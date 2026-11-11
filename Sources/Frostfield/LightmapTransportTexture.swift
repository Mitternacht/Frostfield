import Foundation
import Metal
import simd

/// Base-level linear texels shared by every binding of the same texture.
final class LightmapTexturePixels: Sendable {
    let width: Int
    let height: Int
    let texels: [SIMD4<Float>]
    let maximum: SIMD3<Float>
    let hasValidAlpha: Bool

    init(width: Int, height: Int, texels: [SIMD4<Float>], maximum: SIMD3<Float>, hasValidAlpha: Bool) {
        self.width = width
        self.height = height
        self.texels = texels
        self.maximum = maximum
        self.hasValidAlpha = hasValidAlpha
    }
}

/// A binding's UV transform and level-zero sampler, applied after the hit.
struct LightmapTextureSample: Sendable {
    typealias Addressing = PhysicallyBasedMaterial.TextureSampling.Addressing

    let pixels: LightmapTexturePixels
    let set: MeshTextureCoordinateSet
    let transform: (SIMD3<Double>, SIMD3<Double>)
    let sampling: PhysicallyBasedMaterial.TextureSampling

    var maximum: SIMD3<Float> { pixels.maximum }

    init(_ binding: PhysicallyBasedMaterial.TextureBinding, pixels: LightmapTexturePixels) {
        self.pixels = pixels
        set = binding.textureCoordinateSet
        let (first, second) = binding.transform.rows
        transform = (SIMD3(Double(first.x), Double(first.y), Double(first.z)),
                     SIMD3(Double(second.x), Double(second.y), Double(second.z)))
        sampling = binding.sampling ?? .default
    }

    func coordinates(_ vertex: MeshVertex) -> SIMD2<Double> {
        // Surface validates UV1 before merging, which preserves the coordinates.
        SIMD2<Double>(set == .primary ? vertex.textureCoordinate : vertex.secondaryTextureCoordinate!)
    }

    func sample(_ coordinate: SIMD2<Double>) -> SIMD4<Double> {
        let uv = SIMD3(coordinate.x, coordinate.y, 1)
        let u = Self.address(simd_dot(transform.0, uv), sampling.addressU)
        let v = Self.address(simd_dot(transform.1, uv), sampling.addressV)
        let x = u * Double(pixels.width)
        let y = v * Double(pixels.height)
        func texel(_ x: Int, _ y: Int) -> SIMD4<Double> {
            let column = Self.index(x, size: pixels.width, mode: sampling.addressU)
            let row = Self.index(y, size: pixels.height, mode: sampling.addressV)
            return SIMD4<Double>(pixels.texels[row * pixels.width + column])
        }
        if sampling.magnification == .nearest {
            return texel(Int(x), Int(y))
        }
        // The half-texel offset centers the bilinear footprint on stored texels.
        let x0 = floor(x - 0.5), y0 = floor(y - 0.5)
        let tx = x - 0.5 - x0, ty = y - 0.5 - y0
        let ix = Int(x0), iy = Int(y0)
        let top = texel(ix, iy) * (1 - tx) + texel(ix + 1, iy) * tx
        let bottom = texel(ix, iy + 1) * (1 - tx) + texel(ix + 1, iy + 1) * tx
        return top * (1 - ty) + bottom * ty
    }

    private static func address(_ value: Double, _ mode: Addressing) -> Double {
        switch mode {
        case .clampToEdge: min(max(value, 0), 1)
        case .repeat: value - floor(value)
        case .mirroredRepeat: 1 - abs(1 - abs(value.truncatingRemainder(dividingBy: 2)))
        }
    }

    private static func index(_ value: Int, size: Int, mode: Addressing) -> Int {
        if mode == .repeat { return (value + size) % size }
        // Both mirror and clamp duplicate their edge texel on a bilinear footprint.
        return min(max(value, 0), size - 1)
    }
}

/// Reads shader-readable color textures through Metal, preserving linear HDR
/// values and letting the device decode sRGB and supported compressed formats.
enum LightmapTextureReader {
    private typealias Failure = ProgressiveLightmapBaker.Failure

    static func snapshot(
        _ binding: PhysicallyBasedMaterial.TextureBinding, queue: (any MTLCommandQueue)?
    ) throws -> LightmapTextureSample {
        let texture = binding.texture
        if let queue, queue.device !== texture.device { throw Failure.textureDevice }
        guard binding.sampler == nil || binding.sampling != nil else { throw Failure.textureSampler }
        guard texture.storageMode != .memoryless,
              PhysicallyBasedMaterial.isSurfaceTexture(texture, content: .color) else { throw Failure.unsupportedTexture }
        let (count, overflow) = texture.width.multipliedReportingOverflow(by: texture.height)
        guard !overflow, count > 0, count <= ProgressiveLightmapBaker.maximumTextureTexelCount else {
            throw Failure.textureLimit
        }
        let pixels = try read(texture, count: count, queue: queue)
        return .init(binding, pixels: pixels)
    }

    private static func read(
        _ texture: any MTLTexture, count: Int, queue: (any MTLCommandQueue)?
    ) throws -> LightmapTexturePixels {
        let device = texture.device
        let pipeline: any MTLComputePipelineState
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let function = library.makeFunction(name: "frostfieldLightmapReadColor") else {
                throw Failure.textureReadback("Metal function is missing.")
            }
            pipeline = try device.makeComputePipelineState(function: function)
        } catch let error as Failure {
            throw error
        } catch {
            throw Failure.textureReadback(error.localizedDescription)
        }
        let bytes = count * MemoryLayout<SIMD4<Float>>.stride
        guard let command = (queue ?? device.makeCommandQueue())?.makeCommandBuffer(),
              let output = device.makeBuffer(length: bytes, options: .storageModeShared),
              let encoder = command.makeComputeCommandEncoder() else {
            throw Failure.textureReadback("Metal could not allocate a readback command or buffer.")
        }
        command.label = "Frostfield lightmap texture snapshot"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBuffer(output, offset: 0, index: 0)
        let width = pipeline.threadExecutionWidth
        let height = min(8, pipeline.maxTotalThreadsPerThreadgroup / width)
        encoder.dispatchThreadgroups(
            MTLSize(width: (texture.width + width - 1) / width, height: (texture.height + height - 1) / height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1)
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw Failure.textureReadback(command.error?.localizedDescription ?? "Metal did not complete the readback.")
        }
        let source = output.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        var texels: [SIMD4<Float>] = []
        texels.reserveCapacity(count)
        var maximum = SIMD3<Float>.zero
        var hasValidAlpha = true
        for index in 0..<count {
            let value = source[index]
            let rgb = SIMD3(value.x, value.y, value.z)
            guard rgb.isFiniteVector, simd_reduce_min(rgb) >= 0 else { throw Failure.textureValues }
            maximum = simd_max(maximum, rgb)
            if !(0...1).contains(value.w) { hasValidAlpha = false }
            texels.append(value)
        }
        return .init(width: texture.width, height: texture.height, texels: texels, maximum: maximum, hasValidAlpha: hasValidAlpha)
    }
}
