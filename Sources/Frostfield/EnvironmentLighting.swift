import Foundation
import Metal
import simd
import Synchronization

/// Prepared split-sum lighting from a linear HDR panorama or cube.
public final class EnvironmentLighting {
    public struct Settings: Equatable, Sendable {
        public let cubeSize: Int
        public let diffuseSize: Int
        public let brdfSize: Int
        public let sampleCount: Int

        /// Cube sizes are powers of two. Use 8...256 for radiance and specular,
        /// 4...64 for diffuse, 8...256 for the LUT and 16...512 samples per texel.
        public init(cubeSize: Int = 128, diffuseSize: Int = 16, brdfSize: Int = 128, sampleCount: Int = 128) {
            self.cubeSize = cubeSize
            self.diffuseSize = diffuseSize
            self.brdfSize = brdfSize
            self.sampleCount = sampleCount
        }

        var isValid: Bool {
            (8...256).contains(cubeSize) && cubeSize.nonzeroBitCount == 1
                && (4...64).contains(diffuseSize) && diffuseSize.nonzeroBitCount == 1
                && (8...256).contains(brdfSize)
                && (16...512).contains(sampleCount)
        }
    }

    public var device: any MTLDevice { skyCube.device }
    private let pathEnvironmentCache = Mutex<PathTracer.EnvironmentMap?>(nil)

    /// HDR sky radiance. Use mip zero for a sharp background.
    public let skyCube: any MTLTexture
    /// Cosine-convolved radiance divided by π, ready to multiply by diffuse albedo.
    public let diffuseCube: any MTLTexture
    /// Specular radiance. Sample mip `perceptualRoughness * maxSpecularLOD`.
    public let specularCube: any MTLTexture
    /// Split-sum coefficients A and B. Sample at (N·V, perceptual roughness)
    /// and multiply the prefiltered radiance by `F0 * A + B`.
    public let brdfLUT: any MTLTexture
    public var maxSpecularLOD: Float { Float(specularCube.mipmapLevelCount - 1) }

    /// Prepares the textures on the supplied conventional command queue and waits
    /// for completion. Perform this work before starting the frame loop.
    /// The input must be a shader-readable, linear RGBA float 2:1 texture on this
    /// device. Complete its upload before calling, or submit a hazard-tracked
    /// upload on the same queue first.
    /// The top row faces +Y, the center faces +X, and increasing U turns toward +Z.
    /// Treat the prepared textures as read-only.
    public convenience init(
        queue: any MTLCommandQueue,
        equirectangular: any MTLTexture,
        settings: Settings = .init()
    ) throws {
        try self.init(queue: queue, input: equirectangular, projection: .equirectangular, settings: settings)
    }

    /// Prepares a static lighting probe from mip zero of a shader-readable RGBA8,
    /// sRGB RGBA8, RGBA16Float, or filterable RGBA32Float cube. Finish
    /// writing the cube before calling, or
    /// submit the tracked writes on this queue first. The lighting owns independent
    /// textures, so recapturing into the source cube does not modify the probe.
    public convenience init(
        queue: any MTLCommandQueue, cube: any MTLTexture, settings: Settings = .init()
    ) throws {
        try self.init(queue: queue, input: cube, projection: .cube, settings: settings)
    }

    private enum InputProjection {
        case equirectangular, cube
    }

    private init(
        queue: any MTLCommandQueue, input: any MTLTexture, projection: InputProjection, settings: Settings
    ) throws {
        guard settings.isValid else { throw Failure.settings }
        let device = queue.device
        guard input.device === device else { throw Failure.wrongDevice }
        let readable = input.sampleCount == 1 && !input.isFramebufferOnly && input.storageMode != .memoryless
            && (input.usage.isEmpty || input.usage.contains(.shaderRead))
        let floatingPoint = input.pixelFormat == .rgba16Float || input.pixelFormat == .rgba32Float
        let conversionFunction: String
        let conversionLabel: String
        switch projection {
        case .equirectangular:
            guard readable, floatingPoint, input.textureType == .type2D, input.height > 0, input.width <= 8_192,
                  input.width == 2 * input.height else { throw Failure.inputTexture }
            conversionFunction = "frostfield_environment_sky"
            conversionLabel = "equirectangular to cube"
        case .cube:
            let rgba8 = input.pixelFormat == .rgba8Unorm || input.pixelFormat == .rgba8Unorm_srgb
            guard readable, floatingPoint || rgba8, input.textureType == .typeCube, input.width == input.height,
                  input.width > 0, input.pixelFormat != .rgba32Float || device.supports32BitFloatFiltering else {
                throw Failure.inputCube
            }
            conversionFunction = "frostfield_environment_cube"
            conversionLabel = "cube radiance resampling"
        }

        let library: any MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw Failure.library(error.localizedDescription)
        }
        let skyPipeline = try Self.pipeline(conversionFunction, library: library)
        let diffusePipeline = try Self.pipeline("frostfield_environment_diffuse", library: library)
        let specularPipeline = try Self.pipeline("frostfield_environment_specular", library: library)
        let brdfPipeline = try Self.pipeline("frostfield_environment_brdf", library: library)

        let sky = try Self.makeCube(size: settings.cubeSize, mipmapped: true, device: device, label: "HDR sky")
        let specular = try Self.makeCube(
            size: settings.cubeSize, mipmapped: true, device: device, label: "GGX environment"
        )
        let diffuse = try Self.makeCube(
            size: settings.diffuseSize, mipmapped: false, device: device, label: "Diffuse environment"
        )
        let lutDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rg16Float, width: settings.brdfSize, height: settings.brdfSize, mipmapped: false
        )
        let lut = try Self.makeTexture(lutDescriptor, device: device, label: "Environment BRDF LUT")
        guard let command = queue.makeCommandBuffer() else { throw Failure.commandBuffer }
        command.label = "Frostfield environment lighting preparation"

        guard let conversion = command.makeComputeCommandEncoder() else { throw Failure.encoder(conversionLabel) }
        conversion.label = "Frostfield \(conversionLabel)"
        conversion.setComputePipelineState(skyPipeline)
        conversion.setTexture(input, index: 0)
        conversion.setTexture(sky, index: 1)
        Self.dispatch(conversion, pipeline: skyPipeline, side: sky.width, faces: 6)
        conversion.endEncoding()

        guard let blit = command.makeBlitCommandEncoder() else { throw Failure.encoder("sky mipmap") }
        blit.generateMipmaps(for: sky)
        blit.copy(
            from: sky, sourceSlice: 0, sourceLevel: 0, to: specular, destinationSlice: 0,
            destinationLevel: 0, sliceCount: 6, levelCount: 1
        )
        blit.endEncoding()

        guard let filters = command.makeComputeCommandEncoder() else { throw Failure.encoder("filter") }
        filters.label = "Frostfield IBL convolution and integration"
        var sampleCount = UInt32(settings.sampleCount)
        filters.setComputePipelineState(diffusePipeline)
        filters.setTexture(sky, index: 0)
        filters.setTexture(diffuse, index: 1)
        filters.setBytes(&sampleCount, length: MemoryLayout<UInt32>.size, index: 0)
        Self.dispatch(filters, pipeline: diffusePipeline, side: diffuse.width, faces: 6)

        filters.setComputePipelineState(specularPipeline)
        filters.setTexture(specular, index: 1)
        for mip in 1..<specular.mipmapLevelCount {
            let side = specular.width >> mip
            var parameters = SIMD4(UInt32(side), UInt32(mip), UInt32(specular.mipmapLevelCount), sampleCount)
            filters.setBytes(&parameters, length: MemoryLayout<SIMD4<UInt32>>.size, index: 0)
            Self.dispatch(filters, pipeline: specularPipeline, side: side, faces: 6)
        }

        filters.setComputePipelineState(brdfPipeline)
        filters.setTexture(lut, index: 0)
        filters.setTexture(nil, index: 1)
        filters.setBytes(&sampleCount, length: MemoryLayout<UInt32>.size, index: 0)
        Self.dispatch(filters, pipeline: brdfPipeline, side: lut.width, faces: 1)
        filters.endEncoding()

        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else { throw Failure.execution(command.error?.localizedDescription) }

        skyCube = sky
        diffuseCube = diffuse
        specularCube = specular
        brdfLUT = lut
    }

    /// Wraps textures prepared by an earlier instance, for `SceneArchive`.
    package init(skyCube: any MTLTexture, diffuseCube: any MTLTexture, specularCube: any MTLTexture, brdfLUT: any MTLTexture) {
        self.skyCube = skyCube
        self.diffuseCube = diffuseCube
        self.specularCube = specularCube
        self.brdfLUT = brdfLUT
    }

    /// Returns the immutable path map for this prepared radiance cube. Cache it by
    /// asset identity so re-snapshotting a scene can extend an existing history.
    func pathEnvironmentMap(queue: any MTLCommandQueue) throws -> PathTracer.EnvironmentMap {
        guard queue.device === device else { throw PathTracer.Failure.environmentDevice }
        return try pathEnvironmentCache.withLock { cached in
            if let cached { return cached }
            let map = try preparePathEnvironmentMap(queue: queue)
            cached = map
            return map
        }
    }

    private func preparePathEnvironmentMap(queue: any MTLCommandQueue) throws -> PathTracer.EnvironmentMap {
        // Public preparations produce linear RGBA16Float. Archives can wrap
        // compatible sampled formats, so validate before creating an encoder.
        let format = skyCube.pixelFormat
        guard skyCube.textureType == .typeCube, skyCube.width > 0, skyCube.sampleCount == 1,
              !skyCube.isFramebufferOnly, skyCube.storageMode != .memoryless,
              skyCube.usage.isEmpty || skyCube.usage.contains(.shaderRead),
              format == .rgba16Float || format == .rgba8Unorm || format == .rgba8Unorm_srgb
                || format == .rgba32Float && device.supports32BitFloatFiltering else {
            throw Failure.inputCube
        }
        // At native cube sizes this samples at least the equatorial cube density.
        // Bound archived cube inputs to the largest standard raster preparation.
        let faceSize = min(skyCube.width, 256)
        let width = faceSize * 4
        let height = faceSize * 2
        let count = width * height
        guard let output = device.makeBuffer(
            length: count * MemoryLayout<SIMD4<Float16>>.stride,
            options: [.storageModeShared, .hazardTrackingModeTracked]
        ) else { throw Failure.resource("path environment panorama") }
        output.label = "Frostfield path environment panorama staging"

        let library: any MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw Failure.library(error.localizedDescription)
        }
        let pipeline = try Self.pipeline("frostfield_environment_panorama", library: library)
        guard let command = queue.makeCommandBuffer() else { throw Failure.commandBuffer }
        command.label = "Frostfield path environment preparation"
        guard let encoder = command.makeComputeCommandEncoder() else { throw Failure.encoder("cube to panorama") }
        var size = SIMD2(UInt32(width), UInt32(height))
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(skyCube, index: 0)
        encoder.setBuffer(output, offset: 0, index: 0)
        encoder.setBytes(&size, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
        Self.dispatch(encoder, pipeline: pipeline, width: width, height: height, faces: 1)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else { throw Failure.execution(command.error?.localizedDescription) }

        let pixels = withExtendedLifetime(output) {
            let storage = output.contents().bindMemory(to: SIMD4<Float16>.self, capacity: count)
            return Array(UnsafeBufferPointer(start: storage, count: count))
        }
        return try PathTracer.EnvironmentMap(device: device, width: width, height: height, pixels: pixels)
    }

    private static func makeCube(
        size: Int, mipmapped: Bool, device: any MTLDevice, label: String
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.textureCubeDescriptor(
            pixelFormat: .rgba16Float, size: size, mipmapped: mipmapped
        )
        return try makeTexture(descriptor, device: device, label: label)
    }

    private static func makeTexture(
        _ descriptor: MTLTextureDescriptor, device: any MTLDevice, label: String
    ) throws -> any MTLTexture {
        descriptor.storageMode = .private
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.hazardTrackingMode = .tracked
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.resource(label) }
        texture.label = "Frostfield \(label)"
        return texture
    }

    private static func pipeline(_ name: String, library: any MTLLibrary) throws -> any MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else { throw Failure.shader(name) }
        do {
            return try library.device.makeComputePipelineState(function: function)
        } catch {
            throw Failure.pipeline(name, error.localizedDescription)
        }
    }

    private static func dispatch(
        _ encoder: any MTLComputeCommandEncoder, pipeline: any MTLComputePipelineState, side: Int, faces: Int
    ) {
        dispatch(encoder, pipeline: pipeline, width: side, height: side, faces: faces)
    }

    private static func dispatch(
        _ encoder: any MTLComputeCommandEncoder, pipeline: any MTLComputePipelineState, width: Int, height: Int, faces: Int
    ) {
        let groupWidth = min(pipeline.maxTotalThreadsPerThreadgroup, 8)
        let groupHeight = min(pipeline.maxTotalThreadsPerThreadgroup / groupWidth, 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: (width + groupWidth - 1) / groupWidth, height: (height + groupHeight - 1) / groupHeight, depth: faces),
            threadsPerThreadgroup: MTLSize(width: groupWidth, height: groupHeight, depth: 1)
        )
    }

    public enum Failure: LocalizedError, Equatable {
        case settings
        case wrongDevice
        case inputTexture
        case inputCube
        case library(String)
        case shader(String)
        case pipeline(String, String)
        case resource(String)
        case commandBuffer
        case encoder(String)
        case execution(String?)

        public var errorDescription: String? {
            switch self {
            case .settings: "Use environment resolutions and sample counts within the supported ranges."
            case .wrongDevice: "Use a texture from the environment queue's Metal device."
            case .inputTexture: "Use a shader-readable 2:1 linear RGBA16Float or RGBA32Float panorama up to 8192 pixels wide."
            case .inputCube: "Use a shader-readable RGBA8, RGBA16Float, or filterable RGBA32Float cube."
            case .library(let detail): "Could not load Frostfield's Metal library: \(detail)"
            case .shader(let name): "The environment shader \(name) is missing."
            case .pipeline(let name, let detail): "Could not prepare \(name): \(detail)"
            case .resource(let name): "Metal could not allocate \(name)."
            case .commandBuffer: "Metal could not create an environment command buffer."
            case .encoder(let name): "Metal could not create the environment \(name) encoder."
            case .execution(let detail): detail ?? "Metal could not finish preparing the environment."
            }
        }
    }
}

/// Scene-selected environment bindings for a raster pass. Resources are read-only
/// through GPU completion, including when the caller uses unretained references.
struct RasterSceneEnvironment: @unchecked Sendable {
    let lighting: EnvironmentLighting?
    private let intensity: Float
    private let yaw: Float
    private let sharedLookup: RasterEnvironmentLookup?

    init(lighting: EnvironmentLighting? = nil, intensity: Float = 1, yaw: Float = 0, eulerRotation: SIMD3<Float> = .zero) {
        self.lighting = lighting
        self.intensity = intensity
        self.yaw = yaw
        sharedLookup = lighting.map {
            RasterEnvironmentLookup(EnvironmentLookup(
                eulerRotation: eulerRotation, intensity: intensity, maxSpecularLOD: $0.maxSpecularLOD
            ))
        }
    }

    func resolve(_ material: PhysicallyBasedMaterial) -> (EnvironmentLighting, RasterEnvironmentLookup)? {
        if let authored = material.environment {
            // Keep the earlier global adjustment API for scenes without a shared source.
            let rotation = SIMD3(0, lighting == nil ? authored.yaw + yaw : authored.yaw, 0)
            let scale = lighting == nil ? authored.intensity * intensity : authored.intensity
            let lookup = EnvironmentLookup(eulerRotation: rotation, intensity: scale, maxSpecularLOD: authored.lighting.maxSpecularLOD)
            return (authored.lighting, RasterEnvironmentLookup(lookup, projection: authored.boxProjection))
        }
        guard material.shadingModel == .physicallyBased, let lighting, let sharedLookup else { return nil }
        return (lighting, sharedLookup)
    }

    func retainIfNeeded(on commandBuffer: any MTLCommandBuffer) {
        guard lighting != nil, !commandBuffer.retainedReferences else { return }
        commandBuffer.addCompletedHandler { [self] _ in withExtendedLifetime(self) {} }
    }
}

/// Raster-only uniforms, keeping the shared path sky layout unchanged.
struct RasterEnvironmentLookup {
    let orientation: EnvironmentLookup
    let minimum: SIMD4<Float>
    let maximum: SIMD4<Float>
    let capture: SIMD4<Float>

    init(_ orientation: EnvironmentLookup, projection: EnvironmentBoxProjection? = nil) {
        self.orientation = orientation
        minimum = projection.map { SIMD4($0.bounds.minimum, 0) } ?? .zero
        maximum = projection.map { SIMD4($0.bounds.maximum, 0) } ?? .zero
        capture = projection.map { SIMD4($0.capturePosition, 1) } ?? .zero
    }
}

/// Inverse environment orientation shared by raster and path GPU uniforms.
/// Columns store the world-to-map matrix; the first two w lanes hold controls.
struct EnvironmentLookup {
    let x: SIMD4<Float>
    let y: SIMD4<Float>
    let z: SIMD4<Float>

    init(eulerRotation: SIMD3<Float> = .zero, intensity: Float = 1, maxSpecularLOD: Float = 0) {
        let cy = cos(eulerRotation.y), sy = sin(eulerRotation.y)
        if eulerRotation.x == 0, eulerRotation.z == 0 {
            x = SIMD4(cy, 0, sy, intensity)
            y = SIMD4(0, 1, 0, maxSpecularLOD)
            z = SIMD4(-sy, 0, cy, 0)
        } else {
            let cx = cos(eulerRotation.x), sx = sin(eulerRotation.x)
            let cz = cos(eulerRotation.z), sz = sin(eulerRotation.z)
            // Transpose the proper XYZ rotation Rx * Ry * Rz for texture lookup.
            x = SIMD4(cy * cz, -cy * sz, sy, intensity)
            y = SIMD4(cx * sz + sx * cz * sy, cx * cz - sx * sz * sy, -sx * cy, maxSpecularLOD)
            z = SIMD4(sx * sz - cx * cz * sy, sx * cz + cx * sz * sy, cx * cy, 0)
        }
    }

    var localFromWorld: simd_float3x3 {
        simd_float3x3(SIMD3(x.x, x.y, x.z), SIMD3(y.x, y.y, y.z), SIMD3(z.x, z.y, z.z))
    }
}
