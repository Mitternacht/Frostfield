import Frostfield
import Metal
import Testing
import simd

/// Traces, rasterizes and reads back small images for the path update and sampling suites.
struct PathTestFixture {
    let device: any MTLDevice
    let queue: any MTLCommandQueue
    let camera: PerspectiveCamera

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 30)
    }

    func pbr(
        _ geometry: MeshGeometry, color: SIMD3<Float> = SIMD3(repeating: 0.8), roughness: Float = 1,
        metallic: Float = 0, emission: SIMD3<Float> = .zero, skin: MeshSkin? = nil
    ) throws -> MeshRenderer {
        try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .physicallyBased(try .init(
                metallic: metallic, roughness: roughness, lightIntensity: 0, ambientIntensity: 0,
                baseColorFactor: SIMD4(color, 1), emissiveFactor: emission, alphaMode: .opaque
            )),
            skin: skin
        )
    }

    /// Accumulates `samples` per pixel in batches of up to 16 and returns linear RGBA.
    func trace(
        _ tracer: PathTracer, width: Int, height: Int, samples: Int, settings: PathTracer.Settings,
        lighting: PathTracer.Lighting? = nil, timing: PathTracer.Timing? = nil
    ) throws -> [SIMD4<Float>] {
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let target = try output(width: width, height: height)
        var history: PathTracer.History?
        while (history?.sampleCount ?? 0) < samples {
            let buffer = try #require(queue.makeCommandBuffer())
            history = try tracer.encode(
                camera: camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
                previous: history, commandBuffer: buffer, samplesPerPixel: min(16, samples - (history?.sampleCount ?? 0)),
                settings: settings, lighting: lighting, timing: timing
            )
            try finish(buffer)
        }
        return try readback(target)
    }

    /// Draws the graph with the raster renderer over a transparent black clear.
    func rasterize(_ graph: SceneGraph, width: Int, height: Int) throws -> [SIMD4<Float>] {
        let color = try output(width: width, height: height, usage: [.renderTarget, .shaderRead])
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        let buffer = try #require(queue.makeCommandBuffer())
        try graph.encode(pass: pass, commandBuffer: buffer)
        try finish(buffer)
        return try readback(color)
    }

    /// Reads the center normal from the raster PBR surface attributes pass.
    func rasterSurfaceNormal(_ graph: SceneGraph, side: Int) throws -> SIMD3<Float> {
        let pass = MTLRenderPassDescriptor()
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: SceneGraph.surfaceAttributesPixelFormat, width: side, height: side, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        let color = try #require(device.makeTexture(descriptor: colorDescriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: side, height: side, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = try #require(device.makeTexture(descriptor: depthDescriptor))
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        let command = try #require(queue.makeCommandBuffer())
        try graph.encodeSurfaceAttributes(pass: pass, commandBuffer: command)
        try finish(command)
        var words = SIMD2<UInt32>.zero
        color.getBytes(&words, bytesPerRow: 8, from: MTLRegionMake2D(side / 2, side / 2, 1, 1), mipmapLevel: 0)
        let x = Float(Int16(bitPattern: UInt16(truncatingIfNeeded: words.x))) / 32767
        let y = Float(Int16(bitPattern: UInt16(truncatingIfNeeded: words.x >> 16))) / 32767
        return simd_normalize(SIMD3(x, y, 1 - abs(x) - abs(y)))
    }

    func output(
        width: Int, height: Int, usage: MTLTextureUsage = [.shaderWrite, .shaderRead]
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = usage
        descriptor.hazardTrackingMode = .tracked
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    func finish(_ buffer: any MTLCommandBuffer) throws {
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed, "\(buffer.error?.localizedDescription ?? "No Metal error")")
    }

    func readback(_ texture: any MTLTexture) throws -> [SIMD4<Float>] {
        var result = Array(repeating: SIMD4<Float16>.zero, count: texture.width * texture.height)
        try result.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: texture.width * 8,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return result.map(SIMD4<Float>.init)
    }

    /// The largest per-channel RGB difference between two images.
    static func maximumDifference(_ a: [SIMD4<Float>], _ b: [SIMD4<Float>]) -> Float {
        zip(a, b).reduce(0) { max($0, simd_reduce_max(abs(SIMD3($1.0.x, $1.0.y, $1.0.z) - SIMD3($1.1.x, $1.1.y, $1.1.z)))) }
    }

    /// Root-mean-square RGB error against a reference.
    static func rmse(_ estimate: [SIMD4<Float>], _ reference: [SIMD4<Float>]) -> Double {
        let sum = zip(estimate, reference).reduce(0.0) { total, pair in
            let error = SIMD3<Double>(SIMD3(pair.0.x, pair.0.y, pair.0.z) - SIMD3(pair.1.x, pair.1.y, pair.1.z))
            return total + simd_length_squared(error)
        }
        return (sum / Double(estimate.count * 3)).squareRoot()
    }
}
