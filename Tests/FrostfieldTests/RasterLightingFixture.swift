import CoreGraphics
import Foundation
import Frostfield
import ImageIO
import Metal
import Testing
import UniformTypeIdentifiers
import simd

/// Renders scene graphs at arbitrary sizes for the cascaded shadow, ambient occlusion
/// and reflection suites, and reads RGBA16Float results back as float4 pixels.
struct RasterLightingFixture {
    let device: any MTLDevice
    let queue: any MTLCommandQueue
    let width: Int
    let height: Int

    init(width: Int, height: Int) throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
        self.width = width
        self.height = height
    }

    func texture(
        _ format: MTLPixelFormat, usage: MTLTextureUsage, storage: MTLStorageMode = .private
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = storage
        descriptor.usage = usage
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    /// Shared RGBA16Float color that can be read back, rendered to and sampled.
    func colorTarget() throws -> any MTLTexture {
        try texture(.rgba16Float, usage: [.renderTarget, .shaderRead], storage: .shared)
    }

    func depthTarget() throws -> any MTLTexture {
        try texture(.depth32Float, usage: [.renderTarget, .shaderRead])
    }

    func pass(
        color: any MTLTexture, depth: any MTLTexture, depthMode: MeshDepthMode = .forward
    ) -> MTLRenderPassDescriptor {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .store
        pass.depthAttachment.clearDepth = depthMode == .forward ? 1 : 0
        return pass
    }

    func run(_ body: (any MTLCommandBuffer) throws -> Void) throws {
        let buffer = try #require(queue.makeCommandBuffer())
        try body(buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed, "\(buffer.error?.localizedDescription ?? "The GPU work did not complete.")")
    }

    /// Renders a graph into new color and depth targets.
    func render(
        _ graph: SceneGraph, shadowMap: DirectionalShadowMap? = nil
    ) throws -> (color: any MTLTexture, depth: any MTLTexture) {
        let color = try colorTarget()
        let depth = try depthTarget()
        try run { buffer in
            try graph.encode(pass: pass(color: color, depth: depth), commandBuffer: buffer, shadowMap: shadowMap)
        }
        return (color, depth)
    }

    func pixels(of texture: any MTLTexture) throws -> [SIMD4<Float>] {
        var halves = [SIMD4<Float16>](repeating: .zero, count: texture.width * texture.height)
        try halves.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: texture.width * MemoryLayout<SIMD4<Float16>>.stride,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return halves.map(SIMD4<Float>.init)
    }

    /// Writes an inspection PNG when `FROSTFIELD_TEST_IMAGES` names a directory. Values
    /// map linearly from 0...`scale` to 8-bit sRGB-agnostic gray levels.
    func writeImage(_ pixels: [SIMD4<Float>], name: String, scale: Float = 1) {
        guard let directory = ProcessInfo.processInfo.environment["FROSTFIELD_TEST_IMAGES"] else { return }
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for (index, pixel) in pixels.enumerated() {
            for channel in 0..<3 {
                let value = pixel[channel] / scale
                bytes[index * 4 + channel] = UInt8(max(0, min(255, (value.squareRoot() * 255).rounded())))
            }
        }
        let url = URL(fileURLWithPath: directory).appendingPathComponent("\(name).png")
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
              ),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(destination, image, nil)
        CGImageDestinationFinalize(destination)
    }

    /// Image-based lighting from a small equirectangular panorama whose texel (column, row)
    /// has the given linear color.
    func environment(
        width: Int = 16, height: Int = 8, intensity: Float = 1, color: (Int, Int) -> SIMD3<Float>
    ) throws -> PhysicallyBasedMaterial.Environment {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let panorama = try #require(device.makeTexture(descriptor: descriptor))
        var texels: [SIMD4<Float16>] = []
        for row in 0..<height {
            for column in 0..<width {
                texels.append(SIMD4<Float16>(SIMD4(color(column, row), 1)))
            }
        }
        try texels.withUnsafeBytes { bytes in
            panorama.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: try #require(bytes.baseAddress), bytesPerRow: width * 8
            )
        }
        let lighting = try EnvironmentLighting(
            queue: queue, equirectangular: panorama,
            settings: .init(cubeSize: 16, diffuseSize: 8, brdfSize: 32, sampleCount: 64)
        )
        return try PhysicallyBasedMaterial.Environment(lighting: lighting, intensity: intensity)
    }

    /// A one-texel RGBA8 texture, for constant material maps.
    func constantTexture(_ bytes: SIMD4<UInt8>) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        var value = bytes
        texture.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &value, bytesPerRow: 4)
        return texture
    }

    /// The Hermite smoothstep that Metal's `smoothstep` computes.
    static func smoothstep(_ edge0: Float, _ edge1: Float, _ x: Float) -> Float {
        let t = min(max((x - edge0) / (edge1 - edge0), 0), 1)
        return t * t * (3 - 2 * t)
    }

    /// Prints a measured value for the evidence page when `FROSTFIELD_TEST_IMAGES` is set.
    func note(_ message: @autoclosure () -> String) {
        guard ProcessInfo.processInfo.environment["FROSTFIELD_TEST_IMAGES"] != nil else { return }
        print("[raster-lighting] \(message())")
    }

    /// A ground plane at y = 0 facing +Y, built from the XY plane rotated about X.
    static func groundTransform(y: Float = 0) -> SceneTransform {
        SceneTransform(translation: SIMD3(0, y, 0), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0)))
    }

    /// A camera at `position` pitched by `pitch` radians about X (negative looks down).
    static func cameraTransform(position: SIMD3<Float>, pitch: Float, yaw: Float = 0) -> SceneTransform {
        SceneTransform(
            translation: position,
            rotation: simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0)) * simd_quatf(angle: pitch, axis: SIMD3(1, 0, 0))
        )
    }

    /// The world ray through a pixel center for a perspective camera.
    func ray(
        x: Int, y: Int, camera: PerspectiveCamera, transform: SceneTransform
    ) -> (origin: SIMD3<Float>, direction: SIMD3<Float>) {
        let tangent = tan(camera.verticalFieldOfView / 2)
        let aspect = Float(width) / Float(height)
        let ndc = SIMD2((Float(x) + 0.5) / Float(width) * 2 - 1, 1 - (Float(y) + 0.5) / Float(height) * 2)
        let local = SIMD3(ndc.x * tangent * aspect, ndc.y * tangent, -1)
        return (transform.translation, simd_normalize(transform.rotation.act(local)))
    }

    /// A camera's projection made off-center and anisotropic, like one eye of a headset:
    /// the principal point moves to NDC −`center`, and x is scaled by `horizontalScale`.
    static func offCenterProjection(
        _ camera: PerspectiveCamera, aspectRatio: Float, center: SIMD2<Float>, horizontalScale: Float,
        depthMode: MeshDepthMode
    ) throws -> simd_float4x4 {
        var projection = try camera.projection(aspectRatio: aspectRatio)
        projection.columns.0.x *= horizontalScale
        projection.columns.2.x = center.x
        projection.columns.2.y = center.y
        guard depthMode == .reversed else { return projection }
        var reverse = matrix_identity_float4x4
        reverse.columns.2.z = -1
        reverse.columns.3.z = 1
        return reverse * projection
    }

    /// The pixel that sees a world point through a view-to-clip matrix.
    func project(_ point: SIMD3<Float>, clipFromView: simd_float4x4, worldFromView: simd_float4x4) -> SIMD2<Float> {
        let clip = clipFromView * simd_inverse(worldFromView) * SIMD4(point, 1)
        let ndc = SIMD2(clip.x, clip.y) / clip.w
        return SIMD2((ndc.x + 1) / 2 * Float(width), (1 - ndc.y) / 2 * Float(height))
    }

    /// The world ray through a pixel center, by inverting a view-to-clip matrix.
    func ray(
        x: Int, y: Int, clipFromView: simd_float4x4, worldFromView: simd_float4x4
    ) -> (origin: SIMD3<Float>, direction: SIMD3<Float>) {
        let ndc = SIMD2((Float(x) + 0.5) / Float(width) * 2 - 1, 1 - (Float(y) + 0.5) / Float(height) * 2)
        let view = simd_inverse(clipFromView) * SIMD4<Float>(ndc.x, ndc.y, 0.5, 1)
        let local = SIMD3(view.x, view.y, view.z) / view.w
        let world = worldFromView * SIMD4(local, 0)
        let origin = worldFromView.columns.3
        return (SIMD3(origin.x, origin.y, origin.z), simd_normalize(SIMD3(world.x, world.y, world.z)))
    }

    /// The pixel that sees a world point, as continuous coordinates with a top-left origin.
    func project(
        _ point: SIMD3<Float>, camera: PerspectiveCamera, transform: SceneTransform
    ) -> SIMD2<Float> {
        let local = transform.rotation.inverse.act(point - transform.translation)
        let tangent = tan(camera.verticalFieldOfView / 2)
        let aspect = Float(width) / Float(height)
        let ndc = SIMD2(local.x / (-local.z * tangent * aspect), local.y / (-local.z * tangent))
        return SIMD2((ndc.x + 1) / 2 * Float(width), (1 - ndc.y) / 2 * Float(height))
    }
}
