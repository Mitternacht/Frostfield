import CoreGraphics
import Foundation
import Frostfield
import ImageIO
import Metal
import Testing
import simd

/// Renders camera-model, exposure and light-profile scenes for the physical camera suites.
struct PathCameraFixture {
    let base: PathTestFixture
    var device: any MTLDevice { base.device }
    var queue: any MTLCommandQueue { base.queue }

    init() throws {
        base = try PathTestFixture()
    }

    /// A rough Lambertian PBR material: KHR specular factor zero removes dielectric reflection.
    func lambert(_ geometry: MeshGeometry, albedo: Float, emission: SIMD3<Float> = .zero) throws -> MeshRenderer {
        try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .physicallyBased(try .init(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0,
                baseColorFactor: SIMD4(albedo, albedo, albedo, 1), emissiveFactor: emission, alphaMode: .opaque,
                specular: try .init(factor: 0)
            ))
        )
    }

    /// Accumulates `samples` per pixel through a camera model and returns linear RGBA.
    func trace(
        _ tracer: PathTracer, model: PathTracer.CameraModel, transform: SceneTransform = .init(),
        width: Int, height: Int, samples: Int, settings: PathTracer.Settings = .init(maxBounces: 1, seed: 7),
        lighting: PathTracer.Lighting? = nil
    ) throws -> [SIMD4<Float>] {
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let target = try base.output(width: width, height: height)
        var history: PathTracer.History?
        while (history?.sampleCount ?? 0) < samples {
            let buffer = try #require(queue.makeCommandBuffer())
            history = try tracer.encode(
                cameraModel: model, cameraTransform: transform, destination: target, accumulation: accumulation,
                previous: history, commandBuffer: buffer, samplesPerPixel: min(16, samples - (history?.sampleCount ?? 0)),
                settings: settings, lighting: lighting
            )
            try base.finish(buffer)
        }
        return try base.readback(target)
    }

    /// Draws the graph over transparent black and keeps a shader-readable Depth32Float copy.
    func rasterize(_ graph: SceneGraph, width: Int, height: Int) throws -> (color: any MTLTexture, depth: any MTLTexture) {
        let color = try base.output(width: width, height: height, usage: [.renderTarget, .shaderRead])
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = [.renderTarget, .shaderRead]
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        pass.depthAttachment.storeAction = .store
        let buffer = try #require(queue.makeCommandBuffer())
        try graph.encode(pass: pass, commandBuffer: buffer)
        try base.finish(buffer)
        return (color, depth)
    }

    /// Writes an 8-bit sRGB PNG of `scale × color` when FROSTFIELD_TEST_IMAGES names a directory.
    static func writeImage(_ pixels: [SIMD4<Float>], width: Int, height: Int, name: String, scale: Float = 1) {
        guard let directory = ProcessInfo.processInfo.environment["FROSTFIELD_TEST_IMAGES"] else { return }
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for (index, pixel) in pixels.enumerated() {
            for channel in 0..<3 {
                let linear = min(max(pixel[channel] * scale, 0), 1)
                let encoded = linear <= 0.0031308 ? 12.92 * linear : 1.055 * pow(linear, 1 / 2.4) - 0.055
                bytes[index * 4 + channel] = UInt8((encoded * 255).rounded())
            }
        }
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
              ) else { return }
        let url = URL(fileURLWithPath: directory).appendingPathComponent("\(name).png")
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(destination, image, nil)
        CGImageDestinationFinalize(destination)
    }
}
