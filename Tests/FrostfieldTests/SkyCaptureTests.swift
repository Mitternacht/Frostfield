import CoreGraphics
import Foundation
import Frostfield
import ImageIO
import Metal
import Testing
import UniformTypeIdentifiers
import simd

/// Writes the sky evidence images in docs/evidence/sky.md when
/// `TEST_RUNNER_FROSTFIELD_SKY_CAPTURE` names a directory; skipped otherwise.
@Suite(.enabled(
    if: ProcessInfo.processInfo.environment["FROSTFIELD_SKY_CAPTURE"] != nil && MTLCreateSystemDefaultDevice() != nil,
    "Set TEST_RUNNER_FROSTFIELD_SKY_CAPTURE to a directory to capture sky images"
))
struct SkyCaptureTests {
    private let folder = ProcessInfo.processInfo.environment["FROSTFIELD_SKY_CAPTURE"] ?? ""

    @Test("Capture sky panoramas, a hazy raster scene and its path-traced counterpart")
    func capture() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        try FileManager.default.createDirectory(atPath: folder, withIntermediateDirectories: true)

        // One panorama row per Sun elevation, each exposed for its horizon.
        let width = 512
        var sheet = [SIMD3<Float>]()
        for (elevation, turbidity) in [(60.0, 2.0), (20.0, 2.0), (5.0, 3.0), (1.0, 2.0), (-3.0, 2.0)] {
            let e = elevation * .pi / 180
            let model = try SkyModel(sunDirection: SIMD3(Float(cos(e)), Float(sin(e)), 0), atmosphere: try .init(turbidity: turbidity), luminanceScale: 0.001)
            let sky = try SkyEnvironment(queue: queue, model: model, settings: .init(panoramaWidth: width))
            let pixels = try SkyEnvironmentTests.pixels(sky.panorama)
            let horizon = pixels[(width / 4 - 2) * width + width / 2]
            let exposure = 0.6 / max(0.2126 * horizon.x + 0.7152 * horizon.y + 0.0722 * horizon.z, 1e-6)
            sheet += pixels.map { SIMD3($0.x, $0.y, $0.z) * exposure }
        }
        try write(sheet, width: width, height: width / 2 * 5, exposure: 1, name: "sky-panoramas")

        // Interlaken, 21 June 2026 at 20:15: the Sun about 10° above the west-northwest horizon.
        let position = try SolarPosition(
            localTime: DateComponents(year: 2026, month: 6, day: 21, hour: 20, minute: 15),
            timeZone: try #require(TimeZone(identifier: "Europe/Zurich")), latitude: 46.5, longitude: 7.9
        )
        let model = try SkyModel(solarPosition: position, atmosphere: try .init(turbidity: 3, groundAlbedo: 0.2), altitude: 500, luminanceScale: 0.001)
        let sky = try SkyEnvironment(queue: queue, model: model)
        let rig = try LightRig(lights: [sky.sunLight(castsShadow: false)])
        let environment = try sky.materialEnvironment()
        func material(_ color: SIMD3<Float>, roughness: Float, metallic: Float = 0) throws -> MeshMaterial {
            .physicallyBased(try PhysicallyBasedMaterial(
                metallic: metallic, roughness: roughness, lightIntensity: 0, ambientIntensity: 0,
                baseColorFactor: SIMD4(color, 1), alphaMode: .opaque, environment: environment
            ))
        }
        let sphere = try MeshGeometry.uvSphere(radius: 1, segments: 64, rings: 32)
        let ground = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 40_000, height: 40_000),
            material: material(SIMD3(0.18, 0.2, 0.12), roughness: 0.9)
        )
        let white = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: sphere, material: material(SIMD3(repeating: 0.9), roughness: 0.35))
        let chrome = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: sphere, material: material(SIMD3(0.95, 0.93, 0.9), roughness: 0.08, metallic: 1))
        let clay = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: sphere, material: material(SIMD3(0.6, 0.3, 0.2), roughness: 0.6))
        let sun = simd_normalize(SIMD2(model.sunDirection.x, model.sunDirection.z))
        let yaw = simd_quatf(angle: atan2(-sun.x, -sun.y) + 0.35, axis: SIMD3(0, 1, 0))
        var roots = [
            SceneNode(transform: .init(rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))), mesh: ground),
            SceneNode(transform: .init(translation: yaw.act(SIMD3(-9, 6, -60)), scale: SIMD3(repeating: 6)), mesh: white),
            SceneNode(transform: .init(translation: yaw.act(SIMD3(9, 6, -66)), scale: SIMD3(repeating: 6)), mesh: chrome),
        ]
        // Spheres from 150 m to 19 km away.
        for index in 0..<8 {
            let distance = 150 * pow(2, Float(index))
            let radius = 15 * pow(1.6, Float(index))
            let side = Float(index) / 7 * 1.2 - 0.6
            roots.append(SceneNode(
                transform: .init(translation: yaw.act(SIMD3(side * distance * 0.5, radius, -distance)), scale: SIMD3(repeating: radius)),
                mesh: clay
            ))
        }
        let camera = try PerspectiveCamera(verticalFieldOfView: 0.75, near: 0.1, far: 30_000)
        let cameraTransform = SceneTransform(translation: SIMD3(0, 25, 0), rotation: yaw * simd_quatf(angle: -0.06, axis: SIMD3(1, 0, 0)))
        var graph = SceneGraph(roots: roots + (try rig.sceneNodes()), camera: camera)
        graph.cameraTransform = cameraTransform

        let size = (width: 960, height: 540)
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: size.width, height: size.height, mipmapped: false)
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = [.renderTarget, .shaderRead]
        let color = try #require(device.makeTexture(descriptor: colorDescriptor))
        let output = try #require(device.makeTexture(descriptor: colorDescriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: size.width, height: size.height, mipmapped: false)
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = [.renderTarget, .shaderRead]
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].storeAction = .store
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        pass.depthAttachment.storeAction = .store
        let aerial = try AerialPerspective(environment: sky)
        let buffer = try #require(queue.makeCommandBuffer())
        try graph.encode(pass: pass, commandBuffer: buffer)
        try aerial.encode(
            source: color, depth: depth, destination: output, camera: camera,
            cameraTransform: cameraTransform, commandBuffer: buffer
        )
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)
        try write(try halves(color), width: size.width, height: size.height, exposure: 0.06, name: "sky-scene-raster")
        try write(try halves(output), width: size.width, height: size.height, exposure: 0.06, name: "sky-scene-aerial")

        let tracer = try PathTracer(
            queue: queue, scene: PathTracingScene(queue: queue, graph: SceneGraph(roots: roots, camera: camera)),
            lighting: try rig.pathLighting(sky: sky.pathEnvironment)
        )
        let pathDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: size.width, height: size.height, mipmapped: false)
        pathDescriptor.storageMode = .shared
        pathDescriptor.usage = .shaderWrite
        let traced = try #require(device.makeTexture(descriptor: pathDescriptor))
        let accumulation = try tracer.makeAccumulation(width: size.width, height: size.height)
        var history: PathTracer.History?
        for _ in 0..<32 {
            let buffer = try #require(queue.makeCommandBuffer())
            history = try tracer.encode(
                camera: camera, cameraTransform: cameraTransform, destination: traced, accumulation: accumulation,
                previous: history, commandBuffer: buffer, samplesPerPixel: 16, settings: .init(maxBounces: 4)
            )
            buffer.commit()
            buffer.waitUntilCompleted()
            try #require(buffer.status == .completed)
        }
        try write(try halves(traced), width: size.width, height: size.height, exposure: 0.06, name: "sky-scene-path")
    }

    private func halves(_ texture: any MTLTexture) throws -> [SIMD3<Float>] {
        var values = [SIMD4<Float16>](repeating: .zero, count: texture.width * texture.height)
        try values.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: texture.width * 8,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return values.map { SIMD3<Float>(Float($0.x), Float($0.y), Float($0.z)) }
    }

    /// Exposes, applies Narkowicz's ACES fit and sRGB-encodes linear values into a PNG.
    private func write(_ rgb: [SIMD3<Float>], width: Int, height: Int, exposure: Float, name: String) throws {
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for (index, value) in rgb.enumerated() {
            for channel in 0..<3 {
                let v = max(value[channel] * exposure, 0)
                let mapped = simd_clamp(v * (2.51 * v + 0.03) / (v * (2.43 * v + 0.59) + 0.14), 0, 1)
                let encoded = mapped <= 0.0031308 ? 12.92 * mapped : 1.055 * pow(mapped, 1 / 2.4) - 0.055
                bytes[index * 4 + channel] = UInt8((encoded * 255).rounded())
            }
        }
        let provider = try #require(CGDataProvider(data: Data(bytes) as CFData))
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let image = try #require(CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
            space: space,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        ))
        let url = URL(fileURLWithPath: folder).appendingPathComponent(name + ".png")
        let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
    }
}
