import CoreGraphics
import Foundation
import Frostfield
import ImageIO
import Metal
import simd
import Testing
import UniformTypeIdentifiers

/// Writes the images and timings in docs/evidence/graphlang.md. It runs only when
/// FF_CAPTURE_DIR names an output directory, for example:
///
/// ```sh
/// TEST_RUNNER_FF_CAPTURE_DIR=/tmp/graphlang xcodebuild test -scheme Frostfield-Package \
///   -destination 'platform=macOS,arch=arm64' -only-testing:FrostfieldTests/GraphLanguageCapture
/// ```
@Suite(.enabled(if: ProcessInfo.processInfo.environment["FF_CAPTURE_DIR"] != nil, "Set FF_CAPTURE_DIR to capture"))
struct GraphLanguageCapture {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let directory: String
    private let camera: PerspectiveCamera

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
        directory = try #require(ProcessInfo.processInfo.environment["FF_CAPTURE_DIR"])
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 4, near: 0.1, far: 40)
    }

    private static var wide: SceneTransform {
        SceneTransform(translation: SIMD3(0, 2.0, 4.2), rotation: simd_quatf(angle: -0.42, axis: SIMD3(1, 0, 0)))
    }
    private static var closeUp: SceneTransform {
        SceneTransform(translation: SIMD3(0, 0.75, 1.55), rotation: simd_quatf(angle: -0.55, axis: SIMD3(1, 0, 0)))
    }

    @Test("Captures the graph language scene in raster and path tracing, with and without parallax")
    func capture() throws {
        let parallax = try scene(parallax: true)
        let flat = try scene(parallax: false)
        try write(try raster(parallax, view: Self.wide, width: 960, height: 540), "graphlang-raster")
        try write(try trace(parallax, view: Self.wide, width: 960, height: 540), "graphlang-path")
        try write(try raster(parallax, view: Self.closeUp, width: 960, height: 540), "graphlang-pom-raster")
        try write(try raster(flat, view: Self.closeUp, width: 960, height: 540), "graphlang-pom-flat")
        try write(try trace(parallax, view: Self.closeUp, width: 960, height: 540), "graphlang-pom-path")

        // GPU time of the close-up frame at 1920 × 1080, where the brick plane covers
        // most pixels: parallax with 32 layers against the same material without it.
        // The two alternate after a warm-up so both see the same GPU clock state. The
        // minimum is the run least disturbed by other GPU work; the median is typical.
        var times: [[Double]] = [[], []]
        for round in 0..<70 {
            for (index, roots) in [parallax, flat].enumerated() {
                let (_, duration) = try rasterTimed(roots, view: Self.closeUp, width: 1920, height: 1080)
                if round >= 10 { times[index].append(duration * 1000) }
            }
        }
        let sorted = times.map { $0.sorted() }
        print(String(
            format: "Close-up frame at 1920 × 1080 over %d alternating runs: 32-layer parallax min %.3f ms, median %.3f ms; "
                + "without parallax min %.3f ms, median %.3f ms",
            sorted[0].count, sorted[0][0], sorted[0][sorted[0].count / 2], sorted[1][0], sorted[1][sorted[1].count / 2]
        ))
    }

    // MARK: - Scene

    private func scene(parallax: Bool) throws -> [SceneNode] {
        let checker = try TextureShaderGraph { t in
            let cells = t.checker(t.gridUV * 8)
            let lines = (t.gridUV * 8).fract().remap(0, 1, -1, 1).abs()
            let edge = lines.x.max(lines.y).smoothstep(0.9, 0.97)
            let tiles = t.constant(SIMD3<Float>(0.85, 0.82, 0.74)).mix(t.constant(SIMD3<Float>(0.12, 0.25, 0.42)), by: cells)
            return t.vector(tiles.mix(t.constant(SIMD3<Float>(0.95, 0.6, 0.15)), by: edge), t.constant(Float(1)))
        }
        let bricks = try TextureShaderGraph { t in
            let uv = t.gridUV * t.constant(SIMD2<Float>(4, 8))
            let row = uv.y.floor()
            let shifted = t.vector(uv.x + row.mod(2) * 0.5, uv.y)
            let local = shifted.fract()
            let mortar = local.x.smoothstep(0.03, 0.1) * local.y.smoothstep(0.08, 0.2)
                * (1 - local.x.smoothstep(0.9, 0.97)) * (1 - local.y.smoothstep(0.8, 0.92))
            let chips = t.noise(t.vector(shifted * 9, t.constant(Float(0.5))), kind: .gradient) * 0.25
            return (mortar * (0.75 + chips)).clamped(0, 1)
        }
        let checkerTexture = try ShaderTextureGenerator(graph: checker, device: device)
            .makeTexture(width: 512, height: 512, format: .rgba8Unorm_srgb, mipmapped: true, queue: queue)
        let brickTexture = try ShaderTextureGenerator(graph: bricks, device: device)
            .makeTexture(width: 512, height: 512, format: .rgba16Float, mipmapped: true, queue: queue)

        // Object-space triplanar checker: the pattern stays attached as the sphere turns.
        let triplanar = try SurfaceShaderGraph { s in
            let local = s.modelMatrix.inverse().transformPoint(s.worldPosition)
            let color = s.texture("Checker").triplanar(position: local * 0.9, normal: s.normal, sharpness: 6)
            return .init(baseColor: color.rgb, roughness: s.constant(Float(0.35)))
        }
        let brickGraph = try SurfaceShaderGraph { s in
            let height = s.texture("Height", content: .data)
            let uv = parallax
                ? height.parallaxOcclusion(s.uv0, view: s.tangentSpace(s.viewDirection), depth: s.constant(Float(0.025)), steps: 32)
                : s.uv0
            let h = height.sample(uv).x
            let grain = s.fractalNoise(s.vector(uv * 20, s.constant(Float(1))), octaves: 3)
            let brick = s.constant(SIMD3<Float>(0.55, 0.16, 0.08)) * (0.75 + grain * 0.5)
            let color = s.constant(SIMD3<Float>(0.42, 0.40, 0.36)).mix(brick, by: h.smoothstep(0.05, 0.3))
            return .init(baseColor: color, roughness: h.remap(0, 1, 0.95, 0.6))
        }
        // Polar bands, a rotated checker, a loop of noise octaves and a branch between them.
        let spiral = try SurfaceShaderGraph { s in
            let p = s.polar(s.uv0)
            let bands = s.oscillator(.triangle, p.y * 4 + p.x * 6)
            let tiles = s.checker(s.rotateUV(s.uv0 * 4, angle: s.constant(Float(0.5)), center: SIMD2(2, 2)))
            let swirl = s.loop(count: 4, initial: s.constant(Float(0))) { i, total in
                total + s.noise(s.vector(s.uv0 * (i + 2) * 3, i), kind: .value) / 4
            }
            let glow = s.branch(tiles.greaterThan(s.constant(Float(0.5)))) {
                s.constant(SIMD3<Float>(1.0, 0.45, 0.1)) * bands
            } else: {
                s.constant(SIMD3<Float>(0.1, 0.5, 1.0)) * swirl
            }
            return .init(baseColor: s.constant(SIMD3<Float>(0.05, 0.05, 0.05)), emission: glow * 2)
        }
        let ground = try SurfaceShaderGraph { s in
            let cells = s.cellularNoise(s.worldPosition * 2.5)
            let tone = s.constant(SIMD3<Float>(0.32, 0.33, 0.35)).mix(s.constant(SIMD3<Float>(0.5, 0.48, 0.44)), by: cells.y)
            return .init(baseColor: tone * (0.8 + cells.x * 0.4), roughness: s.constant(Float(0.8)))
        }

        func mesh(_ geometry: MeshGeometry, _ graph: SurfaceShaderGraph, _ textures: [String: any MTLTexture] = [:]) throws
            -> MeshRenderer {
            try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: geometry,
                material: .physicallyBased(try .init(
                    directionToLight: simd_normalize(SIMD3(-0.5, 1, 0.7)), lightIntensity: 2.2, ambientIntensity: 0.25,
                    alphaMode: .opaque, surfaceGraph: graph, surfaceTextures: textures
                ))
            )
        }
        let floor = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
        return [
            SceneNode(transform: .init(translation: SIMD3(0, -0.001, 0), rotation: floor), mesh: try mesh(.plane(width: 8, height: 8), ground)),
            SceneNode(
                transform: .init(translation: SIMD3(-1.35, 0.65, 0), rotation: simd_quatf(angle: 0.6, axis: simd_normalize(SIMD3(0.3, 1, 0.2)))),
                mesh: try mesh(.uvSphere(radius: 0.65, segments: 64, rings: 32), triplanar, ["Checker": checkerTexture])
            ),
            SceneNode(transform: .init(translation: SIMD3(0, 0.01, 0.35), rotation: floor),
                      mesh: try mesh(.plane(width: 1.5, height: 1.5), brickGraph, ["Height": brickTexture])),
            SceneNode(
                transform: .init(translation: SIMD3(1.35, 0.55, 0), rotation: simd_quatf(angle: 0.55, axis: SIMD3(0, 1, 0)),
                                 scale: SIMD3(repeating: 1.1)),
                mesh: try mesh(.cube(), spiral)
            ),
        ]
    }

    // MARK: - Rendering

    private func target(width: Int, height: Int, usage: MTLTextureUsage) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = usage
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    private func rasterTimed(_ roots: [SceneNode], view: SceneTransform, width: Int, height: Int) throws
        -> (any MTLTexture, Double) {
        let color = try target(width: width, height: height, usage: [.renderTarget, .shaderRead])
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: width, height: height, mipmapped: false)
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.18, green: 0.2, blue: 0.24, alpha: 1)
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = try #require(device.makeTexture(descriptor: depthDescriptor))
        pass.depthAttachment.clearDepth = 1
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        let buffer = try #require(queue.makeCommandBuffer())
        try SceneGraph(roots: roots, camera: camera, cameraTransform: view).encode(pass: pass, commandBuffer: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)
        return (color, buffer.gpuEndTime - buffer.gpuStartTime)
    }

    private func raster(_ roots: [SceneNode], view: SceneTransform, width: Int, height: Int) throws -> any MTLTexture {
        try rasterTimed(roots, view: view, width: width, height: height).0
    }

    private func trace(_ roots: [SceneNode], view: SceneTransform, width: Int, height: Int) throws -> any MTLTexture {
        let sun = try PathTracer.AnalyticLight(
            PunctualLight(kind: .directional, intensity: 2.2), direction: -simd_normalize(SIMD3(-0.5, 1, 0.7)),
            angularDiameter: 0.02
        )
        let lighting = try PathTracer.Lighting(analyticLights: [sun], environment: SIMD3(0.18, 0.2, 0.24))
        let scene = try PathTracingScene(queue: queue, graph: SceneGraph(roots: roots, camera: camera, cameraTransform: view))
        let tracer = try PathTracer(queue: queue, scene: scene, lighting: lighting)
        let traced = try target(width: width, height: height, usage: [.shaderWrite, .shaderRead])
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        var history: PathTracer.History?
        for _ in 0..<16 {
            let buffer = try #require(queue.makeCommandBuffer())
            history = try tracer.encode(
                camera: camera, cameraTransform: view, destination: traced, accumulation: accumulation, previous: history,
                commandBuffer: buffer, samplesPerPixel: 16, settings: .init(maxBounces: 4)
            )
            buffer.commit()
            buffer.waitUntilCompleted()
            try #require(buffer.status == .completed)
        }
        return traced
    }

    /// Writes a PNG with a soft Reinhard shoulder and the sRGB transfer.
    private func write(_ texture: any MTLTexture, _ name: String) throws {
        let w = texture.width, h = texture.height
        var halves = [Float16](repeating: 0, count: w * h * 4)
        halves.withUnsafeMutableBytes {
            texture.getBytes($0.baseAddress!, bytesPerRow: w * 8, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        }
        var bytes = [UInt8](repeating: 255, count: w * h * 4)
        for pixel in 0..<(w * h) {
            for channel in 0..<3 {
                let linear = max(0, Float(halves[pixel * 4 + channel]))
                let mapped = linear / (1 + linear * 0.15)
                let encoded = mapped <= 0.0031308 ? mapped * 12.92 : 1.055 * pow(mapped, 1 / 2.4) - 0.055
                bytes[pixel * 4 + channel] = UInt8(max(0, min(255, (encoded * 255).rounded())))
            }
        }
        let provider = try #require(CGDataProvider(data: Data(bytes) as CFData))
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let image = try #require(CGImage(
            width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
            space: space,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        ))
        let url = URL(fileURLWithPath: directory).appendingPathComponent(name + ".png")
        let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
    }
}
