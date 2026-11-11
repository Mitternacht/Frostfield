import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SurfaceShaderRenderingTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let camera: PerspectiveCamera
    private let color: any MTLTexture
    private let pass: MTLRenderPassDescriptor
    private static let size = 65

    init() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        self.device = device
        queue = try #require(device.makeCommandQueue())
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20)
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: Self.size, height: Self.size, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        color = try #require(device.makeTexture(descriptor: colorDescriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: Self.size, height: Self.size, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))
        pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.clearDepth = 1
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
    }

    @Test("Generated surface inputs drive PBR color and emission independently of alpha")
    func generatedChannelsAndCoverage() throws {
        let graph = try SurfaceShaderGraph(nodes: [
            .color(SIMD3(0.4, 0.2, 0.1)), .scalar(1), .parameter(.x), .parameter(.w), .scalar(0),
            .rgb(red: 2, green: 3, blue: 4), .scalar(0.7),
        ], outputs: .init(baseColor: 0, metallic: 1, roughness: 6, emission: 5))
        let material = try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0.5, baseColorFactor: SIMD4(1, 1, 1, 0.5), surfaceGraph: graph
        )
        let renderer = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(), material: .physicallyBased(material)
        )
        func render(_ parameters: SIMD4<Float>) throws -> SIMD4<Float> {
            try renderScene([SceneNode(
                transform: .init(translation: SIMD3(0, 0, -2)), mesh: renderer, materialParameters: parameters
            )])
            return try readPixel()
        }
        let red = try render(SIMD4(0.75, 0, 0, 0))
        let green = try render(SIMD4(0.25, 0, 0, 1))
        for (actual, expected) in [(red, SIMD4<Float>(0.475, 0.05, 0.025, 0.5)),
                                   (green, SIMD4<Float>(0.225, 0.55, 0.025, 0.5))] {
            for channel in 0..<4 {
                #expect(abs(actual[channel] - expected[channel]) < 0.002)
            }
        }
    }

    @Test("Generated masked PBR meshes cast and receive standard directional shadows")
    func maskedSurfaceParticipatesInShadows() throws {
        let surface = try SurfaceShaderGraph(nodes: [.color(SIMD3(repeating: 0.8))], outputs: .init(baseColor: 0))
        let receiver = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 3, height: 3),
            material: .physicallyBased(try .init(
                roughness: 0.8, ambientIntensity: 0, receivesShadows: true, alphaMode: .opaque, surfaceGraph: surface
            ))
        )
        let blocker = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .cube(),
            material: .physicallyBased(try .init(
                lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(1, 1, 1, 0.75),
                alphaMode: .mask(cutoff: 0.5), surfaceGraph: surface
            ))
        )
        let shadow = try DirectionalShadowMap(
            queue: queue, resolution: 128, directionToLight: SIMD3(1, 0, 1),
            center: SIMD3(0, 0, -3), halfExtent: 2, intensity: 2
        )
        let wall = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: receiver, castsShadow: false)
        func center(castsShadow: Bool) throws -> SIMD4<Float> {
            let occluder = SceneNode(
                transform: .init(translation: SIMD3(0.65, 0, -2.35), scale: SIMD3(repeating: 0.4)),
                mesh: blocker, materialParameters: .zero, castsShadow: castsShadow
            )
            try renderScene([wall, occluder], shadowMap: shadow)
            return try readPixel()
        }
        let lit = try center(castsShadow: false)
        let shadowed = try center(castsShadow: true)
        #expect(lit.x > 0.2)
        #expect(shadowed.x < lit.x * 0.25)
        #expect(lit.w == 1 && shadowed.w == 1)
    }

    @Test("Path tracing dispatches multiple generated shaders with the same graph inputs as raster PBR")
    func generatedSurfacesMatchAcrossRenderers() throws {
        let leftGraph = try SurfaceShaderGraph(nodes: [
            .color(SIMD3(1, 0, 0)), .color(SIMD3(0, 1, 0)), .parameter(.x), .mix(0, 1, factor: 2),
        ], outputs: .init(emission: 3))
        let rightGraph = try SurfaceShaderGraph(nodes: [
            .color(SIMD3(0, 0, 1)), .parameter(.w), .multiply(0, 1),
        ], outputs: .init(emission: 2))
        let plane = try MeshGeometry.plane(width: 2, height: 4)
        let nodes = try [(leftGraph, Float(-1)), (rightGraph, Float(1))].map { graph, x in
            let mesh = try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: plane,
                material: .physicallyBased(try .init(lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, surfaceGraph: graph))
            )
            return SceneNode(transform: .init(translation: SIMD3(x, 0, -2)), mesh: mesh,
                             materialParameters: SIMD4(0.25, 0, 0, 0.4))
        }
        try renderScene(nodes)
        let graph = SceneGraph(roots: nodes, camera: camera)
        let scene = try PathTracingScene(queue: queue, graph: graph)
        let tracer = try PathTracer(queue: queue, scene: scene)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: Self.size, height: Self.size, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderWrite
        let traced = try #require(device.makeTexture(descriptor: descriptor))
        let accumulation = try tracer.makeAccumulation(width: Self.size, height: Self.size)
        let buffer = try #require(queue.makeCommandBuffer())
        _ = try tracer.encode(camera: camera, cameraTransform: .init(), destination: traced,
                              accumulation: accumulation, commandBuffer: buffer, samplesPerPixel: 4)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)

        for (x, expected) in [(16, SIMD4<Float>(0.75, 0.25, 0, 1)), (48, SIMD4<Float>(0, 0, 0.4, 1))] {
            let raster = try readPixel(x: x)
            let path = try readPixel(from: traced, x: x)
            for channel in 0..<4 {
                #expect(abs(raster[channel] - expected[channel]) < 0.002)
                #expect(abs(path[channel] - raster[channel]) < 0.002)
            }
        }
    }

    private func renderScene(_ roots: [SceneNode], shadowMap: DirectionalShadowMap? = nil) throws {
        let graph = SceneGraph(roots: roots, camera: camera)
        let buffer = try #require(queue.makeCommandBuffer())
        try graph.encode(pass: pass, commandBuffer: buffer, shadowMap: shadowMap)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)
    }

    private func readPixel(from texture: (any MTLTexture)? = nil, x: Int = Self.size / 2) throws -> SIMD4<Float> {
        var pixel = SIMD4<Float16>.zero
        try withUnsafeMutableBytes(of: &pixel) { bytes in
            (texture ?? color).getBytes(
                try #require(bytes.baseAddress), bytesPerRow: 8,
                from: MTLRegionMake2D(x, Self.size / 2, 1, 1), mipmapLevel: 0
            )
        }
        return SIMD4<Float>(pixel)
    }
}
