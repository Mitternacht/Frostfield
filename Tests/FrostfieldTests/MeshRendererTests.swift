import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MeshRendererTests {
    private static let size = 128
    private static let colorFormat: MTLPixelFormat = .rgba8Unorm
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let camera: PerspectiveCamera

    init() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        self.device = device
        queue = try #require(device.makeCommandQueue())
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
    }

    @Test("Cube moves to its projected position with a rotated face normal")
    func transformedCube() throws {
        let renderer = try makeRenderer()
        var scene = MeshScene(
            camera: camera,
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            materialParameters: SIMD4(0.25, 0.5, 0.75, 1)
        )

        let centered = try render(scene, using: renderer)
        expectPixel(centered, x: 64, y: 64, rgba: SIMD4(14, 28, 191, 255))
        expectPixel(centered, x: 89, y: 51, rgba: .zero)

        scene.cameraTransform = SceneTransform(translation: SIMD3(0.15, -0.15, 0))
        scene.transform = SceneTransform(
            translation: SIMD3(0.9, 0.35, -3),
            rotation: simd_quatf(angle: .pi / 6, axis: SIMD3(0, 1, 0)),
            scale: SIMD3(0.8, 1.2, 1)
        )

        let moved = try render(scene, using: renderer)
        // In camera space the front face center is (1, 0.5, -2.57).
        // At a 90° FOV it projects near (89, 51), with normal (0.5, 0, 0.866).
        expectPixel(moved, x: 89, y: 51, rgba: SIMD4(39, 28, 171, 255))
        expectPixel(moved, x: 64, y: 64, rgba: .zero)
    }

    @Test("Nested meshes share depth and picking selects the visible mesh")
    func sceneGraphDepthAndPicking() throws {
        let renderer = try makeRenderer()
        let near = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            mesh: renderer, materialParameters: SIMD4(1, 0, 0, 1)
        )
        let far = SceneNode(
            transform: SceneTransform(translation: SIMD3(0.85, 0, 0)),
            mesh: renderer, materialParameters: SIMD4(0, 1, 0, 1)
        )
        let parent = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -4)), children: [far]
        )
        let graph = SceneGraph(roots: [near, parent], camera: camera)
        let pixels = try render { pass, buffer in
            try graph.encode(pass: pass, commandBuffer: buffer)
        }
        expectPixel(pixels, x: 73, y: 64, rgba: SIMD4(56, 0, 0, 255))
        expectPixel(pixels, x: 82, y: 64, rgba: SIMD4(0, 56, 0, 255))
        expectPixel(pixels, x: 100, y: 64, rgba: .zero)

        let overlap = try graph.pick(at: SIMD2(73.5 / 128, 64.5 / 128), aspectRatio: 1)
        let farOnly = try graph.pick(at: SIMD2(82.5 / 128, 64.5 / 128), aspectRatio: 1)
        let background = try graph.pick(at: SIMD2(100.5 / 128, 64.5 / 128), aspectRatio: 1)
        #expect(overlap?.nodeID == near.id)
        #expect(farOnly?.nodeID == far.id)
        #expect(background == nil)
    }

    @Test("Lambert intensity follows the transformed normal")
    func directionalLight() throws {
        let renderer = try makeRenderer(material: .lambert)
        var scene = MeshScene(
            camera: camera,
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            materialParameters: SIMD4(0, 0, 1, 0.5)
        )
        expectPixel(try render(scene, using: renderer), x: 64, y: 64, rgba: SIMD4(128, 128, 128, 255))

        scene.transform.rotation = simd_quatf(angle: .pi / 6, axis: SIMD3(0, 1, 0))
        scene.transform.scale = SIMD3(0.8, 1.2, 1)
        expectPixel(try render(scene, using: renderer), x: 64, y: 64, rgba: SIMD4(110, 110, 110, 255))

        scene.materialParameters.z = -1
        expectPixel(try render(scene, using: renderer), x: 64, y: 64, rgba: SIMD4(0, 0, 0, 255))
    }

    @Test("Caller shader receives new parameters on each draw")
    func callerParameters() throws {
        let library = try makeCallerLibrary()
        let renderer = try makeRenderer(material: .custom(library: library, fragment: "tint"))
        var scene = MeshScene(
            camera: camera,
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            materialParameters: SIMD4(0.2, 0.4, 0.8, 1)
        )

        expectPixel(try render(scene, using: renderer), x: 64, y: 64, rgba: SIMD4(51, 102, 204, 255))
        scene.materialParameters = SIMD4(0.6, 0.2, 0.4, 1)
        expectPixel(try render(scene, using: renderer), x: 64, y: 64, rgba: SIMD4(153, 51, 102, 255))
    }

    @Test("Caller shader rejects a missing function and incorrect buffer index")
    func invalidCallerShader() throws {
        let library = try makeCallerLibrary()

        let missing = #expect(throws: MeshRenderer.Failure.self) {
            try makeRenderer(material: .custom(library: library, fragment: "missing"))
        }
        if case .function(let name)? = missing {
            #expect(name == "missing")
        } else {
            Issue.record("Expected a missing function error")
        }

        let wrongBinding = #expect(throws: MeshRenderer.Failure.self) {
            try makeRenderer(material: .custom(library: library, fragment: "wrong_buffer"))
        }
        if case .fragmentBindings(let name)? = wrongBinding {
            #expect(name == "wrong_buffer")
        } else {
            Issue.record("Expected an incompatible fragment binding error")
        }
    }

    private func makeCallerLibrary() throws -> any MTLLibrary {
        try device.makeLibrary(source: """
            #include <metal_stdlib>
            using namespace metal;

            fragment float4 tint(constant float4& parameters [[buffer(0)]]) {
                return parameters;
            }

            fragment float4 wrong_buffer(constant float4& parameters [[buffer(1)]]) {
                return parameters;
            }
            """, options: nil)
    }

    private func makeRenderer(material: MeshMaterial = .unlit) throws -> MeshRenderer {
        try MeshRenderer(
            device: device,
            colorPixelFormat: Self.colorFormat,
            geometry: .cube(),
            material: material
        )
    }

    private func render(_ scene: MeshScene, using renderer: MeshRenderer) throws -> [UInt8] {
        try render { pass, buffer in
            try renderer.encode(scene, pass: pass, commandBuffer: buffer)
        }
    }

    private func render(
        _ encode: (MTLRenderPassDescriptor, any MTLCommandBuffer) throws -> Void
    ) throws -> [UInt8] {
        let size = Self.size
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: Self.colorFormat, width: size, height: size, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        let color = try #require(device.makeTexture(descriptor: colorDescriptor))

        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: MeshRenderer.depthPixelFormat, width: size, height: size, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1

        let commandBuffer = try #require(queue.makeCommandBuffer())
        try encode(pass, commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try #require(commandBuffer.status == .completed)

        var pixels = [UInt8](repeating: 0, count: size * size * 4)
        try pixels.withUnsafeMutableBytes { bytes in
            color.getBytes(
                try #require(bytes.baseAddress),
                bytesPerRow: size * 4,
                from: MTLRegionMake2D(0, 0, size, size),
                mipmapLevel: 0
            )
        }
        return pixels
    }

    private func expectPixel(_ pixels: [UInt8], x: Int, y: Int, rgba: SIMD4<Int>) {
        let offset = (y * Self.size + x) * 4
        for channel in 0..<4 {
            #expect(abs(Int(pixels[offset + channel]) - rgba[channel]) <= 2)
        }
    }
}
