import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct RawMeshProgramTests {
    private let gpu: GeometryRenderFixture
    private let vertex: any MTLFunction
    private let fragment: any MTLFunction
    private let indirectFragment: any MTLFunction
    private let offset: any MTLBuffer
    private let sampler: any MTLSamplerState

    init() throws {
        gpu = try GeometryRenderFixture()
        let library = try gpu.device.makeLibrary(source: """
            #include <metal_stdlib>
            using namespace metal;

            struct RawVaryings {
                float4 position [[position]];
                float weight [[user(raw_weight)]];
            };

            vertex RawVaryings raw_vertex(
                uint id [[vertex_id]], const device float4 *geometry [[buffer(0)]],
                constant float4x4 &clip [[buffer(1)]], constant float &offset [[buffer(4)]]
            ) {
                float4 position = geometry[4 * id] + float4(offset, 0, 0, 0);
                return { clip * position, geometry[4 * id + 1].z };
            }

            fragment float4 raw_fragment(
                RawVaryings input [[stage_in]], constant float4 &tint [[buffer(0)]],
                texture2d<float> image [[texture(2)]], sampler imageSampler [[sampler(1)]]
            ) {
                return tint * input.weight * image.sample(imageSampler, float2(0.5));
            }

            struct ResourceBlock {
                texture2d<float> image [[id(0)]];
                sampler imageSampler [[id(1)]];
            };

            fragment float4 raw_indirect(
                RawVaryings input [[stage_in]], constant ResourceBlock &resources [[buffer(5)]]
            ) {
                return input.weight * resources.image.sample(resources.imageSampler, float2(0.5));
            }
            """, options: nil)
        vertex = try #require(library.makeFunction(name: "raw_vertex"))
        fragment = try #require(library.makeFunction(name: "raw_fragment"))
        indirectFragment = try #require(library.makeFunction(name: "raw_indirect"))
        offset = try #require(gpu.device.makeBuffer(length: 256 + MemoryLayout<Float>.size, options: .storageModeShared))
        offset.contents().storeBytes(of: Float(0.8), toByteOffset: 256, as: Float.self)
        sampler = try #require(gpu.device.makeSamplerState(descriptor: MTLSamplerDescriptor()))
    }

    @Test("Both raw stages draw indexed geometry and read user resources")
    func colorDraws() throws {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let texel: [UInt8] = [255, 255, 0, 255]
        texture.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: texel, bytesPerRow: 4)
        let renderer = try makeRenderer(program(sampling: texture))
        let pose = SceneTransform(translation: SIMD3(0, 0, -2))
        let tint = SIMD4<Float>(0.25, 0.5, 1, 1)
        let projected = gpu.column(x: 0.8, depth: 2)
        let middle = gpu.side / 2

        let target = try gpu.makeTarget()
        let commandBuffer = try #require(gpu.queue.makeCommandBuffer())
        try renderer.encode(
            MeshScene(camera: gpu.camera, transform: pose, materialParameters: tint),
            pass: gpu.pass(color: target.color, depth: target.depth), commandBuffer: commandBuffer
        )
        try gpu.finish(commandBuffer)
        let direct = try gpu.pixels(of: target.color)
        #expect(gpu.pixel(direct, x: middle, y: middle) == .zero)
        #expect(gpu.pixel(direct, x: projected, y: middle) == SIMD4(0.25, 0.5, 0, 1))

        let graph = SceneGraph(
            roots: [SceneNode(transform: pose, mesh: renderer, materialParameters: tint)], camera: gpu.camera
        )
        #expect(try gpu.render(graph).pixels == direct)
    }

    @Test("Raw shaders reject missing resources, target feedback and auxiliary layouts")
    func validation() throws {
        let target = try gpu.makeTarget()
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: gpu.side, height: gpu.side, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = [.renderTarget, .shaderRead]
        let color = try #require(gpu.device.makeTexture(descriptor: colorDescriptor))
        #expect(throws: RawMeshProgram.Failure.self) {
            try makeRenderer(RawMeshProgram(vertex: vertex, fragment: fragment))
        }
        #expect(throws: RawMeshProgram.Failure.functions) {
            try makeRenderer(RawMeshProgram(vertex: fragment, fragment: vertex))
        }
        let indirect = #expect(throws: RawMeshProgram.Failure.self) {
            try makeRenderer(RawMeshProgram(
                vertex: vertex, fragment: indirectFragment,
                vertexBindings: .init(buffers: [4: .init(offset, offset: 256)]),
                fragmentBindings: .init(buffers: [5: .init(offset)])
            ))
        }
        if case .bindings(let stage, let detail) = indirect {
            #expect(stage == "fragment")
            #expect(detail.contains("indirect shader resources"))
        } else {
            Issue.record("Expected an indirect resource error")
        }

        colorDescriptor.pixelFormat = .r8Uint
        colorDescriptor.usage = .shaderRead
        let integers = try #require(gpu.device.makeTexture(descriptor: colorDescriptor))
        #expect(throws: RawMeshProgram.Failure.self) { try makeRenderer(program(sampling: integers)) }

        let renderer = try makeRenderer(program(sampling: color))
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        let pass = gpu.pass(color: color, depth: target.depth)
        #expect(throws: RawMeshProgram.Failure.feedback) {
            try renderer.encode(MeshScene(camera: gpu.camera), pass: pass, commandBuffer: buffer)
        }
        let graph = SceneGraph(roots: [SceneNode(mesh: renderer)], camera: gpu.camera)
        #expect(throws: RawMeshProgram.Failure.feedback) {
            try graph.encode(pass: pass, commandBuffer: buffer)
        }
        #expect(throws: RawMeshProgram.Failure.unsupportedAuxiliaryPass) {
            try renderer.prepareSurfaceAttributes()
        }
    }

    private func program(sampling texture: any MTLTexture) -> RawMeshProgram {
        RawMeshProgram(
            vertex: vertex, fragment: fragment,
            vertexBindings: .init(buffers: [4: .init(offset, offset: 256)]),
            fragmentBindings: .init(textures: [2: texture], samplers: [1: sampler])
        )
    }

    private func makeRenderer(_ program: RawMeshProgram) throws -> MeshRenderer {
        try gpu.unlit(GeometryRenderFixture.square(0.35), material: .rawMetal(program))
    }
}
