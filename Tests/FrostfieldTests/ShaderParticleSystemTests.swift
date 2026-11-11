import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ShaderParticleSystemTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
    }

    /// position += velocity × dt and age += dt, with dt as a uniform.
    private static func ballistic() throws -> BufferShaderGraph {
        try BufferShaderGraph(fieldCount: 3) { s in
            let dt = s.uniform("DeltaTime", Float(0))
            let state = s.elementField(0)
            let velocity = s.elementField(1)
            let position = state.xyz + velocity.xyz * dt
            return [0: s.vector(position, state.w + dt)]
        }
    }

    @Test("A graph step moves soft quads and depth tests them in either convention",
          arguments: [MeshDepthMode.forward, .reversed])
    func simulateAndDraw(depthMode: MeshDepthMode) throws {
        let graph = try Self.ballistic()
        let uniforms = try ShaderUniforms(graph.uniforms)
        let initial: [SIMD4<Float>] = [
            SIMD4(0, -0.5, -2, 5), SIMD4(0, 1, 0, 10), SIMD4(1, 0.5, 0.25, 0.4),
            SIMD4(5, 0, -2, 0), SIMD4(0, 0, 0, 0), SIMD4(1, 1, 1, 0.1),
        ]
        let system = try ShaderParticleSystem(
            device: device, capacity: 2, simulation: graph, initial: initial, uniforms: uniforms, depthMode: depthMode
        )
        try uniforms.set("DeltaTime", to: 0.5)
        let step = try #require(queue.makeCommandBuffer())
        try system.encodeUpdate(commandBuffer: step)
        step.commit()
        step.waitUntilCompleted()
        let fields = system.fields()
        #expect(fields[0] == SIMD4(0, 0, -2, 5.5))
        // A particle with zero lifetime is hidden and its unconnected fields are copied.
        #expect(fields[3] == SIMD4(5, 0, -2, 0.5))
        #expect(fields[5] == SIMD4(1, 1, 1, 0.1))

        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        func center(clearDepth: Double) throws -> SIMD4<Float> {
            let size = 33
            let color = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: size, height: size, mipmapped: false)
            color.storageMode = .shared
            color.usage = .renderTarget
            let target = try #require(device.makeTexture(descriptor: color))
            let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: size, height: size, mipmapped: false)
            depthDescriptor.storageMode = .private
            depthDescriptor.usage = .renderTarget
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = target
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = try #require(device.makeTexture(descriptor: depthDescriptor))
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.clearDepth = clearDepth
            let buffer = try #require(queue.makeCommandBuffer())
            try system.encodeDraw(pass: pass, camera: camera, cameraTransform: .init(), commandBuffer: buffer)
            buffer.commit()
            buffer.waitUntilCompleted()
            var value = SIMD4<Float16>.zero
            try withUnsafeMutableBytes(of: &value) { target.getBytes(try #require($0.baseAddress), bytesPerRow: 8, from: MTLRegionMake2D(16, 16, 1, 1), mipmapLevel: 0) }
            return SIMD4<Float>(value)
        }
        // Mid-life, the quad's center has full alpha and the particle's own color.
        let clearDepth: Double = depthMode == .forward ? 1 : 0
        let lit = try center(clearDepth: clearDepth)
        #expect(simd_distance(lit, SIMD4(1, 0.5, 0.25, 1)) < 0.01, "\(lit)")
        // The opposite endpoint occludes the particle.
        #expect(try center(clearDepth: 1 - clearDepth) == .zero)
    }

    @Test("Particle systems validate their fields and appearance")
    func validation() throws {
        let graph = try Self.ballistic()
        #expect(throws: ShaderParticleSystem.Failure.initialValues) {
            try ShaderParticleSystem(device: device, capacity: 2, simulation: graph, initial: [SIMD4<Float>](repeating: .zero, count: 5))
        }
        let single = try BufferShaderGraph(fieldCount: 1) { s in [0: s.elementField(0)] }
        #expect(throws: ShaderParticleSystem.Failure.fieldCount) {
            try ShaderParticleSystem(device: device, capacity: 1, simulation: single, initial: [.zero])
        }
        #expect(throws: ShaderParticleSystem.Failure.uniforms) {
            try ShaderParticleSystem(device: device, capacity: 1, simulation: graph, initial: [.zero, .zero, .zero],
                                     uniforms: try ShaderUniforms([ShaderUniform("Other", Float(0))]))
        }
        #expect(throws: ShaderParticleSystem.Failure.appearance) {
            try ShaderParticleSystem(device: device, capacity: 1, simulation: graph, initial: [.zero, .zero, .zero], defaultSize: 0)
        }
    }
}
