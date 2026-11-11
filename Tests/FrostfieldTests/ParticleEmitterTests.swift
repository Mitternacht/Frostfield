import Frostfield
import Metal
import simd
import Testing

struct ParticleEmitterTests {
    @Test
    func `Seeded GPU updates draw deterministically without a depth attachment`() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { try Test.cancel("No Metal device") }
        let queue = try #require(device.makeCommandQueue())
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20)
        let clip = try camera.projection(aspectRatio: 1) * simd_inverse(
            SceneTransform(translation: SIMD3(0, 0, 3)).matrix
        )
        let side = 64

        func render(seed: UInt64) throws -> [UInt8] {
            let emitter = try ParticleEmitter(queue: queue, colorPixelFormat: .rgba8Unorm, capacity: 64, seed: seed)
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba8Unorm, width: side, height: side, mipmapped: false
            )
            descriptor.storageMode = .shared
            descriptor.usage = .renderTarget
            let target = try #require(device.makeTexture(descriptor: descriptor))
            let buffer = try #require(queue.makeCommandBuffer())
            for _ in 0..<3 {
                try emitter.encodeUpdate(deltaTime: 1.0 / 60, commandBuffer: buffer)
            }
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = target
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1)
            try emitter.encodeDraw(pass: pass, clipFromWorld: clip, commandBuffer: buffer)
            buffer.commit()
            buffer.waitUntilCompleted()
            try #require(buffer.status == .completed)

            var pixels = [UInt8](repeating: 0, count: side * side * 4)
            pixels.withUnsafeMutableBytes { bytes in
                if let address = bytes.baseAddress {
                    target.getBytes(address, bytesPerRow: side * 4, from: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0)
                }
            }
            return pixels
        }

        let first = try render(seed: 23)
        let repeated = try render(seed: 23)
        let different = try render(seed: 24)
        #expect(stride(from: 0, to: first.count, by: 4).contains { first[$0] > 0 })
        #expect(first == repeated)
        #expect(first != different)
    }

    @Test("Surface flow advects deterministic particles and adds RGB to its target")
    func surfaceFlowRendering() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { try Test.cancel("No Metal device") }
        let queue = try #require(device.makeCommandQueue())
        let geometry = try MeshGeometry(vertices: [
            MeshVertex(position: SIMD3(-0.4, -0.35, 0.4), normal: SIMD3(0, 0, 1)),
            MeshVertex(position: SIMD3(0.4, -0.35, 0.4), normal: SIMD3(0, 0, 1)),
            MeshVertex(position: SIMD3(0, 0.35, 0.4), normal: SIMD3(0, 0, 1)),
        ], indices: [0, 1, 2])
        let samples = try MeshSurfaceSampler(geometry: geometry).samples(count: 48, seed: 13)
        let side = 64
        let flow = ParticleEmitter.SurfaceFlow(speed: 1, outwardSpeed: 0.1, pointSize: 8, intensity: 0.5)

        func render(steps: Int, clear: Double = 0) throws -> [UInt8] {
            let emitter = try ParticleEmitter(
                queue: queue, colorPixelFormat: .rgba8Unorm, surface: samples, flow: flow, seed: 42
            )
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba8Unorm, width: side, height: side, mipmapped: false
            )
            descriptor.storageMode = .shared
            descriptor.usage = .renderTarget
            let target = try #require(device.makeTexture(descriptor: descriptor))
            let buffer = try #require(queue.makeCommandBuffer())
            for _ in 0..<steps {
                try emitter.encodeUpdate(deltaTime: 1.0 / 60, commandBuffer: buffer)
            }
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = target
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColorMake(clear, clear, clear, 1)
            try emitter.encodeDraw(pass: pass, clipFromWorld: matrix_identity_float4x4, commandBuffer: buffer)
            buffer.commit()
            buffer.waitUntilCompleted()
            try #require(buffer.status == .completed, "\(buffer.error?.localizedDescription ?? "Metal did not finish.")")

            var pixels = [UInt8](repeating: 0, count: side * side * 4)
            try pixels.withUnsafeMutableBytes { bytes in
                target.getBytes(
                    try #require(bytes.baseAddress), bytesPerRow: side * 4,
                    from: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0
                )
            }
            return pixels
        }

        let initial = try render(steps: 0)
        let advected = try render(steps: 9)
        #expect(try render(steps: 9) == advected)
        #expect(initial != advected)

        let base: UInt8 = 32
        let overBase = try render(steps: 9, clear: Double(base) / 255)
        let visibleBlue = try #require(stride(from: 2, to: advected.count, by: 4).first {
            (24...180).contains(advected[$0])
        })
        #expect(abs(Int(overBase[visibleBlue]) - Int(advected[visibleBlue]) - Int(base)) <= 2)
        #expect(overBase[visibleBlue + 1] == 255)
    }

    @Test("Surface particles preserve occlusion against loaded mesh depth",
          arguments: [MeshDepthMode.forward, .reversed])
    func surfaceFlowDepth(depthMode: MeshDepthMode) throws {
        guard MTLCreateSystemDefaultDevice() != nil else { try Test.cancel("No Metal device") }
        let gpu = try GeometryRenderFixture()
        let normal = SIMD3<Float>(0, 0, 1)
        let near = MeshSurfaceSample(position: SIMD3(-0.7, 0, -2), normal: normal)
        let far = MeshSurfaceSample(position: SIMD3(1.4, 0, -4), normal: normal)
        let emitter = try ParticleEmitter(
            queue: gpu.queue, colorPixelFormat: .rgba16Float, depthPixelFormat: .depth32Float,
            surface: [near, far], flow: .init(pointSize: 10, color: SIMD3(1, 0, 0), intensity: 2),
            seed: 17, depthMode: depthMode
        )
        let occluder = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(1.8),
            depthMode: depthMode
        )
        func render(withOccluder: Bool) throws -> [SIMD4<Float>] {
            let target = try gpu.makeTarget()
            let pass = gpu.pass(color: target.color, depth: target.depth)
            pass.depthAttachment.storeAction = .store
            pass.depthAttachment.clearDepth = depthMode == .forward ? 1 : 0
            let command = try #require(gpu.queue.makeCommandBuffer())
            if withOccluder {
                try occluder.encode(
                    MeshScene(camera: gpu.camera, transform: SceneTransform(translation: SIMD3(0, 0, -3)),
                              materialParameters: SIMD4(0, 0, 0.25, 1)),
                    pass: pass, commandBuffer: command
                )
                pass.colorAttachments[0].loadAction = .load
                pass.depthAttachment.loadAction = .load
            }
            try emitter.encodeDraw(
                pass: pass, clipFromWorld: gpu.camera.projection(aspectRatio: 1, depthMode: depthMode),
                commandBuffer: command
            )
            try gpu.finish(command)
            return try gpu.pixels(of: target.color)
        }

        let open = try render(withOccluder: false)
        let occluded = try render(withOccluder: true)
        let nearColumn = gpu.column(x: near.position.x, depth: 2)
        let farColumn = gpu.column(x: far.position.x, depth: 4)
        let row = gpu.side / 2
        let nearPixel = gpu.pixel(occluded, x: nearColumn, y: row)
        let farPixel = gpu.pixel(occluded, x: farColumn, y: row)
        #expect(gpu.pixel(open, x: farColumn, y: row).x > 0.03)
        #expect(nearPixel.x > 0.03)
        #expect(farPixel == SIMD4(0, 0, 0.25, 1))
    }

    @Test("Surface seeds validate positions, normals and flow settings")
    func surfaceFlowValidation() throws {
        let sample = MeshSurfaceSample(position: .zero, normal: SIMD3(0, 4, 0))
        let state = try ParticleEmitter.seededFlowState(surface: [sample], flow: .init(), seed: 7)
        try #require(state.count == 3)
        #expect(state[0].w >= 0 && state[0].w < state[1].w)
        #expect(state[2].y == 1)
        #expect(throws: ParticleEmitter.Failure.flowSettings) {
            try ParticleEmitter.seededFlowState(surface: [sample], flow: .init(frequency: .nan), seed: 7)
        }
        #expect(throws: ParticleEmitter.Failure.surfaceSample(0)) {
            try ParticleEmitter.seededFlowState(
                surface: [MeshSurfaceSample(position: .zero, normal: .zero)], flow: .init(), seed: 7
            )
        }
    }

    @Test(arguments: [0, -1, ParticleEmitter.maximumCapacity + 1])
    func `Emitter rejects capacity outside its allocation bounds`(capacity: Int) {
        #expect(throws: ParticleEmitter.Failure.capacity(capacity)) {
            try ParticleEmitter.seededState(capacity: capacity, seed: 17)
        }
    }

    @Test
    func `Equal seeds start with the same particle states`() throws {
        let state = try ParticleEmitter.seededState(capacity: 32, seed: 17)
        let repeated = try ParticleEmitter.seededState(capacity: 32, seed: 17)
        let different = try ParticleEmitter.seededState(capacity: 32, seed: 18)
        #expect(state == repeated)
        #expect(state != different)
        try #require(state.count == 64)
        for particle in stride(from: 0, to: state.count, by: 2) {
            let age = state[particle].w
            let lifetime = state[particle + 1].w
            #expect(age >= 0 && age < lifetime)
            #expect(state[particle].x.isFinite && state[particle].y.isFinite && state[particle].z.isFinite)
        }
    }

    @Test(arguments: [Float(0), -1, .nan, .infinity, -.infinity])
    func `Paused and invalid time deltas skip simulation`(delta: Float) {
        #expect(ParticleEmitter.timeStep(delta) == 0)
    }

    @Test
    func `Large time gaps clamp without changing regular frame steps`() {
        let regular: Float = 1.0 / 60
        let maximum: Float = 1.0 / 30
        #expect(ParticleEmitter.timeStep(regular) == regular)
        #expect(ParticleEmitter.timeStep(12) == maximum)
    }
}
