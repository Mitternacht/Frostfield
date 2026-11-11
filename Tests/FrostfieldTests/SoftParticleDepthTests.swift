import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SoftParticleDepthTests {
    private let gpu: GeometryRenderFixture

    init() throws { gpu = try GeometryRenderFixture(side: 128) }

    @Test("Shader particles smoothly intersect a rendered sloped plane in both depth modes",
          arguments: [MeshDepthMode.forward, .reversed])
    func slopedIntersection(depthMode: MeshDepthMode) throws {
        let graph = try BufferShaderGraph(fieldCount: 2) { s in [0: s.elementField(0)] }
        let system = try ShaderParticleSystem(
            device: gpu.device, capacity: 1, simulation: graph,
            initial: [SIMD4(0, 0, -2, 0.5), SIMD4(0, 0, 0, 1)],
            blending: .alpha, defaultColor: SIMD3(1, 0.5, 0.25), defaultSize: 2.4,
            depthMode: depthMode
        )
        let pose = SceneTransform()
        let camera = try graphCamera(pose: pose, mode: depthMode)
        func draw(contrast: Float?) throws -> [SIMD4<Float>] {
            try render(camera: camera, pose: pose, mode: depthMode, depth: 2.5, slope: 1, contrast: contrast) {
                pass, snapshot, command in
                try system.encodeDraw(
                    pass: pass, camera: gpu.camera, cameraTransform: pose,
                    commandBuffer: command, softParticles: snapshot
                )
            }
        }
        let hard = try draw(contrast: nil)
        let linear = try draw(contrast: 1)
        let eased = try draw(contrast: 2)
        let middle = gpu.side / 2
        #expect(gpu.pixel(hard, x: 42, y: middle).x == 0)
        #expect(gpu.pixel(eased, x: 42, y: middle).x == 0)
        var last: Float = 0
        for x in [49, 52, 59, 64, 71, 76] {
            let reference = gpu.pixel(hard, x: x, y: middle).x
            #expect(reference > 0.25)
            let u = (Float(x) + 0.5) / Float(gpu.side) * 2 - 1
            // The quad is z = −2. The ray meets z = −2.5 − x at distance 2.5 / (1 − u).
            let gap = min(max(2.5 / (1 - u) - 2, 0), 1)
            let folded = min(gap, 1 - gap) * 2
            let curved = gap > 0.5 ? 1 - 0.5 * folded * folded : 0.5 * folded * folded
            let ratio = gpu.pixel(eased, x: x, y: middle).x / reference
            #expect(abs(gpu.pixel(linear, x: x, y: middle).x / reference - gap) < 0.004)
            #expect(abs(ratio - curved) < 0.004)
            #expect(ratio > last)
            last = ratio
        }
        let outside = gpu.pixel(hard, x: middle, y: 46)
        #expect(outside.x > 0.1 && outside.w > 0)
        #expect(gpu.pixel(eased, x: middle, y: 46) == outside)
    }

    @Test("Fixed fountain and HDR additive surface flow use the same soft depth",
          arguments: [MeshDepthMode.forward, .reversed], [false, true])
    func fixedEmitter(depthMode: MeshDepthMode, surfaceFlow: Bool) throws {
        let seed: UInt64 = 47
        let emitter: ParticleEmitter
        let pose: SceneTransform
        if surfaceFlow {
            emitter = try ParticleEmitter(
                queue: gpu.queue, colorPixelFormat: .rgba16Float, depthPixelFormat: .depth32Float,
                surface: [.init(position: SIMD3(0, 0, -2), normal: SIMD3(0, 0, 1))],
                flow: .init(pointSize: 32, color: SIMD3(1, 0.4, 0.2), intensity: 2),
                seed: seed, depthMode: depthMode
            )
            pose = .init()
        } else {
            emitter = try ParticleEmitter(
                queue: gpu.queue, colorPixelFormat: .rgba16Float, depthPixelFormat: .depth32Float,
                capacity: 1, seed: seed, depthMode: depthMode
            )
            let seeded = try ParticleEmitter.seededState(capacity: 1, seed: seed)[0]
            let position = SIMD3(seeded.x, seeded.y, seeded.z)
            pose = .init(translation: position + SIMD3(0, 0, 2))
        }
        // Also exercise orthographic unprojection through the explicit emitter camera.
        let camera = try graphCamera(pose: pose, mode: depthMode, orthographic: true)
        func draw(contrast: Float?) throws -> [SIMD4<Float>] {
            try render(camera: camera, pose: pose, mode: depthMode, depth: 2.25, slope: 0, contrast: contrast) {
                pass, snapshot, command in
                if let snapshot {
                    try emitter.encodeDraw(pass: pass, camera: camera, softParticles: snapshot, commandBuffer: command)
                } else {
                    try emitter.encodeDraw(
                        pass: pass, clipFromWorld: camera.projectionMatrix * camera.viewMatrix, commandBuffer: command
                    )
                }
            }
        }
        let hard = try draw(contrast: nil)
        let soft = try draw(contrast: 2)
        let index = (gpu.side / 2) * gpu.side + gpu.side / 2
        #expect(hard[index].x > 0.03)
        #expect(abs(soft[index].x / hard[index].x - 0.125) < 0.002)
        #expect(abs(soft[index].y / hard[index].y - 0.125) < 0.002)
    }

    @Test("Soft particles reject overlapping attachments, dimensions and wrong depth mapping")
    func validation() throws {
        let target = try gpu.makeTarget()
        let depth = try snapshot(side: gpu.side, usage: [.shaderRead, .renderTarget])
        let alias = try #require(depth.makeTextureView(pixelFormat: .depth32Float))
        let pass = gpu.pass(color: target.color, depth: alias)
        let input = try SoftParticleDepth(texture: depth)
        let emitter = try ParticleEmitter(
            queue: gpu.queue, colorPixelFormat: .rgba16Float, depthPixelFormat: .depth32Float, capacity: 1, seed: 17
        )
        let command = try #require(gpu.queue.makeCommandBuffer())
        let forward = try graphCamera(pose: .init(), mode: .forward)
        #expect(throws: SoftParticleDepth.Failure.storageOverlap) {
            try emitter.encodeDraw(pass: pass, camera: forward, softParticles: input, commandBuffer: command)
        }
        pass.depthAttachment.texture = target.depth
        let reversed = try graphCamera(pose: .init(), mode: .reversed)
        #expect(throws: SoftParticleDepth.Failure.projection) {
            try emitter.encodeDraw(pass: pass, camera: reversed, softParticles: input, commandBuffer: command)
        }
        let small = try SoftParticleDepth(texture: snapshot(side: 8))
        #expect(throws: SoftParticleDepth.Failure.attachments) {
            try emitter.encodeDraw(pass: pass, camera: forward, softParticles: small, commandBuffer: command)
        }
        #expect(throws: SoftParticleDepth.Failure.settings) { try SoftParticleDepth(texture: depth, distance: 0) }
        #expect(throws: SoftParticleDepth.Failure.settings) { try SoftParticleDepth(texture: depth, contrast: .nan) }
        #expect(throws: SoftParticleDepth.Failure.texture) { try SoftParticleDepth(texture: target.depth) }
    }

    private func graphCamera(
        pose: SceneTransform, mode: MeshDepthMode, orthographic: Bool = false
    ) throws -> ShaderGraphCamera {
        let projection: simd_float4x4
        if orthographic {
            let sign: Float = mode == .forward ? -1 : 1
            projection = simd_float4x4(columns: (
                SIMD4(0.5, 0, 0, 0), SIMD4(0, 0.5, 0, 0), SIMD4(0, 0, sign / 10, 0),
                SIMD4(0, 0, mode == .forward ? 0 : 1, 1)
            ))
        } else {
            projection = try gpu.camera.projection(aspectRatio: 1, depthMode: mode)
        }
        return try ShaderGraphCamera(
            viewMatrix: simd_inverse(pose.matrix), projectionMatrix: projection,
            near: orthographic ? 0 : gpu.camera.near, far: orthographic ? 10 : gpu.camera.far
        )
    }

    private func snapshot(side: Int, usage: MTLTextureUsage = .shaderRead) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: side, height: side, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.hazardTrackingMode = .tracked
        descriptor.usage = usage
        return try #require(gpu.device.makeTexture(descriptor: descriptor))
    }

    private func render(
        camera: ShaderGraphCamera, pose: SceneTransform, mode: MeshDepthMode, depth: Float, slope: Float,
        contrast: Float?,
        draw: (MTLRenderPassDescriptor, SoftParticleDepth?, any MTLCommandBuffer) throws -> Void
    ) throws -> [SIMD4<Float>] {
        let vertices: [MeshVertex] = [SIMD2<Float>(-1.25, -0.25), SIMD2(1.25, -0.25), SIMD2(1.25, 0.25), SIMD2(-1.25, 0.25)].map {
            MeshVertex(position: SIMD3($0.x, $0.y, -depth - slope * $0.x), normal: SIMD3(0, 0, 1))
        }
        let geometry = try MeshGeometry(vertices: vertices, indices: [0, 1, 2, 0, 2, 3])
        let black = try ClassicMaterial(model: .basic, color: .zero)
        let mesh = try gpu.unlit(geometry, material: .classic(black), depthMode: mode)
        let graph = SceneGraph(roots: [SceneNode(transform: pose, mesh: mesh)], camera: gpu.camera)
        let target = try gpu.makeTarget()
        let command = try #require(gpu.queue.makeCommandBuffer())
        let pass = gpu.pass(color: target.color, depth: target.depth, depthMode: mode)
        pass.depthAttachment.storeAction = .store
        try graph.encode(
            pass: pass, commandBuffer: command, clipFromWorld: camera.projectionMatrix * camera.viewMatrix,
            cameraPosition: pose.translation, depthMode: mode, graphCamera: camera
        )
        let input: SoftParticleDepth?
        if let contrast {
            let opaqueDepth = try snapshot(side: gpu.side)
            try TextureTransfer.encodeCopy(from: target.depth, to: opaqueDepth, commandBuffer: command)
            input = try SoftParticleDepth(texture: opaqueDepth, contrast: contrast)
        } else {
            input = nil
        }
        pass.colorAttachments[0].loadAction = .load
        pass.depthAttachment.loadAction = .load
        try draw(pass, input, command)
        try gpu.finish(command)
        return try gpu.pixels(of: target.color)
    }
}
