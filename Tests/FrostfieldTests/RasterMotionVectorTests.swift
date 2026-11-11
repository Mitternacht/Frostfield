import Foundation
import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct RasterMotionVectorTests {
    private let gpu: GeometryRenderFixture

    init() throws { gpu = try GeometryRenderFixture(side: 64) }

    @Test("Object, camera and jitter motion use previous-minus-current pixels in both depth conventions",
          arguments: [MeshDepthMode.forward, .reversed], [false, true])
    func rigidMotion(mode: MeshDepthMode, rebased: Bool) throws {
        let mesh = try gpu.unlit(.plane(width: 1.4, height: 1.4), depthMode: mode)
        let blocker = try gpu.unlit(.plane(width: 0.16, height: 0.16), depthMode: mode)
        let nodeID = UUID()
        let origin = rebased ? SIMD3<Float>(1e9, -1e9, 1e9) : .zero
        let parentID = UUID()
        func graph(position: SIMD3<Float>, cameraX: Float, jitter: SIMD2<Float>, addBlocker: Bool) -> SceneGraph {
            var children = [SceneNode(id: nodeID, transform: .init(translation: position), mesh: mesh)]
            if addBlocker {
                children.append(SceneNode(transform: .init(translation: SIMD3(0.25 + cameraX, -0.125, -1)), mesh: blocker))
            }
            var result = SceneGraph(
                roots: [SceneNode(id: parentID, transform: .init(translation: origin), children: children)],
                camera: gpu.camera, cameraTransform: .init(translation: origin + SIMD3(cameraX, 0, 0))
            )
            result.projectionJitter = jitter
            result.highPrecision = rebased
            return result
        }
        let motion = try RasterMotionVectors(queue: gpu.queue)
        let first = try render(graph(position: SIMD3(-0.25, 0.25, -2), cameraX: 0, jitter: SIMD2(0.125, -0.25), addBlocker: false),
                               motion: motion, mode: mode)
        #expect(first.pixels.allSatisfy { $0 == .zero })
        let second = try render(graph(position: SIMD3(0.25, -0.25, -2), cameraX: rebased ? 0 : 0.125,
                                      jitter: SIMD2(-0.375, 0.125), addBlocker: true),
                                motion: motion, mode: mode, previous: first.history)
        let pixel = gpu.pixel(second.pixels, x: rebased ? 35 : 34, y: 36)
        let clip = try gpu.camera.projection(aspectRatio: 1, depthMode: mode) * SIMD4(0, 0, -2, 1)
        #expect(abs(pixel.x - (rebased ? -7.5 : -5.5)) < 0.02)
        #expect(abs(pixel.y + 8.375) < 0.02)
        #expect(abs(pixel.z - clip.z / clip.w) < 0.001 && pixel.w == 1)
        #expect(gpu.pixel(second.pixels, x: 40, y: 36) == .zero)
        #expect(gpu.pixel(second.pixels, x: 5, y: 5) == .zero)
    }

    @Test("Stored scene depth keeps a moving material's cutout holes on the underlying surface")
    func maskVisibility() throws {
        let graph = try SurfaceShaderGraph { b in
            .init(mask: b.uv0.x.lessThan(b.constant(Float(0.5))))
        }
        let foreground = try gpu.unlit(
            .plane(width: 2, height: 2),
            material: .physicallyBased(.init(surfaceGraph: graph, shadingModel: .unlit))
        )
        let background = try gpu.unlit(.plane(width: 4, height: 4))
        let nodeID = UUID()
        let backdrop = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: background)
        func scene(x: Float) -> SceneGraph {
            SceneGraph(roots: [
                backdrop, SceneNode(id: nodeID, transform: .init(translation: SIMD3(x, 0, -2)), mesh: foreground)
            ], camera: gpu.camera)
        }
        let motion = try RasterMotionVectors(queue: gpu.queue)
        let first = try render(scene(x: -0.25), motion: motion)
        let second = try render(scene(x: 0.25), motion: motion, previous: first.history)
        let left = gpu.pixel(second.pixels, x: 28, y: 32)
        let hole = gpu.pixel(second.pixels, x: 42, y: 32)
        let projection = try gpu.camera.projection(aspectRatio: 1)
        let backgroundClip = projection * SIMD4(0, 0, -3, 1)
        #expect(abs(left.x + 8) < 0.02 && left.w == 1)
        #expect(abs(hole.x) < 0.002 && abs(hole.y) < 0.002 && hole.w == 1)
        #expect(abs(hole.z - backgroundClip.z / backgroundClip.w) < 0.001)
    }

    @Test("Instance transforms retain slot history until a bulk replacement")
    func instanceHistory() throws {
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.45, height: 0.45),
            supportsInstancing: true
        )
        func transform(_ x: Float) -> simd_float4x4 { SceneTransform(translation: SIMD3(x, 0, 0)).matrix }
        let instances = try InstanceSet(device: gpu.device, transforms: [transform(-0.5), transform(0.5)])
        let scene = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh, instances: instances)
        ], camera: gpu.camera)
        let motion = try RasterMotionVectors(queue: gpu.queue)
        let first = try render(scene, motion: motion)
        try instances.setTransform(transform(-0.25), at: 0)
        let second = try render(scene, motion: motion, previous: first.history)
        let changed = gpu.pixel(second.pixels, x: 28, y: 32)
        let stable = gpu.pixel(second.pixels, x: 40, y: 32)
        #expect(abs(changed.x + 4) < 0.02 && changed.w == 1)
        #expect(abs(stable.x) < 0.002 && stable.w == 1)
        try instances.replaceAll(transforms: instances.transforms)
        let third = try render(scene, motion: motion, previous: second.history)
        #expect(third.pixels.allSatisfy { $0 == .zero })
    }

    @Test("Inline skin palettes and morph-only weights each produce posed motion", arguments: [false, true])
    func individualPoses(skinned: Bool) throws {
        let geometry = try MeshGeometry.plane(width: 1.4, height: 1.4)
        let offset = SIMD3<Float>(0.25, 0.125, 0)
        let skin = try MeshSkin(
            jointCount: 1,
            influences: Array(repeating: VertexSkinInfluence(joints: .zero, weights: SIMD4(1, 0, 0, 0)),
                              count: geometry.vertices.count)
        )
        let targets = try MeshMorphTargets(
            vertexCount: geometry.vertices.count,
            targets: [.init(positionOffsets: Array(repeating: offset, count: geometry.vertices.count))]
        )
        let mesh = try gpu.unlit(geometry, skin: skinned ? skin : nil, morphTargets: skinned ? nil : targets)
        let id = UUID()
        func scene(posed: Bool) throws -> SceneGraph {
            let palette = skinned ? try JointPalette(matrices: [
                SceneTransform(translation: posed ? offset : .zero).matrix
            ]) : nil
            let node = SceneNode(
                id: id, transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh, jointPalette: palette,
                morphWeights: !skinned && posed ? [1] : nil
            )
            return SceneGraph(roots: [node], camera: gpu.camera)
        }
        let motion = try RasterMotionVectors(queue: gpu.queue)
        let before = try render(scene(posed: !skinned), motion: motion)
        let after = try render(scene(posed: skinned), motion: motion, previous: before.history)
        let vector = gpu.pixel(after.pixels, x: 32, y: 32)
        let sign: Float = skinned ? 1 : -1
        #expect(abs(vector.x + 4 * sign) < 0.01 && abs(vector.y - 2 * sign) < 0.01 && vector.w == 1)
    }

    @Test("Perspective motion replays morph, buffered skin and instance changes in vertex-stage order",
          arguments: [MeshDepthMode.forward, .reversed])
    func combinedPoses(mode: MeshDepthMode) throws {
        let geometry = try MeshGeometry.plane(width: 1, height: 1)
        let skin = try MeshSkin(
            jointCount: 65,
            influences: Array(repeating: VertexSkinInfluence(joints: SIMD4(64, 0, 0, 0), weights: SIMD4(1, 0, 0, 0)),
                              count: geometry.vertices.count)
        )
        let targets = try MeshMorphTargets(
            vertexCount: geometry.vertices.count,
            targets: [.init(positionOffsets: Array(repeating: SIMD3(0.5, 0, 0), count: geometry.vertices.count))]
        )
        let mesh = try gpu.unlit(geometry, skin: skin, morphTargets: targets, supportsInstancing: true, depthMode: mode)
        let instances = try InstanceSet(device: gpu.device, transforms: [matrix_identity_float4x4])
        let id = UUID()
        func scene(posed: Bool) throws -> SceneGraph {
            var joints = Array(repeating: matrix_identity_float4x4, count: 65)
            if posed { joints[64] = SceneTransform(translation: SIMD3(0, 0, 0.5), scale: SIMD3(2, 1, 1)).matrix }
            let node = SceneNode(
                id: id, transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh,
                jointPalette: try JointPalette(matrices: joints), instances: instances, morphWeights: posed ? [0.5] : nil
            )
            return SceneGraph(roots: [node], camera: gpu.camera)
        }
        let motion = try RasterMotionVectors(queue: gpu.queue)
        let first = try render(scene(posed: false), motion: motion, mode: mode)
        #expect(first.pixels.allSatisfy { $0 == .zero })
        try instances.setTransform(SceneTransform(translation: SIMD3(-0.25, 0, 0)).matrix, at: 0)
        let second = try render(scene(posed: true), motion: motion, mode: mode, previous: first.history)
        let priorClip = try gpu.camera.projection(aspectRatio: 1, depthMode: mode) * SIMD4(0, 0, -2, 1)
        for x in [24, 36, 49] {
            let sampleX = Float(x) + 0.5
            let currentWorldX = (sampleX - 32) * (1.5 / 32)
            // Current pose is skin × (base + 0.25), then instance translation.
            let priorWorldX = (currentWorldX + 0.25) / 2 - 0.25
            let vector = gpu.pixel(second.pixels, x: x, y: 32)
            #expect(abs(vector.x - (32 + 16 * priorWorldX - sampleX)) < 0.02 && abs(vector.y + 0.125) < 0.01)
            #expect(abs(vector.z - priorClip.z / priorClip.w) < 0.001 && vector.w == 1)
        }
        #expect(gpu.pixel(second.pixels, x: 5, y: 5) == .zero)
    }

    @Test("Motion snapshots keep the GPU-displaced pose when intervening draws overwrite it",
          arguments: [MeshDepthMode.forward, .reversed])
    func displacedVertexHistory(mode: MeshDepthMode) throws {
        let graph = try SurfaceShaderGraph { builder in
            .init(positionOffset: builder.uniform("Offset", SIMD3<Float>.zero))
        }
        let uniforms = try ShaderUniforms(graph: graph)
        let mesh = try gpu.unlit(
            .plane(width: 0.5, height: 0.5),
            material: .physicallyBased(.init(
                alphaMode: .opaque, surfaceGraph: graph, surfaceUniforms: uniforms, shadingModel: .unlit
            )),
            depthMode: mode
        )
        let nodes = [-0.7 as Float, 0.7].map {
            SceneNode(transform: .init(translation: SIMD3($0, 0, -2)), mesh: mesh)
        }
        let scene = SceneGraph(roots: nodes, camera: gpu.camera)
        let motion = try RasterMotionVectors(queue: gpu.queue)

        func encode(
            _ offset: SIMD3<Float>, recording: RasterMotionVectors.Recording? = nil
        ) throws -> any MTLCommandBuffer {
            try uniforms.set("Offset", to: offset)
            let targets = try gpu.makeTarget()
            let pass = gpu.pass(color: targets.color, depth: targets.depth, depthMode: mode)
            pass.depthAttachment.storeAction = .store
            let command = try #require(gpu.queue.makeCommandBuffer())
            try scene.encode(pass: pass, commandBuffer: command, depthMode: mode, motionVectors: recording)
            return command
        }

        let firstOutput = try texture()
        let first = motion.recording(to: firstOutput)
        let accepted = try encode(SIMD3(-0.25, 0, 0), recording: first)
        accepted.commit()
        let history = try #require(first.history)
        let unrecorded = try encode(SIMD3(0.75, 0, 0.25))
        unrecorded.commit()

        let output = try texture()
        let last = motion.recording(to: output, previous: history)
        let resolved = try encode(SIMD3(0.25, 0, 0.5), recording: last)
        try gpu.finish(resolved)
        #expect(accepted.status == .completed && unrecorded.status == .completed)
        #expect(try gpu.pixels(of: firstOutput).allSatisfy { $0 == .zero })

        let pixels = try gpu.pixels(of: output)
        let priorClip = try gpu.camera.projection(aspectRatio: 1, depthMode: mode) * SIMD4(0, 0, -2, 1)
        for x in [22, 52] {
            let currentPixel = Float(x) + 0.5
            let currentWorldX = (currentPixel - 32) * (1.5 / 32)
            let priorPixel = 32 + (currentWorldX - 0.5) * 16
            let vector = gpu.pixel(pixels, x: x, y: 32)
            #expect(abs(vector.x - (priorPixel - currentPixel)) < 0.02 && abs(vector.y + 0.125) < 0.01)
            #expect(abs(vector.z - priorClip.z / priorClip.w) < 0.001 && vector.w == 1)
        }
        #expect(gpu.pixel(pixels, x: 5, y: 5) == .zero)

        let otherQueue = try #require(gpu.device.makeCommandQueue())
        let unrelated = try RasterMotionVectors(queue: otherQueue).recording(to: output, previous: history)
        let targets = try gpu.makeTarget()
        let pass = gpu.pass(color: targets.color, depth: targets.depth, depthMode: mode)
        pass.depthAttachment.storeAction = .store
        let command = try #require(otherQueue.makeCommandBuffer())
        #expect(throws: RasterMotionVectors.Failure.history) {
            try scene.encode(pass: pass, commandBuffer: command, depthMode: mode, motionVectors: unrelated)
        }
    }

    @Test("Curve placements retain their last recorded pose across unrecorded draws",
          arguments: [CurveFlow.Execution.cpu, .metal], [MeshDepthMode.forward, .reversed])
    func curveFlowHistory(execution: CurveFlow.Execution, mode: MeshDepthMode) throws {
        func spine(y: Double) throws -> CurveDeformer {
            try CurveDeformer(curve: Curve3D(position: { SIMD3(-1 + 2 * $0, y, 0) }), subdivisions: 8)
        }
        let flow = try CurveFlow(
            device: gpu.device, mesh: .plane(width: 0.25, height: 0.25),
            placements: [.init(curve: spine(y: -0.35), offset: 0.5), .init(curve: spine(y: 0.35), offset: 1.5)],
            execution: execution
        )
        let mesh = try MeshRenderer(curveFlow: flow, colorPixelFormat: .rgba16Float, depthMode: mode)
        let scene = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)], camera: gpu.camera
        )
        let motion = try RasterMotionVectors(queue: gpu.queue)
        let first = try render(scene, motion: motion, mode: mode)
        try flow.move(along: 0.5, at: 0)
        _ = try gpu.render(scene, depthMode: mode)
        try flow.move(along: 0.25, at: 0)
        let result = try render(scene, motion: motion, mode: mode, previous: first.history)
        let moved = gpu.pixel(result.pixels, x: gpu.column(x: 0.25, depth: 2), y: gpu.row(y: -0.35, depth: 2))
        let stable = gpu.pixel(result.pixels, x: gpu.column(x: 0.5, depth: 2), y: gpu.row(y: 0.35, depth: 2))
        #expect(abs(moved.x + 12) < 0.02 && abs(moved.y) < 0.02 && moved.w == 1)
        #expect(abs(stable.x) < 0.02 && abs(stable.y) < 0.02 && stable.w == 1)
    }

    @Test("Dynamic mesh motion follows declared correspondence and restarts on replacement",
          arguments: [MeshDepthMode.forward, .reversed])
    func dynamicGeometryHistory(mode: MeshDepthMode) throws {
        let source = try MeshGeometry.plane(width: 0.8, height: 0.8)
        func translated(_ x: Float, indices: [UInt32]? = nil) throws -> MeshGeometry {
            try MeshGeometry(vertices: source.vertices.map {
                MeshVertex(position: $0.position + SIMD3(x, 0, 0), normal: $0.normal,
                           textureCoordinate: $0.textureCoordinate)
            }, indices: indices ?? source.indices)
        }
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, dynamicGeometry: source,
            capacity: .init(vertices: 32, triangles: 16), depthMode: mode
        )
        let scene = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)], camera: gpu.camera
        )
        let motion = try RasterMotionVectors(queue: gpu.queue)
        let first = try render(scene, motion: motion, mode: mode)
        let invalid = try translated(0.25, indices: Array(source.indices.reversed()))
        #expect(throws: MeshRenderer.Failure.dynamicGeometryTopology) {
            try mesh.updateGeometry(invalid, preservingVertexIdentity: true)
        }
        #expect(mesh.geometryRevision == 0 && mesh.geometry.indices == source.indices)

        try mesh.updateGeometry(translated(0.125), preservingVertexIdentity: true)
        _ = try gpu.render(scene, depthMode: mode)
        let shifted = try translated(0.25)
        try mesh.updateGeometry(shifted, preservingVertexIdentity: true)
        let second = try render(scene, motion: motion, mode: mode, previous: first.history)
        let moved = gpu.pixel(second.pixels, x: 32, y: 32)
        #expect(abs(moved.x + 4) < 0.02 && abs(moved.y) < 0.02 && moved.w == 1)

        try mesh.updateGeometry(shifted)
        let replaced = try render(scene, motion: motion, mode: mode, previous: second.history)
        #expect(replaced.pixels.allSatisfy { $0 == .zero })
        try mesh.updateGeometry(translated(0.5), preservingVertexIdentity: true)
        let recovered = try render(scene, motion: motion, mode: mode, previous: replaced.history)
        let movedAgain = gpu.pixel(recovered.pixels, x: 40, y: 32)
        #expect(abs(movedAgain.x + 4) < 0.02 && abs(movedAgain.y) < 0.02 && movedAgain.w == 1)
    }

    @Test("Scene motion drives blur on the moving edge and preserves invalid pixels")
    func motionBlurIntegration() throws {
        let mesh = try gpu.unlit(.plane(width: 1, height: 1))
        let id = UUID()
        let vectors = try texture()
        let source = try texture()
        let blurred = try texture()
        let targets = try gpu.makeTarget()
        let motion = try RasterMotionVectors(queue: gpu.queue)
        let effect = try MotionBlur(device: gpu.device, motionTexture: vectors, settings: .init(sampleCount: 3))

        func draw(x: Float, previous: RasterMotionVectors.History? = nil) throws -> RasterMotionVectors.History {
            let scene = SceneGraph(roots: [
                SceneNode(id: id, transform: .init(translation: SIMD3(x, 0, -2)), mesh: mesh)
            ], camera: gpu.camera)
            let pass = gpu.pass(color: source, depth: targets.depth)
            pass.depthAttachment.storeAction = .store
            let recording = motion.recording(to: vectors, previous: previous)
            let command = try #require(gpu.queue.makeCommandBuffer())
            try scene.encode(pass: pass, commandBuffer: command, motionVectors: recording)
            try effect.encode(source: source, destination: blurred, commandBuffer: command)
            try gpu.finish(command)
            return try #require(recording.history)
        }

        let history = try draw(x: -0.25)
        #expect(try gpu.pixels(of: source) == gpu.pixels(of: blurred))
        _ = try draw(x: 0.25, previous: history)

        let sharp = try gpu.pixels(of: source)
        let result = try gpu.pixels(of: blurred)
        let left = gpu.pixel(sharp, x: 28, y: 32)
        let right = gpu.pixel(sharp, x: 43, y: 32)
        #expect(left.w == 1 && right.w == 1)
        #expect(isNearly(gpu.pixel(result, x: 28, y: 32), left / 2))
        #expect(gpu.pixel(result, x: 43, y: 32) == right)
        #expect(gpu.pixel(result, x: 27, y: 32) == .zero)
    }

    @Test("Scene motion reprojects moved surface history and rejects the newly revealed background",
          arguments: [MeshDepthMode.forward, .reversed])
    func temporalIntegration(mode: MeshDepthMode) throws {
        let surface = try SurfaceShaderGraph { builder in
            .init(baseColor: builder.parameters.rgb)
        }
        let material = try PhysicallyBasedMaterial(alphaMode: .opaque, surfaceGraph: surface, shadingModel: .unlit)
        let foreground = try gpu.unlit(.plane(width: 0.75, height: 0.75), material: .physicallyBased(material), depthMode: mode)
        let background = try gpu.unlit(.plane(width: 8, height: 8), material: .physicallyBased(material), depthMode: mode)
        let frontID = UUID()
        let back = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -4)), mesh: background, materialParameters: SIMD4(0, 1, 0, 1)
        )
        let source = try texture()
        let vectors = try texture()
        let resolved = try texture()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: gpu.side, height: gpu.side, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = [.renderTarget, .shaderRead]
        let depth = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let projection = try gpu.camera.projection(aspectRatio: 1, depthMode: mode)
        let motion = try RasterMotionVectors(queue: gpu.queue)
        let taa = try TemporalAntiAliasing(device: gpu.device, settings: .accumulation)

        func draw(x: Float, tint: SIMD4<Float>, jitter: SIMD2<Float>,
                  previous: RasterMotionVectors.History? = nil) throws -> RasterMotionVectors.History {
            var scene = SceneGraph(roots: [
                back, SceneNode(id: frontID, transform: .init(translation: SIMD3(x, 0, -2)),
                                mesh: foreground, materialParameters: tint)
            ], camera: gpu.camera)
            scene.projectionJitter = jitter
            let pass = gpu.pass(color: source, depth: depth, depthMode: mode)
            pass.depthAttachment.storeAction = .store
            let recording = motion.recording(to: vectors, previous: previous)
            let command = try #require(gpu.queue.makeCommandBuffer())
            try scene.encode(pass: pass, commandBuffer: command, depthMode: mode, motionVectors: recording)
            try taa.encode(
                source: source, depth: depth, clipFromWorld: projection, jitter: jitter, depthMode: mode,
                motionTexture: vectors, destination: resolved, commandBuffer: command
            )
            try gpu.finish(command)
            return try #require(recording.history)
        }

        let history = try draw(x: -0.625, tint: SIMD4(1, 0, 0, 1), jitter: SIMD2(0.25, -0.25))
        #expect(try gpu.pixels(of: source) == gpu.pixels(of: resolved))
        let oldColor = try gpu.pixels(of: source)
        _ = try draw(x: 0.625, tint: SIMD4(0, 0, 1, 1), jitter: SIMD2(-0.5, 0.125), previous: history)

        let current = try gpu.pixels(of: source)
        let actual = try gpu.pixels(of: resolved)
        let oldForeground = gpu.pixel(oldColor, x: 22, y: 32)
        let newForeground = gpu.pixel(current, x: 42, y: 32)
        let revealed = gpu.pixel(current, x: 22, y: 32)
        #expect(oldForeground.x > 0.9 && newForeground.z > 0.9 && revealed.y > 0.9)
        #expect(isNearly(gpu.pixel(actual, x: 42, y: 32), (oldForeground + newForeground) / 2))
        #expect(gpu.pixel(actual, x: 22, y: 32) == revealed)
    }

    @Test("The raster motion request validates stored depth and distinct attachments before encoding")
    func invalidTargets() throws {
        let motion = try RasterMotionVectors(queue: gpu.queue)
        let scene = SceneGraph(roots: [], camera: gpu.camera)
        let targets = try gpu.makeTarget()
        let pass = gpu.pass(color: targets.color, depth: targets.depth)
        let output = try texture()
        let command = try #require(gpu.queue.makeCommandBuffer())
        let recording = motion.recording(to: output)
        #expect(throws: RasterMotionVectors.Failure.depth) {
            try scene.encode(pass: pass, commandBuffer: command, motionVectors: recording)
        }
        pass.depthAttachment.storeAction = .store
        pass.colorAttachments[0].texture = output
        #expect(throws: RasterMotionVectors.Failure.destination) {
            try scene.encode(pass: pass, commandBuffer: command, motionVectors: recording)
        }
        #expect(recording.history == nil)
    }

    private func render(
        _ scene: SceneGraph, motion: RasterMotionVectors, mode: MeshDepthMode = .forward,
        previous: RasterMotionVectors.History? = nil
    ) throws -> (pixels: [SIMD4<Float>], history: RasterMotionVectors.History) {
        let targets = try gpu.makeTarget()
        let pass = gpu.pass(color: targets.color, depth: targets.depth, depthMode: mode)
        pass.depthAttachment.storeAction = .store
        let output = try texture()
        let recording = motion.recording(to: output, previous: previous)
        let command = try #require(gpu.queue.makeCommandBuffer())
        try scene.encode(pass: pass, commandBuffer: command, depthMode: mode, motionVectors: recording)
        try gpu.finish(command)
        return (try gpu.pixels(of: output), try #require(recording.history))
    }

    private func texture() throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: RasterMotionVectors.pixelFormat, width: gpu.side, height: gpu.side, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.renderTarget, .shaderRead]
        return try #require(gpu.device.makeTexture(descriptor: descriptor))
    }
}
