import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathCameraMotionTests {
    private let fixture: PathTestFixture
    private let width = 17
    private let height = 13

    init() throws {
        fixture = try PathTestFixture()
    }

    @Test("Static hits project into the previous camera in top-left input pixels", arguments: PathAccelerationTests.modes)
    func perspectivePans(acceleration: PathTracingScene.Options.Acceleration) throws {
        let (_, _, tracer) = try makeTracer(acceleration: acceleration)
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let first = try frame(tracer, accumulation: accumulation)
        #expect(first.motion.allSatisfy { $0 == .zero })

        let moved = SceneTransform(translation: SIMD3(0.4, 0.3, -0.5))
        let second = try frame(tracer, accumulation: accumulation, camera: moved, previous: first.history)
        let index = (height / 2) * width + width / 2 + 1
        let focal = Float(height) / (2 * tan(fixture.camera.verticalFieldOfView / 2))
        let expected = SIMD2<Float>(focal * 0.4 / 3 + 2.5 / 3 - 1, -focal * 0.3 / 3)
        #expect(abs(second.depth[index] - 2.5) < 1e-5)
        #expect(simd_length(second.motion[index] - expected) < 2e-5)
        #expect(second.depth[0].isInfinite && simd_length(second.motion[0]) < 2e-5)

        let back = try frame(tracer, accumulation: accumulation, previous: second.history)
        let inverse = SIMD2<Float>((3 - focal * 0.4) / 2.5 - 1, focal * 0.3 / 2.5)
        #expect(abs(back.depth[index] - 3) < 1e-5)
        #expect(simd_length(back.motion[index] - inverse) < 2e-5)
    }

    @Test("Perspective sky motion follows rotation, and rays behind the previous camera point outside history")
    func skyReprojection() throws {
        let (_, _, tracer) = try makeTracer()
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let first = try frame(tracer, accumulation: accumulation)
        let angle = Float.pi / 12
        let turned = try frame(
            tracer, accumulation: accumulation, camera: .init(rotation: simd_quatf(angle: angle, axis: SIMD3(0, 1, 0))),
            previous: first.history
        )
        let center = (height / 2) * width + width / 2
        // The plane's left edge is past the center ray after the pan.
        #expect(turned.depth[center].isInfinite)
        let focal = Float(height) / (2 * tan(fixture.camera.verticalFieldOfView / 2))
        #expect(simd_length(turned.motion[center] - SIMD2(-tan(angle) * focal, 0)) < 2e-5)

        let opposite = try frame(
            tracer, accumulation: accumulation, camera: .init(rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))),
            previous: turned.history
        )
        #expect(opposite.depth[center].isInfinite)
        #expect(opposite.motion[center] == SIMD2(Float(width * 2), Float(height * 2)))
    }

    @Test("Motion honors prior and current lens shift and perspective zoom")
    func zoomAndShift() throws {
        let (_, _, tracer) = try makeTracer()
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let oldFOV = Float.pi / 3
        let newFOV = Float.pi / 4
        let oldShift = SIMD2<Float>(-0.1, 0.2)
        let newShift = SIMD2<Float>(0.15, -0.05)
        let first = try frame(
            tracer, accumulation: accumulation,
            model: .init(projection: .perspective(verticalFieldOfView: oldFOV, shift: oldShift))
        )
        let jitter = SIMD2<Float>(0.2, 0.15)
        let next = try frame(
            tracer, accumulation: accumulation,
            model: .init(projection: .perspective(verticalFieldOfView: newFOV, shift: newShift)),
            previous: first.history, jitter: jitter
        )
        let pixelsFromNDC = SIMD2(Float(width) / 2, -Float(height) / 2)
        let sampleNDC = jitter / pixelsFromNDC
        let scale = tan(newFOV / 2) / tan(oldFOV / 2)
        let expected = ((sampleNDC + newShift) * scale - oldShift - sampleNDC) * pixelsFromNDC
        let center = (height / 2) * width + width / 2
        #expect(next.depth[center].isFinite)
        #expect(simd_length(next.motion[center] - expected) < 2e-5)
    }

    @Test("Frame jitter aligns beauty and guides without generating motion on its own")
    func jitterAndOrthographic() throws {
        let (_, _, tracer) = try makeTracer(planeWidth: 0.2)
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let center = (height / 2) * width + width / 2
        let first = try frame(tracer, accumulation: accumulation)
        #expect(first.depth[center].isFinite)
        #expect(first.beauty[center].x > 0.75)
        let jittered = try frame(
            tracer, accumulation: accumulation, previous: first.history, jitter: SIMD2(0.48, -0.4)
        )
        #expect(jittered.depth[center].isInfinite)
        #expect(jittered.beauty[center].x == 0)
        #expect(jittered.motion.allSatisfy { simd_length($0) < 2e-5 })

        let camera = PathTracer.CameraModel(projection: .orthographic(height: 2))
        let ortho = try frame(tracer, accumulation: accumulation, model: camera, jitter: SIMD2(-0.2, 0.35))
        let moved = try frame(
            tracer, accumulation: accumulation, model: camera, camera: .init(translation: SIMD3(0.02, 0.1, 0)),
            previous: ortho.history, jitter: SIMD2(0.1, 0.25), format: .rg16Float
        )
        #expect(moved.depth[center].isFinite)
        #expect(simd_length(moved.motion[center] - SIMD2(0.13, -0.65)) < 3e-4)
        #expect(moved.motion[0] == .zero)
    }

    @Test("Motion validates completed history and invalidates replaced geometry")
    func validationAndReset() throws {
        let (scene, graph, tracer) = try makeTracer()
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let other = try tracer.makeAccumulation(width: width, height: height)
        let pending = try #require(fixture.queue.makeCommandBuffer())
        let destination = try fixture.output(width: width, height: height)
        let vectors = try texture(.rg32Float)
        let history = try tracer.encode(
            camera: fixture.camera, cameraTransform: .init(), destination: destination, accumulation: accumulation,
            commandBuffer: pending, aovs: .init(cameraMotion: .init(vectors: vectors))
        )
        #expect(throws: PathTracer.Failure.cameraMotionHistory) {
            try frame(tracer, accumulation: other, previous: history)
        }
        try fixture.finish(pending)

        let command = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: PathTracer.Failure.cameraMotion) {
            try tracer.encode(
                camera: fixture.camera, cameraTransform: .init(), destination: destination, accumulation: other,
                commandBuffer: command, samplesPerPixel: 2, aovs: .init(cameraMotion: .init(vectors: vectors))
            )
        }
        #expect(throws: PathTracer.Failure.cameraMotion) {
            try tracer.encode(
                camera: fixture.camera, cameraTransform: .init(), destination: destination, accumulation: other,
                commandBuffer: command, settings: .init(apertureRadius: 0.1),
                aovs: .init(cameraMotion: .init(vectors: vectors))
            )
        }
        #expect(throws: PathTracer.Failure.cameraMotion) {
            try tracer.encode(
                camera: fixture.camera, cameraTransform: .init(), destination: destination, accumulation: other,
                commandBuffer: command, aovs: .init(cameraMotion: .init(vectors: vectors, jitter: SIMD2(0.51, 0)))
            )
        }
        let tile = try PathTracer.Region(imageWidth: width + 1, imageHeight: height, x: 1, y: 0)
        #expect(throws: PathTracer.Failure.cameraMotion) {
            try tracer.encode(
                camera: fixture.camera, cameraTransform: .init(), destination: destination, accumulation: other,
                commandBuffer: command, region: tile, aovs: .init(cameraMotion: .init(vectors: vectors))
            )
        }
        #expect(throws: PathTracer.Failure.cameraMotion) {
            try frame(tracer, accumulation: other, model: .init(projection: .equirectangular))
        }
        #expect(throws: PathTracer.Failure.cameraMotionHistory) {
            try frame(
                tracer, accumulation: other, model: .init(projection: .orthographic(height: 2)), previous: history
            )
        }

        var changed = graph
        changed.roots[0].transform.translation.x += 0.2
        try scene.updateGeometry(for: [changed.roots[0].id], in: changed)
        #expect(throws: PathTracer.Failure.sceneChanged) {
            try frame(tracer, accumulation: other, previous: history)
        }
        let reset = try frame(tracer, accumulation: accumulation)
        #expect(reset.motion.allSatisfy { $0 == .zero })
        #expect(throws: PathTracer.Failure.cameraMotionHistory) {
            try frame(tracer, accumulation: other, previous: history)
        }
    }

    @Test("Rigid instance hits use their own previous transform after skipped refits", arguments: PathAccelerationTests.modes)
    func rigidInstanceMotion(acceleration mode: PathTracingScene.Options.Acceleration) throws {
        let material = try PhysicallyBasedMaterial(
            baseColorFactor: SIMD4(1, 0.4, 0.2, 1), alphaMode: .opaque
        ).withLiveFactors()
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.72, height: 1.2),
            material: .physicallyBased(material), supportsInstancing: true
        )
        func placement(_ x: Float, _ y: Float, angle: Float = 0) -> simd_float4x4 {
            SceneTransform(
                translation: SIMD3(x, y, 0), rotation: simd_quatf(angle: angle, axis: SIMD3(0, 0, 1))
            ).matrix
        }
        let original = [placement(-0.8, 0, angle: -0.2), placement(0.8, 0, angle: 0.15)]
        let placements = try InstanceSet(device: fixture.device, transforms: original, capacity: 3)
        let graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh, instances: placements)],
            camera: fixture.camera
        )
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: mode))
        let tracer = try PathTracer(queue: fixture.queue, scene: scene)
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let camera = PathTracer.CameraModel(projection: .orthographic(height: 2))
        let first = try frame(tracer, accumulation: accumulation, model: camera, jitter: SIMD2(-0.33, 0.3))
        try placements.setTransform(placement(-1.5, 0.3), at: 0)
        #expect(try scene.updateTransforms(graph))
        let moved = [placement(-0.6, -0.2, angle: 0.3), placement(0.67, 0.16, angle: -0.5)]
        for (index, transform) in moved.enumerated() { try placements.setTransform(transform, at: index) }
        #expect(try scene.updateTransforms(graph))
        let pan = SIMD3<Float>(0.1, -0.05, 0)
        let jitter = SIMD2<Float>(0.22, -0.31)
        let next = try frame(
            tracer, accumulation: accumulation, model: camera, camera: .init(translation: pan),
            previous: first.history, jitter: jitter
        )
        let extent = SIMD2(Float(width), Float(height))
        let pixelScale = SIMD2(Float(height) / 2, -Float(height) / 2)
        for (index, x) in [3, 13].enumerated() {
            let y = height / 2
            let pixel = SIMD2(Float(x), Float(y)) + 0.5 + jitter
            let view = (pixel - extent * 0.5) / pixelScale
            let current = SIMD4(view.x + pan.x, view.y + pan.y, 0, 1)
            let prior = original[index] * moved[index].inverse * current
            let expected = SIMD2(prior.x - view.x, prior.y - view.y) * pixelScale
            #expect(next.depth[y * width + x].isFinite)
            #expect(simd_length(next.motion[y * width + x] - expected) < 2e-5)
        }
        #expect(next.depth[0].isInfinite && next.motion[0] == .zero)

        let revision = scene.revision
        // Rejecting a structural edit leaves the active scene and history intact.
        try placements.replaceAll(transforms: [moved[0]])
        #expect(throws: PathTracingScene.Failure.structureChanged(graph.roots[0].id)) {
            try scene.updateTransforms(graph)
        }
        let unchanged = try frame(tracer, accumulation: accumulation, model: camera, previous: next.history)
        #expect(unchanged.depth[(height / 2) * width + 13].isFinite)

        // A same-count bulk replacement may reassign slots, even when all the
        // transforms happen to match. It invalidates motion without a refit.
        try placements.replaceAll(transforms: moved)
        #expect(try !scene.updateTransforms(graph))
        #expect(scene.revision == revision)
        #expect(throws: PathTracer.Failure.sceneChanged) {
            try frame(tracer, accumulation: accumulation, model: camera, previous: unchanged.history)
        }
        let reset = try frame(tracer, accumulation: accumulation, model: camera)
        #expect(reset.motion.allSatisfy { $0 == .zero })
        let resumed = try frame(tracer, accumulation: accumulation, model: camera, previous: reset.history, jitter: jitter)
        #expect(resumed.motion.allSatisfy { simd_length($0) < 2e-5 })

        let factors = try #require(material.liveFactors)
        try factors.update { $0.baseColorFactor.x = 0.5 }
        #expect(try scene.updateMaterials())
        #expect(throws: PathTracer.Failure.sceneChanged) {
            try frame(tracer, accumulation: accumulation, model: camera, previous: resumed.history)
        }
    }

    enum Pose: CaseIterable, Sendable { case cpuSkin, computeSkin, morph, shaderGraph }

    @Test("Posed hits reproject prior vertices across skipped rebuilds",
          arguments: PathAccelerationTests.modes, Pose.allCases)
    func deformationMotion(acceleration: PathTracingScene.Options.Acceleration, pose: Pose) throws {
        let geometry = try MeshGeometry.plane(width: 3, height: 1.6, columns: 6, rows: 5)
        let weights = geometry.vertices.map { $0.position.y / 1.6 + 0.5 }
        let usesSkin = pose == .cpuSkin || pose == .computeSkin
        let skin = usesSkin ? try MeshSkin(
            jointCount: 2, influences: weights.map {
                VertexSkinInfluence(joints: SIMD4(0, 1, 0, 0), weights: SIMD4(1 - $0, $0, 0, 0))
            }
        ) : nil
        let morphs = pose == .morph ? try MeshMorphTargets(
            vertexCount: weights.count, targets: [.init(positionOffsets: weights.map { SIMD3($0, 0, 0) })]
        ) : nil
        let surface: SurfaceShaderGraph? = pose == .shaderGraph ? try SurfaceShaderGraph { builder in
            let amount = builder.uniform("Pose", Float(0))
            let weight = builder.objectPosition.y * Float(1 / 1.6) + Float(0.5)
            let zero = builder.constant(Float(0))
            let emission = builder.uniform("Emission", SIMD3<Float>(0.04, 0.01, 0.02))
            return .init(emission: emission, positionOffset: builder.vector(amount * weight, zero, zero))
        } : nil
        let uniforms = try surface.map { try ShaderUniforms(graph: $0) }
        let material = try PhysicallyBasedMaterial(
            baseColorFactor: SIMD4(0.9, 0.2, 0.1, 1), alphaMode: .opaque,
            surfaceGraph: surface, surfaceUniforms: uniforms, shadingModel: .unlit
        )
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .physicallyBased(material), skin: skin, morphTargets: morphs
        )
        func apply(_ amount: Float, to node: inout SceneNode) throws {
            if skin != nil {
                node.jointPalette = try JointPalette(matrices: [
                    matrix_identity_float4x4, SceneTransform(translation: SIMD3(amount, 0, 0)).matrix,
                ])
            } else if let uniforms {
                try uniforms.set("Pose", to: amount)
            } else {
                node.morphWeights = [amount]
            }
        }
        let oldAmount: Float = -0.35
        let oldTranslation = SIMD3<Float>(-0.12, 0.09, -3)
        var node = SceneNode(transform: .init(translation: oldTranslation), mesh: mesh)
        try apply(oldAmount, to: &node)
        var graph = SceneGraph(roots: [node], camera: fixture.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        if pose == .computeSkin { scene.computeSkinning = try ComputeSkinning(device: fixture.device) }
        func softwareTriangleOrder() throws -> [UInt32]? {
            guard acceleration == .software else { return nil }
            let key: PathTracingScene.GeometryKey = pose == .shaderGraph ? .shared(ObjectIdentifier(mesh)) : .node(node.id)
            let level = try #require(scene.acceleration.bottomLevels[key])
            let primitives = try #require(level.primitives)
            return Array(UnsafeBufferPointer(
                start: primitives.contents().assumingMemoryBound(to: UInt32.self), count: level.triangleCount
            ))
        }
        let oldOrder = try softwareTriangleOrder()
        let tracer = try PathTracer(queue: fixture.queue, scene: scene)
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let model = PathTracer.CameraModel(projection: .orthographic(height: 2))
        let first = try frame(tracer, accumulation: accumulation, model: model, jitter: SIMD2(-0.2, 0.3))

        let progressive = try tracer.makeAccumulation(width: width, height: height)
        let destination = try fixture.output(width: width, height: height)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let mean = try tracer.encode(
            cameraModel: model, cameraTransform: .init(), destination: destination,
            accumulation: progressive, commandBuffer: command
        )
        try fixture.finish(command)
        let oldRevision = scene.revision

        // The intermediate pose is never rendered and must not replace the
        // vertices retained by the actual preceding frame.
        try apply(0.63, to: &graph.roots[0])
        try autoreleasepool { try scene.updateGeometry(for: [node.id], in: graph) }
        let newAmount: Float = 1.1
        let newTranslation = SIMD3<Float>(0.15, -0.14, -3)
        graph.roots[0].transform.translation = newTranslation
        try apply(newAmount, to: &graph.roots[0])
        try autoreleasepool { try scene.updateGeometry(for: [node.id], in: graph) }
        #expect(scene.revision == oldRevision + 2)
        #expect(scene.statistics.computeSkinnedBuildCount == (pose == .computeSkin ? 2 : 0))
        if let oldOrder { #expect(try softwareTriangleOrder() != oldOrder) }

        let nextCommand = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: PathTracer.Failure.sceneChanged) {
            try tracer.encode(
                cameraModel: model, cameraTransform: .init(), destination: destination,
                accumulation: progressive, previous: mean, commandBuffer: nextCommand
            )
        }

        let pan = SIMD3<Float>(0.1, -0.05, 0)
        let jitter = SIMD2<Float>(0.22, -0.31)
        let next = try frame(
            tracer, accumulation: accumulation, model: model, camera: .init(translation: pan),
            previous: first.history, jitter: jitter
        )
        let extent = SIMD2(Float(width), Float(height))
        let pixelsPerUnit = SIMD2(Float(height) / 2, -Float(height) / 2)
        for (x, y) in [(6, 3), (9, 6), (11, 9), (7, 9)] {
            let pixel = SIMD2(Float(x), Float(y)) + 0.5 + jitter
            let view = (pixel - extent * 0.5) / pixelsPerUnit
            let localY = view.y + pan.y - newTranslation.y
            let weight = localY / 1.6 + 0.5
            let priorDelta = SIMD2(
                pan.x + oldTranslation.x - newTranslation.x + (oldAmount - newAmount) * weight,
                pan.y + oldTranslation.y - newTranslation.y
            )
            let index = y * width + x
            #expect(next.depth[index].isFinite)
            #expect(simd_length(next.motion[index] - priorDelta * pixelsPerUnit) < 4e-5)
        }
        #expect(next.depth[0].isInfinite && next.motion[0] == .zero)

        if let uniforms {
            try uniforms.set("Emission", to: SIMD3<Float>(0.01, 0.04, 0.02))
            #expect(throws: PathTracer.Failure.surfaceUniformsChanged) {
                try frame(tracer, accumulation: accumulation, model: model, camera: .init(translation: pan),
                          previous: next.history, jitter: jitter)
            }
        }

        // Even identical indices cannot prove an external replacement kept
        // the renderer's vertex identities.
        try scene.updateGeometry(for: [], in: graph, replacements: [node.id: geometry])
        #expect(throws: PathTracer.Failure.sceneChanged) {
            try frame(tracer, accumulation: accumulation, model: model, previous: next.history)
        }
        let reset = try frame(tracer, accumulation: accumulation, model: model)
        #expect(reset.motion.allSatisfy { $0 == .zero })

        // The first pose after a replacement also needs a clean history.
        try scene.updateGeometry(for: [node.id], in: graph)
        #expect(throws: PathTracer.Failure.sceneChanged) {
            try frame(tracer, accumulation: accumulation, model: model, previous: reset.history)
        }
    }

    private func makeTracer(
        planeWidth: Float = 1.5, acceleration: PathTracingScene.Options.Acceleration = .software
    ) throws -> (PathTracingScene, SceneGraph, PathTracer) {
        let material = try PhysicallyBasedMaterial(
            baseColorFactor: SIMD4(1, 0.4, 0.2, 1), alphaMode: .opaque, shadingModel: .unlit
        )
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: planeWidth, height: 1.5),
            material: .physicallyBased(material)
        )
        let graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)], camera: fixture.camera
        )
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        return (scene, graph, try PathTracer(queue: fixture.queue, scene: scene))
    }

    private struct Frame {
        let history: PathTracer.History
        let motion: [SIMD2<Float>]
        let depth: [Float]
        let beauty: [SIMD4<Float>]
    }

    private func frame(
        _ tracer: PathTracer, accumulation: PathTracer.Accumulation, model: PathTracer.CameraModel? = nil,
        camera: SceneTransform = .init(), previous: PathTracer.History? = nil,
        jitter: SIMD2<Float> = .zero, format: MTLPixelFormat = .rg32Float
    ) throws -> Frame {
        let command = try #require(fixture.queue.makeCommandBuffer())
        let destination = try fixture.output(width: width, height: height)
        let vectors = try texture(format)
        let depth = try texture(.r32Float)
        let cameraModel = model ?? .init(
            projection: .perspective(verticalFieldOfView: fixture.camera.verticalFieldOfView),
            near: fixture.camera.near, far: fixture.camera.far
        )
        let history = try tracer.encode(
            cameraModel: cameraModel, cameraTransform: camera, destination: destination, accumulation: accumulation,
            commandBuffer: command, aovs: .init(depth: depth, cameraMotion: .init(vectors: vectors, previousFrame: previous, jitter: jitter))
        )
        try fixture.finish(command)
        let motion: [SIMD2<Float>] = format == .rg16Float
            ? read(vectors, zero: SIMD2<Float16>.zero).map(SIMD2<Float>.init)
            : read(vectors, zero: SIMD2<Float>.zero)
        return Frame(history: history, motion: motion, depth: read(depth, zero: Float(0)), beauty: try fixture.readback(destination))
    }

    private func texture(_ format: MTLPixelFormat) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.hazardTrackingMode = .tracked
        return try #require(fixture.device.makeTexture(descriptor: descriptor))
    }

    private func read<T>(_ texture: any MTLTexture, zero: T) -> [T] {
        var values = [T](repeating: zero, count: width * height)
        values.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                bytes.baseAddress!, bytesPerRow: width * MemoryLayout<T>.stride,
                from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0
            )
        }
        return values
    }
}
