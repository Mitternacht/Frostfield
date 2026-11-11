import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MeshStencilTests {
    private let gpu: PostProcessFixture
    private let white: any MTLTexture

    init() throws {
        gpu = try PostProcessFixture()
        white = try gpu.color(width: 1, height: 1) { _, _ in SIMD4(repeating: 1) }
    }

    @Test("Graph stencil read and write masks distinguish depth fail, stencil fail and pass",
          arguments: [MeshDepthMode.forward, .reversed])
    func graphOperations(depthMode: MeshDepthMode) throws {
        let color = try gpu.color(width: 8, height: 4)
        let target = try pass(color: color, depthMode: depthMode, stencilClear: 0xA0)
        let whole = try MeshGeometry.plane(width: 2, height: 2)
        let half = try MeshGeometry.plane(width: 1, height: 2)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let red = SIMD4<Float>(1, 0, 0, 1)
        let green = SIMD4<Float>(0, 1, 0, 1)

        let occluder = try mesh(half, state: .init(colorWriteMask: []), depthMode: depthMode)
        let depthSplit = try mesh(whole, state: .init(depthWrite: false, colorWriteMask: [], stencil: .init(
            reference: 0x35, writeMask: 0x0F, depthFailure: .replace, pass: .incrementClamp
        )), depthMode: depthMode)
        let invert = try mesh(half, state: .init(depthTest: false, colorWriteMask: [], stencil: .init(
            compare: .never, writeMask: 0x03, stencilFailure: .invert
        )), depthMode: depthMode)
        let readLow = try mesh(whole, state: .init(depthTest: false, stencil: .init(
            reference: 0x15, compare: .equal, readMask: 0x0F
        )), depthMode: depthMode)
        let readAll = try mesh(whole, state: .init(depthTest: false, stencil: .init(
            reference: 0xA2, compare: .equal
        )), depthMode: depthMode)
        let graph = SceneGraph(roots: [
            .init(transform: .init(translation: SIMD3(-0.5, 0, -1)), mesh: occluder, renderOrder: 1),
            .init(transform: .init(translation: SIMD3(0, 0, -2)), mesh: depthSplit, renderOrder: 2),
            .init(transform: .init(translation: SIMD3(0.5, 0, -3)), mesh: invert, renderOrder: 3),
            .init(transform: .init(translation: SIMD3(0, 0, -3)), mesh: readLow, materialParameters: red, renderOrder: 4),
            .init(transform: .init(translation: SIMD3(0, 0, -3)), mesh: readAll, materialParameters: green, renderOrder: 5),
        ], camera: camera)
        let archive = try SceneArchive(capturing: graph)
        let restored = try archive.load(device: gpu.device).graph
        #expect(restored.roots[0].mesh?.usesStencilAttachment == true)
        #expect(restored.roots[0].mesh?.renderState.stencil == nil)
        #expect(restored.roots[1].mesh?.renderState.stencil == depthSplit.renderState.stencil)

        let clip = orthographicClip(depthMode: depthMode)
        let expected = (0..<32).map { $0 % 8 < 4 ? red : green }
        for scene in [graph, restored] {
            try gpu.run { buffer in
                let stats = try scene.encode(
                    pass: target, commandBuffer: buffer, clipFromWorld: clip,
                    cameraPosition: .zero, directionToCamera: SIMD3(0, 0, 1), depthMode: depthMode
                )
                #expect(stats.submittedTriangleCount == 10)
            }
            #expect(try gpu.read(color) == expected)
        }
    }

    @Test("Separate standalone draws store and load the same stencil texture")
    func standaloneLoad() throws {
        let color = try gpu.color(width: 4, height: 4)
        let target = try pass(color: color)
        let plane = try MeshGeometry.plane(width: 3, height: 3)
        let writer = try mesh(plane, state: .init(depthTest: false, colorWriteMask: [], stencil: .init(reference: 42, pass: .replace)))
        let reader = try mesh(plane, state: .init(depthTest: false, stencil: .init(reference: 41, compare: .less)))
        let output = SIMD4<Float>(0.5, 0.25, 0, 1)
        let scene = MeshScene(
            camera: try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10),
            transform: .init(translation: SIMD3(0, 0, -1)), materialParameters: output
        )

        try gpu.run { buffer in
            try writer.encode(scene, pass: target, commandBuffer: buffer)
            let load = try #require(target.copy() as? MTLRenderPassDescriptor)
            load.stencilAttachment.loadAction = .load
            try reader.encode(scene, pass: load, commandBuffer: buffer)
        }
        #expect(try gpu.read(color) == Array(repeating: output, count: 16))
    }

    @Test("Stencil silhouettes follow morph-before-skin and separate rigid instances, including after archive load")
    func posedInstances() throws {
        let color = try gpu.color(width: 16, height: 4)
        let target = try pass(color: color)
        let geometry = try MeshGeometry.plane(width: 0.125, height: 2)
        let influence = VertexSkinInfluence(joints: .zero, weights: SIMD4(1, 0, 0, 0))
        let skin = try MeshSkin(jointCount: 1, influences: Array(repeating: influence, count: geometry.vertices.count))
        let pose = try JointPalette(matrices: [
            SceneTransform(translation: SIMD3(0.125, 0, 0), scale: SIMD3(2, 1, 1)).matrix,
        ])
        let offset = Array(repeating: SIMD3<Float>(0.125, 0, 0), count: geometry.vertices.count)
        let morph = try MeshMorphTargets(vertexCount: geometry.vertices.count, targets: [.init(positionOffsets: offset)])
        let instances = try InstanceSet(device: gpu.device, transforms: [
            SceneTransform(translation: SIMD3(-1, 0, 0)).matrix,
            SceneTransform(translation: SIMD3(0.25, 0, 0)).matrix,
        ])
        let writer = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .textured(white),
            skin: skin, morphTargets: morph, supportsInstancing: true,
            renderState: .init(depthTest: false, colorWriteMask: [], stencil: .init(reference: 7, pass: .replace))
        )
        let reader = try mesh(.plane(width: 2, height: 2), state: .init(
            depthTest: false, stencil: .init(reference: 7, compare: .equal)
        ))
        let transform = SceneTransform(translation: SIMD3(0, 0, -2))
        let output = SIMD4<Float>(1, 0.5, 0.25, 1)
        let graph = SceneGraph(roots: [
            .init(transform: transform, mesh: writer, jointPalette: pose, instances: instances, morphWeights: [1], renderOrder: 1),
            .init(transform: transform, mesh: reader, materialParameters: output, renderOrder: 2),
        ], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
        let archive = try SceneArchive(capturing: graph)
        let restored = try archive.load(device: gpu.device).graph
        let expected = (0..<64).map { [2, 3, 12, 13].contains($0 % 16) ? output : .zero }

        for scene in [graph, restored] {
            try gpu.run { buffer in
                let stats = try scene.encode(
                    pass: target, commandBuffer: buffer, clipFromWorld: orthographicClip(),
                    cameraPosition: .zero, directionToCamera: SIMD3(0, 0, 1)
                )
                #expect(stats.drawnInstanceCount == 3)
                #expect(stats.submittedTriangleCount == 6)
            }
            let pixels = try gpu.read(color)
            #expect(pixels == expected, "first row: \(pixels.prefix(16))")
        }

        let soloColor = try gpu.color(width: 8, height: 8)
        let soloTarget = try pass(color: soloColor)
        let soloCamera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
        let soloTransform = SceneTransform(translation: SIMD3(0, 0, -1))
        try gpu.run { buffer in
            try writer.encode(
                MeshScene(camera: soloCamera, transform: soloTransform, jointPalette: pose, morphWeights: [1]),
                pass: soloTarget, commandBuffer: buffer
            )
            let load = try #require(soloTarget.copy() as? MTLRenderPassDescriptor)
            load.stencilAttachment.loadAction = .load
            try reader.encode(
                MeshScene(camera: soloCamera, transform: soloTransform, materialParameters: output),
                pass: load, commandBuffer: buffer
            )
        }
        #expect(try gpu.read(soloColor) == (0..<64).map { $0 % 8 == 5 ? output : .zero })
    }

    @Test("An in-flight stencil draw keeps its indexed geometry when the live revision moves and grows")
    func dynamicStencilRevisions() throws {
        let width = 16, height = 4
        let strip = try MeshGeometry.plane(width: 0.25, height: 2)
        let oldGeometry = try strip.transformed(by: SceneTransform(translation: SIMD3(-0.625, 0, 0)).matrix)
        let newGeometry = try MeshGeometry.merged([Float(0.125), 0.625].map { center in
            .init(strip, transform: SceneTransform(translation: SIMD3(center, 0, 0)).matrix)
        }).geometry
        let writer = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, dynamicGeometry: oldGeometry,
            capacity: .init(vertices: newGeometry.vertices.count, triangles: newGeometry.indices.count / 3),
            material: .textured(white),
            renderState: .init(depthTest: false, colorWriteMask: [], stencil: .init(reference: 7, pass: .replace))
        )
        let reader = try mesh(.plane(width: 2, height: 2), state: .init(
            depthTest: false, stencil: .init(reference: 7, compare: .equal)
        ))
        let output = SIMD4<Float>(0, 1, 0.5, 1)
        let transform = SceneTransform(translation: SIMD3(0, 0, -2))
        let graph = SceneGraph(roots: [
            .init(transform: transform, mesh: writer, renderOrder: 1),
            .init(transform: transform, mesh: reader, materialParameters: output, renderOrder: 2),
        ], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))

        let oldColor = try gpu.color(width: width, height: height)
        let newColor = try gpu.color(width: width, height: height)
        let gate = try #require(gpu.device.makeSharedEvent())
        let old = try #require(gpu.queue.makeCommandBuffer())
        defer { gate.signaledValue = 1 }
        old.encodeWaitForEvent(gate, value: 1)
        let oldStats = try graph.encode(
            pass: pass(color: oldColor), commandBuffer: old, clipFromWorld: orthographicClip(),
            cameraPosition: .zero, directionToCamera: SIMD3(0, 0, 1)
        )
        old.commit()

        try writer.updateGeometry(newGeometry)
        #expect(writer.geometryRevision == 1 && writer.bounds == newGeometry.bounds)
        let new = try #require(gpu.queue.makeCommandBuffer())
        let newStats = try graph.encode(
            pass: pass(color: newColor), commandBuffer: new, clipFromWorld: orthographicClip(),
            cameraPosition: .zero, directionToCamera: SIMD3(0, 0, 1)
        )
        new.commit()
        gate.signaledValue = 1
        for command in [old, new] {
            command.waitUntilCompleted()
            try #require(command.status == .completed)
        }

        #expect(oldStats.submittedTriangleCount == 4)
        #expect(newStats.submittedTriangleCount == 6)
        let oldExpected = (0..<(width * height)).map { [2, 3].contains($0 % width) ? output : .zero }
        let newExpected = (0..<(width * height)).map { [8, 9, 12, 13].contains($0 % width) ? output : .zero }
        let oldPixels = try gpu.read(oldColor)
        let newPixels = try gpu.read(newColor)
        #expect(oldPixels == oldExpected, "old first row: \(oldPixels.prefix(width))")
        #expect(newPixels == newExpected, "new first row: \(newPixels.prefix(width))")
    }

    @Test("Curve flow keeps its selected indexed copy while the stencil silhouette moves",
          arguments: [CurveFlow.Execution.cpu, .metal])
    func movingCurveFlow(execution: CurveFlow.Execution) throws {
        let strip = try MeshGeometry.plane(width: 0.25, height: 2)
        let curve = try CurveDeformer(curve: Curve3D(position: { SIMD3(-1 + 2 * $0, 0, 0) }), subdivisions: 8)
        let flow = try CurveFlow(device: gpu.device, mesh: strip, placements: [
            .init(curve: curve, offset: 0.375), .init(curve: curve, offset: 1.625),
        ], execution: execution)
        let writer = try MeshRenderer(
            curveFlow: flow, colorPixelFormat: .rgba16Float, material: .textured(white),
            renderState: .init(depthTest: false, colorWriteMask: [], stencil: .init(reference: 9, pass: .replace))
        )
        writer.drawRange = try .init(start: 6, count: 6)
        let output = SIMD4<Float>(0.25, 1, 0.5, 1)
        let graph = try mixedStencilGraph(moving: [
            .init(transform: .init(translation: SIMD3(0, 0, -2)), mesh: writer, renderOrder: 2),
        ], output: output)
        let original = try gpu.color(width: 16, height: 4)
        let changed = try gpu.color(width: 16, height: 4)
        let originalPass = try pass(color: original), changedPass = try pass(color: changed)
        let old = try #require(gpu.queue.makeCommandBufferWithUnretainedReferences())
        let new = try #require(gpu.queue.makeCommandBuffer())
        #expect(try encodeMixedStencil(graph, pass: originalPass, commandBuffer: old) == 6)
        try flow.setOffset(1.125, at: 1)
        #expect(try encodeMixedStencil(graph, pass: changedPass, commandBuffer: new) == 6)
        // Submit in reverse order. Each command must bind its own flow revision.
        try withExtendedLifetime((originalPass, changedPass)) {
            try complete([new, old])
        }
        try expectStencilColumns(original, visible: [2, 3, 12, 13], color: output)
        try expectStencilColumns(changed, visible: [2, 3, 8, 9], color: output)
        try expectStandaloneStencil(writer, column: 4, color: output)
    }

    @Test("Typed position offsets use command-local vertices and the original draw range")
    func graphPositionSnapshots() throws {
        let strip = try MeshGeometry.plane(width: 0.25, height: 2)
        let geometry = try MeshGeometry.merged([Float(-0.625), 0.125].map { center in
            .init(strip, transform: SceneTransform(translation: SIMD3(center, 0, 0)).matrix)
        }).geometry
        let shader = try SurfaceShaderGraph { graph in
            let move = graph.uniform("Move", Float(0))
            let zero = graph.constant(Float(0))
            return .init(positionOffset: graph.vector(move, zero, zero))
        }
        let material = try PhysicallyBasedMaterial(surfaceGraph: shader, shadingModel: .unlit)
        let uniforms = try #require(material.surfaceUniforms)
        let writer = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(material),
            renderState: .init(depthTest: false, colorWriteMask: [], stencil: .init(reference: 9, pass: .replace))
        )
        writer.drawRange = try .init(start: 6, count: 6)
        let output = SIMD4<Float>(1, 0.25, 0.5, 1)
        let graph = try mixedStencilGraph(moving: [
            .init(transform: .init(translation: SIMD3(0, 0, -2)), mesh: writer, renderOrder: 2),
            .init(transform: .init(translation: SIMD3(-0.25, 0, -2)), mesh: writer, renderOrder: 3),
        ], output: output)
        let images = try (0..<3).map { _ in try gpu.color(width: 16, height: 4) }
        let targets = try images.map { try pass(color: $0) }
        let old = try #require(gpu.queue.makeCommandBufferWithUnretainedReferences())
        let next = try #require(gpu.queue.makeCommandBuffer())
        let repeated = try #require(gpu.queue.makeCommandBuffer())
        #expect(try encodeMixedStencil(graph, pass: targets[0], commandBuffer: old) == 8)
        try uniforms.set("Move", to: 0.5)
        #expect(try encodeMixedStencil(graph, pass: targets[1], commandBuffer: next) == 8)
        #expect(try encodeMixedStencil(graph, pass: targets[2], commandBuffer: repeated) == 8)
        #expect(writer.deformationEncodeCount == 3, "The two nodes share one deformation within each pass.")
        // A repeated uniform revision must be valid even when its first command
        // has not been submitted. The old command still uses its own uniform values.
        try withExtendedLifetime(targets) {
            try complete([repeated, old, next])
        }
        try expectStencilColumns(images[0], visible: [2, 3, 6, 7, 8, 9], color: output)
        for image in images.dropFirst() {
            try expectStencilColumns(image, visible: [2, 3, 10, 11, 12, 13], color: output)
        }
        try expectStandaloneStencil(writer, column: 6, color: output)
        #expect(throws: MeshRenderer.Failure.stencilPass) {
            try writer.prepareSurfaceAttributes()
        }
    }

    @Test("Multisample stencil writes gate each sample before the HDR color resolve",
          arguments: [MTLStoreAction.multisampleResolve, .storeAndMultisampleResolve])
    func multisampledStencil(storeAction: MTLStoreAction) throws {
        let width = 8, height = 4, sampleCount = 4
        let color = try multisampleColor(width: width, height: height, sampleCount: sampleCount)
        let resolved = try gpu.color(width: width, height: height)
        let target = try pass(color: color)
        target.colorAttachments[0].storeAction = storeAction
        target.colorAttachments[0].resolveTexture = resolved
        let writer = try mesh(.plane(width: 1.125, height: 2), state: .init(
            depthTest: false, colorWriteMask: [], stencil: .init(reference: 19, pass: .replace)
        ), sampleCount: sampleCount)
        let reader = try mesh(.plane(width: 2, height: 2), state: .init(
            depthTest: false, stencil: .init(reference: 19, compare: .equal)
        ), sampleCount: sampleCount)
        let output = SIMD4<Float>(2, 1, 0.5, 1)
        let graph = SceneGraph(roots: [
            .init(transform: .init(translation: SIMD3(-0.4375, 0, -2)), mesh: writer, renderOrder: 1),
            .init(transform: .init(translation: SIMD3(0, 0, -2)), mesh: reader, materialParameters: output, renderOrder: 2),
        ], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
        try gpu.run { buffer in
            let statistics = try graph.encode(
                pass: target, commandBuffer: buffer, clipFromWorld: orthographicClip(),
                cameraPosition: .zero, directionToCamera: SIMD3(0, 0, 1)
            )
            #expect(statistics.submittedTriangleCount == 4)
        }

        // The stencil silhouette ends at the center of column four. The second
        // mesh covers the whole frame, so only stencil can create this partial resolve.
        let positions = gpu.device.getDefaultSamplePositions(sampleCount: sampleCount)
        let covered = positions.count { $0.x < 0.5 }
        try #require(covered > 0 && covered < sampleCount)
        let boundary = output * (Float(covered) / Float(sampleCount))
        let expected = (0..<(width * height)).map { index in
            let column = index % width
            return column < 4 ? output : column == 4 ? boundary : .zero
        }
        let pixels = try gpu.read(resolved)
        #expect(pixels == expected, "first row: \(pixels.prefix(width))")
    }

    @Test("Invalid multisample stencil resolves leave the resolved color untouched")
    func multisampledValidation() throws {
        let sampleCount = 4
        let color = try multisampleColor(width: 8, height: 4, sampleCount: sampleCount)
        let target = try pass(color: color)
        let sentinel = SIMD4<Float>(0.25, 0.5, 0.75, 1)
        let resolved = try gpu.color(width: 8, height: 4) { _, _ in sentinel }
        let wrongSize = try gpu.color(width: 4, height: 4)
        let singleSample = try pass(color: resolved)
        let renderer = try mesh(.plane(width: 2, height: 2), state: .init(stencil: .init()), sampleCount: sampleCount)
        let scene = MeshScene(
            camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10),
            transform: .init(translation: SIMD3(0, 0, -2))
        )
        try gpu.run { buffer in
            target.colorAttachments[0].storeAction = .multisampleResolve
            for invalid in [nil, wrongSize, color] {
                target.colorAttachments[0].resolveTexture = invalid
                #expect(throws: MeshRenderer.Failure.stencilAttachments) {
                    try renderer.encode(scene, pass: target, commandBuffer: buffer)
                }
            }
            target.colorAttachments[0].resolveTexture = resolved
            target.stencilAttachment.resolveTexture = singleSample.depthAttachment.texture
            #expect(throws: MeshRenderer.Failure.stencilAttachments) {
                try renderer.encode(scene, pass: target, commandBuffer: buffer)
            }
            #expect(throws: MeshRenderer.Failure.sampleCount) {
                try renderer.encode(scene, pass: singleSample, commandBuffer: buffer)
            }
        }
        #expect(try gpu.read(resolved) == Array(repeating: sentinel, count: 32))
    }

    @Test("Unsupported stencil attachments and graph draws are rejected before changing color")
    func validation() throws {
        let sentinel = SIMD4<Float>(0.25, 0.5, 0.75, 1)
        let color = try gpu.color(width: 4, height: 4) { _, _ in sentinel }
        let target = try pass(color: color)
        let plane = try MeshGeometry.plane(width: 3, height: 3)
        let stencil = try mesh(plane, state: .init(stencil: .init(reference: 1, pass: .replace)))
        let ordinary = try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float, geometry: plane)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let transform = SceneTransform(translation: SIMD3(0, 0, -2))
        let graph = SceneGraph(roots: [.init(transform: transform, mesh: stencil), .init(transform: transform, mesh: ordinary)],
                               camera: camera)
        let stencilOnly = SceneGraph(roots: [.init(transform: transform, mesh: stencil)], camera: camera)
        let scene = MeshScene(camera: camera, transform: transform)
        #expect(throws: MeshRenderer.Failure.stencilPass) {
            try BatchedMesh(
                device: gpu.device, colorPixelFormat: .rgba16Float, geometries: [plane], capacity: 1,
                renderState: .init(stencil: .init())
            )
        }

        try gpu.run { buffer in
            #expect(throws: MeshRenderer.Failure.stencilPass) {
                try graph.encode(pass: target, commandBuffer: buffer)
            }
            #expect(throws: MeshRenderer.Failure.stencilPass) {
                try ordinary.encode(scene, pass: target, commandBuffer: buffer)
            }
            let motionTexture = try gpu.color(width: 4, height: 4)
            let motion = try RasterMotionVectors(queue: gpu.queue)
            #expect(throws: MeshRenderer.Failure.stencilPass) {
                try stencilOnly.encode(pass: target, commandBuffer: buffer, motionVectors: motion.recording(to: motionTexture))
            }
            let mismatched = try #require(target.copy() as? MTLRenderPassDescriptor)
            mismatched.stencilAttachment.texture = nil
            #expect(throws: MeshRenderer.Failure.stencilAttachments) {
                try stencil.encode(scene, pass: mismatched, commandBuffer: buffer)
            }
            #expect(throws: MeshRenderer.Failure.stencilPass) {
                try stencil.prepareSurfaceAttributes()
            }
        }
        #expect(try gpu.read(color) == Array(repeating: sentinel, count: 16))
    }

    @Test("A rejected stencil selection leaves the previous outline mask usable")
    func selectionRejection() throws {
        let source = try gpu.color(width: 8, height: 8) { _, _ in SIMD4(0, 0, 0, 1) }
        let destination = try gpu.color(width: 8, height: 8)
        let depth = try gpu.depth(width: 8, height: 8) { _, _ in 1 }
        let plane = try MeshGeometry.plane(width: 1, height: 1)
        let ordinary = try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float, geometry: plane)
        let stencil = try mesh(plane, state: .init(stencil: .init()))
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let transform = SceneTransform(translation: SIMD3(0, 0, -2))
        let selected = SceneNode(transform: transform, mesh: ordinary)
        let unsupported = SceneNode(transform: transform, mesh: stencil)
        let graph = SceneGraph(roots: [selected], camera: camera)
        let stencilGraph = SceneGraph(roots: [unsupported], camera: camera)
        let outline = try SelectionOutline(device: gpu.device)

        try gpu.run { buffer in
            let count = try outline.encodeSelection(
                graph: graph, selection: [selected.id], width: 8, height: 8, commandBuffer: buffer
            )
            #expect(count == 1)
            try outline.encode(source: source, depth: depth, destination: destination, commandBuffer: buffer)
        }
        let previous = try gpu.read(destination)
        #expect(previous != Array(repeating: SIMD4(0, 0, 0, 1), count: 64))
        try gpu.run { buffer in
            #expect(throws: MeshRenderer.Failure.stencilPass) {
                try outline.encodeSelection(
                    graph: stencilGraph, selection: [unsupported.id], width: 4, height: 4,
                    depthMode: .reversed, commandBuffer: buffer
                )
            }
            try outline.encode(source: source, depth: depth, destination: destination, commandBuffer: buffer)
        }
        #expect(try gpu.read(destination) == previous)
    }

    @Test("Archives preserve stencil format version and reject malformed state")
    func archiveValidation() throws {
        let mesh = try mesh(.plane(width: 1, height: 1), state: .init(stencil: .init()))
        let graph = SceneGraph(
            roots: [.init(mesh: mesh)], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        )
        let archive = try SceneArchive(capturing: graph)
        var json = try #require(try JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(json["version"] as? Int == 21)
        json["version"] = 20
        #expect(throws: SceneArchive.Failure.decoding("mesh stencil state requires archive version 21")) {
            try SceneArchive(manifest: JSONSerialization.data(withJSONObject: json), binary: archive.binary)
        }
        json["version"] = 21
        var meshes = try #require(json["meshes"] as? [[String: Any]])
        var state = try #require(meshes[0]["renderState"] as? [String: Any])
        var stencil = try #require(state["stencil"] as? [String: Any])
        stencil["pass"] = 999
        state["stencil"] = stencil
        meshes[0]["renderState"] = state
        json["meshes"] = meshes
        let malformed = try SceneArchive(manifest: JSONSerialization.data(withJSONObject: json), binary: archive.binary)
        #expect(throws: SceneArchive.Failure.invalidReference("mesh stencil comparison or operation")) {
            try malformed.load(device: gpu.device)
        }
    }

    private func mixedStencilGraph(moving: [SceneNode], output: SIMD4<Float>) throws -> SceneGraph {
        let fixed = try mesh(.plane(width: 0.25, height: 2), state: .init(
            depthTest: false, colorWriteMask: [], stencil: .init(reference: 9, pass: .replace)
        ))
        let reader = try mesh(.plane(width: 2, height: 2), state: .init(
            depthTest: false, stencil: .init(reference: 9, compare: .equal)
        ))
        let fixedNode = SceneNode(transform: .init(translation: SIMD3(-0.625, 0, -2)), mesh: fixed, renderOrder: 1)
        let reveal = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -2)), mesh: reader, materialParameters: output, renderOrder: 4
        )
        return SceneGraph(
            roots: [fixedNode] + moving + [reveal], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        )
    }

    private func encodeMixedStencil(
        _ graph: SceneGraph, pass: MTLRenderPassDescriptor, commandBuffer: any MTLCommandBuffer
    ) throws -> Int {
        try graph.encode(
            pass: pass, commandBuffer: commandBuffer, clipFromWorld: orthographicClip(),
            cameraPosition: .zero, directionToCamera: SIMD3(0, 0, 1)
        ).submittedTriangleCount
    }

    private func expectStencilColumns(_ image: any MTLTexture, visible: Set<Int>, color: SIMD4<Float>) throws {
        let expected = (0..<(image.width * image.height)).map { visible.contains($0 % image.width) ? color : .zero }
        let pixels = try gpu.read(image)
        #expect(pixels == expected, "first row: \(pixels.prefix(image.width))")
    }

    private func expectStandaloneStencil(_ writer: MeshRenderer, column: Int, color: SIMD4<Float>) throws {
        let image = try gpu.color(width: 8, height: 8)
        let target = try pass(color: image)
        let reveal = try mesh(.plane(width: 2, height: 2), state: .init(
            depthTest: false, stencil: .init(reference: 9, compare: .equal)
        ))
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
        let scene = MeshScene(
            camera: camera, transform: .init(translation: SIMD3(0, 0, -1)), materialParameters: color
        )
        try gpu.run { buffer in
            try writer.encode(scene, pass: target, commandBuffer: buffer)
            let load = try #require(target.copy() as? MTLRenderPassDescriptor)
            load.stencilAttachment.loadAction = .load
            try reveal.encode(scene, pass: load, commandBuffer: buffer)
        }
        try expectStencilColumns(image, visible: [column], color: color)
    }

    private func complete(_ commands: [any MTLCommandBuffer]) throws {
        for command in commands { command.commit() }
        for command in commands {
            command.waitUntilCompleted()
            try #require(command.status == .completed, "\(String(describing: command.error))")
        }
    }

    private func orthographicClip(depthMode: MeshDepthMode = .forward) -> simd_float4x4 {
        let near: Float = 0.1, far: Float = 10
        var clip = matrix_identity_float4x4
        let reversed = depthMode == .reversed
        clip.columns.2.z = (reversed ? 1 : -1) / (far - near)
        clip.columns.3.z = (reversed ? far : -near) / (far - near)
        return clip
    }

    private func mesh(
        _ geometry: MeshGeometry, state: MeshRenderState, depthMode: MeshDepthMode = .forward, sampleCount: Int = 1
    ) throws -> MeshRenderer {
        try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .textured(white), depthMode: depthMode,
            renderState: state, sampleCount: sampleCount, usesStencilAttachment: state.stencil == nil
        )
    }

    private func multisampleColor(width: Int, height: Int, sampleCount: Int) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.textureType = .type2DMultisample
        descriptor.sampleCount = sampleCount
        descriptor.storageMode = .private
        descriptor.usage = .renderTarget
        return try #require(gpu.device.makeTexture(descriptor: descriptor))
    }

    private func pass(
        color: any MTLTexture, depthMode: MeshDepthMode = .forward, stencilClear: UInt32 = 0
    ) throws -> MTLRenderPassDescriptor {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: MeshRenderer.stencilDepthPixelFormat, width: color.width, height: color.height, mipmapped: false
        )
        descriptor.textureType = color.textureType
        descriptor.sampleCount = color.sampleCount
        descriptor.storageMode = .private
        descriptor.usage = .renderTarget
        let depth = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = depthMode == .reversed ? 0 : 1
        pass.depthAttachment.storeAction = .store
        pass.stencilAttachment.texture = depth
        pass.stencilAttachment.loadAction = .clear
        pass.stencilAttachment.clearStencil = stencilClear
        pass.stencilAttachment.storeAction = .store
        return pass
    }
}
