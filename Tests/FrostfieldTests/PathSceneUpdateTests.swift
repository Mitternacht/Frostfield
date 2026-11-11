import Foundation
import Frostfield
import Metal
import Testing
import simd

struct PathSceneLayoutTests {
    @Test("Instance and dynamics records match the Metal layouts")
    func layout() {
        #expect(MemoryLayout<PathTracingScene.InstanceRecord>.stride == 160)
        #expect(MemoryLayout<PathTracingScene.InstanceRecord>.alignment == 16)
        #expect(MemoryLayout<PathTracingScene.InstanceRecord>.offset(of: \.worldToObject0) == 48)
        #expect(MemoryLayout<PathTracingScene.InstanceRecord>.offset(of: \.meta) == 96)
        #expect(MemoryLayout<PathTracingScene.InstanceRecord>.offset(of: \.surface) == 112)
        #expect(MemoryLayout<PathTracingScene.InstanceRecord>.offset(of: \.vertices) == 128)
        #expect(MemoryLayout<PathTracingScene.InstanceRecord>.offset(of: \.indices) == 136)
        #expect(MemoryLayout<PathTracingScene.InstanceRecord>.offset(of: \.nodes) == 144)
        #expect(MemoryLayout<PathTracingScene.InstanceRecord>.offset(of: \.primitives) == 152)
        #expect(MemoryLayout<PathTracer.Dynamics>.stride == 32)
        #expect(MemoryLayout<PathTracer.Dynamics>.alignment == 16)
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathSceneUpdateTests {
    private let fixture: PathTestFixture
    private let width = 48
    private let height = 32

    init() throws {
        fixture = try PathTestFixture()
    }

    @Test("Moving a node refits the top level, keeps every bottom level and matches a fresh preparation", arguments: PathAccelerationTests.modes)
    func movedInstance(acceleration: PathTracingScene.Options.Acceleration) throws {
        let glow = try fixture.pbr(.plane(width: 10, height: 8), color: .zero, emission: SIMD3(1, 0.5, 0.25))
        let blocker = try fixture.pbr(.cube(), color: .zero)
        var graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -5)), mesh: glow),
            SceneNode(transform: .init(translation: SIMD3(-1, 0, -3)), mesh: blocker),
        ], camera: fixture.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        let tracer = try PathTracer(queue: fixture.queue, scene: scene)
        let settings = PathTracer.Settings(maxBounces: 1, seed: 5)
        let left = pixel(SIMD3(-1, 0, -2.5))
        let right = pixel(SIMD3(1, 0, -2.5))
        let glowColor = SIMD4<Float>(1, 0.5, 0.25, 1)

        let before = try fixture.trace(tracer, width: width, height: height, samples: 4, settings: settings)
        #expect(before[left] == SIMD4(0, 0, 0, 1))
        #expect(before[right] == glowColor)
        let initial = scene.statistics
        #expect(initial.bottomLevelBuildCount == 2 && initial.topLevelBuildCount == 1 && initial.topLevelRefitCount == 0)
        #expect(initial.revision == 0 && scene.revision == 0)

        graph.roots[1].transform.translation = SIMD3(1, 0, -3)
        #expect(try scene.updateTransforms(graph))
        let moved = scene.statistics
        #expect(moved.bottomLevelBuildCount == 2)
        #expect(moved.topLevelBuildCount + moved.topLevelRefitCount == 2)
        #expect(moved.revision == 1)
        #expect(moved.bufferByteCount == initial.bufferByteCount)

        let after = try fixture.trace(tracer, width: width, height: height, samples: 4, settings: settings)
        #expect(after[left] == glowColor)
        #expect(after[right] == SIMD4(0, 0, 0, 1))
        let fresh = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        let reference = try fixture.trace(
            try PathTracer(queue: fixture.queue, scene: fresh), width: width, height: height, samples: 4, settings: settings
        )
        #expect(PathTestFixture.maximumDifference(after, reference) < 1e-3)
        #expect(after.map(\.w) == reference.map(\.w))

        // An unchanged graph keeps the revision, and so the history.
        #expect(try scene.updateTransforms(graph) == false)
        #expect(scene.revision == 1)
    }

    @Test("An empty instance set does not block path scene updates", arguments: PathAccelerationTests.modes)
    func emptyInstanceSet(acceleration: PathTracingScene.Options.Acceleration) throws {
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .cube(),
            material: .physicallyBased(try .init(alphaMode: .opaque)), supportsInstancing: true
        )
        let empty = try InstanceSet(device: fixture.device, transforms: [], capacity: 1)
        let visible = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)
        let group = SceneNode(mesh: mesh, instances: empty)
        var graph = SceneGraph(roots: [group, visible], camera: fixture.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        let prepared = scene.statistics
        #expect(prepared.instanceCount == 1)

        graph.roots[0].transform.translation.x = 1
        #expect(try scene.updateTransforms(graph) == false)
        #expect(scene.statistics == prepared)

        graph.roots[1].transform.translation.x = 1
        #expect(try scene.updateTransforms(graph))
        #expect(scene.revision == 1)
        #expect(scene.statistics.bottomLevelBuildCount == prepared.bottomLevelBuildCount)

        try empty.replaceAll(transforms: [matrix_identity_float4x4])
        let moved = scene.statistics
        #expect(throws: PathTracingScene.Failure.structureChanged(group.id)) {
            try scene.updateTransforms(graph)
        }
        #expect(scene.statistics == moved)
    }

    @Test("Shared, scaled and mirrored instances trace like the same geometry baked into world space", arguments: PathAccelerationTests.modes)
    func instancesMatchBakedGeometry(acceleration: PathTracingScene.Options.Acceleration) throws {
        let box = MeshGeometry.cube()
        let color = SIMD3<Float>(0.7, 0.4, 0.2)
        let transforms = [
            SceneTransform(
                translation: SIMD3(-1.1, -0.15, -3.4), rotation: simd_quatf(angle: 0.6, axis: simd_normalize(SIMD3(1, 1, 0)))
            ),
            SceneTransform(
                translation: SIMD3(0.15, 0.3, -3.9), rotation: simd_quatf(angle: -0.4, axis: SIMD3(0, 1, 0)),
                scale: SIMD3(1.4, 0.6, 0.8)
            ),
            // A negative scale mirrors the winding; back faces must stay culled.
            SceneTransform(
                translation: SIMD3(1.2, -0.3, -3.2), rotation: simd_quatf(angle: 0.5, axis: SIMD3(0, 1, 0)),
                scale: SIMD3(-0.7, 0.7, 0.7)
            ),
        ]
        let floorNode = SceneNode(
            transform: .init(translation: SIMD3(0, -1, -4), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))),
            mesh: try fixture.pbr(.plane(width: 8, height: 8), color: SIMD3(repeating: 0.6))
        )
        let shared = try fixture.pbr(box, color: color, roughness: 0.45)
        // A mirrored glass block also checks which side of each boundary a ray enters.
        let glassTransform = SceneTransform(
            translation: SIMD3(-0.25, -0.55, -2.6), rotation: simd_quatf(angle: 0.3, axis: SIMD3(0, 1, 0)),
            scale: SIMD3(-0.5, 0.5, 0.5)
        )
        func glass(_ geometry: MeshGeometry) throws -> MeshRenderer {
            try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry,
                material: .physicallyBased(try .init(
                    roughness: 0.05, alphaMode: .opaque,
                    transmission: .init(factor: 1, indexOfRefraction: 1.5, thickness: 0.5, attenuationDistance: 0.4,
                                        attenuationColor: SIMD3(0.3, 0.9, 0.6))
                ))
            )
        }
        let instanced = SceneGraph(
            roots: [floorNode] + transforms.map { SceneNode(transform: $0, mesh: shared) }
                + [SceneNode(transform: glassTransform, mesh: try glass(box))],
            camera: fixture.camera
        )
        let baked = SceneGraph(
            roots: [floorNode] + (try transforms.map {
                SceneNode(mesh: try fixture.pbr(try bake(box, $0.matrix), color: color, roughness: 0.45))
            }) + [SceneNode(mesh: try glass(try bake(box, glassTransform.matrix)))],
            camera: fixture.camera
        )
        let lighting = try PathTracer.Lighting(
            lights: [try PathTracer.AreaLight(
                center: SIMD3(0.5, 2, -2.5), aim: SIMD3(0, -1, -3.5), size: SIMD2(1.5, 1.5),
                radiance: SIMD3(repeating: 6), cameraVisible: false
            )],
            environment: SIMD3(0.05, 0.06, 0.08)
        )
        let settings = PathTracer.Settings(maxBounces: 6, seed: 11)
        let instancedScene = try PathTracingScene(queue: fixture.queue, graph: instanced, options: .init(acceleration: acceleration))
        let bakedScene = try PathTracingScene(queue: fixture.queue, graph: baked, options: .init(acceleration: acceleration))
        #expect(instancedScene.statistics.bottomLevelCount == 3 && instancedScene.statistics.instanceCount == 5)
        #expect(bakedScene.statistics.bottomLevelCount == 5)
        #expect(instancedScene.triangleCount == 26 && instancedScene.statistics.instancedTriangleCount == 50)

        let a = try fixture.trace(
            try PathTracer(queue: fixture.queue, scene: instancedScene, lighting: lighting),
            width: width, height: height, samples: 16, settings: settings
        )
        let b = try fixture.trace(
            try PathTracer(queue: fixture.queue, scene: bakedScene, lighting: lighting),
            width: width, height: height, samples: 16, settings: settings
        )
        let differences = zip(a, b).map { simd_reduce_max(abs($0 - $1)) }
        let mean = differences.reduce(0, +) / Float(differences.count)
        let outliers = differences.filter { $0 > 0.02 }.count
        print("Path instancing: mean difference \(mean), \(outliers) pixels above 0.02, largest \(differences.max() ?? 0)")
        #expect(mean < 0.002, "mean difference \(mean)")
        #expect(outliers <= differences.count / 100, "\(outliers) pixels differ by more than 0.02")
        // The mirrored box's front faces stay visible and lit.
        let mirrored = a[pixel(SIMD3(1.2, -0.3, -2.85))]
        #expect(mirrored.x > 0.01 && mirrored.x > mirrored.z)
    }

    @Test("Scene updates reject stale history and leave the scene unchanged when validation fails", arguments: PathAccelerationTests.modes)
    func historyAndValidation(acceleration: PathTracingScene.Options.Acceleration) throws {
        let plane = try fixture.pbr(.plane(width: 6, height: 6), color: SIMD3(0.7, 0.7, 0.7))
        let cube = try fixture.pbr(.cube(), color: SIMD3(0.8, 0.3, 0.2))
        let other = try fixture.pbr(.cube(), color: SIMD3(0.8, 0.3, 0.2))
        let cubeID = UUID()
        let groupID = UUID()
        var graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: plane),
            SceneNode(id: groupID, transform: .init(translation: SIMD3(-0.5, 0, -3)), children: [
                SceneNode(id: cubeID, transform: .init(scale: SIMD3(repeating: 0.5)), mesh: cube),
            ]),
        ], camera: fixture.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        let light = try PathTracer.AreaLight(
            center: SIMD3(0, 1.5, -2), aim: SIMD3(0, 0, -4), size: SIMD2(1, 1), radiance: SIMD3(repeating: 5),
            cameraVisible: false
        )
        let tracer = try PathTracer(queue: fixture.queue, scene: scene, lights: [light])
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let target = try fixture.output(width: width, height: height)
        func encode(after previous: PathTracer.History?) throws -> PathTracer.History {
            let buffer = try #require(fixture.queue.makeCommandBuffer())
            let history = try tracer.encode(
                camera: fixture.camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
                previous: previous, commandBuffer: buffer, settings: .init(maxBounces: 2, seed: 3)
            )
            try fixture.finish(buffer)
            return history
        }

        let first = try encode(after: nil)
        // Moving a parent moves the mesh beneath it.
        graph.roots[1].transform.translation = SIMD3(0.5, 0, -3)
        #expect(try scene.updateTransforms(graph))
        #expect(throws: PathTracer.Failure.sceneChanged) { try encode(after: first) }
        let second = try encode(after: nil)
        // Non-mesh nodes may change freely.
        graph.roots.append(SceneNode(light: try PunctualLight(kind: .point(), color: SIMD3(repeating: 1), intensity: 1)))
        #expect(try scene.updateTransforms(graph) == false)
        let third = try encode(after: second)
        #expect(third.sampleCount == 2)
        try scene.updateGeometry(for: [cubeID], in: graph)
        #expect(scene.revision == 2)
        #expect(throws: PathTracer.Failure.sceneChanged) { try encode(after: third) }

        let revision = scene.revision
        let statistics = scene.statistics
        var removed = graph
        removed.roots[1].children = []
        #expect(throws: PathTracingScene.Failure.structureChanged(cubeID)) { try scene.updateTransforms(removed) }
        var retinted = graph
        retinted.roots[1].children[0].materialParameters = SIMD4(0.5, 0.5, 0.5, 1)
        #expect(throws: PathTracingScene.Failure.structureChanged(cubeID)) { try scene.updateTransforms(retinted) }
        var swapped = graph
        swapped.roots[1].children[0].mesh = other
        #expect(throws: PathTracingScene.Failure.structureChanged(cubeID)) { try scene.updateTransforms(swapped) }
        var added = graph
        let addedID = UUID()
        added.roots.append(SceneNode(id: addedID, mesh: cube))
        #expect(throws: PathTracingScene.Failure.structureChanged(addedID)) { try scene.updateTransforms(added) }
        let stranger = UUID()
        #expect(throws: PathTracingScene.Failure.unknownNode(stranger)) {
            try scene.updateGeometry(for: [stranger], in: graph)
        }
        #expect(throws: PathTracingScene.Failure.unknownNode(groupID)) {
            try scene.updateGeometry(for: [], in: graph, replacements: [groupID: .cube()])
        }
        var palette = graph
        palette.roots[1].children[0].jointPalette = try JointPalette(matrices: [matrix_identity_float4x4])
        #expect(throws: PathTracingScene.Failure.skinning(cubeID, .unexpectedPalette)) {
            try scene.updateTransforms(palette)
        }
        var far = graph
        far.roots[1].transform.translation = SIMD3(2e12, 0, 0)
        #expect(throws: PathTracingScene.Failure.invalidGeometry(cubeID)) { try scene.updateTransforms(far) }
        #expect(scene.revision == revision)
        #expect(scene.statistics == statistics)
        _ = try encode(after: nil)
    }

    @Test("A skinned mesh traces its raster silhouette, and re-posing rebuilds only its bottom level", arguments: PathAccelerationTests.modes)
    func skinnedSilhouette(acceleration: PathTracingScene.Options.Acceleration) throws {
        let strip = try MeshGeometry.plane(width: 2, height: 0.4, columns: 8, rows: 1)
        let influences = strip.vertices.map { vertex in
            VertexSkinInfluence(joints: SIMD4(vertex.position.x > 0.01 ? 1 : 0, 0, 0, 0), weights: SIMD4(1, 0, 0, 0))
        }
        let skinned = try fixture.pbr(strip, skin: try MeshSkin(jointCount: 2, influences: influences))
        let cube = try fixture.pbr(.cube())
        func pose(_ angle: Float) throws -> JointPalette {
            try JointPalette(matrices: [
                matrix_identity_float4x4, SceneTransform(rotation: simd_quatf(angle: angle, axis: SIMD3(0, 0, 1))).matrix,
            ])
        }
        let stripID = UUID()
        var graph = SceneGraph(roots: [
            SceneNode(id: stripID, transform: .init(translation: SIMD3(0, -0.3, -3)), mesh: skinned, jointPalette: try pose(0.9)),
            SceneNode(
                transform: .init(translation: SIMD3(-1.6, 0.9, -4), rotation: simd_quatf(angle: 0.5, axis: SIMD3(0, 1, 0))),
                mesh: cube
            ),
        ], camera: fixture.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        let tracer = try PathTracer(queue: fixture.queue, scene: scene)
        let lighting = try PathTracer.Lighting(backgroundVisible: false)
        #expect(scene.statistics.bottomLevelCount == 2 && scene.statistics.bottomLevelBuildCount == 2)

        func compare(_ graph: SceneGraph) throws -> [Bool] {
            let raster = try fixture.rasterize(graph, width: width, height: height)
            let traced = try fixture.trace(
                tracer, width: width, height: height, samples: 64, settings: .init(maxBounces: 1, seed: 9),
                lighting: lighting
            )
            var mismatches = 0
            for (rasterPixel, tracedPixel) in zip(raster, traced) where (rasterPixel.w > 0.5) != (tracedPixel.w > 0.5) {
                mismatches += 1
                // A pixel center and its covered area can disagree only along an edge.
                #expect(tracedPixel.w > 0.15 && tracedPixel.w < 0.85, "coverage \(tracedPixel.w)")
            }
            #expect(mismatches <= raster.count / 50, "\(mismatches) silhouette mismatches")
            print("Path skinning: \(mismatches) of \(raster.count) pixels differ from the raster silhouette")
            return traced.map { $0.w > 0.5 }
        }

        let raised = try compare(graph)
        graph.roots[0].jointPalette = try pose(-0.7)
        #expect(throws: PathTracingScene.Failure.staleGeometry(stripID)) { try scene.updateTransforms(graph) }
        try scene.updateGeometry(for: [stripID], in: graph)
        #expect(scene.statistics.bottomLevelBuildCount == 3 && scene.statistics.bottomLevelCount == 2)
        let lowered = try compare(graph)
        #expect(zip(raised, lowered).filter { $0 != $1 }.count > 40)

        graph.roots[0].jointPalette = nil
        #expect(throws: PathTracingScene.Failure.skinning(stripID, .missingPalette)) {
            try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        }
    }

    @Test("Replacement geometry gives one node its own bottom level until it rejoins the shared one", arguments: PathAccelerationTests.modes)
    func replacementGeometry(acceleration: PathTracingScene.Options.Acceleration) throws {
        let card = try MeshGeometry.plane(width: 0.8, height: 0.8)
        let shared = try fixture.pbr(card)
        let aID = UUID()
        let graph = SceneGraph(roots: [
            SceneNode(id: aID, transform: .init(translation: SIMD3(-0.8, 0, -3)), mesh: shared),
            SceneNode(transform: .init(translation: SIMD3(0.8, 0, -3)), mesh: shared),
        ], camera: fixture.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        let tracer = try PathTracer(queue: fixture.queue, scene: scene)
        let lighting = try PathTracer.Lighting(backgroundVisible: false)
        func coverage() throws -> [SIMD4<Float>] {
            try fixture.trace(
                tracer, width: width, height: height, samples: 16, settings: .init(maxBounces: 1, seed: 4), lighting: lighting
            )
        }
        #expect(scene.statistics.bottomLevelCount == 1 && scene.statistics.instancedTriangleCount == 4)
        let original = try coverage()
        let aboveA = pixel(SIMD3(-0.8, 0.65, -3))
        let aboveB = pixel(SIMD3(0.8, 0.65, -3))
        #expect(original[aboveA].w == 0 && original[aboveB].w == 0)

        // Raise the top edge, as a morph target would.
        let morphs = try MeshMorphTargets(vertexCount: card.vertices.count, targets: [
            .init(positionOffsets: card.vertices.map { $0.position.y > 0 ? SIMD3(0, 0.6, 0) : .zero }),
        ])
        let raised = try morphs.apply(weights: [1], to: card)
        try scene.updateGeometry(for: [], in: graph, replacements: [aID: raised])
        #expect(scene.statistics.bottomLevelCount == 2 && scene.statistics.bottomLevelBuildCount == 2)
        #expect(scene.triangleCount == 4)
        let morphed = try coverage()
        #expect(morphed[aboveA].w == 1 && morphed[aboveB].w == 0)

        try scene.updateGeometry(for: [aID], in: graph)
        #expect(scene.statistics.bottomLevelCount == 1 && scene.statistics.bottomLevelBuildCount == 3)
        #expect(scene.triangleCount == 2)
        let restored = try coverage()
        #expect(PathTestFixture.maximumDifference(restored, original) < 1e-3)
        #expect(restored.map(\.w) == original.map(\.w))
    }

    @Test("A uniform-driven deformation rebuilds its renderer's shared level for every node on request", arguments: PathAccelerationTests.modes)
    func deformedRendererUpdate(acceleration: PathTracingScene.Options.Acceleration) throws {
        let graph = try SurfaceShaderGraph { s in
            .init(positionOffset: s.vector(s.uniform("Shift", Float(0)), s.constant(Float(0)), s.constant(Float(0))))
        }
        let uniforms = try ShaderUniforms(graph: graph)
        let card = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try .plane(width: 0.5, height: 0.5),
            material: .physicallyBased(try .init(
                lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, surfaceGraph: graph, surfaceUniforms: uniforms
            ))
        )
        let aID = UUID()
        let scene = SceneGraph(roots: [
            SceneNode(id: aID, transform: .init(translation: SIMD3(-1, 0.5, -3)), mesh: card),
            SceneNode(transform: .init(translation: SIMD3(-1, -0.5, -3)), mesh: card),
        ], camera: fixture.camera)
        let prepared = try PathTracingScene(queue: fixture.queue, graph: scene, options: .init(acceleration: acceleration))
        let tracer = try PathTracer(queue: fixture.queue, scene: prepared)
        let lighting = try PathTracer.Lighting(backgroundVisible: false)
        func coverage(_ point: SIMD3<Float>) throws -> Float {
            let image = try fixture.trace(
                tracer, width: width, height: height, samples: 4, settings: .init(maxBounces: 1, seed: 2), lighting: lighting
            )
            return image[pixel(point)].w
        }
        let shifted = [SIMD3<Float>(0.2, 0.5, -3), SIMD3<Float>(0.2, -0.5, -3)]
        #expect(try coverage(SIMD3(-1, 0.5, -3)) == 1 && coverage(shifted[0]) == 0)

        try uniforms.set("Shift", to: 1.2)
        // The snapshot keeps the prepared geometry until it is updated.
        #expect(try coverage(shifted[0]) == 0)
        try prepared.updateGeometry(for: [aID], in: scene)
        #expect(prepared.statistics.bottomLevelCount == 1 && prepared.statistics.bottomLevelBuildCount == 2)
        for point in shifted { #expect(try coverage(point) == 1) }
        #expect(try coverage(SIMD3(-1, 0.5, -3)) == 0)
    }

    @Test("Statistics count every level and report the bytes of every scene buffer", arguments: PathAccelerationTests.modes)
    func statistics(acceleration: PathTracingScene.Options.Acceleration) throws {
        let gridGeometry = try MeshGeometry.plane(width: 1, height: 1, columns: 5, rows: 3)
        let cubeGeometry = try MeshGeometry.cube()
        let grid = try fixture.pbr(gridGeometry)
        let cube = try fixture.pbr(cubeGeometry)
        var graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(-1, 0, -4)), mesh: grid),
            SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: grid),
            SceneNode(transform: .init(translation: SIMD3(1, 0, -4)), mesh: grid, materialParameters: SIMD4(0.5, 0.5, 0.5, 1)),
            SceneNode(transform: .init(translation: SIMD3(0, 1, -4)), mesh: cube),
        ], camera: fixture.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        let statistics = scene.statistics
        let software = acceleration == .software
        #expect(statistics.acceleration == acceleration && scene.accelerationKind == acceleration)
        #expect(statistics.instanceCount == 4 && statistics.bottomLevelCount == 2)
        #expect(statistics.triangleCount == 42 && statistics.instancedTriangleCount == 102)
        let bottomNodes = statistics.bottomLevelNodeCount
        let topNodes = statistics.topLevelNodeCount
        if software {
            // Leaves hold up to four triangles: at least 8 + 3 leaves, at most one per triangle.
            #expect((2 * 11 - 2)...(2 * 42 - 2) ~= bottomNodes && topNodes == 7)
            #expect(statistics.accelerationStructureByteCount == 0)
        } else {
            #expect(bottomNodes == 0 && topNodes == 0)
            #expect(statistics.accelerationStructureByteCount > 0)
        }
        #expect(scene.nodeCount == bottomNodes + topNodes && scene.triangleCount == 42 && scene.materialCount == 3)
        #expect(statistics.bufferByteCount == scene.buffers.reduce(0) { $0 + $1.length })
        let vertexCount = gridGeometry.vertices.count + cubeGeometry.vertices.count
        // Vertices and index triples, plus primitive numbers and nodes for the software BVH or one
        // unused top-level node and the instance descriptors for Metal.
        let levels = 48 * vertexCount + 12 * 42 + (software ? 4 * 42 + 48 * (bottomNodes + topNodes) : 48 + 64 * 4)
        let records = levels + 160 * 4
            + MemoryLayout<PathTracingScene.Material>.stride * 3 + MemoryLayout<UInt32>.stride
            + MemoryLayout<PathTracingScene.TextureTableEntry>.stride
        #expect(MemoryLayout<PathTracingScene.TextureTableEntry>.stride == 24)
        #expect(statistics.bufferByteCount == records)
        #expect(statistics.textureByteCount >= 8)
        #expect(statistics.boundTextureCount == 0 && statistics.bakedTextureCount == 0)
        #expect(statistics.referencedTextureByteCount == 0)

        graph.roots[0].transform.translation.y = -1
        try scene.updateTransforms(graph)
        #expect(scene.statistics.bufferByteCount == records)
        #expect(scene.statistics.revision == 1 && scene.statistics.bottomLevelBuildCount == 2)
        #expect(scene.statistics.topLevelRefitCount == 1)
    }

    // The pixel whose center sees a world point from the origin camera looking down -Z.
    private func pixel(_ point: SIMD3<Float>) -> Int {
        let tanY = Float(tan(Double.pi / 6))
        let tanX = tanY * Float(width) / Float(height)
        let x = (point.x / -point.z) / tanX
        let y = (point.y / -point.z) / tanY
        let column = min(width - 1, max(0, Int((x + 1) / 2 * Float(width))))
        let row = min(height - 1, max(0, Int((1 - y) / 2 * Float(height))))
        return row * width + column
    }

    // World-space geometry for an identity node, with the winding a mirrored transform reverses.
    private func bake(_ geometry: MeshGeometry, _ matrix: simd_float4x4) throws -> MeshGeometry {
        let linear = simd_float3x3(
            SIMD3(matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z),
            SIMD3(matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z),
            SIMD3(matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z)
        )
        let normalMatrix = simd_transpose(simd_inverse(linear))
        let vertices = geometry.vertices.map { vertex in
            let position = matrix * SIMD4(vertex.position, 1)
            return MeshVertex(
                position: SIMD3(position.x, position.y, position.z),
                normal: simd_normalize(normalMatrix * vertex.normal), textureCoordinate: vertex.textureCoordinate
            )
        }
        var indices = geometry.indices
        if simd_determinant(linear) < 0 {
            for offset in stride(from: 0, to: indices.count, by: 3) { indices.swapAt(offset + 1, offset + 2) }
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }
}
