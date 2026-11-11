import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct CullingAndLODTests {
    private let fixture: GeometryRenderFixture

    init() throws {
        fixture = try GeometryRenderFixture()
    }

    private func node(_ mesh: MeshRenderer, at position: SIMD3<Float>) -> SceneNode {
        SceneNode(transform: SceneTransform(translation: position), mesh: mesh)
    }

    @Test("Meshes outside the camera frustum are skipped and the image matches drawing everything")
    func frustumCulling() throws {
        let mesh = try fixture.unlit(try GeometryRenderFixture.square(0.25))
        // Visible, behind the camera, beside the view, and beyond the far plane.
        let roots = [
            node(mesh, at: SIMD3(0, 0, -2)), node(mesh, at: SIMD3(0, 0, 2)),
            node(mesh, at: SIMD3(-10, 0, -2)), node(mesh, at: SIMD3(0, 0, -60)),
        ]
        let culled = try fixture.render(SceneGraph(roots: roots, camera: fixture.camera))
        #expect(culled.statistics.drawnMeshCount == 1)
        #expect(culled.statistics.culledMeshCount == 3)
        #expect(culled.statistics.drawnInstanceCount == 1)

        let unculled = try fixture.render(SceneGraph(
            roots: roots.map { node in var node = node; node.frustumCulled = false; return node }, camera: fixture.camera
        ))
        #expect(unculled.statistics.drawnMeshCount == 4)
        #expect(unculled.statistics.culledMeshCount == 0)
        #expect(culled.pixels == unculled.pixels)
        #expect(fixture.pixel(culled.pixels, x: 32, y: 32).w == 1)

        // An explicit view culls with its own matrix: this one looks at the square beside the default view.
        let target = try fixture.makeTarget()
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        let eye = SceneTransform(translation: SIMD3(-10, 0, 0))
        let clipFromWorld = try fixture.camera.projection(aspectRatio: 1) * simd_inverse(eye.matrix)
        let statistics = try SceneGraph(roots: roots, camera: fixture.camera).encode(
            pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: buffer,
            clipFromWorld: clipFromWorld, cameraPosition: eye.translation
        )
        try fixture.finish(buffer)
        #expect(statistics.drawnMeshCount == 1 && statistics.culledMeshCount == 3)
        #expect(fixture.pixel(try fixture.pixels(of: target.color), x: 32, y: 32).w == 1)
    }

    @Test("Morphed, instanced, and skinned draws cull with bounds that enclose what they draw")
    func conservativeBounds() throws {
        // Morph offsets widen the box by each weighted target, including negative weights.
        let square = try GeometryRenderFixture.square(0.25)
        let slide = try MeshMorphTargets(vertexCount: 4, targets: [
            .init(positionOffsets: Array(repeating: SIMD3(3, 0, 0), count: 4)),
        ])
        let morphed = try fixture.unlit(square, morphTargets: slide)
        for (x, weight, visible) in [(Float(-3), Float(1), true), (-3, 0, false), (3, -1, true), (3, 0.5, false)] {
            var draw = node(morphed, at: SIMD3(x, 0, -2))
            draw.morphWeights = [weight]
            let result = try fixture.render(SceneGraph(roots: [draw], camera: fixture.camera))
            #expect(result.statistics.culledMeshCount == (visible ? 0 : 1), "x \(x), weight \(weight)")
            #expect((fixture.pixel(result.pixels, x: 32, y: 32).w == 1) == visible)
        }

        // An instanced node is culled only when every instance is outside.
        let instanced = try fixture.unlit(square, supportsInstancing: true)
        let set = try InstanceSet(device: fixture.device, transforms: [translationMatrix(-10), translationMatrix(0)])
        var group = node(instanced, at: SIMD3(0, 0, -2))
        group.instances = set
        let partly = try fixture.render(SceneGraph(roots: [group], camera: fixture.camera))
        #expect(partly.statistics.culledMeshCount == 0 && partly.statistics.drawnInstanceCount == 2)
        #expect(partly.statistics.submittedTriangleCount == 4)
        // The union of the instance boxes must leave the view, so both move to the same side.
        try set.setTransform(translationMatrix(-12), at: 1)
        let outside = try fixture.render(SceneGraph(roots: [group], camera: fixture.camera))
        #expect(outside.statistics.culledMeshCount == 1 && outside.statistics.drawnMeshCount == 0)

        // Skinned meshes are never culled unless the node supplies bounds for every pose.
        let skin = try MeshSkin(
            jointCount: 1,
            influences: Array(repeating: VertexSkinInfluence(joints: .zero, weights: SIMD4(1, 0, 0, 0)), count: 4)
        )
        let skinned = try fixture.unlit(square, skin: skin)
        var posed = node(skinned, at: SIMD3(-10, 0, -2))
        posed.jointPalette = try JointPalette(matrices: [translationMatrix(10)])
        let unbounded = try fixture.render(SceneGraph(roots: [posed], camera: fixture.camera))
        #expect(unbounded.statistics.culledMeshCount == 0)
        #expect(fixture.pixel(unbounded.pixels, x: 32, y: 32).w == 1)
        posed.cullingBounds = try AxisAlignedBox(minimum: SIMD3(-0.25, -0.25, 0), maximum: SIMD3(10.25, 0.25, 0))
        let bounded = try fixture.render(SceneGraph(roots: [posed], camera: fixture.camera))
        #expect(bounded.statistics.culledMeshCount == 0)
        #expect(bounded.pixels == unbounded.pixels)
        posed.jointPalette = try JointPalette(matrices: [matrix_identity_float4x4])
        posed.cullingBounds = try AxisAlignedBox(minimum: SIMD3(-0.25, -0.25, 0), maximum: SIMD3(0.25, 0.25, 0))
        let hidden = try fixture.render(SceneGraph(roots: [posed], camera: fixture.camera))
        #expect(hidden.statistics.culledMeshCount == 1)
    }

    @Test("Shadow passes keep casters outside the camera view and skip casters outside a light's view")
    func shadowCasters() throws {
        let receiver = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            mesh: try fixture.unlit(try GeometryRenderFixture.square(1.6), material: .shadowedLambert),
            castsShadow: false
        )
        let blockerMesh = try fixture.unlit(try GeometryRenderFixture.square(0.25))
        // Behind the camera, this blocker shades the receiver's center along (0.6, 0, 0.8).
        let behind = node(blockerMesh, at: SIMD3(2.625, 0, 0.5))
        // Far above, this one is outside both the camera and the light's volume.
        let above = node(blockerMesh, at: SIMD3(0, 12, -3))
        let shadowMap = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 512, directionToLight: SIMD3(0.6, 0, 0.8),
            center: SIMD3(0, 0, -3), halfExtent: 5
        )
        let roots = [receiver, behind, above]
        let culled = try fixture.render(SceneGraph(roots: roots, camera: fixture.camera), shadowMap: shadowMap)
        #expect(culled.statistics.drawnMeshCount == 1)
        #expect(culled.statistics.culledMeshCount == 2)
        #expect(culled.statistics.shadowDrawCount == 1)
        #expect(culled.statistics.submittedTriangleCount == 2 && culled.statistics.shadowTriangleCount == 2)
        #expect(culled.statistics.culledShadowCount == 1)
        #expect(abs(fixture.pixel(culled.pixels, x: 32, y: 32).x - 0.16) < 0.01)
        #expect(abs(fixture.pixel(culled.pixels, x: fixture.column(x: -1, depth: 3), y: 32).x - 0.832) < 0.01)
        let unculled = try fixture.render(SceneGraph(
            roots: roots.map { node in var node = node; node.frustumCulled = false; return node }, camera: fixture.camera
        ), shadowMap: shadowMap)
        #expect(unculled.statistics.culledMeshCount == 0)
        #expect(culled.pixels == unculled.pixels)

        // Punctual: one occluder inside the spot's view casts; one beyond its cone and one behind the light do not.
        let lighting = try PunctualLightingFixture()
        let floor = SceneNode(mesh: try lighting.floor(halfExtent: 4))
        let spot = SceneNode(
            transform: PunctualLightingFixture.aim(SIMD3(0, 3, 0), at: .zero),
            light: try PunctualLight(
                kind: .spot(innerConeAngle: 0.7, outerConeAngle: 0.87), intensity: 20, shadow: PunctualLight.Shadow()
            )
        )
        let box = try lighting.box()
        let occluders = [SIMD3<Float>(1, 1.5, 0), SIMD3(3.5, 1.5, 0), SIMD3(0, 5, 0)].map {
            SceneNode(transform: SceneTransform(translation: $0, scale: SIMD3(repeating: 0.6)), mesh: box)
        }
        let maps = try PunctualShadowMaps(queue: lighting.queue, resolution: 512, spotCapacity: 1, pointCapacity: 0)
        let graph = SceneGraph(roots: [floor, spot] + occluders, camera: try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 20))
        let target = try fixture.makeTarget()
        let buffer = try #require(lighting.queue.makeCommandBuffer())
        let scale: Float = 1 / 4
        let clipFromWorld = simd_float4x4(columns: (
            SIMD4(scale, 0, 0, 0), SIMD4(0, 0, -1.0 / 20, 0), SIMD4(0, -scale, 0, 0), SIMD4(0, 0, 0.5, 1)
        ))
        let statistics = try graph.encode(
            pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: buffer,
            clipFromWorld: clipFromWorld, cameraPosition: SIMD3(0, 10, 0), directionToCamera: SIMD3(0, 1, 0),
            punctualShadowMaps: maps
        )
        try fixture.finish(buffer)
        // The floor and the in-cone occluder cast into the one spot slice.
        #expect(statistics.shadowDrawCount == 2)
        #expect(statistics.culledShadowCount == 2)
        #expect(statistics.drawnMeshCount == 4)
        let shadowed = try lighting.render([floor, spot, occluders[0]], maps: maps)
        let pixels = try lighting.render([floor, spot] + occluders, maps: maps)
        let probe = lighting.pixel(2.2, 0)
        #expect(pixels[probe].x < 0.05)
        #expect(abs(pixels[probe].x - shadowed[probe].x) <= 0.001)
    }

    @Test("Levels switch at their distances, with hysteresis toward finer levels")
    func levelSelection() throws {
        let mesh = try fixture.unlit(try GeometryRenderFixture.square(0.25))
        let levels = [
            LevelOfDetail.Level(mesh: mesh, distance: 0),
            LevelOfDetail.Level(mesh: mesh, distance: 10, hysteresis: 0.1),
            LevelOfDetail.Level(mesh: mesh, distance: 20, hysteresis: 0.1),
        ]
        let fine = try LevelOfDetail(levels: levels)
        #expect([9.99, 10, 19.99, 20, 100].map { fine.level(forDistance: $0) } == [0, 1, 1, 2, 2])
        let coarse = try LevelOfDetail(levels: levels, currentLevel: 2)
        // Leaving level 2 needs 20 × (1 − 0.1) = 18.
        #expect([19, 18, 17.99, 9.99].map { coarse.level(forDistance: $0) } == [2, 2, 1, 0])
        let middle = try LevelOfDetail(levels: levels, currentLevel: 1)
        #expect([9.5, 9, 8.99, 19.99].map { middle.level(forDistance: $0) } == [1, 1, 0, 1])
        #expect(middle.level(forDistance: .nan) == 1)

        // A graph carries the selection between frames through updateLevelsOfDetail.
        let lod = SceneNode(levelsOfDetail: fine)
        var graph = SceneGraph(roots: [lod], camera: fixture.camera)
        var selections: [Int] = []
        for distance: Float in [5, 15, 19, 21, 19, 17, 9.5, 8.5] {
            graph.cameraTransform.translation = SIMD3(0, 0, distance)
            #expect(try graph.resolve().first?.levelOfDetail == graph.resolve(cameraPosition: SIMD3(0, 0, distance)).first?.levelOfDetail)
            try graph.updateLevelsOfDetail()
            selections.append(try #require(graph.roots[0].levelsOfDetail?.currentLevel))
        }
        #expect(selections == [0, 1, 1, 2, 2, 1, 1, 0])
        // Without updates, each resolve compares against the stored level only.
        graph.cameraTransform.translation = SIMD3(0, 0, 25)
        #expect(try graph.resolve().first?.levelOfDetail == 2)
        #expect(graph.roots[0].levelsOfDetail?.currentLevel == 0)
        // An explicit position overrides the graph camera.
        try graph.updateLevelsOfDetail(cameraPosition: SIMD3(0, 12, 0))
        #expect(graph.roots[0].levelsOfDetail?.currentLevel == 1)
    }

    @Test("Color, shadow, picking, and path tracing all use the level selected for the camera")
    func selectedLevelEverywhere() throws {
        // Level 0 is a large square; level 1 a small one offset up and away from the receiver's center.
        let near = try fixture.unlit(try GeometryRenderFixture.square(0.25))
        let far = try fixture.unlit(try MeshGeometry(
            vertices: try GeometryRenderFixture.square(0.05).vertices.map {
                MeshVertex(position: $0.position + SIMD3(0, 1, 0), normal: $0.normal)
            },
            indices: [0, 1, 2, 0, 2, 3]
        ))
        let lod = try LevelOfDetail(levels: [.init(mesh: near, distance: 0), .init(mesh: far, distance: 3)])
        let blocker = SceneNode(transform: SceneTransform(translation: SIMD3(0.75, 0, -2)), levelsOfDetail: lod)
        let receiver = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            mesh: try fixture.unlit(try GeometryRenderFixture.square(1.6), material: .shadowedLambert),
            castsShadow: false
        )
        let shadowMap = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 256, directionToLight: SIMD3(0.6, 0, 0.8),
            center: SIMD3(0, 0, -3), halfExtent: 3
        )
        var graph = SceneGraph(roots: [receiver, blocker], camera: fixture.camera)
        let close = try fixture.render(graph, shadowMap: shadowMap).pixels
        // From the origin the blocker is about 2.1 m away: level 0 shadows the receiver's center.
        #expect(try graph.resolve()[1].levelOfDetail == 0)
        #expect(abs(fixture.pixel(close, x: 32, y: 32).x - 0.16) < 0.01)
        #expect(try graph.pick(at: SIMD2(0.75 / 2 * 0.5 + 0.5, 0.5), aspectRatio: 1)?.nodeID == blocker.id)

        graph.cameraTransform.translation = SIMD3(0, 0, 3)
        let distant = try fixture.render(graph, shadowMap: shadowMap).pixels
        #expect(try graph.resolve()[1].levelOfDetail == 1)
        #expect(abs(fixture.pixel(distant, x: 32, y: 32).x - 0.832) < 0.01)
        #expect(try graph.pick(at: SIMD2(0.75 / 5 * 0.5 + 0.5, 0.5), aspectRatio: 1)?.nodeID != blocker.id)

        // A path-tracing snapshot bakes the level selected for the graph's camera.
        let material = MeshMaterial.physicallyBased(try PhysicallyBasedMaterial(alphaMode: .opaque))
        let cube = try MeshRenderer(device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .cube(), material: material)
        let card = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try GeometryRenderFixture.square(0.5),
            material: material
        )
        let pathLOD = try LevelOfDetail(levels: [.init(mesh: cube, distance: 0), .init(mesh: card, distance: 3)])
        var pathGraph = SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), levelsOfDetail: pathLOD)],
            camera: fixture.camera
        )
        #expect(try PathTracingScene(queue: fixture.queue, graph: pathGraph, textureSize: 16).triangleCount == 12)
        pathGraph.cameraTransform.translation = SIMD3(0, 0, 2)
        #expect(try PathTracingScene(queue: fixture.queue, graph: pathGraph, textureSize: 16).triangleCount == 2)
    }

    @Test("Level-of-detail settings are validated")
    func validation() throws {
        let mesh = try fixture.unlit(try GeometryRenderFixture.square(0.25))
        #expect(throws: LevelOfDetail.Failure.levelCount) { try LevelOfDetail(levels: []) }
        #expect(throws: LevelOfDetail.Failure.levelCount) {
            try LevelOfDetail(levels: (0...LevelOfDetail.maximumLevelCount).map { .init(mesh: mesh, distance: Float($0)) })
        }
        #expect(throws: LevelOfDetail.Failure.distance(1)) {
            try LevelOfDetail(levels: [.init(mesh: mesh, distance: 1), .init(mesh: mesh, distance: 1)])
        }
        #expect(throws: LevelOfDetail.Failure.distance(0)) { try LevelOfDetail(levels: [.init(mesh: mesh, distance: -1)]) }
        #expect(throws: LevelOfDetail.Failure.distance(1)) {
            try LevelOfDetail(levels: [.init(mesh: mesh, distance: 0), .init(mesh: mesh, distance: .infinity)])
        }
        #expect(throws: LevelOfDetail.Failure.hysteresis(0)) {
            try LevelOfDetail(levels: [.init(mesh: mesh, distance: 0, hysteresis: 1)])
        }
        #expect(throws: LevelOfDetail.Failure.hysteresis(0)) {
            try LevelOfDetail(levels: [.init(mesh: mesh, distance: 0, hysteresis: -0.1)])
        }
        #expect(throws: LevelOfDetail.Failure.currentLevel) {
            try LevelOfDetail(levels: [.init(mesh: mesh, distance: 0)], currentLevel: 1)
        }

        let lod = try LevelOfDetail(levels: [.init(mesh: mesh, distance: 0)])
        let both = SceneNode(mesh: mesh, levelsOfDetail: lod)
        #expect(throws: SceneGraph.ValidationError.meshAndLevelsOfDetail(both.id)) {
            try SceneGraph(roots: [both], camera: fixture.camera).resolve()
        }
        var lost = SceneGraph(roots: [SceneNode(levelsOfDetail: lod)], camera: fixture.camera)
        lost.cameraTransform.translation = SIMD3(.nan, 0, 0)
        #expect(throws: SceneGraph.ValidationError.invalidCameraPosition) { try lost.resolve() }
        // Without level-of-detail nodes, resolution does not depend on the camera.
        lost.roots = [SceneNode(mesh: mesh)]
        #expect(try lost.resolve().count == 1)
    }
}
