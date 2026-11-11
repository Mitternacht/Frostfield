import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

struct PosedMeshBVHTests {
    @Test func `Recent poses reuse immutable snapshots and refit rather than repartition`() throws {
        let surface = try MeshGeometry.plane(width: 4, height: 4, columns: 4, rows: 4)
        let collapsedVertices = surface.vertices.map { MeshVertex(position: .zero, normal: $0.normal) }
        let collapsed = try MeshGeometry(vertices: collapsedVertices, indices: surface.indices)
        let cache = PosedMeshBVHCache()
        let full = surface.indices.indices
        var evaluations = 0
        let first = try cache.bvh(palette: nil, weights: [1], range: full) {
            evaluations += 1
            return collapsed
        }
        let second = try cache.bvh(palette: nil, weights: [2], range: full) {
            evaluations += 1
            return surface
        }
        #expect(first.nodeCount == 1)
        #expect(second.nodeCount == first.nodeCount)
        #expect(try MeshBVH(geometry: surface).nodeCount > 1)
        #expect(first.bounds.minimum == .zero)
        #expect(second.bounds == surface.bounds)

        enum CacheTestError: Error { case unexpectedEvaluation, invalidPose }
        let retained = try cache.bvh(palette: nil, weights: [1], range: full) { throw CacheTestError.unexpectedEvaluation }
        #expect(retained.bounds == first.bounds)
        #expect(throws: CacheTestError.invalidPose) {
            try cache.bvh(palette: nil, weights: [3], range: full) { throw CacheTestError.invalidPose }
        }
        _ = try cache.bvh(palette: nil, weights: [2], range: full) { throw CacheTestError.unexpectedEvaluation }
        let partial = try cache.bvh(palette: nil, weights: [2], range: 3..<6) {
            evaluations += 1
            return surface
        }
        #expect(partial.triangleCount == 1)
        #expect(partial.geometry.indices == Array(surface.indices[3..<6]))
        #expect(evaluations == 3)
        _ = try cache.bvh(palette: nil, weights: [1], range: full) {
            evaluations += 1
            return collapsed
        }
        #expect(evaluations == 4)
        #expect(first.geometry.vertices.allSatisfy { $0.position == .zero })
        #expect(second.geometry.vertices[0].position == surface.vertices[0].position)
    }

    @Test func `A moved joint and morph match the pinned skinned BVH refit fixture`() throws {
        let fixture = try GeometryRenderFixture()
        let source = try MeshGeometry.plane(width: 2, height: 2, columns: 4, rows: 4)
        let influence = VertexSkinInfluence(joints: .zero, weights: SIMD4(1, 0, 0, 0))
        let skin = try MeshSkin(jointCount: 1, influences: Array(repeating: influence, count: source.vertices.count))
        let morph = try MeshMorphTargets(vertexCount: source.vertices.count, targets: [
            .init(positionOffsets: Array(repeating: SIMD3(1, 0, 0.5), count: source.vertices.count)),
        ])
        let renderer = try fixture.unlit(source, skin: skin, morphTargets: morph)
        let bind = try JointPalette(matrices: [matrix_identity_float4x4])
        var graph = SceneGraph(roots: [SceneNode(mesh: renderer, jointPalette: bind)], camera: fixture.camera)
        let ray = try SceneRay(origin: SIMD3(3.25, 0.125, 3.5), direction: SIMD3(0, 0, -1), maximumDistance: 2)
        #expect(try graph.pick(ray: ray) == nil)

        graph.roots[0].jointPalette = try JointPalette(matrices: [
            SceneTransform(translation: SIMD3(2.5, 0, 1.25)).matrix,
        ])
        graph.roots[0].morphWeights = [0.5]
        let hit = try #require(try graph.pick(ray: ray))
        #expect(hit.intersection.position == SIMD3(3.25, 0.125, 1.5))
        #expect(hit.intersection.distance == 2)
        let short = try SceneRay(origin: ray.origin, direction: ray.direction, maximumDistance: Float(2).nextDown)
        #expect(try graph.pick(ray: short) == nil)
    }

    @Test func `Scene queries use the exact current skin and morph across shared draws and instance transforms`() throws {
        let fixture = try GeometryRenderFixture()
        let grid = try MeshGeometry.plane(width: 4, height: 4, columns: 4, rows: 4)
        let source = try MeshGeometry(vertices: grid.vertices.map { vertex in
            MeshVertex(position: vertex.position, normal: vertex.normal,
                       textureCoordinate: vertex.textureCoordinate,
                       secondaryTextureCoordinate: SIMD2(0.125, 0.25) + vertex.textureCoordinate * 0.5)
        }, indices: grid.indices)
        let skin = try MeshSkin(jointCount: 2, influences: source.vertices.map { vertex in
            VertexSkinInfluence(joints: SIMD4(0, 1, 0, 0), weights: vertex.position.x < 0
                                ? SIMD4(0.75, 0.25, 0, 0) : SIMD4(0.25, 0.75, 0, 0))
        })
        let morph = try MeshMorphTargets(vertexCount: source.vertices.count, targets: [
            .init(positionOffsets: source.vertices.map { SIMD3(0.5, 0, $0.position.x * 0.125) },
                  normalOffsets: Array(repeating: SIMD3(0.25, 0, 0), count: source.vertices.count)),
        ])
        let renderer = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: source, material: .unlit, skin: skin, morphTargets: morph,
            supportsInstancing: true, renderState: .init(side: .double)
        )
        let first = try JointPalette(matrices: [
            SceneTransform(translation: SIMD3(0.125, 0, 0.5)).matrix,
            SceneTransform(translation: SIMD3(-0.125, 0, 0.75), scale: SIMD3(1, 1.25, 1)).matrix,
        ])
        let second = try JointPalette(matrices: [
            SceneTransform(translation: SIMD3(0.5, 0, -0.5)).matrix,
            SceneTransform(translation: SIMD3(0, 0, -0.125), scale: SIMD3(0.75, 1, 1)).matrix,
        ])
        let instances = try InstanceSet(device: fixture.device, transforms: [
            matrix_identity_float4x4, SceneTransform(translation: SIMD3(0, 0, -1)).matrix,
        ])
        var graph = SceneGraph(roots: [
            SceneNode(
                transform: SceneTransform(translation: SIMD3(-5, 0, -3), scale: SIMD3(-1.5, 0.75, 1.25)),
                mesh: renderer, jointPalette: first, instances: instances, morphWeights: [1]
            ),
            SceneNode(transform: SceneTransform(translation: SIMD3(5, 0, -4)),
                      mesh: renderer, jointPalette: second, morphWeights: [-0.5]),
        ], camera: fixture.camera)

        func compareQueries(in graph: SceneGraph, range: Range<Int>) throws {
            let nodes = graph.roots
            let poses = try nodes.map { node in
                try skin.deform(morph.apply(weights: node.morphWeights ?? [0], to: source), using: #require(node.jointPalette))
            }
            let meshes = try poses.map {
                try MeshGeometry(vertices: $0.vertices, indices: Array($0.indices[range]))
            }
            for (index, mesh) in meshes.enumerated() {
                let node = nodes[index]
                let center = mesh.indices.prefix(3).reduce(SIMD3<Float>.zero) {
                    $0 + mesh.vertices[Int($1)].position / 3
                }
                let placed = node.transform.matrix * SIMD4(center, 1)
                let world = SIMD3(placed.x, placed.y, placed.z)
                for side: Float in [1, -1] {
                    let ray = try SceneRay(
                        origin: world + SIMD3(0, 0, side * 3),
                        direction: SIMD3(0, 0, -side), maximumDistance: 6
                    )
                    var oracle: [SceneHit] = []
                    for (candidate, other) in nodes.enumerated() {
                        let transforms = other.instances?.transforms ?? [matrix_identity_float4x4]
                        for (instance, transform) in transforms.enumerated() {
                            let hits = try meshes[candidate].intersections(
                                ray: ray, localToWorld: other.transform.matrix * transform, side: .double
                            )
                            oracle += hits.map {
                                SceneHit(nodeID: other.id, intersection: $0.offsetTriangleIndex(by: range.lowerBound / 3),
                                         instanceIndex: other.instances == nil ? nil : instance)
                            }
                        }
                    }
                    oracle.sort { $0.intersection.distance < $1.intersection.distance }
                    #expect(try graph.intersections(ray: ray) == oracle)
                    #expect(try graph.raycast(ray: ray) == oracle.map(SceneRaycastHit.mesh))
                    #expect(try graph.pick(ray: ray) == oracle.first)
                    let hit = try #require(oracle.first)
                    #expect(hit.nodeID == node.id)
                    #expect(hit.intersection.isFrontFace == (side > 0))
                    #expect(hit.intersection.secondaryTextureCoordinate != nil)
                    #expect(hit.intersection.triangleIndex >= range.lowerBound / 3)
                    if index == 0 {
                        #expect(Set(oracle.compactMap(\.instanceIndex)) == [0, 1])
                    }
                }
            }
        }

        let selected = 12..<48
        renderer.drawRange = try MeshDrawRange(start: selected.lowerBound, count: selected.count)
        try compareQueries(in: graph, range: selected)
        graph.roots[0].morphWeights = [-0.75]
        graph.roots[1].jointPalette = first
        try compareQueries(in: graph, range: selected)
        let shifted = 24..<60
        renderer.drawRange = try MeshDrawRange(start: shifted.lowerBound, count: shifted.count)
        try compareQueries(in: graph, range: shifted)

        let probe = try SceneRay(origin: SIMD3(-5, 0, 1), direction: SIMD3(0, 0, -1))
        graph.roots[0].morphWeights = [.nan]
        #expect(throws: MeshMorphTargets.ValidationError.nonfiniteWeight) { try graph.intersections(ray: probe) }
        graph.roots[0].morphWeights = nil
        graph.roots[0].jointPalette = nil
        #expect(throws: SkinningError.missingPalette) { try graph.pick(ray: probe) }
    }
}
