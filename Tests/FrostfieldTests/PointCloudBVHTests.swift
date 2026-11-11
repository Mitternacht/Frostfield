import Foundation
@testable import Frostfield
import Testing
import simd

struct PointCloudBVHTests {
    @Test func `A bounded identity ray matches the pinned PointsBVH fixture`() throws {
        let cloud = try PointCloud(positions: [
            SIMD3(0, 0.25, 1), SIMD3(0.625, 0, 1), SIMD3(0, 0, 3), SIMD3(0, 0, 4.125), SIMD3(0.375, 0, 4),
        ])
        let ray = try SceneRay(origin: .zero, direction: SIMD3(0, 0, 1), maximumDistance: 4)
        let hits = try PointCloudBVH(cloud: cloud).intersections(ray: ray, threshold: 0.5)
        #expect(hits.map(\.element) == [.point(0), .point(2), .point(4)])
        #expect(hits.map(\.distance) == [1, 3, 4])
        #expect(hits.map(\.distanceToRay) == [0.25, 0, 0.375])
    }

    @Test func `Mirrored nonuniform points preserve exact threshold, ray limit and source ordering`() throws {
        let near: [SIMD3<Float>] = [
            SIMD3(0, 2, 4), SIMD3(0, 0, 2), SIMD3(0, -1, 4), SIMD3(0.5, 0, 3),
            SIMD3(0, 0, 5.5), SIMD3(0, 0, -0.25), SIMD3(0, 0, 5),
        ]
        let far = (0..<200).map { SIMD3(Float($0 + 8), Float($0 % 4), Float($0 % 5)) }
        let cloud = try PointCloud(positions: near + far, sizes: [1, 0, 1, 1, 1, 1, 1] + Array(repeating: 1, count: far.count))
        let bvh = PointCloudBVH(cloud: cloud)
        #expect(bvh.nodeCount > 1)
        let transform = SceneTransform(translation: SIMD3(1, 0, 0), scale: SIMD3(-2, 0.25, 1)).matrix
        let ray = try SceneRay(origin: SIMD3(1, 0, 0), direction: SIMD3(0, 0, 4), maximumDistance: 5)
        let hits = try bvh.intersections(ray: ray, localToWorld: transform, threshold: 0.5)
        #expect(hits == (try cloud.intersections(ray: ray, localToWorld: transform, threshold: 0.5)))
        #expect(hits.map(\.element) == [.point(5), .point(0), .point(2), .point(6)])
        #expect(hits.map(\.distanceToRay) == [0.25, 0.5, 0.25, 0])
        let range = try GeometryDrawRange(start: 2, count: 4)
        #expect(try bvh.intersections(ray: ray, localToWorld: transform, threshold: 0.5, drawRange: range)
            .map(\.element) == [.point(5), .point(2)])
        let empty = try GeometryDrawRange(start: Int.max, count: Int.max)
        #expect(try bvh.intersections(ray: ray, localToWorld: transform, drawRange: empty).isEmpty)

        let originOnly = try SceneRay(origin: ray.origin, direction: ray.direction, maximumDistance: 0)
        #expect(try bvh.intersections(ray: originOnly, localToWorld: transform, threshold: 0.25).map(\.element) == [.point(5)])
    }

    @Test func `Pruned point queries equal the linear scan for affine placements and draw windows`() throws {
        var random = BVHTestRandom(seed: 0xB7D)
        let points = (0..<3_000).map { _ in random.vector(in: -30...30) }
        let sizes: [Float] = points.indices.map { $0.isMultiple(of: 7) ? 0 : 1 }
        let cloud = try PointCloud(positions: points, sizes: sizes)
        let bvh = PointCloudBVH(cloud: cloud)
        #expect(bvh.nodeCount > 100)
        let transforms = [
            matrix_identity_float4x4,
            simd_float4x4(SIMD4(-2, 0.5, 0, 0), SIMD4(0.25, 0.5, 0.125, 0),
                          SIMD4(0, 0.25, 3, 0), SIMD4(12, -3, 5, 1)),
            SceneTransform(translation: SIMD3(2e6, -1e6, 3e6),
                           rotation: simd_quatf(angle: 0.45, axis: SIMD3(0, 1, 0)),
                           scale: SIMD3(0.0625, 2, -2)).matrix,
        ]
        let directions: [SIMD3<Float>] = [SIMD3(0, 0, 1), SIMD3(3, 2, -1), SIMD3(-1, 1, 2)]
        var hitQueries = 0, missQueries = 0
        for transform in transforms {
            for queryIndex in 0..<80 {
                let source = Int.random(in: points.indices, using: &random)
                let world = transform * SIMD4(points[source], 1)
                let direction = directions[queryIndex % directions.count]
                let origin = SIMD3(world.x, world.y, world.z) - simd_normalize(direction) * 4
                let maximumDistance: Float = queryIndex.isMultiple(of: 3) ? 3 : 7
                let ray = try SceneRay(origin: origin, direction: direction, maximumDistance: maximumDistance)
                let threshold: Float = queryIndex.isMultiple(of: 5) ? 0.75 : 0.25
                let range: GeometryDrawRange = queryIndex.isMultiple(of: 2)
                    ? .all : try .init(start: max(0, source - 7), count: 80)
                let expected = try cloud.intersections(ray: ray, localToWorld: transform, threshold: threshold, drawRange: range)
                #expect(try bvh.intersections(ray: ray, localToWorld: transform, threshold: threshold, drawRange: range) == expected)
                if expected.isEmpty { missQueries += 1 } else { hitQueries += 1 }
            }
        }
        #expect(hitQueries > 80)
        #expect(missQueries > 50)
    }

    @Test func `Coincident and uneven clouds keep source IDs and prune a narrow ray through a large grid`() throws {
        let grid = (0..<128).flatMap { x in
            (0..<64).map { y in SIMD3(Float(x), Float(y), Float(1)) }
        }
        let cloud = try PointCloud(positions: grid)
        let bvh = PointCloudBVH(cloud: cloud)
        let ray = try SceneRay(origin: SIMD3(63, 27, 0), direction: SIMD3(0, 0, 1), maximumDistance: 1)
        let query = try PrimitiveWorldRay(ray, threshold: 0, localToWorld: matrix_identity_float4x4)
        var candidateCount = 0
        bvh.forEachRayCandidate(query, range: grid.indices) { _ in candidateCount += 1 }
        #expect(candidateCount > 0 && candidateCount <= 32)
        #expect(try bvh.intersections(ray: ray, threshold: 0).map(\.element) == [.point(63 * 64 + 27)])

        let repeated = Array(repeating: SIMD3<Float>(1, 1, 0), count: 128)
        let uneven = try PointCloud(positions: repeated + [SIMD3(10_000, 10_000, 10_000)])
        let split = PointCloudBVH(cloud: uneven)
        let diagonal = try SceneRay(origin: .zero, direction: SIMD3(4, 4, 0))
        let range = try GeometryDrawRange(start: 53, count: 3)
        #expect(split.nodeCount > 1)
        #expect(try split.intersections(ray: diagonal, threshold: 0, drawRange: range).map(\.element)
            == [.point(53), .point(54), .point(55)])
    }

    @Test func `Zero tolerance and extreme affine inputs preserve the linear query contract`() throws {
        let cloud = try PointCloud(positions: [SIMD3(1, 1, 0), SIMD3(1, 1, 0.001)])
        let bvh = PointCloudBVH(cloud: cloud)
        let translation = SIMD3<Float>(1e30, -1e30, 2e30)
        let placement = SceneTransform(translation: translation).matrix
        let ray = try SceneRay(origin: translation, direction: SIMD3(1, 1, 0))
        let result = try bvh.intersections(ray: ray, localToWorld: placement, threshold: 0)
        #expect(result.map(\.element) == [.point(0)])
        #expect(try cloud.intersections(ray: ray, localToWorld: placement, threshold: 0) == result)

        let collapsed = SceneTransform(scale: SIMD3(1, 0, 1)).matrix
        #expect(throws: ScenePickingError.invalidTransform) { try bvh.intersections(ray: ray, localToWorld: collapsed) }
        for threshold: Float in [-1, .nan, .infinity] {
            #expect(throws: PrimitivePickingError.tolerance) { try bvh.intersections(ray: ray, threshold: threshold) }
        }
        let extreme = try PointCloudBVH(cloud: PointCloud(positions: [SIMD3(.greatestFiniteMagnitude, 0, 0)]))
        let unbounded = try SceneRay(origin: .zero, direction: SIMD3(1, 0, 0))
        let scale = SceneTransform(scale: SIMD3(2, 1, 1)).matrix
        #expect(throws: ScenePickingError.unrepresentableResult) {
            try extreme.intersections(ray: unbounded, localToWorld: scale, threshold: 0)
        }
        let bounded = try SceneRay(origin: .zero, direction: SIMD3(1, 0, 0), maximumDistance: 1)
        #expect(try extreme.intersections(ray: bounded, localToWorld: scale, threshold: 0).isEmpty)
    }

    @Test func `Scene world picking uses the renderer's BVH and current source draw range`() throws {
        let fixture = try GeometryRenderFixture()
        let points = (0..<96).map { SIMD3(Float($0 % 3) * 0.25, 0, -Float($0 / 3)) }
        let cloud = try PointCloud(positions: points)
        let material = PointsMaterial(size: .pixels(8))
        let ordinary = try PointsRenderer(device: fixture.device, cloud: cloud, material: material)
        let bvh = PointCloudBVH(cloud: cloud)
        let accelerated = try PointsRenderer(device: fixture.device, pickingBVH: bvh, material: material)
        #expect(ordinary.pickingBVH == nil)
        #expect(accelerated.pickingBVH?.nodeCount == bvh.nodeCount)
        let pose = SceneTransform(translation: SIMD3(1, 0, 0), scale: SIMD3(-2, 0.5, 1))
        var node = SceneNode(transform: pose, primitive: .points(ordinary))
        let baseline = SceneGraph(roots: [node], camera: fixture.camera)
        node.primitive = .points(accelerated)
        let graph = SceneGraph(roots: [node], camera: fixture.camera)
        let ray = try SceneRay(origin: SIMD3(1, 0, 1), direction: SIMD3(0, 0, -1), maximumDistance: 5)
        for range: GeometryDrawRange in [.all, try .init(start: 4, count: 7), try .init(start: 4, count: 0)] {
            ordinary.drawRange = range
            accelerated.drawRange = range
            let actual = try graph.raycast(ray: ray, pointThreshold: 0.5)
            #expect(try baseline.raycast(ray: ray, pointThreshold: 0.5) == actual)
            let sources = actual.compactMap { hit -> PrimitiveHit.Element? in
                if case .primitive(let value) = hit { return value.element }
                return nil
            }
            if range == .all {
                #expect(sources == (0..<15).filter { $0 % 3 != 2 }.map(PrimitiveHit.Element.point))
            } else if range.count == 7 {
                #expect(sources == [.point(4), .point(6), .point(7), .point(9), .point(10)])
            } else {
                #expect(sources.isEmpty)
            }
        }
    }
}
