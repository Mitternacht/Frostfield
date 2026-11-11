import Frostfield
import Testing
import simd

struct MeshBVHShapeCastTests {
    @Test func `A custom sphere cast agrees with exhaustive triangle testing and prunes the tree`() throws {
        let geometry = try MeshGeometry.plane(width: 16, height: 16, columns: 16, rows: 16)
        let bvh = try MeshBVH(geometry: geometry)
        let sphere = try BoundingSphere(center: SIMD3(0.2, 0.3, 0.4), radius: 3)
        func overlaps(_ triangle: Triangle3D) throws -> Bool {
            let point = try triangle.closestPoint(to: sphere.center)
            return simd_length_squared(point - sphere.center) <= sphere.radius * sphere.radius
        }
        var expected = Set<Int>()
        for index in 0..<bvh.triangleCount {
            let offset = index * 3
            let corners = geometry.indices[offset..<(offset + 3)].map { geometry.vertices[Int($0)].position }
            if try overlaps(Triangle3D(corners[0], corners[1], corners[2])) { expected.insert(index) }
        }

        var found = Set<Int>()
        var outsideCount = 0, containedCount = 0, primitiveChecks = 0
        let stopped = try bvh.shapecast(classifyBounds: { node, score in
            #expect(score == nil)
            let bounds = node.bounds
            guard bounds.intersects(sphere) else {
                outsideCount += 1
                return .outside
            }
            let far = simd_max(simd_abs(bounds.minimum - sphere.center), simd_abs(bounds.maximum - sphere.center))
            if simd_length_squared(far) <= sphere.radius * sphere.radius {
                containedCount += 1
                return .contained
            }
            return .intersecting
        }, visitTriangle: { triangle, index, _, contained in
            if !contained { primitiveChecks += 1 }
            if try contained || overlaps(triangle) {
                #expect(found.insert(index).inserted)
            }
            return false
        })
        #expect(!stopped)
        #expect(found == expected)
        #expect(found.count == 80)
        #expect(found.reduce(0, +) == 19_432)
        #expect(outsideCount > 0)
        #expect(containedCount > 0)
        #expect(primitiveChecks < bvh.triangleCount / 3)
    }

    @Test func `A contained subtree gives one range and can stop in either callback`() throws {
        let bvh = try MeshBVH(geometry: .plane(width: 8, height: 8, columns: 6, rows: 6))
        #expect(bvh.depth > 1)
        var visitedNodes = 0
        var savedRange: MeshBVH.ShapeCastRange?
        var visitedTriangles: [Int] = []
        let stopped = try bvh.shapecast(classifyBounds: { node, _ in
            visitedNodes += 1
            #expect(node.depth == 1)
            #expect(node.index == 0)
            #expect(node.bounds == bvh.bounds)
            return .contained
        }, visitRange: { node, range, contained in
            #expect(!node.isLeaf)
            #expect(contained)
            #expect(range.startIndex == 0)
            #expect(range.endIndex == bvh.triangleCount)
            #expect(range.sorted() == Array(0..<bvh.triangleCount))
            savedRange = range
            return false
        }, visitTriangle: { triangle, index, node, contained in
            #expect(node.index == 0)
            #expect(contained)
            let expected = bvh.geometry.vertices[Int(bvh.geometry.indices[index * 3])].position
            #expect(triangle.a == expected)
            visitedTriangles.append(index)
            return visitedTriangles.count == 3
        })
        #expect(stopped)
        #expect(visitedNodes == 1)
        let range = try #require(savedRange)
        #expect(visitedTriangles == Array(range.prefix(3)))

        #expect(try bvh.shapecast(classifyBounds: { _, _ in .contained }, visitRange: { _, _, _ in true },
                                 visitTriangle: { _, _, _, _ in
            Issue.record("Stopping a range must skip its triangle callbacks")
            return true
        }))
    }

    @Test func `Sibling scores control order while ties keep the original tree order`() throws {
        let right = SIMD3<Float>(2, 0, 0), left = SIMD3<Float>(-2, 0, 0)
        let vertices = [right, left].map { MeshVertex(position: $0, normal: SIMD3(0, 0, 1)) }
        let bvh = try MeshBVH(geometry: MeshGeometry(vertices: vertices,
                                                    indices: [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]))
        #expect(bvh.depth == 2)
        func first(_ order: ((MeshBVH.ShapeCastNode) -> Double)?) throws -> (Float, Double?) {
            var x = Float.nan
            var score: Double?
            let stopped = try bvh.shapecast(orderBounds: order, classifyBounds: { node, value in
                if node.isLeaf { score = value }
                return .intersecting
            }, visitTriangle: { triangle, _, node, contained in
                #expect(node.depth == 2)
                #expect(node.isLeaf)
                #expect(!contained)
                x = triangle.a.x
                return true
            })
            #expect(stopped)
            return (x, score)
        }
        let original = try first(nil)
        #expect(original.0 == -2)
        #expect(original.1 == nil)
        #expect(try first({ _ in 2 }).0 == -2)
        let reordered = try first { $0.bounds.center.x < 0 ? .nan : -Double($0.bounds.center.x) }
        #expect(reordered.0 == 2)
        #expect(reordered.1 == -2)
    }

    @Test func `A single leaf can be rejected and callback failures are propagated`() throws {
        let bvh = try MeshBVH(geometry: .plane(width: 2, height: 2))
        #expect(bvh.nodeCount == 1)
        var visited = 0
        let stopped = try bvh.shapecast(classifyBounds: { node, _ in
            #expect(node.isLeaf)
            visited += 1
            return .outside
        }, visitRange: { _, _, _ in
            Issue.record("Rejected root must skip its range")
            return true
        })
        #expect(!stopped)
        #expect(visited == 1)

        enum CallbackError: Error { case score, classify, range, triangle }
        #expect(throws: CallbackError.score) {
            try bvh.shapecast(orderBounds: { _ in throw CallbackError.score }, classifyBounds: { _, _ in .contained })
        }
        #expect(throws: CallbackError.classify) {
            try bvh.shapecast(classifyBounds: { _, _ in throw CallbackError.classify })
        }
        #expect(throws: CallbackError.range) {
            try bvh.shapecast(classifyBounds: { _, _ in .contained }, visitRange: { _, _, _ in throw CallbackError.range })
        }
        #expect(throws: CallbackError.triangle) {
            try bvh.shapecast(classifyBounds: { _, _ in .contained },
                              visitTriangle: { _, _, _, _ in throw CallbackError.triangle })
        }
    }

    @Test func `Node indices and triangle ranges are stable across refits`() throws {
        let geometry = try MeshGeometry.plane(width: 4, height: 4, columns: 4, rows: 4)
        let original = try MeshBVH(geometry: geometry)
        let vertices = geometry.vertices.map { MeshVertex(position: $0.position * 4, normal: $0.normal) }
        let moved = try original.refitted(to: MeshGeometry(vertices: vertices, indices: geometry.indices))
        func ranges(_ bvh: MeshBVH) throws -> [Int: [Int]] {
            var ranges: [Int: [Int]] = [:]
            _ = try bvh.shapecast(classifyBounds: { _, _ in .intersecting }, visitRange: { node, range, contained in
                #expect(node.isLeaf)
                #expect(!contained)
                ranges[node.index] = Array(range)
                return false
            })
            return ranges
        }
        #expect(try ranges(moved) == ranges(original))
    }
}
