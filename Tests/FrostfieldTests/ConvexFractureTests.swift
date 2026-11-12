import Testing
import simd
@testable import Frostfield

struct ConvexFractureTests {
    @Test func planeCutHalvesACubeAndImpactKeepsBothPieces() throws {
        let fracture = try ConvexFracture()
        let body = ConvexFracture.Body(
            geometry: MeshGeometry.cube(), mass: 2, velocity: SIMD3(1, 0, 0), angularVelocity: SIMD3(0, 2, 0)
        )
        let plane = try ConvexFracture.Plane(normal: SIMD3(0, 0, 1), through: .zero)
        let pieces = try fracture.cut(body, by: plane)
        let negative = try #require(pieces.negative)
        let positive = try #require(pieces.positive)
        #expect(abs(negative.position.z + 0.375) < 0.02)
        #expect(abs(positive.position.z - 0.375) < 0.02)
        #expect(negative.mass == 1 && positive.mass == 1)
        #expect(negative.velocity == body.velocity && positive.angularVelocity == body.angularVelocity)
        #expect(negative.breakable == false && positive.breakable == false)
        #expect(abs(try ConvexHull(points: negative.geometry.vertices.map(\.position)).volume - 0.5) < 0.05)
        #expect(abs(try ConvexHull(points: positive.geometry.vertices.map(\.position)).volume - 0.5) < 0.05)

        let moved = ConvexFracture.Body(geometry: MeshGeometry.cube(), position: SIMD3(0, 0, 3), mass: 2)
        let split = try fracture.cut(moved, by: try ConvexFracture.Plane(normal: SIMD3(0, 0, 1), through: SIMD3(0, 0, 3)))
        #expect(abs((try #require(split.positive).position.z) - 3.375) < 0.02)

        let missed = try fracture.cut(body, by: try ConvexFracture.Plane(normal: SIMD3(0, 0, 1), through: SIMD3(0, 0, 2)))
        #expect(missed.positive == nil)
        #expect(try #require(missed.negative).mass == 1)

        let debris = try fracture.subdivideByImpact(
            body, at: SIMD3(0.5, 0.5, 0.5), normal: SIMD3(0, 0, 1), random: { 1 }
        )
        #expect(debris.count == 2)
        #expect(debris.allSatisfy { $0.mass == 1 })
    }
}
