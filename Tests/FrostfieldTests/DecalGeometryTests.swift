import Foundation
import Frostfield
import Testing
import simd

struct DecalGeometryTests {
    /// Projector-space coordinates of a world point.
    private func local(_ point: SIMD3<Float>, _ projector: DecalProjector) -> SIMD3<Float> {
        projector.orientation.inverse.act(point - projector.position)
    }

    @Test("A rotated projector over a plane cuts exactly its footprint, with UVs mapping the box to 0...1")
    func planeFootprint() throws {
        let plane = try MeshGeometry.plane(width: 4, height: 4, columns: 9, rows: 7)
        let spin = simd_quatf(angle: .pi / 6, axis: SIMD3(0, 0, 1))
        let projector = DecalProjector(position: SIMD3(0.3, -0.2, 0.5), orientation: spin, size: SIMD3(1, 0.5, 2))
        let decal = try #require(try plane.decal(projector: projector))
        // The footprint is the box's cross-section: width × height.
        #expect(abs(ShapeCheck.surfaceArea(decal) - 0.5) < 1e-5)
        let right = spin.act(SIMD3<Float>(1, 0, 0))
        for vertex in decal.vertices {
            let p = local(vertex.position, projector)
            #expect(abs(p.x) <= 0.5 + 1e-5 && abs(p.y) <= 0.25 + 1e-5)
            let expected = SIMD2(p.x / 1 + 0.5, 0.5 - p.y / 0.5)
            #expect(simd_distance(vertex.textureCoordinate, expected) < 1e-5)
            #expect(vertex.normal == SIMD3(0, 0, 1))
            let tangent = try #require(vertex.tangent)
            #expect(simd_distance(SIMD3(tangent.x, tangent.y, tangent.z), right) < 1e-5)
        }
        // All four box corners appear as UV corners.
        for corner: SIMD2<Float> in [SIMD2(0, 0), SIMD2(1, 0), SIMD2(0, 1), SIMD2(1, 1)] {
            #expect(decal.vertices.contains { simd_distance($0.textureCoordinate, corner) < 1e-5 })
        }
        #expect(ShapeCheck.tangentsFollowTextureCoordinates(decal))
    }

    @Test("On a sphere the decal keeps front faces inside the box, and a hierarchy gives the same result")
    func sphereDecal() throws {
        let sphere = try MeshGeometry.uvSphere(radius: 1, segments: 48, rings: 24)
        let world = SceneTransform(translation: SIMD3(0, 0, -3), scale: SIMD3(repeating: 2)).matrix
        let projector = try DecalProjector(
            position: SIMD3(0.4, 0.3, 0), direction: SIMD3(0, 0, -1), angle: 0.4, size: SIMD3(1.2, 0.8, 10)
        )
        let decal = try #require(try sphere.decal(projector: projector, localToWorld: world))
        let axis = projector.orientation.act(SIMD3<Float>(0, 0, 1))
        for triangle in 0..<(decal.indices.count / 3) {
            let (a, b, c) = ShapeCheck.corners(decal, triangle)
            let normal = simd_normalize(simd_cross(b - a, c - a))
            // Only faces turned toward the projector receive the decal.
            #expect(simd_dot(normal, SIMD3<Double>(axis)) > 0)
        }
        for vertex in decal.vertices {
            let p = local(vertex.position, projector)
            #expect(abs(p.x) <= 0.6 + 1e-4 && abs(p.y) <= 0.4 + 1e-4)
            // Vertices stay on the scaled sphere's facets, within its radius of 2.
            let radius = simd_distance(vertex.position, SIMD3(0, 0, -3))
            #expect(radius <= 2 + 1e-4 && radius > 1.98)
            #expect(vertex.textureCoordinate.x >= -1e-5 && vertex.textureCoordinate.x <= 1 + 1e-5)
        }
        // A near-perpendicular projection onto a sphere covers more area than the footprint.
        #expect(ShapeCheck.surfaceArea(decal) > 0.96)

        let bvh = try MeshBVH(geometry: sphere)
        let accelerated = try #require(try sphere.decal(projector: projector, localToWorld: world, bvh: bvh))
        #expect(accelerated.indices.count == decal.indices.count)
        #expect(Set(accelerated.vertices.map(\.position)) == Set(decal.vertices.map(\.position)))

        // Allowing every angle also projects through to the back of the sphere.
        let through = try #require(try sphere.decal(projector: projector, localToWorld: world, maximumAngle: .pi))
        #expect(ShapeCheck.surfaceArea(through) > 1.9 * ShapeCheck.surfaceArea(decal))

        // A surface offset lifts every vertex along its normal.
        let lifted = try #require(try sphere.decal(projector: projector, localToWorld: world, surfaceOffset: 0.01))
        #expect(lifted.vertices.allSatisfy { simd_distance($0.position, SIMD3(0, 0, -3)) > 1.99 })
    }

    @Test("A projector that misses returns nil, and invalid projectors throw")
    func missesAndValidation() throws {
        let plane = try MeshGeometry.plane()
        #expect(try plane.decal(projector: DecalProjector(position: SIMD3(5, 5, 0), size: SIMD3(1, 1, 1))) == nil)
        // Facing away from the plane's front: nothing within the default 90°.
        let behind = try DecalProjector(position: .zero, direction: SIMD3(0, 0, 1), size: SIMD3(1, 1, 1))
        #expect(try plane.decal(projector: behind) == nil)
        #expect(throws: MeshGeometry.DecalError.invalidProjector) {
            try plane.decal(projector: DecalProjector(position: .zero, size: SIMD3(1, 0, 1)))
        }
        #expect(throws: MeshGeometry.DecalError.invalidAngle) {
            try plane.decal(projector: DecalProjector(position: .zero, size: SIMD3(1, 1, 1)), maximumAngle: 0)
        }
        #expect(throws: MeshGeometry.DecalError.invalidProjector) {
            try DecalProjector(position: .zero, direction: .zero, size: SIMD3(1, 1, 1))
        }
    }
}
