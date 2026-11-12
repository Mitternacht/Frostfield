import Frostfield
import Testing
import simd

struct MeshDepenetrationTests {
    @Test("Capsule and sphere resolve both planes of a corner")
    func corner() throws {
        let corner = try cornerMesh()
        let capsule = try Capsule(start: SIMD3(0.2, 0.3, 1.7), end: SIMD3(0.2, 1.1, 1.7), radius: 0.5)
        let resolved = try #require(try corner.depenetration(for: capsule))
        #expect(resolved.isResolved && resolved.iterations == 2)
        #expect(simd_distance(resolved.translation, SIMD3(0.3, 0.2, 0)) < 1e-6)
        #expect(try corner.depenetration(for: capsule.translated(by: resolved.translation)) == nil)

        let sphere = try BoundingSphere(center: SIMD3(0.125, 0.125, 1.2), radius: 0.5)
        let partial = try #require(try corner.depenetration(for: sphere, maximumIterations: 1))
        #expect(!partial.isResolved && partial.iterations == 1)
        #expect(partial.translation == SIMD3(0.375, 0, 0))
        let sphereResolved = try #require(try corner.depenetration(for: sphere))
        #expect(sphereResolved.isResolved && sphereResolved.iterations == 2)
        #expect(sphereResolved.translation == SIMD3(0.375, 0.375, 0))
        let moved = try BoundingSphere(center: sphere.center + sphereResolved.translation, radius: sphere.radius)
        #expect(try corner.contacts(intersecting: moved).contains { $0.depth == 0 })
        #expect(try corner.depenetration(for: moved) == nil)
    }

    @Test("Axis-aligned and oriented boxes resolve both planes of a corner")
    func boxCorner() throws {
        let corner = try cornerMesh()
        let box = try AxisAlignedBox(minimum: SIMD3(-0.25, -0.25, 2), maximum: SIMD3(0.75, 0.5, 2.5))
        let partial = try #require(try corner.depenetration(for: box, maximumIterations: 1))
        #expect(!partial.isResolved && partial.iterations == 1)
        #expect(partial.translation == SIMD3(0.25, 0, 0))
        let resolved = try #require(try corner.depenetration(for: box))
        #expect(resolved.isResolved && resolved.iterations == 2)
        #expect(resolved.translation == SIMD3(0.25, 0.25, 0))
        let moved = try box.translated(by: resolved.translation)
        #expect(try corner.contacts(intersecting: moved).contains { $0.depth == 0 })
        #expect(try corner.depenetration(for: moved) == nil)

        let angle: Float = .pi / 6
        let oriented = try OrientedBoundingBox(
            center: SIMD3(0.25, 0.25, 2.3), halfSize: SIMD3(0.5, 0.4, 0.3),
            rotation: simd_quatf(angle: angle, axis: SIMD3(0, 0, 1))
        )
        let orientedResolution = try #require(try corner.depenetration(for: oriented))
        #expect(orientedResolution.isResolved && orientedResolution.iterations == 2)
        let expected = SIMD3<Float>(
            0.5 * cos(angle) + 0.4 * sin(angle) - 0.25,
            0.5 * sin(angle) + 0.4 * cos(angle) - 0.25,
            0
        )
        #expect(simd_distance(orientedResolution.translation, expected) < 1e-6)
        let movedOriented = try oriented.translated(by: orientedResolution.translation)
        #expect(movedOriented.rotation == oriented.rotation)
        #expect(movedOriented.size == SIMD3(1, 0.8, 0.6))
        #expect(try corner.depenetration(for: movedOriented) == nil)
    }

    @Test("Opposing walls report unresolved penetration at the iteration limit")
    func opposingWalls() throws {
        let plane = try MeshGeometry.plane(width: 20, height: 20)
        let corridor = try MeshBVH(geometry: MeshGeometry.merged([
            .init(plane, transform: wall(at: 0)), .init(plane, transform: wall(at: 0.5)),
        ]).geometry)
        let capsule = try Capsule(start: SIMD3(0.25, 1, 2), end: SIMD3(0.25, 2, 2), radius: 0.4)
        let response = try #require(try corridor.depenetration(for: capsule, maximumIterations: 5))
        #expect(!response.isResolved && response.iterations == 5)
        #expect(simd_distance(response.translation, SIMD3(0.15, 0, 0)) < 1e-6)
        let displaced = try capsule.translated(by: response.translation)
        #expect(try corridor.contacts(intersecting: displaced).contains { $0.depth > 0.25 })

        let sphere = try BoundingSphere(center: capsule.center, radius: capsule.radius)
        let sphereResponse = try #require(try corridor.depenetration(for: sphere, maximumIterations: 4))
        #expect(!sphereResponse.isResolved && sphereResponse.iterations == 4)
        #expect(simd_distance(sphereResponse.translation, SIMD3(-0.15, 0, 0)) < 1e-6)

        let box = try AxisAlignedBox(minimum: SIMD3(-0.125, 1, 2), maximum: SIMD3(0.625, 2, 3))
        let boxResponse = try #require(try corridor.depenetration(for: box, maximumIterations: 3))
        #expect(!boxResponse.isResolved && boxResponse.iterations == 3)
        #expect(boxResponse.translation == SIMD3(0.125, 0, 0))
    }

    @Test("A capsule move rests on the floor and keeps its slide")
    func characterMove() throws {
        let bvh = try MeshBVH(geometry: try MeshGeometry.plane(width: 20, height: 20))
        let standing = try Capsule(start: SIMD3(0, 0, 1), end: SIMD3(0, 0, 2), radius: 0.5)
        let dropped = try bvh.moved(standing, by: SIMD3(0.4, 0, -0.7))
        #expect(abs(dropped.start.x - 0.4) < 1e-3)
        #expect(abs(dropped.start.y) < 1e-4)
        #expect(abs(dropped.start.z - 0.5) < 1e-3)
        #expect(abs(dropped.end.z - 1.5) < 1e-3)

        let clear = try Capsule(start: SIMD3(0, 0, 3), end: SIMD3(0, 0, 4), radius: 0.5)
        let walked = try bvh.moved(clear, by: SIMD3(0.25, 0, 0))
        let free = try clear.translated(by: SIMD3(0.25, 0, 0))
        #expect(walked == free)
        #expect(throws: MeshCollisionResolution.Failure.invalidDisplacement) {
            try bvh.moved(clear, by: SIMD3(.nan, 0, 0))
        }
    }

    @Test("Invalid resolution settings fail before querying the mesh")
    func settings() throws {
        let bvh = try MeshBVH(geometry: .plane())
        let sphere = try BoundingSphere(center: .zero, radius: 1)
        #expect(throws: MeshCollisionResolution.Failure.invalidSettings) {
            try bvh.depenetration(for: sphere, maximumIterations: 0)
        }
        #expect(throws: MeshCollisionResolution.Failure.invalidSettings) {
            try bvh.depenetration(for: sphere, maximumIterations: 65)
        }
        #expect(throws: MeshCollisionResolution.Failure.invalidSettings) {
            try bvh.depenetration(for: sphere, tolerance: .nan)
        }
    }

    @Test("Depenetration reports when a Float position cannot represent the movement")
    func unrepresentableStep() throws {
        let plane = try MeshGeometry.plane(width: 20, height: 20)
        let x: Float = 100_000_000
        let bvh = try MeshBVH(geometry: MeshGeometry.merged([.init(plane, transform: wall(at: x))]).geometry)
        let sphere = try BoundingSphere(center: SIMD3(x, 1, 2), radius: 0.1)
        let response = try #require(try bvh.depenetration(for: sphere))
        #expect(!response.isResolved && response.iterations == 0)
        #expect(response.translation == .zero)
    }

    private func wall(at x: Float) -> simd_float4x4 {
        simd_float4x4(columns: (
            SIMD4(0, 0, -1, 0), SIMD4(0, 1, 0, 0), SIMD4(1, 0, 0, 0), SIMD4(x, 0, 0, 1)
        ))
    }

    private func cornerMesh() throws -> MeshBVH {
        let plane = try MeshGeometry.plane(width: 20, height: 20)
        let floor = simd_float4x4(columns: (
            SIMD4(1, 0, 0, 0), SIMD4(0, 0, -1, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, 0, 1)
        ))
        return try MeshBVH(geometry: MeshGeometry.merged([
            .init(plane, transform: wall(at: 0)), .init(plane, transform: floor),
        ]).geometry)
    }
}
