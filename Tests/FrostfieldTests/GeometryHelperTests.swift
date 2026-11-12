import Frostfield
import Testing
import simd

struct GeometryHelperTests {
    @Test("An outline shell moves along the normal and turns its front face inward")
    func outlineShell() throws {
        let plane = try MeshGeometry.plane(width: 2, height: 2)
        let shell = try plane.outlineShell(thickness: 0.25)
        #expect(shell.vertices.count == plane.vertices.count)
        #expect(shell.indices == [0, 1, 2, 1, 3, 2])
        for (source, expanded) in zip(plane.vertices, shell.vertices) {
            #expect(expanded.position == source.position + SIMD3(0, 0, 0.25))
            #expect(expanded.normal == SIMD3(0, 0, -1))
        }
        #expect(throws: MeshGeometry.ValidationError.invalidOutlineThickness) {
            try plane.outlineShell(thickness: 0)
        }
    }

    @Test("A position trail is a ribbon of the requested width")
    func positionTrail() throws {
        var trail = try PositionTrail(width: 0.2, capacity: 4)
        #expect(try trail.ribbon() == nil)
        try trail.append(.zero)
        try trail.append(.zero)
        #expect(trail.positions.count == 1)
        try trail.append(SIMD3(1, 0, 0))
        let ribbon = try #require(try trail.ribbon())
        #expect(ribbon.vertices[0].position == SIMD3(0, 0, -0.1))
        #expect(ribbon.vertices[1].position == SIMD3(0, 0, 0.1))
        #expect(ribbon.vertices[2].position == SIMD3(1, 0, -0.1))
        #expect(ribbon.vertices[3].position == SIMD3(1, 0, 0.1))
        #expect(ribbon.vertices[0].normal == SIMD3(0, -1, 0))
        try trail.append(SIMD3(2, 0, 0))
        try trail.append(SIMD3(3, 0, 0))
        #expect(throws: PositionTrail.Failure.capacity) {
            try trail.append(SIMD3(4, 0, 0))
        }
        #expect(throws: PositionTrail.Failure.invalidPosition) {
            try trail.append(SIMD3(.nan, 0, 0))
        }
    }
}
