import CoreText
import Foundation
import Frostfield
import Testing
import simd

struct MeshBooleanTests {
    private func shifted(_ x: Float, _ y: Float = 0, _ z: Float = 0) -> simd_float4x4 {
        SceneTransform(translation: SIMD3(x, y, z)).matrix
    }

    private func evaluate(
        _ operation: MeshBooleanOperation, _ a: MeshGeometry, _ b: MeshGeometry, transform: simd_float4x4 = matrix_identity_float4x4
    ) throws -> MeshGeometry {
        try MeshBoolean.evaluate(operation, MeshBooleanOperand(a), MeshBooleanOperand(b, transform: transform))
    }

    @Test("Overlapping unit cubes give exact union, intersection and difference volumes as closed meshes")
    func overlappingCubes() throws {
        let cube = MeshGeometry.cube()
        let offset = shifted(0.5, 0.5, 0.5)
        let expected: [(MeshBooleanOperation, Double)] = [(.union, 1.875), (.intersection, 0.125), (.subtraction, 0.875)]
        for (operation, volume) in expected {
            let result = try evaluate(operation, cube, cube, transform: offset)
            #expect(ShapeCheck.isClosed(result))
            #expect(abs(result.signedVolume - volume) < 1e-6)
            let bounds = ShapeCheck.bounds(result.vertices.map(\.position))
            switch operation {
            case .union: #expect(bounds.lower == SIMD3(repeating: -0.5) && bounds.upper == SIMD3(repeating: 1))
            case .intersection: #expect(bounds.lower == SIMD3(repeating: 0) && bounds.upper == SIMD3(repeating: 0.5))
            case .subtraction: #expect(bounds.lower == SIMD3(repeating: -0.5) && bounds.upper == SIMD3(repeating: 0.5))
            }
        }
    }

    @Test("Coplanar faces are classified as same or opposite: shared planes merge without duplicate faces")
    func coplanarCubes() throws {
        let cube = MeshGeometry.cube()
        // Shifted along X only, the top, bottom, front and back faces are coplanar.
        let half = shifted(0.5)
        for (operation, volume) in [(MeshBooleanOperation.union, 1.5), (.intersection, 0.5), (.subtraction, 0.5)] {
            let result = try evaluate(operation, cube, cube, transform: half)
            #expect(ShapeCheck.isClosed(result))
            #expect(abs(result.signedVolume - volume) < 1e-6)
            // Coplanar regions appear once: total area equals the exact box surface.
            let size: SIMD3<Double> = operation == .union ? SIMD3(1.5, 1, 1) : SIMD3(0.5, 1, 1)
            let area = 2 * (size.x * size.y + size.y * size.z + size.z * size.x)
            #expect(abs(ShapeCheck.surfaceArea(result) - area) < 1e-6)
        }
        // Cubes touching face to face merge into one box; their intersection is empty.
        let touching = shifted(1)
        let merged = try evaluate(.union, cube, cube, transform: touching)
        #expect(ShapeCheck.isClosed(merged))
        #expect(abs(merged.signedVolume - 2) < 1e-6)
        #expect(abs(ShapeCheck.surfaceArea(merged) - 10) < 1e-6)
        #expect(abs(try evaluate(.subtraction, cube, cube, transform: touching).signedVolume - 1) < 1e-6)
        #expect(throws: MeshBoolean.Failure.emptyResult) {
            try evaluate(.intersection, cube, cube, transform: touching)
        }
        // Identical cubes: union and intersection are the cube, the difference is empty.
        #expect(abs(try evaluate(.union, cube, cube).signedVolume - 1) < 1e-6)
        #expect(abs(try evaluate(.intersection, cube, cube).signedVolume - 1) < 1e-6)
        #expect(throws: MeshBoolean.Failure.emptyResult) { try evaluate(.subtraction, cube, cube) }
    }

    @Test("A cube intersected with its copy turned 45° about Z is a regular octagonal prism of volume 2(√2 − 1)")
    func rotatedOperand() throws {
        let cube = MeshGeometry.cube()
        let turned = SceneTransform(rotation: simd_quatf(angle: .pi / 4, axis: SIMD3(0, 0, 1))).matrix
        let result = try evaluate(.intersection, cube, cube, transform: turned)
        #expect(ShapeCheck.isClosed(result))
        #expect(abs(result.signedVolume - 2 * (2.squareRoot() - 1)) < 1e-6)
        // Tilted on two axes there are no coplanar faces; inclusion–exclusion must still hold.
        let tilted = SceneTransform(
            translation: SIMD3(0.2, 0.1, 0.3), rotation: simd_quatf(angle: 0.6, axis: simd_normalize(SIMD3(1, 2, 0.5)))
        ).matrix
        let union = try evaluate(.union, cube, cube, transform: tilted)
        let intersection = try evaluate(.intersection, cube, cube, transform: tilted)
        let difference = try evaluate(.subtraction, cube, cube, transform: tilted)
        for mesh in [union, intersection, difference] { #expect(ShapeCheck.isClosed(mesh)) }
        #expect(abs(union.signedVolume + intersection.signedVolume - 2) < 1e-6)
        #expect(abs(difference.signedVolume + intersection.signedVolume - 1) < 1e-6)
    }

    @Test("Random rigid placements of cubes, spheres, rounded boxes and cylinders keep inclusion–exclusion and closed results",
          arguments: [21, 22, 23] as [UInt64])
    func randomPlacements(seed: UInt64) throws {
        var random = ShapeTestRandom(seed: seed)
        let cube = MeshGeometry.cube()
        let sphere = try MeshGeometry.uvSphere(radius: 0.6, segments: 24, rings: 12)
        let rounded = try MeshGeometry.bevelBox(size: SIMD3(0.8, 0.6, 1.1), bevel: 0.1, segments: 3)
        let cylinder = try MeshGeometry.lathe(
            profile: [LatheProfilePoint(radius: 0.3, y: -0.5), LatheProfilePoint(radius: 0.3, y: 0.5, hard: true)], segments: 20
        )
        for (a, b) in [(cube, cube), (cube, sphere), (sphere, sphere), (rounded, cylinder)] {
            let axis = simd_normalize(SIMD3<Float>(
                Float.random(in: -1...1, using: &random), Float.random(in: -1...1, using: &random), Float.random(in: 0.1...1, using: &random)
            ))
            let placement = SceneTransform(
                translation: SIMD3(Float.random(in: -0.5...0.5, using: &random), Float.random(in: -0.5...0.5, using: &random), Float.random(in: -0.5...0.5, using: &random)),
                rotation: simd_quatf(angle: Float.random(in: 0...3, using: &random), axis: axis)
            ).matrix
            var volumes: [Double] = []
            for operation in [MeshBooleanOperation.union, .intersection, .subtraction] {
                do {
                    let result = try evaluate(operation, a, b, transform: placement)
                    #expect(ShapeCheck.isClosed(result))
                    volumes.append(result.signedVolume)
                } catch MeshBoolean.Failure.emptyResult {
                    volumes.append(0)
                }
            }
            #expect(abs(volumes[0] + volumes[1] - a.signedVolume - b.signedVolume) < 1e-5)
            #expect(abs(volumes[2] + volumes[1] - a.signedVolume) < 1e-5)
        }
    }

    @Test("Surfaces that coincide only to Float rounding still merge as coplanar")
    func floatCoincidence() throws {
        // Turned by exactly one segment, the sphere's facets land on themselves up to Float rounding.
        let sphere = try MeshGeometry.uvSphere(radius: 0.6, segments: 24, rings: 12)
        let turned = SceneTransform(rotation: simd_quatf(angle: 2 * .pi / 24, axis: SIMD3(0, 1, 0))).matrix
        for operation in [MeshBooleanOperation.union, .intersection] {
            let result = try evaluate(operation, sphere, sphere, transform: turned)
            #expect(ShapeCheck.isClosed(result))
            #expect(abs(result.signedVolume - sphere.signedVolume) < 1e-5)
        }
        #expect(throws: MeshBoolean.Failure.emptyResult) { try evaluate(.subtraction, sphere, sphere, transform: turned) }
        // Cubes sharing only an edge would make a non-manifold union, which is reported.
        #expect(throws: MeshBoolean.Failure.nonManifoldResult) {
            try evaluate(.union, MeshGeometry.cube(), MeshGeometry.cube(), transform: shifted(1, 1))
        }
    }

    @Test("Cube and sphere: volumes satisfy inclusion–exclusion and match the analytic ball–cube intersection")
    func cubeAndSphere() throws {
        let cube = MeshGeometry.cube()
        let radius = 0.65
        let sphere = try MeshGeometry.uvSphere(radius: Float(radius), segments: 96, rings: 48)
        let union = try evaluate(.union, cube, sphere)
        let intersection = try evaluate(.intersection, cube, sphere)
        let difference = try evaluate(.subtraction, cube, sphere)
        let carved = try evaluate(.subtraction, sphere, cube)
        for mesh in [union, intersection, difference, carved] { #expect(ShapeCheck.isClosed(mesh)) }
        let a = cube.signedVolume, b = sphere.signedVolume
        #expect(abs(union.signedVolume + intersection.signedVolume - (a + b)) < 1e-5)
        #expect(abs(difference.signedVolume - (a - intersection.signedVolume)) < 1e-5)
        #expect(abs(carved.signedVolume - (b - intersection.signedVolume)) < 1e-5)

        // The ball pokes through all six faces: its volume less six caps of height R − 0.5.
        let h = radius - 0.5
        let analytic = 4.0 / 3 * .pi * pow(radius, 3) - 6 * .pi * h * h * (3 * radius - h) / 3
        // The faceted sphere is inscribed in the ball; compare within its own volume deficit.
        let deficit = 4.0 / 3 * .pi * pow(radius, 3) - b
        #expect(intersection.signedVolume < analytic)
        #expect(analytic - intersection.signedVolume < deficit)
        // Every result vertex lies on the cube or on the sphere's facets.
        for vertex in difference.vertices {
            let p = SIMD3<Double>(vertex.position)
            let onCube = abs(simd_reduce_max(simd_abs(p)) - 0.5) < 1e-6
            let r = simd_length(p)
            #expect(onCube || (r <= radius + 1e-6 && r > radius * 0.999))
        }
    }

    @Test("Split faces interpolate UVs, normals and tangents from their source triangles; subtracted faces flip")
    func attributeInterpolation() throws {
        let box = try MeshGeometry.bevelBox(bevel: 0)
        let offset = shifted(0.5, 0.5, 0.5)
        let difference = try evaluate(.subtraction, box, box, transform: offset)
        #expect(ShapeCheck.isClosed(difference))
        #expect(ShapeCheck.tangentsFollowTextureCoordinates(difference))
        // The cavity wall at x = 0 comes from the second box's −X face, turned to face +X.
        // That face maps local (z, y) to UV (0.5 + z, 0.5 − y), and local = world − 0.5.
        let cavity = difference.vertices.filter { $0.position.x == 0 && $0.normal.x > 0.5 }
        // The wall is the square y, z ∈ [0, 0.5] at x = 0.
        #expect(Set(cavity.map(\.position)).count >= 4)
        for vertex in cavity {
            #expect(vertex.normal == SIMD3(1, 0, 0))
            #expect(vertex.position.y >= 0 && vertex.position.y <= 0.5 && vertex.position.z >= 0 && vertex.position.z <= 0.5)
            let expected = SIMD2(0.5 + (vertex.position.z - 0.5), 0.5 - (vertex.position.y - 0.5))
            #expect(simd_distance(vertex.textureCoordinate, expected) < 1e-6)
        }
        // Faces kept from the first box keep its outward normals and UVs at split points.
        let top = difference.vertices.filter { $0.position.y == 0.5 && $0.normal.y == 1 }
        for vertex in top {
            // bevelBox's +Y face maps (x, z) to UV (0.5 + x, 0.5 + z).
            #expect(simd_distance(vertex.textureCoordinate, SIMD2(0.5 + vertex.position.x, 0.5 + vertex.position.z)) < 1e-6)
        }
        #expect(difference.vertices.allSatisfy { abs(simd_length($0.normal) - 1) < 1e-5 })
    }

    @Test("Text engraved into a plaque leaves a closed solid whose volume is the plaque less the letters")
    func engravedText() throws {
        let font = CTFontCreateWithName("Helvetica" as CFString, 72, nil)
        let letters = try MeshGeometry.text("Ag", font: font, size: 1, extrusion: ShapeExtrusion(depth: 0.4))
        let plaque = try MeshGeometry.bevelBox(size: SIMD3(2, 1.4, 0.5), bevel: 0)
        // The letters' back half sinks into the plaque's front face at z = 0.25.
        let placement = shifted(-0.6, -0.3, 0.25)
        let engraved = try evaluate(.subtraction, plaque, letters, transform: placement)
        let embossed = try evaluate(.union, plaque, letters, transform: placement)
        #expect(ShapeCheck.isClosed(engraved))
        #expect(ShapeCheck.isClosed(embossed))
        let letterVolume = letters.signedVolume
        #expect(abs(engraved.signedVolume - (1.4 - letterVolume / 2)) < 1e-5)
        #expect(abs(embossed.signedVolume - (1.4 + letterVolume / 2)) < 1e-5)
    }

    @Test("Open, non-manifold, inverted and degenerate operands are rejected before any splitting")
    func validation() throws {
        let cube = MeshGeometry.cube()
        let plane = try MeshGeometry.plane()
        #expect(throws: MeshBoolean.Failure.openBoundary(operand: 1)) { try cube.union(plane) }
        let inverted = try MeshGeometry(
            vertices: cube.vertices, indices: stride(from: 0, to: cube.indices.count, by: 3).flatMap {
                [cube.indices[$0], cube.indices[$0 + 2], cube.indices[$0 + 1]]
            }
        )
        #expect(throws: MeshBoolean.Failure.invertedOrientation(operand: 0)) { try inverted.union(cube) }
        let doubled = try MeshGeometry(vertices: cube.vertices, indices: cube.indices + cube.indices)
        #expect(throws: MeshBoolean.Failure.nonManifoldEdge(operand: 1)) { try cube.intersection(doubled) }
        // A repeated index collapses a triangle, which leaves the cube open.
        let collapsed = try MeshGeometry(vertices: cube.vertices, indices: cube.indices.map { $0 == 1 ? 0 : $0 })
        #expect(throws: MeshBoolean.Failure.openBoundary(operand: 0)) { try collapsed.subtracting(cube) }
        // A crack narrower than the tolerance welds shut.
        var cracked = cube.vertices
        cracked[0] = MeshVertex(position: cracked[0].position + SIMD3(1e-7, -1e-7, 0), normal: cracked[0].normal)
        let healed = try MeshGeometry(vertices: cracked, indices: cube.indices).union(cube)
        #expect(ShapeCheck.isClosed(healed))
        #expect(abs(healed.signedVolume - 1) < 1e-6)
        var singular = matrix_identity_float4x4
        singular.columns.0 = .zero
        #expect(throws: MeshBoolean.Failure.invalidTransform(operand: 1)) {
            try evaluate(.union, cube, cube, transform: singular)
        }
    }
}
