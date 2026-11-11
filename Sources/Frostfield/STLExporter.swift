import Foundation
import simd

/// Writes ASCII or binary STL with a geometric unit normal per triangle.
/// Positions retain their original units. A degenerate facet gets a zero normal.
public enum STLExporter {
    public enum Format: Equatable, Sendable {
        case ascii
        case binary
    }

    public static func data(from geometry: MeshGeometry, format: Format = .ascii, name: String = "mesh") throws -> Data {
        try data(from: [geometry], format: format, name: name)
    }

    /// Flattens all resolved triangle meshes into one STL solid. Instances,
    /// world transforms, mirrored winding, selected LODs and current poses are
    /// reflected in the output. The format carries geometry only.
    public static func data(from graph: SceneGraph, format: Format = .ascii, name: String = "scene") throws -> Data {
        let meshes = try MeshFileSnapshot.meshes(in: graph).map { $0.geometry }
        return try data(from: meshes, format: format, name: name)
    }

    private static func data(from meshes: [MeshGeometry], format: Format, name: String) throws -> Data {
        guard !meshes.isEmpty else { throw Failure.noTriangles }
        let label = MeshFileSnapshot.name(name, fallback: "mesh")
        switch format {
        case .ascii:
            var text = "solid \(label)\n"
            forEachFacet(meshes) { normal, a, b, c in
                text += "  facet normal \(normal.x) \(normal.y) \(normal.z)\n    outer loop\n"
                for point in [a, b, c] { text += "      vertex \(point.x) \(point.y) \(point.z)\n" }
                text += "    endloop\n  endfacet\n"
            }
            text += "endsolid \(label)\n"
            return Data(text.utf8)
        case .binary:
            var count: UInt32 = 0
            for mesh in meshes {
                guard let triangles = UInt32(exactly: mesh.indices.count / 3),
                      triangles <= UInt32.max - count else { throw Failure.triangleCount }
                count += triangles
            }
            var bytes = Data(("Frostfield " + label).utf8.prefix(80))
            bytes.append(Data(repeating: 0, count: 80 - bytes.count))
            bytes.reserveCapacity(84 + Int(count) * 50)
            bytes.appendLittleEndian(count)
            forEachFacet(meshes) { normal, a, b, c in
                for vector in [normal, a, b, c] {
                    bytes.appendLittleEndian(vector.x.bitPattern)
                    bytes.appendLittleEndian(vector.y.bitPattern)
                    bytes.appendLittleEndian(vector.z.bitPattern)
                }
                bytes.appendLittleEndian(UInt16.zero)
            }
            return bytes
        }
    }

    private static func forEachFacet(
        _ meshes: [MeshGeometry], _ emit: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>) -> Void
    ) {
        for mesh in meshes {
            for i in stride(from: 0, to: mesh.indices.count, by: 3) {
                let a = mesh.vertices[Int(mesh.indices[i])].position
                let b = mesh.vertices[Int(mesh.indices[i + 1])].position
                let c = mesh.vertices[Int(mesh.indices[i + 2])].position
                let origin = SIMD3<Double>(a)
                let cross = simd_cross(SIMD3<Double>(b) - origin, SIMD3<Double>(c) - origin)
                let normal = cross == .zero ? SIMD3<Float>.zero : SIMD3<Float>(simd_normalize(cross))
                emit(normal, a, b, c)
            }
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case noTriangles
        case triangleCount

        public var errorDescription: String? {
            switch self {
            case .noTriangles: "The scene contains no STL triangles."
            case .triangleCount: "Binary STL supports at most 4,294,967,295 facets."
            }
        }
    }
}
