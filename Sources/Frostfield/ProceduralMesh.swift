import Foundation
import simd

extension MeshGeometry {
    /// A centered XY plane facing positive Z. UV (0, 0) is the top-left corner.
    /// Each dimension must be positive and normal, with 1 to 256 subdivisions per axis.
    public static func plane(
        width: Float = 1,
        height: Float = 1,
        columns: Int = 1,
        rows: Int = 1
    ) throws -> MeshGeometry {
        guard width.isNormal, width > 0, height.isNormal, height > 0 else {
            throw ProceduralValidationError.invalidPlaneSize
        }
        guard (1...256).contains(columns), (1...256).contains(rows) else {
            throw ProceduralValidationError.invalidPlaneSubdivisions
        }

        let stride = columns + 1
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        vertices.reserveCapacity(stride * (rows + 1))
        indices.reserveCapacity(columns * rows * 6)

        for row in 0...rows {
            let v = Float(row) / Float(rows)
            for column in 0...columns {
                let u = Float(column) / Float(columns)
                vertices.append(MeshVertex(
                    position: SIMD3((u - 0.5) * width, (0.5 - v) * height, 0),
                    normal: SIMD3(0, 0, 1), textureCoordinate: SIMD2(u, v)
                ))
            }
        }
        for row in 0..<rows {
            for column in 0..<columns {
                let top = UInt32(row * stride + column)
                let bottom = top + UInt32(stride)
                indices += [top, bottom, top + 1, top + 1, bottom, bottom + 1]
            }
        }

        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    /// A sphere centered at the origin with poles on Y. The longitudinal seam
    /// starts on positive X. V increases from the north pole toward the south.
    /// Supply 3 to 256 longitude segments and 2 to 256 pole-to-pole bands.
    public static func uvSphere(
        radius: Float = 0.5,
        segments: Int = 32,
        rings: Int = 16
    ) throws -> MeshGeometry {
        guard radius.isNormal, radius > 0 else { throw ProceduralValidationError.invalidSphereRadius }
        guard (3...256).contains(segments), (2...256).contains(rings) else {
            throw ProceduralValidationError.invalidSphereSubdivisions
        }

        let stride = segments + 1
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        vertices.reserveCapacity((rings - 1) * stride + 2 * segments)
        indices.reserveCapacity(6 * segments * (rings - 1))

        for row in 1..<rings {
            let v = Double(row) / Double(rings)
            let latitude = v * .pi
            let ringRadius = sin(latitude)
            let y = cos(latitude)
            for column in 0...segments {
                let u = Double(column) / Double(segments)
                // Reuse the exact endpoint direction on the other side of the UV seam.
                let longitude = column == segments ? 0 : u * 2 * .pi
                let normal = SIMD3<Float>(
                    Float(ringRadius * cos(longitude)), Float(y), Float(ringRadius * sin(longitude))
                )
                vertices.append(MeshVertex(
                    position: normal * radius, normal: normal, textureCoordinate: SIMD2(Float(u), Float(v))
                ))
            }
        }

        for row in 0..<(rings - 2) {
            for column in 0..<segments {
                let upper = UInt32(row * stride + column)
                let lower = upper + UInt32(stride)
                indices += [upper, upper + 1, lower, upper + 1, lower + 1, lower]
            }
        }

        for column in 0..<segments {
            // Each cap wedge has its own pole UV centered between its two rim vertices.
            let u = (Float(column) + 0.5) / Float(segments)
            let north = UInt32(vertices.count)
            vertices.append(MeshVertex(
                position: SIMD3(0, radius, 0), normal: SIMD3(0, 1, 0), textureCoordinate: SIMD2(u, 0)
            ))
            let south = UInt32(vertices.count)
            vertices.append(MeshVertex(
                position: SIMD3(0, -radius, 0), normal: SIMD3(0, -1, 0), textureCoordinate: SIMD2(u, 1)
            ))
            let upper = UInt32(column)
            let lower = UInt32((rings - 2) * stride + column)
            indices += [north, upper + 1, upper, lower, lower + 1, south]
        }

        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    public enum ProceduralValidationError: LocalizedError, Equatable {
        case invalidPlaneSize
        case invalidPlaneSubdivisions
        case invalidSphereRadius
        case invalidSphereSubdivisions

        public var errorDescription: String? {
            switch self {
            case .invalidPlaneSize: "Plane width and height must be positive normal numbers."
            case .invalidPlaneSubdivisions: "A plane needs 1 to 256 columns and rows."
            case .invalidSphereRadius: "The sphere radius must be a positive normal number."
            case .invalidSphereSubdivisions: "A sphere needs 3 to 256 segments and 2 to 256 rings."
            }
        }
    }
}
