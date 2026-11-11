import Foundation
import simd

public extension MeshGeometry {
    /// Tessellates a regular surface over UV [0, 1]². The front faces the cross
    /// product of increasing U and V. Wrapped axes sample zero at the seam and
    /// duplicate their vertices at UV one for continuous texture interpolation.
    /// Supply a regular, non-collapsing parameterization. The mesh includes
    /// smooth normals and a tangent basis estimated from neighboring samples.
    static func parametricSurface(
        uSegments: Int, vSegments: Int, wrapU: Bool = false, wrapV: Bool = false,
        position: (SIMD2<Float>) throws -> SIMD3<Float>
    ) throws -> MeshGeometry {
        guard ((wrapU ? 3 : 1)...1024).contains(uSegments),
              ((wrapV ? 3 : 1)...1024).contains(vSegments),
              (uSegments + 1) * (vSegments + 1) <= 1_000_000 else {
            throw ParametricSurfaceError.resolution
        }

        let columns = uSegments + 1
        var points: [SIMD3<Float>] = []
        points.reserveCapacity(columns * (vSegments + 1))
        for v in 0...vSegments {
            for u in 0...uSegments {
                let point: SIMD3<Float>
                if wrapV, v == vSegments {
                    point = points[u]
                } else if wrapU, u == uSegments {
                    point = points[v * columns]
                } else {
                    point = try position(SIMD2(Float(u) / Float(uSegments), Float(v) / Float(vSegments)))
                }
                guard point.x.isFinite, point.y.isFinite, point.z.isFinite else {
                    throw ParametricSurfaceError.nonfiniteSample(u: u, v: v)
                }
                points.append(point)
            }
        }

        func neighbors(_ index: Int, end: Int, wraps: Bool) -> (Int, Int) {
            if wraps, index == 0 || index == end { return (end - 1, 1) }
            return (max(0, index - 1), min(end, index + 1))
        }

        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        vertices.reserveCapacity(points.count)
        indices.reserveCapacity(uSegments * vSegments * 6)
        for v in 0...vSegments {
            let (beforeV, afterV) = neighbors(v, end: vSegments, wraps: wrapV)
            for u in 0...uSegments {
                let (beforeU, afterU) = neighbors(u, end: uSegments, wraps: wrapU)
                let index = v * columns + u
                let du = SIMD3<Double>(points[v * columns + afterU]) - SIMD3<Double>(points[v * columns + beforeU])
                let dv = SIMD3<Double>(points[afterV * columns + u]) - SIMD3<Double>(points[beforeV * columns + u])
                let normal = simd_cross(du, dv)
                guard simd_length_squared(normal) > 0 else {
                    throw ParametricSurfaceError.singularSample(u: u, v: v)
                }
                let tangent = SIMD3<Float>(simd_normalize(du))
                vertices.append(MeshVertex(
                    position: points[index], normal: SIMD3<Float>(simd_normalize(normal)),
                    textureCoordinate: SIMD2(Float(u) / Float(uSegments), Float(v) / Float(vSegments)),
                    tangent: SIMD4(tangent, 1)
                ))
                if u < uSegments, v < vSegments {
                    let a = UInt32(index)
                    let c = a + UInt32(columns)
                    indices += [a, a + 1, c, a + 1, c + 1, c]
                }
            }
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    enum ParametricSurfaceError: LocalizedError, Equatable {
        case resolution
        case nonfiniteSample(u: Int, v: Int)
        case singularSample(u: Int, v: Int)

        public var errorDescription: String? {
            switch self {
            case .resolution: "Use 1 to 1024 segments per open axis or 3 to 1024 per wrapped axis, with at most one million vertices."
            case .nonfiniteSample(let u, let v): "The surface sample at (\(u), \(v)) is nonfinite."
            case .singularSample(let u, let v): "The surface sample at (\(u), \(v)) has a degenerate tangent basis."
            }
        }
    }
}
