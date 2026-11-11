import Foundation
import simd

extension MeshGeometry {
    /// Tessellates the Utah teapot's bicubic patches in their original order.
    /// `size` is the half-height of the complete teapot. Body includes the
    /// rim, handle and spout. Each patch has independent top-left UVs.
    ///
    /// Each patch edge uses at least two segments, with a combined budget
    /// of one million vertices. Fitting expands the lid horizontally by
    /// 7.7%. Set `blinn` to false for the original taller proportions,
    /// normalized to the same half-height.
    public static func teapot(
        size: Float = 1, segments: Int = 10, bottom: Bool = true, lid: Bool = true, body: Bool = true,
        fitLid: Bool = true, blinn: Bool = true
    ) throws -> MeshGeometry {
        guard size.isNormal, size > 0 else { throw TeapotError.size }
        guard body || lid || bottom else { throw TeapotError.noPatches }
        // Even four bottom patches would exceed the budget above 499 segments.
        guard segments <= 499 else { throw TeapotError.resolution }
        let steps = max(2, segments), stride = steps + 1
        let patchCount = (body ? 20 : 0) + (lid ? 8 : 0) + (bottom ? 4 : 0)
        let vertexCount = patchCount * stride * stride
        guard vertexCount <= 1_000_000 else { throw TeapotError.resolution }

        let heightScale = blinn ? 1.0 : 1.3
        let halfHeight = 3.15 * heightScale / 2
        let scale = Double(size) / halfHeight
        let bases = (0...steps).map { TeapotBasis(Double($0) / Double(steps)) }
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        vertices.reserveCapacity(vertexCount)
        let poleCount = (lid ? 4 : 0) + (bottom ? 4 : 0)
        indices.reserveCapacity(3 * (patchCount * 2 * steps * steps - poleCount * steps))

        func appendTriangle(_ a: Int, _ b: Int, _ c: Int) throws {
            let origin = SIMD3<Double>(vertices[a].position)
            let edgeA = SIMD3<Double>(vertices[b].position) - origin
            let edgeB = SIMD3<Double>(vertices[c].position) - origin
            guard simd_length_squared(simd_cross(edgeA, edgeB)) > 0 else { throw TeapotError.unrepresentableSurface }
            indices.append(UInt32(a))
            indices.append(UInt32(b))
            indices.append(UInt32(c))
        }

        for patch in 0..<32 {
            if patch < 20, !body { continue }
            if (20..<28).contains(patch), !lid { continue }
            if patch >= 28, !bottom { continue }
            let planarScale = fitLid && (20..<28).contains(patch) ? 1.077 : 1.0
            let controlScale = SIMD3(planarScale, planarScale, heightScale)
            let controls = TeapotPatchData.indices[(patch * 16)..<(patch * 16 + 16)]
                .map { TeapotPatchData.point($0) * controlScale }
            let rows = (0..<4).map { row in
                let start = row * 4
                return TeapotCubic(a: controls[start], b: controls[start + 1], c: controls[start + 2], d: controls[start + 3])
            }
            let columns = bases.map { t in
                let points = TeapotCubic(a: rows[0].point(t), b: rows[1].point(t), c: rows[2].point(t), d: rows[3].point(t))
                let derivatives = TeapotCubic(
                    a: rows[0].derivative(t), b: rows[1].derivative(t), c: rows[2].derivative(t), d: rows[3].derivative(t)
                )
                return (points: points, derivatives: derivatives)
            }
            let start = vertices.count
            let poleSign: Double = switch patch {
            case 20..<24: 1
            case 28..<32: -1
            default: 0
            }
            for (row, s) in bases.enumerated() {
                for (columnIndex, column) in columns.enumerated() {
                    let point = column.points.point(s)
                    let position = SIMD3<Float>(scale * teapotYUp(point - SIMD3(0, 0, halfHeight)))
                    let atPole = row == 0 && poleSign != 0
                    let ds = column.points.derivative(s)
                    let dt = column.derivatives.point(s)
                    let normalDirection = atPole ? SIMD3(0, poleSign, 0) : teapotYUp(simd_cross(dt, ds))
                    // The collapsed row has zero dP/dt. Its limiting tangent
                    // comes from the next row of controls.
                    let alongU = teapotYUp(-(atPole ? column.derivatives.b : dt))
                    guard boundsFinite(position), simd_length_squared(normalDirection) > 0,
                          simd_length_squared(alongU) > 0 else { throw TeapotError.unrepresentableSurface }
                    vertices.append(MeshVertex(
                        position: position, normal: SIMD3<Float>(simd_normalize(normalDirection)),
                        textureCoordinate: SIMD2(Float(1 - bases[columnIndex].parameter), Float(s.parameter)),
                        tangent: SIMD4(SIMD3<Float>(simd_normalize(alongU)), -1)
                    ))
                }
            }
            for row in 0..<steps {
                for column in 0..<steps {
                    let a = start + row * stride + column
                    let c = a + stride + 1
                    if row != 0 || poleSign == 0 { try appendTriangle(a, a + 1, c) }
                    try appendTriangle(a, c, c - 1)
                }
            }
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    public enum TeapotError: LocalizedError, Equatable {
        case size
        case noPatches
        case resolution
        case unrepresentableSurface

        public var errorDescription: String? {
            switch self {
            case .size: "Teapot size must be a positive normal number."
            case .noPatches: "Enable at least one of the teapot body, lid or bottom."
            case .resolution: "The teapot may have at most one million patch vertices."
            case .unrepresentableSurface: "The teapot surface exceeds Float position precision at this size."
            }
        }
    }
}

private func teapotYUp(_ vector: SIMD3<Double>) -> SIMD3<Double> {
    SIMD3(vector.x, vector.z, -vector.y)
}

private struct TeapotBasis {
    let parameter: Double
    let weights: SIMD3<Double>
    let derivativeWeights: SIMD3<Double>

    init(_ t: Double) {
        parameter = t
        let u = 1 - t
        weights = SIMD3(3 * u * u * t, 3 * u * t * t, t * t * t)
        derivativeWeights = SIMD3(3 * u * u, 6 * u * t, 3 * t * t)
    }
}

private struct TeapotCubic {
    let a: SIMD3<Double>
    let b: SIMD3<Double>
    let c: SIMD3<Double>
    let d: SIMD3<Double>

    func point(_ basis: TeapotBasis) -> SIMD3<Double> {
        if basis.parameter == 0 { return a }
        if basis.parameter == 1 { return d }
        let w = basis.weights
        // Evaluate relative to the first control so constant coordinates
        // remain exact, including across a collapsed pole row.
        return a + w.x * (b - a) + w.y * (c - a) + w.z * (d - a)
    }

    func derivative(_ basis: TeapotBasis) -> SIMD3<Double> {
        let w = basis.derivativeWeights
        return w.x * (b - a) + w.y * (c - b) + w.z * (d - c)
    }
}
