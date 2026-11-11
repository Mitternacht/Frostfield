import Foundation
import simd

/// An oriented box that projects a decal along its local −Z axis.
///
/// The box is centered on `position`, turned by `orientation`, and spans
/// `size.x` along its local X, `size.y` along Y and `size.z` along the
/// projection axis. The decal image's top-left corner maps to the box's −X,
/// +Y edge, so a projector looking down −Z shows the image upright.
public struct DecalProjector: Equatable, Sendable {
    public var position: SIMD3<Float>
    public var orientation: simd_quatf
    public var size: SIMD3<Float>

    public init(
        position: SIMD3<Float>, orientation: simd_quatf = simd_quatf(angle: 0, axis: SIMD3(0, 1, 0)),
        size: SIMD3<Float>
    ) {
        self.position = position
        self.orientation = orientation
        self.size = size
    }

    /// A projector at `position` whose −Z axis points along `direction`,
    /// with its +Y axis as close to `up` as the direction allows, turned by
    /// `angle` radians about the projection axis.
    public init(position: SIMD3<Float>, direction: SIMD3<Float>, up: SIMD3<Float> = SIMD3(0, 1, 0), angle: Float = 0, size: SIMD3<Float>) throws {
        let back = -SIMD3<Double>(direction)
        guard simd_length(back) > 0, simd_length(back).isFinite else { throw MeshGeometry.DecalError.invalidProjector }
        let z = simd_normalize(back)
        var x = simd_cross(SIMD3<Double>(up), z)
        if simd_length(x) < 1e-6 { x = simd_cross(abs(z.y) < 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0), z) }
        x = simd_normalize(x)
        let y = simd_cross(z, x)
        let basis = simd_quatd(simd_double3x3(columns: (x, y, z)))
        let spin = simd_quatd(angle: Double(angle), axis: SIMD3(0, 0, 1))
        self.init(position: position, orientation: simd_quatf(vector: SIMD4<Float>((basis * spin).normalized.vector)), size: size)
    }
}

extension MeshGeometry {
    /// Projects a decal onto this mesh, like Three.js `DecalGeometry`, and
    /// returns the clipped surface in world space, or nil when no triangle
    /// facing the projector reaches the box.
    ///
    /// Each triangle is transformed by `localToWorld`, clipped against the
    /// box's six planes and fanned back into triangles. UVs map projector X
    /// and −Y to 0...1 across the box, so `u = x / size.x + 0.5` and
    /// `v = 0.5 − y / size.y`. Normals interpolate the mesh's own normals,
    /// and tangents follow +U on the surface, so normal-mapped decals shade
    /// correctly. Triangles whose geometric normal turns more than
    /// `maximumAngle` from the projector's +Z are skipped; the default of
    /// 90° keeps decals off back faces. `surfaceOffset` moves vertices along
    /// their normals in world units to avoid depth fighting.
    ///
    /// Supplying a hierarchy built from this geometry visits only the
    /// triangles that touch the box.
    public func decal(
        projector: DecalProjector, localToWorld: simd_float4x4 = matrix_identity_float4x4,
        maximumAngle: Float = .pi / 2, surfaceOffset: Float = 0, bvh: MeshBVH? = nil
    ) throws -> MeshGeometry? {
        let size = SIMD3<Double>(projector.size)
        guard (0..<3).allSatisfy({ size[$0].isFinite && size[$0] > 0 }),
              simd_length(projector.position).isFinite,
              abs(simd_length(SIMD4<Double>(projector.orientation.vector)) - 1) < 1e-3 else {
            throw DecalError.invalidProjector
        }
        guard maximumAngle.isFinite, maximumAngle > 0, maximumAngle <= .pi else { throw DecalError.invalidAngle }
        guard surfaceOffset.isFinite else { throw DecalError.invalidOffset }
        let world = simd_double4x4(columns: (
            SIMD4<Double>(localToWorld.columns.0), SIMD4<Double>(localToWorld.columns.1),
            SIMD4<Double>(localToWorld.columns.2), SIMD4<Double>(localToWorld.columns.3)
        ))
        let linear = simd_double3x3(
            SIMD3(world.columns.0.x, world.columns.0.y, world.columns.0.z),
            SIMD3(world.columns.1.x, world.columns.1.y, world.columns.1.z),
            SIMD3(world.columns.2.x, world.columns.2.y, world.columns.2.z)
        )
        let determinant = linear.determinant
        guard determinant.isFinite, determinant != 0 else { throw DecalError.invalidTransform }
        let normalMatrix = linear.inverse.transpose

        let rotation = simd_double3x3(simd_quatd(vector: SIMD4<Double>(projector.orientation.vector)).normalized)
        let center = SIMD3<Double>(projector.position)
        let axisX = rotation.columns.0, axisY = rotation.columns.1, axisZ = rotation.columns.2
        let half = size / 2
        let cosineLimit = cos(Double(maximumAngle))

        let candidates: [Int]
        if let bvh {
            guard bvh.isBuilt(from: self) else { throw MeshBVH.Failure.geometryMismatch }
            // The box's corners in mesh space bound the triangles worth clipping.
            let inverse = world.inverse
            var corners: [SIMD3<Float>] = []
            for sx in [-1.0, 1.0] { for sy in [-1.0, 1.0] { for sz in [-1.0, 1.0] {
                let point = center + axisX * (sx * half.x) + axisY * (sy * half.y) + axisZ * (sz * half.z)
                let local = inverse * SIMD4(point, 1)
                corners.append(SIMD3<Float>(SIMD3(local.x, local.y, local.z) / local.w))
            } } }
            let box = try AxisAlignedBox(enclosing: corners)
            let pad = SIMD3<Float>(repeating: 1e-5 * max(1, simd_reduce_max(simd_abs(box.maximum - box.minimum))))
            candidates = bvh.triangles(intersecting: try AxisAlignedBox(minimum: box.minimum - pad, maximum: box.maximum + pad))
        } else {
            candidates = Array(0..<(indices.count / 3))
        }

        struct ClipVertex {
            var local: SIMD3<Double>
            var normal: SIMD3<Double>
        }
        var output: [MeshVertex] = []
        var outputIndices: [UInt32] = []
        var lookup: [DecalVertexKey: UInt32] = [:]
        let flips = determinant < 0
        for triangle in candidates {
            var corners: [ClipVertex] = []
            var worldCorners: [SIMD3<Double>] = []
            for slot in 0..<3 {
                let vertex = vertices[Int(indices[triangle * 3 + slot])]
                let transformed = world * SIMD4(SIMD3<Double>(vertex.position), 1)
                let position = SIMD3(transformed.x, transformed.y, transformed.z)
                worldCorners.append(position)
                let offset = position - center
                corners.append(ClipVertex(
                    local: SIMD3(simd_dot(offset, axisX), simd_dot(offset, axisY), simd_dot(offset, axisZ)),
                    normal: simd_normalize(normalMatrix * SIMD3<Double>(vertex.normal))
                ))
            }
            if flips { corners.swapAt(1, 2); worldCorners.swapAt(1, 2) }
            let faceNormal = simd_cross(worldCorners[1] - worldCorners[0], worldCorners[2] - worldCorners[0])
            let faceLength = simd_length(faceNormal)
            guard faceLength > 0, simd_dot(faceNormal / faceLength, axisZ) > cosineLimit else { continue }

            // Sutherland–Hodgman clipping against each slab of the box.
            var polygon = corners
            for axis in 0..<3 {
                for sign in [1.0, -1.0] {
                    guard !polygon.isEmpty else { break }
                    var clipped: [ClipVertex] = []
                    for index in polygon.indices {
                        let a = polygon[index], b = polygon[(index + 1) % polygon.count]
                        let da = half[axis] - sign * a.local[axis], db = half[axis] - sign * b.local[axis]
                        if da >= 0 { clipped.append(a) }
                        if (da >= 0) != (db >= 0) {
                            let t = da / (da - db)
                            clipped.append(ClipVertex(
                                local: a.local + (b.local - a.local) * t, normal: a.normal + (b.normal - a.normal) * t
                            ))
                        }
                    }
                    polygon = clipped
                }
            }
            guard polygon.count >= 3 else { continue }
            let tangentNormal = faceNormal / faceLength
            var ids: [UInt32] = []
            for corner in polygon {
                var normal = corner.normal
                normal = simd_length_squared(normal) > 1e-24 ? simd_normalize(normal) : tangentNormal
                let position = center + axisX * corner.local.x + axisY * corner.local.y + axisZ * corner.local.z
                    + normal * Double(surfaceOffset)
                let uv = SIMD2(corner.local.x / size.x + 0.5, 0.5 - corner.local.y / size.y)
                // +U runs along projector X and +V along projector −Y, both projected onto the surface.
                var tangent = axisX - normal * simd_dot(axisX, normal)
                if simd_length_squared(tangent) < 1e-12 { tangent = -axisY - normal * simd_dot(-axisY, normal) }
                tangent = simd_normalize(tangent)
                let handedness: Float = simd_dot(simd_cross(normal, tangent), -axisY) < 0 ? -1 : 1
                let vertex = MeshVertex(
                    position: SIMD3<Float>(position), normal: SIMD3<Float>(normal), textureCoordinate: SIMD2<Float>(uv),
                    tangent: SIMD4(SIMD3<Float>(tangent), handedness)
                )
                let key = DecalVertexKey(vertex)
                if let existing = lookup[key] {
                    ids.append(existing)
                } else {
                    output.append(vertex)
                    lookup[key] = UInt32(output.count - 1)
                    ids.append(UInt32(output.count - 1))
                }
            }
            for index in 1..<(ids.count - 1) where ids[0] != ids[index] && ids[index] != ids[index + 1] && ids[0] != ids[index + 1] {
                outputIndices += [ids[0], ids[index], ids[index + 1]]
            }
        }
        guard !outputIndices.isEmpty else { return nil }
        return try MeshGeometry(vertices: output, indices: outputIndices)
    }

    public enum DecalError: LocalizedError, Equatable {
        case invalidProjector
        case invalidAngle
        case invalidOffset
        case invalidTransform

        public var errorDescription: String? {
            switch self {
            case .invalidProjector:
                "A decal projector needs a finite position, a unit orientation and a positive, finite size."
            case .invalidAngle: "The decal's maximum angle must be greater than zero and at most pi radians."
            case .invalidOffset: "The decal's surface offset must be finite."
            case .invalidTransform: "The mesh transform must be finite and invertible."
            }
        }
    }
}

/// Bitwise identity of a decal vertex, so clipped corners shared by neighbouring triangles merge.
private struct DecalVertexKey: Hashable {
    let bits: [UInt32]

    init(_ vertex: MeshVertex) {
        let tangent = vertex.tangent ?? .zero
        bits = [
            vertex.position.x, vertex.position.y, vertex.position.z, vertex.normal.x, vertex.normal.y, vertex.normal.z,
            vertex.textureCoordinate.x, vertex.textureCoordinate.y, tangent.x, tangent.y, tangent.z, tangent.w,
        ].map(\.bitPattern)
    }
}
