import Foundation
import simd

/// A world-space ray measured from its origin along a unit direction.
public struct SceneRay: Equatable, Sendable {
    public let origin: SIMD3<Float>
    public let direction: SIMD3<Float>
    public let maximumDistance: Float

    /// The maximum distance is inclusive. Use infinity for an unbounded ray.
    public init(
        origin: SIMD3<Float>,
        direction: SIMD3<Float>,
        maximumDistance: Float = .infinity
    ) throws {
        guard isFinite(origin) else { throw ScenePickingError.invalidOrigin }
        guard isFinite(direction), direction != .zero else { throw ScenePickingError.invalidDirection }
        guard maximumDistance >= 0 else { throw ScenePickingError.invalidMaximumDistance }
        self.origin = origin
        self.direction = floatVector(simd_normalize(doubleVector(direction)))
        self.maximumDistance = maximumDistance
    }
}

/// An indexed triangle hit in world space.
public struct MeshRayHit: Equatable, Sendable {
    public let distance: Float
    public let position: SIMD3<Float>
    /// Zero-based triangle index, counting groups of three indices.
    public let triangleIndex: Int
    /// Weights of the triangle's three vertices in index order.
    public let barycentric: SIMD3<Float>
    /// The first texture coordinate set interpolated at the hit, with a top-left origin.
    public let textureCoordinate: SIMD2<Float>
    /// The second texture coordinate set, when all three corners supply it.
    public let secondaryTextureCoordinate: SIMD2<Float>?
    /// Whether the ray struck the authored front face, including on mirrored meshes.
    public let isFrontFace: Bool
    /// The unit geometric normal in mesh coordinates, oriented by the index order.
    public let localFaceNormal: SIMD3<Float>
    /// The unit interpolated normal in mesh coordinates, facing against the ray.
    public let localNormal: SIMD3<Float>
    /// The unit geometric normal in world coordinates, preserving the authored front under reflection.
    public let faceNormal: SIMD3<Float>
    /// The unit interpolated normal in world coordinates, facing against the ray.
    public let normal: SIMD3<Float>

    func offsetTriangleIndex(by offset: Int) -> Self {
        guard offset != 0 else { return self }
        return Self(
            distance: distance, position: position, triangleIndex: triangleIndex + offset, barycentric: barycentric,
            textureCoordinate: textureCoordinate, secondaryTextureCoordinate: secondaryTextureCoordinate,
            isFrontFace: isFrontFace, localFaceNormal: localFaceNormal, localNormal: localNormal,
            faceNormal: faceNormal, normal: normal
        )
    }
}

public enum ScenePickingError: LocalizedError, Equatable {
    case invalidOrigin
    case invalidDirection
    case invalidMaximumDistance
    case invalidNormalizedPoint
    case nonfiniteQueryPoint
    case invalidTransform
    case unrepresentableResult

    public var errorDescription: String? {
        switch self {
        case .invalidOrigin: "A picking ray needs a finite origin."
        case .invalidDirection: "A picking ray needs a finite, nonzero direction."
        case .invalidMaximumDistance: "A picking ray's maximum distance must be nonnegative."
        case .invalidNormalizedPoint: "A picking point must lie inside the normalized drawable."
        case .nonfiniteQueryPoint: "A ray proximity query needs finite points."
        case .invalidTransform: "Picking needs a finite, invertible affine transform."
        case .unrepresentableResult: "The picking result is outside the representable world coordinates."
        }
    }
}

extension PerspectiveCamera {
    /// Creates a ray through the drawable, starting at the near plane and ending
    /// at the far plane. Normalized (0, 0) is the top-left and (1, 1) the bottom-right.
    public func ray(
        at normalized: SIMD2<Float>,
        aspectRatio: Float,
        cameraTransform: SceneTransform = .init()
    ) throws -> SceneRay {
        guard (0...1).contains(normalized.x), (0...1).contains(normalized.y) else {
            throw ScenePickingError.invalidNormalizedPoint
        }

        let projection = try projection(aspectRatio: aspectRatio)
        let transform = try PickingAffine(cameraTransform.matrix)
        // At camera-space z = -1, undo the projection's horizontal and vertical scales.
        let cameraStep = SIMD3<Double>(
            (2 * Double(normalized.x) - 1) / Double(projection.columns.0.x),
            (1 - 2 * Double(normalized.y)) / Double(projection.columns.1.y),
            -1
        )
        let worldStep = transform.linear * cameraStep
        let worldStepLength = simd_length(worldStep)
        let origin = floatVector(transform.translation + worldStep * Double(near))
        let length = Float(worldStepLength * (Double(far) - Double(near)))
        guard isFinite(origin), length.isFinite, length > 0 else {
            throw ScenePickingError.unrepresentableResult
        }

        return try SceneRay(
            origin: origin, direction: floatVector(worldStep / worldStepLength), maximumDistance: length
        )
    }
}

public extension ShaderGraphCamera {
    /// Creates a near-to-far ray through a top-left normalized drawable point.
    /// Supply the depth mode used to build the projection matrix. Supports
    /// orthographic, asymmetric perspective, and infinite perspective views.
    func ray(at normalized: SIMD2<Float>, depthMode: MeshDepthMode = .forward) throws -> SceneRay {
        guard (0...1).contains(normalized.x), (0...1).contains(normalized.y) else {
            throw ScenePickingError.invalidNormalizedPoint
        }
        let worldFromClip = simd_inverse(simdPickingMatrix(viewMatrix))
            * simd_inverse(simdPickingMatrix(projectionMatrix))
        let x = 2 * Double(normalized.x) - 1
        let y = 1 - 2 * Double(normalized.y)
        func unproject(_ depth: Double) throws -> SIMD3<Double> {
            let point = worldFromClip * SIMD4(x, y, depth, 1)
            let world = SIMD3(point.x, point.y, point.z) / point.w
            guard world.x.isFinite, world.y.isFinite, world.z.isFinite else {
                throw ScenePickingError.unrepresentableResult
            }
            return world
        }
        let nearDepth: Double = depthMode == .reversed ? 1 : 0
        let origin = try unproject(nearDepth)
        let step = try unproject(far.isInfinite ? 0.5 : 1 - nearDepth) - origin
        let length = simd_length(step)
        let limit = far.isInfinite ? Float.infinity : Float(length)
        guard length.isFinite, length > 0, far.isInfinite || (limit.isFinite && limit > 0),
              isFinite(floatVector(origin)) else {
            throw ScenePickingError.unrepresentableResult
        }
        return try SceneRay(origin: floatVector(origin), direction: floatVector(step / length), maximumDistance: limit)
    }

    internal func pickingPosition() throws -> SIMD3<Float> {
        let position = simd_inverse(simdPickingMatrix(viewMatrix)).columns.3
        let world = SIMD3<Float>(SIMD3(position.x, position.y, position.z) / position.w)
        guard isFinite(world) else { throw ScenePickingError.invalidTransform }
        return world
    }
}

func simdPickingMatrix(_ matrix: simd_float4x4) -> simd_double4x4 {
    let (x, y, z, w) = matrix.columns
    return simd_double4x4(columns: (SIMD4<Double>(x), SIMD4<Double>(y), SIMD4<Double>(z), SIMD4<Double>(w)))
}

extension MeshGeometry {
    /// Finds the nearest front-facing indexed triangle. Mirrored transforms
    /// preserve the mesh's original front face. Equal distances prefer the lower
    /// triangle index. Degenerate triangles are ignored.
    public func intersection(ray: SceneRay, localToWorld: simd_float4x4) throws -> MeshRayHit? {
        try intersection(ray: ray, localToWorld: localToWorld, cullsBackFaces: true)
    }

    /// Finds the nearest indexed triangle, including its back face when culling is disabled.
    /// This scans every triangle; a ``MeshBVH`` answers repeated queries identically and faster.
    public func intersection(
        ray: SceneRay, localToWorld: simd_float4x4, cullsBackFaces: Bool
    ) throws -> MeshRayHit? {
        try intersection(ray: ray, localToWorld: localToWorld, side: cullsBackFaces ? .front : .double)
    }

    /// Finds the nearest indexed triangle accepted by the supplied material side.
    public func intersection(
        ray: SceneRay, localToWorld: simd_float4x4, side: MeshRenderState.Side
    ) throws -> MeshRayHit? {
        let transform = try PickingAffine(localToWorld)
        let query = PickingQuery(ray: ray, transform: transform, side: side)
        var closestDistance = query.maximumDistance
        var closestTriangle: Int?
        for offset in stride(from: 0, to: indices.count, by: 3) {
            guard let distance = query.distance(to: offset / 3, in: self),
                  closestTriangle == nil || distance < closestDistance else { continue }
            closestDistance = distance
            closestTriangle = offset / 3
        }

        guard let closestTriangle else { return nil }
        return try query.hit(distance: closestDistance, triangleIndex: closestTriangle, in: self, transform: transform)
    }

    /// Finds every indexed triangle the ray crosses within its maximum distance,
    /// nearest first. Equal distances are ordered by triangle index. Culling,
    /// degenerate triangles and distances follow ``intersection(ray:localToWorld:cullsBackFaces:)``.
    public func intersections(
        ray: SceneRay, localToWorld: simd_float4x4, cullsBackFaces: Bool = true
    ) throws -> [MeshRayHit] {
        try intersections(ray: ray, localToWorld: localToWorld, side: cullsBackFaces ? .front : .double)
    }

    /// Finds every indexed triangle the ray crosses on the supplied material side.
    public func intersections(
        ray: SceneRay, localToWorld: simd_float4x4, side: MeshRenderState.Side
    ) throws -> [MeshRayHit] {
        let transform = try PickingAffine(localToWorld)
        let query = PickingQuery(ray: ray, transform: transform, side: side)
        var hits: [(distance: Double, triangle: Int)] = []
        for triangle in 0..<(indices.count / 3) {
            if let distance = query.distance(to: triangle, in: self) { hits.append((distance, triangle)) }
        }
        return try query.sortedHits(hits, in: self, transform: transform)
    }
}

/// One ray prepared in a mesh's local space. The linear scan and ``MeshBVH``
/// share this triangle test, so both report identical distances.
struct PickingQuery {
    let worldOrigin: SIMD3<Double>
    let worldDirection: SIMD3<Double>
    let localOrigin: SIMD3<Double>
    /// Unnormalized, so its parameter is the world-space distance.
    let localDirection: SIMD3<Double>
    let maximumDistance: Double
    let side: MeshRenderState.Side

    init(ray: SceneRay, transform: PickingAffine, side: MeshRenderState.Side) {
        worldOrigin = doubleVector(ray.origin)
        worldDirection = simd_normalize(doubleVector(ray.direction))
        localOrigin = transform.inverse * (worldOrigin - transform.translation)
        // Keep this direction unnormalized to preserve the world-space ray parameter.
        localDirection = transform.inverse * worldDirection
        maximumDistance = Double(ray.maximumDistance)
        self.side = side
    }

    /// A double-precision ray for opaque tests in a mesh already in world coordinates.
    init(origin: SIMD3<Double>, unitDirection: SIMD3<Double>, maximumDistance: Double = .infinity) {
        worldOrigin = origin
        worldDirection = unitDirection
        localOrigin = origin
        localDirection = unitDirection
        self.maximumDistance = maximumDistance
        side = .double
    }

    /// The world-space distance to one triangle, or nil when the ray misses it,
    /// meets a culled face, or the hit lies outside `0...maximumDistance`.
    @inline(__always)
    func distance(to triangle: Int, in geometry: MeshGeometry) -> Double? {
        let offset = triangle * 3
        let a = doubleVector(geometry.vertices[Int(geometry.indices[offset])].position)
        let edgeAB = doubleVector(geometry.vertices[Int(geometry.indices[offset + 1])].position) - a
        let edgeAC = doubleVector(geometry.vertices[Int(geometry.indices[offset + 2])].position) - a

        // Möller–Trumbore in local space. A positive determinant
        // selects the CCW front face before the model's transform is applied.
        let normal = simd_cross(edgeAB, edgeAC)
        let signedDeterminant = -simd_dot(localDirection, normal)
        guard signedDeterminant != 0, signedDeterminant > 0 ? side != .back : side != .front else { return nil }
        let determinant = abs(signedDeterminant)
        let facing = signedDeterminant < 0 ? -1.0 : 1.0
        let fromA = localOrigin - a
        let u = facing * simd_dot(fromA, simd_cross(localDirection, edgeAC))
        guard u >= 0, u <= determinant else { return nil }
        let q = simd_cross(fromA, edgeAB)
        let v = facing * simd_dot(localDirection, q)
        guard v >= 0, u + v <= determinant else { return nil }

        let distance = simd_dot(edgeAC, q) / signedDeterminant
        guard distance.isFinite, distance >= 0, distance <= maximumDistance else { return nil }
        return distance
    }

    func hit(distance: Double, triangleIndex: Int, in geometry: MeshGeometry, transform: PickingAffine) throws -> MeshRayHit {
        let rounded = Float(distance)
        let position = floatVector(worldOrigin + worldDirection * distance)
        guard rounded.isFinite, isFinite(position) else { throw ScenePickingError.unrepresentableResult }
        let offset = triangleIndex * 3
        let a = geometry.vertices[Int(geometry.indices[offset])]
        let b = geometry.vertices[Int(geometry.indices[offset + 1])]
        let c = geometry.vertices[Int(geometry.indices[offset + 2])]
        let weights = barycentricWeights(a.position, b.position, c.position)
        let sum = weights.x + weights.y + weights.z
        func uv(_ a: SIMD2<Float>, _ b: SIMD2<Float>, _ c: SIMD2<Float>) -> SIMD2<Float> {
            SIMD2<Float>((weights.x * SIMD2<Double>(a) + weights.y * SIMD2<Double>(b) + weights.z * SIMD2<Double>(c)) / sum)
        }
        let secondary: SIMD2<Float>?
        if let first = a.secondaryTextureCoordinate, let second = b.secondaryTextureCoordinate,
           let third = c.secondaryTextureCoordinate {
            secondary = uv(first, second, third)
        } else {
            secondary = nil
        }
        let pa = doubleVector(a.position)
        let face = pickingUnit(simd_cross(doubleVector(b.position) - pa, doubleVector(c.position) - pa))
        let isFront = simd_dot(face, localDirection) < 0
        let interpolated = weights.x * doubleVector(a.normal) + weights.y * doubleVector(b.normal)
            + weights.z * doubleVector(c.normal)
        let local = interpolated == .zero ? face : pickingUnit(interpolated)
        let oriented = simd_dot(local, localDirection) > 0 ? -local : local
        let normalFromLocal = simd_transpose(transform.inverse)
        return MeshRayHit(
            distance: rounded, position: position, triangleIndex: triangleIndex,
            barycentric: floatVector(weights / sum), textureCoordinate: uv(a.textureCoordinate, b.textureCoordinate, c.textureCoordinate),
            secondaryTextureCoordinate: secondary, isFrontFace: isFront,
            localFaceNormal: floatVector(face), localNormal: floatVector(oriented),
            faceNormal: floatVector(pickingUnit(normalFromLocal * face)),
            normal: floatVector(pickingUnit(normalFromLocal * oriented))
        )
    }

    /// Reconstructs weights for a triangle already accepted by the distance test.
    /// Renormalize before interpolation to account for round-off on a shared edge.
    func barycentricWeights(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) -> SIMD3<Double> {
        let edgeAB = doubleVector(b) - doubleVector(a)
        let edgeAC = doubleVector(c) - doubleVector(a)
        let fromA = localOrigin - doubleVector(a)
        // Reconstruct only for reported hits or for material coverage tests.
        // Opaque visibility rays test triangles without fetching their UVs.
        let determinant = -simd_dot(localDirection, simd_cross(edgeAB, edgeAC))
        let u = simd_dot(fromA, simd_cross(localDirection, edgeAC)) / determinant
        let v = simd_dot(localDirection, simd_cross(fromA, edgeAB)) / determinant
        return SIMD3(max(0, 1 - u - v), u, v)
    }

    func sortedHits(
        _ hits: [(distance: Double, triangle: Int)], in geometry: MeshGeometry, transform: PickingAffine
    ) throws -> [MeshRayHit] {
        try hits.sorted { $0.distance == $1.distance ? $0.triangle < $1.triangle : $0.distance < $1.distance }
            .map { try hit(distance: $0.distance, triangleIndex: $0.triangle, in: geometry, transform: transform) }
    }
}

struct PickingAffine {
    let linear: simd_double3x3
    let translation: SIMD3<Double>
    let inverse: simd_double3x3

    init(_ matrix: simd_float4x4) throws {
        let (x, y, z, w) = matrix.columns
        guard x.w == 0, y.w == 0, z.w == 0, w.w == 1 else {
            throw ScenePickingError.invalidTransform
        }
        let basisX = SIMD3(x.x, x.y, x.z)
        let basisY = SIMD3(y.x, y.y, y.z)
        let basisZ = SIMD3(z.x, z.y, z.z)
        let offset = SIMD3(w.x, w.y, w.z)
        guard isFinite(basisX), isFinite(basisY), isFinite(basisZ), isFinite(offset) else {
            throw ScenePickingError.invalidTransform
        }
        let linear = simd_double3x3(columns: (doubleVector(basisX), doubleVector(basisY), doubleVector(basisZ)))
        let determinant = simd_determinant(linear)
        guard determinant.isFinite, determinant != 0 else { throw ScenePickingError.invalidTransform }
        self.linear = linear
        inverse = simd_inverse(linear)
        translation = doubleVector(offset)
    }
}

private func isFinite(_ value: SIMD3<Float>) -> Bool {
    value.x.isFinite && value.y.isFinite && value.z.isFinite
}

private func doubleVector(_ value: SIMD3<Float>) -> SIMD3<Double> {
    SIMD3(Double(value.x), Double(value.y), Double(value.z))
}

private func floatVector(_ value: SIMD3<Double>) -> SIMD3<Float> {
    SIMD3(Float(value.x), Float(value.y), Float(value.z))
}

private func pickingUnit(_ value: SIMD3<Double>) -> SIMD3<Double> {
    let scale = simd_reduce_max(simd_abs(value))
    return scale > 0 ? simd_normalize(value / scale) : .zero
}
