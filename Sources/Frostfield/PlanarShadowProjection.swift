import Foundation
import simd

/// Projects an authored mesh onto an infinite receiver plane. A homogeneous
/// light with W = 0 points toward a directional light. For W != 0, divide
/// XYZ by W to get the positional light in world coordinates.
public struct PlanarShadowProjection: Sendable {
    public let plane: Plane3D
    private let normal: SIMD3<Double>
    private let constant: Double
    private let light: SIMD3<Double>
    private let lightSide: Double
    private let directional: Bool

    public init(plane: Plane3D, light: SIMD4<Float>) throws {
        guard (0..<4).allSatisfy({ light[$0].isFinite }) else { throw Failure.invalidLight }
        let coefficients = plane.homogeneousCoefficients
        let normal = SIMD3(coefficients.x, coefficients.y, coefficients.z)
        let isDirectional = light.w == 0
        let xyz = SIMD3<Double>(Double(light.x), Double(light.y), Double(light.z))
        guard !isDirectional || xyz != .zero else { throw Failure.invalidLight }
        let canonical = isDirectional ? xyz : xyz / Double(light.w)
        let offset = isDirectional ? 0 : coefficients.w
        let products = normal * canonical
        let side = simd_reduce_add(products) + offset
        guard Self.usable(side, scale: simd_reduce_add(simd_abs(products)) + abs(offset)) else {
            throw isDirectional ? Failure.parallelLight : Failure.lightOnPlane
        }
        self.plane = plane
        self.normal = normal
        constant = coefficients.w
        self.light = canonical
        lightSide = side
        directional = isDirectional
    }

    public func project(_ worldPoint: SIMD3<Float>) throws -> SIMD3<Float> {
        guard boundsFinite(worldPoint) else { throw Failure.unrepresentablePoint }
        return try projected(boundsDouble(worldPoint))
    }

    /// Applies the caster's affine world transform before projecting. The
    /// output uses world coordinates, with original indices, UVs, colors and
    /// material groups. Normals use the receiver normal. A positional caster
    /// must lie on the receiver side of the plane through the light.
    public func project(
        _ geometry: MeshGeometry, sourceToWorld: simd_float4x4 = matrix_identity_float4x4
    ) throws -> MeshGeometry {
        let (a, b, c, d) = sourceToWorld.columns
        guard planarShadowFinite(sourceToWorld), a.w == 0, b.w == 0, c.w == 0, d.w == 1 else {
            throw Failure.sourceTransform
        }
        let world = simd_double4x4(columns: (SIMD4<Double>(a), SIMD4<Double>(b), SIMD4<Double>(c), SIMD4<Double>(d)))
        let receiverNormal = plane.normal
        let vertices = try geometry.vertices.map { vertex in
            let point = world * SIMD4(boundsDouble(vertex.position), 1)
            return MeshVertex(
                position: try projected(SIMD3(point.x, point.y, point.z)), normal: receiverNormal,
                textureCoordinate: vertex.textureCoordinate, secondaryTextureCoordinate: vertex.secondaryTextureCoordinate,
                color: vertex.color
            )
        }
        return try MeshGeometry(vertices: vertices, indices: geometry.indices).withGroups(geometry.groups)
    }

    private func projected(_ point: SIMD3<Double>) throws -> SIMD3<Float> {
        let fromCaster = directional ? light : light - point
        let denominator: Double
        if directional {
            denominator = lightSide
        } else {
            let products = normal * fromCaster
            let dot = simd_reduce_add(products)
            guard Self.usable(dot, scale: simd_reduce_add(simd_abs(products))), (dot > 0) == (lightSide > 0) else {
                throw Failure.rayHorizon
            }
            denominator = dot
        }
        let planeValue = simd_dot(normal, point) + constant
        let result = SIMD3<Float>(point - fromCaster * (planeValue / denominator))
        guard boundsFinite(result) else { throw Failure.unrepresentablePoint }
        return result
    }

    private static func usable(_ value: Double, scale: Double) -> Bool {
        value.isFinite && abs(value) > 8 * Double.ulpOfOne * scale
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidLight
        case parallelLight
        case lightOnPlane
        case sourceTransform
        case rayHorizon
        case unrepresentablePoint

        public var errorDescription: String? {
            switch self {
            case .invalidLight: "Use a finite homogeneous light or a nonzero directional light vector."
            case .parallelLight: "The directional light is parallel to the receiver plane."
            case .lightOnPlane: "A positional light needs a nonzero distance from the receiver plane."
            case .sourceTransform: "The caster needs a finite affine world transform."
            case .rayHorizon: "The caster reaches or crosses the positional light's projection horizon."
            case .unrepresentablePoint: "The projected shadow position exceeds Float precision."
            }
        }
    }
}

func planarShadowFinite(_ matrix: simd_float4x4) -> Bool {
    (0..<4).allSatisfy { column in (0..<4).allSatisfy { matrix[column][$0].isFinite } }
}
