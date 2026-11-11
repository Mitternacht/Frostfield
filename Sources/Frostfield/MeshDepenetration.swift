import Foundation
import simd

/// The translation produced by iterative resolution against a static mesh.
public struct MeshCollisionResolution: Equatable, Sendable {
    /// Add to the original collider's position or capsule endpoints.
    public let translation: SIMD3<Float>
    /// True when every remaining penetration is within the requested tolerance.
    /// False when the iteration budget expires or the next translation cannot
    /// move the collider at Float precision.
    public let isResolved: Bool
    public let iterations: Int

    public enum Failure: LocalizedError, Equatable {
        case invalidSettings
        case unrepresentableTranslation

        public var errorDescription: String? {
            switch self {
            case .invalidSettings: "Use 1...64 collision iterations and a finite, nonnegative tolerance."
            case .unrepresentableTranslation: "The collision translation exceeds the Float range."
            }
        }
    }
}

public extension MeshBVH {
    /// Resolves penetration in geometry coordinates by applying the deepest
    /// contact and querying again after each translation. Ties prefer the lower
    /// triangle index. Returns nil if every original penetration is within the
    /// tolerance, including tangencies.
    ///
    /// `maximumIterations` is 1...64 and `tolerance` is a finite, nonnegative
    /// distance in geometry units. An unresolved result is a partial translation,
    /// such as when two surfaces are closer together than the capsule's diameter.
    func depenetration(
        for capsule: Capsule, maximumIterations: Int = 8, tolerance: Float = 1e-5
    ) throws -> MeshCollisionResolution? {
        try depenetration(
            capsule, maximumIterations: maximumIterations, tolerance: tolerance,
            translate: { try $0.translated(by: $1) }, contacts: { try contacts(intersecting: $0) }
        )
    }

    /// Resolves sphere penetration with the same limits and result semantics
    /// as the capsule overload.
    func depenetration(
        for sphere: BoundingSphere, maximumIterations: Int = 8, tolerance: Float = 1e-5
    ) throws -> MeshCollisionResolution? {
        try depenetration(
            sphere, maximumIterations: maximumIterations, tolerance: tolerance,
            translate: { try BoundingSphere(center: $0.center + $1, radius: $0.radius) },
            contacts: { try contacts(intersecting: $0) }
        )
    }

    /// Resolves an axis-aligned box's penetration with the same limits and
    /// result semantics as the capsule overload.
    func depenetration(
        for box: AxisAlignedBox, maximumIterations: Int = 8, tolerance: Float = 1e-5
    ) throws -> MeshCollisionResolution? {
        try depenetration(
            box, maximumIterations: maximumIterations, tolerance: tolerance,
            translate: { try $0.translated(by: $1) }, contacts: { try contacts(intersecting: $0) }
        )
    }

    /// Resolves an oriented box's penetration by translation, preserving its orientation.
    func depenetration(
        for box: OrientedBoundingBox, maximumIterations: Int = 8, tolerance: Float = 1e-5
    ) throws -> MeshCollisionResolution? {
        try depenetration(
            box, maximumIterations: maximumIterations, tolerance: tolerance,
            translate: { try $0.translated(by: $1) }, contacts: { try contacts(intersecting: $0) }
        )
    }

    private func depenetration<Collider: Equatable>(
        _ initial: Collider, maximumIterations: Int, tolerance: Float,
        translate: (Collider, SIMD3<Float>) throws -> Collider,
        contacts: (Collider) throws -> [MeshCollisionContact]
    ) throws -> MeshCollisionResolution? {
        guard (1...64).contains(maximumIterations), tolerance.isFinite, tolerance >= 0 else {
            throw MeshCollisionResolution.Failure.invalidSettings
        }
        var collider = initial
        var displacement = SIMD3<Double>.zero
        var translation = SIMD3<Float>.zero
        var iterations = 0
        while true {
            var deepest: MeshCollisionContact?
            for contact in try contacts(collider) where contact.depth > (deepest?.depth ?? tolerance) {
                deepest = contact
            }
            guard let deepest else {
                return iterations == 0 ? nil : .init(translation: translation, isResolved: true, iterations: iterations)
            }
            guard iterations < maximumIterations else {
                return .init(translation: translation, isResolved: false, iterations: iterations)
            }
            displacement += Double(deepest.depth) * boundsDouble(deepest.normal)
            let nextTranslation = SIMD3<Float>(displacement)
            guard boundsFinite(nextTranslation) else { throw MeshCollisionResolution.Failure.unrepresentableTranslation }
            let nextCollider = try translate(initial, nextTranslation)
            guard nextCollider != collider else {
                return .init(translation: translation, isResolved: false, iterations: iterations)
            }
            collider = nextCollider
            translation = nextTranslation
            iterations += 1
        }
    }
}
