import Foundation

/// Direction-independent diffuse irradiance for raster materials.
/// Attach it to a scene node to use the node's visibility and camera layers.
public struct AmbientLight: Hashable, Sendable {
    /// Linear RGB components in zero to one.
    public let color: SIMD3<Float>
    /// Nonnegative scale for the light's irradiance.
    public let intensity: Float

    public init(color: SIMD3<Float> = SIMD3(repeating: 1), intensity: Float = 1) throws {
        guard (0..<3).allSatisfy({ (0...1).contains(color[$0]) }) else { throw Failure.color }
        guard intensity.isFinite, intensity >= 0 else { throw Failure.intensity }
        self.color = color
        self.intensity = intensity
    }

    public enum Failure: LocalizedError, Equatable {
        case color
        case intensity
        case irradianceRange

        public var errorDescription: String? {
            switch self {
            case .color: "Ambient light color components must be in zero to one."
            case .intensity: "Ambient light intensity must be finite and nonnegative."
            case .irradianceRange: "The scene's ambient irradiance exceeds the supported Float range."
            }
        }
    }
}
