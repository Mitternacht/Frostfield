import Foundation
import simd

/// Draws expanded, back-facing hulls in the scene's color and depth attachments.
/// The initializer targets classic toon meshes. Use `outlineEffect` to outline
/// compatible scene meshes after their regular color draws.
public struct ToonOutlinePass: Equatable, Sendable {
    /// Clip-space displacement before multiplication by the vertex's clip W.
    /// On a square orthographic view, 0.01 moves a screen-aligned normal by
    /// half a percent of the viewport width. Use zero to disable the pass.
    public let thickness: Float
    /// Unassociated linear RGB, suitable for an HDR scene color target.
    public let color: SIMD3<Float>
    /// Source-over alpha in 0...1. Use zero to disable the pass.
    public let alpha: Float

    /// An override for one SceneNode ID. Nil properties inherit the pass defaults
    /// or the material opacity. Set visible to false to omit the hull.
    public struct Override: Equatable, Sendable {
        public let thickness: Float?
        public let color: SIMD3<Float>?
        public let alpha: Float?
        public let visible: Bool

        public init(
            thickness: Float? = nil, color: SIMD3<Float>? = nil, alpha: Float? = nil, visible: Bool = true
        ) throws {
            try ToonOutlinePass.validate(thickness: thickness, color: color, alpha: alpha)
            self.thickness = thickness
            self.color = color
            self.alpha = alpha
            self.visible = visible
        }
    }

    let drawsAfterScene: Bool
    private let inheritsOpacity: Bool
    private let overrides: [UUID: Override]

    public init(
        thickness: Float = 0.003, color: SIMD3<Float> = .zero, alpha: Float = 1
    ) throws {
        try self.init(thickness: thickness, color: color, alpha: alpha, drawsAfterScene: false, overrides: [:])
    }

    /// Creates the classic OutlineEffect mode for visible meshes with a depth
    /// test and stock vertex inputs. The pass draws after the regular scene
    /// color. Nil alpha uses each material's opacity. A node override can
    /// change thickness, RGB, alpha or visibility for that draw.
    public static func outlineEffect(
        thickness: Float = 0.003, color: SIMD3<Float> = .zero, alpha: Float? = nil,
        overrides: [UUID: Override] = [:]
    ) throws -> Self {
        try Self(thickness: thickness, color: color, alpha: alpha, drawsAfterScene: true, overrides: overrides)
    }

    private init(
        thickness: Float, color: SIMD3<Float>, alpha: Float?, drawsAfterScene: Bool, overrides: [UUID: Override]
    ) throws {
        try Self.validate(thickness: thickness, color: color, alpha: alpha)
        self.thickness = thickness
        self.color = color
        self.alpha = alpha ?? 1
        self.drawsAfterScene = drawsAfterScene
        inheritsOpacity = alpha == nil
        self.overrides = overrides
    }

    private static func validate(thickness: Float?, color: SIMD3<Float>?, alpha: Float?) throws {
        if let thickness, !thickness.isFinite || thickness < 0 { throw Failure.thickness }
        if let color, !(0..<3).allSatisfy({ (0...65_504).contains(color[$0]) }) { throw Failure.color }
        if let alpha, !(0...1).contains(alpha) { throw Failure.alpha }
    }

    var isEnabled: Bool {
        (thickness > 0 && alpha > 0) || overrides.values.contains {
            $0.visible && ($0.thickness ?? thickness) > 0 && ($0.alpha ?? alpha) > 0
        }
    }

    struct DrawStyle {
        let thickness: Float
        let premultiplied: SIMD4<Float>
    }

    func style(for node: UUID, materialOpacity: Float) -> DrawStyle? {
        let options = overrides[node]
        let width = options?.thickness ?? thickness
        let opacity = options?.alpha ?? (inheritsOpacity ? materialOpacity : alpha)
        guard options?.visible != false, width > 0, opacity > 0 else { return nil }
        return DrawStyle(thickness: width, premultiplied: SIMD4((options?.color ?? color) * opacity, opacity))
    }

    public enum Failure: LocalizedError, Equatable {
        case thickness, color, alpha
        case colorAttachments

        public var errorDescription: String? {
            switch self {
            case .thickness: "Toon outline thickness must be finite and nonnegative."
            case .color: "Toon outline RGB must be linear and in 0...65504."
            case .alpha: "Toon outline alpha must be in 0...1."
            case .colorAttachments: "Toon outlines need one scene color attachment."
            }
        }
    }
}
