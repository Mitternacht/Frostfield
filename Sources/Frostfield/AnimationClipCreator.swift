import Foundation
import simd

/// Factories for common clips with typed targets and times in seconds.
public enum AnimationClipCreator {
    public enum Axis: Int, Sendable {
        case x, y, z
    }

    /// Makes one full turn about a local axis, starting at the supplied orientation.
    public static func rotation(
        node: UUID, period: Float, axis: Axis = .x,
        initialRotation: simd_quatf = simd_quatf(real: 1, imag: .zero), name: String? = nil
    ) throws -> AnimationClip {
        try validateDuration(period)
        var direction = SIMD3<Float>.zero
        direction[axis.rawValue] = 1
        let times = [0, period * 0.25, period * 0.5, period * 0.75, period]
        // Quarter turns keep shortest-arc interpolation from collapsing the full turn.
        let keys = [initialRotation] + (1...3).map { step in
            initialRotation * simd_quatf(angle: Float(step) * (.pi / 2), axis: direction)
        } + [simd_quatf(vector: -initialRotation.vector)]
        let track = try AnimationTrack.quaternion(.node(node, .rotation), times: times, values: keys)
        return try AnimationClip(name: name, tracks: [track], duration: period)
    }

    /// Scales one local axis from zero to one and holds the other two axes at their supplied values.
    public static func scaleAxis(
        node: UUID, period: Float, axis: Axis = .x,
        restScale: SIMD3<Float> = SIMD3(repeating: 1), name: String? = nil
    ) throws -> AnimationClip {
        try validateDuration(period)
        var first = restScale
        var last = restScale
        first[axis.rawValue] = 0
        last[axis.rawValue] = 1
        let track = try AnimationTrack.vector(.node(node, .scale), times: [0, period], values: [first, last])
        return try AnimationClip(name: name, tracks: [track], duration: period)
    }

    /// Samples local translation offsets ten times per second. Supply a seed to repeat the clip.
    public static func shake(
        node: UUID, duration: Float, amplitude: SIMD3<Float>, center: SIMD3<Float> = .zero,
        seed: UInt64? = nil, name: String? = nil
    ) throws -> AnimationClip {
        let times = try randomKeyTimes(duration)
        var random = SceneSeedGenerator(state: seed ?? UInt64.random(in: .min ... .max))
        let keys = times.map { _ -> SIMD3<Float> in
            let unit = SIMD3(random.nextUnit(), random.nextUnit(), random.nextUnit())
            return center + (2 * unit - 1) * amplitude
        }
        let track = try AnimationTrack.vector(.node(node, .translation), times: times, values: keys)
        return try AnimationClip(name: name, tracks: [track], duration: duration)
    }

    /// Samples a uniform local scale between zero and the supplied maximum ten times per second.
    public static func pulsation(
        node: UUID, duration: Float, maximumScale: Float, seed: UInt64? = nil, name: String? = nil
    ) throws -> AnimationClip {
        guard maximumScale.isFinite, maximumScale >= 0 else { throw Failure.scale }
        let times = try randomKeyTimes(duration)
        var random = SceneSeedGenerator(state: seed ?? UInt64.random(in: .min ... .max))
        let keys = times.map { _ in SIMD3(repeating: random.nextUnit() * maximumScale) }
        let track = try AnimationTrack.vector(.node(node, .scale), times: times, values: keys)
        return try AnimationClip(name: name, tracks: [track], duration: duration)
    }

    /// Shows the node for the first half of the clip, hides it for the second, and shows it at the end.
    public static func visibility(node: UUID, duration: Float, name: String? = nil) throws -> AnimationClip {
        try validateDuration(duration)
        let track = try AnimationTrack.boolean(
            .node(node, .isVisible), times: [0, duration * 0.5, duration], values: [true, false, true]
        )
        return try AnimationClip(name: name, tracks: [track], duration: duration)
    }

    /// Interpolates normalized linear RGB stops at even intervals and preserves live alpha.
    /// Prepare the target material with `withLiveFactors()`.
    public static func materialColor(
        _ material: PhysicallyBasedMaterial.LiveFactors, duration: Float, colors: [SIMD3<Float>], name: String? = nil
    ) throws -> AnimationClip {
        try validateDuration(duration)
        guard !colors.isEmpty,
              colors.allSatisfy({ all($0 .>= .zero) && all($0 .<= SIMD3(repeating: 1)) })
        else { throw Failure.colors }
        let times: [Float] = colors.count == 1 ? [0] : colors.indices.map { index in
            Float(Double(duration) * Double(index) / Double(colors.count - 1))
        }
        let track = try AnimationTrack.color(.material(material, .baseColorRGB), times: times, values: colors)
        return try AnimationClip(name: name, tracks: [track], duration: duration)
    }

    private static func validateDuration(_ duration: Float) throws {
        guard duration.isFinite, duration > 0 else { throw AnimationClip.Failure.invalidDuration }
    }

    private static func randomKeyTimes(_ duration: Float) throws -> [Float] {
        try validateDuration(duration)
        let count = (Double(duration) * 10).rounded(.up)
        guard count <= 1_000_000 else { throw Failure.sampleCount }
        // Round times to track precision before comparing with the clip boundary.
        var times = (0..<Int(count)).map { Float($0) / 10 }
        if let last = times.last, last >= duration { times.removeLast() }
        return times
    }

    public enum Failure: LocalizedError, Equatable {
        case colors, scale, sampleCount

        public var errorDescription: String? {
            switch self {
            case .colors: "Supply at least one normalized linear RGB color."
            case .scale: "Use a finite nonnegative maximum scale."
            case .sampleCount: "A procedural shake or pulse supports at most one million keys."
            }
        }
    }
}
