import Foundation
import simd

/// Parallel-transported coordinates for a swept cross-section.
struct CurveSweepFrame {
    let center: SIMD3<Double>
    let tangent: SIMD3<Double>
    let normal: SIMD3<Double>
    let binormal: SIMD3<Double>

    init(center: SIMD3<Double>, tangent: SIMD3<Double>, normal: SIMD3<Double>) {
        self.center = center
        self.tangent = tangent
        self.normal = normal
        binormal = simd_cross(tangent, normal)
    }

    func map(_ point: SIMD2<Double>) -> SIMD3<Double> { normal * point.x + binormal * point.y }

    enum Failure: Error {
        case invalidClosedPath
        case invalidSample(Int)
    }

    static func make(
        samples: [(point: SIMD3<Double>, tangent: SIMD3<Double>)], length: Double, closed: Bool
    ) throws(Failure) -> [Self] {
        let last = samples.count - 1
        let start = samples[0]
        if closed {
            guard simd_distance(start.point, samples[last].point) <= length * 1e-6,
                  simd_dot(start.tangent, samples[last].tangent) > 1 - 1e-6 else {
                throw .invalidClosedPath
            }
        }
        let axis: SIMD3<Double>
        let alignment = simd_abs(start.tangent)
        // Three.js picks the last least-aligned axis and uses a double cross.
        if alignment.z <= min(alignment.x, alignment.y) { axis = SIMD3(0, 0, 1) }
        else if alignment.y <= alignment.x { axis = SIMD3(0, 1, 0) }
        else { axis = SIMD3(1, 0, 0) }
        let reference = simd_normalize(start.tangent * simd_dot(axis, start.tangent) - axis)
        var frames = [Self(center: start.point, tangent: start.tangent, normal: reference)]
        for index in 1...last {
            let sample = closed && index == last ? start : samples[index]
            let previous = frames[index - 1]
            guard SIMD3<Float>(sample.point) != SIMD3<Float>(previous.center),
                  let normal = sweepTransportReference(previous.normal, from: previous.tangent, to: sample.tangent) else {
                throw .invalidSample(index)
            }
            frames.append(Self(center: sample.point, tangent: sample.tangent, normal: normal))
        }
        if closed {
            let end = frames[last].normal
            let twist = atan2(simd_dot(start.tangent, simd_cross(end, reference)), simd_dot(end, reference))
            for index in 1..<last {
                let frame = frames[index]
                let roll = simd_quatd(angle: twist * Double(index) / Double(last), axis: frame.tangent)
                frames[index] = Self(center: frame.center, tangent: frame.tangent, normal: roll.act(frame.normal))
            }
            frames[last] = frames[0]
        }
        return frames
    }
}
