import Foundation
import simd

/// A fixed-base, unconstrained 3D joint chain solved by forward and backward
/// reaching. Each pass moves points while preserving the input bone lengths.
/// The result applies the solved directions as local rotations to a graph value.
public struct FABRIKChain: Sendable {
    /// The base, each direct child joint, and the final effector, in that order.
    public let joints: [UUID]
    /// The maximum number of forward and backward passes.
    public let iterations: Int
    /// The stopping distance in world units.
    public let tolerance: Float

    public init(joints: [UUID], iterations: Int = 32, tolerance: Float = 1e-5) throws {
        guard joints.count >= 2, Set(joints).count == joints.count else { throw Failure.invalidJoints }
        guard iterations > 0, tolerance.isFinite, tolerance >= 0 else { throw Failure.invalidSettings }
        self.joints = joints
        self.iterations = iterations
        self.tolerance = tolerance
    }

    public struct Solution {
        public let graph: SceneGraph
        /// Distance from the returned effector's origin to the target, in world units.
        public let distance: Double
        /// Zero when the input pose already meets tolerance, or one for a direct
        /// solution at the minimum or maximum reach of the chain.
        public let iterations: Int
        public let reachedTarget: Bool
    }

    /// Poses a contiguous chain toward a world-space target. The base joint's
    /// pivot stays fixed, or its origin if it has no pivot. The effector's
    /// origin is the endpoint. Only rotating joints' local rotations change.
    /// The chain and its ancestors require positive uniform scale and inherited
    /// orientation. For targets outside the reachable distance interval, returns
    /// the closest point along the direction from the base to the target.
    public func solve(in graph: SceneGraph, target: SIMD3<Float>) throws -> Solution {
        guard boundsFinite(target) else { throw Failure.invalidTarget }
        let effectorID = joints[joints.count - 1]
        guard let effector = graph.node(withID: effectorID), let ancestors = graph.ancestors(of: effectorID) else {
            throw Failure.missingEffector(effectorID)
        }
        let path = Array(ancestors.reversed()) + [effector]
        let chain = Array(path.suffix(joints.count))
        guard chain.map(\.id) == joints else { throw Failure.disconnectedChain }

        let localRotations = try path.map { node -> simd_quatd in
            guard node.orientationMode == .inherited else { throw Failure.worldOrientedNode(node.id) }
            let rotation = SIMD4<Double>(node.transform.rotation.vector)
            let scale = node.transform.scale
            guard boundsFinite(node.transform.translation), boundsFinite(node.pivot ?? .zero),
                  abs(simd_length_squared(rotation) - 1) <= 0.0001 else {
                throw Failure.invalidTransform(node.id)
            }
            guard scale.x.isFinite, scale.x > 0, scale == SIMD3(repeating: scale.x) else {
                throw Failure.nonuniformScale(node.id)
            }
            return simd_normalize(simd_quatd(vector: rotation))
        }

        let prefixCount = path.count - chain.count
        var parent = matrix_identity_double4x4
        var parentRotation = Self.identity
        var worldScale = 1.0
        for index in 0..<prefixCount {
            let node = path[index]
            parent = parent * node.transform.doubleMatrix(around: node.pivot)
            parentRotation = simd_normalize(parentRotation * localRotations[index])
            worldScale *= Double(node.transform.scale.x)
        }
        let base = SIMD3<Double>(chain[0].transform.translation) + SIMD3<Double>(chain[0].pivot ?? .zero)
        // Work near the base in its parent's frame. The uniform ancestor scale
        // converts the solver's distances to world units.
        let goal = parentRotation.inverse.act(SIMD3<Double>(target) - Self.xyz(parent[3])) / worldScale - base
        guard worldScale.isNormal, allFinite(goal) else { throw Failure.unrepresentablePose }

        let boneCount = chain.count - 1
        let lastAnchor = Self.xyz(effector.transform.doubleMatrix(around: effector.pivot)[3])
        var offsets: [SIMD3<Double>] = []
        var directions: [SIMD3<Double>] = []
        var lengths: [Double] = []
        var initial: [SIMD3<Double>] = [.zero]
        var rotation = Self.identity
        var scale = 1.0
        for index in 0..<boneCount {
            let joint = chain[index]
            let child = chain[index + 1]
            let childAnchor = index + 1 == boneCount ? lastAnchor
                : SIMD3<Double>(child.transform.translation) + SIMD3<Double>(child.pivot ?? .zero)
            let offset = childAnchor - SIMD3<Double>(joint.pivot ?? .zero)
            let offsetLength = simd_length(offset)
            guard offsetLength > 0 else { throw Failure.zeroLengthBone(joint.id) }
            rotation = simd_normalize(rotation * localRotations[prefixCount + index])
            scale *= Double(joint.transform.scale.x)
            let length = offsetLength * scale
            let direction = rotation.act(offset / offsetLength)
            let point = initial[index] + direction * length
            guard length.isNormal, allFinite(point), point != initial[index] else {
                throw Failure.unrepresentablePose
            }
            offsets.append(offset / offsetLength)
            directions.append(direction)
            lengths.append(length)
            initial.append(point)
        }

        let localTolerance = Double(tolerance) / worldScale
        let (points, passes) = try solvePoints(
            initial, directions: directions, lengths: lengths, target: goal, tolerance: localTolerance
        )

        var rotations: [UUID: simd_quatf] = [:]
        if points != initial {
            var parent = Self.identity
            for index in 0..<boneCount {
                let rest = localRotations[prefixCount + index]
                let segment = Self.direction(points[index + 1] - points[index], fallback: directions[index])
                let desired = parent.inverse.act(segment)
                let swing = Self.shortestArc(from: rest.act(offsets[index]), to: desired)
                let solved = simd_normalize(swing * rest)
                let local = simd_normalize(simd_quatf(vector: SIMD4<Float>(solved.vector)))
                rotations[chain[index].id] = local
                parent = simd_normalize(parent * simd_quatd(vector: SIMD4<Double>(local.vector)))
            }
        }
        let posed = try graph.updatingTransforms { id, transform in
            if let rotation = rotations[id] { transform.rotation = rotation }
        }
        var world = parent
        for joint in chain {
            var transform = joint.transform
            if let rotation = rotations[joint.id] { transform.rotation = rotation }
            world = world * transform.doubleMatrix(around: joint.pivot)
        }
        let distance = simd_distance(Self.xyz(world[3]), SIMD3<Double>(target))
        guard distance.isFinite else { throw Failure.unrepresentablePose }
        return Solution(graph: posed, distance: distance, iterations: passes, reachedTarget: distance <= Double(tolerance))
    }

    private func solvePoints(
        _ initial: [SIMD3<Double>], directions: [SIMD3<Double>], lengths: [Double],
        target: SIMD3<Double>, tolerance: Double
    ) throws -> ([SIMD3<Double>], Int) {
        let end = lengths.count
        var bestDistance = simd_distance(initial[end], target)
        guard bestDistance.isFinite else { throw Failure.unrepresentablePose }
        if bestDistance <= tolerance { return (initial, 0) }

        let longest = lengths.indices.max { lengths[$0] < lengths[$1] }!
        let maximum = lengths.reduce(0, +)
        let minimum = max(0, lengths[longest] - (maximum - lengths[longest]))
        let targetDistance = simd_length(target)
        guard maximum.isFinite, targetDistance.isFinite else { throw Failure.unrepresentablePose }
        let aim = Self.direction(target, fallback: directions[0])
        var points = initial
        if targetDistance >= maximum || targetDistance <= minimum && minimum > 0 {
            // Outside the outer radius, extend every bone. Inside a nonzero
            // inner radius, point the longest toward the target and fold the rest back.
            for index in lengths.indices {
                let sign: Double = targetDistance >= maximum || index == longest ? 1 : -1
                points[index + 1] = points[index] + aim * (sign * lengths[index])
            }
            return (points, 1)
        }

        if end > 1, directions.allSatisfy({ simd_length_squared(simd_cross($0, aim)) < 1e-14 }) {
            // A straight chain and collinear interior target cannot discover a
            // bend on their own. Seed a consistent bend, then restore lengths
            // in the first forward and backward pass.
            points[1] += Self.perpendicular(to: aim) * (0.25 * min(lengths[0], lengths[1]))
        }

        var best = initial
        var previous = Double.infinity
        let resolution = maximum * (8 * Double.ulpOfOne)
        var passes = 0
        while passes < iterations, bestDistance > tolerance {
            points[end] = target
            for index in lengths.indices.reversed() {
                let direction = Self.direction(points[index] - points[index + 1], fallback: -directions[index])
                points[index] = points[index + 1] + direction * lengths[index]
            }
            points[0] = .zero
            for index in lengths.indices {
                let direction = Self.direction(points[index + 1] - points[index], fallback: directions[index])
                points[index + 1] = points[index] + direction * lengths[index]
            }
            passes += 1
            let distance = simd_distance(points[end], target)
            guard distance.isFinite else { throw Failure.unrepresentablePose }
            if distance < bestDistance {
                bestDistance = distance
                best = points
            }
            if abs(distance - previous) <= resolution { break }
            previous = distance
        }
        return (best, passes)
    }

    private static let identity = simd_quatd(ix: 0, iy: 0, iz: 0, r: 1)

    private static func xyz(_ vector: SIMD4<Double>) -> SIMD3<Double> {
        SIMD3(vector.x, vector.y, vector.z)
    }

    private static func direction(_ vector: SIMD3<Double>, fallback: SIMD3<Double>) -> SIMD3<Double> {
        let length = simd_length(vector)
        return length > 0 && length.isFinite ? vector / length : fallback
    }

    private static func perpendicular(to direction: SIMD3<Double>) -> SIMD3<Double> {
        let axis: SIMD3<Double> = abs(direction.x) < 0.9 ? SIMD3(1, 0, 0) : SIMD3(0, 1, 0)
        return simd_normalize(simd_cross(direction, axis))
    }

    private static func shortestArc(from source: SIMD3<Double>, to destination: SIMD3<Double>) -> simd_quatd {
        let cross = simd_cross(source, destination)
        let sine = simd_length(cross)
        let cosine = simd_dot(source, destination)
        if sine > 1e-14 { return simd_quatd(angle: atan2(sine, cosine), axis: cross / sine) }
        return cosine >= 0 ? identity : simd_quatd(vector: SIMD4(perpendicular(to: source), 0))
    }

    private func allFinite(_ vector: SIMD3<Double>) -> Bool {
        vector.x.isFinite && vector.y.isFinite && vector.z.isFinite
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidJoints
        case invalidSettings
        case invalidTarget
        case missingEffector(UUID)
        case disconnectedChain
        case worldOrientedNode(UUID)
        case invalidTransform(UUID)
        case nonuniformScale(UUID)
        case zeroLengthBone(UUID)
        case unrepresentablePose

        public var errorDescription: String? {
            switch self {
            case .invalidJoints: "FABRIK needs at least two distinct joint IDs."
            case .invalidSettings: "Use positive iterations and a finite nonnegative tolerance."
            case .invalidTarget: "The FABRIK target must be finite."
            case .missingEffector(let id): "The scene graph has no FABRIK effector \(id)."
            case .disconnectedChain: "The FABRIK joints must be consecutive parent and child nodes, starting at the base."
            case .worldOrientedNode(let id): "FABRIK needs inherited orientation. Node \(id) uses world orientation."
            case .invalidTransform(let id): "Joint \(id) needs a finite translation and pivot and a unit rotation."
            case .nonuniformScale(let id): "Joint \(id) needs a finite positive uniform scale."
            case .zeroLengthBone(let id): "The FABRIK bone starting at \(id) needs distinct endpoints."
            case .unrepresentablePose: "The FABRIK pose exceeds the representable coordinate range."
            }
        }
    }
}
