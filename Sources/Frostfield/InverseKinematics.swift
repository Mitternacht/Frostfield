import Foundation
import simd

/// A joint chain solved by cyclic coordinate descent, like the Three.js `CCDIKSolver`
/// addon. Each iteration turns every link, from the effector's parent toward the
/// root, so the effector moves toward the target, then applies the link's
/// constraint. Solving reads and returns scene graph values; the source graph is
/// unchanged.
public struct CCDIKChain: Sendable {
    /// A limit on one link's local rotation.
    public enum Constraint: Equatable, Sendable {
        /// Any rotation.
        case free
        /// A rotation about one axis of the parent's frame by an angle within
        /// `minimumAngle...maximumAngle` radians, like Three.js `limitation` with angle
        /// limits. The local rotation becomes exactly `rotation(angle, axis)`; each
        /// step turns the hinge by the signed angle between the effector and target
        /// directions projected onto the hinge plane.
        case hinge(axis: SIMD3<Float>, minimumAngle: Float, maximumAngle: Float)
        /// XYZ Euler angles of the local rotation clamped componentwise, like Three.js
        /// `rotationMin` and `rotationMax`. The rotation is `Rx · Ry · Rz`.
        case euler(minimum: SIMD3<Float>, maximum: SIMD3<Float>)
    }

    public struct Link: Equatable, Sendable {
        public var nodeID: UUID
        public var constraint: Constraint

        public init(nodeID: UUID, constraint: Constraint = .free) {
            self.nodeID = nodeID
            self.constraint = constraint
        }
    }

    /// The node whose world position should reach the target.
    public let effectorID: UUID
    /// Ancestors of the effector, from its parent toward the root, like Three.js `links`.
    public let links: [Link]
    /// The most passes over the links.
    public let iterations: Int
    /// The largest rotation of one link in one step, in radians, like Three.js `maxAngle`.
    public let maximumStepAngle: Float
    /// Solving stops once the effector is this close to the target, in world units.
    public let tolerance: Float
    /// Interpolates each link from its input rotation toward the solution, like Three.js `blendFactor`.
    public let blendFactor: Float

    /// Links must be distinct, hinge axes finite and nonzero, angle limits ordered,
    /// iterations positive, the step angle positive, the tolerance nonnegative and
    /// the blend factor in zero to one.
    public init(
        effectorID: UUID, links: [Link], iterations: Int = 32, maximumStepAngle: Float = .pi,
        tolerance: Float = 1e-5, blendFactor: Float = 1
    ) throws {
        guard !links.isEmpty, Set(links.map(\.nodeID)).count == links.count,
              !links.contains(where: { $0.nodeID == effectorID }) else { throw Failure.invalidLinks }
        for link in links {
            switch link.constraint {
            case .free: break
            case .hinge(let axis, let minimum, let maximum):
                let length = simd_length(axis)
                guard length.isFinite, length > 0, minimum.isFinite, maximum.isFinite, minimum <= maximum else {
                    throw Failure.invalidConstraint(link.nodeID)
                }
            case .euler(let minimum, let maximum):
                guard (0..<3).allSatisfy({ minimum[$0].isFinite && maximum[$0].isFinite && minimum[$0] <= maximum[$0] }) else {
                    throw Failure.invalidConstraint(link.nodeID)
                }
            }
        }
        guard iterations > 0, maximumStepAngle.isFinite, maximumStepAngle > 0, tolerance.isFinite, tolerance >= 0,
              blendFactor.isFinite, (0...1).contains(blendFactor) else {
            throw Failure.invalidSettings
        }
        self.effectorID = effectorID
        self.links = links
        self.iterations = iterations
        self.maximumStepAngle = maximumStepAngle
        self.tolerance = tolerance
        self.blendFactor = blendFactor
    }

    /// The solved graph, the effector's remaining distance to the target, and the passes used.
    public struct Solution {
        public let graph: SceneGraph
        public let distance: Float
        public let iterations: Int
    }

    /// Turns the links so the effector approaches a world-space target and returns
    /// the posed graph. Only the links' local rotations change. Every link must be an
    /// ancestor of the effector. Throws `Failure.missingNode`, `notAncestor` or
    /// `invalidTarget`, and the graph's validation errors.
    public func solve(in graph: SceneGraph, target: SIMD3<Float>) throws -> Solution {
        guard target.x.isFinite, target.y.isFinite, target.z.isFinite else { throw Failure.invalidTarget }
        let path = try ancestry(in: graph)
        if let node = path.dropFirst().first(where: { $0.orientationMode == .world }) {
            throw Failure.worldOrientedNode(node.id)
        }
        var locals = path.map(\.transform)
        let original = locals
        let goal = SIMD3<Double>(target)
        let linkPositions = try links.map { link -> Int in
            guard let index = path.firstIndex(where: { $0.id == link.nodeID }), index < path.count - 1 else {
                throw Failure.notAncestor(link.nodeID)
            }
            return index
        }

        // World rotations and matrices along the path, recomputed below a changed link.
        var rotations = [simd_quatd](repeating: simd_quatd(ix: 0, iy: 0, iz: 0, r: 1), count: path.count)
        var matrices = [simd_double4x4](repeating: matrix_identity_double4x4, count: path.count)
        func update(from start: Int) {
            for index in start..<path.count {
                let local = locals[index]
                let rotation = simd_quatd(vector: SIMD4<Double>(local.rotation.vector))
                let columns = local.matrix(around: path[index].pivot).columns
                let matrix = simd_double4x4(columns: (
                    SIMD4<Double>(columns.0), SIMD4<Double>(columns.1), SIMD4<Double>(columns.2), SIMD4<Double>(columns.3)
                ))
                rotations[index] = index == 0 ? rotation : simd_normalize(rotations[index - 1] * rotation)
                matrices[index] = index == 0 ? matrix : matrices[index - 1] * matrix
            }
        }
        func position(_ index: Int) -> SIMD3<Double> {
            let column = matrices[index].columns.3
            return SIMD3(column.x, column.y, column.z)
        }
        update(from: 0)
        let effector = path.count - 1
        var distance = simd_distance(position(effector), goal)
        var passes = 0

        while passes < iterations, distance > Double(tolerance) {
            passes += 1
            var rotated = false
            for (link, index) in zip(links, linkPositions) {
                let pivot = path[index].pivot ?? .zero
                let joint = matrices[index] * SIMD4<Double>(SIMD3<Double>(pivot), 1)
                let origin = SIMD3(joint.x, joint.y, joint.z)
                let toEffector = position(effector) - origin
                let toTarget = goal - origin
                guard simd_length(toEffector) > 1e-12, simd_length(toTarget) > 1e-12 else { continue }
                let parent = index == 0 ? simd_quatd(ix: 0, iy: 0, iz: 0, r: 1) : rotations[index - 1]
                var local = simd_quatd(vector: SIMD4<Double>(locals[index].rotation.vector))
                switch link.constraint {
                case .hinge(let axisValue, let minimum, let maximum):
                    // Work in the parent's frame, where the hinge axis is fixed.
                    let axis = simd_normalize(SIMD3<Double>(axisValue))
                    let effectorDirection = parent.inverse.act(toEffector)
                    let targetDirection = parent.inverse.act(toTarget)
                    let e = effectorDirection - axis * simd_dot(effectorDirection, axis)
                    let t = targetDirection - axis * simd_dot(targetDirection, axis)
                    guard simd_length(e) > 1e-12, simd_length(t) > 1e-12 else { continue }
                    var step = atan2(simd_dot(axis, simd_cross(e, t)), simd_dot(e, t))
                    step = min(max(step, -Double(maximumStepAngle)), Double(maximumStepAngle))
                    let current = Self.twistAngle(of: local, about: axis)
                    let angle = min(max(current + step, Double(minimum)), Double(maximum))
                    let next = simd_quatd(angle: angle, axis: axis)
                    if abs(angle - current) < 1e-12, abs(abs(simd_dot(next.vector, local.vector)) - 1) < 1e-12 { continue }
                    local = next
                case .free, .euler:
                    // Three.js turns the link in its own frame by the angle between the directions.
                    let linkRotation = rotations[index]
                    let e = simd_normalize(linkRotation.inverse.act(toEffector))
                    let t = simd_normalize(linkRotation.inverse.act(toTarget))
                    var angle = acos(min(max(simd_dot(e, t), -1), 1))
                    guard angle >= 1e-9 else { continue }
                    angle = min(angle, Double(maximumStepAngle))
                    let cross = simd_cross(e, t)
                    let axis: SIMD3<Double>
                    if simd_length(cross) > 1e-12 {
                        axis = simd_normalize(cross)
                    } else {
                        // Opposite directions: any perpendicular axis turns toward the target.
                        let helper: SIMD3<Double> = abs(e.x) < 0.9 ? SIMD3(1, 0, 0) : SIMD3(0, 1, 0)
                        axis = simd_normalize(simd_cross(e, helper))
                    }
                    local = simd_normalize(local * simd_quatd(angle: angle, axis: axis))
                    if case .euler(let minimum, let maximum) = link.constraint {
                        local = Self.clampEuler(local, minimum: SIMD3<Double>(minimum), maximum: SIMD3<Double>(maximum))
                    }
                }
                locals[index].rotation = simd_quatf(vector: SIMD4<Float>(local.vector))
                update(from: index)
                rotated = true
            }
            distance = simd_distance(position(effector), goal)
            if !rotated { break }
        }

        if blendFactor < 1 {
            for index in linkPositions {
                locals[index].rotation = simd_normalize(simd_slerp(original[index].rotation, locals[index].rotation, blendFactor))
            }
            update(from: 0)
            distance = simd_distance(position(effector), goal)
        }

        let changes = Dictionary(uniqueKeysWithValues: linkPositions.map { (path[$0].id, locals[$0].rotation) })
        let solved = try graph.updatingTransforms { id, transform in
            if let rotation = changes[id] { transform.rotation = rotation }
        }
        return Solution(graph: solved, distance: Float(distance), iterations: passes)
    }

    /// The effector and its ancestors, root first.
    private func ancestry(in graph: SceneGraph) throws -> [SceneNode] {
        var pending = graph.roots.reversed().map { (node: $0, depth: 0) }
        var stack: [SceneNode] = []
        while let (node, depth) = pending.popLast() {
            stack.removeLast(stack.count - depth)
            stack.append(node)
            if node.id == effectorID { return stack }
            for child in node.children.reversed() { pending.append((child, depth + 1)) }
        }
        throw Failure.missingNode(effectorID)
    }

    /// The signed angle of a rotation's twist about a unit axis, in (-π, π].
    static func twistAngle(of rotation: simd_quatd, about axis: SIMD3<Double>) -> Double {
        let projection = simd_dot(rotation.imag, axis)
        var angle = 2 * atan2(projection, rotation.real)
        if angle > .pi { angle -= 2 * .pi } else if angle <= -.pi { angle += 2 * .pi }
        return angle
    }

    /// XYZ Euler angles with `R = Rx · Ry · Rz`, as Three.js `Euler.setFromRotationMatrix`.
    static func eulerAngles(_ rotation: simd_quatd) -> SIMD3<Double> {
        let m = simd_double3x3(rotation)
        let m13 = m.columns.2.x
        let y = asin(min(max(m13, -1), 1))
        if abs(m13) < 0.9999999 {
            return SIMD3(atan2(-m.columns.2.y, m.columns.2.z), y, atan2(-m.columns.1.x, m.columns.0.x))
        }
        return SIMD3(atan2(m.columns.1.z, m.columns.1.y), y, 0)
    }

    static func rotation(eulerAngles angles: SIMD3<Double>) -> simd_quatd {
        simd_quatd(angle: angles.x, axis: SIMD3(1, 0, 0)) * simd_quatd(angle: angles.y, axis: SIMD3(0, 1, 0))
            * simd_quatd(angle: angles.z, axis: SIMD3(0, 0, 1))
    }

    private static func clampEuler(_ rotation: simd_quatd, minimum: SIMD3<Double>, maximum: SIMD3<Double>) -> simd_quatd {
        var angles = eulerAngles(rotation)
        let center = (minimum + maximum) / 2
        for axis in 0..<3 {
            // Wrap to within π of the limits' center, then clamp. Three.js clamps the decomposed
            // angle directly, so an angle just past ±π can jump to the far limit there.
            var offset = (angles[axis] - center[axis] + .pi).truncatingRemainder(dividingBy: 2 * .pi)
            if offset < 0 { offset += 2 * .pi }
            angles[axis] = min(max(center[axis] + offset - .pi, minimum[axis]), maximum[axis])
        }
        return simd_normalize(Self.rotation(eulerAngles: angles))
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidLinks
        case invalidConstraint(UUID)
        case invalidSettings
        case invalidTarget
        case missingNode(UUID)
        case notAncestor(UUID)
        case worldOrientedNode(UUID)

        public var errorDescription: String? {
            switch self {
            case .invalidLinks: "An IK chain needs at least one distinct link that is not the effector."
            case .invalidConstraint(let id): "Link \(id) needs a finite nonzero hinge axis and ordered, finite limits."
            case .invalidSettings:
                "Use positive iterations and step angle, a finite nonnegative tolerance, and a blend factor in zero to one."
            case .invalidTarget: "The IK target must be finite."
            case .missingNode(let id): "The scene graph has no IK effector \(id)."
            case .notAncestor(let id): "IK link \(id) is not an ancestor of the effector."
            case .worldOrientedNode(let id): "CCD IK needs inherited orientation along the chain. Node \(id) uses world orientation."
            }
        }
    }
}
