import Foundation
import simd

// Lookup, traversal, coordinate conversion and editing for the value-type scene tree,
// covering Three.js `Object3D.getObjectById/ByName/ByProperty`, `traverse`,
// `traverseVisible`, `traverseAncestors`, `localToWorld`, `worldToLocal`,
// `getWorldPosition`, `lookAt`, `add`, `remove`, `clear` and `attach`.
//
// Nodes have no parent pointers, so every operation walks from the roots. Walks
// are iterative, so deep chains cannot overflow the stack.

/// Which local axis `lookAt` turns toward the target.
public enum SceneForwardAxis: Sendable {
    /// Local −Z faces the target, as Frostfield cameras, lights and Three.js cameras do.
    case negativeZ
    /// Local +Z faces the target, as Three.js `Object3D.lookAt` does for meshes.
    case positiveZ
}

public extension SceneTransform {
    /// Splits an affine matrix into translation, unit rotation and scale.
    ///
    /// A mirrored matrix gets a negative X scale. Throws when the matrix is not
    /// finite, not affine, singular, or sheared beyond `tolerance` (the largest
    /// allowed cosine between scaled axes), since TRS cannot represent shear.
    init(decomposing matrix: simd_float4x4, tolerance: Double = 1e-4) throws {
        let (c0, c1, c2, c3) = matrix.columns
        guard [c0, c1, c2, c3].allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite }),
              c0.w == 0, c1.w == 0, c2.w == 0, c3.w == 1 else {
            throw SceneGraph.EditingError.unrepresentableTransform
        }
        func double(_ column: SIMD4<Float>) -> SIMD3<Double> { SIMD3(Double(column.x), Double(column.y), Double(column.z)) }
        let x = double(c0), y = double(c1), z = double(c2)
        let lengths = SIMD3(simd_length(x), simd_length(y), simd_length(z))
        guard lengths.min() > 0, lengths.max().isFinite else { throw SceneGraph.EditingError.unrepresentableTransform }
        var axisX = x / lengths.x
        let axisY = y / lengths.y, axisZ = z / lengths.z
        let determinant = simd_dot(simd_cross(axisX, axisY), axisZ)
        guard abs(simd_dot(axisX, axisY)) <= tolerance, abs(simd_dot(axisX, axisZ)) <= tolerance,
              abs(simd_dot(axisY, axisZ)) <= tolerance, abs(abs(determinant) - 1) <= 3 * tolerance else {
            throw SceneGraph.EditingError.unrepresentableTransform
        }
        var scaleX = lengths.x
        if determinant < 0 {
            axisX = -axisX
            scaleX = -scaleX
        }
        let rotation = simd_quatd(simd_double3x3(columns: (axisX, axisY, simd_cross(axisX, axisY))))
        let unit = simd_normalize(rotation.vector)
        self.init(
            translation: SIMD3(c3.x, c3.y, c3.z),
            rotation: simd_quatf(vector: SIMD4<Float>(Float(unit.x), Float(unit.y), Float(unit.z), Float(unit.w))),
            scale: SIMD3(Float(scaleX), Float(lengths.y), Float(lengths.z))
        )
    }

    /// Turns the transform in place so `forward` faces `target`, keeping local +Y as
    /// close to `up` as possible, like Three.js `lookAt`. The target and up vector
    /// are in the same space as the translation (the parent's space). Translation and
    /// scale are unchanged. Throws when the target is at the translation, when a value
    /// is not finite, or when `up` is zero; an up vector parallel to the view direction
    /// falls back to world +Z, as `LightRig` does.
    mutating func look(
        at target: SIMD3<Float>, up: SIMD3<Float> = SIMD3(0, 1, 0), forward: SceneForwardAxis = .negativeZ
    ) throws {
        rotation = try Self.rotation(from: translation, lookingAt: target, up: up, forward: forward)
    }

    /// The unit rotation whose `forward` axis points from `eye` toward `target`.
    internal static func rotation(
        from eye: SIMD3<Float>, lookingAt target: SIMD3<Float>, up: SIMD3<Float>, forward axis: SceneForwardAxis
    ) throws -> simd_quatf {
        let delta = SIMD3<Double>(target) - SIMD3<Double>(eye)
        let length = simd_length(delta)
        let upLength = simd_length(SIMD3<Double>(up))
        guard length.isFinite, length > 1e-9, upLength.isFinite, upLength > 0 else {
            throw SceneGraph.EditingError.invalidLookAt
        }
        let direction = delta / length
        var upAxis = SIMD3<Double>(up) / upLength
        if abs(simd_dot(direction, upAxis)) > 0.9999 {
            upAxis = abs(direction.z) < 0.9999 ? SIMD3(0, 0, 1) : SIMD3(1, 0, 0)
        }
        // Local +Z is the backward axis for -Z facing and the view direction for +Z facing.
        let back = axis == .negativeZ ? -direction : direction
        let right = simd_normalize(simd_cross(upAxis, back))
        let trueUp = simd_cross(back, right)
        let rotation = simd_quatd(simd_double3x3(columns: (right, trueUp, back)))
        let unit = simd_normalize(rotation.vector)
        return simd_quatf(vector: SIMD4<Float>(Float(unit.x), Float(unit.y), Float(unit.z), Float(unit.w)))
    }
}

public extension SceneGraph {
    /// Errors from editing and querying a scene graph's hierarchy.
    enum EditingError: LocalizedError, Equatable {
        case missingNode(UUID)
        case duplicateID(UUID)
        case cycle(UUID)
        case unrepresentableTransform
        case invalidLookAt
        case invalidTransform(UUID)

        public var errorDescription: String? {
            switch self {
            case .missingNode(let id): "Scene node \(id) is not in the graph."
            case .duplicateID(let id): "Scene node \(id) is already in the graph."
            case .cycle(let id): "Scene node \(id) cannot become a descendant of itself."
            case .unrepresentableTransform:
                "The transform must be finite, affine and invertible without shear to be stored as translation, rotation and scale."
            case .invalidLookAt: "Look-at needs finite values, a target away from the eye, and a nonzero up vector."
            case .invalidTransform(let id): "Scene node \(id) or one of its ancestors has a transform that cannot be inverted."
            }
        }
    }

    // MARK: Lookup

    /// The node with an ID, searching depth-first, like Three.js `getObjectById`.
    func node(withID id: UUID) -> SceneNode? {
        firstNode { $0.id == id }
    }

    /// The first node in depth-first order with a name, like Three.js `getObjectByName`.
    func node(named name: String) -> SceneNode? {
        firstNode { $0.name == name }
    }

    /// Every node with a name in depth-first order, like Three.js `getObjectsByProperty("name", …)`.
    func nodes(named name: String) -> [SceneNode] {
        nodes { $0.name == name }
    }

    /// The first node in depth-first pre-order that satisfies a predicate, like
    /// Three.js `getObjectByProperty` for any property.
    func firstNode(where predicate: (SceneNode) throws -> Bool) rethrows -> SceneNode? {
        var pending = Array(roots.reversed())
        while let node = pending.popLast() {
            if try predicate(node) { return node }
            pending.append(contentsOf: node.children.reversed())
        }
        return nil
    }

    /// Every node in depth-first pre-order that satisfies a predicate.
    func nodes(where predicate: (SceneNode) throws -> Bool) rethrows -> [SceneNode] {
        var result: [SceneNode] = []
        try forEachNode { node, _ in if try predicate(node) { result.append(node) } }
        return result
    }

    // MARK: Traversal

    /// Visits every node in depth-first pre-order with its parent's ID, like Three.js `traverse`.
    func forEachNode(_ body: (SceneNode, _ parentID: UUID?) throws -> Void) rethrows {
        var pending = roots.reversed().map { (node: $0, parent: UUID?.none) }
        while let (node, parent) = pending.popLast() {
            try body(node, parent)
            pending.append(contentsOf: node.children.reversed().map { ($0, node.id) })
        }
    }

    /// Visits nodes whose own and ancestors' `isVisible` are true, skipping hidden
    /// subtrees, like Three.js `traverseVisible`. Layers are not considered.
    func forEachVisibleNode(_ body: (SceneNode, _ parentID: UUID?) throws -> Void) rethrows {
        var pending = roots.reversed().filter(\.isVisible).map { (node: $0, parent: UUID?.none) }
        while let (node, parent) = pending.popLast() {
            try body(node, parent)
            pending.append(contentsOf: node.children.reversed().filter(\.isVisible).map { ($0, node.id) })
        }
    }

    /// The node's ancestors from its parent up to its root, like Three.js
    /// `traverseAncestors`. Nil when the node is not in the graph.
    func ancestors(of id: UUID) -> [SceneNode]? {
        guard let path = path(to: id) else { return nil }
        var result: [SceneNode] = []
        var siblings = roots
        for index in path.dropLast() {
            let node = siblings[index]
            result.append(node)
            siblings = node.children
        }
        return result.reversed()
    }

    /// The ID of the node's parent, or nil for a root or a missing node.
    func parentID(of id: UUID) -> UUID? {
        ancestors(of: id)?.first?.id
    }

    // MARK: World and local coordinates

    /// The node's local-to-world matrix, including its orientation inheritance mode.
    func worldTransform(of id: UUID) throws -> simd_float4x4 {
        guard let chain = chain(to: id) else { throw EditingError.missingNode(id) }
        return Self.worldTransform(along: chain) ?? matrix_identity_float4x4
    }

    /// The node's world-space origin, like Three.js `getWorldPosition`.
    func worldPosition(of id: UUID) throws -> SIMD3<Float> {
        let world = try worldTransform(of: id)
        return SIMD3(world.columns.3.x, world.columns.3.y, world.columns.3.z)
    }

    /// Converts a point from the node's local space to world space, like Three.js `localToWorld`.
    func localToWorld(_ point: SIMD3<Float>, in id: UUID) throws -> SIMD3<Float> {
        let world = try worldTransform(of: id) * SIMD4(point, 1)
        return SIMD3(world.x, world.y, world.z)
    }

    /// Converts a world-space point into the node's local space, like Three.js `worldToLocal`.
    /// Uses double precision for the inverse.
    func worldToLocal(_ point: SIMD3<Float>, in id: UUID) throws -> SIMD3<Float> {
        let inverse = try Self.inverse(worldTransform(of: id), id: id)
        let local = inverse * SIMD4<Double>(SIMD4(point, 1))
        return SIMD3(Float(local.x), Float(local.y), Float(local.z))
    }

    // MARK: Editing

    /// Applies an edit to one node in place. Throws `missingNode` when it is absent.
    /// The edit may change anything but the node's identity; changing children is allowed.
    mutating func updateNode<Result>(withID id: UUID, _ body: (inout SceneNode) throws -> Result) throws -> Result {
        guard let path = path(to: id) else { throw EditingError.missingNode(id) }
        return try modify(at: path, body)
    }

    /// Adds a subtree as the last child of a parent, or as the last root when the
    /// parent is nil, like Three.js `add`. The local transform is kept, so the subtree
    /// moves with its new parent. Throws when an ID in the subtree is already in the graph.
    ///
    /// Copying a `SceneNode` value copies its ID, so the graph is searched for each new
    /// ID on every call; the cost grows with the graph's size. Add many subtrees with
    /// `add(contentsOf:toParent:)`, which searches once.
    mutating func add(_ node: SceneNode, toParent parentID: UUID? = nil) throws {
        try add(contentsOf: [node], toParent: parentID)
    }

    /// Adds subtrees in order as the last children of a parent, or as the last roots when
    /// the parent is nil. The graph is searched for duplicate IDs once, so adding n nodes
    /// to a graph of m nodes costs O(n + m). Throws when an ID repeats among the new
    /// subtrees or is already in the graph; the graph is unchanged when it throws.
    mutating func add(contentsOf nodes: [SceneNode], toParent parentID: UUID? = nil) throws {
        var newIDs = Set<UUID>()
        var pending = nodes
        while let next = pending.popLast() {
            guard newIDs.insert(next.id).inserted else { throw EditingError.duplicateID(next.id) }
            pending.append(contentsOf: next.children)
        }
        if let existing = firstID(in: newIDs) { throw EditingError.duplicateID(existing) }
        if let parentID {
            try updateNode(withID: parentID) { $0.children.append(contentsOf: nodes) }
        } else {
            roots.append(contentsOf: nodes)
        }
    }

    /// Removes a node and its subtree and returns it, like Three.js `remove`.
    /// Returns nil when the node is not in the graph.
    @discardableResult
    mutating func removeNode(withID id: UUID) -> SceneNode? {
        guard let path = path(to: id) else { return nil }
        let index = path[path.count - 1]
        if path.count == 1 { return roots.remove(at: index) }
        return try? modify(at: Array(path.dropLast())) { $0.children.remove(at: index) }
    }

    /// Removes and returns every child of a node, like Three.js `clear`.
    @discardableResult
    mutating func removeChildren(of id: UUID) throws -> [SceneNode] {
        try updateNode(withID: id) { node in
            defer { node.children = [] }
            return node.children
        }
    }

    /// Moves a node under a new parent, or to the roots when the parent is nil, while
    /// keeping its world transform, like Three.js `attach`. The new local transform is
    /// solved in double precision. The pivot is kept. Inherited nodes use
    /// `parentWorld⁻¹ × world`. World-oriented nodes solve their local scale
    /// along each fixed world axis.
    ///
    /// Throws `cycle` when the new parent is the node or its descendant and
    /// `unrepresentableTransform` when the pose cannot be stored as local TRS.
    /// The graph is unchanged when it throws.
    mutating func attach(_ id: UUID, to newParentID: UUID?) throws {
        guard let node = node(withID: id) else { throw EditingError.missingNode(id) }
        let world = try worldTransform(of: id)
        var parentWorld = matrix_identity_float4x4
        if let newParentID {
            guard self.node(withID: newParentID) != nil else { throw EditingError.missingNode(newParentID) }
            if newParentID == id || ancestors(of: newParentID)?.contains(where: { $0.id == id }) == true {
                throw EditingError.cycle(id)
            }
            parentWorld = try worldTransform(of: newParentID)
        }
        let local = try Self.inverse(parentWorld, id: newParentID ?? id) * Self.double(world)
        var moved = node
        let worldOriented = moved.orientationMode == .world && newParentID != nil
        if worldOriented {
            let origin = local.columns.3
            moved.transform = try SceneNode.worldOrientedLocalPose(
                preserving: world, under: parentWorld, localOrigin: SIMD3(Float(origin.x), Float(origin.y), Float(origin.z))
            )
        } else {
            moved.transform = try SceneTransform(decomposing: Self.float(local))
        }
        if let pivot = moved.pivot {
            let pivotMatrix = worldOriented ? Self.double(moved.transform.matrix) : local
            let worldPivot = pivotMatrix * SIMD4<Double>(SIMD3<Double>(pivot), 1)
            moved.transform.translation = SIMD3<Float>(SIMD3(worldPivot.x, worldPivot.y, worldPivot.z) - SIMD3<Double>(pivot))
            guard moved.transform.translation.isFiniteVector else { throw EditingError.unrepresentableTransform }
        }
        var edited = self
        edited.removeNode(withID: id)
        try edited.add(moved, toParent: newParentID)
        self = edited
    }

    /// Turns a node so its `forward` axis points at a world-space target, like Three.js
    /// `Object3D.lookAt`. The look-at uses the node's current origin in its
    /// parent's space or, for a world-oriented child, in world space. A pivot
    /// can move that origin when the rotation changes. Authored translation
    /// and scale are unchanged.
    mutating func lookAt(
        _ id: UUID, worldTarget target: SIMD3<Float>, up: SIMD3<Float> = SIMD3(0, 1, 0),
        forward: SceneForwardAxis = .negativeZ
    ) throws {
        guard let chain = chain(to: id) else { throw EditingError.missingNode(id) }
        let parent = Self.worldTransform(along: chain.dropLast())
        let inverse = try Self.inverse(parent ?? matrix_identity_float4x4, id: id)
        try updateNode(withID: id) { node in
            if node.orientationMode == .world, let parent {
                let origin = node.resolvedWorldTransform(relativeTo: parent).columns.3
                let orientation = try SceneTransform.rotation(
                    from: SIMD3(origin.x, origin.y, origin.z), lookingAt: target, up: up, forward: forward
                )
                node.transform.rotation = orientation * node.worldOrientationCorrection.conjugate
            } else {
                let localTarget = inverse * SIMD4<Double>(SIMD4(target, 1))
                let localUp = inverse * SIMD4<Double>(SIMD4(up, 0))
                let origin = node.localTransformMatrix.columns.3
                node.transform.rotation = try SceneTransform.rotation(
                    from: SIMD3(origin.x, origin.y, origin.z),
                    lookingAt: SIMD3(Float(localTarget.x), Float(localTarget.y), Float(localTarget.z)),
                    up: SIMD3(Float(localUp.x), Float(localUp.y), Float(localUp.z)), forward: forward
                )
            }
        }
    }

    /// Points the graph's camera at a world-space target, keeping its position.
    mutating func pointCamera(at target: SIMD3<Float>, up: SIMD3<Float> = SIMD3(0, 1, 0)) throws {
        try cameraTransform.look(at: target, up: up, forward: .negativeZ)
    }

    // MARK: Paths

    /// The first graph ID found in `ids`. Walks child arrays by index and reads only IDs,
    /// without copying node values.
    private func firstID(in ids: Set<UUID>) -> UUID? {
        guard !ids.isEmpty else { return nil }
        var levels = [roots]
        while let level = levels.popLast() {
            for index in level.indices {
                let id = level[index].id
                if ids.contains(id) { return id }
                if !level[index].children.isEmpty { levels.append(level[index].children) }
            }
        }
        return nil
    }

    /// Child indices from a root to the node, or nil when it is absent.
    internal func path(to id: UUID) -> [Int]? {
        var pending = roots.indices.reversed().map { (node: roots[$0], path: [$0]) }
        while let (node, path) = pending.popLast() {
            if node.id == id { return path }
            for index in node.children.indices.reversed() {
                pending.append((node.children[index], path + [index]))
            }
        }
        return nil
    }

    private static func worldTransform(along chain: some Sequence<SceneNode>) -> simd_float4x4? {
        chain.reduce(nil) { $1.resolvedWorldTransform(relativeTo: $0) }
    }

    /// The nodes from a root down to the node, inclusive.
    private func chain(to id: UUID) -> [SceneNode]? {
        guard let path = path(to: id) else { return nil }
        var result: [SceneNode] = []
        var siblings = roots
        for index in path {
            result.append(siblings[index])
            siblings = siblings[index].children
        }
        return result
    }

    /// Edits the node at a path without recursion, rebuilding each ancestor's child list.
    private mutating func modify<Result>(at path: [Int], _ body: (inout SceneNode) throws -> Result) throws -> Result {
        var ancestors: [SceneNode] = []
        var node = roots[path[0]]
        for index in path.dropFirst() {
            ancestors.append(node)
            node = node.children[index]
        }
        let id = node.id
        let result = try body(&node)
        guard node.id == id else { throw EditingError.missingNode(id) }
        for index in path.dropFirst().reversed() {
            var parent = ancestors.removeLast()
            parent.children[index] = node
            node = parent
        }
        roots[path[0]] = node
        return result
    }

    private static func double(_ matrix: simd_float4x4) -> simd_double4x4 {
        simd_double4x4(columns: (
            SIMD4<Double>(matrix.columns.0), SIMD4<Double>(matrix.columns.1),
            SIMD4<Double>(matrix.columns.2), SIMD4<Double>(matrix.columns.3)
        ))
    }

    private static func float(_ matrix: simd_double4x4) -> simd_float4x4 {
        simd_float4x4(columns: (
            SIMD4<Float>(matrix.columns.0), SIMD4<Float>(matrix.columns.1),
            SIMD4<Float>(matrix.columns.2), SIMD4<Float>(matrix.columns.3)
        ))
    }

    private static func inverse(_ matrix: simd_float4x4, id: UUID) throws -> simd_double4x4 {
        let value = double(matrix)
        let determinant = simd_determinant(value)
        guard determinant.isFinite, determinant != 0 else { throw EditingError.invalidTransform(id) }
        return simd_inverse(value)
    }
}
