import Foundation
import simd

/// Ray-driven controls for editing a scene node's translation, rotation or scale.
/// Supply rays from a pointer, touch location, or any other world-space input.
public struct TransformGizmo: Sendable {
    public enum Mode: Sendable { case translate, rotate, scale }
    public enum Space: Sendable { case world, local }
    public enum SnapReference: Sendable { case relative, absolute }
    public enum Axis: Int, CaseIterable, Hashable, Sendable {
        case x, y, z

        var direction: SIMD3<Float> {
            var value = SIMD3<Float>.zero
            value[rawValue] = 1
            return value
        }
    }

    public enum Plane: Int, CaseIterable, Hashable, Sendable {
        case xy, yz, xz

        var axes: (first: Axis, second: Axis, normal: Axis) {
            switch self {
            case .xy: (.x, .y, .z)
            case .yz: (.y, .z, .x)
            case .xz: (.z, .x, .y)
            }
        }
    }

    public enum Handle: Hashable, Sendable {
        case axis(Axis)
        case plane(Plane)
        /// Free translation in the camera plane or uniform scaling along camera up.
        case screen
        /// Rotation about the line from the pivot to the eye at pointer-down.
        case viewRotation
    }

    public let mode: Mode
    public let space: Space
    public let axes: Set<Axis>
    public let planes: Set<Plane>
    public let showsScreenHandle: Bool
    public let showsViewRotation: Bool
    public let sizeInPixels: Float
    public let snap: Float?
    public let snapReference: SnapReference
    /// An optional parent-space box. Clips the requested displacement at its
    /// boundary, preserving the selected constraint and direction of movement.
    public let translationBounds: AxisAlignedBox?

    /// The size is the nominal radius or axis length in drawable pixels.
    /// Relative snaps measure from the starting pose, in world units,
    /// radians, or scale factors. Absolute snaps use the world origin projected
    /// onto the translation constraint or the magnitudes of the local scale
    /// components. Uniform absolute scaling snaps the largest component's
    /// magnitude and applies its factor to all axes. Rotation snaps are always
    /// relative, and scale uses local axes.
    public init(
        mode: Mode = .translate, space: Space = .world, axes: Set<Axis> = Set(Axis.allCases),
        planes: Set<Plane> = Set(Plane.allCases), showsScreenHandle: Bool = true, showsViewRotation: Bool = true,
        sizeInPixels: Float = 80, snap: Float? = nil, snapReference: SnapReference = .relative,
        translationBounds: AxisAlignedBox? = nil
    ) throws {
        guard sizeInPixels.isFinite, sizeInPixels > 0,
              snap.map({ $0.isFinite && $0 > 0 }) != false else { throw Failure.invalidOptions }
        self.mode = mode
        self.space = mode == .scale ? .local : space
        self.axes = axes
        self.planes = planes
        self.showsScreenHandle = showsScreenHandle
        self.showsViewRotation = showsViewRotation
        self.sizeInPixels = sizeInPixels
        self.snap = snap
        self.snapReference = snapReference
        self.translationBounds = translationBounds
    }

    /// A pose shared by the visible handles, hit testing and drag solver.
    public struct Frame: Sendable {
        public let mode: Mode
        public let position: SIMD3<Float>
        public let rotation: simd_quatf
        public let size: Float
        public let visibleHandles: [Handle]

        let viewAxis: SIMD3<Float>
        fileprivate let nodeID: UUID
        fileprivate let start: SceneTransform
        fileprivate let localPivot: SIMD3<Float>?
        fileprivate let orientationMode: SceneNode.OrientationMode
        fileprivate let parentID: UUID?
        fileprivate let parentMatrix: simd_float4x4
        fileprivate let parentInverse: simd_double3x3
        fileprivate let parentRotation: simd_quatf
        fileprivate let cameraRotation: simd_quatf
        fileprivate let snap: Double?
        fileprivate let snapReference: SnapReference
        fileprivate let translationBounds: AxisAlignedBox?

        /// Begins a drag for a picked handle. The host may also select the screen
        /// translation handle directly after picking the target mesh. A ray
        /// parallel to the working plane, or too close to the pivot for rotation
        /// or axis scaling, cannot begin a stable drag.
        public func beginDrag(_ handle: Handle, ray: SceneRay) -> Drag? {
            guard visibleHandles.contains(handle) else { return nil }
            let pivot = SIMD3<Double>(position)
            switch handle {
            case .axis(let axis):
                let direction = simd_normalize(SIMD3<Double>(rotation.act(axis.direction)))
                if mode == .rotate {
                    return rotationDrag(handle, ray: ray, axis: direction)
                }
                let view = SIMD3<Double>(ray.direction)
                let perpendicular = view - direction * simd_dot(view, direction)
                guard simd_length_squared(perpendicular) > 1e-6 else { return nil }
                let normal = simd_normalize(perpendicular)
                guard let point = TransformGizmo.intersect(ray, pivot: pivot, normal: normal) else { return nil }
                let initial = simd_dot(point - pivot, direction)
                guard mode != .scale || initial > Double(size) * 0.10 else { return nil }
                return Drag(frame: self, handle: handle, normal: normal, measurement: .axis(axis, direction, initial))
            case .plane(let plane):
                let (first, second, face) = plane.axes
                let u = SIMD3<Double>(rotation.act(first.direction))
                let v = SIMD3<Double>(rotation.act(second.direction))
                let normal = SIMD3<Double>(rotation.act(face.direction))
                guard let point = TransformGizmo.intersect(ray, pivot: pivot, normal: normal) else { return nil }
                let initial = SIMD2(simd_dot(point - pivot, u), simd_dot(point - pivot, v))
                guard mode != .scale || initial.min() > Double(size) * 0.10 else { return nil }
                return Drag(frame: self, handle: handle, normal: normal, measurement: .plane(plane, u, v, initial))
            case .screen:
                let right = SIMD3<Double>(cameraRotation.act(Axis.x.direction))
                let up = SIMD3<Double>(cameraRotation.act(Axis.y.direction))
                let normal = SIMD3<Double>(cameraRotation.act(Axis.z.direction))
                guard let initial = TransformGizmo.intersect(ray, pivot: pivot, normal: normal) else { return nil }
                return Drag(frame: self, handle: handle, normal: normal, measurement: .screen(right, up, initial))
            case .viewRotation:
                return rotationDrag(handle, ray: ray, axis: SIMD3<Double>(viewAxis))
            }
        }

        private func rotationDrag(_ handle: Handle, ray: SceneRay, axis: SIMD3<Double>) -> Drag? {
            let pivot = SIMD3<Double>(position)
            guard let point = TransformGizmo.intersect(ray, pivot: pivot, normal: axis),
                  let arm = radial(point - pivot) else { return nil }
            return Drag(frame: self, handle: handle, normal: axis, measurement: .angle(axis, arm, 0))
        }

        fileprivate func radial(_ vector: SIMD3<Double>) -> SIMD3<Double>? {
            let length = simd_length(vector)
            return length > Double(size) * 1e-5 ? vector / length : nil
        }
    }

    /// Builds a frame for a visible node. Returns nil when the pivot is outside
    /// the perspective camera's depth range or no handles can be used.
    ///
    /// World translation accepts an invertible affine parent. Rotation and
    /// local gizmos need an orientation-preserving, uniformly scaled parent
    /// so that the displayed orthogonal axes agree with editable TRS values.
    /// World-oriented children use their own orientation for the axes.
    public func frame(for nodeID: UUID, in graph: SceneGraph, drawableHeight: Float) throws -> Frame? {
        guard drawableHeight.isFinite, drawableHeight > 0 else { throw Failure.invalidDrawable }
        guard let node = graph.node(withID: nodeID), let ancestors = graph.ancestors(of: nodeID) else {
            throw SceneGraph.EditingError.missingNode(nodeID)
        }
        guard node.isVisible, node.layers.intersects(graph.cameraLayers),
              ancestors.allSatisfy(\.isVisible) else { return nil }
        let parentID = ancestors.first?.id
        let parent = try parentID.map { try graph.worldTransform(of: $0) } ?? matrix_identity_float4x4
        func xyz(_ column: SIMD4<Float>) -> SIMD3<Double> {
            SIMD3(Double(column.x), Double(column.y), Double(column.z))
        }
        let parentBasis = simd_double3x3(xyz(parent[0]), xyz(parent[1]), xyz(parent[2]))
        let parentOrigin = xyz(parent[3])
        let determinant = parentBasis.determinant
        let start = node.transform
        let camera = graph.cameraTransform
        guard determinant.isFinite, determinant != 0, parentOrigin.indices.allSatisfy({ parentOrigin[$0].isFinite }),
              Self.valid(start), Self.valid(camera), Self.isUniformPositive(camera.scale) else { throw Failure.invalidTransform }
        if mode == .translate, let translationBounds, !translationBounds.contains(start.translation) {
            throw Failure.outsideTranslationBounds
        }
        let worldOriented = node.orientationMode == .world && parentID != nil
        let parentRotation: simd_quatf
        if !worldOriented && (mode == .rotate || space == .local) {
            guard let pose = try? SceneTransform(decomposing: parent), Self.isUniformPositive(pose.scale) else {
                throw Failure.unsupportedParentTransform
            }
            parentRotation = pose.rotation
        } else {
            parentRotation = simd_quatf(real: 1, imag: .zero)
        }
        let position: SIMD3<Float>
        if worldOriented {
            let anchor = node.resolvedWorldTransform(relativeTo: parent) * SIMD4(node.pivot ?? .zero, 1)
            position = SIMD3(anchor.x, anchor.y, anchor.z)
        } else {
            let localAnchor = SIMD3<Double>(start.translation) + SIMD3<Double>(node.pivot ?? .zero)
            position = SIMD3<Float>(parentBasis * localAnchor + parentOrigin)
        }
        let eye = SIMD3<Double>(camera.translation) - SIMD3<Double>(position)
        let cameraRotation = simd_normalize(camera.rotation)
        let depth = simd_dot(eye, SIMD3<Double>(cameraRotation.act(Axis.z.direction)))
        let cameraDepth = depth / Double(camera.scale.z)
        guard position.isFiniteVector else { throw Failure.invalidTransform }
        let clips = graph.activeClipDistances
        guard (Double(clips.near)...Double(clips.far)).contains(cameraDepth) else { return nil }
        let size: Float
        if let orthographic = graph.orthographicCamera {
            size = Float(2 * Double(orthographic.halfHeight) * Double(sizeInPixels) / Double(drawableHeight))
        } else {
            let field = Double(graph.camera.verticalFieldOfView)
            size = Float(2 * depth * tan(field / 2) * Double(sizeInPixels) / Double(drawableHeight))
        }
        let farthestHandle = simd_abs(SIMD3<Double>(position)).max() + Double(size) * 1.25
        guard size.isNormal, size > 0, farthestHandle <= Double(Float.greatestFiniteMagnitude) else {
            throw Failure.invalidDrawable
        }
        let localRotation = worldOriented ? node.worldOrientedRotation : start.rotation
        let rotation = space == .local ? simd_normalize(parentRotation * localRotation) : simd_quatf(real: 1, imag: .zero)
        let view = SIMD3<Float>(simd_normalize(eye))
        var handles = Axis.allCases.compactMap { axis -> Handle? in
            let alignment = abs(simd_dot(rotation.act(axis.direction), view))
            guard axes.contains(axis), mode == .rotate ? alignment > 0.10 : alignment < 0.985 else { return nil }
            return .axis(axis)
        }
        if mode == .rotate {
            if showsViewRotation { handles.append(.viewRotation) }
        } else {
            for plane in Plane.allCases where planes.contains(plane) {
                let (first, second, normal) = plane.axes
                if axes.contains(first), axes.contains(second),
                   abs(simd_dot(rotation.act(normal.direction), view)) > 0.14 {
                    handles.append(.plane(plane))
                }
            }
            if showsScreenHandle { handles.append(.screen) }
        }
        guard !handles.isEmpty else { return nil }
        return Frame(
            mode: mode, position: position, rotation: rotation, size: size, visibleHandles: handles,
            viewAxis: view, nodeID: nodeID, start: start, localPivot: node.pivot, orientationMode: node.orientationMode,
            parentID: parentID, parentMatrix: parent,
            parentInverse: parentBasis.inverse, parentRotation: parentRotation, cameraRotation: cameraRotation,
            snap: snap.map(Double.init), snapReference: snapReference, translationBounds: translationBounds
        )
    }

    /// One pointer's editing session. End it with `finish()` to keep the edit,
    /// or `cancel(in:)` to restore the starting transform.
    public struct Drag: Sendable {
        public let handle: Handle
        public private(set) var isActive = true

        fileprivate enum Measurement: Sendable {
            case axis(Axis, SIMD3<Double>, Double)
            case plane(Plane, SIMD3<Double>, SIMD3<Double>, SIMD2<Double>)
            case screen(SIMD3<Double>, SIMD3<Double>, SIMD3<Double>)
            case angle(SIMD3<Double>, SIMD3<Double>, Double)
        }
        private let frame: Frame
        private let normal: SIMD3<Double>
        private var measurement: Measurement
        private var applied: SceneTransform

        fileprivate init(frame: Frame, handle: Handle, normal: SIMD3<Double>, measurement: Measurement) {
            self.frame = frame
            self.handle = handle
            self.normal = normal
            self.measurement = measurement
            self.applied = frame.start
        }

        /// Applies a new pointer sample. A grazing ray keeps the previous pose
        /// and returns false. Rotation assumes less than half a turn between
        /// samples and accumulates across the signed angle's wrap.
        ///
        /// If the target has moved independently, been reparented, or its
        /// parent's world transform changed, throws without overwriting the edit.
        @discardableResult
        public mutating func update(ray: SceneRay, in graph: inout SceneGraph) throws -> Bool {
            guard isActive else { return false }
            try validateTarget(in: graph)
            let pivot = SIMD3<Double>(frame.position)
            guard let point = TransformGizmo.intersect(ray, pivot: pivot, normal: normal) else { return false }
            var next = frame.start
            var sample = measurement
            switch measurement {
            case .axis(let axis, let direction, let initial):
                let amount = simd_dot(point - pivot, direction)
                if frame.mode == .translate {
                    next.translation = translation(by: direction * translationStep(amount - initial, along: direction))
                } else {
                    next.scale[axis.rawValue] = scaled(axis, factor: amount / initial)
                }
            case .plane(let plane, let u, let v, let initial):
                let a = simd_dot(point - pivot, u), b = simd_dot(point - pivot, v)
                if frame.mode == .translate {
                    next.translation = translation(by: u * translationStep(a - initial.x, along: u) +
                        v * translationStep(b - initial.y, along: v))
                } else {
                    next.scale[plane.axes.first.rawValue] = scaled(plane.axes.first, factor: a / initial.x)
                    next.scale[plane.axes.second.rawValue] = scaled(plane.axes.second, factor: b / initial.y)
                }
            case .screen(let right, let up, let initial):
                let delta = point - initial
                let vertical = simd_dot(delta, up)
                if frame.mode == .translate {
                    let world = right * translationStep(simd_dot(delta, right), along: right) + up * translationStep(vertical, along: up)
                    next.translation = translation(by: world)
                } else {
                    let factor = scaleFactor(1 + vertical / Double(frame.size), reference: simd_abs(frame.start.scale).max())
                    next.scale = SIMD3<Float>(SIMD3<Double>(frame.start.scale) * factor)
                }
            case .angle(let axis, let previous, let accumulated):
                guard let arm = frame.radial(point - pivot) else { return false }
                let angle = accumulated + atan2(simd_dot(axis, simd_cross(previous, arm)), simd_dot(previous, arm))
                sample = .angle(axis, arm, angle)
                let radians = Float(snapped(angle).truncatingRemainder(dividingBy: 2 * .pi))
                if radians != 0 {
                    let parentAxis = frame.parentRotation.inverse.act(SIMD3<Float>(axis))
                    let delta = simd_quatf(angle: radians, axis: simd_normalize(parentAxis))
                    next.rotation = simd_normalize(delta * frame.start.rotation)
                }
            }
            guard TransformGizmo.valid(next) else { throw Failure.invalidTransform }
            if next == applied {
                measurement = sample
                return false
            }
            try graph.updateNode(withID: frame.nodeID) { $0.transform = next }
            measurement = sample
            applied = next
            return true
        }

        public mutating func finish() { isActive = false }

        /// Restores the original pose when the graph still owns this edit.
        public mutating func cancel(in graph: inout SceneGraph) throws {
            guard isActive else { return }
            try validateTarget(in: graph)
            try graph.updateNode(withID: frame.nodeID) { $0.transform = frame.start }
            finish()
        }

        private func validateTarget(in graph: SceneGraph) throws {
            guard let node = graph.node(withID: frame.nodeID), node.transform == applied, node.pivot == frame.localPivot,
                  node.orientationMode == frame.orientationMode,
                  node.isVisible, node.layers.intersects(graph.cameraLayers),
                  let ancestors = graph.ancestors(of: frame.nodeID), ancestors.first?.id == frame.parentID,
                  ancestors.allSatisfy(\.isVisible) else { throw Failure.staleDrag }
            let current = try frame.parentID.map { try graph.worldTransform(of: $0) } ?? matrix_identity_float4x4
            guard current == frame.parentMatrix else { throw Failure.staleDrag }
        }

        private func snapped(_ delta: Double) -> Double {
            guard let step = frame.snap else { return delta }
            return (delta / step).rounded() * step
        }

        private func translationStep(_ delta: Double, along direction: SIMD3<Double>) -> Double {
            if frame.snapReference == .absolute, frame.snap != nil {
                let origin = simd_dot(SIMD3<Double>(frame.position), direction)
                return snapped(origin + delta) - origin
            }
            return snapped(delta)
        }

        private func translation(by worldDelta: SIMD3<Double>) -> SIMD3<Float> {
            let start = SIMD3<Double>(frame.start.translation)
            let local = frame.parentInverse * worldDelta
            guard let bounds = frame.translationBounds else { return SIMD3<Float>(start + local) }
            var fraction = 1.0
            for i in local.indices where local[i] != 0 {
                let boundary = local[i] > 0 ? bounds.maximum[i] : bounds.minimum[i]
                fraction = min(fraction, (Double(boundary) - start[i]) / local[i])
            }
            return simd_clamp(SIMD3<Float>(start + local * fraction), bounds.minimum, bounds.maximum)
        }

        private func scaleFactor(_ factor: Double, reference: Float) -> Double {
            if frame.snapReference == .absolute, frame.snap != nil {
                let magnitude = abs(Double(reference))
                return max(1e-4, snapped(magnitude * factor) / magnitude)
            }
            return max(1e-4, 1 + snapped(factor - 1))
        }

        private func scaled(_ axis: Axis, factor: Double) -> Float {
            let value = frame.start.scale[axis.rawValue]
            return Float(Double(value) * scaleFactor(factor, reference: value))
        }
    }

    private static func intersect(_ ray: SceneRay, pivot: SIMD3<Double>, normal: SIMD3<Double>) -> SIMD3<Double>? {
        let origin = SIMD3<Double>(ray.origin)
        let direction = SIMD3<Double>(ray.direction)
        let divisor = simd_dot(direction, normal)
        guard abs(divisor) > 1e-4 else { return nil }
        let distance = simd_dot(pivot - origin, normal) / divisor
        guard distance >= 0, distance <= Double(ray.maximumDistance) else { return nil }
        return origin + direction * distance
    }

    private static func isUniformPositive(_ scale: SIMD3<Float>) -> Bool {
        scale.min() > 0 && scale.max() - scale.min() <= 1e-4 * scale.max()
    }

    private static func valid(_ pose: SceneTransform) -> Bool {
        pose.translation.isFiniteVector && pose.scale.isFiniteVector &&
        simd_abs(pose.scale).min() > 0 && abs(simd_length_squared(pose.rotation.vector) - 1) <= 1e-4
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidOptions
        case invalidDrawable
        case invalidTransform
        case unsupportedParentTransform
        case outsideTranslationBounds
        case staleDrag

        public var errorDescription: String? {
            switch self {
            case .invalidOptions: "A transform gizmo needs positive finite size and snap increments."
            case .invalidDrawable: "A transform gizmo needs a positive drawable height and a representable projected size."
            case .invalidTransform: "The edited node needs a finite, invertible pose and the camera needs a positive uniform scale."
            case .unsupportedParentTransform: "Local and rotation gizmos need a uniformly scaled, orientation-preserving parent."
            case .outsideTranslationBounds: "The node must start inside its parent-space translation bounds."
            case .staleDrag: "The target or its parent changed outside this drag."
            }
        }
    }
}
