import Foundation
import Metal
import simd

/// Reusable meshes and hit testing for a ``TransformGizmo``.
public final class TransformGizmoOverlay {
    private struct Drawable {
        let renderer: MeshRenderer
        let acceleration: MeshBVH

        init(
            _ geometry: MeshGeometry, device: any MTLDevice, format: MTLPixelFormat,
            depthMode: MeshDepthMode, sampleCount: Int
        ) throws {
            renderer = try MeshRenderer(
                device: device, colorPixelFormat: format, geometry: geometry, depthMode: depthMode,
                renderState: .init(blending: .normal, transparent: true, depthTest: false, depthWrite: false, side: .double),
                sampleCount: sampleCount
            )
            acceleration = try MeshBVH(geometry: geometry)
        }
    }

    private let arrow: Drawable
    private let ring: Drawable
    private let scaleAxis: Drawable
    private let center: Drawable
    private let plane: Drawable
    private let axisIDs = TransformGizmo.Axis.allCases.map { _ in UUID() }
    private let planeIDs = TransformGizmo.Plane.allCases.map { _ in UUID() }
    private let centerID = UUID()
    private let viewRotationID = UUID()
    private static let z = SIMD3<Float>(0, 0, 1)

    public init(
        device: any MTLDevice, colorPixelFormat: MTLPixelFormat,
        depthMode: MeshDepthMode = .forward, sampleCount: Int = 1
    ) throws {
        func prepare(_ geometry: MeshGeometry) throws -> Drawable {
            try Drawable(geometry, device: device, format: colorPixelFormat, depthMode: depthMode, sampleCount: sampleCount)
        }
        let shaft = try MeshGeometry.cylinder(radiusTop: 0.031, radiusBottom: 0.031, height: 0.60, radialSegments: 10)
        let tip = try MeshGeometry.cone(radius: 0.090, height: 0.24, radialSegments: 16)
        let turn = simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(1, 0, 0))
        let stem = MeshGeometry.MergePart(shaft, transform: SceneTransform(translation: Self.z * 0.46, rotation: turn).matrix)
        arrow = try prepare(MeshGeometry.merged([
            stem, .init(tip, transform: SceneTransform(translation: Self.z * 0.88, rotation: turn).matrix),
        ]).geometry)
        let box = MeshGeometry.cube()
        scaleAxis = try prepare(MeshGeometry.merged([
            stem, .init(box, transform: SceneTransform(translation: Self.z * 0.87, scale: SIMD3(repeating: 0.22)).matrix),
        ]).geometry)
        ring = try prepare(.torus(radius: 0.98, tube: 0.034, radialSegments: 8, tubularSegments: 64))
        center = try prepare(box)
        plane = try prepare(.plane())
    }

    /// Root-level helper nodes with depth testing and writes disabled. Use a
    /// separate overlay pass with loaded color after the main scene for full
    /// control over compositing. Reuse an overlay instance across frames to
    /// retain handle identities and prepared GPU resources.
    public func nodes(
        for frame: TransformGizmo.Frame, highlighting handle: TransformGizmo.Handle? = nil,
        layers: SceneLayers = .default
    ) -> [SceneNode] {
        frame.visibleHandles.map { item in
            let id: UUID
            let tint: SIMD4<Float>
            switch item {
            case .axis(let axis):
                id = axisIDs[axis.rawValue]
                tint = Self.axisColors[axis.rawValue]
            case .plane(let plane):
                id = planeIDs[plane.rawValue]
                let (a, b, _) = plane.axes
                let color = simd_max(Self.axisColors[a.rawValue], Self.axisColors[b.rawValue])
                tint = SIMD4(color.x, color.y, color.z, 0.8)
            case .screen:
                id = centerID
                tint = SIMD4(0.8, 0.84, 0.92, 1)
            case .viewRotation:
                id = viewRotationID
                tint = SIMD4(0.92, 0.92, 0.81, 1)
            }
            return SceneNode(
                id: id, transform: placement(of: item, in: frame), mesh: drawable(for: item, mode: frame.mode).renderer,
                materialParameters: item == handle ? SIMD4(1, 0.82, 0.12, 1) : tint,
                castsShadow: false, frustumCulled: false, layers: layers, renderOrder: Int.max
            )
        }
    }

    /// Picks the nearest visible handle using the same triangles as drawing.
    /// This query is independent of scene geometry because the handles overlay it.
    public func hitTest(_ ray: SceneRay, in frame: TransformGizmo.Frame) throws -> TransformGizmo.Handle? {
        var nearest: (handle: TransformGizmo.Handle, distance: Float)?
        for handle in frame.visibleHandles {
            let drawable = drawable(for: handle, mode: frame.mode)
            guard let hit = try drawable.acceleration.intersection(
                ray: ray, localToWorld: placement(of: handle, in: frame).matrix, cullsBackFaces: false
            ), nearest.map({ hit.distance < $0.distance }) != false else { continue }
            nearest = (handle, hit.distance)
        }
        return nearest?.handle
    }

    private func drawable(for handle: TransformGizmo.Handle, mode: TransformGizmo.Mode) -> Drawable {
        switch handle {
        case .screen: return center
        case .plane: return plane
        case .viewRotation: return ring
        case .axis:
            switch mode {
            case .translate: return arrow
            case .rotate: return ring
            case .scale: return scaleAxis
            }
        }
    }

    private func placement(of handle: TransformGizmo.Handle, in frame: TransformGizmo.Frame) -> SceneTransform {
        var position = frame.position
        let direction: simd_quatf
        let size: Float
        switch handle {
        case .axis(let axis):
            direction = frame.rotation * simd_quatf(from: Self.z, to: axis.direction)
            size = frame.size
        case .plane(let plane):
            let (a, b, n) = plane.axes
            position += frame.rotation.act(a.direction + b.direction) * (frame.size * 0.35)
            direction = frame.rotation * simd_quatf(from: Self.z, to: n.direction)
            size = frame.size * 0.22
        case .screen:
            direction = frame.rotation
            size = frame.size * 0.20
        case .viewRotation:
            direction = simd_quatf(from: Self.z, to: frame.viewAxis)
            size = frame.size * 1.24
        }
        return SceneTransform(translation: position, rotation: direction, scale: SIMD3(repeating: size))
    }

    private static let axisColors: [SIMD4<Float>] = [
        SIMD4(1, 0.12, 0.10, 1), SIMD4(0.15, 0.94, 0.18, 1), SIMD4(0.18, 0.42, 1, 1),
    ]
}
