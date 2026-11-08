import Frostfield
import simd
import UIKit

/// FF-07 fixture: the procedural cube as one indexed mesh, drawn with depth
/// testing and back-face culling and colored by its model-space normals.
///
/// Each face keeps its own color however the cube turns: +x reddish
/// (1.0, 0.5, 0.5), -x teal (0.0, 0.5, 0.5), +y light green (0.5, 1.0, 0.5),
/// -y purple (0.5, 0.0, 0.5), +z lavender blue (0.5, 0.5, 1.0), and -z olive
/// (0.5, 0.5, 0.0). A wrong winding convention would show the far faces, and
/// a broken depth test would show whichever face was drawn last.
///
/// The mesh is uploaded once, when the fixture is created. Poses and the
/// continuous spin change only the transform, so the renderer's buffer
/// counters stay constant while the cube moves.
final class GeometryFixture {
    /// Deterministic poses at animation time 0.
    enum Pose: String, CaseIterable {
        case front, corner, stretched

        var title: String {
            switch self {
            case .front: "Front"
            case .corner: "Corner"
            case .stretched: "Stretched"
            }
        }

        var transform: Transform {
            let cornerRotation = simd_quatf(angle: 0.6, axis: [1, 0, 0]) * simd_quatf(angle: .pi / 4, axis: [0, 1, 0])
            return switch self {
            case .front:
                // Only the +z face is visible.
                .identity
            case .corner:
                // 45° about +y brings the -x and +z faces to the front; the
                // tilt about +x then shows +y on top.
                Transform(rotation: cornerRotation)
            case .stretched:
                Transform(rotation: cornerRotation, scale: [1.6, 0.6, 1])
            }
        }
    }

    /// Radians per second of spin while drawing continuously.
    static let spinSpeed: Float = 0.6
    static let spinAxis = simd_normalize(SIMD3<Float>(0.3, 1, 0.2))

    let mesh: Mesh
    let camera: Camera
    private(set) var pose: Pose

    /// Called after the app changes the pose.
    var onChange: (() -> Void)?

    private let poseControl = UISegmentedControl(items: Pose.allCases.map(\.title))

    /// Uploads the cube with `renderer`; this is the fixture's only buffer
    /// allocation.
    init(renderer: MeshRenderer, pose: Pose = .front) throws(RendererError) {
        mesh = try renderer.makeMesh(.cube(size: 1))
        self.pose = pose
        do {
            camera = Camera(
                position: [0, -0.5, 5],
                projection: try PerspectiveProjection(verticalFieldOfView: 50 * .pi / 180, near: 0.1, far: 20)
            )
        } catch {
            preconditionFailure("The fixture's projection is valid: \(error)")
        }
    }

    /// The model transform at an animation time: the pose, then the spin.
    func transform(animationTime: Double) -> Transform {
        var transform = pose.transform
        let spin = simd_quatf(angle: Float(animationTime) * Self.spinSpeed, axis: Self.spinAxis)
        transform.rotation = spin * transform.rotation
        return transform
    }

    /// Fields for the status label's accessibility value.
    var statusFields: [(String, String)] {
        [("pose", pose.rawValue), ("meshVertices", "\(mesh.vertexCount)"), ("meshIndices", "\(mesh.indexCount)")]
    }

    func makeControls() -> UIView {
        poseControl.selectedSegmentIndex = Pose.allCases.firstIndex(of: pose) ?? 0
        poseControl.accessibilityIdentifier = "cubePose"
        poseControl.accessibilityLabel = "Cube pose"
        poseControl.addAction(UIAction { [weak self] _ in self?.poseChanged() }, for: .valueChanged)
        return poseControl
    }

    private func poseChanged() {
        pose = Pose.allCases[poseControl.selectedSegmentIndex]
        onChange?()
    }
}
