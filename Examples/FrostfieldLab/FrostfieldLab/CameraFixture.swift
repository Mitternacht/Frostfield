import Frostfield
import simd
import UIKit

/// FF-06 fixture: the diagnostic triangle placed by an app-supplied model
/// transform and viewed through a perspective camera.
///
/// The app owns all of this state. A pose button or the field-of-view slider
/// changes a value, and the host draws again; the renderer only rewrites its
/// per-frame uniforms, which the status panel's pipeline and buffer counters
/// show. While the host draws continuously, the triangle also spins about its
/// own y axis, so translation, rotation, scale, and perspective combine.
final class CameraFixture {
    /// Deterministic poses: at animation time 0 each shows one part of the
    /// transform on its own.
    enum Pose: String, CaseIterable {
        case rest, moved, turned, scaled

        var title: String {
            switch self {
            case .rest: "Rest"
            case .moved: "Moved"
            case .turned: "Turned"
            case .scaled: "Scaled"
            }
        }

        var transform: Transform {
            switch self {
            case .rest:
                .identity
            case .moved:
                // Right, up, and farther from the camera.
                Transform(translation: [0.4, 0.5, -1.5])
            case .turned:
                // 60° about +y: the right (blue) corner turns away.
                Transform(rotation: simd_quatf(angle: .pi / 3, axis: [0, 1, 0]))
            case .scaled:
                Transform(scale: [1.5, 0.6, 1])
            }
        }
    }

    /// Radians per second of spin about the model's y axis while drawing
    /// continuously.
    static let spinSpeed: Float = 0.8
    /// The eye position: 4.5 units in front of the origin and 0.5 below it,
    /// looking along -z, so the triangle appears above the control panel.
    static let cameraPosition: SIMD3<Float> = [0, -0.5, 4.5]
    static let fieldOfViewRange: ClosedRange<Float> = 20...100

    private(set) var pose: Pose
    private(set) var fieldOfViewDegrees: Float = 60
    private(set) var camera: Camera

    /// Called after the app changes the pose or the field of view.
    var onChange: (() -> Void)?

    private let poseControl = UISegmentedControl(items: Pose.allCases.map(\.title))
    private let fieldOfViewSlider = UISlider()
    private let fieldOfViewLabel = UILabel()

    init(pose: Pose = .rest) {
        self.pose = pose
        guard let projection = Self.projection(degrees: fieldOfViewDegrees) else {
            preconditionFailure("The default field of view lies inside the valid range")
        }
        camera = Camera(position: Self.cameraPosition, projection: projection)
    }

    /// The model transform at an animation time: the pose, plus the spin.
    func transform(animationTime: Double) -> Transform {
        var transform = pose.transform
        let spin = simd_quatf(angle: Float(animationTime) * Self.spinSpeed, axis: [0, 1, 0])
        transform.rotation = transform.rotation * spin
        return transform
    }

    /// Fields for the status label's accessibility value.
    var statusFields: [(String, String)] {
        [("pose", pose.rawValue), ("fieldOfView", "\(Int(fieldOfViewDegrees.rounded()))")]
    }

    // MARK: Controls

    /// The pose picker and the field-of-view slider.
    func makeControls() -> UIView {
        poseControl.selectedSegmentIndex = Pose.allCases.firstIndex(of: pose) ?? 0
        poseControl.accessibilityIdentifier = "cameraPose"
        poseControl.accessibilityLabel = "Pose"
        poseControl.addAction(UIAction { [weak self] _ in self?.poseChanged() }, for: .valueChanged)

        fieldOfViewSlider.minimumValue = Self.fieldOfViewRange.lowerBound
        fieldOfViewSlider.maximumValue = Self.fieldOfViewRange.upperBound
        fieldOfViewSlider.value = fieldOfViewDegrees
        fieldOfViewSlider.accessibilityIdentifier = "fieldOfView"
        fieldOfViewSlider.accessibilityLabel = "Field of view"
        fieldOfViewSlider.addAction(UIAction { [weak self] _ in self?.fieldOfViewChanged() }, for: .valueChanged)

        fieldOfViewLabel.font = .preferredFont(forTextStyle: .footnote)
        fieldOfViewLabel.adjustsFontForContentSizeCategory = true
        fieldOfViewLabel.textColor = .label
        fieldOfViewLabel.setContentHuggingPriority(.required, for: .horizontal)
        fieldOfViewLabel.isAccessibilityElement = false
        updateFieldOfViewLabel()

        let sliderRow = UIStackView(arrangedSubviews: [fieldOfViewLabel, fieldOfViewSlider])
        sliderRow.axis = .horizontal
        sliderRow.spacing = 8
        let controls = UIStackView(arrangedSubviews: [poseControl, sliderRow])
        controls.axis = .vertical
        controls.spacing = 8
        return controls
    }

    private func poseChanged() {
        pose = Pose.allCases[poseControl.selectedSegmentIndex]
        onChange?()
    }

    private func fieldOfViewChanged() {
        let degrees = fieldOfViewSlider.value.rounded()
        guard degrees != fieldOfViewDegrees, let projection = Self.projection(degrees: degrees) else { return }
        fieldOfViewDegrees = degrees
        camera.projection = projection
        updateFieldOfViewLabel()
        onChange?()
    }

    private func updateFieldOfViewLabel() {
        fieldOfViewLabel.text = "FOV \(Int(fieldOfViewDegrees))°"
        fieldOfViewSlider.accessibilityValue = "\(Int(fieldOfViewDegrees)) degrees"
    }

    /// A projection for a field of view in degrees, or `nil` when the values
    /// are rejected (never for the slider's range).
    private static func projection(degrees: Float) -> PerspectiveProjection? {
        do {
            return try PerspectiveProjection(verticalFieldOfView: degrees * .pi / 180, near: 0.1, far: 20)
        } catch {
            print("FrostfieldLab rejected a field of view: \(error)")
            return nil
        }
    }
}
