import Frostfield
import Metal
import simd
import UIKit

/// FF-08 fixture: the cube fixture's mesh shaded by a material instead of
/// the normal diagnostic.
///
/// - **Unlit**: Frostfield's built-in unlit material. The color picker changes
///   its parameter.
/// - **Pulse**: `lab_pulse_fragment`, a fragment function in the app's own
///   shader library (`LabShaders.metal`), with a parameter record declared in
///   `LabShaderTypes.h`. While the host draws continuously, the app writes the
///   animation time into the material's parameters every frame, so bands of
///   color travel up the cube.
///
/// Both materials are made when the scene loads, which prepares their
/// pipelines before any interaction. Choosing a material or a color and
/// animating the time change only parameter values, so the renderer's
/// pipeline counter stays constant. The cube itself stays in the corner pose.
final class MaterialFixture {
    enum Choice: String, CaseIterable {
        case unlit, pulse

        var title: String {
            switch self {
            case .unlit: "Unlit"
            case .pulse: "Pulse"
            }
        }
    }

    enum UnlitColor: String, CaseIterable {
        case amber, teal

        var title: String {
            switch self {
            case .amber: "Amber"
            case .teal: "Teal"
            }
        }

        var color: SIMD4<Float> {
            switch self {
            case .amber: [1.0, 0.6, 0.1, 1]
            case .teal: [0.1, 0.7, 0.7, 1]
            }
        }
    }

    /// The pulse bands' speed, in radians of phase per second.
    static let pulseSpeed: Float = 2.5

    let mesh: Mesh
    let camera: Camera
    let transform = GeometryFixture.Pose.corner.transform
    private(set) var choice: Choice
    private(set) var unlitColor = UnlitColor.amber
    private var unlit: Material<UnlitParameters>
    private var pulse: Material<LabPulseParameters>

    /// Called after the app changes the material or its color.
    var onChange: (() -> Void)?

    private let choiceControl = UISegmentedControl(items: Choice.allCases.map(\.title))
    private let colorControl = UISegmentedControl(items: UnlitColor.allCases.map(\.title))

    /// Makes both materials with `renderer`, preparing their pipelines. The
    /// pulse function comes from the app's default library.
    init(renderer: MeshRenderer, mesh: Mesh, camera: Camera, choice: Choice) throws {
        let appLibrary = try renderer.device.makeDefaultLibrary(bundle: .main)
        self.mesh = mesh
        self.camera = camera
        self.choice = choice
        unlit = try renderer.makeUnlitMaterial(color: UnlitColor.amber.color)
        pulse = try renderer.makeMaterial(
            fragmentFunction: "lab_pulse_fragment",
            library: appLibrary,
            parameters: LabPulseParameters(
                coolColor: [0.15, 0.25, 0.85, 1],
                warmColor: [1.0, 0.45, 0.2, 1],
                time: 0,
                speed: Self.pulseSpeed,
                padding: (0, 0)
            )
        )
    }

    /// Draws the mesh with the chosen material. The pulse material receives
    /// the animation time as a parameter first.
    func drawFrame(renderer: MeshRenderer, animationTime: Double, target: () -> FrameTarget?) -> FrameOutcome {
        switch choice {
        case .unlit:
            return renderer.drawFrame(mesh: mesh, material: unlit, transform: transform, camera: camera, target: target)
        case .pulse:
            pulse.parameters.time = Float(animationTime)
            return renderer.drawFrame(mesh: mesh, material: pulse, transform: transform, camera: camera, target: target)
        }
    }

    /// Fields for the status label's accessibility value.
    var statusFields: [(String, String)] {
        [
            ("material", choice.rawValue),
            ("materialFunction", choice == .unlit ? unlit.fragmentFunctionName : pulse.fragmentFunctionName),
            ("unlitColor", unlitColor.rawValue),
            ("materialTime", String(format: "%.3f", pulse.parameters.time)),
            ("meshVertices", "\(mesh.vertexCount)"),
        ]
    }

    func makeControls() -> UIView {
        choiceControl.selectedSegmentIndex = Choice.allCases.firstIndex(of: choice) ?? 0
        choiceControl.accessibilityIdentifier = "material"
        choiceControl.accessibilityLabel = "Material"
        choiceControl.addAction(UIAction { [weak self] _ in self?.choiceChanged() }, for: .valueChanged)

        colorControl.selectedSegmentIndex = UnlitColor.allCases.firstIndex(of: unlitColor) ?? 0
        colorControl.accessibilityIdentifier = "unlitColor"
        colorControl.accessibilityLabel = "Unlit color"
        colorControl.addAction(UIAction { [weak self] _ in self?.colorChanged() }, for: .valueChanged)
        colorControl.isEnabled = choice == .unlit

        let controls = UIStackView(arrangedSubviews: [choiceControl, colorControl])
        controls.axis = .vertical
        controls.spacing = 8
        return controls
    }

    private func choiceChanged() {
        choice = Choice.allCases[choiceControl.selectedSegmentIndex]
        colorControl.isEnabled = choice == .unlit
        onChange?()
    }

    private func colorChanged() {
        unlitColor = UnlitColor.allCases[colorControl.selectedSegmentIndex]
        unlit.parameters.color = unlitColor.color
        onChange?()
    }
}
