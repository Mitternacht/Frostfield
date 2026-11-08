import Frostfield
import MetalKit
import UIKit

/// One window scene's render host: an `MTKView` driven by the package's
/// `TriangleRenderer`. Each scene creates its own controller, renderer, and
/// view; nothing here is shared between scenes.
final class SceneViewController: UIViewController {
    private var renderer: TriangleRenderer?
    private var renderView: MTKView?
    private let statusLabel = UILabel()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black

        let device = LabLaunchOptions.simulatesMissingDevice ? nil : MTLCreateSystemDefaultDevice()
        let renderView = MTKView(frame: view.bounds, device: device)
        do {
            renderer = try TriangleRenderer(
                device: device,
                colorPixelFormat: renderView.colorPixelFormat,
                onFrameCompleted: { [weak self] _ in
                    Task { @MainActor in self?.updateStatus() }
                }
            )
        } catch {
            showFailure(error)
            return
        }

        renderView.translatesAutoresizingMaskIntoConstraints = false
        renderView.accessibilityIdentifier = "renderView"
        renderView.clearColor = MTLClearColor(red: 0.08, green: 0.09, blue: 0.11, alpha: 1)
        // Draw when invalidated rather than on every display refresh.
        renderView.isPaused = true
        renderView.enableSetNeedsDisplay = true
        renderView.delegate = self
        view.addSubview(renderView)
        NSLayoutConstraint.activate([
            renderView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            renderView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            renderView.topAnchor.constraint(equalTo: view.topAnchor),
            renderView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
        self.renderView = renderView

        installStatusLabel()
        updateStatus()
    }

    private func installStatusLabel() {
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        statusLabel.accessibilityIdentifier = "frameStatus"
        statusLabel.numberOfLines = 0
        statusLabel.font = .preferredFont(forTextStyle: .footnote)
        statusLabel.adjustsFontForContentSizeCategory = true
        statusLabel.textColor = .white
        view.addSubview(statusLabel)
        let guide = view.safeAreaLayoutGuide
        NSLayoutConstraint.activate([
            statusLabel.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 16),
            statusLabel.trailingAnchor.constraint(lessThanOrEqualTo: guide.trailingAnchor, constant: -16),
            statusLabel.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -16),
        ])
    }

    private func updateStatus() {
        guard let renderer else { return }
        let statistics = renderer.statistics
        statusLabel.text = "Frames submitted \(statistics.submitted), completed \(statistics.completed)"
        statusLabel.accessibilityValue = "submitted=\(statistics.submitted);completed=\(statistics.completed)"
    }

    /// Shows the renderer's error instead of a render view, so a failure is
    /// never mistaken for an empty scene.
    private func showFailure(_ error: RendererError) {
        view.backgroundColor = .systemBackground
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.accessibilityIdentifier = "renderFailure"
        label.numberOfLines = 0
        label.textAlignment = .center
        label.font = .preferredFont(forTextStyle: .body)
        label.adjustsFontForContentSizeCategory = true
        label.textColor = .label
        label.text = "Rendering unavailable\n\(error.description)"
        view.addSubview(label)
        let guide = view.readableContentGuide
        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: guide.leadingAnchor),
            label.trailingAnchor.constraint(equalTo: guide.trailingAnchor),
            label.centerYAnchor.constraint(equalTo: view.safeAreaLayoutGuide.centerYAnchor),
        ])
    }
}

extension SceneViewController: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        view.setNeedsDisplay()
    }

    func draw(in view: MTKView) {
        guard let renderer else { return }
        _ = renderer.drawFrame(rotation: 0) {
            guard let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable else {
                return nil
            }
            return FrameTarget(passDescriptor: pass, drawable: drawable)
        }
        updateStatus()
    }
}

/// Launch arguments that select lab fixtures.
enum LabLaunchOptions {
    /// `-FrostfieldLabSimulateMissingDevice` hands the renderer no Metal device,
    /// so the failure state can be inspected on hardware that has one.
    static var simulatesMissingDevice: Bool {
        ProcessInfo.processInfo.arguments.contains("-FrostfieldLabSimulateMissingDevice")
    }
}
