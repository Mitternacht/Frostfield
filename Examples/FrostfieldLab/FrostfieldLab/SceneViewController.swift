import Frostfield
import MetalKit
import UIKit

/// One window scene's render host: a ``SceneRenderView`` driven by the
/// package's `TriangleRenderer` and paced by a `RenderSchedule`.
///
/// The host draws one ``LabFixture`` at a time: the clip-space diagnostic
/// triangle used by the lifecycle tests, or the ``CameraFixture``.
///
/// Each scene creates its own controller, renderer, schedule, and view. The
/// host reports facts from its own context to the schedule (the scene's
/// activity from its scene delegate, visibility, and the view's bounds and
/// display scale) and applies the schedule's answer to the `MTKView`. Nothing
/// here is shared between scenes.
final class SceneViewController: UIViewController {
    /// Radians of triangle rotation per second of animation time.
    private static let rotationSpeed: Float = 0.8

    private var renderer: TriangleRenderer?
    private var renderView: SceneRenderView?
    private var schedule = RenderSchedule(mode: .requested)
    private var fixture = LabLaunchOptions.fixture
    private let cameraFixture = CameraFixture(pose: LabLaunchOptions.cameraPose)
    private var counters = HostCounters()
    private var isAppeared = false
    private var displayScale: CGFloat = 0

    private let statusLabel = UILabel()
    private let fixtureControl = UISegmentedControl(items: LabFixture.allCases.map(\.title))
    private var cameraControls: UIView?
    private let modeControl = UISegmentedControl(items: ["Requested", "Continuous"])
    private let redrawButton = UIButton(configuration: .gray())
    private let visibilityButton = UIButton(configuration: .gray())
    private let newWindowButton = UIButton(configuration: .gray())

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black

        let device = LabLaunchOptions.simulatesMissingDevice ? nil : MTLCreateSystemDefaultDevice()
        let renderView = SceneRenderView(frame: view.bounds, device: device)
        do {
            renderer = try TriangleRenderer(
                device: device,
                colorPixelFormat: renderView.colorPixelFormat,
                onFrameCompleted: { [weak self] _ in
                    Task { @MainActor in self?.frameDidComplete() }
                }
            )
        } catch {
            showFailure(error)
            return
        }

        renderView.translatesAutoresizingMaskIntoConstraints = false
        renderView.accessibilityIdentifier = "renderView"
        renderView.clearColor = MTLClearColor(red: 0.08, green: 0.09, blue: 0.11, alpha: 1)
        // Stopped until the schedule knows the scene, visibility, and size.
        renderView.isPaused = true
        renderView.enableSetNeedsDisplay = false
        renderView.delegate = self
        renderView.layoutDidChange = { [weak self] boundsSize, displayScale in
            self?.renderViewDidLayout(boundsSize: boundsSize, displayScale: displayScale)
        }
        view.addSubview(renderView)
        NSLayoutConstraint.activate([
            renderView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            renderView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            renderView.topAnchor.constraint(equalTo: view.topAnchor),
            renderView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
        self.renderView = renderView

        installControls()
        updateStatus()
    }

    override func viewIsAppearing(_ animated: Bool) {
        super.viewIsAppearing(animated)
        isAppeared = true
        updateVisibility()
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        isAppeared = false
        updateVisibility()
    }

    // MARK: Facts reported to the schedule

    /// Called by this scene's delegate for each lifecycle transition of its
    /// own window scene.
    func sceneActivityDidChange(_ activity: RenderSchedule.Activity) {
        schedule.setActivity(activity)
        applySchedule()
    }

    private func updateVisibility() {
        guard let renderView else { return }
        schedule.setVisible(isAppeared && !renderView.isHidden)
        applySchedule()
    }

    private func renderViewDidLayout(boundsSize: CGSize, displayScale: CGFloat) {
        guard let renderView else { return }
        self.displayScale = displayScale
        if schedule.updateDrawableSize(boundsSize: boundsSize, displayScale: displayScale),
           let drawableSize = schedule.drawableSize {
            renderView.drawableSize = drawableSize
        }
        applySchedule()
    }

    private func frameDidComplete() {
        schedule.frameDidComplete()
        applySchedule()
    }

    /// Applies the schedule's draw-loop settings and display request to the
    /// view. Never waits for a frame slot or a drawable.
    private func applySchedule() {
        guard let renderView else { return }
        let loop = schedule.drawLoop
        if renderView.enableSetNeedsDisplay != loop.enableSetNeedsDisplay {
            renderView.enableSetNeedsDisplay = loop.enableSetNeedsDisplay
        }
        if renderView.isPaused != loop.isPaused {
            renderView.isPaused = loop.isPaused
        }
        if schedule.shouldRequestDisplay {
            renderView.setNeedsDisplay()
        }
        updateStatus()
    }

    // MARK: Controls

    private func installControls() {
        fixtureControl.selectedSegmentIndex = LabFixture.allCases.firstIndex(of: fixture) ?? 0
        fixtureControl.accessibilityIdentifier = "fixture"
        fixtureControl.accessibilityLabel = "Fixture"
        fixtureControl.addAction(UIAction { [weak self] _ in self?.fixtureChanged() }, for: .valueChanged)

        let cameraControls = cameraFixture.makeControls()
        cameraFixture.onChange = { [weak self] in self?.redraw() }
        self.cameraControls = cameraControls
        updateFixtureControls()

        modeControl.selectedSegmentIndex = 0
        modeControl.accessibilityIdentifier = "drawingMode"
        modeControl.addAction(UIAction { [weak self] _ in self?.modeChanged() }, for: .valueChanged)

        redrawButton.configuration?.title = "Redraw"
        redrawButton.accessibilityIdentifier = "redraw"
        redrawButton.addAction(UIAction { [weak self] _ in self?.redraw() }, for: .primaryActionTriggered)

        visibilityButton.accessibilityIdentifier = "toggleVisibility"
        visibilityButton.addAction(UIAction { [weak self] _ in self?.toggleVisibility() }, for: .primaryActionTriggered)
        updateVisibilityButton()

        newWindowButton.configuration?.title = "New Window"
        newWindowButton.accessibilityIdentifier = "newWindow"
        newWindowButton.addAction(UIAction { [weak self] _ in self?.openNewWindow() }, for: .primaryActionTriggered)
        newWindowButton.isHidden = !UIApplication.shared.supportsMultipleScenes

        statusLabel.accessibilityIdentifier = "frameStatus"
        statusLabel.accessibilityLabel = "Frame statistics"
        statusLabel.numberOfLines = 0
        statusLabel.font = UIFontMetrics(forTextStyle: .footnote)
            .scaledFont(for: .monospacedDigitSystemFont(ofSize: 13, weight: .regular))
        statusLabel.adjustsFontForContentSizeCategory = true
        statusLabel.textColor = .label

        let buttons = UIStackView(arrangedSubviews: [redrawButton, visibilityButton, newWindowButton])
        buttons.axis = .horizontal
        buttons.spacing = 8
        buttons.distribution = .fillProportionally

        let panel = UIStackView(arrangedSubviews: [statusLabel, fixtureControl, cameraControls, modeControl, buttons])
        panel.axis = .vertical
        panel.spacing = 8
        panel.isLayoutMarginsRelativeArrangement = true
        panel.directionalLayoutMargins = NSDirectionalEdgeInsets(top: 12, leading: 12, bottom: 12, trailing: 12)
        panel.backgroundColor = .systemBackground.withAlphaComponent(0.85)
        panel.layer.cornerRadius = 16
        panel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(panel)

        let guide = view.safeAreaLayoutGuide
        NSLayoutConstraint.activate([
            panel.leadingAnchor.constraint(equalTo: guide.leadingAnchor, constant: 12),
            panel.trailingAnchor.constraint(equalTo: guide.trailingAnchor, constant: -12),
            panel.bottomAnchor.constraint(equalTo: guide.bottomAnchor, constant: -12),
        ])
    }

    private func fixtureChanged() {
        fixture = LabFixture.allCases[fixtureControl.selectedSegmentIndex]
        updateFixtureControls()
        redraw()
    }

    private func updateFixtureControls() {
        cameraControls?.isHidden = fixture != .camera
    }

    private func modeChanged() {
        schedule.setMode(modeControl.selectedSegmentIndex == 1 ? .continuous : .requested)
        applySchedule()
    }

    private func redraw() {
        schedule.invalidate()
        applySchedule()
    }

    private func toggleVisibility() {
        guard let renderView else { return }
        renderView.isHidden.toggle()
        updateVisibilityButton()
        updateVisibility()
    }

    private func updateVisibilityButton() {
        visibilityButton.configuration?.title = (renderView?.isHidden ?? false) ? "Show" : "Hide"
    }

    private func openNewWindow() {
        let request = UISceneSessionActivationRequest(role: .windowApplication)
        UIApplication.shared.activateSceneSession(for: request) { error in
            print("FrostfieldLab could not open a window: \(error)")
        }
    }

    // MARK: Status

    private func updateStatus() {
        guard let renderer else { return }
        let statistics = renderer.statistics
        let resources = renderer.resourceStatistics
        let size = schedule.drawableSize.map { "\(Int($0.width))x\(Int($0.height))" } ?? "none"
        statusLabel.text = """
            \(schedule.mode) · \(schedule.activity) · \(schedule.pacing) · \(size) px @\(displayScale.formatted())x
            submitted \(statistics.submitted) · completed \(statistics.completed) · \
            skipped \(statistics.skippedNoSlot) slot / \(statistics.skippedNoTarget) drawable
            callbacks \(counters.callbacks) · refused \(counters.refused) · \
            while not active \(counters.callbacksWhileNotActive) / \(counters.submittedWhileNotActive)
            created \(resources.renderPipelinesCreated) pipelines · \(resources.buffersAllocated) buffers \
            (\(resources.bufferBytesAllocated) B)
            """
        var fields: [(String, String)] = [
            ("fixture", fixture.rawValue),
            ("mode", "\(schedule.mode)"),
            ("activity", "\(schedule.activity)"),
            ("pacing", "\(schedule.pacing)"),
            ("drawable", size),
            ("scale", "\(Double(displayScale))"),
            ("submitted", "\(statistics.submitted)"),
            ("completed", "\(statistics.completed)"),
            ("skippedNoSlot", "\(statistics.skippedNoSlot)"),
            ("skippedNoTarget", "\(statistics.skippedNoTarget)"),
            ("callbacks", "\(counters.callbacks)"),
            ("refused", "\(counters.refused)"),
            ("callbacksWhileNotActive", "\(counters.callbacksWhileNotActive)"),
            ("submittedWhileNotActive", "\(counters.submittedWhileNotActive)"),
            ("longestDrawableWaitMicroseconds", "\(counters.longestDrawableWaitMicroseconds)"),
            ("longestDrawableWaitWhileNotActiveMicroseconds", "\(counters.longestDrawableWaitWhileNotActiveMicroseconds)"),
            ("pipelinesCreated", "\(resources.renderPipelinesCreated)"),
            ("buffersAllocated", "\(resources.buffersAllocated)"),
            ("bufferBytesAllocated", "\(resources.bufferBytesAllocated)"),
        ]
        if fixture == .camera {
            fields += cameraFixture.statusFields
        }
        statusLabel.accessibilityValue = fields.map { "\($0)=\($1)" }.joined(separator: ";")
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
        // The schedule already invalidated when it computed this size in
        // `renderViewDidLayout`; nothing else depends on the drawable size.
    }

    func draw(in view: MTKView) {
        guard let renderer else { return }
        let isActive = schedule.activity == .active
        counters.callbacks += 1
        if !isActive { counters.callbacksWhileNotActive += 1 }

        guard let animationTime = schedule.frameWillBegin(at: CACurrentMediaTime()) else {
            counters.refused += 1
            updateStatus()
            return
        }
        // The renderer asks for a target only after it holds a free frame
        // slot, so a drawable is never held while every slot is busy.
        let target = { [self] () -> FrameTarget? in
            let start = ContinuousClock.now
            let pass = view.currentRenderPassDescriptor
            let drawable = view.currentDrawable
            counters.recordDrawableWait(start.duration(to: .now), sceneIsActive: isActive)
            guard let pass, let drawable else { return nil }
            return FrameTarget(passDescriptor: pass, drawable: drawable)
        }
        let outcome = switch fixture {
        case .triangle:
            renderer.drawFrame(rotation: Float(animationTime) * Self.rotationSpeed, target: target)
        case .camera:
            renderer.drawFrame(
                transform: cameraFixture.transform(animationTime: animationTime),
                camera: cameraFixture.camera,
                target: target
            )
        }
        if case .submitted = outcome, !isActive {
            counters.submittedWhileNotActive += 1
        }
        schedule.frameDidEnd(outcome)
        applySchedule()
    }
}

/// Host-side counters the lab shows next to the renderer's statistics. They
/// measure what the view and schedule did, independently of the renderer.
struct HostCounters {
    /// `draw(in:)` callbacks from the view.
    var callbacks = 0
    /// Callbacks the schedule did not allow to submit a frame.
    var refused = 0
    /// Callbacks while the scene was not active.
    var callbacksWhileNotActive = 0
    /// Frames submitted while the scene was not active.
    var submittedWhileNotActive = 0
    /// The longest time the main thread spent obtaining the view's drawable
    /// (`CAMetalLayer.nextDrawable()` inside `MTKView.currentDrawable`)
    /// while the scene was active.
    var longestDrawableWaitMicroseconds = 0
    /// The same while the scene was not active.
    var longestDrawableWaitWhileNotActiveMicroseconds = 0

    mutating func recordDrawableWait(_ duration: Duration, sceneIsActive: Bool) {
        let microseconds = Int(duration / .microseconds(1))
        if sceneIsActive {
            longestDrawableWaitMicroseconds = max(longestDrawableWaitMicroseconds, microseconds)
        } else {
            longestDrawableWaitWhileNotActiveMicroseconds = max(longestDrawableWaitWhileNotActiveMicroseconds, microseconds)
        }
    }
}

/// What a scene's host draws.
enum LabFixture: String, CaseIterable {
    /// The clip-space diagnostic triangle (FF-02/FF-03 host fixture).
    case triangle
    /// The triangle placed by a transform under a perspective camera (FF-06).
    case camera

    var title: String {
        switch self {
        case .triangle: "Triangle"
        case .camera: "Camera"
        }
    }
}

/// Launch arguments that select lab fixtures.
enum LabLaunchOptions {
    /// `-FrostfieldLabFixture <name>` starts with that ``LabFixture``
    /// (`triangle` by default).
    static var fixture: LabFixture {
        value(after: "-FrostfieldLabFixture").flatMap(LabFixture.init(rawValue:)) ?? .triangle
    }

    /// `-FrostfieldLabCameraPose <name>` starts the camera fixture in that
    /// pose (`rest` by default).
    static var cameraPose: CameraFixture.Pose {
        value(after: "-FrostfieldLabCameraPose").flatMap(CameraFixture.Pose.init(rawValue:)) ?? .rest
    }

    private static func value(after flag: String) -> String? {
        let arguments = ProcessInfo.processInfo.arguments
        guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else { return nil }
        return arguments[index + 1]
    }

    /// `-FrostfieldLabSimulateMissingDevice` hands the renderer no Metal device,
    /// so the failure state can be inspected on hardware that has one.
    static var simulatesMissingDevice: Bool {
        ProcessInfo.processInfo.arguments.contains("-FrostfieldLabSimulateMissingDevice")
    }
}
