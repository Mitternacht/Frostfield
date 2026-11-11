import Foundation
import Frostfield
import Metal
import QuartzCore
#if os(macOS)
import AppKit
typealias LabViewController = NSViewController
private typealias LabSelectionGesture = NSClickGestureRecognizer
private typealias LabOrbitGesture = NSPanGestureRecognizer
private typealias LabZoomGesture = NSMagnificationGestureRecognizer
private typealias LabPlatformView = NSView
private typealias LabWindow = NSWindow
#else
import UIKit
typealias LabViewController = UIViewController
private typealias LabSelectionGesture = UITapGestureRecognizer
private typealias LabOrbitGesture = UIPanGestureRecognizer
private typealias LabZoomGesture = UIPinchGestureRecognizer
private typealias LabPlatformView = UIView
private typealias LabWindow = UIWindow
#endif

@MainActor
private final class LabMetalView: LabPlatformView {
    var activityChanged: (() -> Void)?
    var metalLayer: CAMetalLayer? { layer as? CAMetalLayer }

    #if os(macOS)
    var wheelZoom: ((Float) -> Void)?

    override func makeBackingLayer() -> CALayer { CAMetalLayer() }

    override func scrollWheel(with event: NSEvent) {
        if let wheelZoom {
            wheelZoom(exp(Float(event.scrollingDeltaY) * 0.018))
        } else {
            super.scrollWheel(with: event)
        }
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        activityChanged?()
    }

    override func viewDidHide() {
        super.viewDidHide()
        activityChanged?()
    }

    override func viewDidUnhide() {
        super.viewDidUnhide()
        activityChanged?()
    }
    #else
    override class var layerClass: AnyClass { CAMetalLayer.self }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        activityChanged?()
    }

    override var isHidden: Bool {
        didSet {
            if oldValue != isHidden { activityChanged?() }
        }
    }
    #endif
}

final class TriangleViewController: LabViewController, @preconcurrency CAMetalDisplayLinkDelegate {
    private var renderer: TriangleRenderer?
    private var meshFixture: LabMeshFixture?
    private var graphFixture: LabGraphFixture?
    private var oceanFixture: LabOceanFixture?
    private var studioFixture: LabStudioFixture?
    private var pathTracingFixture: LabPathTracingFixture?
    private var pathLightingSettings: LabPathLightSettings?
    private var labScene: (any LabSceneFixture)?
    private var sceneControls: LabSceneControls?
    private var oceanPaused = false
    private var schedule = RenderSchedule()
    private var isPresented = false
    private var displayLink: CAMetalDisplayLink?
    private var drawableSize: CGSize?
    private var depthTexture: (any MTLTexture)?
    private weak var observedWindow: LabWindow?
    #if !os(macOS)
    private weak var observedScene: UIWindowScene?
    private var isDeactivating = false
    #endif
    private var submittedFrames = 0
    private var renderMetrics: LabRenderMetrics?
    private let traceSchedule = ProcessInfo.processInfo.arguments.contains("--trace-render-schedule")

    private var hasFixture: Bool {
        renderer != nil || meshFixture != nil || graphFixture != nil || oceanFixture != nil
            || studioFixture != nil || pathTracingFixture != nil || labScene != nil
    }

    override func loadView() {
        let arguments = ProcessInfo.processInfo.arguments
        if arguments.contains("--render-metrics") || arguments.contains("--render-metrics-sustained") {
            renderMetrics = LabRenderMetrics(arguments: arguments)
        }
        guard !arguments.contains("--simulate-no-metal"), let device = MTLCreateSystemDefaultDevice() else {
            view = failureView("No Metal device is available.")
            return
        }

        let graphMode = LabGraphFixture.Mode.requested(arguments)
        let showGraph = graphMode != nil
        let showMesh = LabMeshFixture.isRequested(arguments)
        let showOcean = arguments.contains(LabOceanFixture.argument)
        let showAssetStudio = arguments.contains(LabStudioFixture.assetStudioArgument)
        let showProcedural = arguments.contains(LabStudioFixture.proceduralArgument)
        let showStudio = showAssetStudio || showProcedural || arguments.contains(LabStudioFixture.argument)
        let showIceberg = arguments.contains(LabPathTracingFixture.icebergArgument)
        let showPorsche = arguments.contains(LabPathTracingFixture.porscheArgument)
        let showPathTracing = showIceberg || showPorsche || arguments.contains(LabPathTracingFixture.argument) || LabPathLightingHost.isRequested
        let metalView = LabMetalView(frame: .zero)
        #if os(macOS)
        metalView.wantsLayer = true
        #endif
        guard let layer = metalView.metalLayer else {
            view = failureView("The native view could not create a Metal layer.")
            return
        }
        layer.device = device
        layer.pixelFormat = .bgra8Unorm_srgb
        layer.framebufferOnly = true
        layer.isOpaque = true
        let requestedScene = LabScenes.requested(arguments)
        let label: String
        if let requestedScene {
            label = requestedScene.title
        } else if showOcean {
            label = "Cyber Ocean animated dolphin"
        } else if showPorsche {
            label = "Procedural Porsche 911 GT3 in a photography studio"
        } else if showIceberg {
            label = "Iceberg in an open ocean"
        } else if showPathTracing {
            label = arguments.contains("--path-light-room") ? "Path-traced lighting room" : "Path-traced procedural rangefinder camera"
        } else if showAssetStudio {
            label = "Asset Studio rangefinder with a baked finish and floor shadows"
        } else if showStudio {
            label = showProcedural ? "Procedural rangefinder camera" : "Textured antique camera in a photography studio"
        } else {
            label = showGraph ? "Scene hierarchy" : (showMesh ? "Colored cube" : "Multicolored triangle")
        }
        #if os(macOS)
        metalView.setAccessibilityElement(true)
        metalView.setAccessibilityLabel(label)
        #else
        metalView.isAccessibilityElement = true
        metalView.accessibilityLabel = label
        #endif

        do {
            if let requestedScene {
                let controls = sceneControls ?? LabSceneControls()
                sceneControls = controls
                labScene = try LabScenes.make(
                    requestedScene, device: device, destinationFormat: layer.pixelFormat, controls: controls
                )
            } else if showOcean {
                oceanFixture = try LabOceanFixture(device: device, destinationFormat: layer.pixelFormat)
            } else if showPathTracing {
                let subject: LabPathTracingFixture.Subject
                if showPorsche { subject = .porsche }
                else if showIceberg { subject = .iceberg }
                else if arguments.contains("--path-light-room") { subject = .lightRoom }
                else { subject = .rangefinder }
                let fixture = try LabPathTracingFixture(
                    device: device, destinationFormat: layer.pixelFormat, detail: arguments.contains("--studio-detail"),
                    subject: subject,
                    generatedFinish: LabPathLightingHost.isRequested || arguments.contains("--shader-finish")
                )
                if let pathLightingSettings { try fixture.setLighting(pathLightingSettings) }
                pathTracingFixture = fixture
            } else if showStudio {
                studioFixture = try LabStudioFixture(
                    device: device, destinationFormat: layer.pixelFormat,
                    closeUp: arguments.contains("--studio-detail"), showsEnvironment: arguments.contains("--studio-sky"),
                    usesImportedLights: arguments.contains("--studio-light-rig"),
                    occlusionEnabled: !arguments.contains("--studio-no-ao"),
                    sourceURL: try LabStudioFixture.sourceURL(arguments: arguments), procedural: showProcedural,
                    generatedFinish: arguments.contains("--shader-finish"), bakesAssets: showAssetStudio
                )
            } else if let graphMode {
                graphFixture = try LabGraphFixture(
                    device: device, pixelFormat: layer.pixelFormat, mode: graphMode
                )
            } else if showMesh {
                meshFixture = try LabMeshFixture(
                    device: device, pixelFormat: layer.pixelFormat, arguments: arguments
                )
            } else if arguments.contains("--simulate-bad-shader") {
                renderer = try TriangleRenderer(
                    device: device, pixelFormat: layer.pixelFormat,
                    vertexFunctionName: "missing_fixture_vertex"
                )
            } else {
                renderer = try TriangleRenderer(device: device, pixelFormat: layer.pixelFormat)
            }
        } catch {
            trace("setup error=\(error.localizedDescription)")
            view = failureView(error.localizedDescription)
            return
        }

        if let graphFixture {
            let gesture = LabSelectionGesture(target: self, action: #selector(selectGraphNode(_:)))
            #if !os(macOS)
            gesture.cancelsTouchesInView = false
            #endif
            metalView.addGestureRecognizer(gesture)
            updateSelectionAccessibility(in: metalView, description: graphFixture.selectionDescription)
        }

        if oceanFixture != nil || studioFixture != nil || pathTracingFixture != nil || labScene != nil {
            metalView.addGestureRecognizer(LabOrbitGesture(target: self, action: #selector(orbitScene(_:))))
            metalView.addGestureRecognizer(LabZoomGesture(target: self, action: #selector(zoomScene(_:))))
            #if os(macOS)
            metalView.wheelZoom = { [weak self] scale in self?.zoomScene(by: scale) }
            #endif
        }

        if let oceanFixture {
            oceanFixture.setPaused(oceanPaused)
            schedule.setMode(oceanPaused ? .onDemand : .continuous)
        } else if arguments.contains("--continuous") || graphMode?.animates == true || labScene?.animates == true {
            schedule.setMode(.continuous)
        }
        metalView.activityChanged = { [weak self] in self?.viewActivityChanged() }
        view = metalView
        #if !os(macOS)
        registerForTraitChanges([UITraitDisplayScale.self]) { (controller: TriangleViewController, _) in
            controller.updateDrawableSize()
        }
        #endif
    }

    private func beginPresentation() {
        isPresented = true
        viewActivityChanged()
    }

    private func endPresentation() {
        isPresented = false
        updateActivity()
        stopDisplayLink()
        NotificationCenter.default.removeObserver(self)
        observedWindow = nil
        #if !os(macOS)
        observedScene = nil
        isDeactivating = false
        #endif
    }

    #if os(macOS)
    override func viewDidAppear() {
        super.viewDidAppear()
        beginPresentation()
    }

    override func viewWillDisappear() {
        endPresentation()
        super.viewWillDisappear()
    }

    override func viewDidLayout() {
        super.viewDidLayout()
        updateDrawableSize()
    }
    #else
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        beginPresentation()
    }

    override func viewWillDisappear(_ animated: Bool) {
        endPresentation()
        super.viewWillDisappear(animated)
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        updateDrawableSize()
    }
    #endif

    private func stopDisplayLink() {
        displayLink?.invalidate()
        displayLink = nil
    }

    private func viewActivityChanged() {
        guard isPresented, let metalView = view as? LabMetalView else { return }
        let window = metalView.window
        #if os(macOS)
        let sourceChanged = observedWindow !== window
        #else
        let scene = window?.windowScene
        let sourceChanged = observedWindow !== window || observedScene !== scene
        #endif
        if sourceChanged {
            stopDisplayLink()
            schedule.invalidate()
            NotificationCenter.default.removeObserver(self)
            observedWindow = window
            #if !os(macOS)
            observedScene = scene
            isDeactivating = false
            if let scene {
                for event in [UIScene.didActivateNotification, UIScene.willDeactivateNotification] {
                    NotificationCenter.default.addObserver(
                        self, selector: #selector(windowChanged(_:)), name: event, object: scene
                    )
                }
            }
            #endif
            if let window {
                #if os(macOS)
                let windowEvents = [
                    NSWindow.didBecomeKeyNotification, NSWindow.didResignKeyNotification,
                    NSWindow.didMiniaturizeNotification, NSWindow.didDeminiaturizeNotification,
                    NSWindow.didChangeOcclusionStateNotification, NSWindow.didChangeBackingPropertiesNotification
                ]
                #else
                let windowEvents = [UIWindow.didBecomeVisibleNotification, UIWindow.didBecomeHiddenNotification]
                #endif
                for event in windowEvents {
                    NotificationCenter.default.addObserver(
                        self, selector: #selector(windowChanged(_:)), name: event, object: window
                    )
                }
            }
        }
        updateDrawableSize()
        updateActivity()
    }

    @objc private func windowChanged(_ notification: Notification) {
        #if !os(macOS)
        if notification.name == UIScene.willDeactivateNotification { isDeactivating = true }
        if notification.name == UIScene.didActivateNotification { isDeactivating = false }
        #endif
        viewActivityChanged()
    }

    private func updateActivity() {
        guard hasFixture, let metalView = view as? LabMetalView else { return }
        let active: Bool
        #if os(macOS)
        if let window = metalView.window {
            active = isPresented && window.isKeyWindow && window.isVisible && !window.isMiniaturized
                && window.occlusionState.contains(.visible) && !metalView.isHiddenOrHasHiddenAncestor
        } else {
            active = false
        }
        #else
        active = isPresented && !isDeactivating && !metalView.isHidden
            && metalView.window?.isHidden == false
            && metalView.window?.windowScene?.activationState == .foregroundActive
        #endif
        let wasActive = schedule.isActive
        schedule.setActive(active)
        let continuous = schedule.shouldRunContinuously
        renderMetrics?.updateActivity(active: active, continuous: continuous)
        if active != wasActive {
            trace("active=\(active) continuous=\(continuous)")
        }
        requestDisplay(in: metalView)
    }

    private func updateDrawableSize() {
        guard let metalView = view as? LabMetalView, let layer = metalView.metalLayer else { return }
        guard metalView.window != nil else {
            drawableSize = nil
            depthTexture = nil
            stopDisplayLink()
            return
        }
        let backingSize: CGSize
        #if os(macOS)
        let scale = metalView.convertToBacking(CGSize(width: 1, height: 1)).width
        backingSize = metalView.convertToBacking(metalView.bounds.size)
        #else
        let scale = metalView.traitCollection.displayScale
        backingSize = CGSize(width: metalView.bounds.width * scale, height: metalView.bounds.height * scale)
        #endif
        guard scale.isFinite, scale > 0, backingSize.width.isFinite, backingSize.height.isFinite,
              backingSize.width > 0, backingSize.height > 0 else {
            drawableSize = nil
            displayLink?.isPaused = true
            return
        }
        #if os(macOS)
        let pathTracingLimit: CGFloat = 960
        #else
        let pathTracingLimit: CGFloat = 640
        #endif
        let previewLimit = labScene?.previewLimit ?? (pathTracingFixture == nil ? nil : pathTracingLimit)
        let previewScale = previewLimit.map { min(1, $0 / max(backingSize.width, backingSize.height)) } ?? 1
        let size = CGSize(width: (backingSize.width * previewScale).rounded(.up), height: (backingSize.height * previewScale).rounded(.up))
        if layer.contentsScale != scale { layer.contentsScale = scale }
        guard drawableSize != size else { return }
        drawableSize = size
        layer.drawableSize = size
        schedule.invalidate()
        trace("drawable=\(Int(size.width))x\(Int(size.height))")
        requestDisplay(in: metalView)
    }

    private func requestDisplay(in view: LabMetalView) {
        guard schedule.canDraw || schedule.shouldRunContinuously,
              drawableSize != nil, view.window != nil, let layer = view.metalLayer else {
            displayLink?.isPaused = true
            return
        }
        if let displayLink {
            displayLink.isPaused = false
            return
        }

        let link = CAMetalDisplayLink(metalLayer: layer)
        link.isPaused = true
        link.delegate = self
        displayLink = link
        link.add(to: .main, forMode: .common)
        link.isPaused = false
    }

    func metalDisplayLink(_ link: CAMetalDisplayLink, needsUpdate update: CAMetalDisplayLink.Update) {
        guard link === displayLink, schedule.canDraw, hasFixture, let drawableSize else { return }
        let frameNumber = submittedFrames + 1
        let measureFrame = renderMetrics?.wantsSample(frameNumber) == true
        let cpuStart = measureFrame ? CACurrentMediaTime() : nil
        let drawable = update.drawable
        let color = drawable.texture
        guard CGFloat(color.width) == drawableSize.width, CGFloat(color.height) == drawableSize.height else {
            renderMetrics?.missedDrawable()
            return
        }
        guard schedule.beginFrame() else { return }
        if !schedule.shouldRunContinuously { link.isPaused = true }

        #if os(macOS)
        if measureFrame, let renderMetrics {
            drawable.addPresentedHandler { [weak renderMetrics] drawable in
                let time = drawable.presentedTime
                Task { @MainActor in renderMetrics?.recordPresentation(frame: frameNumber, time: time) }
            }
        }
        #endif

        do {
            let pass = try makePass(color: color)
            let finish: @Sendable (String?, Double?) -> Void = { [weak self] errorMessage, gpuSeconds in
                Task { @MainActor in
                    self?.renderMetrics?.recordCompletion(
                        frame: frameNumber, succeeded: errorMessage == nil, gpuSeconds: gpuSeconds
                    )
                    self?.frameCompleted(errorMessage: errorMessage)
                }
            }
            let completed: @Sendable (String?) -> Void = { finish($0, nil) }
            if let labScene {
                try labScene.draw(drawable: drawable, completed: completed)
            } else if let oceanFixture {
                try oceanFixture.draw(pass: pass, drawable: drawable, completed: completed)
            } else if let studioFixture {
                try studioFixture.draw(pass: pass, drawable: drawable, completed: completed)
            } else if let pathTracingFixture {
                try pathTracingFixture.draw(drawable: drawable, completed: completed)
            } else if let graphFixture {
                let frame = schedule.shouldRunContinuously ? submittedFrames : 0
                try graphFixture.draw(pass: pass, drawable: drawable, frame: frame, completed: completed)
            } else if let meshFixture {
                let frame = schedule.shouldRunContinuously ? submittedFrames : 0
                try meshFixture.draw(
                    pass: pass, drawable: drawable, frame: frame,
                    collectGPUTiming: measureFrame, completed: finish
                )
            } else if let renderer {
                try renderer.draw(pass: pass, drawable: drawable, completed: completed)
            }
            submittedFrames += 1
            if let renderMetrics {
                renderMetrics.recordSubmission(
                    frame: frameNumber, cpuSeconds: cpuStart.map { CACurrentMediaTime() - $0 },
                    width: drawable.texture.width, height: drawable.texture.height
                )
            }
            if submittedFrames <= 3 || submittedFrames.isMultiple(of: 30) {
                trace("submitted")
            }
        } catch {
            frameCompleted(errorMessage: error.localizedDescription)
        }
    }

    private func makePass(color: any MTLTexture) throws -> MTLRenderPassDescriptor {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.012, green: 0.02, blue: 0.035, alpha: 1)

        if meshFixture != nil || graphFixture != nil || oceanFixture != nil || studioFixture != nil {
            if depthTexture?.width != color.width || depthTexture?.height != color.height {
                let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: MeshRenderer.depthPixelFormat, width: color.width, height: color.height,
                    mipmapped: false
                )
                descriptor.storageMode = .private
                descriptor.usage = studioFixture == nil ? .renderTarget : [.renderTarget, .shaderRead]
                guard let depth = color.device.makeTexture(descriptor: descriptor) else {
                    throw MeshRenderer.Failure.resource("lab depth texture")
                }
                depth.label = "FrostfieldLab depth"
                depthTexture = depth
            }
            pass.depthAttachment.texture = depthTexture
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.storeAction = .dontCare
            pass.depthAttachment.clearDepth = 1
        }
        return pass
    }

    private func frameCompleted(errorMessage: String?) {
        schedule.finishFrame()
        guard let metalView = view as? LabMetalView else { return }
        guard let errorMessage else {
            if pathTracingFixture?.needsMoreSamples == true || labScene?.needsMoreSamples == true { schedule.invalidate() }
            requestDisplay(in: metalView)
            return
        }

        schedule.setActive(false)
        renderMetrics?.updateActivity(active: false, continuous: false)
        stopDisplayLink()
        depthTexture = nil
        metalView.activityChanged = nil
        renderer = nil
        meshFixture = nil
        graphFixture = nil
        oceanFixture = nil
        studioFixture = nil
        pathTracingFixture = nil
        labScene = nil
        trace("frame error=\(errorMessage)")
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.view = self.failureView(errorMessage)
        }
    }

    func setPathLighting(_ settings: LabPathLightSettings) {
        guard settings != pathLightingSettings else { return }
        pathLightingSettings = settings
        guard let pathTracingFixture, let metalView = view as? LabMetalView else { return }
        do {
            try pathTracingFixture.setLighting(settings)
            redrawCamera(in: metalView)
        } catch {
            frameCompleted(errorMessage: error.localizedDescription)
        }
    }

    /// Shares a Lab scene's inspector values. Call before the view loads, then after each edit.
    func setSceneControls(_ controls: LabSceneControls) {
        sceneControls = controls
        guard let labScene, let metalView = view as? LabMetalView else { return }
        do {
            try labScene.applyControls()
            updateDrawableSize()
            redrawCamera(in: metalView)
        } catch {
            controls.status = error.localizedDescription
        }
    }

    func setOceanPaused(_ paused: Bool) {
        guard oceanPaused != paused else { return }
        oceanPaused = paused
        guard let oceanFixture else { return }
        oceanFixture.setPaused(paused)
        schedule.setMode(paused ? .onDemand : .continuous)
        updateActivity()
        trace("oceanPaused=\(paused)")
    }

    @objc private func orbitScene(_ gesture: LabOrbitGesture) {
        guard gesture.state == .changed, let view = gesture.view as? LabMetalView else { return }
        let point = gesture.translation(in: view)
        gesture.setTranslation(.zero, in: view)
        #if os(macOS)
        let vertical = -point.y
        #else
        let vertical = point.y
        #endif
        oceanFixture?.orbit(horizontal: Float(point.x), vertical: Float(vertical))
        studioFixture?.orbit(horizontal: Float(point.x), vertical: Float(vertical))
        pathTracingFixture?.orbit(horizontal: Float(point.x), vertical: Float(vertical))
        labScene?.orbit(horizontal: Float(point.x), vertical: Float(vertical))
        redrawCamera(in: view)
    }

    @objc private func zoomScene(_ gesture: LabZoomGesture) {
        guard gesture.state == .changed else { return }
        #if os(macOS)
        let scale = 1 + gesture.magnification
        gesture.magnification = 0
        #else
        let scale = gesture.scale
        gesture.scale = 1
        #endif
        zoomScene(by: Float(scale))
    }

    private func zoomScene(by scale: Float) {
        guard let metalView = view as? LabMetalView else { return }
        oceanFixture?.zoom(by: scale)
        studioFixture?.zoom(by: scale)
        pathTracingFixture?.zoom(by: scale)
        labScene?.zoom(by: scale)
        redrawCamera(in: metalView)
    }

    private func redrawCamera(in metalView: LabMetalView) {
        guard !schedule.shouldRunContinuously else { return }
        schedule.invalidate()
        requestDisplay(in: metalView)
    }

    @objc private func selectGraphNode(_ gesture: LabSelectionGesture) {
        guard gesture.state == .ended, isPresented, let graphFixture,
              let metalView = gesture.view as? LabMetalView, let drawable = drawableSize else { return }
        let bounds = metalView.bounds
        guard bounds.width > 0, bounds.height > 0, drawable.width > 0, drawable.height > 0 else { return }

        let point = gesture.location(in: metalView)
        let x = (point.x - bounds.minX) / bounds.width
        let localY = (point.y - bounds.minY) / bounds.height
        #if os(macOS)
        let y = metalView.isFlipped ? localY : 1 - localY
        #else
        let y = localY
        #endif
        guard (0...1).contains(x), (0...1).contains(y) else { return }

        do {
            let changed = try graphFixture.select(
                at: SIMD2(Float(x), Float(y)), aspectRatio: Float(drawable.width / drawable.height)
            )
            guard changed else { return }
            updateSelectionAccessibility(in: metalView, description: graphFixture.selectionDescription)
            trace("selection=\(graphFixture.selectionDescription)")
            schedule.invalidate()
            requestDisplay(in: metalView)
        } catch {
            frameCompleted(errorMessage: error.localizedDescription)
        }
    }

    private func updateSelectionAccessibility(in view: LabMetalView, description: String) {
        #if os(macOS)
        view.setAccessibilityValue(description)
        #else
        view.accessibilityValue = description
        #endif
    }

    private func trace(_ event: String) {
        guard traceSchedule else { return }
        let line = "FrostfieldLab \(ObjectIdentifier(self)) \(event) frames=\(submittedFrames)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    #if os(macOS)
    private func failureView(_ message: String) -> NSView {
        let background = NSView()
        let label = NSTextField(wrappingLabelWithString: "Rendering unavailable\n\n\(message)")
        label.translatesAutoresizingMaskIntoConstraints = false
        label.alignment = .center
        background.addSubview(label)

        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: background.leadingAnchor, constant: 24),
            label.trailingAnchor.constraint(equalTo: background.trailingAnchor, constant: -24),
            label.centerYAnchor.constraint(equalTo: background.centerYAnchor)
        ])
        return background
    }
    #else
    private func failureView(_ message: String) -> UIView {
        let background = UIView()
        background.backgroundColor = .systemBackground

        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.font = .preferredFont(forTextStyle: .body)
        label.adjustsFontForContentSizeCategory = true
        label.numberOfLines = 0
        label.textAlignment = .center
        label.text = "Rendering unavailable\n\n\(message)"
        background.addSubview(label)

        let margins = background.layoutMarginsGuide
        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: margins.leadingAnchor),
            label.trailingAnchor.constraint(equalTo: margins.trailingAnchor),
            label.centerYAnchor.constraint(equalTo: background.safeAreaLayoutGuide.centerYAnchor)
        ])
        return background
    }
    #endif
}
