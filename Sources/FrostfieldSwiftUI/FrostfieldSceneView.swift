import CoreGraphics
import Foundation
import Frostfield
import Metal
import QuartzCore
import SwiftUI
import simd
#if os(macOS)
import AppKit
private typealias SceneWindow = NSWindow
private typealias ScenePlatformView = NSView
#else
import UIKit
private typealias SceneWindow = UIWindow
private typealias ScenePlatformView = UIView
#endif

/// Displays a prepared mesh or scene graph in a native Metal view.
/// The view draws on demand and lets native overlays handle input.
@MainActor
public struct FrostfieldSceneView: View {
    @Environment(\.scenePhase) private var scenePhase
    private let content: SceneContent
    private let onError: (@MainActor (any Error) -> Void)?
    private var drawableClearColor = SIMD4<Double>(0, 0, 0, 1)
    private var resolution: DrawableResolution = .display
    private var presentation: ScenePresentation?
    private var onDrawableResize: (@MainActor (CGSize) -> Void)?

    /// Use an 8-bit BGRA renderer for direct presentation or RGBA16Float with
    /// `.hdrDisplay(_:settings:)`. The callback receives failures on the main actor.
    public init(
        renderer: MeshRenderer,
        scene: MeshScene,
        onError: (@MainActor (any Error) -> Void)? = nil
    ) {
        content = .mesh(renderer, scene)
        self.onError = onError
    }

    /// Keep the command queue stable across updates. Prepare the graph's meshes
    /// on its device with the scene color format and matching depth mode. If supplied,
    /// create the shadow map on the same command queue. The view fits a cascaded
    /// shadow map to the graph's camera and the drawable's aspect ratio every frame.
    public init(
        commandQueue: any MTLCommandQueue,
        graph: SceneGraph,
        colorPixelFormat: MTLPixelFormat = .bgra8Unorm_srgb,
        depthMode: MeshDepthMode = .forward,
        shadowMap: DirectionalShadowMap? = nil,
        onError: (@MainActor (any Error) -> Void)? = nil
    ) {
        content = .graph(commandQueue, colorPixelFormat, graph, shadowMap, depthMode)
        self.onError = onError
    }

    public var body: some View {
        SceneHost(
            content: content, clearColor: drawableClearColor, resolution: resolution, presentation: presentation,
            active: scenePhase == .active, onError: onError, onDrawableResize: onDrawableResize
        )
        .id(SceneHost.Identity(
            content: content.identity, usesHDRDisplay: presentation != nil,
            drawableFormat: presentation?.display.destinationPixelFormat ?? content.colorPixelFormat
        ))
        .allowsHitTesting(false)
    }

    /// Clears each acquired drawable to this premultiplied linear RGBA color.
    /// Alpha below one lets the native view composite with its surroundings.
    /// Changing the color requests a redraw and keeps the current Metal host.
    public func clearColor(_ rgba: SIMD4<Double>) -> Self {
        var view = self
        view.drawableClearColor = rgba
        return view
    }

    /// Controls drawable resolution while SwiftUI continues to lay out the view
    /// in points. Changes apply to the current Metal host.
    public func drawableResolution(_ resolution: DrawableResolution) -> Self {
        var view = self
        view.resolution = resolution
        return view
    }

    /// Presents an RGBA16Float scene through a prepared display pass. The display
    /// must use the scene's Metal device and a BGRA8 or RGBA16Float destination.
    /// RGBA16Float enables EDR and uses the display's matching linear profile.
    /// Supply the current display headroom in the settings when it changes.
    public func hdrDisplay(_ display: HDRDisplay, settings: HDRDisplay.Settings = .init()) -> Self {
        var view = self
        view.presentation = ScenePresentation(display: display, settings: settings)
        return view
    }

    /// Receives the first completed drawable size and subsequent size changes
    /// in pixels. Rapid resizes may be coalesced before a frame completes.
    public func onDrawableSizeChange(_ action: @escaping @MainActor (CGSize) -> Void) -> Self {
        var view = self
        view.onDrawableResize = action
        return view
    }
}

/// Displays and progressively refines a prepared path-traced scene in SwiftUI.
/// Use a stable tracer and replace it when the scene topology or materials need
/// fresh preparation. Camera, lighting and completed scene revision changes
/// restart samples. Native overlays can supply interaction.
@MainActor
public struct FrostfieldPathTracingView: View {
    @Environment(\.scenePhase) private var scenePhase
    private var input: PathViewContent
    private let onError: (@MainActor (any Error) -> Void)?
    private var resolution: FrostfieldSceneView.DrawableResolution = .display
    private var presentation: ScenePresentation?
    private var onProgress: (@MainActor (Int) -> Void)?

    /// The view reads prepared resources on the tracer's queue. Publish a new
    /// invalidation ID after an in-place texture or shader-uniform edit. Pausing
    /// keeps the latest image and resumes from its samples if the view is unchanged.
    public init(
        tracer: PathTracer, camera: PathTracer.CameraModel, cameraTransform: SceneTransform = .init(),
        settings: PathTracer.Settings = .init(), lighting: PathTracer.Lighting? = nil,
        preview: PathProgressivePreview.Configuration = .init(), isPaused: Bool = false,
        invalidationID: UInt64 = 0, onError: (@MainActor (any Error) -> Void)? = nil
    ) {
        input = PathViewContent(
            tracer: tracer, camera: camera, pose: cameraTransform, settings: settings, lighting: lighting,
            preview: preview, isPaused: isPaused, sceneRevision: tracer.sceneRevision, invalidationID: invalidationID
        )
        self.onError = onError
    }

    public var body: some View {
        let content = SceneContent.path(input)
        SceneHost(
            content: content, clearColor: SIMD4(0, 0, 0, 1), resolution: resolution, presentation: presentation,
            active: scenePhase == .active, onError: onError, onDrawableResize: nil, onPathProgress: onProgress
        )
        .id(SceneHost.Identity(
            content: content.identity, usesHDRDisplay: presentation != nil,
            drawableFormat: presentation?.display.destinationPixelFormat ?? content.defaultDrawableFormat
        ))
        .allowsHitTesting(false)
    }

    /// Controls the drawable's size in pixels. The progressive trace can use
    /// fewer pixels independently, according to its preview configuration.
    public func drawableResolution(_ resolution: FrostfieldSceneView.DrawableResolution) -> Self {
        var view = self
        view.resolution = resolution
        return view
    }

    /// Changes tone mapping and the drawable format. The default is neutral
    /// tone mapping into an sRGB BGRA8 drawable.
    public func hdrDisplay(_ display: HDRDisplay, settings: HDRDisplay.Settings = .init()) -> Self {
        var view = self
        view.presentation = ScenePresentation(display: display, settings: settings)
        return view
    }

    /// Uses a low-resolution preview before revealing a fixed-resolution trace.
    public func previewTransition(_ transition: Transition?) -> Self {
        var view = self
        view.input.transition = transition
        return view
    }

    /// Receives the completed sample count after each successful current-view
    /// trace batch. A reset or a resolution change starts the count again.
    public func onProgress(_ action: @escaping @MainActor (Int) -> Void) -> Self {
        var view = self
        view.onProgress = action
        return view
    }
}

private struct ScenePresentation: Equatable {
    let display: HDRDisplay
    let settings: HDRDisplay.Settings

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.display === rhs.display && lhs.settings == rhs.settings
    }
}

private enum SceneContent {
    case mesh(MeshRenderer, MeshScene)
    case graph(any MTLCommandQueue, MTLPixelFormat, SceneGraph, DirectionalShadowMap?, MeshDepthMode)
    case path(PathViewContent)

    enum Identity: Hashable {
        case mesh(ObjectIdentifier)
        case graph(ObjectIdentifier, MTLPixelFormat)
        case path(ObjectIdentifier)
    }

    var identity: Identity {
        switch self {
        case .mesh(let renderer, _): .mesh(ObjectIdentifier(renderer))
        case .graph(let queue, let format, _, _, _): .graph(ObjectIdentifier(queue), format)
        case .path(let input): .path(ObjectIdentifier(input.tracer.queue))
        }
    }

    var device: any MTLDevice {
        switch self {
        case .mesh(let renderer, _): renderer.device
        case .graph(let queue, _, _, _, _): queue.device
        case .path(let input): input.tracer.device
        }
    }

    var colorPixelFormat: MTLPixelFormat {
        switch self {
        case .mesh(let renderer, _): renderer.colorPixelFormat
        case .graph(_, let format, _, _, _): format
        case .path: .rgba16Float
        }
    }

    var pathInput: PathViewContent? {
        if case .path(let input) = self { return input }
        return nil
    }

    var defaultDrawableFormat: MTLPixelFormat {
        pathInput == nil ? colorPixelFormat : .bgra8Unorm_srgb
    }

    var depthClearValue: Double {
        let mode: MeshDepthMode = switch self {
        case .mesh(let renderer, _): renderer.depthMode
        case .graph(_, _, _, _, let depthMode): depthMode
        case .path: .forward
        }
        return mode == .reversed ? 0 : 1
    }

    func makeQueue() -> (any MTLCommandQueue)? {
        switch self {
        case .mesh(let renderer, _): renderer.device.makeCommandQueue()
        case .graph(let queue, _, _, _, _): queue
        case .path(let input): input.tracer.queue
        }
    }

    func matches(_ other: Self) -> Bool {
        switch (self, other) {
        case (.mesh(let renderer, let scene), .mesh(let otherRenderer, let otherScene)):
            renderer === otherRenderer && scene.matches(otherScene)
        case (.graph(let queue, let format, let graph, let shadow, let depthMode),
              .graph(let otherQueue, let otherFormat, let otherGraph, let otherShadow, let otherDepthMode)):
            queue === otherQueue && format == otherFormat && depthMode == otherDepthMode
                && shadow === otherShadow && graph.matches(otherGraph)
        case (.path(let input), .path(let other)):
            input == other
        default:
            false
        }
    }

    func encode(pass: MTLRenderPassDescriptor, commandBuffer: any MTLCommandBuffer) throws {
        switch self {
        case .mesh(let renderer, let scene):
            try renderer.encode(scene, pass: pass, commandBuffer: commandBuffer)
        case .graph(_, _, let graph, let shadowMap, let depthMode):
            if let shadowMap, shadowMap.cascades != nil, let color = pass.colorAttachments[0].texture, color.height > 0 {
                let aspect = Float(color.width) / Float(color.height)
                if graph.orthographicCamera != nil {
                    try shadowMap.fitCascades(view: graph.shaderGraphCamera(aspectRatio: aspect, depthMode: depthMode))
                } else {
                    try shadowMap.fitCascades(
                        camera: graph.camera, cameraTransform: graph.cameraTransform, aspectRatio: aspect
                    )
                }
            }
            try graph.encode(pass: pass, commandBuffer: commandBuffer, shadowMap: shadowMap, depthMode: depthMode)
        case .path:
            throw HostError.pathFrame
        }
    }
}

@MainActor
private struct SceneHost {
    struct Identity: Hashable {
        let content: SceneContent.Identity
        let usesHDRDisplay: Bool
        let drawableFormat: MTLPixelFormat
    }

    let content: SceneContent
    let clearColor: SIMD4<Double>
    let resolution: FrostfieldSceneView.DrawableResolution
    let presentation: ScenePresentation?
    let active: Bool
    let onError: (@MainActor (any Error) -> Void)?
    let onDrawableResize: (@MainActor (CGSize) -> Void)?
    var onPathProgress: (@MainActor (Int) -> Void)? = nil

    func makeCoordinator() -> SceneCoordinator {
        SceneCoordinator(content: content, clearColor: clearColor, resolution: resolution, presentation: presentation)
    }

    func update(_ coordinator: SceneCoordinator) {
        coordinator.update(
            content: content, clearColor: clearColor, resolution: resolution, presentation: presentation,
            active: active, onError: onError, onDrawableResize: onDrawableResize, onPathProgress: onPathProgress
        )
    }
}

#if os(macOS)
extension SceneHost: NSViewRepresentable {
    func makeNSView(context: Context) -> SceneMetalView {
        context.coordinator.makeView()
    }

    func updateNSView(_ view: SceneMetalView, context: Context) {
        update(context.coordinator)
    }

    static func dismantleNSView(_ view: SceneMetalView, coordinator: SceneCoordinator) {
        coordinator.dismantle(view)
    }
}
#else
extension SceneHost: UIViewRepresentable {
    func makeUIView(context: Context) -> SceneMetalView {
        context.coordinator.makeView()
    }

    func updateUIView(_ view: SceneMetalView, context: Context) {
        update(context.coordinator)
    }

    static func dismantleUIView(_ view: SceneMetalView, coordinator: SceneCoordinator) {
        coordinator.dismantle(view)
    }
}
#endif

@MainActor
private final class SceneMetalView: ScenePlatformView {
    var activityChanged: (() -> Void)?

    var metalLayer: CAMetalLayer? { layer as? CAMetalLayer }

    var isVisibleForRendering: Bool {
        guard let window else { return false }
        #if os(macOS)
        return window.isVisible && !window.isMiniaturized
            && window.occlusionState.contains(.visible) && !isHiddenOrHasHiddenAncestor
        #else
        return !window.isHidden && !isHidden
        #endif
    }

    func synchronizeDrawableSize(
        resolution: FrostfieldSceneView.DrawableResolution
    ) -> Result<CGSize?, FrostfieldSceneView.DrawableResolution.Failure> {
        guard let metalLayer, window != nil else { return .success(nil) }
        #if os(macOS)
        let scale = convertToBacking(CGSize(width: 1, height: 1)).width
        let backing = convertToBacking(bounds.size)
        #else
        let scale = traitCollection.displayScale
        let backing = CGSize(width: bounds.width * scale, height: bounds.height * scale)
        #endif
        return resolution.apply(to: metalLayer, logicalSize: bounds.size, backingSize: backing, displayScale: scale)
    }

    #if os(macOS)
    override func makeBackingLayer() -> CALayer { CAMetalLayer() }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        activityChanged?()
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        activityChanged?()
    }

    override func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
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

    override func layoutSubviews() {
        super.layoutSubviews()
        activityChanged?()
    }

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

@MainActor
private final class SceneCoordinator: NSObject, @preconcurrency CAMetalDisplayLinkDelegate {
    private let queue: (any MTLCommandQueue)?
    private let drawableFormat: MTLPixelFormat
    private let hasDrawableFormat: Bool
    private var content: SceneContent
    private var clearColor: SIMD4<Double>
    private var resolution: FrostfieldSceneView.DrawableResolution
    private var presentation: ScenePresentation?
    private var sceneIsActive = false
    private var onError: (@MainActor (any Error) -> Void)?
    private var onDrawableResize: (@MainActor (CGSize) -> Void)?
    private var onPathProgress: (@MainActor (Int) -> Void)?
    private var pathRenderer: PathViewRenderer?
    private var pathGeneration: UInt64 = 0
    private var schedule = RenderSchedule()
    private var displayLink: CAMetalDisplayLink?
    private var drawableSizeResult: Result<CGSize?, FrostfieldSceneView.DrawableResolution.Failure> = .success(nil)
    private var drawableSize: CGSize? { try? drawableSizeResult.get() }
    private var reportedDrawableSize: CGSize?
    private var depthTexture: (any MTLTexture)?
    private var hdrTexture: (any MTLTexture)?
    private weak var view: SceneMetalView?
    private weak var observedWindow: SceneWindow?
    #if !os(macOS)
    private weak var observedScene: UIWindowScene?
    private var isDeactivating = false
    #endif

    init(
        content: SceneContent, clearColor: SIMD4<Double>, resolution: FrostfieldSceneView.DrawableResolution,
        presentation: ScenePresentation?
    ) {
        self.content = content
        self.clearColor = clearColor
        self.resolution = resolution
        self.presentation = presentation
        queue = content.makeQueue()
        let format = presentation?.display.destinationPixelFormat ?? content.defaultDrawableFormat
        drawableFormat = format
        hasDrawableFormat = format == .bgra8Unorm || format == .bgra8Unorm_srgb
            || (presentation != nil && format == .rgba16Float)
    }

    func makeView() -> SceneMetalView {
        let view = SceneMetalView(frame: .zero)
        #if os(macOS)
        view.wantsLayer = true
        #endif
        if let layer = view.metalLayer {
            layer.device = content.device
            if hasDrawableFormat { layer.pixelFormat = drawableFormat }
            layer.framebufferOnly = true
            configurePresentation(of: layer)
        }
        configureOpacity(of: view)
        view.activityChanged = { [weak self] in self?.viewActivityChanged() }
        #if !os(macOS)
        view.registerForTraitChanges([UITraitDisplayScale.self]) { (view: SceneMetalView, _) in
            view.activityChanged?()
        }
        #endif
        self.view = view
        return view
    }

    func update(
        content: SceneContent, clearColor: SIMD4<Double>, resolution: FrostfieldSceneView.DrawableResolution,
        presentation: ScenePresentation?, active: Bool, onError: (@MainActor (any Error) -> Void)?,
        onDrawableResize: (@MainActor (CGSize) -> Void)?, onPathProgress: (@MainActor (Int) -> Void)?
    ) {
        self.onError = onError
        self.onPathProgress = onPathProgress
        if (self.onDrawableResize == nil) != (onDrawableResize == nil) {
            reportedDrawableSize = nil
            schedule.invalidate()
        }
        self.onDrawableResize = onDrawableResize
        self.resolution = resolution
        if self.presentation != presentation {
            let previousDisplay = self.presentation?.display
            self.presentation = presentation
            if previousDisplay !== presentation?.display, let layer = view?.metalLayer {
                configurePresentation(of: layer)
            }
            schedule.invalidate()
        }
        if !self.content.matches(content) {
            self.content = content
            pathGeneration &+= 1
            schedule.invalidate()
        }
        if self.clearColor != clearColor {
            self.clearColor = clearColor
            if let view { configureOpacity(of: view) }
            schedule.invalidate()
        }
        if content.pathInput?.isPaused == true { pathRenderer?.suspend() }
        sceneIsActive = active
        viewActivityChanged()
    }

    private func configurePresentation(of layer: CAMetalLayer) {
        layer.colorspace = presentation?.display.presentationColorSpace
            ?? (content.pathInput == nil ? nil : CGColorSpace(name: CGColorSpace.sRGB))
        layer.wantsExtendedDynamicRangeContent = presentation != nil && drawableFormat == .rgba16Float
    }

    private func configureOpacity(of view: SceneMetalView) {
        let opaque = clearColor.w == 1
        view.metalLayer?.isOpaque = opaque
        #if !os(macOS)
        view.isOpaque = opaque
        #endif
    }

    func dismantle(_ view: SceneMetalView) {
        NotificationCenter.default.removeObserver(self)
        schedule.setActive(false)
        stopDisplayLink()
        drawableSizeResult = .success(nil)
        reportedDrawableSize = nil
        depthTexture = nil
        hdrTexture = nil
        view.activityChanged = nil
        self.view = nil
        observedWindow = nil
        #if !os(macOS)
        observedScene = nil
        isDeactivating = false
        #endif
        onError = nil
        onDrawableResize = nil
        onPathProgress = nil
        pathRenderer = nil
        pathGeneration &+= 1
    }

    private func stopDisplayLink() {
        displayLink?.invalidate()
        displayLink = nil
    }

    private func viewActivityChanged() {
        let window = view?.window
        #if os(macOS)
        let sourceChanged = observedWindow !== window
        #else
        let scene = window?.windowScene
        let sourceChanged = observedWindow !== window || observedScene !== scene
        #endif
        if sourceChanged {
            stopDisplayLink()
            NotificationCenter.default.removeObserver(self)
            observedWindow = window
            #if !os(macOS)
            observedScene = scene
            isDeactivating = false
            if let scene {
                for notification in [UIScene.didActivateNotification, UIScene.willDeactivateNotification] {
                    NotificationCenter.default.addObserver(
                        self, selector: #selector(windowDidChange(_:)), name: notification, object: scene
                    )
                }
            }
            #endif
            if let window {
                #if os(macOS)
                let notifications = [
                    NSWindow.didMiniaturizeNotification, NSWindow.didDeminiaturizeNotification,
                    NSWindow.didChangeOcclusionStateNotification, NSWindow.didChangeScreenNotification,
                ]
                #else
                let notifications = [UIWindow.didBecomeVisibleNotification, UIWindow.didBecomeHiddenNotification]
                #endif
                for notification in notifications {
                    NotificationCenter.default.addObserver(
                        self, selector: #selector(windowDidChange(_:)), name: notification, object: window
                    )
                }
            }
        }
        updateActivity()
    }

    @objc private func windowDidChange(_ notification: Notification) {
        #if !os(macOS)
        if notification.name == UIScene.willDeactivateNotification { isDeactivating = true }
        if notification.name == UIScene.didActivateNotification { isDeactivating = false }
        #endif
        updateActivity()
    }

    private func updateActivity() {
        guard let view, view.window != nil else {
            schedule.setActive(false)
            pathRenderer?.suspend()
            stopDisplayLink()
            drawableSizeResult = .success(nil)
            reportedDrawableSize = nil
            depthTexture = nil
            hdrTexture = nil
            return
        }

        #if os(macOS)
        let active = sceneIsActive
        #else
        // A view hosted by UIHostingController uses its UIKit window scene lifecycle.
        let active = !isDeactivating && view.window?.windowScene?.activationState == .foregroundActive
        #endif
        let isActive = active && view.isVisibleForRendering
        schedule.setActive(isActive)
        if !isActive { pathRenderer?.suspend() }
        let result = view.synchronizeDrawableSize(resolution: resolution)
        if result != drawableSizeResult {
            drawableSizeResult = result
            schedule.invalidate()
        }
        requestDisplay()
    }

    private func requestDisplay() {
        guard schedule.shouldRequestDisplay, let view,
              view.bounds.width > 0, view.bounds.height > 0 else {
            displayLink?.isPaused = true
            return
        }
        guard hasDrawableFormat else { return rejectFrame(HostError.drawableFormat) }
        if let presentation {
            guard content.colorPixelFormat == .rgba16Float else { return rejectFrame(HostError.hdrSceneFormat) }
            guard presentation.display.device === content.device else { return rejectFrame(HostError.hdrDevice) }
        }
        guard queue != nil else { return rejectFrame(HostError.queue) }
        guard let layer = view.metalLayer else { return rejectFrame(HostError.layer) }
        if case .failure(let error) = drawableSizeResult { return rejectFrame(error) }
        guard drawableSize != nil else {
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
        // Core Animation calls the delegate on this run loop, so the callback
        // can synchronously use the main actor without moving the drawable.
        link.add(to: .main, forMode: .common)
        link.isPaused = false
    }

    func metalDisplayLink(_ link: CAMetalDisplayLink, needsUpdate update: CAMetalDisplayLink.Update) {
        guard link === displayLink, schedule.canDraw, let queue, let drawableSize else { return }
        let drawable = update.drawable
        let color = drawable.texture
        guard CGFloat(color.width) == drawableSize.width, CGFloat(color.height) == drawableSize.height,
              color.pixelFormat == drawableFormat, schedule.beginFrame() else { return }

        link.isPaused = true
        do {
            guard let buffer = queue.makeCommandBuffer() else { throw HostError.commandBuffer }
            let pathSubmission: (generation: UInt64, refined: Bool)?
            if let input = content.pathInput {
                let renderer = pathRenderer ?? PathViewRenderer()
                pathRenderer = renderer
                let refined = try renderer.encode(
                    input, display: presentation?.display, displaySettings: presentation?.settings ?? .init(),
                    destination: color, commandBuffer: buffer
                )
                pathSubmission = (pathGeneration, refined)
            } else if let presentation {
                pathSubmission = nil
                let source = try hdrSource(matching: color)
                try content.encode(pass: makePass(color: source), commandBuffer: buffer)
                try presentation.display.encode(
                    source: source, destination: color, settings: presentation.settings, commandBuffer: buffer
                )
            } else {
                pathSubmission = nil
                try content.encode(pass: makePass(color: color), commandBuffer: buffer)
            }
            buffer.addCompletedHandler { [weak self] buffer in
                let error = buffer.status == .completed
                    ? nil : HostError.submission(buffer.error?.localizedDescription)
                Task { @MainActor in
                    self?.finishFrame(error, completedSize: drawableSize, pathSubmission: pathSubmission)
                }
            }
            buffer.present(drawable)
            buffer.commit()
        } catch {
            finishFrame(error)
        }
    }

    private func hdrSource(matching drawable: any MTLTexture) throws -> any MTLTexture {
        if let hdrTexture, hdrTexture.width == drawable.width, hdrTexture.height == drawable.height { return hdrTexture }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: drawable.width, height: drawable.height, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = [.renderTarget, .shaderRead]
        guard let source = content.device.makeTexture(descriptor: descriptor) else { throw HostError.hdrTexture }
        source.label = "Frostfield SwiftUI HDR scene"
        hdrTexture = source
        return source
    }

    private func makePass(color: any MTLTexture) throws -> MTLRenderPassDescriptor {
        if depthTexture?.width != color.width || depthTexture?.height != color.height {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: MeshRenderer.depthPixelFormat, width: color.width, height: color.height, mipmapped: false
            )
            descriptor.storageMode = .private
            descriptor.usage = .renderTarget
            guard let depth = content.device.makeTexture(descriptor: descriptor) else {
                throw HostError.depthTexture
            }
            depth.label = "Frostfield SwiftUI depth"
            depthTexture = depth
        }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depthTexture
        pass.depthAttachment.storeAction = .dontCare
        try FramebufferClear(color: clearColor, depth: content.depthClearValue).apply(to: pass)
        return pass
    }

    private func rejectFrame(_ error: any Error) {
        if schedule.beginFrame() { finishFrame(error) }
    }

    private func finishFrame(
        _ error: (any Error)?, completedSize: CGSize? = nil,
        pathSubmission: (generation: UInt64, refined: Bool)? = nil
    ) {
        schedule.finishFrame()
        if let error {
            if pathSubmission == nil || pathSubmission?.generation == pathGeneration {
                Task { @MainActor [weak self] in self?.onError?(error) }
            }
        } else if let completedSize, completedSize == drawableSize, completedSize != reportedDrawableSize,
                  let onDrawableResize {
            reportedDrawableSize = completedSize
            onDrawableResize(completedSize)
        }
        if error == nil, let pathSubmission, pathSubmission.generation == pathGeneration,
           completedSize == drawableSize, schedule.isActive,
           let input = content.pathInput, !input.isPaused, let pathRenderer {
            if pathSubmission.refined, let samples = pathRenderer.sampleCount { onPathProgress?(samples) }
            if pathRenderer.shouldRefine { schedule.invalidate() }
        }
        requestDisplay()
    }
}

private enum HostError: LocalizedError {
    case drawableFormat
    case hdrSceneFormat
    case hdrDevice
    case hdrTexture
    case pathFrame
    case layer
    case queue
    case commandBuffer
    case depthTexture
    case submission(String?)

    var errorDescription: String? {
        switch self {
        case .drawableFormat: "Use a BGRA8 drawable, or an RGBA16Float drawable with HDR display."
        case .hdrSceneFormat: "Prepare the scene with an RGBA16Float color target for HDR display."
        case .hdrDevice: "Prepare HDR display on the same Metal device as the scene."
        case .hdrTexture: "Metal could not allocate the SwiftUI HDR scene texture."
        case .pathFrame: "A path-traced frame requires the progressive host."
        case .layer: "The SwiftUI host could not create a Metal layer."
        case .queue: "Metal could not create a command queue."
        case .commandBuffer: "Metal could not create a command buffer."
        case .depthTexture: "Metal could not allocate the SwiftUI depth texture."
        case .submission(let detail): detail ?? "Metal could not complete the frame."
        }
    }
}

private extension MeshScene {
    func matches(_ other: Self) -> Bool {
        transform.matches(other.transform) && cameraTransform.matches(other.cameraTransform)
            && camera.matches(other.camera)
            && materialParameters == other.materialParameters && jointPalette == other.jointPalette
            && morphWeights == other.morphWeights
    }
}

private extension SceneGraph {
    func matches(_ other: Self) -> Bool {
        guard camera.matches(other.camera), cameraTransform.matches(other.cameraTransform),
              cameraLayers == other.cameraLayers, environmentIntensity == other.environmentIntensity,
              environmentRotation == other.environmentRotation, environmentEulerRotation == other.environmentEulerRotation,
              projectionJitter == other.projectionJitter,
              highPrecision == other.highPrecision, roots.count == other.roots.count else { return false }
        var pending = Array(zip(roots, other.roots))
        while let (node, otherNode) = pending.popLast() {
            guard node.id == otherNode.id, node.mesh === otherNode.mesh,
                  node.transform.matches(otherNode.transform), node.materialParameters == otherNode.materialParameters,
                  node.castsShadow == otherNode.castsShadow, node.jointPalette == otherNode.jointPalette,
                  node.light == otherNode.light,
                  // Instance sets are compared by identity; their edits do not request a redraw.
                  node.instances === otherNode.instances, node.morphWeights == otherNode.morphWeights,
                  node.frustumCulled == otherNode.frustumCulled, node.cullingBounds == otherNode.cullingBounds,
                  node.levelsOfDetail.matches(otherNode.levelsOfDetail),
                  node.isVisible == otherNode.isVisible, node.layers == otherNode.layers,
                  node.renderOrder == otherNode.renderOrder, node.primitive.matches(otherNode.primitive),
                  node.children.count == otherNode.children.count else {
                return false
            }
            pending.append(contentsOf: zip(node.children, otherNode.children))
        }
        return true
    }
}

private extension Optional where Wrapped == ScenePrimitive {
    func matches(_ other: Self) -> Bool {
        switch (self, other) {
        case (nil, nil): true
        case (.points(let a)?, .points(let b)?): a === b
        case (.lines(let a)?, .lines(let b)?): a === b
        case (.sprites(let a)?, .sprites(let b)?): a === b
        default: false
        }
    }
}

private extension Optional where Wrapped == LevelOfDetail {
    func matches(_ other: Self) -> Bool {
        switch (self, other) {
        case (nil, nil):
            return true
        case (let lod?, let otherLOD?):
            return lod.currentLevel == otherLOD.currentLevel && lod.levels.count == otherLOD.levels.count
                && zip(lod.levels, otherLOD.levels).allSatisfy { level, other in
                    level.mesh === other.mesh && level.distance == other.distance && level.hysteresis == other.hysteresis
                }
        default:
            return false
        }
    }
}

private extension PerspectiveCamera {
    func matches(_ other: Self) -> Bool {
        verticalFieldOfView == other.verticalFieldOfView && near == other.near && far == other.far
    }
}

private extension SceneTransform {
    func matches(_ other: Self) -> Bool {
        translation == other.translation && rotation.vector == other.rotation.vector && scale == other.scale
    }
}
