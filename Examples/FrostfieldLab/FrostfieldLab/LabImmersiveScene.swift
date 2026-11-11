#if os(visionOS)
import ARKit
import CompositorServices
import Frostfield
import Metal
import OSLog
import SwiftUI
import simd

/// Coordinates the launcher window and immersive scene on the main actor.
@MainActor
final class LabImmersivePresentation {
    static let windowID = "frostfield-lab-window"

    private enum State { case idle, opening, visible, endedWhileOpening }
    private var state = State.idle
    private var attemptedAutostart = false

    func claimAutostart() -> Bool {
        guard !attemptedAutostart else { return false }
        attemptedAutostart = true
        return true
    }

    func beginOpening() -> Bool {
        guard state == .idle else { return false }
        state = .opening
        return true
    }

    func didOpen() -> Bool {
        guard state == .opening else {
            state = .idle
            return false
        }
        state = .visible
        return true
    }

    func cancelOpening() {
        state = .idle
    }

    func didEnd() -> Bool {
        switch state {
        case .visible:
            state = .idle
            return true
        case .opening:
            state = .endedWhileOpening
            return false
        case .idle, .endedWhileOpening:
            return false
        }
    }
}

struct LabImmersiveLauncher: View {
    static var isRequested: Bool {
        ProcessInfo.processInfo.arguments.contains("--immersive-scene")
    }

    let presentation: LabImmersivePresentation
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.dismissWindow) private var dismissWindow
    @State private var isOpening = false
    @State private var failedToOpen = false

    var body: some View {
        VStack(spacing: 16) {
            Text("Immersive Metal Preview").font(.title)
            Button("Open Immersive Preview") {
                Task { await enterPreview() }
            }
            .disabled(isOpening)
            if failedToOpen {
                Text("The immersive preview could not open.")
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .task {
            #if DEBUG && targetEnvironment(simulator)
            // Explicit development launch option for the visionOS simulator smoke test.
            guard ProcessInfo.processInfo.arguments.contains("--immersive-autostart"),
                  presentation.claimAutostart() else { return }
            await enterPreview()
            #endif
        }
    }

    private func enterPreview() async {
        guard !isOpening, presentation.beginOpening() else { return }
        isOpening = true
        failedToOpen = false
        defer { isOpening = false }
        switch await openImmersiveSpace(id: LabImmersiveScene.spaceID) {
        case .opened:
            if presentation.didOpen() {
                dismissWindow(id: LabImmersivePresentation.windowID)
            } else {
                failedToOpen = true
            }
        case .userCancelled:
            presentation.cancelOpening()
        case .error:
            presentation.cancelOpening()
            failedToOpen = true
        @unknown default:
            presentation.cancelOpening()
            failedToOpen = true
        }
    }
}

struct LabImmersiveScene: Scene {
    static let spaceID = "frostfield-immersive-preview"
    let presentation: LabImmersivePresentation
    @Environment(\.openWindow) private var openWindow

    var body: some Scene {
        ImmersiveSpace(id: Self.spaceID) {
            CompositorLayer(configuration: LabImmersiveConfiguration()) { layer in
                LabImmersiveRenderer.launch(layer) {
                    if presentation.didEnd() {
                        openWindow(id: LabImmersivePresentation.windowID)
                    }
                }
            }
        }
    }
}

private struct LabImmersiveConfiguration: CompositorLayerConfiguration {
    func makeConfiguration(
        capabilities: LayerRenderer.Capabilities,
        configuration: inout LayerRenderer.Configuration
    ) {
        configuration.isFoveationEnabled = false
        if capabilities.supportedLayouts(options: []).contains(.dedicated) {
            configuration.layout = .dedicated
        }
        if capabilities.supportedDepthFormats.contains(MeshRenderer.depthPixelFormat) {
            configuration.depthFormat = MeshRenderer.depthPixelFormat
        }
    }
}

private nonisolated final class LabImmersiveExecutor: TaskExecutor {
    private let queue = DispatchQueue(label: "FrostfieldLab.immersive-renderer", qos: .userInteractive)

    func enqueue(_ job: UnownedJob) {
        queue.async {
            job.runSynchronously(on: self.asUnownedTaskExecutor())
        }
    }
}

// CompositorLayer supplies this reference on the main actor but requires the
// render loop on a dedicated thread. The callback hands it to that owner once.
private nonisolated struct LabCompositorHandoff: @unchecked Sendable {
    let layer: LayerRenderer
}

private nonisolated struct LabImmersiveRenderer {
    private static let log = Logger(subsystem: "org.frostfield.FrostfieldLab", category: "ImmersivePreview")

    private let layer: LayerRenderer
    private let queue: any MTLCommandQueue
    private let graph: SceneGraph
    private let tracking: WorldTrackingProvider?
    private let clock = LayerRenderer.Clock()

    static func launch(_ layer: LayerRenderer, onEnd: @escaping @MainActor @Sendable () -> Void) {
        let handoff = LabCompositorHandoff(layer: layer)
        Task.detached(executorPreference: LabImmersiveExecutor(), priority: .high) {
            do {
                let tracking = await startTracking()
                defer { tracking?.session.stop() }
                try LabImmersiveRenderer(layer: handoff.layer, tracking: tracking?.provider).render()
            } catch {
                log.error("Immersive preview failed: \(error.localizedDescription, privacy: .public)")
            }
            await onEnd()
        }
    }

    private static func startTracking() async -> (session: ARKitSession, provider: WorldTrackingProvider)? {
        guard WorldTrackingProvider.isSupported else { return nil }
        let session = ARKitSession()
        let provider = WorldTrackingProvider()
        do {
            try await session.run([provider])
            return (session, provider)
        } catch {
            log.notice("Using a fixed immersive preview pose: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

    private init(layer: LayerRenderer, tracking: WorldTrackingProvider?) throws {
        guard layer.configuration.layout == .dedicated,
              layer.configuration.depthFormat == MeshRenderer.depthPixelFormat else {
            throw PreviewFailure.unsupportedConfiguration
        }
        guard let queue = layer.device.makeCommandQueue() else { throw PreviewFailure.commandQueue }

        let renderer = try MeshRenderer(
            device: layer.device, colorPixelFormat: layer.configuration.colorFormat,
            geometry: .cube(), depthMode: .reversed
        )
        let yaw = simd_quatf(angle: 0.55, axis: SIMD3(0, 1, 0))
        let pitch = simd_quatf(angle: -0.32, axis: SIMD3(1, 0, 0))
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 30)
        graph = SceneGraph(roots: [
            SceneNode(
                transform: SceneTransform(
                    translation: SIMD3(0, 0, -2.4), rotation: yaw * pitch,
                    scale: SIMD3(repeating: 0.75)
                ),
                mesh: renderer,
                materialParameters: SIMD4(0.65, 0.82, 1, 1)
            ),
        ], camera: camera)
        self.layer = layer
        self.queue = queue
        self.tracking = tracking
    }

    private func render() throws {
        var previous: (any MTLCommandBuffer)?
        var submitted = false
        defer { previous?.waitUntilCompleted() }

        while true {
            switch layer.state {
            case .paused:
                layer.waitUntilRunning()
            case .running:
                // One submitted buffer at a time, on the dedicated render owner.
                if let buffer = previous {
                    buffer.waitUntilCompleted()
                    if buffer.status == .error {
                        throw PreviewFailure.gpu(buffer.error?.localizedDescription)
                    }
                    previous = nil
                }
                if let frame = try autoreleasepool(invoking: submitFrame) {
                    previous = frame.buffer
                    if !submitted {
                        Self.log.notice("Immersive preview submitted \(frame.viewCount, privacy: .public) view(s).")
                        submitted = true
                    }
                } else {
                    Thread.sleep(forTimeInterval: 0.002)
                }
            case .invalidated:
                return
            @unknown default:
                return
            }
        }
    }

    private func submitFrame() throws -> (buffer: any MTLCommandBuffer, viewCount: Int)? {
        guard let frame = layer.queryNextFrame(), let timing = frame.predictTiming() else { return nil }

        frame.startUpdate()
        frame.endUpdate()
        clock.wait(until: timing.optimalInputTime)
        frame.startSubmission()
        defer { frame.endSubmission() }

        let drawables = frame.queryDrawables()
        guard !drawables.isEmpty else { return nil }
        guard let buffer = queue.makeCommandBuffer() else { throw PreviewFailure.commandBuffer }
        var viewCount = 0

        for drawable in drawables {
            let anchor = deviceAnchor(for: drawable)
            drawable.deviceAnchor = anchor
            let worldFromDevice = anchor?.originFromAnchorTransform ?? matrix_identity_float4x4
            let colors = drawable.colorTextures
            let depths = drawable.depthTextures

            for (index, view) in drawable.views.enumerated() {
                let map = view.textureMap
                let textureIndex = map.textureIndex
                guard colors.indices.contains(textureIndex), depths.indices.contains(textureIndex) else {
                    throw PreviewFailure.missingViewTarget
                }

                let eye = worldFromDevice * view.transform
                let projection = drawable.computeProjection(viewIndex: index)
                let position = eye.columns.3
                let pass = MTLRenderPassDescriptor()
                pass.colorAttachments[0].texture = colors[textureIndex]
                pass.colorAttachments[0].loadAction = .clear
                pass.colorAttachments[0].storeAction = .store
                pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.012, green: 0.02, blue: 0.035, alpha: 1)
                pass.depthAttachment.texture = depths[textureIndex]
                pass.depthAttachment.loadAction = .clear
                pass.depthAttachment.storeAction = .store
                pass.depthAttachment.clearDepth = 0

                try graph.encode(
                    pass: pass, commandBuffer: buffer,
                    clipFromWorld: projection * simd_inverse(eye),
                    cameraPosition: SIMD3(position.x, position.y, position.z),
                    viewport: map.viewport, depthMode: .reversed
                )
                viewCount += 1
            }
            drawable.encodePresent(commandBuffer: buffer)
        }
        guard viewCount > 0 else { return nil }
        buffer.commit()
        return (buffer, viewCount)
    }

    private func deviceAnchor(for drawable: LayerRenderer.Drawable) -> DeviceAnchor? {
        let duration = LayerRenderer.Clock.Instant.epoch.duration(to: drawable.frameTiming.presentationTime)
        let components = duration.components
        let timestamp = Double(components.seconds) + Double(components.attoseconds) / 1e18
        return tracking?.queryDeviceAnchor(atTimestamp: timestamp)
    }

    private enum PreviewFailure: LocalizedError {
        case unsupportedConfiguration
        case commandQueue
        case commandBuffer
        case missingViewTarget
        case gpu(String?)

        var errorDescription: String? {
            switch self {
            case .unsupportedConfiguration: "The immersive preview requires dedicated color and depth targets."
            case .commandQueue: "Metal could not create the immersive command queue."
            case .commandBuffer: "Metal could not create an immersive command buffer."
            case .missingViewTarget: "The compositor did not provide a target for an immersive view."
            case .gpu(let detail): detail ?? "Metal could not complete the immersive frame."
            }
        }
    }
}
#endif
