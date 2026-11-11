import Frostfield
import Metal

/// Values that invalidate a progressive view. Scene mutations remain owned by
/// the caller, and a changed token requests a fresh history on the same tracer.
struct PathViewContent: Equatable {
    let tracer: PathTracer
    let camera: PathTracer.CameraModel
    let pose: SceneTransform
    let settings: PathTracer.Settings
    let lighting: PathTracer.Lighting?
    let preview: PathProgressivePreview.Configuration
    let isPaused: Bool
    let sceneRevision: UInt64
    let invalidationID: UInt64
    var transition: FrostfieldPathTracingView.Transition? = nil

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.matchesTrace(rhs) && lhs.isPaused == rhs.isPaused
    }

    func matchesTrace(_ other: Self) -> Bool {
        tracer === other.tracer && camera == other.camera && pose == other.pose
            && settings == other.settings && lighting == other.lighting && preview == other.preview
            && sceneRevision == other.sceneRevision && invalidationID == other.invalidationID
            && transition == other.transition
    }
}

/// Runs inside the same serialized frame slot as the raster host.
@MainActor
final class PathViewRenderer {
    private struct Image {
        let texture: any MTLTexture
        var from: (any MTLTexture)?
        var weight: Float = 1
    }

    private struct PendingDisplay {
        let command: any MTLCommandBuffer
        let image: Image?
        let finishesFade: Bool
    }

    private var content: PathViewContent?
    private var preview: PathProgressivePreview?
    private var handoff: PathPreviewHandoff?
    private var pendingDisplay: PendingDisplay?
    private var lastPresented: Image?
    private var sceneRevision: UInt64?
    private var defaultDisplay: HDRDisplay?

    var sampleCount: Int? { preview?.latestFrame?.sampleCount }

    var shouldRefine: Bool {
        _ = retireDisplay()
        if let handoff, !handoff.isComplete { return true }
        return preview?.reachedSampleLimit != true
    }

    func suspend() { handoff?.suspend() }

    /// Returns whether this command advanced the accumulation.
    func encode(
        _ input: PathViewContent, display: HDRDisplay?, displaySettings: HDRDisplay.Settings,
        destination: any MTLTexture, commandBuffer: any MTLCommandBuffer,
        time: ContinuousClock.Instant = .now
    ) throws -> Bool {
        guard retireDisplay() else { throw PathProgressivePreview.Failure.frameInFlight }
        let preview = try prepare(input, width: destination.width, height: destination.height, time: time)

        let presentation: HDRDisplay
        if let display {
            presentation = display
        } else if let defaultDisplay {
            presentation = defaultDisplay
        } else {
            presentation = try HDRDisplay(device: input.tracer.device, destinationPixelFormat: destination.pixelFormat)
            defaultDisplay = presentation
        }

        let weight = input.isPaused ? nil : handoff?.fadeWeight(after: preview.currentFrame)
        let next = input.isPaused ? nil : try preview.encode(
            cameraModel: input.camera, cameraTransform: input.pose, settings: input.settings, lighting: input.lighting,
            commandBuffer: commandBuffer
        )
        let image: Image?
        if input.isPaused {
            image = lastPresented
        } else if let low = handoff?.retainedPreview {
            if let weight, weight > 0, let final = next ?? preview.currentFrame?.texture {
                image = Image(texture: final, from: weight < 1 ? low : nil, weight: weight)
            } else {
                image = Image(texture: low)
            }
        } else {
            image = (next ?? preview.latestFrame?.texture).map { Image(texture: $0) }
        }

        do {
            if let image, let from = image.from {
                try presentation.encodeCrossfade(
                    from: from, to: image.texture, weight: image.weight, destination: destination,
                    settings: displaySettings, commandBuffer: commandBuffer
                )
            } else if let image {
                try presentation.encodeScaled(
                    source: image.texture, destination: destination, settings: displaySettings, commandBuffer: commandBuffer
                )
            } else {
                let pass = MTLRenderPassDescriptor()
                pass.colorAttachments[0].texture = destination
                pass.colorAttachments[0].loadAction = .clear
                pass.colorAttachments[0].storeAction = .store
                pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
                guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
                    throw PathTracer.Failure.encoder
                }
                encoder.endEncoding()
            }
        } catch {
            if next != nil {
                try? preview.discardUnsubmittedFrame(commandBuffer)
                handoff?.restartFade()
            }
            throw error
        }
        pendingDisplay = PendingDisplay(
            command: commandBuffer, image: image, finishesFade: weight == 1 && handoff?.isComplete == false
        )
        return next != nil
    }

    private func prepare(
        _ input: PathViewContent, width: Int, height: Int, time: ContinuousClock.Instant
    ) throws -> PathProgressivePreview {
        let revision = input.tracer.sceneRevision
        let preview: PathProgressivePreview
        if let transition = input.transition {
            var stage: PathPreviewHandoff
            var current: PathProgressivePreview
            if let handoff, let existing = self.preview, content?.matchesTrace(input) == true,
               sceneRevision == revision, existing.canvasWidth == width, existing.canvasHeight == height {
                stage = handoff
                current = existing
            } else {
                stage = try PathPreviewHandoff(transition, preview: input.preview, width: width, height: height)
                current = try PathProgressivePreview(
                    tracer: input.tracer, canvasWidth: width, canvasHeight: height, configuration: stage.lowConfiguration
                )
            }
            if input.isPaused {
                stage.suspend()
            } else {
                stage.tick(at: time)
                if stage.beginFinal(after: current.currentFrame) {
                    current = try PathProgressivePreview(
                        tracer: input.tracer, canvasWidth: width, canvasHeight: height, configuration: stage.finalConfiguration
                    )
                }
            }
            handoff = stage
            preview = current
        } else if handoff == nil, let current = self.preview,
                  content?.tracer === input.tracer, content?.preview == input.preview {
            if sceneRevision != revision || content?.invalidationID != input.invalidationID {
                current.reset()
            }
            try current.resize(width: width, height: height)
            preview = current
        } else {
            preview = try PathProgressivePreview(
                tracer: input.tracer, canvasWidth: width, canvasHeight: height, configuration: input.preview
            )
            handoff = nil
        }
        self.preview = preview
        content = input
        sceneRevision = revision
        return preview
    }

    private func retireDisplay() -> Bool {
        guard let pendingDisplay else { return true }
        switch pendingDisplay.command.status {
        case .completed:
            lastPresented = pendingDisplay.image
            if pendingDisplay.finishesFade { handoff?.finishFade() }
        case .error:
            break
        default:
            return false
        }
        self.pendingDisplay = nil
        return true
    }
}
