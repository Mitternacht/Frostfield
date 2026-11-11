import Foundation
import Frostfield
import Metal

public extension FrostfieldPathTracingView {
    /// Keeps a low-resolution path preview visible until a fixed-size trace
    /// reaches a sample gate, then fades between their displayed images.
    struct Transition: Equatable, Sendable {
        public var renderDelay: Duration
        public var minimumSamples: Int
        public var fadeDuration: Duration
        /// Upper scale bound for the adaptive preview.
        public var maximumPreviewScale: Double
        /// Fixed final scale, capped by the path tracer's allocation limits.
        public var finalRenderScale: Double

        public init(
            renderDelay: Duration = .milliseconds(16), minimumSamples: Int = 1,
            fadeDuration: Duration = .milliseconds(250), maximumPreviewScale: Double = 0.5,
            finalRenderScale: Double = 1
        ) {
            self.renderDelay = renderDelay
            self.minimumSamples = minimumSamples
            self.fadeDuration = fadeDuration
            self.maximumPreviewScale = maximumPreviewScale
            self.finalRenderScale = finalRenderScale
        }
    }

    enum TransitionFailure: LocalizedError, Equatable {
        case configuration

        public var errorDescription: String? {
            "Use nonnegative transition durations, a minimum within the sample limit, and preview and final scales in (0, 1]."
        }
    }
}

/// One view epoch. The preview and final stages never share an accumulation.
struct PathPreviewHandoff {
    let configuration: FrostfieldPathTracingView.Transition
    let lowConfiguration: PathProgressivePreview.Configuration
    let finalConfiguration: PathProgressivePreview.Configuration
    var retainedPreview: (any MTLTexture)?
    private(set) var isFinal = false
    private(set) var isComplete = false
    private var lastTick: ContinuousClock.Instant?
    private var elapsed = Duration.zero
    private var fadeStart: Duration?

    init(
        _ configuration: FrostfieldPathTracingView.Transition, preview: PathProgressivePreview.Configuration,
        width: Int, height: Int
    ) throws {
        guard configuration.renderDelay >= .zero, configuration.fadeDuration >= .zero,
              configuration.minimumSamples >= 0, configuration.minimumSamples <= preview.maximumSamples,
              configuration.maximumPreviewScale.isFinite, configuration.maximumPreviewScale > 0,
              configuration.finalRenderScale.isFinite, configuration.finalRenderScale <= 1,
              configuration.maximumPreviewScale <= configuration.finalRenderScale else {
            throw FrostfieldPathTracingView.TransitionFailure.configuration
        }
        let scales = preview.renderScales
        guard scales.lowerBound.isFinite, scales.lowerBound > 0, scales.upperBound <= 1,
              preview.initialScale.map(scales.contains) != false else {
            throw PathProgressivePreview.Failure.configuration
        }
        self.configuration = configuration
        let finalScale = try PathProgressivePreview.feasibleScale(configuration.finalRenderScale, width: width, height: height)
        var final = preview
        final.renderScales = finalScale...finalScale
        final.initialScale = nil
        finalConfiguration = final

        var low = preview
        let limit = min(finalScale, configuration.maximumPreviewScale, low.renderScales.upperBound)
        low.renderScales = min(low.renderScales.lowerBound, limit)...limit
        low.initialScale = low.initialScale.map { min($0, limit) }
        lowConfiguration = low
    }

    mutating func tick(at time: ContinuousClock.Instant) {
        if let lastTick { elapsed += max(.zero, lastTick.duration(to: time)) }
        lastTick = time
    }

    mutating func suspend() { lastTick = nil }

    mutating func beginFinal(after completed: PathProgressivePreview.Frame?) -> Bool {
        guard !isFinal, elapsed >= configuration.renderDelay, let completed else { return false }
        retainedPreview = completed.texture
        isFinal = true
        return true
    }

    /// Nil keeps displaying the preview until the completed final count qualifies.
    mutating func fadeWeight(after completed: PathProgressivePreview.Frame?) -> Float? {
        if isComplete { return 1 }
        guard isFinal, let completed, completed.sampleCount >= configuration.minimumSamples else { return nil }
        if fadeStart == nil { fadeStart = elapsed }
        if configuration.fadeDuration == .zero { return 1 }
        return Float(min(1, (elapsed - (fadeStart ?? elapsed)) / configuration.fadeDuration))
    }

    mutating func restartFade() { fadeStart = nil }

    mutating func finishFade() {
        isComplete = true
        retainedPreview = nil
    }
}
