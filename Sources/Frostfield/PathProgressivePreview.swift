import Foundation
import Metal

/// An opt-in, GPU-timed progressive path view. Keep one per view and use it
/// serially on the tracer's queue. The caller encodes presentation and commits
/// the command buffer. Each render size owns a separate accumulation.
public final class PathProgressivePreview {
    public struct Configuration: Equatable, Sendable {
        public var targetGPUSeconds: Double
        public var renderScales: ClosedRange<Double>
        /// Defaults to the largest feasible scale in `renderScales`.
        public var initialScale: Double?
        public var samplesPerFrame: Int
        public var maximumSamples: Int
        /// Valid completed timings at a size before reconsidering it, from 2 to 120.
        public var measurementFrames: Int

        public init(
            targetGPUSeconds: Double = 0.016, renderScales: ClosedRange<Double> = 0.25...1,
            initialScale: Double? = nil, samplesPerFrame: Int = 1,
            maximumSamples: Int = PathTracer.maximumSamples, measurementFrames: Int = 8
        ) {
            self.targetGPUSeconds = targetGPUSeconds
            self.renderScales = renderScales
            self.initialScale = initialScale
            self.samplesPerFrame = samplesPerFrame
            self.maximumSamples = maximumSamples
            self.measurementFrames = measurementFrames
        }
    }

    public struct Frame {
        /// Private tracked RGBA16Float, with shader-read usage. Destinations
        /// alternate within a render size and are retired on reset or resize.
        public let texture: any MTLTexture
        public let renderScale: Double
        public let sampleCount: Int
        /// The duration of the most recent trace pass when available.
        public let gpuSeconds: Double?
    }

    public let configuration: Configuration
    public var canvasWidth: Int { policy.canvasWidth }
    public var canvasHeight: Int { policy.canvasHeight }
    /// The selected scale for the next encode. Read `latestFrame` to process
    /// the outstanding completion and its timing before querying this value.
    public var renderScale: Double { policy.scale }
    /// A device without compute pass timestamps continues at the selected scale.
    public var adaptsToGPUTime: Bool { timing != nil }

    private let tracer: PathTracer
    private var policy: ResolutionPolicy
    private var timing: PathTracer.Timing?
    private var generation: UInt64 = 0
    private var work: Work?
    private var previous: PathTracer.History?
    private var pending: Pending?
    private var published: Frame?

    private struct View: Equatable {
        let camera: PathTracer.CameraModel
        let pose: SceneTransform
        let settings: PathTracer.Settings
        let lighting: PathTracer.Lighting
    }

    private struct Work {
        let view: View
        let accumulation: PathTracer.Accumulation
        let colors: [any MTLTexture]
    }

    private struct Pending {
        let command: any MTLCommandBuffer
        let history: PathTracer.History
        let texture: any MTLTexture
        let scale: Double
        let samples: Int
        let generation: UInt64
    }

    public init(
        tracer: PathTracer, canvasWidth: Int, canvasHeight: Int, configuration: Configuration = .init()
    ) throws {
        self.tracer = tracer
        self.configuration = configuration
        policy = try ResolutionPolicy(configuration: configuration, width: canvasWidth, height: canvasHeight)
        timing = configuration.renderScales.lowerBound < configuration.renderScales.upperBound
            ? try? PathTracer.Timing(device: tracer.device) : nil
    }

    /// The most recent successful image. It remains available while a new
    /// resolution or reset is awaiting its first successful frame.
    public var latestFrame: Frame? {
        _ = retireCompletedWork()
        return published
    }

    /// A successful frame from the current history, excluding any retained
    /// image displayed while an explicitly reset accumulation starts again.
    package var currentFrame: Frame? {
        _ = retireCompletedWork()
        return previous == nil ? nil : published
    }

    /// Finds the requested fixed scale, bounded by the trace allocation limits.
    package static func feasibleScale(
        _ requested: Double, width: Int, height: Int
    ) throws -> Double {
        guard requested.isFinite, requested > 0, requested <= 1 else { throw Failure.configuration }
        let policy = try ResolutionPolicy(
            configuration: .init(renderScales: Double.leastNonzeroMagnitude...requested),
            width: width, height: height
        )
        return policy.scale
    }

    /// A scaled grid selected after a completed timing still needs a new trace.
    package var reachedSampleLimit: Bool {
        guard retireCompletedWork(), let previous, let work else { return false }
        let size = policy.renderSize
        return previous.sampleCount >= configuration.maximumSamples
            && work.accumulation.width == size.width && work.accumulation.height == size.height
    }

    /// Restarts progressive samples while keeping the selected scale and the
    /// last completed image. Call after scene, live uniform or in-place texture
    /// edits. Camera, pose, settings and lighting changes restart automatically.
    /// A frame invalidated by a reset may finish but cannot become current.
    public func reset() {
        generation &+= 1
        work = nil
        previous = nil
        policy.clearMeasurements()
    }

    /// Changes the presentation extent. The new trace preserves this aspect
    /// ratio even if the reduced width and height round to different fractions.
    public func resize(width: Int, height: Int) throws {
        guard width != canvasWidth || height != canvasHeight else { return }
        try policy.resize(width: width, height: height)
        reset()
    }

    /// Encodes a progressive batch and returns its texture for downstream
    /// passes in the same command buffer. Returns nil at the sample limit.
    /// A pending command must complete before the next encode. Camera rays use
    /// the canvas aspect and the rounded render size's pixel centers.
    @discardableResult
    public func encode(
        cameraModel: PathTracer.CameraModel, cameraTransform: SceneTransform = .init(),
        settings: PathTracer.Settings = .init(), lighting: PathTracer.Lighting? = nil,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (any MTLTexture)? {
        guard commandBuffer.commandQueue === tracer.queue, commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw PathTracer.Failure.commandBuffer
        }
        guard retireCompletedWork() else { throw Failure.frameInFlight }
        if case .stereoEquirectangular = cameraModel.projection { throw Failure.stereo }
        let size = policy.renderSize
        var model = cameraModel
        model.projectionAspectRatio = Float(canvasWidth) / Float(canvasHeight)
        let view = View(camera: model, pose: cameraTransform, settings: settings, lighting: lighting ?? tracer.defaultLighting)
        let existing = work.flatMap {
            $0.view == view && $0.accumulation.width == size.width && $0.accumulation.height == size.height ? $0 : nil
        }
        let history = existing == nil ? nil : previous
        let remaining = configuration.maximumSamples - (history?.sampleCount ?? 0)
        guard remaining > 0 else { return nil }
        let current = try existing ?? makeWork(view: view, width: size.width, height: size.height)
        let destination = published?.texture === current.colors[0] ? current.colors[1] : current.colors[0]
        let samples = min(configuration.samplesPerFrame, remaining)
        let next = try tracer.encode(
            cameraModel: model, cameraTransform: cameraTransform, destination: destination,
            accumulation: current.accumulation, previous: history, commandBuffer: commandBuffer,
            samplesPerPixel: samples, settings: settings, lighting: view.lighting, timing: timing
        )
        work = current
        previous = history
        pending = Pending(command: commandBuffer, history: next, texture: destination, scale: policy.scale,
                          samples: samples, generation: generation)
        return destination
    }

    /// Drops an outstanding command that was never enqueued. Discard that
    /// entire buffer. The last completed image remains valid.
    public func discardUnsubmittedFrame(_ commandBuffer: any MTLCommandBuffer) throws {
        guard let pending, pending.command === commandBuffer, commandBuffer.status == .notEnqueued else {
            throw Failure.discard
        }
        self.pending = nil
        if timing != nil { timing = try? PathTracer.Timing(device: tracer.device) }
        reset()
    }

    private func retireCompletedWork() -> Bool {
        guard let pending else { return true }
        let status = pending.command.status
        guard status == .completed || status == .error, timing?.isMeasuring != true else { return false }
        if status == .completed && pending.generation == generation {
            let seconds = timing?.duration
            previous = pending.history
            published = Frame(texture: pending.texture, renderScale: pending.scale,
                              sampleCount: pending.history.sampleCount, gpuSeconds: seconds)
            if pending.samples == configuration.samplesPerFrame { policy.record(seconds: seconds) }
        }
        self.pending = nil
        return true
    }

    private func makeWork(view: View, width: Int, height: Int) throws -> Work {
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.hazardTrackingMode = .tracked
        descriptor.usage = [.shaderRead, .shaderWrite]
        guard let first = tracer.device.makeTexture(descriptor: descriptor),
              let second = tracer.device.makeTexture(descriptor: descriptor) else { throw Failure.texture }
        first.label = "Frostfield path preview A"
        second.label = "Frostfield path preview B"
        return Work(view: view, accumulation: accumulation, colors: [first, second])
    }

    public enum Failure: LocalizedError, Equatable {
        case configuration
        case dimensions
        case stereo
        case frameInFlight
        case discard
        case texture

        public var errorDescription: String? {
            switch self {
            case .configuration: "Use a positive GPU target, scales in (0, 1], and valid sample and measurement limits."
            case .dimensions: "Use a canvas of up to 32,768 pixels per axis and scales within the tracer's accumulation limits."
            case .stereo: "Adaptive path preview supports planar and single-eye panoramic cameras."
            case .frameInFlight: "Wait for the outstanding preview command to complete before encoding again."
            case .discard: "Only the outstanding unsubmitted preview command can be discarded."
            case .texture: "Metal could not allocate the path preview color textures."
            }
        }
    }
}

extension PathProgressivePreview {
    struct ResolutionPolicy {
        private let configuration: Configuration
        private(set) var canvasWidth: Int
        private(set) var canvasHeight: Int
        private(set) var scale: Double
        private var scaleLimit: Double
        private var count = 0
        private var mean = 0.0

        init(configuration: Configuration, width: Int, height: Int) throws {
            let scales = configuration.renderScales
            let initial = configuration.initialScale ?? scales.upperBound
            guard configuration.targetGPUSeconds.isFinite, configuration.targetGPUSeconds > 0,
                  scales.lowerBound.isFinite, scales.upperBound.isFinite, scales.lowerBound > 0, scales.upperBound <= 1,
                  scales.contains(initial), (1...16).contains(configuration.samplesPerFrame),
                  (1...PathTracer.maximumSamples).contains(configuration.maximumSamples),
                  (2...120).contains(configuration.measurementFrames) else { throw Failure.configuration }
            self.configuration = configuration
            canvasWidth = width
            canvasHeight = height
            scaleLimit = try Self.scaleLimit(configuration, width: width, height: height)
            scale = min(initial, scaleLimit)
        }

        var renderSize: (width: Int, height: Int) {
            (max(1, Int(Double(canvasWidth) * scale)), max(1, Int(Double(canvasHeight) * scale)))
        }

        mutating func resize(width: Int, height: Int) throws {
            let limit = try Self.scaleLimit(configuration, width: width, height: height)
            canvasWidth = width
            canvasHeight = height
            scaleLimit = limit
            scale = min(scale, limit)
            clearMeasurements()
        }

        mutating func clearMeasurements() {
            count = 0
            mean = 0
        }

        mutating func record(seconds: Double?) {
            guard let seconds, seconds.isFinite, seconds > 0 else { return }
            count += 1
            mean += (seconds - mean) / Double(count)
            guard count >= configuration.measurementFrames else { return }
            let ratio = configuration.targetGPUSeconds / mean
            clearMeasurements()
            // Keep the same accumulation around the target and limit area
            // changes to a factor of two per measurement window.
            guard ratio < 1 / 1.15 || ratio > 1 / 0.85 else { return }
            scale = min(scaleLimit, max(configuration.renderScales.lowerBound, scale * sqrt(min(2, max(0.5, ratio)))))
        }

        private static func scaleLimit(_ configuration: Configuration, width: Int, height: Int) throws -> Double {
            guard (1...32_768).contains(width), (1...32_768).contains(height) else { throw Failure.dimensions }
            let longest = Double(max(width, height))
            let sideLimit = Double(PathTracer.maximumAccumulationSide) / longest
            let areaLimit = sqrt(Double(PathTracer.maximumAccumulationPixels) / (Double(width) * Double(height)))
            let limit = min(configuration.renderScales.upperBound, sideLimit, areaLimit)
            guard limit >= configuration.renderScales.lowerBound else { throw Failure.dimensions }
            return limit
        }
    }
}
