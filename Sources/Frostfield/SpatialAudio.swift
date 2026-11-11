import Accelerate
import AVFAudio
import Foundation
import simd
import Synchronization

/// Distance attenuation with the Web Audio `PannerNode` formulas that Three.js
/// `PositionalAudio` uses (`distanceModel`, `refDistance`, `maxDistance`, `rolloffFactor`).
public struct AudioDistanceModel: Equatable, Sendable, Codable {
    public enum Curve: String, Equatable, Sendable, Codable {
        /// `1 − f (clamp(d, ref, max) − ref) / (max − ref)`, with the rolloff clamped to 0...1.
        case linear
        /// `ref / (ref + f (max(d, ref) − ref))`, the Three.js default.
        case inverse
        /// `(max(d, ref) / ref)^−f`.
        case exponential
    }

    public let curve: Curve
    /// The distance in meters within which the gain is one.
    public let referenceDistance: Float
    /// The distance where the linear curve reaches its floor. The other curves ignore it.
    public let maximumDistance: Float
    public let rolloffFactor: Float

    /// Distances are meters: the reference distance must be positive, the maximum at least
    /// the reference, and the rolloff finite and nonnegative. The defaults match Three.js.
    public init(curve: Curve = .inverse, referenceDistance: Float = 1, maximumDistance: Float = 10_000, rolloffFactor: Float = 1) throws {
        guard referenceDistance.isFinite, referenceDistance > 0, maximumDistance.isFinite,
              maximumDistance >= referenceDistance, rolloffFactor.isFinite, rolloffFactor >= 0 else {
            throw SceneAudio.Failure.distanceModel
        }
        self.curve = curve
        self.referenceDistance = referenceDistance
        self.maximumDistance = maximumDistance
        self.rolloffFactor = rolloffFactor
    }

    /// Decodes through the validating initializer, so stored settings meet the same rules.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        do {
            try self.init(
                curve: container.decode(Curve.self, forKey: .curve),
                referenceDistance: container.decode(Float.self, forKey: .referenceDistance),
                maximumDistance: container.decode(Float.self, forKey: .maximumDistance),
                rolloffFactor: container.decode(Float.self, forKey: .rolloffFactor)
            )
        } catch let error as SceneAudio.Failure {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: error.localizedDescription))
        }
    }

    /// The gain at a distance in meters, computed in double precision.
    public func gain(atDistance distance: Float) -> Float {
        let d = Double(max(distance, 0))
        let reference = Double(referenceDistance), maximum = Double(maximumDistance), rolloff = Double(rolloffFactor)
        switch curve {
        case .linear:
            let factor = min(max(rolloff, 0), 1)
            guard maximum > reference else { return Float(1 - factor) }
            return Float(1 - factor * (min(max(d, reference), maximum) - reference) / (maximum - reference))
        case .inverse:
            return Float(reference / (reference + rolloff * (max(d, reference) - reference)))
        case .exponential:
            return Float(pow(max(d, reference) / reference, -rolloff))
        }
    }
}

/// A directional cone like Three.js `PositionalAudio.setDirectionalCone` and the Web Audio
/// cone attributes, with full angles in radians instead of degrees.
public struct AudioCone: Equatable, Sendable, Codable {
    /// Inside half of this angle around the source's facing, the gain is one.
    public let innerAngle: Float
    /// Outside half of this angle, the gain is `outerGain`.
    public let outerAngle: Float
    public let outerGain: Float

    /// An omnidirectional source, the default.
    public static let omnidirectional = try! AudioCone(innerAngle: 2 * .pi, outerAngle: 2 * .pi, outerGain: 0)

    /// Angles are full cone angles in 0...2π with inner ≤ outer; the outer gain is in 0...1.
    public init(innerAngle: Float, outerAngle: Float, outerGain: Float) throws {
        guard innerAngle.isFinite, outerAngle.isFinite, innerAngle >= 0, innerAngle <= outerAngle,
              outerAngle <= 2 * .pi + 1e-6, (0...1).contains(outerGain) else {
            throw SceneAudio.Failure.cone
        }
        self.innerAngle = innerAngle
        self.outerAngle = outerAngle
        self.outerGain = outerGain
    }

    /// Decodes through the validating initializer, so stored cones meet the same rules.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        do {
            try self.init(
                innerAngle: container.decode(Float.self, forKey: .innerAngle),
                outerAngle: container.decode(Float.self, forKey: .outerAngle),
                outerGain: container.decode(Float.self, forKey: .outerGain)
            )
        } catch let error as SceneAudio.Failure {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: error.localizedDescription))
        }
    }

    /// The Web Audio cone gain for a source facing `forward` heard from `toListener`.
    public func gain(forward: SIMD3<Float>, toListener: SIMD3<Float>) -> Float {
        let full = 2 * Float.pi - 1e-6
        guard innerAngle < full || outerAngle < full else { return 1 }
        let a = SIMD3<Double>(forward), b = SIMD3<Double>(toListener)
        let lengths = simd_length(a) * simd_length(b)
        guard lengths > 0, lengths.isFinite else { return 1 }
        let angle = acos(min(max(simd_dot(a, b) / lengths, -1), 1))
        let inner = Double(innerAngle) / 2, outer = Double(outerAngle) / 2
        if angle <= inner { return 1 }
        if angle >= outer { return outerGain }
        let x = (angle - inner) / (outer - inner)
        return Float((1 - x) + Double(outerGain) * x)
    }
}

/// How a positional source is spatialized.
public enum AudioPanning: Sendable, Equatable {
    /// Apple's high-quality head-related transfer function, the counterpart of Three.js's
    /// default `panningModel = "HRTF"`.
    case hrtf
    /// Web Audio equal-power panning, `gL = cos(xπ/2)` and `gR = sin(xπ/2)` with
    /// `x = (azimuth + 90°) / 180°` after folding rear azimuths forward. Elevation is ignored.
    case equalPower
    /// Apple's spherical-head model, a lighter alternative to HRTF.
    case sphericalHead
}

/// A listener-relative direction and the gains a source receives, all from Web Audio formulas.
public struct AudioSpatialization: Equatable, Sendable {
    public var distance: Float
    public var distanceGain: Float
    public var coneGain: Float
    /// `volume × distanceGain × coneGain`, the level applied to the source.
    public var gain: Float
    /// Degrees in −180...180; positive is to the listener's right, zero straight ahead.
    public var azimuth: Float
    /// Degrees in −90...90; positive is above the listener.
    public var elevation: Float
    /// Equal-power left and right gains for this azimuth.
    public var equalPowerGains: SIMD2<Float>
}

/// The spatial math the engine applies, exposed for tests, tools and custom mixers.
public enum SpatialAudioMath {
    /// The Web Audio azimuth and elevation of a source in degrees, for a listener whose
    /// facing and up vectors need not be unit length or exactly orthogonal.
    public static func azimuthElevation(
        source: SIMD3<Float>, listener: SIMD3<Float>, forward: SIMD3<Float>, up: SIMD3<Float>
    ) -> (azimuth: Float, elevation: Float) {
        let offset = SIMD3<Double>(source) - SIMD3<Double>(listener)
        let length = simd_length(offset)
        guard length > 0 else { return (0, 0) }
        let toSource = offset / length
        let front = simd_normalize(SIMD3<Double>(forward))
        let right = simd_normalize(simd_cross(front, SIMD3<Double>(up)))
        let trueUp = simd_cross(right, front)
        let upProjection = simd_dot(toSource, trueUp)
        let horizontal = toSource - upProjection * trueUp
        var azimuth: Double = 0
        let horizontalLength = simd_length(horizontal)
        if horizontalLength > 1e-12 {
            let projected = horizontal / horizontalLength
            azimuth = 180 * acos(min(max(simd_dot(projected, right), -1), 1)) / .pi
            if simd_dot(projected, front) < 0 { azimuth = 360 - azimuth }
            azimuth = azimuth <= 270 ? 90 - azimuth : 450 - azimuth
        }
        var elevation = 90 - 180 * acos(min(max(simd_dot(toSource, trueUp), -1), 1)) / .pi
        if elevation > 90 { elevation = 180 - elevation } else if elevation < -90 { elevation = -180 - elevation }
        return (Float(azimuth), Float(elevation))
    }

    /// Folds a rear azimuth into −90...90 degrees, as Web Audio's equal-power panner does.
    public static func foldedAzimuth(_ azimuth: Float) -> Float {
        if azimuth < -90 { return -180 - azimuth }
        if azimuth > 90 { return 180 - azimuth }
        return azimuth
    }

    /// Web Audio's equal-power gains for a mono source at an azimuth in degrees.
    public static func equalPowerGains(azimuth: Float) -> SIMD2<Float> {
        let x = (Double(foldedAzimuth(azimuth)) + 90) / 180
        return SIMD2(Float(cos(x * .pi / 2)), Float(sin(x * .pi / 2)))
    }

    /// Every gain and angle for a source pose heard by a listener pose.
    public static func spatialize(
        sourcePosition: SIMD3<Float>, sourceForward: SIMD3<Float>, listenerPosition: SIMD3<Float>,
        listenerForward: SIMD3<Float>, listenerUp: SIMD3<Float>, distanceModel: AudioDistanceModel,
        cone: AudioCone, volume: Float
    ) -> AudioSpatialization {
        let distance = Float(simd_distance(SIMD3<Double>(sourcePosition), SIMD3<Double>(listenerPosition)))
        let distanceGain = distanceModel.gain(atDistance: distance)
        let coneGain = cone.gain(forward: sourceForward, toListener: listenerPosition - sourcePosition)
        let angles = azimuthElevation(source: sourcePosition, listener: listenerPosition, forward: listenerForward, up: listenerUp)
        return AudioSpatialization(
            distance: distance, distanceGain: distanceGain, coneGain: coneGain, gain: volume * distanceGain * coneGain,
            azimuth: angles.azimuth, elevation: angles.elevation, equalPowerGains: equalPowerGains(azimuth: angles.azimuth)
        )
    }
}

/// Spatial audio for a scene graph on `AVAudioEngine`, like Three.js `AudioListener`,
/// `Audio` and `PositionalAudio`.
///
/// The listener follows the graph's camera transform and positional sources follow the
/// world origin of the node they are attached to, facing its local −Z for cones, as
/// Frostfield lights do. Call `update(from:)` once per frame after animating the graph.
///
/// Frostfield computes each source's distance and cone gains with the Web Audio formulas
/// and applies them as the source's volume; the environment node's own attenuation is
/// disabled, so every source keeps its own model, as in Three.js. HRTF and spherical-head
/// sources use `AVAudioEnvironmentNode` at their true positions. Equal-power sources are
/// placed at half their folded azimuth, because Apple's equal-power panner reaches a full
/// side at 45° where Web Audio's reaches it at 90°; the result matches Web Audio's gains.
///
/// Sources need a mono signal for spatialization; stereo buffers are averaged to mono.
/// Non-positional sources play through a varispeed unit into the main mixer, like
/// Three.js `Audio`; the environment node applies positional sources' rates. Rate changes
/// speed and pitch together, like Web Audio `playbackRate`. Volumes are at most one. Use
/// one instance from one thread at a time.
public final class SceneAudio {
    public enum Failure: LocalizedError, Equatable {
        case distanceModel
        case cone
        case volume
        case rate
        case analyser
        case missingNode(UUID)
        case missingSource
        case emptyBuffer
        case format
        case fftSize(Int)

        public var errorDescription: String? {
            switch self {
            case .distanceModel:
                "Audio distance models need a positive reference distance, a maximum at least that large, and a finite nonnegative rolloff."
            case .cone: "Audio cones need full angles in 0...2π with inner ≤ outer and an outer gain in 0...1."
            case .volume: "Audio volume must be finite and in 0...1."
            case .rate: "Audio playback rate must be finite and in 0.5...2."
            case .analyser: "Analyser smoothing must be in 0...1 with finite, ordered decibel bounds."
            case .missingNode(let id): "Audio source node \(id) is not in the scene graph."
            case .missingSource: "The audio source is not part of this scene audio."
            case .emptyBuffer: "Audio sources need a buffer with at least one frame."
            case .format: "The audio buffer must hold deinterleaved float samples."
            case .fftSize(let size): "An analyser needs a power-of-two FFT size from 32 through 32768, not \(size)."
            }
        }
    }

    /// A source's spatial behavior.
    public struct SourceSettings: Equatable, Sendable {
        public var distanceModel: AudioDistanceModel
        public var cone: AudioCone
        /// A linear gain in 0...1 applied before distance and cone, like Three.js `setVolume`.
        public var volume: Float
        public var panning: AudioPanning
        /// Playback rate, 1 for normal speed, from 0.5 through 2.
        public var rate: Float

        public init(
            distanceModel: AudioDistanceModel = try! AudioDistanceModel(), cone: AudioCone = .omnidirectional,
            volume: Float = 1, panning: AudioPanning = .hrtf, rate: Float = 1
        ) {
            self.distanceModel = distanceModel
            self.cone = cone
            self.volume = volume
            self.panning = panning
            self.rate = rate
        }
    }

    public struct SourceID: Hashable, Sendable {
        fileprivate let value: Int
    }

    private struct Source {
        let player: AVAudioPlayerNode
        /// Resamples a non-positional source, which bypasses the environment node's rate.
        let varispeed: AVAudioUnitVarispeed?
        let nodeID: UUID?
        let buffer: AVAudioPCMBuffer
        let playbackOptions: AVAudioPlayerNodeBufferOptions
        var settings: SourceSettings

        func schedule() {
            player.scheduleBuffer(buffer, at: nil, options: playbackOptions)
        }

        func apply(rate: Float) {
            if let varispeed { varispeed.rate = rate } else { player.rate = rate }
        }
    }

    public let engine: AVAudioEngine
    /// The spatializer every positional source feeds. Its reverb and output type remain
    /// available; its distance attenuation stays disabled.
    public let environment: AVAudioEnvironmentNode
    private var sources: [SourceID: Source] = [:]
    private var nextSource = 0
    private var analyser: AudioAnalyser?

    /// Attaches an environment node to the engine's main mixer. Enable manual rendering on the
    /// engine before calling this to render offline.
    public init(engine: AVAudioEngine = AVAudioEngine()) {
        self.engine = engine
        environment = AVAudioEnvironmentNode()
        engine.attach(environment)
        let sampleRate = engine.mainMixerNode.outputFormat(forBus: 0).sampleRate
        let stereo = AVAudioFormat(standardFormatWithSampleRate: sampleRate > 0 ? sampleRate : 48_000, channels: 2)
        engine.connect(environment, to: engine.mainMixerNode, format: stereo)
        let attenuation = environment.distanceAttenuationParameters
        attenuation.distanceAttenuationModel = .exponential
        attenuation.referenceDistance = 1
        attenuation.maximumDistance = 1_000_000
        attenuation.rolloffFactor = 0
        environment.listenerPosition = AVAudio3DPoint(x: 0, y: 0, z: 0)
        environment.listenerVectorOrientation = AVAudio3DVectorOrientation(
            forward: AVAudio3DVector(x: 0, y: 0, z: -1), up: AVAudio3DVector(x: 0, y: 1, z: 0)
        )
    }

    /// Adds a source that plays a buffer, attached to a scene node for positional audio or
    /// nil for a non-positional source. Looping sources repeat until stopped.
    public func addSource(
        _ buffer: AVAudioPCMBuffer, attachedTo nodeID: UUID?, settings: SourceSettings = .init(), loops: Bool = true
    ) throws -> SourceID {
        guard buffer.frameLength > 0 else { throw Failure.emptyBuffer }
        try Self.validate(settings)
        let playable = try nodeID == nil ? buffer : Self.mono(buffer)
        let player = AVAudioPlayerNode()
        engine.attach(player)
        var varispeed: AVAudioUnitVarispeed?
        if nodeID != nil {
            engine.connect(player, to: environment, format: playable.format)
            player.renderingAlgorithm = Self.algorithm(settings.panning)
            player.sourceMode = .pointSource
        } else {
            let unit = AVAudioUnitVarispeed()
            engine.attach(unit)
            engine.connect(player, to: unit, format: buffer.format)
            engine.connect(unit, to: engine.mainMixerNode, format: buffer.format)
            varispeed = unit
        }
        let id = SourceID(value: nextSource)
        nextSource += 1
        let source = Source(
            player: player, varispeed: varispeed, nodeID: nodeID,
            buffer: playable, playbackOptions: loops ? .loops : [], settings: settings
        )
        source.schedule()
        sources[id] = source
        player.volume = nodeID == nil ? settings.volume : 0
        source.apply(rate: settings.rate)
        return id
    }

    /// Changes a source's distance model, cone, volume, panning or rate. Takes effect at the next update.
    public func setSettings(_ settings: SourceSettings, for id: SourceID) throws {
        guard var source = sources[id] else { throw Failure.missingSource }
        try Self.validate(settings)
        source.settings = settings
        if source.nodeID != nil { source.player.renderingAlgorithm = Self.algorithm(settings.panning) }
        source.apply(rate: settings.rate)
        if source.nodeID == nil { source.player.volume = settings.volume }
        sources[id] = source
    }

    /// The underlying player node, for scheduling more audio or reading its time.
    public func player(for id: SourceID) -> AVAudioPlayerNode? {
        sources[id]?.player
    }

    public func play(_ id: SourceID) throws {
        guard let source = sources[id] else { throw Failure.missingSource }
        source.player.play()
    }

    /// Stops playback and resets the source to the beginning of its original buffer.
    public func stop(_ id: SourceID) throws {
        guard let source = sources[id] else { throw Failure.missingSource }
        source.player.stop()
        source.schedule()
    }

    public func removeSource(_ id: SourceID) {
        guard let source = sources.removeValue(forKey: id) else { return }
        source.player.stop()
        engine.detach(source.player)
        if let varispeed = source.varispeed { engine.detach(varispeed) }
    }

    /// Moves the listener to the graph's camera and every positional source to its node,
    /// and applies each source's gain. Returns the spatialization of each positional source.
    @discardableResult
    public func update(from graph: SceneGraph) throws -> [SourceID: AudioSpatialization] {
        let camera = graph.cameraTransform
        let listener = camera.translation
        let forward = camera.rotation.act(SIMD3(0, 0, -1))
        let up = camera.rotation.act(SIMD3(0, 1, 0))
        environment.listenerPosition = Self.point(listener)
        environment.listenerVectorOrientation = AVAudio3DVectorOrientation(forward: Self.vector(forward), up: Self.vector(up))
        let positional = sources.filter { $0.value.nodeID != nil }
        guard !positional.isEmpty else { return [:] }
        var worlds: [UUID: simd_float4x4] = [:]
        for node in try graph.resolve() { worlds[node.id] = node.worldTransform }
        var result: [SourceID: AudioSpatialization] = [:]
        for (id, source) in positional {
            guard let nodeID = source.nodeID, let world = worlds[nodeID] else { throw Failure.missingNode(source.nodeID ?? UUID()) }
            let position = SIMD3(world.columns.3.x, world.columns.3.y, world.columns.3.z)
            let facing = world * SIMD4<Float>(0, 0, -1, 0)
            let spatial = SpatialAudioMath.spatialize(
                sourcePosition: position, sourceForward: SIMD3(facing.x, facing.y, facing.z), listenerPosition: listener,
                listenerForward: forward, listenerUp: up, distanceModel: source.settings.distanceModel,
                cone: source.settings.cone, volume: source.settings.volume
            )
            source.player.volume = min(max(spatial.gain, 0), 1)
            if source.settings.panning == .equalPower {
                // Apple's equal-power panner turns twice as fast as Web Audio's; halve the folded azimuth.
                let angle = Double(SpatialAudioMath.foldedAzimuth(spatial.azimuth)) / 2 * .pi / 180
                let front = simd_normalize(forward)
                let right = simd_normalize(simd_cross(front, up))
                let distance = max(spatial.distance, 0.001)
                let virtual = listener + distance * (Float(sin(angle)) * right + Float(cos(angle)) * front)
                source.player.position = Self.point(virtual)
            } else {
                source.player.position = Self.point(position)
            }
            result[id] = spatial
        }
        return result
    }

    /// An analyser fed by a tap on the main mixer, like Three.js `AudioAnalyser`. One analyser
    /// per scene audio; calling again replaces it.
    public func makeAnalyser(fftSize: Int = 2_048, smoothingTimeConstant: Float = 0.8) throws -> AudioAnalyser {
        let analyser = try AudioAnalyser(fftSize: fftSize, smoothingTimeConstant: smoothingTimeConstant)
        if self.analyser != nil { engine.mainMixerNode.removeTap(onBus: 0) }
        engine.mainMixerNode.installTap(onBus: 0, bufferSize: AVAudioFrameCount(fftSize), format: nil) { buffer, _ in
            analyser.process(buffer)
        }
        self.analyser = analyser
        return analyser
    }

    deinit {
        if analyser != nil { engine.mainMixerNode.removeTap(onBus: 0) }
    }

    private static func validate(_ settings: SourceSettings) throws {
        guard settings.volume.isFinite, (0...1).contains(settings.volume) else { throw Failure.volume }
        guard settings.rate.isFinite, (0.5...2).contains(settings.rate) else { throw Failure.rate }
    }

    private static func algorithm(_ panning: AudioPanning) -> AVAudio3DMixingRenderingAlgorithm {
        switch panning {
        case .hrtf: .HRTFHQ
        case .equalPower: .equalPowerPanning
        case .sphericalHead: .sphericalHead
        }
    }

    private static func mono(_ buffer: AVAudioPCMBuffer) throws -> AVAudioPCMBuffer {
        let channels = Int(buffer.format.channelCount)
        guard let data = buffer.floatChannelData, !buffer.format.isInterleaved else { throw Failure.format }
        if channels == 1 { return buffer }
        guard let format = AVAudioFormat(standardFormatWithSampleRate: buffer.format.sampleRate, channels: 1),
              let mono = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: buffer.frameLength),
              let output = mono.floatChannelData else { throw Failure.format }
        mono.frameLength = buffer.frameLength
        for frame in 0..<Int(buffer.frameLength) {
            var sum: Float = 0
            for channel in 0..<channels { sum += data[channel][frame] }
            output[0][frame] = sum / Float(channels)
        }
        return mono
    }

    private static func point(_ value: SIMD3<Float>) -> AVAudio3DPoint {
        AVAudio3DPoint(x: value.x, y: value.y, z: value.z)
    }

    private static func vector(_ value: SIMD3<Float>) -> AVAudio3DVector {
        AVAudio3DVector(x: value.x, y: value.y, z: value.z)
    }
}

/// Frequency analysis with the Web Audio `AnalyserNode` definition that Three.js
/// `AudioAnalyser` reads: a Blackman window, magnitudes `|X[k]| / N`, exponential smoothing
/// by `smoothingTimeConstant`, decibels, and bytes scaled between `minDecibels` and `maxDecibels`.
///
/// Feed it samples from any thread; read the spectrum from any thread.
public final class AudioAnalyser: Sendable {
    public let fftSize: Int
    public let smoothingTimeConstant: Float
    public let minDecibels: Float
    public let maxDecibels: Float

    private struct State {
        var samples: [Float]
        var smoothed: [Float]
        var processed = 0
    }

    private let state: Mutex<State>
    private let window: [Float]

    /// The FFT size is a power of two from 32 through 32768; smoothing is in 0...1; the
    /// decibel range must be ordered. Defaults match Web Audio.
    public init(fftSize: Int = 2_048, smoothingTimeConstant: Float = 0.8, minDecibels: Float = -100, maxDecibels: Float = -30) throws {
        guard (32...32_768).contains(fftSize), fftSize.nonzeroBitCount == 1 else { throw SceneAudio.Failure.fftSize(fftSize) }
        guard (0...1).contains(smoothingTimeConstant), minDecibels.isFinite, maxDecibels.isFinite, minDecibels < maxDecibels else {
            throw SceneAudio.Failure.analyser
        }
        self.fftSize = fftSize
        self.smoothingTimeConstant = smoothingTimeConstant
        self.minDecibels = minDecibels
        self.maxDecibels = maxDecibels
        window = (0..<fftSize).map { n in
            let phase = 2 * Double.pi * Double(n) / Double(fftSize)
            return Float(0.42 - 0.5 * cos(phase) + 0.08 * cos(2 * phase))
        }
        state = Mutex(State(samples: Array(repeating: 0, count: fftSize), smoothed: Array(repeating: 0, count: fftSize / 2)))
    }

    /// The number of frequency bins, half the FFT size.
    public var frequencyBinCount: Int { fftSize / 2 }

    /// Appends samples, keeping the most recent `fftSize`. Multichannel buffers are averaged.
    public func process(_ buffer: AVAudioPCMBuffer) {
        guard let data = buffer.floatChannelData, !buffer.format.isInterleaved else { return }
        let channels = Int(buffer.format.channelCount), frames = Int(buffer.frameLength)
        var mono = [Float](repeating: 0, count: frames)
        for channel in 0..<channels {
            for frame in 0..<frames { mono[frame] += data[channel][frame] / Float(channels) }
        }
        process(mono)
    }

    /// The number of samples received so far. Engine taps can deliver after a render returns.
    public var processedSampleCount: Int {
        state.withLock { $0.processed }
    }

    /// Appends mono samples, keeping the most recent `fftSize`.
    public func process(_ samples: [Float]) {
        state.withLock { state in
            state.processed += samples.count
            if samples.count >= fftSize {
                state.samples = Array(samples.suffix(fftSize))
            } else {
                state.samples.removeFirst(samples.count)
                state.samples.append(contentsOf: samples)
            }
        }
    }

    /// Computes the spectrum of the latest samples, applies smoothing, and returns decibels
    /// per bin, like `getFloatFrequencyData`. Each call advances the smoothing once.
    public func floatFrequencyData() -> [Float] {
        state.withLock { state in
            let magnitudes = spectrum(of: state.samples)
            let tau = smoothingTimeConstant
            for bin in state.smoothed.indices {
                let value = tau * state.smoothed[bin] + (1 - tau) * magnitudes[bin]
                state.smoothed[bin] = value.isFinite ? value : 0
            }
            return state.smoothed.map { 20 * log10($0) }
        }
    }

    /// Bytes per bin, `255 (dB − min) / (max − min)` clamped to 0...255, like `getByteFrequencyData`.
    public func frequencyData() -> [UInt8] {
        floatFrequencyData().map { decibels in
            let scaled = 255 * (decibels - minDecibels) / (maxDecibels - minDecibels)
            return UInt8(min(max(scaled.isFinite ? scaled : 0, 0), 255))
        }
    }

    /// The mean of `frequencyData()`, like Three.js `getAverageFrequency`.
    public func averageFrequency() -> Float {
        let data = frequencyData()
        return Float(data.reduce(0) { $0 + Int($1) }) / Float(data.count)
    }

    /// The frequency in hertz at the center of a bin, for a sample rate.
    public func frequency(ofBin bin: Int, sampleRate: Double) -> Double {
        Double(bin) * sampleRate / Double(fftSize)
    }

    private func spectrum(of samples: [Float]) -> [Float] {
        let windowed = vDSP.multiply(samples, window)
        let imaginary = [Float](repeating: 0, count: fftSize)
        guard let transform = try? vDSP.DiscreteFourierTransform(
            count: fftSize, direction: .forward, transformType: .complexComplex, ofType: Float.self
        ) else { return Array(repeating: 0, count: fftSize / 2) }
        let (real, imag) = transform.transform(real: windowed, imaginary: imaginary)
        let scale = 1 / Float(fftSize)
        return (0..<(fftSize / 2)).map { bin in
            (real[bin] * real[bin] + imag[bin] * imag[bin]).squareRoot() * scale
        }
    }
}
