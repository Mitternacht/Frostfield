import AVFAudio
import Foundation
import Frostfield
import Testing
import simd

/// Web Audio distance, cone and panning formulas, and the AVAudioEngine graph that applies them.
struct SpatialAudioTests {
    private static let sampleRate = 48_000.0
    private static let amplitude: Float = 0.5
    /// RMS of the test sine: amplitude / √2.
    private static let sineRMS = Double(amplitude) / 2.0.squareRoot()

    // MARK: Math

    @Test
    func `Distance models follow the Web Audio formulas`() throws {
        let inverse = try AudioDistanceModel(curve: .inverse, referenceDistance: 2, rolloffFactor: 0.5)
        for (d, expected) in [(0.5, 1.0), (2, 1), (10, 2.0 / (2 + 0.5 * 8)), (100, 2.0 / (2 + 0.5 * 98))] as [(Float, Double)] {
            #expect(abs(Double(inverse.gain(atDistance: d)) - expected) < 1e-6, "inverse \(d)")
        }
        let linear = try AudioDistanceModel(curve: .linear, referenceDistance: 1, maximumDistance: 11, rolloffFactor: 1)
        for (d, expected) in [(0.2, 1.0), (6, 0.5), (11, 0), (40, 0)] as [(Float, Double)] {
            #expect(abs(Double(linear.gain(atDistance: d)) - expected) < 1e-6, "linear \(d)")
        }
        // Web Audio clamps the linear rolloff to one.
        #expect(try AudioDistanceModel(curve: .linear, referenceDistance: 1, maximumDistance: 11, rolloffFactor: 3).gain(atDistance: 6) == 0.5)
        #expect(try AudioDistanceModel(curve: .linear, referenceDistance: 1, maximumDistance: 11, rolloffFactor: 0.5).gain(atDistance: 11) == 0.5)
        let exponential = try AudioDistanceModel(curve: .exponential, referenceDistance: 1, rolloffFactor: 2)
        #expect(abs(exponential.gain(atDistance: 4) - 1.0 / 16) < 1e-7)
        #expect(exponential.gain(atDistance: 0.3) == 1)
        #expect(throws: SceneAudio.Failure.distanceModel) { try AudioDistanceModel(referenceDistance: 0) }
        #expect(throws: SceneAudio.Failure.distanceModel) { try AudioDistanceModel(referenceDistance: 5, maximumDistance: 2) }
    }

    @Test
    func `Cones interpolate between the inner and outer angles`() throws {
        let cone = try AudioCone(innerAngle: .pi / 2, outerAngle: .pi, outerGain: 0.2)
        let forward = SIMD3<Float>(0, 0, -1)
        func direction(degrees: Float) -> SIMD3<Float> {
            let a = degrees * .pi / 180
            return SIMD3(sin(a), 0, -cos(a))
        }
        #expect(cone.gain(forward: forward, toListener: direction(degrees: 0)) == 1)
        #expect(cone.gain(forward: forward, toListener: direction(degrees: 44)) == 1)
        // Halfway between the 45° and 90° half-angles: (1 − 0.5) + 0.2 × 0.5.
        #expect(abs(cone.gain(forward: forward, toListener: direction(degrees: 67.5)) - 0.6) < 1e-5)
        #expect(cone.gain(forward: forward, toListener: direction(degrees: 120)) == 0.2)
        #expect(AudioCone.omnidirectional.gain(forward: forward, toListener: direction(degrees: 180)) == 1)
        #expect(throws: SceneAudio.Failure.cone) { try AudioCone(innerAngle: 2, outerAngle: 1, outerGain: 0) }
    }

    @Test
    func `Decoding distance models and cones runs their validation`() throws {
        let model = try AudioDistanceModel(curve: .linear, referenceDistance: 2, maximumDistance: 9, rolloffFactor: 0.5)
        #expect(try JSONDecoder().decode(AudioDistanceModel.self, from: JSONEncoder().encode(model)) == model)
        let cone = try AudioCone(innerAngle: 1, outerAngle: 2, outerGain: 0.3)
        #expect(try JSONDecoder().decode(AudioCone.self, from: JSONEncoder().encode(cone)) == cone)
        for json in [
            #"{"curve":"inverse","referenceDistance":0,"maximumDistance":1,"rolloffFactor":1}"#,
            #"{"curve":"linear","referenceDistance":1,"maximumDistance":-3,"rolloffFactor":-2}"#,
        ] {
            #expect(throws: DecodingError.self, "\(json)") { try JSONDecoder().decode(AudioDistanceModel.self, from: Data(json.utf8)) }
        }
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(AudioCone.self, from: Data(#"{"innerAngle":3,"outerAngle":1,"outerGain":7}"#.utf8))
        }
    }

    @Test
    func `Azimuth, elevation and equal-power gains follow the Web Audio panner`() {
        let listener = SIMD3<Float>(0, 0, 0), forward = SIMD3<Float>(0, 0, -1), up = SIMD3<Float>(0, 1, 0)
        func angles(_ source: SIMD3<Float>) -> (Float, Float) {
            let result = SpatialAudioMath.azimuthElevation(source: source, listener: listener, forward: forward, up: up)
            return (result.azimuth, result.elevation)
        }
        #expect(angles(SIMD3(3, 0, 0)) == (90, 0))
        #expect(angles(SIMD3(-3, 0, 0)) == (-90, 0))
        #expect(angles(SIMD3(0, 0, -3)) == (0, 0))
        #expect(angles(SIMD3(0, 0, 3)).0 == -180)
        let diagonal = angles(SIMD3(1, 0, -1))
        #expect(abs(diagonal.0 - 45) < 1e-4 && diagonal.1 == 0)
        let raised = angles(SIMD3(1, 1, 0))
        #expect(abs(raised.0 - 90) < 1e-4 && abs(raised.1 - 45) < 1e-4)
        #expect(abs(angles(SIMD3(0, 2, 0)).1 - 90) < 1e-4)
        // A listener turned to face −X hears a source at −Z on its right.
        let turned = SpatialAudioMath.azimuthElevation(source: SIMD3(0, 0, -1), listener: .zero, forward: SIMD3(-1, 0, 0), up: up)
        #expect(abs(turned.azimuth - 90) < 1e-4)

        func near(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Bool { simd_reduce_max(abs(a - b)) < 1e-6 }
        #expect(near(SpatialAudioMath.equalPowerGains(azimuth: 0), SIMD2(repeating: Float(0.5).squareRoot())))
        #expect(near(SpatialAudioMath.equalPowerGains(azimuth: 30), SIMD2(0.5, Float(0.75).squareRoot())))
        #expect(near(SpatialAudioMath.equalPowerGains(azimuth: -90), SIMD2(1, 0)))
        // Rear azimuths fold forward: 150° pans like 30°.
        #expect(near(SpatialAudioMath.equalPowerGains(azimuth: 150), SIMD2(0.5, Float(0.75).squareRoot())))
    }

    // MARK: Offline engine

    private func sine(
        frequency: Double = 440, frames: AVAudioFrameCount = AVAudioFrameCount(SpatialAudioTests.sampleRate)
    ) -> AVAudioPCMBuffer {
        let format = AVAudioFormat(standardFormatWithSampleRate: Self.sampleRate, channels: 1)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)!
        buffer.frameLength = frames
        for index in 0..<Int(frames) {
            buffer.floatChannelData![0][index] = Self.amplitude * Float(sin(2 * .pi * frequency * Double(index) / Self.sampleRate))
        }
        return buffer
    }

    private func offlineEngine() throws -> AVAudioEngine {
        let engine = AVAudioEngine()
        let format = try #require(AVAudioFormat(standardFormatWithSampleRate: Self.sampleRate, channels: 2))
        try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4_096)
        return engine
    }

    /// Renders about a second after a short warm-up and returns each channel's RMS.
    private func render(_ engine: AVAudioEngine, play: () throws -> Void) throws -> SIMD2<Double> {
        try engine.start()
        try play()
        let output = try #require(AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: 4_096))
        var sums = SIMD2<Double>.zero
        var count = 0
        for block in 0..<12 {
            #expect(try engine.renderOffline(4_096, to: output) == .success)
            guard block >= 2 else { continue }
            for frame in 0..<Int(output.frameLength) {
                let sample = SIMD2(Double(output.floatChannelData![0][frame]), Double(output.floatChannelData![1][frame]))
                sums += sample * sample
                count += 1
            }
        }
        engine.stop()
        return (sums / Double(count)).squareRoot()
    }

    /// One positional source on a node at `position`, heard from the graph's camera.
    private func renderSource(
        at position: SIMD3<Float>, nodeRotation: simd_quatf = simd_quatf(angle: 0, axis: SIMD3(0, 1, 0)),
        camera: SceneTransform = SceneTransform(), settings: SceneAudio.SourceSettings
    ) throws -> (rms: SIMD2<Double>, spatial: AudioSpatialization) {
        let engine = try offlineEngine()
        let audio = SceneAudio(engine: engine)
        let node = SceneNode(transform: SceneTransform(translation: position, rotation: nodeRotation))
        var graph = SceneGraph(roots: [node], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 100))
        graph.cameraTransform = camera
        let source = try audio.addSource(sine(), attachedTo: node.id, settings: settings)
        let spatial = try #require(try audio.update(from: graph)[source])
        let rms = try render(engine) { try audio.play(source) }
        return (rms, spatial)
    }

    @Test
    func `Offline equal-power renders match Web Audio gains for azimuth and distance`() throws {
        let inverse = try AudioDistanceModel(curve: .inverse, referenceDistance: 1, rolloffFactor: 1)
        let settings = SceneAudio.SourceSettings(distanceModel: inverse, panning: .equalPower)
        for degrees in [Float(0), 15, 30, 60, 120, -45, -150] {
            let angle = degrees * .pi / 180
            let (rms, spatial) = try renderSource(at: SIMD3(2 * sin(angle), 0, -2 * cos(angle)), settings: settings)
            #expect(abs(spatial.distanceGain - 0.5) < 1e-6)
            #expect(abs(spatial.azimuth - degrees) < 1e-3)
            let expected = SIMD2<Double>(SpatialAudioMath.equalPowerGains(azimuth: degrees)) * Self.sineRMS * 0.5
            #expect(simd_reduce_max(abs(rms - expected)) < 0.005 * Self.sineRMS, "azimuth \(degrees): \(rms) vs \(expected)")
        }
        // The exponential model scales the same direction by 1, 1/3 and 1/9.
        let exponential = SceneAudio.SourceSettings(
            distanceModel: try AudioDistanceModel(curve: .exponential, referenceDistance: 1, rolloffFactor: 1), panning: .equalPower
        )
        for distance in [Float(1), 3, 9] {
            let (rms, _) = try renderSource(at: SIMD3(0, 0, -distance), settings: exponential)
            let expected = Self.sineRMS * 0.5.squareRoot() / Double(distance)
            #expect(abs(rms.x - expected) < 0.005 * expected && abs(rms.y - expected) < 0.005 * expected, "distance \(distance)")
        }
    }

    @Test
    func `Cones follow the source node's facing and volume scales linearly`() throws {
        let cone = try AudioCone(innerAngle: .pi / 3, outerAngle: 2 * .pi / 3, outerGain: 0.25)
        let settings = SceneAudio.SourceSettings(
            distanceModel: try AudioDistanceModel(rolloffFactor: 0), cone: cone, volume: 0.8, panning: .equalPower
        )
        // Facing the listener (local −Z toward +Z after a half turn): full level.
        let facing = try renderSource(at: SIMD3(0, 0, -2), nodeRotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0)), settings: settings)
        #expect(facing.spatial.coneGain == 1)
        // Facing away: the outer gain.
        let away = try renderSource(at: SIMD3(0, 0, -2), settings: settings)
        #expect(away.spatial.coneGain == 0.25)
        let full = Self.sineRMS * 0.5.squareRoot() * 0.8
        #expect(abs(facing.rms.x - full) < 0.005 * full)
        #expect(abs(away.rms.x - full * 0.25) < 0.005 * full)
    }

    @Test
    func `HRTF follows the camera as the listener and keeps per-source distance gains`() throws {
        let settings = SceneAudio.SourceSettings(distanceModel: try AudioDistanceModel(rolloffFactor: 1), panning: .hrtf)
        let right = try renderSource(at: SIMD3(2, 0, 0), settings: settings)
        #expect(right.rms.y > 1.4 * right.rms.x)
        // Turning the camera around puts the same source on the listener's left.
        let turned = try renderSource(
            at: SIMD3(2, 0, 0), camera: SceneTransform(rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))), settings: settings
        )
        #expect(abs(turned.spatial.azimuth + 90) < 1e-3)
        #expect(turned.rms.x > 1.4 * turned.rms.y)
        // Straight ahead, the inverse model's 0.5 and 0.125 give a 4:1 level ratio through the HRTF.
        let near = try renderSource(at: SIMD3(0, 0, -2), settings: settings)
        let far = try renderSource(at: SIMD3(0, 0, -8), settings: settings)
        #expect(abs(near.rms.x / far.rms.x - 4) < 0.02)
        #expect(abs(near.rms.y / far.rms.y - 4) < 0.02)
    }

    @Test
    func `The scene audio graph builds and renders offline with non-positional sources and an analyser`() throws {
        let engine = try offlineEngine()
        let audio = SceneAudio(engine: engine)
        #expect(engine.isInManualRenderingMode)
        // At 48 kHz, 48,128 frames hold 940 whole cycles. The loop stays
        // phase-continuous even when the latest FFT window straddles the wrap.
        let tone = sine(frequency: 937.5, frames: 48_128)
        let ambient = try audio.addSource(tone, attachedTo: nil, settings: .init(volume: 0.5))
        let analyser = try audio.makeAnalyser(fftSize: 2_048, smoothingTimeConstant: 0)
        let graph = SceneGraph(roots: [], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
        #expect(try audio.update(from: graph).isEmpty)
        let rms = try render(engine) { try audio.play(ambient) }
        #expect(rms.x > 0.1 && abs(rms.x - rms.y) < 1e-6)
        // Taps may deliver after rendering returns; wait for a full window.
        for _ in 0..<500 where analyser.processedSampleCount < 2_048 { Thread.sleep(forTimeInterval: 0.002) }
        #expect(analyser.processedSampleCount >= 2_048)
        // 937.5 Hz is bin 40 at 48 kHz with 2048 points.
        let spectrum = analyser.floatFrequencyData()
        #expect(spectrum.indices.max { spectrum[$0] < spectrum[$1] } == 40)
        #expect(analyser.frequency(ofBin: 40, sampleRate: Self.sampleRate) == 937.5)
        audio.removeSource(ambient)
        #expect(throws: SceneAudio.Failure.missingSource) { try audio.play(ambient) }
    }

    @Test("Rejecting a positional buffer leaves the audio engine unchanged")
    func rejectsUnsupportedSourceWithoutAttachingPlayer() throws {
        let engine = try offlineEngine()
        let audio = SceneAudio(engine: engine)
        let before = engine.attachedNodes
        let format = try #require(AVAudioFormat(
            commonFormat: .pcmFormatInt16, sampleRate: Self.sampleRate, channels: 1, interleaved: false
        ))
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1))
        buffer.frameLength = 1
        #expect(throws: SceneAudio.Failure.format) { try audio.addSource(buffer, attachedTo: UUID()) }
        #expect(engine.attachedNodes == before)
    }

    @Test("Playing a stopped source schedules its original audio again", arguments: [false, true])
    func replaysStoppedSource(positional: Bool) throws {
        let engine = try offlineEngine()
        let audio = SceneAudio(engine: engine)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let node = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -1)))
        let source = try audio.addSource(
            sine(), attachedTo: positional ? node.id : nil, settings: .init(panning: .equalPower), loops: false
        )
        try audio.update(from: SceneGraph(roots: [node], camera: camera))
        try engine.start()
        defer { engine.stop() }
        let output = try #require(AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: 4_096))
        func peakAfterWarmUp() throws -> Float {
            for _ in 0..<3 { try #require(try engine.renderOffline(4_096, to: output) == .success) }
            let channel = try #require(output.floatChannelData?[0])
            return UnsafeBufferPointer(start: channel, count: Int(output.frameLength)).map(abs).max() ?? 0
        }

        try audio.play(source)
        let initialPeak = try peakAfterWarmUp()
        #expect(initialPeak > 0.1)
        try audio.stop(source)
        #expect(try peakAfterWarmUp() < 1e-6)
        try audio.play(source)
        #expect(abs(try peakAfterWarmUp() - initialPeak) < 1e-3)
    }

    /// The frequency of a rendered tone from its rising zero crossings, after a short warm-up.
    private func renderedFrequency(_ engine: AVAudioEngine, play: () throws -> Void) throws -> Double {
        try engine.start()
        try play()
        let output = try #require(AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: 4_096))
        var samples: [Float] = []
        for block in 0..<12 {
            #expect(try engine.renderOffline(4_096, to: output) == .success)
            if block >= 2 { samples += UnsafeBufferPointer(start: output.floatChannelData![0], count: Int(output.frameLength)) }
        }
        engine.stop()
        let rising = zip(samples, samples.dropFirst()).filter { $0 < 0 && $1 >= 0 }.count
        return Double(rising) * Self.sampleRate / Double(samples.count)
    }

    @Test
    func `Playback rate shifts pitch for non-positional and positional sources`() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let node = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -1)))
        let graph = SceneGraph(roots: [node], camera: camera)
        for positional in [false, true] {
            for rate in [Float(1), 2, 0.5] {
                let engine = try offlineEngine()
                let audio = SceneAudio(engine: engine)
                let source = try audio.addSource(
                    sine(frequency: 440), attachedTo: positional ? node.id : nil,
                    settings: .init(panning: .equalPower, rate: rate)
                )
                try audio.update(from: graph)
                let frequency = try renderedFrequency(engine) { try audio.play(source) }
                #expect(abs(frequency - 440 * Double(rate)) < 3, "positional \(positional) rate \(rate): \(frequency) Hz")
            }
            // Changing the rate later takes effect too.
            let engine = try offlineEngine()
            let audio = SceneAudio(engine: engine)
            let source = try audio.addSource(sine(frequency: 440), attachedTo: positional ? node.id : nil)
            try audio.setSettings(.init(panning: .equalPower, rate: 1.5), for: source)
            try audio.update(from: graph)
            let frequency = try renderedFrequency(engine) { try audio.play(source) }
            #expect(abs(frequency - 660) < 3, "positional \(positional) after setSettings: \(frequency) Hz")
            audio.removeSource(source)
        }
    }

    @Test
    func `The analyser matches the Web Audio Blackman spectrum in closed form`() throws {
        // A sine centered on bin 40 leaks only into bins 39–41 and 38–42 with a periodic Blackman window:
        // |X[k]| = A/2 × 0.42, |X[k±1]| = A/2 × 0.25, |X[k±2]| = A/2 × 0.04.
        let size = 2_048
        let samples = (0..<size).map { Self.amplitude * Float(sin(2 * Double.pi * 40 * Double($0) / Double(size) + 0.3)) }
        let analyser = try AudioAnalyser(fftSize: size, smoothingTimeConstant: 0, maxDecibels: 0)
        analyser.process(samples)
        let decibels = analyser.floatFrequencyData()
        func db(_ magnitude: Double) -> Float { Float(20 * log10(magnitude)) }
        #expect(abs(decibels[40] - db(0.25 * 0.42)) < 0.01)
        #expect(abs(decibels[39] - db(0.25 * 0.25)) < 0.01 && abs(decibels[41] - db(0.25 * 0.25)) < 0.01)
        #expect(abs(decibels[38] - db(0.25 * 0.04)) < 0.05 && abs(decibels[42] - db(0.25 * 0.04)) < 0.05)
        #expect(decibels[30] < -100)
        #expect(analyser.frequencyData()[40] == UInt8(255 * (db(0.105) + 100) / 100))

        // Smoothing 0.8 reaches 1 − 0.8ⁿ of the magnitude after n reads of a steady signal.
        let smoothed = try AudioAnalyser(fftSize: size, smoothingTimeConstant: 0.8)
        smoothed.process(samples)
        #expect(abs(smoothed.floatFrequencyData()[40] - db(0.105 * 0.2)) < 0.01)
        #expect(abs(smoothed.floatFrequencyData()[40] - db(0.105 * 0.36)) < 0.01)
        #expect(throws: SceneAudio.Failure.fftSize(1000)) { try AudioAnalyser(fftSize: 1000) }
    }
}
