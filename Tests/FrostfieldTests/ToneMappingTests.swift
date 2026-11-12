import Foundation
import Frostfield
import Metal
import Testing
import simd

/// CPU transliterations of the reference tone curves. Each one follows its cited
/// source line by line, independently of Frostfield's Metal implementation.
enum ToneReference {
    typealias RGB = SIMD3<Double>

    /// Column-major 3×3 product, as GLSL `mat3(c0, c1, c2) * v`.
    static func multiply(_ columns: (RGB, RGB, RGB), _ v: RGB) -> RGB {
        columns.0 * v.x + columns.1 * v.y + columns.2 * v.z
    }

    static func saturate(_ v: RGB) -> RGB { simd_clamp(v, RGB(repeating: 0), RGB(repeating: 1)) }
    static func pow(_ v: RGB, _ e: Double) -> RGB { RGB(Foundation.pow(v.x, e), Foundation.pow(v.y, e), Foundation.pow(v.z, e)) }

    // three.js r187 src/renderers/shaders/ShaderChunk/tonemapping_pars_fragment.glsl.js
    static func linear(_ c: RGB) -> RGB { saturate(c) }

    static func reinhard(_ c: RGB) -> RGB { saturate(c / (1 + c)) }

    static func cineon(_ color: RGB) -> RGB {
        let c = simd_max(RGB(repeating: 0), color - 0.004)
        return pow((c * (6.2 * c + 0.5)) / (c * (6.2 * c + 1.7) + 0.06), 2.2)
    }

    static func acesFilmic(_ color: RGB) -> RGB {
        let input = (RGB(0.59719, 0.07600, 0.02840), RGB(0.35458, 0.90834, 0.13383), RGB(0.04823, 0.01566, 0.83777))
        let output = (RGB(1.60475, -0.10208, -0.00327), RGB(-0.53108, 1.10813, -0.07276), RGB(-0.07367, -0.00605, 1.07602))
        let v = multiply(input, color / 0.6)
        let a = v * (v + 0.0245786) - 0.000090537
        let b = v * (0.983729 * v + 0.4329510) + 0.238081
        return saturate(multiply(output, a / b))
    }

    static func agx(_ input: RGB, look: HDRDisplay.AgXLook = .default) -> RGB {
        let rec2020ToSRGB = (RGB(1.6605, -0.1246, -0.0182), RGB(-0.5876, 1.1329, -0.1006), RGB(-0.0728, -0.0083, 1.1187))
        let srgbToRec2020 = (RGB(0.6274, 0.0691, 0.0164), RGB(0.3293, 0.9195, 0.0880), RGB(0.0433, 0.0113, 0.8956))
        let inset = (
            RGB(0.856627153315983, 0.137318972929847, 0.11189821299995),
            RGB(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
            RGB(0.0482516061458583, 0.101439036467562, 0.811302368396859)
        )
        let outset = (
            RGB(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
            RGB(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
            RGB(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405)
        )
        let minEV = -12.47393, maxEV = 4.026069
        var color = multiply(inset, multiply(srgbToRec2020, input))
        color = simd_max(color, RGB(repeating: 1e-10))
        color = RGB(log2(color.x), log2(color.y), log2(color.z))
        color = simd_clamp((color - minEV) / (maxEV - minEV), RGB(repeating: 0), RGB(repeating: 1))
        let x2 = color * color, x4 = x2 * x2
        color = 15.5 * x4 * x2 - 40.14 * x4 * color + 31.96 * x4 - 6.868 * x2 * color + 0.4298 * x2 + 0.1191 * color - 0.00232
        // sobotka/AgX config.ocio looks, applied as ASC CDL: slope, power, then saturation on Rec.709 luma.
        switch look {
        case .default: break
        case .punchy, .golden:
            let slope = look == .golden ? RGB(1, 0.9, 0.5) : RGB(1, 1, 1)
            let power = look == .golden ? 0.8 : 1.35
            let saturation = look == .golden ? 1.3 : 1.4
            color = pow(simd_max(color * slope, RGB(repeating: 0)), power)
            let luma = simd_dot(color, RGB(0.2126, 0.7152, 0.0722))
            color = luma + saturation * (color - luma)
        }
        color = pow(simd_max(RGB(repeating: 0), multiply(outset, color)), 2.2)
        return simd_clamp(multiply(rec2020ToSRGB, color), RGB(repeating: 0), RGB(repeating: 1))
    }

    static func neutral(_ input: RGB) -> RGB {
        let startCompression = 0.8 - 0.04, desaturation = 0.15
        var color = input
        let x = min(color.x, min(color.y, color.z))
        let offset = x < 0.08 ? x - 6.25 * x * x : 0.04
        color -= offset
        let peak = max(color.x, max(color.y, color.z))
        if peak < startCompression { return color }
        let d = 1 - startCompression
        let newPeak = 1 - d * d / (peak + d - startCompression)
        color *= newPeak / peak
        let g = 1 - 1 / (desaturation * (peak - newPeak) + 1)
        return color + (RGB(repeating: newPeak) - color) * g
    }

    // https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    static func acesNarkowicz(_ x: RGB) -> RGB {
        saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14))
    }

    // Reinhard, Stark, Shirley and Ferwerda 2002, equation 4, per channel.
    static func reinhardExtended(_ x: RGB, white: Double) -> RGB {
        saturate(x * (1 + x / (white * white)) / (1 + x))
    }

    /// Khronos PBR Neutral with its shoulder aimed at `peak` instead of 1.
    static func neutral(_ input: RGB, peak: Double) -> RGB {
        var color = input
        let x = min(color.x, min(color.y, color.z))
        color -= x < 0.08 ? x - 6.25 * x * x : 0.04
        let high = max(color.x, max(color.y, color.z))
        if high < 0.76 { return color }
        let d = peak - 0.76
        let newPeak = peak - d * d / (high + d - 0.76)
        color *= newPeak / high
        let g = 1 - 1 / (0.15 * (high - newPeak) + 1)
        return color + (RGB(repeating: newPeak) - color) * g
    }

    /// The operator's reference curve for an SDR display.
    static func sdr(_ operation: HDRDisplay.ToneMapping, _ c: RGB) -> RGB {
        switch operation {
        case .none: return simd_min(c, RGB(repeating: 65504))
        case .linear: return linear(c)
        case .reinhard: return reinhard(c)
        case .reinhardExtended(let white): return reinhardExtended(c, white: Double(white))
        case .cineon: return cineon(c)
        case .acesFilmic: return acesFilmic(c)
        case .acesNarkowicz: return acesNarkowicz(c)
        case .agx(let look): return agx(c, look: look)
        case .neutral: return neutral(c)
        }
    }

    /// The expected output of an operator in SDR, and with headroom `h` by the
    /// documented highlight extension or the case's own generalization.
    static func expected(_ operation: HDRDisplay.ToneMapping, _ c: RGB, headroom h: Double = 1) -> RGB {
        switch operation {
        case .none: return simd_min(c, RGB(repeating: 65504))
        case .linear: return simd_clamp(c, RGB(repeating: 0), RGB(repeating: h))
        case .neutral: return simd_clamp(neutral(c, peak: h), RGB(repeating: 0), RGB(repeating: h))
        default:
            let mapped = sdr(operation, c)
            guard h > 1 else { return mapped }
            // E = (H − 1) Δ² / ((Δ + H − w)(Δ + 1 − w)), Δ = w (m − 1), added to the
            // largest display channel with RGB scaled along.
            let w = simd_reduce_max(sdr(operation, RGB(repeating: 1)))
            let delta = w * max(simd_reduce_max(c) - 1, 0)
            let high = simd_reduce_max(mapped)
            guard delta > 0, high > 0 else { return mapped }
            let gain = (h - 1) * delta * delta / ((delta + h - w) * (delta + 1 - w))
            return simd_min(mapped * ((high + gain) / high), RGB(repeating: h))
        }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ToneMappingTests {
    private let fixture: PostProcessFixture

    init() throws {
        fixture = try PostProcessFixture()
    }

    /// Scene-linear test colors: a gray ramp across 13 stops and saturated colors.
    static let samples: [SIMD3<Float>] = [
        SIMD3(0, 0, 0), SIMD3(repeating: 0.002), SIMD3(repeating: 0.01), SIMD3(repeating: 0.05),
        SIMD3(repeating: 0.18), SIMD3(repeating: 0.5), SIMD3(repeating: 0.76), SIMD3(repeating: 1),
        SIMD3(repeating: 2), SIMD3(repeating: 4), SIMD3(repeating: 16), SIMD3(repeating: 120),
        SIMD3(4, 2, 1), SIMD3(0.9, 0.1, 0.05), SIMD3(0.02, 0.5, 3), SIMD3(10, 0.5, 0.1),
        SIMD3(0.05, 0.08, 0.6), SIMD3(1.5, 1.2, 0.02), SIMD3(0.3, 0.7, 0.25), SIMD3(60, 30, 2),
    ]

    static let operators: [HDRDisplay.ToneMapping] = [
        .none, .linear, .reinhard, .reinhardExtended(whitePoint: 4), .cineon, .acesFilmic, .acesNarkowicz,
        .agx(), .agx(look: .punchy), .agx(look: .golden), .neutral,
    ]

    @Test("Every operator matches its reference formula within 1e-3", arguments: operators)
    func referenceCurves(operation: HDRDisplay.ToneMapping) throws {
        let pixels = try render(Self.samples, settings: .init(toneMapping: operation))
        var largest = 0.0
        for (index, sample) in Self.samples.enumerated() {
            let expected = ToneReference.expected(operation, Self.halved(sample))
            let actual = Self.rgb(pixels[index])
            let error = simd_reduce_max(simd_abs(actual - expected))
            largest = max(largest, error)
            #expect(error <= 1e-3, "\(operation) at \(sample): GPU \(actual), reference \(expected)")
            #expect(pixels[index].w == 1)
        }
        print("[rasterlook] \(operation): largest error \(largest) over \(Self.samples.count) colors")
    }

    @Test("Exposure scales scene color before every curve")
    func exposure() throws {
        for operation in [HDRDisplay.ToneMapping.reinhard, .acesFilmic, .agx(), .cineon] {
            let pixels = try render(Self.samples, settings: .init(exposureStops: 1.5, toneMapping: operation))
            for (index, sample) in Self.samples.enumerated() {
                let exposed = Self.halved(sample) * Double(exp2f(1.5))
                let expected = ToneReference.expected(operation, exposed)
                let actual = Self.rgb(pixels[index])
                #expect(simd_reduce_max(simd_abs(actual - expected)) <= 1e-3, "\(operation) at \(sample)")
            }
        }
    }

    @Test("Known anchor values: ACES maps 1 to 0.8 and Reinhard's white point to display white")
    func anchors() throws {
        let unit = try render([SIMD3(repeating: 1)], settings: .init(toneMapping: .acesNarkowicz))
        // Narkowicz: "1 on input maps to ~0.8 on output".
        #expect(abs(unit[0].x - 0.8037) < 1e-3)
        let white = try render([SIMD3(repeating: 4), SIMD3(repeating: 8)], settings: .init(toneMapping: .reinhardExtended(whitePoint: 4)))
        #expect(abs(white[0].x - 1) < 1e-3)
        #expect(white[1].x == 1)
        let gray = try render([SIMD3(repeating: 0.18)], settings: .init(toneMapping: .agx()))
        // Three.js's AgX lifts middle gray to 0.2145 (evaluated from its shader chunk in double precision).
        #expect(abs(gray[0].x - 0.2145) < 1e-3)
    }

    @Test("Extended-range headroom keeps highlights above SDR white", arguments: operators)
    func headroom(operation: HDRDisplay.ToneMapping) throws {
        for headroom in [Float(2), 16] {
            let pixels = try render(Self.samples, settings: .init(toneMapping: operation, displayHeadroom: headroom))
            for (index, sample) in Self.samples.enumerated() {
                let expected = ToneReference.expected(operation, Self.halved(sample), headroom: Double(headroom))
                let actual = Self.rgb(pixels[index])
                // Half floats store values up to 16 with a relative step of 1/1,024.
                let tolerance = 1e-3 * max(1, simd_reduce_max(expected))
                #expect(
                    simd_reduce_max(simd_abs(actual - expected)) <= tolerance,
                    "\(operation) at \(sample), headroom \(headroom): \(actual) vs \(expected)"
                )
                if operation != .none { #expect(simd_reduce_max(actual) <= Double(headroom) * 1.0005) }
            }
            // Bright gray uses the extra range; in SDR it would stop at 1.
            let brightest = pixels[Self.samples.firstIndex(of: SIMD3(repeating: 120))!]
            if operation != .none { #expect(brightest.x > 1 + 0.75 * (headroom - 1), "\(operation): \(brightest.x)") }
        }
    }

    @Test("Headroom leaves shadows and mid-tones at their SDR values", arguments: operators)
    func headroomKeepsMidTones(operation: HDRDisplay.ToneMapping) throws {
        // Colors whose exposed channels are all at most 1, and Neutral's colors below its shoulder.
        let low = Self.samples.filter { simd_reduce_max($0) <= (operation == .neutral ? 0.76 : 1) }
        #expect(low.count >= 8)
        let sdr = try render(low, settings: .init(toneMapping: operation))
        for headroom in [Float(1.5), 4, 16, 64] {
            let pixels = try render(low, settings: .init(toneMapping: operation, displayHeadroom: headroom))
            #expect(pixels == sdr, "\(operation) at headroom \(headroom)")
        }
        // Middle gray in particular: scaling a curve as H · T(x / H) would darken it with headroom.
        let gray = try render([SIMD3(repeating: 0.18)], settings: .init(toneMapping: operation, displayHeadroom: 16))
        let reference = ToneReference.expected(operation, SIMD3(repeating: 0.18))
        #expect(abs(Double(gray[0].x) - reference.x) <= 1e-3, "\(operation): middle gray \(gray[0].x) vs \(reference.x)")
    }

    @Test("Highlights rise smoothly toward the headroom and keep the SDR curve's hue", arguments: operators)
    func highlightExtension(operation: HDRDisplay.ToneMapping) throws {
        let headroom: Float = 16
        // A gray ramp from 1/64 to 4,096 in quarter stops.
        let ramp = (0..<73).map { SIMD3<Float>(repeating: exp2(Float($0) / 4 - 6)) }
        let pixels = try render(ramp, settings: .init(toneMapping: operation, displayHeadroom: headroom))
        for index in 1..<ramp.count {
            #expect(pixels[index].x >= pixels[index - 1].x, "\(operation) decreases at \(ramp[index].x)")
        }
        if operation != .none {
            #expect(pixels.allSatisfy { $0.x <= headroom })
            #expect(pixels[72].x > 0.95 * headroom, "\(operation) reaches \(pixels[72].x) at 4,096")
        }
        guard operation != .none, operation != .linear, operation != .neutral else { return }
        // Above exposed white the extension scales RGB, so channel ratios match the SDR curve.
        for color in [SIMD3<Float>(10, 0.5, 0.1), SIMD3(0.02, 0.5, 3), SIMD3(60, 30, 2)] {
            let hdr = Self.rgb(try render([color], settings: .init(toneMapping: operation, displayHeadroom: headroom))[0])
            let sdr = ToneReference.sdr(operation, Self.halved(color))
            #expect(simd_reduce_max(hdr) > 1.5 * simd_reduce_max(sdr))
            let ratio = simd_abs(hdr / simd_reduce_max(hdr) - sdr / simd_reduce_max(sdr))
            #expect(simd_reduce_max(ratio) <= 2e-3, "\(operation) at \(color): \(hdr) vs \(sdr)")
        }
    }

    @Test("Extended Reinhard stays finite for any positive white point")
    func extendedReinhardWhitePoints() throws {
        let colors: [SIMD3<Float>] = [SIMD3(0, 0, 0), SIMD3(0, 0.5, 0), SIMD3(2, 0, 0.001), SIMD3(60000, 0, 1)]
        for white in [Float(1e-20), .leastNormalMagnitude, .leastNonzeroMagnitude] {
            let pixels = try render(colors, settings: .init(toneMapping: .reinhardExtended(whitePoint: white)))
            // Any nonzero channel is far above such a white point and clips; zero stays zero.
            #expect(Self.rgb(pixels[0]) == SIMD3(0, 0, 0), "white \(white)")
            #expect(Self.rgb(pixels[1]) == SIMD3(0, 1, 0), "white \(white)")
            #expect(Self.rgb(pixels[2]) == SIMD3(1, 0, 1), "white \(white)")
            #expect(Self.rgb(pixels[3]) == SIMD3(1, 0, 1), "white \(white)")
        }
        // The largest white point is plain Reinhard.
        let huge = try render(Self.samples, settings: .init(toneMapping: .reinhardExtended(whitePoint: .greatestFiniteMagnitude)))
        let plain = try render(Self.samples, settings: .init(toneMapping: .reinhard))
        #expect(huge == plain)
    }

    @Test("Premultiplied input is tone mapped straight and premultiplied again")
    func premultipliedAlpha() throws {
        let straight = SIMD3<Float>(3, 0.6, 0.2)
        let source = try fixture.color(width: 2, height: 1) { x, _ in
            x == 0 ? SIMD4(straight * 0.25, 0.25) : SIMD4(0, 0, 0, 0)
        }
        let target = try fixture.color(width: 2, height: 1)
        let display = try HDRDisplay(device: fixture.device, destinationPixelFormat: .rgba16Float)
        try fixture.run { try display.encode(source: source, destination: target, settings: .init(toneMapping: .acesFilmic), commandBuffer: $0) }
        let pixels = try fixture.read(target)
        let expected = ToneReference.acesFilmic(Self.halved(straight)) * 0.25
        #expect(simd_reduce_max(simd_abs(Self.rgb(pixels[0]) - expected)) <= 1e-3)
        #expect(pixels[0].w == 0.25)
        #expect(pixels[1] == .zero)
    }

    @Test("An sRGB destination stores the encoded curve within one code value")
    func srgbDestination() throws {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm_srgb, width: Self.samples.count, height: 1, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .renderTarget
        let target = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let source = try fixture.color(width: Self.samples.count, height: 1) { x, _ in SIMD4(Self.samples[x], 1) }
        let display = try HDRDisplay(device: fixture.device, destinationPixelFormat: .bgra8Unorm_srgb)
        try fixture.run { try display.encode(source: source, destination: target, settings: .init(toneMapping: .agx(look: .punchy)), commandBuffer: $0) }
        var bytes = [UInt8](repeating: 0, count: Self.samples.count * 4)
        bytes.withUnsafeMutableBytes { raw in
            target.getBytes(raw.baseAddress!, bytesPerRow: Self.samples.count * 4, from: MTLRegionMake2D(0, 0, Self.samples.count, 1), mipmapLevel: 0)
        }
        func encode(_ linear: Double) -> Double {
            linear <= 0.0031308 ? 12.92 * linear : 1.055 * Foundation.pow(linear, 1 / 2.4) - 0.055
        }
        for (index, sample) in Self.samples.enumerated() {
            let expected = ToneReference.agx(Self.halved(sample), look: .punchy)
            let stored = [bytes[index * 4 + 2], bytes[index * 4 + 1], bytes[index * 4]]
            for channel in 0..<3 {
                #expect(abs(Double(stored[channel]) - encode(expected[channel]) * 255) <= 1.01, "\(sample) channel \(channel)")
            }
        }
    }

    @Test("A post-process graph hook applies a custom curve with exposure and alpha handling")
    func customOperator() throws {
        // Reinhard written as a graph must match the built-in operator.
        let reinhard = try PostProcessGraph.toneMapping(exposureStops: 0.5) { p, color in
            Self.divide(p, color, color + 1)
        }
        let builtIn = try render(Self.samples, settings: .init(exposureStops: 0.5, toneMapping: .reinhard))
        let custom = try renderGraph(reinhard, Self.samples)
        for index in Self.samples.indices {
            #expect(simd_reduce_max(simd_abs(custom[index] - builtIn[index])) <= 1e-3, "\(Self.samples[index])")
        }

        // John Hable's Uncharted 2 curve, which Three.js does not ship, against its CPU formula.
        let hable = try PostProcessGraph.toneMapping { p, color in
            func partial(_ x: ShaderValue<SIMD3<Float>>) -> ShaderValue<SIMD3<Float>> {
                Self.divide(p, x * (x * 0.15 + 0.05) + 0.004, x * (x * 0.15 + 0.5) + 0.06) - 0.02 / 0.3
            }
            let whiteScale: Float = 1 / Self.hablePartial(11.2)
            return partial(color * 2) * whiteScale
        }
        let pixels = try renderGraph(hable, Self.samples)
        for (index, sample) in Self.samples.enumerated() {
            let x = Self.halved(sample)
            let expected = SIMD3<Double>(
                Double(Self.hablePartial(Float(x.x) * 2)), Double(Self.hablePartial(Float(x.y) * 2)),
                Double(Self.hablePartial(Float(x.z) * 2))
            ) / Double(Self.hablePartial(11.2))
            let actual = Self.rgb(pixels[index])
            // Above white the custom curve is not clamped; half floats store it to 1/1,024.
            #expect(simd_reduce_max(simd_abs(actual - expected)) <= 1.5e-3, "\(sample)")
        }

        // Transparent pixels stay transparent and translucent ones are premultiplied.
        let source = try fixture.color(width: 2, height: 1) { x, _ in x == 0 ? SIMD4(0.5, 0.5, 0.5, 0.5) : .zero }
        let target = try fixture.color(width: 2, height: 1)
        let pass = try PostProcessPass(graph: reinhard, device: fixture.device)
        try pass.uniforms.set("ExposureStops", to: 0)
        try fixture.run { try pass.encode(source: source, destination: target, commandBuffer: $0) }
        let translucent = try fixture.read(target)
        #expect(abs(translucent[0].x - 0.25) < 1e-3)
        #expect(translucent[0].w == 0.5)
        #expect(translucent[1] == .zero)
    }

    @Test("A material can keep its linear color when the display applies a tone curve")
    func materialToneMapped() throws {
        let scene = try GeometryRenderFixture(side: 32)
        func quad(x: Float, toneMapped: Bool) throws -> SceneNode {
            let material = try PhysicallyBasedMaterial(
                lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
                emissiveFactor: SIMD3(repeating: 1), alphaMode: .opaque, emissiveStrength: 4,
                toneMapped: toneMapped
            )
            let mesh = try MeshRenderer(
                device: scene.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.5, height: 0.5),
                material: .physicallyBased(material)
            )
            return SceneNode(transform: .init(translation: SIMD3(x, 0, -3)), mesh: mesh)
        }
        let graph = SceneGraph(
            roots: [try quad(x: -0.5, toneMapped: true), try quad(x: 0.5, toneMapped: false)],
            camera: scene.camera
        )
        let target = try scene.makeTarget()
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: scene.side, height: scene.side, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = [.renderTarget, .shaderRead]
        let color = try #require(scene.device.makeTexture(descriptor: colorDescriptor))
        let pass = scene.pass(color: color, depth: target.depth)
        pass.depthAttachment.storeAction = .store
        let maskDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm, width: scene.side, height: scene.side, mipmapped: false
        )
        maskDescriptor.usage = [.renderTarget, .shaderRead]
        maskDescriptor.storageMode = .private
        let mask = try #require(scene.device.makeTexture(descriptor: maskDescriptor))
        let displayed = try scene.makeTarget().color
        let display = try HDRDisplay(device: scene.device, destinationPixelFormat: .rgba16Float)
        let buffer = try #require(scene.queue.makeCommandBuffer())
        _ = try graph.encode(pass: pass, commandBuffer: buffer)
        try graph.encodeToneMapMask(depth: target.depth, into: mask, commandBuffer: buffer)
        try display.encode(
            source: color, destination: displayed, settings: .init(toneMapping: .reinhard),
            untonemappedMask: mask, commandBuffer: buffer
        )
        try scene.finish(buffer)
        let pixels = try scene.pixels(of: displayed)
        func pixel(_ x: Int) -> SIMD4<Float> { pixels[16 * scene.side + x] }
        let mapped = pixel(13)
        let raw = pixel(19)
        #expect(abs(mapped.x - 0.8) < 0.05, "\(mapped)")
        #expect(abs(raw.x - 4) < 0.05, "\(raw)")
    }

    @Test("Invalid headroom, white point and formats are rejected before encoding")
    func validation() throws {
        let source = try fixture.color(width: 1, height: 1) { _, _ in SIMD4(repeating: 1) }
        let target = try fixture.color(width: 1, height: 1)
        let display = try HDRDisplay(device: fixture.device, destinationPixelFormat: .rgba16Float)
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: HDRDisplay.Failure.invalidHeadroom) {
            try display.encode(source: source, destination: target, settings: .init(displayHeadroom: 0.5), commandBuffer: buffer)
        }
        #expect(throws: HDRDisplay.Failure.invalidHeadroom) {
            try display.encode(source: source, destination: target, settings: .init(displayHeadroom: .infinity), commandBuffer: buffer)
        }
        #expect(throws: HDRDisplay.Failure.invalidWhitePoint) {
            try display.encode(
                source: source, destination: target,
                settings: .init(toneMapping: .reinhardExtended(whitePoint: 0)), commandBuffer: buffer
            )
        }
        let srgb = try HDRDisplay(device: fixture.device, destinationPixelFormat: .bgra8Unorm_srgb)
        let srgbTarget = try fixture.color(width: 1, height: 1, pixelFormat: .bgra8Unorm_srgb)
        #expect(throws: HDRDisplay.Failure.headroomNeedsExtendedRange) {
            try srgb.encode(source: source, destination: srgbTarget, settings: .init(displayHeadroom: 2), commandBuffer: buffer)
        }
    }

    // MARK: - Helpers

    /// The value the GPU reads after the sample is stored as a half float.
    static func halved(_ value: SIMD3<Float>) -> SIMD3<Double> {
        SIMD3(Double(Float(Float16(value.x))), Double(Float(Float16(value.y))), Double(Float(Float16(value.z))))
    }

    static func rgb(_ pixel: SIMD4<Float>) -> SIMD3<Double> {
        SIMD3(Double(pixel.x), Double(pixel.y), Double(pixel.z))
    }

    /// Component-wise division; graph division takes a scalar divisor.
    static func divide(
        _ p: ShaderGraphBuilder, _ a: ShaderValue<SIMD3<Float>>, _ b: ShaderValue<SIMD3<Float>>
    ) -> ShaderValue<SIMD3<Float>> {
        p.vector(a.x / b.x, a.y / b.y, a.z / b.z)
    }

    static func hablePartial(_ x: Float) -> Float {
        (x * (x * 0.15 + 0.05) + 0.004) / (x * (x * 0.15 + 0.5) + 0.06) - 0.02 / 0.3
    }

    private func render(_ colors: [SIMD3<Float>], settings: HDRDisplay.Settings) throws -> [SIMD4<Float>] {
        let source = try fixture.color(width: colors.count, height: 1) { x, _ in SIMD4(colors[x], 1) }
        let target = try fixture.color(width: colors.count, height: 1)
        let display = try HDRDisplay(device: fixture.device, destinationPixelFormat: .rgba16Float)
        try fixture.run { try display.encode(source: source, destination: target, settings: settings, commandBuffer: $0) }
        return try fixture.read(target)
    }

    private func renderGraph(_ graph: PostProcessGraph, _ colors: [SIMD3<Float>]) throws -> [SIMD4<Float>] {
        let source = try fixture.color(width: colors.count, height: 1) { x, _ in SIMD4(colors[x], 1) }
        let target = try fixture.color(width: colors.count, height: 1)
        let pass = try PostProcessPass(graph: graph, device: fixture.device)
        try fixture.run { try pass.encode(source: source, destination: target, commandBuffer: $0) }
        return try fixture.read(target)
    }
}
