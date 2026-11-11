import CryptoKit
import CoreGraphics
import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct HDRDisplayDitheringTests {
    private typealias Reference = R187DitheringReference

    private let device: any MTLDevice
    private let queue: any MTLCommandQueue

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
        print("DITHER-ENV MTL_DEBUG_LAYER=\(ProcessInfo.processInfo.environment["MTL_DEBUG_LAYER"] ?? "unset") device=\(device.name)")
    }

    // MARK: - Fixture

    private func texture(
        _ format: MTLPixelFormat, width: Int, height: Int, usage: MTLTextureUsage
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = usage
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    private func source(width: Int, height: Int, pixel: (Int, Int) -> SIMD4<Float>) throws -> any MTLTexture {
        let source = try texture(.rgba16Float, width: width, height: height, usage: [.shaderRead, .renderTarget])
        var halves: [Float16] = []
        for y in 0..<height {
            for x in 0..<width {
                let rgba = pixel(x, y)
                halves += [Float16(rgba.x), Float16(rgba.y), Float16(rgba.z), Float16(rgba.w)]
            }
        }
        try halves.withUnsafeBytes { bytes in
            source.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: try #require(bytes.baseAddress), bytesPerRow: width * 8
            )
        }
        return source
    }

    /// Renders through `HDRDisplay` and returns the destination's bytes, row by row.
    private func render(
        _ source: any MTLTexture, format: MTLPixelFormat, space: HDRDisplay.OutputColorSpace = .sRGB,
        settings: HDRDisplay.Settings = .init(), width: Int? = nil, height: Int? = nil,
        encode: ((HDRDisplay, any MTLTexture, any MTLCommandBuffer) throws -> Void)? = nil
    ) throws -> [UInt8] {
        let width = width ?? source.width
        let height = height ?? source.height
        let target = try texture(format, width: width, height: height, usage: .renderTarget)
        let display = try HDRDisplay(device: device, destinationPixelFormat: format, outputColorSpace: space)
        let buffer = try #require(queue.makeCommandBuffer())
        if let encode {
            try encode(display, target, buffer)
        } else {
            try display.encode(source: source, destination: target, settings: settings, commandBuffer: buffer)
        }
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)
        return Self.bytes(of: target)
    }

    private static func bytes(of target: any MTLTexture) -> [UInt8] {
        let bytesPerPixel = target.pixelFormat == .rgba16Float ? 8 : 4
        var bytes = [UInt8](repeating: 0, count: target.width * target.height * bytesPerPixel)
        bytes.withUnsafeMutableBytes { raw in
            target.getBytes(
                raw.baseAddress!, bytesPerRow: target.width * bytesPerPixel,
                from: MTLRegionMake2D(0, 0, target.width, target.height), mipmapLevel: 0
            )
        }
        return bytes
    }

    private static func hex(_ bytes: [UInt8]) -> String {
        SHA256.hash(data: Data(bytes)).map { String(format: "%02x", $0) }.joined()
    }

    /// The byte offsets of red, green and blue within a pixel of an 8-bit destination.
    private static func channelOrder(_ format: MTLPixelFormat) -> [Int] {
        switch format {
        case .bgra8Unorm, .bgra8Unorm_srgb: [2, 1, 0]
        default: [0, 1, 2]
        }
    }

    private static func isSRGB(_ format: MTLPixelFormat) -> Bool {
        format == .bgra8Unorm_srgb || format == .rgba8Unorm_srgb
    }

    /// Metal's conversion of a color in 0...1 to an 8-bit unorm code.
    private static func code(_ value: Double) -> Int { Int((255 * min(max(value, 0), 1)).rounded(.toNearestOrEven)) }

    // MARK: - Dithering off leaves the output unchanged

    /// A 64 x 32 premultiplied HDR frame with gradients, an HDR highlight, three alpha bands and
    /// a diagonal color edge for the anti-aliasing path.
    private static func goldenPixel(_ x: Int, _ y: Int) -> SIMD4<Float> {
        let alpha: Float = x < 8 ? 0 : (x < 16 ? 0.5 : 1)
        var rgb = SIMD3<Float>(Float(x) / 63 * 2, Float(y) / 31 * 1.5, Float((x ^ y) & 15) / 15 * 0.8)
        if x == 40, y == 10 { rgb = SIMD3(4, 2, 1) }
        if x >= 24, x - 24 <= y { rgb = SIMD3(1.2, 0.2, 0.2) }
        return SIMD4(rgb * alpha, alpha)
    }

    private struct GoldenCase {
        var name: String
        var format: MTLPixelFormat
        var space: HDRDisplay.OutputColorSpace = .sRGB
        var settings: HDRDisplay.Settings = .init()
    }

    private static let goldenCases: [GoldenCase] = [
        GoldenCase(name: "float-neutral", format: .rgba16Float),
        GoldenCase(name: "float-none", format: .rgba16Float, settings: .init(toneMapping: .none)),
        GoldenCase(name: "float-aces-plus1", format: .rgba16Float, settings: .init(exposureStops: 1, toneMapping: .acesFilmic)),
        GoldenCase(name: "float-agx-punchy", format: .rgba16Float, settings: .init(toneMapping: .agx(look: .punchy))),
        GoldenCase(name: "float-reinhard-extended", format: .rgba16Float, settings: .init(toneMapping: .reinhardExtended(whitePoint: 2.5))),
        GoldenCase(name: "float-neutral-headroom4", format: .rgba16Float, settings: .init(displayHeadroom: 4)),
        GoldenCase(name: "float-p3-none", format: .rgba16Float, space: .displayP3, settings: .init(toneMapping: .none)),
        GoldenCase(name: "float-neutral-aa", format: .rgba16Float, settings: .init(antiAliasing: true)),
        GoldenCase(name: "srgb8-neutral", format: .bgra8Unorm_srgb),
        GoldenCase(name: "srgb8-aces-aa", format: .rgba8Unorm_srgb, settings: .init(antiAliasing: true, toneMapping: .acesFilmic)),
        GoldenCase(name: "srgb8-p3-neutral", format: .bgra8Unorm_srgb, space: .displayP3),
        GoldenCase(name: "linear8-neutral", format: .rgba8Unorm),
        GoldenCase(name: "linear8-linear-curve", format: .bgra8Unorm, settings: .init(toneMapping: .linear)),
    ]

    /// SHA-256 digests of the destination bytes, rendered by the display pass before dithering
    /// existed (the integrated tip 8f473f7) on the Apple M4 Max, from the same frame and settings.
    /// The iPad Pro (M5) simulator rendered the same bytes.
    private static let goldenDigests: [String: String] = [
        "float-neutral": "70286aee84a38f877bff4db5656f46f31f83f8851ff2af1f67e4013016f6b643",
        "float-none": "3d989f1c20b520be986aed1944a27f40746fed96c0dfba8b5b49da0f2a6b022c",
        "float-aces-plus1": "5b255ae081a14a2c25d44410388a01e9bd8fdb8f4fe9c88b9de90ee2a52ce0c1",
        "float-agx-punchy": "24d21051001ed0c0559e31f21ff1156c7e02e86ecbe1523d2f58d4115a134b82",
        "float-reinhard-extended": "a850dded014df4cf361a0cb2834c5e7b4e51313f9a12ffe1225e9cfe691979e9",
        "float-neutral-headroom4": "b082c8d9a1a04a24dd10444fe42452ccac3596f6e7b4952bb2973686391478dd",
        "float-p3-none": "93d4c9f465114e17a6b1276fa0c59449c6e5823c4c5262364bb9f57b3cd67f47",
        "float-neutral-aa": "57846bdc16cfea1f329e11db8bf35758a163e605e7edbd945a126c88194a676e",
        "srgb8-neutral": "912437baade99416b44a71cc56406a45deeaa6787e3ec95b2a4b262d6225a9ee",
        "srgb8-aces-aa": "0a61e50ecf2199121e16a38a3a80e76c0e08fe1ff0688d6a5868f1646875b7e1",
        "srgb8-p3-neutral": "fb37253487f1b7a2349e8a699cd3ebfc7e37551311a9271d23c8ec7f61992942",
        "linear8-neutral": "a21281ae49a3208d114ccaac1bf4fc6b15c635dc2b587e7b944f5ae648a688cb",
        "linear8-linear-curve": "6a86da34dde3078fb5d0abd73b706ab54152e8508a1793fa55000f91e2c74d1c",
        "srgb8-scaled": "bb35acc8641af9c8d300bc2e9310a4b68b512b0d5e3d22a15cee4333414f6e72",
        "srgb8-crossfade": "8c7ddee5dfa366ad6039e809cc7f13e83f77994ad868527d2a046b5939c19c0a",
        "srgb8-chain": "912437baade99416b44a71cc56406a45deeaa6787e3ec95b2a4b262d6225a9ee",
    ]

    @Test func `with dithering off the display pass writes the bytes it wrote before the option existed`() throws {
        #expect(HDRDisplay.Settings().dithering == false)
        #expect(HDRDisplay.Settings(dithering: false) == HDRDisplay.Settings())
        let frame = try source(width: 64, height: 32, pixel: Self.goldenPixel)
        var digests: [String: String] = [:]
        for item in Self.goldenCases {
            digests[item.name] = Self.hex(try render(frame, format: item.format, space: item.space, settings: item.settings))
            var explicit = item.settings
            explicit.dithering = false
            #expect(
                Self.hex(try render(frame, format: item.format, space: item.space, settings: explicit)) == digests[item.name],
                "an explicit false must equal the default for \(item.name)"
            )
        }
        // The package-level scaled and crossfade entry points share the shader.
        let small = try source(width: 32, height: 16) { x, y in Self.goldenPixel(x * 2, y * 2) }
        digests["srgb8-scaled"] = Self.hex(try render(small, format: .bgra8Unorm_srgb, width: 64, height: 32) { display, target, buffer in
            try display.encodeScaled(source: small, destination: target, settings: .init(), commandBuffer: buffer)
        })
        digests["srgb8-crossfade"] = Self.hex(try render(frame, format: .rgba8Unorm_srgb) { display, target, buffer in
            try display.encodeCrossfade(from: small, to: frame, weight: 0.35, destination: target, settings: .init(toneMapping: .acesFilmic), commandBuffer: buffer)
        })
        // A chain's final tone-mapping stage is the same pass.
        let chain = try PostProcessChain(queue: queue, destinationPixelFormat: .bgra8Unorm_srgb, effects: [.toneMapping(.init())])
        digests["srgb8-chain"] = Self.hex(try render(frame, format: .bgra8Unorm_srgb) { _, target, buffer in
            try chain.encode(source: frame, destination: target, commandBuffer: buffer)
        })
        #expect(Set(digests.keys) == Set(Self.goldenDigests.keys))
        for (name, digest) in digests.sorted(by: { $0.key < $1.key }) {
            #expect(digest == Self.goldenDigests[name], "\(name) changed")
        }
    }

    // MARK: - Dithering on

    @Test("An 8-bit target without a transfer function stores the r187 shift, code for code",
          arguments: [MTLPixelFormat.rgba8Unorm, .bgra8Unorm])
    func linearTargetMatchesReference(format: MTLPixelFormat) throws {
        let width = 128
        let height = 64
        // Every channel sits half a code above a whole code, so the sign of the shift decides
        // the stored code at nearly every pixel and a wrong shift cannot hide.
        func value(_ x: Int, _ y: Int) -> SIMD3<Float> {
            let red = Float(60 + x % 120) + 0.5
            let green = Float(40 + 2 * y) + 0.5
            return SIMD3<Float>(red / 255, green / 255, 0.5)
        }
        let frame = try source(width: width, height: height) { x, y in SIMD4<Float>(value(x, y), 1) }
        let settings = HDRDisplay.Settings(toneMapping: .none, dithering: true)
        let off = try render(frame, format: format, settings: .init(toneMapping: .none))
        let on = try render(frame, format: format, settings: settings)
        #expect(on == (try render(frame, format: format, settings: settings)), "the shift depends only on the pixel")
        let order = Self.channelOrder(format)
        var exact = 0
        var withinOne = 0
        var wrongOriginAgreement = 0
        var changed = 0
        var alphaChanged = 0
        for y in 0..<height {
            for x in 0..<width {
                let shift = Reference.mirrorShift(column: x, row: y, height: height)
                let unflipped = Reference.mirrorShift(column: x, row: y, height: height, flipsRows: false)
                for channel in 0..<3 {
                    let stored = Double(Float(Float16(value(x, y)[channel])))
                    let actual = Int(on[(y * width + x) * 4 + order[channel]])
                    let expected = Self.code(stored + shift[channel])
                    if actual == expected { exact += 1 }
                    if abs(actual - expected) <= 1 { withinOne += 1 }
                    if actual == Self.code(stored + unflipped[channel]) { wrongOriginAgreement += 1 }
                    if actual != Int(off[(y * width + x) * 4 + order[channel]]) { changed += 1 }
                }
                if on[(y * width + x) * 4 + 3] != off[(y * width + x) * 4 + 3] { alphaChanged += 1 }
            }
        }
        let total = Double(width * height * 3)
        print("DITHER-E2E linear \(format.rawValue): code equal to the float32-mirror reference \(Double(exact) / total), within one code \(Double(withinOne) / total), equal to a top-left-origin reference \(Double(wrongOriginAgreement) / total), codes changed by dithering \(Double(changed) / total), alpha bytes changed \(alphaChanged)")
        #expect(withinOne == width * height * 3)
        #expect(Double(exact) / total >= 0.98)
        // A reference that counts rows from the top instead of from gl_FragCoord's bottom-left
        // origin must not fit: two independent shifts agree on a half-code color about half the time.
        #expect(Double(wrongOriginAgreement) / total < 0.65)
        #expect(Double(changed) / total > 0.4)
        #expect(alphaChanged == 0)
    }

    @Test("An sRGB target shifts the encoded code by at most half a step at every brightness",
          arguments: [MTLPixelFormat.bgra8Unorm_srgb, .rgba8Unorm_srgb])
    func srgbTargetShiftsEncodedValue(format: MTLPixelFormat) throws {
        let width = 256
        let height = 32
        // Encoded values from 0.002, where a linear-space shift of 0.5 / 255 would be about
        // 6 codes, to 0.952.
        func linear(_ x: Int) -> Float { Float(Reference.srgbDecode(0.002 + 0.95 * Double(x) / Double(width - 1))) }
        let frame = try source(width: width, height: height) { x, _ in SIMD4<Float>(linear(x), linear(x), linear(x), 1) }
        let off = try render(frame, format: format, settings: .init(toneMapping: .none))
        let on = try render(frame, format: format, settings: .init(toneMapping: .none, dithering: true))
        let order = Self.channelOrder(format)
        var exact = 0
        var exactUndithered = 0
        var withinOne = 0
        var shiftedByMoreThanOne = 0
        var darkMismatches = 0
        var darkTotal = 0
        for y in 0..<height {
            for x in 0..<width {
                let shift = Reference.mirrorShift(column: x, row: y, height: height)
                let encoded = Reference.srgbEncode(Double(Float(Float16(linear(x)))))
                for channel in 0..<3 {
                    let actual = Int(on[(y * width + x) * 4 + order[channel]])
                    let undithered = Int(off[(y * width + x) * 4 + order[channel]])
                    let expected = Self.code(encoded + shift[channel])
                    if actual == expected { exact += 1 }
                    if undithered == Self.code(encoded) { exactUndithered += 1 }
                    if abs(actual - expected) <= 1 { withinOne += 1 }
                    if abs(actual - undithered) > 1 { shiftedByMoreThanOne += 1 }
                    if x < 32 {
                        darkTotal += 1
                        if actual != expected { darkMismatches += 1 }
                    }
                }
            }
        }
        let total = Double(width * height * 3)
        print("DITHER-E2E srgb \(format.rawValue): code equal to the encoded-space reference \(Double(exact) / total) (undithered code equal to its own reference \(Double(exactUndithered) / total)), within one code \(Double(withinOne) / total), pixels moved by more than one code from the undithered code \(shiftedByMoreThanOne), mismatches in the darkest 32 columns \(darkMismatches) of \(darkTotal)")
        #expect(withinOne == width * height * 3)
        #expect(shiftedByMoreThanOne == 0)
        // Metal's sRGB store places some code boundaries away from the encoded midpoints (see the
        // measurement below), so even the undithered code differs from its reference at some
        // pixels. Dithering must not make agreement worse than that baseline by more than a little.
        #expect(Double(exactUndithered) / total >= 0.9)
        #expect(Double(exact) / total >= Double(exactUndithered) / total - 0.02)
    }

    /// Measures where Metal's sRGB store switches from one code to the next, against the
    /// midpoints between codes in the encoded domain, to separate the hardware's own
    /// quantizer from the dither.
    @Test func `measures the sRGB store's code boundaries against the encoded midpoints`() throws {
        let width = 8192
        func value(_ x: Int) -> Float { Float(x) / Float(width - 1) * 0.02 }
        let frame = try source(width: width, height: 1) { x, _ in SIMD4<Float>(value(x), value(x), value(x), 1) }
        let bytes = try render(frame, format: .rgba8Unorm_srgb, settings: .init(toneMapping: .none))
        var previous = Int(bytes[0])
        var deviations: [Int: Double] = [:]
        for x in 1..<width {
            let code = Int(bytes[x * 4])
            if code != previous {
                if code == previous + 1 {
                    let low = Double(Float(Float16(value(x - 1))))
                    let high = Double(Float(Float16(value(x))))
                    deviations[previous] = 255 * Reference.srgbEncode((low + high) / 2) - (Double(previous) + 0.5)
                }
                previous = code
            }
        }
        let worst = deviations.values.map { abs($0) }.max() ?? 0
        let early = (0..<12).map { String(format: "%.3f", deviations[$0] ?? .nan) }.joined(separator: " ")
        print("DITHER-HARDWARE sRGB store: \(deviations.count) code boundaries found, worst deviation of a boundary from the encoded midpoint \(worst) codes; boundaries 0 to 11: \(early)")
        #expect(deviations.count >= 30)
        // Measured 0.345 codes on the Apple M4 Max: the store's codes 0 to 38 are not evenly spaced
        // in the encoded domain. The bound flags a different quantizer.
        #expect(worst < 0.4)
    }

    // MARK: - Banding

    private struct BandingResult {
        var undithered: Double
        var dithered: Double
        var unditheredMean: Double
        var ditheredMean: Double
    }

    /// Renders a 256 x 64 ramp of four 8-bit codes (64 pixels per code) with and without
    /// dithering and compares the 16 x 16 window averages of the stored codes with the average
    /// of the source's true values, in codes. `firstEncoded` is the ramp's encoded start.
    private func banding(format: MTLPixelFormat, firstEncoded: Double) throws -> BandingResult {
        let width = 256
        let height = 64
        let srgb = Self.isSRGB(format)
        func linear(_ x: Int) -> Float {
            let encoded = firstEncoded + 4.0 / 255 * Double(x) / Double(width - 1)
            return Float(srgb ? Reference.srgbDecode(encoded) : encoded)
        }
        func truth(_ x: Int) -> Double {
            let value = Double(Float(Float16(linear(x))))
            return 255 * (srgb ? Reference.srgbEncode(value) : value)
        }
        let frame = try source(width: width, height: height) { x, _ in SIMD4(linear(x), linear(x), linear(x), 1) }
        let off = try render(frame, format: format, settings: .init(toneMapping: .none))
        let on = try render(frame, format: format, settings: .init(toneMapping: .none, dithering: true))
        let order = Self.channelOrder(format)
        func statistics(_ bytes: [UInt8]) -> (rms: Double, mean: Double) {
            var squares = 0.0
            var sum = 0.0
            var count = 0
            for windowY in Swift.stride(from: 0, to: height, by: 16) {
                for windowX in Swift.stride(from: 0, to: width, by: 16) {
                    for channel in order {
                        var stored = 0.0
                        var expected = 0.0
                        for y in windowY..<(windowY + 16) {
                            for x in windowX..<(windowX + 16) {
                                stored += Double(bytes[(y * width + x) * 4 + channel])
                                expected += truth(x)
                            }
                        }
                        let error = (stored - expected) / 256
                        squares += error * error
                        sum += error
                        count += 1
                    }
                }
            }
            return ((squares / Double(count)).squareRoot(), sum / Double(count))
        }
        let (offRMS, offMean) = statistics(off)
        let (onRMS, onMean) = statistics(on)
        return BandingResult(undithered: offRMS, dithered: onRMS, unditheredMean: offMean, ditheredMean: onMean)
    }

    @Test("A dithered gradient's local average tracks the true value more closely than the quantized gradient",
          arguments: [
              (MTLPixelFormat.bgra8Unorm, 0.4, "8-bit linear, codes 102 to 106"),
              (MTLPixelFormat.bgra8Unorm_srgb, 0.4, "sRGB, encoded codes 102 to 106"),
              (MTLPixelFormat.rgba8Unorm, 0.02, "8-bit linear, dark ramp, codes 5.1 to 9.1"),
          ])
    func ditheringReducesBanding(format: MTLPixelFormat, start: Double, label: String) throws {
        let result = try banding(format: format, firstEncoded: start)
        print("DITHER-BANDING \(label): RMS error of 16x16 window averages, in 8-bit codes: undithered \(result.undithered), dithered \(result.dithered), ratio \(result.dithered / result.undithered); mean error undithered \(result.unditheredMean), dithered \(result.ditheredMean)")
        // Quantizing a ramp to whole codes leaves a sawtooth of up to 0.29 codes RMS. A dither
        // that moves the quantizer's decision by a uniform half step makes the expected
        // stored code the true value, leaving noise of about 0.4 / 16 codes in a 256-pixel window.
        #expect(result.undithered > 0.15)
        #expect(result.dithered < 0.1)
        #expect(result.dithered < result.undithered / 2)
        #expect(abs(result.ditheredMean) < 0.02)
    }

    @Test func `a dark sRGB ramp improves less because the hardware store is uneven there`() throws {
        let result = try banding(format: .rgba8Unorm_srgb, firstEncoded: 0.02)
        print("DITHER-BANDING sRGB, dark ramp, encoded codes 5.1 to 9.1: RMS error of 16x16 window averages, in 8-bit codes: undithered \(result.undithered), dithered \(result.dithered), ratio \(result.dithered / result.undithered); mean error undithered \(result.unditheredMean), dithered \(result.ditheredMean)")
        // Metal's sRGB store puts its code boundaries up to 0.345 codes from the encoded midpoints
        // below code 39, so the average of dithered stores follows the store's own curve: the
        // remaining error is bounded by that deviation and still below the quantized ramp's.
        #expect(result.dithered < result.undithered * 0.85)
        #expect(result.dithered < 0.345)
    }

    // MARK: - Alpha, anti-aliasing, destinations

    @Test func `a fully transparent pixel stays zero and alpha is never shifted`() throws {
        let frame = try source(width: 64, height: 32, pixel: Self.goldenPixel)
        let format = MTLPixelFormat.rgba8Unorm
        let off = try render(frame, format: format, settings: .init(toneMapping: .none))
        let on = try render(frame, format: format, settings: .init(toneMapping: .none, dithering: true))
        var transparentChanged = 0
        var alphaChanged = 0
        var translucentWithinReference = 0
        var translucent = 0
        for y in 0..<32 {
            for x in 0..<64 {
                let base = (y * 64 + x) * 4
                if on[base + 3] != off[base + 3] { alphaChanged += 1 }
                if x < 8, on[base..<base + 4].contains(where: { $0 != 0 }) { transparentChanged += 1 }
                if (8..<16).contains(x) {
                    let shift = Reference.mirrorShift(column: x, row: y, height: 32)
                    let pixel = Self.goldenPixel(x, y)
                    for channel in 0..<3 {
                        translucent += 1
                        let expected = Self.code(Double(Float(Float16(pixel[channel]))) + shift[channel])
                        if abs(Int(on[base + channel]) - expected) <= 1 { translucentWithinReference += 1 }
                    }
                }
            }
        }
        #expect(transparentChanged == 0)
        #expect(alphaChanged == 0)
        #expect(translucent > 0 && translucentWithinReference == translucent,
                "a half-transparent pixel shifts its premultiplied color like any other")
    }

    @Test func `anti-aliasing and dithering combine: dithering moves each code by at most one`() throws {
        let frame = try source(width: 64, height: 32, pixel: Self.goldenPixel)
        for format in [MTLPixelFormat.bgra8Unorm_srgb, .rgba8Unorm] {
            let off = try render(frame, format: format, settings: .init(antiAliasing: true, toneMapping: .acesFilmic))
            let on = try render(frame, format: format, settings: .init(antiAliasing: true, toneMapping: .acesFilmic, dithering: true))
            var worst = 0
            var changed = 0
            for index in off.indices where abs(Int(on[index]) - Int(off[index])) != 0 {
                changed += 1
                worst = max(worst, abs(Int(on[index]) - Int(off[index])))
            }
            #expect(changed > 0)
            #expect(worst <= 1)
            var alphaChanged = 0
            for pixel in 0..<(64 * 32) where on[pixel * 4 + 3] != off[pixel * 4 + 3] { alphaChanged += 1 }
            #expect(alphaChanged == 0)
        }
    }

    @Test func `dithering needs an 8-bit destination written directly`() throws {
        let frame = try source(width: 8, height: 8) { _, _ in SIMD4(0.5, 0.5, 0.5, 1) }
        let dithered = HDRDisplay.Settings(dithering: true)
        #expect(throws: HDRDisplay.Failure.ditheringNeedsEightBitDestination) {
            try render(frame, format: .rgba16Float, settings: dithered)
        }
        for format in [MTLPixelFormat.rgba8Unorm, .bgra8Unorm, .rgba8Unorm_srgb, .bgra8Unorm_srgb] {
            _ = try render(frame, format: format, settings: dithered)
        }

        // In a chain the stage must be the last one and write the chain's 8-bit destination.
        #expect(throws: HDRDisplay.Failure.ditheringNeedsEightBitDestination) {
            try PostProcessChain(queue: queue, destinationPixelFormat: .rgba16Float, effects: [.toneMapping(dithered)])
        }
        let notLast = try PostProcessChain(
            queue: queue, destinationPixelFormat: .bgra8Unorm_srgb, effects: [.toneMapping(dithered), .colorGrade(strength: 0.5)]
        )
        let target = try texture(.bgra8Unorm_srgb, width: 8, height: 8, usage: .renderTarget)
        let buffer = try #require(queue.makeCommandBuffer())
        #expect(throws: HDRDisplay.Failure.ditheringNeedsEightBitDestination) {
            try notLast.encode(source: frame, destination: target, commandBuffer: buffer)
        }

        let last = try PostProcessChain(queue: queue, destinationPixelFormat: .bgra8Unorm_srgb, effects: [.toneMapping(dithered)])
        let direct = try render(frame, format: .bgra8Unorm_srgb, settings: dithered)
        let chained = try render(frame, format: .bgra8Unorm_srgb) { _, target, buffer in
            try last.encode(source: frame, destination: target, commandBuffer: buffer)
        }
        #expect(chained == direct, "a final tone-mapping stage dithers exactly like the display pass")
        #expect(chained != (try render(frame, format: .bgra8Unorm_srgb)), "and the dither is visible at 0.5 gray")

        // A scaled or masked dithered stage writes an RGBA16Float intermediate instead of the destination.
        let scaled = try PostProcessChain(queue: queue, destinationPixelFormat: .bgra8Unorm_srgb, effects: [.toneMapping(dithered)])
        try scaled.setResolutionScale(0.5, at: 0)
        let scaledBuffer = try #require(queue.makeCommandBuffer())
        #expect(throws: HDRDisplay.Failure.ditheringNeedsEightBitDestination) {
            try scaled.encode(source: frame, destination: target, commandBuffer: scaledBuffer)
        }
        let masked = try PostProcessChain(queue: queue, destinationPixelFormat: .bgra8Unorm_srgb, effects: [.toneMapping(dithered)])
        let maskTexture = try texture(.r8Unorm, width: 8, height: 8, usage: .shaderRead)
        try masked.setMask(.init(texture: maskTexture), stages: 0..<1)
        let maskedBuffer = try #require(queue.makeCommandBuffer())
        #expect(throws: HDRDisplay.Failure.ditheringNeedsEightBitDestination) {
            try masked.encode(source: frame, destination: target, commandBuffer: maskedBuffer)
        }
    }

    @Test func `a reduced scene capture cannot keep a dither through its resample`() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let scene = SceneGraph(roots: [], camera: camera)
        let chain = try PostProcessChain(
            queue: queue, destinationPixelFormat: .bgra8Unorm_srgb, effects: [.toneMapping(.init(dithering: true))]
        )
        let target = try texture(.bgra8Unorm_srgb, width: 16, height: 16, usage: .renderTarget)

        let reduced = PostProcessScenePass(queue: queue)
        try reduced.setResolutionScale(0.5)
        let rejected = try #require(queue.makeCommandBuffer())
        #expect(throws: HDRDisplay.Failure.ditheringNeedsEightBitDestination) {
            try chain.encode(scene: scene, using: reduced, destination: target, commandBuffer: rejected)
        }

        // At full size the capture is not resampled, so the same chain encodes.
        let full = PostProcessScenePass(queue: queue)
        let accepted = try #require(queue.makeCommandBuffer())
        _ = try chain.encode(scene: scene, using: full, destination: target, commandBuffer: accepted)
        accepted.commit()
        accepted.waitUntilCompleted()
        #expect(accepted.error == nil)
    }
}
