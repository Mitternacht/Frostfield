import Foundation
import Frostfield
import Metal
import simd
import Testing

struct TextureUtilitiesTests {
    @Test("Centered fit maps wide, tall and matching images into the expected UV ranges")
    func fitTransforms() throws {
        let coverWide = try TextureUtilities.fit(imageAspectRatio: 2, surfaceAspectRatio: 1, mode: .cover)
        #expect(coverWide.scale == SIMD2(0.5, 1) && coverWide.offset == SIMD2(0.25, 0))
        #expect(coverWide.apply(to: .zero) == SIMD2(0.25, 0))
        #expect(coverWide.apply(to: SIMD2(repeating: 1)) == SIMD2(0.75, 1))
        let containWide = try TextureUtilities.fit(imageAspectRatio: 2, surfaceAspectRatio: 1, mode: .contain)
        #expect(containWide.scale == SIMD2(1, 2) && containWide.offset == SIMD2(0, -0.5))
        #expect(containWide.apply(to: SIMD2(0.5, 0.25)) == SIMD2(0.5, 0))
        #expect(containWide.apply(to: SIMD2(0.5, 0.75)) == SIMD2(0.5, 1))

        let coverTall = try TextureUtilities.fit(imageAspectRatio: 0.5, surfaceAspectRatio: 1, mode: .cover)
        #expect(coverTall.scale == SIMD2(1, 0.5) && coverTall.offset == SIMD2(0, 0.25))
        let containTall = try TextureUtilities.fit(imageAspectRatio: 0.5, surfaceAspectRatio: 1, mode: .contain)
        #expect(containTall.scale == SIMD2(2, 1) && containTall.offset == SIMD2(-0.5, 0))
        #expect(try TextureUtilities.fit(imageAspectRatio: 2, surfaceAspectRatio: 2, mode: .cover) == .identity)
        #expect(try TextureUtilities.fit(imageAspectRatio: 2, surfaceAspectRatio: 2, mode: .contain) == .identity)
        #expect(try TextureUtilities.fit(imageAspectRatio: 2, surfaceAspectRatio: 0.5, mode: .fill) == .identity)
    }

    @Test("Fitting rejects invalid dimensions and unrepresentable native scales")
    func invalidFit() {
        for aspect: Double in [0, -.infinity, .infinity, .nan, -1] {
            #expect(throws: TextureUtilities.Failure.aspectRatio) {
                try TextureUtilities.fit(imageAspectRatio: aspect, surfaceAspectRatio: 1, mode: .cover)
            }
            #expect(throws: TextureUtilities.Failure.aspectRatio) {
                try TextureUtilities.fit(imageAspectRatio: 1, surfaceAspectRatio: aspect, mode: .contain)
            }
        }
        #expect(throws: TextureUtilities.Failure.aspectRatio) {
            try TextureUtilities.fit(imageAspectRatio: 1e300, surfaceAspectRatio: 1, mode: .cover)
        }
        #expect(throws: TextureUtilities.Failure.aspectRatio) {
            try TextureUtilities.fit(imageAspectRatio: 1e300, surfaceAspectRatio: 1, mode: .contain)
        }
    }

    @Test("Tightly packed sizes include whole edge blocks and native packed format widths")
    func packedByteCounts() throws {
        let layouts: [(MTLPixelFormat, Int, Int, Int)] = [
            (.a8Unorm, 3, 5, 15), (.rgba8Uint, 3, 5, 60), (.rgb9e5Float, 3, 5, 60),
            (.rgba16Float, 3, 5, 120), (.rgba32Float, 3, 5, 240), (.depth32Float, 3, 5, 60),
            (.bgr10_xr, 3, 5, 60), (.bgra10_xr_srgb, 3, 5, 120),
            (.bc1_rgba, 7, 9, 48), (.bc3_rgba_srgb, 7, 9, 96), (.bc4_rSnorm, 7, 9, 48),
            (.bc5_rgUnorm, 7, 9, 96), (.bc6H_rgbFloat, 7, 9, 96), (.bc7_rgbaUnorm, 1, 1, 16),
            (.etc2_rgb8a1, 7, 9, 48), (.eac_r11Snorm, 7, 9, 48), (.eac_rg11Unorm, 7, 9, 96),
            (.astc_5x4_ldr, 11, 7, 96), (.astc_10x6_srgb, 11, 7, 64), (.astc_12x10_hdr, 11, 17, 32),
            (.pvrtc_rgba_2bpp, 1, 1, 32), (.pvrtc_rgb_2bpp_srgb, 32, 16, 128),
            (.pvrtc_rgb_4bpp, 4, 4, 32), (.pvrtc_rgba_4bpp_srgb, 16, 32, 256),
        ]
        for (format, width, height, bytes) in layouts {
            #expect(try TextureUtilities.byteCount(width: width, height: height, pixelFormat: format) == bytes,
                    "format \(format.rawValue), \(width)×\(height)")
        }
    }

    /// Block width, block height and bytes per block from Apple's pixel format tables and the Khronos
    /// formats, written out independently of the package's layout tables. PVRTC1 is checked separately.
    private static let documentedLayouts: [MTLPixelFormat: (width: Int, height: Int, bytes: Int)] = {
        var layouts: [MTLPixelFormat: (width: Int, height: Int, bytes: Int)] = [:]
        func add(_ formats: [MTLPixelFormat], _ width: Int, _ height: Int, _ bytes: Int) {
            for format in formats { layouts[format] = (width, height, bytes) }
        }
        add([.a8Unorm, .r8Unorm, .r8Unorm_srgb, .r8Snorm, .r8Uint, .r8Sint, .stencil8], 1, 1, 1)
        add([
            .r16Unorm, .r16Snorm, .r16Uint, .r16Sint, .r16Float, .rg8Unorm, .rg8Unorm_srgb, .rg8Snorm, .rg8Uint,
            .rg8Sint, .b5g6r5Unorm, .a1bgr5Unorm, .abgr4Unorm, .bgr5A1Unorm, .depth16Unorm,
        ], 1, 1, 2)
        add([
            .r32Uint, .r32Sint, .r32Float, .rg16Unorm, .rg16Snorm, .rg16Uint, .rg16Sint, .rg16Float, .rgba8Unorm,
            .rgba8Unorm_srgb, .rgba8Snorm, .rgba8Uint, .rgba8Sint, .bgra8Unorm, .bgra8Unorm_srgb, .rgb10a2Unorm,
            .rgb10a2Uint, .rg11b10Float, .rgb9e5Float, .bgr10a2Unorm, .depth32Float, .bgr10_xr, .bgr10_xr_srgb,
        ], 1, 1, 4)
        add([
            .rg32Uint, .rg32Sint, .rg32Float, .rgba16Unorm, .rgba16Snorm, .rgba16Uint, .rgba16Sint, .rgba16Float,
            .bgra10_xr, .bgra10_xr_srgb,
        ], 1, 1, 8)
        add([.rgba32Uint, .rgba32Sint, .rgba32Float], 1, 1, 16)
        add([.bc1_rgba, .bc1_rgba_srgb, .bc4_rUnorm, .bc4_rSnorm], 4, 4, 8)
        add([
            .bc2_rgba, .bc2_rgba_srgb, .bc3_rgba, .bc3_rgba_srgb, .bc5_rgUnorm, .bc5_rgSnorm, .bc6H_rgbFloat,
            .bc6H_rgbuFloat, .bc7_rgbaUnorm, .bc7_rgbaUnorm_srgb,
        ], 4, 4, 16)
        add([.etc2_rgb8, .etc2_rgb8_srgb, .etc2_rgb8a1, .etc2_rgb8a1_srgb, .eac_r11Unorm, .eac_r11Snorm], 4, 4, 8)
        add([.eac_rgba8, .eac_rgba8_srgb, .eac_rg11Unorm, .eac_rg11Snorm], 4, 4, 16)
        let astc: [(Int, Int, MTLPixelFormat, MTLPixelFormat, MTLPixelFormat)] = [
            (4, 4, .astc_4x4_ldr, .astc_4x4_srgb, .astc_4x4_hdr), (5, 4, .astc_5x4_ldr, .astc_5x4_srgb, .astc_5x4_hdr),
            (5, 5, .astc_5x5_ldr, .astc_5x5_srgb, .astc_5x5_hdr), (6, 5, .astc_6x5_ldr, .astc_6x5_srgb, .astc_6x5_hdr),
            (6, 6, .astc_6x6_ldr, .astc_6x6_srgb, .astc_6x6_hdr), (8, 5, .astc_8x5_ldr, .astc_8x5_srgb, .astc_8x5_hdr),
            (8, 6, .astc_8x6_ldr, .astc_8x6_srgb, .astc_8x6_hdr), (8, 8, .astc_8x8_ldr, .astc_8x8_srgb, .astc_8x8_hdr),
            (10, 5, .astc_10x5_ldr, .astc_10x5_srgb, .astc_10x5_hdr), (10, 6, .astc_10x6_ldr, .astc_10x6_srgb, .astc_10x6_hdr),
            (10, 8, .astc_10x8_ldr, .astc_10x8_srgb, .astc_10x8_hdr), (10, 10, .astc_10x10_ldr, .astc_10x10_srgb, .astc_10x10_hdr),
            (12, 10, .astc_12x10_ldr, .astc_12x10_srgb, .astc_12x10_hdr), (12, 12, .astc_12x12_ldr, .astc_12x12_srgb, .astc_12x12_hdr),
        ]
        for (width, height, ldr, srgb, hdr) in astc { add([ldr, srgb, hdr], width, height, 16) }
        return layouts
    }()

    private static let pvrtcFormats: [(format: MTLPixelFormat, bitsPerPixel: Int, minimumWidth: Int)] = [
        (.pvrtc_rgb_2bpp, 2, 16), (.pvrtc_rgb_2bpp_srgb, 2, 16), (.pvrtc_rgba_2bpp, 2, 16), (.pvrtc_rgba_2bpp_srgb, 2, 16),
        (.pvrtc_rgb_4bpp, 4, 8), (.pvrtc_rgb_4bpp_srgb, 4, 8), (.pvrtc_rgba_4bpp, 4, 8), (.pvrtc_rgba_4bpp_srgb, 4, 8),
    ]

    @Test("Every documented layout is sized at odd and minimal dimensions, and every other Metal format is rejected")
    func everyClaimedLayout() throws {
        for (format, block) in Self.documentedLayouts {
            for (width, height) in [(1, 1), (11, 7), (16, 16), (13, 33)] {
                let expected = (width + block.width - 1) / block.width * ((height + block.height - 1) / block.height) * block.bytes
                #expect(try TextureUtilities.byteCount(width: width, height: height, pixelFormat: format) == expected,
                        "format \(format.rawValue), \(width)×\(height)")
            }
        }
        // The WebGL PVRTC extension sizes a level as floor((max(w, minW) * max(h, 8) * bpp + 7) / 8).
        for (format, bitsPerPixel, minimumWidth) in Self.pvrtcFormats {
            for (width, height) in [(1, 1), (2, 16), (8, 8), (16, 8), (32, 16), (64, 64), (4, 128)] {
                let expected = (max(width, minimumWidth) * max(height, 8) * bitsPerPixel + 7) / 8
                #expect(try TextureUtilities.byteCount(width: width, height: height, pixelFormat: format) == expected,
                        "format \(format.rawValue), \(width)×\(height)")
            }
        }
        // Nothing else is accepted, including combined depth-stencil, video and invalid formats.
        let accepted = Set(Self.documentedLayouts.keys).union(Self.pvrtcFormats.map(\.format))
        var rejected = 0
        for raw in 0..<UInt(1_024) {
            guard let format = MTLPixelFormat(rawValue: raw), !accepted.contains(format) else { continue }
            rejected += 1
            #expect(throws: TextureUtilities.Failure.pixelFormat(format)) {
                try TextureUtilities.byteCount(width: 4, height: 4, pixelFormat: format)
            }
        }
        #expect(rejected > 0)
    }

    @Test("Byte count rejects unsupported and overflowing requests without trapping")
    func invalidByteCount() throws {
        #expect(throws: TextureUtilities.Failure.dimensions) {
            try TextureUtilities.byteCount(width: 0, height: 8, pixelFormat: .rgba8Unorm)
        }
        #expect(throws: TextureUtilities.Failure.pvrtcDimensions) {
            try TextureUtilities.byteCount(width: 12, height: 8, pixelFormat: .pvrtc_rgb_4bpp)
        }
        #expect(throws: TextureUtilities.Failure.pixelFormat(.invalid)) {
            try TextureUtilities.byteCount(width: 1, height: 1, pixelFormat: .invalid)
        }
        #expect(throws: TextureUtilities.Failure.pixelFormat(.depth32Float_stencil8)) {
            try TextureUtilities.byteCount(width: 1, height: 1, pixelFormat: .depth32Float_stencil8)
        }
        #expect(throws: TextureUtilities.Failure.byteCountOverflow) {
            try TextureUtilities.byteCount(width: Int.max, height: Int.max, pixelFormat: .bc7_rgbaUnorm)
        }
        #expect(throws: TextureUtilities.Failure.byteCountOverflow) {
            try TextureUtilities.byteCount(width: Int.max, height: 1, pixelFormat: .rgba16Float)
        }
        #expect(try TextureUtilities.byteCount(width: Int.max, height: 1, pixelFormat: .r8Unorm) == Int.max)
    }

    @Test("Binary16 matches IEEE tie rounding, subnormals and special values")
    func halfRounding() {
        #expect(Binary16.encode(0) == 0)
        #expect(Binary16.encode(-0.0) == 0x8000)
        #expect(Binary16.encode(1 + 0x1p-11) == 0x3c00)
        #expect(Binary16.encode(1 + 3 * 0x1p-11) == 0x3c02)
        #expect(Binary16.encode(0x1p-25) == 0)
        #expect(Binary16.encode(3 * 0x1p-25) == 2)
        #expect(Binary16.decode(0x0001) == 0x1p-24)
        #expect(Binary16.decode(0x0400) == 0x1p-14)
        #expect(Binary16.encode(65_504) == 0x7bff)
        // IEEE overflow: below the 65,520 midpoint rounds down to the largest finite half, and at or
        // above it goes to infinity. r187 DataUtils.toHalfFloat clamps both to ±65,504 instead.
        #expect(Binary16.encode(Float(65_520).nextDown) == 0x7bff)
        #expect(Binary16.encode(65_520) == 0x7c00)
        #expect(Binary16.encode(1e6) == 0x7c00 && Binary16.encode(-1e6) == 0xfc00)
        #expect(Binary16.encode(.infinity) == 0x7c00)
        #expect(Binary16.encode(-.infinity) == 0xfc00)
        #expect(Binary16.decode(0x8000).sign == .minus)
        #expect(Binary16.decode(0x7c00) == .infinity)
        #expect(Binary16.decode(Binary16.encode(.nan)).isNaN)
    }

    @Test("Every binary16 pattern decodes and finite values and infinities round-trip")
    func halfPatterns() {
        #expect((UInt16.min...UInt16.max).allSatisfy { bits in
            let exponent = bits & 0x7c00
            let fraction = bits & 0x03ff
            let value = Binary16.decode(bits)
            return exponent == 0x7c00 && fraction != 0 ? value.isNaN : Binary16.encode(value) == bits
        })
    }

    @Test("Float sRGB conversion transforms RGB and keeps straight alpha, which r187 would also convert")
    func colorTransfer() {
        let floating = ImageUtilities.sRGBToLinear(SIMD4<Float>(0.5, 0.04045, 1, 0.25))
        #expect(isNearly(floating, SIMD4(0.21404114, 0.003130805, 1, 0.25), tolerance: 1e-6))
        // r187 ImageUtils.sRGBToLinear loops over every array element, so it would turn this 0.25 into 0.0509.
        #expect(floating.w == 0.25)
    }

    @Test("Float sRGB conversion matches the r187 transfer formula with its rounded constants")
    func floatTransferMatchesR187() {
        // r187 ColorManagement.SRGBToLinear: c < 0.04045 ? c * 0.0773993808 : pow(c * 0.9478672986 + 0.0521327014, 2.4).
        func r187(_ value: Double) -> Double {
            value < 0.04045 ? value * 0.0773993808 : pow(value * 0.9478672986 + 0.0521327014, 2.4)
        }
        // Includes the segment boundary and values outside 0...1, which neither implementation clamps.
        let samples: [Float] = (0...1_000).map { Float($0) / 1_000 } + [-0.5, 0.04045, 0.040449999, 2]
        var worst: (sample: Float, error: Double) = (0, 0)
        for sample in samples {
            let expected = r187(Double(sample))
            let rgb = ImageUtilities.sRGBToLinear(SIMD4(sample, sample, sample, 0.5))
            for lane in [rgb.x, rgb.y, rgb.z] {
                // Float32 arithmetic carries about 1e-7 relative error, so compare relative to the magnitude.
                let error = abs(Double(lane) - expected) / max(1, abs(expected))
                if error > worst.error { worst = (sample, error) }
            }
        }
        #expect(worst.error < 1e-6, "worst relative error \(worst.error) at \(worst.sample)")
    }

    @Test("Byte sRGB conversion rounds each linear code to the nearest value rather than truncating")
    func byteTransfer() {
        func linearCode(_ code: Int) -> Double {
            let value = Double(code) / 255
            return (value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4)) * 255
        }
        var separating = 0
        var mismatches: [Int] = []
        for code in 0...255 {
            let nearest = UInt8(linearCode(code).rounded())
            if nearest != UInt8(linearCode(code).rounded(.down)) { separating += 1 }
            let other = 255 - code
            let alpha = UInt8((code * 7) % 256)
            let converted = ImageUtilities.sRGBToLinear(SIMD4<UInt8>(UInt8(code), UInt8(other), UInt8(code), alpha))
            let expected = SIMD4(nearest, UInt8(linearCode(other).rounded()), nearest, alpha)
            if converted != expected { mismatches.append(code) }
        }
        #expect(mismatches.isEmpty, "codes converted differently: \(mismatches)")
        // Truncation and rounding disagree for 135 of the 256 codes, so the loop above can tell them apart.
        #expect(separating >= 100)

        // Codes 7, 10 and 220 land at 0.54, 0.77 and 182.50 and round up. Truncating gives 0, 0 and 182.
        // r187's raw-byte path would also turn the alpha code 128 into 55; it stays 128 here.
        #expect(ImageUtilities.sRGBToLinear(SIMD4<UInt8>(7, 10, 220, 128)) == SIMD4(1, 1, 183, 128))
        #expect(ImageUtilities.sRGBToLinear(SIMD4<UInt8>(128, 64, 255, 17)) == SIMD4(55, 13, 255, 17))
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TextureUtilitiesMetalTests {
    private let gpu: GeometryRenderFixture

    init() throws { gpu = try GeometryRenderFixture(side: 64) }

    @Test("Cover crops the outside columns and contain keeps them with transparent bars")
    func renderFits() throws {
        let row0: [UInt8] = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255]
        let row1: [UInt8] = [0, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255]
        let texture = try makeTexture(.rgba8Unorm, width: 4, height: 2, bytes: row0 + row1)
        let cover = try TextureUtilities.fit(texture, surfaceAspectRatio: 1, mode: .cover)
        let sampling = PhysicallyBasedMaterial.TextureSampling(
            addressU: .clampToEdge, addressV: .clampToEdge, magnification: .nearest, minification: .nearest,
            mipFilter: .notMipmapped
        )
        let coverMaterial = try PhysicallyBasedMaterial(
            baseColorTexture: texture, baseColorSampler: sampling.makeSamplerState(device: gpu.device),
            alphaMode: .opaque, baseColorTextureTransform: cover, shadingModel: .unlit, baseColorSampling: sampling
        )
        let cropped = try render(coverMaterial)
        #expect(gpu.pixel(cropped, x: 4, y: 16) == SIMD4(0, 1, 0, 1))
        #expect(gpu.pixel(cropped, x: 59, y: 16) == SIMD4(0, 0, 1, 1))

        let contain = try TextureUtilities.fit(texture, surfaceAspectRatio: 1, mode: .contain)
        let graph = try SurfaceShaderGraph { surface in
            let map = surface.texture("Image", addressing: .clampToEdge, filtering: .nearest)
            let uv = surface.uv0 * contain.scale + contain.offset
            let inside = uv.y.greaterThan(surface.constant(Float(0))) * uv.y.lessThan(surface.constant(Float(1)))
            return .init(baseColor: map.sample(uv).rgb, mask: inside)
        }
        let containMaterial = try PhysicallyBasedMaterial(
            alphaMode: .opaque, surfaceGraph: graph, surfaceTextures: ["Image": texture], shadingModel: .unlit
        )
        let letterboxed = try render(containMaterial)
        #expect(gpu.pixel(letterboxed, x: 4, y: 20) == SIMD4(1, 0, 0, 1))
        #expect(gpu.pixel(letterboxed, x: 59, y: 20) == SIMD4(1, 1, 0, 1))
        #expect(gpu.pixel(letterboxed, x: 32, y: 4) == .zero)
        #expect(gpu.pixel(letterboxed, x: 32, y: 59) == .zero)
    }

    @Test("PNG data URL reopens exact sRGB or explicitly linearized RGB with unchanged alpha")
    func pngDataURL() async throws {
        let bytes: [UInt8] = [128, 64, 255, 17, 16, 192, 32, 127]
        let texture = try makeTexture(.rgba8Unorm_srgb, width: 2, height: 1, bytes: bytes)
        for interpretation in [TextureLoader.Interpretation.encodedSRGB, .linearData] {
            let url = try ImageUtilities.pngDataURL(from: texture, interpretation: interpretation, commandQueue: gpu.queue)
            let prefix = "data:image/png;base64,"
            try #require(url.hasPrefix(prefix))
            let png = try #require(Data(base64Encoded: String(url.dropFirst(prefix.count))))
            #expect(png.starts(with: [0x89, 0x50, 0x4e, 0x47]))
            let decoded = try TextureLoader(device: gpu.device).load(data: png, interpretation: interpretation)
            let stored = try await TextureTransfer.read(from: decoded, commandQueue: gpu.queue)
            let size = try TextureUtilities.byteCount(width: stored.size.width, height: stored.size.height, pixelFormat: stored.pixelFormat)
            let expected = interpretation == .encodedSRGB ? bytes : [55, 13, 255, 17, 1, 134, 4, 127]
            #expect(stored.bytes == Data(expected))
            #expect(stored.bytes.count == size && size == 8)
        }
    }

    @Test("The default PNG interpretation encodes linear-format texels instead of dumping their stored bytes")
    func pngDataURLEncodesLinearTexels() async throws {
        let stored: [UInt8] = [55, 13, 255, 17, 1, 134, 4, 127]
        let texture = try makeTexture(.rgba8Unorm, width: 2, height: 1, bytes: stored)
        // Default interpretation: sRGB-encoded PNG codes; only alpha is copied.
        let url = try ImageUtilities.pngDataURL(from: texture, commandQueue: gpu.queue)
        let png = try #require(Data(base64Encoded: String(url.dropFirst("data:image/png;base64,".count))))
        let decoded = try TextureLoader(device: gpu.device).load(data: png, interpretation: .encodedSRGB)
        #expect(decoded.pixelFormat == .rgba8Unorm_srgb)
        let encoded = try await TextureTransfer.read(from: decoded, commandQueue: gpu.queue)
        #expect(encoded.bytes == Data([128, 64, 255, 17, 13, 192, 34, 127]))
        #expect(encoded.bytes != Data(stored))

        // Linear data copies a linear format's bytes unchanged.
        let linearURL = try ImageUtilities.pngDataURL(from: texture, interpretation: .linearData, commandQueue: gpu.queue)
        let linearPNG = try #require(Data(base64Encoded: String(linearURL.dropFirst("data:image/png;base64,".count))))
        let linear = try TextureLoader(device: gpu.device).load(data: linearPNG, interpretation: .linearData)
        #expect(try await TextureTransfer.read(from: linear, commandQueue: gpu.queue).bytes == Data(stored))
    }

    private func render(_ material: PhysicallyBasedMaterial) throws -> [SIMD4<Float>] {
        let mesh = try gpu.unlit(MeshGeometry.plane(width: 4, height: 4), material: .physicallyBased(material))
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)
        return try gpu.render(SceneGraph(roots: [node], camera: gpu.camera)).pixels
    }

    private func makeTexture(
        _ format: MTLPixelFormat, width: Int, height: Int, bytes: [UInt8]
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(gpu.device.makeTexture(descriptor: descriptor))
        try bytes.withUnsafeBytes { storage in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: try #require(storage.baseAddress), bytesPerRow: width * 4
            )
        }
        return texture
    }
}
