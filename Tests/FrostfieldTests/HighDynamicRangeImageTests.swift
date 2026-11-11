import CoreGraphics
import Foundation
import Frostfield
import ImageIO
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct HighDynamicRangeImageTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let loader: TextureLoader

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
        loader = TextureLoader(device: device)
    }

    /// Scene-linear values across magnitudes, signs, and float edge cases, 37 × 19 so ZIP's last block is partial.
    private static let width = 37, height = 19
    private static let pixels: [SIMD4<Float>] = (0..<(width * height)).map { index in
        let specials: [Float] = [0, -0, 1e-30, .leastNormalMagnitude / 4, 0.1, 1, 7.5, 1000, 65_504, 1e6, -2.5, 1.0e-5]
        var state = UInt32(truncatingIfNeeded: index &* 2_654_435_761 &+ 12_345)
        func next() -> Float {
            state = state &* 1_664_525 &+ 1_013_904_223
            if state % 7 == 0 { return specials[Int(state >> 8) % specials.count] }
            let exponent = Float(Int(state >> 16) % 24) - 12
            return pow(2, exponent) * (Float(state & 0xFFFF) / 65_536 + 0.5) * (state & 0x10000 == 0 ? 1 : -1)
        }
        return SIMD4(next(), next(), next(), index.isMultiple(of: 5) ? 0.25 : 1)
    }

    @Test(arguments: [EXRExporter.Compression.uncompressed, .zipScanline, .zip], [EXRExporter.PixelType.float, .half])
    func `EXRExporter output decodes in ImageIO to the stored bits`(
        compression: EXRExporter.Compression, pixelType: EXRExporter.PixelType
    ) throws {
        let data = try EXRExporter(pixelType: pixelType, compression: compression)
            .data(width: Self.width, height: Self.height, pixels: Self.pixels)
        let source = try #require(CGImageSourceCreateWithData(data as CFData, nil))
        #expect(CGImageSourceGetType(source) as String? == "com.ilm.openexr-image")
        let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        #expect(image.width == Self.width && image.height == Self.height)
        let decoded = try rawFloats(image)
        var mismatches = 0
        for (index, pixel) in Self.pixels.enumerated() {
            for lane in 0..<4 {
                let expected = pixelType == .float ? pixel[lane] : Float(Float16(pixel[lane]))
                if decoded[index][lane].bitPattern != expected.bitPattern { mismatches += 1 }
            }
        }
        #expect(mismatches == 0, "\(mismatches) channel values differ")
        if compression != .uncompressed {
            let stored = try EXRExporter(pixelType: pixelType, compression: .uncompressed)
                .data(width: Self.width, height: Self.height, pixels: Self.pixels)
            #expect(data.count < stored.count)
        }
    }

    @Test(arguments: [EXRExporter.Compression.rle, .piz], [EXRExporter.PixelType.float, .half])
    func `ImageIO decodes compressed and final raw scanlines bit for bit`(
        compression: EXRExporter.Compression, pixelType: EXRExporter.PixelType
    ) throws {
        let width = 129, height = 65
        // Paired byte values produce long predictor runs even in 32-bit RGBA.
        let reds = [Float(bitPattern: 0x4000_4000), Float(bitPattern: 0x4080_4080)]
        let greens = [Float(bitPattern: 0x3E80_3E80), Float(bitPattern: 0x3F00_3F00)]
        let blue = Float(bitPattern: 0xC000_C000)
        let pixels: [SIMD4<Float>] = (0..<(width * height)).map { index in
            let x = index % width, y = index / width
            if y == height - 1 { return Self.pixels[x] }
            return SIMD4(reds[x < 65 ? 0 : 1], greens[y < 16 ? 0 : 1], blue, 1)
        }
        let exporter = EXRExporter(pixelType: pixelType, compression: compression)
        let data = try exporter.data(width: width, height: height, pixels: pixels)
        let raw = try EXRExporter(pixelType: pixelType, compression: .uncompressed)
            .data(width: width, height: height, pixels: pixels)
        #expect(data.count < raw.count / 3)

        let source = try #require(CGImageSourceCreateWithData(data as CFData, nil))
        let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        #expect(image.width == width && image.height == height)
        let expected = pixels.flatMap { pixel in
            (0..<4).map { lane in
                let stored = pixelType == .float ? pixel[lane] : Float(Float16(pixel[lane]))
                return stored.bitPattern
            }
        }
        #expect(try rawFloats(image).flatMap { pixel in (0..<4).map { pixel[$0].bitPattern } } == expected)
    }

    @Test
    func `TextureLoader keeps EXR floats exactly in RGBA32Float and clamps half precision`() throws {
        let data = try EXRExporter(pixelType: .float).data(width: Self.width, height: Self.height, pixels: Self.pixels)
        let exact = try loader.loadHighDynamicRange(data: data, precision: .float)
        #expect(exact.source == .openEXR && exact.texture.pixelFormat == .rgba32Float)
        let floats = try read(exact.texture, as: SIMD4<Float>.self)
        #expect(floats.map { $0.x.bitPattern } == Self.pixels.map { $0.x.bitPattern })
        #expect(floats == Self.pixels)
        #expect(exact.maximumValue == Self.pixels.map { max($0.x, $0.y, $0.z) }.max())

        // Every interpretation of `load(data:)` routes EXR to the linear half-float path.
        for interpretation in [TextureLoader.Interpretation.color, .encodedSRGB, .linearData] {
            let half = try loader.load(data: data, interpretation: interpretation)
            #expect(half.pixelFormat == .rgba16Float)
            let values = try read(half, as: SIMD4<Float16>.self)
            let expected = Self.pixels.map { SIMD4<Float16>(simd_clamp($0, SIMD4(repeating: -65_504), SIMD4(repeating: 65_504))) }
            #expect(values == expected)
        }
    }

    @Test
    func `An EXR written by ImageIO round-trips within half precision of each pixel's peak`() throws {
        let values: [SIMD4<Float>] = [SIMD4(0.25, 0.5, 1, 1), SIMD4(4, 16, 100, 1), SIMD4(0.001, 0.002, 0.003, 1), SIMD4(1000, 0.1, 7.5, 1)]
        let data = try imageIOEXR(values, width: 2, height: 2)
        let loaded = try loader.loadHighDynamicRange(data: data, precision: .float)
        let decoded = try read(loaded.texture, as: SIMD4<Float>.self)
        var worst: Float = 0
        for (source, result) in zip(values, decoded) {
            let peak = max(abs(source.x), abs(source.y), abs(source.z))
            for lane in 0..<3 {
                worst = max(worst, abs(result[lane] - source[lane]) / peak)
            }
            #expect(result.w == source.w)
        }
        // ImageIO's encoder converts color; the error stays below one half-precision step of the pixel's peak.
        #expect(worst > 0 && worst < Float(Float16.ulpOfOne), "worst relative error \(worst)")
        #expect(abs(decoded[3].y - 0.1) > 0.01) // the cross-talk the Frostfield exporter avoids
    }

    @Test
    func `Radiance RGBE files load at any aspect ratio with the reference decoding`() throws {
        let rgbe: [SIMD4<UInt8>] = [SIMD4(128, 64, 32, 129), SIMD4(200, 100, 50, 136), SIMD4(255, 0, 0, 140),
                                    SIMD4(0, 0, 0, 0), SIMD4(10, 20, 30, 120), SIMD4(128, 128, 128, 128)]
        var data = Data("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\nEXPOSURE=1\n\n-Y 2 +X 3\n".utf8)
        for pixel in rgbe { data.append(contentsOf: [pixel.x, pixel.y, pixel.z, pixel.w]) }
        func reference(_ pixel: SIMD4<UInt8>) -> SIMD4<Float> {
            guard pixel.w != 0 else { return SIMD4(0, 0, 0, 1) }
            let scale = pow(2, Float(Int(pixel.w) - 136))
            return SIMD4((Float(pixel.x) + 0.5) * scale, (Float(pixel.y) + 0.5) * scale, (Float(pixel.z) + 0.5) * scale, 1)
        }
        let texture = try loader.load(data: data)
        #expect(texture.pixelFormat == .rgba16Float && texture.width == 3 && texture.height == 2)
        #expect(try read(texture, as: SIMD4<Float16>.self) == rgbe.map { SIMD4<Float16>(reference($0)) })
        let loaded = try loader.loadHighDynamicRange(data: data, precision: .float)
        #expect(loaded.source == .radiance)
        // Every RGBE value in the half-float normal range is exact in half precision.
        #expect(try read(loaded.texture, as: SIMD4<Float>.self) == rgbe.map(reference))
        #expect(loaded.maximumValue == reference(rgbe[2]).x)
    }

    @Test
    func `Radiance values outside half precision's range load exactly in RGBA32Float`() throws {
        // 102,656 and 2.16 × 10⁹ exceed half precision; 1.87 × 10⁻⁹ and 2⁻¹³⁶ are below its subnormals.
        let rgbe: [SIMD4<UInt8>] = [SIMD4(200, 100, 50, 145), SIMD4(128, 128, 128, 160), SIMD4(128, 128, 128, 100),
                                    SIMD4(0, 1, 255, 1)]
        var data = Data("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 4\n".utf8)
        for pixel in rgbe { data.append(contentsOf: [pixel.x, pixel.y, pixel.z, pixel.w]) }
        // (m + 0.5) · 2^(e − 136) in double precision; each value is exact in single precision.
        func reference(_ pixel: SIMD4<UInt8>) -> SIMD4<Float> {
            let scale = pow(2, Double(Int(pixel.w) - 136))
            return SIMD4(Float((Double(pixel.x) + 0.5) * scale), Float((Double(pixel.y) + 0.5) * scale),
                         Float((Double(pixel.z) + 0.5) * scale), 1)
        }
        let expected = rgbe.map(reference)
        #expect(expected[0].x == 102_656 && expected[1].x == 2_155_872_256 && expected[3].z == 255.5 * pow(2, -135))

        let exact = try loader.loadHighDynamicRange(data: data, precision: .float)
        #expect(try read(exact.texture, as: SIMD4<Float>.self) == expected)
        #expect(exact.maximumValue == 2_155_872_256)

        let half = try loader.loadHighDynamicRange(data: data, precision: .half)
        let clamped = expected.map { SIMD4<Float16>(pointwiseMin($0, SIMD4(repeating: 65_504))) }
        #expect(try read(half.texture, as: SIMD4<Float16>.self) == clamped)
        #expect(half.maximumValue == 65_504)
        #expect(clamped[2] == SIMD4(0, 0, 0, 1)) // flushed below half precision
    }

    @Test
    func `Radiance images use TextureLoader's size limits at any aspect ratio`() throws {
        func header(_ resolution: String) -> Data { Data("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n\(resolution)\n".utf8) }
        // 9,000,000 pixels pass the size check, which EnvironmentLighting's 8,388,608-pixel
        // panorama limit would fail, and decoding stops at the missing first scanline.
        let square = header("-Y 3000 +X 3000")
        let truncated = EnvironmentLighting.RadianceFailure.scanline(0).localizedDescription
        #expect(throws: TextureLoader.HighDynamicRangeFailure.invalidImage(truncated)) {
            try loader.loadHighDynamicRange(data: square, precision: .float)
        }
        #expect(throws: TextureLoader.Failure.imageDimensions(width: 9_000, height: 10)) {
            try loader.loadHighDynamicRange(data: header("-Y 10 +X 9000"))
        }
        #expect(throws: TextureLoader.Failure.imageDimensions(width: 4_097, height: 4_097)) {
            try loader.loadHighDynamicRange(data: header("+X 4097 -Y 4097"))
        }
        // EnvironmentLighting keeps its panorama requirement and message.
        #expect(throws: EnvironmentLighting.RadianceFailure.size) { try RadianceHDRImage.decode(square) }
    }

    @Test(.enabled(if: HighDynamicRangeImageTests.canEncodeGainMaps, "Requires ImageIO's ISO gain-map encoder"))
    func `Gain-map HEIC, JPEG, and AVIF images decode to linear HDR values above SDR white`() throws {
        let types = ["public.heic", "public.jpeg", "public.avif"].filter {
            (CGImageDestinationCopyTypeIdentifiers() as? [String])?.contains($0) == true
        }
        #expect(types.contains("public.heic") && types.contains("public.jpeg"))
        for type in types {
            let data = try gainMapImage(type: type, orientation: nil) { x, _ in x < 32 ? SIMD3(repeating: 0.5) : SIMD3(repeating: 4) }
            let hdr = try loader.loadHighDynamicRange(data: data, precision: .float)
            #expect(hdr.source == .gainMap, "\(type)")
            #expect(abs((hdr.contentHeadroom ?? 0) - 4) < 0.01, "\(type) headroom \(hdr.contentHeadroom ?? 0)")
            let pixels = try read(hdr.texture, as: SIMD4<Float>.self)
            let shadow = pixels[10 * 64 + 5], highlight = pixels[10 * 64 + 50]
            #expect(abs(shadow.x - 0.5) < 0.02 && abs(shadow.y - 0.5) < 0.02, "\(type) shadow \(shadow)")
            // The encoder tone-maps the SDR base, and the gain map restores up to the 4× headroom.
            #expect(highlight.x > 3 && highlight.x <= 4.01, "\(type) highlight \(highlight)")
            #expect(hdr.maximumValue > 3)

            // The ordinary loader keeps the SDR base, which cannot exceed one.
            let sdr = try loader.load(data: data, interpretation: .color)
            #expect(sdr.pixelFormat == .rgba8Unorm_srgb)
            let base = try loader.loadHighDynamicRange(data: data, precision: .half)
            #expect(base.texture.pixelFormat == .rgba16Float)
        }
    }

    @Test(.enabled(if: HighDynamicRangeImageTests.canEncodeGainMaps, "Requires ImageIO's ISO gain-map encoder"))
    func `HDR decoding applies EXIF orientation`() throws {
        // Orientation 6 stores the image rotated; displaying it turns the left column into the top row.
        let data = try gainMapImage(type: "public.heic", orientation: 6) { x, _ in x < 32 ? SIMD3(4, 0, 0) : SIMD3(0, 0.5, 0) }
        let hdr = try loader.loadHighDynamicRange(data: data, precision: .float)
        #expect(hdr.texture.width == 32 && hdr.texture.height == 64)
        let pixels = try read(hdr.texture, as: SIMD4<Float>.self)
        #expect(pixels[5 * 32 + 16].x > 3 && pixels[5 * 32 + 16].y < 0.05)
        #expect(pixels[58 * 32 + 16].y > 0.45 && pixels[58 * 32 + 16].x < 0.05)
    }

    @Test
    func `An EXR panorama lights EnvironmentLighting with its HDR radiance`() throws {
        // Uniform radiance L: the sky is L everywhere, and the cosine-convolved diffuse cube
        // stores irradiance / π = L.
        let radiance: Float = 3.25
        let panorama = Array(repeating: SIMD4<Float>(radiance, radiance * 0.5, radiance * 2, 1), count: 64 * 32)
        let exr = try EXRExporter(pixelType: .half).data(width: 64, height: 32, pixels: panorama)
        let loaded = try loader.loadHighDynamicRange(data: exr)
        let lighting = try EnvironmentLighting(
            queue: queue, equirectangular: loaded.texture, settings: .init(cubeSize: 16, diffuseSize: 4, brdfSize: 16, sampleCount: 64)
        )
        for (cube, tolerance) in [(lighting.skyCube, Float(1e-3)), (lighting.diffuseCube, 0.01)] {
            for face in 0..<6 {
                for texel in try readCubeFace(cube, face: face) {
                    let expected = SIMD3(radiance, radiance * 0.5, radiance * 2)
                    let error = simd_reduce_max(abs(SIMD3(texel.x, texel.y, texel.z) - expected) / expected)
                    #expect(error < tolerance, "face \(face) texel \(texel)")
                }
            }
        }
    }

    @Test
    func `EXRExporter reads private float textures and rejects unsupported input`() throws {
        let side = 8
        let values = (0..<(side * side)).map { SIMD4<Float>(Float($0) * 0.5, -Float($0), 1e4 + Float($0), 0.5) }
        let shared = try makeTexture(.rgba32Float, side: side, storage: .shared)
        values.withUnsafeBytes {
            shared.replace(region: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: side * 16)
        }
        let privateTexture = try makeTexture(.rgba32Float, side: side, storage: .private)
        let command = try #require(queue.makeCommandBuffer())
        let blit = try #require(command.makeBlitCommandEncoder())
        blit.copy(from: shared, to: privateTexture)
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        let exr = try EXRExporter(pixelType: .float, compression: .zip, commandQueue: queue).data(from: privateTexture)
        let loaded = try loader.loadHighDynamicRange(data: exr, precision: .float)
        #expect(try read(loaded.texture, as: SIMD4<Float>.self) == values)

        #expect(throws: EXRExporter.Failure.unsupportedTextureFormat(.rgba8Unorm)) {
            try EXRExporter().data(from: try makeTexture(.rgba8Unorm, side: 2, storage: .shared))
        }
        #expect(throws: EXRExporter.Failure.invalidDimensions(width: 2, height: 2, pixelCount: 3)) {
            try EXRExporter().data(width: 2, height: 2, pixels: Array(repeating: .zero, count: 3))
        }
        #expect(throws: TextureLoader.HighDynamicRangeFailure.self) {
            try loader.loadHighDynamicRange(data: Data("#?RADIANCE\nnot an image".utf8))
        }
    }

    // MARK: Helpers

    static let canEncodeGainMaps: Bool = {
        guard let space = CGColorSpace(name: CGColorSpace.extendedLinearSRGB),
              let image = floatImage(width: 64, height: 32, space: space, headroom: 4, { _, _ in SIMD3(repeating: 2) }) else { return false }
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output as CFMutableData, "public.heic" as CFString, 1, nil) else {
            return false
        }
        CGImageDestinationAddImage(destination, image, [kCGImageDestinationEncodeRequest: kCGImageDestinationEncodeToISOGainmap] as CFDictionary)
        return CGImageDestinationFinalize(destination)
    }()

    private static func floatImage(
        width: Int, height: Int, space: CGColorSpace, headroom: Float?, _ color: (Int, Int) -> SIMD3<Float>
    ) -> CGImage? {
        var values = [Float]()
        for y in 0..<height {
            for x in 0..<width {
                let rgb = color(x, y)
                values += [rgb.x, rgb.y, rgb.z, 1]
            }
        }
        guard let provider = CGDataProvider(data: values.withUnsafeBytes { Data($0) } as CFData) else { return nil }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.floatComponents.rawValue
            | CGBitmapInfo.byteOrder32Little.rawValue)
        if let headroom {
            return CGImage(headroom: headroom, width: width, height: height, bitsPerComponent: 32, bitsPerPixel: 128,
                           bytesPerRow: width * 16, space: space, bitmapInfo: info, provider: provider, decode: nil,
                           shouldInterpolate: false, intent: .defaultIntent)
        }
        return CGImage(width: width, height: height, bitsPerComponent: 32, bitsPerPixel: 128, bytesPerRow: width * 16,
                       space: space, bitmapInfo: info, provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    /// A 64 × 32 HDR image with 4× headroom encoded with an ISO 21496-1 gain map.
    private func gainMapImage(type: String, orientation: Int?, _ color: (Int, Int) -> SIMD3<Float>) throws -> Data {
        let space = try #require(CGColorSpace(name: CGColorSpace.extendedLinearSRGB))
        let created = Self.floatImage(width: 64, height: 32, space: space, headroom: 4, color)
        let image = try #require(created)
        let output = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(output as CFMutableData, type as CFString, 1, nil))
        var properties: [CFString: Any] = [kCGImageDestinationEncodeRequest: kCGImageDestinationEncodeToISOGainmap]
        if let orientation { properties[kCGImagePropertyOrientation] = orientation }
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        try #require(CGImageDestinationFinalize(destination))
        return output as Data
    }

    private func imageIOEXR(_ values: [SIMD4<Float>], width: Int, height: Int) throws -> Data {
        let space = try #require(CGColorSpace(name: CGColorSpace.extendedLinearSRGB))
        let created = Self.floatImage(width: width, height: height, space: space, headroom: nil) { x, y in
            let value = values[y * width + x]
            return SIMD3(value.x, value.y, value.z)
        }
        let image = try #require(created)
        let output = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(output as CFMutableData, "com.ilm.openexr-image" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        try #require(CGImageDestinationFinalize(destination))
        return output as Data
    }

    /// The decoded CGImage's raw RGBA components, widened to Float.
    private func rawFloats(_ image: CGImage) throws -> [SIMD4<Float>] {
        let data = try #require(image.dataProvider?.data)
        try #require(image.bitmapInfo.contains(.floatComponents) && image.bitsPerPixel == image.bitsPerComponent * 4)
        let bytes = try #require(CFDataGetBytePtr(data))
        return withExtendedLifetime(data) {
            (0..<(image.width * image.height)).map { index in
                let row = index / image.width, column = index % image.width
                let base = UnsafeRawPointer(bytes).advanced(by: row * image.bytesPerRow)
                if image.bitsPerComponent == 16 {
                    let value = base.loadUnaligned(fromByteOffset: column * 8, as: SIMD4<Float16>.self)
                    return SIMD4<Float>(value)
                }
                return base.loadUnaligned(fromByteOffset: column * 16, as: SIMD4<Float>.self)
            }
        }
    }

    private func read<T>(_ texture: any MTLTexture, as type: T.Type) throws -> [T] {
        try #require(texture.storageMode == .shared)
        let stride = MemoryLayout<T>.stride
        var bytes = [UInt8](repeating: 0, count: texture.width * texture.height * stride)
        bytes.withUnsafeMutableBytes { raw in
            texture.getBytes(raw.baseAddress!, bytesPerRow: texture.width * stride,
                             from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0)
        }
        return bytes.withUnsafeBytes { raw in (0..<(texture.width * texture.height)).map { raw.loadUnaligned(fromByteOffset: $0 * stride, as: T.self) } }
    }

    private func readCubeFace(_ cube: any MTLTexture, face: Int) throws -> [SIMD4<Float>] {
        let side = cube.width
        let buffer = try #require(device.makeBuffer(length: side * side * 8, options: .storageModeShared))
        let command = try #require(queue.makeCommandBuffer())
        let blit = try #require(command.makeBlitCommandEncoder())
        blit.copy(from: cube, sourceSlice: face, sourceLevel: 0, sourceOrigin: MTLOrigin(), sourceSize: MTLSize(width: side, height: side, depth: 1),
                  to: buffer, destinationOffset: 0, destinationBytesPerRow: side * 8, destinationBytesPerImage: side * side * 8)
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        let pointer = buffer.contents().bindMemory(to: SIMD4<Float16>.self, capacity: side * side)
        return (0..<(side * side)).map { SIMD4<Float>(pointer[$0]) }
    }

    private func makeTexture(_ format: MTLPixelFormat, side: Int, storage: MTLStorageMode) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: side, height: side, mipmapped: false)
        descriptor.storageMode = storage
        descriptor.usage = .shaderRead
        return try #require(device.makeTexture(descriptor: descriptor))
    }
}
