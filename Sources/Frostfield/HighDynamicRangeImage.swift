import CoreGraphics
import Foundation
import FrostfieldJPEGXL
import ImageIO
import Metal

public extension TextureLoader {
    /// The float format `loadHighDynamicRange` writes.
    enum HighDynamicRangePrecision: Equatable, Sendable {
        /// RGBA16Float: half the memory of `.float` and filterable on every GPU. Magnitudes
        /// above 65,504 clamp to 65,504, and half precision keeps about three decimal digits.
        case half
        /// RGBA32Float: the decoded float values unchanged. Linear filtering and mipmaps
        /// need a device that supports 32-bit float filtering.
        case float
    }

    /// A decoded linear high-dynamic-range texture.
    struct HighDynamicRangeTexture {
        public enum Source: Equatable, Sendable {
            /// OpenEXR decoded by ImageIO. RGB and alpha are the file's values; OpenEXR alpha is
            /// associated (premultiplied), and the loader leaves it that way.
            case openEXR
            /// Radiance RGBE decoded by Frostfield with the reference formula (m + 0.5) · 2^(e − 136),
            /// which `EnvironmentLighting(radianceHDR:)` also uses. Every decoded value is exact in single
            /// precision, so `.float` keeps values above 65,504 and below half precision's normal range.
            case radiance
            /// An image with an ISO 21496-1 or Apple HDR gain map, decoded to its HDR rendition by ImageIO.
            case gainMap
            /// An image decoded in a PQ, HLG, or extended-range color space without a gain map.
            case highDynamicRangeEncoding
            /// A standard-dynamic-range image, converted to linear values between zero and one.
            case standardDynamicRange
        }

        /// A shader-readable RGBA16Float or RGBA32Float 2D texture in linear sRGB (Rec. 709)
        /// primaries with extended range. For display-referred sources, 1.0 is SDR reference white.
        public let texture: any MTLTexture
        public let source: Source
        /// The ratio of the image's peak to SDR reference white when ImageIO reports one, such as a gain map's headroom.
        public let contentHeadroom: Float?
        /// The largest RGB component stored in the texture, after any half-precision clamp.
        public let maximumValue: Float
    }

    enum HighDynamicRangeFailure: LocalizedError, Equatable {
        /// The data is not an image this loader can decode as high dynamic range.
        case invalidImage(String)
        /// RGBA32Float mipmaps need a device that supports 32-bit float filtering.
        case floatFilteringUnsupported

        public var errorDescription: String? {
            switch self {
            case .invalidImage(let reason): "Could not decode a high-dynamic-range image. \(reason)"
            case .floatFilteringUnsupported: "This device cannot filter RGBA32Float textures, so it cannot generate their mipmaps."
            }
        }
    }

    /// Loads a regular local file. The caller supplies any required sandbox access.
    func loadHighDynamicRange(
        contentsOf url: URL, precision: HighDynamicRangePrecision = .half, mipmapped: Bool = false
    ) throws -> HighDynamicRangeTexture {
        guard url.isFileURL else { throw Failure.localFileRequired }
        let data: Data
        do {
            data = try Self.readLocalFile(url)
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.fileRead(error.localizedDescription)
        }
        return try loadHighDynamicRange(data: data, precision: precision, mipmapped: mipmapped)
    }

    /// Decodes up to 64 MiB of encoded data into a linear float texture, like Three.js
    /// `EXRLoader`, `RGBELoader`, and `UltraHDRLoader`.
    ///
    /// - OpenEXR uses ImageIO's decoder. Float data in the linear sRGB space ImageIO reports
    ///   is copied without conversion, so `.float` keeps every 32-bit value bit for bit.
    /// - Radiance RGBE uses Frostfield's decoder at any orientation and aspect ratio.
    /// - Still JPEG XL uses a bounded native decoder and color management to preserve
    ///   extended linear sRGB radiance and straight alpha.
    /// - Other formats ImageIO reads, including gain-map JPEG, HEIC, and AVIF, are decoded with
    ///   `kCGImageSourceDecodeToHDR`, oriented upright, and drawn into an extended linear sRGB
    ///   float context without tone mapping. Their alpha is straight. ImageIO on macOS 27.2 returns
    ///   that rendition as 10-bit PQ, so values carry PQ quantization, about 1% near 4× SDR white.
    ///
    /// NaN becomes zero and infinities clamp to the largest finite value of the chosen precision.
    /// The image may be at most 8192 pixels per side and 16,777,216 pixels in total; larger
    /// images throw `TextureLoader.Failure.imageDimensions` before any pixel is decoded.
    func loadHighDynamicRange(
        data: Data, precision: HighDynamicRangePrecision = .half, mipmapped: Bool = false
    ) throws -> HighDynamicRangeTexture {
        guard data.count <= Self.maxEncodedBytes else { throw Failure.inputTooLarge }
        if mipmapped, precision == .float, !device.supports32BitFloatFiltering {
            throw HighDynamicRangeFailure.floatFilteringUnsupported
        }
        var image = try HighDynamicRangeImage.decode(data)
        let limit: Float = precision == .half ? Float(Float16.greatestFiniteMagnitude) : .greatestFiniteMagnitude
        var maximum: Float = 0
        let lower = SIMD4<Float>(repeating: -limit), upper = SIMD4<Float>(repeating: limit)
        // The decoded array is uniquely referenced, so it is clamped in place without a copy.
        image.pixels.withUnsafeMutableBufferPointer { buffer in
            for index in buffer.indices {
                var pixel = buffer[index]
                pixel.replace(with: 0, where: pixel .!= pixel)
                pixel = pixel.clamped(lowerBound: lower, upperBound: upper)
                maximum = max(maximum, SIMD3(pixel.x, pixel.y, pixel.z).max())
                buffer[index] = pixel
            }
        }

        let format: MTLPixelFormat = precision == .half ? .rgba16Float : .rgba32Float
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: image.width, height: image.height, mipmapped: mipmapped
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw Failure.textureAllocation(width: image.width, height: image.height)
        }
        texture.label = "Frostfield HDR image"
        let region = MTLRegionMake2D(0, 0, image.width, image.height)
        switch precision {
        case .half:
            let halves = image.pixels.map(SIMD4<Float16>.init)
            halves.withUnsafeBytes { raw in
                texture.replace(region: region, mipmapLevel: 0, withBytes: raw.baseAddress!, bytesPerRow: image.width * 8)
            }
        case .float:
            image.pixels.withUnsafeBytes { raw in
                texture.replace(region: region, mipmapLevel: 0, withBytes: raw.baseAddress!, bytesPerRow: image.width * 16)
            }
        }
        if texture.mipmapLevelCount > 1 { try generateMipmaps(for: texture) }
        return HighDynamicRangeTexture(
            texture: texture, source: image.source, contentHeadroom: image.contentHeadroom, maximumValue: maximum
        )
    }
}

/// Decodes HDR containers to linear float RGBA rows, top row first.
struct HighDynamicRangeImage {
    let width: Int
    let height: Int
    var pixels: [SIMD4<Float>]
    let source: TextureLoader.HighDynamicRangeTexture.Source
    let contentHeadroom: Float?

    private typealias Failure = TextureLoader.HighDynamicRangeFailure

    /// Formats that are linear HDR by definition.
    enum Container {
        case openEXR, radiance

        init?(_ data: Data) {
            let bytes = [UInt8](data.prefix(4))
            if bytes == [0x76, 0x2F, 0x31, 0x01] {
                self = .openEXR
            } else if bytes.starts(with: [0x23, 0x3F]) { // "#?", the Radiance program identifier
                self = .radiance
            } else {
                return nil
            }
        }
    }

    static func decode(_ data: Data) throws -> Self {
        if data.starts(with: [0xff, 0x0a]) ||
            data.starts(with: [0, 0, 0, 12, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a]) {
            return try decodeJPEGXL(data)
        }
        switch Container(data) {
        case .radiance:
            do {
                let image = try RadianceHDRImage.decodeLinear(data, validateSize: TextureLoader.validate)
                return Self(width: image.width, height: image.height, pixels: image.pixels, source: .radiance, contentHeadroom: nil)
            } catch let failure as TextureLoader.Failure {
                throw failure
            } catch {
                throw Failure.invalidImage(error.localizedDescription)
            }
        case .openEXR:
            return try decodeImageIO(data, source: .openEXR)
        case nil:
            return try decodeImageIO(data, source: nil)
        }
    }

    private static func decodeJPEGXL(_ data: Data) throws -> Self {
        var decoded = FrostfieldJPEGXLImage()
        let status = data.withUnsafeBytes { bytes in
            FrostfieldJPEGXLDecode(
                bytes.bindMemory(to: UInt8.self).baseAddress, bytes.count,
                UInt32(TextureLoader.maxDimension), TextureLoader.maxPixels, &decoded
            )
        }
        if status == FrostfieldJPEGXLLimit {
            if decoded.width > 0, decoded.height > 0 {
                try TextureLoader.validate(width: Int(decoded.width), height: Int(decoded.height))
            }
            throw Failure.invalidImage("The JPEG XL decoder reached its allocation limit.")
        }
        if status == FrostfieldJPEGXLAnimated {
            throw Failure.invalidImage("The HDR loader requires a still JPEG XL image.")
        }
        if status == FrostfieldJPEGXLColorProfile {
            throw Failure.invalidImage("The JPEG XL color profile cannot convert to linear sRGB.")
        }
        guard status == FrostfieldJPEGXLSuccess, let storage = decoded.rgba else {
            throw Failure.invalidImage("The JPEG XL data is invalid or truncated.")
        }
        defer { FrostfieldJPEGXLFree(storage) }

        let width = Int(decoded.width), height = Int(decoded.height)
        var pixels = [SIMD4<Float>](repeating: .zero, count: width * height)
        pixels.withUnsafeMutableBytes { destination in
            destination.copyMemory(from: UnsafeRawBufferPointer(start: storage, count: destination.count))
        }
        let extended = decoded.reference_headroom > 0 || pixels.contains { pixel in
            let rgb = SIMD3(pixel.x, pixel.y, pixel.z)
            return rgb.min() < 0 || rgb.max() > 1
        }
        return Self(
            width: width, height: height, pixels: pixels,
            source: extended ? .highDynamicRangeEncoding : .standardDynamicRange,
            contentHeadroom: decoded.reference_headroom >= 1 ? decoded.reference_headroom : nil
        )
    }

    private static func decodeImageIO(_ data: Data, source kind: TextureLoader.HighDynamicRangeTexture.Source?) throws -> Self {
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        if let format = ImageSignature(data), !format.isDecodable {
            throw TextureLoader.Failure.unsupportedImageFormat(format.name)
        }
        guard let imageSource = CGImageSourceCreateWithData(data as CFData, options), CGImageSourceGetCount(imageSource) > 0,
              let properties = CGImageSourceCopyPropertiesAtIndex(imageSource, 0, options) as? [String: Any],
              let width = properties[kCGImagePropertyPixelWidth as String] as? Int,
              let height = properties[kCGImagePropertyPixelHeight as String] as? Int else {
            throw Failure.invalidImage("ImageIO cannot read the data.")
        }
        try TextureLoader.validate(width: width, height: height)

        if kind == .openEXR {
            guard let image = CGImageSourceCreateImageAtIndex(imageSource, 0, options) else {
                throw Failure.invalidImage("ImageIO cannot decode the OpenEXR image.")
            }
            let pixels = try copyLinearFloat(image) ?? draw(image, width: image.width, height: image.height, straightAlpha: false)
            return Self(width: image.width, height: image.height, pixels: pixels, source: .openEXR, contentHeadroom: nil)
        }

        let hasGainMap = CGImageSourceCopyAuxiliaryDataInfoAtIndex(imageSource, 0, kCGImageAuxiliaryDataTypeISOGainMap) != nil
            || CGImageSourceCopyAuxiliaryDataInfoAtIndex(imageSource, 0, kCGImageAuxiliaryDataTypeHDRGainMap) != nil
        // The thumbnail path applies EXIF orientation and honors the HDR decode request.
        let decode: [CFString: Any] = [
            kCGImageSourceDecodeRequest: kCGImageSourceDecodeToHDR,
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(width, height),
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(imageSource, 0, decode as CFDictionary) else {
            throw Failure.invalidImage("ImageIO cannot decode the image.")
        }
        try TextureLoader.validate(width: image.width, height: image.height)
        let space = image.colorSpace
        let isHDR = space.map { CGColorSpaceUsesITUR_2100TF($0) || CGColorSpaceUsesExtendedRange($0) } ?? false
        let headroom = image.contentHeadroom
        return Self(
            width: image.width, height: image.height,
            pixels: try draw(image, width: image.width, height: image.height, straightAlpha: true),
            source: hasGainMap ? .gainMap : (isHDR ? .highDynamicRangeEncoding : .standardDynamicRange),
            contentHeadroom: headroom >= 1 ? headroom : nil
        )
    }

    /// Copies float RGBA or RGBX rows that ImageIO already stores in extended linear sRGB.
    private static func copyLinearFloat(_ image: CGImage) throws -> [SIMD4<Float>]? {
        guard let space = image.colorSpace,
              space.name == CGColorSpace.extendedLinearSRGB || space.name == CGColorSpace.linearSRGB,
              image.bitmapInfo.contains(.floatComponents),
              [16, 32].contains(image.bitsPerComponent), image.bitsPerPixel == image.bitsPerComponent * 4,
              [.premultipliedLast, .last, .noneSkipLast].contains(image.alphaInfo),
              let provider = image.dataProvider?.data else { return nil }
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        let little: CGBitmapInfo = image.bitsPerComponent == 16 ? .byteOrder16Little : .byteOrder32Little
        guard order == little || (order == [] && UInt16(littleEndian: 1) == 1) else { return nil }
        let rowBytes = image.bytesPerRow, width = image.width, height = image.height
        guard rowBytes >= width * image.bitsPerPixel / 8, CFDataGetLength(provider) >= rowBytes * height,
              let bytes = CFDataGetBytePtr(provider) else { return nil }
        let opaque = image.alphaInfo == .noneSkipLast
        var pixels = [SIMD4<Float>](repeating: .zero, count: width * height)
        // Little-endian rows match the host layout of SIMD4<Float> and SIMD4<Float16>.
        withExtendedLifetime(provider) {
            let base = UnsafeRawPointer(bytes)
            pixels.withUnsafeMutableBufferPointer { output in
                for row in 0..<height {
                    let line = base.advanced(by: row * rowBytes)
                    let destination = UnsafeMutableBufferPointer(rebasing: output[(row * width)..<((row + 1) * width)])
                    if image.bitsPerComponent == 32 {
                        UnsafeMutableRawPointer(destination.baseAddress!).copyMemory(from: line, byteCount: width * 16)
                    } else {
                        for column in 0..<width {
                            destination[column] = SIMD4<Float>(line.loadUnaligned(fromByteOffset: column * 8, as: SIMD4<Float16>.self))
                        }
                    }
                    if opaque {
                        for column in 0..<width { destination[column].w = 1 }
                    }
                }
            }
        }
        return pixels
    }

    /// Draws into an extended linear sRGB float context. An EDR target headroom of zero
    /// disables Core Graphics tone mapping, so values above one survive.
    private static func draw(_ image: CGImage, width: Int, height: Int, straightAlpha: Bool) throws -> [SIMD4<Float>] {
        guard let space = CGColorSpace(name: CGColorSpace.extendedLinearSRGB) else {
            throw Failure.invalidImage("The extended linear sRGB color space is unavailable.")
        }
        var pixels = [SIMD4<Float>](repeating: .zero, count: width * height)
        let drawn = pixels.withUnsafeMutableBytes { raw -> Bool in
            guard let context = CGContext(
                data: raw.baseAddress, width: width, height: height, bitsPerComponent: 32, bytesPerRow: width * 16,
                space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.floatComponents.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue
            ) else { return false }
            _ = context.setEDRTargetHeadroom(0)
            context.setBlendMode(.copy)
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw Failure.invalidImage("Core Graphics cannot create a float context.") }
        if straightAlpha {
            for index in pixels.indices where pixels[index].w > 0 && pixels[index].w < 1 {
                let alpha = pixels[index].w
                pixels[index] = SIMD4(pixels[index].x / alpha, pixels[index].y / alpha, pixels[index].z / alpha, alpha)
            }
        }
        return pixels
    }
}
