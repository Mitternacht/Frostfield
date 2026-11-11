import CoreGraphics
import Foundation
import ImageIO
import Metal

extension GLTFExporter {
    /// Encodes level 0 of a 2D texture as an 8-bit, straight-alpha RGBA PNG that
    /// `TextureLoader` decodes with the same interpretation to the values Metal samples.
    ///
    /// Color interpretations (`.color` and `.encodedSRGB`) write sRGB-encoded RGB: sRGB
    /// formats copy their stored bytes, while linear and float formats are clamped to zero to
    /// one and encoded. `.linearData` writes linear values: sRGB formats are decoded first.
    /// Alpha is always linear. R and RG formats expand to RGBA as Metal samples them, with zero
    /// for missing color channels and opaque alpha.
    ///
    /// Shared textures are read directly, so finish GPU writes first. Private and managed
    /// textures are copied with a blit on `commandQueue`, or on a new queue for the texture's
    /// device, and this call waits for the copy.
    ///
    /// - Throws: `Failure.unsupportedTextureFormat` for other pixel formats,
    ///   `Failure.unsupportedTexture` for non-2D, multisample, memoryless, or framebuffer-only
    ///   textures, and `Failure.wrongDevice` when `commandQueue` uses another device.
    public func pngData(from texture: any MTLTexture, interpretation: TextureLoader.Interpretation = .encodedSRGB) throws -> Data {
        var reader = GLTFTextureReader(queue: commandQueue)
        return try reader.png(texture, color: interpretation != .linearData)
    }
}

/// Reads textures for one export and reuses one command queue per device.
struct GLTFTextureReader {
    private typealias Failure = GLTFExporter.Failure

    private let suppliedQueue: (any MTLCommandQueue)?
    private var queues: [ObjectIdentifier: any MTLCommandQueue] = [:]

    init(queue: (any MTLCommandQueue)?) {
        suppliedQueue = queue
    }

    mutating func queue(for device: any MTLDevice) throws -> any MTLCommandQueue {
        if let suppliedQueue {
            guard suppliedQueue.device === device else { throw Failure.wrongDevice }
            return suppliedQueue
        }
        if let existing = queues[ObjectIdentifier(device)] { return existing }
        guard let queue = device.makeCommandQueue() else { throw Failure.resource("export command queue") }
        queues[ObjectIdentifier(device)] = queue
        return queue
    }

    mutating func png(_ texture: any MTLTexture, color: Bool) throws -> Data {
        let layout = try Layout(texture.pixelFormat)
        let width = texture.width, height = texture.height
        let source = try levelZero(texture, bytesPerPixel: layout.bytesPerPixel)
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        source.withUnsafeBytes { raw in
            for pixel in 0..<(width * height) {
                let rgba = layout.decode(raw, at: pixel * layout.bytesPerPixel, color: color)
                for lane in 0..<4 { pixels[pixel * 4 + lane] = rgba[lane] }
            }
        }
        return try Self.encodePNG(pixels, width: width, height: height)
    }

    private mutating func levelZero(_ texture: any MTLTexture, bytesPerPixel: Int) throws -> [UInt8] {
        guard texture.textureType == .type2D, texture.sampleCount == 1 else {
            throw Failure.unsupportedTexture("Only single-sample 2D textures can be exported.")
        }
        guard !texture.isFramebufferOnly else {
            throw Failure.unsupportedTexture("Framebuffer-only textures cannot be read.")
        }
        let width = texture.width, height = texture.height
        let bytesPerRow = width * bytesPerPixel
        var bytes = [UInt8](repeating: 0, count: bytesPerRow * height)
        let region = MTLRegionMake2D(0, 0, width, height)
        switch texture.storageMode {
        case .shared:
            bytes.withUnsafeMutableBytes { raw in
                texture.getBytes(raw.baseAddress!, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
            }
        case .private, .managed:
            let queue = try queue(for: texture.device)
            guard let buffer = texture.device.makeBuffer(length: bytes.count, options: .storageModeShared),
                  let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder() else {
                throw Failure.resource("texture readback buffer")
            }
            command.label = "Frostfield glTF texture export"
            blit.copy(
                from: texture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                sourceSize: MTLSize(width: width, height: height, depth: 1), to: buffer, destinationOffset: 0,
                destinationBytesPerRow: bytesPerRow, destinationBytesPerImage: bytesPerRow * height
            )
            blit.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else { throw Failure.textureReadback }
            bytes.withUnsafeMutableBytes { raw in
                raw.baseAddress!.copyMemory(from: buffer.contents(), byteCount: raw.count)
            }
        case .memoryless:
            throw Failure.unsupportedTexture("Memoryless textures cannot be read.")
        @unknown default:
            throw Failure.unsupportedTexture("The texture storage mode cannot be read.")
        }
        return bytes
    }

    private static func encodePNG(_ pixels: [UInt8], width: Int, height: Int) throws -> Data {
        // glTF ignores embedded color profiles. The tag only helps other image viewers.
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let provider = CGDataProvider(data: Data(pixels) as CFData),
              let image = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
              ) else {
            throw Failure.imageEncoding
        }
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output as CFMutableData, "public.png" as CFString, 1, nil) else {
            throw Failure.imageEncoding
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw Failure.imageEncoding }
        return output as Data
    }

    /// Decodes one stored texel to the 8-bit code values written to the PNG.
    private enum Layout {
        case rgba8(sRGB: Bool, bgra: Bool)
        case rgba16Float
        case unorm8(channels: Int)
        case float16(channels: Int)

        init(_ format: MTLPixelFormat) throws {
            switch format {
            case .rgba8Unorm: self = .rgba8(sRGB: false, bgra: false)
            case .rgba8Unorm_srgb: self = .rgba8(sRGB: true, bgra: false)
            case .bgra8Unorm: self = .rgba8(sRGB: false, bgra: true)
            case .bgra8Unorm_srgb: self = .rgba8(sRGB: true, bgra: true)
            case .rgba16Float: self = .rgba16Float
            case .r8Unorm: self = .unorm8(channels: 1)
            case .rg8Unorm: self = .unorm8(channels: 2)
            case .r16Float: self = .float16(channels: 1)
            case .rg16Float: self = .float16(channels: 2)
            default: throw Failure.unsupportedTextureFormat(format)
            }
        }

        var bytesPerPixel: Int {
            switch self {
            case .rgba8: 4
            case .rgba16Float: 8
            case .unorm8(let channels): channels
            case .float16(let channels): channels * 2
            }
        }

        func decode(_ raw: UnsafeRawBufferPointer, at offset: Int, color: Bool) -> SIMD4<UInt8> {
            switch self {
            case .rgba8(let sRGB, let bgra):
                let stored = SIMD4(raw[offset], raw[offset + 1], raw[offset + 2], raw[offset + 3])
                let rgba = bgra ? SIMD4(stored.z, stored.y, stored.x, stored.w) : stored
                // Matching encodings copy bytes exactly; otherwise convert through linear values.
                if sRGB == color { return rgba }
                let table = sRGB ? GLTFColorTables.decodeSRGB : GLTFColorTables.encodeSRGB
                return SIMD4(table[Int(rgba.x)], table[Int(rgba.y)], table[Int(rgba.z)], rgba.w)
            case .unorm8(let channels):
                let red = raw[offset], green = channels > 1 ? raw[offset + 1] : 0
                if !color { return SIMD4(red, green, 0, 255) }
                let table = GLTFColorTables.encodeSRGB
                return SIMD4(table[Int(red)], table[Int(green)], table[0], 255)
            case .rgba16Float:
                let values = (0..<4).map { lane in
                    Float(Float16(bitPattern: raw.loadUnaligned(fromByteOffset: offset + lane * 2, as: UInt16.self)))
                }
                return Self.quantize(SIMD4(values[0], values[1], values[2], values[3]), color: color)
            case .float16(let channels):
                func lane(_ index: Int) -> Float {
                    index < channels
                        ? Float(Float16(bitPattern: raw.loadUnaligned(fromByteOffset: offset + index * 2, as: UInt16.self))) : 0
                }
                return Self.quantize(SIMD4(lane(0), lane(1), 0, 1), color: color)
            }
        }

        private static func quantize(_ value: SIMD4<Float>, color: Bool) -> SIMD4<UInt8> {
            func code(_ value: Float, encode: Bool) -> UInt8 {
                // NaN clamps to zero, as it does when sampled through saturate in the shaders.
                let clamped = value.isNaN ? 0 : min(max(value, 0), 1)
                let stored = encode ? GLTFColorTables.linearToSRGB(clamped) : clamped
                return UInt8((stored * 255).rounded())
            }
            return SIMD4(
                code(value.x, encode: color), code(value.y, encode: color), code(value.z, encode: color),
                code(value.w, encode: false)
            )
        }
    }
}

enum GLTFColorTables {
    /// Maps an 8-bit linear code to the nearest 8-bit sRGB code.
    static let encodeSRGB: [UInt8] = (0...255).map { UInt8((linearToSRGB(Float($0) / 255) * 255).rounded()) }
    /// Maps an 8-bit sRGB code to the nearest 8-bit linear code.
    static let decodeSRGB: [UInt8] = (0...255).map { UInt8((sRGBToLinear(Float($0) / 255) * 255).rounded()) }

    static func linearToSRGB(_ value: Float) -> Float {
        value <= 0.003_130_8 ? value * 12.92 : 1.055 * pow(value, 1 / 2.4) - 0.055
    }

    static func sRGBToLinear(_ value: Float) -> Float {
        value <= 0.040_45 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4)
    }
}
