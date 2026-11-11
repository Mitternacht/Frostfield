import Foundation
import Metal

public extension EnvironmentLighting {
    /// Decodes an RGBE Radiance panorama and waits for lighting preparation.
    convenience init(
        queue: any MTLCommandQueue, radianceHDR: Data, settings: Settings = .init()
    ) throws {
        guard settings.isValid else { throw Failure.settings }
        let panorama = try Self.makeEquirectangularTexture(radianceHDR: radianceHDR, device: queue.device)
        try self.init(queue: queue, equirectangular: panorama, settings: settings)
    }

    /// Uploads the first Radiance RGBE image as top-left RGBA16Float. Encoded RGB
    /// is interpreted in the caller's linear working space. Exposure and color
    /// metadata are left to the caller. Values above 65504 saturate on upload.
    static func makeEquirectangularTexture(radianceHDR: Data, device: any MTLDevice) throws -> any MTLTexture {
        let image = try RadianceHDRImage.decode(radianceHDR)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: image.width, height: image.height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw Failure.resource("HDR panorama")
        }
        texture.label = "Frostfield Radiance HDR panorama"
        image.pixels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, image.width, image.height), mipmapLevel: 0,
                withBytes: bytes.baseAddress!, bytesPerRow: image.width * MemoryLayout<SIMD4<Float16>>.stride
            )
        }
        return texture
    }

    enum RadianceFailure: LocalizedError, Equatable {
        case size
        case header
        case format
        case scanline(Int)

        public var errorDescription: String? {
            switch self {
            case .size: "Use a Radiance 2:1 panorama within the input size limits."
            case .header: "The Radiance header or resolution line is invalid."
            case .format: "The panorama must use the Radiance 32-bit RGBE format."
            case .scanline(let row): "Radiance scanline \(row) is truncated or invalid."
            }
        }
    }
}

package struct RadianceHDRImage {
    package let width: Int
    package let height: Int
    package let pixels: [SIMD4<Float16>]

    private static let maximumBytes = 64 * 1_024 * 1_024
    private static let maximumPixels = 8_388_608
    private static let exponentScales = (0...255).map {
        Float(sign: .plus, exponent: $0 - 136, significand: 1)
    }

    /// Decodes a 2:1 panorama of at most 8,388,608 pixels (4,096 × 2,048) to half precision,
    /// clamping values above 65,504, for environment lighting.
    package static func decode(_ data: Data) throws -> Self {
        let image = try decode(data, validateSize: { width, height in
            guard (2...8_192).contains(width), (1...4_096).contains(height),
                  width == height * 2, width * height <= maximumPixels else {
                throw EnvironmentLighting.RadianceFailure.size
            }
        }, pixel: { SIMD4<Float16>(pointwiseMin(decodeLinear($0), SIMD4(repeating: Float(Float16.greatestFiniteMagnitude)))) })
        return RadianceHDRImage(width: image.width, height: image.height, pixels: image.pixels)
    }

    /// Decodes an image of any orientation and aspect ratio to single-precision values,
    /// without the half-float clamp. `validateSize` receives the width and height before
    /// any pixel is decoded and throws to reject them.
    package static func decodeLinear(
        _ data: Data, validateSize: (Int, Int) throws -> Void
    ) throws -> (width: Int, height: Int, pixels: [SIMD4<Float>]) {
        try decode(data, validateSize: validateSize, pixel: decodeLinear)
    }

    private static func decode<Pixel>(
        _ data: Data, validateSize: (Int, Int) throws -> Void, pixel: (SIMD4<UInt8>) -> Pixel
    ) throws -> (width: Int, height: Int, pixels: [Pixel]) {
        guard data.count <= maximumBytes else { throw EnvironmentLighting.RadianceFailure.size }
        return try data.withUnsafeBytes { raw in
            var reader = Reader(bytes: raw.bindMemory(to: UInt8.self))
            return try reader.readImage(validateSize: validateSize, pixel: pixel)
        }
    }

    /// The Radiance reference decoding (m + 0.5) · 2^(e − 136). Every result is exact in
    /// single precision: the largest is 255.5 · 2^119 and the smallest nonzero is 2^−136.
    private static func decodeLinear(_ pixel: SIMD4<UInt8>) -> SIMD4<Float> {
        guard pixel.w != 0 else { return SIMD4(0, 0, 0, 1) }
        let scale = exponentScales[Int(pixel.w)]
        return SIMD4((Float(pixel.x) + 0.5) * scale, (Float(pixel.y) + 0.5) * scale, (Float(pixel.z) + 0.5) * scale, 1)
    }

    private struct Reader {
        let bytes: UnsafeBufferPointer<UInt8>
        var offset = 0

        mutating func readImage<Pixel>(
            validateSize: (Int, Int) throws -> Void, pixel decode: (SIMD4<UInt8>) -> Pixel
        ) throws -> (width: Int, height: Int, pixels: [Pixel]) {
            let signature = try line()
            guard signature == "#?RADIANCE" || signature == "#?RGBE" else {
                throw EnvironmentLighting.RadianceFailure.header
            }
            var format: String?
            while true {
                let entry = try line()
                if entry.isEmpty { break }
                if entry.hasPrefix("FORMAT=") {
                    guard format == nil else { throw EnvironmentLighting.RadianceFailure.header }
                    format = String(entry.dropFirst("FORMAT=".count))
                }
            }
            guard format == "32-bit_rle_rgbe" else { throw EnvironmentLighting.RadianceFailure.format }

            let tokens = try line().split(whereSeparator: \.isWhitespace)
            guard tokens.count == 4, let majorCount = Int(tokens[1]), let minorCount = Int(tokens[3]),
                  let major = Axis(rawValue: String(tokens[0])), let minor = Axis(rawValue: String(tokens[2])),
                  major.isHorizontal != minor.isHorizontal else {
                throw EnvironmentLighting.RadianceFailure.header
            }

            let x = major.isHorizontal ? major : minor
            let y = major.isHorizontal ? minor : major
            let width = major.isHorizontal ? majorCount : minorCount
            let height = major.isHorizontal ? minorCount : majorCount
            try validateSize(width, height)

            var pixels = Array(repeating: decode(.zero), count: width * height)
            var scanline = Array(repeating: SIMD4<UInt8>.zero, count: minorCount)
            for row in 0..<majorCount {
                guard readScanline(into: &scanline) else {
                    throw EnvironmentLighting.RadianceFailure.scanline(row)
                }
                for (column, pixel) in scanline.enumerated() {
                    let inputX = major.isHorizontal ? row : column
                    let inputY = major.isHorizontal ? column : row
                    let outputX = x == .positiveX ? inputX : width - 1 - inputX
                    let outputY = y == .negativeY ? inputY : height - 1 - inputY
                    pixels[outputY * width + outputX] = decode(pixel)
                }
            }
            return (width, height, pixels)
        }

        private enum Axis: String {
            case positiveX = "+X"
            case negativeX = "-X"
            case positiveY = "+Y"
            case negativeY = "-Y"

            var isHorizontal: Bool { self == .positiveX || self == .negativeX }
        }

        private mutating func line() throws -> String {
            let start = offset
            let limit = min(bytes.count, 65_536)
            while offset < limit, bytes[offset] != 10 { offset += 1 }
            guard offset < limit else { throw EnvironmentLighting.RadianceFailure.header }
            var end = offset
            offset += 1
            if end > start, bytes[end - 1] == 13 { end -= 1 }
            return String(decoding: bytes[start..<end], as: UTF8.self)
        }

        private mutating func readScanline(into line: inout [SIMD4<UInt8>]) -> Bool {
            guard let first = readPixel() else { return false }
            if first.x == 2, first.y == 2, first.z < 128 {
                guard Int(first.z) * 256 + Int(first.w) == line.count else { return false }
                return readChannels(into: &line)
            }
            return readFlat(first: first, into: &line)
        }

        private mutating func readChannels(into line: inout [SIMD4<UInt8>]) -> Bool {
            for channel in 0..<4 {
                var index = 0
                while index < line.count {
                    guard let code = readByte(), code > 0 else { return false }
                    let run = code > 128
                    let count = run ? Int(code) - 128 : Int(code)
                    guard count <= line.count - index else { return false }
                    if run {
                        guard let value = readByte() else { return false }
                        for pixel in index..<index + count { line[pixel][channel] = value }
                    } else {
                        guard count <= bytes.count - offset else { return false }
                        for pixel in index..<index + count {
                            line[pixel][channel] = bytes[offset]
                            offset += 1
                        }
                    }
                    index += count
                }
            }
            return true
        }

        private mutating func readFlat(first: SIMD4<UInt8>, into line: inout [SIMD4<UInt8>]) -> Bool {
            guard !isRepeat(first) else { return false }
            line[0] = first
            var index = 1
            var shift = 0
            while index < line.count {
                guard let pixel = readPixel() else { return false }
                if isRepeat(pixel) {
                    guard shift <= 16 else { return false }
                    let count = Int(pixel.w) << shift
                    guard count <= line.count - index else { return false }
                    let previous = line[index - 1]
                    for repeatIndex in index..<index + count { line[repeatIndex] = previous }
                    index += count
                    shift += 8
                } else {
                    line[index] = pixel
                    index += 1
                    shift = 0
                }
            }
            return true
        }

        private func isRepeat(_ pixel: SIMD4<UInt8>) -> Bool {
            pixel.x == 1 && pixel.y == 1 && pixel.z == 1
        }

        private mutating func readPixel() -> SIMD4<UInt8>? {
            guard bytes.count - offset >= 4 else { return nil }
            let pixel = SIMD4(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3])
            offset += 4
            return pixel
        }

        private mutating func readByte() -> UInt8? {
            guard offset < bytes.count else { return nil }
            let byte = bytes[offset]
            offset += 1
            return byte
        }
    }
}
