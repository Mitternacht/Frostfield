import Foundation

/// The noninterleaved TGA layouts supported by Three.js TGALoader.
struct TargaImage {
    static let imageIOType = "com.truevision.tga-image"

    private enum ColorLayout: Equatable {
        case indexed, gray, grayAlpha, rgb555, bgr, bgra

        var byteCount: Int {
            switch self {
            case .indexed, .gray: 1
            case .grayAlpha, .rgb555: 2
            case .bgr: 3
            case .bgra: 4
            }
        }
    }

    let width: Int
    let height: Int
    private let layout: ColorLayout
    private let runLengthEncoded: Bool
    private let rightOrigin: Bool
    private let topOrigin: Bool
    private let hasAlpha: Bool
    private let paletteOffset: Int
    private let paletteFirst: Int
    private let paletteCount: Int

    init?(_ data: Data) {
        let bytes = Array(data.prefix(18))
        guard bytes.count == 18, bytes[17] & 0xC0 == 0 else { return nil }
        switch (bytes[2], bytes[16]) {
        case (1, 8), (9, 8): layout = .indexed
        case (3, 8), (11, 8): layout = .gray
        case (3, 16), (11, 16): layout = .grayAlpha
        case (2, 16), (10, 16): layout = .rgb555
        case (2, 24), (10, 24): layout = .bgr
        case (2, 32), (10, 32): layout = .bgra
        default: return nil
        }

        paletteOffset = 18 + Int(bytes[0])
        paletteFirst = Int(bytes[3]) | Int(bytes[4]) << 8
        paletteCount = Int(bytes[5]) | Int(bytes[6]) << 8
        if layout == .indexed {
            guard bytes[1] == 1, bytes[7] == 24, paletteFirst < 256,
                  (1...256 - paletteFirst).contains(paletteCount) else { return nil }
        } else {
            guard bytes[1] == 0 else { return nil }
        }
        width = Int(bytes[12]) | Int(bytes[13]) << 8
        height = Int(bytes[14]) | Int(bytes[15]) << 8
        runLengthEncoded = bytes[2] >= 9
        rightOrigin = bytes[17] & 0x10 != 0
        topOrigin = bytes[17] & 0x20 != 0
        hasAlpha = bytes[17] & 0x0F != 0
    }

    /// ImageIO handles the common opaque left-origin variants. These cases need
    /// exact samples, palette addressing, packet decoding, or horizontal orientation.
    func requiresNativeDecode(interpretation: TextureLoader.Interpretation) -> Bool {
        rightOrigin || runLengthEncoded || layout == .grayAlpha || layout == .rgb555 ||
            (layout == .indexed && paletteFirst != 0) ||
            (layout == .bgra && hasAlpha && interpretation == .color)
    }

    func pixels(_ data: Data, clearTransparent: Bool, recognizedByImageIO: Bool) throws -> [SIMD4<UInt8>] {
        try TextureLoader.validate(width: width, height: height)
        let count = width * height
        let pixelOffset = paletteOffset + (layout == .indexed ? paletteCount * 3 : 0)
        return try data.withUnsafeBytes { bytes in
            let end = try Self.payloadEnd(bytes, pixelOffset: pixelOffset)
            guard pixelOffset <= end else { throw TextureLoader.Failure.invalidImage }
            let pixelBytes = layout.byteCount
            let minimumBytes = runLengthEncoded ? (count + 127) / 128 * (pixelBytes + 1) : count * pixelBytes
            guard minimumBytes <= end - pixelOffset else { throw TextureLoader.Failure.invalidImage }

            var result = [SIMD4<UInt8>](repeating: .zero, count: count)
            var source = pixelOffset
            var pixel = 0
            while pixel < count {
                var length = count
                var repeatPixel = false
                if runLengthEncoded {
                    guard source < end else { throw TextureLoader.Failure.invalidImage }
                    let packet = bytes[source]
                    source += 1
                    length = Int(packet & 0x7F) + 1
                    repeatPixel = packet & 0x80 != 0
                }
                let bytesInPacket = (repeatPixel ? 1 : length) * pixelBytes
                guard length <= count - pixel, bytesInPacket <= end - source else {
                    throw TextureLoader.Failure.invalidImage
                }
                let repeated = repeatPixel ? try sample(bytes, at: source) : .zero
                for i in 0..<length {
                    let rgba = repeatPixel ? repeated : try sample(bytes, at: source + i * pixelBytes)
                    let x = pixel % width, y = pixel / width
                    let column = rightOrigin ? width - x - 1 : x
                    let row = topOrigin ? y : height - y - 1
                    result[row * width + column] = clearTransparent && rgba.w == 0 ? .zero : rgba
                    pixel += 1
                }
                source += bytesInPacket
            }
            // Original TGA has no signature. Without one or ImageIO identification,
            // recognize only a complete legacy image with no bytes after its pixels.
            guard recognizedByImageIO || end != bytes.count || source == end else {
                throw TextureLoader.Failure.invalidImage
            }
            return result
        }
    }

    private static let footerSignature = Array("TRUEVISION-XFILE.\0".utf8)

    private static func payloadEnd(_ bytes: UnsafeRawBufferPointer, pixelOffset: Int) throws -> Int {
        guard bytes.count >= 26, bytes.suffix(18).elementsEqual(footerSignature) else { return bytes.count }
        let footer = bytes.count - 26
        var end = footer
        for field in [footer, footer + 4] {
            let offset = Int(UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: field, as: UInt32.self)))
            if offset == 0 { continue }
            guard offset >= pixelOffset, offset < footer else { throw TextureLoader.Failure.invalidImage }
            end = min(end, offset)
        }
        return end
    }

    private func sample(_ bytes: UnsafeRawBufferPointer, at start: Int) throws -> SIMD4<UInt8> {
        switch layout {
        case .indexed:
            let index = Int(bytes[start]) - paletteFirst
            guard index >= 0, index < paletteCount else { throw TextureLoader.Failure.invalidImage }
            let color = paletteOffset + index * 3
            return SIMD4(bytes[color + 2], bytes[color + 1], bytes[color], 255)
        case .gray, .grayAlpha:
            let gray = bytes[start]
            return SIMD4(gray, gray, gray, layout == .gray ? 255 : bytes[start + 1])
        case .rgb555:
            let packed = UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: start, as: UInt16.self))
            // Match r187's 5-bit expansion and its interpretation of bit 15.
            return SIMD4(
                UInt8((packed >> 10) & 31) << 3, UInt8((packed >> 5) & 31) << 3, UInt8(packed & 31) << 3,
                !hasAlpha || packed & 0x8000 == 0 ? 255 : 0
            )
        case .bgr, .bgra:
            let alpha = layout == .bgra && hasAlpha ? bytes[start + 3] : 255
            return SIMD4(bytes[start + 2], bytes[start + 1], bytes[start], alpha)
        }
    }
}
