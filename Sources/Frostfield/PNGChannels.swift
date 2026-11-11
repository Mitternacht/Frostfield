import Foundation
import zlib

/// Reads numeric samples without applying color profiles or premultiplying.
enum PNGChannels {
    private enum Layout: UInt8 {
        case gray = 0, rgb = 2, indexed = 3, grayAlpha = 4, rgba = 6

        var channelCount: Int {
            switch self {
            case .gray, .indexed: 1
            case .grayAlpha: 2
            case .rgb: 3
            case .rgba: 4
            }
        }
    }

    private struct ImageData {
        let layout: Layout
        let compressed: Data
        let palette: [UInt8]
    }

    /// Restores straight RGBA8 when ImageIO premultiplies a PNG on decode.
    static func copyRGBA8(
        _ png: Data, width: Int, height: Int, into output: UnsafeMutableRawBufferPointer
    ) -> Bool {
        guard let image = imageData(in: png, width: width, height: height, expectedLayout: .rgba) else {
            return false
        }
        return copySamples(image, width: width, height: height, into: output)
    }

    /// Reads an 8-bit PNG as RGB triples, retaining RGB values under transparent pixels.
    static func readRGB8(_ png: Data, width: Int, height: Int) -> [UInt8]? {
        guard let image = imageData(in: png, width: width, height: height) else { return nil }
        let channels = image.layout.channelCount
        let count = width * height
        var samples = [UInt8](repeating: 0, count: count * channels)
        guard samples.withUnsafeMutableBytes({ copySamples(image, width: width, height: height, into: $0) }) else {
            return nil
        }
        if image.layout == .rgb { return samples }

        var rgb: [UInt8] = []
        rgb.reserveCapacity(count * 3)
        for offset in stride(from: 0, to: samples.count, by: channels) {
            switch image.layout {
            case .gray, .grayAlpha:
                let gray = samples[offset]
                rgb.append(gray)
                rgb.append(gray)
                rgb.append(gray)
            case .indexed:
                let index = Int(samples[offset]) * 3
                guard index + 3 <= image.palette.count else { return nil }
                rgb.append(contentsOf: image.palette[index..<(index + 3)])
            case .rgb, .rgba:
                rgb.append(contentsOf: samples[offset..<(offset + 3)])
            }
        }
        return rgb
    }

    private static func copySamples(
        _ image: ImageData, width: Int, height: Int, into output: UnsafeMutableRawBufferPointer
    ) -> Bool {
        let channels = image.layout.channelCount
        let rowBytes = width * channels
        let filteredStride = rowBytes + 1
        guard output.count == rowBytes * height, let destination = output.baseAddress else { return false }
        var filtered = [UInt8](repeating: 0, count: filteredStride * height)
        var decodedLength = uLongf(filtered.count)
        let result = filtered.withUnsafeMutableBufferPointer { target in
            image.compressed.withUnsafeBytes { source in
                uncompress(target.baseAddress, &decodedLength, source.bindMemory(to: UInt8.self).baseAddress, uLong(source.count))
            }
        }
        guard result == Z_OK, decodedLength == filtered.count,
              stride(from: 0, to: filtered.count, by: filteredStride).allSatisfy({ filtered[$0] <= 4 }) else { return false }
        return filtered.withUnsafeBytes { source in
            let pixels = output.bindMemory(to: UInt8.self)
            for row in 0..<height {
                let filterIndex = row * filteredStride
                let filter = source[filterIndex]
                let pixelRow = row * rowBytes
                if filter == 0 {
                    destination.advanced(by: pixelRow).copyMemory(
                        from: source.baseAddress!.advanced(by: filterIndex + 1), byteCount: rowBytes
                    )
                    continue
                }
                for column in 0..<rowBytes {
                    let index = pixelRow + column
                    let left = column >= channels ? pixels[index - channels] : 0
                    let up = row > 0 ? pixels[index - rowBytes] : 0
                    let upperLeft = row > 0 && column >= channels ? pixels[index - rowBytes - channels] : 0
                    let predictor: UInt8
                    switch filter {
                    case 1: predictor = left
                    case 2: predictor = up
                    case 3: predictor = UInt8((Int(left) + Int(up)) / 2)
                    default: predictor = paeth(left, up, upperLeft)
                    }
                    pixels[index] = source[filterIndex + 1 + column] &+ predictor
                }
            }
            return true
        }
    }

    /// Recognizes static, noninterlaced PNG with 8-bit samples or indices.
    private static func imageData(
        in png: Data, width: Int, height: Int, expectedLayout: Layout? = nil
    ) -> ImageData? {
        guard png.count <= TextureLoader.maxEncodedBytes, ImageSignature(png) == .png,
              (try? TextureLoader.validate(width: width, height: height)) != nil else { return nil }
        return png.withUnsafeBytes { bytes in
            func word(_ offset: Int) -> UInt32 {
                UInt32(bigEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
            }
            func validCRC(at offset: Int, length: Int) -> Bool {
                let start = bytes.bindMemory(to: UInt8.self).baseAddress?.advanced(by: offset + 4)
                return crc32(0, start, UInt32(length + 4)) == uLong(word(offset + 8 + length))
            }

            let ihdr: UInt32 = 0x4948_4452, idat: UInt32 = 0x4944_4154, iend: UInt32 = 0x4945_4E44
            let plte: UInt32 = 0x504C_5445
            guard bytes.count >= 33, word(8) == 13, word(12) == ihdr,
                  word(16) == width, word(20) == height, bytes[24] == 8, let layout = Layout(rawValue: bytes[25]),
                  expectedLayout == nil || layout == expectedLayout,
                  bytes[26] == 0, bytes[27] == 0, bytes[28] == 0,
                  validCRC(at: 8, length: 13) else { return nil }

            var offset = 33
            var compressed = Data()
            var palette: [UInt8] = []
            var sawData = false
            var finishedData = false
            while offset <= bytes.count - 12 {
                let length = Int(word(offset))
                guard length <= bytes.count - offset - 12 else { return nil }
                let type = word(offset + 4)
                // The first animation frame can differ from the default IDAT image.
                if type == 0x6163_544C || type == 0x6663_544C || type == 0x6664_4154 { return nil }
                if type == iend {
                    guard length == 0, !compressed.isEmpty, validCRC(at: offset, length: 0),
                          layout != .indexed || !palette.isEmpty else { return nil }
                    return ImageData(layout: layout, compressed: compressed, palette: palette)
                }
                if type == idat {
                    guard !finishedData, validCRC(at: offset, length: length) else { return nil }
                    compressed.append(contentsOf: bytes[(offset + 8)..<(offset + 8 + length)])
                    sawData = true
                } else {
                    // Other unknown critical chunks need ImageIO's interpretation.
                    guard bytes[offset + 4] & 0x20 != 0 || type == plte else { return nil }
                    if type == plte, layout == .indexed {
                        guard !sawData, palette.isEmpty, (3...768).contains(length),
                              length.isMultiple(of: 3), validCRC(at: offset, length: length) else { return nil }
                        palette = Array(bytes[(offset + 8)..<(offset + 8 + length)])
                    }
                    finishedData = sawData
                }
                offset += length + 12
            }
            return nil
        }
    }

    private static func paeth(_ left: UInt8, _ up: UInt8, _ upperLeft: UInt8) -> UInt8 {
        let a = Int(left), b = Int(up), c = Int(upperLeft)
        let da = abs(b - c), db = abs(a - c), dc = abs(a + b - 2 * c)
        if da <= db, da <= dc { return left }
        return db <= dc ? up : upperLeft
    }
}
