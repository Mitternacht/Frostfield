import CoreGraphics
import Foundation
import ImageIO

extension ColorLookupTable {
    /// Reads a horizontal `N² × N` or vertical `N × N²` 3D lookup strip.
    ///
    /// Red varies across each square, green down it, and blue by square. Set
    /// `flipVertically` to read source rows bottom-to-top before extracting the
    /// squares. RGB samples map to the unit domain as numeric values without
    /// applying image color profiles or a transfer function.
    ///
    /// Accepts noninterlaced 8-bit PNG (grayscale, RGB, alpha variants, or
    /// indexed color) and single-frame 8-bit opaque RGB or grayscale images
    /// that ImageIO can decode. PNG alpha is ignored. EXIF orientation is ignored.
    public init(image data: Data, flipVertically: Bool = false) throws {
        guard data.count <= TextureLoader.maxEncodedBytes else { throw Failure.imageInputTooLarge }
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithData(data as CFData, options),
              CGImageSourceGetCount(source) == 1,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, options) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int else { throw Failure.imageDecode }

        let size = min(width, height)
        guard (2...Self.maximumSize).contains(size) else { throw Failure.size(size) }
        guard max(width, height) == size * size else { throw Failure.imageLayout(width: width, height: height) }

        let samples: [UInt8]
        let channels: Int
        if ImageSignature(data) == .png {
            guard let rgb = PNGChannels.readRGB8(data, width: width, height: height) else { throw Failure.imageDecode }
            samples = rgb
            channels = 3
        } else {
            guard let image = CGImageSourceCreateImageAtIndex(source, 0, options),
                  image.width == width, image.height == height,
                  let decoded = Self.decodeOpaqueImage(image) else { throw Failure.imageDecode }
            samples = decoded.samples
            channels = decoded.channels
        }

        let horizontal = width > height
        var values: [SIMD3<Float>] = []
        values.reserveCapacity(size * size * size)
        for blue in 0..<size {
            for green in 0..<size {
                let sourceRow = horizontal ? green : blue * size + green
                let row = flipVertically ? height - 1 - sourceRow : sourceRow
                let start = row * width + (horizontal ? blue * size : 0)
                for red in 0..<size {
                    let i = (start + red) * channels
                    if channels == 1 {
                        values.append(SIMD3(repeating: Float(samples[i]) / 255))
                    } else {
                        values.append(SIMD3(Float(samples[i]), Float(samples[i + 1]), Float(samples[i + 2])) / 255)
                    }
                }
            }
        }
        try self.init(size: size, values: values)
    }

    /// Reads a bounded regular local lookup-strip image.
    public init(imageContentsOf url: URL, flipVertically: Bool = false) throws {
        guard url.isFileURL else { throw TextureLoader.Failure.localFileRequired }
        let data: Data
        do {
            data = try TextureLoader.readLocalFile(url)
        } catch TextureLoader.Failure.inputTooLarge {
            throw Failure.imageInputTooLarge
        }
        try self.init(image: data, flipVertically: flipVertically)
    }

    private static func decodeOpaqueImage(_ image: CGImage) -> (samples: [UInt8], channels: Int)? {
        guard image.bitsPerComponent == 8, !image.isMask, image.decode == nil,
              [.none, .noneSkipFirst, .noneSkipLast].contains(image.alphaInfo) else { return nil }
        let grayscale = image.colorSpace?.model == .monochrome
        guard grayscale || image.colorSpace?.model == .rgb else { return nil }

        let space = grayscale ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = grayscale ? CGImageAlphaInfo.none.rawValue :
            CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue
        let channels = grayscale ? 1 : 4
        let rowBytes = image.width * channels
        var samples = [UInt8](repeating: 0, count: rowBytes * image.height)
        // Retag the source and destination identically to preserve decoded
        // components when changing the bitmap layout.
        let decoded = samples.withUnsafeMutableBytes { bytes in
            guard let source = image.copy(colorSpace: space),
                  let context = CGContext(data: bytes.baseAddress, width: image.width, height: image.height,
                                          bitsPerComponent: 8, bytesPerRow: rowBytes, space: space,
                                          bitmapInfo: bitmapInfo) else { return false }
            context.setBlendMode(.copy)
            context.draw(source, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            return true
        }
        return decoded ? (samples, channels) : nil
    }
}
