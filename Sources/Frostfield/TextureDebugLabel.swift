import CoreGraphics
import CoreText
import Foundation
import Metal

extension TextureDebugPass {
    /// A one-line Core Text label prepared once for repeated overlay draws.
    public struct Label {
        let mask: any MTLTexture
        public var width: Int { mask.width }
        public var height: Int { mask.height }

        /// Uses a 14-point system font by default. Supply another font for a
        /// different pixel size. The label has two pixels of clear padding.
        public init(_ text: String, device: any MTLDevice, font: CTFont? = nil) throws {
            guard !text.isEmpty, text.utf16.count <= 1_024, !text.contains(where: \.isNewline) else {
                throw Failure.labelText
            }
            let font = font ?? CTFontCreateUIFontForLanguage(.system, 14, nil)
                ?? CTFontCreateWithName("Helvetica" as CFString, 14, nil)
            let attributed = NSAttributedString(string: text, attributes: [
                NSAttributedString.Key(kCTFontAttributeName as String): font,
                NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(gray: 1, alpha: 1),
            ])
            let line = CTLineCreateWithAttributedString(attributed)
            let bounds = CTLineGetImageBounds(line, nil).integral
            guard bounds.minX.isFinite, bounds.minY.isFinite,
                  (1...4_092).contains(bounds.width), (1...1_020).contains(bounds.height) else {
                throw Failure.labelText
            }
            let width = Int(bounds.width) + 4
            let height = Int(bounds.height) + 4
            guard let space = CGColorSpace(name: CGColorSpace.sRGB),
                  let bitmap = CGContext(
                    data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                    space: space, bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
                  ), let bytes = bitmap.data else { throw Failure.labelResource }
            bitmap.clear(CGRect(x: 0, y: 0, width: width, height: height))
            bitmap.setShouldSmoothFonts(false)
            bitmap.textPosition = CGPoint(x: 2 - bounds.minX, y: 2 - bounds.minY)
            CTLineDraw(line, bitmap)
            let rgba = bytes.assumingMemoryBound(to: UInt8.self)
            var coverage = [UInt8](repeating: 0, count: width * height)
            for y in 0..<height {
                for x in 0..<width {
                    coverage[y * width + x] = rgba[y * bitmap.bytesPerRow + 4 * x + 3]
                }
            }
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .r8Unorm, width: width, height: height, mipmapped: false
            )
            descriptor.storageMode = .shared
            descriptor.usage = .shaderRead
            guard let mask = device.makeTexture(descriptor: descriptor) else { throw Failure.labelResource }
            mask.label = "Frostfield texture viewer label"
            coverage.withUnsafeBytes {
                mask.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                             withBytes: $0.baseAddress!, bytesPerRow: width)
            }
            self.mask = mask
        }
    }
}
