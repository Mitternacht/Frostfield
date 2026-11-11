import Foundation

extension TextureLoader {
    /// Pixel transforms for an ordinary decoded raster image.
    public struct UploadOptions: Equatable, Sendable {
        public enum AlphaMode: Equatable, Sendable {
            /// Keeps RGB independent of alpha.
            case straight
            /// Multiplies the stored RGB codes by alpha, matching browser uploads.
            case premultipliedEncoded
            /// Multiplies RGB by alpha in the texture's shader sampling space.
            /// sRGB textures decode, multiply, and re-encode their RGB channels.
            case premultipliedLinear
        }

        /// Reverses decoded rows after any orientation applied by the interpretation.
        public var flipY: Bool
        /// Alpha stays linear and unchanged. Use associated RGB with a consumer
        /// configured for that association.
        public var alphaMode: AlphaMode

        public init(flipY: Bool = false, alphaMode: AlphaMode = .straight) {
            self.flipY = flipY
            self.alphaMode = alphaMode
        }

        func validateStoredContainer(_ container: String) throws {
            guard self == .init() else { throw Failure.unsupportedUploadTransform(container) }
        }

        func apply(
            to rgba: UnsafeMutableRawBufferPointer, width: Int, height: Int, interpretation: Interpretation
        ) {
            if flipY, height > 1 {
                let rowBytes = width * 4
                let base = rgba.baseAddress!
                var scratch = [UInt8](repeating: 0, count: rowBytes)
                scratch.withUnsafeMutableBytes { row in
                    for y in 0..<(height / 2) {
                        let top = base.advanced(by: y * rowBytes)
                        let bottom = base.advanced(by: (height - 1 - y) * rowBytes)
                        row.copyMemory(from: UnsafeRawBufferPointer(start: top, count: rowBytes))
                        top.copyMemory(from: bottom, byteCount: rowBytes)
                        bottom.copyMemory(from: row.baseAddress!, byteCount: rowBytes)
                    }
                }
            }

            guard alphaMode != .straight else { return }
            if alphaMode == .premultipliedLinear, interpretation != .linearData {
                let table = Self.linearSRGBPremultiplication
                for offset in stride(from: 0, to: rgba.count, by: 4) where rgba[offset + 3] < 255 {
                    let row = Int(rgba[offset + 3]) * 256
                    for channel in offset..<(offset + 3) {
                        rgba[channel] = table[row + Int(rgba[channel])]
                    }
                }
            } else {
                for offset in stride(from: 0, to: rgba.count, by: 4) where rgba[offset + 3] < 255 {
                    let alpha = Int(rgba[offset + 3])
                    for channel in offset..<(offset + 3) {
                        rgba[channel] = UInt8((Int(rgba[channel]) * alpha + 127) / 255)
                    }
                }
            }
        }

        /// Indexed by alpha, then encoded RGB code. Reuse the same transfer
        /// functions as export while keeping full precision until quantization.
        private static let linearSRGBPremultiplication: [UInt8] = {
            let linear = (0...255).map { GLTFColorTables.sRGBToLinear(Float($0) / 255) }
            var codes: [UInt8] = []
            codes.reserveCapacity(256 * 256)
            for alpha in 0...255 {
                let factor = Float(alpha) / 255
                for value in linear {
                    let code = GLTFColorTables.linearToSRGB(value * factor) * 255
                    codes.append(UInt8(clamping: Int(code.rounded())))
                }
            }
            return codes
        }()
    }
}
