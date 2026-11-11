import Foundation
import Metal
import simd

/// UV fitting and tightly packed texel storage for native textures.
public enum TextureUtilities {
    public enum FitMode: Sendable {
        /// Fills the surface while preserving aspect ratio, cropping the image around its center.
        case cover
        /// Fits the entire image at the center while preserving aspect ratio.
        case contain
        /// Maps the full image over the surface, allowing different aspect ratios.
        case fill
    }

    /// Maps top-left surface UV to source UV. With `contain`, the unused bars
    /// map outside 0...1. Mask those UVs when the bars need a separate color.
    /// Both aspect ratios are width divided by height and must be finite and positive.
    public static func fit(
        imageAspectRatio: Double, surfaceAspectRatio: Double, mode: FitMode
    ) throws -> PhysicallyBasedMaterial.TextureTransform {
        guard imageAspectRatio.isFinite, imageAspectRatio > 0,
              surfaceAspectRatio.isFinite, surfaceAspectRatio > 0 else { throw Failure.aspectRatio }
        let ratio = surfaceAspectRatio / imageAspectRatio
        let scale: SIMD2<Float> = switch mode {
        case .cover: ratio < 1 ? SIMD2(Float(ratio), 1) : SIMD2(1, Float(1 / ratio))
        case .contain: ratio < 1 ? SIMD2(1, Float(1 / ratio)) : SIMD2(Float(ratio), 1)
        case .fill: SIMD2(repeating: 1)
        }
        guard scale.x.isFinite, scale.y.isFinite, scale.x > 0, scale.y > 0 else { throw Failure.aspectRatio }
        return try .init(offset: (SIMD2<Float>(repeating: 1) - scale) * 0.5, scale: scale)
    }

    /// Uses the texture's level-zero width and height.
    public static func fit(
        _ texture: any MTLTexture, surfaceAspectRatio: Double, mode: FitMode
    ) throws -> PhysicallyBasedMaterial.TextureTransform {
        try fit(imageAspectRatio: Double(texture.width) / Double(texture.height), surfaceAspectRatio: surfaceAspectRatio, mode: mode)
    }

    /// Bytes in one tightly packed 2D mip level or slice. This excludes
    /// row alignment, allocator overhead, tile memory and other subresources.
    /// Block formats round up to whole blocks. PVRTC1 requires power-of-two
    /// dimensions and includes its minimum 2×2 block footprint.
    /// Format geometry is independent of whether a particular device can allocate it.
    public static func byteCount(width: Int, height: Int, pixelFormat: MTLPixelFormat) throws -> Int {
        guard width > 0, height > 0 else { throw Failure.dimensions }
        let layout: ArchiveTextureCodec.Layout
        let paddedWidth: Int, paddedHeight: Int
        if let pvrtc = pvrtcFormats[pixelFormat] {
            guard width.nonzeroBitCount == 1, height.nonzeroBitCount == 1 else { throw Failure.pvrtcDimensions }
            paddedWidth = max(width, pvrtc.minimumWidth)
            paddedHeight = max(height, pvrtc.minimumHeight)
            layout = .init(blockWidth: pvrtc.minimumWidth / 2, blockHeight: pvrtc.minimumHeight / 2, bytesPerBlock: 8)
        } else {
            guard let known = ArchiveTextureCodec.layout(pixelFormat) ?? additionalLayouts[pixelFormat] else {
                throw Failure.pixelFormat(pixelFormat)
            }
            paddedWidth = width
            paddedHeight = height
            layout = known
        }
        let columns = 1 + (paddedWidth - 1) / layout.blockWidth
        let rows = 1 + (paddedHeight - 1) / layout.blockHeight
        let (blocks, blockOverflow) = columns.multipliedReportingOverflow(by: rows)
        let (bytes, byteOverflow) = blocks.multipliedReportingOverflow(by: layout.bytesPerBlock)
        guard !blockOverflow, !byteOverflow else { throw Failure.byteCountOverflow }
        return bytes
    }

    private static let pvrtcFormats = Dictionary(uniqueKeysWithValues: PVRTC1Format.allCases.flatMap { format in
        [(format.pixelFormat(sRGB: false), format), (format.pixelFormat(sRGB: true), format)]
    })

    private static let additionalLayouts: [MTLPixelFormat: ArchiveTextureCodec.Layout] = {
        let r11 = ArchiveTextureCodec.Layout(blockWidth: 4, blockHeight: 4, bytesPerBlock: 8)
        var layouts: [MTLPixelFormat: ArchiveTextureCodec.Layout] = [.eac_r11Unorm: r11, .eac_r11Snorm: r11]
        let packed: [(MTLPixelFormat, Int)] = [
            (.stencil8, 1), (.depth16Unorm, 2), (.depth32Float, 4),
            (.bgr10_xr, 4), (.bgr10_xr_srgb, 4), (.bgra10_xr, 8), (.bgra10_xr_srgb, 8),
        ]
        for (format, size) in packed {
            layouts[format] = .init(blockWidth: 1, blockHeight: 1, bytesPerBlock: size)
        }
        for footprint in KTX2Format.astc {
            layouts[footprint.hdr] = .init(blockWidth: footprint.width, blockHeight: footprint.height, bytesPerBlock: 16)
        }
        return layouts
    }()

    public enum Failure: LocalizedError, Equatable {
        case aspectRatio, dimensions, pvrtcDimensions, byteCountOverflow
        case pixelFormat(MTLPixelFormat)

        public var errorDescription: String? {
            switch self {
            case .aspectRatio: "Use finite positive aspect ratios that produce a representable UV scale."
            case .dimensions: "Texture dimensions must be positive."
            case .pvrtcDimensions: "PVRTC1 texture dimensions must be powers of two."
            case .byteCountOverflow: "The packed texture byte count exceeds Int."
            case .pixelFormat(let format): "The packed layout for Metal format \(format.rawValue) is unavailable."
            }
        }
    }
}

/// IEEE 754 binary16 bit conversion using Swift's native half-precision type.
public enum Binary16 {
    /// Rounds to nearest with ties to even, preserving signed zero, infinities
    /// and NaNs. Finite overflow rounds according to IEEE 754.
    public static func encode(_ value: Float) -> UInt16 {
        Float16(value).bitPattern
    }

    /// Converts a binary16 bit pattern to Float. NaN payloads can be quieted
    /// by the floating-point conversion.
    public static func decode(_ bits: UInt16) -> Float {
        Float(Float16(bitPattern: bits))
    }
}
