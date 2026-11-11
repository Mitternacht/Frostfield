import Foundation
import Metal
import simd

/// Native image encoding and explicit sRGB transfer for linear pixel data.
public enum ImageUtilities {
    /// Encodes texture level zero as a base64 PNG data URL. See
    /// `GLTFExporter.pngData(from:interpretation:)` for formats, color and GPU ordering.
    public static func pngDataURL(
        from texture: any MTLTexture, interpretation: TextureLoader.Interpretation = .encodedSRGB,
        commandQueue: (any MTLCommandQueue)? = nil
    ) throws -> String {
        let bytes = try GLTFExporter(commandQueue: commandQueue).pngData(from: texture, interpretation: interpretation)
        return "data:image/png;base64," + bytes.base64EncodedString()
    }

    /// Applies the sRGB transfer to RGB in a straight-alpha pixel. Alpha stays linear.
    public static func sRGBToLinear(_ rgba: SIMD4<Float>) -> SIMD4<Float> {
        SIMD4(
            GLTFColorTables.sRGBToLinear(rgba.x), GLTFColorTables.sRGBToLinear(rgba.y),
            GLTFColorTables.sRGBToLinear(rgba.z), rgba.w
        )
    }

    /// Converts RGB to linear codes rounded to the nearest 8-bit value. Alpha is unchanged.
    public static func sRGBToLinear(_ rgba: SIMD4<UInt8>) -> SIMD4<UInt8> {
        let table = GLTFColorTables.decodeSRGB
        return SIMD4(table[Int(rgba.x)], table[Int(rgba.y)], table[Int(rgba.z)], rgba.w)
    }
}
