import CoreText
import Foundation

/// Loads TrueType and OpenType font files, including collections, into Core Text fonts.
///
/// This is the native stand-in for Three.js `TTFLoader` and `FontLoader`. Pass the returned
/// font to ``TextOutline`` or `MeshGeometry.text`. Typeface.js JSON is not a container here:
/// Core Text reads the font bytes directly.
public enum FontFile {
    public enum Failure: LocalizedError, Equatable {
        case size
        case unrecognized

        public var errorDescription: String? {
            switch self {
            case .size: "The font size must be finite and positive."
            case .unrecognized: "Core Text did not find a font in the data."
            }
        }
    }

    /// Every face in the file, at `size` points. The point size selects the face's optical size.
    /// ``TextOutline`` still scales the em into world units separately.
    public static func fonts(data: Data, size: CGFloat = 12) throws -> [CTFont] {
        guard size.isFinite, size > 0 else { throw Failure.size }
        guard let descriptors = CTFontManagerCreateFontDescriptorsFromData(data as CFData) as? [CTFontDescriptor],
              !descriptors.isEmpty else {
            throw Failure.unrecognized
        }
        return descriptors.map { CTFontCreateWithFontDescriptor($0, size, nil) }
    }
}
