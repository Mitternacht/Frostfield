import Foundation

/// The Metal source of the Three.js r187 dithering functions, for shaders that Frostfield
/// or a test compiles at run time. `HDRDisplay` includes the same header at build time.
package enum DitheringShader {
    /// The text of `FrostfieldDithering.h`: `frostfieldR187Rand`, `frostfieldR187DitherShift`,
    /// `frostfieldR187Dithering` and `frostfieldR187DitherOutput`. It includes
    /// `<metal_stdlib>` and uses explicit `metal::` names, so it can follow any prologue.
    package static func metalSource() throws -> String {
        guard let url = Bundle.module.url(forResource: "FrostfieldDithering", withExtension: "h"),
              let source = try? String(contentsOf: url, encoding: .utf8) else {
            throw HDRDisplay.Failure.missingShader
        }
        return source
    }
}
