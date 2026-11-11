import Foundation
internal import FrostfieldBasis
import Metal

extension FrostfieldBCFormat {
    private var bytesPerTexel: Int {
        self == FrostfieldBC6HSigned || self == FrostfieldBC6HUnsigned ? 8 : 4
    }

    func decodedPixelFormat(sRGB: Bool) -> MTLPixelFormat {
        switch self {
        case FrostfieldBC6HSigned, FrostfieldBC6HUnsigned: .rgba16Float
        case FrostfieldBC4Signed, FrostfieldBC5Signed: .rgba8Snorm
        case FrostfieldBC4, FrostfieldBC5: .rgba8Unorm
        default: sRGB ? .rgba8Unorm_srgb : .rgba8Unorm
        }
    }

    func decodedRowBytes(width: Int) -> Int { width * bytesPerTexel }

    /// Expands a validated slice to a tightly packed Metal-compatible format.
    func decode(_ blocks: UnsafeRawBufferPointer, width: Int, height: Int) -> Data? {
        var result = Data(count: decodedRowBytes(width: width) * height)
        let decoded = result.withUnsafeMutableBytes { bytes in
            frostfield_basis_decode_bc(
                self, blocks.baseAddress, blocks.count, UInt32(width), UInt32(height),
                bytes.baseAddress, bytes.count
            )
        }
        return decoded == 1 ? result : nil
    }
}
