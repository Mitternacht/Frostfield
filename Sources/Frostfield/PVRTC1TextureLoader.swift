import Foundation
internal import FrostfieldBasis
import Metal

struct StoredTextureLevel {
    let width: Int
    let height: Int
    /// Byte ranges ordered by Metal slice within this mip.
    let faces: [Range<Int>]
}

/// The raw values match the compressed pixel format IDs in the PVR v3 header.
enum PVRTC1Format: UInt32, CaseIterable {
    case rgb2, rgba2, rgb4, rgba4

    var isTwoBPP: Bool { self == .rgb2 || self == .rgba2 }
    var isOpaque: Bool { self == .rgb2 || self == .rgb4 }
    var minimumWidth: Int { isTwoBPP ? 16 : 8 }
    var minimumHeight: Int { 8 }

    /// PVRTC1 dimensions are powers of two, so only the minimum footprint needs padding.
    func byteCount(width: Int, height: Int) -> Int {
        max(width, minimumWidth) * max(height, minimumHeight) / (isTwoBPP ? 4 : 2)
    }

    func pixelFormat(sRGB: Bool) -> MTLPixelFormat {
        switch (self, sRGB) {
        case (.rgb2, false): .pvrtc_rgb_2bpp
        case (.rgb2, true): .pvrtc_rgb_2bpp_srgb
        case (.rgba2, false): .pvrtc_rgba_2bpp
        case (.rgba2, true): .pvrtc_rgba_2bpp_srgb
        case (.rgb4, false): .pvrtc_rgb_4bpp
        case (.rgb4, true): .pvrtc_rgb_4bpp_srgb
        case (.rgba4, false): .pvrtc_rgba_4bpp
        case (.rgba4, true): .pvrtc_rgba_4bpp_srgb
        }
    }
}

extension TextureLoader {
    /// Uploads validated KTX1 or PVR PVRTC1 surfaces.
    func loadPVRTC1(
        _ data: Data, levels: ArraySlice<StoredTextureLevel>, format: PVRTC1Format, sRGB: Bool,
        decodeFailure: Failure
    ) throws -> any MTLTexture {
        let base = levels[levels.startIndex]
        let isCube = base.faces.count == 6
        // Metal requires a square descriptor. For small mips the CPU decoder
        // explicitly crops the reconstructed minimum compression footprint.
        let fallback = !device.supportsFamily(.apple2) || base.width != base.height ||
            levels.contains { $0.width < format.minimumWidth || $0.height < format.minimumHeight }

        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = isCube ? .typeCube : .type2D
        descriptor.pixelFormat = fallback ? (sRGB ? .rgba8Unorm_srgb : .rgba8Unorm) : format.pixelFormat(sRGB: sRGB)
        descriptor.width = base.width
        descriptor.height = base.height
        descriptor.mipmapLevelCount = levels.count
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        // Some devices expose the encoded alpha endpoint for PVRTC RGB.
        if format.isOpaque, !fallback {
            descriptor.swizzle = MTLTextureSwizzleChannels(red: .red, green: .green, blue: .blue, alpha: .one)
        }
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw Failure.textureAllocation(width: base.width, height: base.height)
        }

        try data.withUnsafeBytes { file in
            for (mip, level) in levels.enumerated() {
                let region = MTLRegionMake2D(0, 0, level.width, level.height)
                for (face, range) in level.faces.enumerated() {
                    let bytes = UnsafeRawBufferPointer(rebasing: file[range])
                    func upload(_ pixels: UnsafeRawBufferPointer, rowBytes: Int) {
                        if isCube {
                            texture.replace(region: region, mipmapLevel: mip, slice: face,
                                            withBytes: pixels.baseAddress!, bytesPerRow: rowBytes, bytesPerImage: 0)
                        } else {
                            texture.replace(region: region, mipmapLevel: mip,
                                            withBytes: pixels.baseAddress!, bytesPerRow: rowBytes)
                        }
                    }
                    if fallback {
                        let rowBytes = level.width * 4
                        var pixels = Data(count: rowBytes * level.height)
                        let decoded = pixels.withUnsafeMutableBytes { raw in
                            frostfield_basis_decode_pvrtc1(
                                bytes.baseAddress, bytes.count, UInt32(level.width), UInt32(level.height),
                                format.isTwoBPP ? 1 : 0, format.isOpaque ? 1 : 0, raw.baseAddress, raw.count
                            )
                        }
                        guard decoded == 1 else { throw decodeFailure }
                        pixels.withUnsafeBytes { upload($0, rowBytes: rowBytes) }
                    } else {
                        upload(bytes, rowBytes: 0)
                    }
                }
            }
        }
        return texture
    }
}
