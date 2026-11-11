import Foundation
internal import FrostfieldBasis
import Metal

extension TextureLoader {
    /// Loads a 2D texture or six-face cube from DDS data. Stored faces use
    /// Metal's +X, −X, +Y, −Y, +Z, −Z order with top-left rows.
    ///
    /// Select `loadMipmaps: false` to upload only the base level of each face.
    /// `mipmapped` can generate levels for a single uploaded uncompressed level.
    /// Block formats keep their selected stored mip levels. BC formats decode
    /// to RGBA8 or RGBA16Float when the device cannot sample BC directly.
    /// BC6H always samples as linear HDR, independent of `interpretation`.
    public func loadDDS(
        data: Data, interpretation: Interpretation = .color, loadMipmaps: Bool = true, mipmapped: Bool = false
    ) throws -> any MTLTexture {
        try loadDDSData(data, interpretation: interpretation, loadMipmaps: loadMipmaps, mipmapped: mipmapped,
                        allowingCube: true)
    }

    /// Loads a regular local DDS file. The caller supplies any sandbox access.
    public func loadDDS(
        contentsOf url: URL, interpretation: Interpretation = .color, loadMipmaps: Bool = true, mipmapped: Bool = false
    ) throws -> any MTLTexture {
        guard url.isFileURL else { throw Failure.localFileRequired }
        let data: Data
        do {
            data = try Self.readLocalFile(url)
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.fileRead(error.localizedDescription)
        }
        return try loadDDS(data: data, interpretation: interpretation, loadMipmaps: loadMipmaps, mipmapped: mipmapped)
    }

    func loadDDSData(
        _ data: Data, interpretation: Interpretation, loadMipmaps: Bool, mipmapped: Bool, allowingCube: Bool
    ) throws -> any MTLTexture {
        guard data.count <= Self.maxEncodedBytes else { throw Failure.inputTooLarge }
        let container = try DDSContainer(data, allowingCube: allowingCube)
        let format = container.format
        let sRGB = interpretation != .linearData
        let bcFallback = device.supportsBCTextureCompression ? nil : format.bcFallback
        if format == .etc1, !device.supportsFamily(.apple2) { throw Failure.unsupportedTextureFormat(format.rawValue) }
        let pixelFormat = bcFallback?.decodedPixelFormat(sRGB: sRGB) ?? format.pixelFormat(sRGB: sRGB)
        let uploadCount = loadMipmaps ? container.mipCount : 1
        let generates = mipmapped && uploadCount == 1 && !format.isCompressed
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = container.faceCount == 6 ? .typeCube : .type2D
        descriptor.pixelFormat = pixelFormat
        descriptor.width = container.width
        descriptor.height = container.height
        descriptor.mipmapLevelCount = generates
            ? Int.bitWidth - max(container.width, container.height).leadingZeroBitCount : uploadCount
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        if format == .bc1, bcFallback == nil {
            // Three.js DDSLoader exposes DXT1 as RGB, including in three-color blocks.
            descriptor.swizzle = MTLTextureSwizzleChannels(red: .red, green: .green, blue: .blue, alpha: .one)
        }
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw Failure.textureAllocation(width: container.width, height: container.height)
        }

        try container.data.withUnsafeBytes { file in
            for face in 0..<container.faceCount {
                for level in 0..<uploadCount {
                    let surface = container.surfaces[face * container.mipCount + level]
                    let bytes = UnsafeRawBufferPointer(rebasing: file[surface.range])
                    let region = MTLRegionMake2D(0, 0, surface.width, surface.height)
                    func upload(_ pixels: UnsafeRawBufferPointer, rowBytes: Int) {
                        if container.faceCount == 1 {
                            texture.replace(region: region, mipmapLevel: level,
                                            withBytes: pixels.baseAddress!, bytesPerRow: rowBytes)
                        } else {
                            texture.replace(region: region, mipmapLevel: level, slice: face,
                                            withBytes: pixels.baseAddress!, bytesPerRow: rowBytes, bytesPerImage: pixels.count)
                        }
                    }
                    if let bcFallback {
                        guard let pixels = bcFallback.decode(bytes, width: surface.width, height: surface.height) else {
                            throw Failure.invalidDDS("the BC blocks could not be decoded")
                        }
                        pixels.withUnsafeBytes { upload($0, rowBytes: bcFallback.decodedRowBytes(width: surface.width)) }
                    } else if format.isCompressed {
                        upload(bytes, rowBytes: surface.rowBytes)
                    } else {
                        let pixels = Self.ddsRGBA(bytes, surface: surface, pixelStride: format.bytesPerBlock)
                        pixels.withUnsafeBytes { upload($0, rowBytes: surface.width * 4) }
                    }
                }
            }
        }
        if generates, texture.mipmapLevelCount > 1 { try generateMipmaps(for: texture) }
        return texture
    }

    private static func ddsRGBA(
        _ bytes: UnsafeRawBufferPointer, surface: DDSContainer.Surface, pixelStride: Int
    ) -> [SIMD4<UInt8>] {
        var pixels = [SIMD4<UInt8>](repeating: .zero, count: surface.width * surface.height)
        for row in 0..<surface.height {
            for column in 0..<surface.width {
                let offset = row * surface.rowBytes + column * pixelStride
                pixels[row * surface.width + column] = SIMD4(
                    bytes[offset + 2], bytes[offset + 1], bytes[offset], pixelStride == 4 ? bytes[offset + 3] : 255
                )
            }
        }
        return pixels
    }
}

private extension DDSContainer.Format {
    var bcFallback: FrostfieldBCFormat? {
        switch self {
        case .bc1: FrostfieldBC1Opaque
        case .bc2: FrostfieldBC2
        case .bc3: FrostfieldBC3
        case .bc6HSigned: FrostfieldBC6HSigned
        case .bc6HUnsigned: FrostfieldBC6HUnsigned
        default: nil
        }
    }
}
