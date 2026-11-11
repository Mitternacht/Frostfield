import Foundation
internal import FrostfieldBasis
import Metal

extension TextureLoader {
    /// Loads compressed KTX 1.0 data as a 2D texture or complete cube.
    /// Cubes store faces in +X, −X, +Y, −Y, +Z, −Z order within each mip.
    /// The loader uploads stored levels, or only level zero when requested.
    /// A declared sRGB encoding is retained in `.color` mode.
    public func loadKTX1(
        data: Data, interpretation: Interpretation = .color, loadMipmaps: Bool = true
    ) throws -> any MTLTexture {
        try loadKTX1Data(data, interpretation: interpretation, loadMipmaps: loadMipmaps, allowingCube: true)
    }

    /// Loads a regular local KTX 1.0 file. The caller supplies sandbox access.
    public func loadKTX1(
        contentsOf url: URL, interpretation: Interpretation = .color, loadMipmaps: Bool = true
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
        return try loadKTX1(data: data, interpretation: interpretation, loadMipmaps: loadMipmaps)
    }

    func loadKTX1Data(
        _ data: Data, interpretation: Interpretation, loadMipmaps: Bool, allowingCube: Bool
    ) throws -> any MTLTexture {
        guard data.count <= Self.maxEncodedBytes else { throw Failure.inputTooLarge }
        let container = try KTX1Container(data, allowingCube: allowingCube)
        let format = container.format
        let wantsSRGB = format.sRGB != nil && (interpretation == .encodedSRGB || (interpretation == .color && format.isSRGB))
        let uploadLevels = loadMipmaps ? container.levels[...] : container.levels.prefix(1)
        if let pvrtc = format.pvrtc {
            return try loadPVRTC1(container.data, levels: uploadLevels, format: pvrtc, sRGB: wantsSRGB,
                                  decodeFailure: .invalidKTX1("the PVRTC blocks could not be decoded"))
        }
        let bcFallback = device.supportsBCTextureCompression ? nil : format.bc
        let native = wantsSRGB ? (format.sRGB ?? format.linear) : format.linear
        let pixelFormat = bcFallback?.decodedPixelFormat(sRGB: wantsSRGB) ?? native
        if format.bc == nil, !device.supportsFamily(.apple2) {
            throw Failure.unsupportedTextureFormat(String(describing: pixelFormat))
        }

        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = container.faceCount == 6 ? .typeCube : .type2D
        descriptor.pixelFormat = pixelFormat
        descriptor.width = container.width
        descriptor.height = container.height
        descriptor.mipmapLevelCount = uploadLevels.count
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        if format.bc == FrostfieldBC1Opaque, bcFallback == nil {
            descriptor.swizzle = MTLTextureSwizzleChannels(red: .red, green: .green, blue: .blue, alpha: .one)
        }
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw Failure.textureAllocation(width: container.width, height: container.height)
        }

        try container.data.withUnsafeBytes { file in
            for (mip, level) in uploadLevels.enumerated() {
                let region = MTLRegionMake2D(0, 0, level.width, level.height)
                let storedRowBytes = format.bytesPerRow(width: level.width)
                for (face, range) in level.faces.enumerated() {
                    let bytes = UnsafeRawBufferPointer(rebasing: file[range])
                    func upload(_ pixels: UnsafeRawBufferPointer, rowBytes: Int) {
                        if container.faceCount == 1 {
                            texture.replace(region: region, mipmapLevel: mip,
                                            withBytes: pixels.baseAddress!, bytesPerRow: rowBytes)
                        } else {
                            texture.replace(region: region, mipmapLevel: mip, slice: face,
                                            withBytes: pixels.baseAddress!, bytesPerRow: rowBytes, bytesPerImage: 0)
                        }
                    }
                    if let bcFallback {
                        guard let pixels = bcFallback.decode(bytes, width: level.width, height: level.height) else {
                            throw Failure.invalidKTX1("the BC blocks could not be decoded")
                        }
                        pixels.withUnsafeBytes { upload($0, rowBytes: bcFallback.decodedRowBytes(width: level.width)) }
                    } else {
                        upload(bytes, rowBytes: storedRowBytes)
                    }
                }
            }
        }
        return texture
    }
}
