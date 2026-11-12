import CoreGraphics
import Foundation
internal import FrostfieldBasis
import ImageIO
import Metal

/// Loads the first image from encoded data as a single-level sRGB color texture.
/// ImageIO rows start at the top left after applying image orientation. RGB samples are
/// linear in the shader, with straight alpha. Color-managed decoding clears RGB
/// at zero alpha. Data interpretations can preserve those encoded RGB values.
/// Pass `UploadOptions` to flip the decoded rows or associate RGB with alpha
/// before uploading an ordinary raster image.
///
/// ImageIO decodes PNG, JPEG, and the other formats it supports at runtime, including
/// WebP and AVIF on current systems; a WebP or AVIF this system cannot decode fails
/// with `Failure.unsupportedImageFormat`. TIFF uses the first ImageIO image.
/// TGA uses a native fallback for layouts that ImageIO cannot decode accurately.
/// DDS, KTX 1.0, and PVR textures retain stored mip levels. Use the corresponding
/// `loadDDS`, `loadKTX1`, or `loadPVR` method for cubes.
/// KTX2 containers upload their stored texel
/// blocks and mip levels directly: R, RG, RGBA and BGRA 8-bit UNORM or sRGB,
/// R, RG and RGBA Float16 and Float32, and the
/// BC1 RGBA, BC3, BC7, ETC2, and LDR ASTC block formats the device can sample,
/// stored uncompressed or with zlib supercompression. ETC1S and UASTC LDR Basis
/// textures decode to RGBA8 by default, including Zstandard-compressed UASTC levels.
/// Linear HDR Basis profiles decode to RGBA16Float with values above one preserved.
/// Use `loadKTX2(data:interpretation:mipmapped:)` for complete cubes, 2D
/// arrays, and uncompressed 3D textures.
///
/// OpenEXR and Radiance HDR files are scene-linear by definition, so every interpretation
/// loads them as linear RGBA16Float through `loadHighDynamicRange(data:precision:mipmapped:)`.
/// Gain-map JPEG and HEIC images load their SDR base here; decode their HDR rendition
/// with `loadHighDynamicRange`.
public struct TextureLoader {
    /// The linear primaries a color-managed image is converted into. Stored containers keep their own encoding.
    public enum WorkingPrimaries: Equatable, Sendable {
        case sRGB
        case displayP3
        case rec2020

        var linearSpaceName: CFString {
            switch self {
            case .sRGB: CGColorSpace.extendedLinearSRGB
            case .displayP3: CGColorSpace.extendedLinearDisplayP3
            case .rec2020: CGColorSpace.extendedLinearITUR_2020
            }
        }
    }

    public enum Interpretation: Equatable, Sendable {
        /// Applies image orientation and converts tagged color to sRGB.
        case color
        /// Treats encoded RGB values as sRGB, independent of source color metadata.
        case encodedSRGB
        /// Uploads the encoded data channels without an sRGB sampling transfer.
        case linearData
    }

    public enum BasisStorage: Equatable, Sendable {
        /// Decodes LDR Basis textures into uncompressed RGBA8.
        case rgba8
        /// Transcodes LDR Basis into a block format the Metal device can sample.
        /// A missing format or a request to generate extra mips throws.
        case deviceCompressed
    }

    let device: any MTLDevice

    static let maxEncodedBytes = 64 * 1_024 * 1_024
    static let maxDimension = 8_192
    static let maxPixels = 16_777_216

    public init(device: any MTLDevice) {
        self.device = device
    }

    /// Loads a regular local file. The caller supplies any required sandbox access.
    public func load(
        contentsOf url: URL, interpretation: Interpretation = .color, mipmapped: Bool = false
    ) throws -> any MTLTexture {
        try load(contentsOf: url, interpretation: interpretation, mipmapped: mipmapped, uploadOptions: .init())
    }

    /// Loads a regular local image file with explicit pixel transforms.
    public func load(
        contentsOf url: URL, interpretation: Interpretation = .color, mipmapped: Bool = false,
        uploadOptions: UploadOptions
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
        return try load(data: data, interpretation: interpretation, mipmapped: mipmapped, uploadOptions: uploadOptions)
    }

    /// Loads up to 64 MiB of encoded image data. The decoded image may be at most
    /// 8192 pixels per side and 16,777,216 pixels in total.
    ///
    /// For KTX2, `.color` keeps the container's sRGB or linear encoding, `.encodedSRGB`
    /// samples 8-bit and block formats as sRGB, and `.linearData` samples them without
    /// an sRGB transfer. Float data stays linear. Stored mip levels are uploaded
    /// as they are. `mipmapped` generates levels for single-level KTX2 textures
    /// when their pixel format supports mipmap generation.
    public func load(
        data: Data, interpretation: Interpretation = .color, mipmapped: Bool = false,
        workingPrimaries: WorkingPrimaries = .sRGB
    ) throws -> any MTLTexture {
        try load(
            data: data, interpretation: interpretation, mipmapped: mipmapped, uploadOptions: .init(),
            workingPrimaries: workingPrimaries
        )
    }

    /// Loads an ordinary raster image with optional row reversal and alpha association.
    /// When requested, mips are generated from the transformed base level.
    /// HDR and DDS, KTX, or PVR containers require default upload options.
    public func load(
        data: Data, interpretation: Interpretation = .color, mipmapped: Bool = false,
        uploadOptions: UploadOptions, workingPrimaries: WorkingPrimaries = .sRGB
    ) throws -> any MTLTexture {
        guard data.count <= Self.maxEncodedBytes else { throw Failure.inputTooLarge }
        if workingPrimaries != .sRGB, interpretation != .color {
            throw Failure.workingPrimaries
        }
        if HighDynamicRangeImage.Container(data) != nil {
            if workingPrimaries != .sRGB { throw Failure.workingPrimaries }
            try uploadOptions.validateStoredContainer("HDR")
            return try loadHighDynamicRange(data: data, precision: .half, mipmapped: mipmapped).texture
        }
        if KTX2Container.hasIdentifier(data) {
            if workingPrimaries != .sRGB { throw Failure.workingPrimaries }
            try uploadOptions.validateStoredContainer("KTX2")
            return try loadKTX2Data(data, interpretation: interpretation, mipmapped: mipmapped, allowingSpatialTextures: false)
        }
        if KTX1Container.hasIdentifier(data) {
            if workingPrimaries != .sRGB { throw Failure.workingPrimaries }
            try uploadOptions.validateStoredContainer("KTX 1.0")
            return try loadKTX1Data(data, interpretation: interpretation, loadMipmaps: true, allowingCube: false)
        }
        if PVRContainer.hasIdentifier(data) {
            if workingPrimaries != .sRGB { throw Failure.workingPrimaries }
            try uploadOptions.validateStoredContainer("PVR")
            return try loadPVRData(data, interpretation: interpretation, loadMipmaps: true, allowingCube: false)
        }
        if DDSContainer.hasIdentifier(data) {
            if workingPrimaries != .sRGB { throw Failure.workingPrimaries }
            try uploadOptions.validateStoredContainer("DDS")
            return try loadDDSData(data, interpretation: interpretation, loadMipmaps: true, mipmapped: mipmapped,
                                   allowingCube: false)
        }
        let format = ImageSignature(data)
        if let format, !format.isDecodable {
            throw Failure.unsupportedImageFormat(format.name)
        }
        let sourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary
        let imageSource = CGImageSourceCreateWithData(data as CFData, sourceOptions)
        let sourceType = imageSource.flatMap(CGImageSourceGetType).map { $0 as String }
        let targa = format == nil && (sourceType == nil || sourceType == TargaImage.imageIOType) ? TargaImage(data) : nil
        if let targa, sourceType == nil || targa.requiresNativeDecode(interpretation: interpretation) {
            if workingPrimaries != .sRGB { throw Failure.workingPrimaries }
            return try loadTarga(targa, data: data, interpretation: interpretation, mipmapped: mipmapped,
                                 uploadOptions: uploadOptions, recognizedByImageIO: sourceType == TargaImage.imageIOType)
        }
        if let source = imageSource, CGImageSourceGetCount(source) > 0,
           let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, sourceOptions) as? [String: Any],
           let width = properties[kCGImagePropertyPixelWidth as String] as? Int,
           let height = properties[kCGImagePropertyPixelHeight as String] as? Int {
            try Self.validate(width: width, height: height)

            let image: CGImage?
            if interpretation == .color {
                // Force a decode from the full image so an embedded preview cannot replace it.
                let decodeOptions: [CFString: Any] = [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceCreateThumbnailWithTransform: true,
                    kCGImageSourceThumbnailMaxPixelSize: max(width, height),
                    kCGImageSourceShouldCacheImmediately: true,
                ]
                image = CGImageSourceCreateThumbnailAtIndex(source, 0, decodeOptions as CFDictionary)
            } else {
                image = CGImageSourceCreateImageAtIndex(source, 0, sourceOptions)
            }
            if let image {
                try Self.validate(width: image.width, height: image.height)
                return try makeTexture(
                    from: image, png: format == .png ? data : nil, interpretation: interpretation,
                    workingPrimaries: workingPrimaries,
                    mipmapped: mipmapped, uploadOptions: uploadOptions
                )
            }
        }
        if let targa {
            return try loadTarga(targa, data: data, interpretation: interpretation, mipmapped: mipmapped,
                                 uploadOptions: uploadOptions, recognizedByImageIO: true)
        }
        throw Failure.invalidImage
    }

    /// Loads KTX2 texels, including complete cubes, 2D arrays and uncompressed
    /// 3D textures. ETC1S and UASTC LDR decode to RGBA8.
    /// HDR Basis always decodes to scene-linear RGBA16Float.
    /// A nonzero layer count denotes an array, even when there is one layer.
    /// The returned texture retains stored mip levels. The image-oriented
    /// `load(data:interpretation:mipmapped:)` stays 2D.
    public func loadKTX2(
        data: Data, interpretation: Interpretation = .color, mipmapped: Bool = false
    ) throws -> any MTLTexture {
        try loadKTX2Data(data, interpretation: interpretation, mipmapped: mipmapped, allowingSpatialTextures: true)
    }

    /// Selects how ETC1S and UASTC LDR data is stored on the device.
    /// Files that declare a native pixel format use their stored format.
    public func loadKTX2(
        data: Data, interpretation: Interpretation = .color, mipmapped: Bool = false, basisStorage: BasisStorage
    ) throws -> any MTLTexture {
        return try loadKTX2Data(data, interpretation: interpretation, mipmapped: mipmapped, allowingSpatialTextures: true,
                                basisStorage: basisStorage)
    }

    static func readLocalFile(_ url: URL) throws -> Data {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values.isRegularFile == true else { throw Failure.localFileRequired }
        if let size = values.fileSize, size > maxEncodedBytes {
            throw Failure.inputTooLarge
        }

        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }

        var data = Data()
        // Read at most one extra byte so a file changed after the metadata check
        // cannot exceed the allocation bound.
        while data.count <= maxEncodedBytes {
            let remaining = maxEncodedBytes + 1 - data.count
            guard let chunk = try file.read(upToCount: min(remaining, 1_024 * 1_024)),
                  !chunk.isEmpty else { break }
            data.append(chunk)
        }
        guard data.count <= maxEncodedBytes else { throw Failure.inputTooLarge }
        return data
    }

    static func validate(width: Int, height: Int) throws {
        guard width > 0, height > 0,
              width <= maxDimension, height <= maxDimension,
              width <= maxPixels / height else {
            throw Failure.imageDimensions(width: width, height: height)
        }
    }

    private func makeTexture(
        from image: CGImage, png: Data?, interpretation: Interpretation, workingPrimaries: WorkingPrimaries,
        mipmapped: Bool, uploadOptions: UploadOptions
    ) throws -> any MTLTexture {
        if workingPrimaries != .sRGB {
            return try uploadLinearPrimaries(
                image, workingPrimaries: workingPrimaries, mipmapped: mipmapped, uploadOptions: uploadOptions
            )
        }
        let width = image.width
        let height = image.height
        let rowBytes = width * 4
        var pixels = [UInt8](repeating: 0, count: rowBytes * height)
        return try pixels.withUnsafeMutableBytes { raw in
            guard let sRGB = CGColorSpace(name: CGColorSpace.sRGB) else {
                throw Failure.colorConversion
            }

            if interpretation != .color, Self.copyStraightRGBA(image, into: raw) ||
                png.map({ PNGChannels.copyRGBA8($0, width: width, height: height, into: raw) }) == true {
                // ImageIO may round RGB during 8-bit premultiplication. For an
                // RGBA8 PNG, recover the original channels before uploading.
            } else if interpretation == .linearData, image.colorSpace?.model == .monochrome {
                try Self.copyGray(image, into: raw)
            } else {
                let input: CGImage
                if interpretation == .color || image.colorSpace?.model == .indexed
                    || image.colorSpace?.model == .monochrome {
                    input = image
                } else if let tagged = image.copy(colorSpace: sRGB) {
                    input = tagged
                } else {
                    throw Failure.colorConversion
                }
                try Self.copyColor(input, into: raw, colorSpace: sRGB)
            }

            return try uploadRGBA(raw, width: width, height: height, interpretation: interpretation, mipmapped: mipmapped,
                                  uploadOptions: uploadOptions)
        }
    }

    private func loadTarga(
        _ image: TargaImage, data: Data, interpretation: Interpretation, mipmapped: Bool,
        uploadOptions: UploadOptions, recognizedByImageIO: Bool
    ) throws -> any MTLTexture {
        var pixels = try image.pixels(data, clearTransparent: interpretation == .color, recognizedByImageIO: recognizedByImageIO)
        return try pixels.withUnsafeMutableBytes { bytes in
            try uploadRGBA(bytes, width: image.width, height: image.height, interpretation: interpretation,
                           mipmapped: mipmapped, uploadOptions: uploadOptions)
        }
    }

    private func uploadRGBA(
        _ pixels: UnsafeMutableRawBufferPointer, width: Int, height: Int, interpretation: Interpretation,
        mipmapped: Bool, uploadOptions: UploadOptions
    ) throws -> any MTLTexture {
        guard let address = pixels.baseAddress else { throw Failure.colorConversion }
        uploadOptions.apply(to: pixels, width: width, height: height, interpretation: interpretation)
        let pixelFormat: MTLPixelFormat = interpretation == .linearData ? .rgba8Unorm : .rgba8Unorm_srgb
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat, width: width, height: height, mipmapped: mipmapped
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw Failure.textureAllocation(width: width, height: height)
        }
        texture.replace(
            region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: address, bytesPerRow: width * 4
        )
        if texture.mipmapLevelCount > 1 { try generateMipmaps(for: texture) }
        return texture
    }

    private static func copyStraightRGBA(_ image: CGImage, into output: UnsafeMutableRawBufferPointer) -> Bool {
        let stride = image.width * 4
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        guard image.colorSpace?.model == .rgb, image.bitsPerComponent == 8, image.bitsPerPixel == 32,
              image.alphaInfo == .last, order == [] || order == .byteOrder32Big,
              image.bytesPerRow >= stride, let data = image.dataProvider?.data,
              image.bytesPerRow <= CFDataGetLength(data) / image.height,
              let source = CFDataGetBytePtr(data), let destination = output.baseAddress else {
            return false
        }
        for row in 0..<image.height {
            destination.advanced(by: row * stride).copyMemory(
                from: source.advanced(by: row * image.bytesPerRow), byteCount: stride
            )
        }
        return true
    }

    private static func copyGray(_ image: CGImage, into raw: UnsafeMutableRawBufferPointer) throws {
        let space = CGColorSpaceCreateDeviceGray()
        guard let tagged = image.copy(colorSpace: space),
              let context = CGContext(
                data: nil, width: image.width, height: image.height, bitsPerComponent: 8,
                bytesPerRow: 0, space: space, bitmapInfo: CGImageAlphaInfo.none.rawValue
              ) else { throw Failure.colorConversion }
        context.setBlendMode(.copy)
        context.draw(tagged, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard let pixels = context.data?.assumingMemoryBound(to: UInt8.self) else { throw Failure.colorConversion }
        for row in 0..<image.height {
            for column in 0..<image.width {
                let value = pixels[row * context.bytesPerRow + column]
                let index = (row * image.width + column) * 4
                raw[index] = value
                raw[index + 1] = value
                raw[index + 2] = value
                raw[index + 3] = 255
            }
        }
    }

    private func uploadLinearPrimaries(
        _ image: CGImage, workingPrimaries: WorkingPrimaries, mipmapped: Bool, uploadOptions: UploadOptions
    ) throws -> any MTLTexture {
        guard uploadOptions == .init() else { throw Failure.unsupportedUploadTransform("a non-sRGB working space") }
        guard let space = CGColorSpace(name: workingPrimaries.linearSpaceName) else { throw Failure.colorConversion }
        let width = image.width
        let height = image.height
        var samples = [Float](repeating: 0, count: width * height * 4)
        try samples.withUnsafeMutableBytes { raw in
            guard let context = CGContext(
                data: raw.baseAddress, width: width, height: height, bitsPerComponent: 32,
                bytesPerRow: width * 16, space: space,
                bitmapInfo: CGBitmapInfo.floatComponents.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
                    | CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { throw Failure.colorConversion }
            context.setBlendMode(.copy)
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        for index in stride(from: 0, to: samples.count, by: 4) {
            let alpha = samples[index + 3]
            if alpha == 0 {
                samples[index] = 0
                samples[index + 1] = 0
                samples[index + 2] = 0
            } else if alpha < 1 {
                samples[index] /= alpha
                samples[index + 1] /= alpha
                samples[index + 2] /= alpha
            }
        }
        let stored = samples.map { Float16($0) }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: mipmapped
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw Failure.textureAllocation(width: width, height: height)
        }
        stored.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: bytes.baseAddress!, bytesPerRow: width * 8
            )
        }
        if texture.mipmapLevelCount > 1 { try generateMipmaps(for: texture) }
        return texture
    }

    private static func copyColor(
        _ image: CGImage, into raw: UnsafeMutableRawBufferPointer, colorSpace: CGColorSpace
    ) throws {
        guard let context = CGContext(
            data: raw.baseAddress, width: image.width, height: image.height, bitsPerComponent: 8,
            bytesPerRow: image.width * 4, space: colorSpace,
            bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { throw Failure.colorConversion }
        context.setBlendMode(.copy)
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))

        // Convert the Core Graphics result back to straight alpha before sampling.
        for index in stride(from: 0, to: raw.count, by: 4) {
            let alpha = Int(raw[index + 3])
            for channel in index..<(index + 3) where alpha < 255 {
                raw[channel] = alpha == 0 ? 0 : UInt8(min(255, (Int(raw[channel]) * 255 + alpha / 2) / alpha))
            }
        }
    }

    func loadKTX2Data(
        _ data: Data, interpretation: Interpretation, mipmapped: Bool, allowingSpatialTextures: Bool,
        basisStorage: BasisStorage? = nil
    ) throws -> any MTLTexture {
        guard data.count <= Self.maxEncodedBytes else { throw Failure.inputTooLarge }
        do {
            let container = try KTX2Container(data, allowingSpatialTextures: allowingSpatialTextures)
            return try loadKTX2(container, interpretation: interpretation, mipmapped: mipmapped)
        } catch Failure.ktx2TranscodingRequired {
            return try loadBasisKTX2(data, interpretation: interpretation, mipmapped: mipmapped,
                                     allowingSpatialTextures: allowingSpatialTextures, storage: basisStorage)
        }
    }

    private func loadKTX2(
        _ container: KTX2Container, interpretation: Interpretation, mipmapped: Bool
    ) throws -> any MTLTexture {
        let format = container.format
        let pixelFormat: MTLPixelFormat = switch interpretation {
        case .color: format.storedPixelFormat
        case .encodedSRGB: format.sRGB ?? format.linear
        case .linearData: format.linear
        }
        guard format.isSupported(by: device, samplingAs: pixelFormat) else { throw Failure.unsupportedTextureFormat(format.name) }
        let generates = !format.isCompressed && container.levels.count == 1
            && (mipmapped || container.requestsGeneratedMipmaps)
        let largest = max(container.width, container.height, container.depth)
        if generates, largest > 1, format.typeSize == 4, !device.supports32BitFloatFiltering {
            throw Failure.mipmapGeneration
        }
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = container.textureType
        descriptor.pixelFormat = pixelFormat
        descriptor.width = container.width
        descriptor.height = container.height
        descriptor.depth = max(1, container.depth)
        descriptor.arrayLength = max(1, container.layerCount)
        descriptor.mipmapLevelCount = generates ? Int.bitWidth - largest.leadingZeroBitCount : container.levels.count
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw Failure.textureAllocation(width: container.width, height: container.height)
        }
        for level in container.levels.indices {
            let width = max(1, container.width >> level)
            let height = max(1, container.height >> level)
            let row = format.bytesPerRow(width: width)
            let bytes = try container.bytes(level: level)
            bytes.withUnsafeBytes { raw in
                if container.textureType == .type2D {
                    texture.replace(
                        region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: level,
                        withBytes: raw.baseAddress!, bytesPerRow: row
                    )
                } else {
                    let depth = max(1, container.depth >> level)
                    let imageBytes = format.byteCount(width: width, height: height)
                    let region = MTLRegionMake3D(0, 0, 0, width, height, depth)
                    for slice in 0..<container.sliceCount {
                        texture.replace(
                            region: region, mipmapLevel: level, slice: slice,
                            withBytes: raw.baseAddress! + slice * imageBytes, bytesPerRow: row, bytesPerImage: imageBytes
                        )
                    }
                }
            }
        }
        if generates, texture.mipmapLevelCount > 1 { try generateMipmaps(for: texture) }
        return texture
    }

    private func loadBasisKTX2(
        _ data: Data, interpretation: Interpretation, mipmapped: Bool, allowingSpatialTextures: Bool, storage: BasisStorage?
    ) throws -> any MTLTexture {
        try data.withUnsafeBytes { (source: UnsafeRawBufferPointer) in
            var info = FrostfieldBasisInfo()
            var status: Int32 = 0
            guard let decoder = frostfield_basis_open(
                source.baseAddress, source.count, allowingSpatialTextures ? 1 : 0, &info, &status
            ) else {
                throw BasisKTX2.failure(status)
            }
            defer { frostfield_basis_close(decoder) }

            let width = Int(info.width), height = Int(info.height)
            let layers = max(1, Int(info.layers))
            let faces = Int(info.faces)
            let target: BasisKTX2.Target
            if info.is_hdr != 0, storage != nil {
                throw Failure.unsupportedTextureFormat("requested LDR storage for HDR Basis Universal")
            }
            switch storage {
            case nil:
                target = info.is_hdr != 0 ? .rgba16Float : .rgba8
            case .rgba8:
                target = .rgba8
            case .deviceCompressed:
                guard let compressed = BasisKTX2.Target.deviceCompressed(
                    isUASTC: info.is_uastc != 0, hasAlpha: info.has_alpha != 0,
                    supportsAppleBlockFormats: device.supportsFamily(.apple2),
                    supportsBC: device.supportsBCTextureCompression
                ) else {
                    throw Failure.unsupportedTextureFormat("Basis Universal device-compressed storage")
                }
                target = compressed
            }
            let (transcodeFormat, format) = target.formats
            let usesSRGB = info.is_hdr == 0
                && (interpretation == .encodedSRGB || (interpretation == .color && info.is_srgb != 0))
            let generates = info.levels == 1 && mipmapped
            if generates, format.isCompressed, max(width, height) > 1 { throw Failure.mipmapGeneration }
            let descriptor = MTLTextureDescriptor()
            descriptor.textureType = faces == 6 ? .typeCube : (info.layers > 0 ? .type2DArray : .type2D)
            descriptor.pixelFormat = usesSRGB ? (format.sRGB ?? format.linear) : format.linear
            descriptor.width = width
            descriptor.height = height
            descriptor.arrayLength = layers
            descriptor.mipmapLevelCount = generates
                ? Int.bitWidth - max(width, height).leadingZeroBitCount : Int(info.levels)
            descriptor.storageMode = .shared
            descriptor.usage = .shaderRead
            guard let texture = device.makeTexture(descriptor: descriptor) else {
                throw Failure.textureAllocation(width: width, height: height)
            }
            for level in 0..<Int(info.levels) {
                let mipWidth = max(1, width >> level), mipHeight = max(1, height >> level)
                let row = format.bytesPerRow(width: mipWidth)
                let region = MTLRegionMake2D(0, 0, mipWidth, mipHeight)
                var bytes = Data(count: format.byteCount(width: mipWidth, height: mipHeight))
                // Keep each Zstandard mip hot while decoding all of its slices.
                try bytes.withUnsafeMutableBytes { (output: UnsafeMutableRawBufferPointer) in
                    for slice in 0..<(layers * faces) {
                        guard frostfield_basis_transcode(
                            decoder, transcodeFormat, UInt32(level), UInt32(slice / faces), UInt32(slice % faces),
                            output.baseAddress, output.count
                        ) == 0 else {
                            throw Failure.invalidKTX2("Basis Universal could not decode level \(level)")
                        }
                        texture.replace(
                            region: region, mipmapLevel: level, slice: slice,
                            withBytes: output.baseAddress!, bytesPerRow: row, bytesPerImage: output.count
                        )
                    }
                }
            }
            if generates, texture.mipmapLevelCount > 1 { try generateMipmaps(for: texture) }
            return texture
        }
    }

    /// Whether ImageIO lists a uniform type identifier among its decodable types.
    static func imageIOCanDecode(_ typeIdentifier: String) -> Bool {
        (CGImageSourceCopyTypeIdentifiers() as? [String])?.contains(typeIdentifier) == true
    }

    func generateMipmaps(for texture: any MTLTexture) throws {
        guard let queue = device.makeCommandQueue(), let command = queue.makeCommandBuffer(),
              let blit = command.makeBlitCommandEncoder() else { throw Failure.mipmapGeneration }
        blit.generateMipmaps(for: texture)
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else { throw Failure.mipmapGeneration }
    }

    public enum Failure: LocalizedError, Equatable {
        case localFileRequired
        case fileRead(String)
        case inputTooLarge
        case invalidImage
        case imageDimensions(width: Int, height: Int)
        case colorConversion
        case workingPrimaries
        case mipmapGeneration
        case textureAllocation(width: Int, height: Int)
        /// ImageIO on this system cannot decode the named format.
        case unsupportedImageFormat(String)
        case unsupportedUploadTransform(String)
        case invalidDDS(String)
        case unsupportedDDS(String)
        case invalidKTX1(String)
        case unsupportedKTX1(String)
        case invalidPVR(String)
        case unsupportedPVR(String)
        /// The KTX2 container is malformed.
        case invalidKTX2(String)
        /// The KTX2 container uses a valid feature this loader does not upload.
        case unsupportedKTX2(String)
        /// The KTX2 payload is Basis Universal data, which needs transcoding.
        case ktx2TranscodingRequired(String)
        /// This Metal device cannot sample the named block-compressed format.
        case unsupportedTextureFormat(String)
        case invalidDocument(String)

        public var errorDescription: String? {
            switch self {
            case .localFileRequired: "A texture URL must point to a regular local file."
            case .fileRead(let detail): "Could not read the texture file. \(detail)"
            case .inputTooLarge: "The encoded texture exceeds the 64 MiB limit."
            case .invalidImage: "Could not decode an image from the texture data."
            case .imageDimensions(let width, let height):
                "Texture dimensions \(width) × \(height) exceed the supported limits."
            case .colorConversion: "Could not prepare the image's RGBA channels."
            case .workingPrimaries:
                "Working primaries apply to a tagged color image. Stored containers and data interpretations keep their encoding."
            case .mipmapGeneration: "Metal could not generate the texture's mip levels."
            case .textureAllocation(let width, let height):
                "Metal could not allocate a \(width) × \(height) texture."
            case .unsupportedImageFormat(let name): "ImageIO on this system cannot decode \(name) images."
            case .unsupportedUploadTransform(let container):
                "Row flipping and alpha association are only available for ordinary raster images. Got \(container)."
            case .invalidDDS(let reason): "Invalid DDS texture. \(reason)"
            case .unsupportedDDS(let feature): "DDS \(feature) is not supported."
            case .invalidKTX1(let reason): "Invalid KTX 1.0 texture. \(reason)"
            case .unsupportedKTX1(let feature): "KTX 1.0 \(feature) is not supported."
            case .invalidPVR(let reason): "Invalid PVR texture. \(reason)"
            case .unsupportedPVR(let feature): "PVR \(feature) is not supported."
            case .invalidKTX2(let reason): "Invalid KTX2 texture. \(reason)"
            case .unsupportedKTX2(let feature): "KTX2 \(feature) is not supported."
            case .ktx2TranscodingRequired(let format):
                "The KTX2 texture stores \(format) data, which needs Basis Universal transcoding. Frostfield does not transcode Basis payloads; use a GPU-native KTX2 format or a PNG or JPEG image."
            case .unsupportedTextureFormat(let name): "This Metal device cannot sample \(name) textures."
            case .invalidDocument(let reason): "The texture document is not usable. \(reason)"
            }
        }
    }
}

/// Encoded image containers recognized from their leading bytes.
enum ImageSignature: CaseIterable, Hashable {
    case png, jpeg, webP, avif, ktx2

    init?(_ data: Data) {
        let bytes = [UInt8](data.prefix(64))
        if bytes.starts(with: [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
            self = .png
        } else if bytes.starts(with: [0xFF, 0xD8, 0xFF]) {
            self = .jpeg
        } else if bytes.starts(with: KTX2Container.identifier) {
            self = .ktx2
        } else if bytes.count >= 12, bytes[0..<4].elementsEqual("RIFF".utf8), bytes[8..<12].elementsEqual("WEBP".utf8) {
            self = .webP
        } else if Self.hasAVIFBrand(bytes) {
            self = .avif
        } else {
            return nil
        }
    }

    init?(mimeType: String) {
        guard let match = Self.allCases.first(where: { $0.mimeType == mimeType }) else { return nil }
        self = match
    }

    var mimeType: String {
        switch self {
        case .png: "image/png"
        case .jpeg: "image/jpeg"
        case .webP: "image/webp"
        case .avif: "image/avif"
        case .ktx2: "image/ktx2"
        }
    }

    var name: String {
        switch self {
        case .png: "PNG"
        case .jpeg: "JPEG"
        case .webP: "WebP"
        case .avif: "AVIF"
        case .ktx2: "KTX2"
        }
    }

    /// WebP and AVIF depend on the system's ImageIO decoders; the others are always available.
    var isDecodable: Bool {
        switch self {
        case .webP: TextureLoader.imageIOCanDecode("org.webmproject.webp")
        case .avif: TextureLoader.imageIOCanDecode("public.avif")
        default: true
        }
    }

    /// Checks an ISO BMFF `ftyp` box for an AVIF image or sequence brand.
    private static func hasAVIFBrand(_ bytes: [UInt8]) -> Bool {
        guard bytes.count >= 16, bytes[4..<8].elementsEqual("ftyp".utf8) else { return false }
        let size = min(bytes.count, Int(bytes[0]) << 24 | Int(bytes[1]) << 16 | Int(bytes[2]) << 8 | Int(bytes[3]))
        let brands = [8] + Array(stride(from: 16, to: size - 3, by: 4))
        return brands.contains { offset in
            offset + 4 <= bytes.count && ["avif", "avis"].contains { bytes[offset..<(offset + 4)].elementsEqual($0.utf8) }
        }
    }
}


/// Header preflight shared by native texture loading and glTF source selection.
enum BasisKTX2 {
    enum Target: Equatable {
        case rgba8, rgba16Float, etcRGB, etcRGBA, astc4x4, bc7

        static func deviceCompressed(
            isUASTC: Bool, hasAlpha: Bool, supportsAppleBlockFormats: Bool, supportsBC: Bool
        ) -> Self? {
            if supportsAppleBlockFormats {
                if isUASTC { return .astc4x4 }
                return hasAlpha ? .etcRGBA : .etcRGB
            }
            return supportsBC ? .bc7 : nil
        }

        var formats: (FrostfieldBasisFormat, KTX2Format) {
            let (transcode, vkFormat): (FrostfieldBasisFormat, UInt32) = switch self {
            case .rgba8: (FrostfieldBasisRGBA8, 37)
            case .rgba16Float: (FrostfieldBasisRGBA16Float, 97)
            case .etcRGB: (FrostfieldBasisETC1RGB, 147)
            case .etcRGBA: (FrostfieldBasisETC2RGBA, 151)
            case .astc4x4: (FrostfieldBasisASTC4x4, 157)
            case .bc7: (FrostfieldBasisBC7, 145)
            }
            return (transcode, KTX2Format(vkFormat: vkFormat)!)
        }
    }

    static func validate(_ data: Data) throws {
        let status = data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
            frostfield_basis_validate(bytes.baseAddress, bytes.count)
        }
        guard status == 0 else { throw failure(status) }
    }

    static func failure(_ status: Int32) -> TextureLoader.Failure {
        if status == 2 {
            return .unsupportedKTX2("Basis Universal layout, transfer function, alpha mode, or supercompression")
        }
        return .invalidKTX2("invalid ETC1S or UASTC Basis Universal data")
    }
}
