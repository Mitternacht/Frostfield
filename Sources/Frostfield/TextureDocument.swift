import Metal

/// Parsed bytes for a data texture or a block-compressed texture.
///
/// This is the loader base behind Three.js `DataTextureLoader` and
/// `CompressedTextureLoader`. A caller parses a file format into a document.
/// ``TextureLoader`` checks the size limits and uploads it. The built-in KTX,
/// DDS, PVR and HDR methods stay direct.
public struct TextureDocument: Sendable, Equatable {
    public enum Contents: Sendable, Equatable {
        /// Straight RGBA8, tightly packed, `width × height × 4` bytes.
        case rgba8(Data)
        /// Block bytes for each mip, starting at level zero. The level count is the array count.
        case blocks(format: MTLPixelFormat, levels: [Data])
    }

    public let width: Int
    public let height: Int
    public let contents: Contents

    public init(width: Int, height: Int, contents: Contents) {
        self.width = width
        self.height = height
        self.contents = contents
    }
}

/// Turns file bytes into a ``TextureDocument``.
public protocol TextureDocumentParser: Sendable {
    func parse(_ data: Data) throws -> TextureDocument
}

extension TextureLoader {
    /// Parses with `parser`, then uploads. The encoded input stays within 64 MiB.
    public func load(data: Data, parser: any TextureDocumentParser) throws -> any MTLTexture {
        guard data.count <= Self.maxEncodedBytes else { throw Failure.inputTooLarge }
        return try upload(try parser.parse(data))
    }

    /// Uploads a document the caller already parsed.
    public func upload(_ document: TextureDocument) throws -> any MTLTexture {
        try Self.validate(width: document.width, height: document.height)
        switch document.contents {
        case .rgba8(let bytes):
            let count = document.width * document.height * 4
            guard bytes.count == count else {
                throw Failure.invalidDocument("RGBA8 needs \(count) bytes, not \(bytes.count).")
            }
            return try makeDocumentTexture(
                format: .rgba8Unorm, width: document.width, height: document.height, levels: [bytes]
            )
        case .blocks(let format, let levels):
            guard let layout = Self.blockLayout(format) else {
                throw Failure.invalidDocument("Pixel format \(format.rawValue) is not a supported block layout.")
            }
            let limit = 1 + Int(log2(Double(max(document.width, document.height))))
            guard (1...limit).contains(levels.count) else {
                throw Failure.invalidDocument("A \(document.width) × \(document.height) texture has 1 to \(limit) mip levels.")
            }
            for (level, bytes) in levels.enumerated() {
                let width = max(1, document.width >> level)
                let height = max(1, document.height >> level)
                let needed = Self.blockCount(width, layout.blockWidth) * Self.blockCount(height, layout.blockHeight) * layout.bytes
                guard bytes.count == needed else {
                    throw Failure.invalidDocument("Level \(level) needs \(needed) bytes, not \(bytes.count).")
                }
            }
            return try makeDocumentTexture(
                format: format, width: document.width, height: document.height, levels: levels
            )
        }
    }

    private func makeDocumentTexture(
        format: MTLPixelFormat, width: Int, height: Int, levels: [Data]
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: levels.count > 1
        )
        descriptor.mipmapLevelCount = levels.count
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw Failure.unsupportedTextureFormat("\(format.rawValue)")
        }
        let layout = Self.blockLayout(format)
        for (level, bytes) in levels.enumerated() {
            let levelWidth = max(1, width >> level)
            let levelHeight = max(1, height >> level)
            let row = layout.map { Self.blockCount(levelWidth, $0.blockWidth) * $0.bytes } ?? levelWidth * 4
            bytes.withUnsafeBytes { raw in
                texture.replace(
                    region: MTLRegionMake2D(0, 0, levelWidth, levelHeight), mipmapLevel: level,
                    withBytes: raw.baseAddress!, bytesPerRow: row
                )
            }
        }
        texture.label = "Frostfield texture document"
        return texture
    }

    private struct BlockLayout {
        var blockWidth: Int
        var blockHeight: Int
        var bytes: Int
    }

    private static func blockLayout(_ format: MTLPixelFormat) -> BlockLayout? {
        switch format {
        case .astc_4x4_ldr, .astc_4x4_srgb:
            return BlockLayout(blockWidth: 4, blockHeight: 4, bytes: 16)
        case .bc1_rgba, .bc1_rgba_srgb:
            return BlockLayout(blockWidth: 4, blockHeight: 4, bytes: 8)
        default:
            return nil
        }
    }

    private static func blockCount(_ texels: Int, _ block: Int) -> Int {
        max(1, (texels + block - 1) / block)
    }
}
