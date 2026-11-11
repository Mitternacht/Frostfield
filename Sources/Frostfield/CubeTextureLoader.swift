import Foundation
import Metal

/// Loads six encoded images into a shader-readable Metal cube.
///
/// Supply faces in Metal slice order: +X, −X, +Y, −Y, +Z, −Z, looking
/// outward from the center. Rows start at the top left of each oriented face.
/// All faces must have the same square dimensions and be at most 4096 pixels
/// per side. The cube is ready for use when the call returns.
public struct CubeTextureLoader {
    private let queue: any MTLCommandQueue
    private let loader: TextureLoader

    public init(queue: any MTLCommandQueue) {
        self.queue = queue
        loader = TextureLoader(device: queue.device)
    }

    /// Loads six local files. The caller supplies any required sandbox access.
    public func load(
        contentsOf urls: [URL], interpretation: TextureLoader.Interpretation = .color,
        mipmapped: Bool = true
    ) throws -> any MTLTexture {
        try assemble(urls, mipmapped: mipmapped) { try loader.load(contentsOf: $0, interpretation: interpretation) }
    }

    /// Loads the base level of each face, optionally generating a cube mip chain.
    /// Faces must all decode to the same RGBA8 or floating-point format.
    public func load(
        data faces: [Data], interpretation: TextureLoader.Interpretation = .color,
        mipmapped: Bool = true
    ) throws -> any MTLTexture {
        try assemble(faces, mipmapped: mipmapped) { try loader.load(data: $0, interpretation: interpretation) }
    }

    /// Loads six local files into a linear sRGB floating-point cube.
    public func loadHighDynamicRange(
        contentsOf urls: [URL], precision: TextureLoader.HighDynamicRangePrecision = .half,
        mipmapped: Bool = false
    ) throws -> any MTLTexture {
        try assemble(urls, mipmapped: mipmapped) {
            try loader.loadHighDynamicRange(contentsOf: $0, precision: precision).texture
        }
    }

    /// Converts Radiance, EXR, gain-map, or SDR faces to a common linear float format.
    public func loadHighDynamicRange(
        data faces: [Data], precision: TextureLoader.HighDynamicRangePrecision = .half,
        mipmapped: Bool = false
    ) throws -> any MTLTexture {
        try assemble(faces, mipmapped: mipmapped) {
            try loader.loadHighDynamicRange(data: $0, precision: precision).texture
        }
    }

    private func assemble<Source>(
        _ sources: [Source], mipmapped: Bool, decode: (Source) throws -> any MTLTexture
    ) throws -> any MTLTexture {
        guard sources.count == 6 else { throw Failure.faceCount }
        var faces: [any MTLTexture] = []
        for (index, source) in sources.enumerated() {
            let face: any MTLTexture
            do {
                face = try decode(source)
            } catch {
                throw Failure.decodeFace(index, error.localizedDescription)
            }
            guard (1...4096).contains(face.width), face.width == face.height,
                  faces.first.map({ $0.width == face.width }) ?? true else {
                throw Failure.faceSize(index, face.width, face.height)
            }
            switch face.pixelFormat {
            case .rgba8Unorm, .rgba8Unorm_srgb, .rgba16Float, .rgba32Float: break
            default: throw Failure.faceFormat(index)
            }
            guard faces.first.map({ $0.pixelFormat == face.pixelFormat }) ?? true else {
                throw Failure.faceFormat(index)
            }
            faces.append(face)
        }

        let first = faces[0]
        if mipmapped, first.width > 1, first.pixelFormat == .rgba32Float,
           !queue.device.supports32BitFloatFiltering {
            throw TextureLoader.HighDynamicRangeFailure.floatFilteringUnsupported
        }
        let descriptor = MTLTextureDescriptor.textureCubeDescriptor(
            pixelFormat: first.pixelFormat, size: first.width, mipmapped: mipmapped
        )
        descriptor.storageMode = .private
        descriptor.hazardTrackingMode = .tracked
        descriptor.usage = .shaderRead
        guard let cube = queue.device.makeTexture(descriptor: descriptor) else { throw Failure.allocation }
        cube.label = "Frostfield loaded cube"

        guard let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder() else {
            throw Failure.upload(nil)
        }
        blit.label = "Frostfield upload cube faces"
        let origin = MTLOrigin(x: 0, y: 0, z: 0)
        let size = MTLSize(width: first.width, height: first.height, depth: 1)
        for (index, face) in faces.enumerated() {
            blit.copy(
                from: face, sourceSlice: 0, sourceLevel: 0, sourceOrigin: origin, sourceSize: size,
                to: cube, destinationSlice: index, destinationLevel: 0, destinationOrigin: origin
            )
        }
        if cube.mipmapLevelCount > 1 { blit.generateMipmaps(for: cube) }
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else { throw Failure.upload(command.error?.localizedDescription) }
        return cube
    }

    public enum Failure: LocalizedError, Equatable {
        case faceCount
        case decodeFace(Int, String)
        case faceSize(Int, Int, Int)
        case faceFormat(Int)
        case allocation
        case upload(String?)

        public var errorDescription: String? {
            switch self {
            case .faceCount: "Supply exactly six cube face images."
            case .decodeFace(let index, let detail): "Could not decode cube face \(index): \(detail)"
            case .faceSize(let index, let width, let height):
                "Cube face \(index) is \(width) by \(height). Use matching square faces up to 4096 pixels per side."
            case .faceFormat(let index): "Cube face \(index) must match the other faces' RGBA8 or floating-point format."
            case .allocation: "Metal could not allocate the cube texture."
            case .upload(let detail): detail ?? "Metal could not upload the cube faces."
            }
        }
    }
}
