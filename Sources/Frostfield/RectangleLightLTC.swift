import Foundation
import Metal
import Synchronization

/// Immutable GGX transform and Fresnel tables from the pinned Three.js r187
/// rectangle light, originally fitted by Heitz, Dupuy, Hill and Neubelt.
final class RectangleLightLTC: @unchecked Sendable {
    private static let cached = Mutex<RectangleLightLTC?>(nil)
    let texture: any MTLTexture

    private init(device: any MTLDevice) throws {
        let size = 64
        let rowBytes = size * MemoryLayout<SIMD4<Float16>>.stride
        let imageBytes = rowBytes * size
        guard let url = Bundle.module.url(
            forResource: "Three-r187-64x64-rgba16f", withExtension: "bin", subdirectory: "RectangleLTC"
        ), let bytes = try? Data(contentsOf: url), bytes.count == imageBytes * 2 else {
            throw MeshRenderer.Failure.resource("Rectangle light LTC tables")
        }
        let texture = try LightTextureSet.makeArray(
            device: device, format: .rgba16Float, width: size, height: size, layers: 2
        )
        texture.label = "Frostfield rectangle light LTC tables"
        bytes.withUnsafeBytes { raw in
            for slice in 0..<2 {
                texture.replace(
                    region: MTLRegionMake2D(0, 0, size, size), mipmapLevel: 0, slice: slice,
                    withBytes: raw.baseAddress!.advanced(by: slice * imageBytes), bytesPerRow: rowBytes, bytesPerImage: imageBytes
                )
            }
        }
        self.texture = texture
    }

    static func load(device: any MTLDevice) throws -> RectangleLightLTC {
        try cached.withLock { entry in
            if let entry, entry.texture.device === device { return entry }
            let tables = try RectangleLightLTC(device: device)
            entry = tables
            return tables
        }
    }

    func retainIfNeeded(on commandBuffer: any MTLCommandBuffer) {
        guard !commandBuffer.retainedReferences else { return }
        commandBuffer.addCompletedHandler { [self] _ in withExtendedLifetime(self) {} }
    }
}
