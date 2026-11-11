import Foundation
import Metal

extension TextureLoader {
    /// Loads a PVR v2 or v3 PVRTC1 2D texture or complete cube.
    /// The loader uploads stored levels, or only the base level when requested.
    /// The v3 color-space field determines the transfer in `.color` mode.
    public func loadPVR(
        data: Data, interpretation: Interpretation = .color, loadMipmaps: Bool = true
    ) throws -> any MTLTexture {
        try loadPVRData(data, interpretation: interpretation, loadMipmaps: loadMipmaps, allowingCube: true)
    }

    /// Loads a regular local PVR file. The caller supplies sandbox access.
    public func loadPVR(
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
        return try loadPVR(data: data, interpretation: interpretation, loadMipmaps: loadMipmaps)
    }

    func loadPVRData(
        _ data: Data, interpretation: Interpretation, loadMipmaps: Bool, allowingCube: Bool
    ) throws -> any MTLTexture {
        guard data.count <= Self.maxEncodedBytes else { throw Failure.inputTooLarge }
        let container = try PVRContainer(data, allowingCube: allowingCube)
        let sRGB = interpretation == .encodedSRGB || (interpretation == .color && container.isSRGB)
        let levels = loadMipmaps ? container.levels[...] : container.levels.prefix(1)
        return try loadPVRTC1(container.data, levels: levels, format: container.format, sRGB: sRGB,
                              decodeFailure: .invalidPVR("the PVRTC blocks could not be decoded"))
    }
}
