import Foundation

/// Loads a standalone Wavefront MTL library using the same material conversion
/// as OBJ scene import. Each result prepares as a Frostfield PBR material.
public enum MTLLoader {
    public struct Library {
        /// The material names declared by the file.
        public let materials: [String: GLTFScene.Material]
        public let warnings: [ModelIOSceneLoader.Warning]
    }

    /// Loads a regular local file and resolves textures inside the file's folder.
    /// Supports UTF-8 and ISO Latin-1. Material colors and map placement follow
    /// the OBJ importer, including dissolve-map packing, bump maps and PBR extensions.
    /// Phong shininess maps to roughness. Classic specular color, specular
    /// and displacement maps are outside the converted material model.
    public static func load(contentsOf url: URL) throws -> Library {
        guard url.isFileURL else { throw Failure.localFileRequired }
        let bytes: Data
        do {
            bytes = try TextureLoader.readLocalFile(url)
        } catch {
            throw Failure.fileRead(error.localizedDescription)
        }
        guard let text = String(data: bytes, encoding: .utf8) ?? String(data: bytes, encoding: .isoLatin1) else {
            throw Failure.textEncoding
        }
        var importer = ModelIOMaterialImporter(assetURL: url, isUSD: false)
        importer.importWavefront(WavefrontMaterialLibrary(text))
        let materials = Dictionary(uniqueKeysWithValues: importer.materials.compactMap { material in
            material.name.map { ($0, material) }
        })
        guard !materials.isEmpty else { throw Failure.emptyLibrary }
        return Library(materials: materials, warnings: importer.warnings)
    }

    public enum Failure: LocalizedError, Equatable {
        case localFileRequired
        case fileRead(String)
        case textEncoding
        case emptyLibrary

        public var errorDescription: String? {
            switch self {
            case .localFileRequired: "Provide a local MTL file URL."
            case .fileRead(let reason): "Could not read the MTL file. \(reason)"
            case .textEncoding: "Use a UTF-8 or ISO Latin-1 MTL file."
            case .emptyLibrary: "The MTL file contains no named materials."
            }
        }
    }
}
