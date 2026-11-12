import Foundation

extension GLTFMeshLoader {
    /// Extensions the loader can honor, including when an asset lists them in
    /// `extensionsRequired`. A required extension outside this set, or one whose
    /// loading option is off, fails with `Failure.unsupportedFeature(name)`.
    ///
    /// | Extension | Condition |
    /// | --- | --- |
    /// | `KHR_mesh_quantization` | Always. 8-bit and 16-bit attributes dequantize to `Float` with the glTF equations. |
    /// | `EXT_meshopt_compression`, `KHR_meshopt_compression` | Always. Compressed views decode on load; fallback buffers are never read. |
    /// | `KHR_draco_mesh_compression` | Triangle-list primitives. Compressed attributes and indices are decoded before the primitive's other accessors. |
    /// | `EXT_mesh_gpu_instancing` | `includeGPUInstancing` in `loadScene`. Fills `GLTFScene.Node.instances`. |
    /// | `KHR_lights_punctual` | `includeLights`. |
    /// | `KHR_animation_pointer` | `includeAnimations`. Node, material, light and camera pointers listed in `GLTFAnimationPointer`; pointers to data that was not loaded are skipped. Other pointers are skipped and listed in `GLTFAnimationClip.unsupportedPointers`, or fail with `unsupportedFeature` when the extension is required. |
    /// | `KHR_materials_clearcoat`, `KHR_materials_transmission`, `KHR_materials_volume`, `KHR_materials_ior`, `KHR_materials_emissive_strength`, `KHR_materials_specular`, `KHR_materials_sheen`, `KHR_materials_anisotropy`, `KHR_materials_iridescence`, `KHR_materials_dispersion`, `KHR_materials_unlit`, `KHR_materials_diffuse_transmission`, `KHR_materials_diffuse_roughness` (draft), `KHR_materials_volume_scatter` (draft), `EXT_materials_bump`, `KHR_materials_variants`, `KHR_texture_transform` | `includeMaterials`. |
    /// | `EXT_texture_webp`, `EXT_texture_avif` | `includeMaterials`. Used when ImageIO can decode the format, otherwise the texture's PNG or JPEG `source` is used, or loading fails. |
    /// | `KHR_texture_basisu` | `includeMaterials`. ETC1S and UASTC LDR textures decode to RGBA8 with stored mips by default, including Zstandard UASTC. Material preparation can prefer device-compressed storage. A PNG or JPEG source can provide a fallback. |
    public static let supportedExtensions: Set<String> = geometryExtensions
        .union([GLTFHierarchy.gpuInstancingExtension, GLTFLightDocument.extensionName, GLTFAnimationPointer.extensionName])
        .union(GLTFMaterialDocument.supportedExtensions)
        .union(GLTFMaterialDocument.textureExtensions)

    static let quantizationExtension = "KHR_mesh_quantization"
    static let dracoExtension = "KHR_draco_mesh_compression"
    private static let geometryExtensions: Set<String> = Set(
        [quantizationExtension, dracoExtension] + MeshoptDecoder.Specification.allCases.map(\.rawValue)
    )

    /// Whether a required extension is honored with these loading options.
    static func honors(
        _ name: String, includeLights: Bool, includeMaterials: Bool, includeGPUInstancing: Bool,
        includeAnimations: Bool = false
    ) -> Bool {
        if geometryExtensions.contains(name) { return true }
        switch name {
        case GLTFLightDocument.extensionName: return includeLights
        case GLTFAnimationPointer.extensionName: return includeAnimations
        case GLTFHierarchy.gpuInstancingExtension: return includeGPUInstancing
        default:
            return includeMaterials
                && (GLTFMaterialDocument.supportedExtensions.contains(name) || GLTFMaterialDocument.textureExtensions.contains(name))
        }
    }
}
