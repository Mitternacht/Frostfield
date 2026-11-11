import Foundation
import simd

/// A `KHR_animation_pointer` target that Frostfield binds to a typed property.
///
/// Node pointers (`/nodes/{i}/translation`, `rotation`, `scale` and `weights`)
/// import as ordinary node channels. The others are:
///
/// | Pointer | Binding |
/// | --- | --- |
/// | `/materials/{i}/pbrMetallicRoughness/baseColorFactor`, `metallicFactor`, `roughnessFactor` | `.material(_, .baseColorFactor / .metallic / .roughness)` |
/// | `/materials/{i}/emissiveFactor`, `alphaCutoff`, `normalTexture/scale`, `occlusionTexture/strength` | the matching factor |
/// | `/materials/{i}/extensions/KHR_materials_emissive_strength/emissiveStrength` | `.emissiveStrength` |
/// | `KHR_materials_clearcoat` `clearcoatFactor`, `clearcoatRoughnessFactor`; `KHR_materials_transmission` `transmissionFactor`; `KHR_materials_specular` `specularFactor`, `specularColorFactor`; `KHR_materials_sheen` `sheenColorFactor`, `sheenRoughnessFactor`; `KHR_materials_iridescence` `iridescenceFactor`; `KHR_materials_anisotropy` `anisotropyStrength`, `anisotropyRotation` | the matching layer factor |
/// | `.../{baseColor,metallicRoughness,normal,occlusion,emissive}Texture/extensions/KHR_texture_transform/{offset,rotation,scale}` | `.textureOffset`, `.textureRotation`, `.textureScale` |
/// | `/extensions/KHR_lights_punctual/lights/{i}/color`, `intensity`, `range`, `spot/innerConeAngle`, `spot/outerConeAngle` | the light property of every node using the light |
/// | `/cameras/{i}/perspective/{yfov,aspectRatio,znear,zfar}`, `/cameras/{i}/orthographic/{xmag,ymag,znear,zfar}` | `.gltfCamera(i, _)` |
///
/// Channels with any other pointer are skipped and listed in
/// `GLTFAnimationClip.unsupportedPointers`, as clients without the extension ignore
/// pointer channels. When the asset lists `KHR_animation_pointer` in
/// `extensionsRequired`, such a pointer fails to load with `GLTFMeshLoader.Failure.unsupportedFeature`.
public enum GLTFAnimationPointer: Hashable, Sendable {
    case material(Int, AnimationProperty.Material)
    /// A light definition's property, applied to every node that uses the light.
    case light(Int, AnimationProperty.Node)
    case camera(Int, AnimationProperty.GLTFCamera)

    static let extensionName = "KHR_animation_pointer"

    enum Parsed: Equatable {
        case node(Int, GLTFAnimationClip.Channel.Target)
        case pointer(GLTFAnimationPointer)
    }

    /// Floats per output value.
    var componentCount: Int {
        switch self {
        case .material(_, let property):
            switch property {
            case .baseColorFactor: 4
            case .emissiveFactor, .specularColor, .sheenColor: 3
            case .textureOffset, .textureScale: 2
            default: 1
            }
        case .light(_, let property): property == .lightColor ? 3 : 1
        case .camera: 1
        }
    }

    /// Parses an RFC 6901 JSON pointer, or returns nil when Frostfield cannot bind it.
    static func parse(_ pointer: String) -> Parsed? {
        guard pointer.hasPrefix("/") else { return nil }
        let tokens = pointer.dropFirst().split(separator: "/", omittingEmptySubsequences: false).map {
            $0.replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")
        }
        func index(_ token: String) -> Int? {
            guard let value = Int(token), value >= 0, String(value) == token else { return nil }
            return value
        }
        switch tokens.first {
        case "nodes":
            guard tokens.count == 3, let node = index(tokens[1]),
                  let target = GLTFAnimationClip.Channel.Target(rawValue: tokens[2]) else { return nil }
            return .node(node, target)
        case "materials":
            guard tokens.count >= 3, let material = index(tokens[1]),
                  let property = materialProperty(Array(tokens.dropFirst(2))) else { return nil }
            return .pointer(.material(material, property))
        case "extensions":
            guard tokens.count >= 5, tokens[1] == "KHR_lights_punctual", tokens[2] == "lights",
                  let light = index(tokens[3]) else { return nil }
            let property: AnimationProperty.Node
            switch Array(tokens.dropFirst(4)) {
            case ["color"]: property = .lightColor
            case ["intensity"]: property = .lightIntensity
            case ["range"]: property = .lightRange
            case ["spot", "innerConeAngle"]: property = .spotInnerConeAngle
            case ["spot", "outerConeAngle"]: property = .spotOuterConeAngle
            default: return nil
            }
            return .pointer(.light(light, property))
        case "cameras":
            guard tokens.count == 4, let camera = index(tokens[1]) else { return nil }
            let property: AnimationProperty.GLTFCamera
            switch (tokens[2], tokens[3]) {
            case ("perspective", "yfov"): property = .verticalFieldOfView
            case ("perspective", "aspectRatio"): property = .aspectRatio
            case ("perspective", "znear"), ("orthographic", "znear"): property = .near
            case ("perspective", "zfar"), ("orthographic", "zfar"): property = .far
            case ("orthographic", "xmag"): property = .halfWidth
            case ("orthographic", "ymag"): property = .halfHeight
            default: return nil
            }
            return .pointer(.camera(camera, property))
        default:
            return nil
        }
    }

    private static func materialProperty(_ path: [String]) -> AnimationProperty.Material? {
        let textures: [String: AnimationProperty.MaterialMap] = [
            "baseColorTexture": .baseColor, "metallicRoughnessTexture": .metallicRoughness,
            "normalTexture": .normal, "occlusionTexture": .occlusion, "emissiveTexture": .emissive,
        ]
        // Texture transforms: [pbrMetallicRoughness/]<texture>/extensions/KHR_texture_transform/<field>.
        var texturePath = path
        if texturePath.first == "pbrMetallicRoughness" { texturePath.removeFirst() }
        if texturePath.count == 4, let map = textures[texturePath[0]], texturePath[1] == "extensions",
           texturePath[2] == "KHR_texture_transform",
           (map == .baseColor || map == .metallicRoughness) == (path.first == "pbrMetallicRoughness") {
            switch texturePath[3] {
            case "offset": return .textureOffset(map)
            case "rotation": return .textureRotation(map)
            case "scale": return .textureScale(map)
            default: return nil
            }
        }
        switch path {
        case ["pbrMetallicRoughness", "baseColorFactor"]: return .baseColorFactor
        case ["pbrMetallicRoughness", "metallicFactor"]: return .metallic
        case ["pbrMetallicRoughness", "roughnessFactor"]: return .roughness
        case ["emissiveFactor"]: return .emissiveFactor
        case ["alphaCutoff"]: return .alphaCutoff
        case ["normalTexture", "scale"]: return .normalScale
        case ["occlusionTexture", "strength"]: return .occlusionStrength
        case ["extensions", "KHR_materials_emissive_strength", "emissiveStrength"]: return .emissiveStrength
        case ["extensions", "KHR_materials_clearcoat", "clearcoatFactor"]: return .clearcoatFactor
        case ["extensions", "KHR_materials_clearcoat", "clearcoatRoughnessFactor"]: return .clearcoatRoughness
        case ["extensions", "KHR_materials_transmission", "transmissionFactor"]: return .transmissionFactor
        case ["extensions", "KHR_materials_specular", "specularFactor"]: return .specularFactor
        case ["extensions", "KHR_materials_specular", "specularColorFactor"]: return .specularColor
        case ["extensions", "KHR_materials_sheen", "sheenColorFactor"]: return .sheenColor
        case ["extensions", "KHR_materials_sheen", "sheenRoughnessFactor"]: return .sheenRoughness
        case ["extensions", "KHR_materials_iridescence", "iridescenceFactor"]: return .iridescenceFactor
        case ["extensions", "KHR_materials_anisotropy", "anisotropyStrength"]: return .anisotropyStrength
        case ["extensions", "KHR_materials_anisotropy", "anisotropyRotation"]: return .anisotropyRotation
        default: return nil
        }
    }
}

extension GLTFAnimationClip {
    /// A `KHR_animation_pointer` channel whose target is not a node transform or weights.
    public struct PointerChannel {
        /// The JSON pointer as written in the asset.
        public let pointer: String
        public let target: GLTFAnimationPointer
        public let interpolation: Channel.Interpolation
        public let times: [Float]
        /// Output floats, `target.componentCount` per value; cubic splines store
        /// in-tangent, value and out-tangent per key.
        let values: [Float]

        /// The channel as a typed track for a property.
        func track(for property: AnimationProperty) throws -> AnimationTrack {
            let count = target.componentCount
            let kind = property.valueKind ?? .number
            let animationValues = stride(from: 0, to: values.count, by: count).map { start in
                AnimationValue.continuous(kind, values[start..<(start + count)].map(Double.init))
            }
            let mode: AnimationTrack.Interpolation = switch interpolation {
            case .linear: .linear
            case .step: .step
            case .cubicSpline: .cubicSpline
            }
            return try AnimationTrack(property: property, times: times, values: animationValues, interpolation: mode)
        }
    }
}

extension AnimationClip {
    /// Binds an imported glTF clip to a prepared scene, including its
    /// `KHR_animation_pointer` channels: node channels target `nodeIDs`, material
    /// pointers the scene's `materialFactors`, light pointers every node using the
    /// light, and camera pointers `.gltfCamera`. Play it with an `AnimationMixer`,
    /// then use `apply(to:)` for the graph, `applyLiveProperties()` for materials and
    /// `PreparedGLTFScene.cameraView(nodeIndex:in:viewportAspectRatio:animatedBy:)`
    /// for cameras. Pointers to an imported material that no primitive of the
    /// prepared scene draws, such as one only another material variant selects, are
    /// skipped. Throws `Failure.unmappedTrack` for a pointer whose target the
    /// prepared scene does not have.
    public init(_ clip: GLTFAnimationClip, in scene: PreparedGLTFScene) throws {
        let nodes = try AnimationClip(clip, nodeIDs: scene.nodeIDs)
        var tracks: [AnimationTrack] = []
        for channel in clip.pointerChannels {
            switch channel.target {
            case .material(let index, let property):
                guard let live = scene.materialFactors[index] else {
                    if scene.importedMaterialIndices.contains(index) { continue }
                    throw Failure.unmappedTrack(channel.pointer)
                }
                tracks.append(try channel.track(for: .material(live, property)))
            case .light(let index, let property):
                guard let ids = scene.lightNodeIDs[index], !ids.isEmpty else { throw Failure.unmappedTrack(channel.pointer) }
                for id in ids { tracks.append(try channel.track(for: .node(id, property))) }
            case .camera(let index, let property):
                tracks.append(try channel.track(for: .gltfCamera(index, property)))
            }
        }
        var seen = Set<AnimationProperty>()
        for track in tracks where !seen.insert(track.property).inserted {
            throw Failure.duplicateTrack(track.property.description)
        }
        self.init(
            name: clip.name, duration: clip.duration, blendMode: .normal, targets: nodes.targets, reference: [:],
            tracks: tracks, trackReference: [:]
        )
    }
}
