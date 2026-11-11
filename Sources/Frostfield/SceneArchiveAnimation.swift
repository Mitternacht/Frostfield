import Foundation
import simd

/// Stable targets of typed tracks. Live bindings use the resource tables, not
/// ObjectIdentifier values from the process that created the archive.
enum ArchivePropertyBinding: Codable {
    case node(id: UUID, field: AnimationProperty.Node)
    case camera(field: AnimationProperty.Camera)
    case environment(field: AnimationProperty.Environment)
    case uniform(index: Int, name: String)
    case material(index: Int, field: AnimationProperty.Material)
    case rigLight(id: String, field: AnimationProperty.RigLight)
    case gltfCamera(index: Int, field: AnimationProperty.GLTFCamera)
    case custom(name: String)
}

struct ArchivePropertyTrack: Codable {
    var binding: ArchivePropertyBinding
    var interpolation: AnimationTrack.Interpolation
    var times: Int
    var values: ArchivePropertyValues
    var reference: ArchivePropertyValues?

    init(
        track: AnimationTrack, binding: ArchivePropertyBinding, reference: AnimationValue?,
        writer: inout ArchiveBinaryWriter
    ) throws {
        try Self.validateCubicQuaternionKeys(track.values, interpolation: track.interpolation, property: track.property)
        self.binding = binding
        interpolation = track.interpolation
        times = writer.append(track.times)
        values = .capture(track.values, kind: track.valueKind, writer: &writer)
        self.reference = reference.map { .capture([$0], kind: $0.kind, writer: &writer) }
    }

    func restore(
        property: AnimationProperty, additive: Bool, reader: ArchiveBinaryReader
    ) throws -> (AnimationTrack, AnimationValue?) {
        let keyTimes = try reader.array(times, of: Float.self)
        let stride = interpolation == .cubicSpline ? 3 : 1
        let count = try reader.product(keyTimes.count, stride, view: times)
        let keys = try values.restore(count: count, reader: reader)
        try Self.validateCubicQuaternionKeys(keys, interpolation: interpolation, property: property)
        let track: AnimationTrack
        do {
            track = try AnimationTrack(
                property: property, times: keyTimes, values: keys, interpolation: interpolation,
                unitKeys: false, normalizeQuaternionKeys: false
            )
        } catch {
            throw SceneArchive.Failure.invalidAnimation(error.localizedDescription)
        }
        guard (reference != nil) == (additive && track.valueKind.isContinuous) else {
            throw SceneArchive.Failure.invalidAnimation("unexpected or missing additive reference for \(property)")
        }
        guard let reference else { return (track, nil) }
        let value = try reference.restore(count: 1, reader: reader)[0]
        guard value.kind == track.valueKind, value.isFinite,
              value.quaternionValue.map({ abs(simd_length($0.vector) - 1) <= 1e-3 }) != false else {
            throw SceneArchive.Failure.invalidAnimation("invalid additive reference for \(property)")
        }
        return (track, value)
    }

    private static func validateCubicQuaternionKeys(
        _ keys: [AnimationValue], interpolation: AnimationTrack.Interpolation, property: AnimationProperty
    ) throws {
        // Exact subclips can have non-unit cubic keys. A zero quaternion has
        // no sampled orientation, regardless of interpolation.
        guard interpolation == .cubicSpline, keys.first?.kind == .quaternion else { return }
        for index in stride(from: 1, to: keys.count, by: 3) where keys[index].quaternionValue?.vector == .zero {
            throw SceneArchive.Failure.invalidAnimation("zero cubic quaternion key for \(property)")
        }
    }
}

/// Numeric and boolean key data live in the sidecar. String keys are JSON.
enum ArchivePropertyValues: Codable {
    case continuous(kind: AnimationValue.Kind, view: Int)
    case booleans(view: Int)
    case strings([String])

    static func capture(_ values: [AnimationValue], kind: AnimationValue.Kind, writer: inout ArchiveBinaryWriter) -> Self {
        switch kind {
        case .boolean:
            return .booleans(view: writer.append(values.map { UInt8($0.booleanValue == true ? 1 : 0) }))
        case .string:
            return .strings(values.compactMap { if case .string(let text) = $0 { text } else { nil } })
        default:
            return .continuous(kind: kind, view: writer.append(values.flatMap { $0.components?.map(Float.init) ?? [] }))
        }
    }

    func restore(count: Int, reader: ArchiveBinaryReader) throws -> [AnimationValue] {
        switch self {
        case .booleans(let view):
            let bytes = try reader.array(view, of: UInt8.self, count: count)
            guard bytes.allSatisfy({ $0 <= 1 }) else { throw SceneArchive.Failure.invalidAnimation("invalid boolean key") }
            return bytes.map { .boolean($0 == 1) }
        case .strings(let strings):
            guard strings.count == count else { throw SceneArchive.Failure.invalidAnimation("string key count") }
            return strings.map(AnimationValue.string)
        case .continuous(let kind, let view):
            guard kind.isContinuous, kind.componentCount >= 0 else {
                throw SceneArchive.Failure.invalidAnimation("invalid numeric value kind")
            }
            let width = kind.componentCount
            let components = try reader.array(view, of: Float.self, count: try reader.product(count, width, view: view))
            return (0..<count).map { index in
                let start = index * width
                return .continuous(kind, components[start..<(start + width)].map(Double.init))
            }
        }
    }
}

struct ArchiveLiveFactor: Codable {
    var field: AnimationProperty.Material
    var value: ArchivePropertyValues

    /// The factor list is ordered so live state has deterministic sidecar offsets.
    static let fields: [AnimationProperty.Material] = [
        .baseColorFactor, .metallic, .roughness, .emissiveFactor, .emissiveStrength, .normalScale, .occlusionStrength,
        .alphaCutoff, .clearcoatFactor, .clearcoatRoughness, .transmissionFactor, .specularFactor, .specularColor,
        .sheenColor, .sheenRoughness, .iridescenceFactor, .anisotropyStrength, .anisotropyRotation,
    ] + AnimationProperty.MaterialMap.allCases.flatMap { [.textureOffset($0), .textureRotation($0), .textureScale($0)] }
}

/// Validates target existence and shape at both capture and load. Key values
/// still use the normal AnimationTrack and AnimationMixer validation.
struct ArchiveAnimationTargets {
    let nodes: [UUID: ArchiveNode]
    let meshes: [ArchiveMesh]
    let rigLights: [String: LightRig.Light]

    init(nodes: [ArchiveNode], meshes: [ArchiveMesh], lightRig: LightRig?) {
        self.nodes = Dictionary(uniqueKeysWithValues: nodes.map { ($0.id, $0) })
        self.meshes = meshes
        rigLights = Dictionary(uniqueKeysWithValues: (lightRig?.lights ?? []).map { ($0.id, $0) })
    }

    func validate(_ property: AnimationProperty, kind: AnimationValue.Kind) throws {
        let failure = SceneArchive.Failure.invalidReference(property.description)
        switch property {
        case .node(let id, let field):
            guard let node = nodes[id] else { throw failure }
            let count = node.mesh.flatMap { meshes.indices.contains($0) ? meshes[$0].morphTargets?.targets.count : nil }
                ?? node.morphWeights?.count ?? 0
            switch field {
            case .morphWeights:
                guard count > 0, kind == .numbers(count: count) else { throw failure }
            case .morphWeight(let index):
                guard (0..<count).contains(index) else { throw failure }
            case .lightColor, .lightIntensity:
                guard node.light != nil else { throw failure }
            case .lightRange:
                switch node.light?.kind {
                case .point(.some), .spot(.some, _, _): break
                default: throw failure
                }
            case .spotInnerConeAngle, .spotOuterConeAngle:
                guard case .spot? = node.light?.kind else { throw failure }
            default: break
            }
        case .material(let live, let field):
            guard live.rest.value(of: field) != nil else { throw failure }
        case .uniform:
            guard property.valueKind != nil else { throw failure }
        case .rigLight(let id, let field):
            guard let light = rigLights[id] else { throw failure }
            switch field {
            case .intensity: guard case .intensity = light.emission else { throw failure }
            case .power: guard case .power = light.emission else { throw failure }
            case .innerAngle, .outerAngle: guard case .spot = light.kind else { throw failure }
            case .width, .height: guard case .rectangle = light.kind else { throw failure }
            case .range:
                switch light.kind {
                case .point(.some), .spot(_, _, .some): break
                default: throw failure
                }
            case .position, .target, .color: break
            }
        case .gltfCamera(let index, _):
            guard index >= 0 else { throw failure }
        case .camera, .environment, .custom: break
        }
    }
}
