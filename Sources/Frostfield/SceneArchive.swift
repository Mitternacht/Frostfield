import Foundation
import Metal
import simd

/// A versioned, self-contained snapshot of a supported scene graph. The JSON
/// manifest and binary sidecar play the role of Three.js `Object3D.toJSON`
/// and `ObjectLoader`.
///
/// The manifest holds the node tree as a flat pre-order list with parent indices, so
/// hierarchies of any depth encode without recursion (IDs, names, transforms, pivots, orientation, visibility,
/// layers, render order, metadata, lights, morph weights, culling bounds, levels of detail),
/// the camera, environment settings and diffuse light probes, meshes, materials with every PBR extension,
/// texture and sampler records, surface shader graphs with their uniform values,
/// instance sets, animation clips and an optional light rig. The sidecar holds
/// geometry, skins, morph targets, palettes, instance records, keyframes and every
/// mip level and slice of every texture, as little-endian bytes in their stored Metal
/// formats. Photometric profiles and light cookie pixels also use the sidecar.
/// Loading the archive recreates renderers that draw the same pixels.
///
/// Shared objects stay shared: a renderer, texture, environment, uniform set or
/// instance set or live PBR material used by several nodes is stored once and
/// recreated once. Typed tracks address the recreated targets, and live PBR
/// factors retain both their authored rest values and current values.
///
/// Capture rejects unsupported inputs, including custom materials, unrecorded
/// samplers without a matching description, and multisample, memoryless
/// or framebuffer-only textures.
public struct SceneArchive: Sendable, Equatable {
    /// The newest manifest version this build reads and writes. Captures use the
    /// earliest version that can store the scene.
    public static let formatVersion = 30
    /// Maximum root-to-leaf node count. Keeps nested scene values safe to release
    /// on a Swift task stack even though archive traversal itself is iterative.
    public static let maximumHierarchyDepth = 1_024
    public static let formatName = "frostfield.scene"
    /// The manifest's file name inside an archive directory.
    public static let manifestFileName = "scene.json"
    /// The sidecar's file name inside an archive directory.
    public static let binaryFileName = "scene.bin"

    /// UTF-8 JSON with sorted keys, so equal scenes produce equal bytes.
    public let manifest: Data
    /// The views the manifest references, each aligned to 16 bytes.
    public let binary: Data

    /// Wraps archive bytes after checking the format name, version and view bounds.
    public init(manifest: Data, binary: Data) throws {
        let document = try Self.decode(manifest)
        guard document.binaryLength == binary.count else { throw Failure.binaryLength }
        _ = try ArchiveBinaryReader(data: binary, views: document.views)
        self.manifest = manifest
        self.binary = binary
    }

    /// Reads an archive directory written by `write(to:)`. Both files must be regular files.
    public init(contentsOf directory: URL) throws {
        func read(_ name: String) throws -> Data {
            let url = directory.appendingPathComponent(name, isDirectory: false)
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            guard values.isRegularFile == true, values.isSymbolicLink != true else { throw Failure.notRegularFile(name) }
            return try Data(contentsOf: url)
        }
        try self.init(manifest: read(Self.manifestFileName), binary: read(Self.binaryFileName))
    }

    /// Writes the manifest and sidecar into a directory, creating it when needed.
    public func write(to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try manifest.write(to: directory.appendingPathComponent(Self.manifestFileName), options: .atomic)
        try binary.write(to: directory.appendingPathComponent(Self.binaryFileName), options: .atomic)
    }

    /// Captures a graph, its clips and an optional light rig.
    ///
    /// Validates the graph and optional light rig first. Textures are read back with blits on
    /// `commandQueue`, or on a new queue for each texture's device, and this call waits
    /// for them; finish GPU writes to those textures first. Surface-graph uniforms and
    /// instance sets are captured with their current values.
    public init(
        capturing graph: SceneGraph, animations: [AnimationClip] = [], lightRig: LightRig? = nil,
        commandQueue: (any MTLCommandQueue)? = nil
    ) throws {
        if let batchNode = try graph.resolve().first(where: { $0.batch != nil }) {
            throw Failure.unsupportedBatch(batchNode.id)
        }
        let lightRig = try lightRig.map { try LightRig(lights: $0.lights) }
        var capture = ArchiveCapture(queue: commandQueue)
        let document = try capture.document(graph: graph, animations: animations, lightRig: lightRig)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes]
        do {
            manifest = try encoder.encode(document)
        } catch {
            throw Failure.encoding(error.localizedDescription)
        }
        binary = capture.writer.data
    }

    /// What an archive recreates.
    public struct Contents {
        public var graph: SceneGraph
        public var animations: [AnimationClip]
        public var lightRig: LightRig?
    }

    /// Recreates the scene on a device. Renderers use their archived color format, or
    /// `colorPixelFormat` when supplied. Textures upload with blits on `commandQueue`,
    /// or on a new queue, and this call waits for them.
    public func load(
        device: any MTLDevice, commandQueue: (any MTLCommandQueue)? = nil, colorPixelFormat: MTLPixelFormat? = nil
    ) throws -> Contents {
        if let commandQueue, commandQueue.device !== device { throw Failure.wrongDevice }
        guard let queue = commandQueue ?? device.makeCommandQueue() else { throw Failure.resource("command queue") }
        let document = try Self.decode(manifest)
        var loader = ArchiveLoader(
            document: document, reader: try ArchiveBinaryReader(data: binary, views: document.views), queue: queue,
            colorPixelFormat: colorPixelFormat
        )
        return try loader.contents()
    }

    private static func decode(_ manifest: Data) throws -> ArchiveManifest {
        struct Header: Decodable {
            var format: String
            var version: Int
        }
        let header: Header
        do {
            header = try JSONDecoder().decode(Header.self, from: manifest)
        } catch {
            throw Failure.decoding(error.localizedDescription)
        }
        guard header.format == formatName else { throw Failure.format(header.format) }
        guard (1...formatVersion).contains(header.version) else { throw Failure.unsupportedVersion(header.version) }
        let document: ArchiveManifest
        do {
            document = try JSONDecoder().decode(ArchiveManifest.self, from: manifest)
        } catch {
            throw Failure.decoding(String(describing: error))
        }
        if document.version == 1,
           document.animations.contains(where: { $0.propertyTracks != nil })
            || document.materials.contains(where: { $0.liveFactors != nil }) {
            throw Failure.invalidAnimation("typed tracks and live materials require archive version 2")
        }
        if document.version < 3,
           document.photometry != nil || document.scene.nodes.contains(where: { $0.photometry != nil }) {
            throw Failure.invalidPhotometry("photometric resources require archive version 3")
        }
        if document.scene.lightProbes != nil {
            guard document.version >= 6 else { throw Failure.invalidLightProbe("light probes require archive version 6") }
            _ = try document.scene.decodedLightProbes()
        }
        if document.version < 7, document.surfaceGraphs.contains(where: \.hasCoverageOutputs) {
            throw Failure.decoding("surface graph coverage requires archive version 7")
        }
        if document.version < 8, document.scene.nodes.contains(where: { $0.pivot != nil }) {
            throw Failure.decoding("node pivots require archive version 8")
        }
        if document.version < 9, document.meshes.contains(where: { $0.drawRange != nil }) {
            throw Failure.decoding("mesh draw ranges require archive version 9")
        }
        if document.version < 10, document.scene.nodes.contains(where: { $0.orientationMode != nil }) {
            throw Failure.decoding("world orientation requires archive version 10")
        }
        if document.version < 11, (document.primitives ?? []).contains(where: { $0.drawRange != nil }) {
            throw Failure.decoding("primitive draw ranges require archive version 11")
        }
        if document.version < 12, document.scene.nodes.contains(where: { $0.ambientLight != nil }) {
            throw Failure.decoding("ambient lights require archive version 12")
        }
        if document.version < 13, document.materials.contains(where: { $0.classic?.hasExplicitDepthOutput == true }) {
            throw Failure.decoding("depth and distance outputs require archive version 13")
        }
        if document.version < 14, document.materials.contains(where: { $0.physicallyBased?.bumpMap != nil }) {
            throw Failure.decoding("bump maps require archive version 14")
        }
        if document.version < 15, document.materials.contains(where: { $0.classic?.hasSurfaceDetail == true }) {
            throw Failure.decoding("classic surface detail requires archive version 15")
        }
        if document.version < 16, document.scene.environment != nil {
            throw Failure.decoding("scene environment requires archive version 16")
        }
        if document.version < 17, document.surfaceGraphs.contains(where: \.hasViewportMipLevelNode) {
            throw Failure.decoding("surface viewport mip levels require archive version 17")
        }
        if document.version < 18, document.scene.environmentEulerRotation != nil {
            throw Failure.decoding("scene environment Euler rotation requires archive version 18")
        }
        if document.version < 19, document.surfaceGraphs.contains(where: \.hasSoftParticleNode) {
            throw Failure.decoding("surface soft particles require archive version 19")
        }
        if document.version < 20, (document.primitives ?? []).contains(where: { $0.softParticleFade != nil }) {
            throw Failure.decoding("soft point and sprite materials require archive version 20")
        }
        if document.version < 21, document.meshes.contains(where: \.hasStencilState) {
            throw Failure.decoding("mesh stencil state requires archive version 21")
        }
        if document.version < 22, document.materials.contains(where: { $0.physicallyBased?.hasLightMapSettings == true }) {
            throw Failure.decoding("PBR light maps require archive version 22")
        }
        if document.version < 23, document.materials.contains(where: { $0.physicallyBased?.retroreflectivity != nil }) {
            throw Failure.decoding("PBR retroreflectivity requires archive version 23")
        }
        if document.version < 24, document.materials.contains(where: { $0.classic?.environmentMap?.refractionRatio != nil }) {
            throw Failure.decoding("classic environment refraction requires archive version 24")
        }
        if document.version < 25, document.animations.contains(where: \.hasVisibilityOrRGBTrack) {
            throw Failure.invalidAnimation("node visibility and material RGB bindings require archive version 25")
        }
        if document.version < 26, document.scene.nodes.contains(where: \.hasPunctualShadowStrength) {
            throw Failure.decoding("punctual shadow strength requires archive version 26")
        }
        if document.version < 27, document.materials.contains(where: { $0.physicallyBased?.unlitUsesGraphEmission == true }) {
            throw Failure.decoding("unlit graph emission requires archive version 27")
        }
        if document.version < 28, document.surfaceGraphs.contains(where: \.hasViewportSafeUVNode) {
            throw Failure.decoding("depth-safe viewport coordinates require archive version 28")
        }
        if document.version < 29, document.surfaceGraphs.contains(where: \.hasCastShadowOutput) {
            throw Failure.decoding("surface graph colored shadow outputs require archive version 29")
        }
        if document.version < 30, document.materials.contains(where: { $0.classic?.surfaceGraph != nil }) {
            throw Failure.decoding("classic node graphs require archive version 30")
        }
        let rotation = (document.scene.environmentEulerRotation ?? .zero) + SIMD3(0, document.scene.environmentRotation, 0)
        guard rotation.isFiniteVector else { throw Failure.decoding("scene environment rotation angles must be finite") }
        return document
    }

    public enum Failure: LocalizedError, Equatable {
        case format(String)
        case unsupportedVersion(Int)
        case decoding(String)
        case encoding(String)
        case binaryLength
        case invalidView(Int)
        case invalidReference(String)
        case notRegularFile(String)
        case unsupportedMaterial(String)
        case unsupportedBatch(UUID)
        case hierarchyDepth
        case unrecordedSampler(String)
        case invalidSampler
        case unsupportedTexture(String)
        case unsupportedTextureFormat(UInt)
        case simulatorCompressedMipChain
        case invalidTexture(String)
        case invalidAnimation(String)
        case invalidPhotometry(String)
        case invalidLightProbe(String)
        case nonfiniteMetadata(UUID)
        case wrongDevice
        case resource(String)
        case gpu(String)

        public var errorDescription: String? {
            switch self {
            case .format(let name): "The manifest's format is \(name), not \(SceneArchive.formatName)."
            case .unsupportedVersion(let version):
                "Scene archive version \(version) is not supported; this build reads versions 1 through \(SceneArchive.formatVersion)."
            case .decoding(let detail): "The scene manifest could not be decoded: \(detail)"
            case .encoding(let detail): "The scene manifest could not be encoded: \(detail)"
            case .binaryLength: "The binary sidecar's length differs from the length the manifest records."
            case .invalidView(let index): "Binary view \(index) is missing, outside the sidecar, or the wrong size for its data."
            case .invalidReference(let detail): "The manifest refers to a missing \(detail)."
            case .notRegularFile(let name): "\(name) must be a regular file, not a link or directory."
            case .unsupportedMaterial(let detail): "This material cannot be archived: \(detail)"
            case .unsupportedBatch(let id): "Scene archive cannot capture the batched mesh at node \(id)."
            case .hierarchyDepth: "Scene archives support hierarchies up to \(SceneArchive.maximumHierarchyDepth) levels deep."
            case .unrecordedSampler(let detail):
                "A sampler on \(detail) was not created through TextureSampling or glTF preparation, so its settings are unknown."
            case .invalidSampler: "Sampler anisotropy must be 1 through 16 with finite, ordered, nonnegative LOD clamps."
            case .unsupportedTexture(let label):
                "Texture \(label) cannot be archived. Use a single-sample 2D, 2D array, cube, cube array or 3D texture that is not framebuffer-only or memoryless."
            case .unsupportedTextureFormat(let raw): "Pixel format \(raw) is not supported by scene archives."
            case .simulatorCompressedMipChain: "Metal Simulator cannot archive a mipmapped ASTC texture without risking corrupt lower mip levels."
            case .invalidTexture(let label): "The archive's texture \(label) has an invalid descriptor or data size."
            case .invalidAnimation(let detail): "An archived animation channel is invalid: \(detail)."
            case .invalidPhotometry(let detail): "An archived light resource is invalid: \(detail)."
            case .invalidLightProbe(let detail): "Invalid archived light probe. \(detail)"
            case .nonfiniteMetadata(let id): "Scene node \(id) has nonfinite metadata numbers, which JSON cannot store."
            case .wrongDevice: "The command queue must use the same device as the resources."
            case .resource(let name): "Metal could not create the archive's \(name)."
            case .gpu(let detail): "An archive GPU copy failed: \(detail)"
            }
        }
    }
}

// MARK: Manifest records

struct ArchiveManifest: Codable {
    var format: String
    var version: Int
    var generator: String
    var binaryLength: Int
    var views: [ArchiveView]
    var scene: ArchiveScene
    var meshes: [ArchiveMesh]
    var primitives: [ArchivePrimitive]?
    var materials: [ArchiveMaterial]
    var textures: [ArchiveTextureRecord]
    var samplers: [TextureSampling]
    var environments: [ArchiveEnvironment]
    var surfaceGraphs: [SurfaceShaderGraph]
    var uniforms: [ArchiveUniforms]
    var instanceSets: [ArchiveInstanceSet]
    var animations: [ArchiveClip]
    var lightRig: LightRig?
    var photometry: ArchivePhotometry?
}

struct ArchiveScene: Codable {
    var camera: PerspectiveCamera
    var cameraTransform: SceneTransform
    var cameraLayers: SceneLayers
    var environment: Int?
    var environmentIntensity: Float
    var environmentRotation: Float
    var environmentEulerRotation: SIMD3<Float>?
    var lightProbes: [ArchiveLightProbe]?
    var projectionJitter: SIMD2<Float>?
    /// Every node in depth-first pre-order, so each parent precedes its children and
    /// siblings keep their order. A flat list keeps encoding and decoding depth constant
    /// however deep the hierarchy is.
    var nodes: [ArchiveNode]

    func decodedLightProbes() throws -> [LightProbe] {
        guard let lightProbes else { return [] }
        do {
            let probes = try lightProbes.map { try LightProbe(coefficients: $0.coefficients, intensity: $0.intensity) }
            _ = try LightProbe.rasterLighting(probes)
            return probes
        } catch {
            throw SceneArchive.Failure.invalidLightProbe(error.localizedDescription)
        }
    }
}

struct ArchiveLightProbe: Codable {
    var coefficients: [SIMD3<Float>]
    var intensity: Float

    init(_ probe: LightProbe) {
        coefficients = probe.coefficients
        intensity = probe.intensity
    }
}

struct ArchiveNode: Codable {
    /// The parent's index in `ArchiveScene.nodes`, which must be lower than this node's; nil for a root.
    var parent: Int?
    var id: UUID
    var name: String?
    var transform: SceneTransform
    var pivot: SIMD3<Float>?
    var orientationMode: SceneNode.OrientationMode?
    var mesh: Int?
    var primitive: Int?
    var materialParameters: SIMD4<Float>
    var castsShadow: Bool
    var jointPalette: Int?
    var light: PunctualLight?
    var hasPunctualShadowStrength: Bool { (light?.shadow?.shadowStrength ?? 1) != 1 }
    var ambientLight: AmbientLight?
    var instances: Int?
    var morphWeights: [Float]?
    var frustumCulled: Bool
    var cullingBounds: AxisAlignedBox?
    var levelsOfDetail: ArchiveLevels?
    var isVisible: Bool
    var layers: SceneLayers
    var renderOrder: Int
    var metadata: [String: SceneMetadataValue]?
    var photometry: ArchivePhotometry.References?
}

struct ArchiveLevels: Codable {
    struct Level: Codable {
        var mesh: Int
        var distance: Float
        var hysteresis: Float
    }

    var levels: [Level]
    var currentLevel: Int
}

struct ArchiveMesh: Codable {
    var colorPixelFormat: UInt
    var reversedDepth: Bool
    var supportsInstancing: Bool
    var sampleCount: Int?
    var renderState: ArchiveMeshState?
    var usesStencilAttachment: Bool?
    var material: Int
    var geometry: ArchiveGeometry
    var drawRange: ArchiveDrawRange?
    var skin: ArchiveSkin?
    var morphTargets: ArchiveMorphTargets?

    var hasStencilState: Bool { usesStencilAttachment != nil || renderState?.stencil != nil }
}

struct ArchiveDrawRange: Codable {
    var start: Int
    var count: Int?
}

struct ArchiveGeometry: Codable {
    var vertexCount: Int
    var indexCount: Int
    /// Three floats per vertex.
    var positions: Int
    var normals: Int
    /// Two floats per vertex.
    var textureCoordinates: Int
    var secondaryTextureCoordinates: Int?
    /// One byte per vertex, present when only some vertices have the attribute.
    var secondaryPresence: Int?
    /// Four floats per vertex.
    var tangents: Int?
    var tangentPresence: Int?
    var colors: Int?
    var colorPresence: Int?
    var groups: [ArchiveGeometryGroup]?
    /// UInt32 indices.
    var indices: Int
}

struct ArchiveGeometryGroup: Codable {
    var start: Int
    var count: Int
    var materialIndex: Int
}

struct ArchiveSkin: Codable {
    var jointCount: Int
    /// Four UInt16 joints per vertex.
    var joints: Int
    /// Four floats per vertex.
    var weights: Int
}

struct ArchiveMorphTargets: Codable {
    struct Target: Codable {
        var positions: Int?
        var normals: Int?
        var tangents: Int?
    }

    var vertexCount: Int
    var targets: [Target]
}

struct ArchiveMaterial: Codable {
    var kind: String
    var texture: Int?
    var physicallyBased: ArchivePBR?
    var classic: ArchiveClassicMaterial?
    var liveFactors: [ArchiveLiveFactor]?
}

struct ArchiveBinding: Codable {
    var texture: Int
    var sampler: Int?
    var sampling: PhysicallyBasedMaterial.TextureSampling?
    var coordinateSet: MeshTextureCoordinateSet
    var transform: PhysicallyBasedMaterial.TextureTransform
    var normalEncoding: PhysicallyBasedMaterial.NormalEncoding?
    var normalSpace: PhysicallyBasedMaterial.NormalSpace?
}

struct ArchivePBR: Codable {
    struct LightMap: Codable {
        var texture: Int
        var coordinateSet: MeshTextureCoordinateSet
        var transform: PhysicallyBasedMaterial.TextureTransform
    }

    struct BumpMap: Codable {
        var texture: ArchiveBinding
        var scale: Float
    }

    struct Environment: Codable {
        var index: Int
        var intensity: Float
        var yaw: Float
    }

    struct Clearcoat: Codable {
        var factor: Float
        var roughness: Float
        var factorTexture: ArchiveBinding?
        var roughnessTexture: ArchiveBinding?
        var normalTexture: ArchiveBinding?
        var normalScale: Float
    }

    struct Transmission: Codable {
        var factor: Float
        var indexOfRefraction: Float
        var thickness: Float
        var attenuationDistance: Float?
        var attenuationColor: SIMD3<Float>
        var factorTexture: ArchiveBinding?
        var thicknessTexture: ArchiveBinding?
        var dispersion: Float
    }

    struct Specular: Codable {
        var factor: Float
        var colorFactor: SIMD3<Float>
        var factorTexture: ArchiveBinding?
        var colorTexture: ArchiveBinding?
    }

    struct Sheen: Codable {
        var colorFactor: SIMD3<Float>
        var roughness: Float
        var colorTexture: ArchiveBinding?
        var roughnessTexture: ArchiveBinding?
    }

    struct Anisotropy: Codable {
        var strength: Float
        var rotation: Float
        var texture: ArchiveBinding?
    }

    struct Iridescence: Codable {
        var factor: Float
        var indexOfRefraction: Float
        var minimumThickness: Float
        var maximumThickness: Float
        var factorTexture: ArchiveBinding?
        var thicknessTexture: ArchiveBinding?
    }

    struct DiffuseTransmission: Codable {
        var factor: Float
        var colorFactor: SIMD3<Float>
        var factorTexture: ArchiveBinding?
        var colorTexture: ArchiveBinding?
    }

    struct DiffuseRoughness: Codable {
        var factor: Float
        var texture: ArchiveBinding?
    }

    struct VolumeScatter: Codable {
        var multiscatterColor: SIMD3<Float>
        var anisotropy: Float
        var multiscatterColorTexture: ArchiveBinding?
    }

    struct Subsurface: Codable {
        var weight: Float
        var color: SIMD3<Float>
        var radius: SIMD3<Float>
        var scale: Float
        var anisotropy: Float
        var colorTexture: ArchiveBinding?
    }

    var baseColorTexture: Int?
    var baseColorSampler: Int?
    var baseColorSampling: PhysicallyBasedMaterial.TextureSampling?
    var baseColorTextureCoordinateSet: MeshTextureCoordinateSet
    var baseColorTextureTransform: PhysicallyBasedMaterial.TextureTransform
    var baseColorFactor: SIMD4<Float>
    var metallic: Float
    var roughness: Float
    var directionToLight: SIMD3<Float>
    var lightIntensity: Float
    var ambientIntensity: Float
    var receivesShadows: Bool
    var metallicRoughnessTexture: ArchiveBinding?
    var normalTexture: ArchiveBinding?
    var normalScale: Float
    var bumpMap: BumpMap?
    var lightMap: LightMap?
    var lightMapIntensity: Float?
    var retroreflectivity: Float?
    var hasLightMapSettings: Bool { lightMap != nil || lightMapIntensity != nil }
    var occlusionTexture: ArchiveBinding?
    var occlusionStrength: Float
    var emissiveTexture: ArchiveBinding?
    var emissiveFactor: SIMD3<Float>
    var emissiveStrength: Float
    var alphaTexture: ArchiveBinding?
    var alphaMode: String
    var alphaCutoff: Float?
    var isDoubleSided: Bool
    var environment: Environment?
    var clearcoat: Clearcoat?
    var transmission: Transmission?
    var specular: Specular?
    var sheen: Sheen?
    var anisotropy: Anisotropy?
    var iridescence: Iridescence?
    var diffuseTransmission: DiffuseTransmission?
    var diffuseRoughness: DiffuseRoughness?
    var volumeScatter: VolumeScatter?
    var subsurface: Subsurface?
    var surfaceGraph: Int?
    var surfaceUniforms: Int?
    var surfaceTextures: [String: Int]?
    var unlit: Bool
    var unlitUsesGraphEmission: Bool?
}

struct ArchiveEnvironment: Codable {
    var sky: Int
    var diffuse: Int
    var specular: Int
    var brdf: Int
}

struct ArchiveUniforms: Codable {
    var declarations: [ShaderUniform]
    /// One float4 per value slot, in declaration order. A matrix contributes one slot per column.
    /// An integer uniform's slot is zero here. JSON numbers cannot carry every IEEE bit pattern,
    /// and a signed integer such as -2 is a NaN when those bits are read as a float.
    var values: [SIMD4<Float>]
    /// Signed values for `.int` uniforms, in declaration order. Absent when the set has none.
    var integerValues: [Int32]?

    init(declarations: [ShaderUniform], values: [SIMD4<Float>]) {
        var stored = values
        var integers: [Int32] = []
        var cursor = 0
        for declaration in declarations {
            if declaration.type == .int, stored.indices.contains(cursor) {
                integers.append(Int32(bitPattern: stored[cursor].x.bitPattern))
                stored[cursor] = .zero
            }
            cursor += declaration.type.uniformSlotCount
        }
        self.declarations = declarations
        self.values = stored
        self.integerValues = integers.isEmpty ? nil : integers
    }

    private enum CodingKeys: String, CodingKey { case declarations, values, integerValues }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        declarations = try container.decode([ShaderUniform].self, forKey: .declarations)
        values = try container.decode([SIMD4<Float>].self, forKey: .values)
        integerValues = try container.decodeIfPresent([Int32].self, forKey: .integerValues)
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(declarations, forKey: .declarations)
        try container.encode(values, forKey: .values)
        try container.encodeIfPresent(integerValues, forKey: .integerValues)
    }
}

struct ArchiveInstanceSet: Codable {
    var capacity: Int
    var count: Int
    /// Four float4 columns per instance.
    var transforms: Int
    /// One float4 per instance.
    var tints: Int
}

struct ArchiveClip: Codable {
    struct Channel: Codable {
        var sourceNodeIndex: Int
        var target: String
        var interpolation: String
        var times: Int
        var values: Int?
        var weights: Int?
        var weightCount: Int
    }

    struct Target: Codable {
        var node: UUID
        /// Eleven floats per key: time, translation, rotation (x, y, z, w), scale.
        var keyframes: Int?
        var channels: [Channel]?
    }

    struct Pose: Codable {
        var translation: SIMD3<Float>?
        var rotation: SIMD4<Float>?
        var scale: SIMD3<Float>?
        var morphWeights: [Float]?
    }

    var name: String?
    var duration: Float
    var additive: Bool
    var targets: [Target]
    var reference: [String: Pose]?
    var propertyTracks: [ArchivePropertyTrack]?

    var hasVisibilityOrRGBTrack: Bool {
        propertyTracks?.contains { track in
            switch track.binding {
            case .node(_, .isVisible), .material(_, .baseColorRGB): true
            default: false
            }
        } ?? false
    }
}

// MARK: Capture

private struct ArchiveCapture {
    private let suppliedQueue: (any MTLCommandQueue)?
    private var queues: [ObjectIdentifier: any MTLCommandQueue] = [:]
    var writer = ArchiveBinaryWriter()
    private var meshes: [ArchiveMesh] = []
    private var meshIndices: [ObjectIdentifier: Int] = [:]
    private var primitives: [ArchivePrimitive] = []
    private var primitiveIndices: [ObjectIdentifier: Int] = [:]
    private var materials: [ArchiveMaterial] = []
    private var liveMaterialIndices: [ObjectIdentifier: Int] = [:]
    private var textures: [ArchiveTextureRecord] = []
    private var textureIndices: [ObjectIdentifier: Int] = [:]
    private var samplers: [TextureSampling] = []
    private var environments: [ArchiveEnvironment] = []
    private var environmentIndices: [ObjectIdentifier: Int] = [:]
    private var graphs: [SurfaceShaderGraph] = []
    private var uniforms: [ArchiveUniforms] = []
    private var uniformIndices: [ObjectIdentifier: Int] = [:]
    private var instanceSets: [ArchiveInstanceSet] = []
    private var instanceIndices: [ObjectIdentifier: Int] = [:]
    private var photometry = ArchivePhotometryCapture()
    private var hasPackedNormals = false
    private var hasObjectSpaceNormals = false
    // Keep captured objects alive so their identifiers cannot be reused during capture.
    private var retained: [AnyObject] = []

    init(queue: (any MTLCommandQueue)?) {
        suppliedQueue = queue
    }

    mutating func document(graph: SceneGraph, animations: [AnimationClip], lightRig: LightRig?) throws -> ArchiveManifest {
        guard graph.lightProbeGridLighting == nil else {
            throw SceneArchive.Failure.encoding("scene light probe volumes need to be restored separately")
        }
        guard graph.effectiveEnvironmentEulerRotation.isFiniteVector else {
            throw SceneArchive.Failure.encoding("scene environment rotation angles must be finite")
        }
        // Iterative pre-order walk: deep hierarchies cannot overflow the stack.
        var nodes: [ArchiveNode] = []
        var pending = graph.roots.reversed().map { (node: $0, parent: Int?.none, depth: 1) }
        while let (node, parent, depth) = pending.popLast() {
            guard depth <= SceneArchive.maximumHierarchyDepth else { throw SceneArchive.Failure.hierarchyDepth }
            let index = nodes.count
            nodes.append(try self.node(node, parent: parent))
            pending.append(contentsOf: node.children.reversed().map { ($0, index, depth + 1) })
        }
        let scene = ArchiveScene(
            camera: graph.camera, cameraTransform: graph.cameraTransform, cameraLayers: graph.cameraLayers,
            environment: try graph.environment.map { try environment($0) },
            environmentIntensity: graph.environmentIntensity, environmentRotation: graph.environmentRotation,
            environmentEulerRotation: graph.environmentEulerRotation == .zero ? nil : graph.environmentEulerRotation,
            lightProbes: graph.lightProbes.isEmpty ? nil : graph.lightProbes.map(ArchiveLightProbe.init),
            projectionJitter: graph.projectionJitter == .zero ? nil : graph.projectionJitter, nodes: nodes
        )
        _ = try scene.decodedLightProbes()
        let bindings = ArchiveAnimationTargets(nodes: nodes, meshes: meshes, lightRig: lightRig)
        let clips = try animations.map { try clip($0, bindings: bindings) }
        let needsVersion2 = clips.contains { $0.propertyTracks != nil } || materials.contains { $0.liveFactors != nil }
        let rig = photometry.rig(lightRig, writer: &writer)
        let lighting = photometry.record
        let version = if materials.contains(where: { $0.classic?.surfaceGraph != nil }) { 30 }
        else if graphs.contains(where: \.hasCastShadowOutput) { 29 }
        else if graphs.contains(where: \.hasViewportSafeUVNode) { 28 }
        else if materials.contains(where: { $0.physicallyBased?.unlitUsesGraphEmission == true }) { 27 }
        else if nodes.contains(where: \.hasPunctualShadowStrength) { 26 }
        else if clips.contains(where: \.hasVisibilityOrRGBTrack) { 25 }
        else if materials.contains(where: { $0.classic?.environmentMap?.refractionRatio != nil }) { 24 }
        else if materials.contains(where: { $0.physicallyBased?.retroreflectivity != nil }) { 23 }
        else if materials.contains(where: { $0.physicallyBased?.hasLightMapSettings == true }) { 22 }
        else if meshes.contains(where: \.hasStencilState) { 21 }
        else if primitives.contains(where: { $0.softParticleFade != nil }) { 20 }
        else if graphs.contains(where: \.hasSoftParticleNode) { 19 }
        else if scene.environmentEulerRotation != nil { 18 }
        else if graphs.contains(where: \.hasViewportMipLevelNode) { 17 }
        else if scene.environment != nil { 16 }
        else if materials.contains(where: { $0.classic?.hasSurfaceDetail == true }) { 15 }
        else if materials.contains(where: { $0.physicallyBased?.bumpMap != nil }) { 14 }
        else if materials.contains(where: { $0.classic?.hasExplicitDepthOutput == true }) { 13 }
        else if nodes.contains(where: { $0.ambientLight != nil }) { 12 }
        else if primitives.contains(where: { $0.drawRange != nil }) { 11 }
        else if nodes.contains(where: { $0.orientationMode != nil }) { 10 }
        else if meshes.contains(where: { $0.drawRange != nil }) { 9 }
        else if nodes.contains(where: { $0.pivot != nil }) { 8 }
        else if graphs.contains(where: \.hasCoverageOutputs) { 7 }
        else if scene.lightProbes != nil { 6 }
        else if hasObjectSpaceNormals { 5 }
        else if hasPackedNormals { 4 }
        else if lighting != nil { 3 }
        else { needsVersion2 ? 2 : 1 }
        return ArchiveManifest(
            format: SceneArchive.formatName, version: version, generator: "Frostfield",
            binaryLength: writer.data.count, views: writer.views, scene: scene, meshes: meshes,
            primitives: primitives.isEmpty ? nil : primitives, materials: materials,
            textures: textures, samplers: samplers, environments: environments, surfaceGraphs: graphs,
            uniforms: uniforms, instanceSets: instanceSets, animations: clips, lightRig: rig, photometry: lighting
        )
    }

    private mutating func node(_ node: SceneNode, parent: Int?) throws -> ArchiveNode {
        guard node.metadata.values.allSatisfy(\.isFinite) else { throw SceneArchive.Failure.nonfiniteMetadata(node.id) }
        let levels = try node.levelsOfDetail.map { lod in
            ArchiveLevels(
                levels: try lod.levels.map { .init(mesh: try mesh($0.mesh), distance: $0.distance, hysteresis: $0.hysteresis) },
                currentLevel: lod.currentLevel
            )
        }
        let lighting = try photometry.node(node.light, writer: &writer)
        return ArchiveNode(
            parent: parent, id: node.id, name: node.name, transform: node.transform, pivot: node.pivot,
            orientationMode: node.orientationMode == .inherited ? nil : node.orientationMode,
            mesh: try node.mesh.map { try mesh($0) },
            primitive: try node.primitive.map { try primitive($0) },
            materialParameters: node.materialParameters, castsShadow: node.castsShadow,
            jointPalette: node.jointPalette.map { writer.append(ArchiveMatrices.pack($0.matrices)) },
            light: lighting.light, ambientLight: node.ambientLight,
            instances: node.instances.map { instanceSet($0) }, morphWeights: node.morphWeights,
            frustumCulled: node.frustumCulled, cullingBounds: node.cullingBounds, levelsOfDetail: levels,
            isVisible: node.isVisible, layers: node.layers, renderOrder: node.renderOrder,
            metadata: node.metadata.isEmpty ? nil : node.metadata, photometry: lighting.resources
        )
    }

    private mutating func instanceSet(_ set: InstanceSet) -> Int {
        if let index = instanceIndices[ObjectIdentifier(set)] { return index }
        retained.append(set)
        instanceSets.append(ArchiveInstanceSet(
            capacity: set.capacity, count: set.count, transforms: writer.append(ArchiveMatrices.pack(set.transforms)),
            tints: writer.append(set.tints)
        ))
        instanceIndices[ObjectIdentifier(set)] = instanceSets.count - 1
        return instanceSets.count - 1
    }

    private mutating func primitive(_ primitive: ScenePrimitive) throws -> Int {
        let identity: ObjectIdentifier = switch primitive {
        case .points(let renderer): ObjectIdentifier(renderer)
        case .lines(let renderer): ObjectIdentifier(renderer)
        case .sprites(let renderer): ObjectIdentifier(renderer)
        }
        if let existing = primitiveIndices[identity] { return existing }

        let color: SIMD4<Float>
        let drawRange: GeometryDrawRange
        let content: ArchivePrimitive.Content
        switch primitive {
        case .points(let renderer):
            let cloud = renderer.cloud
            let material = renderer.material
            color = material.color
            drawRange = renderer.drawRange
            content = .points(.init(
                count: cloud.positions.count, positions: writer.append(Self.flatten(cloud.positions)),
                colors: cloud.colors.map { writer.append($0) }, sizes: cloud.sizes.map { writer.append($0) },
                extent: .init(material.size), circular: material.shape == .circle,
                map: try material.map.map { try texture($0) }, alphaTest: material.alphaTest
            ))
        case .lines(let renderer):
            let geometry = renderer.geometry
            let material = renderer.material
            color = material.color
            drawRange = renderer.drawRange
            let extent: ArchivePrimitive.Extent? = switch material.width {
            case .hairline: nil
            case .pixels(let value): .pixels(value)
            case .world(let value): .world(value)
            }
            content = .lines(.init(
                count: geometry.positions.count, positions: writer.append(Self.flatten(geometry.positions)),
                colors: geometry.colors.map { writer.append($0) }, topology: .init(geometry.topology),
                extent: extent, join: .init(material.join), cap: .init(material.cap),
                dash: material.dash.map { .init(length: $0.length, gap: $0.gap, offset: $0.offset) }
            ))
        case .sprites(let renderer):
            let material = renderer.material
            color = material.color
            drawRange = .all
            content = .sprites(.init(
                count: renderer.sprites.count, records: writer.append(ArchivePrimitive.Sprites.pack(renderer.sprites)),
                map: try material.map.map { try texture($0) }, alphaTest: material.alphaTest,
                sizeAttenuation: material.sizeAttenuation
            ))
        }
        let index = primitives.count
        primitives.append(ArchivePrimitive(
            colorPixelFormat: primitive.colorPixelFormat.rawValue, sampleCount: primitive.sampleCount,
            color: color, blending: .init(primitive.blending), depthTest: primitive.depthTest,
            drawRange: drawRange == .all ? nil : ArchiveDrawRange(start: drawRange.start, count: drawRange.count),
            content: content, softParticleFade: primitive.softParticleFade.map(ArchivePrimitive.Fade.init)
        ))
        primitiveIndices[identity] = index
        return index
    }

    private mutating func mesh(_ renderer: MeshRenderer) throws -> Int {
        if let index = meshIndices[ObjectIdentifier(renderer)] { return index }
        guard renderer.depthMode != .logarithmic else {
            throw SceneArchive.Failure.unsupportedMaterial("logarithmic depth needs runtime pipeline preparation")
        }
        guard renderer.surfaceRenderTargets == nil else {
            throw SceneArchive.Failure.unsupportedMaterial("named surface render targets require a runtime pass layout")
        }
        retained.append(renderer)
        let material = try self.material(renderer.material)
        let record = ArchiveMesh(
            colorPixelFormat: renderer.colorPixelFormat.rawValue, reversedDepth: renderer.depthMode == .reversed,
            supportsInstancing: renderer.supportsInstancing, sampleCount: renderer.sampleCount,
            renderState: ArchiveMeshState(renderer.renderState), usesStencilAttachment: renderer.usesStencilAttachment ? true : nil,
            material: material, geometry: geometry(renderer.geometry),
            drawRange: renderer.drawRange == .all ? nil : ArchiveDrawRange(
                start: renderer.drawRange.start, count: renderer.drawRange.count
            ),
            skin: renderer.skin.map { skin in
                ArchiveSkin(
                    jointCount: skin.jointCount, joints: writer.append(skin.influences.map(\.joints)),
                    weights: writer.append(skin.influences.map(\.weights))
                )
            },
            morphTargets: renderer.morphTargets.map { morph in
                ArchiveMorphTargets(vertexCount: morph.vertexCount, targets: morph.targets.map { target in
                    .init(
                        positions: target.positionOffsets.map { writer.append(Self.flatten($0)) },
                        normals: target.normalOffsets.map { writer.append(Self.flatten($0)) },
                        tangents: target.tangentOffsets.map { writer.append(Self.flatten($0)) }
                    )
                })
            }
        )
        meshes.append(record)
        meshIndices[ObjectIdentifier(renderer)] = meshes.count - 1
        return meshes.count - 1
    }

    private mutating func geometry(_ geometry: MeshGeometry) -> ArchiveGeometry {
        let vertices = geometry.vertices
        func optional<T>(_ values: [T?], zero: T) -> (Int?, Int?) {
            let present = values.filter { $0 != nil }.count
            guard present > 0 else { return (nil, nil) }
            let data = writer.append(values.map { $0 ?? zero })
            let mask = present == values.count ? nil : writer.append(values.map { $0 == nil ? UInt8(0) : 1 })
            return (data, mask)
        }
        let secondary = optional(vertices.map { $0.secondaryTextureCoordinate }, zero: SIMD2<Float>.zero)
        let tangents = optional(vertices.map(\.tangent), zero: SIMD4<Float>.zero)
        let colors = optional(vertices.map(\.color), zero: SIMD4<Float>.zero)
        let groups = geometry.groups.map {
            ArchiveGeometryGroup(start: $0.start, count: $0.count, materialIndex: $0.materialIndex)
        }
        return ArchiveGeometry(
            vertexCount: vertices.count, indexCount: geometry.indices.count,
            positions: writer.append(Self.flatten(vertices.map(\.position))),
            normals: writer.append(Self.flatten(vertices.map(\.normal))),
            textureCoordinates: writer.append(vertices.map(\.textureCoordinate)),
            secondaryTextureCoordinates: secondary.0, secondaryPresence: secondary.1,
            tangents: tangents.0, tangentPresence: tangents.1, colors: colors.0, colorPresence: colors.1,
            groups: groups.isEmpty ? nil : groups, indices: writer.append(geometry.indices)
        )
    }

    /// Packs float3 values as three floats, leaving out the SIMD padding lane.
    static func flatten(_ values: [SIMD3<Float>]) -> [Float] {
        values.flatMap { [$0.x, $0.y, $0.z] }
    }

    private mutating func material(_ material: MeshMaterial) throws -> Int {
        let record: ArchiveMaterial
        switch material {
        case .unlit: record = ArchiveMaterial(kind: "unlit")
        case .lambert: record = ArchiveMaterial(kind: "lambert")
        case .shadowedLambert: record = ArchiveMaterial(kind: "shadowedLambert")
        case .textured(let texture): record = ArchiveMaterial(kind: "textured", texture: try self.texture(texture))
        case .physicallyBased(let pbr):
            if let live = pbr.liveFactors { return try liveMaterial(live) }
            record = ArchiveMaterial(kind: "physicallyBased", physicallyBased: try physicallyBased(pbr))
        case .classic(let classic):
            record = ArchiveMaterial(kind: "classic", classic: try classicMaterial(classic))
        case .custom, .rawMetal:
            throw SceneArchive.Failure.unsupportedMaterial(
                "a custom Metal library does not expose its source; describe the surface with a SurfaceShaderGraph"
            )
        }
        materials.append(record)
        return materials.count - 1
    }

    private mutating func liveMaterial(_ live: PhysicallyBasedMaterial.LiveFactors) throws -> Int {
        let identity = ObjectIdentifier(live)
        if let index = liveMaterialIndices[identity] { return index }
        let base = try physicallyBased(live.base)
        let current = live.current
        let changes = ArchiveLiveFactor.fields.compactMap { field -> ArchiveLiveFactor? in
            guard let value = current.value(of: field), value != live.rest.value(of: field) else { return nil }
            return ArchiveLiveFactor(field: field, value: .capture([value], kind: value.kind, writer: &writer))
        }
        let index = materials.count
        materials.append(ArchiveMaterial(kind: "physicallyBased", physicallyBased: base, liveFactors: changes))
        liveMaterialIndices[identity] = index
        retained.append(live)
        return index
    }

    private mutating func binding(_ binding: PhysicallyBasedMaterial.TextureBinding?, _ slot: String) throws -> ArchiveBinding? {
        try binding.map { try archiveBinding($0, slot) }
    }

    private mutating func archiveBinding(
        _ binding: PhysicallyBasedMaterial.TextureBinding, _ slot: String
    ) throws -> ArchiveBinding {
        let encoding = binding.normalEncoding == .rgb ? nil : binding.normalEncoding
        let normalSpace = binding.normalSpace == .tangentSpace ? nil : binding.normalSpace
        hasPackedNormals = hasPackedNormals || encoding != nil
        hasObjectSpaceNormals = hasObjectSpaceNormals || normalSpace != nil
        return ArchiveBinding(
            texture: try texture(binding.texture),
            sampler: try binding.sampler.map { try sampler($0, slot, fallback: binding.sampling) },
            sampling: binding.sampling, coordinateSet: binding.textureCoordinateSet, transform: binding.transform,
            normalEncoding: encoding, normalSpace: normalSpace
        )
    }

    private mutating func classicMaterial(_ material: ClassicMaterial) throws -> ArchiveClassicMaterial {
        let model: ArchiveClassicMaterial.Model = switch material.model {
        case .basic: .basic
        case .lambert: .lambert
        case .phong(let specular, let shininess): .phong(specular: specular, shininess: shininess)
        case .toon(let gradient): .toon(gradient: try binding(gradient, "toon gradient"))
        case .matcap(let map):
            .matcap(try archiveBinding(map, "matcap"))
        case .normal: .normal
        case .depth: .depth
        case .distance(let reference, let near, let far): .distance(reference: reference, near: near, far: far)
        case .shadow: .shadow
        }
        var nodeTextures: [String: Int]?
        if let graph = material.surfaceGraph, !graph.textures.isEmpty {
            var indices: [String: Int] = [:]
            for (name, image) in zip(graph.textures.map(\.name), material.surfaceTextures) {
                indices[name] = try texture(image)
            }
            nodeTextures = indices
        }
        return ArchiveClassicMaterial(
            model: model, color: material.color, opacity: material.opacity, map: try binding(material.map, "color"),
            alphaMap: try binding(material.alphaMap, "alpha"), alphaTest: material.alphaTest,
            aoMap: try binding(material.aoMap, "ambient occlusion"), aoMapIntensity: material.aoMapIntensity,
            lightMap: try binding(material.lightMap, "light"), lightMapIntensity: material.lightMapIntensity,
            emissive: material.emissive, emissiveMap: try binding(material.emissiveMap, "emissive"),
            environmentMap: try material.environmentMap.map {
                .init(texture: try texture($0.texture), blend: .init($0.blend), reflectivity: $0.reflectivity,
                      intensity: $0.intensity, yaw: $0.yaw, refractionRatio: $0.mapping.refractionRatio)
            },
            ambient: material.ambient, receivesShadows: material.receivesShadows, flatShading: material.flatShading,
            depthPacking: material.depthPacking, distanceOutput: material.distanceOutput == .red ? .red : nil,
            bumpMap: try material.bumpMap.map { .init(texture: try archiveBinding($0.texture, "classic bump"), scale: $0.scale) },
            normalMap: try binding(material.normalMap, "classic normal"),
            normalScale: material.normalScale == SIMD2(repeating: 1) ? nil : material.normalScale,
            surfaceGraph: material.surfaceGraph.map { graph in
                if let index = graphs.firstIndex(of: graph) { return index }
                graphs.append(graph)
                return graphs.count - 1
            },
            surfaceUniforms: try material.surfaceUniforms.map { try uniformSet($0) },
            surfaceTextures: nodeTextures
        )
    }

    private mutating func physicallyBased(_ material: PhysicallyBasedMaterial) throws -> ArchivePBR {
        guard material.environment?.boxProjection == nil else {
            throw SceneArchive.Failure.unsupportedMaterial("box-projected environment bindings")
        }
        let inputs = material.sourceInputs
        let alphaMode: String
        var cutoff: Float?
        switch material.alphaMode {
        case .materialAlpha: alphaMode = "materialAlpha"
        case .opaque: alphaMode = "opaque"
        case .mask(let value): alphaMode = "mask"; cutoff = value
        case .blend: alphaMode = "blend"
        }
        var surfaceTextures: [String: Int]?
        if let graph = material.surfaceGraph, !graph.textures.isEmpty {
            var named: [String: Int] = [:]
            for (declaration, texture) in zip(graph.textures, material.surfaceTextures) {
                named[declaration.name] = try self.texture(texture)
            }
            surfaceTextures = named
        }
        return ArchivePBR(
            baseColorTexture: try material.baseColorTexture.map { try texture($0) },
            baseColorSampler: try inputs.baseColorSampler.map { try sampler($0, "base color", fallback: material.baseColorSampling) },
            baseColorSampling: material.baseColorSampling,
            baseColorTextureCoordinateSet: inputs.baseColorTextureCoordinateSet,
            baseColorTextureTransform: material.baseColorTextureTransform, baseColorFactor: material.baseColorFactor,
            metallic: inputs.metallic, roughness: inputs.roughness, directionToLight: inputs.directionToLight,
            lightIntensity: inputs.lightIntensity, ambientIntensity: inputs.ambientIntensity,
            receivesShadows: material.receivesShadows,
            metallicRoughnessTexture: try binding(inputs.metallicRoughnessTexture, "metallic-roughness"),
            normalTexture: try binding(inputs.normalTexture, "normal"), normalScale: inputs.normalScale,
            bumpMap: try material.bumpMap.map {
                .init(texture: try archiveBinding($0.texture, "bump"), scale: $0.scale)
            },
            lightMap: try material.lightMap.map {
                .init(texture: try texture($0.texture), coordinateSet: $0.textureCoordinateSet, transform: $0.transform)
            },
            lightMapIntensity: material.lightMapIntensity == 1 ? nil : material.lightMapIntensity,
            retroreflectivity: material.retroreflectivity == 0 ? nil : material.retroreflectivity,
            occlusionTexture: try binding(inputs.occlusionTexture, "occlusion"), occlusionStrength: inputs.occlusionStrength,
            emissiveTexture: try binding(inputs.emissiveTexture, "emissive"), emissiveFactor: inputs.emissiveFactor,
            emissiveStrength: material.emissiveStrength, alphaTexture: try binding(material.alphaTexture, "alpha"),
            alphaMode: alphaMode, alphaCutoff: cutoff,
            isDoubleSided: material.isDoubleSided,
            environment: try material.environment.map {
                .init(index: try environment($0.lighting), intensity: $0.intensity, yaw: $0.yaw)
            },
            clearcoat: try material.clearcoat.map {
                .init(
                    factor: $0.factor, roughness: $0.roughness, factorTexture: try binding($0.factorTexture, "clearcoat"),
                    roughnessTexture: try binding($0.roughnessTexture, "clearcoat roughness"),
                    normalTexture: try binding($0.normalTexture, "clearcoat normal"), normalScale: $0.normalScale
                )
            },
            transmission: try material.transmission.map {
                .init(
                    factor: $0.factor, indexOfRefraction: $0.indexOfRefraction, thickness: $0.thickness,
                    attenuationDistance: $0.attenuationDistance, attenuationColor: $0.attenuationColor,
                    factorTexture: try binding($0.factorTexture, "transmission"),
                    thicknessTexture: try binding($0.thicknessTexture, "thickness"), dispersion: $0.dispersion
                )
            },
            specular: try material.specular.map {
                .init(
                    factor: $0.factor, colorFactor: $0.colorFactor, factorTexture: try binding($0.factorTexture, "specular"),
                    colorTexture: try binding($0.colorTexture, "specular color")
                )
            },
            sheen: try material.sheen.map {
                .init(
                    colorFactor: $0.colorFactor, roughness: $0.roughness, colorTexture: try binding($0.colorTexture, "sheen color"),
                    roughnessTexture: try binding($0.roughnessTexture, "sheen roughness")
                )
            },
            anisotropy: try material.anisotropy.map {
                .init(strength: $0.strength, rotation: $0.rotation, texture: try binding($0.texture, "anisotropy"))
            },
            iridescence: try material.iridescence.map {
                .init(
                    factor: $0.factor, indexOfRefraction: $0.indexOfRefraction, minimumThickness: $0.minimumThickness,
                    maximumThickness: $0.maximumThickness, factorTexture: try binding($0.factorTexture, "iridescence"),
                    thicknessTexture: try binding($0.thicknessTexture, "iridescence thickness")
                )
            },
            diffuseTransmission: try material.diffuseTransmission.map {
                .init(
                    factor: $0.factor, colorFactor: $0.colorFactor,
                    factorTexture: try binding($0.factorTexture, "diffuse transmission factor"),
                    colorTexture: try binding($0.colorTexture, "diffuse transmission color")
                )
            },
            diffuseRoughness: try material.diffuseRoughness.map {
                .init(factor: $0.factor, texture: try binding($0.texture, "diffuse roughness"))
            },
            volumeScatter: try material.volumeScatter.map {
                .init(
                    multiscatterColor: $0.multiscatterColor, anisotropy: $0.anisotropy,
                    multiscatterColorTexture: try binding($0.multiscatterColorTexture, "volume scattering")
                )
            },
            subsurface: try material.subsurface.map {
                .init(
                    weight: $0.weight, color: $0.color, radius: $0.radius, scale: $0.scale,
                    anisotropy: $0.anisotropy, colorTexture: try binding($0.colorTexture, "subsurface color")
                )
            },
            surfaceGraph: material.surfaceGraph.map { graph in
                if let index = graphs.firstIndex(of: graph) { return index }
                graphs.append(graph)
                return graphs.count - 1
            },
            surfaceUniforms: material.surfaceUniforms.map { uniformSet($0) },
            surfaceTextures: surfaceTextures, unlit: material.shadingModel == .unlit,
            unlitUsesGraphEmission: material.unlitUsesGraphEmission ? true : nil
        )
    }

    private mutating func uniformSet(_ set: ShaderUniforms) -> Int {
        if let index = uniformIndices[ObjectIdentifier(set)] { return index }
        retained.append(set)
        uniforms.append(ArchiveUniforms(declarations: set.declarations, values: set.snapshot().values))
        uniformIndices[ObjectIdentifier(set)] = uniforms.count - 1
        return uniforms.count - 1
    }

    private mutating func sampler(
        _ sampler: any MTLSamplerState, _ slot: String, fallback: PhysicallyBasedMaterial.TextureSampling?
    ) throws -> Int {
        guard let sampling = TextureSampling.recorded(for: sampler) ?? fallback.map({ TextureSampling(descriptor: $0.descriptor) }) else {
            throw SceneArchive.Failure.unrecordedSampler("the \(slot) map")
        }
        if let index = samplers.firstIndex(of: sampling) { return index }
        samplers.append(sampling)
        return samplers.count - 1
    }

    private mutating func environment(_ lighting: EnvironmentLighting) throws -> Int {
        if let index = environmentIndices[ObjectIdentifier(lighting)] { return index }
        retained.append(lighting)
        environments.append(ArchiveEnvironment(
            sky: try texture(lighting.skyCube), diffuse: try texture(lighting.diffuseCube),
            specular: try texture(lighting.specularCube), brdf: try texture(lighting.brdfLUT)
        ))
        environmentIndices[ObjectIdentifier(lighting)] = environments.count - 1
        return environments.count - 1
    }

    private mutating func texture(_ texture: any MTLTexture) throws -> Int {
        if let index = textureIndices[ObjectIdentifier(texture)] { return index }
        retained.append(texture)
        var (record, bytes) = try ArchiveTextureCodec.read(texture, queue: queue(for: texture.device))
        record.view = writer.append(bytes)
        textures.append(record)
        textureIndices[ObjectIdentifier(texture)] = textures.count - 1
        return textures.count - 1
    }

    private mutating func queue(for device: any MTLDevice) throws -> any MTLCommandQueue {
        if let suppliedQueue, suppliedQueue.device === device { return suppliedQueue }
        if let existing = queues[ObjectIdentifier(device)] { return existing }
        guard let queue = device.makeCommandQueue() else { throw SceneArchive.Failure.resource("readback command queue") }
        queues[ObjectIdentifier(device)] = queue
        return queue
    }

    private mutating func clip(_ clip: AnimationClip, bindings: ArchiveAnimationTargets) throws -> ArchiveClip {
        let targets = clip.targets.map { target in
            ArchiveClip.Target(
                node: target.nodeID,
                keyframes: target.transform.map { track in
                    writer.append(track.keyframes.flatMap { key -> [Float] in
                        let t = key.transform
                        return [key.time, t.translation.x, t.translation.y, t.translation.z]
                            + [t.rotation.vector.x, t.rotation.vector.y, t.rotation.vector.z, t.rotation.vector.w]
                            + [t.scale.x, t.scale.y, t.scale.z]
                    })
                },
                channels: target.channels.isEmpty ? nil : target.channels.map { channel in
                    ArchiveClip.Channel(
                        sourceNodeIndex: channel.sourceNodeIndex, target: channel.target.rawValue,
                        interpolation: channel.interpolation.rawValue, times: writer.append(channel.times),
                        values: channel.target == .weights ? nil : writer.append(channel.values),
                        weights: channel.target == .weights ? writer.append(channel.weightValues) : nil,
                        weightCount: channel.weightCount
                    )
                }
            )
        }
        var reference: [String: ArchiveClip.Pose]?
        if !clip.reference.isEmpty {
            reference = Dictionary(uniqueKeysWithValues: clip.reference.map { id, pose in
                (id.uuidString, ArchiveClip.Pose(
                    translation: pose.translation, rotation: pose.rotation?.vector, scale: pose.scale,
                    morphWeights: pose.morphWeights
                ))
            })
        }
        let propertyTracks = try clip.tracks.map { track in
            try bindings.validate(track.property, kind: track.valueKind)
            let binding = try animationBinding(track.property)
            return try ArchivePropertyTrack(
                track: track, binding: binding, reference: clip.trackReference[track.property], writer: &writer
            )
        }
        return ArchiveClip(
            name: clip.name, duration: clip.duration, additive: clip.blendMode == .additive, targets: targets,
            reference: reference, propertyTracks: propertyTracks.isEmpty ? nil : propertyTracks
        )
    }

    private mutating func animationBinding(_ property: AnimationProperty) throws -> ArchivePropertyBinding {
        switch property {
        case .node(let id, let field): .node(id: id, field: field)
        case .camera(let field): .camera(field: field)
        case .environment(let field): .environment(field: field)
        case .uniform(let set, let name): .uniform(index: uniformSet(set), name: name)
        case .material(let live, let field): .material(index: try liveMaterial(live), field: field)
        case .rigLight(let id, let field): .rigLight(id: id, field: field)
        case .gltfCamera(let index, let field): .gltfCamera(index: index, field: field)
        case .custom(let name): .custom(name: name)
        }
    }
}

// MARK: Load

private struct ArchiveLoader {
    let document: ArchiveManifest
    let reader: ArchiveBinaryReader
    let queue: any MTLCommandQueue
    let colorPixelFormat: MTLPixelFormat?
    private var device: any MTLDevice { queue.device }
    private var textures: [Int: any MTLTexture] = [:]
    private var samplers: [Int: any MTLSamplerState] = [:]
    private var environments: [Int: EnvironmentLighting] = [:]
    private var uniforms: [Int: ShaderUniforms] = [:]
    private var liveMaterials: [Int: PhysicallyBasedMaterial] = [:]
    private var meshes: [Int: MeshRenderer] = [:]
    private var primitives: [Int: ScenePrimitive] = [:]
    private var instanceSets: [Int: InstanceSet] = [:]
    private var photometry: ArchivePhotometryLoader

    init(document: ArchiveManifest, reader: ArchiveBinaryReader, queue: any MTLCommandQueue, colorPixelFormat: MTLPixelFormat?) {
        self.document = document
        self.reader = reader
        self.queue = queue
        self.colorPixelFormat = colorPixelFormat
        photometry = ArchivePhotometryLoader(record: document.photometry, reader: reader)
    }

    mutating func contents() throws -> SceneArchive.Contents {
        let scene = document.scene
        let records = scene.nodes
        var depths = [Int](repeating: 1, count: records.count)
        for (index, record) in records.enumerated() {
            guard let parent = record.parent else { continue }
            guard (0..<index).contains(parent) else {
                throw SceneArchive.Failure.invalidReference("parent \(parent) of node \(index)")
            }
            let depth = depths[parent] + 1
            guard depth <= SceneArchive.maximumHierarchyDepth else { throw SceneArchive.Failure.hierarchyDepth }
            depths[index] = depth
        }
        // Create nodes and their resources in document order, then attach children from the
        // last node back to the first, so no step recurses.
        let nodes = try records.map { try node($0) }
        var children = Array(repeating: [SceneNode](), count: records.count)
        var roots: [SceneNode] = []
        for index in records.indices.reversed() {
            var node = nodes[index]
            node.children = children[index].reversed()
            children[index] = []
            if let parent = records[index].parent { children[parent].append(node) } else { roots.append(node) }
        }
        var graph = SceneGraph(roots: roots.reversed(), camera: scene.camera, cameraTransform: scene.cameraTransform)
        graph.cameraLayers = scene.cameraLayers
        graph.environment = try scene.environment.map { try environment($0) }
        graph.environmentIntensity = scene.environmentIntensity
        graph.environmentRotation = scene.environmentRotation
        graph.environmentEulerRotation = scene.environmentEulerRotation ?? .zero
        graph.lightProbes = try scene.decodedLightProbes()
        graph.projectionJitter = scene.projectionJitter ?? .zero
        _ = try graph.resolve()
        let rig = try photometry.rig(document.lightRig)
        let bindings = ArchiveAnimationTargets(nodes: records, meshes: document.meshes, lightRig: rig)
        let animations = try document.animations.map { try clip($0, bindings: bindings) }
        return SceneArchive.Contents(graph: graph, animations: animations, lightRig: rig)
    }

    private func element<T>(_ array: [T], _ index: Int, _ kind: String) throws -> T {
        guard array.indices.contains(index) else { throw SceneArchive.Failure.invalidReference("\(kind) \(index)") }
        return array[index]
    }

    private mutating func node(_ record: ArchiveNode) throws -> SceneNode {
        let levels = try record.levelsOfDetail.map { lod in
            try LevelOfDetail(
                levels: try lod.levels.map { .init(mesh: try mesh($0.mesh), distance: $0.distance, hysteresis: $0.hysteresis) },
                currentLevel: lod.currentLevel
            )
        }
        let light = try photometry.node(record.light, resources: record.photometry, id: record.id)
        var node = SceneNode(
            id: record.id, transform: record.transform, mesh: try record.mesh.map { try mesh($0) },
            materialParameters: record.materialParameters, castsShadow: record.castsShadow,
            jointPalette: try record.jointPalette.map {
                try JointPalette(matrices: ArchiveMatrices.unpack(try reader.array($0, of: SIMD4<Float>.self)))
            },
            light: light, ambientLight: record.ambientLight, instances: try record.instances.map { try instanceSet($0) },
            morphWeights: record.morphWeights, frustumCulled: record.frustumCulled, cullingBounds: record.cullingBounds,
            levelsOfDetail: levels, name: record.name, isVisible: record.isVisible, layers: record.layers,
            renderOrder: record.renderOrder, metadata: record.metadata ?? [:], pivot: record.pivot,
            orientationMode: record.orientationMode ?? .inherited
        )
        node.primitive = try record.primitive.map { try primitive($0) }
        return node
    }

    private mutating func instanceSet(_ index: Int) throws -> InstanceSet {
        if let existing = instanceSets[index] { return existing }
        let record = try element(document.instanceSets, index, "instance set")
        let transforms = ArchiveMatrices.unpack(
            try reader.array(record.transforms, of: SIMD4<Float>.self, count: try reader.product(record.count, 4, view: record.transforms))
        )
        let tints = try reader.array(record.tints, of: SIMD4<Float>.self, count: record.count)
        let set = try InstanceSet(device: device, transforms: transforms, tints: tints, capacity: record.capacity)
        instanceSets[index] = set
        return set
    }

    private mutating func primitive(_ index: Int) throws -> ScenePrimitive {
        if let existing = primitives[index] { return existing }
        let record = try element(document.primitives ?? [], index, "primitive")
        guard let archivedFormat = MTLPixelFormat(rawValue: record.colorPixelFormat), archivedFormat != .invalid else {
            throw SceneArchive.Failure.invalidReference("primitive color pixel format \(record.colorPixelFormat)")
        }
        let format = colorPixelFormat ?? archivedFormat
        let blending = record.blending.runtime
        let drawRange = try record.drawRange.map { try GeometryDrawRange(start: $0.start, count: $0.count) }
        let softParticleFade = try record.softParticleFade?.runtime()
        let result: ScenePrimitive
        switch record.content {
        case .points(let cloud):
            guard (1...PointCloud.maximumPointCount).contains(cloud.count) else { throw PrimitiveGeometryError.pointCount }
            let points = try PointCloud(
                positions: try float3(cloud.positions, count: cloud.count),
                colors: try cloud.colors.map { try reader.array($0, of: SIMD4<Float>.self, count: cloud.count) },
                sizes: try cloud.sizes.map { try reader.array($0, of: Float.self, count: cloud.count) }
            )
            let material = try PointsMaterial(
                size: cloud.extent.pointSize, shape: cloud.circular ? .circle : .square, color: record.color,
                map: cloud.map.map { try texture($0) }, alphaTest: cloud.alphaTest, blending: blending, depthTest: record.depthTest,
                softParticleFade: softParticleFade
            )
            let renderer = try PointsRenderer(
                device: device, colorPixelFormat: format, sampleCount: record.sampleCount, cloud: points, material: material
            )
            renderer.drawRange = drawRange ?? .all
            result = .points(renderer)
        case .lines(let line):
            guard softParticleFade == nil else { throw SceneArchive.Failure.decoding("line materials do not support soft particles") }
            guard (2...LineGeometry.maximumPointCount).contains(line.count) else { throw PrimitiveGeometryError.pointCount }
            let geometry = try LineGeometry(
                positions: try float3(line.positions, count: line.count),
                colors: try line.colors.map { try reader.array($0, of: SIMD4<Float>.self, count: line.count) },
                topology: line.topology.runtime
            )
            let material = try LineMaterial(
                width: line.extent?.lineWidth ?? .hairline, join: line.join.runtime, cap: line.cap.runtime,
                color: record.color, dash: line.dash.map { try .init(length: $0.length, gap: $0.gap, offset: $0.offset) },
                blending: blending, depthTest: record.depthTest
            )
            let renderer = try LineRenderer(
                device: device, colorPixelFormat: format, sampleCount: record.sampleCount, geometry: geometry, material: material
            )
            if let drawRange { try renderer.setDrawRange(drawRange) }
            result = .lines(renderer)
        case .sprites(let sprites):
            guard drawRange == nil else { throw SceneArchive.Failure.invalidReference("sprite draw range") }
            let material = try SpriteMaterial(
                map: sprites.map.map { try texture($0) }, color: record.color, sizeAttenuation: sprites.sizeAttenuation,
                alphaTest: sprites.alphaTest, blending: blending, depthTest: record.depthTest, softParticleFade: softParticleFade
            )
            result = .sprites(try SpriteRenderer(
                device: device, colorPixelFormat: format, sampleCount: record.sampleCount,
                sprites: sprites.unpack(using: reader), material: material
            ))
        }
        primitives[index] = result
        return result
    }

    private mutating func mesh(_ index: Int) throws -> MeshRenderer {
        if let existing = meshes[index] { return existing }
        let record = try element(document.meshes, index, "mesh")
        let geometry = try self.geometry(record.geometry)
        let count = record.geometry.vertexCount
        let skin = try record.skin.map { skin in
            let joints = try reader.array(skin.joints, of: SIMD4<UInt16>.self, count: count)
            let weights = try reader.array(skin.weights, of: SIMD4<Float>.self, count: count)
            return try MeshSkin(
                jointCount: skin.jointCount,
                influences: zip(joints, weights).map { VertexSkinInfluence(joints: $0, weights: $1) }
            )
        }
        let morph = try record.morphTargets.map { morph in
            try MeshMorphTargets(vertexCount: morph.vertexCount, targets: try morph.targets.map { target in
                .init(
                    positionOffsets: try target.positions.map { try float3($0, count: morph.vertexCount) },
                    normalOffsets: try target.normals.map { try float3($0, count: morph.vertexCount) },
                    tangentOffsets: try target.tangents.map { try float3($0, count: morph.vertexCount) }
                )
            })
        }
        guard let archivedFormat = MTLPixelFormat(rawValue: record.colorPixelFormat) else {
            throw SceneArchive.Failure.invalidReference("color pixel format \(record.colorPixelFormat)")
        }
        let renderer = try MeshRenderer(
            device: device, colorPixelFormat: colorPixelFormat ?? archivedFormat, geometry: geometry,
            material: try material(record.material), depthMode: record.reversedDepth ? .reversed : .forward,
            skin: skin, morphTargets: morph, supportsInstancing: record.supportsInstancing,
            renderState: try record.renderState?.makeState(), sampleCount: record.sampleCount ?? 1,
            usesStencilAttachment: record.usesStencilAttachment ?? false
        )
        if let range = record.drawRange { renderer.drawRange = try MeshDrawRange(start: range.start, count: range.count) }
        meshes[index] = renderer
        return renderer
    }

    private func float3(_ view: Int, count: Int) throws -> [SIMD3<Float>] {
        let floats = try reader.array(view, of: Float.self, count: try reader.product(count, 3, view: view))
        return (0..<count).map { SIMD3(floats[3 * $0], floats[3 * $0 + 1], floats[3 * $0 + 2]) }
    }

    private func geometry(_ record: ArchiveGeometry) throws -> MeshGeometry {
        let count = record.vertexCount
        guard count > 0 else { throw SceneArchive.Failure.invalidView(record.positions) }
        let positions = try float3(record.positions, count: count)
        let normals = try float3(record.normals, count: count)
        let uv0 = try reader.array(record.textureCoordinates, of: SIMD2<Float>.self, count: count)
        func optional<T>(_ data: Int?, _ mask: Int?, _ type: T.Type) throws -> [T?] {
            guard let data else { return Array(repeating: nil, count: count) }
            let values = try reader.array(data, of: type, count: count)
            guard let mask else { return values }
            let present = try reader.array(mask, of: UInt8.self, count: count)
            return zip(values, present).map { $1 == 0 ? nil : $0 }
        }
        let uv1 = try optional(record.secondaryTextureCoordinates, record.secondaryPresence, SIMD2<Float>.self)
        let tangents = try optional(record.tangents, record.tangentPresence, SIMD4<Float>.self)
        let colors = try optional(record.colors, record.colorPresence, SIMD4<Float>.self)
        let vertices = (0..<count).map {
            MeshVertex(
                position: positions[$0], normal: normals[$0], textureCoordinate: uv0[$0],
                secondaryTextureCoordinate: uv1[$0], tangent: tangents[$0], color: colors[$0]
            )
        }
        return try MeshGeometry(
            vertices: vertices, indices: try reader.array(record.indices, of: UInt32.self, count: record.indexCount)
        ).withGroups((record.groups ?? []).map {
            MeshGeometryGroup(start: $0.start, count: $0.count, materialIndex: $0.materialIndex)
        })
    }

    private mutating func material(_ index: Int) throws -> MeshMaterial {
        if let existing = liveMaterials[index] { return .physicallyBased(existing) }
        let record = try element(document.materials, index, "material")
        guard record.liveFactors == nil || record.kind == "physicallyBased" else {
            throw SceneArchive.Failure.invalidReference("live factors of material \(index)")
        }
        switch record.kind {
        case "unlit": return .unlit
        case "lambert": return .lambert
        case "shadowedLambert": return .shadowedLambert
        case "textured":
            guard let texture = record.texture else { throw SceneArchive.Failure.invalidReference("texture of material \(index)") }
            return .textured(try self.texture(texture))
        case "classic":
            guard let classic = record.classic else {
                throw SceneArchive.Failure.invalidReference("classic settings of material \(index)")
            }
            return .classic(try classicMaterial(classic))
        case "physicallyBased":
            guard let pbr = record.physicallyBased else {
                throw SceneArchive.Failure.invalidReference("PBR settings of material \(index)")
            }
            let base = try physicallyBased(pbr)
            guard let changes = record.liveFactors else { return .physicallyBased(base) }
            let material = base.withLiveFactors()
            guard let live = material.liveFactors else { throw SceneArchive.Failure.invalidReference("live material \(index)") }
            var factors = live.rest
            var fields = Set<AnimationProperty.Material>()
            for change in changes {
                guard fields.insert(change.field).inserted else {
                    throw SceneArchive.Failure.invalidAnimation("duplicate live material factor \(change.field)")
                }
                let value = try change.value.restore(count: 1, reader: reader)[0]
                do {
                    try factors.set(change.field, to: value)
                } catch {
                    throw SceneArchive.Failure.invalidAnimation(error.localizedDescription)
                }
            }
            do {
                try live.set(factors)
            } catch {
                throw SceneArchive.Failure.invalidAnimation(error.localizedDescription)
            }
            liveMaterials[index] = material
            return .physicallyBased(material)
        default:
            throw SceneArchive.Failure.invalidReference("material kind \(record.kind)")
        }
    }

    private mutating func binding(_ record: ArchiveBinding?) throws -> PhysicallyBasedMaterial.TextureBinding? {
        try record.map { try textureBinding($0) }
    }

    private mutating func textureBinding(_ record: ArchiveBinding) throws -> PhysicallyBasedMaterial.TextureBinding {
        PhysicallyBasedMaterial.TextureBinding(
            texture: try texture(record.texture), sampler: try record.sampler.map { try sampler($0) },
            textureCoordinateSet: record.coordinateSet, transform: record.transform, sampling: record.sampling,
            normalEncoding: record.normalEncoding ?? .rgb, normalSpace: record.normalSpace ?? .tangentSpace
        )
    }

    private mutating func classicMaterial(_ record: ArchiveClassicMaterial) throws -> ClassicMaterial {
        let model: ClassicMaterial.Model = switch record.model {
        case .basic: .basic
        case .lambert: .lambert
        case .phong(let specular, let shininess): .phong(specular: specular, shininess: shininess)
        case .toon(let gradient): .toon(gradient: try binding(gradient))
        case .matcap(let map): .matcap(try textureBinding(map))
        case .normal: .normal
        case .depth: .depth
        case .distance(let reference, let near, let far): .distance(reference: reference, near: near, far: far)
        case .shadow: .shadow
        }
        return try ClassicMaterial(
            model: model, color: record.color, opacity: record.opacity, map: try binding(record.map),
            alphaMap: try binding(record.alphaMap), alphaTest: record.alphaTest, aoMap: try binding(record.aoMap),
            aoMapIntensity: record.aoMapIntensity, lightMap: try binding(record.lightMap),
            lightMapIntensity: record.lightMapIntensity, emissive: record.emissive, emissiveMap: try binding(record.emissiveMap),
            environmentMap: try record.environmentMap.map {
                try .init(texture: texture($0.texture), blend: $0.blend.runtime, reflectivity: $0.reflectivity,
                          intensity: $0.intensity, yaw: $0.yaw,
                          mapping: $0.refractionRatio.map { .refraction(ratio: $0) } ?? .reflection)
            },
            ambient: record.ambient, receivesShadows: record.receivesShadows, flatShading: record.flatShading,
            depthPacking: record.depthPacking, distanceOutput: record.distanceOutput ?? .grayscale,
            bumpMap: try record.bumpMap.map { try .init(texture: textureBinding($0.texture), scale: $0.scale) },
            normalMap: try binding(record.normalMap), normalScale: record.normalScale ?? SIMD2(repeating: 1),
            surfaceGraph: try record.surfaceGraph.map { try element(document.surfaceGraphs, $0, "surface graph") },
            surfaceUniforms: try record.surfaceUniforms.map { try uniformSet($0) },
            surfaceTextures: try record.surfaceTextures?.mapValues { try texture($0) } ?? [:]
        )
    }

    private mutating func physicallyBased(_ record: ArchivePBR) throws -> PhysicallyBasedMaterial {
        let alphaMode: PhysicallyBasedMaterial.AlphaMode
        switch record.alphaMode {
        case "materialAlpha": alphaMode = .materialAlpha
        case "opaque": alphaMode = .opaque
        case "mask": alphaMode = .mask(cutoff: record.alphaCutoff ?? 0.5)
        case "blend": alphaMode = .blend
        default: throw SceneArchive.Failure.invalidReference("alpha mode \(record.alphaMode)")
        }
        var surfaceTextures: [String: any MTLTexture] = [:]
        for (name, index) in record.surfaceTextures ?? [:] { surfaceTextures[name] = try texture(index) }
        return try PhysicallyBasedMaterial(
            baseColorTexture: try record.baseColorTexture.map { try texture($0) },
            metallic: record.metallic, roughness: record.roughness, directionToLight: record.directionToLight,
            lightIntensity: record.lightIntensity, ambientIntensity: record.ambientIntensity,
            receivesShadows: record.receivesShadows, baseColorFactor: record.baseColorFactor,
            baseColorSampler: try record.baseColorSampler.map { try sampler($0) },
            metallicRoughnessTexture: try binding(record.metallicRoughnessTexture),
            normalTexture: try binding(record.normalTexture), normalScale: record.normalScale,
            occlusionTexture: try binding(record.occlusionTexture), occlusionStrength: record.occlusionStrength,
            emissiveTexture: try binding(record.emissiveTexture), emissiveFactor: record.emissiveFactor,
            alphaMode: alphaMode, isDoubleSided: record.isDoubleSided,
            environment: try record.environment.map {
                try PhysicallyBasedMaterial.Environment(lighting: try environment($0.index), intensity: $0.intensity, yaw: $0.yaw)
            },
            baseColorTextureCoordinateSet: record.baseColorTextureCoordinateSet,
            clearcoat: try record.clearcoat.map {
                try .init(
                    factor: $0.factor, roughness: $0.roughness, factorTexture: try binding($0.factorTexture),
                    roughnessTexture: try binding($0.roughnessTexture), normalTexture: try binding($0.normalTexture),
                    normalScale: $0.normalScale
                )
            },
            transmission: try record.transmission.map {
                try .init(
                    factor: $0.factor, indexOfRefraction: $0.indexOfRefraction, thickness: $0.thickness,
                    attenuationDistance: $0.attenuationDistance, attenuationColor: $0.attenuationColor,
                    factorTexture: try binding($0.factorTexture), thicknessTexture: try binding($0.thicknessTexture),
                    dispersion: $0.dispersion
                )
            },
            surfaceGraph: try record.surfaceGraph.map { try element(document.surfaceGraphs, $0, "surface graph") },
            surfaceUniforms: try record.surfaceUniforms.map { try uniformSet($0) },
            surfaceTextures: surfaceTextures, baseColorTextureTransform: record.baseColorTextureTransform,
            emissiveStrength: record.emissiveStrength,
            specular: try record.specular.map {
                try .init(
                    factor: $0.factor, colorFactor: $0.colorFactor, factorTexture: try binding($0.factorTexture),
                    colorTexture: try binding($0.colorTexture)
                )
            },
            sheen: try record.sheen.map {
                try .init(
                    colorFactor: $0.colorFactor, roughness: $0.roughness, colorTexture: try binding($0.colorTexture),
                    roughnessTexture: try binding($0.roughnessTexture)
                )
            },
            anisotropy: try record.anisotropy.map {
                try .init(strength: $0.strength, rotation: $0.rotation, texture: try binding($0.texture))
            },
            iridescence: try record.iridescence.map {
                try .init(
                    factor: $0.factor, indexOfRefraction: $0.indexOfRefraction, minimumThickness: $0.minimumThickness,
                    maximumThickness: $0.maximumThickness, factorTexture: try binding($0.factorTexture),
                    thicknessTexture: try binding($0.thicknessTexture)
                )
            },
            shadingModel: record.unlit ? .unlit : .physicallyBased,
            baseColorSampling: record.baseColorSampling, alphaTexture: try binding(record.alphaTexture),
            diffuseTransmission: try record.diffuseTransmission.map {
                try .init(
                    factor: $0.factor, colorFactor: $0.colorFactor,
                    factorTexture: try binding($0.factorTexture), colorTexture: try binding($0.colorTexture)
                )
            },
            diffuseRoughness: try record.diffuseRoughness.map {
                try .init(factor: $0.factor, texture: try binding($0.texture))
            },
            volumeScatter: try record.volumeScatter.map {
                try .init(
                    multiscatterColor: $0.multiscatterColor, anisotropy: $0.anisotropy,
                    multiscatterColorTexture: try binding($0.multiscatterColorTexture)
                )
            },
            subsurface: try record.subsurface.map {
                try .init(
                    weight: $0.weight, color: $0.color, radius: $0.radius, scale: $0.scale,
                    anisotropy: $0.anisotropy, colorTexture: try binding($0.colorTexture)
                )
            },
            bumpMap: try record.bumpMap.map {
                try .init(texture: textureBinding($0.texture), scale: $0.scale)
            },
            lightMap: try record.lightMap.map {
                try .init(texture: texture($0.texture), textureCoordinateSet: $0.coordinateSet, transform: $0.transform)
            },
            lightMapIntensity: record.lightMapIntensity ?? 1,
            retroreflectivity: record.retroreflectivity ?? 0,
            unlitUsesGraphEmission: record.unlitUsesGraphEmission ?? false
        )
    }

    private mutating func uniformSet(_ index: Int) throws -> ShaderUniforms {
        if let existing = uniforms[index] { return existing }
        let record = try element(document.uniforms, index, "uniform set")
        let slotCount = record.declarations.reduce(0) { $0 + $1.type.uniformSlotCount }
        guard record.values.count == slotCount else {
            throw SceneArchive.Failure.invalidReference("values of uniform set \(index)")
        }
        let set = try ShaderUniforms(record.declarations)
        var cursor = 0
        var integerCursor = 0
        for declaration in record.declarations {
            let count = declaration.type.uniformSlotCount
            let slots = Array(record.values[cursor..<(cursor + count)])
            cursor += count
            switch declaration.type {
            case .scalar: try set.set(declaration.name, to: slots[0].x)
            case .vector2: try set.set(declaration.name, to: SIMD2(slots[0].x, slots[0].y))
            case .vector3: try set.set(declaration.name, to: SIMD3(slots[0].x, slots[0].y, slots[0].z))
            case .vector4: try set.set(declaration.name, to: slots[0])
            case .matrix3:
                try set.set(declaration.name, to: simd_float3x3(
                    SIMD3(slots[0].x, slots[0].y, slots[0].z),
                    SIMD3(slots[1].x, slots[1].y, slots[1].z),
                    SIMD3(slots[2].x, slots[2].y, slots[2].z)
                ))
            case .matrix4:
                try set.set(declaration.name, to: simd_float4x4(columns: (slots[0], slots[1], slots[2], slots[3])))
            case .int:
                let value: Int32
                if let integerValues = record.integerValues {
                    guard integerValues.indices.contains(integerCursor) else {
                        throw SceneArchive.Failure.invalidReference("integer values of uniform set \(index)")
                    }
                    value = integerValues[integerCursor]
                    integerCursor += 1
                } else {
                    value = Int32(bitPattern: slots[0].x.bitPattern)
                }
                try set.setInteger(declaration.name, value)
            case .bool: try set.setBoolean(declaration.name, slots[0].x != 0)
            case .structure:
                throw SceneArchive.Failure.invalidReference("struct uniform \(declaration.name)")
            }
        }
        if let integerValues = record.integerValues, integerCursor != integerValues.count {
            throw SceneArchive.Failure.invalidReference("integer values of uniform set \(index)")
        }
        uniforms[index] = set
        return set
    }

    private mutating func sampler(_ index: Int) throws -> any MTLSamplerState {
        if let existing = samplers[index] { return existing }
        let sampler = try element(document.samplers, index, "sampler").makeSamplerState(device: device)
        samplers[index] = sampler
        return sampler
    }

    private mutating func environment(_ index: Int) throws -> EnvironmentLighting {
        if let existing = environments[index] { return existing }
        let record = try element(document.environments, index, "environment")
        let sky = try texture(record.sky)
        let diffuse = try texture(record.diffuse)
        let specular = try texture(record.specular)
        let brdf = try texture(record.brdf)
        guard sky.textureType == .typeCube, diffuse.textureType == .typeCube, specular.textureType == .typeCube,
              brdf.textureType == .type2D, [sky, diffuse, specular, brdf].allSatisfy({
                  $0.usage.isEmpty || $0.usage.contains(.shaderRead)
              }) else {
            throw SceneArchive.Failure.invalidReference("environment texture types")
        }
        let lighting = EnvironmentLighting(skyCube: sky, diffuseCube: diffuse, specularCube: specular, brdfLUT: brdf)
        environments[index] = lighting
        return lighting
    }

    private mutating func texture(_ index: Int) throws -> any MTLTexture {
        if let existing = textures[index] { return existing }
        let record = try element(document.textures, index, "texture")
        let texture = try ArchiveTextureCodec.make(record, bytes: try reader.bytes(record.view), queue: queue)
        textures[index] = texture
        return texture
    }

    private mutating func clip(_ record: ArchiveClip, bindings: ArchiveAnimationTargets) throws -> AnimationClip {
        guard record.duration.isFinite, record.duration >= 0 else {
            throw SceneArchive.Failure.invalidAnimation("clip duration \(record.duration)")
        }
        let targets = try record.targets.map { target in
            let transform = try target.keyframes.map { view in
                let floats = try reader.array(view, of: Float.self)
                guard floats.count % 11 == 0 else { throw SceneArchive.Failure.invalidView(view) }
                let keys = stride(from: 0, to: floats.count, by: 11).map { start in
                    TransformKeyframe(time: floats[start], transform: SceneTransform(
                        translation: SIMD3(floats[start + 1], floats[start + 2], floats[start + 3]),
                        rotation: simd_quatf(vector: SIMD4(floats[start + 4], floats[start + 5], floats[start + 6], floats[start + 7])),
                        scale: SIMD3(floats[start + 8], floats[start + 9], floats[start + 10])
                    ))
                }
                return try SceneAnimationTrack(nodeID: target.node, keyframes: keys)
            }
            var paths = Set<GLTFAnimationClip.Channel.Target>()
            let channels = try (target.channels ?? []).map { channel in
                guard let path = GLTFAnimationClip.Channel.Target(rawValue: channel.target),
                      let interpolation = GLTFAnimationClip.Channel.Interpolation(rawValue: channel.interpolation) else {
                    throw SceneArchive.Failure.invalidReference("animation channel \(channel.target) \(channel.interpolation)")
                }
                guard paths.insert(path).inserted else {
                    throw SceneArchive.Failure.invalidAnimation("node \(target.node) has two \(path.rawValue) channels")
                }
                // The checks GLTFMeshLoader applies to imported samplers, so sampling cannot index out of range.
                let times = try reader.array(channel.times, of: Float.self)
                guard let first = times.first, first.isFinite, first >= 0, times.last?.isFinite == true,
                      zip(times, times.dropFirst()).allSatisfy({ $0.0 < $0.1 }) else {
                    throw SceneArchive.Failure.invalidAnimation("key times must be finite, nonnegative and strictly increasing")
                }
                let cubic = interpolation == .cubicSpline
                if cubic, times.count < 2 { throw SceneArchive.Failure.invalidAnimation("a cubic spline needs at least two keys") }
                let elements = try reader.product(times.count, cubic ? 3 : 1, view: channel.times)
                if path == .weights {
                    guard let weights = channel.weights else { throw SceneArchive.Failure.invalidReference("morph weight keys") }
                    guard (1...MeshRenderer.maximumMorphTargetCount).contains(channel.weightCount) else {
                        throw SceneArchive.Failure.invalidAnimation("morph weight count \(channel.weightCount)")
                    }
                    let values = try reader.array(
                        weights, of: Float.self, count: try reader.product(elements, channel.weightCount, view: weights)
                    )
                    guard values.allSatisfy(\.isFinite) else { throw SceneArchive.Failure.invalidAnimation("nonfinite morph weights") }
                    return GLTFAnimationClip.Channel(
                        sourceNodeIndex: channel.sourceNodeIndex, interpolation: interpolation, times: times,
                        weights: values, weightCount: channel.weightCount
                    )
                }
                guard let view = channel.values else { throw SceneArchive.Failure.invalidReference("animation channel values") }
                let values = try reader.array(view, of: SIMD4<Float>.self, count: elements)
                guard values.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite }) else {
                    throw SceneArchive.Failure.invalidAnimation("nonfinite \(path.rawValue) keys")
                }
                if path == .rotation {
                    // Tangents are derivatives; only the spline values must be unit quaternions.
                    for key in stride(from: cubic ? 1 : 0, to: values.count, by: cubic ? 3 : 1)
                    where abs(simd_length(values[key]) - 1) > 1e-3 {
                        throw SceneArchive.Failure.invalidAnimation("rotation key \(key) is not a unit quaternion")
                    }
                }
                return GLTFAnimationClip.Channel(
                    sourceNodeIndex: channel.sourceNodeIndex, target: path, interpolation: interpolation, times: times,
                    values: values
                )
            }
            return AnimationNodeTracks(nodeID: target.node, transform: transform, channels: channels)
        }
        var reference: [UUID: AnimationNodePose] = [:]
        for (key, pose) in record.reference ?? [:] {
            guard let id = UUID(uuidString: key) else { throw SceneArchive.Failure.invalidReference("reference node \(key)") }
            reference[id] = AnimationNodePose(
                translation: pose.translation, rotation: pose.rotation.map { simd_quatf(vector: $0) }, scale: pose.scale,
                morphWeights: pose.morphWeights
            )
        }
        guard Set(targets.map(\.nodeID)).count == targets.count else {
            throw SceneArchive.Failure.invalidReference("clip with a node animated twice")
        }
        var tracks: [AnimationTrack] = []
        var trackReference: [AnimationProperty: AnimationValue] = [:]
        var properties = Set<AnimationProperty>()
        for archived in record.propertyTracks ?? [] {
            let property = try animationProperty(archived.binding)
            guard properties.insert(property).inserted else {
                throw SceneArchive.Failure.invalidAnimation("duplicate property track for \(property)")
            }
            let (track, reference) = try archived.restore(property: property, additive: record.additive, reader: reader)
            try bindings.validate(property, kind: track.valueKind)
            tracks.append(track)
            trackReference[property] = reference
        }
        return AnimationClip(
            name: record.name, duration: record.duration, blendMode: record.additive ? .additive : .normal,
            targets: targets, reference: reference, tracks: tracks, trackReference: trackReference
        )
    }

    private mutating func animationProperty(_ binding: ArchivePropertyBinding) throws -> AnimationProperty {
        switch binding {
        case .node(let id, let field): return .node(id, field)
        case .camera(let field): return .camera(field)
        case .environment(let field): return .environment(field)
        case .uniform(let index, let name):
            let property = AnimationProperty.uniform(try uniformSet(index), name: name)
            guard property.valueKind != nil else { throw SceneArchive.Failure.invalidReference(property.description) }
            return property
        case .material(let index, let field):
            guard case .physicallyBased(let material) = try material(index), let live = material.liveFactors else {
                throw SceneArchive.Failure.invalidReference("live material \(index)")
            }
            return .material(live, field)
        case .rigLight(let id, let field): return .rigLight(id: id, field)
        case .gltfCamera(let index, let field): return .gltfCamera(index, field)
        case .custom(let name): return .custom(name)
        }
    }
}

extension PhysicallyBasedMaterial {
    /// Initializer arguments that the material folds into packed shader values, kept
    /// verbatim so `SceneArchive` can rebuild an identical material.
    package struct SourceInputs {
        let metallic: Float
        let roughness: Float
        let directionToLight: SIMD3<Float>
        let lightIntensity: Float
        let ambientIntensity: Float
        let baseColorSampler: (any MTLSamplerState)?
        let metallicRoughnessTexture: TextureBinding?
        let normalTexture: TextureBinding?
        let normalScale: Float
        let occlusionTexture: TextureBinding?
        let occlusionStrength: Float
        let emissiveTexture: TextureBinding?
        let emissiveFactor: SIMD3<Float>
        let baseColorTextureCoordinateSet: MeshTextureCoordinateSet
    }
}

public extension AssetLoader {
    /// Reads a scene archive directory once. Its cost is the manifest and sidecar size.
    func sceneArchive(contentsOf url: URL) async throws -> SceneArchive {
        try await load(url, variant: "scene-archive", as: SceneArchive.self, cost: { $0.manifest.count + $0.binary.count }) { context in
            try Task.checkCancellation()
            return try SceneArchive(contentsOf: context.url)
        }
    }
}
