import Darwin
import Foundation
import simd

/// Loads static triangle primitives and node hierarchies from glTF 2.0 assets.
/// Node animation, skinning, punctual lights, cameras, morph targets, and GPU
/// instancing are opt-in. Quantized attributes (`KHR_mesh_quantization`) and
/// meshopt-compressed buffer views and Draco-compressed triangle meshes decode on load.
/// The document and total loaded or decoded buffers each have a 64 MiB limit.
public enum GLTFMeshLoader {
    private static let maximumBytes = 64 * 1_024 * 1_024
    private static let maximumElements = 1_000_000
    /// Bounds the offsets materialized from sparse morph accessors without base data.
    private static let maximumMorphOffsets = 8_000_000

    /// Supports local files and their adjacent regular buffer files, without following buffer symlinks.
    public static func load(contentsOf url: URL, meshIndex: Int = 0, primitiveIndex: Int = 0) throws -> MeshGeometry {
        try load(readLocalAsset(url), directory: url.deletingLastPathComponent(), meshIndex: meshIndex, primitiveIndex: primitiveIndex)
    }

    /// Supports JSON with base64 buffers and GLB binary chunks.
    public static func load(data: Data, meshIndex: Int = 0, primitiveIndex: Int = 0) throws -> MeshGeometry {
        try load(data, directory: nil, meshIndex: meshIndex, primitiveIndex: primitiveIndex)
    }

    /// Loads the selected scene using the same local resource rules as primitive loading.
    /// With no index, selects the asset's default scene or its first scene.
    /// Material metadata, node TRS and morph weight animations, skin data, punctual
    /// lights, cameras, morph targets, and `EXT_mesh_gpu_instancing` transforms are
    /// opt-in. Morph targets import POSITION, NORMAL, and TANGENT offsets, including
    /// sparse accessors, for primitives with authored normals.
    public static func loadScene(
        contentsOf url: URL, sceneIndex: Int? = nil, includeMaterials: Bool = false,
        includeAnimations: Bool = false, includeSkins: Bool = false, includeLights: Bool = false,
        includeCameras: Bool = false, includeMorphTargets: Bool = false, includeGPUInstancing: Bool = false
    ) throws -> GLTFScene {
        try loadScene(
            readLocalAsset(url), directory: url.deletingLastPathComponent(),
            sceneIndex: sceneIndex, includeMaterials: includeMaterials,
            includeAnimations: includeAnimations, includeSkins: includeSkins, includeLights: includeLights,
            includeCameras: includeCameras, includeMorphTargets: includeMorphTargets,
            includeGPUInstancing: includeGPUInstancing
        )
    }

    /// Loads a scene from GLB or JSON with embedded resources.
    public static func loadScene(
        data: Data, sceneIndex: Int? = nil, includeMaterials: Bool = false,
        includeAnimations: Bool = false, includeSkins: Bool = false, includeLights: Bool = false,
        includeCameras: Bool = false, includeMorphTargets: Bool = false, includeGPUInstancing: Bool = false
    ) throws -> GLTFScene {
        try loadScene(
            data, directory: nil, sceneIndex: sceneIndex, includeMaterials: includeMaterials,
            includeAnimations: includeAnimations, includeSkins: includeSkins, includeLights: includeLights,
            includeCameras: includeCameras, includeMorphTargets: includeMorphTargets,
            includeGPUInstancing: includeGPUInstancing
        )
    }

    private static func readLocalAsset(_ url: URL) throws -> Data {
        guard url.isFileURL, !url.path.utf8.contains(0),
              url.host == nil || url.host == "" || url.host == "localhost" else {
            throw Failure.localFileRequired
        }
        let descriptor = url.path.withCString { open($0, O_RDONLY | O_CLOEXEC | O_NONBLOCK) }
        return try readFile(descriptor, name: url.lastPathComponent)
    }

    private static func load(_ data: Data, directory: URL?, meshIndex: Int, primitiveIndex: Int) throws -> MeshGeometry {
        let (json, binary) = try unpack(data)
        let document = try decode(Document.self, from: json)
        try validate(document)
        guard let meshes = document.meshes, meshes.indices.contains(meshIndex) else { throw Failure.missingMesh(meshIndex) }
        let primitives = meshes[meshIndex].primitives
        guard primitives.indices.contains(primitiveIndex) else { throw Failure.missingPrimitive(primitiveIndex) }
        var reader = Reader(document: document, binary: binary, directory: directory)
        let primitive = primitives[primitiveIndex]
        let draco = try reader.decodeDraco(primitive)
        return try loadPrimitive(primitive, reader: &reader, allowsMorphTargets: false, draco: draco).geometry
    }

    private static func loadScene(
        _ data: Data, directory: URL?, sceneIndex: Int?, includeMaterials: Bool,
        includeAnimations: Bool, includeSkins: Bool, includeLights: Bool, includeCameras: Bool,
        includeMorphTargets: Bool, includeGPUInstancing: Bool
    ) throws -> GLTFScene {
        let (json, binary) = try unpack(data)
        let decoded = try decode(SceneDocument.self, from: json)
        try validate(
            decoded.document, includeLights: includeLights, includeMaterials: includeMaterials,
            includeGPUInstancing: includeGPUInstancing, includeAnimations: includeAnimations
        )
        let lightDocument = includeLights ? try decode(GLTFLightDocument.self, from: json) : nil
        let lightIndices = try lightDocument?.nodeReferences(
            required: decoded.document.extensionsRequired?.contains(GLTFLightDocument.extensionName) == true,
            maximumElements: maximumElements
        ) ?? [:]
        let cameraDocument = includeCameras ? try decode(GLTFCameraDocument.self, from: json) : nil
        let scene = try decoded.hierarchy.resolve(
            sceneIndex: sceneIndex, includeSkins: includeSkins, includeMorphTargets: includeMorphTargets,
            includeGPUInstancing: includeGPUInstancing,
            lightIndices: lightIndices, cameraIndices: cameraDocument?.nodeCameraIndices ?? [:]
        )
        let lights = try lightDocument?.resolve(nodes: scene.nodes) ?? []
        let cameras = try cameraDocument?.resolve(nodes: scene.nodes) ?? []
        let sourceMeshes = decoded.document.meshes ?? []
        var reader = Reader(document: decoded.document, binary: binary, directory: directory)
        let skins: [GLTFScene.Skin]
        if scene.nodes.contains(where: { $0.skinIndex != nil }) {
            skins = try decode(GLTFSkinDocument.self, from: json).resolve(
                nodes: scene.nodes, roots: scene.roots, maximumElements: maximumElements,
                readMatrices: { try reader.bindMatrices($0, jointCount: $1) }
            )
        } else {
            skins = []
        }
        let jointsBySkin = Dictionary(uniqueKeysWithValues: skins.map { ($0.sourceIndex, $0.jointNodeIndices.count) })
        var jointLimitByMesh: [Int: Int] = [:]
        for node in scene.nodes {
            if let mesh = node.meshIndex, let skin = node.skinIndex, let joints = jointsBySkin[skin] {
                jointLimitByMesh[mesh] = min(jointLimitByMesh[mesh] ?? joints, joints)
            }
        }

        var meshes: [GLTFScene.Mesh] = []
        var materialUsage: [Int: GLTFMaterialDocument.Usage] = [:]
        let variants = includeMaterials
            ? try decode(GLTFMaterialVariantsDocument.self, from: json).resolve(meshIndices: Set(scene.nodes.compactMap(\.meshIndex)))
            : nil
        var vertexCount = 0
        var indexCount = 0
        var morphBudget = maximumMorphOffsets

        for meshIndex in Set(scene.nodes.compactMap(\.meshIndex)).sorted() {
            guard sourceMeshes.indices.contains(meshIndex) else { throw Failure.missingMesh(meshIndex) }
            let mesh = sourceMeshes[meshIndex]
            guard !mesh.primitives.isEmpty else { throw Failure.missingPrimitive(0) }
            var primitives: [MeshGeometry] = []
            var skinInfluences: [[VertexSkinInfluence]?] = []
            var morphPrimitives: [MeshMorphTargets] = []
            for (primitiveIndex, primitive) in mesh.primitives.enumerated() {
                if includeMaterials {
                    // Materials that a variant selects must suit the primitive as well.
                    let variantMaterials = variants?.mappings[meshIndex]?[primitiveIndex].values.sorted() ?? []
                    for materialIndex in Set([primitive.material].compactMap { $0 } + variantMaterials) {
                        var available = Set<MeshTextureCoordinateSet>()
                        if primitive.attributes?["TEXCOORD_0"] != nil { available.insert(.primary) }
                        if primitive.attributes?["TEXCOORD_1"] != nil { available.insert(.secondary) }
                        let authoredTangentSpace = primitive.attributes?["NORMAL"] != nil && primitive.attributes?["TANGENT"] != nil
                        let previous = materialUsage[materialIndex]
                        materialUsage[materialIndex] = GLTFMaterialDocument.Usage(
                            coordinates: previous?.coordinates.intersection(available) ?? available,
                            hasAuthoredTangentSpace: (previous?.hasAuthoredTangentSpace ?? true) && authoredTangentSpace
                        )
                    }
                }
                let draco = try reader.decodeDraco(primitive)
                let loaded = try loadPrimitive(
                    primitive, reader: &reader, allowsMorphTargets: includeMorphTargets, draco: draco
                )
                let geometry = loaded.geometry
                guard geometry.vertices.count <= maximumElements - vertexCount,
                      geometry.indices.count <= maximumElements - indexCount else {
                    throw Failure.sceneSizeLimit
                }
                vertexCount += geometry.vertices.count
                indexCount += geometry.indices.count
                primitives.append(geometry)
                let influences = try jointLimitByMesh[meshIndex].map {
                    try reader.skinInfluences(
                        attributes: primitive.attributes ?? [:], jointCount: $0,
                        sourceVertexCount: loaded.sourceVertexCount, sourceIndices: loaded.sourceIndices, draco: draco
                    )
                }
                skinInfluences.append(influences)
                if let targets = primitive.targets, !targets.isEmpty {
                    // Generated flat normals duplicate vertices and would need recomputing after morphing.
                    guard loaded.sourceIndices == nil else {
                        throw Failure.unsupportedFeature("morph targets on mesh \(meshIndex) without vertex normals")
                    }
                    morphPrimitives.append(try reader.morphTargets(
                        targets, attributes: primitive.attributes ?? [:], mesh: meshIndex,
                        vertexCount: geometry.vertices.count, budget: &morphBudget
                    ))
                }
            }
            var importedMesh = GLTFScene.Mesh(
                sourceIndex: meshIndex, name: mesh.name, primitives: primitives,
                primitiveMaterialIndices: mesh.primitives.map(\.material), primitiveSkinInfluences: skinInfluences,
                morphTargets: includeMorphTargets ? try morphTargets(morphPrimitives, of: mesh, at: meshIndex) : nil
            )
            importedMesh.primitiveMaterialVariants = variants?.mappings[meshIndex] ?? []
            meshes.append(importedMesh)
        }

        var nodes = scene.nodes
        var instanceBudget = maximumElements
        for index in nodes.indices {
            guard let attributes = scene.instancing[nodes[index].sourceIndex] else { continue }
            nodes[index].instances = try reader.instanceTransforms(
                attributes, node: nodes[index].sourceIndex, budget: &instanceBudget
            )
        }
        var morphTargetCounts: [Int: Int] = [:]
        if includeMorphTargets {
            let defaults = Dictionary(uniqueKeysWithValues: meshes.compactMap { mesh in
                mesh.morphTargets.map { (mesh.sourceIndex, $0.defaultWeights) }
            })
            for index in nodes.indices {
                let node = nodes[index]
                let meshDefaults = node.meshIndex.flatMap { defaults[$0] }
                if let weights = node.morphWeights {
                    guard let meshDefaults, weights.count == meshDefaults.count, weights.allSatisfy(\.isFinite) else {
                        throw Failure.invalidMorphWeights(node.sourceIndex, "weights need one finite value per morph target of the node's mesh")
                    }
                } else {
                    nodes[index].morphWeights = meshDefaults
                }
                if let meshDefaults { morphTargetCounts[node.sourceIndex] = meshDefaults.count }
            }
        }

        let materials: [GLTFScene.Material]
        if materialUsage.isEmpty {
            materials = []
        } else {
            materials = try decode(GLTFMaterialDocument.self, from: json).resolve(
                usage: materialUsage, maximumBytes: maximumBytes,
                readView: { try reader.imageData(bufferView: $0) },
                readAdjacent: { try adjacentBuffer($0, directory: directory) }
            )
        }
        let animations: [GLTFAnimationClip]
        if includeAnimations {
            animations = try decode(GLTFAnimationDocument.self, from: json).resolve(
                nodes: decoded.hierarchy.nodes ?? [], selectedNodes: Set(scene.nodes.map(\.sourceIndex)),
                maximumElements: maximumElements, morphTargetCounts: includeMorphTargets ? morphTargetCounts : nil,
                pointerTargets: .init(
                    materials: includeMaterials ? Set(materials.map(\.sourceIndex)) : nil,
                    lights: includeLights ? Set(lights.map(\.sourceIndex)) : nil,
                    cameras: includeCameras ? Set(cameras.map(\.sourceIndex)) : nil,
                    required: decoded.document.extensionsRequired?.contains(GLTFAnimationPointer.extensionName) == true
                ),
                readSampler: { input, output, rotation, valueMultiplier, remaining in
                    let times = try reader.animationTimes(input)
                    let outputCount = times.count * valueMultiplier
                    guard times.count <= remaining, outputCount <= remaining - times.count else {
                        throw Failure.animationSizeLimit
                    }
                    let values = try reader.animationValues(output, rotation: rotation, count: outputCount)
                    return (times, values.values, values.unitTolerance)
                },
                readWeights: { input, output, valuesPerKey, remaining in
                    let times = try reader.animationTimes(input)
                    guard times.count <= remaining, valuesPerKey <= (remaining - times.count) / times.count else {
                        throw Failure.animationSizeLimit
                    }
                    return (times, try reader.animationWeights(output, count: times.count * valuesPerKey))
                },
                readPointer: { input, output, components, valuesPerKey, remaining in
                    let times = try reader.animationTimes(input)
                    guard times.count <= remaining, components * valuesPerKey <= (remaining - times.count) / times.count else {
                        throw Failure.animationSizeLimit
                    }
                    return (times, try reader.animationPointerValues(
                        output, components: components, count: times.count * valuesPerKey
                    ))
                }
            )
        } else {
            animations = []
        }
        var imported = GLTFScene(
            sourceIndex: scene.index, name: scene.name, rootNodeIndices: scene.roots,
            nodes: nodes, meshes: meshes, materials: materials, animations: animations, skins: skins, lights: lights,
            cameras: cameras
        )
        imported.materialVariants = variants?.names ?? []
        return imported
    }

    private static func decode<T: Decodable>(_ type: T.Type, from json: Data) throws -> T {
        do {
            return try JSONDecoder().decode(type, from: json)
        } catch {
            throw Failure.invalidJSON(String(describing: error))
        }
    }

    private static func validate(
        _ document: Document, includeLights: Bool = false, includeMaterials: Bool = false,
        includeGPUInstancing: Bool = false, includeAnimations: Bool = false
    ) throws {
        guard document.asset.version == "2.0" else { throw Failure.unsupportedVersion(document.asset.version) }
        if let minimum = document.asset.minVersion, minimum != "2.0" {
            throw Failure.unsupportedVersion(minimum)
        }
        for name in document.extensionsRequired ?? [] {
            guard honors(
                name, includeLights: includeLights, includeMaterials: includeMaterials,
                includeGPUInstancing: includeGPUInstancing, includeAnimations: includeAnimations
            ) else {
                throw Failure.unsupportedFeature(name)
            }
        }
    }

    /// Validates a mesh's per-primitive morph targets and its default weights.
    private static func morphTargets(
        _ primitives: [MeshMorphTargets], of mesh: Mesh, at index: Int
    ) throws -> GLTFScene.Mesh.MorphTargets? {
        func invalid(_ reason: String) -> Failure { .invalidMorphTarget(index, reason) }
        guard let first = primitives.first else {
            if let weights = mesh.weights, !weights.isEmpty { throw invalid("default weights require morph targets") }
            return nil
        }
        let count = first.targets.count
        guard primitives.count == mesh.primitives.count, primitives.allSatisfy({ $0.targets.count == count }) else {
            throw invalid("every primitive needs the same number of morph targets")
        }
        let weights = mesh.weights ?? Array(repeating: 0, count: count)
        guard weights.count == count, weights.allSatisfy(\.isFinite) else {
            throw invalid("default weights need one finite value per morph target")
        }
        let names = mesh.extras?.targetNames.flatMap { $0.count == count ? $0 : nil }
        return .init(primitives: primitives, defaultWeights: weights, names: names)
    }

    private static func loadPrimitive(
        _ primitive: Primitive, reader: inout Reader, allowsMorphTargets: Bool, draco: DracoPrimitive?
    ) throws -> (geometry: MeshGeometry, sourceVertexCount: Int, sourceIndices: [UInt32]?) {
        let mode = primitive.mode ?? 4
        guard mode == 4 else { throw Failure.unsupportedMode(mode) }
        if !allowsMorphTargets, let targets = primitive.targets, !targets.isEmpty {
            throw Failure.unsupportedFeature("morph targets")
        }
        let attributes = primitive.attributes ?? [:]
        guard let positionIndex = attributes["POSITION"] else { throw Failure.missingPosition }

        let positions = try reader.layout(positionIndex, semantic: "POSITION", components: 3, usage: .position, draco: draco).vectors3()
        let normals = try attributes["NORMAL"].map {
            try reader.layout($0, semantic: "NORMAL", components: 3, usage: .direction, draco: draco).unitVectors3()
        }
        let coordinates = try attributes["TEXCOORD_0"].map {
            try reader.layout($0, semantic: "TEXCOORD_0", components: 2, usage: .textureCoordinate, draco: draco).vectors2()
        }
        let secondaryCoordinates = try attributes["TEXCOORD_1"].map { index in
            guard coordinates != nil else {
                throw Failure.invalidAccessor(index, "TEXCOORD_1 requires TEXCOORD_0")
            }
            return try reader.layout(index, semantic: "TEXCOORD_1", components: 2, usage: .textureCoordinate, draco: draco).vectors2()
        }
        // COLOR_0 multiplies the base color and alpha; VEC3 colors are opaque.
        let colors = try attributes["COLOR_0"].map { try reader.colors($0, draco: draco) }
        var tangents: [SIMD4<Float>]?
        // glTF discards authored tangents when normals have to be generated.
        if normals != nil, let index = attributes["TANGENT"] {
            let values = try reader.layout(index, semantic: "TANGENT", components: 4, usage: .direction, draco: draco).tangents()
            guard values.allSatisfy(MeshGeometry.isValidTangent) else {
                throw Failure.invalidAccessor(index, "tangents require a unit direction and a handedness sign of -1 or +1")
            }
            tangents = values
        }
        guard normals == nil || normals?.count == positions.count,
              coordinates == nil || coordinates?.count == positions.count,
              secondaryCoordinates == nil || secondaryCoordinates?.count == positions.count,
              tangents == nil || tangents?.count == positions.count,
              colors == nil || colors?.count == positions.count else {
            throw Failure.attributeCounts
        }

        let indices: [UInt32]
        if let draco {
            guard positions.count == draco.pointCount else { throw Failure.attributeCounts }
            indices = draco.indices
        } else if let index = primitive.indices {
            let layout = try reader.layout(index, semantic: "indices", components: 1, usage: .indices)
            indices = try layout.indices(vertexCount: positions.count)
        } else {
            indices = positions.indices.map { UInt32($0) }
        }
        guard !indices.isEmpty, indices.count.isMultiple(of: 3) else { throw Failure.invalidTriangleCount }

        if let normals {
            let vertices = positions.indices.map {
                MeshVertex(
                    position: positions[$0], normal: normals[$0], textureCoordinate: coordinates?[$0] ?? .zero,
                    secondaryTextureCoordinate: secondaryCoordinates?[$0], tangent: tangents?[$0],
                    color: colors?[$0]
                )
            }
            return (try MeshGeometry(vertices: vertices, indices: indices), positions.count, nil)
        }

        // A shared index can belong to faces with different flat normals.
        var vertices: [MeshVertex] = []
        vertices.reserveCapacity(indices.count)
        for offset in stride(from: 0, to: indices.count, by: 3) {
            let a = Int(indices[offset])
            let b = Int(indices[offset + 1])
            let c = Int(indices[offset + 2])
            let first = doubleVector(positions[a])
            let normal = simd_cross(doubleVector(positions[b]) - first, doubleVector(positions[c]) - first)
            let length = simd_length(normal)
            guard length > 0 else { throw Failure.degenerateTriangle(offset / 3) }
            let unit = normal / length
            let faceNormal = SIMD3(Float(unit.x), Float(unit.y), Float(unit.z))
            for index in [a, b, c] {
                vertices.append(MeshVertex(
                    position: positions[index], normal: faceNormal, textureCoordinate: coordinates?[index] ?? .zero,
                    secondaryTextureCoordinate: secondaryCoordinates?[index], color: colors?[index]
                ))
            }
        }
        let geometry = try MeshGeometry(vertices: vertices, indices: vertices.indices.map { UInt32($0) })
        return (geometry, positions.count, indices)
    }

    private static func doubleVector(_ value: SIMD3<Float>) -> SIMD3<Double> {
        SIMD3(Double(value.x), Double(value.y), Double(value.z))
    }

    private static func unpack(_ data: Data) throws -> (json: Data, binary: Data?) {
        guard data.count <= maximumBytes else { throw Failure.sizeLimit }
        return try data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
            guard bytes.count >= 4, word(bytes, 0) == 0x4654_6C67 else { return (data, nil) }
            guard bytes.count >= 20 else { throw Failure.invalidGLB("missing header or JSON chunk") }
            guard word(bytes, 4) == 2 else { throw Failure.unsupportedVersion(String(word(bytes, 4))) }
            guard Int(word(bytes, 8)) == bytes.count else { throw Failure.invalidGLB("incorrect total length") }

            var offset = 12
            var chunk = 0
            var json: Data?
            var binary: Data?
            while offset < bytes.count {
                guard bytes.count - offset >= 8 else { throw Failure.invalidGLB("truncated chunk header") }
                let length = Int(word(bytes, offset))
                let type = word(bytes, offset + 4)
                let start = offset + 8
                guard length.isMultiple(of: 4), length <= bytes.count - start else {
                    throw Failure.invalidGLB("invalid chunk length")
                }
                let end = start + length
                if chunk == 0 {
                    guard type == 0x4E4F_534A, length > 0 else { throw Failure.invalidGLB("first chunk must be JSON") }
                    json = Data(bytes[start..<end])
                } else if type == 0x004E_4942 {
                    guard chunk == 1 else { throw Failure.invalidGLB("BIN must be the second chunk") }
                    binary = Data(bytes[start..<end])
                } else if type == 0x4E4F_534A {
                    throw Failure.invalidGLB("duplicate JSON chunk")
                }
                offset = end
                chunk += 1
            }
            guard let json else { throw Failure.invalidGLB("missing JSON chunk") }
            return (json, binary)
        }
    }

    private static func word(_ bytes: UnsafeRawBufferPointer, _ offset: Int) -> UInt32 {
        UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
    }

    private static func adjacentBuffer(_ uri: String, directory: URL?) throws -> Data {
        // POSIX recognizes the slash byte even within a Unicode grapheme cluster.
        guard let name = uri.removingPercentEncoding, !name.isEmpty, name != ".", name != "..",
              name.unicodeScalars.allSatisfy({ !CharacterSet.controlCharacters.contains($0) }),
              !name.utf8.contains(where: { "/\\:?#".utf8.contains($0) }) else {
            throw Failure.unsafeBufferURI(uri)
        }
        guard let directory else { throw Failure.externalBufferRequiresURL(uri) }
        let parent = directory.path.withCString { open($0, O_RDONLY | O_DIRECTORY | O_CLOEXEC) }
        guard parent >= 0 else { throw Failure.unreadableFile(directory.lastPathComponent) }
        defer { _ = close(parent) }
        let descriptor = name.withCString { openat(parent, $0, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK) }
        return try readFile(descriptor, name: name)
    }

    private static func readFile(_ descriptor: Int32, name: String) throws -> Data {
        guard descriptor >= 0 else { throw Failure.unreadableFile(name) }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close() }
        var info = stat()
        guard fstat(descriptor, &info) == 0, (Int32(info.st_mode) & Int32(S_IFMT)) == Int32(S_IFREG) else {
            throw Failure.unreadableFile(name)
        }
        guard info.st_size >= 0, info.st_size <= Int64(maximumBytes) else { throw Failure.sizeLimit }

        do {
            var data = Data()
            data.reserveCapacity(Int(info.st_size))
            while data.count <= maximumBytes,
                  let chunk = try handle.read(upToCount: min(65_536, maximumBytes + 1 - data.count)),
                  !chunk.isEmpty {
                data.append(chunk)
            }
            guard data.count <= maximumBytes else { throw Failure.sizeLimit }
            return data
        } catch let error as Failure {
            throw error
        } catch {
            throw Failure.unreadableFile(name)
        }
    }

    private struct Reader {
        let document: Document
        let binary: Data?
        let directory: URL?
        /// True when the asset declares `KHR_mesh_quantization`, which widens attribute component types.
        let quantized: Bool
        private var cache: [Int: Data] = [:]
        private var decodedViews: [Int: Data] = [:]
        private var totalBytes = 0

        init(document: Document, binary: Data?, directory: URL?) {
            self.document = document
            self.binary = binary
            self.directory = directory
            let declared = (document.extensionsUsed ?? []) + (document.extensionsRequired ?? [])
            quantized = declared.contains(quantizationExtension)
        }

        /// Decodes compressed attributes before interpreting the primitive's accessors.
        /// The result stays scoped to one primitive, even if another reuses an accessor.
        mutating func decodeDraco(_ primitive: Primitive) throws -> DracoPrimitive? {
            guard let record = primitive.extensions?.draco else { return nil }
            let mode = primitive.mode ?? 4
            guard mode == 4 else { throw Failure.unsupportedMode(mode) }
            func invalid(_ reason: String) -> Failure { .invalidDracoCompression(record.bufferView, reason) }
            let mesh: DracoDecoder.Mesh
            do {
                let data = try packedView(record.bufferView, invalid: invalid)
                mesh = try DracoDecoder.decode(data)
            } catch DracoDecoder.Failure.unsupportedGeometry {
                throw Failure.dracoCompressionUnsupported
            } catch let error as DracoDecoder.Failure {
                throw invalid(error.localizedDescription)
            }
            let primitiveAttributes = primitive.attributes ?? [:]
            guard !record.attributes.isEmpty, record.attributes.keys.allSatisfy({ primitiveAttributes[$0] != nil }) else {
                throw invalid("compressed attributes must be declared on the primitive")
            }
            var byID: [UInt32: DracoDecoder.Attribute] = [:]
            for attribute in mesh.attributes {
                guard byID.updateValue(attribute, forKey: attribute.id) == nil else {
                    throw invalid("compressed attribute IDs must be unique")
                }
            }
            var attributes: [String: DracoDecoder.Attribute] = [:]
            var size = mesh.indices.count * MemoryLayout<UInt32>.stride
            for (name, id) in record.attributes {
                guard let attribute = byID[id], let index = primitiveAttributes[name],
                      let accessors = document.accessors, accessors.indices.contains(index) else {
                    throw invalid("attribute \(name) has no Draco data or accessor")
                }
                let accessor = accessors[index]
                let dimensions = ["SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4][accessor.type]
                guard accessor.count == mesh.pointCount, accessor.sparse == nil,
                      dimensions == attribute.components,
                      Component(rawValue: accessor.componentType)?.draco == attribute.component else {
                    throw invalid("attribute \(name) does not match its accessor")
                }
                attributes[name] = attribute
                size += attribute.data.count
            }
            if let index = primitive.indices {
                guard let accessors = document.accessors, accessors.indices.contains(index) else {
                    throw Failure.invalidAccessor(index, "missing Draco index accessor")
                }
                let accessor = accessors[index]
                guard accessor.type == "SCALAR", accessor.count == mesh.indices.count,
                      accessor.normalized != true, accessor.sparse == nil,
                      let component = Component(rawValue: accessor.componentType), let restart = component.restartIndex,
                      mesh.indices.allSatisfy({ $0 < restart }) else {
                    throw Failure.invalidAccessor(index, "Draco indices do not match their accessor")
                }
            }
            guard size <= maximumBytes - totalBytes else { throw Failure.sizeLimit }
            totalBytes += size
            return DracoPrimitive(pointCount: mesh.pointCount, indices: mesh.indices, attributes: attributes)
        }

        /// Reads COLOR_0 as linear RGBA: floats or normalized unsigned bytes or shorts,
        /// VEC3 (opaque) or VEC4.
        mutating func colors(_ index: Int, draco: DracoPrimitive? = nil) throws -> [SIMD4<Float>] {
            guard let accessors = document.accessors, accessors.indices.contains(index) else {
                throw Failure.invalidAccessor(index, "missing accessor")
            }
            let components = accessors[index].type == "VEC3" ? 3 : 4
            let values = try layout(
                index, semantic: "COLOR_0", components: components, usage: .color, draco: draco
            ).vectors4(components: components)
            return components == 4 ? values : values.map { SIMD4($0.x, $0.y, $0.z, 1) }
        }

        /// Reads an accessor's buffer view. Sparse substitution is the caller's job
        /// when `allowsSparse` is true; otherwise sparse accessors are unsupported.
        mutating func layout(
            _ index: Int, semantic: String, components: Int, usage: AccessorUsage = .vertexFloat,
            allowsSparse: Bool = false, draco: DracoPrimitive? = nil
        ) throws -> Layout {
            func invalid(_ reason: String) -> Failure { .invalidAccessor(index, reason) }
            guard let accessors = document.accessors, accessors.indices.contains(index) else { throw invalid("missing accessor") }
            let accessor = accessors[index]
            if accessor.sparse != nil, !allowsSparse { throw Failure.unsupportedFeature("sparse accessor \(index)") }
            let expectedType = components == 1 ? "SCALAR" : components == 16 ? "MAT4" : "VEC\(components)"
            let normalized = accessor.normalized ?? false
            guard accessor.type == expectedType, let component = Component(rawValue: accessor.componentType),
                  usage.allows(component, normalized: normalized, quantized: quantized) else {
                throw invalid("\(semantic) requires \(expectedType) with \(usage.componentDescription(quantized: quantized))")
            }
            guard (1...maximumElements).contains(accessor.count) else { throw invalid("element count is outside the supported range") }
            if let attribute = draco?.attributes[semantic] {
                return Layout(
                    accessor: index, data: attribute.data, start: 0, stride: components * component.width,
                    count: accessor.count, component: component, normalized: normalized
                )
            }
            guard let viewIndex = accessor.bufferView else { throw invalid("missing buffer view") }
            return try viewLayout(
                accessor: index, view: viewIndex, byteOffset: accessor.byteOffset ?? 0, component: component,
                normalized: normalized, components: components, count: accessor.count,
                allowsStride: usage.isVertexAttribute
            )
        }

        private mutating func viewLayout(
            accessor index: Int, view viewIndex: Int, byteOffset accessorOffset: Int, component: Component,
            normalized: Bool = false, components: Int, count: Int, allowsStride: Bool
        ) throws -> Layout {
            func invalid(_ reason: String) -> Failure { .invalidAccessor(index, reason) }
            guard let views = document.bufferViews, views.indices.contains(viewIndex) else {
                throw invalid("missing buffer view")
            }
            let view = views[viewIndex]
            guard let buffers = document.buffers, buffers.indices.contains(view.buffer) else { throw invalid("missing buffer") }
            let buffer = buffers[view.buffer]
            let viewOffset = view.byteOffset ?? 0
            let elementSize = components * component.width
            let stride = view.byteStride ?? elementSize
            if let declaredStride = view.byteStride {
                guard allowsStride, (4...252).contains(declaredStride), declaredStride.isMultiple(of: 4) else {
                    throw invalid("invalid byte stride")
                }
            }
            guard buffer.byteLength > 0, viewOffset >= 0, viewOffset <= buffer.byteLength,
                  view.byteLength > 0, view.byteLength <= buffer.byteLength - viewOffset,
                  accessorOffset >= 0, accessorOffset <= view.byteLength,
                  stride >= elementSize, accessorOffset.isMultiple(of: component.width),
                  (viewOffset + accessorOffset).isMultiple(of: component.width),
                  elementSize <= view.byteLength - accessorOffset,
                  count - 1 <= (view.byteLength - accessorOffset - elementSize) / stride else {
                throw invalid("unaligned access or out of bounds")
            }
            if view.extensions?.meshopt != nil {
                let data = try decompressedView(viewIndex)
                return Layout(
                    accessor: index, data: data, start: accessorOffset, stride: stride, count: count,
                    component: component, normalized: normalized
                )
            }
            guard buffer.extensions?.isMeshoptFallback != true else {
                throw invalid("buffer view \(viewIndex) reads meshopt fallback buffer \(view.buffer) without compression")
            }
            let data = try payload(view.buffer, buffer: buffer)
            return Layout(
                accessor: index, data: data, start: viewOffset + accessorOffset, stride: stride, count: count,
                component: component, normalized: normalized
            )
        }

        /// Decodes a meshopt-compressed view once. Its parent buffer is never read, so
        /// placeholder and fallback buffers need no data.
        private mutating func decompressedView(_ index: Int) throws -> Data {
            if let decoded = decodedViews[index] { return decoded }
            func invalid(_ reason: String) -> Failure { .invalidCompressedBufferView(index, reason) }
            guard let view = document.bufferViews?[index], let extensions = view.extensions,
                  let record = extensions.meshopt else { throw invalid("missing compression data") }
            guard !extensions.hasConflictingMeshopt else {
                throw invalid("a buffer view cannot use both EXT_meshopt_compression and KHR_meshopt_compression")
            }
            guard let source = record.buffer, let length = record.byteLength, let stride = record.byteStride,
                  let count = record.count, let modeName = record.mode else {
                throw invalid("\(record.specification.rawValue) requires buffer, byteLength, byteStride, count, and mode")
            }
            guard let mode = MeshoptDecoder.Mode(rawValue: modeName) else { throw invalid("unknown mode \(modeName)") }
            guard let filter = MeshoptDecoder.Filter(rawValue: record.filter ?? "NONE") else {
                throw invalid("unknown filter \(record.filter ?? "")")
            }
            guard view.byteStride == nil || view.byteStride == stride else {
                throw invalid("the buffer view and extension byte strides differ")
            }
            let (size, overflow) = stride.multipliedReportingOverflow(by: count)
            guard !overflow, size == view.byteLength else {
                throw invalid("byteStride × count must equal the buffer view byteLength")
            }
            guard let buffers = document.buffers, buffers.indices.contains(source) else {
                throw invalid("missing compressed buffer \(source)")
            }
            let buffer = buffers[source]
            guard buffer.extensions?.isMeshoptFallback != true else {
                throw invalid("compressed data cannot come from fallback buffer \(source)")
            }
            let offset = record.byteOffset ?? 0
            guard offset >= 0, length >= 0, offset <= buffer.byteLength, length <= buffer.byteLength - offset else {
                throw invalid("the compressed range is outside buffer \(source)")
            }
            guard size <= maximumBytes - totalBytes else { throw Failure.sizeLimit }
            let bytes = try payload(source, buffer: buffer)
            guard size <= maximumBytes - totalBytes else { throw Failure.sizeLimit }
            let decoded: Data
            do {
                decoded = try MeshoptDecoder.decode(
                    bytes.subdata(in: (bytes.startIndex + offset)..<(bytes.startIndex + offset + length)),
                    count: count, byteStride: stride, mode: mode, filter: filter, specification: record.specification
                )
            } catch let failure as MeshoptDecoder.Failure {
                throw invalid(failure.localizedDescription)
            }
            totalBytes += decoded.count
            decodedViews[index] = decoded
            return decoded
        }

        /// Reads one primitive's morph targets, with offsets in its vertex order.
        mutating func morphTargets(
            _ records: [[String: Int]], attributes: [String: Int], mesh: Int, vertexCount: Int, budget: inout Int
        ) throws -> MeshMorphTargets {
            var targets: [MeshMorphTargets.Target] = []
            for record in records {
                if let name = record.keys.sorted().first(where: { !["POSITION", "NORMAL", "TANGENT"].contains($0) }) {
                    throw Failure.unsupportedFeature("morph target attribute \(name)")
                }
                guard !record.isEmpty else { throw Failure.invalidMorphTarget(mesh, "a morph target needs at least one attribute") }
                if let name = record.keys.sorted().first(where: { attributes[$0] == nil }) {
                    throw Failure.invalidMorphTarget(mesh, "target attribute \(name) is missing from the base primitive")
                }
                func offsets(_ semantic: String) throws -> [SIMD3<Float>]? {
                    guard let index = record[semantic] else { return nil }
                    guard vertexCount <= budget else { throw Failure.morphTargetSizeLimit }
                    budget -= vertexCount
                    return try morphOffsets(
                        index, semantic: "\(semantic) morph target", count: vertexCount,
                        usage: semantic == "POSITION" ? .morphPosition : .direction
                    )
                }
                targets.append(.init(
                    positionOffsets: try offsets("POSITION"), normalOffsets: try offsets("NORMAL"),
                    tangentOffsets: try offsets("TANGENT")
                ))
            }
            return try MeshMorphTargets(vertexCount: vertexCount, targets: targets)
        }

        /// Reads VEC3 offsets, applying sparse substitution over the dense values or
        /// over zeros when the accessor has no buffer view. Quantized offsets are
        /// dequantized but not normalized.
        private mutating func morphOffsets(
            _ index: Int, semantic: String, count: Int, usage: AccessorUsage
        ) throws -> [SIMD3<Float>] {
            func invalid(_ reason: String) -> Failure { .invalidAccessor(index, reason) }
            guard let accessors = document.accessors, accessors.indices.contains(index) else { throw invalid("missing accessor") }
            let accessor = accessors[index]
            guard accessor.count == count else { throw invalid("morph target offsets need one value per vertex") }
            guard let sparse = accessor.sparse else {
                return try layout(index, semantic: semantic, components: 3, usage: usage).vectors3()
            }

            var values: [SIMD3<Float>]
            let normalized = accessor.normalized ?? false
            guard let component = Component(rawValue: accessor.componentType) else {
                throw invalid("\(semantic) requires VEC3 with \(usage.componentDescription(quantized: quantized))")
            }
            if accessor.bufferView != nil {
                values = try layout(index, semantic: semantic, components: 3, usage: usage, allowsSparse: true).vectors3()
            } else {
                guard accessor.type == "VEC3", usage.allows(component, normalized: normalized, quantized: quantized) else {
                    throw invalid("\(semantic) requires VEC3 with \(usage.componentDescription(quantized: quantized))")
                }
                values = Array(repeating: .zero, count: count)
            }
            guard let substitutions = sparse.count, (1...count).contains(substitutions),
                  let indices = sparse.indices, let indexView = indices.bufferView,
                  let indexComponent = indices.componentType.flatMap(Component.init(rawValue:)),
                  [.unsignedByte, .unsignedShort, .unsignedInt].contains(indexComponent),
                  let replacements = sparse.values, let valueView = replacements.bufferView else {
                throw invalid("sparse storage requires a count, unsigned indices, and values")
            }
            let positions = try viewLayout(
                accessor: index, view: indexView, byteOffset: indices.byteOffset ?? 0, component: indexComponent,
                components: 1, count: substitutions, allowsStride: false
            ).unsignedIntegers()
            let substitutes = try viewLayout(
                accessor: index, view: valueView, byteOffset: replacements.byteOffset ?? 0, component: component,
                normalized: normalized, components: 3, count: substitutions, allowsStride: false
            ).vectors3()
            var previous: UInt32?
            for (position, value) in zip(positions, substitutes) {
                guard Int(position) < count, previous.map({ position > $0 }) != false else {
                    throw invalid("sparse indices must increase strictly within the accessor")
                }
                values[Int(position)] = value
                previous = position
            }
            return values
        }

        mutating func animationWeights(_ index: Int, count: Int) throws -> [Float] {
            let layout = try layout(index, semantic: "animation weights", components: 1, usage: .animationWeights)
            guard layout.count == count else {
                throw Failure.invalidAccessor(index, "animation output count does not match the sampler")
            }
            return try layout.numbers()
        }

        /// KHR_animation_pointer outputs: float scalars or vectors, flattened. Cubic
        /// splines count three values per key.
        mutating func animationPointerValues(_ index: Int, components: Int, count: Int) throws -> [Float] {
            let layout = try layout(index, semantic: "animation pointer values", components: components, usage: .animationFloat)
            guard layout.count == count else {
                throw Failure.invalidAccessor(index, "animation output count does not match the sampler")
            }
            let values: [Float] = switch components {
            case 1: try layout.scalars()
            case 2: try layout.vectors2().flatMap { [$0.x, $0.y] }
            case 3: try layout.vectors3().flatMap { [$0.x, $0.y, $0.z] }
            default: try layout.vectors4(components: 4).flatMap { [$0.x, $0.y, $0.z, $0.w] }
            }
            guard values.allSatisfy(\.isFinite) else { throw Failure.invalidAccessor(index, "animation values must be finite") }
            return values
        }

        mutating func animationTimes(_ index: Int) throws -> [Float] {
            let layout = try layout(index, semantic: "animation times", components: 1, usage: .animationFloat)
            let times = try layout.scalars()
            guard let first = times.first, first >= 0, let last = times.last,
                  zip(times, times.dropFirst()).allSatisfy({ $0.0 < $0.1 }) else {
                throw Failure.invalidAccessor(index, "animation times must be nonnegative and strictly increasing")
            }
            let accessor = document.accessors?[index]
            guard accessor?.min == [first], accessor?.max == [last] else {
                throw Failure.invalidAccessor(index, "animation times require matching min and max bounds")
            }
            return times
        }

        mutating func animationValues(
            _ index: Int, rotation: Bool, count: Int
        ) throws -> (values: [SIMD4<Float>], unitTolerance: Float) {
            let components = rotation ? 4 : 3
            let layout = try layout(
                index, semantic: "animation values", components: components,
                usage: rotation ? .animationRotation : .animationFloat
            )
            guard layout.count == count else {
                throw Failure.invalidAccessor(index, "animation output count does not match the sampler")
            }
            return (try layout.vectors4(components: components), layout.component.quaternionTolerance)
        }

        mutating func bindMatrices(_ index: Int, jointCount: Int) throws -> [simd_float4x4] {
            let layout = try layout(index, semantic: "inverse bind matrices", components: 16, usage: .matrixFloat)
            guard layout.count >= jointCount else {
                throw Failure.invalidAccessor(index, "inverse bind matrix count is less than the joint count")
            }
            return try layout.affineMatrices(count: jointCount)
        }

        mutating func skinInfluences(
            attributes: [String: Int], jointCount: Int, sourceVertexCount: Int, sourceIndices: [UInt32]?,
            draco: DracoPrimitive? = nil
        ) throws -> [VertexSkinInfluence] {
            if attributes.keys.contains(where: { key in
                (key.hasPrefix("JOINTS_") && key != "JOINTS_0") || (key.hasPrefix("WEIGHTS_") && key != "WEIGHTS_0")
            }) {
                throw Failure.unsupportedFeature("more than four skin influences per vertex")
            }
            guard let jointIndex = attributes["JOINTS_0"], let weightIndex = attributes["WEIGHTS_0"] else {
                throw Failure.invalidSkinPrimitive("a skinned primitive requires JOINTS_0 and WEIGHTS_0")
            }
            let joints = try layout(jointIndex, semantic: "JOINTS_0", components: 4, usage: .jointIndex, draco: draco)
            let weights = try layout(weightIndex, semantic: "WEIGHTS_0", components: 4, usage: .weights, draco: draco)
            guard joints.count == sourceVertexCount, weights.count == sourceVertexCount else {
                throw Failure.attributeCounts
            }

            let jointVectors = try joints.jointVectors()
            let weightVectors = try weights.vectors4(components: 4)
            var influences: [VertexSkinInfluence] = []
            influences.reserveCapacity(sourceVertexCount)
            for (indices, values) in zip(jointVectors, weightVectors) {
                let total = values.sum()
                // glTF recommends near-unit Float weights and requires exact encoded integer sums.
                guard Int(indices.max()) < jointCount, values.min() >= 0, values.max() <= 1, total > 0,
                      weights.component == .float || abs(total - 1) <= 0.0000008 else {
                    throw Failure.invalidSkinPrimitive("joint indices or weights are out of range")
                }
                for first in 0..<4 where values[first] > 0 {
                    for second in (first + 1)..<4 where values[second] > 0 && indices[first] == indices[second] {
                        throw Failure.invalidSkinPrimitive("a vertex cannot repeat a nonzero joint influence")
                    }
                }
                influences.append(VertexSkinInfluence(joints: indices, weights: values / total))
            }
            if let sourceIndices { return sourceIndices.map { influences[Int($0)] } }
            return influences
        }

        /// Reads the TRS accessors of an `EXT_mesh_gpu_instancing` node. Attributes whose
        /// names start with an underscore are application data; only their counts are checked.
        mutating func instanceTransforms(
            _ attributes: [String: Int], node: Int, budget: inout Int
        ) throws -> [SceneTransform] {
            func invalid(_ reason: String) -> Failure { .invalidInstancing(node, reason) }
            let standard = ["TRANSLATION", "ROTATION", "SCALE"]
            if let unknown = attributes.keys.sorted().first(where: { !standard.contains($0) && !$0.hasPrefix("_") }) {
                throw invalid("unknown attribute \(unknown); custom attributes need a leading underscore")
            }
            guard !attributes.isEmpty else { throw invalid("instancing needs at least one attribute") }
            var count: Int?
            for (name, index) in attributes.sorted(by: { $0.key < $1.key }) {
                guard let accessors = document.accessors, accessors.indices.contains(index) else {
                    throw Failure.invalidAccessor(index, "missing accessor")
                }
                let accessor = accessors[index]
                guard count == nil || count == accessor.count else {
                    throw invalid("attribute \(name) has a different instance count")
                }
                count = accessor.count
            }
            guard let count, (1...maximumElements).contains(count) else { throw invalid("instance count is outside the supported range") }
            guard count <= budget else { throw Failure.sceneSizeLimit }
            budget -= count

            let translations = try attributes["TRANSLATION"].map {
                try layout($0, semantic: "TRANSLATION", components: 3, usage: .vertexFloat).vectors3()
            }
            let scales = try attributes["SCALE"].map {
                try layout($0, semantic: "SCALE", components: 3, usage: .vertexFloat).vectors3()
            }
            let rotations = try attributes["ROTATION"].map { index -> (values: [SIMD4<Float>], tolerance: Float) in
                let layout = try layout(index, semantic: "ROTATION", components: 4, usage: .instanceRotation)
                return (try layout.vectors4(components: 4), layout.component.quaternionTolerance)
            }
            return try (0..<count).map { instance in
                var rotation = simd_quatf(real: 1, imag: .zero)
                if let (values, tolerance) = rotations {
                    let length = simd_length(values[instance])
                    guard abs(length - 1) <= tolerance else { throw invalid("instance \(instance) rotation must be a unit quaternion") }
                    rotation = simd_quatf(vector: values[instance] / length)
                }
                let transform = SceneTransform(
                    translation: translations?[instance] ?? .zero, rotation: rotation,
                    scale: scales?[instance] ?? SIMD3(repeating: 1)
                )
                let matrix = transform.matrix
                let determinant = simd_determinant(matrix)
                let inverse = simd_inverse(matrix)
                let (x, y, z, w) = inverse.columns
                guard determinant.isFinite, determinant != 0,
                      [x, y, z, w].allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite }) else {
                    throw invalid("instance \(instance) transform must be finite and invertible")
                }
                return transform
            }
        }

        mutating func imageData(bufferView index: Int) throws -> Data {
            if let views = document.bufferViews, views.indices.contains(index),
               let extensionName = views[index].extensions?.names.first {
                throw Failure.unsupportedFeature("\(extensionName) on image buffer view \(index)")
            }
            return try packedView(index) { _ in .invalidImageView(index) }
        }

        private mutating func packedView(_ index: Int, invalid: (String) -> Failure) throws -> Data {
            guard let views = document.bufferViews, views.indices.contains(index) else { throw invalid("missing buffer view") }
            let view = views[index]
            guard view.byteStride == nil, view.extensions?.names.isEmpty != false,
                  let buffers = document.buffers, buffers.indices.contains(view.buffer),
                  buffers[view.buffer].extensions?.isMeshoptFallback != true else {
                throw invalid("expected a contiguous uncompressed buffer view")
            }
            let buffer = buffers[view.buffer]
            let offset = view.byteOffset ?? 0
            guard offset >= 0, offset <= buffer.byteLength, view.byteLength > 0,
                  view.byteLength <= buffer.byteLength - offset else {
                throw invalid("buffer view is out of bounds")
            }
            let bytes = try payload(view.buffer, buffer: buffer)
            return bytes.subdata(in: offset..<(offset + view.byteLength))
        }

        private mutating func payload(_ index: Int, buffer: Buffer) throws -> Data {
            if let stored = cache[index] { return stored }
            guard buffer.byteLength <= maximumBytes else { throw Failure.sizeLimit }
            let bytes: Data
            if let uri = buffer.uri {
                if uri.prefix(5).lowercased() == "data:" {
                    let types = ["data:application/octet-stream;base64", "data:application/gltf-buffer;base64"]
                    guard let comma = uri.firstIndex(of: ","), types.contains(uri[..<comma].lowercased()),
                          let payload = String(uri[uri.index(after: comma)...]).removingPercentEncoding,
                          let decoded = Data(base64Encoded: payload) else {
                        throw Failure.invalidBuffer(index)
                    }
                    bytes = decoded
                } else {
                    bytes = try adjacentBuffer(uri, directory: directory)
                }
            } else {
                guard index == 0, let binary, binary.count >= buffer.byteLength,
                      binary.count - buffer.byteLength <= 3,
                      binary.suffix(binary.count - buffer.byteLength).allSatisfy({ $0 == 0 }) else {
                    throw Failure.invalidBuffer(index)
                }
                bytes = binary
            }
            guard bytes.count >= buffer.byteLength else { throw Failure.invalidBuffer(index) }
            guard bytes.count <= maximumBytes - totalBytes else { throw Failure.sizeLimit }
            totalBytes += bytes.count
            cache[index] = bytes
            return bytes
        }
    }

    private enum AccessorUsage {
        case vertexFloat, indices, animationFloat, animationRotation, animationWeights, matrixFloat, jointIndex, weights
        case textureCoordinate
        /// POSITION; `KHR_mesh_quantization` adds 8-bit and 16-bit integers, normalized or not.
        case position
        /// NORMAL, TANGENT, and their morph offsets; quantization adds normalized BYTE and SHORT.
        case direction
        /// Morph POSITION offsets; quantization adds BYTE and SHORT, normalized or not.
        case morphPosition
        /// `EXT_mesh_gpu_instancing` ROTATION.
        case instanceRotation
        /// COLOR_0: floats or normalized unsigned bytes or shorts.
        case color

        var isVertexAttribute: Bool {
            switch self {
            case .vertexFloat, .jointIndex, .weights, .textureCoordinate, .position, .direction, .morphPosition,
                 .instanceRotation, .color: true
            default: false
            }
        }

        func componentDescription(quantized: Bool) -> String {
            switch self {
            case .indices: "unsigned byte, short, or int components"
            case .animationRotation, .animationWeights, .instanceRotation: "floats or normalized byte or short components"
            case .jointIndex: "unsigned byte or short components"
            case .weights, .color: "floats or normalized unsigned byte or short components"
            case .textureCoordinate:
                quantized ? "floats or 8-bit or 16-bit integer components" : "floats or normalized unsigned byte or short components"
            case .position: quantized ? "floats or 8-bit or 16-bit integer components" : "float components"
            case .direction: quantized ? "floats or normalized byte or short components" : "float components"
            case .morphPosition: quantized ? "floats or signed 8-bit or 16-bit integer components" : "float components"
            default: "float components"
            }
        }

        func allows(_ component: Component, normalized: Bool, quantized: Bool) -> Bool {
            switch self {
            case .indices:
                return !normalized && (component == .unsignedByte || component == .unsignedShort || component == .unsignedInt)
            case .jointIndex:
                return !normalized && (component == .unsignedByte || component == .unsignedShort)
            default:
                if component == .float { return !normalized }
            }
            switch self {
            case .vertexFloat, .animationFloat, .matrixFloat, .indices, .jointIndex:
                return false
            case .animationRotation, .animationWeights:
                return normalized && component.isNormalizable
            case .weights, .color:
                return normalized && (component == .unsignedByte || component == .unsignedShort)
            case .textureCoordinate:
                if normalized, component == .unsignedByte || component == .unsignedShort { return true }
                return quantized && component.isNormalizable
                    && (component == .byte || component == .short || !normalized)
            case .position:
                return quantized && component.isNormalizable
            case .direction, .instanceRotation:
                return (quantized || self == .instanceRotation) && normalized
                    && (component == .byte || component == .short)
            case .morphPosition:
                return quantized && (component == .byte || component == .short)
            }
        }
    }

    private enum Component: Int {
        case byte = 5_120, unsignedByte = 5_121, short = 5_122, unsignedShort = 5_123
        case unsignedInt = 5_125, float = 5_126

        var draco: DracoDecoder.Component {
            switch self {
            case .byte: .int8
            case .unsignedByte: .uint8
            case .short: .int16
            case .unsignedShort: .uint16
            case .unsignedInt: .uint32
            case .float: .float32
            }
        }

        var restartIndex: UInt32? {
            switch self {
            case .unsignedByte: UInt32(UInt8.max)
            case .unsignedShort: UInt32(UInt16.max)
            case .unsignedInt: UInt32.max
            default: nil
            }
        }

        var width: Int {
            switch self {
            case .byte, .unsignedByte: 1
            case .short, .unsignedShort: 2
            case .unsignedInt, .float: 4
            }
        }

        var isNormalizable: Bool { width < 4 }

        var quaternionTolerance: Float {
            switch self {
            case .byte: 1 / 127 + 0.0001
            case .unsignedByte: 1 / 255 + 0.0001
            case .short: 1 / 32_767 + 0.0001
            case .unsignedShort: 1 / 65_535 + 0.0001
            default: 0.0001
            }
        }
    }

    private struct Layout {
        let accessor: Int
        let data: Data
        let start: Int
        let stride: Int
        let count: Int
        let component: Component
        /// Integer components map to [0, 1] or [-1, 1] when true and keep their value otherwise.
        var normalized = false

        func vectors3() throws -> [SIMD3<Float>] {
            let width = component.width
            return try read { bytes, offset in
                SIMD3(try number(bytes, offset), try number(bytes, offset + width), try number(bytes, offset + 2 * width))
            }
        }

        /// Quantized directions are not exactly unit length, so they are renormalized.
        /// Zero vectors are left for geometry validation to reject.
        func unitVectors3() throws -> [SIMD3<Float>] {
            let values = try vectors3()
            guard component != .float else { return values }
            return values.map { $0 == .zero ? $0 : simd_normalize($0) }
        }

        /// TANGENT values, with quantized directions renormalized and the handedness kept.
        func tangents() throws -> [SIMD4<Float>] {
            let values = try vectors4(components: 4)
            guard component != .float else { return values }
            return values.map { value in
                let direction = SIMD3(value.x, value.y, value.z)
                guard direction != .zero else { return value }
                return SIMD4(simd_normalize(direction), value.w)
            }
        }

        func vectors2() throws -> [SIMD2<Float>] {
            try read { bytes, offset in
                SIMD2(try number(bytes, offset), try number(bytes, offset + component.width))
            }
        }

        func scalars() throws -> [Float] {
            try read { bytes, offset in try float(bytes, offset) }
        }

        /// Floats or normalized integers, one per element.
        func numbers() throws -> [Float] {
            try read { bytes, offset in try number(bytes, offset) }
        }

        func unsignedIntegers() throws -> [UInt32] {
            try read { bytes, offset in
                switch component {
                case .unsignedByte: return UInt32(bytes[offset])
                case .unsignedShort: return UInt32(short(bytes, offset))
                case .unsignedInt: return word(bytes, offset)
                case .byte, .short, .float: throw Failure.invalidAccessor(accessor, "index component must be unsigned")
                }
            }
        }

        func vectors4(components: Int) throws -> [SIMD4<Float>] {
            let width = component.width
            return try read { bytes, offset in
                SIMD4(
                    try number(bytes, offset), try number(bytes, offset + width),
                    try number(bytes, offset + 2 * width), components == 4 ? try number(bytes, offset + 3 * width) : 0
                )
            }
        }

        func jointVectors() throws -> [SIMD4<UInt16>] {
            let width = component.width
            return try read { bytes, offset in
                func joint(_ lane: Int) -> UInt16 {
                    let start = offset + lane * width
                    return width == 1 ? UInt16(bytes[start]) : short(bytes, start)
                }
                return SIMD4(joint(0), joint(1), joint(2), joint(3))
            }
        }

        func affineMatrices(count: Int) throws -> [simd_float4x4] {
            try read(count: count) { bytes, offset in
                func column(_ lane: Int) throws -> SIMD4<Float> {
                    let base = offset + 16 * lane
                    return SIMD4(
                        try float(bytes, base), try float(bytes, base + 4),
                        try float(bytes, base + 8), try float(bytes, base + 12)
                    )
                }
                let x = try column(0), y = try column(1), z = try column(2), w = try column(3)
                guard x.w == 0, y.w == 0, z.w == 0, w.w == 1 else {
                    throw Failure.invalidAccessor(accessor, "inverse bind matrices must be affine")
                }
                return simd_float4x4(columns: (x, y, z, w))
            }
        }

        func indices(vertexCount: Int) throws -> [UInt32] {
            try read { bytes, offset in
                let value: UInt32
                let restart: UInt32
                switch component {
                case .unsignedByte:
                    value = UInt32(bytes[offset])
                    restart = UInt32(UInt8.max)
                case .unsignedShort:
                    value = UInt32(UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self)))
                    restart = UInt32(UInt16.max)
                case .unsignedInt:
                    value = word(bytes, offset)
                    restart = UInt32.max
                case .byte, .short, .float:
                    throw Failure.invalidAccessor(accessor, "index component must be unsigned")
                }
                guard value != restart, value < UInt32(vertexCount) else { throw Failure.invalidIndex(value) }
                return value
            }
        }

        private func read<T>(count limit: Int? = nil, _ element: (UnsafeRawBufferPointer, Int) throws -> T) throws -> [T] {
            try data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
                try (0..<(limit ?? count)).map { try element(bytes, start + $0 * stride) }
            }
        }

        private func short(_ bytes: UnsafeRawBufferPointer, _ offset: Int) -> UInt16 {
            UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
        }

        /// Dequantizes with the glTF normalization equations, or converts integers directly.
        private func number(_ bytes: UnsafeRawBufferPointer, _ offset: Int) throws -> Float {
            switch component {
            case .float: try float(bytes, offset)
            case .byte:
                normalized ? max(Float(Int8(bitPattern: bytes[offset])) / 127, -1) : Float(Int8(bitPattern: bytes[offset]))
            case .unsignedByte: normalized ? Float(bytes[offset]) / 255 : Float(bytes[offset])
            case .short:
                normalized
                    ? max(Float(Int16(bitPattern: short(bytes, offset))) / 32_767, -1)
                    : Float(Int16(bitPattern: short(bytes, offset)))
            case .unsignedShort: normalized ? Float(short(bytes, offset)) / 65_535 : Float(short(bytes, offset))
            case .unsignedInt: throw Failure.invalidAccessor(accessor, "unsupported floating-point component")
            }
        }

        private func float(_ bytes: UnsafeRawBufferPointer, _ offset: Int) throws -> Float {
            let value = Float(bitPattern: word(bytes, offset))
            guard value.isFinite else { throw Failure.invalidAccessor(accessor, "nonfinite float component") }
            return value
        }
    }

    private struct Document: Decodable {
        let asset: Asset
        let extensionsUsed: [String]?
        let extensionsRequired: [String]?
        let meshes: [Mesh]?
        let accessors: [Accessor]?
        let bufferViews: [BufferView]?
        let buffers: [Buffer]?
    }

    private struct SceneDocument: Decodable {
        let document: Document
        let hierarchy: GLTFHierarchy

        init(from decoder: any Decoder) throws {
            document = try Document(from: decoder)
            hierarchy = try GLTFHierarchy(from: decoder)
        }
    }

    private struct Asset: Decodable { let version: String; let minVersion: String? }
    private struct Mesh: Decodable {
        let name: String?
        let primitives: [Primitive]
        let weights: [Float]?
        let extras: MeshExtras?
    }
    /// Application-specific data; only a well-formed `targetNames` string array is used.
    private struct MeshExtras: Decodable {
        let targetNames: [String]?

        private enum CodingKeys: String, CodingKey { case targetNames }

        init(from decoder: any Decoder) throws {
            let container = try? decoder.container(keyedBy: CodingKeys.self)
            targetNames = (try? container?.decodeIfPresent([String].self, forKey: .targetNames)) ?? nil
        }
    }
    private struct Primitive: Decodable {
        let mode: Int?
        let attributes: [String: Int]?
        let indices: Int?
        let material: Int?
        let extensions: Extensions?
        let targets: [[String: Int]]?

        struct Extensions: Decodable {
            let draco: DracoRecord?
            private enum CodingKeys: String, CodingKey { case draco = "KHR_draco_mesh_compression" }
        }
    }
    private struct DracoRecord: Decodable {
        let bufferView: Int
        let attributes: [String: UInt32]
    }
    private struct DracoPrimitive {
        let pointCount: Int
        let indices: [UInt32]
        let attributes: [String: DracoDecoder.Attribute]
    }
    private struct Accessor: Decodable {
        let bufferView: Int?
        let byteOffset: Int?
        let componentType: Int
        let normalized: Bool?
        let count: Int
        let type: String
        let min: [Float]?
        let max: [Float]?
        let sparse: Sparse?
    }
    /// Fields are optional so that unsupported sparse use reports the feature, not the JSON shape.
    private struct Sparse: Decodable {
        let count: Int?
        let indices: SparseIndices?
        let values: SparseValues?
    }
    private struct SparseIndices: Decodable { let bufferView: Int?; let byteOffset: Int?; let componentType: Int? }
    private struct SparseValues: Decodable { let bufferView: Int?; let byteOffset: Int? }
    private struct BufferView: Decodable {
        let buffer: Int
        let byteOffset: Int?
        let byteLength: Int
        let byteStride: Int?
        let extensions: BufferViewExtensions?
    }
    private struct Buffer: Decodable { let uri: String?; let byteLength: Int; let extensions: BufferExtensions? }

    /// Extension names on a buffer view, with the meshopt record decoded when present.
    private struct BufferViewExtensions: Decodable {
        let names: [String]
        let meshopt: MeshoptRecord?
        let hasConflictingMeshopt: Bool

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: ExtensionName.self)
            names = container.allKeys.map(\.stringValue).sorted()
            let records = try MeshoptDecoder.Specification.allCases.compactMap { specification -> MeshoptRecord? in
                let key = ExtensionName(stringValue: specification.rawValue)
                guard container.contains(key) else { return nil }
                var record = try container.decode(MeshoptRecord.self, forKey: key)
                record.specification = specification
                return record
            }
            meshopt = records.first
            hasConflictingMeshopt = records.count > 1
        }
    }

    /// Fields are optional so that missing values report the extension, not the JSON shape.
    private struct MeshoptRecord: Decodable {
        let buffer: Int?
        let byteOffset: Int?
        let byteLength: Int?
        let byteStride: Int?
        let count: Int?
        let mode: String?
        let filter: String?
        var specification = MeshoptDecoder.Specification.ext

        private enum CodingKeys: String, CodingKey { case buffer, byteOffset, byteLength, byteStride, count, mode, filter }
    }

    private struct BufferExtensions: Decodable {
        /// A meshopt fallback buffer holds no data this loader reads.
        let isMeshoptFallback: Bool

        private struct Fallback: Decodable { let fallback: Bool? }

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: ExtensionName.self)
            isMeshoptFallback = try MeshoptDecoder.Specification.allCases.contains { specification in
                let key = ExtensionName(stringValue: specification.rawValue)
                return try container.decodeIfPresent(Fallback.self, forKey: key)?.fallback == true
            }
        }
    }

    private struct ExtensionName: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }

    public enum Failure: LocalizedError, Equatable {
        case localFileRequired
        case unreadableFile(String)
        case unsafeBufferURI(String)
        case externalBufferRequiresURL(String)
        case sizeLimit
        case invalidJSON(String)
        case invalidGLB(String)
        case unsupportedVersion(String)
        case unsupportedFeature(String)
        case missingScene(Int)
        case missingNode(Int)
        case duplicateNodeReference(Int)
        case multipleParents(Int)
        case nodeCycle(Int)
        case sceneRootHasParent(Int)
        case invalidNodeTransform(Int, String)
        case sceneSizeLimit
        case invalidAnimation(Int, String)
        case animationSizeLimit
        case invalidMorphTarget(Int, String)
        case invalidMorphWeights(Int, String)
        case morphTargetSizeLimit
        case missingSkin(Int)
        case missingLight(Int)
        case invalidLight(Int, String)
        case invalidLightExtension
        case lightInstanceLimit(Int)
        case missingCamera(Int)
        case invalidCamera(Int, String)
        case invalidSkin(Int, String)
        case invalidSkinPrimitive(String)
        case missingMaterial(Int)
        case invalidMaterial(Int, String)
        case invalidMaterialVariants(String)
        case missingMaterialVariant(String)
        case missingTexture(Int)
        case invalidTexture(Int, String)
        case missingImage(Int)
        case invalidImage(Int, String)
        case invalidImageView(Int)
        case missingMesh(Int)
        case missingPrimitive(Int)
        case unsupportedMode(Int)
        case missingPosition
        case invalidAccessor(Int, String)
        case invalidBuffer(Int)
        /// A meshopt-compressed buffer view has an invalid layout or bitstream.
        case invalidCompressedBufferView(Int, String)
        /// A Draco payload uses a geometry type other than a triangle mesh.
        case dracoCompressionUnsupported
        /// A compressed primitive's buffer view, stream or accessor mapping is invalid.
        case invalidDracoCompression(Int, String)
        case invalidInstancing(Int, String)
        /// The image needs Basis Universal transcoding and its texture has no usable fallback.
        case textureTranscodingUnsupported(image: Int, String)
        case attributeCounts
        case invalidTriangleCount
        case invalidIndex(UInt32)
        case degenerateTriangle(Int)

        public var errorDescription: String? {
            switch self {
            case .localFileRequired: "The glTF URL must identify a local file."
            case .unreadableFile(let name): "Could not read glTF file \(name)."
            case .unsafeBufferURI(let uri): "The glTF resource URI must name one adjacent file. Received \(uri)."
            case .externalBufferRequiresURL(let uri): "Load the glTF from a URL to resolve resource \(uri)."
            case .sizeLimit: "The glTF document or its referenced buffers exceed the 64 MiB limit."
            case .invalidJSON(let reason): "Invalid glTF JSON. \(reason)"
            case .invalidGLB(let reason): "Invalid GLB container. \(reason)"
            case .unsupportedVersion(let version): "glTF version \(version) is unsupported. Expected 2.0."
            case .unsupportedFeature(let feature): "Unsupported glTF feature \(feature)."
            case .missingScene(let index): "glTF scene \(index) does not exist."
            case .missingNode(let index): "glTF node \(index) does not exist."
            case .duplicateNodeReference(let index): "glTF node \(index) is referenced more than once in the same root or child list."
            case .multipleParents(let index): "glTF node \(index) has more than one parent."
            case .nodeCycle(let index): "The glTF hierarchy has a cycle at node \(index)."
            case .sceneRootHasParent(let index): "glTF scene root node \(index) already has a parent."
            case .invalidNodeTransform(let index, let reason): "Invalid glTF node \(index) transform. \(reason)"
            case .sceneSizeLimit: "The glTF scene exceeds the limit of one million nodes, vertices, or indices."
            case .invalidAnimation(let index, let reason): "Invalid glTF animation \(index). \(reason)"
            case .animationSizeLimit: "The glTF animation data exceeds the supported element limit."
            case .invalidMorphTarget(let index, let reason): "Invalid glTF morph targets on mesh \(index). \(reason)"
            case .invalidMorphWeights(let index, let reason): "Invalid glTF morph weights on node \(index). \(reason)"
            case .morphTargetSizeLimit: "The glTF morph targets exceed the limit of eight million offsets."
            case .missingSkin(let index): "glTF skin \(index) does not exist."
            case .missingLight(let index): "glTF light \(index) does not exist."
            case .invalidLight(let index, let reason): "Invalid glTF light \(index). \(reason)"
            case .invalidLightExtension: "KHR_lights_punctual requires at least one light definition."
            case .lightInstanceLimit(let limit): "A glTF scene supports at most \(limit) punctual light instances."
            case .missingCamera(let index): "glTF camera \(index) does not exist."
            case .invalidCamera(let index, let reason): "Invalid glTF camera \(index). \(reason)"
            case .invalidSkin(let index, let reason): "Invalid glTF skin \(index). \(reason)"
            case .invalidSkinPrimitive(let reason): "Invalid glTF skinned primitive. \(reason)"
            case .missingMaterial(let index): "glTF material \(index) does not exist."
            case .invalidMaterial(let index, let reason): "Invalid glTF material \(index). \(reason)"
            case .invalidMaterialVariants(let reason): "Invalid KHR_materials_variants data: \(reason)."
            case .missingMaterialVariant(let variant): "The glTF scene has no material variant \(variant)."
            case .missingTexture(let index): "glTF texture \(index) does not exist."
            case .invalidTexture(let index, let reason): "Invalid glTF texture \(index). \(reason)"
            case .missingImage(let index): "glTF image \(index) does not exist."
            case .invalidImage(let index, let reason): "Invalid glTF image \(index). \(reason)"
            case .invalidImageView(let index): "Invalid glTF image buffer view \(index)."
            case .missingMesh(let index): "glTF mesh \(index) does not exist."
            case .missingPrimitive(let index): "glTF mesh primitive \(index) does not exist."
            case .unsupportedMode(let mode): "glTF primitive mode \(mode) is unsupported. Expected TRIANGLES (4)."
            case .missingPosition: "The glTF primitive has no POSITION accessor."
            case .invalidAccessor(let index, let reason): "Invalid glTF accessor \(index). \(reason)"
            case .invalidBuffer(let index): "The data for glTF buffer \(index) is missing or invalid."
            case .invalidCompressedBufferView(let index, let reason): "Invalid meshopt-compressed glTF buffer view \(index). \(reason)"
            case .dracoCompressionUnsupported:
                "Draco-compressed geometry (KHR_draco_mesh_compression) must contain a triangle mesh."
            case .invalidDracoCompression(let index, let reason):
                "Invalid Draco-compressed glTF buffer view \(index). \(reason)"
            case .invalidInstancing(let index, let reason): "Invalid EXT_mesh_gpu_instancing data on glTF node \(index). \(reason)"
            case .textureTranscodingUnsupported(let index, let format):
                "glTF image \(index) stores \(format), which needs Basis Universal transcoding. Frostfield does not transcode Basis payloads; add a PNG or JPEG fallback source or use a KTX2 file with a GPU-native format."
            case .attributeCounts: "The glTF attributes must have the same vertex count."
            case .invalidTriangleCount: "A glTF triangle primitive needs a positive multiple of three indices."
            case .invalidIndex(let index): "glTF vertex index \(index) is out of bounds or a reserved restart value."
            case .degenerateTriangle(let index): "Cannot generate a flat normal for degenerate glTF triangle \(index)."
            }
        }
    }
}
