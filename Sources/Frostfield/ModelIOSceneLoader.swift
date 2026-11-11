import CoreGraphics
import Foundation
import ImageIO
import ModelIO
import simd

/// Imports USD (`.usd`, `.usda`, `.usdc`, `.usdz`), Wavefront OBJ with MTL, PLY, STL, and
/// Alembic files through ModelIO into a `GLTFScene`, so the result prepares with
/// `PreparedGLTFScene`, exports with `GLTFExporter` or `USDZExporter`, and its materials
/// prepare with `GLTFScene.Material.prepare(device:)`.
///
/// The scene keeps ModelIO's object hierarchy, names, and local transforms in depth-first
/// order. Each `MDLMesh` becomes a mesh whose primitives are its submeshes. ModelIO keeps
/// polygon topology and Frostfield ear-clips each face in its plane, so concave faces stay
/// inside their outline, where ModelIO's own triangulation and Three.js `OBJLoader` fan them.
/// Primitives carry positions, normals, and the first two texture coordinate sets flipped to
/// Frostfield's top-left origin. ModelIO exposes no tangent attribute from any format it
/// reads (USD `primvars:tangents` included), so primitives have no tangents and normal maps
/// use the frame the renderer derives from texture coordinates. Meshes without normals get
/// ModelIO-generated normals. Perspective `MDLCamera` objects become cameras, and
/// time-sampled transforms become one LINEAR animation clip sampled at ModelIO's key times.
///
/// Materials map UsdPreviewSurface inputs as ModelIO reports them (`diffuseColor`,
/// `metallic`, `roughness`, `normal`, `occlusion`, `emissiveColor`, `opacity`, `clearcoat`,
/// `clearcoatRoughness`, `ior`) and ModelIO's physically plausible semantics for other
/// formats. OBJ materials come from the referenced MTL files, parsed by Frostfield, so
/// `Kd`, `d`/`Tr`, `Ke`, `Ns`, `Ni`, `Pr`, `Pm`, `Pc`, `Pcr`, and their maps survive.
/// `bump` and `map_bump` retain a native linear height field and its authored options.
/// Portable glTF or USDZ export requires converting an active height map to a normal map.
/// Separate metallic, roughness, and opacity maps are packed into glTF channel layout as
/// PNG images. Texture files must be regular files inside the asset's folder or the
/// USDZ package itself. A USD mesh whose text root layer authors `doubleSided = 1` gets a
/// double-sided copy of its bound material.
///
/// ModelIO reports UsdUVTexture file paths but not their `scale`, `bias`, channel
/// selection, wrap modes, or UsdTransform2d nodes, so imported USD maps use repeat
/// wrapping, identity transforms, and unit factors; a map shared by `metallic` and
/// `roughness` is read with glTF's green-roughness, blue-metallic layout. ModelIO does
/// not import UsdLux lights or PLY texture coordinates.
public enum ModelIOSceneLoader {
    public struct Options: Equatable, Sendable {
        /// Imports materials and their textures. Without them, primitives use the default material.
        public var includeMaterials: Bool
        public var includeCameras: Bool
        /// Samples time-varying node transforms into a LINEAR animation clip.
        public var includeAnimations: Bool
        /// Scales text USD stages that declare `metersPerUnit` so the scene is in meters.
        public var convertsToMeters: Bool
        /// Rotates Z-up stages by −90° about X so their +Z axis becomes Frostfield's +Y.
        public var convertsToYUp: Bool
        /// ModelIO's crease threshold for generated normals, from zero (smooth across any
        /// angle) to one (faceted). Adjacent faces whose normals' dot product exceeds it share normals.
        public var normalCreaseThreshold: Float

        public init(
            includeMaterials: Bool = true, includeCameras: Bool = true, includeAnimations: Bool = true,
            convertsToMeters: Bool = true, convertsToYUp: Bool = true, normalCreaseThreshold: Float = 0.9
        ) {
            self.includeMaterials = includeMaterials
            self.includeCameras = includeCameras
            self.includeAnimations = includeAnimations
            self.convertsToMeters = convertsToMeters
            self.convertsToYUp = convertsToYUp
            self.normalCreaseThreshold = normalCreaseThreshold
        }
    }

    /// A lossy or inferred conversion. Names are ModelIO object paths or material names.
    public enum Warning: Equatable, Sendable, CustomStringConvertible {
        /// The mesh had no normals, so ModelIO generated them.
        case normalsGenerated(mesh: String)
        /// Zero or non-finite normals were replaced by their triangle's face normal.
        case normalsRepaired(mesh: String, count: Int)
        /// The mesh's color format or encoding was not imported.
        case vertexColorsDropped(mesh: String)
        /// A submesh of points, lines, or another non-triangle type was skipped.
        case submeshSkipped(mesh: String, geometryType: Int)
        /// ModelIO reported an all-zero transform matrix, which was replaced by identity.
        case degenerateTransformIgnored(node: String)
        /// The object resets its parent transform, which `GLTFScene` cannot express.
        case resetsTransformIgnored(node: String)
        /// The object is a light. ModelIO lights are not converted.
        case lightIgnored(node: String)
        /// Orthographic or invalid cameras were skipped.
        case cameraSkipped(node: String)
        /// A material texture could not be read: it is missing, outside the asset folder, or not an image.
        case textureNotLoaded(material: String, path: String)
        /// A non-PNG/JPEG/WebP/AVIF/KTX2 image was decoded and stored as PNG.
        case imageTranscoded(path: String)
        /// The material uses a statement or input the importer does not translate.
        case materialInputIgnored(material: String, input: String)
        /// Legacy warning retained for callers. Imported MTL bump maps now keep their heights.
        case bumpMapUsedAsNormalMap(material: String)
        /// Maps that should share one packed image use different transforms or coordinates;
        /// the first map's placement was used.
        case packedMapPlacementMerged(material: String)
        /// The USD root layer is binary, so its `metersPerUnit` and `doubleSided` opinions could not
        /// be read; one meter per unit and single-sided meshes were assumed.
        case binaryRootLayer
        /// A time-sampled transform could not be decomposed at some key and was not animated.
        case animationSkipped(node: String)

        public var description: String {
            switch self {
            case .normalsGenerated(let mesh): "Mesh \(mesh) had no normals; ModelIO generated them."
            case .normalsRepaired(let mesh, let count): "Mesh \(mesh) had \(count) invalid normals replaced by face normals."
            case .vertexColorsDropped(let mesh): "Mesh \(mesh) vertex colors were dropped."
            case .submeshSkipped(let mesh, let type): "Mesh \(mesh) skipped a submesh of ModelIO geometry type \(type)."
            case .degenerateTransformIgnored(let node): "Node \(node) had an all-zero transform, replaced by identity."
            case .resetsTransformIgnored(let node): "Node \(node) resets its parent transform, which was ignored."
            case .lightIgnored(let node): "Light \(node) was not imported."
            case .cameraSkipped(let node): "Camera \(node) is orthographic or invalid and was skipped."
            case .textureNotLoaded(let material, let path): "Material \(material) could not load texture \(path)."
            case .imageTranscoded(let path): "Image \(path) was transcoded to PNG."
            case .materialInputIgnored(let material, let input): "Material \(material) ignored \(input)."
            case .bumpMapUsedAsNormalMap(let material): "Material \(material) used its bump map as a normal map."
            case .packedMapPlacementMerged(let material): "Material \(material) packed maps with different placements."
            case .binaryRootLayer:
                "The binary USD root layer's metersPerUnit and doubleSided were not read; one meter per unit and single-sided meshes were assumed."
            case .animationSkipped(let node): "Node \(node) has time samples that could not be imported."
            }
        }
    }

    /// An imported scene and its import report.
    public struct Asset {
        public let scene: GLTFScene
        public let warnings: [Warning]
        /// The `metersPerUnit` a text USD root layer declares, before conversion.
        public let sourceMetersPerUnit: Double?
        /// The up axis ModelIO reports, before conversion.
        public let sourceUpAxis: SIMD3<Float>
        /// Each imported node's ModelIO object path, keyed by node source index.
        public let objectPaths: [Int: String]
    }

    public enum Failure: LocalizedError, Equatable {
        case localFileRequired
        /// ModelIO on this system cannot import the file extension.
        case unsupportedFormat(String)
        case inputTooLarge
        case invalidOptions
        /// ModelIO failed to read the file.
        case modelIO(String)
        /// The file contains no objects or no triangles.
        case emptyAsset
        case invalidGeometry(mesh: String, reason: String)
        /// The transform contains shear or a projection, which `SceneTransform` cannot express.
        case unsupportedTransform(String)

        public var errorDescription: String? {
            switch self {
            case .localFileRequired: "The asset URL must point to a regular local file."
            case .unsupportedFormat(let ext): "ModelIO on this system cannot import .\(ext) files."
            case .inputTooLarge: "The asset exceeds the 512 MiB file or 16,777,216-vertex import limit."
            case .invalidOptions: "The normal crease threshold must be between zero and one."
            case .modelIO(let detail): "ModelIO could not read the asset. \(detail)"
            case .emptyAsset: "The asset contains no triangle meshes, cameras, or nodes."
            case .invalidGeometry(let mesh, let reason): "Mesh \(mesh) is invalid: \(reason)"
            case .unsupportedTransform(let node): "Node \(node) has a sheared or projective transform."
            }
        }
    }

    /// The extensions this loader accepts that ModelIO can import on this system.
    public static var supportedFileExtensions: [String] {
        ["usd", "usda", "usdc", "usdz", "obj", "ply", "stl", "abc"].filter { MDLAsset.canImportFileExtension($0) }
    }

    private static let maximumFileBytes = 512 * 1_024 * 1_024
    private static let maximumVertices = 16_777_216
    // ModelIO's thread safety is undocumented; serialize parsing process-wide.
    private static let parseLock = NSLock()

    /// Imports a regular local file. The caller supplies any required sandbox access.
    public static func load(contentsOf url: URL, options: Options = .init()) throws -> Asset {
        guard url.isFileURL else { throw Failure.localFileRequired }
        guard (0...1).contains(options.normalCreaseThreshold) else { throw Failure.invalidOptions }
        let fileExtension = url.pathExtension.lowercased()
        guard supportedFileExtensions.contains(fileExtension) else { throw Failure.unsupportedFormat(fileExtension) }
        let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values?.isRegularFile == true else { throw Failure.localFileRequired }
        guard (values?.fileSize ?? 0) <= maximumFileBytes else { throw Failure.inputTooLarge }

        parseLock.lock()
        defer { parseLock.unlock() }
        var error: NSError?
        // Keep polygons so concave faces are ear-clipped here instead of fanned by ModelIO.
        let asset = MDLAsset(url: url, vertexDescriptor: nil, bufferAllocator: nil, preserveTopology: true, error: &error)
        if let error {
            throw Failure.modelIO((error.userInfo["MDLErrorKey"] as? String) ?? error.localizedDescription)
        }
        guard asset.count > 0 else { throw Failure.emptyAsset }
        var importer = ModelIOImporter(url: url, fileExtension: fileExtension, asset: asset, options: options)
        return try importer.run()
    }
}

private struct ModelIOImporter {
    private typealias Failure = ModelIOSceneLoader.Failure
    private typealias Warning = ModelIOSceneLoader.Warning

    let url: URL
    let fileExtension: String
    let asset: MDLAsset
    let options: ModelIOSceneLoader.Options
    private var warnings: [Warning] = []
    private var materials: ModelIOMaterialImporter
    private var nodeRecords: [(name: String, path: String, parent: Int?, transform: SceneTransform,
                               mesh: Int?, camera: Int?)] = []
    private var children: [[Int]] = []
    private var meshes: [GLTFScene.Mesh] = []
    private var cameras: [GLTFScene.Camera] = []
    private var channels: [GLTFAnimationClip.Channel] = []
    private var vertexBudget = 16_777_216
    private var conversion = matrix_identity_float4x4
    private var unitScale: Float = 1
    private var metersPerUnit: Double?
    /// Prim paths whose `doubleSided` is true in a text USD root layer.
    private var doubleSidedPaths: Set<String> = []

    init(url: URL, fileExtension: String, asset: MDLAsset, options: ModelIOSceneLoader.Options) {
        self.url = url
        self.fileExtension = fileExtension
        self.asset = asset
        self.options = options
        materials = ModelIOMaterialImporter(assetURL: url, isUSD: ["usd", "usda", "usdc", "usdz"].contains(fileExtension))
    }

    mutating func run() throws -> ModelIOSceneLoader.Asset {
        if ["usd", "usda", "usdc", "usdz"].contains(fileExtension) {
            switch USDLayerMetadata.rootLayer(of: url) {
            case .text(let layer):
                metersPerUnit = USDLayerMetadata.metersPerUnit(inUSDA: layer)
                doubleSidedPaths = USDLayerMetadata.doubleSidedPrims(inUSDA: layer)
            case .binary: warnings.append(.binaryRootLayer)
            case .unreadable: break
            }
        }
        if options.convertsToMeters, let metersPerUnit, metersPerUnit.isFinite, metersPerUnit > 0 {
            unitScale = Float(metersPerUnit)
        }
        var rotation = matrix_identity_float4x4
        if options.convertsToYUp, asset.upAxis == SIMD3(0, 0, 1) {
            rotation = simd_float4x4(simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0)))
        }
        conversion = rotation * simd_float4x4(diagonal: SIMD4(unitScale, unitScale, unitScale, 1))
        if options.includeMaterials, fileExtension == "obj" { materials.loadMaterialLibraries(forOBJ: url) }

        var roots: [Int] = []
        for index in 0..<asset.count {
            roots.append(try visit(asset.object(at: index), parent: nil))
        }
        guard !meshes.isEmpty || !cameras.isEmpty else { throw Failure.emptyAsset }
        warnings += materials.warnings

        let nodes = nodeRecords.enumerated().map { index, record in
            GLTFScene.Node(
                sourceIndex: index, name: record.name, parentIndex: record.parent, childIndices: children[index],
                meshIndex: record.mesh, skinIndex: nil, lightIndex: nil, cameraIndex: record.camera,
                localTransform: record.transform
            )
        }
        var animations: [GLTFAnimationClip] = []
        if !channels.isEmpty {
            animations.append(GLTFAnimationClip(sourceIndex: 0, name: nil, channels: channels))
        }
        let scene = GLTFScene(
            sourceIndex: 0, name: url.deletingPathExtension().lastPathComponent, rootNodeIndices: roots, nodes: nodes,
            meshes: meshes, materials: materials.materials, animations: animations, skins: [], lights: [], cameras: cameras
        )
        return ModelIOSceneLoader.Asset(
            scene: scene, warnings: warnings, sourceMetersPerUnit: metersPerUnit, sourceUpAxis: asset.upAxis,
            objectPaths: Dictionary(uniqueKeysWithValues: nodeRecords.enumerated().map { ($0.offset, $0.element.path) })
        )
    }

    private mutating func visit(_ object: MDLObject, parent: Int?) throws -> Int {
        let index = nodeRecords.count
        let path = object.path
        let transform = try localTransform(object, parent: parent, time: asset.startTime)
        nodeRecords.append((object.name, path, parent, transform, nil, nil))
        children.append([])
        if let parent { children[parent].append(index) }
        if object.transform?.resetsTransform == true { warnings.append(.resetsTransformIgnored(node: path)) }
        if options.includeAnimations { sampleAnimation(object, node: index, parent: parent) }

        if let mesh = object as? MDLMesh {
            if let imported = try importMesh(mesh, path: path) {
                nodeRecords[index].mesh = meshes.count
                meshes.append(imported)
            }
        } else if let camera = object as? MDLCamera, options.includeCameras {
            if let projection = Self.projection(camera, scale: unitScale) {
                nodeRecords[index].camera = cameras.count
                cameras.append(GLTFScene.Camera(sourceIndex: cameras.count, name: object.name, projection: projection))
            } else {
                warnings.append(.cameraSkipped(node: path))
            }
        } else if object is MDLLight {
            warnings.append(.lightIgnored(node: path))
        }
        for child in object.children.objects {
            _ = try visit(child, parent: index)
        }
        return index
    }

    private mutating func localTransform(_ object: MDLObject, parent: Int?, time: TimeInterval) throws -> SceneTransform {
        guard let component = object.transform else { return parent == nil ? try decompose(conversion, object) : SceneTransform() }
        var matrix = component.localTransform?(atTime: time) ?? component.matrix
        if matrix == simd_float4x4() {
            warnings.append(.degenerateTransformIgnored(node: object.path))
            matrix = matrix_identity_float4x4
        }
        if parent == nil { matrix = conversion * matrix }
        return try decompose(matrix, object)
    }

    private func decompose(_ matrix: simd_float4x4, _ object: MDLObject) throws -> SceneTransform {
        guard let transform = SceneTransform(decomposing: matrix) else { throw Failure.unsupportedTransform(object.path) }
        return transform
    }

    private mutating func sampleAnimation(_ object: MDLObject, node: Int, parent: Int?) {
        guard let component = object.transform, component.keyTimes.count > 1 else { return }
        let times = component.keyTimes.map(\.doubleValue)
        guard times.allSatisfy({ $0.isFinite && $0 >= 0 }), zip(times, times.dropFirst()).allSatisfy({ $0 < $1 }) else {
            warnings.append(.animationSkipped(node: object.path))
            return
        }
        var poses: [SceneTransform] = []
        for time in times {
            var matrix = component.localTransform?(atTime: time) ?? component.matrix
            if parent == nil { matrix = conversion * matrix }
            guard let pose = SceneTransform(decomposing: matrix) else {
                warnings.append(.animationSkipped(node: object.path))
                return
            }
            // Keep consecutive quaternions in one hemisphere so linear keys take the short arc.
            if let previous = poses.last, simd_dot(previous.rotation.vector, pose.rotation.vector) < 0 {
                poses.append(SceneTransform(translation: pose.translation, rotation: simd_quatf(vector: -pose.rotation.vector), scale: pose.scale))
            } else {
                poses.append(pose)
            }
        }
        let keys = times.map(Float.init)
        channels.append(.init(sourceNodeIndex: node, target: .translation, interpolation: .linear, times: keys,
                              values: poses.map { SIMD4($0.translation, 0) }))
        channels.append(.init(sourceNodeIndex: node, target: .rotation, interpolation: .linear, times: keys,
                              values: poses.map(\.rotation.vector)))
        channels.append(.init(sourceNodeIndex: node, target: .scale, interpolation: .linear, times: keys,
                              values: poses.map { SIMD4($0.scale, 0) }))
    }

    private static func projection(_ camera: MDLCamera, scale: Float) -> GLTFScene.Camera.Projection? {
        guard camera.projection == .perspective else { return nil }
        let fieldOfView = camera.fieldOfView * .pi / 180
        let near = camera.nearVisibilityDistance * scale, far = camera.farVisibilityDistance * scale
        let aspect = camera.sensorAspect
        guard fieldOfView.isFinite, fieldOfView > 0, fieldOfView < .pi, near.isFinite, near > 0,
              far.isFinite, far > near else { return nil }
        return .perspective(
            verticalFieldOfView: fieldOfView, aspectRatio: aspect.isFinite && aspect > 0 ? aspect : nil, near: near, far: far
        )
    }

    // MARK: Meshes

    private mutating func importMesh(_ mesh: MDLMesh, path: String) throws -> GLTFScene.Mesh? {
        func has(_ name: String) -> Bool {
            (mesh.vertexDescriptor.attributeNamed(name)).map { $0.format != .invalid } ?? false
        }
        guard has(MDLVertexAttributePosition), mesh.vertexCount > 0 else { return nil }
        if !has(MDLVertexAttributeNormal) {
            mesh.addNormals(withAttributeNamed: MDLVertexAttributeNormal, creaseThreshold: options.normalCreaseThreshold)
            warnings.append(.normalsGenerated(mesh: path))
        }
        let colors = fileExtension == "ply" ? Self.plyColors(in: mesh) : nil
        if colors == nil, has(MDLVertexAttributeColor) { warnings.append(.vertexColorsDropped(mesh: path)) }
        let count = mesh.vertexCount
        guard let positionData = mesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributePosition, as: .float3),
              let normalData = mesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeNormal, as: .float3) else {
            throw Failure.invalidGeometry(mesh: path, reason: "ModelIO did not provide positions and normals.")
        }
        func vectors3(_ data: MDLVertexAttributeData) -> [SIMD3<Float>] {
            (0..<count).map { index in
                let value = data.dataStart.advanced(by: index * data.stride).loadUnaligned(as: (Float, Float, Float).self)
                return SIMD3(value.0, value.1, value.2)
            }
        }
        func vectors2(_ name: String) -> [SIMD2<Float>]? {
            guard has(name), let data = mesh.vertexAttributeData(forAttributeNamed: name, as: .float2) else { return nil }
            return (0..<count).map { data.dataStart.advanced(by: $0 * data.stride).loadUnaligned(as: SIMD2<Float>.self) }
        }
        let positions = vectors3(positionData)
        let normals = vectors3(normalData)
        let primary = vectors2(MDLVertexAttributeTextureCoordinate)
        let secondary = vectors2(MDLVertexAttributeTextureCoordinate + "1")
        let isDoubleSided = doubleSidedPaths.contains(path)
        guard positions.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else {
            throw Failure.invalidGeometry(mesh: path, reason: "positions must be finite.")
        }

        var primitives: [MeshGeometry] = []
        var primitiveMaterials: [Int?] = []
        var repaired = 0
        for case let submesh as MDLSubmesh in mesh.submeshes ?? [] {
            var indices = Self.indices(submesh)
            switch submesh.geometryType {
            case .triangles, .quads, .variableTopology:
                let counts = Self.faceCounts(submesh, indexCount: indices.count)
                guard counts.reduce(0, +) == indices.count, counts.allSatisfy({ $0 >= 3 }) else {
                    throw Failure.invalidGeometry(mesh: path, reason: "the face sizes do not match the index count.")
                }
                guard indices.allSatisfy({ Int($0) < count }) else {
                    throw Failure.invalidGeometry(mesh: path, reason: "an index is outside the vertex array.")
                }
                if counts.contains(where: { $0 != 3 }) {
                    indices = PolygonTriangulation.triangulate(indices, faceCounts: counts, positions: positions)
                }
            case .triangleStrips:
                indices = Self.triangles(fromStrip: indices)
            default:
                warnings.append(.submeshSkipped(mesh: path, geometryType: submesh.geometryType.rawValue))
                continue
            }
            guard !indices.isEmpty, indices.count.isMultiple(of: 3) else { continue }
            guard indices.allSatisfy({ Int($0) < count }) else {
                throw Failure.invalidGeometry(mesh: path, reason: "an index is outside the vertex array.")
            }
            // Compact to the vertices this submesh uses, in first-use order.
            var remap: [UInt32: UInt32] = [:]
            var vertices: [MeshVertex] = []
            var faceNormals: [UInt32: SIMD3<Float>] = [:]
            for triangle in stride(from: 0, to: indices.count, by: 3) {
                let a = positions[Int(indices[triangle])], b = positions[Int(indices[triangle + 1])]
                let c = positions[Int(indices[triangle + 2])]
                let cross = simd_cross(b - a, c - a)
                let face = simd_length(cross) > 0 ? simd_normalize(cross) : SIMD3<Float>(0, 1, 0)
                for corner in 0..<3 where faceNormals[indices[triangle + corner]] == nil {
                    faceNormals[indices[triangle + corner]] = face
                }
            }
            var local: [UInt32] = []
            local.reserveCapacity(indices.count)
            for source in indices {
                if let mapped = remap[source] {
                    local.append(mapped)
                    continue
                }
                let index = Int(source)
                var normal = normals[index]
                let length = simd_length(normal)
                if !(length.isFinite && length > 0) {
                    normal = faceNormals[source] ?? SIMD3(0, 1, 0)
                    repaired += 1
                } else {
                    normal /= length
                }
                func flipped(_ uv: SIMD2<Float>) -> SIMD2<Float> { SIMD2(uv.x, 1 - uv.y) }
                guard vertexBudget > 0 else { throw Failure.inputTooLarge }
                vertexBudget -= 1
                let mapped = UInt32(vertices.count)
                vertices.append(MeshVertex(
                    position: positions[index], normal: normal,
                    textureCoordinate: primary.map { flipped($0[index]) } ?? .zero,
                    secondaryTextureCoordinate: secondary.map { flipped($0[index]) },
                    color: colors?[index]
                ))
                remap[source] = mapped
                local.append(mapped)
            }
            do {
                primitives.append(try MeshGeometry(vertices: vertices, indices: local))
            } catch {
                throw Failure.invalidGeometry(mesh: path, reason: error.localizedDescription)
            }
            primitiveMaterials.append(options.includeMaterials
                ? submesh.material.map { materials.index(for: $0, doubleSided: isDoubleSided) } : nil)
        }
        if repaired > 0 { warnings.append(.normalsRepaired(mesh: path, count: repaired)) }
        guard !primitives.isEmpty else { return nil }
        return GLTFScene.Mesh(
            sourceIndex: meshes.count, name: mesh.name, primitives: primitives,
            primitiveMaterialIndices: primitiveMaterials,
            primitiveSkinInfluences: Array(repeating: nil, count: primitives.count), morphTargets: nil
        )
    }

    private static func plyColors(in mesh: MDLMesh) -> [SIMD4<Float>]? {
        let name = MDLVertexAttributeColor
        let bytes = 3 * MemoryLayout<Float>.size
        guard let attribute = mesh.vertexDescriptor.attributeNamed(name),
              let data = mesh.vertexAttributeData(forAttributeNamed: name),
              data.format == .float3, data.stride >= bytes,
              data.bufferSize >= bytes, attribute.offset <= data.bufferSize - bytes,
              mesh.vertexCount - 1 <= (data.bufferSize - bytes - attribute.offset) / data.stride else {
            return nil
        }
        // Model I/O exposes PLY RGB as normalized float3. Requesting float4
        // can misalign this interleaved layout, and PLY alpha is not exposed.
        var colors: [SIMD4<Float>] = []
        colors.reserveCapacity(mesh.vertexCount)
        for index in 0..<mesh.vertexCount {
            let value = data.dataStart.advanced(by: index * data.stride).loadUnaligned(as: (Float, Float, Float).self)
            guard (0...1).contains(value.0), (0...1).contains(value.1), (0...1).contains(value.2) else { return nil }
            colors.append(SIMD4(
                GLTFColorTables.sRGBToLinear(value.0), GLTFColorTables.sRGBToLinear(value.1),
                GLTFColorTables.sRGBToLinear(value.2), 1
            ))
        }
        return colors
    }

    private static func indices(_ submesh: MDLSubmesh) -> [UInt32] {
        let buffer = submesh.indexBuffer(asIndexType: .uInt32)
        let map = buffer.map()
        return Array(UnsafeBufferPointer(start: map.bytes.assumingMemoryBound(to: UInt32.self), count: submesh.indexCount))
    }

    /// Vertex counts per face: the submesh topology when ModelIO preserved it, else uniform faces.
    private static func faceCounts(_ submesh: MDLSubmesh, indexCount: Int) -> [Int] {
        if let topology = submesh.topology, let buffer = topology.faceTopology, topology.faceCount > 0,
           buffer.length >= topology.faceCount {
            let bytes = buffer.map().bytes.assumingMemoryBound(to: UInt8.self)
            return (0..<topology.faceCount).map { Int(bytes[$0]) }
        }
        let size = submesh.geometryType == .quads ? 4 : 3
        return indexCount.isMultiple(of: size) ? Array(repeating: size, count: indexCount / size) : [indexCount]
    }

    private static func triangles(fromStrip strip: [UInt32]) -> [UInt32] {
        guard strip.count >= 3 else { return [] }
        var result: [UInt32] = []
        for index in 0..<(strip.count - 2) {
            let (a, b, c) = (strip[index], strip[index + 1], strip[index + 2])
            guard a != b, b != c, a != c else { continue }
            result += index.isMultiple(of: 2) ? [a, b, c] : [b, a, c]
        }
        return result
    }
}

extension SceneTransform {
    /// Splits an affine matrix into translation, rotation, and scale. A mirrored matrix gets a
    /// negative X scale. Nil for a sheared, projective, singular, or non-finite matrix.
    init?(decomposing matrix: simd_float4x4) {
        let (c0, c1, c2, c3) = matrix.columns
        guard c0.w == 0, c1.w == 0, c2.w == 0, c3.w == 1 else { return nil }
        func double(_ column: SIMD4<Float>) -> SIMD3<Double> { SIMD3(Double(column.x), Double(column.y), Double(column.z)) }
        let x = double(c0), y = double(c1), z = double(c2)
        let lengths = SIMD3(simd_length(x), simd_length(y), simd_length(z))
        guard lengths.x > 0, lengths.y > 0, lengths.z > 0, lengths.x.isFinite, lengths.y.isFinite, lengths.z.isFinite,
              c3.x.isFinite, c3.y.isFinite, c3.z.isFinite else { return nil }
        var axisX = x / lengths.x
        let axisY = y / lengths.y, axisZ = z / lengths.z
        let tolerance = 0.0001
        let determinant = simd_dot(simd_cross(axisX, axisY), axisZ)
        guard abs(simd_dot(axisX, axisY)) <= tolerance, abs(simd_dot(axisX, axisZ)) <= tolerance,
              abs(simd_dot(axisY, axisZ)) <= tolerance, abs(abs(determinant) - 1) <= tolerance else { return nil }
        var scaleX = lengths.x
        if determinant < 0 {
            axisX = -axisX
            scaleX = -scaleX
        }
        func float(_ value: SIMD3<Double>) -> SIMD3<Float> { SIMD3(Float(value.x), Float(value.y), Float(value.z)) }
        let basis = simd_float3x3(columns: (float(axisX), float(axisY), float(axisZ)))
        self.init(
            translation: SIMD3(c3.x, c3.y, c3.z), rotation: simd_normalize(simd_quatf(basis)),
            scale: SIMD3(Float(scaleX), Float(lengths.y), Float(lengths.z))
        )
    }
}

/// Reads opinions ModelIO does not report from text USD root layers, without a USD runtime.
/// Composition arcs are not followed, so only what the root layer itself authors is found.
package enum USDLayerMetadata {
    enum RootLayer {
        /// The USDA text of the root layer: the file itself, or a package's first file.
        case text(Data)
        /// The root layer is a binary crate file.
        case binary
        case unreadable
    }

    /// The root layer of a USD file or package. For a package, the first file is the root layer.
    static func rootLayer(of url: URL) -> RootLayer {
        guard var data = try? Data(contentsOf: url, options: .mappedIfSafe) else { return .unreadable }
        if data.starts(with: [0x50, 0x4B, 0x03, 0x04]) {
            guard let first = try? USDZArchive.records(in: data).first, first.method == 0 else { return .unreadable }
            let path = first.path.lowercased()
            guard path.hasSuffix(".usda") || path.hasSuffix(".usd") else {
                return path.hasSuffix(".usdc") ? .binary : .unreadable
            }
            guard let layer = try? USDZArchive.file(first, in: data) else { return .unreadable }
            data = layer
        }
        if data.starts(with: Array("PXR-USDC".utf8)) { return .binary }
        return data.starts(with: Array("#usda".utf8)) ? .text(data) : .unreadable
    }

    /// The layer's `metersPerUnit`, which layer metadata declares before the first prim.
    package static func metersPerUnit(inUSDA layer: Data) -> Double? {
        var header = ""
        for line in String(decoding: layer.prefix(65_536), as: UTF8.self).split(separator: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if ["def ", "over ", "class "].contains(where: { trimmed.hasPrefix($0) }) { break }
            header += trimmed + "\n"
        }
        guard let range = header.range(of: #"metersPerUnit\s*=\s*([0-9.eE+-]+)"#, options: .regularExpression) else {
            return nil
        }
        return header[range].split(separator: "=").last.flatMap { Double($0.trimmingCharacters(in: .whitespaces)) }
    }

    /// The paths of prims whose `doubleSided` attribute the layer sets to true. Prims inside
    /// variant blocks get the path they compose to when their variant is selected.
    package static func doubleSidedPrims(inUSDA layer: Data) -> Set<String> {
        layer.withUnsafeBytes { raw in
            var scanner = USDATextScanner(bytes: raw.bindMemory(to: UInt8.self))
            return scanner.doubleSidedPrims()
        }
    }
}

/// Tokenizes USDA text just enough to track prim scopes. Strings, asset paths, path
/// references, and comments are single tokens, and a brace opens a scope only outside
/// metadata parentheses and array brackets.
private struct USDATextScanner {
    let bytes: UnsafeBufferPointer<UInt8>
    private var offset = 0

    init(bytes: UnsafeBufferPointer<UInt8>) {
        self.bytes = bytes
    }

    private enum Token {
        case word(Range<Int>)
        /// A quoted string's contents.
        case string(Range<Int>)
        case symbol(UInt8)
    }

    private enum Specifier {
        case none
        /// After `def`, `over`, or `class`, before the prim name.
        case started
        /// After the prim name, before its body.
        case named(String)
    }

    mutating func doubleSidedPrims() -> Set<String> {
        var result: Set<String> = []
        // One entry per open brace outside parentheses and brackets: a prim name, or nil for
        // a dictionary, time-sample, or variant block.
        var scopes: [String?] = []
        var parentheses = 0, brackets = 0
        var specifier = Specifier.none
        // 1 after a prim body's `doubleSided`, 2 after its `=`.
        var assignment = 0
        while let token = next() {
            if case .symbol(let symbol) = token, parentheses > 0 || brackets > 0 || symbol == UInt8(ascii: "(") || symbol == UInt8(ascii: "[") {
                switch symbol {
                case UInt8(ascii: "("): parentheses += 1
                case UInt8(ascii: ")"): parentheses = max(0, parentheses - 1)
                case UInt8(ascii: "["): brackets += 1
                case UInt8(ascii: "]"): brackets = max(0, brackets - 1)
                default: break
                }
                // Metadata parentheses sit between a prim's name and its body.
                if case .named = specifier {} else { specifier = .none }
                assignment = 0
                continue
            }
            guard parentheses == 0, brackets == 0 else { continue }

            switch token {
            case .symbol(UInt8(ascii: "{")):
                if case .named(let name) = specifier { scopes.append(name) } else { scopes.append(nil) }
                specifier = .none
            case .symbol(UInt8(ascii: "}")):
                _ = scopes.popLast()
                specifier = .none
            case .word(let range) where matches(range, "def") || matches(range, "over") || matches(range, "class"):
                specifier = .started
            case .word:
                if case .started = specifier {} else { specifier = .none } // a type name keeps `.started`
            case .string(let range):
                if case .started = specifier { specifier = .named(text(range)) } else { specifier = .none }
            default:
                specifier = .none
            }

            let inPrimBody = (scopes.last ?? nil) != nil
            switch token {
            case .word(let range) where inPrimBody && matches(range, "doubleSided"):
                assignment = 1
            case .symbol(UInt8(ascii: "=")) where assignment == 1:
                assignment = 2
            case .word(let range) where assignment == 2:
                let path = "/" + scopes.compactMap { $0 }.joined(separator: "/")
                if matches(range, "1") || matches(range, "true") { result.insert(path) } else { result.remove(path) }
                assignment = 0
            default:
                assignment = 0
            }
        }
        return result
    }

    private func matches(_ range: Range<Int>, _ literal: String) -> Bool {
        range.count == literal.utf8.count && zip(range, literal.utf8).allSatisfy { bytes[$0] == $1 }
    }

    private func text(_ range: Range<Int>) -> String {
        String(decoding: UnsafeBufferPointer(rebasing: bytes[range]), as: UTF8.self)
    }

    private func byte(_ index: Int) -> UInt8? {
        index < bytes.count ? bytes[index] : nil
    }

    private mutating func next() -> Token? {
        while let current = byte(offset) {
            switch current {
            case UInt8(ascii: " "), UInt8(ascii: "\t"), UInt8(ascii: "\n"), UInt8(ascii: "\r"):
                offset += 1
            case UInt8(ascii: "#"):
                skipLine()
            case UInt8(ascii: "/") where byte(offset + 1) == UInt8(ascii: "/"):
                skipLine()
            case UInt8(ascii: "/") where byte(offset + 1) == UInt8(ascii: "*"):
                offset += 2
                while let value = byte(offset), !(value == UInt8(ascii: "*") && byte(offset + 1) == UInt8(ascii: "/")) { offset += 1 }
                offset = min(offset + 2, bytes.count)
            case UInt8(ascii: "\""), UInt8(ascii: "'"):
                return .string(delimited(by: current, escapes: true))
            case UInt8(ascii: "@"):
                _ = delimited(by: current, escapes: false)
                return .symbol(current)
            case UInt8(ascii: "<"):
                while let value = byte(offset), value != UInt8(ascii: ">") { offset += 1 }
                offset = min(offset + 1, bytes.count)
                return .symbol(current)
            default:
                guard Self.isWordByte(current) else {
                    offset += 1
                    return .symbol(current)
                }
                let start = offset
                while let value = byte(offset), Self.isWordByte(value) { offset += 1 }
                return .word(start..<offset)
            }
        }
        return nil
    }

    private mutating func skipLine() {
        while let value = byte(offset), value != UInt8(ascii: "\n") { offset += 1 }
    }

    private static func isWordByte(_ byte: UInt8) -> Bool {
        switch byte {
        case UInt8(ascii: "a")...UInt8(ascii: "z"), UInt8(ascii: "A")...UInt8(ascii: "Z"), UInt8(ascii: "0")...UInt8(ascii: "9"),
             UInt8(ascii: "_"), UInt8(ascii: ":"), UInt8(ascii: "."), UInt8(ascii: "-"), UInt8(ascii: "+"), 0x80...0xFF:
            true
        default: false
        }
    }

    /// Skips a single or tripled delimiter pair, such as `"name"`, `"""text"""`, `@a.usd@`,
    /// or `@@@a.usd@@@`, and returns the range between the delimiters.
    private mutating func delimited(by mark: UInt8, escapes: Bool) -> Range<Int> {
        func isTriple(at index: Int) -> Bool { byte(index) == mark && byte(index + 1) == mark && byte(index + 2) == mark }
        let triple = isTriple(at: offset)
        offset += triple ? 3 : 1
        let start = offset
        while let value = byte(offset) {
            if escapes, value == UInt8(ascii: "\\") {
                offset += 2
            } else if value == mark, !triple || isTriple(at: offset) {
                let end = offset
                offset += triple ? 3 : 1
                return start..<end
            } else {
                offset += 1
            }
        }
        offset = bytes.count
        return start..<bytes.count
    }
}

/// Splits simple polygons into triangles that keep their winding.
package enum PolygonTriangulation {
    /// Ear-clips each face in the plane of its Newell normal, so concave faces stay inside
    /// their outline. Faces that are degenerate, self-intersecting, or larger than 1,024
    /// corners fall back to a fan from their first corner. The search resumes beside each
    /// clipped ear and tests only non-convex corners for containment, so a face costs about
    /// O(n · r) for r reflex corners instead of restarting its scan after every ear.
    package static func triangulate(_ indices: [UInt32], faceCounts: [Int], positions: [SIMD3<Float>]) -> [UInt32] {
        var result: [UInt32] = []
        result.reserveCapacity(indices.count * 3)
        var start = 0
        for count in faceCounts {
            let face = Array(indices[start..<(start + count)])
            start += count
            result += triangulate(face, positions: positions)
        }
        return result
    }

    package static func triangulate(_ face: [UInt32], positions: [SIMD3<Float>]) -> [UInt32] {
        let count = face.count
        guard count > 3 else { return count == 3 ? face : [] }
        func fan() -> [UInt32] { (1..<(count - 1)).flatMap { [face[0], face[$0], face[$0 + 1]] } }
        guard count <= 1_024 else { return fan() }
        let points = face.map { SIMD3<Double>(positions[Int($0)]) }
        var normal = SIMD3<Double>.zero
        for index in 0..<count {
            let a = points[index], b = points[(index + 1) % count]
            normal += SIMD3((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y))
        }
        let length = simd_length(normal)
        guard length > 0, length.isFinite else { return fan() }
        normal /= length
        let helper: SIMD3<Double> = abs(normal.x) < 0.9 ? SIMD3(1, 0, 0) : SIMD3(0, 1, 0)
        let u = simd_normalize(simd_cross(helper, normal)), v = simd_cross(normal, u)
        // Counterclockwise about the normal in (u, v).
        let flat = points.map { SIMD2(simd_dot($0, u), simd_dot($0, v)) }
        let extent = flat.reduce(0.0) { max($0, abs($1.x), abs($1.y)) }
        let epsilon = 1e-12 * max(extent * extent, 1e-300)
        func cross(_ a: Int, _ b: Int, _ c: Int) -> Double {
            let ab = flat[b] - flat[a], ac = flat[c] - flat[a]
            return ab.x * ac.y - ab.y * ac.x
        }
        // A doubly linked ring of the corners not yet clipped.
        var previous = (0..<count).map { ($0 + count - 1) % count }
        var next = (0..<count).map { ($0 + 1) % count }
        func isConvex(_ corner: Int) -> Bool { cross(previous[corner], corner, next[corner]) > epsilon }
        // Only corners that are not strictly convex can lie inside an ear of a simple polygon,
        // so only they are tested for containment (O'Rourke, Computational Geometry in C, 1.6).
        var isCandidate = (0..<count).map { !isConvex($0) }
        var candidates = (0..<count).filter { isCandidate[$0] }
        var triangles: [UInt32] = []
        triangles.reserveCapacity((count - 2) * 3)
        var remaining = count
        var corner = 0
        // Corners examined since the last clip; a full lap without an ear means the face is not simple.
        var examined = 0
        while remaining > 3 {
            guard examined < remaining else { return fan() }
            let before = previous[corner], after = next[corner]
            var isEar = isConvex(corner)
            if isEar {
                for other in candidates where isCandidate[other] && other != before && other != corner && other != after {
                    guard flat[other] != flat[before], flat[other] != flat[corner], flat[other] != flat[after] else { continue }
                    if cross(before, corner, other) >= -epsilon && cross(corner, after, other) >= -epsilon
                        && cross(after, before, other) >= -epsilon {
                        isEar = false
                        break
                    }
                }
            }
            guard isEar else {
                corner = after
                examined += 1
                continue
            }
            triangles += [face[before], face[corner], face[after]]
            next[before] = after
            previous[after] = before
            isCandidate[corner] = false
            remaining -= 1
            // Clipping usually makes a reflex neighbor convex; a degenerate one may turn non-convex.
            for neighbor in [before, after] {
                let convex = isConvex(neighbor)
                if isCandidate[neighbor], convex {
                    isCandidate[neighbor] = false
                } else if !isCandidate[neighbor], !convex {
                    isCandidate[neighbor] = true
                    candidates.append(neighbor)
                }
            }
            if candidates.count > 2 * remaining { candidates.removeAll { !isCandidate[$0] } }
            // Resume at the previous corner, whose angle just changed, instead of restarting the lap.
            corner = before
            examined = 0
        }
        return triangles + [face[previous[corner]], face[corner], face[next[corner]]]
    }
}
