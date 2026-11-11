import Foundation
import simd

/// A selected glTF scene with its node hierarchy and CPU-side triangle geometry.
/// Node and mesh arrays are sorted by source index. The index fields refer to the
/// original glTF arrays, so use them as keys rather than offsets into these arrays.
public struct GLTFScene {
    public let sourceIndex: Int
    public let name: String?
    public let rootNodeIndices: [Int]
    public let nodes: [Node]
    public internal(set) var meshes: [Mesh]
    /// Used material records, sorted by source index, including materials that only
    /// a material variant selects. Empty when materials are not requested.
    public let materials: [Material]
    /// Clips with channels targeting this scene. Empty when animations are not requested.
    public let animations: [GLTFAnimationClip]
    /// Used skin records, sorted by source index.
    public let skins: [Skin]
    /// Used punctual light definitions, sorted by source index.
    public let lights: [Light]
    /// Camera definitions used by the selected scene, sorted by source index.
    public let cameras: [Camera]
    /// KHR_materials_variants names, indexed by variant. Empty when materials are not requested.
    public internal(set) var materialVariants: [String] = []
    /// The variant chosen with `selectingMaterialVariant(_:)`, or nil for the default materials.
    public internal(set) var selectedMaterialVariant: Int?

    public struct Node {
        public let sourceIndex: Int
        public let name: String?
        public let parentIndex: Int?
        public let childIndices: [Int]
        public let meshIndex: Int?
        public let skinIndex: Int?
        public let lightIndex: Int?
        public let cameraIndex: Int?
        public let localTransform: SceneTransform
        /// The node's initial morph target weights: its own `weights`, else its
        /// mesh's defaults. Nil when its mesh has no imported morph targets.
        public internal(set) var morphWeights: [Float]?
        /// Per-instance transforms from `EXT_mesh_gpu_instancing`, in the node's local
        /// space: an instance's world transform is the node's world transform times
        /// the instance transform. A node with instances draws its mesh once per
        /// instance and not at the node itself. Nil when the node has no instancing
        /// or the scene was loaded without `includeGPUInstancing`.
        public internal(set) var instances: [SceneTransform]? = nil
    }

    public struct Mesh {
        public let sourceIndex: Int
        public let name: String?
        /// In the original primitive order.
        public let primitives: [MeshGeometry]
        /// Aligned with primitives. Nil selects glTF's default material. Reflects the
        /// scene's selected material variant.
        public internal(set) var primitiveMaterialIndices: [Int?]
        /// Aligned with primitives: the material source index each variant index selects.
        /// Empty maps, or an empty array without loaded materials, mean no variant mappings.
        public internal(set) var primitiveMaterialVariants: [[Int: Int]] = []
        /// The authored materials, kept once a variant has been selected.
        var defaultPrimitiveMaterialIndices: [Int?]?
        /// Aligned with primitives and each imported primitive's vertex order.
        /// Nil means the primitive is unused by skinned nodes in this scene.
        public let primitiveSkinInfluences: [[VertexSkinInfluence]?]
        /// Present when the scene was loaded with morph targets and this mesh has them.
        public let morphTargets: MorphTargets?

        /// A mesh's morph target offsets and default weights. Apply them on the CPU
        /// with `MeshMorphTargets.apply(weights:to:)`.
        public struct MorphTargets {
            /// Aligned with the mesh's primitives and their imported vertex order.
            /// Every primitive has the same number of targets, in the same order.
            public let primitives: [MeshMorphTargets]
            /// The mesh's `weights`, or zeros when the mesh omits them.
            public let defaultWeights: [Float]
            /// Names from the common `extras.targetNames` convention, when present
            /// with one string per target.
            public let names: [String]?
        }
    }
}

struct GLTFHierarchy: Decodable {
    private typealias Failure = GLTFMeshLoader.Failure
    static let gpuInstancingExtension = "EXT_mesh_gpu_instancing"

    let scene: Int?
    let scenes: [SceneRecord]?
    let nodes: [NodeRecord]?

    struct SceneRecord: Decodable {
        let name: String?
        let nodes: [Int]?
    }

    struct NodeRecord: Decodable {
        let name: String?
        let children: [Int]?
        let mesh: Int?
        let skin: Int?
        let weights: [Float]?
        let matrix: [Float]?
        let translation: [Float]?
        let rotation: [Float]?
        let scale: [Float]?
        let extensions: Extensions?
    }

    struct Extensions: Decodable {
        let gpuInstancing: GPUInstancing?

        struct GPUInstancing: Decodable {
            let attributes: [String: Int]?
        }

        private enum CodingKeys: String, CodingKey {
            case gpuInstancing = "EXT_mesh_gpu_instancing"
        }
    }

    /// With morph targets, nodes carry their raw `weights` for the loader to validate.
    /// `instancing` maps instanced node indices to their attribute accessors.
    func resolve(
        sceneIndex requested: Int?, includeSkins: Bool, includeMorphTargets: Bool, includeGPUInstancing: Bool,
        lightIndices: [Int: Int], cameraIndices: [Int: Int]
    ) throws -> (index: Int, name: String?, roots: [Int], nodes: [GLTFScene.Node], instancing: [Int: [String: Int]]) {
        let index = requested ?? scene ?? 0
        guard let scenes, scenes.indices.contains(index) else { throw Failure.missingScene(index) }
        let records = nodes ?? []
        guard records.count <= 1_000_000 else { throw Failure.sceneSizeLimit }
        let parents = try parentIndices(in: records)
        try checkCycles(parents)
        for scene in scenes {
            var seen = Set<Int>()
            for root in scene.nodes ?? [] {
                guard records.indices.contains(root) else { throw Failure.missingNode(root) }
                guard seen.insert(root).inserted else { throw Failure.duplicateNodeReference(root) }
                guard parents[root] == nil else { throw Failure.sceneRootHasParent(root) }
            }
        }

        let selected = scenes[index]
        let roots = selected.nodes ?? []
        var pending = roots
        var imported: [GLTFScene.Node] = []
        var instancing: [Int: [String: Int]] = [:]
        while let nodeIndex = pending.popLast() {
            let node = records[nodeIndex]
            if let skin = node.skin {
                guard includeSkins else { throw Failure.unsupportedFeature("skin on node \(nodeIndex)") }
                guard node.mesh != nil else { throw Failure.invalidSkin(skin, "node \(nodeIndex) has no mesh") }
            }
            let weights = node.weights.flatMap { $0.isEmpty ? nil : $0 }
            if weights != nil, !includeMorphTargets {
                throw Failure.unsupportedFeature("morph weights on node \(nodeIndex)")
            }
            if let instances = node.extensions?.gpuInstancing {
                guard includeGPUInstancing else {
                    throw Failure.unsupportedFeature("EXT_mesh_gpu_instancing on node \(nodeIndex)")
                }
                guard node.mesh != nil else { throw Failure.invalidInstancing(nodeIndex, "instancing needs a node with a mesh") }
                guard node.skin == nil else {
                    throw Failure.unsupportedFeature("EXT_mesh_gpu_instancing on skinned node \(nodeIndex)")
                }
                instancing[nodeIndex] = instances.attributes ?? [:]
            }
            let children = node.children ?? []
            imported.append(GLTFScene.Node(
                sourceIndex: nodeIndex, name: node.name, parentIndex: parents[nodeIndex],
                childIndices: children, meshIndex: node.mesh, skinIndex: node.skin, lightIndex: lightIndices[nodeIndex],
                cameraIndex: cameraIndices[nodeIndex], localTransform: try transform(node, at: nodeIndex),
                morphWeights: weights
            ))
            pending.append(contentsOf: children)
        }

        return (index, selected.name, roots, imported.sorted { $0.sourceIndex < $1.sourceIndex }, instancing)
    }

    private func parentIndices(in records: [NodeRecord]) throws -> [Int?] {
        var parents = [Int?](repeating: nil, count: records.count)
        for (index, node) in records.enumerated() {
            for child in node.children ?? [] {
                guard records.indices.contains(child) else { throw Failure.missingNode(child) }
                if let parent = parents[child] {
                    throw parent == index ? Failure.duplicateNodeReference(child) : Failure.multipleParents(child)
                }
                parents[child] = index
            }
        }
        return parents
    }

    private func checkCycles(_ parents: [Int?]) throws {
        // Follow parent links without recursion, including nodes outside the selected scene.
        enum Visit { case unseen, visiting, finished }
        var visits = [Visit](repeating: .unseen, count: parents.count)
        var path: [Int] = []
        for index in parents.indices where visits[index] == .unseen {
            var current: Int? = index
            while let node = current, visits[node] == .unseen {
                visits[node] = .visiting
                path.append(node)
                current = parents[node]
            }
            if let node = current, visits[node] == .visiting { throw Failure.nodeCycle(node) }
            for node in path { visits[node] = .finished }
            path.removeAll(keepingCapacity: true)
        }
    }

    private func transform(_ node: NodeRecord, at index: Int) throws -> SceneTransform {
        let result: SceneTransform
        if let matrix = node.matrix {
            guard node.translation == nil, node.rotation == nil, node.scale == nil else {
                throw Failure.invalidNodeTransform(index, "matrix and TRS cannot be combined")
            }
            result = try decompose(matrix, at: index)
        } else {
            let translation = try vector(node.translation, fallback: .zero, at: index)
            let scale = try vector(node.scale, fallback: SIMD3(repeating: 1), at: index)
            var rotation = simd_quatf(real: 1, imag: .zero)
            if let values = node.rotation {
                guard values.count == 4, values.allSatisfy(\.isFinite) else {
                    throw Failure.invalidNodeTransform(index, "rotation needs four finite XYZW components")
                }
                let vector = SIMD4(values[0], values[1], values[2], values[3])
                guard abs(simd_length_squared(vector) - 1) <= 0.0001 else {
                    throw Failure.invalidNodeTransform(index, "rotation must be a unit quaternion")
                }
                rotation = simd_quatf(vector: simd_normalize(vector))
            }
            result = SceneTransform(translation: translation, rotation: rotation, scale: scale)
        }

        let matrix = result.matrix
        let determinant = simd_determinant(matrix)
        guard finite(matrix), determinant.isFinite, determinant != 0, finite(simd_inverse(matrix)) else {
            throw Failure.invalidNodeTransform(index, "transform must be finite and invertible")
        }
        return result
    }

    private func decompose(_ values: [Float], at index: Int) throws -> SceneTransform {
        guard values.count == 16, values.allSatisfy(\.isFinite),
              values[3] == 0, values[7] == 0, values[11] == 0, values[15] == 1 else {
            throw Failure.invalidNodeTransform(index, "matrix must be finite, column-major, and affine")
        }
        let x = SIMD3(Double(values[0]), Double(values[1]), Double(values[2]))
        let y = SIMD3(Double(values[4]), Double(values[5]), Double(values[6]))
        let z = SIMD3(Double(values[8]), Double(values[9]), Double(values[10]))
        let lengths = SIMD3(simd_length(x), simd_length(y), simd_length(z))
        guard lengths.x > 0, lengths.y > 0, lengths.z > 0 else {
            throw Failure.invalidNodeTransform(index, "matrix scale must be nonzero")
        }

        var axisX = x / lengths.x
        let axisY = y / lengths.y
        let axisZ = z / lengths.z
        let determinant = simd_dot(simd_cross(axisX, axisY), axisZ)
        let tolerance = 0.0001
        guard abs(simd_dot(axisX, axisY)) <= tolerance,
              abs(simd_dot(axisX, axisZ)) <= tolerance,
              abs(simd_dot(axisY, axisZ)) <= tolerance,
              abs(abs(determinant) - 1) <= tolerance else {
            throw Failure.invalidNodeTransform(index, "matrix cannot be decomposed into TRS without shear")
        }

        var scaleX = lengths.x
        if determinant < 0 {
            axisX = -axisX
            scaleX = -scaleX
        }
        let basis = simd_float3x3(columns: (floatVector(axisX), floatVector(axisY), floatVector(axisZ)))
        return SceneTransform(
            translation: SIMD3(values[12], values[13], values[14]),
            rotation: simd_normalize(simd_quatf(basis)),
            scale: SIMD3(Float(scaleX), Float(lengths.y), Float(lengths.z))
        )
    }

    private func vector(_ values: [Float]?, fallback: SIMD3<Float>, at index: Int) throws -> SIMD3<Float> {
        guard let values else { return fallback }
        guard values.count == 3, values.allSatisfy(\.isFinite) else {
            throw Failure.invalidNodeTransform(index, "translation and scale need three finite components")
        }
        return SIMD3(values[0], values[1], values[2])
    }

    private func floatVector(_ value: SIMD3<Double>) -> SIMD3<Float> {
        SIMD3(Float(value.x), Float(value.y), Float(value.z))
    }

    private func finite(_ matrix: simd_float4x4) -> Bool {
        let (x, y, z, w) = matrix.columns
        return [x, y, z, w].allSatisfy {
            $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite
        }
    }
}
