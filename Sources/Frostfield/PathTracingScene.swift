import Foundation
import Metal
import simd

/// A two-level acceleration structure and material snapshot for the path tracer.
///
/// Preparation builds one bottom level per distinct geometry in object space: every
/// unskinned node that uses a renderer shares that renderer's level, while each
/// skinned node, and each node given replacement geometry, owns one. A top level
/// over the node instances holds their world transforms. With Metal ray tracing, the
/// levels are compacted primitive acceleration structures built on the GPU under one
/// instance acceleration structure; the software fallback builds binned-SAH BVHs on the
/// CPU that the kernel traverses itself. `updateTransforms(_:)` moves instances by
/// refitting the top level, and `updateGeometry(for:in:replacements:)` rebuilds chosen
/// bottom levels. Neither re-bakes textures nor recompiles shaders, and each update that
/// changes the traced scene increments `revision`, which makes `PathTracer` reject older
/// accumulation history.
///
/// Updates replace the affected Metal buffers and acceleration structures instead of
/// writing into them, so command buffers that are already encoded keep reading the
/// previous state. The scene is not thread-safe: update it on the thread that encodes
/// with it. `prepare(queue:graph:options:progress:)` builds the levels off the caller.
public final class PathTracingScene {
    /// The most triangles stored across bottom levels. Instances of a shared level count once.
    public static let maximumTriangleCount = 16_777_216
    /// The most top-level instances: mesh nodes, or one per entry of a node's `InstanceSet`.
    public static let maximumInstanceCount = 4_194_304
    public static let maximumMaterialCount = 65_536
    /// The most texture table entries: distinct pairs of a texture and its sampling description.
    public static let maximumTextureCount = 4_096
    public static let maximumBVHDepth = 32
    public static let maximumSurfaceGraphCount = 16
    /// Distinct live uniform blocks share 256 float4 slots, copied into each dispatch.
    /// The first ten hold the camera that surface graphs read, and each block takes one
    /// slot per declared uniform plus one for its graph clock.
    public static let maximumSurfaceUniformSlots = 256

    /// How preparation stores material textures.
    public struct Options: Equatable, Sendable {
        /// Where traced rays read material maps.
        public enum Textures: Equatable, Sendable {
            /// Samples each map's own texture through a bindless table of texture handles and sampler
            /// states: its full resolution, its mip chain and its pixel format, sRGB and
            /// block-compressed formats included. The sampler state comes from the binding's
            /// `TextureSampling`, with addressing per axis, magnification, minification and mip
            /// filters, and anisotropic filtering when minification is linear. Ray cones choose
            /// the level and anisotropy at every hit. A map bakes into a `fallbackSize`² layer, as
            /// `baked(size:)` does, only when the table cannot hold it: on a GPU without tier 2
            /// argument buffers, or for a 32-bit float texture that its sampler or a surface
            /// graph filters linearly on a GPU that cannot filter 32-bit floats. `fallbackSize`
            /// is from 16 to 512.
            case fullResolution(fallbackSize: Int = 256)
            /// Bakes every map into one RGBA16Float array of `size`² layers with a full mip chain,
            /// from 16 to 512 texels, the behavior before full-resolution sampling.
            case baked(size: Int)
        }

        /// The structure that rays traverse.
        public enum Acceleration: Hashable, Sendable {
            /// Metal ray tracing when the device reports `supportsRaytracing`, otherwise the software BVH.
            case automatic
            /// Primitive and instance `MTLAccelerationStructure`s built and compacted on the GPU and
            /// traversed by Metal's intersector. Preparation throws on devices without ray tracing.
            case metal
            /// Binned-SAH BVHs built on the CPU and traversed by the kernel, on any Metal device.
            case software
        }

        public var textures: Textures
        public var acceleration: Acceleration
        /// The most bytes of directly sampled material textures plus baked layers that one
        /// scene may keep resident during a dispatch. A nil budget uses the device's
        /// `recommendedMaxWorkingSetSize` when reported and 64 MiB otherwise.
        public var textureMemoryBudget: Int?
        /// The most anisotropic samples per lookup, from 1 to 16, for directly sampled maps
        /// whose `TextureSampling` minifies linearly. One filters isotropically, as raster does.
        public var maximumAnisotropy: Int
        /// Adds explicit light sampling and MIS for opaque mesh triangles with constant
        /// or textured emission. Surface graphs, nonopaque coverage, unlit surfaces and closed-walk
        /// boundaries keep their existing path-hit behavior.
        public var sampleEmissiveMeshes: Bool

        public init(
            textures: Textures = .fullResolution(), acceleration: Acceleration = .automatic,
            textureMemoryBudget: Int? = nil, maximumAnisotropy: Int = 16,
            sampleEmissiveMeshes: Bool = false
        ) {
            self.textures = textures
            self.acceleration = acceleration
            self.textureMemoryBudget = textureMemoryBudget
            self.maximumAnisotropy = maximumAnisotropy
            self.sampleEmissiveMeshes = sampleEmissiveMeshes
        }
    }

    /// Sizes and update counters for benchmarks and memory reports.
    public struct Statistics: Equatable, Sendable {
        /// `.metal` or `.software`: the structure rays traverse.
        public let acceleration: Options.Acceleration
        /// Top-level instances.
        public let instanceCount: Int
        public let bottomLevelCount: Int
        /// Triangles stored in bottom levels; a shared level counts once.
        public let triangleCount: Int
        /// Triangles traced across every instance.
        public let instancedTriangleCount: Int
        /// Software BVH nodes; zero with Metal ray tracing.
        public let bottomLevelNodeCount: Int
        public let topLevelNodeCount: Int
        /// Total length of the scene's Metal buffers: bottom-level vertices, indices and,
        /// for the software BVH, primitive numbers and nodes; instance records, top-level
        /// nodes or Metal instance descriptors, materials, the texture table and the surface
        /// texture table.
        public let bufferByteCount: Int
        /// Allocated size of the Metal acceleration structures; zero for the software BVH.
        public let accelerationStructureByteCount: Int
        /// Allocated size of the baked texture array, or its logical mip bytes if larger.
        public let textureByteCount: Int
        /// Distinct textures the kernel samples directly at full resolution.
        public let boundTextureCount: Int
        /// Texture table entries that sample a baked layer instead.
        public let bakedTextureCount: Int
        /// Allocated size of the directly sampled textures. The materials own them; the scene
        /// keeps them resident while it traces.
        public let referencedTextureByteCount: Int
        public let revision: UInt64
        /// Bottom levels built since preparation, including the initial ones.
        public let bottomLevelBuildCount: Int
        /// Top-level builds since preparation, including the initial one.
        public let topLevelBuildCount: Int
        /// Updates that refit the existing top-level tree instead of rebuilding it.
        public let topLevelRefitCount: Int
        /// Skinned bottom levels rebuilt from `computeSkinning` output since preparation.
        public let computeSkinnedBuildCount: Int
    }

    public let device: any MTLDevice
    /// `.metal` or `.software`: the structure that preparation built and rays traverse.
    public let accelerationKind: Options.Acceleration
    /// Triangles stored in bottom levels; instances of a shared level count once.
    public var triangleCount: Int { acceleration.bottomLevels.values.reduce(0) { $0 + $1.triangleCount } }
    /// Software BVH nodes across the top level and every bottom level; zero with Metal ray tracing.
    public var nodeCount: Int {
        acceleration.topLevelNodes.count + acceleration.bottomLevels.values.reduce(0) { $0 + $1.nodeCount }
    }
    public let materialCount: Int
    /// World-space triangles in the explicit emitter distribution, including instances.
    /// At most 262,144 triangles are sampled, in prepared instance and triangle order.
    /// Other emissive triangles retain their ordinary path-hit contribution.
    public var sampledEmissiveTriangleCount: Int { emissiveTriangles?.count ?? 0 }
    /// Texture table entries: directly sampled textures with their sampler states, and baked layers.
    public let textureCount: Int
    /// Whether a material is an index-matched volume boundary that shadow rays can cross.
    package private(set) var hasNullBoundaries: Bool
    /// Increments after each update that changes traced transforms or geometry.
    public private(set) var revision: UInt64 = 0
    /// Changes when consecutive frames no longer share stable vertex and instance identities or materials.
    private(set) var motionRevision: UInt64 = 0
    /// When set, `updateGeometry(for:in:replacements:)` skins every rebuilt skinned
    /// node in one compute pass on the scene's queue, then waits and reads the
    /// vertices back for the bottom-level build, instead of `MeshSkin.deform(_:using:)`
    /// on the CPU. It must use the scene's device. Preparation always skins on the CPU.
    public var computeSkinning: ComputeSkinning?
    /// Bind-pose uploads for compute skinning, by node.
    private var skinningMeshes: [UUID: ComputeSkinning.Mesh] = [:]

    /// The packed material records. Live edits replace the entire buffer.
    package private(set) var materials: any MTLBuffer
    private(set) var emissiveTriangles: EmissiveTriangleTable?
    let textureArray: any MTLTexture
    /// `TextureTableEntry` records that material and surface-graph texture indices address.
    let textureTable: any MTLBuffer
    /// Every directly sampled texture, made resident for each dispatch.
    let boundTextures: [any MTLTexture]
    let textureSamplers: [any MTLSamplerState]
    private let bakedTextureCount: Int
    let generatedSurfaceLibrary: (any MTLLibrary)?
    /// Live uniform blocks, where each starts in the dispatch's graph uniforms, and
    /// whether any traced graph that shares the block reads its clock.
    let surfaceUniforms: [(uniforms: ShaderUniforms, offset: Int, mask: UInt32, readsClock: Bool)]
    let surfaceUniformSlotCount: Int
    let surfaceReadsProjectionMatrix: Bool
    let graphTextureTable: any MTLBuffer
    /// The acceleration structure that the next encode binds.
    package private(set) var acceleration: Acceleration
    private let queue: any MTLCommandQueue
    // The top-level build only reads completed bottom levels. Its own queue lets
    // refits finish even when a caller has already enqueued a trace on `queue`.
    private let topLevelQueue: any MTLCommandQueue
    /// Every top-level instance of a mesh node: one, or one per `InstanceSet` entry. A node whose
    /// set was empty at preparation maps to no instances.
    private let instanceIndices: [UUID: [Int]]
    private var instanceLayouts: [UUID: UUID]
    private var bottomLevelBuildCount = 0
    /// What each material record was packed from, for `updateMaterials()`.
    private let materialSources: [MaterialSource]
    private let materialPackingOptions: Options
    private let materialTextureSize: Int
    /// The live factor revisions in the current material buffer.
    private var packedLiveRevisions: [ObjectIdentifier: UInt64]
    /// The prepared texture table indices, including the sampler of each entry.
    private let materialTextureIDs: [Builder.TextureKey: UInt32]
    private var topLevelBuildCount = 0
    private var topLevelRefitCount = 0
    private var computeSkinnedBuildCount = 0

    /// Bakes static metallic-roughness meshes from the graph. The tracer uses
    /// emissive geometry plus the area lights, analytic lights and environment in
    /// its `PathTracer.Lighting`. Punctual lights on nodes are not geometry, so the
    /// snapshot skips them; pass `PathTracer.AnalyticLight.lights(in:)` to the
    /// lighting to trace them. Other raster-only lighting and baked occlusion have
    /// no effect. World positions must be finite and within ±10¹² units.
    ///
    /// Skinned nodes snapshot their geometry with the node's joint palette, as
    /// `MeshSkin.deform(_:using:)` computes it; meshes whose surface graph offsets
    /// positions snapshot their current deformation.
    ///
    /// By default, maps sample their own textures at full resolution with their binding's
    /// `TextureSampling`: repeat, clamp-to-edge or mirrored-repeat addressing per axis, the
    /// magnification, minification and mip filters, and the texture's own mip chain.
    /// Anisotropic ray cones pick the level at every hit; see `Options`. A binding with a
    /// custom `sampler` needs a `sampling` description. Texture transforms apply to every
    /// traced map. Masked surfaces test their cutoff. Blended surfaces whose alpha can fall below one trace as
    /// stochastic opacity: every camera, bounce and shadow ray accepts a candidate hit
    /// with probability equal to its alpha, decided by a hash of the ray, instance and
    /// triangle, so the converged image is unbiased alpha coverage rather than sorted
    /// compositing. Material-alpha surfaces stay opaque to every ray, as their raster
    /// depth writes and shadows are; a camera ray that first hits one scales its
    /// radiance and coverage by the alpha, as raster writes it. Clearcoat normal maps
    /// tilt the coat lobe only; without one the coat follows the interpolated vertex normal.
    /// Surface graphs are compiled once per snapshot. Volume boundaries use actual
    /// ray exit distances, so thickness maps do not apply. Specular, sheen,
    /// anisotropy, iridescence, dispersion and unlit materials are traced, as are
    /// diffuse transmission, rough (Oren-Nayar) diffuse, volume scattering and
    /// subsurface scattering, whose media need closed, outward-facing meshes.
    /// Texture inputs must be ready, or uploaded with hazard tracking on this
    /// queue. A fallback bake waits for completion before returning. Directly sampled
    /// textures are referenced, not copied, so edits to their contents appear in later
    /// dispatches; their total size must fit the options' memory budget.
    ///
    /// With `.automatic` or `.metal` acceleration on a ray-tracing GPU, bottom levels build and
    /// compact on the scene's queue and preparation waits for them. The software triangle test
    /// is watertight (Woop et al. 2013): a ray through an edge that two triangles share hits
    /// one of them. Both backends trace the same scene to the same image up to rounding in
    /// hit points.
    public convenience init(queue: any MTLCommandQueue, graph: SceneGraph, options: Options = Options()) throws {
        let preparation = try Preparation(queue: queue, graph: graph, options: options)
        let levels = try Self.buildLevels(preparation.jobs, queue: queue, metal: preparation.metal, progress: nil)
        try self.init(preparation: preparation, levels: levels)
    }

    /// Prepares a scene as `init(queue:graph:options:)` does, building the bottom levels on
    /// the concurrent pool so the caller's actor stays free. The graph is resolved, geometry
    /// is read and skinned, and materials and textures are packed on the caller's executor
    /// first; the top level is built there afterwards. `progress` receives each finished
    /// bottom level from the build task. Cancelling the task stops between levels and throws
    /// `CancellationError`.
    nonisolated(nonsending) public static func prepare(
        queue: any MTLCommandQueue, graph: SceneGraph, options: Options = Options(),
        progress: (@Sendable (PreparationProgress) -> Void)? = nil
    ) async throws -> PathTracingScene {
        let preparation = try Preparation(queue: queue, graph: graph, options: options)
        let levels = try await buildLevelsConcurrently(preparation.jobs, queue: queue, metal: preparation.metal, progress: progress)
        try Task.checkCancellation()
        return try PathTracingScene(preparation: preparation, levels: levels)
    }

    @concurrent private static func buildLevelsConcurrently(
        _ jobs: [LevelJob], queue: any MTLCommandQueue, metal: Bool, progress: (@Sendable (PreparationProgress) -> Void)?
    ) async throws -> [GeometryKey: BottomLevel] {
        try buildLevels(jobs, queue: queue, metal: metal, cancellable: true, progress: progress)
    }

    /// The caller-side work of a preparation: resolved instances, packed materials and
    /// textures, and the bottom-level jobs.
    private struct Preparation {
        let queue: any MTLCommandQueue
        let metal: Bool
        let builder: Builder
        let instances: [Instance]
        let indices: [UUID: [Int]]
        let instanceLayouts: [UUID: UUID]
        let jobs: [LevelJob]
        let library: (any MTLLibrary)?
        let textureArray: any MTLTexture

        init(queue: any MTLCommandQueue, graph: SceneGraph, options: Options) throws {
            let textureSize: Int
            switch options.textures {
            case .fullResolution(let size), .baked(let size): textureSize = size
            }
            guard (16...512).contains(textureSize) else { throw Failure.textureSize }
            guard (1...16).contains(options.maximumAnisotropy), options.textureMemoryBudget.map({ $0 > 0 }) != false else {
                throw Failure.options
            }
            let device = queue.device
            switch options.acceleration {
            case .automatic: metal = device.supportsRaytracing
            case .metal:
                guard device.supportsRaytracing else { throw Failure.rayTracingUnsupported }
                metal = true
            case .software: metal = false
            }
            var builder = Builder(device: device, textureSize: textureSize, options: options)
            var instances: [Instance] = []
            var indices: [UUID: [Int]] = [:]
            var instanceLayouts: [UUID: UUID] = [:]
            for node in try graph.resolve() where graph.renders(node) {
                if node.batch?.hasVisibleInstances == true { throw Failure.unsupported(node.id, "batched meshes") }
                guard let renderer = node.mesh, renderer.hasActiveTriangles else { continue }
                let placed = try builder.instances(node, renderer: renderer)
                guard instances.count + placed.count <= PathTracingScene.maximumInstanceCount else {
                    throw Failure.instanceLimit
                }
                indices[node.id] = Array(instances.count..<(instances.count + placed.count))
                if !placed.isEmpty, let set = node.instances { instanceLayouts[node.id] = set.layoutID }
                instances.append(contentsOf: placed)
            }
            guard !instances.isEmpty else { throw Failure.emptyScene }

            var sources = GeometrySources(queue: queue)
            var jobs: [LevelJob] = []
            var queued = Set<GeometryKey>()
            var stored = 0
            for instance in instances where queued.insert(instance.geometry).inserted {
                let job = try PathTracingScene.job(for: instance, geometry: try PathTracingScene.source(of: instance, sources: &sources))
                stored += job.triangleCount
                guard stored <= PathTracingScene.maximumTriangleCount else { throw Failure.triangleLimit }
                jobs.append(job)
            }

            let referenced = builder.boundTextures.reduce(0) { $0 + $1.allocatedSize }
            let baked = builder.bakeSources.isEmpty ? 0 : builder.bakeSources.count * PathTracingScene.mipTexelCount(textureSize) * 8
            let recommended = Int(clamping: device.recommendedMaxWorkingSetSize)
            // Simulator may report no recommendation. The 64 MiB default allows for
            // allocation padding beyond the separate 32 MiB cap on baked texels.
            let budget = options.textureMemoryBudget ?? (recommended > 0 ? recommended : 64 * 1_048_576)
            guard referenced + baked <= budget else { throw Failure.textureMemory(referenced + baked, budget) }
            library = try PathTracingScene.compileSurfaceLibrary(graphs: builder.graphs, device: device)
            textureArray = try PathTracingScene.bake(
                builder.bakeSources, size: textureSize, queue: queue, referencedBytes: referenced, budget: budget
            )
            self.queue = queue
            self.builder = builder
            self.instances = instances
            self.indices = indices
            self.instanceLayouts = instanceLayouts
            self.jobs = jobs
        }
    }

    private init(preparation: Preparation, levels: [GeometryKey: BottomLevel]) throws {
        let builder = preparation.builder
        let device = preparation.queue.device
        let metal = preparation.metal
        let topLevelQueue: any MTLCommandQueue
        if metal {
            guard let queue = device.makeCommandQueue() else { throw Failure.resource("path top-level command queue") }
            queue.label = "Frostfield path top-level builds"
            topLevelQueue = queue
        } else {
            topLevelQueue = preparation.queue
        }
        let topLevel = try Self.topLevel(
            instances: preparation.instances, bottomLevels: levels, refitting: nil, metal: metal, queue: topLevelQueue
        )
        generatedSurfaceLibrary = preparation.library
        let materialBuffer = try Self.makeBuffer(builder.materials, device: device, label: "path materials")
        materials = materialBuffer
        if builder.options.sampleEmissiveMeshes {
            emissiveTriangles = try Self.makeEmissiveTriangles(
                instances: preparation.instances, bottomLevels: levels, materials: materialBuffer, device: device
            )
        }
        surfaceUniforms = builder.uniformBlocks
        surfaceUniformSlotCount = builder.uniformSlotCount
        surfaceReadsProjectionMatrix = builder.graphs.contains { $0.readsProjectionMatrix }
        graphTextureTable = try Self.makeBuffer(
            builder.graphTextureTable.isEmpty ? [UInt32(0)] : builder.graphTextureTable, device: device,
            label: "path surface texture table"
        )
        textureArray = preparation.textureArray
        textureTable = try Self.makeBuffer(
            builder.entries.isEmpty ? [TextureTableEntry.empty] : builder.entries, device: device, label: "path texture table"
        )
        boundTextures = builder.boundTextures
        textureSamplers = Array(builder.samplers.values)
        bakedTextureCount = builder.entries.count { $0.layer != UInt32.max }
        self.device = device
        queue = preparation.queue
        self.topLevelQueue = topLevelQueue
        accelerationKind = metal ? .metal : .software
        materialCount = builder.materials.count
        hasNullBoundaries = builder.hasNullBoundaries
        textureCount = builder.entries.count
        materialSources = builder.sources
        materialPackingOptions = builder.options
        materialTextureSize = builder.textureSize
        packedLiveRevisions = builder.liveSnapshots.mapValues(\.revision)
        materialTextureIDs = builder.textureIDs
        instanceIndices = preparation.indices
        instanceLayouts = preparation.instanceLayouts
        acceleration = Acceleration(
            bottomLevels: levels, instances: preparation.instances, instanceBuffer: topLevel.instanceBuffer,
            topLevel: topLevel.nodeBuffer, topLevelNodes: topLevel.nodes, topLevelBuildCost: topLevel.buildCost,
            metal: topLevel.metal
        )
        bottomLevelBuildCount = levels.count
        topLevelBuildCount = 1
    }

    /// Bakes every map at `textureSize`² texels, from 16 to 512.
    /// Use `init(queue:graph:options:)` to sample maps at full resolution.
    public convenience init(queue: any MTLCommandQueue, graph: SceneGraph, textureSize: Int) throws {
        try self.init(queue: queue, graph: graph, options: Options(textures: .baked(size: textureSize)))
    }

    /// Moves every instance to its node's current world transform in O(instances).
    /// The software top level is refit around the moved bounds, or rebuilt when refitting
    /// has more than doubled its traversal cost; the Metal instance structure is refit on
    /// the GPU into a new structure and published after its build completes. Bottom levels,
    /// materials, textures and shaders are reused.
    ///
    /// The graph must hold the same mesh nodes that contribute instances, by ID,
    /// with the same renderers, instance sets and material parameters. Empty instance
    /// sets contribute no geometry. Other nodes and the hierarchy may change.
    /// Skinned nodes must keep the joint palette of their last geometry snapshot,
    /// because posing changes geometry: use `updateGeometry(for:in:replacements:)` for them.
    /// Temporal frames track the same nodes and instance slots across transform edits.
    /// Replacing an entire instance set or its slots invalidates temporal history.
    /// Returns false, and keeps `revision`, when no transform changed. Throws without
    /// changing the scene when validation fails.
    @discardableResult
    public func updateTransforms(_ graph: SceneGraph) throws -> Bool {
        let (nodes, layouts) = try matchedNodes(in: graph, rebuilding: [])
        var instances = acceleration.instances
        var changed = false
        for (index, _, world) in nodes where !Self.equal(instances[index].world, world) {
            instances[index].world = world
            changed = true
        }
        if changed {
            try commit(instances: instances, bottomLevels: acceleration.bottomLevels, built: 0, rebuild: false)
        }
        if instanceLayouts != layouts {
            instanceLayouts = layouts
            motionRevision &+= 1
        }
        return changed
    }

    /// Rebuilds the bottom levels of the given mesh nodes from the graph's current
    /// state, then applies every node's world transform as `updateTransforms(_:)` does.
    ///
    /// Each node's geometry is its renderer's current draw range, including a surface
    /// graph's position offset, or its entry in `replacements`, such as the output of
    /// `MeshMorphTargets.apply(weights:to:)`. Skinned nodes then apply their node's
    /// joint palette. Replacement geometry is object-space, may change topology, and
    /// is updated even when `nodeIDs` omits its node. A skinned or replaced node owns
    /// its bottom level. An unskinned node without a replacement shares its
    /// renderer's level with the other unskinned nodes of that renderer, so
    /// rebuilding it changes all of them.
    ///
    /// Renderers whose surface graph offsets positions run that deformation on the
    /// scene's queue and wait for it. The graph must keep the prepared mesh nodes,
    /// renderers and material parameters, and nodes outside the update must keep
    /// their joint palettes. Prepare a new scene if a renderer starts or stops
    /// contributing triangles. Cost grows with the rebuilt triangles plus the instance count.
    /// The update throws without changing the scene when any node is unknown or
    /// any geometry is invalid, and always increments `revision` when it succeeds.
    /// Temporal motion can keep history for native skin, morph and surface-graph
    /// poses with the same renderer vertex correspondence. Explicit replacements
    /// and other geometry rebuilds restart temporal history.
    public func updateGeometry(
        for nodeIDs: [UUID], in graph: SceneGraph, replacements: [UUID: MeshGeometry] = [:]
    ) throws {
        if let unknown = (nodeIDs + Array(replacements.keys)).first(where: { instanceIndices[$0] == nil }) {
            throw Failure.unknownNode(unknown)
        }
        // A node whose instance set was empty at preparation has no geometry to rebuild.
        let requested = Set(nodeIDs).union(replacements.keys).filter { instanceIndices[$0]?.isEmpty == false }
        let (nodes, layouts) = try matchedNodes(in: graph, rebuilding: requested)
        var instances = acceleration.instances
        for (index, node, world) in nodes {
            instances[index].world = world
            instances[index].palette = node.jointPalette
            instances[index].morphWeights = node.morphWeights
        }

        var sources = GeometrySources(queue: queue)
        var jobs: [LevelJob] = []
        var queued = Set<GeometryKey>()
        // Sorting keeps the work, and the first reported failure, independent of set order.
        let order = requested.sorted(by: { instanceIndices[$0]![0] < instanceIndices[$1]![0] })
        let skinned = try computeSkinned(order, instances: instances, replacements: replacements, sources: &sources)
        for id in order {
            let indices = instanceIndices[id]!
            var first = instances[indices[0]]
            let replacement = replacements[id]
            let key: GeometryKey = first.renderer.skin != nil || replacement != nil || Self.isMorphed(first)
                ? .node(id) : .shared(ObjectIdentifier(first.renderer))
            for index in indices { instances[index].geometry = key }
            guard queued.insert(key).inserted else { continue }
            let source = try replacement ?? skinned[id] ?? Self.source(of: first, sources: &sources)
            first.geometry = key
            jobs.append(try Self.job(
                for: first, geometry: source, preskinned: skinned[id] != nil, usesDrawRange: replacement == nil
            ))
        }
        let keepsMotion = instanceLayouts == layouts && jobs.allSatisfy {
            $0.tracksPoseMotion && acceleration.bottomLevels[$0.key]?.tracksPoseMotion == true
        }
        let used = Set(instances.map(\.geometry))
        let kept = acceleration.bottomLevels.filter { used.contains($0.key) && !queued.contains($0.key) }
        let stored = kept.values.reduce(0) { $0 + $1.triangleCount } + jobs.reduce(0) { $0 + $1.triangleCount }
        guard stored <= Self.maximumTriangleCount else { throw Failure.triangleLimit }
        let rebuilt = try Self.buildLevels(jobs, queue: queue, metal: accelerationKind == .metal, progress: nil)
        try commit(instances: instances, bottomLevels: kept.merging(rebuilt) { _, new in new }, built: rebuilt.count, rebuild: true)
        computeSkinnedBuildCount += skinned.count
        instanceLayouts = layouts
        if !keepsMotion { motionRevision &+= 1 }
    }

    /// Repacks materials whose live factors changed since the scene last packed them,
    /// such as factors an `AnimationMixer` writes with `applyLiveProperties()`, into a
    /// new material buffer and increments `revision`. Textures, surface graphs and
    /// geometry are reused, and frames in flight keep the previous buffer. Returns
    /// false, and keeps `revision`, when no live factors changed.
    @discardableResult
    public func updateMaterials() throws -> Bool {
        var snapshots: [ObjectIdentifier: PhysicallyBasedMaterial.LiveFactors.Snapshot] = [:]
        for source in materialSources {
            guard let live = source.material.liveFactors else { continue }
            let id = ObjectIdentifier(live)
            if snapshots[id] == nil { snapshots[id] = live.snapshot }
        }
        let revisions = snapshots.mapValues(\.revision)
        guard revisions != packedLiveRevisions else { return false }
        // Reuse the prepared texture indices and preserve the uniform order.
        // The table and its sampler states stay unchanged while factors animate.
        var builder = Builder(device: device, textureSize: materialTextureSize, options: materialPackingOptions)
        builder.textureIDs = materialTextureIDs
        builder.liveSnapshots = snapshots
        let records = try materialSources.map { source in
            try builder.pack(
                source.material, parameters: source.parameters, instanceTint: source.instanceTint,
                readsVertexColors: source.readsVertexColors, nodeID: source.nodeID
            )
        }
        guard builder.textureIDs == materialTextureIDs else {
            throw Failure.structureChanged(materialSources[0].nodeID)
        }
        let replacement = try Self.makeBuffer(records, device: device, label: "path materials")
        let emitters = materialPackingOptions.sampleEmissiveMeshes ? try Self.makeEmissiveTriangles(
            instances: acceleration.instances, bottomLevels: acceleration.bottomLevels, materials: replacement, device: device
        ) : nil
        materials = replacement
        emissiveTriangles = emitters
        hasNullBoundaries = builder.hasNullBoundaries
        packedLiveRevisions = revisions
        revision &+= 1
        motionRevision &+= 1
        return true
    }

    /// Skins the rebuilt skinned nodes without replacements in one compute pass.
    private func computeSkinned(
        _ order: [UUID], instances: [Instance], replacements: [UUID: MeshGeometry], sources: inout GeometrySources
    ) throws -> [UUID: MeshGeometry] {
        guard let computeSkinning, let first = order.first else { return [:] }
        guard computeSkinning.device === device else { throw Failure.wrongDevice(first) }
        var jobs: [(mesh: ComputeSkinning.Mesh, palette: JointPalette)] = []
        var ids: [UUID] = []
        for id in order where replacements[id] == nil {
            let instance = instances[instanceIndices[id]![0]]
            guard let skin = instance.renderer.skin else { continue }
            guard let palette = instance.palette else { throw Failure.skinning(id, .missingPalette) }
            let geometry = try Self.source(of: instance, sources: &sources)
            let mesh: ComputeSkinning.Mesh
            // Unmorphed nodes reuse their bind-pose upload; morphed ones upload their current weights' geometry.
            if !Self.isMorphed(instance), let cached = skinningMeshes[id], cached.skin.influences == skin.influences,
               cached.geometry.indices == geometry.indices,
               zip(cached.geometry.vertices, geometry.vertices).allSatisfy({ a, b in
                   a.position == b.position && a.normal == b.normal && a.textureCoordinate == b.textureCoordinate
                       && a.secondaryTextureCoordinate == b.secondaryTextureCoordinate && a.tangent == b.tangent
                       && a.color == b.color
               }), cached.geometry.vertices.count == geometry.vertices.count {
                mesh = cached
            } else {
                mesh = try computeSkinning.makeMesh(geometry: geometry, skin: skin)
                if !Self.isMorphed(instance) { skinningMeshes[id] = mesh }
            }
            jobs.append((mesh, palette))
            ids.append(id)
        }
        guard !jobs.isEmpty else { return [:] }
        do {
            return Dictionary(uniqueKeysWithValues: zip(ids, try computeSkinning.skin(jobs, queue: queue)))
        } catch let error as SkinningError {
            throw Failure.skinning(ids[0], error)
        }
    }

    /// Current sizes, memory and update counts.
    public var statistics: Statistics {
        let levels = acceleration.bottomLevels
        return Statistics(
            acceleration: accelerationKind,
            instanceCount: acceleration.instances.count,
            bottomLevelCount: levels.count,
            triangleCount: triangleCount,
            instancedTriangleCount: acceleration.instances.reduce(0) { $0 + (levels[$1.geometry]?.triangleCount ?? 0) },
            bottomLevelNodeCount: levels.values.reduce(0) { $0 + $1.nodeCount },
            topLevelNodeCount: acceleration.topLevelNodes.count,
            bufferByteCount: buffers.reduce(0) { $0 + $1.length },
            accelerationStructureByteCount: acceleration.structures.reduce(0) { $0 + $1.size },
            textureByteCount: Self.bakedTextureBytes(textureArray),
            boundTextureCount: boundTextures.count,
            bakedTextureCount: bakedTextureCount,
            referencedTextureByteCount: boundTextures.reduce(0) { $0 + $1.allocatedSize },
            revision: revision,
            bottomLevelBuildCount: bottomLevelBuildCount,
            topLevelBuildCount: topLevelBuildCount,
            topLevelRefitCount: topLevelRefitCount,
            computeSkinnedBuildCount: computeSkinnedBuildCount
        )
    }

    /// Every Metal buffer the scene currently owns.
    package var buffers: [any MTLBuffer] {
        acceleration.bottomLevelBuffers + [acceleration.metal?.descriptorBuffer].compactMap { $0 }
            + [acceleration.instanceBuffer, acceleration.topLevel, materials, textureTable, graphTextureTable]
            + [emissiveTriangles?.buffer].compactMap { $0 }
    }

    // Validates the graph against the prepared instances and returns each instance's resolved node.
    // Each prepared instance with its node and current world transform. Instance
    // tints and counts are part of the structure; morph weights and palettes are geometry.
    private func matchedNodes(
        in graph: SceneGraph, rebuilding: Set<UUID>
    ) throws -> (nodes: [(Int, ResolvedSceneNode, simd_float4x4)], layouts: [UUID: UUID]) {
        var matched: [(Int, ResolvedSceneNode, simd_float4x4)] = []
        var layouts = instanceLayouts
        matched.reserveCapacity(acceleration.instances.count)
        for node in try graph.resolve() where graph.renders(node) {
            if node.batch?.hasVisibleInstances == true { throw Failure.unsupported(node.id, "batched meshes") }
            guard let renderer = node.mesh else { continue }
            let instanceSet = node.instances
            if instanceIndices[node.id] == nil, instanceSet?.count == 0 || !renderer.hasActiveTriangles { continue }
            guard renderer.hasActiveTriangles,
                  let indices = instanceIndices[node.id], indices.count == (instanceSet?.count ?? 1) else {
                throw Failure.structureChanged(node.id)
            }
            // A node that still has an empty instance set contributes nothing.
            guard let firstIndex = indices.first else { continue }
            let wasInstanced = acceleration.instances[firstIndex].instanceIndex != nil
            guard wasInstanced == (instanceSet != nil) else { throw Failure.structureChanged(node.id) }
            if let instanceSet, layouts[node.id] != instanceSet.layoutID {
                layouts[node.id] = instanceSet.layoutID
            }
            try Self.validatePalette(node.jointPalette, renderer: renderer, nodeID: node.id)
            for index in indices {
                let instance = acceleration.instances[index]
                let placement = instanceSet.flatMap { set in
                    instance.instanceIndex.map { (set.transforms[$0], set.tints[$0]) }
                }
                guard instance.renderer === renderer, instance.parameters == node.materialParameters,
                      instance.tint == (placement?.1 ?? SIMD4(repeating: 1)) else {
                    throw Failure.structureChanged(node.id)
                }
                if !rebuilding.contains(node.id),
                   (renderer.skin != nil && node.jointPalette != instance.palette)
                    || (Self.isMorphed(instance) && node.morphWeights != instance.morphWeights) {
                    throw Failure.staleGeometry(node.id)
                }
                matched.append((index, node, placement.map { node.worldTransform * $0.0 } ?? node.worldTransform))
            }
        }
        if matched.count != acceleration.instances.count {
            let present = Set(matched.map(\.0))
            let missing = acceleration.instances.indices.first { !present.contains($0) } ?? 0
            throw Failure.structureChanged(acceleration.instances[missing].nodeID)
        }
        return (matched, layouts)
    }

    // `rebuild` builds a new Metal instance structure, as changed bottom levels require.
    private func commit(instances: [Instance], bottomLevels: [GeometryKey: BottomLevel], built: Int, rebuild: Bool) throws {
        let emitters = materialPackingOptions.sampleEmissiveMeshes ? try Self.makeEmissiveTriangles(
            instances: instances, bottomLevels: bottomLevels, materials: materials, device: device
        ) : nil
        let topLevel = try Self.topLevel(
            instances: instances, bottomLevels: bottomLevels,
            refitting: (acceleration.topLevelNodes, acceleration.topLevelBuildCost, rebuild ? nil : acceleration.metal),
            metal: accelerationKind == .metal, queue: topLevelQueue
        )
        acceleration = Acceleration(
            bottomLevels: bottomLevels, instances: instances, instanceBuffer: topLevel.instanceBuffer,
            topLevel: topLevel.nodeBuffer, topLevelNodes: topLevel.nodes,
            topLevelBuildCost: topLevel.refit ? acceleration.topLevelBuildCost : topLevel.buildCost, metal: topLevel.metal
        )
        emissiveTriangles = emitters
        bottomLevelBuildCount += built
        if topLevel.refit { topLevelRefitCount += 1 } else { topLevelBuildCount += 1 }
        revision &+= 1
    }

    private static func validatePalette(_ palette: JointPalette?, renderer: MeshRenderer, nodeID: UUID) throws {
        guard let skin = renderer.skin else {
            if palette != nil { throw Failure.skinning(nodeID, .unexpectedPalette) }
            return
        }
        guard let palette else { throw Failure.skinning(nodeID, .missingPalette) }
        guard palette.matrices.count == skin.jointCount else { throw Failure.skinning(nodeID, .paletteCount) }
    }

    private static func equal(_ a: simd_float4x4, _ b: simd_float4x4) -> Bool {
        a.columns.0 == b.columns.0 && a.columns.1 == b.columns.1 && a.columns.2 == b.columns.2 && a.columns.3 == b.columns.3
    }

    // MARK: - Records

    // Keep these records in lockstep with PathTracingShading.h.
    package struct BVHNode {
        package var minimum: SIMD4<Float>
        package var maximum: SIMD4<Float>
        package var meta: SIMD4<UInt32>
    }

    package struct Material {
        var baseColor: SIMD4<Float>
        var factors: SIMD4<Float>
        /// Emission (rgb), then the alpha test (w): a mask cutoff of zero or more,
        /// `opaqueCoverage`, `stochasticCoverage` or `writtenCoverage`.
        var emission: SIMD4<Float>
        var optical: SIMD4<Float>
        var layers: SIMD4<Float>
        var textureIndices: SIMD4<UInt32>
        var textureUVs: SIMD4<UInt32>
        var graphParameters: SIMD4<Float>
        var graphResources: SIMD4<UInt32>
        package var extensions: MaterialExtensions
        package var lobes: MaterialLobes
    }

    /// Diffuse transmission, diffuse roughness, clearcoat normal and medium data after the
    /// extension block, matching `PTMaterialLobes` in PathTracingShading.h (384 bytes).
    package struct MaterialLobes {
        /// Diffuse transmission color (rgb) and the translucent share of the diffuse lobe (a):
        /// the diffuse transmission factor or subsurface weight.
        package var translucency: SIMD4<Float>
        /// Multiple-scattering color before the entry-point texture (rgb) and Henyey-Greenstein g (a).
        package var scattering: SIMD4<Float>
        /// Extinction per world unit (rgb), unused.
        package var extinction: SIMD4<Float>
        /// Diffuse roughness, clearcoat normal scale, light-map intensity, retroreflectivity.
        package var surface: SIMD4<Float>
        /// x: `lobe*` flags; y/z: base and clearcoat normal sources; w: object-space normal flags.
        package var flags: SIMD4<UInt32>
        /// Texture layers for clearcoat normal, diffuse transmission, diffuse transmission color,
        /// diffuse roughness, multiple-scattering color and baked irradiance, then two unused slots.
        package var textures: InlineArray<8, UInt32>
        /// Bit i of x selects the secondary UV set for slot i; bits 2i and 2i + 1 of y and z hold
        /// its U and V addressing and bit i of w selects nearest filtering.
        package var coordinates: SIMD4<UInt32>
        /// Two transform rows per slot.
        package var transforms: InlineArray<16, SIMD4<Float>>
    }

    /// `MaterialLobes.flags.x` bits, matching `ptLobe*` in PathTracingShading.h.
    package static let lobeMedium: UInt32 = 1
    package static let lobeDiffuseBoundary: UInt32 = 2
    package static let lobeNullBoundary: UInt32 = 4
    package static let lobeScattering: UInt32 = 8
    package static let lobeSubsurfaceColor: UInt32 = 16
    /// The path tracer's largest subsurface extinction: mean free paths shorter than 10⁻⁴
    /// world units would fall inside the ray offset that keeps walks off the boundary.
    static let maximumSubsurfaceExtinction: Float = 1e4

    /// KHR material extension data at the end of each material, matching
    /// `PTMaterialExtensions` in PathTracingShading.h (592 bytes).
    package struct MaterialExtensions {
        /// Specular color factor (rgb) and factor (a).
        package var specular: SIMD4<Float>
        /// Sheen color (rgb) and perceptual roughness (a).
        package var sheen: SIMD4<Float>
        /// Strength, cosine and sine of the rotation, unused.
        package var anisotropy: SIMD4<Float>
        /// Factor, film index of refraction, minimum and maximum thickness in nanometers.
        package var iridescence: SIMD4<Float>
        /// One for unlit, dispersion, one when specular modulates the dielectric Fresnel,
        /// and one when the base tint is saturated after applying vertex colors.
        package var options: SIMD4<Float>
        /// Texture layers for clearcoat factor, clearcoat roughness, transmission, specular,
        /// specular color, sheen color, sheen roughness, anisotropy, iridescence and
        /// iridescence thickness, then two unused slots. Missing maps hold `UInt32.max`.
        package var textures: InlineArray<12, UInt32>
        /// Bit i of x selects the secondary UV set for texture slot i. For each map m
        /// (0 base color, 1 normal, 2 metallic-roughness, 3 emissive, then 4 + texture slot),
        /// bits 2m and 2m + 1 of y and z hold its U and V addressing (0 repeat, 1 clamp to
        /// edge, 2 mirrored repeat) and bit m of w selects nearest filtering. Zero repeats
        /// with linear filtering.
        package var coordinates: SIMD4<UInt32>
        /// Two transform rows each for base color, normal, metallic-roughness and emissive,
        /// then for the ten texture slots.
        package var transforms: InlineArray<28, SIMD4<Float>>
    }

    /// One top-level instance, matching PTInstance (160 bytes). Rows of the affine
    /// object-to-world and world-to-object transforms; meta holds the material, bottom-level
    /// node and triangle counts, and flags; surface holds the surface graph and the object ID. The addresses
    /// are the bottom level's GPU buffers, with zero for absent software-only buffers.
    package struct InstanceRecord {
        package var objectToWorld0: SIMD4<Float>
        package var objectToWorld1: SIMD4<Float>
        package var objectToWorld2: SIMD4<Float>
        package var worldToObject0: SIMD4<Float>
        package var worldToObject1: SIMD4<Float>
        package var worldToObject2: SIMD4<Float>
        package var meta: SIMD4<UInt32>
        package var surface: SIMD4<UInt32>
        package var vertices: UInt64
        package var indices: UInt64
        package var nodes: UInt64
        package var primitives: UInt64
    }

    package static let mirroredInstance: UInt32 = 1
    package static let cullsBackFacesInstance: UInt32 = 2
    package static let coloredInstance: UInt32 = 4
    /// `Material.emission.w` without an alpha test; a mask stores its cutoff of zero or more.
    static let opaqueCoverage: Float = -1
    /// `Material.emission.w` for stochastic opacity: each ray accepts a candidate hit with
    /// probability equal to its alpha.
    static let stochasticCoverage: Float = -2
    /// `Material.emission.w` for material alpha: opaque to rays, while a camera ray that first
    /// hits the surface scales its radiance and coverage by the alpha, as raster writes it.
    static let writtenCoverage: Float = -3

    /// A source texture and whether it bakes with nearest resampling.
    struct BakeSource {
        let texture: any MTLTexture
        let nearest: Bool
    }

    /// One texture table entry, matching PTTextureBinding in PathTracingShading.h (24 bytes).
    /// A directly sampled texture stores its handle and sampler state with `layer` set to
    /// `UInt32.max`; a fallback stores its baked layer and zero handles.
    package struct TextureTableEntry {
        var texture: MTLResourceID
        var sampler: MTLResourceID
        var layer: UInt32
        var flags: UInt32

        static let empty = TextureTableEntry(texture: MTLResourceID(), sampler: MTLResourceID(), layer: 0, flags: 0)
    }

    package enum GeometryKey: Hashable, Sendable {
        /// The current geometry of an unskinned renderer, shared by its nodes.
        case shared(ObjectIdentifier)
        /// Geometry owned by one node: skinned, or replaced by the caller.
        case node(UUID)
    }

    struct Instance {
        let nodeID: UUID
        let objectID: UInt32
        /// The entry in the node's `InstanceSet`, or nil for an ordinary node.
        let instanceIndex: Int?
        let tint: SIMD4<Float>
        let renderer: MeshRenderer
        let parameters: SIMD4<Float>
        let material: UInt32
        let surfaceGraph: UInt32
        let cullsBackFaces: UInt32
        /// The material has a mask or stochastic coverage, so Metal tests its candidates.
        let alphaTested: Bool
        var geometry: GeometryKey
        var world: simd_float4x4
        var palette: JointPalette?
        var morphWeights: [Float]?
    }

    private static func isMorphed(_ instance: Instance) -> Bool {
        instance.renderer.morphTargets != nil || instance.morphWeights != nil
    }

    // Morphed nodes bake their weights; other meshes share their renderer's current geometry.
    private static func source(of instance: Instance, sources: inout GeometrySources) throws -> MeshGeometry {
        isMorphed(instance) ? try instance.renderer.morphedGeometry(weights: instance.morphWeights)
            : try sources.geometry(of: instance.renderer)
    }

    package struct Acceleration {
        let bottomLevels: [GeometryKey: BottomLevel]
        let instances: [Instance]
        let instanceBuffer: any MTLBuffer
        /// Software top-level nodes, or one unused node under Metal ray tracing.
        let topLevel: any MTLBuffer
        let topLevelNodes: [BVHNode]
        /// Summed node surface area over root area when the top level was last built.
        let topLevelBuildCost: Float
        let metal: MetalTopLevel?

        var bottomLevelBuffers: [any MTLBuffer] { bottomLevels.values.flatMap(\.buffers) }
        var poseVertexBuffers: [any MTLBuffer] {
            bottomLevels.values.compactMap { $0.tracksPoseMotion ? $0.vertices : nil }
        }
        /// The instance structure first, then every bottom-level structure.
        var structures: [any MTLAccelerationStructure] {
            guard let metal else { return [] }
            return [metal.structure] + metal.levels
        }
    }

    // Reads each renderer's current geometry once per preparation or update.
    private struct GeometrySources {
        let queue: any MTLCommandQueue
        var cache: [ObjectIdentifier: MeshGeometry] = [:]

        mutating func geometry(of renderer: MeshRenderer) throws -> MeshGeometry {
            let key = ObjectIdentifier(renderer)
            if let cached = cache[key] { return cached }
            let geometry = try renderer.deformedGeometry(queue: queue)
            cache[key] = geometry
            return geometry
        }
    }

    // MARK: - Levels

    // Applies the node's joint palette unless the compute pass already skinned the geometry.
    private static func job(
        for instance: Instance, geometry: MeshGeometry, preskinned: Bool = false, usesDrawRange: Bool = true
    ) throws -> LevelJob {
        let id = instance.nodeID
        var source = geometry
        if !preskinned, let skin = instance.renderer.skin {
            guard let palette = instance.palette else { throw Failure.skinning(id, .missingPalette) }
            do {
                source = try skin.deform(source, using: palette)
            } catch let error as SkinningError {
                throw Failure.skinning(id, error)
            }
        }
        let renderer = instance.renderer
        if usesDrawRange {
            guard let selected = renderer.geometryInDrawRange(source) else { throw Failure.structureChanged(id) }
            source = selected
        }
        if let material = renderer.pathTracingMaterial, material.hasLightMap,
           material.lightMap?.textureCoordinateSet == .secondary,
           source.vertices.contains(where: { $0.secondaryTextureCoordinate == nil }) {
            throw MeshRenderer.Failure.missingSecondaryTextureCoordinates
        }
        guard source.indices.count / 3 <= maximumTriangleCount else { throw Failure.triangleLimit }
        // Skin, morph and typed position offsets preserve the renderer's vertex
        // identities. Matching indices alone cannot establish this for replacements.
        let hasPose = renderer.skin != nil || renderer.morphTargets != nil
            || renderer.pathTracingMaterial?.surfaceGraph?.hasVertexOutputs == true
        let tracksPoseMotion = usesDrawRange && hasPose
            && source.vertices.count == renderer.geometry.vertices.count && source.indices == renderer.geometry.indices
        return LevelJob(key: instance.geometry, nodeID: id, geometry: source, tracksPoseMotion: tracksPoseMotion)
    }

    private struct TopLevel {
        let instanceBuffer: any MTLBuffer
        let nodeBuffer: any MTLBuffer
        let nodes: [BVHNode]
        let buildCost: Float
        let refit: Bool
        let metal: MetalTopLevel?
    }

    // Writes the instance records. The software path refits the previous tree when its cost
    // stays within twice the cost at its last build, and otherwise builds a new tree; the
    // Metal path refits the previous instance structure when given one, and otherwise builds.
    // Metal returns some new buffers and command buffers autoreleased, so each build drains its
    // own pool; a caller's loop that never drains one would otherwise keep every replaced buffer.
    private static func topLevel(
        instances: [Instance], bottomLevels: [GeometryKey: BottomLevel],
        refitting previous: (nodes: [BVHNode], buildCost: Float, metal: MetalTopLevel?)?, metal: Bool,
        queue: any MTLCommandQueue
    ) throws -> TopLevel {
        try autoreleasepool {
            try buildTopLevel(instances: instances, bottomLevels: bottomLevels, refitting: previous, metal: metal, queue: queue)
        }
    }

    private static func buildTopLevel(
        instances: [Instance], bottomLevels: [GeometryKey: BottomLevel],
        refitting previous: (nodes: [BVHNode], buildCost: Float, metal: MetalTopLevel?)?, metal: Bool,
        queue: any MTLCommandQueue
    ) throws -> TopLevel {
        let device = queue.device
        let recordStride = MemoryLayout<InstanceRecord>.stride
        guard let instanceBuffer = device.makeBuffer(
            length: recordStride * instances.count, options: [.storageModeShared, .hazardTrackingModeTracked]
        ) else { throw Failure.resource("path instances") }
        instanceBuffer.label = "Frostfield path instances"
        let records = instanceBuffer.contents().bindMemory(to: InstanceRecord.self, capacity: instances.count)
        var bounds: [(SIMD3<Float>, SIMD3<Float>)] = []
        bounds.reserveCapacity(metal ? 0 : instances.count)
        for (index, instance) in instances.enumerated() {
            guard let level = bottomLevels[instance.geometry] else { throw Failure.resource("path bottom level") }
            let world = instance.world
            let linear = simd_float3x3(
                SIMD3(world.columns.0.x, world.columns.0.y, world.columns.0.z),
                SIMD3(world.columns.1.x, world.columns.1.y, world.columns.1.z),
                SIMD3(world.columns.2.x, world.columns.2.y, world.columns.2.z)
            )
            // The world bounds of the level's box: the transformed center plus the absolute
            // linear part applied to the half extent.
            let center = (level.minimum + level.maximum) * 0.5
            let half = (level.maximum - level.minimum) * 0.5
            let worldCenter = linear * center + SIMD3(world.columns.3.x, world.columns.3.y, world.columns.3.z)
            let worldHalf = abs(linear.columns.0) * half.x + abs(linear.columns.1) * half.y + abs(linear.columns.2) * half.z
            let minimum = worldCenter - worldHalf
            let maximum = worldCenter + worldHalf
            guard (0..<3).allSatisfy({ minimum[$0].isFinite && maximum[$0].isFinite
                && abs(minimum[$0]) <= 1e12 && abs(maximum[$0]) <= 1e12 }) else {
                throw Failure.invalidGeometry(instance.nodeID)
            }
            if !metal { bounds.append((minimum, maximum)) }
            let rows = simd_transpose(world)
            let inverseRows = simd_transpose(world.inverse)
            var flags: UInt32 = simd_determinant(linear) < 0 ? mirroredInstance : 0
            if instance.cullsBackFaces != 0 { flags |= cullsBackFacesInstance }
            if instance.renderer.readsVertexColors && level.hasVertexColors { flags |= coloredInstance }
            records[index] = InstanceRecord(
                objectToWorld0: rows.columns.0, objectToWorld1: rows.columns.1, objectToWorld2: rows.columns.2,
                worldToObject0: inverseRows.columns.0, worldToObject1: inverseRows.columns.1,
                worldToObject2: inverseRows.columns.2,
                meta: SIMD4(instance.material, UInt32(level.nodeCount), UInt32(level.triangleCount), flags),
                surface: SIMD4(instance.surfaceGraph, instance.objectID, 0, 0),
                vertices: level.vertices.gpuAddress, indices: level.indices.gpuAddress,
                nodes: level.nodes?.gpuAddress ?? 0, primitives: level.primitives?.gpuAddress ?? 0
            )
        }

        if metal {
            let structure = try metalTopLevel(
                instances: instances, bottomLevels: bottomLevels, refitting: previous?.metal, queue: queue
            )
            let unused = [BVHNode(minimum: .zero, maximum: .zero, meta: .zero)]
            return TopLevel(
                instanceBuffer: instanceBuffer, nodeBuffer: try makeBuffer(unused, device: device, label: "path top-level BVH"),
                nodes: [], buildCost: 0, refit: previous?.metal != nil, metal: structure
            )
        }

        var nodes: [BVHNode]
        var buildCost: Float
        var refit = false
        if var refitted = previous?.nodes, !refitted.isEmpty, let previousCost = previous?.buildCost {
            for index in refitted.indices.reversed() {
                let meta = refitted[index].meta
                if meta.w != 0 {
                    let leaf = bounds[Int(meta.z)]
                    refitted[index].minimum = SIMD4(leaf.0, 0)
                    refitted[index].maximum = SIMD4(leaf.1, 0)
                } else {
                    let first = refitted[Int(meta.x)]
                    let second = refitted[Int(meta.y)]
                    refitted[index].minimum = simd_min(first.minimum, second.minimum)
                    refitted[index].maximum = simd_max(first.maximum, second.maximum)
                }
            }
            buildCost = cost(refitted)
            refit = buildCost <= 2 * previousCost
            nodes = refitted
        } else {
            nodes = []
            buildCost = 0
        }
        if !refit {
            let tree = try buildTree(minimum: bounds.map(\.0), maximum: bounds.map(\.1), leafSize: 1)
            nodes = tree.nodes
            for index in nodes.indices where nodes[index].meta.w != 0 {
                nodes[index].meta.z = tree.order[Int(nodes[index].meta.z)]
            }
            buildCost = cost(nodes)
        }
        return TopLevel(
            instanceBuffer: instanceBuffer, nodeBuffer: try makeBuffer(nodes, device: device, label: "path top-level BVH"),
            nodes: nodes, buildCost: buildCost, refit: refit, metal: nil
        )
    }

    // The surface-area traversal estimate: summed node area relative to the root.
    private static func cost(_ nodes: [BVHNode]) -> Float {
        func area(_ node: BVHNode) -> Float {
            let size = simd_max(node.maximum - node.minimum, .zero)
            return size.x * size.y + size.y * size.z + size.z * size.x
        }
        let root = max(area(nodes[0]), .leastNormalMagnitude)
        return nodes.reduce(0) { $0 + area($1) } / root
    }

    // MARK: - Material packing

    private struct MaterialSource {
        let material: PhysicallyBasedMaterial
        let parameters: SIMD4<Float>
        let instanceTint: SIMD4<Float>
        let readsVertexColors: Bool
        let nodeID: UUID
    }

    private struct Builder {
        let device: any MTLDevice
        let textureSize: Int
        let options: Options
        var materials: [Material] = []
        var sources: [MaterialSource] = []
        var liveSnapshots: [ObjectIdentifier: PhysicallyBasedMaterial.LiveFactors.Snapshot] = [:]
        /// The texture table, in index order.
        var entries: [TextureTableEntry] = []
        var bakeSources: [BakeSource] = []
        var boundTextures: [any MTLTexture] = []
        var samplers: [TextureSampling: any MTLSamplerState] = [:]
        var materialIDs: [MaterialKey: UInt32] = [:]
        var uniformBlockIndices: [ObjectIdentifier: Int] = [:]
        var uniformBlocks: [(uniforms: ShaderUniforms, offset: Int, mask: UInt32, readsClock: Bool)] = []
        // The dispatch writes the camera that surface graphs read into the leading slots.
        var uniformSlotCount = PathTracingScene.graphViewSlotCount
        var graphTextureTable: [UInt32] = []
        var graphs: [SurfaceShaderGraph] = []
        var graphIDs: [SurfaceShaderGraph: UInt32] = [:]
        /// Whether a material has an index-matched boundary that shadow rays cross.
        var hasNullBoundaries = false

        struct MaterialKey: Hashable {
            let mesh: ObjectIdentifier
            let tint: SIMD4<Float>
            let instanceTint: SIMD4<Float>
        }

        /// A table entry: a texture with the sampling of a material map, or nil for a surface graph
        /// texture, whose generated code picks its sampler. `nearest` selects a nearest bake.
        struct TextureKey: Hashable {
            let texture: ObjectIdentifier
            let sampling: TextureSampling?
            let nearest: Bool
        }
        var textureIDs: [TextureKey: UInt32] = [:]
        var bakeLayers: [BakeKey: UInt32] = [:]
        var boundIDs: Set<ObjectIdentifier> = []

        struct BakeKey: Hashable {
            let texture: ObjectIdentifier
            let nearest: Bool
        }

        typealias TextureSampling = PhysicallyBasedMaterial.TextureSampling

        init(device: any MTLDevice, textureSize: Int, options: Options) {
            self.device = device
            self.textureSize = textureSize
            self.options = options
        }

        // Packs each map's sampling into MaterialExtensions.coordinates y, z and w.
        struct SamplingModes {
            var addressU: UInt32 = 0
            var addressV: UInt32 = 0
            var nearest: UInt32 = 0

            /// Records the binding's sampling for map `map`, which a baked fallback uses, and
            /// returns it. A Metal sampler cannot be inspected, so one without a description is rejected.
            mutating func add(
                _ binding: PhysicallyBasedMaterial.TextureBinding, map: Int, nodeID: UUID
            ) throws -> TextureSampling {
                let sampling = try Self.description(of: binding, nodeID: nodeID)
                // Addressing codes are 0 repeat, 1 clamp to edge and 2 mirrored repeat, as for graph textures.
                addressU |= sampling.addressU.code << UInt32(2 * map)
                addressV |= sampling.addressV.code << UInt32(2 * map)
                if sampling.magnification == .nearest { nearest |= 1 << UInt32(map) }
                return sampling
            }

            static func description(
                of binding: PhysicallyBasedMaterial.TextureBinding, nodeID: UUID
            ) throws -> TextureSampling {
                guard binding.sampler == nil || binding.sampling != nil else {
                    throw Failure.unsupported(nodeID, "custom texture samplers without a TextureSampling description")
                }
                return binding.sampling ?? .default
            }
        }

        // Validates a mesh node and packs its materials; bottom levels are built separately.
        // An instanced node yields one top-level instance per entry, all sharing one bottom level.
        mutating func instances(_ node: ResolvedSceneNode, renderer: MeshRenderer) throws -> [Instance] {
            guard renderer.device === device else { throw Failure.wrongDevice(node.id) }
            guard let material = renderer.pathTracingMaterial else {
                throw Failure.unsupported(node.id, "materials outside metallic-roughness PBR")
            }
            try Self.validate(renderer.renderState, material: material, nodeID: node.id)
            try PathTracingScene.validatePalette(node.jointPalette, renderer: renderer, nodeID: node.id)
            let parameters = node.materialParameters
            guard Self.finite(parameters) else { throw Failure.invalidTint(node.id) }
            let graphIndex = try surfaceIndex(material.surfaceGraph)
            if let graph = material.surfaceGraph {
                try PathTracingScene.validatePathInputs(graph, node: node)
            }
            let placements: [(index: Int?, local: simd_float4x4, tint: SIMD4<Float>)]
            if let set = node.instances {
                guard renderer.supportsInstancing else { throw MeshRenderer.Failure.instancingNotPrepared }
                placements = set.transforms.indices.map { ($0, set.transforms[$0], set.tints[$0]) }
            } else {
                placements = [(nil, matrix_identity_float4x4, SIMD4(repeating: 1))]
            }
            let morphed = renderer.morphTargets != nil || node.morphWeights != nil
            let geometry: GeometryKey = renderer.skin != nil || morphed ? .node(node.id) : .shared(ObjectIdentifier(renderer))
            let objectID = PathTracer.objectID(for: node.id)
            return try placements.map { placement in
                let key = MaterialKey(mesh: ObjectIdentifier(renderer), tint: parameters, instanceTint: placement.tint)
                let materialIndex: UInt32
                if let existing = materialIDs[key] {
                    materialIndex = existing
                } else {
                    guard materials.count < PathTracingScene.maximumMaterialCount else { throw Failure.materialLimit }
                    materialIndex = UInt32(materials.count)
                    materials.append(try pack(
                        material, parameters: parameters, instanceTint: placement.tint,
                        readsVertexColors: renderer.readsVertexColors, nodeID: node.id
                    ))
                    sources.append(MaterialSource(
                        material: material, parameters: parameters, instanceTint: placement.tint,
                        readsVertexColors: renderer.readsVertexColors, nodeID: node.id
                    ))
                    materialIDs[key] = materialIndex
                }
                return Instance(
                    nodeID: node.id, objectID: objectID, instanceIndex: placement.index, tint: placement.tint, renderer: renderer,
                    parameters: parameters, material: materialIndex, surfaceGraph: graphIndex,
                    cullsBackFaces: renderer.renderState.side == .double || material.hasTransmission
                        || material.medium != nil || material.hasDiffuseTransmission ? 0 : 1,
                    alphaTested: material.surfaceGraph?.hasCutoutOutputs == true || materials[Int(materialIndex)].emission.w >= 0
                        || materials[Int(materialIndex)].emission.w == PathTracingScene.stochasticCoverage
                        || (material.liveFactors != nil && material.alphaMode == .blend),
                    geometry: geometry,
                    world: node.worldTransform * placement.local, palette: node.jointPalette, morphWeights: node.morphWeights
                )
            }
        }
        /// Rejects raster render state that changes what a surface shows, since the tracer
        /// renders every surface as physically present. Depth writes, polygon offset, the
        /// transparent flag and render order only order raster draws, so they are ignored;
        /// alpha to coverage converges to the traced per-sample alpha cutoff. Sides follow
        /// the state: `.double` traces both faces and `.front` culls back faces.
        private static func validate(
            _ state: MeshRenderState, material: PhysicallyBasedMaterial, nodeID: UUID
        ) throws {
            // A `.blend` material uses stochastic opacity instead of raster blending.
            if state.blending != .none, material.alphaMode != .blend {
                throw Failure.unsupported(nodeID, "render state blending")
            }
            if state.stencil != nil { throw Failure.unsupported(nodeID, "raster stencil state") }
            if state.colorWriteMask != .all { throw Failure.unsupported(nodeID, "color write masks") }
            if state.alphaHash { throw Failure.unsupported(nodeID, "alpha hashing") }
            if state.side == .back { throw Failure.unsupported(nodeID, "back-face-only sides") }
            guard state.depthTest, state.depthCompare == .less || state.depthCompare == .lessEqual else {
                throw Failure.unsupported(nodeID, "depth tests other than nearest-surface visibility")
            }
        }

        private mutating func surfaceIndex(_ graph: SurfaceShaderGraph?) throws -> UInt32 {
            guard let graph else { return 0 }
            if let existing = graphIDs[graph] { return existing }
            guard graphs.count < PathTracingScene.maximumSurfaceGraphCount else { throw Failure.surfaceGraphLimit }
            graphs.append(graph)
            let index = UInt32(graphs.count)
            graphIDs[graph] = index
            return index
        }

        mutating func pack(
            _ packed: PhysicallyBasedMaterial, parameters: SIMD4<Float>, instanceTint: SIMD4<Float>,
            readsVertexColors: Bool, nodeID: UUID
        ) throws -> Material {
            // Live factors can turn an authored layer off and back on. Reserve its
            // map even when the current material omits the layer, with its authored sampler.
            // This lets `updateMaterials()` reuse the prepared table.
            let material: PhysicallyBasedMaterial
            if let live = packed.liveFactors {
                for binding in (packed.pathTracingMaps + packed.pathTracingLayerMaps + packed.pathTracingLobeMaps).compactMap(\.self) {
                    let sampling = try SamplingModes.description(of: binding, nodeID: nodeID)
                    _ = try packTexture(binding.texture, sampling: sampling, nodeID: nodeID)
                }
                let id = ObjectIdentifier(live)
                let snapshot = liveSnapshots[id] ?? live.snapshot
                liveSnapshots[id] = snapshot
                material = snapshot.material
            } else {
                material = packed
            }
            let maps = material.pathTracingMaps
            let transmission = material.transmission
            let coat = material.clearcoat
            if material.alphaTexture != nil {
                throw Failure.unsupported(nodeID, "alpha maps")
            }

            // The raster tint multiplies parameters (without a graph) and the instance tint into base color.
            let drawTint = material.surfaceGraph == nil ? parameters * instanceTint : instanceTint
            // Raster saturates the tint after multiplying in the vertex color, so a colored
            // mesh keeps the unclamped product and the shader saturates it per vertex color.
            let colored = readsVertexColors
            let tint = colored ? drawTint : simd_clamp(drawTint, .zero, SIMD4(repeating: 1))
            let base = tint * material.baseColorFactor
            let graphOpacity = material.surfaceGraph?.outputs.opacity != nil
            let alphaCutoff: Float
            switch material.alphaMode {
            case .opaque:
                alphaCutoff = PathTracingScene.opaqueCoverage
            case .mask(let cutoff):
                alphaCutoff = cutoff
            case .materialAlpha:
                // Raster writes depth and the premultiplied color with its alpha, without blending.
                alphaCutoff = graphOpacity || colored || base.w < 1 || maps[0] != nil
                    ? PathTracingScene.writtenCoverage : PathTracingScene.opaqueCoverage
            case .blend:
                // Alpha below one traces as stochastic opacity; a constant alpha of one stays opaque.
                alphaCutoff = graphOpacity || colored || base.w < 1 || maps[0] != nil
                    ? PathTracingScene.stochasticCoverage : PathTracingScene.opaqueCoverage
            }

            var textureIndices = SIMD4<UInt32>(repeating: .max)
            var textureUVs = SIMD4<UInt32>.zero
            var transforms = InlineArray<28, SIMD4<Float>>(repeating: .zero)
            var modes = SamplingModes()
            for (index, binding) in maps.enumerated() {
                let rows = (binding?.transform ?? .identity).rows
                transforms[2 * index] = rows.0
                transforms[2 * index + 1] = rows.1
                guard let binding else { continue }
                let sampling = try modes.add(binding, map: index, nodeID: nodeID)
                textureIndices[index] = try packTexture(binding.texture, sampling: sampling, nodeID: nodeID)
                textureUVs[index] = binding.textureCoordinateSet.rawValue
            }
            var layerTextures = InlineArray<12, UInt32>(repeating: .max)
            var layerUVs: UInt32 = 0
            for (index, binding) in material.pathTracingLayerMaps.enumerated() {
                let rows = (binding?.transform ?? .identity).rows
                transforms[8 + 2 * index] = rows.0
                transforms[9 + 2 * index] = rows.1
                guard let binding else { continue }
                let sampling = try modes.add(binding, map: 4 + index, nodeID: nodeID)
                layerTextures[index] = try packTexture(binding.texture, sampling: sampling, nodeID: nodeID)
                layerUVs |= binding.textureCoordinateSet.rawValue << UInt32(index)
            }
            let packed = material.packedExtensions
            let extensions = MaterialExtensions(
                specular: packed.specular, sheen: packed.sheen, anisotropy: packed.anisotropy,
                iridescence: packed.iridescence,
                options: SIMD4(
                    packed.options.x, material.hasTransmission ? transmission?.dispersion ?? 0 : 0,
                    material.shadingModel == .physicallyBased && material.specular != nil ? 1 : 0, colored ? 1 : 0
                ),
                textures: layerTextures, coordinates: SIMD4(layerUVs, modes.addressU, modes.addressV, modes.nearest),
                transforms: transforms
            )

            let attenuation = transmission?.attenuationColor ?? SIMD3(repeating: 1)
            let lobes = try packLobes(material, alphaCutoff: alphaCutoff, nodeID: nodeID)
            let volumetric = lobes.flags.x & PathTracingScene.lobeMedium != 0
            return Material(
                baseColor: base,
                factors: SIMD4(
                    material.settings.y, material.settings.x, material.hasTransmission ? transmission?.factor ?? 0 : 0,
                    transmission?.indexOfRefraction ?? 1.5
                ),
                emission: SIMD4(material.pathTracingEmission, alphaCutoff),
                optical: SIMD4(attenuation, transmission?.attenuationDistance ?? 0),
                layers: SIMD4(
                    material.pathTracingNormalScale, coat?.factor ?? 0, coat?.roughness ?? 0,
                    volumetric || (transmission?.thickness ?? 0) > 0 ? 0 : 1
                ),
                textureIndices: textureIndices,
                textureUVs: textureUVs,
                graphParameters: material.surfaceGraph == nil ? .zero : parameters,
                graphResources: try graphResources(material, nodeID: nodeID),
                extensions: extensions,
                lobes: lobes
            )
        }

        // Packs diffuse transmission, diffuse roughness, the clearcoat normal map and the volume's medium.
        private mutating func packLobes(
            _ material: PhysicallyBasedMaterial, alphaCutoff: Float, nodeID: UUID
        ) throws -> MaterialLobes {
            var lobes = MaterialLobes(
                translucency: .zero, scattering: .zero, extinction: .zero,
                surface: SIMD4(
                    material.hasDiffuseRoughness ? material.diffuseRoughness?.factor ?? 0 : 0,
                    material.hasClearcoat && material.clearcoat?.normalTexture != nil ? material.clearcoat?.normalScale ?? 1 : 0,
                    material.hasLightMap ? material.lightMapIntensity : 0, material.packedExtensions.options.z
                ),
                flags: SIMD4(0, material.baseNormalKind, material.coatNormalEncoding.rawValue, material.objectNormalMaps),
                textures: InlineArray(repeating: .max), coordinates: .zero,
                transforms: InlineArray(repeating: .zero)
            )
            var modes = SamplingModes()
            for (index, binding) in material.pathTracingLobeMaps.enumerated() {
                let rows = (binding?.transform ?? .identity).rows
                lobes.transforms[2 * index] = rows.0
                lobes.transforms[2 * index + 1] = rows.1
                guard let binding else { continue }
                let sampling = try modes.add(binding, map: index, nodeID: nodeID)
                lobes.textures[index] = try packTexture(binding.texture, sampling: sampling, nodeID: nodeID)
                lobes.coordinates.x |= binding.textureCoordinateSet.rawValue << UInt32(index)
            }
            lobes.coordinates.y = modes.addressU
            lobes.coordinates.z = modes.addressV
            lobes.coordinates.w = modes.nearest

            var flags: UInt32 = 0
            if let subsurface = material.hasSubsurface ? material.subsurface : nil {
                lobes.scattering = SIMD4(subsurface.color, subsurface.anisotropy)
                if simd_reduce_max(subsurface.meanFreePath) > 0 {
                    // The walk replaces `weight` of the diffuse lobe, entering and leaving untinted.
                    lobes.translucency = SIMD4(1, 1, 1, subsurface.weight)
                    let path = subsurface.meanFreePath
                    let limit = PathTracingScene.maximumSubsurfaceExtinction
                    lobes.extinction = SIMD4(
                        SIMD3((0..<3).map { path[$0] > 0 ? min(1 / path[$0], limit) : limit }), 0
                    )
                    flags |= PathTracingScene.lobeMedium | PathTracingScene.lobeDiffuseBoundary
                    if simd_reduce_max(subsurface.color) > 0 || subsurface.colorTexture != nil {
                        flags |= PathTracingScene.lobeScattering
                    }
                } else {
                    // A zero mean free path scatters at the entry point: diffuse with the subsurface color.
                    lobes.translucency = SIMD4(0, 0, 0, subsurface.weight)
                    flags |= PathTracingScene.lobeSubsurfaceColor
                }
            } else {
                if let diffuse = material.hasDiffuseTransmission ? material.diffuseTransmission : nil {
                    lobes.translucency = SIMD4(diffuse.colorFactor, diffuse.factor)
                }
                if let medium = material.medium {
                    lobes.extinction = SIMD4(medium.extinction, 0)
                    lobes.scattering = SIMD4(medium.multiscatterColor, medium.anisotropy)
                    flags |= PathTracingScene.lobeMedium
                    if medium.boundary == .diffuse { flags |= PathTracingScene.lobeDiffuseBoundary }
                    let scatters = simd_reduce_max(medium.multiscatterColor) > 0 || material.volumeScatter?.multiscatterColorTexture != nil
                    if scatters, simd_reduce_max(medium.extinction) > 0 { flags |= PathTracingScene.lobeScattering }
                    // Shadow rays cross an index-matched boundary whose BSDF is a straight-through delta.
                    if medium.boundary == .dielectric, let transmission = material.transmission,
                       transmission.factor == 1, transmission.factorTexture == nil, transmission.indexOfRefraction == 1,
                       material.settings.x == 0, material.settings.y == 0, !material.hasClearcoat, !material.hasSheen, !material.hasIridescence,
                       !material.hasDiffuseTransmission, material.surfaceGraph == nil,
                       alphaCutoff == PathTracingScene.opaqueCoverage {
                        flags |= PathTracingScene.lobeNullBoundary
                    }
                }
            }
            lobes.flags.x = flags
            if flags & PathTracingScene.lobeNullBoundary != 0 { hasNullBoundaries = true }
            return lobes
        }

        // Places live uniform blocks once per instance and appends the material's texture layers to the table.
        private mutating func graphResources(_ material: PhysicallyBasedMaterial, nodeID: UUID) throws -> SIMD4<UInt32> {
            guard let graph = material.surfaceGraph else { return .zero }
            var uniformOffset: UInt32 = 0
            if let uniforms = material.surfaceUniforms {
                let key = ObjectIdentifier(uniforms)
                if let block = uniformBlockIndices[key] {
                    uniformOffset = UInt32(uniformBlocks[block].offset)
                    uniformBlocks[block].mask |= graph.surfaceUniformMask
                    if graph.surfaceReadsClock { uniformBlocks[block].readsClock = true }
                } else {
                    // Each block holds its declared value slots and then its graph clock.
                    let blockSlots = uniforms.valueSlotCount + 1
                    guard uniformSlotCount + blockSlots <= PathTracingScene.maximumSurfaceUniformSlots else {
                        throw Failure.surfaceUniformLimit
                    }
                    uniformOffset = UInt32(uniformSlotCount)
                    uniformBlockIndices[key] = uniformBlocks.count
                    uniformBlocks.append((uniforms, Int(uniformOffset), graph.surfaceUniformMask, graph.surfaceReadsClock))
                    uniformSlotCount += blockSlots
                }
            }
            let tableOffset = UInt32(graphTextureTable.count)
            for texture in material.surfaceTextures {
                graphTextureTable.append(try packTexture(texture, sampling: nil, nodeID: nodeID))
            }
            let coverageFlags: UInt32 = (graph.hasCoverageOutputs ? 1 : 0) | (graph.hasCutoutOutputs ? 2 : 0)
            return SIMD4(uniformOffset, tableOffset, UInt32(material.surfaceTextures.count), coverageFlags)
        }

        // Adds a table entry for a material map with its sampling, or for a surface graph texture
        // (nil sampling). The entry samples the texture directly unless the table cannot hold it;
        // see `Options.Textures.fullResolution(fallbackSize:)`. Nearest-magnified maps bake with
        // nearest texels on each axis the bake magnifies, so their texel edges stay hard, and
        // filter linearly on each axis it shrinks, so no texels drop out.
        private mutating func packTexture(_ texture: any MTLTexture, sampling: TextureSampling?, nodeID: UUID) throws -> UInt32 {
            let nearest = sampling?.magnification == .nearest
            let key = TextureKey(texture: ObjectIdentifier(texture), sampling: sampling, nearest: nearest)
            if let existing = textureIDs[key] { return existing }
            guard texture.device === device, texture.textureType == .type2D, texture.sampleCount == 1,
                  texture.storageMode != .memoryless, !texture.isFramebufferOnly,
                  texture.usage.isEmpty || texture.usage.contains(.shaderRead) else {
                throw Failure.invalidTexture(nodeID)
            }
            guard entries.count < PathTracingScene.maximumTextureCount else { throw Failure.textureLimit }
            let entry: TextureTableEntry
            if directlySampled(texture, sampling: sampling) {
                let state = try sampler(for: sampling ?? .default)
                if boundIDs.insert(ObjectIdentifier(texture)).inserted { boundTextures.append(texture) }
                entry = TextureTableEntry(texture: texture.gpuResourceID, sampler: state.gpuResourceID, layer: .max, flags: 0)
            } else {
                let bakeKey = BakeKey(texture: ObjectIdentifier(texture), nearest: nearest)
                let layer: UInt32
                if let existing = bakeLayers[bakeKey] {
                    layer = existing
                } else {
                    guard (bakeSources.count + 1) * PathTracingScene.mipTexelCount(textureSize) <= 4_194_304 else {
                        throw Failure.textureLimit
                    }
                    layer = UInt32(bakeSources.count)
                    bakeSources.append(BakeSource(texture: texture, nearest: nearest))
                    bakeLayers[bakeKey] = layer
                }
                entry = TextureTableEntry(texture: MTLResourceID(), sampler: MTLResourceID(), layer: layer, flags: 0)
            }
            let index = UInt32(entries.count)
            entries.append(entry)
            textureIDs[key] = index
            return index
        }

        // A surface graph may filter its textures linearly, so a nil sampling counts as linear.
        private func directlySampled(_ texture: any MTLTexture, sampling: TextureSampling?) -> Bool {
            guard case .fullResolution = options.textures, device.argumentBuffersSupport == .tier2 else { return false }
            let filters = sampling.map { $0.magnification == .linear || $0.minification == .linear || $0.mipFilter == .linear } ?? true
            return !(filters && MaterialTextureFormat.isFloat32(texture.pixelFormat) && !device.supports32BitFloatFiltering)
        }

        // One argument-buffer sampler state per distinct description; linear minification adds anisotropy.
        private mutating func sampler(for sampling: TextureSampling) throws -> any MTLSamplerState {
            if let existing = samplers[sampling] { return existing }
            guard samplers.count < device.maxArgumentBufferSamplerCount else { throw Failure.textureLimit }
            let descriptor = sampling.descriptor
            descriptor.supportArgumentBuffers = true
            if sampling.minification == .linear { descriptor.maxAnisotropy = options.maximumAnisotropy }
            descriptor.label = "Frostfield path texture sampler"
            guard let state = device.makeSamplerState(descriptor: descriptor) else { throw Failure.resource("path texture sampler") }
            samplers[sampling] = state
            return state
        }

        private static func finite(_ value: SIMD4<Float>) -> Bool {
            value.x.isFinite && value.y.isFinite && value.z.isFinite && value.w.isFinite
        }
    }

    static func mipTexelCount(_ size: Int) -> Int {
        var total = 0
        var level = size
        while level > 0 {
            total += level * level
            level /= 2
        }
        return total
    }

    private static func bakedTextureBytes(_ texture: any MTLTexture) -> Int {
        let logicalBytes = texture.arrayLength * mipTexelCount(texture.width) * MemoryLayout<SIMD4<Float16>>.stride
        return max(logicalBytes, texture.allocatedSize)
    }

    static func bake(
        _ sources: [BakeSource], size: Int, queue: any MTLCommandQueue, referencedBytes: Int, budget: Int
    ) throws -> any MTLTexture {
        let device = queue.device
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type2DArray
        descriptor.pixelFormat = .rgba16Float
        descriptor.width = sources.isEmpty ? 1 : size
        descriptor.height = descriptor.width
        descriptor.arrayLength = max(1, sources.count)
        descriptor.mipmapLevelCount = Int.bitWidth - descriptor.width.leadingZeroBitCount
        descriptor.storageMode = sources.isEmpty ? .shared : .private
        descriptor.usage = sources.isEmpty ? .shaderRead : [.shaderRead, .shaderWrite]
        descriptor.hazardTrackingMode = .tracked
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.resource("path texture array") }
        let residentBytes = referencedBytes + (sources.isEmpty ? 0 : bakedTextureBytes(texture))
        guard residentBytes <= budget else { throw Failure.textureMemory(residentBytes, budget) }
        texture.label = "Frostfield linear path textures"
        guard !sources.isEmpty else {
            var white = SIMD4<Float16>(repeating: 1)
            withUnsafePointer(to: &white) {
                let stride = MemoryLayout<SIMD4<Float16>>.stride
                texture.replace(
                    region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, slice: 0,
                    withBytes: $0, bytesPerRow: stride, bytesPerImage: stride
                )
            }
            return texture
        }

        let pipeline: any MTLComputePipelineState
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let shader = library.makeFunction(name: "frostfield_path_bake_texture") else { throw Failure.bakeShader }
            pipeline = try device.makeComputePipelineState(function: shader)
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.bake(error.localizedDescription)
        }
        guard let buffer = queue.makeCommandBuffer(), let encoder = buffer.makeComputeCommandEncoder() else {
            throw Failure.resource("path texture bake encoder")
        }
        buffer.label = "Frostfield path texture bake"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(texture, index: 1)
        let width = min(8, pipeline.maxTotalThreadsPerThreadgroup)
        let height = min(8, pipeline.maxTotalThreadsPerThreadgroup / width)
        let groups = MTLSize(width: (size + width - 1) / width, height: (size + height - 1) / height, depth: 1)
        let groupSize = MTLSize(width: width, height: height, depth: 1)
        for (index, source) in sources.enumerated() {
            var slice = UInt32(index)
            encoder.setTexture(source.texture, index: 0)
            encoder.setBytes(&slice, length: MemoryLayout<UInt32>.size, index: 0)
            // 0 filters linearly, 1 keeps nearest texels on each axis the bake magnifies, and 2
            // point-samples 32-bit float sources on GPUs that cannot filter them.
            let format = source.texture.pixelFormat
            let unfilterable = MaterialTextureFormat.isFloat32(format) && !device.supports32BitFloatFiltering
            var mode: UInt32 = source.nearest ? 1 : unfilterable ? 2 : 0
            encoder.setBytes(&mode, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: groupSize)
        }
        encoder.endEncoding()
        guard let blit = buffer.makeBlitCommandEncoder() else { throw Failure.resource("path mip encoder") }
        blit.generateMipmaps(for: texture)
        blit.endEncoding()
        buffer.commit()
        buffer.waitUntilCompleted()
        guard buffer.status == .completed else { throw Failure.bake(buffer.error?.localizedDescription) }
        return texture
    }

    // Graph uniforms live at the material's offset in the dispatch's uniform slots. Textures are
    // texture table entries, found through the material's graph table: directly sampled textures,
    // or layers of the scene's baked array. Each address mode has its own sampler, so clamp and
    // mirror edges stay exact at every mip level. The level and anisotropy come from the
    // primary UV set's ray-cone gradients.
    private static let pathGraphResources = """
    struct FrostfieldGraphResources {
        constant float4* uniforms;
        device const uint* table;
        texture2d_array<half> maps;
        device const PTTextureBinding* textures;
        uint textureCount;
        uint uniformOffset;
        uint textureOffset;
        float4 gradients[2];
        float3 footprintX, footprintY;
    };

    static inline float4 frostfield_graph_uniform(thread const FrostfieldGraphResources& resources, uint index) {
        return resources.uniforms[resources.uniformOffset + index];
    }

    static inline float4 frostfield_graph_sample(
        thread const FrostfieldGraphResources& resources, uint slot, uint address, uint filtering, float2 uv,
        float4 gradients
    ) {
        constexpr sampler repeatLinear(
            coord::normalized, address::repeat, filter::linear, mip_filter::linear, max_anisotropy(16)
        );
        constexpr sampler repeatNearest(coord::normalized, address::repeat, filter::nearest, mip_filter::nearest);
        constexpr sampler clampLinear(
            coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::linear, max_anisotropy(16)
        );
        constexpr sampler clampNearest(coord::normalized, address::clamp_to_edge, filter::nearest, mip_filter::nearest);
        constexpr sampler mirrorLinear(
            coord::normalized, address::mirrored_repeat, filter::linear, mip_filter::linear, max_anisotropy(16)
        );
        constexpr sampler mirrorNearest(coord::normalized, address::mirrored_repeat, filter::nearest, mip_filter::nearest);
        uint entry = resources.table[resources.textureOffset + slot];
        if (entry >= resources.textureCount) { return float4(1); }
        device const PTTextureBinding& binding = resources.textures[entry];
        gradient2d lod(gradients.xy, gradients.zw);
        bool linear = filtering == 0u;
        if (binding.layer == 0xffffffffu) {
            texture2d<half> map = binding.texture;
            if (address == 1u) { return float4(linear ? map.sample(clampLinear, uv, lod) : map.sample(clampNearest, uv, lod)); }
            if (address == 2u) { return float4(linear ? map.sample(mirrorLinear, uv, lod) : map.sample(mirrorNearest, uv, lod)); }
            return float4(linear ? map.sample(repeatLinear, uv, lod) : map.sample(repeatNearest, uv, lod));
        }
        uint layer = binding.layer;
        texture2d_array<half> maps = resources.maps;
        if (address == 1u) {
            return float4(linear ? maps.sample(clampLinear, uv, layer, lod) : maps.sample(clampNearest, uv, layer, lod));
        }
        if (address == 2u) {
            return float4(linear ? maps.sample(mirrorLinear, uv, layer, lod) : maps.sample(mirrorNearest, uv, layer, lod));
        }
        return float4(linear ? maps.sample(repeatLinear, uv, layer, lod) : maps.sample(repeatNearest, uv, layer, lod));
    }

    static inline float4 frostfield_graph_sample(
        thread const FrostfieldGraphResources& resources, uint slot, uint address, uint filtering, float2 uv
    ) {
        return frostfield_graph_sample(resources, slot, address, filtering, uv, resources.gradients[0]);
    }

    static inline float3 frostfield_graph_bump(
        thread const FrostfieldSurfaceInput& input, thread const FrostfieldGraphResources& resources,
        uint slot, uint address, uint filtering, uint coordinateSet, float4 row0, float4 row1, float scale
    ) {
        float2 uv = frostfieldTransformUV(coordinateSet == 0u ? input.uv0 : input.uv1, row0, row1);
        float4 source = resources.gradients[coordinateSet & 1u];
        float4 gradients = float4(
            dot(row0.xy, source.xy), dot(row1.xy, source.xy), dot(row0.xy, source.zw), dot(row1.xy, source.zw)
        );
        float height = frostfield_graph_sample(resources, slot, address, filtering, uv, gradients).r;
        float2 delta = float2(
            frostfield_graph_sample(resources, slot, address, filtering, uv + gradients.xy, gradients).r,
            frostfield_graph_sample(resources, slot, address, filtering, uv + gradients.zw, gradients).r
        ) - height;
        float3 bumped = frostfieldBumpNormal(input.vertexNormal, resources.footprintX, resources.footprintY, scale * delta);
        return frostfield_graph_to_tangent(input, bumped);
    }
    """

    static func compileSurfaceLibrary(
        graphs: [SurfaceShaderGraph], device: any MTLDevice
    ) throws -> (any MTLLibrary)? {
        guard !graphs.isEmpty else { return nil }
        let marker = "// FROSTFIELD_GENERATED_SURFACE_FUNCTIONS"
        guard let url = Bundle.module.url(forResource: "PathTracingShading", withExtension: "h"),
              let template = try? String(contentsOf: url, encoding: .utf8),
              var source = try? MaterialExtensionsSource.splice(into: BVHRayTriangleSource.spliceMeshBVH(into: template)),
              let insertion = source.range(of: marker) else { throw Failure.surfaceShaderTemplate }
        let entries = try graphs.enumerated().map { offset, graph in
            (index: offset + 1, source: try graph.metalFunction(named: "frostfield_path_graph_\(offset + 1)"))
        }
        let cases = entries.map { entry in
            "case \(entry.index)u: frostfield_path_graph_\(entry.index)(input, resources, surface); break;"
        }.joined(separator: "\n        ")
        let functions = entries.map(\.source).joined(separator: "\n\n")
        let coverage = try graphs.enumerated().filter { $0.element.hasCoverageOutputs }.map { offset, graph in
            (index: offset + 1, source: try graph.metalCoverageFunction(named: "frostfield_path_coverage_\(offset + 1)"))
        }
        let coverageCases = coverage.map { entry in
            "case \(entry.index)u: frostfield_path_coverage_\(entry.index)(input, resources, surface); break;"
        }.joined(separator: "\n        ")
        let coverageDispatch = coverage.isEmpty ? "" : """
        #define FROSTFIELD_HAS_COVERAGE_GRAPHS 1
        \(coverage.map(\.source).joined(separator: "\n\n"))
        static inline void ptCoverageGraph(
            uint index, thread const FrostfieldSurfaceInput& input, thread const FrostfieldGraphResources& resources,
            thread FrostfieldSurfaceOutput& surface
        ) {
            switch (index) {
                \(coverageCases)
                default: break;
            }
        }
        """
        source.replaceSubrange(insertion, with: """
        #define FROSTFIELD_HAS_SURFACE_GRAPHS 1
        \(SurfaceShaderGraph.supportSource)
        \(pathGraphResources)
        \(functions)
        static inline void ptSurfaceGraph(
            uint index, thread const FrostfieldSurfaceInput& input, thread const FrostfieldGraphResources& resources,
            thread FrostfieldSurfaceOutput& surface
        ) {
            switch (index) {
                \(cases)
                default: break;
            }
        }
        \(coverageDispatch)
        """)
        do {
            // Re-preparing geometry with the same graphs reuses the compiled library.
            return try ShaderLibraryCache.library(source: source, device: device)
        } catch {
            throw Failure.surfaceShaderCompilation(ShaderProgram.diagnose(error, source: source))
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case textureSize
        case emptyScene
        case wrongDevice(UUID)
        case unsupported(UUID, String)
        case invalidTint(UUID)
        case invalidGeometry(UUID)
        case degenerateTriangle(UUID, Int)
        case invalidTexture(UUID)
        case triangleLimit
        case materialLimit
        case textureLimit
        case textureMemory(Int, Int)
        case options
        case surfaceGraphLimit
        case surfaceUniformLimit
        case surfaceShaderTemplate
        case surfaceShaderCompilation(String)
        case bvhDepth
        case resource(String)
        case bakeShader
        case bake(String?)
        case instanceLimit
        case skinning(UUID, SkinningError)
        case unknownNode(UUID)
        case structureChanged(UUID)
        case staleGeometry(UUID)
        case rayTracingUnsupported
        case accelerationStructure(String?)

        public var errorDescription: String? {
            switch self {
            case .textureSize: "Path texture bake sizes must be between 16 and 512."
            case .emptyScene: "The path tracing scene needs at least one triangle."
            case .wrongDevice(let id): "Mesh node \(id) uses a different Metal device."
            case .unsupported(let id, let feature): "Path tracing node \(id) does not support \(feature)."
            case .invalidTint(let id): "Mesh node \(id) needs finite material parameters."
            case .invalidGeometry(let id): "Mesh node \(id) exceeds the finite path-tracing geometry range."
            case .degenerateTriangle(let id, let index): "Mesh node \(id) has a degenerate triangle at index \(index)."
            case .invalidTexture(let id): "Mesh node \(id) needs shader-readable textures on the scene's device."
            case .triangleLimit: "A path tracing scene supports up to \(maximumTriangleCount) triangles."
            case .materialLimit: "A path tracing scene supports up to \(maximumMaterialCount) materials."
            case .textureLimit:
                "Use at most \(maximumTextureCount) path texture and sampling pairs, the device's argument-buffer sampler limit, and 32 MiB of baked texels."
            case .textureMemory(let required, let budget):
                "The path tracing textures need \(required) bytes, more than the \(budget)-byte texture memory budget."
            case .options: "Path tracing options need an anisotropy from 1 to 16 and a positive texture memory budget."
            case .surfaceGraphLimit: "A path tracing scene supports up to \(maximumSurfaceGraphCount) distinct surface graphs."
            case .surfaceUniformLimit:
                "A path tracing scene supports up to \(maximumSurfaceUniformSlots) uniform values across its distinct uniform sets."
            case .surfaceShaderTemplate: "The path tracing shader source template is missing."
            case .surfaceShaderCompilation(let detail): "Could not compile the path tracing surface graphs. \(detail)"
            case .bvhDepth: "The path BVH exceeded its traversal stack depth."
            case .resource(let name): "Metal could not allocate \(name)."
            case .bakeShader: "The path texture bake shader is missing."
            case .bake(let detail): detail ?? "Metal could not finish baking the path textures."
            case .instanceLimit: "A path tracing scene supports up to \(maximumInstanceCount) mesh nodes."
            case .skinning(let id, let error):
                "Path tracing node \(id) cannot be skinned. \(error.errorDescription ?? "")"
            case .unknownNode(let id): "Node \(id) is not a mesh node of this path tracing scene."
            case .structureChanged(let id):
                "Mesh node \(id) was added, removed, or changed its mesh or material parameters; prepare a new path tracing scene."
            case .staleGeometry(let id):
                "Skinned node \(id) has a new joint palette; update its geometry before tracing it."
            case .rayTracingUnsupported: "This Metal device does not support ray tracing; use the software acceleration."
            case .accelerationStructure(let detail): detail ?? "Metal could not build the path tracing acceleration structures."
            }
        }
    }
}
