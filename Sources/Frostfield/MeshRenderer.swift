import Foundation
import Metal
import simd

/// The fragment shader receives linear color and writes premultiplied RGBA.
public enum MeshMaterial {
    case unlit
    /// Samples a straight-alpha color texture and multiplies it by the scene's RGBA tint.
    /// Metal decodes sRGB formats on sample. Use top-left UV coordinates.
    case textured(any MTLTexture)
    /// Lambertian white surface. Parameters are direction toward the light (xyz)
    /// and nonnegative intensity (w), in linear color space.
    case lambert
    /// A directional Lambert surface that receives a scene shadow map.
    /// Material parameters are a linear RGBA tint.
    case shadowedLambert
    case physicallyBased(PhysicallyBasedMaterial)
    /// A Three.js-style basic, Lambert, Phong, toon, matcap, normal, depth, distance
    /// or shadow material. Material parameters tint its color and opacity.
    case classic(ClassicMaterial)
    /// The fragment takes the stage-in normal and one read-only float4 at buffer 0.
    /// Its library must have been created on the renderer's device.
    case custom(library: any MTLLibrary, fragment: String)
    /// Replaces both Metal stages and binds the program's declared resources.
    case rawMetal(RawMeshProgram)
}

/// Per-frame state for a prepared mesh. The camera looks down its local negative Z axis.
public struct MeshScene {
    public var transform: SceneTransform
    public var camera: PerspectiveCamera
    public var cameraTransform: SceneTransform
    /// Per-draw material values. A PBR surface graph uses all four as graph inputs.
    public var materialParameters: SIMD4<Float>
    /// Required when rendering this scene with a skinned mesh.
    public var jointPalette: JointPalette?
    /// One weight per morph target of the mesh; nil draws the base mesh.
    public var morphWeights: [Float]?
    /// Stored render textures for surface graphs that read scene color or depth.
    public var viewportSnapshot: SceneViewportSnapshot? = nil

    public init(
        camera: PerspectiveCamera,
        transform: SceneTransform = SceneTransform(),
        cameraTransform: SceneTransform = SceneTransform(),
        materialParameters: SIMD4<Float> = SIMD4(repeating: 1),
        jointPalette: JointPalette? = nil,
        morphWeights: [Float]? = nil
    ) {
        self.camera = camera
        self.transform = transform
        self.cameraTransform = cameraTransform
        self.materialParameters = materialParameters
        self.jointPalette = jointPalette
        self.morphWeights = morphWeights
    }
}

/// The ordering of Metal's normalized depth values for a render pass.
public enum MeshDepthMode: Hashable, Sendable {
    /// Near geometry has smaller depth values. Clear the depth attachment to 1.
    case forward
    /// Near geometry has larger depth values. Clear the depth attachment to 0.
    case reversed
    /// Logarithmic perspective depth, with near at zero and far at one. Clear to 1.
    /// Requires a supported single-sample mesh pipeline and finite camera clip planes.
    /// Orthographic views keep linear depth. The camera's clip projection is forward.
    case logarithmic
}

/// A prepared indexed mesh and render pipeline on a single Metal device.
public final class MeshRenderer {
    public static let depthPixelFormat: MTLPixelFormat = .depth32Float
    public static let stencilDepthPixelFormat: MTLPixelFormat = .depth32Float_stencil8
    /// The most morph targets one mesh can upload. Every draw blends all of its
    /// nonzero weights in the vertex stage, so cost grows with the active count.
    public static let maximumMorphTargetCount = 256

    public let device: any MTLDevice
    /// The format at color attachment zero.
    public let colorPixelFormat: MTLPixelFormat
    /// Pipeline color formats in Metal attachment index order.
    public let colorPixelFormats: [MTLPixelFormat]
    /// The named PBR layout, when this renderer was prepared with one.
    public let surfaceRenderTargets: SurfaceRenderTargets?
    public let depthMode: MeshDepthMode
    /// Whether drawing this material requires a scene graph transmission backdrop.
    public var requiresTransmissionBackdrop: Bool { physicallyBased?.hasTransmission == true }
    /// The indexed source used to prepare this renderer, or the current geometry
    /// when the renderer draws a curve flow or supports dynamic geometry.
    public var geometry: MeshGeometry { curveFlow?.geometry ?? sourceGeometry }
    private var sourceGeometry: MeshGeometry
    /// The direct draw range shared by nodes using this renderer. Change it
    /// between encodes. The next draw captures the range without rebuilding GPU
    /// buffers, and picking uses the same complete triangles.
    public var drawRange: MeshDrawRange = .all {
        didSet { if drawRange != oldValue { pickingBVH = MeshBVHCache() } }
    }

    var hasActiveTriangles: Bool { !drawRange.resolve(indexCount: sourceGeometry.indices.count).isEmpty }

    func geometryInDrawRange(_ geometry: MeshGeometry) -> MeshGeometry? {
        let range = drawRange.resolve(indexCount: geometry.indices.count)
        guard !range.isEmpty else { return nil }
        return geometry.restrictingIndices(to: range)
    }
    /// The latest accepted dynamic geometry revision. Static renderers return nil.
    public private(set) var geometryRevision: UInt64?
    /// True when the latest dynamic upload wrote only recorded vertex spans into an existing buffer.
    package private(set) var lastVertexUploadWasPartial = false
    private var geometryPatches: [Range<Int>] = []
    private var patchBaseRevision: UInt64 = 0
    private var motionTopologyRevision: UInt64 = 0
    /// The color pass's blending, depth, stencil, face and coverage state.
    public let renderState: MeshRenderState
    /// Whether this mesh's color pipeline requires a combined depth-stencil attachment.
    public let usesStencilAttachment: Bool
    /// The color and depth attachments' sample count; one unless prepared for MSAA.
    public let sampleCount: Int
    /// Picking tests front faces of front-sided meshes and both faces otherwise.
    /// Transparent meshes draw after the rest, back to front.
    private var isBlended: Bool { renderState.transparent }
    private var drawsAfterOpaque: Bool { isBlended || requiresTransmissionBackdrop }
    private let classic: (material: ClassicMaterial, prepared: ClassicMaterial.Prepared)?
    /// Per-vertex linear RGBA at vertex buffer 10, when the material reads vertex colors.
    private let vertexColors: (any MTLBuffer)?
    /// Whether the material multiplies the geometry's vertex colors into its base color.
    let readsVertexColors: Bool
    /// Whether the fragment reads the graph's punctual lights and their shadows.
    private var usesPunctualLights: Bool { physicallyBased != nil || classic?.material.usesPunctualLights == true }
    private var usesNativeLambert: Bool {
        switch material {
        case .lambert, .shadowedLambert: true
        default: false
        }
    }
    /// The material this renderer was prepared with.
    public let material: MeshMaterial
    private var rawProgram: RawMeshProgram? {
        guard case .rawMetal(let program) = material else { return nil }
        return program
    }
    public let skin: MeshSkin?
    /// Offsets blended per draw on the GPU with the node's `morphWeights`.
    public let morphTargets: MeshMorphTargets?
    /// Whether scene nodes may draw this mesh with an `InstanceSet`.
    public let supportsInstancing: Bool
    /// The local box around `geometry`, updated with curve flow or dynamic vertices.
    /// Culling uses it for meshes that are not skinned or deformed by a surface graph.
    public var bounds: AxisAlignedBox { curveFlow?.bounds ?? sourceBounds }
    private var sourceBounds: AxisAlignedBox
    private let skinInfluences: (any MTLBuffer)?
    private let morph: PreparedMorph?
    private let pipeline: any MTLRenderPipelineState
    private var preparedToonOutline: ToonOutlinePipeline?
    /// Depth pipelines for specialized vertex stages, which the shadow maps do not prepare.
    private let shadowPipelines: ShadowPipelines?
    private let graphShadow: (pipeline: any MTLRenderPipelineState, cameraNode: MaterialGraph.NodeID?)?
    private let graphTransmittedShadow: DirectionalTransmissionPass.GraphCaster?
    private let colorTexture: (any MTLTexture)?
    private let textureHelperOptions: SIMD4<UInt32>?
    private let indirectColorArguments: BatchTextureArguments?
    private let physicallyBased: PhysicallyBasedMaterial?
    private let viewportInputs: SurfaceViewportInputs
    private let readsSoftParticles: Bool
    var pathTracingMaterial: PhysicallyBasedMaterial? { physicallyBased }
    private let pbrBindings: PhysicallyBasedMaterial.Prepared?
    private let receivesShadows: Bool
    // An unsampled single-slice array bound to PBR fragments when no punctual shadow maps are supplied.
    private let punctualShadowFallback: (any MTLTexture)?
    private let lightTableCache = LightTextureSet.Cache()
    let screenInputsFallback: PBRScreenInputs.Binding?
    // The surface attribute pipeline's vertex stage and fragment, prepared on first use.
    private let attributeVertex: any MTLFunction
    private let attributeFragment: (library: any MTLLibrary, name: String, constants: MTLFunctionConstantValues?)
    private var attributePipeline: (any MTLRenderPipelineState)?
    private let sceneEnvironmentSource: SceneEnvironmentPipelineSource?
    private var sceneEnvironmentColorPipeline: (any MTLRenderPipelineState)?
    private var sceneEnvironmentAttributePipeline: (any MTLRenderPipelineState)?
    private let depthState: any MTLDepthStencilState
    private let vertices: any MTLBuffer
    private let deformation: MeshDeformation?
    private var curveFlow: CurveFlow?
    /// How many times scene encodes have run this mesh's graph deformation.
    package var deformationEncodeCount: Int { deformation?.encodeCount ?? 0 }
    private let indices: any MTLBuffer
    private let dynamicGeometry: DynamicGeometryBuffers?
    private var indirectValidation: MeshIndirectDraws.Validator?

    func recordResources(in inventory: inout SceneResourceInventory) {
        guard inventory.recordGeometry(self) else { return }
        if let dynamicGeometry {
            dynamicGeometry.recordResources(in: &inventory)
        } else {
            inventory.recordBuffer(vertices)
            inventory.recordBuffer(deformation?.completedVertices)
            inventory.recordBuffer(indices)
            inventory.recordBuffer(vertexColors)
        }
        inventory.recordBuffer(skinInfluences)
        inventory.recordBuffer(morph?.offsets)
        indirectColorArguments?.recordResources(in: &inventory)
        inventory.recordPipeline(pipeline)
        inventory.recordPipeline(attributePipeline)
        inventory.recordPipeline(sceneEnvironmentColorPipeline)
        inventory.recordPipeline(sceneEnvironmentAttributePipeline)
        inventory.recordPipeline(shadowPipelines?.opaque)
        inventory.recordPipeline(shadowPipelines?.masked)
        inventory.recordPipeline(graphTransmittedShadow?.pipeline)
        inventory.recordTexture(colorTexture)
        if let rawProgram {
            for texture in rawProgram.vertexBindings.textures.values { inventory.recordTexture(texture) }
            for texture in rawProgram.fragmentBindings.textures.values { inventory.recordTexture(texture) }
        }
        inventory.recordTexture(punctualShadowFallback)
        inventory.recordBuffer(screenInputsFallback?.arguments)
        if let classic = classic?.prepared {
            for texture in classic.textures { inventory.recordTexture(texture) }
            inventory.recordTexture(classic.environment)
            inventory.recordTexture(classic.shadowFallback)
            for texture in classic.graphTextures { inventory.recordTexture(texture) }
        }
        pbrBindings?.recordResources(in: &inventory)
    }

    /// Prepares the mesh, its material pipeline and optional vertex features.
    ///
    /// Morph targets blend in the bind pose before skinning. They need one
    /// offset per vertex, at most `maximumMorphTargetCount` targets, base
    /// tangents for tangent offsets, and no surface-graph position offset.
    /// Their offsets upload as three packed float sections, `12 × targets ×
    /// vertices` bytes per attribute that any target uses. `supportsInstancing`
    /// prepares vertex stages that read an `InstanceSet`; draws without
    /// instances then bind one identity instance. Both features prepare their
    /// own shadow depth pipelines.
    ///
    /// `renderState` defaults to the material's own: physically based `.blend`
    /// materials draw with ``MeshRenderState/transparent`` and double-sided ones with
    /// `.double` sides; everything else draws ``MeshRenderState/opaque`` (classic
    /// shadow materials blend). A `sampleCount` above one prepares the pipeline for
    /// multisampled color and depth attachments with that count. Physically based and
    /// classic materials multiply vertex colors into their base color; the unlit,
    /// textured, Lambert and custom fragments ignore them.
    ///
    /// A stencil state automatically selects the combined depth-stencil layout.
    /// Set `usesStencilAttachment` on meshes with no stencil state that share
    /// that pass. Both Metal attachment slots must reference the same
    /// `stencilDepthPixelFormat` texture. Its sample count matches color.
    /// Multisampled color can resolve into a separate texture. The combined
    /// depth-stencil attachment stays at its authored sample count.
    /// This layout supports forward or reversed depth, skinning, morph targets,
    /// graph position offsets, and graph nodes with ordinary instance sets.
    public convenience init(
        device: any MTLDevice,
        colorPixelFormat: MTLPixelFormat,
        geometry: MeshGeometry,
        material: MeshMaterial = .unlit,
        depthMode: MeshDepthMode = .forward,
        skin: MeshSkin? = nil,
        morphTargets: MeshMorphTargets? = nil,
        supportsInstancing: Bool = false,
        renderState: MeshRenderState? = nil,
        sampleCount: Int = 1,
        usesStencilAttachment: Bool = false
    ) throws {
        try self.init(
            device: device, colorPixelFormat: colorPixelFormat, geometry: geometry, material: material, depthMode: depthMode,
            skin: skin, morphTargets: morphTargets, supportsInstancing: supportsInstancing, renderState: renderState,
            sampleCount: sampleCount, indirectCommands: false, usesStencilAttachment: usesStencilAttachment
        )
    }

    /// Prepares a renderer whose indexed geometry can change within fixed capacities.
    /// Each encode captures the current revision and retains its buffers until the
    /// command completes or is discarded. Up to three distinct revisions can have
    /// draws in flight. Serialize edits and encoding on one thread or actor.
    ///
    /// Supports the normal scene color and depth passes, or a combined stencil
    /// color pass when prepared with stencil. Use a separate renderer for skinning,
    /// morphs, GPU indirect arguments, instances or vertex graph deformation.
    public convenience init(
        device: any MTLDevice, colorPixelFormat: MTLPixelFormat, dynamicGeometry geometry: MeshGeometry,
        capacity: DynamicGeometryCapacity, material: MeshMaterial = .unlit, depthMode: MeshDepthMode = .forward,
        renderState: MeshRenderState? = nil, sampleCount: Int = 1, usesStencilAttachment: Bool = false
    ) throws {
        try self.init(
            device: device, colorPixelFormat: colorPixelFormat, geometry: geometry, material: material,
            depthMode: depthMode, supportsInstancing: false, renderState: renderState, sampleCount: sampleCount,
            indirectCommands: false, dynamicCapacity: capacity, usesStencilAttachment: usesStencilAttachment
        )
    }

    /// Replaces the current geometry and bounds without rebuilding the material
    /// pipeline. The next encode uploads the revision when it has a free slot.
    /// Already encoded draws keep their previous vertices, indices and count.
    /// The replacement starts new raster motion history for this mesh.
    public func updateGeometry(_ geometry: MeshGeometry) throws {
        try updateGeometry(geometry, preservingVertexIdentity: false)
    }

    /// Updates dynamic geometry. Pass `true` when each vertex slot still denotes
    /// the same vertex as in the previous revision, so raster motion can follow it.
    /// Preserving identity requires the same vertex count and index sequence.
    public func updateGeometry(_ geometry: MeshGeometry, preservingVertexIdentity: Bool) throws {
        guard let dynamicGeometry, let revision = geometryRevision else { throw Failure.dynamicGeometryNotPrepared }
        try dynamicGeometry.capacity.validate(geometry)
        try validateMaterialCoordinates(geometry)
        if preservingVertexIdentity {
            guard geometry.vertices.count == sourceGeometry.vertices.count, geometry.indices == sourceGeometry.indices else {
                throw Failure.dynamicGeometryTopology
            }
        }
        let (next, overflow) = revision.addingReportingOverflow(1)
        guard !overflow else { throw Failure.dynamicGeometryRevision }
        sourceGeometry = geometry
        sourceBounds = geometry.bounds
        geometryRevision = next
        geometryPatches = []
        patchBaseRevision = next
        if !preservingVertexIdentity { motionTopologyRevision = next }
        pickingBVH = MeshBVHCache()
    }

    /// Replaces a contiguous span of the current dynamic mesh and uploads only that span
    /// when the previous vertices are still in an idle buffer. Indices and the other
    /// vertices stay. The next encode uses the new bounds. A renderer that is not dynamic
    /// throws `Failure.dynamicGeometryNotPrepared`.
    public func updateVertices(_ vertices: [MeshVertex], startingAt start: Int) throws {
        guard dynamicGeometry != nil, let revision = geometryRevision else { throw Failure.dynamicGeometryNotPrepared }
        guard start >= 0, !vertices.isEmpty, start <= sourceGeometry.vertices.count - vertices.count else {
            throw Failure.dynamicGeometryRange
        }
        var replaced = sourceGeometry.vertices
        for (offset, vertex) in vertices.enumerated() {
            replaced[start + offset] = vertex
        }
        let geometry = try sourceGeometry.replacingVertices(replaced)
        try dynamicGeometry?.capacity.validate(geometry)
        let (next, overflow) = revision.addingReportingOverflow(1)
        guard !overflow else { throw Failure.dynamicGeometryRevision }
        if geometryPatches.isEmpty { patchBaseRevision = revision }
        geometryPatches.append(start..<(start + vertices.count))
        sourceGeometry = geometry
        sourceBounds = geometry.bounds
        geometryRevision = next
        pickingBVH = MeshBVHCache()
    }

    /// Prepares a custom fragment for two to eight color attachments.
    /// Declare the native shader outputs at the corresponding `[[color(i)]]`
    /// indices and keep the custom material's read-only `float4` binding at buffer 0.
    /// The same write mask applies to every output. Use an unblended render state.
    ///
    /// Passes use equal-sized, single-sample 2D color textures at mip and slice
    /// zero and a matching Depth32Float texture. Metal validates device-specific
    /// format and aggregate attachment limits when preparing the pipeline.
    public convenience init(
        device: any MTLDevice,
        colorPixelFormats: [MTLPixelFormat],
        geometry: MeshGeometry,
        material: MeshMaterial,
        depthMode: MeshDepthMode = .forward,
        skin: MeshSkin? = nil,
        morphTargets: MeshMorphTargets? = nil,
        supportsInstancing: Bool = false,
        renderState: MeshRenderState? = nil
    ) throws {
        switch material {
            case .custom, .rawMetal: break
            default: throw Failure.multipleRenderTargets
        }
        guard (2...8).contains(colorPixelFormats.count), let first = colorPixelFormats.first,
              !colorPixelFormats.contains(.invalid),
              (renderState ?? .opaque).blending == .none else {
            throw Failure.multipleRenderTargets
        }
        try self.init(
            device: device, colorPixelFormat: first, geometry: geometry, material: material, depthMode: depthMode,
            skin: skin, morphTargets: morphTargets, supportsInstancing: supportsInstancing, renderState: renderState,
            sampleCount: 1, indirectCommands: false, additionalColorPixelFormats: Array(colorPixelFormats.dropFirst())
        )
    }

    /// Prepares PBR color and named secondary surface textures in the same pass.
    /// Surface-graph changes and cutout coverage apply to every output.
    public convenience init(
        device: any MTLDevice,
        surfaceRenderTargets: SurfaceRenderTargets,
        geometry: MeshGeometry,
        material: PhysicallyBasedMaterial,
        depthMode: MeshDepthMode = .forward,
        skin: MeshSkin? = nil,
        morphTargets: MeshMorphTargets? = nil,
        supportsInstancing: Bool = false,
        renderState: MeshRenderState? = nil
    ) throws {
        let meshMaterial = MeshMaterial.physicallyBased(material)
        let state = renderState ?? MeshRenderState.defaultState(for: meshMaterial)
        guard state.blending == .none, material.alphaMode != .blend, !material.hasTransmission else {
            throw SurfaceRenderTargets.Failure.material
        }
        try self.init(
            device: device, colorPixelFormat: surfaceRenderTargets.colorPixelFormat, geometry: geometry,
            material: meshMaterial, depthMode: depthMode, skin: skin, morphTargets: morphTargets,
            supportsInstancing: supportsInstancing, renderState: state, sampleCount: 1, indirectCommands: false,
            surfaceRenderTargets: surfaceRenderTargets
        )
    }

    init(
        device: any MTLDevice, colorPixelFormat: MTLPixelFormat, geometry: MeshGeometry, material: MeshMaterial,
        depthMode: MeshDepthMode, skin: MeshSkin? = nil, morphTargets: MeshMorphTargets? = nil,
        supportsInstancing: Bool, renderState: MeshRenderState?, sampleCount: Int, indirectCommands: Bool,
        textureHelper: TextureHelperBinding? = nil, dynamicCapacity: DynamicGeometryCapacity? = nil,
        additionalColorPixelFormats: [MTLPixelFormat] = [], surfaceRenderTargets: SurfaceRenderTargets? = nil,
        sharingGeometryWith source: MeshRenderer? = nil, usesStencilAttachment: Bool = false
    ) throws {
        if case .rawMetal = material, skin != nil || morphTargets != nil || supportsInstancing || indirectCommands {
            throw RawMeshProgram.Failure.unsupportedGeometry
        }
        if let dynamicCapacity {
            guard skin == nil, morphTargets == nil, !supportsInstancing, !indirectCommands, textureHelper == nil else {
                throw Failure.unsupportedDynamicGeometry
            }
            if case .physicallyBased(let material) = material, material.surfaceGraph?.hasVertexOutputs == true {
                throw Failure.unsupportedDynamicGeometry
            }
            try dynamicCapacity.validate(geometry)
        }
        guard RendererCapabilities.supportsRasterSampleCount(sampleCount, on: device) else {
            throw Failure.sampleCount
        }
        let state = renderState ?? MeshRenderState.defaultState(for: material)
        let usesStencilAttachment = usesStencilAttachment || state.stencil != nil
        if usesStencilAttachment {
            guard depthMode != .logarithmic,
                  !indirectCommands, textureHelper == nil, source == nil,
                  additionalColorPixelFormats.isEmpty, surfaceRenderTargets == nil else { throw Failure.stencilPass }
            switch material {
            case .rawMetal, .shadowedLambert:
                throw Failure.stencilPass
            case .physicallyBased(let material) where material.hasTransmission:
                throw Failure.stencilPass
            default:
                break
            }
        }
        let readsVertexColors: Bool
        switch material {
        case .physicallyBased, .classic: readsVertexColors = true
        default: readsVertexColors = false
        }
        try state.validate(sampleCount: sampleCount, supportsAlphaControls: readsVertexColors)
        let usesLogDepth = depthMode == .logarithmic
        if usesLogDepth {
            guard sampleCount == 1, !indirectCommands, textureHelper == nil, state.depthOffset == nil,
                  additionalColorPixelFormats.isEmpty, surfaceRenderTargets == nil else {
                throw Failure.logarithmicDepthPass
            }
            switch material {
            case .unlit, .textured, .lambert, .shadowedLambert, .physicallyBased: break
            default: throw Failure.logarithmicDepthPass
            }
        }
        let depthSuffix = usesLogDepth ? "_log" : ""
        let colored = readsVertexColors && (geometry.hasVertexColors || dynamicCapacity != nil || source?.dynamicGeometry != nil)
        if let skin, skin.influences.count != geometry.vertices.count { throw SkinningError.influenceCount }
        if let morphTargets {
            try Self.validate(morphTargets, geometry: geometry, material: material)
        }
        let library: any MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw Failure.library(error.localizedDescription)
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Frostfield mesh"
        let colorPixelFormats = surfaceRenderTargets?.colorPixelFormats ?? ([colorPixelFormat] + additionalColorPixelFormats)
        for (index, format) in colorPixelFormats.enumerated() {
            let attachment = descriptor.colorAttachments[index]!
            attachment.pixelFormat = format
            attachment.writeMask = state.colorWriteMask
        }
        descriptor.depthAttachmentPixelFormat = usesStencilAttachment ? Self.stencilDepthPixelFormat : Self.depthPixelFormat
        if usesStencilAttachment { descriptor.stencilAttachmentPixelFormat = Self.stencilDepthPixelFormat }
        descriptor.rasterSampleCount = sampleCount
        if indirectCommands {
            descriptor.supportIndirectCommandBuffers = true
            descriptor.inputPrimitiveTopology = .triangle
        }
        state.configure(descriptor)
        let specialization = VertexSpecialization(
            skinned: skin != nil, morphed: morphTargets != nil, instanced: supportsInstancing, colored: colored
        )
        if specialization.isNeeded {
            let objectNormals = switch material {
            case .physicallyBased(let pbr): pbr.objectNormalMaps != 0
            case .classic(let classic): classic.normalMap?.normalSpace == .objectSpace
            default: false
            }
            descriptor.vertexFunction = try specialization.function(
                "frostfield_mesh_vertex_specialized", in: library, objectNormals: objectNormals
            )
        } else {
            let vertexFunction = skin == nil ? "frostfield_mesh_vertex" : "frostfield_skinned_mesh_vertex"
            descriptor.vertexFunction = try Self.function(vertexFunction, in: library)
        }

        let colorTexture: (any MTLTexture)?
        var indirectColorArguments: BatchTextureArguments?
        var physicallyBased: PhysicallyBasedMaterial?
        var viewportInputs: SurfaceViewportInputs = []
        var readsSoftParticles = false
        var pbrBindings: PhysicallyBasedMaterial.Prepared?
        var graphShadow: (pipeline: any MTLRenderPipelineState, cameraNode: MaterialGraph.NodeID?)?
        var graphTransmittedShadow: DirectionalTransmissionPass.GraphCaster?
        var punctualShadowFallback: (any MTLTexture)?
        var screenInputsFallback: PBRScreenInputs.Binding?
        var attributeFragment: (library: any MTLLibrary, name: String, constants: MTLFunctionConstantValues?) =
            (library, "frostfield_mesh_attributes" + depthSuffix, nil)
        var sceneEnvironmentSource: SceneEnvironmentPipelineSource?
        var receivesShadows = false
        var classic: (material: ClassicMaterial, prepared: ClassicMaterial.Prepared)?
        let preparedMaterial: MeshMaterial = textureHelper.map {
            .custom(library: library, fragment: $0.fragment)
        } ?? material
        switch preparedMaterial {
        case .textured(let texture):
            guard texture.device === device else { throw Failure.wrongDevice }
            guard Self.isColorTexture(texture) else { throw Failure.colorTexture }
            colorTexture = texture
            let fragment = indirectCommands ? "frostfield_mesh_textured_indirect" : "frostfield_mesh_textured" + depthSuffix
            let function = try Self.function(fragment, in: library)
            descriptor.fragmentFunction = function
            pipeline = try Self.makePipeline(device: device, descriptor: descriptor)
            if indirectCommands { indirectColorArguments = try BatchTextureArguments(texture: texture, fragment: function) }
        case .unlit:
            colorTexture = nil
            descriptor.fragmentFunction = try Self.function("frostfield_mesh_unlit" + depthSuffix, in: library)
            pipeline = try Self.makePipeline(device: device, descriptor: descriptor)
        case .lambert:
            colorTexture = nil
            descriptor.fragmentFunction = try Self.function("frostfield_mesh_lambert" + depthSuffix, in: library)
            pipeline = try Self.makePipeline(device: device, descriptor: descriptor)
        case .shadowedLambert:
            colorTexture = nil
            receivesShadows = true
            descriptor.fragmentFunction = try Self.function("frostfield_mesh_shadowed" + depthSuffix, in: library)
            pipeline = try Self.makePipeline(device: device, descriptor: descriptor)
        case .physicallyBased(let material):
            physicallyBased = material
            viewportInputs = material.surfaceGraph?.viewportInputs ?? []
            readsSoftParticles = material.surfaceGraph?.readsSoftParticles ?? false
            receivesShadows = material.receivesShadows
            colorTexture = nil
            pbrBindings = try PhysicallyBasedMaterial.Prepared(material: material, device: device)
            punctualShadowFallback = try Self.makePunctualShadowFallback(device: device)
            screenInputsFallback = try .init(device: device, encoder: PBRScreenInputs.Binding.argumentEncoder(on: device))
            let fragment = receivesShadows ? "frostfield_mesh_pbr_shadowed" : "frostfield_mesh_pbr"
            let constants = MTLFunctionConstantValues()
            var hasAlphaMap = material.alphaTexture != nil
            constants.setConstantValue(&hasAlphaMap, type: .bool, index: 4)
            var hasRasterMaps = material.hasLayers || material.hasLightMap
            constants.setConstantValue(&hasRasterMaps, type: .bool, index: 9)
            Self.setCoverageConstants(constants, colored: colored, state: state)
            var hasEnvironment = material.environment != nil
            constants.setConstantValue(&hasEnvironment, type: .bool, index: 0)
            var hasClearcoat = material.hasClearcoat
            constants.setConstantValue(&hasClearcoat, type: .bool, index: 1)
            var hasTransmission = material.hasTransmission
            constants.setConstantValue(&hasTransmission, type: .bool, index: 2)
            var hasLayers = material.hasLayers
            constants.setConstantValue(&hasLayers, type: .bool, index: 3)
            var hasObjectNormalMaps = material.objectNormalMaps != 0
            constants.setConstantValue(&hasObjectNormalMaps, type: .bool, index: 8)
            do {
                let fragmentLibrary: any MTLLibrary
                if material.surfaceGraph != nil || surfaceRenderTargets != nil || usesLogDepth {
                    fragmentLibrary = try Self.surfaceLibrary(
                        material.surfaceGraph, targets: surfaceRenderTargets, logarithmicDepth: usesLogDepth,
                        viewportInputs: viewportInputs, readsSoftParticles: readsSoftParticles, device: device
                    )
                } else {
                    fragmentLibrary = library
                }
                descriptor.fragmentFunction = try fragmentLibrary.makeFunction(name: fragment, constantValues: constants)
                attributeFragment = (fragmentLibrary, "frostfield_mesh_pbr_attributes", constants)
                if material.shadingModel == .physicallyBased, material.environment == nil {
                    sceneEnvironmentSource = SceneEnvironmentPipelineSource(
                        descriptor: descriptor, library: fragmentLibrary, fragment: fragment, constants: constants
                    )
                }
                let masked = if case .mask = material.alphaMode { true } else { false }
                if let graph = material.surfaceGraph {
                    let depthOnly = !state.transparent && !material.hasTransmission
                        && (graph.hasShadowCutoutOutputs || (graph.outputs.opacity != nil && masked))
                    let castsColored = graph.hasCastShadowOutput
                    if depthOnly || castsColored {
                        let shadowConstants = MTLFunctionConstantValues()
                        var disabled = false
                        var usesColors = colored
                        shadowConstants.setConstantValue(&hasAlphaMap, type: .bool, index: 4)
                        shadowConstants.setConstantValue(&usesColors, type: .bool, index: 5)
                        shadowConstants.setConstantValue(&disabled, type: .bool, index: 6)
                        shadowConstants.setConstantValue(&disabled, type: .bool, index: 7)
                        shadowConstants.setConstantValue(&hasObjectNormalMaps, type: .bool, index: 8)
                        let depth = MTLRenderPipelineDescriptor()
                        depth.vertexFunction = descriptor.vertexFunction
                        depth.depthAttachmentPixelFormat = .depth32Float
                        depth.inputPrimitiveTopology = .triangle
                        if depthOnly {
                            depth.label = "Frostfield surface graph shadow depth"
                            depth.fragmentFunction = try fragmentLibrary.makeFunction(
                                name: "frostfield_mesh_graph_shadow", constantValues: shadowConstants
                            )
                            depth.supportIndirectCommandBuffers = indirectCommands
                            graphShadow = (try Self.makePipeline(device: device, descriptor: depth), graph.shadowCoverageCameraNode)
                        }
                        if castsColored {
                            depth.label = "Frostfield surface graph colored shadow"
                            depth.fragmentFunction = try fragmentLibrary.makeFunction(
                                name: "frostfield_mesh_graph_transmitted_shadow", constantValues: shadowConstants
                            )
                            depth.colorAttachments[0].pixelFormat = .rgba32Float
                            depth.supportIndirectCommandBuffers = false
                            graphTransmittedShadow = .init(
                                pipeline: try Self.makePipeline(device: device, descriptor: depth), material: material
                            )
                        }
                    }
                }
            } catch {
                throw Failure.pipeline(error.localizedDescription)
            }
            pipeline = try Self.makePipeline(device: device, descriptor: descriptor)
        case .classic(let material):
            var depthOutput = material.depthOutputCode
            if depthOutput != 0 {
                guard sampleCount == 1, !state.blends, !state.alphaToCoverage else {
                    throw Failure.renderState("Explicit depth and distance outputs require a single sample and no blending.")
                }
                guard material.supportsDepthOutput(colorPixelFormat) else {
                    throw Failure.depthOutputPixelFormat
                }
            }
            colorTexture = nil
            receivesShadows = material.receivesShadows
            punctualShadowFallback = try Self.makePunctualShadowFallback(device: device)
            classic = (material, try ClassicMaterial.Prepared(material: material, device: device, depthMode: depthMode))
            let constants = MTLFunctionConstantValues()
            Self.setCoverageConstants(constants, colored: colored, state: state)
            var model = material.modelCode
            constants.setConstantValue(&model, type: .uint, index: 40)
            var blends = state.blends
            constants.setConstantValue(&blends, type: .bool, index: 41)
            var directional = material.receivesShadows
            constants.setConstantValue(&directional, type: .bool, index: 42)
            constants.setConstantValue(&depthOutput, type: .uint, index: 43)
            var detail = material.hasSurfaceDetail
            var objectNormal = material.normalMap?.normalSpace == .objectSpace
            constants.setConstantValue(&objectNormal, type: .bool, index: 8)
            constants.setConstantValue(&detail, type: .bool, index: 44)
            do {
                let fragmentLibrary: any MTLLibrary
                if let graph = material.surfaceGraph {
                    fragmentLibrary = try Self.classicNodeLibrary(graph, device: device)
                } else {
                    fragmentLibrary = library
                }
                descriptor.fragmentFunction = try fragmentLibrary.makeFunction(name: "frostfield_mesh_classic", constantValues: constants)
                let attributeConstants = MTLFunctionConstantValues()
                Self.setCoverageConstants(attributeConstants, colored: colored, state: state)
                attributeConstants.setConstantValue(&objectNormal, type: .bool, index: 8)
                attributeConstants.setConstantValue(&detail, type: .bool, index: 44)
                attributeFragment = (library, "frostfield_mesh_classic_attributes", attributeConstants)
            } catch {
                throw Failure.pipeline(error.localizedDescription)
            }
            pipeline = try Self.makePipeline(device: device, descriptor: descriptor)
        case .custom(let customLibrary, let fragment):
            guard customLibrary.device === device else { throw Failure.wrongDevice }
            descriptor.fragmentFunction = try Self.function(fragment, in: customLibrary)
            if let textureHelper {
                colorTexture = textureHelper.texture
                pipeline = try Self.makePipeline(device: device, descriptor: descriptor)
                break
            }
            colorTexture = nil
            var reflection: MTLRenderPipelineReflection?
            do {
                pipeline = try device.makeRenderPipelineState(
                    descriptor: descriptor, options: [.bindingInfo, .bufferTypeInfo], reflection: &reflection
                )
            } catch {
                throw Failure.pipeline(error.localizedDescription)
            }
            let bindings = reflection?.fragmentBindings.filter { $0.isUsed && $0.isArgument } ?? []
            guard bindings.count == 1,
                  let parameters = bindings.first as? any MTLBufferBinding,
                  parameters.index == 0, parameters.access == .readOnly,
                  parameters.bufferDataType == .float4, parameters.bufferDataSize == 16 else {
                throw Failure.fragmentBindings(fragment)
            }
        case .rawMetal(let program):
            colorTexture = nil
            pipeline = try program.pipeline(on: device, descriptor: descriptor)
        }

        // Blended and transmissive meshes skip opaque shadows, but selection depth passes
        // (`encodeDepthOnly`) still need their specialized vertex stage.
        if specialization.isNeeded {
            var masked = false
            if case .mask? = physicallyBased?.alphaMode { masked = true }
            shadowPipelines = try ShadowPipelines(
                specialization: specialization, masked: masked, library: library, device: device, indirectCommands: indirectCommands
            )
        } else {
            shadowPipelines = nil
        }

        let depth = state.depthDescriptor(depthMode: depthMode)
        guard let depthState = device.makeDepthStencilState(descriptor: depth) else {
            throw Failure.resource("depth state")
        }

        if physicallyBased?.usesSecondaryTextureCoordinates == true
            || classic?.material.usesSecondaryTextureCoordinates == true,
           geometry.vertices.contains(where: { $0.secondaryTextureCoordinate == nil }) {
            throw Failure.missingSecondaryTextureCoordinates
        }
        let dynamicGeometry = try dynamicCapacity.map {
            try DynamicGeometryBuffers(device: device, capacity: $0, geometry: geometry, colored: colored)
        }
        self.dynamicGeometry = dynamicGeometry
        geometryRevision = dynamicGeometry == nil ? nil : 0
        self.skin = skin
        self.skinInfluences = try source?.skinInfluences ?? skin.map { skin in
            let influences = skin.influences.map { influence in
                let joints = influence.joints
                return PackedSkinInfluence(
                    joints: SIMD4(UInt32(joints.x), UInt32(joints.y), UInt32(joints.z), UInt32(joints.w)),
                    weights: influence.weights
                )
            }
            return try Self.makeBuffer(influences, on: device, label: "mesh skin influences")
        }
        let vertices = try source?.vertices ?? dynamicGeometry?.initial.vertices
            ?? Self.makeBuffer(geometry.vertices.map(PackedMeshVertex.init), on: device, label: "mesh vertices")
        self.vertices = vertices
        self.readsVertexColors = colored
        if let source, source.dynamicGeometry != nil {
            vertexColors = nil // The source leases the current revision's colors.
        } else if let dynamicGeometry {
            vertexColors = dynamicGeometry.initial.colors
        } else {
            vertexColors = colored
                ? try source?.vertexColors
                    ?? Self.makeBuffer(geometry.vertices.map { $0.color ?? SIMD4(repeating: 1) }, on: device, label: "mesh vertex colors")
                : nil
        }
        self.morphTargets = morphTargets
        posedPickingBVH = skin != nil || morphTargets != nil ? PosedMeshBVHCache() : nil
        morph = try source?.morph ?? morphTargets.map { try PreparedMorph($0, device: device) }
        self.supportsInstancing = supportsInstancing
        sourceBounds = geometry.bounds
        if case .physicallyBased(let material) = material, let graph = material.surfaceGraph, graph.hasVertexOutputs {
            deformation = try MeshDeformation(
                device: device, graph: graph, material: material, source: vertices, geometry: geometry,
                fallbackTexture: { try PhysicallyBasedMaterial.Prepared.makeWhiteTexture(device: device) }
            )
        } else {
            deformation = nil
        }
        self.indices = try source?.indices ?? dynamicGeometry?.initial.indices
            ?? Self.makeBuffer(geometry.indices, on: device, label: "mesh indices")
        self.depthState = depthState
        self.depthMode = depthMode
        self.device = device
        self.colorPixelFormat = colorPixelFormat
        self.colorPixelFormats = colorPixelFormats
        self.surfaceRenderTargets = surfaceRenderTargets
        self.sourceGeometry = geometry
        self.material = preparedMaterial
        self.colorTexture = colorTexture
        self.textureHelperOptions = textureHelper?.options
        self.indirectColorArguments = indirectColorArguments
        self.physicallyBased = physicallyBased
        self.viewportInputs = viewportInputs
        self.readsSoftParticles = readsSoftParticles
        self.pbrBindings = pbrBindings
        self.graphShadow = graphShadow
        self.graphTransmittedShadow = graphTransmittedShadow
        self.punctualShadowFallback = punctualShadowFallback
        self.screenInputsFallback = screenInputsFallback
        self.receivesShadows = receivesShadows
        self.renderState = state
        self.usesStencilAttachment = usesStencilAttachment
        self.sampleCount = sampleCount
        self.classic = classic
        guard let attributeVertex = descriptor.vertexFunction else { throw Failure.function("vertex") }
        self.attributeVertex = attributeVertex
        self.attributeFragment = attributeFragment
        self.sceneEnvironmentSource = sceneEnvironmentSource
    }

    func validateMaterialCoordinates(_ geometry: MeshGeometry) throws {
        if physicallyBased?.usesSecondaryTextureCoordinates == true
            || classic?.material.usesSecondaryTextureCoordinates == true,
           geometry.vertices.contains(where: { $0.secondaryTextureCoordinate == nil }) {
            throw Failure.missingSecondaryTextureCoordinates
        }
    }

    var supportsPlanarMirrorMaterial: Bool {
        !usesStencilAttachment && curveFlow == nil && deformation == nil && rawProgram == nil && geometryRevision == nil
            && skin == nil && morphTargets == nil
    }

    func makeMaterialOverride(
        _ material: MeshMaterial, state: MeshRenderState?, layout: SceneMaterialOverride.Layout
    ) throws -> MeshRenderer {
        guard !usesStencilAttachment, state?.stencil == nil else { throw Failure.stencilPass }
        let state = state ?? MeshRenderState.defaultState(for: material)
        if layout.targets != nil {
            guard case .physicallyBased(let pbr) = material, state.blending == .none,
                  pbr.alphaMode != .blend, !pbr.hasTransmission else { throw SurfaceRenderTargets.Failure.material }
        } else if layout.formats.count > 1 {
            guard case .custom = material, state.blending == .none, layout.samples == 1 else {
                throw Failure.multipleRenderTargets
            }
        }
        return try MeshRenderer(
            device: device, colorPixelFormat: layout.formats[0], geometry: sourceGeometry,
            material: material, depthMode: layout.depth, skin: skin, morphTargets: morphTargets,
            supportsInstancing: supportsInstancing, renderState: state, sampleCount: layout.samples,
            indirectCommands: false,
            additionalColorPixelFormats: Array(layout.formats.dropFirst()), surfaceRenderTargets: layout.targets,
            sharingGeometryWith: self
        )
    }

    /// Prepares a renderer once for a live curve flow. Its indices, colors and
    /// material pipeline remain usable across offset and curve edits. An optional
    /// ordinary `InstanceSet` applies rigid transforms to the entire bent batch.
    public convenience init(
        curveFlow: CurveFlow, colorPixelFormat: MTLPixelFormat, material: MeshMaterial = .unlit,
        depthMode: MeshDepthMode = .forward, supportsInstancing: Bool = false,
        renderState: MeshRenderState? = nil, sampleCount: Int = 1, usesStencilAttachment: Bool = false
    ) throws {
        if case .physicallyBased(let material) = material, material.surfaceGraph?.hasVertexOutputs == true {
            throw CurveFlow.Failure.vertexGraph
        }
        try self.init(
            device: curveFlow.device, colorPixelFormat: colorPixelFormat, geometry: curveFlow.geometry,
            material: material, depthMode: depthMode, supportsInstancing: supportsInstancing,
            renderState: renderState, sampleCount: sampleCount, usesStencilAttachment: usesStencilAttachment
        )
        self.curveFlow = curveFlow
        curveFlow.adoptInitialBuffer(vertices)
    }

    /// Prepares the pipeline that `SceneGraph.encodeSurfaceAttributes` draws this mesh
    /// with. Encoding prepares it on first use; call this beforehand to avoid compiling
    /// a material variant during a frame.
    public func prepareSurfaceAttributes() throws {
        _ = try surfaceAttributePipeline()
    }

    private struct SceneEnvironmentPipelineSource {
        let descriptor: MTLRenderPipelineDescriptor
        let library: any MTLLibrary
        let fragment: String
        let constants: MTLFunctionConstantValues

        func environmentFunction(named name: String) throws -> any MTLFunction {
            guard let constants = constants.copy() as? MTLFunctionConstantValues else {
                throw Failure.resource("PBR environment specialization")
            }
            var hasEnvironment = true
            constants.setConstantValue(&hasEnvironment, type: .bool, index: 0)
            do {
                return try library.makeFunction(name: name, constantValues: constants)
            } catch {
                throw Failure.pipeline(error.localizedDescription)
            }
        }
    }

    private func environmentPipeline(surfaceAttributes: Bool) throws -> (any MTLRenderPipelineState)? {
        guard let source = sceneEnvironmentSource else { return nil }
        if surfaceAttributes { return try surfaceAttributePipeline(sceneEnvironment: source) }
        if let sceneEnvironmentColorPipeline { return sceneEnvironmentColorPipeline }
        guard let descriptor = source.descriptor.copy() as? MTLRenderPipelineDescriptor else {
            throw Failure.resource("PBR environment pipeline descriptor")
        }
        descriptor.label = "Frostfield mesh with scene environment"
        descriptor.fragmentFunction = try source.environmentFunction(named: source.fragment)
        let pipeline = try Self.makePipeline(device: device, descriptor: descriptor)
        sceneEnvironmentColorPipeline = pipeline
        return pipeline
    }

    private func surfaceAttributePipeline(
        sceneEnvironment: SceneEnvironmentPipelineSource? = nil
    ) throws -> any MTLRenderPipelineState {
        guard !usesStencilAttachment else { throw Failure.stencilPass }
        guard rawProgram == nil else { throw RawMeshProgram.Failure.unsupportedAuxiliaryPass }
        if let cached = sceneEnvironment == nil ? attributePipeline : sceneEnvironmentAttributePipeline { return cached }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Frostfield surface attributes"
        if pipeline.supportIndirectCommandBuffers {
            descriptor.supportIndirectCommandBuffers = true
            descriptor.inputPrimitiveTopology = .triangle
        }
        descriptor.vertexFunction = attributeVertex
        descriptor.colorAttachments[0].pixelFormat = SceneGraph.surfaceAttributesPixelFormat
        descriptor.depthAttachmentPixelFormat = Self.depthPixelFormat
        let source = attributeFragment
        if let sceneEnvironment {
            descriptor.fragmentFunction = try sceneEnvironment.environmentFunction(named: source.name)
        } else if let constants = source.constants {
            do {
                descriptor.fragmentFunction = try source.library.makeFunction(name: source.name, constantValues: constants)
            } catch {
                throw Failure.pipeline(error.localizedDescription)
            }
        } else {
            descriptor.fragmentFunction = try Self.function(source.name, in: source.library)
        }
        let pipeline = try Self.makePipeline(device: device, descriptor: descriptor)
        if sceneEnvironment == nil {
            attributePipeline = pipeline
        } else {
            sceneEnvironmentAttributePipeline = pipeline
        }
        return pipeline
    }

    /// Function constants 5 to 7 of the PBR and classic fragments.
    private static func setCoverageConstants(
        _ constants: MTLFunctionConstantValues, colored: Bool, state: MeshRenderState
    ) {
        var colored = colored
        var hash = state.alphaHash
        var coverage = state.alphaToCoverage
        constants.setConstantValue(&colored, type: .bool, index: 5)
        constants.setConstantValue(&hash, type: .bool, index: 6)
        constants.setConstantValue(&coverage, type: .bool, index: 7)
    }

    /// Encodes one draw into the caller's command buffer without submitting it.
    /// Supply color and depth attachments of equal size whose sample count equals
    /// the renderer's. Clear depth to 1 for a forward-depth renderer or 0 for
    /// a reversed-depth renderer.
    /// Supply every prepared color format in attachment index order.
    /// Keep the renderer and attachments
    /// alive through completion if using a command buffer with unretained references.
    public func encode(
        _ scene: MeshScene,
        pass: MTLRenderPassDescriptor,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try encode(scene, pass: pass, commandBuffer: commandBuffer, indirect: nil, instances: nil)
    }

    /// Encodes indexed draws whose counts and start indices come from GPU
    /// argument records. The regular material, depth and mesh deformation
    /// bindings and depth clear requirements apply. Use a renderer prepared for instancing when passing
    /// an `InstanceSet`. Without one, only one instance at base zero can draw.
    ///
    /// End the argument producer's encoder before calling this method. A
    /// validation pass reads the tracked source buffer and writes private
    /// draw arguments on this command buffer. Resources are held through
    /// completion. With unretained references, keep the attachments alive.
    public func encode(
        _ scene: MeshScene, pass: MTLRenderPassDescriptor, commandBuffer: any MTLCommandBuffer,
        indirectDraws: MeshIndirectDraws, instances: InstanceSet? = nil
    ) throws {
        try encode(scene, pass: pass, commandBuffer: commandBuffer, indirect: indirectDraws, instances: instances)
    }

    private func encode(
        _ scene: MeshScene, pass: MTLRenderPassDescriptor, commandBuffer: any MTLCommandBuffer,
        indirect: MeshIndirectDraws?, instances: InstanceSet?
    ) throws {
        guard commandBuffer.device === device else { throw Failure.wrongDevice }
        try indirect?.validate(commandBuffer: commandBuffer)
        guard !requiresTransmissionBackdrop else { throw Failure.transmissionBackdrop }
        guard !receivesShadows else { throw Failure.shadowMap }
        let (color, formats, size) = try Self.colorAttachments(in: pass, device: device)
        guard usesStencilAttachment == Self.usesStencil(pass),
              !usesStencilAttachment || (indirect == nil && scene.viewportSnapshot == nil) else {
            throw Failure.stencilPass
        }
        if scene.viewportSnapshot?.source == .opaquePass { throw SceneViewportSnapshot.Failure.capturePass }
        try scene.viewportSnapshot?.validate(pass: pass, commandBuffer: commandBuffer)
        guard formats == colorPixelFormats else { throw Failure.attachments }
        guard color.sampleCount == sampleCount else { throw Failure.sampleCount }
        try validateTextureHelper(color: color, resolve: pass.colorAttachments[0].resolveTexture)
        try rawProgram?.validateAttachments(pass, commandBuffer: commandBuffer)
        let clipFromWorld = try Self.viewProjection(
            camera: scene.camera, cameraTransform: scene.cameraTransform, size: size, depthMode: depthMode
        )
        var draw = try prepare(
            world: scene.transform.matrix, clipFromWorld: clipFromWorld,
            shadingCamera: SIMD4(scene.cameraTransform.translation, 1), parameters: scene.materialParameters,
            jointPalette: scene.jointPalette, morphWeights: scene.morphWeights, instances: instances
        )
        let lightTables = try punctualShadowFallback.map { _ in
            try lightTableCache.make(device: device, profiles: [], cookies: [])
        }
        let graphView = try GraphViewContext(
            camera: scene.camera, cameraTransform: scene.cameraTransform, width: size.x, height: size.y,
            depthMode: depthMode
        )
        let logarithmicDepth = try depthMode == .logarithmic ? LogarithmicRasterDepth(view: graphView, pass: pass) : nil
        draw.indirectSource = indirect
        if indirect == nil, draw.indexRange.isEmpty || draw.instanceCount == 0 {
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.encoder }
            encoder.endEncoding()
            return
        }
        try viewportInputs.validate(in: scene.viewportSnapshot)
        if readsSoftParticles { _ = try graphView.softParticleProjection() }
        let viewportTextures = try viewportInputs.isEmpty ? nil : scene.viewportSnapshot?.prepare(device: device)
        draw.deformedVertices = try deformation?.acquireVertices(for: commandBuffer)
        try acquireDrawInputs(&draw, for: commandBuffer)
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            throw Failure.encoder
        }
        logarithmicDepth?.bind(into: encoder)
        draw.palette?.retainBufferIfNeeded(on: commandBuffer)
        if usesNativeLambert || punctualShadowFallback != nil {
            Self.bindLightProbe(LightProbe.noRasterLighting, into: encoder)
        }
        if let punctualShadowFallback, let lightTables {
            lightTables.retainIfNeeded(on: commandBuffer)
            Self.bindPunctualLights(
                [], shadowTexture: punctualShadowFallback, shadowViews: [], tables: lightTables, into: encoder
            )
        }
        encode(
            draw, into: encoder, shadowMap: nil, graphView: graphView,
            viewBasis: Self.viewBasis(cameraTransform: scene.cameraTransform), viewportTextures: viewportTextures
        )
        encoder.endEncoding()
    }

    /// Encodes the graph's color or attribute pass. When the graph has an inspector, it is told
    /// about the encode's start, each draw and the finish or the failure. A validation-only
    /// call encodes nothing and is not reported.
    static func encode(
        _ graph: SceneGraph,
        pass: MTLRenderPassDescriptor,
        commandBuffer: any MTLCommandBuffer,
        shadowMap: DirectionalShadowMap?,
        punctualShadowMaps: PunctualShadowMaps? = nil,
        view: (clipFromWorld: simd_float4x4, cameraPosition: SIMD3<Float>, directionToCamera: SIMD3<Float>?)? = nil,
        viewport: MTLViewport? = nil,
        depthMode: MeshDepthMode = .forward,
        transmissionBackdrop: (any MTLTexture)? = nil,
        graphCamera: ShaderGraphCamera? = nil,
        screenOcclusion: (any MTLTexture)? = nil,
        screenDirectionalShadow: ScreenSpaceShadows.DirectionalVisibility? = nil,
        surfaceAttributes: Bool = false,
        excluding: Set<UUID> = [],
        scissor: MTLScissorRect? = nil,
        indirectDraws: [UUID: MeshIndirectDraws] = [:],
        occlusionQueries: SceneOcclusionRecording? = nil,
        gpuTiming: SceneGPUTimingRecording? = nil,
        surfaceRenderTargets: SurfaceRenderTargets? = nil,
        ordering: SceneRenderOrdering = .init(),
        toonOutline: ToonOutlinePass? = nil,
        reflectionTargets: [any MTLTexture] = [],
        motionVectors: RasterMotionVectors.Recording? = nil,
        viewLayers: SceneLayers? = nil,
        viewIndex: (index: Int, count: Int)? = nil,
        validateOnly: Bool = false
    ) throws -> SceneDrawStatistics {
        let inspector = validateOnly ? nil : graph.encodeInspector
        let info = SceneEncodeInfo(
            id: inspector == nil ? 0 : SceneEncodeInfo.nextID(), pass: surfaceAttributes ? .surfaceAttributes : .color,
            viewIndex: viewIndex?.index, viewCount: viewIndex?.count, depthMode: depthMode
        )
        inspector?.sceneEncodeDidBegin(info, graph: graph)
        do {
            let statistics = try encodeScene(
                graph,
                pass: pass,
                commandBuffer: commandBuffer,
                shadowMap: shadowMap,
                punctualShadowMaps: punctualShadowMaps,
                view: view,
                viewport: viewport,
                depthMode: depthMode,
                transmissionBackdrop: transmissionBackdrop,
                graphCamera: graphCamera,
                screenOcclusion: screenOcclusion,
                screenDirectionalShadow: screenDirectionalShadow,
                surfaceAttributes: surfaceAttributes,
                excluding: excluding,
                scissor: scissor,
                indirectDraws: indirectDraws,
                occlusionQueries: occlusionQueries,
                gpuTiming: gpuTiming,
                surfaceRenderTargets: surfaceRenderTargets,
                ordering: ordering,
                toonOutline: toonOutline,
                reflectionTargets: reflectionTargets,
                motionVectors: motionVectors,
                viewLayers: viewLayers,
                validateOnly: validateOnly,
                info: info, inspector: inspector
            )
            inspector?.sceneEncodeDidFinish(info, statistics: statistics)
            return statistics
        } catch {
            inspector?.sceneEncodeDidFail(info, error: error)
            throw error
        }
    }

    private static func encodeScene(
        _ graph: SceneGraph,
        pass: MTLRenderPassDescriptor,
        commandBuffer: any MTLCommandBuffer,
        shadowMap: DirectionalShadowMap?,
        punctualShadowMaps: PunctualShadowMaps? = nil,
        view: (clipFromWorld: simd_float4x4, cameraPosition: SIMD3<Float>, directionToCamera: SIMD3<Float>?)? = nil,
        viewport: MTLViewport? = nil,
        depthMode: MeshDepthMode = .forward,
        transmissionBackdrop: (any MTLTexture)? = nil,
        graphCamera: ShaderGraphCamera? = nil,
        screenOcclusion: (any MTLTexture)? = nil,
        screenDirectionalShadow: ScreenSpaceShadows.DirectionalVisibility? = nil,
        surfaceAttributes: Bool = false,
        excluding: Set<UUID> = [],
        scissor: MTLScissorRect? = nil,
        indirectDraws: [UUID: MeshIndirectDraws] = [:],
        occlusionQueries: SceneOcclusionRecording? = nil,
        gpuTiming: SceneGPUTimingRecording? = nil,
        surfaceRenderTargets: SurfaceRenderTargets? = nil,
        ordering: SceneRenderOrdering = .init(),
        toonOutline: ToonOutlinePass? = nil,
        reflectionTargets: [any MTLTexture] = [],
        motionVectors: RasterMotionVectors.Recording? = nil,
        viewLayers: SceneLayers? = nil,
        validateOnly: Bool = false,
        info: SceneEncodeInfo,
        inspector: (any SceneEncodeInspector)?
    ) throws -> SceneDrawStatistics {
        let device = commandBuffer.device
        let (color, formats, size) = try colorAttachments(in: pass, device: device)
        let usesStencil = Self.usesStencil(pass)
        if usesStencil {
            guard !surfaceAttributes, depthMode != .logarithmic, shadowMap == nil, punctualShadowMaps == nil,
                  motionVectors == nil, transmissionBackdrop == nil, toonOutline?.isEnabled != true,
                  indirectDraws.isEmpty, graph.viewportSnapshot == nil,
                  graph.overrideMaterial == nil, graph.primitiveMaterialOverride == nil else { throw Failure.stencilPass }
        } else if graph.overrideMaterial?.renderState?.stencil != nil {
            throw Failure.stencilPass
        }
        try motionVectors?.validate(pass: pass, size: size, depthMode: depthMode, commandBuffer: commandBuffer)
        try graph.viewportSnapshot?.validate(pass: pass, commandBuffer: commandBuffer)
        let opaqueSnapshot = !surfaceAttributes && graph.viewportSnapshot?.source == .opaquePass ? graph.viewportSnapshot : nil
        let usesMultipleRenderTargets = formats.count > 1
        if usesMultipleRenderTargets, surfaceAttributes || screenOcclusion != nil || transmissionBackdrop != nil {
            throw Failure.multipleRenderTargets
        }
        let scissorIsEmpty = scissor.map { $0.width == 0 || $0.height == 0 } ?? false
        try validateScissor(scissor, size: size)
        if let screenOcclusion {
            guard screenOcclusion.device === device, screenOcclusion.pixelFormat == .rgba16Float,
                  !TextureTransfer.sharesStorage(screenOcclusion, color),
                  pass.colorAttachments[0].resolveTexture.map({ !TextureTransfer.sharesStorage(screenOcclusion, $0) }) ?? true,
                  screenOcclusion.textureType == .type2D, screenOcclusion.usage.contains(.shaderRead),
                  screenOcclusion.storageMode != .memoryless,
                  screenOcclusion.width == size.x, screenOcclusion.height == size.y else {
                throw Failure.screenOcclusion
            }
        }

        if graph.highPrecision, view != nil, graphCamera == nil { throw Failure.highPrecisionView }
        var clipFromWorld = try view?.clipFromWorld ?? viewProjection(
            graph: graph, size: size, depthMode: depthMode
        )
        let environmentRotation = graph.effectiveEnvironmentEulerRotation
        guard graph.environmentIntensity.isFinite, graph.environmentIntensity >= 0, environmentRotation.isFiniteVector else {
            throw Failure.environment
        }
        if let lighting = graph.environment, lighting.device !== device { throw Failure.wrongDevice }
        let directionalVisibility = try screenDirectionalShadow?.prepare(
            pass: pass, size: size, viewport: viewport, commandBuffer: commandBuffer
        )
        let pbrScreenInputs = try PBRScreenInputs(device: device, ambient: screenOcclusion, directional: directionalVisibility)
        if let pbrScreenInputs, !commandBuffer.retainedReferences {
            let retained = RetainedScreenInputs(value: pbrScreenInputs)
            commandBuffer.addCompletedHandler { _ in withExtendedLifetime(retained) {} }
        }
        let sceneEnvironment = RasterSceneEnvironment(
            lighting: graph.environment, intensity: graph.environmentIntensity,
            yaw: graph.environmentRotation, eulerRotation: environmentRotation
        )
        let cameraPosition = view?.cameraPosition ?? graph.cameraTransform.translation
        let viewBasis = view.map {
            Self.viewBasis(clipFromWorld: clipFromWorld, directionToCamera: $0.directionToCamera, depthMode: depthMode)
        } ?? Self.viewBasis(cameraTransform: graph.cameraTransform)
        let shadingCamera = try view.map {
            try explicitShadingCamera(
                clipFromWorld: clipFromWorld, position: $0.cameraPosition, direction: $0.directionToCamera,
                highPrecision: graph.highPrecision
            )
        } ?? graph.orthographicDirectionToCamera.map { SIMD4($0, 0) } ?? SIMD4(cameraPosition, 1)
        try validateViewport(viewport, size: size)
        if graph.projectionJitter != .zero {
            guard TemporalAntiAliasing.isValidJitter(graph.projectionJitter) else { throw Failure.projectionJitter }
            clipFromWorld = TemporalAntiAliasing.jitteredClipFromWorld(
                clipFromWorld, jitter: graph.projectionJitter,
                width: viewport?.width ?? Double(size.x), height: viewport?.height ?? Double(size.y)
            )
        }

        // Surface graphs read the camera and viewport. A caller-supplied clip matrix
        // separates the view and projection matrices and gives clip distances only
        // through a graph camera, which must agree with it.
        let graphView: GraphViewContext
        if let view {
            if let graphCamera, !graphCamera.matches(clipFromWorld: view.clipFromWorld, perColumn: graph.highPrecision) {
                throw Failure.graphCameraInputs
            }
            let target = SIMD2<Float>(size)
            let box = viewport.map {
                MTLViewportBox(origin: SIMD2(Float($0.originX), Float($0.originY)), size: SIMD2(Float($0.width), Float($0.height)))
            } ?? MTLViewportBox(origin: .zero, size: target)
            graphView = GraphViewContext(
                cameraPosition: view.cameraPosition, viewport: box, targetSize: target, camera: graphCamera,
                windowDepth: viewport.map { SIMD2(Float($0.znear), Float($0.zfar)) } ?? SIMD2(0, 1), depthMode: depthMode
            )
        } else {
            graphView = try GraphViewContext(graph: graph, width: size.x, height: size.y, depthMode: depthMode)
        }
        let logarithmicDepth = try depthMode == .logarithmic
            ? LogarithmicRasterDepth(view: graphView, pass: pass, viewport: viewport) : nil
        let precision: HighPrecisionModelView?
        if graph.highPrecision {
            let origin = SIMD3<Double>(cameraPosition)
            let relativeView: simd_double4x4
            if view != nil {
                guard let authored = graphCamera?.relativeView(at: origin) else { throw Failure.highPrecisionView }
                relativeView = authored
            } else {
                relativeView = simd_inverse(graph.cameraTransform.doubleMatrix(relativeTo: origin))
            }
            let projection = graph.projectionJitter == .zero ? graphView.projectionMatrix
                : TemporalAntiAliasing.jitteredClipFromWorld(
                    graphView.projectionMatrix, jitter: graph.projectionJitter,
                    width: viewport?.width ?? Double(size.x), height: viewport?.height ?? Double(size.y)
                )
            precision = HighPrecisionModelView(
                graph: graph, origin: origin, viewFromRelativeWorld: relativeView,
                projection: projection, excluding: excluding
            )
        } else {
            precision = nil
        }

        // Validate every draw before creating the encoder so a failed node does
        // not leave a partially encoded scene in the caller's command buffer.
        var draws: [GraphDraw] = []
        var lights: [PackedPunctualLight] = []
        let cameraOrigin = SIMD3<Double>(cameraPosition)
        var shadowSlots = PunctualShadowSlots(maps: punctualShadowMaps, cameraOrigin: precision == nil ? nil : cameraOrigin)
        var statistics = SceneDrawStatistics()
        var batchBytes = 0
        if let shadowMap, shadowMap.automaticallyFitsCamera {
            try shadowMap.validate(commandBuffer: commandBuffer)
            if view != nil {
                guard let graphCamera else { throw DirectionalShadowMap.Failure.invalidCamera }
                try shadowMap.fitCascades(view: graphCamera)
            } else {
                if graph.orthographicCamera != nil {
                    try shadowMap.fitCascades(view: graph.shaderGraphCamera(
                        aspectRatio: Float(size.x) / Float(size.y), depthMode: depthMode
                    ))
                } else {
                    try shadowMap.fitCascades(
                        camera: graph.camera, cameraTransform: graph.cameraTransform,
                        aspectRatio: Float(size.x) / Float(size.y)
                    )
                }
            }
        }
        try shadowMap?.validateFitted()
        let shadowViews = shadowMap?.cascadeViews ?? []
        // Level-of-detail nodes select for the camera that this pass views from. Hidden
        // subtrees and nodes outside the camera layers neither draw, light nor cast shadows.
        let resolved = try graph.resolve(
            cameraPosition: cameraPosition, excluding: excluding, relativeWorldTransforms: precision?.relativeWorldTransforms
        )
        let nodes = try resolved.filter { node in
            guard graph.renders(node) else { return false }
            if usesStencil, node.primitive != nil || node.batch != nil {
                throw Failure.stencilPass
            }
            if let mesh = node.mesh ?? node.batch?.renderer, mesh.usesStencilAttachment != usesStencil {
                throw Failure.stencilPass
            }
            return true
        }
        try validateIndirectTargets(indirectDraws, nodes: resolved, commandBuffer: commandBuffer)
        if depthMode == .logarithmic, !surfaceAttributes,
           toonOutline != nil || nodes.contains(where: { $0.primitive != nil }) {
            throw Failure.logarithmicDepthPass
        }
        let materialOverride = graph.overrideMaterial
        materialOverride?.removeRetiredSources()
        let primitiveOverride = graph.primitiveMaterialOverride
        primitiveOverride?.removeRetiredSources()
        let overrideLayout = SceneMaterialOverride.Layout(
            formats: formats, depth: depthMode, samples: color.sampleCount, targets: surfaceRenderTargets
        )
        if usesMultipleRenderTargets, nodes.contains(where: { $0.primitive != nil }) {
            throw Failure.multipleRenderTargets
        }
        func cameraFrustum(for clip: simd_float4x4) -> ViewFrustum? {
            try? ViewFrustum(clipFromWorld: logarithmicDepth?.cullingClip(clip) ?? clip)
        }
        let worldFrustum = cameraFrustum(for: clipFromWorld)
        let relativeFrustum = precision.flatMap { cameraFrustum(for: $0.clipMatrix(for: matrix_identity_double4x4)) }
        var primitiveDraws: [PrimitiveDraw] = []
        var primitiveIndirectRecords = 0
        if !graph.primitiveIndirectDraws.isEmpty {
            for (id, _) in graph.primitiveIndirectDraws {
                guard let node = nodes.first(where: { $0.id == id }) else {
                    throw PrimitiveIndirectDraws.Failure.unknownNode(id)
                }
                guard node.primitive != nil else { throw PrimitiveIndirectDraws.Failure.incompatibleNode(id) }
            }
        }
        let primitiveView = !surfaceAttributes && nodes.contains { $0.primitive != nil }
            ? try PrimitiveView(
                clipFromWorld: clipFromWorld, viewport: viewport, attachment: color, depthMode: depthMode,
                precision: precision, orthographicBack: view?.directionToCamera ?? graph.orthographicDirectionToCamera
            )
            : nil
        // Assign shadow slots before preparing casters, which need every shadow view.
        var resolvedLights: [ResolvedPunctualLight] = []
        var ambientIrradiance = SIMD3<Double>.zero
        if !surfaceAttributes {
            shadowSlots.preserveAssignments(for: nodes, commandBuffer: commandBuffer)
            for node in nodes {
                if let ambient = node.ambientLight {
                    ambientIrradiance += SIMD3<Double>(ambient.color) * Double(ambient.intensity)
                }
                if let light = node.light, light.light.intensity > 0, light.light.color != .zero {
                    lights.append(try shadowSlots.pack(light, id: node.id, preciseLight: precision?.lightPositions[node.id]))
                    resolvedLights.append(light)
                }
            }
        }
        let hasPreciseLight = shadowSlots.relativeLightPositions.contains { $0.w != 0 }
        let probeLighting = surfaceAttributes ? LightProbe.noRasterLighting
            : try LightProbe.rasterLighting(graph.lightProbes, ambient: ambientIrradiance)
        let probeGrids = surfaceAttributes ? nil : graph.lightProbeGridLighting
        if let probeGrids, probeGrids.device !== device { throw Failure.wrongDevice }
        var inventory = SceneResourceInventory(forbiddenTextures: reflectionTargets)
        if !reflectionTargets.isEmpty, let snapshot = graph.viewportSnapshot {
            inventory.recordTexture(snapshot.color)
            inventory.recordTexture(snapshot.depth)
            guard !inventory.hasTextureFeedback else { throw PlanarReflection.Failure.feedback }
        }
        for (nodeIndex, node) in nodes.enumerated() {
            // An array-camera view draws only nodes on its own layers, on top of the graph's
            // camera layers. Lights, shadow casters and validation still follow the graph's.
            let drawsInView = viewLayers.map { node.layers.intersects($0) } ?? true
            if let source = node.primitive, drawsInView, let primitiveView, let elements = source.drawElements {
                guard source.device === device else { throw Failure.wrongDevice }
                let replacement = try primitiveOverride?.renderer(
                    for: source, node: node.id, colorPixelFormat: color.pixelFormat, sampleCount: color.sampleCount
                )
                if let map = replacement?.colorMap,
                   TextureTransfer.sharesStorage(map, color)
                    || pass.colorAttachments[0].resolveTexture.map({ TextureTransfer.sharesStorage(map, $0) }) == true {
                    throw PrimitiveRendererError.mapAliasing
                }
                let primitive = replacement ?? source
                if !reflectionTargets.isEmpty {
                    primitive.recordResources(in: &inventory)
                    guard !inventory.hasTextureFeedback else { throw PlanarReflection.Failure.feedback }
                }
                if var draw = try primitiveView.draw(
                    primitive, elements: elements, node: node, nodeIndex: nodeIndex, frustum: worldFrustum,
                    device: device, color: color, recordsOpaqueDepth: ordering.recordsOpaqueDepth,
                    overrideTint: replacement == nil ? nil : primitiveOverride?.parameters
                ) {
                    if let source = graph.primitiveIndirectDraws[node.id] {
                        try source.validate(commandBuffer: commandBuffer)
                        let limits = try draw.indirectLimits()
                        let validated = try source.encodeValidation(
                            into: commandBuffer, vertexCount: limits.vertexCount, instanceCount: limits.instanceCount
                        )
                        primitiveIndirectRecords += validated.count
                        guard primitiveIndirectRecords <= PrimitiveIndirectDraws.maximumSceneCount else {
                            throw PrimitiveIndirectDraws.Failure.sceneBudget
                        }
                        draw.indirect = validated
                        if !commandBuffer.retainedReferences {
                            let retained = RetainedPrimitiveIndirect(source: source.buffer, validated: validated.buffer)
                            commandBuffer.addCompletedHandler { _ in withExtendedLifetime(retained) {} }
                        }
                    }
                    primitiveDraws.append(draw)
                } else {
                    statistics.culledPrimitiveCount += 1
                }
            }
            guard let renderer = node.mesh ?? node.batch?.renderer else { continue }
            guard renderer.device === device else { throw Failure.wrongDevice }
            let replacement: (renderer: MeshRenderer, parameters: SIMD4<Float>)?
            if let materialOverride, !materialOverride.excludedNodes.contains(node.id) {
                replacement = (try materialOverride.renderer(for: renderer, layout: overrideLayout), materialOverride.parameters)
            } else {
                replacement = nil
            }
            let colorRenderer = replacement?.renderer ?? renderer
            if !reflectionTargets.isEmpty {
                renderer.recordResources(in: &inventory)
                colorRenderer.recordResources(in: &inventory)
                guard !inventory.hasTextureFeedback else { throw PlanarReflection.Failure.feedback }
            }
            if replacement != nil, colorRenderer.requiresTransmissionBackdrop {
                if node.batch != nil { throw BatchedMesh.Failure.transmission }
                if node.instances != nil { throw Failure.instancedTransmission }
            }
            if surfaceAttributes {
                guard !colorRenderer.isBlended else { continue }
                _ = try colorRenderer.surfaceAttributePipeline()
            } else {
                if let surfaceRenderTargets, colorRenderer.surfaceRenderTargets != surfaceRenderTargets {
                    throw SurfaceRenderTargets.Failure.rendererLayout
                }
                guard colorRenderer.colorPixelFormats == formats else { throw Failure.attachments }
                guard colorRenderer.sampleCount == color.sampleCount else { throw Failure.sampleCount }
                guard !colorRenderer.receivesShadows || shadowMap != nil else { throw Failure.shadowMap }
                try colorRenderer.validateTextureHelper(color: color, resolve: pass.colorAttachments[0].resolveTexture)
                try colorRenderer.rawProgram?.validateAttachments(pass, commandBuffer: commandBuffer)
            }
            guard colorRenderer.depthMode == depthMode else { throw Failure.depthMode }
            if !graphView.hasProjection, colorRenderer.physicallyBased?.surfaceGraph?.readsCameraProjection == true {
                throw Failure.graphCameraInputs
            }
            let precise = precision?.matrices(for: node.id)
            let preciseWorld = precision?.relativeWorldTransforms[node.id]
            var draw = try renderer.prepare(
                world: node.worldTransform, clipFromWorld: clipFromWorld,
                shadingCamera: shadingCamera, parameters: node.materialParameters,
                jointPalette: node.jointPalette, morphWeights: node.morphWeights, instances: node.instances,
                precise: precise
            )
            draw.suppressesShadows = !node.receivesShadow
            draw.indirectSource = indirectDraws[node.id]
            let castsShadow = node.castsShadow && renderer.rawProgram == nil
                && !renderer.isBlended && !renderer.requiresTransmissionBackdrop
            let transmittedCaster: DirectionalTransmissionPass.Caster?
            if node.castsShadow, let shadowMap, shadowMap.transmissionLayers > 0,
               renderer.requiresTransmissionBackdrop || renderer.physicallyBased?.surfaceGraph?.hasCastShadowOutput == true {
                transmittedCaster = try renderer.transmittedShadowCaster(
                    draw, light: shadowMap.lightParameters, node: node.id, batched: node.batch != nil
                )
            } else {
                transmittedCaster = nil
            }
            let castsDirectionalShadow = castsShadow || transmittedCaster != nil
            let relativeWorld = node.batch == nil && renderer.skin == nil ? preciseWorld : nil
            if hasPreciseLight, colorRenderer.usesPunctualLights, let relativeWorld {
                let relative = HighPrecisionModelView.packed(relativeWorld)
                guard isFinite(relative) else { throw Failure.transform }
                draw.relativeLightModel = relative
            }
            if (castsDirectionalShadow || colorRenderer.receivesShadows), let shadowMap, let relativeWorld {
                let relative = shadowMap.relativeDraw(relativeWorld, cameraOrigin: cameraOrigin)
                guard isFinite(relative.splitPlane), relative.clips.allSatisfy(isFinite) else { throw Failure.transform }
                draw.relativeDirectional = relative
            }
            // Use the same model-to-light transforms for caster projection and receiver lookup.
            let shadowClips: [simd_float4x4] = if !castsDirectionalShadow {
                []
            } else if let relative = draw.relativeDirectional {
                relative.clips
            } else {
                shadowViews.map { $0 * node.worldTransform }
            }
            guard shadowClips.allSatisfy(isFinite) else { throw Failure.transform }
            let punctualClips = castsShadow
                ? shadowSlots.casterClips(world: node.worldTransform, relativeWorld: relativeWorld) : []
            guard punctualClips.allSatisfy(isFinite) else { throw Failure.transform }
            guard draw.instanceCount > 0,
                  node.batch != nil || draw.indirectSource != nil || !draw.indexRange.isEmpty else { continue }
            let batch: (owner: BatchedMesh, snapshot: BatchedMesh.DrawSnapshot)?
            let localBounds: AxisAlignedBox?
            let worldBounds: AxisAlignedBox?
            var visible = !scissorIsEmpty && drawsInView
            if let owner = node.batch, let snapshot = try owner.drawSnapshot(world: node.worldTransform, relativeWorld: preciseWorld) {
                let objects = snapshot.objects
                guard !objects.isEmpty else { continue }
                batch = (owner, snapshot)
                localBounds = nil
                worldBounds = owner.enclosing(snapshot)
                let isRelative = preciseWorld != nil
                let bounds = isRelative ? owner.enclosing(snapshot, cameraRelative: true) : worldBounds
                let frustum = isRelative ? relativeFrustum : worldFrustum
                let explicitBounds = node.cullingBounds.flatMap { box in
                    if let preciseWorld { return try? box.transformed(by: preciseWorld) }
                    return try? box.transformed(by: node.worldTransform)
                }
                if drawsInView, node.frustumCulled, let frustum, let overall = explicitBounds ?? bounds,
                   frustum.classify(box: overall) == .outside {
                    visible = false
                    statistics.culledMeshCount += objects.count
                } else if visible {
                    let back = viewBasis.back
                    let context = BatchedMesh.SortContext(
                        pass: surfaceAttributes ? .surfaceAttributes : .color,
                        clipFromWorld: clipFromWorld, isBlended: colorRenderer.isBlended
                    )
                    let plan = try owner.plan(
                        snapshot, frustum: node.frustumCulled && snapshot.perObjectFrustumCulled ? frustum : nil,
                        context: context, sortingBack: SIMD3(back.x, back.y, back.z), cameraRelative: isRelative
                    )
                    statistics.culledMeshCount += plan.culledCount
                    // An arbitrary alternate fragment may not support Metal ICBs.
                    // Reuse the plan's direct runs for the override color pass.
                    draw.batch = try owner.acquire(
                        plan, commandBuffer: commandBuffer, bytesUsed: &batchBytes,
                        useIndirect: replacement == nil && colorRenderer.viewportInputs.isEmpty
                    )
                    visible = draw.batch != nil
                }
            } else {
                batch = nil
                localBounds = try renderer.localCullingBounds(of: node, revisionBounds: draw.dynamicRevision?.bounds)
                worldBounds = try? localBounds?.transformed(by: node.worldTransform)
                if drawsInView, node.frustumCulled {
                    let outside: Bool
                    if let precise, let localBounds {
                        outside = cameraFrustum(for: precise.clip)?.classify(box: localBounds) == .outside
                    } else if let worldBounds, let worldFrustum {
                        outside = worldFrustum.classify(box: worldBounds) == .outside
                    } else {
                        outside = false
                    }
                    if outside {
                        visible = false
                        statistics.culledMeshCount += 1
                    }
                }
            }
            if visible, !colorRenderer.viewportInputs.isEmpty {
                if graph.viewportSnapshot?.source == .opaquePass, !colorRenderer.drawsAfterOpaque {
                    throw SceneViewportSnapshot.Failure.captureReader
                }
                try colorRenderer.viewportInputs.validate(in: graph.viewportSnapshot)
            }
            let order = SceneColorOrder(
                phase: colorRenderer.isBlended ? .blended : .opaque, kind: node.batch == nil ? .mesh : .batch, node: node,
                nodeIndex: nodeIndex, clipPosition: draw.clip.columns.3, depthMode: depthMode,
                recordsOpaqueDepth: ordering.recordsOpaqueDepth, logarithmicDepth: logarithmicDepth
            )
            let environmentPipeline = try visible && sceneEnvironment.lighting != nil
                ? colorRenderer.environmentPipeline(surfaceAttributes: surfaceAttributes) : nil
            draws.append(GraphDraw(
                renderer: renderer, replacement: replacement, draw: draw, environmentPipeline: environmentPipeline,
                shadowClips: shadowClips, transmittedCaster: transmittedCaster, punctualClips: punctualClips,
                relativePunctualCaster: relativeWorld != nil, localBounds: localBounds, worldBounds: worldBounds,
                visible: visible, order: order, batch: batch
            ))
        }

        let readsViewport = draws.contains { $0.visible && !$0.colorRenderer.viewportInputs.isEmpty }
        let viewportTextures = try readsViewport ? graph.viewportSnapshot?.prepare(device: device) : nil
        let primitiveSoftParticles: PrimitiveSoftParticleFrame?
        if !scissorIsEmpty, primitiveDraws.contains(where: { $0.primitive.softParticleFade != nil }) {
            guard let depth = graph.viewportSnapshot?.depth else { throw SceneViewportSnapshot.Failure.missingDepth }
            primitiveSoftParticles = try PrimitiveSoftParticleFrame(depth: depth, view: graphView)
        } else {
            primitiveSoftParticles = nil
        }
        if draws.contains(where: { $0.visible && $0.colorRenderer.readsSoftParticles }) {
            _ = try graphView.softParticleProjection()
        }
        if let toonOutline, toonOutline.isEnabled, !surfaceAttributes {
            for index in draws.indices where draws[index].visible {
                let renderer = draws[index].colorRenderer
                let opacity: Float
                if toonOutline.drawsAfterScene {
                    guard draws[index].renderer.rawProgram == nil,
                          let supported = renderer.outlineEffectOpacity(parameters: draws[index].colorDraw.parameters) else { continue }
                    opacity = supported
                } else {
                    guard case .toon? = renderer.classic?.material.model else { continue }
                    opacity = 1
                }
                let id = nodes[draws[index].order.nodeIndex].id
                guard let style = toonOutline.style(for: id, materialOpacity: opacity) else { continue }
                guard !usesMultipleRenderTargets else { throw ToonOutlinePass.Failure.colorAttachments }
                draws[index].toonOutline = (try renderer.toonOutlinePipeline(), style)
            }
        }

        if draws.contains(where: { $0.environmentPipeline != nil }) {
            sceneEnvironment.retainIfNeeded(on: commandBuffer)
        }

        // The first renderer with punctual bindings owns the shared tables for this light list.
        let lightTableOwner = surfaceAttributes ? nil : draws.first { $0.colorRenderer.punctualShadowFallback != nil }?.colorRenderer
        let usesDiffuseLighting = !surfaceAttributes && (lightTableOwner != nil || draws.contains { $0.colorRenderer.usesNativeLambert })
        if usesDiffuseLighting, let buffer = probeGrids?.buffer, !commandBuffer.retainedReferences {
            commandBuffer.addCompletedHandler { _ in withExtendedLifetime(buffer) {} }
        }
        let lightTables = try lightTableOwner.map {
            try PackedPunctualLight.applyTables(to: &lights, from: resolvedLights, device: device, cache: $0.lightTableCache)
        }
        lightTables?.retainIfNeeded(on: commandBuffer)
        let rectangleLTC: RectangleLightLTC? = if !surfaceAttributes,
            lights.contains(where: { $0.positionKind.w == 3 }),
            draws.contains(where: { $0.visible && $0.renderer.physicallyBased?.shadingModel == .physicallyBased }) {
            try RectangleLightLTC.load(device: device)
        } else {
            nil
        }
        rectangleLTC?.retainIfNeeded(on: commandBuffer)

        // Bind punctual depth only when a PBR or lit classic surface can receive it.
        let receivesPunctualLights = draws.contains { $0.colorRenderer.usesPunctualLights }
        let usesPunctualShadows = receivesPunctualLights && !shadowSlots.renderedSlices.isEmpty
        let punctualUse: ShadowMapUpdateCache.Use? = if let punctualShadowMaps, usesPunctualShadows {
            try punctualShadowMaps.beginUse(commandBuffer: commandBuffer, layout: shadowSlots.cacheLayout)
        } else {
            nil
        }
        var accessedShadows = false
        defer { if !accessedShadows { punctualUse?.cancelBeforeAccess() } }
        let directionalUse: ShadowMapUpdateCache.Use? = if let shadowMap, draws.contains(where: { $0.colorRenderer.receivesShadows }) {
            try shadowMap.beginUse(commandBuffer: commandBuffer, usingRelativeView: precision != nil)
        } else {
            nil
        }
        defer { if !accessedShadows { directionalUse?.cancelBeforeAccess() } }
        let punctualShadowPasses: [(slot: Int, slice: Int)] = if let punctualUse {
            punctualUse.writesAll ? shadowSlots.renderedSlices : shadowSlots.renderedSlices.filter { punctualUse.writes(slice: $0.slice) }
        } else {
            []
        }
        let drawsDirectionalShadows = directionalUse?.writesDepth == true
        for item in draws where (drawsDirectionalShadows && !item.shadowClips.isEmpty && item.transmittedCaster == nil)
            || (!punctualShadowPasses.isEmpty && !item.punctualClips.isEmpty) {
            if let graphNode = item.renderer.graphShadow?.cameraNode {
                throw Failure.graphShadowCameraInput(nodes[item.order.nodeIndex].id, graphNode)
            }
        }
        for index in draws.indices {
            guard let batch = draws[index].batch else { continue }
            func packet(for view: simd_float4x4, pass: BatchedMesh.SortPass) throws -> BatchedMesh.Packet? {
                let frustum = batch.snapshot.perObjectFrustumCulled ? try? ViewFrustum(clipFromWorld: view) : nil
                let context = BatchedMesh.SortContext(pass: pass, clipFromWorld: view, isBlended: false)
                let back = Self.viewBasis(clipFromWorld: view, directionToCamera: nil, depthMode: .forward).back
                let plan = try batch.owner.plan(batch.snapshot, frustum: frustum, context: context,
                                                sortingBack: SIMD3(back.x, back.y, back.z))
                statistics.culledShadowCount += plan.culledCount
                return try batch.owner.acquire(plan, commandBuffer: commandBuffer, bytesUsed: &batchBytes)
            }
            if drawsDirectionalShadows, !draws[index].shadowClips.isEmpty, draws[index].transmittedCaster == nil {
                for (slice, view) in shadowViews.enumerated() {
                    draws[index].directionalBatches[slice] = try packet(for: view, pass: .directionalShadow(cascade: slice))
                }
            }
            if !draws[index].punctualClips.isEmpty {
                for (slot, _) in punctualShadowPasses {
                    draws[index].punctualBatches[slot] = try packet(for: shadowSlots.views[slot], pass: .punctualShadow(slice: slot))
                }
            }
        }

        let transmission: TransmissionPasses?
        if !surfaceAttributes, !scissorIsEmpty, draws.contains(where: { $0.colorRenderer.requiresTransmissionBackdrop }) {
            transmission = try transmissionPasses(
                pass, color: color, size: size, backdrop: transmissionBackdrop, clipFromWorld: clipFromWorld,
                viewport: viewport, commandBuffer: commandBuffer
            )
        } else {
            transmission = nil
        }
        let opaqueCapture = try opaqueSnapshot.map { snapshot in
            if let depth = snapshot.depth, [shadowMap?.texture, punctualShadowMaps?.texture].contains(where: {
                $0.map { TextureTransfer.sharesStorage(depth, $0) } ?? false
            }) {
                throw SceneViewportSnapshot.Failure.feedback
            }
            return try SceneViewportSnapshot.OpaqueCapture(
                snapshot: snapshot, pass: pass, transmissionBackdrop: transmission?.frame.backdrop
            )
        }

        // A validation-only call stops after every CPU-side check that precedes encoding and
        // before any encoder, deformation lease or indirect-argument pass touches the command
        // buffer. The shadow-use defers above cancel their update-cache reservations.
        if validateOnly {
            if drawsDirectionalShadows { try shadowMap?.validate(commandBuffer: commandBuffer) }
            if !punctualShadowPasses.isEmpty { try punctualShadowMaps?.validate(commandBuffer: commandBuffer) }
            return statistics
        }

        // Color, auxiliary draws and motion use the same leased vertex revision.
        var preparedDeformations: [ObjectIdentifier: any MTLBuffer] = [:]
        for index in draws.indices {
            let renderer = draws[index].renderer
            if !usesStencil || draws[index].visible {
                draws[index].draw.deformedVertices = try renderer.prepareDeformation(
                    for: commandBuffer, cached: &preparedDeformations
                )
            }
            let needsColors = draws[index].replacement?.renderer.readsVertexColors == true
            try renderer.acquireDrawInputs(&draws[index].draw, for: commandBuffer, requiringColors: needsColors)
        }
        let motion = try motionVectors?.prepare(
            draws: scissorIsEmpty ? [] : draws.compactMap { $0.motionDraw(nodeID: nodes[$0.order.nodeIndex].id) },
            size: size, viewport: viewport, depth: pass.depthAttachment.texture!
        )
        if !commandBuffer.retainedReferences, materialOverride != nil {
            let retained = RetainedOverrideRenderers(renderers: draws.compactMap { $0.replacement?.renderer })
            commandBuffer.addCompletedHandler { _ in withExtendedLifetime(retained) {} }
        }

        let spotShadowMoments: (any MTLTexture)? = if let punctualShadowMaps, usesPunctualShadows, !shadowSlots.varianceSlices.isEmpty {
            try punctualShadowMaps.prepareVarianceMoments()
        } else {
            nil
        }
        accessedShadows = true
        if let shadowMap, drawsDirectionalShadows {
            let coverageView = graphView.shadowViewport(resolution: shadowMap.resolution)
            for (slice, view) in shadowViews.enumerated() {
                let depthEncoder = try shadowMap.makeEncoder(commandBuffer: commandBuffer, slice: slice) { descriptor in
                    try gpuTiming?.record(.directionalShadow(cascade: slice), into: descriptor)
                }
                let lightView = try? ViewFrustum(clipFromWorld: view)
                for caster in draws where !caster.shadowClips.isEmpty && caster.transmittedCaster == nil {
                    var draw = caster.draw
                    if caster.batch != nil {
                        guard let batch = caster.directionalBatches[slice] else { continue }
                        draw.batch = batch
                    } else if caster.isOutsideDirectionalShadow(slice: slice, lightView: lightView) {
                        statistics.culledShadowCount += 1
                        continue
                    }
                    caster.renderer.encodeShadowCaster(
                        draw, clip: caster.shadowClips[slice], into: depthEncoder, coverageView: coverageView
                    ) { skinned, masked in
                        shadowMap.setPipeline(on: depthEncoder, skinned: skinned, masked: masked)
                    }
                    statistics.shadowDrawCount += draw.drawCallCount
                    statistics.shadowTriangleCount += caster.renderer.submittedTriangleCount(draw)
                    statistics.indirectBatchExecutionCount += draw.batch?.bundle?.executionCount ?? 0
                    statistics.indirectArgumentDrawCount += draw.indirect?.count ?? 0
                }
                depthEncoder.endEncoding()
                statistics.shadowPassCount += 1
                if let transmitted = shadowMap.transmittedShadows {
                    var peelView = coverageView
                    peelView.windowDepth = SIMD2(0, 1)
                    peelView.depthMode = .forward
                    let glass = draws.compactMap { caster -> (GraphDraw, DirectionalTransmissionPass.Caster)? in
                        guard let optics = caster.transmittedCaster else { return nil }
                        if caster.isOutsideDirectionalShadow(slice: slice, lightView: lightView) {
                            statistics.culledShadowCount += 1
                            return nil
                        }
                        return (caster, optics)
                    }
                    // Clear every layer when refreshing an on-demand map. Removed
                    // panes must leave no stale colored shadow in the cache.
                    for layer in 0..<transmitted.layerCount {
                        let encoder = try transmitted.makeEncoder(slice: slice, layer: layer, commandBuffer: commandBuffer)
                        for (caster, optics) in glass {
                            caster.renderer.encodeTransmittedShadow(
                                caster.draw, clip: caster.shadowClips[slice], optics: optics,
                                pass: transmitted, view: peelView, into: encoder
                            )
                            statistics.shadowDrawCount += caster.draw.drawCallCount
                            statistics.shadowTriangleCount += caster.renderer.submittedTriangleCount(caster.draw)
                        }
                        encoder.endEncoding()
                        statistics.shadowPassCount += 1
                    }
                }
            }
            try shadowMap.encodeMoments(commandBuffer: commandBuffer)
        }

        if let punctualShadowMaps {
            let coverageView = graphView.shadowViewport(resolution: punctualShadowMaps.resolution)
            for (slot, slice) in punctualShadowPasses {
                let depthEncoder = try punctualShadowMaps.makeEncoder(slice: slice, commandBuffer: commandBuffer) { descriptor in
                    try gpuTiming?.record(.punctualShadow(slice: slice), into: descriptor)
                }
                let lightView = try? ViewFrustum(clipFromWorld: shadowSlots.views[slot])
                for caster in draws where !caster.punctualClips.isEmpty {
                    var draw = caster.draw
                    if caster.batch != nil {
                        guard let batch = caster.punctualBatches[slot] else { continue }
                        draw.batch = batch
                    } else {
                        let view: ViewFrustum?
                        let bounds: AxisAlignedBox?
                        if caster.relativePunctualCaster, shadowSlots.hasRelativeLightView(slot: slot) {
                            view = try? ViewFrustum(clipFromWorld: caster.punctualClips[slot])
                            bounds = caster.localBounds
                        } else {
                            view = lightView
                            bounds = caster.worldBounds
                        }
                        if let view, let bounds, view.classify(box: bounds) == .outside {
                            statistics.culledShadowCount += 1
                            continue
                        }
                    }
                    caster.renderer.encodeShadowCaster(
                        draw, clip: caster.punctualClips[slot], into: depthEncoder, coverageView: coverageView
                    ) { skinned, masked in
                        punctualShadowMaps.setPipeline(on: depthEncoder, skinned: skinned, masked: masked)
                    }
                    statistics.shadowDrawCount += draw.drawCallCount
                    statistics.shadowTriangleCount += caster.renderer.submittedTriangleCount(draw)
                    statistics.indirectBatchExecutionCount += draw.batch?.bundle?.executionCount ?? 0
                    statistics.indirectArgumentDrawCount += draw.indirect?.count ?? 0
                }
                depthEncoder.endEncoding()
                statistics.shadowPassCount += 1
            }
            let momentSlices = shadowSlots.varianceSlices.filter { punctualUse?.writes(slice: Int($0.x)) == true }
            try punctualShadowMaps.encodeMoments(slices: momentSlices, commandBuffer: commandBuffer)
        }

        // The color passes draw only what the camera can see; shadows above used every caster.
        // The attribute prepass draws what writes depth: everything except blended meshes.
        var colorDraws = scissorIsEmpty ? [] : draws.filter(\.visible)
        for item in colorDraws { Self.count(item, into: &statistics) }
        func precedes(_ lhs: SceneColorOrder, _ rhs: SceneColorOrder) -> Bool {
            ordering.precedes(lhs, rhs, nodes: nodes, depthMode: depthMode)
        }
        if ordering.recordsOpaqueDepth || colorDraws.contains(where: { $0.order.needsSorting }) {
            colorDraws.sort { precedes($0.order, $1.order) }
        }
        let orderedPrimitives = scissorIsEmpty ? [] : primitiveDraws.sorted { precedes($0.order, $1.order) }
        statistics.drawnPrimitiveCount = orderedPrimitives.count
        for draw in orderedPrimitives { Self.count(draw, into: &statistics) }
        let drawsOutlinesAfterScene = toonOutline?.drawsAfterScene == true
        let finishingOutlines: [GraphDraw] = if drawsOutlinesAfterScene {
            colorDraws.compactMap { item -> (draw: GraphDraw, order: SceneColorOrder)? in
                guard let outline = item.toonOutline else { return nil }
                let index = item.order.nodeIndex
                let blended = outline.style.premultiplied.w < 1 || item.colorRenderer.isBlended
                let order = SceneColorOrder(
                    phase: blended ? .blended : .opaque, kind: item.order.kind, node: nodes[index], nodeIndex: index,
                    clipPosition: item.draw.clip.columns.3, depthMode: depthMode, recordsOpaqueDepth: ordering.recordsOpaqueDepth
                )
                return (item, order)
            }.sorted { precedes($0.order, $1.order) }.map(\.draw)
        } else {
            []
        }
        // Render hooks and the inspector see every color or attribute draw in encode order. The index
        // counts all draws, hooked or not, so it names a draw's place in the encode. An encode that
        // splits into opaque and overlay passes keeps one count.
        var drawIndex = 0
        // One scan per encode keeps an unobserved draw from copying its node.
        let observing = inspector != nil || nodes.contains { $0.onBeforeEncode != nil || $0.onAfterEncode != nil }
        func observed(_ nodeIndex: Int) -> ResolvedSceneNode? {
            guard observing else { return nil }
            let node = nodes[nodeIndex]
            return inspector != nil || node.onBeforeEncode != nil || node.onAfterEncode != nil ? node : nil
        }
        func objectContext(
            _ node: ResolvedSceneNode, kind: SceneRenderOrdering.Kind, parameters: SIMD4<Float>,
            overridden: Bool, blended: Bool, instanceCount: Int?, at index: Int
        ) -> SceneRenderObjectContext {
            SceneRenderObjectContext(
                nodeID: node.id, kind: kind, pass: info.pass, drawIndex: index, viewIndex: info.viewIndex,
                viewCount: info.viewCount, depthMode: depthMode, worldTransform: node.worldTransform,
                materialParameters: parameters, usesMaterialOverride: overridden, isBlended: blended,
                instanceCount: instanceCount
            )
        }
        func meshContext(_ item: GraphDraw, node: ResolvedSceneNode, at index: Int) -> SceneRenderObjectContext {
            objectContext(
                node, kind: item.order.kind, parameters: item.colorDraw.parameters, overridden: item.replacement != nil,
                blended: item.colorRenderer.isBlended, instanceCount: node.batch == nil ? node.instances?.count : nil,
                at: index
            )
        }
        func primitiveContext(_ draw: PrimitiveDraw, node: ResolvedSceneNode, at index: Int) -> SceneRenderObjectContext {
            objectContext(
                node, kind: draw.order.kind, parameters: draw.tint, overridden: draw.isMaterialOverride,
                blended: draw.isBlended, instanceCount: nil, at: index
            )
        }
        func meshCounts(_ item: GraphDraw) -> SceneDrawStatistics {
            var counts = SceneDrawStatistics()
            Self.count(item, into: &counts)
            return counts
        }
        func primitiveCounts(_ draw: PrimitiveDraw) -> SceneDrawStatistics {
            var counts = SceneDrawStatistics()
            counts.drawnPrimitiveCount = 1
            Self.count(draw, into: &counts)
            return counts
        }
        /// Reports a finished draw to the node's after hook, then to the inspector with the draw's own counts.
        func didEncode(_ context: SceneRenderObjectContext, node: ResolvedSceneNode, counts: () -> SceneDrawStatistics) {
            node.onAfterEncode?(context)
            if let inspector { inspector.sceneEncode(info, didEncode: context, statistics: counts()) }
        }
        func encodeDraws(
            _ draws: [GraphDraw], primitives: [PrimitiveDraw], pass: MTLRenderPassDescriptor,
            transmission: TransmissionFrame? = nil, timingPass: SceneGPUTimingFrame.Pass = .color,
            finishingOutlines: [GraphDraw] = []
        ) throws {
            let visibility = try occlusionQueries?.prepare(
                pass, meshIndices: draws.map(\.order.nodeIndex), primitiveIndices: primitives.map(\.order.nodeIndex),
                nodes: nodes
            )
            let descriptor: MTLRenderPassDescriptor
            if let gpuTiming {
                if let visibility {
                    descriptor = visibility.descriptor
                } else if let copied = pass.copy() as? MTLRenderPassDescriptor {
                    descriptor = copied
                } else {
                    throw SceneGPUTimingError.resource
                }
                try gpuTiming.record(timingPass, into: descriptor)
            } else {
                descriptor = visibility?.descriptor ?? pass
            }
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else {
                throw Failure.encoder
            }
            if let viewport { encoder.setViewport(viewport) }
            if let scissor, !scissorIsEmpty { encoder.setScissorRect(scissor) }
            logarithmicDepth?.bind(into: encoder)
            if surfaceAttributes {
                encoder.label = "Frostfield surface attributes"
                for item in draws {
                    item.draw.palette?.retainBufferIfNeeded(on: commandBuffer)
                    let node = observed(item.order.nodeIndex)
                    let context = node.map { meshContext(item, node: $0, at: drawIndex) }
                    drawIndex += 1
                    if let node, let context { node.onBeforeEncode?(context) }
                    try item.colorRenderer.encodeSurfaceAttributes(
                        item.colorDraw, into: encoder, sceneEnvironment: sceneEnvironment, graphView: graphView,
                        environmentPipeline: item.environmentPipeline, viewportTextures: viewportTextures
                    )
                    if let node, let context { didEncode(context, node: node) { meshCounts(item) } }
                }
                encoder.endEncoding()
                return
            }
            if usesDiffuseLighting {
                bindLightProbe(probeLighting, grids: probeGrids, layers: graph.cameraLayers, into: encoder)
            }
            if let lightTables, let fallback = draws.lazy.compactMap(\.colorRenderer.punctualShadowFallback).first {
                if let punctualShadowMaps, usesPunctualShadows {
                    bindPunctualLights(
                        lights, shadowTexture: punctualShadowMaps.texture, shadowViews: shadowSlots.sliceViews,
                        shadowMoments: spotShadowMoments, relativeLightPositions: shadowSlots.relativeLightPositions,
                        tables: lightTables, rectangleLTC: rectangleLTC?.texture, into: encoder
                    )
                } else {
                    // No sampled light holds a shadow slot.
                    bindPunctualLights(
                        lights, shadowTexture: fallback, shadowViews: [], relativeLightPositions: shadowSlots.relativeLightPositions,
                        tables: lightTables, rectangleLTC: rectangleLTC?.texture, into: encoder
                    )
                }
            }
            var punctualShadowsSuppressed = false
            func bindScenePunctualLights(suppressShadows: Bool) {
                guard let lightTables, let fallback = draws.lazy.compactMap(\.colorRenderer.punctualShadowFallback).first else {
                    return
                }
                let bound = suppressShadows ? lights.map { light in
                    var light = light
                    light.shadowReceiver.x = 0
                    return light
                } : lights
                if let punctualShadowMaps, usesPunctualShadows {
                    bindPunctualLights(
                        bound, shadowTexture: punctualShadowMaps.texture, shadowViews: shadowSlots.sliceViews,
                        shadowMoments: spotShadowMoments, relativeLightPositions: shadowSlots.relativeLightPositions,
                        tables: lightTables, rectangleLTC: rectangleLTC?.texture, into: encoder
                    )
                } else {
                    bindPunctualLights(
                        bound, shadowTexture: fallback, shadowViews: [],
                        relativeLightPositions: shadowSlots.relativeLightPositions,
                        tables: lightTables, rectangleLTC: rectangleLTC?.texture, into: encoder
                    )
                }
            }
            // Merge both sorted lists. For equal keys on the same node, draw the mesh first.
            var querying = false
            func selectVisibility(_ offset: Int?) {
                if let offset {
                    encoder.setVisibilityResultMode(.boolean, offset: offset)
                    querying = true
                } else if querying {
                    encoder.setVisibilityResultMode(.disabled, offset: 0)
                    querying = false
                }
            }
            func encodePrimitive(_ primitive: PrimitiveDraw) {
                selectVisibility(visibility?.primitives[primitive.order.nodeIndex])
                primitive.retainBuffersIfNeeded(on: commandBuffer)
                let node = observed(primitive.order.nodeIndex)
                let context = node.map { primitiveContext(primitive, node: $0, at: drawIndex) }
                drawIndex += 1
                if let node, let context { node.onBeforeEncode?(context) }
                primitive.encode(into: encoder, depthMode: depthMode, softParticles: primitiveSoftParticles)
                if let node, let context { didEncode(context, node: node) { primitiveCounts(primitive) } }
            }
            func encodeOutline(_ item: GraphDraw) {
                guard let outline = item.toonOutline else { return }
                item.colorRenderer.encodeToonOutline(item.colorDraw, style: outline.style, pipeline: outline.pipeline, into: encoder)
                statistics.toonOutlineDrawCount += item.draw.drawCallCount
                statistics.submittedTriangleCount += item.colorRenderer.submittedTriangleCount(item.colorDraw)
            }
            var pendingPrimitives = primitives[...]
            for item in draws {
                while let next = pendingPrimitives.first, precedes(next.order, item.order) {
                    encodePrimitive(next)
                    pendingPrimitives.removeFirst()
                }
                selectVisibility(visibility?.meshes[item.order.nodeIndex])
                item.draw.palette?.retainBufferIfNeeded(on: commandBuffer)
                let node = observed(item.order.nodeIndex)
                let context = node.map { meshContext(item, node: $0, at: drawIndex) }
                drawIndex += 1
                if let node, let context { node.onBeforeEncode?(context) }
                if !drawsOutlinesAfterScene { encodeOutline(item) }
                if item.colorDraw.suppressesShadows != punctualShadowsSuppressed {
                    bindScenePunctualLights(suppressShadows: item.colorDraw.suppressesShadows)
                    punctualShadowsSuppressed = item.colorDraw.suppressesShadows
                }
                item.colorRenderer.encode(
                    item.colorDraw, into: encoder, shadowMap: shadowMap, transmission: transmission,
                    sceneEnvironment: sceneEnvironment, graphView: graphView, viewBasis: viewBasis,
                    screenInputs: pbrScreenInputs, environmentPipeline: item.environmentPipeline,
                    viewportTextures: viewportTextures
                )
                if let node, let context { didEncode(context, node: node) { meshCounts(item) } }
            }
            for primitive in pendingPrimitives { encodePrimitive(primitive) }
            if !finishingOutlines.isEmpty {
                selectVisibility(nil)
                for item in finishingOutlines { encodeOutline(item) }
            }
            encoder.endEncoding()
        }

        let opaquePass = transmission?.opaque ?? opaqueCapture?.opaque
        let overlayPass = transmission?.overlay ?? opaqueCapture?.overlay
        if let opaquePass, let overlayPass {
            let opaque = colorDraws.filter { !$0.colorRenderer.drawsAfterOpaque }
            let overlay = colorDraws.filter { $0.colorRenderer.drawsAfterOpaque }
            // Opaque points, lines and sprites are part of the backdrop.
            let opaquePrimitives = orderedPrimitives.filter { !$0.isBlended && !$0.isOverlay }
            let overlayPrimitives = orderedPrimitives.filter { $0.isBlended || $0.isOverlay }
            try encodeDraws(opaque, primitives: opaquePrimitives, pass: opaquePass, timingPass: .opaqueColor)
            if let transmission {
                try copyBackdrop(
                    from: color, attachment: pass.colorAttachments[0]!, to: transmission.frame.backdrop,
                    commandBuffer: commandBuffer
                )
            }
            try opaqueCapture?.encodeCopy(commandBuffer: commandBuffer)
            try encodeDraws(
                overlay, primitives: overlayPrimitives, pass: overlayPass, transmission: transmission?.frame,
                timingPass: transmission == nil ? .color : .transmissionColor, finishingOutlines: finishingOutlines
            )
        } else {
            try encodeDraws(colorDraws, primitives: orderedPrimitives, pass: pass, finishingOutlines: finishingOutlines)
        }
        try motion?.encode(commandBuffer: commandBuffer, scissor: scissor)
        directionalUse?.didEncode()
        punctualUse?.didEncode()
        return statistics
    }

    /// The counts one mesh draw adds to the color or attribute pass statistics.
    private static func count(_ item: GraphDraw, into statistics: inout SceneDrawStatistics) {
        let draw = item.draw
        statistics.drawnMeshCount += draw.drawCallCount
        statistics.submittedTriangleCount += item.renderer.submittedTriangleCount(draw)
        if draw.indirect == nil { statistics.drawnInstanceCount += draw.batch?.instanceCount ?? draw.instanceCount }
        statistics.indirectBatchExecutionCount += draw.batch?.bundle?.executionCount ?? 0
        statistics.indirectArgumentDrawCount += draw.indirect?.count ?? 0
    }

    /// The point, line and sprite counts one primitive draw adds, not its `drawnPrimitiveCount`.
    private static func count(_ draw: PrimitiveDraw, into statistics: inout SceneDrawStatistics) {
        if let indirect = draw.indirect {
            statistics.indirectArgumentDrawCount += indirect.count
            return
        }
        switch draw.elements {
        case .points(let range): statistics.submittedPointCount += range.count
        case .lines(let lines): statistics.submittedLineSegmentCount += lines.segments.count
        case .sprites(let count):
            statistics.submittedSpriteCount += count
            statistics.submittedTriangleCount += count * 2
        }
    }

    private struct GraphDraw {
        let renderer: MeshRenderer
        let replacement: (renderer: MeshRenderer, parameters: SIMD4<Float>)?
        var draw: PreparedDraw
        let environmentPipeline: (any MTLRenderPipelineState)?
        var toonOutline: (pipeline: ToonOutlinePipeline, style: ToonOutlinePass.DrawStyle)? = nil
        var colorRenderer: MeshRenderer { replacement?.renderer ?? renderer }
        var colorDraw: PreparedDraw {
            guard let replacement else { return draw }
            var draw = draw
            draw.parameters = replacement.parameters
            return draw
        }

        func motionDraw(nodeID: UUID) -> RasterMotionVectors.Draw? {
            let material = colorRenderer.renderState
            guard visible, material.depthTest, material.depthWrite, material.depthOffset == nil,
                  colorRenderer.rawProgram == nil, draw.indirectSource == nil, batch == nil,
                  !draw.indexRange.isEmpty else { return nil }
            let instances = draw.instances.map {
                RasterMotionVectors.Instances(owner: $0, layout: $0.layoutID, transforms: $0.transforms)
            }
            let vertices = renderer.vertexBuffer(for: draw)
            let vertexSnapshotLength: Int? = if let revision = draw.dynamicRevision {
                revision.geometry.vertices.count * MemoryLayout<PackedMeshVertex>.stride
            } else if draw.curveVertices != nil || draw.deformedVertices != nil {
                vertices.length
            } else {
                nil
            }
            return RasterMotionVectors.Draw(
                id: nodeID,
                entry: .init(
                    renderer: renderer, clip: draw.clip, indexRange: draw.indexRange,
                    topologyRevision: draw.dynamicRevision?.topologyRevision,
                    instances: instances, palette: draw.palette, morph: draw.morph
                ),
                vertices: vertices, vertexSnapshotLength: vertexSnapshotLength,
                indices: draw.dynamicBuffers?.indices ?? renderer.indices, skinInfluences: renderer.skinInfluences,
                morphOffsets: renderer.morph?.offsets, winding: draw.frontFacing, cull: material.cullMode
            )
        }
        /// The caster's clip transform for each directional map slice; empty when it casts no shadow.
        let shadowClips: [simd_float4x4]
        let transmittedCaster: DirectionalTransmissionPass.Caster?
        let punctualClips: [simd_float4x4]
        let relativePunctualCaster: Bool
        let localBounds: AxisAlignedBox?
        /// The world box that culls this draw, or nil when it is never culled.
        let worldBounds: AxisAlignedBox?
        /// Whether the color passes draw it. Shadow passes ignore the camera frustum.
        let visible: Bool
        let order: SceneColorOrder
        let batch: (owner: BatchedMesh, snapshot: BatchedMesh.DrawSnapshot)?
        var directionalBatches: [Int: BatchedMesh.Packet] = [:]
        var punctualBatches: [Int: BatchedMesh.Packet] = [:]

        func isOutsideDirectionalShadow(slice: Int, lightView: ViewFrustum?) -> Bool {
            let view: ViewFrustum?
            let bounds: AxisAlignedBox?
            if draw.relativeDirectional != nil {
                view = try? ViewFrustum(clipFromWorld: shadowClips[slice])
                bounds = localBounds
            } else {
                view = lightView
                bounds = worldBounds
            }
            guard let view, let bounds else { return false }
            return view.classify(box: bounds) == .outside
        }
    }

    private struct TransmissionFrame {
        let backdrop: any MTLTexture
        var clipFromWorld: simd_float4x4
        var viewportUV: SIMD4<Float>
    }

    private struct TransmissionPasses {
        let opaque: MTLRenderPassDescriptor
        let overlay: MTLRenderPassDescriptor
        let frame: TransmissionFrame
    }

    private static func transmissionPasses(
        _ pass: MTLRenderPassDescriptor,
        color: any MTLTexture,
        size: SIMD2<Int>,
        backdrop: (any MTLTexture)?,
        clipFromWorld: simd_float4x4,
        viewport: MTLViewport?,
        commandBuffer: any MTLCommandBuffer
    ) throws -> TransmissionPasses {
        guard let backdrop else { throw Failure.transmissionBackdrop }
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.transmissionCommandBuffer
        }
        let colorAttachment = pass.colorAttachments[0]!
        let depthAttachment = pass.depthAttachment!
        guard let depth = depthAttachment.texture,
              color.pixelFormat == .rgba16Float, !color.isFramebufferOnly,
              color.parent == nil, color.buffer == nil, color.heap == nil, color.iosurface == nil,
              isTransmissionAttachment(colorAttachment, texture: color),
              isTransmissionAttachment(depthAttachment, texture: depth),
              pass.stencilAttachment.texture == nil,
              (1..<8).allSatisfy({ pass.colorAttachments[$0].texture == nil }),
              pass.renderTargetArrayLength <= 1, pass.rasterizationRateMap == nil,
              attachmentExtent(color, colorAttachment) == size else {
            throw Failure.transmissionAttachments
        }
        let largestDimension = max(size.x, size.y)
        let mipLevels = Int.bitWidth - largestDimension.leadingZeroBitCount
        guard backdrop.device === commandBuffer.device, !TextureTransfer.sharesStorage(backdrop, color),
              backdrop.pixelFormat == .rgba16Float, backdrop.textureType == .type2D, backdrop.sampleCount == 1,
              backdrop.width == size.x, backdrop.height == size.y, backdrop.mipmapLevelCount == mipLevels,
              backdrop.usage.contains(.shaderRead), backdrop.storageMode != .memoryless,
              backdrop.hazardTrackingMode != .untracked, !backdrop.isFramebufferOnly,
              backdrop.parent == nil, backdrop.buffer == nil, backdrop.heap == nil, backdrop.iosurface == nil else {
            throw Failure.invalidTransmissionBackdrop
        }

        guard let opaque = pass.copy() as? MTLRenderPassDescriptor,
              let overlay = pass.copy() as? MTLRenderPassDescriptor else { throw Failure.transmissionAttachments }
        opaque.colorAttachments[0].storeAction = .store
        opaque.depthAttachment.storeAction = .store
        overlay.colorAttachments[0].loadAction = .load
        overlay.depthAttachment.loadAction = .load
        let viewportUV = viewport.map {
            SIMD4<Float>(
                Float($0.width / Double(size.x)), Float($0.height / Double(size.y)),
                Float($0.originX / Double(size.x)), Float($0.originY / Double(size.y))
            )
        } ?? SIMD4(1, 1, 0, 0)
        return TransmissionPasses(
            opaque: opaque, overlay: overlay,
            frame: TransmissionFrame(backdrop: backdrop, clipFromWorld: clipFromWorld, viewportUV: viewportUV)
        )
    }

    private static func isTransmissionAttachment(
        _ attachment: MTLRenderPassAttachmentDescriptor, texture: any MTLTexture
    ) -> Bool {
        guard texture.storageMode != .memoryless, texture.hazardTrackingMode != .untracked,
              texture.sampleCount == 1, texture.usage.contains(.renderTarget),
              attachment.resolveTexture == nil else { return false }
        return (attachment.loadAction == .clear || attachment.loadAction == .load)
            && (attachment.storeAction == .store || attachment.storeAction == .dontCare)
    }

    private static func copyBackdrop(
        from color: any MTLTexture, attachment: MTLRenderPassColorAttachmentDescriptor,
        to backdrop: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw Failure.transmissionBlitEncoder }
        blit.label = "Frostfield transmission backdrop"
        blit.copy(
            from: color, sourceSlice: attachment.slice, sourceLevel: attachment.level,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: attachment.depthPlane),
            sourceSize: MTLSize(width: backdrop.width, height: backdrop.height, depth: 1),
            to: backdrop, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin()
        )
        if backdrop.mipmapLevelCount > 1 { blit.generateMipmaps(for: backdrop) }
        blit.endEncoding()
    }

    /// One light record, matching `FrostfieldPunctualLight` in PhysicallyBasedShading.h:
    /// eight float4 slots with a 128-byte stride and no padding.
    package struct PackedPunctualLight {
        /// World position or rectangle center; w is the kind: 0 directional, 1 point, 2 spot, 3 rectangle.
        package var positionKind: SIMD4<Float> = .zero
        /// Unit emission direction along local -Z; w is the range in meters, or zero.
        package var directionRange: SIMD4<Float> = .zero
        /// Linear color; w is lux, candela, or nits by kind.
        package var colorIntensity: SIMD4<Float> = .zero
        /// Cosines of the inner and outer spot cone, 1 when ranged, and the shadow view's half-angle tangent.
        package var cone: SIMD4<Float> = .zero
        /// Unit rectangle width axis; w is the half width in meters. A point or spot light
        /// with a profile or cookie stores its local +X and one plus its profile layer.
        package var axisU: SIMD4<Float> = .zero
        /// Unit rectangle height axis; w is the half height in meters. A spot light with a
        /// cookie stores one plus its cookie layer and the tangent of its outer cone in x and y.
        package var axisV: SIMD4<Float> = .zero
        /// First shadow slice or -1, depth bias in meters, normal bias in texels, filter radius or -1 basic / -2 variance.
        package var shadow = SIMD4<Float>(-1, 0, 0, 0)
        /// Shadow strength in x. The remaining lanes align the Metal light record.
        package var shadowReceiver = SIMD4<Float>(1, 0, 0, 0)

        package init() {}

        init(_ resolved: ResolvedPunctualLight) {
            let light = resolved.light
            var range: Float?
            let kind: Float
            switch light.kind {
            case .directional:
                kind = 0
            case .point(let cutoff):
                kind = 1
                range = cutoff
            case .spot(let cutoff, let inner, let outer):
                kind = 2
                range = cutoff
                cone.x = cos(inner)
                cone.y = max(cos(outer), 0)
            case .rectangle(let width, let height):
                kind = 3
                axisU = SIMD4(resolved.right, width / 2)
                axisV = SIMD4(resolved.up, height / 2)
            }
            cone.z = range == nil ? 0 : 1
            positionKind = SIMD4(resolved.position, kind)
            directionRange = SIMD4(resolved.direction, range ?? 0)
            colorIntensity = SIMD4(light.color, light.intensity)
        }
    }

    /// Assigns punctual shadow slices and records their views.
    private struct PunctualShadowSlots {
        private struct RelativeLightView {
            let position: SIMD3<Double>
            let projection: simd_double4x4

            func casterClip(_ relativeWorld: simd_double4x4) -> simd_float4x4 {
                var model = relativeWorld
                model[3] -= SIMD4(position, 0)
                return HighPrecisionModelView.packed(projection * model)
            }
        }

        let maps: PunctualShadowMaps?
        let cameraOrigin: SIMD3<Double>?
        private var spots = 0
        private var points = 0
        /// World-to-clip shadow views in assignment order.
        private(set) var views: [simd_float4x4] = []
        /// The array slice that each view in `views` renders.
        private(set) var renderedSlices: [(slot: Int, slice: Int)] = []
        /// The variance spot slices and their filter radii, in the filter's upload layout.
        private(set) var varianceSlices: [SIMD2<UInt32>] = []
        private var groups: [ShadowMapUpdateCache.Layout.Group] = []
        private var assignments: [UUID: Int]?
        private var relativeViews: [Int: RelativeLightView] = [:]
        /// Parallel to the light records in high-precision frames. W marks rebased lights.
        private(set) var relativeLightPositions: [SIMD4<Float>] = []

        init(maps: PunctualShadowMaps?, cameraOrigin: SIMD3<Double>?) {
            self.maps = maps
            self.cameraOrigin = cameraOrigin
        }

        func hasRelativeLightView(slot: Int) -> Bool { relativeViews[slot] != nil }

        func casterClips(world: simd_float4x4, relativeWorld: simd_double4x4?) -> [simd_float4x4] {
            views.enumerated().map { slot, view in
                if let relativeWorld, let relative = relativeViews[slot] {
                    return relative.casterClip(relativeWorld)
                }
                return view * world
            }
        }

        /// Every slice's view, indexed by slice, for the PBR fragment.
        var sliceViews: [simd_float4x4] {
            var result = Array(repeating: matrix_identity_float4x4, count: maps?.sliceCount ?? 1)
            for (slot, slice) in renderedSlices { result[slice] = views[slot] }
            return result
        }

        var cacheLayout: ShadowMapUpdateCache.Layout {
            .init(groups: groups)
        }

        mutating func preserveAssignments(for nodes: [ResolvedSceneNode], commandBuffer: any MTLCommandBuffer) {
            guard let maps, maps.usesCachedUpdates else { return }
            let previous = maps.existingSlots(commandBuffer: commandBuffer)
            let spots = Array(0..<maps.spotCapacity)
            let points = (0..<maps.pointCapacity).map { maps.spotCapacity + 6 * $0 }
            let eligible: [(id: UUID, slots: [Int])] = nodes.compactMap { node in
                guard let light = node.light?.light, light.shadow != nil,
                      light.intensity > 0, light.color != .zero else { return nil }
                switch light.kind {
                case .directional: return nil
                case .spot: return (node.id, spots)
                case .point, .rectangle: return (node.id, points)
                }
            }
            var assignments: [UUID: Int] = [:]
            var occupied = Set<Int>()
            for (id, slots) in eligible {
                guard let slot = previous[id], slots.contains(slot) else { continue }
                assignments[id] = slot
                occupied.insert(slot)
            }
            for (id, slots) in eligible where assignments[id] == nil {
                guard let slot = slots.first(where: { !occupied.contains($0) }) else { continue }
                assignments[id] = slot
                occupied.insert(slot)
            }
            self.assignments = assignments
        }

        mutating func pack(
            _ resolved: ResolvedPunctualLight, id: UUID, preciseLight: HighPrecisionModelView.LightPosition?
        ) throws -> PackedPunctualLight {
            var packed = PackedPunctualLight(resolved)
            let lightPosition = cameraOrigin.flatMap { preciseLight?.relative(to: $0) }
            if cameraOrigin != nil {
                let relative = lightPosition.map { SIMD4(SIMD3<Float>($0), 1) } ?? .zero
                guard MeshRenderer.isFinite(relative) else { throw Failure.transform }
                relativeLightPositions.append(relative)
            }
            guard let maps, let shadow = resolved.light.shadow else { return packed }
            let firstView = views.count
            let firstSlice: Int
            let tangent: Double
            let directions: [SIMD3<Float>]
            switch resolved.light.kind {
            case .directional:
                return packed
            case .spot(_, _, let outer):
                guard let slice = nextSlice(for: id, spot: true, maps: maps) else { return packed }
                firstSlice = slice
                tangent = maps.projectionTangent(covering: tan(Double(outer)), shadow: shadow)
                directions = [resolved.direction]
            case .point, .rectangle:
                guard let slice = nextSlice(for: id, spot: false, maps: maps) else { return packed }
                firstSlice = slice
                tangent = maps.projectionTangent(covering: 1, shadow: shadow)
                directions = PunctualShadowMaps.faceDirections
            }
            var preciseProjections: [ShadowMapUpdateCache.Layout.Group.RelativeProjection] = []
            for (face, direction) in directions.enumerated() {
                let view = try PunctualShadowMaps.projection(
                    origin: resolved.position, forward: direction, tangent: tangent, near: shadow.nearPlane
                )
                if let preciseLight, let lightPosition {
                    let projection = PunctualShadowMaps.projectionMatrix(
                        forward: direction, tangent: tangent, near: shadow.nearPlane
                    )
                    relativeViews[views.count] = RelativeLightView(position: lightPosition, projection: projection)
                    if assignments != nil {
                        // Keep cache identity in the light's root coordinates, independent of the viewing camera.
                        let anchored = PunctualShadowMaps.projectionMatrix(
                            origin: preciseLight.offsetFromRoot, forward: direction, tangent: tangent,
                            near: shadow.nearPlane
                        )
                        preciseProjections.append(.init(origin: preciseLight.rootOrigin, clipFromOrigin: anchored))
                    }
                }
                renderedSlices.append((views.count, firstSlice + face))
                views.append(view)
            }
            let momentRadius = shadow.sampling == .variance ? shadow.varianceBlurRadius : nil
            if let momentRadius {
                varianceSlices.append(SIMD2(UInt32(firstSlice), UInt32(momentRadius)))
            }
            if assignments != nil {
                let end = firstSlice + views.count - firstView
                groups.append(.init(
                    lightID: id, slices: firstSlice..<end, views: views[firstView...], momentBlurRadius: momentRadius,
                    relativeProjections: preciseProjections
                ))
            }
            packed.cone.w = Float(tangent)
            let radius: Float = switch shadow.sampling {
            case .filtered: shadow.filterRadius
            case .basic: -1
            case .variance: -2
            }
            packed.shadow = SIMD4(Float(firstSlice), shadow.depthBias, shadow.normalBias, radius)
            packed.shadowReceiver.x = shadow.shadowStrength
            return packed
        }

        private mutating func nextSlice(for id: UUID, spot: Bool, maps: PunctualShadowMaps) -> Int? {
            if let assignments { return assignments[id] }
            if spot {
                guard spots < maps.spotCapacity else { return nil }
                defer { spots += 1 }
                return spots
            }
            guard points < maps.pointCapacity else { return nil }
            defer { points += 1 }
            return maps.spotCapacity + 6 * points
        }
    }

    private static func bindLightProbe(
        _ coefficients: [SIMD4<Float>], grids: LightProbeGridLighting? = nil, layers: SceneLayers = .default,
        into encoder: any MTLRenderCommandEncoder
    ) {
        var packed = coefficients
        if let grids {
            packed[4].w = Float(layers.rawValue & 0xffff)
            packed[5].w = Float(layers.rawValue >> 16)
            encoder.setFragmentBuffer(grids.buffer, offset: 0, index: 5)
        } else {
            var empty = SIMD4<Float>.zero
            encoder.setFragmentBytes(&empty, length: MemoryLayout<SIMD4<Float>>.stride, index: 5)
        }
        packed.withUnsafeBytes {
            encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: 26)
        }
    }

    private static let unusedRelativeLightPositions = Array(repeating: SIMD4<Float>.zero, count: SceneGraph.maximumPunctualLightCount)

    private static func bindPunctualLights(
        _ lights: [PackedPunctualLight], shadowTexture: any MTLTexture, shadowViews: [simd_float4x4],
        shadowMoments: (any MTLTexture)? = nil, relativeLightPositions: [SIMD4<Float>] = [],
        tables: LightTextureSet, rectangleLTC: (any MTLTexture)? = nil, into encoder: any MTLRenderCommandEncoder
    ) {
        var count = UInt32(lights.count)
        if lights.isEmpty {
            var unused = PackedPunctualLight()
            encoder.setFragmentBytes(&unused, length: MemoryLayout<PackedPunctualLight>.stride, index: 11)
        } else {
            lights.withUnsafeBytes { bytes in
                encoder.setFragmentBytes(bytes.baseAddress!, length: bytes.count, index: 11)
            }
        }
        encoder.setFragmentBytes(&count, length: MemoryLayout<UInt32>.size, index: 12)
        let positions = relativeLightPositions.isEmpty ? unusedRelativeLightPositions : relativeLightPositions
        positions.withUnsafeBytes { bytes in
            encoder.setFragmentBytes(bytes.baseAddress!, length: bytes.count, index: 28)
        }
        // Fragment buffer 20 and texture 24 belong to punctual shadows.
        if shadowViews.isEmpty {
            var unused = matrix_identity_float4x4
            encoder.setFragmentBytes(&unused, length: MemoryLayout<simd_float4x4>.stride, index: 20)
        } else {
            shadowViews.withUnsafeBytes { bytes in
                encoder.setFragmentBytes(bytes.baseAddress!, length: bytes.count, index: 20)
            }
        }
        encoder.setFragmentTexture(shadowTexture, index: 24)
        // Fragment textures 25 and 26 hold IES profile tables and spot cookies.
        encoder.setFragmentTexture(tables.profiles, index: 25)
        encoder.setFragmentTexture(tables.cookies, index: 26)
        // The profile array is a valid unused binding when no spot samples moments.
        encoder.setFragmentTexture(shadowMoments ?? tables.profiles, index: 29)
        encoder.setFragmentTexture(rectangleLTC ?? tables.profiles, index: 30)
    }

    private struct PackedSkinInfluence {
        let joints: SIMD4<UInt32>
        let weights: SIMD4<Float>
    }

    enum PreparedPalette {
        case inline(JointPalette)
        case buffer(any MTLBuffer)

        func bind(into encoder: any MTLRenderCommandEncoder, index: Int = 5) {
            switch self {
            case .inline(let palette):
                palette.matrices.withUnsafeBytes { bytes in
                    encoder.setVertexBytes(bytes.baseAddress!, length: bytes.count, index: index)
                }
            case .buffer(let buffer):
                encoder.setVertexBuffer(buffer, offset: 0, index: index)
            }
        }

        func retainBufferIfNeeded(on commandBuffer: any MTLCommandBuffer) {
            guard !commandBuffer.retainedReferences, case .buffer(let buffer) = self else { return }
            let retained = RetainedPaletteBuffer(buffer: buffer)
            commandBuffer.addCompletedHandler { _ in withExtendedLifetime(retained) {} }
        }
    }

    // The buffer is immutable after creation. The completion handler only owns its lifetime.
    private struct RetainedPaletteBuffer: @unchecked Sendable {
        let buffer: any MTLBuffer
    }

    private struct RetainedIndirectDraw: @unchecked Sendable {
        let renderer: MeshRenderer
        let source: any MTLBuffer
        let validated: any MTLBuffer
    }

    private struct RetainedPrimitiveIndirect: @unchecked Sendable {
        let source: any MTLBuffer
        let validated: any MTLBuffer
    }

    private struct RetainedScreenInputs: @unchecked Sendable {
        let value: PBRScreenInputs
    }

    private struct RetainedOverrideRenderers: @unchecked Sendable {
        let renderers: [MeshRenderer]
    }

    private struct PreparedDraw {
        var clip: simd_float4x4
        var normal: simd_float4x4
        var parameters: SIMD4<Float>
        var world: simd_float4x4
        var relativeDirectional: DirectionalShadowMap.RelativeDraw?
        var relativeLightModel: simd_float4x4?
        var shadingCamera: SIMD4<Float>
        let frontFacing: MTLWinding
        let palette: PreparedPalette?
        let morph: MorphDraw?
        let instances: InstanceSet?
        /// The instance count when the draw was prepared; one without instances.
        let instanceCount: Int
        let instanceRevision: UInt64?
        let indexRange: Range<Int>
        /// Assigned once the whole graph has been validated.
        var instanceBuffer: (any MTLBuffer)?
        var curveVertices: (any MTLBuffer)?
        var deformedVertices: (any MTLBuffer)?
        var dynamicRevision: DynamicGeometryBuffers.Revision?
        var dynamicBuffers: DynamicGeometryBuffers.Buffers?
        var indirectSource: MeshIndirectDraws?
        var indirect: MeshIndirectDraws.Validated?
        var batch: BatchedMesh.Packet?
        /// The node asked to stay lit. Directional strength is cleared and punctual maps are unbound for this draw.
        var suppressesShadows = false
        var drawCallCount: Int { indirect?.count ?? batch?.runs.count ?? 1 }
    }

    private func prepare(
        world: simd_float4x4, clipFromWorld: simd_float4x4,
        shadingCamera: SIMD4<Float>, parameters: SIMD4<Float>, jointPalette: JointPalette?,
        morphWeights: [Float]?, instances: InstanceSet?, precise: HighPrecisionModelView.DrawMatrices? = nil
    ) throws -> PreparedDraw {
        let determinant = simd_determinant(world)
        guard determinant.isFinite, determinant != 0 else { throw Failure.transform }
        var normal = simd_transpose(simd_inverse(world))
        if let precise {
            for index in 0..<3 { normal[index] = SIMD4(precise.normal[index], normal[index].w) }
        }
        let clip = precise?.clip ?? (clipFromWorld * world)
        guard Self.isFinite(clip), Self.isFinite(normal), Self.isFinite(parameters) else {
            throw Failure.transform
        }
        try validateInstances(instances)
        let weights = try validatedMorphWeights(morphWeights)
        let palette: PreparedPalette?
        if let jointPalette = try validatedPalette(jointPalette) {
            let byteCount = jointPalette.matrices.count * MemoryLayout<simd_float4x4>.stride
            palette = byteCount < 4_096 ? .inline(jointPalette)
                : .buffer(try Self.makeBuffer(jointPalette.matrices, on: device, label: "joint palette"))
        } else {
            palette = nil
        }
        // One draw has one winding, so a mirrored instance set flips it for every instance.
        let mirrored = (determinant < 0) != (instances?.isMirrored == true)
        return PreparedDraw(
            clip: clip, normal: normal, parameters: parameters, world: world,
            shadingCamera: shadingCamera,
            frontFacing: mirrored ? .clockwise : .counterClockwise, palette: palette,
            morph: morph?.draw(weights: weights), instances: instances, instanceCount: instances?.count ?? 1,
            instanceRevision: instances?.revision,
            indexRange: drawRange.resolve(indexCount: sourceGeometry.indices.count),
            dynamicRevision: geometryRevision.map {
                .init(
                    number: $0, topologyRevision: motionTopologyRevision, geometry: sourceGeometry, bounds: sourceBounds,
                    patchBase: geometryPatches.isEmpty ? nil : patchBaseRevision, vertexPatches: geometryPatches
                )
            }
        )
    }

    private func validateInstances(_ instances: InstanceSet?) throws {
        guard let instances else { return }
        guard supportsInstancing else { throw Failure.instancingNotPrepared }
        guard !requiresTransmissionBackdrop else { throw Failure.instancedTransmission }
        guard instances.device === device else { throw Failure.wrongDevice }
    }

    private static func validateIndirectTargets(
        _ draws: [UUID: MeshIndirectDraws], nodes: [ResolvedSceneNode], commandBuffer: any MTLCommandBuffer
    ) throws {
        guard !draws.isEmpty else { return }
        let ids = Set(nodes.map(\.id))
        if let missing = draws.keys.filter({ !ids.contains($0) }).min(by: { $0.uuidString < $1.uuidString }) {
            throw MeshIndirectDraws.Failure.unknownNode(missing)
        }
        var recordCount = 0
        for node in nodes {
            guard let source = draws[node.id] else { continue }
            guard let renderer = node.mesh, node.levelOfDetail == nil, node.batch == nil,
                  renderer.dynamicGeometry == nil else {
                throw MeshIndirectDraws.Failure.incompatibleNode(node.id)
            }
            guard renderer.device === commandBuffer.device else { throw Failure.wrongDevice }
            guard source.byteOffsets.count <= MeshIndirectDraws.maximumSceneCount - recordCount else {
                throw MeshIndirectDraws.Failure.sceneBudget
            }
            guard renderer.drawRange == .all else { throw Failure.drawRangeWithIndirectArguments }
            guard UInt32(exactly: renderer.sourceGeometry.indices.count) != nil,
                  UInt32(exactly: renderer.sourceGeometry.vertices.count) != nil else {
                throw MeshIndirectDraws.Failure.indexLimit
            }
            try renderer.validateInstances(node.instances)
            try source.validate(commandBuffer: commandBuffer)
            recordCount += source.byteOffsets.count
        }
    }

    /// Checks a draw's morph weights. Nil means the base mesh.
    private func validatedMorphWeights(_ weights: [Float]?) throws -> [Float]? {
        guard let morphTargets else {
            if weights != nil { throw Failure.unexpectedMorphWeights }
            return nil
        }
        guard let weights else { return nil }
        guard weights.count == morphTargets.targets.count else {
            throw MeshMorphTargets.ValidationError.weightCount(expected: morphTargets.targets.count, actual: weights.count)
        }
        guard weights.allSatisfy(\.isFinite) else { throw MeshMorphTargets.ValidationError.nonfiniteWeight }
        return weights
    }

    /// The world box that culls a node's draw: the node's culling bounds, or the
    /// mesh's bounds (widened by morph offsets and enclosing every instance). Nil
    /// when the node never culls: skinned or graph-deformed meshes without node
    /// bounds, or bounds outside the Float range.
    func cullingBounds(of node: ResolvedSceneNode, revisionBounds: AxisAlignedBox? = nil) throws -> AxisAlignedBox? {
        let local = try localCullingBounds(of: node, revisionBounds: revisionBounds)
        return try? local?.transformed(by: node.worldTransform)
    }

    private func localCullingBounds(of node: ResolvedSceneNode, revisionBounds: AxisAlignedBox?) throws -> AxisAlignedBox? {
        var local = node.cullingBounds
        if local == nil, skin == nil, deformation == nil {
            let weights = try validatedMorphWeights(node.morphWeights)
            local = revisionBounds ?? morph.map { $0.bounds(base: bounds, weights: weights) } ?? curveFlow?.cullingBounds ?? bounds
            if let mesh = local, let instances = node.instances {
                local = try? instances.bounds(enclosing: mesh)
            }
        }
        return local
    }

    private func prepareDeformation(
        for commandBuffer: any MTLCommandBuffer, cached: inout [ObjectIdentifier: any MTLBuffer]
    ) throws -> (any MTLBuffer)? {
        guard let deformation else { return nil }
        let key = ObjectIdentifier(deformation)
        if let vertices = cached[key] { return vertices }
        let vertices = try deformation.acquireVertices(for: commandBuffer)
        cached[key] = vertices
        return vertices
    }

    private func acquireDrawInputs(
        _ draw: inout PreparedDraw, for commandBuffer: any MTLCommandBuffer, requiringColors: Bool = false
    ) throws {
        if let source = draw.indirectSource {
            guard dynamicGeometry == nil else { throw Failure.unsupportedDynamicGeometry }
            guard drawRange == .all else { throw Failure.drawRangeWithIndirectArguments }
            guard draw.instances?.revision == draw.instanceRevision else { throw MeshIndirectDraws.Failure.instancesChanged }
            let validator = try indirectValidation ?? MeshIndirectDraws.Validator(geometry: sourceGeometry, indices: indices)
            indirectValidation = validator
            let validated = try source.encodeValidation(
                into: commandBuffer, validator: validator, instanceCount: UInt32(draw.instanceCount)
            )
            draw.indirect = validated
            if !commandBuffer.retainedReferences {
                let retained = RetainedIndirectDraw(renderer: self, source: source.buffer, validated: validated.buffer)
                commandBuffer.addCompletedHandler { _ in withExtendedLifetime(retained) {} }
            }
        }
        if let dynamicGeometry, let revision = draw.dynamicRevision {
            draw.dynamicBuffers = try dynamicGeometry.acquire(revision, for: commandBuffer, requiringColors: requiringColors)
            lastVertexUploadWasPartial = dynamicGeometry.lastUploadWasPartial
        }
        draw.curveVertices = try curveFlow?.acquireVertices(for: commandBuffer)
        draw.instanceBuffer = try draw.instances?.acquireBuffer(for: commandBuffer)
    }

    private func vertexBuffer(for draw: PreparedDraw) -> any MTLBuffer {
        draw.dynamicBuffers?.vertices ?? draw.curveVertices ?? draw.deformedVertices ?? vertices
    }

    /// Binds the vertex inputs that every pass shares: vertices, skin, instances and morph targets.
    private func bindVertexInputs(_ draw: PreparedDraw, into encoder: any MTLRenderCommandEncoder) {
        encoder.setVertexBuffer(vertexBuffer(for: draw), offset: 0, index: 0)
        if let vertexColors = draw.dynamicBuffers?.colors ?? vertexColors {
            encoder.setVertexBuffer(vertexColors, offset: 0, index: 10)
        }
        bindSkin(draw.palette, into: encoder)
        if supportsInstancing {
            if let buffer = draw.batch?.buffer ?? draw.instanceBuffer {
                encoder.setVertexBuffer(buffer, offset: 0, index: 6)
            } else {
                var identity = InstanceSet.PackedInstance.identity
                encoder.setVertexBytes(&identity, length: MemoryLayout<InstanceSet.PackedInstance>.stride, index: 6)
            }
        }
        if let morph, let state = draw.morph {
            encoder.setVertexBuffer(morph.offsets, offset: 0, index: 7)
            state.bind(into: encoder)
        }
    }

    /// Encodes one caster into a directional or punctual depth pass. The shadow
    /// map's pipelines serve plain and skinned meshes; specialized vertex stages use this mesh's own.
    private func encodeShadowCaster(
        _ draw: PreparedDraw, clip: simd_float4x4, into encoder: any MTLRenderCommandEncoder,
        coverageView: GraphViewContext? = nil,
        mapPipeline: (_ skinned: Bool, _ masked: Bool) -> Void
    ) {
        var clip = clip
        encoder.setFrontFacing(draw.frontFacing)
        if let graphShadow, let coverageView, let physicallyBased {
            encoder.setRenderPipelineState(graphShadow.pipeline)
            encodeGraphShadowInputs(physicallyBased, draw: draw, view: coverageView, into: encoder)
        } else {
            let cutoff: Float?
            if case .mask(let value)? = physicallyBased?.current.alphaMode { cutoff = value } else { cutoff = nil }
            if let shadowPipelines {
                encoder.setRenderPipelineState(cutoff != nil ? shadowPipelines.masked ?? shadowPipelines.opaque : shadowPipelines.opaque)
            } else {
                mapPipeline(draw.palette != nil, cutoff != nil)
            }
            if let cutoff {
                pbrBindings?.encodeShadowMask(tintAlpha: draw.parameters.w, cutoff: cutoff, into: encoder)
            }
        }
        bindVertexInputs(draw, into: encoder)
        encoder.setVertexBytes(&clip, length: MemoryLayout<simd_float4x4>.size, index: 1)
        drawIndices(draw, into: encoder)
    }

    private func encodeGraphShadowInputs(
        _ material: PhysicallyBasedMaterial, draw: PreparedDraw, view: GraphViewContext, into encoder: any MTLRenderCommandEncoder
    ) {
        var normal = draw.normal
        var world = draw.world
        var parameters = draw.parameters
        encoder.setVertexBytes(&normal, length: MemoryLayout<simd_float4x4>.size, index: 2)
        encoder.setVertexBytes(&world, length: MemoryLayout<simd_float4x4>.size, index: 3)
        encoder.setFragmentBytes(&parameters, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
        encodePBRInputs(material, draw: draw, into: encoder, sceneEnvironment: .init(), graphView: view)
    }

    private func transmittedShadowCaster(
        _ draw: PreparedDraw, light: SIMD4<Float>, node: UUID, batched: Bool
    ) throws -> DirectionalTransmissionPass.Caster? {
        if let graph = physicallyBased?.surfaceGraph, graph.hasCastShadowOutput {
            // The cast outputs were validated when the graph was built. Their coverage
            // outputs are shared with the ordinary shadow passes and reject camera inputs the same way.
            if let input = graph.shadowCoverageCameraNode { throw Failure.graphShadowCameraInput(node, input) }
            guard let graphTransmittedShadow else { throw Failure.transmittedShadowCaster(node) }
            guard !batched, draw.indirectSource == nil else { throw Failure.graphCastShadowGeometry(node) }
            return .graph(graphTransmittedShadow)
        }
        guard let material = physicallyBased?.current, let optical = material.transmission, let prepared = pbrBindings,
              material.surfaceGraph == nil, shadowPipelines == nil, draw.indirectSource == nil else {
            throw Failure.transmittedShadowCaster(node)
        }
        let tint = simd_clamp(draw.parameters * material.baseColorFactor, .zero, SIMD4(repeating: 1))
        let opticalDepth = Self.transmittedShadowOpticalDepth(optical, normal: draw.normal, light: light)
        if material.baseColorTexture != nil || material.alphaTexture != nil
            || material.sourceInputs.metallicRoughnessTexture != nil || optical.factorTexture != nil || optical.thicknessTexture != nil {
            let inputs = prepared.transmissionShadow(
                material: material, optical: optical, tint: tint, opticalDepth: Float(min(opticalDepth, 1e20))
            )
            return .textured(.init(material: prepared, inputs: inputs))
        }

        let coverage: Float
        switch material.alphaMode {
        case .opaque: coverage = 1
        case .mask(let cutoff): coverage = tint.w >= cutoff ? 1 : 0
        case .materialAlpha, .blend: coverage = tint.w
        }
        guard coverage > 0 else { return nil }

        var filter = SIMD3(tint.x, tint.y, tint.z) * (optical.factor * (1 - material.settings.y))
        if opticalDepth > 0 {
            for channel in 0..<3 {
                filter[channel] *= Float(pow(Double(optical.attenuationColor[channel]), opticalDepth))
            }
        }
        let full = SIMD3<Float>(repeating: 1)
        let transmitted = full - coverage * (full - filter)
        return transmitted == full ? nil : .uniform(transmitted)
    }

    private static func transmittedShadowOpticalDepth(
        _ optical: PhysicallyBasedMaterial.Transmission, normal: simd_float4x4, light: SIMD4<Float>
    ) -> Double {
        guard let distance = optical.attenuationDistance, optical.thickness > 0 else { return 0 }
        // Convert mesh-space thickness to a world-space straight light path
        // using the linear world-to-model transform (normal transposed).
        let local = simd_transpose(normal) * SIMD4(light.x, light.y, light.z, 0)
        let rate = simd_length(SIMD3<Double>(Double(local.x), Double(local.y), Double(local.z)))
        guard rate > 0, rate.isFinite else { return 0 }
        return Double(optical.thickness) / (rate * Double(distance))
    }

    private func encodeTransmittedShadow(
        _ draw: PreparedDraw, clip: simd_float4x4, optics: DirectionalTransmissionPass.Caster,
        pass: DirectionalTransmissionPass, view: GraphViewContext, into encoder: any MTLRenderCommandEncoder
    ) {
        var clip = clip
        let skinned = draw.palette != nil
        switch optics {
        case .uniform(var tint):
            pass.setPipeline(on: encoder, skinned: skinned, textured: false)
            encoder.setFragmentBytes(&tint, length: MemoryLayout<SIMD3<Float>>.stride, index: 0)
        case .textured(let caster):
            pass.setPipeline(on: encoder, skinned: skinned, textured: true)
            caster.material.encodeTransmissionShadow(caster.inputs, into: encoder)
        case .graph(let caster):
            encoder.setRenderPipelineState(caster.pipeline)
            encodeGraphShadowInputs(caster.material, draw: draw, view: view, into: encoder)
        }
        // A graph caster draws both faces like its ordinary shadow depth pass, so enabling
        // colored layers never changes which faces cast. Stock glass keeps its material's culling.
        if case .graph = optics {
            encoder.setCullMode(.none)
        } else {
            encoder.setCullMode(renderState.cullMode)
        }
        encoder.setFrontFacing(draw.frontFacing)
        bindVertexInputs(draw, into: encoder)
        encoder.setVertexBytes(&clip, length: MemoryLayout<simd_float4x4>.size, index: 1)
        drawIndices(draw, into: encoder)
    }

    private func validateTextureHelper(color: any MTLTexture, resolve: (any MTLTexture)?) throws {
        guard textureHelperOptions != nil, let colorTexture else { return }
        guard !TextureDebugPass.aliases(colorTexture, color),
              resolve.map({ !TextureDebugPass.aliases(colorTexture, $0) }) ?? true else {
            throw TextureDebugPass.Failure.destination
        }
    }

    private func encode(
        _ prepared: PreparedDraw,
        into encoder: any MTLRenderCommandEncoder,
        shadowMap: DirectionalShadowMap?,
        transmission: TransmissionFrame? = nil,
        sceneEnvironment: RasterSceneEnvironment = .init(),
        graphView: GraphViewContext,
        viewBasis: ViewBasis,
        screenInputs: PBRScreenInputs? = nil,
        environmentPipeline: (any MTLRenderPipelineState)? = nil,
        viewportTextures: SceneViewportSnapshot.Prepared? = nil
    ) {
        var draw = prepared
        encoder.setRenderPipelineState(environmentPipeline ?? pipeline)
        encoder.setDepthStencilState(depthState)
        if let stencil = renderState.stencil { encoder.setStencilReferenceValue(UInt32(stencil.reference)) }
        encoder.setFrontFacing(draw.frontFacing)
        encoder.setCullMode(renderState.cullMode)
        setDepthBias(on: encoder)
        if case .custom(let function) = renderState.blending {
            let constant = function.constantColor
            encoder.setBlendColor(red: constant.x, green: constant.y, blue: constant.z, alpha: constant.w)
        }
        bindVertexInputs(draw, into: encoder)
        encoder.setVertexBytes(&draw.clip, length: MemoryLayout<simd_float4x4>.size, index: 1)
        encoder.setVertexBytes(&draw.normal, length: MemoryLayout<simd_float4x4>.size, index: 2)
        encoder.setVertexBytes(&draw.world, length: MemoryLayout<simd_float4x4>.size, index: 3)
        encoder.setFragmentBytes(&draw.parameters, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
        rawProgram?.bind(into: encoder)
        if let colorTexture {
            if let indirectColorArguments {
                indirectColorArguments.encode(into: encoder)
            } else {
                encoder.setFragmentTexture(colorTexture, index: 0)
            }
        }
        if var options = textureHelperOptions {
            encoder.setFragmentBytes(&options, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 1)
        }
        if physicallyBased != nil || classic != nil {
            var relative = draw.relativeLightModel ?? simd_float4x4()
            encoder.setFragmentBytes(&relative, length: MemoryLayout<simd_float4x4>.stride, index: 27)
        }
        if let physicallyBased {
            encodePBRInputs(
                physicallyBased, draw: draw, into: encoder, sceneEnvironment: sceneEnvironment, graphView: graphView,
                viewportTextures: viewportTextures
            )
            var light = receivesShadows ? shadowMap?.lightParameters ?? physicallyBased.lightParameters
                : physicallyBased.lightParameters
            encoder.setFragmentBytes(&light, length: MemoryLayout<SIMD4<Float>>.size, index: 1)
            encoder.setFragmentBytes(&draw.shadingCamera, length: MemoryLayout<SIMD4<Float>>.size, index: 3)
            // Blended surfaces lie in front of the geometry measured by the prepass.
            let screen = isBlended ? nil : screenInputs?.binding(for: light)
            (screen ?? screenInputsFallback)?.bind(into: encoder)
        }
        if let classic {
            classic.prepared.encode(into: encoder)
            var light = receivesShadows ? shadowMap?.lightParameters ?? .zero : .zero
            encoder.setFragmentBytes(&light, length: MemoryLayout<SIMD4<Float>>.size, index: 2)
            encoder.setFragmentBytes(&draw.shadingCamera, length: MemoryLayout<SIMD4<Float>>.size, index: 3)
            var basis = viewBasis
            encoder.setFragmentBytes(&basis, length: MemoryLayout<ViewBasis>.stride, index: 6)
            if !receivesShadows || shadowMap == nil {
                DirectionalShadowMap.unusedShaderUniforms.withUnsafeBytes { bytes in
                    encoder.setFragmentBytes(bytes.baseAddress!, length: bytes.count, index: 4)
                }
                encoder.setFragmentTexture(classic.prepared.shadowFallback, index: 1)
            }
            if classic.material.surfaceGraph != nil {
                classic.prepared.encodeNodeGraph(into: encoder)
                var view = graphView.packed(model: draw.world, normal: draw.normal)
                encoder.setFragmentBytes(&view, length: MemoryLayout<PackedGraphView>.stride, index: 24)
            }
        }
        if requiresTransmissionBackdrop, var transmission {
            encoder.setFragmentTexture(transmission.backdrop, index: 13)
            encoder.setFragmentBytes(&transmission.clipFromWorld, length: MemoryLayout<simd_float4x4>.size, index: 16)
            encoder.setFragmentBytes(&draw.normal, length: MemoryLayout<simd_float4x4>.size, index: 17)
            encoder.setFragmentBytes(&transmission.viewportUV, length: MemoryLayout<SIMD4<Float>>.size, index: 18)
        }
        if receivesShadows, let shadowMap {
            // The FrostfieldDirectionalShadow record in FrostfieldShadowSampling.h.
            // biasUp.w, slot 25, is the shadow strength. A node that does not receive shadows forces it to zero.
            var uniforms = draw.relativeDirectional.map { shadowMap.relativeShaderUniforms(for: $0) }
                ?? shadowMap.shaderUniforms
            if draw.suppressesShadows, uniforms.count > 25 { uniforms[25].w = 0 }
            uniforms.withUnsafeBytes { bytes in
                encoder.setFragmentBytes(bytes.baseAddress!, length: bytes.count, index: physicallyBased != nil || classic != nil ? 4 : 1)
            }
            if physicallyBased == nil {
                var light = shadowMap.lightParameters
                encoder.setFragmentBytes(&light, length: MemoryLayout<SIMD4<Float>>.size, index: 2)
            }
            encoder.setFragmentTexture(shadowMap.texture, index: 1)
            encoder.setFragmentTexture(shadowMap.shadowMoments, index: 28)
            if physicallyBased != nil {
                encoder.setFragmentTexture(shadowMap.transmittedShadows?.layers ?? shadowMap.shadowMoments, index: 27)
            }
        }
        drawIndices(draw, into: encoder)
    }

    /// World-space right, up and backward view axes, matching `FrostfieldViewBasis` in
    /// ClassicMaterials.metal. Normal and matcap materials measure normals in it.
    struct ViewBasis {
        var right: SIMD4<Float>
        var up: SIMD4<Float>
        var back: SIMD4<Float>
    }

    static func viewBasis(cameraTransform: SceneTransform) -> ViewBasis {
        let rotation = simd_normalize(cameraTransform.rotation)
        return ViewBasis(
            right: SIMD4(rotation.act(SIMD3(1, 0, 0)), 0), up: SIMD4(rotation.act(SIMD3(0, 1, 0)), 0),
            back: SIMD4(rotation.act(SIMD3(0, 0, 1)), 0)
        )
    }

    /// The view axes of a caller-supplied world-to-clip matrix. Perspective views take
    /// the backward axis from the w row, orthographic views from the direction to the
    /// camera or the depth row, and the right axis from the x row.
    static func viewBasis(
        clipFromWorld m: simd_float4x4, directionToCamera: SIMD3<Float>?, depthMode: MeshDepthMode
    ) -> ViewBasis {
        func row(_ index: Int) -> SIMD3<Double> {
            SIMD3(Double(m.columns.0[index]), Double(m.columns.1[index]), Double(m.columns.2[index]))
        }
        func unit(_ value: SIMD3<Double>, _ fallback: SIMD3<Double>) -> SIMD3<Double> {
            let length = simd_length(value)
            return length > 1e-12 && length.isFinite ? value / length : fallback
        }
        var back: SIMD3<Double>
        if let directionToCamera {
            back = unit(SIMD3<Double>(directionToCamera), SIMD3(0, 0, 1))
        } else if simd_length(row(3)) > 1e-12 {
            back = unit(-row(3), SIMD3(0, 0, 1))
        } else {
            back = unit(depthMode == .reversed ? row(2) : -row(2), SIMD3(0, 0, 1))
        }
        let x = row(0)
        var right = unit(x - back * simd_dot(x, back), SIMD3(1, 0, 0))
        if abs(simd_dot(right, back)) > 0.5 { right = unit(simd_cross(SIMD3(0, 1, 0), back), SIMD3(1, 0, 0)) }
        let up = simd_cross(back, right)
        back = simd_cross(right, up)
        return ViewBasis(
            right: SIMD4(SIMD3<Float>(right), 0), up: SIMD4(SIMD3<Float>(up), 0), back: SIMD4(SIMD3<Float>(back), 0)
        )
    }

    /// Draws one mesh into the surface attribute prepass with its material's normal,
    /// roughness and specular inputs.
    private func encodeSurfaceAttributes(
        _ prepared: PreparedDraw, into encoder: any MTLRenderCommandEncoder, sceneEnvironment: RasterSceneEnvironment,
        graphView: GraphViewContext, environmentPipeline: (any MTLRenderPipelineState)?,
        viewportTextures: SceneViewportSnapshot.Prepared?
    ) throws {
        var draw = prepared
        encoder.setRenderPipelineState(try environmentPipeline ?? surfaceAttributePipeline())
        encoder.setDepthStencilState(depthState)
        encoder.setFrontFacing(draw.frontFacing)
        encoder.setCullMode(renderState.cullMode)
        setDepthBias(on: encoder)
        bindVertexInputs(draw, into: encoder)
        encoder.setVertexBytes(&draw.clip, length: MemoryLayout<simd_float4x4>.size, index: 1)
        encoder.setVertexBytes(&draw.normal, length: MemoryLayout<simd_float4x4>.size, index: 2)
        encoder.setVertexBytes(&draw.world, length: MemoryLayout<simd_float4x4>.size, index: 3)
        if let physicallyBased {
            encodePBRInputs(
                physicallyBased, draw: draw, into: encoder, sceneEnvironment: sceneEnvironment, graphView: graphView,
                viewportTextures: viewportTextures
            )
        }
        if let classic {
            classic.prepared.encode(into: encoder)
        }
        if physicallyBased != nil || classic != nil {
            encoder.setFragmentBytes(&draw.parameters, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
            encoder.setFragmentBytes(&draw.shadingCamera, length: MemoryLayout<SIMD4<Float>>.size, index: 3)
        }
        drawIndices(draw, into: encoder)
    }

    private func setDepthBias(on encoder: any MTLRenderCommandEncoder) {
        // Reversed depth stores nearer surfaces as larger values, so the bias flips to keep its direction.
        let biasSign: Float = depthMode == .reversed ? -1 : 1
        let offset = renderState.depthOffset
        encoder.setDepthBias(
            biasSign * (offset?.units ?? 0), slopeScale: biasSign * (offset?.factor ?? 0),
            clamp: biasSign * (offset?.clamp ?? 0)
        )
    }

    private func encodePBRInputs(
        _ material: PhysicallyBasedMaterial, draw: PreparedDraw, into encoder: any MTLRenderCommandEncoder,
        sceneEnvironment: RasterSceneEnvironment, graphView: GraphViewContext,
        viewportTextures: SceneViewportSnapshot.Prepared? = nil
    ) {
        pbrBindings?.encode(into: encoder, sceneEnvironment: sceneEnvironment)
        var settings = material.current.settings
        encoder.setFragmentBytes(&settings, length: MemoryLayout<SIMD4<Float>>.size, index: 2)
        if material.surfaceGraph != nil {
            var view = graphView.packed(model: draw.world, normal: draw.normal)
            encoder.setFragmentBytes(&view, length: MemoryLayout<PackedGraphView>.stride, index: 24)
            if !viewportInputs.isEmpty { viewportTextures?.bind(into: encoder) }
        }
    }

    private func validatedPalette(_ palette: JointPalette?) throws -> JointPalette? {
        guard let skin else {
            if palette != nil { throw SkinningError.unexpectedPalette }
            return nil
        }
        guard let palette else { throw SkinningError.missingPalette }
        guard palette.matrices.count == skin.jointCount else { throw SkinningError.paletteCount }
        return palette
    }

    private func bindSkin(_ palette: PreparedPalette?, into encoder: any MTLRenderCommandEncoder) {
        guard let palette, let skinInfluences else { return }
        encoder.setVertexBuffer(skinInfluences, offset: 0, index: 4)
        palette.bind(into: encoder)
    }

    /// One active index range and, when applicable, one revision of graph-displaced geometry.
    private var pickingBVH = MeshBVHCache()

    private let posedPickingBVH: PosedMeshBVHCache?

    /// Prepares the active index range for subsequent scene picks and world rays.
    /// The hierarchy builds off the caller's actor. Use the renderer from one
    /// actor during preparation and when editing its geometry or draw range.
    /// Returns nil for an empty draw range. This prepares rest geometry for a
    /// morphable mesh and the current uniform and clock pose for a mesh with a
    /// graph position offset. Skinned and curve-flow meshes, and graph offsets
    /// that read mutable textures, use their pose-specific picking paths.
    ///
    /// Graph poses await GPU readback before building the hierarchy off-actor.
    /// Cancellation or a change to the captured range, geometry or pose prevents
    /// installation. A hierarchy already prepared by a scene pick is reused
    /// while those inputs match.
    nonisolated(nonsending) public func preparePickingAcceleration(
        progress: (@Sendable (MeshBVH.BuildProgress) -> Void)? = nil
    ) async throws -> MeshBVH? {
        try Task.checkCancellation()
        guard skin == nil, curveFlow == nil, deformation?.supportsPickingAcceleration != false else {
            throw Failure.pickingPreparation
        }
        let range = drawRange.resolve(indexCount: sourceGeometry.indices.count)
        guard !range.isEmpty else { return nil }
        let cache = pickingBVH
        let snapshot = deformation?.uniformSnapshot()
        if let prepared = cache.prepared(revision: snapshot?.revision) { return prepared }
        let geometry: MeshGeometry
        if let deformation, let snapshot {
            guard let queue = device.makeCommandQueue() else { throw Failure.resource("deformation queue") }
            geometry = try await deformation.readPickingGeometry(queue: queue, snapshot: snapshot)
            guard cache === pickingBVH, deformation.uniformRevision == snapshot.revision else {
                throw Failure.pickingGeometryChanged
            }
        } else {
            geometry = sourceGeometry
        }
        let hierarchy = try await MeshBVH.build(geometry: geometry.restrictingIndices(to: range), progress: progress)
        try Task.checkCancellation()
        guard cache === pickingBVH, deformation?.uniformRevision == snapshot?.revision else {
            throw Failure.pickingGeometryChanged
        }
        return cache.installIfAbsent(hierarchy, revision: snapshot?.revision)
    }

    private func pickingSurface(
        _ node: ResolvedSceneNode
    ) throws -> (geometry: MeshGeometry, bvh: MeshBVH?, triangleOffset: Int)? {
        let geometry = self.geometry
        let range = drawRange.resolve(indexCount: geometry.indices.count)
        guard !range.isEmpty else { return nil }
        let palette = try validatedPalette(node.jointPalette)
        let weights = try validatedMorphWeights(node.morphWeights)
        let activeWeights = weights?.contains(where: { $0 != 0 }) == true ? weights : nil
        let offset = range.lowerBound / 3
        if let curveFlow, range.count == geometry.indices.count {
            return (geometry, try curveFlow.pickingAcceleration(), offset)
        }
        if let deformation, deformation.supportsPickingAcceleration, palette == nil {
            let snapshot = deformation.uniformSnapshot()
            let bvh = try pickingBVH.bvh(
                for: deformedGeometry(queue: nil, picking: snapshot).restrictingIndices(to: range),
                revision: snapshot.revision
            )
            return (bvh.geometry, bvh, offset)
        }
        if curveFlow == nil, deformation == nil {
            if palette == nil, activeWeights == nil {
                let bvh = try pickingBVH.bvh(for: geometry.restrictingIndices(to: range))
                return (bvh.geometry, bvh, offset)
            }
            if let posedPickingBVH, geometryRevision == nil {
                let bvh = try posedPickingBVH.bvh(palette: palette, weights: activeWeights, range: range) {
                    try pickingGeometry(jointPalette: palette, morphWeights: activeWeights)
                }
                return (bvh.geometry, bvh, offset)
            }
        }
        let posed = try pickingGeometry(jointPalette: node.jointPalette, morphWeights: node.morphWeights)
        return (posed.restrictingIndices(to: range), nil, offset)
    }

    func selectionCenter(_ node: ResolvedSceneNode) throws -> SIMD3<Float> {
        let palette = try validatedPalette(node.jointPalette)
        let weights = try validatedMorphWeights(node.morphWeights)
        if deformation == nil, palette == nil, weights?.contains(where: { $0 != 0 }) != true {
            return bounds.center
        }
        return try pickingGeometry(jointPalette: node.jointPalette, morphWeights: node.morphWeights).bounds.center
    }

    /// The nearest hit on a node's mesh and, for an instanced node, the lowest
    /// instance index among equally near hits. Instances test the same mesh-space
    /// triangles through each instance's transform.
    fileprivate func pickingIntersection(
        ray: SceneRay, node: ResolvedSceneNode
    ) throws -> (hit: MeshRayHit, instance: Int?)? {
        guard let surface = try pickingSurface(node) else { return nil }
        func intersect(_ transform: simd_float4x4) throws -> MeshRayHit? {
            let hit: MeshRayHit?
            if let bvh = surface.bvh {
                hit = try bvh.intersection(ray: ray, localToWorld: transform, side: renderState.side)
            } else {
                hit = try surface.geometry.intersection(ray: ray, localToWorld: transform, side: renderState.side)
            }
            return hit?.offsetTriangleIndex(by: surface.triangleOffset)
        }
        guard let instances = node.instances else {
            return try intersect(node.worldTransform).map { ($0, nil) }
        }
        var nearest: (hit: MeshRayHit, instance: Int?)?
        for (index, transform) in instances.transforms.enumerated() {
            guard let hit = try intersect(node.worldTransform * transform) else { continue }
            if let nearest, hit.distance >= nearest.hit.distance { continue }
            nearest = (hit, index)
        }
        return nearest
    }

    fileprivate func appendPickingHits(ray: SceneRay, node: ResolvedSceneNode, to result: inout [SceneHit]) throws {
        guard let surface = try pickingSurface(node) else { return }
        func hits(_ transform: simd_float4x4, instanceIndex: Int?) throws -> [SceneHit] {
            let intersections = try surface.bvh?.intersections(
                ray: ray, localToWorld: transform, side: renderState.side
            ) ?? surface.geometry.intersections(ray: ray, localToWorld: transform, side: renderState.side)
            return intersections.map {
                SceneHit(nodeID: node.id, intersection: $0.offsetTriangleIndex(by: surface.triangleOffset), instanceIndex: instanceIndex)
            }
        }
        if let instances = node.instances {
            for (index, transform) in instances.transforms.enumerated() {
                result.append(contentsOf: try hits(node.worldTransform * transform, instanceIndex: index))
            }
        } else {
            result.append(contentsOf: try hits(node.worldTransform, instanceIndex: nil))
        }
    }

    /// The local geometry a draw rasterizes: deformed by a surface graph, or
    /// morphed with the CPU `MeshMorphTargets.apply`, then skinned.
    func pickingGeometry(jointPalette: JointPalette?, morphWeights: [Float]? = nil) throws -> MeshGeometry {
        let base = try morphedGeometry(weights: morphWeights)
        guard let palette = try validatedPalette(jointPalette), let skin else { return base }
        return try skin.deform(base, using: palette)
    }

    /// The source geometry morphed on the CPU with validated weights, or the
    /// deformed geometry for a mesh without morph targets. Path tracing uses it.
    func morphedGeometry(weights: [Float]?) throws -> MeshGeometry {
        let weights = try validatedMorphWeights(weights)
        guard let morphTargets, let weights else { return try deformedGeometry() }
        return try morphTargets.apply(weights: weights, to: geometry)
    }

    /// The object-space geometry that draws use. For a surface graph with a
    /// position offset, this runs the deformation with the current uniform values
    /// on the GPU, waits, and returns validated geometry with rebuilt normals.
    /// Picking and path-tracing preparation use it. Pass a queue to avoid creating one.
    public func deformedGeometry(queue: (any MTLCommandQueue)? = nil) throws -> MeshGeometry {
        try deformedGeometry(queue: queue, picking: nil)
    }

    private func deformedGeometry(
        queue: (any MTLCommandQueue)?, picking snapshot: MeshDeformation.UniformSnapshot?
    ) throws -> MeshGeometry {
        guard let deformation else { return geometry }
        if let queue, queue.device !== device { throw Failure.wrongDevice }
        guard let queue = queue ?? device.makeCommandQueue() else { throw Failure.resource("deformation queue") }
        return try deformation.deformedGeometry(queue: queue, picking: snapshot)
    }

    func prepareBatchBundle(_ runs: [BatchedMesh.Run]) throws -> BatchRenderBundle {
        try BatchRenderBundle(indices: indices, runs: runs)
    }

    private func submittedTriangleCount(_ draw: PreparedDraw) -> Int {
        guard draw.indirect == nil else { return 0 }
        if let batch = draw.batch {
            return batch.runs.reduce(0) { $0 + ($1.indices.count / 3) * $1.instanceCount }
        }
        return (draw.indexRange.count / 3) * draw.instanceCount
    }

    private func drawIndices(
        _ draw: PreparedDraw, into encoder: any MTLRenderCommandEncoder, useBatchBundle: Bool = true
    ) {
        if let indirect = draw.indirect {
            for index in 0..<indirect.count {
                encoder.drawIndexedPrimitives(
                    type: .triangle, indexType: .uint32, indexBuffer: indices, indexBufferOffset: 0,
                    indirectBuffer: indirect.buffer, indirectBufferOffset: index * MeshIndirectDraws.recordByteCount
                )
            }
            return
        }
        if let batch = draw.batch {
            if useBatchBundle, let bundle = batch.bundle {
                bundle.encode(into: encoder)
                return
            }
            for run in batch.runs {
                encoder.setFrontFacing(run.mirrored ? .clockwise : .counterClockwise)
                encoder.setVertexBufferOffset(run.firstInstance * MemoryLayout<InstanceSet.PackedInstance>.stride, index: 6)
                encoder.drawIndexedPrimitives(
                    type: .triangle, indexCount: run.indices.count, indexType: .uint32,
                    indexBuffer: indices, indexBufferOffset: run.indices.lowerBound * MemoryLayout<UInt32>.stride,
                    instanceCount: run.instanceCount
                )
            }
            return
        }
        guard !draw.indexRange.isEmpty else { return }
        encoder.drawIndexedPrimitives(
            type: .triangle, indexCount: draw.indexRange.count, indexType: .uint32,
            indexBuffer: draw.dynamicBuffers?.indices ?? indices,
            indexBufferOffset: draw.indexRange.lowerBound * MemoryLayout<UInt32>.stride,
            instanceCount: draw.instanceCount
        )
    }

    private struct ToonOutlinePipeline {
        let render: any MTLRenderPipelineState
        let depth: any MTLDepthStencilState
    }

    private func outlineEffectOpacity(parameters: SIMD4<Float>) -> Float? {
        guard renderState.depthTest, textureHelperOptions == nil else { return nil }
        switch material {
        case .classic(let material):
            switch material.model {
            case .depth, .distance, .shadow: return nil
            default: return material.opacity
            }
        case .physicallyBased(let material):
            return material.liveFactors?.current.baseColorFactor.w ?? material.baseColorFactor.w
        case .unlit, .textured, .shadowedLambert:
            return parameters.w.isFinite ? min(max(parameters.w, 0), 1) : 0
        case .lambert, .custom:
            return 1
        case .rawMetal:
            return nil
        }
    }

    private func toonOutlinePipeline() throws -> ToonOutlinePipeline {
        if let preparedToonOutline { return preparedToonOutline }
        let library: any MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw Failure.library(error.localizedDescription)
        }
        let specialization = VertexSpecialization(
            skinned: skin != nil, morphed: morphTargets != nil, instanced: supportsInstancing
        )
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Frostfield toon outline hull"
        descriptor.vertexFunction = try specialization.function("frostfield_toon_outline_vertex", in: library)
        descriptor.fragmentFunction = try Self.function("frostfield_toon_outline_fragment", in: library)
        descriptor.colorAttachments[0].pixelFormat = colorPixelFormat
        descriptor.depthAttachmentPixelFormat = Self.depthPixelFormat
        descriptor.rasterSampleCount = sampleCount
        let state = MeshRenderState(blending: .normal, depthCompare: .lessEqual, side: .back)
        state.configure(descriptor)
        guard let depth = device.makeDepthStencilState(descriptor: state.depthDescriptor(depthMode: depthMode)) else {
            throw Failure.pipeline("Toon outline depth state could not be created.")
        }
        let pipeline = ToonOutlinePipeline(render: try Self.makePipeline(device: device, descriptor: descriptor), depth: depth)
        preparedToonOutline = pipeline
        return pipeline
    }

    private func encodeToonOutline(
        _ draw: PreparedDraw, style: ToonOutlinePass.DrawStyle, pipeline: ToonOutlinePipeline,
        into encoder: any MTLRenderCommandEncoder
    ) {
        var clip = draw.clip
        var thickness = style.thickness
        var premultiplied = style.premultiplied
        encoder.setRenderPipelineState(pipeline.render)
        encoder.setDepthStencilState(pipeline.depth)
        encoder.setFrontFacing(draw.frontFacing)
        encoder.setCullMode(.front)
        encoder.setDepthBias(0, slopeScale: 0, clamp: 0)
        bindVertexInputs(draw, into: encoder)
        encoder.setVertexBytes(&clip, length: MemoryLayout<simd_float4x4>.size, index: 1)
        encoder.setVertexBytes(&thickness, length: MemoryLayout<Float>.size, index: 11)
        encoder.setFragmentBytes(&premultiplied, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
        // The outline pipeline reads the same batch runs with its own shader and cull state.
        drawIndices(draw, into: encoder, useBatchBundle: false)
    }

    private static func attachmentExtent(
        _ texture: any MTLTexture, _ attachment: MTLRenderPassAttachmentDescriptor
    ) -> SIMD2<Int>? {
        let level = attachment.level
        guard level >= 0, level < texture.mipmapLevelCount, attachment.slice >= 0, attachment.depthPlane >= 0 else {
            return nil
        }
        let slices: Int
        switch texture.textureType {
        case .type2D, .type2DMultisample, .type3D: slices = 1
        case .type2DArray: slices = texture.arrayLength
        case .typeCube: slices = 6
        case .typeCubeArray: slices = texture.arrayLength * 6
        default: return nil
        }
        let planes = texture.textureType == .type3D ? max(1, texture.depth >> level) : 1
        guard attachment.slice < slices, attachment.depthPlane < planes else { return nil }
        return SIMD2(max(1, texture.width >> level), max(1, texture.height >> level))
    }

    private static func usesStencil(_ pass: MTLRenderPassDescriptor) -> Bool {
        pass.stencilAttachment.texture != nil || pass.stencilAttachment.resolveTexture != nil
            || pass.depthAttachment.texture?.pixelFormat == stencilDepthPixelFormat
    }

    private static func validateStencilAttachments(
        in pass: MTLRenderPassDescriptor, color: any MTLTexture, depth: any MTLTexture
    ) throws {
        let colorAttachment = pass.colorAttachments[0]!
        let depthAttachment = pass.depthAttachment!
        let stencil = pass.stencilAttachment!
        let multisampled = color.sampleCount > 1
        let type: MTLTextureType = multisampled ? .type2DMultisample : .type2D

        func isRenderable(_ texture: any MTLTexture) -> Bool {
            texture.storageMode != .memoryless && !texture.isAliasable()
                && (texture.usage.isEmpty || texture.usage.contains(.renderTarget))
        }

        let attachments: [MTLRenderPassAttachmentDescriptor] = [colorAttachment, depthAttachment, stencil]
        guard depth.pixelFormat == stencilDepthPixelFormat, stencil.texture === depth,
              color.textureType == type, depth.textureType == type, isRenderable(color), isRenderable(depth),
              pass.rasterizationRateMap == nil, !TextureTransfer.sharesStorage(color, depth) else {
            throw Failure.stencilAttachments
        }
        let supportedDepthStores = attachments.dropFirst().allSatisfy { attachment in
            attachment.resolveTexture == nil && (attachment.storeAction == .store || attachment.storeAction == .dontCare)
        }
        guard supportedDepthStores, stencil.loadAction == .clear || stencil.loadAction == .load,
              stencil.loadAction != .clear || stencil.clearStencil <= UInt32(UInt8.max),
              (1..<8).allSatisfy({ pass.colorAttachments[$0].texture == nil && pass.colorAttachments[$0].resolveTexture == nil }),
              attachments.allSatisfy({ $0.level == 0 && $0.slice == 0 && $0.depthPlane == 0 }) else {
            throw Failure.stencilAttachments
        }

        switch colorAttachment.storeAction {
        case .store, .dontCare:
            guard colorAttachment.resolveTexture == nil else { throw Failure.stencilAttachments }
        case .multisampleResolve, .storeAndMultisampleResolve:
            guard multisampled, let resolve = colorAttachment.resolveTexture, resolve.device === color.device,
                  resolve.textureType == .type2D, resolve.sampleCount == 1, resolve.pixelFormat == color.pixelFormat,
                  resolve.width == color.width, resolve.height == color.height, isRenderable(resolve),
                  colorAttachment.resolveLevel == 0, colorAttachment.resolveSlice == 0, colorAttachment.resolveDepthPlane == 0,
                  !TextureTransfer.sharesStorage(color, resolve), !TextureTransfer.sharesStorage(depth, resolve) else {
                throw Failure.stencilAttachments
            }
        default:
            throw Failure.stencilAttachments
        }
    }

    private static func colorAttachments(
        in pass: MTLRenderPassDescriptor, device: any MTLDevice
    ) throws -> (first: any MTLTexture, formats: [MTLPixelFormat], size: SIMD2<Int>) {
        let colors = pass.colorAttachments
        let first = colors[0]!
        let depthAttachment = pass.depthAttachment!
        let usesStencil = Self.usesStencil(pass)
        guard let color = first.texture, let depth = depthAttachment.texture,
              color.device === device, depth.device === device,
              usesStencil || depth.pixelFormat == Self.depthPixelFormat,
              depth.sampleCount == color.sampleCount, pass.renderTargetArrayLength <= 1,
              let extent = attachmentExtent(color, first), attachmentExtent(depth, depthAttachment) == extent,
              pass.renderTargetWidth >= 0, pass.renderTargetWidth <= extent.x,
              pass.renderTargetHeight >= 0, pass.renderTargetHeight <= extent.y else {
            throw Failure.attachments
        }
        let size = SIMD2(
            pass.renderTargetWidth == 0 ? extent.x : pass.renderTargetWidth,
            pass.renderTargetHeight == 0 ? extent.y : pass.renderTargetHeight
        )
        if usesStencil {
            try validateStencilAttachments(in: pass, color: color, depth: depth)
            return (color, [color.pixelFormat], size)
        }
        guard let last = (1..<8).last(where: { colors[$0].texture != nil || colors[$0].resolveTexture != nil }) else {
            let ordinaryColor = color.textureType == .type2D || color.textureType == .type2DMultisample
            let ordinaryDepth = depth.textureType == .type2D || depth.textureType == .type2DMultisample
            if !ordinaryColor || !ordinaryDepth {
                guard color.sampleCount == 1, first.resolveTexture == nil, depthAttachment.resolveTexture == nil,
                      color.usage.isEmpty || color.usage.contains(.renderTarget),
                      depth.usage.isEmpty || depth.usage.contains(.renderTarget),
                      first.storeAction == .store || first.storeAction == .dontCare,
                      depthAttachment.storeAction == .store || depthAttachment.storeAction == .dontCare else {
                    throw Failure.attachments
                }
            }
            return (color, [color.pixelFormat], size)
        }
        guard color.sampleCount == 1, size == extent else { throw Failure.attachments }
        var outputs: [any MTLTexture] = []
        let attachments: [MTLRenderPassAttachmentDescriptor] =
            [pass.depthAttachment] + (0...last).map { colors[$0]! }
        for attachment in attachments {
            guard let texture = attachment.texture,
                  texture.device === device, texture.sampleCount == 1, attachmentExtent(texture, attachment) == extent,
                  texture.storageMode != .memoryless, !texture.isAliasable(),
                  texture.usage.isEmpty || texture.usage.contains(.renderTarget),
                  attachment.resolveTexture == nil,
                  attachment.storeAction == .store || attachment.storeAction == .dontCare,
                  !outputs.contains(where: { TextureTransfer.sharesStorage($0, texture) }) else {
                throw Failure.attachments
            }
            outputs.append(texture)
        }
        return (color, outputs.dropFirst().map(\.pixelFormat), size)
    }

    static func renderTargetSize(in pass: MTLRenderPassDescriptor, device: any MTLDevice) throws -> SIMD2<Int> {
        try colorAttachments(in: pass, device: device).size
    }

    static func validateScissor(_ scissor: MTLScissorRect?, size: SIMD2<Int>) throws {
        guard let scissor else { return }
        let width = size.x, height = size.y
        guard scissor.x >= 0, scissor.y >= 0, scissor.x <= width, scissor.y <= height,
              scissor.width >= 0, scissor.height >= 0,
              scissor.width <= width - scissor.x, scissor.height <= height - scissor.y else {
            throw Failure.scissor
        }
    }

    static func validateViewport(_ viewport: MTLViewport?, size: SIMD2<Int>) throws {
        guard let viewport else { return }
        guard viewport.originX >= 0, viewport.originY >= 0,
              viewport.width > 0, viewport.height > 0,
              viewport.originX + viewport.width <= Double(size.x),
              viewport.originY + viewport.height <= Double(size.y),
              viewport.znear >= 0, viewport.zfar <= 1, viewport.znear < viewport.zfar else {
            throw Failure.viewport
        }
    }

    static func explicitShadingCamera(
        clipFromWorld: simd_float4x4, position: SIMD3<Float>, direction: SIMD3<Float>?, highPrecision: Bool
    ) throws -> SIMD4<Float> {
        let camera = SIMD4(position, 1)
        guard isFinite(clipFromWorld), isFinite(camera) else { throw Failure.transform }
        if !highPrecision {
            let determinant = simd_determinant(clipFromWorld)
            guard determinant.isFinite, determinant != 0 else { throw Failure.transform }
        }
        guard let direction else { return camera }
        let length = simd_length(direction)
        guard length.isFinite, length.isNormal else { throw Failure.transform }
        return SIMD4(direction / length, 0)
    }

    private static func viewProjection(
        camera: PerspectiveCamera, cameraTransform: SceneTransform, size: SIMD2<Int>, depthMode: MeshDepthMode
    ) throws -> simd_float4x4 {
        let worldFromCamera = cameraTransform.matrix
        let determinant = simd_determinant(worldFromCamera)
        guard determinant.isFinite, determinant != 0 else { throw Failure.transform }
        let aspect = Float(size.x) / Float(size.y)
        let clipFromWorld = try camera.projection(aspectRatio: aspect, depthMode: depthMode) * simd_inverse(worldFromCamera)
        guard isFinite(clipFromWorld) else { throw Failure.transform }
        return clipFromWorld
    }

    private static func viewProjection(
        graph: SceneGraph, size: SIMD2<Int>, depthMode: MeshDepthMode
    ) throws -> simd_float4x4 {
        let worldFromCamera = graph.cameraTransform.matrix
        let determinant = simd_determinant(worldFromCamera)
        guard determinant.isFinite, determinant != 0 else { throw Failure.transform }
        let aspect = Float(size.x) / Float(size.y)
        let clipFromWorld = try graph.activeProjection(aspectRatio: aspect, depthMode: depthMode) * simd_inverse(worldFromCamera)
        guard isFinite(clipFromWorld) else { throw Failure.transform }
        return clipFromWorld
    }

    private static func isColorTexture(_ texture: any MTLTexture) -> Bool {
        guard texture.textureType == .type2D, !texture.isFramebufferOnly,
              texture.usage.isEmpty || texture.usage.contains(.shaderRead) else { return false }
        switch texture.pixelFormat {
        case .rgba8Unorm, .rgba8Unorm_srgb, .bgra8Unorm, .bgra8Unorm_srgb, .rgba16Float:
            return true
        case let format:
            return MaterialTextureFormat.compressedEncoding(format).isCompressed
        }
    }

    private static func isFinite(_ value: SIMD4<Float>) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite && value.w.isFinite
    }

    private static func isFinite(_ matrix: simd_float4x4) -> Bool {
        let (x, y, z, w) = matrix.columns
        return isFinite(x) && isFinite(y) && isFinite(z) && isFinite(w)
    }

    private static func function(_ name: String, in library: any MTLLibrary) throws -> any MTLFunction {
        guard let function = library.makeFunction(name: name) else { throw Failure.function(name) }
        return function
    }

    private static func surfaceLibrary(
        _ graph: SurfaceShaderGraph?, targets: SurfaceRenderTargets?, logarithmicDepth: Bool,
        viewportInputs: SurfaceViewportInputs, readsSoftParticles: Bool, device: any MTLDevice
    ) throws -> any MTLLibrary {
        guard let shadowURL = Bundle.module.url(forResource: "FrostfieldShadowSampling", withExtension: "h"),
              let screenURL = Bundle.module.url(forResource: "FrostfieldScreenSpace", withExtension: "h"),
              let probeURL = Bundle.module.url(forResource: "FrostfieldLightProbe", withExtension: "h"),
              let pbrURL = Bundle.module.url(forResource: "PhysicallyBasedShading", withExtension: "h") else {
            throw Failure.library("The package is missing its PBR shader sources.")
        }
        let shadow = try String(contentsOf: shadowURL, encoding: .utf8)
        let screenSpace = try String(contentsOf: screenURL, encoding: .utf8)
        let lightProbe = try String(contentsOf: probeURL, encoding: .utf8)
        let shading = try MaterialExtensionsSource.splice(into: String(contentsOf: pbrURL, encoding: .utf8))
        let graphSource: String
        if let graph {
            let function = try graph.metalFunction(named: "frostfield_generated_surface")
            // Only shadow passes evaluate this function, so it follows the shadow mask precedence.
            let coverage = graph.hasShadowCoverageOutputs
                ? try graph.metalShadowCoverageFunction(named: "frostfield_generated_coverage") : ""
            let castShadow = try graph.metalCastShadowFunction(named: "frostfield_generated_cast_shadow")
            let softParticleSource: String
            if readsSoftParticles {
                guard let url = Bundle.module.url(forResource: "FrostfieldSoftParticle", withExtension: "h") else {
                    throw Failure.resource("FrostfieldSoftParticle.h")
                }
                softParticleSource = try String(contentsOf: url, encoding: .utf8) + "\n" + rasterSoftParticleSource
            } else {
                softParticleSource = ""
            }
            graphSource = """
            \(SurfaceShaderGraph.supportSource)
            \(viewportInputs.contains(.color) ? "#define FROSTFIELD_GRAPH_VIEWPORT_COLOR 1" : "")
            \(viewportInputs.contains(.depth) ? "#define FROSTFIELD_GRAPH_VIEWPORT_DEPTH 1" : "")
            \(rasterGraphResources)
            \(softParticleSource)
            #define FROSTFIELD_GENERATED_SURFACE 1
            \(graph.readsDirectionalShadow ? "#define FROSTFIELD_GRAPH_DIRECTIONAL_SHADOW 1" : "")
            \(graph.hasShadowCoverageOutputs ? "#define FROSTFIELD_GRAPH_COVERAGE 1" : "")
            \(castShadow != nil ? "#define FROSTFIELD_GRAPH_CAST_SHADOW 1" : "")
            \(function)
            \(coverage)
            \(castShadow ?? "")
            """
        } else {
            graphSource = ""
        }
        let depthSource: String
        if logarithmicDepth {
            guard let url = Bundle.module.url(forResource: "FrostfieldLogDepth", withExtension: "h") else {
                throw Failure.resource("FrostfieldLogDepth.h")
            }
            depthSource = "#define FROSTFIELD_PBR_LOG_DEPTH 1\n" + (try String(contentsOf: url, encoding: .utf8))
        } else {
            depthSource = ""
        }
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        \(depthSource)
        \(shadow)
        \(screenSpace)
        \(lightProbe)
        \(graphSource)
        \(targets?.metalSource ?? "")
        \(shading)
        """
        do {
            return try ShaderLibraryCache.library(source: source, device: device)
        } catch {
            throw Failure.library(ShaderProgram.diagnose(error, source: source))
        }
    }

    /// Compiles `frostfield_mesh_classic` with the graph's base color wired in before lighting.
    /// The bundled classic shader stays unchanged for materials without a graph.
    private static func classicNodeLibrary(_ graph: SurfaceShaderGraph, device: any MTLDevice) throws -> any MTLLibrary {
        guard let shadowURL = Bundle.module.url(forResource: "FrostfieldShadowSampling", withExtension: "h"),
              let screenURL = Bundle.module.url(forResource: "FrostfieldScreenSpace", withExtension: "h"),
              let probeURL = Bundle.module.url(forResource: "FrostfieldLightProbe", withExtension: "h"),
              let pbrURL = Bundle.module.url(forResource: "PhysicallyBasedShading", withExtension: "h"),
              let classicURL = Bundle.module.url(forResource: "ClassicMaterialShading", withExtension: "h") else {
            throw Failure.library("The package is missing its classic shader sources.")
        }
        let function = try graph.metalFunction(named: "frostfield_generated_surface")
        let shading = try MaterialExtensionsSource.splice(into: String(contentsOf: pbrURL, encoding: .utf8))
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        #define FROSTFIELD_CLASSIC_NODE 1
        #define FROSTFIELD_PBR_HELPERS_ONLY 1
        \(SurfaceShaderGraph.supportSource)
        \(rasterGraphResources)
        \(function)
        \(try String(contentsOf: shadowURL, encoding: .utf8))
        \(try String(contentsOf: screenURL, encoding: .utf8))
        \(try String(contentsOf: probeURL, encoding: .utf8))
        \(shading)
        \(try String(contentsOf: classicURL, encoding: .utf8))
        """
        do {
            return try ShaderLibraryCache.library(source: source, device: device)
        } catch {
            throw Failure.library(ShaderProgram.diagnose(error, source: source))
        }
    }

    private static let rasterSoftParticleSource = """
    static inline float frostfield_graph_soft_particle(
        thread const FrostfieldSurfaceInput& input, float opacity, float distance, float contrast, float rawDepth
    ) {
        if (!(distance >= 0x1p-126f && contrast >= 0x1p-126f)
            || !frostfield_graph_finite(distance) || !frostfield_graph_finite(contrast)) return opacity;
        float3 range = input.depthRange;
        float depth = (rawDepth - range.x) / (range.y - range.x);
        float clear = range.z == 1.0f ? 0.0f : 1.0f;
        if (!(depth >= 0.0f && depth <= 1.0f) || depth == clear) return opacity;
        float4 p = float4(input.projectionMatrix[2].z, input.projectionMatrix[3].z,
                          input.projectionMatrix[2].w, input.projectionMatrix[3].w);
        float particleZ = frostfieldParticleFragmentViewZ(input.particleDepth, range.xy, p);
        float sceneZ = frostfieldParticleViewZ(depth, p);
        if (range.z == 2.0f && p.z < 0.0f) {
            sceneZ = -exp2(mix(log2(max(input.cameraClip.x, 1.0e-6f)), log2(input.cameraClip.y), depth));
        }
        if (!frostfield_graph_finite(particleZ) || !frostfield_graph_finite(sceneZ)) return opacity;
        return opacity * frostfieldParticleContrast(particleZ, sceneZ, distance, contrast);
    }
    """

    // The raster fragment binds uniforms at buffer 19, eight textures from slot 16, and the
    // per-draw camera and model values at buffer 24. Viewport textures use an argument
    // buffer at 30 only when the graph reads them. Keep the member order in lockstep with
    // the fragment functions in PhysicallyBasedShading.h.
    private static let rasterGraphResources = PackedGraphView.metalSource + """

    #if defined(FROSTFIELD_GRAPH_VIEWPORT_COLOR) || defined(FROSTFIELD_GRAPH_VIEWPORT_DEPTH)
    #define FROSTFIELD_GRAPH_VIEWPORT_INPUT 1
    struct FrostfieldViewportTextures {
        texture2d<float> color [[id(0)]];
        depth2d<float> depth [[id(1)]];
    };
    #define FROSTFIELD_GRAPH_VIEWPORT_ARG , constant FrostfieldViewportTextures& viewport [[buffer(30)]]
    #define FROSTFIELD_GRAPH_VIEWPORT_VALUE , &viewport
    #else
    #define FROSTFIELD_GRAPH_VIEWPORT_ARG
    #define FROSTFIELD_GRAPH_VIEWPORT_VALUE
    #endif

    struct FrostfieldGraphResources {
        constant float4* uniforms;
        texture2d<float> texture0, texture1, texture2, texture3, texture4, texture5, texture6, texture7;
        constant FrostfieldGraphView* view;
    #if defined(FROSTFIELD_GRAPH_VIEWPORT_INPUT)
        constant FrostfieldViewportTextures* viewport;
    #endif
    };

    #if defined(FROSTFIELD_GRAPH_VIEWPORT_COLOR)
    static inline float4 frostfield_graph_scene(thread const FrostfieldGraphResources& resources, float2 uv) {
        constexpr sampler bilinear(coord::normalized, address::clamp_to_edge, filter::linear);
        return resources.viewport->color.sample(bilinear, uv, level(0));
    }

    static inline float4 frostfield_graph_scene(thread const FrostfieldGraphResources& resources, float2 uv, float lod) {
        constexpr sampler trilinear(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::linear);
        auto source = resources.viewport->color;
        lod = isfinite(lod) ? clamp(lod, 0.0f, float(source.get_num_mip_levels() - 1)) : 0.0f;
        return source.sample(trilinear, uv, level(lod));
    }
    #endif
    #if defined(FROSTFIELD_GRAPH_VIEWPORT_DEPTH)
    static inline float frostfield_graph_depth(thread const FrostfieldGraphResources& resources, float2 uv) {
        constexpr sampler nearest(coord::normalized, address::clamp_to_edge, filter::nearest);
        return resources.viewport->depth.sample(nearest, uv, level(0));
    }

    static inline float2 frostfield_graph_viewport_safe_uv(
        thread const FrostfieldSurfaceInput& input, thread const FrostfieldGraphResources& resources,
        float2 uv, float2 fallback
    ) {
        if (!all(frostfield_graph_finite(uv))) { return fallback; }
        float sampledDepth = frostfield_graph_depth(resources, uv);
        if (!(sampledDepth >= 0.0f && sampledDepth <= 1.0f)) { return fallback; }
        float fragmentDepth = input.particleDepth.x;
    #if defined(FROSTFIELD_PBR_LOG_DEPTH)
        if (input.projectionMatrix[2].w < 0.0f) {
            float near = max(input.cameraClip.x, 1.0e-6f);
            float far = input.cameraClip.y;
            float ratio = far / near;
            float span = frostfield_graph_finite(ratio) ? log2(ratio) : log2(far) - log2(near);
            if (!(span > 0.0f)) { return fallback; }
            float4 position = float4(0, 0, input.particleDepth);
            fragmentDepth = frostfieldLogDepth(position, float4(near, 1.0f / span, log2(near), far));
        }
    #endif
        bool isNearer = input.depthRange.z == 1.0f ? sampledDepth > fragmentDepth : sampledDepth < fragmentDepth;
        return isNearer ? fallback : uv;
    }
    #endif

    static inline float4 frostfield_graph_uniform(thread const FrostfieldGraphResources& resources, uint index) {
        return resources.uniforms[index];
    }

    static inline texture2d<float> frostfield_graph_texture(thread const FrostfieldGraphResources& resources, uint slot) {
        switch (slot) {
            case 0u: return resources.texture0;
            case 1u: return resources.texture1;
            case 2u: return resources.texture2;
            case 3u: return resources.texture3;
            case 4u: return resources.texture4;
            case 5u: return resources.texture5;
            case 6u: return resources.texture6;
            default: return resources.texture7;
        }
    }

    static inline float4 frostfield_graph_sample(
        thread const FrostfieldGraphResources& resources, uint slot, uint address, uint filtering, float2 uv
    ) {
        constexpr sampler repeatLinear(address::repeat, filter::linear, mip_filter::linear);
        constexpr sampler clampLinear(address::clamp_to_edge, filter::linear, mip_filter::linear);
        constexpr sampler mirrorLinear(address::mirrored_repeat, filter::linear, mip_filter::linear);
        constexpr sampler repeatNearest(address::repeat, filter::nearest, mip_filter::nearest);
        constexpr sampler clampNearest(address::clamp_to_edge, filter::nearest, mip_filter::nearest);
        constexpr sampler mirrorNearest(address::mirrored_repeat, filter::nearest, mip_filter::nearest);
        texture2d<float> texture = frostfield_graph_texture(resources, slot);
        if (filtering == 0u) {
            return address == 1u ? texture.sample(clampLinear, uv)
                : address == 2u ? texture.sample(mirrorLinear, uv) : texture.sample(repeatLinear, uv);
        }
        return address == 1u ? texture.sample(clampNearest, uv)
            : address == 2u ? texture.sample(mirrorNearest, uv) : texture.sample(repeatNearest, uv);
    }

    static inline float frostfield_graph_lod(texture2d<float> texture, float level) {
        float last = float(texture.get_num_mip_levels() - 1);
        return isfinite(level) ? clamp(level, 0.0f, last) : 0.0f;
    }

    static inline float4 frostfield_graph_sample(
        thread const FrostfieldGraphResources& resources, uint slot, uint address, uint filtering, float2 uv, float requested
    ) {
        constexpr sampler repeatLinear(address::repeat, filter::linear, mip_filter::nearest);
        constexpr sampler clampLinear(address::clamp_to_edge, filter::linear, mip_filter::nearest);
        constexpr sampler mirrorLinear(address::mirrored_repeat, filter::linear, mip_filter::nearest);
        constexpr sampler repeatNearest(address::repeat, filter::nearest, mip_filter::nearest);
        constexpr sampler clampNearest(address::clamp_to_edge, filter::nearest, mip_filter::nearest);
        constexpr sampler mirrorNearest(address::mirrored_repeat, filter::nearest, mip_filter::nearest);
        texture2d<float> texture = frostfield_graph_texture(resources, slot);
        float mip = frostfield_graph_lod(texture, requested);
        if (filtering == 0u) {
            return address == 1u ? texture.sample(clampLinear, uv, level(mip))
                : address == 2u ? texture.sample(mirrorLinear, uv, level(mip)) : texture.sample(repeatLinear, uv, level(mip));
        }
        return address == 1u ? texture.sample(clampNearest, uv, level(mip))
            : address == 2u ? texture.sample(mirrorNearest, uv, level(mip)) : texture.sample(repeatNearest, uv, level(mip));
    }

    static inline float frostfield_graph_gradient_lod(texture2d<float> texture, float2 ddx, float2 ddy) {
        if (!all(isfinite(ddx)) || !all(isfinite(ddy))) return 0.0f;
        float2 size = float2(texture.get_width(), texture.get_height());
        float rho = max(length(ddx * size), length(ddy * size));
        return rho > 0.0f && isfinite(rho) ? log2(rho) : 0.0f;
    }

    static inline float4 frostfield_graph_sample(
        thread const FrostfieldGraphResources& resources, uint slot, uint address, uint filtering,
        float2 uv, float2 ddx, float2 ddy
    ) {
        texture2d<float> texture = frostfield_graph_texture(resources, slot);
        return frostfield_graph_sample(
            resources, slot, address, filtering, uv, frostfield_graph_gradient_lod(texture, ddx, ddy)
        );
    }

    // Implemented by the shared material extensions included after the graph source.
    static inline float3 frostfieldBumpNormal(float3 normal, float3 dx, float3 dy, float2 heightDelta);

    static inline float3 frostfield_graph_bump(
        thread const FrostfieldSurfaceInput& input, thread const FrostfieldGraphResources& resources,
        uint slot, uint address, uint filtering, uint coordinateSet, float4 row0, float4 row1, float scale
    ) {
        float2 source = coordinateSet == 0u ? input.uv0 : input.uv1;
        float2 uv = float2(dot(row0.xy, source) + row0.z, dot(row1.xy, source) + row1.z);
        float2 dx = dfdx(uv), dy = dfdy(uv);
        float height = frostfield_graph_sample(resources, slot, address, filtering, uv).r;
        float2 delta = float2(
            frostfield_graph_sample(resources, slot, address, filtering, uv + dx).r,
            frostfield_graph_sample(resources, slot, address, filtering, uv + dy).r
        ) - height;
        float3 bumped = frostfieldBumpNormal(input.vertexNormal, dfdx(input.worldPosition), dfdy(input.worldPosition), scale * delta);
        return frostfield_graph_to_tangent(input, bumped);
    }
    """

    private static func makePipeline(
        device: any MTLDevice, descriptor: MTLRenderPipelineDescriptor
    ) throws -> any MTLRenderPipelineState {
        do {
            return try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
    }

    private static func makePunctualShadowFallback(device: any MTLDevice) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: 1, height: 1, mipmapped: false
        )
        descriptor.textureType = .type2DArray
        descriptor.arrayLength = 1
        descriptor.storageMode = .private
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw Failure.resource("unused punctual shadow array")
        }
        texture.label = "Frostfield unused punctual shadow array"
        return texture
    }

    private static func makeBuffer<T>(_ values: [T], on device: any MTLDevice, label: String) throws -> any MTLBuffer {
        let buffer = values.withUnsafeBytes { bytes in
            bytes.baseAddress.flatMap { device.makeBuffer(bytes: $0, length: bytes.count, options: .storageModeShared) }
        }
        guard let buffer else { throw Failure.resource(label) }
        buffer.label = label
        return buffer
    }

    public enum Failure: LocalizedError, Equatable {
        case library(String)
        case function(String)
        case pipeline(String)
        case fragmentBindings(String)
        case resource(String)
        case dynamicGeometryCapacity
        case dynamicGeometryNotPrepared
        case dynamicGeometryRange
        case unsupportedDynamicGeometry
        case drawRangeWithIndirectArguments
        case dynamicGeometryCommandBuffer
        case dynamicGeometryBuffersBusy
        case dynamicGeometryRevision
        case dynamicGeometryTopology
        case wrongDevice
        case colorTexture
        case missingSecondaryTextureCoordinates
        case shadowMap
        case attachments
        case multipleRenderTargets
        case stencilAttachments
        case stencilPass
        case depthMode
        case logarithmicDepthCamera
        case logarithmicDepthPass
        case logarithmicDepthConsumer
        case depthOutputPixelFormat
        case highPrecisionView
        case viewport
        case scissor
        case transmissionBackdrop
        case invalidTransmissionBackdrop
        case transmissionAttachments
        case transmissionCommandBuffer
        case transmissionBlitEncoder
        case transform
        case encoder
        case unsupportedSurfaceGraph
        /// A surface graph reads the view or projection matrix or clip distances in a
        /// pass that supplies a combined clip matrix without a `ShaderGraphCamera`, or
        /// the graph camera's projection times view does not equal the clip matrix.
        case graphCameraInputs
        /// A shadow caster's coverage graph reads camera data its light view cannot provide.
        case graphShadowCameraInput(UUID, MaterialGraph.NodeID)
        /// A colored shadow caster is batched or draws from an indirect argument buffer.
        case graphCastShadowGeometry(UUID)
        case transmittedShadowCaster(UUID)
        case environment
        case morphTargetCount
        case morphedDeformation
        case unexpectedMorphWeights
        case instancingNotPrepared
        case instancedTransmission
        case projectionJitter
        case sampleCount
        case renderState(String)
        case pickingPreparation
        case pickingGeometryChanged
        case screenOcclusion
        case surfaceAttributes

        public var errorDescription: String? {
            switch self {
            case .library(let message): "Could not load Frostfield’s shader library: \(message)"
            case .function(let name): "The shader library does not contain \(name)."
            case .pipeline(let message): "Could not create the mesh pipeline: \(message)"
            case .fragmentBindings(let name): "\(name) must bind one read-only float4 at fragment buffer 0."
            case .resource(let name): "Metal could not allocate \(name)."
            case .dynamicGeometryCapacity: "The dynamic geometry capacity is invalid or the replacement exceeds it."
            case .dynamicGeometryNotPrepared: "Prepare a dynamic geometry renderer before replacing its geometry."
            case .dynamicGeometryRange: "A dynamic vertex update needs a non-empty span inside the current vertex list."
            case .unsupportedDynamicGeometry: "Use static geometry for vertex deformation, skinning, morphs, instances or GPU indirect draws."
            case .drawRangeWithIndirectArguments: "GPU indirect records supply their own range. Reset the renderer's draw range before using them."
            case .dynamicGeometryCommandBuffer: "Use an uncommitted command buffer for dynamic geometry."
            case .dynamicGeometryBuffersBusy: "Complete or discard an earlier command buffer before encoding another geometry revision."
            case .dynamicGeometryRevision: "The renderer has reached its dynamic geometry revision limit."
            case .dynamicGeometryTopology: "Preserving vertex identity requires the current vertex count and index sequence."
            case .wrongDevice: "The supplied Metal object belongs to a different device."
            case .colorTexture: "Use a shader-readable 2D RGBA, BGRA, or RGBA block-compressed color texture."
            case .missingSecondaryTextureCoordinates: "The material requires secondary texture coordinates on every vertex."
            case .shadowMap: "A material that receives shadows needs a directional shadow map on its scene graph."
            case .attachments:
                "Use valid color and Depth32Float render subresources with matching dimensions and sample counts, "
                    + "and the renderer's color formats in attachment order."
            case .multipleRenderTargets:
                "Use two to eight defined color formats with unblended custom meshes. "
                    + "Primitive, transmission, screen occlusion and surface attribute passes use a single color target."
            case .stencilAttachments:
                "Use a shared Depth32Float_Stencil8 attachment with matching color dimensions and sample count, "
                    + "stencil load or clear, and a compatible color resolve when needed."
            case .stencilPass:
                "Stencil attachments require compatible indexed mesh color pipelines."
            case .depthMode: "Prepare each mesh with the depth mode used by the render pass."
            case .logarithmicDepthCamera:
                "Logarithmic depth needs a finite forward perspective or orthographic camera and its split projection."
            case .logarithmicDepthPass:
                "Logarithmic depth requires supported single-sample mesh pipelines, a full viewport depth range, and a clear depth of one."
            case .logarithmicDepthConsumer:
                "This pass requires forward or reversed projection depth."
            case .depthOutputPixelFormat: "Use a linear color format with enough channels for the selected depth or distance output."
            case .highPrecisionView: "High-precision explicit views need a ShaderGraphCamera with an affine view and matching camera position."
            case .viewport: "The viewport must fit within the color attachment and use increasing depth values in zero to one."
            case .scissor: "The scissor rectangle must fit inside the render attachment."
            case .transmissionBackdrop: "Render transmitting materials in a scene graph with a transmission backdrop."
            case .invalidTransmissionBackdrop:
                "Use a separate, tracked, shader-readable RGBA16Float backdrop with matching dimensions and a full mip chain."
            case .transmissionAttachments:
                "Transmission needs stored, tracked RGBA16Float color and depth attachments with defined load actions."
            case .transmissionCommandBuffer: "Transmission needs an uncommitted command buffer that retains resources."
            case .transmissionBlitEncoder: "Metal could not create the transmission backdrop blit encoder."
            case .transform: "The mesh and camera transforms must be finite and invertible, with finite material parameters."
            case .encoder: "Metal could not create a mesh render encoder."
            case .unsupportedSurfaceGraph: "The surface graph has no vertex offset to deform the mesh with."
            case .graphCameraInputs:
                "A surface graph reads the camera's view or projection matrix or clip distances; with a caller-supplied clip matrix, "
                    + "pass a ShaderGraphCamera whose projection times view matrix equals that clip matrix."
            case .transmittedShadowCaster(let node):
                "Transmitted shadow caster \(node) needs supported PBR geometry and a stock glass or castShadowColor output."
            case .graphCastShadowGeometry(let node):
                "Colored shadow caster \(node) is batched or drawn indirectly, which the colored directional shadow pass does not support. "
                    + "Draw it as a plain, skinned or instanced mesh, or set transmissionLayers to zero."
            case .graphShadowCameraInput(let node, let input):
                "Shadow caster \(node) reads a camera input or viewport snapshot at surface graph node \(input). "
                    + "Use material, geometry, or light-pass screen inputs for raster shadow coverage."
            case .environment: "Scene environment intensity must be finite and nonnegative, and its rotation finite."
            case .morphTargetCount: "A mesh supports up to \(MeshRenderer.maximumMorphTargetCount) morph targets."
            case .morphedDeformation: "Morph targets cannot combine with a surface graph's position offset."
            case .unexpectedMorphWeights: "Morph weights need a mesh prepared with morph targets."
            case .instancingNotPrepared: "Prepare the mesh with supportsInstancing to draw a node with instances."
            case .instancedTransmission: "Transmitting materials cannot be drawn with instances."
            case .projectionJitter: "Scene projection jitter must be finite, with each component between -1 and 1 pixel."
            case .sampleCount:
                "Use a sample count of 1, 2, 4 or 8 that the device supports, matching the color and depth attachments."
            case .renderState(let message): message
            case .pickingPreparation:
                "This mesh needs pose-specific picking for skinning, curve flow or a texture-driven graph offset."
            case .pickingGeometryChanged: "The mesh geometry, active draw range or shader pose changed while preparing picking."
            case .screenOcclusion:
                "Screen-space occlusion must be a shader-readable 2D RGBA16Float texture on the device, sized like the color attachment."
            case .surfaceAttributes:
                "Surface attributes need a single-sample RG32Uint color attachment and Depth32Float depth of equal size."
            }
        }
    }
}

/// Draw counts from one scene graph encode.
public struct SceneDrawStatistics: Equatable, Sendable {
    /// Ordinary mesh commands in the color or attribute pass, including indirect records.
    public var drawnMeshCount = 0
    /// Extra commands submitted for toon hulls in the scene color pass.
    public var toonOutlineDrawCount = 0
    /// Instances in CPU-counted draws. Counts read by the GPU remain on the GPU.
    public var drawnInstanceCount = 0
    /// Ordinary nodes and batched objects left out by the camera frustum.
    public var culledMeshCount = 0
    /// Caster draws encoded across every directional and punctual depth pass.
    public var shadowDrawCount = 0
    /// Directional and punctual depth passes encoded, including clear-only updates.
    public var shadowPassCount = 0
    /// Ordinary casters and batched objects outside the light's view.
    public var culledShadowCount = 0
    /// Metal indirect batch executions across the color, attribute and shadow passes.
    public var indirectBatchExecutionCount = 0
    /// Indexed mesh and primitive argument records submitted in the color, attribute and shadow passes.
    /// Records that the GPU turns into empty draws still count as submissions.
    /// Primitive records replace the CPU point, segment and sprite counts for that draw.
    public var indirectArgumentDrawCount = 0
    /// Point, line and sprite nodes drawn in the color pass.
    public var drawnPrimitiveCount = 0
    /// Point, line and sprite nodes left out of the color pass by the camera frustum.
    public var culledPrimitiveCount = 0
    /// CPU-known triangles submitted to the color or attribute pass, including two per sprite.
    /// Point and wide-line expansion is counted by the logical primitive fields below.
    public var submittedTriangleCount = 0
    /// CPU-known triangles submitted across all shadow passes. GPU-authored counts stay on the GPU.
    public var shadowTriangleCount = 0
    /// Logical point and line counts before fragment or clip rejection.
    public var submittedPointCount = 0
    public var submittedLineSegmentCount = 0
    public var submittedSpriteCount = 0

    public init() {}
}

/// The nearest front-facing mesh intersection in a scene graph.
public struct SceneHit: Equatable, Sendable {
    public let nodeID: UUID
    public let intersection: MeshRayHit
    /// The hit instance's index in the node's `InstanceSet`, or nil for a node without instances.
    public let instanceIndex: Int?
    /// The stable slot of a hit inside a `BatchedMesh`.
    public let batchInstanceID: BatchedMesh.InstanceID?

    init(nodeID: UUID, intersection: MeshRayHit, instanceIndex: Int?, batchInstanceID: BatchedMesh.InstanceID? = nil) {
        self.nodeID = nodeID
        self.intersection = intersection
        self.instanceIndex = instanceIndex
        self.batchInstanceID = batchInstanceID
    }
}

extension ResolvedSceneNode {
    func appendMeshHits(ray: SceneRay, to hits: inout [SceneHit]) throws {
        if let batch {
            try batch.appendHits(ray: ray, node: self, to: &hits)
        } else if let mesh, instances?.count != 0 {
            try mesh.appendPickingHits(ray: ray, node: self, to: &hits)
        }
    }

    func nearestMeshHit(ray: SceneRay) throws -> SceneHit? {
        if let batch { return try batch.nearestHit(ray: ray, node: self) }
        guard let mesh, instances?.count != 0 else { return nil }
        return try mesh.pickingIntersection(ray: ray, node: self).map {
            SceneHit(nodeID: id, intersection: $0.hit, instanceIndex: $0.instance)
        }
    }
}

public extension SceneGraph {
    /// Picks a mesh at normalized window coordinates with a top-left origin.
    /// A hit must lie between the camera's clip planes. Equal distances use
    /// the first mesh in depth-first order, matching the depth test. The first
    /// pick builds a ``MeshBVH`` for a static mesh or a CPU skin/morph pose.
    /// Later poses with the same draw range refit the hierarchy. Shader-deformed
    /// meshes use their current triangles from the GPU. Instanced nodes test
    /// every instance and report its index; equal distances keep the lower index.
    /// Level-of-detail nodes test the level selected for the graph's camera.
    /// Hidden subtrees and layers outside the graph's `cameraLayers` are skipped.
    func pick(at normalized: SIMD2<Float>, aspectRatio: Float) throws -> SceneHit? {
        try pick(at: normalized, aspectRatio: aspectRatio, layers: cameraLayers)
    }

    /// Picks a mesh using the supplied layer mask in place of the graph's camera layers,
    /// like Three.js `Raycaster.layers`. Hidden subtrees are skipped.
    func pick(at normalized: SIMD2<Float>, aspectRatio: Float, layers: SceneLayers) throws -> SceneHit? {
        let ray = try cameraRay(at: normalized, aspectRatio: aspectRatio)
        return try nearestMeshHit(ray: ray, cameraPosition: cameraTransform.translation, layers: layers)
    }

    /// Picks the nearest visible mesh in the scene, or among the direct children
    /// of the supplied group. The graph camera selects levels of detail and layers.
    /// An unknown group ID throws `EditingError.missingNode`.
    func pick(ray: SceneRay, in group: UUID? = nil) throws -> SceneHit? {
        try nearestMeshHit(ray: ray, cameraPosition: cameraTransform.translation, layers: cameraLayers, group: group)
    }

    /// Picks using an explicit render camera, including an orthographic camera.
    /// The view's world position selects levels of detail.
    func pick(
        at normalized: SIMD2<Float>, camera: ShaderGraphCamera, depthMode: MeshDepthMode = .forward,
        in group: UUID? = nil
    ) throws -> SceneHit? {
        try nearestMeshHit(
            ray: camera.ray(at: normalized, depthMode: depthMode),
            cameraPosition: camera.pickingPosition(), layers: cameraLayers, group: group
        )
    }

    /// Returns every visible mesh triangle crossed by a world ray, nearest first.
    /// Equal distances keep depth-first node order, then instance and triangle
    /// order. Use `pick` when only the nearest triangle is needed.
    func intersections(ray: SceneRay) throws -> [SceneHit] {
        try meshHits(ray: ray, cameraPosition: cameraTransform.translation)
    }

    /// Returns every mesh hit between the graph camera's clip planes.
    func intersections(at normalized: SIMD2<Float>, aspectRatio: Float) throws -> [SceneHit] {
        try intersections(ray: cameraRay(at: normalized, aspectRatio: aspectRatio))
    }

    /// Returns every mesh hit between the explicit render camera's clip planes.
    func intersections(
        at normalized: SIMD2<Float>, camera: ShaderGraphCamera, depthMode: MeshDepthMode = .forward
    ) throws -> [SceneHit] {
        try meshHits(ray: camera.ray(at: normalized, depthMode: depthMode), cameraPosition: camera.pickingPosition())
    }

    private func meshHits(ray: SceneRay, cameraPosition: SIMD3<Float>) throws -> [SceneHit] {
        var hits: [SceneHit] = []
        for node in try resolve(cameraPosition: cameraPosition) where renders(node) {
            try node.appendMeshHits(ray: ray, to: &hits)
        }
        hits.sort { $0.intersection.distance < $1.intersection.distance }
        return hits
    }

    package func nearestMeshHit(
        ray: SceneRay, cameraPosition: SIMD3<Float>, layers: SceneLayers, viewLayers: SceneLayers? = nil,
        group: UUID? = nil
    ) throws -> SceneHit? {
        let children = try pickingChildren(in: group)
        guard children?.isEmpty != true else { return nil }
        var nearest: SceneHit?
        for node in try resolve(cameraPosition: cameraPosition)
            where children?.contains(node.id) != false && node.isVisible && node.layers.intersects(layers)
                && viewLayers.map({ node.layers.intersects($0) }) != false {
            guard let hit = try node.nearestMeshHit(ray: ray),
                  nearest.map({ hit.intersection.distance < $0.intersection.distance }) != false else { continue }
            nearest = hit
        }
        return nearest
    }

    internal func pickingChildren(in group: UUID?) throws -> Set<UUID>? {
        guard let group else { return nil }
        guard let node = node(withID: group) else { throw EditingError.missingNode(group) }
        return Set(node.children.map(\.id))
    }

    /// Encodes the graph without submitting the command buffer and returns its
    /// draw counts. A graph with no meshes still applies the pass's load and store actions.
    /// For multiple color attachments, prepare every rendered mesh with the
    /// same ordered custom formats. Primitives require a separate single-target pass.
    /// To render to a single array layer, cube face, or volume plane, select its
    /// slice or depth plane and mip level in the Metal attachment descriptor.
    /// The depth attachment's selected mip must match the color dimensions.
    /// For stencil, bind one Depth32Float_Stencil8 texture as both attachments and
    /// prepare each mesh with the stencil layout. Clear stencil to a known value
    /// or load a previous pass's stored values. Multisampled color can resolve
    /// into a separate texture with matching dimensions and format.
    /// Stencil applies to color draws only.
    /// Prepare each mesh for the pass's `depthMode`. Clear depth to 1 for forward
    /// depth or 0 for reversed depth.
    ///
    /// Level-of-detail nodes draw the level selected for the camera in every pass.
    /// A node whose `frustumCulled` is true is left out of the color pass when its
    /// world bounds lie outside the camera frustum. The bounds come from the node's
    /// `cullingBounds`, else from the mesh's bounds widened by the node's morph
    /// offsets and every instance; skinned and graph-deformed meshes without
    /// node bounds are never culled. An instanced node is culled as a whole.
    /// Shadow passes never use the camera frustum: they skip a caster only when
    /// its bounds are outside that light's view, where it could not write depth.
    ///
    /// Transmission requires a separate, tracked, shader-readable RGBA16Float
    /// backdrop with the color attachment's dimensions and a complete mip chain.
    /// Use a retained command buffer and tracked, single-sample attachments
    /// with `.clear` or `.load`. Use standalone, blit-readable RGBA16Float color
    /// and store-capable depth. With `.load`, preload only the background and
    /// opaque or masked draws intended to be visible through glass. The pass
    /// snapshots the selected color mip and layer before drawing transmission and alpha blends. It omits
    /// transmitting meshes from the opaque shadow map and uses one snapshot for
    /// every transmitting draw. Discard the buffer on failure.
    ///
    /// With `punctualShadowMaps`, lights that carry a `PunctualLight.Shadow` take
    /// the maps' slots in depth-first order until each kind's capacity is used;
    /// the rest, and every light when the maps are nil, render unshadowed. When a
    /// PBR mesh is present, one depth pass per assigned slice runs in this command
    /// buffer before the color pass. Casters follow the same rules as the
    /// directional map, and every PBR surface receives these shadows.
    ///
    /// A cascaded `shadowMap` renders one depth pass per cascade, culling casters by each
    /// cascade's light view. Fit it to the viewing camera before encoding.
    /// Maps prepared with `.onDemand` reuse depth until their projection or light-slot
    /// layout changes or the caller requests an update. Submit and finish a command
    /// buffer that uses an on-demand map before encoding it into another buffer.
    ///
    /// `screenSpaceOcclusion` is the RGBA16Float output of
    /// `AmbientOcclusion.encodeVisibility`, sized like the color attachment. Opaque and
    /// masked PBR surfaces then light image-based and ambient terms around its bent normal,
    /// darken diffuse light by its multiple-bounce visibility and specular light by the
    /// overlap of the reflection cone with the visibility cone. Direct light and emission
    /// are not occluded.
    ///
    /// `scissor` clips color and depth draws to top-left attachment pixels.
    /// An empty rectangle skips the camera's draws. Pass load and store
    /// actions and shadow-map draws still run.
    ///
    /// `indirectDraws` maps a node ID to ordered GPU-written arguments for that
    /// node's ordinary mesh. Color and shadow passes share the validated records.
    /// Visibility, layers, transforms, render order and authored culling bounds
    /// apply as usual. End the producer encoder before calling this method.
    /// Use `ordering` to override opaque or transparent ordering, or use
    /// traversal order within each phase. Depth-disabled overlays stay last.
    ///
    /// `toonOutline` draws a hull for each classic toon mesh by default. Use
    /// `ToonOutlinePass.outlineEffect` for eligible meshes after the scene color.
    /// Both modes use the scene depth and the same prepared geometry.
    /// `screenSpaceDirectionalShadow` consumes a matching visibility mask on
    /// opaque and masked PBR draws, multiplying any directional shadow map.
    /// The mask needs the same full view and may have a smaller resolution.
    /// `motionVectors` records mesh and pose velocity after the color pass.
    /// Store the single-sample Depth32Float attachment for that recording.
    ///
    /// `updatingSkeletons`, `updatingBoxes` and `updatingPlanes` refresh the
    /// attachments for this draw only. Supply a command buffer with retained references.
    @discardableResult
    func encode(
        pass: MTLRenderPassDescriptor,
        commandBuffer: any MTLCommandBuffer,
        shadowMap: DirectionalShadowMap? = nil,
        transmissionBackdrop: (any MTLTexture)? = nil,
        punctualShadowMaps: PunctualShadowMaps? = nil,
        depthMode: MeshDepthMode = .forward,
        screenSpaceOcclusion: (any MTLTexture)? = nil,
        screenSpaceDirectionalShadow: ScreenSpaceShadows.DirectionalVisibility? = nil,
        scissor: MTLScissorRect? = nil,
        indirectDraws: [UUID: MeshIndirectDraws] = [:],
        ordering: SceneRenderOrdering = .init(),
        updatingSkeletons: [DebugHelpers.SkeletonAttachment] = [],
        updatingBoxes: [DebugHelpers.BoxAttachment] = [],
        updatingPlanes: [DebugHelpers.PlaneAttachment] = [],
        toonOutline: ToonOutlinePass? = nil,
        motionVectors: RasterMotionVectors.Recording? = nil
    ) throws -> SceneDrawStatistics {
        let graph = try refreshingDebugHelpers(
            skeletons: updatingSkeletons, boxes: updatingBoxes, planes: updatingPlanes, commandBuffer: commandBuffer
        )
        return try MeshRenderer.encode(
            graph, pass: pass, commandBuffer: commandBuffer, shadowMap: shadowMap,
            punctualShadowMaps: punctualShadowMaps, depthMode: depthMode, transmissionBackdrop: transmissionBackdrop,
            screenOcclusion: screenSpaceOcclusion, screenDirectionalShadow: screenSpaceDirectionalShadow,
            scissor: scissor, indirectDraws: indirectDraws, ordering: ordering,
            toonOutline: toonOutline, motionVectors: motionVectors
        )
    }

    /// Encodes a view using a caller-supplied Metal projection and camera position.
    /// Supply a world-to-clip matrix whose depth mapping matches the prepared meshes.
    /// Clear depth to 0 for reversed depth or to 1 for forward depth. Use a separate
    /// render pass for each view. A nil viewport covers the entire attachment.
    /// `scissor` clips draws independently of the viewport and uses top-left attachment pixels.
    /// For orthographic shading, provide the fixed direction from surfaces toward the camera.
    /// Transmission snapshots the full attachment. Rough transmission may sample
    /// neighboring viewports through the generated mipmaps. Use a dedicated
    /// attachment for each view when those mipmaps must stay isolated.
    /// Punctual shadow maps render once per call, as described for the default view.
    /// Culling and level-of-detail selection use this view's matrix and camera position.
    /// Surface graphs that read the view or projection matrix or the clip distances
    /// need `graphCamera`, whose projection times view must equal `clipFromWorld`;
    /// a glTF camera view supplies one as `GLTFScene.Camera.View.graphCamera`.
    /// High precision also needs this split camera and its world position to rebase
    /// ordinary mesh and primitive draws before projection.
    /// `indirectDraws` uses the same node IDs and GPU argument contract as the default view.
    /// Live debug attachments use the same draw-only snapshot and command-buffer retention contract.
    /// `toonOutline` uses this view's projection and the standard scene depth attachment.
    /// `screenSpaceDirectionalShadow` needs this view's matching depth, projection
    /// and full attachment viewport. Use a separate mask for each view.
    /// `motionVectors` uses this view and the stored single-sample depth attachment.
    @discardableResult
    func encode(
        pass: MTLRenderPassDescriptor,
        commandBuffer: any MTLCommandBuffer,
        clipFromWorld: simd_float4x4,
        cameraPosition: SIMD3<Float>,
        directionToCamera: SIMD3<Float>? = nil,
        viewport: MTLViewport? = nil,
        depthMode: MeshDepthMode = .forward,
        shadowMap: DirectionalShadowMap? = nil,
        transmissionBackdrop: (any MTLTexture)? = nil,
        punctualShadowMaps: PunctualShadowMaps? = nil,
        graphCamera: ShaderGraphCamera? = nil,
        screenSpaceOcclusion: (any MTLTexture)? = nil,
        screenSpaceDirectionalShadow: ScreenSpaceShadows.DirectionalVisibility? = nil,
        scissor: MTLScissorRect? = nil,
        indirectDraws: [UUID: MeshIndirectDraws] = [:],
        ordering: SceneRenderOrdering = .init(),
        updatingSkeletons: [DebugHelpers.SkeletonAttachment] = [],
        updatingBoxes: [DebugHelpers.BoxAttachment] = [],
        updatingPlanes: [DebugHelpers.PlaneAttachment] = [],
        toonOutline: ToonOutlinePass? = nil,
        motionVectors: RasterMotionVectors.Recording? = nil
    ) throws -> SceneDrawStatistics {
        let graph = try refreshingDebugHelpers(
            skeletons: updatingSkeletons, boxes: updatingBoxes, planes: updatingPlanes, commandBuffer: commandBuffer
        )
        return try MeshRenderer.encode(
            graph, pass: pass, commandBuffer: commandBuffer, shadowMap: shadowMap,
            punctualShadowMaps: punctualShadowMaps,
            view: (clipFromWorld, cameraPosition, directionToCamera), viewport: viewport, depthMode: depthMode,
            transmissionBackdrop: transmissionBackdrop, graphCamera: graphCamera,
            screenOcclusion: screenSpaceOcclusion, screenDirectionalShadow: screenSpaceDirectionalShadow,
            scissor: scissor, indirectDraws: indirectDraws, ordering: ordering,
            toonOutline: toonOutline, motionVectors: motionVectors
        )
    }
}

// MARK: - Offscreen mesh depth

/// Plain and skinned depth pipelines shared by selection and receiver-plane captures.
struct MeshDepthCapturePipelines {
    let plain: any MTLRenderPipelineState
    private let skinned: any MTLRenderPipelineState
    private let masked: any MTLRenderPipelineState
    private let skinnedMasked: any MTLRenderPipelineState

    init(device: any MTLDevice, label: String) throws {
        let library: any MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw PostProcessEffectError.pipeline(error.localizedDescription)
        }
        guard let vertex = library.makeFunction(name: "frostfield_shadow_vertex"),
              let skinnedVertex = library.makeFunction(name: "frostfield_skinned_shadow_vertex"),
              let mask = library.makeFunction(name: "frostfield_shadow_mask") else {
            throw PostProcessEffectError.missingShader("frostfield_shadow_vertex")
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = label
        descriptor.depthAttachmentPixelFormat = .depth32Float
        descriptor.inputPrimitiveTopology = .triangle
        do {
            descriptor.vertexFunction = vertex
            plain = try device.makeRenderPipelineState(descriptor: descriptor)
            descriptor.vertexFunction = skinnedVertex
            skinned = try device.makeRenderPipelineState(descriptor: descriptor)
            descriptor.fragmentFunction = mask
            skinnedMasked = try device.makeRenderPipelineState(descriptor: descriptor)
            descriptor.vertexFunction = vertex
            masked = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            throw PostProcessEffectError.pipeline(error.localizedDescription)
        }
    }

    func setPipeline(on encoder: any MTLRenderCommandEncoder, skinned: Bool, masked: Bool) {
        if masked {
            encoder.setRenderPipelineState(skinned ? skinnedMasked : self.masked)
        } else {
            encoder.setRenderPipelineState(skinned ? self.skinned : plain)
        }
    }
}

extension MeshRenderer {
    /// Encodes depth-only draws of the resolved graph nodes that `includes` accepts,
    /// with the shadow vertex stages, so selection masks follow skinning, morphing,
    /// instancing, deformation and alpha cutoffs exactly as the color pass does.
    /// Every draw is validated before `makeEncoder` runs. Returns the draw count.
    static func encodeDepthOnly(
        _ graph: SceneGraph,
        clipFromWorld: simd_float4x4,
        commandBuffer: any MTLCommandBuffer,
        includes: (UUID) -> Bool,
        indirectDraws: [UUID: MeshIndirectDraws] = [:],
        shadowView: GraphViewContext? = nil,
        makeEncoder: () throws -> any MTLRenderCommandEncoder,
        setPipeline: (any MTLRenderCommandEncoder, _ skinned: Bool, _ masked: Bool) -> Void
    ) throws -> Int {
        let device = commandBuffer.device
        let cameraPosition = shadowView?.cameraPosition ?? graph.cameraTransform.translation
        var draws: [(renderer: MeshRenderer, draw: PreparedDraw)] = []
        var batchBytes = 0
        let frustum = try? ViewFrustum(clipFromWorld: clipFromWorld)
        let nodes = try graph.resolve(cameraPosition: cameraPosition)
        let selected = nodes.filter { graph.renders($0) && includes($0.id) && (shadowView == nil || $0.castsShadow) }
        guard !selected.contains(where: { $0.mesh?.usesStencilAttachment == true }) else { throw Failure.stencilPass }
        try validateIndirectTargets(indirectDraws, nodes: nodes, commandBuffer: commandBuffer)
        for node in selected {
            guard let renderer = node.mesh ?? node.batch?.renderer else { continue }
            guard renderer.device === device else { throw Failure.wrongDevice }
            guard renderer.rawProgram == nil else { throw RawMeshProgram.Failure.unsupportedAuxiliaryPass }
            let shadingCamera = graph.orthographicDirectionToCamera.map { SIMD4($0, 0) } ?? SIMD4(cameraPosition, 1)
            if shadowView != nil {
                guard !renderer.isBlended, !renderer.requiresTransmissionBackdrop else { continue }
                if let graphNode = renderer.graphShadow?.cameraNode {
                    throw Failure.graphShadowCameraInput(node.id, graphNode)
                }
            }
            var draw = try renderer.prepare(
                world: node.worldTransform, clipFromWorld: clipFromWorld, shadingCamera: shadingCamera,
                parameters: node.materialParameters, jointPalette: node.jointPalette, morphWeights: node.morphWeights,
                instances: node.instances
            )
            guard draw.instanceCount > 0 else { continue }
            draw.indirectSource = indirectDraws[node.id]
            guard node.batch != nil || draw.indirectSource != nil || !draw.indexRange.isEmpty else { continue }
            if let batch = node.batch, let snapshot = try batch.drawSnapshot(world: node.worldTransform) {
                let context = BatchedMesh.SortContext(pass: .selectionDepth, clipFromWorld: clipFromWorld, isBlended: false)
                let plan = try batch.plan(
                    snapshot, frustum: node.frustumCulled && snapshot.perObjectFrustumCulled ? frustum : nil, context: context
                )
                guard let packet = try batch.acquire(plan, commandBuffer: commandBuffer, bytesUsed: &batchBytes) else { continue }
                draw.batch = packet
            }
            draws.append((renderer, draw))
        }
        var preparedDeformations: [ObjectIdentifier: any MTLBuffer] = [:]
        for index in draws.indices {
            let renderer = draws[index].renderer
            draws[index].draw.deformedVertices = try renderer.prepareDeformation(for: commandBuffer, cached: &preparedDeformations)
            try renderer.acquireDrawInputs(&draws[index].draw, for: commandBuffer)
        }
        let encoder = try makeEncoder()
        for item in draws {
            item.draw.palette?.retainBufferIfNeeded(on: commandBuffer)
            encoder.setCullMode(item.renderer.renderState.cullMode)
            item.renderer.encodeShadowCaster(item.draw, clip: item.draw.clip, into: encoder, coverageView: shadowView) { skinned, masked in
                setPipeline(encoder, skinned, masked)
            }
        }
        encoder.endEncoding()
        return draws.reduce(0) { $0 + $1.draw.drawCallCount }
    }
}
