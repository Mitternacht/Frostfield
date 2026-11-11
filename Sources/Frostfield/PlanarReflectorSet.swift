import Foundation
import Metal
import simd

/// Discovers authored mirrors visible to each view and supplies their projected
/// reflection materials. Serialize access on one render thread and queue.
public final class PlanarReflectorSet {
    /// An opt-in projected reflection overlay for an authored PBR material.
    /// Texture resources follow the renderer's usual Metal ownership rules.
    public struct Appearance: @unchecked Sendable {
        public let blur: PlanarReflectionBlur.Settings
        public let mix: PlanarReflectionMix
        public let distortionMap: (any MTLTexture)?
        public let usesAuthoredNormalMap: Bool

        public init(
            blur: PlanarReflectionBlur.Settings = .init(), mix: PlanarReflectionMix,
            distortionMap: (any MTLTexture)? = nil, usesAuthoredNormalMap: Bool = true
        ) throws {
            guard blur.isValid else { throw PlanarReflectionBlur.Failure.settings }
            self.blur = blur
            self.mix = mix
            self.distortionMap = distortionMap
            self.usesAuthoredNormalMap = usesAuthoredNormalMap
        }
    }

    public struct Mirror: Sendable {
        public let nodeID: UUID
        /// The mesh's oriented plane in local coordinates. The positive side reflects.
        public let localPlane: Plane3D
        /// Nil uses the original sharp unlit mirror. An overlay retains the
        /// source PBR material's ordinary lighting, maps and roughness.
        public let appearance: Appearance?

        public init(nodeID: UUID, localPlane: Plane3D) {
            self.init(nodeID: nodeID, localPlane: localPlane, appearance: nil)
        }

        public init(nodeID: UUID, localPlane: Plane3D, appearance: Appearance?) {
            self.nodeID = nodeID
            self.localPlane = localPlane
            self.appearance = appearance
        }
    }

    public struct Frame {
        /// Draw this graph with the source camera in the same command buffer
        /// before committing it. Prepare a new frame for each source view.
        public let graph: SceneGraph
        public let captureCount: Int
    }

    private static let maximumCaptures = 32
    private static let maximumTargetBytes = 256 * 1024 * 1024
    private let queue: any MTLCommandQueue
    private let size: SIMD2<Int>
    private let maximumDepth: Int
    private let captureBudget: Int
    private let mirrors: [UUID: Mirror]
    private let surface: SurfaceShaderGraph
    private let blurPass: PlanarReflectionBlur?
    private var slots: [[UUID]: Slot] = [:]
    private var validatedGeometry: [UUID: MeshRenderer] = [:]
    private weak var activeCommand: (any MTLCommandBuffer)?

    /// Use one to four reflection levels. Each capture owns a single-sample
    /// RGBA16Float color and a Depth32Float texture. The set budgets up to 32
    /// captures and 256 MiB of logical capture and filter texel storage. Shallower
    /// views take priority in scene order when the budget is reached.
    public init(queue: any MTLCommandQueue, width: Int, height: Int, mirrors: [Mirror], maximumDepth: Int = 2) throws {
        guard (1...4_096).contains(width), (1...4_096).contains(height), (1...4).contains(maximumDepth),
              mirrors.count <= Self.maximumCaptures else { throw Failure.configuration }
        var authored: [UUID: Mirror] = [:]
        for mirror in mirrors {
            guard authored.updateValue(mirror, forKey: mirror.nodeID) == nil else {
                throw Failure.duplicateMirror(mirror.nodeID)
            }
        }
        self.queue = queue
        size = SIMD2(width, height)
        self.maximumDepth = maximumDepth
        let filterSettings = mirrors.compactMap { $0.appearance?.blur }.filter(\.needsPass)
        let scratchBytesPerPixel = filterSettings.contains { $0.radius != .zero } ? 16 : 0
        let outputBytesPerPixel = filterSettings.isEmpty ? 0 : 8 + (filterSettings.contains { $0.depthScale > 0 } ? 2 : 0)
        captureBudget = min(
            Self.maximumCaptures, (Self.maximumTargetBytes / (width * height) - scratchBytesPerPixel) / (12 + outputBytesPerPixel)
        )
        guard captureBudget > 0 else { throw Failure.configuration }
        self.mirrors = authored
        blurPass = try filterSettings.isEmpty ? nil : PlanarReflectionBlur(device: queue.device)
        surface = try SurfaceShaderGraph { graph in
            let reflection = graph.texture("Reflection", addressing: .clampToEdge)
            return .init(baseColor: reflection.samplePlanarReflection().rgb)
        }
    }

    /// Discover visible front-facing mirrors on the camera's layers. Inactive
    /// mirrors, leaves and views beyond the budget use the authored material.
    /// Use static planar meshes with unblended single-sample RGBA16Float pipelines.
    /// The explicit camera includes any projection crop or jitter. Optional
    /// shadows and transmission attachments must be distinct for each capture path.
    /// Finish the previous command buffer before reusing this set. Discard the
    /// supplied retained-reference command buffer if encoding throws.
    public func encode(
        _ scene: SceneGraph, camera: ShaderGraphCamera, depthMode: MeshDepthMode = .forward,
        background: EnvironmentBackground? = nil, backgroundOptions: EnvironmentBackground.Options = .init(),
        commandBuffer: any MTLCommandBuffer,
        attachmentsForView: (_ path: [UUID]) throws -> PlanarReflection.Capture.Attachments = { _ in .init() }
    ) throws -> Frame {
        guard commandBuffer.commandQueue === queue, commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else { throw Failure.commandBuffer }
        if let activeCommand, activeCommand.status != .completed, activeCommand.status != .error {
            throw Failure.inFlight
        }
        guard depthMode == .forward || depthMode == .reversed else { throw PlanarReflection.Failure.projection }
        let candidates = try candidates(in: scene, camera: camera, depthMode: depthMode)
        let (views, rootCount) = try plan(candidates, camera: camera, depthMode: depthMode)
        let sources = Dictionary(uniqueKeysWithValues: views.map { ($0.path, $0.candidate.mesh) })
        slots = slots.filter { path, slot in slot.source === sources[path] && slot.renderer.depthMode == depthMode }
        let prepared = try views.map { view in
            let source = view.candidate.mesh
            let slot = try slots[view.path] ?? makeSlot(candidate: view.candidate, depthMode: depthMode)
            slot.renderer.drawRange = source.drawRange
            if let uniforms = slot.projectionUniforms {
                let camera = try view.view.unclippedCamera()
                let projection = camera.projectionMatrix * camera.viewMatrix
                try uniforms.set(Self.normalDepthRow, to: Self.row(2, of: projection))
                try uniforms.set(Self.normalWRow, to: Self.row(3, of: projection))
            }
            return slot
        }
        let nextSlots = Dictionary(uniqueKeysWithValues: zip(views.map(\.path), prepared))
        slots = nextSlots

        func capture(_ index: Int) throws -> PlanarReflection.Capture {
            let view = views[index]
            let slot = prepared[index]
            var result = try PlanarReflection.Capture(
                view.candidate.reflection, color: slot.color, depth: slot.depth,
                children: view.children.map(capture), attachments: attachmentsForView(view.path)
            )
            result.filter = slot.filter
            return result
        }
        func graph(_ path: [UUID], colors: [UUID: any MTLTexture]) -> SceneGraph {
            var result = colors.isEmpty ? scene : scene.updatingNodes { node in
                guard colors[node.id] != nil, let slot = nextSlots[path + [node.id]] else { return }
                node.mesh = slot.renderer
            }
            result.projectionJitter = .zero
            return result
        }

        if rootCount > 0 { activeCommand = commandBuffer }
        let roots = try (0..<rootCount).map(capture)
        let rootColors = try PlanarReflection.encodeBounces(
            roots, camera: camera, depthMode: depthMode, background: background,
            backgroundOptions: backgroundOptions, commandBuffer: commandBuffer
        ) { graph($0, colors: $1) }
        return Frame(graph: graph([], colors: rootColors), captureCount: views.count)
    }

    private struct Candidate {
        let reflection: PlanarReflection
        let ancestors: Set<UUID>
        let mesh: MeshRenderer
        let worldBounds: AxisAlignedBox?
        let appearance: Appearance?
    }

    private func candidates(in graph: SceneGraph, camera: ShaderGraphCamera, depthMode: MeshDepthMode) throws -> [Candidate] {
        var geometry: [UUID: MeshRenderer] = [:]
        let candidates = try graph.resolve(cameraPosition: camera.pickingPosition()).compactMap { node -> Candidate? in
            guard let mirror = mirrors[node.id], graph.renders(node) else { return nil }
            let plane = mirror.localPlane
            if let override = graph.overrideMaterial, !override.excludedNodes.contains(node.id) { return nil }
            guard let mesh = node.mesh, node.batch == nil, node.levelOfDetail == nil, node.instances == nil,
                  node.primitive == nil, mesh.device === queue.device, mesh.supportsPlanarMirrorMaterial,
                  mesh.colorPixelFormats == [.rgba16Float], mesh.sampleCount == 1, mesh.depthMode == depthMode,
                  mesh.renderState.blending == .none else { throw Failure.unsupportedMirror(node.id) }
            guard mesh.hasActiveTriangles else { return nil }
            if validatedGeometry[node.id] !== mesh {
                let bounds = mesh.bounds
                let size = simd_reduce_max(SIMD3<Double>(bounds.maximum) - SIMD3<Double>(bounds.minimum))
                let tolerance = max(size * 1e-5, 1e-6)
                guard mesh.geometry.vertices.allSatisfy({ abs(plane.signedDistance(to: $0.position)) <= tolerance }) else {
                    throw Failure.nonplanarMirror(node.id)
                }
            }
            geometry[node.id] = mesh
            let reflection = PlanarReflection(plane: try Self.worldPlane(plane, transform: node.worldTransform), reflector: node.id)
            let ancestors = Set((graph.ancestors(of: node.id) ?? []).map(\.id).filter { mirrors[$0] != nil })
            let bounds = node.frustumCulled ? try (node.cullingBounds ?? mesh.bounds).transformed(by: node.worldTransform) : nil
            return Candidate(
                reflection: reflection, ancestors: ancestors, mesh: mesh, worldBounds: bounds, appearance: mirror.appearance
            )
        }
        validatedGeometry = geometry
        return candidates
    }

    private static func worldPlane(_ plane: Plane3D, transform: simd_float4x4) throws -> Plane3D {
        let world = simd_double4x4(
            SIMD4<Double>(transform[0]), SIMD4<Double>(transform[1]), SIMD4<Double>(transform[2]), SIMD4<Double>(transform[3])
        )
        let coefficients = simd_inverse(world).transpose * SIMD4(plane.unitNormal, plane.constant)
        let normal = SIMD3(coefficients.x, coefficients.y, coefficients.z)
        let length = simd_length(normal)
        return try Plane3D(normal: SIMD3<Float>(normal / length), constant: coefficients.w / length)
    }

    private struct PlannedView {
        let candidate: Candidate
        let path: [UUID]
        let view: PlanarReflection.View
        var children = 0..<0
    }

    private func plan(
        _ candidates: [Candidate], camera: ShaderGraphCamera, depthMode: MeshDepthMode
    ) throws -> ([PlannedView], rootCount: Int) {
        func visible(
            from camera: ShaderGraphCamera, clipFromWorld: simd_float4x4, path: [UUID], limit: Int
        ) throws -> [PlannedView] {
            let frustum = try ViewFrustum(clipFromWorld: clipFromWorld)
            let excluded = Set(path)
            var result: [PlannedView] = []
            for candidate in candidates {
                if result.count == limit { break }
                let id = candidate.reflection.reflector
                guard !excluded.contains(id), candidate.ancestors.isDisjoint(with: excluded),
                      candidate.worldBounds.map(frustum.intersects(box:)) ?? true,
                      let view = try candidate.reflection.reflectedView(camera, depthMode: depthMode) else { continue }
                result.append(PlannedView(candidate: candidate, path: path + [id], view: view))
            }
            return result
        }

        var plan = try visible(
            from: camera, clipFromWorld: camera.projectionMatrix * camera.viewMatrix, path: [], limit: captureBudget
        )
        let rootCount = plan.count
        var index = 0
        while index < plan.count, plan.count < captureBudget {
            let parent = plan[index]
            guard parent.path.count < maximumDepth else { break }
            let reflected = parent.view.camera
            let children = try visible(
                from: parent.view.unclippedCamera(), clipFromWorld: reflected.projectionMatrix * reflected.viewMatrix,
                path: parent.path, limit: captureBudget - plan.count
            )
            plan[index].children = plan.count..<(plan.count + children.count)
            plan.append(contentsOf: children)
            index += 1
        }
        return (plan, rootCount)
    }

    private struct Slot {
        let source: MeshRenderer
        let renderer: MeshRenderer
        let color: any MTLTexture
        let depth: any MTLTexture
        let filter: PlanarReflection.Capture.Filter?
        let projectionUniforms: ShaderUniforms?
    }

    private static let normalDepthRow = "ReflectorNormalDepthRow"
    private static let normalWRow = "ReflectorNormalWRow"

    private static func row(_ index: Int, of matrix: simd_float4x4) -> SIMD4<Float> {
        SIMD4(matrix[0][index], matrix[1][index], matrix[2][index], matrix[3][index])
    }

    private func makeSlot(candidate: Candidate, depthMode: MeshDepthMode) throws -> Slot {
        let source = candidate.mesh
        let id = candidate.reflection.reflector
        let appearance = candidate.appearance
        let label = "Frostfield automatic planar reflection"
        func texture(_ format: MTLPixelFormat, name: String, usage: MTLTextureUsage = [.renderTarget, .shaderRead]) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: size.x, height: size.y, mipmapped: false)
            descriptor.storageMode = .private
            descriptor.hazardTrackingMode = .tracked
            descriptor.usage = usage
            guard let texture = queue.device.makeTexture(descriptor: descriptor) else { throw Failure.allocation }
            texture.label = "\(label) \(name)"
            return texture
        }
        let color = try texture(.rgba16Float, name: "color")
        let usesDepth = (appearance?.blur.depthScale ?? 0) > 0
        let depth = try texture(.depth32Float, name: "depth", usage: usesDepth ? [.renderTarget, .shaderRead] : .renderTarget)
        let filter: PlanarReflection.Capture.Filter?
        if let appearance, appearance.blur.needsPass, let blurPass {
            filter = try .init(
                pass: blurPass, settings: appearance.blur, color: texture(.rgba16Float, name: "filtered"),
                depthFactor: usesDepth ? texture(.r16Float, name: "proximity") : nil
            )
        } else {
            filter = nil
        }
        let material: PhysicallyBasedMaterial
        let uniforms: ShaderUniforms?
        if let appearance {
            guard case .physicallyBased(let authored) = source.material,
                  authored.surfaceGraph == nil, authored.liveFactors == nil else { throw Failure.unsupportedAppearance(id) }
            (material, uniforms) = try Self.filteredMaterial(
                authored: authored, appearance: appearance, color: color, filter: filter, id: id, depthMode: depthMode
            )
        } else {
            material = try PhysicallyBasedMaterial(
                surfaceGraph: surface, surfaceTextures: ["Reflection": color], shadingModel: .unlit
            )
            uniforms = nil
        }
        let layout = SceneMaterialOverride.Layout(formats: [.rgba16Float], depth: depthMode, samples: 1, targets: nil)
        let renderer = try source.makeMaterialOverride(.physicallyBased(material), state: source.renderState, layout: layout)
        return Slot(source: source, renderer: renderer, color: color, depth: depth, filter: filter, projectionUniforms: uniforms)
    }

    private static func filteredMaterial(
        authored: PhysicallyBasedMaterial, appearance: Appearance, color: any MTLTexture,
        filter: PlanarReflection.Capture.Filter?, id: UUID, depthMode: MeshDepthMode
    ) throws -> (PhysicallyBasedMaterial, ShaderUniforms?) {
        let blurred = appearance.blur.radius == .zero ? nil : filter?.color
        let depthFactor = filter?.depthFactor
        let distortion = appearance.mix.distortion == 0 ? nil : appearance.distortionMap
        let normal = appearance.usesAuthoredNormalMap ? authored.sourceInputs.normalTexture : nil
        let sampling = normal?.sampling ?? .default
        if let normal {
            let linear = sampling.magnification == .linear
            guard normal.normalEncoding == .rgb, normal.normalSpace == .tangentSpace,
                  normal.sampler == nil || normal.sampling != nil,
                  sampling.addressU == sampling.addressV, sampling.magnification == sampling.minification,
                  normal.texture.mipmapLevelCount == 1 || sampling.mipFilter == (linear ? .linear : .nearest) else {
                throw Failure.unsupportedProjectionNormal(id)
            }
        }
        let surface = try SurfaceShaderGraph { graph in
            let sharp = graph.texture("Reflection", addressing: .clampToEdge)
            let soft = blurred.map { _ in graph.texture("ReflectionBlur", addressing: .clampToEdge) }
            let factor = depthFactor.map { _ in
                graph.texture("ReflectionDepth", content: .data, addressing: .clampToEdge, filtering: .nearest)
            }
            let distortionMap = distortion.map { _ in graph.texture("ReflectionDistortion", content: .data) }
            var normalMap: ShaderTexture?
            var normalCoordinates: ShaderValue<SIMD2<Float>>?
            var projectionDepth: ShaderValue<Float>?
            if let normal {
                normalMap = graph.texture(
                    "ReflectionNormal", content: .data, addressing: sampling.addressU,
                    filtering: sampling.magnification == .linear ? .linear : .nearest
                )
                let uv = normal.textureCoordinateSet == .secondary ? graph.uv1 : graph.uv0
                let (x, y) = normal.transform.rows
                let scale = authored.sourceInputs.normalScale
                normalCoordinates = graph.vector(
                    uv.x * x.x + uv.y * x.y + x.z, uv.x * y.x + uv.y * y.y + y.z
                ) * scale
                let position = graph.vector(graph.worldPosition, graph.constant(Float(1)))
                let depth = graph.uniform(normalDepthRow, SIMD4<Float>.zero).dot(position)
                let w = graph.uniform(normalWRow, SIMD4<Float>(0, 0, 0, 1)).dot(position)
                projectionDepth = depth / w.max(0.000001)
            }
            return .init(baseColor: graph.baseColor * appearance.mix.multiplier(
                sharp: sharp, blurred: soft, depthFactor: factor, distortionMap: distortionMap,
                normalMap: normalMap, normalCoordinates: normalCoordinates, normalProjectedDepth: projectionDepth, depthMode: depthMode
            ))
        }
        var textures = ["Reflection": color]
        textures["ReflectionBlur"] = blurred
        textures["ReflectionDepth"] = depthFactor
        textures["ReflectionDistortion"] = distortion
        textures["ReflectionNormal"] = normal?.texture
        let uniforms = try normal.map { _ in try ShaderUniforms(graph: surface) }
        let material = try authored.applying(.init(authored), surfaceOverride: (surface, uniforms, textures))
        return (material, uniforms)
    }

    public enum Failure: LocalizedError, Equatable {
        case configuration, duplicateMirror(UUID), unsupportedMirror(UUID), nonplanarMirror(UUID)
        case unsupportedAppearance(UUID), unsupportedProjectionNormal(UUID)
        case commandBuffer, inFlight, allocation

        public var errorDescription: String? {
            switch self {
            case .configuration: "Use up to 4096 texels per side, 32 mirrors and four reflection levels within the capture storage budget."
            case .duplicateMirror(let id): "The mirror node \(id) was registered twice."
            case .unsupportedMirror(let id): "Mirror \(id) needs a static unblended, single-sample RGBA16Float mesh."
            case .nonplanarMirror(let id): "The geometry of mirror \(id) must lie on its local reflection plane."
            case .unsupportedAppearance(let id): "Mirror \(id) needs an authored PBR material with fixed factors and no surface graph."
            case .unsupportedProjectionNormal(let id): "Mirror \(id) needs an RGB tangent-space normal map with matching U/V addressing and min/mag filtering, or disable normal-map projection."
            case .commandBuffer: "Use an uncommitted retaining command buffer from the reflector set's queue."
            case .inFlight: "Finish the previous reflector frame before reusing this set."
            case .allocation: "Metal could not allocate the planar reflector attachments."
            }
        }
    }
}
