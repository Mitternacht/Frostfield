import Foundation
import Metal
import simd

/// Outlines selected scene nodes with separate colors for visible and hidden edges,
/// like Three.js `OutlinePass` with `visibleEdgeColor` and `hiddenEdgeColor`.
///
/// `encodeSelection` renders the depth of the selected nodes and their descendants
/// into a Depth32Float mask with the same vertex stages as the color pass, so skinned,
/// morphed, instanced, deformed and alpha-masked meshes outline their true
/// silhouettes. Alpha-blended and transmissive meshes outline their full geometry.
/// The outline pass compares that mask with the scene depth: a selected pixel is
/// visible where its depth is not behind the scene's by more than `depthBias`, else
/// hidden. The outline is a band `thickness` pixels wide outside the selection's
/// edge, drawn in the visible color, or in the hidden color when only hidden selected
/// pixels are near. A pixel whose center lies `d` pixels (Euclidean, from two
/// separable passes) from the nearest selected pixel's center is covered by
/// `saturate(thickness + 1 − d)`, the part of the band inside it, so a straight edge
/// receives exactly `thickness` pixels of coverage. Hidden selected pixels can be
/// tinted by `hiddenFill`. Colors are linear RGB with an opacity in alpha and are
/// composited over the scene.
public final class SelectionOutline: PostProcessEffect {
    public struct Settings: Sendable, Hashable {
        public var visibleColor: SIMD4<Float>
        public var hiddenColor: SIMD4<Float>
        /// Tint over selected pixels hidden behind other geometry; zero opacity disables it.
        public var hiddenFill: SIMD4<Float>
        /// Outline width in pixels beyond the selection's edge, from 0.5 to 16.
        public var thickness: Float
        /// Stored-depth tolerance for deciding that a selected pixel is visible.
        public var depthBias: Float
        /// Three.js OutlinePass pulse parameter. A positive value gives a cycle of
        /// `2π × pulsePeriod / 10` seconds. Zero disables the pulse.
        public var pulsePeriod: Float
        /// Time in seconds for this frame's pulse. Set it explicitly to animate.
        public var time: Double
        /// Gain of the blurred edge light, from 0 to 16. Zero skips the glow passes.
        public var edgeGlow: Float
        /// Glow's Gaussian standard deviation in output pixels, from 0.5 to 16.
        public var glowSigma: Float
        /// Pattern repetitions across the full frame in X and Y.
        public var patternRepeat: SIMD2<Float>
        /// Opacity of the repeating pattern. Hidden selection uses half this value.
        public var patternOpacity: Float

        public init(
            visibleColor: SIMD4<Float> = SIMD4(1, 0.55, 0.1, 1), hiddenColor: SIMD4<Float> = SIMD4(0.2, 0.35, 1, 0.8),
            hiddenFill: SIMD4<Float> = .zero, thickness: Float = 2, depthBias: Float = 1e-5,
            pulsePeriod: Float = 0, time: Double = 0, edgeGlow: Float = 0, glowSigma: Float = 4,
            patternRepeat: SIMD2<Float> = SIMD2(repeating: 6), patternOpacity: Float = 1
        ) {
            self.visibleColor = visibleColor
            self.hiddenColor = hiddenColor
            self.hiddenFill = hiddenFill
            self.thickness = thickness
            self.depthBias = depthBias
            self.pulsePeriod = pulsePeriod
            self.time = time
            self.edgeGlow = edgeGlow
            self.glowSigma = glowSigma
            self.patternRepeat = patternRepeat
            self.patternOpacity = patternOpacity
        }
    }

    public let device: any MTLDevice
    public var readsDepth: Bool { true }
    public var settings: Settings
    /// An optional tracked, shader-readable 2D texture on this device. Its
    /// inverse red channel forms a repeating white pattern inside the selection.
    public var patternTexture: (any MTLTexture)?
    /// The selected nodes' depth from the last `encodeSelection`, cleared where nothing is selected.
    public private(set) var selectionMask: (any MTLTexture)?
    /// The depth mode of the last `encodeSelection`.
    public private(set) var depthMode: MeshDepthMode = .forward
    private let shaders: PostProcessShaders
    private let pipelines: MeshDepthCapturePipelines
    private let forwardDepth: any MTLDepthStencilState
    private let reversedDepth: any MTLDepthStencilState
    private var rows: (any MTLTexture)?
    private var glowTargets: GlowTargets?

    public init(
        device: any MTLDevice, settings: Settings = Settings(), patternTexture: (any MTLTexture)? = nil
    ) throws {
        self.device = device
        self.settings = settings
        self.patternTexture = patternTexture
        shaders = try PostProcessShaders(device: device)
        pipelines = try MeshDepthCapturePipelines(device: device, label: "Frostfield selection depth")
        let state = MTLDepthStencilDescriptor()
        state.isDepthWriteEnabled = true
        state.depthCompareFunction = .less
        guard let forward = device.makeDepthStencilState(descriptor: state) else { throw PostProcessEffectError.allocation }
        state.depthCompareFunction = .greater
        guard let reversed = device.makeDepthStencilState(descriptor: state) else { throw PostProcessEffectError.allocation }
        forwardDepth = forward
        reversedDepth = reversed
    }

    /// The IDs of the selected nodes and every descendant in the graph.
    public static func expandedSelection(_ selection: Set<UUID>, in graph: SceneGraph) -> Set<UUID> {
        var result = Set<UUID>()
        var pending = graph.roots.map { ($0, false) }
        while let (node, inherited) = pending.popLast() {
            let selected = inherited || selection.contains(node.id)
            if selected { result.insert(node.id) }
            pending.append(contentsOf: node.children.map { ($0, selected) })
        }
        return result
    }

    /// Renders the selected nodes' depth for a `width` × `height` frame. Pass the same
    /// view and depth mode the scene used: `clipFromWorld` defaults to the graph
    /// camera's projection for this aspect ratio, mapping near and far to 0 and 1 for
    /// `.forward` depth or to 1 and 0 for `.reversed`. Encode this before the outline
    /// on the same or an earlier command buffer from one queue.
    /// Supply the same `indirectDraws` node arguments as the corresponding color pass.
    /// Returns the number of encoded draws. Zero leaves an empty mask.
    @discardableResult
    public func encodeSelection(
        graph: SceneGraph, selection: Set<UUID>, width: Int, height: Int, depthMode: MeshDepthMode = .forward,
        clipFromWorld: simd_float4x4? = nil, commandBuffer: any MTLCommandBuffer,
        indirectDraws: [UUID: MeshIndirectDraws] = [:]
    ) throws -> Int {
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw PostProcessEffectError.commandBuffer
        }
        guard commandBuffer.device === device else { throw PostProcessEffectError.wrongDevice }
        try depthMode.requireProjectionDepth()
        guard width > 0, height > 0 else { throw PostProcessEffectError.sizeMismatch }
        let view = try clipFromWorld ?? {
            try Self.projection(graph.camera, aspectRatio: Float(width) / Float(height), depthMode: depthMode)
                * simd_inverse(graph.cameraTransform.matrix)
        }()
        let mask: any MTLTexture
        if let selectionMask, selectionMask.width == width, selectionMask.height == height {
            mask = selectionMask
        } else {
            mask = try PostProcessShaders.scratch(
                device: device, width: width, height: height, pixelFormat: .depth32Float,
                usage: [.renderTarget, .shaderRead], label: "Frostfield selection depth"
            )
        }
        let included = Self.expandedSelection(selection, in: graph)
        let depthState = depthMode == .forward ? forwardDepth : reversedDepth
        let pipelines = pipelines
        let drawCount = try MeshRenderer.encodeDepthOnly(
            graph, clipFromWorld: view, commandBuffer: commandBuffer, includes: { included.contains($0) },
            indirectDraws: indirectDraws, makeEncoder: {
                let pass = MTLRenderPassDescriptor()
                pass.depthAttachment.texture = mask
                pass.depthAttachment.loadAction = .clear
                pass.depthAttachment.storeAction = .store
                pass.depthAttachment.clearDepth = depthMode == .forward ? 1 : 0
                guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
                    throw PostProcessEffectError.encoder
                }
                encoder.label = "Frostfield selection depth"
                encoder.setRenderPipelineState(pipelines.plain)
                encoder.setDepthStencilState(depthState)
                return encoder
            },
            setPipeline: pipelines.setPipeline
        )
        selectionMask = mask
        self.depthMode = depthMode
        return drawCount
    }

    /// The camera's projection with forward depth, or with near and far mapped to 1 and 0
    /// for reversed depth, computed in double precision as the glTF camera projection is.
    static func projection(_ camera: PerspectiveCamera, aspectRatio: Float, depthMode: MeshDepthMode) throws -> simd_float4x4 {
        var projection = try camera.projection(aspectRatio: aspectRatio)
        if depthMode == .reversed {
            let near = Double(camera.near)
            let far = Double(camera.far)
            projection.columns.2.z = Float(near / (far - near))
            projection.columns.3.z = Float(near * far / (far - near))
        }
        return projection
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render("frostfield_outline_composite", formats: [pixelFormat])
        _ = try shaders.compute("frostfield_outline_rows")
        if settings.edgeGlow > 0 || (settings.patternOpacity > 0 && patternTexture != nil) {
            _ = try shaders.render("frostfield_outline_styled_composite", formats: [pixelFormat])
        }
        if settings.edgeGlow > 0 {
            _ = try shaders.render("frostfield_outline_edges", formats: [.rgba16Float])
            _ = try shaders.compute("frostfield_blur_kernel")
            _ = try shaders.render("frostfield_blur_fragment", formats: [.rgba16Float])
        }
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        let s = settings
        let colors = [s.visibleColor, s.hiddenColor, s.hiddenFill]
        guard s.thickness.isFinite, (0.5...16).contains(s.thickness), s.depthBias.isFinite, s.depthBias >= 0,
              colors.allSatisfy({ $0.max().isFinite && $0.min() >= 0 && $0.w <= 1 }) else {
            throw PostProcessEffectError.invalidSettings(
                "Outline thickness must be 0.5 to 16 pixels, the depth bias nonnegative, and colors nonnegative with opacity up to 1."
            )
        }
        guard s.pulsePeriod.isFinite, s.pulsePeriod >= 0, s.time.isFinite,
              (0...16).contains(s.edgeGlow), (0.5...16).contains(s.glowSigma),
              s.patternRepeat.x.isFinite, s.patternRepeat.x > 0, s.patternRepeat.y.isFinite, s.patternRepeat.y > 0,
              (0...1).contains(s.patternOpacity) else {
            throw PostProcessEffectError.invalidSettings(
                "Use finite pulse time and nonnegative period, glow gain 0...16, sigma 0.5...16, positive repeat counts, and opacity 0...1."
            )
        }
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination, commandBuffer: commandBuffer,
            requiresDepth: true
        )
        if s.patternOpacity > 0, let patternTexture {
            guard patternTexture.device === device else { throw PostProcessEffectError.wrongDevice }
            guard patternTexture.textureType == .type2D, patternTexture.sampleCount == 1,
                  patternTexture.usage.contains(.shaderRead), patternTexture.storageMode != .memoryless,
                  patternTexture.hazardTrackingMode != .untracked, !patternTexture.isFramebufferOnly,
                  PostProcessPass.isFloatSampleable(patternTexture.pixelFormat, linear: true, device: device)
            else { throw PostProcessEffectError.unsupportedInput("Outline patterns need a tracked, float-sampleable 2D texture.") }
            guard !TextureTransfer.sharesStorage(patternTexture, destination) else { throw PostProcessEffectError.destination }
        }
        guard let mask = selectionMask, mask.width == source.width, mask.height == source.height else {
            throw PostProcessEffectError.missingSelection
        }
        if rows?.width != source.width || rows?.height != source.height {
            rows = try PostProcessShaders.scratch(
                device: device, width: source.width, height: source.height, label: "Frostfield outline rows"
            )
        }
        if s.edgeGlow > 0, glowTargets?.edges.width != source.width || glowTargets?.edges.height != source.height {
            glowTargets = try GlowTargets(device: device, width: source.width, height: source.height)
        }
    }

    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        guard let depth, let rows, let mask = selectionMask else { throw PostProcessEffectError.missingSelection }
        let rowKernel = try shaders.compute("frostfield_outline_rows")
        let s = settings
        let glow = s.edgeGlow > 0 ? glowTargets : nil
        let pattern = s.patternOpacity > 0 ? patternTexture : nil
        let styled = glow != nil || pattern != nil
        let composite = try shaders.render(
            styled ? "frostfield_outline_styled_composite" : "frostfield_outline_composite", formats: [destination.pixelFormat]
        )
        var pulse: Float = 1
        if s.pulsePeriod > 0 {
            let cycle = Double(s.pulsePeriod) * 2 * Double.pi / 10
            let phase = 2 * Double.pi * (s.time.truncatingRemainder(dividingBy: cycle) / cycle)
            pulse = Float(0.625 + 0.375 * cos(phase))
        }
        let pulseColor = SIMD4(pulse, pulse, pulse, 1)
        var parameters = Parameters(
            visibleColor: s.visibleColor * pulseColor, hiddenColor: s.hiddenColor * pulseColor, hiddenFill: s.hiddenFill,
            thickness: s.thickness, depthBias: s.depthBias, reversed: depthMode == .reversed ? 1 : 0,
            radius: Int32(s.thickness.rounded(.up))
        )
        try PostProcessShaders.compute(
            label: "Frostfield outline rows", commandBuffer: commandBuffer, pipeline: rowKernel,
            width: rows.width, height: rows.height
        ) { encoder in
            encoder.setTexture(mask, index: 0)
            encoder.setTexture(depth, index: 1)
            encoder.setTexture(rows, index: 2)
            encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 0)
        }
        if let glow {
            try encodeGlow(rows: rows, targets: glow, parameters: &parameters, sigma: s.glowSigma, commandBuffer: commandBuffer)
        }
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield selection outline", commandBuffer: commandBuffer, pipeline: composite
        ) { encoder in
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentTexture(rows, index: 1)
            encoder.setFragmentBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 0)
            if styled {
                encoder.setFragmentTexture(glow?.edges ?? rows, index: 2)
                encoder.setFragmentTexture(glow?.blurred ?? rows, index: 3)
                encoder.setFragmentTexture(pattern ?? rows, index: 4)
                var style = SIMD4(s.patternRepeat.x, s.patternRepeat.y, pattern == nil ? 0 : s.patternOpacity, s.edgeGlow)
                encoder.setFragmentBytes(&style, length: MemoryLayout<SIMD4<Float>>.stride, index: 1)
            }
        }
    }

    private func encodeGlow(
        rows: any MTLTexture, targets: GlowTargets, parameters: inout Parameters, sigma: Float,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        let pipeline = try shaders.render("frostfield_outline_edges", formats: [.rgba16Float])
        try PostProcessShaders.drawFullScreen(
            into: targets.edges, label: "Frostfield outline edges", commandBuffer: commandBuffer, pipeline: pipeline
        ) { encoder in
            encoder.setFragmentTexture(rows, index: 0)
            encoder.setFragmentBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 0)
        }
        let weights = try GaussianBlur.kernel(sigma: sigma)
        try SeparableBlur.encode(
            shaders: shaders, source: targets.edges, depth: nil, scratch: targets.scratch,
            destination: targets.blurred, weights: weights, parameters: { axis in
                SeparableBlur.Parameters(radius: Int32(weights.count / 2), axis: axis, depthSigma: 0, colorSigma: 0, depth: .zero)
            }, label: "Frostfield outline glow", commandBuffer: commandBuffer
        )
    }

    private struct GlowTargets {
        let edges: any MTLTexture
        let scratch: any MTLTexture
        let blurred: any MTLTexture

        init(device: any MTLDevice, width: Int, height: Int) throws {
            let renderUsage: MTLTextureUsage = [.renderTarget, .shaderRead]
            edges = try PostProcessShaders.scratch(
                device: device, width: width, height: height, usage: renderUsage, label: "Frostfield outline edges"
            )
            scratch = try PostProcessShaders.scratch(
                device: device, width: width, height: height, label: "Frostfield outline glow horizontal"
            )
            blurred = try PostProcessShaders.scratch(
                device: device, width: width, height: height, usage: renderUsage, label: "Frostfield outline glow"
            )
        }
    }

    private struct Parameters {
        var visibleColor: SIMD4<Float>
        var hiddenColor: SIMD4<Float>
        var hiddenFill: SIMD4<Float>
        var thickness: Float
        var depthBias: Float
        var reversed: UInt32
        var radius: Int32
    }
}

extension SelectionOutline: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}
