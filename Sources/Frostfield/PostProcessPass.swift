import Foundation
import Metal

/// One prepared full-screen render pipeline for a `PostProcessGraph`.
///
/// Preparation compiles the graph once per device and output format. Passes on
/// one device whose graphs generate identical Metal share a compiled library.
/// Uniform values are copied from `uniforms` into every encode, so editing them
/// never rebuilds the pipeline.
///
/// Bindings: fragment texture 0 is the source, 1 the depth when the graph reads
/// it, and 2...9 the declared graph textures in order. Fragment buffer 0 holds one
/// float4 per declared uniform.
public final class PostProcessPass {
    /// Linear RGBA16Float for further processing, or an 8-bit sRGB format for
    /// final display, where Metal encodes the linear result when it stores it.
    public static let supportedPixelFormats: [MTLPixelFormat] = [.rgba16Float, .bgra8Unorm_srgb, .rgba8Unorm_srgb]

    public let graph: PostProcessGraph
    public let device: any MTLDevice
    public let pixelFormat: MTLPixelFormat
    /// Live values for the graph's uniforms, copied into each encode.
    public let uniforms: ShaderUniforms
    /// The compiled graph. Passes on this device with identical generated source share it.
    public let library: any MTLLibrary
    public let pipelineState: any MTLRenderPipelineState
    /// The bound graph textures in declaration order.
    private let textures: [any MTLTexture]
    private let denoiseNormalSlots: [Int]
    private let readsDenoise: Bool

    /// Compiles the graph, or reuses its cached library, and prepares the pipeline.
    ///
    /// Pass `uniforms` to share live values with other passes or materials; they
    /// must declare the graph's uniforms in the same order, names and types. Nil
    /// creates values at the graph's defaults. `textures` binds every declared graph
    /// texture by name to a 2D, shader-readable, float-sampleable texture on `device`.
    /// 32-bit float textures with linear filtering need a device that filters them.
    /// Synchronize writes to untracked graph textures yourself.
    public init(
        graph: PostProcessGraph,
        device: any MTLDevice,
        pixelFormat: MTLPixelFormat = .rgba16Float,
        uniforms: ShaderUniforms? = nil,
        textures: [String: any MTLTexture] = [:]
    ) throws {
        guard Self.supportedPixelFormats.contains(pixelFormat) else { throw Failure.pixelFormat }
        if let uniforms {
            guard uniforms.matches(graph.uniforms) else { throw Failure.uniformDeclarations }
            self.uniforms = uniforms
        } else {
            self.uniforms = try ShaderUniforms(graph: graph)
        }
        if let unknown = textures.keys.sorted().first(where: { name in !graph.textures.contains { $0.name == name } }) {
            throw Failure.unknownTexture(unknown)
        }
        self.textures = try graph.textures.map { declaration in
            guard let texture = textures[declaration.name] else { throw Failure.missingTexture(declaration.name) }
            guard texture.device === device, texture.textureType == .type2D,
                  texture.usage.contains(.shaderRead), !texture.isFramebufferOnly,
                  texture.storageMode != .memoryless,
                  Self.isFloatSampleable(texture.pixelFormat, linear: declaration.filtering == .linear, device: device)
            else { throw Failure.incompatibleTexture(declaration.name) }
            return texture
        }

        denoiseNormalSlots = graph.denoiseNormalSlots
        readsDenoise = graph.readsDenoise
        for slot in denoiseNormalSlots {
            let texture = self.textures[slot]
            guard (texture.pixelFormat == .rgba16Float || texture.pixelFormat == .rgba32Float),
                  texture.sampleCount == 1, texture.hazardTrackingMode != .untracked else {
                throw Failure.incompatibleTexture(graph.textures[slot].name)
            }
        }
        self.graph = graph
        self.device = device
        self.pixelFormat = pixelFormat
        let source = try Self.metalSource(for: graph)
        do {
            library = try ShaderLibraryCache.library(source: source, device: device)
        } catch {
            throw Failure.compilation(ShaderProgram.diagnose(error, source: source))
        }
        do {
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.label = "Frostfield post-process graph"
            descriptor.vertexFunction = library.makeFunction(name: "frostfield_post_vertex")
            descriptor.fragmentFunction = library.makeFunction(name: "frostfield_post_fragment")
            descriptor.colorAttachments[0].pixelFormat = pixelFormat
            pipelineState = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
    }

    /// Encodes one full-screen draw on the caller's command buffer without submitting it.
    ///
    /// The source is a shader-readable, single-sample, hazard-tracked RGBA16Float
    /// texture of premultiplied linear color. `depth` is required when
    /// `graph.readsDepth` is true and otherwise optional; it is shader-readable,
    /// hazard-tracked Depth32Float. The destination is a separate render target with
    /// this pass's pixel format. All three share one size. The current uniform values
    /// are copied into the command buffer. Use a command buffer that retains its
    /// references. Nothing is encoded when validation throws.
    public func encode(
        source: any MTLTexture,
        depth: (any MTLTexture)? = nil,
        destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        let pass = Self.descriptor(for: destination)
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.encoder }
        draw(source: source, depth: depth, encoder: encoder)
        encoder.endEncoding()
    }

    /// Encodes the usual draw and records its vertex and fragment GPU intervals.
    /// Submit the command buffer and read the timing after it completes.
    public func encodeTimed(
        source: any MTLTexture,
        depth: (any MTLTexture)? = nil,
        destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws -> RenderGPUTiming {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        return try RenderGPUTiming.encode(pass: Self.descriptor(for: destination), commandBuffer: commandBuffer) { encoder in
            draw(source: source, depth: depth, encoder: encoder)
        }
    }

    private static func descriptor(for destination: any MTLTexture) -> MTLRenderPassDescriptor {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = destination
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        return pass
    }

    private func draw(source: any MTLTexture, depth: (any MTLTexture)?, encoder: any MTLRenderCommandEncoder) {
        encoder.label = "Frostfield post-process graph"
        encoder.setRenderPipelineState(pipelineState)
        encoder.setFragmentTexture(source, index: 0)
        if graph.readsDepth { encoder.setFragmentTexture(depth, index: 1) }
        for (slot, texture) in textures.enumerated() {
            encoder.setFragmentTexture(texture, index: 2 + slot)
        }
        if graph.program.uniformSlotCount > 0 {
            let values = uniforms.bufferSnapshot().values
            values.withUnsafeBytes { bytes in
                if let base = bytes.baseAddress { encoder.setFragmentBytes(base, length: bytes.count, index: 0) }
            }
        }
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    }

    /// Checks everything `encode` requires without encoding.
    func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard commandBuffer.device === device, source.device === device, destination.device === device,
              depth.map({ $0.device === device }) ?? true else { throw Failure.wrongDevice }
        guard Self.isReadable(source, format: .rgba16Float), source.width > 0, source.height > 0 else {
            throw Failure.source
        }
        guard destination !== source, !textures.contains(where: { $0 === destination }),
              destination.pixelFormat == pixelFormat, destination.textureType == .type2D,
              destination.sampleCount == 1, destination.usage.contains(.renderTarget),
              destination.storageMode != .memoryless else { throw Failure.destination }
        for slot in denoiseNormalSlots {
            let normals = textures[slot]
            guard normals.width == source.width, normals.height == source.height else { throw Failure.sizeMismatch }
            guard !TextureTransfer.sharesStorage(normals, destination) else { throw Failure.destination }
        }
        if readsDenoise, TextureTransfer.sharesStorage(source, destination) { throw Failure.destination }
        if let depth {
            guard Self.isReadable(depth, format: .depth32Float) else { throw Failure.depth }
        } else if graph.readsDepth {
            throw Failure.missingDepth
        }
        guard destination.width == source.width, destination.height == source.height,
              depth.map({ $0.width == source.width && $0.height == source.height }) ?? true else {
            throw Failure.sizeMismatch
        }
    }

    static func isReadable(_ texture: any MTLTexture, format: MTLPixelFormat) -> Bool {
        texture.pixelFormat == format && texture.textureType == .type2D && texture.sampleCount == 1
            && texture.usage.contains(.shaderRead) && !texture.isFramebufferOnly
            && texture.storageMode != .memoryless && texture.hazardTrackingMode != .untracked
    }

    static func isFloatSampleable(_ format: MTLPixelFormat, linear: Bool, device: any MTLDevice) -> Bool {
        switch format {
        case .invalid, .depth16Unorm, .depth32Float, .stencil8, .depth32Float_stencil8, .x32_stencil8,
             .r8Uint, .r8Sint, .r16Uint, .r16Sint, .rg8Uint, .rg8Sint, .r32Uint, .r32Sint, .rg16Uint, .rg16Sint,
             .rgba8Uint, .rgba8Sint, .rgb10a2Uint, .rg32Uint, .rg32Sint, .rgba16Uint, .rgba16Sint,
             .rgba32Uint, .rgba32Sint:
            return false
        case .r32Float, .rg32Float, .rgba32Float:
            return !linear || device.supports32BitFloatFiltering
        default:
            #if os(macOS)
            if format == .depth24Unorm_stencil8 || format == .x24_stencil8 { return false }
            #endif
            return true
        }
    }

    // MARK: - Generated source

    private static let functionName = "frostfield_post_graph"

    /// The complete library for a graph: support declarations, resources, the hooks
    /// the graph uses, the generated function, and the full-screen entry points.
    static func metalSource(for graph: PostProcessGraph) throws -> String {
        let function = try graph.metalFunction(named: functionName)
        let textureSlots = graph.textures.indices
        let hasUniforms = graph.program.uniformSlotCount > 0

        var members = ["texture2d<float> scene;"]
        var parameters = ["float4 position [[position]]", "texture2d<float> scene [[texture(0)]]"]
        var arguments = ["scene"]
        var hooks: [String] = []
        if graph.readsScene { hooks.append(sceneHook) }
        if graph.readsDepth {
            members.append("depth2d<float> depth;")
            parameters.append("depth2d<float> depth [[texture(1)]]")
            arguments.append("depth")
            hooks.append(depthHook)
            if graph.readsDepthNormal { hooks.append(PostDepthGraphSource.normalHook) }
        }
        if hasUniforms {
            members.append("constant float4* uniforms;")
            parameters.append("constant float4* uniforms [[buffer(0)]]")
            arguments.append("uniforms")
            hooks.append(uniformHook)
        }
        for slot in textureSlots {
            members.append("texture2d<float> texture\(slot);")
            parameters.append("texture2d<float> texture\(slot) [[texture(\(2 + slot))]]")
            arguments.append("texture\(slot)")
        }
        if !textureSlots.isEmpty {
            let cases = textureSlots.map { "        case \($0)u: return frostfield_post_sample(r.texture\($0), address, filter, uv);" }
            let levelCases = textureSlots.map {
                "        case \($0)u: return frostfield_post_sample(r.texture\($0), address, filter, uv, level);"
            }
            let gradientCases = textureSlots.map {
                "        case \($0)u: return frostfield_post_sample(r.texture\($0), address, filter, uv, ddx, ddy);"
            }
            hooks.append(sampleHelper)
            hooks.append("""
            static inline float4 frostfield_graph_sample(
                thread const FrostfieldGraphResources& r, uint slot, uint address, uint filter, float2 uv
            ) {
                switch (slot) {
            \(cases.joined(separator: "\n"))
                    default: return float4(0);
                }
            }
            static inline float4 frostfield_graph_sample(
                thread const FrostfieldGraphResources& r, uint slot, uint address, uint filter, float2 uv, float level
            ) {
                switch (slot) {
            \(levelCases.joined(separator: "\n"))
                    default: return float4(0);
                }
            }
            static inline float4 frostfield_graph_sample(
                thread const FrostfieldGraphResources& r, uint slot, uint address, uint filter,
                float2 uv, float2 ddx, float2 ddy
            ) {
                switch (slot) {
            \(gradientCases.joined(separator: "\n"))
                    default: return float4(0);
                }
            }
            """)
        }

        if graph.readsDenoise { hooks.append(AODenoiseSource.hook(normalSlots: graph.denoiseNormalSlots)) }

        return """
        #include <metal_stdlib>
        using namespace metal;
        // The shared support source defines helpers that a given graph may not call.
        #pragma clang diagnostic ignored "-Wunused-function"

        \(PostProcessGraph.supportSource)

        struct FrostfieldGraphResources {
        \(members.map { "    " + $0 }.joined(separator: "\n"))
        };

        \(hooks.joined(separator: "\n\n"))

        \(function)

        vertex float4 frostfield_post_vertex(uint vertexID [[vertex_id]]) {
            const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
            return float4(positions[vertexID], 0, 1);
        }

        fragment float4 frostfield_post_fragment(
        \(parameters.map { "    " + $0 }.joined(separator: ",\n"))
        ) {
            float2 size = float2(scene.get_width(), scene.get_height());
            FrostfieldPostInput input;
            input.screenUV = position.xy / size;
            input.texelSize = 1.0f / size;
            input.viewportSize = size;
            input.screenCoordinate = position.xy;
            input.screenSize = size;
            FrostfieldGraphResources resources = { \(arguments.joined(separator: ", ")) };
            return \(functionName)(input, resources);
        }
        """
    }

    private static let sceneHook = """
    static inline float4 frostfield_graph_scene(thread const FrostfieldGraphResources& r, float2 uv) {
        constexpr sampler bilinear(coord::normalized, address::clamp_to_edge, filter::linear);
        return r.scene.sample(bilinear, uv, level(0));
    }

    static inline float4 frostfield_graph_scene(thread const FrostfieldGraphResources& r, float2 uv, float lod) {
        constexpr sampler trilinear(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::linear);
        lod = isfinite(lod) ? clamp(lod, 0.0f, float(r.scene.get_num_mip_levels() - 1)) : 0.0f;
        return r.scene.sample(trilinear, uv, level(lod));
    }
    """

    private static let depthHook = """
    static inline float frostfield_graph_depth(thread const FrostfieldGraphResources& r, float2 uv) {
        constexpr sampler nearest(coord::normalized, address::clamp_to_edge, filter::nearest);
        return r.depth.sample(nearest, uv, level(0));
    }
    """

    private static let uniformHook = """
    static inline float4 frostfield_graph_uniform(thread const FrostfieldGraphResources& r, uint index) {
        return r.uniforms[index];
    }
    """

    // Address codes: 0 repeat, 1 clamp to edge, 2 mirrored repeat. Filter codes: 0 linear, 1 nearest.
    private static let sampleHelper = """
    static inline float4 frostfield_post_sample(texture2d<float> t, uint address, uint filter, float2 uv) {
        constexpr sampler repeatLinear(coord::normalized, address::repeat, filter::linear, mip_filter::linear);
        constexpr sampler repeatNearest(coord::normalized, address::repeat, filter::nearest, mip_filter::nearest);
        constexpr sampler clampLinear(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::linear);
        constexpr sampler clampNearest(coord::normalized, address::clamp_to_edge, filter::nearest, mip_filter::nearest);
        constexpr sampler mirrorLinear(coord::normalized, address::mirrored_repeat, filter::linear, mip_filter::linear);
        constexpr sampler mirrorNearest(coord::normalized, address::mirrored_repeat, filter::nearest, mip_filter::nearest);
        switch (address * 2u + filter) {
            case 0u: return t.sample(repeatLinear, uv);
            case 1u: return t.sample(repeatNearest, uv);
            case 2u: return t.sample(clampLinear, uv);
            case 3u: return t.sample(clampNearest, uv);
            case 4u: return t.sample(mirrorLinear, uv);
            default: return t.sample(mirrorNearest, uv);
        }
    }
    static inline float frostfield_post_lod(texture2d<float> t, float level) {
        float last = float(t.get_num_mip_levels() - 1);
        return isfinite(level) ? clamp(level, 0.0f, last) : 0.0f;
    }
    static inline float4 frostfield_post_sample(texture2d<float> t, uint address, uint filter, float2 uv, float requested) {
        constexpr sampler repeatLinear(coord::normalized, address::repeat, filter::linear, mip_filter::nearest);
        constexpr sampler repeatNearest(coord::normalized, address::repeat, filter::nearest, mip_filter::nearest);
        constexpr sampler clampLinear(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::nearest);
        constexpr sampler clampNearest(coord::normalized, address::clamp_to_edge, filter::nearest, mip_filter::nearest);
        constexpr sampler mirrorLinear(coord::normalized, address::mirrored_repeat, filter::linear, mip_filter::nearest);
        constexpr sampler mirrorNearest(coord::normalized, address::mirrored_repeat, filter::nearest, mip_filter::nearest);
        float mip = frostfield_post_lod(t, requested);
        switch (address * 2u + filter) {
            case 0u: return t.sample(repeatLinear, uv, level(mip));
            case 1u: return t.sample(repeatNearest, uv, level(mip));
            case 2u: return t.sample(clampLinear, uv, level(mip));
            case 3u: return t.sample(clampNearest, uv, level(mip));
            case 4u: return t.sample(mirrorLinear, uv, level(mip));
            default: return t.sample(mirrorNearest, uv, level(mip));
        }
    }
    static inline float frostfield_post_gradient_lod(texture2d<float> t, float2 ddx, float2 ddy) {
        if (!all(isfinite(ddx)) || !all(isfinite(ddy))) return 0.0f;
        float2 size = float2(t.get_width(), t.get_height());
        float rho = max(length(ddx * size), length(ddy * size));
        return rho > 0.0f && isfinite(rho) ? log2(rho) : 0.0f;
    }
    static inline float4 frostfield_post_sample(
        texture2d<float> t, uint address, uint filter, float2 uv, float2 ddx, float2 ddy
    ) {
        return frostfield_post_sample(t, address, filter, uv, frostfield_post_gradient_lod(t, ddx, ddy));
    }
    """

    public enum Failure: LocalizedError, Equatable {
        case pixelFormat
        case uniformDeclarations
        case unknownTexture(String)
        case missingTexture(String)
        case incompatibleTexture(String)
        case compilation(String)
        case pipeline(String)
        case commandBuffer
        case wrongDevice
        case source
        case destination
        case depth
        case missingDepth
        case sizeMismatch
        case encoder

        public var errorDescription: String? {
            switch self {
            case .pixelFormat: "A post-process pass renders to RGBA16Float, BGRA8Unorm_sRGB or RGBA8Unorm_sRGB."
            case .uniformDeclarations:
                "The shader uniforms must declare the graph's uniforms in the same order, with the same names and types."
            case .unknownTexture(let name): "The post-process graph does not declare a texture named \"\(name)\"."
            case .missingTexture(let name): "Bind a texture for the post-process graph's declared texture \"\(name)\"."
            case .incompatibleTexture(let name):
                "Graph texture \"\(name)\" must be a 2D, shader-readable, float-sampleable texture on the pass's device."
            case .compilation(let detail): "Could not compile the post-process graph. \(detail)"
            case .pipeline(let detail): "Could not prepare the post-process pipeline. \(detail)"
            case .commandBuffer: "Post-processing needs an uncommitted command buffer that retains its references."
            case .wrongDevice: "Post-process textures and the command buffer must use the pass's Metal device."
            case .source: "The post-process source must be a shader-readable, tracked, single-sample 2D RGBA16Float texture."
            case .destination:
                "The post-process destination must be a separate single-sample 2D render target in the pass's pixel format."
            case .depth: "Post-process depth must be a shader-readable, tracked, single-sample 2D Depth32Float texture."
            case .missingDepth: "This post-process graph samples scene depth. Pass the scene's Depth32Float texture."
            case .sizeMismatch: "The post-process source, depth and destination must have the same dimensions."
            case .encoder: "Metal could not create the post-process encoder."
            }
        }
    }
}
