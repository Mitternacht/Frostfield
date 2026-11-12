import Foundation
import Metal
import Synchronization

/// Generates textures by evaluating a `TextureShaderGraph` at every texel on the GPU.
///
/// The compute pipeline is prepared once in the initializer. Generators of the
/// same graph on the same device share one cached pipeline, and uniform edits
/// never rebuild it. Supported formats:
///
/// - `.rgba8Unorm_srgb`: linear graph RGB encoded with the sRGB transfer function in
///   the shader and stored through a non-sRGB view of the same texture, so it never
///   depends on hardware sRGB shader writes. Sampling decodes it back to linear.
/// - `.rgba8Unorm`, `.r8Unorm`: linear data clamped to 0...1.
/// - `.rgba16Float`, `.r16Float`: clamped to ±65504.
/// - `.rgba32Float`, `.r32Float`: stored unchanged.
///
/// 8-bit values are rounded half away from zero, like `ProceduralSurface`.
/// Single-channel formats store the red component.
public final class ShaderTextureGenerator: Sendable {
    public typealias Failure = ShaderComputeFailure

    /// The largest width or height of a generated texture.
    public static let maximumDimension = 8_192
    /// The largest number of texels in a generated texture.
    public static let maximumTexelCount = 16_777_216
    public static let supportedFormats: Set<MTLPixelFormat> = [
        .rgba8Unorm_srgb, .rgba8Unorm, .rgba16Float, .rgba32Float, .r8Unorm, .r16Float, .r32Float,
    ]

    public let graph: TextureShaderGraph
    public let device: any MTLDevice
    /// The prepared pipeline. It stays the same for the generator's lifetime.
    public let pipelineState: any MTLComputePipelineState
    private let layout: ShaderComputeResources

    private static let kernelName = "frostfield_texture_graph_kernel"

    /// Compiles the graph, or reuses the pipeline already compiled for the same
    /// generated source on this device. Compiler errors name the generated line.
    public init(graph: TextureShaderGraph, device: any MTLDevice) throws {
        layout = ShaderComputeResources(uniformCount: graph.program.uniformSlotCount, textures: graph.textures)
        let function = try graph.metalFunction(named: "frostfield_texture_graph")
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        \(ShaderProgram.supportSource)

        \(layout.declarations)

        \(function)

        // dispatch.xy is the output size in texels; dispatch.z selects the storage conversion:
        // 0 unchanged, 1 sRGB-encoded 8-bit through a unorm view, 2 8-bit unorm, 3 16-bit float.
        static inline float4 frostfield_texture_store(float4 value, uint conversion) {
            switch (conversion) {
            case 1u: {
                float4 encoded = float4(frostfield_graph_linear_to_srgb(saturate(value.rgb)), saturate(value.a));
                return round(encoded * 255.0f) / 255.0f;
            }
            case 2u: return round(saturate(value) * 255.0f) / 255.0f;
            case 3u: return clamp(value, float4(-65504.0f), float4(65504.0f));
            default: return value;
            }
        }

        kernel void \(Self.kernelName)(
            texture2d<float, access::write> output [[texture(0)]],
            constant uint4& dispatch [[buffer(0)]]\(layout.kernelParameters(firstBuffer: 1, firstTexture: 1))
            uint2 texel [[thread_position_in_grid]]
        ) {
            if (texel.x >= dispatch.x || texel.y >= dispatch.y) {
                return;
            }
            FrostfieldComputeInput input = {};
            float2 size = float2(dispatch.xy);
            input.texelSize = 1.0f / size;
            input.gridUV = (float2(texel) + 0.5f) / size;
            \(layout.initializer)
            output.write(frostfield_texture_store(frostfield_texture_graph(input, resources), dispatch.z), texel);
        }
        """
        pipelineState = try ShaderComputeCache.pipeline(
            source: source, function: Self.kernelName, label: "Frostfield texture graph", device: device
        )
        self.graph = graph
        self.device = device
    }

    /// Allocates a shared-storage 2D texture, generates it on `queue`, and waits for
    /// completion, including the full mip chain when `mipmapped` is true. The
    /// returned texture is ready for use on any queue. Its usage is
    /// `[.shaderRead, .shaderWrite]`.
    ///
    /// - Parameters:
    ///   - uniforms: Values matching the graph's declarations, or nil for their defaults.
    ///   - textures: One texture per declared texture name; see `encode(into:commandBuffer:uniforms:textures:)`.
    public func makeTexture(
        width: Int, height: Int, format: MTLPixelFormat, mipmapped: Bool = false,
        uniforms: ShaderUniforms? = nil, textures: [String: any MTLTexture] = [:],
        queue: any MTLCommandQueue
    ) throws -> any MTLTexture {
        guard queue.device === device else { throw Failure.wrongDevice }
        try Self.validate(width: width, height: height)
        try validate(format: format, mipmapped: mipmapped)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: mipmapped
        )
        descriptor.storageMode = .shared
        // sRGB output is written through a linear view, which Metal validation requires to be declared.
        descriptor.usage = format == .rgba8Unorm_srgb ? [.shaderRead, .shaderWrite, .pixelFormatView] : [.shaderRead, .shaderWrite]
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw Failure.resource("a \(width) × \(height) generated texture")
        }
        texture.label = "Frostfield generated texture"
        guard let commandBuffer = queue.makeCommandBuffer() else { throw Failure.resource("a command buffer") }
        commandBuffer.label = "Frostfield texture graph"
        try encode(into: texture, commandBuffer: commandBuffer, uniforms: uniforms, textures: textures)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw Failure.execution(commandBuffer.error?.localizedDescription ?? "The command buffer did not complete.")
        }
        return texture
    }

    /// Encodes generation of base level 0 into `texture`, followed by a blit that
    /// regenerates every lower mip level when the texture has more than one.
    ///
    /// Nothing runs until the caller commits the command buffer; the texture is
    /// ready when that command buffer completes. Metal orders these writes against
    /// later passes in the same queue only for hazard-tracked textures; with
    /// untracked textures or heaps, the caller adds the fences or barriers.
    ///
    /// - Parameters:
    ///   - texture: A 2D, single-sample texture from the generator's device in a
    ///     supported format, with `.shaderWrite` usage and CPU- or GPU-only storage.
    ///     An `.rgba8Unorm_srgb` texture also needs `.pixelFormatView` usage, because the
    ///     encoded sRGB values are written through a linear view.
    ///   - commandBuffer: An uncommitted command buffer with retained references from the generator's device.
    ///   - uniforms: Values matching the graph's declarations, or nil for their defaults.
    ///   - textures: Exactly one texture per declared texture name: 2D, single-sample,
    ///     `.shaderRead`, float-sampled, from this device, and never `texture` itself or a
    ///     view of its storage. Data textures cannot use sRGB formats. Linear filtering
    ///     of 32-bit float textures requires `supports32BitFloatFiltering`.
    public func encode(
        into texture: any MTLTexture, commandBuffer: any MTLCommandBuffer,
        uniforms: ShaderUniforms? = nil, textures: [String: any MTLTexture] = [:]
    ) throws {
        try encode(into: texture, commandBuffer: commandBuffer, uniforms: uniforms, textures: textures, timing: nil)
    }

    /// Generates the same texture and reports GPU time for the graph dispatch.
    /// Mipmap generation follows in a separate blit. Read the timing after completion.
    public func encodeTimed(
        into texture: any MTLTexture, commandBuffer: any MTLCommandBuffer,
        uniforms: ShaderUniforms? = nil, textures: [String: any MTLTexture] = [:]
    ) throws -> ComputeGPUTiming {
        let timing = ComputeGPUTiming(commandBuffer: commandBuffer)
        try encode(into: texture, commandBuffer: commandBuffer, uniforms: uniforms, textures: textures, timing: timing)
        return timing
    }

    private func encode(
        into texture: any MTLTexture, commandBuffer: any MTLCommandBuffer,
        uniforms: ShaderUniforms?, textures: [String: any MTLTexture], timing: ComputeGPUTiming?
    ) throws {
        try ShaderComputeResources.validate(commandBuffer, device: device)
        guard texture.device === device else { throw Failure.wrongDevice }
        guard texture.textureType == .type2D, texture.sampleCount == 1, !texture.isFramebufferOnly,
              texture.usage.contains(.shaderWrite), texture.storageMode != .memoryless,
              texture.pixelFormat != .rgba8Unorm_srgb || texture.usage.contains(.pixelFormatView) else {
            throw Failure.outputTexture
        }
        try Self.validate(width: texture.width, height: texture.height)
        try validate(format: texture.pixelFormat, mipmapped: texture.mipmapLevelCount > 1)
        let values = try ShaderComputeResources.uniformValues(for: graph.uniforms, from: uniforms)
        let inputs = try ShaderComputeResources.textures(
            for: graph.textures, from: textures, device: device, output: texture
        )

        let target: any MTLTexture
        if texture.pixelFormat == .rgba8Unorm_srgb {
            // The Simulator's validation layer requires pixelFormatView usage even for an sRGB/linear view.
            guard let view = texture.makeTextureView(pixelFormat: .rgba8Unorm) else {
                throw Failure.resource("a linear view of the sRGB output texture")
            }
            target = view
        } else {
            target = texture
        }
        let conversion: UInt32 = switch texture.pixelFormat {
        case .rgba8Unorm_srgb: 1
        case .rgba8Unorm, .r8Unorm: 2
        case .rgba16Float, .r16Float: 3
        default: 0
        }

        guard let encoder = try timing?.makeEncoder(in: commandBuffer) ?? commandBuffer.makeComputeCommandEncoder() else {
            throw Failure.encoder("compute")
        }
        encoder.label = "Frostfield texture graph"
        encoder.setComputePipelineState(pipelineState)
        encoder.setTexture(target, index: 0)
        var dispatch = SIMD4<UInt32>(UInt32(texture.width), UInt32(texture.height), conversion, 0)
        encoder.setBytes(&dispatch, length: MemoryLayout<SIMD4<UInt32>>.size, index: 0)
        layout.bind(uniforms: values, textures: inputs, firstBuffer: 1, firstTexture: 1, encoder: encoder)
        let width = pipelineState.threadExecutionWidth
        let height = max(1, pipelineState.maxTotalThreadsPerThreadgroup / width)
        encoder.dispatchThreadgroups(
            MTLSize(width: (texture.width + width - 1) / width, height: (texture.height + height - 1) / height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1)
        )
        encoder.endEncoding()

        if texture.mipmapLevelCount > 1 {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw Failure.encoder("blit") }
            blit.label = "Frostfield texture graph mipmaps"
            blit.generateMipmaps(for: texture)
            blit.endEncoding()
        }
    }

    private static func validate(width: Int, height: Int) throws {
        guard (1...maximumDimension).contains(width), (1...maximumDimension).contains(height),
              width <= maximumTexelCount / height else {
            throw Failure.dimensions(width: width, height: height)
        }
    }

    private func validate(format: MTLPixelFormat, mipmapped: Bool) throws {
        guard Self.supportedFormats.contains(format) else { throw Failure.unsupportedFormat(format) }
        // Blit mip generation filters the texture, which 32-bit float formats support only on some GPUs.
        if mipmapped, format == .rgba32Float || format == .r32Float, !device.supports32BitFloatFiltering {
            throw Failure.mipmapFiltering(format)
        }
    }
}

/// Runs a `BufferShaderGraph` once per element, reading one buffer and writing another.
///
/// The compute pipeline is prepared once in the initializer and shared by kernels
/// of the same graph on the same device; uniform edits never rebuild it.
public final class ShaderBufferKernel: Sendable {
    public typealias Failure = ShaderComputeFailure

    /// The largest element count, which keeps `elementIndex` exact as a float.
    public static let maximumElementCount = 16_777_216

    public let graph: BufferShaderGraph
    public let device: any MTLDevice
    /// The prepared pipeline. It stays the same for the kernel's lifetime.
    public let pipelineState: any MTLComputePipelineState
    private let kernelSource: String
    private let compiledIndirect = Mutex<(any MTLComputePipelineState)?>(nil)
    private let layout: ShaderComputeResources

    private static let kernelName = "frostfield_buffer_graph_kernel"
    private static let indirectKernelName = "frostfield_buffer_graph_kernel_indirect"

    /// Threadgroup counts followed by the element count, in that order. Metal reads the first three fields.
    /// `threadgroupsY` and `threadgroupsZ` are 1 because the kernel uses the grid's x index as the element index.
    /// `threadgroupsX` counts groups of `indirectPipelineState.threadExecutionWidth` threads.
    public struct IndirectArguments: Sendable, Equatable {
        public var threadgroupsX: UInt32
        public var threadgroupsY: UInt32
        public var threadgroupsZ: UInt32
        public var elementCount: UInt32

        /// Covers `elementCount` elements with groups of `threadgroupWidth` threads.
        public init(elementCount: Int, threadgroupWidth: Int) throws {
            guard (0...ShaderBufferKernel.maximumElementCount).contains(elementCount), threadgroupWidth > 0 else {
                throw ShaderComputeFailure.indirectArguments
            }
            let width = UInt32(threadgroupWidth)
            let count = UInt32(elementCount)
            self.threadgroupsX = count == 0 ? 0 : (count + width - 1) / width
            self.threadgroupsY = 1
            self.threadgroupsZ = 1
            self.elementCount = count
        }
    }

    /// Compiles the graph, or reuses the pipeline already compiled for the same
    /// generated source on this device. Compiler errors name the generated line.
    public init(graph: BufferShaderGraph, device: any MTLDevice) throws {
        layout = ShaderComputeResources(uniformCount: graph.program.uniformSlotCount, textures: graph.textures)
        let function = try graph.metalFunction(named: "frostfield_buffer_graph")
        let fields = graph.fieldCount
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        \(ShaderProgram.supportSource)

        \(layout.declarations)

        \(function)

        // Each element is \(fields) consecutive float4 fields. dispatch.x is the element count.
        kernel void \(Self.kernelName)(
            device const float4* source [[buffer(0)]],
            device float4* destination [[buffer(1)]],
            constant uint4& dispatch [[buffer(2)]]\(layout.kernelParameters(firstBuffer: 3, firstTexture: 0))
            uint element [[thread_position_in_grid]]
        ) {
            if (element >= dispatch.x) {
                return;
            }
            FrostfieldComputeInput input = {};
            input.elementIndex = float(element);
            input.elementCount = float(dispatch.x);
            uint base = element * \(fields)u;
            for (uint field = 0; field < \(fields)u; ++field) {
                input.fields[field] = source[base + field];
            }
            \(layout.initializer)
            float4 fields[4] = { input.fields[0], input.fields[1], input.fields[2], input.fields[3] };
            frostfield_buffer_graph(input, resources, fields);
            for (uint field = 0; field < \(fields)u; ++field) {
                destination[base + field] = fields[field];
            }
        }
        """
        pipelineState = try ShaderComputeCache.pipeline(
            source: source, function: Self.kernelName, label: "Frostfield buffer graph", device: device
        )
        kernelSource = source
        self.graph = graph
        self.device = device
    }

    /// The pipeline used by an indirect dispatch. Its threadgroup width is the width the argument buffer must cover.
    public func indirectPipeline() throws -> any MTLComputePipelineState {
        if let existing = compiledIndirect.withLock({ $0 }) { return existing }
        let fields = graph.fieldCount
        let source = kernelSource + """

        kernel void \(Self.indirectKernelName)(
            device const float4* source [[buffer(0)]],
            device float4* destination [[buffer(1)]],
            constant uint4& dispatch [[buffer(2)]],
            constant uint& capacity [[buffer(3)]]\(layout.kernelParameters(firstBuffer: 4, firstTexture: 0))
            uint element [[thread_position_in_grid]]
        ) {
            uint count = min(dispatch.w, capacity);
            if (element >= count) { return; }
            FrostfieldComputeInput input = {};
            input.elementIndex = float(element);
            input.elementCount = float(count);
            uint base = element * \(fields)u;
            for (uint field = 0; field < \(fields)u; ++field) {
                input.fields[field] = source[base + field];
            }
            \(layout.initializer)
            float4 fields[4] = { input.fields[0], input.fields[1], input.fields[2], input.fields[3] };
            frostfield_buffer_graph(input, resources, fields);
            for (uint field = 0; field < \(fields)u; ++field) {
                destination[base + field] = fields[field];
            }
        }
        """
        let pipeline = try ShaderComputeCache.pipeline(
            source: source, function: Self.indirectKernelName, label: "Frostfield indirect buffer graph", device: device
        )
        return compiledIndirect.withLock { existing in
            if let existing { return existing }
            existing = pipeline
            return pipeline
        }
    }

    /// Encodes one update of `elementCount` elements from `source` into `destination`.
    ///
    /// Every element's fields are written to the destination: connected fields get
    /// the graph's value, the rest are copied. The kernel never writes `source`, so
    /// ping-pong two buffers across steps. Metal orders one step's writes before the
    /// next step's reads in the same queue only for hazard-tracked buffers; with
    /// untracked buffers or heaps, the caller adds the fences or barriers. Do not
    /// read or write the destination range on the CPU until the command buffer completes.
    ///
    /// - Parameters:
    ///   - elementCount: One through `maximumElementCount` elements.
    ///   - source: A buffer from this device holding `elementCount` × `graph.elementStride`
    ///     bytes after `sourceOffset`.
    ///   - destination: A different buffer from this device with the same capacity after `destinationOffset`.
    ///   - sourceOffset, destinationOffset: Nonnegative multiples of 16 bytes.
    ///   - uniforms: Values matching the graph's declarations, or nil for their defaults.
    ///   - textures: Exactly one 2D, `.shaderRead`, float-sampled texture from this
    ///     device per declared texture name.
    ///   - commandBuffer: An uncommitted command buffer with retained references from this device.
    public func encode(
        elementCount: Int,
        from source: any MTLBuffer, sourceOffset: Int = 0,
        to destination: any MTLBuffer, destinationOffset: Int = 0,
        uniforms: ShaderUniforms? = nil, textures: [String: any MTLTexture] = [:],
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try encode(
            elementCount: elementCount, from: source, sourceOffset: sourceOffset, to: destination,
            destinationOffset: destinationOffset, uniforms: uniforms, textures: textures, commandBuffer: commandBuffer, timing: nil
        )
    }

    /// Updates the same buffer range and reports GPU time for the graph dispatch.
    /// Read the timing after the caller's command buffer completes.
    public func encodeTimed(
        elementCount: Int,
        from source: any MTLBuffer, sourceOffset: Int = 0,
        to destination: any MTLBuffer, destinationOffset: Int = 0,
        uniforms: ShaderUniforms? = nil, textures: [String: any MTLTexture] = [:],
        commandBuffer: any MTLCommandBuffer
    ) throws -> ComputeGPUTiming {
        let timing = ComputeGPUTiming(commandBuffer: commandBuffer)
        try encode(
            elementCount: elementCount, from: source, sourceOffset: sourceOffset, to: destination,
            destinationOffset: destinationOffset, uniforms: uniforms, textures: textures, commandBuffer: commandBuffer, timing: timing
        )
        return timing
    }

    /// Dispatches the graph with threadgroup counts stored in `arguments`.
    ///
    /// The buffer holds one ``IndirectArguments`` value at `argumentsOffset`: three threadgroup
    /// counts, then the element count. A previous encoder in this command buffer may write that
    /// record. The kernel writes only elements that fit in both buffers, and reads the count from
    /// the buffer rather than from the CPU. `threadgroupsY` and `threadgroupsZ` should be 1.
    public func encode(
        indirect arguments: any MTLBuffer, argumentsOffset: Int = 0,
        from source: any MTLBuffer, sourceOffset: Int = 0,
        to destination: any MTLBuffer, destinationOffset: Int = 0,
        uniforms: ShaderUniforms? = nil, textures: [String: any MTLTexture] = [:],
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try ShaderComputeResources.validate(commandBuffer, device: device)
        guard source.device === device, destination.device === device, arguments.device === device else {
            throw Failure.wrongDevice
        }
        guard source !== destination, source !== arguments, destination !== arguments else { throw Failure.bufferAliasing }
        guard arguments.storageMode != .memoryless else { throw Failure.indirectArguments }
        guard argumentsOffset >= 0, argumentsOffset.isMultiple(of: 16),
              argumentsOffset <= arguments.length - MemoryLayout<IndirectArguments>.stride else {
            throw Failure.indirectArguments
        }
        for (role, buffer, offset) in [("source", source, sourceOffset), ("destination", destination, destinationOffset)] {
            guard offset >= 0, offset.isMultiple(of: 16), offset <= buffer.length else {
                throw offset >= 0 && offset.isMultiple(of: 16)
                    ? Failure.bufferTooSmall(role, required: offset, available: buffer.length)
                    : Failure.bufferOffset(offset)
            }
        }
        let values = try ShaderComputeResources.uniformValues(for: graph.uniforms, from: uniforms)
        let inputs = try ShaderComputeResources.textures(for: graph.textures, from: textures, device: device, output: nil)
        let sourceRoom = (source.length - sourceOffset) / graph.elementStride
        let destinationRoom = (destination.length - destinationOffset) / graph.elementStride
        var capacity = UInt32(min(sourceRoom, destinationRoom, Self.maximumElementCount))
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw Failure.encoder("compute") }
        encoder.label = "Frostfield indirect buffer graph"
        let pipeline = try indirectPipeline()
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: sourceOffset, index: 0)
        encoder.setBuffer(destination, offset: destinationOffset, index: 1)
        encoder.setBuffer(arguments, offset: argumentsOffset, index: 2)
        encoder.setBytes(&capacity, length: MemoryLayout<UInt32>.stride, index: 3)
        layout.bind(uniforms: values, textures: inputs, firstBuffer: 4, firstTexture: 0, encoder: encoder)
        let width = pipeline.threadExecutionWidth
        encoder.dispatchThreadgroups(
            indirectBuffer: arguments, indirectBufferOffset: argumentsOffset,
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    private func encode(
        elementCount: Int, from source: any MTLBuffer, sourceOffset: Int,
        to destination: any MTLBuffer, destinationOffset: Int,
        uniforms: ShaderUniforms?, textures: [String: any MTLTexture],
        commandBuffer: any MTLCommandBuffer, timing: ComputeGPUTiming?
    ) throws {
        try ShaderComputeResources.validate(commandBuffer, device: device)
        guard (1...Self.maximumElementCount).contains(elementCount) else { throw Failure.elementCount(elementCount) }
        guard source.device === device, destination.device === device else { throw Failure.wrongDevice }
        guard source !== destination else { throw Failure.bufferAliasing }
        let length = elementCount * graph.elementStride
        for (role, buffer, offset) in [("source", source, sourceOffset), ("destination", destination, destinationOffset)] {
            guard offset >= 0, offset.isMultiple(of: 16) else { throw Failure.bufferOffset(offset) }
            let (end, overflow) = offset.addingReportingOverflow(length)
            guard !overflow, end <= buffer.length else {
                throw Failure.bufferTooSmall(role, required: overflow ? .max : end, available: buffer.length)
            }
        }
        let values = try ShaderComputeResources.uniformValues(for: graph.uniforms, from: uniforms)
        let inputs = try ShaderComputeResources.textures(for: graph.textures, from: textures, device: device, output: nil)

        guard let encoder = try timing?.makeEncoder(in: commandBuffer) ?? commandBuffer.makeComputeCommandEncoder() else {
            throw Failure.encoder("compute")
        }
        encoder.label = "Frostfield buffer graph"
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(source, offset: sourceOffset, index: 0)
        encoder.setBuffer(destination, offset: destinationOffset, index: 1)
        var dispatch = SIMD4<UInt32>(UInt32(elementCount), 0, 0, 0)
        encoder.setBytes(&dispatch, length: MemoryLayout<SIMD4<UInt32>>.size, index: 2)
        layout.bind(uniforms: values, textures: inputs, firstBuffer: 3, firstTexture: 0, encoder: encoder)
        let width = pipelineState.threadExecutionWidth
        encoder.dispatchThreadgroups(
            MTLSize(width: (elementCount + width - 1) / width, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }
}

/// Errors from preparing or encoding compute graphs.
public enum ShaderComputeFailure: LocalizedError, Equatable {
    case compilation(String)
    case pipeline(String)
    case wrongDevice
    case commandBuffer
    case dimensions(width: Int, height: Int)
    case unsupportedFormat(MTLPixelFormat)
    case mipmapFiltering(MTLPixelFormat)
    case outputTexture
    case missingTexture(String)
    case unknownTexture(String)
    case invalidTexture(String)
    case textureAliasesOutput(String)
    case uniforms
    case elementCount(Int)
    case bufferAliasing
    case bufferOffset(Int)
    case bufferTooSmall(String, required: Int, available: Int)
    case indirectArguments
    case encoder(String)
    case resource(String)
    case execution(String)

    public var errorDescription: String? {
        switch self {
        case .compilation(let detail): "Could not compile the compute graph. \(detail)"
        case .pipeline(let detail): "Could not create the compute graph pipeline: \(detail)"
        case .wrongDevice: "Use a queue, command buffer, texture and buffers from the compute graph's device."
        case .commandBuffer: "Use an uncommitted command buffer with retained references."
        case .dimensions(let width, let height):
            "Generated textures are 1–\(ShaderTextureGenerator.maximumDimension) texels per side and at most "
                + "\(ShaderTextureGenerator.maximumTexelCount) texels, not \(width) × \(height)."
        case .unsupportedFormat(let format):
            "Pixel format \(format.rawValue) is unsupported. Use rgba8Unorm_srgb, rgba8Unorm, rgba16Float, "
                + "rgba32Float, r8Unorm, r16Float or r32Float."
        case .mipmapFiltering(let format):
            "This GPU cannot filter pixel format \(format.rawValue) to generate mipmaps. Use a 16-bit float format or no mipmaps."
        case .outputTexture:
            "The output must be a 2D single-sample texture with shaderWrite usage, CPU- or GPU-only storage, and pixelFormatView usage when it is sRGB."
        case .missingTexture(let name): "Bind a texture for the graph's declared texture \"\(name)\"."
        case .unknownTexture(let name): "The graph does not declare a texture named \"\(name)\"."
        case .invalidTexture(let name):
            "Texture \"\(name)\" must be a single-sample, shaderRead, float-sampled texture from the graph's device, "
                + "and a 2D image, cube map, or 3D texture as declared; "
                + "data textures cannot use sRGB formats, and linear filtering of 32-bit float textures needs GPU support."
        case .textureAliasesOutput(let name):
            "Texture \"\(name)\" shares storage with the output texture. Read from a separate texture."
        case .uniforms: "The uniform values must match the graph's uniform declarations in order and type."
        case .elementCount(let count):
            "A buffer graph dispatch covers 1–\(ShaderBufferKernel.maximumElementCount) elements, not \(count)."
        case .bufferAliasing: "Use separate source and destination buffers, and swap them between steps."
        case .bufferOffset(let offset): "Buffer offset \(offset) must be a nonnegative multiple of 16 bytes."
        case .bufferTooSmall(let role, let required, let available):
            "The \(role) buffer needs \(required) bytes from its offset but has \(available)."
        case .indirectArguments:
            "Indirect dispatch arguments need 16 bytes at a 16-byte offset in a tracked buffer from this device."
        case .encoder(let kind): "Metal could not create a compute graph \(kind) encoder."
        case .resource(let name): "Metal could not allocate \(name)."
        case .execution(let detail): "The GPU could not finish the compute graph. \(detail)"
        }
    }
}

/// Resource declarations, kernel parameters, and bindings shared by generated
/// compute kernels. The struct holds the uniform pointer and one sampled texture per
/// declared slot; hooks exist only for declared resources.
struct ShaderComputeResources: Sendable {
    let uniformCount: Int
    let textureCount: Int
    let shapes: [ShaderTextureInput.Shape]

    init(uniformCount: Int, textures: [ShaderTextureInput]) {
        self.uniformCount = uniformCount
        self.textureCount = textures.count
        self.shapes = textures.map(\.shape)
    }

    private static func sampledType(_ shape: ShaderTextureInput.Shape) -> String {
        switch shape {
        case .twoDimensional: "texture2d<float>"
        case .cube: "texturecube<float>"
        case .texture3D: "texture3d<float>"
        }
    }

    /// `FrostfieldGraphResources` and the `frostfield_graph_uniform` and `frostfield_graph_sample` hooks.
    var declarations: String {
        var members: [String] = []
        if uniformCount > 0 { members.append("    constant float4* uniforms;") }
        for (slot, shape) in shapes.enumerated() {
            members.append("    \(Self.sampledType(shape)) texture\(slot);")
        }
        var source = members.isEmpty
            ? "struct FrostfieldGraphResources {};"
            : "struct FrostfieldGraphResources {\n\(members.joined(separator: "\n"))\n};"
        if uniformCount > 0 {
            source += """


            static inline float4 frostfield_graph_uniform(thread const FrostfieldGraphResources& resources, uint index) {
                return resources.uniforms[index];
            }
            """
        }
        if textureCount > 0 {
            let flatSlots = shapes.enumerated().compactMap { $0.element == .twoDimensional ? $0.offset : nil }
            let cases = flatSlots.map {
                "    case \($0)u: return frostfield_compute_sample(resources.texture\($0), address, filter, uv, level);"
            }.joined(separator: "\n")
            let gradientCases = flatSlots.map {
                "    case \($0)u: return frostfield_compute_sample(resources.texture\($0), address, filter, uv, frostfield_compute_gradient_lod(resources.texture\($0), ddx, ddy));"
            }.joined(separator: "\n")
            source += """


            // address: 0 repeat, 1 clamp, 2 mirror; filter: 0 linear, 1 nearest.
            // A nonfinite lod reads level 0. The lod is clamped to the last mip.
            static inline float frostfield_compute_lod(texture2d<float> texture, float level) {
                float last = float(texture.get_num_mip_levels() - 1);
                return isfinite(level) ? clamp(level, 0.0f, last) : 0.0f;
            }
            // log2 of the larger UV footprint in texels. A nonfinite derivative reads level 0.
            static inline float frostfield_compute_gradient_lod(texture2d<float> texture, float2 ddx, float2 ddy) {
                if (!all(isfinite(ddx)) || !all(isfinite(ddy))) return 0.0f;
                float2 size = float2(texture.get_width(), texture.get_height());
                float rho = max(length(ddx * size), length(ddy * size));
                return rho > 0.0f && isfinite(rho) ? log2(rho) : 0.0f;
            }
            static inline float4 frostfield_compute_sample(
                texture2d<float> texture, uint address, uint filter, float2 uv, float lod
            ) {
                constexpr sampler repeatLinear(address::repeat, filter::linear, mip_filter::nearest);
                constexpr sampler repeatNearest(address::repeat, filter::nearest, mip_filter::nearest);
                constexpr sampler clampLinear(address::clamp_to_edge, filter::linear, mip_filter::nearest);
                constexpr sampler clampNearest(address::clamp_to_edge, filter::nearest, mip_filter::nearest);
                constexpr sampler mirrorLinear(address::mirrored_repeat, filter::linear, mip_filter::nearest);
                constexpr sampler mirrorNearest(address::mirrored_repeat, filter::nearest, mip_filter::nearest);
                float2 coordinates = select(float2(0), uv, frostfield_graph_finite(uv));
                float mip = frostfield_compute_lod(texture, lod);
                switch (address * 2u + filter) {
                case 0u: return texture.sample(repeatLinear, coordinates, level(mip));
                case 1u: return texture.sample(repeatNearest, coordinates, level(mip));
                case 2u: return texture.sample(clampLinear, coordinates, level(mip));
                case 3u: return texture.sample(clampNearest, coordinates, level(mip));
                case 4u: return texture.sample(mirrorLinear, coordinates, level(mip));
                default: return texture.sample(mirrorNearest, coordinates, level(mip));
                }
            }

            static inline float4 frostfield_graph_sample(
                thread const FrostfieldGraphResources& resources, uint slot, uint address, uint filter, float2 uv, float level
            ) {
                switch (slot) {
            \(cases)
                default: return float4(0);
                }
            }

            static inline float4 frostfield_graph_sample(
                thread const FrostfieldGraphResources& resources, uint slot, uint address, uint filter, float2 uv
            ) {
                return frostfield_graph_sample(resources, slot, address, filter, uv, 0.0f);
            }

            static inline float4 frostfield_graph_sample(
                thread const FrostfieldGraphResources& resources, uint slot, uint address, uint filter,
                float2 uv, float2 ddx, float2 ddy
            ) {
                switch (slot) {
            \(gradientCases)
                default: return float4(0);
                }
            }
            """
            if shapes.contains(where: { $0 != .twoDimensional }) {
                let spatialCases = shapes.enumerated().compactMap { slot, shape -> String? in
                    let function: String
                    switch shape {
                    case .twoDimensional: return nil
                    case .cube: function = "frostfield_compute_cube"
                    case .texture3D: function = "frostfield_compute_volume"
                    }
                    return "    case \(slot)u: return \(function)(resources.texture\(slot), address, filter, coordinates);"
                }.joined(separator: "\n")
                if shapes.contains(.cube) {
                    source += """


            // A nonfinite or zero direction reads the +X face. Compute reads mip 0.
            static inline float4 frostfield_compute_cube(
                texturecube<float> texture, uint address, uint filter, float3 direction
            ) {
                constexpr sampler repeatLinear(address::repeat, filter::linear, mip_filter::nearest);
                constexpr sampler repeatNearest(address::repeat, filter::nearest, mip_filter::nearest);
                constexpr sampler clampLinear(address::clamp_to_edge, filter::linear, mip_filter::nearest);
                constexpr sampler clampNearest(address::clamp_to_edge, filter::nearest, mip_filter::nearest);
                constexpr sampler mirrorLinear(address::mirrored_repeat, filter::linear, mip_filter::nearest);
                constexpr sampler mirrorNearest(address::mirrored_repeat, filter::nearest, mip_filter::nearest);
                float3 coordinates = all(isfinite(direction)) && length(direction) > 0.0f ? direction : float3(1, 0, 0);
                switch (address * 2u + filter) {
                case 0u: return texture.sample(repeatLinear, coordinates, level(0.0f));
                case 1u: return texture.sample(repeatNearest, coordinates, level(0.0f));
                case 2u: return texture.sample(clampLinear, coordinates, level(0.0f));
                case 3u: return texture.sample(clampNearest, coordinates, level(0.0f));
                case 4u: return texture.sample(mirrorLinear, coordinates, level(0.0f));
                default: return texture.sample(mirrorNearest, coordinates, level(0.0f));
                }
            }
            """
                }
                if shapes.contains(.texture3D) {
                    source += """


            // A nonfinite coordinate reads zero. Compute reads mip 0.
            static inline float4 frostfield_compute_volume(
                texture3d<float> texture, uint address, uint filter, float3 uv
            ) {
                constexpr sampler repeatLinear(address::repeat, filter::linear, mip_filter::nearest);
                constexpr sampler repeatNearest(address::repeat, filter::nearest, mip_filter::nearest);
                constexpr sampler clampLinear(address::clamp_to_edge, filter::linear, mip_filter::nearest);
                constexpr sampler clampNearest(address::clamp_to_edge, filter::nearest, mip_filter::nearest);
                constexpr sampler mirrorLinear(address::mirrored_repeat, filter::linear, mip_filter::nearest);
                constexpr sampler mirrorNearest(address::mirrored_repeat, filter::nearest, mip_filter::nearest);
                float3 coordinates = all(isfinite(uv)) ? uv : float3(0);
                switch (address * 2u + filter) {
                case 0u: return texture.sample(repeatLinear, coordinates, level(0.0f));
                case 1u: return texture.sample(repeatNearest, coordinates, level(0.0f));
                case 2u: return texture.sample(clampLinear, coordinates, level(0.0f));
                case 3u: return texture.sample(clampNearest, coordinates, level(0.0f));
                case 4u: return texture.sample(mirrorLinear, coordinates, level(0.0f));
                default: return texture.sample(mirrorNearest, coordinates, level(0.0f));
                }
            }
            """
                }
                source += """


            static inline float4 frostfield_graph_sample(
                thread const FrostfieldGraphResources& resources, uint slot, uint address, uint filter, float3 coordinates
            ) {
                switch (slot) {
            \(spatialCases)
                default: return float4(0);
                }
            }
            """
            }
        }
        return source
    }

    /// Kernel parameters after the fixed ones, each preceded by a comma and ending with one.
    func kernelParameters(firstBuffer: Int, firstTexture: Int) -> String {
        var parameters: [String] = []
        if uniformCount > 0 { parameters.append("constant float4* graphUniforms [[buffer(\(firstBuffer))]]") }
        for (slot, shape) in shapes.enumerated() {
            parameters.append("\(Self.sampledType(shape)) graphTexture\(slot) [[texture(\(firstTexture + slot))]]")
        }
        return parameters.map { ",\n    \($0)" }.joined() + ","
    }

    /// Declares `resources` inside the kernel.
    var initializer: String {
        var members: [String] = []
        if uniformCount > 0 { members.append("graphUniforms") }
        for slot in 0..<textureCount { members.append("graphTexture\(slot)") }
        return "FrostfieldGraphResources resources = { \(members.joined(separator: ", ")) };"
    }

    func bind(
        uniforms: [SIMD4<Float>], textures: [any MTLTexture], firstBuffer: Int, firstTexture: Int,
        encoder: any MTLComputeCommandEncoder
    ) {
        if uniformCount > 0 {
            uniforms.withUnsafeBytes { bytes in
                if let address = bytes.baseAddress {
                    encoder.setBytes(address, length: bytes.count, index: firstBuffer)
                }
            }
        }
        for (slot, texture) in textures.enumerated() {
            encoder.setTexture(texture, index: firstTexture + slot)
        }
    }

    static func validate(_ commandBuffer: any MTLCommandBuffer, device: any MTLDevice) throws {
        guard commandBuffer.device === device else { throw ShaderComputeFailure.wrongDevice }
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw ShaderComputeFailure.commandBuffer
        }
    }

    /// The current values, or the declared defaults without a uniforms object.
    /// Every value slot followed by the graph clock, which nil uniforms leave at zero.
    static func uniformValues(for declarations: [ShaderUniform], from uniforms: ShaderUniforms?) throws -> [SIMD4<Float>] {
        guard let uniforms else { return declarations.flatMap(\.columns) + [.zero] }
        guard uniforms.matches(declarations) else { throw ShaderComputeFailure.uniforms }
        return uniforms.bufferSnapshot().values
    }

    /// Orders the bound textures by declared slot after checking each one.
    static func textures(
        for declarations: [ShaderTextureInput], from provided: [String: any MTLTexture],
        device: any MTLDevice, output: (any MTLTexture)?
    ) throws -> [any MTLTexture] {
        if let unknown = provided.keys.sorted().first(where: { name in !declarations.contains { $0.name == name } }) {
            throw ShaderComputeFailure.unknownTexture(unknown)
        }
        return try declarations.map { declaration in
            guard let texture = provided[declaration.name] else {
                throw ShaderComputeFailure.missingTexture(declaration.name)
            }
            let format = texture.pixelFormat
            let isSRGB = format == .rgba8Unorm_srgb || format == .bgra8Unorm_srgb
            let isFloat32 = format == .r32Float || format == .rg32Float || format == .rgba32Float
            let expected: MTLTextureType = switch declaration.shape {
            case .twoDimensional: .type2D
            case .cube: .typeCube
            case .texture3D: .type3D
            }
            guard texture.device === device, texture.textureType == expected, texture.sampleCount == 1,
                  texture.usage.isEmpty || texture.usage.contains(.shaderRead),
                  sampledFormats.contains(format),
                  !(isSRGB && declaration.content == .data),
                  !(isFloat32 && declaration.filtering == .linear && !device.supports32BitFloatFiltering) else {
                throw ShaderComputeFailure.invalidTexture(declaration.name)
            }
            if let output, storage(of: texture) === storage(of: output) {
                throw ShaderComputeFailure.textureAliasesOutput(declaration.name)
            }
            return texture
        }
    }

    private static let sampledFormats: Set<MTLPixelFormat> = [
        .r8Unorm, .rg8Unorm, .rgba8Unorm, .rgba8Unorm_srgb, .bgra8Unorm, .bgra8Unorm_srgb,
        .r16Unorm, .rg16Unorm, .rgba16Unorm, .r16Float, .rg16Float, .rgba16Float,
        .r32Float, .rg32Float, .rgba32Float, .rgb10a2Unorm, .rg11b10Float, .rgb9e5Float,
    ]

    /// The texture that owns the storage behind a view chain.
    private static func storage(of texture: any MTLTexture) -> any MTLTexture {
        var current = texture
        while let parent = current.parent { current = parent }
        return current
    }
}

/// Compute pipelines keyed by device and generated source, so equal graphs compile
/// once per device. Bounded so editing sessions that generate many graphs do not grow without limit.
enum ShaderComputeCache {
    private struct Key: Hashable {
        let device: UInt64
        let source: String
        let function: String
    }

    private struct State {
        var pipelines: [Key: any MTLComputePipelineState] = [:]
        var order: [Key] = []
    }

    static let capacity = 64
    private static let state = Mutex(State())

    static func pipeline(
        source: String, function name: String, label: String, device: any MTLDevice
    ) throws -> any MTLComputePipelineState {
        let key = Key(device: device.registryID, source: source, function: name)
        if let cached = state.withLock({ $0.pipelines[key] }), cached.device === device { return cached }

        let library: any MTLLibrary
        do {
            library = try ShaderLibraryCache.library(source: source, device: device)
        } catch {
            throw ShaderComputeFailure.compilation(ShaderProgram.diagnose(error, source: source))
        }
        guard let function = library.makeFunction(name: name) else {
            throw ShaderComputeFailure.pipeline("The generated library does not contain \(name).")
        }
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.label = label
        descriptor.computeFunction = function
        let pipeline: any MTLComputePipelineState
        do {
            (pipeline, _) = try device.makeComputePipelineState(descriptor: descriptor, options: [])
        } catch {
            throw ShaderComputeFailure.pipeline(error.localizedDescription)
        }
        return state.withLock { state in
            if let existing = state.pipelines[key], existing.device === device { return existing }
            if state.pipelines.updateValue(pipeline, forKey: key) == nil {
                state.order.append(key)
            }
            while state.order.count > capacity {
                state.pipelines[state.order.removeFirst()] = nil
            }
            return pipeline
        }
    }
}
