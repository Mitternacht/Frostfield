import Foundation
import Metal

/// An application-authored pair of Metal raster functions and their resources.
/// Both functions may define their own varyings and use `[[vertex_id]]` for
/// indexed geometry.
public struct RawMeshProgram {
    public struct Buffer {
        public let resource: any MTLBuffer
        public let offset: Int

        public init(_ resource: any MTLBuffer, offset: Int = 0) {
            self.resource = resource
            self.offset = offset
        }
    }

    public struct Bindings {
        public let buffers: [Int: Buffer]
        public let textures: [Int: any MTLTexture]
        public let samplers: [Int: any MTLSamplerState]

        /// Use each argument's Metal index. Supplied resources must be read-only
        /// in the shader and remain valid for the duration of GPU execution.
        public init(
            buffers: [Int: Buffer] = [:], textures: [Int: any MTLTexture] = [:],
            samplers: [Int: any MTLSamplerState] = [:]
        ) {
            self.buffers = buffers
            self.textures = textures
            self.samplers = samplers
        }
    }

    public let vertex: any MTLFunction
    public let fragment: any MTLFunction
    public let vertexBindings: Bindings
    public let fragmentBindings: Bindings

    /// Vertex buffer 0 supplies four float4 values per vertex (position,
    /// normal, UV0/UV1 and tangent). Vertex buffers 1, 2 and 3 supply the
    /// object-to-clip, normal and object-to-world float4x4 matrices. Fragment
    /// buffer 0 supplies the per-draw float4 material parameters. Declare
    /// only the built-in bindings your functions use.
    public init(
        vertex: any MTLFunction, fragment: any MTLFunction,
        vertexBindings: Bindings = .init(), fragmentBindings: Bindings = .init()
    ) {
        self.vertex = vertex
        self.fragment = fragment
        self.vertexBindings = vertexBindings
        self.fragmentBindings = fragmentBindings
    }

    func pipeline(
        on device: any MTLDevice, descriptor: MTLRenderPipelineDescriptor
    ) throws -> any MTLRenderPipelineState {
        guard vertex.device === device, fragment.device === device else { throw MeshRenderer.Failure.wrongDevice }
        guard vertex.functionType == .vertex, fragment.functionType == .fragment else { throw Failure.functions }
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment

        var reflection: MTLRenderPipelineReflection?
        let pipeline: any MTLRenderPipelineState
        do {
            pipeline = try device.makeRenderPipelineState(
                descriptor: descriptor, options: [.bindingInfo, .bufferTypeInfo], reflection: &reflection
            )
        } catch {
            throw MeshRenderer.Failure.pipeline(error.localizedDescription)
        }
        guard let reflection else { throw Failure.reflection }
        try vertexBindings.validate(reflection.vertexBindings, stage: .vertex, device: device)
        try fragmentBindings.validate(reflection.fragmentBindings, stage: .fragment, device: device)
        return pipeline
    }

    func validateAttachments(_ pass: MTLRenderPassDescriptor, commandBuffer: any MTLCommandBuffer) throws {
        guard commandBuffer.retainedReferences else { throw Failure.commandBuffer }
        guard !vertexBindings.textures.isEmpty || !fragmentBindings.textures.isEmpty else { return }
        let attachments: [MTLRenderPassAttachmentDescriptor] =
            (0..<8).map { pass.colorAttachments[$0]! } + [pass.depthAttachment, pass.stencilAttachment]
        for attachment in attachments {
            for target in [attachment.texture, attachment.resolveTexture].compactMap({ $0 }) {
                guard !vertexBindings.aliases(target), !fragmentBindings.aliases(target) else {
                    throw Failure.feedback
                }
            }
        }
    }

    func bind(into encoder: any MTLRenderCommandEncoder) {
        for (index, value) in vertexBindings.buffers {
            encoder.setVertexBuffer(value.resource, offset: value.offset, index: index)
        }
        for (index, value) in vertexBindings.textures { encoder.setVertexTexture(value, index: index) }
        for (index, value) in vertexBindings.samplers { encoder.setVertexSamplerState(value, index: index) }
        for (index, value) in fragmentBindings.buffers {
            encoder.setFragmentBuffer(value.resource, offset: value.offset, index: index)
        }
        for (index, value) in fragmentBindings.textures { encoder.setFragmentTexture(value, index: index) }
        for (index, value) in fragmentBindings.samplers { encoder.setFragmentSamplerState(value, index: index) }
    }

    fileprivate enum Stage: String {
        case vertex, fragment

        var reservedBuffers: Range<Int> { self == .vertex ? 0..<4 : 0..<1 }

        func acceptsBuiltIn(_ binding: any MTLBufferBinding) -> Bool {
            if self == .vertex, binding.index == 0 {
                return binding.bufferDataSize <= MemoryLayout<PackedMeshVertex>.stride
                    && binding.bufferAlignment <= MemoryLayout<PackedMeshVertex>.alignment
            }
            let isMatrix = self == .vertex
            return binding.bufferDataType == (isMatrix ? .float4x4 : .float4)
                && binding.bufferDataSize == (isMatrix ? 64 : 16)
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case functions
        case reflection
        case bindings(stage: String, detail: String)
        case unsupportedGeometry
        case unsupportedAuxiliaryPass
        case commandBuffer
        case feedback

        public var errorDescription: String? {
            switch self {
                case .functions: "Supply a Metal vertex function and a Metal fragment function."
                case .reflection: "Metal did not provide shader binding reflection."
                case .bindings(let stage, let detail): "The raw \(stage) shader bindings are invalid. \(detail)"
                case .unsupportedGeometry: "Raw programs use ordinary indexed meshes without skinning, morphing, instancing or indirect command buffers."
                case .unsupportedAuxiliaryPass: "Raw programs do not declare the surface attributes needed for this pass."
                case .commandBuffer: "Use a Metal command buffer with retained references for raw shader draws."
                case .feedback: "Raw shader textures must not alias render attachments in the same pass."
            }
        }
    }
}

private extension RawMeshProgram.Bindings {
    func aliases(_ target: any MTLTexture) -> Bool {
        textures.values.contains { TextureDebugPass.aliases($0, target) }
    }

    static func containsIndirectResources(_ structure: MTLStructType?) -> Bool {
        structure?.members.contains { member in
            containsIndirectResources(member.dataType, structure: member.structType(), array: member.arrayType())
        } ?? false
    }

    static func containsIndirectResources(_ type: MTLDataType, structure: MTLStructType?, array: MTLArrayType?) -> Bool {
        switch type {
            case .texture, .sampler, .pointer, .renderPipeline, .computePipeline, .indirectCommandBuffer,
                 .visibleFunctionTable, .intersectionFunctionTable, .primitiveAccelerationStructure,
                 .instanceAccelerationStructure:
                true
            case .struct:
                containsIndirectResources(structure)
            case .array:
                array.map {
                    containsIndirectResources($0.elementType, structure: $0.elementStructType(), array: $0.element())
                } ?? false
            default:
                false
        }
    }

    static func matches(_ binding: any MTLTextureBinding, _ format: MTLPixelFormat) -> Bool {
        switch format {
            case .depth16Unorm, .depth32Float, .depth24Unorm_stencil8, .depth32Float_stencil8:
                binding.isDepthTexture
            case .r8Uint, .rg8Uint, .rgba8Uint, .r16Uint, .rg16Uint, .rgba16Uint, .r32Uint, .rg32Uint, .rgba32Uint,
                 .rgb10a2Uint, .stencil8, .x24_stencil8, .x32_stencil8:
                !binding.isDepthTexture && binding.textureDataType == .uint
            case .r8Sint, .rg8Sint, .rgba8Sint, .r16Sint, .rg16Sint, .rgba16Sint, .r32Sint, .rg32Sint, .rgba32Sint:
                !binding.isDepthTexture && binding.textureDataType == .int
            default:
                !binding.isDepthTexture && (binding.textureDataType == .float || binding.textureDataType == .half)
        }
    }

    func validate(_ reflection: [any MTLBinding], stage: RawMeshProgram.Stage, device: any MTLDevice) throws {
        func invalid(_ detail: String) -> RawMeshProgram.Failure {
            .bindings(stage: stage.rawValue, detail: detail)
        }

        guard buffers.keys.allSatisfy({ !stage.reservedBuffers.contains($0) }) else {
            throw invalid("Frostfield provides the reserved buffers.")
        }
        var usedBuffers: Set<Int> = []
        var usedTextures: Set<Int> = []
        var usedSamplers: Set<Int> = []
        for argument in reflection where argument.isUsed && argument.isArgument {
            guard argument.access == .readOnly else { throw invalid("Argument \(argument.name) must be read-only.") }
            let index = argument.index
            switch argument.type {
                case .buffer:
                    guard let layout = argument as? any MTLBufferBinding else { throw RawMeshProgram.Failure.reflection }
                    guard layout.bufferPointerType?.elementIsArgumentBuffer != true,
                          !Self.containsIndirectResources(layout.bufferStructType) else {
                        throw invalid("Buffer \(index) contains indirect shader resources.")
                    }
                    if stage.reservedBuffers.contains(index) {
                        guard stage.acceptsBuiltIn(layout) else { throw invalid("Buffer \(index) has an incompatible built-in layout.") }
                        continue
                    }
                    guard let buffer = buffers[index], buffer.resource.device === device,
                          buffer.offset >= 0, buffer.offset < buffer.resource.length,
                          buffer.offset.isMultiple(of: max(layout.bufferAlignment, 1)),
                          layout.bufferDataSize <= buffer.resource.length - buffer.offset else {
                        throw invalid("Supply a device-local buffer with enough aligned storage at index \(index).")
                    }
                    usedBuffers.insert(index)
                case .texture:
                    guard let layout = argument as? any MTLTextureBinding, layout.arrayLength <= 1,
                          let texture = textures[index], texture.device === device, texture.textureType == layout.textureType,
                          !texture.isFramebufferOnly, texture.storageMode != .memoryless,
                          texture.usage.isEmpty || texture.usage.contains(.shaderRead),
                          Self.matches(layout, texture.pixelFormat) else {
                        throw invalid("Supply one device-local shader-readable texture of the declared type at index \(index).")
                    }
                    usedTextures.insert(index)
                case .sampler:
                    guard let sampler = samplers[index], sampler.device === device else {
                        throw invalid("Supply a device-local sampler at index \(index).")
                    }
                    usedSamplers.insert(index)
                default:
                    throw invalid("Argument \(argument.name) uses an unsupported Metal resource type.")
            }
        }
        guard usedBuffers == Set(buffers.keys), usedTextures == Set(textures.keys), usedSamplers == Set(samplers.keys) else {
            throw invalid("Every supplied resource must match an active shader argument.")
        }
    }
}
