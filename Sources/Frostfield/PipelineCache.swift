import Metal
internal import FrostfieldShaderTypes

/// A fragment function that passed the material checks, with the render
/// pipeline that pairs it with Frostfield's mesh vertex stage.
final class MaterialProgram {
    let functionName: String
    let pipeline: any MTLRenderPipelineState
    /// The size of the parameter block the function declares at
    /// `FF_MATERIAL_PARAMETERS_BUFFER_INDEX`, or `nil` when it declares none.
    let parameterBlockLength: Int?
    /// The cache that prepared this program; a renderer draws only programs
    /// from its own cache.
    private(set) weak var cache: MaterialPipelineCache?

    init(functionName: String, pipeline: any MTLRenderPipelineState, parameterBlockLength: Int?, cache: MaterialPipelineCache) {
        self.functionName = functionName
        self.pipeline = pipeline
        self.parameterBlockLength = parameterBlockLength
        self.cache = cache
    }
}

/// Prepares and keeps the render pipelines of one ``MeshRenderer``'s
/// materials.
///
/// Every pipeline pairs `ff_mesh_vertex` and the mesh vertex descriptor with
/// one fragment function, for the renderer's color and depth formats. A
/// pipeline is created the first time a fragment function is prepared and
/// reused for every later material that names the same function in the same
/// library object, so materials and parameter values can be added without
/// compiling anything again.
///
/// Before a pipeline is cached, its fragment bindings are checked against the
/// material contract through Metal's pipeline reflection: the function may
/// declare only a read-only parameter block at
/// `FF_MATERIAL_PARAMETERS_BUFFER_INDEX`. The cache keeps each library it has
/// seen alive, so a library's identity cannot be reused by another object.
///
/// Owned by one renderer and used from that renderer's thread or actor only.
final class MaterialPipelineCache {
    /// Parameter blocks are copied with `setFragmentBytes`, which Metal
    /// documents for data smaller than 4 KB.
    static let parameterLengthLimit = 4096

    private struct Key: Hashable {
        let library: ObjectIdentifier
        let functionName: String
    }

    private let device: any MTLDevice
    private let vertexFunction: any MTLFunction
    private let vertexDescriptor: MTLVertexDescriptor
    private let colorPixelFormat: MTLPixelFormat
    private let depthPixelFormat: MTLPixelFormat
    private var programs: [Key: MaterialProgram] = [:]
    private var libraries: [ObjectIdentifier: any MTLLibrary] = [:]

    init(
        device: any MTLDevice,
        vertexFunction: any MTLFunction,
        vertexDescriptor: MTLVertexDescriptor,
        colorPixelFormat: MTLPixelFormat,
        depthPixelFormat: MTLPixelFormat
    ) {
        self.device = device
        self.vertexFunction = vertexFunction
        self.vertexDescriptor = vertexDescriptor
        self.colorPixelFormat = colorPixelFormat
        self.depthPixelFormat = depthPixelFormat
    }

    /// Returns the prepared program for `functionName` in `library`, creating
    /// and checking its pipeline on first use, and checks that the function's
    /// parameter block matches `parameterLength` bytes.
    ///
    /// - Parameter resources: Incremented for every pipeline Metal creates,
    ///   including one rejected afterwards by the binding checks.
    func program(
        functionName: String,
        library: any MTLLibrary,
        parameterLength: Int,
        resources: inout ResourceStatistics
    ) throws(MaterialError) -> MaterialProgram {
        guard parameterLength < Self.parameterLengthLimit else {
            throw .parametersTooLarge(bytes: parameterLength, limit: Self.parameterLengthLimit)
        }
        let key = Key(library: ObjectIdentifier(library), functionName: functionName)
        let program: MaterialProgram
        if let cached = programs[key] {
            program = cached
        } else {
            program = try prepare(functionName: functionName, library: library, resources: &resources)
            programs[key] = program
            libraries[key.library] = library
        }
        try Self.check(parameterLength: parameterLength, against: program)
        return program
    }

    private func prepare(
        functionName: String,
        library: any MTLLibrary,
        resources: inout ResourceStatistics
    ) throws(MaterialError) -> MaterialProgram {
        guard library.device.registryID == device.registryID else {
            throw .libraryDeviceMismatch(libraryDevice: library.device.name, rendererDevice: device.name)
        }
        guard let function = library.makeFunction(name: functionName) else {
            throw .functionMissing(name: functionName, fragmentFunctions: Self.fragmentFunctionNames(in: library))
        }
        // Handing Metal a function of another type is an API-validation abort,
        // not a recoverable error, so it is rejected here.
        guard function.functionType == .fragment else {
            throw .notAFragmentFunction(name: functionName, functionType: function.functionType)
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Frostfield.Material.\(functionName)"
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = function
        descriptor.vertexDescriptor = vertexDescriptor
        descriptor.colorAttachments[0].pixelFormat = colorPixelFormat
        descriptor.depthAttachmentPixelFormat = depthPixelFormat
        let pipeline: any MTLRenderPipelineState
        let reflection: MTLRenderPipelineReflection?
        do {
            (pipeline, reflection) = try device.makeRenderPipelineState(
                descriptor: descriptor,
                options: [.bindingInfo, .bufferTypeInfo]
            )
        } catch {
            throw .pipelineCreationFailed(function: functionName, reason: error.localizedDescription)
        }
        resources.renderPipelinesCreated += 1
        guard let reflection else {
            throw .pipelineCreationFailed(function: functionName, reason: "Metal returned no binding reflection")
        }

        var parameterBlockLength: Int?
        for binding in reflection.fragmentBindings {
            let parameterIndex = Int(FF_MATERIAL_PARAMETERS_BUFFER_INDEX)
            if binding.type == .buffer, binding.index == parameterIndex,
               binding.access == .readOnly, let buffer = binding as? any MTLBufferBinding {
                parameterBlockLength = buffer.bufferDataSize
            } else {
                throw .unsupportedBinding(function: functionName, argument: binding.name, binding: Self.describe(binding))
            }
        }
        return MaterialProgram(
            functionName: functionName,
            pipeline: pipeline,
            parameterBlockLength: parameterBlockLength,
            cache: self
        )
    }

    private static func check(parameterLength: Int, against program: MaterialProgram) throws(MaterialError) {
        switch (program.parameterBlockLength, parameterLength) {
        case (nil, 0):
            return
        case (nil, let hostBytes):
            throw .parametersNotDeclared(function: program.functionName, hostBytes: hostBytes)
        case (let shaderBytes?, 0):
            throw .parametersMissing(function: program.functionName, shaderBytes: shaderBytes)
        case (let shaderBytes?, let hostBytes) where shaderBytes != hostBytes:
            throw .parameterSizeMismatch(function: program.functionName, shaderBytes: shaderBytes, hostBytes: hostBytes)
        default:
            return
        }
    }

    /// The fragment functions in `library`, sorted, for a missing-function
    /// diagnostic.
    private static func fragmentFunctionNames(in library: any MTLLibrary) -> [String] {
        library.functionNames
            .filter { library.makeFunction(name: $0)?.functionType == .fragment }
            .sorted()
    }

    /// The binding's attribute as it appears in MSL, such as `texture(0)`.
    private static func describe(_ binding: any MTLBinding) -> String {
        let index = binding.index
        return switch binding.type {
        case .buffer:
            binding.access == .readOnly ? "[[buffer(\(index))]]" : "[[buffer(\(index))]] with write access"
        case .texture: "[[texture(\(index))]]"
        case .sampler: "[[sampler(\(index))]]"
        case .threadgroupMemory: "[[threadgroup(\(index))]]"
        default: "binding index \(index) (binding type \(binding.type.rawValue))"
        }
    }
}
