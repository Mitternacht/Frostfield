import Metal

/// A fragment function prepared to shade meshes drawn by one
/// ``MeshRenderer``, together with the values it reads.
///
/// A material is made by ``MeshRenderer/makeUnlitMaterial(color:)`` (the
/// built-in unlit material) or ``MeshRenderer/makeMaterial(fragmentFunction:library:parameters:)``
/// (a fragment function the app supplies). Making it checks the function and
/// prepares its render pipeline; drawing never creates one.
///
/// ``parameters`` is a plain value. Change it and draw again to animate the
/// material: the renderer copies its bytes into the frame's commands when it
/// encodes the frame, so a change never affects a frame already submitted, and
/// it reuses the prepared pipeline. Every material is drawn with the same
/// vertex stage, depth test, culling, and frame slots as the built-in ones;
/// see `docs/materials.md` for the binding contract.
public struct Material<Parameters: BitwiseCopyable> {
    /// The values the fragment function reads from its parameter block,
    /// `[[buffer(0)]]` in the fragment argument table.
    public var parameters: Parameters

    /// The name of the material's fragment function.
    public var fragmentFunctionName: String { program.functionName }

    let program: MaterialProgram

    init(program: MaterialProgram, parameters: Parameters) {
        self.program = program
        self.parameters = parameters
    }
}

/// The parameter block of the built-in unlit material.
///
/// Its layout is the shader record `FFUnlitParameters`: 16 bytes, 16-byte
/// aligned, with `color` at offset 0.
public struct UnlitParameters: Sendable, Equatable, BitwiseCopyable {
    /// Red, green, blue, and alpha. The renderer writes them to the color
    /// attachment unchanged: there is no blending and no color conversion, so
    /// with a non-sRGB format such as `bgra8Unorm` the stored values are these
    /// values.
    public var color: SIMD4<Float>

    public init(color: SIMD4<Float>) {
        self.color = color
    }
}

/// Why a fragment function cannot become a ``Material``.
///
/// Every case is detected while the material is made, before any frame uses
/// it, and its description names the function, what is wrong, and how to fix
/// it.
public enum MaterialError: Error, Equatable, Sendable, CustomStringConvertible {
    /// The library was made by a different device than the renderer's.
    case libraryDeviceMismatch(libraryDevice: String, rendererDevice: String)
    /// The library has no function with this name. `fragmentFunctions` lists
    /// the fragment functions it does have.
    case functionMissing(name: String, fragmentFunctions: [String])
    /// The named function is not a fragment function.
    case notAFragmentFunction(name: String, functionType: MTLFunctionType)
    /// Metal could not combine Frostfield's mesh vertex function with this
    /// fragment function, for example because the fragment function reads a
    /// varying the vertex function does not write. `reason` is Metal's
    /// message.
    case pipelineCreationFailed(function: String, reason: String)
    /// The fragment function declares an argument that Frostfield does not
    /// bind, such as a texture or a second buffer.
    case unsupportedBinding(function: String, argument: String, binding: String)
    /// The fragment function's parameter block and the Swift parameters have
    /// different sizes.
    case parameterSizeMismatch(function: String, shaderBytes: Int, hostBytes: Int)
    /// The Swift parameters are not empty, but the fragment function declares
    /// no parameter block.
    case parametersNotDeclared(function: String, hostBytes: Int)
    /// The fragment function declares a parameter block, but the Swift
    /// parameters are empty.
    case parametersMissing(function: String, shaderBytes: Int)
    /// The Swift parameters are too large to copy into a frame's commands.
    case parametersTooLarge(bytes: Int, limit: Int)

    public var description: String {
        switch self {
        case .libraryDeviceMismatch(let libraryDevice, let rendererDevice):
            "The shader library belongs to the device \(libraryDevice), but the renderer draws with "
                + "\(rendererDevice). Make the library with the renderer's device."
        case .functionMissing(let name, let fragmentFunctions):
            "The shader library has no function named \(name). "
                + (fragmentFunctions.isEmpty
                    ? "It has no fragment functions at all; check that the .metal file is compiled into this library."
                    : "Its fragment functions are: \(fragmentFunctions.joined(separator: ", ")).")
        case .notAFragmentFunction(let name, let functionType):
            "\(name) is a \(Self.describe(functionType)) function. A material needs a function declared with "
                + "the `fragment` qualifier."
        case .pipelineCreationFailed(let function, let reason):
            "Metal could not combine Frostfield's mesh vertex function with the fragment function \(function): "
                + "\(reason). A material's [[stage_in]] struct may contain only `float4 [[position]]`, "
                + "`float3 [[user(ff_model_normal)]]`, and `float3 [[user(ff_model_position)]]` members, "
                + "and its return type must suit the target's color format."
        case .unsupportedBinding(let function, let argument, let binding):
            "The fragment function \(function) declares the argument \(argument) at \(binding), which Frostfield "
                + "does not bind. A material may read only its parameter block, declared as "
                + "`constant T &parameters [[buffer(0)]]`, and the varyings in its [[stage_in]] struct."
        case .parameterSizeMismatch(let function, let shaderBytes, let hostBytes):
            "The fragment function \(function) reads a \(shaderBytes)-byte parameter block, but the Swift "
                + "parameters are \(hostBytes) bytes (MemoryLayout.size). Declare the record once in a C header "
                + "that both Swift and the shader include, or make both declarations match, including padding."
        case .parametersNotDeclared(let function, let hostBytes):
            "The Swift parameters are \(hostBytes) bytes, but the fragment function \(function) declares no "
                + "parameter block. Add `constant T &parameters [[buffer(0)]]` to the function, or pass an empty "
                + "parameters value."
        case .parametersMissing(let function, let shaderBytes):
            "The fragment function \(function) reads a \(shaderBytes)-byte parameter block at [[buffer(0)]], "
                + "but the Swift parameters are empty. Pass a parameters value with the same layout."
        case .parametersTooLarge(let bytes, let limit):
            "The parameters are \(bytes) bytes. A material's parameter block must be smaller than \(limit) "
                + "bytes, because the renderer copies it into each frame's commands."
        }
    }

    private static func describe(_ functionType: MTLFunctionType) -> String {
        switch functionType {
        case .vertex: "vertex"
        case .fragment: "fragment"
        case .kernel: "kernel"
        case .visible: "visible"
        case .intersection: "intersection"
        case .mesh: "mesh"
        case .object: "object"
        @unknown default: "non-fragment (type \(functionType.rawValue))"
        }
    }
}
