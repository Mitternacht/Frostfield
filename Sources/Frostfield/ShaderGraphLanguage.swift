import Foundation

// The vocabulary shared by every typed shader graph: stages, built-in inputs,
// operations, and the declarations a graph can bind at runtime.
extension MaterialGraph {
    /// Where a graph's expressions run. Validation rejects inputs and nodes
    /// that a stage cannot provide.
    public enum Stage: String, Hashable, Sendable {
        /// Per-pixel raster shading and per-hit path tracing.
        case surface
        /// Object-space vertex deformation, evaluated per prepared mesh.
        case vertex
        /// Full-screen image processing of a rendered frame.
        case postProcess
        /// Per-texel or per-element compute work.
        case compute
    }

    /// Built-in values a stage provides to a graph.
    public enum Input: Hashable, Sendable {
        /// Primary top-left texture coordinates. Surface and vertex stages.
        case uv0
        /// Secondary texture coordinates, or zero when a mesh has none. Surface and vertex stages.
        case uv1
        /// World-space shading position. Surface stage.
        case worldPosition
        /// Unit world-space shading normal after authored normal maps. Surface stage.
        case normal
        /// Unit world-space tangent along increasing U of UV0, or zero where a renderer
        /// cannot derive one. Surface stage.
        case tangent
        /// Unit world-space bitangent along increasing V of UV0: `±cross(normal, tangent)`
        /// with the sign of dP/dv, in raster PBR and the path tracer alike, whatever the
        /// sign of an authored vertex tangent's w. Zero where a renderer cannot derive
        /// one. Surface stage.
        case bitangent
        /// Unit direction from the surface toward the viewer. Surface stage.
        case viewDirection
        /// The per-node `materialParameters`. Surface stage.
        case parameters
        /// Linear base color sampled from the authored material before the graph runs. Surface stage.
        case baseColor
        /// Authored metallic factor after its texture. Surface stage.
        case metallic
        /// Authored perceptual roughness after its texture. Surface stage.
        case roughness
        /// Authored linear emission. Surface stage.
        case emission
        /// Undeformed object-space position. Vertex stage.
        case objectPosition
        /// Unit undeformed object-space normal. Vertex stage.
        case objectNormal
        /// Top-left normalized coordinates of the pixel center within the viewport, which
        /// is Three.js `viewportUV`; post-processing covers its whole output. Post-process
        /// and raster surface stages; the path tracer rejects it.
        case screenUV
        /// The output or viewport size in pixels. Post-process and raster surface stages;
        /// the path tracer rejects it.
        case viewportSize
        /// Render target pixel coordinates of the pixel center, top-left origin, with
        /// centers at half-integers: Three.js `screenCoordinate`. Post-process and raster
        /// surface stages; the path tracer rejects it.
        case screenCoordinate
        /// The render target size in pixels: Three.js `screenSize`. Post-process and raster
        /// surface stages; the path tracer rejects it.
        case screenSize
        /// One over the output dimensions, for neighboring samples. Post-process and texture compute.
        case texelSize
        /// Top-left normalized coordinates of a generated texel center. Texture compute.
        case gridUV
        /// The element index of a buffer compute dispatch. Exact below 2²⁴.
        case elementIndex
        /// The element count of a buffer compute dispatch.
        case elementCount
        /// One of four float4 fields of the current buffer element. Buffer compute.
        case elementField(Int)
        /// Seconds on the graph's clock, from `ShaderUniforms.frame`. Every stage.
        case time
        /// Seconds since the previous frame on the graph's clock. Every stage.
        case deltaTime
        /// The graph clock's frame counter modulo 2²⁴, exact as a float. Every stage.
        case frame
        /// World-space camera position. Surface stage.
        case cameraPosition
        /// Distance to the camera's near clip plane. Surface stage.
        case cameraNear
        /// Distance to the camera's far clip plane. Surface stage.
        case cameraFar
        /// Visibility of the raster directional shadow map. Surface color stage.
        case directionalShadowVisibility
        /// The world-to-camera matrix. Surface stage.
        case viewMatrix
        /// The camera-to-clip projection matrix with Metal's 0...1 depth. Surface stage.
        case projectionMatrix
        /// The mesh node's object-to-world matrix. Instanced draws report the node's
        /// matrix without the per-instance transform, as Three.js `modelWorldMatrix` does.
        /// Surface stage.
        case modelMatrix
        /// The inverse transpose of the model matrix's upper 3 × 3, which maps object normals
        /// to world space. Surface stage.
        case normalMatrix

        public var type: ValueType {
            switch self {
            case .uv0, .uv1, .screenUV, .viewportSize, .screenCoordinate, .screenSize, .texelSize, .gridUV: .vector2
            case .worldPosition, .normal, .tangent, .bitangent, .viewDirection, .baseColor, .emission,
                 .objectPosition, .objectNormal, .cameraPosition: .vector3
            case .parameters, .elementField: .vector4
            case .metallic, .roughness, .elementIndex, .elementCount, .time, .deltaTime, .frame,
                 .cameraNear, .cameraFar, .directionalShadowVisibility: .scalar
            case .viewMatrix, .projectionMatrix, .modelMatrix: .matrix4
            case .normalMatrix: .matrix3
            }
        }

        /// Whether a stage provides this input.
        public func isAvailable(in stage: Stage) -> Bool {
            switch self {
            case .uv0, .uv1: stage == .surface || stage == .vertex
            case .worldPosition, .normal, .tangent, .bitangent, .viewDirection, .parameters,
                 .baseColor, .metallic, .roughness, .emission, .cameraPosition, .cameraNear, .cameraFar,
                 .viewMatrix, .projectionMatrix, .modelMatrix, .normalMatrix, .directionalShadowVisibility: stage == .surface
            case .objectPosition, .objectNormal: stage == .vertex
            case .screenUV, .viewportSize, .screenCoordinate, .screenSize: stage == .postProcess || stage == .surface
            case .texelSize: stage == .postProcess || stage == .compute
            case .gridUV, .elementIndex, .elementCount: stage == .compute
            case .elementField(let index): stage == .compute && (0..<4).contains(index)
            case .time, .deltaTime, .frame: true
            }
        }

        /// Whether the input reads the graph clock, which renderers bind after the declared uniforms.
        var readsClock: Bool { self == .time || self == .deltaTime || self == .frame }

        /// Whether only a rasterizer can provide the input: the path tracer has no pixel
        /// grid at secondary bounces.
        var isScreenSpace: Bool {
            [.screenUV, .viewportSize, .screenCoordinate, .screenSize, .directionalShadowVisibility].contains(self)
        }

        var expression: String {
            switch self {
            case .uv0: "input.uv0"
            case .uv1: "input.uv1"
            case .worldPosition: "input.worldPosition"
            case .normal: "frostfield_graph_unit(input.normal)"
            case .tangent: "frostfield_graph_unit(input.tangent)"
            case .bitangent: "frostfield_graph_unit(input.bitangent)"
            case .viewDirection: "frostfield_graph_unit(input.viewDirection)"
            case .parameters: "input.parameters"
            case .baseColor: "input.baseColor"
            case .metallic: "input.metallic"
            case .roughness: "input.roughness"
            case .emission: "input.emission"
            case .objectPosition: "input.objectPosition"
            case .objectNormal: "frostfield_graph_unit(input.objectNormal)"
            case .screenUV: "input.screenUV"
            case .viewportSize: "input.viewportSize"
            case .screenCoordinate: "input.screenCoordinate"
            case .screenSize: "input.screenSize"
            case .texelSize: "input.texelSize"
            case .gridUV: "input.gridUV"
            case .elementIndex: "input.elementIndex"
            case .elementCount: "input.elementCount"
            case .elementField(let index): "input.fields[\(index)]"
            case .cameraPosition: "input.cameraPosition"
            case .cameraNear: "input.cameraClip.x"
            case .cameraFar: "input.cameraClip.y"
            case .directionalShadowVisibility: "input.directionalShadowVisibility"
            case .viewMatrix: "input.viewMatrix"
            case .projectionMatrix: "input.projectionMatrix"
            case .modelMatrix: "input.modelMatrix"
            case .normalMatrix: "input.normalMatrix"
            // The clock follows the declared uniforms; ShaderProgram substitutes the slot.
            case .time, .deltaTime, .frame: ""
            }
        }
    }

    /// Componentwise operations unless noted. Results stay finite for finite inputs
    /// where the comment says so; renderers sanitize nonfinite graph outputs.
    public enum UnaryOperation: String, Hashable, Sendable, CaseIterable {
        case abs, sign, floor, ceil, round, fract
        /// Square root of the input clamped to zero.
        case sqrt
        /// Reciprocal square root, or zero for nonpositive inputs.
        case inverseSqrt
        case exp, exp2
        /// Natural logarithm of the input clamped to a tiny positive value.
        case log
        /// Base-two logarithm of the input clamped to a tiny positive value.
        case log2
        case sin, cos, tan
        /// Arcsine of the input clamped to -1...1.
        case asin
        /// Arccosine of the input clamped to -1...1.
        case acos
        case atan
        case negate
        /// Returns 1 - x.
        case oneMinus
        /// Clamps to zero through one.
        case saturate
        /// Unit vector, or zero for a zero or nonfinite vector. Vectors only.
        case normalize
        /// Euclidean length. Returns a scalar.
        case length
        /// Rec. 709 relative luminance of a linear color. Vector3 input, scalar result.
        case luminance
        /// Converts hue, saturation and value in 0...1 to linear RGB. Vector3 only.
        case hsvToRGB
        /// Converts RGB to hue, saturation and value. Vector3 only.
        case rgbToHSV
        /// Decodes the sRGB transfer function. Vector3 only.
        case srgbToLinear
        /// Encodes the sRGB transfer function. Vector3 only.
        case linearToSRGB
    }

    public enum BinaryOperation: String, Hashable, Sendable, CaseIterable {
        case minimum, maximum
        /// Raises the first operand, clamped to zero, to the second.
        case power
        /// Floored modulo. A zero divisor returns zero.
        case modulo
        /// Two-argument arctangent of y (first) over x (second).
        case atan2
        /// Returns 0 where the second operand is below the first (the edge), else 1.
        case step
        /// Dot product. Returns a scalar.
        case dot
        /// Cross product of two vector3 values.
        case cross
        /// Euclidean distance. Returns a scalar.
        case distance
        /// Reflects a vector about the normalized second vector.
        case reflect
        /// 1 where the first operand is greater, else 0, componentwise.
        case greaterThan
        /// 1 where the first operand is smaller, else 0, componentwise.
        case lessThan
    }

    /// Lattice noise variants. Each returns approximately 0...1.
    public enum NoiseKind: String, Hashable, Sendable {
        /// Trilinear-smoothed value noise.
        case value
        /// Quintic-faded gradient (Perlin-style) noise.
        case gradient
    }

    /// Periodic waves with period 1 and range 0...1 that start at 0, as in Three.js
    /// `oscSine`, `oscSquare`, `oscTriangle` and `oscSawtooth`. Componentwise.
    public enum Oscillator: String, Hashable, Sendable, CaseIterable {
        /// `0.5 − 0.5 cos(2π t)`, which equals Three.js `sin(2π (t + 0.75)) / 2 + 0.5`.
        case sine
        /// 0 where `fract(t)` is below 0.5, else 1.
        case square
        /// `|2 fract(t + 0.5) − 1|`: 0 at whole numbers and 1 halfway between.
        case triangle
        /// `fract(t)`.
        case sawtooth
    }

    /// Screen-space partial derivatives, as in Three.js `dFdx`, `dFdy` and `fwidth`.
    /// Only rasterized fragments provide them: the raster surface and post-process stages.
    public enum Derivative: String, Hashable, Sendable, CaseIterable {
        /// Change per pixel to the right.
        case dx
        /// Change per pixel downward in Metal window coordinates.
        case dy
        /// `|dx| + |dy|`.
        case width
    }

    /// Bitwise operations on unsigned 24-bit integers carried in floats. Operands
    /// truncate toward zero and clamp to 0...16,777,215; nonfinite operands read as zero.
    /// Results keep their low 24 bits, so every result is exact as a float.
    public enum BitOperation: String, Hashable, Sendable, CaseIterable {
        case and, or, xor
        /// Shifts the first operand left by the second, 0...23 bits.
        case shiftLeft
        /// Shifts the first operand right by the second, 0...23 bits.
        case shiftRight
    }

    /// Operations on a single matrix.
    public enum MatrixOperation: String, Hashable, Sendable, CaseIterable {
        case transpose
        /// The inverse, or the zero matrix when the determinant is zero or nonfinite.
        case inverse
        /// The determinant, a scalar.
        case determinant
        /// The upper-left 3 × 3 of a matrix4.
        case upperLeft
    }
}

/// A named value that an app can change after the graph's pipeline is prepared.
/// Uniform names are labels for diagnostics and lookup. They are never emitted into Metal.
public struct ShaderUniform: Hashable, Sendable {
    public let name: String
    public let type: MaterialGraph.ValueType
    /// The initial value in its leading components. Unused components are zero.
    public let defaultValue: SIMD4<Float>

    public init(_ name: String, _ value: Float) {
        self.init(name: name, type: .scalar, value: SIMD4(value, 0, 0, 0))
    }

    public init(_ name: String, _ value: SIMD2<Float>) {
        self.init(name: name, type: .vector2, value: SIMD4(value.x, value.y, 0, 0))
    }

    public init(_ name: String, _ value: SIMD3<Float>) {
        self.init(name: name, type: .vector3, value: SIMD4(value, 0))
    }

    public init(_ name: String, _ value: SIMD4<Float>) {
        self.init(name: name, type: .vector4, value: value)
    }

    private init(name: String, type: MaterialGraph.ValueType, value: SIMD4<Float>) {
        self.name = name
        self.type = type
        defaultValue = type.masked(value)
    }
}

/// A texture a graph samples by slot. Bind the actual `MTLTexture` when preparing
/// the material, pass, or bake that uses the graph.
public struct ShaderTextureInput: Hashable, Sendable {
    public enum Content: String, Hashable, Sendable {
        /// Color stored in an sRGB format, which Metal decodes when sampling, or linear float color.
        case color
        /// Linear values such as masks, heights, or packed normals.
        case data
    }

    public enum Addressing: String, Hashable, Sendable {
        case `repeat`
        case clampToEdge
        case mirroredRepeat

        var code: UInt32 {
            switch self {
            case .repeat: 0
            case .clampToEdge: 1
            case .mirroredRepeat: 2
            }
        }
    }

    public enum Filtering: String, Hashable, Sendable {
        /// Bilinear filtering with linear mip selection.
        case linear
        /// Nearest texel with nearest mip selection.
        case nearest

        var code: UInt32 { self == .linear ? 0 : 1 }
    }

    public let name: String
    public let content: Content
    public let addressing: Addressing
    public let filtering: Filtering

    public init(
        _ name: String, content: Content = .color, addressing: Addressing = .repeat, filtering: Filtering = .linear
    ) {
        self.name = name
        self.content = content
        self.addressing = addressing
        self.filtering = filtering
    }
}

/// A caller-written Metal function that a graph can call like a built-in node.
/// The body is ordinary Metal statements that return the declared type. It may
/// use the standard library and the `frostfield_graph_` helpers but no other graph state.
public struct ShaderFunction: Hashable, Sendable {
    public struct Parameter: Hashable, Sendable {
        public let name: String
        public let type: MaterialGraph.ValueType

        public init(_ name: String, _ type: MaterialGraph.ValueType) {
            self.name = name
            self.type = type
        }
    }

    public static let maximumBodyLength = 16_384

    public let name: String
    public let parameters: [Parameter]
    public let returnType: MaterialGraph.ValueType
    public let body: String

    public init(name: String, parameters: [Parameter], returns returnType: MaterialGraph.ValueType, body: String) {
        self.name = name
        self.parameters = parameters
        self.returnType = returnType
        self.body = body
    }
}

extension MaterialGraph.ValueType {
    func masked(_ value: SIMD4<Float>) -> SIMD4<Float> {
        switch self {
        case .scalar: SIMD4(value.x, 0, 0, 0)
        case .vector2: SIMD4(value.x, value.y, 0, 0)
        case .vector3: SIMD4(value.x, value.y, value.z, 0)
        case .vector4, .matrix3, .matrix4: value
        }
    }
}
