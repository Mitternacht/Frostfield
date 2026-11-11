import Foundation
import Metal

/// A bounded graph for a mesh fragment's straight linear RGB color.
/// The compiled material multiplies RGB by the saturated alpha in
/// MeshScene.materialParameters.w and writes premultiplied RGBA.
///
/// `MaterialGraph` also defines the typed node vocabulary shared by
/// `SurfaceShaderGraph`, `PostProcessGraph`, and the compute graphs.
public struct MaterialGraph: Sendable {
    /// A node's index in the graph's node array. Forward references are allowed.
    public typealias NodeID = Int

    public static let maximumNodeCount = 1_024
    public static let maximumUniformCount = 32
    public static let maximumTextureCount = 8
    public static let maximumFunctionCount = 16
    /// The most iterations one `.loop` node may run, including a dynamic count's maximum.
    public static let maximumLoopIterations = 1_024
    /// The most arms one `.switchSelect` node may have, besides its fallback.
    public static let maximumSwitchCases = 16
    /// The deepest nesting of branch arms and loop bodies in generated code.
    public static let maximumNestingDepth = 16
    /// The most layers a `.parallaxOcclusion` node may march.
    public static let maximumParallaxSteps = 128
    /// The most structs one graph may declare.
    public static let maximumStructCount = 16
    /// The most fields one struct may have.
    public static let maximumStructFields = 16
    /// The most loop iterations and parallax layers one evaluation of a stage may run.
    /// Each loop counts its iteration count times the counts of the loops around it,
    /// and each parallax node its layers plus one, times the same factor. This keeps a
    /// graph that validates from running long enough to trip the GPU watchdog.
    public static let maximumTotalIterations = 65_536

    public enum ValueType: String, Hashable, Sendable {
        case scalar
        case vector2
        case vector3
        case vector4
        /// A 3 × 3 float matrix in column-major order.
        case matrix3
        /// A 4 × 4 float matrix in column-major order.
        case matrix4
        /// A signed 32-bit integer.
        case int
        /// A boolean. Comparisons and branches can read it directly.
        case bool
        /// A declared struct. The struct's index lives beside the node; this case only
        /// marks the value as a struct so it cannot be used as a float.
        case structure

        /// A linear RGB color is a vector3.
        public static let color = ValueType.vector3

        public var componentCount: Int {
            switch self {
            case .scalar: 1
            case .vector2: 2
            case .vector3: 3
            case .vector4: 4
            case .matrix3: 9
            case .matrix4: 16
            case .int, .bool: 1
            case .structure: 0
            }
        }

        /// Whether the type is a matrix. Only matrix nodes accept matrix operands.
        public var isMatrix: Bool { self == .matrix3 || self == .matrix4 }

        /// Float4 slots in a uniform buffer. A matrix uses one slot per column.
        var uniformSlotCount: Int {
            switch self {
            case .matrix3: 3
            case .matrix4: 4
            case .structure: 0
            default: 1
            }
        }

        init?(componentCount: Int) {
            switch componentCount {
            case 1: self = .scalar
            case 2: self = .vector2
            case 3: self = .vector3
            case 4: self = .vector4
            default: return nil
            }
        }

        var metalType: String {
            switch self {
            case .scalar: "float"
            case .vector2: "float2"
            case .vector3: "float3"
            case .vector4: "float4"
            case .matrix3: "float3x3"
            case .matrix4: "float4x4"
            case .int: "int"
            case .bool: "bool"
            case .structure: "void"
            }
        }
    }

    public enum Parameter: String, Sendable {
        case x, y, z, w
    }

    public enum Axis: String, Sendable {
        case x, y, z
    }

    public enum UV: Hashable, Sendable {
        case u, v
    }

    /// The associated values define each operation's input count.
    public enum Node: Hashable, Sendable {
        case scalar(Float)
        /// A signed 32-bit integer constant.
        case integer(Int32)
        /// A boolean constant.
        case boolean(Bool)
        /// A vector3 constant, usually a linear RGB color.
        case color(SIMD3<Float>)
        case vector2(SIMD2<Float>)
        case vector4(SIMD4<Float>)
        /// A component of the scene's per-draw float4. For animation, pass time in x.
        case parameter(Parameter)
        case uv(UV)
        case normal(Axis)
        case worldPosition(Axis)
        /// A built-in value provided by the graph's stage.
        case input(Input)
        /// The value of a declared uniform, typed by its declaration.
        case uniform(Int)
        /// Samples a declared texture at vector2 coordinates and returns RGBA.
        /// Vertex and compute stages sample the base mip level.
        case sampleTexture(slot: Int, coordinates: NodeID)
        /// Samples a declared texture at an explicit mip level. A nonfinite level reads level 0.
        /// The level is clamped to the texture's last mip. Unlike `sampleTexture`, this may run inside a branch.
        case sampleTextureLevel(slot: Int, coordinates: NodeID, level: NodeID)
        /// Samples a declared texture with explicit UV derivatives. The mip is
        /// log2 of the larger footprint in texels, rounded by the nearest mip filter
        /// and clamped to the last mip. A nonfinite derivative reads level 0.
        /// Like `sampleTextureLevel`, this may run inside a branch.
        case sampleTextureGradient(slot: Int, coordinates: NodeID, ddx: NodeID, ddy: NodeID)
        /// Samples a cube map along a direction and returns RGBA. Compute graphs read mip 0.
        case sampleCube(slot: Int, direction: NodeID)
        /// Samples the viewport or post-process image at mip zero. Returns premultiplied linear RGBA.
        case sampleScene(coordinates: NodeID)
        /// Samples the viewport or post-process image at a scalar mip level.
        case sampleSceneLevel(coordinates: NodeID, level: NodeID)
        /// Samples viewport or post-process depth at vector2 coordinates. Returns stored Metal depth in 0...1.
        case sampleDepth(coordinates: NodeID)
        /// Chooses the fallback UV when a stored raster depth is in front of the fragment.
        case viewportSafeUV(coordinates: NodeID, fallback: NodeID)
        /// Combines authored opacity with a raster particle's view-space fade against stored depth.
        case softParticleOpacity(opacity: NodeID, distance: NodeID, contrast: NodeID, depth: NodeID)
        /// Post-process view position from top-left UV, raw Metal depth and an inverse projection.
        case viewPositionFromDepth(coordinates: NodeID, depth: NodeID, inverseProjection: NodeID)
        /// Post-process view normal from neighboring texels in the bound depth texture.
        case normalFromDepth(coordinates: NodeID, inverseProjection: NodeID, reversedDepth: Bool)
        /// Edge-preserving full-screen denoise of the scene source.
        case denoise(
            coordinates: NodeID, inverseProjection: NodeID, settings: NodeID, sampleIndex: NodeID,
            normalSlot: Int?, reversedDepth: Bool
        )
        /// Post-process linear near-to-far fraction from raw Metal depth and finite clip distances.
        case linearDepth(coordinates: NodeID, depth: NodeID, inverseProjection: NodeID, near: NodeID, far: NodeID)
        /// Adds two values of the same type.
        case add(NodeID, NodeID)
        /// Subtracts two values of the same type.
        case subtract(NodeID, NodeID)
        /// Multiplies matching types componentwise or scales any value by a scalar.
        case multiply(NodeID, NodeID)
        /// Divides any value by a scalar. A zero or nonfinite divisor returns zero.
        case divide(NodeID, NodeID)
        /// Interpolates matching values by a scalar factor clamped to zero through one.
        case mix(NodeID, NodeID, factor: NodeID)
        /// Takes the sine of a scalar in radians.
        case sine(NodeID)
        /// Takes the cosine of a scalar in radians.
        case cosine(NodeID)
        case floor(NodeID)
        /// Returns a scalar's fractional part, including for negative inputs.
        case fract(NodeID)
        /// Hermite interpolation between scalar edges. Nonincreasing edges make a step at edge0.
        case smoothstep(edge0: NodeID, edge1: NodeID, value: NodeID)
        /// Deterministic 3D value noise. Coordinates clamp to ±1,048,576.
        case noise(x: NodeID, y: NodeID, z: NodeID)
        /// Clamps a scalar or each vector component to zero through one.
        case clamp01(NodeID)
        case rgb(red: NodeID, green: NodeID, blue: NodeID)
        /// A componentwise function of one value. See `UnaryOperation` for result types.
        case unary(UnaryOperation, NodeID)
        /// A function of two values. Operands share a type unless `BinaryOperation` says otherwise.
        case binary(BinaryOperation, NodeID, NodeID)
        /// Clamps a value between bounds of its own type or scalar bounds.
        case clamp(NodeID, minimum: NodeID, maximum: NodeID)
        /// Chooses the second value where the scalar condition exceeds 0.5, else the third.
        case select(condition: NodeID, ifTrue: NodeID, ifFalse: NodeID)
        /// Maps a value linearly from one scalar range to another. An empty source range returns `toMinimum`.
        case remap(NodeID, fromMinimum: NodeID, fromMaximum: NodeID, toMinimum: NodeID, toMaximum: NodeID)
        /// Selects one to four components with `xyzw` or `rgba` letters, such as "zyx" or "a".
        /// A scalar can be repeated with "x" letters.
        case swizzle(NodeID, String)
        /// Concatenates two to four values into a vector of two to four components.
        case combine([NodeID])
        /// 3D lattice noise of a vector3 position, in approximately 0...1.
        case latticeNoise(NoiseKind, position: NodeID)
        /// Fractal sum of 1–8 noise octaves at a vector3 position, normalized to approximately 0...1.
        case fractalNoise(NoiseKind, position: NodeID, octaves: Int, lacunarity: Float, gain: Float)
        /// Cellular (Worley) noise of a vector3 position. Returns the distance to the
        /// nearest feature point (x) and a 0...1 identifier of its cell (y).
        case cellularNoise(position: NodeID)
        /// Three.js's polynomial-gradient simplex noise at a vector3 position.
        case simplexNoise(position: NodeID)
        /// Curl of the r187 three-component simplex field at a vector3 position.
        case curlNoise(position: NodeID)
        /// Squared distance to the nearest animated feature in two or three dimensions.
        case animatedVoronoi(position: NodeID, time: NodeID)
        /// Calls a declared `ShaderFunction` with typed arguments.
        case function(Int, arguments: [NodeID])
        /// Reinterprets a float's bits as an int, or an int's bits as a float.
        case bitcast(NodeID)
        /// The number of set bits. A float uses its exact bits; an int uses its two's-complement bits.
        case countOneBits(NodeID)
        /// Truncates a float toward zero, or converts an int to a float.
        case numericCast(NodeID)
        /// Builds a declared struct from one value per field, in declaration order.
        case constructStruct(index: Int, fields: [NodeID])
        /// One field of a struct value. `field` is the field's position in the declaration.
        case structField(value: NodeID, field: Int)

        // MARK: Control flow

        /// Evaluates only the chosen arm: `ifTrue` where the scalar condition exceeds 0.5,
        /// else `ifFalse`. Nodes that only one arm reads are generated inside that arm.
        /// Texture samples, derivatives and loops that contain them stay outside branches
        /// so that their screen-space derivatives remain defined. Nest branches in
        /// `ifFalse` for else-if chains.
        case branch(condition: NodeID, ifTrue: NodeID, ifFalse: NodeID)
        /// Chooses the value whose match equals `selector`, or `fallback`. Only the chosen
        /// arm's private nodes run. Matches are exact, so a scalar selector should be an
        /// integer-valued float. One through `maximumSwitchCases` arms.
        case switchSelect(selector: NodeID, matches: [NodeID], values: [NodeID], fallback: NodeID)
        /// Runs a bounded loop of 1...`maximumLoopIterations` iterations. The value starts
        /// at `initial`; each iteration evaluates `next` from this loop's `loopIndex` and
        /// `loopValue` nodes and stores it as the new value. When `iterations` is set, the
        /// loop runs that many times after truncating toward zero, clamped to 0...`count`,
        /// and a nonfinite count runs none. When `exit` exceeds 0.5 the loop stops after
        /// that iteration. When `skip` exceeds 0.5 the iteration keeps the accumulator and
        /// does not run the nodes only `next` reads. Nodes that read neither loop node are
        /// evaluated once before the loop.
        case loop(count: Int, iterations: NodeID?, initial: NodeID, next: NodeID, exit: NodeID?, skip: NodeID?)
        /// Sets the surface mask false where `condition` exceeds 0.5, which raster and path
        /// coverage treat as a discarded fragment, and otherwise yields `value`. Surface stage.
        case discard(condition: NodeID, value: NodeID)
        /// The zero-based iteration number of the `.loop` node `loop`, as a scalar.
        /// Valid only in nodes that reach that loop through its `next` or `exit`.
        case loopIndex(loop: NodeID)
        /// The value of the `.loop` node `loop` at the start of the current iteration,
        /// typed like its `initial`. Valid only inside that loop, like `loopIndex`.
        case loopValue(loop: NodeID)

        // MARK: Patterns and coordinates

        /// A periodic wave of any scalar or vector, componentwise. See `Oscillator`.
        case oscillator(Oscillator, NodeID)
        /// Rotates vector2 coordinates by a scalar angle in radians about a vector2 center:
        /// `center + (c x − s y, s x + c y)` for the offset `(x, y)`, as Three.js `rotateUV`.
        case rotateUV(NodeID, angle: NodeID, center: NodeID)
        /// Bulges vector2 coordinates about a vector2 center: `uv + d · |d|⁴ · strength`,
        /// where `d = uv − center` and strength is a scalar or vector2, as Three.js `spherizeUV`.
        case spherizeUV(NodeID, strength: NodeID, center: NodeID)
        /// Polar coordinates of vector2 coordinates about a vector2 center: the distance in x
        /// and the angle `atan2(dy, dx)` in turns, wrapped to 0...1, in y.
        case polar(NodeID, center: NodeID)
        /// 1 where `floor(x) + floor(y)` of vector2 coordinates is odd, else 0. Three.js
        /// `checker(uv)` equals this node at `uv × 2`.
        case checker(NodeID)
        /// Triplanar blend weights from a vector3 normal and a scalar sharpness `k ≥ 0`:
        /// `|n|ᵏ / (|n.x|ᵏ + |n.y|ᵏ + |n.z|ᵏ)`, which sum to 1. A zero normal gives equal
        /// weights. Sharpness 1 matches Three.js `triplanarTexture`.
        case triplanarWeights(normal: NodeID, sharpness: NodeID)
        /// Samples one declared texture, or three (for the x, y and z projections), at a
        /// vector3 position projected onto the yz, zx and xy planes, blended by
        /// `triplanarWeights`. Returns RGBA, as Three.js `triplanarTexture`.
        case triplanarSample(slots: [Int], position: NodeID, normal: NodeID, sharpness: NodeID)
        /// Parallax occlusion mapping. Marches `steps` (1...`maximumParallaxSteps`) equal
        /// layers along a vector3 tangent-space direction toward the viewer into the height
        /// field stored in the red channel of a declared texture, where 1 is the surface and
        /// 0 lies `depth` UV units below it, then interpolates linearly between the layers
        /// that bracket the first intersection. Returns the offset vector2 coordinates.
        /// The direction's z is clamped to at least 0.05; a direction facing away from the
        /// surface returns the coordinates unchanged.
        case parallaxOcclusion(slot: Int, coordinates: NodeID, view: NodeID, depth: NodeID, steps: Int)

        /// A tangent-space normal from the linear red height channel of a texture.
        /// The selected mesh UV set and affine placement carry a local sample footprint.
        case bumpMap(slot: Int, coordinateSet: MeshTextureCoordinateSet,
                     transform: PhysicallyBasedMaterial.TextureTransform, scale: NodeID)

        // MARK: Derivatives, bits and dithering

        /// A screen-space derivative of a scalar or vector. Raster surface and post-process
        /// stages only; vertex and compute stages and the path tracer reject it.
        case derivative(Derivative, NodeID)
        /// A bitwise operation on two scalars. See `BitOperation`.
        case bits(BitOperation, NodeID, NodeID)
        /// A uniform 0...1 hash of the exact bits of a scalar or vector, with negative
        /// zero read as zero. Equal inputs give equal results on the CPU and GPU.
        case hash(NodeID)
        /// Jimenez's interleaved gradient noise of vector2 pixel coordinates:
        /// `fract(52.9829189 · fract(0.06711056 x + 0.00583715 y))`, in 0...1.
        case interleavedGradientNoise(NodeID)

        // MARK: Matrices

        /// Builds a matrix from three vector3 columns or four vector4 columns.
        case matrix([NodeID])
        /// Multiplies matrix × matrix of the same size, or matrix × vector of matching size.
        case matrixMultiply(NodeID, NodeID)
        /// A transpose, inverse, determinant or upper-left 3 × 3. See `MatrixOperation`.
        case matrixUnary(MatrixOperation, NodeID)
        /// One column of a matrix, 0...2 for matrix3 and 0...3 for matrix4.
        case matrixColumn(NodeID, Int)
    }

    public let nodes: [Node]
    public let output: NodeID
    private let program: ShaderProgram

    public init(nodes: [Node], output: NodeID) throws {
        program = try ShaderProgram(
            nodes: nodes, declarations: .init(), outputs: [(output, .surface)], allowsInput: Self.unlitInput
        ) { _, id, type in
            guard type == .vector3 else { throw Failure.outputMustBeColor(id) }
        }
        self.nodes = nodes
        self.output = output
    }

    /// Compiles the graph's fragment function for one device. Call during material
    /// preparation, then reuse the renderer and update the scene's parameters.
    public func compile(device: any MTLDevice) throws -> MeshMaterial {
        let options = MTLCompileOptions()
        options.mathMode = .safe
        let source = try metalSource()
        do {
            let library = try device.makeLibrary(source: source, options: options)
            return .custom(library: library, fragment: Self.fragmentName)
        } catch {
            throw Failure.compilation(ShaderProgram.diagnose(error, source: source))
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case tooManyNodes(limit: Int)
        case invalidOutput(NodeID)
        case invalidReference(node: NodeID, referenced: NodeID)
        case cycle(NodeID)
        case nonfiniteConstant(NodeID)
        case typeMismatch(node: NodeID, input: NodeID, expected: ValueType, actual: ValueType)
        case outputMustBeColor(NodeID)
        case compilation(String)
        /// The node reads an input or resource that its stage cannot provide.
        case unavailable(node: NodeID, stage: Stage)
        case invalidUniform(node: NodeID, index: Int)
        case invalidTexture(node: NodeID, slot: Int)
        case invalidFunction(node: NodeID, index: Int)
        case argumentCount(node: NodeID, expected: Int, actual: Int)
        case invalidSwizzle(node: NodeID, components: String)
        case invalidCombine(node: NodeID, components: Int)
        case invalidNoise(NodeID)
        case invalidDeclaration(String)
        /// The node received a matrix where it needs a scalar or vector.
        case matrixOperand(node: NodeID, input: NodeID)
        /// A matrix node received operands of incompatible sizes or a column out of range.
        case invalidMatrix(NodeID)
        /// A loop count outside 1...`maximumLoopIterations`.
        case invalidLoopCount(node: NodeID, count: Int)
        /// A switch with no cases, too many cases, or cases that do not line up.
        case invalidSwitch(NodeID)
        /// A `loopIndex` or `loopValue` node names a node that is not a `.loop`.
        case notALoop(node: NodeID, loop: NodeID)
        /// A value computed from a loop's iteration is read outside that loop's `next` or `exit`.
        case loopValueEscapes(node: NodeID, loop: NodeID)
        /// Branches and loops nest deeper than `maximumNestingDepth`.
        case nestingTooDeep(NodeID)
        /// A texture sample or derivative depends on a loop that can exit early, where
        /// neighboring pixels may stop at different iterations.
        case divergentDerivative(node: NodeID, loop: NodeID)
        /// A pattern node received invalid constant parameters, such as a triplanar slot count
        /// other than one or three or a parallax step count out of range.
        case invalidPattern(NodeID)
        /// The node brings one evaluation's loop iterations and parallax layers above
        /// `maximumTotalIterations`.
        case tooManyIterations(node: NodeID, iterations: Int)
        /// A struct node names a missing declaration or a missing field.
        case invalidStruct(NodeID)
        /// Two struct values were required to be the same declaration and were not.
        case structMismatch(node: NodeID, expected: Int, actual: Int)

        public var errorDescription: String? {
            switch self {
            case .tooManyNodes(let limit): "A material graph may contain at most \(limit) nodes."
            case .invalidOutput(let node): "Material graph output \(node) is outside the node array."
            case .invalidReference(let node, let referenced):
                "Material graph node \(node) refers to missing node \(referenced)."
            case .cycle(let node): "Material graph node \(node) participates in a cycle."
            case .nonfiniteConstant(let node): "Material graph node \(node) has a nonfinite constant."
            case .typeMismatch(let node, let input, let expected, let actual):
                "Material graph node \(node) expects \(expected.rawValue) from node \(input), which is \(actual.rawValue)."
            case .outputMustBeColor(let node): "Material graph output node \(node) must produce a color."
            case .compilation(let detail): "Could not compile the material graph. \(detail)"
            case .unavailable(let node, let stage):
                "Material graph node \(node) is not available in the \(stage.rawValue) stage."
            case .invalidUniform(let node, let index):
                "Material graph node \(node) reads uniform \(index), which the graph does not declare."
            case .invalidTexture(let node, let slot):
                "Material graph node \(node) samples texture slot \(slot), which the graph does not declare."
            case .invalidFunction(let node, let index):
                "Material graph node \(node) calls function \(index), which the graph does not declare."
            case .argumentCount(let node, let expected, let actual):
                "Material graph node \(node) passes \(actual) arguments where \(expected) are required."
            case .invalidSwizzle(let node, let components):
                "Material graph node \(node) cannot select components \"\(components)\" from its input."
            case .invalidCombine(let node, let components):
                "Material graph node \(node) combines \(components) components; use two to four."
            case .invalidNoise(let node):
                "Material graph node \(node) needs 1–8 octaves and finite positive lacunarity and gain."
            case .invalidDeclaration(let detail): "Invalid material graph declaration. \(detail)"
            case .matrixOperand(let node, let input):
                "Material graph node \(node) needs a scalar or vector, but node \(input) is a matrix; use the matrix nodes."
            case .invalidMatrix(let node):
                "Material graph node \(node) combines matrices or vectors of different sizes, or selects a missing column."
            case .invalidLoopCount(let node, let count):
                "Material graph loop \(node) runs \(count) iterations; use 1–\(MaterialGraph.maximumLoopIterations)."
            case .invalidSwitch(let node):
                "Material graph switch \(node) needs 1–\(MaterialGraph.maximumSwitchCases) cases."
            case .notALoop(let node, let loop):
                "Material graph node \(node) reads the iteration of node \(loop), which is not a loop."
            case .loopValueEscapes(let node, let loop):
                "Material graph node \(node) uses a value from inside loop \(loop) outside that loop's next value or exit condition."
            case .nestingTooDeep(let node):
                "Material graph node \(node) nests branches and loops more than \(MaterialGraph.maximumNestingDepth) levels deep."
            case .divergentDerivative(let node, let loop):
                "Material graph node \(node) samples a texture or takes a derivative inside loop \(loop), which can exit early; "
                    + "neighboring pixels may stop at different iterations. Remove the exit condition or move the node out of the loop."
            case .invalidPattern(let node):
                "Material graph node \(node) needs one or three texture slots, or 1–\(MaterialGraph.maximumParallaxSteps) parallax steps."
            case .tooManyIterations(let node, let iterations):
                "Material graph node \(node) raises one evaluation to \(iterations) loop iterations and parallax layers, "
                    + "counting nested loops multiplied together; the limit is \(MaterialGraph.maximumTotalIterations)."
            case .invalidStruct(let node):
                "Material graph node \(node) uses a struct the graph does not declare, or a field it does not have."
            case .structMismatch(let node, let expected, let actual):
                "Material graph node \(node) needs struct \(expected), but received struct \(actual)."
            }
        }
    }

    // The unlit fragment receives UV0, the interpolated normal, world position, and parameters.
    private static func unlitInput(_ input: Input) -> Bool {
        switch input {
        case .uv0, .normal, .worldPosition, .parameters: true
        default: false
        }
    }

    private static let fragmentName = "frostfield_material_graph"

    private func metalSource() throws -> String {
        let name = Self.fragmentName
        return """
        #include <metal_stdlib>
        using namespace metal;

        \(ShaderProgram.supportSource)

        struct FrostfieldGraphResources {};

        \(program.functionDefinitions(prefix: name))

        struct GraphInput {
            float4 position [[position]];
            float3 normal [[user(frostfield_normal)]];
            float2 uv [[user(frostfield_uv)]];
            float3 worldPosition [[user(frostfield_world_position)]];
        };

        fragment float4 \(name)(
            GraphInput stageInput [[stage_in]],
            constant float4& parameters [[buffer(0)]]
        ) {
            FrostfieldSurfaceInput input = {};
            input.uv0 = stageInput.uv;
            input.normal = stageInput.normal;
            input.worldPosition = stageInput.worldPosition;
            input.parameters = parameters;
            FrostfieldGraphResources resources = {};
            (void)resources;
        \(try program.statements(for: [output], prefix: name))
            float3 finiteRGB = select(float3(0), n\(output), frostfield_graph_finite(n\(output)));
            float3 rgb = clamp(finiteRGB, float3(0), float3(65504.0f));
            float alpha = saturate(parameters.w);
            return float4(rgb * alpha, alpha);
        }
        """
    }
}
