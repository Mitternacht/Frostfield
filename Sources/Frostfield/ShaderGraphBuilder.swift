import Foundation
import simd

/// Any Swift value type that a shader graph can carry: the scalars and vectors of
/// `ShaderValueType`, plus `simd_float3x3` and `simd_float4x4`. Branches, loops,
/// selections and declared functions accept every graph value.
public protocol ShaderGraphValue {
    static var shaderType: MaterialGraph.ValueType { get }
}

/// `Float`, `SIMD2<Float>`, `SIMD3<Float>`, or `SIMD4<Float>`: the values that
/// constants, uniforms, arithmetic and swizzles accept. Matrices conform only to
/// `ShaderGraphValue` and use the matrix operations instead.
public protocol ShaderValueType: ShaderGraphValue {
    var shaderConstant: MaterialGraph.Node { get }
    static func shaderUniform(_ name: String, _ value: Self) -> ShaderUniform
}

/// The vector members of `ShaderValueType`.
public protocol ShaderVectorType: ShaderValueType {
    static var shaderComponents: String { get }
    static func shaderSplat(_ value: Float) -> Self
}

/// `simd_float3x3` and `simd_float4x4`, with the vector type of their columns.
/// Graphs can build, multiply, invert, slice and declare these matrices as uniforms.
public protocol ShaderMatrixType: ShaderGraphValue {
    /// The vector type of one column: `SIMD3<Float>` or `SIMD4<Float>`.
    associatedtype Column: ShaderVectorType
    /// The columns in order, for building a constant matrix node.
    var shaderColumns: [Column] { get }
}

extension simd_float3x3: ShaderGraphValue, ShaderMatrixType {
    public typealias Column = SIMD3<Float>
    public static var shaderType: MaterialGraph.ValueType { .matrix3 }
    public var shaderColumns: [SIMD3<Float>] { [columns.0, columns.1, columns.2] }
}

extension simd_float4x4: ShaderGraphValue, ShaderMatrixType {
    public typealias Column = SIMD4<Float>
    public static var shaderType: MaterialGraph.ValueType { .matrix4 }
    public var shaderColumns: [SIMD4<Float>] { [columns.0, columns.1, columns.2, columns.3] }
}

extension Int32: ShaderGraphValue, ShaderValueType {
    public static var shaderType: MaterialGraph.ValueType { .int }
    public var shaderConstant: MaterialGraph.Node { .integer(self) }
    public static func shaderUniform(_ name: String, _ value: Int32) -> ShaderUniform { ShaderUniform(name, value) }
}

extension Bool: ShaderGraphValue, ShaderValueType {
    public static var shaderType: MaterialGraph.ValueType { .bool }
    public var shaderConstant: MaterialGraph.Node { .boolean(self) }
    public static func shaderUniform(_ name: String, _ value: Bool) -> ShaderUniform { ShaderUniform(name, value) }
}

extension Float: ShaderGraphValue, ShaderValueType {
    public static var shaderType: MaterialGraph.ValueType { .scalar }
    public var shaderConstant: MaterialGraph.Node { .scalar(self) }
    public static func shaderUniform(_ name: String, _ value: Float) -> ShaderUniform { ShaderUniform(name, value) }
}

extension SIMD2: ShaderGraphValue, ShaderValueType, ShaderVectorType where Scalar == Float {
    public static var shaderType: MaterialGraph.ValueType { .vector2 }
    public var shaderConstant: MaterialGraph.Node { .vector2(self) }
    public static func shaderUniform(_ name: String, _ value: Self) -> ShaderUniform { ShaderUniform(name, value) }
    public static var shaderComponents: String { "xx" }
    public static func shaderSplat(_ value: Float) -> Self { Self(repeating: value) }
}

extension SIMD3: ShaderGraphValue, ShaderValueType, ShaderVectorType where Scalar == Float {
    public static var shaderType: MaterialGraph.ValueType { .vector3 }
    public var shaderConstant: MaterialGraph.Node { .color(self) }
    public static func shaderUniform(_ name: String, _ value: Self) -> ShaderUniform { ShaderUniform(name, value) }
    public static var shaderComponents: String { "xxx" }
    public static func shaderSplat(_ value: Float) -> Self { Self(repeating: value) }
}

extension SIMD4: ShaderGraphValue, ShaderValueType, ShaderVectorType where Scalar == Float {
    public static var shaderType: MaterialGraph.ValueType { .vector4 }
    public var shaderConstant: MaterialGraph.Node { .vector4(self) }
    public static func shaderUniform(_ name: String, _ value: Self) -> ShaderUniform { ShaderUniform(name, value) }
    public static var shaderComponents: String { "xxxx" }
    public static func shaderSplat(_ value: Float) -> Self { Self(repeating: value) }
}

/// Records nodes and declarations while Swift code describes a shader graph.
///
/// Values carry their component type, so the Swift compiler rejects mixing a
/// color with a scalar where the graph would reject it. Values belong to the
/// builder that made them: using one in another builder's graph makes that
/// graph's initializer throw. The finished graph still validates stage
/// availability, declarations, and limits.
public final class ShaderGraphBuilder {
    private(set) var nodes: [MaterialGraph.Node] = []
    private(set) var uniforms: [ShaderUniform] = []
    private(set) var textures: [ShaderTextureInput] = []
    private(set) var functions: [ShaderFunction] = []
    private(set) var structs: [ShaderStruct] = []
    private(set) var labels: [MaterialGraph.NodeID: String] = [:]
    private var constants: [MaterialGraph.Node: MaterialGraph.NodeID] = [:]
    /// Set when a node received an operand that this builder did not make.
    private var hasForeignOperand = false
    private var invalidDeclaration: String?

    init() {}

    func add<T: ShaderGraphValue>(_ node: MaterialGraph.Node, as type: T.Type = T.self) -> ShaderValue<T> {
        ShaderValue(id: append(node), builder: self)
    }

    /// Appends a node that reads the given values. Node IDs only identify nodes in
    /// the builder that made them, so an operand from another builder, or an
    /// argument that is not a builder value, is recorded and `checkOperands()` throws.
    func add<T: ShaderGraphValue>(
        _ node: MaterialGraph.Node, as type: T.Type = T.self, operands: [any ShaderArgument]
    ) -> ShaderValue<T> {
        if !operands.allSatisfy({ ($0 as? any ShaderBuilderOwned)?.owner === self }) {
            hasForeignOperand = true
        }
        return add(node)
    }

    /// Records an invalid declaration for nonthrowing node-building methods.
    func rejectDeclaration(_ reason: String) {
        if invalidDeclaration == nil { invalidDeclaration = reason }
    }

    func checkOperands() throws {
        if let invalidDeclaration { throw MaterialGraph.Failure.invalidDeclaration(invalidDeclaration) }
        guard !hasForeignOperand else {
            throw MaterialGraph.Failure.invalidDeclaration("A graph value belongs to a different builder.")
        }
    }

    func addStruct(_ structure: ShaderStruct) {
        structs.append(structure)
    }

    func append(_ node: MaterialGraph.Node) -> MaterialGraph.NodeID {
        switch node {
        case .scalar, .color, .vector2, .vector4, .input, .uniform, .parameter, .uv, .normal, .worldPosition:
            // Leaf values are shared so repeated literals and inputs emit once.
            if let existing = constants[node] { return existing }
            nodes.append(node)
            constants[node] = nodes.count - 1
        default:
            nodes.append(node)
        }
        return nodes.count - 1
    }

    func label(_ id: MaterialGraph.NodeID, _ name: String) {
        labels[id] = name
    }

    /// Records whether every operand belongs to this builder, as `add(_:as:operands:)` does.
    func noteOperands(_ operands: [any ShaderArgument]) {
        if !operands.allSatisfy({ ($0 as? any ShaderBuilderOwned)?.owner === self }) {
            hasForeignOperand = true
        }
    }

    /// Appends a placeholder that `fill(_:with:)` replaces, so loop nodes can be
    /// referenced by their body before the body exists.
    func reserve() -> MaterialGraph.NodeID {
        nodes.append(.scalar(0))
        return nodes.count - 1
    }

    func fill(_ id: MaterialGraph.NodeID, with node: MaterialGraph.Node) {
        nodes[id] = node
    }

    var declarations: ShaderDeclarations {
        ShaderDeclarations(uniforms: uniforms, textures: textures, functions: functions, structs: structs, labels: labels)
    }

    // MARK: Values

    public func constant<T: ShaderValueType>(_ value: T) -> ShaderValue<T> { add(value.shaderConstant) }

    /// A constant matrix built from constant columns.
    public func constant<M: ShaderMatrixType>(_ value: M) -> ShaderValue<M> {
        add(.matrix(value.shaderColumns.map { constant($0).id }))
    }

    private func input<T: ShaderGraphValue>(_ input: MaterialGraph.Input) -> ShaderValue<T> { add(.input(input)) }

    /// Primary texture coordinates. Surface and vertex stages.
    public var uv0: ShaderValue<SIMD2<Float>> { input(.uv0) }
    /// Secondary texture coordinates. Surface and vertex stages.
    public var uv1: ShaderValue<SIMD2<Float>> { input(.uv1) }
    public var worldPosition: ShaderValue<SIMD3<Float>> { input(.worldPosition) }
    /// Unit world-space shading normal.
    public var normal: ShaderValue<SIMD3<Float>> { input(.normal) }
    public var tangent: ShaderValue<SIMD3<Float>> { input(.tangent) }
    /// Unit direction toward the viewer.
    public var viewDirection: ShaderValue<SIMD3<Float>> { input(.viewDirection) }
    /// The node's `materialParameters`.
    public var parameters: ShaderValue<SIMD4<Float>> { input(.parameters) }
    /// The authored material's linear base color before the graph runs.
    public var baseColor: ShaderValue<SIMD3<Float>> { input(.baseColor) }
    public var metallic: ShaderValue<Float> { input(.metallic) }
    public var roughness: ShaderValue<Float> { input(.roughness) }
    public var emission: ShaderValue<SIMD3<Float>> { input(.emission) }
    /// Undeformed object-space position. Vertex stage.
    public var objectPosition: ShaderValue<SIMD3<Float>> { input(.objectPosition) }
    /// Undeformed unit object-space normal. Vertex stage.
    public var objectNormal: ShaderValue<SIMD3<Float>> { input(.objectNormal) }
    /// The pixel center in top-left normalized coordinates of the viewport (Three.js
    /// `viewportUV`). Post-process and raster surface stages.
    public var screenUV: ShaderValue<SIMD2<Float>> { input(.screenUV) }
    /// One over the output dimensions. Post-process and texture compute stages.
    public var texelSize: ShaderValue<SIMD2<Float>> { input(.texelSize) }
    /// Generated texel center in top-left normalized coordinates. Texture compute stage.
    public var gridUV: ShaderValue<SIMD2<Float>> { input(.gridUV) }
    public var elementIndex: ShaderValue<Float> { input(.elementIndex) }
    public var elementCount: ShaderValue<Float> { input(.elementCount) }
    /// Unit bitangent along increasing V of UV0, completing the frame with `tangent`
    /// and `normal`. Surface stage.
    public var bitangent: ShaderValue<SIMD3<Float>> { input(.bitangent) }
    /// The output or viewport size in pixels. Post-process and raster surface stages.
    public var viewportSize: ShaderValue<SIMD2<Float>> { input(.viewportSize) }
    /// Pixel coordinates of the output pixel center within the viewport:
    /// `screenUV × viewportSize`, with centers at half-integers (Three.js
    /// `viewportCoordinate`). Post-process and raster surface stages.
    public var fragmentCoordinate: ShaderValue<SIMD2<Float>> { screenUV * viewportSize }
    /// Render target pixel coordinates of the pixel center, top-left origin (Three.js
    /// `screenCoordinate`). Post-process and raster surface stages.
    public var screenCoordinate: ShaderValue<SIMD2<Float>> { input(.screenCoordinate) }
    /// The render target size in pixels (Three.js `screenSize`). Post-process and raster surface stages.
    public var screenSize: ShaderValue<SIMD2<Float>> { input(.screenSize) }
    /// Seconds on the graph clock, set with `ShaderUniforms.setFrame(_:)` or `advanceFrame(by:)`.
    public var time: ShaderValue<Float> { input(.time) }
    /// Seconds since the previous frame on the graph clock.
    public var deltaTime: ShaderValue<Float> { input(.deltaTime) }
    /// The graph clock's frame counter modulo 2²⁴.
    public var frame: ShaderValue<Float> { input(.frame) }
    /// World-space camera position. Surface stage.
    public var cameraPosition: ShaderValue<SIMD3<Float>> { input(.cameraPosition) }
    /// The camera's near clip distance. Surface stage.
    public var cameraNear: ShaderValue<Float> { input(.cameraNear) }
    /// The camera's far clip distance. Surface stage.
    public var cameraFar: ShaderValue<Float> { input(.cameraFar) }
    /// Fraction of directional sunlight visible at this raster surface. Set the
    /// material's `receivesShadows` and supply a scene directional shadow map.
    /// Unshadowed raster variants return one. Coverage outputs cannot read it.
    public var directionalShadowVisibility: ShaderValue<Float> { input(.directionalShadowVisibility) }
    /// The world-to-camera matrix. Surface stage.
    public var viewMatrix: ShaderValue<simd_float4x4> { input(.viewMatrix) }
    /// The camera-to-clip matrix. Surface stage.
    public var projectionMatrix: ShaderValue<simd_float4x4> { input(.projectionMatrix) }
    /// The mesh node's object-to-world matrix. Surface stage.
    public var modelMatrix: ShaderValue<simd_float4x4> { input(.modelMatrix) }
    /// The inverse transpose of the model matrix's upper 3 × 3. Surface stage.
    public var normalMatrix: ShaderValue<simd_float3x3> { input(.normalMatrix) }

    /// One of the current buffer element's four float4 fields. Buffer compute stage.
    public func elementField(_ index: Int) -> ShaderValue<SIMD4<Float>> { input(.elementField(index)) }

    /// Declares, or reuses by name and type, a uniform that can be edited without recompiling.
    /// A second declaration of the same name and type returns the first one and keeps its default.
    public func uniform<T: ShaderValueType>(_ name: String, _ defaultValue: T) -> ShaderValue<T> {
        declareUniform(T.shaderUniform(name, defaultValue))
    }

    /// A 3 × 3 column-major uniform, as a Three.js `mat3` uniform. Edit it with `ShaderUniforms.set(_:to:)`.
    public func uniform(_ name: String, _ defaultValue: simd_float3x3) -> ShaderValue<simd_float3x3> {
        declareUniform(ShaderUniform(name, defaultValue))
    }

    /// A 4 × 4 column-major uniform, as a Three.js `mat4` uniform. Edit it with `ShaderUniforms.set(_:to:)`.
    public func uniform(_ name: String, _ defaultValue: simd_float4x4) -> ShaderValue<simd_float4x4> {
        declareUniform(ShaderUniform(name, defaultValue))
    }

    private func declareUniform<T: ShaderGraphValue>(_ declaration: ShaderUniform) -> ShaderValue<T> {
        if let index = uniforms.firstIndex(where: { $0.name == declaration.name && $0.type == declaration.type }) {
            return add(.uniform(index))
        }
        uniforms.append(declaration)
        return add(.uniform(uniforms.count - 1))
    }

    /// Declares, or reuses by name, a texture to bind when preparing the graph.
    public func texture(
        _ name: String, content: ShaderTextureInput.Content = .color,
        addressing: ShaderTextureInput.Addressing = .repeat, filtering: ShaderTextureInput.Filtering = .linear
    ) -> ShaderTexture {
        declareTexture(name, content: content, addressing: addressing, filtering: filtering, shape: .twoDimensional)
    }

    /// Declares, or reuses by name, a cube map. Compute graphs sample it by direction at mip 0.
    public func cube(
        _ name: String, content: ShaderTextureInput.Content = .color,
        addressing: ShaderTextureInput.Addressing = .clampToEdge, filtering: ShaderTextureInput.Filtering = .linear
    ) -> ShaderTexture {
        declareTexture(name, content: content, addressing: addressing, filtering: filtering, shape: .cube)
    }

    /// Declares, or reuses by name, a 3D texture. Compute graphs sample it at normalized
    /// coordinates and mip 0.
    public func texture3D(
        _ name: String, content: ShaderTextureInput.Content = .data,
        addressing: ShaderTextureInput.Addressing = .clampToEdge, filtering: ShaderTextureInput.Filtering = .linear
    ) -> ShaderTexture {
        declareTexture(name, content: content, addressing: addressing, filtering: filtering, shape: .texture3D)
    }

    private func declareTexture(
        _ name: String, content: ShaderTextureInput.Content, addressing: ShaderTextureInput.Addressing,
        filtering: ShaderTextureInput.Filtering, shape: ShaderTextureInput.Shape
    ) -> ShaderTexture {
        if let slot = textures.firstIndex(where: { $0.name == name }) {
            if textures[slot].shape != shape {
                rejectDeclaration("Texture \(name) is already declared with a different shape.")
            }
            return ShaderTexture(slot: slot, builder: self)
        }
        textures.append(ShaderTextureInput(
            name, content: content, addressing: addressing, filtering: filtering, shape: shape
        ))
        return ShaderTexture(slot: textures.count - 1, builder: self)
    }

    /// Declares a Metal function written by the caller. See `ShaderFunction`.
    public func function<R: ShaderGraphValue>(
        _ name: String, returns: R.Type, parameters: [ShaderFunction.Parameter], body: String
    ) -> ShaderFunctionReference<R> {
        if let index = functions.firstIndex(where: { $0.name == name }) {
            return ShaderFunctionReference(index: index, builder: self)
        }
        functions.append(ShaderFunction(name: name, parameters: parameters, returns: R.shaderType, body: body))
        return ShaderFunctionReference(index: functions.count - 1, builder: self)
    }

    /// Declares a Metal function that takes and returns graph structs. The body may name each
    /// declared struct by its Swift name; that name is a typedef for the generated Metal type.
    public func function(
        _ name: String, returns structure: ShaderStruct, parameters: [ShaderFunction.Parameter], body: String
    ) -> ShaderStructFunction {
        if let index = functions.firstIndex(where: { $0.name == name }) {
            let existing = functions[index]
            if existing.returnType != .structure || existing.returnStructure != structure.index {
                rejectDeclaration("Function \(name) is already declared with a different type.")
            }
            return ShaderStructFunction(index: index, structure: structure.index, builder: self)
        }
        functions.append(ShaderFunction(
            name: name, parameters: parameters, returns: .structure, returnStructure: structure.index, body: body
        ))
        return ShaderStructFunction(index: functions.count - 1, structure: structure.index, builder: self)
    }

    /// Premultiplied linear RGBA at top-left normalized coordinates, sampled
    /// bilinearly. Reads the source image in post-processing and a
    /// `SceneViewportSnapshot` in raster surface graphs.
    public func sceneColor(at coordinates: ShaderValue<SIMD2<Float>>) -> ShaderValue<SIMD4<Float>> {
        add(.sampleScene(coordinates: coordinates.id), operands: [coordinates])
    }

    /// Samples scene color at an explicit mip level. Fractional levels blend
    /// adjacent mips. Levels outside the available range clamp to that range.
    public func sceneColor(at coordinates: ShaderValue<SIMD2<Float>>, mipLevel: ShaderValue<Float>) -> ShaderValue<SIMD4<Float>> {
        add(.sampleSceneLevel(coordinates: coordinates.id, level: mipLevel.id), operands: [coordinates, mipLevel])
    }

    /// Stored 0...1 Metal depth at top-left normalized coordinates, sampled
    /// nearest. Reads the source depth in post-processing and a
    /// `SceneViewportSnapshot` in raster surface graphs.
    public func sceneDepth(at coordinates: ShaderValue<SIMD2<Float>>) -> ShaderValue<Float> {
        add(.sampleDepth(coordinates: coordinates.id), operands: [coordinates])
    }

    /// Returns the undistorted attachment UV if a nearer opaque object occupies
    /// the proposed refraction sample. Requires scene depth from the same view,
    /// viewport and depth mode. Coordinates are normalized over the whole target.
    public func viewportSafeUV(
        at coordinates: ShaderValue<SIMD2<Float>>, fallback: ShaderValue<SIMD2<Float>>? = nil
    ) -> ShaderValue<SIMD2<Float>> {
        let original = fallback ?? attachmentUV(at: screenCoordinate)
        return add(.viewportSafeUV(coordinates: coordinates.id, fallback: original.id), operands: [coordinates, original])
    }

    func attachmentUV(at pixel: ShaderValue<SIMD2<Float>>) -> ShaderValue<SIMD2<Float>> {
        let size = screenSize
        return vector(pixel.x / size.x, pixel.y / size.y)
    }

    /// Multiplies opacity by a soft intersection fade for a raster surface.
    /// The default depth sample uses the whole attachment. An explicit depth
    /// expression can select different UVs. Use depth from the same view,
    /// viewport depth range and depth mode as the particle.
    public func softParticleOpacity(
        opacity: ShaderValue<Float>? = nil, distance: ShaderValue<Float>? = nil,
        contrast: ShaderValue<Float>? = nil, viewportDepth: ShaderValue<Float>? = nil
    ) -> ShaderValue<Float> {
        let opacity = opacity ?? constant(Float(1))
        let distance = distance ?? constant(Float(1))
        let contrast = contrast ?? constant(Float(2))
        let depth = viewportDepth ?? sceneDepth(
            at: vector(screenCoordinate.x / screenSize.x, screenCoordinate.y / screenSize.y)
        )
        return add(
            .softParticleOpacity(opacity: opacity.id, distance: distance.id, contrast: contrast.id, depth: depth.id),
            operands: [opacity, distance, contrast, depth]
        )
    }

    // MARK: Procedural functions

    /// Lattice noise in approximately 0...1.
    public func noise(_ position: ShaderValue<SIMD3<Float>>, kind: MaterialGraph.NoiseKind = .gradient) -> ShaderValue<Float> {
        add(.latticeNoise(kind, position: position.id), operands: [position])
    }

    /// Fractal Brownian motion: summed octaves normalized to approximately 0...1.
    public func fractalNoise(
        _ position: ShaderValue<SIMD3<Float>>, kind: MaterialGraph.NoiseKind = .gradient,
        octaves: Int = 5, lacunarity: Float = 2, gain: Float = 0.5
    ) -> ShaderValue<Float> {
        add(
            .fractalNoise(kind, position: position.id, octaves: octaves, lacunarity: lacunarity, gain: gain),
            operands: [position]
        )
    }

    /// Worley noise: nearest feature distance in x and a 0...1 cell identifier in y.
    public func cellularNoise(_ position: ShaderValue<SIMD3<Float>>) -> ShaderValue<SIMD2<Float>> {
        add(.cellularNoise(position: position.id), operands: [position])
    }

    /// Polynomial-gradient simplex noise as used by Three.js's TSL curl field.
    public func simplexNoise(_ position: ShaderValue<SIMD3<Float>>) -> ShaderValue<Float> {
        add(.simplexNoise(position: position.id), operands: [position])
    }

    /// A three-dimensional curl field from central differences of simplex noise.
    public func curlNoise(_ position: ShaderValue<SIMD3<Float>>) -> ShaderValue<SIMD3<Float>> {
        add(.curlNoise(position: position.id), operands: [position])
    }

    /// Animated squared Voronoi distance, using nine neighboring cells.
    public func animatedVoronoi(_ position: ShaderValue<SIMD2<Float>>, time: ShaderValue<Float>) -> ShaderValue<Float> {
        add(.animatedVoronoi(position: position.id, time: time.id), operands: [position, time])
    }

    /// Animated squared Voronoi distance, using 27 neighboring cells.
    public func animatedVoronoi(_ position: ShaderValue<SIMD3<Float>>, time: ShaderValue<Float>) -> ShaderValue<Float> {
        add(.animatedVoronoi(position: position.id, time: time.id), operands: [position, time])
    }

    /// Chooses `ifTrue` where the condition exceeds 0.5.
    public func select<T: ShaderGraphValue>(
        _ condition: ShaderValue<Float>, _ ifTrue: ShaderValue<T>, _ ifFalse: ShaderValue<T>
    ) -> ShaderValue<T> {
        add(.select(condition: condition.id, ifTrue: ifTrue.id, ifFalse: ifFalse.id), operands: [condition, ifTrue, ifFalse])
    }

    /// 0 where `value` is below `edge`, else 1.
    public func step<T: ShaderValueType>(_ edge: ShaderValue<T>, _ value: ShaderValue<T>) -> ShaderValue<T> {
        add(.binary(.step, edge.id, value.id), operands: [edge, value])
    }

    public func vector(_ x: ShaderValue<Float>, _ y: ShaderValue<Float>) -> ShaderValue<SIMD2<Float>> {
        add(.combine([x.id, y.id]), operands: [x, y])
    }

    public func vector(_ x: ShaderValue<Float>, _ y: ShaderValue<Float>, _ z: ShaderValue<Float>) -> ShaderValue<SIMD3<Float>> {
        add(.combine([x.id, y.id, z.id]), operands: [x, y, z])
    }

    public func vector(_ xy: ShaderValue<SIMD2<Float>>, _ z: ShaderValue<Float>) -> ShaderValue<SIMD3<Float>> {
        add(.combine([xy.id, z.id]), operands: [xy, z])
    }

    public func vector(_ xyz: ShaderValue<SIMD3<Float>>, _ w: ShaderValue<Float>) -> ShaderValue<SIMD4<Float>> {
        add(.combine([xyz.id, w.id]), operands: [xyz, w])
    }

    public func vector(
        _ x: ShaderValue<Float>, _ y: ShaderValue<Float>, _ z: ShaderValue<Float>, _ w: ShaderValue<Float>
    ) -> ShaderValue<SIMD4<Float>> {
        add(.combine([x.id, y.id, z.id, w.id]), operands: [x, y, z, w])
    }

    /// An object-space offset along the undeformed normal, for `positionOffset`.
    public func displacement(_ amount: ShaderValue<Float>) -> ShaderValue<SIMD3<Float>> {
        objectNormal * amount
    }
}

/// A typed handle to one node of a graph under construction.
public struct ShaderValue<Value: ShaderGraphValue> {
    public let id: MaterialGraph.NodeID
    let builder: ShaderGraphBuilder

    init(id: MaterialGraph.NodeID, builder: ShaderGraphBuilder) {
        self.id = id
        self.builder = builder
    }

    /// Appends a node to this value's builder. Pass every other value the node
    /// reads as an operand, so values from another builder are rejected.
    func make<T: ShaderGraphValue>(
        _ node: MaterialGraph.Node, as type: T.Type = T.self, with operands: any ShaderArgument...
    ) -> ShaderValue<T> {
        builder.add(node, operands: operands)
    }

    /// Names this node in generated Metal comments and diagnostics.
    @discardableResult
    public func named(_ label: String) -> ShaderValue<Value> {
        builder.label(id, label)
        return self
    }
}

extension ShaderValue where Value: ShaderValueType {
    func lift(_ value: Value) -> ShaderValue<Value> { builder.constant(value) }

    private func unary(_ operation: MaterialGraph.UnaryOperation) -> ShaderValue<Value> { make(.unary(operation, id)) }

    private func binary(_ operation: MaterialGraph.BinaryOperation, _ other: ShaderValue<Value>) -> ShaderValue<Value> {
        make(.binary(operation, id, other.id), with: other)
    }

    public func abs() -> Self { unary(.abs) }
    public func sign() -> Self { unary(.sign) }
    public func floor() -> Self { unary(.floor) }
    public func ceil() -> Self { unary(.ceil) }
    public func rounded() -> Self { unary(.round) }
    public func fract() -> Self { unary(.fract) }
    public func squareRoot() -> Self { unary(.sqrt) }
    public func inverseSquareRoot() -> Self { unary(.inverseSqrt) }
    public func exp() -> Self { unary(.exp) }
    public func exp2() -> Self { unary(.exp2) }
    public func log() -> Self { unary(.log) }
    public func log2() -> Self { unary(.log2) }
    public func sin() -> Self { unary(.sin) }
    public func cos() -> Self { unary(.cos) }
    public func tan() -> Self { unary(.tan) }
    public func asin() -> Self { unary(.asin) }
    public func acos() -> Self { unary(.acos) }
    public func atan() -> Self { unary(.atan) }
    public func oneMinus() -> Self { unary(.oneMinus) }
    public func saturated() -> Self { unary(.saturate) }
    public func length() -> ShaderValue<Float> { make(.unary(.length, id)) }

    public func min(_ other: Self) -> Self { binary(.minimum, other) }
    public func max(_ other: Self) -> Self { binary(.maximum, other) }
    public func pow(_ exponent: Self) -> Self { binary(.power, exponent) }
    public func mod(_ divisor: Self) -> Self { binary(.modulo, divisor) }
    public func atan2(_ x: Self) -> Self { binary(.atan2, x) }
    public func greaterThan(_ other: Self) -> Self { binary(.greaterThan, other) }
    public func lessThan(_ other: Self) -> Self { binary(.lessThan, other) }
    public func min(_ other: Value) -> Self { binary(.minimum, lift(other)) }
    public func max(_ other: Value) -> Self { binary(.maximum, lift(other)) }
    public func pow(_ exponent: Value) -> Self { binary(.power, lift(exponent)) }
    public func mod(_ divisor: Value) -> Self { binary(.modulo, lift(divisor)) }

    public func clamped(_ minimum: Self, _ maximum: Self) -> Self {
        make(.clamp(id, minimum: minimum.id, maximum: maximum.id), with: minimum, maximum)
    }

    public func clamped(_ minimum: Float, _ maximum: Float) -> Self {
        make(.clamp(id, minimum: builder.constant(minimum).id, maximum: builder.constant(maximum).id))
    }

    /// Interpolates toward `other` by a factor clamped to 0...1.
    public func mix(_ other: Self, by factor: ShaderValue<Float>) -> Self {
        make(.mix(id, other.id, factor: factor.id), with: other, factor)
    }

    public func mix(_ other: Self, by factor: Float) -> Self { mix(other, by: builder.constant(factor)) }

    public func remap(_ fromMinimum: Float, _ fromMaximum: Float, _ toMinimum: Float, _ toMaximum: Float) -> Self {
        let bounds = [fromMinimum, fromMaximum, toMinimum, toMaximum].map { builder.constant($0).id }
        return make(.remap(id, fromMinimum: bounds[0], fromMaximum: bounds[1], toMinimum: bounds[2], toMaximum: bounds[3]))
    }

    /// Selects components with `xyzw` or `rgba` letters. Validation checks the letters and count.
    public func swizzle<T: ShaderValueType>(_ components: String, as type: T.Type) -> ShaderValue<T> {
        make(.swizzle(id, components))
    }
}

extension ShaderValue where Value == Float {
    public func smoothstep(_ edge0: Float, _ edge1: Float) -> Self {
        make(.smoothstep(edge0: builder.constant(edge0).id, edge1: builder.constant(edge1).id, value: id))
    }

    public func smoothstep(_ edge0: Self, _ edge1: Self) -> Self {
        make(.smoothstep(edge0: edge0.id, edge1: edge1.id, value: id), with: edge0, edge1)
    }

    public var xx: ShaderValue<SIMD2<Float>> { make(.swizzle(id, "xx")) }
    public var xxx: ShaderValue<SIMD3<Float>> { make(.swizzle(id, "xxx")) }
    public var xxxx: ShaderValue<SIMD4<Float>> { make(.swizzle(id, "xxxx")) }
}

extension ShaderValue where Value: ShaderVectorType {
    public func normalized() -> Self { make(.unary(.normalize, id)) }
    public func dot(_ other: Self) -> ShaderValue<Float> { make(.binary(.dot, id, other.id), with: other) }
    public func distance(to other: Self) -> ShaderValue<Float> { make(.binary(.distance, id, other.id), with: other) }
    /// Reflects this incident direction about the normalized normal.
    public func reflect(_ normal: Self) -> Self { make(.binary(.reflect, id, normal.id), with: normal) }

    /// Refracts this unit incident direction through a unit normal. Eta is the
    /// incident medium's index divided by the transmitted medium's index.
    /// Total internal reflection produces the zero vector.
    public func refract(_ normal: Self, eta: ShaderValue<Float>) -> Self {
        let projection = dot(normal)
        let discriminant = 1 - eta * eta * (1 - projection * projection)
        let transmitted = self * eta - normal * (eta * projection + discriminant.squareRoot())
        let zero = builder.constant(Value.shaderSplat(0))
        return builder.select(discriminant.lessThan(builder.constant(Float(0))), zero, transmitted)
    }

    public func refract(_ normal: Self, eta: Float) -> Self { refract(normal, eta: builder.constant(eta)) }

    /// Keeps this normal when the reference normal opposes the incident vector.
    /// Otherwise returns its negation, including at a zero dot product.
    public func faceForward(incident: Self, referenceNormal: Self) -> Self {
        let facesIncident = referenceNormal.dot(incident).lessThan(builder.constant(Float(0)))
        return builder.select(facesIncident, self, -self)
    }

    public var x: ShaderValue<Float> { make(.swizzle(id, "x")) }
    public var y: ShaderValue<Float> { make(.swizzle(id, "y")) }

    func splat(_ scalar: ShaderValue<Float>) -> Self { make(.swizzle(scalar.id, Value.shaderComponents), with: scalar) }
}

extension ShaderValue where Value == SIMD2<Float> {
    public var yx: Self { make(.swizzle(id, "yx")) }
    /// The signed perpendicular component of the two-dimensional cross product.
    public func cross(_ other: Self) -> ShaderValue<Float> { x * other.y - y * other.x }
}

extension ShaderValue where Value == SIMD3<Float> {
    public var z: ShaderValue<Float> { make(.swizzle(id, "z")) }
    public var xy: ShaderValue<SIMD2<Float>> { make(.swizzle(id, "xy")) }
    public var xz: ShaderValue<SIMD2<Float>> { make(.swizzle(id, "xz")) }
    public var yz: ShaderValue<SIMD2<Float>> { make(.swizzle(id, "yz")) }
    public var r: ShaderValue<Float> { make(.swizzle(id, "x")) }
    public var g: ShaderValue<Float> { make(.swizzle(id, "y")) }
    public var b: ShaderValue<Float> { make(.swizzle(id, "z")) }
    public func cross(_ other: Self) -> Self { make(.binary(.cross, id, other.id), with: other) }
    public func luminance() -> ShaderValue<Float> { make(.unary(.luminance, id)) }
    public func hsvToRGB() -> Self { make(.unary(.hsvToRGB, id)) }
    public func rgbToHSV() -> Self { make(.unary(.rgbToHSV, id)) }
    public func srgbToLinear() -> Self { make(.unary(.srgbToLinear, id)) }
    public func linearToSRGB() -> Self { make(.unary(.linearToSRGB, id)) }
}

extension ShaderValue where Value == SIMD4<Float> {
    public var z: ShaderValue<Float> { make(.swizzle(id, "z")) }
    public var w: ShaderValue<Float> { make(.swizzle(id, "w")) }
    public var xy: ShaderValue<SIMD2<Float>> { make(.swizzle(id, "xy")) }
    public var zw: ShaderValue<SIMD2<Float>> { make(.swizzle(id, "zw")) }
    public var xyz: ShaderValue<SIMD3<Float>> { make(.swizzle(id, "xyz")) }
    public var rgb: ShaderValue<SIMD3<Float>> { make(.swizzle(id, "xyz")) }
    public var r: ShaderValue<Float> { make(.swizzle(id, "x")) }
    public var g: ShaderValue<Float> { make(.swizzle(id, "y")) }
    public var b: ShaderValue<Float> { make(.swizzle(id, "z")) }
    public var a: ShaderValue<Float> { make(.swizzle(id, "w")) }
}

// MARK: - Operators

public func + <T: ShaderValueType>(left: ShaderValue<T>, right: ShaderValue<T>) -> ShaderValue<T> { left.make(.add(left.id, right.id), with: right) }
public func - <T: ShaderValueType>(left: ShaderValue<T>, right: ShaderValue<T>) -> ShaderValue<T> { left.make(.subtract(left.id, right.id), with: right) }
public func * <T: ShaderValueType>(left: ShaderValue<T>, right: ShaderValue<T>) -> ShaderValue<T> { left.make(.multiply(left.id, right.id), with: right) }
public func + <T: ShaderValueType>(left: ShaderValue<T>, right: T) -> ShaderValue<T> { left + left.lift(right) }
public func - <T: ShaderValueType>(left: ShaderValue<T>, right: T) -> ShaderValue<T> { left - left.lift(right) }
public func * <T: ShaderValueType>(left: ShaderValue<T>, right: T) -> ShaderValue<T> { left * left.lift(right) }
public func + <T: ShaderValueType>(left: T, right: ShaderValue<T>) -> ShaderValue<T> { right.lift(left) + right }
public func - <T: ShaderValueType>(left: T, right: ShaderValue<T>) -> ShaderValue<T> { right.lift(left) - right }
public func * <T: ShaderValueType>(left: T, right: ShaderValue<T>) -> ShaderValue<T> { right.lift(left) * right }
public prefix func - <T: ShaderValueType>(value: ShaderValue<T>) -> ShaderValue<T> { value.make(.unary(.negate, value.id)) }

/// Divides by a scalar. A zero or nonfinite divisor yields zero.
public func / <T: ShaderValueType>(left: ShaderValue<T>, right: ShaderValue<Float>) -> ShaderValue<T> {
    left.make(.divide(left.id, right.id), with: right)
}

public func / <T: ShaderValueType>(left: ShaderValue<T>, right: Float) -> ShaderValue<T> { left / left.builder.constant(right) }

public func * <V: ShaderVectorType>(left: ShaderValue<V>, right: ShaderValue<Float>) -> ShaderValue<V> {
    left.make(.multiply(left.id, right.id), with: right)
}

public func * <V: ShaderVectorType>(left: ShaderValue<Float>, right: ShaderValue<V>) -> ShaderValue<V> {
    right.make(.multiply(left.id, right.id), with: left)
}

public func * <V: ShaderVectorType>(left: ShaderValue<V>, right: Float) -> ShaderValue<V> {
    left * left.builder.constant(right)
}

public func * <V: ShaderVectorType>(left: Float, right: ShaderValue<V>) -> ShaderValue<V> {
    right.builder.constant(left) * right
}

public func + <V: ShaderVectorType>(left: ShaderValue<V>, right: ShaderValue<Float>) -> ShaderValue<V> {
    left + left.splat(right)
}

public func - <V: ShaderVectorType>(left: ShaderValue<V>, right: ShaderValue<Float>) -> ShaderValue<V> {
    left - left.splat(right)
}

public func + <V: ShaderVectorType>(left: ShaderValue<V>, right: Float) -> ShaderValue<V> {
    left + left.lift(V.shaderSplat(right))
}

public func - <V: ShaderVectorType>(left: ShaderValue<V>, right: Float) -> ShaderValue<V> {
    left - left.lift(V.shaderSplat(right))
}

public func - <V: ShaderVectorType>(left: Float, right: ShaderValue<V>) -> ShaderValue<V> {
    right.lift(V.shaderSplat(left)) - right
}

/// A declared texture under construction.
public struct ShaderTexture {
    public let slot: Int
    let builder: ShaderGraphBuilder

    /// RGBA at top-left normalized coordinates. Vertex and compute stages sample the base level.
    public func sample(_ coordinates: ShaderValue<SIMD2<Float>>) -> ShaderValue<SIMD4<Float>> {
        require2D("A cube map is sampled with a direction.")
        return builder.add(.sampleTexture(slot: slot, coordinates: coordinates.id), operands: [coordinates])
    }

    /// RGBA at an explicit mip level. A nonfinite level reads level 0, and the level is clamped
    /// to the last mip. This sample may run inside a branch.
    public func sample(
        _ coordinates: ShaderValue<SIMD2<Float>>, level: ShaderValue<Float>
    ) -> ShaderValue<SIMD4<Float>> {
        require2D("A cube map has no 2D mip sample.")
        return builder.add(
            .sampleTextureLevel(slot: slot, coordinates: coordinates.id, level: level.id),
            operands: [coordinates, level]
        )
    }

    /// RGBA from explicit UV derivatives. The mip is log2 of the larger footprint in
    /// texels, then clamped to the last mip. A nonfinite derivative reads level 0.
    /// This sample may run inside a branch.
    public func sample(
        _ coordinates: ShaderValue<SIMD2<Float>>,
        ddx: ShaderValue<SIMD2<Float>>,
        ddy: ShaderValue<SIMD2<Float>>
    ) -> ShaderValue<SIMD4<Float>> {
        require2D("A cube map has no UV derivatives.")
        return builder.add(
            .sampleTextureGradient(slot: slot, coordinates: coordinates.id, ddx: ddx.id, ddy: ddy.id),
            operands: [coordinates, ddx, ddy]
        )
    }

    /// RGBA of a cube map along a direction, or of a 3D texture at normalized coordinates.
    /// Compute graphs read mip 0. A nonfinite or zero cube direction reads the +X face.
    /// A nonfinite 3D coordinate reads zero.
    public func sample(_ coordinates: ShaderValue<SIMD3<Float>>) -> ShaderValue<SIMD4<Float>> {
        let shape = builder.textures.indices.contains(slot) ? builder.textures[slot].shape : .twoDimensional
        switch shape {
        case .cube:
            return builder.add(.sampleCube(slot: slot, direction: coordinates.id), operands: [coordinates])
        case .texture3D:
            return builder.add(.sampleTexture3D(slot: slot, coordinates: coordinates.id), operands: [coordinates])
        case .twoDimensional:
            builder.rejectDeclaration("A 2D texture is sampled with a vector2.")
            return builder.add(.sampleCube(slot: slot, direction: coordinates.id), operands: [coordinates])
        }
    }

    private func require2D(_ cubeReason: String) {
        guard builder.textures.indices.contains(slot) else { return }
        switch builder.textures[slot].shape {
        case .twoDimensional:
            return
        case .cube:
            builder.rejectDeclaration(cubeReason)
        case .texture3D:
            builder.rejectDeclaration("A 3D texture is sampled with a vector3.")
        }
    }

    /// Decodes a tangent-space normal map texel from 0...1 storage to -1...1.
    public func sampleNormal(_ coordinates: ShaderValue<SIMD2<Float>>, strength: Float = 1) -> ShaderValue<SIMD3<Float>> {
        let encoded = sample(coordinates).xyz * 2 - 1
        let xy = encoded.xy * strength
        return builder.vector(xy, encoded.z)
    }

    /// A tangent-space normal from the red channel of a linear height texture.
    /// Connect to the surface graph's normal output. The selected mesh UV set
    /// and affine transform apply to the raster pixel or path ray footprint.
    /// The output uses the surface graph's UV0 tangent frame.
    public func bumpMap(
        coordinateSet: MeshTextureCoordinateSet = .primary,
        transform: PhysicallyBasedMaterial.TextureTransform = .identity,
        scale: ShaderValue<Float>? = nil
    ) -> ShaderValue<SIMD3<Float>> {
        let amount = scale ?? builder.constant(Float(1))
        return builder.add(
            .bumpMap(slot: slot, coordinateSet: coordinateSet, transform: transform, scale: amount.id),
            operands: [amount]
        )
    }
}

/// A declared function under construction.
public struct ShaderFunctionReference<Result: ShaderGraphValue> {
    public let index: Int
    let builder: ShaderGraphBuilder

    /// Calls the function. Validation checks the argument count and types.
    public func callAsFunction(_ arguments: any ShaderArgument...) -> ShaderValue<Result> {
        builder.add(.function(index, arguments: arguments.map(\.shaderNodeID)), operands: arguments)
    }
}

/// A declared function whose result is a struct.
public struct ShaderStructFunction {
    public let index: Int
    /// Index of the struct this function returns.
    public let structure: Int
    let builder: ShaderGraphBuilder

    /// Calls the function. Validation checks the argument count and struct identity.
    public func callAsFunction(_ arguments: any ShaderArgument...) -> ShaderStructValue {
        builder.noteOperands(arguments)
        return ShaderStructValue(
            id: builder.append(.function(index, arguments: arguments.map(\.shaderNodeID))),
            structure: structure, builder: builder
        )
    }
}

/// A graph value that can be passed to a declared function.
public protocol ShaderArgument {
    var shaderNodeID: MaterialGraph.NodeID { get }
}

extension ShaderValue: ShaderArgument {
    public var shaderNodeID: MaterialGraph.NodeID { id }
}

// MARK: - Surface graphs

extension SurfaceShaderGraph {
    /// Typed outputs for `SurfaceShaderGraph(building:)`.
    public struct Result {
        public var baseColor: ShaderValue<SIMD3<Float>>?
        public var metallic: ShaderValue<Float>?
        public var roughness: ShaderValue<Float>?
        public var emission: ShaderValue<SIMD3<Float>>?
        public var transmissionMask: ShaderValue<Float>?
        /// A tangent-space normal.
        public var normal: ShaderValue<SIMD3<Float>>?
        public var occlusion: ShaderValue<Float>?
        /// An object-space vertex offset, evaluated in the vertex stage.
        public var positionOffset: ShaderValue<SIMD3<Float>>?
        /// A point size in the material's units, evaluated per point.
        public var pointSize: ShaderValue<Float>?
        /// A line width in the material's units. Hairlines ignore it.
        public var lineWidth: ShaderValue<Float>?
        public var opacity: ShaderValue<Float>?
        public var alphaTest: ShaderValue<Float>?
        public var mask: ShaderValue<Float>?
        /// Straight RGB transmission and alpha coverage for colored directional shadows.
        /// See `SurfaceShaderGraph.Outputs.castShadowColor`.
        public var castShadowColor: ShaderValue<SIMD4<Float>>?
        /// A predicate that replaces `mask` in raster shadow passes. Requires `castShadowColor`.
        public var castShadowMask: ShaderValue<Float>?

        public init(
            baseColor: ShaderValue<SIMD3<Float>>? = nil, metallic: ShaderValue<Float>? = nil,
            roughness: ShaderValue<Float>? = nil, emission: ShaderValue<SIMD3<Float>>? = nil,
            transmissionMask: ShaderValue<Float>? = nil, normal: ShaderValue<SIMD3<Float>>? = nil,
            occlusion: ShaderValue<Float>? = nil, positionOffset: ShaderValue<SIMD3<Float>>? = nil,
            pointSize: ShaderValue<Float>? = nil, lineWidth: ShaderValue<Float>? = nil,
            opacity: ShaderValue<Float>? = nil, alphaTest: ShaderValue<Float>? = nil, mask: ShaderValue<Float>? = nil,
            castShadowColor: ShaderValue<SIMD4<Float>>? = nil, castShadowMask: ShaderValue<Float>? = nil
        ) {
            self.baseColor = baseColor
            self.metallic = metallic
            self.roughness = roughness
            self.emission = emission
            self.transmissionMask = transmissionMask
            self.normal = normal
            self.occlusion = occlusion
            self.positionOffset = positionOffset
            self.pointSize = pointSize
            self.lineWidth = lineWidth
            self.opacity = opacity
            self.alphaTest = alphaTest
            self.mask = mask
            self.castShadowColor = castShadowColor
            self.castShadowMask = castShadowMask
        }
    }

    /// Describes a graph in Swift. Values from other builders, as outputs or as
    /// operands, are rejected.
    ///
    /// ```swift
    /// let finish = try SurfaceShaderGraph { s in
    ///     let tint = s.uniform("Tint", SIMD3<Float>(0.7, 0.3, 0.1))
    ///     let grain = s.fractalNoise(s.worldPosition * 6)
    ///     return .init(baseColor: s.baseColor * tint, roughness: grain.remap(0, 1, 0.3, 0.8))
    /// }
    /// ```
    public init(building body: (ShaderGraphBuilder) throws -> Result) throws {
        let builder = ShaderGraphBuilder()
        let result = try body(builder)
        let values: [(any ShaderArgument)?] = [
            result.baseColor, result.metallic, result.roughness, result.emission,
            result.transmissionMask, result.normal, result.occlusion, result.positionOffset, result.pointSize,
            result.lineWidth,
            result.opacity, result.alphaTest, result.mask, result.castShadowColor, result.castShadowMask,
        ]
        for case let value? in values {
            guard let owned = value as? any ShaderBuilderOwned, owned.owner === builder else {
                throw MaterialGraph.Failure.invalidDeclaration("A graph output belongs to a different builder.")
            }
        }
        try builder.checkOperands()
        let declarations = builder.declarations
        try self.init(
            nodes: builder.nodes, uniforms: declarations.uniforms, textures: declarations.textures,
            functions: declarations.functions, structs: declarations.structs, labels: declarations.labels,
            outputs: Outputs(
                baseColor: result.baseColor?.id, metallic: result.metallic?.id, roughness: result.roughness?.id,
                emission: result.emission?.id, transmissionMask: result.transmissionMask?.id,
                normal: result.normal?.id, occlusion: result.occlusion?.id, positionOffset: result.positionOffset?.id,
                pointSize: result.pointSize?.id, lineWidth: result.lineWidth?.id,
                opacity: result.opacity?.id, alphaTest: result.alphaTest?.id, mask: result.mask?.id,
                castShadowColor: result.castShadowColor?.id, castShadowMask: result.castShadowMask?.id
            )
        )
    }
}

protocol ShaderBuilderOwned {
    var owner: ShaderGraphBuilder { get }
}

extension ShaderValue: ShaderBuilderOwned {
    var owner: ShaderGraphBuilder { builder }
}
