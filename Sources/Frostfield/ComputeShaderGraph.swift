import Foundation

/// A typed compute-stage graph evaluated once at each texel center of a generated texture.
///
/// The graph reads `gridUV` (top-left normalized texel centers), `texelSize`,
/// the graph clock (`time`, `deltaTime`, `frame`), declared uniforms, declared textures sampled at their base level, and declared
/// functions. Its output is RGBA: a vector4 node directly, a vector3 node with
/// alpha 1, or a scalar node as gray with alpha 1. Generate textures with
/// `ShaderTextureGenerator`.
public struct TextureShaderGraph: Hashable, Sendable {
    public let nodes: [MaterialGraph.Node]
    public let uniforms: [ShaderUniform]
    public let textures: [ShaderTextureInput]
    public let functions: [ShaderFunction]
    /// Structs the graph can construct. Field order is the Metal layout.
    public let structs: [ShaderStruct]
    /// Optional node names that appear in generated Metal comments and diagnostics.
    public let labels: [MaterialGraph.NodeID: String]
    /// A scalar, vector3, or vector4 node.
    public let output: MaterialGraph.NodeID
    let program: ShaderProgram
    private let rgba: String

    /// Validates the graph for the compute stage. Inputs other than the clock, `gridUV` and
    /// `texelSize`, scene and depth samples, and vector2 outputs are rejected.
    public init(
        nodes: [MaterialGraph.Node],
        uniforms: [ShaderUniform] = [],
        textures: [ShaderTextureInput] = [],
        functions: [ShaderFunction] = [],
        structs: [ShaderStruct] = [],
        labels: [MaterialGraph.NodeID: String] = [:],
        output: MaterialGraph.NodeID
    ) throws {
        program = try ShaderProgram(
            nodes: nodes,
            declarations: ShaderDeclarations(
                uniforms: uniforms, textures: textures, functions: functions, structs: structs, labels: labels
            ),
            outputs: [(output, .compute)],
            allowsInput: { $0 == .gridUV || $0 == .texelSize || $0.readsClock }
        ) { _, id, type in
            guard type != .vector2, !type.isMatrix else { throw Failure.outputType(node: id, actual: type) }
        }
        let value = "n\(output)"
        switch program.types[output] {
        case .scalar: rgba = "float4(float3(\(value)), 1.0f)"
        case .vector3: rgba = "float4(\(value), 1.0f)"
        case .vector4: rgba = value
        case let type: throw Failure.outputType(node: output, actual: type)
        }
        self.nodes = nodes
        self.uniforms = uniforms
        self.textures = textures
        self.functions = functions
        self.structs = structs
        self.labels = labels
        self.output = output
    }

    /// Declarations every generated graph function depends on. Place this source
    /// after `#include <metal_stdlib>` and `using namespace metal;`.
    public static var supportSource: String { ShaderProgram.supportSource }

    /// Emits the per-texel function:
    ///
    /// ```metal
    /// static inline float4 name(thread const FrostfieldComputeInput& input,
    ///     thread const FrostfieldGraphResources& resources)
    /// ```
    ///
    /// The result is RGBA with nonfinite components replaced by zero. The caller
    /// fills `input.gridUV` and `input.texelSize`, and defines `FrostfieldGraphResources`,
    /// plus `frostfield_graph_uniform` when the graph declares uniforms and
    /// `frostfield_graph_sample` when it declares textures. Choose a unique name of
    /// up to 128 ASCII letters, digits or single separating underscores, starting
    /// with a letter. Declared functions use it as a prefix.
    public func metalFunction(named name: String) throws -> String {
        guard ShaderProgram.isIdentifier(name) else { throw Failure.invalidFunctionName(name) }
        return """
        \(program.functionDefinitions(prefix: name))

        static inline float4 \(name)(
            thread const FrostfieldComputeInput& input,
            thread const FrostfieldGraphResources& resources
        ) {
        \(try program.statements(for: [output], prefix: name))
            float4 rgba = \(rgba);
            return select(float4(0), rgba, frostfield_graph_finite(rgba));
        }
        """
    }

    public enum Failure: LocalizedError, Equatable {
        case outputType(node: MaterialGraph.NodeID, actual: MaterialGraph.ValueType)
        case invalidFunctionName(String)

        public var errorDescription: String? {
            switch self {
            case .outputType(let node, let actual):
                "A texture graph outputs a scalar, vector3 or vector4, but node \(node) produces \(actual.rawValue)."
            case .invalidFunctionName(let name): "The generated Metal function name \(name) is invalid."
            }
        }
    }
}

/// A typed compute-stage graph evaluated once per element of a float4 record buffer.
///
/// Each element holds `fieldCount` consecutive float4 fields (16 bytes each, no
/// padding). The graph reads `elementIndex`, `elementCount`, `elementField(i)`
/// for `i < fieldCount`, the graph clock, declared uniforms, declared textures sampled at their
/// base level, and declared functions. Each connected output replaces one field;
/// unconnected fields keep their value. Run it with `ShaderBufferKernel`.
public struct BufferShaderGraph: Hashable, Sendable {
    public static let maximumFieldCount = 4

    public let nodes: [MaterialGraph.Node]
    /// The number of float4 fields in each element, from one through four.
    public let fieldCount: Int
    public let uniforms: [ShaderUniform]
    public let textures: [ShaderTextureInput]
    public let functions: [ShaderFunction]
    /// Structs the graph can construct. Field order is the Metal layout.
    public let structs: [ShaderStruct]
    /// Optional node names that appear in generated Metal comments and diagnostics.
    public let labels: [MaterialGraph.NodeID: String]
    /// The vector4 node written to each connected field index.
    public let outputs: [Int: MaterialGraph.NodeID]
    let program: ShaderProgram

    /// The byte distance between consecutive elements: 16 × `fieldCount`.
    public var elementStride: Int { fieldCount * MemoryLayout<SIMD4<Float>>.stride }

    /// Validates the graph for the compute stage. Outputs must be vector4 nodes at
    /// field indices below `fieldCount`. Texture-grid inputs, scene and depth
    /// samples, and reads of fields at or beyond `fieldCount` are rejected.
    public init(
        nodes: [MaterialGraph.Node],
        fieldCount: Int,
        uniforms: [ShaderUniform] = [],
        textures: [ShaderTextureInput] = [],
        functions: [ShaderFunction] = [],
        structs: [ShaderStruct] = [],
        labels: [MaterialGraph.NodeID: String] = [:],
        outputs: [Int: MaterialGraph.NodeID]
    ) throws {
        guard (1...Self.maximumFieldCount).contains(fieldCount) else { throw Failure.fieldCount(fieldCount) }
        guard !outputs.isEmpty else { throw Failure.noOutputs }
        let connected = outputs.sorted { $0.key < $1.key }
        if let field = connected.first(where: { !(0..<fieldCount).contains($0.key) })?.key {
            throw Failure.outputField(field, fieldCount: fieldCount)
        }
        program = try ShaderProgram(
            nodes: nodes,
            declarations: ShaderDeclarations(
                uniforms: uniforms, textures: textures, functions: functions, structs: structs, labels: labels
            ),
            outputs: connected.map { ($0.value, .compute) },
            allowsInput: { input in
                switch input {
                case .elementIndex, .elementCount, .elementField, .time, .deltaTime, .frame: true
                default: false
                }
            }
        ) { index, id, type in
            guard type == .vector4 else { throw Failure.outputType(field: connected[index].key, node: id, actual: type) }
        }
        for id in program.order(for: connected.map(\.value)) {
            if case .input(.elementField(let field)) = nodes[id], field >= fieldCount {
                throw Failure.fieldOutOfRange(node: id, field: field, fieldCount: fieldCount)
            }
        }
        self.nodes = nodes
        self.fieldCount = fieldCount
        self.uniforms = uniforms
        self.textures = textures
        self.functions = functions
        self.structs = structs
        self.labels = labels
        self.outputs = outputs
    }

    /// Declarations every generated graph function depends on. Place this source
    /// after `#include <metal_stdlib>` and `using namespace metal;`.
    public static var supportSource: String { ShaderProgram.supportSource }

    /// Emits the per-element update:
    ///
    /// ```metal
    /// static inline void name(thread const FrostfieldComputeInput& input,
    ///     thread const FrostfieldGraphResources& resources, thread float4 (&fields)[4])
    /// ```
    ///
    /// Initialize `fields` with the element's current fields. Each connected output
    /// overwrites its field; nonfinite components keep the field's previous value.
    /// The caller fills `input.elementIndex`, `input.elementCount` and `input.fields`,
    /// and defines the same resources and hooks as `TextureShaderGraph.metalFunction(named:)`.
    public func metalFunction(named name: String) throws -> String {
        guard ShaderProgram.isIdentifier(name) else { throw Failure.invalidFunctionName(name) }
        let connected = outputs.sorted { $0.key < $1.key }
        let assignments = connected.map { field, id in
            "    fields[\(field)] = select(fields[\(field)], n\(id), frostfield_graph_finite(n\(id)));"
        }.joined(separator: "\n")
        return """
        \(program.functionDefinitions(prefix: name))

        static inline void \(name)(
            thread const FrostfieldComputeInput& input,
            thread const FrostfieldGraphResources& resources,
            thread float4 (&fields)[4]
        ) {
        \(try program.statements(for: connected.map(\.value), prefix: name))
        \(assignments)
        }
        """
    }

    public enum Failure: LocalizedError, Equatable {
        case fieldCount(Int)
        case noOutputs
        case outputField(Int, fieldCount: Int)
        case outputType(field: Int, node: MaterialGraph.NodeID, actual: MaterialGraph.ValueType)
        case fieldOutOfRange(node: MaterialGraph.NodeID, field: Int, fieldCount: Int)
        case invalidFunctionName(String)

        public var errorDescription: String? {
            switch self {
            case .fieldCount(let count):
                "A buffer graph element has one to four float4 fields, not \(count)."
            case .noOutputs: "A buffer shader graph needs at least one connected field output."
            case .outputField(let field, let fieldCount):
                "Buffer graph output field \(field) is outside the element's \(fieldCount) fields."
            case .outputType(let field, let node, let actual):
                "Buffer graph field \(field) needs a vector4, but node \(node) produces \(actual.rawValue)."
            case .fieldOutOfRange(let node, let field, let fieldCount):
                "Buffer graph node \(node) reads field \(field), but each element has \(fieldCount) fields."
            case .invalidFunctionName(let name): "The generated Metal function name \(name) is invalid."
            }
        }
    }
}

// MARK: - Builders

extension TextureShaderGraph {
    /// Describes a texture graph in Swift. Return a scalar, vector3 or vector4 value.
    ///
    /// ```swift
    /// let checker = try TextureShaderGraph { t in
    ///     let cells = (t.gridUV * 8).floor()
    ///     return (cells.x + cells.y).mod(2)
    /// }
    /// ```
    public init<Value: ShaderGraphValue>(building body: (ShaderGraphBuilder) throws -> ShaderValue<Value>) throws {
        let builder = ShaderGraphBuilder()
        let result = try body(builder)
        guard result.builder === builder else {
            throw MaterialGraph.Failure.invalidDeclaration("A graph output belongs to a different builder.")
        }
        try builder.checkOperands()
        let declarations = builder.declarations
        try self.init(
            nodes: builder.nodes, uniforms: declarations.uniforms, textures: declarations.textures,
            functions: declarations.functions, structs: declarations.structs, labels: declarations.labels, output: result.id
        )
    }
}

extension BufferShaderGraph {
    /// Describes a buffer graph in Swift. Return the new value of each field to write, by field index.
    ///
    /// ```swift
    /// let advect = try BufferShaderGraph(fieldCount: 2) { b in
    ///     let position = b.elementField(0)
    ///     let velocity = b.elementField(1)
    ///     return [0: position + velocity * b.uniform("DeltaTime", Float(1.0 / 60))]
    /// }
    /// ```
    public init(
        fieldCount: Int, building body: (ShaderGraphBuilder) throws -> [Int: ShaderValue<SIMD4<Float>>]
    ) throws {
        let builder = ShaderGraphBuilder()
        let result = try body(builder)
        guard result.values.allSatisfy({ $0.builder === builder }) else {
            throw MaterialGraph.Failure.invalidDeclaration("A graph output belongs to a different builder.")
        }
        try builder.checkOperands()
        let declarations = builder.declarations
        try self.init(
            nodes: builder.nodes, fieldCount: fieldCount, uniforms: declarations.uniforms,
            textures: declarations.textures, functions: declarations.functions, structs: declarations.structs,
            labels: declarations.labels,
            outputs: result.mapValues(\.id)
        )
    }
}

extension ShaderUniforms {
    /// Starts every uniform the texture graph declares at its default.
    public convenience init(graph: TextureShaderGraph) throws {
        try self.init(graph.uniforms)
    }

    /// Starts every uniform the buffer graph declares at its default.
    public convenience init(graph: BufferShaderGraph) throws {
        try self.init(graph.uniforms)
    }
}
