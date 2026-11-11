import Foundation

/// The declarations a graph can reference by index.
struct ShaderDeclarations: Hashable, Sendable {
    var uniforms: [ShaderUniform] = []
    var textures: [ShaderTextureInput] = []
    var functions: [ShaderFunction] = []
    var structs: [ShaderStruct] = []
    var labels: [MaterialGraph.NodeID: String] = [:]

    /// Float4 slots taken by the declared values. One per scalar or vector, one per matrix column.
    var uniformValueSlotCount: Int { uniforms.reduce(0) { $0 + $1.type.uniformSlotCount } }

    func uniformSlotOffset(_ index: Int) -> Int {
        uniforms.prefix(index).reduce(0) { $0 + $1.type.uniformSlotCount }
    }
}

/// A validated, typed graph that emits Metal statements for selected outputs.
/// Each statement declares `n<id>` and ends with a `// node <id>` marker, so
/// compiler diagnostics can be traced back to graph nodes.
struct ShaderProgram: Hashable, Sendable {
    typealias Node = MaterialGraph.Node
    typealias NodeID = MaterialGraph.NodeID
    typealias ValueType = MaterialGraph.ValueType
    typealias Failure = MaterialGraph.Failure

    let nodes: [Node]
    let types: [ValueType]
    /// The declared struct index for each node that produces a struct, otherwise nil.
    let structIndices: [Int?]
    let declarations: ShaderDeclarations
    /// Every output the graph was validated for, in the order given.
    let roots: [NodeID]

    /// Validates declarations, every node's types, and that each output's
    /// reachable nodes are available in that output's stage. Disconnected
    /// nodes must type-check but are never emitted.
    init(
        nodes: [Node],
        declarations: ShaderDeclarations,
        outputs: [(NodeID, MaterialGraph.Stage)],
        allowsInput: (MaterialGraph.Input) -> Bool = { _ in true },
        allowsSurfaceViewportSamples: Bool = false,
        validateOutput: (Int, NodeID, ValueType) throws -> Void
    ) throws {
        guard nodes.count <= MaterialGraph.maximumNodeCount else {
            throw Failure.tooManyNodes(limit: MaterialGraph.maximumNodeCount)
        }
        try Self.validate(declarations)

        var types = [ValueType?](repeating: nil, count: nodes.count)
        var visiting = [Bool](repeating: false, count: nodes.count)
        let identity = StructIdentity()

        // Types a node and everything it depends on with an explicit stack, so a
        // long dependency chain cannot overflow the thread's stack. `type(of:)`
        // reads inputs in a fixed order. When it reaches an input that has no type
        // yet, the node is set aside, the input is typed first, and the node is
        // typed again from the start; typed inputs answer at once. This reports the
        // same failure for the same node as a depth-first recursive walk.
        func visit(_ root: NodeID, referencedBy parent: NodeID) throws -> ValueType {
            guard nodes.indices.contains(root) else { throw Failure.invalidReference(node: parent, referenced: root) }
            if let type = types[root] { return type }
            visiting[root] = true
            var stack = [root]
            while let id = stack.last {
                do {
                    let type = try Self.type(of: id, in: nodes, declarations: declarations, identity: identity) { input in
                        guard nodes.indices.contains(input) else {
                            throw Failure.invalidReference(node: id, referenced: input)
                        }
                        if let type = types[input] { return type }
                        if visiting[input] { throw Failure.cycle(input) }
                        throw UntypedInput(id: input)
                    }
                    types[id] = type
                    visiting[id] = false
                    stack.removeLast()
                } catch let untyped as UntypedInput {
                    visiting[untyped.id] = true
                    stack.append(untyped.id)
                }
            }
            return types[root]!
        }

        for (index, output) in outputs.enumerated() {
            guard nodes.indices.contains(output.0) else { throw Failure.invalidOutput(output.0) }
            try validateOutput(index, output.0, visit(output.0, referencedBy: output.0))
        }
        for id in nodes.indices {
            _ = try visit(id, referencedBy: id)
        }
        self.nodes = nodes
        self.types = types.map { $0! }
        self.structIndices = (0..<nodes.count).map { identity.index[$0] }
        self.declarations = declarations
        roots = outputs.map(\.0)

        for (output, stage) in outputs {
            for id in order(for: [output]) where !Self.isAvailable(
                nodes[id], in: stage, allowsInput: allowsInput, allowsSurfaceViewportSamples: allowsSurfaceViewportSamples
            ) {
                throw Failure.unavailable(node: id, stage: stage)
            }
        }
        // Each stage's outputs are generated together, so check their control flow together.
        var stages: [MaterialGraph.Stage] = []
        for (_, stage) in outputs where !stages.contains(stage) { stages.append(stage) }
        for stage in stages {
            _ = try ShaderSchedule(program: self, outputs: outputs.filter { $0.1 == stage }.map(\.0))
        }
    }

    /// Whether any output reads `time`, `deltaTime` or `frame`.
    var readsClock: Bool { readsClock(from: roots) }

    /// Whether the given outputs read `time`, `deltaTime` or `frame`.
    func readsClock(from outputs: [NodeID]) -> Bool {
        uses({ if case .input(let input) = $0 { input.readsClock } else { false } }, from: outputs)
    }

    /// The float4 slots the uniform buffer needs: the declared value slots, then the clock when read.
    var uniformSlotCount: Int { declarations.uniformValueSlotCount + (readsClock ? 1 : 0) }

    /// The first node reachable from the outputs that only a rasterizer can evaluate:
    /// a derivative, screen input, or stored viewport texture sample.
    func firstScreenSpaceNode(from outputs: [NodeID]) -> NodeID? {
        order(for: outputs).first { id in
            switch nodes[id] {
            case .derivative, .sampleScene, .sampleSceneLevel, .sampleDepth, .viewportSafeUV, .softParticleOpacity: true
            case .input(let input): input.isScreenSpace
            default: false
            }
        }
    }

    /// Whether the outputs read the camera's view or projection matrix or clip distances.
    func readsCameraProjection(from outputs: [NodeID]) -> Bool {
        uses({ node in
            if case .softParticleOpacity = node { return true }
            guard case .input(let input) = node else { return false }
            return [.viewMatrix, .projectionMatrix, .cameraNear, .cameraFar].contains(input)
        }, from: outputs)
    }

    /// Reachable nodes in dependency order, starting from the given outputs. The
    /// walk is a depth-first post-order on an explicit stack, so graph depth does
    /// not grow the call stack.
    func order(for outputs: [NodeID]) -> [NodeID] {
        var emitted = [Bool](repeating: false, count: nodes.count)
        var ordered: [NodeID] = []
        var stack: [(id: NodeID, inputs: [NodeID], next: Int)] = []
        for output in outputs where !emitted[output] {
            emitted[output] = true
            stack.append((output, Self.inputs(of: nodes[output]), 0))
            while let top = stack.last {
                if top.next < top.inputs.count {
                    let input = top.inputs[top.next]
                    stack[stack.count - 1].next += 1
                    guard !emitted[input] else { continue }
                    emitted[input] = true
                    stack.append((input, Self.inputs(of: nodes[input]), 0))
                } else {
                    ordered.append(top.id)
                    stack.removeLast()
                }
            }
        }
        return ordered
    }

    /// Signals that `type(of:)` needs an input typed before it can finish.
    private struct UntypedInput: Error {
        let id: NodeID
    }

    func uniformMask(from outputs: [NodeID]) -> UInt32 {
        var mask: UInt32 = 0
        for id in order(for: outputs) {
            if case .uniform(let index) = nodes[id] { mask |= UInt32(1) << index }
        }
        return mask
    }

    func uses(_ predicate: (Node) -> Bool, from outputs: [NodeID]) -> Bool {
        order(for: outputs).contains { predicate(nodes[$0]) }
    }

    /// Statements that declare every node reachable from the outputs. The prefix
    /// names the caller's generated function and its declared functions. Branches
    /// and loops become `if` and `for` blocks; see `ShaderSchedule`.
    func statements(for outputs: [NodeID], prefix: String) throws -> String {
        let schedule = try ShaderSchedule(program: self, outputs: outputs)
        var lines: [String] = []
        emit(scope: 0, schedule: schedule, indent: "    ", prefix: prefix, into: &lines)
        return lines.joined(separator: "\n")
    }

    private func marker(_ id: NodeID, _ kind: String = "") -> String {
        var marker = "// node \(id)" + (kind.isEmpty ? "" : " \(kind)")
        if let label = declarations.labels[id] { marker += " \(Self.commentSafe(label))" }
        return marker
    }

    /// Exact equality for a switch arm. Vectors match when every component is equal.
    private static func switchEqual(_ selector: NodeID, _ match: NodeID, types: [ValueType]) -> String {
        let comparison = "n\(selector) == n\(match)"
        return types[selector].componentCount == 1 ? comparison : "all(\(comparison))"
    }

    /// Emits one scope's nodes in dependency order, recursing into branch arms and loop bodies.
    private func emit(scope: Int, schedule: ShaderSchedule, indent: String, prefix: String, into lines: inout [String]) {
        let inner = indent + "    "
        for id in schedule.nodes(in: scope) {
            let type = valueName(id)
            switch nodes[id] {
            case .branch(let condition, let ifTrue, let ifFalse):
                lines.append("\(indent)\(type) n\(id); \(marker(id, "branch"))")
                let conditionTest = types[condition] == .bool ? "n\(condition)" : "n\(condition) > 0.5f"
                lines.append("\(indent)if (\(conditionTest)) {")
                emit(scope: schedule.trueScope[id]!, schedule: schedule, indent: inner, prefix: prefix, into: &lines)
                lines.append("\(inner)n\(id) = n\(ifTrue);")
                lines.append("\(indent)} else {")
                emit(scope: schedule.falseScope[id]!, schedule: schedule, indent: inner, prefix: prefix, into: &lines)
                lines.append("\(inner)n\(id) = n\(ifFalse);")
                lines.append("\(indent)}")
            case .switchSelect(let selector, let matches, let values, let fallback):
                lines.append("\(indent)\(type) n\(id); \(marker(id, "switch"))")
                let arms = schedule.caseScopes[id]!
                for (index, match) in matches.enumerated() {
                    let keyword = index == 0 ? "if" : "} else if"
                    lines.append("\(indent)\(keyword) (\(Self.switchEqual(selector, match, types: types))) {")
                    emit(scope: arms[index], schedule: schedule, indent: inner, prefix: prefix, into: &lines)
                    lines.append("\(inner)n\(id) = n\(values[index]);")
                }
                lines.append("\(indent)} else {")
                emit(scope: arms[matches.count], schedule: schedule, indent: inner, prefix: prefix, into: &lines)
                lines.append("\(inner)n\(id) = n\(fallback);")
                lines.append("\(indent)}")
            case .loop(let count, let iterations, let initial, let next, let exit, let skip):
                lines.append("\(indent)\(type) n\(id) = n\(initial); \(marker(id, "loop \(count)"))")
                if let iterations {
                    lines.append("\(indent)float n\(id)_count = n\(iterations);")
                    lines.append(
                        "\(indent)int n\(id)_limit = frostfield_graph_finite(n\(id)_count) && n\(id)_count > 0.0f "
                            + "? min((int)n\(id)_count, \(count)) : 0;"
                    )
                    lines.append("\(indent)for (int n\(id)_i = 0; n\(id)_i < n\(id)_limit; ++n\(id)_i) {")
                } else {
                    lines.append("\(indent)for (int n\(id)_i = 0; n\(id)_i < \(count); ++n\(id)_i) {")
                }
                emit(scope: schedule.bodyScope[id]!, schedule: schedule, indent: inner, prefix: prefix, into: &lines)
                if let skip, let update = schedule.updateScope[id] {
                    lines.append("\(inner)if (!(n\(skip) > 0.5f)) {")
                    emit(scope: update, schedule: schedule, indent: inner + "    ", prefix: prefix, into: &lines)
                    lines.append("\(inner)    n\(id) = n\(next);")
                    lines.append("\(inner)}")
                } else {
                    lines.append("\(inner)n\(id) = n\(next);")
                }
                if let exit { lines.append("\(inner)if (n\(exit) > 0.5f) { break; }") }
                lines.append("\(indent)}")
            case .discard(let condition, let value):
                lines.append("\(indent)if (n\(condition) > 0.5f) { surface.mask = false; } \(marker(id, "discard"))")
                lines.append("\(indent)\(type) n\(id) = n\(value);")
            default:
                lines.append("\(indent)\(type) n\(id) = \(expression(id, prefix: prefix)); \(marker(id))")
            }
        }
    }

    /// Metal definitions for the graph's declared functions and the helpers its
    /// parallax nodes generate. Place them after the renderer's resource hooks.
    func functionDefinitions(prefix: String) -> String {
        let declared = declarations.functions.map { function in
            let parameters = function.parameters.map { "\($0.type.metalType) \($0.name)" }.joined(separator: ", ")
            return """
            // function \(function.name)
            static inline \(function.returnType.metalType) \(prefix)_fn_\(function.name)(\(parameters)) {
            \(function.body)
            }
            """
        }
        let parallax = nodes.indices.compactMap { id -> String? in
            guard case .parallaxOcclusion(let slot, _, _, _, let steps) = nodes[id] else { return nil }
            return Self.parallaxHelper(name: "\(prefix)_parallax_\(id)", texture: declarations.textures[slot], slot: slot, steps: steps)
        }
        let hasProceduralNoise = nodes.contains { node in
            switch node {
                case .simplexNoise, .curlNoise, .animatedVoronoi: true
                default: false
            }
        }
        let procedural = hasProceduralNoise ? [Self.proceduralNoiseFunctions(prefix: prefix)] : []
        return (structDefinitions() + procedural + declared + parallax).filter { !$0.isEmpty }.joined(separator: "\n\n")
    }

    /// Metal layout for every declared struct, in declaration order so a nested field's
    /// type is already defined.
    private func structDefinitions() -> [String] {
        declarations.structs.map { structure in
            let fields = structure.fields.map { field -> String in
                let typeName: String
                switch field.content {
                case .value(let type): typeName = type.metalType
                case .structure(let nested): typeName = declarations.structs[nested].metalName()
                }
                return "    \(typeName) \(field.name);"
            }.joined(separator: "\n")
            return "struct \(structure.metalName()) {\n\(fields)\n};"
        }
    }

    /// The Metal type of one node's value. Struct nodes use their declared layout.
    private func valueName(_ id: NodeID) -> String {
        if types[id] == .structure, let index = structIndices[id], declarations.structs.indices.contains(index) {
            return declarations.structs[index].metalName()
        }
        return types[id].metalType
    }

    /// A fixed-count march with one sample per layer and no early exit or return, so
    /// every pixel of a quad runs every sample and implicit mip selection stays
    /// defined. Invalid inputs (a view at or below the surface, a nonpositive or
    /// nonfinite depth, nonfinite coordinates) march with no shift and return `uv`.
    private static func parallaxHelper(name: String, texture: ShaderTextureInput, slot: Int, steps: Int) -> String {
        let sample = "frostfield_graph_sample(resources, \(slot)u, \(texture.addressing.code)u, \(texture.filtering.code)u"
        return """
        // parallax occlusion, \(steps) layers
        static inline float2 \(name)(
            thread const FrostfieldGraphResources& resources, float2 uv, float3 view, float depth
        ) {
            float3 direction = frostfield_graph_unit(view);
            bool valid = direction.z > 0.0f && depth > 0.0f && frostfield_graph_finite(depth)
                && all(frostfield_graph_finite(uv));
            float2 shift = valid ? -direction.xy / max(direction.z, 0.05f) * depth : float2(0);
            float previousDepth = 0.0f;
            float previousGap = -(1.0f - saturate(\(sample), uv).r));
            float hitDepth = 0.0f;
            bool found = previousGap >= 0.0f;
            for (int layer = 1; layer <= \(steps); ++layer) {
                float layerDepth = float(layer) / \(steps).0f;
                float surfaceDepth = 1.0f - saturate(\(sample), uv + shift * layerDepth).r);
                float gap = layerDepth - surfaceDepth;
                if (!found && gap >= 0.0f) {
                    hitDepth = previousDepth + (layerDepth - previousDepth) * (previousGap / (previousGap - gap));
                    found = true;
                }
                previousDepth = layerDepth;
                previousGap = gap;
            }
            return valid ? uv + shift * hitDepth : uv;
        }
        """
    }

    // MARK: - Typing

    /// Struct declaration chosen by each struct-producing node. Filled while typing.
    private final class StructIdentity {
        var index: [NodeID: Int] = [:]
    }

    private static func type(
        of id: NodeID, in nodes: [Node], declarations: ShaderDeclarations,
        identity: StructIdentity,
        visit visitAny: (NodeID) throws -> ValueType
    ) throws -> ValueType {
        // Scalar and vector operations reject matrix operands; matrix nodes use visitAny.
        func visit(_ input: NodeID) throws -> ValueType {
            let type = try visitAny(input)
            guard !type.isMatrix else { throw Failure.matrixOperand(node: id, input: input) }
            return type
        }
        func require(_ input: NodeID, _ expected: ValueType) throws {
            let actual = try visitAny(input)
            guard actual == expected else {
                throw Failure.typeMismatch(node: id, input: input, expected: expected, actual: actual)
            }
        }
        func adoptStructure(of origin: NodeID, type: ValueType) {
            if type == .structure, let index = identity.index[origin] {
                identity.index[id] = index
            }
        }
        /// Same value type, and the same struct declaration when the type is a struct.
        func requireMatching(_ input: NodeID, _ origin: NodeID, _ type: ValueType) throws {
            let actual = try visitAny(input)
            guard actual == type else {
                throw Failure.typeMismatch(node: id, input: input, expected: type, actual: actual)
            }
            if type == .structure, identity.index[input] != identity.index[origin] {
                throw Failure.structMismatch(
                    node: id, expected: identity.index[origin] ?? -1, actual: identity.index[input] ?? -1
                )
            }
        }
        /// Float scalars and vectors. Integers and bools have their own operations.
        func requireFloatOperand(_ input: NodeID) throws -> ValueType {
            let actual = try visit(input)
            guard actual != .int, actual != .bool, actual != .structure else {
                throw Failure.typeMismatch(node: id, input: input, expected: .scalar, actual: actual)
            }
            return actual
        }
        func requireMatrix(_ input: NodeID) throws -> ValueType {
            let actual = try visitAny(input)
            guard actual.isMatrix else {
                throw Failure.typeMismatch(node: id, input: input, expected: .matrix4, actual: actual)
            }
            return actual
        }
        func requireTexture(_ slot: Int) throws {
            guard declarations.textures.indices.contains(slot) else { throw Failure.invalidTexture(node: id, slot: slot) }
        }
        func requireFlatTexture(_ slot: Int) throws {
            try requireTexture(slot)
            if declarations.textures[slot].shape == .cube {
                throw Failure.invalidDeclaration("Cube map \(declarations.textures[slot].name) is sampled with a direction.")
            }
        }
        func requireCube(_ slot: Int) throws {
            try requireTexture(slot)
            guard declarations.textures[slot].shape == .cube else {
                throw Failure.invalidDeclaration("Texture \(declarations.textures[slot].name) is 2D. Sample it with a vector2.")
            }
        }
        func requireVector(_ input: NodeID) throws -> ValueType {
            let actual = try visit(input)
            guard actual != .scalar else {
                throw Failure.typeMismatch(node: id, input: input, expected: .vector3, actual: actual)
            }
            return actual
        }
        func finite(_ values: Float...) throws {
            guard values.allSatisfy(\.isFinite) else { throw Failure.nonfiniteConstant(id) }
        }

        switch nodes[id] {
        case .scalar(let value):
            try finite(value)
            return .scalar
        case .integer:
            return .int
        case .boolean:
            return .bool
        case .color(let value):
            try finite(value.x, value.y, value.z)
            return .vector3
        case .vector2(let value):
            try finite(value.x, value.y)
            return .vector2
        case .vector4(let value):
            try finite(value.x, value.y, value.z, value.w)
            return .vector4
        case .parameter, .uv, .normal, .worldPosition:
            return .scalar
        case .input(let input):
            if case .elementField(let index) = input, !(0..<4).contains(index) {
                throw Failure.unavailable(node: id, stage: .compute)
            }
            return input.type
        case .uniform(let index):
            guard declarations.uniforms.indices.contains(index) else {
                throw Failure.invalidUniform(node: id, index: index)
            }
            return declarations.uniforms[index].type
        case .sampleTexture(let slot, let coordinates):
            try requireFlatTexture(slot)
            try require(coordinates, .vector2)
            return .vector4
        case .sampleTextureLevel(let slot, let coordinates, let level):
            try requireFlatTexture(slot)
            try require(coordinates, .vector2)
            try require(level, .scalar)
            return .vector4
        case .sampleTextureGradient(let slot, let coordinates, let ddx, let ddy):
            try requireFlatTexture(slot)
            try require(coordinates, .vector2)
            try require(ddx, .vector2)
            try require(ddy, .vector2)
            return .vector4
        case .sampleCube(let slot, let direction):
            try requireCube(slot)
            try require(direction, .vector3)
            return .vector4
        case .sampleScene(let coordinates):
            try require(coordinates, .vector2)
            return .vector4
        case .sampleSceneLevel(let coordinates, let level):
            try require(coordinates, .vector2)
            try require(level, .scalar)
            return .vector4
        case .sampleDepth(let coordinates):
            try require(coordinates, .vector2)
            return .scalar
        case .viewportSafeUV(let coordinates, let fallback):
            try require(coordinates, .vector2)
            try require(fallback, .vector2)
            return .vector2
        case .softParticleOpacity(let opacity, let distance, let contrast, let depth):
            for input in [opacity, distance, contrast, depth] { try require(input, .scalar) }
            return .scalar
        case .viewPositionFromDepth(let coordinates, let depth, let inverse):
            try require(coordinates, .vector2)
            try require(depth, .scalar)
            try require(inverse, .matrix4)
            return .vector3
        case .normalFromDepth(let coordinates, let inverse, _):
            try require(coordinates, .vector2)
            try require(inverse, .matrix4)
            return .vector3
        case .denoise(let coordinates, let inverse, let settings, let sampleIndex, let normalSlot, _):
            try require(coordinates, .vector2)
            try require(inverse, .matrix4)
            try require(settings, .vector4)
            try require(sampleIndex, .scalar)
            if let normalSlot {
                guard declarations.textures.indices.contains(normalSlot) else {
                    throw Failure.invalidTexture(node: id, slot: normalSlot)
                }
                guard declarations.textures[normalSlot].content == .data else {
                    throw Failure.invalidDeclaration("Denoise view normals need a data texture.")
                }
            }
            return .vector4
        case .linearDepth(let coordinates, let depth, let inverse, let near, let far):
            try require(coordinates, .vector2)
            try require(depth, .scalar)
            try require(inverse, .matrix4)
            try require(near, .scalar)
            try require(far, .scalar)
            return .scalar
        case .add(let a, let b), .subtract(let a, let b):
            let type = try visit(a)
            guard type != .bool, type != .structure else {
                throw Failure.typeMismatch(node: id, input: a, expected: .scalar, actual: .bool)
            }
            try require(b, type)
            return type
        case .multiply(let a, let b):
            let left = try visit(a)
            guard left != .bool, left != .structure else {
                throw Failure.typeMismatch(node: id, input: a, expected: .scalar, actual: .bool)
            }
            let right = try visit(b)
            if left == right { return left }
            // A float scalar scales vectors. It does not scale an integer; that would be `int * float`.
            if right == .scalar, left != .int { return left }
            if left == .scalar, right != .int, right != .bool { return right }
            throw Failure.typeMismatch(node: id, input: b, expected: left, actual: right)
        case .divide(let dividend, let divisor):
            let type = try visit(dividend)
            if type == .int {
                try require(divisor, .int)
                return .int
            }
            guard type != .bool, type != .structure else {
                throw Failure.typeMismatch(node: id, input: dividend, expected: .scalar, actual: type)
            }
            try require(divisor, .scalar)
            return type
        case .mix(let a, let b, let factor):
            let type = try requireFloatOperand(a)
            try require(b, type)
            try require(factor, .scalar)
            return type
        case .sine(let input), .cosine(let input), .floor(let input), .fract(let input):
            try require(input, .scalar)
            return .scalar
        case .smoothstep(let edge0, let edge1, let value), .noise(let edge0, let edge1, let value):
            try require(edge0, .scalar)
            try require(edge1, .scalar)
            try require(value, .scalar)
            return .scalar
        case .clamp01(let input):
            return try requireFloatOperand(input)
        case .rgb(let red, let green, let blue):
            try require(red, .scalar)
            try require(green, .scalar)
            try require(blue, .scalar)
            return .vector3
        case .unary(let operation, let input):
            switch operation {
            case .normalize:
                _ = try requireFloatOperand(input)
                return try requireVector(input)
            case .length:
                _ = try requireFloatOperand(input)
                return .scalar
            case .luminance:
                try require(input, .vector3)
                return .scalar
            case .hsvToRGB, .rgbToHSV, .srgbToLinear, .linearToSRGB:
                try require(input, .vector3)
                return .vector3
            case .abs, .negate:
                let actual = try visit(input)
                guard actual != .bool else {
                    throw Failure.typeMismatch(node: id, input: input, expected: .scalar, actual: actual)
                }
                return actual
            default:
                return try requireFloatOperand(input)
            }
        case .binary(let operation, let a, let b):
            let type = try visit(a)
            try require(b, type)
            switch operation {
            case .minimum, .maximum:
                guard type != .bool, type != .structure else {
                    throw Failure.typeMismatch(node: id, input: a, expected: .scalar, actual: type)
                }
                return type
            case .dot, .distance:
                _ = try requireFloatOperand(a)
                return .scalar
            case .cross:
                guard type == .vector3 else {
                    throw Failure.typeMismatch(node: id, input: a, expected: .vector3, actual: type)
                }
                return .vector3
            case .reflect:
                _ = try requireFloatOperand(a)
                return try requireVector(a)
            default:
                return try requireFloatOperand(a)
            }
        case .clamp(let value, let minimum, let maximum):
            let type = try visit(value)
            guard type != .bool, type != .structure else {
                throw Failure.typeMismatch(node: id, input: value, expected: .scalar, actual: type)
            }
            for bound in [minimum, maximum] {
                let actual = try visit(bound)
                let allowed = type == .int ? actual == .int : actual == type || actual == .scalar
                guard allowed else {
                    throw Failure.typeMismatch(node: id, input: bound, expected: type, actual: actual)
                }
            }
            return type
        case .select(let condition, let ifTrue, let ifFalse), .branch(let condition, let ifTrue, let ifFalse):
            let conditionType = try visit(condition)
            guard conditionType == .scalar || conditionType == .bool else {
                throw Failure.typeMismatch(node: id, input: condition, expected: .scalar, actual: conditionType)
            }
            let type = try visitAny(ifTrue)
            try requireMatching(ifFalse, ifTrue, type)
            adoptStructure(of: ifTrue, type: type)
            return type
        case .remap(let value, let fromMinimum, let fromMaximum, let toMinimum, let toMaximum):
            let type = try requireFloatOperand(value)
            for bound in [fromMinimum, fromMaximum, toMinimum, toMaximum] { try require(bound, .scalar) }
            return type
        case .swizzle(let input, let components):
            let source = try requireFloatOperand(input)
            guard let count = swizzleCount(components, sourceComponents: source.componentCount),
                  let type = ValueType(componentCount: count) else {
                throw Failure.invalidSwizzle(node: id, components: components)
            }
            return type
        case .combine(let inputs):
            var count = 0
            for input in inputs {
                count += try requireFloatOperand(input).componentCount
            }
            guard (2...4).contains(inputs.count), let type = ValueType(componentCount: count), count >= 2 else {
                throw Failure.invalidCombine(node: id, components: count)
            }
            return type
        case .latticeNoise(_, let position), .cellularNoise(let position):
            try require(position, .vector3)
            if case .cellularNoise = nodes[id] { return .vector2 }
            return .scalar
        case .simplexNoise(let position):
            try require(position, .vector3)
            return .scalar
        case .curlNoise(let position):
            try require(position, .vector3)
            return .vector3
        case .animatedVoronoi(let position, let time):
            let type = try visit(position)
            guard type == .vector2 || type == .vector3 else {
                throw Failure.typeMismatch(node: id, input: position, expected: .vector3, actual: type)
            }
            try require(time, .scalar)
            return .scalar
        case .fractalNoise(_, let position, let octaves, let lacunarity, let gain):
            guard (1...8).contains(octaves), lacunarity.isFinite, lacunarity > 0, gain.isFinite, gain > 0 else {
                throw Failure.invalidNoise(id)
            }
            try require(position, .vector3)
            return .scalar
        case .function(let index, let arguments):
            guard declarations.functions.indices.contains(index) else {
                throw Failure.invalidFunction(node: id, index: index)
            }
            let function = declarations.functions[index]
            guard arguments.count == function.parameters.count else {
                throw Failure.argumentCount(node: id, expected: function.parameters.count, actual: arguments.count)
            }
            for (argument, parameter) in zip(arguments, function.parameters) {
                try require(argument, parameter.type)
            }
            return function.returnType
        case .switchSelect(let selector, let matches, let values, let fallback):
            guard matches.count == values.count, (1...MaterialGraph.maximumSwitchCases).contains(matches.count) else {
                throw Failure.invalidSwitch(id)
            }
            let selectorType = try visit(selector)
            guard selectorType != .structure else {
                throw Failure.typeMismatch(node: id, input: selector, expected: .scalar, actual: selectorType)
            }
            for match in matches { try require(match, selectorType) }
            let result = try visitAny(values[0])
            for value in values.dropFirst() { try requireMatching(value, values[0], result) }
            try requireMatching(fallback, values[0], result)
            adoptStructure(of: values[0], type: result)
            return result
        case .loop(let count, let iterations, let initial, let next, let exit, let skip):
            guard (1...MaterialGraph.maximumLoopIterations).contains(count) else {
                throw Failure.invalidLoopCount(node: id, count: count)
            }
            if let iterations { try require(iterations, .scalar) }
            let type = try visitAny(initial)
            try requireMatching(next, initial, type)
            if let exit { try require(exit, .scalar) }
            if let skip { try require(skip, .scalar) }
            adoptStructure(of: initial, type: type)
            return type
        case .discard(let condition, let value):
            try require(condition, .scalar)
            let type = try visitAny(value)
            adoptStructure(of: value, type: type)
            return type
        case .loopIndex(let loop), .loopValue(let loop):
            guard nodes.indices.contains(loop) else { throw Failure.invalidReference(node: id, referenced: loop) }
            guard case .loop(_, _, let initial, _, _, _) = nodes[loop] else { throw Failure.notALoop(node: id, loop: loop) }
            if case .loopIndex = nodes[id] { return .scalar }
            let type = try visitAny(initial)
            adoptStructure(of: initial, type: type)
            return type
        case .oscillator(_, let input), .derivative(_, let input):
            return try requireFloatOperand(input)
        case .rotateUV(let uv, let angle, let center):
            try require(uv, .vector2)
            try require(angle, .scalar)
            try require(center, .vector2)
            return .vector2
        case .spherizeUV(let uv, let strength, let center):
            try require(uv, .vector2)
            let amount = try visit(strength)
            guard amount == .scalar || amount == .vector2 else {
                throw Failure.typeMismatch(node: id, input: strength, expected: .vector2, actual: amount)
            }
            try require(center, .vector2)
            return .vector2
        case .polar(let uv, let center):
            try require(uv, .vector2)
            try require(center, .vector2)
            return .vector2
        case .checker(let uv):
            try require(uv, .vector2)
            return .scalar
        case .triplanarWeights(let normal, let sharpness):
            try require(normal, .vector3)
            try require(sharpness, .scalar)
            return .vector3
        case .triplanarSample(let slots, let position, let normal, let sharpness):
            guard slots.count == 1 || slots.count == 3 else { throw Failure.invalidPattern(id) }
            for slot in slots { try requireTexture(slot) }
            try require(position, .vector3)
            try require(normal, .vector3)
            try require(sharpness, .scalar)
            return .vector4
        case .parallaxOcclusion(let slot, let coordinates, let view, let depth, let steps):
            guard (1...MaterialGraph.maximumParallaxSteps).contains(steps) else { throw Failure.invalidPattern(id) }
            try requireTexture(slot)
            try require(coordinates, .vector2)
            try require(view, .vector3)
            try require(depth, .scalar)
            return .vector2
        case .bumpMap(let slot, _, _, let scale):
            try requireTexture(slot)
            guard declarations.textures[slot].content == .data else {
                throw Failure.invalidDeclaration("A bump map needs a linear data texture.")
            }
            try require(scale, .scalar)
            return .vector3
        case .bits(_, let a, let b):
            try require(a, .scalar)
            try require(b, .scalar)
            return .scalar
        case .bitcast(let input):
            switch try visitAny(input) {
            case .scalar: return .int
            case .int: return .scalar
            case let actual:
                throw Failure.typeMismatch(node: id, input: input, expected: .scalar, actual: actual)
            }
        case .countOneBits(let input):
            let source = try visitAny(input)
            guard source == .scalar || source == .int else {
                throw Failure.typeMismatch(node: id, input: input, expected: .scalar, actual: source)
            }
            return .int
        case .numericCast(let input):
            switch try visitAny(input) {
            case .scalar: return .int
            case .int: return .scalar
            case let actual:
                throw Failure.typeMismatch(node: id, input: input, expected: .scalar, actual: actual)
            }
        case .hash(let input):
            let source = try visit(input)
            guard source != .int, source != .bool, source != .structure else {
                throw Failure.typeMismatch(node: id, input: input, expected: .scalar, actual: source)
            }
            return .scalar
        case .interleavedGradientNoise(let pixel):
            try require(pixel, .vector2)
            return .scalar
        case .matrix(let columns):
            guard columns.count == 3 || columns.count == 4 else { throw Failure.invalidMatrix(id) }
            for column in columns { try require(column, columns.count == 3 ? .vector3 : .vector4) }
            return columns.count == 3 ? .matrix3 : .matrix4
        case .matrixMultiply(let a, let b):
            let left = try requireMatrix(a)
            switch (left, try visitAny(b)) {
            case (.matrix3, .matrix3): return .matrix3
            case (.matrix3, .vector3): return .vector3
            case (.matrix4, .matrix4): return .matrix4
            case (.matrix4, .vector4): return .vector4
            default: throw Failure.invalidMatrix(id)
            }
        case .matrixUnary(let operation, let input):
            let type = try requireMatrix(input)
            switch operation {
            case .transpose, .inverse: return type
            case .determinant: return .scalar
            case .upperLeft:
                guard type == .matrix4 else { throw Failure.invalidMatrix(id) }
                return .matrix3
            }
        case .matrixColumn(let input, let index):
            let type = try requireMatrix(input)
            let count = type == .matrix3 ? 3 : 4
            guard (0..<count).contains(index) else { throw Failure.invalidMatrix(id) }
            return count == 3 ? .vector3 : .vector4
        case .constructStruct(let index, let fields):
            guard declarations.structs.indices.contains(index) else { throw Failure.invalidStruct(id) }
            let structure = declarations.structs[index]
            guard fields.count == structure.fields.count else {
                throw Failure.argumentCount(node: id, expected: structure.fields.count, actual: fields.count)
            }
            for (input, field) in zip(fields, structure.fields) {
                switch field.content {
                case .value(let expected):
                    try require(input, expected)
                case .structure(let nested):
                    let actual = try visitAny(input)
                    guard actual == .structure, identity.index[input] == nested else {
                        throw Failure.structMismatch(node: id, expected: nested, actual: identity.index[input] ?? -1)
                    }
                }
            }
            identity.index[id] = index
            return .structure
        case .structField(let value, let field):
            let actual = try visitAny(value)
            guard actual == .structure, let index = identity.index[value],
                  declarations.structs.indices.contains(index),
                  declarations.structs[index].fields.indices.contains(field) else {
                throw Failure.invalidStruct(id)
            }
            switch declarations.structs[index].fields[field].content {
            case .value(let type):
                return type
            case .structure(let nested):
                identity.index[id] = nested
                return .structure
            }
        }
    }

    static func inputs(of node: Node) -> [NodeID] {
        switch node {
        case .scalar, .integer, .boolean, .color, .vector2, .vector4, .parameter, .uv, .normal, .worldPosition, .input, .uniform:
            []
        case .sampleTexture(_, let coordinates), .sampleScene(let coordinates), .sampleDepth(let coordinates):
            [coordinates]
        case .sampleTextureLevel(_, let coordinates, let level):
            [coordinates, level]
        case .sampleTextureGradient(_, let coordinates, let ddx, let ddy):
            [coordinates, ddx, ddy]
        case .sampleCube(_, let direction):
            [direction]
        case .sampleSceneLevel(let coordinates, let level):
            [coordinates, level]
        case .viewportSafeUV(let coordinates, let fallback):
            [coordinates, fallback]
        case .softParticleOpacity(let opacity, let distance, let contrast, let depth):
            [opacity, distance, contrast, depth]
        case .viewPositionFromDepth(let coordinates, let depth, let inverse):
            [coordinates, depth, inverse]
        case .normalFromDepth(let coordinates, let inverse, _):
            [coordinates, inverse]
        case .denoise(let coordinates, let inverse, let settings, let sampleIndex, _, _):
            [coordinates, inverse, settings, sampleIndex]
        case .linearDepth(let coordinates, let depth, let inverse, let near, let far):
            [coordinates, depth, inverse, near, far]
        case .add(let a, let b), .subtract(let a, let b), .multiply(let a, let b), .divide(let a, let b),
             .binary(_, let a, let b):
            [a, b]
        case .mix(let a, let b, let c), .smoothstep(let a, let b, let c), .noise(let a, let b, let c),
             .rgb(let a, let b, let c), .clamp(let a, let b, let c), .select(let a, let b, let c):
            [a, b, c]
        case .sine(let a), .cosine(let a), .floor(let a), .fract(let a), .clamp01(let a), .unary(_, let a),
             .swizzle(let a, _), .latticeNoise(_, let a), .fractalNoise(_, let a, _, _, _), .cellularNoise(let a),
             .simplexNoise(let a), .curlNoise(let a):
            [a]
        case .remap(let a, let b, let c, let d, let e):
            [a, b, c, d, e]
        case .combine(let inputs), .function(_, let inputs), .matrix(let inputs):
            inputs
        case .branch(let a, let b, let c), .rotateUV(let a, let b, let c), .spherizeUV(let a, let b, let c),
             .triplanarSample(_, let a, let b, let c), .parallaxOcclusion(_, let a, let b, let c, _):
            [a, b, c]
        case .switchSelect(let selector, let matches, let values, let fallback):
            [selector] + matches + values + [fallback]
        case .discard(let condition, let value):
            [condition, value]
        // A loop's operands: initial, next, exit, skip, then the dynamic count.
        case .loop(_, let iterations, let initial, let next, let exit, let skip):
            [initial, next] + [exit, skip, iterations].compactMap { $0 }
        // Loop nodes are scope references, not data dependencies.
        case .loopIndex, .loopValue:
            []
        case .bitcast(let value), .countOneBits(let value), .numericCast(let value), .structField(let value, _):
            [value]
        case .constructStruct(_, let fields):
            fields
        case .oscillator(_, let a), .checker(let a), .derivative(_, let a), .hash(let a),
             .interleavedGradientNoise(let a), .matrixUnary(_, let a), .matrixColumn(let a, _),
             .bumpMap(_, _, _, let a):
            [a]
        case .polar(let a, let b), .triplanarWeights(let a, let b), .bits(_, let a, let b), .matrixMultiply(let a, let b),
             .animatedVoronoi(let a, let b):
            [a, b]
        }
    }

    private static func isAvailable(
        _ node: Node, in stage: MaterialGraph.Stage, allowsInput: (MaterialGraph.Input) -> Bool,
        allowsSurfaceViewportSamples: Bool
    ) -> Bool {
        func provides(_ input: MaterialGraph.Input) -> Bool { input.isAvailable(in: stage) && allowsInput(input) }
        switch node {
        case .parameter: return provides(.parameters)
        case .uv: return provides(.uv0)
        case .normal: return provides(.normal)
        case .worldPosition: return provides(.worldPosition)
        case .input(let input): return provides(input)
        case .sampleScene, .sampleSceneLevel, .sampleDepth:
            return stage == .postProcess || (stage == .surface && allowsSurfaceViewportSamples)
        case .viewportSafeUV, .softParticleOpacity: return stage == .surface && allowsSurfaceViewportSamples
        case .viewPositionFromDepth, .normalFromDepth, .linearDepth, .denoise:
            return stage == .postProcess
        case .derivative: return stage == .surface || stage == .postProcess
        case .discard: return stage == .surface
        case .bumpMap: return stage == .surface && provides(.tangent)
        default: return true
        }
    }

    private static func swizzleCount(_ components: String, sourceComponents: Int) -> Int? {
        let letters = Array(components)
        guard (1...4).contains(letters.count) else { return nil }
        for set in ["xyzw", "rgba"] {
            let alphabet = Array(set)
            let indices = letters.compactMap { alphabet.firstIndex(of: $0) }
            if indices.count == letters.count, indices.allSatisfy({ $0 < sourceComponents }) {
                return letters.count
            }
        }
        return nil
    }

    // MARK: - Expressions

    private func expression(_ id: NodeID, prefix: String) -> String {
        let type = types[id]
        let t = type.metalType
        func n(_ input: NodeID) -> String { "n\(input)" }
        func cast(_ input: NodeID) -> String { types[input] == type ? n(input) : "\(t)(\(n(input)))" }

        switch nodes[id] {
        case .scalar(let value): return Self.literal(value)
        case .integer(let value): return "int(\(value))"
        case .boolean(let value): return value ? "true" : "false"
        case .color(let value): return "float3(\(Self.literal(value.x)), \(Self.literal(value.y)), \(Self.literal(value.z)))"
        case .vector2(let value): return "float2(\(Self.literal(value.x)), \(Self.literal(value.y)))"
        case .vector4(let value):
            return "float4(\(Self.literal(value.x)), \(Self.literal(value.y)), \(Self.literal(value.z)), \(Self.literal(value.w)))"
        case .parameter(let component): return "input.parameters.\(component.rawValue)"
        case .uv(let component): return component == .u ? "input.uv0.x" : "input.uv0.y"
        case .normal(let axis): return "frostfield_graph_unit(input.normal).\(axis.rawValue)"
        case .worldPosition(let axis): return "input.worldPosition.\(axis.rawValue)"
        case .input(let input):
            guard input.readsClock else { return input.expression }
            let component = input == .time ? "x" : input == .deltaTime ? "y" : "z"
            return "frostfield_graph_uniform(resources, \(declarations.uniformValueSlotCount)u).\(component)"
        case .uniform(let index):
            let origin = declarations.uniformSlotOffset(index)
            switch type {
            case .matrix3:
                let columns = (0..<3).map { "frostfield_graph_uniform(resources, \(origin + $0)u).xyz" }
                return "float3x3(\(columns.joined(separator: ", ")))"
            case .matrix4:
                let columns = (0..<4).map { "frostfield_graph_uniform(resources, \(origin + $0)u)" }
                return "float4x4(\(columns.joined(separator: ", ")))"
            case .int:
                return "as_type<int>(frostfield_graph_uniform(resources, \(origin)u).x)"
            case .bool:
                return "frostfield_graph_uniform(resources, \(origin)u).x != 0.0f"
            default:
                let swizzle = ["", ".x", ".xy", ".xyz", ""][type.componentCount]
                return "frostfield_graph_uniform(resources, \(origin)u)\(swizzle)"
            }
        case .sampleTexture(let slot, let coordinates):
            let texture = declarations.textures[slot]
            return "frostfield_graph_sample(resources, \(slot)u, \(texture.addressing.code)u, "
                + "\(texture.filtering.code)u, \(n(coordinates)))"
        case .sampleTextureLevel(let slot, let coordinates, let level):
            let texture = declarations.textures[slot]
            return "frostfield_graph_sample(resources, \(slot)u, \(texture.addressing.code)u, "
                + "\(texture.filtering.code)u, \(n(coordinates)), \(n(level)))"
        case .sampleTextureGradient(let slot, let coordinates, let ddx, let ddy):
            let texture = declarations.textures[slot]
            return "frostfield_graph_sample(resources, \(slot)u, \(texture.addressing.code)u, "
                + "\(texture.filtering.code)u, \(n(coordinates)), \(n(ddx)), \(n(ddy)))"
        case .sampleCube(let slot, let direction):
            let texture = declarations.textures[slot]
            return "frostfield_graph_sample(resources, \(slot)u, \(texture.addressing.code)u, "
                + "\(texture.filtering.code)u, \(n(direction)))"
        case .sampleScene(let coordinates): return "frostfield_graph_scene(resources, \(n(coordinates)))"
        case .sampleSceneLevel(let coordinates, let level): return "frostfield_graph_scene(resources, \(n(coordinates)), \(n(level)))"
        case .sampleDepth(let coordinates): return "frostfield_graph_depth(resources, \(n(coordinates)))"
        case .viewportSafeUV(let coordinates, let fallback):
            return "frostfield_graph_viewport_safe_uv(input, resources, \(n(coordinates)), \(n(fallback)))"
        case .softParticleOpacity(let opacity, let distance, let contrast, let depth):
            return "frostfield_graph_soft_particle(input, \(n(opacity)), \(n(distance)), \(n(contrast)), \(n(depth)))"
        case .viewPositionFromDepth(let coordinates, let depth, let inverse):
            return "frostfield_post_view_position(\(n(coordinates)), \(n(depth)), \(n(inverse)))"
        case .normalFromDepth(let coordinates, let inverse, let reversed):
            return "frostfield_graph_depth_normal(resources, \(n(coordinates)), \(n(inverse)), \(reversed))"
        case .denoise(let coordinates, let inverse, let settings, let sampleIndex, let normalSlot, let reversed):
            return "frostfield_graph_denoise(resources, \(n(coordinates)), \(n(inverse)), \(n(settings)), "
                + "\(n(sampleIndex)), \(normalSlot ?? -1), \(reversed))"
        case .linearDepth(let coordinates, let depth, let inverse, let near, let far):
            return "frostfield_post_linear_depth(\(n(coordinates)), \(n(depth)), \(n(inverse)), \(n(near)), \(n(far)))"
        case .add(let a, let b): return "\(n(a)) + \(n(b))"
        case .subtract(let a, let b): return "\(n(a)) - \(n(b))"
        case .multiply(let a, let b): return "\(n(a)) * \(n(b))"
        case .divide(let a, let b):
            if types[id] == .int { return "\(n(b)) != 0 ? \(n(a)) / \(n(b)) : 0" }
            return "frostfield_graph_finite(\(n(b))) && \(n(b)) != 0.0f ? \(n(a)) / \(n(b)) : \(t)(0)"
        case .mix(let a, let b, let factor): return "mix(\(n(a)), \(n(b)), saturate(\(n(factor))))"
        case .sine(let input): return "sin(\(n(input)))"
        case .cosine(let input): return "cos(\(n(input)))"
        case .floor(let input): return "floor(\(n(input)))"
        case .fract(let input): return "fract(\(n(input)))"
        case .smoothstep(let edge0, let edge1, let value):
            return "frostfield_graph_smoothstep(\(n(edge0)), \(n(edge1)), \(n(value)))"
        case .noise(let x, let y, let z): return "frostfield_graph_value_noise(float3(\(n(x)), \(n(y)), \(n(z))))"
        case .clamp01(let input): return "saturate(\(n(input)))"
        case .rgb(let red, let green, let blue): return "float3(\(n(red)), \(n(green)), \(n(blue)))"
        case .unary(let operation, let input):
            let x = n(input)
            let source = types[input].metalType
            switch operation {
            case .abs: return "abs(\(x))"
            case .sign: return "sign(\(x))"
            case .floor: return "floor(\(x))"
            case .ceil: return "ceil(\(x))"
            case .round: return "round(\(x))"
            case .fract: return "fract(\(x))"
            case .sqrt: return "sqrt(max(\(x), \(t)(0)))"
            case .inverseSqrt: return "frostfield_graph_rsqrt(\(x))"
            case .exp: return "exp(\(x))"
            case .exp2: return "exp2(\(x))"
            case .log: return "log(max(\(x), \(t)(1e-30f)))"
            case .log2: return "log2(max(\(x), \(t)(1e-30f)))"
            case .sin: return "sin(\(x))"
            case .cos: return "cos(\(x))"
            case .tan: return "tan(\(x))"
            case .asin: return "asin(clamp(\(x), \(t)(-1), \(t)(1)))"
            case .acos: return "acos(clamp(\(x), \(t)(-1), \(t)(1)))"
            case .atan: return "atan(\(x))"
            case .negate: return "-\(x)"
            case .oneMinus: return "\(t)(1) - \(x)"
            case .saturate: return "saturate(\(x))"
            case .normalize: return "frostfield_graph_unit(\(x))"
            case .length: return source == "float" ? "abs(\(x))" : "length(\(x))"
            case .luminance: return "dot(\(x), float3(0.2126f, 0.7152f, 0.0722f))"
            case .hsvToRGB: return "frostfield_graph_hsv_to_rgb(\(x))"
            case .rgbToHSV: return "frostfield_graph_rgb_to_hsv(\(x))"
            case .srgbToLinear: return "frostfield_graph_srgb_to_linear(\(x))"
            case .linearToSRGB: return "frostfield_graph_linear_to_srgb(\(x))"
            }
        case .binary(let operation, let a, let b):
            let operands = types[a].metalType
            switch operation {
            case .minimum: return "min(\(n(a)), \(n(b)))"
            case .maximum: return "max(\(n(a)), \(n(b)))"
            case .power: return "pow(max(\(n(a)), \(t)(0)), \(n(b)))"
            case .modulo: return "frostfield_graph_mod(\(n(a)), \(n(b)))"
            case .atan2: return "atan2(\(n(a)), \(n(b)))"
            case .step: return "step(\(n(a)), \(n(b)))"
            case .dot: return operands == "float" ? "\(n(a)) * \(n(b))" : "dot(\(n(a)), \(n(b)))"
            case .cross: return "cross(\(n(a)), \(n(b)))"
            case .distance: return operands == "float" ? "abs(\(n(a)) - \(n(b)))" : "distance(\(n(a)), \(n(b)))"
            case .reflect: return "reflect(\(n(a)), frostfield_graph_unit(\(n(b))))"
            case .greaterThan: return "select(\(t)(0), \(t)(1), \(n(a)) > \(n(b)))"
            case .lessThan: return "select(\(t)(0), \(t)(1), \(n(a)) < \(n(b)))"
            }
        case .clamp(let value, let minimum, let maximum):
            return "clamp(\(n(value)), \(cast(minimum)), \(cast(maximum)))"
        case .select(let condition, let ifTrue, let ifFalse):
            let conditionTest = types[condition] == .bool ? n(condition) : "\(n(condition)) > 0.5f"
            return "\(conditionTest) ? \(n(ifTrue)) : \(n(ifFalse))"
        case .remap(let value, let fromMinimum, let fromMaximum, let toMinimum, let toMaximum):
            return "frostfield_graph_remap(\(n(value)), \(n(fromMinimum)), \(n(fromMaximum)), \(n(toMinimum)), \(n(toMaximum)))"
        case .swizzle(let input, let components):
            if types[input] == .scalar {
                return type == .scalar ? n(input) : "\(t)(\(n(input)))"
            }
            return "\(n(input)).\(components)"
        case .combine(let inputs):
            return "\(t)(\(inputs.map(n).joined(separator: ", ")))"
        case .latticeNoise(let kind, let position):
            return kind == .value ? "frostfield_graph_value_noise(\(n(position)))"
                : "frostfield_graph_gradient_noise(\(n(position)))"
        case .fractalNoise(let kind, let position, let octaves, let lacunarity, let gain):
            return "frostfield_graph_fbm(\(n(position)), \(octaves), \(Self.literal(lacunarity)), "
                + "\(Self.literal(gain)), \(kind == .value ? 0 : 1)u)"
        case .cellularNoise(let position):
            return "frostfield_graph_cellular(\(n(position)))"
        case .simplexNoise(let position):
            return "\(prefix)_simplex(\(n(position)))"
        case .curlNoise(let position):
            return "\(prefix)_curl(\(n(position)))"
        case .animatedVoronoi(let position, let time):
            return "\(prefix)_voronoi(\(n(position)), \(n(time)))"
        case .function(let index, let arguments):
            return "\(prefix)_fn_\(declarations.functions[index].name)(\(arguments.map(n).joined(separator: ", ")))"
        case .branch, .loop, .switchSelect, .discard:
            // Generated as blocks by `emit(scope:schedule:indent:prefix:into:)`.
            return ""
        case .loopIndex(let loop): return "float(n\(loop)_i)"
        case .loopValue(let loop): return "n\(loop)"
        case .oscillator(let kind, let input):
            let x = n(input)
            switch kind {
            case .sine: return "0.5f - 0.5f * cos(6.28318530717958647692f * \(x))"
            case .square: return "select(\(t)(0), \(t)(1), fract(\(x)) >= \(t)(0.5f))"
            case .triangle: return "abs(fract(\(x) + 0.5f) * 2.0f - 1.0f)"
            case .sawtooth: return "fract(\(x))"
            }
        case .rotateUV(let uv, let angle, let center):
            return "frostfield_graph_rotate_uv(\(n(uv)), \(n(angle)), \(n(center)))"
        case .spherizeUV(let uv, let strength, let center):
            return "frostfield_graph_spherize_uv(\(n(uv)), float2(\(n(strength))), \(n(center)))"
        case .polar(let uv, let center): return "frostfield_graph_polar(\(n(uv)), \(n(center)))"
        case .checker(let uv): return "frostfield_graph_checker(\(n(uv)))"
        case .triplanarWeights(let normal, let sharpness):
            return "frostfield_graph_triplanar_weights(\(n(normal)), \(n(sharpness)))"
        case .triplanarSample(let slots, let position, let normal, let sharpness):
            let projections = zip(slots.count == 1 ? Array(repeating: slots[0], count: 3) : slots, ["yz", "zx", "xy"])
            let samples = projections.map { slot, plane -> String in
                let texture = declarations.textures[slot]
                return "frostfield_graph_sample(resources, \(slot)u, \(texture.addressing.code)u, "
                    + "\(texture.filtering.code)u, \(n(position)).\(plane))"
            }
            return "frostfield_graph_triplanar_blend(frostfield_graph_triplanar_weights(\(n(normal)), \(n(sharpness))), "
                + samples.joined(separator: ", ") + ")"
        case .parallaxOcclusion(_, let coordinates, let view, let depth, _):
            return "\(prefix)_parallax_\(id)(resources, \(n(coordinates)), \(n(view)), \(n(depth)))"
        case .bumpMap(let slot, let set, let transform, let scale):
            let texture = declarations.textures[slot]
            let (row0, row1) = transform.rows
            func metalRow(_ row: SIMD4<Float>) -> String {
                "float4(\(Self.literal(row.x)), \(Self.literal(row.y)), \(Self.literal(row.z)), 0.0f)"
            }
            return "frostfield_graph_bump(input, resources, \(slot)u, \(texture.addressing.code)u, "
                + "\(texture.filtering.code)u, \(set.rawValue)u, \(metalRow(row0)), \(metalRow(row1)), \(n(scale)))"
        case .derivative(let kind, let input):
            switch kind {
            case .dx: return "dfdx(\(n(input)))"
            case .dy: return "dfdy(\(n(input)))"
            case .width: return "fwidth(\(n(input)))"
            }
        case .bits(let operation, let a, let b):
            let code = MaterialGraph.BitOperation.allCases.firstIndex(of: operation)!
            return "frostfield_graph_bits(\(n(a)), \(n(b)), \(code)u)"
        case .bitcast(let input):
            return "as_type<\(t)>(\(n(input)))"
        case .countOneBits(let input):
            return "int(popcount(as_type<uint>(\(n(input)))))"
        case .numericCast(let input):
            if type == .int {
                return "frostfield_graph_finite(\(n(input))) ? int(\(n(input))) : 0"
            }
            return "float(\(n(input)))"
        case .hash(let input):
            // Adding positive zero turns negative zero into positive zero.
            let x = "\(n(input)) + 0.0f"
            switch types[input] {
            case .scalar: return "frostfield_graph_hash_bits(uint4(as_type<uint>(\(x)), 0u, 0u, 0u))"
            case .vector2: return "frostfield_graph_hash_bits(uint4(as_type<uint2>(\(x)), 0u, 0u))"
            case .vector3: return "frostfield_graph_hash_bits(uint4(as_type<uint3>(\(x)), 0u))"
            default: return "frostfield_graph_hash_bits(as_type<uint4>(\(x)))"
            }
        case .interleavedGradientNoise(let pixel): return "frostfield_graph_ign(\(n(pixel)))"
        case .matrix(let columns): return "\(t)(\(columns.map(n).joined(separator: ", ")))"
        case .matrixMultiply(let a, let b): return "\(n(a)) * \(n(b))"
        case .matrixUnary(let operation, let input):
            switch operation {
            case .transpose: return "transpose(\(n(input)))"
            case .inverse: return "frostfield_graph_inverse(\(n(input)))"
            case .determinant: return "determinant(\(n(input)))"
            case .upperLeft: return "float3x3(\(n(input))[0].xyz, \(n(input))[1].xyz, \(n(input))[2].xyz)"
            }
        case .matrixColumn(let input, let index): return "\(n(input))[\(index)]"
        case .constructStruct(_, let fields):
            return "{ \(fields.map(n).joined(separator: ", ")) }"
        case .structField(let value, let field):
            let index = structIndices[value] ?? 0
            let name = declarations.structs[index].fields[field].name
            return "\(n(value)).\(name)"
        }
    }

    static func literal(_ value: Float) -> String {
        var text = "\(value)"
        if !text.contains(".") && !text.contains("e") { text += ".0" }
        return text + "f"
    }

    /// Printable ASCII for a `//` comment. Other characters become `?`, and a
    /// backslash becomes `/`, because a backslash that ends the line would splice
    /// the next generated statement into the comment.
    private static func commentSafe(_ label: String) -> String {
        String(label.unicodeScalars.prefix(80).map { scalar -> Character in
            if scalar == "\\" { return "/" }
            return (32...126).contains(scalar.value) ? Character(scalar) : "?"
        })
    }

    // MARK: - Declarations

    private static func validate(_ declarations: ShaderDeclarations) throws {
        func fail(_ detail: String) -> Failure { Failure.invalidDeclaration(detail) }
        guard declarations.uniforms.count <= MaterialGraph.maximumUniformCount else {
            throw fail("A graph may declare at most \(MaterialGraph.maximumUniformCount) uniforms.")
        }
        guard declarations.textures.count <= MaterialGraph.maximumTextureCount else {
            throw fail("A graph may declare at most \(MaterialGraph.maximumTextureCount) textures.")
        }
        guard declarations.functions.count <= MaterialGraph.maximumFunctionCount else {
            throw fail("A graph may declare at most \(MaterialGraph.maximumFunctionCount) functions.")
        }
        guard declarations.structs.count <= MaterialGraph.maximumStructCount else {
            throw fail("A graph may declare at most \(MaterialGraph.maximumStructCount) structs.")
        }
        var uniformNames = Set<String>()
        for uniform in declarations.uniforms {
            guard !uniform.name.isEmpty, uniform.name.count <= 64, uniformNames.insert(uniform.name).inserted else {
                throw fail("Uniform names must be unique and 1–64 characters: \"\(uniform.name)\".")
            }
            guard uniform.type != .structure else {
                throw fail("Uniform \"\(uniform.name)\" cannot be a struct. Build the struct in the graph.")
            }
            let finite = uniform.type == .int || uniform.columns.allSatisfy { column in
                column.x.isFinite && column.y.isFinite && column.z.isFinite && column.w.isFinite
            }
            guard finite else {
                throw fail("Uniform \"\(uniform.name)\" needs a finite default value.")
            }
        }
        var textureNames = Set<String>()
        for texture in declarations.textures {
            guard !texture.name.isEmpty, texture.name.count <= 64, textureNames.insert(texture.name).inserted else {
                throw fail("Texture names must be unique and 1–64 characters: \"\(texture.name)\".")
            }
        }
        var functionNames = Set<String>()
        for function in declarations.functions {
            guard isIdentifier(function.name, limit: 64), functionNames.insert(function.name).inserted else {
                throw fail("Function names must be unique Metal identifiers: \"\(function.name)\".")
            }
            var parameterNames = Set<String>()
            for parameter in function.parameters {
                guard isIdentifier(parameter.name, limit: 64), parameterNames.insert(parameter.name).inserted else {
                    throw fail("Function \(function.name) has an invalid parameter name \"\(parameter.name)\".")
                }
            }
            guard function.parameters.count <= 8 else {
                throw fail("Function \(function.name) may take at most eight parameters.")
            }
            if function.returnType == .structure || function.parameters.contains(where: { $0.type == .structure }) {
                throw fail("Function \(function.name) cannot use a struct type. Construct the struct in the graph and pass its fields.")
            }
            let body = function.body
            guard !body.isEmpty, body.utf8.count <= ShaderFunction.maximumBodyLength, !body.contains("#"),
                  balancedBraces(body) else {
                throw fail("Function \(function.name) needs a nonempty body with balanced braces and no preprocessor directives.")
            }
        }
        var structNames = Set<String>()
        for (index, structure) in declarations.structs.enumerated() {
            guard structure.index == index, isIdentifier(structure.name, limit: 64),
                  structNames.insert(structure.name).inserted else {
                throw fail("Struct names must be unique Metal identifiers and stay in declaration order: \"\(structure.name)\".")
            }
            guard (1...MaterialGraph.maximumStructFields).contains(structure.fields.count) else {
                throw fail("Struct \(structure.name) needs 1–\(MaterialGraph.maximumStructFields) fields.")
            }
            var fieldNames = Set<String>()
            for field in structure.fields {
                guard isIdentifier(field.name, limit: 64), fieldNames.insert(field.name).inserted else {
                    throw fail("Struct \(structure.name) has an invalid field name \"\(field.name)\".")
                }
                switch field.content {
                case .value(let type):
                    guard type != .structure else {
                        throw fail("Struct \(structure.name) field \(field.name) needs a concrete type or an earlier struct.")
                    }
                case .structure(let nested):
                    guard declarations.structs.indices.contains(nested), nested < index else {
                        throw fail("Struct \(structure.name) field \(field.name) must name a struct declared before it.")
                    }
                }
            }
        }
    }

    private static func balancedBraces(_ body: String) -> Bool {
        var depth = 0
        for character in body {
            if character == "{" { depth += 1 }
            if character == "}" { depth -= 1 }
            if depth < 0 { return false }
        }
        return depth == 0
    }

    /// ASCII letters, digits, and single separating underscores, starting with a
    /// letter. C++ and Metal keywords, vector type names, and names that may be
    /// Metal predefined macros are rejected; see `isMacroName(_:)`.
    static func isIdentifier(_ name: String, limit: Int = 128) -> Bool {
        let bytes = name.utf8
        func isLetter(_ byte: UInt8) -> Bool { (65...90).contains(byte) || (97...122).contains(byte) }
        guard bytes.count <= limit, let first = bytes.first, isLetter(first),
              bytes.dropFirst().allSatisfy({ isLetter($0) || $0 == 95 || (48...57).contains($0) }),
              !name.contains("__"), !name.hasSuffix("_"), !reservedNames.contains(name),
              !isMacroName(name) else { return false }
        return true
    }

    /// Whether a name may be a macro that `<metal_stdlib>` defines. The preprocessor
    /// would replace it in generated source, so a parameter named `FLT_MAX` fails to
    /// compile. Names that start with an underscore or contain `__`, such as
    /// `__METAL_VERSION__` and the `__HAVE_*` feature macros, already fail
    /// `isIdentifier`. This rule covers the rest:
    /// - the lowercase function-like macros `assert` and `is_function_constant_defined`;
    /// - `INFINITY`, `NAN`, `NULL`, `MAXFLOAT`, `MAXHALF`, `MAXDOUBLE`, and `MAXBFLOAT`;
    /// - all-caps names (uppercase letters, digits, and underscores) that begin with a
    ///   macro family prefix: the float limits `FLT_`, `DBL_`, `HALF_`, and `BFLT_`;
    ///   math constants `M_`, such as `M_PI_F` and `M_SQRT2_H`; `METAL_`; `TARGET_OS_`;
    ///   `HUGE_VAL`; `FP_ILOGB`; `HAS_INTERFACE`; and the integer limits `CHAR_`,
    ///   `SCHAR_`, `UCHAR_`, `SHRT_`, `USHRT_`, `INT_`, `UINT_`, `LONG_`, `ULONG_`,
    ///   `LLONG_`, and `ULLONG_`.
    ///
    /// Families are matched by prefix so that macros a header defines only for some
    /// language versions or targets, and future members, are also rejected. Other
    /// all-caps names, such as `N`, `L`, or `UV`, remain valid.
    private static func isMacroName(_ name: String) -> Bool {
        if macroNames.contains(name) { return true }
        let isAllCaps = name.utf8.allSatisfy { (65...90).contains($0) || (48...57).contains($0) || $0 == 95 }
        return isAllCaps && macroPrefixes.contains { name.hasPrefix($0) }
    }

    private static let macroNames: Set<String> = [
        "assert", "is_function_constant_defined",
        "INFINITY", "NAN", "NULL", "MAXFLOAT", "MAXHALF", "MAXDOUBLE", "MAXBFLOAT",
    ]

    private static let macroPrefixes = [
        "FLT_", "DBL_", "HALF_", "BFLT_", "M_", "METAL_", "TARGET_OS_", "HUGE_VAL", "FP_ILOGB", "HAS_INTERFACE",
        "CHAR_", "SCHAR_", "UCHAR_", "SHRT_", "USHRT_", "INT_", "UINT_", "LONG_", "ULONG_", "LLONG_", "ULLONG_",
    ]

    private static let reservedNames: Set<String> = {
        let keywords = """
        alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t char32_t
        class compl concept const consteval constexpr constinit const_cast continue co_await co_return co_yield
        decltype default delete do double dynamic_cast else enum explicit export extern false float for friend
        goto if inline int long mutable namespace new noexcept not not_eq nullptr operator or or_eq private
        protected public register reinterpret_cast requires return short signed sizeof static static_assert
        static_cast struct switch template this thread_local throw true try typedef typeid typename union
        unsigned using virtual void volatile wchar_t while xor xor_eq
        constant device fragment half kernel metal object mesh ray_data thread threadgroup threadgroup_imageblock
        uchar ushort uint ulong vertex input resources surface
        """
        var names = Set(keywords.split(whereSeparator: \.isWhitespace).map(String.init))
        for scalar in ["bool", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong", "half", "float"] {
            for width in 2...4 { names.insert("\(scalar)\(width)") }
        }
        return names
    }()

    // MARK: - Diagnostics

    /// Summarizes Metal compiler errors and the generated lines they point to.
    static func diagnose(_ error: any Error, source: String) -> String {
        let message = (error as NSError).localizedDescription
        let lines = source.components(separatedBy: "\n")
        var notes: [String] = []
        let pattern = /program_source:(\d+):(\d+): error: ([^\n]*)/
        for match in message.matches(of: pattern).prefix(5) {
            guard let line = Int(match.output.1) else { continue }
            var note = "Line \(line): \(match.output.3)"
            if lines.indices.contains(line - 1) {
                note += " — `\(lines[line - 1].trimmingCharacters(in: .whitespaces))`"
            }
            notes.append(note)
        }
        return notes.isEmpty ? message : notes.joined(separator: "\n")
    }

    // MARK: - Support source

    /// Declarations shared by all generated graph functions. Renderers place this
    /// first, then define `FrostfieldGraphResources` and any resource hooks the
    /// graph calls: `frostfield_graph_uniform`, `frostfield_graph_sample`,
    /// `frostfield_graph_scene`, and `frostfield_graph_depth`.
    static let supportSource = """
    // Frostfield shader graph support.
    struct FrostfieldSurfaceInput {
        float2 uv0;
        float2 uv1;
        float3 worldPosition;
        float3 normal;
        float3 vertexNormal;
        float3 tangent;
        float3 viewDirection;
        float4 parameters;
        float3 baseColor;
        float metallic;
        float roughness;
        float3 emission;
        float3 bitangent;
        float3 cameraPosition;
        float2 cameraClip; // near, far
        float directionalShadowVisibility;
        float2 screenUV;
        float2 viewportSize;
        float2 screenCoordinate;
        float2 screenSize;
        float4x4 viewMatrix;
        float4x4 projectionMatrix;
        float4x4 modelMatrix;
        float3x3 normalMatrix;
        float2 particleDepth; // window depth, reciprocal clip W
        float3 depthRange;   // viewport near, viewport far, depth mode
    };

    static inline float3 frostfield_graph_to_tangent(thread const FrostfieldSurfaceInput& input, float3 normal) {
        return float3(dot(normal, input.tangent), dot(normal, input.bitangent), dot(normal, input.vertexNormal));
    }

    // Graph outputs overwrite the authored values a renderer stores here.
    struct FrostfieldSurfaceOutput {
        float3 baseColor;
        float metallic;
        float roughness;
        float3 emission;
        float transmissionMask;
        float occlusion;
        float3 normal; // Unit tangent-space normal when hasNormal is true.
        bool hasNormal;
        float opacity;   // Multiplier, initially one.
        float alphaTest; // Minus one when the graph leaves the cutoff unset.
        bool mask;       // Initially true.
    };

    struct FrostfieldVertexInput {
        float2 uv0;
        float2 uv1;
        float3 objectPosition;
        float3 objectNormal;
    };

    struct FrostfieldPostInput {
        float2 screenUV;
        float2 texelSize;
        float2 viewportSize;
        float2 screenCoordinate;
        float2 screenSize;
    };

    struct FrostfieldComputeInput {
        float2 gridUV;
        float2 texelSize;
        float elementIndex;
        float elementCount;
        float4 fields[4];
    };

    static inline bool frostfield_graph_finite(float value) {
        return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
    }

    static inline bool2 frostfield_graph_finite(float2 value) {
        return (as_type<uint2>(value) & uint2(0x7f800000u)) != uint2(0x7f800000u);
    }

    static inline bool3 frostfield_graph_finite(float3 value) {
        return (as_type<uint3>(value) & uint3(0x7f800000u)) != uint3(0x7f800000u);
    }

    static inline bool4 frostfield_graph_finite(float4 value) {
        return (as_type<uint4>(value) & uint4(0x7f800000u)) != uint4(0x7f800000u);
    }

    // Zero for a zero vector or any nonfinite component. Metal's max drops a NaN
    // operand, so every component is checked rather than only the largest.
    static inline float2 frostfield_graph_unit(float2 value) {
        float largest = max(abs(value.x), abs(value.y));
        return largest > 0.0f && all(frostfield_graph_finite(value)) ? normalize(value / largest) : float2(0);
    }

    static inline float3 frostfield_graph_unit(float3 value) {
        float largest = max(abs(value.x), max(abs(value.y), abs(value.z)));
        return largest > 0.0f && all(frostfield_graph_finite(value)) ? normalize(value / largest) : float3(0);
    }

    static inline float4 frostfield_graph_unit(float4 value) {
        float largest = max(max(abs(value.x), abs(value.y)), max(abs(value.z), abs(value.w)));
        return largest > 0.0f && all(frostfield_graph_finite(value)) ? normalize(value / largest) : float4(0);
    }

    static inline float frostfield_graph_smoothstep(float low, float high, float value) {
        if (!frostfield_graph_finite(low) || !frostfield_graph_finite(high) || !frostfield_graph_finite(value)) {
            return 0.0f;
        }
        return high > low ? smoothstep(low, high, value) : step(low, value);
    }

    template <typename T> static inline T frostfield_graph_rsqrt(T value) {
        return select(T(0), rsqrt(max(value, T(1e-30f))), value > T(0));
    }

    template <typename T> static inline T frostfield_graph_mod(T value, T divisor) {
        T result = value - divisor * floor(value / divisor);
        return select(result, T(0), divisor == T(0));
    }

    template <typename T> static inline T frostfield_graph_remap(T value, float low, float high, float to, float toHigh) {
        float span = high - low;
        return span != 0.0f && frostfield_graph_finite(span) ? T(to) + (value - T(low)) * ((toHigh - to) / span) : T(to);
    }

    static inline float3 frostfield_graph_hsv_to_rgb(float3 hsv) {
        float3 k = fract(float3(hsv.x) + float3(1.0f, 2.0f / 3.0f, 1.0f / 3.0f)) * 6.0f - 3.0f;
        return hsv.z * mix(float3(1.0f), saturate(abs(k) - 1.0f), saturate(hsv.y));
    }

    static inline float3 frostfield_graph_rgb_to_hsv(float3 c) {
        float4 k = float4(0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f);
        float4 p = mix(float4(c.bg, k.wz), float4(c.gb, k.xy), step(c.b, c.g));
        float4 q = mix(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
        float d = q.x - min(q.w, q.y);
        return float3(abs(q.z + (q.w - q.y) / (6.0f * d + 1e-10f)), d / (q.x + 1e-10f), q.x);
    }

    static inline float3 frostfield_graph_srgb_to_linear(float3 c) {
        float3 x = max(c, float3(0));
        return select(pow((x + 0.055f) / 1.055f, float3(2.4f)), x / 12.92f, x <= 0.04045f);
    }

    static inline float3 frostfield_graph_linear_to_srgb(float3 c) {
        float3 x = max(c, float3(0));
        return select(1.055f * pow(x, float3(1.0f / 2.4f)) - 0.055f, x * 12.92f, x <= 0.0031308f);
    }

    static inline float frostfield_graph_hash(int3 cell, uint seed = 0u) {
        uint3 bits = as_type<uint3>(cell);
        uint h = bits.x * 0x8da6b343u ^ bits.y * 0xd8163841u ^ bits.z * 0xcb1ab31fu ^ seed * 0x9e3779b9u;
        h ^= h >> 16u;
        h *= 0x7feb352du;
        h ^= h >> 15u;
        h *= 0x846ca68bu;
        h ^= h >> 16u;
        return float(h >> 8u) * 0x1p-24f;
    }

    // The noise functions spell out every multiply-add as fma, so the rounding does not
    // depend on compiler contraction and ShaderNoise reproduces them on the CPU.
    static inline float frostfield_graph_lerp(float a, float b, float t) {
        return fma(t, b - a, a);
    }

    static inline float frostfield_graph_dot3(float3 a, float3 b) {
        return fma(a.z, b.z, fma(a.y, b.y, a.x * b.x));
    }

    static inline float3 frostfield_graph_noise_point(float3 position) {
        float3 finitePosition = select(float3(0), position, frostfield_graph_finite(position));
        return clamp(finitePosition, float3(-1048576.0f), float3(1048576.0f));
    }

    static inline float frostfield_graph_value_noise(float3 position) {
        float3 point = frostfield_graph_noise_point(position);
        int3 cell = int3(floor(point));
        float3 f = fract(point);
        float3 weight = f * f * fma(float3(-2.0f), f, float3(3.0f));
        float lowerA = frostfield_graph_lerp(frostfield_graph_hash(cell), frostfield_graph_hash(cell + int3(1, 0, 0)), weight.x);
        float lowerB = frostfield_graph_lerp(frostfield_graph_hash(cell + int3(0, 1, 0)), frostfield_graph_hash(cell + int3(1, 1, 0)), weight.x);
        float upperA = frostfield_graph_lerp(frostfield_graph_hash(cell + int3(0, 0, 1)), frostfield_graph_hash(cell + int3(1, 0, 1)), weight.x);
        float upperB = frostfield_graph_lerp(frostfield_graph_hash(cell + int3(0, 1, 1)), frostfield_graph_hash(cell + int3(1, 1, 1)), weight.x);
        return frostfield_graph_lerp(
            frostfield_graph_lerp(lowerA, lowerB, weight.y), frostfield_graph_lerp(upperA, upperB, weight.y), weight.z
        );
    }

    static inline float frostfield_graph_corner(int3 cell, int3 offset, float3 local) {
        int3 corner = cell + offset;
        float3 gradient = fma(float3(
            frostfield_graph_hash(corner, 1u), frostfield_graph_hash(corner, 2u), frostfield_graph_hash(corner, 3u)
        ), float3(2.0f), float3(-1.0f));
        float lengthSquared = frostfield_graph_dot3(gradient, gradient);
        float3 unit = lengthSquared > 0.0f ? gradient / sqrt(lengthSquared) : float3(0);
        return frostfield_graph_dot3(unit, local - float3(offset));
    }

    static inline float frostfield_graph_gradient_noise(float3 position) {
        float3 point = frostfield_graph_noise_point(position);
        int3 cell = int3(floor(point));
        float3 local = fract(point);
        float3 w = local * local * local * fma(local, fma(local, float3(6.0f), float3(-15.0f)), float3(10.0f));
        float x00 = frostfield_graph_lerp(frostfield_graph_corner(cell, int3(0, 0, 0), local), frostfield_graph_corner(cell, int3(1, 0, 0), local), w.x);
        float x10 = frostfield_graph_lerp(frostfield_graph_corner(cell, int3(0, 1, 0), local), frostfield_graph_corner(cell, int3(1, 1, 0), local), w.x);
        float x01 = frostfield_graph_lerp(frostfield_graph_corner(cell, int3(0, 0, 1), local), frostfield_graph_corner(cell, int3(1, 0, 1), local), w.x);
        float x11 = frostfield_graph_lerp(frostfield_graph_corner(cell, int3(0, 1, 1), local), frostfield_graph_corner(cell, int3(1, 1, 1), local), w.x);
        float value = frostfield_graph_lerp(frostfield_graph_lerp(x00, x10, w.y), frostfield_graph_lerp(x01, x11, w.y), w.z);
        // Unit gradients bound 3D gradient noise by about ±0.866.
        return saturate(fma(0.57735f, value, 0.5f));
    }

    static inline float frostfield_graph_fbm(float3 position, int octaves, float lacunarity, float gain, uint kind) {
        float sum = 0.0f;
        float total = 0.0f;
        float amplitude = 1.0f;
        float3 point = position;
        for (int octave = 0; octave < octaves; ++octave) {
            float value = kind == 0u ? frostfield_graph_value_noise(point) : frostfield_graph_gradient_noise(point);
            sum = fma(value, amplitude, sum);
            total += amplitude;
            amplitude *= gain;
            point = fma(point, float3(lacunarity), float3(19.19f, 7.31f, 13.73f));
        }
        return total > 0.0f && frostfield_graph_finite(sum / total) ? sum / total : 0.0f;
    }

    static inline float2 frostfield_graph_cellular(float3 position) {
        float3 point = frostfield_graph_noise_point(position);
        int3 cell = int3(floor(point));
        float3 local = fract(point);
        float nearest = 8.0f;
        float identifier = 0.0f;
        for (int z = -1; z <= 1; ++z) {
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    int3 offset = int3(x, y, z);
                    int3 neighbor = cell + offset;
                    float3 feature = float3(offset) + float3(
                        frostfield_graph_hash(neighbor, 4u), frostfield_graph_hash(neighbor, 5u),
                        frostfield_graph_hash(neighbor, 6u)
                    );
                    float3 delta = feature - local;
                    float distanceSquared = frostfield_graph_dot3(delta, delta);
                    if (distanceSquared < nearest) {
                        nearest = distanceSquared;
                        identifier = frostfield_graph_hash(neighbor, 7u);
                    }
                }
            }
        }
        return float2(sqrt(nearest), identifier);
    }

    // A 0...1 hash of four words; ShaderNoise.hash reproduces it.
    static inline float frostfield_graph_hash_bits(uint4 bits) {
        return frostfield_graph_hash(as_type<int3>(bits.xyz), bits.w ^ 0x68e31da4u);
    }

    // Interleaved gradient noise (Jimenez 2014) of pixel coordinates.
    static inline float frostfield_graph_ign(float2 pixel) {
        float inner = fract(fma(0.00583715f, pixel.y, 0.06711056f * pixel.x));
        return fract(52.9829189f * inner);
    }

    static inline float frostfield_graph_bits(float a, float b, uint operation) {
        uint x = frostfield_graph_finite(a) ? uint(clamp(a, 0.0f, 16777215.0f)) : 0u;
        uint y = frostfield_graph_finite(b) ? uint(clamp(b, 0.0f, 16777215.0f)) : 0u;
        uint result;
        switch (operation) {
            case 0u: result = x & y; break;
            case 1u: result = x | y; break;
            case 2u: result = x ^ y; break;
            case 3u: result = y < 24u ? x << y : 0u; break;
            default: result = y < 24u ? x >> y : 0u; break;
        }
        return float(result & 0xFFFFFFu);
    }

    static inline float2 frostfield_graph_rotate_uv(float2 uv, float angle, float2 center) {
        float2 d = uv - center;
        float c = cos(angle);
        float s = sin(angle);
        return center + float2(c * d.x - s * d.y, s * d.x + c * d.y);
    }

    static inline float2 frostfield_graph_spherize_uv(float2 uv, float2 strength, float2 center) {
        float2 d = uv - center;
        float d2 = dot(d, d);
        return uv + d * (d2 * d2 * strength);
    }

    static inline float2 frostfield_graph_polar(float2 uv, float2 center) {
        float2 d = uv - center;
        return float2(length(d), fract(atan2(d.y, d.x) * 0.15915494309189533577f));
    }

    static inline float frostfield_graph_checker(float2 p) {
        float sum = floor(p.x) + floor(p.y);
        return sum - 2.0f * floor(sum * 0.5f);
    }

    // |n|^k / sum |n|^k, computed as (|n| / max |n|)^k so the dominant axis stays at
    // exactly 1: high sharpness and short normals cannot underflow every weight.
    static inline float3 frostfield_graph_triplanar_weights(float3 normal, float sharpness) {
        float k = frostfield_graph_finite(sharpness) ? max(sharpness, 0.0f) : 1.0f;
        float3 a = abs(select(float3(0), normal, frostfield_graph_finite(normal)));
        float largest = max(a.x, max(a.y, a.z));
        if (!(largest > 0.0f)) {
            return float3(1.0f / 3.0f);
        }
        float3 r = a / largest;
        float3 w = k > 0.0f ? select(pow(r, float3(k)), float3(0), r <= 0.0f) : float3(1);
        return w / (w.x + w.y + w.z);
    }

    static inline float4 frostfield_graph_triplanar_blend(float3 w, float4 x, float4 y, float4 z) {
        return x * w.x + y * w.y + z * w.z;
    }

    static inline float3x3 frostfield_graph_inverse(float3x3 m) {
        float3 r0 = cross(m[1], m[2]);
        float3 r1 = cross(m[2], m[0]);
        float3 r2 = cross(m[0], m[1]);
        float det = dot(m[0], r0);
        if (!(abs(det) > 0.0f) || !frostfield_graph_finite(det)) {
            return float3x3(float3(0), float3(0), float3(0));
        }
        return transpose(float3x3(r0, r1, r2)) * (1.0f / det);
    }

    static inline float4x4 frostfield_graph_inverse(float4x4 m) {
        float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2], a03 = m[0][3];
        float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2], a13 = m[1][3];
        float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2], a23 = m[2][3];
        float a30 = m[3][0], a31 = m[3][1], a32 = m[3][2], a33 = m[3][3];
        float b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
        float b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
        float b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
        float b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
        float det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
        if (!(abs(det) > 0.0f) || !frostfield_graph_finite(det)) {
            return float4x4(float4(0), float4(0), float4(0), float4(0));
        }
        float inverse = 1.0f / det;
        return float4x4(
            float4(a11 * b11 - a12 * b10 + a13 * b09, a02 * b10 - a01 * b11 - a03 * b09,
                   a31 * b05 - a32 * b04 + a33 * b03, a22 * b04 - a21 * b05 - a23 * b03) * inverse,
            float4(a12 * b08 - a10 * b11 - a13 * b07, a00 * b11 - a02 * b08 + a03 * b07,
                   a32 * b02 - a30 * b05 - a33 * b01, a20 * b05 - a22 * b02 + a23 * b01) * inverse,
            float4(a10 * b10 - a11 * b08 + a13 * b06, a01 * b08 - a00 * b10 - a03 * b06,
                   a30 * b04 - a31 * b02 + a33 * b00, a21 * b02 - a20 * b04 - a23 * b00) * inverse,
            float4(a11 * b07 - a10 * b09 - a12 * b06, a00 * b09 - a01 * b07 + a02 * b06,
                   a31 * b01 - a30 * b03 - a32 * b00, a20 * b03 - a21 * b01 + a22 * b00) * inverse
        );
    }
    """
}
