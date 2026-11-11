import Foundation

/// A reusable typed graph that overrides selected PBR channels before lighting
/// and can deform a mesh's vertices. The same validated graph compiles into the
/// raster PBR pipeline and the path tracer.
///
/// Build one from an array of nodes or with `SurfaceShaderGraph(building:)`.
/// Declared uniforms are edited at runtime through `ShaderUniforms` without
/// recompiling. Declared textures are bound by name when preparing a material.
public struct SurfaceShaderGraph: Hashable, Sendable {
    public enum Channel: String, Hashable, Sendable {
        case baseColor
        case metallic
        case roughness
        case emission
        case transmissionMask
        /// A tangent-space normal. The renderer normalizes it and builds the frame from tangents or UV derivatives.
        case normal
        /// Ambient occlusion in 0...1. Raster image-based and ambient lighting use it; path tracing derives occlusion from geometry.
        case occlusion
        /// An object-space position offset evaluated per vertex.
        case positionOffset
        /// Multiplies the sampled material alpha before coverage is decided.
        case opacity
        /// Rejects alpha at or below the graph cutoff, replacing any material cutoff.
        case alphaTest
        /// Rejects the fragment or ray hit when the scalar is zero.
        case mask
        /// Straight RGB transmission and alpha coverage for a colored directional shadow.
        case castShadowColor
        /// A scalar predicate for every raster shadow pass. Takes precedence over `mask` there.
        case castShadowMask

        var type: MaterialGraph.ValueType {
            switch self {
            case .baseColor, .emission, .normal, .positionOffset: .vector3
            case .castShadowColor: .vector4
            case .metallic, .roughness, .transmissionMask, .occlusion, .opacity, .alphaTest, .mask, .castShadowMask: .scalar
            }
        }

        var stage: MaterialGraph.Stage { self == .positionOffset ? .vertex : .surface }
        var isCoverage: Bool { self == .opacity || self == .alphaTest || self == .mask }
        var isCastShadow: Bool { self == .castShadowColor || self == .castShadowMask }
    }

    /// Unset outputs leave the sampled material channel as supplied by the renderer.
    public struct Outputs: Hashable, Sendable {
        public let baseColor: MaterialGraph.NodeID?
        public let metallic: MaterialGraph.NodeID?
        public let roughness: MaterialGraph.NodeID?
        public let emission: MaterialGraph.NodeID?
        /// A scalar in 0...1 that multiplies the material's configured transmission factor.
        public let transmissionMask: MaterialGraph.NodeID?
        /// A vector3 tangent-space normal, such as a decoded normal map or a procedural perturbation.
        public let normal: MaterialGraph.NodeID?
        public let occlusion: MaterialGraph.NodeID?
        /// A vector3 object-space offset added to each vertex position.
        public let positionOffset: MaterialGraph.NodeID?
        /// A scalar multiplier clamped to zero through one.
        public let opacity: MaterialGraph.NodeID?
        /// A cutoff clamped to zero through one. Alpha equal to the cutoff is discarded.
        public let alphaTest: MaterialGraph.NodeID?
        /// A scalar predicate. Zero discards and nonzero preserves coverage.
        public let mask: MaterialGraph.NodeID?
        /// A float4 with straight RGB transmission and alpha coverage in 0...1,
        /// evaluated only when a directional shadow map has `transmissionLayers`.
        /// The material's alpha mode, base alpha, `opacity`, `alphaTest` and the
        /// shadow mask decide coverage exactly as for stock glass; this alpha then
        /// scales that coverage. Camera, viewport, soft-particle and directional
        /// shadow inputs are rejected when the graph is built or decoded.
        public let castShadowColor: MaterialGraph.NodeID?
        /// A scalar predicate where zero discards the fragment from every raster
        /// shadow pass (directional, colored directional, spot and point). It
        /// replaces `mask` in those passes and leaves `mask` alone in color
        /// passes and the path tracer. Requires `castShadowColor`.
        public let castShadowMask: MaterialGraph.NodeID?

        public init(
            baseColor: MaterialGraph.NodeID? = nil, metallic: MaterialGraph.NodeID? = nil,
            roughness: MaterialGraph.NodeID? = nil, emission: MaterialGraph.NodeID? = nil,
            transmissionMask: MaterialGraph.NodeID? = nil, normal: MaterialGraph.NodeID? = nil,
            occlusion: MaterialGraph.NodeID? = nil, positionOffset: MaterialGraph.NodeID? = nil,
            opacity: MaterialGraph.NodeID? = nil, alphaTest: MaterialGraph.NodeID? = nil, mask: MaterialGraph.NodeID? = nil,
            castShadowColor: MaterialGraph.NodeID? = nil, castShadowMask: MaterialGraph.NodeID? = nil
        ) {
            self.baseColor = baseColor
            self.metallic = metallic
            self.roughness = roughness
            self.emission = emission
            self.transmissionMask = transmissionMask
            self.normal = normal
            self.occlusion = occlusion
            self.positionOffset = positionOffset
            self.opacity = opacity
            self.alphaTest = alphaTest
            self.mask = mask
            self.castShadowColor = castShadowColor
            self.castShadowMask = castShadowMask
        }

        var connected: [(Channel, MaterialGraph.NodeID)] {
            [
                (Channel.baseColor, baseColor), (.metallic, metallic), (.roughness, roughness),
                (.emission, emission), (.transmissionMask, transmissionMask), (.normal, normal),
                (.occlusion, occlusion), (.positionOffset, positionOffset),
                (.opacity, opacity), (.alphaTest, alphaTest), (.mask, mask),
                (.castShadowColor, castShadowColor), (.castShadowMask, castShadowMask),
            ].compactMap { channel, id in id.map { (channel, $0) } }
        }

        var surfaceChannels: [(Channel, MaterialGraph.NodeID)] {
            connected.filter { $0.0.stage == .surface && !$0.0.isCastShadow }
        }

        /// The coverage outputs every raster shadow pass evaluates: the shadow
        /// mask, when authored, stands in for `mask`.
        var shadowCoverageChannels: [(Channel, MaterialGraph.NodeID)] {
            let channels: [(Channel, MaterialGraph.NodeID?)] = [
                (.opacity, opacity), (.alphaTest, alphaTest), (.mask, castShadowMask ?? mask),
            ]
            return channels.compactMap { channel, id in id.map { (channel, $0) } }
        }
    }

    public let nodes: [MaterialGraph.Node]
    public let uniforms: [ShaderUniform]
    public let textures: [ShaderTextureInput]
    public let functions: [ShaderFunction]
    /// Structs the graph can construct. Field order is the Metal layout.
    public let structs: [ShaderStruct]
    /// Optional node names that appear in generated Metal comments and diagnostics.
    public let labels: [MaterialGraph.NodeID: String]
    public let outputs: Outputs
    let program: ShaderProgram
    let bumpUsesSecondaryTextureCoordinates: Bool

    /// Validates the entire graph and prepares shader code for its connected outputs.
    public init(
        nodes: [MaterialGraph.Node],
        uniforms: [ShaderUniform] = [],
        textures: [ShaderTextureInput] = [],
        functions: [ShaderFunction] = [],
        structs: [ShaderStruct] = [],
        labels: [MaterialGraph.NodeID: String] = [:],
        outputs: Outputs
    ) throws {
        let connected = outputs.connected
        guard !connected.isEmpty else { throw Failure.noOutputs }
        guard outputs.castShadowMask == nil || outputs.castShadowColor != nil else { throw Failure.shadowMaskWithoutColor }
        program = try ShaderProgram(
            nodes: nodes,
            declarations: ShaderDeclarations(
                uniforms: uniforms, textures: textures, functions: functions, structs: structs, labels: labels
            ),
            outputs: connected.map { ($0.1, $0.0.stage) },
            allowsSurfaceViewportSamples: true
        ) { index, id, type in
            let channel = connected[index].0
            guard type == channel.type else { throw Failure.outputType(channel: channel, node: id, actual: type) }
        }
        let coverage = connected.filter { $0.0.isCoverage }.map(\.1) + [outputs.castShadowMask].compactMap { $0 }
        guard !program.uses({ $0 == .input(.directionalShadowVisibility) }, from: coverage) else {
            throw Failure.directionalShadowCoverage
        }
        // A colored shadow pass runs in light space with no camera or viewport snapshot,
        // so reject those inputs now instead of when a draw reaches the shadow map.
        for (channel, root) in [(Channel.castShadowColor, outputs.castShadowColor), (.castShadowMask, outputs.castShadowMask)] {
            guard let root else { continue }
            if let node = Self.shadowUnsupportedNode(in: program, nodes: nodes, from: [root], rejectsShadowVisibility: true) {
                throw Failure.castShadowInput(channel: channel, node: node)
            }
        }
        bumpUsesSecondaryTextureCoordinates = program.uses({ node in
            if case .bumpMap(_, .secondary, _, _) = node { return true }
            return false
        }, from: connected.map { $0.1 })
        self.nodes = nodes
        self.uniforms = uniforms
        self.textures = textures
        self.functions = functions
        self.structs = structs
        self.labels = labels
        self.outputs = outputs
    }

    /// Whether the graph overrides any per-pixel channel.
    public var hasSurfaceOutputs: Bool { !outputs.surfaceChannels.isEmpty }
    /// Whether the graph deforms vertices.
    public var hasVertexOutputs: Bool { outputs.positionOffset != nil }
    var hasCoverageOutputs: Bool { outputs.opacity != nil || hasCutoutOutputs }
    var hasCutoutOutputs: Bool { outputs.alphaTest != nil || outputs.mask != nil || discardsFragment }
    /// Coverage in the raster shadow passes. Differs from the color pass only when `castShadowMask` replaces `mask`.
    var hasShadowCoverageOutputs: Bool { outputs.opacity != nil || hasShadowCutoutOutputs }
    var hasShadowCutoutOutputs: Bool {
        outputs.alphaTest != nil || (outputs.castShadowMask ?? outputs.mask) != nil || discardsFragment
    }

    /// Outputs whose evaluation can discard. Coverage passes re-run these roots, not the
    /// discard node alone, so a discard inside a branch stays inside that branch.
    private var discardsFragment: Bool { !discardRoots.isEmpty }

    private var discardRoots: [MaterialGraph.NodeID] {
        outputs.connected.map(\.1).filter { root in
            program.order(for: [root]).contains { if case .discard = program.nodes[$0] { true } else { false } }
        }
    }
    var hasCastShadowOutput: Bool { outputs.castShadowColor != nil }

    /// Declarations every generated graph function depends on. Place this source
    /// after `#include <metal_stdlib>` and `using namespace metal;`.
    public static var supportSource: String { ShaderProgram.supportSource }

    /// Emits the per-pixel surface function:
    ///
    /// ```metal
    /// static inline void name(thread const FrostfieldSurfaceInput& input,
    ///     thread const FrostfieldGraphResources& resources, thread FrostfieldSurfaceOutput& surface)
    /// ```
    ///
    /// Initialize `surface` with the authored material. Connected outputs overwrite
    /// their channel with finite, clamped values; nonfinite results keep the authored value.
    /// Initialize coverage fields `opacity`, `alphaTest` and `mask` to one, minus one and true.
    /// The renderer defines `FrostfieldGraphResources`, plus `frostfield_graph_uniform`
    /// when the graph declares uniforms and `frostfield_graph_sample` when it declares
    /// textures. A bump map node calls the renderer's `frostfield_graph_bump` hook.
    /// Choose a unique name of up to 128 ASCII letters, digits or single
    /// separating underscores, starting with a letter. Declared functions use it as a prefix.
    public func metalFunction(named name: String) throws -> String {
        try metalFunction(named: name, outputs: outputs.surfaceChannels)
    }

    func metalCoverageFunction(named name: String) throws -> String {
        try metalFunction(named: name, outputs: outputs.connected.filter { $0.0.isCoverage }, extraRoots: discardRoots)
    }

    /// The coverage function of the raster shadow passes. Identical to `metalCoverageFunction`
    /// unless the graph authors `castShadowMask`, which replaces `mask` here.
    func metalShadowCoverageFunction(named name: String) throws -> String {
        try metalFunction(named: name, outputs: outputs.shadowCoverageChannels, extraRoots: discardRoots)
    }

    func metalCastShadowFunction(named name: String) throws -> String? {
        guard let color = outputs.castShadowColor else { return nil }
        return try metalFunction(named: name, outputs: outputs.shadowCoverageChannels, castShadow: color, extraRoots: discardRoots)
    }

    private func metalFunction(
        named name: String, outputs surfaceOutputs: [(Channel, MaterialGraph.NodeID)],
        castShadow: MaterialGraph.NodeID? = nil, extraRoots: [MaterialGraph.NodeID] = []
    ) throws -> String {
        try Self.checkName(name)
        let assignments = surfaceOutputs.map { channel, id -> String in
            switch channel {
            case .baseColor, .emission:
                let field = channel == .emission ? "emission" : "baseColor"
                let limit = channel == .emission ? "65504.0f" : "1.0f"
                return """
                    float3 finite_\(field) = select(float3(0), surface.\(field), frostfield_graph_finite(surface.\(field)));
                    surface.\(field) = clamp(select(finite_\(field), n\(id), frostfield_graph_finite(n\(id))), float3(0), float3(\(limit)));
                """
            case .normal:
                return """
                    if (all(frostfield_graph_finite(n\(id))) && dot(n\(id), n\(id)) > 1e-12f) {
                        surface.normal = frostfield_graph_unit(n\(id));
                        surface.hasNormal = true;
                    }
                """
            case .opacity, .alphaTest:
                return "    if (frostfield_graph_finite(n\(id))) { surface.\(channel.rawValue) = saturate(n\(id)); }"
            case .mask:
                return "    if (frostfield_graph_finite(n\(id))) { surface.mask = n\(id) != 0.0f; }"
            default:
                let field = channel.rawValue
                return """
                    float finite_\(field) = frostfield_graph_finite(surface.\(field)) ? surface.\(field) : 0.0f;
                    surface.\(field) = saturate(frostfield_graph_finite(n\(id)) ? n\(id) : finite_\(field));
                """
            }
        }.joined(separator: "\n")

        let roots = surfaceOutputs.map(\.1) + extraRoots + (castShadow.map { [$0] } ?? [])
        let result = castShadow.map { id in
            "\n    return all(frostfield_graph_finite(n\(id))) ? saturate(n\(id)) : float4(0);"
        } ?? ""
        let returnType = castShadow == nil ? "void" : "float4"
        return """
        \(program.functionDefinitions(prefix: name))

        static inline \(returnType) \(name)(
            thread const FrostfieldSurfaceInput& input,
            thread const FrostfieldGraphResources& resources,
            thread FrostfieldSurfaceOutput& surface
        ) {
        \(try program.statements(for: roots, prefix: name))
        \(assignments)\(result)
        }
        """
    }

    /// Emits the per-vertex deformation, or nil when the graph has no position offset:
    ///
    /// ```metal
    /// static inline float3 name(thread const FrostfieldVertexInput& input,
    ///     thread const FrostfieldGraphResources& resources)
    /// ```
    ///
    /// The result is a finite object-space offset bounded to ±10⁶ units per axis.
    /// Emit it in a separate library from `metalFunction(named:)` or under a different name.
    public func metalVertexFunction(named name: String) throws -> String? {
        try Self.checkName(name)
        guard let offset = outputs.positionOffset else { return nil }
        return """
        \(program.functionDefinitions(prefix: name))

        static inline float3 \(name)(
            thread const FrostfieldVertexInput& input,
            thread const FrostfieldGraphResources& resources
        ) {
        \(try program.statements(for: [offset], prefix: name))
            return clamp(select(float3(0), n\(offset), frostfield_graph_finite(n\(offset))), float3(-1e6f), float3(1e6f));
        }
        """
    }

    private static func checkName(_ name: String) throws {
        guard ShaderProgram.isIdentifier(name) else { throw Failure.invalidFunctionName(name) }
    }

    public enum Failure: LocalizedError, Equatable {
        case noOutputs
        case outputType(channel: Channel, node: MaterialGraph.NodeID, actual: MaterialGraph.ValueType)
        case invalidFunctionName(String)
        case shadowMaskWithoutColor
        case castShadowInput(channel: Channel, node: MaterialGraph.NodeID)
        case directionalShadowCoverage

        public var errorDescription: String? {
            switch self {
            case .noOutputs: "A surface shader graph needs at least one connected output."
            case .shadowMaskWithoutColor: "A colored shadow mask needs a castShadowColor output."
            case .castShadowInput(let channel, let node):
                "Surface channel \(channel.rawValue) reads a camera, viewport, soft-particle or directional shadow input at node \(node), "
                    + "which a light-space shadow pass cannot supply."
            case .outputType(let channel, let node, let actual):
                "Surface channel \(channel.rawValue) needs a \(channel.type.rawValue), but node \(node) produces \(actual.rawValue)."
            case .invalidFunctionName(let name): "The generated Metal function name \(name) is invalid."
            case .directionalShadowCoverage: "Surface coverage cannot depend on its directional shadow visibility."
            }
        }
    }
}
