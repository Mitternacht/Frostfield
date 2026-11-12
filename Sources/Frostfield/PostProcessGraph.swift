import Foundation

/// A typed full-screen image graph in the `.postProcess` stage.
///
/// The graph reads the output pixel's `screenUV` (top-left normalized pixel
/// center) and `texelSize` (one over the output size), declared uniforms, textures
/// and functions, and two frame inputs:
/// - `sceneColor(at:)`: the source's premultiplied linear RGBA, sampled bilinearly
///   from its base level with clamp-to-edge addressing.
/// - `sceneColor(at:mipLevel:)`: the same source with explicit trilinear mip sampling.
/// - `sceneDepth(at:)`: the stored Metal depth in 0...1 at the nearest texel, with
///   clamp-to-edge addressing.
///
/// Surface and vertex inputs such as `worldPosition` are rejected.
///
/// The single output is either:
/// - a vector4 of premultiplied linear RGBA, written as the pixel; or
/// - a vector3 of premultiplied linear RGB, in the same space `sceneColor` returns,
///   which the pass pairs with the source alpha at the pixel center. The RGB is
///   not multiplied by that alpha again, so gains, tints, swizzles, and blurs of
///   `sceneColor` stay consistent. Effects that are not linear in color, such as
///   inversion or thresholds, either assume opaque frames or should unpremultiply
///   explicitly and return a vector4.
///
/// Prepare it with `PostProcessPass` or add it to a `PostProcessChain`.
public struct PostProcessGraph: Hashable, Sendable {
    public let nodes: [MaterialGraph.Node]
    public let uniforms: [ShaderUniform]
    public let textures: [ShaderTextureInput]
    public let functions: [ShaderFunction]
    /// Structs the graph can construct. Field order is the Metal layout.
    public let structs: [ShaderStruct]
    /// Optional node names that appear in generated Metal comments and diagnostics.
    public let labels: [MaterialGraph.NodeID: String]
    /// A vector4 premultiplied RGBA node or a vector3 premultiplied RGB node.
    public let output: MaterialGraph.NodeID
    let program: ShaderProgram

    /// Validates the entire graph, its declarations, and that every node reachable
    /// from the output is available in the post-process stage.
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
            outputs: [(output, .postProcess)]
        ) { _, id, type in
            guard type == .vector4 || type == .vector3 else { throw Failure.outputType(node: id, actual: type) }
        }
        self.nodes = nodes
        self.uniforms = uniforms
        self.textures = textures
        self.functions = functions
        self.structs = structs
        self.labels = labels
        self.output = output
    }

    /// Describes a graph that returns premultiplied linear RGBA. Values from other
    /// builders are rejected.
    ///
    /// ```swift
    /// let invert = try PostProcessGraph { p in
    ///     let color = p.sceneColor(at: p.screenUV)
    ///     return p.vector(color.a - color.rgb, color.a)
    /// }
    /// ```
    public init(building body: (ShaderGraphBuilder) throws -> ShaderValue<SIMD4<Float>>) throws {
        let builder = ShaderGraphBuilder()
        let result = try body(builder)
        try self.init(builder: builder, output: result)
    }

    /// Describes a graph that returns premultiplied linear RGB and keeps the source
    /// alpha. Values from other builders are rejected.
    ///
    /// ```swift
    /// let gain = try PostProcessGraph { p in
    ///     p.sceneColor(at: p.screenUV).rgb * p.uniform("Gain", Float(1))
    /// }
    /// ```
    public init(building body: (ShaderGraphBuilder) throws -> ShaderValue<SIMD3<Float>>) throws {
        let builder = ShaderGraphBuilder()
        let result = try body(builder)
        try self.init(builder: builder, output: result)
    }

    private init<T>(builder: ShaderGraphBuilder, output: ShaderValue<T>) throws {
        guard output.builder === builder else {
            throw MaterialGraph.Failure.invalidDeclaration("A graph output belongs to a different builder.")
        }
        try builder.checkOperands()
        let declarations = builder.declarations
        try self.init(
            nodes: builder.nodes, uniforms: declarations.uniforms, textures: declarations.textures,
            functions: declarations.functions, structs: declarations.structs, labels: declarations.labels, output: output.id
        )
    }

    /// `.vector4` for an RGBA output or `.vector3` for an RGB output that keeps the source alpha.
    public var outputType: MaterialGraph.ValueType { program.types[output] }

    /// Whether the output depends on scene depth, so encoding needs a depth texture.
    public var readsDepth: Bool {
        program.uses({ node in
            switch node {
            case .sampleDepth, .normalFromDepth, .denoise: true
            default: false
            }
        }, from: [output])
    }

    var readsDepthNormal: Bool {
        program.uses({ node in
            switch node {
            case .normalFromDepth, .denoise: true
            default: false
            }
        }, from: [output])
    }

    var readsDenoise: Bool {
        program.uses({ if case .denoise = $0 { true } else { false } }, from: [output])
    }

    var denoiseNormalSlots: [Int] {
        let slots = program.order(for: [output]).compactMap { id -> Int? in
            guard case .denoise(_, _, _, _, let slot, _) = nodes[id] else { return nil }
            return slot
        }
        return Set(slots).sorted()
    }

    /// Whether the generated function calls `frostfield_graph_scene`: the graph
    /// samples the scene, or its vector3 output reads the source alpha.
    var readsScene: Bool {
        outputType == .vector3
            || program.uses({ node in
                switch node {
                case .sampleScene, .sampleSceneLevel: true
                default: false
                }
            }, from: [output])
    }

    /// Declarations every generated graph function depends on. Place this source
    /// after `#include <metal_stdlib>` and `using namespace metal;`.
    public static var supportSource: String { ShaderProgram.supportSource + "\n" + PostDepthGraphSource.math }

    /// Emits the per-pixel function:
    ///
    /// ```metal
    /// static inline float4 name(thread const FrostfieldPostInput& input,
    ///     thread const FrostfieldGraphResources& resources)
    /// ```
    ///
    /// The result is premultiplied linear RGBA. Nonfinite components become zero,
    /// then RGB is clamped to 0...65,504 and alpha to 0...1. A vector3 output takes
    /// its alpha from `frostfield_graph_scene(resources, input.screenUV).a`.
    ///
    /// The renderer defines `FrostfieldGraphResources`, then the hooks the graph uses:
    /// `frostfield_graph_scene` (bilinear, clamp to edge) when it samples the scene or
    /// has a vector3 output, `frostfield_graph_depth` (nearest, clamp to edge, raw
    /// Metal depth) for a raw depth sample, and `frostfield_graph_depth_normal` for
    /// a reconstructed normal, and `frostfield_graph_denoise` for a denoise node.
    /// Supply `frostfield_graph_uniform` for declared
    /// uniforms and `frostfield_graph_sample` for declared textures.
    /// Choose a unique name of up to 128 ASCII letters, digits or single separating
    /// underscores, starting with a letter. Declared functions use it as a prefix.
    public func metalFunction(named name: String) throws -> String {
        guard ShaderProgram.isIdentifier(name) else { throw Failure.invalidFunctionName(name) }
        let result: String
        if outputType == .vector4 {
            result = """
                float4 finite_result = select(float4(0), n\(output), frostfield_graph_finite(n\(output)));
                return float4(clamp(finite_result.rgb, float3(0), float3(65504.0f)), saturate(finite_result.a));
            """
        } else {
            result = """
                float3 finite_result = select(float3(0), n\(output), frostfield_graph_finite(n\(output)));
                float source_alpha = frostfield_graph_scene(resources, input.screenUV).a;
                source_alpha = frostfield_graph_finite(source_alpha) ? saturate(source_alpha) : 0.0f;
                return float4(clamp(finite_result, float3(0), float3(65504.0f)), source_alpha);
            """
        }
        return """
        \(program.functionDefinitions(prefix: name))

        static inline float4 \(name)(
            thread const FrostfieldPostInput& input,
            thread const FrostfieldGraphResources& resources
        ) {
        \(try program.statements(for: [output], prefix: name))
        \(result)
        }
        """
    }

    public enum Failure: LocalizedError, Equatable {
        case outputType(node: MaterialGraph.NodeID, actual: MaterialGraph.ValueType)
        case invalidFunctionName(String)
        case fogDensity

        public var errorDescription: String? {
            switch self {
            case .outputType(let node, let actual):
                "A post-process graph output must be a vector4 RGBA or vector3 RGB, but node \(node) produces \(actual.rawValue)."
            case .invalidFunctionName(let name): "The generated Metal function name \(name) is invalid."
            case .fogDensity: "Fog density must be finite and nonnegative."
            }
        }
    }
}

extension ShaderUniforms {
    /// Starts every uniform the post-process graph declares at its default.
    public convenience init(graph: PostProcessGraph) throws {
        try self.init(graph.uniforms)
    }
}

// MARK: - Presets

extension PostProcessGraph {
    /// Darkens RGB toward the frame corners and keeps alpha.
    ///
    /// Each pixel is multiplied by `1 − Strength · smoothstep(0, 1, r²/2)`, where
    /// `r` is the distance from the frame center measured in half-extents, so edge
    /// midpoints receive half and corners the full `Strength`. The scalar uniform
    /// "Strength" starts at `strength`; values from 0 to 1 are meaningful.
    public static func vignette(strength: Float = 0.5) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let amount = p.uniform("Strength", strength)
            let centered = p.screenUV * 2 - 1
            let falloff = (centered.dot(centered) * 0.5).smoothstep(0, 1)
            return p.sceneColor(at: p.screenUV).rgb * (1 - amount * falloff)
        }
    }

    /// Displaces red outward and blue inward along the direction from the frame
    /// center, using three scene samples.
    ///
    /// The scalar uniform "Amount" starts at `amount` and gives the displacement in
    /// output pixels at the frame edges. Displaced channels are unpremultiplied by
    /// their sample's alpha and premultiplied by the center alpha, so translucent
    /// edges do not fringe darker. Fully transparent samples contribute zero.
    public static func chromaticAberration(amount: Float = 2) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let shift = (p.screenUV * 2 - 1) * p.texelSize * p.uniform("Amount", amount)
            let center = p.sceneColor(at: p.screenUV)
            let red = p.sceneColor(at: p.screenUV + shift)
            let blue = p.sceneColor(at: p.screenUV - shift)
            return p.vector(red.r * center.a / red.a, center.g, blue.b * center.a / blue.a, center.a)
        }
    }

    /// Averages a 3 × 3 grid of premultiplied RGBA samples.
    ///
    /// The scalar uniform "Spacing" starts at `spacing` and sets the distance between
    /// taps in output pixels. One gives an exact 3 × 3 box; larger values widen it
    /// with bilinear taps. Samples clamp to the frame edge.
    public static func boxBlur(spacing: Float = 1) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let step = p.texelSize * p.uniform("Spacing", spacing)
            var sum = p.sceneColor(at: p.screenUV)
            for (x, y) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                sum = sum + p.sceneColor(at: p.screenUV + step * SIMD2<Float>(Float(x), Float(y)))
            }
            return sum / 9
        }
    }

    /// Sharpens RGB with a five-tap unsharp kernel and keeps alpha.
    ///
    /// The result is `center + Amount · (4 · center − north − south − east − west)`
    /// with neighbors one output pixel away. The scalar uniform "Amount" starts at
    /// `amount`. Negative results clamp to zero.
    public static func sharpen(amount: Float = 0.5) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let center = p.sceneColor(at: p.screenUV).rgb
            let offsets = [SIMD2<Float>(0, -1), SIMD2(0, 1), SIMD2(-1, 0), SIMD2(1, 0)]
            let neighbors = offsets.map { p.sceneColor(at: p.screenUV + p.texelSize * $0).rgb }
            let sum = neighbors.dropFirst().reduce(neighbors[0]) { $0 + $1 }
            return center + (center * 4 - sum) * p.uniform("Amount", amount)
        }
    }

    /// Blends toward an opaque fog color by view-axis distance reconstructed from depth.
    ///
    /// The pass converts stored perspective depth to camera distance with the
    /// "CameraNear" and "CameraFar" uniforms, which start at the camera's clip
    /// planes. The fog factor is `smoothstep(Near, Far, distance)`, as in Three.js
    /// `Fog`, with "Near" and "Far" in world units; equal or reversed bounds make a
    /// step at Near. When `respectsMaterialFog` is true, white texels in the
    /// "FogExclusion" mask keep the source color. The result is `mix(scene, (Color, 1), factor)` in premultiplied
    /// RGBA, so fully fogged transparent pixels become opaque fog. Cleared depth
    /// reads as the far plane. Update the camera uniforms when the camera changes.
    public static func depthFog(
        color: SIMD3<Float>, near: Float, far: Float, camera: PerspectiveCamera, depthMode: MeshDepthMode = .forward,
        respectsMaterialFog: Bool = false
    ) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            try depthMode.requireProjectionDepth()
            let distance = perspectiveDistance(p, camera: camera, depthMode: depthMode)
            let factor = fogCoverage(
                distance.smoothstep(p.uniform("Near", near), p.uniform("Far", far)),
                respectsMaterialFog: respectsMaterialFog, p
            )
            let fog = p.vector(p.uniform("Color", color), p.constant(Float(1)))
            return p.sceneColor(at: p.screenUV).mix(fog, by: factor)
        }
    }

    /// Blends toward "Color" by `1 - exp(-(Density * viewDistance)^2)`, keeping source alpha.
    /// Camera planes initialize "CameraNear" and "CameraFar". Update both when the camera changes.
    public static func exponentialDepthFog(
        color: SIMD3<Float>, density: Float, camera: PerspectiveCamera, depthMode: MeshDepthMode = .forward,
        respectsMaterialFog: Bool = false
    ) throws -> PostProcessGraph {
        try exponentialFog(
            color: color, density: density, camera: camera, depthMode: depthMode, respectsMaterialFog: respectsMaterialFog
        ) { _, distance in
            distance
        }
    }

    /// Exponential-squared depth fog scaled by `max(Height - worldY, 0)`, keeping source alpha.
    ///
    /// The graph uses the output aspect ratio to reconstruct world height from perspective
    /// depth. Update "CameraNear", "CameraFar", "CameraTanHalfFOV" and "CameraWorldY" when
    /// the camera changes. "CameraWorldY" is the Y row of its local-to-world matrix.
    public static func exponentialHeightFog(
        color: SIMD3<Float>, density: Float, height: Float, camera: PerspectiveCamera,
        cameraTransform: SceneTransform, depthMode: MeshDepthMode = .forward, respectsMaterialFog: Bool = false
    ) throws -> PostProcessGraph {
        let matrix = cameraTransform.matrix
        let worldYRow = SIMD4(matrix.columns.0.y, matrix.columns.1.y, matrix.columns.2.y, matrix.columns.3.y)
        let tanHalfFOV = tan(camera.verticalFieldOfView * 0.5)
        return try exponentialFog(
            color: color, density: density, camera: camera, depthMode: depthMode, respectsMaterialFog: respectsMaterialFog
        ) { p, distance in
            let fieldOfView = p.uniform("CameraTanHalfFOV", tanHalfFOV)
            let worldY = p.uniform("CameraWorldY", worldYRow)
            let aspect = p.texelSize.y / p.texelSize.x
            let viewX = (p.screenUV.x * 2 - 1) * aspect * fieldOfView * distance
            let viewY = (1 - p.screenUV.y * 2) * fieldOfView * distance
            let viewPosition = p.vector(viewX, viewY, -distance, p.constant(Float(1)))
            let heightBelowFog = (p.uniform("Height", height) - viewPosition.dot(worldY)).max(0)
            return heightBelowFog * distance
        }
    }

    private static func exponentialFog(
        color: SIMD3<Float>, density: Float, camera: PerspectiveCamera, depthMode: MeshDepthMode,
        respectsMaterialFog: Bool = false,
        distance: (ShaderGraphBuilder, ShaderValue<Float>) -> ShaderValue<Float>
    ) throws -> PostProcessGraph {
        try depthMode.requireProjectionDepth()
        guard density.isFinite, density >= 0 else { throw Failure.fogDensity }
        return try PostProcessGraph { p in
            let viewDistance = perspectiveDistance(p, camera: camera, depthMode: depthMode)
            let scaled = p.uniform("Density", density).max(0) * distance(p, viewDistance)
            let factor = fogCoverage((-(scaled * scaled)).exp().oneMinus(), respectsMaterialFog: respectsMaterialFog, p)
            let scene = p.sceneColor(at: p.screenUV)
            let fogColor = p.uniform("Color", color) * scene.a
            return p.vector(scene.rgb.mix(fogColor, by: factor), scene.a)
        }
    }

    /// White texels in the "FogExclusion" mask keep the source color. The default leaves the factor unchanged.
    private static func fogCoverage(
        _ factor: ShaderValue<Float>, respectsMaterialFog: Bool, _ p: ShaderGraphBuilder
    ) -> ShaderValue<Float> {
        guard respectsMaterialFog else { return factor }
        let excluded = p.texture("FogExclusion", content: .data, addressing: .clampToEdge, filtering: .nearest)
            .sample(p.screenUV).r
        return factor * excluded.oneMinus()
    }

    private static func perspectiveDistance(
        _ p: ShaderGraphBuilder, camera: PerspectiveCamera, depthMode: MeshDepthMode
    ) -> ShaderValue<Float> {
        let near = p.uniform("CameraNear", camera.near)
        let far = p.uniform("CameraFar", camera.far)
        let stored = p.sceneDepth(at: p.screenUV)
        let nearWeight = depthMode == .forward ? 1 - stored : stored
        let farWeight = depthMode == .forward ? stored : 1 - stored
        return near / (nearWeight + farWeight * (near / far))
    }
}
