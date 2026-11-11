import Foundation
import Metal

/// Alternate raster materials for the scene's point, line and sprite draws.
/// Each type uses its own material and keeps the source geometry and draw range.
/// Use one override per render owner.
public final class ScenePrimitiveMaterialOverride {
    public let points: PointsMaterial?
    public let lines: LineMaterial?
    public let sprites: SpriteMaterial?
    /// Parameters applied only to draws using an override.
    public let parameters: SIMD4<Float>
    /// Nodes that keep their authored primitive materials.
    public let excludedNodes: Set<UUID>

    /// Omit a material to keep the authored materials of that primitive type.
    public init(
        points: PointsMaterial? = nil, lines: LineMaterial? = nil, sprites: SpriteMaterial? = nil,
        parameters: SIMD4<Float> = SIMD4(repeating: 1), excluding nodes: Set<UUID> = []
    ) throws {
        guard [parameters.x, parameters.y, parameters.z, parameters.w].allSatisfy(\.isFinite) else {
            throw SceneMaterialOverride.Failure.parameters
        }
        self.points = points
        self.lines = lines
        self.sprites = sprites
        self.parameters = parameters
        excludedNodes = nodes
    }

    /// Prepares alternate material pipelines for the graph's current primitives.
    /// Additional source renderers or color formats prepare when used.
    public func prepare(
        in graph: SceneGraph, colorPixelFormat: MTLPixelFormat, sampleCount: Int = 1
    ) throws {
        guard colorPixelFormat != .invalid else { throw MeshRenderer.Failure.attachments }
        guard sampleCount > 0 else { throw MeshRenderer.Failure.sampleCount }
        removeRetiredSources()
        try graph.forEachNode { node, _ in
            guard let source = node.primitive else { return }
            _ = try renderer(for: source, node: node.id, colorPixelFormat: colorPixelFormat, sampleCount: sampleCount)
        }
    }

    private struct Layout: Hashable {
        let format: MTLPixelFormat
        let samples: Int
    }

    private final class Source {
        weak var renderer: AnyObject?
        var variants: [Layout: ScenePrimitive] = [:]

        init(renderer: AnyObject) { self.renderer = renderer }
    }

    private var prepared: [ObjectIdentifier: Source] = [:]

    func removeRetiredSources() {
        prepared = prepared.filter { $0.value.renderer != nil }
    }

    func renderer(
        for source: ScenePrimitive, node: UUID, colorPixelFormat: MTLPixelFormat, sampleCount: Int
    ) throws -> ScenePrimitive? {
        guard !excludedNodes.contains(node) else { return nil }
        let identity = source.rendererIdentity
        let key = ObjectIdentifier(identity)
        let layout = Layout(format: colorPixelFormat, samples: sampleCount)
        let entry = prepared[key].flatMap { $0.renderer === identity ? $0 : nil } ?? Source(renderer: identity)
        if let cached = entry.variants[layout] { return cached }
        let variant: ScenePrimitive
        switch source {
        case .points(let renderer):
            guard let points else { return nil }
            variant = .points(try renderer.replacingMaterial(points, format: colorPixelFormat, samples: sampleCount))
        case .lines(let renderer):
            guard let lines else { return nil }
            variant = .lines(try renderer.replacingMaterial(lines, format: colorPixelFormat, samples: sampleCount))
        case .sprites(let renderer):
            guard let sprites else { return nil }
            variant = .sprites(try renderer.replacingMaterial(sprites, format: colorPixelFormat, samples: sampleCount))
        }
        entry.variants[layout] = variant
        prepared[key] = entry
        return variant
    }
}
