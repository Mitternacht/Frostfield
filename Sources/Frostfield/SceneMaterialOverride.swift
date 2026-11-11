import Foundation
import Metal

/// A prepared-material selection for a scene's indexed triangle draws.
/// It uses the live geometry and native mesh deformation of each draw,
/// including mesh batches and levels of detail. Exclude raw Metal meshes
/// that need their custom vertex program. Use one override per render owner.
public final class SceneMaterialOverride {
    public let material: MeshMaterial
    public let renderState: MeshRenderState?
    public let parameters: SIMD4<Float>
    /// Nodes that keep their authored material. Exclusions do not include descendants.
    public let excludedNodes: Set<UUID>

    /// `parameters` replaces each selected node's material parameters while
    /// instance tints and vertex colors keep their ordinary meaning.
    public init(
        material: MeshMaterial, renderState: MeshRenderState? = nil,
        parameters: SIMD4<Float> = SIMD4(repeating: 1), excluding nodes: Set<UUID> = []
    ) throws {
        switch material {
        case .rawMetal:
            throw Failure.vertexStage
        case .physicallyBased(let material) where material.surfaceGraph?.hasVertexOutputs == true:
            throw Failure.vertexStage
        default:
            break
        }
        guard [parameters.x, parameters.y, parameters.z, parameters.w].allSatisfy(\.isFinite) else {
            throw Failure.parameters
        }
        self.material = material
        self.renderState = renderState
        self.parameters = parameters
        excludedNodes = nodes
    }

    /// Prepares a standard color pass for all current mesh renderers, including
    /// every LOD. A new renderer or target layout prepares on first use.
    public func prepare(
        in graph: SceneGraph, colorPixelFormat: MTLPixelFormat,
        depthMode: MeshDepthMode = .forward, sampleCount: Int = 1
    ) throws {
        guard colorPixelFormat != .invalid else { throw MeshRenderer.Failure.attachments }
        guard sampleCount > 0 else { throw MeshRenderer.Failure.sampleCount }
        let layout = Layout(formats: [colorPixelFormat], depth: depthMode, samples: sampleCount, targets: nil)
        var pending = graph.roots
        removeRetiredSources()
        while let node = pending.popLast() {
            pending.append(contentsOf: node.children)
            guard !excludedNodes.contains(node.id) else { continue }
            if let mesh = node.mesh ?? node.batch?.renderer { _ = try renderer(for: mesh, layout: layout) }
            for level in node.levelsOfDetail?.levels ?? [] { _ = try renderer(for: level.mesh, layout: layout) }
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case vertexStage
        case parameters

        public var errorDescription: String? {
            switch self {
            case .vertexStage: "Use a scene material override without a custom vertex program or graph vertex output."
            case .parameters: "Scene material override parameters must be finite."
            }
        }
    }

    struct Layout: Hashable {
        let formats: [MTLPixelFormat]
        let depth: MeshDepthMode
        let samples: Int
        let targets: SurfaceRenderTargets?
    }

    private final class Source {
        weak var mesh: MeshRenderer?
        var variants: [Layout: Variant] = [:]

        init(mesh: MeshRenderer) { self.mesh = mesh }
    }

    private struct Variant {
        let renderer: MeshRenderer
        var revision: UInt64?
    }

    private var prepared: [ObjectIdentifier: Source] = [:]

    func removeRetiredSources() {
        prepared = prepared.filter { $0.value.mesh != nil }
    }

    func renderer(for source: MeshRenderer, layout: Layout) throws -> MeshRenderer {
        let key = ObjectIdentifier(source)
        let entry = prepared[key].flatMap { $0.mesh === source ? $0 : nil } ?? Source(mesh: source)
        if var variant = entry.variants[layout] {
            if variant.revision != source.geometryRevision {
                try variant.renderer.validateMaterialCoordinates(source.geometry)
                variant.revision = source.geometryRevision
                entry.variants[layout] = variant
            }
            return variant.renderer
        }
        let renderer = try source.makeMaterialOverride(material, state: renderState, layout: layout)
        entry.variants[layout] = Variant(renderer: renderer, revision: source.geometryRevision)
        prepared[key] = entry
        return renderer
    }
}
