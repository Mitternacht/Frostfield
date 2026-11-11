import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Scene graphs and pixel digests for the render-hook tests. The default camera sits at the
/// origin looking down -Z with a 90 degree field of view, so at depth 4 the view spans
/// 4 world units either way and each node below covers its own pixels.
struct RenderHooksScenes {
    let fixture: PrimitiveRenderFixture
    private var gpu: GeometryRenderFixture { fixture.base }

    /// A drawable node with the label and the render kind the encode should report for it.
    struct Labeled {
        let label: String
        var node: SceneNode
        let kind: SceneRenderOrdering.Kind
    }

    init(side: Int = 64) throws {
        fixture = try PrimitiveRenderFixture(side: side)
    }

    /// The pixel that a world point in front of the camera projects to, clamped to the target.
    /// A point at or behind the camera, which only a wrong transform produces, maps to the
    /// corner so that the caller's expectation fails instead of the test crashing.
    func pixel(of world: SIMD3<Float>) -> (x: Int, y: Int) {
        let depth = -world.z
        guard depth > 0 else { return (0, 0) }
        let side = Float(fixture.side)
        let x = side / 2 + side / 2 * world.x / depth
        let y = side / 2 - side / 2 * world.y / depth
        return (Int(min(max(x, 0), side - 1)), Int(min(max(y, 0), side - 1)))
    }

    /// One node of every drawable kind: a plain, an instanced, a skinned and a batched mesh,
    /// then points, a line and sprites. All sit at depth 4 inside the view and share no pixels.
    func kinds() throws -> [Labeled] {
        let square = try GeometryRenderFixture.square(0.5)
        let plain = SceneNode(
            transform: .init(translation: SIMD3(-2.6, 2.4, -4)), mesh: try gpu.unlit(square),
            materialParameters: SIMD4(1, 0, 0, 1), name: "plain"
        )
        let instances = try InstanceSet(
            device: gpu.device, transforms: [translationMatrix(-0.7), translationMatrix(0.7)],
            tints: [SIMD4(1, 0, 1, 1), SIMD4(0.5, 1, 0.25, 1)]
        )
        let instanced = SceneNode(
            transform: .init(translation: SIMD3(0, 2.4, -4)),
            mesh: try gpu.unlit(try GeometryRenderFixture.square(0.3), supportsInstancing: true),
            instances: instances, name: "instanced"
        )
        let influence = VertexSkinInfluence(joints: SIMD4(repeating: 0), weights: SIMD4(1, 0, 0, 0))
        let skin = try MeshSkin(jointCount: 1, influences: Array(repeating: influence, count: 4))
        let skinned = SceneNode(
            transform: .init(translation: SIMD3(2.6, 2.4, -4)), mesh: try gpu.unlit(square, skin: skin),
            materialParameters: SIMD4(0, 0, 1, 1), jointPalette: try JointPalette(matrices: [matrix_identity_float4x4]),
            name: "skinned"
        )
        let batch = try BatchedMesh(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometries: [try GeometryRenderFixture.square(0.3)],
            capacity: 4, renderState: nil
        )
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(-0.7), tint: SIMD4(0, 1, 1, 1))
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0.7), tint: SIMD4(1, 1, 0, 1))
        let batched = SceneNode(transform: .init(translation: SIMD3(-2.4, 0, -4)), name: "batched", batch: batch)

        var points = SceneNode(
            transform: .init(translation: SIMD3(0.4, 0, -4)), primitive: .points(try PointsRenderer(
                device: gpu.device,
                cloud: try PointCloud(positions: [SIMD3(-0.6, 0, 0), SIMD3(0, 0, 0), SIMD3(0.6, 0, 0)]),
                material: PointsMaterial(size: .pixels(6), color: SIMD4(1, 1, 1, 1))
            ))
        )
        points.name = "points"
        var line = SceneNode(
            transform: .init(translation: SIMD3(2.4, 0, -4)),
            primitive: try fixture.lines(
                try LineGeometry(positions: [SIMD3(-0.8, -0.6, 0), SIMD3(0.8, 0.6, 0)]),
                LineMaterial(width: .pixels(3), cap: .butt, color: SIMD4(1, 0.5, 0, 1))
            )
        )
        line.name = "line"
        var sprites = SceneNode(
            transform: .init(translation: SIMD3(-1.5, -2.4, -4)), primitive: .sprites(try SpriteRenderer(
                device: gpu.device,
                sprites: [Sprite(position: SIMD3(-0.5, 0, 0), size: SIMD2(0.8, 0.8)), Sprite(position: SIMD3(0.5, 0, 0), size: SIMD2(0.8, 0.8))],
                material: SpriteMaterial(sizeAttenuation: true, blending: .opaque)
            ))
        )
        sprites.name = "sprites"
        return [
            Labeled(label: "plain", node: plain, kind: .mesh), Labeled(label: "instanced", node: instanced, kind: .mesh),
            Labeled(label: "skinned", node: skinned, kind: .mesh), Labeled(label: "batched", node: batched, kind: .batch),
            Labeled(label: "points", node: points, kind: .points), Labeled(label: "line", node: line, kind: .lines),
            Labeled(label: "sprites", node: sprites, kind: .sprites),
        ]
    }

    func graph(_ nodes: [SceneNode]) -> SceneGraph {
        fixture.graph(nodes)
    }

    /// Named scenes for the default-path digests: each node kind alone, then all together.
    func digestScenes() throws -> [(name: String, graph: SceneGraph)] {
        let all = try kinds()
        var scenes = all.map { (name: $0.label, graph: graph([$0.node])) }
        scenes.append((name: "all", graph: graph(all.map(\.node))))
        return scenes
    }

    /// FNV-1a over the bit pattern of every channel, so two renders agree only if every
    /// pixel is byte-for-byte equal.
    static func digest(_ pixels: [SIMD4<Float>]) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for pixel in pixels {
            for channel in 0..<4 {
                var bits = pixel[channel].bitPattern
                for _ in 0..<4 {
                    hash = (hash ^ UInt64(bits & 0xff)) &* 0x0000_0100_0000_01b3
                    bits >>= 8
                }
            }
        }
        return hash
    }
}
