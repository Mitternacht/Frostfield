import Foundation
import Frostfield
import Metal
import Testing
import simd

/// GPU time for large point, line and sprite batches at 1920 × 1080. The bounds
/// are loose regression guards; the printed medians are the measurements.
struct PrimitivePerformanceTests {
    private struct Bench {
        let device: any MTLDevice
        let queue: any MTLCommandQueue
        let color: any MTLTexture
        let depth: any MTLTexture
        let camera: PerspectiveCamera

        init() throws {
            device = try #require(MTLCreateSystemDefaultDevice())
            queue = try #require(device.makeCommandQueue())
            let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float, width: 1920, height: 1080, mipmapped: false
            )
            colorDescriptor.storageMode = .private
            colorDescriptor.usage = .renderTarget
            let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .depth32Float, width: 1920, height: 1080, mipmapped: false
            )
            depthDescriptor.storageMode = .private
            depthDescriptor.usage = .renderTarget
            color = try #require(device.makeTexture(descriptor: colorDescriptor))
            depth = try #require(device.makeTexture(descriptor: depthDescriptor))
            camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 100)
        }

        /// The median GPU time in milliseconds over seven frames after one warm-up.
        func milliseconds(_ node: SceneNode) throws -> Double {
            let graph = SceneGraph(roots: [node], camera: camera)
            var samples: [Double] = []
            for frame in 0..<8 {
                let pass = MTLRenderPassDescriptor()
                pass.colorAttachments[0].texture = color
                pass.colorAttachments[0].loadAction = .clear
                pass.colorAttachments[0].storeAction = .store
                pass.depthAttachment.texture = depth
                pass.depthAttachment.loadAction = .clear
                pass.depthAttachment.storeAction = .dontCare
                pass.depthAttachment.clearDepth = 1
                let buffer = try #require(queue.makeCommandBuffer())
                try graph.encode(pass: pass, commandBuffer: buffer)
                buffer.commit()
                buffer.waitUntilCompleted()
                try #require(buffer.status == .completed)
                if frame > 0 { samples.append((buffer.gpuEndTime - buffer.gpuStartTime) * 1000) }
            }
            return samples.sorted()[samples.count / 2]
        }
    }

    /// Deterministic positions filling the view between depths 5 and 25.
    private func positions(_ count: Int) -> [SIMD3<Float>] {
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(state >> 40) / Float(1 << 24)
        }
        return (0..<count).map { _ in
            let depth = 5 + 20 * next()
            return SIMD3((next() * 2 - 1) * depth * 0.9, (next() * 2 - 1) * depth * 0.5, -depth)
        }
    }

    @Test
    func `Large point, line and sprite batches render within a frame budget`() throws {
        let bench = try Bench()
        let device = bench.device
        let cloud = try PointCloud(positions: positions(1_000_000))
        let squares = try bench.milliseconds(SceneNode(primitive: .points(try PointsRenderer(
            device: device, cloud: cloud, material: PointsMaterial(size: .pixels(2))
        ))))
        let discs = try bench.milliseconds(SceneNode(primitive: .points(try PointsRenderer(
            device: device, cloud: cloud, material: PointsMaterial(size: .world(0.05), shape: .circle, blending: .alpha)
        ))))
        // A random walk with short steps, like a track or a sampled curve, and a
        // stress case of random points joined across the whole view.
        var walk: [SIMD3<Float>] = [SIMD3(0, 0, -10)]
        let steps = positions(100_000)
        for step in steps {
            let next = walk[walk.count - 1] + simd_normalize(step + SIMD3(0, 0, 15)) * 0.03
            walk.append(simd_clamp(next, SIMD3(-8, -4, -14), SIMD3(8, 4, -6)))
        }
        let track = try LineGeometry(positions: walk)
        let scribble = try LineGeometry(positions: positions(100_001))
        let dashed = LineMaterial(width: .pixels(3), join: .miter(), cap: .butt, dash: try LineMaterial.Dash(length: 0.2, gap: 0.1))
        let wide = try bench.milliseconds(SceneNode(primitive: .lines(try LineRenderer(device: device, geometry: track, material: dashed))))
        let stress = try bench.milliseconds(SceneNode(primitive: .lines(try LineRenderer(device: device, geometry: scribble, material: dashed))))
        let hairlines = try bench.milliseconds(SceneNode(primitive: .lines(try LineRenderer(
            device: device, geometry: scribble, material: LineMaterial(width: .hairline, blending: .opaque)
        ))))
        // Average projected segment length in pixels: focal scale × height / 2 × length / depth.
        let focal = 1 / tan(Float.pi / 6) * 540
        func averagePixels(_ line: LineGeometry) -> Float {
            line.segments.reduce(Float(0)) { sum, segment in
                let a = line.positions[segment.start]
                let b = line.positions[segment.end]
                let pa = SIMD2(a.x, a.y) / -a.z * focal
                let pb = SIMD2(b.x, b.y) / -b.z * focal
                return sum + simd_distance(pa, pb)
            } / Float(line.segments.count)
        }
        let sprites = try bench.milliseconds(SceneNode(primitive: .sprites(try SpriteRenderer(
            device: device, sprites: positions(100_000).map { Sprite(position: $0, size: SIMD2(0.2, 0.1), rotation: $0.x) }
        ))))
        print(String(
            format: "Primitive GPU ms at 1920x1080 — 1M 2px squares %.2f, 1M AA discs %.2f, 100k-segment 3px dashed miter track %.2f (%.1f px per segment), same over a 100k-segment scribble %.2f (%.0f px per segment), scribble hairlines %.2f, 100k sprites %.2f",
            squares, discs, wide, averagePixels(track), stress, averagePixels(scribble), hairlines, sprites
        ))
        for value in [squares, discs, wide, stress, hairlines, sprites] { #expect(value < 100) }
    }
}
