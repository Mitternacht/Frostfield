import Metal
import Testing
import simd
@testable import Frostfield

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct LightformerTests {
    @Test func shapesAndCapturedRadiance() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let rectangle = try Lightformer.node(device: device, color: SIMD3(1, 0, 0), intensity: 4, position: SIMD3(0, 0, 2), scale: SIMD3(repeating: 4))
        let geometry = try #require(rectangle.mesh?.geometry)
        #expect(geometry.vertices.count == 4)
        #expect(geometry.vertices.allSatisfy { abs($0.position.x) == 0.5 && abs($0.position.y) == 0.5 && $0.position.z == 0 })
        #expect(abs(rectangle.transform.matrix.columns.2.z - 4) < 1e-4)

        let ring = try Lightformer.node(device: device, form: .ring, segments: 8)
        let radii = try #require(ring.mesh?.geometry.vertices.map { simd_length(SIMD2($0.position.x, $0.position.y)) })
        #expect(radii.allSatisfy { abs($0 - 0.5) < 1e-5 || abs($0 - 1) < 1e-5 })

        let circle = try Lightformer.node(device: device, form: .circle, segments: 4)
        let circleVertices = try #require(circle.mesh?.geometry.vertices)
        #expect(circleVertices.contains { $0.position == .zero })
        let box = try Lightformer.node(device: device, form: .box)
        let boxCount = try #require(box.mesh).geometry.vertices.count
        #expect(MeshGeometry.cube().vertices.count == boxCount)

        #expect(throws: Lightformer.Failure.color) {
            try Lightformer.node(device: device, color: SIMD3(2, 0, 0))
        }
        #expect(throws: Lightformer.Failure.segments) {
            try Lightformer.node(device: device, form: .circle, segments: 2)
        }

        let gpu = try RasterLightingFixture(width: 8, height: 8)
        let graph = SceneGraph(
            roots: [rectangle],
            camera: try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 100)
        )
        let settings = EnvironmentLighting.Settings(cubeSize: 32, diffuseSize: 8, brdfSize: 32, sampleCount: 16)
        let lighting = try EnvironmentLighting.capture(graph, at: .zero, queue: gpu.queue, size: 32, settings: settings)
        let sample = try gpu.sampleCube(lighting.skyCube, directions: [SIMD3(0, 0, 1)])[0]
        #expect(abs(sample.x - 4) < 0.15 && sample.y < 0.05 && sample.z < 0.05, "\(sample)")
    }
}
