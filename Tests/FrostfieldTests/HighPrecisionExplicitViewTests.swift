import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct HighPrecisionExplicitViewTests {
    enum Projection: CaseIterable, Sendable {
        case offAxis, reversedOrthographic

        var depthMode: MeshDepthMode { self == .offAxis ? .forward : .reversed }
        var matrix: simd_float4x4 {
            switch self {
            case .offAxis:
                simd_float4x4(
                    SIMD4(1.2, 0, 0, 0), SIMD4(0, 1.35, 0, 0), SIMD4(0.19, -0.13, -1.0125, -1),
                    SIMD4(0, 0, -0.10125, 0)
                )
            case .reversedOrthographic:
                simd_float4x4(
                    SIMD4(0.5, 0, 0, 0), SIMD4(0, 0.5, 0, 0), SIMD4(0, 0, 0.125, 0),
                    SIMD4(0.1, -0.08, 1, 1)
                )
            }
        }
    }

    private let anchor = SIMD3<Float>(0x1p60, -0x1p60, 0x1p60)
    private let rotation = simd_quatf(angle: 0.21, axis: simd_normalize(SIMD3<Float>(0.24, 1, 0.31)))

    @Test("Explicit off-axis and reversed orthographic draws preserve inherited pivots, primitives, culling and sorting",
          arguments: Projection.allCases)
    func projectedFrame(_ projection: Projection) throws {
        let gpu = try GeometryRenderFixture(side: 96)
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(0.26),
            material: .physicallyBased(try PhysicallyBasedMaterial(shadingModel: .unlit)), depthMode: projection.depthMode
        )
        let green = SIMD4<Float>(0, 1, 0, 1)
        let cyan = SIMD4<Float>(0, 1, 1, 1)
        let red = SIMD4<Float>(1, 0, 0, 0.5)
        let blue = SIMD4<Float>(0, 0, 1, 0.5)
        let points = try PointsRenderer(
            device: gpu.device, cloud: PointCloud(positions: [.zero]), material: .init(size: .pixels(7), color: cyan)
        )
        func sprite(_ color: SIMD4<Float>, z: Float) throws -> SceneNode {
            let renderer = try SpriteRenderer(
                device: gpu.device, sprites: [Sprite(position: .zero, size: SIMD2(20, 20))],
                material: .init(color: color, sizeAttenuation: false)
            )
            return SceneNode(transform: .init(translation: SIMD3(0.15, 0.15, z)), primitive: .sprites(renderer))
        }
        let nodes = [
            SceneNode(transform: .init(translation: SIMD3(-1.1, -0.5, -3.6)), mesh: mesh, materialParameters: green),
            SceneNode(transform: .init(translation: SIMD3(8, 0, -3.6)), mesh: mesh),
            try sprite(red, z: -3.7),
            try sprite(blue, z: -5.1),
            SceneNode(transform: .init(translation: SIMD3(1.1, 0.45, -3.5)), primitive: .points(points)),
            SceneNode(transform: .init(translation: SIMD3(8, 2, -4)), primitive: .points(points)),
        ]
        let ordinary = try render(gpu, projection: projection, origin: .zero, nodes: nodes, precise: false)
        let precise = try render(gpu, projection: projection, origin: anchor, nodes: nodes, precise: true)
        #expect(zip(ordinary.pixels, precise.pixels).allSatisfy { isNearly($0, $1, tolerance: 0.001) })
        #expect(precise.statistics.drawnMeshCount == 1)
        #expect(precise.statistics.culledMeshCount == 1)
        #expect(precise.statistics.drawnPrimitiveCount == 3)
        #expect(precise.statistics.culledPrimitiveCount == 1)
        #expect(precise.pixels.contains(green))
        #expect(precise.pixels.contains(cyan))
        #expect(precise.pixels.contains(SIMD4(0.5, 0, 0.25, 0.75)))
    }

    @Test("Explicit high precision validates the split view and each combined clip column before encoding")
    func validation() throws {
        let gpu = try GeometryRenderFixture()
        let target = try gpu.makeTarget()
        var graph = SceneGraph(roots: [], camera: gpu.camera)
        graph.highPrecision = true
        let camera = try makeCamera(.offAxis, origin: anchor)
        let clip = camera.projectionMatrix * camera.viewMatrix
        let pass = gpu.pass(color: target.color, depth: target.depth)
        let command = try #require(gpu.queue.makeCommandBuffer())
        var mismatched = clip
        mismatched[0].x += 0.25
        #expect(throws: MeshRenderer.Failure.graphCameraInputs) {
            try graph.encode(
                pass: pass, commandBuffer: command, clipFromWorld: mismatched,
                cameraPosition: anchor, graphCamera: camera
            )
        }
        #expect(throws: MeshRenderer.Failure.highPrecisionView) {
            try graph.encode(
                pass: pass, commandBuffer: command, clipFromWorld: clip,
                cameraPosition: .zero, graphCamera: camera
            )
        }
        var projectiveView = matrix_identity_float4x4
        projectiveView[2].w = 0.125
        let invalid = try ShaderGraphCamera(
            viewMatrix: projectiveView, projectionMatrix: camera.projectionMatrix, near: camera.near, far: camera.far
        )
        #expect(throws: MeshRenderer.Failure.highPrecisionView) {
            try graph.encode(
                pass: pass, commandBuffer: command, clipFromWorld: invalid.projectionMatrix * invalid.viewMatrix,
                cameraPosition: .zero, graphCamera: invalid
            )
        }
        #expect(command.status == .notEnqueued)
    }

    private func makeCamera(_ projection: Projection, origin: SIMD3<Float>) throws -> ShaderGraphCamera {
        var view = simd_inverse(SceneTransform(rotation: rotation).matrix)
        let translation = -(view * SIMD4(origin, 0))
        view[3] = SIMD4(translation.x, translation.y, translation.z, 1)
        return try ShaderGraphCamera(
            viewMatrix: view, projectionMatrix: projection.matrix,
            near: projection == .offAxis ? 0.1 : 0, far: 8
        )
    }

    private func render(
        _ gpu: GeometryRenderFixture, projection: Projection, origin: SIMD3<Float>, nodes: [SceneNode], precise: Bool
    ) throws -> (pixels: [SIMD4<Float>], statistics: SceneDrawStatistics) {
        let root = SceneNode(
            transform: .init(
                translation: origin, rotation: simd_quatf(angle: 0.15, axis: SIMD3(0, 0, 1)),
                scale: SIMD3(0.95, 0.8, 1.15)
            ),
            children: nodes, pivot: SIMD3(0.18, -0.13, 0.22)
        )
        var graph = SceneGraph(
            roots: [root], camera: gpu.camera, cameraTransform: .init(translation: SIMD3(500, 0, 0))
        )
        graph.highPrecision = precise
        graph.projectionJitter = SIMD2(0.375, 0.125)
        let camera = try makeCamera(projection, origin: origin)
        let target = try gpu.makeTarget()
        let pass = gpu.pass(color: target.color, depth: target.depth)
        pass.depthAttachment.clearDepth = projection.depthMode == .reversed ? 0 : 1
        let command = try #require(gpu.queue.makeCommandBuffer())
        let statistics = try graph.encode(
            pass: pass, commandBuffer: command, clipFromWorld: camera.projectionMatrix * camera.viewMatrix,
            cameraPosition: origin,
            directionToCamera: projection == .reversedOrthographic ? rotation.act(SIMD3(0, 0, 1)) : nil,
            depthMode: projection.depthMode, graphCamera: precise ? camera : nil
        )
        try gpu.finish(command)
        return (try gpu.pixels(of: target.color), statistics)
    }
}
