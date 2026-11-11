import CoreGraphics
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PixelatedScenePassTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Reduced scene silhouettes repeat nearest integer blocks within a normalized viewport")
    func viewportSilhouette() throws {
        let sentinel = SIMD4<Float>(0, 0.125, 0.25, 1)
        let width = 24, height = 12
        let destination = try gpu.color(width: width, height: height) { _, _ in sentinel }
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let view = try orthographic(depthMode: .forward)
        let white = try gpu.color(width: 1, height: 1) { _, _ in SIMD4(repeating: 1) }
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 1.05, height: 1.4),
            material: .textured(white)
        )
        let scene = SceneGraph(roots: [SceneNode(
            transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh, materialParameters: SIMD4(repeating: 1)
        )], camera: camera)
        let pass = try PixelatedScenePass(
            queue: gpu.queue, settings: .init(pixelSize: 4, normalEdgeStrength: 0, depthEdgeStrength: 0)
        )
        let viewport = CGRect(x: 0.125, y: 0.25, width: 0.75, height: 0.5)
        try gpu.run { buffer in
            let frame = try pass.encode(scene, destination: destination, commandBuffer: buffer, viewport: viewport, camera: view)
            #expect(frame.color.width == 5 && frame.color.height == 2)
            #expect(frame.depth.width == 5 && frame.depth.height == 2)
            #expect(frame.statistics.submittedTriangleCount == 2)
            #expect(frame.camera == view)
        }
        let result = try gpu.read(destination)
        for y in 0..<height {
            for x in 0..<width {
                let inside = (3..<21).contains(x) && (3..<9).contains(y)
                let expected: SIMD4<Float> = inside ? ((7..<19).contains(x) ? SIMD4(repeating: 1) : .zero) : sentinel
                #expect(result[y * width + x] == expected, "output pixel \(x), \(y)")
            }
        }

        let projection = try camera.projection(aspectRatio: 3)
        try gpu.run { buffer in
            let defaultFrame = try pass.encode(scene, destination: destination, commandBuffer: buffer, viewport: viewport)
            #expect(defaultFrame.camera.projectionMatrix == projection)
            let reuse = try pass.encode(scene, destination: destination, commandBuffer: buffer, viewport: viewport)
            #expect(reuse.color === defaultFrame.color && reuse.depth === defaultFrame.depth)
            pass.settings.pixelSize = 1024
            let single = try pass.encode(scene, destination: destination, commandBuffer: buffer, viewport: viewport, camera: view)
            #expect(single.color.width == 1 && single.color.height == 1)
        }
        let single = try gpu.read(destination)
        #expect(single[3 * width + 3] == SIMD4(repeating: 1))
        #expect(single[8 * width + 20] == SIMD4(repeating: 1))
        #expect(single[3 * width + 21] == sentinel)
    }

    @Test("Rendered depth and normal edges can be enabled independently in both depth modes",
          arguments: [MeshDepthMode.forward, .reversed])
    func renderedEdges(depthMode: MeshDepthMode) throws {
        let width = 24, height = 8
        let destination = try gpu.color(width: width, height: height)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let view = try orthographic(depthMode: depthMode)
        let plane = try MeshGeometry.plane(width: 2 / 3, height: 2)
        let angledNormal = try MeshGeometry(vertices: plane.vertices.map {
            MeshVertex(position: $0.position, normal: SIMD3(-1, 0, 0), textureCoordinate: $0.textureCoordinate)
        }, indices: plane.indices)
        let white = try gpu.color(width: 1, height: 1) { _, _ in SIMD4(repeating: 1) }
        let flat = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: plane, material: .textured(white), depthMode: depthMode
        )
        let angled = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: angledNormal, material: .textured(white), depthMode: depthMode
        )
        let color = SIMD4<Float>(1.25, 1, 0.5, 1)
        let scene = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(-2 / 3, 0, -2)), mesh: flat, materialParameters: color),
            SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: angled, materialParameters: color),
            SceneNode(transform: .init(translation: SIMD3(2 / 3, 0, -4)), mesh: angled, materialParameters: color),
        ], camera: camera)
        let pass = try PixelatedScenePass(queue: gpu.queue)
        let strengths: [(Float, Float)] = [(0, 0), (0.25, 0), (0, 0.5), (0.25, 0.5)]
        for (normal, depth) in strengths {
            pass.settings = .init(pixelSize: 4, normalEdgeStrength: normal, depthEdgeStrength: depth, depthMode: depthMode)
            try gpu.run { buffer in
                let frame = try pass.encode(scene, destination: destination, commandBuffer: buffer, camera: view)
                #expect(frame.color.width == 6 && frame.color.height == 2 && frame.depthMode == depthMode)
            }
            let result = try gpu.read(destination)
            for (index, pixel) in result.enumerated() {
                let block = (index % width) / 4
                let gain = block == 1 ? 1 + normal : block == 3 ? 1 - depth : 1
                let expected = SIMD4(SIMD3(color.x, color.y, color.z) * gain, color.w)
                #expect(pixel == expected, "block \(block), strengths \(normal), \(depth)")
            }
        }
    }

    @Test("Scene pixelation rejects a wrong queue and out-of-range normalized viewport")
    func validation() throws {
        let destination = try gpu.color(width: 6, height: 4)
        let scene = SceneGraph(roots: [], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
        let pass = try PixelatedScenePass(queue: gpu.queue)
        let otherQueue = try #require(gpu.device.makeCommandQueue())
        let foreign = try #require(otherQueue.makeCommandBuffer())
        #expect(throws: PostProcessScenePass.Failure.commandBuffer) {
            try pass.encode(scene, destination: destination, commandBuffer: foreign)
        }
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        for viewport in [CGRect(x: 0.9, y: 0.5, width: 0.5, height: 0.5), CGRect(x: 0, y: 0, width: 0.01, height: 0.01)] {
            #expect(throws: PostProcessEffectError.self) {
                try pass.encode(scene, destination: destination, commandBuffer: buffer, viewport: viewport)
            }
        }
    }

    private func orthographic(depthMode: MeshDepthMode) throws -> ShaderGraphCamera {
        let near: Float = 0.1, far: Float = 10
        let range = far - near
        let reversed = depthMode == .reversed
        var projection = matrix_identity_float4x4
        projection.columns.2.z = (reversed ? 1 : -1) / range
        projection.columns.3.z = (reversed ? far : -near) / range
        return try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: projection, near: near, far: far
        )
    }
}
