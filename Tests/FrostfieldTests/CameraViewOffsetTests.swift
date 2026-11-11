import Frostfield
import Metal
import Testing
import simd

struct CameraViewOffsetTests {
    @Test("The upper-left window uses the full image aspect and preserves clip depth")
    func perspectiveWindow() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.25, far: 16)
        let window = try CameraViewOffset(fullWidth: 800, fullHeight: 400, x: 0, y: 0, width: 300, height: 100)
        let full = try camera.projection(aspectRatio: 2)
        let cropped = try camera.projection(viewOffset: window)
        let topLeft = SIMD4<Float>(-4, 2, -2, 1)
        let bottomRight = SIMD4<Float>(-1, 1, -2, 1)
        #expect(window.fullAspectRatio == 2)
        let a = cropped * topLeft
        let b = cropped * bottomRight
        #expect(simd_distance(SIMD2(a.x, a.y) / a.w, SIMD2(-1, 1)) < 0.00001)
        #expect(simd_distance(SIMD2(b.x, b.y) / b.w, SIMD2(1, -1)) < 0.00001)
        #expect(a.z == (full * topLeft).z)
        #expect(b.w == (full * bottomRight).w)
    }

    @Test("A full window leaves a shifted projection unchanged and signed origins permit overscan")
    func overscan() throws {
        var projection = matrix_identity_float4x4
        projection.columns.3 = SIMD4(0.125, -0.375, 0, 1)
        let full = try CameraViewOffset(fullWidth: 640, fullHeight: 320, x: 0, y: 0, width: 640, height: 320)
        #expect(try full.applying(to: projection) == projection)

        let padded = try CameraViewOffset(fullWidth: 640, fullHeight: 320, x: -320, y: -160, width: 1280, height: 640)
        let center = SIMD4<Float>(0, 0, 0.25, 1)
        #expect(try padded.applying(to: projection) * center == SIMD4<Float>(0.0625, -0.1875, 0.25, 1))
    }

    @Test("Invalid dimensions and an overflowing cropped projection report errors")
    func invalidWindow() throws {
        #expect(throws: CameraViewOffset.Failure.invalidWindow) {
            try CameraViewOffset(fullWidth: 0, fullHeight: 16, x: 0, y: 0, width: 8, height: 8)
        }
        #expect(throws: CameraViewOffset.Failure.invalidWindow) {
            try CameraViewOffset(fullWidth: 16, fullHeight: 16, x: 0, y: 0, width: 8, height: -1)
        }
        let window = try CameraViewOffset(fullWidth: 16, fullHeight: 16, x: 0, y: 0, width: 8, height: 8)
        var tooLarge = matrix_identity_float4x4
        tooLarge.columns.0.x = .greatestFiniteMagnitude
        #expect(throws: CameraViewOffset.Failure.unrepresentableProjection) { try window.applying(to: tooLarge) }
    }

    @Test("An off-center tile matches a full Metal frame and its picking ray", arguments: [false, true])
    func rasterTile(orthographicReversed: Bool) throws {
        let full = try GeometryRenderFixture(side: 64)
        let tile = try GeometryRenderFixture(side: 32)
        let depthMode: MeshDepthMode = orthographicReversed ? .reversed : .forward
        let projection: simd_float4x4
        if orthographicReversed {
            projection = simd_float4x4(columns: (
                SIMD4(0.25, 0, 0, 0), SIMD4(0, 0.25, 0, 0), SIMD4(0, 0, 0.125, 0), SIMD4(0, 0, 1, 1)
            ))
        } else {
            projection = try full.camera.projection(aspectRatio: 1)
        }
        let fullCamera = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: projection,
            near: orthographicReversed ? 0 : full.camera.near, far: orthographicReversed ? 8 : full.camera.far
        )
        let offset = try CameraViewOffset(fullWidth: 64, fullHeight: 64, x: 9, y: 23, width: 32, height: 32)
        let cropped = try fullCamera.cropped(to: offset)
        #expect(cropped.near == fullCamera.near && cropped.far == fullCamera.far)
        #expect(cropped.viewMatrix == fullCamera.viewMatrix)

        let mesh = try MeshRenderer(
            device: full.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(0.48),
            material: .unlit, depthMode: depthMode
        )
        let first = SceneNode(
            transform: .init(translation: SIMD3(-0.875, -0.875, -4)), mesh: mesh,
            materialParameters: SIMD4(1, 0, 0, 1)
        )
        let second = SceneNode(
            transform: .init(translation: SIMD3(0.93, 0.71, -4)), mesh: mesh,
            materialParameters: SIMD4(0, 1, 0, 1)
        )
        let outside = SceneNode(transform: .init(translation: SIMD3(3.1, 0, -4)), mesh: mesh)
        let graph = SceneGraph(roots: [first, second, outside], camera: full.camera)
        func render(_ fixture: GeometryRenderFixture, camera: ShaderGraphCamera) throws -> ([SIMD4<Float>], SceneDrawStatistics) {
            let target = try fixture.makeTarget()
            let pass = fixture.pass(color: target.color, depth: target.depth)
            pass.depthAttachment.clearDepth = orthographicReversed ? 0 : 1
            let command = try #require(fixture.queue.makeCommandBuffer())
            let stats = try graph.encode(
                pass: pass, commandBuffer: command, clipFromWorld: camera.projectionMatrix * camera.viewMatrix,
                cameraPosition: .zero, directionToCamera: orthographicReversed ? SIMD3(0, 0, 1) : nil,
                depthMode: depthMode, graphCamera: camera
            )
            try fixture.finish(command)
            return (try fixture.pixels(of: target.color), stats)
        }

        let (fullPixels, fullStats) = try render(full, camera: fullCamera)
        let (tilePixels, tileStats) = try render(tile, camera: cropped)
        #expect(fullStats.drawnMeshCount == 3)
        #expect(tileStats.drawnMeshCount == 2 && tileStats.culledMeshCount == 1)
        #expect(tile.pixel(tilePixels, x: 16, y: 16).x > 0.2)
        let reference = (offset.y..<(offset.y + offset.height)).flatMap { y in
            let start = y * full.side + offset.x
            return fullPixels[start..<(start + offset.width)]
        }
        #expect(tilePixels == reference)
        #expect(try graph.pick(at: SIMD2(0.5, 0.5), camera: cropped, depthMode: depthMode)?.nodeID == first.id)
        #expect(try graph.pick(at: SIMD2(25.0 / 64, 39.0 / 64), camera: fullCamera, depthMode: depthMode)?.nodeID == first.id)
    }
}
