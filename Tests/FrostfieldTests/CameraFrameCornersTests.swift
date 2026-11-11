import Frostfield
import Metal
import Testing
import simd

struct CameraFrameCornersTests {
    private let eye = SIMD3<Float>(2, -1, 3)
    private let bottomLeft = SIMD3<Float>(2.67420016, -1.24398556, -1.20308520)
    private let bottomRight = SIMD3<Float>(5.87755325, 0.49247383, 0.44713834)
    private let topLeft = SIMD3<Float>(1.69220392, 0.49674122, -1.12856238)
    private let back = SIMD3<Float>(-0.34289781, -0.23240453, 0.91016989)

    @Test("The tilted rectangle matches r187 framing after converting OpenGL depth to Metal")
    func reference() throws {
        let frame = try framed(far: 20, depthMode: .forward)
        let view = simd_float4x4(columns: (
            SIMD4(0.80083827, -0.49099812, -0.34289781, 0),
            SIMD4(0.43411485, 0.87036339, -0.23240453, 0),
            SIMD4(0.41255589, 0.03726141, 0.91016989, 0),
            SIMD4(-2.40522935, 1.74057540, -2.27711858, 1)
        ))
        let projection = simd_float4x4(columns: (
            SIMD4(2, 0, 0, 0), SIMD4(0, 4, 0, 0),
            SIMD4(0.35, 0.3, -1.01265823, -1), SIMD4(0, 0, -0.25316456, 0)
        ))
        for column in 0..<4 {
            #expect(simd_distance(frame.graphCamera.viewMatrix[column], view[column]) < 0.000_01)
            #expect(simd_distance(frame.graphCamera.projectionMatrix[column], projection[column]) < 0.000_01)
        }
        #expect(frame.clipFromWorld == frame.graphCamera.projectionMatrix * frame.graphCamera.viewMatrix)
        #expect(frame.cameraPosition == eye)
        #expect(abs(frame.aspectRatio - 2) < 0.000_01)
        let opticalAxis = ndc(eye - back * 4, frame)
        #expect(simd_distance(SIMD2(opticalAxis.x, opticalAxis.y), SIMD2(-0.35, -0.3)) < 0.000_01)
    }

    @Test("The four screen corners and clip depths are exact for forward and reversed views",
          arguments: [MeshDepthMode.forward, .reversed])
    func cornersAndDepth(_ mode: MeshDepthMode) throws {
        let topRight = bottomRight + topLeft - bottomLeft
        let corners: [(SIMD3<Float>, SIMD2<Float>)] = [
            (bottomLeft, SIMD2(-1, -1)), (bottomRight, SIMD2(1, -1)),
            (topLeft, SIMD2(-1, 1)), (topRight, SIMD2(1, 1)),
        ]
        let nearDepth: Float = mode == .reversed ? 1 : 0
        let farDepth = 1 - nearDepth
        for far: Float in [20, .infinity] {
            let frame = try framed(far: far, depthMode: mode)
            #expect(frame.depthMode == mode && frame.graphCamera.near == 0.25 && frame.graphCamera.far == far)
            for (corner, expected) in corners {
                let projected = ndc(corner, frame)
                #expect(simd_distance(SIMD2(projected.x, projected.y), expected) < 0.000_01)
            }
            #expect(abs(ndc(eye - back * 0.25, frame).z - nearDepth) < 0.000_02)
            let farPoint: Float = far.isFinite ? far : 100_000
            #expect(abs(ndc(eye - back * farPoint, frame).z - farDepth) < 0.000_02)
            let ray = try frame.graphCamera.ray(at: SIMD2(0.5, 0.5), depthMode: mode)
            let center = (bottomRight + topLeft) * 0.5
            #expect(simd_distance(ray.direction, simd_normalize(center - eye)) < 0.000_02)
            #expect(ray.maximumDistance.isInfinite == far.isInfinite)
        }
    }

    @Test("The off-axis raster pass and picking agree for both Metal depth mappings",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"),
          arguments: [MeshDepthMode.forward, .reversed])
    func raster(_ mode: MeshDepthMode) throws {
        let gpu = try RasterLightingFixture(width: 80, height: 40)
        let eye = SIMD3<Float>(0.8, 0.4, 0)
        let frame = try CameraUtils.frameCorners(
            eye: eye, bottomLeft: SIMD3(-2, -1, -4), bottomRight: SIMD3(2, -1, -4), topLeft: SIMD3(-2, 1, -4),
            near: 0.25, far: 12, depthMode: mode
        )
        let material = try PhysicallyBasedMaterial(shadingModel: .unlit)
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.55, height: 0.48),
            material: .physicallyBased(material), depthMode: mode
        )
        let red = SIMD4<Float>(1, 0, 0, 1), green = SIMD4<Float>(0, 1, 0, 1), blue = SIMD4<Float>(0, 0, 1, 1)
        let upperLeft = SIMD3<Float>(-1, 0.5, -4)
        let near = SceneNode(
            transform: .init(translation: eye + (upperLeft - eye) * 0.75), mesh: mesh, materialParameters: red
        )
        let far = SceneNode(
            transform: .init(translation: eye + (upperLeft - eye) * 1.5), mesh: mesh, materialParameters: green
        )
        let lowerRight = SceneNode(transform: .init(translation: SIMD3(1, -0.5, -4)), mesh: mesh, materialParameters: blue)
        let outside = SceneNode(transform: .init(translation: SIMD3(4, 0, -4)), mesh: mesh)
        let graph = SceneGraph(
            roots: [near, far, lowerRight, outside], camera: try PerspectiveCamera(verticalFieldOfView: 0.6, near: 0.25, far: 12)
        )
        let color = try gpu.colorTarget(), depth = try gpu.depthTarget()
        try gpu.run { command in
            let stats = try graph.encode(
                pass: gpu.pass(color: color, depth: depth, depthMode: mode), commandBuffer: command,
                clipFromWorld: frame.clipFromWorld, cameraPosition: frame.cameraPosition,
                depthMode: frame.depthMode, graphCamera: frame.graphCamera
            )
            #expect(stats.drawnMeshCount == 3 && stats.culledMeshCount == 1)
        }
        let pixels = try gpu.pixels(of: color)
        #expect(pixels[10 * 80 + 20] == red)
        #expect(pixels[30 * 80 + 60] == blue)
        #expect(pixels[20 * 80 + 40] == .zero)
        #expect(try graph.pick(at: SIMD2(0.25, 0.25), camera: frame.graphCamera, depthMode: mode)?.nodeID == near.id)
        #expect(try graph.pick(at: SIMD2(0.75, 0.75), camera: frame.graphCamera, depthMode: mode)?.nodeID == lowerRight.id)
    }

    @Test("Invalid rectangles, eye locations and clip planes fail before a view is returned")
    func validation() throws {
        func frame(
            eye: SIMD3<Float> = SIMD3(0, 0, 1), right: SIMD3<Float> = SIMD3(1, 0, 0), up: SIMD3<Float> = SIMD3(0, 1, 0),
            near: Float = 0.25, far: Float = 20
        ) throws -> CameraUtils.FramedView {
            try CameraUtils.frameCorners(eye: eye, bottomLeft: .zero, bottomRight: right, topLeft: up, near: near, far: far)
        }
        #expect(throws: CameraUtils.Failure.nonfinitePoint) { try frame(eye: SIMD3(.nan, 0, 1)) }
        #expect(throws: CameraUtils.Failure.invalidRectangle) { try frame(right: .zero) }
        #expect(throws: CameraUtils.Failure.invalidRectangle) { try frame(up: SIMD3(0.25, 1, 0)) }
        #expect(throws: CameraUtils.Failure.eyeBehindScreen) { try frame(eye: SIMD3(0, 0, -1)) }
        #expect(throws: CameraUtils.Failure.eyeBehindScreen) { try frame(eye: SIMD3(1, 2, 0)) }
        #expect(throws: CameraUtils.Failure.eyeBehindScreen) { try frame(right: SIMD3(-1, 0, 0)) }
        #expect(throws: CameraUtils.Failure.invalidClippingPlanes) { try frame(near: 0) }
        #expect(throws: CameraUtils.Failure.invalidClippingPlanes) { try frame(near: 1, far: 1) }
        #expect(throws: CameraUtils.Failure.invalidClippingPlanes) { try frame(far: .nan) }
        #expect(throws: CameraUtils.Failure.unrepresentableView) {
            try frame(right: SIMD3(.leastNonzeroMagnitude, 0, 0))
        }
        #expect(throws: CameraUtils.Failure.unrepresentableView) { try frame(far: .greatestFiniteMagnitude) }

        let offscreenAxis = try frame(eye: SIMD3(4, 0.2, 1))
        let leftCorner = ndc(.zero, offscreenAxis)
        #expect(simd_distance(SIMD2(leftCorner.x, leftCorner.y), SIMD2(-1, -1)) < 0.000_01)
    }

    private func framed(far: Float, depthMode: MeshDepthMode) throws -> CameraUtils.FramedView {
        try CameraUtils.frameCorners(
            eye: eye, bottomLeft: bottomLeft, bottomRight: bottomRight, topLeft: topLeft,
            near: 0.25, far: far, depthMode: depthMode
        )
    }

    private func ndc(_ point: SIMD3<Float>, _ frame: CameraUtils.FramedView) -> SIMD3<Float> {
        let clip = frame.clipFromWorld * SIMD4(point, 1)
        return SIMD3(clip.x, clip.y, clip.z) / clip.w
    }
}
