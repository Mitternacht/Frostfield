import Frostfield
import Testing
import simd

struct SceneMathTests {
    @Test
    func `Transform scales before rotation and translates last`() {
        let transform = SceneTransform(
            translation: SIMD3<Float>(5, 6, 7),
            rotation: simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(0, 0, 1)),
            scale: SIMD3<Float>(2, 3, 4)
        )

        let point = SIMD4<Float>(1, 2, 3, 1)
        let worldPoint = transform.matrix * point
        #expect(simd_length(worldPoint - SIMD4<Float>(-1, 8, 19, 1)) < 0.0001)
    }

    @Test
    func `Perspective maps clip planes and frustum edges to Metal coordinates`() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.5, far: 12)
        let projection = try camera.projection(aspectRatio: 2)

        let nearPoint = projection * SIMD4<Float>(0, 0, -0.5, 1)
        let farPoint = projection * SIMD4<Float>(0, 0, -12, 1)
        let upperRight = projection * SIMD4<Float>(4, 2, -2, 1)

        #expect(abs(nearPoint.z / nearPoint.w) < 0.0001)
        #expect(abs(farPoint.z / farPoint.w - 1) < 0.0001)
        #expect(abs(upperRight.x / upperRight.w - 1) < 0.0001)
        #expect(abs(upperRight.y / upperRight.w - 1) < 0.0001)
        #expect(nearPoint.w > 0)
    }

    @Test(arguments: [Float(0), -1, .pi, .infinity, .nan])
    func `Camera rejects invalid fields of view`(fieldOfView: Float) {
        #expect(throws: PerspectiveCamera.ValidationError.invalidFieldOfView) {
            try PerspectiveCamera(verticalFieldOfView: fieldOfView, near: 0.1, far: 10)
        }
    }

    @Test
    func `Camera rejects invalid clip distances`() {
        #expect(throws: PerspectiveCamera.ValidationError.invalidClippingPlanes) {
            try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0, far: 10)
        }
        #expect(throws: PerspectiveCamera.ValidationError.invalidClippingPlanes) {
            try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 2, far: 1)
        }
        #expect(throws: PerspectiveCamera.ValidationError.invalidClippingPlanes) {
            try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 1, far: .infinity)
        }
    }

    @Test(arguments: [Float(0), -1, .infinity, .nan])
    func `Projection rejects invalid drawable aspect ratios`(aspectRatio: Float) throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)

        #expect(throws: PerspectiveCamera.ValidationError.invalidAspectRatio) {
            try camera.projection(aspectRatio: aspectRatio)
        }
    }

    @Test
    func `Projection rejects a finite aspect ratio that overflows its matrix`() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)

        #expect(throws: PerspectiveCamera.ValidationError.unrepresentableProjection) {
            try camera.projection(aspectRatio: .leastNonzeroMagnitude)
        }
    }
}
