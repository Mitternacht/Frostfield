import Testing
import simd
@testable import Frostfield

/// Applies a column-vector matrix to a point and divides by w.
private func transformPoint(_ matrix: simd_float4x4, _ point: SIMD3<Float>) -> SIMD3<Float> {
    let result = matrix * SIMD4(point, 1)
    return SIMD3(result.x, result.y, result.z) / result.w
}

private func isClose(_ a: SIMD3<Float>, _ b: SIMD3<Float>, tolerance: Float = 1e-5) -> Bool {
    simd_reduce_max(simd_abs(a - b)) <= tolerance
}

private func isClose(_ a: simd_float4x4, _ b: simd_float4x4, tolerance: Float = 1e-5) -> Bool {
    (0..<4).allSatisfy { simd_reduce_max(simd_abs(a[$0] - b[$0])) <= tolerance }
}

private func degrees(_ value: Float) -> Float { value * .pi / 180 }

/// The transform convention: right-handed, column vectors, scale then rotate
/// then translate.
struct TransformTests {
    @Test func `The identity transform leaves points unchanged`() {
        #expect(Transform.identity.matrix == matrix_identity_float4x4)
        #expect(transformPoint(Transform.identity.matrix, [1, 2, 3]) == [1, 2, 3])
    }

    @Test func `Translation occupies the last column`() {
        let matrix = Transform(translation: [3, -4, 5]).matrix
        #expect(matrix.columns.3 == SIMD4(3, -4, 5, 1))
        #expect(transformPoint(matrix, [1, 1, 1]) == [4, -3, 6])
    }

    /// A point is scaled first, then rotated, then translated: with a scale of
    /// 2 along x, a quarter turn about z, and a translation along z, the point
    /// (1, 0, 0) goes to (2, 0, 0), then (0, 2, 0), then (0, 2, 5).
    @Test func `Scale applies first, then rotation, then translation`() {
        let transform = Transform(
            translation: [0, 0, 5],
            rotation: simd_quatf(angle: .pi / 2, axis: [0, 0, 1]),
            scale: [2, 1, 1]
        )
        // Scaling after rotating would have stretched y instead: (0, 1, 5).
        #expect(isClose(transformPoint(transform.matrix, [1, 0, 0]), [0, 2, 5]))
        // A quarter turn about y takes (0, 0, 1) to (1, 0, 0), and the
        // translation then gives (2, 0, 0). Translating before rotating would
        // have given (1, 0, -1).
        let turned = Transform(translation: [1, 0, 0], rotation: simd_quatf(angle: .pi / 2, axis: [0, 1, 0]))
        #expect(isClose(transformPoint(turned.matrix, [0, 0, 1]), [2, 0, 0]))
    }

    @Test func `The matrix equals translation times rotation times scale`() {
        let translation: SIMD3<Float> = [0.5, -2, 3]
        let rotation = simd_quatf(angle: 0.7, axis: simd_normalize([1, 2, 3]))
        let scale: SIMD3<Float> = [2, 0.5, 3]
        var t = matrix_identity_float4x4
        t.columns.3 = SIMD4(translation, 1)
        let r = simd_float4x4(rotation)
        let s = simd_float4x4(diagonal: SIMD4(scale, 1))
        let transform = Transform(translation: translation, rotation: rotation, scale: scale)
        #expect(isClose(transform.matrix, t * r * s))
        #expect(!isClose(transform.matrix, s * r * t))
    }

    /// Right-handed rotations: a positive angle turns counter-clockwise seen
    /// from the positive end of the axis.
    @Test(arguments: [
        (SIMD3<Float>(0, 0, 1), SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0)),  // about z: x to y
        (SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0), SIMD3<Float>(0, 0, 1)),  // about x: y to z
        (SIMD3<Float>(0, 1, 0), SIMD3<Float>(0, 0, 1), SIMD3<Float>(1, 0, 0)),  // about y: z to x
    ])
    func `A positive quarter turn is counter-clockwise about its axis`(
        axis: SIMD3<Float>, point: SIMD3<Float>, expected: SIMD3<Float>
    ) {
        let matrix = Transform(rotation: simd_quatf(angle: .pi / 2, axis: axis)).matrix
        #expect(isClose(transformPoint(matrix, point), expected))
    }

    @Test func `Only the direction of the rotation quaternion matters`() {
        let unit = simd_quatf(angle: 1.1, axis: simd_normalize([0, 1, 1]))
        let scaled = simd_quatf(vector: unit.vector * 3)
        #expect(isClose(Transform(rotation: scaled).matrix, Transform(rotation: unit).matrix))
    }

    @Test func `A negative scale mirrors the object`() {
        let matrix = Transform(scale: [-1, 1, 1]).matrix
        #expect(transformPoint(matrix, [2, 3, 4]) == [-2, 3, 4])
        #expect(simd_determinant(matrix) < 0)
    }
}

/// The perspective projection maps the view frustum into Metal clip space:
/// x and y in [-1, 1], depth in [0, 1] from near to far.
struct PerspectiveProjectionTests {
    @Test func `Near maps to depth 0 and far to depth 1`() throws {
        let projection = try PerspectiveProjection(verticalFieldOfView: degrees(60), near: 0.5, far: 20)
        let matrix = try projection.matrix(aspectRatio: 1.5)
        #expect(abs(transformPoint(matrix, [0, 0, -0.5]).z - 0) < 1e-6)
        #expect(abs(transformPoint(matrix, [0, 0, -20]).z - 1) < 1e-5)
        // Depth increases with distance in between.
        let depths = [1, 2, 5, 10, 19].map { transformPoint(matrix, [0, 0, -Float($0)]).z }
        #expect(depths == depths.sorted())
        #expect(depths.allSatisfy { $0 > 0 && $0 < 1 })
    }

    @Test func `Points outside the near and far planes leave the depth range`() throws {
        let projection = try PerspectiveProjection(verticalFieldOfView: degrees(90), near: 1, far: 10)
        let matrix = try projection.matrix(aspectRatio: 1)
        #expect(transformPoint(matrix, [0, 0, -0.9]).z < 0)
        #expect(transformPoint(matrix, [0, 0, -11]).z > 1)
    }

    @Test func `Clip-space w is the distance in front of the camera`() throws {
        let matrix = try PerspectiveProjection(verticalFieldOfView: 1, near: 0.1, far: 100).matrix(aspectRatio: 1)
        #expect((matrix * SIMD4(0, 0, -7, 1)).w == 7)
    }

    /// The top of the field of view maps to y = 1; the right edge, which is
    /// wider by the aspect ratio, maps to x = 1.
    @Test(arguments: [Float(0.5), 1, 2.25])
    func `The field of view and aspect ratio set the frustum edges`(aspectRatio: Float) throws {
        let fieldOfView = degrees(50)
        let matrix = try PerspectiveProjection(verticalFieldOfView: fieldOfView, near: 0.1, far: 100)
            .matrix(aspectRatio: aspectRatio)
        let distance: Float = 4
        let halfHeight = tan(fieldOfView / 2) * distance
        let top = transformPoint(matrix, [0, halfHeight, -distance])
        let right = transformPoint(matrix, [halfHeight * aspectRatio, 0, -distance])
        #expect(abs(top.y - 1) < 1e-5)
        #expect(abs(right.x - 1) < 1e-5)
        // +y stays up and +x stays right.
        #expect(transformPoint(matrix, [-halfHeight * aspectRatio, -halfHeight, -distance]).x < 0)
    }

    @Test(arguments: [Float(0), -0.5, .pi, 4, .nan, .infinity])
    func `An invalid field of view is rejected`(fieldOfView: Float) {
        let error = #expect(throws: ProjectionError.self) {
            try PerspectiveProjection(verticalFieldOfView: fieldOfView, near: 0.1, far: 10)
        }
        guard case .invalidFieldOfView(let value)? = error else {
            Issue.record("Unexpected error \(String(describing: error))")
            return
        }
        #expect(value.bitPattern == fieldOfView.bitPattern)
        #expect(error?.description.contains("less than π radians") == true)
    }

    @Test func `A field of view in degrees gets a conversion hint`() {
        let error = #expect(throws: ProjectionError.invalidFieldOfView(60)) {
            try PerspectiveProjection(verticalFieldOfView: 60, near: 0.1, far: 10)
        }
        #expect(error?.description.contains("degrees") == true)
        #expect(ProjectionError.invalidFieldOfView(0).description.contains("degrees") == false)
    }

    @Test(arguments: [Float(0), -1, .nan, .infinity])
    func `A near plane at or behind the eye is rejected`(near: Float) {
        let error = #expect(throws: ProjectionError.self) {
            try PerspectiveProjection(verticalFieldOfView: 1, near: near, far: 10)
        }
        guard case .invalidNearPlane(let value)? = error else {
            Issue.record("Unexpected error \(String(describing: error))")
            return
        }
        #expect(value.bitPattern == near.bitPattern)
        #expect(error?.description.contains("greater than 0") == true)
    }

    @Test(arguments: [Float(1), 0.5, .infinity, .nan])
    func `A far plane that is not beyond the near plane is rejected`(far: Float) {
        let error = #expect(throws: ProjectionError.self) {
            try PerspectiveProjection(verticalFieldOfView: 1, near: 1, far: far)
        }
        guard case .invalidFarPlane(let near, let value)? = error else {
            Issue.record("Unexpected error \(String(describing: error))")
            return
        }
        #expect(near == 1)
        #expect(value.bitPattern == far.bitPattern)
        #expect(error?.description.contains("greater than the near-plane distance (1.0)") == true)
    }

    @Test(arguments: [Float(0), -1.5, .nan, .infinity])
    func `An invalid aspect ratio is rejected`(aspectRatio: Float) throws {
        let projection = try PerspectiveProjection(verticalFieldOfView: 1, near: 0.1, far: 10)
        let camera = Camera(position: .zero, projection: projection)
        for error in [
            #expect(throws: ProjectionError.self) { try projection.matrix(aspectRatio: aspectRatio) },
            #expect(throws: ProjectionError.self) { try camera.viewProjectionMatrix(aspectRatio: aspectRatio) },
        ] {
            guard case .invalidAspectRatio(let value)? = error else {
                Issue.record("Unexpected error \(String(describing: error))")
                continue
            }
            #expect(value.bitPattern == aspectRatio.bitPattern)
            #expect(error?.description.contains("pixel size") == true)
        }
    }
}

/// The camera's view matrix undoes its placement in the world.
struct CameraTests {
    let projection: PerspectiveProjection

    init() throws {
        projection = try PerspectiveProjection(verticalFieldOfView: degrees(90), near: 0.1, far: 100)
    }

    @Test func `A camera with the identity orientation looks along negative z`() {
        let camera = Camera(position: [0, 0, 5], projection: projection)
        #expect(transformPoint(camera.viewMatrix, .zero) == [0, 0, -5])
        #expect(transformPoint(camera.viewMatrix, [1, 2, 5]) == [1, 2, 0])
    }

    /// A camera turned a quarter turn about +y looks along -x, so a point on
    /// the negative x axis lies straight ahead.
    @Test func `The orientation turns the view direction`() {
        let camera = Camera(
            position: [1, 0, 0],
            orientation: simd_quatf(angle: .pi / 2, axis: [0, 1, 0]),
            projection: projection
        )
        #expect(isClose(transformPoint(camera.viewMatrix, [-3, 0, 0]), [0, 0, -4]))
    }

    @Test func `The view matrix inverts the camera's placement`() {
        let camera = Camera(
            position: [2, -1, 4],
            orientation: simd_quatf(angle: 0.8, axis: simd_normalize([1, -2, 0.5])),
            projection: projection
        )
        let placement = Transform(translation: camera.position, rotation: camera.orientation).matrix
        #expect(isClose(camera.viewMatrix * placement, matrix_identity_float4x4))
    }

    @Test func `View projection is projection times view`() throws {
        let camera = Camera(position: [0, 1, 3], orientation: simd_quatf(angle: 0.3, axis: [1, 0, 0]), projection: projection)
        let expected = try projection.matrix(aspectRatio: 1.25) * camera.viewMatrix
        #expect(try camera.viewProjectionMatrix(aspectRatio: 1.25) == expected)
        #expect(camera.uncheckedViewProjectionMatrix(aspectRatio: 1.25) == expected)
    }
}
