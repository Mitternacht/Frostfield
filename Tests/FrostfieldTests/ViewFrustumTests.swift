import Frostfield
import Testing
import simd

struct ViewFrustumTests {
    private static let near: Float = 1
    private static let far: Float = 10

    /// A 90° perspective with Metal depth, reversed when requested, optionally with an infinite far plane.
    private static func perspective(reversed: Bool, infinite: Bool = false) -> simd_float4x4 {
        let (z, offset): (Float, Float)
        switch (reversed, infinite) {
        case (false, false): (z, offset) = (far / (near - far), near * far / (near - far))
        case (true, false): (z, offset) = (near / (far - near), near * far / (far - near))
        case (false, true): (z, offset) = (-1, -near)
        case (true, true): (z, offset) = (0, near)
        }
        return simd_float4x4(columns: (
            SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, z, -1), SIMD4(0, 0, offset, 0)
        ))
    }

    private static func box(center: SIMD3<Float>, half: Float) throws -> AxisAlignedBox {
        try AxisAlignedBox(minimum: center - half, maximum: center + half)
    }

    @Test
    func `Forward projection matches PerspectiveCamera and maps depth to zero through one`() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: Self.near, far: Self.far)
        let projection = try camera.projection(aspectRatio: 1)
        let mine = Self.perspective(reversed: false)
        for column in 0..<4 {
            #expect(simd_length(projection[column] - mine[column]) < 1e-6)
        }
        let reversed = Self.perspective(reversed: true)
        let nearPoint = reversed * SIMD4<Float>(0, 0, -Self.near, 1)
        let farPoint = reversed * SIMD4<Float>(0, 0, -Self.far, 1)
        #expect(abs(nearPoint.z / nearPoint.w - 1) < 1e-6)
        #expect(abs(farPoint.z / farPoint.w) < 1e-6)
    }

    @Test(arguments: [false, true])
    func `Frustum classifies points and boxes for forward and reversed depth`(reversed: Bool) throws {
        let frustum = try ViewFrustum(clipFromWorld: Self.perspective(reversed: reversed))
        #expect(frustum.planes.count == 6)
        for plane in frustum.planes {
            #expect(abs(simd_length(plane.normal) - 1) < 1e-6)
        }

        #expect(frustum.contains(point: SIMD3(0, 0, -5)))
        #expect(frustum.contains(point: SIMD3(4.9, -4.9, -5)))
        #expect(!frustum.contains(point: SIMD3(5.1, 0, -5)))
        #expect(!frustum.contains(point: SIMD3(0, 0, -0.5)))
        #expect(!frustum.contains(point: SIMD3(0, 0, -10.5)))
        #expect(!frustum.contains(point: SIMD3(0, 0, 5)))

        #expect(frustum.classify(box: try Self.box(center: SIMD3(0, 0, -5), half: 0.5)) == .inside)
        #expect(frustum.classify(box: try Self.box(center: SIMD3(0, 0, 5), half: 0.5)) == .outside)
        #expect(frustum.classify(box: try Self.box(center: SIMD3(0, 9, -5), half: 0.5)) == .outside)
        #expect(frustum.classify(box: try Self.box(center: SIMD3(0, 0, -12), half: 0.5)) == .outside)
        #expect(frustum.classify(box: try Self.box(center: SIMD3(0, 0, -1), half: 0.25)) == .intersecting)
        #expect(frustum.classify(box: try Self.box(center: SIMD3(0, 0, -10), half: 0.25)) == .intersecting)
        #expect(frustum.classify(box: try Self.box(center: SIMD3(5, 0, -5), half: 0.5)) == .intersecting)
        #expect(frustum.classify(box: try Self.box(center: .zero, half: 100)) == .intersecting)
        #expect(frustum.intersects(box: try Self.box(center: SIMD3(-5, 5, -5), half: 0.5)))
        #expect(!frustum.intersects(box: try Self.box(center: SIMD3(-7, 0, -5), half: 0.5)))

        #expect(frustum.intersects(sphere: try BoundingSphere(center: SIMD3(0, 0, -5), radius: 1)))
        #expect(frustum.intersects(sphere: try BoundingSphere(center: SIMD3(0, 0, 0), radius: 1.2)))
        #expect(!frustum.intersects(sphere: try BoundingSphere(center: SIMD3(0, 0, 0), radius: 0.8)))
        #expect(!frustum.intersects(sphere: try BoundingSphere(center: SIMD3(0, 0, -20), radius: 5)))
    }

    @Test(arguments: [false, true])
    func `Infinite projections omit the far plane`(reversed: Bool) throws {
        let frustum = try ViewFrustum(clipFromWorld: Self.perspective(reversed: reversed, infinite: true))
        #expect(frustum.planes.count == 5)
        #expect(frustum.contains(point: SIMD3(0, 0, -1_000_000)))
        #expect(!frustum.contains(point: SIMD3(0, 0, -0.5)))
        #expect(frustum.classify(box: try Self.box(center: SIMD3(0, 0, -5000), half: 10)) == .inside)
    }

    @Test(arguments: [false, true])
    func `Camera transforms and orthographic projections move the frustum`(reversed: Bool) throws {
        // The camera sits at x = 10 and looks toward −X.
        let camera = SceneTransform(translation: SIMD3(10, 0, 0), rotation: simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0)))
        let view = simd_inverse(camera.matrix)
        let frustum = try ViewFrustum(clipFromWorld: Self.perspective(reversed: reversed) * view)
        #expect(frustum.contains(point: SIMD3(5, 0, 0)))
        #expect(!frustum.contains(point: SIMD3(15, 0, 0)))
        #expect(frustum.classify(box: try Self.box(center: SIMD3(4, 1, -1), half: 0.5)) == .inside)
        #expect(frustum.classify(box: try Self.box(center: SIMD3(4, 0, 9), half: 0.5)) == .outside)

        // An 8 × 6 orthographic view spanning depths 2...20.
        let depthScale: Float = reversed ? 1 / 18 : -1 / 18
        let depthOffset: Float = reversed ? 20 / 18 : -2 / 18
        let orthographic = simd_float4x4(columns: (
            SIMD4(0.25, 0, 0, 0), SIMD4(0, 1 / 3, 0, 0), SIMD4(0, 0, depthScale, 0), SIMD4(0, 0, depthOffset, 1)
        ))
        let box = try ViewFrustum(clipFromWorld: orthographic * view)
        #expect(box.planes.count == 6)
        #expect(box.contains(point: SIMD3(-9, 2.9, 3.9)))
        #expect(!box.contains(point: SIMD3(-9, 3.1, 0)))
        #expect(!box.contains(point: SIMD3(9, 0, 0)))
        #expect(!box.contains(point: SIMD3(-11, 0, 0)))
        #expect(box.classify(box: try Self.box(center: SIMD3(0, 0, 0), half: 1)) == .inside)
        #expect(box.classify(box: try Self.box(center: SIMD3(0, 0, 4), half: 1)) == .intersecting)
    }

    @Test
    func `Transformed mesh bounds feed frustum culling`() throws {
        let frustum = try ViewFrustum(clipFromWorld: Self.perspective(reversed: false))
        let cube = MeshGeometry.cube()
        let visible = SceneTransform(translation: SIMD3(0, 0, -4), rotation: simd_quatf(angle: 0.7, axis: SIMD3(0, 1, 0)))
        let hidden = SceneTransform(translation: SIMD3(0, 0, 4))
        #expect(frustum.classify(box: try cube.bounds.transformed(by: visible.matrix)) == .inside)
        #expect(frustum.classify(box: try cube.bounds.transformed(by: hidden.matrix)) == .outside)
        #expect(frustum.intersects(sphere: try cube.boundingSphere.transformed(by: visible.matrix)))
        #expect(!frustum.intersects(sphere: try cube.boundingSphere.transformed(by: hidden.matrix)))
    }

    @Test
    func `Frustum rejects non-finite and degenerate matrices`() {
        var invalid = matrix_identity_float4x4
        invalid.columns.2.z = .nan
        #expect(throws: ViewFrustum.Failure.nonFiniteMatrix) { try ViewFrustum(clipFromWorld: invalid) }
        #expect(throws: ViewFrustum.Failure.degenerateMatrix) { try ViewFrustum(clipFromWorld: simd_float4x4()) }
    }

    @Test
    func `Mesh bounds enclose every vertex`() throws {
        let cube = MeshGeometry.cube()
        #expect(cube.bounds == (try AxisAlignedBox(minimum: SIMD3(repeating: -0.5), maximum: SIMD3(repeating: 0.5))))
        #expect(cube.bounds.center == .zero)
        #expect(cube.bounds.size == SIMD3(repeating: 1))
        #expect(abs(cube.boundingSphere.radius - Float(0.75).squareRoot()) < 1e-6)

        let sphere = try MeshGeometry.uvSphere(radius: 2, segments: 32, rings: 16)
        let bounding = sphere.boundingSphere
        #expect(simd_length(bounding.center) < 1e-6)
        #expect(bounding.radius >= 2 && bounding.radius < 2 * (1 + 1e-6))
        for vertex in sphere.vertices {
            #expect(bounding.contains(vertex.position))
            #expect(sphere.bounds.contains(vertex.position))
        }
        // The vertex sphere is tighter than the box's corner sphere.
        #expect(bounding.radius < sphere.bounds.boundingSphere.radius)
    }

    @Test
    func `Box transforms enclose the transformed corners tightly`() throws {
        let box = try AxisAlignedBox(minimum: SIMD3(-1, 0.5, -2), maximum: SIMD3(3, 1.5, 0))
        let transform = SceneTransform(
            translation: SIMD3(4, -2, 7),
            rotation: simd_quatf(angle: 0.6, axis: simd_normalize(SIMD3(1, 1, 0))),
            scale: SIMD3(2, -0.5, 1.5)
        ).matrix
        let corners = (0..<8).map { index -> SIMD3<Float> in
            let corner = SIMD4<Float>(
                index & 1 == 0 ? box.minimum.x : box.maximum.x,
                index & 2 == 0 ? box.minimum.y : box.maximum.y,
                index & 4 == 0 ? box.minimum.z : box.maximum.z, 1
            )
            let moved = transform * corner
            return SIMD3(moved.x, moved.y, moved.z)
        }
        let expected = try AxisAlignedBox(enclosing: corners)
        let transformed = try box.transformed(by: transform)
        #expect(simd_length(transformed.minimum - expected.minimum) < 1e-5)
        #expect(simd_length(transformed.maximum - expected.maximum) < 1e-5)
        #expect(transformed.intersects(expected))

        var projective = matrix_identity_float4x4
        projective.columns.2.w = -1
        #expect(throws: AxisAlignedBox.Failure.invalidTransform) { try box.transformed(by: projective) }
        let huge = SceneTransform(scale: SIMD3(repeating: .greatestFiniteMagnitude)).matrix
        #expect(throws: AxisAlignedBox.Failure.unrepresentableResult) { try box.transformed(by: huge) }
    }

    @Test
    func `Sphere transforms enclose sheared and scaled points`() throws {
        let sphere = try BoundingSphere(center: SIMD3(1, -1, 2), radius: 1.5)
        var shear = SceneTransform(translation: SIMD3(0, 3, -1), scale: SIMD3(1, 2, 0.5)).matrix
        shear.columns.1.x = 1.2
        let transformed = try sphere.transformed(by: shear)
        var random = BVHTestRandom(seed: 21)
        var farthest: Float = 0
        for _ in 0..<2000 {
            var direction = random.vector(in: -1...1)
            if direction == .zero { continue }
            direction = simd_normalize(direction)
            let point = shear * SIMD4(sphere.center + direction * sphere.radius, 1)
            let moved = SIMD3(point.x, point.y, point.z)
            #expect(transformed.contains(moved))
            farthest = max(farthest, simd_length(moved - transformed.center))
        }
        // Random samples approach, but never exceed, the ellipsoid's longest semi-axis.
        #expect(farthest > transformed.radius * 0.97)

        let similar = SceneTransform(
            translation: SIMD3(5, 0, 0), rotation: simd_quatf(angle: 1.1, axis: SIMD3(0, 0, 1)), scale: SIMD3(repeating: 3)
        )
        let scaled = try sphere.transformed(by: similar.matrix)
        #expect(abs(scaled.radius - 4.5) < 1e-5)
    }

    @Test
    func `Bounds validate their inputs`() throws {
        #expect(throws: AxisAlignedBox.Failure.invertedCorners) {
            try AxisAlignedBox(minimum: SIMD3(1, 0, 0), maximum: SIMD3(0, 1, 1))
        }
        #expect(throws: AxisAlignedBox.Failure.nonFiniteCorner) {
            try AxisAlignedBox(minimum: SIMD3(.nan, 0, 0), maximum: SIMD3(1, 1, 1))
        }
        #expect(throws: AxisAlignedBox.Failure.noPoints) { try AxisAlignedBox(enclosing: [SIMD3<Float>]()) }
        #expect(throws: AxisAlignedBox.Failure.nonFinitePoint(1)) {
            try AxisAlignedBox(enclosing: [SIMD3<Float>(0, 0, 0), SIMD3(.infinity, 0, 0)])
        }
        #expect(throws: BoundingSphere.Failure.invalidRadius) { try BoundingSphere(center: .zero, radius: -1) }
        #expect(throws: BoundingSphere.Failure.nonFiniteCenter) { try BoundingSphere(center: SIMD3(0, .nan, 0), radius: 1) }

        let point = try AxisAlignedBox(minimum: SIMD3(1, 1, 1), maximum: SIMD3(1, 1, 1))
        #expect(point.contains(SIMD3(1, 1, 1)))
        let touching = try AxisAlignedBox(minimum: SIMD3(1, 0, 0), maximum: SIMD3(2, 1, 1))
        #expect(point.intersects(touching))
        #expect(!point.intersects(try AxisAlignedBox(minimum: SIMD3(1.01, 0, 0), maximum: SIMD3(2, 2, 2))))
    }
}
