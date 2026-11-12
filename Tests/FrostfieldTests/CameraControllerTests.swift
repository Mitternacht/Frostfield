import Frostfield
import Testing
import simd

struct CameraControllerTests {
    private static let fieldOfView = Float.pi / 3

    private func forward(_ transform: SceneTransform) -> SIMD3<Float> {
        transform.rotation.act(SIMD3(0, 0, -1))
    }

    private func close(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ tolerance: Float = 1e-5) -> Bool {
        simd_length(a - b) <= tolerance
    }

    private func close(_ a: SIMD2<Float>, _ b: SIMD2<Float>, _ tolerance: Float = 1e-4) -> Bool {
        simd_length(a - b) <= tolerance
    }

    /// Projects a world point to normalized top-left drawable coordinates.
    private func normalized(
        _ point: SIMD3<Float>, camera: SceneTransform, aspectRatio: Float,
        fieldOfView: Float = fieldOfView, orthographicHalfHeight: Float? = nil
    ) -> SIMD2<Float> {
        let view = simd_inverse(camera.matrix) * SIMD4(point, 1)
        let ndc: SIMD2<Float>
        if let halfHeight = orthographicHalfHeight {
            ndc = SIMD2(view.x / (halfHeight * aspectRatio), view.y / halfHeight)
        } else {
            let reach = tan(fieldOfView / 2) * -view.z
            ndc = SIMD2(view.x / (reach * aspectRatio), view.y / reach)
        }
        return SIMD2((ndc.x + 1) / 2, (1 - ndc.y) / 2)
    }

    private func orbit(
        target: SIMD3<Float> = SIMD3(0.5, -0.25, -1), distance: Float = 6, yaw: Float = 0.4, pitch: Float = 0.3,
        projection: OrbitCameraController.Projection = .perspective(verticalFieldOfView: fieldOfView),
        limits: OrbitCameraController.Limits = .init(), panMode: OrbitCameraController.PanMode = .viewPlane,
        dampingRate: Float? = nil
    ) throws -> OrbitCameraController {
        try OrbitCameraController(
            target: target, distance: distance, yaw: yaw, pitch: pitch, projection: projection,
            orthographicHalfHeight: 3, limits: limits, panMode: panMode, dampingRate: dampingRate
        )
    }

    @Test
    func `Orbit sits on the target's +Z side at zero yaw and looks at the target`() throws {
        var controller = try orbit(target: SIMD3(1, 2, 3), distance: 4, yaw: 0, pitch: 0)
        #expect(close(controller.cameraTransform.translation, SIMD3(1, 2, 7)))
        #expect(close(forward(controller.cameraTransform), SIMD3(0, 0, -1)))

        try controller.rotate(by: SIMD2(.pi / 2, .pi / 4))
        try controller.update(deltaTime: 0)
        let half = 4 * Float(0.5).squareRoot()
        #expect(close(controller.cameraTransform.translation, SIMD3(1 + half, 2 + half, 3), 1e-5))
        let expected = simd_normalize(SIMD3<Float>(-1, -1, 0))
        #expect(close(forward(controller.cameraTransform), expected))
        // The camera's up vector stays in the vertical plane through the target.
        #expect(abs(controller.cameraTransform.rotation.act(SIMD3(0, 1, 0)).z) < 1e-6)
    }

    @Test
    func `Rotating around the target keeps the distance and the aim`() throws {
        var controller = try orbit()
        var random = BVHTestRandom(seed: 11)
        for _ in 0..<50 {
            try controller.rotate(by: SIMD2(random.float(in: -2...2), random.float(in: -0.7...0.7)))
            try controller.update(deltaTime: 1 / 60)
            let transform = controller.cameraTransform
            let toTarget = controller.pose.target - transform.translation
            #expect(abs(simd_length(toTarget) - 6) < 1e-4)
            #expect(close(forward(transform), simd_normalize(toTarget)))
            #expect(abs(controller.goal.yaw) <= .pi + 1e-6)
        }
    }

    @Test
    func `Pitch and optional yaw limits clamp the goal`() throws {
        var controller = try orbit(pitch: 0)
        try controller.rotate(by: SIMD2(0, 10))
        #expect(controller.goal.pitch == Float.pi / 2)
        try controller.rotate(by: SIMD2(0, -20))
        #expect(controller.goal.pitch == -Float.pi / 2)

        let limits = OrbitCameraController.Limits(pitch: -0.2...0.6, yaw: -1...1)
        var limited = try orbit(yaw: 0, pitch: 0, limits: limits)
        try limited.rotate(by: SIMD2(5, 5))
        #expect(limited.goal.yaw == 1)
        #expect(limited.goal.pitch == 0.6)
        try limited.rotate(by: SIMD2(-5, -5))
        #expect(limited.goal.yaw == -1)
        #expect(limited.goal.pitch == -0.2)
    }

    @Test
    func `Panning moves the target in the view plane and the scene follows the drag`() throws {
        var controller = try orbit()
        let before = controller.cameraTransform
        let start = controller.goal.target
        let aspect: Float = 1.5
        #expect(close(normalized(start, camera: before, aspectRatio: aspect), SIMD2(0.5, 0.5)))

        // Drag 120 points right and 60 down in a 600-point-tall view.
        try controller.pan(by: SIMD2(120, 60), viewportHeight: 600)
        try controller.update(deltaTime: 0)
        let moved = controller.goal.target - start
        #expect(abs(simd_dot(moved, forward(before))) < 1e-5)
        #expect(close(forward(controller.cameraTransform), forward(before)))
        #expect(controller.goal.distance == 6)
        let followed = normalized(start, camera: controller.cameraTransform, aspectRatio: aspect)
        #expect(close(followed, SIMD2(0.5 + 120 / (600 * aspect), 0.5 + 60.0 / 600)))

        // A full-height drag moves the target by the visible height at the target.
        var vertical = try orbit()
        try vertical.pan(by: SIMD2(0, 600), viewportHeight: 600)
        let visibleHeight = 2 * 6 * tan(Self.fieldOfView / 2)
        #expect(abs(simd_length(vertical.goal.target - start) - visibleHeight) < 1e-4)
    }

    @Test
    func `Horizontal panning moves the target along the ground like map controls`() throws {
        var controller = try orbit(target: SIMD3(0, 1, 0), yaw: 0, pitch: 0.8, panMode: .horizontal)
        try controller.pan(by: SIMD2(0, 100), viewportHeight: 400)
        #expect(controller.goal.target.y == 1)
        #expect(controller.goal.target.x == 0)
        // Dragging down moves the target forward, away from the camera on +Z.
        #expect(controller.goal.target.z < 0)
        try controller.pan(by: SIMD2(100, 0), viewportHeight: 400)
        #expect(controller.goal.target.y == 1)
        #expect(controller.goal.target.x < 0)
    }

    @Test
    func `Dolly divides the distance or orthographic height within limits`() throws {
        let limits = OrbitCameraController.Limits(distance: 1...20, orthographicHalfHeight: 0.5...8)
        var controller = try orbit(limits: limits)
        try controller.dolly(by: 2)
        #expect(controller.goal.distance == 3)
        try controller.dolly(by: 100)
        #expect(controller.goal.distance == 1)
        try controller.dolly(by: 0.001)
        #expect(controller.goal.distance == 20)
        try controller.update(deltaTime: 0)
        #expect(abs(simd_length(controller.cameraTransform.translation - controller.goal.target) - 20) < 1e-4)

        var orthographic = try orbit(projection: .orthographic, limits: limits)
        try orthographic.dolly(by: 2)
        #expect(orthographic.goal.orthographicHalfHeight == 1.5)
        #expect(orthographic.goal.distance == 6)
        try orthographic.dolly(by: 100)
        #expect(orthographic.goal.orthographicHalfHeight == 0.5)

        for factor: Float in [0, -1, .infinity, .nan] {
            #expect(throws: OrbitCameraController.Failure.invalidFactor) { try controller.dolly(by: factor) }
        }
    }

    @Test
    func `Damping converges at a frame-rate independent pace`() throws {
        var controller = try orbit(dampingRate: 6)
        try controller.rotate(by: SIMD2(0.9, -0.2))
        try controller.dolly(by: 1.8)
        try controller.pan(by: SIMD2(-40, 25), viewportHeight: 300)
        #expect(!controller.isSettled)

        var once = controller
        try once.update(deltaTime: 1 / 60)
        var twice = controller
        try twice.update(deltaTime: 1 / 120)
        try twice.update(deltaTime: 1 / 120)
        #expect(close(once.pose.target, twice.pose.target, 1e-5))
        #expect(abs(once.pose.distance - twice.pose.distance) < 1e-5)
        #expect(abs(once.pose.yaw - twice.pose.yaw) < 1e-6)
        #expect(abs(once.pose.pitch - twice.pose.pitch) < 1e-6)

        // One 60 Hz step covers 1 − e^(−0.1) of every gap.
        let covered = (once.pose.distance - controller.pose.distance) / (controller.goal.distance - controller.pose.distance)
        #expect(abs(covered - (1 - exp(-0.1))) < 1e-4)

        var frames = 0
        while !controller.isSettled, frames < 600 {
            try controller.update(deltaTime: 1 / 60)
            frames += 1
        }
        #expect(controller.isSettled)
        #expect(controller.pose == controller.goal)
        #expect(frames > 30 && frames < 300)
    }

    @Test
    func `Zooming toward a point keeps world points under the cursor`() throws {
        let aspect: Float = 1.5
        let cursor = SIMD2<Float>(0.8, 0.3)
        let camera = try PerspectiveCamera(verticalFieldOfView: Self.fieldOfView, near: 0.01, far: 100)
        for factor: Float in [2.5, 0.6, 1000] {
            var controller = try orbit(limits: .init(distance: 0.5...50), dampingRate: 5)
            let before = controller.cameraTransform
            let ray = try camera.ray(at: cursor, aspectRatio: aspect, cameraTransform: before)
            let view = forward(before)
            let onTargetPlane = simd_dot(controller.goal.target - ray.origin, view) / simd_dot(ray.direction, view)
            let points = [ray.origin + ray.direction * onTargetPlane, ray.origin + ray.direction * onTargetPlane * 1.7]

            try controller.zoom(by: factor, toward: cursor, aspectRatio: aspect)
            let expected = min(max(6 / factor, 0.5), 50)
            #expect(abs(controller.goal.distance - expected) <= expected * 1e-6)
            // The damped camera moves along the cursor ray, so the points stay put throughout.
            for _ in 0..<3 {
                try controller.update(deltaTime: 1 / 30)
                for point in points {
                    #expect(close(normalized(point, camera: controller.cameraTransform, aspectRatio: aspect), cursor))
                }
            }
            controller.settle()
            #expect(close(forward(controller.cameraTransform), view))
            for point in points {
                #expect(close(normalized(point, camera: controller.cameraTransform, aspectRatio: aspect), cursor))
            }
        }
    }

    @Test
    func `Orthographic zoom keeps the point under the cursor`() throws {
        let aspect: Float = 0.75
        let cursor = SIMD2<Float>(0.2, 0.85)
        var controller = try orbit(projection: .orthographic)
        let before = controller.cameraTransform
        let screen = SIMD2(2 * cursor.x - 1, 1 - 2 * cursor.y)
        let points = [Float(-6), -2].map { depth in
            let local = SIMD4(screen.x * 3 * aspect, screen.y * 3, depth, 1)
            let world = before.matrix * local
            return SIMD3(world.x, world.y, world.z)
        }
        for point in points {
            #expect(close(normalized(point, camera: before, aspectRatio: aspect, orthographicHalfHeight: 3), cursor))
        }

        try controller.zoom(by: 2, toward: cursor, aspectRatio: aspect)
        controller.settle()
        #expect(controller.goal.orthographicHalfHeight == 1.5)
        #expect(controller.goal.distance == 6)
        for point in points {
            let projected = normalized(
                point, camera: controller.cameraTransform, aspectRatio: aspect, orthographicHalfHeight: 1.5
            )
            #expect(close(projected, cursor))
        }
    }

    @Test
    func `Framing fits the box's bounding sphere in the field of view`() throws {
        let box = try AxisAlignedBox(minimum: SIMD3(-1, -2, -3), maximum: SIMD3(3, 2, 1))
        let radius = Float(12).squareRoot()
        let corners = (0..<8).map { index in
            SIMD3(index & 1 == 0 ? box.minimum.x : box.maximum.x,
                  index & 2 == 0 ? box.minimum.y : box.maximum.y,
                  index & 4 == 0 ? box.minimum.z : box.maximum.z)
        }

        for aspect: Float in [1.6, 0.5] {
            var controller = try orbit(target: .zero, distance: 1)
            try controller.frame(box, aspectRatio: aspect)
            controller.settle()
            #expect(controller.goal.target == SIMD3(1, 0, -1))
            let halfVertical = Self.fieldOfView / 2
            let limiting = min(halfVertical, atan(tan(halfVertical) * aspect))
            // The sphere is tangent to the narrower pair of frustum planes.
            #expect(abs(controller.goal.distance * sin(limiting) - radius) < 1e-4)
            for corner in corners {
                let point = normalized(corner, camera: controller.cameraTransform, aspectRatio: aspect)
                #expect((0...1).contains(point.x) && (0...1).contains(point.y))
            }
        }
        var wide = try orbit(target: .zero, distance: 1)
        try wide.frame(box, aspectRatio: 1.6)
        #expect(abs(wide.goal.distance - 2 * radius) < 1e-4)

        var orthographic = try orbit(target: .zero, distance: 1, projection: .orthographic)
        try orthographic.frame(box, aspectRatio: 0.5)
        #expect(abs(orthographic.goal.orthographicHalfHeight - 2 * radius) < 1e-4)
        #expect(abs(orthographic.goal.distance - 2 * radius) < 1e-4)
    }

    @Test
    func `Orbit controller rejects invalid configuration and input`() throws {
        #expect(throws: OrbitCameraController.Failure.invalidFieldOfView) {
            try orbit(projection: .perspective(verticalFieldOfView: .pi))
        }
        #expect(throws: OrbitCameraController.Failure.invalidLimits) {
            try orbit(limits: .init(distance: 0...10))
        }
        #expect(throws: OrbitCameraController.Failure.invalidLimits) {
            try orbit(limits: .init(pitch: -2...0))
        }
        #expect(throws: OrbitCameraController.Failure.poseOutsideLimits) {
            try orbit(distance: 30, limits: .init(distance: 1...20))
        }
        #expect(throws: OrbitCameraController.Failure.invalidTarget) { try orbit(target: SIMD3(.nan, 0, 0)) }
        #expect(throws: OrbitCameraController.Failure.invalidDampingRate) { try orbit(dampingRate: 0) }

        var controller = try orbit()
        #expect(throws: OrbitCameraController.Failure.invalidDelta) { try controller.rotate(by: SIMD2(.nan, 0)) }
        #expect(throws: OrbitCameraController.Failure.invalidViewportHeight) {
            try controller.pan(by: SIMD2(1, 1), viewportHeight: 0)
        }
        #expect(throws: OrbitCameraController.Failure.invalidNormalizedPoint) {
            try controller.zoom(by: 2, toward: SIMD2(1.5, 0.5), aspectRatio: 1)
        }
        #expect(throws: OrbitCameraController.Failure.invalidAspectRatio) {
            try controller.zoom(by: 2, toward: SIMD2(0.5, 0.5), aspectRatio: 0)
        }
        #expect(throws: OrbitCameraController.Failure.invalidDeltaTime) { try controller.update(deltaTime: -1) }
        #expect(controller.goal == (try orbit()).goal)
    }

    @Test
    func `Fly camera moves along its view axes`() throws {
        var controller = try FlyCameraController(maximumSpeed: 2)
        try controller.update(deltaTime: 0.5, movement: SIMD3(0, 0, 1))
        #expect(controller.position.x == 0 && controller.position.y == 0)
        #expect(controller.position.z < 0)
        #expect(close(simd_normalize(controller.velocity), SIMD3(0, 0, -1)))
        #expect(close(forward(controller.cameraTransform), SIMD3(0, 0, -1)))

        var turned = try FlyCameraController(yaw: .pi / 2, maximumSpeed: 2)
        try turned.update(deltaTime: 0.5, movement: SIMD3(0, 0, 1))
        #expect(turned.position.x < 0)
        #expect(abs(turned.position.z) < 1e-6)
        #expect(close(forward(turned.cameraTransform), SIMD3(-1, 0, 0)))

        var strafing = try FlyCameraController(maximumSpeed: 2)
        try strafing.update(deltaTime: 0.5, movement: SIMD3(1, 1, 0))
        #expect(strafing.position.x > 0)
        #expect(abs(strafing.position.x - strafing.position.y) < 1e-6)
        #expect(strafing.position.z == 0)

        var climbing = try FlyCameraController(pitch: .pi / 4, maximumSpeed: 2)
        try climbing.update(deltaTime: 0.5, movement: SIMD3(0, 0, 1))
        #expect(climbing.position.y > 0)
        #expect(abs(climbing.position.y + climbing.position.z) < 1e-6)
        #expect(close(simd_normalize(climbing.position), forward(climbing.cameraTransform)))
    }

    @Test
    func `Fly camera velocity is frame-rate independent and coasts to rest`() throws {
        let start = try FlyCameraController(yaw: 0.3, pitch: -0.2, maximumSpeed: 3, accelerationRate: 4, dampingRate: 2)
        var once = start
        var twice = start
        let movement = SIMD3<Float>(0.4, 0, 1)
        try once.update(deltaTime: 1 / 60, movement: movement)
        try twice.update(deltaTime: 1 / 120, movement: movement)
        try twice.update(deltaTime: 1 / 120, movement: movement)
        #expect(close(once.position, twice.position, 1e-6))
        #expect(close(once.velocity, twice.velocity, 1e-5))

        var cruising = start
        for _ in 0..<300 { try cruising.update(deltaTime: 1 / 60, movement: SIMD3(0, 0, 4)) }
        // Input longer than one unit is shortened, so the speed stays at the maximum.
        #expect(abs(simd_length(cruising.velocity) - 3) < 1e-3)

        let coastStart = cruising.position
        var steps = 0
        while cruising.velocity != .zero, steps < 1000 {
            try cruising.update(deltaTime: 1 / 60)
            steps += 1
        }
        #expect(cruising.velocity == .zero)
        // Coasting from speed v at rate r travels v / r.
        #expect(abs(simd_length(cruising.position - coastStart) - 1.5) < 1e-3)
        let resting = cruising.position
        try cruising.update(deltaTime: 1)
        #expect(cruising.position == resting)
    }

    @Test
    func `Fly camera clamps pitch, wraps yaw and validates input`() throws {
        var controller = try FlyCameraController(maximumSpeed: 1, pitchLimits: -1...1)
        try controller.look(by: SIMD2(0, 3))
        #expect(controller.pitch == 1)
        try controller.look(by: SIMD2(0, -5))
        #expect(controller.pitch == -1)
        try controller.look(by: SIMD2(3 * .pi, 0))
        #expect(abs(abs(controller.yaw) - .pi) < 1e-5)

        #expect(throws: FlyCameraController.Failure.invalidDelta) { try controller.look(by: SIMD2(.infinity, 0)) }
        #expect(throws: FlyCameraController.Failure.invalidMovement) {
            try controller.update(deltaTime: 0.1, movement: SIMD3(.nan, 0, 0))
        }
        #expect(throws: FlyCameraController.Failure.invalidDeltaTime) { try controller.update(deltaTime: .nan) }
        #expect(throws: FlyCameraController.Failure.invalidSpeed) { try FlyCameraController(maximumSpeed: 0) }
        #expect(throws: FlyCameraController.Failure.invalidRate) {
            try FlyCameraController(maximumSpeed: 1, dampingRate: 0)
        }
        #expect(throws: FlyCameraController.Failure.invalidPitchLimits) {
            try FlyCameraController(maximumSpeed: 1, pitchLimits: -2...0)
        }
        #expect(throws: FlyCameraController.Failure.invalidAngle) {
            try FlyCameraController(pitch: 1.2, maximumSpeed: 1, pitchLimits: -1...1)
        }
    }

    @Test("A shared control event moves an enabled camera and leaves a disabled one")
    func enabledEvents() throws {
        var orbit = try OrbitCameraController(
            distance: 4, projection: .perspective(verticalFieldOfView: Self.fieldOfView), dampingRate: nil
        )
        let settled = orbit.pose
        orbit.isEnabled = false
        try orbit.handle(.init(.pointer(phase: .changed, delta: SIMD2(0.4, 0))))
        try orbit.update(deltaTime: 1)
        #expect(orbit.pose == settled)
        #expect(throws: OrbitCameraController.Failure.invalidDelta) {
            try orbit.handle(.init(.pointer(phase: .changed, delta: SIMD2(.nan, 0))))
        }
        orbit.isEnabled = true
        try orbit.handle(.init(.pointer(phase: .changed, delta: SIMD2(0.4, 0))))
        try orbit.update(deltaTime: 1)
        #expect(abs(orbit.pose.yaw - 0.4) < 1e-4)
        let dolly = orbit.pose.distance
        try orbit.handle(.init(.scroll(delta: SIMD2(0, log(Float(2))))))
        try orbit.update(deltaTime: 0)
        #expect(abs(orbit.pose.distance - dolly / 2) < 1e-4)

        var fly = try FlyCameraController(maximumSpeed: 2, accelerationRate: 100, dampingRate: 100)
        try fly.handle(.init(.movement(direction: SIMD3(0, 0, 1), deltaTime: 1)))
        let moved = fly.position
        #expect(simd_length(moved) > 0.5)
        fly.isEnabled = false
        try fly.handle(.init(.movement(direction: SIMD3(0, 0, 1), deltaTime: 1)))
        try fly.update(deltaTime: 1, movement: SIMD3(0, 0, 1))
        #expect(fly.position == moved)
        #expect(throws: FlyCameraController.Failure.invalidDeltaTime) {
            try fly.handle(.init(.movement(direction: .zero, deltaTime: -1)))
        }
    }

    @Test("A look goal, a timed move and frame padding place the camera")
    func timedLookAndPadding() throws {
        var controller = try orbit(target: .zero, distance: 4, yaw: 0, pitch: 0)
        try controller.look(at: SIMD3(1, 2, 3), from: SIMD3(1, 2, 8))
        try controller.update(deltaTime: 0)
        #expect(close(controller.cameraTransform.translation, SIMD3(1, 2, 8)))
        #expect(close(forward(controller.cameraTransform), SIMD3(0, 0, -1)))

        controller.transitionDuration = 1
        try controller.look(at: SIMD3(1, 2, 3), from: SIMD3(1, 2, 12))
        try controller.update(deltaTime: 0.5)
        #expect(abs(controller.pose.distance - 7) < 1e-3)
        try controller.update(deltaTime: 0.5)
        #expect(close(controller.cameraTransform.translation, SIMD3(1, 2, 12)))
        controller.transitionDuration = -1
        #expect(throws: OrbitCameraController.Failure.invalidTransitionDuration) {
            try controller.update(deltaTime: 0)
        }
        controller.transitionDuration = nil

        let box = try AxisAlignedBox(minimum: SIMD3(repeating: -1), maximum: SIMD3(repeating: 1))
        var tight = try orbit(target: .zero, distance: 1, yaw: 0, pitch: 0)
        var padded = tight
        try tight.frame(box, aspectRatio: 1)
        try padded.frame(box, aspectRatio: 1, padding: 0.5)
        #expect(abs(padded.goal.distance / tight.goal.distance - 2) < 1e-3)
        #expect(throws: OrbitCameraController.Failure.invalidPadding) {
            try tight.frame(box, aspectRatio: 1, padding: 1)
        }
    }

    @Test
    func `A target box and a mesh keep the orbit outside them`() throws {
        var bounded = try orbit(target: .zero, distance: 4, yaw: 0, pitch: 0)
        bounded.targetBoundary = try AxisAlignedBox(minimum: SIMD3(repeating: -1), maximum: SIMD3(repeating: 1))
        try bounded.pan(by: SIMD2(200, 0), viewportHeight: 100)
        try bounded.update(deltaTime: 0)
        #expect(abs(bounded.goal.target.x + 1) < 1e-4)
        #expect(abs(bounded.pose.target.x + 1) < 1e-4)

        var blocked = try orbit(target: .zero, distance: 2, yaw: 0, pitch: 0)
        blocked.colliders = [try MeshBVH(geometry: try MeshGeometry.plane(width: 4, height: 4))]
        blocked.collisionRadius = 0.5
        try blocked.dolly(by: 20)
        try blocked.update(deltaTime: 0)
        #expect(abs(blocked.goal.cameraPosition.z - 0.5) < 1e-3)
        #expect(abs(blocked.pose.cameraPosition.z - 0.5) < 1e-3)
        #expect(abs(blocked.goal.distance - 0.5) < 1e-3)
    }
}
