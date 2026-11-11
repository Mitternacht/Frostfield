import CoreText
import Frostfield
import Metal
import Testing
import simd

struct AxisViewHelperTests {
    @Test
    func `Layout applies inverse camera rotation and resolves overlapping hit discs`() throws {
        let identity = SceneTransform().rotation
        let origin = SIMD2(18, 29)
        let center = SIMD2<Float>(82, 93)
        let layout = try AxisViewHelper.Layout(cameraRotation: identity, origin: origin)
        #expect(layout.endpoints.map(\.axis) == [.negativeZ, .positiveX, .positiveY, .negativeX, .negativeY, .positiveZ])
        #expect(layout.hitTest(at: center) == .positiveZ)
        #expect(layout.hitTest(at: center + SIMD2(32, 0)) == .positiveX)
        #expect(layout.hitTest(at: center + SIMD2(-32, 0)) == .negativeX)
        #expect(layout.hitTest(at: center + SIMD2(0, -32)) == .positiveY)
        #expect(layout.hitTest(at: center + SIMD2(0, 32)) == .negativeY)
        #expect(layout.hitTest(at: center + SIMD2(39, 0)) == .positiveX)
        #expect(layout.hitTest(at: center + SIMD2(39, 1)) == nil)
        #expect(layout.hitTest(at: center + SIMD2(16, 0)) == nil)
        #expect(layout.hitTest(at: SIMD2(.infinity, 0)) == nil)

        let roll = simd_quatf(angle: .pi / 2, axis: SIMD3(0, 0, 1))
        let rolled = try AxisViewHelper.Layout(cameraRotation: roll, origin: origin)
        #expect(rolled.hitTest(at: center + SIMD2(0, 32)) == .positiveX)
        #expect(rolled.hitTest(at: center + SIMD2(32, 0)) == .positiveY)
        #expect(rolled.hitTest(at: center) == .positiveZ)
        let scaled = try AxisViewHelper.Layout(cameraRotation: simd_quatf(vector: roll.vector * 8), origin: origin)
        #expect(rolled == scaled)
    }

    @Test
    func `Layout validation catches invalid pixel bounds and rotation`() throws {
        #expect(throws: AxisViewHelper.Failure.rotation) {
            try AxisViewHelper.Layout(cameraRotation: simd_quatf(vector: .zero), origin: .zero)
        }
        #expect(throws: AxisViewHelper.Failure.rotation) {
            try AxisViewHelper.Layout(cameraRotation: simd_quatf(vector: SIMD4(.nan, 0, 0, 1)), origin: .zero)
        }
        #expect(throws: AxisViewHelper.Failure.region) {
            try AxisViewHelper.Layout(cameraRotation: SceneTransform().rotation, origin: SIMD2(-1, 0))
        }
        #expect(throws: AxisViewHelper.Failure.region) {
            try AxisViewHelper.Layout(cameraRotation: SceneTransform().rotation, origin: .zero, size: 0)
        }
        #expect(throws: AxisViewHelper.Failure.region) {
            try AxisViewHelper.Layout(cameraRotation: SceneTransform().rotation, origin: SIMD2(Int.max, 0))
        }
        #expect(throws: AxisViewHelper.Failure.markerRadius) {
            try AxisViewHelper.Layout(cameraRotation: SceneTransform().rotation, origin: .zero, markerRadius: .nan)
        }
        #expect(throws: AxisViewHelper.Failure.markerRadius) {
            try AxisViewHelper.Layout(cameraRotation: SceneTransform().rotation, origin: .zero, markerRadius: 33)
        }
    }

    @Test(arguments: AxisViewHelper.Axis.allCases)
    func `The six views align the camera while retaining orbit center, radius and zoom`(_ axis: AxisViewHelper.Axis) throws {
        let focus = SIMD3<Float>(4, -3, 8)
        var controller = try OrbitCameraController(
            target: focus, distance: 7, yaw: 0.4, pitch: -0.2, projection: .orthographic, orthographicHalfHeight: 2.5
        )
        try controller.orient(to: axis, animated: false)
        #expect(controller.isSettled)
        #expect(controller.pose.target == focus)
        #expect(controller.pose.distance == 7)
        #expect(controller.pose.orthographicHalfHeight == 2.5)
        #expect(simd_distance(controller.cameraTransform.translation, focus + axis.direction * 7) < 2e-6)
        #expect(simd_distance(controller.cameraTransform.rotation.act(SIMD3(0, 0, 1)), axis.direction) < 2e-6)
        let displayed = try AxisViewHelper.Layout(cameraRotation: controller.cameraTransform.rotation, origin: .zero)
        #expect(displayed.hitTest(at: SIMD2(64, 64)) == axis)
    }

    @Test
    func `Axis selection animates along the short yaw branch with a fixed presented orbit`() throws {
        var controller = try OrbitCameraController(
            target: SIMD3(1, 2, 3), distance: 8, yaw: -.pi + 0.1, pitch: 0.25,
            projection: .perspective(verticalFieldOfView: 1), dampingRate: 5
        )
        let starting = controller.pose
        try controller.pan(by: SIMD2(30, -20), viewportHeight: 128)
        try controller.dolly(by: 2)
        #expect(controller.goal.target != starting.target)
        #expect(controller.goal.distance != starting.distance)
        let oldPosition = controller.cameraTransform.translation
        let initialLayout = try AxisViewHelper.Layout(cameraRotation: controller.cameraTransform.rotation, origin: .zero)
        let negativeZ = try #require(initialLayout.endpoints.first { $0.axis == .negativeZ })
        let selection = try #require(initialLayout.hitTest(at: negativeZ.position))
        #expect(selection == .negativeZ)
        try controller.orient(to: selection)
        #expect(simd_distance(controller.cameraTransform.translation, oldPosition) < 2e-6)
        #expect(controller.goal.target == starting.target)
        #expect(controller.goal.distance == starting.distance)
        #expect(abs(controller.pose.yaw - controller.goal.yaw) < 0.101)
        #expect(!controller.isSettled)

        try controller.update(deltaTime: 0.1)
        #expect(!controller.isSettled)
        #expect(abs(controller.pose.yaw - controller.goal.yaw) < 0.1)
        #expect(controller.pose.target == starting.target)
        #expect(abs(simd_distance(controller.cameraTransform.translation, starting.target) - 8) < 1e-6)
        try controller.update(deltaTime: 5)
        #expect(controller.isSettled)
        #expect(simd_distance(controller.cameraTransform.translation, starting.target + SIMD3(0, 0, -8)) < 2e-6)
    }

    @Test
    func `Alignment finds an equivalent legal yaw and excludes prohibited axes transactionally`() throws {
        var controller = try OrbitCameraController(
            distance: 3, projection: .orthographic,
            limits: .init(pitch: -0.3...0.3, yaw: 0...2 * .pi)
        )
        try controller.orient(to: .negativeX)
        #expect(abs(controller.pose.yaw - 1.5 * .pi) < 1e-6)
        #expect(controller.isSettled)
        let saved = controller
        #expect(throws: OrbitCameraController.Failure.poseOutsideLimits) { try controller.orient(to: .positiveY) }
        #expect(controller == saved)

        var restricted = try OrbitCameraController(
            distance: 3, yaw: -.pi, projection: .orthographic,
            limits: .init(yaw: -.pi ... -.pi)
        )
        try restricted.orient(to: .negativeZ)
        #expect(restricted.pose.yaw == -.pi)
        #expect(throws: OrbitCameraController.Failure.poseOutsideLimits) { try restricted.orient(to: .positiveX) }
    }

    @Test
    func `Default constant speed completes a quarter turn in a quarter second`() throws {
        let focus = SIMD3<Float>(3, 1, 2)
        var controller = try OrbitCameraController(
            target: focus, distance: 8, projection: .orthographic, orthographicHalfHeight: 2
        )
        try controller.orient(to: .positiveX, transition: .constantSpeed())
        #expect(!controller.isSettled)
        #expect(controller.cameraTransform.translation == focus + SIMD3(0, 0, 8))
        try controller.update(deltaTime: 0.125)
        #expect(!controller.isSettled)
        #expect(abs(controller.pose.yaw - .pi / 4) < 1e-6)
        let expected = focus + SIMD3<Float>(1, 0, 1) * (8 / Float(2).squareRoot())
        #expect(simd_distance(controller.cameraTransform.translation, expected) < 1e-6)
        try controller.update(deltaTime: 0.125)
        #expect(controller.isSettled)
        #expect(simd_distance(controller.cameraTransform.translation, focus + SIMD3(8, 0, 0)) < 1e-6)
        #expect(controller.pose.target == focus)
        #expect(controller.pose.distance == 8)
        #expect(controller.pose.orthographicHalfHeight == 2)
    }

    @Test
    func `Compound constant speed respects orbit limits and subdivided time`() throws {
        var controller = try OrbitCameraController(
            distance: 3, yaw: -0.5, pitch: 0.6, projection: .orthographic,
            limits: .init(pitch: 0...0.6, yaw: -0.5...1.6), dampingRate: 5
        )
        let initial = controller.cameraTransform.rotation
        let speed: Float = 1.25
        try controller.orient(to: .positiveX, transition: .constantSpeed(radiansPerSecond: speed))
        var split = controller
        try controller.update(deltaTime: 0.0625)
        try split.update(deltaTime: 0.03125)
        try split.update(deltaTime: 0.03125)
        #expect(controller == split)
        let mid = controller.cameraTransform.rotation
        let a = simd_normalize(SIMD4<Double>(initial.vector))
        let b = simd_normalize(SIMD4<Double>(mid.vector))
        let angularStep = 2 * acos(min(1, abs(simd_dot(a, b))))
        #expect(abs(angularStep / 0.0625 - Double(speed)) < 0.001)
        #expect(controller.limits.yaw?.contains(controller.pose.yaw) == true)
        #expect(controller.limits.pitch.contains(controller.pose.pitch))
        #expect(abs(mid.act(SIMD3(1, 0, 0)).y) < 1e-6)
        #expect(abs(simd_length(controller.cameraTransform.translation) - 3) < 1e-6)

        let saved = controller
        #expect(throws: OrbitCameraController.Failure.invalidAngularSpeed) {
            try controller.orient(to: .positiveZ, transition: .constantSpeed(radiansPerSecond: 0))
        }
        #expect(controller == saved)
        try controller.update(deltaTime: 2)
        #expect(controller.isSettled)
        #expect(simd_distance(controller.cameraTransform.translation, SIMD3(3, 0, 0)) < 1e-6)
    }

    @Test
    func `Failed input preserves a constant turn and successful input resumes normal timing`() throws {
        var controller = try OrbitCameraController(
            distance: 3, projection: .orthographic, orthographicHalfHeight: 1_000_000
        )
        try controller.orient(to: .negativeZ, transition: .constantSpeed())
        try controller.update(deltaTime: 0.125)
        let saved = controller
        #expect(throws: OrbitCameraController.Failure.unrepresentablePose) {
            try controller.zoom(by: 2, toward: SIMD2(1, 0.5), aspectRatio: .greatestFiniteMagnitude)
        }
        #expect(controller == saved)
        try controller.dolly(by: 2)
        try controller.update(deltaTime: 0)
        #expect(controller.isSettled)
        #expect(controller.pose.orthographicHalfHeight == 500_000)
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Metal overlay and hit test share visible ordering and preserve scene pixels`() throws {
        let fixture = try RasterLightingFixture(width: 176, height: 160)
        let overlay = try AxisViewHelper(device: fixture.device)
        let first = try fixture.colorTarget()
        let second = try fixture.colorTarget()
        let origin = SIMD2(26, 18)
        let identity = try AxisViewHelper.Layout(cameraRotation: SceneTransform().rotation, origin: origin)
        let side = try AxisViewHelper.Layout(
            cameraRotation: simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0)), origin: origin
        )
        let background = SIMD4<Float>(0.25, 0.5, 0.75, 1)
        try fixture.run { buffer in
            for (target, layout) in [(first, identity), (second, side)] {
                let pass = MTLRenderPassDescriptor()
                pass.colorAttachments[0].texture = target
                pass.colorAttachments[0].loadAction = .clear
                pass.colorAttachments[0].storeAction = .store
                pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.25, green: 0.5, blue: 0.75, alpha: 1)
                let encoder = try #require(buffer.makeRenderCommandEncoder(descriptor: pass))
                encoder.endEncoding()
                try overlay.encode(layout: layout, destination: target, commandBuffer: buffer)
            }
        }
        let firstPixels = try fixture.pixels(of: first)
        let secondPixels = try fixture.pixels(of: second)
        func pixel(_ array: [SIMD4<Float>], _ x: Int, _ y: Int) -> SIMD4<Float> { array[y * 176 + x] }

        #expect(identity.hitTest(at: SIMD2(90.5, 82.5)) == .positiveZ)
        #expect(side.hitTest(at: SIMD2(90.5, 82.5)) == .positiveX)
        #expect(simd_distance(pixel(firstPixels, 90, 82), LineGeometry.helperColor(0x4488ff)) < 0.002)
        #expect(simd_distance(pixel(secondPixels, 90, 82), LineGeometry.helperColor(0xff4466)) < 0.002)
        #expect(simd_distance(pixel(firstPixels, 122, 82), LineGeometry.helperColor(0xff4466)) < 0.002)
        #expect(simd_distance(pixel(firstPixels, 90, 50), LineGeometry.helperColor(0x88ff44)) < 0.002)
        #expect(simd_distance(pixel(firstPixels, 58, 82), SIMD4(0.2, 0.4, 0.6, 1)) < 0.002)
        #expect(pixel(firstPixels, 106, 82).x > 0.7)
        #expect(identity.hitTest(at: SIMD2(106.5, 82.5)) == nil)
        #expect(pixel(firstPixels, 26, 18) == background)
        #expect(pixel(firstPixels, 15, 82) == background)
        #expect(pixel(secondPixels, 175, 159) == background)
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Metal overlay encodes sRGB and rejects incompatible destinations`() throws {
        let fixture = try RasterLightingFixture(width: 128, height: 128)
        let overlay = try AxisViewHelper(device: fixture.device, pixelFormat: .rgba8Unorm_srgb)
        let target = try fixture.texture(.rgba8Unorm_srgb, usage: .renderTarget, storage: .shared)
        let mismatched = try fixture.colorTarget()
        let layout = try AxisViewHelper.Layout(cameraRotation: SceneTransform().rotation, origin: .zero)
        let outside = try AxisViewHelper.Layout(cameraRotation: SceneTransform().rotation, origin: SIMD2(1, 0))
        try fixture.run { buffer in
            #expect(throws: AxisViewHelper.Failure.destination) {
                try overlay.encode(layout: layout, destination: mismatched, commandBuffer: buffer)
            }
            #expect(throws: AxisViewHelper.Failure.region) {
                try overlay.encode(layout: outside, destination: target, commandBuffer: buffer)
            }
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = target
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            let encoder = try #require(buffer.makeRenderCommandEncoder(descriptor: pass))
            encoder.endEncoding()
            try overlay.encode(layout: layout, destination: target, commandBuffer: buffer)
        }
        var bytes = [UInt8](repeating: 0, count: 4)
        bytes.withUnsafeMutableBytes {
            target.getBytes($0.baseAddress!, bytesPerRow: 4, from: MTLRegionMake2D(64, 64, 1, 1), mipmapLevel: 0)
        }
        #expect(bytes == [0x44, 0x88, 0xff, 255])
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Core Text labels honor font, color, depth order and the same marker radius as picking`() throws {
        let fixture = try RasterLightingFixture(width: 144, height: 132)
        let overlay = try AxisViewHelper(device: fixture.device)
        let targets = try (0..<3).map { _ in try fixture.colorTarget() }
        let layout = try AxisViewHelper.Layout(
            cameraRotation: SceneTransform().rotation, origin: SIMD2(8, 2), markerRadius: 14
        )
        let font = CTFontCreateWithName("Helvetica" as CFString, 18, nil)
        let front = try AxisViewHelper.Label("F", device: fixture.device, font: font, color: SIMD4(1, 1, 1, 1))
        let side = try AxisViewHelper.Label("X", device: fixture.device, color: SIMD4(1, 1, 0, 1))
        let rear = try AxisViewHelper.Label("W", device: fixture.device, font: font, color: SIMD4(1, 0, 1, 1))
        #expect(throws: AxisViewHelper.Failure.labelColor) {
            try AxisViewHelper.Label("X", device: fixture.device, color: SIMD4(0, 0, 0, .nan))
        }
        let labels: [[AxisViewHelper.Axis: AxisViewHelper.Label]] = [
            [:], [.positiveZ: front, .negativeX: side], [.positiveZ: front, .negativeX: side, .negativeZ: rear],
        ]
        try fixture.run { buffer in
            for (target, labels) in zip(targets, labels) {
                let pass = MTLRenderPassDescriptor()
                pass.colorAttachments[0].texture = target
                pass.colorAttachments[0].loadAction = .clear
                pass.colorAttachments[0].storeAction = .store
                let encoder = try #require(buffer.makeRenderCommandEncoder(descriptor: pass))
                encoder.endEncoding()
                try overlay.encode(layout: layout, labels: labels, destination: target, commandBuffer: buffer)
            }
        }
        let base = try fixture.pixels(of: targets[0])
        let ink = try fixture.pixels(of: targets[1])
        #expect(try fixture.pixels(of: targets[2]) == ink)
        func changedPixels(x: Range<Int>, y: Range<Int>, channel: Int) -> [SIMD2<Int>] {
            y.flatMap { row in
                x.compactMap { column in
                    let index = row * 144 + column
                    return ink[index][channel] - base[index][channel] > 0.06 ? SIMD2(column, row) : nil
                }
            }
        }
        let upperF = changedPixels(x: 64..<80, y: 59..<64, channel: 0)
        let lowerF = changedPixels(x: 64..<80, y: 68..<73, channel: 0)
        let sideX = changedPixels(x: 32..<48, y: 58..<74, channel: 1)
        #expect(upperF.count > 8)
        #expect(lowerF.count > 4)
        let upperRight = try #require(upperF.map(\.x).max())
        let lowerRight = try #require(lowerF.map(\.x).max())
        #expect(upperRight - lowerRight > 2)
        #expect(sideX.count > 8)
        #expect(layout.hitTest(at: SIMD2(72, 66)) == .positiveZ)
        #expect(layout.hitTest(at: SIMD2(104, 79)) == .positiveX)
        #expect(layout.hitTest(at: SIMD2(104, 81)) == nil)
    }
}
