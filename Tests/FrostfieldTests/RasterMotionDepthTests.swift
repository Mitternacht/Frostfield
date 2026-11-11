import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct RasterMotionDepthTests {
    @Test("A translated PBR floor records motion across both visible test patches",
          arguments: [MeshDepthMode.forward, .reversed])
    func translatedFloor(mode: MeshDepthMode) throws {
        let gpu = try RasterLightingFixture(width: 128, height: 128)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 35)
        let pose = SceneTransform(translation: SIMD3(0, 1.5, 0), rotation: simd_quatf(angle: -0.08, axis: SIMD3(1, 0, 0)))
        let floorRotation = simd_quatf(angle: -.pi / 2, axis: SIMD3<Float>(1, 0, 0))
        func mesh(_ geometry: MeshGeometry, color: SIMD4<Float>, emission: SIMD3<Float> = .zero) throws -> MeshRenderer {
            let material = try PhysicallyBasedMaterial(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0,
                baseColorFactor: color, emissiveFactor: emission, emissiveStrength: 3
            )
            return try MeshRenderer(
                device: gpu.device, colorPixelFormat: .rgba16Float, geometry: geometry,
                material: .physicallyBased(material), depthMode: mode
            )
        }
        let floor = SceneNode(transform: .init(translation: SIMD3(0, 0, -4.5), rotation: floorRotation),
                              mesh: try mesh(.plane(width: 4, height: 7), color: SIMD4(0.4, 0.8, 0.6, 1)))
        func wall(emission: SIMD3<Float>) throws -> SceneNode {
            .init(transform: .init(translation: SIMD3(0, 0.9, -5)),
                  mesh: try mesh(.plane(width: 3, height: 1.8), color: SIMD4(1, 0, 0, 1), emission: emission))
        }
        let receiver = SIMD3<Float>(-0.55, 0, -4.05)
        let hidden = SIMD3<Float>(0.72, 0, -4.05)
        let blocker = pose.translation + (hidden - pose.translation) * 0.78
        let foreground = SceneNode(transform: .init(
            translation: blocker, rotation: floorRotation, scale: SIMD3(0.19, 0.08, 1)
        ), mesh: floor.mesh)
        var before = SceneGraph(roots: [floor, try wall(emission: SIMD3(1, 0.04, 0.01)), foreground],
                                camera: camera, cameraTransform: pose)
        before.projectionJitter = SIMD2(0.125, -0.375)
        let displacement = SIMD3<Float>(0.48, 0.31, 0.20)
        var moved = floor
        moved.transform.translation += displacement
        var after = SceneGraph(roots: [moved, try wall(emission: .zero)], camera: camera, cameraTransform: pose)
        after.projectionJitter = SIMD2(-0.375, 0.25)

        let beauty = try gpu.colorTarget(), depth = try gpu.depthTarget(), vectors = try gpu.colorTarget()
        let producer = try RasterMotionVectors(queue: gpu.queue)
        func render(_ graph: SceneGraph, previous: RasterMotionVectors.History? = nil) throws -> RasterMotionVectors.History {
            let recording = producer.recording(to: vectors, previous: previous)
            try gpu.run { command in
                try graph.encode(pass: gpu.pass(color: beauty, depth: depth, depthMode: mode),
                                 commandBuffer: command, depthMode: mode, motionVectors: recording)
            }
            return try #require(recording.history)
        }

        let previous = try render(before)
        _ = try render(after, previous: previous)
        let samples = try gpu.pixels(of: vectors)
        let colors = try gpu.pixels(of: beauty)
        let view = simd_inverse(pose.matrix)
        let projection = try camera.projection(aspectRatio: 1, depthMode: mode)
        let current = TemporalAntiAliasing.jitteredClipFromWorld(projection, jitter: after.projectionJitter, width: 128, height: 128)
        for point in [receiver, hidden] {
            let clip = current * view * SIMD4(point + displacement, 1)
            let x = Int((0.5 + 0.5 * clip.x / clip.w) * 128)
            let y = Int((0.5 - 0.5 * clip.y / clip.w) * 128)
            let motion = samples[y * 128 + x]
            let color = colors[y * 128 + x]
            let prior = projection * view * SIMD4(point, 1)
            #expect(color == SIMD4(0, 0, 0, 1), "Current beauty at \(x), \(y) = \(color)")
            #expect(motion.w == 1, "Current motion at \(x), \(y) = \(motion)")
            #expect(abs(motion.z - prior.z / prior.w) < 0.001, "Prior raw depth at \(x), \(y)")
        }
    }
}
