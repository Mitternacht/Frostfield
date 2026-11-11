import Metal
import simd

/// An authored HDR room fixture for image-based lighting.
/// The room layout follows Three.js r187 `RoomEnvironment`.
public enum RoomEnvironment {
    /// Makes the capture scene with an inward-facing room, one point light,
    /// six white boxes and six emissive panels. Meshes use linear RGBA16Float
    /// and forward depth, ready to render through `CubeCamera`.
    public static func makeScene(device: any MTLDevice) throws -> SceneGraph {
        let cube = MeshGeometry.cube()
        var inward = cube.indices
        for start in stride(from: 0, to: inward.count, by: 3) {
            inward.swapAt(start + 1, start + 2)
        }
        let room = try MeshGeometry(
            vertices: cube.vertices.map { MeshVertex(position: $0.position, normal: -$0.normal) }, indices: inward
        )
        let matte = try PhysicallyBasedMaterial(roughness: 1, lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque)
        let roomMesh = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: room, material: .physicallyBased(matte))
        let boxMesh = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: cube, material: .physicallyBased(matte))
        let key = try PunctualLight(kind: .point(range: 28), intensity: 900)
        var children = [
            SceneNode(transform: .init(translation: SIMD3(0.418, 16.199, 0.300)), light: key, name: "Key point light"),
            SceneNode(transform: .init(translation: SIMD3(-0.757, 13.219, 0.717), scale: SIMD3(31.713, 28.305, 28.591)),
                      mesh: roomMesh, name: "Room shell")
        ]
        for (index, box) in boxes.enumerated() {
            let rotation = simd_quatf(angle: box.yaw, axis: SIMD3(0, 1, 0))
            children.append(SceneNode(
                transform: .init(translation: box.position, rotation: rotation, scale: box.size),
                mesh: boxMesh, name: "Box \(index + 1)"
            ))
        }
        for (index, panel) in panels.enumerated() {
            let emission = try PhysicallyBasedMaterial(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
                emissiveFactor: SIMD3(repeating: 1), alphaMode: .opaque, emissiveStrength: panel.radiance,
                specular: .init(factor: 0)
            )
            let mesh = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: cube, material: .physicallyBased(emission))
            children.append(SceneNode(
                transform: .init(translation: panel.position, scale: panel.size), mesh: mesh, name: "Emitter \(index + 1)"
            ))
        }
        let root = SceneNode(transform: .init(translation: SIMD3(0, -3.5, 0)), children: children, name: "RoomEnvironment")
        return SceneGraph(roots: [root], camera: try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 100))
    }

    /// Captures the room at a world position and prepares reusable HDR lighting.
    /// The default captures at the origin with a 256-texel cube face. This waits
    /// for Metal capture and filtering, so prepare it before starting the frame loop.
    public static func prepare(
        queue: any MTLCommandQueue, at position: SIMD3<Float> = .zero,
        settings: EnvironmentLighting.Settings = .init(cubeSize: 256)
    ) throws -> EnvironmentLighting {
        guard settings.isValid else { throw EnvironmentLighting.Failure.settings }
        guard position.isFiniteVector else { throw CubeCamera.Failure.position }
        let scene = try makeScene(device: queue.device)
        let cube = try CubeCamera.makeTarget(device: queue.device, size: settings.cubeSize, mipmapped: false)
        let camera = try CubeCamera(queue: queue, target: cube, near: scene.camera.near, far: scene.camera.far)
        guard let command = queue.makeCommandBuffer() else { throw EnvironmentLighting.Failure.commandBuffer }
        command.label = "Frostfield room environment capture"
        try camera.encode(scene, at: position, commandBuffer: command)
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else { throw EnvironmentLighting.Failure.execution(command.error?.localizedDescription) }
        return try EnvironmentLighting(queue: queue, cube: cube, settings: settings)
    }

    // Authored transforms from Three.js r187 RoomEnvironment.js (MIT).
    // See LICENSES/Three-RoomEnvironment.txt.
    private static let boxes: [(position: SIMD3<Float>, yaw: Float, size: SIMD3<Float>)] = [
        (SIMD3(-10.906, 2.009, 1.846), -0.195, SIMD3(2.328, 7.905, 4.651)),
        (SIMD3(-5.607, -0.754, -0.758), 0.994, SIMD3(1.970, 1.534, 3.955)),
        (SIMD3(6.167, 0.857, 7.803), 0.561, SIMD3(3.927, 6.285, 3.687)),
        (SIMD3(-2.017, 0.018, 6.124), 0.333, SIMD3(2.002, 4.566, 2.064)),
        (SIMD3(2.291, -0.756, -2.621), -0.286, SIMD3(1.546, 1.552, 1.496)),
        (SIMD3(-2.193, -0.369, -5.547), 0.516, SIMD3(3.875, 3.487, 2.986))
    ]

    private static let panels: [(position: SIMD3<Float>, size: SIMD3<Float>, radiance: Float)] = [
        (SIMD3(-16.116, 14.370, 8.208), SIMD3(0.1, 2.428, 2.739), 50),
        (SIMD3(-16.109, 18.021, -8.207), SIMD3(0.1, 2.425, 2.751), 50),
        (SIMD3(14.904, 12.198, -1.832), SIMD3(0.15, 4.265, 6.331), 17),
        (SIMD3(-0.462, 8.890, 14.520), SIMD3(4.38, 5.441, 0.088), 43),
        (SIMD3(3.235, 11.486, -12.541), SIMD3(2.5, 2.0, 0.1), 20),
        (SIMD3(0, 20, 0), SIMD3(1, 0.1, 1), 100)
    ]
}
