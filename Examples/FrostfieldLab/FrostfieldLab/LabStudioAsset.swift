import Foundation
import Frostfield
import Metal
import simd

/// A studio subject with authored materials and a floor-aligned pose.
struct LabStudioAsset {
    /// The lens faces +Z at zero yaw. This angle gives the default three-quarter view.
    static let suggestedAngle: Float = 0.28
    let dimensions: SIMD3<Float>
    let requiresTransmissionBackdrop: Bool

    private let roots: [SceneNode]
    private let origin: SIMD3<Float>
    private let scale: Float
    private let placementID = UUID()

    init(
        device: any MTLDevice, pixelFormat: MTLPixelFormat,
        directionToLight: SIMD3<Float> = SIMD3(-0.6, 0.9, 0.7), lightIntensity: Float = 1,
        ambientIntensity: Float = 0.12, receivesShadows: Bool = false,
        environment: PhysicallyBasedMaterial.Environment? = nil, sourceURL: URL? = nil
    ) throws {
        guard let url = sourceURL ?? Bundle.main.url(forResource: "StudioCamera", withExtension: "glb") else {
            throw Failure.missingResource
        }
        let source = try GLTFMeshLoader.loadScene(contentsOf: url, includeMaterials: true, includeLights: true)
        let prepared = try PreparedGLTFScene(
            device: device, colorPixelFormat: pixelFormat, scene: source,
            directionToLight: directionToLight, lightIntensity: lightIntensity,
            ambientIntensity: ambientIntensity, receivesShadows: receivesShadows, environment: environment
        )
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 4, near: 0.1, far: 20)
        try self.init(reference: prepared.graph(camera: camera))
    }

    init(reference: SceneGraph) throws {
        var needsBackdrop = false
        var minimum = SIMD3<Float>(repeating: .infinity)
        var maximum = SIMD3<Float>(repeating: -.infinity)
        for node in try reference.resolve() {
            guard let mesh = node.mesh else { continue }
            needsBackdrop = needsBackdrop || mesh.requiresTransmissionBackdrop
            for vertex in mesh.geometry.vertices {
                let world = node.worldTransform * SIMD4(vertex.position, 1)
                let point = SIMD3(world.x, world.y, world.z)
                minimum = simd_min(minimum, point)
                maximum = simd_max(maximum, point)
            }
        }
        let size = maximum - minimum
        guard size.x.isFinite, size.y.isFinite, size.z.isFinite, size.y > 0 else {
            throw Failure.missingGeometry
        }

        let scale = 2.7 / size.y
        self.scale = scale
        requiresTransmissionBackdrop = needsBackdrop
        dimensions = size * scale
        origin = SIMD3((minimum.x + maximum.x) * 0.5, minimum.y, (minimum.z + maximum.z) * 0.5)
        roots = reference.roots
    }

    func graph(
        camera: PerspectiveCamera, cameraTransform: SceneTransform, angle: Float = Self.suggestedAngle
    ) throws -> SceneGraph {
        guard angle.isFinite else { throw Failure.invalidAngle }
        let rotation = simd_quatf(angle: angle, axis: SIMD3(0, 1, 0))
        let placement = SceneTransform(
            translation: -rotation.act(origin * scale), rotation: rotation, scale: SIMD3(repeating: scale)
        )
        return SceneGraph(
            roots: [SceneNode(id: placementID, transform: placement, children: roots)],
            camera: camera, cameraTransform: cameraTransform
        )
    }

    private enum Failure: LocalizedError {
        case missingResource
        case missingGeometry
        case invalidAngle

        var errorDescription: String? {
            switch self {
            case .missingResource: "The bundled StudioCamera.glb resource could not be found."
            case .missingGeometry: "The studio camera is missing usable geometry."
            case .invalidAngle: "The studio camera's placement angle must be finite."
            }
        }
    }
}
