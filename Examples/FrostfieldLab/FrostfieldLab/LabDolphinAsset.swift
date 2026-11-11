import Foundation
import Frostfield
import Metal
import simd

/// The bundled Cyber Ocean dolphin and its imported swimming animation.
struct LabDolphinAsset {
    let sourceNodeIndex: Int
    let animationSourceIndex: Int
    let duration: Float
    let bindGeometry: MeshGeometry
    /// Places the bind mesh near the swimming pose for approximate surface emission.
    let bindSurfaceWorldTransform: simd_float4x4

    private let prepared: PreparedGLTFScene
    private let framing: SceneTransform
    private let framingID = UUID()

    init(device: any MTLDevice, pixelFormat: MTLPixelFormat) throws {
        guard let url = Bundle.main.url(forResource: "Dolphin", withExtension: "glb") else {
            throw Failure.missingResource
        }
        let source = try GLTFMeshLoader.loadScene(
            contentsOf: url, includeMaterials: true, includeAnimations: true, includeSkins: true
        )
        guard let node = source.nodes.first(where: { $0.meshIndex != nil && $0.skinIndex != nil }),
              let clip = source.animations.first, clip.duration > 0 else {
            throw Failure.missingAnimation
        }
        let material = try PhysicallyBasedMaterial(
            metallic: 0.38, roughness: 0.34, directionToLight: SIMD3(-0.42, 0.8, 0.9),
            lightIntensity: 2.5, ambientIntensity: 0.27,
            baseColorFactor: SIMD4(0.27, 0.67, 0.96, 1),
            emissiveFactor: SIMD3(0.008, 0.02, 0.04), alphaMode: .opaque
        )
        let prepared = try PreparedGLTFScene(
            device: device, colorPixelFormat: pixelFormat, scene: source, defaultMaterial: .physicallyBased(material)
        )
        let inspectionCamera = try PerspectiveCamera(verticalFieldOfView: .pi / 4, near: 0.1, far: 20)
        let reference = try prepared.graph(camera: inspectionCamera, animationIndex: clip.sourceIndex)
        guard let draw = try reference.resolve().first(where: {
            $0.mesh != nil && prepared.sourceNodeIndicesByID[$0.id] == node.sourceIndex
        }), let mesh = draw.mesh, let skin = mesh.skin, let palette = draw.jointPalette else {
            throw Failure.missingSurface
        }

        // The source swims along +Z. Turn it to face mostly +X with the nose
        // slightly toward the camera, then fit the first sampled pose.
        let rotation = simd_quatf(angle: .pi * 0.43, axis: SIMD3(0, 1, 0))
        let referenceGeometry = try skin.deform(mesh.geometry, using: palette)
        let posedBounds = Self.bounds(referenceGeometry, world: draw.worldTransform, rotation: rotation)
        let width = posedBounds.maximum.x - posedBounds.minimum.x
        guard width.isFinite, width > 0 else { throw Failure.missingSurface }
        let scale = 3.25 / width
        let posedCenter = (posedBounds.minimum + posedBounds.maximum) * 0.5
        let bindBounds = Self.bounds(mesh.geometry, world: draw.worldTransform, rotation: rotation)
        let bindCenter = (bindBounds.minimum + bindBounds.maximum) * 0.5
        let bindFraming = SceneTransform(
            translation: -bindCenter * scale, rotation: rotation, scale: SIMD3(repeating: scale)
        )

        sourceNodeIndex = node.sourceIndex
        animationSourceIndex = clip.sourceIndex
        duration = clip.duration
        bindGeometry = mesh.geometry
        bindSurfaceWorldTransform = bindFraming.matrix * draw.worldTransform
        self.prepared = prepared
        framing = SceneTransform(
            translation: -posedCenter * scale, rotation: rotation, scale: SIMD3(repeating: scale)
        )
    }

    func graph(camera: PerspectiveCamera, cameraTransform: SceneTransform, time: Float) throws -> SceneGraph {
        var graph = try prepared.graph(
            camera: camera, cameraTransform: cameraTransform, animationIndex: animationSourceIndex,
            time: time, looping: true
        )
        // The normalization is a common parent of both the joints and their
        // draw nodes, so the prepared draw-local joint palettes remain valid.
        graph.roots = [SceneNode(id: framingID, transform: framing, children: graph.roots)]
        return graph
    }

    private static func bounds(
        _ geometry: MeshGeometry, world: simd_float4x4, rotation: simd_quatf
    ) -> (minimum: SIMD3<Float>, maximum: SIMD3<Float>) {
        var minimum = SIMD3<Float>(repeating: .infinity)
        var maximum = SIMD3<Float>(repeating: -.infinity)
        for vertex in geometry.vertices {
            let worldPoint = world * SIMD4(vertex.position, 1)
            let point = rotation.act(SIMD3(worldPoint.x, worldPoint.y, worldPoint.z))
            minimum = simd_min(minimum, point)
            maximum = simd_max(maximum, point)
        }
        return (minimum, maximum)
    }

    private enum Failure: LocalizedError {
        case missingResource
        case missingAnimation
        case missingSurface

        var errorDescription: String? {
            switch self {
            case .missingResource: "The bundled Dolphin.glb resource could not be found."
            case .missingAnimation: "The bundled dolphin is missing its swimming animation."
            case .missingSurface: "The bundled dolphin is missing a usable skinned mesh."
            }
        }
    }
}
