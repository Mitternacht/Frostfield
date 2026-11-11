import Foundation
import Frostfield
import MetalKit
import simd

final class LabGraphFixture {
    enum Mode {
        case cubes, obj, gltf, physicallyBased, shadows, physicallyBasedShadows

        var animates: Bool { self == .obj || self == .gltf || self == .physicallyBased }
        var usesShadows: Bool { self == .shadows || self == .physicallyBasedShadows }

        static func requested(_ arguments: [String]) -> Mode? {
            if arguments.contains("--pbr-shadow-scene") { return .physicallyBasedShadows }
            if arguments.contains("--shadow-scene") { return .shadows }
            if arguments.contains("--pbr-scene") { return .physicallyBased }
            if arguments.contains("--gltf-scene") { return .gltf }
            if arguments.contains("--asset-scene") { return .obj }
            if arguments.contains("--scene-graph") { return .cubes }
            return nil
        }
    }

    private let queue: any MTLCommandQueue
    private let mesh: MeshRenderer
    private let importedScene: GLTFScene?
    private let importedTint: SIMD4<Float>
    private let floor: MeshRenderer?
    private let shadowMap: DirectionalShadowMap?
    private let mode: Mode
    private let floorID = UUID()
    private let camera: PerspectiveCamera
    private let groupID = UUID()
    private let subgroupID = UUID()
    private let leftID = UUID()
    private let centerID = UUID()
    private let rightID = UUID()
    private var selected: UUID?
    private var cachedGraph: (aspect: Float, graph: SceneGraph)?
    private var submittedGraph: (aspect: Float, graph: SceneGraph)?
    private var displayedGraph: (aspect: Float, graph: SceneGraph)?
    private var animation: SceneAnimation?

    init(device: any MTLDevice, pixelFormat: MTLPixelFormat, mode: Mode) throws {
        guard let queue = device.makeCommandQueue() else { throw MeshRenderer.Failure.resource("command queue") }
        self.queue = queue
        self.mode = mode
        var importedScene: GLTFScene?
        var importedTint = SIMD4<Float>(repeating: 1)
        if mode == .gltf {
            guard let data = Data(base64Encoded: Self.tetrahedronGLB) else {
                throw MeshRenderer.Failure.resource("fixture glTF")
            }
            let scene = try GLTFMeshLoader.loadScene(data: data, includeMaterials: true)
            guard let assetMesh = scene.meshes.first, let primitive = assetMesh.primitives.first,
                  let index = assetMesh.primitiveMaterialIndices.first.flatMap({ $0 }),
                  let source = scene.materials.first(where: { $0.sourceIndex == index }),
                  let image = source.baseColorImage else {
                throw MeshRenderer.Failure.resource("fixture glTF material and geometry")
            }
            let texture = try TextureLoader(device: device).load(data: image.data)
            let material = try PhysicallyBasedMaterial(
                baseColorTexture: texture, metallic: source.metallicFactor, roughness: source.roughnessFactor,
                directionToLight: SIMD3(-0.35, 0.75, 1), lightIntensity: 2.7, ambientIntensity: 0.48
            )
            mesh = try MeshRenderer(
                device: device, colorPixelFormat: pixelFormat, geometry: primitive, material: .physicallyBased(material)
            )
            importedScene = scene
            importedTint = source.baseColorFactor
        } else if mode.animates {
            guard let image = Data(base64Encoded: Self.checkerPNG) else {
                throw MeshRenderer.Failure.resource("fixture texture")
            }
            let texture = try TextureLoader(device: device).load(data: image)
            let geometry = try OBJMeshLoader.load(Self.octahedron)
            let material: MeshMaterial
            if mode == .physicallyBased {
                material = .physicallyBased(try PhysicallyBasedMaterial(
                    baseColorTexture: texture, metallic: 0.35, roughness: 0.32,
                    directionToLight: SIMD3(-0.45, 1, 0.75), lightIntensity: 2.4
                ))
            } else {
                material = .textured(texture)
            }
            mesh = try MeshRenderer(device: device, colorPixelFormat: pixelFormat, geometry: geometry, material: material)
        } else if mode == .physicallyBasedShadows {
            let material = try PhysicallyBasedMaterial(metallic: 0.35, roughness: 0.26, receivesShadows: true)
            mesh = try MeshRenderer(
                device: device, colorPixelFormat: pixelFormat, geometry: .uvSphere(radius: 0.72),
                material: .physicallyBased(material)
            )
        } else {
            mesh = try MeshRenderer(
                device: device, colorPixelFormat: pixelFormat, geometry: .cube(),
                material: mode == .shadows ? .shadowedLambert : .unlit
            )
        }
        self.importedScene = importedScene
        self.importedTint = importedTint
        if mode.usesShadows {
            let geometry = try MeshGeometry.plane(width: 6, height: 6)
            floor = try MeshRenderer(device: device, colorPixelFormat: pixelFormat, geometry: geometry, material: .shadowedLambert)
            shadowMap = try DirectionalShadowMap(
                queue: queue, resolution: 1024, directionToLight: SIMD3(-0.65, 1, 0.55), halfExtent: 5
            )
        } else {
            floor = nil
            shadowMap = nil
        }
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 4, near: 0.1, far: 100)
        if mode.animates {
            let axis = SIMD3<Float>(0, 1, 0)
            animation = try SceneAnimation(tracks: [
                SceneAnimationTrack(nodeID: groupID, keyframes: [
                    TransformKeyframe(time: 0, transform: SceneTransform(rotation: simd_quatf(angle: -0.36, axis: axis))),
                    TransformKeyframe(time: 2, transform: SceneTransform(rotation: simd_quatf(angle: 0.52, axis: axis))),
                    TransformKeyframe(time: 4, transform: SceneTransform(rotation: simd_quatf(angle: -0.36, axis: axis))),
                ])
            ])
        }
    }

    var selectionDescription: String {
        let object = switch mode {
        case .obj, .physicallyBased: "octahedron"
        case .gltf: "tetrahedron"
        case .physicallyBasedShadows: "sphere"
        case .cubes, .shadows: "cube"
        }
        if selected == leftID { return "Left \(object) selected" }
        if selected == centerID { return "Center \(object) selected" }
        if selected == rightID { return "Right \(object) selected" }
        return "No \(object) selected"
    }

    func select(at point: SIMD2<Float>, aspectRatio: Float) throws -> Bool {
        guard let displayedGraph, displayedGraph.aspect == aspectRatio else { return false }
        let hit = try displayedGraph.graph.pick(at: point, aspectRatio: aspectRatio)
        let id = hit?.nodeID == floorID ? nil : hit?.nodeID
        guard id != selected else { return false }
        selected = id
        cachedGraph = nil
        return true
    }

    func draw(
        pass: MTLRenderPassDescriptor,
        drawable: any CAMetalDrawable,
        frame: Int,
        completed: @escaping @Sendable (String?) -> Void
    ) throws {
        guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("command buffer") }
        let aspectRatio = Float(drawable.texture.width) / Float(drawable.texture.height)
        let base = try graph(aspectRatio: aspectRatio)
        let scene = try animation?.applying(to: base, at: Float(frame % 240) / 60, looping: true) ?? base
        try scene.encode(pass: pass, commandBuffer: buffer, shadowMap: shadowMap)
        submittedGraph = (aspectRatio, scene)
        buffer.addCompletedHandler { [weak self] command in
            let error = command.status == .error
                ? command.error?.localizedDescription ?? "Metal could not finish the scene frame." : nil
            Task { @MainActor in
                if error == nil { self?.displayedGraph = self?.submittedGraph }
                self?.submittedGraph = nil
                completed(error)
            }
        }
        buffer.present(drawable)
        buffer.commit()
    }

    private func graph(aspectRatio: Float) throws -> SceneGraph {
        if let cachedGraph, cachedGraph.aspect == aspectRatio { return cachedGraph.graph }

        let pitch = simd_quatf(angle: -0.22, axis: SIMD3(1, 0, 0))
        let yaw = simd_quatf(angle: 0.14, axis: SIMD3(0, 1, 0))
        let group = SceneNode(
            id: groupID,
            transform: SceneTransform(rotation: yaw * pitch),
            children: [
                cube(id: leftID, position: SIMD3(-1.45, 0, 0.15), color: SIMD4(0.44, 0.62, 0.96, 1)),
                cube(id: centerID, position: SIMD3(0, 0.25, 0.03), color: SIMD4(0.87, 0.32, 0.57, 1)),
                SceneNode(
                    id: subgroupID,
                    transform: SceneTransform(translation: SIMD3(1.35, -0.1, -0.15)),
                    children: [
                        cube(id: rightID, position: SIMD3(0.14, 0.04, 0.12), color: SIMD4(0.23, 0.8, 0.52, 1)),
                    ]
                ),
            ]
        )
        let roots: [SceneNode]
        let pose: SceneTransform
        if let floor {
            roots = [
                SceneNode(
                    id: floorID,
                    transform: SceneTransform(
                        translation: SIMD3(0, -0.82, 0),
                        rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
                    ),
                    mesh: floor, materialParameters: SIMD4(0.68, 0.7, 0.73, 1), castsShadow: false
                ),
                group,
            ]
            let eye = SIMD3<Float>(0, 3.1, 7.2 / min(1, aspectRatio))
            let forward = simd_normalize(SIMD3<Float>(0, -0.15, 0) - eye)
            pose = SceneTransform(translation: eye, rotation: simd_quatf(from: SIMD3(0, 0, -1), to: forward))
        } else {
            roots = try importedScene.map(importedRoots) ?? [group]
            pose = SceneTransform(translation: SIMD3(0, 0, 5.8 / min(1, aspectRatio)))
        }
        let result = SceneGraph(roots: roots, camera: camera, cameraTransform: pose)
        cachedGraph = (aspectRatio, result)
        return result
    }

    private func importedRoots(_ scene: GLTFScene) throws -> [SceneNode] {
        let nodes = Dictionary(uniqueKeysWithValues: scene.nodes.map { ($0.sourceIndex, $0) })
        let identities = [groupID, leftID, centerID, subgroupID, rightID]
        let colors: [Int: SIMD4<Float>] = [
            1: SIMD4(0.44, 0.62, 0.96, 1),
            2: SIMD4(0.87, 0.32, 0.57, 1),
            4: SIMD4(0.23, 0.8, 0.52, 1),
        ]
        func makeNode(_ sourceIndex: Int) throws -> SceneNode {
            guard let node = nodes[sourceIndex], identities.indices.contains(sourceIndex) else {
                throw MeshRenderer.Failure.resource("fixture glTF node")
            }
            let id = identities[sourceIndex]
            return SceneNode(
                id: id, transform: node.localTransform, mesh: node.meshIndex == nil ? nil : mesh,
                materialParameters: selected == id ? SIMD4(1, 0.64, 0.04, 1)
                    : importedTint * colors[sourceIndex, default: SIMD4(repeating: 1)],
                children: try node.childIndices.map(makeNode)
            )
        }
        return try scene.rootNodeIndices.map(makeNode)
    }

    private static let checkerPNG = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAAXNSR0IArs4c6QAAACFJREFUeJxj+HUp7z8I62fsAWN0PgNBBbgkYHzCCgaBGwBU7bFhmJAkZgAAAABJRU5ErkJggg=="

    private static let tetrahedronGLB = "Z2xURgIAAADoBgAAZAYAAEpTT057ImFzc2V0Ijp7InZlcnNpb24iOiIyLjAifSwiYnVmZmVycyI6W3siYnl0ZUxlbmd0aCI6MTA0fV0sImJ1ZmZlclZpZXdzIjpbeyJidWZmZXIiOjAsImJ5dGVPZmZzZXQiOjAsImJ5dGVMZW5ndGgiOjQ4fSx7ImJ1ZmZlciI6MCwiYnl0ZU9mZnNldCI6NDgsImJ5dGVMZW5ndGgiOjMyfSx7ImJ1ZmZlciI6MCwiYnl0ZU9mZnNldCI6ODAsImJ5dGVMZW5ndGgiOjI0fV0sImFjY2Vzc29ycyI6W3siYnVmZmVyVmlldyI6MCwiY29tcG9uZW50VHlwZSI6NTEyNiwiY291bnQiOjQsInR5cGUiOiJWRUMzIiwibWluIjpbLTAuNjUsLTAuNiwtMC42NV0sIm1heCI6WzAuNjUsMC43LDAuNV19LHsiYnVmZmVyVmlldyI6MSwiY29tcG9uZW50VHlwZSI6NTEyNiwiY291bnQiOjQsInR5cGUiOiJWRUMyIn0seyJidWZmZXJWaWV3IjoyLCJjb21wb25lbnRUeXBlIjo1MTIzLCJjb3VudCI6MTIsInR5cGUiOiJTQ0FMQVIifV0sIm1lc2hlcyI6W3sicHJpbWl0aXZlcyI6W3siYXR0cmlidXRlcyI6eyJQT1NJVElPTiI6MCwiVEVYQ09PUkRfMCI6MX0sImluZGljZXMiOjIsIm1hdGVyaWFsIjowfV19XSwic2NlbmUiOjAsInNjZW5lcyI6W3sibmFtZSI6IlRldHJhaGVkcm9uIGdyb3VwIiwibm9kZXMiOlswXX1dLCJub2RlcyI6W3sibmFtZSI6Ikdyb3VwIiwiY2hpbGRyZW4iOlsxLDIsM119LHsibmFtZSI6IkxlZnQiLCJtZXNoIjowLCJ0cmFuc2xhdGlvbiI6Wy0xLjQ1LDAsMC4xNV0sInJvdGF0aW9uIjpbLTAuMTIxMjQ3OTY2MjE4OTU3NTksMC4yMzEwMjYyNjcwNTYxMzIzOCwwLjAyOTAyOTYzNzEzOTM4MDk5NywwLjk2NDkyNjQ2MDgxNzMxNThdLCJzY2FsZSI6WzAuOSwwLjksMC45XX0seyJuYW1lIjoiQ2VudGVyIiwibWVzaCI6MCwidHJhbnNsYXRpb24iOlswLDAuMjUsMC4wM10sInJvdGF0aW9uIjpbLTAuMTIxMjQ3OTY2MjE4OTU3NTksMC4yMzEwMjYyNjcwNTYxMzIzOCwwLjAyOTAyOTYzNzEzOTM4MDk5NywwLjk2NDkyNjQ2MDgxNzMxNThdLCJzY2FsZSI6WzAuOSwwLjksMC45XX0seyJuYW1lIjoiU3ViZ3JvdXAiLCJjaGlsZHJlbiI6WzRdLCJ0cmFuc2xhdGlvbiI6WzEuMzUsLTAuMSwtMC4xNV19LHsibmFtZSI6IlJpZ2h0IiwibWVzaCI6MCwidHJhbnNsYXRpb24iOlswLjE0LDAuMDQsMC4xMl0sInJvdGF0aW9uIjpbLTAuMTIxMjQ3OTY2MjE4OTU3NTksMC4yMzEwMjYyNjcwNTYxMzIzOCwwLjAyOTAyOTYzNzEzOTM4MDk5NywwLjk2NDkyNjQ2MDgxNzMxNThdLCJzY2FsZSI6WzAuOSwwLjksMC45XX1dLCJpbWFnZXMiOlt7InVyaSI6ImRhdGE6aW1hZ2UvcG5nO2Jhc2U2NCxpVkJPUncwS0dnb0FBQUFOU1VoRVVnQUFBQWdBQUFBSUNBWUFBQURFRDc2TEFBQUFBWE5TUjBJQXJzNGM2UUFBQUNGSlJFRlVlSnhqK0hVcDd6OEk2MmZzQVdOMFBnTkJCYmdrWUh6Q0NnYUJHd0JVN2JGaG1KQWtaZ0FBQUFCSlJVNUVya0pnZ2c9PSJ9XSwidGV4dHVyZXMiOlt7InNvdXJjZSI6MH1dLCJtYXRlcmlhbHMiOlt7Im5hbWUiOiJGaXh0dXJlIGNoZWNrZXIiLCJwYnJNZXRhbGxpY1JvdWdobmVzcyI6eyJiYXNlQ29sb3JUZXh0dXJlIjp7ImluZGV4IjowfSwiYmFzZUNvbG9yRmFjdG9yIjpbMC45MiwxLjAsMC44OCwxLjBdLCJtZXRhbGxpY0ZhY3RvciI6MC4yMiwicm91Z2huZXNzRmFjdG9yIjowLjQ1fX1dfSAgaAAAAEJJTgBmZia/mpkZvwAAAD9mZiY/mpkZvwAAAD8AAAAAmpkZv2ZmJr8AAAAAMzMzPwAAAAAAAAAAAACAPwAAgD8AAIA/AAAAP5qZWT8AAAA/AAAAAAAAAgABAAAAAQADAAEAAgADAAIAAAADAA=="

    private static let octahedron = """
    v 0 0.7 0
    v 0.6 0 0
    v 0 0 0.6
    v -0.6 0 0
    v 0 0 -0.6
    v 0 -0.7 0
    vt 0.5 1
    vt 0 0
    vt 1 0
    f 1/1 3/2 2/3
    f 1/1 4/2 3/3
    f 1/1 5/2 4/3
    f 1/1 2/2 5/3
    f 6/1 2/2 3/3
    f 6/1 3/2 4/3
    f 6/1 4/2 5/3
    f 6/1 5/2 2/3
    """

    private func cube(id: UUID, position: SIMD3<Float>, color: SIMD4<Float>) -> SceneNode {
        let tilt = simd_quatf(angle: -0.25, axis: SIMD3(1, 0, 0))
        let turn = simd_quatf(angle: 0.47, axis: SIMD3(0, 1, 0))
        return SceneNode(
            id: id,
            transform: SceneTransform(translation: position, rotation: turn * tilt, scale: SIMD3(repeating: 0.9)),
            mesh: mesh,
            materialParameters: selected == id ? SIMD4(1, 0.64, 0.04, 1) : color
        )
    }
}
