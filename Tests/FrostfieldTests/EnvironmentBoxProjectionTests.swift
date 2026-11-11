import Frostfield
import Metal
import Testing
import simd

struct EnvironmentBoxProjectionTests {
    @Test func `Projection follows r187 in translated bounds and supports an off-center capture`() throws {
        let bounds = try AxisAlignedBox(minimum: SIMD3(4, 1, -12), maximum: SIMD3(12, 7, -2))
        let rayOrigin = SIMD3<Float>(10, 5, -7)
        let ray = SIMD3<Float>(1, 2, 4)
        let centered = try EnvironmentBoxProjection(bounds: bounds, capturePosition: bounds.center)
        let pinned = SIMD3<Float>(0.514495755, 0.514495755, 0.685994341)
        #expect(try simd_distance(centered.direction(from: rayOrigin, along: ray), pinned) < 0.000001)

        let capture = SIMD3<Float>(7, 4, -7)
        let offCenter = try EnvironmentBoxProjection(bounds: bounds, capturePosition: capture)
        let direction = try offCenter.direction(from: SIMD3(10, 4, -7), along: SIMD3(0, 0, 7))
        #expect(simd_distance(direction, simd_normalize(SIMD3(3, 0, 5))) < 0.000001)
        #expect(try simd_distance(offCenter.direction(from: capture, along: ray), simd_normalize(ray)) < 0.000001)
    }

    @Test func `Parallel faces, outside rays and invalid configurations are defined`() throws {
        let bounds = try AxisAlignedBox(minimum: SIMD3(4, 1, -12), maximum: SIMD3(12, 7, -2))
        let probe = try EnvironmentBoxProjection(bounds: bounds, capturePosition: SIMD3(7, 4, -7))
        let tangent = try probe.direction(from: SIMD3(12, 4, -7), along: SIMD3(0, 1, 0))
        #expect(simd_distance(tangent, simd_normalize(SIMD3(5, 3, 0))) < 0.000001)
        #expect(try probe.direction(from: SIMD3(12, 4, -7), along: SIMD3(-1, 0, 0)) == SIMD3(-1, 0, 0))
        #expect(try probe.direction(from: SIMD3(13, 4, -7), along: SIMD3(-2, 0, 0)) == SIMD3(-1, 0, 0))
        #expect(throws: EnvironmentBoxProjection.Failure.ray) { try probe.direction(from: .zero, along: .zero) }
        #expect(throws: EnvironmentBoxProjection.Failure.capturePosition) {
            try EnvironmentBoxProjection(bounds: bounds, capturePosition: SIMD3(4, 2, -5))
        }
        let flat = try AxisAlignedBox(minimum: .zero, maximum: SIMD3(1, 0, 1))
        #expect(throws: EnvironmentBoxProjection.Failure.bounds) {
            try EnvironmentBoxProjection(bounds: flat, capturePosition: .zero)
        }
        let wide = try AxisAlignedBox(minimum: .zero, maximum: SIMD3(1e30, 2, 2))
        let nearWall = try EnvironmentBoxProjection(bounds: wide, capturePosition: SIMD3(1e-20, 1, 1))
        #expect(try nearWall.direction(from: SIMD3(1e20, 1, 1), along: SIMD3(-1, 0, 0)) == SIMD3(-1, 0, 0))
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct EnvironmentBoxProjectionRasterTests {
    @Test func `A translated room reflection tracks the correct wall patch for an off-center probe`() throws {
        let gpu = try RasterLightingFixture(width: 25, height: 25)
        let bounds = try AxisAlignedBox(minimum: SIMD3(4, 1, -12), maximum: SIMD3(12, 7, -2))
        let capture = SIMD3<Float>(7, 4, -7)
        let projection = try EnvironmentBoxProjection(bounds: bounds, capturePosition: capture)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 30)
        let cube = try CubeCamera.makeTarget(device: gpu.device, size: 64, mipmapped: false)
        let cubeCamera = try CubeCamera(queue: gpu.queue, target: cube, near: 0.1, far: 30)
        let captureGraph = SceneGraph(roots: try room(gpu, bounds: bounds), camera: camera)
        try gpu.run { try cubeCamera.encode(captureGraph, at: capture, commandBuffer: $0) }
        let lighting = try EnvironmentLighting(queue: gpu.queue, cube: cube, settings: .init(
            cubeSize: 64, diffuseSize: 8, brdfSize: 32, sampleCount: 32
        ))
        let raw = try gpu.sampleCube(lighting.skyCube, directions: [SIMD3(0, 0, 1), SIMD3(3, 0, 5)])
        #expect(raw[0].z > 7 && raw[0].x < 0.1)
        #expect(raw[1].x > 7 && raw[1].z < 0.1)

        let ordinary = try PhysicallyBasedMaterial.Environment(lighting: lighting)
        let explicitDefault = try PhysicallyBasedMaterial.Environment(lighting: lighting, boxProjection: nil)
        let boxed = try PhysicallyBasedMaterial.Environment(lighting: lighting, boxProjection: projection)
        let mirror = try MeshGeometry.plane(width: 2, height: 2)
        func graph(_ environment: PhysicallyBasedMaterial.Environment, position: SIMD3<Float>) throws -> SceneGraph {
            let material = try PhysicallyBasedMaterial(
                metallic: 1, roughness: 0, lightIntensity: 0, ambientIntensity: 0, environment: environment
            )
            let mesh = try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float, geometry: mirror, material: .physicallyBased(material))
            let receiver = SceneNode(transform: .init(translation: position), mesh: mesh)
            return SceneGraph(roots: [receiver], camera: camera, cameraTransform: .init(translation: position + SIMD3(0, 0, 2.5)))
        }
        func center(_ graph: SceneGraph) throws -> SIMD4<Float> {
            try gpu.pixels(of: gpu.render(graph).color)[gpu.width * gpu.height / 2]
        }
        let receiver = SIMD3<Float>(10, 4, -7)
        let ordinaryPixel = try center(graph(ordinary, position: receiver))
        let boxedGraph = try graph(boxed, position: receiver)
        let boxedPixel = try center(boxedGraph)
        #expect(ordinaryPixel.z > 6 && ordinaryPixel.x < 0.1)
        #expect(boxedPixel.x > 6 && boxedPixel.z < 0.1)
        #expect(try center(graph(explicitDefault, position: receiver)) == ordinaryPixel)
        #expect(try center(graph(boxed, position: capture)) == center(graph(ordinary, position: capture)))
        let outside = SIMD3<Float>(14, 4, -7)
        #expect(try center(graph(boxed, position: outside)) == center(graph(ordinary, position: outside)))
        #expect(throws: SceneArchive.Failure.unsupportedMaterial("box-projected environment bindings")) {
            try SceneArchive(capturing: boxedGraph)
        }
    }

    private func room(_ gpu: RasterLightingFixture, bounds: AxisAlignedBox) throws -> [SceneNode] {
        let wall = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .cube(),
            material: .classic(try ClassicMaterial(model: .basic, color: SIMD3(repeating: 0.25))), renderState: .init(side: .back)
        )
        let shell = SceneNode(transform: .init(translation: bounds.center, scale: bounds.size), mesh: wall)
        let panel = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 1.5, height: 1.5),
            material: .classic(try ClassicMaterial(model: .basic))
        )
        let flip = simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
        return [
            shell,
            SceneNode(transform: .init(translation: SIMD3(7, 4, -2.01), rotation: flip), mesh: panel, materialParameters: SIMD4(0, 0, 8, 1)),
            SceneNode(transform: .init(translation: SIMD3(10, 4, -2.01), rotation: flip), mesh: panel, materialParameters: SIMD4(8, 0, 0, 1))
        ]
    }
}
