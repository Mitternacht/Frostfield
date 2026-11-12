import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct OrthographicSceneCameraTests {
    private let fixture: GeometryRenderFixture

    init() throws {
        fixture = try GeometryRenderFixture(side: 64)
    }

    @Test("Equal quads at two depths cover the same pixels when the scene camera is orthographic")
    func equalCoverage() throws {
        func quad(x: Float, depth: Float, color: SIMD3<Float>) throws -> SceneNode {
            let material = try PhysicallyBasedMaterial(
                lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(color, 1),
                alphaMode: .opaque, shadingModel: .unlit
            )
            let mesh = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float,
                geometry: .plane(width: 0.8, height: 0.8), material: .physicallyBased(material)
            )
            return SceneNode(transform: .init(translation: SIMD3(x, 0, -depth)), mesh: mesh)
        }
        var graph = SceneGraph(
            roots: [
                try quad(x: -0.7, depth: 2, color: SIMD3(1, 0, 0)),
                try quad(x: 0.7, depth: 6, color: SIMD3(0, 0, 1)),
            ],
            camera: fixture.camera
        )
        let perspective = try fixture.render(graph).pixels
        graph.orthographicCamera = try OrthographicCamera(halfHeight: 2, near: 0.1, far: 50)
        let orthographic = try fixture.render(graph).pixels
        func count(_ pixels: [SIMD4<Float>], _ channel: KeyPath<SIMD4<Float>, Float>) -> Int {
            pixels.filter { $0[keyPath: channel] > 0.5 }.count
        }
        let nearPerspective = count(perspective, \.x)
        let farPerspective = count(perspective, \.z)
        let nearOrthographic = count(orthographic, \.x)
        let farOrthographic = count(orthographic, \.z)
        #expect(nearPerspective > farPerspective * 2)
        #expect(nearOrthographic == farOrthographic)
        #expect(nearOrthographic > 20)

        let center = try graph.cameraRay(at: SIMD2(0.5, 0.5), aspectRatio: 1)
        let edge = try graph.cameraRay(at: SIMD2(1, 0.5), aspectRatio: 1)
        #expect(simd_distance(center.direction, SIMD3(0, 0, -1)) < 1e-4)
        #expect(simd_distance(edge.direction, center.direction) < 1e-4)
        #expect(edge.origin.x > center.origin.x + 1)

        let archive = try SceneArchive(capturing: graph, commandQueue: fixture.queue)
        let restored = try archive.load(device: fixture.device, commandQueue: fixture.queue).graph
        #expect(restored.orthographicCamera?.halfHeight == 2)
        #expect(restored.orthographicCamera?.near == 0.1 && restored.orthographicCamera?.far == 50)

        let target = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float,
            geometry: .plane(width: 0.2, height: 0.2), material: .unlit
        )
        var picked = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(1, 0, -4)), mesh: target)],
            camera: fixture.camera
        )
        picked.orthographicCamera = try OrthographicCamera(halfHeight: 2, near: 0.1, far: 50)
        #expect(try picked.pick(at: SIMD2(0.75, 0.5), aspectRatio: 1) != nil)
        picked.orthographicCamera = nil
        #expect(try picked.pick(at: SIMD2(0.75, 0.5), aspectRatio: 1) == nil)
    }

    @Test("An orthographic view keeps the specular highlight even across a facing plane")
    func parallelShading() throws {
        let material = try PhysicallyBasedMaterial(
            metallic: 1, roughness: 0.05, directionToLight: SIMD3(0, 0, 1), lightIntensity: 8, ambientIntensity: 0,
            baseColorFactor: SIMD4(repeating: 1)
        )
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float,
            geometry: .plane(width: 2, height: 2), material: .physicallyBased(material)
        )
        var graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: mesh)],
            camera: fixture.camera
        )
        func spread(_ pixels: [SIMD4<Float>]) -> Float {
            let center = pixels[32 * fixture.side + 32].x
            let edge = pixels[32 * fixture.side + 20].x
            return abs(center - edge)
        }
        let perspectivePixels = try fixture.render(graph).pixels
        graph.orthographicCamera = try OrthographicCamera(halfHeight: 2, near: 0.1, far: 50)
        let orthographicPixels = try fixture.render(graph).pixels
        let center = orthographicPixels[32 * fixture.side + 32].x
        #expect(center > 0.2, "center \(center)")
        #expect(spread(perspectivePixels) > 0.05)
        #expect(spread(orthographicPixels) < spread(perspectivePixels) * 0.25)
    }
}
