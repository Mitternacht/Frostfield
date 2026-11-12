import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct WireframeShadingTests {
    private let fixture: GeometryRenderFixture

    init() throws {
        fixture = try GeometryRenderFixture()
    }

    private func square(normal: SIMD3<Float>) throws -> MeshGeometry {
        let corners = [SIMD3<Float>(-1, -1, 0), SIMD3(1, -1, 0), SIMD3(1, 1, 0), SIMD3(-1, 1, 0)]
        return try MeshGeometry(
            vertices: corners.map {
                MeshVertex(position: $0, normal: normal, textureCoordinate: SIMD2($0.x, $0.y), tangent: SIMD4(1, 0, 0, 1))
            },
            indices: [0, 1, 2, 0, 2, 3]
        )
    }

    private func render(_ mesh: MeshRenderer, lights: [SceneNode] = []) throws -> [SIMD4<Float>] {
        let graph = SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: mesh)] + lights,
            camera: fixture.camera
        )
        return try fixture.render(graph).pixels
    }

    private func covered(_ pixels: [SIMD4<Float>]) -> Int {
        pixels.count(where: { $0.w > 0.5 })
    }

    @Test("A wireframe material draws edges and leaves the triangle interior empty")
    func wireframeLeavesTheInteriorEmpty() throws {
        let filled = try ClassicMaterial(model: .basic, color: SIMD3(1, 0, 0))
        let lines = try ClassicMaterial(model: .basic, color: SIMD3(1, 0, 0), wireframe: true)
        let solid = try render(try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try square(normal: SIMD3(0, 0, 1)),
            material: .classic(filled)
        ))
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try square(normal: SIMD3(0, 0, 1)),
            material: .classic(lines)
        )
        let drawn = try render(mesh)
        let interior = fixture.pixel(drawn, x: 40, y: 40)
        #expect(fixture.pixel(solid, x: 40, y: 40).x > 0.5)
        #expect(interior.w == 0)
        #expect(covered(drawn) > 8)
        #expect(covered(drawn) < covered(solid) / 4)
        let scene = SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: mesh)],
            camera: fixture.camera
        )
        let archive = try SceneArchive(capturing: scene, commandQueue: fixture.queue)
        let loaded = try archive.load(device: fixture.device, commandQueue: fixture.queue).graph
        let restored = try fixture.render(loaded).pixels
        #expect(restored == drawn)
        #expect(fixture.pixel(restored, x: 40, y: 40).w == 0)
    }

    @Test("Flat shading uses the face normal without baking a new mesh")
    func flatShadingUsesTheFaceNormal() throws {
        let geometry = try square(normal: SIMD3(1, 0, 0))
        let light = SceneNode(light: try PunctualLight(kind: .directional, intensity: .pi))
        func center(flat: Bool) throws -> SIMD4<Float> {
            let material = try ClassicMaterial(model: .lambert, color: SIMD3(1, 0, 0), flatShading: flat)
            let mesh = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .classic(material)
            )
            return fixture.pixel(try render(mesh, lights: [light]), x: 40, y: 40)
        }
        #expect(try center(flat: false).x < 0.05)
        #expect(try center(flat: true).x > 0.5)
    }

    @Test("A physically based material can draw its edges or shade each face flat")
    func physicallyBasedWireframeAndFlatShading() throws {
        let geometry = try square(normal: SIMD3(1, 0, 0))
        let lines = try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(1, 0, 0, 1),
            shadingModel: .unlit, wireframe: true
        )
        let drawn = try render(try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(lines)
        ))
        #expect(fixture.pixel(drawn, x: 40, y: 40).w == 0)
        #expect(covered(drawn) > 8)

        func center(flat: Bool) throws -> SIMD4<Float> {
            let material = try PhysicallyBasedMaterial(
                roughness: 1, directionToLight: SIMD3(0, 0, 1), lightIntensity: .pi, ambientIntensity: 0,
                baseColorFactor: SIMD4(1, 0, 0, 1), flatShading: flat
            )
            let mesh = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry,
                material: .physicallyBased(material)
            )
            return fixture.pixel(try render(mesh), x: 40, y: 40)
        }
        let smooth = try center(flat: false)
        let flat = try center(flat: true)
        #expect(smooth.x < 0.05)
        #expect(flat.x > 0.5)
        let scene = SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry,
                material: .physicallyBased(try PhysicallyBasedMaterial(
                    roughness: 1, directionToLight: SIMD3(0, 0, 1), lightIntensity: .pi, ambientIntensity: 0,
                    baseColorFactor: SIMD4(1, 0, 0, 1), flatShading: true
                ))
            ))],
            camera: fixture.camera
        )
        let archive = try SceneArchive(capturing: scene, commandQueue: fixture.queue)
        let loaded = try archive.load(device: fixture.device, commandQueue: fixture.queue).graph
        #expect(try fixture.render(loaded).pixels == fixture.render(scene).pixels)
    }
}
