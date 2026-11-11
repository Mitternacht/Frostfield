import Foundation
import Frostfield
import Metal
import Testing
import simd

/// glTF round-tripping of diffuse transmission, diffuse roughness, and volume scattering.
struct GLTFTranslucencyTests {
    private static let diffuseTransmission = "KHR_materials_diffuse_transmission"
    private static let diffuseRoughness = "KHR_materials_diffuse_roughness"
    private static let volumeScatter = "KHR_materials_volume_scatter"
    private static let volume = "KHR_materials_volume"

    private func asset(extensions: [String: [String: Any]], required: Bool = false) throws -> Data {
        var fixture = GLTFExtensionFixture()
        let positions = fixture.accessor(
            GLTFExtensionFixture.floats([0, 0, 0, 1, 0, 0, 0, 1, 0]), component: 5126, type: "VEC3", count: 3
        )
        let uv = fixture.accessor(GLTFExtensionFixture.floats([0, 0, 1, 0, 0, 1]), component: 5126, type: "VEC2", count: 3)
        let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMQSTnxHwAEEAJAKtk0QwAAAABJRU5ErkJggg=="
        var sections: [String: Any] = [
            "nodes": [["mesh": 0]], "scenes": [["nodes": [0]]],
            "meshes": [["primitives": [["attributes": ["POSITION": positions, "TEXCOORD_0": uv], "material": 0]]]],
            "materials": [["extensions": extensions]],
            "textures": [["source": 0]], "images": [["uri": "data:image/png;base64,\(png)"]],
            "extensionsUsed": extensions.keys.sorted(),
        ]
        if required { sections["extensionsRequired"] = extensions.keys.sorted() }
        return try fixture.document(sections)
    }

    @Test("Diffuse transmission, diffuse roughness and volume scattering import with their textures and prepare a translucent material")
    func importsAndPrepares() throws {
        let data = try asset(extensions: [
            Self.diffuseTransmission: [
                "diffuseTransmissionFactor": 0.75, "diffuseTransmissionColorFactor": [1, 0.5, 0.25],
                "diffuseTransmissionTexture": ["index": 0], "diffuseTransmissionColorTexture": ["index": 0],
            ],
            Self.diffuseRoughness: ["diffuseRoughnessFactor": 0.6, "diffuseRoughnessTexture": ["index": 0]],
            Self.volume: ["thicknessFactor": 0.2, "attenuationDistance": 0.01, "attenuationColor": [0.9, 0.6, 0.3]],
            Self.volumeScatter: ["multiscatterColorFactor": [0.8, 0.4, 0.2], "scatterAnisotropy": 0.3,
                                 "multiscatterColorTexture": ["index": 0]],
        ], required: true)
        let scene = try GLTFMeshLoader.loadScene(data: data, includeMaterials: true)
        let material = try #require(scene.materials.first)
        let diffuse = try #require(material.diffuseTransmission)
        #expect(diffuse.factor == 0.75 && diffuse.colorFactor == SIMD3(1, 0.5, 0.25))
        #expect(diffuse.factorTexture != nil && diffuse.colorTexture != nil)
        #expect(material.diffuseRoughness?.factor == 0.6 && material.diffuseRoughness?.texture != nil)
        let scatter = try #require(material.volumeScatter)
        #expect(scatter.multiscatterColorFactor == SIMD3(0.8, 0.4, 0.2) && scatter.scatterAnisotropy == 0.3)
        #expect(scatter.multiscatterColorTexture != nil)
        // The volume is valid without KHR_materials_transmission because diffuse transmission enters it.
        #expect(material.transmission == nil && material.volume?.thickness == 0.2)

        let device = try #require(MTLCreateSystemDefaultDevice())
        let prepared = try material.prepare(device: device)
        #expect(prepared.diffuseTransmission?.factor == 0.75 && prepared.diffuseTransmission?.colorTexture != nil)
        #expect(prepared.diffuseRoughness?.factor == 0.6)
        #expect(prepared.volumeScatter?.anisotropy == 0.3 && prepared.volumeScatter?.multiscatterColorTexture != nil)
        // The volume becomes a zero-factor transmission carrying its thickness and attenuation.
        #expect(prepared.transmission?.factor == 0 && prepared.transmission?.thickness == 0.2)
        #expect(prepared.transmission?.attenuationDistance == 0.01)
        // The path tracer accepts it: a closed-volume medium entered diffusely.
        let queue = try #require(device.makeCommandQueue())
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let renderer = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: try MeshGeometry.uvSphere(), material: .physicallyBased(prepared)
        )
        let traced = try PathTracingScene(queue: queue, graph: SceneGraph(roots: [SceneNode(mesh: renderer)], camera: camera), textureSize: 16)
        #expect(traced.materialCount == 1)
    }

    @Test("Translucency extensions validate their factors and dependencies")
    func validation() throws {
        func failure(_ extensions: [String: [String: Any]]) throws -> GLTFMeshLoader.Failure? {
            do {
                _ = try GLTFMeshLoader.loadScene(data: try asset(extensions: extensions), includeMaterials: true)
                return nil
            } catch let failure as GLTFMeshLoader.Failure {
                return failure
            }
        }
        #expect(try failure([Self.diffuseTransmission: ["diffuseTransmissionFactor": 1.5]]) == .invalidMaterial(
            0, "diffuse transmission needs a factor and three color components between zero and one"
        ))
        #expect(try failure([Self.diffuseTransmission: ["diffuseTransmissionColorFactor": [1, 1]]]) == .invalidMaterial(
            0, "diffuse transmission needs a factor and three color components between zero and one"
        ))
        #expect(try failure([Self.diffuseRoughness: ["diffuseRoughnessFactor": -0.5]]) == .invalidMaterial(
            0, "diffuse roughness must be between zero and one"
        ))
        #expect(try failure([Self.volumeScatter: ["multiscatterColorFactor": [0.5, 0.5, 0.5]]]) == .unsupportedFeature(
            "KHR_materials_volume_scatter without KHR_materials_volume on material 0"
        ))
        #expect(try failure([
            Self.diffuseTransmission: ["diffuseTransmissionFactor": 1], Self.volume: ["thicknessFactor": 1],
            Self.volumeScatter: ["scatterAnisotropy": 1],
        ]) == .invalidMaterial(
            0, "volume scatter needs three multiscatter color components between zero and one and an anisotropy between -1 and 1"
        ))
        #expect(try failure([Self.diffuseTransmission: ["extensions": ["EXT_unknown": [:]]]]) == .unsupportedFeature(
            "EXT_unknown on diffuse transmission on material 0"
        ))
        // A thin-walled volume (zero thickness) keeps no medium, so scattering is dropped on preparation.
        let thin = try GLTFMeshLoader.loadScene(data: try asset(extensions: [
            Self.diffuseTransmission: ["diffuseTransmissionFactor": 1], Self.volume: ["thicknessFactor": 0],
            Self.volumeScatter: ["multiscatterColorFactor": [0.5, 0.5, 0.5]],
        ]), includeMaterials: true)
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(try thin.materials[0].prepare(device: device).volumeScatter == nil)
        #expect(GLTFMeshLoader.supportedExtensions.isSuperset(of: [Self.diffuseTransmission, Self.diffuseRoughness, Self.volumeScatter]))
    }

    @Test("Imported and native translucency retain textures, transforms and volume properties on export",
          arguments: [GLTFExporter.Format.gltf, .glb])
    func roundTrip(format: GLTFExporter.Format) throws {
        let transformed: [String: Any] = [
            "index": 0, "extensions": ["KHR_texture_transform": ["offset": [0.125, 0.25]]],
        ]
        let data = try asset(extensions: [
            Self.diffuseTransmission: [
                "diffuseTransmissionFactor": 0.75, "diffuseTransmissionColorFactor": [1, 0.5, 0.25],
                "diffuseTransmissionTexture": transformed, "diffuseTransmissionColorTexture": ["index": 0],
            ],
            Self.diffuseRoughness: ["diffuseRoughnessFactor": 0.5, "diffuseRoughnessTexture": ["index": 0]],
            Self.volume: ["thicknessFactor": 0.2, "attenuationDistance": 0.5],
            Self.volumeScatter: [
                "multiscatterColorFactor": [0.75, 0.5, 0.25], "scatterAnisotropy": 0.25,
                "multiscatterColorTexture": ["index": 0],
            ],
        ])
        let scene = try GLTFMeshLoader.loadScene(data: data, includeMaterials: true)
        let original = try #require(scene.materials.first)
        let exporter = GLTFExporter(format: format)
        let importedOutput = try exporter.export(scene)

        let device = try #require(MTLCreateSystemDefaultDevice())
        let renderer = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: try MeshGeometry.uvSphere(),
            material: .physicallyBased(try original.prepare(device: device))
        )
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let graphOutput = try exporter.export(SceneGraph(roots: [SceneNode(mesh: renderer)], camera: camera))

        for output in [importedOutput, graphOutput] {
            #expect(output.warnings.isEmpty)
            let scene = try GLTFMeshLoader.loadScene(data: output.data, includeMaterials: true)
            let material = try #require(scene.materials.first)
            let diffuse = try #require(material.diffuseTransmission)
            #expect(diffuse.factor == 0.75 && diffuse.colorFactor == SIMD3(1, 0.5, 0.25))
            #expect(diffuse.factorTexture?.transform.offset == SIMD2(0.125, 0.25))
            #expect(diffuse.colorTexture != nil)
            #expect(material.diffuseRoughness?.factor == 0.5 && material.diffuseRoughness?.texture != nil)
            #expect(material.volume?.thickness == 0.2 && material.volume?.attenuationDistance == 0.5)
            #expect(material.transmission == nil)
            #expect(material.volumeScatter?.multiscatterColorFactor == SIMD3(0.75, 0.5, 0.25))
            #expect(material.volumeScatter?.scatterAnisotropy == 0.25)
            #expect(material.volumeScatter?.multiscatterColorTexture != nil)
            let prepared = try material.prepare(device: device)
            #expect(prepared.diffuseTransmission?.factorTexture?.transform.offset == SIMD2(0.125, 0.25))
            #expect(prepared.volumeScatter?.multiscatterColorTexture != nil)
        }
    }

    @Test("glTF export reports the native subsurface layer it cannot represent")
    func exportWarnings() throws {
        // Scene graph export reports Frostfield's subsurface scattering, which glTF cannot carry.
        let device = try #require(MTLCreateSystemDefaultDevice())
        let renderer = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: try MeshGeometry.uvSphere(),
            material: .physicallyBased(try PhysicallyBasedMaterial(subsurface: .init()))
        )
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let graphOutput = try GLTFExporter(format: .gltf).export(SceneGraph(roots: [SceneNode(mesh: renderer)], camera: camera))
        #expect(graphOutput.warnings.contains(.materialLayersDropped(material: 0, layers: ["subsurface scattering"])))
    }
}
