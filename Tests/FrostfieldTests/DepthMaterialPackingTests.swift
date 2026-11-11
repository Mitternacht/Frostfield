import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct DepthMaterialPackingTests {
    private let gpu: GeometryRenderFixture
    private let geometry: MeshGeometry

    init() throws {
        gpu = try GeometryRenderFixture(side: 17)
        geometry = try GeometryRenderFixture.square(0.9)
    }

    @Test("Depth material writes the r187 Basic, RGBA, RGB and RG bytes at endpoints and fractional depths")
    func depthPacking() throws {
        let modes = ClassicMaterial.DepthPacking.allCases
        #expect(modes.map(\.rawValue) == [3200, 3201, 3202, 3203])
        let scenes = try modes.map {
            try scene(ClassicMaterial(model: .depth, opacity: 0.25, depthPacking: $0))
        }
        // Float32 inputs and exact RGBA8 bytes from the pinned r187 packing chunk.
        let cases: [(Float, [SIMD4<UInt8>])] = [
            (0, [SIMD4(255, 255, 255, 64), .zero, SIMD4(0, 0, 0, 255), SIMD4(0, 0, 0, 255)]),
            (1, [SIMD4(0, 0, 0, 64), SIMD4(repeating: 255), SIMD4(repeating: 255), SIMD4(255, 255, 0, 255)]),
            (0.1234567, [SIMD4(224, 224, 224, 64), SIMD4(31, 154, 219, 191), SIMD4(31, 154, 219, 255), SIMD4(31, 154, 0, 255)]),
            (0.0001234567, [SIMD4(255, 255, 255, 64), SIMD4(0, 8, 23, 66), SIMD4(0, 8, 23, 255), SIMD4(0, 8, 0, 255)]),
            (0x1p-26, [SIMD4(255, 255, 255, 64), SIMD4(0, 0, 0, 64), SIMD4(0, 0, 0, 255), SIMD4(0, 0, 0, 255)]),
            (Float(1).nextDown, [
                SIMD4(0, 0, 0, 64), SIMD4(255, 255, 255, 0), SIMD4(255, 255, 254, 255), SIMD4(255, 255, 0, 255),
            ]),
        ]
        for (depth, expected) in cases {
            for (index, graph) in scenes.enumerated() {
                #expect(try pixel(graph, depth: depth) == expected[index], "Mode \(modes[index]), depth \(depth)")
            }
        }
    }

    @Test("Reversed depth packs raw window Z and explicit Basic inverts it, while legacy stays near-white")
    func reversedDepth() throws {
        let legacy = try ClassicMaterial(model: .depth)
        let basic = try ClassicMaterial(model: .depth, opacity: 0.25, depthPacking: .basic)
        let packed = try ClassicMaterial(model: .depth, depthPacking: .rgba)
        for (material, forward, reversed): (ClassicMaterial, SIMD4<UInt8>, SIMD4<UInt8>) in [
            (legacy, SIMD4(191, 191, 191, 255), SIMD4(191, 191, 191, 255)),
            (basic, SIMD4(191, 191, 191, 64), SIMD4(64, 64, 64, 64)),
            (packed, SIMD4(64, 0, 0, 0), SIMD4(192, 0, 0, 0)),
        ] {
            #expect(try pixel(scene(material), depth: 0.25) == forward)
            #expect(try pixel(scene(material, depthMode: .reversed), depth: 0.75, depthMode: .reversed) == reversed)
        }
    }

    @Test("r187 distance output writes the clamped radial distance only in red")
    func distanceOutput() throws {
        let material = try ClassicMaterial(model: .distance(reference: .zero, near: 1, far: 5), distanceOutput: .red)
        var graph = try scene(material)
        for (z, expected): (Float, UInt8) in [(0, 0), (-3, 128), (-7, 255)] {
            graph.roots[0].transform.translation.z = z
            #expect(try pixel(graph, depth: 0.125) == SIMD4(expected, 0, 0, 255))
        }
        graph.roots[0].transform.translation.z = -3
        #expect(try pixel(graph, depth: 0.125, x: 13) == SIMD4(131, 0, 0, 255))
    }

    @Test("Depth packing retains cutout and rejects incompatible material and pipeline settings")
    func coverageAndValidation() throws {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let map = try #require(gpu.device.makeTexture(descriptor: descriptor))
        func setAlpha(_ alpha: UInt8) {
            let bytes: [UInt8] = [255, 255, 255, alpha]
            map.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: bytes, bytesPerRow: 4)
        }
        let material = try ClassicMaterial(model: .depth, map: .init(texture: map), alphaTest: 0.5, depthPacking: .rgba)
        let cutout = try scene(material)
        setAlpha(0)
        #expect(try pixel(cutout, depth: 0.1234567) == SIMD4(255, 0, 255, 255))
        setAlpha(255)
        #expect(try pixel(cutout, depth: 0.1234567) == SIMD4(31, 154, 219, 191))

        #expect(throws: ClassicMaterial.Failure.unusedProperty("depthPacking")) {
            try ClassicMaterial(model: .basic, depthPacking: .rgba)
        }
        #expect(throws: ClassicMaterial.Failure.unusedProperty("distanceOutput")) {
            try ClassicMaterial(model: .depth, distanceOutput: .red)
        }
        for format: MTLPixelFormat in [.rgba8Unorm_srgb, .rg8Unorm] {
            #expect(throws: MeshRenderer.Failure.depthOutputPixelFormat) {
                try MeshRenderer(device: gpu.device, colorPixelFormat: format, geometry: geometry, material: .classic(material))
            }
        }
        #expect(throws: MeshRenderer.Failure.renderState("Explicit depth and distance outputs require a single sample and no blending.")) {
            try MeshRenderer(
                device: gpu.device, colorPixelFormat: .rgba8Unorm, geometry: geometry,
                material: .classic(material), renderState: .transparent
            )
        }
    }

    @Test("Version 13 preserves explicit depth and distance settings and their exact rendered bytes")
    func archive() throws {
        let legacy = try scene(ClassicMaterial(model: .depth))
        let oldArchive = try SceneArchive(capturing: legacy)
        #expect(try manifest(oldArchive)["version"] as? Int == 1)
        let restoredLegacy = try oldArchive.load(device: gpu.device).graph
        #expect(try pixel(restoredLegacy, depth: 0.25) == SIMD4(191, 191, 191, 255))

        let depth = try ClassicMaterial(model: .depth, depthPacking: .rgba)
        let distance = try ClassicMaterial(model: .distance(reference: SIMD3(0, 0, 3), near: 1, far: 5), distanceOutput: .red)
        var graph = try scene(depth)
        graph.roots += try scene(distance).roots
        let archive = try SceneArchive(capturing: graph)
        var document = try manifest(archive)
        #expect(document["version"] as? Int == 13)
        let restored = try archive.load(device: gpu.device).graph
        #expect(try SceneArchive(capturing: restored) == archive)
        let expected = [SIMD4<UInt8>(31, 154, 219, 191), SIMD4<UInt8>(128, 0, 0, 255)]
        for index in expected.indices {
            let original = SceneGraph(roots: [graph.roots[index]], camera: gpu.camera)
            let decoded = SceneGraph(roots: [restored.roots[index]], camera: gpu.camera)
            #expect(try pixel(original, depth: 0.1234567) == expected[index])
            #expect(try pixel(decoded, depth: 0.1234567) == expected[index])
        }

        document["version"] = 12
        #expect(throws: SceneArchive.Failure.decoding("depth and distance outputs require archive version 13")) {
            try SceneArchive(manifest: JSONSerialization.data(withJSONObject: document), binary: archive.binary)
        }
    }

    private func scene(_ material: ClassicMaterial, depthMode: MeshDepthMode = .forward) throws -> SceneGraph {
        let mesh = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba8Unorm, geometry: geometry, material: .classic(material),
            depthMode: depthMode, renderState: .init(depthCompare: .always)
        )
        return SceneGraph(roots: [SceneNode(mesh: mesh, frustumCulled: false)], camera: gpu.camera)
    }

    private func pixel(
        _ graph: SceneGraph, depth: Float, depthMode: MeshDepthMode = .forward, x: Int = 8
    ) throws -> SIMD4<UInt8> {
        let targets = try gpu.makeTarget()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: gpu.side, height: gpu.side, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .renderTarget
        let color = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let pass = gpu.pass(color: color, depth: targets.depth)
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
        pass.depthAttachment.clearDepth = depthMode == .reversed ? 0 : 1
        let command = try #require(gpu.queue.makeCommandBuffer())
        var projection = matrix_identity_float4x4
        projection.columns.2.z = -0.0625
        projection.columns.3.z = depth
        try graph.encode(
            pass: pass, commandBuffer: command, clipFromWorld: projection,
            cameraPosition: SIMD3(0, 0, 2), directionToCamera: SIMD3(0, 0, 1), depthMode: depthMode
        )
        try gpu.finish(command)
        var result = SIMD4<UInt8>.zero
        try withUnsafeMutableBytes(of: &result) { bytes in
            color.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: 4,
                from: MTLRegionMake2D(x, 8, 1, 1), mipmapLevel: 0
            )
        }
        return result
    }

    private func manifest(_ archive: SceneArchive) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
    }
}
