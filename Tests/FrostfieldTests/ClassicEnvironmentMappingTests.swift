import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ClassicEnvironmentMappingTests {
    private let fixture: GeometryRenderFixture
    private let tilt = simd_quatf(angle: 0.35 * .pi, axis: SIMD3(0, 1, 0))
    // +X, -X, +Y, -Y, +Z, -Z, with distinct linear HDR colors.
    private let faces: [SIMD3<Float>] = [
        SIMD3(2, 0.125, 0), SIMD3(0.25, 3, 0), SIMD3(0, 0, 4),
        SIMD3(0.5, 0.25, 0), SIMD3(1.25, 0.25, 0.5), SIMD3(0.125, 0.5, 2)
    ]

    init() throws {
        fixture = try GeometryRenderFixture(side: 9)
    }

    @Test("Classic cube reflection and refraction select different directions and share yaw")
    func directions() throws {
        let texture = try cube()
        func pixel(_ mapping: ClassicMaterial.EnvironmentMapping, yaw: Float = 0) throws -> SIMD4<Float> {
            let map = try ClassicMaterial.EnvironmentMap(texture: texture, blend: .mix, yaw: yaw, mapping: mapping)
            return try center(scene(.init(model: .basic, environmentMap: map), rotation: tilt))
        }

        // At 63 degrees from the normal, reflection points at +X. A small
        // air/material IOR ratio bends into -X. Unit ratio follows the ray into -Z.
        #expect(try pixel(.reflection) == SIMD4(faces[0], 1))
        #expect(try pixel(.refraction(ratio: 0.25)) == SIMD4(faces[1], 1))
        #expect(try pixel(.refraction(ratio: 1)) == SIMD4(faces[5], 1))
        #expect(try pixel(.refraction()) == SIMD4(faces[5], 1))
        #expect(try pixel(.refraction(ratio: 1), yaw: .pi / 2) == SIMD4(faces[0], 1))
        for ratio: Float in [-0.01, 1.01, .nan, .infinity] {
            #expect(throws: ClassicMaterial.Failure.invalidValue("refractionRatio")) {
                try ClassicMaterial.EnvironmentMap(texture: texture, mapping: .refraction(ratio: ratio))
            }
        }
    }

    @Test("Basic, Lambert and Phong combine refracted HDR with the shaded color")
    func combine() throws {
        let texture = try cube()
        let color = SIMD3<Float>(0.25, 0.5, 0.75)
        let intensity: Float = 1.5
        let reflectivity: Float = 0.375
        let radiance = faces[5] * intensity
        let cases: [(ClassicMaterial.EnvironmentBlend, SIMD3<Float>)] = [
            (.multiply, color * (1 - reflectivity) + color * radiance * reflectivity),
            (.mix, color * (1 - reflectivity) + radiance * reflectivity),
            (.add, color + radiance * reflectivity)
        ]
        for model: ClassicMaterial.Model in [.basic, .lambert, .phong()] {
            let ambient: SIMD3<Float> = if case .basic = model { .zero } else { SIMD3(repeating: .pi) }
            let base = try center(scene(.init(model: model, color: color, ambient: ambient)))
            #expect(close(base, SIMD4(color, 1)))
            for (blend, expected) in cases {
                func pixel(_ reflectivity: Float) throws -> SIMD4<Float> {
                    let map = try ClassicMaterial.EnvironmentMap(
                        texture: texture, blend: blend, reflectivity: reflectivity, intensity: intensity,
                        mapping: .refraction(ratio: 1)
                    )
                    return try center(scene(.init(model: model, color: color, environmentMap: map, ambient: ambient)))
                }
                #expect(close(try pixel(reflectivity), SIMD4(expected, 1)))
                #expect(try pixel(0) == base)
            }
        }
    }

    @Test("Explicit orthographic camera direction keeps refracted lookups parallel")
    func orthographicDirections() throws {
        let map = try ClassicMaterial.EnvironmentMap(
            texture: cube(), blend: .mix, mapping: .refraction(ratio: 1)
        )
        let graph = try scene(.init(model: .basic, environmentMap: map))
        let clipFromWorld = simd_float4x4(diagonal: SIMD4(0.25, 0.25, -0.1, 1))
        func image(directionToCamera: SIMD3<Float>?) throws -> [SIMD4<Float>] {
            let target = try fixture.makeTarget()
            let buffer = try #require(fixture.queue.makeCommandBuffer())
            try graph.encode(
                pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: buffer,
                clipFromWorld: clipFromWorld, cameraPosition: .zero, directionToCamera: directionToCamera
            )
            try fixture.finish(buffer)
            return try fixture.pixels(of: target.color)
        }
        let ortho = try image(directionToCamera: SIMD3(0, 0, 1))
        let fromEye = try image(directionToCamera: nil)
        for x in [1, 4, 7] {
            #expect(fixture.pixel(ortho, x: x, y: 4) == SIMD4(faces[5], 1))
        }
        #expect(fixture.pixel(fromEye, x: 1, y: 4) == SIMD4(faces[1], 1))
        #expect(fixture.pixel(fromEye, x: 7, y: 4) == SIMD4(faces[0], 1))
    }

    @Test("Classic environment archives preserve old reflection and version refraction")
    func archive() throws {
        let texture = try cube()
        let legacy = try ClassicMaterial.EnvironmentMap(texture: texture, blend: .mix)
        let refraction = try ClassicMaterial.EnvironmentMap(texture: texture, blend: .mix, mapping: .refraction(ratio: 0))
        for (map, version, face) in [(legacy, 1, 0), (refraction, 24, 1)] {
            let graph = try scene(.init(model: .basic, environmentMap: map), rotation: tilt)
            let archive = try SceneArchive(capturing: graph)
            let json = try #require(try JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
            #expect(json["version"] as? Int == version)
            let restored = try archive.load(device: fixture.device).graph
            #expect(try center(restored) == center(graph))
            #expect(try center(restored) == SIMD4(faces[face], 1))
            if version == 24 {
                var incompatible = json
                incompatible["version"] = 23
                #expect(throws: SceneArchive.Failure.decoding("classic environment refraction requires archive version 24")) {
                    try SceneArchive(manifest: JSONSerialization.data(withJSONObject: incompatible), binary: archive.binary)
                }
            }
        }
    }

    private func scene(_ material: ClassicMaterial, rotation: simd_quatf? = nil) throws -> SceneGraph {
        let renderer = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float,
            geometry: GeometryRenderFixture.square(4), material: .classic(material)
        )
        let transform = SceneTransform(
            translation: SIMD3(0, 0, -2), rotation: rotation ?? simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))
        )
        return SceneGraph(roots: [SceneNode(transform: transform, mesh: renderer)], camera: fixture.camera)
    }

    private func center(_ scene: SceneGraph) throws -> SIMD4<Float> {
        fixture.pixel(try fixture.render(scene).pixels, x: 4, y: 4)
    }

    private func cube() throws -> any MTLTexture {
        let side = 16
        let descriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: .rgba16Float, size: side, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let rowBytes = side * MemoryLayout<SIMD4<Float16>>.stride
        for (face, rgb) in faces.enumerated() {
            let pixels = Array(repeating: SIMD4<Float16>(SIMD4(rgb, 0)), count: side * side)
            pixels.withUnsafeBytes { bytes in
                texture.replace(
                    region: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0, slice: face,
                    withBytes: bytes.baseAddress!, bytesPerRow: rowBytes, bytesPerImage: rowBytes * side
                )
            }
        }
        return texture
    }

    private func close(_ actual: SIMD4<Float>, _ expected: SIMD4<Float>) -> Bool {
        simd_reduce_max(abs(actual - expected)) < 0.001
    }
}
