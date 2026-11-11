import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MeshLightmapBakerTests {
    private func pixel(_ map: MeshLightmapBaker, x: Int, y: Int) -> SIMD4<Float> {
        var value = SIMD4<Float16>.zero
        map.irradianceTexture.getBytes(
            &value, bytesPerRow: MemoryLayout<SIMD4<Float16>>.stride,
            from: MTLRegionMake2D(x, y, 1, 1), mipmapLevel: 0
        )
        return SIMD4<Float>(value)
    }

    private func receiver() throws -> MeshGeometry {
        let plane = try MeshGeometry.plane(width: 2, height: 2)
        return try MeshGeometry(vertices: plane.vertices.map { vertex in
            let uv = vertex.textureCoordinate
            return MeshVertex(position: vertex.position, normal: vertex.normal, textureCoordinate: uv,
                              secondaryTextureCoordinate: SIMD2(1 - uv.y, uv.x))
        }, indices: plane.indices)
    }

    private func blocker() throws -> MeshGeometry.MergePart {
        let plane = try MeshGeometry.plane(width: 1, height: 2)
        let degenerateUV = try MeshGeometry(vertices: plane.vertices.map { vertex in
            MeshVertex(position: vertex.position, normal: vertex.normal, secondaryTextureCoordinate: .zero)
        }, indices: plane.indices)
        return .init(degenerateUV, transform: SceneTransform(translation: SIMD3(-0.5, 0, 0.2)).matrix)
    }

    @Test("Directional irradiance sums cosines and rejects light from behind the receiver")
    func directionalShadow() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let geometry = try receiver()
        let key = SIMD3<Float>(0.75, 0.5, 0.125)
        let fill = SIMD3<Float>(0.125, 0.25, 0.5)
        let lights: [MeshLightmapBaker.DirectionalLight] = [
            try .init(directionToLight: SIMD3(0, 0, 1), irradiance: key),
            try .init(directionToLight: SIMD3(0.6, 0, 0.8), irradiance: fill),
            try .init(directionToLight: SIMD3(0, 0, -1), irradiance: SIMD3(repeating: 10)),
        ]
        let expected = SIMD4<Float>(SIMD4<Float16>(SIMD4(key + fill * 0.8, 1)))
        let dark = SIMD4<Float>(0, 0, 0, 1)
        let caster = try blocker()
        let blocked = try MeshLightmapBaker(
            device: device, geometry: geometry, width: 16, height: 16, lights: lights, casters: [caster]
        )
        let noCaster = try MeshLightmapBaker(
            device: device, geometry: geometry, width: 16, height: 16, lights: lights, receiverCastsShadows: false
        )
        let receiverOnly = try MeshLightmapBaker(device: device, geometry: geometry, width: 16, height: 16, lights: lights)

        #expect(pixel(blocked, x: 8, y: 4) == dark)
        #expect(pixel(blocked, x: 8, y: 12) == expected)
        #expect(pixel(noCaster, x: 8, y: 4) == expected)
        #expect(pixel(receiverOnly, x: 8, y: 4) == expected)

        // The same blocker can be part of the receiver geometry, with a zero-area UV chart.
        let combined = try MeshGeometry.merged([.init(geometry), caster]).geometry
        let selfShadow = try MeshLightmapBaker(device: device, geometry: combined, width: 16, height: 16, lights: lights)
        let selfOff = try MeshLightmapBaker(
            device: device, geometry: combined, width: 16, height: 16, lights: lights, receiverCastsShadows: false
        )
        #expect(pixel(selfShadow, x: 8, y: 4) == dark)
        #expect(pixel(selfShadow, x: 8, y: 12) == expected)
        #expect(pixel(selfOff, x: 8, y: 4) == expected)
    }

    @Test("A transformed receiver and caster keep world-space incidence and visibility under reflection")
    func worldTransform() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let geometry = try receiver()
        let pose = SceneTransform(
            translation: SIMD3(4, 2, -5), rotation: simd_quatf(angle: 0.53, axis: SIMD3(0, 1, 0)),
            scale: SIMD3(-2, 0.75, 1.5)
        )
        let caster = try blocker()
        let irradiance = SIMD3<Float>(0.5, 1, 2)
        let lights = [try MeshLightmapBaker.DirectionalLight(
            directionToLight: pose.rotation.act(SIMD3(0, 0, 1)), irradiance: irradiance
        )]
        let map = try MeshLightmapBaker(
            device: device, geometry: geometry, width: 16, height: 16, worldFromObject: pose.matrix,
            lights: lights, casters: [.init(caster.geometry, transform: pose.matrix * caster.transform)]
        )
        #expect(pixel(map, x: 8, y: 4) == SIMD4(0, 0, 0, 1))
        #expect(pixel(map, x: 8, y: 12) == SIMD4(irradiance, 1))
    }

    @Test("A baked UV1 map reproduces Lambertian radiance and contains its baked shadow")
    func materialWorkflow() throws {
        let fixture = try GeometryRenderFixture()
        let geometry = try receiver()
        let color = SIMD3<Float>(0.75, 0.5, 0.25)
        let albedo = SIMD3<Float>(0.8, 0.6, 0.4)
        let direction = SIMD3<Float>(0, 0, 1)
        let lights = [try MeshLightmapBaker.DirectionalLight(directionToLight: direction, irradiance: color)]
        let noCaster = try MeshLightmapBaker(
            device: fixture.device, geometry: geometry, width: 32, height: 32, lights: lights
        )
        let occluded = try MeshLightmapBaker(
            device: fixture.device, geometry: geometry, width: 32, height: 32, lights: lights, casters: [blocker()]
        )

        func render(_ material: PhysicallyBasedMaterial, lights: [SceneNode] = []) throws -> [SIMD4<Float>] {
            let mesh = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(material)
            )
            let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)
            return try fixture.render(SceneGraph(roots: [node] + lights, camera: fixture.camera)).pixels
        }
        func bakedMaterial(_ map: MeshLightmapBaker) throws -> PhysicallyBasedMaterial {
            try PhysicallyBasedMaterial(
                lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
                emissiveTexture: .init(texture: map.irradianceTexture, textureCoordinateSet: map.textureCoordinateSet),
                emissiveFactor: albedo / .pi, alphaMode: .opaque
            )
        }

        let authored = try render(
            PhysicallyBasedMaterial(
                lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(albedo, 1),
                alphaMode: .opaque, specular: .init(factor: 0)
            ), lights: [SceneNode(light: try PunctualLight(kind: .directional, color: color, intensity: 1))]
        )
        let baked = try render(bakedMaterial(noCaster))
        let shadowed = try render(bakedMaterial(occluded))
        let expected = SIMD4(albedo * color / .pi, 1)
        for column in [20, 26, 38, 44] {
            let direct = fixture.pixel(authored, x: column, y: 32)
            let unblocked = fixture.pixel(baked, x: column, y: 32)
            #expect(simd_reduce_max(abs(direct - expected)) < 0.001)
            #expect(simd_reduce_max(abs(direct - unblocked)) < 0.001)
            let withShadow = fixture.pixel(shadowed, x: column, y: 32)
            let shadowReference = column < 32 ? SIMD4<Float>(0, 0, 0, 1) : expected
            #expect(simd_reduce_max(abs(withShadow - shadowReference)) < 0.001)
        }
    }

    @Test("The lightmap rejects invalid lights, bias and unrepresentable irradiance")
    func validation() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let geometry = try receiver()
        #expect(throws: MeshLightmapBaker.Failure.light) {
            try MeshLightmapBaker.DirectionalLight(directionToLight: .zero, irradiance: SIMD3(repeating: 1))
        }
        #expect(throws: MeshLightmapBaker.Failure.shadowBias) {
            try MeshLightmapBaker(device: device, geometry: geometry, width: 4, height: 4, lights: [], shadowBias: -1)
        }
        let light = try MeshLightmapBaker.DirectionalLight(
            directionToLight: SIMD3(0, 0, 1), irradiance: SIMD3(repeating: 40_000)
        )
        #expect(throws: MeshLightmapBaker.Failure.lightCount) {
            try MeshLightmapBaker(device: device, geometry: geometry, width: 4, height: 4, lights: Array(repeating: light, count: 17))
        }
        #expect(throws: MeshLightmapBaker.Failure.irradianceRange) {
            try MeshLightmapBaker(device: device, geometry: geometry, width: 4, height: 4, lights: [light, light])
        }
    }
}
