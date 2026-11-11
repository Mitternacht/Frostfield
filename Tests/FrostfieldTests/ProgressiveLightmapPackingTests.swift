@testable import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ProgressiveLightmapPackingTests {
    @Test("A shared atlas preserves authored UV1 transport and cross-lights reflected receiver meshes")
    func crossLighting() throws {
        let fixture = try GeometryRenderFixture()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 2, height: 2, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(fixture.device.makeTexture(descriptor: descriptor))
        // Reusing packed UV1 for transport would pick green or the wrong receiver color.
        let colors: [SIMD4<UInt8>] = [
            SIMD4(255, 0, 0, 255), SIMD4(0, 255, 0, 255),
            SIMD4(0, 255, 0, 255), SIMD4(0, 0, 255, 255),
        ]
        colors.withUnsafeBytes {
            texture.replace(region: MTLRegionMake2D(0, 0, 2, 2), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 8)
        }
        let map = try ProgressiveLightmapBaker.TextureSnapshot(.init(
            texture: texture, textureCoordinateSet: .secondary, sampling: .init(magnification: .nearest)
        ))
        let plane = try MeshGeometry.plane(width: 4, height: 4)
        func receiver(uv: SIMD2<Float>, world: simd_float4x4) throws -> ProgressiveLightmapBaker.Surface {
            let geometry = try MeshGeometry(vertices: plane.vertices.map {
                MeshVertex(position: $0.position, normal: $0.normal, textureCoordinate: $0.textureCoordinate,
                           secondaryTextureCoordinate: uv, tangent: $0.tangent)
            }, indices: plane.indices)
            return try .init(geometry: geometry, worldFromObject: world, diffuseAlbedo: SIMD3(repeating: 0.8), diffuseTexture: map)
        }
        let lower = try receiver(uv: SIMD2(0.25, 0.25), world: SceneTransform(scale: SIMD3(-1, 1.25, 1)).matrix)
        let upper = try receiver(uv: SIMD2(0.75, 0.75), world: simd_float4x4(columns: (
            SIMD4(1, 0, 0, 0), SIMD4(0, -1.25, 0, 0), SIMD4(0, 0, -1, 0), SIMD4(0, 0, 1, 1)
        )))
        let lights: [MeshLightmapBaker.DirectionalLight] = try [
            .init(directionToLight: SIMD3(5, 0, 1), irradiance: SIMD3(repeating: 1)),
            .init(directionToLight: SIMD3(-5, 0, -1), irradiance: SIMD3(repeating: 1)),
            .init(directionToLight: SIMD3(0, 0, 1), irradiance: SIMD3(repeating: 10)),
            .init(directionToLight: SIMD3(0, 0, -1), irradiance: SIMD3(repeating: 10)),
        ]
        let resolution = 32
        let baker = try ProgressiveLightmapBaker(
            device: fixture.device, receivers: [lower, upper], resolution: resolution,
            lights: lights, maximumBounces: 1, seed: 19, padding: 2
        )
        #expect(baker.textureCoordinateSet == .secondary)
        #expect(baker.receiverBindings.count == 2)
        let lowerSamples = try covered(baker.receiverBindings[0].geometry, resolution: resolution)
        let upperSamples = try covered(baker.receiverBindings[1].geometry, resolution: resolution)
        let lowerPixels = Set(lowerSamples.keys)
        let upperPixels = Set(upperSamples.keys)
        #expect(lowerPixels.count > 100 && upperPixels.count > 100)
        #expect(lowerPixels.isDisjoint(with: upperPixels))
        let allInterior = lowerPixels.union(upperPixels)
        let reference = try baker.snapshot()
        let coverage: [UInt8] = read(reference.coverageTexture, as: UInt8.self, zero: 0)
        #expect(Set(coverage.indices.filter { coverage[$0] == 255 }) == allInterior)
        #expect(coverage.contains(128))
        let direct = readIrradiance(reference.irradianceTexture)
        // Each receiver blocks the opposing normal-incidence light. The
        // unobstructed grazing light contributes 1 / √26 on either face.
        let expected = Float(1 / Double(26).squareRoot())
        let unexpected = allInterior.filter { simd_distance(direct[$0], SIMD4(expected, expected, expected, 1)) >= 0.001 }
        #expect(unexpected.isEmpty, "Unexpected irradiance at \(unexpected.sorted().map { (lowerSamples[$0] ?? upperSamples[$0])! })")

        try baker.accumulate(samples: 16)
        try baker.accumulate(samples: 48)
        let final = try baker.snapshot()
        let output = readIrradiance(final.irradianceTexture)
        let blueGain = lowerPixels.reduce(Float(0)) { $0 + output[$1].z - direct[$1].z } / Float(lowerPixels.count)
        let redGain = upperPixels.reduce(Float(0)) { $0 + output[$1].x - direct[$1].x } / Float(upperPixels.count)
        #expect(blueGain > 0.06 && redGain > 0.06)
        #expect(lowerPixels.allSatisfy { output[$0].x == direct[$0].x && output[$0].y == direct[$0].y })
        #expect(upperPixels.allSatisfy { output[$0].y == direct[$0].y && output[$0].z == direct[$0].z })
        #expect(readIrradiance(reference.irradianceTexture) == direct)
        baker.reset()
        try baker.accumulate(samples: 64)
        #expect(try readIrradiance(baker.snapshot().irradianceTexture) == output)

        let binding = try final.lightMapBinding()
        #expect(binding.texture === final.irradianceTexture)
        #expect(binding.textureCoordinateSet == .secondary)
        let material = try ClassicMaterial(model: .basic, color: SIMD3(repeating: 1), lightMap: binding)
        func center(_ receiver: ProgressiveLightmapBaker.ReceiverBinding) throws -> SIMD4<Float> {
            let renderer = try MeshRenderer(device: fixture.device, colorPixelFormat: .rgba16Float,
                                            geometry: receiver.geometry, material: .classic(material))
            let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: renderer)
            let raster = try fixture.render(SceneGraph(roots: [node], camera: fixture.camera))
            return fixture.pixel(raster.pixels, x: 32, y: 32)
        }
        let blue = try center(baker.receiverBindings[0])
        let red = try center(baker.receiverBindings[1])
        #expect(blue.z > blue.x + 0.02 && abs(blue.x - blue.y) < 0.001)
        #expect(red.x > red.z + 0.02 && abs(red.y - red.z) < 0.001)
    }

    @Test("Chart splits map back to original local vertices, winding, and material groups")
    func splitChartSourceMapping() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let points: [SIMD3<Float>] = [.zero, SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(-1, 0, 0), SIMD3(0, -1, 0)]
        let vertices = points.enumerated().map { index, point in
            MeshVertex(position: point, normal: SIMD3(0, 0, 1), textureCoordinate: SIMD2(Float(index) / 4, 0.25),
                       secondaryTextureCoordinate: SIMD2(0.125, 0.625), tangent: SIMD4(1, 0, 0, 1),
                       color: SIMD4(Float(index) / 4, 0.5, 0, 1))
        }
        // The triangles meet only at vertex zero, so that vertex is split between charts.
        let source = try MeshGeometry(vertices: vertices, indices: [0, 1, 2, 0, 3, 4]).withGroups([
            .init(start: 0, count: 3, materialIndex: 4), .init(start: 3, count: 3, materialIndex: 9),
        ])
        let first = try ProgressiveLightmapBaker.Surface(
            geometry: source, worldFromObject: SceneTransform(scale: SIMD3(-1, 2, 1)).matrix
        )
        let second = try ProgressiveLightmapBaker.Surface(
            geometry: source, worldFromObject: SceneTransform(translation: SIMD3(3, 0, 1)).matrix
        )
        let baker = try ProgressiveLightmapBaker(device: device, receivers: [first, second], resolution: 48, padding: 2)
        for (binding, input) in zip(baker.receiverBindings, [first, second]) {
            let indices = try #require(binding.sourceVertexIndices)
            #expect(indices.filter { $0 == 0 }.count == 2)
            #expect(binding.worldFromObject == input.worldFromObject)
            #expect(binding.geometry.groups == source.groups)
            #expect(binding.geometry.indices.map { indices[Int($0)] } == source.indices.map(Int.init))
            #expect(binding.geometry.vertices.contains { $0.secondaryTextureCoordinate != vertices[0].secondaryTextureCoordinate })
            for (vertex, index) in zip(binding.geometry.vertices, indices) {
                let input = vertices[index]
                #expect(vertex.position == input.position && vertex.normal == input.normal)
                #expect(vertex.textureCoordinate == input.textureCoordinate)
                #expect(vertex.tangent == input.tangent && vertex.color == input.color)
                let uv = try #require(vertex.secondaryTextureCoordinate)
                #expect((0...1).contains(uv.x) && (0...1).contains(uv.y))
            }
        }
        #expect(try Set(covered(baker.receiverBindings[0].geometry, resolution: 48).keys)
            .isDisjoint(with: covered(baker.receiverBindings[1].geometry, resolution: 48).keys))
    }

    @Test("Shared atlas validates receiver count, resolution and the existing geometry budget")
    func invalidInputs() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let receiver = try ProgressiveLightmapBaker.Surface(geometry: .plane(width: 1, height: 1))
        #expect(throws: ProgressiveLightmapBaker.Failure.receiverCount) {
            try ProgressiveLightmapBaker(device: device, receivers: [], resolution: 32)
        }
        #expect(throws: MeshGeometry.UVAtlasError.resolution) {
            try ProgressiveLightmapBaker(device: device, receivers: [receiver], resolution: 15)
        }
        #expect(throws: ProgressiveLightmapBaker.Failure.texelLimit) {
            try ProgressiveLightmapBaker(device: device, receivers: [receiver], resolution: 513)
        }
        #expect(throws: ProgressiveLightmapBaker.Failure.geometryLimit) {
            try ProgressiveLightmapBaker(
                device: device, receivers: [receiver], resolution: 32,
                surfaces: Array(repeating: receiver, count: ProgressiveLightmapBaker.maximumSurfaceCount)
            )
        }
        let authored = try ProgressiveLightmapBaker(
            device: device, receiver: receiver, width: 8, height: 8, textureCoordinateSet: .primary
        )
        #expect(authored.receiverBindings.count == 1)
        #expect(authored.receiverBindings[0].sourceVertexIndices == nil)
        #expect(authored.receiverBindings[0].geometry.indices == receiver.geometry.indices)
    }

    private func covered(_ geometry: MeshGeometry, resolution: Int) throws -> [Int: SIMD3<Double>] {
        var atlas = try UVBakeAtlas(width: resolution, height: resolution, padding: 0)
        let coordinates = try atlas.coordinates(of: geometry, set: .secondary)
        var pixels: [Int: SIMD3<Double>] = [:]
        for start in stride(from: 0, to: geometry.indices.count, by: 3) {
            let a = Int(geometry.indices[start]), b = Int(geometry.indices[start + 1]), c = Int(geometry.indices[start + 2])
            let pa = SIMD3<Double>(geometry.vertices[a].position)
            let pb = SIMD3<Double>(geometry.vertices[b].position)
            let pc = SIMD3<Double>(geometry.vertices[c].position)
            try atlas.rasterize(
                coordinates[a], coordinates[b], coordinates[c], triangle: start / 3, overlap: .reject
            ) { index, weights in
                pixels[index] = pa * weights.x + pb * weights.y + pc * weights.z
            }
        }
        return pixels
    }

    private func readIrradiance(_ texture: any MTLTexture) -> [SIMD4<Float>] {
        read(texture, as: SIMD4<Float16>.self, zero: .zero).map { SIMD4<Float>($0) }
    }

    private func read<T>(_ texture: any MTLTexture, as: T.Type, zero: T) -> [T] {
        var pixels = [T](repeating: zero, count: texture.width * texture.height)
        pixels.withUnsafeMutableBytes {
            texture.getBytes($0.baseAddress!, bytesPerRow: texture.width * MemoryLayout<T>.stride,
                             from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0)
        }
        return pixels
    }
}
