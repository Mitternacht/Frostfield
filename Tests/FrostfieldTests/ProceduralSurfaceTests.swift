import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ProceduralSurfaceTests {
    @Test
    func generatesRepeatableCoordinatedMapsWithMipmaps() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let first = try ProceduralSurface(device: device, style: .leatherette, resolution: 64, seed: 82)
        let repeatSurface = try ProceduralSurface(device: device, style: .leatherette, resolution: 64, seed: 82)
        let varied = try ProceduralSurface(device: device, style: .leatherette, resolution: 64, seed: 83)

        #expect(first.baseColorTexture.pixelFormat == .rgba8Unorm_srgb)
        #expect(first.normalTexture.pixelFormat == .rgba8Unorm)
        #expect(first.metallicRoughnessTexture.pixelFormat == .rgba8Unorm)
        for (a, b) in zip(
            [first.baseColorTexture, first.normalTexture, first.metallicRoughnessTexture],
            [repeatSurface.baseColorTexture, repeatSurface.normalTexture, repeatSurface.metallicRoughnessTexture]
        ) {
            #expect(a.mipmapLevelCount == 7)
            #expect(pixels(a) == pixels(b))
            #expect(pixels(a, level: 6)[3] == 255)
        }
        #expect(pixels(first.baseColorTexture) != pixels(varied.baseColorTexture))

        let roughness = pixels(first.metallicRoughnessTexture)
        let starts = stride(from: 0, to: roughness.count, by: 4)
        #expect(starts.allSatisfy { roughness[$0 + 2] == 0 })
        #expect(Set(starts.map { roughness[$0 + 1] }).count > 12)
        let material = try first.material()
        #expect(material.baseColorTexture === first.baseColorTexture)
        #expect(material.alphaMode == .opaque)
    }

    @Test
    func brushedMetalNormalDetailHasAConsistentGrainDirection() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let surface = try ProceduralSurface(device: device, style: .brushedMetal, resolution: 128, seed: 9)
        let normals = pixels(surface.normalTexture)
        let packed = pixels(surface.metallicRoughnessTexture)
        let starts = stride(from: 0, to: normals.count, by: 4)
        let alongGrain = starts.reduce(0) { $0 + abs(Int(normals[$1]) - 128) }
        let acrossGrain = starts.reduce(0) { $0 + abs(Int(normals[$1 + 1]) - 128) }
        #expect(acrossGrain > 3 * alongGrain)
        #expect(starts.allSatisfy { packed[$0 + 2] == 255 })
    }

    @Test
    func rejectsUnboundedResolutionAndNonfiniteColor() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(throws: ProceduralSurface.Failure.resolution) {
            try ProceduralSurface(device: device, style: .wood, resolution: 1025, seed: 0)
        }
        #expect(throws: ProceduralSurface.Failure.color) {
            try ProceduralSurface(device: device, style: .wood, seed: 0, baseColor: SIMD3(0, .nan, 0))
        }
    }

    @Test
    func samplesAuthoredChannelsAndDerivesWrappingNormals() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        var coordinates: [SIMD2<Float>] = []
        let surface = try ProceduralSurface(device: device, resolution: 32, heightScale: 0.2) { uv in
            coordinates.append(uv)
            return .init(
                baseColor: SIMD3(0.25, 0.5, 0.75),
                height: 0.5 + 0.25 * sin(2 * .pi * uv.x),
                roughness: uv.y, metallic: 0.5, occlusion: 0.25
            )
        }

        #expect(coordinates.count == 32 * 32)
        #expect(coordinates.first == SIMD2(1.0 / 64, 1.0 / 64))
        #expect(coordinates.last == SIMD2(63.0 / 64, 63.0 / 64))
        #expect(Array(pixels(surface.baseColorTexture).prefix(4)) == [137, 188, 225, 255])
        let packed = pixels(surface.metallicRoughnessTexture)
        #expect(Array(packed.prefix(4)) == [64, 4, 128, 255])
        #expect(Array(packed.suffix(4)) == [64, 251, 128, 255])
        let normals = pixels(surface.normalTexture)
        #expect(normals[0] < 110)
        #expect(normals[16 * 4] > 145)
        #expect(normals[0] == normals[31 * 4])
        #expect(normals[1] == 128)
        #expect(surface.normalTexture.mipmapLevelCount == 6)
    }

    @Test
    func reportsTheFirstInvalidAuthoredFieldAndTexel() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(throws: ProceduralSurface.Failure.invalidSample(x: 1, y: 0, field: .roughness)) {
            try ProceduralSurface(device: device, resolution: 32) { uv in
                .init(baseColor: .zero, roughness: uv.x < 0.02 ? 0.5 : .infinity)
            }
        }
    }

    private func pixels(_ texture: any MTLTexture, level: Int = 0) -> [UInt8] {
        let width = max(1, texture.width >> level)
        let height = max(1, texture.height >> level)
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        bytes.withUnsafeMutableBytes { raw in
            if let address = raw.baseAddress {
                texture.getBytes(
                    address, bytesPerRow: width * 4, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: level
                )
            }
        }
        return bytes
    }
}
