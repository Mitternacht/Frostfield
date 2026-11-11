import Foundation
import Frostfield
import Metal
import Testing

struct RadianceHDRTests {
    @Test("RGBE expands HDR values and normalizes X-major image orientation")
    func orientedFlatPixels() throws {
        var source = header("-X 4 +Y 2")
        for index in UInt8(0)..<8 {
            source.append(contentsOf: [160 + index, 64, 32, 130])
        }

        let decoded = try RadianceHDRImage.decode(source)
        #expect(decoded.width == 4)
        #expect(decoded.height == 2)
        #expect(decoded.pixels[0] == SIMD4<Float16>(2.6171875, 1.0078125, 0.5078125, 1))
        #expect(decoded.pixels[7] == SIMD4<Float16>(2.5078125, 1.0078125, 0.5078125, 1))
    }

    @Test("Radiance decodes literal and repeat packets and rejects truncated rows")
    func scanlinePackets() throws {
        var source = header("-Y 4 +X 8")
        let row: [UInt8] = [
            2, 2, 0, 8,
            8, 128, 129, 130, 131, 132, 133, 134, 135,
            136, 64,
            136, 32,
            136, 130,
        ]
        for _ in 0..<4 { source.append(contentsOf: row) }
        let decoded = try RadianceHDRImage.decode(source)
        #expect(decoded.pixels[0] == SIMD4<Float16>(2.0078125, 1.0078125, 0.5078125, 1))
        #expect(decoded.pixels[31] == SIMD4<Float16>(2.1171875, 1.0078125, 0.5078125, 1))
        #expect(throws: EnvironmentLighting.RadianceFailure.scanline(3)) {
            try RadianceHDRImage.decode(Data(source.dropLast()))
        }

        var oldStyle = header("-Y 1 +X 2")
        oldStyle.append(contentsOf: [160, 64, 32, 130, 1, 1, 1, 1])
        let repeated = try RadianceHDRImage.decode(oldStyle)
        #expect(repeated.pixels[0] == repeated.pixels[1])
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct EnvironmentLightingTests {
    private let queue: any MTLCommandQueue

    init() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
    }

    @Test("Prepared lighting preserves constant HDR radiance at every specular roughness")
    func constantRadiance() throws {
        var source = header("-Y 2 +X 4")
        for _ in 0..<8 { source.append(contentsOf: [160, 64, 32, 130]) }
        let lighting = try EnvironmentLighting(
            queue: queue, radianceHDR: source,
            settings: .init(cubeSize: 16, diffuseSize: 4, brdfSize: 16, sampleCount: 32)
        )

        #expect(lighting.skyCube.textureType == .typeCube)
        #expect(lighting.diffuseCube.pixelFormat == .rgba16Float)
        #expect(lighting.specularCube.mipmapLevelCount == 5)
        #expect(lighting.brdfLUT.pixelFormat == .rg16Float)
        #expect(lighting.maxSpecularLOD == 4)
        var inputs = (0..<6).flatMap { face in
            [
                Texel(lighting.skyCube, face: face, x: 3, y: 5),
                Texel(lighting.diffuseCube, face: face, x: 1, y: 2),
                Texel(lighting.specularCube, face: face, x: 3, y: 5),
                Texel(lighting.specularCube, face: face, mip: 4),
            ]
        }
        inputs.append(Texel(lighting.brdfLUT, x: 15))
        let pixels = try read(inputs)
        for pixel in pixels.dropLast() {
            #expect(abs(pixel[0] - 2.5078125) < 0.025)
            #expect(abs(pixel[1] - 1.0078125) < 0.025)
            #expect(abs(pixel[2] - 0.5078125) < 0.025)
            #expect(pixel[3] == 1)
        }
        let mirror = try #require(pixels.last)
        #expect(mirror[0] > 0.95 && mirror[0] < 1.05)
        #expect(mirror[1] >= 0 && mirror[1] < 0.02)
    }

    @Test("Diffuse and rough specular lighting spread a sharp directional HDR source")
    func directionalRadiance() throws {
        var source = header("-Y 32 +X 64")
        for y in 0..<32 {
            for x in 0..<64 {
                let hot = (27...36).contains(x) && (13...18).contains(y)
                let pixel: [UInt8] = hot ? [128, 96, 64, 134] : [128, 128, 128, 125]
                source.append(contentsOf: pixel)
            }
        }
        let lighting = try EnvironmentLighting(
            queue: queue, radianceHDR: source,
            settings: .init(cubeSize: 32, diffuseSize: 8, brdfSize: 16, sampleCount: 128)
        )
        let pixels = try read([
            Texel(lighting.skyCube, face: 0, x: 16, y: 16),
            Texel(lighting.specularCube, face: 0, x: 16, y: 16),
            Texel(lighting.specularCube, face: 0, mip: 5),
            Texel(lighting.diffuseCube, face: 0, x: 4, y: 4),
            Texel(lighting.diffuseCube, face: 1, x: 4, y: 4),
        ])
        #expect(pixels[0][0] > 20)
        #expect(pixels[1] == pixels[0])
        #expect(pixels[2][0] > 0.25 && pixels[2][0] < pixels[1][0] * 0.5)
        #expect(pixels[3][0] > pixels[4][0] + 0.5)
    }

    private struct Texel {
        let texture: any MTLTexture
        let face: Int
        let mip: Int
        let x: Int
        let y: Int

        init(_ texture: any MTLTexture, face: Int = 0, mip: Int = 0, x: Int = 0, y: Int = 0) {
            self.texture = texture
            self.face = face
            self.mip = mip
            self.x = x
            self.y = y
        }
    }

    private func read(_ inputs: [Texel]) throws -> [[Float]] {
        let stride = 256
        let buffer = try #require(queue.device.makeBuffer(length: inputs.count * stride, options: .storageModeShared))
        let command = try #require(queue.makeCommandBuffer())
        let blit = try #require(command.makeBlitCommandEncoder())
        for (index, sample) in inputs.enumerated() {
            blit.copy(
                from: sample.texture, sourceSlice: sample.face, sourceLevel: sample.mip,
                sourceOrigin: MTLOrigin(x: sample.x, y: sample.y, z: 0), sourceSize: MTLSize(width: 1, height: 1, depth: 1),
                to: buffer, destinationOffset: index * stride, destinationBytesPerRow: stride,
                destinationBytesPerImage: stride
            )
        }
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed, "\(command.error?.localizedDescription ?? "Readback failed.")")
        return inputs.enumerated().map { index, sample in
            let count = sample.texture.pixelFormat == .rg16Float ? 2 : 4
            let values = buffer.contents().advanced(by: index * stride).bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(values[$0]) }
        }
    }
}

private func header(_ resolution: String) -> Data {
    Data("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n\(resolution)\n".utf8)
}
