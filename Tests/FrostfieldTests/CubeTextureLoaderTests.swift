import CoreGraphics
import Foundation
import Frostfield
import ImageIO
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct CubeTextureLoaderTests {
    @Test("Six PNG faces preserve Metal direction, top-left orientation and sRGB sampling")
    func sRGBFaces() throws {
        let fixture = try RasterLightingFixture(width: 4, height: 4)
        let bases: [UInt8] = [32, 64, 96, 128, 160, 192]
        let faces = try bases.map { try png(width: 2, height: 2, reds: [$0, 16, 8, 0]) }
        let cube = try CubeTextureLoader(queue: fixture.queue).load(data: faces)
        #expect(cube.textureType == .typeCube)
        #expect(cube.pixelFormat == .rgba8Unorm_srgb)
        #expect(cube.mipmapLevelCount == 2)

        // These are the face-local centers of the upper-left and lower-right texels.
        let upperLeft: [SIMD3<Float>] = [
            SIMD3(1, 0.5, 0.5), SIMD3(-1, 0.5, -0.5),
            SIMD3(-0.5, 1, -0.5), SIMD3(-0.5, -1, 0.5),
            SIMD3(-0.5, 0.5, 1), SIMD3(0.5, 0.5, -1),
        ]
        let lowerRight: [SIMD3<Float>] = [
            SIMD3(1, -0.5, -0.5), SIMD3(-1, -0.5, 0.5),
            SIMD3(0.5, 1, 0.5), SIMD3(0.5, -1, -0.5),
            SIMD3(0.5, -0.5, 1), SIMD3(-0.5, -0.5, -1),
        ]
        let samples = try fixture.sampleCube(cube, directions: upperLeft + lowerRight)
        let expected = bases.map(linear)
        for index in bases.indices {
            #expect(simd_length(samples[index] - SIMD4(expected[index], 0, 0, 1)) < 0.003)
            #expect(samples[index + bases.count] == SIMD4(0, 0, 0, 1))
        }
        let mips = try fixture.sampleCube(cube, directions: Self.axes, mip: 1)
        for index in bases.indices {
            let mean = (expected[index] + linear(16) + linear(8)) / 4
            #expect(abs(mips[index].x - mean) < 0.004)
        }

        let lighting = try EnvironmentLighting(
            queue: fixture.queue, cube: cube, settings: .init(cubeSize: 8, diffuseSize: 4, brdfSize: 8, sampleCount: 16)
        )
        let sky = try fixture.sampleCube(lighting.skyCube, directions: Self.axes)
        #expect(sky.last!.x > sky.first!.x)
        #expect(sky.allSatisfy { $0.x > 0 && $0.y == 0 && $0.z == 0 })
    }

    @Test("Radiance face files keep full float HDR values and default half output")
    func hdrFiles() throws {
        let fixture = try RasterLightingFixture(width: 2, height: 2)
        let loader = CubeTextureLoader(queue: fixture.queue)
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let reds: [UInt8] = [10, 20, 30, 40, 50, 60]
        let faces = reds.map(radiance)
        let urls = try faces.enumerated().map { index, face in
            let url = directory.appending(path: "\(index).hdr")
            try face.write(to: url)
            return url
        }
        let cube = try loader.loadHighDynamicRange(contentsOf: urls, precision: .float)
        #expect(cube.pixelFormat == .rgba32Float)
        #expect(cube.mipmapLevelCount == 1)
        let samples = try fixture.sampleCube(cube, directions: Self.axes)
        for (index, red) in reds.enumerated() {
            #expect(samples[index] == SIMD4((Float(red) + 0.5) * 2048, 1024, 1024, 1))
        }
        let half = try loader.loadHighDynamicRange(data: faces, mipmapped: true)
        #expect(half.pixelFormat == .rgba16Float && half.mipmapLevelCount == 2)
        let mip = try fixture.sampleCube(half, directions: Self.axes, mip: 1)
        #expect(mip[0] == SIMD4(21504, 1024, 1024, 1))
        #expect(mip[5].x == 65504)
    }

    @Test("Missing, malformed, mismatched and non-square faces fail with face indices")
    func invalidFaces() throws {
        let fixture = try RasterLightingFixture(width: 2, height: 2)
        let loader = CubeTextureLoader(queue: fixture.queue)
        let square = try png(width: 2, height: 2, reds: [0, 0, 0, 0])
        var faces = [Data](repeating: square, count: 6)
        #expect(throws: CubeTextureLoader.Failure.faceCount) { try loader.load(data: Array(faces.dropLast())) }
        faces[2] = try png(width: 1, height: 2, reds: [0, 0])
        #expect(throws: CubeTextureLoader.Failure.faceSize(2, 1, 2)) { try loader.load(data: faces) }
        faces[2] = try png(width: 1, height: 1, reds: [0])
        #expect(throws: CubeTextureLoader.Failure.faceSize(2, 1, 1)) { try loader.load(data: faces) }
        faces[2] = radiance(10)
        #expect(throws: CubeTextureLoader.Failure.faceFormat(2)) { try loader.load(data: faces) }
        faces[2] = Data([1, 2, 3])
        do {
            _ = try loader.load(data: faces)
            Issue.record("Malformed face unexpectedly loaded")
        } catch CubeTextureLoader.Failure.decodeFace(let index, _) {
            #expect(index == 2)
        }
    }

    private static let axes: [SIMD3<Float>] = [
        SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0),
        SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1),
    ]

    private func linear(_ byte: UInt8) -> Float {
        let encoded = Float(byte) / 255
        return encoded <= 0.04045 ? encoded / 12.92 : pow((encoded + 0.055) / 1.055, 2.4)
    }

    private func png(width: Int, height: Int, reds: [UInt8]) throws -> Data {
        let data = Data(reds.flatMap { [$0, 0, 0, 255] })
        let provider = try #require(CGDataProvider(data: data as CFData))
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let image = try #require(CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
            space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        ))
        let encoded = NSMutableData()
        let writer = try #require(CGImageDestinationCreateWithData(encoded, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(writer, image, nil)
        try #require(CGImageDestinationFinalize(writer))
        return encoded as Data
    }

    private func radiance(_ red: UInt8) -> Data {
        var data = Data("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 2\n".utf8)
        for _ in 0..<4 { data.append(contentsOf: [red, 0, 0, 147]) }
        return data
    }
}
