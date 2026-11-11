import CoreGraphics
import Foundation

import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct HDRDisplayTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
    }

    @Test("PBR Neutral compresses highlights and tone maps straight color before repremultiplying")
    func toneMappingAndExposure() throws {
        let source = try input(width: 4, height: 1) { x, _ in
            switch x {
            case 0: SIMD4(0.5, 0.5, 0.5, 1)
            case 1: SIMD4(0.25, 0.25, 0.25, 0.5)
            case 2: SIMD4(4, 2, 1, 1)
            default: SIMD4(0, 0, 0, 0)
            }
        }
        let target = try texture(.rgba16Float, width: 4, height: 1, usage: .renderTarget)
        let display = try HDRDisplay(device: device, destinationPixelFormat: .rgba16Float)
        try render(display, source: source, destination: target)
        let pixels = read(target, zero: Float16.zero)
        #expect(abs(Float(pixels[0]) - 0.46) < 0.001)
        #expect(abs(Float(pixels[4]) - 0.23) < 0.001)
        #expect(pixels[7] == 0.5)
        #expect(abs(Float(pixels[8]) - 0.98326) < 0.001)
        #expect(abs(Float(pixels[9]) - 0.63995) < 0.001)
        #expect(abs(Float(pixels[10]) - 0.46830) < 0.001)
        #expect(pixels[12...15].allSatisfy { $0 == 0 })

        try render(display, source: source, destination: target, settings: .init(exposureStops: 1))
        let exposed = read(target, zero: Float16.zero)
        #expect(abs(Float(exposed[0]) - 0.86909) < 0.001)
        #expect(exposed[7] == 0.5)
    }

    @Test("sRGB destination encodes RGB and keeps alpha linear")
    func srgbEncoding() throws {
        let source = try input(width: 1, height: 1) { _, _ in SIMD4(0.25, 0.125, 0.0625, 0.5) }
        let target = try texture(.bgra8Unorm_srgb, width: 1, height: 1, usage: .renderTarget)
        let display = try HDRDisplay(device: device, destinationPixelFormat: .bgra8Unorm_srgb)
        try render(display, source: source, destination: target)
        let pixels = read(target, zero: UInt8.zero)
        for (actual, expected) in zip(pixels, [58, 91, 132, 128]) {
            #expect(abs(Int(actual) - expected) <= 1)
        }
    }

    @Test("Wide-gamut float output preserves extended linear color and premultiplied alpha",
          arguments: [HDRDisplay.OutputColorSpace.displayP3, .rec2020])
    func wideGamut(_ space: HDRDisplay.OutputColorSpace) throws {
        let values: [SIMD4<Float>] = [
            SIMD4(1.225, -0.042, -0.020, 1),
            SIMD4(-0.125, 1.5, 0.5, 0.5),
            SIMD4(2, 2, 2, 1),
            .zero,
        ]
        let source = try input(width: values.count, height: 1) { x, _ in values[x] }
        let target = try texture(.rgba16Float, width: values.count, height: 1, usage: .renderTarget)
        let display = try HDRDisplay(device: device, destinationPixelFormat: .rgba16Float, outputColorSpace: space)
        let expectedName = space == .displayP3 ? CGColorSpace.extendedLinearDisplayP3 : CGColorSpace.extendedLinearITUR_2020
        #expect(display.presentationColorSpace.name as String? == expectedName as String)
        #expect(display.outputColorSpace == space)
        try render(display, source: source, destination: target, settings: .init(toneMapping: .none))
        let pixels = read(target, zero: Float16.zero)
        for (index, input) in values.enumerated() {
            let rgb = SIMD3<Double>(Double(Float16(input.x)), Double(Float16(input.y)), Double(Float16(input.z)))
            let expected = Self.xyzToOutput(space) * (Self.sRGBToXYZ * rgb)
            for channel in 0..<3 {
                #expect(abs(Double(pixels[index * 4 + channel]) - expected[channel]) < 0.0015)
            }
            #expect(pixels[index * 4 + 3] == Float16(input.w))
        }
        if space == .displayP3 {
            #expect(abs(Float(pixels[0]) - 1) < 0.0015)
            #expect(abs(Float(pixels[1])) < 0.0015)
            #expect(abs(Float(pixels[2])) < 0.0015)
        }
    }

    @Test("Display P3 applies the Metal sRGB transfer after conversion and keeps alpha linear")
    func displayP3Encoding() throws {
        let source = try input(width: 1, height: 1) { _, _ in SIMD4(0.5, 0, 0, 0.5) }
        let target = try texture(.bgra8Unorm_srgb, width: 1, height: 1, usage: .renderTarget)
        let display = try HDRDisplay(
            device: device, destinationPixelFormat: .bgra8Unorm_srgb, outputColorSpace: .displayP3
        )
        #expect(display.presentationColorSpace.name as String? == CGColorSpace.displayP3 as String)
        try render(display, source: source, destination: target, settings: .init(toneMapping: .none))
        let stored = read(target, zero: UInt8.zero)
        let expected = Self.xyzToOutput(.displayP3) * (Self.sRGBToXYZ * SIMD3<Double>(0.5, 0, 0))
        for (channel, reference) in zip(stored.prefix(3).reversed(), [expected.x, expected.y, expected.z]) {
            let encoded = reference <= 0.0031308 ? 12.92 * reference : 1.055 * pow(reference, 1 / 2.4) - 0.055
            #expect(abs(Int(channel) - Int((encoded * 255).rounded())) <= 1)
        }
        #expect(stored[3] == 128)
        #expect(throws: HDRDisplay.Failure.outputColorSpace) {
            try HDRDisplay(device: device, destinationPixelFormat: .bgra8Unorm_srgb, outputColorSpace: .rec2020)
        }
    }

    // Independent XYZ transforms from the r187 ColorManagement and ColorSpaces matrices.
    private static let sRGBToXYZ = simd_double3x3(
        SIMD3(0.4123908, 0.2126390, 0.0193308),
        SIMD3(0.3575843, 0.7151687, 0.1191948),
        SIMD3(0.1804808, 0.0721923, 0.9505322)
    )

    private static func xyzToOutput(_ space: HDRDisplay.OutputColorSpace) -> simd_double3x3 {
        if space == .displayP3 {
            return simd_double3x3(
                SIMD3(2.4934969, -0.8294890, 0.0358458),
                SIMD3(-0.9313836, 1.7626641, -0.0761724),
                SIMD3(-0.4027108, 0.0236247, 0.9568845)
            )
        }
        return simd_double3x3(
            SIMD3(1.7166512, -0.6666844, 0.0176399),
            SIMD3(-0.3556708, 1.6164812, -0.0427706),
            SIMD3(-0.2533663, 0.0157685, 0.9421031)
        )
    }

    @Test("Spatial AA smooths diagonal coverage and keeps an axis-aligned edge sharp")
    func antiAliasing() throws {
        let diagonal = try input(width: 7, height: 7) { x, y in SIMD4(repeating: x <= y ? 1 : 0) }
        let vertical = try input(width: 7, height: 7) { x, _ in SIMD4(repeating: x <= 3 ? 1 : 0) }
        let target = try texture(.rgba16Float, width: 7, height: 7, usage: .renderTarget)
        let display = try HDRDisplay(device: device, destinationPixelFormat: .rgba16Float)
        let center = (3 * 7 + 3) * 4
        try render(display, source: diagonal, destination: target)
        let baseline = read(target, zero: Float16.zero)
        #expect(baseline[center + 3] == 1)
        #expect(baseline[center + 7] == 0)

        try render(display, source: diagonal, destination: target, settings: .init(antiAliasing: true))
        let filtered = read(target, zero: Float16.zero)
        #expect(filtered[center + 3] == 0.75)
        #expect(filtered[center + 7] == 0.25)
        #expect(abs(Float(filtered[center]) - 0.75 * Float(baseline[center])) < 0.001)

        try render(display, source: vertical, destination: target, settings: .init(antiAliasing: true))
        let straight = read(target, zero: Float16.zero)
        #expect(straight[center + 3] == 1)
        #expect(straight[center + 7] == 0)
    }

    @Test("Presentation rejects aliased targets and nonfinite exposure")
    func validation() throws {
        let source = try input(width: 1, height: 1) { _, _ in SIMD4(repeating: 1) }
        let target = try texture(.rgba16Float, width: 1, height: 1, usage: .renderTarget)
        let display = try HDRDisplay(device: device, destinationPixelFormat: .rgba16Float)
        let buffer = try #require(queue.makeCommandBuffer())
        #expect(throws: HDRDisplay.Failure.incompatibleTextures) {
            try display.encode(source: source, destination: source, commandBuffer: buffer)
        }
        #expect(throws: HDRDisplay.Failure.invalidExposure) {
            try display.encode(source: source, destination: target, settings: .init(exposureStops: .nan), commandBuffer: buffer)
        }
    }

    private func input(
        width: Int, height: Int, pixel: (Int, Int) -> SIMD4<Float>
    ) throws -> any MTLTexture {
        let source = try texture(.rgba16Float, width: width, height: height, usage: [.shaderRead, .renderTarget])
        var pixels: [SIMD4<Float16>] = []
        for y in 0..<height {
            for x in 0..<width {
                let rgba = pixel(x, y)
                pixels.append(SIMD4(Float16(rgba.x), Float16(rgba.y), Float16(rgba.z), Float16(rgba.w)))
            }
        }
        try pixels.withUnsafeBytes { bytes in
            source.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: try #require(bytes.baseAddress), bytesPerRow: width * 8
            )
        }
        return source
    }

    private func texture(
        _ format: MTLPixelFormat, width: Int, height: Int, usage: MTLTextureUsage
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = usage
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    private func render(
        _ display: HDRDisplay, source: any MTLTexture, destination: any MTLTexture,
        settings: HDRDisplay.Settings = .init()
    ) throws {
        let buffer = try #require(queue.makeCommandBuffer())
        try display.encode(source: source, destination: destination, settings: settings, commandBuffer: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)
    }

    private func read<Channel>(_ texture: any MTLTexture, zero: Channel) -> [Channel] {
        var pixels = Array(repeating: zero, count: texture.width * texture.height * 4)
        pixels.withUnsafeMutableBytes { bytes in
            guard let address = bytes.baseAddress else { return }
            texture.getBytes(
                address, bytesPerRow: texture.width * 4 * MemoryLayout<Channel>.stride,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return pixels
    }
}
