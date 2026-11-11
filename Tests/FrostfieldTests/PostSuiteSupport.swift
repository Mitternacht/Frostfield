import CoreGraphics
import Foundation
import ImageIO
import Metal
import Testing
import UniformTypeIdentifiers

extension PostProcessFixture {
    /// A shared Depth32Float render target that shaders can read, for scenes drawn in tests.
    func depthTarget(width: Int, height: Int) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = [.renderTarget, .shaderRead]
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    /// A shared single-channel float texture that compute kernels can write.
    func scalarTexture(width: Int, height: Int) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.shaderRead, .shaderWrite]
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    func readScalars(_ texture: any MTLTexture) throws -> [Float] {
        var values = [Float](repeating: 0, count: texture.width * texture.height)
        try values.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: texture.width * 4,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return values
    }

    /// Writes linear RGBA as an sRGB PNG when `FROSTFIELD_POSTSUITE_IMAGES` names a directory,
    /// so rendered results can be inspected. Does nothing otherwise.
    func savePNG(_ pixels: [SIMD4<Float>], width: Int, height: Int, name: String) {
        guard let directory = ProcessInfo.processInfo.environment["FROSTFIELD_POSTSUITE_IMAGES"] else { return }
        func encode(_ value: Float) -> UInt8 {
            let v = min(max(value, 0), 1)
            let s = v <= 0.0031308 ? v * 12.92 : 1.055 * pow(v, 1 / 2.4) - 0.055
            return UInt8((s * 255).rounded())
        }
        var bytes: [UInt8] = []
        for pixel in pixels {
            let alpha = min(max(pixel.w, 0), 1)
            let straight = alpha > 0 ? SIMD3(pixel.x, pixel.y, pixel.z) / alpha : .zero
            bytes += [encode(straight.x), encode(straight.y), encode(straight.z), UInt8((alpha * 255).rounded())]
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider, decode: nil,
                shouldInterpolate: false, intent: .defaultIntent
              ) else { return }
        let url = URL(fileURLWithPath: directory).appendingPathComponent("\(name).png")
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            return
        }
        CGImageDestinationAddImage(destination, image, nil)
        CGImageDestinationFinalize(destination)
    }
}

/// Stored forward Metal depth for a view distance, `f (z − n) / ((f − n) z)`.
func storedDepth(distance z: Float, near n: Float, far f: Float, reversed: Bool = false) -> Float {
    let forward = Float(Double(f) * (Double(z) - Double(n)) / ((Double(f) - Double(n)) * Double(z)))
    return reversed ? 1 - forward : forward
}

func sum(_ pixels: [SIMD4<Float>]) -> SIMD4<Double> {
    pixels.reduce(SIMD4<Double>.zero) { $0 + SIMD4<Double>($1) }
}
