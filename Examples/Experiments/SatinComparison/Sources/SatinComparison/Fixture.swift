import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd
import UniformTypeIdentifiers

/// The shared scene: one indexed grid mesh, one transform, one orthographic
/// camera, one target size. Both renderers draw exactly this.
enum Fixture {
    static let width = 512
    static let height = 512
    static let pixelFormat = MTLPixelFormat.bgra8Unorm
    static let clearColor = SIMD4<Float>(0, 0, 0, 1)

    /// Cells per side of the grid; (n + 1)² vertices, 6n² indices.
    static let gridCells = 4

    // Model transform: translate · rotate(z) · scale.
    static let translation = SIMD3<Float>(0.1, -0.05, 0)
    static let rotationZ: Float = .pi / 6
    static let scale = SIMD3<Float>(1.2, 0.8, 1)

    // Orthographic camera at z = 2 looking at the origin.
    static let cameraZ: Float = 2
    static let near: Float = 0.1
    static let far: Float = 10

    struct Mesh {
        var positions: [SIMD3<Float>]
        var texcoords: [SIMD2<Float>]
        var indices: [UInt32]
    }

    /// A grid over [-0.5, 0.5]² with counter-clockwise triangles.
    static func grid() -> Mesh {
        let n = gridCells
        var positions: [SIMD3<Float>] = []
        var texcoords: [SIMD2<Float>] = []
        for row in 0...n {
            for column in 0...n {
                let u = Float(column) / Float(n), v = Float(row) / Float(n)
                positions.append(SIMD3(u - 0.5, v - 0.5, 0))
                texcoords.append(SIMD2(u, v))
            }
        }
        var indices: [UInt32] = []
        for row in 0..<n {
            for column in 0..<n {
                let a = UInt32(row * (n + 1) + column)
                let b = a + 1, c = a + UInt32(n + 1), d = c + 1
                indices += [a, b, d, a, d, c]
            }
        }
        return Mesh(positions: positions, texcoords: texcoords, indices: indices)
    }

    static var modelMatrix: simd_float4x4 {
        let t = simd_float4x4(columns: (
            SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, 1, 0),
            SIMD4(translation.x, translation.y, translation.z, 1)))
        let r = simd_float4x4(simd_quatf(angle: rotationZ, axis: SIMD3(0, 0, 1)))
        let s = simd_float4x4(diagonal: SIMD4(scale.x, scale.y, scale.z, 1))
        return t * r * s
    }

    /// Right-handed view looking down -z, Metal clip depth in [0, 1].
    static var viewProjection: simd_float4x4 {
        let (l, r, b, t): (Float, Float, Float, Float) = (-1, 1, -1, 1)
        let projection = simd_float4x4(columns: (
            SIMD4(2 / (r - l), 0, 0, 0),
            SIMD4(0, 2 / (t - b), 0, 0),
            SIMD4(0, 0, -1 / (far - near), 0),
            SIMD4(-(r + l) / (r - l), -(t + b) / (t - b), -near / (far - near), 1)))
        let view = simd_float4x4(columns: (
            SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, 1, 0), SIMD4(0, 0, -cameraZ, 1)))
        return projection * view
    }

    static func makeTarget(device: any MTLDevice) -> (any MTLTexture)? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat, width: width, height: height, mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        return device.makeTexture(descriptor: descriptor)
    }

    static func shaderURL(_ relativePath: String) -> URL? {
        Bundle.module.resourceURL?.appendingPathComponent("Shaders").appendingPathComponent(relativePath)
    }
}

/// A CPU copy of a BGRA8 texture, read only after the GPU completed the frame
/// that wrote it.
struct Image: Sendable {
    let width: Int, height: Int
    let bgra: [UInt8]

    init(texture: any MTLTexture) {
        width = texture.width
        height = texture.height
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        texture.getBytes(&bytes, bytesPerRow: width * 4, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        bgra = bytes
    }

    /// (r, g, b) at a clip-space point, y up.
    func rgb(clipX: Float, clipY: Float) -> SIMD3<Int> {
        let x = min(width - 1, Int((clipX + 1) / 2 * Float(width)))
        let y = min(height - 1, Int((1 - clipY) / 2 * Float(height)))
        let i = (y * width + x) * 4
        return SIMD3(Int(bgra[i + 2]), Int(bgra[i + 1]), Int(bgra[i]))
    }

    var coveredPixels: Int {
        stride(from: 0, to: bgra.count, by: 4).reduce(0) { count, i in
            count + ((bgra[i] | bgra[i + 1] | bgra[i + 2]) > 8 ? 1 : 0)
        }
    }

    func writePNG(to url: URL) throws {
        // Swap BGRA to RGBA for an sRGB-tagged CGImage.
        var rgba = bgra
        for i in stride(from: 0, to: rgba.count, by: 4) { rgba.swapAt(i, i + 2) }
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        let image = CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )!
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
    }
}

struct ImageDifference: CustomStringConvertible {
    var differingPixels = 0
    var maxChannelDelta = 0
    var meanAbsoluteDifference = 0.0
    let pixelCount: Int

    init(_ a: Image, _ b: Image) {
        precondition(a.width == b.width && a.height == b.height)
        pixelCount = a.width * a.height
        var total = 0
        for pixel in 0..<pixelCount {
            var differs = false
            for channel in 0..<3 {
                let delta = abs(Int(a.bgra[pixel * 4 + channel]) - Int(b.bgra[pixel * 4 + channel]))
                total += delta
                if delta > 0 { differs = true }
                maxChannelDelta = max(maxChannelDelta, delta)
            }
            if differs { differingPixels += 1 }
        }
        meanAbsoluteDifference = Double(total) / Double(pixelCount * 3 * 255)
    }

    var description: String {
        String(format: "differing_pixels=%d (%.4f%%) max_channel_delta=%d mean_abs_diff=%.6f",
               differingPixels, 100 * Double(differingPixels) / Double(pixelCount), maxChannelDelta, meanAbsoluteDifference)
    }
}
