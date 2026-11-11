#if os(macOS)
import CoreGraphics
import Foundation
import Frostfield
import ImageIO
import Metal
import UniformTypeIdentifiers

enum LabImageCapture {
    static func target(device: any MTLDevice, width: Int, height: Int) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm_srgb, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .renderTarget
        guard let target = device.makeTexture(descriptor: descriptor) else {
            throw MeshRenderer.Failure.resource("capture color target")
        }
        return target
    }

    /// Read after the command buffer that renders the target has completed.
    static func write(_ target: any MTLTexture, to url: URL) throws {
        let width = target.width
        let height = target.height
        let rowBytes = width * 4
        var pixels = [UInt8](repeating: 0, count: rowBytes * height)
        try pixels.withUnsafeMutableBytes { raw in
            guard let address = raw.baseAddress else { throw MeshRenderer.Failure.resource("capture pixels") }
            target.getBytes(
                address, bytesPerRow: rowBytes,
                from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0
            )
        }
        let bitmap = CGBitmapInfo.byteOrder32Little.union(CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue))
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: rowBytes,
                space: colorSpace, bitmapInfo: bitmap, provider: provider, decode: nil,
                shouldInterpolate: false, intent: .defaultIntent
              ),
              let writer = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw MeshRenderer.Failure.resource("capture PNG writer")
        }
        CGImageDestinationAddImage(writer, image, nil)
        guard CGImageDestinationFinalize(writer) else { throw MeshRenderer.Failure.resource("capture PNG") }
    }
}
#endif
