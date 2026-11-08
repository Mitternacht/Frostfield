import Metal
import Testing
@testable import Frostfield

/// A CPU-readable BGRA8 render target cleared to opaque black, optionally
/// with a depth attachment cleared to 1.
struct OffscreenTarget {
    struct Pixel: CustomStringConvertible, Equatable {
        enum Channel: Sendable { case red, green, blue }

        var blue: UInt8, green: UInt8, red: UInt8, alpha: UInt8
        var isClear: Bool { red < 8 && green < 8 && blue < 8 }
        var description: String { "rgba(\(red), \(green), \(blue), \(alpha))" }

        /// The channel that is bright and clearly stronger than the other two,
        /// or `nil` for clear or mixed pixels.
        var dominantChannel: Channel? {
            let values: [(Channel, Int)] = [(.red, Int(red)), (.green, Int(green)), (.blue, Int(blue))]
            let sorted = values.sorted { $0.1 > $1.1 }
            guard sorted[0].1 >= 120, sorted[0].1 - sorted[1].1 >= 60 else { return nil }
            return sorted[0].0
        }
    }

    let texture: any MTLTexture
    /// A private depth texture, kept after the pass so tests can copy it out.
    let depthTexture: (any MTLTexture)?

    init(device: any MTLDevice, width: Int, height: Int, depthPixelFormat: MTLPixelFormat? = nil) throws {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false
        )
        descriptor.usage = .renderTarget
        descriptor.storageMode = .shared
        texture = try #require(device.makeTexture(descriptor: descriptor))

        if let depthPixelFormat {
            let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: depthPixelFormat, width: width, height: height, mipmapped: false
            )
            depthDescriptor.usage = .renderTarget
            depthDescriptor.storageMode = .private
            let depthTexture: any MTLTexture = try #require(device.makeTexture(descriptor: depthDescriptor))
            self.depthTexture = depthTexture
        } else {
            depthTexture = nil
        }
    }

    var frameTarget: FrameTarget {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        if let depthTexture {
            pass.depthAttachment.texture = depthTexture
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.storeAction = .store
            pass.depthAttachment.clearDepth = 1
        }
        return FrameTarget(passDescriptor: pass, drawable: nil)
    }

    /// The pixel position of a normalized-device-coordinate point.
    func pixelPosition(ndc: SIMD2<Float>) -> (x: Int, y: Int) {
        let x = min(texture.width - 1, Int((ndc.x + 1) / 2 * Float(texture.width)))
        let y = min(texture.height - 1, Int((1 - ndc.y) / 2 * Float(texture.height)))
        return (x, y)
    }

    /// The NDC point at the centre of a pixel.
    func ndc(x: Int, y: Int) -> SIMD2<Float> {
        SIMD2(
            (Float(x) + 0.5) / Float(texture.width) * 2 - 1,
            1 - (Float(y) + 0.5) / Float(texture.height) * 2
        )
    }

    /// Copies a `depth32Float` attachment into a shared buffer on the GPU and
    /// returns its values in row-major order, top row first.
    func depthValues(queue: any MTLCommandQueue) async throws -> [Float] {
        let depthTexture = try #require(depthTexture)
        try #require(depthTexture.pixelFormat == .depth32Float)
        let bytesPerRow = depthTexture.width * MemoryLayout<Float>.stride
        let length = bytesPerRow * depthTexture.height
        let buffer = try #require(queue.device.makeBuffer(length: length, options: .storageModeShared))
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let blit = try #require(commandBuffer.makeBlitCommandEncoder())
        blit.copy(
            from: depthTexture, sourceSlice: 0, sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: depthTexture.width, height: depthTexture.height, depth: 1),
            to: buffer, destinationOffset: 0,
            destinationBytesPerRow: bytesPerRow, destinationBytesPerImage: length
        )
        blit.endEncoding()
        let status = await commitAndWait(commandBuffer)
        try #require(status == .completed)
        return Array(UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: Float.self),
            count: depthTexture.width * depthTexture.height
        ))
    }

    /// The pixel containing a normalized-device-coordinate point.
    func pixel(at ndc: SIMD2<Float>) -> Pixel {
        pixel(clipX: ndc.x, clipY: ndc.y)
    }

    /// The pixel containing a clip-space point (y up, as in the shader).
    func pixel(clipX: Float, clipY: Float) -> Pixel {
        let (x, y) = pixelPosition(ndc: SIMD2(clipX, clipY))
        return pixel(x: x, y: y)
    }

    /// Every pixel in row-major order, top row first, read in one copy.
    func pixels() -> [Pixel] {
        let count = texture.width * texture.height
        var bytes = [UInt8](repeating: 0, count: count * 4)
        texture.getBytes(
            &bytes, bytesPerRow: texture.width * 4,
            from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
        )
        return (0..<count).map { i in
            Pixel(blue: bytes[i * 4], green: bytes[i * 4 + 1], red: bytes[i * 4 + 2], alpha: bytes[i * 4 + 3])
        }
    }

    func pixel(x: Int, y: Int) -> Pixel {
        var bytes = [UInt8](repeating: 0, count: 4)
        texture.getBytes(&bytes, bytesPerRow: 4, from: MTLRegionMake2D(x, y, 1, 1), mipmapLevel: 0)
        return Pixel(blue: bytes[0], green: bytes[1], red: bytes[2], alpha: bytes[3])
    }
}
