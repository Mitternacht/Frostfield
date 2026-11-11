import Metal
import Testing

/// Creates known frames and reads results back for the post-process suites.
struct PostProcessFixture {
    let device: any MTLDevice
    let queue: any MTLCommandQueue

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
    }

    /// A shared RGBA16Float texture usable as a source or render target.
    func color(
        width: Int, height: Int, pixelFormat: MTLPixelFormat = .rgba16Float,
        pixel: ((Int, Int) -> SIMD4<Float>)? = nil
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.shaderRead, .renderTarget]
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        if let pixel {
            var halves: [Float16] = []
            for y in 0..<height {
                for x in 0..<width {
                    let rgba = pixel(x, y)
                    halves += [Float16(rgba.x), Float16(rgba.y), Float16(rgba.z), Float16(rgba.w)]
                }
            }
            try halves.withUnsafeBytes { bytes in
                texture.replace(
                    region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                    withBytes: try #require(bytes.baseAddress), bytesPerRow: width * 8
                )
            }
        }
        return texture
    }

    /// A private Depth32Float texture filled with known values through a blit.
    func depth(width: Int, height: Int, value: (Int, Int) -> Float) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = [.shaderRead]
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        var values: [Float] = []
        for y in 0..<height {
            for x in 0..<width { values.append(value(x, y)) }
        }
        let buffer = try #require(values.withUnsafeBytes { bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        })
        let command = try #require(queue.makeCommandBuffer())
        let blit = try #require(command.makeBlitCommandEncoder())
        blit.copy(
            from: buffer, sourceOffset: 0, sourceBytesPerRow: width * 4, sourceBytesPerImage: width * height * 4,
            sourceSize: MTLSize(width: width, height: height, depth: 1), to: texture,
            destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
        )
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed)
        return texture
    }

    /// Commits a buffer encoded by `body` and waits for it.
    func run(_ body: (any MTLCommandBuffer) throws -> Void) throws {
        let command = try #require(queue.makeCommandBuffer())
        try body(command)
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed, "\(String(describing: command.error))")
    }

    /// RGBA16Float pixels as float4 values in row-major order.
    func read(_ texture: any MTLTexture) throws -> [SIMD4<Float>] {
        var halves = Array(repeating: Float16.zero, count: texture.width * texture.height * 4)
        try halves.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: texture.width * 8,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return stride(from: 0, to: halves.count, by: 4).map {
            SIMD4(Float(halves[$0]), Float(halves[$0 + 1]), Float(halves[$0 + 2]), Float(halves[$0 + 3]))
        }
    }

    /// Whether every component differs by at most the tolerance.
    static func close(_ a: SIMD4<Float>, _ b: SIMD4<Float>, tolerance: Float = 2e-3) -> Bool {
        let difference = a - b
        return max(max(abs(difference.x), abs(difference.y)), max(abs(difference.z), abs(difference.w))) <= tolerance
    }
}
