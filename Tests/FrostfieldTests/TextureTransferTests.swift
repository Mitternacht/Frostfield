import Foundation
import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TextureTransferTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
    }

    @Test("GPU render targets read their selected lower mip and preserve stored BGRA order")
    func renderTargetMip() async throws {
        let texture = try texture2D(.bgra8Unorm, side: 8, mipmapped: true, usage: .renderTarget)
        let producer = try #require(queue.makeCommandBuffer())
        for (level, color) in [(0, MTLClearColor(red: 0, green: 0, blue: 1, alpha: 1)),
                               (1, MTLClearColor(red: 1, green: 1, blue: 0, alpha: 1))] {
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = texture
            pass.colorAttachments[0].level = level
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = color
            try #require(producer.makeRenderCommandEncoder(descriptor: pass)).endEncoding()
        }
        producer.commit()

        let result = try await TextureTransfer.read(
            from: texture, mipLevel: 1, region: MTLRegionMake2D(1, 2, 3, 2), commandQueue: queue
        )
        #expect(result.pixelFormat == .bgra8Unorm)
        #expect(result.size.width == 3 && result.size.height == 2 && result.size.depth == 1)
        #expect(result.bytesPerRow == 12 && result.bytesPerImage == 24)
        #expect(result.bytes == Data(Array(repeating: [UInt8(0), 255, 255, 255], count: 6).joined()))
        let base = try await TextureTransfer.read(from: texture, region: MTLRegionMake2D(0, 0, 1, 1), commandQueue: queue)
        #expect(base.bytes == Data([255, 0, 0, 255]))
        #expect(producer.status == .completed)
    }

    @Test("Copies select source and destination mips and preserve the surrounding texels")
    func regionCopy() async throws {
        let source = try texture2D(.rgba8Uint, side: 8, mipmapped: true)
        let destination = try texture2D(.rgba8Uint, side: 8, mipmapped: true)
        let command = try #require(queue.makeCommandBuffer())
        let colors: [UInt8] = (0..<64).map { UInt8($0) }
        try upload(colors, to: source, level: 1, region: MTLRegionMake2D(0, 0, 4, 4), command: command)
        try upload(Array(repeating: 233, count: 16), to: destination, level: 2,
                   region: MTLRegionMake2D(0, 0, 2, 2), command: command)
        try TextureTransfer.encodeCopy(
            from: source, sourceMipLevel: 1, sourceRegion: MTLRegionMake2D(1, 2, 2, 1),
            to: destination, destinationMipLevel: 2, destinationOrigin: MTLOrigin(x: 0, y: 1, z: 0),
            commandBuffer: command
        )
        command.commit()
        let result = try await TextureTransfer.read(from: destination, mipLevel: 2, commandQueue: queue)
        #expect(result.pixelFormat == .rgba8Uint)
        #expect(result.bytes == Data(Array(repeating: 233, count: 8) + colors[36..<44]))
    }

    @Test("CPU uploads patch one array layer and mip while preserving adjacent texels and other layers")
    func arrayUpload() async throws {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Uint, width: 8, height: 8, mipmapped: true)
        descriptor.textureType = .type2DArray
        descriptor.arrayLength = 2
        descriptor.storageMode = .private
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let command = try #require(queue.makeCommandBuffer())
        let firstLayer = Data(repeating: 13, count: 64)
        let secondLayer = Data(repeating: 29, count: 64)
        let baseMip = Data(repeating: 71, count: 256)
        try TextureTransfer.encodeUpload(firstLayer, to: texture, mipLevel: 1, commandBuffer: command)
        try TextureTransfer.encodeUpload(secondLayer, to: texture, mipLevel: 1, slice: 1, commandBuffer: command)
        try TextureTransfer.encodeUpload(baseMip, to: texture, slice: 1, commandBuffer: command)
        let region = MTLRegionMake2D(1, 2, 2, 1)
        var patch = Data([1, 2, 3, 4, 10, 20, 30, 40])
        #expect(throws: TextureTransfer.Failure.uploadByteCount(expected: 8)) {
            try TextureTransfer.encodeUpload(Data([1]), to: texture, mipLevel: 1, slice: 1, region: region,
                                             commandBuffer: command)
        }
        try TextureTransfer.encodeUpload(patch, to: texture, mipLevel: 1, slice: 1, region: region, commandBuffer: command)
        var expected = secondLayer
        expected.replaceSubrange(36..<44, with: patch)
        patch.resetBytes(in: 0..<patch.count)
        command.commit()
        let updated = try await TextureTransfer.read(from: texture, mipLevel: 1, slice: 1, commandQueue: queue)
        let neighbor = try await TextureTransfer.read(from: texture, mipLevel: 1, commandQueue: queue)
        let base = try await TextureTransfer.read(from: texture, slice: 1, commandQueue: queue)
        #expect(updated.bytes == expected)
        #expect(neighbor.bytes == firstLayer)
        #expect(base.bytes == baseMip)
    }

    @Test("An array layer or cube face at a selected mip can copy to an ordinary texture",
          arguments: sliceTypes)
    func slices(_ type: MTLTextureType) async throws {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 4, height: 4, mipmapped: true)
        descriptor.textureType = type
        descriptor.arrayLength = type == .typeCube ? 1 : 2
        descriptor.storageMode = .private
        let source = try #require(device.makeTexture(descriptor: descriptor))
        let destination = try texture2D(.rgba8Unorm, side: 2)
        let slice = type == .type2DArray ? 1 : (type == .typeCube ? 4 : 10)
        let colors: [UInt8] = (32..<48).map { UInt8($0) }
        let command = try #require(queue.makeCommandBuffer())
        try upload(colors, to: source, level: 1, slice: slice, region: MTLRegionMake2D(0, 0, 2, 2), command: command)
        try TextureTransfer.encodeCopy(from: source, sourceMipLevel: 1, sourceSlice: slice, to: destination, commandBuffer: command)
        command.commit()
        let face = try await TextureTransfer.read(from: source, mipLevel: 1, slice: slice, commandQueue: queue)
        let copy = try await TextureTransfer.read(from: destination, commandQueue: queue)
        #expect(face.bytes == Data(colors) && copy.bytes == face.bytes)
    }

    private static let sliceTypes: [MTLTextureType] = {
        var types: [MTLTextureType] = [.type2DArray, .typeCube]
        if let device = MTLCreateSystemDefaultDevice(), device.supportsFamily(.apple4) || device.supportsFamily(.mac2) {
            types.append(.typeCubeArray)
        }
        return types
    }()

    @Test("A one-dimensional array layer copies a horizontal subregion")
    func oneDimension() async throws {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type1DArray
        descriptor.pixelFormat = .rgba8Uint
        descriptor.width = 4
        descriptor.arrayLength = 2
        descriptor.storageMode = .private
        let source = try #require(device.makeTexture(descriptor: descriptor))
        descriptor.textureType = .type1D
        descriptor.arrayLength = 1
        descriptor.width = 2
        let destination = try #require(device.makeTexture(descriptor: descriptor))
        let command = try #require(queue.makeCommandBuffer())
        let pixels: [UInt8] = (0..<16).map { UInt8($0 + 7) }
        try upload(pixels, to: source, slice: 1, region: MTLRegionMake1D(0, 4), command: command)
        try TextureTransfer.encodeCopy(
            from: source, sourceSlice: 1, sourceRegion: MTLRegionMake1D(1, 2), to: destination, commandBuffer: command
        )
        command.commit()
        let result = try await TextureTransfer.read(from: destination, commandQueue: queue)
        #expect(result.size.height == 1 && result.bytesPerRow == 8 && result.bytes == Data(pixels[4..<12]))
    }

    @Test("Volume regions preserve the selected Z planes and tight image pitch")
    func volumes() async throws {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type3D
        descriptor.pixelFormat = .rgba8Unorm
        (descriptor.width, descriptor.height, descriptor.depth, descriptor.mipmapLevelCount) = (4, 4, 4, 3)
        descriptor.storageMode = .private
        let source = try #require(device.makeTexture(descriptor: descriptor))
        let destination = try #require(device.makeTexture(descriptor: descriptor))
        let colors: [UInt8] = (0..<32).map { UInt8($0 + 32) }
        let command = try #require(queue.makeCommandBuffer())
        try upload(colors, to: source, level: 1, region: MTLRegionMake3D(0, 0, 0, 2, 2, 2), command: command)
        let region = MTLRegionMake3D(1, 0, 0, 1, 2, 2)
        let offset = MTLOrigin(x: 2, y: 1, z: 2)
        try TextureTransfer.encodeCopy(
            from: source, sourceMipLevel: 1, sourceRegion: region, to: destination,
            destinationOrigin: offset, commandBuffer: command
        )
        command.commit()
        let result = try await TextureTransfer.read(from: destination, region: .init(origin: offset, size: region.size), commandQueue: queue)
        #expect(result.bytesPerRow == 4 && result.bytesPerImage == 8 && result.size.depth == 2)
        #expect(result.bytes == Data([4, 12, 20, 28].flatMap { colors[$0..<($0 + 4)] }))
    }

    @Test("Single-channel depth readback preserves the rendered Float32 values")
    func depth() async throws {
        let texture = try texture2D(.depth32Float, side: 2, usage: .renderTarget)
        let command = try #require(queue.makeCommandBuffer())
        let pass = MTLRenderPassDescriptor()
        pass.depthAttachment.texture = texture
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .store
        pass.depthAttachment.clearDepth = 0.25
        try #require(command.makeRenderCommandEncoder(descriptor: pass)).endEncoding()
        command.commit()
        let result = try await TextureTransfer.read(from: texture, commandQueue: queue)
        #expect(result.pixelFormat == .depth32Float && result.bytesPerRow == 8)
        let expected = Array(repeating: Float(0.25), count: 4).withUnsafeBytes { Data($0) }
        #expect(result.bytes == expected)
    }

    @Test("Compressed transfers return the stored EAC blocks with block-row pitch")
    func compressed() async throws {
        guard device.supportsFamily(.apple2) else { return }
        let blocks: [[UInt8]] = (0..<4).map { index in
            [UInt8(10 + index), 0x10, 0, 0, 0, 0, 0, 0, UInt8(20 + index), 0x10, 0, 0, 0, 0, 0, 0]
        }
        let fixture = KTX1Fixture.data(gl: 0x9272, base: 0x8227, width: 8, height: 8, levels: [[Array(blocks.joined())]])
        let source = try TextureLoader(device: device).loadKTX1(data: fixture, interpretation: .linearData)
        let destination = try texture2D(.eac_rg11Unorm, side: 4)
        let command = try #require(queue.makeCommandBuffer())
        try TextureTransfer.encodeCopy(
            from: source, sourceRegion: MTLRegionMake2D(4, 0, 4, 4), to: destination, commandBuffer: command
        )
        command.commit()
        let result = try await TextureTransfer.read(from: destination, commandQueue: queue)
        #expect(result.pixelFormat == .eac_rg11Unorm && result.bytesPerRow == 16 && result.bytesPerImage == 16)
        #expect(result.bytes == Data(blocks[1]))
        let update = try #require(queue.makeCommandBuffer())
        try TextureTransfer.encodeUpload(Data(blocks[3]), to: destination, commandBuffer: update)
        update.commit()
        let updated = try await TextureTransfer.read(from: destination, commandQueue: queue)
        #expect(updated.bytes == Data(blocks[3]))
        await #expect(throws: TextureTransfer.Failure.invalidRegion) {
            try await TextureTransfer.read(from: source, region: MTLRegionMake2D(1, 0, 4, 4), commandQueue: queue)
        }
    }

    @Test("Invalid coordinates, format, storage, command state and overlapping views fail before encoding")
    func validation() async throws {
        let source = try texture2D(.rgba8Unorm, side: 4, mipmapped: true, usage: .pixelFormatView)
        let otherFormat = try texture2D(.bgra8Unorm, side: 4)
        let view = try #require(source.makeTextureView(pixelFormat: .rgba8Unorm, textureType: .type2D, levels: 1..<2, slices: 0..<1))
        let command = try #require(queue.makeCommandBuffer())
        #expect(throws: TextureTransfer.Failure.incompatibleTextures) {
            try TextureTransfer.encodeCopy(from: source, to: otherFormat, commandBuffer: command)
        }
        #expect(throws: TextureTransfer.Failure.overlappingCopy) {
            try TextureTransfer.encodeCopy(from: source, sourceMipLevel: 1, to: view, commandBuffer: command)
        }
        #expect(throws: TextureTransfer.Failure.overlappingCopy) {
            try TextureTransfer.encodeCopy(from: source, sourceRegion: MTLRegionMake2D(0, 0, 3, 3), to: source,
                                           destinationOrigin: MTLOrigin(x: 1, y: 1, z: 0), commandBuffer: command)
        }
        #expect(throws: TextureTransfer.Failure.invalidRegion) {
            try TextureTransfer.encodeCopy(from: source, to: source, destinationOrigin: MTLOrigin(x: Int.max, y: 0, z: 0), commandBuffer: command)
        }
        await #expect(throws: TextureTransfer.Failure.invalidSubresource) {
            try await TextureTransfer.read(from: source, mipLevel: 3, commandQueue: queue)
        }
        await #expect(throws: TextureTransfer.Failure.invalidSubresource) {
            try await TextureTransfer.read(from: source, slice: -1, commandQueue: queue)
        }
        let untracked = try texture2D(.rgba8Unorm, side: 4, hazardTracking: .untracked)
        await #expect(throws: TextureTransfer.Failure.untrackedTexture) {
            try await TextureTransfer.read(from: untracked, commandQueue: queue)
        }
        if device.supportsTextureSampleCount(4) {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba8Unorm, width: 4, height: 4, mipmapped: false
            )
            descriptor.textureType = .type2DMultisample
            descriptor.sampleCount = 4
            descriptor.storageMode = .private
            descriptor.usage = .renderTarget
            let multisample = try #require(device.makeTexture(descriptor: descriptor))
            await #expect(throws: TextureTransfer.Failure.unsupportedTexture) {
                try await TextureTransfer.read(from: multisample, commandQueue: queue)
            }
        }
        let noRetention = try #require(queue.makeCommandBufferWithUnretainedReferences())
        #expect(throws: TextureTransfer.Failure.commandBuffer) {
            try TextureTransfer.encodeCopy(from: source, to: source, commandBuffer: noRetention)
        }

        let pixel: [UInt8] = [12, 23, 34, 45]
        try upload(pixel, to: source, region: MTLRegionMake2D(0, 0, 1, 1), command: command)
        try TextureTransfer.encodeCopy(from: source, sourceRegion: MTLRegionMake2D(0, 0, 1, 1), to: source,
                                       destinationOrigin: MTLOrigin(x: 2, y: 2, z: 0), commandBuffer: command)
        command.commit()
        #expect(throws: TextureTransfer.Failure.commandBuffer) {
            try TextureTransfer.encodeCopy(from: source, to: source, commandBuffer: command)
        }
        let result = try await TextureTransfer.read(from: source, region: MTLRegionMake2D(2, 2, 1, 1), commandQueue: queue)
        #expect(result.bytes == Data(pixel))
    }

    private func texture2D(
        _ format: MTLPixelFormat, side: Int, mipmapped: Bool = false, usage: MTLTextureUsage = [],
        hazardTracking: MTLHazardTrackingMode = .tracked
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: side, height: side, mipmapped: mipmapped)
        descriptor.storageMode = .private
        descriptor.hazardTrackingMode = hazardTracking
        descriptor.usage = usage
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    private func upload(
        _ bytes: [UInt8], to texture: any MTLTexture, level: Int = 0, slice: Int = 0,
        region: MTLRegion, command: any MTLCommandBuffer
    ) throws {
        try TextureTransfer.encodeUpload(Data(bytes), to: texture, mipLevel: level, slice: slice, region: region,
                                         commandBuffer: command)
    }
}
