import Dispatch
import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TextureOutputTests {
    @Test
    func `Color grade bypass preserves pixels and enabled grade preserves alpha`() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let grade = try ColorGrade(device: device, destinationPixelFormat: .rgba16Float)
        let source = try makeTexture(device, usage: [.shaderRead, .renderTarget])
        let destination = try makeTexture(device, usage: .renderTarget)
        let pixels: [Float16] = [
            0.25, 0.125, 0.125, 0.5,  0.0625, 0.25, 0, 1,
            0, 0, 0, 0,  0.25, 0, 0.5, 0.75,
        ]
        let region = MTLRegionMake2D(0, 0, 2, 2)
        pixels.withUnsafeBytes { source.replace(region: region, mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 16) }

        let bypass = try #require(queue.makeCommandBuffer())
        #expect(throws: ColorGrade.Failure.incompatibleTextures) {
            try grade.encode(source: source, destination: source, strength: 0, commandBuffer: bypass)
        }
        try grade.encode(source: source, destination: destination, strength: 0, commandBuffer: bypass)
        bypass.commit()
        bypass.waitUntilCompleted()
        try #require(bypass.status == .completed)
        #expect(readPixels(destination) == pixels)

        let enabled = try #require(queue.makeCommandBuffer())
        try grade.encode(source: source, destination: destination, strength: 1, commandBuffer: enabled)
        enabled.commit()
        enabled.waitUntilCompleted()
        try #require(enabled.status == .completed)
        let graded = readPixels(destination)
        #expect(Array(graded.prefix(4)) == [0.28125, 0.125, 0.1015625, 0.5])
        #expect(graded[8...11].allSatisfy { $0 == 0 })
        #expect([graded[3], graded[7], graded[11], graded[15]] == [0.5, 1, 0, 0.75])
    }

    @Test
    func `A dropped frame holds its slot until its producer finishes`() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let output = try TextureOutput(queue: queue, capacity: 1)
        let gate = try #require(device.makeSharedEvent())
        let finished = DispatchSemaphore(value: 0)
        defer { gate.signaledValue = 1 }

        var frame = try output.produce(width: 4, height: 4) { texture, buffer in
            buffer.encodeWaitForEvent(gate, value: 1)
            try clear(texture, in: buffer)
            buffer.addCompletedHandler { _ in finished.signal() }
        }
        #expect(frame?.readiness == .pending)
        #expect(throws: TextureOutput.Failure.frameNotReady) {
            try output.consume(try #require(frame)) { _, _ in Issue.record("Pending frame reached its consumer") }
        }
        let id = try #require(frame?.id)
        frame = nil
        #expect(try output.produce(width: 8, height: 4) { _, _ in Issue.record("In-flight producer slot was reused") } == nil)

        gate.signaledValue = 1
        try #require(finished.wait(timeout: .now() + 5) == .success)
        let replacement = try #require(try output.produce(width: 8, height: 4, encode: clear))
        #expect(replacement.width == 8)
        #expect(replacement.height == 4)
        #expect(replacement.id > id)
        #expect(replacement.pixelFormat == .rgba16Float)
        #expect(replacement.device === device)
    }

    @Test
    func `A delayed consumer reads the original frame and retains the slot`() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let otherQueue = try #require(device.makeCommandQueue())
        let output = try TextureOutput(queue: queue, capacity: 1)
        let otherOutput = try TextureOutput(queue: otherQueue, capacity: 1)
        let gate = try #require(device.makeSharedEvent())
        let readback = try #require(device.makeBuffer(length: 256, options: .storageModeShared))
        let produced = DispatchSemaphore(value: 0)
        let consumed = DispatchSemaphore(value: 0)
        defer { gate.signaledValue = 1 }

        var frame = try output.produce(width: 4, height: 4) { texture, buffer in
            try clear(texture, in: buffer)
            buffer.addCompletedHandler { _ in produced.signal() }
        }
        try #require(produced.wait(timeout: .now() + 5) == .success)
        #expect(frame?.readiness == .ready)
        #expect(throws: TextureOutput.Failure.foreignFrame) {
            try otherOutput.consume(try #require(frame)) { _, _ in Issue.record("Foreign frame reached its consumer") }
        }

        try output.consume(try #require(frame), completed: { error in
            #expect(error == nil)
            consumed.signal()
        }) { texture, buffer in
            buffer.encodeWaitForEvent(gate, value: 1)
            let blit = try #require(buffer.makeBlitCommandEncoder())
            blit.copy(
                from: texture, sourceSlice: 0, sourceLevel: 0,
                sourceOrigin: .init(x: 0, y: 0, z: 0), sourceSize: .init(width: 1, height: 1, depth: 1),
                to: readback, destinationOffset: 0, destinationBytesPerRow: 256, destinationBytesPerImage: 256
            )
            blit.endEncoding()
        }
        frame = nil
        #expect(try output.produce(width: 4, height: 4) { _, _ in Issue.record("In-flight consumer slot was reused") } == nil)

        gate.signaledValue = 1
        try #require(consumed.wait(timeout: .now() + 5) == .success)
        let bytes = readback.contents().bindMemory(to: UInt16.self, capacity: 4)
        #expect((0..<4).map { Float16(bitPattern: bytes[$0]) } == [0.25, 0.125, 0.0625, 0.5])
        _ = try #require(try output.produce(width: 4, height: 4, encode: clear))
    }

    private func clear(_ texture: any MTLTexture, in buffer: any MTLCommandBuffer) throws {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = .init(red: 0.25, green: 0.125, blue: 0.0625, alpha: 0.5)
        let encoder = try #require(buffer.makeRenderCommandEncoder(descriptor: pass))
        encoder.endEncoding()
    }

    private func makeTexture(_ device: any MTLDevice, usage: MTLTextureUsage) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: 2, height: 2, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = usage
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    private func readPixels(_ texture: any MTLTexture) -> [Float16] {
        var result = [Float16](repeating: 0, count: 16)
        result.withUnsafeMutableBytes {
            texture.getBytes($0.baseAddress!, bytesPerRow: 16, from: MTLRegionMake2D(0, 0, 2, 2), mipmapLevel: 0)
        }
        return result
    }
}
