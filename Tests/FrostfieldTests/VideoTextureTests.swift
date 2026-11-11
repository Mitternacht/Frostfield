import AVFoundation
import CoreVideo
import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Video frames decoded by AVFoundation, read through Metal and drawn in materials.
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct VideoTextureTests {
    /// Each frame has an sRGB-encoded top half and bottom half color.
    static let frames: [(top: SIMD3<UInt8>, bottom: SIMD3<UInt8>)] = [
        (SIMD3(200, 60, 30), SIMD3(30, 180, 90)),
        (SIMD3(40, 70, 220), SIMD3(230, 210, 40)),
        (SIMD3(128, 128, 128), SIMD3(64, 64, 64)),
    ]
    static let side = 64
    static let frameDuration = CMTime(value: 1, timescale: 10)

    private let fixture: GeometryRenderFixture

    init() throws {
        fixture = try GeometryRenderFixture(side: Self.side)
    }

    @Test func `Video texture detaches its output when released`() throws {
        let item = AVPlayerItem(url: URL(fileURLWithPath: "/unused-video.mov"))
        do {
            let video = try VideoTexture(device: fixture.device, playerItem: item, width: 4, height: 4)
            #expect(item.outputs.contains { $0 === video.output })
        }
        #expect(item.outputs.isEmpty)
    }

    /// Writes an H.264 movie of sRGB-tagged BGRA frames, as a camera or editor would deliver them.
    static func makeMovie() async throws -> URL {
        let url = FileManager.default.temporaryDirectory.appending(path: "frostfield-video-\(UUID().uuidString).mov")
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: side, AVVideoHeightKey: side,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
            ],
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: side, kCVPixelBufferHeightKey as String: side,
        ])
        writer.add(input)
        try #require(writer.startWriting())
        writer.startSession(atSourceTime: .zero)
        for (index, frame) in frames.enumerated() {
            while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(2)) }
            var created: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, try #require(adaptor.pixelBufferPool), &created)
            let buffer = try #require(created)
            CVBufferSetAttachment(buffer, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
            CVBufferSetAttachment(buffer, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_sRGB, .shouldPropagate)
            CVPixelBufferLockBaseAddress(buffer, [])
            let base = try #require(CVPixelBufferGetBaseAddress(buffer)).assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRow(buffer)
            for y in 0..<side {
                let color = y < side / 2 ? frame.top : frame.bottom
                for x in 0..<side {
                    let pixel = base + y * stride + x * 4
                    (pixel[0], pixel[1], pixel[2], pixel[3]) = (color.z, color.y, color.x, 255)
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            try #require(adaptor.append(buffer, withPresentationTime: CMTimeMultiply(frameDuration, multiplier: Int32(index))))
        }
        input.markAsFinished()
        await writer.finishWriting()
        try #require(writer.status == .completed, "\(writer.error?.localizedDescription ?? "writing failed")")
        return url
    }

    /// The linear range an sRGB code decodes to after up to four codes of codec error.
    static func linearRange(_ code: UInt8) -> ClosedRange<Float> {
        func linear(_ value: Int) -> Float {
            let c = Float(min(max(value, 0), 255)) / 255
            return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
        }
        return linear(Int(code) - 4)...linear(Int(code) + 4)
    }

    /// A camera-filling quad at depth 1 with top-left texture coordinates.
    private func screenQuad(_ material: MeshMaterial) throws -> MeshRenderer {
        let corners: [(SIMD3<Float>, SIMD2<Float>)] = [
            (SIMD3(-1, -1, 0), SIMD2(0, 1)), (SIMD3(1, -1, 0), SIMD2(1, 1)),
            (SIMD3(1, 1, 0), SIMD2(1, 0)), (SIMD3(-1, 1, 0), SIMD2(0, 0)),
        ]
        let geometry = try MeshGeometry(
            vertices: corners.map { MeshVertex(position: $0.0, normal: SIMD3(0, 0, 1), textureCoordinate: $0.1, tangent: SIMD4(1, 0, 0, 1)) },
            indices: [0, 1, 2, 0, 2, 3]
        )
        return try MeshRenderer(device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: material)
    }

    private func render(_ renderer: MeshRenderer, before: ((any MTLCommandBuffer) throws -> Void)? = nil) throws -> [SIMD4<Float>] {
        let target = try fixture.makeTarget()
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        try before?(buffer)
        try SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -1)), mesh: renderer)], camera: fixture.camera
        ).encode(pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: buffer)
        try fixture.finish(buffer)
        return try fixture.pixels(of: target.color)
    }

    private func expectFrame(_ pixels: [SIMD4<Float>], _ index: Int, _ comment: String) {
        let expected = Self.frames[index]
        for (row, color) in [(16, expected.top), (48, expected.bottom)] {
            let pixel = fixture.pixel(pixels, x: 32, y: row)
            for channel in 0..<3 {
                #expect(Self.linearRange(color[channel]).contains(pixel[channel]), "\(comment) frame \(index) row \(row) channel \(channel): \(pixel)")
            }
            #expect(pixel.w == 1)
        }
    }

    @Test("Decoded frames share their IOSurface with Metal and sample the source colors in linear sRGB")
    func zeroCopyFrames() async throws {
        let url = try await Self.makeMovie()
        defer { try? FileManager.default.removeItem(at: url) }
        let asset = AVURLAsset(url: url)
        let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: VideoColor.outputSettings())
        reader.add(output)
        try #require(reader.startReading())
        let cache = try VideoTextureCache(device: fixture.device)
        var index = 0
        while let sample = output.copyNextSampleBuffer() {
            let pixelBuffer = try #require(CMSampleBufferGetImageBuffer(sample))
            let frame = try cache.frame(from: pixelBuffer, presentationTime: CMSampleBufferGetPresentationTimeStamp(sample))
            // The texture is the pixel buffer's own memory, not a copy.
            let surface = try #require(CVPixelBufferGetIOSurface(pixelBuffer)?.takeUnretainedValue())
            #expect(frame.texture.iosurface === surface)
            #expect(frame.texture.pixelFormat == .bgra8Unorm_srgb)
            #expect(frame.isSRGBEncoded)
            #expect(frame.presentationTime == CMTimeMultiply(Self.frameDuration, multiplier: Int32(index)))
            let pixels = try render(try screenQuad(.textured(frame.texture))) { frame.retain(until: $0) }
            expectFrame(pixels, index, "reader")
            index += 1
        }
        #expect(index == Self.frames.count)
        #expect(reader.status == .completed)
    }

    @Test("A video texture holds opaque black on every mip level before its first frame")
    func blackBeforeFirstFrame() throws {
        let item = AVPlayerItem(url: FileManager.default.temporaryDirectory.appending(path: "frostfield-missing-\(UUID().uuidString).mov"))
        let video = try VideoTexture(
            device: fixture.device, playerItem: item, width: 40, height: 24, mipmapped: true, commandQueue: fixture.queue
        )
        #expect(video.texture.mipmapLevelCount == 6 && video.frameCount == 0)
        let buffer = try #require(fixture.device.makeBuffer(length: 40 * 24 * 4, options: .storageModeShared))
        for level in 0..<video.texture.mipmapLevelCount {
            let width = max(1, 40 >> level), height = max(1, 24 >> level)
            let command = try #require(fixture.queue.makeCommandBuffer())
            let blit = try #require(command.makeBlitCommandEncoder())
            blit.copy(
                from: video.texture, sourceSlice: 0, sourceLevel: level, sourceOrigin: MTLOrigin(),
                sourceSize: MTLSize(width: width, height: height, depth: 1), to: buffer, destinationOffset: 0,
                destinationBytesPerRow: width * 4, destinationBytesPerImage: width * height * 4
            )
            blit.endEncoding()
            try fixture.finish(command)
            let texels = UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: SIMD4<UInt8>.self), count: width * height)
            #expect(texels.allSatisfy { $0 == SIMD4(0, 0, 0, 255) }, "level \(level): \(texels.first.map(String.init(describing:)) ?? "")")
        }
        // A material bound before any frame draws black, not uninitialized memory.
        let pixels = try render(try screenQuad(.textured(video.texture)))
        #expect(fixture.pixel(pixels, x: 32, y: 32) == SIMD4(0, 0, 0, 1))
        let other = try #require(MTLCreateSystemDefaultDevice()?.makeCommandQueue())
        if other.device !== fixture.device {
            #expect(throws: VideoTexture.Failure.wrongDevice) {
                try VideoTexture(device: fixture.device, playerItem: item, width: 4, height: 4, commandQueue: other)
            }
        }
    }

    @Test("A player-driven video texture updates materials prepared once, including a surface graph texture")
    @MainActor
    func playerTexture() async throws {
        let url = try await Self.makeMovie()
        defer { try? FileManager.default.removeItem(at: url) }
        let item = AVPlayerItem(url: url)
        let video = try VideoTexture(device: fixture.device, playerItem: item, width: Self.side, height: Self.side)
        let player = AVPlayer(playerItem: item)
        for _ in 0..<500 where item.status != .readyToPlay { try await Task.sleep(for: .milliseconds(10)) }
        try #require(item.status == .readyToPlay, "\(item.error?.localizedDescription ?? "the item did not become ready")")

        // Both materials bind the stable texture once, before any frame arrives.
        let textured = try screenQuad(.textured(video.texture))
        let graph = try SurfaceShaderGraph(
            nodes: [.input(.uv0), .sampleTexture(slot: 0, coordinates: 0), .swizzle(1, "rgb")],
            textures: [ShaderTextureInput("video")], outputs: .init(emission: 2)
        )
        let surface = try screenQuad(.physicallyBased(try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1), alphaMode: .opaque,
            surfaceGraph: graph, surfaceTextures: ["video": video.texture]
        )))

        for index in Self.frames.indices {
            let time = CMTimeMultiply(Self.frameDuration, multiplier: Int32(index))
            try #require(await item.seek(to: time, toleranceBefore: .zero, toleranceAfter: .zero))
            var updated = false
            for _ in 0..<300 where !updated {
                let buffer = try #require(fixture.queue.makeCommandBuffer())
                updated = try video.update(itemTime: time, commandBuffer: buffer)
                if updated { try fixture.finish(buffer) } else { try await Task.sleep(for: .milliseconds(10)) }
            }
            try #require(updated, "No frame arrived for \(time.seconds) s")
            #expect(video.currentFrame?.isSRGBEncoded == true)
            #expect(video.frameCount == index + 1)
            expectFrame(try render(textured), index, "textured material")
            let surfacePixels = try render(surface)
            expectFrame(surfacePixels, index, "surface graph")
        }
        // A time with no new frame leaves the texture unchanged.
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        #expect(try video.update(itemTime: CMTimeMultiply(Self.frameDuration, multiplier: 2), commandBuffer: buffer) == false)
        _ = player
    }
}
