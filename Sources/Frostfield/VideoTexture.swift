import AVFoundation
import CoreVideo
import Foundation
import Metal

/// Color handling shared by video sources.
///
/// Frostfield asks AVFoundation for 8-bit BGRA frames converted to BT.709 (sRGB) primaries
/// and the IEC sRGB transfer function, then views them as `bgra8Unorm_srgb`, so shaders
/// sample linear sRGB like any other color texture. Without the explicit transfer,
/// AVFoundation returns BT.709-encoded values that sampled as sRGB would be off by up
/// to about 12 code values in the midtones.
public enum VideoColor {
    /// Pixel buffer attributes and color properties for `AVPlayerItemVideoOutput` and
    /// `AVAssetReaderTrackOutput`. Width and height scale the decoded frames when given.
    public static func outputSettings(width: Int? = nil, height: Int? = nil) -> [String: any Sendable] {
        var settings: [String: any Sendable] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_IEC_sRGB,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
            ],
        ]
        if let width { settings[kCVPixelBufferWidthKey as String] = width }
        if let height { settings[kCVPixelBufferHeightKey as String] = height }
        return settings
    }

    /// The pixel format Frostfield views video frames with.
    public static let pixelFormat: MTLPixelFormat = .bgra8Unorm_srgb
}

/// One video frame as a Metal texture that aliases the pixel buffer's IOSurface, with no
/// copy, like the texture behind Three.js `VideoFrameTexture`. Keep the frame alive until
/// GPU work that reads its texture completes, for example with `retain(until:)`.
public final class VideoFrame {
    /// A `bgra8Unorm_srgb` view of the pixel buffer; sampling returns linear sRGB.
    public let texture: any MTLTexture
    public let pixelBuffer: CVPixelBuffer
    /// The frame's time in its source, or `.invalid` when unknown.
    public let presentationTime: CMTime
    /// The pixel buffer's transfer function tag, such as `kCVImageBufferTransferFunction_sRGB`.
    public let transferFunction: String?
    /// The pixel buffer's color primaries tag.
    public let colorPrimaries: String?
    private let metalTexture: CVMetalTexture

    fileprivate init(texture: any MTLTexture, metalTexture: CVMetalTexture, pixelBuffer: CVPixelBuffer, presentationTime: CMTime) {
        self.texture = texture
        self.metalTexture = metalTexture
        self.pixelBuffer = pixelBuffer
        self.presentationTime = presentationTime
        transferFunction = CVBufferCopyAttachment(pixelBuffer, kCVImageBufferTransferFunctionKey, nil) as? String
        colorPrimaries = CVBufferCopyAttachment(pixelBuffer, kCVImageBufferColorPrimariesKey, nil) as? String
    }

    /// Whether the pixel buffer was converted to the sRGB transfer function that `texture` decodes.
    public var isSRGBEncoded: Bool {
        transferFunction == (kCVImageBufferTransferFunction_sRGB as String)
    }

    /// Keeps the frame, and so its pixel buffer, alive until the command buffer completes.
    public func retain(until commandBuffer: any MTLCommandBuffer) {
        let box = FrameBox(self)
        commandBuffer.addCompletedHandler { _ in _ = box }
    }
}

/// Holds a frame in a command buffer's completion handler, which runs on another thread
/// only to release it.
private final class FrameBox: @unchecked Sendable {
    let frame: VideoFrame

    init(_ frame: VideoFrame) {
        self.frame = frame
    }
}

/// Wraps CoreVideo pixel buffers as Metal textures without copying, through a
/// `CVMetalTextureCache`, like Three.js `VideoFrameTexture`. Use it for frames from
/// `AVAssetReader`, `AVCaptureVideoDataOutput` or your own pixel buffer pools.
public final class VideoTextureCache {
    public let device: any MTLDevice
    private let cache: CVMetalTextureCache

    public init(device: any MTLDevice) throws {
        var cache: CVMetalTextureCache?
        let status = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
        guard status == kCVReturnSuccess, let cache else { throw VideoTexture.Failure.textureCache(status) }
        self.device = device
        self.cache = cache
    }

    /// Creates a texture that shares a BGRA pixel buffer's memory. The buffer must be
    /// IOSurface-backed and Metal compatible, as `VideoColor.outputSettings` requests.
    public func frame(from pixelBuffer: CVPixelBuffer, presentationTime: CMTime = .invalid) throws -> VideoFrame {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else {
            throw VideoTexture.Failure.pixelFormat(CVPixelBufferGetPixelFormatType(pixelBuffer))
        }
        guard CVPixelBufferGetIOSurface(pixelBuffer) != nil else { throw VideoTexture.Failure.notIOSurfaceBacked }
        let width = CVPixelBufferGetWidth(pixelBuffer), height = CVPixelBufferGetHeight(pixelBuffer)
        var metalTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, pixelBuffer, nil, VideoColor.pixelFormat, width, height, 0, &metalTexture
        )
        guard status == kCVReturnSuccess, let metalTexture, let texture = CVMetalTextureGetTexture(metalTexture) else {
            throw VideoTexture.Failure.textureCache(status)
        }
        return VideoFrame(texture: texture, metalTexture: metalTexture, pixelBuffer: pixelBuffer, presentationTime: presentationTime)
    }

    /// Releases cached textures that no frame uses any more.
    public func flush() {
        CVMetalTextureCacheFlush(cache, 0)
    }
}

/// Plays an `AVPlayerItem` into one stable texture that any material slot or surface graph
/// texture can bind once, like Three.js `VideoTexture`.
///
/// Decoded frames arrive as IOSurface-backed pixel buffers that Metal reads without a CPU
/// copy; `update` encodes one GPU blit of a new frame into `texture` (and its mip chain when
/// mipmapped), so materials prepared with `texture` show each frame without being rebuilt.
/// Browsers running Three.js also copy each frame on the GPU. Colors follow `VideoColor`.
/// Until the first frame arrives, `texture` holds opaque black.
///
/// Drive it from the render loop: pass the display link's target presentation host time
/// to `update(hostTime:commandBuffer:)`. Use one instance from one thread at a time.
/// Submit texture writes and dependent reads to the same queue in frame order, or
/// synchronize across queues with Metal events.
public final class VideoTexture {
    public enum Failure: LocalizedError, Equatable {
        case textureCache(CVReturn)
        case pixelFormat(OSType)
        case notIOSurfaceBacked
        case size
        case frameSize(width: Int, height: Int)
        case wrongDevice
        case resource(String)

        public var errorDescription: String? {
            switch self {
            case .textureCache(let status): "Core Video could not create a Metal texture (status \(status))."
            case .pixelFormat(let format): "Video frames must be 32-bit BGRA; got pixel format \(format)."
            case .notIOSurfaceBacked: "Video frames must be IOSurface-backed to share memory with Metal."
            case .size: "A video texture needs a positive width and height of at most 16384."
            case .frameSize(let width, let height): "A \(width) × \(height) frame does not match the video texture."
            case .wrongDevice: "The command buffer must use the video texture's device."
            case .resource(let name): "Metal could not create the video \(name)."
            }
        }
    }

    public let device: any MTLDevice
    public let playerItem: AVPlayerItem
    /// The output attached to `playerItem`, delivering frames at `texture`'s size.
    public let output: AVPlayerItemVideoOutput
    /// A private `bgra8Unorm_srgb` texture that holds the latest frame. Bind it in materials.
    public let texture: any MTLTexture
    /// The latest frame copied into `texture`, or nil before the first.
    public private(set) var currentFrame: VideoFrame?
    /// Frames copied so far.
    public private(set) var frameCount = 0
    private let cache: VideoTextureCache

    /// Attaches an output to the item that decodes frames at `width` × `height`; use the
    /// video track's `naturalSize` for full resolution or a smaller size to scale on decode.
    /// Mipmapped textures regenerate their chain after each frame.
    ///
    /// Every level of `texture` is cleared to opaque black on `commandQueue`, or on a new
    /// queue, and this call waits for the clear, so materials that bind the texture before
    /// the first frame sample black rather than uninitialized memory.
    public init(
        device: any MTLDevice, playerItem: AVPlayerItem, width: Int, height: Int, mipmapped: Bool = false,
        commandQueue: (any MTLCommandQueue)? = nil
    ) throws {
        guard (1...16_384).contains(width), (1...16_384).contains(height) else { throw Failure.size }
        if let commandQueue, commandQueue.device !== device { throw Failure.wrongDevice }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: VideoColor.pixelFormat, width: width, height: height, mipmapped: mipmapped
        )
        descriptor.storageMode = .private
        descriptor.usage = [.shaderRead, .renderTarget]
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.resource("texture") }
        texture.label = "Frostfield video"
        guard let queue = commandQueue ?? device.makeCommandQueue(), let clear = queue.makeCommandBuffer() else {
            throw Failure.resource("clear command buffer")
        }
        for level in 0..<texture.mipmapLevelCount {
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = texture
            pass.colorAttachments[0].level = level
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            guard let encoder = clear.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.resource("clear pass") }
            encoder.endEncoding()
        }
        clear.commit()
        clear.waitUntilCompleted()
        guard clear.status == .completed else { throw Failure.resource("cleared texture") }
        cache = try VideoTextureCache(device: device)
        output = AVPlayerItemVideoOutput(outputSettings: VideoColor.outputSettings(width: width, height: height))
        self.device = device
        self.playerItem = playerItem
        self.texture = texture
        playerItem.add(output)
    }

    deinit {
        playerItem.remove(output)
    }

    /// The item time to show at a host time, such as a display link's target presentation time.
    public func itemTime(forHostTime hostTime: CFTimeInterval) -> CMTime {
        output.itemTime(forHostTime: hostTime)
    }

    /// Copies the frame for a host time into `texture` when the output has a new one.
    /// Returns whether the texture changed.
    @discardableResult
    public func update(hostTime: CFTimeInterval, commandBuffer: any MTLCommandBuffer) throws -> Bool {
        try update(itemTime: itemTime(forHostTime: hostTime), commandBuffer: commandBuffer)
    }

    /// Copies the frame for an item time into `texture` when the output has a new one,
    /// encoding a blit (and mipmap generation) into the command buffer.
    @discardableResult
    public func update(itemTime: CMTime, commandBuffer: any MTLCommandBuffer) throws -> Bool {
        guard commandBuffer.device === device else { throw Failure.wrongDevice }
        guard itemTime.isValid, output.hasNewPixelBuffer(forItemTime: itemTime) else { return false }
        var displayTime = CMTime.invalid
        guard let pixelBuffer = output.copyPixelBuffer(forItemTime: itemTime, itemTimeForDisplay: &displayTime) else {
            return false
        }
        let frame = try cache.frame(from: pixelBuffer, presentationTime: displayTime)
        guard frame.texture.width == texture.width, frame.texture.height == texture.height else {
            throw Failure.frameSize(width: frame.texture.width, height: frame.texture.height)
        }
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw Failure.resource("blit encoder") }
        blit.copy(
            from: frame.texture, sourceSlice: 0, sourceLevel: 0, to: texture, destinationSlice: 0, destinationLevel: 0,
            sliceCount: 1, levelCount: 1
        )
        if texture.mipmapLevelCount > 1 { blit.generateMipmaps(for: texture) }
        blit.endEncoding()
        frame.retain(until: commandBuffer)
        currentFrame = frame
        frameCount += 1
        if frameCount.isMultiple(of: 60) { cache.flush() }
        return true
    }
}
