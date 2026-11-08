import Foundation
import Metal
internal import FrostfieldShaderTypes

/// Why a ``TriangleRenderer`` could not be created.
public enum RendererError: Error, Equatable, Sendable, CustomStringConvertible {
    /// No Metal device was supplied, for example because
    /// `MTLCreateSystemDefaultDevice()` returned `nil`.
    case noMetalDevice
    /// The device could not create a command queue.
    case commandQueueUnavailable
    /// Frostfield's packaged shader library could not be loaded from the
    /// package's resource bundle.
    case shaderLibraryUnavailable(reason: String)
    /// The packaged shader library has no function with this name.
    case shaderFunctionMissing(name: String)
    /// Metal rejected the render pipeline description.
    case pipelineCreationFailed(reason: String)
    /// The device could not allocate a per-frame buffer.
    case bufferAllocationFailed

    public var description: String {
        switch self {
        case .noMetalDevice:
            "No Metal device is available."
        case .commandQueueUnavailable:
            "The Metal device could not create a command queue."
        case .shaderLibraryUnavailable(let reason):
            "Frostfield's shader library could not be loaded: \(reason)"
        case .shaderFunctionMissing(let name):
            "Frostfield's shader library has no function named \(name)."
        case .pipelineCreationFailed(let reason):
            "The render pipeline could not be created: \(reason)"
        case .bufferAllocationFailed:
            "The Metal device could not allocate a per-frame buffer."
        }
    }
}

/// The render pass and optional drawable one frame renders into.
public struct FrameTarget {
    /// A pass whose first color attachment has the renderer's pixel format.
    public var passDescriptor: MTLRenderPassDescriptor
    /// The drawable to present after rendering, or `nil` for off-screen output.
    public var drawable: (any MTLDrawable)?

    public init(passDescriptor: MTLRenderPassDescriptor, drawable: (any MTLDrawable)?) {
        self.passDescriptor = passDescriptor
        self.drawable = drawable
    }
}

/// What happened to one frame request. Skips are reported, never waited out.
public enum FrameOutcome: Sendable, Equatable {
    /// The frame was committed to the GPU with this one-based frame number.
    case submitted(frame: UInt64)
    /// Every frame slot was still in use by the GPU; nothing was encoded.
    case skippedNoSlot
    /// The target provider returned no usable target; nothing was encoded.
    case skippedNoTarget
    /// Metal could not create a command buffer or encoder; nothing was committed.
    case encodingFailed
}

/// The GPU's report for one submitted frame.
public struct FrameCompletion: Sendable, Equatable {
    /// The frame number from ``FrameOutcome/submitted(frame:)``.
    public var frame: UInt64
    /// Whether the command buffer completed without an error status.
    public var succeeded: Bool
}

/// Renders Frostfield's diagnostic triangle through a conventional Metal
/// command queue with a bounded number of frames in flight.
///
/// The renderer owns its command queue, pipeline, and one uniform buffer per
/// frame slot. A frame writes only to a slot whose previous frame the GPU has
/// completed. When every slot is busy, ``drawFrame(rotation:target:)`` returns
/// ``FrameOutcome/skippedNoSlot`` immediately instead of waiting.
///
/// Use one renderer from one thread or actor at a time. Completion reports
/// arrive on a Metal-owned thread.
public final class TriangleRenderer {
    /// Names of the packaged shader functions.
    static let vertexFunctionName = "ff_triangle_vertex"
    static let fragmentFunctionName = "ff_triangle_fragment"

    /// The bundle that holds the package's compiled `default.metallib`.
    static var shaderBundle: Bundle { .module }

    /// The device every resource of this renderer belongs to.
    public let device: any MTLDevice
    /// The color format the pipeline renders; targets must match it.
    public let colorPixelFormat: MTLPixelFormat
    /// The maximum number of frames that may be in flight at once.
    public var maxFramesInFlight: Int { slots.capacity }
    /// Counters for submitted, completed, and skipped frames.
    public var statistics: FrameStatistics { slots.statistics }

    private let commandQueue: any MTLCommandQueue
    private let pipeline: any MTLRenderPipelineState
    private let uniformBuffers: [any MTLBuffer]
    private let slots: FrameSlots
    private let onFrameCompleted: @Sendable (FrameCompletion) -> Void

    /// Creates the queue, pipeline, and per-frame buffers.
    ///
    /// - Parameters:
    ///   - device: The device to render with. Passing `nil`, for example the
    ///     result of `MTLCreateSystemDefaultDevice()` on a system without
    ///     Metal, throws ``RendererError/noMetalDevice``.
    ///   - colorPixelFormat: The pixel format of every target's first color
    ///     attachment, such as an `MTKView`'s `colorPixelFormat`.
    ///   - maxFramesInFlight: Frame slots, from 1 to 3. A `CAMetalLayer` holds
    ///     at most three drawables, so more slots cannot help on-screen output.
    ///   - onFrameCompleted: Called on a Metal-owned thread after each submitted
    ///     frame completes and its slot is free again.
    public convenience init(
        device: (any MTLDevice)?,
        colorPixelFormat: MTLPixelFormat,
        maxFramesInFlight: Int = 2,
        onFrameCompleted: @escaping @Sendable (FrameCompletion) -> Void = { _ in }
    ) throws(RendererError) {
        try self.init(
            device: device,
            colorPixelFormat: colorPixelFormat,
            maxFramesInFlight: maxFramesInFlight,
            vertexFunctionName: Self.vertexFunctionName,
            fragmentFunctionName: Self.fragmentFunctionName,
            onFrameCompleted: onFrameCompleted
        )
    }

    /// Designated initializer; tests name missing functions to exercise the
    /// failure path.
    init(
        device: (any MTLDevice)?,
        colorPixelFormat: MTLPixelFormat,
        maxFramesInFlight: Int,
        vertexFunctionName: String,
        fragmentFunctionName: String,
        onFrameCompleted: @escaping @Sendable (FrameCompletion) -> Void
    ) throws(RendererError) {
        precondition((1...3).contains(maxFramesInFlight), "maxFramesInFlight must be 1, 2, or 3")
        guard let device else { throw .noMetalDevice }
        guard let commandQueue = device.makeCommandQueue() else { throw .commandQueueUnavailable }
        commandQueue.label = "Frostfield.TriangleRenderer"

        let library = try Self.makeShaderLibrary(device: device)
        guard let vertexFunction = library.makeFunction(name: vertexFunctionName) else {
            throw .shaderFunctionMissing(name: vertexFunctionName)
        }
        guard let fragmentFunction = library.makeFunction(name: fragmentFunctionName) else {
            throw .shaderFunctionMissing(name: fragmentFunctionName)
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Frostfield.Triangle"
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.colorAttachments[0].pixelFormat = colorPixelFormat
        let pipeline: any MTLRenderPipelineState
        do {
            pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            throw .pipelineCreationFailed(reason: String(describing: error))
        }

        var uniformBuffers: [any MTLBuffer] = []
        for slot in 0..<maxFramesInFlight {
            guard let buffer = device.makeBuffer(
                length: MemoryLayout<FFTriangleUniforms>.stride,
                options: .storageModeShared
            ) else { throw .bufferAllocationFailed }
            buffer.label = "Frostfield.TriangleUniforms[\(slot)]"
            uniformBuffers.append(buffer)
        }

        self.device = device
        self.colorPixelFormat = colorPixelFormat
        self.commandQueue = commandQueue
        self.pipeline = pipeline
        self.uniformBuffers = uniformBuffers
        self.slots = FrameSlots(capacity: maxFramesInFlight)
        self.onFrameCompleted = onFrameCompleted
    }

    /// Loads the library Xcode compiles from `Sources/Frostfield/Shaders` into
    /// the package's resource bundle, not the host app's default library.
    static func makeShaderLibrary(device: any MTLDevice) throws(RendererError) -> any MTLLibrary {
        do {
            return try device.makeDefaultLibrary(bundle: shaderBundle)
        } catch {
            throw .shaderLibraryUnavailable(reason: String(describing: error))
        }
    }

    /// The clip-space scale that keeps the triangle's shape in a target of
    /// the given pixel size: the shorter axis spans clip space.
    static func aspectScale(width: Int, height: Int) -> SIMD2<Float> {
        if width >= height {
            SIMD2(Float(height) / Float(width), 1)
        } else {
            SIMD2(1, Float(width) / Float(height))
        }
    }

    /// Encodes and commits one frame, or reports why it did not.
    ///
    /// The renderer first claims a free frame slot without waiting. Only then
    /// does it call `target`, so a host can fetch its drawable as late as
    /// possible and does not hold one while every slot is busy. A `nil` target,
    /// or one without a non-empty color texture, releases the slot unused.
    ///
    /// - Parameters:
    ///   - rotation: Counter-clockwise rotation of the triangle in radians.
    ///   - target: Supplies the render pass and drawable for this frame.
    public func drawFrame(rotation: Float, target: () -> FrameTarget?) -> FrameOutcome {
        guard let slot = slots.acquire() else { return .skippedNoSlot }

        guard let target = target(),
              let colorTexture = target.passDescriptor.colorAttachments[0].texture,
              colorTexture.width > 0, colorTexture.height > 0
        else {
            slots.releaseUnusedForMissingTarget(slot)
            return .skippedNoTarget
        }
        precondition(
            colorTexture.pixelFormat == colorPixelFormat,
            "Target pixel format \(colorTexture.pixelFormat.rawValue) does not match the pipeline's \(colorPixelFormat.rawValue)"
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: target.passDescriptor)
        else {
            slots.releaseUnusedForEncodingFailure(slot)
            return .encodingFailed
        }
        commandBuffer.label = "Frostfield.TriangleFrame"

        // The slot's previous frame has completed, so the GPU no longer reads
        // this buffer.
        let uniforms = FFTriangleUniforms(
            scale: Self.aspectScale(width: colorTexture.width, height: colorTexture.height),
            rotation: rotation,
            padding: 0
        )
        let uniformBuffer = uniformBuffers[slot]
        uniformBuffer.contents().storeBytes(of: uniforms, as: FFTriangleUniforms.self)

        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: Int(FF_TRIANGLE_UNIFORMS_BUFFER_INDEX))
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()

        if let drawable = target.drawable {
            commandBuffer.present(drawable)
        }

        let frame = slots.markSubmitted(slot)
        let slots = self.slots
        let onFrameCompleted = self.onFrameCompleted
        commandBuffer.addCompletedHandler { buffer in
            let succeeded = buffer.status == .completed
            slots.complete(slot, failed: !succeeded)
            onFrameCompleted(FrameCompletion(frame: frame, succeeded: succeeded))
        }
        commandBuffer.commit()
        return .submitted(frame: frame)
    }
}
