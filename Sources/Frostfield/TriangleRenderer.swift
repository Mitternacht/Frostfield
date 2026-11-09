import Foundation
import Metal
internal import FrostfieldShaderTypes

/// Why a ``TriangleRenderer`` or ``MeshRenderer`` could not be created, or
/// why a renderer could not create a resource.
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
    /// The device could not allocate a per-frame or mesh buffer.
    case bufferAllocationFailed
    /// The device could not create a depth-stencil state.
    case depthStencilStateUnavailable

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
            "The Metal device could not allocate a buffer."
        case .depthStencilStateUnavailable:
            "The Metal device could not create a depth-stencil state."
        }
    }
}

/// The render pass and optional drawable one frame renders into.
///
/// The pass must match the renderer's pipelines: a single-sample first color
/// attachment of the renderer's color format, a depth attachment only for a
/// ``MeshRenderer`` (of its ``MeshRenderer/depthPixelFormat``), and no stencil
/// attachment. A mismatched target stops on a precondition before anything is
/// encoded.
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
/// The triangle can be drawn two ways:
///
/// - ``drawFrame(rotation:target:)`` rotates it in clip space and keeps its
///   shape in any aspect ratio. The host lifecycle fixtures use this.
/// - ``drawFrame(transform:camera:target:)`` treats it as model-space geometry
///   in the z = 0 plane, places it with a ``Transform``, and projects it
///   through a perspective ``Camera``.
///
/// The renderer owns its command queue, both pipelines, and one uniform buffer
/// per frame slot, all created by the initializer. Drawing writes the frame's
/// values into its slot's buffer and creates no GPU object, so
/// ``resourceStatistics`` stays constant while transforms and cameras change.
/// A frame writes only to a slot whose previous frame the GPU has completed.
/// When every slot is busy, drawing returns ``FrameOutcome/skippedNoSlot``
/// immediately instead of waiting.
///
/// Use one renderer from one thread or actor at a time. Completion reports
/// arrive on a Metal-owned thread.
public final class TriangleRenderer {
    /// Names of the packaged shader functions.
    static let vertexFunctionName = "ff_triangle_vertex"
    static let cameraVertexFunctionName = "ff_triangle_camera_vertex"
    static let fragmentFunctionName = "ff_triangle_fragment"

    /// The length of each per-slot uniform buffer: large enough for the
    /// record of either drawing path, written at offset 0.
    static let uniformBufferLength = max(
        MemoryLayout<FFTriangleUniforms>.stride,
        MemoryLayout<FFTransformUniforms>.stride
    )

    /// The bundle that holds the package's compiled `default.metallib`.
    static var shaderBundle: Bundle { .module }

    /// The device every resource of this renderer belongs to.
    public let device: any MTLDevice
    /// The color format the pipeline renders; targets must match it.
    public let colorPixelFormat: MTLPixelFormat
    /// The maximum number of frames that may be in flight at once.
    public var maxFramesInFlight: Int { submitter.slots.capacity }
    /// Counters for submitted, completed, and skipped frames.
    public var statistics: FrameStatistics { submitter.slots.statistics }
    /// Counters for the pipelines and buffers this renderer created.
    public let resourceStatistics: ResourceStatistics

    private let submitter: FrameSubmitter
    private let pipeline: any MTLRenderPipelineState
    private let cameraPipeline: any MTLRenderPipelineState
    /// One uniform buffer per frame slot, indexed by slot.
    var uniformBuffers: [any MTLBuffer] { submitter.uniformBuffers }

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
        cameraVertexFunctionName: String = TriangleRenderer.cameraVertexFunctionName,
        fragmentFunctionName: String,
        onFrameCompleted: @escaping @Sendable (FrameCompletion) -> Void
    ) throws(RendererError) {
        guard let device else { throw .noMetalDevice }
        var resources = ResourceStatistics()
        let submitter = try FrameSubmitter(
            device: device,
            rendererName: "TriangleRenderer",
            colorPixelFormat: colorPixelFormat,
            depthPixelFormat: nil,
            maxFramesInFlight: maxFramesInFlight,
            uniformBufferLength: Self.uniformBufferLength,
            resources: &resources,
            onFrameCompleted: onFrameCompleted
        )

        let library = try Self.makeShaderLibrary(device: device)
        guard let vertexFunction = library.makeFunction(name: vertexFunctionName) else {
            throw .shaderFunctionMissing(name: vertexFunctionName)
        }
        guard let cameraVertexFunction = library.makeFunction(name: cameraVertexFunctionName) else {
            throw .shaderFunctionMissing(name: cameraVertexFunctionName)
        }
        guard let fragmentFunction = library.makeFunction(name: fragmentFunctionName) else {
            throw .shaderFunctionMissing(name: fragmentFunctionName)
        }

        func makePipeline(label: String, vertexFunction: any MTLFunction) throws(RendererError) -> any MTLRenderPipelineState {
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.label = label
            descriptor.vertexFunction = vertexFunction
            descriptor.fragmentFunction = fragmentFunction
            descriptor.colorAttachments[0].pixelFormat = colorPixelFormat
            do {
                let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
                resources.renderPipelinesCreated += 1
                return pipeline
            } catch {
                throw .pipelineCreationFailed(reason: String(describing: error))
            }
        }
        let pipeline = try makePipeline(label: "Frostfield.Triangle", vertexFunction: vertexFunction)
        let cameraPipeline = try makePipeline(label: "Frostfield.TriangleCamera", vertexFunction: cameraVertexFunction)

        self.device = device
        self.colorPixelFormat = colorPixelFormat
        self.submitter = submitter
        self.pipeline = pipeline
        self.cameraPipeline = cameraPipeline
        self.resourceStatistics = resources
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

    /// Encodes and commits one frame of the clip-space triangle, or reports
    /// why it did not.
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
        submitter.submit(label: "Frostfield.TriangleFrame", target: target) { encoder, uniformBuffer, _, colorTexture in
            let uniforms = FFTriangleUniforms(
                scale: Self.aspectScale(width: colorTexture.width, height: colorTexture.height),
                rotation: rotation,
                padding: 0
            )
            uniformBuffer.contents().storeBytes(of: uniforms, as: FFTriangleUniforms.self)
            encoder.setRenderPipelineState(pipeline)
            encoder.setVertexBuffer(uniformBuffer, offset: 0, index: Int(FF_TRIANGLE_UNIFORMS_BUFFER_INDEX))
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        }
    }

    /// Encodes and commits one frame of the triangle placed by `transform`
    /// and viewed through `camera`, or reports why it did not.
    ///
    /// The triangle's model-space corners lie in the z = 0 plane, centred on
    /// the origin with a circumradius of 0.75: a red apex at (0, 0.75) and
    /// green and blue corners at (∓0.65, -0.375). The renderer computes
    /// `projection * view * model` for the target's aspect ratio and writes it
    /// into the frame slot's existing uniform buffer. Faces are not culled, so
    /// the triangle is visible from both sides.
    ///
    /// Slots and targets behave as in ``drawFrame(rotation:target:)``.
    public func drawFrame(transform: Transform, camera: Camera, target: () -> FrameTarget?) -> FrameOutcome {
        submitter.submit(label: "Frostfield.TriangleCameraFrame", target: target) { encoder, uniformBuffer, _, colorTexture in
            // The target is non-empty, so the ratio is finite and positive.
            let aspectRatio = Float(colorTexture.width) / Float(colorTexture.height)
            let uniforms = FFTransformUniforms(
                modelViewProjection: camera.uncheckedViewProjectionMatrix(aspectRatio: aspectRatio) * transform.matrix
            )
            uniformBuffer.contents().storeBytes(of: uniforms, as: FFTransformUniforms.self)
            encoder.setRenderPipelineState(cameraPipeline)
            encoder.setVertexBuffer(uniformBuffer, offset: 0, index: Int(FF_TRANSFORM_UNIFORMS_BUFFER_INDEX))
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        }
    }
}
