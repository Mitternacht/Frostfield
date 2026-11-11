import Foundation
import Metal
import simd

/// Temporal anti-aliasing for jittered HDR frames, like Three.js `TAARenderPass`
/// and the `traa` TSL node, and progressive supersampling of a still view like
/// `TAARenderPass` accumulation.
///
/// Each frame, set `SceneGraph.projectionJitter` to `nextJitter`, a Halton (2, 3)
/// sub-pixel offset, render the scene, and encode this pass with the frame's color,
/// depth and unjittered view-projection. The pass reconstructs each pixel's surface
/// from depth, reprojects it into the previous frame with the previous unjittered
/// matrix, reads the history there with a Catmull-Rom filter, clips that history
/// to the YCoCg bounds of the current 3 × 3 neighborhood, and blends the two. It
/// uses the nearest depth in the neighborhood for reprojection so edges move with
/// the foreground. Because the history accumulates the jittered samples, a still
/// camera converges toward a box-filtered supersampled image.
///
/// Supply the raster motion attachment to reproject moving objects and reject
/// history across depth discontinuities. Without it, reprojection follows the
/// camera. Run the pass on scene-linear HDR color before `HDRDisplay`. Call
/// `reset()` after a cut, a large camera jump, or a scene change, and after
/// discarding a command buffer that held an encode.
///
/// The object owns two history textures, RGBA16Float while clipping and
/// RGBA32Float without it, and is not thread-safe. Encode on retained command
/// buffers from one queue, in frame order.
public final class TemporalAntiAliasing {
    /// Three.js r187-style weights and depth-edge handling for motion resolves.
    public struct TRAASettings: Sendable, Hashable {
        /// Motion length in input pixels that gives full current-frame weight.
        public var maxVelocityLength: Float
        /// Adds current-frame weight for fractional unjittered motion.
        public var useSubpixelCorrection: Bool
        /// A 3 × 3 normalized depth range above this value enables the
        /// depth-validated history footprint at silhouettes.
        public var edgeDepthDifference: Float

        public init(maxVelocityLength: Float = 128, useSubpixelCorrection: Bool = true, edgeDepthDifference: Float = 0.001) {
            self.maxVelocityLength = maxVelocityLength
            self.useSubpixelCorrection = useSubpixelCorrection
            self.edgeDepthDifference = edgeDepthDifference
        }

        fileprivate var isValid: Bool {
            maxVelocityLength.isNormal && maxVelocityLength > 0 && (0...1).contains(edgeDepthDifference)
        }
    }

    /// Blending and jitter parameters.
    public struct Settings: Sendable, Hashable {
        /// The weight of the current frame once history has built up, in
        /// (0, 1]. Until then each frame is weighted `1 / (frames + 1)`, an exact
        /// running mean. Smaller values converge further and respond more slowly.
        public var currentFrameWeight: Float
        /// Clips history to the current neighborhood's YCoCg and alpha bounds.
        /// Turn it off only for a still camera and scene, for supersampling. Without
        /// clipping, history is kept in RGBA32Float so long means stay exact, and is
        /// read at the nearest texel of the reprojected position, which is the
        /// pixel itself for a still camera. Changing it starts the history over.
        public var clipsHistory: Bool
        /// How many Halton (2, 3) offsets `nextJitter` cycles through, 1...1,024.
        public var jitterSequenceLength: Int
        /// Maximum normalized depth difference accepted for motion history. The
        /// default covers the motion texture's half-float depth quantization.
        public var motionDepthThreshold: Float
        /// Opts motion-texture frames into velocity weighting, variance
        /// clipping and luminance weighting. Camera-only frames use the
        /// standard current-frame weight and clipping settings.
        public var traa: TRAASettings?

        public init(
            currentFrameWeight: Float = 0.1, clipsHistory: Bool = true, jitterSequenceLength: Int = 16,
            motionDepthThreshold: Float = 0.0005, traa: TRAASettings? = nil
        ) {
            self.currentFrameWeight = currentFrameWeight
            self.clipsHistory = clipsHistory
            self.jitterSequenceLength = jitterSequenceLength
            self.motionDepthThreshold = motionDepthThreshold
            self.traa = traa
        }

        /// The reference resolve with the r187 32-frame Halton jitter cycle.
        public static let referenceTRAA = Settings(jitterSequenceLength: 32, traa: .init())

        /// Progressive supersampling for a still camera and scene, like Three.js
        /// `TAARenderPass` with `accumulate`: an equal-weight mean of up to 1,024
        /// differently jittered frames in 32-bit float history, without clipping.
        /// Later frames keep a running average with weight 1/1,024. For a
        /// supersampled image within one frame, like `SSAARenderPass`, call
        /// `reset()` and then render and encode `N` jittered frames in a row.
        public static let accumulation = Settings(currentFrameWeight: 1.0 / 1024, clipsHistory: false, jitterSequenceLength: 1024)

        var isValid: Bool {
            currentFrameWeight.isFinite && currentFrameWeight > 0 && currentFrameWeight <= 1
                && (1...1024).contains(jitterSequenceLength) && (0...1).contains(motionDepthThreshold)
                && (traa?.isValid ?? true)
        }
    }

    public let device: any MTLDevice
    /// Validated by each encode.
    public var settings: Settings
    /// Frames encoded since creation or the last reset. Selects `nextJitter`.
    public private(set) var frameIndex = 0
    /// Frames blended into the current history; zero after `reset()`.
    public private(set) var accumulatedFrameCount = 0

    /// The offset in pixels, x right and y down, to give `SceneGraph.projectionJitter`
    /// for the next frame.
    public var nextJitter: SIMD2<Float> {
        Self.jitter(at: frameIndex % max(1, settings.jitterSequenceLength))
    }

    /// Resolves into RGBA16Float history (clipping) and RGBA32Float history (no clipping).
    private let pipeline: any MTLRenderPipelineState
    private let exactPipeline: any MTLRenderPipelineState
    private let library: any MTLLibrary
    private let vertex: any MTLFunction
    private var motionPipelines: [MTLPixelFormat: any MTLRenderPipelineState] = [:]
    private var history: [any MTLTexture] = []
    private var motionDepthHistory: [any MTLTexture] = []
    private var readIndex = 0
    private var previousClipFromWorld: simd_float4x4?
    private var previousJitter = SIMD2<Float>.zero
    private var previousMotionDepthMode: MeshDepthMode?
    private var previousUsesTRAA = false

    /// Matches `TAAOptions` in AntiAliasing.metal.
    private struct ShaderOptions {
        var worldFromJitteredClip: simd_float4x4
        var currentClipFromWorld: simd_float4x4
        var previousClipFromWorld: simd_float4x4
        var currentWeight: Float
        var flags: UInt32
        var jitterDelta: SIMD2<Float>
        var motionDepthThreshold: Float
        var maxVelocityLength: Float
        var edgeDepthDifference: Float
    }

    public init(device: any MTLDevice, settings: Settings = Settings()) throws {
        self.device = device
        self.settings = settings
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let vertex = library.makeFunction(name: "frostfield_aa_vertex"),
                  let fragment = library.makeFunction(name: "frostfield_taa_fragment") else {
                throw Failure.missingShader
            }
            self.library = library
            self.vertex = vertex
            pipeline = try Self.makePipeline(device: device, vertex: vertex, fragment: fragment, format: .rgba16Float)
            exactPipeline = try Self.makePipeline(device: device, vertex: vertex, fragment: fragment, format: .rgba32Float)
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
    }

    /// Discards the history and restarts the jitter sequence. The next encode
    /// writes its current frame unchanged.
    public func reset() {
        frameIndex = 0
        accumulatedFrameCount = 0
        previousClipFromWorld = nil
        previousJitter = .zero
        previousMotionDepthMode = nil
        previousUsesTRAA = false
    }

    /// Encodes one resolve on the caller's command buffer without submitting it,
    /// then advances the history.
    ///
    /// - Parameters:
    ///   - source: This frame's color, rendered with `jitter`: shader-readable,
    ///     tracked, single-sample 2D RGBA16Float of premultiplied linear color.
    ///   - depth: This frame's stored Depth32Float depth, rendered with the same
    ///     jitter: shader-readable, tracked, single-sample, the source's size.
    ///   - clipFromWorld: The unjittered world-to-clip matrix of this frame, such as
    ///     `SceneGraph.clipFromWorld(width:height:)`. Finite and invertible.
    ///   - jitter: The offset the frame was rendered with, in pixels, each
    ///     component in -1...1.
    ///   - depthMode: Whether near surfaces have smaller (forward) or larger
    ///     (reversed) depth, which chooses the nearest neighbor for reprojection.
    ///   - destination: A separate RGBA16Float render target of the same size that
    ///     receives the anti-aliased frame. It may be the next frame's source.
    ///   - commandBuffer: An uncommitted buffer that retains its references.
    ///
    /// Everything is validated before encoding. A size change reallocates the
    /// history and starts it over.
    public func encode(
        source: any MTLTexture,
        depth: any MTLTexture,
        clipFromWorld: simd_float4x4,
        jitter: SIMD2<Float>,
        depthMode: MeshDepthMode = .forward,
        destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try encode(
            source: source, depth: depth, clipFromWorld: clipFromWorld, jitter: jitter, depthMode: depthMode,
            motionTexture: nil, destination: destination, commandBuffer: commandBuffer
        )
    }

    /// Resolves with optional raster motion in same-size, shader-readable RGBA16Float.
    /// RG holds previous-minus-current motion in top-left pixels, including each
    /// frame's raster jitter. B holds the previous stored projection depth and A
    /// is one for valid history. Both frames must use the same depth mode.
    ///
    /// The first motion frame after a resize, reset or camera-only frame seeds
    /// the depth history. Invalid or occluded pixels keep their current color.
    public func encode(
        source: any MTLTexture,
        depth: any MTLTexture,
        clipFromWorld: simd_float4x4,
        jitter: SIMD2<Float>,
        depthMode: MeshDepthMode = .forward,
        motionTexture: (any MTLTexture)?,
        destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try depthMode.requireProjectionDepth()
        guard settings.isValid else { throw Failure.invalidSettings }
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard commandBuffer.device === device, source.device === device, depth.device === device,
              destination.device === device, motionTexture.map({ $0.device === device }) ?? true else {
            throw Failure.wrongDevice
        }
        guard PostProcessPass.isReadable(source, format: .rgba16Float), source.width > 0, source.height > 0 else {
            throw Failure.source
        }
        guard PostProcessPass.isReadable(depth, format: .depth32Float) else { throw Failure.depth }
        if let motionTexture, !PostProcessPass.isReadable(motionTexture, format: .rgba16Float) { throw Failure.motion }
        guard !TextureTransfer.sharesStorage(source, destination),
              !TextureTransfer.sharesStorage(depth, destination),
              motionTexture.map({ !TextureTransfer.sharesStorage($0, destination) }) ?? true,
              !history.contains(where: { $0 === destination }),
              destination.pixelFormat == .rgba16Float, destination.textureType == .type2D,
              destination.sampleCount == 1, destination.usage.contains(.renderTarget),
              destination.storageMode != .memoryless else { throw Failure.destination }
        guard depth.width == source.width, depth.height == source.height,
              destination.width == source.width, destination.height == source.height,
              motionTexture.map({ $0.width == source.width && $0.height == source.height }) ?? true else {
            throw Failure.sizeMismatch
        }
        guard Self.isValidJitter(jitter) else { throw Failure.jitter }
        let jittered = Self.jitteredClipFromWorld(
            clipFromWorld, jitter: jitter, width: Double(source.width), height: Double(source.height)
        )
        let determinant = simd_determinant(jittered)
        let worldFromJittered = simd_inverse(jittered)
        guard Self.isFinite(clipFromWorld), determinant.isFinite, determinant != 0,
              Self.isFinite(worldFromJittered) else { throw Failure.transform }

        let traa = motionTexture == nil ? nil : settings.traa
        let clipsHistory = settings.clipsHistory || traa != nil
        let historyFormat: MTLPixelFormat = clipsHistory ? .rgba16Float : .rgba32Float
        let renderPipeline = try motionTexture == nil
            ? (clipsHistory ? pipeline : exactPipeline)
            : motionPipeline(format: historyFormat)
        if motionTexture != nil { try prepareMotionDepthHistory(width: source.width, height: source.height) }
        let (read, write) = try historyTextures(width: source.width, height: source.height, format: historyFormat)
        let historyValid = accumulatedFrameCount > 0 && previousClipFromWorld != nil
            && (motionTexture == nil || (previousMotionDepthMode == depthMode && previousUsesTRAA == (traa != nil)))
        var flags: UInt32 = historyValid ? 1 : 0
        if clipsHistory { flags |= 2 } else { flags |= 8 }
        if depthMode == .reversed { flags |= 4 }
        if let traa {
            flags |= 16
            if traa.useSubpixelCorrection { flags |= 32 }
        }
        var options = ShaderOptions(
            worldFromJitteredClip: worldFromJittered,
            currentClipFromWorld: clipFromWorld,
            previousClipFromWorld: previousClipFromWorld ?? clipFromWorld,
            currentWeight: historyValid
                ? max(settings.currentFrameWeight, 1 / Float(accumulatedFrameCount + 1)) : 1,
            flags: flags, jitterDelta: previousJitter - jitter, motionDepthThreshold: settings.motionDepthThreshold,
            maxVelocityLength: traa?.maxVelocityLength ?? 1, edgeDepthDifference: traa?.edgeDepthDifference ?? 0
        )

        let pass = MTLRenderPassDescriptor()
        let targets = motionTexture == nil ? [write, destination] : [write, destination, motionDepthHistory[1 - readIndex]]
        for (index, texture) in targets.enumerated() {
            pass.colorAttachments[index].texture = texture
            pass.colorAttachments[index].loadAction = .dontCare
            pass.colorAttachments[index].storeAction = .store
        }
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.encoder }
        encoder.label = "Frostfield temporal anti-aliasing"
        encoder.setRenderPipelineState(renderPipeline)
        encoder.setFragmentTexture(source, index: 0)
        encoder.setFragmentTexture(depth, index: 1)
        encoder.setFragmentTexture(read, index: 2)
        if let motionTexture {
            encoder.setFragmentTexture(motionTexture, index: 3)
            encoder.setFragmentTexture(motionDepthHistory[readIndex], index: 4)
        }
        encoder.setFragmentBytes(&options, length: MemoryLayout<ShaderOptions>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()

        readIndex = 1 - readIndex
        previousClipFromWorld = clipFromWorld
        previousJitter = jitter
        previousMotionDepthMode = motionTexture == nil ? nil : depthMode
        previousUsesTRAA = traa != nil
        accumulatedFrameCount = historyValid ? min(accumulatedFrameCount + 1, 1 << 20) : 1
        frameIndex = frameIndex == Int.max ? 0 : frameIndex + 1
    }

    private static func makePipeline(
        device: any MTLDevice, vertex: any MTLFunction, fragment: any MTLFunction,
        format: MTLPixelFormat, writesMotionDepth: Bool = false
    ) throws -> any MTLRenderPipelineState {
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = writesMotionDepth ? "Frostfield motion-guided TAA" : "Frostfield temporal anti-aliasing"
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        descriptor.colorAttachments[0].pixelFormat = format
        descriptor.colorAttachments[1].pixelFormat = .rgba16Float
        if writesMotionDepth { descriptor.colorAttachments[2].pixelFormat = .r32Float }
        return try device.makeRenderPipelineState(descriptor: descriptor)
    }

    private func motionPipeline(format: MTLPixelFormat) throws -> any MTLRenderPipelineState {
        if let existing = motionPipelines[format] { return existing }
        guard let fragment = library.makeFunction(name: "frostfield_taa_motion_fragment") else { throw Failure.missingShader }
        do {
            let pipeline = try Self.makePipeline(
                device: device, vertex: vertex, fragment: fragment, format: format, writesMotionDepth: true
            )
            motionPipelines[format] = pipeline
            return pipeline
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
    }

    private func prepareMotionDepthHistory(width: Int, height: Int) throws {
        if let first = motionDepthHistory.first, first.width == width, first.height == height { return }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = [.renderTarget, .shaderRead]
        let textures = try (0..<2).map { index in
            guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.allocation }
            texture.label = "Frostfield TAA motion depth history \(index)"
            return texture
        }
        motionDepthHistory = textures
        previousMotionDepthMode = nil
    }

    // MARK: - Jitter

    /// The Halton (2, 3) point at `index + 1`, centered on the pixel: an offset in
    /// pixels with each component in -0.5..<0.5. Negative indices use 0.
    public static func jitter(at index: Int) -> SIMD2<Float> {
        let i = max(index, 0) + 1
        return SIMD2(halton(i, base: 2) - 0.5, halton(i, base: 3) - 0.5)
    }

    /// Shifts a world-to-clip matrix so the rendered image moves by `jitter` pixels
    /// (x right, y down) in a `width` × `height` viewport. The translation is
    /// applied in clip space after projection, so depth is unchanged.
    public static func jitteredClipFromWorld(
        _ clipFromWorld: simd_float4x4, jitter: SIMD2<Float>, width: Double, height: Double
    ) -> simd_float4x4 {
        var translation = matrix_identity_float4x4
        translation.columns.3 = SIMD4(
            Float(2 * Double(jitter.x) / width), Float(-2 * Double(jitter.y) / height), 0, 1
        )
        return translation * clipFromWorld
    }

    static func isValidJitter(_ jitter: SIMD2<Float>) -> Bool {
        jitter.x.isFinite && jitter.y.isFinite && abs(jitter.x) <= 1 && abs(jitter.y) <= 1
    }

    private static func halton(_ index: Int, base: Int) -> Float {
        var fraction: Float = 1
        var result: Float = 0
        var i = index
        while i > 0 {
            fraction /= Float(base)
            result += fraction * Float(i % base)
            i /= base
        }
        return result
    }

    private static func isFinite(_ matrix: simd_float4x4) -> Bool {
        let (x, y, z, w) = matrix.columns
        return [x, y, z, w].allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite }
    }

    /// The history pair, reallocated (which starts the history over) when the size
    /// or the format changes.
    private func historyTextures(
        width: Int, height: Int, format: MTLPixelFormat
    ) throws -> (read: any MTLTexture, write: any MTLTexture) {
        if let first = history.first, first.width != width || first.height != height || first.pixelFormat != format {
            history.removeAll()
            reset()
        }
        while history.count < 2 {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: width, height: height, mipmapped: false
            )
            descriptor.storageMode = .private
            descriptor.usage = [.renderTarget, .shaderRead]
            guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.allocation }
            texture.label = "Frostfield TAA history \(history.count)"
            history.append(texture)
            accumulatedFrameCount = 0
        }
        return (history[readIndex], history[1 - readIndex])
    }

    public enum Failure: LocalizedError, Equatable {
        case missingShader
        case pipeline(String)
        case invalidSettings
        case commandBuffer
        case wrongDevice
        case source
        case depth
        case motion
        case destination
        case sizeMismatch
        case jitter
        case transform
        case allocation
        case encoder

        public var errorDescription: String? {
            switch self {
            case .missingShader: "Frostfield's temporal anti-aliasing shaders are missing."
            case .pipeline(let detail): "Could not prepare temporal anti-aliasing. \(detail)"
            case .invalidSettings:
                "Temporal anti-aliasing needs a current-frame weight in (0, 1], a jitter sequence of 1 to 1,024 offsets, normalized depth thresholds and a positive finite TRAA velocity limit."
            case .commandBuffer: "Temporal anti-aliasing needs an uncommitted command buffer that retains its references."
            case .wrongDevice: "Temporal anti-aliasing textures and the command buffer must use its Metal device."
            case .source: "The TAA source must be a shader-readable, tracked, single-sample 2D RGBA16Float texture."
            case .depth: "TAA depth must be a shader-readable, tracked, single-sample 2D Depth32Float texture."
            case .motion: "TAA motion must be a shader-readable, tracked, single-sample 2D RGBA16Float texture."
            case .destination:
                "The TAA destination must be a separate single-sample 2D RGBA16Float render target, not a history texture."
            case .sizeMismatch: "The TAA source, depth, optional motion and destination must have the same dimensions."
            case .jitter: "Projection jitter must be finite, with each component between -1 and 1 pixel."
            case .transform: "The TAA view-projection matrix must be finite and invertible."
            case .allocation: "Metal could not allocate the temporal anti-aliasing history."
            case .encoder: "Metal could not create the temporal anti-aliasing encoder."
            }
        }
    }
}

public extension SceneGraph {
    /// The unjittered world-to-clip matrix that `encode(pass:commandBuffer:)` uses
    /// for a color attachment of this size, for `TemporalAntiAliasing`.
    func clipFromWorld(width: Int, height: Int) throws -> simd_float4x4 {
        guard width > 0, height > 0 else { throw PerspectiveCamera.ValidationError.invalidAspectRatio }
        let worldFromCamera = cameraTransform.matrix
        let determinant = simd_determinant(worldFromCamera)
        guard determinant.isFinite, determinant != 0 else { throw TemporalAntiAliasing.Failure.transform }
        return try camera.projection(aspectRatio: Float(width) / Float(height)) * simd_inverse(worldFromCamera)
    }
}
