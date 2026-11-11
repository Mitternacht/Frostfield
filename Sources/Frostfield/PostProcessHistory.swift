import Foundation
import Metal

/// A previous-frame texture for post-process graphs, and a chain stage that captures it.
///
/// Bind `texture` to a `PostProcessGraph` texture declaration (for example the
/// "Previous" texture of `PostProcessGraph.motionTrail`) and place this object as a
/// `PostProcessChain.Effect.pass` stage after the graph. Each encode copies its input
/// to the destination and into `texture`, so the next frame's graph reads this
/// frame's result. Metal orders the copy after the earlier read on one queue.
///
/// The texture starts transparent black, cleared on the queue given at creation;
/// submit later frames on that queue. Its size is fixed: create a new history and
/// re-prepare the graph stage when the frame size changes.
public final class PostProcessHistory: PostProcessEffect {
    public let device: any MTLDevice
    public var readsDepth: Bool { false }
    /// The last captured frame, as private, tracked RGBA16Float.
    public let texture: any MTLTexture
    /// Frames captured since creation or `reset`.
    public private(set) var capturedFrameCount = 0
    private let shaders: PostProcessShaders
    private let queue: any MTLCommandQueue

    public init(queue: any MTLCommandQueue, width: Int, height: Int) throws {
        guard width > 0, height > 0, width <= 16_384, height <= 16_384 else {
            throw PostProcessEffectError.invalidSettings("History textures need 1 to 16,384 pixels per side.")
        }
        self.queue = queue
        device = queue.device
        shaders = try PostProcessShaders(device: device)
        texture = try PostProcessShaders.scratch(
            device: device, width: width, height: height, usage: [.shaderRead, .renderTarget],
            label: "Frostfield post-process history"
        )
        try reset()
    }

    /// Clears the history to transparent black on the creation queue.
    public func reset() throws {
        guard let buffer = queue.makeCommandBuffer() else { throw PostProcessEffectError.encoder }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = buffer.makeRenderCommandEncoder(descriptor: pass) else { throw PostProcessEffectError.encoder }
        encoder.label = "Frostfield history clear"
        encoder.endEncoding()
        buffer.commit()
        capturedFrameCount = 0
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render("frostfield_copy_fragment", formats: [pixelFormat])
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination, commandBuffer: commandBuffer,
            requiresDepth: false
        )
        guard source.width == texture.width, source.height == texture.height else {
            throw PostProcessEffectError.sizeMismatch
        }
    }

    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)? = nil, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        let copy = try shaders.render("frostfield_copy_fragment", formats: [destination.pixelFormat])
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield history pass-through", commandBuffer: commandBuffer, pipeline: copy
        ) { $0.setFragmentTexture(source, index: 0) }
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw PostProcessEffectError.encoder }
        blit.label = "Frostfield history capture"
        blit.copy(from: source, to: texture)
        blit.endEncoding()
        capturedFrameCount += 1
    }
}

/// Persistence of previous frames, like Three.js `AfterimagePass`.
///
/// `.maximum(damp:)` reproduces `AfterimageShader` exactly: each component of the
/// previous output is multiplied by `damp` where it exceeds 0.1 and zeroed
/// elsewhere, and the result is the componentwise maximum with the new frame.
/// `.blend(persistence:)` is an exponential moving average,
/// `mix(new, previous, persistence)`, for linear motion trails. The first frame and
/// the first frame after a size change or `reset()` pass through. The effect owns two
/// RGBA16Float history textures and writes the next one together with the destination.
public final class Afterimage: PostProcessEffect {
    public enum Mode: Sendable, Hashable {
        case maximum(damp: Float)
        case blend(persistence: Float)
    }

    public let device: any MTLDevice
    public var readsDepth: Bool { false }
    public var mode: Mode
    private let shaders: PostProcessShaders
    private var history: [any MTLTexture] = []
    private var current = 0
    private var isValid = false

    public init(device: any MTLDevice, mode: Mode = .maximum(damp: 0.96)) throws {
        self.device = device
        self.mode = mode
        shaders = try PostProcessShaders(device: device)
    }

    /// Forgets the stored frame, so the next encode passes through.
    public func reset() {
        isValid = false
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render("frostfield_afterimage_fragment", formats: [pixelFormat, .rgba16Float])
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        let factor: Float = switch mode {
        case .maximum(let damp): damp
        case .blend(let persistence): persistence
        }
        guard factor.isFinite, (0...1).contains(factor) else {
            throw PostProcessEffectError.invalidSettings("Afterimage damp and persistence must be from 0 to 1.")
        }
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination, commandBuffer: commandBuffer,
            requiresDepth: false
        )
        if history.first?.width != source.width || history.first?.height != source.height {
            history = try (0..<2).map { index in
                try PostProcessShaders.scratch(
                    device: device, width: source.width, height: source.height, usage: [.shaderRead, .renderTarget],
                    label: "Frostfield afterimage history \(index)"
                )
            }
            isValid = false
        }
    }

    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)? = nil, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        let pipeline = try shaders.render("frostfield_afterimage_fragment", formats: [destination.pixelFormat, .rgba16Float])
        let previous = history[current]
        let next = history[1 - current]
        var parameters: SIMD4<Float> = switch mode {
        case .maximum(let damp): SIMD4(damp, 0, isValid ? 1 : 0, 0)
        case .blend(let persistence): SIMD4(persistence, 1, isValid ? 1 : 0, 0)
        }
        try PostProcessShaders.drawFullScreen(
            into: destination, extra: [next], label: "Frostfield afterimage", commandBuffer: commandBuffer,
            pipeline: pipeline
        ) { encoder in
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentTexture(previous, index: 1)
            encoder.setFragmentBytes(&parameters, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
        }
        current = 1 - current
        isValid = true
    }
}

extension PostProcessHistory: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}

extension Afterimage: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}
