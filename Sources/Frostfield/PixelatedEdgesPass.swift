import Metal
import simd

/// Nearest-center pixel blocks with optional depth outlines and normal crease
/// highlights. Use directly or as a `PostProcessChain.Effect.pass(_:)`.
public final class PixelatedEdgesPass: PostProcessEffect {
    public struct Settings: Sendable, Hashable {
        /// Width and height of a block in source pixels, from 1 to 1024.
        public var pixelSize: Int
        /// Brightens the preferred side of a normal crease, from zero to one.
        public var normalEdgeStrength: Float
        /// Darkens the shallower side of a depth discontinuity, from zero to one.
        public var depthEdgeStrength: Float
        /// The stored projection depth's ordering. Edge comparisons support forward and reversed depth.
        public var depthMode: MeshDepthMode

        public init(
            pixelSize: Int = 6, normalEdgeStrength: Float = 0.3,
            depthEdgeStrength: Float = 0.4, depthMode: MeshDepthMode = .forward
        ) {
            self.pixelSize = pixelSize
            self.normalEdgeStrength = normalEdgeStrength
            self.depthEdgeStrength = depthEdgeStrength
            self.depthMode = depthMode
        }

        func check() throws {
            guard (1...1024).contains(pixelSize), (0...1).contains(normalEdgeStrength),
                  (0...1).contains(depthEdgeStrength) else {
                throw PostProcessEffectError.invalidSettings("Pixelation needs a block size from 1 to 1024 and edge strengths from 0 to 1.")
            }
            if normalEdgeStrength > 0 || depthEdgeStrength > 0 { try depthMode.requireProjectionDepth() }
        }
    }

    public let device: any MTLDevice
    public var settings: Settings
    public var readsDepth: Bool { settings.depthEdgeStrength > 0 || settings.normalEdgeStrength > 0 }
    /// Set to the RG32Uint output of `SceneGraph.encodeSurfaceAttributes` for the
    /// current view when normal highlights are enabled. Use the prepass's own depth.
    public var surfaceAttributes: (any MTLTexture)?
    /// The current camera's world orientation, used to place the crease highlight.
    public var cameraRotation: simd_quatf

    private let shaders: PostProcessShaders

    private enum Variant: String, CaseIterable {
        case color = "frostfield_pixelated_color"
        case depth = "frostfield_pixelated_depth"
        case normals = "frostfield_pixelated_normals"
    }

    private struct Parameters {
        var worldBias: SIMD4<Float>
        var strengths: SIMD2<Float>
        var pixelSize: UInt32
        var reversed: UInt32
    }

    public init(
        device: any MTLDevice, settings: Settings = .init(), surfaceAttributes: (any MTLTexture)? = nil,
        cameraRotation: simd_quatf = .init(real: 1, imag: .zero)
    ) throws {
        try settings.check()
        self.device = device
        self.settings = settings
        self.surfaceAttributes = surfaceAttributes
        self.cameraRotation = cameraRotation
        shaders = try PostProcessShaders(device: device)
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        for variant in Variant.allCases {
            _ = try shaders.render(variant.rawValue, formats: [pixelFormat])
        }
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        _ = try parameters(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
    }

    /// Samples each block's center and compares its four adjacent blocks.
    /// Pass matching single-sample scene color, stored Depth32Float and, for
    /// normal edges, the packed attribute texture. Gain applies to linear HDR
    /// RGB and preserves the selected center texel's coverage.
    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)? = nil, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try encode(source: source, depth: depth, normalDepth: nil, destination: destination, commandBuffer: commandBuffer)
    }

    func encode(
        source: any MTLTexture, depth: (any MTLTexture)?, normalDepth: (any MTLTexture)?,
        destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        let (variant, constants) = try parameters(
            source: source, depth: depth, destination: destination, commandBuffer: commandBuffer
        )
        if let normalDepth {
            guard normalDepth.device === device else { throw PostProcessEffectError.wrongDevice }
            guard PostProcessPass.isReadable(normalDepth, format: .depth32Float) else { throw PostProcessEffectError.depth }
            guard normalDepth.width == source.width, normalDepth.height == source.height else { throw PostProcessEffectError.sizeMismatch }
            guard !TextureTransfer.sharesStorage(normalDepth, destination) else { throw PostProcessEffectError.destination }
        }
        var values = constants
        let pipeline = try shaders.render(variant.rawValue, formats: [destination.pixelFormat])
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield pixelation", commandBuffer: commandBuffer, pipeline: pipeline
        ) { encoder in
            encoder.setFragmentTexture(source, index: 0)
            if variant != .color { encoder.setFragmentTexture(depth, index: 1) }
            if variant == .normals {
                encoder.setFragmentTexture(surfaceAttributes, index: 2)
                encoder.setFragmentTexture(normalDepth ?? depth, index: 3)
            }
            encoder.setFragmentBytes(&values, length: MemoryLayout<Parameters>.stride, index: 0)
        }
    }

    private func parameters(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws -> (Variant, Parameters) {
        try settings.check()
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination,
            commandBuffer: commandBuffer, requiresDepth: readsDepth
        )
        guard !TextureTransfer.sharesStorage(source, destination),
              depth.map({ !TextureTransfer.sharesStorage($0, destination) }) ?? true else {
            throw PostProcessEffectError.destination
        }
        let variant: Variant
        var bias = SIMD3<Float>.zero
        if settings.normalEdgeStrength > 0 {
            guard let surfaceAttributes else {
                throw PostProcessEffectError.unsupportedInput("Pixelation normal edges need the scene's RG32Uint surface attributes.")
            }
            guard surfaceAttributes.device === device else { throw PostProcessEffectError.wrongDevice }
            guard PostProcessPass.isReadable(surfaceAttributes, format: SceneGraph.surfaceAttributesPixelFormat) else {
                throw PostProcessEffectError.unsupportedInput("Pixelation needs shader-readable, tracked, single-sample 2D RG32Uint surface attributes.")
            }
            guard surfaceAttributes.width == source.width, surfaceAttributes.height == source.height else {
                throw PostProcessEffectError.sizeMismatch
            }
            guard !TextureTransfer.sharesStorage(surfaceAttributes, destination) else { throw PostProcessEffectError.destination }
            guard simd_length_squared(cameraRotation.vector).isNormal else {
                throw PostProcessEffectError.invalidSettings("Pixelation needs a finite nonzero camera rotation for normal edges.")
            }
            bias = simd_act(cameraRotation.normalized, SIMD3(repeating: 1))
            variant = .normals
        } else {
            variant = readsDepth ? .depth : .color
        }
        return (variant, Parameters(
            worldBias: SIMD4(bias, 0), strengths: SIMD2(settings.depthEdgeStrength, settings.normalEdgeStrength),
            pixelSize: UInt32(settings.pixelSize), reversed: settings.depthMode == .reversed ? 1 : 0
        ))
    }
}

extension PixelatedEdgesPass: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}
