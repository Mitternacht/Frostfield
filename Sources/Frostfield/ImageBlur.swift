import Foundation
import Metal

/// A separable Gaussian blur of premultiplied linear RGBA, like the Three.js
/// `GaussianBlurShader` or `HorizontalBlurShader`/`VerticalBlurShader` pair.
///
/// Each axis convolves with `kernel(sigma:)`: the Gaussian integrated over every
/// pixel's footprint, truncated at three standard deviations and normalized to
/// sum to one. Taps read exact texels and clamp to the frame edge. The horizontal
/// pass writes an RGBA16Float scratch texture; the vertical pass writes the destination.
public final class GaussianBlur: PostProcessEffect {
    /// The widest supported standard deviation, in pixels.
    public static let maximumSigma: Float = 64

    public struct Settings: Sendable, Hashable {
        /// Standard deviation in output pixels, from 0 to 64. Zero copies the source.
        public var sigma: Float

        public init(sigma: Float = 2) {
            self.sigma = sigma
        }
    }

    public let device: any MTLDevice
    public var readsDepth: Bool { false }
    public var settings: Settings
    private let shaders: PostProcessShaders
    private var scratch: (any MTLTexture)?

    public init(device: any MTLDevice, settings: Settings = Settings()) throws {
        self.device = device
        self.settings = settings
        shaders = try PostProcessShaders(device: device)
    }

    /// The normalized one-dimensional weights for offsets `-radius...radius`, where
    /// `radius = ceil(3σ)`. Weight `i` is the probability mass of a Gaussian with
    /// standard deviation σ over pixel `[i − ½, i + ½]`, renormalized over the
    /// truncated support. Sigma zero returns `[1]`.
    public static func kernel(sigma: Float) throws -> [Float] {
        guard sigma.isFinite, (0...maximumSigma).contains(sigma) else {
            throw PostProcessEffectError.invalidSettings("Gaussian sigma must be from 0 to \(Int(maximumSigma)) pixels.")
        }
        guard sigma > 0 else { return [1] }
        let radius = max(1, Int((3 * Double(sigma)).rounded(.up)))
        let scale = 1 / (Double(sigma) * 2.0.squareRoot())
        let mass = (-radius...radius).map { offset -> Double in
            0.5 * (erf((Double(offset) + 0.5) * scale) - erf((Double(offset) - 0.5) * scale))
        }
        let total = mass.reduce(0, +)
        return mass.map { Float($0 / total) }
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render("frostfield_blur_fragment", formats: [pixelFormat])
        _ = try shaders.render("frostfield_copy_fragment", formats: [pixelFormat])
        _ = try shaders.compute("frostfield_blur_kernel")
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        _ = try Self.kernel(sigma: settings.sigma)
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination, commandBuffer: commandBuffer,
            requiresDepth: false
        )
        if settings.sigma > 0 { _ = try scratchTexture(width: source.width, height: source.height) }
    }

    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)? = nil, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        let weights = try Self.kernel(sigma: settings.sigma)
        guard weights.count > 1 else {
            let copy = try shaders.render("frostfield_copy_fragment", formats: [destination.pixelFormat])
            try PostProcessShaders.drawFullScreen(
                into: destination, label: "Frostfield blur copy", commandBuffer: commandBuffer, pipeline: copy
            ) { $0.setFragmentTexture(source, index: 0) }
            return
        }
        try SeparableBlur.encode(
            shaders: shaders, source: source, depth: nil, scratch: try scratchTexture(width: source.width, height: source.height),
            destination: destination, weights: weights, parameters: { axis in
                SeparableBlur.Parameters(radius: Int32(weights.count / 2), axis: axis, depthSigma: 0, colorSigma: 0, depth: .zero)
            }, label: "Frostfield Gaussian blur", commandBuffer: commandBuffer
        )
    }

    private func scratchTexture(width: Int, height: Int) throws -> any MTLTexture {
        if let scratch, scratch.width == width, scratch.height == height { return scratch }
        let texture = try PostProcessShaders.scratch(device: device, width: width, height: height, label: "Frostfield blur scratch")
        scratch = texture
        return texture
    }
}

/// An edge-preserving separable blur weighted by view-depth and color similarity.
///
/// Each tap's Gaussian spatial weight is multiplied by
/// `exp(−½ ((z − z₀) / (depthTolerance · z₀))²)`, where `z` is view-axis distance
/// reconstructed from stored depth with `camera` and `depthMode`, and, when
/// `colorSigma` is positive, by `exp(−½ (|c − c₀| / colorSigma)²)` over premultiplied
/// RGBA. Weights are renormalized per pixel, so a surface never averages with a
/// layer at a different distance. Applying the two axes in sequence approximates the
/// full 2D bilateral filter, as is common for denoising and SSAO/SSR resolves.
public final class BilateralBlur: PostProcessEffect {
    public struct Settings: Sendable, Hashable {
        /// Spatial standard deviation in pixels, from 0 to 32.
        public var sigma: Float
        /// Depth standard deviation as a fraction of the center pixel's view distance.
        /// Zero ignores depth.
        public var depthTolerance: Float
        /// Color standard deviation over premultiplied RGBA. Zero ignores color.
        public var colorSigma: Float

        public init(sigma: Float = 3, depthTolerance: Float = 0.02, colorSigma: Float = 0) {
            self.sigma = sigma
            self.depthTolerance = depthTolerance
            self.colorSigma = colorSigma
        }
    }

    public let device: any MTLDevice
    /// True while `depthTolerance` is positive.
    public var readsDepth: Bool { settings.depthTolerance > 0 }
    public var settings: Settings
    /// The camera whose clip planes stored depth uses.
    public var camera: PerspectiveCamera
    public var depthMode: MeshDepthMode
    private let shaders: PostProcessShaders
    private var scratch: (any MTLTexture)?

    public init(
        device: any MTLDevice, camera: PerspectiveCamera, depthMode: MeshDepthMode = .forward,
        settings: Settings = Settings()
    ) throws {
        self.device = device
        self.camera = camera
        self.depthMode = depthMode
        self.settings = settings
        shaders = try PostProcessShaders(device: device)
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render("frostfield_blur_fragment", formats: [pixelFormat])
        _ = try shaders.compute("frostfield_blur_kernel")
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard settings.sigma.isFinite, (0...32).contains(settings.sigma), settings.sigma > 0 else {
            throw PostProcessEffectError.invalidSettings("Bilateral sigma must be greater than 0 and at most 32 pixels.")
        }
        guard settings.depthTolerance.isFinite, settings.depthTolerance >= 0,
              settings.colorSigma.isFinite, settings.colorSigma >= 0 else {
            throw PostProcessEffectError.invalidSettings("Bilateral depth tolerance and color sigma must be finite and nonnegative.")
        }
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination, commandBuffer: commandBuffer,
            requiresDepth: readsDepth
        )
        if scratch?.width != source.width || scratch?.height != source.height {
            scratch = try PostProcessShaders.scratch(
                device: device, width: source.width, height: source.height, label: "Frostfield bilateral scratch"
            )
        }
    }

    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        guard let scratch else { throw PostProcessEffectError.allocation }
        let weights = try GaussianBlur.kernel(sigma: settings.sigma)
        let usesDepth = readsDepth
        if usesDepth { try depthMode.requireProjectionDepth() }
        let depthParameters = PostProcessValidation.depthParameters(camera: camera, depthMode: depthMode)
        try SeparableBlur.encode(
            shaders: shaders, source: source, depth: usesDepth ? depth : nil, scratch: scratch, destination: destination,
            weights: weights, parameters: { axis in
                SeparableBlur.Parameters(
                    radius: Int32(weights.count / 2), axis: axis, depthSigma: usesDepth ? self.settings.depthTolerance : 0,
                    colorSigma: self.settings.colorSigma, depth: depthParameters
                )
            }, label: "Frostfield bilateral blur", commandBuffer: commandBuffer
        )
    }
}

/// Shared encoding for the separable Gaussian and bilateral passes.
enum SeparableBlur {
    struct Parameters {
        var radius: Int32
        var axis: Int32
        var depthSigma: Float
        var colorSigma: Float
        var depth: SIMD4<Float>
    }

    static func encode(
        shaders: PostProcessShaders, source: any MTLTexture, depth: (any MTLTexture)?, scratch: any MTLTexture,
        destination: any MTLTexture, weights: [Float], parameters: (Int32) -> Parameters, label: String,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        let kernel = try shaders.compute("frostfield_blur_kernel")
        let fragment = try shaders.render("frostfield_blur_fragment", formats: [destination.pixelFormat])
        let depthTexture = try depth ?? shaders.placeholderDepth()
        var useDepth: UInt32 = depth == nil ? 0 : 1
        var horizontal = parameters(0)
        var vertical = parameters(1)
        try weights.withUnsafeBytes { bytes in
            try PostProcessShaders.compute(
                label: "\(label) horizontal", commandBuffer: commandBuffer, pipeline: kernel,
                width: scratch.width, height: scratch.height
            ) { encoder in
                encoder.setTexture(source, index: 0)
                encoder.setTexture(scratch, index: 1)
                encoder.setTexture(depthTexture, index: 2)
                encoder.setBytes(&horizontal, length: MemoryLayout<Parameters>.stride, index: 0)
                encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 1)
                encoder.setBytes(&useDepth, length: MemoryLayout<UInt32>.size, index: 2)
            }
            try PostProcessShaders.drawFullScreen(
                into: destination, label: "\(label) vertical", commandBuffer: commandBuffer, pipeline: fragment
            ) { encoder in
                encoder.setFragmentTexture(scratch, index: 0)
                encoder.setFragmentTexture(depthTexture, index: 2)
                encoder.setFragmentBytes(&vertical, length: MemoryLayout<Parameters>.stride, index: 0)
                encoder.setFragmentBytes(bytes.baseAddress!, length: bytes.count, index: 1)
                encoder.setFragmentBytes(&useDepth, length: MemoryLayout<UInt32>.size, index: 2)
            }
        }
    }
}

/// An iterated four-tap Kawase blur, like pmndrs/postprocessing `KawaseBlurPass`.
///
/// When `resolutionScale` is below 1, the frame is first reduced to the blur grid by
/// averaging every grid texel's exact footprint, which keeps the mean and never skips
/// a source texel. Iteration `i` then averages four bilinear samples at
/// `±(offsets[i] + ½)` blur-grid texels diagonally from each pixel; each sample lands
/// on a texel corner and averages four texels, so one iteration is a 16-texel average
/// that conserves energy away from the frame edge. Iterations run in RGBA16Float
/// scratch textures and the last one writes the destination at full size.
public final class KawaseBlur: PostProcessEffect {
    public enum KernelSize: Sendable, CaseIterable {
        case verySmall, small, medium, large, veryLarge, huge

        /// The per-iteration offsets for this preset.
        public var offsets: [Float] {
            switch self {
            case .verySmall: [0, 1, 2, 2, 3]
            case .small: [0, 1, 2, 3, 4, 4, 5]
            case .medium: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]
            case .large: [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 14, 15]
            case .veryLarge: [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 17, 19, 21]
            case .huge: [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]
            }
        }
    }

    public struct Settings: Sendable, Hashable {
        /// One to 32 offsets, each from 0 to 64 blur-grid texels.
        public var offsets: [Float]
        /// The blur grid's size relative to the frame, from 0.125 to 1.
        public var resolutionScale: Float

        public init(offsets: [Float] = KernelSize.medium.offsets, resolutionScale: Float = 0.5) {
            self.offsets = offsets
            self.resolutionScale = resolutionScale
        }

        public init(kernelSize: KernelSize, resolutionScale: Float = 0.5) {
            self.init(offsets: kernelSize.offsets, resolutionScale: resolutionScale)
        }
    }

    public let device: any MTLDevice
    public var readsDepth: Bool { false }
    public var settings: Settings
    private let shaders: PostProcessShaders
    private var scratch: [any MTLTexture] = []

    public init(device: any MTLDevice, settings: Settings = Settings()) throws {
        self.device = device
        self.settings = settings
        shaders = try PostProcessShaders(device: device)
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try prepare(pixelFormat: pixelFormat, tiltShift: false)
    }

    func prepare(pixelFormat: MTLPixelFormat, tiltShift: Bool) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render(tiltShift ? "frostfield_tilt_shift_fragment" : "frostfield_kawase_fragment", formats: [pixelFormat])
        _ = try shaders.compute(tiltShift ? "frostfield_tilt_shift_kernel" : "frostfield_kawase_kernel")
        _ = try shaders.compute("frostfield_area_downsample")
    }

    /// The blur grid for a frame size.
    func gridSize(width: Int, height: Int) -> (Int, Int) {
        (max(1, Int((Float(width) * settings.resolutionScale).rounded())),
         max(1, Int((Float(height) * settings.resolutionScale).rounded())))
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard (1...32).contains(settings.offsets.count),
              settings.offsets.allSatisfy({ $0.isFinite && (0...64).contains($0) }),
              settings.resolutionScale.isFinite, (0.125...1).contains(settings.resolutionScale) else {
            throw PostProcessEffectError.invalidSettings(
                "Kawase blur needs 1 to 32 offsets from 0 to 64 and a resolution scale from 0.125 to 1."
            )
        }
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination, commandBuffer: commandBuffer,
            requiresDepth: false
        )
        let (width, height) = gridSize(width: source.width, height: source.height)
        if scratch.first?.width != width || scratch.first?.height != height {
            scratch = []
        }
        // A reduced grid holds the downsampled frame in the second scratch texture.
        let reduced = width != source.width || height != source.height
        let needed = reduced ? 2 : min(2, settings.offsets.count - 1)
        while scratch.count < needed {
            scratch.append(try PostProcessShaders.scratch(
                device: device, width: width, height: height, label: "Frostfield Kawase scratch \(scratch.count)"
            ))
        }
    }

    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)? = nil, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try encode(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer, focus: nil)
    }

    func encode(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer, focus: FocusBand?
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        let kernel = try shaders.compute(focus == nil ? "frostfield_kawase_kernel" : "frostfield_tilt_shift_kernel")
        let fragment = try shaders.render(
            focus == nil ? "frostfield_kawase_fragment" : "frostfield_tilt_shift_fragment", formats: [destination.pixelFormat]
        )
        let aspect = Float(source.width) / Float(source.height)
        var band = focus.map { SIMD4($0.normal.x * aspect, $0.normal.y, $0.offset, $0.sharpHalfWidth) } ?? .zero
        let fullBlur = focus?.fullBlurHalfWidth ?? 0
        let (width, height) = gridSize(width: source.width, height: source.height)
        let texel = SIMD2(1 / Float(width), 1 / Float(height))
        var input = source
        if width != source.width || height != source.height {
            let downsample = try shaders.compute("frostfield_area_downsample")
            let reduced = scratch[1]
            try PostProcessShaders.compute(
                label: "Frostfield Kawase downsample", commandBuffer: commandBuffer, pipeline: downsample,
                width: width, height: height
            ) { encoder in
                encoder.setTexture(source, index: 0)
                encoder.setTexture(reduced, index: 1)
            }
            input = reduced
        }
        for (index, offset) in settings.offsets.enumerated() {
            if index == settings.offsets.count - 1 {
                var parameters = KawaseParameters(
                    taps: SIMD4(offset, texel.x, texel.y, fullBlur),
                    output: SIMD4(1 / Float(destination.width), 1 / Float(destination.height), 0, 0)
                )
                try PostProcessShaders.drawFullScreen(
                    into: destination, label: "Frostfield Kawase blur", commandBuffer: commandBuffer, pipeline: fragment
                ) { encoder in
                    encoder.setFragmentTexture(input, index: 0)
                    encoder.setFragmentBytes(&parameters, length: MemoryLayout<KawaseParameters>.stride, index: 0)
                    if focus != nil {
                        encoder.setFragmentTexture(source, index: 1)
                        encoder.setFragmentBytes(&band, length: MemoryLayout<SIMD4<Float>>.stride, index: 1)
                    }
                }
            } else {
                let output = scratch[index % 2]
                var parameters = KawaseParameters(
                    taps: SIMD4(offset, texel.x, texel.y, fullBlur), output: SIMD4(texel.x, texel.y, 0, 0)
                )
                try PostProcessShaders.compute(
                    label: "Frostfield Kawase iteration \(index)", commandBuffer: commandBuffer, pipeline: kernel,
                    width: width, height: height
                ) { encoder in
                    encoder.setTexture(input, index: 0)
                    encoder.setTexture(output, index: 1)
                    encoder.setBytes(&parameters, length: MemoryLayout<KawaseParameters>.stride, index: 0)
                    if focus != nil { encoder.setBytes(&band, length: MemoryLayout<SIMD4<Float>>.stride, index: 1) }
                }
                input = output
            }
        }
    }

    struct FocusBand {
        let normal: SIMD2<Float>
        let offset: Float
        let sharpHalfWidth: Float
        let fullBlurHalfWidth: Float
    }

    private struct KawaseParameters {
        var taps: SIMD4<Float>
        var output: SIMD4<Float>
    }
}

extension GaussianBlur: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}

extension BilateralBlur: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}

extension KawaseBlur: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}
