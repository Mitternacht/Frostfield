import Foundation
import Metal

/// A screen-space focus band with aspect-correct tilt and feathered blur.
/// Use it as a `PostProcessChain.Effect.pass` or encode it directly.
public final class TiltShift: PostProcessEffect {
    public struct Settings: Sendable, Hashable {
        /// Counterclockwise rotation of the sharp band, in radians.
        public var rotation: Float
        /// Signed displacement along the band normal. At zero rotation positive values move it down.
        /// One unit is half the output height, regardless of aspect ratio.
        public var offset: Float
        /// Distance from the centerline at which blur reaches its full strength, in half-frame-height units.
        public var focusArea: Float
        /// Width of the smooth transition on either edge. The sharp half-width is
        /// `max(focusArea - feather, 0)`. Zero gives a hard transition.
        public var feather: Float
        /// The Kawase sample offsets and resolution scale used outside the band.
        public var blur: KawaseBlur.Settings

        public init(
            rotation: Float = 0, offset: Float = 0, focusArea: Float = 0.4, feather: Float = 0.3,
            blur: KawaseBlur.Settings = .init()
        ) {
            self.rotation = rotation
            self.offset = offset
            self.focusArea = focusArea
            self.feather = feather
            self.blur = blur
        }
    }

    public var settings: Settings
    public var device: any MTLDevice { blur.device }
    public var readsDepth: Bool { false }
    private let blur: KawaseBlur

    public init(device: any MTLDevice, settings: Settings = .init()) throws {
        self.settings = settings
        blur = try KawaseBlur(device: device, settings: settings.blur)
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try blur.prepare(pixelFormat: pixelFormat, tiltShift: true)
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        _ = try focusBand(source: source, destination: destination)
        try blur.validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
    }

    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)? = nil, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        let focus = try focusBand(source: source, destination: destination)
        try blur.encode(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer, focus: focus)
    }

    private func focusBand(source: any MTLTexture, destination: any MTLTexture) throws -> KawaseBlur.FocusBand {
        let settings = settings
        guard settings.rotation.isFinite, settings.offset.isFinite,
              (0...2).contains(settings.focusArea), (0...2).contains(settings.feather) else {
            throw PostProcessEffectError.invalidSettings(
                "Tilt shift needs finite rotation and offset, with focus area and feather from 0 to 2."
            )
        }
        guard !TextureTransfer.sharesStorage(source, destination) else { throw PostProcessEffectError.destination }
        blur.settings = settings.blur
        return KawaseBlur.FocusBand(
            normal: SIMD2(sin(settings.rotation), cos(settings.rotation)),
            offset: settings.offset, sharpHalfWidth: max(settings.focusArea - settings.feather, 0),
            fullBlurHalfWidth: settings.focusArea
        )
    }
}

extension TiltShift: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { blur.builtPipelineCount }
}
