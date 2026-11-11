import Foundation
import Metal

/// Scatter-as-gather bokeh depth of field driven by a physical `CameraLens`, like
/// Three.js `BokehPass` and pmndrs/postprocessing `DepthOfFieldEffect`.
///
/// Each pixel's signed circle-of-confusion radius is the thin-lens value
/// `K · (1 − s / z)` in pixels, from view distance `z` reconstructed from stored
/// depth with `camera` and `depthMode`, focus distance `s`, and `K` from the lens's
/// aperture, focal length and sensor width; it clamps to `maximumRadius`.
/// A 16-pixel tile pass records the largest radius that can reach each tile.
///
/// The gather visits a Vogel disk of `sampleCount` taps within that radius. A tap
/// contributes when its own blur disk, shaped by the aperture blades, covers the
/// center (scatter as gather), weighted by its represented area over its disk area,
/// so light spreads as it would through the lens. Taps nearer than the center
/// form a foreground layer whose estimated coverage composites over the rest, so
/// defocused foreground spreads across in-focus edges. Taps behind the center
/// spread with the center's own radius, so blurred background never bleeds onto a
/// sharp subject, and the scene behind a defocused foreground shows through its
/// edge, which fades symmetrically across the silhouette. Pixels whose tile cannot
/// be reached by any blur copy through.
public final class BokehDepthOfField: PostProcessEffect {
    public static let tileSize = 16
    public static let maximumRadiusLimit: Float = 48

    public struct Settings: Sendable, Hashable {
        /// Largest circle-of-confusion radius in pixels, from 1 to 48.
        public var maximumRadius: Float
        /// Gather taps per blurred pixel, from 16 to 512.
        public var sampleCount: Int
        /// Aperture blades: 0 for a circular aperture, else 3 to 16 for a polygonal bokeh.
        public var bladeCount: Int
        /// Rotation of the polygonal aperture in radians.
        public var bladeRotation: Float
        /// Rotates the tap pattern per pixel with interleaved gradient noise, trading banding for noise.
        public var dither: Bool

        public init(
            maximumRadius: Float = 16, sampleCount: Int = 96, bladeCount: Int = 0, bladeRotation: Float = 0,
            dither: Bool = true
        ) {
            self.maximumRadius = maximumRadius
            self.sampleCount = sampleCount
            self.bladeCount = bladeCount
            self.bladeRotation = bladeRotation
            self.dither = dither
        }
    }

    public let device: any MTLDevice
    public var readsDepth: Bool { true }
    public var settings: Settings
    public var lens: CameraLens
    /// The camera whose clip planes the stored depth uses. Its field of view does
    /// not affect blur; `lens` defines it.
    public var camera: PerspectiveCamera
    public var depthMode: MeshDepthMode
    private let shaders: PostProcessShaders
    private var tiles: (raw: any MTLTexture, dilated: any MTLTexture)?

    public init(
        device: any MTLDevice, lens: CameraLens, camera: PerspectiveCamera, depthMode: MeshDepthMode = .forward,
        settings: Settings = Settings()
    ) throws {
        self.device = device
        self.lens = lens
        self.camera = camera
        self.depthMode = depthMode
        self.settings = settings
        shaders = try PostProcessShaders(device: device)
    }

    public func prepare(pixelFormat: MTLPixelFormat) throws {
        try PostProcessValidation.checkFormat(pixelFormat)
        _ = try shaders.render("frostfield_bokeh_gather", formats: [pixelFormat])
        _ = try shaders.compute("frostfield_bokeh_tiles")
        _ = try shaders.compute("frostfield_bokeh_dilate")
    }

    public func validate(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        try validateSettings()
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination, commandBuffer: commandBuffer,
            requiresDepth: true
        )
        let width = (source.width + Self.tileSize - 1) / Self.tileSize
        let height = (source.height + Self.tileSize - 1) / Self.tileSize
        if tiles?.raw.width != width || tiles?.raw.height != height {
            tiles = (
                try PostProcessShaders.scratch(device: device, width: width, height: height, pixelFormat: .r16Float,
                                               label: "Frostfield bokeh tiles"),
                try PostProcessShaders.scratch(device: device, width: width, height: height, pixelFormat: .r16Float,
                                               label: "Frostfield bokeh dilated tiles")
            )
        }
    }

    public func encode(
        source: any MTLTexture, depth: (any MTLTexture)?, destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(source: source, depth: depth, destination: destination, commandBuffer: commandBuffer)
        guard let depth, let tiles else { throw PostProcessEffectError.missingDepth }
        try depthMode.requireProjectionDepth()
        let tileKernel = try shaders.compute("frostfield_bokeh_tiles")
        let dilateKernel = try shaders.compute("frostfield_bokeh_dilate")
        let gather = try shaders.render("frostfield_bokeh_gather", formats: [destination.pixelFormat])
        var parameters = try parameters(width: source.width)
        try PostProcessShaders.compute(
            label: "Frostfield bokeh tiles", commandBuffer: commandBuffer, pipeline: tileKernel,
            width: tiles.raw.width, height: tiles.raw.height
        ) { encoder in
            encoder.setTexture(depth, index: 0)
            encoder.setTexture(tiles.raw, index: 1)
            encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 0)
        }
        try PostProcessShaders.compute(
            label: "Frostfield bokeh tile dilation", commandBuffer: commandBuffer, pipeline: dilateKernel,
            width: tiles.raw.width, height: tiles.raw.height
        ) { encoder in
            encoder.setTexture(tiles.raw, index: 0)
            encoder.setTexture(tiles.dilated, index: 1)
            encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 0)
        }
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield bokeh gather", commandBuffer: commandBuffer, pipeline: gather
        ) { encoder in
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentTexture(depth, index: 1)
            encoder.setFragmentTexture(tiles.dilated, index: 2)
            encoder.setFragmentBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 0)
        }
    }

    /// Writes each pixel's signed, unclamped thin-lens circle-of-confusion radius in
    /// pixels to an R16Float or R32Float texture with shader-write usage and the
    /// depth's size: negative in front of focus, positive behind it.
    public func encodeCircleOfConfusion(
        depth: any MTLTexture, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        try validateSettings()
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw PostProcessEffectError.commandBuffer
        }
        guard commandBuffer.device === device, depth.device === device, destination.device === device else {
            throw PostProcessEffectError.wrongDevice
        }
        guard PostProcessPass.isReadable(depth, format: .depth32Float) else { throw PostProcessEffectError.depth }
        guard destination.pixelFormat == .r16Float || destination.pixelFormat == .r32Float,
              destination.textureType == .type2D, destination.usage.contains(.shaderWrite),
              destination.width == depth.width, destination.height == depth.height else {
            throw PostProcessEffectError.destination
        }
        try depthMode.requireProjectionDepth()
        let kernel = try shaders.compute("frostfield_bokeh_coc")
        var parameters = try parameters(width: depth.width)
        try PostProcessShaders.compute(
            label: "Frostfield circle of confusion", commandBuffer: commandBuffer, pipeline: kernel,
            width: depth.width, height: depth.height
        ) { encoder in
            encoder.setTexture(depth, index: 0)
            encoder.setTexture(destination, index: 1)
            encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 0)
        }
    }

    private func validateSettings() throws {
        guard settings.maximumRadius.isFinite, (1...Self.maximumRadiusLimit).contains(settings.maximumRadius),
              (16...512).contains(settings.sampleCount),
              settings.bladeCount == 0 || (3...16).contains(settings.bladeCount),
              settings.bladeRotation.isFinite else {
            throw PostProcessEffectError.invalidSettings(
                "Bokeh needs a maximum radius from 1 to 48 pixels, 16 to 512 samples, and 0 or 3 to 16 blades."
            )
        }
        do {
            try lens.validate()
        } catch {
            throw PostProcessEffectError.invalidSettings(error.localizedDescription)
        }
        guard lens.focusDistance >= camera.near else {
            throw PostProcessEffectError.invalidSettings("The focus distance must lie beyond the camera's near plane.")
        }
    }

    private func parameters(width: Int) throws -> Parameters {
        let scale = Float(lens.pixelRadiusScale(imageWidth: width))
        guard scale.isFinite else { throw PostProcessEffectError.invalidSettings("The lens blur scale is not representable.") }
        let dilation = (settings.maximumRadius / Float(Self.tileSize)).rounded(.up)
        return Parameters(
            depth: PostProcessValidation.depthParameters(camera: camera, depthMode: depthMode),
            lens: SIMD4(scale, lens.focusDistance, settings.maximumRadius, Float(settings.sampleCount)),
            aperture: SIMD4(Float(settings.bladeCount), settings.bladeRotation, settings.dither ? 1 : 0, Float(Self.tileSize)),
            tiles: SIMD4(dilation, 0, 0, 0)
        )
    }

    private struct Parameters {
        var depth: SIMD4<Float>
        var lens: SIMD4<Float>
        var aperture: SIMD4<Float>
        var tiles: SIMD4<Float>
    }
}

extension BokehDepthOfField: PostProcessPipelineCounting {
    package var builtPipelineCount: Int { shaders.builtPipelineCount }
}
