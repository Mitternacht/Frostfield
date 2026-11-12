import CoreGraphics
import Foundation
import Metal
import simd
import Synchronization

/// Presents an HDR image as display-referred color with a choice of tone mapping
/// operators, like Three.js `renderer.toneMapping` and `toneMappingExposure`.
///
/// The default operator is Khronos PBR Neutral. Every operator receives straight
/// (unpremultiplied), nonnegative linear sRGB scaled by the exposure, and the
/// result is premultiplied again. For a custom curve, like Three.js
/// `CustomToneMapping`, build a graph with `PostProcessGraph.toneMapping(exposureStops:curve:)`.
///
/// Standard and extended range: with the default `Settings.displayHeadroom` of 1,
/// every operator except `.none` writes 0...1, as for an SDR display. To keep EDR
/// headroom, render to RGBA16Float presented with extended range (for example a
/// `CAMetalLayer` with `wantsExtendedDynamicRangeContent` and an extended linear
/// sRGB color space) and pass the display's current headroom, such as
/// `NSScreen.maximumExtendedDynamicRangeColorComponentValue` or
/// `UIScreen.currentEDRHeadroom`. Each operator then writes 0...headroom as its
/// case describes, and `.none` never clamps highlights.
///
/// Dithering: `Settings.dithering` adds Three.js r187's `material.dithering` shift to the
/// final color, so that quantizing to 8 bits hides banding. It applies only when the pass
/// writes an 8-bit destination directly, because that is where the color is quantized, and
/// it is off by default, which leaves every output byte unchanged. See
/// `docs/evidence/dithering.md` for the exact departures from r187, including why the
/// switch belongs to the display pass and not to a material.
///
/// The headroom only adds range for highlights: shadows and mid-tones keep their
/// SDR values for every headroom, so a headroom that follows the screen's
/// brightness does not change them. `.neutral` keeps its output below 0.76 and aims
/// its shoulder at the headroom. The other curves keep their output wherever every
/// exposed scene channel is at most 1 (exposed diffuse white). Above that, the
/// pixel's largest display channel rises by
/// `E = (H − 1) Δ² / ((Δ + H − w)(Δ + 1 − w))` with `Δ = w (m − 1)`, where `H` is
/// the headroom, `m` the largest exposed scene channel and `w` the curve's output
/// at exposed white, and RGB scales with it, keeping the SDR curve's hue and
/// saturation. `E` starts at zero with zero slope, so the curve stays smooth, and
/// approaches `H − 1`: highlights continue at about the curve's gain at white and
/// roll off toward the peak.
public final class HDRDisplay {
    /// Output primaries for the display pass. The source is linear sRGB.
    public enum OutputColorSpace: UInt32, Sendable {
        case sRGB
        case displayP3
        /// Extended linear Rec. 2020, stored in RGBA16Float.
        case rec2020

        fileprivate func profileName(for format: MTLPixelFormat) -> CFString? {
            switch (self, format) {
                case (.sRGB, .rgba16Float): CGColorSpace.extendedLinearSRGB
                case (.sRGB, .rgba8Unorm_srgb), (.sRGB, .bgra8Unorm_srgb): CGColorSpace.sRGB
                case (.sRGB, _): CGColorSpace.linearSRGB
                case (.displayP3, .rgba16Float): CGColorSpace.extendedLinearDisplayP3
                case (.displayP3, .rgba8Unorm_srgb), (.displayP3, .bgra8Unorm_srgb): CGColorSpace.displayP3
                case (.displayP3, _): CGColorSpace.linearDisplayP3
                case (.rec2020, .rgba16Float): CGColorSpace.extendedLinearITUR_2020
                case (.rec2020, _): nil
            }
        }
    }

    /// A curve from exposed scene-linear color to display-linear color.
    ///
    /// Formulas follow the Three.js r187 shader chunk where Three.js has the operator.
    /// With a display headroom above 1, `.none`, `.linear` and `.neutral` behave as
    /// their cases describe, and every other curve keeps its SDR output up to exposed
    /// white and extends brighter highlights as `HDRDisplay` describes.
    public enum ToneMapping: Sendable, Hashable {
        /// No curve, like Three.js `NoToneMapping` with zero exposure stops. Exposed
        /// color is written unchanged, clamped only to 0...65,504, whatever the headroom.
        /// An 8-bit destination clips it to 1 when stored.
        case none
        /// Three.js `LinearToneMapping`: exposed color clamped to 0...1, or to the
        /// headroom when it is above 1.
        case linear
        /// Three.js `ReinhardToneMapping`, `x / (1 + x)` per channel.
        case reinhard
        /// Reinhard et al. 2002 (equation 4) per channel, `x (1 + x / W²) / (1 + x)`:
        /// the exposed scene value `whitePoint` (W, finite and positive) maps to SDR
        /// white and brighter values clip in SDR. With headroom, values above exposed
        /// white continue into it like the other curves.
        case reinhardExtended(whitePoint: Float)
        /// Three.js `CineonToneMapping`, Hejl and Burgess-Dawson's filmic curve.
        case cineon
        /// Three.js `ACESFilmicToneMapping`: Stephen Hill's RRT and ODT fit with
        /// its sRGB-to-AP1 matrices, including Three.js's 1/0.6 exposure.
        case acesFilmic
        /// Krzysztof Narkowicz's per-channel ACES fit, pre-exposed so that 1 maps
        /// to about 0.8. It saturates bright colors more than `.acesFilmic`.
        case acesNarkowicz
        /// Three.js `AgXToneMapping` (Blender's Rec.2020 AgX base via Filament) with
        /// an optional look. Write `.agx()` for the base look.
        case agx(look: AgXLook = .default)
        /// Khronos PBR Neutral, Three.js `NeutralToneMapping`. With headroom the
        /// shoulder approaches the headroom instead of 1, so colors below 0.76 keep
        /// their SDR values; only highlights use the extra range.
        case neutral

        var shaderCode: UInt32 {
            switch self {
            case .none: 0
            case .linear: 1
            case .reinhard: 2
            case .reinhardExtended: 3
            case .cineon: 4
            case .acesFilmic: 5
            case .acesNarkowicz: 6
            case .agx: 7
            case .neutral: 8
            }
        }
    }

    /// An ASC CDL look applied after the AgX sigmoid, with the slope, power and
    /// saturation of Troy Sobotka's AgX configuration.
    public enum AgXLook: Sendable, Hashable {
        /// The AgX base image with no look, as Three.js.
        case `default`
        /// Power 1.35 and saturation 1.4: denser, more saturated color.
        case punchy
        /// Slope (1, 0.9, 0.5), power 0.8 and saturation 1.3: a warm, lifted look.
        case golden

        var shaderCode: UInt32 {
            switch self {
            case .default: 0
            case .punchy: 1
            case .golden: 2
            }
        }
    }

    public struct Settings: Sendable, Equatable {
        /// Exposure adjustment before tone mapping, from -16 to +16 stops.
        public var exposureStops: Float
        /// Applies spatial smoothing to high-contrast diagonal edges in the same pass.
        public var antiAliasing: Bool
        /// The curve from exposed scene color to display color. Defaults to `.neutral`.
        public var toneMapping: ToneMapping
        /// The brightest value the destination can show, as a multiple of SDR white,
        /// from 1 to 64. Values above 1 need an RGBA16Float destination presented
        /// with extended dynamic range. See `HDRDisplay`.
        public var displayHeadroom: Float
        /// Adds the Three.js r187 `material.dithering` shift to the final premultiplied color:
        /// a screen-space hash of the pixel moves each channel by up to half of one 8-bit step,
        /// plus, minus, plus for red, green and blue. It needs an 8-bit RGBA or BGRA destination
        /// and throws `Failure.ditheringNeedsEightBitDestination` for RGBA16Float. An sRGB
        /// destination is shifted in its encoded values, so the stored code moves by half a step
        /// at every brightness. Fully transparent pixels stay zero. Defaults to `false`.
        public var dithering: Bool

        public init(
            exposureStops: Float = 0,
            antiAliasing: Bool = false,
            toneMapping: ToneMapping = .neutral,
            displayHeadroom: Float = 1,
            dithering: Bool = false
        ) {
            self.exposureStops = exposureStops
            self.antiAliasing = antiAliasing
            self.toneMapping = toneMapping
            self.displayHeadroom = displayHeadroom
            self.dithering = dithering
        }
    }

    /// The largest supported `Settings.displayHeadroom`.
    public static let maximumDisplayHeadroom: Float = 64

    /// The fragment options. Matches `HDRDisplayOptions` in HDRDisplay.metal.
    private struct ShaderOptions {
        var exposure: Float
        var toneMapping: UInt32
        var look: UInt32
        var inverseWhiteSquared: Float
        var headroom: Float
        var outputColorSpace: UInt32
        var ditherMode: UInt32
    }

    package let device: any MTLDevice
    public let destinationPixelFormat: MTLPixelFormat
    public let outputColorSpace: OutputColorSpace
    /// The profile for presenting the destination texture, including its stored transfer.
    /// Set this on the destination's compositor or image metadata.
    public let presentationColorSpace: CGColorSpace
    private let pipeline: any MTLRenderPipelineState

    public init(
        device: any MTLDevice, destinationPixelFormat: MTLPixelFormat, outputColorSpace: OutputColorSpace = .sRGB
    ) throws {
        switch destinationPixelFormat {
        case .rgba16Float, .rgba8Unorm, .rgba8Unorm_srgb, .bgra8Unorm, .bgra8Unorm_srgb: break
        default: throw Failure.destinationFormat
        }
        guard let profileName = outputColorSpace.profileName(for: destinationPixelFormat),
              let profile = CGColorSpace(name: profileName) else {
            throw Failure.outputColorSpace
        }
        self.device = device
        self.destinationPixelFormat = destinationPixelFormat
        self.outputColorSpace = outputColorSpace
        presentationColorSpace = profile
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let vertex = library.makeFunction(name: "frostfield_hdr_vertex"),
                  let fragment = library.makeFunction(name: "frostfield_hdr_fragment") else {
                throw Failure.missingShader
            }
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.label = "Frostfield HDR display"
            descriptor.vertexFunction = vertex
            descriptor.fragmentFunction = fragment
            descriptor.colorAttachments[0].pixelFormat = destinationPixelFormat
            pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
    }

    /// Encode after the HDR scene and post effects on the caller's retained command
    /// buffer. The source must be a standalone, tracked RGBA16Float texture with
    /// finite, premultiplied linear sRGB color. The destination is a separate,
    /// equal-sized render target. Tone curves run in sRGB primaries before the
    /// output transform. With `.none` a wide-gamut transform uses signed linear
    /// color, preserving source components outside sRGB. Metal applies its RGB
    /// transfer function when the destination has an sRGB pixel format.
    /// Alpha remains linear and is filtered
    /// together with premultiplied RGB when anti-aliasing is enabled. Settings are
    /// validated before encoding: a headroom above 1 needs a display prepared for
    /// RGBA16Float, and an extended Reinhard white point must be finite and positive.
    public func encode(
        source: any MTLTexture,
        destination: any MTLTexture,
        settings: Settings = Settings(),
        untonemappedMask: (any MTLTexture)? = nil,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try encode(
            source: source, destination: destination, settings: settings, resamples: false,
            fade: nil, untonemappedMask: untonemappedMask, commandBuffer: commandBuffer
        )
    }

    /// Samples a progressive preview at normalized output pixel centers.
    package func encodeScaled(
        source: any MTLTexture, destination: any MTLTexture, settings: Settings,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try encode(
            source: source, destination: destination, settings: settings, resamples: true,
            fade: nil, untonemappedMask: nil, commandBuffer: commandBuffer
        )
    }

    /// Blends independently scaled premultiplied linear images before tone mapping.
    package func encodeCrossfade(
        from: any MTLTexture, to: any MTLTexture, weight: Float, destination: any MTLTexture,
        settings: Settings, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard (0...1).contains(weight) else { throw Failure.incompatibleTextures }
        _ = try validate(
            source: from, destination: destination, settings: settings, resamples: true, commandBuffer: commandBuffer
        )
        try encode(
            source: to, destination: destination, settings: settings, resamples: true,
            fade: (from, weight), untonemappedMask: nil, commandBuffer: commandBuffer
        )
    }

    private func encode(
        source: any MTLTexture, destination: any MTLTexture, settings: Settings, resamples: Bool,
        fade: (texture: any MTLTexture, weight: Float)?, untonemappedMask: (any MTLTexture)?,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        var options = try validate(
            source: source, destination: destination, settings: settings,
            resamples: resamples, commandBuffer: commandBuffer
        )
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = destination
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        if let untonemappedMask {
            guard untonemappedMask.device === device, untonemappedMask.textureType == .type2D,
                  untonemappedMask.pixelFormat == .r8Unorm, untonemappedMask.sampleCount == 1,
                  untonemappedMask.width == destination.width, untonemappedMask.height == destination.height,
                  untonemappedMask.usage.contains(.shaderRead), !untonemappedMask.isFramebufferOnly else {
                throw Failure.incompatibleTextures
            }
        }
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            throw Failure.encoder
        }
        var antiAliasing: Float = settings.antiAliasing ? 1 : 0
        encoder.label = "Frostfield HDR display"
        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBytes(&options, length: MemoryLayout<ShaderOptions>.size, index: 0)
        encoder.setFragmentTexture(source, index: 0)
        encoder.setFragmentTexture(fade?.texture ?? source, index: 1)
        encoder.setFragmentBytes(&options, length: MemoryLayout<ShaderOptions>.size, index: 0)
        encoder.setFragmentBytes(&antiAliasing, length: MemoryLayout<Float>.size, index: 1)
        var destinationSize = SIMD2(UInt32(destination.width), UInt32(destination.height))
        encoder.setFragmentBytes(&destinationSize, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
        var weight = fade?.weight ?? 1
        encoder.setFragmentBytes(&weight, length: MemoryLayout<Float>.size, index: 3)
        encoder.setFragmentTexture(untonemappedMask ?? source, index: 2)
        var hasMask: UInt32 = untonemappedMask == nil ? 0 : 1
        encoder.setFragmentBytes(&hasMask, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    /// Checks everything `encode` requires without encoding and returns the shader options.
    private func validate(
        source: any MTLTexture, destination: any MTLTexture, settings: Settings, resamples: Bool,
        commandBuffer: any MTLCommandBuffer
    ) throws -> ShaderOptions {
        let (inverseWhiteSquared, look) = try Self.check(settings)
        guard settings.displayHeadroom == 1 || destinationPixelFormat == .rgba16Float else {
            throw Failure.headroomNeedsExtendedRange
        }
        let ditherMode: UInt32
        switch (settings.dithering, destinationPixelFormat) {
        case (false, _): ditherMode = 0
        case (true, .rgba8Unorm), (true, .bgra8Unorm): ditherMode = 1
        case (true, .rgba8Unorm_srgb), (true, .bgra8Unorm_srgb): ditherMode = 2
        case (true, _): throw Failure.ditheringNeedsEightBitDestination
        }
        guard commandBuffer.retainedReferences else { throw Failure.commandBuffer }
        guard commandBuffer.device === device, source.device === device, destination.device === device else {
            throw Failure.wrongDevice
        }
        guard source !== destination, source.textureType == .type2D, destination.textureType == .type2D,
              source.sampleCount == 1, destination.sampleCount == 1,
              source.width > 0, source.height > 0,
              resamples || (source.width == destination.width && source.height == destination.height),
              source.pixelFormat == .rgba16Float, destination.pixelFormat == destinationPixelFormat,
              source.usage.contains(.shaderRead), !source.isFramebufferOnly,
              destination.usage.contains(.renderTarget), destination.storageMode != .memoryless,
              source.hazardTrackingMode != .untracked,
              source.parent == nil, destination.parent == nil,
              source.buffer == nil, source.iosurface == nil, source.heap == nil else {
            throw Failure.incompatibleTextures
        }
        return ShaderOptions(
            exposure: exp2f(settings.exposureStops), toneMapping: settings.toneMapping.shaderCode,
            look: look.shaderCode, inverseWhiteSquared: inverseWhiteSquared, headroom: settings.displayHeadroom,
            outputColorSpace: outputColorSpace.rawValue, ditherMode: ditherMode
        )
    }

    /// Checks the settings that do not depend on textures and returns the operator's
    /// parameters: extended Reinhard's 1 / W², evaluated in double precision and
    /// limited to the largest finite float so that a tiny white point cannot produce
    /// infinity times zero on the GPU, and the AgX look.
    static func check(_ settings: Settings) throws -> (inverseWhiteSquared: Float, look: AgXLook) {
        guard (-16...16).contains(settings.exposureStops) else { throw Failure.invalidExposure }
        guard settings.displayHeadroom.isFinite, (1...maximumDisplayHeadroom).contains(settings.displayHeadroom) else {
            throw Failure.invalidHeadroom
        }
        switch settings.toneMapping {
        case .reinhardExtended(let white):
            guard white.isFinite, white > 0 else { throw Failure.invalidWhitePoint }
            let inverse = 1 / (Double(white) * Double(white))
            return (Float(min(inverse, Double(Float.greatestFiniteMagnitude))), .default)
        case .agx(let look):
            return (1, look)
        default:
            return (1, .default)
        }
    }

    /// Validation for `PostProcessChain`, which checks every stage before encoding any.
    func check(
        source: any MTLTexture, destination: any MTLTexture, settings: Settings, commandBuffer: any MTLCommandBuffer
    ) throws {
        _ = try validate(
            source: source, destination: destination, settings: settings, resamples: false, commandBuffer: commandBuffer
        )
    }

    public enum Failure: LocalizedError, Equatable {
        case missingShader
        case pipeline(String)
        case destinationFormat
        case outputColorSpace
        case invalidExposure
        case invalidHeadroom
        case headroomNeedsExtendedRange
        case ditheringNeedsEightBitDestination
        case invalidWhitePoint
        case commandBuffer
        case wrongDevice
        case incompatibleTextures
        case encoder

        public var errorDescription: String? {
            switch self {
            case .missingShader: "Frostfield’s HDR display shaders are missing."
            case .pipeline(let detail): "Could not prepare Frostfield’s HDR display. \(detail)"
            case .destinationFormat: "HDR display needs RGBA16Float or an 8-bit RGBA or BGRA target."
            case .outputColorSpace: "Use RGBA16Float for the extended linear Rec. 2020 output profile."
            case .invalidExposure: "Exposure must be finite and between -16 and +16 stops."
            case .invalidHeadroom: "Display headroom must be finite and between 1 and \(Int(HDRDisplay.maximumDisplayHeadroom))."
            case .headroomNeedsExtendedRange:
                "Display headroom above 1 needs an HDR display prepared for an RGBA16Float, extended-range destination."
            case .ditheringNeedsEightBitDestination:
                "Dithering needs the display pass to write an 8-bit RGBA or BGRA destination directly. In a post-process chain, use it on the final, unscaled and unmasked tone-mapping stage of a chain that writes an 8-bit destination, and not with a reduced scene capture that is resampled afterwards."
            case .invalidWhitePoint: "The extended Reinhard white point must be finite and positive."
            case .commandBuffer: "HDR display needs a command buffer that retains resources."
            case .wrongDevice: "HDR display textures and command buffer must use its Metal device."
            case .incompatibleTextures:
                "HDR display needs a tracked, shader-readable RGBA16Float source and a separate, equal-sized render target."
            case .encoder: "Metal could not create the HDR display encoder."
            }
        }
    }
}

// MARK: - Custom tone mapping

extension SceneGraph {
    /// Marks visible pixels whose material has `toneMapped == false`.
    /// The depth texture must be the one stored by the color pass. The mask is cleared to zero,
    /// then each opted-out mesh is drawn white with an equal depth test.
    public func encodeToneMapMask(
        depth: any MTLTexture, into mask: any MTLTexture, commandBuffer: any MTLCommandBuffer,
        depthMode: MeshDepthMode = .forward
    ) throws {
        guard mask.pixelFormat == .r8Unorm, mask.textureType == .type2D, depth.textureType == .type2D,
              mask.width == depth.width, mask.height == depth.height, mask.width > 0,
              mask.usage.contains(.renderTarget), depth.pixelFormat == .depth32Float,
              commandBuffer.device === mask.device, depth.device === mask.device else {
            throw HDRDisplay.Failure.incompatibleTextures
        }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = mask
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .load
        pass.depthAttachment.storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            throw HDRDisplay.Failure.encoder
        }
        try encodeMaterialMask(
            into: mask, depthMode: depthMode, label: "Frostfield tone-map exclusion", encoder: encoder
        ) { mesh in
            mesh.isMaterialVisible && !mesh.toneMapped
        }
    }

    /// Marks visible pixels whose material has `fog == false`.
    /// The depth texture must be the one stored by the color pass. The mask is cleared to zero,
    /// then each opted-out mesh is drawn white with an equal depth test.
    public func encodeFogExclusionMask(
        depth: any MTLTexture, into mask: any MTLTexture, commandBuffer: any MTLCommandBuffer,
        depthMode: MeshDepthMode = .forward
    ) throws {
        guard mask.pixelFormat == .r8Unorm, mask.textureType == .type2D, depth.textureType == .type2D,
              mask.width == depth.width, mask.height == depth.height, mask.width > 0,
              mask.usage.contains(.renderTarget), depth.pixelFormat == .depth32Float,
              commandBuffer.device === mask.device, depth.device === mask.device else {
            throw HDRDisplay.Failure.incompatibleTextures
        }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = mask
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .load
        pass.depthAttachment.storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            throw HDRDisplay.Failure.encoder
        }
        try encodeMaterialMask(
            into: mask, depthMode: depthMode, label: "Frostfield fog exclusion", encoder: encoder
        ) { mesh in
            mesh.isMaterialVisible && !mesh.participatesInFog
        }
    }

    private func encodeMaterialMask(
        into mask: any MTLTexture, depthMode: MeshDepthMode, label: String, encoder: any MTLRenderCommandEncoder,
        include: (MeshRenderer) -> Bool
    ) throws {
        encoder.label = label
        let aspect = Float(mask.width) / Float(mask.height)
        let clip = try activeProjection(aspectRatio: aspect, depthMode: depthMode) * simd_inverse(cameraTransform.matrix)
        for node in try resolve() where renders(node) {
            guard let mesh = node.mesh, include(mesh) else { continue }
            let resources = try ToneMapMaskRenderer.resources(device: mesh.device, skinned: mesh.skin != nil)
            try mesh.encodeToneMapMask(
                world: node.worldTransform, clipFromWorld: clip, into: encoder,
                pipeline: resources.pipeline, depthState: resources.depth
            )
        }
        encoder.endEncoding()
    }
}

private enum ToneMapMaskRenderer {
    struct Resources {
        let pipeline: any MTLRenderPipelineState
        let depth: any MTLDepthStencilState
    }

    private static let cache = Mutex<[ObjectIdentifier: (plain: Resources, skinned: Resources)]>([:])

    static func resources(device: any MTLDevice, skinned: Bool) throws -> Resources {
        let key = ObjectIdentifier(device)
        if let cached = cache.withLock({ $0[key] }) {
            return skinned ? cached.skinned : cached.plain
        }
        let library = try device.makeDefaultLibrary(bundle: .module)
        func make(_ vertex: String) throws -> Resources {
            guard let vertexFunction = library.makeFunction(name: vertex),
                  let fragment = library.makeFunction(name: "frostfield_tonemap_mask_fragment") else {
                throw HDRDisplay.Failure.missingShader
            }
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = vertexFunction
            descriptor.fragmentFunction = fragment
            descriptor.colorAttachments[0].pixelFormat = .r8Unorm
            descriptor.depthAttachmentPixelFormat = .depth32Float
            let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
            let depthDescriptor = MTLDepthStencilDescriptor()
            depthDescriptor.depthCompareFunction = .equal
            depthDescriptor.isDepthWriteEnabled = false
            guard let depth = device.makeDepthStencilState(descriptor: depthDescriptor) else {
                throw HDRDisplay.Failure.encoder
            }
            return Resources(pipeline: pipeline, depth: depth)
        }
        let pair = (plain: try make("frostfield_mesh_vertex"), skinned: try make("frostfield_skinned_mesh_vertex"))
        cache.withLock { store in
            if store[key] == nil { store[key] = pair }
        }
        return skinned ? pair.skinned : pair.plain
    }
}

extension PostProcessGraph {
    /// A display pass that applies a caller-written tone curve, like Three.js
    /// `CustomToneMapping`.
    ///
    /// The graph does the bookkeeping `HDRDisplay` does: it unpremultiplies the
    /// source pixel, clamps negative RGB to zero, multiplies by `2^ExposureStops`,
    /// calls `curve` with that straight scene-linear RGB, and premultiplies the
    /// curve's display-linear RGB by the source alpha. Fully transparent pixels stay
    /// zero. The scalar uniform "ExposureStops" starts at `exposureStops`.
    ///
    /// Prepare it with `PostProcessPass` for a BGRA8Unorm_sRGB or RGBA8Unorm_sRGB
    /// destination, where Metal encodes the result, or for RGBA16Float. The curve's
    /// output is not clamped to 1, so an RGBA16Float destination presented with
    /// extended range keeps values above SDR white; 8-bit destinations clip them.
    ///
    /// ```swift
    /// // Reinhard on luminance instead of per channel.
    /// let display = try PostProcessGraph.toneMapping { _, color in
    ///     let luminance = color.luminance()
    ///     return color / (luminance + 1)
    /// }
    /// ```
    public static func toneMapping(
        exposureStops: Float = 0,
        curve: (ShaderGraphBuilder, ShaderValue<SIMD3<Float>>) throws -> ShaderValue<SIMD3<Float>>
    ) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let source = p.sceneColor(at: p.screenUV)
            let alpha = source.a.saturated()
            let exposure = p.uniform("ExposureStops", exposureStops).exp2()
            // Division by zero alpha yields zero, so transparent pixels stay zero.
            let straight = source.rgb.max(SIMD3<Float>(repeating: 0)) * exposure / alpha
            return p.vector(try curve(p, straight) * alpha, alpha)
        }
    }
}
