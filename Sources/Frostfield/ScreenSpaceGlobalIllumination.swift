import Foundation
import Metal
import simd

/// Diffuse indirect light and ambient visibility from one finished HDR scene view.
/// Use a tracked, retaining command buffer and serialize calls to this pass.
public final class ScreenSpaceGlobalIllumination {
    public enum Normals {
        /// Reconstruct view-space normals from the closest depth derivatives.
        case reconstructFromDepth
        /// The RG32Uint result of `SceneGraph.encodeSurfaceAttributes`.
        case surfaceAttributes(any MTLTexture)
        /// Signed world-space normals from a named RGBA16Float or RGBA32Float surface target.
        case worldNormals(any MTLTexture)
    }

    public struct TemporalSettings: Sendable, Hashable {
        /// Weight of a matching previous sample, from 0 to 0.95.
        public var historyWeight: Float

        public init(historyWeight: Float = 0.85) { self.historyWeight = historyWeight }
    }

    public struct Settings: Sendable, Hashable {
        /// Maximum sampling distance in world units, finite and positive.
        public var radius: Float
        /// Estimated thickness of visible surfaces along the view ray, in world units.
        public var thickness: Float
        /// Maximum search distance in pixels, from 2 to 512.
        public var maximumScreenRadius: Float
        /// Slice directions per pixel, from 1 to 8.
        public var sliceCount: Int
        /// Samples on either side of each slice, from 2 to 32.
        public var stepCount: Int
        /// Step spacing exponent, from 1 (uniform) to 3 (denser near the center).
        public var stepExponent: Float
        /// Indirect radiance multiplier, from 0 to 100.
        public var intensity: Float
        /// Exponent applied to ambient visibility, from 0 to 4. Zero disables AO.
        public var ambientOcclusionPower: Float
        /// Fade length for light samples near the image border, from 0 to 0.5 of the frame.
        public var edgeFade: Float
        /// Share of visible backface radiance allowed to bounce, from 0 to 1.
        public var backfaceLighting: Float
        /// Apply the depth- and normal-aware spatial resolve.
        public var denoise: Bool
        /// Fraction of each source dimension used by the trace, from 0.25 to 1.
        /// Reduced traces always get a full-size edge-aware resolve.
        public var resolutionScale: Float
        /// Use the r187 screen-relative search based on radius and image width.
        public var screenSpaceSampling: Bool
        /// Scale thickness by 100 times the sample distance divided by camera far.
        public var linearThickness: Bool
        /// Rotate and accumulate the trace across frames with matching depth and normals.
        public var temporal: TemporalSettings?

        public init(
            radius: Float = 2, thickness: Float = 0.25, maximumScreenRadius: Float = 128,
            sliceCount: Int = 3, stepCount: Int = 8, stepExponent: Float = 2, intensity: Float = 4,
            ambientOcclusionPower: Float = 1, edgeFade: Float = 0.06, backfaceLighting: Float = 0, denoise: Bool = true,
            resolutionScale: Float = 1, screenSpaceSampling: Bool = false, linearThickness: Bool = false,
            temporal: TemporalSettings? = nil
        ) {
            self.radius = radius
            self.thickness = thickness
            self.maximumScreenRadius = maximumScreenRadius
            self.sliceCount = sliceCount
            self.stepCount = stepCount
            self.stepExponent = stepExponent
            self.intensity = intensity
            self.ambientOcclusionPower = ambientOcclusionPower
            self.edgeFade = edgeFade
            self.backfaceLighting = backfaceLighting
            self.denoise = denoise
            self.resolutionScale = resolutionScale
            self.screenSpaceSampling = screenSpaceSampling
            self.linearThickness = linearThickness
            self.temporal = temporal
        }

        fileprivate var isValid: Bool {
            radius.isFinite && radius > 0 && thickness.isFinite && thickness > 0
                && (2...512).contains(maximumScreenRadius) && (1...8).contains(sliceCount) && (2...32).contains(stepCount)
                && (1...3).contains(stepExponent) && (0...100).contains(intensity) && (0...4).contains(ambientOcclusionPower)
                && (0...0.5).contains(edgeFade) && (0...1).contains(backfaceLighting)
                && (0.25...1).contains(resolutionScale)
                && (0...0.95).contains(temporal?.historyWeight ?? 0)
        }
    }

    public let device: any MTLDevice
    private let shaders: PostProcessShaders
    private let unusedPacked: any MTLTexture
    private let unusedFloat: any MTLTexture
    private var scratch: (any MTLTexture)?
    private var spatial: (any MTLTexture)?
    private var history: History?
    private weak var scratchQueue: (any MTLCommandQueue)?

    var traceTextureSize: SIMD2<Int>? { scratch.map { SIMD2($0.width, $0.height) } }

    public init(device: any MTLDevice) throws {
        self.device = device
        shaders = try PostProcessShaders(device: device)
        unusedPacked = try PostProcessShaders.scratch(
            device: device, width: 1, height: 1, pixelFormat: .rg32Uint, usage: .shaderRead, label: "Frostfield unused SSGI attributes"
        )
        unusedFloat = try PostProcessShaders.scratch(
            device: device, width: 1, height: 1, usage: .shaderRead, label: "Frostfield unused SSGI surface data"
        )
    }

    /// Discards history for a camera cut or abrupt same-surface lighting change.
    public func resetTemporalHistory() { history = nil }

    /// Writes indirect linear HDR radiance to RGB and ambient visibility to A.
    /// Cleared depth writes zero radiance and full visibility. Inputs must be
    /// shader-readable, tracked and from the same view and size. Use RGBA16Float
    /// beauty, Depth32Float depth, and a separate RGBA16Float render target.
    /// `camera` is the source perspective view, including its off-center crop or jitter.
    /// With temporal filtering, optional `motionTexture` uses the current view's
    /// RGBA16Float `RasterMotionVectors` format. RG is previous minus current
    /// in top-left pixels, B is previous stored depth, and A is validity.
    /// Supply vectors for the preceding SSGI frame, with the standard 0...1 depth range.
    /// Without vectors, temporal reprojection follows camera motion.
    public func encode(
        source: any MTLTexture, depth: any MTLTexture, normals: Normals = .reconstructFromDepth,
        motionTexture: (any MTLTexture)? = nil, destination: any MTLTexture, camera: ShaderGraphCamera, depthMode: MeshDepthMode = .forward,
        settings: Settings = .init(), commandBuffer: any MTLCommandBuffer
    ) throws {
        guard settings.isValid, motionTexture == nil || settings.temporal != nil else { throw Failure.settings }
        try depthMode.requireProjectionDepth()
        try PostProcessValidation.check(
            device: device, source: source, depth: depth, destination: destination, commandBuffer: commandBuffer, requiresDepth: true
        )
        var packed = unusedPacked
        var world = unusedFloat
        var normalKind: Float = 0
        switch normals {
        case .reconstructFromDepth: break
        case .surfaceAttributes(let texture):
            try validateInput(texture, formats: [.rg32Uint], matching: source)
            packed = texture
            normalKind = 1
        case .worldNormals(let texture):
            try validateInput(texture, formats: [.rgba16Float, .rgba32Float], matching: source)
            world = texture
            normalKind = 2
        }
        var inputs = [source, depth, packed, world]
        if let motionTexture {
            try validateInput(motionTexture, formats: [RasterMotionVectors.pixelFormat], matching: source)
            inputs.append(motionTexture)
        }
        try validateDestination(destination, reading: inputs)
        let pose = simd_inverse(camera.viewMatrix)
        guard let projection = ScreenSpaceProjection(clipFromView: camera.projectionMatrix, depthMode: depthMode),
              pose[0].w == 0, pose[1].w == 0, pose[2].w == 0, abs(pose[3].w - 1) < 1e-5,
              let rotation = ScreenSpaceProjection.viewFromWorld(pose) else { throw Failure.camera }
        let size = SIMD2(source.width, source.height)
        let traceSize = SIMD2(
            max(1, Int((Float(size.x) * settings.resolutionScale).rounded())),
            max(1, Int((Float(size.y) * settings.resolutionScale).rounded()))
        )
        let needsResolve = traceSize != size || settings.denoise
        let queue = commandBuffer.commandQueue
        if scratchQueue !== queue {
            scratch = nil
            spatial = nil
            scratchQueue = queue
        }

        let previous: History?
        let next: History?
        if settings.temporal != nil {
            previous = history.flatMap {
                $0.canRead(size: size, settings: settings, normalKind: normalKind, mode: depthMode, on: commandBuffer) ? $0 : nil
            }
            next = try History(
                reusing: previous, device: device, size: size, settings: settings, normalKind: normalKind,
                mode: depthMode, camera: camera, projection: projection, command: commandBuffer
            )
        } else {
            previous = nil
            next = nil
        }
        let phase = next?.phase ?? .zero
        let constants = projection.constants(extra: settings.linearThickness ? 1 / camera.far : 0) + [
            SIMD4(settings.radius, settings.thickness, settings.maximumScreenRadius, settings.stepExponent),
            SIMD4(Float(settings.sliceCount), Float(settings.stepCount), normalKind, depthMode == .reversed ? 1 : 0),
            SIMD4(settings.intensity, settings.ambientOcclusionPower, settings.edgeFade, settings.backfaceLighting),
            SIMD4(rotation[0], 0), SIMD4(rotation[1], 0), SIMD4(rotation[2], 0),
            SIMD4(Float(traceSize.x), Float(traceSize.y), settings.screenSpaceSampling ? 1 : 0, settings.linearThickness ? 1 : 0),
            SIMD4(phase.x, phase.y, 0, 0),
        ]
        let gather = try shaders.render("frostfield_ssgi_gather", formats: [.rgba16Float])
        let resolve = try needsResolve ? shaders.render("frostfield_ssgi_resolve", formats: [.rgba16Float]) : nil
        let temporal = try next.map { _ in
            try shaders.render("frostfield_ssgi_temporal", formats: [.rgba16Float, .rgba16Float, .rg32Uint])
        }
        let resolved: any MTLTexture
        if next != nil {
            let intermediate = try workingTexture(spatial, size: size, label: "Frostfield spatial SSGI")
            spatial = intermediate
            resolved = intermediate
        } else {
            resolved = destination
        }
        let trace: any MTLTexture
        if resolve != nil {
            let intermediate = try workingTexture(scratch, size: traceSize, label: "Frostfield raw SSGI")
            scratch = intermediate
            trace = intermediate
        } else {
            scratch = nil
            trace = resolved
        }
        func bind(_ input: any MTLTexture, _ encoder: any MTLRenderCommandEncoder) {
            encoder.setFragmentTexture(input, index: 0)
            encoder.setFragmentTexture(depth, index: 1)
            encoder.setFragmentTexture(packed, index: 2)
            encoder.setFragmentTexture(world, index: 3)
            constants.withUnsafeBytes { encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: 0) }
        }
        try PostProcessShaders.drawFullScreen(
            into: trace, label: "Frostfield SSGI visibility slices", commandBuffer: commandBuffer, pipeline: gather
        ) { bind(source, $0) }
        if let resolve {
            try PostProcessShaders.drawFullScreen(
                into: resolved, label: "Frostfield SSGI spatial resolve", commandBuffer: commandBuffer, pipeline: resolve
            ) { bind(trace, $0) }
        }
        if let temporal, let next {
            var reprojection = History.Reprojection(
                previous: previous, camera: camera, pose: pose, weight: settings.temporal?.historyWeight ?? 0,
                usesMotion: motionTexture != nil
            )
            try PostProcessShaders.drawFullScreen(
                into: destination, extra: [next.sample.light, next.sample.geometry],
                label: "Frostfield SSGI temporal resolve", commandBuffer: commandBuffer, pipeline: temporal
            ) { encoder in
                bind(resolved, encoder)
                encoder.setFragmentTexture(previous?.sample.light ?? unusedFloat, index: 4)
                encoder.setFragmentTexture(previous?.sample.geometry ?? unusedPacked, index: 5)
                encoder.setFragmentTexture(motionTexture ?? unusedFloat, index: 6)
                encoder.setFragmentBytes(&reprojection, length: MemoryLayout<History.Reprojection>.stride, index: 1)
            }
        }
        history = next
    }

    private func workingTexture(_ current: (any MTLTexture)?, size: SIMD2<Int>, label: String) throws -> any MTLTexture {
        if let current, current.width == size.x, current.height == size.y { return current }
        return try PostProcessShaders.scratch(
            device: device, width: size.x, height: size.y, usage: [.renderTarget, .shaderRead], label: label
        )
    }

    /// Adds the indirect radiance to beauty in linear HDR space and preserves
    /// beauty alpha. Supply straight linear diffuse albedo to tint the receiver,
    /// such as an opaque diffuse material's named base-color target. Nil uses
    /// unit albedo. AO stays separate for use on the scene's indirect light.
    public func composite(
        source: any MTLTexture, illumination: any MTLTexture, diffuseAlbedo: (any MTLTexture)? = nil,
        destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        try PostProcessValidation.check(
            device: device, source: source, depth: nil, destination: destination, commandBuffer: commandBuffer, requiresDepth: false
        )
        try validateInput(illumination, formats: [.rgba16Float], matching: source)
        if let diffuseAlbedo { try validateInput(diffuseAlbedo, formats: [.rgba16Float, .rgba32Float], matching: source) }
        try validateDestination(destination, reading: [source, illumination, diffuseAlbedo ?? unusedFloat])
        let pipeline = try shaders.render("frostfield_ssgi_composite", formats: [.rgba16Float])
        var hasAlbedo: UInt32 = diffuseAlbedo == nil ? 0 : 1
        try PostProcessShaders.drawFullScreen(
            into: destination, label: "Frostfield SSGI composite", commandBuffer: commandBuffer, pipeline: pipeline
        ) { encoder in
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentTexture(illumination, index: 1)
            encoder.setFragmentTexture(diffuseAlbedo ?? unusedFloat, index: 2)
            encoder.setFragmentBytes(&hasAlbedo, length: MemoryLayout<UInt32>.size, index: 0)
        }
    }

    private func validateInput(_ texture: any MTLTexture, formats: [MTLPixelFormat], matching source: any MTLTexture) throws {
        guard texture.device === device else { throw PostProcessEffectError.wrongDevice }
        guard formats.contains(texture.pixelFormat), PostProcessPass.isReadable(texture, format: texture.pixelFormat) else {
            throw Failure.input
        }
        guard texture.width == source.width, texture.height == source.height else { throw PostProcessEffectError.sizeMismatch }
    }

    private func validateDestination(_ destination: any MTLTexture, reading: [any MTLTexture]) throws {
        guard destination.pixelFormat == .rgba16Float, !destination.isFramebufferOnly,
              destination.hazardTrackingMode != .untracked, !destination.isAliasable(),
              !reading.contains(where: { $0.isAliasable() || TextureTransfer.sharesStorage($0, destination) }) else {
            throw Failure.destination
        }
    }

    private struct History {
        struct Sample {
            let light: any MTLTexture
            let geometry: any MTLTexture

            init(device: any MTLDevice, size: SIMD2<Int>) throws {
                light = try PostProcessShaders.scratch(
                    device: device, width: size.x, height: size.y, usage: [.renderTarget, .shaderRead], label: "Frostfield SSGI history"
                )
                geometry = try PostProcessShaders.scratch(
                    device: device, width: size.x, height: size.y, pixelFormat: .rg32Uint,
                    usage: [.renderTarget, .shaderRead], label: "Frostfield SSGI history geometry"
                )
            }
        }

        struct Reprojection {
            let previousViewFromCurrent: simd_float4x4
            let previousClipFromCurrent: simd_float4x4
            let options: SIMD4<Float>

            init(previous: History?, camera: ShaderGraphCamera, pose: simd_float4x4, weight: Float, usesMotion: Bool) {
                if let previous {
                    previousViewFromCurrent = previous.camera.viewMatrix == camera.viewMatrix
                        ? matrix_identity_float4x4 : previous.camera.viewMatrix * pose
                    previousClipFromCurrent = previous.camera.projectionMatrix * previousViewFromCurrent
                    options = SIMD4(weight, usesMotion ? 1 : 0, previous.projection.farInverseDistance,
                                    previous.projection.inverseDistanceRange)
                } else {
                    previousViewFromCurrent = matrix_identity_float4x4
                    previousClipFromCurrent = matrix_identity_float4x4
                    options = .zero
                }
            }
        }

        let sample: Sample
        private let spare: Sample
        private let size: SIMD2<Int>
        private let settings: Settings
        private let normalKind: Float
        private let mode: MeshDepthMode
        let camera: ShaderGraphCamera
        let projection: ScreenSpaceProjection
        private let command: any MTLCommandBuffer
        private let frame: Int

        var phase: SIMD2<Float> {
            SIMD2([60, 300, 180, 240, 120, 0][frame % 6] / 360.0, [0, 0.5, 0.25, 0.75][frame % 4])
        }

        init(
            reusing previous: History?, device: any MTLDevice, size: SIMD2<Int>, settings: Settings, normalKind: Float,
            mode: MeshDepthMode, camera: ShaderGraphCamera, projection: ScreenSpaceProjection, command: any MTLCommandBuffer
        ) throws {
            sample = try previous?.spare ?? Sample(device: device, size: size)
            spare = try previous?.sample ?? Sample(device: device, size: size)
            self.size = size
            self.settings = settings
            self.normalKind = normalKind
            self.mode = mode
            self.camera = camera
            self.projection = projection
            self.command = command
            frame = ((previous?.frame ?? -1) + 1) % 12
        }

        func canRead(size: SIMD2<Int>, settings: Settings, normalKind: Float, mode: MeshDepthMode, on next: any MTLCommandBuffer) -> Bool {
            // An already-enqueued producer precedes this frame on the same Metal queue.
            // With a different, unqueued producer there is no ordering promise.
            self.size == size && self.settings == settings && self.normalKind == normalKind && self.mode == mode
                && command.commandQueue === next.commandQueue && command.status != .error
                && (command === next || command.status == .completed
                    || (command.status != .notEnqueued && next.status == .notEnqueued))
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case settings, camera, input, destination

        public var errorDescription: String? {
            switch self {
            case .settings: "Use SSGI settings within their documented bounds and enable temporal filtering to supply motion."
            case .camera: "SSGI needs a rigid camera and a perspective projection with the selected depth convention."
            case .input: "SSGI needs tracked shader-readable normals, motion, illumination or albedo in the selected format."
            case .destination: "SSGI writes a distinct, tracked RGBA16Float render target."
            }
        }
    }
}
