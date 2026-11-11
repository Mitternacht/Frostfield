import Foundation
import Metal
import simd

/// Ground-truth ambient occlusion (GTAO) from stored depth, like Three.js `GTAOPass`
/// and `GTAONode`.
///
/// For each pixel the pass searches the depth buffer for the highest horizon on both sides
/// of several screen-space slices through the view vector, then integrates the
/// cosine-weighted visible arc between them in closed form (Jimenez et al. 2016). It also
/// returns the bent normal, the mean unoccluded direction. A 4×4 rotation pattern spreads
/// slice directions across neighboring pixels, and a depth- and normal-aware 4×4 filter
/// averages them. The visible integrals of all slices, weighted by the projected normal's
/// length, are divided once by the same weighted sum of their unoccluded integrals, so open
/// surfaces read exactly one. By default depth beyond the horizon is assumed solid.
///
/// Use `encodeVisibility` before the color pass to feed PBR shading through
/// `SceneGraph.encode(...screenSpaceOcclusion:)`, which occludes only indirect light with
/// multiple-bounce and specular occlusion, or `encode(source:...)` to darken a finished
/// frame. Serialize this pass's encodes and use command buffers that retain references.
public final class AmbientOcclusion {
    /// Settings for the optional legacy GTAO Poisson spatial resolve.
    public struct PoissonSettings: Sendable, Hashable {
        /// Samples on the spiral, from 2 to 64, plus the center.
        public var sampleCount: Int
        /// Turns around the spiral, from 1 to 32.
        public var rings: Int
        /// Maximum radius in full-size pixels, from 1 to 64.
        public var radius: Float
        /// Positive exponent for spacing taps between one pixel and radius.
        public var radiusExponent: Float
        /// Positive tolerance for visibility differences.
        public var lumaPhi: Float
        /// Positive receiver-plane distance tolerance in world units.
        public var depthPhi: Float
        /// Positive exponent for normal similarity.
        public var normalPhi: Float
        /// Selects one of four deterministic kernel rotations.
        public var sampleIndex: Int

        public init(
            sampleCount: Int = 16, rings: Int = 2, radius: Float = 8, radiusExponent: Float = 2,
            lumaPhi: Float = 10, depthPhi: Float = 2, normalPhi: Float = 3, sampleIndex: Int = 0
        ) {
            self.sampleCount = sampleCount
            self.rings = rings
            self.radius = radius
            self.radiusExponent = radiusExponent
            self.lumaPhi = lumaPhi
            self.depthPhi = depthPhi
            self.normalPhi = normalPhi
            self.sampleIndex = sampleIndex
        }

        fileprivate var isValid: Bool {
            (2...64).contains(sampleCount) && (1...32).contains(rings) && (1...64).contains(radius)
                && (0...3).contains(sampleIndex) && [radiusExponent, lumaPhi, depthPhi, normalPhi].allSatisfy {
                    $0.isFinite && $0 > 0
                }
        }

        fileprivate var offsets: [SIMD2<Float>] {
            (0..<sampleCount).map { index in
                let angle = 2 * Double.pi * Double(rings) * Double(index) / Double(sampleCount)
                let distance = 1 + Double(radius - 1) * pow(Double(index) / Double(sampleCount - 1), Double(radiusExponent))
                return SIMD2(Float(cos(angle) * distance), Float(sin(angle) * distance))
            }
        }
    }

    public struct Settings: Sendable {
        /// Maximum occluder distance in camera/world units. Must be finite and positive.
        public var radius: Float
        /// Blend amount from 0 to 1, scaling both the occlusion and the normal's bend. Zero
        /// leaves color unchanged, visibility at one and the normal unbent.
        public var strength: Float
        /// Slice directions per pixel, from 1 to 8. The filter combines 16 rotations of them.
        public var sliceCount: Int
        /// Depth samples on each side of a slice, from 2 to 32.
        public var stepCount: Int
        /// The outer fraction of the radius, from 0 to 1, over which occluders fade out.
        /// Zero counts occluders exactly within the radius.
        public var falloff: Float
        /// The largest screen-space search radius in pixels, from 2 to 512.
        public var maximumScreenRadius: Float
        /// Whether to apply the 4×4 depth- and normal-aware filter.
        public var denoise: Bool
        /// Optional maximum view-axis separation of a horizon sample, in world units.
        public var thickness: Float?
        /// Ignore samples this close to the receiver's tangent plane, in world units.
        public var normalBias: Float
        /// Exponent on integrated visibility, before strength and spatial filtering.
        public var visibilityExponent: Float
        /// Fraction of each output dimension used for the horizon gather, from 0.25 to 1.
        public var resolutionScale: Float
        /// Optional fixed radius in full-size pixels, from 2 to 512, bounded by
        /// maximumScreenRadius. When supplied this replaces the projected world radius.
        public var screenSpaceRadius: Float?
        /// Selects the legacy Poisson filter when denoise is enabled.
        /// Nil uses the standard depth- and normal-aware filter.
        public var poisson: PoissonSettings?
        /// Distribution of horizon taps between the receiver and the search radius.
        /// Must be finite and positive. One spaces taps linearly and two favors the receiver.
        public var sampleDistanceExponent: Float
        /// Occlusion has full strength inside this box and fades over one search radius outside.
        /// The box is in world space when a view pose is supplied, and in view space otherwise.
        public var sceneClipBox: AxisAlignedBox?

        public init(
            radius: Float = 0.35, strength: Float = 0.6, sliceCount: Int = 3, stepCount: Int = 8,
            falloff: Float = 0.6, maximumScreenRadius: Float = 128, denoise: Bool = true,
            thickness: Float? = nil, normalBias: Float = 0, visibilityExponent: Float = 1,
            resolutionScale: Float = 1, screenSpaceRadius: Float? = nil, poisson: PoissonSettings? = nil,
            sampleDistanceExponent: Float = 2, sceneClipBox: AxisAlignedBox? = nil
        ) {
            self.radius = radius
            self.strength = strength
            self.sliceCount = sliceCount
            self.stepCount = stepCount
            self.falloff = falloff
            self.maximumScreenRadius = maximumScreenRadius
            self.denoise = denoise
            self.thickness = thickness
            self.normalBias = normalBias
            self.visibilityExponent = visibilityExponent
            self.resolutionScale = resolutionScale
            self.screenSpaceRadius = screenSpaceRadius
            self.poisson = poisson
            self.sampleDistanceExponent = sampleDistanceExponent
            self.sceneClipBox = sceneClipBox
        }

        var isValid: Bool {
            radius.isFinite && radius > 0 && (0...1).contains(strength) && (1...8).contains(sliceCount)
                && (2...32).contains(stepCount) && (0...1).contains(falloff)
                && maximumScreenRadius.isFinite && (2...512).contains(maximumScreenRadius)
                && (thickness.map { $0.isFinite && $0 > 0 } ?? true)
                && normalBias.isFinite && normalBias >= 0
                && visibilityExponent.isFinite && visibilityExponent > 0
                && (0.25...1).contains(resolutionScale)
                && (screenSpaceRadius.map { (2...512).contains($0) } ?? true)
                && (poisson?.isValid ?? true)
                && sampleDistanceExponent.isFinite && sampleDistanceExponent > 0
        }

        fileprivate var activePoisson: PoissonSettings? { denoise ? poisson : nil }

        fileprivate var usesControls: Bool {
            thickness != nil || normalBias > 0 || visibilityExponent != 1 || screenSpaceRadius != nil
        }

        fileprivate func traceSize(width: Int, height: Int) -> SIMD2<Int> {
            SIMD2(max(1, Int((Float(width) * resolutionScale).rounded())),
                  max(1, Int((Float(height) * resolutionScale).rounded())))
        }
    }

    /// Optional sample phasing and caller-owned history for a visibility encode.
    public struct TemporalFrame {
        public struct Previous {
            public let visibility: any MTLTexture
            public let depth: any MTLTexture
            public let surfaceAttributes: (any MTLTexture)?
            public let camera: ShaderGraphCamera
            public let depthMode: MeshDepthMode
            public let settings: Settings?

            /// Supply the stored outputs of the same prior raster view.
            /// Include its settings when the prior visibility used a scene clip box.
            public init(
                visibility: any MTLTexture, depth: any MTLTexture, surfaceAttributes: (any MTLTexture)? = nil,
                camera: ShaderGraphCamera, depthMode: MeshDepthMode = .forward, settings: Settings? = nil
            ) {
                self.visibility = visibility
                self.depth = depth
                self.surfaceAttributes = surfaceAttributes
                self.camera = camera
                self.depthMode = depthMode
                self.settings = settings
            }
        }

        public let index: UInt64
        public let previous: Previous?
        public let historyWeight: Float
        /// Optional full-size motion for the current view and the supplied prior view.
        /// See `RasterMotionVectors` for the RG pixel offset, B previous raw depth and A validity.
        public let motionTexture: (any MTLTexture)?

        /// Advance the index each rendered frame. Omit previous history after a cut.
        /// With motion, invalid vectors leave the current visibility unchanged.
        /// Omit motion to reproject stationary geometry from the camera matrices.
        public init(
            index: UInt64, previous: Previous? = nil, historyWeight: Float = 0.85,
            motionTexture: (any MTLTexture)? = nil
        ) {
            self.index = index
            self.previous = previous
            self.historyWeight = historyWeight
            self.motionTexture = motionTexture
        }

        fileprivate var phase: SIMD2<Float> {
            SIMD2([60, 300, 180, 240, 120, 0][Int(index % 6)] / 360.0, [0, 0.5, 0.25, 0.75][Int(index % 4)])
        }
    }

    private let device: any MTLDevice
    private let library: any MTLLibrary
    private let vertex: any MTLFunction
    private var spatialPipelines: [SpatialVariant: any MTLRenderPipelineState] = [:]
    private var temporalPipelines: [TemporalVariant: any MTLRenderPipelineState] = [:]
    private let unusedAttributes: any MTLTexture
    private var intermediate: (any MTLTexture)?
    private var spatial: (any MTLTexture)?
    private weak var intermediateQueue: (any MTLCommandQueue)?

    var horizonTextureSize: SIMD2<Int>? { intermediate.map { SIMD2($0.width, $0.height) } }

    public init(device: any MTLDevice) throws {
        self.device = device
        var defaults: [SpatialVariant: any MTLRenderPipelineState] = [:]
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let vertex = library.makeFunction(name: "frostfield_ao_vertex") else { throw Failure.missingShader }
            self.library = library
            self.vertex = vertex
            for (name, label) in [
                ("frostfield_gtao_fragment", "Frostfield GTAO horizons"),
                ("frostfield_gtao_visibility", "Frostfield GTAO visibility"),
                ("frostfield_gtao_apply", "Frostfield ambient occlusion"),
            ] {
                let fragment = try Self.projectedFragment(name, library: library, projection: .perspective)
                let key = SpatialVariant(function: name, projection: .perspective, usesExtraControl: false)
                defaults[key] = try Self.prepare(device, vertex: vertex, fragment: fragment, label: label)
            }
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: SceneGraph.surfaceAttributesPixelFormat, width: 1, height: 1, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = .shaderRead
        guard let unused = device.makeTexture(descriptor: descriptor) else { throw Failure.resource }
        unused.label = "Frostfield unused surface attributes"
        unusedAttributes = unused
        spatialPipelines = defaults
    }

    private static func prepare(
        _ device: any MTLDevice, vertex: any MTLFunction, fragment: any MTLFunction, label: String
    ) throws -> any MTLRenderPipelineState {
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = label
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        descriptor.colorAttachments[0].pixelFormat = .rgba16Float
        do { return try device.makeRenderPipelineState(descriptor: descriptor) }
        catch { throw Failure.pipeline(error.localizedDescription) }
    }

    private enum ProjectionKind: Hashable {
        case perspective, orthographic, logarithmic
    }

    private struct SpatialVariant: Hashable {
        let function: String
        let projection: ProjectionKind
        let usesExtraControl: Bool
    }

    private struct TemporalVariant: Hashable {
        let usesMotion: Bool
        let usesLogarithmicDepth: Bool
        let usesClipBox: Bool
    }

    private func spatialPipeline(
        _ name: String, projection: ProjectionKind, usesExtraControl: Bool
    ) throws -> any MTLRenderPipelineState {
        let variant = SpatialVariant(function: name, projection: projection, usesExtraControl: usesExtraControl)
        if let cached = spatialPipelines[variant] { return cached }
        let fragment = try Self.projectedFragment(
            name, library: library, projection: projection, usesExtraControl: usesExtraControl
        )
        let pipeline = try Self.prepare(device, vertex: vertex, fragment: fragment, label: name)
        spatialPipelines[variant] = pipeline
        return pipeline
    }

    private func resolvePipeline(
        color: Bool, reduced: Bool, projection: ProjectionKind, poisson: Bool, usesClipBox: Bool
    ) throws -> any MTLRenderPipelineState {
        let name = color ? "frostfield_gtao_apply" : "frostfield_gtao_visibility"
        let suffix = poisson ? "_poisson" : reduced ? "_reduced" : ""
        return try spatialPipeline(name + suffix, projection: projection, usesExtraControl: usesClipBox)
    }

    private static func projectedFragment(
        _ name: String, library: any MTLLibrary, projection: ProjectionKind, usesExtraControl: Bool = false
    ) throws -> any MTLFunction {
        let constants = MTLFunctionConstantValues()
        var orthographic = projection == .orthographic
        var logarithmic = projection == .logarithmic
        var extraControl = usesExtraControl
        constants.setConstantValue(&orthographic, type: .bool, index: 0)
        constants.setConstantValue(&logarithmic, type: .bool, index: 2)
        constants.setConstantValue(&extraControl, type: .bool, index: 4)
        do { return try library.makeFunction(name: name, constantValues: constants) }
        catch { throw Failure.pipeline(error.localizedDescription) }
    }

    private struct Projection {
        let scale: SIMD2<Float>
        let near: Float
        let kind: ProjectionKind
        var orthographic: Bool { kind == .orthographic }
        let position: SIMD4<Float>
        let depth: SIMD2<Float>
        let logarithmicFar: Float

        init(_ perspective: ScreenSpaceProjection, logarithmic: LogarithmicRasterDepth? = nil) {
            scale = perspective.scale
            near = perspective.near
            position = SIMD4(1 / scale.x, 1 / scale.y, perspective.center.x, perspective.center.y)
            if let logarithmic {
                kind = .logarithmic
                depth = SIMD2(logarithmic.range.z, logarithmic.range.y)
                logarithmicFar = logarithmic.range.w
            } else {
                kind = .perspective
                depth = SIMD2(perspective.farInverseDistance, perspective.inverseDistanceRange)
                logarithmicFar = 0
            }
        }

        init?(camera: PerspectiveCamera, aspectRatio: Float, depthMode: MeshDepthMode) {
            guard let perspective = ScreenSpaceProjection(camera: camera, aspectRatio: aspectRatio) else { return nil }
            if depthMode == .logarithmic {
                guard let matrix = try? camera.projection(aspectRatio: aspectRatio),
                      let logarithmic = try? LogarithmicRasterDepth(projection: matrix, near: camera.near, far: camera.far)
                else { return nil }
                self.init(perspective, logarithmic: logarithmic)
            } else {
                self.init(perspective)
            }
        }

        init?(camera: ShaderGraphCamera, depthMode: MeshDepthMode) {
            let matrix = camera.projectionMatrix
            if depthMode == .logarithmic {
                guard let logarithmic = try? LogarithmicRasterDepth(projection: matrix, near: camera.near, far: camera.far)
                else { return nil }
                if logarithmic.range.w > 0 {
                    guard let perspective = ScreenSpaceProjection(clipFromView: matrix, depthMode: .forward) else { return nil }
                    self.init(perspective, logarithmic: logarithmic)
                } else {
                    self.init(orthographic: matrix, depthMode: .forward)
                }
                return
            }
            self.init(clipFromView: matrix, depthMode: depthMode)
        }

        init?(clipFromView matrix: simd_float4x4, depthMode: MeshDepthMode) {
            if depthMode != .logarithmic,
               let perspective = ScreenSpaceProjection(clipFromView: matrix, depthMode: depthMode) {
                self.init(perspective)
                return
            }
            self.init(orthographic: matrix, depthMode: depthMode)
        }

        private init?(orthographic matrix: simd_float4x4, depthMode: MeshDepthMode) {
            let (x, y, z, w) = matrix.columns
            guard [x, y, z, w].allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite }),
                  x.y == 0, x.z == 0, x.w == 0, y.x == 0, y.z == 0, y.w == 0,
                  z.x == 0, z.y == 0, z.w == 0, w.w > 0 else { return nil }
            let clipW = Double(w.w)
            let scales = SIMD2(Float(Double(x.x) / clipW), Float(Double(y.y) / clipW))
            let slope = Double(z.z) / clipW
            let offset = Double(w.z) / clipW
            let reversed = depthMode == .reversed
            guard reversed ? slope > 0 : slope < 0 else { return nil }
            let near = (offset - (reversed ? 1 : 0)) / slope
            let span = 1 / abs(slope)
            let center = SIMD2(Float(-Double(w.x) / clipW), Float(-Double(w.y) / clipW))
            guard scales.x.isNormal, scales.x > 0, scales.y.isNormal, scales.y > 0,
                  (1 / scales.x).isFinite, (1 / scales.y).isFinite,
                  center.x.isFinite, center.y.isFinite, near >= 0, Float(near + span).isFinite,
                  Float(span).isNormal else { return nil }
            scale = scales
            self.near = Float(near)
            kind = .orthographic
            position = SIMD4(1 / scales.x, 1 / scales.y, center.x, center.y)
            depth = SIMD2(Float(near), Float(span))
            logarithmicFar = 0
        }

        func constants(extra: Float) -> [SIMD4<Float>] { [position, SIMD4(depth.x, depth.y, near, extra)] }
    }

    /// Encodes after opaque scene color and depth are stored. The source is a
    /// standalone, shader-readable, tracked RGBA16Float texture. The depth input
    /// is shader-readable, tracked Depth32Float using the supplied camera and
    /// depth mode. Use single-sample textures at the same size and a separate,
    /// non-aliased RGBA16Float render target. Synchronize any untracked destination.
    ///
    /// Normals are reconstructed from depth, and cleared depth is never occluded. The
    /// visibility multiplies scene RGB, including direct light and emission, while alpha
    /// is preserved. To occlude only indirect light, use `encodeVisibility` instead.
    public func encode(
        source: any MTLTexture,
        depth: any MTLTexture,
        destination: any MTLTexture,
        camera: PerspectiveCamera,
        depthMode: MeshDepthMode = .forward,
        settings: Settings = Settings(),
        commandBuffer: any MTLCommandBuffer
    ) throws {
        let projection = Projection(camera: camera, aspectRatio: Float(depth.width) / Float(depth.height), depthMode: depthMode)
        try encodeColor(
            source: source, depth: depth, destination: destination, projection: projection,
            depthMode: depthMode, settings: settings, commandBuffer: commandBuffer
        )
    }

    /// Applies GTAO to color with a caller-supplied Metal projection.
    /// Supply an axis-aligned orthographic or perspective projection matching
    /// the source depth, including any lens shift or jitter. Use the
    /// `ShaderGraphCamera` overload for logarithmic perspective depth.
    public func encode(
        source: any MTLTexture, depth: any MTLTexture, destination: any MTLTexture,
        clipFromView: simd_float4x4, depthMode: MeshDepthMode = .forward,
        settings: Settings = Settings(), commandBuffer: any MTLCommandBuffer
    ) throws {
        try encodeColor(
            source: source, depth: depth, destination: destination,
            projection: Projection(clipFromView: clipFromView, depthMode: depthMode),
            depthMode: depthMode, settings: settings, commandBuffer: commandBuffer
        )
    }

    /// Applies GTAO to color using the authored clip planes and projection.
    /// Supply a forward clip matrix when the perspective attachment uses logarithmic depth.
    public func encode(
        source: any MTLTexture, depth: any MTLTexture, destination: any MTLTexture,
        camera: ShaderGraphCamera, depthMode: MeshDepthMode = .forward,
        settings: Settings = Settings(), commandBuffer: any MTLCommandBuffer
    ) throws {
        try encodeColor(
            source: source, depth: depth, destination: destination,
            projection: Projection(camera: camera, depthMode: depthMode),
            worldFromView: settings.sceneClipBox == nil ? matrix_identity_float4x4 : simd_inverse(camera.viewMatrix),
            depthMode: depthMode, settings: settings, commandBuffer: commandBuffer
        )
    }

    private func encodeColor(
        source: any MTLTexture, depth: any MTLTexture, destination: any MTLTexture, projection: Projection?,
        worldFromView: simd_float4x4 = matrix_identity_float4x4,
        depthMode: MeshDepthMode, settings: Settings, commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(settings: settings, commandBuffer: commandBuffer)
        guard source.device === device, depth.device === device, destination.device === device else {
            throw Failure.wrongDevice
        }
        guard source !== destination, Self.isReadable(source, format: .rgba16Float),
              source.parent == nil, source.buffer == nil, source.iosurface == nil, source.heap == nil,
              source.width == depth.width, source.height == depth.height else {
            throw Failure.incompatibleTextures
        }
        try validate(depth: depth, attributes: nil, destination: destination)
        guard let projection, let rotation = ScreenSpaceProjection.viewFromWorld(worldFromView) else {
            throw Failure.cameraRange
        }
        let trace = settings.traceSize(width: depth.width, height: depth.height)
        let reduced = trace != SIMD2(depth.width, depth.height)
        let view = try viewConstants(
            projection: projection, rotation: rotation, width: depth.width, height: depth.height,
            depthMode: depthMode, settings: settings, hasAttributes: false, traceSize: trace, worldFromView: worldFromView
        )
        let raw = try horizonsTarget(width: trace.x, height: trace.y, queue: commandBuffer.commandQueue)
        let pipeline = try resolvePipeline(
            color: true, reduced: reduced, projection: projection.kind,
            poisson: settings.activePoisson != nil, usesClipBox: settings.sceneClipBox != nil
        )
        try encodeHorizons(depth: depth, attributes: nil, raw: raw, view: view,
                           controlled: settings.usesControls || reduced, projection: projection.kind,
                           customSampleSpacing: settings.sampleDistanceExponent != 2, commandBuffer: commandBuffer)
        try encodeResolve(
            pipeline: pipeline, raw: raw, depth: depth, attributes: nil, color: source,
            destination: destination, view: view, settings: settings, commandBuffer: commandBuffer
        )
    }

    /// Encodes per-pixel world-space bends (RGB) and visibility (A) into an RGBA16Float
    /// render target, for `SceneGraph.encode(...screenSpaceOcclusion:)`.
    ///
    /// Encode it after `SceneGraph.encodeSurfaceAttributes` and before the color pass,
    /// passing that prepass's depth and attributes. Without attributes the normals are
    /// reconstructed from depth, with one-sided differences at the screen border. The bend
    /// is `strength · (bent normal − measured normal)`; shading adds it to each fragment's
    /// own shading normal, so open surfaces, where it is zero, keep their normal maps.
    /// Visibility is `1 − strength · occlusion`. Pixels with cleared depth, or without a
    /// usable normal, read zero bend and visibility one. All textures must share one size;
    /// depth and attributes must be shader-readable and tracked. Supply an optional
    /// temporal frame to vary the sample pattern and reproject prior visibility.
    /// The history and any motion texture must remain available until the command
    /// buffer completes. History with logarithmic depth uses camera reprojection.
    /// Changing the scene clip box discards history from the previous box.
    public func encodeVisibility(
        depth: any MTLTexture,
        surfaceAttributes: (any MTLTexture)? = nil,
        destination: any MTLTexture,
        camera: PerspectiveCamera,
        cameraTransform: SceneTransform,
        depthMode: MeshDepthMode = .forward,
        settings: Settings = Settings(),
        temporal: TemporalFrame? = nil,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(settings: settings, commandBuffer: commandBuffer)
        guard depth.device === device, destination.device === device,
              surfaceAttributes.map({ $0.device === device }) ?? true else { throw Failure.wrongDevice }
        try validate(depth: depth, attributes: surfaceAttributes, destination: destination)
        let rotation = simd_matrix3x3(cameraTransform.rotation.normalized)
        guard Self.isFinite(rotation.columns.0), Self.isFinite(rotation.columns.1), Self.isFinite(rotation.columns.2),
              let projection = Projection(camera: camera, aspectRatio: Float(depth.width) / Float(depth.height), depthMode: depthMode)
        else {
            throw Failure.cameraRange
        }
        try encodeVisibility(
            depth: depth, surfaceAttributes: surfaceAttributes, destination: destination, projection: projection,
            viewFromWorld: rotation.transpose, worldFromView: cameraTransform.matrix, depthMode: depthMode,
            settings: settings, temporal: temporal, commandBuffer: commandBuffer
        )
    }

    /// Encodes visibility for a caller-supplied view, such as one eye of a visionOS
    /// compositor drawable, whose projection may be off-center.
    ///
    /// `clipFromView` maps right-handed view space, looking down −Z, to Metal clip space
    /// with the projection depth mapping that `depthMode` names. Axis-aligned
    /// perspective and orthographic projections accept lens shifts, crops and
    /// jitter. Use the `ShaderGraphCamera` overload for logarithmic perspective depth.
    /// `worldFromView` is the rigid view pose. Its translation is used for
    /// temporal reprojection. Otherwise this matches the camera overload.
    public func encodeVisibility(
        depth: any MTLTexture,
        surfaceAttributes: (any MTLTexture)? = nil,
        destination: any MTLTexture,
        clipFromView: simd_float4x4,
        worldFromView: simd_float4x4,
        depthMode: MeshDepthMode = .forward,
        settings: Settings = Settings(),
        temporal: TemporalFrame? = nil,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try encodeExplicitVisibility(
            depth: depth, surfaceAttributes: surfaceAttributes, destination: destination,
            projection: Projection(clipFromView: clipFromView, depthMode: depthMode), worldFromView: worldFromView,
            depthMode: depthMode, settings: settings, temporal: temporal, commandBuffer: commandBuffer
        )
    }

    /// Encodes visibility using the authored camera pose, projection and clip planes.
    /// Logarithmic perspective depth uses the camera's forward projection and finite clip planes.
    public func encodeVisibility(
        depth: any MTLTexture, surfaceAttributes: (any MTLTexture)? = nil, destination: any MTLTexture,
        camera: ShaderGraphCamera, depthMode: MeshDepthMode = .forward, settings: Settings = Settings(),
        temporal: TemporalFrame? = nil, commandBuffer: any MTLCommandBuffer
    ) throws {
        try encodeExplicitVisibility(
            depth: depth, surfaceAttributes: surfaceAttributes, destination: destination,
            projection: Projection(camera: camera, depthMode: depthMode), worldFromView: simd_inverse(camera.viewMatrix),
            depthMode: depthMode, settings: settings, temporal: temporal, commandBuffer: commandBuffer
        )
    }

    private func encodeExplicitVisibility(
        depth: any MTLTexture, surfaceAttributes: (any MTLTexture)?, destination: any MTLTexture,
        projection: Projection?, worldFromView: simd_float4x4, depthMode: MeshDepthMode, settings: Settings,
        temporal: TemporalFrame?, commandBuffer: any MTLCommandBuffer
    ) throws {
        try validate(settings: settings, commandBuffer: commandBuffer)
        guard depth.device === device, destination.device === device,
              surfaceAttributes.map({ $0.device === device }) ?? true else { throw Failure.wrongDevice }
        try validate(depth: depth, attributes: surfaceAttributes, destination: destination)
        guard let projection, let rotation = ScreenSpaceProjection.viewFromWorld(worldFromView) else {
            throw Failure.cameraRange
        }
        try encodeVisibility(
            depth: depth, surfaceAttributes: surfaceAttributes, destination: destination, projection: projection,
            viewFromWorld: rotation, worldFromView: worldFromView, depthMode: depthMode,
            settings: settings, temporal: temporal, commandBuffer: commandBuffer
        )
    }

    private func encodeVisibility(
        depth: any MTLTexture, surfaceAttributes: (any MTLTexture)?, destination: any MTLTexture,
        projection: Projection, viewFromWorld: simd_float3x3, worldFromView: simd_float4x4, depthMode: MeshDepthMode,
        settings: Settings, temporal: TemporalFrame?, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard temporal.map({ (0...0.95).contains($0.historyWeight) }) ?? true else { throw Failure.history }
        let motion = temporal?.motionTexture
        if let motion {
            guard motion.device === device else { throw Failure.wrongDevice }
            guard depthMode != .logarithmic, temporal?.previous?.depthMode != .logarithmic,
                  Self.isReadable(motion, format: RasterMotionVectors.pixelFormat),
                  motion.width == depth.width, motion.height == depth.height,
                  !motion.isAliasable(), !TextureTransfer.sharesStorage(motion, destination)
            else { throw Failure.history }
        }
        let history = try temporal.flatMap { frame -> PreparedHistory? in
            guard let previous = frame.previous, previous.settings?.sceneClipBox == settings.sceneClipBox else { return nil }
            return try prepareHistory(previous, weight: frame.historyWeight, worldFromView: worldFromView,
                                      size: SIMD2(depth.width, depth.height), destination: destination, settings: settings)
        }
        let trace = settings.traceSize(width: depth.width, height: depth.height)
        let reduced = trace != SIMD2(depth.width, depth.height)
        let view = try viewConstants(
            projection: projection, rotation: viewFromWorld, width: depth.width, height: depth.height,
            depthMode: depthMode, settings: settings, hasAttributes: surfaceAttributes != nil,
            phase: temporal?.phase ?? .zero, traceSize: trace, worldFromView: worldFromView
        )
        let raw = try horizonsTarget(width: trace.x, height: trace.y, queue: commandBuffer.commandQueue)
        let pipeline = try resolvePipeline(
            color: false, reduced: reduced, projection: projection.kind,
            poisson: settings.activePoisson != nil, usesClipBox: settings.sceneClipBox != nil
        )
        let output: any MTLTexture
        if history != nil {
            let target = try target(spatial, width: depth.width, height: depth.height, label: "Frostfield spatial GTAO")
            spatial = target
            output = target
        } else {
            output = destination
        }
        try encodeHorizons(depth: depth, attributes: surfaceAttributes, raw: raw, view: view,
                           controlled: settings.usesControls || temporal != nil || reduced,
                           projection: projection.kind, customSampleSpacing: settings.sampleDistanceExponent != 2,
                           commandBuffer: commandBuffer)
        try encodeResolve(
            pipeline: pipeline, raw: raw, depth: depth, attributes: surfaceAttributes, color: nil,
            destination: output, view: view, settings: settings, commandBuffer: commandBuffer
        )
        if let history {
            try encodeTemporal(
                history, current: output, depth: depth, attributes: surfaceAttributes, motion: motion,
                view: view, usesLogarithmicDepth: projection.kind == .logarithmic || history.usesLogarithmicDepth,
                usesClipBox: settings.sceneClipBox != nil,
                destination: destination, commandBuffer: commandBuffer
            )
        }
    }

    private struct PreparedHistory {
        let previous: TemporalFrame.Previous
        let view: [SIMD4<Float>]
        let clipFromCurrent: simd_float4x4
        let options: SIMD4<Float>
        let usesLogarithmicDepth: Bool
    }

    private func prepareHistory(
        _ previous: TemporalFrame.Previous, weight: Float, worldFromView: simd_float4x4,
        size: SIMD2<Int>, destination: any MTLTexture, settings: Settings
    ) throws -> PreparedHistory {
        var textures = [previous.visibility, previous.depth]
        if let attributes = previous.surfaceAttributes { textures.append(attributes) }
        guard textures.allSatisfy({ $0.device === device }) else { throw Failure.wrongDevice }
        guard Self.isReadable(previous.visibility, format: .rgba16Float), Self.isReadable(previous.depth, format: .depth32Float),
              previous.surfaceAttributes.map({ Self.isReadable($0, format: SceneGraph.surfaceAttributesPixelFormat) }) ?? true,
              textures.allSatisfy({
                  $0.width == size.x && $0.height == size.y && !$0.isAliasable()
                      && !TextureTransfer.sharesStorage($0, destination)
              }),
              let projection = Projection(camera: previous.camera, depthMode: previous.depthMode)
        else { throw Failure.history }

        let priorView = previous.camera.viewMatrix
        let priorWorld = simd_inverse(priorView)
        let priorSettings = previous.settings ?? settings
        guard priorSettings.isValid else { throw Failure.history }
        guard Self.isRigid(priorView), Self.isRigid(worldFromView),
              let rotation = ScreenSpaceProjection.viewFromWorld(priorWorld) else { throw Failure.history }
        func widen(_ matrix: simd_float4x4, normalizeAffine: Bool) -> simd_double4x4 {
            let scale = normalizeAffine ? 1 / Double(matrix[3].w) : 1
            return simd_double4x4(
                SIMD4<Double>(matrix[0]) * scale, SIMD4<Double>(matrix[1]) * scale,
                SIMD4<Double>(matrix[2]) * scale, SIMD4<Double>(matrix[3]) * scale
            )
        }
        let relative = priorView == simd_inverse(worldFromView) ? matrix_identity_double4x4
            : widen(priorView, normalizeAffine: true) * widen(worldFromView, normalizeAffine: true)
        let clip = widen(previous.camera.projectionMatrix, normalizeAffine: false) * relative
        let priorConstants = try viewConstants(
            projection: projection, rotation: rotation, width: size.x, height: size.y,
            depthMode: previous.depthMode, settings: priorSettings, hasAttributes: previous.surfaceAttributes != nil,
            worldFromView: priorWorld
        )
        return PreparedHistory(
            previous: previous, view: priorConstants,
            clipFromCurrent: simd_float4x4(SIMD4(clip[0]), SIMD4(clip[1]), SIMD4(clip[2]), SIMD4(clip[3])),
            options: SIMD4(projection.orthographic ? 0 : -1 / previous.camera.projectionMatrix[2].w, weight, 0, 0),
            usesLogarithmicDepth: projection.kind == .logarithmic
        )
    }

    private static func isRigid(_ matrix: simd_float4x4) -> Bool {
        matrix[0].w == 0 && matrix[1].w == 0 && matrix[2].w == 0 && abs(matrix[3].w - 1) < 1e-5
            && isFinite(SIMD3(matrix[3].x, matrix[3].y, matrix[3].z))
            && ScreenSpaceProjection.viewFromWorld(matrix) != nil
    }

    private func encodeTemporal(
        _ history: PreparedHistory, current: any MTLTexture, depth: any MTLTexture, attributes: (any MTLTexture)?,
        motion: (any MTLTexture)?, view: [SIMD4<Float>], usesLogarithmicDepth: Bool,
        usesClipBox: Bool, destination: any MTLTexture, commandBuffer: any MTLCommandBuffer
    ) throws {
        let pipeline: any MTLRenderPipelineState
        var usesMotion = motion != nil
        var logarithmic = usesLogarithmicDepth
        var clips = usesClipBox
        let variant = TemporalVariant(usesMotion: usesMotion, usesLogarithmicDepth: logarithmic, usesClipBox: clips)
        if let cached = temporalPipelines[variant] { pipeline = cached }
        else {
            let constants = MTLFunctionConstantValues()
            constants.setConstantValue(&usesMotion, type: .bool, index: 1)
            constants.setConstantValue(&logarithmic, type: .bool, index: 2)
            constants.setConstantValue(&clips, type: .bool, index: 3)
            let fragment = try library.makeFunction(name: "frostfield_gtao_temporal", constantValues: constants)
            pipeline = try Self.prepare(device, vertex: vertex, fragment: fragment, label: "Frostfield temporal GTAO")
            temporalPipelines[variant] = pipeline
        }
        do {
            try PostProcessShaders.drawFullScreen(
                into: destination, label: "Frostfield temporal GTAO", commandBuffer: commandBuffer, pipeline: pipeline
            ) { encoder in
                encoder.setFragmentTexture(current, index: 0)
                encoder.setFragmentTexture(depth, index: 1)
                encoder.setFragmentTexture(attributes ?? unusedAttributes, index: 2)
                encoder.setFragmentTexture(history.previous.visibility, index: 3)
                encoder.setFragmentTexture(history.previous.depth, index: 4)
                encoder.setFragmentTexture(history.previous.surfaceAttributes ?? unusedAttributes, index: 5)
                if let motion { encoder.setFragmentTexture(motion, index: 6) }
                view.withUnsafeBytes { encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: 0) }
                history.view.withUnsafeBytes { encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: 1) }
                var clip = history.clipFromCurrent
                var options = history.options
                encoder.setFragmentBytes(&clip, length: MemoryLayout<simd_float4x4>.stride, index: 2)
                encoder.setFragmentBytes(&options, length: MemoryLayout<SIMD4<Float>>.stride, index: 3)
            }
        } catch { throw Failure.encoder }
    }

    /// The float4 slots of FrostfieldOcclusionView in AmbientOcclusion.metal.
    private func viewConstants(
        projection: Projection, rotation: simd_float3x3, width: Int, height: Int,
        depthMode: MeshDepthMode, settings: Settings, hasAttributes: Bool,
        phase: SIMD2<Float> = .zero, traceSize: SIMD2<Int>? = nil,
        worldFromView: simd_float4x4 = matrix_identity_float4x4
    ) throws -> [SIMD4<Float>] {
        let focalLengthPixels = 0.5 * Float(height) * projection.scale.y
        // Horizontal over vertical focal length in pixels; one for square pixels.
        let focalRatio = 0.5 * Float(width) * projection.scale.x / focalLengthPixels
        guard focalLengthPixels.isNormal, focalRatio.isNormal,
              settings.sceneClipBox == nil || Self.isRigid(worldFromView),
              (settings.radius * focalLengthPixels / (projection.orthographic ? 1 : projection.near)).isFinite else {
            throw Failure.cameraRange
        }
        return projection.constants(extra: focalRatio) + [
            SIMD4(settings.radius, focalLengthPixels, depthMode == .reversed ? 1 : 0, settings.falloff),
            SIMD4(Float(settings.sliceCount), Float(settings.stepCount), hasAttributes ? 1 : 0, settings.maximumScreenRadius),
            SIMD4(rotation.columns.0, settings.strength),
            SIMD4(rotation.columns.1, projection.orthographic ? 1 : 0),
            SIMD4(rotation.columns.2, projection.logarithmicFar),
            SIMD4(settings.thickness ?? 0, settings.normalBias, settings.visibilityExponent, settings.screenSpaceRadius ?? 0),
            SIMD4(phase.x, phase.y, Float(traceSize?.x ?? width), Float(traceSize?.y ?? height)),
            SIMD4(settings.sceneClipBox?.minimum ?? .zero, settings.sceneClipBox == nil ? 0 : 1),
            SIMD4(settings.sceneClipBox?.maximum ?? .zero, settings.sampleDistanceExponent),
            worldFromView.columns.3,
        ]
    }

    private func encodeHorizons(
        depth: any MTLTexture, attributes: (any MTLTexture)?, raw: any MTLTexture,
        view: [SIMD4<Float>], controlled: Bool, projection: ProjectionKind, customSampleSpacing: Bool,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        let name = controlled ? "frostfield_gtao_controlled" : "frostfield_gtao_fragment"
        let pipeline = try spatialPipeline(name, projection: projection, usesExtraControl: customSampleSpacing)
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = raw
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.encoder }
        encoder.label = "Frostfield GTAO horizons"
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(depth, index: 0)
        encoder.setFragmentTexture(attributes ?? unusedAttributes, index: 1)
        view.withUnsafeBytes { encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: 0) }
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    private struct PoissonUniforms {
        var sampleCount: UInt32
        var sampleIndex: UInt32
        var lumaPhi: Float
        var depthPhi: Float
        var normalPhi: Float

        init(_ settings: PoissonSettings) {
            sampleCount = UInt32(settings.sampleCount)
            sampleIndex = UInt32(settings.sampleIndex)
            lumaPhi = settings.lumaPhi
            depthPhi = settings.depthPhi
            normalPhi = settings.normalPhi
        }
    }

    private func encodeResolve(
        pipeline: any MTLRenderPipelineState, raw: any MTLTexture, depth: any MTLTexture,
        attributes: (any MTLTexture)?, color: (any MTLTexture)?, destination: any MTLTexture,
        view: [SIMD4<Float>], settings: Settings, commandBuffer: any MTLCommandBuffer
    ) throws {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = destination
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.encoder }
        encoder.label = color == nil ? "Frostfield GTAO visibility" : "Frostfield ambient occlusion"
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(raw, index: 0)
        encoder.setFragmentTexture(depth, index: 1)
        encoder.setFragmentTexture(attributes ?? unusedAttributes, index: 2)
        if let color { encoder.setFragmentTexture(color, index: 3) }
        view.withUnsafeBytes { encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: 0) }
        if let poisson = settings.activePoisson {
            var uniforms = PoissonUniforms(poisson)
            encoder.setFragmentBytes(&uniforms, length: MemoryLayout<PoissonUniforms>.stride, index: 1)
            poisson.offsets.withUnsafeBytes { encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: 2) }
        } else {
            var filter = SIMD4<Float>(settings.denoise ? 1 : 0, 0, 0, 0)
            encoder.setFragmentBytes(&filter, length: MemoryLayout<SIMD4<Float>>.size, index: 1)
        }
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    private func horizonsTarget(width: Int, height: Int, queue: any MTLCommandQueue) throws -> any MTLTexture {
        if intermediateQueue !== queue {
            intermediate = nil
            spatial = nil
            intermediateQueue = queue
        }
        let texture = try target(intermediate, width: width, height: height, label: "Frostfield GTAO horizons")
        intermediate = texture
        return texture
    }

    private func target(_ existing: (any MTLTexture)?, width: Int, height: Int, label: String) throws -> any MTLTexture {
        if let existing, existing.width == width, existing.height == height { return existing }
        do {
            return try PostProcessShaders.scratch(
                device: device, width: width, height: height, usage: [.renderTarget, .shaderRead], label: label
            )
        } catch { throw Failure.resource }
    }

    private func validate(settings: Settings, commandBuffer: any MTLCommandBuffer) throws {
        guard settings.isValid else { throw Failure.invalidSettings }
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard commandBuffer.device === device else { throw Failure.wrongDevice }
    }

    private func validate(depth: any MTLTexture, attributes: (any MTLTexture)?, destination: any MTLTexture) throws {
        guard Self.isReadable(depth, format: .depth32Float), depth.width > 0, depth.height > 0,
              destination !== depth, destination.pixelFormat == .rgba16Float,
              destination.textureType == .type2D, destination.sampleCount == 1,
              depth.width == destination.width, depth.height == destination.height,
              destination.usage.contains(.renderTarget), destination.storageMode != .memoryless,
              destination.parent == nil else {
            throw Failure.incompatibleTextures
        }
        if let attributes {
            guard Self.isReadable(attributes, format: SceneGraph.surfaceAttributesPixelFormat),
                  attributes.width == depth.width, attributes.height == depth.height else {
                throw Failure.incompatibleTextures
            }
        }
    }

    private static func isReadable(_ texture: any MTLTexture, format: MTLPixelFormat) -> Bool {
        texture.pixelFormat == format && texture.textureType == .type2D && texture.sampleCount == 1
            && texture.usage.contains(.shaderRead) && !texture.isFramebufferOnly
            && texture.storageMode != .memoryless && texture.hazardTrackingMode != .untracked
    }

    private static func isFinite(_ value: SIMD3<Float>) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite
    }

    public enum Failure: LocalizedError, Equatable {
        case missingShader
        case pipeline(String)
        case invalidSettings
        case commandBuffer
        case wrongDevice
        case incompatibleTextures
        case history
        case cameraRange
        case resource
        case encoder

        public var errorDescription: String? {
            switch self {
            case .missingShader: "Frostfield’s ambient occlusion shaders are missing."
            case .pipeline(let detail): "Could not prepare Frostfield’s ambient occlusion. \(detail)"
            case .invalidSettings:
                "Ambient occlusion needs valid sample counts and finite settings within their documented bounds."
            case .commandBuffer: "Ambient occlusion needs an uncommitted command buffer that retains resources."
            case .wrongDevice: "Ambient occlusion textures and command buffer must use its Metal device."
            case .incompatibleTextures:
                "Use equal-sized, shader-readable RGBA16Float color, Depth32Float depth and RG32Uint surface attributes with a separate RGBA16Float render target."
            case .history:
                "Temporal GTAO needs matching tracked history and optional RGBA16Float motion textures, rigid views and a history weight from 0 to 0.95."
            case .cameraRange:
                "Ambient occlusion needs a supported projection, a rigid view rotation and a representable radius. Logarithmic perspective depth needs the authored finite clip planes."
            case .resource: "Metal could not allocate ambient occlusion textures."
            case .encoder: "Metal could not create the ambient occlusion encoder."
            }
        }
    }
}
