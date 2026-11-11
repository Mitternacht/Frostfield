import Foundation
import Metal
import simd

/// Screen-space reflections for a finished HDR frame, like Three.js `SSRPass` and `SSRNode`.
///
/// For each pixel with a surface, the pass reflects the view ray about the normal stored by
/// `SceneGraph.encodeSurfaceAttributes` and marches it across the screen with
/// perspective-correct depth, one step per pixel at most, then refines the first crossing
/// by bisection. A crossing counts as a hit when the ray lies behind the stored depth by
/// less than `thickness` and reaches the front of the hit surface. The hit pixel's color
/// replaces the environment's reflection: the pass adds `confidence × reflectance ×
/// (hit radiance − environment radiance)`, where the reflectance is the split-sum value
/// the prepass stored (already scaled by material occlusion) and the environment term is
/// subtracted only when one is supplied. Confidence fades with roughness, with the reflected
/// ray's world length over the last 20% of `maximumDistance`, and with distance from the
/// screen edge. Optionally trace at reduced resolution, blur rough receivers,
/// or restrict the effect using a per-pixel receiver mask.
public final class ScreenSpaceReflections {
    /// Previous composited reflections or beauty with its actual raster view.
    /// Preserve both textures until this frame finishes on the Metal queue.
    public struct PreviousFrame {
        public enum Reprojection: Sendable, Equatable {
            /// Require exactly the current view, projection and depth convention.
            case registered
            /// Reproject the current ray hit into the supplied prior perspective camera.
            /// This follows the camera while geometry stays in the same world space.
            case cameraMotion
        }

        public let radiance: any MTLTexture
        public let depth: any MTLTexture
        public let camera: ShaderGraphCamera
        public let depthMode: MeshDepthMode
        public let reprojection: Reprojection

        public init(
            radiance: any MTLTexture, depth: any MTLTexture, camera: ShaderGraphCamera,
            depthMode: MeshDepthMode = .forward, reprojection: Reprojection = .registered
        ) {
            self.radiance = radiance
            self.depth = depth
            self.camera = camera
            self.depthMode = depthMode
            self.reprojection = reprojection
        }
    }

    public struct Settings: Sendable {
        /// The longest reflected ray, in world units. Finite and positive.
        public var maximumDistance: Float
        /// How far behind stored depth, in world units along the view axis, a ray may be and
        /// still hit that surface. Finite and positive.
        public var thickness: Float
        /// Linear march steps along the screen-space ray, from 1 to 512. Rays shorter than
        /// this many pixels take one step per pixel.
        public var stepCount: Int
        /// Bisection steps after the first crossing, from 0 to 16.
        public var refinementCount: Int
        /// Reflections fade out between half this perceptual roughness and this value,
        /// from 0.01 to 1.
        public var maximumRoughness: Float
        /// The fraction of the screen, from 0 to 0.5, over which hits fade toward the edges.
        public var edgeFade: Float
        /// Scales the reflection, from 0 to 1.
        public var intensity: Float
        /// Fraction of each input dimension used for tracing, from 0.25 to 1.
        public var resolutionScale: Float
        /// Maximum spatial blur radius in full-size pixels, from 0 to 32.
        /// The receiver's squared perceptual roughness scales the radius.
        public var roughnessBlurRadius: Float
        /// Selects a deterministic GGX visible-normal ray per trace pixel.
        /// Nil keeps mirror rays. Change the seed between frames when accumulating samples.
        public var stochasticSeed: UInt32?

        public init(
            maximumDistance: Float = 10, thickness: Float = 0.25, stepCount: Int = 96, refinementCount: Int = 6,
            maximumRoughness: Float = 0.6, edgeFade: Float = 0.1, intensity: Float = 1,
            resolutionScale: Float = 1, roughnessBlurRadius: Float = 0, stochasticSeed: UInt32? = nil
        ) {
            self.maximumDistance = maximumDistance
            self.thickness = thickness
            self.stepCount = stepCount
            self.refinementCount = refinementCount
            self.maximumRoughness = maximumRoughness
            self.edgeFade = edgeFade
            self.intensity = intensity
            self.resolutionScale = resolutionScale
            self.roughnessBlurRadius = roughnessBlurRadius
            self.stochasticSeed = stochasticSeed
        }

        var isValid: Bool {
            maximumDistance.isFinite && maximumDistance > 0 && thickness.isFinite && thickness > 0
                && (1...512).contains(stepCount) && (0...16).contains(refinementCount)
                && maximumRoughness.isFinite && (0.01...1).contains(maximumRoughness)
                && edgeFade.isFinite && (0...0.5).contains(edgeFade) && intensity.isFinite && (0...1).contains(intensity)
                && (0.25...1).contains(resolutionScale) && (0...32).contains(roughnessBlurRadius)
        }
    }

    private let device: any MTLDevice
    private let shaders: PostProcessShaders
    private let pipeline: any MTLRenderPipelineState
    private let unusedCube: any MTLTexture
    private let unusedMask: any MTLTexture
    private var trace: (any MTLTexture)?
    private weak var traceQueue: (any MTLCommandQueue)?

    var traceTextureSize: SIMD2<Int>? { trace.map { SIMD2($0.width, $0.height) } }

    public init(device: any MTLDevice) throws {
        self.device = device
        do {
            shaders = try PostProcessShaders(device: device)
            pipeline = try shaders.render("frostfield_ssr_fragment", formats: [.rgba16Float])
        } catch PostProcessEffectError.missingShader(_) {
            throw Failure.missingShader
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
        let descriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: .rgba16Float, size: 1, mipmapped: false)
        descriptor.storageMode = .private
        descriptor.usage = .shaderRead
        guard let cube = device.makeTexture(descriptor: descriptor) else { throw Failure.resource }
        cube.label = "Frostfield unused reflection environment"
        unusedCube = cube
        do {
            unusedMask = try PostProcessShaders.scratch(
                device: device, width: 1, height: 1, pixelFormat: .r8Unorm, usage: .shaderRead, label: "Frostfield unused SSR mask"
            )
        } catch {
            throw Failure.resource
        }
    }

    /// Encodes reflections from `source` into `destination` without submitting the buffer.
    ///
    /// `source` is the shader-readable, tracked RGBA16Float scene color; `depth` the
    /// shader-readable, tracked Depth32Float depth of the same view, using `camera` at
    /// `cameraTransform` with `depthMode`; and `surfaceAttributes` the RG32Uint output of
    /// `SceneGraph.encodeSurfaceAttributes` for that view. The destination is a separate
    /// RGBA16Float render target. All share one size. Pass the materials' `environment`, with
    /// the scene's `environmentIntensity` and `environmentRotation`, so a hit replaces the
    /// image-based reflection that shading already added; without it the reflection adds to
    /// the frame. Alpha is preserved. An optional tracked, shader-readable
    /// R8Unorm, R16Float, or R32Float `receiverMask` scales reflection per pixel.
    /// Masked objects can still appear in reflections. A `previousFrame` supplies
    /// prior radiance only at hits whose depth agrees. Use the prior frame's
    /// actual projection, including jitter and lens offsets. Serialize use of
    /// this pass and order any prior-frame producers before this encode on the GPU.
    public func encode(
        source: any MTLTexture,
        depth: any MTLTexture,
        surfaceAttributes: any MTLTexture,
        receiverMask: (any MTLTexture)? = nil,
        previousFrame: PreviousFrame? = nil,
        destination: any MTLTexture,
        camera: PerspectiveCamera,
        cameraTransform: SceneTransform,
        depthMode: MeshDepthMode = .forward,
        settings: Settings = Settings(),
        environment: PhysicallyBasedMaterial.Environment? = nil,
        environmentIntensity: Float = 1,
        environmentRotation: Float = 0,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try depthMode.requireProjectionDepth()
        try validate(
            source: source, depth: depth, surfaceAttributes: surfaceAttributes, receiverMask: receiverMask,
            previousFrame: previousFrame, destination: destination,
            settings: settings, environment: environment, environmentIntensity: environmentIntensity,
            environmentRotation: environmentRotation, commandBuffer: commandBuffer
        )
        let rotation = simd_matrix3x3(cameraTransform.rotation.normalized).transpose
        guard [rotation.columns.0, rotation.columns.1, rotation.columns.2].allSatisfy({
                  $0.x.isFinite && $0.y.isFinite && $0.z.isFinite
              }),
              let projection = ScreenSpaceProjection(camera: camera, aspectRatio: Float(source.width) / Float(source.height))
        else {
            throw Failure.cameraRange
        }
        let history = try previousFrame.map {
            let matrix = try camera.projection(aspectRatio: Float(source.width) / Float(source.height), depthMode: depthMode)
            return try HistoryProjection($0, projection: matrix, worldFromView: cameraTransform.matrix, depthMode: depthMode)
        }
        try encode(
            source: source, depth: depth, surfaceAttributes: surfaceAttributes, receiverMask: receiverMask,
            previousFrame: previousFrame, history: history, destination: destination,
            projection: projection, viewFromWorld: rotation, depthMode: depthMode, settings: settings,
            environment: environment, environmentIntensity: environmentIntensity,
            environmentRotation: environmentRotation, commandBuffer: commandBuffer
        )
    }

    /// Encodes reflections for a caller-supplied view, such as one eye of a visionOS
    /// compositor drawable, whose projection may be off-center.
    ///
    /// `clipFromView` maps right-handed view space, looking down −Z, to Metal clip space
    /// with the depth mapping that `depthMode` names: a perspective projection whose clip w
    /// is the view distance up to scale, with x depending only on view x and z, and y only
    /// on view y and z. `worldFromView` is the view's rigid transform; only its rotation is
    /// used. Otherwise this matches the camera overload.
    public func encode(
        source: any MTLTexture,
        depth: any MTLTexture,
        surfaceAttributes: any MTLTexture,
        receiverMask: (any MTLTexture)? = nil,
        previousFrame: PreviousFrame? = nil,
        destination: any MTLTexture,
        clipFromView: simd_float4x4,
        worldFromView: simd_float4x4,
        depthMode: MeshDepthMode = .forward,
        settings: Settings = Settings(),
        environment: PhysicallyBasedMaterial.Environment? = nil,
        environmentIntensity: Float = 1,
        environmentRotation: Float = 0,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try depthMode.requireProjectionDepth()
        try validate(
            source: source, depth: depth, surfaceAttributes: surfaceAttributes, receiverMask: receiverMask,
            previousFrame: previousFrame, destination: destination,
            settings: settings, environment: environment, environmentIntensity: environmentIntensity,
            environmentRotation: environmentRotation, commandBuffer: commandBuffer
        )
        guard let projection = ScreenSpaceProjection(clipFromView: clipFromView, depthMode: depthMode),
              let rotation = ScreenSpaceProjection.viewFromWorld(worldFromView) else {
            throw Failure.cameraRange
        }
        let history = try previousFrame.map {
            try HistoryProjection($0, projection: clipFromView, worldFromView: worldFromView, depthMode: depthMode)
        }
        try encode(
            source: source, depth: depth, surfaceAttributes: surfaceAttributes, receiverMask: receiverMask,
            previousFrame: previousFrame, history: history, destination: destination,
            projection: projection, viewFromWorld: rotation, depthMode: depthMode, settings: settings,
            environment: environment, environmentIntensity: environmentIntensity,
            environmentRotation: environmentRotation, commandBuffer: commandBuffer
        )
    }

    private func validate(
        source: any MTLTexture, depth: any MTLTexture, surfaceAttributes: any MTLTexture,
        receiverMask: (any MTLTexture)?, previousFrame: PreviousFrame?, destination: any MTLTexture,
        settings: Settings, environment: PhysicallyBasedMaterial.Environment?, environmentIntensity: Float,
        environmentRotation: Float, commandBuffer: any MTLCommandBuffer
    ) throws {
        guard settings.isValid else { throw Failure.invalidSettings }
        guard environmentIntensity.isFinite, environmentIntensity >= 0, environmentRotation.isFinite else {
            throw Failure.invalidSettings
        }
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard commandBuffer.device === device, source.device === device, depth.device === device,
              surfaceAttributes.device === device, destination.device === device,
              environment.map({ $0.lighting.device === device }) ?? true,
              receiverMask.map({ $0.device === device }) ?? true else {
            throw Failure.wrongDevice
        }
        guard source !== destination, Self.isReadable(source, format: .rgba16Float),
              Self.isReadable(depth, format: .depth32Float),
              Self.isReadable(surfaceAttributes, format: SceneGraph.surfaceAttributesPixelFormat),
              destination.pixelFormat == .rgba16Float, destination.textureType == .type2D,
              destination.sampleCount == 1, destination.usage.contains(.renderTarget),
              destination.storageMode != .memoryless, destination.parent == nil, !destination.isAliasable(),
              destination.hazardTrackingMode != .untracked, !destination.isFramebufferOnly,
              source.parent == nil, source.buffer == nil, source.iosurface == nil, source.heap == nil,
              source.width > 0, source.height > 0,
              [depth, surfaceAttributes, destination].allSatisfy({ $0.width == source.width && $0.height == source.height })
        else {
            throw Failure.incompatibleTextures
        }
        if let receiverMask {
            guard [.r8Unorm, .r16Float, .r32Float].contains(receiverMask.pixelFormat),
                  Self.isReadable(receiverMask, format: receiverMask.pixelFormat), !receiverMask.isAliasable(),
                  receiverMask.width == source.width, receiverMask.height == source.height,
                  !TextureTransfer.sharesStorage(receiverMask, destination) else { throw Failure.receiverMask }
        }
        if let previousFrame {
            let prior = [previousFrame.radiance, previousFrame.depth]
            guard prior.allSatisfy({ $0.device === device }) else { throw Failure.wrongDevice }
            guard Self.isReadable(previousFrame.radiance, format: .rgba16Float),
                  Self.isReadable(previousFrame.depth, format: .depth32Float),
                  prior.allSatisfy({
                      $0.width == source.width && $0.height == source.height
                          && !$0.isAliasable() && !TextureTransfer.sharesStorage($0, destination)
                  }) else { throw Failure.historyTextures }
        }
    }

    private struct HistoryProjection {
        var viewFromCurrent = matrix_identity_float4x4
        var clipFromCurrent = matrix_identity_float4x4
        var depth = SIMD4<Float>.zero

        init() {}

        init(
            _ previous: PreviousFrame, projection: simd_float4x4,
            worldFromView: simd_float4x4, depthMode: MeshDepthMode
        ) throws {
            guard previous.depthMode != .logarithmic,
                  let priorProjection = ScreenSpaceProjection(clipFromView: previous.camera.projectionMatrix, depthMode: previous.depthMode)
            else { throw Failure.historyView }
            let samePose = previous.camera.viewMatrix == simd_inverse(worldFromView)
            switch previous.reprojection {
            case .registered:
                guard samePose, previous.depthMode == depthMode,
                      previous.camera.projectionMatrix == projection else { throw Failure.historyView }
            case .cameraMotion:
                guard Self.isRigid(worldFromView), Self.isRigid(previous.camera.viewMatrix) else { throw Failure.historyView }
                func wide(_ matrix: simd_float4x4, normalizeAffine: Bool = false) -> simd_double4x4 {
                    let scale = normalizeAffine ? 1.0 / Double(matrix[3].w) : 1.0
                    return simd_double4x4(
                        SIMD4<Double>(matrix[0]) * scale, SIMD4<Double>(matrix[1]) * scale,
                        SIMD4<Double>(matrix[2]) * scale, SIMD4<Double>(matrix[3]) * scale
                    )
                }
                func narrow(_ matrix: simd_double4x4) -> simd_float4x4 {
                    simd_float4x4(SIMD4(matrix[0]), SIMD4(matrix[1]), SIMD4(matrix[2]), SIMD4(matrix[3]))
                }
                // Float matrix inversion can leave the homogeneous scale a few ulps off one.
                let relative = samePose ? matrix_identity_double4x4
                    : wide(previous.camera.viewMatrix, normalizeAffine: true) * wide(worldFromView, normalizeAffine: true)
                viewFromCurrent = narrow(relative)
                clipFromCurrent = narrow(wide(previous.camera.projectionMatrix) * relative)
            }
            depth = SIMD4(
                priorProjection.farInverseDistance, priorProjection.inverseDistanceRange,
                previous.depthMode == .reversed ? 1 : 0, previous.reprojection == .registered ? 1 : 2
            )
        }

        private static func isRigid(_ matrix: simd_float4x4) -> Bool {
            matrix[0].w == 0 && matrix[1].w == 0 && matrix[2].w == 0 && abs(matrix[3].w - 1) < 1e-5
                && matrix[3].x.isFinite && matrix[3].y.isFinite && matrix[3].z.isFinite
                && ScreenSpaceProjection.viewFromWorld(matrix) != nil
        }
    }

    private func encode(
        source: any MTLTexture, depth: any MTLTexture, surfaceAttributes: any MTLTexture,
        receiverMask: (any MTLTexture)?, previousFrame: PreviousFrame?, history: HistoryProjection?, destination: any MTLTexture,
        projection: ScreenSpaceProjection, viewFromWorld rotation: simd_float3x3, depthMode: MeshDepthMode,
        settings: Settings, environment: PhysicallyBasedMaterial.Environment?, environmentIntensity: Float,
        environmentRotation: Float, commandBuffer: any MTLCommandBuffer
    ) throws {
        var environmentValues = SIMD4<Float>(1, 0, 0, 0)
        if let environment {
            let yaw = environment.yaw + environmentRotation
            environmentValues = SIMD4(
                cos(yaw), sin(yaw), environment.intensity * environmentIntensity, environment.lighting.maxSpecularLOD
            )
        }
        let width = max(1, Int((Float(source.width) * settings.resolutionScale).rounded()))
        let height = max(1, Int((Float(source.height) * settings.resolutionScale).rounded()))
        let seed = settings.stochasticSeed ?? 0
        let view: [SIMD4<Float>] = projection.constants() + [
            SIMD4(settings.maximumDistance, settings.thickness, Float(settings.stepCount), Float(settings.refinementCount)),
            SIMD4(settings.maximumRoughness, settings.edgeFade, settings.intensity, depthMode == .reversed ? 1 : 0),
            SIMD4(rotation.columns.0, 0),
            SIMD4(rotation.columns.1, 0),
            SIMD4(rotation.columns.2, 0),
            environmentValues,
            SIMD4(environment == nil ? 0 : 1, receiverMask == nil ? 0 : 1, settings.roughnessBlurRadius, settings.stochasticSeed == nil ? 0 : 1),
            SIMD4(Float(width), Float(height), Float(seed & 0xffff), Float(seed >> 16)),
        ]

        var historyProjection = history ?? HistoryProjection()
        func draw(
            _ target: any MTLTexture, pipeline: any MTLRenderPipelineState, samples: (any MTLTexture)? = nil, label: String
        ) throws {
            do {
                try PostProcessShaders.drawFullScreen(into: target, label: label, commandBuffer: commandBuffer, pipeline: pipeline) { encoder in
                    encoder.setFragmentTexture(source, index: 0)
                    encoder.setFragmentTexture(depth, index: 1)
                    encoder.setFragmentTexture(surfaceAttributes, index: 2)
                    encoder.setFragmentTexture(environment?.lighting.specularCube ?? unusedCube, index: 3)
                    encoder.setFragmentTexture(receiverMask ?? unusedMask, index: 4)
                    encoder.setFragmentTexture(samples, index: 5)
                    encoder.setFragmentTexture(previousFrame?.radiance ?? source, index: 6)
                    encoder.setFragmentTexture(previousFrame?.depth ?? depth, index: 7)
                    view.withUnsafeBytes { encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: 0) }
                    encoder.setFragmentBytes(&historyProjection, length: MemoryLayout<HistoryProjection>.stride, index: 1)
                }
            } catch { throw Failure.encoder }
        }
        if width == source.width, height == source.height, settings.roughnessBlurRadius == 0 {
            trace = nil
            try draw(destination, pipeline: pipeline, label: "Frostfield screen-space reflections")
            return
        }
        let gather: any MTLRenderPipelineState
        let resolve: any MTLRenderPipelineState
        do {
            gather = try shaders.render("frostfield_ssr_trace", formats: [.rgba16Float])
            resolve = try shaders.render("frostfield_ssr_resolve", formats: [.rgba16Float])
        } catch PostProcessEffectError.missingShader(_) {
            throw Failure.missingShader
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
        let target: any MTLTexture
        if let trace, trace.width == width, trace.height == height, traceQueue === commandBuffer.commandQueue {
            target = trace
        } else {
            do {
                target = try PostProcessShaders.scratch(
                    device: device, width: width, height: height, usage: [.renderTarget, .shaderRead], label: "Frostfield raw SSR"
                )
            } catch { throw Failure.resource }
            trace = target
            traceQueue = commandBuffer.commandQueue
        }
        try draw(target, pipeline: gather, label: "Frostfield SSR trace")
        try draw(destination, pipeline: resolve, samples: target, label: "Frostfield SSR spatial resolve")
    }

    private static func isReadable(_ texture: any MTLTexture, format: MTLPixelFormat) -> Bool {
        texture.pixelFormat == format && texture.textureType == .type2D && texture.sampleCount == 1
            && texture.usage.contains(.shaderRead) && !texture.isFramebufferOnly
            && texture.storageMode != .memoryless && texture.hazardTrackingMode != .untracked
    }

    public enum Failure: LocalizedError, Equatable {
        case missingShader
        case pipeline(String)
        case invalidSettings
        case commandBuffer
        case wrongDevice
        case incompatibleTextures
        case receiverMask
        case historyTextures
        case historyView
        case cameraRange
        case resource
        case encoder

        public var errorDescription: String? {
            switch self {
            case .missingShader: "Frostfield’s screen-space reflection shaders are missing."
            case .pipeline(let detail): "Could not prepare Frostfield’s screen-space reflections. \(detail)"
            case .invalidSettings:
                "Reflections need a positive finite distance and thickness, 1 to 512 steps, 0 to 16 refinements, a maximum roughness from 0.01 to 1, an edge fade from 0 to 0.5, an intensity from 0 to 1, a resolution scale from 0.25 to 1, a blur radius from 0 to 32 pixels, and a finite nonnegative environment intensity."
            case .commandBuffer: "Reflections need an uncommitted command buffer that retains resources."
            case .wrongDevice: "Reflection textures, environment and command buffer must use its Metal device."
            case .incompatibleTextures:
                "Use equal-sized, shader-readable RGBA16Float color, Depth32Float depth and RG32Uint surface attributes with a separate RGBA16Float render target."
            case .receiverMask: "Use a distinct shader-readable R8Unorm, R16Float, or R32Float mask matching the scene extent."
            case .historyTextures: "Use separate tracked prior HDR color and depth textures matching the current view size."
            case .historyView: "Prior reflection radiance needs matching registered views or rigid perspective cameras for reprojection."
            case .cameraRange:
                "Reflections need a perspective projection with a positive near distance and a rigid view rotation."
            case .resource: "Metal could not allocate a reflection texture."
            case .encoder: "Metal could not create the reflection encoder."
            }
        }
    }
}
