import Foundation
import Metal
import simd
import Synchronization

/// Progressively traces an immutable scene into linear HDR color on one Metal queue.
public final class PathTracer {
    /// A planar emitter whose front faces the cross product of its half axes.
    public struct AreaLight: Equatable, Sendable {
        public enum Shape: Equatable, Sendable {
            case rectangle
            /// A circle or ellipse defined by the two perpendicular radii.
            case disk
        }

        public let center: SIMD3<Float>
        public let halfU: SIMD3<Float>
        public let halfV: SIMD3<Float>
        public let radiance: SIMD3<Float>
        public let shape: Shape
        /// Controls primary-ray visibility. The emitter still lights other surfaces.
        public let cameraVisible: Bool

        public init(
            center: SIMD3<Float>, halfU: SIMD3<Float>, halfV: SIMD3<Float>,
            radiance: SIMD3<Float>, shape: Shape = .rectangle, cameraVisible: Bool = true
        ) throws {
            let uLength2 = simd_length_squared(halfU)
            let vLength2 = simd_length_squared(halfV)
            let crossLength2 = simd_length_squared(simd_cross(halfU, halfV))
            let extent = abs(SIMD3<Double>(center)) + abs(SIMD3<Double>(halfU)) + abs(SIMD3<Double>(halfV))
            guard PathTracer.isFinite(center), uLength2.isNormal, vLength2.isNormal, crossLength2.isNormal,
                  abs(simd_dot(halfU / sqrt(uLength2), halfV / sqrt(vLength2))) < 1e-4,
                  simd_reduce_max(extent) <= Double(Float.greatestFiniteMagnitude),
                  PathTracer.isRadiance(radiance) else { throw Failure.invalidAreaLight }
            self.center = center
            self.halfU = halfU
            self.halfV = halfV
            self.radiance = radiance
            self.shape = shape
            self.cameraVisible = cameraVisible
        }

        /// Aims the front face at a world-space point. Size is the full width
        /// and height in world units, or the two diameters of a disk.
        public init(
            center: SIMD3<Float>, aim: SIMD3<Float>, size: SIMD2<Float>,
            radiance: SIMD3<Float>, shape: Shape = .rectangle, cameraVisible: Bool = true
        ) throws {
            let direction = aim - center
            let length2 = simd_length_squared(direction)
            guard length2.isNormal, size.x.isFinite, size.y.isFinite, size.x > 0, size.y > 0 else {
                throw Failure.invalidAreaLight
            }
            let normal = direction / sqrt(length2)
            let up = abs(normal.y) < 0.999 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(0, 0, 1)
            let right = simd_normalize(simd_cross(up, normal))
            try self.init(
                center: center, halfU: right * (size.x / 2), halfV: simd_cross(normal, right) * (size.y / 2),
                radiance: radiance, shape: shape, cameraVisible: cameraVisible
            )
        }

        /// Sets the total linear RGB power emitted over the front hemisphere.
        /// With dimensions in meters and power in watts, radiance is in W/(m²·sr).
        public init(
            center: SIMD3<Float>, aim: SIMD3<Float>, size: SIMD2<Float>,
            radiantPower: SIMD3<Float>, shape: Shape = .rectangle, cameraVisible: Bool = true
        ) throws {
            guard PathTracer.isFinite(radiantPower), simd_reduce_min(radiantPower) >= 0 else {
                throw Failure.invalidAreaLight
            }
            let area = Double(size.x) * Double(size.y) * (shape == .disk ? Double.pi / 4 : 1)
            let radiance = SIMD3<Float>(SIMD3<Double>(radiantPower) / (.pi * area))
            try self.init(center: center, aim: aim, size: size, radiance: radiance, shape: shape, cameraVisible: cameraVisible)
        }
    }

    /// The emitters and environment for one accumulation. Histories compare the
    /// complete value, so any change requires a restart.
    public struct Lighting: Equatable, Sendable {
        /// Up to 16 rectangular or disk emitters.
        public let lights: [AreaLight]
        /// Up to 16 point, spot and directional lights.
        public let analyticLights: [AnalyticLight]
        /// Radiance for camera and scattered rays that escape the scene.
        public let sky: Environment
        /// The map's full XYZ orientation in radians, including scalar yaw.
        /// Constant skies have no orientation and return zero.
        public let skyEulerRotation: SIMD3<Float>
        /// Whether camera rays that miss the scene show the environment. When false
        /// they add no radiance and no coverage, while surfaces still receive
        /// environment light and scattered rays still see it. The destination alpha
        /// is the fraction of a pixel's samples whose camera ray hit geometry or a
        /// camera-visible area light, and the RGB is premultiplied by it. A camera ray
        /// whose first hit is a material-alpha surface counts as that surface's alpha,
        /// as raster writes it. Otherwise a visible background keeps alpha at one.
        public let backgroundVisible: Bool
        /// The first authored fog, if any. Read `fogVolumes` to inspect the full medium.
        public var fog: Fog? { fogVolumes.first }

        /// Up to eight fog regions. Extinction adds where regions overlap.
        /// Each box keeps its own scattering albedo and phase function.
        public let fogVolumes: [Fog]
        public static let maximumFogVolumes = 8

        /// The constant linear RGB world radiance, or zero when the environment is a map.
        public var environment: SIMD3<Float> {
            if case .constant(let radiance) = sky { radiance } else { .zero }
        }

        /// A constant environment needs radiance from 0 to 65,504 per channel. A map
        /// needs an intensity from 0 to 65,504 and a finite yaw.
        public init(
            lights: [AreaLight] = [], analyticLights: [AnalyticLight] = [],
            sky: Environment = .constant(.zero), backgroundVisible: Bool = true, fog: Fog? = nil
        ) throws {
            try self.init(
                lights: lights, analyticLights: analyticLights, sky: sky,
                backgroundVisible: backgroundVisible, fogVolumes: fog.map { [$0] } ?? []
            )
        }

        /// A camera ray that scatters in any fog volume counts toward coverage.
        public init(
            lights: [AreaLight] = [], analyticLights: [AnalyticLight] = [],
            sky: Environment = .constant(.zero), backgroundVisible: Bool = true, fogVolumes: [Fog]
        ) throws {
            try self.init(
                lights: lights, analyticLights: analyticLights, sky: sky, additionalSkyEulerRotation: .zero,
                backgroundVisible: backgroundVisible, fogVolumes: fogVolumes
            )
        }

        init(
            lights: [AreaLight], analyticLights: [AnalyticLight], sky: Environment,
            additionalSkyEulerRotation: SIMD3<Float>, backgroundVisible: Bool, fogVolumes: [Fog]
        ) throws {
            guard lights.count <= 16, analyticLights.count <= 16 else { throw Failure.invalidLighting }
            guard fogVolumes.count <= Self.maximumFogVolumes else { throw Failure.fogVolumeLimit }
            switch sky {
            case .constant(let radiance):
                guard PathTracer.isRadiance(radiance) else { throw Failure.invalidLighting }
                skyEulerRotation = .zero
            case .map(_, let intensity, let yaw):
                let angles = additionalSkyEulerRotation + SIMD3(0, yaw, 0)
                guard (0...65_504).contains(intensity), angles.isFiniteVector else { throw Failure.invalidLighting }
                skyEulerRotation = angles
            }
            self.lights = lights
            self.analyticLights = analyticLights
            self.sky = sky
            self.backgroundVisible = backgroundVisible
            self.fogVolumes = fogVolumes
        }

        /// Uses constant linear RGB world radiance from 0 to 65,504 per channel.
        public init(
            lights: [AreaLight] = [], analyticLights: [AnalyticLight] = [],
            environment: SIMD3<Float>, backgroundVisible: Bool = true, fog: Fog? = nil
        ) throws {
            try self.init(
                lights: lights, analyticLights: analyticLights, sky: .constant(environment),
                backgroundVisible: backgroundVisible, fog: fog
            )
        }

        /// Uses constant linear RGB world radiance with multiple fog volumes.
        public init(
            lights: [AreaLight] = [], analyticLights: [AnalyticLight] = [],
            environment: SIMD3<Float>, backgroundVisible: Bool = true, fogVolumes: [Fog]
        ) throws {
            try self.init(
                lights: lights, analyticLights: analyticLights, sky: .constant(environment),
                backgroundVisible: backgroundVisible, fogVolumes: fogVolumes
            )
        }
    }

    /// How each camera sample draws its random numbers. All modes are deterministic for
    /// a seed, image pixel and sample index, so split batches use the same sequence.
    public enum Sampler: Hashable, Sendable {
        /// An independent PCG stream per pixel sample, the original estimator.
        case independent
        /// Owen-scrambled Sobol points: each pixel shuffles and scrambles every
        /// dimension with its own hash, so a pixel's first 2ⁿ samples stratify each
        /// pair of path dimensions while neighboring pixels stay decorrelated. Error
        /// falls faster than with independent samples for smooth integrands such as
        /// area-light illumination. Unbiased, with up to 65,536 samples per pixel.
        case sobol
        /// A common scrambled Sobol sequence with a periodic 64 × 64 blue-noise
        /// shift at each image pixel. Early sample errors shift toward finer
        /// spatial frequencies. Tiled traces use full-image coordinates.
        case blueDither

        fileprivate var shaderIndex: UInt32 {
            switch self {
            case .independent: 0
            case .sobol: 1
            case .blueDither: 2
            }
        }
    }

    public struct Settings: Equatable, Sendable {
        /// Number of scattering events, from 1 to 8.
        public var maxBounces: Int
        public var seed: UInt32
        /// Thin-lens aperture radius in world units. Zero produces a pinhole.
        public var apertureRadius: Float
        /// Positive distance to the focal plane along the camera axis.
        public var focusDistance: Float
        /// The random-number sequence; `.sobol` by default.
        public var sampler: Sampler
        /// The largest linear radiance one contribution may add after the path's first
        /// scattering event, or nil to keep every contribution. Each next-event or
        /// scattered emission term whose largest channel exceeds the limit is scaled
        /// down to it, preserving hue. This suppresses fireflies from rare bright
        /// paths but biases the converged image darker wherever such paths carry
        /// energy, so it no longer converges to the unclamped result. This legacy
        /// limit leaves camera-visible emission and background at their original values.
        /// Positive and finite.
        public var maximumSampleRadiance: Float?
        /// Limits the mean absolute RGB of each camera-visible or one-scatter
        /// contribution. Zero disables this limit. Nil uses the legacy setting.
        public var clampDirect: Float?
        /// Limits the mean absolute RGB of each contribution at depth two and
        /// beyond. Zero disables this limit. Nil uses the legacy setting.
        public var clampIndirect: Float?
        /// Combines light and scattering samples for area emitters, sampled
        /// environments and visible directional disks. Turning it off uses
        /// scattering samples for those lights and keeps analytic-only lights.
        public var multipleImportanceSampling: Bool

        public init(
            maxBounces: Int = 5, seed: UInt32 = 42, apertureRadius: Float = 0, focusDistance: Float = 1,
            sampler: Sampler = .sobol, maximumSampleRadiance: Float? = nil,
            clampDirect: Float? = nil, clampIndirect: Float? = nil, multipleImportanceSampling: Bool = true
        ) {
            self.maxBounces = maxBounces
            self.seed = seed
            self.apertureRadius = apertureRadius
            self.focusDistance = focusDistance
            self.sampler = sampler
            self.maximumSampleRadiance = maximumSampleRadiance
            self.clampDirect = clampDirect
            self.clampIndirect = clampIndirect
            self.multipleImportanceSampling = multipleImportanceSampling
        }
    }

    /// Optional first-hit outputs for the initial sample batch of a view.
    // Texture identities are immutable. The caller orders GPU access on the queue.
    public struct Guides: @unchecked Sendable {
        /// Signed world normal in RGB and primary ray distance in alpha. Misses store zero.
        public let normalDepth: any MTLTexture
        /// Linear base color in RGB and perceptual roughness in alpha.
        public let albedoRoughness: any MTLTexture

        public init(normalDepth: any MTLTexture, albedoRoughness: any MTLTexture) {
            self.normalDepth = normalDepth
            self.albedoRoughness = albedoRoughness
        }
    }

    /// A window of a larger image. The accumulation supplies the window's size.
    ///
    /// Origins use whole-image pixel coordinates from the top left. Both image
    /// axes can reach 32,768 pixels. Each accumulation retains its own size limit.
    public struct Region: Equatable, Sendable {
        public let imageWidth: Int
        public let imageHeight: Int
        public let x: Int
        public let y: Int

        public init(imageWidth: Int, imageHeight: Int, x: Int, y: Int) throws {
            guard (1...32_768).contains(imageWidth), (1...32_768).contains(imageHeight),
                  (0..<imageWidth).contains(x), (0..<imageHeight).contains(y) else {
                throw Failure.region
            }
            self.imageWidth = imageWidth
            self.imageHeight = imageHeight
            self.x = x
            self.y = y
        }

        fileprivate init(whole accumulation: Accumulation) {
            imageWidth = accumulation.width
            imageHeight = accumulation.height
            x = 0
            y = 0
        }
    }

    /// Scratch textures for a single view. Create separate accumulations for independent views.
    // Private textures stay on the tracer queue and their write ownership is locked.
    public final class Accumulation: @unchecked Sendable {
        public let width: Int
        public let height: Int
        fileprivate let owner: UUID
        fileprivate let first: any MTLTexture
        fileprivate let second: any MTLTexture
        fileprivate let state = Mutex(AccumulationState())

        fileprivate init(owner: UUID, first: any MTLTexture, second: any MTLTexture) {
            self.owner = owner
            self.first = first
            self.second = second
            width = first.width
            height = first.height
        }

        fileprivate func texture(at index: Int) -> any MTLTexture { index == 0 ? first : second }
    }

    /// The pending result of an encode. Promote it only after the command buffer succeeds.
    public struct History: Sendable {
        public let sampleCount: Int
        fileprivate let accumulation: Accumulation
        fileprivate let sequence: UInt64
        fileprivate let configuration: Configuration
        fileprivate let lighting: Lighting
        fileprivate let surfaceRevisions: [ShaderUniformRevision]
        fileprivate let sceneRevision: UInt64
        fileprivate let motionScene: MotionScene?
        fileprivate let guides: Guides?
    }

    // Motion history holds the actual trace's instance records and only the
    // vertex buffers that may be replaced without invalidating correspondence.
    // The current scene keeps unchanged vertices resident.
    fileprivate struct MotionScene: @unchecked Sendable {
        let revision: UInt64
        let instances: any MTLBuffer
        let poseVertices: [any MTLBuffer]
    }

    public static let maximumSamples = 65_536
    public var device: any MTLDevice { queue.device }
    package let queue: any MTLCommandQueue
    private let scene: PathTracingScene
    package var sceneRevision: UInt64 { scene.revision }
    let defaultLighting: Lighting
    private let library: any MTLLibrary

    private struct PipelineVariant: Hashable {
        let guides: Bool
        let nullBoundaries: Bool
        let motion: Bool
        let meshEmitters: Bool
    }

    // Variants omit unused bindings and shader paths. Live edits can change null
    // boundaries and can add or remove the scene's first sampled mesh emitter.
    private let pipelines = Mutex<[PipelineVariant: any MTLComputePipelineState]>([:])
    private let aovPipelines = Mutex<[PipelineVariant: AOVPipeline]>([:])
    private let bvhHeatmapPipeline = Mutex<(any MTLComputePipelineState)?>(nil)
    /// A black stand-in bound when the lighting has no environment map.
    private let placeholderSky: any MTLTexture
    private let lightTableCache = LightTextureSet.Cache()
    private let blueNoiseRanks = Mutex<BlueNoiseRanks?>(nil)
    private let owner = UUID()

    /// Prepares color tracing for up to 16 planar area emitters and a constant environment.
    /// The scene must belong to the queue's device. Each encode traces the scene's
    /// state at that time, so updates to it apply to later encodes.
    public convenience init(
        queue: any MTLCommandQueue,
        scene: PathTracingScene,
        lights: [AreaLight] = [],
        environment: SIMD3<Float> = .zero
    ) throws {
        try self.init(queue: queue, scene: scene, lighting: Lighting(lights: lights, environment: environment))
    }

    /// Prepares color tracing with a default rig of area lights, analytic lights
    /// and an environment. Guide and AOV variants prepare on their first use.
    /// The scene and any environment map must belong to the queue's device.
    public init(queue: any MTLCommandQueue, scene: PathTracingScene, lighting: Lighting) throws {
        guard scene.device === queue.device else { throw Failure.wrongDevice }
        guard Self.usesDevice(lighting, queue.device) else { throw Failure.environmentDevice }
        _ = try Self.fogBuffers(lighting, device: queue.device)
        self.queue = queue
        self.scene = scene
        defaultLighting = lighting
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let placeholder = queue.device.makeTexture(descriptor: descriptor) else { throw Failure.environmentAllocation }
        var black = SIMD4<Float16>.zero
        placeholder.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &black, bytesPerRow: 8)
        placeholder.label = "Frostfield path environment placeholder"
        placeholderSky = placeholder
        do {
            library = try scene.generatedSurfaceLibrary ?? queue.device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
        _ = try preparePipeline(.init(
            guides: false, nullBoundaries: scene.hasNullBoundaries, motion: false, meshEmitters: scene.emissiveTriangles != nil
        ))
    }

    /// Prepares the first-hit guide pipeline for the current material snapshot.
    /// Guided viewers can call this at setup to keep compilation out of the first draw.
    public func prepareGuides() throws {
        _ = try preparePipeline(.init(
            guides: true, nullBoundaries: scene.hasNullBoundaries, motion: false, meshEmitters: scene.emissiveTriangles != nil
        ))
    }

    static let maximumAccumulationSide = 4_096
    static let maximumAccumulationPixels = 4_194_304

    private static func supportsAccumulationDimensions(width: Int, height: Int) -> Bool {
        (1...maximumAccumulationSide).contains(width) && (1...maximumAccumulationSide).contains(height)
            && width <= maximumAccumulationPixels / height
    }

    /// Creates two tracked RGBA32Float mean textures for up to 4,194,304 pixels.
    /// Each dimension is limited to 4,096 pixels.
    public func makeAccumulation(width: Int, height: Int) throws -> Accumulation {
        guard Self.supportsAccumulationDimensions(width: width, height: height) else {
            throw Failure.dimensions
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.hazardTrackingMode = .tracked
        descriptor.usage = [.shaderRead, .shaderWrite]
        guard let first = device.makeTexture(descriptor: descriptor),
              let second = device.makeTexture(descriptor: descriptor) else { throw Failure.allocation }
        first.label = "Frostfield path-tracing mean A"
        second.label = "Frostfield path-tracing mean B"
        return Accumulation(owner: owner, first: first, second: second)
    }

    /// Combines the latest successfully completed histories of disjoint windows.
    /// An image may contain up to 4,194,304 pixels and at most 4,096 on either axis.
    /// Copies the counts; subsequent renders cannot change the returned snapshot.
    public func sampleDensity(_ histories: [History]) throws -> SampleDensity {
        guard let region = histories.first?.configuration.region,
              Self.supportsAccumulationDimensions(width: region.imageWidth, height: region.imageHeight) else {
            throw SampleDensity.Failure.imageDimensions
        }
        var counts = [UInt32](repeating: 0, count: region.imageWidth * region.imageHeight)
        for history in histories {
            let window = history.configuration.region
            let accumulation = history.accumulation
            guard window.imageWidth == region.imageWidth, window.imageHeight == region.imageHeight else {
                throw SampleDensity.Failure.imageDimensions
            }
            guard accumulation.owner == owner, accumulation.state.withLock({ state in
                // A newer in-flight write does not supersede the last successful one.
                try? state.reapFinished()
                return state.completed?.sequence == history.sequence
            }) else {
                throw SampleDensity.Failure.history
            }
            let count = UInt32(history.sampleCount)
            for y in window.y..<(window.y + accumulation.height) {
                let start = y * region.imageWidth + window.x
                for index in start..<(start + accumulation.width) {
                    guard counts[index] == 0 else { throw SampleDensity.Failure.overlappingWindows }
                    counts[index] = count
                }
            }
        }
        return SampleDensity(width: region.imageWidth, height: region.imageHeight, counts: counts)
    }

    /// Encodes one pixel-center camera ray per texel and displays intersected
    /// software BVH boxes as a blue-to-red heat ramp. Magenta marks rays that
    /// exhaust the node-test budget. The output is standalone tracked RGBA16Float
    /// with shader-write usage. Use an uncommitted retained command buffer from
    /// this tracer's queue. No accumulation or lighting is needed.
    public func encodeBVHHeatmap(
        cameraModel: CameraModel, cameraTransform: SceneTransform = .init(), destination: any MTLTexture,
        commandBuffer: any MTLCommandBuffer, options: BVHHeatmapOptions = .init(), region: Region? = nil
    ) throws {
        guard scene.accelerationKind == .software else { throw PathTracingScene.BVHBounds.Failure.hardwareAcceleration }
        guard commandBuffer.commandQueue === queue, commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard Self.supportsAccumulationDimensions(width: destination.width, height: destination.height),
              isStandaloneOutput(destination, formats: [.rgba16Float]),
              !scene.boundTextures.contains(where: { $0 === destination }) else {
            throw Failure.boundsHeatmapOutput
        }
        guard (1...65_536).contains(options.saturationCount), (1...65_536).contains(options.maximumNodeVisits) else {
            throw Failure.boundsHeatmapSettings
        }
        let window = try region ?? Region(imageWidth: destination.width, imageHeight: destination.height, x: 0, y: 0)
        guard window.x + destination.width <= window.imageWidth,
              window.y + destination.height <= window.imageHeight else { throw Failure.region }
        let camera = try CameraParameters(
            model: cameraModel, transform: cameraTransform, width: window.imageWidth, height: window.imageHeight
        )
        let levels: UInt32 = (options.displayTopLevel ? 1 : 0) | (options.displayBottomLevel ? 2 : 0)
        let clipsAtSurface = options.stopAtSurface && levels != 0
        let surfaceValues = clipsAtSurface
            ? try snapshotSurfaceUniforms(model: cameraModel, cameraTransform: cameraTransform, region: window).values
            : [.zero]
        let pipeline = try bvhHeatmapPipeline.withLock { cached in
            if let cached { return cached }
            guard let function = library.makeFunction(name: "frostfield_path_bvh_heatmap") else {
                throw Failure.pipeline("The BVH heatmap kernel is missing.")
            }
            do {
                let result = try device.makeComputePipelineState(function: function)
                cached = result
                return result
            } catch {
                throw Failure.pipeline(error.localizedDescription)
            }
        }
        let acceleration = scene.acceleration
        var cameraUniforms = Uniforms(
            right: camera.right, up: camera.up, forward: camera.forward, position: camera.position,
            environment: .zero, projection: camera.clip, image: SIMD4(UInt32(window.imageWidth), UInt32(window.imageHeight), 0, 0),
            controls: SIMD4(0, 0, 0, UInt32(acceleration.topLevelNodes.count)),
            sky: .init(intensity: 0), skyInfo: .zero, lens: camera.lens, shift: camera.shift
        )
        var controls = SIMD4(
            UInt32(acceleration.instances.count), levels, clipsAtSurface ? 1 : 0, UInt32(options.saturationCount)
        )
        var dimensions = SIMD4(UInt32(window.x), UInt32(window.y), UInt32(options.maximumNodeVisits), UInt32(scene.textureCount))
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw Failure.encoder }
        encoder.label = "Frostfield software BVH heatmap"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(scene.materials, offset: 0, index: 2)
        encoder.setBytes(&cameraUniforms, length: MemoryLayout<Uniforms>.stride, index: 4)
        surfaceValues.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 5) }
        encoder.setBuffer(scene.graphTextureTable, offset: 0, index: 6)
        encoder.setBuffer(scene.textureTable, offset: 0, index: 9)
        encoder.setBuffer(acceleration.instanceBuffer, offset: 0, index: 11)
        encoder.setBuffer(acceleration.topLevel, offset: 0, index: 12)
        encoder.setBytes(&controls, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 13)
        encoder.setBytes(&dimensions, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 14)
        encoder.setTexture(scene.textureArray, index: 0)
        encoder.setTexture(destination, index: 3)
        let bottomLevels = acceleration.bottomLevelBuffers
        encoder.useResources(bottomLevels, usage: .read)
        if !scene.boundTextures.isEmpty { encoder.useResources(scene.boundTextures, usage: .read) }
        let width = pipeline.threadExecutionWidth
        let height = max(1, min(8, pipeline.maxTotalThreadsPerThreadgroup / width))
        encoder.dispatchThreadgroups(
            MTLSize(width: (destination.width + width - 1) / width,
                    height: (destination.height + height - 1) / height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1)
        )
        encoder.endEncoding()
        let retained = RetainedBuffers(
            buffers: bottomLevels, textures: scene.boundTextures, samplers: scene.textureSamplers, structures: []
        )
        commandBuffer.addCompletedHandler { _ in withExtendedLifetime(retained) {} }
    }

    /// Encodes 1 to 16 samples per pixel. Use a retained, uncommitted command
    /// buffer from the tracer queue and standalone, tracked RGBA16Float shader-write
    /// textures matching the accumulation size. Each accumulation permits one
    /// pending command buffer. Keep the previous history only after successful GPU
    /// completion. Releasing an abandoned buffer permits another attempt.
    ///
    /// Pass nil history to restart on camera, setting, or lighting changes, and
    /// after any `PathTracingScene` update that changed its `revision`. Also restart
    /// after editing a directly sampled texture. The scene cannot observe external
    /// texture writes; complete them before tracing or order them on the tracer's
    /// queue with hazard tracking.
    /// Omit the lighting override to use the rig passed to the initializer. The
    /// destination alpha is camera-ray coverage, as `Lighting.backgroundVisible`
    /// describes; it stays one while the background is visible, except over
    /// material-alpha surfaces. Guide textures receive a pinhole first-hit sample on
    /// restart, which sees a blended surface where its alpha is at least one half,
    /// and may be reused until the next restart. Later frames may omit guides or
    /// pass the same pair.
    ///
    /// With `timing`, the compute pass records GPU timestamps at its start and end;
    /// read `Timing.duration` after the command buffer completes.
    ///
    /// Pass a `region` to render a tile. The full image determines the camera and
    /// sample sequence. Output and guide textures remain the accumulation's size.
    /// Changing regions requires nil history.
    ///
    /// Pass `aovs` to write first-hit outputs on this encode, including when it
    /// extends an accumulation. These textures cannot alias any other output.
    public func encode(
        camera: PerspectiveCamera,
        cameraTransform: SceneTransform,
        destination: any MTLTexture,
        accumulation: Accumulation,
        previous: History? = nil,
        guides: Guides? = nil,
        commandBuffer: any MTLCommandBuffer,
        samplesPerPixel: Int = 1,
        settings: Settings = Settings(),
        lighting: Lighting? = nil,
        timing: Timing? = nil,
        region: Region? = nil,
        aovs: AOVs? = nil
    ) throws -> History {
        try encode(
            model: CameraModel(perspective: camera, settings: settings), cameraTransform: cameraTransform,
            destination: destination, accumulation: accumulation, previous: previous, guides: guides,
            commandBuffer: commandBuffer, samplesPerPixel: samplesPerPixel, settings: settings, lighting: lighting,
            timing: timing, region: region, aovs: aovs
        )
    }

    /// The shared encode for every camera model; `PathTracer.encode(cameraModel:...)` documents the contract.
    func encode(
        model: CameraModel,
        cameraTransform: SceneTransform,
        destination: any MTLTexture,
        accumulation: Accumulation,
        previous: History?,
        guides: Guides?,
        commandBuffer: any MTLCommandBuffer,
        samplesPerPixel: Int,
        settings: Settings,
        lighting: Lighting?,
        timing: Timing?,
        region: Region?,
        aovs: AOVs?
    ) throws -> History {
        guard accumulation.owner == owner else { throw Failure.accumulation }
        guard commandBuffer.commandQueue === queue, commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard (1...16).contains(samplesPerPixel), (1...8).contains(settings.maxBounces),
              settings.apertureRadius.isFinite, settings.apertureRadius >= 0,
              settings.focusDistance.isNormal, settings.focusDistance > 0,
              settings.maximumSampleRadiance.map({ $0.isFinite && $0 > 0 }) != false,
              settings.clampDirect.map({ $0.isFinite && $0 >= 0 }) != false,
              settings.clampIndirect.map({ $0.isFinite && $0 >= 0 }) != false else { throw Failure.settings }
        if let timing, timing.device !== device { throw Failure.timingDevice }
        guard isOutput(destination, for: accumulation) else { throw Failure.output }
        if let guides = guides ?? previous?.guides {
            guard isOutput(guides.normalDepth, for: accumulation), isOutput(guides.albedoRoughness, for: accumulation),
                  guides.normalDepth !== guides.albedoRoughness,
                  guides.normalDepth !== destination, guides.albedoRoughness !== destination else { throw Failure.output }
        }
        let outputs = aovs?.textures ?? []
        var outputMask: UInt32 = 0
        if !outputs.isEmpty {
            var identities: Set<ObjectIdentifier> = [ObjectIdentifier(destination)]
            if let guides = guides ?? previous?.guides {
                identities.insert(ObjectIdentifier(guides.normalDepth))
                identities.insert(ObjectIdentifier(guides.albedoRoughness))
            }
            for (index, output) in outputs.enumerated() {
                guard let output else { continue }
                guard isOutput(output, for: accumulation, formats: AOVs.formats[index]),
                      identities.insert(ObjectIdentifier(output)).inserted else { throw Failure.aovs }
                outputMask |= 1 << index
            }
        }
        let region = region ?? Region(whole: accumulation)
        guard region.x + accumulation.width <= region.imageWidth,
              region.y + accumulation.height <= region.imageHeight else { throw Failure.region }
        let cameraParameters = try CameraParameters(
            model: model, transform: cameraTransform, width: region.imageWidth, height: region.imageHeight
        )
        if aovs?.projectedDepth != nil {
            switch model.projection {
            case .perspective, .orthographic: break
            case .equirectangular, .stereoEquirectangular: throw Failure.aovProjection
            }
        }
        let motion = aovs?.cameraMotion
        let previousMotionFrame = motion?.previousFrame
        let previousMotionScene = previousMotionFrame?.motionScene
        let previousInstances = previousMotionScene?.instances
        // Later scene updates replace, not mutate, the buffers of this dispatch.
        let acceleration = scene.acceleration
        let sceneRevision = scene.revision
        let motionScene = motion.map { _ in
            MotionScene(
                revision: scene.motionRevision, instances: acceleration.instanceBuffer,
                poseVertices: acceleration.poseVertexBuffers
            )
        }
        let instancesChanged = previousInstances.map { $0 !== acceleration.instanceBuffer } ?? false
        let priorPoseVertices = instancesChanged ? previousMotionScene?.poseVertices ?? [] : []
        let motionUniforms: MotionUniforms?
        if let motion {
            guard previous == nil, samplesPerPixel == 1, region == Region(whole: accumulation),
                  cameraParameters.clip.z < 2, cameraParameters.forward.w == 0,
                  (-0.5...0.5).contains(motion.jitter.x), (-0.5...0.5).contains(motion.jitter.y) else {
                throw Failure.cameraMotion
            }
            if let prior = previousMotionFrame {
                guard prior.accumulation.owner == owner, prior.configuration.frameJitter != nil,
                      prior.configuration.region == region, prior.configuration.camera.clip.z == cameraParameters.clip.z,
                      prior.accumulation.state.withLock({ state in
                          state.completed?.sequence == prior.sequence
                              || (state.pending?.sequence == prior.sequence && state.commandBuffer?.status == .completed)
                      }) else { throw Failure.cameraMotionHistory }
                guard canReuseMotionScene(prior) else { throw Failure.sceneChanged }
            }
            motionUniforms = MotionUniforms(
                prior: previousMotionFrame?.configuration.camera, current: cameraParameters, jitter: motion.jitter,
                instancesChanged: instancesChanged
            )
        } else {
            motionUniforms = nil
        }
        let radianceLimits = SIMD4(
            settings.maximumSampleRadiance ?? 0, settings.clampDirect ?? -1, settings.clampIndirect ?? -1, 0
        )
        let configuration = Configuration(
            camera: cameraParameters, maxBounces: settings.maxBounces, seed: settings.seed, sampler: settings.sampler,
            radianceLimits: radianceLimits, multipleImportanceSampling: settings.multipleImportanceSampling,
            region: region, frameJitter: motion?.jitter
        )
        let lighting = lighting ?? defaultLighting
        guard Self.usesDevice(lighting, device) else { throw Failure.environmentDevice }
        let fogBuffers = try Self.fogBuffers(lighting, device: device)
        if let prior = previousMotionFrame, prior.lighting != lighting { throw Failure.lightingChanged }
        let lightTables = try lightTableCache.make(
            device: device, profiles: lighting.analyticLights.map(\.light.iesProfile),
            cookies: lighting.analyticLights.map(\.light.cookie)
        )
        if let previous {
            guard previous.accumulation === accumulation, previous.configuration == configuration else {
                throw Failure.history
            }
            guard previous.lighting == lighting else { throw Failure.lightingChanged }
            guard previous.sceneRevision == scene.revision else { throw Failure.sceneChanged }
            if let guides {
                guard let prior = previous.guides, prior.normalDepth === guides.normalDepth,
                      prior.albedoRoughness === guides.albedoRoughness else { throw Failure.history }
            }
        }
        let sampleStart = previous?.sampleCount ?? 0
        let sampleCount = sampleStart + samplesPerPixel
        guard sampleCount <= Self.maximumSamples else { throw Failure.sampleLimit }
        let firstHitGuides = previous == nil ? guides : nil
        let emissiveTriangles = settings.multipleImportanceSampling ? scene.emissiveTriangles : nil
        let variant = PipelineVariant(
            guides: firstHitGuides != nil, nullBoundaries: scene.hasNullBoundaries, motion: motion != nil,
            meshEmitters: emissiveTriangles != nil
        )
        let aovPipeline = outputMask != 0 ? try makeAOVPipeline(variant) : nil
        let pipeline = try aovPipeline?.pipeline ?? preparePipeline(variant)
        let ranks = settings.sampler == .blueDither ? try prepareBlueNoiseRanks() : nil
        var dynamics = Dynamics(
            counts: SIMD4(
                UInt32(acceleration.instances.count), settings.sampler.shaderIndex,
                UInt32(scene.textureCount), UInt32(emissiveTriangles?.count ?? 0)
            ),
            limits: radianceLimits
        )
        var skyFlags: UInt32 = lighting.backgroundVisible ? 0 : Self.skyHidden
        if lighting.analyticLights.contains(where: \.diskVisible) { skyFlags |= Self.skyDirectionalDisks }
        if !settings.multipleImportanceSampling { skyFlags |= Self.disableMIS }
        var uniforms = Uniforms(
            right: cameraParameters.right, up: cameraParameters.up, forward: cameraParameters.forward,
            position: cameraParameters.position, environment: SIMD4(lighting.environment, 0),
            projection: cameraParameters.clip,
            image: SIMD4(UInt32(region.imageWidth), UInt32(region.imageHeight), UInt32(sampleStart), UInt32(samplesPerPixel)),
            controls: SIMD4(
                UInt32(settings.maxBounces), settings.seed, UInt32(lighting.lights.count),
                UInt32(acceleration.topLevelNodes.count)
            ),
            sky: .init(),
            skyInfo: SIMD4(0, 0, UInt32(lighting.analyticLights.count), skyFlags),
            lens: cameraParameters.lens, shift: cameraParameters.shift
        )
        var map: EnvironmentMap?
        if case .map(let environment, let intensity, _) = lighting.sky {
            map = environment
            uniforms.sky = EnvironmentLookup(eulerRotation: lighting.skyEulerRotation, intensity: intensity)
            uniforms.skyInfo.x = UInt32(environment.width)
            uniforms.skyInfo.y = UInt32(environment.height)
            uniforms.skyInfo.w |= Self.skyMap | (environment.isSampled ? Self.skySampled : 0)
        }

        let (surfaceValues, surfaceRevisions) = try snapshotSurfaceUniforms(
            model: model, cameraTransform: cameraTransform, region: region
        )
        if let prior = previous ?? previousMotionFrame, prior.surfaceRevisions != surfaceRevisions {
            throw Failure.surfaceUniformsChanged
        }

        return try accumulation.state.withLock { state in
            try state.reapFinished()
            if let previous, previous.sequence != state.completed?.sequence { throw Failure.history }
            let writeIndex = state.completed.map { 1 - $0.index } ?? 0
            let encoder: (any MTLComputeCommandEncoder)?
            if let timing {
                encoder = commandBuffer.makeComputeCommandEncoder(descriptor: try timing.begin(commandBuffer))
            } else {
                encoder = commandBuffer.makeComputeCommandEncoder()
            }
            guard let encoder else {
                timing?.cancel()
                throw Failure.encoder
            }
            encoder.label = "Frostfield path tracing"
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(scene.materials, offset: 0, index: 2)
            encoder.setBuffer(acceleration.instanceBuffer, offset: 0, index: 11)
            encoder.setBuffer(acceleration.topLevel, offset: 0, index: 12)
            encoder.setBytes(&dynamics, length: MemoryLayout<Dynamics>.stride, index: 13)
            // MSL uint2 at buffer 14, independent of the full image size in Uniforms.
            var regionOrigin = SIMD2(UInt32(region.x), UInt32(region.y))
            encoder.setBytes(&regionOrigin, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 14)
            let media = MediaUniforms.pack(lighting.fogVolumes)
            media.withUnsafeBytes { bytes in
                encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 16)
            }
            if !fogBuffers.isEmpty { encoder.useResources(fogBuffers, usage: .read) }
            if let ranks {
                encoder.setBuffer(ranks, offset: 0, index: 19)
            } else {
                var unusedRank: UInt16 = 0
                encoder.setBytes(&unusedRank, length: MemoryLayout<UInt16>.stride, index: 19)
            }
            // Instances address their bottom levels by GPU pointer, and the texture table holds
            // handles to the material textures it samples directly.
            let bottomLevels = acceleration.bottomLevelBuffers
            encoder.useResources(bottomLevels, usage: .read)
            if !priorPoseVertices.isEmpty { encoder.useResources(priorPoseVertices, usage: .read) }
            if let metal = acceleration.metal {
                // The instance structure references every bottom-level structure indirectly.
                encoder.setAccelerationStructure(metal.structure, bufferIndex: 10)
                encoder.useResources(metal.levels, usage: .read)
            }
            if let emissiveTriangles {
                encoder.setBuffer(emissiveTriangles.buffer, offset: 0, index: 20)
            }
            encoder.setBuffer(scene.textureTable, offset: 0, index: 9)
            if !scene.boundTextures.isEmpty { encoder.useResources(scene.boundTextures, usage: .read) }
            Self.bindLights(lighting.lights, into: encoder)
            encoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 4)
            surfaceValues.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 5) }
            encoder.setBuffer(scene.graphTextureTable, offset: 0, index: 6)
            Self.bindAnalyticLights(lighting.analyticLights, tables: lightTables, into: encoder)
            if let map {
                encoder.setBuffer(map.distribution, offset: 0, index: 8)
                encoder.setTexture(map.texture, index: 6)
            } else {
                var unused = SIMD4<Float>.zero
                encoder.setBytes(&unused, length: MemoryLayout<SIMD4<Float>>.stride, index: 8)
                encoder.setTexture(placeholderSky, index: 6)
            }
            encoder.setTexture(scene.textureArray, index: 0)
            encoder.setTexture(accumulation.texture(at: 1 - writeIndex), index: 1)
            encoder.setTexture(accumulation.texture(at: writeIndex), index: 2)
            encoder.setTexture(destination, index: 3)
            if let aovPipeline {
                encoder.setBytes(&outputMask, length: MemoryLayout<UInt32>.stride, index: 15)
                for (index, output) in outputs.dropLast(motion == nil ? 1 : 0).enumerated() {
                    let placeholder = AOVs.formats[index] == [.r32Uint]
                        ? aovPipeline.integerPlaceholder : aovPipeline.colorPlaceholder
                    encoder.setTexture(output ?? placeholder, index: 9 + index)
                }
            }
            if var motionUniforms {
                encoder.setBytes(&motionUniforms, length: MemoryLayout<MotionUniforms>.stride, index: 17)
                encoder.setBuffer(previousInstances ?? acceleration.instanceBuffer, offset: 0, index: 18)
            }
            if let firstHitGuides {
                encoder.setTexture(firstHitGuides.normalDepth, index: 4)
                encoder.setTexture(firstHitGuides.albedoRoughness, index: 5)
            }
            let width = pipeline.threadExecutionWidth
            let height = max(1, min(8, pipeline.maxTotalThreadsPerThreadgroup / width))
            let groups = MTLSize(
                width: (accumulation.width + width - 1) / width,
                height: (accumulation.height + height - 1) / height, depth: 1
            )
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1))
            encoder.endEncoding()

            state.sequence &+= 1
            let sequence = state.sequence
            state.pending = .init(sequence: sequence, index: writeIndex)
            state.commandBuffer = commandBuffer
            let retained = RetainedBuffers(
                buffers: bottomLevels + priorPoseVertices + fogBuffers,
                textures: scene.boundTextures, samplers: scene.textureSamplers,
                structures: acceleration.structures
            )
            commandBuffer.addCompletedHandler { buffer in
                withExtendedLifetime(retained) {}
                accumulation.state.withLock { $0.finish(sequence: sequence, succeeded: buffer.status == .completed) }
            }
            return History(
                sampleCount: sampleCount, accumulation: accumulation, sequence: sequence,
                configuration: configuration, lighting: lighting, surfaceRevisions: surfaceRevisions,
                sceneRevision: sceneRevision, motionScene: motionScene, guides: guides ?? previous?.guides
            )
        }
    }

    private func snapshotSurfaceUniforms(
        model: CameraModel, cameraTransform: SceneTransform, region: Region
    ) throws -> (values: [SIMD4<Float>], revisions: [ShaderUniformRevision]) {
        // Copy every live uniform block into this dispatch and remember the revisions it used.
        var surfaceValues = [SIMD4<Float>](repeating: .zero, count: max(1, scene.surfaceUniformSlotCount))
        if scene.generatedSurfaceLibrary != nil {
            let graphView = try PathTracingScene.graphViewBlock(
                model: model, cameraTransform: cameraTransform, width: region.imageWidth, height: region.imageHeight,
                readsProjectionMatrix: scene.surfaceReadsProjectionMatrix
            )
            surfaceValues.replaceSubrange(0..<graphView.count, with: graphView)
        }
        var surfaceRevisions: [ShaderUniformRevision] = []
        surfaceRevisions.reserveCapacity(scene.surfaceUniforms.count)
        for (uniforms, offset, mask, readsClock) in scene.surfaceUniforms {
            let snapshot = uniforms.bufferSnapshot(reading: mask)
            surfaceValues.replaceSubrange(offset..<(offset + snapshot.values.count), with: snapshot.values)
            // History follows traced shading inputs. Vertex-only edits are tracked by the scene pose.
            surfaceRevisions.append(ShaderUniformRevision(
                values: snapshot.revision, frame: readsClock ? snapshot.frameRevision : nil
            ))
        }
        return (surfaceValues, surfaceRevisions)
    }

    func canReuseMotionScene(_ history: History) -> Bool {
        history.motionScene?.revision == scene.motionRevision
    }

    // The cache publishes a buffer only after its contents and label are set.
    private struct BlueNoiseRanks: @unchecked Sendable {
        let buffer: any MTLBuffer
    }

    private func prepareBlueNoiseRanks() throws -> any MTLBuffer {
        try blueNoiseRanks.withLock { cached in
            if let cached { return cached }
            // The bundled field contains a periodic 64 × 64 permutation of 12-bit ranks.
            guard let url = Bundle.module.url(forResource: "PathBlueNoiseRanks", withExtension: "bin"),
                  let data = try? Data(contentsOf: url),
                  data.count == 64 * 64 * MemoryLayout<UInt16>.stride,
                  let buffer = data.withUnsafeBytes({
                      device.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                                        options: [.storageModeShared, .hazardTrackingModeTracked])
                  }) else { throw Failure.blueNoiseRanks }
            buffer.label = "Frostfield path blue-noise ranks"
            let ranks = BlueNoiseRanks(buffer: buffer)
            cached = ranks
            return ranks
        }.buffer
    }

    private static func bindLights(_ lights: [AreaLight], into encoder: any MTLComputeCommandEncoder) {
        if lights.isEmpty {
            var dummy = PackedAreaLight()
            encoder.setBytes(&dummy, length: MemoryLayout<PackedAreaLight>.stride, index: 3)
        } else {
            // Metal copies the complete light rig into this command's argument data.
            lights.map(PackedAreaLight.init).withUnsafeBytes { bytes in
                encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 3)
            }
        }
    }

    private static func bindAnalyticLights(
        _ lights: [AnalyticLight], tables: LightTextureSet, into encoder: any MTLComputeCommandEncoder
    ) {
        if lights.isEmpty {
            var unused = PackedAnalyticLight()
            encoder.setBytes(&unused, length: MemoryLayout<PackedAnalyticLight>.stride, index: 7)
        } else {
            lights.indices.map { index in
                PackedAnalyticLight(
                    lights[index], profileLayer: tables.profileLayers[index], cookieLayer: tables.cookieLayers[index]
                )
            }.withUnsafeBytes { bytes in
                encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 7)
            }
        }
        // Textures 7 and 8 hold the IES profile tables and spot cookies.
        encoder.setTexture(tables.profiles, index: 7)
        encoder.setTexture(tables.cookies, index: 8)
    }

    private static func usesDevice(_ lighting: Lighting, _ device: any MTLDevice) -> Bool {
        if case .map(let map, _, _) = lighting.sky { return map.device === device }
        return true
    }

    private func isOutput(
        _ texture: any MTLTexture, for accumulation: Accumulation, formats: [MTLPixelFormat] = [.rgba16Float]
    ) -> Bool {
        isStandaloneOutput(texture, formats: formats)
            && texture.width == accumulation.width && texture.height == accumulation.height
    }

    private func isStandaloneOutput(_ texture: any MTLTexture, formats: [MTLPixelFormat]) -> Bool {
        texture.device === device && formats.contains(texture.pixelFormat) && texture.textureType == .type2D
            && texture.sampleCount == 1
            && texture.usage.contains(.shaderWrite) && texture.storageMode != .memoryless && !texture.isFramebufferOnly
            && texture.hazardTrackingMode != .untracked && texture.parent == nil && texture.buffer == nil
            && texture.iosurface == nil && texture.heap == nil
    }

    // The pipeline and texture identities are immutable. Omitted slots never
    // write the placeholder textures, even if several encodes share them.
    private struct AOVPipeline: @unchecked Sendable {
        let pipeline: any MTLComputePipelineState
        let colorPlaceholder: any MTLTexture
        let integerPlaceholder: any MTLTexture
    }

    private func preparePipeline(_ variant: PipelineVariant) throws -> any MTLComputePipelineState {
        try pipelines.withLock { cache in
            if let pipeline = cache[variant] { return pipeline }
            let pipeline = try makePipeline(variant)
            cache[variant] = pipeline
            return pipeline
        }
    }

    private func makeAOVPipeline(_ variant: PipelineVariant) throws -> AOVPipeline {
        try aovPipelines.withLock { cache in
            if let prepared = cache[variant] { return prepared }
            // Every specialized texture slot must be bound. The mask prevents
            // writes to these stand-ins when the caller omits a texture.
            func placeholder(_ format: MTLPixelFormat) throws -> any MTLTexture {
                let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: format, width: 1, height: 1, mipmapped: false
                )
                descriptor.usage = .shaderWrite
                descriptor.storageMode = .private
                guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.allocation }
                return texture
            }
            let pipeline = try makePipeline(variant, aovs: true)
            let prepared = try AOVPipeline(
                pipeline: pipeline, colorPlaceholder: placeholder(.rgba16Float), integerPlaceholder: placeholder(.r32Uint)
            )
            cache[variant] = prepared
            return prepared
        }
    }

    private func makePipeline(_ variant: PipelineVariant, aovs: Bool = false) throws -> any MTLComputePipelineState {
        var guides = variant.guides
        var nullBoundaries = variant.nullBoundaries
        var metal = scene.accelerationKind == .metal
        var aovs = aovs
        var motion = variant.motion
        var meshEmitters = variant.meshEmitters
        let constants = MTLFunctionConstantValues()
        constants.setConstantValue(&guides, type: .bool, index: 0)
        constants.setConstantValue(&metal, type: .bool, index: 1)
        constants.setConstantValue(&aovs, type: .bool, index: 2)
        constants.setConstantValue(&motion, type: .bool, index: 3)
        constants.setConstantValue(&nullBoundaries, type: .bool, index: 4)
        constants.setConstantValue(&meshEmitters, type: .bool, index: 5)
        do {
            let function = try library.makeFunction(name: "frostfield_path_trace", constantValues: constants)
            return try device.makeComputePipelineState(function: function)
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
    }

    private static func isRadiance(_ value: SIMD3<Float>) -> Bool {
        (0...65_504).contains(value.x) && (0...65_504).contains(value.y) && (0...65_504).contains(value.z)
    }

    private static func isFinite(_ value: SIMD3<Float>) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite
    }

    private struct PackedAreaLight {
        var center: SIMD4<Float> = .zero
        var halfU: SIMD4<Float> = .zero
        var halfV: SIMD4<Float> = .zero
        var radiance: SIMD4<Float> = .zero

        init() {}

        init(_ light: AreaLight) {
            center = SIMD4(light.center, light.cameraVisible ? 1 : 0)
            halfU = SIMD4(light.halfU, light.shape == .disk ? 1 : 0)
            halfV = SIMD4(light.halfV, 0)
            radiance = SIMD4(light.radiance, 0)
        }
    }

    // Environment flags in Uniforms.skyInfo.w, matching PathTracingShading.h.
    private static let skyMap: UInt32 = 1
    private static let skySampled: UInt32 = 2
    private static let skyHidden: UInt32 = 4
    private static let skyDirectionalDisks: UInt32 = 8
    private static let disableMIS: UInt32 = 16

    // Keep in lockstep with PTUniforms in PathTracingShading.h.
    package struct Uniforms {
        var right: SIMD4<Float>
        var up: SIMD4<Float>
        var forward: SIMD4<Float>
        var position: SIMD4<Float>
        /// Constant environment radiance, zero when a map supplies it.
        var environment: SIMD4<Float>
        var projection: SIMD4<Float>
        var image: SIMD4<UInt32>
        /// Bounces, seed, area light count, BVH node count.
        var controls: SIMD4<UInt32>
        /// World-to-map orientation and intensity.
        var sky: EnvironmentLookup
        /// Map width, map height, analytic light count, environment flags.
        var skyInfo: SIMD4<UInt32>
        /// Aperture blade count (0 round), rotation, and horizontal and vertical aperture scales.
        var lens: SIMD4<Float>
        /// Perspective window shift in half-window units, then unused.
        var shift: SIMD4<Float>
    }

    fileprivate struct Configuration: Equatable, Sendable {
        let camera: CameraParameters
        let maxBounces: Int
        let seed: UInt32
        let sampler: Sampler
        let radianceLimits: SIMD4<Float>
        let multipleImportanceSampling: Bool
        let region: Region
        let frameJitter: SIMD2<Float>?
    }

    // Keep in lockstep with PTMotionUniforms. position.w is 0 without history,
    // 1 for unchanged instances, and 2 when the previous instance snapshot must be used;
    // shiftAndJitter packs the previous window shift and the current pixel jitter.
    private struct MotionUniforms {
        let right: SIMD4<Float>
        let up: SIMD4<Float>
        let forward: SIMD4<Float>
        let position: SIMD4<Float>
        let clip: SIMD4<Float>
        let shiftAndJitter: SIMD4<Float>

        init(prior: CameraParameters?, current: CameraParameters, jitter: SIMD2<Float>, instancesChanged: Bool) {
            let camera = prior ?? current
            right = camera.right
            up = camera.up
            forward = camera.forward
            let historyMode: Float = prior == nil ? 0 : (instancesChanged ? 2 : 1)
            position = SIMD4(camera.position.x, camera.position.y, camera.position.z, historyMode)
            clip = camera.clip
            shiftAndJitter = SIMD4(camera.shift.x, camera.shift.y, jitter.x, jitter.y)
        }
    }

    // Keep in lockstep with PTDynamics in PathTracingShading.h.
    package struct Dynamics {
        /// Instance count, sampler (0 independent, 1 Sobol, 2 blue dither), texture and emissive triangle counts.
        var counts: SIMD4<UInt32>
        /// Legacy peak limit, direct and indirect mean limits (-1 uses legacy), then unused.
        var limits: SIMD4<Float>
    }

    // The bottom levels and directly sampled textures are reachable only through GPU addresses
    // and handles, so the completion handler keeps them alive until the GPU finishes. Metal
    // resources are thread-safe to retain and release.
    private struct RetainedBuffers: @unchecked Sendable {
        let buffers: [any MTLBuffer]
        let textures: [any MTLTexture]
        let samplers: [any MTLSamplerState]
        let structures: [any MTLAccelerationStructure]
    }

    // right.w and up.w hold the half window (tangents for perspective, world units for
    // orthographic). clip holds near, far, projection mode, and half the stereo eye separation.
    fileprivate struct CameraParameters: Equatable, Sendable {
        let right: SIMD4<Float>
        let up: SIMD4<Float>
        let forward: SIMD4<Float>
        let position: SIMD4<Float>
        let clip: SIMD4<Float>
        let lens: SIMD4<Float>
        let shift: SIMD4<Float>

        init(model: CameraModel, transform: SceneTransform, width: Int, height: Int) throws {
            let rotationLength = simd_length_squared(transform.rotation.vector)
            guard transform.scale == SIMD3(repeating: 1), PathTracer.isFinite(transform.translation),
                  abs(rotationLength - 1) < 1e-4, model.isValid(width: width, height: height) else { throw Failure.camera }
            let window = model.window(width: width, height: height)
            let rotation = simd_normalize(transform.rotation)
            right = SIMD4(rotation.act(SIMD3(1, 0, 0)), window.x)
            up = SIMD4(rotation.act(SIMD3(0, 1, 0)), window.y)
            forward = SIMD4(rotation.act(SIMD3(0, 0, -1)), model.lens?.apertureRadius ?? 0)
            position = SIMD4(transform.translation, model.lens?.focusDistance ?? 1)
            clip = SIMD4(model.near, model.far, model.mode, model.halfEyeSeparation)
            lens = model.lens?.shape ?? SIMD4(0, 0, 1, 1)
            shift = SIMD4(model.shift.x, model.shift.y, 0, 0)
        }
    }

    // All access to the weak buffer identity is serialized by Accumulation.state.
    fileprivate struct AccumulationState: @unchecked Sendable {
        struct Write {
            let sequence: UInt64
            let index: Int
        }

        weak var commandBuffer: (any MTLCommandBuffer)?
        var sequence: UInt64 = 0
        var pending: Write?
        var completed: Write?

        mutating func reapFinished() throws {
            guard let pending else { return }
            guard let commandBuffer else {
                self.pending = nil
                return
            }
            switch commandBuffer.status {
            case .completed: finish(sequence: pending.sequence, succeeded: true)
            case .error: finish(sequence: pending.sequence, succeeded: false)
            default: throw Failure.accumulationBusy
            }
        }

        mutating func finish(sequence: UInt64, succeeded: Bool) {
            guard pending?.sequence == sequence else { return }
            if succeeded { completed = pending }
            pending = nil
            commandBuffer = nil
        }
    }

    private static func fogBuffers(_ lighting: Lighting, device: any MTLDevice) throws -> [any MTLBuffer] {
        let densities = lighting.fogVolumes.compactMap { $0.densityGrid?.buffer }
        guard densities.allSatisfy({ $0.device === device }) else { throw Failure.fogDensityDevice }
        let boundaries = lighting.fogVolumes.compactMap { $0.meshBoundary?.buffer }
        guard boundaries.allSatisfy({ $0.device === device }) else { throw Failure.fogBoundaryDevice }
        return densities + boundaries
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidAreaLight
        case invalidAnalyticLight
        case invalidLighting
        case invalidEnvironmentMap
        case environmentAllocation
        case environmentDevice
        case pipeline(String)
        case dimensions
        case region
        case allocation
        case blueNoiseRanks
        case wrongDevice
        case output
        case aovs
        case aovProjection
        case boundsHeatmapOutput
        case boundsHeatmapSettings
        case cameraMotion
        case cameraMotionHistory
        case commandBuffer
        case settings
        case camera
        case graphCameraProjection
        case accumulation
        case accumulationBusy
        case history
        case lightingChanged
        case surfaceUniformsChanged
        case sceneChanged
        case sampleLimit
        case encoder
        case timingUnsupported
        case timingDevice
        case timingBusy
        case frameBudget
        case invalidFog
        case invalidFogBoundary
        case invalidFogTransform
        case fogBoundaryDevice
        case fogVolumeLimit
        case invalidFogDensityGrid
        case fogDensityAllocation
        case fogDensityDevice

        public var errorDescription: String? {
            switch self {
            case .invalidAreaLight: "An area light needs finite perpendicular half axes and nonnegative HDR radiance."
            case .invalidAnalyticLight:
                "An analytic light needs a finite position within ±10¹² units and a nonzero finite direction. "
                    + "Only directional lights accept a nonzero angular diameter up to π/2. "
                    + "A finite, nonnegative spot radius must place its aperture within ±10¹² units."
            case .invalidLighting:
                "Path tracing supports up to 16 area lights, 16 analytic lights, and environment radiance or map intensity from 0 to 65,504 with finite rotation angles."
            case .invalidEnvironmentMap:
                "Use a 2:1 linear RGB panorama up to 8,192 × 4,096 pixels with values from 0 to 65,504."
            case .environmentAllocation: "Metal could not allocate the path-tracing environment texture or sampling table."
            case .environmentDevice: "Use an environment map prepared on the path tracer’s Metal device."
            case .pipeline(let detail): "Could not prepare the path-tracing pipeline. \(detail)"
            case .dimensions: "Path tracing dimensions may reach 4,096 per axis and 4,194,304 pixels total."
            case .region: "The path tracing region must fit an image of up to 32,768 pixels per axis."
            case .allocation: "Metal could not allocate the path-tracing mean textures."
            case .blueNoiseRanks: "The path tracer could not prepare its blue-noise rank field."
            case .wrongDevice: "The path-tracing scene must use the command queue’s Metal device."
            case .output: "Use separate, tracked RGBA16Float shader-write textures matching the accumulation size."
            case .aovs: "Use distinct, tracked shader-write textures with the accumulation size and documented AOV formats."
            case .aovProjection: "Projected depth requires a perspective or orthographic camera."
            case .boundsHeatmapOutput: "Use a standalone tracked RGBA16Float shader-write heatmap texture within the path-tracing size limits."
            case .boundsHeatmapSettings: "BVH heatmap saturation and node-test budget must be between 1 and 65,536."
            case .cameraMotion:
                "Motion vectors need a full planar pinhole frame, one sample, no progressive history, and jitter within ±0.5 pixels."
            case .cameraMotionHistory:
                "Use the latest completed motion frame for its accumulation from this tracer with matching dimensions and projection kind, or reset history."
            case .commandBuffer: "Use a retained, uncommitted command buffer from the path tracer’s queue."
            case .settings:
                "Use 1 to 16 samples per pixel, 1 to 8 bounces, a finite aperture and focus distance, "
                    + "and finite radiance limits. The legacy limit is positive, and direct and indirect limits allow zero."
            case .camera:
                "Path tracing requires a finite rigid camera pose and a valid camera model: a representable perspective or orthographic window, finite lens values with a perspective, an even image height for stereo panoramas, and 0 < near < far."
            case .graphCameraProjection: "A panoramic camera has no planar projection matrix for a surface graph."
            case .accumulation: "Create the accumulation with this path tracer."
            case .accumulationBusy: "Wait for the accumulation’s current command buffer to finish."
            case .history: "Use the latest successful history for this view or pass nil to restart."
            case .lightingChanged: "Pass nil history to restart after changing the path-tracing lighting."
            case .surfaceUniformsChanged: "Pass nil history to restart after editing surface shader uniforms."
            case .sceneChanged: "Pass nil history to restart after updating the path tracing scene."
            case .timingUnsupported: "This Metal device cannot sample GPU timestamps at compute pass boundaries."
            case .timingDevice: "Create the path tracing timing on the tracer's Metal device."
            case .timingBusy: "Wait for the timed command buffer to finish before reusing its timing."
            case .frameBudget: "Use a positive finite frame budget, a sample range in 1...16, and responsiveness in (0, 1]."
            case .sampleLimit: "Path tracing accumulates up to 65,536 samples per pixel before restarting."
            case .encoder: "Metal could not create the path-tracing compute encoder."
            case .invalidFogDensityGrid:
                "Use 1 to 128 voxels per axis, with one finite density from zero to one per voxel."
            case .fogDensityAllocation: "Metal could not allocate the fog density grid."
            case .fogDensityDevice: "Use a fog density grid prepared on the path tracer’s Metal device."
            case .fogVolumeLimit: "Use up to eight path-traced fog volumes."
            case .invalidFogBoundary:
                "Use a watertight fog boundary with consistently wound, nondegenerate faces. Outer shells wind outward, cavity shells wind toward the cavity, and shells do not intersect, touch or nest with the same orientation."
            case .invalidFogTransform:
                "Place the fog boundary with finite values, a unit rotation, scale magnitudes of at least 10⁻⁶ and a ratio between the largest and smallest scale of at most 10⁶."
            case .fogBoundaryDevice: "Prepare the fog boundary on the path tracer's Metal device."
            case .invalidFog:
                "Fog needs extinction from 0 to 10⁶ per unit, albedo from zero to one, anisotropy greater than -1 and less than 1, and a finite box within ±10¹² units whose minimum is below its maximum."
            }
        }
    }
}


extension PathTracer {
    /// First-hit layers available alongside beauty in a streamed still.
    public enum StillEXRLayer: String, CaseIterable, Sendable {
        case albedo
        case normal
        case depth
        case roughness
        case emission
        case objectID
        case materialID
        case specularAlbedo
        /// Forward Metal device depth. Available for planar cameras only.
        case projectedDepth

        fileprivate var format: MTLPixelFormat {
            switch self {
            case .albedo, .normal, .emission, .specularAlbedo: .rgba16Float
            case .depth, .roughness, .projectedDepth: .r32Float
            case .objectID, .materialID: .r32Uint
            }
        }

        fileprivate var binding: WritableKeyPath<AOVs, (any MTLTexture)?> {
            switch self {
            case .albedo: \.albedo
            case .normal: \.normal
            case .depth: \.depth
            case .roughness: \.roughness
            case .emission: \.emission
            case .objectID: \.objectID
            case .materialID: \.materialID
            case .specularAlbedo: \.specularAlbedo
            case .projectedDepth: \.projectedDepth
            }
        }
    }

    /// Traces an HDR still to OpenEXR, with half-float beauty RGBA and optional
    /// first-hit layers. Colors and normals use half, depths and roughness use
    /// float, and IDs use unsigned integer channels. The image coordinates
    /// and sample sequence match a single full view.
    ///
    /// GPU tiles are up to 1,024 columns wide and one compression block high.
    /// CPU channel storage covers one full-width scanline block. The call waits for
    /// each sample batch. Keep the scene and its textures stable for the export.
    /// A failed dispatch, scene revision change or task cancellation leaves any
    /// existing destination file intact.
    ///
    /// Select linear `depth` for distance along a panoramic camera ray.
    /// Panoramas reject `projectedDepth`. Motion belongs to temporal frames.
    public func writeStillEXR(
        to url: URL,
        cameraModel: CameraModel,
        cameraTransform: SceneTransform = SceneTransform(),
        width: Int,
        height: Int,
        samplesPerPixel: Int,
        settings: Settings = Settings(),
        lighting: Lighting? = nil,
        compression: EXRExporter.Compression = .zip,
        layers: Set<StillEXRLayer> = []
    ) throws {
        guard (1...Self.maximumSamples).contains(samplesPerPixel) else { throw Failure.sampleLimit }
        if layers.contains(.projectedDepth) {
            switch cameraModel.projection {
            case .perspective, .orthographic: break
            case .equirectangular, .stereoEquirectangular: throw Failure.aovProjection
            }
        }

        struct Tile {
            let accumulation: Accumulation
            let colors: [any MTLTexture]
            let scalars: [any MTLTexture]
            let ids: [any MTLTexture]
            let aovs: AOVs?
        }
        let orderedLayers = StillEXRLayer.allCases.filter(layers.contains)
        let colorLayers = orderedLayers.filter { $0.format == .rgba16Float }
        let scalarLayers = orderedLayers.filter { $0.format == .r32Float }
        let idLayers = orderedLayers.filter { $0.format == .r32Uint }
        let tileWidth = 1_024
        var tiles: [SIMD2<Int>: Tile] = [:]
        var reference: History?
        try EXRExporter(compression: compression).write(to: url, width: width, height: height) { rows in
            let pixelCount = width * rows.count
            var colors = (0...colorLayers.count).map { _ in [SIMD4<Float16>](repeating: .zero, count: pixelCount) }
            var scalars = scalarLayers.map { _ in [Float](repeating: 0, count: pixelCount) }
            var ids = idLayers.map { _ in [UInt32](repeating: 0, count: pixelCount) }
            for x in stride(from: 0, to: width, by: tileWidth) {
                let columns = min(tileWidth, width - x)
                let size = SIMD2(columns, rows.count)
                let tile: Tile
                if let cached = tiles[size] {
                    tile = cached
                } else {
                    func texture(_ format: MTLPixelFormat) throws -> any MTLTexture {
                        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                            pixelFormat: format, width: columns, height: rows.count, mipmapped: false
                        )
                        descriptor.storageMode = .shared
                        descriptor.usage = .shaderWrite
                        guard let output = device.makeTexture(descriptor: descriptor) else { throw Failure.output }
                        return output
                    }

                    var aovs = AOVs()
                    func textures(_ group: [StillEXRLayer]) throws -> [any MTLTexture] {
                        try group.map { layer in
                            let output = try texture(layer.format)
                            aovs[keyPath: layer.binding] = output
                            return output
                        }
                    }

                    let colorTextures = [try texture(.rgba16Float)] + (try textures(colorLayers))
                    let scalarTextures = try textures(scalarLayers)
                    let idTextures = try textures(idLayers)
                    let accumulation = try makeAccumulation(width: columns, height: rows.count)
                    tile = Tile(
                        accumulation: accumulation, colors: colorTextures, scalars: scalarTextures, ids: idTextures,
                        aovs: orderedLayers.isEmpty ? nil : aovs
                    )
                    tiles[size] = tile
                }
                let region = try Region(imageWidth: width, imageHeight: height, x: x, y: rows.lowerBound)
                var history: History?
                for start in stride(from: 0, to: samplesPerPixel, by: 16) {
                    try Task.checkCancellation()
                    guard let command = queue.makeCommandBuffer() else { throw Failure.commandBuffer }
                    let next = try encode(
                        cameraModel: cameraModel, cameraTransform: cameraTransform, destination: tile.colors[0],
                        accumulation: tile.accumulation, previous: history, commandBuffer: command,
                        samplesPerPixel: min(16, samplesPerPixel - start), settings: settings,
                        lighting: lighting, region: region, aovs: start == 0 ? tile.aovs : nil
                    )
                    if let reference {
                        guard next.sceneRevision == reference.sceneRevision else { throw Failure.sceneChanged }
                        guard next.surfaceRevisions == reference.surfaceRevisions else { throw Failure.surfaceUniformsChanged }
                    }
                    command.commit()
                    command.waitUntilCompleted()
                    guard command.status == .completed else { throw command.error ?? Failure.commandBuffer }
                    history = next
                    if reference == nil { reference = next }
                }

                func read<T>(_ textures: [any MTLTexture], into bands: inout [[T]]) {
                    let stride = MemoryLayout<T>.stride
                    for (index, texture) in textures.enumerated() {
                        bands[index].withUnsafeMutableBytes { bytes in
                            texture.getBytes(
                                bytes.baseAddress!.advanced(by: x * stride), bytesPerRow: width * stride,
                                from: MTLRegionMake2D(0, 0, columns, rows.count), mipmapLevel: 0
                            )
                        }
                    }
                }
                read(tile.colors, into: &colors)
                read(tile.scalars, into: &scalars)
                read(tile.ids, into: &ids)
            }

            var channels = Self.stillColorChannels(colors[0], layer: nil)
            for (layer, pixels) in zip(colorLayers, colors.dropFirst()) {
                channels += Self.stillColorChannels(pixels, layer: layer)
            }
            for (layer, values) in zip(scalarLayers, scalars) {
                let component = layer == .roughness ? "R" : "Z"
                channels.append(.init(name: "\(layer.rawValue).\(component)", samples: .float(values)))
            }
            for (layer, values) in zip(idLayers, ids) {
                channels.append(.init(name: "\(layer.rawValue).id", samples: .unsigned(values)))
            }
            return channels
        }
    }

    private static func stillColorChannels(
        _ pixels: [SIMD4<Float16>], layer: StillEXRLayer?
    ) -> [EXRExporter.Channel] {
        let components: [KeyPath<SIMD4<Float16>, Float16>] = [\.x, \.y, \.z, \.w]
        let names = layer == .normal ? ["X", "Y", "Z", "A"] : ["R", "G", "B", "A"]
        let prefix = layer.map { $0.rawValue + "." } ?? ""
        return zip(names, components).map { name, component in
            .init(name: prefix + name, samples: .half(pixels.map { $0[keyPath: component] }))
        }
    }
}
