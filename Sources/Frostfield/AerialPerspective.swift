import Foundation
import Metal
import simd

/// Atmospheric perspective for a raster frame: the light a `SkyEnvironment`'s
/// atmosphere scatters into each view ray and the extinction of the surface
/// behind it, plus the sky and limb-darkened solar disk where depth is empty.
///
/// Each frame a compute pass fills two 64³ froxel volumes along the camera's
/// pixel rays: in-scattered radiance and transmittance from the camera to
/// distances `maximumDistance × (k / 63)²`, using the sky's transmittance and
/// multiple-scattering tables at 16 wavelengths. A full-screen pass reads stored
/// depth, converts it to meters along the pixel's ray and composites
/// `color × transmittance + inscattering × alpha`. The world origin is at
/// `SkyModel.altitude`, +Y points up and the planet curves away below.
///
/// Background pixels show the sky as seen from the camera, so a raised camera sees
/// a lower, bluer horizon that matches the haze on distant geometry. Within
/// `panoramaAltitudeTolerance` (10 m) of `SkyModel.altitude` and 1 km horizontally
/// of the world origin, the sky panorama is that sky and is read directly. Elsewhere
/// the compute pass also ray-marches a 256 × 256 sky table at the camera's altitude
/// with 48 steps per texel. Columns cover the azimuth from the Sun's vertical plane.
/// Rows pack zenith angles toward the horizon as in Hillaire's sky-view table. The
/// ground below the horizon is lit by the Sun and by the sky irradiance at the world
/// origin. The solar disk adds wavelength-dependent power-law limb darkening and the
/// extinction along the camera's line of sight to the Sun. The disk's luminance, about 1.6 × 10⁹ cd/m²
/// at noon, exceeds the half-float range at the default scale and saturates at
/// 65,504; with `SkyModel.luminanceScale` at or below about 3 × 10⁻⁵ the limb
/// darkening stays visible.
///
/// Spectral transmittance becomes RGB by weighting each wavelength with the
/// positive part of the sRGB channel's response to sunlight, an approximation for
/// surfaces whose color is not lit by sunlight.
public final class AerialPerspective {
    public struct Settings: Equatable, Sendable {
        /// Meters per world unit.
        public var metersPerUnit: Float
        /// Distance in meters covered by the froxel volumes. Farther surfaces use
        /// the value at this distance.
        public var maximumDistance: Float
        /// Whether pixels without geometry show the sky. When false they keep the source color.
        public var drawsSky: Bool
        /// Whether the sky includes the solar disk.
        public var drawsSun: Bool

        public init(metersPerUnit: Float = 1, maximumDistance: Float = 100_000, drawsSky: Bool = true, drawsSun: Bool = true) {
            self.metersPerUnit = metersPerUnit
            self.maximumDistance = maximumDistance
            self.drawsSky = drawsSky
            self.drawsSun = drawsSun
        }
    }

    package static let volumeSize = 64
    package static let skyTableSize = (width: 256, height: 256)
    /// Altitude difference in meters from `SkyModel.altitude` within which the camera
    /// reuses the sky panorama as its background. Ten meters move the horizon by
    /// under 0.1° and change the sky's radiance by well under 1%.
    public static let panoramaAltitudeTolerance: Double = 10
    /// Horizontal distance in meters from the world origin within which the camera
    /// reuses the panorama; 1 km tilts the local vertical by 0.009°.
    public static let panoramaDistanceTolerance: Double = 1_000

    public let environment: SkyEnvironment
    private let volumePipeline: any MTLComputePipelineState
    private let skyPipeline: any MTLComputePipelineState
    private let compositePipeline: any MTLRenderPipelineState
    private let inscattering: any MTLTexture
    private let transmittance: any MTLTexture
    private let skyTable: any MTLTexture

    /// Prepares the pipelines and the froxel volumes. Encode frames on one queue:
    /// Metal's hazard tracking orders each frame's volume update after the previous
    /// frame's reads.
    public init(environment: SkyEnvironment) throws {
        let device = environment.device
        do {
            let library = try device.makeDefaultLibrary(bundle: .module)
            guard let kernel = library.makeFunction(name: "frostfield_sky_aerial_volume"),
                  let skyKernel = library.makeFunction(name: "frostfield_sky_aerial_sky"),
                  let vertex = library.makeFunction(name: "frostfield_sky_aerial_vertex"),
                  let fragment = library.makeFunction(name: "frostfield_sky_aerial_fragment") else {
                throw Failure.missingShader
            }
            volumePipeline = try device.makeComputePipelineState(function: kernel)
            skyPipeline = try device.makeComputePipelineState(function: skyKernel)
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.label = "Frostfield aerial perspective"
            descriptor.vertexFunction = vertex
            descriptor.fragmentFunction = fragment
            descriptor.colorAttachments[0].pixelFormat = .rgba16Float
            compositePipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch let failure as Failure {
            throw failure
        } catch {
            throw Failure.pipeline(error.localizedDescription)
        }
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type3D
        descriptor.pixelFormat = .rgba16Float
        descriptor.width = Self.volumeSize
        descriptor.height = Self.volumeSize
        descriptor.depth = Self.volumeSize
        descriptor.storageMode = .private
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.hazardTrackingMode = .tracked
        guard let inscattering = device.makeTexture(descriptor: descriptor),
              let transmittance = device.makeTexture(descriptor: descriptor) else { throw Failure.resource }
        inscattering.label = "Frostfield aerial in-scattering"
        transmittance.label = "Frostfield aerial transmittance"
        // Float32 keeps hazy skies beyond half-float range until the composite clamps them;
        // the composite reads it without filtering.
        let skyDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: Self.skyTableSize.width, height: Self.skyTableSize.height, mipmapped: false
        )
        skyDescriptor.storageMode = .private
        skyDescriptor.usage = [.shaderRead, .shaderWrite]
        skyDescriptor.hazardTrackingMode = .tracked
        guard let skyTable = device.makeTexture(descriptor: skyDescriptor) else { throw Failure.resource }
        skyTable.label = "Frostfield aerial sky table"
        self.environment = environment
        self.inscattering = inscattering
        self.transmittance = transmittance
        self.skyTable = skyTable
    }

    /// Encodes on the caller's retained command buffer after the scene's color and
    /// depth passes. Use equally sized, single-sample, shader-readable, hazard-tracked
    /// premultiplied linear RGBA16Float color and Depth32Float depth, rendered with
    /// `camera`, `cameraTransform` and `depthMode`, and a separate RGBA16Float render
    /// target. Cleared depth (1 forward, 0 reversed) marks the sky. The camera
    /// transform needs a unit rotation and unit scale; its translation is in world units.
    public func encode(
        source: any MTLTexture,
        depth: any MTLTexture,
        destination: any MTLTexture,
        camera: PerspectiveCamera,
        cameraTransform: SceneTransform = .init(),
        depthMode: MeshDepthMode = .forward,
        settings: Settings = .init(),
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try depthMode.requireProjectionDepth()
        let device = environment.device
        guard commandBuffer.retainedReferences,
              commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw Failure.commandBuffer
        }
        guard commandBuffer.device === device, source.device === device, depth.device === device,
              destination.device === device else { throw Failure.wrongDevice }
        guard source !== destination, Self.isReadable(source, format: .rgba16Float),
              Self.isReadable(depth, format: .depth32Float), destination.pixelFormat == .rgba16Float,
              destination.textureType == .type2D, destination.sampleCount == 1,
              source.width > 0, source.height > 0,
              source.width == depth.width, source.height == depth.height,
              source.width == destination.width, source.height == destination.height,
              destination.usage.contains(.renderTarget), destination.storageMode != .memoryless else {
            throw Failure.incompatibleTextures
        }
        guard settings.metersPerUnit.isFinite, settings.metersPerUnit > 0,
              settings.maximumDistance.isFinite, (1...1e7).contains(settings.maximumDistance) else {
            throw Failure.settings
        }
        var uniforms = try uniforms(
            camera: camera, cameraTransform: cameraTransform, depthMode: depthMode, settings: settings,
            aspectRatio: Float(source.width) / Float(source.height)
        )
        var constants = environment.constants

        guard let compute = commandBuffer.makeComputeCommandEncoder() else { throw Failure.encoder }
        compute.label = "Frostfield aerial perspective volume"
        compute.setComputePipelineState(volumePipeline)
        compute.setTexture(environment.transmittanceTable, index: 0)
        compute.setTexture(environment.multipleScatteringTable, index: 1)
        compute.setTexture(inscattering, index: 2)
        compute.setTexture(transmittance, index: 3)
        compute.setBytes(&constants, length: MemoryLayout<SkyEnvironment.Constants>.stride, index: 0)
        compute.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        SkyEnvironment.dispatch(compute, pipeline: volumePipeline, width: Self.volumeSize, height: Self.volumeSize, depth: 1)
        if settings.drawsSky, uniforms.options.z > 0.5 {
            compute.setComputePipelineState(skyPipeline)
            compute.setTexture(skyTable, index: 5)
            compute.setBuffer(environment.spectralIrradiance, offset: 0, index: 2)
            SkyEnvironment.dispatch(
                compute, pipeline: skyPipeline, width: Self.skyTableSize.width, height: Self.skyTableSize.height, depth: 1
            )
        }
        compute.endEncoding()

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = destination
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { throw Failure.encoder }
        encoder.label = "Frostfield aerial perspective"
        encoder.setRenderPipelineState(compositePipeline)
        encoder.setFragmentTexture(source, index: 0)
        encoder.setFragmentTexture(depth, index: 1)
        encoder.setFragmentTexture(inscattering, index: 2)
        encoder.setFragmentTexture(transmittance, index: 3)
        encoder.setFragmentTexture(environment.panorama, index: 4)
        encoder.setFragmentTexture(skyTable, index: 5)
        encoder.setFragmentBytes(&constants, length: MemoryLayout<SkyEnvironment.Constants>.stride, index: 0)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    private static func isReadable(_ texture: any MTLTexture, format: MTLPixelFormat) -> Bool {
        texture.pixelFormat == format && texture.textureType == .type2D && texture.sampleCount == 1
            && texture.usage.contains(.shaderRead) && !texture.isFramebufferOnly
            && texture.storageMode != .memoryless && texture.hazardTrackingMode != .untracked
    }

    private func uniforms(
        camera: PerspectiveCamera, cameraTransform: SceneTransform, depthMode: MeshDepthMode,
        settings: Settings, aspectRatio: Float
    ) throws -> Uniforms {
        let rotation = cameraTransform.rotation
        guard abs(simd_length_squared(rotation.vector) - 1) <= 1e-4,
              simd_reduce_max(abs(cameraTransform.scale - 1)) <= 1e-4 else { throw Failure.cameraTransform }
        let translation = SIMD3<Double>(cameraTransform.translation) * Double(settings.metersPerUnit)
        guard translation.x.isFinite, translation.y.isFinite, translation.z.isFinite else { throw Failure.cameraTransform }

        let model = environment.model
        let ground = model.atmosphere.groundRadius
        // The planet's center lies below the world origin; the camera's altitude is
        // its distance from that center minus the ground radius.
        let originHeight = max(model.altitude, 1) + translation.y
        let fromCenter = SIMD3(translation.x, ground + originHeight, translation.z)
        let radius = simd_length(fromCenter)
        let squares = translation.x * translation.x + translation.z * translation.z + originHeight * (2 * ground + originHeight)
        let altitude = max(squares / (radius + ground), 1)
        guard altitude < model.atmosphere.topRadius - ground else { throw Failure.cameraTransform }
        let vertical = fromCenter / radius
        let usesTable = abs(altitude - max(model.altitude, 1)) > Self.panoramaAltitudeTolerance
            || (translation.x * translation.x + translation.z * translation.z).squareRoot() > Self.panoramaDistanceTolerance

        let tanY = tan(camera.verticalFieldOfView / 2)
        let sun = SIMD3<Double>(model.sunDirection)
        let solar = SkyModel.Spectrum.solarIrradiance.map { $0 / (model.earthSunDistance * model.earthSunDistance) }
        let sunTransmittance = model.transmittance(altitude: altitude, cosine: simd_dot(sun, vertical))
        let radiusAngle = model.sunAngularRadius
        // Exact cone solid angle; 1 − cos θ = 2 sin²(θ / 2) keeps precision.
        let half = sin(radiusAngle / 2)
        let solidAngle = 4 * Double.pi * half * half
        let limb = SkyModel.Spectrum.limbDarkening
        // Center radiance of a μ^α disk whose average radiance carries the irradiance: L(1) = (α + 2) / 2 × E / Ω.
        let disk = (0..<SkyModel.Spectrum.count).map { index in
            solar[index] * sunTransmittance[index] * (limb[index] + 2) / 2 / solidAngle
        }

        var red = [Double](), green = [Double](), blue = [Double]()
        for index in 0..<SkyModel.Spectrum.count {
            let weight = simd_max(SkyModel.Spectrum.rgbWeights[index], .zero) * solar[index]
            red.append(weight.x)
            green.append(weight.y)
            blue.append(weight.z)
        }
        let normalize = { (values: [Double]) -> [Double] in
            let total = values.reduce(0, +)
            return values.map { $0 / total }
        }
        let clipFlag: Float = depthMode == .reversed ? 1 : 0
        let batches = SkyEnvironment.Constants.batches
        return Uniforms(
            right: SIMD4(rotation.act(SIMD3(1, 0, 0)), tanY * aspectRatio),
            up: SIMD4(rotation.act(SIMD3(0, 1, 0)), tanY),
            forward: SIMD4(rotation.act(SIMD3(0, 0, -1)), settings.maximumDistance),
            planet: SIMD4(SIMD3<Float>(vertical), Float(altitude)),
            transmittanceRed: batches(normalize(red)),
            transmittanceGreen: batches(normalize(green)),
            transmittanceBlue: batches(normalize(blue)),
            sunDisk: SIMD4(model.sunDirection, Float(cos(radiusAngle))),
            clip: SIMD4(1 / camera.far, 1 / camera.near, clipFlag, settings.metersPerUnit),
            options: SIMD4(settings.drawsSky ? 1 : 0, settings.drawsSun ? 1 : 0, usesTable ? 1 : 0, 0),
            diskCenter: batches(disk),
            limbDarkening: batches(limb)
        )
    }

    /// Mirrors FrostfieldAerialUniforms in Atmosphere.metal: 27 float4 slots.
    package struct Uniforms {
        var right: SIMD4<Float>
        var up: SIMD4<Float>
        var forward: SIMD4<Float>
        var planet: SIMD4<Float>
        var transmittanceRed: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var transmittanceGreen: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var transmittanceBlue: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var sunDisk: SIMD4<Float>
        var clip: SIMD4<Float>
        var options: SIMD4<Float>
        var diskCenter: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var limbDarkening: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
    }

    public enum Failure: LocalizedError, Equatable {
        case missingShader
        case pipeline(String)
        case resource
        case commandBuffer
        case wrongDevice
        case incompatibleTextures
        case settings
        case cameraTransform
        case encoder

        public var errorDescription: String? {
            switch self {
            case .missingShader: "Frostfield's aerial perspective shaders are missing."
            case .pipeline(let detail): "Could not prepare Frostfield's aerial perspective: \(detail)"
            case .resource: "Metal could not allocate the aerial perspective volumes or sky table."
            case .commandBuffer: "Aerial perspective needs an uncommitted command buffer that retains resources."
            case .wrongDevice: "Aerial perspective textures and command buffer must use the sky's Metal device."
            case .incompatibleTextures:
                "Use equal-sized, shader-readable RGBA16Float color and Depth32Float depth with a separate RGBA16Float render target."
            case .settings: "Use a positive meters-per-unit scale and a maximum distance from 1 m to 10,000 km."
            case .cameraTransform: "Use a unit camera rotation and scale with the camera inside the atmosphere."
            case .encoder: "Metal could not create an aerial perspective encoder."
            }
        }
    }
}
