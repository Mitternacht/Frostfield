import Foundation
import Metal
import simd

/// A `SkyModel` rendered on the GPU into lighting for both renderers.
///
/// Preparation builds Bruneton's transmittance table and Hillaire's
/// multiple-scattering table, integrates the sky's horizontal irradiance with
/// cosine-weighted and phase-function-weighted directions combined by multiple
/// importance sampling, and ray-marches every texel of a linear HDR
/// equirectangular panorama at 16 wavelengths. The panorama then feeds raster image-based lighting through
/// `EnvironmentLighting` and an importance-sampled `PathTracer.EnvironmentMap`.
/// Below the horizon the panorama shows the Lambertian ground seen through the
/// atmosphere, lit by the Sun and the sky.
///
/// The panorama holds sky radiance only. Add the Sun once, as `sunLight()`, in a
/// `LightRig` shared by both renderers. The path tracer uses the same directional
/// emitter for direct illumination and for the disk seen on escaping rays:
///
/// ```swift
/// let sky = try SkyEnvironment(queue: queue, model: model)
/// let rig = try LightRig(lights: [sky.sunLight()])
/// // lightIntensity: 0 turns off the material's own raster-only key light.
/// let material = try PhysicallyBasedMaterial(..., lightIntensity: 0, ..., environment: sky.materialEnvironment())
/// let nodes = try rig.sceneNodes()                                  // raster
/// let lighting = try rig.pathLighting(sky: sky.pathEnvironment)     // path tracer
/// ```
///
/// Preparation runs on the supplied queue and waits for completion. The result is
/// immutable and can be shared by renderers on its device.
public final class SkyEnvironment: @unchecked Sendable {
    // Immutable after initialization; the GPU only reads the textures.
    public struct Settings: Equatable, Sendable {
        /// Panorama width in texels, an even number from 64 to 8,192. The height is half.
        public var panoramaWidth: Int
        /// Resolution and sample counts for the raster image-based lighting.
        public var lighting: EnvironmentLighting.Settings

        public init(panoramaWidth: Int = 1_024, lighting: EnvironmentLighting.Settings = .init(cubeSize: 256)) {
            self.panoramaWidth = panoramaWidth
            self.lighting = lighting
        }
    }

    /// Integration resolution. The defaults keep the sky within about 1% of a
    /// double-precision reference integration.
    package struct Quality: Equatable, Sendable {
        package var transmittanceSteps = 500
        /// Azimuths per cosine node; the cosine uses 12 Gauss–Legendre nodes on each side of the horizon.
        package var multipleScatteringDirections = 24
        package var multipleScatteringSteps = 64
        package var viewSteps = 64

        package init() {}
    }

    package static let transmittanceSize = (width: 256, height: 64)
    package static let multipleScatteringSize = 32

    public let model: SkyModel
    public let settings: Settings
    /// Split-sum raster lighting and the background cube.
    public let lighting: EnvironmentLighting
    /// The same panorama prepared for importance-sampled path tracing.
    public let pathMap: PathTracer.EnvironmentMap
    /// Linear sRGB sky radiance in a shared RGBA32Float 2:1 texture. Treat it as read-only.
    public let panorama: any MTLTexture
    /// Illuminance on an upward-facing surface at the world origin from the whole
    /// sky, excluding the Sun's direct beam, in lux times `model.luminanceScale`.
    public let skyIlluminance: SIMD3<Float>
    /// The direct solar beam at the world origin; see `SkyModel.sunIlluminance`.
    public let sunIlluminance: SIMD3<Float>
    /// The brightest panorama component.
    public let peakRadiance: Float

    package let transmittanceTable: any MTLTexture
    package let multipleScatteringTable: any MTLTexture
    /// Four float4 batches of the sky's horizontal spectral irradiance at the origin
    /// per unit solar irradiance, which lights the ground below the horizon.
    package let spectralIrradiance: any MTLBuffer
    package let constants: Constants

    public var device: any MTLDevice { panorama.device }

    /// The importance-sampled panorama for `PathTracer.Lighting` or `LightRig.pathLighting(sky:)`.
    public var pathEnvironment: PathTracer.Environment { .map(pathMap) }

    /// A material environment that lights PBR materials with this sky.
    public func materialEnvironment(intensity: Float = 1) throws -> PhysicallyBasedMaterial.Environment {
        try PhysicallyBasedMaterial.Environment(lighting: lighting, intensity: intensity)
    }

    /// The Sun as a directional `LightRig` light; see `SkyModel.sunLight(id:target:castsShadow:)`.
    /// Its intensity is lux times `model.luminanceScale`, so scale every other light
    /// in the rig by `model.luminanceScale` as well.
    public func sunLight(id: String = "sun", target: SIMD3<Float> = .zero, castsShadow: Bool = true) -> LightRig.Light {
        model.sunLight(id: id, target: target, castsShadow: castsShadow)
    }

    /// Renders the sky on `queue` and prepares both renderers' environments.
    /// Throws `Failure.radianceRange` when part of the sky exceeds the 65,504
    /// half-float limit of the lighting textures, which happens near the Sun in hazy
    /// skies with `SkyModel.luminanceScale` at one; lower the scale and raise display
    /// exposure by the same factor.
    public convenience init(queue: any MTLCommandQueue, model: SkyModel, settings: Settings = .init()) throws {
        try self.init(queue: queue, model: model, settings: settings, quality: Quality())
    }

    package init(
        queue: any MTLCommandQueue, model: SkyModel, settings: Settings, quality: Quality,
        multipleScattering: Bool = true
    ) throws {
        let width = settings.panoramaWidth
        guard width.isMultiple(of: 2), (64...PathTracer.EnvironmentMap.maximumWidth).contains(width) else {
            throw Failure.settings
        }
        let device = queue.device
        let library: any MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw Failure.library(error.localizedDescription)
        }
        let transmittancePipeline = try Self.pipeline("frostfield_sky_transmittance", library: library)
        let multiplePipeline = try Self.pipeline("frostfield_sky_multiple_scattering", library: library)
        let irradiancePipeline = try Self.pipeline("frostfield_sky_irradiance", library: library)
        let panoramaPipeline = try Self.pipeline("frostfield_sky_panorama", library: library)

        var constants = Constants(model: model, multipleScattering: multipleScattering)
        var steps = SIMD4<UInt32>(
            UInt32(quality.transmittanceSteps), UInt32(quality.multipleScatteringDirections),
            UInt32(quality.multipleScatteringSteps), UInt32(quality.viewSteps)
        )
        let transmittance = try Self.texture(
            .type2D, width: Self.transmittanceSize.width, height: Self.transmittanceSize.height, slices: 1,
            device: device, label: "sky transmittance"
        )
        let multiple = try Self.texture(
            .type2DArray, width: Self.multipleScatteringSize, height: Self.multipleScatteringSize, slices: 4,
            device: device, label: "sky multiple scattering"
        )
        let panorama = try Self.texture(.type2D, width: width, height: width / 2, slices: 1, device: device, label: "sky panorama")
        guard let irradiance = device.makeBuffer(length: 4 * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared) else {
            throw Failure.resource("sky irradiance")
        }
        irradiance.label = "Frostfield sky irradiance"

        guard let command = queue.makeCommandBuffer() else { throw Failure.commandBuffer }
        command.label = "Frostfield sky preparation"
        guard let encoder = command.makeComputeCommandEncoder() else { throw Failure.encoder }
        encoder.label = "Frostfield sky tables and panorama"
        encoder.setBytes(&constants, length: MemoryLayout<Constants>.stride, index: 0)
        encoder.setBytes(&steps, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 1)

        encoder.setComputePipelineState(transmittancePipeline)
        encoder.setTexture(transmittance, index: 0)
        Self.dispatch(encoder, pipeline: transmittancePipeline, width: transmittance.width, height: transmittance.height, depth: 1)

        encoder.setComputePipelineState(multiplePipeline)
        encoder.setTexture(transmittance, index: 0)
        encoder.setTexture(multiple, index: 1)
        Self.dispatch(encoder, pipeline: multiplePipeline, width: multiple.width, height: multiple.height, depth: 4)

        encoder.setComputePipelineState(irradiancePipeline)
        encoder.setTexture(multiple, index: 1)
        encoder.setBuffer(irradiance, offset: 0, index: 2)
        var threads = 1
        while threads * 2 <= min(irradiancePipeline.maxTotalThreadsPerThreadgroup, 256) { threads *= 2 }
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )

        encoder.setComputePipelineState(panoramaPipeline)
        encoder.setTexture(panorama, index: 2)
        Self.dispatch(encoder, pipeline: panoramaPipeline, width: panorama.width, height: panorama.height, depth: 1)
        encoder.endEncoding()

        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else { throw Failure.execution(command.error?.localizedDescription) }

        var texels = [SIMD4<Float>](repeating: .zero, count: panorama.width * panorama.height)
        texels.withUnsafeMutableBytes { bytes in
            panorama.getBytes(
                bytes.baseAddress!, bytesPerRow: panorama.width * MemoryLayout<SIMD4<Float>>.stride,
                from: MTLRegionMake2D(0, 0, panorama.width, panorama.height), mipmapLevel: 0
            )
        }
        var peak: Float = 0
        var pixels = [SIMD3<Float>]()
        pixels.reserveCapacity(texels.count)
        for texel in texels {
            let rgb = SIMD3(texel.x, texel.y, texel.z)
            guard rgb.x.isFinite, rgb.y.isFinite, rgb.z.isFinite else { throw Failure.execution("The sky produced nonfinite radiance.") }
            peak = max(peak, simd_reduce_max(rgb))
            pixels.append(rgb)
        }
        guard peak <= 65_504 else { throw Failure.radianceRange(peak) }

        let spectral = irradiance.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        var skySpectrum = [Double]()
        for batch in 0..<4 {
            for lane in 0..<4 {
                let index = batch * 4 + lane
                skySpectrum.append(Double(spectral[batch][lane]) * SkyModel.Spectrum.solarIrradiance[index]
                    / (model.earthSunDistance * model.earthSunDistance))
            }
        }

        self.lighting = try EnvironmentLighting(queue: queue, equirectangular: panorama, settings: settings.lighting)
        self.pathMap = try PathTracer.EnvironmentMap(device: device, pixels: pixels, width: panorama.width, height: panorama.height)
        self.model = model
        self.settings = settings
        self.panorama = panorama
        self.peakRadiance = peak
        self.skyIlluminance = SIMD3<Float>(simd_max(SkyModel.Spectrum.linearSRGB(skySpectrum), .zero) * model.luminanceScale)
        self.sunIlluminance = model.sunIlluminance
        self.transmittanceTable = transmittance
        self.multipleScatteringTable = multiple
        self.spectralIrradiance = irradiance
        self.constants = constants
    }

    private static func texture(
        _ type: MTLTextureType, width: Int, height: Int, slices: Int, device: any MTLDevice, label: String
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = type
        descriptor.pixelFormat = .rgba32Float
        descriptor.width = width
        descriptor.height = height
        descriptor.arrayLength = slices
        descriptor.storageMode = .shared
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.hazardTrackingMode = .tracked
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.resource(label) }
        texture.label = "Frostfield \(label)"
        return texture
    }

    private static func pipeline(_ name: String, library: any MTLLibrary) throws -> any MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else { throw Failure.shader(name) }
        do {
            return try library.device.makeComputePipelineState(function: function)
        } catch {
            throw Failure.pipeline(name, error.localizedDescription)
        }
    }

    static func dispatch(
        _ encoder: any MTLComputeCommandEncoder, pipeline: any MTLComputePipelineState, width: Int, height: Int, depth: Int
    ) {
        let groupWidth = min(pipeline.maxTotalThreadsPerThreadgroup, 8)
        let groupHeight = max(1, min(pipeline.maxTotalThreadsPerThreadgroup / groupWidth, 8))
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (width + groupWidth - 1) / groupWidth, height: (height + groupHeight - 1) / groupHeight, depth: depth
            ),
            threadsPerThreadgroup: MTLSize(width: groupWidth, height: groupHeight, depth: 1)
        )
    }

    /// Mirrors FrostfieldSkyConstants in Atmosphere.metal: 35 float4 slots.
    package struct Constants {
        var radii: SIMD4<Float>
        var profile: SIMD4<Float>
        var sun: SIMD4<Float>
        var rayleigh: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var mieScattering: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var mieExtinction: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var ozone: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var solar: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var red: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var green: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        var blue: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)

        init(model: SkyModel, multipleScattering: Bool) {
            let air = model.atmosphere
            let coefficients = model.coefficients
            let distance2 = model.earthSunDistance * model.earthSunDistance
            let weights = SkyModel.Spectrum.rgbWeights.map { $0 * model.luminanceScale }
            radii = SIMD4(Float(air.groundRadius), Float(air.topRadius), Float(max(model.altitude, 1)), 0)
            profile = SIMD4(
                Float(1 / air.rayleighScaleHeight), Float(1 / air.aerosolScaleHeight),
                Float(air.aerosolAsymmetry), Float(air.groundAlbedo)
            )
            let flags: UInt32 = (multipleScattering ? 1 : 0) | 2
            sun = SIMD4(model.sunDirection, Float(bitPattern: flags))
            rayleigh = Self.batches(coefficients.rayleigh)
            mieScattering = Self.batches(coefficients.mieScattering)
            mieExtinction = Self.batches(coefficients.mieExtinction)
            ozone = Self.batches(coefficients.ozone)
            solar = Self.batches(SkyModel.Spectrum.solarIrradiance.map { $0 / distance2 })
            red = Self.batches(weights.map(\.x))
            green = Self.batches(weights.map(\.y))
            blue = Self.batches(weights.map(\.z))
        }

        static func batches(_ values: [Double]) -> (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>) {
            func batch(_ index: Int) -> SIMD4<Float> {
                SIMD4(Float(values[index * 4]), Float(values[index * 4 + 1]), Float(values[index * 4 + 2]), Float(values[index * 4 + 3]))
            }
            return (batch(0), batch(1), batch(2), batch(3))
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case settings
        case library(String)
        case shader(String)
        case pipeline(String, String)
        case resource(String)
        case commandBuffer
        case encoder
        case execution(String?)
        case radianceRange(Float)

        public var errorDescription: String? {
            switch self {
            case .settings: "Use an even panorama width from 64 to 8,192 texels."
            case .library(let detail): "Could not load Frostfield's Metal library: \(detail)"
            case .shader(let name): "The sky shader \(name) is missing."
            case .pipeline(let name, let detail): "Could not prepare \(name): \(detail)"
            case .resource(let name): "Metal could not allocate the \(name)."
            case .commandBuffer: "Metal could not create a sky command buffer."
            case .encoder: "Metal could not create the sky compute encoder."
            case .execution(let detail): detail ?? "Metal could not finish rendering the sky."
            case .radianceRange(let peak):
                "The sky peaks at \(Int(peak.rounded())), above the 65,504 half-float limit of environment textures. Multiply SkyModel.luminanceScale by \(String(format: "%.3g", 60_000 / peak)) or less and raise display exposure to match."
            }
        }
    }
}
