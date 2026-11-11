import Foundation
import Frostfield
import Testing
import simd

struct SkyModelTests {
    private typealias Atmosphere = SkyModel.Atmosphere

    @Test("Spectral weights reproduce the solar constant's illuminance and color")
    func solarSpectrum() throws {
        // With nothing to scatter or absorb, the beam is the extraterrestrial sun.
        let vacuum = Atmosphere(airDensity: 0, aerosolOpticalDepth: 0, ozoneColumn: 0)
        let model = try SkyModel(sunDirection: SIMD3(0, 1, 0), atmosphere: vacuum, luminanceScale: 1)
        let sun = SIMD3<Double>(model.sunIlluminance)
        let luminance = simd_dot(sun, SIMD3(0.2126, 0.7152, 0.0722))
        // Bruneton's 10 nm ASTM G-173 table integrates to 132,603 lx against the same
        // observer; Darula et al. (2005) recommend a luminous solar constant of 133,334 lx.
        #expect(abs(luminance - 132_603) / 132_603 < 0.002, "\(luminance) lx")
        #expect(abs(luminance - 133_334) / 133_334 < 0.01)
        let inverse = SkyModel.Spectrum.xyzToLinearSRGB.inverse
        let xyz = inverse * sun
        let chromaticity = SIMD2(xyz.x, xyz.y) / (xyz.x + xyz.y + xyz.z)
        #expect(simd_distance(chromaticity, SIMD2(0.3219, 0.3310)) < 0.001, "\(chromaticity)")
        // Linear sRGB luminance equals photometric Y for any spectrum.
        let flat = SkyModel.Spectrum.linearSRGB(Array(repeating: 1, count: 16))
        let integral = SkyModel.Spectrum.colorMatching.reduce(0) { $0 + $1.y } * 683
        #expect(abs(simd_dot(flat, SIMD3(0.2126, 0.7152, 0.0722)) - integral) / integral < 1e-4)
        #expect(abs(integral / 683 - 106.92) < 0.05)

        // Irradiance falls with the square of distance and the disk shrinks with it.
        let perihelion = try SkyModel(sunDirection: SIMD3(0, 1, 0), atmosphere: vacuum, earthSunDistance: 0.9833, luminanceScale: 1)
        #expect(abs(Double(perihelion.sunIlluminance.y / model.sunIlluminance.y) - 1 / (0.9833 * 0.9833)) < 1e-5)
        #expect(abs(model.sunAngularRadius - 0.0046505) < 1e-6)
        #expect(perihelion.sunAngularRadius > model.sunAngularRadius)
    }

    @Test("The direct beam follows Beer–Lambert extinction through closed-form columns")
    func beerLambert() throws {
        let air = Atmosphere()
        for elevation in [90.0, 30.0] {
            let radians = elevation * .pi / 180
            let model = try SkyModel(sunDirection: SIMD3(Float(cos(radians)), Float(sin(radians)), 0), atmosphere: air, luminanceScale: 0.001)
            let transmittance = model.sunTransmittance(altitude: 0)
            let coefficients = model.coefficients
            // Exponential columns from one meter up, the evaluation floor, to the top.
            let top = air.topRadius - air.groundRadius
            func column(_ height: Double) -> Double { height * (exp(-1 / height) - exp(-top / height)) }
            let ozone = 15_000.0
            for index in [0, 7, 15] {
                let vertical = coefficients.rayleigh[index] * column(air.rayleighScaleHeight)
                    + coefficients.mieExtinction[index] * column(air.aerosolScaleHeight) + coefficients.ozone[index] * ozone
                if elevation == 90 {
                    #expect(abs(transmittance[index] - exp(-vertical)) < 1e-6, "\(index)")
                } else {
                    // The slant path matches the independent 3D integration, and its
                    // optical depth stays within 1% of the flat-Earth air mass 1 / sin(elevation).
                    let reference = AtmosphereReference(model)
                    let slant = reference.spectralTransmittance(
                        reference.opticalLengths(from: reference.origin, toward: reference.sun, steps: 4_000)
                    )
                    #expect(abs(log(transmittance[index]) / log(slant[index]) - 1) < 1e-5, "\(index)")
                    #expect(abs(log(transmittance[index]) / (-vertical / sin(radians)) - 1) < 0.01, "\(index)")
                }
            }
        }
        // Rayleigh optical depth at 550 nm and the turbidity relation.
        #expect(abs(air.rayleighOpticalDepth(wavelength: 550) - 0.1084) < 0.0005)
        let hazy = try Atmosphere(turbidity: 3)
        #expect(abs(hazy.aerosolOpticalDepth - 2 * air.rayleighOpticalDepth(wavelength: 550)) < 1e-12)
        #expect(hazy.aerosolAngstromExponent == 1.3)
        let model = try SkyModel(sunDirection: SIMD3(0, 1, 0), atmosphere: hazy, luminanceScale: 0.001)
        let extinction = model.coefficients.mieExtinction
        #expect(abs(extinction[0] / extinction[15] - pow(400.0 / 700, -1.3)) < 1e-9)
        #expect(throws: SkyModel.Failure.self) { try Atmosphere(turbidity: 0.5) }
    }

    @Test("The sun light carries the transmitted beam, the disk size and a normalized color")
    func sunLight() throws {
        let model = try SkyModel(sunDirection: SIMD3(0.5, 0.5, 0), luminanceScale: 0.001)
        let light = model.sunLight(id: "noon", target: SIMD3(1, 0, 0))
        let illuminance = model.sunIlluminance
        #expect(light.id == "noon" && light.emission == .intensity(simd_reduce_max(illuminance)))
        #expect(simd_reduce_max(light.color) == 1 && simd_distance(light.color * simd_reduce_max(illuminance), illuminance) < 1e-3)
        #expect(simd_distance(light.direction, -simd_normalize(SIMD3(0.5, 0.5, 0))) < 1e-6)
        #expect(light.kind == .directional(angularDiameter: Float(2 * model.sunAngularRadius)))
        // A low sun is redder and dimmer.
        let low = try SkyModel(sunDirection: SIMD3(1, 0.05, 0), luminanceScale: 0.001)
        #expect(low.sunIlluminance.x / low.sunIlluminance.z > illuminance.x / illuminance.z)
        #expect(low.sunIlluminance.y < illuminance.y)
        let rig = try LightRig(lights: [light, low.sunLight(id: "low")])
        #expect(rig.lights.count == 2)
        // Below the horizon the beam is gone, and the light stays valid.
        let night = try SkyModel(sunDirection: SIMD3(1, -0.2, 0), luminanceScale: 0.001)
        #expect(night.sunIlluminance == .zero && night.sunLight().emission == .intensity(0))
        _ = try LightRig(lights: [night.sunLight()])
    }

    @Test("Sky models reject invalid parameters")
    func validation() throws {
        #expect(throws: SkyModel.Failure.sunDirection) { try SkyModel(sunDirection: .zero, luminanceScale: 1) }
        #expect(throws: SkyModel.Failure.sunDirection) { try SkyModel(sunDirection: SIMD3(.nan, 1, 0), luminanceScale: 1) }
        #expect(throws: SkyModel.Failure.altitude) { try SkyModel(sunDirection: SIMD3(0, 1, 0), altitude: 70_000, luminanceScale: 1) }
        #expect(throws: SkyModel.Failure.altitude) { try SkyModel(sunDirection: SIMD3(0, 1, 0), altitude: -1, luminanceScale: 1) }
        #expect(throws: SkyModel.Failure.luminanceScale) { try SkyModel(sunDirection: SIMD3(0, 1, 0), luminanceScale: 0) }
        #expect(throws: SkyModel.Failure.earthSunDistance) { try SkyModel(sunDirection: SIMD3(0, 1, 0), earthSunDistance: 3, luminanceScale: 1) }
        for atmosphere in [
            Atmosphere(groundAlbedo: 1.5), Atmosphere(aerosolAsymmetry: 1), Atmosphere(aerosolOpticalDepth: -0.1),
            Atmosphere(groundRadius: 6_360_000, topRadius: 6_359_000), Atmosphere(rayleighScaleHeight: .infinity),
        ] {
            #expect(throws: SkyModel.Failure.self) { try SkyModel(sunDirection: SIMD3(0, 1, 0), atmosphere: atmosphere, luminanceScale: 1) }
        }
        let normalized = try SkyModel(sunDirection: SIMD3(0, 3, 4), luminanceScale: 0.001)
        #expect(simd_distance(normalized.sunDirection, SIMD3(0, 0.6, 0.8)) < 1e-6)
        let position = try SolarPosition(julianDay: 2_452_930.312847, latitude: 39.742476, longitude: -105.1786)
        let placed = try SkyModel(solarPosition: position, luminanceScale: 0.001)
        #expect(simd_distance(placed.sunDirection, position.direction()) < 1e-6)
        #expect(placed.earthSunDistance == position.earthSunDistance && placed.luminanceScale == 0.001)
    }
}
