import Foundation
import simd

/// A physically based clear sky over a spherical Earth.
///
/// The atmosphere follows Bruneton's precomputed scattering model: Rayleigh
/// scattering with an exponential profile, aerosol (Mie) scattering and absorption
/// with an exponential profile and a Henyey–Greenstein phase function, ozone
/// absorption in a layer from 10 to 40 km, and a Lambertian ground. Sky radiance
/// adds single scattering to Hillaire's multiple-scattering approximation, which
/// sums every higher scattering order as a geometric series of an isotropic
/// second order. Light is computed at 16 wavelengths from 400 to 700 nm with the
/// ASTM G-173 extraterrestrial solar spectrum, then converted to linear sRGB
/// (Rec. 709 primaries, D65 white) through the CIE 1931 2° observer.
///
/// Units are photometric, multiplied by `luminanceScale`: sky luminance in candela
/// per square meter and sun illuminance in lux. Frostfield shades these numbers as
/// radiometric scene units, as `LightRig` does. The scale has no default because it
/// sets the units of the whole scene. A scale of 1/1000 gives kilocandela and
/// kilolux: a clear noon sky reaches about 100 kilolux, a white surface about 30
/// units, and an `HDRDisplay` exposure near −5 stops displays it. A scale of one
/// keeps physical units for an exposure near −15 stops (EV 15, "sunny 16"), but
/// hazy skies then exceed half-float range near the Sun and `SkyEnvironment` throws.
/// `LightRig` documents its intensities in lux, candela and nits, so multiply every
/// other light's intensity or power by the same scale to keep their balance with
/// the Sun and sky.
///
/// The Sun itself is not part of the sky radiance. `SkyEnvironment` turns it into a
/// directional `LightRig` light carrying the transmitted solar illuminance and the
/// Sun's angular size, so raster and path-traced lighting both add sky and Sun once.
///
/// - [Bruneton, "Precomputed Atmospheric Scattering: a New Implementation" (2017)](https://ebruneton.github.io/precomputed_atmospheric_scattering/)
/// - [Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique" (EGSR 2020)](https://sebh.github.io/publications/egsr2020.pdf)
public struct SkyModel: Equatable, Sendable {
    /// Physical constituents of the atmosphere. Defaults are the reference values
    /// of Bruneton's 2017 implementation: a clean, dry Earth atmosphere.
    public struct Atmosphere: Equatable, Sendable {
        /// Planet radius in meters.
        public var groundRadius: Double
        /// Radius of the atmosphere's outer boundary in meters.
        public var topRadius: Double
        /// Multiplies the ground-level Rayleigh scattering of standard air,
        /// 1.24062 × 10⁻⁶ λ⁻⁴ per meter with λ in micrometers.
        public var airDensity: Double
        /// Rayleigh density scale height in meters.
        public var rayleighScaleHeight: Double
        /// Vertical aerosol optical depth at 550 nm. About 0.05 is very clear,
        /// 0.1–0.2 a typical clear day and 0.5 hazy.
        public var aerosolOpticalDepth: Double
        /// Ångström exponent of aerosol extinction, `τ(λ) = τ(550 nm) (λ / 550 nm)^−α`.
        public var aerosolAngstromExponent: Double
        /// Aerosol density scale height in meters.
        public var aerosolScaleHeight: Double
        /// Fraction of aerosol extinction that scatters rather than absorbs.
        public var aerosolSingleScatteringAlbedo: Double
        /// Henyey–Greenstein asymmetry of aerosol scattering, from 0 up to 0.99.
        public var aerosolAsymmetry: Double
        /// Total ozone column in Dobson units.
        public var ozoneColumn: Double
        /// Lambertian reflectance of the ground, from 0 to 1, for all wavelengths.
        public var groundAlbedo: Double

        public init(
            groundRadius: Double = 6_360_000, topRadius: Double = 6_420_000, airDensity: Double = 1,
            rayleighScaleHeight: Double = 8_000, aerosolOpticalDepth: Double = 0.005328,
            aerosolAngstromExponent: Double = 0, aerosolScaleHeight: Double = 1_200,
            aerosolSingleScatteringAlbedo: Double = 0.9, aerosolAsymmetry: Double = 0.8,
            ozoneColumn: Double = 300, groundAlbedo: Double = 0.1
        ) {
            self.groundRadius = groundRadius
            self.topRadius = topRadius
            self.airDensity = airDensity
            self.rayleighScaleHeight = rayleighScaleHeight
            self.aerosolOpticalDepth = aerosolOpticalDepth
            self.aerosolAngstromExponent = aerosolAngstromExponent
            self.aerosolScaleHeight = aerosolScaleHeight
            self.aerosolSingleScatteringAlbedo = aerosolSingleScatteringAlbedo
            self.aerosolAsymmetry = aerosolAsymmetry
            self.ozoneColumn = ozoneColumn
            self.groundAlbedo = groundAlbedo
        }

        /// An atmosphere with Linke-style turbidity `T`, the ratio of total to
        /// molecular optical depth at 550 nm that Preetham's sky model defines. The
        /// aerosol optical depth becomes `(T − 1) τ_Rayleigh(550 nm)` with an Ångström
        /// exponent of 1.3, Preetham's assumption. Turbidity runs from 1 (pure air)
        /// to 20; 2 is a clear day, and 10 is an aerosol optical depth near 0.98,
        /// heavy haze.
        ///
        /// Three.js `Sky.turbidity` is not this quantity: its shader scales Mie
        /// scattering linearly with turbidity times a separate `mieCoefficient`, so
        /// its values do not carry over. Match haze by appearance or by aerosol
        /// optical depth instead.
        public init(turbidity: Double, groundAlbedo: Double = 0.1) throws {
            guard turbidity.isFinite, (1...20).contains(turbidity) else { throw Failure.atmosphere("turbidity must be 1 to 20") }
            self.init(aerosolAngstromExponent: 1.3, groundAlbedo: groundAlbedo)
            aerosolOpticalDepth = (turbidity - 1) * rayleighOpticalDepth(wavelength: 550)
        }

        /// Vertical Rayleigh optical depth from the ground to space at a wavelength in nanometers.
        public func rayleighOpticalDepth(wavelength: Double) -> Double {
            Self.rayleighCoefficient(wavelength) * airDensity * rayleighScaleHeight
                * (1 - exp(-(topRadius - groundRadius) / rayleighScaleHeight))
        }

        /// Ground-level Rayleigh scattering of standard air per meter at a wavelength in nanometers.
        static func rayleighCoefficient(_ wavelength: Double) -> Double {
            1.24062e-6 / pow(wavelength / 1_000, 4)
        }

        func validate() throws {
            func check(_ value: Double, _ range: ClosedRange<Double>, _ name: String) throws {
                guard value.isFinite, range.contains(value) else { throw Failure.atmosphere(name) }
            }
            try check(groundRadius, 1_000...1e9, "groundRadius")
            try check(topRadius - groundRadius, 1_000...1e6, "topRadius − groundRadius (1 to 1,000 km)")
            try check(airDensity, 0...100, "airDensity")
            try check(rayleighScaleHeight, 10...1e5, "rayleighScaleHeight")
            try check(aerosolOpticalDepth, 0...10, "aerosolOpticalDepth")
            try check(aerosolAngstromExponent, -1...4, "aerosolAngstromExponent")
            try check(aerosolScaleHeight, 10...1e5, "aerosolScaleHeight")
            try check(aerosolSingleScatteringAlbedo, 0...1, "aerosolSingleScatteringAlbedo")
            try check(aerosolAsymmetry, 0...0.99, "aerosolAsymmetry")
            try check(ozoneColumn, 0...1_000, "ozoneColumn")
            try check(groundAlbedo, 0...1, "groundAlbedo")
        }
    }

    // The properties are constants so that every model has passed `init`'s
    // validation; build a new model to change one.

    /// Unit world direction from the observer toward the Sun's center, +Y up.
    public let sunDirection: SIMD3<Float>
    public let atmosphere: Atmosphere
    /// Altitude of the world origin above the ground in meters. World units are
    /// meters and +Y points away from the planet's center. Values below one meter
    /// are evaluated at one meter so the horizon stays defined.
    public let altitude: Double
    /// Earth–Sun distance in astronomical units. Solar irradiance falls with its
    /// square and the Sun's apparent radius shrinks with it.
    public let earthSunDistance: Double
    /// Multiplies every radiance and illuminance: 0.001 gives kilocandela per square
    /// meter and kilolux, and one keeps physical units.
    public let luminanceScale: Double

    /// Validates the atmosphere, altitude, distance and scale, and normalizes the Sun
    /// direction. `luminanceScale` sets the scene's units; see the type's discussion.
    public init(
        sunDirection: SIMD3<Float>, atmosphere: Atmosphere = .init(), altitude: Double = 0,
        earthSunDistance: Double = 1, luminanceScale: Double
    ) throws {
        let length = simd_length(sunDirection)
        guard length.isFinite, length > 1e-6 else { throw Failure.sunDirection }
        try atmosphere.validate()
        guard altitude.isFinite, altitude >= 0, altitude < atmosphere.topRadius - atmosphere.groundRadius - 1 else {
            throw Failure.altitude
        }
        guard earthSunDistance.isFinite, (0.5...2).contains(earthSunDistance) else { throw Failure.earthSunDistance }
        guard luminanceScale.isFinite, luminanceScale > 0, luminanceScale <= 1e6 else { throw Failure.luminanceScale }
        self.sunDirection = sunDirection / length
        self.atmosphere = atmosphere
        self.altitude = altitude
        self.earthSunDistance = earthSunDistance
        self.luminanceScale = luminanceScale
    }

    /// Places the Sun at a computed solar position, using its refracted direction
    /// (north −Z, east +X) and Earth–Sun distance.
    public init(
        solarPosition: SolarPosition, atmosphere: Atmosphere = .init(), altitude: Double = 0,
        luminanceScale: Double
    ) throws {
        try self.init(
            sunDirection: solarPosition.direction(), atmosphere: atmosphere, altitude: altitude,
            earthSunDistance: solarPosition.earthSunDistance, luminanceScale: luminanceScale
        )
    }

    /// Angular radius of the solar disk in radians, from the IAU nominal solar
    /// radius of 695,700 km: about 0.00465 at one astronomical unit.
    public var sunAngularRadius: Double {
        asin(695_700 / (earthSunDistance * 149_597_870.7))
    }

    /// The direct solar beam at the world origin in linear sRGB: illuminance on a
    /// surface facing the Sun, after extinction along the path through the
    /// atmosphere, in lux times `luminanceScale`. Zero once the Sun's center is
    /// below the horizon. Atmospheric refraction of the beam is ignored, and
    /// components outside the sRGB gamut clamp to zero.
    public var sunIlluminance: SIMD3<Float> {
        SIMD3<Float>(sunIlluminance(altitude: altitude))
    }

    /// The direct solar beam at an altitude in meters above the ground.
    package func sunIlluminance(altitude height: Double) -> SIMD3<Double> {
        let transmittance = sunTransmittance(altitude: height)
        let irradiance = Spectrum.solarIrradiance.indices.map {
            Spectrum.solarIrradiance[$0] * transmittance[$0] / (earthSunDistance * earthSunDistance)
        }
        // A deep red sun near the horizon lies outside the sRGB gamut; drop negative components.
        return simd_max(Spectrum.linearSRGB(irradiance), .zero) * luminanceScale
    }

    /// Spectral transmittance from an altitude toward the Sun's center, at each of
    /// `Spectrum.wavelengths`. Integrates the density profiles numerically in
    /// double precision. Zero where the planet blocks the Sun's center.
    package func sunTransmittance(altitude height: Double) -> [Double] {
        transmittance(altitude: height, cosine: Double(sunDirection.y))
    }

    /// Spectral transmittance from an altitude to the top of the atmosphere along a
    /// ray with the given cosine to the local vertical; zero if it meets the ground.
    package func transmittance(altitude height: Double, cosine: Double) -> [Double] {
        let r = atmosphere.groundRadius + max(height, 1)
        let lengths = Self.opticalLengths(atmosphere: atmosphere, radius: r, cosine: cosine)
        guard let lengths else { return Array(repeating: 0, count: Spectrum.count) }
        let coefficients = self.coefficients
        return (0..<Spectrum.count).map { index in
            exp(-(coefficients.rayleigh[index] * lengths.x + coefficients.mieExtinction[index] * lengths.y
                + coefficients.ozone[index] * lengths.z))
        }
    }

    /// Density-weighted path lengths in meters (Rayleigh, aerosol, ozone) from a
    /// radius toward the top of the atmosphere along a ray with the given cosine
    /// to the local vertical, or nil when the ray meets the ground.
    package static func opticalLengths(atmosphere: Atmosphere, radius: Double, cosine: Double, steps: Int = 2_000) -> SIMD3<Double>? {
        let bottom = atmosphere.groundRadius
        let top = atmosphere.topRadius
        let height = radius - bottom
        let mu = simd_clamp(cosine, -1, 1)
        if mu < 0, radius * radius * mu * mu - height * (2 * bottom + height) >= 0 { return nil }
        let discriminant = radius * radius * mu * mu + (top - radius) * (top + radius)
        let length = -radius * mu + sqrt(max(discriminant, 0))
        var sum = SIMD3<Double>.zero
        // Midpoints of a quadratic partition put more samples near the dense start.
        for index in 0..<steps {
            let s0 = Double(index) / Double(steps)
            let s1 = Double(index + 1) / Double(steps)
            let t = length * 0.25 * (s0 + s1) * (s0 + s1)
            let dt = length * (s1 * s1 - s0 * s0)
            let r2MinusBottom2 = t * t + 2 * radius * mu * t + height * (2 * bottom + height)
            let altitude = max(r2MinusBottom2, 0) / (sqrt(max(r2MinusBottom2, 0) + bottom * bottom) + bottom)
            sum += density(atmosphere: atmosphere, altitude: altitude) * dt
        }
        return sum
    }

    /// Relative Rayleigh, aerosol and ozone densities at an altitude in meters.
    package static func density(atmosphere: Atmosphere, altitude: Double) -> SIMD3<Double> {
        let h = max(altitude, 0)
        return SIMD3(
            exp(-h / atmosphere.rayleighScaleHeight),
            exp(-h / atmosphere.aerosolScaleHeight),
            max(0, 1 - abs(h - 25_000) / 15_000)
        )
    }

    /// A directional light for `LightRig` that carries the transmitted solar beam:
    /// the Sun's angular diameter, its color normalized to a largest component of
    /// one, and its largest RGB illuminance component as the intensity. Positioned
    /// one unit toward the Sun from `target`. The path tracer sees its disk
    /// on escaping rays. Below the horizon its intensity is zero.
    ///
    /// The intensity is lux times `luminanceScale`, not lux as `LightRig` documents:
    /// kilolux at a scale of 0.001. Multiply the intensity or power of every other
    /// light in the rig by `luminanceScale` too.
    public func sunLight(id: String = "sun", target: SIMD3<Float> = .zero, castsShadow: Bool = true) -> LightRig.Light {
        let illuminance = sunIlluminance
        let peak = simd_reduce_max(illuminance)
        let color = peak > 0 ? simd_clamp(illuminance / peak, SIMD3(repeating: 0), SIMD3(repeating: 1)) : SIMD3(repeating: 1)
        return LightRig.Light(
            id: id, kind: .directional(angularDiameter: Float(2 * sunAngularRadius)),
            position: target + sunDirection, target: target, color: color, emission: .intensity(max(peak, 0)),
            castsShadow: castsShadow, isVisibleToCamera: true
        )
    }

    /// Per-wavelength extinction and scattering coefficients per meter at unit density.
    package struct Coefficients: Equatable, Sendable {
        package var rayleigh: [Double]
        package var mieScattering: [Double]
        package var mieExtinction: [Double]
        /// Ozone absorption at the layer's peak density.
        package var ozone: [Double]
    }

    package var coefficients: Coefficients {
        let air = atmosphere
        let wavelengths = Spectrum.wavelengths
        let rayleigh = wavelengths.map { Atmosphere.rayleighCoefficient($0) * air.airDensity }
        // A vertical column of exponential density integrates to H(1 − e^(−top/H)).
        let column = air.aerosolScaleHeight * (1 - exp(-(air.topRadius - air.groundRadius) / air.aerosolScaleHeight))
        let extinction = wavelengths.map {
            air.aerosolOpticalDepth / column * pow($0 / 550, -air.aerosolAngstromExponent)
        }
        // Dobson units are 2.687 × 10²⁰ molecules per square meter; the tent profile
        // from 10 to 40 km integrates to 15 km at unit peak density.
        let peakDensity = air.ozoneColumn * 2.687e20 / 15_000
        return Coefficients(
            rayleigh: rayleigh,
            mieScattering: extinction.map { $0 * air.aerosolSingleScatteringAlbedo },
            mieExtinction: extinction,
            ozone: Spectrum.ozoneCrossSection.map { $0 * peakDensity }
        )
    }

    public enum Failure: LocalizedError, Equatable {
        case sunDirection
        case atmosphere(String)
        case altitude
        case earthSunDistance
        case luminanceScale

        public var errorDescription: String? {
            switch self {
            case .sunDirection: "The Sun direction must be finite and nonzero."
            case .atmosphere(let detail): "The atmosphere parameter is out of range: \(detail)."
            case .altitude: "The observer altitude must be at least zero meters and below the top of the atmosphere."
            case .earthSunDistance: "The Earth–Sun distance must be between 0.5 and 2 astronomical units."
            case .luminanceScale: "The luminance scale must be positive, finite and at most 1,000,000."
            }
        }
    }
}

extension SkyModel {
    /// The spectral sampling shared by the CPU and GPU sky.
    ///
    /// Sixteen wavelengths from 400 to 700 nm at 20 nm steps. A spectrum is the
    /// piecewise-linear interpolation of its samples, held constant below 400 and
    /// above 700 nm, so converting it to color integrates each sample's hat function
    /// against the color-matching functions from 360 to 830 nm. The solar spectrum
    /// and ozone cross sections are the 10 nm tables of Bruneton's reference
    /// implementation, averaged over each 20 nm hat.
    package enum Spectrum {
        package static let count = 16
        package static let wavelengths: [Double] = (0..<count).map { 400 + 20 * Double($0) }

        // Solar spectral irradiance at the top of the atmosphere in W·m⁻²·nm⁻¹, 10 nm
        // bins from 360 to 830 nm, from ASTM G-173 as tabulated in Bruneton's
        // precomputed_atmospheric_scattering demo (BSD 3-clause, © 2017 Eric Bruneton).
        private static let solarTable: [Double] = [
            1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253,
            1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
            1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533,
            1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482,
            1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758, 1.2367, 1.2082,
            1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992,
        ]

        // Ozone absorption cross sections in m² at 233 K for the same bins, from the
        // same source (Serdyuchenko et al. 2014 via Bruneton).
        private static let ozoneTable: [Double] = [
            1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27,
            8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26,
            1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
            4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25,
            2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26,
            6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26,
            2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27,
        ]

        /// The 10 nm table averaged over each sample's 20 nm hat.
        private static func hatAverage(_ table: [Double]) -> [Double] {
            wavelengths.map { wavelength in
                let index = Int((wavelength - 360) / 10)
                return (0.5 * table[index - 1] + table[index] + 0.5 * table[index + 1]) / 2
            }
        }

        /// Solar spectral irradiance at one astronomical unit, W·m⁻²·nm⁻¹.
        package static let solarIrradiance = hatAverage(solarTable)
        /// Ozone absorption cross section in square meters.
        package static let ozoneCrossSection = hatAverage(ozoneTable)

        /// The luminous efficacy of 555 nm light, lumens per watt.
        package static let luminousEfficacy = 683.0

        /// CIE 1931 2° color-matching functions integrated against each sample's hat,
        /// in nanometers. Uses the multi-lobe fit of Wyman, Sloan and Shirley (2013),
        /// as `LightRig.Light.color(temperature:)` does.
        package static let colorMatching: [SIMD3<Double>] = {
            func gaussian(_ x: Double, _ mu: Double, _ low: Double, _ high: Double) -> Double {
                let t = (x - mu) / (x < mu ? low : high)
                return exp(-0.5 * t * t)
            }
            var result = Array(repeating: SIMD3<Double>.zero, count: count)
            let step = 0.25
            var lambda = 360 + step / 2
            while lambda < 830 {
                let x = 1.056 * gaussian(lambda, 599.8, 37.9, 31.0) + 0.362 * gaussian(lambda, 442.0, 16.0, 26.7)
                    - 0.065 * gaussian(lambda, 501.1, 20.4, 26.2)
                let y = 0.821 * gaussian(lambda, 568.8, 46.9, 40.5) + 0.286 * gaussian(lambda, 530.9, 16.3, 31.1)
                let z = 1.217 * gaussian(lambda, 437.0, 11.8, 36.0) + 0.681 * gaussian(lambda, 459.0, 26.0, 13.8)
                let position = (lambda - 400) / 20
                for index in 0..<count {
                    let weight: Double
                    if (index == 0 && position <= 0) || (index == count - 1 && position >= Double(count - 1)) {
                        weight = 1
                    } else {
                        weight = max(0, 1 - abs(position - Double(index)))
                    }
                    if weight > 0 { result[index] += SIMD3(x, y, z) * (weight * step) }
                }
                lambda += step
            }
            return result
        }()

        /// CIE XYZ to linear sRGB with D65 white.
        package static let xyzToLinearSRGB = simd_double3x3(rows: [
            SIMD3(3.2404542, -1.5371385, -0.4985314),
            SIMD3(-0.9692660, 1.8760108, 0.0415560),
            SIMD3(0.0556434, -0.2040259, 1.0572252),
        ])

        /// Converts spectral radiance (W·m⁻²·sr⁻¹·nm⁻¹) to linear sRGB luminance
        /// units (cd·m⁻²), or spectral irradiance to lux. Out-of-gamut components may be negative.
        package static func linearSRGB(_ spectrum: [Double]) -> SIMD3<Double> {
            var xyz = SIMD3<Double>.zero
            for index in 0..<count { xyz += colorMatching[index] * spectrum[index] }
            return xyzToLinearSRGB * xyz * luminousEfficacy
        }

        /// Per-sample linear sRGB weights, `linearSRGB(s) = Σ weights[i] s[i]`.
        package static let rgbWeights: [SIMD3<Double>] = colorMatching.map { xyzToLinearSRGB * $0 * luminousEfficacy }

        /// Exponent of the power-law limb darkening `I(μ) / I(1) = μ^α` at each
        /// sample, with `α = 0.292 / λ − 0.023` (λ in micrometers), a fit to the
        /// visible-band exponents of Hestroffer and Magnan (1998).
        package static let limbDarkening: [Double] = wavelengths.map { 0.292 / ($0 / 1_000) - 0.023 }
    }
}
