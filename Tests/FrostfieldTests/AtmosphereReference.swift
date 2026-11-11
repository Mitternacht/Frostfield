import Foundation
import Frostfield
import simd

/// An independent double-precision integration of `SkyModel`'s atmosphere for
/// tests. It marches 3D positions around the planet's center instead of the
/// shaders' (r, μ) forms, integrates every optical depth directly instead of
/// reading look-up tables, and uses Gauss–Legendre quadrature split at the
/// horizon for the sphere integral of the multiple-scattering term.
struct AtmosphereReference {
    let model: SkyModel
    let air: SkyModel.Atmosphere
    let rayleigh: [Double]
    let mieScattering: [Double]
    let mieExtinction: [Double]
    let ozone: [Double]
    let sun: SIMD3<Double>
    let originRadius: Double

    init(_ model: SkyModel) {
        self.model = model
        air = model.atmosphere
        let wavelengths = (0..<16).map { 400 + 20 * Double($0) }
        rayleigh = wavelengths.map { 1.24062e-6 / pow($0 / 1_000, 4) * model.atmosphere.airDensity }
        let column = air.aerosolScaleHeight * (1 - exp(-(air.topRadius - air.groundRadius) / air.aerosolScaleHeight))
        mieExtinction = wavelengths.map {
            model.atmosphere.aerosolOpticalDepth / column * pow($0 / 550, -model.atmosphere.aerosolAngstromExponent)
        }
        mieScattering = mieExtinction.map { $0 * model.atmosphere.aerosolSingleScatteringAlbedo }
        let peak = model.atmosphere.ozoneColumn * 2.687e20 / 15_000
        ozone = SkyModel.Spectrum.ozoneCrossSection.map { $0 * peak }
        sun = simd_normalize(SIMD3<Double>(model.sunDirection))
        originRadius = air.groundRadius + max(model.altitude, 1)
    }

    var origin: SIMD3<Double> { SIMD3(0, originRadius, 0) }

    func densities(_ point: SIMD3<Double>) -> SIMD3<Double> {
        let h = max(simd_length(point) - air.groundRadius, 0)
        let ozoneDensity: Double = if h < 10_000 || h > 40_000 { 0 } else if h < 25_000 { (h - 10_000) / 15_000 } else { (40_000 - h) / 15_000 }
        return SIMD3(exp(-h / air.rayleighScaleHeight), exp(-h / air.aerosolScaleHeight), ozoneDensity)
    }

    /// The nearest positive distance to a sphere, or nil.
    func intersect(_ point: SIMD3<Double>, _ direction: SIMD3<Double>, radius: Double) -> Double? {
        let b = simd_dot(point, direction)
        let c = simd_length_squared(point) - radius * radius
        let discriminant = b * b - c
        guard discriminant >= 0 else { return nil }
        let root = sqrt(discriminant)
        if -b - root > 0 { return -b - root }
        if -b + root > 0 { return -b + root }
        return nil
    }

    func groundDistance(_ point: SIMD3<Double>, _ direction: SIMD3<Double>) -> Double? {
        let b = simd_dot(point, direction)
        let c = simd_length_squared(point) - air.groundRadius * air.groundRadius
        guard b < 0, b * b - c >= 0 else { return nil }
        return -b - sqrt(b * b - c)
    }

    /// Density-weighted lengths from a point to the top of the atmosphere, or nil through the ground.
    func opticalLengths(from point: SIMD3<Double>, toward direction: SIMD3<Double>, steps: Int = 256) -> SIMD3<Double>? {
        if groundDistance(point, direction) != nil { return nil }
        guard let length = intersect(point, direction, radius: air.topRadius) else { return .zero }
        var sum = SIMD3<Double>.zero
        for index in 0..<steps {
            let s0 = Double(index) / Double(steps)
            let s1 = Double(index + 1) / Double(steps)
            let t = length * (s0 + s1) * (s0 + s1) / 4
            sum += densities(point + t * direction) * (length * (s1 * s1 - s0 * s0))
        }
        return sum
    }

    func spectralTransmittance(_ lengths: SIMD3<Double>?) -> [Double] {
        guard let lengths else { return Array(repeating: 0, count: 16) }
        return (0..<16).map { exp(-(rayleigh[$0] * lengths.x + mieExtinction[$0] * lengths.y + ozone[$0] * lengths.z)) }
    }

    func rayleighPhase(_ nu: Double) -> Double { 3 / (16 * .pi) * (1 + nu * nu) }

    func miePhase(_ nu: Double) -> Double {
        let g = air.aerosolAsymmetry
        return (1 - g * g) / (4 * .pi * pow(1 + g * g - 2 * g * nu, 1.5))
    }

    /// Spectral radiance per unit solar irradiance along a ray from `start` up to
    /// `limit` meters (or the atmosphere's edge), with single scattering and an
    /// optional multiple-scattering source σ_s Ψ(h). Also returns the transmittance.
    /// `phases` replaces the Rayleigh and aerosol phase values for the ray.
    func march(
        from start: SIMD3<Double>, direction: SIMD3<Double>, limit: Double? = nil, steps: Int = 1_000,
        sunSteps: Int = 128, multiple: ((Double) -> [Double])? = nil, phases override: (Double, Double)? = nil
    ) -> (radiance: [Double], transmittance: [Double]) {
        let tEnd = min(limit ?? .infinity, groundDistance(start, direction) ?? intersect(start, direction, radius: air.topRadius) ?? 0)
        let nu = simd_dot(direction, sun)
        let phases = override ?? (rayleighPhase(nu), miePhase(nu))
        var radiance = Array(repeating: 0.0, count: 16)
        var depth = Array(repeating: 0.0, count: 16)
        for index in 0..<steps {
            // A uniform midpoint rule in s with t = tEnd s² for short paths; uniform in t when a limit is given.
            let (t, dt): (Double, Double)
            if limit != nil {
                dt = tEnd / Double(steps)
                t = (Double(index) + 0.5) * dt
            } else {
                let s0 = Double(index) / Double(steps)
                let s1 = Double(index + 1) / Double(steps)
                t = tEnd * (s0 + s1) * (s0 + s1) / 4
                dt = tEnd * (s1 * s1 - s0 * s0)
            }
            let point = start + t * direction
            let density = densities(point)
            var extinction = [Double](repeating: 0, count: 16)
            for k in 0..<16 {
                extinction[k] = rayleigh[k] * density.x + mieExtinction[k] * density.y + ozone[k] * density.z
            }
            let sunTransmittance = spectralTransmittance(opticalLengths(from: point, toward: sun, steps: sunSteps))
            let psi = multiple?(simd_length(point) - air.groundRadius)
            for k in 0..<16 {
                // Optical depth to the midpoint: everything before plus half of this step.
                let transmittance = exp(-(depth[k] + 0.5 * extinction[k] * dt))
                let scatterR = rayleigh[k] * density.x
                let scatterM = mieScattering[k] * density.y
                var source = (scatterR * phases.0 + scatterM * phases.1) * sunTransmittance[k]
                if let psi { source += (scatterR + scatterM) * psi[k] }
                radiance[k] += transmittance * source * dt
                depth[k] += extinction[k] * dt
            }
        }
        return (radiance, depth.map { exp(-$0) })
    }

    /// Hillaire's Ψ_ms at an altitude for the model's Sun direction, at the point
    /// directly above the origin's vertical.
    func psi(altitude h: Double, nodes: Int = 12, azimuths: Int = 24, steps: Int = 80, sunSteps: Int = 64) -> [Double] {
        let r = air.groundRadius + h
        let point = SIMD3(0, r, 0)
        let horizon = -sqrt(max(h * (2 * air.groundRadius + h), 0)) / r
        let (abscissae, weights) = Self.gaussLegendre(nodes)
        var secondOrder = Array(repeating: 0.0, count: 16)
        var transfer = Array(repeating: 0.0, count: 16)
        // Split the cosine integral at the horizon, where the ground cuts rays off.
        for (low, high) in [(-1.0, horizon), (horizon, 1.0)] {
            for (node, weight) in zip(abscissae, weights) {
                let mu = 0.5 * (high - low) * node + 0.5 * (high + low)
                let muWeight = 0.5 * (high - low) * weight
                let sine = sqrt(max(1 - mu * mu, 0))
                for a in 0..<azimuths {
                    let phi = 2 * Double.pi * (Double(a) + 0.5) / Double(azimuths)
                    let direction = SIMD3(sine * cos(phi), mu, sine * sin(phi))
                    let solidAngle = muWeight * 2 * Double.pi / Double(azimuths)
                    let ground = groundDistance(point, direction)
                    let tEnd = ground ?? intersect(point, direction, radius: air.topRadius) ?? 0
                    var depth = Array(repeating: 0.0, count: 16)
                    for k in 0..<steps {
                        // A quadratic partition refines the dense start of each ray.
                        let s0 = Double(k) / Double(steps)
                        let s1 = Double(k + 1) / Double(steps)
                        let dt = tEnd * (s1 * s1 - s0 * s0)
                        let sample = point + tEnd * (s0 + s1) * (s0 + s1) / 4 * direction
                        let density = densities(sample)
                        let sunTransmittance = spectralTransmittance(opticalLengths(from: sample, toward: sun, steps: sunSteps))
                        for band in 0..<16 {
                            let scattering = rayleigh[band] * density.x + mieScattering[band] * density.y
                            let extinction = rayleigh[band] * density.x + mieExtinction[band] * density.y + ozone[band] * density.z
                            let transmittance = exp(-(depth[band] + 0.5 * extinction * dt))
                            secondOrder[band] += solidAngle * transmittance * scattering * sunTransmittance[band] / (4 * .pi) * dt
                            transfer[band] += solidAngle * transmittance * scattering * dt
                            depth[band] += extinction * dt
                        }
                    }
                    if let ground {
                        let hit = point + ground * direction
                        let cosine = simd_dot(simd_normalize(hit), sun)
                        if cosine > 0 {
                            let sunTransmittance = spectralTransmittance(opticalLengths(from: hit, toward: sun, steps: sunSteps))
                            for band in 0..<16 {
                                secondOrder[band] += solidAngle * exp(-depth[band]) * sunTransmittance[band] * cosine
                                    * air.groundAlbedo / .pi
                            }
                        }
                    }
                }
            }
        }
        // Both integrals carry the isotropic phase 1/4π.
        return (0..<16).map { band in
            (secondOrder[band] / (4 * .pi)) / (1 - transfer[band] / (4 * .pi))
        }
    }

    /// Spectral irradiance on an upward-facing surface at the origin from the whole
    /// sky, per unit solar irradiance. The radiance splits by phase function.
    /// Rayleigh single scattering and the isotropic multiple-scattering source
    /// integrate over Gauss–Legendre cosines and uniform azimuths about the zenith.
    /// Aerosol single scattering integrates over a midpoint grid in the
    /// Henyey–Greenstein cumulative distribution about the Sun, where the phase
    /// function cancels, so a narrow forward glow costs no extra directions.
    func skyIrradiance(
        cosines: Int = 12, azimuths: Int = 32, phaseCells: Int = 24, steps: Int = 240, sunSteps: Int = 64,
        multiple: ((Double) -> [Double])? = nil
    ) -> [Double] {
        var rays = [(direction: SIMD3<Double>, weight: Double, aerosol: Bool)]()
        let (nodes, weights) = Self.gaussLegendre(cosines)
        for (node, weight) in zip(nodes, weights) {
            let mu = 0.5 + 0.5 * node
            let sine = sqrt(max(1 - mu * mu, 0))
            for a in 0..<azimuths {
                let phi = 2 * Double.pi * (Double(a) + 0.5) / Double(azimuths)
                rays.append((SIMD3(sine * cos(phi), mu, sine * sin(phi)), 0.5 * weight * mu * 2 * .pi / Double(azimuths), false))
            }
        }
        let g = air.aerosolAsymmetry
        let helper = abs(sun.y) < 0.9 ? SIMD3<Double>(0, 1, 0) : SIMD3<Double>(1, 0, 0)
        let tangent = simd_normalize(simd_cross(helper, sun))
        let bitangent = simd_cross(sun, tangent)
        for i in 0..<phaseCells {
            let xi = (Double(i) + 0.5) / Double(phaseCells)
            // Inverse of the Henyey–Greenstein distribution of the cosine to the Sun.
            let mu = g < 1e-6 ? 2 * xi - 1 : (1 + g * g - pow((1 - g * g) / (1 - g + 2 * g * xi), 2)) / (2 * g)
            let sine = sqrt(max(1 - mu * mu, 0))
            for j in 0..<phaseCells {
                let phi = 2 * Double.pi * (Double(j) + 0.5) / Double(phaseCells)
                let direction = simd_normalize(mu * sun + sine * (cos(phi) * tangent + sin(phi) * bitangent))
                guard direction.y > 0 else { continue }
                rays.append((direction, direction.y / Double(phaseCells * phaseCells), true))
            }
        }
        let radiances = Self.concurrentMap(rays.count) { index in
            let ray = rays[index]
            let phases = ray.aerosol ? (0.0, 1.0) : (rayleighPhase(simd_dot(ray.direction, sun)), 0.0)
            return march(
                from: origin, direction: ray.direction, steps: steps, sunSteps: sunSteps,
                multiple: ray.aerosol ? nil : multiple, phases: phases
            ).radiance.map { $0 * ray.weight }
        }
        return (0..<16).map { band in radiances.reduce(0) { $0 + $1[band] } }
    }

    /// Evaluates `transform` for each index on all cores.
    static func concurrentMap(_ count: Int, _ transform: (Int) -> [Double]) -> [[Double]] {
        var results = [[Double]](repeating: [], count: count)
        results.withUnsafeMutableBufferPointer { buffer in
            guard let output = buffer.baseAddress else { return }
            DispatchQueue.concurrentPerform(iterations: count) { index in
                output[index] = transform(index)
            }
        }
        return results
    }

    /// Linear sRGB in the model's units for a spectral radiance per unit solar irradiance.
    func rgb(_ spectrum: [Double]) -> SIMD3<Double> {
        let distance2 = model.earthSunDistance * model.earthSunDistance
        let scaled = (0..<16).map { spectrum[$0] * SkyModel.Spectrum.solarIrradiance[$0] / distance2 }
        return SkyModel.Spectrum.linearSRGB(scaled) * model.luminanceScale
    }

    static func gaussLegendre(_ count: Int) -> ([Double], [Double]) {
        var nodes = [Double]()
        var weights = [Double]()
        for i in 1...count {
            var x = cos(Double.pi * (Double(i) - 0.25) / (Double(count) + 0.5))
            var derivative = 0.0
            for _ in 0..<100 {
                var p0 = 1.0
                var p1 = x
                for k in 2...max(count, 2) {
                    let p2 = ((2 * Double(k) - 1) * x * p1 - (Double(k) - 1) * p0) / Double(k)
                    p0 = p1
                    p1 = p2
                }
                if count == 1 { p1 = x; p0 = 1 }
                derivative = Double(count) * (x * p1 - p0) / (x * x - 1)
                let step = p1 / derivative
                x -= step
                if abs(step) < 1e-15 { break }
            }
            nodes.append(x)
            weights.append(2 / ((1 - x * x) * derivative * derivative))
        }
        return (nodes, weights)
    }
}
