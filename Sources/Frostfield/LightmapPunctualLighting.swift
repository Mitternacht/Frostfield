import Foundation
import simd

extension ProgressiveLightmapBaker {
    /// A world-space point, spot or rectangle light with Frostfield's native units.
    /// The baker traces shadows using its own bias and the scene geometry.
    public struct PositionedLight: Equatable, Sendable {
        public let light: PunctualLight
        public let position: SIMD3<Float>
        /// Unit emission axis for spot cones, photometric profiles and rectangles.
        public let direction: SIMD3<Float>
        /// Unit world direction of the light's local +X, including node roll.
        public private(set) var right: SIMD3<Float>

        /// Accepts point, spot and one-sided Lambertian rectangle lights, including
        /// IES profiles and spot cookies on their supported light kinds.
        /// A direction-only placement keeps the light's up axis near world +Y.
        public init(
            _ light: PunctualLight, position: SIMD3<Float>, direction: SIMD3<Float> = SIMD3(0, 0, -1)
        ) throws {
            switch light.kind {
            case .point, .spot, .rectangle: break
            case .directional: throw Failure.unsupportedPunctualLight
            }
            guard position.isFiniteVector, direction.isFiniteVector, direction != .zero else { throw Failure.punctualPose }
            self.light = light
            self.position = position
            let axis = SIMD3<Float>(simd_normalize(SIMD3<Double>(direction)))
            self.direction = axis
            self.right = ResolvedPunctualLight.defaultRight(for: axis)
        }

        /// Uses the world position, aim and roll resolved from a scene node.
        public init(_ resolved: ResolvedPunctualLight) throws {
            try self.init(resolved.light, position: resolved.position, direction: resolved.direction)
            right = resolved.right
        }
    }
}

struct LightmapPunctualLighting {
    /// Strongly retains validated CPU resources for the current prepared light list.
    private struct Photometry {
        let profile: IESProfile?
        let cookie: LightCookie?
        let right: SIMD3<Double>
        let up: SIMD3<Double>
        let cookieScale: Double

        init?(_ source: ProgressiveLightmapBaker.PositionedLight, axis: SIMD3<Double>) {
            guard source.light.iesProfile != nil || source.light.cookie != nil else { return nil }
            profile = source.light.iesProfile
            cookie = source.light.cookie
            let authoredRight = SIMD3<Double>(source.right)
            right = simd_normalize(authoredRight - axis * simd_dot(authoredRight, axis))
            up = simd_cross(right, axis)
            if case .spot(_, _, let outer) = source.light.kind {
                cookieScale = Double(tan(outer))
            } else {
                cookieScale = 0
            }
        }

        func modulation(
            toSurface: SIMD3<Double>, axis: SIMD3<Double>, cookieSize: SIMD2<Int>
        ) -> SIMD3<Double> {
            let x = simd_dot(toSurface, right)
            let y = simd_dot(toSurface, up)
            let z = simd_dot(toSurface, axis)
            var result = SIMD3<Double>(repeating: 1)
            if let profile {
                // The table stores normalized half-precision samples, as the GPU does.
                let theta = acos(min(max(z, -1), 1))
                let phi = x == 0 && y == 0 ? 0 : atan2(y, x)
                result *= Double(profile.storage.tableValue(theta: theta, phi: phi))
            }
            if let cookie {
                guard z > 0, cookieScale > 0 else { return .zero }
                let extent = z * cookieScale
                let u = Float(0.5 + 0.5 * x / extent)
                let v = Float(0.5 - 0.5 * y / extent)
                result *= SIMD3<Double>(cookie.packedTransmittance(u: u, v: v, width: cookieSize.x, height: cookieSize.y))
            }
            return result
        }
    }

    private struct Light {
        let position: SIMD3<Double>
        let direction: SIMD3<Double>
        let emitted: SIMD3<Double>
        let range: Double?
        /// Cosines packed as in the raster and path tracing shaders.
        let cone: SIMD2<Double>?
        let photometry: Photometry?
    }

    private struct Rectangle {
        private static let samplesPerAxis = 8

        let center: SIMD3<Double>
        let axis: SIMD3<Double>
        let cellRadiance: SIMD3<Double>
        let sampleOffsets: [SIMD3<Double>]

        init?(_ source: ProgressiveLightmapBaker.PositionedLight) {
            guard case .rectangle(let width, let height) = source.light.kind else { return nil }
            let radiance = SIMD3<Double>(source.light.color) * Double(source.light.intensity)
            guard radiance != .zero else { return nil }
            let count = Double(Self.samplesPerAxis)
            center = SIMD3<Double>(source.position)
            let axis = simd_normalize(SIMD3<Double>(source.direction))
            self.axis = axis
            let authoredRight = SIMD3<Double>(source.right)
            let right = simd_normalize(authoredRight - axis * simd_dot(authoredRight, axis))
            let u = right * (Double(width) / count)
            let v = simd_cross(right, axis) * (Double(height) / count)
            cellRadiance = radiance * (Double(width) * Double(height) / (count * count))
            let midpoint = (count - 1) * 0.5
            sampleOffsets = (0..<Self.samplesPerAxis).flatMap { y in
                (0..<Self.samplesPerAxis).map { x in
                    u * (Double(x) - midpoint) + v * (Double(y) - midpoint)
                }
            }
        }

        func irradiance(
            at position: SIMD3<Double>, shadowOrigin: SIMD3<Double>, normal: SIMD3<Double>,
            geometricNormal: SIMD3<Double>, visibility: MeshBVH?, ignoringTriangle: Int?, acceptsHit: MeshBVH.HitFilter?
        ) -> SIMD3<Double> {
            let toCenter = center - position
            let planeDistance = simd_dot(axis, -toCenter)
            guard planeDistance > 0 else { return .zero }
            var weight = 0.0
            for offset in sampleOffsets {
                let delta = toCenter + offset
                let squared = simd_length_squared(delta)
                guard squared > 0 else { continue }
                let distance = squared.squareRoot()
                let toSample = delta / distance
                let cosine = simd_dot(normal, toSample)
                guard cosine > 0, simd_dot(geometricNormal, toSample) > 0 else { continue }
                if let visibility, LightmapPunctualLighting.occluded(
                    from: position, shadowOrigin: shadowOrigin, to: center + offset, distance: distance,
                    visibility: visibility, ignoringTriangle: ignoringTriangle, acceptsHit: acceptsHit
                ) { continue }
                // Convert each cell's area to solid angle using the emitter cosine.
                weight += cosine * (planeDistance / distance) / squared
            }
            return cellRadiance * weight
        }
    }

    private let lights: [Light]
    private let rectangles: [Rectangle]
    /// The renderer's cookie array uses the largest width and height in the light list.
    private let cookieSize: SIMD2<Int>
    var isEmpty: Bool { lights.isEmpty && rectangles.isEmpty }

    init(_ lights: [ProgressiveLightmapBaker.PositionedLight]) {
        let cookies = lights.compactMap(\.light.cookie)
        cookieSize = SIMD2(cookies.map(\.width).max() ?? 1, cookies.map(\.height).max() ?? 1)
        rectangles = lights.compactMap(Rectangle.init)
        self.lights = lights.compactMap { light in
            let emitted = SIMD3<Double>(light.light.color) * Double(light.light.intensity)
            guard emitted != .zero else { return nil }
            let range: Float?
            let cone: SIMD2<Double>?
            switch light.light.kind {
            case .point(let cutoff):
                range = cutoff
                cone = nil
            case .spot(let cutoff, let inner, let outer):
                range = cutoff
                cone = SIMD2(Double(cos(inner)), Double(max(cos(outer), 0)))
            case .directional, .rectangle:
                return nil // Only point and spot lights use this attenuation model.
            }
            let axis = simd_normalize(SIMD3<Double>(light.direction))
            return Light(position: SIMD3<Double>(light.position), direction: axis, emitted: emitted,
                         range: range.map(Double.init), cone: cone, photometry: Photometry(light, axis: axis))
        }
    }

    func irradiance(
        at position: SIMD3<Double>, shadowOrigin: SIMD3<Double>, normal: SIMD3<Double>,
        geometricNormal: SIMD3<Double>, visibility: MeshBVH?, ignoringTriangle: Int?, acceptsHit: MeshBVH.HitFilter?
    ) -> SIMD3<Double> {
        var result = SIMD3<Double>.zero
        let limit = SIMD3<Double>(repeating: Double(Float16.greatestFiniteMagnitude))
        for light in lights {
            let delta = light.position - position
            let squared = simd_length_squared(delta)
            guard squared > 0 else { continue }
            let distance = squared.squareRoot()
            let toLight = delta / distance
            let cosine = simd_dot(normal, toLight)
            guard cosine > 0, simd_dot(geometricNormal, toLight) > 0 else { continue }
            // Same inverse-square floor and range window as pbrPunctual.
            var attenuation = 1 / max(squared, 0.0001)
            if let range = light.range {
                guard distance < range else { continue }
                let ratio = distance / range
                let ratioSquared = ratio * ratio
                attenuation *= max(0, 1 - ratioSquared * ratioSquared)
            }
            if let cone = light.cone {
                let aim = simd_dot(light.direction, -toLight)
                let width = cone.x - cone.y
                let ramp = width > 0 ? min(max((aim - cone.y) / width, 0), 1) : (aim >= cone.x ? 1.0 : 0.0)
                attenuation *= ramp * ramp
            }
            guard attenuation > 0 else { continue }
            var emitted = light.emitted
            if let photometry = light.photometry {
                emitted *= photometry.modulation(toSurface: -toLight, axis: light.direction, cookieSize: cookieSize)
                guard emitted != .zero else { continue }
            }
            if let visibility, Self.occluded(
                from: position, shadowOrigin: shadowOrigin, to: light.position, distance: distance,
                visibility: visibility, ignoringTriangle: ignoringTriangle, acceptsHit: acceptsHit
            ) { continue }
            result += simd_min(emitted * attenuation, limit) * cosine
        }
        for rectangle in rectangles {
            result += rectangle.irradiance(
                at: position, shadowOrigin: shadowOrigin, normal: normal, geometricNormal: geometricNormal,
                visibility: visibility, ignoringTriangle: ignoringTriangle, acceptsHit: acceptsHit
            )
        }
        return result
    }

    private static func occluded(
        from position: SIMD3<Double>, shadowOrigin: SIMD3<Double>, to target: SIMD3<Double>, distance: Double,
        visibility: MeshBVH, ignoringTriangle: Int?, acceptsHit: MeshBVH.HitFilter?
    ) -> Bool {
        let offset = shadowOrigin - position
        let offsetLength = simd_length(offset)
        // Keep even an unusually large user bias on the receiver's side of the
        // light. Falloff always uses the unbiased position.
        let origin = offsetLength > distance * 0.5
            ? position + offset * (distance * 0.5 / offsetLength) : shadowOrigin
        let shadowDelta = target - origin
        let shadowDistance = simd_length(shadowDelta)
        return shadowDistance > 0 && visibility.occludes(
            origin: origin, unitDirection: shadowDelta / shadowDistance,
            ignoringTriangle: ignoringTriangle, maximumDistance: shadowDistance.nextDown, acceptsHit: acceptsHit
        )
    }
}
