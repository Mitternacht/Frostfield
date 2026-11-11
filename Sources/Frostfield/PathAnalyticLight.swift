import Foundation
import simd

extension PathTracer {
    /// A point, spot or directional light for the path tracer, using the raster
    /// `PunctualLight` model and units.
    ///
    /// Point and spot lights use `color × intensity / d²` incident radiance, the
    /// optional smooth range window, and the squared cosine ramp of the spot cone in the
    /// raster PBR shader. A directional light delivers `color × intensity` irradiance
    /// to a surface facing it. With a nonzero angular diameter it becomes a uniform
    /// disk of that apparent size with the same irradiance, so its shadows soften.
    /// A spot light can instead sample a finite world-space aperture.
    ///
    /// Each bounce traces one shadow ray per analytic light. Finite directional
    /// lights can also appear on escaping camera and scattered rays when their
    /// disk is visible. Both sampling strategies then use multiple importance
    /// sampling. Area lights and geometry occlude the light.
    public struct AnalyticLight: Equatable, Sendable {
        public let light: PunctualLight
        /// World position of a point light or the apex of a spot. Directional lights ignore it.
        public let position: SIMD3<Float>
        /// Unit world direction in which a spot or directional light travels. Point lights ignore it.
        public let direction: SIMD3<Float>
        /// Full apparent diameter of a directional light's disk in radians, from 0
        /// through π/2. Zero is a delta light with hard shadows; the Sun is about 0.0093.
        public let angularDiameter: Float
        /// Shows a finite directional emitter on escaping rays. A hidden
        /// background still hides it from camera rays.
        public let diskVisible: Bool
        /// Radius of a spot's sampling disk in world units. The disk faces
        /// along `direction`, at radius / tan(outer cone angle) from the apex,
        /// with a minimum tangent of 1e-5. Zero uses the authored point.
        public let spotRadius: Float
        fileprivate let spotApertureOffset: Float
        /// Unit world direction of the light's local +X, perpendicular to `direction`:
        /// horizontal angle 0° of an IES profile and the right edge of a cookie. Lights
        /// placed by position and direction keep local +Y near world +Y, as
        /// `LightRig.Light.transform()` does; scene lights use their node's frame.
        public private(set) var right: SIMD3<Float>

        /// Places a light in world space. The direction need not be unit length.
        /// Throws for a nonfinite or out-of-range position, a zero or nonfinite
        /// direction, a nonzero angular diameter on a point or spot light, or
        /// an invalid spot radius or aperture outside the supported world range.
        /// A visible disk also requires a nonzero angular diameter and finite radiance.
        public init(
            _ light: PunctualLight,
            position: SIMD3<Float> = .zero,
            direction: SIMD3<Float> = SIMD3(0, 0, -1),
            angularDiameter: Float = 0,
            spotRadius: Float = 0,
            diskVisible: Bool = false
        ) throws {
            let scale = simd_reduce_max(abs(direction))
            guard (0..<3).allSatisfy({ position[$0].isFinite && abs(position[$0]) <= 1e12 }),
                  direction.isFiniteVector, scale > 0, angularDiameter.isFinite, spotRadius.isFinite, spotRadius >= 0,
                  (0...Float.pi / 2).contains(angularDiameter) else { throw Failure.invalidAnalyticLight }
            if case .directional = light.kind {} else if angularDiameter != 0 { throw Failure.invalidAnalyticLight }
            if diskVisible {
                let spread = Self.diskSpread(angularDiameter)
                let projectedArea = Double.pi * Double(spread) * (2 - Double(spread))
                let beam = Double(light.intensity) * Double(simd_reduce_max(light.color))
                guard spread.isNormal, beam / projectedArea <= Double(Float.greatestFiniteMagnitude) / 2 else {
                    throw Failure.invalidAnalyticLight
                }
            }
            // Rectangles are emitters the tracer samples by area; see `AreaLight(_:)`.
            if case .rectangle = light.kind { throw Failure.invalidAnalyticLight }
            let forward = simd_normalize(direction / scale)
            var apertureOffset: Float = 0
            if spotRadius > 0 {
                guard case .spot(_, _, let outer) = light.kind else { throw Failure.invalidAnalyticLight }
                let setback = Double(spotRadius) / max(tan(Double(outer)), 1e-5)
                let center = SIMD3<Double>(position) + SIMD3<Double>(forward) * setback
                let extent = abs(center) + Double(spotRadius)
                guard simd_reduce_max(extent) <= 1e12 else { throw Failure.invalidAnalyticLight }
                apertureOffset = Float(setback)
            }
            self.light = light
            self.position = position
            self.direction = forward
            self.angularDiameter = angularDiameter
            self.diskVisible = diskVisible
            self.spotRadius = spotRadius
            self.spotApertureOffset = apertureOffset
            right = ResolvedPunctualLight.defaultRight(for: forward)
        }

        /// Uses a scene node's resolved world position and aim. A finite spot
        /// aperture, IES profile or cookie also keeps the node's local +X.
        public init(
            _ resolved: ResolvedPunctualLight, angularDiameter: Float = 0, spotRadius: Float = 0, diskVisible: Bool = false
        ) throws {
            try self.init(
                resolved.light, position: resolved.position, direction: resolved.direction,
                angularDiameter: angularDiameter, spotRadius: spotRadius, diskVisible: diskVisible
            )
            if spotRadius > 0 || resolved.light.iesProfile != nil || resolved.light.cookie != nil { right = resolved.right }
        }

        fileprivate static func diskSpread(_ diameter: Float) -> Float {
            // 1 − cos(α) = 2 sin²(α / 2) keeps precision for small disks.
            let quarter = sin(Double(diameter) / 4)
            return Float(2 * quarter * quarter)
        }

        /// Collects the punctual lights of a scene graph in resolve order, so the
        /// raster rig can drive the tracer. Directional lights receive the supplied
        /// angular diameter. Throws the graph's resolve errors and analytic light
        /// validation errors. The graph allows 16 lights, the tracer's limit.
        public static func lights(in graph: SceneGraph, directionalAngularDiameter: Float = 0) throws -> [AnalyticLight] {
            try graph.resolve().filter(graph.renders).compactMap(\.light).filter { resolved in
                if case .rectangle = resolved.light.kind { return false }
                return true
            }.map { resolved in
                if case .directional = resolved.light.kind {
                    return try AnalyticLight(resolved, angularDiameter: directionalAngularDiameter)
                }
                return try AnalyticLight(resolved)
            }
        }
    }

    // Six float4 slots, matching PTAnalyticLight in PathTracingShading.h. The first
    // three slots and cone.xyz use FrostfieldPunctualLight's raster packing.
    package struct PackedAnalyticLight {
        /// World position and kind: 0 directional, 1 point, 2 spot.
        var positionKind: SIMD4<Float> = .zero
        /// Unit travel direction and range, which is zero when absent.
        var directionRange: SIMD4<Float> = .zero
        var colorIntensity: SIMD4<Float> = .zero
        /// cos(inner) or directional disk visibility, max(cos(outer), 0),
        /// range flag, and 1 − cos(half angle) of a directional disk.
        var cone: SIMD4<Float> = .zero
        /// The light's local +X and one plus its IES profile layer, or zero without one.
        var frame: SIMD4<Float> = .zero
        /// Cookie layer plus one (or zero), outer cone tangent, aperture radius and axial offset.
        var tables: SIMD4<Float> = .zero

        init() {}

        init(_ analytic: AnalyticLight, profileLayer: Float, cookieLayer: Float) {
            self.init(analytic)
            frame = SIMD4(analytic.right, profileLayer)
            if case .spot(_, _, let outer) = analytic.light.kind {
                tables.x = cookieLayer
                tables.y = tan(outer)
            }
        }

        init(_ analytic: AnalyticLight) {
            let light = analytic.light
            var range: Float?
            let kind: Float
            switch light.kind {
            case .directional:
                kind = 0
                cone.x = analytic.diskVisible ? 1 : 0
                cone.w = AnalyticLight.diskSpread(analytic.angularDiameter)
            case .point(let cutoff):
                kind = 1
                range = cutoff
            case .spot(let cutoff, let inner, let outer):
                kind = 2
                range = cutoff
                cone.x = cos(inner)
                cone.y = max(cos(outer), 0)
                tables.z = analytic.spotRadius
                tables.w = analytic.spotApertureOffset
            case .rectangle:
                // Rejected by the initializer; rectangles become area lights.
                kind = 1
            }
            cone.z = range == nil ? 0 : 1
            positionKind = SIMD4(analytic.position, kind)
            directionRange = SIMD4(analytic.direction, range ?? 0)
            colorIntensity = SIMD4(light.color, light.intensity)
        }
    }

    /// Collects a scene graph's rectangle lights as camera-hidden area emitters with
    /// the same luminance, size, position and facing the raster renderer uses.
    public static func areaLights(in graph: SceneGraph, cameraVisible: Bool = false) throws -> [AreaLight] {
        try graph.resolve().filter(graph.renders).compactMap(\.light).compactMap { resolved in
            guard case .rectangle = resolved.light.kind else { return nil }
            return try AreaLight(resolved, cameraVisible: cameraVisible)
        }
    }
}

extension PathTracer.AreaLight {
    /// A rectangle light resolved through its scene node. Its front faces the node's
    /// local -Z, its width spans local X, and its radiance is the light's luminance times color.
    public init(_ resolved: ResolvedPunctualLight, cameraVisible: Bool = false) throws {
        guard case .rectangle(let width, let height) = resolved.light.kind else {
            throw PathTracer.Failure.invalidAreaLight
        }
        // The raster panel's width follows local X and its front faces local -Z, which is
        // cross(right, -up); matching the half axes keeps non-square panels aligned.
        try self.init(
            center: resolved.position, halfU: resolved.right * (width / 2), halfV: -resolved.up * (height / 2),
            radiance: resolved.light.color * resolved.light.intensity, cameraVisible: cameraVisible
        )
    }
}
