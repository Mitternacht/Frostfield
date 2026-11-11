import Foundation
import simd

/// The optics of a photographic lens: focal length, f-number, focus distance and
/// the shape of its aperture.
///
/// Lengths on the camera side are in millimeters and the focus distance is in
/// world units, which Frostfield treats as meters. The model is an ideal thin lens
/// without optical vignetting, cat-eye bokeh, aberrations or focus breathing: every
/// pixel sees the same aperture, and the field of view follows the focal length.
public struct PhysicalLens: Equatable, Sendable {
    /// The most aperture blades a lens may have.
    public static let maximumBladeCount = 16

    /// Focal length in millimeters, from 1 to 10,000.
    public var focalLength: Float
    /// Focal length divided by the entrance pupil diameter, from 0.5 to 256.
    public var fNumber: Float
    /// Distance from the lens to the plane in sharp focus along the view axis, in
    /// world units (meters). It must exceed the focal length.
    public var focusDistance: Float
    /// The number of straight aperture blades from 3 to 16, or 0 for a circular
    /// aperture. Out-of-focus points take the aperture's shape.
    public var apertureBlades: Int
    /// The aperture polygon's rotation in radians, counterclockwise in the image.
    /// At zero one corner points along the image's +X axis.
    public var apertureRotation: Float
    /// The aperture's horizontal-to-vertical extent, from 0.1 to 10. Values above one
    /// shrink the vertical extent by the ratio and values below one shrink the
    /// horizontal extent, as three-gpu-pathtracer's `anamorphicRatio` does. A 2×
    /// anamorphic lens, whose bokeh is twice as tall as wide, uses 0.5.
    public var anamorphicRatio: Float

    /// Throws when a lens parameter falls outside its documented range.
    public init(
        focalLength: Float = 50, fNumber: Float = 2.8, focusDistance: Float = 10,
        apertureBlades: Int = 0, apertureRotation: Float = 0, anamorphicRatio: Float = 1
    ) throws {
        self.focalLength = focalLength
        self.fNumber = fNumber
        self.focusDistance = focusDistance
        self.apertureBlades = apertureBlades
        self.apertureRotation = apertureRotation
        self.anamorphicRatio = anamorphicRatio
        try validate()
    }

    /// The entrance pupil diameter in millimeters: focal length divided by the f-number.
    public var apertureDiameter: Float { focalLength / fNumber }

    /// Throws `PhysicalCamera.Failure.lens` for values outside the documented ranges.
    public func validate() throws {
        guard focalLength.isFinite, (1...10_000).contains(focalLength),
              fNumber.isFinite, (0.5...256).contains(fNumber),
              focusDistance.isFinite, Double(focusDistance) > Double(focalLength) / 1000,
              focusDistance <= 1e9,
              apertureBlades == 0 || (3...Self.maximumBladeCount).contains(apertureBlades),
              apertureRotation.isFinite, abs(apertureRotation) <= 1_000,
              anamorphicRatio.isFinite, (0.1...10).contains(anamorphicRatio) else {
            throw PhysicalCamera.Failure.lens
        }
    }
}

/// A camera described the way photographers set one up: a lens, a sensor, and the
/// shutter time, ISO and exposure compensation that set its exposure.
///
/// The same camera drives the raster renderer through `perspectiveCamera(aspectRatio:)`,
/// the depth-of-field post pass through `depthOfFieldSettings(aspectRatio:maximumRadius:)`,
/// the path tracer through `pathCameraModel(aspectRatio:)`, and display exposure
/// through `displaySettings(calibration:antiAliasing:)`. Lights authored in lux,
/// candela and nits with `LightRig` then render at photographic brightness.
///
/// **Field of view.** The image covers the part of the sensor that `sensorFit`
/// selects; its vertical field of view is `2 atan(filmHeight / (2 × focalLength))`.
///
/// **Circle of confusion.** Both renderers blur a point at distance `d` into a disk
/// whose diameter on the sensor is the thin-lens value
/// `c = f² |d − s| / (N d (s − f))` for focal length `f`, f-number `N` and focus
/// distance `s`, the formula of depth-of-field calculators. The path tracer reaches
/// it with a lens of radius `f / (2N) × s / (s − f)`: it keeps the field of view of
/// the focal length rather than the slightly narrower one of a lens focused at `s`.
/// The path tracer shapes the blur as the aperture polygon; the post pass gathers a disk.
///
/// **Exposure.** `ev100` is `log2(N² / t × 100 / S)` for shutter time `t` in
/// seconds and ISO `S`, less the exposure compensation. `exposure(calibration:)`
/// turns it into the linear scale applied to scene luminance before tone mapping.
public struct PhysicalCamera: Equatable, Sendable {
    /// Which part of the sensor an image of a given aspect ratio covers.
    public enum SensorFit: Equatable, Sendable {
        /// The largest centered region of the image's aspect ratio inside the sensor, like a crop.
        case fill
        /// The sensor width spans the image width; the height follows the image aspect ratio.
        case horizontal
        /// The sensor height spans the image height; the width follows the image aspect ratio.
        case vertical
    }

    /// How an exposure value maps scene luminance to linear image values.
    ///
    /// Both use a reflected-light meter constant `K = 12.5` and ISO 100 to define
    /// EV100: a meter reads `L = 2^EV100 × 12.5 / 100` cd/m² as correctly exposed.
    public enum ExposureCalibration: Equatable, Sendable {
        /// ISO 12232 saturation-based sensitivity with lens attenuation `q = 0.65`, as
        /// in Lagarde and de Rousiers, "Moving Frostbite to PBR" (2014), and Filament:
        /// luminance `1.2 × 2^EV100` cd/m² reaches 1.0, so `exposure = 1 / (1.2 × 2^EV100)`
        /// and a metered surface lands at `1 / 9.6 ≈ 0.104`, leaving headroom for highlights.
        case saturation
        /// The metered luminance lands exactly on middle grey, 0.18:
        /// `exposure = 0.18 × 100 / (12.5 × 2^EV100) = 1.44 / 2^EV100`. A spot-metered
        /// 18% grey card then renders at 0.18 before tone mapping.
        case middleGrey
    }

    /// Full-frame 35 mm sensor dimensions in millimeters.
    public static let fullFrame = SIMD2<Float>(36, 24)
    /// The reflected-light meter calibration constant `K` in cd·s/m² per ISO.
    public static let reflectedMeterConstant: Double = 12.5
    /// The flat incident-light meter calibration constant `C` in lux·s per ISO.
    public static let incidentMeterConstant: Double = 250

    /// Lens optics used for field of view and defocus.
    public var lens: PhysicalLens
    /// Sensor width and height in millimeters, each from 1 to 1,000.
    public var sensorSize: SIMD2<Float>
    /// How the image fits the sensor dimensions.
    public var sensorFit: SensorFit
    /// Exposure time in seconds, from 1/1,000,000 to 3,600.
    public var shutterTime: Float
    /// ISO arithmetic speed, from 1 to 1,000,000.
    public var iso: Float
    /// Exposure compensation in stops, from -16 to +16; positive values brighten.
    public var exposureCompensation: Float
    /// Positive clip distance along the view axis, in world units.
    public var near: Float
    /// Clip distance beyond `near`, in world units.
    public var far: Float

    /// Throws when the optics, sensor, exposure, or clip range is invalid.
    public init(
        lens: PhysicalLens, sensorSize: SIMD2<Float> = PhysicalCamera.fullFrame, sensorFit: SensorFit = .fill,
        shutterTime: Float = 1.0 / 125, iso: Float = 100, exposureCompensation: Float = 0,
        near: Float = 0.1, far: Float = 1_000
    ) throws {
        self.lens = lens
        self.sensorSize = sensorSize
        self.sensorFit = sensorFit
        self.shutterTime = shutterTime
        self.iso = iso
        self.exposureCompensation = exposureCompensation
        self.near = near
        self.far = far
        try validate()
    }

    /// Throws for an invalid lens, sensor, exposure or clip range.
    public func validate() throws {
        try lens.validate()
        guard sensorSize.x.isFinite, sensorSize.y.isFinite,
              (1...1_000).contains(sensorSize.x), (1...1_000).contains(sensorSize.y) else { throw Failure.sensor }
        guard shutterTime.isFinite, (1e-6...3_600).contains(shutterTime),
              iso.isFinite, (1...1_000_000).contains(iso),
              exposureCompensation.isFinite, (-16...16).contains(exposureCompensation) else { throw Failure.exposure }
        guard near.isFinite, far.isFinite, near > 0, far > near else { throw Failure.clipping }
    }

    // MARK: - Projection

    /// The width and height in millimeters of the sensor region an image with this
    /// width-to-height ratio covers.
    public func filmSize(aspectRatio: Float) throws -> SIMD2<Float> {
        try validate()
        guard aspectRatio.isFinite, aspectRatio > 0 else { throw Failure.aspectRatio }
        let sensorAspect = sensorSize.x / sensorSize.y
        let fitsWidth: Bool = switch sensorFit {
        case .fill: aspectRatio >= sensorAspect
        case .horizontal: true
        case .vertical: false
        }
        let size = fitsWidth
            ? SIMD2(sensorSize.x, sensorSize.x / aspectRatio)
            : SIMD2(sensorSize.y * aspectRatio, sensorSize.y)
        guard size.x.isNormal, size.y.isNormal else { throw Failure.aspectRatio }
        return size
    }

    /// The vertical field of view in radians: `2 atan(filmHeight / (2 × focalLength))`.
    public func verticalFieldOfView(aspectRatio: Float) throws -> Float {
        let film = try filmSize(aspectRatio: aspectRatio)
        return Float(2 * atan(Double(film.y) / (2 * Double(lens.focalLength))))
    }

    /// The horizontal field of view in radians: `2 atan(filmWidth / (2 × focalLength))`.
    public func horizontalFieldOfView(aspectRatio: Float) throws -> Float {
        let film = try filmSize(aspectRatio: aspectRatio)
        return Float(2 * atan(Double(film.x) / (2 * Double(lens.focalLength))))
    }

    /// The focal length in millimeters that gives a vertical field of view on film
    /// of the given height, as three.js `PerspectiveCamera.getFocalLength` does.
    public static func focalLength(verticalFieldOfView: Float, filmHeight: Float) throws -> Float {
        guard verticalFieldOfView.isFinite, verticalFieldOfView > 0, verticalFieldOfView < .pi,
              filmHeight.isFinite, filmHeight > 0 else { throw Failure.aspectRatio }
        return Float(Double(filmHeight) / (2 * tan(Double(verticalFieldOfView) / 2)))
    }

    /// The raster camera with this lens's field of view and the camera's clip planes.
    public func perspectiveCamera(aspectRatio: Float) throws -> PerspectiveCamera {
        try PerspectiveCamera(verticalFieldOfView: verticalFieldOfView(aspectRatio: aspectRatio), near: near, far: far)
    }

    // MARK: - Depth of field

    /// The entrance pupil radius in world units: `f / (2N)`.
    public var apertureRadius: Float { lens.focalLength / (2000 * lens.fNumber) }

    /// The thin-lens circle-of-confusion diameter on the sensor in millimeters for a
    /// point at `distance` world units along the view axis:
    /// `c = f² |d − s| / (N d (s − f))`. It is zero at the focus distance.
    public func circleOfConfusionDiameter(distance: Float) throws -> Float {
        try validate()
        guard distance.isFinite, distance > 0 else { throw Failure.distance }
        let f = Double(lens.focalLength) / 1000
        let s = Double(lens.focusDistance)
        let d = Double(distance)
        return Float(1000 * f * f * abs(d - s) / (Double(lens.fNumber) * d * (s - f)))
    }

    /// The circle-of-confusion radius in pixels for an image `imageHeight` pixels
    /// tall with the given aspect ratio: half the diameter, scaled by pixels per
    /// millimeter of film. For a polygonal aperture it is the polygon's circumradius.
    public func circleOfConfusionRadius(distance: Float, imageHeight: Int, aspectRatio: Float) throws -> Float {
        guard imageHeight > 0 else { throw Failure.aspectRatio }
        let film = try filmSize(aspectRatio: aspectRatio)
        return try circleOfConfusionDiameter(distance: distance) / 2 * Float(imageHeight) / film.y
    }

    /// Settings for `DepthOfField` that blur by the same circle of confusion as the
    /// path tracer. Pair them with `perspectiveCamera(aspectRatio:)` for the same aspect.
    public func depthOfFieldSettings(aspectRatio: Float, maximumRadius: Float = 12) throws -> DepthOfField.Settings {
        let film = try filmSize(aspectRatio: aspectRatio)
        guard maximumRadius.isFinite, (0...24).contains(maximumRadius) else { throw Failure.lens }
        return DepthOfField.Settings(
            focusDistance: lens.focusDistance, fNumber: lens.fNumber, sensorHeight: film.y / 1000,
            maximumRadius: maximumRadius
        )
    }

    /// The path tracer's lens radius in world units that reproduces the thin-lens
    /// circle of confusion while keeping the focal length's field of view:
    /// `f / (2N) × s / (s − f)`.
    public var pathApertureRadius: Float {
        let f = Double(lens.focalLength) / 1000
        let s = Double(lens.focusDistance)
        return Float(f / (2 * Double(lens.fNumber)) * s / (s - f))
    }

    // MARK: - Exposure

    /// The exposure value at ISO 100 for these camera settings, less the exposure
    /// compensation: `log2(N² / t) + log2(100 / S) − compensation`.
    public var ev100: Float {
        Float(Self.ev100(fNumber: lens.fNumber, shutterTime: shutterTime, iso: iso)) - exposureCompensation
    }

    /// `log2(N² / t × 100 / S)` for an f-number, a shutter time in seconds and an ISO speed.
    public static func ev100(fNumber: Float, shutterTime: Float, iso: Float) -> Double {
        let n = Double(fNumber)
        return log2(n * n / Double(shutterTime) * 100 / Double(iso))
    }

    /// The EV100 a reflected-light meter (`K = 12.5`) reads for a surface of the given
    /// luminance in cd/m²: `log2(L × 100 / 12.5)`.
    public static func ev100(luminance: Float) throws -> Double {
        guard luminance.isFinite, luminance > 0 else { throw Failure.exposure }
        return log2(Double(luminance) * 100 / reflectedMeterConstant)
    }

    /// The EV100 a flat incident-light meter (`C = 250`) reads under the given
    /// illuminance in lux: `log2(E × 100 / 250)`.
    public static func ev100(illuminance: Float) throws -> Double {
        guard illuminance.isFinite, illuminance > 0 else { throw Failure.exposure }
        return log2(Double(illuminance) * 100 / incidentMeterConstant)
    }

    /// The shutter time in seconds that gives the EV100 at an f-number and ISO speed.
    /// Throws when an input or the resulting exposure time is outside the camera’s ranges.
    public static func shutterTime(ev100: Double, fNumber: Float, iso: Float) throws -> Float {
        let n = Double(fNumber)
        guard ev100.isFinite, (0.5...256).contains(fNumber), (1...1_000_000).contains(iso) else {
            throw Failure.exposure
        }
        let time = n * n * 100 / (Double(iso) * exp2(ev100))
        guard (1e-6...3_600).contains(time) else { throw Failure.exposure }
        return Float(time)
    }

    /// The linear factor applied to scene luminance before tone mapping.
    public func exposure(calibration: ExposureCalibration = .saturation) throws -> Float {
        try validate()
        return Float(Self.exposure(ev100: Double(ev100), calibration: calibration))
    }

    /// The linear exposure factor for an EV100 under a calibration.
    public static func exposure(ev100: Double, calibration: ExposureCalibration) -> Double {
        switch calibration {
        case .saturation: 1 / (1.2 * exp2(ev100))
        case .middleGrey: 0.18 * 100 / (reflectedMeterConstant * exp2(ev100))
        }
    }

    /// The exposure in stops, `log2(exposure(calibration:))`, for `HDRDisplay`.
    public func exposureStops(calibration: ExposureCalibration = .saturation) throws -> Float {
        log2(try exposure(calibration: calibration))
    }

    /// HDR display settings that apply this camera's exposure. Throws when the
    /// exposure falls outside the display's ±16 stops.
    public func displaySettings(
        calibration: ExposureCalibration = .saturation, antiAliasing: Bool = false
    ) throws -> HDRDisplay.Settings {
        let stops = try exposureStops(calibration: calibration)
        guard (-16...16).contains(stops) else { throw Failure.displayExposure(stops) }
        return HDRDisplay.Settings(exposureStops: stops, antiAliasing: antiAliasing)
    }

    public enum Failure: LocalizedError, Equatable {
        case lens
        case sensor
        case exposure
        case clipping
        case aspectRatio
        case distance
        case displayExposure(Float)
        case stereo

        public var errorDescription: String? {
            switch self {
            case .lens:
                "A lens needs a focal length of 1–10,000 mm, an f-number of 0.5–256, a focus distance beyond the focal length, 0 or 3–16 aperture blades, a finite rotation and an anamorphic ratio of 0.1–10."
            case .sensor: "Sensor width and height must each be 1–1,000 mm."
            case .exposure:
                "Use a shutter time of 1 µs to 3,600 s, ISO 1–1,000,000, exposure compensation within ±16 stops, and positive finite meter readings."
            case .clipping: "Clip distances must be finite, with zero < near < far."
            case .aspectRatio: "The image aspect ratio and size must be positive and finite."
            case .distance: "The subject distance must be positive and finite."
            case .displayExposure(let stops):
                "The camera's exposure is \(stops) stops; HDR display accepts -16 to +16. Adjust shutter, ISO or compensation."
            case .stereo: "A stereo pair needs a positive, finite eye separation and a convergence distance beyond the near plane."
            }
        }
    }
}

extension PhysicalCamera {
    /// One eye of a stereo pair: its camera transform and path-tracing model.
    public struct StereoView: Equatable, Sendable {
        /// The eye's local-to-world transform.
        public let cameraTransform: SceneTransform
        /// The eye's off-center projection and optional lens.
        public let cameraModel: PathTracer.CameraModel

        /// The eye's raster world-to-clip matrix with forward Metal depth, for
        /// `SceneGraph.encode(pass:commandBuffer:clipFromWorld:cameraPosition:)`. The
        /// window shift becomes an asymmetric frustum, so raster and path-traced eyes align.
        public func clipFromWorld(aspectRatio: Float) throws -> simd_float4x4 {
            guard case .perspective(let fov, let shift) = cameraModel.projection else { throw PhysicalCamera.Failure.stereo }
            var projection = try PerspectiveCamera(verticalFieldOfView: fov, near: cameraModel.near, far: cameraModel.far)
                .projection(aspectRatio: aspectRatio)
            // ndc' = ndc − shift: clip x gains shift.x × z_view, since w = −z_view.
            projection.columns.2.x = shift.x
            projection.columns.2.y = shift.y
            return projection * cameraTransform.matrix.inverse
        }
    }

    /// The path tracer's model of this camera: the focal length's field of view and,
    /// when the lens is open, a thin lens whose polygon or disk has the circumradius
    /// `pathApertureRadius`, so out-of-focus points blur by `circleOfConfusionDiameter`.
    public func pathCameraModel(aspectRatio: Float) throws -> PathTracer.CameraModel {
        let fov = try verticalFieldOfView(aspectRatio: aspectRatio)
        return PathTracer.CameraModel(
            projection: .perspective(verticalFieldOfView: fov),
            lens: .init(
                apertureRadius: pathApertureRadius, focusDistance: lens.focusDistance,
                apertureBlades: lens.apertureBlades, apertureRotation: lens.apertureRotation,
                anamorphicRatio: lens.anamorphicRatio
            ),
            near: near, far: far
        )
    }

    /// Left and right path-tracing views for a stereo still, such as a visionOS
    /// spatial photo. The eyes sit `eyeSeparation` world units apart along the
    /// camera's +X with parallel view axes. With `convergenceDistance`, each eye's
    /// window shifts so points at that distance have zero parallax, the asymmetric
    /// frustum of three.js `StereoCamera`; nil keeps zero parallax at infinity.
    public func stereoPair(
        cameraTransform: SceneTransform, eyeSeparation: Float = 0.064, convergenceDistance: Float? = nil,
        aspectRatio: Float
    ) throws -> (left: StereoView, right: StereoView) {
        guard eyeSeparation.isNormal, eyeSeparation > 0 else { throw Failure.stereo }
        if let convergenceDistance {
            guard convergenceDistance.isFinite, convergenceDistance > near else { throw Failure.stereo }
        }
        let mono = try pathCameraModel(aspectRatio: aspectRatio)
        let tanX = Float(tan(Double(try horizontalFieldOfView(aspectRatio: aspectRatio)) / 2))
        // A point on the axis at distance c sits (e / 2) / c to the side of each eye's axis.
        let shift = convergenceDistance.map { eyeSeparation / 2 / ($0 * tanX) } ?? 0
        guard shift.isFinite, abs(shift) <= 4 else { throw Failure.stereo }
        let axis = simd_normalize(cameraTransform.rotation).act(SIMD3<Float>(1, 0, 0))
        func view(side: Float) -> StereoView {
            var model = mono
            if case .perspective(let fov, _) = mono.projection {
                model.projection = .perspective(verticalFieldOfView: fov, shift: SIMD2(-side * shift, 0))
            }
            var transform = cameraTransform
            transform.translation += axis * (side * eyeSeparation / 2)
            return StereoView(cameraTransform: transform, cameraModel: model)
        }
        return (view(side: -1), view(side: 1))
    }
}
