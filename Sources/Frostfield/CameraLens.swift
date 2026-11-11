import Foundation

/// Physical thin-lens values for a camera: focal length, aperture, focus distance
/// and sensor width, with the derived field of view and circle of confusion.
///
/// Lengths on the lens side are millimeters, as printed on real lenses and sensors.
/// Scene distances are world units, converted with `metersPerUnit`. The image is
/// assumed to span the sensor width (horizontal sensor fit, as in Blender's default).
public struct CameraLens: Hashable, Sendable {
    /// Focal length in millimeters.
    public var focalLength: Float
    /// Focal length divided by aperture diameter. Smaller values blur more.
    public var fNumber: Float
    /// Distance along the view axis to the plane in focus, in world units.
    public var focusDistance: Float
    /// Sensor width in millimeters; 36 is full frame and 23.6 is APS-C.
    public var sensorWidth: Float
    /// World units to meters; 1 for meter-scale scenes, 0.01 for centimeters.
    public var metersPerUnit: Float

    /// Validates finite positive values and a focus distance beyond the focal length.
    public init(
        focalLength: Float = 50, fNumber: Float = 2.8, focusDistance: Float, sensorWidth: Float = 36,
        metersPerUnit: Float = 1
    ) throws {
        self.focalLength = focalLength
        self.fNumber = fNumber
        self.focusDistance = focusDistance
        self.sensorWidth = sensorWidth
        self.metersPerUnit = metersPerUnit
        try validate()
    }

    /// A lens whose horizontal sensor fit reproduces a `PerspectiveCamera`'s vertical
    /// field of view at the given aspect ratio.
    public init(
        matching camera: PerspectiveCamera, aspectRatio: Float, fNumber: Float, focusDistance: Float,
        sensorWidth: Float = 36, metersPerUnit: Float = 1
    ) throws {
        guard aspectRatio.isFinite, aspectRatio > 0 else { throw Failure.invalidValues }
        let focal = Double(sensorWidth) / (2 * Double(aspectRatio) * tan(Double(camera.verticalFieldOfView) / 2))
        try self.init(
            focalLength: Float(focal), fNumber: fNumber, focusDistance: focusDistance, sensorWidth: sensorWidth,
            metersPerUnit: metersPerUnit
        )
    }

    /// Throws unless every value is finite and positive and focus lies beyond the focal length.
    public func validate() throws {
        guard [focalLength, fNumber, focusDistance, sensorWidth, metersPerUnit].allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw Failure.invalidValues
        }
        guard Double(focusDistance) * Double(metersPerUnit) * 1000 > Double(focalLength) else {
            throw Failure.focusInsideFocalLength
        }
    }

    /// Aperture diameter in millimeters.
    public var apertureDiameter: Float { focalLength / fNumber }

    /// Horizontal angle of view in radians.
    public var horizontalFieldOfView: Float {
        Float(2 * atan(Double(sensorWidth) / (2 * Double(focalLength))))
    }

    /// Vertical angle of view in radians for an image of the given width-to-height ratio.
    public func verticalFieldOfView(aspectRatio: Float) -> Float {
        Float(2 * atan(Double(sensorWidth) / (2 * Double(focalLength) * Double(aspectRatio))))
    }

    /// A camera with this lens's field of view and the given clip planes.
    public func perspectiveCamera(aspectRatio: Float, near: Float, far: Float) throws -> PerspectiveCamera {
        guard aspectRatio.isFinite, aspectRatio > 0 else { throw Failure.invalidValues }
        return try PerspectiveCamera(verticalFieldOfView: verticalFieldOfView(aspectRatio: aspectRatio), near: near, far: far)
    }

    /// Thin-lens circle-of-confusion diameter on the sensor, in millimeters, for a
    /// point at `distance` world units along the view axis:
    /// `c = A · f · |d − s| / (d · (s − f))`, with aperture `A = f / N`.
    public func circleOfConfusionDiameter(atDistance distance: Float) -> Float {
        Float(sensorBlurScale * abs(1 - Double(focusDistance) / Double(distance)))
    }

    /// The signed circle-of-confusion radius in pixels for an image `imageWidth`
    /// pixels wide: negative in front of the focus plane, positive behind it.
    public func circleOfConfusionRadius(atDistance distance: Float, imageWidth: Int) -> Float {
        Float(pixelRadiusScale(imageWidth: imageWidth) * (1 - Double(focusDistance) / Double(distance)))
    }

    /// `K` in `radius(z) = K · (1 − s / z)`: the pixel radius approached far behind focus.
    func pixelRadiusScale(imageWidth: Int) -> Double {
        sensorBlurScale * Double(imageWidth) / (2 * Double(sensorWidth))
    }

    /// `A · f / (s − f)` in millimeters: the sensor blur diameter approached far behind focus.
    private var sensorBlurScale: Double {
        let f = Double(focalLength)
        let s = Double(focusDistance) * Double(metersPerUnit) * 1000
        return f / Double(fNumber) * f / (s - f)
    }

    /// The focus distance in world units beyond which everything to infinity is
    /// acceptably sharp: `H = f² / (N · c) + f`, for a circle of confusion `c` in millimeters.
    public func hyperfocalDistance(circleOfConfusion: Float = 0.03) -> Float {
        let f = Double(focalLength)
        let millimeters = f * f / (Double(fNumber) * Double(circleOfConfusion)) + f
        return Float(millimeters / 1000 / Double(metersPerUnit))
    }

    /// The nearest and farthest distances, in world units, whose blur stays within
    /// `circleOfConfusion` millimeters. The far limit is infinite at or beyond the hyperfocal distance.
    public func depthOfFieldLimits(circleOfConfusion: Float = 0.03) -> (near: Float, far: Float) {
        let f = Double(focalLength)
        let s = Double(focusDistance) * Double(metersPerUnit) * 1000
        let h = f * f / (Double(fNumber) * Double(circleOfConfusion)) + f
        let near = s * (h - f) / (h + s - 2 * f)
        let far = s < h ? s * (h - f) / (h - s) : .infinity
        let scale = 1 / (1000 * Double(metersPerUnit))
        return (Float(near * scale), Float(far * scale))
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidValues
        case focusInsideFocalLength

        public var errorDescription: String? {
            switch self {
            case .invalidValues:
                "Lens focal length, f-number, focus distance, sensor width, scale and aspect ratio must be finite and positive."
            case .focusInsideFocalLength: "The focus distance must lie beyond the focal length."
            }
        }
    }
}
