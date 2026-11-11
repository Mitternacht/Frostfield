import Foundation
import Metal
import simd

extension PathTracer {
    /// How the path tracer turns pixels into camera rays: a projection, an optional
    /// thin lens, and clip distances. The camera transform supplies position and
    /// orientation; the camera looks down its local −Z with +Y up, as elsewhere.
    public struct CameraModel: Equatable, Sendable {
        /// The image projection used to generate primary rays.
        public enum Projection: Equatable, Sendable {
            /// A pinhole, or a thin lens with `lens`, through a window `2 tan(fov / 2)`
            /// tall at unit distance whose width follows the image aspect ratio.
            /// `shift` moves the window in units of its half width and half height,
            /// positive toward +X and +Y, like a shift lens or three.js `filmOffset`;
            /// stereo pairs use it to converge. Each component must be within ±4.
            case perspective(verticalFieldOfView: Float, shift: SIMD2<Float> = .zero)
            /// Parallel rays along the view axis from a window `height` world units
            /// tall, whose width follows the image aspect ratio.
            case orthographic(height: Float)
            /// A 360° × 180° latitude-longitude image from one point. The center column
            /// looks along the camera's −Z, the column a quarter of the width to its
            /// right looks along +X, the top row looks along +Y, and the left and right
            /// edges look backward. Use a 2:1 image for square pixels.
            case equirectangular
            /// Omnidirectional stereo (ODS) for VR stills: two latitude-longitude images
            /// stacked top-bottom, the left eye on top. Each column's ray starts on a
            /// circle of diameter `interpupillaryDistance` around the camera, tangent to
            /// the circle, so every viewing direction sees correct horizontal parallax.
            /// Use an image twice as tall as an eye, such as 4,096 × 4,096 for two
            /// 4,096 × 2,048 eyes; the height must be even. Parallax does not fade
            /// toward the poles.
            case stereoEquirectangular(interpupillaryDistance: Float)
        }

        /// A thin lens for perspective projections, with a round or polygonal aperture.
        public struct Lens: Equatable, Sendable {
            /// The aperture's circumradius in world units. Zero is a pinhole.
            public var apertureRadius: Float
            /// The positive distance to the focal plane along the view axis.
            public var focusDistance: Float
            /// 0 for a round aperture, or 3 to 16 straight blades.
            public var apertureBlades: Int
            /// The polygon's rotation in radians, counterclockwise in the image; at zero
            /// a corner points along the image's +X.
            public var apertureRotation: Float
            /// Horizontal-to-vertical aperture extent from 0.1 to 10; see `PhysicalLens.anamorphicRatio`.
            public var anamorphicRatio: Float

            /// Defines the aperture; the tracer validates it when encoding.
            public init(
                apertureRadius: Float, focusDistance: Float, apertureBlades: Int = 0, apertureRotation: Float = 0,
                anamorphicRatio: Float = 1
            ) {
                self.apertureRadius = apertureRadius
                self.focusDistance = focusDistance
                self.apertureBlades = apertureBlades
                self.apertureRotation = apertureRotation
                self.anamorphicRatio = anamorphicRatio
            }

            var isValid: Bool {
                apertureRadius.isFinite && apertureRadius >= 0 && apertureRadius <= 1e6
                    && focusDistance.isNormal && focusDistance > 0
                    && (apertureBlades == 0 || (3...PhysicalLens.maximumBladeCount).contains(apertureBlades))
                    && apertureRotation.isFinite && abs(apertureRotation) <= 1_000
                    && anamorphicRatio.isFinite && (0.1...10).contains(anamorphicRatio)
            }

            /// Blade count, rotation, and the horizontal and vertical aperture scales.
            var shape: SIMD4<Float> {
                SIMD4(Float(apertureBlades), apertureRotation, min(anamorphicRatio, 1), min(1 / anamorphicRatio, 1))
            }
        }

        /// Maps the image plane to directions or parallel ray origins.
        public var projection: Projection
        /// A thin lens; perspective projections only. Nil is a pinhole.
        public var lens: Lens?
        /// Positive distance at which camera rays start: along the view axis for
        /// planar projections and along the ray for panoramas.
        public var near: Float
        /// Distance beyond `near` at which camera rays stop, measured like `near`.
        public var far: Float
        // Dynamic preview rounds the traced grid independently of the projection.
        var projectionAspectRatio: Float?

        /// Defines a camera model; the tracer validates it against the render size.
        public init(projection: Projection, lens: Lens? = nil, near: Float = 0.01, far: Float = 1e6) {
            self.projection = projection
            self.lens = lens
            self.near = near
            self.far = far
        }

        /// The model of `PathTracer.encode(camera:...)`: a raster camera with the
        /// settings' round thin lens.
        init(perspective camera: PerspectiveCamera, settings: Settings) {
            projection = .perspective(verticalFieldOfView: camera.verticalFieldOfView)
            lens = Lens(apertureRadius: settings.apertureRadius, focusDistance: settings.focusDistance)
            near = camera.near
            far = camera.far
        }

        func isValid(width: Int, height: Int) -> Bool {
            guard near.isNormal, far.isNormal, near > 0, far > near, width > 0, height > 0 else { return false }
            if let lens {
                guard lens.isValid, case .perspective = projection else { return false }
            }
            switch projection {
            case .perspective(let fov, let shift):
                guard fov.isFinite, fov > 0, fov < .pi, shift.x.isFinite, shift.y.isFinite,
                      abs(shift.x) <= 4, abs(shift.y) <= 4 else { return false }
            case .orthographic(let windowHeight):
                guard windowHeight.isNormal, windowHeight > 0, windowHeight <= 1e9 else { return false }
            case .equirectangular:
                break
            case .stereoEquirectangular(let distance):
                guard distance.isNormal, distance > 0, distance <= 1e6, height.isMultiple(of: 2) else { return false }
            }
            let window = window(width: width, height: height)
            return window.x.isFinite && window.y.isFinite && window.x >= 0 && window.y >= 0
        }

        func aspectRatio(width: Int, height: Int) -> Float {
            projectionAspectRatio ?? Float(width) / Float(height)
        }

        /// The half window: tangents for perspective, world units for orthographic.
        func window(width: Int, height: Int) -> SIMD2<Float> {
            let aspect = aspectRatio(width: width, height: height)
            switch projection {
            case .perspective(let fov, _):
                let tanY = Float(tan(Double(fov) / 2))
                return SIMD2(tanY * aspect, tanY)
            case .orthographic(let windowHeight):
                return SIMD2(windowHeight / 2 * aspect, windowHeight / 2)
            case .equirectangular, .stereoEquirectangular:
                return .zero
            }
        }

        // Projection modes, matching ptCamera in PathTracingShading.h.
        var mode: Float {
            switch projection {
            case .perspective: 0
            case .orthographic: 1
            case .equirectangular: 2
            case .stereoEquirectangular: 3
            }
        }

        var halfEyeSeparation: Float {
            if case .stereoEquirectangular(let distance) = projection { return distance / 2 }
            return 0
        }

        var shift: SIMD2<Float> {
            if case .perspective(_, let shift) = projection { return shift }
            return .zero
        }
    }

    /// Encodes 1 to 16 samples per pixel through a camera model, with the contract of
    /// `encode(camera:...)`. The model's lens replaces the settings' round lens, so
    /// `settings.apertureRadius` must be zero. Panoramas measure `near` and `far`
    /// along each ray. Histories compare the model, so changing it requires a restart.
    public func encode(
        cameraModel: CameraModel,
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
        guard settings.apertureRadius == 0 else { throw Failure.settings }
        return try encode(
            model: cameraModel, cameraTransform: cameraTransform, destination: destination, accumulation: accumulation,
            previous: previous, guides: guides, commandBuffer: commandBuffer, samplesPerPixel: samplesPerPixel,
            settings: settings, lighting: lighting, timing: timing, region: region, aovs: aovs
        )
    }
}
