import Foundation
import QuartzCore

public extension FrostfieldSceneView {
    /// The number of Metal drawable pixels per layout point.
    struct DrawableResolution: Equatable, Sendable {
        /// Follow the current window's display backing resolution.
        public static let display = Self()

        /// A fixed number of pixels per point, or nil when following the display.
        public let pixelsPerPoint: CGFloat?

        private init() { pixelsPerPoint = nil }

        /// Render at an explicit scale independent of the current display.
        public init(pixelsPerPoint: CGFloat) throws(Failure) {
            guard pixelsPerPoint.isFinite, pixelsPerPoint > 0 else { throw .pixelRatio }
            self.pixelsPerPoint = pixelsPerPoint
        }

        func apply(
            to layer: CAMetalLayer, logicalSize: CGSize, backingSize: CGSize, displayScale: CGFloat
        ) -> Result<CGSize?, Failure> {
            guard displayScale.isFinite, displayScale > 0,
                  logicalSize.width > 0, logicalSize.height > 0 else { return .success(nil) }
            let pixels = pixelsPerPoint.map { ratio in
                CGSize(width: logicalSize.width * ratio, height: logicalSize.height * ratio)
            } ?? backingSize
            let size = CGSize(width: pixels.width.rounded(.up), height: pixels.height.rounded(.up))
            guard size.width.isFinite, size.height.isFinite,
                  (1...16_384).contains(size.width), (1...16_384).contains(size.height) else {
                return .failure(.size)
            }
            if layer.contentsScale != displayScale { layer.contentsScale = displayScale }
            if layer.drawableSize != size { layer.drawableSize = size }
            return .success(size)
        }

        public enum Failure: LocalizedError, Equatable {
            case pixelRatio
            case size

            public var errorDescription: String? {
                switch self {
                case .pixelRatio: "Use a finite positive number of drawable pixels per point."
                case .size: "The drawable size must be within 1 to 16,384 pixels on each axis."
                }
            }
        }
    }
}
