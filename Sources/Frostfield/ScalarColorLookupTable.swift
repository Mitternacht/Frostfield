import Foundation
import Metal
import simd

/// Maps scalar values to a finite table of linear RGB colors.
/// Like Three.js r187 `Lut`, a table of n intervals stores n + 1 samples
/// including both endpoints, and lookup rounds to the nearest sample.
public struct ScalarColorLookupTable: Hashable, Sendable {
    public static let maximumIntervalCount = 4_095

    public enum Palette: String, CaseIterable, Sendable {
        case rainbow
        case coolToWarm = "cooltowarm"
        case blackbody
        case grayscale

        fileprivate var stops: [Stop] {
            // Three.js r187 examples/jsm/math/Lut.js uses LinearSRGBColorSpace for these bytes.
            let colors: [UInt32] = switch self {
            case .rainbow: [0x0000FF, 0x00FFFF, 0x00FF00, 0xFFFF00, 0xFF0000]
            case .coolToWarm: [0x3C4EC2, 0x9BBCFF, 0xDCDCDC, 0xF6A385, 0xB40426]
            case .blackbody: [0x000000, 0x780000, 0xE63200, 0xFFFF00, 0xFFFFFF]
            case .grayscale: [0x000000, 0x404040, 0x7F7F80, 0xBFBFBF, 0xFFFFFF]
            }
            return zip([0, 0.2, 0.5, 0.8, 1], colors).map { Stop($0, linearRGB: Stop.rgb($1)) }
        }
    }

    public struct Stop: Hashable, Sendable {
        /// Position on the normalized ramp, independent of the scalar domain.
        public let position: Double
        public let linearRGB: SIMD3<Float>

        public init(_ position: Double, linearRGB: SIMD3<Float>) {
            self.position = position
            self.linearRGB = linearRGB
        }

        /// Interprets 0xRRGGBB as normalized linear RGB, as the r187 palettes do.
        public init(_ position: Double, linearRGBHex: UInt32) throws {
            guard linearRGBHex <= 0xFF_FF_FF else { throw Failure.hexColor }
            self.init(position, linearRGB: Self.rgb(linearRGBHex))
        }

        fileprivate static func rgb(_ hex: UInt32) -> SIMD3<Float> {
            SIMD3(Float((hex >> 16) & 0xFF), Float((hex >> 8) & 0xFF), Float(hex & 0xFF)) / 255
        }
    }

    public enum TextureLayout: Hashable, Sendable {
        /// One row of samples from minimum to maximum, left to right.
        case lookup
        /// One column of samples from maximum to minimum, top to bottom.
        case legend
    }

    public private(set) var domain: ClosedRange<Double>
    public let intervalCount: Int
    /// Linear RGB from the minimum to the maximum, including both endpoints.
    public let samples: [SIMD3<Float>]

    public init(
        palette: Palette = .rainbow, intervalCount: Int = 32, domain: ClosedRange<Double> = 0...1
    ) throws {
        try self.init(stops: palette.stops, intervalCount: intervalCount, domain: domain)
    }

    /// Samples a piecewise linear ramp. Supply strictly increasing stop positions
    /// including 0 and 1, and finite linear RGB in 0...65504.
    public init(
        stops: [Stop], intervalCount: Int = 32, domain: ClosedRange<Double> = 0...1
    ) throws {
        guard (1...Self.maximumIntervalCount).contains(intervalCount) else { throw Failure.intervalCount }
        try Self.validate(domain: domain)
        guard stops.count >= 2, stops.first?.position == 0, stops.last?.position == 1,
              zip(stops, stops.dropFirst()).allSatisfy({ $0.position < $1.position }) else { throw Failure.stops }
        let colorRange = 0...Float(Float16.greatestFiniteMagnitude)
        guard stops.allSatisfy({
            colorRange.contains($0.linearRGB.x) && colorRange.contains($0.linearRGB.y) && colorRange.contains($0.linearRGB.z)
        }) else { throw Failure.stopColor }

        self.domain = domain
        self.intervalCount = intervalCount
        var segment = 0
        let step = 1 / Double(intervalCount)
        samples = (0...intervalCount).map { index in
            let position = index == intervalCount ? 1 : Double(index) * step
            while position > stops[segment + 1].position { segment += 1 }
            let lower = stops[segment], upper = stops[segment + 1]
            let fraction = (position - lower.position) / (upper.position - lower.position)
            let start = SIMD3<Double>(lower.linearRGB), end = SIMD3<Double>(upper.linearRGB)
            return SIMD3<Float>(start * (1 - fraction) + end * fraction)
        }
    }

    /// Changes the finite scalar domain without rebuilding the color samples.
    public mutating func setDomain(_ domain: ClosedRange<Double>) throws {
        try Self.validate(domain: domain)
        self.domain = domain
    }

    /// Clamps to the domain and chooses the nearest sample, with ties going
    /// toward the larger index. Infinity selects an endpoint. NaN returns nil.
    public func color(at value: Double) -> SIMD3<Float>? {
        guard !value.isNaN else { return nil }
        let clamped = min(max(value, domain.lowerBound), domain.upperBound)
        let normalized = (clamped - domain.lowerBound) / (domain.upperBound - domain.lowerBound)
        return samples[Int((normalized * Double(intervalCount)).rounded(.toNearestOrAwayFromZero))]
    }

    /// Uploads an opaque linear RGBA16Float strip suitable for shader lookup,
    /// a scene texture helper, or a texture viewer overlay.
    public func makeTexture(device: any MTLDevice, layout: TextureLayout = .lookup) throws -> any MTLTexture {
        let (width, height) = dimensions(layout)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        descriptor.hazardTrackingMode = .tracked
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.textureAllocation }
        texture.label = "Frostfield scalar colormap"
        try write(to: texture, layout: layout)
        return texture
    }

    /// Updates a matching shared lookup or legend texture in place. Wait until
    /// earlier GPU work has finished reading the texture before calling this.
    public func write(to texture: any MTLTexture, layout: TextureLayout = .lookup) throws {
        let (width, height) = dimensions(layout)
        guard texture.textureType == .type2D, texture.pixelFormat == .rgba16Float,
              texture.width == width, texture.height == height, texture.mipmapLevelCount == 1,
              texture.storageMode == .shared, texture.usage.contains(.shaderRead), !texture.isFramebufferOnly,
              texture.hazardTrackingMode != .untracked else { throw Failure.texture }
        let pixels = samples.indices.map { index in
            let sample = samples[layout == .lookup ? index : intervalCount - index]
            return SIMD4<Float16>(SIMD4(sample, 1))
        }
        pixels.withUnsafeBytes {
            texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                            withBytes: $0.baseAddress!, bytesPerRow: width * MemoryLayout<SIMD4<Float16>>.stride)
        }
    }

    private func dimensions(_ layout: TextureLayout) -> (width: Int, height: Int) {
        layout == .lookup ? (samples.count, 1) : (1, samples.count)
    }

    private static func validate(domain: ClosedRange<Double>) throws {
        let span = domain.upperBound - domain.lowerBound
        guard span.isFinite, span > 0 else { throw Failure.domain }
    }

    public enum Failure: LocalizedError, Equatable {
        case intervalCount, domain, stops, stopColor, hexColor, texture, textureAllocation

        public var errorDescription: String? {
            switch self {
            case .intervalCount: "Use 1 through \(ScalarColorLookupTable.maximumIntervalCount) scalar colormap intervals."
            case .domain: "Use a scalar domain with positive finite width."
            case .stops: "Use strictly increasing colormap stops from 0 through 1."
            case .stopColor: "Use finite linear stop RGB in 0...65504."
            case .hexColor: "Use a 24-bit linear RGB color."
            case .texture: "Use a matching shared, shader-readable RGBA16Float colormap strip."
            case .textureAllocation: "Could not allocate the scalar colormap texture."
            }
        }
    }
}
