import Foundation
import Metal
import simd

/// A deterministic linear normal map for metallic paint and clearcoat detail.
public struct FlakesTexture {
    public enum Encoding: Equatable, Sendable {
        /// Matches the red, green, and blue code ranges of Three.js FlakesTexture.
        case threeJS
        /// Maps all three components of a unit normal from −1...1 to 0...1.
        case tangentSpaceRGB
    }

    public enum EdgeMode: Equatable, Sendable {
        /// Clips disks at the texture boundary.
        case clipped
        /// Continues disks across opposite edges for a seamless tile.
        case wrapped
    }

    public struct Settings: Equatable, Sendable {
        public var width: Int
        public var height: Int
        public var flakeCount: Int
        /// Uniform disk radii in texels.
        public var radius: ClosedRange<Double>
        public var encoding: Encoding
        public var edges: EdgeMode
        public var mipmapped: Bool

        public init(
            width: Int = 512, height: Int = 512, flakeCount: Int = 4_000,
            radius: ClosedRange<Double> = 3...6, encoding: Encoding = .threeJS,
            edges: EdgeMode = .clipped, mipmapped: Bool = true
        ) {
            self.width = width
            self.height = height
            self.flakeCount = flakeCount
            self.radius = radius
            self.encoding = encoding
            self.edges = edges
            self.mipmapped = mipmapped
        }
    }

    /// Opaque, top-left RGBA8Unorm. Bind it as a base or clearcoat RGB normal map.
    public let normalTexture: any MTLTexture

    /// Draws uniformly placed disks in seed order. Each disk has a constant
    /// tangent normal from normalized `(x, y, 1.5)`, where x and y are
    /// independent uniform draws from −1..<1. Later disks cover earlier disks.
    /// The same settings and seed generate the same base texels.
    ///
    /// Dimensions are bounded to 8,192 per axis and 16,777,216 base texels.
    /// Radii must be within 0.5...64 texels. The count and stamp budget are
    /// bounded. Zero flakes generates a neutral map. Requested mipmaps complete
    /// before this initializer returns.
    public init(device: any MTLDevice, seed: UInt64, settings: Settings = .init()) throws {
        guard (try? TextureLoader.validate(width: settings.width, height: settings.height)) != nil else {
            throw Failure.dimensions
        }
        guard (0.5...64).contains(settings.radius.lowerBound),
              (0.5...64).contains(settings.radius.upperBound) else { throw Failure.radius }
        let stampSide = Int(ceil(2 * settings.radius.upperBound + 2))
        let stampArea = min(settings.width, stampSide) * min(settings.height, stampSide)
        guard (0...65_536).contains(settings.flakeCount),
              settings.flakeCount <= 16_777_216 / stampArea else { throw Failure.stampBudget }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: settings.width, height: settings.height, mipmapped: settings.mipmapped
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.texture }
        texture.label = "Frostfield flakes normal"

        let pixels = Self.rasterize(settings, seed: seed)
        pixels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, settings.width, settings.height), mipmapLevel: 0,
                withBytes: bytes.baseAddress!, bytesPerRow: settings.width * MemoryLayout<SIMD4<UInt8>>.stride
            )
        }
        if texture.mipmapLevelCount > 1 {
            do {
                try TextureLoader(device: device).generateMipmaps(for: texture)
            } catch {
                throw Failure.mipmaps
            }
        }
        normalTexture = texture
    }

    private static func rasterize(_ settings: Settings, seed: UInt64) -> [SIMD4<UInt8>] {
        let width = settings.width, height = settings.height
        let neutral: UInt8 = settings.encoding == .threeJS ? 127 : 128
        var pixels = [SIMD4<UInt8>](repeating: SIMD4(neutral, neutral, 255, 255), count: width * height)
        var random = SceneSeedGenerator(state: seed)
        let wraps = settings.edges == .wrapped
        let span = settings.radius.upperBound - settings.radius.lowerBound
        for _ in 0..<settings.flakeCount {
            let center = SIMD2(random.nextDouble() * Double(width), random.nextDouble() * Double(height))
            let radius = settings.radius.lowerBound + random.nextDouble() * span
            let normal = simd_normalize(SIMD3(2 * random.nextDouble() - 1, 2 * random.nextDouble() - 1, 1.5))
            let color = encode(normal, as: settings.encoding)
            // A one-texel distance ramp antialiases the disk boundary.
            let reach = radius + 0.5
            let minX = Int(floor(center.x - reach)), minY = Int(floor(center.y - reach))
            let maxX = Int(ceil(center.x + reach)), maxY = Int(ceil(center.y + reach))
            let columns = wraps ? minX..<min(maxX, minX + width) : max(0, minX)..<min(width, maxX)
            let rows = wraps ? minY..<min(maxY, minY + height) : max(0, minY)..<min(height, maxY)
            for y in rows {
                let row = (wraps ? periodicIndex(y, extent: height) : y) * width
                let dy = delta(y, from: center.y, period: wraps ? Double(height) : nil)
                for x in columns {
                    let dx = delta(x, from: center.x, period: wraps ? Double(width) : nil)
                    let coverage = reach - sqrt(dx * dx + dy * dy)
                    guard coverage > 0 else { continue }
                    let index = row + (wraps ? periodicIndex(x, extent: width) : x)
                    if coverage >= 1 {
                        pixels[index] = color
                    } else {
                        for channel in 0..<3 {
                            let current = Double(pixels[index][channel])
                            pixels[index][channel] = byte(current + (Double(color[channel]) - current) * coverage)
                        }
                    }
                }
            }
        }
        return pixels
    }

    private static func periodicIndex(_ index: Int, extent: Int) -> Int {
        let remainder = index % extent
        return remainder < 0 ? remainder + extent : remainder
    }

    private static func delta(_ texel: Int, from center: Double, period: Double?) -> Double {
        let offset = Double(texel) + 0.5 - center
        guard let period else { return offset }
        return offset - (offset / period).rounded() * period
    }

    private static func encode(_ normal: SIMD3<Double>, as encoding: Encoding) -> SIMD4<UInt8> {
        let codes: SIMD3<Double>
        switch encoding {
        case .threeJS: codes = SIMD3(127 + 127 * normal.x, 127 + 127 * normal.y, 255 * normal.z)
        case .tangentSpaceRGB: codes = (normal * 0.5 + 0.5) * 255
        }
        return SIMD4(byte(codes.x), byte(codes.y), byte(codes.z), 255)
    }

    private static func byte(_ value: Double) -> UInt8 { UInt8(clamping: Int(value.rounded())) }

    public enum Failure: LocalizedError, Equatable {
        case dimensions
        case radius
        case stampBudget
        case texture
        case mipmaps

        public var errorDescription: String? {
            switch self {
            case .dimensions: "Use positive flake dimensions within 8,192 per axis and 16,777,216 base texels."
            case .radius: "Flake radii must be finite and within 0.5...64 texels."
            case .stampBudget: "Flake generation supports up to 65,536 disks and 16,777,216 candidate stamp texels."
            case .texture: "Metal could not allocate the flakes normal texture."
            case .mipmaps: "Metal could not generate the flakes normal mipmaps."
            }
        }
    }
}
