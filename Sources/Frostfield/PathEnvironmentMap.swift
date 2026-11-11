import Foundation
import Metal
import simd

extension PathTracer {
    /// Radiance for rays that leave the scene.
    public enum Environment: Equatable, Sendable {
        /// The same linear RGB radiance in every direction. Escaping scattered rays
        /// pick it up without a separate light sample.
        case constant(SIMD3<Float>)
        /// An importance-sampled panorama, scaled by a linear intensity from 0 to
        /// 65,504 and turned by a finite yaw in radians about world +Y. The yaw
        /// matches `PhysicallyBasedMaterial.Environment.yaw` and
        /// `EnvironmentBackground`, so one panorama orients both renderers alike.
        /// Maps compare by identity.
        case map(EnvironmentMap, intensity: Float = 1, yaw: Float = 0)
    }

    /// A linear RGB equirectangular panorama prepared for importance-sampled path tracing.
    ///
    /// The orientation matches `EnvironmentLighting`: the top row faces +Y, the
    /// center column faces +X, and increasing U turns toward +Z. Preparation
    /// stores the image in a shared RGBA16Float texture and builds a sampling
    /// distribution with one marginal CDF over rows and one conditional CDF per
    /// row. Radiance lookups filter bilinearly, wrap horizontally and clamp at the
    /// poles, as the raster panorama conversion does. Each texel's weight is the
    /// Rec. 709 luminance of that filtered radiance averaged over the texel, times
    /// sin θ at the row center, so every lit direction can be light sampled. The
    /// tracer samples the map for next-event estimation and weights it against
    /// BSDF sampling with the power heuristic.
    ///
    /// Memory is `8 × width × height` texture bytes plus `4 × (width + 1) × height`
    /// table bytes: 24 MiB at 2,048 × 1,024 and 96 MiB at 4,096 × 2,048.
    /// Preparation runs on the calling thread and needs no command queue. The map
    /// is immutable, so one map can light many tracers and accumulations on its device.
    public final class EnvironmentMap: Equatable, @unchecked Sendable {
        // Immutable after initialization. The GPU only reads the texture and table.
        public static let maximumWidth = 8_192

        public let device: any MTLDevice
        public let width: Int
        public let height: Int
        /// False for an entirely black map, which needs no light samples.
        public let isSampled: Bool
        let texture: any MTLTexture
        /// Height marginal CDF entries followed by height rows of width conditional CDF entries.
        let distribution: any MTLBuffer

        /// A sample at the center of a texel in native panorama orientation.
        public struct Texel: Sendable {
            /// Integer pixel coordinates from the top left.
            public let coordinates: SIMD2<Int>
            public let dimensions: SIMD2<Int>
            /// Normalized texel center, with V increasing downward.
            public let uv: SIMD2<Float>
            /// Unit direction. +Y is the zenith, +X faces the center meridian,
            /// and increasing U turns toward +Z.
            public let direction: SIMD3<Float>
        }

        /// A latitude gradient in linear RGB. The defaults interpolate from a
        /// black nadir to a white zenith using exponent two.
        public struct VerticalGradient: Equatable, Sendable {
            public var top: SIMD3<Float>
            public var bottom: SIMD3<Float>
            public var exponent: Float

            public init(
                top: SIMD3<Float> = SIMD3(repeating: 1), bottom: SIMD3<Float> = .zero, exponent: Float = 2
            ) {
                self.top = top
                self.bottom = bottom
                self.exponent = exponent
            }
        }

        /// Generates a 2:1 linear HDR panorama in top-left row-major order and
        /// builds its importance distribution from the resulting half-float
        /// pixels. Return finite RGB in 0...65,504. The callback runs on the
        /// calling thread, once per texel, and can throw.
        public convenience init(
            device: any MTLDevice, width: Int = 1_024, height: Int = 512,
            generator: (Texel) throws -> SIMD3<Float>
        ) throws {
            guard Self.isValidSize(width: width, height: height) else { throw Failure.invalidEnvironmentMap }
            let columns = (0..<width).map { column in
                let u = (Double(column) + 0.5) / Double(width)
                let phi = (u - 0.5) * 2 * Double.pi
                return (u: Float(u), cosine: Float(cos(phi)), sine: Float(sin(phi)))
            }
            let dimensions = SIMD2(width, height)
            var pixels: [SIMD4<Float16>] = []
            pixels.reserveCapacity(width * height)
            for row in 0..<height {
                let v = (Double(row) + 0.5) / Double(height)
                let theta = v * Double.pi
                let y = Float(cos(theta))
                let radius = Float(sin(theta))
                for (column, longitude) in columns.enumerated() {
                    let texel = Texel(
                        coordinates: SIMD2(column, row), dimensions: dimensions, uv: SIMD2(longitude.u, Float(v)),
                        direction: SIMD3(radius * longitude.cosine, y, radius * longitude.sine)
                    )
                    pixels.append(try Self.storedPixel(generator(texel)))
                }
            }
            try self.init(device: device, width: width, height: height, pixels: pixels)
        }

        /// Generates a vertical HDR gradient and prepares it for environment
        /// importance sampling. Both colors use the same range as authored pixels.
        /// An exponent of zero gives the top color in every direction.
        public convenience init(
            device: any MTLDevice, width: Int = 1_024, height: Int = 512, gradient: VerticalGradient
        ) throws {
            guard gradient.exponent.isFinite, gradient.exponent >= 0 else { throw Failure.invalidEnvironmentMap }
            _ = try Self.storedPixel(gradient.top)
            _ = try Self.storedPixel(gradient.bottom)
            let lower = SIMD3<Double>(gradient.bottom)
            let difference = SIMD3<Double>(gradient.top) - lower
            try self.init(device: device, width: width, height: height) { texel in
                let t = pow(simd_clamp(0.5 + 0.5 * Double(texel.direction.y), 0, 1), Double(gradient.exponent))
                return SIMD3<Float>(lower + difference * t)
            }
        }

        /// Decodes the first image of a Radiance RGBE file as linear RGB. The file
        /// must be a 2:1 panorama of at most 8,192 × 4,096 pixels; values above
        /// 65,504 saturate as in `EnvironmentLighting(queue:radianceHDR:)`.
        public convenience init(device: any MTLDevice, radianceHDR: Data) throws {
            try self.init(device: device, image: RadianceHDRImage.decode(radianceHDR))
        }

        /// Prepares row-major linear RGB pixels, starting at the top-left texel.
        /// Use a 2:1 panorama of at most 8,192 × 4,096 pixels with finite values from
        /// 0 to 65,504. The texture stores half-precision values.
        public convenience init(device: any MTLDevice, pixels: [SIMD3<Float>], width: Int, height: Int) throws {
            guard Self.isValidSize(width: width, height: height), pixels.count == width * height else {
                throw Failure.invalidEnvironmentMap
            }
            var stored = [SIMD4<Float16>]()
            stored.reserveCapacity(pixels.count)
            for pixel in pixels { stored.append(try Self.storedPixel(pixel)) }
            try self.init(device: device, width: width, height: height, pixels: stored)
        }

        private static func storedPixel(_ pixel: SIMD3<Float>) throws -> SIMD4<Float16> {
            guard (0...65_504).contains(pixel.x), (0...65_504).contains(pixel.y), (0...65_504).contains(pixel.z) else {
                throw Failure.invalidEnvironmentMap
            }
            return SIMD4(SIMD3<Float16>(pixel), 1)
        }

        package convenience init(device: any MTLDevice, image: RadianceHDRImage) throws {
            try self.init(device: device, width: image.width, height: image.height, pixels: image.pixels)
        }

        init(device: any MTLDevice, width: Int, height: Int, pixels: [SIMD4<Float16>]) throws {
            let range = Float16.zero...Float16.greatestFiniteMagnitude
            guard Self.isValidSize(width: width, height: height), pixels.count == width * height,
                  pixels.allSatisfy({ range.contains($0.x) && range.contains($0.y) && range.contains($0.z) }) else {
                throw Failure.invalidEnvironmentMap
            }
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
            )
            descriptor.storageMode = .shared
            descriptor.usage = .shaderRead
            descriptor.hazardTrackingMode = .tracked
            guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.environmentAllocation }
            texture.label = "Frostfield path environment"
            pixels.withUnsafeBytes { bytes in
                texture.replace(
                    region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                    withBytes: bytes.baseAddress!, bytesPerRow: width * MemoryLayout<SIMD4<Float16>>.stride
                )
            }

            let (table, sampled) = Self.makeDistribution(pixels: pixels, width: width, height: height)
            let buffer = table.withUnsafeBytes { bytes in
                device.makeBuffer(
                    bytes: bytes.baseAddress!, length: bytes.count,
                    options: [.storageModeShared, .hazardTrackingModeTracked]
                )
            }
            guard let buffer else { throw Failure.environmentAllocation }
            buffer.label = "Frostfield path environment distribution"
            self.device = device
            self.width = width
            self.height = height
            self.texture = texture
            self.distribution = buffer
            isSampled = sampled
        }

        public static func == (lhs: EnvironmentMap, rhs: EnvironmentMap) -> Bool { lhs === rhs }

        /// The solid-angle density, per steradian, with which the tracer picks a
        /// world direction when the map is turned by `yaw` radians. It integrates to
        /// one over the sphere. Returns zero for directions the sampler never picks,
        /// for an entirely black map, and for zero or nonfinite input.
        public func pdf(direction: SIMD3<Float>, yaw: Float = 0) -> Float {
            pdf(direction: direction, eulerRotation: SIMD3(0, yaw, 0))
        }

        /// Solid-angle density for a map with an authored XYZ rotation in radians.
        public func pdf(direction: SIMD3<Float>, eulerRotation: SIMD3<Float>) -> Float {
            let length = simd_length(direction)
            guard isSampled, length.isNormal, length.isFinite, eulerRotation.isFiniteVector else { return 0 }
            let local = EnvironmentLookup(eulerRotation: eulerRotation).localFromWorld * (direction / length)
            let sine = sqrt(max(0, 1 - local.y * local.y))
            guard sine > 1e-6 else { return 0 }
            let uv = Self.uv(local)
            let column = min(Int(uv.x * Float(width)), width - 1)
            let row = min(Int(uv.y * Float(height)), height - 1)
            return density(row: row, column: column, sine: sine)
        }

        /// Maps two uniform numbers in [0, 1) to a world direction and its density,
        /// using the same tables and arithmetic as the tracer's light sample. The
        /// second number picks the row. Returns nil for a black map, inputs outside
        /// [0, 1) or a sample at a pole.
        public func sample(_ random: SIMD2<Float>, yaw: Float = 0) -> (direction: SIMD3<Float>, pdf: Float)? {
            sample(random, eulerRotation: SIMD3(0, yaw, 0))
        }

        /// Samples a world direction for a map with an authored XYZ rotation in radians.
        public func sample(_ random: SIMD2<Float>, eulerRotation: SIMD3<Float>) -> (direction: SIMD3<Float>, pdf: Float)? {
            guard isSampled, eulerRotation.isFiniteVector, (0..<1).contains(random.x), (0..<1).contains(random.y) else { return nil }
            let table = distribution.contents().assumingMemoryBound(to: Float.self)
            let row = Self.search(table, count: height, value: random.y)
            let rowBefore = row == 0 ? 0 : table[row - 1]
            let rowMass = table[row] - rowBefore
            let conditional = table + height + row * width
            let column = Self.search(conditional, count: width, value: random.x)
            let columnBefore = column == 0 ? 0 : conditional[column - 1]
            let columnMass = conditional[column] - columnBefore
            guard rowMass > 0, columnMass > 0 else { return nil }
            let u = (Float(column) + simd_clamp((random.x - columnBefore) / columnMass, 0, 1)) / Float(width)
            let v = (Float(row) + simd_clamp((random.y - rowBefore) / rowMass, 0, 1)) / Float(height)
            let theta = v * .pi
            let phi = (u - 0.5) * 2 * .pi
            let sine = sin(theta)
            guard sine > 1e-6 else { return nil }
            let local = SIMD3(sine * cos(phi), cos(theta), sine * sin(phi))
            let pdf = rowMass * Float(height) * columnMass * Float(width) / (2 * .pi * .pi * sine)
            return (EnvironmentLookup(eulerRotation: eulerRotation).localFromWorld.transpose * local, pdf)
        }

        private func density(row: Int, column: Int, sine: Float) -> Float {
            let table = distribution.contents().assumingMemoryBound(to: Float.self)
            let conditional = table + height + row * width
            let rowMass = table[row] - (row == 0 ? 0 : table[row - 1])
            let columnMass = conditional[column] - (column == 0 ? 0 : conditional[column - 1])
            return rowMass * Float(height) * columnMass * Float(width) / (2 * .pi * .pi * sine)
        }

        // Matches ptSkyUV and ptSearch in PathTracingShading.h.
        private static func uv(_ local: SIMD3<Float>) -> SIMD2<Float> {
            SIMD2(atan2(local.z, local.x) / (2 * .pi) + 0.5, acos(simd_clamp(local.y, -1, 1)) / .pi)
        }

        private static func search(_ cdf: UnsafePointer<Float>, count: Int, value: Float) -> Int {
            var low = 0
            var high = count - 1
            while low < high {
                let middle = (low + high) / 2
                if cdf[middle] > value { high = middle } else { low = middle + 1 }
            }
            return low
        }

        private static func isValidSize(width: Int, height: Int) -> Bool {
            (1...maximumWidth / 2).contains(height) && width == 2 * height
        }

        /// Builds inclusive CDFs in double precision. Each table ends at exactly one.
        /// Black rows keep a uniform conditional CDF and receive no marginal mass.
        private static func makeDistribution(
            pixels: [SIMD4<Float16>], width: Int, height: Int
        ) -> (table: [Float], sampled: Bool) {
            let luminance = pixels.map { pixel -> Double in
                let value = SIMD4<Float>(pixel)
                return max(0, 0.2126 * Double(value.x) + 0.7152 * Double(value.y) + 0.0722 * Double(value.z))
            }
            // Averaging a bilinear reconstruction over one texel weights its 3 × 3
            // neighborhood by 1/8, 3/4 and 1/8 per axis, wrapping U and clamping V.
            let weights = [0.125, 0.75, 0.125]
            var table = [Float](repeating: 0, count: height + width * height)
            var rowSums = [Double](repeating: 0, count: height)
            var running = [Double](repeating: 0, count: width)
            for row in 0..<height {
                let sine = sin((Double(row) + 0.5) / Double(height) * .pi)
                var sum = 0.0
                for column in 0..<width {
                    var filtered = 0.0
                    for (b, rowWeight) in weights.enumerated() {
                        let source = min(max(row + b - 1, 0), height - 1) * width
                        for (a, columnWeight) in weights.enumerated() {
                            let neighbor = (column + a - 1 + width) % width
                            filtered += rowWeight * columnWeight * luminance[source + neighbor]
                        }
                    }
                    sum += filtered * sine
                    running[column] = sum
                }
                rowSums[row] = sum
                let offset = height + row * width
                for column in 0..<width {
                    table[offset + column] = sum > 0
                        ? Float(running[column] / sum)
                        : Float(Double(column + 1) / Double(width))
                }
                table[offset + width - 1] = 1
            }
            let total = rowSums.reduce(0, +)
            guard total > 0, total.isFinite else {
                for row in 0..<height { table[row] = Float(Double(row + 1) / Double(height)) }
                table[height - 1] = 1
                return (table, false)
            }
            var cumulative = 0.0
            for row in 0..<height {
                cumulative += rowSums[row]
                table[row] = Float(cumulative / total)
            }
            table[height - 1] = 1
            return (table, true)
        }
    }
}
