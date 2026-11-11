import Foundation
import Metal
import simd

/// A reusable set of tileable material textures from a preset or authored sampling function.
public struct ProceduralSurface {
    public enum Style: Sendable {
        case leatherette
        case brushedMetal
        case wood
    }

    /// One authored texel. All fields must be finite and between zero and one.
    public struct Sample: Sendable {
        public enum Field: String, Sendable {
            case baseColor, height, roughness, metallic, occlusion
        }

        /// Linear RGB. The generated base-color texture uses sRGB storage.
        public let baseColor: SIMD3<Float>
        public let height: Float
        public let roughness: Float
        public let metallic: Float
        public let occlusion: Float

        public init(
            baseColor: SIMD3<Float>, height: Float = 0, roughness: Float = 0.5,
            metallic: Float = 0, occlusion: Float = 1
        ) {
            self.baseColor = baseColor
            self.height = height
            self.roughness = roughness
            self.metallic = metallic
            self.occlusion = occlusion
        }

        fileprivate var invalidField: Field? {
            if !(0..<3).allSatisfy({ (0...1).contains(baseColor[$0]) }) { return .baseColor }
            if !(0...1).contains(height) { return .height }
            if !(0...1).contains(roughness) { return .roughness }
            if !(0...1).contains(metallic) { return .metallic }
            if !(0...1).contains(occlusion) { return .occlusion }
            return nil
        }
    }

    /// sRGB color. Samples decode to linear RGB in Metal.
    public let baseColorTexture: any MTLTexture
    /// Linear tangent-space normals.
    public let normalTexture: any MTLTexture
    /// Linear occlusion, roughness, and metallic in the red, green, and blue channels.
    public let metallicRoughnessTexture: any MTLTexture

    /// Generates square, mipmapped textures. The optional color is linear RGB in zero to one.
    /// Preparation waits for the mip chain once so the returned textures can be used on any queue.
    public init(
        device: any MTLDevice, style: Style, resolution: Int = 512, seed: UInt64,
        baseColor: SIMD3<Float>? = nil
    ) throws {
        guard (32...1024).contains(resolution) else { throw Failure.resolution }
        let color = baseColor ?? style.defaultColor
        guard (0..<3).allSatisfy({ (0...1).contains(color[$0]) }) else { throw Failure.color }

        let pattern = Pattern(style: style, seed: seed)
        let maps = Maps(resolution: resolution, heightScale: style.relief) { uv, _, _ in
            let sample = pattern.sample(uv.x, uv.y)
            return Sample(
                baseColor: color * sample.brightness, height: sample.height, roughness: sample.roughness,
                metallic: sample.metallic, occlusion: sample.occlusion
            )
        }
        try self.init(device: device, resolution: resolution, maps: maps)
    }

    /// Samples an authored pattern once at each texel center in row order. UVs
    /// increase rightward and downward from the top-left corner. For a seamless
    /// tile, define a pattern that repeats on both axes.
    ///
    /// Height differences wrap at the tile edges. A unit height difference
    /// represents `heightScale` times the tile width; zero produces flat normals.
    /// Height scale must be finite and between zero and one.
    public init(
        device: any MTLDevice, resolution: Int = 512, heightScale: Float = 0.01,
        sample: (SIMD2<Float>) throws -> Sample
    ) throws {
        guard (32...1024).contains(resolution) else { throw Failure.resolution }
        guard (0...1).contains(heightScale) else { throw Failure.heightScale }

        let maps = try Maps(resolution: resolution, heightScale: heightScale) { uv, x, y in
            let authored = try sample(uv)
            if let field = authored.invalidField { throw Failure.invalidSample(x: x, y: y, field: field) }
            return authored
        }
        try self.init(device: device, resolution: resolution, maps: maps)
    }

    private init(device: any MTLDevice, resolution: Int, maps: Maps) throws {
        let base = try Self.upload(maps.color, resolution: resolution, format: .rgba8Unorm_srgb, device: device)
        let normal = try Self.upload(maps.normal, resolution: resolution, format: .rgba8Unorm, device: device)
        let data = try Self.upload(maps.data, resolution: resolution, format: .rgba8Unorm, device: device)
        base.label = "Frostfield procedural color"
        normal.label = "Frostfield procedural normal"
        data.label = "Frostfield procedural surface data"

        guard let queue = device.makeCommandQueue(), let buffer = queue.makeCommandBuffer(),
              let blit = buffer.makeBlitCommandEncoder() else { throw Failure.mipmaps }
        buffer.label = "Frostfield procedural surface mipmaps"
        for texture in [base, normal, data] { blit.generateMipmaps(for: texture) }
        blit.endEncoding()
        buffer.commit()
        buffer.waitUntilCompleted()
        guard buffer.status == .completed else { throw Failure.mipmaps }

        baseColorTexture = base
        normalTexture = normal
        metallicRoughnessTexture = data
    }

    /// Uses the generated maps in a physically based material with repeat sampling.
    public func material(
        directionToLight: SIMD3<Float> = SIMD3(0, 1, 1),
        lightIntensity: Float = 1,
        ambientIntensity: Float = 0.12,
        receivesShadows: Bool = false,
        environment: PhysicallyBasedMaterial.Environment? = nil,
        normalScale: Float = 1
    ) throws -> PhysicallyBasedMaterial {
        let surfaceData = PhysicallyBasedMaterial.TextureBinding(texture: metallicRoughnessTexture)
        return try PhysicallyBasedMaterial(
            baseColorTexture: baseColorTexture, metallic: 1, roughness: 1,
            directionToLight: directionToLight, lightIntensity: lightIntensity,
            ambientIntensity: ambientIntensity, receivesShadows: receivesShadows,
            metallicRoughnessTexture: surfaceData,
            normalTexture: .init(texture: normalTexture), normalScale: normalScale,
            occlusionTexture: surfaceData, alphaMode: .opaque, environment: environment
        )
    }

    public enum Failure: LocalizedError, Equatable {
        case resolution
        case color
        case heightScale
        case invalidSample(x: Int, y: Int, field: Sample.Field)
        case texture
        case mipmaps

        public var errorDescription: String? {
            switch self {
            case .resolution: "Procedural surface resolution must be between 32 and 1024."
            case .color: "Procedural surface color must have finite linear components between zero and one."
            case .heightScale: "Procedural surface height scale must be finite and between zero and one."
            case .invalidSample(let x, let y, let field):
                "The procedural surface texel at (\(x), \(y)) needs finite \(field.rawValue) values between zero and one."
            case .texture: "Metal could not allocate a procedural surface texture."
            case .mipmaps: "Metal could not generate the procedural surface mipmaps."
            }
        }
    }

    private static func upload(
        _ pixels: [SIMD4<UInt8>], resolution: Int, format: MTLPixelFormat, device: any MTLDevice
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: resolution, height: resolution, mipmapped: true
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.texture }
        pixels.withUnsafeBytes { bytes in
            if let baseAddress = bytes.baseAddress {
                texture.replace(
                    region: MTLRegionMake2D(0, 0, resolution, resolution), mipmapLevel: 0,
                    withBytes: baseAddress, bytesPerRow: resolution * MemoryLayout<SIMD4<UInt8>>.stride
                )
            }
        }
        return texture
    }
}

private extension ProceduralSurface.Style {
    var defaultColor: SIMD3<Float> {
        switch self {
        case .leatherette: SIMD3(0.041, 0.031, 0.024)
        case .brushedMetal: SIMD3(0.62, 0.60, 0.56)
        case .wood: SIMD3(0.30, 0.13, 0.057)
        }
    }

    var relief: Float {
        switch self {
        case .leatherette: 0.018
        case .brushedMetal: 0.0028
        case .wood: 0.008
        }
    }
}

private struct Maps {
    let color: [SIMD4<UInt8>]
    let normal: [SIMD4<UInt8>]
    let data: [SIMD4<UInt8>]

    init(
        resolution: Int, heightScale: Float,
        sample: (SIMD2<Float>, Int, Int) throws -> ProceduralSurface.Sample
    ) rethrows {
        let count = resolution * resolution
        let coordinateScale = 1 / Float(resolution)
        var color = [SIMD4<UInt8>](repeating: .zero, count: count)
        var normal = color
        var data = color
        var height = [Float](repeating: 0, count: count)

        for y in 0..<resolution {
            let v = (Float(y) + 0.5) * coordinateScale
            for x in 0..<resolution {
                let u = (Float(x) + 0.5) * coordinateScale
                let index = y * resolution + x
                let value = try sample(SIMD2(u, v), x, y)
                let linear = value.baseColor
                color[index] = SIMD4(Self.sRGB(linear.x), Self.sRGB(linear.y), Self.sRGB(linear.z), 255)
                data[index] = SIMD4(Self.byte(value.occlusion), Self.byte(value.roughness), Self.byte(value.metallic), 255)
                height[index] = value.height
            }
        }

        let slopeScale = Float(resolution) * heightScale * 0.5
        for y in 0..<resolution {
            let above = ((y + resolution - 1) % resolution) * resolution
            let below = ((y + 1) % resolution) * resolution
            let row = y * resolution
            for x in 0..<resolution {
                let left = row + (x + resolution - 1) % resolution
                let right = row + (x + 1) % resolution
                let slopeX = (height[left] - height[right]) * slopeScale
                let slopeY = (height[above + x] - height[below + x]) * slopeScale
                let encoded = simd_normalize(SIMD3(slopeX, slopeY, 1)) * 0.5 + 0.5
                normal[row + x] = SIMD4(Self.byte(encoded.x), Self.byte(encoded.y), Self.byte(encoded.z), 255)
            }
        }

        self.color = color
        self.normal = normal
        self.data = data
    }

    private static func byte(_ value: Float) -> UInt8 {
        UInt8((min(max(value, 0), 1) * 255).rounded())
    }

    private static func sRGB(_ value: Float) -> UInt8 {
        let linear = min(max(value, 0), 1)
        let encoded = linear <= 0.003_130_8 ? 12.92 * linear : 1.055 * pow(linear, 1 / 2.4) - 0.055
        return byte(encoded)
    }
}

private struct Pattern {
    let style: ProceduralSurface.Style
    let broad: Noise
    let grain: Noise
    let fine: Noise
    let pebbles: Pebbles?

    init(style: ProceduralSurface.Style, seed: UInt64) {
        self.style = style
        let dimensions: ((Int, Int), (Int, Int), (Int, Int))
        switch style {
        case .leatherette: dimensions = ((7, 7), (97, 97), (173, 173))
        case .brushedMetal: dimensions = ((5, 27), (8, 103), (23, 173))
        case .wood: dimensions = ((5, 9), (14, 43), (35, 157))
        }
        broad = Noise(columns: dimensions.0.0, rows: dimensions.0.1, seed: seed)
        grain = Noise(columns: dimensions.1.0, rows: dimensions.1.1, seed: seed &+ 1)
        fine = Noise(columns: dimensions.2.0, rows: dimensions.2.1, seed: seed &+ 2)
        pebbles = style == .leatherette ? Pebbles(size: 39, seed: seed &+ 3) : nil
    }

    struct Sample {
        let height: Float
        let brightness: Float
        let roughness: Float
        let occlusion: Float
        let metallic: Float
    }

    func sample(_ u: Float, _ v: Float) -> Sample {
        let broad = broad.sample(u, v)
        let grain = grain.sample(u, v)
        let fine = fine.sample(u, v)

        switch style {
        case .leatherette:
            let pebble = pebbles?.sample(u, v) ?? (crown: 0, tint: 0)
            let pores = max(0, (fine - 0.68) / 0.32)
            let crown = pebble.crown
            return Sample(
                height: crown * (0.72 + 0.16 * pebble.tint) + 0.12 * grain - 0.14 * pores,
                brightness: 0.56 + 0.34 * crown + 0.16 * broad + 0.15 * pebble.tint - 0.12 * pores,
                roughness: 0.76 - 0.21 * crown + 0.10 * grain,
                occlusion: 0.78 + 0.22 * crown, metallic: 0
            )
        case .brushedMetal:
            let narrow = fine * fine
            let scratch = narrow * narrow * narrow
            return Sample(
                height: 0.48 * grain + 0.26 * fine - 0.38 * scratch,
                brightness: 0.77 + 0.22 * grain + 0.12 * broad - 0.08 * scratch,
                roughness: 0.25 + 0.19 * grain + 0.10 * broad + 0.12 * scratch,
                occlusion: 1, metallic: 1
            )
        case .wood:
            let phase = 2 * Float.pi * (15 * v + 0.65 * broad + 0.18 * grain)
            let ring = 0.5 + 0.5 * sin(phase)
            let latewood = ring * ring * ring
            return Sample(
                height: 0.36 * latewood + 0.25 * grain + 0.11 * fine,
                brightness: 0.59 + 0.31 * (1 - latewood) + 0.27 * broad + 0.07 * fine,
                roughness: 0.42 + 0.17 * latewood + 0.12 * grain,
                occlusion: 0.91 + 0.09 * ring, metallic: 0
            )
        }
    }
}

private struct Noise {
    let columns: Int
    let rows: Int
    let values: [Float]

    init(columns: Int, rows: Int, seed: UInt64) {
        self.columns = columns
        self.rows = rows
        var random = SurfaceRandom(state: seed)
        values = (0..<(columns * rows)).map { _ in random.next() }
    }

    func sample(_ u: Float, _ v: Float) -> Float {
        let px = u * Float(columns)
        let py = v * Float(rows)
        let x = Int(px)
        let y = Int(py)
        let nextX = (x + 1) % columns
        let nextY = ((y + 1) % rows) * columns
        let row = y * columns
        let tx = smooth(px - Float(x))
        let ty = smooth(py - Float(y))
        let top = values[row + x] + (values[row + nextX] - values[row + x]) * tx
        let bottom = values[nextY + x] + (values[nextY + nextX] - values[nextY + x]) * tx
        return top + (bottom - top) * ty
    }
}

private struct Pebbles {
    let size: Int
    let cells: [SIMD3<Float>]

    init(size: Int, seed: UInt64) {
        self.size = size
        var random = SurfaceRandom(state: seed)
        cells = (0..<(size * size)).map { _ in
            SIMD3(0.2 + 0.6 * random.next(), 0.2 + 0.6 * random.next(), random.next())
        }
    }

    func sample(_ u: Float, _ v: Float) -> (crown: Float, tint: Float) {
        let px = u * Float(size)
        let py = v * Float(size)
        let x = Int(px)
        let y = Int(py)
        var nearest = Float.infinity
        var second = Float.infinity
        var tint: Float = 0
        for cellY in (y - 1)...(y + 1) {
            let row = ((cellY + size) % size) * size
            for cellX in (x - 1)...(x + 1) {
                let cell = cells[row + (cellX + size) % size]
                let dx = Float(cellX) + cell.x - px
                let dy = Float(cellY) + cell.y - py
                let distance = dx * dx + dy * dy
                if distance < nearest {
                    second = nearest
                    nearest = distance
                    tint = cell.z
                } else {
                    second = min(second, distance)
                }
            }
        }
        let seam = (sqrt(second) - sqrt(nearest)) * 5.5
        let crown = smooth(min(seam, 1)) * max(0.58, 1 - 0.30 * nearest)
        return (crown, tint)
    }
}

private func smooth(_ value: Float) -> Float {
    value * value * (3 - 2 * value)
}

private struct SurfaceRandom {
    var state: UInt64

    mutating func next() -> Float {
        state &+= 0x9E37_79B9_7F4A_7C15
        var bits = state
        bits = (bits ^ (bits >> 30)) &* 0xBF58_476D_1CE4_E5B9
        bits = (bits ^ (bits >> 27)) &* 0x94D0_49BB_1331_11EB
        bits ^= bits >> 31
        return Float(bits >> 40) / 16_777_216
    }
}
