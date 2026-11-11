import Foundation
import Metal
import simd

/// Distant diffuse lighting represented by three bands of real spherical harmonics.
public struct LightProbe: Hashable, Sendable {
    public static let coefficientCount = 9
    /// Select a lower mip when projecting a larger cube.
    public static let maximumFaceSize = 1_024

    /// Linear RGB radiance coefficients in Three.js `SphericalHarmonics3` order.
    /// The basis is [1, y, z, x, xy, yz, 3z²−1, xz, x²−y²] with each term's
    /// orthonormal normalization. Coefficients may be negative.
    public let coefficients: [SIMD3<Float>]
    public private(set) var intensity: Float

    static let coefficientLimit = Double(Float.greatestFiniteMagnitude) / 16
    static let noRasterLighting = [SIMD4<Float>](repeating: .zero, count: coefficientCount)

    public init(coefficients: [SIMD3<Float>], intensity: Float = 1) throws {
        guard coefficients.count == Self.coefficientCount, coefficients.allSatisfy(\.isFiniteVector) else {
            throw Failure.coefficients
        }
        try Self.validate(intensity: intensity, coefficients: coefficients)
        self.coefficients = coefficients
        self.intensity = intensity
    }

    /// Creates a diffuse hemisphere light from nonnegative linear RGB irradiance.
    /// The sky color faces `direction` in world space. Opposing normals receive
    /// the ground color. The blend varies linearly with the cosine between the
    /// surface normal and that direction.
    public static func hemisphere(
        skyColor: SIMD3<Float>, groundColor: SIMD3<Float>,
        direction: SIMD3<Float> = SIMD3(0, 1, 0), intensity: Float = 1
    ) throws -> LightProbe {
        guard skyColor.isFiniteVector, groundColor.isFiniteVector,
              simd_reduce_min(skyColor) >= 0, simd_reduce_min(groundColor) >= 0 else {
            throw Failure.hemisphereColor
        }
        guard direction.isFiniteVector, direction != .zero else { throw Failure.direction }
        let axis = simd_normalize(SIMD3<Double>(direction))
        let sky = SIMD3<Double>(skyColor)
        let ground = SIMD3<Double>(groundColor)
        let mean = (sky + ground) / (2 * Basis.diffuseConstant)
        let contrast = (sky - ground) / (2 * Basis.diffuseLinear)
        var coefficients = [SIMD3<Float>](repeating: .zero, count: Self.coefficientCount)
        coefficients[0] = SIMD3<Float>(mean)
        coefficients[1] = SIMD3<Float>(contrast * axis.y)
        coefficients[2] = SIMD3<Float>(contrast * axis.z)
        coefficients[3] = SIMD3<Float>(contrast * axis.x)
        return try LightProbe(coefficients: coefficients, intensity: intensity)
    }

    /// Projects a mip of a native Metal cube into nine linear radiance coefficients.
    /// Supports RGBA8, sRGB RGBA8, RGBA16Float and RGBA32Float. Alpha is ignored.
    /// Each face is read asynchronously in Metal order. Submit tracked producer work
    /// on the supplied queue first and keep the cube unchanged until this call completes.
    public init(
        cube: any MTLTexture, commandQueue: any MTLCommandQueue, mipLevel: Int = 0, intensity: Float = 1
    ) async throws {
        try await self.init(projectingCube: cube, commandQueue: commandQueue, mipLevel: mipLevel, intensity: intensity)
    }

    // A grid can project its scratch texture on the same executor that owns it.
    nonisolated(nonsending) init(
        projectingCube cube: any MTLTexture, commandQueue: any MTLCommandQueue, mipLevel: Int = 0, intensity: Float = 1
    ) async throws {
        guard intensity.isFinite, intensity >= 0 else { throw Failure.intensity }
        guard cube.device === commandQueue.device else { throw Failure.wrongDevice }
        guard cube.textureType == .typeCube, cube.width == cube.height,
              let format = CubeColorFormat(cube.pixelFormat) else { throw Failure.cube }
        guard (0..<cube.mipmapLevelCount).contains(mipLevel) else { throw Failure.mip }
        let size = max(cube.width >> mipLevel, 1)
        guard size <= Self.maximumFaceSize else { throw Failure.faceSize }
        var coefficients = [SIMD3<Double>](repeating: .zero, count: Self.coefficientCount)
        var weightSum = 0.0
        for face in 0..<6 {
            let data = try await TextureTransfer.readOnCaller(
                from: cube, mipLevel: mipLevel, slice: face, commandQueue: commandQueue
            )
            try Self.accumulate(data, format: format, face: face, size: size, coefficients: &coefficients, weightSum: &weightSum)
        }
        let normalization = (4 * Double.pi) / weightSum
        try self.init(coefficients: coefficients.map { SIMD3<Float>($0 * normalization) }, intensity: intensity)
    }

    public mutating func setIntensity(_ intensity: Float) throws {
        try Self.validate(intensity: intensity, coefficients: coefficients)
        self.intensity = intensity
    }

    /// Evaluates reconstructed radiance. Direction may have any finite nonzero length.
    public func radiance(in direction: SIMD3<Float>) throws -> SIMD3<Float> {
        try evaluate(direction, diffuse: false)
    }

    /// Evaluates cosine-convolved irradiance at a surface normal, before multiplying
    /// by the Lambertian BRDF. Low-order reconstruction can produce signed values.
    public func irradiance(at normal: SIMD3<Float>) throws -> SIMD3<Float> {
        try evaluate(normal, diffuse: true)
    }

    private func evaluate(_ direction: SIMD3<Float>, diffuse: Bool) throws -> SIMD3<Float> {
        guard direction.isFiniteVector, direction != .zero else { throw Failure.direction }
        let basis = Basis(simd_normalize(SIMD3<Double>(direction)), diffuse: diffuse)
        return SIMD3<Float>(evaluate(basis))
    }

    func evaluate(_ basis: Basis) -> SIMD3<Double> {
        var value = SIMD3<Double>.zero
        for index in 0..<4 {
            value += SIMD3<Double>(coefficients[index]) * basis.low[index]
            value += SIMD3<Double>(coefficients[index + 4]) * basis.high[index]
        }
        value += SIMD3<Double>(coefficients[8]) * basis.last
        return value * Double(intensity)
    }

    private static func validate(intensity: Float, coefficients: [SIMD3<Float>]) throws {
        guard intensity.isFinite, intensity >= 0 else { throw Failure.intensity }
        // Reserve room for summing the nine weighted terms in shader Float arithmetic.
        let limit = coefficientLimit / max(Double(intensity), 1)
        guard coefficients.allSatisfy({ Double(simd_reduce_max(simd_abs($0))) <= limit }) else {
            throw Failure.coefficientRange
        }
    }

    static func rasterLighting(_ probes: [LightProbe], ambient: SIMD3<Double> = .zero) throws -> [SIMD4<Float>] {
        guard simd_reduce_max(ambient) <= coefficientLimit else { throw AmbientLight.Failure.irradianceRange }
        guard !probes.isEmpty || ambient != .zero else { return noRasterLighting }
        var sum = [SIMD3<Double>](repeating: .zero, count: coefficientCount)
        for probe in probes where probe.intensity > 0 {
            for index in sum.indices {
                sum[index] += SIMD3<Double>(probe.coefficients[index]) * Double(probe.intensity)
            }
        }
        guard sum.allSatisfy({ simd_reduce_max(simd_abs($0)) <= coefficientLimit }) else {
            throw Failure.coefficientRange
        }
        let hasProbes = sum.contains { $0 != .zero }
        guard ambient != .zero || hasProbes else { return noRasterLighting }
        var packed = sum.map { SIMD4(SIMD3<Float>($0), 0) }
        packed[0].w = hasProbes ? 1 : 2
        packed[1].w = Float(ambient.x)
        packed[2].w = Float(ambient.y)
        packed[3].w = Float(ambient.z)
        return packed
    }

    struct Basis {
        static let diffuseConstant = 0.886227
        static let diffuseLinear = 1.023328
        let low: SIMD4<Double>
        let high: SIMD4<Double>
        let last: Double

        init(_ direction: SIMD3<Double>, diffuse: Bool) {
            let x = direction.x, y = direction.y, z = direction.z
            if diffuse {
                // The L0, L1 and L2 radiance bands convolved by the clamped cosine.
                low = SIMD4(Self.diffuseConstant, Self.diffuseLinear * y, Self.diffuseLinear * z, Self.diffuseLinear * x)
                high = SIMD4(0.858086 * x * y, 0.858086 * y * z, 0.743125 * z * z - 0.247708, 0.858086 * x * z)
                last = 0.429043 * (x * x - y * y)
            } else {
                low = SIMD4(0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x)
                high = SIMD4(1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z)
                last = 0.546274 * (x * x - y * y)
            }
        }
    }

    private static func accumulate(
        _ data: TextureTransfer.Readback, format: CubeColorFormat, face: Int, size: Int,
        coefficients: inout [SIMD3<Double>], weightSum: inout Double
    ) throws {
        let step = 2 / Double(size)
        try data.bytes.withUnsafeBytes { bytes in
            for y in 0..<size {
                try Task.checkCancellation()
                let v = (Double(y) + 0.5) * step - 1
                for x in 0..<size {
                    let u = (Double(x) + 0.5) * step - 1
                    let color = format.color(bytes, at: y * data.bytesPerRow + x * format.stride)
                    guard color.x.isFinite, color.y.isFinite, color.z.isFinite else { throw Failure.cubeColor }
                    let coordinate = cubeCoordinate(face: face, u: u, v: v)
                    let squared = simd_length_squared(coordinate)
                    let length = squared.squareRoot()
                    // Solid-angle midpoint quadrature, normalized to 4π across the cube.
                    let weight = 4 / (squared * length)
                    let basis = Basis(coordinate / length, diffuse: false)
                    let weighted = color * weight
                    for index in 0..<4 {
                        coefficients[index] += weighted * basis.low[index]
                        coefficients[index + 4] += weighted * basis.high[index]
                    }
                    coefficients[8] += weighted * basis.last
                    weightSum += weight
                }
            }
        }
    }

    private static func cubeCoordinate(face: Int, u: Double, v: Double) -> SIMD3<Double> {
        // Native top-left Metal texel axes, matching EnvironmentLighting.metal.
        switch face {
        case 0: SIMD3(1, -v, -u)
        case 1: SIMD3(-1, -v, u)
        case 2: SIMD3(u, 1, v)
        case 3: SIMD3(u, -1, -v)
        case 4: SIMD3(u, -v, 1)
        default: SIMD3(-u, -v, -1)
        }
    }

    private enum CubeColorFormat {
        case rgba8, sRGB, half, float

        init?(_ format: MTLPixelFormat) {
            switch format {
            case .rgba8Unorm: self = .rgba8
            case .rgba8Unorm_srgb: self = .sRGB
            case .rgba16Float: self = .half
            case .rgba32Float: self = .float
            default: return nil
            }
        }

        var stride: Int {
            switch self {
            case .rgba8, .sRGB: 4
            case .half: 8
            case .float: 16
            }
        }

        private static let linearSRGB = (0...255).map { Double(GLTFColorTables.sRGBToLinear(Float($0) / 255)) }

        func color(_ bytes: UnsafeRawBufferPointer, at offset: Int) -> SIMD3<Double> {
            switch self {
            case .rgba8:
                return SIMD3(Double(bytes[offset]), Double(bytes[offset + 1]), Double(bytes[offset + 2])) / 255
            case .sRGB:
                return SIMD3(Self.linearSRGB[Int(bytes[offset])], Self.linearSRGB[Int(bytes[offset + 1])],
                             Self.linearSRGB[Int(bytes[offset + 2])])
            case .half:
                return SIMD3(Double(bytes.loadUnaligned(fromByteOffset: offset, as: Float16.self)),
                             Double(bytes.loadUnaligned(fromByteOffset: offset + 2, as: Float16.self)),
                             Double(bytes.loadUnaligned(fromByteOffset: offset + 4, as: Float16.self)))
            case .float:
                return SIMD3(Double(bytes.loadUnaligned(fromByteOffset: offset, as: Float.self)),
                             Double(bytes.loadUnaligned(fromByteOffset: offset + 4, as: Float.self)),
                             Double(bytes.loadUnaligned(fromByteOffset: offset + 8, as: Float.self)))
            }
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case coefficients, coefficientRange, intensity, direction, wrongDevice, cube, cubeColor, mip, faceSize
        case hemisphereColor

        public var errorDescription: String? {
            switch self {
            case .coefficients: "Supply nine finite RGB spherical-harmonic coefficients."
            case .coefficientRange: "The scaled light probe coefficients exceed the supported Float range."
            case .intensity: "Use a finite, nonnegative light probe intensity."
            case .direction: "Use a finite, nonzero direction for light probe evaluation."
            case .hemisphereColor: "Use finite, nonnegative linear RGB sky and ground irradiance."
            case .wrongDevice: "Use a command queue on the cube's Metal device."
            case .cube: "Use a Metal cube in RGBA8, sRGB RGBA8, RGBA16Float or RGBA32Float."
            case .cubeColor: "The cube contains a nonfinite RGB value."
            case .mip: "The selected cube mip does not exist."
            case .faceSize: "Select a cube mip of at most \(LightProbe.maximumFaceSize) texels per side."
            }
        }
    }
}
