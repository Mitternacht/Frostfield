import CoreGraphics
import Foundation
import Frostfield
import ImageIO
import Metal
import Testing
import UniformTypeIdentifiers
import simd

/// Traces and rasterizes small scenes with an explicit camera pose, and computes the
/// analytic and Monte Carlo references of the participating-media suites.
struct MediaFixture {
    let device: any MTLDevice
    let queue: any MTLCommandQueue

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
    }

    func renderer(_ geometry: MeshGeometry, _ material: PhysicallyBasedMaterial) throws -> MeshRenderer {
        try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(material))
    }

    /// Accumulates `samples` per pixel in batches of up to 16 and returns linear RGBA.
    func trace(
        _ graph: SceneGraph, camera: PerspectiveCamera, pose: SceneTransform = .init(), width: Int, height: Int,
        samples: Int, settings: PathTracer.Settings, lighting: PathTracer.Lighting,
        options: PathTracingScene.Options = .init(textures: .baked(size: 16))
    ) throws -> [SIMD4<Float>] {
        let scene = try PathTracingScene(queue: queue, graph: graph, options: options)
        let tracer = try PathTracer(queue: queue, scene: scene, lighting: lighting)
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let target = try output(width: width, height: height)
        var history: PathTracer.History?
        while (history?.sampleCount ?? 0) < samples {
            let buffer = try #require(queue.makeCommandBuffer())
            history = try tracer.encode(
                camera: camera, cameraTransform: pose, destination: target, accumulation: accumulation,
                previous: history, commandBuffer: buffer, samplesPerPixel: min(16, samples - (history?.sampleCount ?? 0)),
                settings: settings
            )
            try finish(buffer)
        }
        return try readback(target)
    }

    /// Draws the graph with the raster renderer over an opaque black clear.
    func rasterize(_ graph: SceneGraph, width: Int, height: Int) throws -> [SIMD4<Float>] {
        let color = try output(width: width, height: height, usage: [.renderTarget, .shaderRead])
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        let buffer = try #require(queue.makeCommandBuffer())
        try graph.encode(pass: pass, commandBuffer: buffer)
        try finish(buffer)
        return try readback(color)
    }

    func output(width: Int, height: Int, usage: MTLTextureUsage = [.shaderWrite, .shaderRead]) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = usage
        descriptor.hazardTrackingMode = .tracked
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    func finish(_ buffer: any MTLCommandBuffer) throws {
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed, "\(buffer.error?.localizedDescription ?? "No Metal error")")
    }

    func readback(_ texture: any MTLTexture) throws -> [SIMD4<Float>] {
        var result = Array(repeating: SIMD4<Float16>.zero, count: texture.width * texture.height)
        try result.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: texture.width * 8,
                from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0
            )
        }
        return result.map(SIMD4<Float>.init)
    }

    /// An RGBA8 linear data texture.
    func dataTexture(width: Int, height: Int, pixels: [UInt8]) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false
        )
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        pixels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: bytes.baseAddress!,
                bytesPerRow: width * 4
            )
        }
        return texture
    }

    /// The mean RGB of the pixels in a centered square block.
    static func center(_ image: [SIMD4<Float>], width: Int, height: Int, radius: Int) -> SIMD3<Float> {
        var sum = SIMD3<Float>.zero
        var count: Float = 0
        for y in (height / 2 - radius)...(height / 2 + radius) {
            for x in (width / 2 - radius)...(width / 2 + radius) {
                let pixel = image[y * width + x]
                sum += SIMD3(pixel.x, pixel.y, pixel.z)
                count += 1
            }
        }
        return sum / count
    }

    /// Writes a clamped sRGB PNG when the FROSTFIELD_MEDIA_IMAGES environment variable names a directory.
    static func save(_ image: [SIMD4<Float>], width: Int, height: Int, name: String, exposure: Float = 1) {
        guard let directory = ProcessInfo.processInfo.environment["FROSTFIELD_MEDIA_IMAGES"] else { return }
        var bytes: [UInt8] = []
        for pixel in image {
            for channel in 0..<3 {
                let linear = min(max(pixel[channel] * exposure, 0), 1)
                let encoded = linear <= 0.0031308 ? 12.92 * linear : 1.055 * pow(linear, 1 / 2.4) - 0.055
                bytes.append(UInt8((encoded * 255).rounded()))
            }
            bytes.append(255)
        }
        let data = Data(bytes) as CFData
        guard let provider = CGDataProvider(data: data),
              let cgImage = CGImage(
                  width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                  space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
              ) else { return }
        let url = URL(fileURLWithPath: directory).appendingPathComponent("\(name).png")
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(destination, cgImage, nil)
        CGImageDestinationFinalize(destination)
    }
}

/// Closed-form and numerical references for scattering media.
enum MediaReference {
    /// Kulla and Conty's single-scattering albedo for a multiple-scattering albedo, as KHR_materials_volume_scatter specifies.
    static func singleScatteringAlbedo(_ color: Double) -> Double {
        let root = 4.09712 + 4.20863 * color - (9.59217 + 41.6808 * color + 17.7126 * color * color).squareRoot()
        return 1 - root * root
    }

    /// Gauss-Legendre nodes and weights on [0, 1].
    static func gaussLegendre(_ count: Int) -> [(x: Double, w: Double)] {
        (1...count).map { index in
            var x = cos(Double.pi * (Double(index) - 0.25) / (Double(count) + 0.5))
            var derivative = 0.0
            for _ in 0..<100 {
                var previous = 1.0
                var current = x
                for k in 2...count {
                    (previous, current) = (current, ((2 * Double(k) - 1) * x * current - (Double(k) - 1) * previous) / Double(k))
                }
                derivative = Double(count) * (x * current - previous) / (x * x - 1)
                let step = current / derivative
                x -= step
                if abs(step) < 1e-15 { break }
            }
            return (0.5 * (x + 1), 1 / ((1 - x * x) * derivative * derivative))
        }
    }

    /// Chandrasekhar's H-function for isotropic scattering with single-scattering albedo
    /// `albedo`, solved by iterating 1 / H(mu) = sqrt(1 - a) + (a / 2) Int mu' H(mu') / (mu + mu') dmu'.
    static func chandrasekharH(albedo: Double) -> (Double) -> Double {
        let nodes = gaussLegendre(64)
        let root = (1 - albedo).squareRoot()
        var values = Array(repeating: 1.0, count: nodes.count)
        func evaluate(_ mu: Double, _ h: [Double]) -> Double {
            var sum = 0.0
            for (node, value) in zip(nodes, h) { sum += node.w * node.x * value / (mu + node.x) }
            return 1 / (root + 0.5 * albedo * sum)
        }
        for _ in 0..<5_000 {
            let next = nodes.map { evaluate($0.x, values) }
            let change = zip(next, values).map { abs($0 - $1) }.max() ?? 0
            values = next
            if change < 1e-13 { break }
        }
        let solved = values
        return { evaluate($0, solved) }
    }

    /// The plane albedo of a semi-infinite, index-matched, isotropically scattering medium for
    /// light incident at cosine mu: 1 - H(mu) sqrt(1 - a) (Chandrasekhar 1960).
    static func planeAlbedo(albedo: Double, cosine: Double) -> Double {
        1 - chandrasekharH(albedo: albedo)(cosine) * (1 - albedo).squareRoot()
    }

    /// The Henyey-Greenstein density over the scattering cosine (integrating to one over [-1, 1]).
    static func henyeyGreensteinCDF(_ mu: Double, g: Double) -> Double {
        guard abs(g) > 1e-6 else { return (mu + 1) / 2 }
        return (1 - g * g) / (2 * g) * (1 / (1 + g * g - 2 * g * mu).squareRoot() - 1 / (1 + g))
    }

    static func henyeyGreenstein(_ mu: Double, g: Double) -> Double {
        let denominator = 1 + g * g - 2 * g * mu
        return (1 - g * g) / (4 * Double.pi * denominator * denominator.squareRoot())
    }

    /// Total reflectance and transmittance of an index-matched slab of optical depth `depth`
    /// for a collimated beam at normal incidence, by an analog Monte Carlo random walk with
    /// Henyey-Greenstein scattering. Weights carry the albedo; roulette ends faint photons.
    static func slab(depth: Double, albedo: Double, g: Double, photons: Int, seed: UInt64) -> (reflectance: Double, transmittance: Double) {
        var state = seed | 1
        func random() -> Double {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return Double(state >> 11) * 0x1p-53
        }
        var reflected = 0.0
        var transmitted = 0.0
        for _ in 0..<photons {
            var z = 0.0
            var mu = 1.0
            var weight = 1.0
            while true {
                z += mu * -log(1 - random())
                if z < 0 { reflected += weight; break }
                if z > depth { transmitted += weight; break }
                weight *= albedo
                if weight < 0.01 {
                    if random() > 0.1 { break }
                    weight /= 0.1
                }
                // Scatter by the Henyey-Greenstein cosine about the current direction.
                let u = random()
                let cosine: Double
                if abs(g) < 1e-6 {
                    cosine = 1 - 2 * u
                } else {
                    let ratio = (1 - g * g) / (1 - g + 2 * g * u)
                    cosine = (1 + g * g - ratio * ratio) / (2 * g)
                }
                let sine = max(0, 1 - cosine * cosine).squareRoot()
                let phi = 2 * Double.pi * random()
                mu = min(max(mu * cosine + max(0, 1 - mu * mu).squareRoot() * sine * cos(phi), -1), 1)
            }
        }
        return (reflected / Double(photons), transmitted / Double(photons))
    }

    /// EON (Portsmouth, Kutz and Hill 2025), transcribed from the authors' reference listing for
    /// local directions with z along the normal; returns the BRDF without the cosine.
    static func eon(albedo rho: Double, roughness r: Double, incoming wi: SIMD3<Double>, outgoing wo: SIMD3<Double>) -> Double {
        let constant1 = 0.5 - 2 / (3 * Double.pi)
        let constant2 = 2.0 / 3 - 28 / (15 * Double.pi)
        func albedo(_ mu: Double) -> Double {
            let a = 1 / (1 + constant1 * r)
            let b = r * a
            let m = min(max(mu, -1), 1)
            let si = (1 - m * m).squareRoot()
            let g = si * (acos(m) - si * m) + (2.0 / 3) * (si * m * (1 + si + si * si) / (1 + si) - si)
            return a + b / Double.pi * g
        }
        let muI = wi.z
        let muO = wo.z
        let s = simd_dot(wi, wo) - muI * muO
        let sOverT = s > 0 ? s / max(muI, muO) : s
        let a = 1 / (1 + constant1 * r)
        let single = rho / Double.pi * a * (1 + r * sOverT)
        let average = a * (1 + constant2 * r)
        let multipleAlbedo = rho * rho * average / (1 - rho * (1 - average))
        let epsilon = 1e-7
        let multiple = multipleAlbedo / Double.pi * max(epsilon, 1 - albedo(muO)) * max(epsilon, 1 - albedo(muI))
            / max(epsilon, 1 - average)
        return single + multiple
    }
}
