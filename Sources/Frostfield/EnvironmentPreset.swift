import Metal
import simd

/// A generated linear panorama, named like a drei environment preset.
public enum EnvironmentPreset: Equatable, Sendable, CaseIterable {
    case daylight
    case sunset
    case night

    /// Linear radiance at the zenith and the horizon.
    public var gradient: (zenith: SIMD3<Float>, horizon: SIMD3<Float>) {
        switch self {
        case .daylight: (SIMD3(0.25, 0.45, 0.85), SIMD3(0.75, 0.82, 0.9))
        case .sunset: (SIMD3(0.12, 0.08, 0.22), SIMD3(0.95, 0.35, 0.08))
        case .night: (SIMD3(0.01, 0.015, 0.04), SIMD3(0.04, 0.05, 0.08))
        }
    }
}

extension EnvironmentLighting {
    /// A 2:1 linear RGBA16Float panorama. The top row is the zenith.
    public static func equirectangularTexture(
        for preset: EnvironmentPreset, device: any MTLDevice, width: Int = 64
    ) throws -> any MTLTexture {
        guard (2...8_192).contains(width), width.isMultiple(of: 2) else { throw Failure.inputTexture }
        let height = width / 2
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw Failure.resource("preset panorama") }
        texture.label = "Frostfield \(preset) panorama"
        let gradient = preset.gradient
        var pixels = [Float16](repeating: 0, count: width * height * 4)
        for row in 0..<height {
            let towardHorizon = min(Float(row) / Float(max(height / 2, 1)), 1)
            let color = gradient.zenith + (gradient.horizon - gradient.zenith) * towardHorizon
            for column in 0..<width {
                let index = (row * width + column) * 4
                pixels[index] = Float16(color.x)
                pixels[index + 1] = Float16(color.y)
                pixels[index + 2] = Float16(color.z)
                pixels[index + 3] = 1
            }
        }
        pixels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: bytes.baseAddress!, bytesPerRow: width * 8
            )
        }
        return texture
    }

    /// Prepares split-sum lighting from a named panorama.
    public convenience init(
        queue: any MTLCommandQueue, preset: EnvironmentPreset, settings: Settings = .init()
    ) throws {
        let panorama = try Self.equirectangularTexture(for: preset, device: queue.device)
        try self.init(queue: queue, equirectangular: panorama, settings: settings)
    }

    /// Captures the scene into a cube and prepares lighting from it.
    /// Ground projection of a finished panorama is `GroundedSkybox`.
    public static func capture(
        _ graph: SceneGraph, at position: SIMD3<Float>, queue: any MTLCommandQueue,
        size: Int = 32, near: Float = 0.1, far: Float = 100,
        clearColor: MTLClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1),
        settings: Settings = Settings(cubeSize: 32, diffuseSize: 8, brdfSize: 32, sampleCount: 16)
    ) throws -> EnvironmentLighting {
        let target = try CubeCamera.makeTarget(device: queue.device, size: size, mipmapped: false)
        let camera = try CubeCamera(queue: queue, target: target, near: near, far: far)
        guard let commandBuffer = queue.makeCommandBuffer() else { throw Failure.commandBuffer }
        try camera.encode(graph, at: position, clearColor: clearColor, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else { throw Failure.execution(commandBuffer.error?.localizedDescription) }
        return try EnvironmentLighting(queue: queue, cube: target, settings: settings)
    }
}
