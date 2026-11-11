import Foundation
import Frostfield
import Metal
import simd

/// A cyclorama's softboxes and fill, evaluated directly into a linear HDR panorama.
enum LabProceduralLighting {
    private struct Panel {
        let forward: SIMD3<Float>
        let right: SIMD3<Float>
        let up: SIMD3<Float>
        let halfExtent: SIMD2<Float>
        let color: SIMD3<Float>

        init(_ direction: SIMD3<Float>, _ halfExtent: SIMD2<Float>, _ color: SIMD3<Float>) {
            forward = simd_normalize(direction)
            right = simd_normalize(simd_cross(SIMD3(0, 1, 0), forward))
            up = simd_cross(forward, right)
            self.halfExtent = halfExtent
            self.color = color
        }

        func radiance(_ ray: SIMD3<Float>) -> SIMD3<Float> {
            let depth = simd_dot(ray, forward)
            guard depth > 0 else { return .zero }
            let projected = abs(SIMD2(simd_dot(ray, right), simd_dot(ray, up))) / depth
            let edge = simd_clamp((halfExtent - projected) * 18, .zero, SIMD2(repeating: 1))
            let fade = edge * edge * (3 - 2 * edge)
            return color * (fade.x * fade.y)
        }
    }

    static let panoramaWidth = 512
    static let panoramaHeight = 256

    static func make(queue: any MTLCommandQueue) throws -> EnvironmentLighting {
        let width = panoramaWidth
        let height = panoramaHeight
        let pixels = panorama()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        guard let texture = queue.device.makeTexture(descriptor: descriptor) else {
            throw MeshRenderer.Failure.resource("procedural studio panorama")
        }
        let halves = pixels.map { SIMD4<Float16>(SIMD3<Float16>($0), 1) }
        halves.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: bytes.baseAddress!, bytesPerRow: width * MemoryLayout<SIMD4<Float16>>.stride
            )
        }
        return try EnvironmentLighting(queue: queue, equirectangular: texture)
    }

    /// The same studio panorama for the path tracer's importance-sampled sky.
    static func pathEnvironment(device: any MTLDevice) throws -> PathTracer.EnvironmentMap {
        try PathTracer.EnvironmentMap(device: device, pixels: panorama(), width: panoramaWidth, height: panoramaHeight)
    }

    /// Linear RGB, row-major from the top-left, in the raster equirectangular orientation.
    static func panorama() -> [SIMD3<Float>] {
        let width = panoramaWidth
        let height = panoramaHeight
        let panels: [Panel] = [
            Panel(SIMD3(-0.56, 0.64, 0.58), SIMD2(0.80, 0.38), SIMD3(6.2, 5.1, 3.9)),
            Panel(SIMD3(0.89, 0.29, 0.30), SIMD2(0.23, 0.72), SIMD3(3.2, 4.8, 7.7)),
            Panel(SIMD3(0.32, 0.68, -0.72), SIMD2(0.85, 0.48), SIMD3(6.9, 7.8, 9.8)),
            Panel(SIMD3(-0.19, 0.12, 0.99), SIMD2(0.30, 0.13), SIMD3(2.7, 2.6, 2.3)),
        ]
        var pixels: [SIMD3<Float>] = []
        pixels.reserveCapacity(width * height)
        for y in 0..<height {
            let latitude = (0.5 - (Float(y) + 0.5) / Float(height)) * .pi
            let up = sin(latitude)
            let horizontal = cos(latitude)
            for x in 0..<width {
                let azimuth = ((Float(x) + 0.5) / Float(width) - 0.5) * (2 * .pi)
                let ray = SIMD3(horizontal * cos(azimuth), up, horizontal * sin(azimuth))
                let ambient = SIMD3<Float>(0.19, 0.22, 0.28) * (0.65 + 0.35 * up)
                pixels.append(panels.reduce(ambient) { $0 + $1.radiance(ray) })
            }
        }
        return pixels
    }
}
