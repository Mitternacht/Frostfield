import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct EnvironmentPresetTests {
    @Test("Named panoramas keep distinct zenith colors and prepare lighting")
    func presets() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let daylight = try EnvironmentLighting.equirectangularTexture(for: .daylight, device: device, width: 8)
        let night = try EnvironmentLighting.equirectangularTexture(for: .night, device: device, width: 8)
        let dayPixel = pixel(daylight)
        let nightPixel = pixel(night)
        #expect(dayPixel.z > dayPixel.x)
        #expect(dayPixel.z > nightPixel.z * 4)

        let queue = try #require(device.makeCommandQueue())
        let settings = EnvironmentLighting.Settings(cubeSize: 8, diffuseSize: 4, brdfSize: 8, sampleCount: 16)
        let lighting = try EnvironmentLighting(queue: queue, preset: .sunset, settings: settings)
        #expect(lighting.skyCube.width == 8)
    }

    @Test("One capture prepares lighting from the scene clear color")
    func capture() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 20)
        let graph = SceneGraph(roots: [], camera: camera)
        let settings = EnvironmentLighting.Settings(cubeSize: 8, diffuseSize: 4, brdfSize: 8, sampleCount: 16)
        let lighting = try EnvironmentLighting.capture(
            graph, at: .zero, queue: queue, size: 8, clearColor: MTLClearColor(red: 0.8, green: 0.1, blue: 0.05, alpha: 1),
            settings: settings
        )
        let diffuse = try read(lighting.diffuseCube, queue: queue)
        #expect(diffuse[0] > diffuse[2] * 2)
        #expect(throws: CubeCamera.Failure.position) {
            try EnvironmentLighting.capture(graph, at: SIMD3(.nan, 0, 0), queue: queue, settings: settings)
        }
    }

    private func pixel(_ texture: any MTLTexture) -> SIMD3<Float> {
        var bytes = [UInt16](repeating: 0, count: 4)
        bytes.withUnsafeMutableBytes {
            texture.getBytes($0.baseAddress!, bytesPerRow: 8, from: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0)
        }
        return SIMD3(bytes.prefix(3).map { Float(Float16(bitPattern: $0)) })
    }

    private func read(_ texture: any MTLTexture, queue: any MTLCommandQueue) throws -> [Float] {
        let stride = 8
        let buffer = try #require(queue.device.makeBuffer(length: stride, options: .storageModeShared))
        let command = try #require(queue.makeCommandBuffer())
        let blit = try #require(command.makeBlitCommandEncoder())
        blit.copy(
            from: texture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: 1, height: 1, depth: 1), to: buffer, destinationOffset: 0,
            destinationBytesPerRow: stride, destinationBytesPerImage: stride
        )
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed)
        let values = buffer.contents().bindMemory(to: Float16.self, capacity: 4)
        return (0..<4).map { Float(values[$0]) }
    }
}
