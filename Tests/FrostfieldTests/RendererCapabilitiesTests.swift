import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct RendererCapabilitiesTests {
    @Test("Public capabilities agree with native device queries and Frostfield runtime gates")
    func runtimeCapabilities() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let capabilities = RendererCapabilities(device: device)
        let expected: [RendererCapabilities.Feature: Bool] = [
            .hardwareRayTracing: device.supportsRaytracing,
            .tier2ArgumentBuffers: device.argumentBuffersSupport == .tier2,
            .temporalPathDenoising: PathTemporalDenoiser.isSupported(device: device),
            .batchCommandBundles: BatchedMesh.supportsIndirectSubmission(on: device),
            .indexedIndirectArguments: MeshIndirectDraws.isSupported(on: device),
            .stageTimestamps: SceneGPUTimingFrame.isSupported(on: device),
            .bcTextureCompression: device.supportsBCTextureCompression,
            .appleBlockTextureCompression: device.supportsFamily(.apple2),
            .float32TextureFiltering: device.supports32BitFloatFiltering,
        ]
        #expect(Set(expected.keys) == Set(RendererCapabilities.Feature.allCases))
        for (feature, available) in expected {
            #expect(capabilities.hasFeature(feature) == available, "\(feature) on \(capabilities.deviceName)")
        }
        let threads = device.maxThreadsPerThreadgroup
        #expect(capabilities.deviceName == device.name)
        #expect(capabilities.maximumBufferLength == device.maxBufferLength)
        #expect(capabilities.maximumThreadsPerThreadgroup == SIMD3(threads.width, threads.height, threads.depth))
    }

    @Test("Reported raster sample counts prepare real Metal mesh pipelines")
    func meshSampleCounts() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let capabilities = RendererCapabilities(device: device)
        #expect(capabilities.rasterSampleCounts.contains(1))
        #expect(capabilities.rasterSampleCounts == [1, 2, 4, 8].filter { device.supportsTextureSampleCount($0) })
        let geometry = try MeshGeometry.plane(width: 1, height: 1)
        for count in capabilities.rasterSampleCounts {
            let renderer = try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: geometry, sampleCount: count
            )
            #expect(renderer.sampleCount == count)
        }
    }
}
