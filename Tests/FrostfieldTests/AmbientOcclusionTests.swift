@testable import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct AmbientOcclusionTests {
    @Test("Screen-space occlusion darkens nearby recesses in forward and reversed depth",
          arguments: [Float(1), 0.5])
    func darkensWithinRadius(scale: Float) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let occlusion = try AmbientOcclusion(device: device)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 30)
        let width = 41
        let height = 25
        let split = 18
        let skyRows = 3
        let region = MTLRegionMake2D(0, 0, width, height)
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = [.shaderRead, .renderTarget]
        let source = try #require(device.makeTexture(descriptor: colorDescriptor))
        let destination = try #require(device.makeTexture(descriptor: colorDescriptor))
        let pixel = SIMD4<Float16>(0.5, 0.25, 0.125, 0.5)
        let input = Array(repeating: pixel, count: width * height)
        try input.withUnsafeBytes { bytes in
            source.replace(region: region, mipmapLevel: 0, withBytes: try #require(bytes.baseAddress), bytesPerRow: width * 8)
        }
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = [.renderTarget, .shaderRead]
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))

        let library = try device.makeLibrary(source: Self.depthSource, options: nil)
        let vertex: any MTLFunction = try #require(library.makeFunction(name: "test_ao_depth"))
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertex
        pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        let pipeline = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = .always
        depthStateDescriptor.isDepthWriteEnabled = true
        let depthState = try #require(device.makeDepthStencilState(descriptor: depthStateDescriptor))
        let projection = try camera.projection(aspectRatio: Float(width) / Float(height))
        let raised = projection * SIMD4<Float>(0, 0, -2.65, 1)
        let recessed = projection * SIMD4<Float>(0, 0, -3, 1)

        func render(_ mode: MeshDepthMode, _ settings: AmbientOcclusion.Settings) throws -> [SIMD4<Float16>] {
            let buffer = try #require(queue.makeCommandBuffer())
            let pass = MTLRenderPassDescriptor()
            pass.depthAttachment.texture = depth
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.storeAction = .store
            pass.depthAttachment.clearDepth = mode == .reversed ? 0 : 1
            let encoder = try #require(buffer.makeRenderCommandEncoder(descriptor: pass))
            encoder.setRenderPipelineState(pipeline)
            encoder.setDepthStencilState(depthState)
            for (x, count, clip) in [(0, split, raised), (split, width - split, recessed)] {
                var value = mode == .reversed ? 1 - clip.z / clip.w : clip.z / clip.w
                encoder.setScissorRect(MTLScissorRect(x: x, y: skyRows, width: count, height: height - skyRows))
                encoder.setVertexBytes(&value, length: MemoryLayout<Float>.size, index: 0)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            }
            encoder.endEncoding()
            try occlusion.encode(
                source: source, depth: depth, destination: destination, camera: camera,
                depthMode: mode, settings: settings, commandBuffer: buffer
            )
            buffer.commit()
            buffer.waitUntilCompleted()
            try #require(buffer.status == .completed)
            var output = Array(repeating: SIMD4<Float16>.zero, count: input.count)
            try output.withUnsafeMutableBytes { bytes in
                destination.getBytes(try #require(bytes.baseAddress), bytesPerRow: width * 8, from: region, mipmapLevel: 0)
            }
            return output
        }

        let settings = AmbientOcclusion.Settings(radius: 1.3, strength: 1, resolutionScale: scale)
        let row = height / 2 * width
        let forward = try render(.forward, settings)
        let reversed = try render(.reversed, settings)
        #expect(occlusion.horizonTextureSize == (scale == 1 ? SIMD2(41, 25) : SIMD2(21, 13)))
        for result in [forward, reversed] {
            let nearContact = result[row + split + 2]
            #expect((0.36...0.47).contains(Float(nearContact.x)))
            #expect(nearContact.y == nearContact.x * 0.5)
            #expect(nearContact.w == pixel.w)
            #expect(result[row + width - 7] == pixel)
            #expect(result[row + split - 1] == pixel)
            #expect(result[0..<(skyRows * width)].allSatisfy { $0 == pixel })
        }
        var poisson = settings
        poisson.poisson = .init(radius: 5, depthPhi: 0.05)
        let filtered = try render(.reversed, poisson)
        #expect(filtered[row + split + 2].x < 0.48)
        #expect(filtered[row + split - 1] == pixel)
        #expect(filtered[0..<(skyRows * width)].allSatisfy { $0 == pixel })
        #expect(filtered.allSatisfy { $0.w == pixel.w })
        #expect(abs(Float(forward[row + split + 2].x - reversed[row + split + 2].x)) < 0.001)
        #expect(try render(.forward, .init(radius: 1.3, strength: 0)) == input)
        #expect(try render(.forward, .init(radius: 0.05, strength: 1)) == input)
    }

    private static let depthSource = """
    #include <metal_stdlib>
    using namespace metal;
    vertex float4 test_ao_depth(uint vertexID [[vertex_id]], constant float& depth [[buffer(0)]]) {
        const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
        return float4(positions[vertexID], depth, 1);
    }
    """
}
