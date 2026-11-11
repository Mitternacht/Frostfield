import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct DepthOfFieldTests {
    @Test("Perspective depth blurs a defocused plane, preserves focus and rejects the other layer")
    func gathersAtMatchingDepth() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let effect = try DepthOfField(device: device)
        let camera = try PerspectiveCamera(verticalFieldOfView: 0.65, near: 0.1, far: 10)
        let settings = DepthOfField.Settings(focusDistance: 1, fNumber: 1, sensorHeight: 0.16, maximumRadius: 4)
        let width = 19
        let height = 9
        let split = 8
        let row = height / 2
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = [.shaderRead, .renderTarget]
        let color = try #require(device.makeTexture(descriptor: colorDescriptor))
        let output = try #require(device.makeTexture(descriptor: colorDescriptor))
        let input: [SIMD4<Float16>] = (0..<width * height).map { index in
            let (x, y) = (index % width, index / width)
            if x == split - 1 { return SIMD4(0, 0, 2, 0.5) }
            if x == 4 && y == row { return SIMD4(0, 1, 0, 0.5) }
            if x == 13 && y == row { return SIMD4(1, 0, 0, 0.5) }
            return SIMD4(0, 0, 0, 0.5)
        }
        let region = MTLRegionMake2D(0, 0, width, height)
        try input.withUnsafeBytes { bytes in
            color.replace(region: region, mipmapLevel: 0, withBytes: try #require(bytes.baseAddress), bytesPerRow: width * 8)
        }
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = [.renderTarget, .shaderRead]
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))

        let library = try device.makeLibrary(source: Self.depthSource, options: nil)
        let vertex: any MTLFunction = try #require(library.makeFunction(name: "test_dof_depth"))
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertex
        pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        let pipeline = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        let stateDescriptor = MTLDepthStencilDescriptor()
        stateDescriptor.depthCompareFunction = .always
        stateDescriptor.isDepthWriteEnabled = true
        let state = try #require(device.makeDepthStencilState(descriptor: stateDescriptor))
        let projection = try camera.projection(aspectRatio: Float(width) / Float(height))
        let near = projection * SIMD4<Float>(0, 0, -1, 1)
        let far = projection * SIMD4<Float>(0, 0, -8, 1)

        func render(reversed: Bool, settings: DepthOfField.Settings) throws -> [SIMD4<Float16>] {
            let buffer = try #require(queue.makeCommandBuffer())
            let pass = MTLRenderPassDescriptor()
            pass.depthAttachment.texture = depth
            pass.depthAttachment.loadAction = .dontCare
            pass.depthAttachment.storeAction = .store
            let encoder = try #require(buffer.makeRenderCommandEncoder(descriptor: pass))
            encoder.setRenderPipelineState(pipeline)
            encoder.setDepthStencilState(state)
            for (x, count, clip) in [(0, split, near), (split, width - split, far)] {
                var value = reversed ? 1 - clip.z / clip.w : clip.z / clip.w
                encoder.setScissorRect(MTLScissorRect(x: x, y: 0, width: count, height: height))
                encoder.setVertexBytes(&value, length: MemoryLayout<Float>.size, index: 0)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            }
            encoder.endEncoding()

            try effect.encode(
                source: color, depth: depth, destination: output, camera: camera,
                depthMode: reversed ? .reversed : .forward, settings: settings, commandBuffer: buffer
            )
            buffer.commit()
            buffer.waitUntilCompleted()
            try #require(buffer.status == .completed)
            var result = Array(repeating: SIMD4<Float16>.zero, count: input.count)
            try result.withUnsafeMutableBytes { bytes in
                output.getBytes(try #require(bytes.baseAddress), bytesPerRow: width * 8, from: region, mipmapLevel: 0)
            }
            return result
        }

        let forward = try render(reversed: false, settings: settings)
        let reverse = try render(reversed: true, settings: settings)
        for result in [forward, reverse] {
            #expect(result[row * width + 4] == input[row * width + 4])
            #expect(result[row * width + split].z == 0)
            #expect((0.02...0.65).contains(Float(result[row * width + 13].x)))
            #expect(result[row * width + 15].x > 0.01)
            #expect(result.allSatisfy { $0.w == 0.5 })
        }
        var disabled = settings
        disabled.maximumRadius = 0.5
        #expect(try render(reversed: false, settings: disabled) == input)
    }

    private static let depthSource = """
    #include <metal_stdlib>
    using namespace metal;
    vertex float4 test_dof_depth(uint vertexID [[vertex_id]], constant float& depth [[buffer(0)]]) {
        const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
        return float4(positions[vertexID], depth, 1);
    }
    """
}
