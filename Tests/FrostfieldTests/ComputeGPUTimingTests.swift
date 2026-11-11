import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ComputeGPUTimingTests {
    @Test("Timing measures both graph dispatches on one command and preserves texture mips and buffer output")
    func graphs() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let generator = try ShaderTextureGenerator(graph: TextureShaderGraph { t in
            t.constant(SIMD4<Float>(0.25, 0.5, 0.75, 1))
        }, device: device)
        let kernel = try ShaderBufferKernel(graph: BufferShaderGraph(fieldCount: 1) { b in
            [0: b.elementField(0) * 2]
        }, device: device)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: 256, height: 128, mipmapped: true
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let count = 32_768
        let length = count * MemoryLayout<SIMD4<Float>>.stride
        let source = try #require(device.makeBuffer(length: length, options: .storageModeShared))
        let destination = try #require(device.makeBuffer(length: length, options: .storageModeShared))
        let value = SIMD4<Float>(1, 2, 3, 4)
        source.contents().initializeMemory(as: SIMD4<Float>.self, repeating: value, count: count)
        let command = try #require(queue.makeCommandBuffer())
        let timings: [ComputeGPUTiming]
        if ComputeGPUTiming.isSupported(on: device) {
            let textureTime = try generator.encodeTimed(into: texture, commandBuffer: command)
            let bufferTime = try kernel.encodeTimed(elementCount: count, from: source, to: destination, commandBuffer: command)
            #expect(throws: ComputeGPUTiming.Failure.notReady) { try textureTime.duration() }
            #expect(throws: ComputeGPUTiming.Failure.notReady) { try bufferTime.duration() }
            timings = [textureTime, bufferTime]
        } else {
            #expect(throws: ComputeGPUTiming.Failure.unsupported) {
                try generator.encodeTimed(into: texture, commandBuffer: command)
            }
            #expect(throws: ComputeGPUTiming.Failure.unsupported) {
                try kernel.encodeTimed(elementCount: count, from: source, to: destination, commandBuffer: command)
            }
            try generator.encode(into: texture, commandBuffer: command)
            try kernel.encode(elementCount: count, from: source, to: destination, commandBuffer: command)
            timings = []
        }
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed)
        for timing in timings {
            let duration = try timing.duration()
            #expect(duration.isFinite && duration > 0)
            #expect(duration <= (command.gpuEndTime - command.gpuStartTime) * 1.05 + 0.0001)
        }
        let output = destination.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        #expect(output[0] == value * 2)
        #expect(output[count - 1] == value * 2)
        let lastMip = try readRGBA16FloatMip(texture, level: texture.mipmapLevelCount - 1, queue: queue)
        #expect(lastMip == [0.25, 0.5, 0.75, 1])
    }

    @Test("Custom Metal dispatch timing handles unretained commands and rejects completed commands",
          .enabled(if: MTLCreateSystemDefaultDevice().map { ComputeGPUTiming.isSupported(on: $0) } ?? false,
                   "Requires stage-boundary counters"))
    func customCompute() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let source = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void fill(device uint *values [[buffer(0)]], uint i [[thread_position_in_grid]]) {
                values[i] = 2u * i + 1u;
            }
            """
        let library = try device.makeLibrary(source: source, options: nil)
        let function = try #require(library.makeFunction(name: "fill"))
        let pipeline = try device.makeComputePipelineState(function: function)
        let count = 65_536
        let destination = try #require(device.makeBuffer(length: count * 4, options: .storageModeShared))
        let command = try #require(queue.makeCommandBufferWithUnretainedReferences())
        let timing = try ComputeGPUTiming.encode(commandBuffer: command) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(destination, offset: 0, index: 0)
            encoder.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
            )
        }
        command.commit()
        command.waitUntilCompleted()
        withExtendedLifetime(pipeline) {}
        try #require(command.status == .completed)
        #expect(try timing.duration() > 0)
        let output = destination.contents().bindMemory(to: UInt32.self, capacity: count)
        #expect(output[count - 1] == UInt32(2 * count - 1))
        #expect(throws: ComputeGPUTiming.Failure.commandBufferState) {
            try ComputeGPUTiming.encode(commandBuffer: command) { _ in
                Issue.record("Completed command buffers must reject encoding")
            }
        }
    }
}
