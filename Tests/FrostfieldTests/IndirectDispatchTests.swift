import Metal
import Testing
import simd
@testable import Frostfield

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct IndirectDispatchTests {
    @Test func bufferGraphUsesTheCountWrittenOnTheGPU() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let graph = try BufferShaderGraph(fieldCount: 1) { builder in
            [0: builder.vector(builder.elementIndex, builder.elementCount, builder.constant(Float(0)), builder.constant(Float(1)))]
        }
        let kernel = try ShaderBufferKernel(graph: graph, device: device)
        let width = kernel.indirectPipelineState.threadExecutionWidth
        let arguments = try ShaderBufferKernel.IndirectArguments(elementCount: 7, threadgroupWidth: width)
        #expect(arguments.threadgroupsX == UInt32((7 + width - 1) / width))
        #expect(arguments.threadgroupsY == 1 && arguments.threadgroupsZ == 1)
        #expect(arguments.elementCount == 7)
        #expect(MemoryLayout<ShaderBufferKernel.IndirectArguments>.stride == 16)

        let count = 16
        let length = count * MemoryLayout<SIMD4<Float>>.stride
        let source = try #require(device.makeBuffer(length: length, options: .storageModeShared))
        let destination = try #require(device.makeBuffer(length: length, options: .storageModeShared))
        source.contents().initializeMemory(as: SIMD4<Float>.self, repeating: .zero, count: count)
        destination.contents().initializeMemory(as: SIMD4<Float>.self, repeating: SIMD4(repeating: .nan), count: count)
        let argumentBuffer = try #require(device.makeBuffer(length: 16, options: .storageModePrivate))

        let library = try device.makeLibrary(source: """
        #include <metal_stdlib>
        using namespace metal;
        kernel void frostfield_write_indirect(
            device uint4* arguments [[buffer(0)]],
            constant uint2& countAndWidth [[buffer(1)]]
        ) {
            uint elementCount = countAndWidth.x;
            uint groupWidth = max(countAndWidth.y, 1u);
            uint groups = elementCount == 0u ? 0u : (elementCount + groupWidth - 1u) / groupWidth;
            arguments[0] = uint4(groups, 1u, 1u, elementCount);
        }
        """, options: nil)
        let writer = try device.makeComputePipelineState(function: try #require(library.makeFunction(name: "frostfield_write_indirect")))
        let command = try #require(queue.makeCommandBuffer())
        let prepare = try #require(command.makeComputeCommandEncoder())
        prepare.setComputePipelineState(writer)
        prepare.setBuffer(argumentBuffer, offset: 0, index: 0)
        var countAndWidth = SIMD2<UInt32>(7, UInt32(width))
        prepare.setBytes(&countAndWidth, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
        prepare.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        prepare.endEncoding()
        try kernel.encode(indirect: argumentBuffer, from: source, to: destination, commandBuffer: command)
        command.commit()
        command.waitUntilCompleted()
        #expect(command.status == .completed)

        let output = Array(UnsafeBufferPointer(
            start: destination.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: count
        ))
        for element in 0..<7 {
            #expect(output[element] == SIMD4(Float(element), 7, 0, 1))
        }
        #expect(output[7].x.isNaN)

        let misaligned = try #require(queue.makeCommandBuffer())
        #expect(throws: ShaderComputeFailure.indirectArguments) {
            try kernel.encode(indirect: argumentBuffer, argumentsOffset: 4, from: source, to: destination, commandBuffer: misaligned)
        }
    }
}
