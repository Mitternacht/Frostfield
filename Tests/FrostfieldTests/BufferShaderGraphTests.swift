import Foundation
import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct BufferShaderGraphTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
    }

    /// Position/age, velocity/lifetime, and a diagnostic field per element.
    private func advect() throws -> BufferShaderGraph {
        try BufferShaderGraph(fieldCount: 3) { b in
            let position = b.elementField(0)
            let velocity = b.elementField(1)
            let index = b.elementIndex
            return [
                0: position + velocity * b.uniform("DeltaTime", Float(0.5)),
                2: b.vector(index, b.elementCount, index.mod(8), b.elementField(2).w + 1),
            ]
        }
    }

    @Test("A graph-authored particle step over 1000 elements matches the CPU, ping-ponging buffers")
    func particleStep() throws {
        let graph = try advect()
        #expect(graph.elementStride == 3 * MemoryLayout<SIMD4<Float>>.stride)
        let source = try graph.metalFunction(named: "advect")
        #expect(source.contains("static inline void advect("))
        #expect(source.contains("thread float4 (&fields)[4]"))
        #expect(!source.contains("fields[1] ="))

        let kernel = try ShaderBufferKernel(graph: graph, device: device)
        let pipeline = kernel.pipelineState
        let count = 1_000
        // Quarter-unit values keep every product and sum exact in Float, so GPU and CPU agree bit for bit.
        var state: [SIMD4<Float>] = []
        for element in 0..<count {
            let i = Float(element)
            state.append(SIMD4(i.truncatingRemainder(dividingBy: 97) * 0.25, -i * 0.125, 3, 0))
            state.append(SIMD4(0.5, Float(element % 13) - 6, 0.25 * Float(element % 5), 2))
            state.append(SIMD4(-1, -1, -1, Float(element % 3)))
        }
        let a = try makeBuffer(state)
        let b = try makeBuffer([SIMD4<Float>](repeating: SIMD4(repeating: .nan), count: state.count))
        let uniforms = try ShaderUniforms(graph: graph)

        func step(_ values: [SIMD4<Float>], deltaTime: Float) -> [SIMD4<Float>] {
            var next = values
            for element in 0..<count {
                let base = element * 3
                next[base] = values[base] + values[base + 1] * deltaTime
                let index = Float(element)
                next[base + 2] = SIMD4(index, Float(count), Float(element % 8), values[base + 2].w + 1)
            }
            return next
        }

        let commandBuffer = try #require(queue.makeCommandBuffer())
        try kernel.encode(elementCount: count, from: a, to: b, uniforms: uniforms, commandBuffer: commandBuffer)
        try kernel.encode(elementCount: count, from: b, to: a, uniforms: uniforms, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        let twice = step(step(state, deltaTime: 0.5), deltaTime: 0.5)
        #expect(contents(a, count: state.count) == twice)
        #expect(contents(b, count: state.count) == step(state, deltaTime: 0.5))

        try uniforms.set("DeltaTime", to: Float(-0.25))
        let edited = try #require(queue.makeCommandBuffer())
        try kernel.encode(elementCount: count, from: a, to: b, uniforms: uniforms, commandBuffer: edited)
        edited.commit()
        edited.waitUntilCompleted()
        #expect(contents(b, count: state.count) == step(twice, deltaTime: -0.25))
        #expect(kernel.pipelineState === pipeline)
        #expect(try ShaderBufferKernel(graph: advect(), device: device).pipelineState === pipeline)
    }

    @Test("Offsets select a sub-range, nonfinite results keep the previous field, and textures are sampled")
    func offsetsNonfiniteAndTextures() throws {
        let graph = try BufferShaderGraph(fieldCount: 2) { b in
            let lookup = b.texture("Table", content: .data, addressing: .clampToEdge, filtering: .nearest)
            let u = (b.elementIndex + 0.5) / b.elementCount
            return [
                0: lookup.sample(b.vector(u, b.constant(Float(0.5)))),
                1: b.elementField(1) * b.uniform("Scale", Float(1e30)) * Float(1e30),
            ]
        }
        let kernel = try ShaderBufferKernel(graph: graph, device: device)
        let table: [SIMD4<Float>] = [SIMD4(1, 2, 3, 4), SIMD4(5, 6, 7, 8), SIMD4(9, 10, 11, 12), SIMD4(-1, -2, -3, -4)]
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 4, height: 1, mipmapped: false)
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        texture.replace(region: MTLRegionMake2D(0, 0, 4, 1), mipmapLevel: 0, withBytes: table, bytesPerRow: 64)

        // Element records start one element (32 bytes) into each buffer.
        let input = try makeBuffer([SIMD4<Float>](repeating: .zero, count: 2) + (0..<4).flatMap { element in
            [SIMD4<Float>(repeating: -9), SIMD4(Float(element), 0, 0, 1)]
        })
        let output = try makeBuffer([SIMD4<Float>](repeating: SIMD4(repeating: 7), count: 10))
        let commandBuffer = try #require(queue.makeCommandBuffer())
        try kernel.encode(
            elementCount: 4, from: input, sourceOffset: 32, to: output, destinationOffset: 32,
            textures: ["Table": texture], commandBuffer: commandBuffer
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let result = contents(output, count: 10)
        let untouched = SIMD4<Float>(repeating: 7)
        #expect(result[0] == untouched && result[1] == untouched)
        for element in 0..<4 {
            #expect(result[2 + element * 2] == table[element])
            // x is 0 for element 0, so x stays 0 there; every infinite component keeps its previous value.
            let kept = SIMD4<Float>(Float(element), 0, 0, 1)
            #expect(result[3 + element * 2] == kept)
        }
    }

    @Test("Buffer graphs reject bad layouts, field reads, and stage inputs")
    func graphValidation() throws {
        let field = MaterialGraph.Node.input(.elementField(0))
        #expect(throws: BufferShaderGraph.Failure.fieldCount(0)) {
            try BufferShaderGraph(nodes: [field], fieldCount: 0, outputs: [0: 0])
        }
        #expect(throws: BufferShaderGraph.Failure.fieldCount(5)) {
            try BufferShaderGraph(nodes: [field], fieldCount: 5, outputs: [0: 0])
        }
        #expect(throws: BufferShaderGraph.Failure.noOutputs) {
            try BufferShaderGraph(nodes: [field], fieldCount: 1, outputs: [:])
        }
        #expect(throws: BufferShaderGraph.Failure.outputField(2, fieldCount: 2)) {
            try BufferShaderGraph(nodes: [field], fieldCount: 2, outputs: [2: 0])
        }
        #expect(throws: BufferShaderGraph.Failure.outputType(field: 0, node: 1, actual: .scalar)) {
            try BufferShaderGraph(nodes: [field, .input(.elementIndex)], fieldCount: 1, outputs: [0: 1])
        }
        #expect(throws: BufferShaderGraph.Failure.fieldOutOfRange(node: 1, field: 3, fieldCount: 2)) {
            try BufferShaderGraph(nodes: [field, .input(.elementField(3)), .add(0, 1)], fieldCount: 2, outputs: [1: 2])
        }
        #expect(throws: MaterialGraph.Failure.unavailable(node: 0, stage: .compute)) {
            try BufferShaderGraph(nodes: [.input(.elementField(4))], fieldCount: 4, outputs: [0: 0])
        }
        #expect(throws: MaterialGraph.Failure.unavailable(node: 0, stage: .compute)) {
            try BufferShaderGraph(nodes: [.input(.gridUV), .swizzle(0, "xyxy")], fieldCount: 1, outputs: [0: 1])
        }
        #expect(throws: MaterialGraph.Failure.unavailable(node: 0, stage: .compute)) {
            try BufferShaderGraph(nodes: [.input(.viewDirection), .swizzle(0, "xyzz")], fieldCount: 1, outputs: [0: 1])
        }
    }

    @Test("Dispatch validation rejects aliasing, small buffers, misaligned offsets and missing bindings")
    func dispatchValidation() throws {
        let kernel = try ShaderBufferKernel(graph: advect(), device: device)
        let stride = kernel.graph.elementStride
        let a = try #require(device.makeBuffer(length: stride * 10, options: .storageModeShared))
        let b = try #require(device.makeBuffer(length: stride * 10, options: .storageModePrivate))
        let commandBuffer = try #require(queue.makeCommandBuffer())
        func encode(
            _ count: Int = 10, from source: any MTLBuffer, _ sourceOffset: Int = 0,
            to destination: any MTLBuffer, _ destinationOffset: Int = 0, uniforms: ShaderUniforms? = nil,
            textures: [String: any MTLTexture] = [:]
        ) throws {
            try kernel.encode(
                elementCount: count, from: source, sourceOffset: sourceOffset, to: destination,
                destinationOffset: destinationOffset, uniforms: uniforms, textures: textures,
                commandBuffer: commandBuffer
            )
        }
        #expect(throws: ShaderComputeFailure.bufferAliasing) { try encode(from: a, to: a) }
        #expect(throws: ShaderComputeFailure.elementCount(0)) { try encode(0, from: a, to: b) }
        #expect(throws: ShaderComputeFailure.elementCount(16_777_217)) { try encode(16_777_217, from: a, to: b) }
        #expect(throws: ShaderComputeFailure.bufferTooSmall("source", required: stride * 11, available: stride * 10)) {
            try encode(11, from: a, to: b)
        }
        #expect(throws: ShaderComputeFailure.bufferTooSmall("destination", required: stride * 10 + 16, available: stride * 10)) {
            try encode(from: a, to: b, 16)
        }
        #expect(throws: ShaderComputeFailure.bufferOffset(8)) { try encode(1, from: a, 8, to: b) }
        #expect(throws: ShaderComputeFailure.bufferOffset(-16)) { try encode(1, from: a, to: b, -16) }
        #expect(throws: ShaderComputeFailure.uniforms) {
            try encode(from: a, to: b, uniforms: ShaderUniforms([]))
        }
        #expect(throws: ShaderComputeFailure.unknownTexture("Extra")) {
            let texture = try #require(device.makeTexture(descriptor: .texture2DDescriptor(
                pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false
            )))
            try encode(from: a, to: b, textures: ["Extra": texture])
        }
        try encode(from: a, to: b)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        #expect(throws: ShaderComputeFailure.commandBuffer) { try encode(from: a, to: b) }

        let unretained = try #require(queue.makeCommandBufferWithUnretainedReferences())
        #expect(throws: ShaderComputeFailure.commandBuffer) {
            try kernel.encode(elementCount: 1, from: a, to: b, commandBuffer: unretained)
        }
    }

    // MARK: - Helpers

    private func makeBuffer(_ values: [SIMD4<Float>]) throws -> any MTLBuffer {
        try #require(values.withUnsafeBytes { bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        })
    }

    private func contents(_ buffer: any MTLBuffer, count: Int) -> [SIMD4<Float>] {
        Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: count), count: count))
    }
}
