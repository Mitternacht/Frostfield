import Frostfield
import Metal
import simd
import Testing

/// Evaluates buffer graphs on the GPU through `ShaderBufferKernel`, the production
/// compute hook, so language tests compare generated Metal against CPU references.
struct GraphProbe {
    let device: any MTLDevice
    let queue: any MTLCommandQueue

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
    }

    /// Runs the graph once over `elements`, each holding `graph.fieldCount` float4
    /// fields, and returns every element's fields afterward.
    func run(
        _ graph: BufferShaderGraph, elements: [[SIMD4<Float>]], uniforms: ShaderUniforms? = nil,
        textures: [String: any MTLTexture] = [:]
    ) throws -> [[SIMD4<Float>]] {
        let fieldCount = graph.fieldCount
        var flat: [SIMD4<Float>] = []
        for element in elements {
            try #require(element.count == fieldCount)
            flat += element
        }
        let source = try #require(flat.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        })
        let destination = try #require(device.makeBuffer(length: source.length, options: .storageModeShared))
        let kernel = try ShaderBufferKernel(graph: graph, device: device)
        let command = try #require(queue.makeCommandBuffer())
        try kernel.encode(
            elementCount: elements.count, from: source, to: destination, uniforms: uniforms, textures: textures,
            commandBuffer: command
        )
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed, "\(String(describing: command.error))")
        let values = Array(UnsafeBufferPointer(
            start: destination.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: flat.count
        ))
        return stride(from: 0, to: values.count, by: fieldCount).map { Array(values[$0..<($0 + fieldCount)]) }
    }

    /// Runs a graph over one input field per element and returns output field 1.
    func map(
        _ inputs: [SIMD4<Float>], uniforms: ShaderUniforms? = nil, textures: [String: any MTLTexture] = [:],
        _ body: (ShaderGraphBuilder, ShaderValue<SIMD4<Float>>) throws -> ShaderValue<SIMD4<Float>>
    ) throws -> [SIMD4<Float>] {
        let graph = try BufferShaderGraph(fieldCount: 2) { b in [1: try body(b, b.elementField(0))] }
        return try run(graph, elements: inputs.map { [$0, .zero] }, uniforms: uniforms, textures: textures).map { $0[1] }
    }

    /// A shared RGBA32Float texture with the given texels in row-major order.
    func texture(width: Int, height: Int, texels: [SIMD4<Float>]) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        try texels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: try #require(bytes.baseAddress), bytesPerRow: width * 16
            )
        }
        return texture
    }

    static func maximumDifference(_ a: SIMD4<Float>, _ b: SIMD4<Float>) -> Float {
        let difference = simd_abs(a - b)
        return max(max(difference.x, difference.y), max(difference.z, difference.w))
    }
}

/// A seeded generator so random sample points repeat across runs.
struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
