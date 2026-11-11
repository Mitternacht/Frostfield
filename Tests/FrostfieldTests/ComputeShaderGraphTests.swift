import Foundation
import Frostfield
import Metal
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ComputeShaderGraphTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
    }

    @Test("Texture graphs emit a compute function and evaluate gridUV at exact texel centers")
    func gridGradient() throws {
        let graph = try TextureShaderGraph(
            nodes: [.input(.gridUV), .input(.texelSize), .combine([0, 1])], output: 2
        )
        let source = try graph.metalFunction(named: "fixture_grid")
        #expect(source.contains("static inline float4 fixture_grid("))
        #expect(source.contains("thread const FrostfieldComputeInput& input"))
        #expect(source.contains("thread const FrostfieldGraphResources& resources"))
        #expect(source.contains("input.gridUV"))

        let generator = try ShaderTextureGenerator(graph: graph, device: device)
        let texture = try generator.makeTexture(width: 16, height: 8, format: .rgba32Float, queue: queue)
        #expect(texture.pixelFormat == .rgba32Float && texture.mipmapLevelCount == 1)
        #expect(texture.usage.contains(.shaderRead))
        let texels = floats(texture)
        for y in 0..<8 {
            for x in 0..<16 {
                let index = (y * 16 + x) * 4
                #expect(Array(texels[index..<(index + 4)]) == [(Float(x) + 0.5) / 16, (Float(y) + 0.5) / 8, 1.0 / 16, 1.0 / 8])
            }
        }

        // Linear 8-bit storage rounds half away from zero, like ProceduralSurface.
        let unorm = try generator.makeTexture(width: 16, height: 8, format: .rgba8Unorm, queue: queue)
        let bytes = self.bytes(unorm)
        for x in 0..<16 {
            #expect(bytes[x * 4] == UInt8(((Float(x) + 0.5) / 16 * 255).rounded()))
            #expect(bytes[x * 4 + 2] == 16 && bytes[x * 4 + 3] == 32)
        }
        let mask = try generator.makeTexture(width: 16, height: 8, format: .r8Unorm, queue: queue)
        #expect(self.bytes(mask, bytesPerPixel: 1)[3] == UInt8((3.5 / 16 * 255 as Float).rounded()))
    }

    @Test("Vector graphs transmit or return zero at total internal reflection")
    func refractedDirections() throws {
        let graph = try TextureShaderGraph { t in
            let incident = t.constant(SIMD2<Float>(0.8, -0.6))
            let normal = t.constant(SIMD2<Float>(0, 1))
            let transmitted = incident.refract(normal, eta: t.gridUV.x * 2)
            let facing = normal.faceForward(incident: incident, referenceNormal: normal)
            let cross = t.constant(SIMD2<Float>(2, 3)).cross(t.constant(SIMD2<Float>(5, 7)))
            return t.vector(transmitted.x, transmitted.y, cross, facing.y)
        }
        let generator = try ShaderTextureGenerator(graph: graph, device: device)
        let texture = try generator.makeTexture(width: 2, height: 1, format: .rgba32Float, queue: queue)
        let values = floats(texture)
        #expect(abs(values[0] - 0.4) < 1e-6)
        #expect(abs(values[1] + sqrt(Float(0.84))) < 1e-6)
        #expect(Array(values[2..<4]) == [-1, 1])
        #expect(Array(values[4..<8]) == [0, 0, -1, 1])
    }

    @Test("Four-component directions reflect and face forward at the zero-dot boundary")
    func fourComponentDirections() throws {
        let graph = try TextureShaderGraph { t in
            let normal = t.constant(SIMD4<Float>(0, 2, 0, 0))
            let incident = t.constant(SIMD4<Float>(1, -1, 0, 0))
            let reflected = incident.reflect(normal)
            let perpendicular = t.constant(SIMD4<Float>(1, 0, 0, 0))
            let facing = normal.faceForward(incident: perpendicular, referenceNormal: normal)
            let unitIncident = t.constant(SIMD4<Float>(0.8, -0.6, 0, 0))
            let transmitted = unitIncident.refract(normal.normalized(), eta: 1)
            return t.vector(reflected.x, reflected.y, facing.y, transmitted.y)
        }
        let generator = try ShaderTextureGenerator(graph: graph, device: device)
        let texture = try generator.makeTexture(width: 1, height: 1, format: .rgba32Float, queue: queue)
        let values = floats(texture)
        #expect(Array(values.prefix(3)) == [1, 1, -2])
        #expect(abs(values[3] + 0.6) < 1e-6)
    }

    @Test("sRGB output stores the sRGB transfer of linear graph values, never linear values")
    func srgbEncoding() throws {
        let graph = try TextureShaderGraph { t in
            t.vector(t.gridUV.x.xxx, t.gridUV.y)
        }
        let generator = try ShaderTextureGenerator(graph: graph, device: device)
        let encoded = try generator.makeTexture(width: 64, height: 2, format: .rgba8Unorm_srgb, queue: queue)
        let linear = try generator.makeTexture(width: 64, height: 2, format: .rgba8Unorm, queue: queue)
        #expect(encoded.pixelFormat == .rgba8Unorm_srgb)
        let stored = bytes(encoded)
        let linearBytes = bytes(linear)
        for x in 0..<64 {
            let value = (Float(x) + 0.5) / 64
            let expected = Int(srgbByte(value))
            for channel in 0..<3 {
                #expect(abs(Int(stored[x * 4 + channel]) - expected) <= 1)
            }
            // Alpha is linear in sRGB formats.
            #expect(stored[x * 4 + 3] == UInt8((0.25 * 255 as Float).rounded()))
        }
        // A mid-range linear 0.2 encodes near 124, far from its linear byte 51.
        let fifth = 12
        #expect(abs(Int(stored[fifth * 4]) - Int(srgbByte((Float(fifth) + 0.5) / 64))) <= 1)
        #expect(Int(stored[fifth * 4]) - Int(linearBytes[fifth * 4]) > 60)
    }

    @Test("Mipmapped output carries a full GPU-generated chain whose levels average the level above")
    func mipChain() throws {
        let checker = try TextureShaderGraph { t in
            let cell = (t.gridUV / t.texelSize.x).floor()
            return (cell.x + cell.y).mod(2)
        }
        let generator = try ShaderTextureGenerator(graph: checker, device: device)
        let texture = try generator.makeTexture(
            width: 16, height: 16, format: .rgba16Float, mipmapped: true, queue: queue
        )
        #expect(texture.mipmapLevelCount == 5)
        let base = try readRGBA16FloatMip(texture, level: 0, queue: queue)
        for y in 0..<16 {
            for x in 0..<16 {
                #expect(base[(y * 16 + x) * 4] == Float16((x + y) % 2))
                #expect(base[(y * 16 + x) * 4 + 3] == 1)
            }
        }
        for level in 1..<5 {
            let values = try readRGBA16FloatMip(texture, level: level, queue: queue)
            #expect(values.enumerated().allSatisfy { $0.offset % 4 == 3 ? $0.element == 1 : $0.element == 0.5 })
        }

        // A smooth pattern: level 1 is the 2×2 box average of level 0.
        let ramp = try ShaderTextureGenerator(graph: TextureShaderGraph { t in
            t.vector(t.gridUV.x, t.gridUV.y * t.gridUV.y, (t.gridUV.x * 3).sin().abs())
        }, device: device)
        let smooth = try ramp.makeTexture(width: 32, height: 16, format: .rgba16Float, mipmapped: true, queue: queue)
        #expect(smooth.mipmapLevelCount == 6)
        let fine = try readRGBA16FloatMip(smooth, level: 0, queue: queue).map(Float.init)
        let coarse = try readRGBA16FloatMip(smooth, level: 1, queue: queue).map(Float.init)
        for y in 0..<8 {
            for x in 0..<16 {
                for channel in 0..<3 {
                    let average = [(0, 0), (1, 0), (0, 1), (1, 1)].reduce(Float(0)) { sum, offset in
                        sum + fine[((2 * y + offset.1) * 32 + 2 * x + offset.0) * 4 + channel]
                    } / 4
                    #expect(abs(coarse[(y * 16 + x) * 4 + channel] - average) < 2e-3)
                }
            }
        }
    }

    @Test("Uniform edits change the output while the cached pipeline object is reused")
    func uniformEditsReusePipeline() throws {
        func tinted(_ initial: SIMD3<Float>) throws -> TextureShaderGraph {
            try TextureShaderGraph { t in
                t.uniform("Tint", initial) * t.uniform("Gain", Float(1))
            }
        }
        let graph = try tinted(SIMD3(0.25, 0.5, 0.75))
        let generator = try ShaderTextureGenerator(graph: graph, device: device)
        let pipeline = generator.pipelineState
        let uniforms = try ShaderUniforms(graph: graph)

        let first = floats(try generator.makeTexture(
            width: 2, height: 2, format: .rgba32Float, uniforms: uniforms, queue: queue
        ))
        #expect(Array(first.prefix(4)) == [0.25, 0.5, 0.75, 1])
        try uniforms.set("Tint", to: SIMD3<Float>(1, 0, 0.5))
        try uniforms.set("Gain", to: Float(0.5))
        let second = floats(try generator.makeTexture(
            width: 2, height: 2, format: .rgba32Float, uniforms: uniforms, queue: queue
        ))
        #expect(Array(second.suffix(4)) == [0.5, 0, 0.25, 1])
        #expect(generator.pipelineState === pipeline)

        // Uniform defaults are not part of the generated source, so a rebuilt graph shares the pipeline.
        let rebuilt = try tinted(SIMD3(0.9, 0.1, 0.1))
        #expect(rebuilt != graph)
        let shared = try ShaderTextureGenerator(graph: rebuilt, device: device)
        #expect(shared.pipelineState === pipeline)
        let defaults = floats(try shared.makeTexture(width: 1, height: 1, format: .rgba32Float, queue: queue))
        #expect(defaults == [0.9, 0.1, 0.1, 1])
    }

    @Test("Declared textures are sampled at their base level, and color inputs round-trip through sRGB")
    func textureInputs() throws {
        let pattern: [UInt8] = (0..<16).flatMap { index -> [UInt8] in
            let value = UInt8(index * 16 + 7)
            return [value, 255 - value, UInt8(index * 3), 255]
        }
        let data = try makeTexture(.rgba8Unorm, width: 4, height: 4, bytes: pattern, mipmapped: true)
        let color = try makeTexture(.rgba8Unorm_srgb, width: 4, height: 4, bytes: pattern)
        // Fill data level 1 with distinct values so sampling any level other than 0 is visible.
        data.replace(
            region: MTLRegionMake2D(0, 0, 2, 2), mipmapLevel: 1,
            withBytes: [UInt8](repeating: 1, count: 16), bytesPerRow: 8
        )

        let copy = try TextureShaderGraph { t in
            t.texture("Data", content: .data, addressing: .clampToEdge, filtering: .nearest).sample(t.gridUV)
        }
        let copied = try ShaderTextureGenerator(graph: copy, device: device).makeTexture(
            width: 4, height: 4, format: .rgba8Unorm, textures: ["Data": data], queue: queue
        )
        #expect(bytes(copied) == pattern)

        let roundTrip = try TextureShaderGraph { t in
            t.texture("Albedo", content: .color, filtering: .nearest).sample(t.gridUV)
        }
        let encoded = try ShaderTextureGenerator(graph: roundTrip, device: device).makeTexture(
            width: 4, height: 4, format: .rgba8Unorm_srgb, textures: ["Albedo": color], queue: queue
        )
        #expect(zip(bytes(encoded), pattern).allSatisfy { abs(Int($0) - Int($1)) <= 1 })

        // Repeat addressing wraps a shifted lookup back to the first column.
        let shifted = try TextureShaderGraph { t in
            let offset = t.vector(t.texelSize.x * 4, t.constant(Float(0)))
            return t.texture("Data", content: .data, filtering: .nearest).sample(t.gridUV + offset)
        }
        let wrapped = try ShaderTextureGenerator(graph: shifted, device: device).makeTexture(
            width: 4, height: 4, format: .rgba8Unorm, textures: ["Data": data], queue: queue
        )
        #expect(bytes(wrapped) == pattern)
    }

    @Test("Encoding into a caller texture leaves ordering to the caller's command buffer")
    func encodeIntoCallerTexture() throws {
        let graph = try TextureShaderGraph { t in t.constant(SIMD4<Float>(0.5, 0.25, 1, 0.75)) }
        let generator = try ShaderTextureGenerator(graph: graph, device: device)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: 8, height: 4, mipmapped: true
        )
        descriptor.storageMode = .private
        descriptor.usage = [.shaderRead, .shaderWrite]
        let target = try #require(device.makeTexture(descriptor: descriptor))
        let readback = try #require(device.makeBuffer(length: 8 * 4 * 8, options: .storageModeShared))
        let commandBuffer = try #require(queue.makeCommandBuffer())
        try generator.encode(into: target, commandBuffer: commandBuffer)
        let blit = try #require(commandBuffer.makeBlitCommandEncoder())
        blit.copy(
            from: target, sourceSlice: 0, sourceLevel: 2, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: 2, height: 1, depth: 1), to: readback, destinationOffset: 0,
            destinationBytesPerRow: 16, destinationBytesPerImage: 16
        )
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        let values = readback.contents().bindMemory(to: Float16.self, capacity: 8)
        #expect((0..<8).map { Float(values[$0]) } == [0.5, 0.25, 1, 0.75, 0.5, 0.25, 1, 0.75])
    }

    @Test("Compute graphs reject unavailable inputs and unsupported outputs")
    func graphValidation() throws {
        #expect(throws: MaterialGraph.Failure.unavailable(node: 0, stage: .compute)) {
            try TextureShaderGraph(nodes: [.input(.viewDirection)], output: 0)
        }
        #expect(throws: MaterialGraph.Failure.unavailable(node: 0, stage: .compute)) {
            try TextureShaderGraph(nodes: [.input(.elementIndex)], output: 0)
        }
        #expect(throws: MaterialGraph.Failure.unavailable(node: 0, stage: .compute)) {
            try TextureShaderGraph(nodes: [.uv(.u)], output: 0)
        }
        #expect(throws: MaterialGraph.Failure.unavailable(node: 1, stage: .compute)) {
            try TextureShaderGraph(nodes: [.input(.gridUV), .sampleScene(coordinates: 0)], output: 1)
        }
        #expect(throws: TextureShaderGraph.Failure.outputType(node: 0, actual: .vector2)) {
            try TextureShaderGraph(nodes: [.input(.gridUV)], output: 0)
        }
        #expect(throws: TextureShaderGraph.Failure.invalidFunctionName("kernel")) {
            try TextureShaderGraph(nodes: [.scalar(1)], output: 0).metalFunction(named: "kernel")
        }
        #expect(throws: MaterialGraph.Failure.invalidDeclaration("A graph output belongs to a different builder.")) {
            var foreign: ShaderValue<Float>?
            _ = try TextureShaderGraph { t in
                let value = t.constant(Float(1))
                foreign = value
                return value
            }
            return try TextureShaderGraph { _ in try #require(foreign) }
        }
    }

    @Test("Texture generation validates dimensions, formats, bindings, uniforms and command buffers")
    func generatorValidation() throws {
        let graph = try TextureShaderGraph { t in
            t.texture("Mask", content: .data).sample(t.gridUV) * t.uniform("Gain", Float(1))
        }
        let generator = try ShaderTextureGenerator(graph: graph, device: device)
        let mask = try makeTexture(.rgba8Unorm, width: 2, height: 2, bytes: [UInt8](repeating: 9, count: 16))
        let srgb = try makeTexture(.rgba8Unorm_srgb, width: 2, height: 2, bytes: [UInt8](repeating: 9, count: 16))
        func make(
            _ width: Int = 4, _ height: Int = 4, format: MTLPixelFormat = .rgba8Unorm,
            uniforms: ShaderUniforms? = nil, textures: [String: any MTLTexture]? = nil
        ) throws {
            _ = try generator.makeTexture(
                width: width, height: height, format: format, uniforms: uniforms,
                textures: textures ?? ["Mask": mask], queue: queue
            )
        }

        try make()
        #expect(throws: ShaderComputeFailure.dimensions(width: 0, height: 4)) { try make(0, 4) }
        #expect(throws: ShaderComputeFailure.dimensions(width: 8_193, height: 1)) { try make(8_193, 1) }
        #expect(throws: ShaderComputeFailure.dimensions(width: 8_192, height: 4_096)) { try make(8_192, 4_096) }
        #expect(throws: ShaderComputeFailure.unsupportedFormat(.bgra8Unorm)) { try make(format: .bgra8Unorm) }
        #expect(throws: ShaderComputeFailure.unsupportedFormat(.depth32Float)) { try make(format: .depth32Float) }
        #expect(throws: ShaderComputeFailure.missingTexture("Mask")) { try make(textures: [:]) }
        #expect(throws: ShaderComputeFailure.unknownTexture("Other")) {
            try make(textures: ["Mask": mask, "Other": mask])
        }
        #expect(throws: ShaderComputeFailure.invalidTexture("Mask")) { try make(textures: ["Mask": srgb]) }
        #expect(throws: ShaderComputeFailure.uniforms) {
            try make(uniforms: ShaderUniforms([ShaderUniform("Gain", SIMD2<Float>(1, 1))]))
        }

        let commandBuffer = try #require(queue.makeCommandBuffer())
        let readOnly = try makeTexture(.rgba8Unorm, width: 2, height: 2, bytes: [UInt8](repeating: 0, count: 16))
        #expect(throws: ShaderComputeFailure.outputTexture) {
            try generator.encode(into: readOnly, commandBuffer: commandBuffer, textures: ["Mask": mask])
        }
        let writable = try makeTexture(
            .rgba8Unorm, width: 2, height: 2, bytes: [UInt8](repeating: 0, count: 16), writable: true
        )
        #expect(throws: ShaderComputeFailure.textureAliasesOutput("Mask")) {
            try generator.encode(into: writable, commandBuffer: commandBuffer, textures: ["Mask": writable])
        }
        let view = try #require(writable.makeTextureView(pixelFormat: .rgba8Unorm))
        #expect(throws: ShaderComputeFailure.textureAliasesOutput("Mask")) {
            try generator.encode(into: writable, commandBuffer: commandBuffer, textures: ["Mask": view])
        }
        commandBuffer.commit()
        #expect(throws: ShaderComputeFailure.commandBuffer) {
            try generator.encode(into: writable, commandBuffer: commandBuffer, textures: ["Mask": mask])
        }
        commandBuffer.waitUntilCompleted()
    }

    @Test("Compiler errors from declared functions report the generated line")
    func compilationDiagnostics() throws {
        let graph = try TextureShaderGraph { t in
            let broken = t.function(
                "broken", returns: Float.self, parameters: [.init("x", .scalar)], body: "return x + undefined_value;"
            )
            return broken(t.gridUV.x)
        }
        do {
            _ = try ShaderTextureGenerator(graph: graph, device: device)
            Issue.record("A broken function compiled.")
        } catch let ShaderComputeFailure.compilation(detail) {
            #expect(detail.contains("undefined_value"))
            #expect(detail.contains("Line "))
        }
    }

    // MARK: - Helpers

    private func makeTexture(
        _ format: MTLPixelFormat, width: Int, height: Int, bytes: [UInt8],
        mipmapped: Bool = false, writable: Bool = false
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: mipmapped
        )
        descriptor.storageMode = .shared
        descriptor.usage = writable ? [.shaderRead, .shaderWrite] : .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        texture.replace(
            region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: bytes, bytesPerRow: width * 4
        )
        return texture
    }

    private func bytes(_ texture: any MTLTexture, level: Int = 0, bytesPerPixel: Int = 4) -> [UInt8] {
        read(texture, level: level, as: UInt8.self, components: bytesPerPixel)
    }

    private func floats(_ texture: any MTLTexture, level: Int = 0) -> [Float] {
        read(texture, level: level, as: Float.self, components: 4)
    }

    private func read<T: Numeric>(_ texture: any MTLTexture, level: Int, as type: T.Type, components: Int) -> [T] {
        let width = max(1, texture.width >> level)
        let height = max(1, texture.height >> level)
        var values = [T](repeating: 0, count: width * height * components)
        values.withUnsafeMutableBytes { raw in
            if let address = raw.baseAddress {
                texture.getBytes(
                    address, bytesPerRow: width * components * MemoryLayout<T>.stride,
                    from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: level
                )
            }
        }
        return values
    }

    private func srgbByte(_ linear: Float) -> UInt8 {
        let value = min(max(linear, 0), 1)
        let encoded = value <= 0.003_130_8 ? 12.92 * value : 1.055 * pow(value, 1 / 2.4) - 0.055
        return UInt8((encoded * 255).rounded())
    }
}

/// Explicit GPU mip readback. The simulator's direct shared-texture read can return base-level texels.
func readRGBA16FloatMip(_ texture: any MTLTexture, level: Int, queue: any MTLCommandQueue) throws -> [Float16] {
    let width = max(1, texture.width >> level)
    let height = max(1, texture.height >> level)
    let rowBytes = width * 8
    let imageBytes = rowBytes * height
    let readback = try #require(queue.device.makeBuffer(length: imageBytes, options: .storageModeShared))
    let command = try #require(queue.makeCommandBuffer())
    let blit = try #require(command.makeBlitCommandEncoder())
    blit.copy(
        from: texture, sourceSlice: 0, sourceLevel: level, sourceOrigin: MTLOrigin(),
        sourceSize: MTLSize(width: width, height: height, depth: 1), to: readback, destinationOffset: 0,
        destinationBytesPerRow: rowBytes, destinationBytesPerImage: imageBytes
    )
    blit.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    try #require(command.status == .completed)
    let count = width * height * 4
    let values = readback.contents().bindMemory(to: Float16.self, capacity: count)
    return Array(UnsafeBufferPointer(start: values, count: count))
}
