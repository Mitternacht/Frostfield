import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PixelatedEdgesTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Pixel blocks match depth and normal references in forward and reversed depth")
    func edgeReference() throws {
        let width = 17, height = 11, block = 3
        let gridWidth = (width + block - 1) / block, gridHeight = (height + block - 1) / block
        let source = try gpu.color(width: width, height: height) { x, y in
            let alpha: Float = [0.125, 0.25, 0.5, 0.75, 1][(x + y) % 5]
            let rgb = SIMD3(Float(x + 1) / 8, Float(y + 1) / 16, -0.25)
            return SIMD4(rgb * alpha, alpha)
        }
        let original = try gpu.read(source)
        let destination = try gpu.color(width: width, height: height)
        let depthByColumn: [Float] = [0.2, 0.2175, 0.2175, 0.5, 0.5, 0.5]
        func depthAtCell(_ x: Int, _ y: Int) -> Float {
            y < 2 ? depthByColumn[x] : x == 0 ? 1 : 0.45
        }
        func normalAtCell(_ x: Int, _ y: Int) -> SIMD3<Float> {
            if depthAtCell(x, y) == 1 { return .zero }
            return x < (y < 2 ? 3 : 4) ? SIMD3(0, 0, 1) : SIMD3(-1, 0, 0)
        }
        func isCenter(_ x: Int, _ y: Int) -> Bool {
            x == min(x / block * block + block / 2, width - 1)
                && y == min(y / block * block + block / 2, height - 1)
        }

        let attributes = try makeAttributes(width: width, height: height) { x, y in
            // RG32Uint octahedral snorm16 of -X is (0x8001, 0) and +Z is zero.
            let isMinusX = normalAtCell(x / block, y / block).x < 0
            return isCenter(x, y) && isMinusX ? 0x8001 : 0
        }
        let effect = try PixelatedEdgesPass(device: gpu.device, settings: .init(pixelSize: block), surfaceAttributes: attributes)
        let chain = try PostProcessChain(queue: gpu.queue, effects: [.pass(effect)])
        let pipelineCount = effect.builtPipelineCount
        for mode: MeshDepthMode in [.forward, .reversed] {
            effect.settings.depthMode = mode
            let depth = try gpu.depth(width: width, height: height) { x, y in
                let value = isCenter(x, y) ? depthAtCell(x / block, y / block) : 0.95
                return mode == .forward ? value : 1 - value
            }
            for rotateCamera in [false, true] {
                effect.cameraRotation = simd_quatf(angle: rotateCamera ? .pi : 0, axis: SIMD3(0, 1, 0))
                let bias: SIMD3<Float> = rotateCamera ? SIMD3(-1, 1, -1) : SIMD3(repeating: 1)
                try gpu.run { try chain.encode(source: source, depth: depth, destination: destination, commandBuffer: $0) }
                let result = try gpu.read(destination)
                var seenGains = Set<Float>()
                for (index, actual) in result.enumerated() {
                    let cellX = (index % width) / block, cellY = (index / width) / block
                    let representativeX = min(cellX * block + block / 2, width - 1)
                    let representativeY = min(cellY * block + block / 2, height - 1)
                    let color = original[representativeY * width + representativeX]
                    let centerDepth = depthAtCell(cellX, cellY)
                    let centerNormal = normalAtCell(cellX, cellY)
                    var depthDifference: Float = 0
                    var crease: Float = 0
                    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        let sx = min(max(cellX + dx, 0), gridWidth - 1)
                        let sy = min(max(cellY + dy, 0), gridHeight - 1)
                        let difference = depthAtCell(sx, sy) - centerDepth
                        depthDifference += min(max(difference, 0), 1)
                        if centerDepth < 1 && difference > -0.01 {
                            let adjacent = normalAtCell(sx, sy)
                            let priority = smoothstep(-0.01, 0.01, simd_dot(centerNormal - adjacent, bias))
                            crease += (1 - simd_dot(centerNormal, adjacent)) * priority
                        }
                    }
                    let depthEdge = floor(2 * smoothstep(0.01, 0.02, depthDifference)) / 2
                    let gain: Float = depthEdge > 0 ? 1 - 0.4 * depthEdge : crease >= 0.1 ? 1.3 : 1
                    seenGains.insert(gain)
                    let expected = SIMD4(SIMD3(color.x, color.y, color.z) * gain, color.w)
                    #expect(PostProcessFixture.close(actual, expected, tolerance: 0.002),
                            "\(mode), rotated \(rotateCamera), pixel \(index): \(actual) vs \(expected)")
                    #expect(actual.w == expected.w)
                }
                #expect(seenGains == [0.6, 0.8, 1, 1.3])
                // The equal-depth crease changes sides with the camera's world orientation.
                let creaseCellX = rotateCamera ? 4 : 3
                let representative = 10 * width + creaseCellX * block + block / 2
                #expect(result[representative].x > original[representative].x)
            }
        }
        #expect(effect.builtPipelineCount == pipelineCount)
    }

    @Test("Color-only matches the existing graph, and depth and normal resources are validated")
    func colorOnlyAndValidation() throws {
        let width = 9, height = 7
        let source = try gpu.color(width: width, height: height) { x, y in
            SIMD4(Float(x) / 4, Float(y) / 8, 0.25, 0.5)
        }
        let output = try gpu.color(width: width, height: height)
        let reference = try gpu.color(width: width, height: height)
        let depth = try gpu.depth(width: width, height: height) { x, _ in x < 3 ? 0.2 : 0.7 }
        let effect = try PixelatedEdgesPass(device: gpu.device, settings: .init(pixelSize: 3))
        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PostProcessEffectError.missingDepth) {
            try effect.encode(source: source, destination: output, commandBuffer: command)
        }
        #expect(throws: PostProcessEffectError.depth) {
            try effect.encode(source: source, depth: source, destination: output, commandBuffer: command)
        }
        let smallDepth = try gpu.depth(width: width - 1, height: height) { _, _ in 0.5 }
        #expect(throws: PostProcessEffectError.sizeMismatch) {
            try effect.encode(source: source, depth: smallDepth, destination: output, commandBuffer: command)
        }
        #expect(throws: PostProcessEffectError.self) {
            try effect.encode(source: source, depth: depth, destination: output, commandBuffer: command)
        }
        effect.surfaceAttributes = source
        #expect(throws: PostProcessEffectError.self) {
            try effect.encode(source: source, depth: depth, destination: output, commandBuffer: command)
        }
        effect.surfaceAttributes = try makeAttributes(width: width - 1, height: height) { _, _ in 0 }
        #expect(throws: PostProcessEffectError.sizeMismatch) {
            try effect.encode(source: source, depth: depth, destination: output, commandBuffer: command)
        }
        effect.surfaceAttributes = try makeAttributes(width: width, height: height) { _, _ in 0 }
        effect.cameraRotation = simd_quatf(vector: .zero)
        #expect(throws: PostProcessEffectError.self) {
            try effect.encode(source: source, depth: depth, destination: output, commandBuffer: command)
        }
        effect.settings.depthMode = .logarithmic
        #expect(throws: MeshRenderer.Failure.logarithmicDepthConsumer) {
            try effect.encode(source: source, depth: depth, destination: output, commandBuffer: command)
        }
        effect.settings.depthMode = .forward
        effect.settings.normalEdgeStrength = 0
        effect.surfaceAttributes = nil
        try gpu.run { try effect.encode(source: source, depth: depth, destination: output, commandBuffer: $0) }
        let depthOnlyPixel = try gpu.read(output)[width + 1]
        // Allow one half-float ULP for render-target conversion of the decimal gain.
        #expect(PostProcessFixture.close(depthOnlyPixel, SIMD4(0.15, 0.075, 0.15, 0.5), tolerance: Float(Float16(0.15).ulp)))
        effect.settings.depthEdgeStrength = 0.5
        try gpu.run { try effect.encode(source: source, depth: depth, destination: output, commandBuffer: $0) }
        #expect(try gpu.read(output)[width + 1] == SIMD4(0.125, 0.0625, 0.125, 0.5))

        effect.settings.depthEdgeStrength = 0
        #expect(!effect.readsDepth)
        let graph = try PostProcessPass(graph: .pixelate(size: 3), device: gpu.device)
        try gpu.run {
            try effect.encode(source: source, destination: output, commandBuffer: $0)
            try graph.encode(source: source, destination: reference, commandBuffer: $0)
        }
        #expect(try gpu.read(output) == gpu.read(reference))
        #expect(throws: PostProcessEffectError.destination) {
            try effect.encode(source: source, destination: source, commandBuffer: command)
        }
    }

    private func makeAttributes(width: Int, height: Int, normal: (Int, Int) -> UInt32) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: SceneGraph.surfaceAttributesPixelFormat, width: width, height: height, mipmapped: false
        )
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        let texture = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let pixels = (0..<(width * height)).map { SIMD2(normal($0 % width, $0 / width), 0) }
        pixels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: bytes.baseAddress!, bytesPerRow: width * MemoryLayout<SIMD2<UInt32>>.stride
            )
        }
        return texture
    }

    private func smoothstep(_ a: Float, _ b: Float, _ value: Float) -> Float {
        let t = min(max((value - a) / (b - a), 0), 1)
        return t * t * (3 - 2 * t)
    }
}
