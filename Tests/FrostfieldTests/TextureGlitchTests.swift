import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TextureGlitchTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Authored noise controls nearest-cell displacement, both band directions and RGB split",
          arguments: [Float(1), -1])
    func shaderReference(sign: Float) throws {
        let width = 13, height = 11
        let noiseWidth = 4, noiseHeight = 3
        let noise: [Float] = [0, 0.75, 0.125, 1, 0.5, 0.25, 0.875, 0.125, 1, 0.5, 0, 0.75]
        let displacement = try gpu.color(width: noiseWidth, height: noiseHeight) { x, y in
            SIMD4(noise[y * noiseWidth + x], 1, 0, 1)
        }
        let source = try gpu.color(width: width, height: height) { x, y in
            let alpha: Float = x < 4 && y > 5 ? 0 : ((x + 2 * y).isMultiple(of: 4) ? 0.5 : 1)
            return SIMD4(SIMD3(Float(x) / 4, Float(y) / 8, Float(x + y) / 16) * alpha, alpha)
        }
        let destination = try gpu.color(width: width, height: height)
        let pass = try PostProcessPass(
            graph: .glitchWithDisplacement(), device: gpu.device, textures: ["Displacement": displacement]
        )
        let seed: Float = 0.8
        let seedX: Float = -0.35 * sign, seedY: Float = 0.45 * sign
        let distortionX: Float = 0.55, distortionY: Float = 0.42, band: Float = 0.13
        let amount: Float = 0.075, angle: Float = 0.63
        for (name, value) in [
            ("Seed", seed), ("SeedX", seedX), ("SeedY", seedY), ("DistortionX", distortionX), ("DistortionY", distortionY),
            ("ColumnSize", band), ("Amount", amount), ("Angle", angle), ("SnowStrength", 0)
        ] {
            try pass.uniforms.set(name, to: value)
        }
        try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }
        let input = try gpu.read(source)
        let warped = try gpu.read(destination)
        var affected = 0
        var transparent = 0
        for (index, actual) in warped.enumerated() {
            let x = index % width, y = index / width
            let uv = SIMD2((Float(x) + 0.5) / Float(width), 1 - (Float(y) + 0.5) / Float(height))
            let noiseUV = uv * (seed * seed)
            let nx = min(Int(noiseUV.x * Float(noiseWidth)), noiseWidth - 1)
            let ny = min(Int((1 - noiseUV.y) * Float(noiseHeight)), noiseHeight - 1)
            let disp = noise[ny * noiseWidth + nx]
            var point = uv
            if point.y > distortionX - band * seed && point.y < distortionX + band {
                point.y = seedX > 0 ? 1 - point.y - distortionY : distortionY
            }
            if point.x > distortionY - band * seed && point.x < distortionY + band {
                point.x = seedY > 0 ? distortionX : 1 - point.x - distortionX
            }
            point += SIMD2(seedX, seedY) * (disp * seed / 5)
            let offset = SIMD2(cos(angle), sin(angle)) * amount
            let red = sample(input, width: width, height: height, at: point + offset)
            let center = sample(input, width: width, height: height, at: point)
            let blue = sample(input, width: width, height: height, at: point - offset)
            let r = red.w > 0 ? red.x * center.w / red.w : 0
            let b = blue.w > 0 ? blue.z * center.w / blue.w : 0
            let expected = SIMD4(r, center.y, b, center.w)
            #expect(PostProcessFixture.close(actual, expected, tolerance: 0.009), "pixel \(index): \(actual) vs \(expected)")
            if !PostProcessFixture.close(actual, input[index], tolerance: 0.01) { affected += 1 }
            if center.w == 0 {
                #expect(actual == .zero)
                transparent += 1
            }
        }
        #expect(affected > 70 && transparent > 0)

        try pass.uniforms.set("SnowStrength", to: Float(1))
        try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }
        let snow = try gpu.read(destination)
        #expect(snow != warped)
        #expect(zip(warped, snow).allSatisfy { $0.0.w == $0.1.w && ($0.0.w != 0 || $0.1 == .zero) })
        try pass.uniforms.set("Bypass", to: Float(1))
        try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }
        #expect(try gpu.read(destination) == input)
    }

    @Test("Seeded displacement is replayable and produces a real GPU texture")
    func seededDisplacement() throws {
        let size = 8
        let first = try GlitchRandomizer.makeDisplacementTexture(device: gpu.device, size: size, seed: 42)
        let second = try GlitchRandomizer.makeDisplacementTexture(device: gpu.device, size: size, seed: 42)
        let different = try GlitchRandomizer.makeDisplacementTexture(device: gpu.device, size: size, seed: 17)
        func read(_ texture: any MTLTexture) -> [Float] {
            var values = Array(repeating: Float.zero, count: size * size)
            values.withUnsafeMutableBytes { bytes in
                texture.getBytes(bytes.baseAddress!, bytesPerRow: size * 4, from: MTLRegionMake2D(0, 0, size, size), mipmapLevel: 0)
            }
            return values
        }
        #expect(first.pixelFormat == .r32Float && first.width == size && first.height == size)
        let values = read(first)
        #expect(values == read(second))
        #expect(values != read(different))
        #expect(values.allSatisfy { $0 >= 0 && $0 < 1 } && Set(values).count > 55)
        #expect(throws: PostProcessEffectError.self) {
            try GlitchRandomizer.makeDisplacementTexture(device: gpu.device, size: 1025)
        }

        let source = try gpu.color(width: 16, height: 12) { x, y in SIMD4(Float(x) / 8, Float(y) / 6, 0.75, 1) }
        let destination = try gpu.color(width: 16, height: 12)
        let graph = try PostProcessGraph.glitchWithDisplacement()
        func render(_ map: any MTLTexture) throws -> [SIMD4<Float>] {
            let pass = try PostProcessPass(graph: graph, device: gpu.device, textures: ["Displacement": map])
            try pass.uniforms.set("Seed", to: Float(1))
            try pass.uniforms.set("SeedX", to: Float(0.8))
            try pass.uniforms.set("SeedY", to: Float(-0.6))
            try gpu.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }
            return try gpu.read(destination)
        }
        let result = try render(first)
        #expect(try render(second) == result)
        #expect(try render(different) != result)
    }

    @Test("Texture glitch uses the same deterministic burst, weak and bypass schedule as the hash graph")
    func schedule() throws {
        let hash = try PostProcessGraph.glitch()
        let textured = try PostProcessGraph.glitchWithDisplacement()
        let hashUniforms = try ShaderUniforms(graph: hash)
        let textureUniforms = try ShaderUniforms(graph: textured)
        var hashRandomizer = GlitchRandomizer(seed: 0x47CA)
        var textureRandomizer = GlitchRandomizer(seed: 0x47CA)
        func values(_ uniforms: ShaderUniforms, graph: PostProcessGraph) -> [String: Float] {
            Dictionary(uniqueKeysWithValues: zip(graph.uniforms.map(\.name), uniforms.snapshot().values.map(\.x)))
        }
        var burstFrames: [Int] = []
        var wasBypassed = true
        var weak = 0, bypassed = 0
        for index in 0..<600 {
            try hashRandomizer.advance(hashUniforms)
            try textureRandomizer.advance(textureUniforms)
            var frame = values(textureUniforms, graph: textured)
            #expect(frame.removeValue(forKey: "SnowStrength") == 1)
            #expect(frame == values(hashUniforms, graph: hash))
            let bypass = frame["Bypass"] == 1
            let amount = try #require(frame["Amount"])
            if bypass {
                bypassed += 1
            } else if wasBypassed {
                burstFrames.append(index)
                #expect(amount <= 1 / 30)
            } else {
                weak += 1
                #expect(amount <= 1 / 90)
            }
            wasBypassed = bypass
        }
        #expect(burstFrames.first == 0 && burstFrames.count >= 3)
        #expect(weak > 80 && bypassed > 350)
        #expect(zip(burstFrames.dropFirst(), burstFrames).allSatisfy { (120...239).contains($0.0 - $0.1) })

        textureRandomizer.goWild = true
        for _ in 0..<60 {
            try textureRandomizer.advance(textureUniforms)
            let frame = values(textureUniforms, graph: textured)
            #expect(frame["Bypass"] == 0)
        }
    }

    private func sample(_ image: [SIMD4<Float>], width: Int, height: Int, at glUV: SIMD2<Float>) -> SIMD4<Float> {
        let point = SIMD2(glUV.x * Float(width) - 0.5, (1 - glUV.y) * Float(height) - 0.5)
        let left = Int(floor(point.x)), top = Int(floor(point.y))
        let x = point.x - Float(left), y = point.y - Float(top)
        func texel(_ x: Int, _ y: Int) -> SIMD4<Float> {
            image[min(max(y, 0), height - 1) * width + min(max(x, 0), width - 1)]
        }
        return (texel(left, top) * (1 - x) + texel(left + 1, top) * x) * (1 - y)
            + (texel(left, top + 1) * (1 - x) + texel(left + 1, top + 1) * x) * y
    }
}
