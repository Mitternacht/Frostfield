@testable import Frostfield
import Foundation
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SMAATests {
    /// FNV-1a over bytes.
    private static func fnv1a(_ bytes: [UInt8]) -> UInt64 {
        bytes.reduce(0xcbf2_9ce4_8422_2325 as UInt64) { ($0 ^ UInt64($1)) &* 0x0000_0100_0000_01b3 }
    }

    @Test("Generated lookup tables equal the published SMAA 1x tables")
    func lookupTables() {
        // Hashes of the first 160 × 80 RG8 texels of AreaTex.h and of SearchTex.h from
        // github.com/iryoku/smaa, computed from those files; the tables themselves are not shipped.
        let area = SMAALookupTextures.areaBytes()
        let search = SMAALookupTextures.searchBytes()
        #expect(area.count == 160 * 80 * 2 && search.count == 64 * 16)
        #expect(Self.fnv1a(area) == 0xb3c5_c9f6_6925_88e8)
        #expect(Self.fnv1a(search) == 0x21c1_fcf0_aa63_1065)
        #expect(area.reduce(0) { $0 + Int($1) } == 889_395)
        #expect(search.reduce(0) { $0 + Int($1) } == 22_860)
        // An unfiltered pattern (0) has zero area; an L pattern (1) at distance 0 covers a quarter
        // of the pixel below the edge: area((0, -0.5) → (0.5, 0)) over x 0...1 = 0.125, quantized to 31.
        #expect(area[0] == 0 && area[1] == 0)
        let lPattern = (0 * 160 + 16 * 3) * 2 // pattern 1 is in slot (3, 0)
        #expect(area[lPattern] == 31 && area[lPattern + 1] == 0)
    }

    @Test("SMAA reduces edge aliasing against an 8 × 8 supersampled reference", arguments: [Float(0.25), 0.7])
    func edgeAliasing(angle: Float) throws {
        let scene = try EdgeScene(angle: angle)
        let reference = try scene.supersampledReference()
        let raw = try scene.render()
        let smaa = try SMAA(device: scene.device)
        let output = try scene.makeTarget().color
        try scene.run { try smaa.encode(source: raw.color, destination: output, commandBuffer: $0) }

        let rawPixels = try EdgeScene.read(raw.color)
        let filtered = try EdgeScene.read(output)
        let band = scene.edgeBand(reference)
        let before = EdgeScene.rmse(rawPixels, reference, over: band)
        let after = EdgeScene.rmse(filtered, reference, over: band)
        print("[rasterlook] SMAA angle \(angle): edge RMSE \(before) -> \(after) over \(band.count) pixels")
        EdgeScene.save(filtered, side: scene.side, name: "smaa-\(angle)")
        #expect(after < 0.7 * before, "SMAA edge RMSE \(after) versus \(before) without it")

        var flatCount = 0
        for y in 2..<(scene.side - 2) {
            for x in 2..<(scene.side - 2) {
                let center = rawPixels[y * scene.side + x]
                let flat = (-2...2).allSatisfy { v in (-2...2).allSatisfy { u in rawPixels[(y + v) * scene.side + x + u] == center } }
                guard flat else { continue }
                flatCount += 1
                #expect(filtered[y * scene.side + x] == center, "flat pixel \(x), \(y)")
            }
        }
        #expect(flatCount > scene.side * scene.side / 2)
    }

    @Test("Diagonal detection at high quality recovers a 40° edge better than medium quality")
    func diagonalDetection() throws {
        let scene = try EdgeScene(angle: 0.7)
        let reference = try scene.supersampledReference()
        let band = scene.edgeBand(reference)
        let raw = try scene.render()
        let smaa = try SMAA(device: scene.device)
        var errors: [SMAA.Quality: Float] = [:]
        for quality in [SMAA.Quality.medium, .high] {
            let output = try scene.makeTarget().color
            try scene.run { try smaa.encode(source: raw.color, destination: output, quality: quality, commandBuffer: $0) }
            errors[quality] = EdgeScene.rmse(try EdgeScene.read(output), reference, over: band)
        }
        let medium = try #require(errors[.medium]), high = try #require(errors[.high])
        print("[rasterlook] SMAA 40.1° edge: medium \(medium), high \(high)")
        // Both presets use threshold 0.1, and 8 steps already reach the ends of these short
        // stair steps, so the difference is the diagonal search that only high runs.
        #expect(high < 0.7 * medium, "high \(high) versus medium \(medium)")
    }

    @Test("Corner detection keeps the corners of a pixel-aligned square sharp")
    func cornerDetection() throws {
        let fixture = try PostProcessFixture()
        let side = 48
        // A white square on transparent black with pixel-aligned sides, which a supersampled
        // render reproduces exactly; any blending is error.
        let square = try fixture.color(width: side, height: side) { x, y in
            (12..<36).contains(x) && (12..<36).contains(y) ? SIMD4(1, 1, 1, 1) : .zero
        }
        let original = try fixture.read(square)
        let smaa = try SMAA(device: fixture.device)
        var changes: [SMAA.Quality: (count: Int, sum: Float)] = [:]
        for quality in [SMAA.Quality.medium, .high] {
            let output = try fixture.color(width: side, height: side)
            try fixture.run { try smaa.encode(source: square, destination: output, quality: quality, commandBuffer: $0) }
            let pixels = try fixture.read(output)
            let differences = zip(pixels, original).map { simd_reduce_max(simd_abs($0 - $1)) }
            changes[quality] = (differences.filter { $0 > 0 }.count, differences.reduce(0, +))
        }
        let medium = try #require(changes[.medium]), high = try #require(changes[.high])
        print("[rasterlook] SMAA aligned square: medium changes \(medium.count) pixels by \(medium.sum) in total, high \(high.count) by \(high.sum)")
        // Without corner detection, each end of every side is rounded like an L pattern.
        #expect(medium.sum > 0.5)
        // Corner rounding 25% keeps a quarter of that blending.
        #expect(high.sum < 0.5 * medium.sum, "high \(high.sum) versus medium \(medium.sum)")
    }

    @Test("SMAA finds edges between extended-range highlights")
    func extendedRangeEdges() throws {
        // A 14.3° staircase between display values 2 and 6, as after tone mapping with headroom.
        let scene = try EdgeScene(angle: 0.25)
        let coverage = try EdgeScene.read(scene.render().color).map(\.w)
        let reference = try scene.supersampledReference()
        let band = scene.edgeBand(reference)
        let fixture = try PostProcessFixture()
        func hdr(_ coverage: Float) -> SIMD4<Float> {
            let value = 2 + 4 * coverage
            return SIMD4(value, value, value, 1)
        }
        let source = try fixture.color(width: scene.side, height: scene.side) { x, y in hdr(coverage[y * scene.side + x]) }
        let expected = reference.map { hdr($0.w) }
        let original = try fixture.read(source)
        let output = try fixture.color(width: scene.side, height: scene.side)
        try fixture.run { try SMAA(device: fixture.device).encode(source: source, destination: output, commandBuffer: $0) }
        let filtered = try fixture.read(output)
        let before = EdgeScene.rmse(original, expected, over: band)
        let after = EdgeScene.rmse(filtered, expected, over: band)
        let changed = zip(filtered, original).filter { $0 != $1 }.count
        print("[rasterlook] SMAA on a 2-to-6 staircase: \(changed) pixels changed, edge RMSE \(before) -> \(after)")
        #expect(changed > 0)
        #expect(after < 0.7 * before, "SMAA RMSE \(after) versus \(before) between highlights")
    }

    @Test("SMAA keeps long axis-aligned edges sharp, where FXAA softens them")
    func straightEdges() throws {
        let scene = try EdgeScene(angle: 0)
        let raw = try scene.render()
        let rawPixels = try EdgeScene.read(raw.color)
        let smaaOutput = try scene.makeTarget().color
        let fxaaOutput = try scene.makeTarget().color
        try scene.run { buffer in
            try SMAA(device: scene.device).encode(source: raw.color, destination: smaaOutput, commandBuffer: buffer)
            try FXAA(device: scene.device).encode(source: raw.color, destination: fxaaOutput, commandBuffer: buffer)
        }
        #expect(try EdgeScene.read(smaaOutput) == rawPixels)
        // FXAA's sub-pixel term blends a little across the straight edge.
        #expect(try EdgeScene.read(fxaaOutput) != rawPixels)
    }

    @Test("Every SMAA preset runs, and inputs are validated before encoding")
    func presetsAndValidation() throws {
        let scene = try EdgeScene(side: 32, angle: 0.4)
        let raw = try scene.render()
        let smaa = try SMAA(device: scene.device)
        for quality in [SMAA.Quality.low, .medium, .high, .ultra] {
            let output = try scene.makeTarget().color
            try scene.run { try smaa.encode(source: raw.color, destination: output, quality: quality, commandBuffer: $0) }
            let pixels = try EdgeScene.read(output)
            #expect(pixels.allSatisfy { $0.x.isFinite && $0.w >= 0 && $0.w <= 1 })
            #expect(pixels.contains { $0.w > 0.05 && $0.w < 0.95 }, "\(quality) blends the edge")
        }
        let buffer = try #require(scene.queue.makeCommandBuffer())
        #expect(throws: SMAA.Failure.destination) {
            try smaa.encode(source: raw.color, destination: raw.color, commandBuffer: buffer)
        }
        let large = try scene.makeTarget(side: 64).color
        #expect(throws: SMAA.Failure.sizeMismatch) {
            try smaa.encode(source: raw.color, destination: large, commandBuffer: buffer)
        }
        #expect(throws: SMAA.Failure.destinationFormat) { try SMAA(device: scene.device, destinationPixelFormat: .bgra8Unorm) }

        // As a chain stage it matches the standalone pass.
        let chain = try PostProcessChain(queue: scene.queue, effects: [.smaa(.ultra)])
        let chained = try scene.makeTarget().color
        let direct = try scene.makeTarget().color
        try scene.run { buffer in
            try chain.encode(source: raw.color, destination: chained, commandBuffer: buffer)
            try smaa.encode(source: raw.color, destination: direct, quality: .ultra, commandBuffer: buffer)
        }
        #expect(try EdgeScene.read(chained) == EdgeScene.read(direct))
    }
}
