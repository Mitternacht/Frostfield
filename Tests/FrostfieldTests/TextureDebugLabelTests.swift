import CoreText
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TextureDebugLabelTests {
    @Test("A named light label blends over the shadow preview and leaves other pixels intact")
    func labelOverlay() throws {
        let fixture = try RasterLightingFixture(width: 64, height: 48)
        let viewer = try TextureDebugPass(device: fixture.device)
        let node = SceneNode(light: try PunctualLight(kind: .spot()), name: "L")
        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, 24, nil)
        let label = try TextureDebugPass.Label(try #require(node.name), device: fixture.device, font: font)
        let color = try fixture.colorTarget(), depth = try fixture.depthTarget()
        let origin = SIMD2(8, 7)
        try fixture.run { buffer in
            let clear = MTLRenderPassDescriptor()
            clear.depthAttachment.texture = depth
            clear.depthAttachment.loadAction = .clear
            clear.depthAttachment.storeAction = .store
            clear.depthAttachment.clearDepth = 0.5
            let encoder = try #require(buffer.makeRenderCommandEncoder(descriptor: clear))
            encoder.endEncoding()
            try viewer.encode(source: .depth(depth), destination: color, commandBuffer: buffer)
            try viewer.encode(label: label, destination: color, origin: origin, color: SIMD4(1, 0, 0, 0.5),
                              commandBuffer: buffer)
        }
        let pixels = try fixture.pixels(of: color)
        let unchanged = SIMD4<Float>(0.5, 0.5, 0.5, 1)
        let red = SIMD4<Float>(0.75, 0.25, 0.25, 1)
        #expect(pixels.contains(red))
        var top = 0, bottom = 0
        for y in 0..<48 {
            for x in 0..<64 {
                let pixel = pixels[y * 64 + x]
                let inside = (origin.x..<(origin.x + label.width)).contains(x)
                    && (origin.y..<(origin.y + label.height)).contains(y)
                if !inside {
                    #expect(pixel == unchanged)
                } else if pixel == red {
                    if y < origin.y + label.height / 2 { top += 1 } else { bottom += 1 }
                }
            }
        }
        #expect(top > 0 && bottom > top)
        #expect(pixels.contains { (0.5..<0.75).contains($0.x) && $0.x > 0.5 && $0.w == 1 })
    }

    @Test("Invalid text, tint or position fails before starting an overlay draw")
    func validation() throws {
        let fixture = try RasterLightingFixture(width: 64, height: 48)
        let viewer = try TextureDebugPass(device: fixture.device)
        for invalid in ["", "  ", "A\nB", String(repeating: "X", count: 1_025)] {
            #expect(throws: TextureDebugPass.Failure.labelText) {
                try TextureDebugPass.Label(invalid, device: fixture.device)
            }
        }
        let label = try TextureDebugPass.Label("Spot", device: fixture.device)
        let color = try fixture.colorTarget()
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: TextureDebugPass.Failure.region) {
            try viewer.encode(label: label, destination: color, origin: SIMD2(63, 0), commandBuffer: buffer)
        }
        #expect(throws: TextureDebugPass.Failure.labelColor) {
            try viewer.encode(label: label, destination: color, origin: .zero, color: SIMD4(.nan, 0, 0, 1),
                              commandBuffer: buffer)
        }
    }
}
