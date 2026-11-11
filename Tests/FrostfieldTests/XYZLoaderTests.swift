import Foundation
import Frostfield
import Metal
import Testing
import simd

struct XYZLoaderTests {
    @Test("XYZ and XYZRGB accept whitespace and preserve point and color order")
    func variants() throws {
        let plain = try XYZLoader.load("# point\r\n\t+1 -2 3e0\r\n")
        #expect(plain.positions == [SIMD3(1, -2, 3)] && plain.colors == nil)
        let mixed = try XYZLoader.load("""
            # cloud
            -2 0 -4
             0 0 -4 128 255 0 # colored
             2 0 -4 0 64 255
             3 0 -4
            """)
        #expect(mixed.positions == [SIMD3(-2, 0, -4), SIMD3(0, 0, -4), SIMD3(2, 0, -4), SIMD3(3, 0, -4)])
        let colors = try #require(mixed.colors)
        #expect(colors.count == 4 && colors[0] == SIMD4(repeating: 1) && colors[3] == SIMD4(repeating: 1))
        #expect(isNearly(colors[1], SIMD4(0.21586, 1, 0, 1), tolerance: 0.0001))
        #expect(isNearly(colors[2], SIMD4(0, 0.05127, 1, 1), tolerance: 0.0001))
    }

    @Test("XYZ reports invalid UTF-8, incomplete records, nonfinite positions, and out-of-range RGB")
    func malformed() {
        #expect(throws: XYZLoader.Failure.encoding) { try XYZLoader.load(Data([0xFF])) }
        #expect(throws: XYZLoader.Failure.empty) { try XYZLoader.load("# comment\n\n") }
        #expect(throws: XYZLoader.Failure.invalidRecord(2)) { try XYZLoader.load("# point\n1 2") }
        #expect(throws: XYZLoader.Failure.invalidRecord(1)) { try XYZLoader.load("nan 2 3") }
        #expect(throws: XYZLoader.Failure.invalidRecord(1)) { try XYZLoader.load("1 2 3 256 0 1") }
    }

    @Test("XYZRGB file imports feed Metal point positions and linear colors")
    func render() throws {
        let data = Data("\u{feff}-2 0 -4 128 255 0\n2 0 -4 0 64 255\n".utf8)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("xyz")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let cloud = try XYZLoader.load(contentsOf: url)
        let fixture = try PrimitiveRenderFixture(side: 64)
        let renderer = try PointsRenderer(
            device: fixture.device, cloud: cloud, material: PointsMaterial(size: .pixels(6))
        )
        let pixels = try fixture.render(fixture.graph([SceneNode(primitive: .points(renderer))])).pixels
        #expect(isNearly(pixels[32 * 64 + 16], SIMD4(0.21586, 1, 0, 1)))
        #expect(isNearly(pixels[32 * 64 + 48], SIMD4(0, 0.05127, 1, 1)))
        #expect(pixels.filter { $0.w == 1 }.count == 72)
    }
}
