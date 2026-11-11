import CoreGraphics
import Foundation
import Frostfield
import Metal
import Testing

struct SVGGradientTests {
    @Test("SVG gradient inheritance applies bounding-box units, transforms and opacity")
    func inheritedLinear() throws {
        let document = try SVGLoader.load("""
            <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
              <defs>
                <linearGradient id="base" gradientTransform="translate(.25 0)">
                  <stop offset="0%" stop-color="red" stop-opacity="25%"/>
                  <stop offset="100%" style="stop-color:blue;stop-opacity:.75"/>
                </linearGradient>
                <linearGradient id="copy" xlink:href="#base" spreadMethod="reflect"/>
              </defs>
              <rect width="2" height="1" transform="translate(10 20) scale(3 4)"
                    fill="url( '#copy' )" fill-opacity=".5"/>
            </svg>
            """)
        let element = try #require(document.elements.first)
        guard case .gradient(let gradient) = element.fill else { Issue.record("Expected inherited gradient"); return }
        #expect(gradient.units == .objectBoundingBox)
        #expect(gradient.spread == .reflect)
        #expect(gradient.stops.map(\.offset) == [0, 1])
        #expect(try element.gradientColor(at: CGPoint(x: 11.5, y: 22)) == SIMD4(1, 0, 0, 0.125))
        let center = try #require(try element.gradientColor(at: CGPoint(x: 14.5, y: 22)))
        #expect(isNearly(center, SIMD4(0.214041, 0, 0.214041, 0.25)))
        let reflected = try #require(try element.gradientColor(at: CGPoint(x: 19, y: 22)))
        #expect(isNearly(reflected, SIMD4(0.050876, 0, 0.522522, 0.3125)))
    }

    @Test("SVG user-space percentages, linear interpolation and repeat use the viewport")
    func userSpace() throws {
        let document = try SVGLoader.load("""
            <svg viewBox="-2 -4 20 10">
              <defs><linearGradient id="paint" gradientUnits="userSpaceOnUse" x1="0%" x2="25%"
                                   y1="0" y2="0" spreadMethod="repeat" color-interpolation="linearRGB">
                <stop offset="0" stop-color="red"/>
                <stop offset="50%" stop-color="lime"/>
                <stop offset="1" stop-color="blue"/>
              </linearGradient></defs>
              <rect x="-2" width="16" height="5" fill="url(#paint)"/>
            </svg>
            """)
        let element = try #require(document.elements.first)
        guard case .gradient(let gradient) = element.fill else { Issue.record("Expected gradient"); return }
        #expect(gradient.units == .userSpaceOnUse)
        #expect(gradient.interpolatesInLinearRGB)
        #expect(gradient.stops.map(\.offset) == [0, 0.5, 1])
        #expect(try element.gradientColor(at: CGPoint(x: -0.75, y: 0)) == SIMD4(0.5, 0.5, 0, 1))
        #expect(try element.gradientColor(at: CGPoint(x: 5.5, y: 0)) == SIMD4(0, 1, 0, 1))
        #expect(try element.gradientColor(at: CGPoint(x: -4.5, y: 0)) == SIMD4(0, 1, 0, 1))
    }

    @Test("SVG radial gradient keeps focus and elliptical coordinates through a gradient transform")
    func radial() throws {
        let document = try SVGLoader.load("""
            <svg>
              <defs><radialGradient id="paint" cx=".5" cy=".5" r=".5" fx=".25" fy=".5"
                                   gradientTransform="scale(1 .5)" spreadMethod="repeat">
                <stop stop-color="red"/><stop offset="1" stop-color="blue"/>
              </radialGradient></defs>
              <rect width="4" height="2" fill="url(#paint)"/>
            </svg>
            """)
        let element = try #require(document.elements.first)
        #expect(try element.gradientColor(at: CGPoint(x: 1, y: 0.5)) == SIMD4(1, 0, 0, 1))
        let midpoint = try #require(try element.gradientColor(at: CGPoint(x: 2.5, y: 0.5)))
        #expect(isNearly(midpoint, SIMD4(0.214041, 0, 0.214041, 1)))
        let repeated = try #require(try element.gradientColor(at: CGPoint(x: 4.75, y: 0.5)))
        #expect(isNearly(repeated, SIMD4(0.522522, 0, 0.050876, 1)))
        let offAxis = try #require(try element.gradientColor(at: CGPoint(x: 2.5, y: 0.8)))
        #expect(offAxis.z > midpoint.z)
    }

    @Test("Prepared SVG gradient texture uses the reflected mesh UVs in Metal")
    func render() throws {
        let document = try SVGLoader.load("""
            <svg>
              <defs><linearGradient id="paint" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0" stop-color="red" stop-opacity=".5"/>
                <stop offset="1" stop-color="blue" stop-opacity="1"/>
              </linearGradient></defs>
              <rect x="-1" y="-1" width="2" height="2" fill="url(#paint)" opacity=".75"/>
            </svg>
            """)
        let element = try #require(document.elements.first)
        let fixture = try PrimitiveRenderFixture()
        let prepared = try #require(try element.prepareGradientFill(device: fixture.device, resolution: 128, yUp: true))
        #expect(prepared.texture.pixelFormat == .rgba16Float)
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: prepared.geometry,
            material: .textured(prepared.texture),
            renderState: MeshRenderState(blending: .normal, transparent: true, depthWrite: false)
        )
        let node = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: mesh)
        let pixels = try fixture.render(fixture.graph([node])).pixels
        for row in [24, 40] {
            let y = (Double(row) + 0.5 - 32) / 16
            let expected = try #require(try element.gradientColor(at: CGPoint(x: 0, y: y)))
            let premultiplied = SIMD4(expected.x * expected.w, expected.y * expected.w, expected.z * expected.w, expected.w)
            #expect(isNearly(pixels[row * 64 + 32], premultiplied))
        }
        #expect(pixels[24 * 64 + 32].x > pixels[24 * 64 + 32].z)
        #expect(pixels[40 * 64 + 32].z > pixels[40 * 64 + 32].x)
        #expect(pixels[10 * 64 + 32] == .zero)
    }

    @Test("SVG gradient cycles, singular placements and missing local paint have distinct outcomes")
    func malformed() throws {
        #expect(throws: SVGLoader.Failure.referenceCycle) {
            try SVGLoader.load("""
                <svg><defs><linearGradient id="a" href="#b"/><linearGradient id="b" href="#a"/></defs>
                <rect width="2" height="2" fill="url(#a)"/></svg>
                """)
        }
        let document = try SVGLoader.load("""
            <svg><defs><linearGradient id="a" gradientTransform="scale(0)">
              <stop stop-color="red"/>
            </linearGradient></defs>
            <rect width="2" height="2" fill="url(#a)"/>
            <rect width="2" height="2" fill="url(#missing)"/>
            </svg>
            """)
        #expect(throws: SVGLoader.Failure.gradient("singular transform")) {
            try document.elements[0].gradientColor(at: .zero)
        }
        #expect(document.elements[1].fill == .unresolved("url(#missing)"))
        #expect(try document.elements[1].gradientColor(at: .zero) == nil)
    }
}
