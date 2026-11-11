import CoreGraphics
import Foundation
import Frostfield
import Testing

struct SVGStyleTests {
    @Test("SVG stylesheet classes and IDs layer over presentation attributes under inline paint")
    func selectors() throws {
        let doc = try SVGLoader.load("""
            <svg xmlns="http://www.w3.org/2000/svg">
              <rect id="main" class="red outlined" fill="black" style="fill:blue" width="3" height="2"/>
              <rect id="accent" class="red outlined" fill="black" width="3" height="2"/>
              <defs><style><![CDATA[
                /* Styles declared inside defs are available across the document. */
                @import url("file:///no-style.css");
                .red, .fallback { fill: red; }
                #main, #accent { fill: rebeccapurple; }
                .outlined { stroke: green; stroke-width: 2; }
                /* Unfinished comment leaves earlier rules intact.
              ]]></style></defs>
            </svg>
            """)
        #expect(doc.elements.count == 2)
        #expect(doc.elements[0].fill == .solid(SIMD4(0, 0, 1, 1)))
        guard case .solid(let accent) = doc.elements[1].fill else { Issue.record("Expected solid accent"); return }
        #expect(isNearly(accent, SIMD4(0.132868, 0.033105, 0.318547, 1)))
        for element in doc.elements {
            let style = try #require(element.stroke)
            #expect(style.width == 2)
            guard case .solid(let green) = style.paint else { Issue.record("Expected solid stroke"); return }
            #expect(isNearly(green, SIMD4(0, 0.21586, 0, 1)))
        }
    }

    @Test("Local SVG use expands forward references with inherited paint and independent transforms")
    func instances() throws {
        let doc = try SVGLoader.load("""
            <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
              <use href="#shape" x="2" y="3" transform="translate(10 0) scale(2)" fill="orange"/>
              <use xlink:href="#shape" x="-2" fill="blue"/>
              <defs><symbol id="shape" transform="translate(1 0)">
                <rect id="part" width="3" height="2"/>
              </symbol></defs>
            </svg>
            """)
        #expect(doc.elements.map(\.id) == ["part", "part"])
        #expect(doc.elements[0].path.boundingBoxOfPath == CGRect(x: 16, y: 6, width: 6, height: 4))
        #expect(doc.elements[1].path.boundingBoxOfPath == CGRect(x: -1, y: 0, width: 3, height: 2))
        #expect(doc.elements[1].fill == .solid(SIMD4(0, 0, 1, 1)))
    }

    @Test("SVG class-styled use instances reach Metal in distinct positions and colors")
    func render() throws {
        let doc = try SVGLoader.load("""
            <svg>
              <style>.first {fill:#ff0000} .second {fill:#0000ff}</style>
              <defs><rect id="plate" x="-.35" y="-.35" width=".7" height=".7"/></defs>
              <use href="#plate" class="first" x="-.7"/>
              <use href="#plate" class="second" x=".7"/>
            </svg>
            """)
        let fixture = try PrimitiveRenderFixture()
        let nodes = try doc.elements.map { element in
            let mesh = try #require(try element.fillGeometry())
            guard case .solid(let rgba) = element.fill else { throw SVGLoader.Failure.geometry }
            let renderer = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: mesh,
                material: .classic(ClassicMaterial(model: .basic, color: SIMD3(rgba.x, rgba.y, rgba.z)))
            )
            return SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: renderer)
        }
        let result = try fixture.render(fixture.graph(nodes))
        #expect(result.statistics.drawnMeshCount == 2)
        #expect(result.pixels[32 * 64 + 21] == SIMD4(1, 0, 0, 1))
        #expect(result.pixels[32 * 64 + 43] == SIMD4(0, 0, 1, 1))
        #expect(result.pixels[32 * 64 + 32] == .zero)
    }

    @Test("SVG local expansion rejects cycles and excessive depth and skips external references")
    func referenceLimits() throws {
        #expect(throws: SVGLoader.Failure.referenceCycle) {
            try SVGLoader.load("<svg><use href='#a'/><defs><g id='a'><use href='#a'/></g></defs></svg>")
        }
        let chain = (0..<70).map { "<g id='n\($0)'><use href='#n\($0 + 1)'/></g>" }.joined()
        #expect(throws: SVGLoader.Failure.size) {
            try SVGLoader.load("<svg><use href='#n0'/><defs>\(chain)</defs></svg>")
        }
        let doc = try SVGLoader.load("<svg><use href='file:///no-source.svg#shape'/><use href='#missing'/></svg>")
        #expect(doc.elements.isEmpty)
    }
}
