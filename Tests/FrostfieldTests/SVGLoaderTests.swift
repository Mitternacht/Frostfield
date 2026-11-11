import CoreGraphics
import Foundation
import Frostfield
import Testing

struct SVGLoaderTests {
    @Test("SVG path commands preserve implicit lines, reflected controls and close restarts")
    func commands() throws {
        let element = try one("M1 2 3 4 h2 v3 l-1-1 C5 6 7 8 9 10 s2 1 3 4 q3 0 4 1 t5 2 z l1 0")
        let path = Self.records(element.path)
        #expect(path.map(\.0) == [
            .moveToPoint, .addLineToPoint, .addLineToPoint, .addLineToPoint, .addLineToPoint,
            .addCurveToPoint, .addCurveToPoint, .addQuadCurveToPoint, .addQuadCurveToPoint,
            .closeSubpath, .moveToPoint, .addLineToPoint,
        ])
        #expect(path[1].1 == [CGPoint(x: 3, y: 4)])
        #expect(path[3].1 == [CGPoint(x: 5, y: 7)])
        #expect(path[6].1 == [CGPoint(x: 11, y: 12), CGPoint(x: 11, y: 11), CGPoint(x: 12, y: 14)])
        #expect(path[8].1 == [CGPoint(x: 17, y: 16), CGPoint(x: 21, y: 17)])
        #expect(path[10].1 == [CGPoint(x: 1, y: 2)])
        #expect(path[11].1 == [CGPoint(x: 2, y: 2)])
    }

    @Test("SVG elliptical arcs read adjacent flags, correct radii, and degenerate to a line")
    func arcs() throws {
        let element = try one("M0 0 A2 2 0 0110 0 A0 10 0 0 1 12 3 A7 5 15 0 0 12 3")
        let bounds = element.path.boundingBoxOfPath
        #expect(abs(bounds.minX) < 0.00001)
        #expect(abs(bounds.minY + 5) < 0.00001)
        #expect(bounds.maxX == 12 && bounds.maxY == 3)
        #expect(element.path.currentPoint == CGPoint(x: 12, y: 3))
        #expect(Self.records(element.path).filter { $0.0 == .addCurveToPoint }.count == 4)

        let small = try one("M0 0 a8 5 20 0 1 4 3")
        let large = try one("M0 0 a8 5 20 1 1 4 3")
        #expect(small.path.boundingBoxOfPath.width < large.path.boundingBoxOfPath.width)
        #expect(large.path.currentPoint == CGPoint(x: 4, y: 3))
    }

    @Test("SVG nested affine order, authored viewBox and presentation style are retained")
    func transformsAndStyle() throws {
        let document = try SVGLoader.load("""
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="-8 -4 30 20">
              <g transform="translate(20,10) scale(2)" color="rebeccapurple" opacity=".5">
                <g transform="rotate(90 1 1) skewX(0) skewY(0)">
                  <path id="part" transform="matrix(1 0 0 1 1 0)" fill="red"
                        style="fill:currentColor;fill-opacity:25%" d="M1 0 L2 0 L2 1Z"/>
                </g>
              </g>
              <rect x="1in" y="10px" width="72pt" height="1cm"/>
              <circle cx="10" cy="11" r="2"/>
              <ellipse cx="2" cy="3" rx="4" ry="2"/>
              <line x1="0" y1="0" x2="1" y2="2"/>
              <polyline points="0,0 1,0 1,1"/>
              <polygon points="0,0 1,0 1,1"/>
              <g display="none"><path d="M0 0H2V2Z"/></g>
            </svg>
            """)
        #expect(document.viewBox == CGRect(x: -8, y: -4, width: 30, height: 20))
        #expect(document.elements.count == 7)
        let first = document.elements[0]
        #expect(first.id == "part")
        let start = try #require(Self.records(first.path).first?.1.first)
        #expect(abs(start.x - 24) < 0.00001 && abs(start.y - 14) < 0.00001)
        #expect(first.fillOpacity == 0.125)
        if case .solid(let color) = first.fill {
            #expect(isNearly(color, SIMD4(0.132868, 0.033105, 0.318547, 1)))
        } else {
            Issue.record("Expected a resolved currentColor fill")
        }
        let rect = document.elements[1].path.boundingBoxOfPath
        #expect(rect.minX == 90 && rect.minY == 10 && rect.width == 90)
        #expect(abs(rect.height - 90 / 2.54) < 0.00001)
        #expect(document.elements[2].path.boundingBoxOfPath == CGRect(x: 8, y: 9, width: 4, height: 4))
        #expect(try document.elements[4].fillGeometry() == nil)
        #expect(try document.elements[5].filledShapes().first?.area == 0.5)
        #expect(try document.elements[6].filledShapes().first?.area == 0.5)
    }

    @Test("SVG fill rules leave real holes in the Metal render")
    func fillRuleRendering() throws {
        let data = "M-1-1 H1 V1 H-1Z M-.4-.4 H.4 V.4 H-.4Z"
        let evenOdd = try one(data, style: "fill:#ff8800;fill-opacity:.5;fill-rule:evenodd")
        let nonzero = try one(data)
        let shapes = try evenOdd.filledShapes(yUp: true)
        #expect(shapes.count == 1 && shapes[0].holes.count == 1)
        #expect(abs(shapes[0].area - 3.36) < 0.000001)
        #expect(try nonzero.filledShapes().first?.area == 4)

        let fixture = try PrimitiveRenderFixture()
        let mesh = try #require(try evenOdd.fillGeometry(yUp: true))
        guard case .solid(let paint) = evenOdd.fill else {
            Issue.record("Expected a resolved SVG fill")
            return
        }
        let renderer = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: mesh,
            material: .classic(ClassicMaterial(
                model: .basic, color: SIMD3(paint.x, paint.y, paint.z), opacity: evenOdd.fillOpacity * paint.w
            )),
            renderState: MeshRenderState(blending: .normal, transparent: true, depthWrite: false)
        )
        let node = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: renderer)
        let pixels = try fixture.render(fixture.graph([node])).pixels
        #expect(pixels[32 * 64 + 32] == .zero)
        #expect(isNearly(pixels[32 * 64 + 44], SIMD4(0.5, 0.1231, 0, 0.5)))
        #expect(pixels[10 * 64 + 32] == .zero)
    }

    @Test("The legacy SVG 1.1 public header is discarded before parsing")
    func standardPublicDeclaration() throws {
        let header = """
            <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
                     "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
            """
        let document = """
            <svg xmlns="http://www.w3.org/2000/svg">
              <rect width="5" height="4"/>
            </svg>
            """
        let imported = try SVGLoader.load("<?xml version='1.0' encoding='UTF-8'?>\n" + header + document)
        let direct = try SVGLoader.load(document)
        let importedShapes = try #require(imported.elements.first).filledShapes()
        let directShapes = try #require(direct.elements.first).filledShapes()
        #expect(importedShapes == directShapes)
        #expect(throws: SVGLoader.Failure.unsupportedXML) {
            try SVGLoader.load(header + header + document)
        }
        #expect(throws: SVGLoader.Failure.unsupportedXML) {
            try SVGLoader.load("""
                <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
                         "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd"
                         [<!ENTITY injected "unsafe">]>
                <svg/>
                """)
        }
        #expect(throws: SVGLoader.Failure.xml) {
            try SVGLoader.load(header + "<svg>&injected;</svg>")
        }
    }

    @Test("SVG rejects external XML definitions, malformed commands and excessive fill complexity")
    func validation() throws {
        #expect(throws: SVGLoader.Failure.unsupportedXML) {
            try SVGLoader.load("<!DOCTYPE svg SYSTEM 'file:///etc/passwd'><svg/>")
        }
        #expect(throws: SVGLoader.Failure.unsupportedXML) {
            try SVGLoader.load("<?xml version='1.0' encoding='UTF-7'?><svg/>")
        }
        #expect(throws: SVGLoader.Failure.xml) { try SVGLoader.load("<svg><g></svg>") }
        #expect(throws: SVGLoader.Failure.transform) { try SVGLoader.load("<svg><g transform='matrix(1 0)'/></svg>") }
        #expect(throws: SVGLoader.Failure.pathData(12)) { try one("M0 0 A4 4 0 2 0 5 6") }

        let curves = try one("M0 0 " + String(repeating: "C0 100000 100000 0 0 0 ", count: 10))
        #expect(throws: SVGLoader.Failure.size) { try curves.filledShapes(tolerance: 0.0001) }
        let star = (0..<257).map { i in
            let a = Double((i * 127) % 257) / 257 * 2 * .pi
            return "\(cos(a)),\(sin(a))"
        }.joined(separator: " ")
        let crossing = try #require(try SVGLoader.load("<svg><polygon points='\(star)'/></svg>").elements.first)
        #expect(throws: SVGLoader.Failure.size) { try crossing.filledShapes() }
    }

    private func one(_ path: String, style: String = "") throws -> SVGLoader.Element {
        let doc = try SVGLoader.load("<svg><path style='\(style)' d='\(path)'/></svg>")
        return try #require(doc.elements.first)
    }

    static func records(_ path: CGPath) -> [(CGPathElementType, [CGPoint])] {
        var records: [(CGPathElementType, [CGPoint])] = []
        path.applyWithBlock {
            let element = $0.pointee
            let count: Int
            switch element.type {
            case .moveToPoint, .addLineToPoint: count = 1
            case .addQuadCurveToPoint: count = 2
            case .addCurveToPoint: count = 3
            case .closeSubpath: count = 0
            @unknown default: count = 0
            }
            records.append((element.type, Array(UnsafeBufferPointer(start: element.points, count: count))))
        }
        return records
    }
}
