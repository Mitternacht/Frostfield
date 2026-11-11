import CoreGraphics
import Foundation
import Frostfield
import Metal
import Testing

struct SVGStrokeTests {
    @Test("SVG stroke caps outline local width before a nonuniform transform")
    func caps() throws {
        let doc = try SVGLoader.load("""
            <svg>
              <g stroke="currentColor" color="orange" stroke-width="2" stroke-opacity=".8" opacity=".5">
                <path d="M0 0H4" stroke-linecap="butt"/>
                <path d="M0 0H4" stroke-linecap="round"/>
                <path d="M0 0H4" stroke-linecap="square"/>
                <path d="M0 0H4" stroke-linecap="round" transform="scale(3 .5)"/>
              </g>
            </svg>
            """)
        let strokes = try doc.elements.map { try #require($0.stroke) }
        #expect(strokes.map(\.cap) == [.butt, .round, .square, .round])
        #expect(strokes.allSatisfy { $0.width == 2 && $0.opacity == 0.4 })
        let areas = try doc.elements.map { element in
            try element.strokedShapes(tolerance: 0.001).reduce(0) { $0 + $1.area }
        }
        #expect(areas[0] == 8 && areas[2] == 12)
        #expect(abs(areas[1] - 8 - Double.pi) < 0.005)
        #expect(abs(areas[3] - (8 + Double.pi) * 1.5) < 0.008)
        let outline = try #require(try doc.elements[3].strokedShapes(tolerance: 0.001).first)
        #expect(outline.outer.map(\.x).min() == -3)
        #expect(outline.outer.map(\.x).max() == 15)
        #expect(outline.outer.map(\.y).min() == -0.5)
        #expect(outline.outer.map(\.y).max() == 0.5)
    }

    @Test("SVG joins and miter limits affect outline bounds")
    func joins() throws {
        let data = "M0 2L2 0L4 2"
        let miter = try element(data, style: "stroke:black;stroke-width:1;stroke-linejoin:miter")
        let bevel = try element(data, style: "stroke:black;stroke-width:1;stroke-linejoin:bevel")
        let round = try element(data, style: "stroke:black;stroke-width:1;stroke-linejoin:round")
        let limited = try element(data, style: "stroke:black;stroke-width:1;stroke-linejoin:miter;stroke-miterlimit:1")
        let top = try [miter, bevel, round, limited].map { element in
            let points = try element.strokedShapes(tolerance: 0.001).flatMap(\.outer)
            return try #require(points.map(\.y).min())
        }
        #expect(abs(top[0] + 0.7071068) < 0.00001)
        #expect(abs(top[1] + 0.3535534) < 0.00001)
        #expect(abs(top[2] + 0.5) < 0.0005)
        #expect(top[3] == top[1])
    }

    @Test("Closed SVG stroke preserves its transparent center in Metal")
    func render() throws {
        let shape = try element("M-1-1 H1 V1 H-1Z", style: "fill:none;stroke:#04ff20;stroke-width:.25;stroke-opacity:.75")
        #expect(try shape.fillGeometry() == nil)
        let style = try #require(shape.stroke)
        let regions = try shape.strokedShapes()
        #expect(regions.count == 1 && regions[0].holes.count == 1)
        let geometry = try #require(try shape.strokeGeometry())
        guard case .solid(let rgba) = style.paint else { Issue.record("Expected solid stroke"); return }
        let fixture = try PrimitiveRenderFixture()
        let renderer = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .classic(ClassicMaterial(
                model: .basic, color: SIMD3(rgba.x, rgba.y, rgba.z), opacity: style.opacity * rgba.w
            )),
            renderState: MeshRenderState(blending: .normal, transparent: true, depthWrite: false)
        )
        let node = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2)), mesh: renderer)
        let pixels = try fixture.render(fixture.graph([node])).pixels
        #expect(isNearly(pixels[32 * 64 + 49], SIMD4(0.000911, 0.75, 0.010833, 0.75)))
        #expect(pixels[32 * 64 + 32] == .zero)
        #expect(pixels[32 * 64 + 58] == .zero)
    }

    @Test("Stroke outlines merge shared edges after close-path restarts")
    func overlappingStrokes() throws {
        let path = "M100 100 h-50 v-50 h50 z h50 v-50 h-50 z h50 v50 h-50 z h-50 v50 h50 z"
        let shape = try element(path, style: "stroke:orange;stroke-width:4;fill:none")
        let regions = try shape.strokedShapes()
        #expect(regions.count == 1 && regions[0].holes.count == 4)
        // A 104 × 104 outer boundary surrounds four 46 × 46 interiors.
        #expect(regions.reduce(0) { $0 + $1.area } == 2_352)
    }

    @Test("Empty strokes and unsupported joins are explicit")
    func emptyAndUnsupported() throws {
        let invisible = try element("M0 0H5", style: "stroke:red;stroke-width:0")
        let degenerate = try element("M0 0H0", style: "stroke:red;stroke-linecap:butt")
        let arcs = try element("M0 0H5V5", style: "stroke:red;stroke-linejoin:arcs")
        #expect(try invisible.strokeGeometry() == nil)
        #expect(try degenerate.strokeGeometry() == nil)
        #expect(try degenerate.strokeGeometry(textureCoordinates: .path) == nil)
        #expect(throws: SVGLoader.Failure.strokeJoin("arcs")) { try arcs.strokeGeometry() }
        #expect(throws: SVGLoader.Failure.strokeJoin("arcs")) {
            try arcs.strokeGeometry(textureCoordinates: .path)
        }
    }

    @Test("Clipped miters match the pinned extent and cover both turn directions")
    func clippedMiters() throws {
        let paint = "fill:none;stroke:black;stroke-width:1;stroke-linejoin:miter-clip;stroke-miterlimit:"
        let clipped = try element("M0 2 L2 0 L4 2", style: paint + "1")
        let full = try element("M0 2 L2 0 L4 2", style: paint + "4")
        let reflected = try element("M0 -2 L2 0 L4 -2", style: paint + "1")
        #expect(clipped.stroke?.join == .miterClip)
        let clipOutline = try #require(try clipped.strokedShapes().first)
        let fullOutline = try #require(try full.strokedShapes().first)
        let reflectedOutline = try #require(try reflected.strokedShapes(yUp: true).first)
        // Three.js r187 pointsToStroke, width 1 and miter limits 1 and 4.
        #expect(abs((clipOutline.outer.map(\.y).min() ?? 0) + 0.603553414) < 0.000001)
        #expect(abs((fullOutline.outer.map(\.y).min() ?? 0) + 0.707106769) < 0.000001)
        #expect(abs(clipOutline.area - 5.6461304534) < 0.000002)
        #expect(abs(fullOutline.area - 5.6568538125) < 0.000002)
        #expect(abs(reflectedOutline.area - clipOutline.area) < 0.000002)

        let mesh = try #require(try clipped.strokeGeometry(textureCoordinates: .path))
        let tips = mesh.vertices.filter { $0.position.y < -0.6 }
        #expect(Set(tips.map(\.textureCoordinate)) == [SIMD2(0.5, 0)])
        #expect(Set(tips.map(\.position)).count == 2)
    }

    @Test("Path-relative UVs survive curves, closed seams and independent subpaths")
    func pathCoordinates() throws {
        let shape = try element(
            "M0 0 Q1 2 2 0 M3 0 L5 0 L5 2 L3 2 Z M8 0 H9",
            style: "fill:none;stroke:red;stroke-width:.5;stroke-linejoin:miter"
        )
        let mesh = try #require(try shape.strokeGeometry(textureCoordinates: .path, tolerance: 0.02))
        let curved = mesh.vertices.filter { $0.position.x < 2.5 }
        #expect(curved.contains { $0.textureCoordinate.x > 0 && $0.textureCoordinate.x < 1 && $0.position.y > 1 })
        let seam = mesh.vertices.filter { $0.position.x == 2.75 && $0.position.y == -0.25 }
        #expect(Set(seam.map(\.textureCoordinate)) == [SIMD2(0, 0), SIMD2(1, 0)])
        let last = mesh.vertices.filter { $0.position.x >= 8 }
        #expect(Set(last.map(\.textureCoordinate)) == [SIMD2(0, 0), SIMD2(0, 1), SIMD2(1, 0), SIMD2(1, 1)])
        #expect(throws: PlanarShape.Failure.invalidTolerance) {
            try shape.strokeGeometry(textureCoordinates: .path, tolerance: .nan)
        }
    }

    @Test("Metal samples path UVs along an unequal-length clipped stroke")
    func texturedStroke() throws {
        let shape = try element(
            "M0 0 H4 V2", style: "fill:none;stroke:black;stroke-width:1;stroke-linejoin:miter-clip;stroke-miterlimit:1"
        )
        let geometry = try #require(try shape.strokeGeometry(textureCoordinates: .path, yUp: true))
        let fixture = try PrimitiveRenderFixture()
        let resolution = 64
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: resolution, height: resolution, mipmapped: false
        )
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        let texture = try #require(fixture.device.makeTexture(descriptor: descriptor))
        var texels: [SIMD4<Float16>] = []
        for row in 0..<resolution {
            for column in 0..<resolution {
                texels.append(SIMD4(
                    Float16((Float(column) + 0.5) / Float(resolution)),
                    Float16((Float(row) + 0.5) / Float(resolution)), 0, 1
                ))
            }
        }
        texels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, resolution, resolution), mipmapLevel: 0,
                withBytes: bytes.baseAddress!, bytesPerRow: resolution * MemoryLayout<SIMD4<Float16>>.stride
            )
        }
        let renderer = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .textured(texture)
        )
        let node = SceneNode(transform: SceneTransform(translation: SIMD3(-1.9375, 0.9375, -4)), mesh: renderer)
        let pixels = try fixture.render(fixture.graph([node])).pixels
        // Barycentrically sampled r187 UVs at authored points (1, -.25), (2, .25), (4.25, 1).
        #expect(isNearly(pixels[22 * 64 + 24], SIMD4(0.125, 0.25, 0, 1)))
        #expect(isNearly(pixels[26 * 64 + 32], SIMD4(0.26785714, 0.75, 0, 1)))
        #expect(isNearly(pixels[32 * 64 + 50], SIMD4(0.71875, 0.25, 0, 1)))
        #expect(pixels[16 * 64 + 24] == .zero)
    }

    private func element(_ path: String, style: String) throws -> SVGLoader.Element {
        let document = try SVGLoader.load("<svg><path style='\(style)' d='\(path)'/></svg>")
        return try #require(document.elements.first)
    }
}
