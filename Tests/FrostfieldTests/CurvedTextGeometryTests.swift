import CoreText
import Foundation
import Frostfield
import Metal
import Testing

struct CurvedTextGeometryTests {
    private let font = CTFontCreateWithName("Helvetica" as CFString, 72, nil)

    @Test("A Core Text glyph sweep renders textured caps and walls with an open counter",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func curvedGlyphs() throws {
        try #require(CTFontCopyPostScriptName(font) as String == "Helvetica")
        let string = "O O"
        let size: Float = 1.5
        let tolerance: Float = 0.003
        let outline = try TextOutline(string, font: font, size: size, alignment: .center, curveTolerance: tolerance)
        try #require(outline.glyphs.count == 3 && outline.shapes.count == 2)
        #expect(outline.glyphs.map(\.characterIndex) == [0, 1, 2])
        #expect(outline.glyphs[1].shapes.isEmpty)
        #expect(outline.shapes.allSatisfy { $0.holes.count == 1 })
        let firstCounter = try #require(outline.shapes[0].holes.first)
        let counterCenter = firstCounter.reduce(SIMD2<Float>.zero, +) / Float(firstCounter.count)

        // The endpoints have +Z tangents. The transported profile axes are
        // -Y and +X there, so the first near counter is centered on the view ray.
        let end = SIMD3<Float>(-counterCenter.y, counterCenter.x, -2)
        let start = end + SIMD3(-0.12, 0, -1.1)
        let axisOffset = SIMD3<Float>(0, 0, 0.37)
        let path = try Curve3D.cubicBezier(from: start, control1: start + axisOffset, control2: end - axisOffset, to: end)
        let steps = 6
        let subdivisions = 128
        let crease: Float = .pi / 3
        let mapping = ShapeTextureMapping.world(unitsPerTile: 4)
        let ordinary = try MeshGeometry.extrude(
            outline.shapes, along: path, steps: steps, arcLengthSubdivisions: subdivisions,
            creaseAngle: crease, textureMapping: mapping
        )
        let custom = ShapeExtrusionUVGenerator(
            cap: { face in
                (Self.capUV(face.corners.0.textureCoordinate), Self.capUV(face.corners.1.textureCoordinate),
                 Self.capUV(face.corners.2.textureCoordinate))
            },
            wall: { face in
                (Self.wallUV(face.corners.0.textureCoordinate), Self.wallUV(face.corners.1.textureCoordinate),
                 Self.wallUV(face.corners.2.textureCoordinate), Self.wallUV(face.corners.3.textureCoordinate))
            }
        )
        let mesh = try MeshGeometry.text(
            string, font: font, size: size, along: path, steps: steps, arcLengthSubdivisions: subdivisions,
            creaseAngle: crease, textureMapping: mapping, alignment: .center, curveTolerance: tolerance, uvGenerator: custom
        )
        try #require(mesh.groups == ordinary.groups && mesh.indices.count == ordinary.indices.count)
        #expect(mesh.groups.map(\.materialIndex) == [0, 1, 0, 1])
        #expect(ShapeCheck.isClosed(mesh))
        #expect(mesh.signedVolume > 0)
        #expect(ShapeCheck.tangentsFollowTextureCoordinates(mesh, minimumArea: 1e-7))
        for (shapeIndex, shape) in outline.shapes.enumerated() {
            let capGroup = mesh.groups[shapeIndex * 2], wallGroup = mesh.groups[shapeIndex * 2 + 1]
            let ringEdges = shape.outer.count + shape.holes.reduce(0) { $0 + $1.count }
            #expect(wallGroup.count == ringEdges * steps * 6)
            let capVertices = capGroup.range.map { mesh.vertices[Int(mesh.indices[$0])] }
            #expect(Set(capVertices.map(\.position.z)) == [start.z, end.z])
            #expect(capVertices.allSatisfy { $0.normal == SIMD3(0, 0, $0.position.z == end.z ? 1 : -1) })
        }
        for group in mesh.groups {
            #expect(group.range.allSatisfy { offset in
                let a = mesh.vertices[Int(mesh.indices[offset])], b = ordinary.vertices[Int(ordinary.indices[offset])]
                let expected = group.materialIndex == 0 ? Self.capUV(b.textureCoordinate) : Self.wallUV(b.textureCoordinate)
                return a.position == b.position && a.normal == b.normal && a.textureCoordinate == expected
            })
        }

        let gpu = try GeometryRenderFixture(side: 160)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 2, height: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let bytes: [UInt8] = [255, 0, 0, 255, 0, 255, 0, 255]
        texture.replace(region: MTLRegionMake2D(0, 0, 2, 1), mipmapLevel: 0, withBytes: bytes, bytesPerRow: 8)
        let sampling = PhysicallyBasedMaterial.TextureSampling(
            addressU: .clampToEdge, addressV: .clampToEdge, magnification: .nearest, minification: .nearest
        )
        let cap = try ClassicMaterial(model: .basic, map: .init(
            texture: texture, sampler: sampling.makeSamplerState(device: gpu.device), sampling: sampling
        ))
        let wallColor = SIMD3<Float>(0, 0, 1)
        let node = try SceneNode.grouped(
            mesh, materials: [.classic(cap), .classic(ClassicMaterial(model: .basic, color: wallColor))],
            device: gpu.device, colorPixelFormat: .rgba16Float
        )
        let rendered = try gpu.render(SceneGraph(roots: [node], camera: gpu.camera))
        #expect(rendered.statistics.drawnMeshCount == 4)
        #expect(rendered.pixels.filter { $0 == SIMD4(0, 1, 0, 1) }.count > 200)
        #expect(rendered.pixels.filter { $0 == SIMD4(wallColor, 1) }.count > 20)
        #expect(!rendered.pixels.contains(SIMD4(1, 0, 0, 1)))
        let center = gpu.side / 2
        for y in (center - 3)...(center + 3) {
            for x in (center - 3)...(center + 3) {
                #expect(gpu.pixel(rendered.pixels, x: x, y: y) == .zero)
            }
        }
    }

    @Test("Curved text forwards outline, path and texture validation")
    func invalidOptions() {
        let path = Curve3D { SIMD3(0, 0, $0) }
        #expect(throws: MeshGeometry.ShapeMeshError.noShapes) {
            try MeshGeometry.text(" ", font: font, along: path)
        }
        #expect(throws: MeshGeometry.ShapeMeshError.invalidSteps) {
            try MeshGeometry.text("O", font: font, along: path, steps: 0)
        }
        #expect(throws: MeshGeometry.ShapeMeshError.invalidTextureScale) {
            try MeshGeometry.text("O", font: font, along: path, textureMapping: .world(unitsPerTile: 0))
        }
    }

    private static func capUV(_ uv: SIMD2<Float>) -> SIMD2<Float> { SIMD2(0.75 + 0.25 * uv.x, uv.y) }
    private static func wallUV(_ uv: SIMD2<Float>) -> SIMD2<Float> { uv + SIMD2(0, 2) }
}
