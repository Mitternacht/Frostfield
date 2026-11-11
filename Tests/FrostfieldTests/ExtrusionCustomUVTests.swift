import CoreText
import Foundation
import Frostfield
import Metal
import simd
import Testing

struct ExtrusionCustomUVTests {
    @Test("Beveled cap and wall callbacks remap UV0 and regenerate V-down tangents")
    func beveledMapping() throws {
        let shape = try PlanarShape(outer: square(0.75), holes: [square(0.25)])
        let options = ShapeExtrusion(depth: 1, bevelSize: 0.06, bevelSegments: 3, steps: 3)
        let baseline = try MeshGeometry.extrude([shape], options)
        let mapped = try MeshGeometry.extrude([shape], options, uvGenerator: .init(
            cap: { cap in
                let offset: Float = cap.side == .front ? 2 : 4
                func transform(_ vertex: MeshVertex) -> SIMD2<Float> {
                    SIMD2(offset + vertex.textureCoordinate.y, vertex.textureCoordinate.x)
                }
                return (transform(cap.corners.0), transform(cap.corners.1), transform(cap.corners.2))
            },
            wall: { wall in
                let offset = Float(wall.ringIndex) * 8
                func transform(_ vertex: MeshVertex) -> SIMD2<Float> {
                    SIMD2(offset + 1 - vertex.textureCoordinate.x, vertex.textureCoordinate.y)
                }
                return (
                    transform(wall.corners.0), transform(wall.corners.1),
                    transform(wall.corners.2), transform(wall.corners.3)
                )
            }
        ))
        #expect(sameCorners(mapped, baseline))
        #expect(mapped.groups.isEmpty)
        #expect(ShapeCheck.isClosed(mapped))
        #expect(ShapeCheck.tangentsFollowTextureCoordinates(mapped))
        let front = mapped.vertices.filter { (2...3).contains($0.textureCoordinate.x) }
        let back = mapped.vertices.filter { (4...5).contains($0.textureCoordinate.x) }
        #expect(!front.isEmpty && !back.isEmpty)
        #expect(front.allSatisfy {
            $0.normal == SIMD3(0, 0, 1) && simd_distance($0.tangent ?? .zero, SIMD4(0, -1, 0, 1)) < 0.0001
        })
        #expect(back.allSatisfy {
            $0.normal == SIMD3(0, 0, -1) && simd_distance($0.tangent ?? .zero, SIMD4(0, -1, 0, -1)) < 0.0001
        })
        #expect(mapped.vertices.contains { (8...9).contains($0.textureCoordinate.x) && abs($0.position.z) < 0.2 })

        let unchanged = try MeshGeometry.extrude([shape], options, uvGenerator: .init(
            cap: { ($0.corners.0.textureCoordinate, $0.corners.1.textureCoordinate, $0.corners.2.textureCoordinate) }
        ))
        #expect(unchanged.indices == baseline.indices)
        #expect(unchanged.vertices.map(\.textureCoordinate) == baseline.vertices.map(\.textureCoordinate))
        #expect(unchanged.vertices.map(\.tangent) == baseline.vertices.map(\.tangent))
    }

    @Test("Curve wall quads can use distinct islands without changing indexed group ranges")
    func pathSeams() throws {
        let path = try Curve3D.cubicBezier(
            from: SIMD3(0, 0, 0), control1: SIMD3(0, 0, 1), control2: SIMD3(0.25, 0.1, 2), to: SIMD3(0.6, 0.4, 3)
        )
        let shape = try PlanarShape(outer: square(0.2))
        let steps = 4
        let baseline = try MeshGeometry.extrude([shape], along: path, steps: steps, creaseAngle: .pi)
        let mapped = try MeshGeometry.extrude([shape], along: path, steps: steps, creaseAngle: .pi, uvGenerator: .init(
            wall: { wall in
                let offset = SIMD2<Float>(Float(wall.edgeIndex) * 2, Float(wall.levelIndex) * 2)
                return (
                    wall.corners.0.textureCoordinate + offset, wall.corners.1.textureCoordinate + offset,
                    wall.corners.2.textureCoordinate + offset, wall.corners.3.textureCoordinate + offset
                )
            }
        ))
        #expect(sameCorners(mapped, baseline))
        #expect(mapped.groups == baseline.groups)
        #expect(mapped.vertices.count > baseline.vertices.count)
        #expect(ShapeCheck.isClosed(mapped))
        #expect(ShapeCheck.tangentsFollowTextureCoordinates(mapped))
        let wall = try mapped.subset(mapped.groups[1].range)
        let coordinatesByPosition = Dictionary(grouping: wall.vertices, by: \.position)
        #expect(coordinatesByPosition.values.contains { Set($0.map(\.textureCoordinate)).count == 4 })
        let capRange = mapped.groups[0].range
        #expect(capRange.allSatisfy {
            mapped.vertices[Int(mapped.indices[$0])].textureCoordinate == baseline.vertices[Int(baseline.indices[$0])].textureCoordinate
        })
    }

    @Test("Nonfinite UVs and callback errors fail extrusion before returning a mesh")
    func validation() throws {
        enum Rejected: Error, Equatable { case authored }
        let shape = try PlanarShape(outer: square(0.5))
        let path = Curve3D { SIMD3(0, 0, $0) }
        #expect(throws: MeshGeometry.ShapeMeshError.invalidGeneratedUV(0)) {
            try MeshGeometry.extrude([shape], .init(depth: 1), uvGenerator: .init(
                cap: { _ in (SIMD2(.nan, 0), .zero, .zero) }
            ))
        }
        #expect(throws: MeshGeometry.ShapeMeshError.invalidGeneratedUV(0)) {
            try MeshGeometry.extrude([shape], along: path, uvGenerator: .init(
                wall: { _ in (.zero, .zero, .zero, SIMD2(0, .infinity)) }
            ))
        }
        #expect(throws: Rejected.authored) {
            try MeshGeometry.extrude([shape], along: path, uvGenerator: .init(cap: { _ in throw Rejected.authored }))
        }
    }

    @Test("Extruded text forwards the UV generator to its cap and wall emitters")
    func textForwarding() throws {
        let font = CTFontCreateWithName("Helvetica" as CFString, 72, nil)
        let options = ShapeExtrusion(depth: 0.25, steps: 2)
        let offset = SIMD2<Float>(2, 4)
        let custom = ShapeExtrusionUVGenerator(
            cap: {
                ($0.corners.0.textureCoordinate + offset, $0.corners.1.textureCoordinate + offset,
                 $0.corners.2.textureCoordinate + offset)
            },
            wall: {
                ($0.corners.0.textureCoordinate + offset, $0.corners.1.textureCoordinate + offset,
                 $0.corners.2.textureCoordinate + offset, $0.corners.3.textureCoordinate + offset)
            }
        )
        let baseline = try MeshGeometry.text("A", font: font, extrusion: options)
        let mesh = try MeshGeometry.text("A", font: font, extrusion: options, uvGenerator: custom)
        #expect(sameCorners(mesh, baseline))
        #expect(zip(mesh.indices, baseline.indices).allSatisfy {
            mesh.vertices[Int($0)].textureCoordinate == baseline.vertices[Int($1)].textureCoordinate + offset
        })
        #expect(throws: MeshGeometry.ShapeMeshError.uvGeneratorNeedsExtrusion) {
            try MeshGeometry.text("A", font: font, uvGenerator: custom)
        }
    }

    @Test("A Metal cap draw samples the remapped UV0",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func sampledCap() throws {
        let gpu = try GeometryRenderFixture()
        let shape = try PlanarShape(outer: square(0.75))
        let options = ShapeExtrusion(depth: 0.5)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 2, height: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let map = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let bytes: [UInt8] = [255, 0, 0, 255, 0, 255, 0, 255]
        map.replace(region: MTLRegionMake2D(0, 0, 2, 1), mipmapLevel: 0, withBytes: bytes, bytesPerRow: 8)
        let sampling = PhysicallyBasedMaterial.TextureSampling(
            addressU: .clampToEdge, addressV: .clampToEdge, magnification: .nearest, minification: .nearest
        )
        let material = try ClassicMaterial(model: .basic, map: .init(
            texture: map, sampler: sampling.makeSamplerState(device: gpu.device), sampling: sampling
        ))
        func draw(_ custom: ShapeExtrusionUVGenerator?) throws -> [SIMD4<Float>] {
            let geometry = try MeshGeometry.extrude([shape], options, uvGenerator: custom)
            let renderer = try gpu.unlit(geometry, material: .classic(material))
            let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: renderer)
            return try gpu.render(SceneGraph(roots: [node], camera: gpu.camera)).pixels
        }
        let baseline = try draw(nil)
        let mirrored = try draw(.init(cap: { face in
            func mirror(_ vertex: MeshVertex) -> SIMD2<Float> {
                SIMD2(1 - vertex.textureCoordinate.x, vertex.textureCoordinate.y)
            }
            return (mirror(face.corners.0), mirror(face.corners.1), mirror(face.corners.2))
        }))
        #expect(gpu.pixel(baseline, x: 24, y: 32) == SIMD4(1, 0, 0, 1))
        #expect(gpu.pixel(baseline, x: 40, y: 32) == SIMD4(0, 1, 0, 1))
        #expect(gpu.pixel(mirrored, x: 24, y: 32) == SIMD4(0, 1, 0, 1))
        #expect(gpu.pixel(mirrored, x: 40, y: 32) == SIMD4(1, 0, 0, 1))
    }

    private func square(_ halfExtent: Float) -> [SIMD2<Float>] {
        let h = halfExtent
        return [SIMD2(-h, -h), SIMD2(h, -h), SIMD2(h, h), SIMD2(-h, h)]
    }

    private func sameCorners(_ lhs: MeshGeometry, _ rhs: MeshGeometry) -> Bool {
        lhs.indices.count == rhs.indices.count && zip(lhs.indices, rhs.indices).allSatisfy {
            let a = lhs.vertices[Int($0)], b = rhs.vertices[Int($1)]
            return a.position == b.position && a.normal == b.normal
        }
    }
}
