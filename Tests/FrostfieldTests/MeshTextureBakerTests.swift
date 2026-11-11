import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MeshTextureBakerTests {
    private func pixel(_ texture: any MTLTexture, _ x: Int, _ y: Int) -> SIMD4<Float> {
        var result = SIMD4<Float16>.zero
        texture.getBytes(&result, bytesPerRow: MemoryLayout<SIMD4<Float16>>.stride,
                         from: MTLRegionMake2D(x, y, 1, 1), mipmapLevel: 0)
        return SIMD4<Float>(result)
    }

    private func coverage(_ texture: any MTLTexture, _ x: Int, _ y: Int) -> UInt8 {
        var result: UInt8 = 0
        texture.getBytes(&result, bytesPerRow: 1, from: MTLRegionMake2D(x, y, 1, 1), mipmapLevel: 0)
        return result
    }

    @Test("UV0 shared edges have one owner and preserve straight vertex RGBA")
    func sharedEdges() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let points: [SIMD2<Float>] = [SIMD2(0, 0), SIMD2(1, 0), SIMD2(1, 1), SIMD2(0, 1)]
        let geometry = try MeshGeometry(vertices: points.map { uv in
            MeshVertex(
                position: SIMD3(uv.x, uv.y, 0), normal: SIMD3(0, 0, 1), textureCoordinate: uv,
                color: SIMD4(uv.x * 2, uv.y * -0.5, 0.375, uv.y * 0.5 + 0.25)
            )
        }, indices: [0, 1, 2, 0, 3, 2])
        let baked = try MeshTextureBaker(device: device, geometry: geometry, width: 8, height: 8)
        #expect(baked.vertexColorTexture.pixelFormat == .rgba16Float)
        #expect(baked.normalTexture.pixelFormat == .rgba16Float)
        #expect(baked.coverageTexture.pixelFormat == .r8Unorm)
        for y in 0..<8 {
            for x in 0..<8 {
                let u = (Float(x) + 0.5) / 8
                let v = (Float(y) + 0.5) / 8
                let expected = SIMD4(u * 2, v * -0.5, 0.375, v * 0.5 + 0.25)
                #expect(pixel(baked.vertexColorTexture, x, y) == expected)
                #expect(pixel(baked.normalTexture, x, y) == SIMD4(0.5, 0.5, 1, 1))
                #expect(coverage(baked.coverageTexture, x, y) == 255)
            }
        }
    }

    @Test("Object and world normals interpolate in the selected space")
    func normalSpaces() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let uv: [SIMD2<Float>] = [.zero, SIMD2(1, 0), SIMD2(0, 1)]
        let normals: [SIMD3<Float>] = [SIMD3(1, 1, 1), SIMD3(-1, 2, 1), SIMD3(0, 1, 2)]
        let geometry = try MeshGeometry(vertices: uv.indices.map { index in
            MeshVertex(position: SIMD3(uv[index].x, uv[index].y, 0), normal: normals[index], textureCoordinate: uv[index])
        }, indices: [0, 1, 2])
        let transform = SceneTransform(
            translation: SIMD3(15, -8, 4),
            rotation: simd_quatf(angle: 0.4, axis: SIMD3(0, 0, 1)),
            scale: SIMD3(-2, 0.5, 4)
        )
        for space in [MeshTextureBaker.NormalSpace.object, .world(worldFromObject: transform.matrix)] {
            let mapped = normals.map { normal in
                switch space {
                case .object: simd_normalize(normal)
                case .world: simd_normalize(transform.rotation.act(normal / transform.scale))
                }
            }
            let direction = simd_normalize(mapped[0] * 0.75 + mapped[1] * 0.125 + mapped[2] * 0.125)
            let expected = SIMD4(direction * 0.5 + 0.5, 1)
            let baked = try MeshTextureBaker(device: device, geometry: geometry, width: 4, height: 4, normalSpace: space)
            #expect(simd_reduce_max(abs(pixel(baked.normalTexture, 0, 0) - expected)) < 0.0003)
            #expect(pixel(baked.vertexColorTexture, 0, 0) == SIMD4(repeating: 1))
            #expect(coverage(baked.coverageTexture, 3, 3) == 0)
        }
    }

    private func atlasSeeds(first: Int, second: Int) throws -> MeshGeometry {
        let red = SIMD4<Float>(1, 0, 0, 0)
        let blue = SIMD4<Float>(0, 0, 1, 0.75)
        let offsets: [SIMD2<Float>] = [SIMD2(0.25, 0.25), SIMD2(0.875, 0.25), SIMD2(0.25, 0.875)]
        let vertices = [(first, red, SIMD3<Float>(0, 0, 1)), (second, blue, SIMD3<Float>(1, 0, 0))].flatMap { seed in
            offsets.map { offset in
                let uv = (SIMD2(Float(seed.0), 1) + offset) / SIMD2(8, 4)
                return MeshVertex(position: SIMD3(uv.x, uv.y, 0), normal: seed.2, textureCoordinate: uv, color: seed.1)
            }
        }
        return try MeshGeometry(vertices: vertices, indices: [0, 1, 2, 3, 4, 5])
    }

    @Test("Padding keeps interior and transparent samples and breaks ties in row-major order")
    func padding() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let atlas = try MeshTextureBaker(
            device: device, geometry: atlasSeeds(first: 2, second: 4), width: 8, height: 4, padding: 1
        )
        let red = SIMD4<Float>(1, 0, 0, 0)
        let blue = SIMD4<Float>(0, 0, 1, 0.75)
        #expect(coverage(atlas.coverageTexture, 2, 1) == 255)
        #expect(coverage(atlas.coverageTexture, 4, 1) == 255)
        #expect(pixel(atlas.vertexColorTexture, 2, 1) == red)
        #expect(pixel(atlas.vertexColorTexture, 4, 1) == blue)
        #expect(coverage(atlas.coverageTexture, 3, 1) == 128)
        #expect(pixel(atlas.vertexColorTexture, 3, 1) == red)
        #expect(pixel(atlas.normalTexture, 3, 1) == SIMD4(0.5, 0.5, 1, 1))
        #expect(pixel(atlas.vertexColorTexture, 5, 1) == blue)
        #expect(pixel(atlas.normalTexture, 5, 1) == SIMD4(1, 0.5, 0.5, 1))
        #expect(coverage(atlas.coverageTexture, 1, 2) == 0)
        #expect(pixel(atlas.vertexColorTexture, 1, 2) == .zero)
        #expect(pixel(atlas.normalTexture, 1, 2) == .zero)
    }

    @Test("Overlaps reject with a texel coordinate or follow the selected triangle order")
    func overlap() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let geometry = try atlasSeeds(first: 2, second: 2)
        #expect(throws: MeshTextureBaker.Failure.overlap(triangle: 1, x: 2, y: 1)) {
            try MeshTextureBaker(device: device, geometry: geometry, width: 8, height: 4)
        }
        let first = try MeshTextureBaker(device: device, geometry: geometry, width: 8, height: 4, overlap: .firstTriangle)
        let last = try MeshTextureBaker(device: device, geometry: geometry, width: 8, height: 4, overlap: .lastTriangle)
        #expect(pixel(first.vertexColorTexture, 2, 1) == SIMD4(1, 0, 0, 0))
        #expect(pixel(last.vertexColorTexture, 2, 1) == SIMD4(0, 0, 1, 0.75))
        #expect(pixel(first.normalTexture, 2, 1) == SIMD4(0.5, 0.5, 1, 1))
        #expect(pixel(last.normalTexture, 2, 1) == SIMD4(1, 0.5, 0.5, 1))
    }

    @Test("A UV1-baked base color renders like the authored vertex colors")
    func materialWorkflow() throws {
        let fixture = try GeometryRenderFixture()
        let plane = try MeshGeometry.plane(width: 2, height: 2)
        let vertices = plane.vertices.map { vertex in
            let uv = vertex.textureCoordinate
            let color = SIMD4(0.3 + 0.6 * uv.x, 0.15 + 0.5 * uv.y, 0.75 - 0.45 * uv.x, 0.5 + 0.5 * uv.y)
            return MeshVertex(
                position: vertex.position, normal: vertex.normal, textureCoordinate: uv,
                secondaryTextureCoordinate: SIMD2(1 - uv.y, uv.x), color: color
            )
        }
        let colored = try MeshGeometry(vertices: vertices, indices: plane.indices)
        let maps = try MeshTextureBaker(
            device: fixture.device, geometry: colored, width: 32, height: 16, textureCoordinateSet: .secondary
        )
        let textured = try MeshGeometry(vertices: vertices.map { vertex in
            MeshVertex(position: vertex.position, normal: vertex.normal, textureCoordinate: vertex.textureCoordinate,
                       secondaryTextureCoordinate: vertex.secondaryTextureCoordinate)
        }, indices: plane.indices)
        func draw(_ geometry: MeshGeometry, using material: PhysicallyBasedMaterial) throws -> [SIMD4<Float>] {
            let mesh = try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry,
                material: .physicallyBased(material), renderState: .transparent
            )
            return try fixture.render(SceneGraph(
                roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)],
                camera: fixture.camera
            )).pixels
        }
        let expected = try draw(colored, using: PhysicallyBasedMaterial(alphaMode: .blend, shadingModel: .unlit))
        let actual = try draw(textured, using: PhysicallyBasedMaterial(
            baseColorTexture: maps.vertexColorTexture, alphaMode: .blend,
            baseColorTextureCoordinateSet: maps.textureCoordinateSet, shadingModel: .unlit
        ))
        for y in 18..<46 {
            for x in 18..<46 {
                let a = fixture.pixel(actual, x: x, y: y)
                let e = fixture.pixel(expected, x: x, y: y)
                #expect(simd_reduce_max(abs(a - e)) < 0.0015, "pixel (\(x), \(y)): \(a) vs \(e)")
            }
        }
        #expect(abs(fixture.pixel(actual, x: 20, y: 32).x - fixture.pixel(actual, x: 43, y: 32).x) > 0.2)
    }

    @Test("Bakes validate input UVs, transforms, sizes and sample coverage")
    func validation() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let mesh = try MeshGeometry.plane()
        #expect(throws: MeshTextureBaker.Failure.dimensions) {
            try MeshTextureBaker(device: device, geometry: mesh, width: 0, height: 8)
        }
        #expect(throws: MeshTextureBaker.Failure.padding) {
            try MeshTextureBaker(device: device, geometry: mesh, width: 8, height: 8, padding: 65)
        }
        #expect(throws: MeshTextureBaker.Failure.missingUV1(vertex: 0)) {
            try MeshTextureBaker(device: device, geometry: mesh, width: 8, height: 8, textureCoordinateSet: .secondary)
        }
        #expect(throws: MeshTextureBaker.Failure.normalTransform) {
            try MeshTextureBaker(
                device: device, geometry: mesh, width: 8, height: 8,
                normalSpace: .world(worldFromObject: SceneTransform(scale: SIMD3(0, 1, 1)).matrix)
            )
        }
        let outside = try MeshGeometry(vertices: mesh.vertices.map {
            MeshVertex(position: $0.position, normal: $0.normal, textureCoordinate: $0.textureCoordinate + 1.5)
        }, indices: mesh.indices)
        #expect(throws: MeshTextureBaker.Failure.invalidUV(vertex: 0)) {
            try MeshTextureBaker(device: device, geometry: outside, width: 8, height: 8)
        }
        let tiny = try MeshGeometry(vertices: mesh.vertices.map {
            MeshVertex(position: $0.position, normal: $0.normal, textureCoordinate: $0.textureCoordinate * 0.01)
        }, indices: mesh.indices)
        #expect(throws: MeshTextureBaker.Failure.emptyAtlas) {
            try MeshTextureBaker(device: device, geometry: tiny, width: 8, height: 8)
        }
        let overflow = try MeshGeometry(vertices: mesh.vertices.map {
            MeshVertex(position: $0.position, normal: $0.normal, textureCoordinate: $0.textureCoordinate, color: SIMD4(repeating: 1e8))
        }, indices: mesh.indices)
        #expect(throws: MeshTextureBaker.Failure.colorRange(vertex: 0)) {
            try MeshTextureBaker(device: device, geometry: overflow, width: 8, height: 8)
        }
    }
}
