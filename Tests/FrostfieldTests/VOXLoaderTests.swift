import Foundation
import Frostfield
import Testing
import simd

struct VOXLoaderTests {
    @Test("VOX packs keep source voxel order and apply the final palette to every model")
    func packedModels() throws {
        let palette = Self.palette([SIMD4(255, 128, 0, 96), SIMD4(0, 0, 255, 255)])
        let data = Self.file(
            Self.chunk("PACK", Self.integer(2)) +
            Self.model(SIMD3(2, 1, 1), [SIMD4(1, 0, 0, 1), SIMD4(0, 0, 0, 1)]) +
            Self.chunk("JUNK", Data([1, 2]), children: Self.chunk("MORE", Data([3]))) +
            Self.model(SIMD3(2, 1, 1), [SIMD4(0, 0, 0, 1), SIMD4(1, 0, 0, 2)]) + palette,
            version: 200
        )
        let asset = try VOXLoader.load(data)
        #expect(asset.version == 200)
        #expect(asset.scene == nil)
        #expect(asset.models.map(\.size) == [SIMD3(2, 1, 1), SIMD3(2, 1, 1)])
        #expect(asset.models[0].voxels.map(\.position) == [SIMD3(1, 0, 0), .zero])
        #expect(asset.palette.count == 256)
        #expect(asset.palette[0] == .zero)
        #expect(asset.palette[1] == SIMD4(255, 128, 0, 96))
        #expect(asset.palette[255] == .zero)

        let solid = try #require(try asset.makeMesh(modelIndex: 0))
        let split = try #require(try asset.makeMesh(modelIndex: 1))
        #expect(solid.vertices.count == 24)
        #expect(solid.indices.count == 36)
        #expect(split.vertices.count == 40)
        #expect(split.indices.count == 60)
        #expect(solid.vertices.allSatisfy { isNearly($0.color ?? .zero, SIMD4(1, 0.21586, 0, 1)) })
        #expect(split.vertices.contains { $0.color == SIMD4(0, 0, 1, 1) })
        #expect(solid.vertices.map { $0.position.x }.min() == -1)
        #expect(solid.vertices.map { $0.position.x }.max() == 1)
        for index in stride(from: 0, to: split.indices.count, by: 3) {
            let a = split.vertices[Int(split.indices[index])]
            let b = split.vertices[Int(split.indices[index + 1])]
            let c = split.vertices[Int(split.indices[index + 2])]
            #expect(simd_dot(simd_cross(b.position - a.position, c.position - a.position), a.normal) > 0)
        }
        #expect(throws: VOXLoader.Failure.invalidModelIndex(2)) { try asset.makeMesh(modelIndex: 2) }
    }

    @Test("Default palette includes the ordered color cube and ramps")
    func defaultPalette() throws {
        let asset = try VOXLoader.load(Self.file(Self.model(SIMD3(1, 1, 1), [])))
        #expect(asset.version == 150)
        #expect(try asset.makeMesh(modelIndex: 0) == nil)
        #expect(asset.palette.count == 256)
        #expect(asset.palette[1] == SIMD4(255, 255, 255, 255))
        #expect(asset.palette[2] == SIMD4(255, 255, 204, 255))
        #expect(asset.palette[215] == SIMD4(0, 0, 51, 255))
        #expect(asset.palette[216] == SIMD4(238, 0, 0, 255))
        #expect(asset.palette[226] == SIMD4(0, 238, 0, 255))
        #expect(asset.palette[236] == SIMD4(0, 0, 238, 255))
        #expect(asset.palette[255] == SIMD4(17, 17, 17, 255))
    }

    @Test("First-frame VOX transforms and instanced palette colors reach the native scene graph")
    func nativeScene() throws {
        let data = Self.file(
            Self.model(SIMD3(1, 1, 1), [SIMD4(0, 0, 0, 1)]) +
            Self.transform(0, child: 1, frames: [[("_t", "0 4 0")]]) +
            Self.chunk("nGRP", Self.integer(1) + Self.dictionary([]) + Self.integers([2, 2, 4])) +
            Self.transform(2, child: 3, name: "left", frames: [[("_t", "-1 0 0"), ("_r", "17")], [("_t", "8 0 0")]]) +
            Self.transform(4, child: 3, name: "right", frames: [[("_t", "1 0 0")]]) +
            Self.shape(3, models: [0]) + Self.palette([SIMD4(0, 128, 255, 255)])
        )
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("vox")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let asset = try VOXLoader.load(contentsOf: url)
        let scene = try #require(asset.scene)
        #expect(scene.transform.translation == SIMD3(0, 0, -4))
        let group = try #require(scene.children.first)
        #expect(group.children.map(\.name) == ["left", "right"])
        #expect(group.children.map(\.modelIndex) == [0, 0])
        #expect(group.children[0].transform.translation == SIMD3(-1, 0, 0))
        #expect(isNearly(group.children[0].transform.matrix * SIMD4(1, 0, 0, 0), SIMD4(0, 0, -1, 0)))

        let fixture = try PrimitiveRenderFixture()
        let native = try asset.makeSceneNode(
            device: fixture.device, colorPixelFormat: .rgba16Float, material: .classic(ClassicMaterial(model: .basic))
        )
        let children = native.children[0].children
        #expect(children[0].mesh != nil && children[0].mesh === children[1].mesh)
        let rendered = try fixture.render(fixture.graph([native]))
        #expect(rendered.statistics.drawnMeshCount == 2)
        let expected = SIMD4<Float>(0, 0.21586, 1, 1)
        #expect(isNearly(rendered.pixels[32 * 64 + 24], expected))
        #expect(isNearly(rendered.pixels[32 * 64 + 40], expected))
        #expect(rendered.pixels[32 * 64 + 32] == .zero)
    }

    @Test("VOX occupancy samples sparse source coordinates as R8Unorm")
    func occupancy() throws {
        let asset = try VOXLoader.load(Self.file(Self.model(
            SIMD3(2, 3, 2), [SIMD4(1, 0, 0, 1), SIMD4(0, 2, 1, 255)]
        )))
        let probe = try NRRDTextureProbe()
        let texture = try asset.models[0].makeOccupancyTexture(device: probe.device)
        #expect(texture.textureType == .type3D)
        #expect(texture.pixelFormat == .r8Unorm)
        #expect(SIMD3(texture.width, texture.height, texture.depth) == SIMD3(2, 3, 2))
        #expect(try probe.read(texture, at: SIMD3(1, 0, 0), kind: .float) == Float(1).bitPattern)
        #expect(try probe.read(texture, at: SIMD3(0, 2, 1), kind: .float) == Float(1).bitPattern)
        #expect(try probe.read(texture, at: SIMD3(0, 1, 1), kind: .float) == Float(0).bitPattern)
    }

    @Test("VOX rejects invalid chunk extents, dimensions, records and scene references")
    func malformed() throws {
        let model = Self.model(SIMD3(2, 1, 1), [SIMD4(1, 0, 0, 1)])
        #expect(throws: VOXLoader.Failure.version(151)) { try VOXLoader.load(Self.file(model, version: 151)) }
        #expect(throws: VOXLoader.Failure.chunk("MAIN")) { try VOXLoader.load(Self.file(model).dropLast()) }
        #expect(throws: VOXLoader.Failure.chunk("MAIN")) {
            try VOXLoader.load(Self.file(Self.chunk("JUNK", Data(), children: Data([0, 1])).dropLast()))
        }
        #expect(throws: VOXLoader.Failure.chunk("PACK")) {
            try VOXLoader.load(Self.file(Self.chunk("PACK", Self.integer(2)) + model))
        }
        #expect(throws: VOXLoader.Failure.chunk("SIZE")) {
            try VOXLoader.load(Self.file(Self.model(SIMD3(257, 1, 1), [])))
        }
        for voxels: [SIMD4<UInt8>] in [[SIMD4(2, 0, 0, 1)], [SIMD4(1, 0, 0, 0)]] {
            #expect(throws: VOXLoader.Failure.invalidVoxel(0)) {
                try VOXLoader.load(Self.file(Self.model(SIMD3(2, 1, 1), voxels)))
            }
        }
        #expect(throws: VOXLoader.Failure.invalidVoxel(1)) {
            try VOXLoader.load(Self.file(Self.model(SIMD3(2, 1, 1), [SIMD4(1, 0, 0, 1), SIMD4(1, 0, 0, 2)])))
        }
        for records in [
            Self.transform(0, child: 1, frames: []),
            Self.transform(0, child: 0, frames: []),
            Self.shape(0, models: [1]),
            Self.transform(0, child: 1, frames: [[("_r", "3")]]) + Self.shape(1, models: [0]),
        ] {
            #expect(throws: VOXLoader.Failure.scene) { try VOXLoader.load(Self.file(model + records)) }
        }
    }

    private static func integer(_ value: Int) -> Data {
        withUnsafeBytes(of: Int32(value).littleEndian) { Data($0) }
    }

    private static func integers(_ values: [Int]) -> Data {
        values.reduce(into: Data()) { $0 += integer($1) }
    }

    private static func dictionary(_ pairs: [(String, String)]) -> Data {
        pairs.reduce(into: integer(pairs.count)) { bytes, pair in
            for word in [pair.0, pair.1] {
                bytes += integer(word.utf8.count) + Data(word.utf8)
            }
        }
    }

    private static func chunk(_ id: String, _ payload: Data, children: Data = Data()) -> Data {
        Data(id.utf8) + integers([payload.count, children.count]) + payload + children
    }

    private static func file(_ chunks: Data, version: Int = 150) -> Data {
        Data("VOX ".utf8) + integer(version) + chunk("MAIN", Data(), children: chunks)
    }

    private static func model(_ size: SIMD3<Int>, _ voxels: [SIMD4<UInt8>]) -> Data {
        let bytes = voxels.flatMap { [$0.x, $0.y, $0.z, $0.w] }
        return chunk("SIZE", integers([size.x, size.y, size.z])) +
            chunk("XYZI", integer(voxels.count) + Data(bytes))
    }

    private static func palette(_ colors: [SIMD4<UInt8>]) -> Data {
        var bytes = colors.flatMap { [$0.x, $0.y, $0.z, $0.w] }
        bytes += [UInt8](repeating: 0, count: 1_024 - bytes.count)
        return chunk("RGBA", Data(bytes))
    }

    private static func transform(
        _ id: Int, child: Int, name: String? = nil, frames: [[(String, String)]]
    ) -> Data {
        chunk("nTRN", integer(id) + dictionary(name.map { [("_name", $0)] } ?? []) +
              integers([child, -1, 0, frames.count]) + frames.reduce(into: Data()) { $0 += dictionary($1) })
    }

    private static func shape(_ id: Int, models: [Int]) -> Data {
        chunk("nSHP", integer(id) + dictionary([]) + integer(models.count) +
              models.reduce(into: Data()) { $0 += integer($1) + dictionary([]) })
    }
}
