import Foundation
import Frostfield
import Metal
import Testing
import simd

struct MTLLoaderTests {
    @Test("A standalone MTL library resolves local maps and prepares named materials")
    func loadsLibrary() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        let folder = root.appending(path: "asset")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let image = try #require(Data(base64Encoded:
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAACCAYAAACZgbYnAAAAAXNSR0IArs4c6QAAABJJREFUeJxjaHD438Ag8izlPwAVTASdmr2NEwAAAABJRU5ErkJggg=="
        ))
        try image.write(to: folder.appending(path: "paint.png"))
        try image.write(to: root.appending(path: "outside.png"))
        let url = folder.appending(path: "library.mtl")
        try """
        newmtl Painted Alloy
        Kd 0.25 0.5 0.75
        Pr 0.4
        Pm 0.7
        d 0.25
        map_Kd -o 0.25 0 -clamp on paint.png
        bump -bm 0.5 paint.png

        newmtl Outside
        Ns 98
        map_Kd ../outside.png
        """.write(to: url, atomically: true, encoding: .utf8)

        let library = try MTLLoader.load(contentsOf: url)
        #expect(Set(library.materials.keys) == ["Painted Alloy", "Outside"])
        let painted = try #require(library.materials["Painted Alloy"])
        #expect(painted.baseColorFactor == SIMD4(0.25, 0.5, 0.75, 0.25))
        #expect(painted.metallicFactor == 0.7 && painted.roughnessFactor == 0.4 && painted.alphaMode == .blend)
        let texture = try #require(painted.baseColorTexture)
        #expect(texture.image.data == image && texture.transform.offset == SIMD2(0.25, 0))
        #expect(texture.sampling.wrapS == .clampToEdge)
        #expect(painted.bumpMap?.texture.image.data == image && painted.bumpMap?.scale == 0.5)
        if let device = MTLCreateSystemDefaultDevice() {
            let material = try painted.prepare(device: device)
            #expect(material.baseColorTexture?.pixelFormat == .rgba8Unorm_srgb)
            #expect(material.bumpMap?.texture.texture.pixelFormat == .rgba8Unorm)
        }
        let outside = try #require(library.materials["Outside"])
        #expect(abs(outside.roughnessFactor - pow(2 / Float(100), 0.25)) < 1e-6)
        #expect(outside.baseColorTexture == nil)
        #expect(library.warnings.contains(.textureNotLoaded(material: "Outside", path: "../outside.png")))
    }

    @Test("An MTL load requires a local file and at least one named material")
    func invalidLibrary() throws {
        let remote = try #require(URL(string: "https://example.invalid/library.mtl"))
        #expect(throws: MTLLoader.Failure.localFileRequired) { try MTLLoader.load(contentsOf: remote) }
        let local = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString + ".mtl")
        defer { try? FileManager.default.removeItem(at: local) }
        try "# empty material library\nKd 0 1 0\n".write(to: local, atomically: true, encoding: .utf8)
        #expect(throws: MTLLoader.Failure.emptyLibrary) { try MTLLoader.load(contentsOf: local) }
    }
}
