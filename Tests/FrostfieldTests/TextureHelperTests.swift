import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TextureHelperTests {
    @Test("A nested plane transforms its source texels and sees GPU writes in the same frame")
    func livePlane() throws {
        let fixture = try RasterLightingFixture(width: 64, height: 64)
        let source = try texture(fixture.device, type: .type2D, side: 2)
        let texels: [SIMD4<Float16>] = [
            SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0.25), SIMD4(0, 0, 1, 0.5), SIMD4(1, 1, 1, 0.75),
        ]
        upload(texels, side: 2, to: source)
        let node = try DebugHelpers(device: fixture.device).texture(
            source, size: SIMD3(2, 2, 1), transform: .init(rotation: simd_quatf(angle: .pi / 2, axis: SIMD3(0, 0, 1)))
        )
        #expect(!node.castsShadow)
        let parent = SceneNode(transform: .init(translation: SIMD3(1, 0, -4)), children: [node])
        let scene = SceneGraph(roots: [parent], camera: try camera())
        let first = try fixture.pixels(of: fixture.render(scene).color)
        #expect(first[36 * 64 + 36] == SIMD4(1, 0, 0, 1))
        #expect(first[28 * 64 + 36] == SIMD4(0, 1, 0, 1))
        #expect(first[36 * 64 + 44] == SIMD4(0, 0, 1, 1))
        #expect(first[28 * 64 + 44] == SIMD4(1, 1, 1, 1))
        #expect(first[32 * 64 + 24] == .zero)

        let color = try fixture.colorTarget(), depth = try fixture.depthTarget()
        try fixture.run { buffer in
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = source
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.25, green: 0.5, blue: 2, alpha: 0)
            let producer = try #require(buffer.makeRenderCommandEncoder(descriptor: pass))
            producer.endEncoding()
            try scene.encode(pass: fixture.pass(color: color, depth: depth), commandBuffer: buffer)
        }
        let updated = try fixture.pixels(of: color)
        #expect(updated[36 * 64 + 36] == SIMD4(0.25, 0.5, 2, 1))
        #expect(updated[32 * 64 + 24] == .zero)
    }

    @Test("All six transformed box faces display their selected Metal cube mip and texel orientation")
    func cubeFaces() throws {
        let fixture = try RasterLightingFixture(width: 64, height: 64)
        let source = try texture(fixture.device, type: .typeCube, side: 4, mips: 2)
        let bases: [(right: SIMD3<Float>, up: SIMD3<Float>, normal: SIMD3<Float>)] = [
            (SIMD3(0, 0, -1), SIMD3(0, 1, 0), SIMD3(1, 0, 0)),
            (SIMD3(0, 0, 1), SIMD3(0, 1, 0), SIMD3(-1, 0, 0)),
            (SIMD3(1, 0, 0), SIMD3(0, 0, -1), SIMD3(0, 1, 0)),
            (SIMD3(1, 0, 0), SIMD3(0, 0, 1), SIMD3(0, -1, 0)),
            (SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)),
            (SIMD3(-1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, -1)),
        ]
        var node = try DebugHelpers(device: fixture.device).texture(source, size: SIMD3(1.5, 1, 2), mipLevel: 1)
        let camera = try camera()
        for (face, basis) in bases.enumerated() {
            let red = Float16(face + 1) / 8
            let markers = (1...4).map { SIMD4<Float16>(red, Float16($0) / 8, 0.5, 0) }
            upload(markers, side: 2, to: source, slice: face, mip: 1)
            let orientation = simd_quatf(simd_float3x3(columns: (basis.right, basis.up, basis.normal)).transpose)
            node.transform = .init(translation: SIMD3(0, 0, -3), rotation: orientation)
            let result = try fixture.render(SceneGraph(roots: [node], camera: camera))
            let pixels = try fixture.pixels(of: result.color)
            let corners = [28 * 64 + 28, 28 * 64 + 35, 35 * 64 + 28, 35 * 64 + 35]
            #expect(corners.map { pixels[$0] } == markers.map { SIMD4(Float($0.x), Float($0.y), Float($0.z), 1) })
        }
    }

    @Test("Array and volume layers share their mesh and composite in camera order")
    func stacks() throws {
        let fixture = try RasterLightingFixture(width: 32, height: 32)
        let helper = DebugHelpers(device: fixture.device)
        let camera = try camera()
        for type: MTLTextureType in [.type2DArray, .type3D] {
            let source = try texture(fixture.device, type: type, side: 4, mips: 3)
            upload([SIMD4<Float16>](repeating: SIMD4(1, 0, 0, 0), count: 4), side: 2, to: source, mip: 1)
            upload([SIMD4<Float16>](repeating: SIMD4(0, 0, 1, 0), count: 4), side: 2, to: source, slice: 1, mip: 1)
            var node = try helper.texture(
                source, size: SIMD3(2, 2, 1), mipLevel: 1, transform: .init(translation: SIMD3(0, 0, -3))
            )
            try #require(node.children.count == 2)
            #expect(node.children[0].mesh === node.children[1].mesh)
            #expect(node.children.map(\.transform.translation.z) == [-0.5, 0.5])
            #expect(node.children.allSatisfy { !$0.castsShadow })
            let front = try fixture.render(SceneGraph(roots: [node], camera: camera))
            #expect(try fixture.pixels(of: front.color)[16 * 32 + 16] == SIMD4(0.25, 0, 0.5, 0.75))
            node.transform.rotation = simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
            let back = try fixture.render(SceneGraph(roots: [node], camera: camera))
            #expect(try fixture.pixels(of: back.color)[16 * 32 + 16] == SIMD4(0.5, 0, 0.25, 0.75))

            if type == .type3D {
                upload([SIMD4(0, 1, 0, 0)], side: 1, to: source, mip: 2)
                let single = try helper.texture(source, mipLevel: 2, transform: .init(translation: SIMD3(0, 0, -3)))
                #expect(single.children.map(\.transform.translation) == [.zero])
                let one = try fixture.render(SceneGraph(roots: [single], camera: camera))
                #expect(try fixture.pixels(of: one.color)[16 * 32 + 16] == SIMD4(0, 1, 0, 1))
            }
        }
    }

    @Test("Invalid helper size and mips fail, and its live source cannot also be the scene target")
    func validation() throws {
        let fixture = try RasterLightingFixture(width: 32, height: 32)
        let source = try fixture.colorTarget()
        let depth = try fixture.depthTarget()
        let helper = DebugHelpers(device: fixture.device)
        #expect(throws: PrimitiveGeometryError.invalidHelperParameter("texture helper dimensions must be positive and normal")) {
            try helper.texture(source, size: SIMD3(1, .nan, 1))
        }
        #expect(throws: TextureDebugPass.Failure.subresource) { try helper.texture(source, mipLevel: 1) }
        let alias = try #require(source.makeTextureView(pixelFormat: .rgba16Float))
        let node = try helper.texture(alias, transform: .init(translation: SIMD3(0, 0, -3)))
        let scene = SceneGraph(roots: [node], camera: try camera())
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: TextureDebugPass.Failure.destination) {
            try scene.encode(pass: fixture.pass(color: source, depth: depth), commandBuffer: buffer)
        }
    }

    private func camera() throws -> PerspectiveCamera {
        try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 16)
    }

    private func texture(
        _ device: any MTLDevice, type: MTLTextureType, side: Int, mips: Int = 1
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = type
        descriptor.pixelFormat = .rgba16Float
        descriptor.width = side
        descriptor.height = side
        descriptor.depth = type == .type3D ? side : 1
        descriptor.arrayLength = type == .type2DArray ? 2 : 1
        descriptor.mipmapLevelCount = mips
        descriptor.storageMode = .shared
        descriptor.usage = type == .type2D ? [.shaderRead, .renderTarget] : .shaderRead
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    private func upload(
        _ texels: [SIMD4<Float16>], side: Int, to texture: any MTLTexture, slice: Int = 0, mip: Int = 0
    ) {
        let isVolume = texture.textureType == .type3D
        let region = MTLRegionMake3D(0, 0, isVolume ? slice : 0, side, side, 1)
        let rowBytes = side * MemoryLayout<SIMD4<Float16>>.stride
        texels.withUnsafeBytes {
            texture.replace(region: region, mipmapLevel: mip, slice: isVolume ? 0 : slice,
                            withBytes: $0.baseAddress!, bytesPerRow: rowBytes, bytesPerImage: rowBytes * side)
        }
    }
}
