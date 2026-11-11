import Frostfield
import Metal
import Testing
import simd

struct MeshSkinTests {
    @Test("Bind matrices produce identity and a posed joint moves only its weighted share")
    func bindAndPose() throws {
        let geometry = MeshGeometry.cube()
        let influence = VertexSkinInfluence(joints: SIMD4(0, 1, 0, 0), weights: SIMD4(3, 1, 0, 0))
        let skin = try MeshSkin(jointCount: 2, influences: Array(repeating: influence, count: geometry.vertices.count))
        #expect(skin.influences[0].weights == SIMD4(0.75, 0.25, 0, 0))

        let meshWorld = translation(5)
        let inverseBind = [translation(-1), translation(-2)]
        let bind = try JointPalette(
            meshWorld: meshWorld, jointWorld: [translation(6), translation(7)], inverseBindMatrices: inverseBind
        )
        let identity = try JointPalette(matrices: [matrix_identity_float4x4, matrix_identity_float4x4])
        #expect(bind == identity)
        let bindGeometry = try skin.deform(geometry, using: bind)
        #expect(bindGeometry.vertices.map(\.position) == geometry.vertices.map(\.position))

        let pose = try JointPalette(
            meshWorld: meshWorld, jointWorld: [translation(6), translation(11)], inverseBindMatrices: inverseBind
        )
        let posed = try skin.deform(geometry, using: pose)
        #expect(posed.vertices[0].position == geometry.vertices[0].position + SIMD3(1, 0, 0))
        #expect(posed.vertices[0].normal == geometry.vertices[0].normal)
        #expect(posed.vertices[0].textureCoordinate == geometry.vertices[0].textureCoordinate)
        #expect(posed.indices == geometry.indices)
    }

    @Test("Nonuniform joint scale uses an inverse-transpose normal")
    func normal() throws {
        let sourceNormal = simd_normalize(SIMD3<Float>(1, 1, 0))
        let geometry = try MeshGeometry(vertices: [
            MeshVertex(position: .zero, normal: sourceNormal),
            MeshVertex(position: SIMD3(1, 0, 0), normal: sourceNormal),
            MeshVertex(position: SIMD3(0, 0, 1), normal: sourceNormal),
        ], indices: [0, 1, 2])
        let skin = try oneJointSkin(vertexCount: geometry.vertices.count)
        let pose = try JointPalette(matrices: [SceneTransform(scale: SIMD3(2, 1, 1)).matrix])
        let posed = try skin.deform(geometry, using: pose)
        let expected = simd_normalize(SIMD3<Float>(0.5, 1, 0))
        #expect(simd_distance(posed.vertices[0].normal, expected) < 0.00001)
    }

    @Test("Malformed influences and poses are rejected before drawing")
    func validation() throws {
        let good = VertexSkinInfluence(joints: .zero, weights: SIMD4(1, 0, 0, 0))
        #expect(throws: SkinningError.jointCount) {
            try MeshSkin(jointCount: 257, influences: [good])
        }
        #expect(throws: SkinningError.jointIndex(vertex: 0)) {
            try MeshSkin(jointCount: 1, influences: [
                VertexSkinInfluence(joints: SIMD4(0, 0, 0, 1), weights: good.weights)
            ])
        }
        #expect(throws: SkinningError.weights(vertex: 0)) {
            try MeshSkin(jointCount: 1, influences: [VertexSkinInfluence(joints: .zero, weights: .zero)])
        }
        #expect(throws: SkinningError.weights(vertex: 0)) {
            try MeshSkin(jointCount: 1, influences: [
                VertexSkinInfluence(joints: .zero, weights: SIMD4(1, .nan, 0, 0))
            ])
        }
        #expect(throws: SkinningError.jointTransform(0)) {
            try JointPalette(matrices: [SceneTransform(scale: .zero).matrix])
        }
        var projective = matrix_identity_float4x4
        projective.columns.0.w = 0.25
        #expect(throws: SkinningError.jointTransform(0)) {
            try JointPalette(matrices: [projective])
        }
        #expect(throws: SkinningError.paletteCount) {
            try JointPalette(meshWorld: matrix_identity_float4x4, jointWorld: [], inverseBindMatrices: [matrix_identity_float4x4])
        }
        let oneInfluence = try MeshSkin(jointCount: 1, influences: [good])
        let identity = try JointPalette(matrices: [matrix_identity_float4x4])
        #expect(throws: SkinningError.influenceCount) {
            try oneInfluence.deform(.cube(), using: identity)
        }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MeshSkinRenderingTests {
    private static let side = 128
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let camera: PerspectiveCamera

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
    }

    @Test("Inline and buffer palettes move raster geometry and picking together", arguments: [1, 64])
    func colorAndPicking(jointCount: Int) throws {
        let geometry = try plane(0.35)
        let influence = VertexSkinInfluence(
            joints: SIMD4(repeating: UInt16(jointCount - 1)), weights: SIMD4(1, 0, 0, 0)
        )
        let skin = try MeshSkin(jointCount: jointCount, influences: Array(repeating: influence, count: 4))
        let renderer = try MeshRenderer(device: device, colorPixelFormat: .rgba8Unorm, geometry: geometry, skin: skin)
        var matrices = Array(repeating: matrix_identity_float4x4, count: jointCount)
        matrices[jointCount - 1] = translation(1)
        let pose = try JointPalette(matrices: matrices)
        let transform = SceneTransform(translation: SIMD3(0, 0, -2))
        let pixels = try render { pass, buffer in
            #expect(throws: SkinningError.missingPalette) {
                try renderer.encode(MeshScene(camera: camera, transform: transform), pass: pass, commandBuffer: buffer)
            }
            let scene = MeshScene(camera: camera, transform: transform, jointPalette: pose)
            try renderer.encode(scene, pass: pass, commandBuffer: buffer)
        }
        #expect(pixel(pixels, x: 64, channel: 3) == 0)
        #expect(pixel(pixels, x: 96, channel: 3) == 255)

        let node = SceneNode(transform: transform, mesh: renderer, jointPalette: pose)
        let graph = SceneGraph(roots: [node], camera: camera)
        #expect(try graph.pick(at: SIMD2(0.5, 0.5), aspectRatio: 1) == nil)
        #expect(try graph.pick(at: SIMD2(0.75, 0.5), aspectRatio: 1)?.nodeID == node.id)
    }

    @Test("A posed blocker casts a shadow from its deformed position")
    func skinnedShadow() throws {
        let receiver = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            mesh: try MeshRenderer(
                device: device, colorPixelFormat: .rgba8Unorm, geometry: plane(1.6), material: .shadowedLambert
            ),
            castsShadow: false
        )
        var blocker = SceneNode(
            transform: SceneTransform(translation: SIMD3(1.6, 0, -2)),
            mesh: try MeshRenderer(
                device: device, colorPixelFormat: .rgba8Unorm, geometry: plane(0.25), skin: oneJointSkin(vertexCount: 4)
            ),
            jointPalette: try JointPalette(matrices: [matrix_identity_float4x4])
        )
        let shadowMap = try DirectionalShadowMap(
            queue: queue, resolution: 256, directionToLight: SIMD3(0.6, 0, 0.8),
            center: SIMD3(0, 0, -3), halfExtent: 3
        )
        var graph = SceneGraph(roots: [receiver, blocker], camera: camera)
        let bind = try render { try graph.encode(pass: $0, commandBuffer: $1, shadowMap: shadowMap) }
        blocker.jointPalette = try JointPalette(matrices: [translation(-0.85)])
        graph.roots[1] = blocker
        let posed = try render { try graph.encode(pass: $0, commandBuffer: $1, shadowMap: shadowMap) }
        #expect(Int(pixel(bind, x: 64, channel: 0)) > Int(pixel(posed, x: 64, channel: 0)) + 60)
        #expect(pixel(posed, x: 64, channel: 3) == 255)
    }

    private func plane(_ halfExtent: Float) throws -> MeshGeometry {
        let h = halfExtent
        let positions = [SIMD3(-h, -h, 0), SIMD3(h, -h, 0), SIMD3(h, h, 0), SIMD3(-h, h, 0)]
        return try MeshGeometry(
            vertices: positions.map { MeshVertex(position: $0, normal: SIMD3(0, 0, 1)) },
            indices: [0, 1, 2, 0, 2, 3]
        )
    }

    private func render(
        _ encode: (MTLRenderPassDescriptor, any MTLCommandBuffer) throws -> Void
    ) throws -> [UInt8] {
        let side = Self.side
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: side, height: side, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: side, height: side, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let color = try #require(device.makeTexture(descriptor: colorDescriptor))
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        pass.depthAttachment.clearDepth = 1
        let buffer = try #require(queue.makeCommandBuffer())
        try encode(pass, buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed, "\(buffer.error?.localizedDescription ?? "Render did not complete.")")

        var pixels = [UInt8](repeating: 0, count: side * side * 4)
        try pixels.withUnsafeMutableBytes { bytes in
            color.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: side * 4,
                from: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0
            )
        }
        return pixels
    }

    private func pixel(_ pixels: [UInt8], x: Int, channel: Int) -> UInt8 {
        pixels[(Self.side / 2 * Self.side + x) * 4 + channel]
    }
}

private func translation(_ x: Float) -> simd_float4x4 {
    SceneTransform(translation: SIMD3(x, 0, 0)).matrix
}

private func oneJointSkin(vertexCount: Int) throws -> MeshSkin {
    let influence = VertexSkinInfluence(joints: .zero, weights: SIMD4(1, 0, 0, 0))
    return try MeshSkin(jointCount: 1, influences: Array(repeating: influence, count: vertexCount))
}
