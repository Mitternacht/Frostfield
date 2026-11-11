import Frostfield
import Metal
import Testing
import simd

struct PlanarProjectedShadowTests {
    @Test("Directional and positional projections match the r187 affine caster oracle")
    func referenceProjection() throws {
        let plane = try Plane3D(normal: SIMD3(1, 2, 2), constant: 4)
        let transform = simd_float4x4(columns: (
            SIMD4(1.25, -0.125, 0.125, 0), SIMD4(0.25, 0.75, -0.25, 0),
            SIMD4(0, 0.125, 1, 0), SIMD4(0.5, 0.25, -1.5, 1)
        ))
        let authored = try MeshGeometry(vertices: [
            MeshVertex(position: .zero, normal: SIMD3(0, 0, 1), textureCoordinate: SIMD2(0, 1),
                       secondaryTextureCoordinate: SIMD2(0.25, 0.5), tangent: SIMD4(1, 0, 0, 1), color: SIMD4(1, 0, 0, 1)),
            MeshVertex(position: SIMD3(1, 0, 0), normal: SIMD3(0, 0, 1), textureCoordinate: SIMD2(1, 1)),
            MeshVertex(position: SIMD3(0, 1, 1), normal: SIMD3(0, 0, 1), textureCoordinate: SIMD2(0, 0)),
        ], indices: [0, 1, 2]).withGroups([.init(start: 0, count: 3, materialIndex: 2)])
        let cases: [(SIMD4<Float>, [SIMD3<Float>])] = [
            (SIMD4(2, 3, 4, 0), [SIMD3(0.25, -0.125, -2), SIMD3(1.34375, -0.484375, -2.1875), SIMD3(0.0625, 0.09375, -2.125)]),
            (SIMD4(0, 6, 0, 2), [SIMD3(0.625, -0.4375, -1.875), SIMD3(2.5925926, -1.2592593, -2.037037),
                                  SIMD3(1.6666667, -1.1666667, -1.6666667)]),
        ]
        for (light, expected) in cases {
            let projection = try PlanarShadowProjection(plane: plane, light: light)
            let projected = try projection.project(authored, sourceToWorld: transform)
            #expect(projected.indices == authored.indices && projected.groups == authored.groups)
            for (index, vertex) in projected.vertices.enumerated() {
                let source = authored.vertices[index]
                #expect(simd_distance(vertex.position, expected[index]) < 1e-6)
                #expect(abs(plane.signedDistance(to: vertex.position)) < 1e-6)
                #expect(vertex.normal == plane.normal && vertex.tangent == nil)
                #expect(vertex.textureCoordinate == source.textureCoordinate)
                #expect(vertex.secondaryTextureCoordinate == source.secondaryTextureCoordinate && vertex.color == source.color)
            }
            #expect(try projection.project(SIMD3(0, 0, -2)) == SIMD3(0, 0, -2))
            let scaled = try PlanarShadowProjection(plane: plane, light: -2 * light)
            #expect(try scaled.project(authored, sourceToWorld: transform).vertices.map(\.position) == projected.vertices.map(\.position))
        }
    }

    @Test("Invalid lights, nonaffine casters and the positional horizon fail cleanly")
    func invalidProjection() throws {
        #expect(throws: Plane3D.Failure.invalidNormal) { try Plane3D(normal: .zero, constant: 0) }
        let plane = try Plane3D(normal: SIMD3(0, 1, 0), constant: 0)
        #expect(throws: PlanarShadowProjection.Failure.invalidLight) {
            try PlanarShadowProjection(plane: plane, light: .zero)
        }
        #expect(throws: PlanarShadowProjection.Failure.invalidLight) {
            try PlanarShadowProjection(plane: plane, light: SIMD4(.infinity, 1, 0, 1))
        }
        #expect(throws: PlanarShadowProjection.Failure.parallelLight) {
            try PlanarShadowProjection(plane: plane, light: SIMD4(1, 0, 0, 0))
        }
        #expect(throws: PlanarShadowProjection.Failure.lightOnPlane) {
            try PlanarShadowProjection(plane: plane, light: SIMD4(1, 0, 0, 1))
        }
        let shadow = try PlanarShadowProjection(plane: plane, light: SIMD4(0, 2, 0, 1))
        #expect(throws: PlanarShadowProjection.Failure.rayHorizon) { try shadow.project(SIMD3(1, 2, 0)) }
        #expect(throws: PlanarShadowProjection.Failure.rayHorizon) { try shadow.project(SIMD3(1, 3, 0)) }
        var invalid = matrix_identity_float4x4
        invalid.columns.0.w = 0.1
        #expect(throws: PlanarShadowProjection.Failure.sourceTransform) {
            try shadow.project(MeshGeometry.cube(), sourceToWorld: invalid)
        }
    }

    @Test("Metal blends the projected cube once and respects scene depth",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"),
          arguments: [false, true])
    func renderedShadow(reversed: Bool) throws {
        let gpu = try GeometryRenderFixture(side: 128)
        let mode: MeshDepthMode = reversed ? .reversed : .forward
        let receiverColor = SIMD3<Float>(0.75, 0.5, 0.25)
        let blockerColor = SIMD3<Float>(0, 0.5, 1)
        let casterColor = SIMD3<Float>(1, 0.25, 0)
        func renderer(_ mesh: MeshGeometry, color: SIMD3<Float>) throws -> MeshRenderer {
            try gpu.unlit(mesh, material: .classic(ClassicMaterial(model: .basic, color: color)), depthMode: mode)
        }
        let receiver = try SceneNode(
            transform: .init(translation: SIMD3(0, 0, -4)),
            mesh: renderer(GeometryRenderFixture.square(2.3), color: receiverColor)
        )
        let blocker = try SceneNode(
            transform: .init(translation: SIMD3(0.5625, 0.135, -3)),
            mesh: renderer(GeometryRenderFixture.square(0.075), color: blockerColor)
        )
        let caster = try SceneNode(
            transform: .init(translation: SIMD3(-0.2, 0, -0.4),
                             rotation: simd_quatf(angle: 0.17, axis: SIMD3(0, 1, 0)), scale: SIMD3(0.6, 0.5, 0.55)),
            mesh: renderer(.cube(), color: casterColor)
        )
        let parent = SceneNode(transform: .init(translation: SIMD3(-0.5, 0, -2)), children: [caster])
        let graph = SceneGraph(roots: [receiver, blocker, parent], camera: gpu.camera)
        let resolved = try #require(try graph.resolve().first { $0.id == caster.id })
        let mesh = try #require(resolved.mesh?.geometry)
        let plane = try Plane3D(normal: SIMD3(0, 0, 1), through: SIMD3(0, 0, -4))
        let light: SIMD4<Float> = reversed ? SIMD4(-2, 0, -1, 1) : SIMD4(-0.9, 0, 1, 0)
        let projection = try PlanarShadowProjection(plane: plane, light: light)
        let projected = try projection.project(mesh, sourceToWorld: resolved.worldTransform)
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: gpu.side, height: gpu.side, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        let targetColor = try #require(gpu.device.makeTexture(descriptor: colorDescriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: gpu.side, height: gpu.side, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = [.renderTarget, .shaderRead]
        let targetDepth = try #require(gpu.device.makeTexture(descriptor: depthDescriptor))
        let stencil = try scratchStencil(gpu)
        let command = try #require(gpu.queue.makeCommandBuffer())
        let pass = gpu.pass(color: targetColor, depth: targetDepth, depthMode: mode)
        pass.depthAttachment.storeAction = .store
        try graph.encode(pass: pass, commandBuffer: command, depthMode: mode)
        let shadow = try PlanarProjectedShadowPass(device: gpu.device, depthMode: mode)
        // A cube's front and back faces overlap after projection. Submit it
        // twice as well, to check stencil deduplication across separate draws.
        try shadow.encode(
            projectedMeshes: [projected, projected], clipFromWorld: gpu.camera.projection(aspectRatio: 1, depthMode: mode),
            color: targetColor, depth: targetDepth, stencil: stencil, tint: SIMD4(0, 0, 0, 0.5), commandBuffer: command
        )
        try gpu.finish(command)
        let pixels = try gpu.pixels(of: targetColor)
        func pixel(_ x: Float, _ y: Float, depth: Float = 4) -> SIMD4<Float> {
            gpu.pixel(pixels, x: gpu.column(x: x, depth: depth), y: gpu.row(y: y, depth: depth))
        }
        #expect(pixel(0.75, 0) == SIMD4(receiverColor * 0.5, 1))
        #expect(pixel(0.75, 0.18) == SIMD4(blockerColor, 1))
        #expect(pixel(-1.8, 1.2) == SIMD4(receiverColor, 1))
        #expect(pixel(-0.7, 0, depth: 2.4) == SIMD4(casterColor, 1))
    }

    @Test("The projected shadow pass validates scratch textures and retaining commands",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func passValidation() throws {
        let gpu = try GeometryRenderFixture()
        let shadow = try PlanarProjectedShadowPass(device: gpu.device)
        let target = try gpu.makeTarget()
        let stencil = try scratchStencil(gpu)
        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PlanarProjectedShadowPass.Failure.attachments) {
            try shadow.encode(projectedMeshes: [], clipFromWorld: matrix_identity_float4x4, color: target.color,
                              depth: target.depth, stencil: target.depth, commandBuffer: command)
        }
        #expect(throws: PlanarProjectedShadowPass.Failure.attachments) {
            try shadow.encode(projectedMeshes: [], clipFromWorld: matrix_identity_float4x4, color: target.color,
                              depth: target.depth, stencil: stencil, commandBuffer: command)
        }
        let unretained = try #require(gpu.queue.makeCommandBufferWithUnretainedReferences())
        #expect(throws: PlanarProjectedShadowPass.Failure.commandBuffer) {
            try shadow.encode(projectedMeshes: [], clipFromWorld: matrix_identity_float4x4, color: target.color,
                              depth: target.depth, stencil: stencil, commandBuffer: unretained)
        }
    }

    private func scratchStencil(_ gpu: GeometryRenderFixture) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .stencil8, width: gpu.side, height: gpu.side, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = .renderTarget
        return try #require(gpu.device.makeTexture(descriptor: descriptor))
    }
}
