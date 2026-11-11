import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ToonOutlinePassTests {
    private let fixture: GeometryRenderFixture
    private let outlineRGB = SIMD3<Float>(0.8, 0.1, 0.4)

    init() throws { fixture = try GeometryRenderFixture(side: 128) }

    @Test("Toon hull matches projected thickness, preserves front color and respects scene depth",
          arguments: [MeshDepthMode.forward, .reversed])
    func projectedHull(depth: MeshDepthMode) throws {
        let radius: Float = 0.35
        let sphere = try MeshGeometry.uvSphere(radius: radius, segments: 64, rings: 32)
        let toon = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -3)),
            mesh: try fixture.unlit(sphere, material: .classic(toonMaterial()), depthMode: depth)
        )
        let basic = try ClassicMaterial(model: .basic, color: SIMD3(0.1, 0.8, 0.3))
        let other = SceneNode(
            transform: .init(translation: SIMD3(-0.74, 0, -3)),
            mesh: try fixture.unlit(
                MeshGeometry.uvSphere(radius: 0.14, segments: 32, rings: 16),
                material: .classic(basic), depthMode: depth
            )
        )
        let occluder = SceneNode(
            transform: .init(translation: SIMD3(0.44, 0, -1)),
            mesh: try fixture.unlit(GeometryRenderFixture.square(0.07), material: .classic(basic), depthMode: depth)
        )
        let graph = SceneGraph(roots: [toon, other], camera: fixture.camera)
        let wide = try ToonOutlinePass(thickness: 0.125, color: outlineRGB, alpha: 0.5)
        let narrow = try ToonOutlinePass(thickness: 0.0625, color: outlineRGB, alpha: 0.5)
        let expected = SIMD4(outlineRGB * 0.5, 0.5)
        let plain = try render(graph, depth: depth, orthographic: true)
        let result = try render(graph, outline: wide, depth: depth, orthographic: true)
        let reduced = try render(graph, outline: narrow, depth: depth, orthographic: true)
        #expect(result.statistics.drawnMeshCount == 2)
        #expect(result.statistics.toonOutlineDrawCount == 1)
        #expect(result.statistics.submittedTriangleCount == plain.statistics.submittedTriangleCount + sphere.indices.count / 3)
        #expect(result.pixels.contains { isNearly($0, expected) })
        #expect(isNearly(pixel(result.pixels, 64), pixel(plain.pixels, 64)))
        #expect(pixel(result.pixels, 64).w == 1)
        #expect(isNearly(pixel(result.pixels, 16), pixel(plain.pixels, 16)))
        #expect(pixel(result.pixels, 28) == .zero) // A non-toon mesh has no expanded shell.

        // At the right equator the normal is +X and clip W is 1. Three.js's
        // clip-space formula projects radius .35 and hull radii .4125 / .475.
        #expect(pixel(plain.pixels, 86) == .zero)
        #expect(isNearly(pixel(result.pixels, 86), expected))
        #expect(isNearly(pixel(reduced.pixels, 89), expected))
        #expect(pixel(reduced.pixels, 92) == .zero)
        #expect(isNearly(pixel(result.pixels, 93), expected))
        #expect(pixel(result.pixels, 94) == .zero)

        for nodes in [[occluder, toon, other], [toon, other, occluder]] {
            let blocked = try render(
                SceneGraph(roots: nodes, camera: fixture.camera), outline: wide, depth: depth, orthographic: true
            )
            #expect(isNearly(pixel(blocked.pixels, 92), SIMD4(0.1, 0.8, 0.3, 1)))
            #expect(isNearly(pixel(blocked.pixels, 35), expected))
        }
        let backdrop = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -5)),
            mesh: try fixture.unlit(GeometryRenderFixture.square(1), material: .classic(basic), depthMode: depth)
        )
        let composited = try render(
            SceneGraph(roots: [backdrop, toon, other], camera: fixture.camera),
            outline: wide, depth: depth, orthographic: true
        )
        #expect(isNearly(pixel(composited.pixels, 92), SIMD4(0.45, 0.45, 0.35, 1)))

        for disabled in [
            try ToonOutlinePass(thickness: 0, color: outlineRGB),
            try ToonOutlinePass(thickness: 0.125, color: outlineRGB, alpha: 0),
        ] {
            let bypass = try render(graph, outline: disabled, depth: depth, orthographic: true)
            #expect(bypass.pixels == plain.pixels)
            #expect(bypass.statistics == plain.statistics)
        }
    }

    @Test("A moving skinned and morphed toon hull follows the posed instanced geometry")
    func posedHull() throws {
        let sphere = try MeshGeometry.uvSphere(radius: 0.32)
        let count = sphere.vertices.count
        let morph = try MeshMorphTargets(vertexCount: count, targets: [
            .init(positionOffsets: Array(repeating: SIMD3(0.2, 0.12, 0), count: count))
        ])
        let skin = try MeshSkin(
            jointCount: 1,
            influences: Array(repeating: VertexSkinInfluence(joints: .zero, weights: SIMD4(1, 0, 0, 0)), count: count)
        )
        let palette = try JointPalette(matrices: [translationMatrix(0.08, 0.12, -0.15)])
        let instances = try InstanceSet(device: fixture.device, transforms: [
            translationMatrix(-0.72), translationMatrix(0.62),
        ])
        let material = try MeshMaterial.classic(toonMaterial())
        let gpuMesh = try fixture.unlit(sphere, skin: skin, morphTargets: morph, supportsInstancing: true, material: material)
        let moved = try skin.deform(morph.apply(weights: [1], to: sphere), using: palette)
        let cpuMesh = try fixture.unlit(moved, supportsInstancing: true, material: material)
        let transform = SceneTransform(translation: SIMD3(0, 0, -2))
        let gpu = SceneGraph(roots: [SceneNode(
            transform: transform, mesh: gpuMesh, jointPalette: palette, instances: instances, morphWeights: [1]
        )], camera: fixture.camera)
        let cpu = SceneGraph(roots: [SceneNode(
            transform: transform, mesh: cpuMesh, instances: instances
        )], camera: fixture.camera)
        let outline = try ToonOutlinePass(thickness: 0.065, color: outlineRGB, alpha: 0.6)
        let rendered = try render(gpu, outline: outline)
        let reference = try render(cpu, outline: outline)
        #expect(rendered.statistics.toonOutlineDrawCount == 1)
        #expect(rendered.statistics.drawnInstanceCount == 2)
        #expect(rendered.pixels.contains { isNearly($0, SIMD4(outlineRGB * 0.6, 0.6)) })
        #expect(zip(rendered.pixels, reference.pixels).allSatisfy { isNearly($0, $1) })
    }

    @Test("The toon hull reuses batched ranges and mirrors their winding")
    func batchedHull() throws {
        let sphere = try MeshGeometry.uvSphere(radius: 0.31)
        let small = try MeshGeometry.uvSphere(radius: 0.24)
        let material = try MeshMaterial.classic(toonMaterial())
        let transforms = [
            SceneTransform(translation: SIMD3(-0.53, 0, 0)),
            SceneTransform(translation: SIMD3(0.46, 0, 0), scale: SIMD3(-1, 1, 1)),
        ]
        let transform = SceneTransform(translation: SIMD3(0, 0, -2))
        let outline = try ToonOutlinePass(thickness: 0.1, color: outlineRGB)
        let references = try zip([sphere, small], transforms).map { geometry, transform in
            SceneNode(transform: transform, mesh: try fixture.unlit(geometry, material: material))
        }
        let expected = try render(
            SceneGraph(roots: [SceneNode(transform: transform, children: references)], camera: fixture.camera),
            outline: outline
        )
        let batch = try BatchedMesh(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometries: [sphere, small], capacity: 2,
            material: material
        )
        for (index, placement) in transforms.enumerated() {
            try batch.addInstance(geometryIndex: index, transform: placement.matrix)
        }
        let rendered = try render(
            SceneGraph(roots: [SceneNode(transform: transform, batch: batch)], camera: fixture.camera),
            outline: outline
        )
        #expect(rendered.statistics.toonOutlineDrawCount == 2)
        #expect(rendered.statistics.submittedTriangleCount == expected.statistics.submittedTriangleCount)
        #expect(rendered.pixels.contains { isNearly($0, SIMD4(outlineRGB, 1)) })
        #expect(zip(rendered.pixels, expected.pixels).allSatisfy { isNearly($0, $1) })
    }

    @Test("OutlineEffect uses non-toon meshes, inherits opacity and applies each node's overrides")
    func outlineEffectOverrides() throws {
        let geometry = try MeshGeometry.uvSphere(radius: 0.12, segments: 64, rings: 32)
        let basic = try ClassicMaterial(model: .basic, color: SIMD3(0, 1, 0), opacity: 0.25)
        let live = try PhysicallyBasedMaterial(shadingModel: .unlit).withLiveFactors()
        try #require(live.liveFactors).update { $0.baseColorFactor.w = 0.5 }
        let color = SIMD3<Float>(1, 0.5, 0)
        func node(_ x: Float, _ material: MeshMaterial, state: MeshRenderState? = nil) throws -> SceneNode {
            SceneNode(
                transform: .init(translation: SIMD3(x, 0, -3)),
                mesh: try MeshRenderer(
                    device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: material, renderState: state
                )
            )
        }
        let left = try node(-0.72, .classic(basic))
        let custom = try node(-0.25, .physicallyBased(live))
        let depthDisabled = try node(0.23, .unlit, state: .init(depthTest: false))
        let hidden = try node(0.72, .classic(basic))
        let graph = SceneGraph(roots: [left, custom, depthDisabled, hidden], camera: fixture.camera)
        let effect = try ToonOutlinePass.outlineEffect(
            thickness: 0.05, color: outlineRGB,
            overrides: [custom.id: .init(thickness: 0.11, color: color), hidden.id: .init(visible: false)]
        )
        let plain = try render(graph, orthographic: true)
        let outlined = try render(graph, outline: effect, orthographic: true)
        #expect(outlined.statistics.toonOutlineDrawCount == 2)
        #expect(pixel(plain.pixels, 26) == .zero && pixel(plain.pixels, 60) == .zero)
        #expect(isNearly(pixel(outlined.pixels, 26), SIMD4(outlineRGB * 0.25, 0.25)))
        #expect(isNearly(pixel(outlined.pixels, 60), SIMD4(color * 0.5, 0.5)))
        #expect(pixel(outlined.pixels, 88) == .zero && pixel(outlined.pixels, 121) == .zero)
        #expect(pixel(outlined.pixels, 18) == pixel(plain.pixels, 18))
        #expect(try render(graph, orthographic: true).pixels == plain.pixels)
    }

    @Test("OutlineEffect composites after the scene color even when the backdrop draws later")
    func outlineEffectAfterScene() throws {
        let sphere = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -2)),
            mesh: try fixture.unlit(MeshGeometry.uvSphere(radius: 0.35, segments: 64, rings: 32), material: .classic(toonMaterial()))
        )
        let backdrop = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -5)),
            mesh: try fixture.unlit(GeometryRenderFixture.square(1), material: .classic(.init(model: .basic, color: SIMD3(0, 1, 0)))),
            renderOrder: 1
        )
        let graph = SceneGraph(roots: [sphere, backdrop], camera: fixture.camera)
        let red = SIMD3<Float>(1, 0, 0)
        let inline = try ToonOutlinePass(thickness: 0.125, color: red, alpha: 0.5)
        let effect = try ToonOutlinePass.outlineEffect(
            thickness: 0.125, color: red, overrides: [sphere.id: .init(alpha: 0.5), backdrop.id: .init(visible: false)]
        )
        let before = try render(graph, outline: inline, orthographic: true)
        let after = try render(graph, outline: effect, orthographic: true)
        #expect(isNearly(pixel(before.pixels, 92), SIMD4(0.5, 0, 0, 0.5)))
        #expect(isNearly(pixel(after.pixels, 92), SIMD4(0.5, 0.5, 0, 1)))
        #expect(after.statistics.toonOutlineDrawCount == 1)

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: fixture.side, height: fixture.side, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let opaqueColor = try #require(fixture.device.makeTexture(descriptor: descriptor))
        var splitGraph = graph
        splitGraph.viewportSnapshot = try SceneViewportSnapshot(color: opaqueColor, source: .opaquePass)
        let split = try render(splitGraph, outline: effect, orthographic: true)
        #expect(split.pixels == after.pixels && split.statistics.toonOutlineDrawCount == 1)
        #expect(pixel(try fixture.pixels(of: opaqueColor), 92) == SIMD4(0, 1, 0, 1))
    }

    @Test("Translucent hulls sort back to front independently of their opaque scene draws",
          arguments: [MeshDepthMode.forward, .reversed])
    func outlineEffectSort(depth: MeshDepthMode) throws {
        let geometry = try MeshGeometry.uvSphere(radius: 0.35, segments: 64, rings: 32)
        let mesh = try fixture.unlit(geometry, depthMode: depth)
        let near = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)
        let far = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)
        let effect = try ToonOutlinePass.outlineEffect(thickness: 0.125, alpha: 0.5, overrides: [
            near.id: .init(color: SIMD3(1, 0, 0)), far.id: .init(color: SIMD3(0, 1, 0))
        ])
        for roots in [[near, far], [far, near]] {
            let graph = SceneGraph(roots: roots, camera: fixture.camera)
            let image = try render(graph, outline: effect, depth: depth, orthographic: true)
            #expect(image.statistics.toonOutlineDrawCount == 2)
            #expect(isNearly(pixel(image.pixels, 92), SIMD4(0.5, 0.25, 0, 0.75)))
        }
    }

    @Test("OutlineEffect reuses indexed batch packets while the ordinary draw retains ICB submission")
    func outlineEffectBatch() throws {
        let geometry = try MeshGeometry.uvSphere(radius: 0.28)
        let effect = try ToonOutlinePass.outlineEffect(thickness: 0.075, color: outlineRGB)
        func draw(_ submission: BatchedMesh.Submission) throws -> (pixels: [SIMD4<Float>], statistics: SceneDrawStatistics) {
            let batch = try BatchedMesh(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometries: [geometry], capacity: 2,
                material: .unlit, submission: submission
            )
            for x: Float in [-0.4, 0.4] {
                try batch.addInstance(geometryIndex: 0, transform: translationMatrix(x, 0, -2))
            }
            let graph = SceneGraph(roots: [SceneNode(batch: batch)], camera: fixture.camera)
            return try render(graph, outline: effect, orthographic: true)
        }

        let direct = try draw(.direct)
        #expect(direct.statistics.toonOutlineDrawCount == 1 && direct.statistics.drawnInstanceCount == 2)
        #expect(direct.pixels.contains { isNearly($0, SIMD4(outlineRGB, 1)) })
        if BatchedMesh.supportsIndirectSubmission(on: fixture.device) {
            let bundled = try draw(.indirectCommandBuffer)
            #expect(bundled.statistics.indirectBatchExecutionCount == 1)
            #expect(bundled.statistics.toonOutlineDrawCount == direct.statistics.toonOutlineDrawCount)
            #expect(bundled.pixels == direct.pixels)
        }
    }

    @Test("Invalid hull settings are rejected before rendering")
    func validation() throws {
        let defaults = try ToonOutlinePass()
        #expect(defaults.thickness == 0.003 && defaults.color == .zero && defaults.alpha == 1)
        #expect(throws: ToonOutlinePass.Failure.thickness) { try ToonOutlinePass(thickness: -.infinity) }
        #expect(throws: ToonOutlinePass.Failure.thickness) { try ToonOutlinePass(thickness: -0.01) }
        #expect(throws: ToonOutlinePass.Failure.alpha) { try ToonOutlinePass(alpha: 1.1) }
        #expect(throws: ToonOutlinePass.Failure.color) { try ToonOutlinePass(color: SIMD3(.nan, 0, 0)) }
        #expect(throws: ToonOutlinePass.Failure.thickness) { try ToonOutlinePass.Override(thickness: -0.01) }
        #expect(throws: ToonOutlinePass.Failure.alpha) { try ToonOutlinePass.Override(alpha: .nan) }
    }

    private func toonMaterial() throws -> ClassicMaterial {
        try ClassicMaterial(model: .toon(), color: SIMD3(0.3, 0.5, 0.7), ambient: SIMD3(repeating: .pi))
    }

    private func pixel(_ pixels: [SIMD4<Float>], _ x: Int) -> SIMD4<Float> {
        fixture.pixel(pixels, x: x, y: 64)
    }

    private func render(
        _ graph: SceneGraph, outline: ToonOutlinePass? = nil, depth: MeshDepthMode = .forward, orthographic: Bool = false
    ) throws -> (pixels: [SIMD4<Float>], statistics: SceneDrawStatistics) {
        let target = try fixture.makeTarget()
        let pass = fixture.pass(color: target.color, depth: target.depth, depthMode: depth)
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        let stats: SceneDrawStatistics
        if orthographic {
            let z: Float = depth == .reversed ? 0.1 : -0.1
            let bias: Float = depth == .reversed ? 1 : 0
            let clip = simd_float4x4(columns: (
                SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, z, 0), SIMD4(0, 0, bias, 1)
            ))
            stats = try graph.encode(
                pass: pass, commandBuffer: buffer, clipFromWorld: clip, cameraPosition: .zero,
                directionToCamera: SIMD3(0, 0, 1), depthMode: depth, toonOutline: outline
            )
        } else {
            stats = try graph.encode(pass: pass, commandBuffer: buffer, depthMode: depth, toonOutline: outline)
        }
        try fixture.finish(buffer)
        return (try fixture.pixels(of: target.color), stats)
    }
}
