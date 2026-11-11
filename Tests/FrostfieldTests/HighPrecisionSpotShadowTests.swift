import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct HighPrecisionSpotShadowTests {
    private let farOrigin = SIMD3<Float>(0x1p60, 0, 0x1p60)
    private let lightPosition = SIMD3<Float>(2.1, 0, -1)
    private let receiverDepth: Float = -4

    @Test("Spot caster masks, lighting, depth, and variance moments survive a shared large origin",
          arguments: [ShadowMapSampling.basic, .filtered, .variance])
    func translated(_ sampling: ShadowMapSampling) async throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let fixture = try makeScene(gpu, classic: sampling == .basic)
        let source = try spotlight(sampling)
        let nodes = [fixture.receiver, fixture.caster, fixture.outsideCaster, source]
        let offAxis = sampling != .basic

        func render(origin: SIMD3<Float>, precise: Bool, visible: Float) async throws
            -> (pixels: [SIMD4<Float>], depth: Data, moments: Data?, stats: SceneDrawStatistics) {
            try fixture.uniforms.set("Visible", to: visible)
            let graph = scene(gpu, nodes: nodes, origin: origin, precise: precise)
            let maps = try PunctualShadowMaps(queue: gpu.queue, resolution: 96, spotCapacity: 1, pointCapacity: 0)
            let frame = try renderFrame(gpu, graph: graph, maps: maps, offAxis: offAxis)
            let depth = try await read(maps.texture, gpu: gpu)
            let moments: Data?
            if let image = maps.shadowMoments {
                moments = try await read(image, gpu: gpu)
            } else {
                moments = nil
            }
            return (frame.pixels, depth, moments, frame.statistics)
        }

        let near = try await render(origin: .zero, precise: true, visible: 1)
        let moved = try await render(origin: farOrigin, precise: true, visible: 1)
        let cleared = try await render(origin: farOrigin, precise: true, visible: 0)
        let ordinary = try await render(origin: .zero, precise: false, visible: 1)
        #expect(moved.pixels == near.pixels)
        #expect(moved.depth == near.depth)
        #expect(moved.moments == near.moments)
        #expect(moved.depth.contains { $0 != 0 })
        #expect(cleared.depth.allSatisfy { $0 == 0 })
        if sampling == .variance {
            #expect(try #require(moved.moments).contains { $0 != 0 })
            #expect(try #require(cleared.moments).allSatisfy { $0 == 0 })
        } else {
            #expect(moved.moments == nil)
        }
        #expect(moved.stats.shadowPassCount == 1)
        #expect(moved.stats.shadowDrawCount == 1 && moved.stats.culledShadowCount == 1)

        let center = try centerPixel(gpu, offAxis: offAxis)
        let open = gpu.pixel(cleared.pixels, x: center.x, y: center.y)
        let shadow = gpu.pixel(moved.pixels, x: center.x, y: center.y)
        #expect(open.y > 0.2 && shadow.y < open.y * 0.1, "Open \\(open), shadow \\(shadow)")
        #expect(isNearly(gpu.pixel(ordinary.pixels, x: center.x, y: center.y), shadow, tolerance: 0.003))
        let lit = gpu.pixel(moved.pixels, x: center.x - 14, y: center.y)
        let ordinaryLit = gpu.pixel(ordinary.pixels, x: center.x - 14, y: center.y)
        #expect(lit.y > 0.1 && isNearly(ordinaryLit, lit, tolerance: 0.003))
    }

    @Test("Precise spotlight normal bias survives at 2⁶⁰ and changes the lit pixel")
    func normalBias() throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let fixture = try makeScene(gpu)
        var shallow = fixture.caster
        shallow.transform.translation = SIMD3(-0.065, 0, -3.95)
        func render(origin: SIMD3<Float>, bias: Float) throws -> SIMD4<Float> {
            let graph = try scene(
                gpu, nodes: [fixture.receiver, shallow, spotlight(.basic, normalBias: bias)],
                origin: origin, precise: true
            )
            let maps = try PunctualShadowMaps(queue: gpu.queue, resolution: 64, spotCapacity: 1, pointCapacity: 0)
            let frame = try gpu.render(graph, punctualShadowMaps: maps)
            return gpu.pixel(frame.pixels, x: gpu.side / 2, y: gpu.side / 2)
        }
        let low = try render(origin: farOrigin, bias: 0)
        let high = try render(origin: farOrigin, bias: 12)
        #expect(low.y < 0.01 && high.y > 0.2, "Low \\(low), high \\(high)")
        #expect(low == (try render(origin: .zero, bias: 0)))
        #expect(high == (try render(origin: .zero, bias: 12)))
    }

    @Test("A lit material override receives precise spot shadows on an authored unlit mesh")
    func overriddenReceiver() throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let fixture = try makeScene(gpu)
        let source = try spotlight(.variance)
        let reference = scene(gpu, nodes: [fixture.receiver, fixture.caster, source], origin: .zero, precise: true)
        let original = try #require(fixture.receiver.mesh)
        var receiver = fixture.receiver
        receiver.mesh = try gpu.unlit(original.geometry)
        var translated = scene(gpu, nodes: [receiver, fixture.caster, source], origin: farOrigin, precise: true)
        translated.overrideMaterial = try SceneMaterialOverride(material: original.material, excluding: [fixture.caster.id])
        let maps = try PunctualShadowMaps(queue: gpu.queue, resolution: 96, spotCapacity: 1, pointCapacity: 0)
        let expected = try renderFrame(gpu, graph: reference, maps: maps, offAxis: true)
        let actual = try renderFrame(gpu, graph: translated, maps: maps, offAxis: true)
        #expect(actual.pixels == expected.pixels)
        #expect(actual.statistics.shadowDrawCount == 1)
    }

    @Test("A cached spot follows child placement changes and survives a viewing-camera rebase")
    func cachedPlacement() async throws {
        let gpu = try GeometryRenderFixture()
        let fixture = try makeScene(gpu)
        let source = try spotlight(.variance)
        var graph = scene(gpu, nodes: [fixture.receiver, fixture.caster, source], origin: farOrigin, precise: true)
        let maps = try PunctualShadowMaps(
            queue: gpu.queue, resolution: 64, spotCapacity: 1, pointCapacity: 0, updateMode: .onDemand
        )
        let first = try gpu.render(graph, punctualShadowMaps: maps)
        let initialDepth = try await read(maps.texture, gpu: gpu)
        let moments = try #require(maps.shadowMoments)
        let initialMoments = try await read(moments, gpu: gpu)
        #expect(first.statistics.shadowPassCount == 1 && initialDepth.contains { $0 != 0 })

        graph.cameraTransform.translation.y = 0.25
        let newView = try gpu.render(graph, punctualShadowMaps: maps)
        #expect(newView.statistics.shadowPassCount == 0 && newView.pixels != first.pixels)
        #expect(try await read(maps.texture, gpu: gpu) == initialDepth)

        graph.roots[0].children[2].transform.translation.x += 0.25
        let moved = try gpu.render(graph, punctualShadowMaps: maps)
        #expect(moved.statistics.shadowPassCount == 1)
        #expect(try await read(maps.texture, gpu: gpu) != initialDepth)
        #expect(try await read(moments, gpu: gpu) != initialMoments)
        #expect(try gpu.render(graph, punctualShadowMaps: maps).statistics.shadowPassCount == 0)

        try fixture.uniforms.set("Visible", to: Float(0))
        #expect(maps.requestUpdate(for: source.id))
        #expect(try gpu.render(graph, punctualShadowMaps: maps).statistics.shadowPassCount == 1)
        #expect(try await read(maps.texture, gpu: gpu).allSatisfy { $0 == 0 })
        #expect(try await read(moments, gpu: gpu).allSatisfy { $0 == 0 })
    }

    private func makeScene(_ gpu: GeometryRenderFixture, classic: Bool = false) throws
        -> (receiver: SceneNode, caster: SceneNode, outsideCaster: SceneNode, uniforms: ShaderUniforms) {
        let receiving: MeshMaterial = if classic {
            .classic(try ClassicMaterial(model: .lambert))
        } else {
            .physicallyBased(try PhysicallyBasedMaterial(
                roughness: 0.8, lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque
            ))
        }
        let receiver = try gpu.unlit(GeometryRenderFixture.square(2.2), material: receiving)
        let surface = try SurfaceShaderGraph { builder in
            .init(
                positionOffset: builder.constant(SIMD3<Float>(0.1, 0, 0)),
                mask: builder.uniform("Visible", Float(1))
            )
        }
        let casting = try PhysicallyBasedMaterial(alphaMode: .opaque, isDoubleSided: true, surfaceGraph: surface)
        let uniforms = try #require(casting.surfaceUniforms)
        let caster = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(0.32),
            material: .physicallyBased(casting), renderState: .init(depthWrite: false, colorWriteMask: [], side: .double)
        )
        let outside = try gpu.unlit(GeometryRenderFixture.square(0.32))
        return (
            SceneNode(transform: .init(translation: SIMD3(0, 0, receiverDepth)), mesh: receiver, castsShadow: false),
            SceneNode(transform: .init(translation: SIMD3(0.6, 0, -3)), mesh: caster),
            SceneNode(transform: .init(translation: SIMD3(50, 0, -3)), mesh: outside),
            uniforms
        )
    }

    private func spotlight(_ sampling: ShadowMapSampling, normalBias: Float = 1.5) throws -> SceneNode {
        let aim = simd_normalize(SIMD3<Float>(0, 0, receiverDepth) - lightPosition)
        let shadow = try PunctualLight.Shadow(
            normalBias: normalBias, filterRadius: 1, nearPlane: 0.2, sampling: sampling, varianceBlurRadius: 1
        )
        return SceneNode(
            transform: .init(translation: lightPosition, rotation: simd_quatf(from: SIMD3(0, 0, -1), to: aim)),
            light: try PunctualLight(kind: .spot(innerConeAngle: 0.48, outerConeAngle: 0.7), intensity: 40, shadow: shadow)
        )
    }

    private func scene(
        _ gpu: GeometryRenderFixture, nodes: [SceneNode], origin: SIMD3<Float>, precise: Bool
    ) -> SceneGraph {
        let anchor = SceneTransform(translation: origin)
        var graph = SceneGraph(
            roots: [SceneNode(transform: anchor, children: nodes)], camera: gpu.camera, cameraTransform: anchor
        )
        graph.highPrecision = precise
        return graph
    }

    private func projection(_ gpu: GeometryRenderFixture, offAxis: Bool) throws -> simd_float4x4 {
        var projection = try gpu.camera.projection(aspectRatio: 1)
        if offAxis {
            projection[2].x = 0.18
            projection[2].y = -0.1
        }
        return projection
    }

    private func centerPixel(_ gpu: GeometryRenderFixture, offAxis: Bool) throws -> SIMD2<Int> {
        let clip = try projection(gpu, offAxis: offAxis) * SIMD4<Float>(0, 0, receiverDepth, 1)
        return SIMD2(Int((0.5 + 0.5 * clip.x / clip.w) * Float(gpu.side)),
                     Int((0.5 - 0.5 * clip.y / clip.w) * Float(gpu.side)))
    }

    private func renderFrame(
        _ gpu: GeometryRenderFixture, graph: SceneGraph, maps: PunctualShadowMaps, offAxis: Bool
    ) throws -> (pixels: [SIMD4<Float>], statistics: SceneDrawStatistics) {
        if !offAxis { return try gpu.render(graph, punctualShadowMaps: maps) }
        let camera = try ShaderGraphCamera(
            viewMatrix: simd_inverse(graph.cameraTransform.matrix), projectionMatrix: projection(gpu, offAxis: true),
            near: gpu.camera.near, far: gpu.camera.far
        )
        let target = try gpu.makeTarget()
        let command = try #require(gpu.queue.makeCommandBuffer())
        let statistics = try graph.encode(
            pass: gpu.pass(color: target.color, depth: target.depth), commandBuffer: command,
            clipFromWorld: camera.projectionMatrix * camera.viewMatrix, cameraPosition: graph.cameraTransform.translation,
            punctualShadowMaps: maps, graphCamera: camera
        )
        try gpu.finish(command)
        return (try gpu.pixels(of: target.color), statistics)
    }

    private func read(_ texture: any MTLTexture, gpu: GeometryRenderFixture) async throws -> Data {
        try await TextureTransfer.read(from: texture, commandQueue: gpu.queue).bytes
    }
}
