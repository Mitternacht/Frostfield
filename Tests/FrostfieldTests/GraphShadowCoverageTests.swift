import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct GraphShadowCoverageTests {
    enum LightKind: CaseIterable, Sendable { case directional, spot, point }

    private struct ShadowFixture {
        let directional: DirectionalShadowMap?
        let punctual: PunctualShadowMaps?
        let light: SceneNode?
        let texture: any MTLTexture
        let projection: simd_float4x4
        let slice: Int
        let clearedDepth: Float
    }

    private let gpu: PathTestFixture
    private let side = 33
    private let resolution = 128

    init() throws { gpu = try PathTestFixture() }

    @Test("Graph opacity and masks change depth and lit receivers in directional, spot and point shadows",
          arguments: LightKind.allCases, [ShadowMapSampling.filtered, .basic])
    func graphCaster(_ kind: LightKind, sampling: ShadowMapSampling) async throws {
        let shader = try SurfaceShaderGraph { graph in
            let map = graph.texture("Coverage", content: .data, filtering: .nearest)
            let lightViewport = graph.screenSize.x.greaterThan(graph.constant(Float(64)))
            let projected = graph.screenUV.y.lessThan(graph.constant(Float(0.75)))
            let moved = graph.worldPosition.x.greaterThan(graph.constant(Float(0.58)))
            return .init(
                baseColor: graph.cameraPosition, // Camera inputs outside coverage stay valid.
                positionOffset: graph.constant(SIMD3<Float>(0.3, 0, 0)),
                opacity: map.sample(graph.uv0).x * graph.parameters.w,
                alphaTest: graph.uniform("Cutoff", Float(0.25)),
                mask: graph.uniform("Visible", Float(1)) * lightViewport * projected * moved
            )
        }
        let material = try PhysicallyBasedMaterial(
            baseColorTexture: texture(SIMD4(1, 1, 1, 0.8)), alphaMode: .mask(cutoff: 0.8),
            isDoubleSided: true, surfaceGraph: shader, surfaceTextures: ["Coverage": try texture(SIMD4(0.5, 0, 0, 1))],
            alphaTexture: .init(texture: texture(SIMD4(1, 0.75, 1, 1)))
        )
        let renderer = try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float,
                                        geometry: .plane(width: 0.38, height: 0.55),
                                        material: .physicallyBased(material), supportsInstancing: true)
        let instances = try InstanceSet(device: gpu.device, transforms: [SceneTransform(translation: SIMD3(0.25, 0, 0)).matrix])
        let caster = SceneNode(transform: .init(translation: SIMD3(0.15, 0, -3)), mesh: renderer,
                               materialParameters: SIMD4(1, 1, 1, 0.75), instances: instances)
        let shadows = try shadowFixture(kind, sampling: sampling)
        var roots = [try receiver(directional: kind == .directional), caster]
        if let light = shadows.light { roots.append(light) }
        let graph = SceneGraph(roots: roots, camera: gpu.camera)
        let uniforms = try #require(material.surfaceUniforms)
        let point = SIMD3<Float>(0.7, 0, -3)

        let open = try render(graph, shadows: shadows)
        let openDepth = try await depth(in: shadows, point: point)
        // Base alpha 0.8 × alpha map 0.75 × graph map 0.5 × node parameter 0.75 = 0.225.
        try uniforms.set("Cutoff", to: 0.2)
        let occluded = try render(graph, shadows: shadows)
        let occludedDepth = try await depth(in: shadows, point: point)
        try uniforms.set("Visible", to: 0)
        let masked = try render(graph, shadows: shadows)
        let maskedDepth = try await depth(in: shadows, point: point)

        #expect(open > 0.2 && occluded < open * 0.12 && abs(masked - open) < 0.003,
                "\(kind): open \(open), occluded \(occluded), masked \(masked)")
        #expect(openDepth == shadows.clearedDepth && maskedDepth == openDepth)
        let clip = shadows.projection * SIMD4(point, 1)
        #expect(abs(occludedDepth - clip.z / clip.w) < 0.02 && occludedDepth != openDepth)
    }

    @Test("Basic shadows have binary receiver visibility and retain explicit depth bias",
          arguments: LightKind.allCases)
    func hardEdgesAndBias(_ kind: LightKind) throws {
        let side = 129
        let receiver = try receiver(directional: kind == .directional)
        let blocker = SceneNode(
            transform: .init(translation: SIMD3(0.7, 0, -3)),
            mesh: try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float,
                                   geometry: .plane(width: 0.62, height: 0.8)),
            materialParameters: SIMD4(1, 0, 0, 1)
        )
        let basic = try shadowFixture(kind, sampling: .basic, size: 32)
        let filtered = try shadowFixture(kind, size: 32)
        let biased = try shadowFixture(kind, sampling: .basic, bias: kind == .directional ? 0.4 : 1.6, size: 32)
        func row(_ shadows: ShadowFixture, castShadow: Bool = true) throws -> [SIMD4<Float>] {
            let roots = [receiver] + (castShadow ? [blocker] : []) + [shadows.light].compactMap { $0 }
            let graph = SceneGraph(roots: roots, camera: gpu.camera)
            let pixels = try renderPixels(graph, shadows: shadows, side: side)
            return Array(pixels[(side / 2 * side)..<(side / 2 * side + side)])
        }
        let open = try row(basic, castShadow: false)
        let hard = try row(basic)
        let smooth = try row(filtered)
        let offset = try row(biased)
        let center = side / 2
        #expect(open[center].y > 0.2 && hard[center].y < open[center].y * 0.02)
        #expect(offset[center].y > open[center].y * 0.98)

        // The blocker is red. Compare only white receiver pixels in the two renders.
        let receiverPixels = (side / 4...3 * side / 4).filter { x in
            hard[x].x == hard[x].y && smooth[x].x == smooth[x].y && open[x].y > 0.1
        }
        let hardVisibility = receiverPixels.map { hard[$0].y / open[$0].y }
        let smoothVisibility = receiverPixels.map { smooth[$0].y / open[$0].y }
        #expect(hardVisibility.contains { $0 < 0.02 } && hardVisibility.contains { $0 > 0.98 })
        #expect(hardVisibility.allSatisfy { $0 < 0.02 || $0 > 0.98 }, "\(kind) basic: \(hardVisibility)")
        #expect(smoothVisibility.contains { $0 > 0.05 && $0 < 0.95 }, "\(kind) filtered: \(smoothVisibility)")
    }

    @Test("On-demand shadows keep stale depth and lit pixels until requested or their layout changes",
          arguments: LightKind.allCases)
    func onDemandUpdates(_ kind: LightKind) async throws {
        let shadows = try shadowFixture(kind, sampling: .basic, updateMode: .onDemand)
        let receiver = try receiver(directional: kind == .directional)
        let point = SIMD3<Float>(0.7, 0, -3)
        let caster = SceneNode(
            transform: .init(translation: point),
            mesh: try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float,
                                   geometry: .plane(width: 0.62, height: 0.8)),
            materialParameters: SIMD4(1, 0, 0, 1)
        )
        var graph = SceneGraph(roots: [receiver, caster] + [shadows.light].compactMap { $0 }, camera: gpu.camera)
        let first = try renderFrame(graph, shadows: shadows, side: side)
        let written = try await depth(in: shadows, point: point)
        let center = side / 2 * side + side / 2
        let expectedPasses = kind == .point ? 6 : 1
        #expect(first.statistics.shadowPassCount == expectedPasses && first.statistics.shadowDrawCount > 0)
        #expect(written != shadows.clearedDepth && first.pixels[center].y < 0.005)

        graph.roots[1].transform.translation.x += 2
        let cached = try renderFrame(graph, shadows: shadows, side: side)
        #expect(cached.statistics.shadowPassCount == 0 && cached.statistics.shadowDrawCount == 0)
        #expect(cached.pixels[center] == first.pixels[center])
        #expect(try await depth(in: shadows, point: point) == written)

        shadows.directional?.requestUpdate()
        shadows.punctual?.requestUpdate()
        let rebuilt = try renderFrame(graph, shadows: shadows, side: side)
        #expect(rebuilt.statistics.shadowPassCount == expectedPasses && rebuilt.pixels[center].y > 0.2)
        #expect(try await depth(in: shadows, point: point) == shadows.clearedDepth)
        #expect(try renderFrame(graph, shadows: shadows, side: side).statistics.shadowPassCount == 0)

        if let map = shadows.directional {
            try map.update(directionToLight: SIMD3(0.7, 0, 1), center: SIMD3(0.125, 0, -3.5), halfExtent: 2)
        } else if let light = shadows.light {
            // Same projection and settings, but a different node now owns the slice.
            graph.roots[2] = SceneNode(transform: light.transform, light: light.light)
        }
        #expect(try renderFrame(graph, shadows: shadows, side: side).statistics.shadowPassCount == expectedPasses)
    }

    @Test("Selective punctual refresh keeps stable slots, unrelated depth and six-face atomicity")
    func selectivePunctualUpdates() async throws {
        let maps = try PunctualShadowMaps(
            queue: gpu.queue, resolution: resolution, spotCapacity: 2, pointCapacity: 1, updateMode: .onDemand
        )
        let settings = try PunctualLight.Shadow(normalBias: 0, sampling: .basic)
        let outer: Float = 0.6
        func light(_ kind: PunctualLight.Kind, at position: SIMD3<Float>, color: SIMD3<Float>) throws -> SceneNode {
            let aim = simd_normalize(SIMD3<Float>(0, 0, -4) - position)
            let transform = SceneTransform(translation: position, rotation: simd_quatf(from: SIMD3(0, 0, -1), to: aim))
            return SceneNode(transform: transform, light: try PunctualLight(kind: kind, color: color, intensity: 40, shadow: settings))
        }
        let spot: PunctualLight.Kind = .spot(innerConeAngle: 0.45, outerConeAngle: outer)
        let red = try light(spot, at: SIMD3(2.1, 0, -1), color: SIMD3(1, 0, 0))
        let blue = try light(spot, at: SIMD3(2.35, 0.1, -1), color: SIMD3(0, 0, 1))
        let green = try light(.point(), at: SIMD3(2.1, 0, -1), color: SIMD3(0, 1, 0))
        let caster = SceneNode(
            transform: .init(translation: SIMD3(0.7, 0, -3)),
            mesh: try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float,
                                   geometry: .plane(width: 0.62, height: 0.8))
        )
        var graph = SceneGraph(roots: [try receiver(directional: false), caster, red, blue, green], camera: gpu.camera)
        let shadows = ShadowFixture(
            directional: nil, punctual: maps, light: nil, texture: maps.texture,
            projection: matrix_identity_float4x4, slice: 0, clearedDepth: 0
        )
        let center = side / 2 * side + side / 2
        func slice(_ index: Int) async throws -> Data {
            try await TextureTransfer.read(from: maps.texture, slice: index, commandQueue: gpu.queue).bytes
        }
        #expect(!maps.requestUpdate(for: red.id))
        let initial = try renderFrame(graph, shadows: shadows, side: side)
        #expect(initial.statistics.shadowPassCount == 8)
        #expect(initial.pixels[center].x < 0.005 && initial.pixels[center].y < 0.005)
        let initialRed = try await slice(0), initialBlue = try await slice(1), initialGreen = try await slice(7)
        #expect(initialRed != initialBlue)

        graph.roots.swapAt(2, 3)
        graph.roots[1].transform.translation.x += 2
        let cached = try renderFrame(graph, shadows: shadows, side: side)
        #expect(cached.statistics.shadowPassCount == 0 && cached.pixels[center] == initial.pixels[center])
        let cachedRed = try await slice(0), cachedBlue = try await slice(1)
        #expect(cachedRed == initialRed && cachedBlue == initialBlue)

        #expect(maps.requestUpdate(for: red.id))
        #expect(!maps.requestUpdate(for: UUID()))
        let target = try renderTarget(side: side)
        let writer = try #require(gpu.queue.makeCommandBuffer())
        #expect(try graph.encode(pass: target.pass, commandBuffer: writer, punctualShadowMaps: maps).shadowPassCount == 1)
        let competing = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: ShadowMapUpdateError.commandBufferInUse) {
            try graph.encode(pass: target.pass, commandBuffer: competing, punctualShadowMaps: maps)
        }
        try gpu.finish(writer)
        let redRefreshed = try gpu.readback(target.color)[center]
        #expect(redRefreshed.x > 0.2 && redRefreshed.y < 0.005)
        let refreshedRed = try await slice(0)
        #expect(refreshedRed != initialRed)
        let unchangedBlue = try await slice(1), unchangedGreen = try await slice(7)
        #expect(unchangedBlue == initialBlue && unchangedGreen == initialGreen)

        // Deliberately submit an incomplete point update. Any potentially
        // overwritten face is invalid, while the two spots are still usable.
        #expect(maps.requestUpdate(for: green.id))
        let resolved = Dictionary(uniqueKeysWithValues: try graph.resolve().compactMap { node in
            node.light.map { (node.id, $0) }
        })
        let spotTangent = maps.projectionTangent(covering: tan(Double(outer)), shadow: settings)
        let pointTangent = maps.projectionTangent(covering: 1, shadow: settings)
        let spotGroups: [ShadowMapUpdateCache.Layout.Group] = try [red, blue].enumerated().map { index, light in
            let data = try #require(resolved[light.id])
            let view = try PunctualShadowMaps.projection(origin: data.position, forward: data.direction,
                                                         tangent: spotTangent, near: settings.nearPlane)
            return .init(lightID: light.id, slices: index..<(index + 1), views: [view][...])
        }
        let pointViews = try PunctualShadowMaps.faceDirections.map {
            try PunctualShadowMaps.projection(origin: green.transform.translation, forward: $0,
                                              tangent: pointTangent, near: settings.nearPlane)
        }
        let layout = ShadowMapUpdateCache.Layout(groups: spotGroups + [.init(lightID: green.id, slices: 2..<8, views: pointViews[...])])
        let partial = try #require(gpu.queue.makeCommandBuffer())
        let use = try maps.beginUse(commandBuffer: partial, layout: layout)
        #expect((0..<8).filter { use.writes(slice: $0) } == Array(2..<8))
        try maps.makeEncoder(slice: 7, commandBuffer: partial).endEncoding()
        try gpu.finish(partial)

        let viewer = try TextureDebugPass(device: gpu.device)
        let inspection = try #require(gpu.queue.makeCommandBuffer())
        for face in 2..<8 {
            #expect(throws: ShadowMapUpdateError.depthUnavailable) {
                try viewer.encode(source: .punctualShadow(maps, slice: face), destination: target.color, commandBuffer: inspection)
            }
        }
        try viewer.encode(source: .punctualShadow(maps, slice: 0), destination: target.color, commandBuffer: inspection)
        try gpu.finish(inspection)

        let recovered = try renderFrame(graph, shadows: shadows, side: side)
        #expect(recovered.statistics.shadowPassCount == 6 && recovered.pixels[center].y > 0.2)
        let recoveredRed = try await slice(0), recoveredBlue = try await slice(1), recoveredGreen = try await slice(7)
        #expect(recoveredRed == refreshedRed && recoveredBlue == initialBlue && recoveredGreen != initialGreen)

        graph.roots[2].transform.translation.y += 0.125
        #expect(try renderFrame(graph, shadows: shadows, side: side).statistics.shadowPassCount == 1)
        maps.requestUpdate()
        #expect(try renderFrame(graph, shadows: shadows, side: side).statistics.shadowPassCount == 8)
    }

    @Test("Punctual lights can refresh automatically or independently within one map set",
          arguments: [ShadowMapUpdateMode.automatic, .onDemand])
    func mixedPunctualUpdates(_ defaultMode: ShadowMapUpdateMode) async throws {
        let shadow = try PunctualLight.Shadow(normalBias: 0, sampling: .basic)
        let lightPosition = SIMD3<Float>(2.1, 0, -1)
        let direction = simd_normalize(SIMD3<Float>(0, 0, -4) - lightPosition)
        let spot = SceneNode(
            transform: .init(translation: lightPosition, rotation: simd_quatf(from: SIMD3(0, 0, -1), to: direction)),
            light: try PunctualLight(kind: .spot(innerConeAngle: 0.45, outerConeAngle: 0.6),
                                     color: SIMD3(1, 0, 0), intensity: 40, shadow: shadow)
        )
        let point = SceneNode(
            transform: .init(translation: lightPosition),
            light: try PunctualLight(kind: .point(), color: SIMD3(0, 1, 0), intensity: 40, shadow: shadow)
        )
        let overrides: [UUID: ShadowMapUpdateMode] = defaultMode == .automatic ? [spot.id: .onDemand] : [point.id: .automatic]
        let maps = try PunctualShadowMaps(
            queue: gpu.queue, resolution: resolution, spotCapacity: 1, pointCapacity: 1,
            updateMode: defaultMode, lightUpdateModes: overrides
        )
        #expect(maps.updateMode(for: spot.id) == .onDemand)
        #expect(maps.updateMode(for: point.id) == .automatic)
        let caster = SceneNode(
            transform: .init(translation: SIMD3(0.7, 0, -3)),
            mesh: try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float,
                                   geometry: .plane(width: 0.62, height: 0.8))
        )
        var graph = SceneGraph(roots: [try receiver(directional: false), caster, spot, point], camera: gpu.camera)
        let fixture = ShadowFixture(
            directional: nil, punctual: maps, light: nil, texture: maps.texture,
            projection: matrix_identity_float4x4, slice: 0, clearedDepth: 0
        )
        let center = side / 2 * side + side / 2
        let initial = try renderFrame(graph, shadows: fixture, side: side)
        #expect(initial.statistics.shadowPassCount == 7)
        #expect(initial.pixels[center].x < 0.005 && initial.pixels[center].y < 0.005)
        let cachedSpot = try await TextureTransfer.read(from: maps.texture, slice: 0, commandQueue: gpu.queue).bytes

        let first = try renderTarget(side: side)
        let second = try renderTarget(side: side)
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        graph.roots[1].transform.translation.x += 2
        let moved = try graph.encode(pass: first.pass, commandBuffer: buffer, punctualShadowMaps: maps)
        graph.roots[1].transform.translation.x -= 2
        let restored = try graph.encode(pass: second.pass, commandBuffer: buffer, punctualShadowMaps: maps)
        #expect(moved.shadowPassCount == 6)
        #expect(restored.shadowPassCount == 6)
        try gpu.finish(buffer)
        let movedPixel = try gpu.readback(first.color)[center]
        let restoredPixel = try gpu.readback(second.color)[center]
        #expect(movedPixel.x < 0.005 && movedPixel.y > 0.2)
        #expect(restoredPixel == initial.pixels[center])
        let unchangedSpot = try await TextureTransfer.read(from: maps.texture, slice: 0, commandQueue: gpu.queue).bytes
        #expect(unchangedSpot == cachedSpot)

        try maps.setUpdateMode(.onDemand, for: point.id)
        graph.roots[1].transform.translation.x += 2
        #expect(try renderFrame(graph, shadows: fixture, side: side).statistics.shadowPassCount == 0)
        #expect(maps.requestUpdate(for: spot.id))
        let refreshed = try renderFrame(graph, shadows: fixture, side: side)
        #expect(refreshed.statistics.shadowPassCount == 1)
        #expect(refreshed.pixels[center].x > 0.2 && refreshed.pixels[center].y < 0.005)
        try maps.setUpdateMode(nil, for: point.id)
        #expect(maps.updateMode(for: point.id) == defaultMode)
        let expectedPasses = defaultMode == .automatic ? 6 : 0
        #expect(try renderFrame(graph, shadows: fixture, side: side).statistics.shadowPassCount == expectedPasses)
    }

    @Test("The automatic fast path keeps its existing behavior and reports unavailable overrides")
    func automaticShadowPolicy() throws {
        let maps = try PunctualShadowMaps(queue: gpu.queue, resolution: resolution)
        let id = UUID()
        #expect(maps.updateMode(for: id) == .automatic)
        #expect(!maps.requestUpdate(for: id))
        #expect(throws: ShadowMapUpdateError.lightUpdateModesUnavailable) { try maps.setUpdateMode(.onDemand, for: id) }
        let optedIn = try PunctualShadowMaps(queue: gpu.queue, resolution: resolution, lightUpdateModes: [:])
        try optedIn.setUpdateMode(.onDemand, for: id)
        #expect(optedIn.updateMode(for: id) == .onDemand)
    }

    @Test("Manual shadow updates preserve command ordering and a request made during a pending use")
    func onDemandLeases() throws {
        let shadows = try shadowFixture(.directional, updateMode: .onDemand)
        let map = try #require(shadows.directional)
        let graph = SceneGraph(roots: [try receiver(directional: true)], camera: gpu.camera)
        let target = try renderTarget(side: side)
        let first = try #require(gpu.queue.makeCommandBuffer())
        let next = try #require(gpu.queue.makeCommandBuffer())
        #expect(try graph.encode(pass: target.pass, commandBuffer: first, shadowMap: map).shadowPassCount == 1)
        #expect(try graph.encode(pass: target.pass, commandBuffer: first, shadowMap: map).shadowPassCount == 0)
        map.requestUpdate()
        #expect(throws: ShadowMapUpdateError.commandBufferInUse) {
            try graph.encode(pass: target.pass, commandBuffer: next, shadowMap: map)
        }
        try gpu.finish(first)
        #expect(try graph.encode(pass: target.pass, commandBuffer: next, shadowMap: map).shadowPassCount == 1)
        try gpu.finish(next)

        // Even a cached read holds the texture until its command buffer completes.
        let reader = try #require(gpu.queue.makeCommandBuffer())
        let writer = try #require(gpu.queue.makeCommandBuffer())
        #expect(try graph.encode(pass: target.pass, commandBuffer: reader, shadowMap: map).shadowPassCount == 0)
        map.requestUpdate()
        #expect(throws: ShadowMapUpdateError.commandBufferInUse) {
            try graph.encode(pass: target.pass, commandBuffer: writer, shadowMap: map)
        }
        try gpu.finish(reader)
        #expect(try graph.encode(pass: target.pass, commandBuffer: writer, shadowMap: map).shadowPassCount == 1)
        try gpu.finish(writer)
    }

    @Test("An abandoned or partially encoded manual update cannot become the cached map")
    func onDemandIncomplete() throws {
        let map = try DirectionalShadowMap(
            queue: gpu.queue, resolution: 32, directionToLight: SIMD3(0, 0, 1),
            cascades: .init(count: 2), updateMode: .onDemand
        )
        try map.fitCascades(camera: gpu.camera, cameraTransform: .init(), aspectRatio: 1)
        let graph = SceneGraph(roots: [try receiver(directional: true)], camera: gpu.camera)
        let target = try renderTarget(side: side)
        try autoreleasepool {
            let discarded = try #require(gpu.queue.makeCommandBuffer())
            #expect(try graph.encode(pass: target.pass, commandBuffer: discarded, shadowMap: map).shadowPassCount == 2)
        }
        let first = try #require(gpu.queue.makeCommandBuffer())
        #expect(try graph.encode(pass: target.pass, commandBuffer: first, shadowMap: map).shadowPassCount == 2)
        try gpu.finish(first)

        map.requestUpdate()
        let partial = try #require(gpu.queue.makeCommandBuffer())
        let use = try map.beginUse(commandBuffer: partial)
        #expect(use.writesDepth)
        let depth = try map.makeEncoder(commandBuffer: partial)
        depth.endEncoding()
        // Simulate a failure after only the first cascade was encoded.
        try gpu.finish(partial)
        let viewer = try TextureDebugPass(device: gpu.device)
        let rebuilt = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: ShadowMapUpdateError.depthUnavailable) {
            try viewer.encode(source: .directionalShadow(map), destination: target.color, commandBuffer: rebuilt)
        }
        #expect(try graph.encode(pass: target.pass, commandBuffer: rebuilt, shadowMap: map).shadowPassCount == 2)
        try gpu.finish(rebuilt)
        let cached = try #require(gpu.queue.makeCommandBuffer())
        #expect(try graph.encode(pass: target.pass, commandBuffer: cached, shadowMap: map).shadowPassCount == 0)
        try gpu.finish(cached)
    }

    @Test("An opacity-only graph uses the authored material cutoff in the shadow map")
    func authoredCutoff() async throws {
        let shader = try SurfaceShaderGraph { graph in .init(opacity: graph.uniform("Opacity", Float(0.25))) }
        let material = try PhysicallyBasedMaterial(alphaMode: .mask(cutoff: 0.5), isDoubleSided: true, surfaceGraph: shader)
        let renderer = try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float,
                                        geometry: .plane(width: 0.45, height: 0.55), material: .physicallyBased(material))
        let point = SIMD3<Float>(0.7, 0, -3)
        let graph = SceneGraph(roots: [try receiver(directional: true), SceneNode(transform: .init(translation: point), mesh: renderer)],
                               camera: gpu.camera)
        let shadows = try shadowFixture(.directional)
        let open = try render(graph, shadows: shadows)
        let openDepth = try await depth(in: shadows, point: point)
        #expect(openDepth == 1)
        try #require(material.surfaceUniforms).set("Opacity", to: 0.5)
        let closed = try render(graph, shadows: shadows)
        let closedDepth = try await depth(in: shadows, point: point)
        #expect(closedDepth < 1)
        #expect(open > 0.2 && closed < open * 0.12)
    }

    @Test("A graph shadow mask replaces the mask in directional, spot and point shadow maps", arguments: LightKind.allCases)
    func shadowMaskPrecedence(_ kind: LightKind) throws {
        let shader = try SurfaceShaderGraph { graph in
            .init(
                mask: graph.uniform("Visible", Float(1)),
                castShadowColor: graph.constant(SIMD4<Float>(0.5, 0.5, 0.5, 1)),
                castShadowMask: graph.uniform("Casts", Float(1))
            )
        }
        let material = try PhysicallyBasedMaterial(alphaMode: .opaque, isDoubleSided: true, surfaceGraph: shader)
        let renderer = try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float,
                                        geometry: .plane(width: 0.62, height: 0.8), material: .physicallyBased(material))
        let shadows = try shadowFixture(kind, sampling: .basic)
        let graph = SceneGraph(
            roots: [try receiver(directional: kind == .directional), SceneNode(transform: .init(translation: SIMD3(0.7, 0, -3)), mesh: renderer)]
                + [shadows.light].compactMap { $0 },
            camera: gpu.camera
        )
        let uniforms = try #require(material.surfaceUniforms)
        func shadow(visible: Float, casts: Float) throws -> Float {
            try uniforms.set("Visible", to: visible)
            try uniforms.set("Casts", to: casts)
            return try render(graph, shadows: shadows)
        }
        let open = try shadow(visible: 1, casts: 0)
        #expect(open > 0.2)
        // The shadow mask decides in every raster shadow pass: a visible mesh can opt out and a hidden mesh can still cast.
        let cast = try shadow(visible: 1, casts: 1)
        let hidden = try shadow(visible: 0, casts: 1)
        let neither = try shadow(visible: 0, casts: 0)
        #expect(cast < open * 0.12, "\(kind): cast \(cast), open \(open)")
        #expect(hidden < open * 0.12, "\(kind): hidden \(hidden), open \(open)")
        #expect(abs(neither - open) < 0.003, "\(kind): neither \(neither), open \(open)")
    }

    @Test("A camera-dependent coverage node is rejected only when the mesh casts a raster shadow")
    func unsupportedCamera() throws {
        let shader = try SurfaceShaderGraph { graph in
            .init(mask: graph.cameraPosition.z.greaterThan(graph.constant(Float(-1))))
        }
        let renderer = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: .plane(),
            material: .physicallyBased(.init(alphaMode: .opaque, surfaceGraph: shader))
        )
        let transform = SceneTransform(translation: SIMD3(0.7, 0, -3))
        let caster = SceneNode(transform: transform, mesh: renderer)
        let target = try receiver(directional: true)
        let shadows = try shadowFixture(.directional)
        let graph = SceneGraph(roots: [target, caster], camera: gpu.camera)
        let input = try #require(shader.nodes.firstIndex { if case .input(.cameraPosition) = $0 { true } else { false } })
        #expect(throws: MeshRenderer.Failure.graphShadowCameraInput(caster.id, input)) {
            try render(graph, shadows: shadows)
        }
        let nonCaster = SceneNode(transform: transform, mesh: renderer, castsShadow: false)
        let unshadowed = try render(SceneGraph(roots: [target, nonCaster], camera: gpu.camera), shadows: shadows)
        #expect(unshadowed > 0.2)
    }

    private func receiver(directional: Bool) throws -> SceneNode {
        let material = try PhysicallyBasedMaterial(
            roughness: 0.8, lightIntensity: 0, ambientIntensity: 0, receivesShadows: directional,
            alphaMode: .opaque, specular: .init(factor: 0)
        )
        let renderer = try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float,
                                        geometry: .plane(width: 5, height: 5), material: .physicallyBased(material))
        return SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: renderer, castsShadow: false)
    }

    private func shadowFixture(
        _ kind: LightKind, sampling: ShadowMapSampling = .filtered, bias: Float? = nil, size: Int? = nil,
        updateMode: ShadowMapUpdateMode = .automatic
    ) throws -> ShadowFixture {
        let size = size ?? resolution
        if kind == .directional {
            let map = try DirectionalShadowMap(queue: gpu.queue, resolution: size, directionToLight: SIMD3(0.7, 0, 1),
                                               center: SIMD3(0, 0, -3.5), halfExtent: 2, bias: bias ?? 0.0015, sampling: sampling,
                                               updateMode: updateMode)
            return ShadowFixture(directional: map, punctual: nil, light: nil, texture: map.texture,
                                 projection: map.clipFromWorld, slice: 0, clearedDepth: 1)
        }
        let spot = kind == .spot
        let maps = try PunctualShadowMaps(queue: gpu.queue, resolution: size, spotCapacity: spot ? 1 : 0,
                                          pointCapacity: spot ? 0 : 1, updateMode: updateMode)
        let position = SIMD3<Float>(2.1, 0, -1)
        let direction = simd_normalize(SIMD3<Float>(0, 0, -4) - position)
        let transform = SceneTransform(translation: position, rotation: simd_quatf(from: SIMD3(0, 0, -1), to: direction))
        let settings = try PunctualLight.Shadow(depthBias: bias ?? 0, normalBias: 0, filterRadius: 1, sampling: sampling)
        let outer: Float = 0.6
        let light = try PunctualLight(kind: spot ? .spot(innerConeAngle: 0.45, outerConeAngle: outer) : .point(),
                                      intensity: 40, shadow: settings)
        let slice = spot ? 0 : 5
        let clip = try PunctualShadowMaps.projection(
            origin: position, forward: spot ? direction : PunctualShadowMaps.faceDirections[slice],
            tangent: maps.projectionTangent(covering: spot ? tan(Double(outer)) : 1, shadow: settings), near: settings.nearPlane
        )
        return ShadowFixture(directional: nil, punctual: maps, light: SceneNode(transform: transform, light: light),
                             texture: maps.texture, projection: clip, slice: slice, clearedDepth: 0)
    }

    private func render(_ graph: SceneGraph, shadows: ShadowFixture) throws -> Float {
        try renderPixels(graph, shadows: shadows, side: side)[side / 2 * side + side / 2].x
    }

    private func renderPixels(_ graph: SceneGraph, shadows: ShadowFixture, side: Int) throws -> [SIMD4<Float>] {
        try renderFrame(graph, shadows: shadows, side: side).pixels
    }

    private func renderFrame(
        _ graph: SceneGraph, shadows: ShadowFixture, side: Int
    ) throws -> (pixels: [SIMD4<Float>], statistics: SceneDrawStatistics) {
        let target = try renderTarget(side: side)
        let command = try #require(gpu.queue.makeCommandBuffer())
        let statistics = try graph.encode(
            pass: target.pass, commandBuffer: command, shadowMap: shadows.directional, punctualShadowMaps: shadows.punctual
        )
        try gpu.finish(command)
        return (try gpu.readback(target.color), statistics)
    }

    private func renderTarget(side: Int) throws -> (pass: MTLRenderPassDescriptor, color: any MTLTexture) {
        let color = try gpu.output(width: side, height: side, usage: .renderTarget)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: side, height: side, mipmapped: false)
        descriptor.storageMode = .private
        descriptor.usage = .renderTarget
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = try #require(gpu.device.makeTexture(descriptor: descriptor))
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        return (pass, color)
    }

    private func depth(in shadows: ShadowFixture, point: SIMD3<Float>) async throws -> Float {
        let clip = shadows.projection * SIMD4(point, 1)
        let x = Int((0.5 + 0.5 * clip.x / clip.w) * Float(resolution))
        let y = Int((0.5 - 0.5 * clip.y / clip.w) * Float(resolution))
        let bytes = try await TextureTransfer.read(from: shadows.texture, slice: shadows.slice,
                                                   region: MTLRegionMake2D(x, y, 1, 1), commandQueue: gpu.queue).bytes
        return bytes.withUnsafeBytes { $0.loadUnaligned(as: Float.self) }
    }

    private func texture(_ value: SIMD4<Float16>) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(gpu.device.makeTexture(descriptor: descriptor))
        withUnsafeBytes(of: value) { bytes in
            texture.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: bytes.baseAddress!, bytesPerRow: bytes.count)
        }
        return texture
    }
}
