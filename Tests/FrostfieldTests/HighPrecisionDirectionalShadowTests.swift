import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct HighPrecisionDirectionalShadowTests {
    enum Receiver: CaseIterable, Sendable {
        case lambert, pbrGraph, classic
    }

    private let direction = SIMD3<Float>(0.4, 0, 1)
    private let farOrigin = SIMD3<Float>(0x1p60, 0, 0x1p60)

    @Test("Single directional caster depth and stock receiver shading survive a 2⁶⁰ translation",
          arguments: Receiver.allCases)
    func translated(_ receiver: Receiver) async throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let (material, sampling): (MeshMaterial, ShadowMapSampling)
        switch receiver {
        case .lambert:
            (material, sampling) = (.shadowedLambert, .basic)
        case .pbrGraph:
            let graph = try SurfaceShaderGraph { builder in
                .init(baseColor: builder.constant(SIMD3<Float>(0.8, 0.75, 0.7)))
            }
            material = .physicallyBased(try PhysicallyBasedMaterial(
                roughness: 0.9, ambientIntensity: 0, receivesShadows: true, surfaceGraph: graph
            ))
            sampling = .filtered
        case .classic:
            material = .classic(try ClassicMaterial(model: .lambert, receivesShadows: true))
            sampling = .variance
        }
        let nodes = try sceneNodes(gpu, material: material)

        func render(origin: SIMD3<Float>, highPrecision: Bool) async throws
            -> (pixels: [SIMD4<Float>], depth: [Float], statistics: SceneDrawStatistics) {
            let shadowMap = try map(gpu, origin: origin, sampling: sampling)
            let graph = scene(gpu, nodes: nodes, origin: origin, highPrecision: highPrecision)
            let target = try gpu.makeTarget()
            let command = try #require(gpu.queue.makeCommandBuffer())
            let projection = simd_float4x4(columns: (
                SIMD4(0.4, 0, 0, 0), SIMD4(0, 0.4, 0, 0), SIMD4(0, 0, -0.1, 0), SIMD4(0, 0, 0, 1)
            ))
            let view = simd_inverse(graph.cameraTransform.matrix)
            let camera = try ShaderGraphCamera(viewMatrix: view, projectionMatrix: projection, near: 0, far: 10)
            let statistics = try graph.encode(
                pass: gpu.pass(color: target.color, depth: target.depth), commandBuffer: command,
                clipFromWorld: projection * view, cameraPosition: origin, directionToCamera: SIMD3(0, 0, 1),
                shadowMap: shadowMap, graphCamera: camera
            )
            try gpu.finish(command)
            return (try gpu.pixels(of: target.color), try await depth(gpu, shadowMap), statistics)
        }

        let ordinary = try await render(origin: .zero, highPrecision: false)
        let near = try await render(origin: .zero, highPrecision: true)
        let relocated = try await render(origin: farOrigin, highPrecision: true)
        #expect(relocated.pixels == near.pixels)
        #expect(relocated.depth == near.depth)
        #expect(zip(ordinary.pixels, near.pixels).allSatisfy { isNearly($0, $1, tolerance: 0.002) })
        #expect(zip(ordinary.depth, near.depth).allSatisfy { abs($0 - $1) < 0.000001 })
        #expect(relocated.statistics.shadowDrawCount == 1)
        #expect(relocated.statistics.culledShadowCount == 1)

        let center = gpu.side / 2
        let shadowed = gpu.pixel(relocated.pixels, x: center, y: center)
        let lit = gpu.pixel(relocated.pixels, x: center + 20, y: center)
        let ambient: Float = receiver == .lambert ? 0.16 : 0
        #expect(abs(shadowed.y - ambient) < 0.01 && shadowed.w == 1, "\(shadowed)")
        #expect(lit.y > ambient + 0.12 && lit.w == 1, "\(lit)")
        #expect(relocated.depth.contains { $0 > 0 && $0 < 1 })
        #expect(relocated.depth.contains(1))
    }

    @Test("On-demand maps replace depth across coordinate modes and retain authored receiver bias")
    func updatesAndBias() async throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let nodes = try sceneNodes(gpu, material: .shadowedLambert)
        let shadowMap = try map(gpu, origin: farOrigin, sampling: .basic, updateMode: .onDemand)
        var graph = scene(gpu, nodes: nodes, origin: farOrigin, highPrecision: false)
        let legacy = try gpu.render(graph, shadowMap: shadowMap)
        #expect(legacy.statistics.shadowPassCount == 1)

        graph.highPrecision = true
        let precise = try gpu.render(graph, shadowMap: shadowMap)
        let initialDepth = try await depth(gpu, shadowMap)
        let center = gpu.side / 2
        #expect(precise.statistics.shadowPassCount == 1 && precise.statistics.shadowDrawCount == 1)
        #expect(precise.pixels != legacy.pixels)
        #expect(abs(gpu.pixel(precise.pixels, x: center, y: center).y - 0.16) < 0.001)
        #expect(initialDepth.contains { $0 > 0 && $0 < 1 })

        let reused = try gpu.render(graph, shadowMap: shadowMap)
        #expect(reused.statistics.shadowPassCount == 0)
        #expect(reused.pixels == precise.pixels)
        try shadowMap.update(directionToLight: direction, center: farOrigin, halfExtent: 8, bias: 0.25)
        let biased = try gpu.render(graph, shadowMap: shadowMap)
        #expect(biased.statistics.shadowPassCount == 0)
        #expect(gpu.pixel(biased.pixels, x: center, y: center).y > 0.5)
        #expect(try await depth(gpu, shadowMap) == initialDepth)

        try shadowMap.update(directionToLight: direction, center: farOrigin + SIMD3(0, 0.5, 0), halfExtent: 8)
        let recentered = try gpu.render(graph, shadowMap: shadowMap)
        #expect(recentered.statistics.shadowPassCount == 1)
        #expect(try await depth(gpu, shadowMap) != initialDepth)

        graph.highPrecision = false
        #expect(try gpu.render(graph, shadowMap: shadowMap).statistics.shadowPassCount == 1)
    }

    @Test("Directional cascades retain depth, split selection, and blend bands across a large grid-aligned origin",
          arguments: [ShadowMapSampling.basic, .variance])
    func translatedCascades(_ sampling: ShadowMapSampling) async throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let receiver = try gpu.unlit(GeometryRenderFixture.square(1), material: .shadowedLambert)
        let blocker = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(1),
            renderState: .init(depthWrite: false, colorWriteMask: [], side: .double)
        )
        let sampleY: [Float] = [1.45, 0.68, -0.08, -0.83, -1.58]
        let distances: [Float] = [2, 3.5, 5, 6.625, 7.25]
        let tilt: Float = 0.5
        let receivers = zip(sampleY, distances).enumerated().map { index, point in
            SceneNode(
                transform: .init(
                    translation: SIMD3(0, point.0, -point.1),
                    rotation: simd_quatf(angle: index == 1 ? tilt : 0, axis: SIMD3(0, 1, 0)),
                    scale: SIMD3(0.3, 0.22, 1)
                ),
                mesh: receiver, castsShadow: false
            )
        }
        let casters = [
            SceneNode(
                transform: .init(translation: SIMD3(0, 0.45, -0.5), scale: SIMD3(0.4, 1.4, 1)), mesh: blocker
            ),
            SceneNode(
                transform: .init(translation: SIMD3(0, -1.15, -4.5), scale: SIMD3(0.4, 0.7, 1)), mesh: blocker
            ),
        ]
        let projection = simd_float4x4(columns: (
            SIMD4(1, 0, 0, 0), SIMD4(0, 0.5, 0, 0), SIMD4(0, 0, -0.125, 0), SIMD4(0, 0, 0, 1)
        ))

        func render(origin: SIMD3<Float>, highPrecision: Bool) async throws
            -> (pixels: [SIMD4<Float>], slices: [[Float]], stats: SceneDrawStatistics) {
            let graph = scene(gpu, nodes: receivers + casters, origin: origin, highPrecision: highPrecision)
            let view = simd_inverse(graph.cameraTransform.matrix)
            let camera = try ShaderGraphCamera(viewMatrix: view, projectionMatrix: projection, near: 0, far: 8)
            let shadowMap = try DirectionalShadowMap(
                queue: gpu.queue, resolution: 64, directionToLight: SIMD3(0, 0, 1),
                cascades: .init(count: 2, maximumDistance: 7, blendFraction: 0.25, casterMargin: 0),
                sampling: sampling, varianceBlurRadius: 1
            )
            try shadowMap.fitCascades(view: camera, splitFractions: [4.0 / 7])
            #expect(shadowMap.cascadeSplits == [4, 7])
            #expect(shadowMap.texelSizes == [3.0 / 32, 3.0 / 32])

            let target = try gpu.makeTarget()
            let command = try #require(gpu.queue.makeCommandBuffer())
            let stats = try graph.encode(
                pass: gpu.pass(color: target.color, depth: target.depth), commandBuffer: command,
                clipFromWorld: projection * view, cameraPosition: origin, directionToCamera: SIMD3(0, 0, 1),
                shadowMap: shadowMap, graphCamera: camera
            )
            try gpu.finish(command)
            return (
                try gpu.pixels(of: target.color),
                [try await depth(gpu, shadowMap), try await depth(gpu, shadowMap, slice: 1)],
                stats
            )
        }

        // Both slice spheres have radius 3. This common shift is a whole
        // number of texels (3/32), so the world's sampling phase is the same.
        let translation: Float = 3 * 0x1p60
        let ordinary = try await render(origin: .zero, highPrecision: false)
        let near = try await render(origin: .zero, highPrecision: true)
        let relocated = try await render(origin: SIMD3(translation, 0, translation), highPrecision: true)
        let offGrid = try await render(origin: farOrigin, highPrecision: true)
        #expect(relocated.pixels == near.pixels)
        #expect(relocated.slices == near.slices)
        #expect(relocated.stats.shadowPassCount == 2)
        #expect(relocated.stats.shadowDrawCount == 3 && relocated.stats.culledShadowCount == 1)
        #expect(relocated.slices.allSatisfy { $0.contains(1) && $0.contains(where: { $0 > 0 && $0 < 1 }) })
        for (reference, shifted) in zip(near.slices, offGrid.slices) {
            // At 2⁶⁰ the world grid moves the snapped Z center by -1/16
            // relative to this camera. Both projections span six depth units.
            let delta = try #require(shifted.min()) - #require(reference.min())
            #expect(abs(delta + 1.0 / 96) < 0.000001)
        }
        #expect(zip(ordinary.pixels, near.pixels).allSatisfy { isNearly($0, $1, tolerance: 0.002) })
        let firstBlend = 0.16 + 0.42 * cos(tilt)
        for (y, expected) in zip(sampleY, [0.16, firstBlend, 1, 0.58, 1]) {
            let row = Int((0.5 - 0.25 * y) * Float(gpu.side))
            let value = gpu.pixel(relocated.pixels, x: gpu.side / 2, y: row)
            #expect(abs(value.y - expected) < 0.02 && value.w == 1, "row \(row): \(value), expected \(expected)")
            #expect(isNearly(gpu.pixel(offGrid.pixels, x: gpu.side / 2, y: row), value, tolerance: 0.002))
        }
        let tiltedRow = Int((0.5 - 0.25 * sampleY[1]) * Float(gpu.side))
        let left = gpu.pixel(relocated.pixels, x: gpu.side / 2 - 5, y: tiltedRow)
        let right = gpu.pixel(relocated.pixels, x: gpu.side / 2 + 5, y: tiltedRow)
        #expect(right.y - left.y > 0.08 && left.w == 1 && right.w == 1)
    }

    @Test("Subtexel refits reuse on-demand cascade depth while a whole-cell move refreshes it")
    func stabilizedOnDemandCascades() async throws {
        let gpu = try GeometryRenderFixture(side: 65)
        let nodes = try sceneNodes(gpu, material: .shadowedLambert)
        let graph = scene(gpu, nodes: nodes, origin: .zero, highPrecision: true)
        let shadowMap = try DirectionalShadowMap(
            queue: gpu.queue, resolution: 64, directionToLight: SIMD3(0, 0, 1),
            cascades: .init(count: 2, maximumDistance: 10), updateMode: .onDemand
        )

        func fit(y: Float) throws {
            try shadowMap.fitCascades(
                camera: gpu.camera, cameraTransform: .init(translation: SIMD3(0, y, 0)), aspectRatio: 1
            )
        }
        try fit(y: 0)
        let first = try gpu.render(graph, shadowMap: shadowMap)
        let depthBefore = try await depth(gpu, shadowMap)
        #expect(first.statistics.shadowPassCount == 2)

        let smallestTexel = try #require(shadowMap.texelSizes.min())
        try fit(y: smallestTexel * 0.45)
        let subtexel = try gpu.render(graph, shadowMap: shadowMap)
        #expect(subtexel.statistics.shadowPassCount == 0)
        #expect(subtexel.pixels == first.pixels)
        #expect(try await depth(gpu, shadowMap) == depthBefore)

        try fit(y: 2 * smallestTexel)
        let shifted = try gpu.render(graph, shadowMap: shadowMap)
        #expect(shifted.statistics.shadowPassCount == 2)
        #expect(try await depth(gpu, shadowMap) != depthBefore)
    }

    private func map(
        _ gpu: GeometryRenderFixture, origin: SIMD3<Float>, sampling: ShadowMapSampling,
        updateMode: ShadowMapUpdateMode = .automatic
    ) throws -> DirectionalShadowMap {
        try DirectionalShadowMap(
            queue: gpu.queue, resolution: 64, directionToLight: direction,
            center: origin, halfExtent: 8, sampling: sampling, varianceBlurRadius: 1, updateMode: updateMode
        )
    }

    private func sceneNodes(_ gpu: GeometryRenderFixture, material: MeshMaterial) throws -> [SceneNode] {
        let receiver = try gpu.unlit(GeometryRenderFixture.square(2.25), material: material)
        let blocker = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(0.55),
            renderState: .init(depthWrite: false, colorWriteMask: [], side: .double)
        )
        return [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -5)), mesh: receiver, castsShadow: false),
            SceneNode(transform: .init(translation: SIMD3(0.6, 0, -3.5)), mesh: blocker),
            SceneNode(transform: .init(translation: SIMD3(50, 0, -3.5)), mesh: blocker),
        ]
    }

    private func scene(
        _ gpu: GeometryRenderFixture, nodes: [SceneNode], origin: SIMD3<Float>, highPrecision: Bool
    ) -> SceneGraph {
        let anchor = SceneTransform(translation: origin)
        var graph = SceneGraph(
            roots: [SceneNode(transform: anchor, children: nodes)], camera: gpu.camera, cameraTransform: anchor
        )
        graph.highPrecision = highPrecision
        return graph
    }

    private func depth(
        _ gpu: GeometryRenderFixture, _ map: DirectionalShadowMap, slice: Int = 0
    ) async throws -> [Float] {
        let image = try await TextureTransfer.read(from: map.texture, slice: slice, commandQueue: gpu.queue)
        return image.bytes.withUnsafeBytes { bytes in
            (0..<(map.resolution * map.resolution)).map {
                bytes.loadUnaligned(fromByteOffset: $0 * MemoryLayout<Float>.stride, as: Float.self)
            }
        }
    }
}
