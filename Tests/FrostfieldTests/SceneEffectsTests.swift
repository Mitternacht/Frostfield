import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SelectionOutlineTests {
    private let gpu: PostProcessFixture
    private let side = 64

    init() throws {
        gpu = try PostProcessFixture()
    }

    @Test("Selecting a node includes its descendants")
    func expandedSelection() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let leaf = SceneNode()
        let child = SceneNode(children: [leaf])
        let other = SceneNode()
        let graph = SceneGraph(roots: [SceneNode(children: [child]), other], camera: camera)
        #expect(SelectionOutline.expandedSelection([child.id], in: graph) == [child.id, leaf.id])
        #expect(SelectionOutline.expandedSelection([], in: graph).isEmpty)
    }

    /// Renders a scene and outlines the `selected` nodes, returning the scene color and the result.
    private func outlined(
        _ graph: SceneGraph, selected: Set<UUID>, settings: SelectionOutline.Settings, depthMode: MeshDepthMode = .forward,
        expectedDraws: Int = 1, pattern: (any MTLTexture)? = nil
    ) throws -> (input: [SIMD4<Float>], output: [SIMD4<Float>]) {
        let color = try gpu.color(width: side, height: side)
        let depth = try gpu.depthTarget(width: side, height: side)
        let destination = try gpu.color(width: side, height: side)
        let outline = try SelectionOutline(device: gpu.device, settings: settings, patternTexture: pattern)
        try gpu.run { buffer in
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = color
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = depth
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.clearDepth = depthMode == .forward ? 1 : 0
            pass.depthAttachment.storeAction = .store
            if depthMode == .forward {
                try graph.encode(pass: pass, commandBuffer: buffer)
            } else {
                // The reversed view is built here independently of SelectionOutline's default.
                let n = Double(graph.camera.near)
                let f = Double(graph.camera.far)
                var projection = try graph.camera.projection(aspectRatio: 1)
                projection.columns.2.z = Float(n / (f - n))
                projection.columns.3.z = Float(n * f / (f - n))
                try graph.encode(
                    pass: pass, commandBuffer: buffer, clipFromWorld: projection * simd_inverse(graph.cameraTransform.matrix),
                    cameraPosition: graph.cameraTransform.translation, depthMode: .reversed
                )
            }
            let drawn = try outline.encodeSelection(
                graph: graph, selection: selected, width: side, height: side, depthMode: depthMode, commandBuffer: buffer
            )
            #expect(drawn == expectedDraws)
            try outline.encode(source: color, depth: depth, destination: destination, commandBuffer: buffer)
        }
        return (try gpu.read(color), try gpu.read(destination))
    }

    /// Selected square A at depth 4 covers columns 20...27 and rows 28...35; unselected
    /// square B covers columns 36...43; square C at depth 2 covers columns 17...22 and rows
    /// 29...34, hiding A's left edge there.
    private func threeSquares(depthMode: MeshDepthMode = .forward) throws -> (graph: SceneGraph, a: SceneNode) {
        let renderer = { (half: Float) in
            try MeshRenderer(
                device: gpu.device, colorPixelFormat: .rgba16Float, geometry: try GeometryRenderFixture.square(half),
                material: .unlit, depthMode: depthMode
            )
        }
        let a = SceneNode(transform: SceneTransform(translation: SIMD3(-1, 0, -4)), mesh: try renderer(0.5),
                          materialParameters: SIMD4(0.2, 0.6, 0.2, 1))
        let b = SceneNode(transform: SceneTransform(translation: SIMD3(1, 0, -4)), mesh: try renderer(0.5),
                          materialParameters: SIMD4(0.6, 0.2, 0.2, 1))
        let c = SceneNode(transform: SceneTransform(translation: SIMD3(-0.75, 0, -2)), mesh: try renderer(0.2),
                          materialParameters: SIMD4(0.2, 0.2, 0.6, 1))
        let graph = SceneGraph(
            roots: [a, b, c], camera: try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 50)
        )
        return (graph, a)
    }

    @Test("The outline follows only the selected node's edges, in the hidden color where it is occluded",
          arguments: [MeshDepthMode.forward, .reversed])
    func outline(depthMode: MeshDepthMode) throws {
        let (graph, a) = try threeSquares(depthMode: depthMode)
        let fill = SIMD4<Float>(0.9, 0.1, 0.9, 0.5)
        let (source, result) = try outlined(
            graph, selected: [a.id],
            settings: .init(visibleColor: SIMD4(1, 0.5, 0, 1), hiddenColor: SIMD4(0, 0.5, 1, 1), hiddenFill: fill, thickness: 1.5),
            depthMode: depthMode
        )
        gpu.savePNG(result, width: side, height: side, name: "outline-\(depthMode)")
        func at(_ x: Int, _ y: Int) -> SIMD4<Float> { result[y * side + x] }
        func input(_ x: Int, _ y: Int) -> SIMD4<Float> { source[y * side + x] }

        // A's visible right edge: a 1.5 px band covers one pixel fully and the next half.
        #expect(PostProcessFixture.close(at(28, 32), SIMD4(1, 0.5, 0, 1)))
        let half = SIMD4<Float>(1, 0.5, 0, 1) * 0.5 + input(29, 32) * 0.5
        #expect(PostProcessFixture.close(at(29, 32), half))
        #expect(at(30, 32) == input(30, 32))
        // A's top and bottom edges.
        #expect(PostProcessFixture.close(at(24, 27), SIMD4(1, 0.5, 0, 1)))
        #expect(PostProcessFixture.close(at(24, 36), SIMD4(1, 0.5, 0, 1)))
        // A's interior is unchanged where visible and filled where hidden behind C.
        #expect(at(24, 32) == input(24, 32))
        let filled = SIMD4(fill.x * fill.w, fill.y * fill.w, fill.z * fill.w, fill.w) + input(21, 32) * (1 - fill.w)
        #expect(PostProcessFixture.close(at(21, 32), filled))
        // Left of A, behind C: the hidden color.
        #expect(PostProcessFixture.close(at(19, 32), SIMD4(0, 0.5, 1, 1)))
        // B and everything away from A stay untouched.
        for y in 24..<40 {
            for x in 33..<48 {
                #expect(at(x, y) == input(x, y), "(\(x), \(y))")
            }
        }
        #expect(at(5, 5) == input(5, 5))
    }

    @Test("A straight edge receives exactly `thickness` pixels of outline coverage",
          arguments: [Float(0.5), 0.75, 1, 1.5, 2, 3.25])
    func thickness(thickness: Float) throws {
        let (graph, a) = try threeSquares()
        let (source, result) = try outlined(
            graph, selected: [a.id], settings: .init(visibleColor: SIMD4(1, 1, 1, 1), thickness: thickness)
        )
        // Right of A's edge at column 27, on rows 30...33, the black background takes
        // coverage saturate(thickness + 1 − d) at d = 1, 2, 3, ... pixels out.
        for y in 30...33 {
            var total: Float = 0
            for d in 1...6 {
                let pixel = result[y * side + 27 + d]
                #expect(source[y * side + 27 + d] == SIMD4(0, 0, 0, 1))
                let expected = min(max(thickness + 1 - Float(d), 0), 1)
                #expect(abs(pixel.x - expected) < 2e-3, "row \(y), \(d) px out: \(pixel.x) vs \(expected)")
                total += pixel.x
            }
            #expect(abs(total - thickness) < 4e-3)
        }
    }

    @Test("Outline pulse follows the pinned timing and modulates both edge colors only")
    func pulse() throws {
        let (graph, a) = try threeSquares()
        let visible = SIMD4<Float>(1, 0.5, 0, 1)
        let hidden = SIMD4<Float>(0, 0.5, 1, 1)
        var settings = SelectionOutline.Settings(
            visibleColor: visible, hiddenColor: hidden, hiddenFill: SIMD4(0.5, 0.25, 0.75, 0.5), thickness: 1
        )
        let baseline = try outlined(graph, selected: [a.id], settings: settings).output
        settings.pulsePeriod = 2
        #expect(try outlined(graph, selected: [a.id], settings: settings).output == baseline)
        let cycle = Double(settings.pulsePeriod) * 2 * Double.pi / 10
        settings.time = cycle / 2
        let trough = try outlined(graph, selected: [a.id], settings: settings).output
        #expect(trough[32 * side + 28] == SIMD4(0.25, 0.125, 0, 1))
        #expect(trough[32 * side + 19] == SIMD4(0, 0.125, 0.25, 1))
        #expect(trough[32 * side + 21] == baseline[32 * side + 21])
        settings.time = cycle
        #expect(try outlined(graph, selected: [a.id], settings: settings).output == baseline)
    }

    @Test("Outline glow blurs the visible and hidden silhouette into the exterior")
    func glow() throws {
        let (graph, a) = try threeSquares()
        let visible = SIMD4<Float>(1, 0.5, 0, 1)
        let hidden = SIMD4<Float>(0, 0.5, 1, 1)
        var settings = SelectionOutline.Settings(visibleColor: visible, hiddenColor: hidden, thickness: 1)
        let base = try outlined(graph, selected: [a.id], settings: settings).output
        settings.edgeGlow = 1.5
        settings.glowSigma = 1.5
        let softened = try outlined(graph, selected: [a.id], settings: settings).output
        let kernel = try GaussianBlur.kernel(sigma: settings.glowSigma)
        let radius = kernel.count / 2

        func edge(_ x: Int, _ y: Int) -> SIMD4<Float> {
            if (20...27).contains(x), (28...35).contains(y) { return .zero }
            var visibleDistance = Float.infinity, hiddenDistance = Float.infinity
            for sy in 28...35 {
                for sx in 20...27 {
                    let dx = sx - x, dy = sy - y
                    let distance = Float(dx * dx + dy * dy).squareRoot()
                    if sx <= 22, (29...34).contains(sy) {
                        hiddenDistance = min(hiddenDistance, distance)
                    } else {
                        visibleDistance = min(visibleDistance, distance)
                    }
                }
            }
            let v = min(max(2 - visibleDistance, 0), 1)
            let h = min(max(2 - hiddenDistance, 0), 1) * (1 - v)
            return visible * v + hidden * h
        }

        for (x, y) in [(31, 32), (30, 32), (28, 32), (18, 32), (17, 32), (24, 26)] {
            var filtered = SIMD4<Float>.zero
            for dy in -radius...radius {
                for dx in -radius...radius {
                    filtered += edge(x + dx, y + dy) * (kernel[dx + radius] * kernel[dy + radius])
                }
            }
            let index = y * side + x
            let expected = base[index] + SIMD4(filtered.x, filtered.y, filtered.z, 0) * settings.edgeGlow
            #expect(PostProcessFixture.close(softened[index], expected, tolerance: 0.003),
                    "(\(x), \(y)): \(softened[index]) vs \(expected)")
        }
        #expect(softened[32 * side + 31].x > 0 && softened[32 * side + 17].z > 0)
        #expect(softened[32 * side + 24] == base[32 * side + 24])
        #expect(softened[5 * side + 5] == base[5 * side + 5])
    }

    @Test("The repeated pattern stays inside selected geometry and is half as strong behind occluders",
          arguments: [MeshDepthMode.forward, .reversed])
    func pattern(depthMode: MeshDepthMode) throws {
        let (graph, a) = try threeSquares(depthMode: depthMode)
        let stripe = try gpu.color(width: 4, height: 1, pixel: { x, _ in
            x < 2 ? SIMD4(0, 1, 1, 1) : SIMD4(1, 0, 0, 1)
        })
        var settings = SelectionOutline.Settings(
            visibleColor: SIMD4(1, 0.5, 0, 1), hiddenColor: SIMD4(0, 0.5, 1, 1), thickness: 1,
            patternRepeat: SIMD2(16, 3), patternOpacity: 0.8
        )
        let base = try outlined(graph, selected: [a.id], settings: settings, depthMode: depthMode).output
        let patterned = try outlined(graph, selected: [a.id], settings: settings, depthMode: depthMode, pattern: stripe).output
        for x in 16...31 {
            let index = 32 * side + x
            let coverage: Float = (20...27).contains(x) && x % 4 < 2 ? (x <= 22 ? 0.4 : 0.8) : 0
            let expected = SIMD4(repeating: coverage) + base[index] * (1 - coverage)
            #expect(PostProcessFixture.close(patterned[index], expected), "column \(x)")
        }
        settings.patternOpacity = 0
        let disabled = try outlined(graph, selected: [a.id], settings: settings, depthMode: depthMode, pattern: stripe)
        #expect(disabled.output == base)
    }

    /// The selection mask's covered pixels, read back through a blit.
    private func maskCoverage(_ outline: SelectionOutline, forward: Bool = true) throws -> [Bool] {
        let mask = try #require(outline.selectionMask)
        let buffer = try #require(gpu.device.makeBuffer(length: mask.width * mask.height * 4, options: .storageModeShared))
        try gpu.run { command in
            let blit = try #require(command.makeBlitCommandEncoder())
            blit.copy(
                from: mask, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                sourceSize: MTLSize(width: mask.width, height: mask.height, depth: 1), to: buffer, destinationOffset: 0,
                destinationBytesPerRow: mask.width * 4, destinationBytesPerImage: mask.width * mask.height * 4
            )
            blit.endEncoding()
        }
        let values = buffer.contents().bindMemory(to: Float.self, capacity: mask.width * mask.height)
        return (0..<(mask.width * mask.height)).map { forward ? values[$0] < 1 : values[$0] > 0 }
    }

    @Test("Selection masks cover exactly what instanced, morphed, skinned, blended, transmissive and masked meshes draw")
    func deformedSilhouettes() throws {
        let square = try GeometryRenderFixture.square(0.5)
        let pairs = try InstanceSet(device: gpu.device, transforms: [
            SceneTransform(translation: SIMD3(-1, 0, 0)).matrix, SceneTransform(translation: SIMD3(1, 0.5, 0)).matrix,
        ])
        let slide = try MeshMorphTargets(vertexCount: 4, targets: [
            .init(positionOffsets: [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 0.5, 0), SIMD3(0, 0, 0)]),
        ])
        let skin = try MeshSkin(
            jointCount: 1,
            influences: Array(repeating: VertexSkinInfluence(joints: .zero, weights: SIMD4(1, 0, 0, 0)), count: 4)
        )
        func pbr(_ alphaMode: PhysicallyBasedMaterial.AlphaMode, alpha: Float) throws -> MeshMaterial {
            .physicallyBased(try PhysicallyBasedMaterial(baseColorFactor: SIMD4(0.8, 0.8, 0.8, alpha), alphaMode: alphaMode))
        }
        struct Case {
            let name: String
            let material: MeshMaterial
            var skin: MeshSkin?
            var morph: MeshMorphTargets?
            var instances: InstanceSet?
            var empty = false
        }
        let cases = [
            Case(name: "blended instances", material: try pbr(.blend, alpha: 0.5), instances: pairs),
            Case(name: "blended morph", material: try pbr(.blend, alpha: 0.5), morph: slide),
            Case(name: "skinned morph", material: .unlit, skin: skin, morph: slide),
            Case(name: "skinned instances", material: .unlit, skin: skin, instances: pairs),
            Case(name: "masked instances, kept", material: try pbr(.mask(cutoff: 0.5), alpha: 0.8), instances: pairs),
            Case(name: "masked instances, cut", material: try pbr(.mask(cutoff: 0.5), alpha: 0.3), instances: pairs, empty: true),
        ]
        for item in cases {
            let mesh = try MeshRenderer(
                device: gpu.device, colorPixelFormat: .rgba16Float, geometry: square, material: item.material,
                skin: item.skin, morphTargets: item.morph, supportsInstancing: item.instances != nil
            )
            let node = SceneNode(
                transform: SceneTransform(translation: SIMD3(0, -0.25, -4)), mesh: mesh,
                jointPalette: item.skin == nil ? nil : try JointPalette(matrices: [SceneTransform(translation: SIMD3(-0.75, 0, 0)).matrix]),
                instances: item.instances, morphWeights: item.morph == nil ? nil : [1]
            )
            let graph = SceneGraph(roots: [node], camera: try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 50))
            let color = try gpu.color(width: side, height: side)
            let depth = try gpu.depthTarget(width: side, height: side)
            let outline = try SelectionOutline(device: gpu.device)
            try gpu.run { buffer in
                let pass = MTLRenderPassDescriptor()
                pass.colorAttachments[0].texture = color
                pass.colorAttachments[0].loadAction = .clear
                pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
                pass.colorAttachments[0].storeAction = .store
                pass.depthAttachment.texture = depth
                pass.depthAttachment.loadAction = .clear
                pass.depthAttachment.clearDepth = 1
                pass.depthAttachment.storeAction = .store
                try graph.encode(pass: pass, commandBuffer: buffer)
                try outline.encodeSelection(graph: graph, selection: [node.id], width: side, height: side, commandBuffer: buffer)
            }
            let drawn = try gpu.read(color).map { $0.w > 0 }
            let selected = try maskCoverage(outline)
            #expect(drawn.contains(true) == !item.empty, "\(item.name) draws")
            let mismatches = zip(drawn, selected).filter { $0 != $1 }.count
            #expect(mismatches == 0, "\(item.name): \(mismatches) pixels differ between the color pass and the selection mask")
        }

        // A transmissive mesh needs a backdrop to draw in color (and cannot be instanced),
        // so a morphed one's mask is compared with its projected square: a morph shifting
        // every vertex by (1, 0.5) covers columns 36..<44 and rows 26..<34.
        let shift = try MeshMorphTargets(vertexCount: 4, targets: [
            .init(positionOffsets: Array(repeating: SIMD3(1, 0.5, 0), count: 4)),
        ])
        let glass = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: square,
            material: .physicallyBased(try PhysicallyBasedMaterial(transmission: .init(factor: 1, thickness: 0.1))),
            morphTargets: shift
        )
        #expect(glass.requiresTransmissionBackdrop)
        let node = SceneNode(transform: SceneTransform(translation: SIMD3(0, -0.25, -4)), mesh: glass, morphWeights: [1])
        let graph = SceneGraph(roots: [node], camera: try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 50))
        let outline = try SelectionOutline(device: gpu.device)
        try gpu.run { try outline.encodeSelection(graph: graph, selection: [node.id], width: side, height: side, commandBuffer: $0) }
        let selected = try maskCoverage(outline)
        for y in 0..<side {
            for x in 0..<side {
                let inside = (36..<44).contains(x) && (26..<34).contains(y)
                #expect(selected[y * side + x] == inside, "transmissive morph at (\(x), \(y))")
            }
        }
    }

    @Test("The outline needs this frame's selection mask")
    func missingSelection() throws {
        let outline = try SelectionOutline(device: gpu.device)
        let source = try gpu.color(width: 8, height: 8)
        let depth = try gpu.depth(width: 8, height: 8) { _, _ in 1 }
        let destination = try gpu.color(width: 8, height: 8)
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PostProcessEffectError.missingSelection) {
            try outline.encode(source: source, depth: depth, destination: destination, commandBuffer: buffer)
        }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct GodRaysTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    private func render(
        settings: GodRays.Settings, width: Int = 64, height: Int = 48, depth: @escaping (Int, Int) -> Float,
        color: @escaping (Int, Int) -> SIMD4<Float>
    ) throws -> (input: [SIMD4<Float>], output: [SIMD4<Float>]) {
        let effect = try GodRays(device: gpu.device, settings: settings)
        let depthTexture = try gpu.depth(width: width, height: height, value: depth)
        let source = try gpu.color(width: width, height: height, pixel: color)
        let destination = try gpu.color(width: width, height: height)
        try gpu.run { try effect.encode(source: source, depth: depthTexture, destination: destination, commandBuffer: $0) }
        return (try gpu.read(source), try gpu.read(destination))
    }

    @Test("A uniform open sky adds exactly intensity times color times the mask")
    func uniformSky() throws {
        let sky = SIMD4<Float>(0.2, 0.3, 0.4, 1)
        for passes in 1...3 {
            let settings = GodRays.Settings(
                lightPosition: SIMD2(0.7, 0.2), color: SIMD3(1, 0.5, 0.25), intensity: 2, density: 1, decay: 0.95,
                samples: 24, passes: passes, resolutionScale: 0.5, source: .sky(threshold: 0)
            )
            let (_, output) = try render(settings: settings, depth: { _, _ in 1 }, color: { _, _ in sky })
            let expected = SIMD4<Float>(0.2 + 2 * 0.2, 0.3 + 2 * 0.5 * 0.3, 0.4 + 2 * 0.25 * 0.4, 1)
            for pixel in output {
                #expect(PostProcessFixture.close(pixel, expected, tolerance: 2e-3), "\(passes) passes: \(pixel)")
            }
        }
    }

    @Test("Occluded pixels whose rays stay behind geometry receive no light")
    func occluded() throws {
        let sky = SIMD4<Float>(1, 1, 1, 1)
        let settings = GodRays.Settings(
            lightPosition: SIMD2(1.2, 0.5), intensity: 1, density: 0.1, decay: 1, samples: 16, passes: 1,
            resolutionScale: 1, source: .sky(threshold: 0)
        )
        let (input, output) = try render(settings: settings, depth: { x, _ in x < 32 ? 0.5 : 1 }, color: { _, _ in sky })
        for y in 0..<48 {
            for x in 0..<20 { #expect(output[y * 64 + x] == input[y * 64 + x]) }
            for x in 34..<64 { #expect(output[y * 64 + x].x > 1.5) }
        }
    }

    @Test("An occluder casts a darker shaft away from the light than the open sky beside it")
    func shadowShaft() throws {
        // Light at the top center; a block of geometry below it; sky elsewhere.
        let settings = GodRays.Settings(
            lightPosition: SIMD2(0.5, 0.05), color: .one, intensity: 1, density: 0.8, decay: 0.99, samples: 64, passes: 2,
            resolutionScale: 1, source: .sky(threshold: 0)
        )
        let (_, output) = try render(
            settings: settings, depth: { x, y in (28..<36).contains(x) && (12..<20).contains(y) ? 0.5 : 1 },
            color: { _, _ in SIMD4(1, 1, 1, 1) }
        )
        gpu.savePNG(output.map { SIMD4($0.x - 1, $0.y - 1, $0.z - 1, 1) }, width: 64, height: 48, name: "godrays-shaft")
        let behind = output[40 * 64 + 32].x - 1
        let beside = output[40 * 64 + 12].x - 1
        #expect(behind < beside - 0.1, "behind \(behind), beside \(beside)")
    }

    @Test("A sun disk lights pixels whose rays stay inside its core and leaves rays that never reach it dark")
    func sunDisk() throws {
        // Disk radius 0.25 with a full core to 0.125; density 0.5 marches at most halfway to the light.
        let settings = GodRays.Settings(
            lightPosition: SIMD2(0.5, 0.5), color: SIMD3(1, 0.5, 0.25), intensity: 1, density: 0.5, decay: 1, samples: 16,
            passes: 1, resolutionScale: 1, source: .sun(radius: 0.25)
        )
        let (input, output) = try render(
            settings: settings, width: 64, height: 64, depth: { _, _ in 1 }, color: { _, _ in SIMD4(0, 0, 0, 1) }
        )
        for y in 0..<64 {
            for x in 0..<64 {
                let d = simd_length(SIMD2((Float(x) + 0.5) / 64, (Float(y) + 0.5) / 64) - 0.5)
                if d < 0.1 {
                    #expect(output[y * 64 + x] == SIMD4(1, 0.5, 0.25, 1), "(\(x), \(y)) inside the core")
                } else if d > 0.6 {
                    #expect(output[y * 64 + x] == input[y * 64 + x], "(\(x), \(y)) far from the disk")
                }
            }
        }
    }

    @Test("A light direction projects to the frame position its pixels see")
    func screenPosition() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
        let ahead = try GodRays.screenPosition(
            ofDirection: SIMD3(0, 0, -1), camera: camera, cameraTransform: SceneTransform(), aspectRatio: 1
        )
        #expect(ahead == SIMD2(0.5, 0.5))
        let upRight = try #require(try GodRays.screenPosition(
            ofDirection: SIMD3(0.5, 0.5, -1), camera: camera, cameraTransform: SceneTransform(), aspectRatio: 1
        ))
        #expect(abs(upRight.x - 0.75) < 1e-6 && abs(upRight.y - 0.25) < 1e-6)
        #expect(try GodRays.screenPosition(
            ofDirection: SIMD3(0, 0, 1), camera: camera, cameraTransform: SceneTransform(), aspectRatio: 1
        ) == nil)
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct LensFlareTests {
    private let gpu: PostProcessFixture
    private let width = 64
    private let height = 48

    init() throws {
        gpu = try PostProcessFixture()
    }

    private let ghostOnly = LensFlare.Settings(
        glareRadius: 0, starburstRays: 0, haloIntensity: 0,
        ghosts: [LensFlare.Ghost(distance: 1, radius: 0.05, color: SIMD3(0.5, 0.25, 1), softness: 0.5)],
        ghostBlades: 0, occlusionRadius: 4, occlusionSamples: 32
    )

    private func render(
        lights: [LensFlare.Light], settings: LensFlare.Settings, depth: @escaping (Int, Int) -> Float
    ) throws -> (input: [SIMD4<Float>], output: [SIMD4<Float>]) {
        let flare = try LensFlare(device: gpu.device, lights: lights, settings: settings)
        let depthTexture = try gpu.depth(width: width, height: height, value: depth)
        let source = try gpu.color(width: width, height: height) { _, _ in SIMD4(0, 0, 0, 1) }
        let destination = try gpu.color(width: width, height: height)
        try gpu.run { try flare.encode(source: source, depth: depthTexture, destination: destination, commandBuffer: $0) }
        return (try gpu.read(source), try gpu.read(destination))
    }

    /// The fraction of the flare's Vogel visibility taps around pixel (16, 12) at or right of column 16.
    private func rightHalfFraction() -> Float {
        var visible = 0
        for i in 0..<32 {
            let r = Float(4) * ((Float(i) + 0.5) / 32).squareRoot()
            let theta = Float(i) * 2.39996323
            if (Float(16) + r * cos(theta)).rounded(.down) >= 16 { visible += 1 }
        }
        return Float(visible) / 32
    }

    @Test("A visible light places its ghost mirrored through the center with the exact tint")
    func ghost() throws {
        let light = LensFlare.Light(position: SIMD2(0.25, 0.25), color: SIMD3(1, 0.8, 0.6), intensity: 2)
        let (_, output) = try render(lights: [light], settings: ghostOnly, depth: { _, _ in 1 })
        let expected = SIMD4<Float>(2 * 0.5, 2 * 0.8 * 0.25, 2 * 0.6, 1)
        #expect(PostProcessFixture.close(output[36 * width + 48], expected, tolerance: 2e-3), "\(output[36 * width + 48])")
        #expect(output[12 * width + 16] == SIMD4(0, 0, 0, 1))
        let full = try render(lights: [light], settings: LensFlare.Settings(), depth: { _, _ in 1 })
        gpu.savePNG(full.output, width: width, height: height, name: "lens-flare")
        #expect(full.output[12 * width + 16].x > 1)
    }

    @Test("Occlusion scales the flare by the visible fraction of the light's neighborhood")
    func occlusion() throws {
        let light = LensFlare.Light(position: SIMD2(0.25, 0.25), color: .one, intensity: 1)
        let hidden = try render(lights: [light], settings: ghostOnly, depth: { _, _ in 0.5 })
        #expect(hidden.output == hidden.input)

        let half = try render(lights: [light], settings: ghostOnly, depth: { x, _ in x < 16 ? 0.5 : 1 })
        let fraction = rightHalfFraction()
        #expect(fraction > 0.3 && fraction < 0.7)
        #expect(abs(half.output[36 * width + 48].x - 0.5 * fraction) < 2e-3)

        // A light with depth is hidden only by nearer geometry.
        let near = LensFlare.Light(position: SIMD2(0.25, 0.25), color: .one, intensity: 1, depth: 0.6)
        #expect(try render(lights: [near], settings: ghostOnly, depth: { _, _ in 0.5 }).output[36 * width + 48].x == 0)
        #expect(abs(try render(lights: [near], settings: ghostOnly, depth: { _, _ in 0.7 }).output[36 * width + 48].x - 0.5) < 2e-3)

        // Off-frame lights have no visible taps.
        let outside = LensFlare.Light(position: SIMD2(-0.5, 0.5), color: .one, intensity: 1)
        let offFrame = try render(lights: [outside], settings: ghostOnly, depth: { _, _ in 1 })
        #expect(offFrame.output == offFrame.input)
    }

    @Test("Projected lights beyond the far plane or inside the near plane get depths that encode and occlude correctly")
    func projectedDepthRange() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
        // (z / 2, −z / 2, z) projects to (0.25, 0.25) of a square frame.
        func light(z: Float, depthMode: MeshDepthMode = .forward) throws -> LensFlare.Light {
            try #require(try LensFlare.Light(
                worldPosition: SIMD3(z / 2, -z / 2, z), camera: camera, cameraTransform: SceneTransform(), aspectRatio: 1,
                depthMode: depthMode
            ))
        }
        let far = try light(z: -20)
        #expect(far.depth == nil)
        #expect(try light(z: -20, depthMode: .reversed).depth == nil)
        #expect(abs(far.position.x - 0.25) < 1e-6 && abs(far.position.y - 0.25) < 1e-6)
        // Seen through cleared depth like a light at infinity; any rendered surface hides it.
        let clear = try render(lights: [far], settings: ghostOnly, depth: { _, _ in 1 })
        #expect(abs(clear.output[36 * width + 48].x - 0.5) < 2e-3)
        #expect(try render(lights: [far], settings: ghostOnly, depth: { _, _ in 0.9999 }).output == clear.input)

        // Between the camera and the near plane nothing rendered can be in front of it.
        let near = try light(z: -0.05)
        #expect(near.depth == 0)
        #expect(try light(z: -0.05, depthMode: .reversed).depth == 1)
        let covered = try render(lights: [near], settings: ghostOnly, depth: { _, _ in 0 })
        #expect(abs(covered.output[36 * width + 48].x - 0.5) < 2e-3)

        // A light inside the clip range keeps its stored depth.
        let inside = try light(z: -5)
        #expect(abs(try #require(inside.depth) - storedDepth(distance: 5, near: 0.1, far: 10)) < 1e-6)
    }
}
