import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ShaderGraphUVTests {
    @Test("GPU buffer projections agree with Three.js r187 across the panorama seam and sprite frame wraps")
    func computeProjections() throws {
        let gpu = try GeometryRenderFixture()
        let graph = try BufferShaderGraph(fieldCount: 4) { b in
            let equirect = b.equirectUV(direction: b.elementField(0).xyz)
            let matcap = b.matcapUV(normalInView: b.elementField(1).xyz, viewDirectionInView: b.elementField(2).xyz)
            let sprite = b.spritesheetUV(
                b.elementField(3).xy, uv: b.constant(SIMD2<Float>(0.25, 0.75)), frame: b.elementField(3).z
            )
            let zero = b.constant(Float(0))
            return [0: b.vector(equirect.x, equirect.y, matcap.x, matcap.y), 1: b.vector(sprite.x, sprite.y, zero, zero)]
        }
        // Expected UVs from the pinned TSL function bodies evaluated on the CPU,
        // with their vertical origin translated to Frostfield's top-left origin.
        let panorama: [(SIMD3<Float>, SIMD2<Float>)] = [
            (SIMD3(1, 0, 0), SIMD2(0.5, 0.5)),
            (SIMD3(0, 0, 1), SIMD2(0.75, 0.5)),
            (SIMD3(-1, 0, 0), SIMD2(1, 0.5)),
            (SIMD3(0, 0, -1), SIMD2(0.25, 0.5)),
            (SIMD3(0, 1, 0), SIMD2(0.5, 0)),
            (SIMD3(0, -1, 0), SIMD2(0.5, 1)),
            (SIMD3(0.48, 0.6, -0.64), SIMD2(0.352416382, 0.295167235)),
            (SIMD3(1, 1.5, 0), SIMD2(0.5, 0)),
            (SIMD3(-1, 0, 0.01), SIMD2(0.998408504, 0.5)),
            (SIMD3(-1, 0, -0.01), SIMD2(0.001591496, 0.5)),
        ]
        let matcaps: [(SIMD3<Float>, SIMD3<Float>, SIMD2<Float>)] = [
            (SIMD3(0.6, 0, 0.8), SIMD3(0, 0, 1), SIMD2(0.797, 0.5)),
            (SIMD3(-0.6, 0.8, 0), SIMD3(0.6, 0, 0.8), SIMD2(0.2624, 0.104)),
            (SIMD3(0.6, 0, 0.8), SIMD3(0.36, 0.48, 0.8), SIMD2(0.608336275, 0.731839628)),
            // The explicit overload normalizes its inputs and defines the vertical/zero cases.
            (SIMD3(6, 0, 8), SIMD3(0, 0, 100), SIMD2(0.797, 0.5)),
            (SIMD3(0.6, 0, 0.8), SIMD3(0, 1, 0), SIMD2(0.797, 0.896)),
            (SIMD3(0.6, 0, 0.8), .zero, SIMD2(0.797, 0.5)),
        ]
        let sprites: [(SIMD2<Float>, Float, SIMD2<Float>)] = [
            (SIMD2(3, 2), 0, SIMD2(1.0 / 12, 0.375)),
            (SIMD2(3, 2), 2.9, SIMD2(0.75, 0.375)),
            (SIMD2(3, 2), 3, SIMD2(1.0 / 12, 0.875)),
            (SIMD2(3, 2), 5, SIMD2(0.75, 0.875)),
            (SIMD2(3, 2), 6, SIMD2(1.0 / 12, 0.375)),
            (SIMD2(3, 2), -0.25, SIMD2(0.75, 0.875)),
            (SIMD2(3, 2), -6, SIMD2(1.0 / 12, 0.375)),
            (SIMD2(3, 2), -6.1, SIMD2(0.75, 0.875)),
            (SIMD2(3, 2), 13.9, SIMD2(5.0 / 12, 0.375)),
            // A finite one-cell grid and the maximum Float-exact cell count.
            (SIMD2(-5, 0.5), 9, SIMD2(0.25, 0.75)),
            (SIMD2(16_777_216, 2), -1, SIMD2(0.99999994, 0.75)),
        ]
        var records: [SIMD4<Float>] = []
        var expected: [(SIMD4<Float>, SIMD4<Float>)] = []
        for (index, (grid, frame, sprite)) in sprites.enumerated() {
            let (direction, equirect) = panorama[index % panorama.count]
            let (normal, viewer, matcap) = matcaps[index % matcaps.count]
            records += [SIMD4(direction, 0), SIMD4(normal, 0), SIMD4(viewer, 0), SIMD4(grid.x, grid.y, frame, 0)]
            expected.append((SIMD4(equirect.x, equirect.y, matcap.x, matcap.y), SIMD4(sprite.x, sprite.y, 0, 0)))
        }
        let size = records.count * MemoryLayout<SIMD4<Float>>.stride
        let source = try #require(gpu.device.makeBuffer(bytes: records, length: size, options: .storageModeShared))
        let destination = try #require(gpu.device.makeBuffer(length: size, options: .storageModeShared))
        let command = try #require(gpu.queue.makeCommandBuffer())
        let kernel = try ShaderBufferKernel(graph: graph, device: gpu.device)
        try kernel.encode(elementCount: expected.count, from: source, to: destination, commandBuffer: command)
        try gpu.finish(command)
        let output = destination.contents().bindMemory(to: SIMD4<Float>.self, capacity: records.count)
        for (index, (projection, atlas)) in expected.enumerated() {
            #expect(isNearly(output[index * 4], projection, tolerance: 0.00002), "Projection record \(index)")
            #expect(isNearly(output[index * 4 + 1], atlas, tolerance: 0.00002), "Atlas record \(index)")
        }
    }

    @Test("Surface matcap follows camera roll and the raster or traced viewer direction")
    func surfaceMatcap() throws {
        let gpu = try GeometryRenderFixture(side: 33)
        let graph = try SurfaceShaderGraph { b in
            .init(baseColor: b.vector(b.matcapUV(), b.constant(Float(0))))
        }
        let material = try PhysicallyBasedMaterial(surfaceGraph: graph, shadingModel: .unlit)
        let square = try GeometryRenderFixture.square(1)
        let geometry = try MeshGeometry(vertices: square.vertices.map {
            MeshVertex(position: $0.position, normal: SIMD3(0.6, 0, 0.8), textureCoordinate: $0.textureCoordinate)
        }, indices: square.indices)
        let mesh = try gpu.unlit(geometry, material: .physicallyBased(material))
        let surface = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)
        var scene = SceneGraph(roots: [surface], camera: gpu.camera)
        let unrolled = try gpu.render(scene).pixels
        #expect(isNearly(gpu.pixel(unrolled, x: 16, y: 16), SIMD4(0.797, 0.5, 0, 1), tolerance: 0.001))

        scene.cameraTransform.rotation = simd_quatf(angle: .pi / 2, axis: SIMD3(0, 0, 1))
        let rolled = try gpu.render(scene).pixels
        #expect(isNearly(gpu.pixel(rolled, x: 16, y: 16), SIMD4(0.5, 0.797, 0, 1), tolerance: 0.001))
        #expect(isNearly(gpu.pixel(rolled, x: 21, y: 16), SIMD4(0.614842929, 0.797, 0, 1), tolerance: 0.001))

        let tracer = try PathTracer(queue: gpu.queue, scene: PathTracingScene(queue: gpu.queue, graph: scene))
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: gpu.side, height: gpu.side, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderWrite
        let target = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let command = try #require(gpu.queue.makeCommandBuffer())
        let accumulation = try tracer.makeAccumulation(width: gpu.side, height: gpu.side)
        _ = try tracer.encode(
            cameraModel: .init(projection: .orthographic(height: 2)),
            cameraTransform: scene.cameraTransform, destination: target,
            accumulation: accumulation, commandBuffer: command
        )
        try gpu.finish(command)
        let traced = try gpu.pixels(of: target)
        let expected = SIMD4<Float>(0.5, 0.797, 0, 1)
        #expect(isNearly(gpu.pixel(traced, x: 16, y: 16), expected, tolerance: 0.001))
        #expect(isNearly(gpu.pixel(traced, x: 21, y: 16), expected, tolerance: 0.001))
    }

    @Test("A live surface frame selects the correct tile of a top-left sprite texture")
    func surfaceAtlas() throws {
        let gpu = try GeometryRenderFixture(side: 33)
        let graph = try SurfaceShaderGraph { b in
            let atlas = b.texture("Atlas", content: .data, addressing: .clampToEdge, filtering: .nearest)
            let frame = b.uniform("Frame", Float(0))
            return .init(baseColor: atlas.sample(b.spritesheetUV(b.constant(SIMD2<Float>(3, 2)), frame: frame)).rgb)
        }
        let colors: [SIMD4<Float>] = [
            SIMD4(1, 0, 0, 1), SIMD4(0, 1, 0, 1), SIMD4(0, 0, 1, 1),
            SIMD4(1, 1, 0, 1), SIMD4(0, 1, 1, 1), SIMD4(1, 0, 1, 1),
        ]
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 3, height: 2, mipmapped: false)
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        let atlas = try #require(gpu.device.makeTexture(descriptor: descriptor))
        atlas.replace(region: MTLRegionMake2D(0, 0, 3, 2), mipmapLevel: 0, withBytes: colors, bytesPerRow: 48)
        let uniforms = try ShaderUniforms(graph: graph)
        let material = try PhysicallyBasedMaterial(surfaceGraph: graph, surfaceUniforms: uniforms,
                                                   surfaceTextures: ["Atlas": atlas], shadingModel: .unlit)
        let mesh = try gpu.unlit(.plane(width: 2, height: 2), material: .physicallyBased(material))
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)
        let scene = SceneGraph(roots: [node], camera: gpu.camera)
        for (frame, tile): (Float, Int) in [(0, 0), (2.99, 2), (3, 3), (6, 0), (-0.25, 5)] {
            try uniforms.set("Frame", to: frame)
            let rendered = try gpu.render(scene).pixels
            #expect(gpu.pixel(rendered, x: 16, y: 16) == colors[tile], "Frame \(frame)")
        }
    }
}
