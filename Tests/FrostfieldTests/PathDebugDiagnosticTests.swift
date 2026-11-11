import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathDebugDiagnosticTests {
    @Test("Completed tile sample counts reach the Metal viewer in top-left order")
    func sampleDensity() throws {
        let fixture = try PathTestFixture()
        let mesh = try fixture.pbr(.plane(width: 2, height: 2))
        let graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)], camera: fixture.camera
        )
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: .software))
        let tracer = try PathTracer(queue: fixture.queue, scene: scene)
        let accumulation = try tracer.makeAccumulation(width: 2, height: 2)
        let output = try fixture.output(width: 2, height: 2)
        let region = try PathTracer.Region(imageWidth: 5, imageHeight: 3, x: 1, y: 0)
        let command = try #require(fixture.queue.makeCommandBuffer())
        let first = try tracer.encode(
            camera: fixture.camera, cameraTransform: .init(), destination: output, accumulation: accumulation,
            commandBuffer: command, samplesPerPixel: 3, region: region
        )
        #expect(throws: PathTracer.SampleDensity.Failure.history) { try tracer.sampleDensity([first]) }
        try fixture.finish(command)

        let next = try #require(fixture.queue.makeCommandBuffer())
        let latest = try tracer.encode(
            camera: fixture.camera, cameraTransform: .init(), destination: output, accumulation: accumulation,
            previous: first, commandBuffer: next, samplesPerPixel: 2, region: region
        )
        let firstSnapshot = try tracer.sampleDensity([first])
        #expect(firstSnapshot.counts.max() == 3)
        try fixture.finish(next)

        func tile(region: PathTracer.Region?) throws -> PathTracer.History {
            let buffer = try #require(fixture.queue.makeCommandBuffer())
            let history = try tracer.encode(
                camera: fixture.camera, cameraTransform: .init(), destination: fixture.output(width: 1, height: 2),
                accumulation: tracer.makeAccumulation(width: 1, height: 2), commandBuffer: buffer,
                samplesPerPixel: 2, region: region
            )
            try fixture.finish(buffer)
            return history
        }
        let right = try tile(region: .init(imageWidth: 5, imageHeight: 3, x: 4, y: 1))
        let otherSize = try tile(region: nil)
        #expect(throws: PathTracer.SampleDensity.Failure.history) { try tracer.sampleDensity([first]) }
        #expect(throws: PathTracer.SampleDensity.Failure.overlappingWindows) { try tracer.sampleDensity([latest, latest]) }
        #expect(throws: PathTracer.SampleDensity.Failure.imageDimensions) { try tracer.sampleDensity([latest, otherSize]) }
        #expect(throws: PathTracer.SampleDensity.Failure.imageDimensions) { try tracer.sampleDensity([]) }
        let anotherTracer = try PathTracer(queue: fixture.queue, scene: scene)
        #expect(throws: PathTracer.SampleDensity.Failure.history) { try anotherTracer.sampleDensity([latest]) }

        let density = try tracer.sampleDensity([right, latest])
        #expect(density.width == 5 && density.height == 3)
        #expect(density.counts == [0, 5, 5, 0, 0, 0, 5, 5, 0, 2, 0, 0, 0, 0, 2])
        #expect(firstSnapshot.counts.max() == 3)
        let raw = try density.makeTexture(device: fixture.device)
        #expect(raw.pixelFormat == .r32Float)
        var bytes = [Float](repeating: -1, count: 15)
        bytes.withUnsafeMutableBytes {
            raw.getBytes($0.baseAddress!, bytesPerRow: 5 * MemoryLayout<Float>.stride,
                         from: MTLRegionMake2D(0, 0, 5, 3), mipmapLevel: 0)
        }
        #expect(bytes == density.counts.map(Float.init))

        #expect(throws: PathTracer.SampleDensity.Failure.displayRange) {
            try density.makeTexture(device: fixture.device, displayRange: 2...2)
        }
        let display = try density.makeTexture(device: fixture.device, displayRange: 1...5)
        let viewer = try TextureDebugPass(device: fixture.device)
        let raster = try RasterLightingFixture(width: 5, height: 3)
        let target = try raster.colorTarget()
        let show = try #require(fixture.queue.makeCommandBuffer())
        try viewer.encode(source: .color(display, channel: .red), destination: target, commandBuffer: show)
        try fixture.finish(show)
        #expect(try fixture.readback(target) == density.counts.map {
            let value = $0 == 5 ? Float(1) : $0 == 2 ? 0.25 : 0
            return SIMD4(value, value, value, 1)
        })
    }

    @Test("Software TLAS and instanced BLAS snapshots track refits and draw their actual edges")
    func softwareBounds() throws {
        let fixture = try PrimitiveRenderFixture()
        let geometry = MeshGeometry.cube()
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .physicallyBased(try .init(alphaMode: .opaque)), supportsInstancing: true
        )
        let slotTransforms = [
            SceneTransform(translation: SIMD3(-1, 0, 0), scale: SIMD3(-1, 1, 1)).matrix,
            SceneTransform(
                translation: SIMD3(1, 0, 0), rotation: simd_quatf(angle: 0.4, axis: SIMD3(0, 1, 0)),
                scale: SIMD3(-1, 1, 1)
            ).matrix,
        ]
        let slots = try InstanceSet(device: fixture.device, transforms: slotTransforms)
        let node = SceneNode(
            transform: .init(rotation: simd_quatf(angle: 0.4, axis: SIMD3(0, 0, 1))), mesh: mesh, instances: slots
        )
        let parent = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -4), scale: SIMD3(1.4, 0.9, 1.1)), children: [node]
        )
        var graph = fixture.graph([parent])
        let queue = try #require(fixture.device.makeCommandQueue())
        let scene = try PathTracingScene(queue: queue, graph: graph, options: .init(acceleration: .software))
        let localBox = try AxisAlignedBox(minimum: SIMD3(repeating: -0.5), maximum: SIMD3(repeating: 0.5))
        let worldTransforms = slotTransforms.map { parent.transform.matrix * node.transform.matrix * $0 }
        let worldBoxes = try worldTransforms.map { try localBox.transformed(by: $0) }

        let top = try scene.softwareBVHBounds(maximumDepth: 1)
        #expect(top.nodes.map(\.depth) == [0, 1, 1])
        #expect(top.nodes.map(\.leafPrimitiveCount) == [nil, 1, 1])
        let unionMin = simd_min(worldBoxes[0].minimum, worldBoxes[1].minimum)
        let unionMax = simd_max(worldBoxes[0].maximum, worldBoxes[1].maximum)
        #expect(simd_distance(top.nodes[0].bounds.minimum, unionMin) < 1e-5)
        #expect(simd_distance(top.nodes[0].bounds.maximum, unionMax) < 1e-5)
        #expect(try scene.softwareBVHBounds(maximumDepth: 0).nodes == [top.nodes[0]])
        #expect(throws: PathTracingScene.BVHBounds.Failure.depth) { try scene.softwareBVHBounds(maximumDepth: -1) }
        #expect(throws: PathTracingScene.BVHBounds.Failure.missingInstance) {
            try scene.softwareBVHBounds(level: .bottomLevel(nodeID: node.id))
        }

        let level = PathTracingScene.BVHBounds.Level.bottomLevel(nodeID: node.id, instanceIndex: 1)
        let bottom = try scene.softwareBVHBounds(level: level, maximumDepth: 12)
        #expect(bottom.nodes[0].bounds == localBox)
        #expect(bottom.nodes.compactMap(\.leafPrimitiveCount).reduce(0, +) == geometry.indices.count / 3)
        #expect(Set(bottom.nodes.map(\.index)).count == bottom.nodes.count)
        let root = try scene.softwareBVHBounds(level: level, maximumDepth: 0)
        let lines = try root.lineGeometry(color: SIMD4(1, 0.5, 0, 1))
        #expect(lines.segments.count == 12)
        let expected = worldTransforms[1] * SIMD4<Float>(0.5, 0.5, 0.5, 1)
        #expect(simd_distance(lines.positions[0], SIMD3(expected.x, expected.y, expected.z)) < 1e-6)

        let helpers = DebugHelpers(device: fixture.device, lineMaterial: .init(width: .pixels(2), depthTest: false))
        let drawn = try fixture.render(fixture.graph([helpers.node(lines)]))
        let midpoint = fixture.pixel((lines.positions[0] + lines.positions[1]) / 2)
        let x = Int(midpoint.x), y = Int(midpoint.y)
        try #require((1..<(fixture.side - 1)).contains(x) && (1..<(fixture.side - 1)).contains(y))
        #expect((-1...1).contains { dy in (-1...1).contains { dx in
            let pixel = drawn.pixels[(y + dy) * fixture.side + x + dx]
            return pixel.x > 0.75 && abs(pixel.y / pixel.x - 0.5) < 0.02
        } })
        #expect(drawn.pixels.filter { $0.x > 0.5 }.count > 20)
        #expect(drawn.pixels[0] == .zero)

        graph.roots[0].transform.translation.y += 2
        #expect(try scene.updateTransforms(graph))
        let moved = try scene.softwareBVHBounds(level: level, maximumDepth: 0)
        let movedLines = try moved.lineGeometry()
        #expect(moved.revision == root.revision + 1)
        #expect(moved.nodes == root.nodes)
        #expect(simd_distance(movedLines.positions[0] - lines.positions[0], SIMD3(0, 2, 0)) < 1e-6)
        #expect(try scene.softwareBVHBounds(maximumDepth: 0).nodes[0].bounds.center.y > top.nodes[0].bounds.center.y + 1.9)
    }

    @Test("Metal acceleration does not invent hardware BVH nodes",
          .enabled(if: MTLCreateSystemDefaultDevice()?.supportsRaytracing == true, "Requires Metal ray tracing"))
    func hardwareBounds() throws {
        let fixture = try PathTestFixture()
        let graph = SceneGraph(roots: [SceneNode(mesh: try fixture.pbr(.cube()))], camera: fixture.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: .metal))
        #expect(throws: PathTracingScene.BVHBounds.Failure.hardwareAcceleration) { try scene.softwareBVHBounds() }
    }
}
