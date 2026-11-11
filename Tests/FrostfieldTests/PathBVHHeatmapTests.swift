import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathBVHHeatmapTests {
    private let zero = SIMD4<Float>(0, 0, 0.5, 1)
    private let one = SIMD4<Float>(0, 0, 1, 1)
    private let two = SIMD4<Float>(0, 0.5, 1, 1)
    private let three = SIMD4<Float>(0, 1, 1, 1)
    private let five = SIMD4<Float>(1, 1, 0, 1)

    private struct Fixture {
        let gpu: PathTestFixture
        var graph: SceneGraph
        let scene: PathTracingScene
        let tracer: PathTracer
        let model = PathTracer.CameraModel(projection: .orthographic(height: 1))

        init(
            frontMask: Bool = false, frontBackface: Bool = false, frontMaterial: PhysicallyBasedMaterial? = nil,
            metal: Bool = false
        ) throws {
            gpu = try PathTestFixture()
            let plane = try MeshGeometry.plane(width: 1, height: 1)
            let back = try gpu.pbr(plane)
            let front = try MeshRenderer(
                device: gpu.device, colorPixelFormat: .rgba16Float, geometry: plane,
                material: .physicallyBased(try frontMaterial ?? .init(
                    baseColorFactor: SIMD4(1, 1, 1, frontMask ? 0 : 1),
                    alphaMode: frontMask ? .mask(cutoff: 0.5) : .opaque
                ))
            )
            let turn = frontBackface ? simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
                : simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))
            graph = SceneGraph(roots: [
                SceneNode(transform: .init(translation: SIMD3(0, 0, -2), rotation: turn), mesh: front),
                SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: back),
            ], camera: gpu.camera)
            scene = try PathTracingScene(
                queue: gpu.queue, graph: graph, options: .init(acceleration: metal ? .metal : .software)
            )
            tracer = try PathTracer(queue: gpu.queue, scene: scene)
        }

        func pixels(
            _ options: PathTracer.BVHHeatmapOptions, model: PathTracer.CameraModel? = nil,
            width: Int = 3, region: PathTracer.Region? = nil
        ) throws -> [SIMD4<Float>] {
            let target = try gpu.output(width: width, height: 1)
            let command = try #require(gpu.queue.makeCommandBuffer())
            try tracer.encodeBVHHeatmap(
                cameraModel: model ?? self.model, destination: target, commandBuffer: command,
                options: options, region: region
            )
            try gpu.finish(command)
            return try gpu.readback(target)
        }
    }

    @Test("A two-instance tree counts independent TLAS and BLAS hits on the GPU")
    func boxCounts() throws {
        let fixture = try Fixture()
        // The center ray intersects the TLAS root, two TLAS leaves and one
        // BLAS leaf per quad. The outside rays miss the TLAS root.
        #expect(try fixture.pixels(.init(saturationCount: 8)) == [zero, five, zero])
        #expect(try fixture.pixels(.init(displayBottomLevel: false, saturationCount: 8)) == [zero, three, zero])
        #expect(try fixture.pixels(.init(displayTopLevel: false, saturationCount: 8)) == [zero, two, zero])
        #expect(try fixture.pixels(.init(
            displayTopLevel: false, displayBottomLevel: false, stopAtSurface: true, saturationCount: 8
        )) == [zero, zero, zero])

        let center = try PathTracer.Region(imageWidth: 3, imageHeight: 1, x: 1, y: 0)
        #expect(try fixture.pixels(.init(saturationCount: 8), width: 1, region: center) == [five])
        let offAxis = PathTracer.CameraModel(projection: .perspective(verticalFieldOfView: .pi / 2, shift: SIMD2(0.5, 0)))
        #expect(try fixture.pixels(.init(saturationCount: 8), model: offAxis, width: 1) == [zero])
        let finiteClip = PathTracer.CameraModel(projection: .orthographic(height: 1), far: 3)
        #expect(try fixture.pixels(.init(saturationCount: 8), model: finiteClip) == [zero, three, zero])
    }

    @Test("First-surface cutoff uses mesh culling and material coverage")
    func surfaceCutoff() throws {
        let opaque = try Fixture()
        let mask = try Fixture(frontMask: true)
        let culled = try Fixture(frontBackface: true)
        let options = PathTracer.BVHHeatmapOptions(stopAtSurface: true, saturationCount: 8)
        // Opaque near quad excludes the far TLAS leaf and BLAS. A mask hole or
        // culled back face leaves the same two bounds visible up to the far quad.
        #expect(try opaque.pixels(options) == [zero, three, zero])
        #expect(try mask.pixels(options) == [zero, five, zero])
        #expect(try culled.pixels(options) == [zero, five, zero])
        #expect(try opaque.pixels(.init(displayBottomLevel: false, stopAtSurface: true, saturationCount: 8))
                == [zero, two, zero])
        #expect(try opaque.pixels(.init(displayTopLevel: false, stopAtSurface: true, saturationCount: 8))
                == [zero, one, zero])
    }

    @Test("The cutoff reads a live coverage graph on the prepared scene")
    func liveGraph() throws {
        let graph = try SurfaceShaderGraph { graph in .init(mask: graph.uniform("Visible", Float(0))) }
        let material = try PhysicallyBasedMaterial(alphaMode: .opaque, surfaceGraph: graph)
        let fixture = try Fixture(frontMaterial: material)
        let options = PathTracer.BVHHeatmapOptions(stopAtSurface: true, saturationCount: 8)
        #expect(try fixture.pixels(options) == [zero, five, zero])
        try #require(material.surfaceUniforms).set("Visible", to: Float(1))
        #expect(try fixture.pixels(options) == [zero, three, zero])
    }

    @Test("A split BLAS counts its root and both leaves before applying a surface cutoff")
    func interiorBottomLevel() throws {
        let gpu = try PathTestFixture()
        let plane = try MeshGeometry.plane(width: 1, height: 1)
        let parts: [MeshGeometry.MergePart] = [-2, -4, -6].map { z in
            .init(plane, transform: SceneTransform(translation: SIMD3(0, 0, z)).matrix)
        }
        let geometry = try MeshGeometry.merged(parts).geometry
        let mesh = try gpu.pbr(geometry)
        let graph = SceneGraph(roots: [SceneNode(mesh: mesh)], camera: gpu.camera)
        let scene = try PathTracingScene(queue: gpu.queue, graph: graph, options: .init(acceleration: .software))
        let tracer = try PathTracer(queue: gpu.queue, scene: scene)
        let target = try gpu.output(width: 1, height: 1)
        let model = PathTracer.CameraModel(projection: .orthographic(height: 1))
        for (cutoff, expected) in [(false, three), (true, two)] {
            let command = try #require(gpu.queue.makeCommandBuffer())
            try tracer.encodeBVHHeatmap(
                cameraModel: model, destination: target, commandBuffer: command,
                options: .init(displayTopLevel: false, stopAtSurface: cutoff, saturationCount: 8)
            )
            try gpu.finish(command)
            #expect(try gpu.readback(target) == [expected])
        }
    }

    @Test("Diagnostic work has an explicit node budget and rejects invalid commands and targets")
    func validation() throws {
        let fixture = try Fixture()
        let magenta = SIMD4<Float>(1, 0, 1, 1)
        #expect(try fixture.pixels(.init(saturationCount: 8, maximumNodeVisits: 1)) == [zero, magenta, zero])
        let saturated = SIMD4<Float>(0.5, 0, 0, 1)
        #expect(try fixture.pixels(.init(saturationCount: 1, maximumNodeVisits: 1)) == [zero, saturated, zero])
        #expect(try fixture.pixels(.init(stopAtSurface: true, saturationCount: 8, maximumNodeVisits: 1))
                == [zero, magenta, zero])
        #expect(throws: PathTracer.Failure.boundsHeatmapSettings) {
            try fixture.pixels(.init(saturationCount: 0))
        }
        #expect(throws: PathTracer.Failure.boundsHeatmapSettings) {
            try fixture.pixels(.init(maximumNodeVisits: 65_537))
        }
        let command = try #require(fixture.gpu.queue.makeCommandBuffer())
        let readOnly = try fixture.gpu.output(width: 3, height: 1, usage: .shaderRead)
        #expect(throws: PathTracer.Failure.boundsHeatmapOutput) {
            try fixture.tracer.encodeBVHHeatmap(cameraModel: fixture.model, destination: readOnly, commandBuffer: command)
        }
        let wrongQueue = try #require(fixture.gpu.device.makeCommandQueue())
        let wrongCommand = try #require(wrongQueue.makeCommandBuffer())
        let writable = try fixture.gpu.output(width: 3, height: 1)
        #expect(throws: PathTracer.Failure.commandBuffer) {
            try fixture.tracer.encodeBVHHeatmap(cameraModel: fixture.model, destination: writable, commandBuffer: wrongCommand)
        }
    }

    @Test("Encoded views retain the corresponding BVH across transform changes")
    func retainedRefit() throws {
        var fixture = try Fixture()
        let command = try #require(fixture.gpu.queue.makeCommandBuffer())
        let previous = try fixture.gpu.output(width: 3, height: 1)
        let updated = try fixture.gpu.output(width: 3, height: 1)
        try fixture.tracer.encodeBVHHeatmap(
            cameraModel: fixture.model, destination: previous, commandBuffer: command, options: .init(saturationCount: 8)
        )
        fixture.graph.roots[0].transform.translation.x = 1
        #expect(try fixture.scene.updateTransforms(fixture.graph))
        try fixture.tracer.encodeBVHHeatmap(
            cameraModel: fixture.model, destination: updated, commandBuffer: command, options: .init(saturationCount: 8)
        )
        try fixture.gpu.finish(command)
        #expect(try fixture.gpu.readback(previous) == [zero, five, zero])
        #expect(try fixture.gpu.readback(updated) == [zero, three, three])
    }

    @Test("The diagnostic rejects opaque hardware acceleration structures",
          .enabled(if: MTLCreateSystemDefaultDevice()?.supportsRaytracing == true, "Requires Metal ray tracing"))
    func hardware() throws {
        let fixture = try Fixture(metal: true)
        #expect(throws: PathTracingScene.BVHBounds.Failure.hardwareAcceleration) {
            try fixture.pixels(.init())
        }
    }
}
