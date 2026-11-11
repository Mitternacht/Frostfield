import Foundation
import Frostfield
import Metal
import Synchronization
import Testing
import simd

/// Software and Metal ray-tracing acceleration: watertight hits, matching images, limits,
/// asynchronous preparation and updates.
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathAccelerationTests {
    private let fixture: PathTestFixture

    init() throws {
        fixture = try PathTestFixture()
    }

    /// The accelerations this device can trace.
    static var modes: [PathTracingScene.Options.Acceleration] {
        MTLCreateSystemDefaultDevice()?.supportsRaytracing == true ? [.software, .metal] : [.software]
    }

    /// One pinhole sample through each pixel center: the first-hit albedo guide.
    private func guide(
        _ graph: SceneGraph, width: Int, height: Int, acceleration: PathTracingScene.Options.Acceleration
    ) throws -> [SIMD4<Float>] {
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: PathTracer.Lighting(environment: .zero))
        let accumulation = try tracer.makeAccumulation(width: width, height: height)
        let guides = PathTracer.Guides(
            normalDepth: try fixture.output(width: width, height: height),
            albedoRoughness: try fixture.output(width: width, height: height)
        )
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        _ = try tracer.encode(
            camera: graph.camera, cameraTransform: graph.cameraTransform,
            destination: try fixture.output(width: width, height: height), accumulation: accumulation, guides: guides,
            commandBuffer: buffer, settings: .init(maxBounces: 1, seed: 3)
        )
        try fixture.finish(buffer)
        return try fixture.readback(guides.albedoRoughness)
    }

    private func unlitWhite(_ geometry: MeshGeometry) throws -> MeshRenderer {
        try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .physicallyBased(try PhysicallyBasedMaterial(alphaMode: .opaque, shadingModel: .unlit))
        )
    }

    @Test("Rays through the shared diagonal of a quad's two triangles always hit one of them", arguments: modes)
    func watertightDiagonal(acceleration: PathTracingScene.Options.Acceleration) throws {
        // The plane's triangles share the edge from (-w/2, -h/2) to (w/2, h/2). A 90° view of a
        // square at distance d whose half size is d puts the pixel centers with x + y = side - 1
        // on that edge, up to rounding.
        var diagonal = 0
        var misses = 0
        for (side, distance, size) in [(64, 1, 2), (65, 3, 6), (127, 0.75, 1.5), (128, 7, 14), (255, 2.5, 5)] as [(Int, Float, Float)] {
            for yaw in [Float(0), 0.3, -0.45] {
                let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.01, far: 100)
                let rotation = simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0))
                let node = SceneNode(
                    transform: SceneTransform(translation: rotation.act(SIMD3(0, 0, -distance)), rotation: rotation),
                    mesh: try unlitWhite(.plane(width: size, height: size))
                )
                let view = SceneTransform(rotation: rotation)
                let albedo = try guide(
                    SceneGraph(roots: [node], camera: camera, cameraTransform: view), width: side, height: side,
                    acceleration: acceleration
                )
                for y in 0..<side {
                    let x = side - 1 - y
                    diagonal += 1
                    if albedo[y * side + x].x < 0.5 { misses += 1 }
                }
            }
        }
        #expect(diagonal > 1_500)
        #expect(misses == 0, "\(misses) of \(diagonal) diagonal rays missed both triangles")
    }

    @Test("Automatic acceleration follows the device's ray-tracing capability")
    func automaticAcceleration() throws {
        let mesh = try unlitWhite(.plane(width: 2, height: 2))
        let graph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)], camera: fixture.camera
        )
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph)
        let expected: PathTracingScene.Options.Acceleration = fixture.device.supportsRaytracing ? .metal : .software
        #expect(scene.accelerationKind == expected)
        print("Path acceleration selected \(scene.accelerationKind) (supportsRaytracing: \(fixture.device.supportsRaytracing))")
        let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: .init(environment: .zero))
        let image = try fixture.trace(tracer, width: 16, height: 16, samples: 1, settings: .init(maxBounces: 1))
        #expect(image[8 * 16 + 8] == SIMD4(1, 1, 1, 1))
        if !fixture.device.supportsRaytracing {
            #expect(throws: PathTracingScene.Failure.rayTracingUnsupported) {
                try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: .metal))
            }
        }
    }

    // MARK: - Matching images

    private func texture(size: Int, pixel: (Int, Int) -> [UInt8]) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: size, height: size, mipmapped: true
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let bytes = (0..<(size * size)).flatMap { pixel($0 % size, $0 / size) }
        bytes.withUnsafeBytes { raw in
            texture.replace(region: MTLRegionMake2D(0, 0, size, size), mipmapLevel: 0, withBytes: raw.baseAddress!, bytesPerRow: size * 4)
        }
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        let blit = try #require(buffer.makeBlitCommandEncoder())
        blit.generateMipmaps(for: texture)
        blit.endEncoding()
        try fixture.finish(buffer)
        return texture
    }

    private func renderer(
        _ geometry: MeshGeometry, _ material: PhysicallyBasedMaterial, instancing: Bool = false
    ) throws -> MeshRenderer {
        try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .physicallyBased(material), supportsInstancing: instancing
        )
    }

    /// A textured floor, a rough cube, glass and metal spheres, a masked leaf, a half-transparent
    /// blended pane and three mirrored instances, under one area light and a dim sky.
    private func studio() throws -> SceneGraph {
        let checker = try texture(size: 64) { x, y in (x / 8 + y / 8).isMultiple(of: 2) ? [230, 220, 200, 255] : [40, 50, 70, 255] }
        let disc = try texture(size: 32) { x, y in
            let distance = simd_length(SIMD2(Float(x) - 15.5, Float(y) - 15.5))
            return [60, 160, 60, distance < 13 ? 255 : 0]
        }
        let up = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
        let floor = try renderer(.plane(width: 8, height: 8), try PhysicallyBasedMaterial(
            baseColorTexture: checker, roughness: 0.7, lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque
        ))
        let cube = try renderer(.cube(), try PhysicallyBasedMaterial(
            roughness: 0.9, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0.8, 0.3, 0.2, 1), alphaMode: .opaque
        ))
        let glass = try renderer(.uvSphere(radius: 0.45, segments: 48, rings: 24), try PhysicallyBasedMaterial(
            roughness: 0.05, lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque,
            transmission: try .init(factor: 1, indexOfRefraction: 1.5, thickness: 0.9)
        ))
        let metal = try renderer(.uvSphere(radius: 0.4), try PhysicallyBasedMaterial(
            metallic: 1, roughness: 0.25, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0.95, 0.8, 0.5, 1),
            alphaMode: .opaque
        ))
        let leaf = try renderer(.plane(width: 0.9, height: 0.9), try PhysicallyBasedMaterial(
            baseColorTexture: disc, roughness: 0.6, lightIntensity: 0, ambientIntensity: 0, alphaMode: .mask(cutoff: 0.5),
            isDoubleSided: true
        ))
        let pane = try renderer(.plane(width: 1.2, height: 0.8), try PhysicallyBasedMaterial(
            roughness: 0.4, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0.3, 0.5, 0.9, 0.4), alphaMode: .blend
        ))
        let box = try renderer(.cube(), try PhysicallyBasedMaterial(
            roughness: 0.5, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0.4, 0.7, 0.4, 1), alphaMode: .opaque
        ), instancing: true)
        let mirrored = try (0..<3).map { index -> simd_float4x4 in
            SceneTransform(
                translation: SIMD3(Float(index) * 0.45 - 0.45, 0, 0),
                rotation: simd_quatf(angle: Float(index) * 0.4, axis: SIMD3(0, 1, 0)), scale: SIMD3(-0.3, 0.3, 0.3)
            ).matrix
        }
        return SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, -1, -4), rotation: up), mesh: floor),
            SceneNode(
                transform: .init(translation: SIMD3(-1.4, -0.5, -4.3), rotation: simd_quatf(angle: 0.6, axis: SIMD3(0, 1, 0))),
                mesh: cube
            ),
            SceneNode(transform: .init(translation: SIMD3(-0.2, -0.55, -3.4)), mesh: glass),
            SceneNode(transform: .init(translation: SIMD3(1.1, -0.6, -4.2)), mesh: metal),
            SceneNode(
                transform: .init(translation: SIMD3(0.6, 0.2, -3.2), rotation: simd_quatf(angle: 0.7, axis: SIMD3(0, 1, 0))),
                mesh: leaf
            ),
            SceneNode(transform: .init(translation: SIMD3(-0.9, 0.4, -3.0)), mesh: pane),
            SceneNode(
                transform: .init(translation: SIMD3(0.3, -0.85, -5.0)), mesh: box,
                instances: try InstanceSet(device: fixture.device, transforms: try mirrored)
            ),
        ], camera: fixture.camera)
    }

    private func studioLighting() throws -> PathTracer.Lighting {
        try PathTracer.Lighting(
            lights: [try PathTracer.AreaLight(
                center: SIMD3(0.5, 2.5, -3), aim: SIMD3(0, -1, -4), size: SIMD2(1.5, 1), radiance: SIMD3(repeating: 12)
            )],
            environment: SIMD3(0.12, 0.14, 0.18)
        )
    }

    @Test(
        "Software and Metal acceleration trace the same studio within floating-point tolerance",
        .enabled(if: MTLCreateSystemDefaultDevice()?.supportsRaytracing == true, "Requires Metal ray tracing")
    )
    func matchingImages() throws {
        let graph = try studio()
        let lighting = try studioLighting()
        var images: [[SIMD4<Float>]] = []
        for acceleration in [PathTracingScene.Options.Acceleration.software, .metal] {
            let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
            #expect(scene.accelerationKind == acceleration)
            let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: lighting)
            images.append(try fixture.trace(
                tracer, width: 96, height: 64, samples: 32, settings: .init(maxBounces: 6, seed: 21)
            ))
        }
        try PathFullResolutionTextureTests.save(images[0], width: 96, height: 64, name: "studio-software")
        try PathFullResolutionTextureTests.save(images[1], width: 96, height: 64, name: "studio-metal")
        let differences = zip(images[0], images[1]).map { simd_reduce_max(abs($0 - $1)) }
        let mean = differences.reduce(0, +) / Float(differences.count)
        let close = differences.count { $0 <= 1e-3 }
        let brightness = images[0].reduce(Float(0)) { $0 + $1.x + $1.y + $1.z } / Float(3 * images[0].count)
        print("Software against Metal: mean difference \(mean), \(close) of \(differences.count) pixels within 1e-3, max \(differences.max()!), mean radiance \(brightness)")
        #expect(brightness > 0.05)
        // The same seed follows the same paths; only rounding in hit points and barycentrics differs.
        #expect(mean < 2e-3)
        #expect(close >= differences.count * 95 / 100)
    }

    // MARK: - Updates

    @Test("An instance set that is empty at preparation does not block later transform and geometry updates", arguments: modes)
    func emptyInstanceSet(acceleration: PathTracingScene.Options.Acceleration) throws {
        let box = try renderer(.cube(), try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, emissiveFactor: SIMD3(1, 1, 1), alphaMode: .opaque
        ), instancing: true)
        let empty = try InstanceSet(device: fixture.device, transforms: [], capacity: 4)
        let emptyID = UUID()
        var graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -4)), mesh: box),
            SceneNode(id: emptyID, transform: .init(translation: SIMD3(1, 0, -4)), mesh: box, instances: empty),
        ], camera: fixture.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        #expect(scene.statistics.instanceCount == 1)
        #expect(try scene.updateTransforms(graph) == false)
        graph.roots[0].transform.translation.x = -1
        graph.roots[1].transform.translation.y = 1
        #expect(try scene.updateTransforms(graph))
        #expect(scene.revision == 1)
        try scene.updateGeometry(for: [emptyID], in: graph)
        #expect(scene.revision == 2)
        // Filling the set adds instances, which needs a new scene.
        try empty.replaceAll(transforms: [matrix_identity_float4x4])
        #expect(throws: PathTracingScene.Failure.structureChanged(emptyID)) { try scene.updateTransforms(graph) }

        try empty.replaceAll(transforms: [])
        graph.roots[1] = SceneNode(mesh: box, instances: empty)
        #expect(try scene.updateTransforms(graph) == false)
    }

    @Test("A pre-enqueued trace uses the completed transform update", arguments: modes)
    func refitWithEnqueuedTrace(acceleration: PathTracingScene.Options.Acceleration) throws {
        let glow = try renderer(.plane(width: 10, height: 8), try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1), emissiveFactor: SIMD3(1, 0.5, 0.25),
            alphaMode: .opaque
        ))
        let blocker = try renderer(.cube(), try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1), alphaMode: .opaque
        ))
        var graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -5)), mesh: glow),
            SceneNode(transform: .init(translation: SIMD3(-1, 0, -3)), mesh: blocker),
        ], camera: fixture.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: PathTracer.Lighting(environment: .zero))
        func trace(_ buffer: any MTLCommandBuffer) throws -> [SIMD4<Float>] {
            let accumulation = try tracer.makeAccumulation(width: 48, height: 32)
            let target = try fixture.output(width: 48, height: 32)
            _ = try tracer.encode(
                camera: fixture.camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
                commandBuffer: buffer, samplesPerPixel: 4, settings: .init(maxBounces: 1, seed: 5)
            )
            try fixture.finish(buffer)
            return try fixture.readback(target)
        }
        let before = try trace(#require(fixture.queue.makeCommandBuffer()))
        let nextTrace = try #require(fixture.queue.makeCommandBuffer())
        nextTrace.enqueue()
        // The update must finish without waiting for the caller to commit its trace.
        graph.roots[1].transform.translation.x = 1
        #expect(try scene.updateTransforms(graph))
        let after = try trace(nextTrace)
        let left = 16 * 48 + 12
        let right = 16 * 48 + 36
        #expect(before[left] == SIMD4(0, 0, 0, 1) && before[right] == SIMD4(1, 0.5, 0.25, 1))
        #expect(after[left] == SIMD4(1, 0.5, 0.25, 1) && after[right] == SIMD4(0, 0, 0, 1))
    }

    @Test("Transform updates release the instance buffers they replace", arguments: modes)
    func updatesReleaseMemory(acceleration: PathTracingScene.Options.Acceleration) throws {
        let mesh = try renderer(.cube(), try PhysicallyBasedMaterial(alphaMode: .opaque))
        var graph = SceneGraph(roots: [SceneNode(mesh: mesh)], camera: fixture.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        for step in 1...3 {
            weak let replacedBuffer = scene.buffers.first { $0.label == "Frostfield path instances" }
            try #require(replacedBuffer != nil)
            graph.roots[0].transform.translation.x = Float(step)
            #expect(try scene.updateTransforms(graph))
            #expect(replacedBuffer == nil)
        }
    }

    @Test("Switching between a plain mesh and a singleton instance set throws without changing the scene", arguments: modes)
    func singletonInstanceChange(acceleration: PathTracingScene.Options.Acceleration) throws {
        let mesh = try renderer(.cube(), try PhysicallyBasedMaterial(alphaMode: .opaque), instancing: true)
        let instanceSet = try InstanceSet(device: fixture.device, transforms: [matrix_identity_float4x4])
        let id = UUID()
        for instances in [nil, instanceSet] {
            var graph = SceneGraph(roots: [SceneNode(id: id, mesh: mesh, instances: instances)], camera: fixture.camera)
            let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
            graph.roots[0].instances = instances == nil ? instanceSet : nil
            #expect(throws: PathTracingScene.Failure.structureChanged(id)) { try scene.updateTransforms(graph) }
            #expect(scene.revision == 0)
            #expect(scene.statistics.instanceCount == 1)
        }
    }

    @Test("The software BVH keeps all primitives when their centroid extent is subnormal")
    func subnormalCentroidExtent() throws {
        let vertices = (0..<5).flatMap { index in
            let x = Float(index) * Float.leastNonzeroMagnitude * 8
            return [SIMD3<Float>(x, 0, 0), SIMD3<Float>(x, 1, 0), SIMD3<Float>(x, 0, 1)]
        }
        let tree = try PathTracingScene.buildBVH(vertices: vertices, indices: (0..<15).map(UInt32.init))
        #expect(tree.primitives.sorted() == [0, 1, 2, 3, 4])
        #expect(tree.nodes.count == 3)
    }

    // MARK: - Scale

    @Test(
        "Metal ray tracing holds over a million instances and refits them into a new structure",
        .enabled(if: MTLCreateSystemDefaultDevice()?.supportsRaytracing == true, "Requires Metal ray tracing")
    )
    func millionInstances() throws {
        let box = try renderer(.cube(), try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, emissiveFactor: SIMD3(1, 0.5, 0.25), alphaMode: .opaque
        ), instancing: true)
        // Four sets of 512 × 512 cubes, 2 cm apart, a 10.24 m square each.
        let side = 512
        let transforms = (0..<(side * side)).map { index -> simd_float4x4 in
            SceneTransform(
                translation: SIMD3(Float(index % side) * 0.02 - 5.12, Float(index / side) * 0.02 - 5.12, 0),
                scale: SIMD3(repeating: 0.012)
            ).matrix
        }
        var graph = SceneGraph(roots: try (0..<4).map { quadrant in
            SceneNode(
                transform: .init(translation: SIMD3(quadrant % 2 == 0 ? -5.12 : 5.12, quadrant < 2 ? -5.12 : 5.12, -12)),
                mesh: box, instances: try InstanceSet(device: fixture.device, transforms: transforms)
            )
        }, camera: fixture.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: .metal))
        let statistics = scene.statistics
        #expect(statistics.instanceCount == 1_048_576 && statistics.bottomLevelCount == 1)
        #expect(statistics.instancedTriangleCount == 12 * 1_048_576)
        #expect(statistics.accelerationStructureByteCount > 1_048_576 * 32)
        let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: PathTracer.Lighting(environment: .zero))
        let settings = PathTracer.Settings(maxBounces: 1, seed: 1)
        let before = try fixture.trace(tracer, width: 64, height: 48, samples: 1, settings: settings)
        // About 60 percent of the field is cube faces: each cell is 1.2 cm of 2 cm.
        let lit = before.count { $0.x > 0.5 }
        #expect(lit > before.count / 5 && lit < before.count * 4 / 5, "\(lit) lit pixels")

        graph.roots[0].transform.translation.x += 0.01
        #expect(try scene.updateTransforms(graph))
        #expect(scene.statistics.topLevelRefitCount == 1 && scene.statistics.topLevelBuildCount == 1)
        let fresh = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: .metal))
        let moved = try fixture.trace(tracer, width: 64, height: 48, samples: 1, settings: settings)
        let expected = try fixture.trace(
            try PathTracer(queue: fixture.queue, scene: fresh, lighting: PathTracer.Lighting(environment: .zero)),
            width: 64, height: 48, samples: 1, settings: settings
        )
        #expect(PathTestFixture.maximumDifference(moved, expected) < 1e-3)
        #expect(PathTestFixture.maximumDifference(moved, before) > 0.5)
    }

    @Test("A mesh four times the former triangle limit builds and traces with both accelerations", arguments: modes)
    func millionTriangles(acceleration: PathTracingScene.Options.Acceleration) throws {
        // A 1,024 × 512 grid of quads: 1,048,576 triangles facing the camera.
        let columns = 1_024
        let rows = 512
        var vertices: [MeshVertex] = []
        vertices.reserveCapacity((columns + 1) * (rows + 1))
        for row in 0...rows {
            for column in 0...columns {
                let u = Float(column) / Float(columns)
                let v = Float(row) / Float(rows)
                vertices.append(MeshVertex(
                    position: SIMD3(u * 4 - 2, 1 - v * 2, 0.05 * sin(u * 40) * cos(v * 30)), normal: SIMD3(0, 0, 1),
                    textureCoordinate: SIMD2(u, v)
                ))
            }
        }
        var indices: [UInt32] = []
        indices.reserveCapacity(columns * rows * 6)
        for row in 0..<rows {
            for column in 0..<columns {
                let top = UInt32(row * (columns + 1) + column)
                let bottom = top + UInt32(columns + 1)
                indices += [top, bottom, top + 1, top + 1, bottom, bottom + 1]
            }
        }
        let mesh = try renderer(try MeshGeometry(vertices: vertices, indices: indices), try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, emissiveFactor: SIMD3(1, 1, 1), alphaMode: .opaque
        ))
        let graph = SceneGraph(roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)], camera: fixture.camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, options: .init(acceleration: acceleration))
        #expect(scene.triangleCount == 4 * 262_144)
        let tracer = try PathTracer(queue: fixture.queue, scene: scene, lighting: PathTracer.Lighting(environment: .zero))
        let image = try fixture.trace(tracer, width: 48, height: 32, samples: 1, settings: .init(maxBounces: 1, seed: 2))
        // The 4 × 2 m sheet at 3 m covers the middle of the 60° view and nothing else.
        #expect(image[16 * 48 + 24].x == 1 && image[0].x == 0)
    }

    // MARK: - Asynchronous preparation

    @MainActor
    @Test("Asynchronous preparation builds off the main actor, reports progress and matches the synchronous scene", arguments: modes)
    func asynchronousPreparation(acceleration: PathTracingScene.Options.Acceleration) async throws {
        let graph = try studio()
        let options = PathTracingScene.Options(acceleration: acceleration)
        let reports = Mutex<[(PathTracingScene.PreparationProgress, Bool)]>([])
        let scene = try await PathTracingScene.prepare(queue: fixture.queue, graph: graph, options: options) { progress in
            let main = Thread.isMainThread
            reports.withLock { $0.append((progress, main)) }
        }
        let recorded = reports.withLock { $0 }
        #expect(!recorded.isEmpty && recorded.allSatisfy { !$0.1 })
        #expect(recorded.last?.0.fractionCompleted == 1 && recorded.last?.0.completedLevelCount == scene.statistics.bottomLevelCount)
        #expect(zip(recorded, recorded.dropFirst()).allSatisfy { $0.0.completedTriangleCount <= $1.0.completedTriangleCount })
        #expect(recorded.last?.0.triangleCount == scene.triangleCount)

        let synchronous = try PathTracingScene(queue: fixture.queue, graph: graph, options: options)
        let lighting = try studioLighting()
        let settings = PathTracer.Settings(maxBounces: 4, seed: 8)
        let first = try fixture.trace(
            try PathTracer(queue: fixture.queue, scene: scene, lighting: lighting), width: 48, height: 32, samples: 4, settings: settings
        )
        let second = try fixture.trace(
            try PathTracer(queue: fixture.queue, scene: synchronous, lighting: lighting), width: 48, height: 32, samples: 4,
            settings: settings
        )
        #expect(first == second)
    }

    @Test("Cancelling an asynchronous preparation throws CancellationError")
    func cancelledPreparation() async throws {
        let queue = fixture.queue
        let device = fixture.device
        let task = Task { () async throws -> Int in
            withUnsafeCurrentTask { $0?.cancel() }
            let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 30)
            let mesh = try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: .cube(),
                material: .physicallyBased(try PhysicallyBasedMaterial(alphaMode: .opaque))
            )
            let graph = SceneGraph(roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)], camera: camera)
            return try await PathTracingScene.prepare(queue: queue, graph: graph).triangleCount
        }
        await #expect(throws: CancellationError.self) { try await task.value }
    }
}
