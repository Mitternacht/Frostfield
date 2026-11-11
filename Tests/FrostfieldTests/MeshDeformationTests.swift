import Foundation
import Frostfield
import Metal
import simd
import Synchronization
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MeshDeformationTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let camera: PerspectiveCamera

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20)
    }

    private func renderer(
        _ geometry: MeshGeometry, graph: SurfaceShaderGraph, uniforms: ShaderUniforms? = nil,
        lightIntensity: Float = 0, receivesShadows: Bool = false
    ) throws -> MeshRenderer {
        try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .physicallyBased(try PhysicallyBasedMaterial(
                directionToLight: SIMD3(1, 0, 1), lightIntensity: lightIntensity, ambientIntensity: 0,
                receivesShadows: receivesShadows, alphaMode: .opaque, surfaceGraph: graph, surfaceUniforms: uniforms
            ))
        )
    }

    @Test("Offsets move picking and normals follow the deformed triangles")
    func tiltedPlane() throws {
        // z += 0.5·x tilts the plane; its normal becomes (−0.5, 0, 1) normalized.
        let graph = try SurfaceShaderGraph { s in
            .init(positionOffset: s.vector(s.constant(Float(0)), s.constant(Float(0)), s.objectPosition.x * 0.5))
        }
        let mesh = try renderer(try .plane(width: 2, height: 2, columns: 4, rows: 4), graph: graph)
        let scene = SceneGraph(roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)], camera: camera)
        // Distances start at the 0.1 near plane. Right of center the tilted plane is about
        // 2.75 away; without the offset that point would be 2.92, farther than the center.
        let center = try #require(try scene.pick(at: SIMD2(0.5, 0.5), aspectRatio: 1))
        #expect(abs(center.intersection.distance - 2.9) < 1e-3)
        let hit = try #require(try scene.pick(at: SIMD2(0.6, 0.5), aspectRatio: 1))
        #expect(abs(hit.intersection.distance - 2.754) < 0.01, "\(hit.intersection.distance)")

        let deformed = try PathTracingScene(queue: queue, graph: scene)
        #expect(deformed.triangleCount == 32)
        let expected = simd_normalize(SIMD3<Float>(-0.5, 0, 1))
        let normals = try MeshDeformationTests.deformedNormals(mesh, queue: queue)
        for normal in normals {
            #expect(simd_distance(normal, expected) < 1e-4, "\(normal)")
        }
    }

    @Test("Zero offsets preserve hard edges and seams stay welded")
    func adjacency() throws {
        let still = try SurfaceShaderGraph { s in .init(positionOffset: s.constant(SIMD3<Float>(0, 0, 0))) }
        let cube = MeshGeometry.cube()
        let cubeNormals = try MeshDeformationTests.deformedNormals(try renderer(cube, graph: still), queue: queue)
        for (normal, vertex) in zip(cubeNormals, cube.vertices) {
            #expect(simd_distance(normal, vertex.normal) < 1e-5)
        }

        // Radial bumps on a sphere: vertices duplicated along the UV seam keep identical normals.
        let bumps = try SurfaceShaderGraph { s in
            .init(positionOffset: s.displacement((s.objectPosition.y * 9).sin() * 0.05))
        }
        let sphere = try MeshGeometry.uvSphere(radius: 1, segments: 24, rings: 12)
        let sphereNormals = try MeshDeformationTests.deformedNormals(try renderer(sphere, graph: bumps), queue: queue)
        var checked = 0
        for i in sphere.vertices.indices {
            for j in sphere.vertices.indices where j > i
                && simd_distance(sphere.vertices[i].position, sphere.vertices[j].position) < 1e-6
                && simd_dot(sphere.vertices[i].normal, sphere.vertices[j].normal) > 0.999 {
                #expect(simd_distance(sphereNormals[i], sphereNormals[j]) < 1e-5)
                checked += 1
            }
        }
        #expect(checked >= 11)
    }

    @Test("Uniform edits redeform raster draws, shadows, and path-tracing snapshots")
    func liveDeformation() throws {
        let graph = try SurfaceShaderGraph { s in
            let lift = s.uniform("Lift", Float(0))
            return .init(emission: s.constant(SIMD3<Float>(1, 1, 1)), positionOffset: s.vector(lift, s.constant(Float(0)), s.constant(Float(0))))
        }
        let uniforms = try ShaderUniforms(graph: graph)
        let mesh = try renderer(try .plane(width: 1, height: 1), graph: graph, uniforms: uniforms)
        let scene = SceneGraph(roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)], camera: camera)
        let target = try RenderTarget(device: device, size: 33)

        try target.render(scene, queue: queue)
        #expect(try target.pixel(16, 16).x > 0.99)
        try uniforms.set("Lift", to: 5)
        try target.render(scene, queue: queue)
        #expect(try target.pixel(16, 16).x == 0, "The edited offset moved the plane out of view.")

        let traced = try PathTracingScene(queue: queue, graph: scene)
        let tracer = try PathTracer(queue: queue, scene: traced)
        #expect(try target.trace(tracer, camera: camera, queue: queue).x == 0)
        try uniforms.set("Lift", to: 0)
        let restored = try PathTracer(queue: queue, scene: PathTracingScene(queue: queue, graph: scene))
        #expect(try target.trace(restored, camera: camera, queue: queue).x > 0.99)

        // The shadow pass reads the same deformed vertices as the color pass.
        let receiver = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 3, height: 3),
            material: .physicallyBased(try .init(roughness: 0.8, ambientIntensity: 0, receivesShadows: true, alphaMode: .opaque))
        )
        let blockerGraph = try SurfaceShaderGraph { s in
            .init(positionOffset: s.vector(s.uniform("Shift", Float(0)), s.constant(Float(0)), s.constant(Float(0))))
        }
        let blockerUniforms = try ShaderUniforms(graph: blockerGraph)
        let blocker = try renderer(.cube(), graph: blockerGraph, uniforms: blockerUniforms)
        let shadow = try DirectionalShadowMap(
            queue: queue, resolution: 128, directionToLight: SIMD3(1, 0, 1), center: SIMD3(0, 0, -3), halfExtent: 2, intensity: 2
        )
        let shadowScene = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: receiver, castsShadow: false),
            SceneNode(transform: .init(translation: SIMD3(0.65, 0, -2.35), scale: SIMD3(repeating: 0.4)), mesh: blocker),
        ], camera: camera)
        try target.render(shadowScene, queue: queue, shadowMap: shadow)
        let shadowed = try target.pixel(16, 16).x
        try blockerUniforms.set("Shift", to: 20)
        try target.render(shadowScene, queue: queue, shadowMap: shadow)
        let lit = try target.pixel(16, 16).x
        #expect(lit > 0.2 && shadowed < lit * 0.25, "lit \(lit), shadowed \(shadowed)")
    }

    @MainActor
    @Test("A graph BVH prewarms the active raster pose and live picks refresh when uniform or clock inputs change")
    func prewarmedGraphPicking() async throws {
        let shader = try SurfaceShaderGraph { s in
            let shift = s.uniform("Shift", Float(0)) + s.time
            return .init(
                emission: s.constant(SIMD3<Float>(1, 1, 1)),
                positionOffset: s.vector(shift, s.constant(Float(0)), s.objectPosition.x * 0.15)
            )
        }
        let uniforms = try ShaderUniforms(graph: shader)
        let plane = try MeshGeometry.plane(width: 0.6, height: 0.6)
        let geometry = try MeshGeometry.merged([
            .init(plane, transform: translationMatrix(-0.55)),
            .init(plane, transform: translationMatrix(0.55))
        ]).geometry
        let mesh = try renderer(geometry, graph: shader, uniforms: uniforms)
        mesh.drawRange = try MeshDrawRange(start: 6, count: 6)
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)
        let scene = SceneGraph(roots: [node], camera: camera)
        let target = try RenderTarget(device: device, size: 33)
        try target.render(scene, queue: queue)
        #expect(try target.pixel(22, 16).x > 0.9)
        #expect(try target.pixel(11, 16).x == 0)

        let reports = Mutex<[MeshBVH.BuildProgress]>([])
        let initial = try await mesh.preparePickingAcceleration { report in reports.withLock { $0.append(report) } }
        let prepared = try #require(initial)
        #expect(prepared.geometry.indices == Array(geometry.indices[6..<12]))
        #expect(reports.withLock { $0.last?.fractionCompleted } == 1)
        let fresh = try mesh.deformedGeometry(queue: queue)
        #expect(prepared.geometry.vertices.map(\.position) == fresh.vertices.map(\.position))
        #expect(prepared.geometry.vertices.map(\.normal) == fresh.vertices.map(\.normal))

        func ray(_ x: Float) throws -> SceneRay {
            try SceneRay(origin: SIMD3(x, 0.07, 0), direction: SIMD3(0, 0, -1))
        }
        let initialRay = try ray(0.55)
        let leftRay = try ray(-0.55)
        let shiftedRay = try ray(1.35)
        let frameRay = try ray(2.15)
        #expect(try scene.pick(ray: leftRay) == nil)
        let hit = try #require(try scene.pick(ray: initialRay))
        #expect(hit.nodeID == node.id && (2...3).contains(hit.intersection.triangleIndex))
        #expect(try scene.intersections(ray: initialRay) == [hit])
        #expect(try scene.pick(ray: shiftedRay) == nil)

        try uniforms.set("Shift", to: 0.8)
        let pending = try #require(queue.makeCommandBuffer())
        try scene.encode(pass: target.pass, commandBuffer: pending)
        // The queued pose isn't available yet. Picking evaluates the same inputs
        // without waiting for an unrelated command that has never been submitted.
        #expect(try scene.pick(ray: initialRay) == nil)
        let shifted = try #require(try scene.pick(ray: shiftedRay))
        #expect(shifted.intersection.triangleIndex == hit.intersection.triangleIndex)
        #expect(abs(shifted.intersection.distance - hit.intersection.distance) < 1e-5)
        let repeated = Mutex(0)
        let warmedByPick = try await mesh.preparePickingAcceleration { _ in repeated.withLock { $0 += 1 } }
        #expect(abs((try #require(warmedByPick)).bounds.center.x - 1.35) < 1e-5)
        #expect(repeated.withLock { $0 } == 0)
        let completed: Bool = await withCheckedContinuation { completion in
            pending.addCompletedHandler { completion.resume(returning: $0.status == .completed) }
            pending.commit()
        }
        try #require(completed)
        #expect(try target.pixel(22, 16).x == 0)
        #expect(try target.pixel(30, 16).x > 0.9)
        #expect(try prepared.intersection(ray: initialRay, localToWorld: node.transform.matrix) != nil)
        #expect(try prepared.intersection(ray: shiftedRay, localToWorld: node.transform.matrix) == nil)

        try uniforms.setFrame(ShaderFrame(time: 0.8, index: 1))
        #expect(try scene.intersections(ray: shiftedRay).isEmpty)
        #expect(try scene.intersections(ray: frameRay).first?.nodeID == node.id)

        mesh.drawRange = try MeshDrawRange(start: 0, count: 6)
        let otherRange = try await mesh.preparePickingAcceleration()
        #expect(otherRange?.geometry.indices == Array(geometry.indices[0..<6]))
        let newHit = try #require(try scene.pick(ray: ray(1.05)))
        #expect((0...1).contains(newHit.intersection.triangleIndex))
        #expect(try scene.pick(ray: frameRay) == nil)

        mesh.drawRange = try MeshDrawRange(start: 0, count: 0)
        #expect(try await mesh.preparePickingAcceleration() == nil)
        #expect(try scene.pick(ray: frameRay) == nil)
    }

    @MainActor
    @Test("A live graph edit invalidates an asynchronous picking build", arguments: [false, true])
    func rejectsStaleGraphPicking(clockEdit: Bool) async throws {
        let shader = try SurfaceShaderGraph { s in
            .init(positionOffset: s.vector(s.uniform("Shift", Float(0)) + s.time, s.constant(Float(0)), s.constant(Float(0))))
        }
        let uniforms = try ShaderUniforms(graph: shader)
        let mesh = try renderer(try .plane(width: 0.6, height: 0.6), graph: shader, uniforms: uniforms)
        let (ready, signal) = AsyncStream<Bool>.makeStream()
        let release = DispatchSemaphore(value: 0)
        let build = Task {
            defer { signal.finish() }
            return try await mesh.preparePickingAcceleration { progress in
                if progress.fractionCompleted == 1 {
                    signal.yield(true)
                    _ = release.wait(timeout: .now() + 10)
                }
            }
        }
        defer { release.signal() }
        var iterator = ready.makeAsyncIterator()
        try #require(await iterator.next() == true)
        if clockEdit {
            try uniforms.setFrame(ShaderFrame(time: 1, index: 1))
        } else {
            try uniforms.set("Shift", to: 1)
        }
        release.signal()
        await #expect(throws: MeshRenderer.Failure.pickingGeometryChanged) { try await build.value }

        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)
        let scene = SceneGraph(roots: [node], camera: camera)
        let old = try SceneRay(origin: .zero, direction: SIMD3(0, 0, -1))
        let moved = try SceneRay(origin: SIMD3(1, 0.07, 0), direction: SIMD3(0, 0, -1))
        #expect(try scene.pick(ray: old) == nil)
        #expect(try scene.pick(ray: moved)?.nodeID == node.id)
        let current = try await mesh.preparePickingAcceleration()
        #expect(current?.bounds.center.x == 1)
    }

    @MainActor
    @Test("Texture-driven graph picking refreshes current texels and rejects prewarming")
    func textureDrivenGraphPicking() async throws {
        let shader = try SurfaceShaderGraph { s in
            let texel = s.texture("Shift", content: .data, filtering: .nearest).sample(s.uv0)
            return .init(positionOffset: s.vector(texel.x, s.constant(Float(0)), s.constant(Float(0))))
        }
        let description = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        description.storageMode = .shared
        description.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: description))
        func write(_ red: UInt8) {
            let pixel: [UInt8] = [red, 0, 0, 255]
            texture.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: pixel, bytesPerRow: 4)
        }
        write(0)
        let mesh = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.6, height: 0.6),
            material: .physicallyBased(try PhysicallyBasedMaterial(surfaceGraph: shader, surfaceTextures: ["Shift": texture]))
        )
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)
        let scene = SceneGraph(roots: [node], camera: camera)
        let old = try SceneRay(origin: .zero, direction: SIMD3(0, 0, -1))
        let moved = try SceneRay(origin: SIMD3(1, 0.07, 0), direction: SIMD3(0, 0, -1))
        await #expect(throws: MeshRenderer.Failure.pickingPreparation) { try await mesh.preparePickingAcceleration() }
        #expect(try scene.pick(ray: old)?.nodeID == node.id)
        #expect(try scene.pick(ray: moved) == nil)
        write(255)
        #expect(try scene.pick(ray: old) == nil)
        #expect(try scene.pick(ray: moved)?.nodeID == node.id)
    }

    private static func deformedNormals(_ mesh: MeshRenderer, queue: any MTLCommandQueue) throws -> [SIMD3<Float>] {
        try mesh.deformedGeometry(queue: queue).vertices.map(\.normal)
    }

    private struct RenderTarget {
        let color: any MTLTexture
        let pass: MTLRenderPassDescriptor
        let size: Int

        init(device: any MTLDevice, size: Int) throws {
            self.size = size
            let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float, width: size, height: size, mipmapped: false
            )
            colorDescriptor.storageMode = .shared
            colorDescriptor.usage = [.renderTarget, .shaderWrite]
            color = try #require(device.makeTexture(descriptor: colorDescriptor))
            let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .depth32Float, width: size, height: size, mipmapped: false
            )
            depthDescriptor.storageMode = .private
            depthDescriptor.usage = .renderTarget
            pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = color
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = try #require(device.makeTexture(descriptor: depthDescriptor))
            pass.depthAttachment.clearDepth = 1
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.storeAction = .dontCare
        }

        func render(_ scene: SceneGraph, queue: any MTLCommandQueue, shadowMap: DirectionalShadowMap? = nil) throws {
            let buffer = try #require(queue.makeCommandBuffer())
            try scene.encode(pass: pass, commandBuffer: buffer, shadowMap: shadowMap)
            buffer.commit()
            buffer.waitUntilCompleted()
            try #require(buffer.status == .completed)
        }

        func trace(_ tracer: PathTracer, camera: PerspectiveCamera, queue: any MTLCommandQueue) throws -> SIMD4<Float> {
            let accumulation = try tracer.makeAccumulation(width: size, height: size)
            let buffer = try #require(queue.makeCommandBuffer())
            _ = try tracer.encode(camera: camera, cameraTransform: .init(), destination: color,
                                  accumulation: accumulation, commandBuffer: buffer, samplesPerPixel: 2)
            buffer.commit()
            buffer.waitUntilCompleted()
            try #require(buffer.status == .completed)
            return try pixel(size / 2, size / 2)
        }

        func pixel(_ x: Int, _ y: Int) throws -> SIMD4<Float> {
            var value = SIMD4<Float16>.zero
            try withUnsafeMutableBytes(of: &value) { bytes in
                color.getBytes(try #require(bytes.baseAddress), bytesPerRow: 8,
                               from: MTLRegionMake2D(x, y, 1, 1), mipmapLevel: 0)
            }
            return SIMD4<Float>(value)
        }
    }
}
