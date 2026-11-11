import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathEmissiveSamplingTests {
    private let fixture: PathTestFixture

    init() throws {
        fixture = try PathTestFixture()
    }

    @Test("Mirrored, nonuniform emissive instances match analytic panels and ordinary path hits",
          arguments: PathAccelerationTests.modes)
    func matchesIndependentEstimators(acceleration: PathTracingScene.Options.Acceleration) throws {
        let receiver = try self.receiver()
        let color = SIMD3<Float>(1, 0.625, 0.25)
        let emitter = try self.emitter(.init(
            baseColorFactor: SIMD4(0, 0, 0, 1), emissiveFactor: color, alphaMode: .opaque, emissiveStrength: 4
        ))
        let placements: [(position: SIMD3<Float>, size: SIMD2<Float>)] = [
            (SIMD3(0.59, 0, -2), SIMD2(0.96, 1.3)),
            (SIMD3(-0.64, 0.2, -2.25), SIMD2(0.47, 0.82))
        ]
        let transforms = placements.map { placement in
            facingReceiver(position: placement.position, size: placement.size).matrix
        }
        let lights = try placements.map { placement in
            try PathTracer.AreaLight(
                center: placement.position, halfU: SIMD3(placement.size.x / 2, 0, 0),
                halfV: SIMD3(0, -placement.size.y / 2, 0), radiance: color * 4, cameraVisible: false
            )
        }
        let instances = try InstanceSet(device: fixture.device, transforms: transforms)
        let emitters = SceneNode(mesh: emitter, instances: instances)
        let enabled = try prepared([receiver, emitters], acceleration: acceleration)
        let disabled = try prepared([receiver, emitters], acceleration: acceleration, sampled: false)
        let reference = try prepared([receiver], acceleration: acceleration)
        #expect(enabled.sampledEmissiveTriangleCount == 4)
        #expect(disabled.sampledEmissiveTriangleCount == 0)
        #expect(reference.sampledEmissiveTriangleCount == 0)

        let enabledTracer = try PathTracer(queue: fixture.queue, scene: enabled)
        let disabledTracer = try PathTracer(queue: fixture.queue, scene: disabled)
        let oneSample = try trace(enabledTracer, samples: 1)
        #expect(oneSample.x > 0.1 && oneSample.w == 1, "\(oneSample)")
        let meshMIS = try trace(enabledTracer, samples: 2048)
        let panelMIS = try trace(PathTracer(queue: fixture.queue, scene: reference, lights: lights), samples: 2048)
        let bsdf = try trace(disabledTracer, samples: 4096)
        let tableDisabled = try trace(enabledTracer, samples: 256, multipleImportanceSampling: false)
        let noTable = try trace(disabledTracer, samples: 256, multipleImportanceSampling: false)
        #expect(tableDisabled == noTable)
        #expect(abs(meshMIS.x / panelMIS.x - 1) < 0.025, "mesh \(meshMIS), panel \(panelMIS)")
        #expect(abs(meshMIS.x / bsdf.x - 1) < 0.07, "mesh \(meshMIS), BSDF \(bsdf)")
        #expect(abs(meshMIS.y / meshMIS.x - 0.625) < 0.002)
        #expect(abs(meshMIS.z / meshMIS.x - 0.25) < 0.002)
    }

    @Test("Transform, geometry and material updates replace immutable emitter distributions",
          arguments: PathAccelerationTests.modes)
    func replacesEmitterRevisions(acceleration: PathTracingScene.Options.Acceleration) throws {
        let material = try PhysicallyBasedMaterial(
            baseColorFactor: SIMD4(0, 0, 0, 1), emissiveFactor: SIMD3(1, 0.5, 0.25), alphaMode: .opaque, emissiveStrength: 4
        ).withLiveFactors()
        let live = try #require(material.liveFactors)
        let mesh = try emitter(material)
        let firstTransform = facingReceiver(position: SIMD3(0.75, 0, -2), size: SIMD2(0.8, 0.8)).matrix
        let secondTransform = facingReceiver(position: SIMD3(-0.75, 0, -2), size: SIMD2(0.8, 0.8)).matrix
        let set = try InstanceSet(device: fixture.device, transforms: [firstTransform, secondTransform])
        let node = SceneNode(mesh: mesh, instances: set)
        let nodes = [try receiver(), node]
        let graph = SceneGraph(roots: nodes, camera: fixture.camera)
        let scene = try prepared(nodes, acceleration: acceleration)
        let tracer = try PathTracer(queue: fixture.queue, scene: scene)
        #expect(scene.sampledEmissiveTriangleCount == 4)
        let original = try trace(tracer, samples: 256)

        let target = try fixture.output(width: 1, height: 1)
        let accumulation = try tracer.makeAccumulation(width: 1, height: 1)
        let oldCommand = try #require(fixture.queue.makeCommandBuffer())
        let oldFrame = try encode(tracer, target: target, accumulation: accumulation, command: oldCommand, samples: 16)
        // This encode has already bound the old scene resources when they are replaced.
        try set.setTransform(facingReceiver(position: SIMD3(-0.75, 0, -2), size: SIMD2(0.28, 0.5)).matrix, at: 1)
        #expect(try scene.updateTransforms(graph))
        #expect(scene.sampledEmissiveTriangleCount == 4)
        try fixture.finish(oldCommand)
        let oldImage = try #require(fixture.readback(target).first)
        let oldGraphSet = try InstanceSet(device: fixture.device, transforms: [firstTransform, secondTransform])
        let oldScene = try prepared([nodes[0], SceneNode(mesh: mesh, instances: oldGraphSet)], acceleration: acceleration)
        #expect(try trace(PathTracer(queue: fixture.queue, scene: oldScene), samples: 16) == oldImage)
        let newCommand = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: PathTracer.Failure.sceneChanged) {
            try encode(tracer, target: target, accumulation: accumulation, previous: oldFrame, command: newCommand)
        }
        let moved = try trace(tracer, samples: 256)
        #expect(moved.x < original.x * 0.7, "\(moved) vs \(original)")
        let fresh = try prepared(nodes, acceleration: acceleration)
        #expect(try trace(PathTracer(queue: fixture.queue, scene: fresh), samples: 256) == moved)

        let plane = try MeshGeometry.plane(width: 1, height: 1)
        let triangle = try MeshGeometry(vertices: plane.vertices, indices: Array(plane.indices.prefix(3)))
        try scene.updateGeometry(for: [node.id], in: graph, replacements: [node.id: triangle])
        #expect(scene.sampledEmissiveTriangleCount == 2)
        let rebuilt = try trace(tracer, samples: 256)
        #expect(rebuilt.x > 0 && rebuilt.x < moved.x * 0.8, "\(rebuilt) vs \(moved)")

        try live.update { $0.emissiveStrength = 0 }
        #expect(try scene.updateMaterials())
        #expect(scene.sampledEmissiveTriangleCount == 0)
        #expect(try trace(tracer, samples: 16) == SIMD4(0, 0, 0, 1))
        try live.update { $0.emissiveStrength = 2 }
        #expect(try scene.updateMaterials())
        #expect(scene.sampledEmissiveTriangleCount == 2)
        let dimmed = try trace(tracer, samples: 256)
        #expect(abs(dimmed.x / rebuilt.x - 0.5) < 0.002)
    }

    @Test("UV1 emission maps light the receiver like corresponding colored panels",
          arguments: PathAccelerationTests.modes, [false, true])
    func texturedEmitters(acceleration: PathTracingScene.Options.Acceleration, baked: Bool) throws {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 2, height: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let map = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let pixels: [UInt8] = [255, 0, 0, 255, 0, 0, 255, 255]
        pixels.withUnsafeBytes {
            map.replace(region: MTLRegionMake2D(0, 0, 2, 1), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 8)
        }
        let plane = try MeshGeometry.plane()
        let uv1 = try MeshGeometry(vertices: plane.vertices.map { vertex in
            MeshVertex(
                position: vertex.position, normal: vertex.normal, textureCoordinate: SIMD2(0.25, 0.5),
                secondaryTextureCoordinate: vertex.textureCoordinate
            )
        }, indices: plane.indices)
        let binding = try PhysicallyBasedMaterial.TextureBinding(
            texture: map, textureCoordinateSet: .secondary,
            transform: .init(offset: SIMD2(1, 0), scale: SIMD2(-1, 1)),
            sampling: .init(addressU: .clampToEdge, addressV: .clampToEdge,
                            magnification: .nearest, minification: .nearest, mipFilter: .notMipmapped)
        )
        let lightMesh = try emitter(.init(
            baseColorFactor: SIMD4(0, 0, 0, 1), emissiveTexture: binding,
            emissiveFactor: SIMD3(repeating: 1), alphaMode: .opaque, emissiveStrength: 4
        ), geometry: uv1)
        let receiver = try self.receiver()
        let meshLight = SceneNode(
            transform: facingReceiver(position: SIMD3(0.65, 0, -2), size: SIMD2(0.8, 0.8)), mesh: lightMesh
        )
        let textures: PathTracingScene.Options.Textures = baked ? .baked(size: 16) : .fullResolution(fallbackSize: 16)
        let enabled = try prepared([receiver, meshLight], acceleration: acceleration, textures: textures)
        let disabled = try prepared([receiver, meshLight], acceleration: acceleration, sampled: false, textures: textures)
        #expect(enabled.sampledEmissiveTriangleCount == 2)
        #expect(disabled.sampledEmissiveTriangleCount == 0)
        let explicit = try PathTracer(queue: fixture.queue, scene: enabled)
        let bsdfOnly = try PathTracer(queue: fixture.queue, scene: disabled)
        let reference = try prepared([receiver], acceleration: acceleration)
        let lights = try [
            PathTracer.AreaLight(center: SIMD3(0.45, 0, -2), halfU: SIMD3(0.2, 0, 0), halfV: SIMD3(0, -0.4, 0),
                                 radiance: SIMD3(0, 0, 4), cameraVisible: false),
            PathTracer.AreaLight(center: SIMD3(0.85, 0, -2), halfU: SIMD3(0.2, 0, 0), halfV: SIMD3(0, -0.4, 0),
                                 radiance: SIMD3(4, 0, 0), cameraVisible: false),
        ]
        let meshMIS = try trace(explicit, samples: 2048)
        let panelMIS = try trace(PathTracer(queue: fixture.queue, scene: reference, lights: lights), samples: 2048)
        let bsdf = try trace(bsdfOnly, samples: 4096)
        #expect(meshMIS.x > 0.06 && meshMIS.z > 0.06 && meshMIS.y == 0)
        #expect(abs(meshMIS.x / panelMIS.x - 1) < 0.045 && abs(meshMIS.z / panelMIS.z - 1) < 0.045,
                "mesh \(meshMIS), panels \(panelMIS)")
        #expect(abs(meshMIS.x / bsdf.x - 1) < 0.10 && abs(meshMIS.z / bsdf.z - 1) < 0.10,
                "mesh \(meshMIS), BSDF \(bsdf)")
        #expect(try trace(explicit, samples: 32, multipleImportanceSampling: false) == trace(bsdfOnly, samples: 32))

        let cameraEmitter = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: lightMesh)
        let cameraSampled = try prepared([cameraEmitter], acceleration: acceleration, textures: textures)
        let cameraPlain = try prepared([cameraEmitter], acceleration: acceleration, sampled: false, textures: textures)
        let primary = try trace(PathTracer(queue: fixture.queue, scene: cameraSampled), samples: 16)
        #expect(primary == (try trace(PathTracer(queue: fixture.queue, scene: cameraPlain), samples: 16)))
        #expect(abs(primary.x + primary.z - 4) < 0.005 && primary.w == 1)
    }

    @Test("Light and BSDF strategies both sample the outgoing cone's authored emissive mip",
          .enabled(if: MTLCreateSystemDefaultDevice()?.argumentBuffersSupport == .tier2, "Requires direct mip sampling"),
          arguments: PathAccelerationTests.modes)
    func texturedEmitterMipSelection(acceleration: PathTracingScene.Options.Acceleration) throws {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 64, height: 64, mipmapped: true)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let map = try #require(fixture.device.makeTexture(descriptor: descriptor))
        for level in 0..<map.mipmapLevelCount {
            let side = max(1, map.width >> level)
            let color: SIMD4<UInt8> = level == 0 ? SIMD4(255, 0, 0, 255) : SIMD4(0, 255, 0, 255)
            let pixels = Array(repeating: color, count: side * side)
            pixels.withUnsafeBytes {
                map.replace(
                    region: MTLRegionMake2D(0, 0, side, side), mipmapLevel: level,
                    withBytes: $0.baseAddress!, bytesPerRow: side * MemoryLayout<SIMD4<UInt8>>.stride
                )
            }
        }
        let mesh = try emitter(.init(
            baseColorFactor: SIMD4(0, 0, 0, 1), emissiveTexture: .init(
                texture: map, sampling: .init(addressU: .clampToEdge, addressV: .clampToEdge)
            ), emissiveFactor: SIMD3(repeating: 1), alphaMode: .opaque, emissiveStrength: 4
        ))
        let light = SceneNode(
            transform: facingReceiver(position: SIMD3(0.65, 0, -1.8), size: SIMD2(0.8, 0.8)), mesh: mesh
        )
        let nodes = [try receiver(), light]
        let textureMode = PathTracingScene.Options.Textures.fullResolution(fallbackSize: 16)
        let enabled = try prepared(nodes, acceleration: acceleration, textures: textureMode)
        let disabled = try prepared(nodes, acceleration: acceleration, sampled: false, textures: textureMode)
        let mis = try trace(PathTracer(queue: fixture.queue, scene: enabled), samples: 1024)
        let bsdf = try trace(PathTracer(queue: fixture.queue, scene: disabled), samples: 4096)
        #expect(mis.y > 0.12 && mis.x < 0.002 && mis.z == 0, "mip result \(mis)")
        #expect(bsdf.y > 0.12 && bsdf.x < 0.002 && bsdf.z == 0, "BSDF result \(bsdf)")
        #expect(abs(mis.y / bsdf.y - 1) < 0.08, "MIS \(mis), BSDF \(bsdf)")
    }

    @Test("Opaque emitters honor visibility and unsupported materials retain their hit radiance", arguments: PathAccelerationTests.modes)
    func visibilityAndFallbacks(acceleration: PathTracingScene.Options.Acceleration) throws {
        let plane = try MeshGeometry.plane(width: 1, height: 1)
        let cases = try [
            (PhysicallyBasedMaterial(emissiveFactor: SIMD3(0.8, 0.4, 0.2), alphaMode: .mask(cutoff: 0.5)), plane),
            (
                PhysicallyBasedMaterial(
                    emissiveFactor: SIMD3(0.8, 0.4, 0.2), alphaMode: .opaque,
                    subsurface: .init(weight: 1, color: SIMD3(repeating: 1), radius: SIMD3(repeating: 1), scale: 0.25)
                ), MeshGeometry.cube()
            )
        ]
        for (material, geometry) in cases {
            let nodes = [
                SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: try emitter(material, geometry: geometry))
            ]
            let enabled = try prepared(nodes, acceleration: acceleration)
            let disabled = try prepared(nodes, acceleration: acceleration, sampled: false)
            #expect(enabled.sampledEmissiveTriangleCount == 0)
            let a = try trace(PathTracer(queue: fixture.queue, scene: enabled), samples: 16)
            let b = try trace(PathTracer(queue: fixture.queue, scene: disabled), samples: 16)
            #expect(a == b && a.x > 0.4)
        }

        let receiver = try self.receiver()
        let emitter = try self.emitter(.init(emissiveFactor: SIMD3(repeating: 1), alphaMode: .opaque))
        let front = SceneNode(transform: facingReceiver(position: SIMD3(0.6, 0, -2), size: SIMD2(0.8, 0.8)), mesh: emitter)
        let back = SceneNode(transform: .init(translation: SIMD3(0.6, 0, -2), scale: SIMD3(0.8, 0.8, 1)), mesh: emitter)
        let occluder = try fixture.pbr(.plane(width: 0.5, height: 0.6), color: .zero)
        let blocker = SceneNode(
            transform: .init(translation: SIMD3(0.3, 0, -2.5), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))),
            mesh: occluder
        )
        let doubleSided = SceneNode(
            transform: back.transform, mesh: try self.emitter(.init(
                emissiveFactor: SIMD3(repeating: 1), alphaMode: .opaque, isDoubleSided: true
            ))
        )
        let visibilityCases = [
            ([receiver, front], true), ([receiver, back], false), ([receiver, front, blocker], false),
            ([receiver, doubleSided], true)
        ]
        for (nodes, visible) in visibilityCases {
            let scene = try prepared(nodes, acceleration: acceleration)
            #expect(scene.sampledEmissiveTriangleCount == 2)
            let output = try trace(PathTracer(queue: fixture.queue, scene: scene), samples: 128)
            if visible {
                #expect(output.x > 0.05 && output.w == 1, "\(output)")
            } else {
                #expect(output == SIMD4(0, 0, 0, 1), "\(output)")
            }
        }

        let cameraEmitter = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: emitter)
        let cameraScene = try prepared([cameraEmitter], acceleration: acceleration)
        #expect(cameraScene.sampledEmissiveTriangleCount == 2)
        #expect(try trace(PathTracer(queue: fixture.queue, scene: cameraScene), samples: 1) == SIMD4(repeating: 1))
    }

    private func receiver() throws -> SceneNode {
        let mesh = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 4, height: 4),
            material: .physicallyBased(try .init(roughness: 1, alphaMode: .opaque, specular: .init(factor: 0)))
        )
        return SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)
    }

    private func emitter(_ material: PhysicallyBasedMaterial, geometry: MeshGeometry? = nil) throws -> MeshRenderer {
        try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry ?? .plane(width: 1, height: 1),
            material: .physicallyBased(material), supportsInstancing: true
        )
    }

    private func facingReceiver(position: SIMD3<Float>, size: SIMD2<Float>) -> SceneTransform {
        SceneTransform(
            translation: position, rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0)),
            scale: SIMD3(-size.x, size.y, 1)
        )
    }

    private func prepared(
        _ nodes: [SceneNode], acceleration: PathTracingScene.Options.Acceleration, sampled: Bool = true,
        textures: PathTracingScene.Options.Textures = .baked(size: 16)
    ) throws -> PathTracingScene {
        try PathTracingScene(
            queue: fixture.queue, graph: SceneGraph(roots: nodes, camera: fixture.camera),
            options: .init(textures: textures, acceleration: acceleration, sampleEmissiveMeshes: sampled)
        )
    }

    private func trace(
        _ tracer: PathTracer, samples: Int, multipleImportanceSampling: Bool = true
    ) throws -> SIMD4<Float> {
        let target = try fixture.output(width: 1, height: 1)
        let accumulation = try tracer.makeAccumulation(width: 1, height: 1)
        var previous: PathTracer.History?
        for start in stride(from: 0, to: samples, by: 16) {
            let command = try #require(fixture.queue.makeCommandBuffer())
            previous = try encode(
                tracer, target: target, accumulation: accumulation, previous: previous,
                command: command, samples: min(16, samples - start), multipleImportanceSampling: multipleImportanceSampling
            )
            try fixture.finish(command)
        }
        return try #require(fixture.readback(target).first)
    }

    private func encode(
        _ tracer: PathTracer, target: any MTLTexture, accumulation: PathTracer.Accumulation,
        previous: PathTracer.History? = nil, command: any MTLCommandBuffer, samples: Int = 1,
        multipleImportanceSampling: Bool = true
    ) throws -> PathTracer.History {
        try tracer.encode(
            camera: fixture.camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
            previous: previous, commandBuffer: command, samplesPerPixel: samples,
            settings: .init(maxBounces: 1, seed: 49, multipleImportanceSampling: multipleImportanceSampling),
            region: .init(imageWidth: 63, imageHeight: 63, x: 31, y: 31)
        )
    }
}
