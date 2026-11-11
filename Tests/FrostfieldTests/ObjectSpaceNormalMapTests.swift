import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ObjectSpaceNormalMapTests {
    enum Placement: CaseIterable, Sendable {
        case node, instance, mirroredInstance, backFace

        var transform: SceneTransform {
            let rotation = simd_quatf(angle: self == .backFace ? .pi + 0.22 : 0.22, axis: SIMD3(0, 1, 0))
            return .init(translation: SIMD3(0, 0, -3), rotation: rotation, scale: SIMD3(1.4, 0.8, 1.15))
        }

        var instance: simd_float4x4? {
            guard self == .instance || self == .mirroredInstance else { return nil }
            return SceneTransform(
                rotation: simd_quatf(angle: 0.41, axis: SIMD3(0, 0, 1)),
                scale: SIMD3(self == .mirroredInstance ? -0.62 : 0.62, 1.3, 1.65)
            ).matrix
        }

        func normal(_ local: SIMD3<Float>) -> SIMD3<Float> {
            let world = transform.matrix * (instance ?? matrix_identity_float4x4)
            let transformed = simd_transpose(simd_inverse(world)) * SIMD4(local, 0)
            return simd_normalize(SIMD3(transformed.x, transformed.y, transformed.z)) * (self == .backFace ? -1 : 1)
        }
    }

    @Test("Packed XY uses the same object normal transform as RGB after reconstructing Z")
    func packedMap() throws {
        let pair = try normalMap(encoding: .xyUNorm)
        let expected = Placement.instance.normal(pair.normal)
        let graph = try graph(base(pair.binding), placement: .instance, directionToLight: expected)
        #expect(simd_distance(try rasterNormal(graph), expected) < 0.002)
        let traced = try trace(graph)
        #expect(simd_distance(traced.normal, expected) < 0.002 && traced.beauty > 0.03)
    }

    private let gpu: PathTestFixture
    private let side = 17
    private var center: Int { side / 2 * side + side / 2 }

    init() throws { gpu = try PathTestFixture() }

    @Test("Object normals override vertex and tangent frames in lit pixels under node, instance, mirrored and back-face transforms",
          arguments: Placement.allCases)
    func baseMap(_ placement: Placement) throws {
        let pair = try normalMap()
        let light = simd_normalize(SIMD3<Float>(0.6, 0.4, 1))
        let expected = placement.normal(pair.normal)
        let surface = try graph(base(pair.binding), placement: placement, directionToLight: light)
        let reference = try graph(base(nil), placement: placement, vertexNormal: pair.normal, directionToLight: light)
        let raster = try rasterNormal(surface)
        #expect(simd_distance(raster, expected) < 0.002, "raster \(raster), expected \(expected)")
        let rasterColor = try gpu.rasterize(surface, width: side, height: side)[center].x
        let rasterReference = try gpu.rasterize(reference, width: side, height: side)[center].x
        #expect(rasterColor > 0.03 && abs(rasterColor - rasterReference) < 0.003)

        let traced = try trace(surface)
        let tracedReference = try trace(reference)
        #expect(simd_distance(traced.normal, expected) < 0.002, "path \(traced.normal), expected \(expected)")
        #expect(traced.beauty > 0.03 && abs(traced.beauty - tracedReference.beauty) < 0.003)
        if placement.instance != nil {
            #expect(simd_distance(expected, Placement.node.normal(pair.normal)) > 0.2)
        }
    }

    @Test("Object-space clearcoat replaces only the coat normal and ignores normal scale")
    func coatMap() throws {
        let pair = try normalMap()
        let tilted = Placement.node.normal(pair.normal)
        let mirror = simd_normalize(2 * tilted.z * tilted - SIMD3(0, 0, 1))
        func coat(_ binding: PhysicallyBasedMaterial.TextureBinding?) throws -> PhysicallyBasedMaterial {
            try PhysicallyBasedMaterial(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
                alphaMode: .opaque, clearcoat: .init(factor: 1, roughness: 0.3, normalTexture: binding, normalScale: 0),
                specular: .init(factor: 0)
            )
        }
        let graph = try graph(coat(pair.binding), placement: .node, directionToLight: mirror)
        let reference = try self.graph(coat(nil), placement: .node, vertexNormal: pair.normal, directionToLight: mirror)
        let flat = try self.graph(coat(nil), placement: .node, directionToLight: mirror)
        let raster = try gpu.rasterize(graph, width: side, height: side)[center].x
        let expected = try gpu.rasterize(reference, width: side, height: side)[center].x
        let flatRaster = try gpu.rasterize(flat, width: side, height: side)[center].x
        #expect(raster > flatRaster * 2 && abs(raster - expected) < 0.004, "\(raster), reference \(expected), flat \(flatRaster)")

        let traced = try trace(graph)
        let tracedReference = try trace(reference)
        let tracedFlat = try trace(flat)
        #expect(traced.beauty > tracedFlat.beauty * 2 && abs(traced.beauty - tracedReference.beauty) < 0.004)
        #expect(simd_distance(traced.normal, tracedFlat.normal) < 0.002)
    }

    @Test("An object-space base map leaves the tangent-space coat frame independent")
    func mixedSpaces() throws {
        let pair = try normalMap()
        let plane = try MeshGeometry.plane(width: 5, height: 5)
        let geometry = try MeshGeometry(vertices: plane.vertices.map {
            MeshVertex(position: $0.position, normal: $0.normal,
                       textureCoordinate: SIMD2($0.textureCoordinate.y, $0.textureCoordinate.x),
                       secondaryTextureCoordinate: SIMD2(0.25, 0.5), tangent: SIMD4(1, 0, 0, 1))
        }, indices: plane.indices)
        let tangentBinding = PhysicallyBasedMaterial.TextureBinding(
            texture: pair.binding.texture, transform: try .init(offset: SIMD2(0.25, 0)), sampling: pair.binding.sampling
        )
        let tilted = Placement.node.transform.rotation.act(pair.normal)
        let mirror = simd_normalize(2 * tilted.z * tilted - SIMD3(0, 0, 1))
        func raster(_ binding: PhysicallyBasedMaterial.TextureBinding?) throws -> Float {
            let material = try PhysicallyBasedMaterial(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
                normalTexture: binding, alphaMode: .opaque, clearcoat: .init(factor: 1, roughness: 0.3, normalTexture: tangentBinding),
                specular: .init(factor: 0)
            )
            let surface = try graph(material, placement: .node, directionToLight: mirror, geometry: geometry)
            return try gpu.rasterize(surface, width: side, height: side)[center].x
        }
        let reference = try raster(nil)
        let mixed = try raster(pair.binding)
        #expect(reference > 0.03 && abs(mixed - reference) < 0.003, "mixed \(mixed), reference \(reference)")
    }

    @Test("Object normal space survives live UV edits and archives, and tangent-only exporters reject it")
    func persistence() throws {
        let pair = try normalMap()
        let initial = PhysicallyBasedMaterial.TextureBinding(
            texture: pair.binding.texture, textureCoordinateSet: .secondary,
            sampling: pair.binding.sampling, normalSpace: .objectSpace
        )
        let live = try base(initial).withLiveFactors()
        let factors = try #require(live.liveFactors)
        let source = try graph(live, placement: .instance, directionToLight: SIMD3(0.6, 0.4, 1))
        let before = try rasterNormal(source)
        var update = factors.current
        update.textureTransforms[.normal] = try .init(offset: SIMD2(0.5, 0))
        try factors.set(update)
        let edited = try rasterNormal(source)
        #expect(simd_distance(before, edited) > 0.2)
        #expect(simd_distance(edited, Placement.instance.normal(pair.normal)) < 0.002)
        let archive = try SceneArchive(capturing: source, commandQueue: gpu.queue)
        let manifest = try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(manifest["version"] as? Int == 5)
        let restored = try archive.load(device: gpu.device, commandQueue: gpu.queue).graph
        #expect(try rasterNormal(restored) == rasterNormal(source))
        #expect(try gpu.rasterize(restored, width: side, height: side) == gpu.rasterize(source, width: side, height: side))
        #expect(throws: GLTFExporter.Failure.unsupportedTexture("Convert object-space normal maps to tangent space before glTF export.")) {
            try GLTFExporter(commandQueue: gpu.queue).export(restored)
        }
        #expect(throws: USDZExporter.Failure.invalidData("Convert object-space normal maps to tangent space before USDZ export.")) {
            try USDZExporter(commandQueue: gpu.queue).export(restored)
        }
    }

    private func base(_ binding: PhysicallyBasedMaterial.TextureBinding?) throws -> PhysicallyBasedMaterial {
        try PhysicallyBasedMaterial(
            roughness: 0.7, lightIntensity: 0, ambientIntensity: 0, normalTexture: binding, normalScale: 0,
            alphaMode: .opaque, isDoubleSided: true
        )
    }

    private func normalMap(
        encoding: PhysicallyBasedMaterial.NormalEncoding = .rgb
    ) throws -> (binding: PhysicallyBasedMaterial.TextureBinding, normal: SIMD3<Float>) {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 2, height: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let local = simd_normalize(SIMD3<Float>(0.45, 0.28, 1)) * 0.5 + 0.5
        let stored = SIMD4(Float16(local.x), Float16(local.y), Float16(local.z), 1)
        let texels = [SIMD4<Float16>(0.1, 0.5, 0.8, 1), stored]
        texels.withUnsafeBytes { bytes in
            texture.replace(region: MTLRegionMake2D(0, 0, 2, 1), mipmapLevel: 0, withBytes: bytes.baseAddress!, bytesPerRow: bytes.count)
        }
        var decoded = SIMD3(Float(stored.x), Float(stored.y), Float(stored.z)) * 2 - 1
        if encoding == .xyUNorm { decoded.z = sqrt(1 - decoded.x * decoded.x - decoded.y * decoded.y) }
        let binding = PhysicallyBasedMaterial.TextureBinding(
            texture: texture, textureCoordinateSet: .secondary,
            transform: try .init(offset: SIMD2(0.5, 0)), sampling: .init(magnification: .nearest, minification: .nearest),
            normalEncoding: encoding, normalSpace: .objectSpace
        )
        return (binding, simd_normalize(decoded))
    }

    private func graph(
        _ material: PhysicallyBasedMaterial, placement: Placement, vertexNormal: SIMD3<Float> = SIMD3(0, 0, 1),
        directionToLight: SIMD3<Float>, geometry suppliedGeometry: MeshGeometry? = nil
    ) throws -> SceneGraph {
        let geometry: MeshGeometry
        if let suppliedGeometry {
            geometry = suppliedGeometry
        } else {
            let plane = try MeshGeometry.plane(width: 5, height: 5)
            let vertices = plane.vertices.map {
                // Constant, different UV sets prove the object map is independent of a UV tangent frame.
                MeshVertex(position: $0.position, normal: vertexNormal, textureCoordinate: SIMD2(0.75, 0.5),
                           secondaryTextureCoordinate: SIMD2(0.25, 0.5))
            }
            geometry = try MeshGeometry(vertices: vertices, indices: plane.indices)
        }
        let instances = try placement.instance.map { try InstanceSet(device: gpu.device, transforms: [$0]) }
        let renderer = try MeshRenderer(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .physicallyBased(material), supportsInstancing: instances != nil
        )
        let node = SceneNode(transform: placement.transform, mesh: renderer, instances: instances)
        let light = SceneNode(
            transform: .init(rotation: simd_quatf(from: SIMD3(0, 0, -1), to: -simd_normalize(directionToLight))),
            light: try PunctualLight(kind: .directional, intensity: 1)
        )
        return SceneGraph(roots: [node, light], camera: gpu.camera)
    }

    private func rasterNormal(_ graph: SceneGraph) throws -> SIMD3<Float> {
        try gpu.rasterSurfaceNormal(graph, side: side)
    }

    private func trace(_ graph: SceneGraph) throws -> (normal: SIMD3<Float>, beauty: Float) {
        let scene = try PathTracingScene(queue: gpu.queue, graph: graph, options: .init(acceleration: .software))
        let lights = try PathTracer.AnalyticLight.lights(in: graph)
        let tracer = try PathTracer(queue: gpu.queue, scene: scene, lighting: .init(analyticLights: lights, environment: .zero))
        let accumulation = try tracer.makeAccumulation(width: side, height: side)
        let beauty = try gpu.output(width: side, height: side)
        let normal = try gpu.output(width: side, height: side)
        let command = try #require(gpu.queue.makeCommandBuffer())
        try tracer.encode(
            camera: gpu.camera, cameraTransform: .init(), destination: beauty, accumulation: accumulation,
            commandBuffer: command, samplesPerPixel: 8, settings: .init(maxBounces: 1, seed: 9), aovs: .init(normal: normal)
        )
        try gpu.finish(command)
        let n = try gpu.readback(normal)[center]
        return (SIMD3(n.x, n.y, n.z), try gpu.readback(beauty)[center].x)
    }
}
