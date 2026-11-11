import Foundation
import Frostfield
import Metal
import simd
import Testing

/// Tangent frames, parallax and triplanar mapping, model matrices, screen inputs and
/// graph cameras, each rendered with raster PBR and the path tracer where both apply
/// and compared with closed forms at chosen pixels.
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ShaderGraphViewTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let camera: PerspectiveCamera
    private let cameraTransform = SceneTransform(translation: SIMD3(0, 0, 0.5))
    private static let size = 65
    /// tan of half the 60° vertical field of view.
    private static let halfExtent = Float(tan(Double.pi / 6))

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        queue = try #require(device.makeCommandQueue())
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20)
    }

    // MARK: - Tangent frames

    @Test("Graph tangent frames follow +U and +V of UV0 in both renderers, whatever the authored tangent sign")
    func tangentFrames() throws {
        let base = try MeshGeometry.plane(width: 3, height: 3)
        func withTangent(_ tangent: SIMD4<Float>?, mirrored: Bool = false) throws -> MeshGeometry {
            try MeshGeometry(vertices: base.vertices.map { vertex in
                let uv = vertex.textureCoordinate
                return MeshVertex(
                    position: vertex.position, normal: vertex.normal,
                    textureCoordinate: mirrored ? SIMD2(1 - uv.x, uv.y) : uv, tangent: tangent
                )
            }, indices: base.indices)
        }
        // The plane spans +X with U and -Y with V. Mirroring U makes +U run along -X.
        let cases: [(name: String, geometry: MeshGeometry, tangent: SIMD3<Float>)] = [
            ("derived", base, SIMD3(1, 0, 0)),
            ("w = -1", try withTangent(SIMD4(1, 0, 0, -1)), SIMD3(1, 0, 0)),
            ("w = +1 (glTF convention)", try withTangent(SIMD4(1, 0, 0, 1)), SIMD3(1, 0, 0)),
            ("mirrored U", try withTangent(nil, mirrored: true), SIMD3(-1, 0, 0)),
            ("mirrored U, w = +1", try withTangent(SIMD4(-1, 0, 0, 1), mirrored: true), SIMD3(-1, 0, 0)),
            ("mirrored U, w = -1", try withTangent(SIMD4(-1, 0, 0, -1), mirrored: true), SIMD3(-1, 0, 0)),
        ]
        let frame = try SurfaceShaderGraph { s in
            .init(emission: s.tangent * 0.25 + s.bitangent * 0.125 + 0.5)
        }
        let direction = SIMD3<Float>(0.6, -0.48, 0.64)
        let local = try SurfaceShaderGraph { s in
            .init(emission: s.tangentSpace(s.constant(direction)) * 0.5 + 0.5)
        }
        let bitangent = SIMD3<Float>(0, -1, 0)
        for item in cases {
            let expectedFrame = item.tangent * 0.25 + bitangent * 0.125 + 0.5
            let expectedLocal = SIMD3(simd_dot(direction, item.tangent), simd_dot(direction, bitangent), direction.z) * 0.5 + 0.5
            for (graph, expected) in [(frame, expectedFrame), (local, expectedLocal)] {
                let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -1.5)), mesh: try emissive(graph, item.geometry))
                let raster = try rasterize([node])[32, 32]
                let path = try trace([node], samples: 1)[32, 32]
                #expect(close(raster, SIMD4(expected, 1)), "raster \(item.name): \(raster) vs \(expected)")
                #expect(close(path, SIMD4(expected, 1)), "path \(item.name): \(path) vs \(expected)")
            }

            // A graph normal leaning toward +V uses the same frame: lit from below (+V is -Y),
            // white Lambertian brightness is cos(normal, light) / π in both renderers.
            let fromBelow = simd_normalize(SIMD3<Float>(0, -1, 1))
            let leaning = try SurfaceShaderGraph { s in .init(normal: s.constant(SIMD3<Float>(0, 0.6, 0.8))) }
            let lit = try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: item.geometry,
                material: .physicallyBased(try .init(
                    roughness: 1, directionToLight: fromBelow, ambientIntensity: 0, alphaMode: .opaque,
                    surfaceGraph: leaning, specular: .init(factor: 0)
                ))
            )
            let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -1.5)), mesh: lit)
            let light = try PathTracer.AnalyticLight(PunctualLight(kind: .directional), direction: -fromBelow)
            let lighting = try PathTracer.Lighting(analyticLights: [light], environment: .zero)
            let expected = simd_dot(SIMD3<Float>(0, -0.6, 0.8), fromBelow) / .pi
            let raster = try rasterize([node])[32, 32].x
            let path = try trace([node], samples: 4, lighting: lighting, bounces: 1)[32, 32].x
            #expect(abs(raster - expected) < 0.01 * expected, "raster normal \(item.name): \(raster) vs \(expected)")
            #expect(abs(path - expected) < 0.01 * expected, "path normal \(item.name): \(path) vs \(expected)")
        }
    }

    // MARK: - Parallax and triplanar

    /// The world point a pixel center sees on the plane z = -1.5, two units from the camera.
    private static func planePoint(x: Int, y: Int) -> SIMD3<Float> {
        let ndc = SIMD2((Float(x) + 0.5) / Float(size) * 2 - 1, 1 - (Float(y) + 0.5) / Float(size) * 2)
        return SIMD3(ndc * 2 * halfExtent, -1.5)
    }

    @Test("Parallax occlusion marches away from the viewer in UV0 in both renderers, whatever the authored tangent sign")
    func parallaxInRenderers() throws {
        // A height of zero everywhere: every pixel hits the floor a full depth below the
        // surface, so the offset is -(v.xy / v.z) · depth in tangent space.
        let zero = try dataTexture(SIMD4(0, 0, 0, 1))
        let depth: Float = 0.1
        let graph = try SurfaceShaderGraph { s in
            let height = s.texture("Height", content: .data)
            let shifted = height.parallaxOcclusion(s.uv0, view: s.tangentSpace(s.viewDirection), depth: s.constant(depth), steps: 8)
            return .init(emission: s.vector((shifted - s.uv0) * 4 + 0.5, s.constant(Float(0.5))))
        }
        let base = try MeshGeometry.plane(width: 3, height: 3)
        let geometries: [(String, MeshGeometry)] = try [("derived", base)] + [Float(-1), 1].map { sign in
            ("w = \(sign)", try MeshGeometry(vertices: base.vertices.map {
                MeshVertex(position: $0.position, normal: $0.normal, textureCoordinate: $0.textureCoordinate,
                           tangent: SIMD4(1, 0, 0, sign))
            }, indices: base.indices))
        }
        let pixels = [(12, 12), (52, 20), (20, 50), (32, 32), (60, 58)]
        for (name, geometry) in geometries {
            let node = SceneNode(
                transform: .init(translation: SIMD3(0, 0, -1.5)),
                mesh: try emissive(graph, geometry, textures: ["Height": zero])
            )
            let raster = try rasterize([node])
            let path = try trace([node], samples: 64)
            for (x, y) in pixels {
                // Tangent space is (+U, +V, N) = (+X, -Y, +Z) on this plane.
                let view = simd_normalize(cameraTransform.translation - Self.planePoint(x: x, y: y))
                let local = SIMD3(view.x, -view.y, view.z)
                let offset = -SIMD2(local.x, local.y) / local.z * depth
                let expected = SIMD4(SIMD3(offset * 4 + 0.5, 0.5), 1)
                #expect(close(raster[x, y], expected), "raster \(name) at \(x), \(y): \(raster[x, y]) vs \(expected)")
                #expect(close(path[x, y], expected, slack: 2e-3), "path \(name) at \(x), \(y): \(path[x, y]) vs \(expected)")
            }
        }
    }

    @Test("Triplanar weights and three-texture sampling agree with the closed form in both renderers")
    func triplanarInRenderers() throws {
        let textures = [
            "X": try dataTexture(SIMD4(1, 0, 0, 1)), "Y": try dataTexture(SIMD4(0, 1, 0, 1)),
            "Z": try dataTexture(SIMD4(0, 0, 1, 1)),
        ]
        let weights = try SurfaceShaderGraph { s in
            .init(emission: s.triplanarWeights(normal: s.normal, sharpness: 3))
        }
        let sampled = try SurfaceShaderGraph { s in
            let x = s.texture("X", content: .data), y = s.texture("Y", content: .data), z = s.texture("Z", content: .data)
            let color = s.triplanar(x: x, y: y, z: z, position: s.worldPosition, normal: s.normal, sharpness: 3)
            return .init(emission: color.xyz)
        }
        let rotation = simd_quatf(angle: 0.7, axis: simd_normalize(SIMD3<Float>(1, 0.6, 0)))
        let normal = rotation.act(SIMD3<Float>(0, 0, 1))
        let raised = SIMD3(pow(abs(normal.x), 3), pow(abs(normal.y), 3), pow(abs(normal.z), 3))
        let expected = SIMD4(raised / (raised.x + raised.y + raised.z), 1)
        // The tilt mixes all three projections.
        #expect(simd_reduce_min(SIMD3(expected.x, expected.y, expected.z)) > 0.02)
        for graph in [weights, sampled] {
            let node = SceneNode(
                transform: .init(translation: SIMD3(0, 0, -1.5), rotation: rotation),
                mesh: try emissive(graph, try .plane(width: 1.2, height: 1.2), textures: graph == sampled ? textures : [:])
            )
            let raster = try rasterize([node])[32, 32]
            let path = try trace([node], samples: 1)[32, 32]
            #expect(close(raster, expected), "raster: \(raster) vs \(expected)")
            #expect(close(path, expected), "path: \(path) vs \(expected)")
        }
    }

    // MARK: - Model matrices

    @Test("modelMatrix and normalMatrix follow a rotation about a tilted axis with non-uniform scale in both renderers")
    func modelMatrices() throws {
        let transform = SceneTransform(
            translation: SIMD3(0.1, -0.05, -1.5), rotation: simd_quatf(angle: 0.5, axis: simd_normalize(SIMD3<Float>(1, 1, 0.3))),
            scale: SIMD3(1.2, 0.9, 0.5)
        )
        let probe = SIMD3<Float>(0.3, -0.5, 0.8)
        let normalGraph = try SurfaceShaderGraph { s in .init(emission: s.normalMatrix * s.constant(probe) * 0.25 + 0.5) }
        let modelGraph = try SurfaceShaderGraph { s in
            .init(emission: s.modelMatrix.transformDirection(s.constant(probe)) * 0.25 + 0.5)
        }
        let pointGraph = try SurfaceShaderGraph { s in
            .init(emission: s.modelMatrix.transformPoint(s.constant(probe)) * 0.25 + 0.5)
        }
        let model = transform.matrix
        let upper = simd_float3x3(
            SIMD3(model.columns.0.x, model.columns.0.y, model.columns.0.z),
            SIMD3(model.columns.1.x, model.columns.1.y, model.columns.1.z),
            SIMD3(model.columns.2.x, model.columns.2.y, model.columns.2.z)
        )
        let normalMatrix = simd_inverse(simd_transpose(upper))
        let point = model * SIMD4(probe, 1)
        // The transposed matrix would give a clearly different answer, so this checks the orientation.
        #expect(simd_distance(normalMatrix * probe, simd_transpose(normalMatrix) * probe) > 0.2)
        let cases: [(SurfaceShaderGraph, SIMD3<Float>)] = [
            (normalGraph, normalMatrix * probe * 0.25 + 0.5),
            (modelGraph, upper * probe * 0.25 + 0.5),
            (pointGraph, SIMD3(point.x, point.y, point.z) * 0.25 + 0.5),
        ]
        for (graph, expected) in cases {
            let node = SceneNode(transform: transform, mesh: try emissive(graph, try .plane(width: 1, height: 1)))
            let raster = try rasterize([node])[32, 32]
            let path = try trace([node], samples: 1)[32, 32]
            #expect(close(raster, SIMD4(expected, 1)), "raster: \(raster) vs \(expected)")
            #expect(close(path, SIMD4(expected, 1)), "path: \(path) vs \(expected)")
        }
    }

    // MARK: - Screen inputs

    @Test("A surface graph reads its screen inputs while the same draw reads an alpha map")
    func graphViewWithAlphaMap() throws {
        let graph = try SurfaceShaderGraph { s in
            .init(emission: s.vector(s.screenUV, s.constant(Float(0.25))))
        }
        let geometry = try MeshGeometry.plane(width: 3, height: 3)
        let pose = SceneTransform(translation: SIMD3(0, 0, -1.5))
        let cases: [(Float, SIMD4<Float>)] = [(0, .zero), (1, SIMD4(0.5, 0.5, 0.25, 1))]
        for (green, expected) in cases {
            let alpha = try dataTexture(SIMD4(0, green, 0, 1))
            let material = try PhysicallyBasedMaterial(
                lightIntensity: 0, ambientIntensity: 0, alphaMode: .mask(cutoff: 0.5),
                surfaceGraph: graph, alphaTexture: .init(texture: alpha)
            )
            let mesh = try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(material)
            )
            let composed = try rasterize([SceneNode(transform: pose, mesh: mesh)])
            let direct = try rasterize([]) { _, pass, buffer in
                try mesh.encode(
                    MeshScene(camera: camera, transform: pose, cameraTransform: cameraTransform),
                    pass: pass, commandBuffer: buffer
                )
            }
            #expect(close(composed[32, 32], expected))
            #expect(close(direct[32, 32], expected))
        }
    }

    @Test("screenUV, fragmentCoordinate and viewportSize follow an offset viewport; screenCoordinate and screenSize cover the target")
    func offsetViewport() throws {
        let origin = SIMD2<Float>(10, 6), extent = SIMD2<Float>(40, 52)
        let viewport = MTLViewport(
            originX: Double(origin.x), originY: Double(origin.y), width: Double(extent.x), height: Double(extent.y),
            znear: 0, zfar: 1
        )
        let clip = try camera.projection(aspectRatio: extent.x / extent.y) * simd_inverse(cameraTransform.matrix)
        let graphs = [
            try SurfaceShaderGraph { s in .init(emission: s.vector(s.screenUV, s.fragmentCoordinate.x / 100)) },
            try SurfaceShaderGraph { s in
                .init(emission: s.vector(s.fragmentCoordinate.y / 100, s.screenCoordinate.x / 100, s.screenCoordinate.y / 100))
            },
            try SurfaceShaderGraph { s in .init(emission: s.vector(s.screenSize / 100, s.viewportSize.y / 100)) },
        ]
        func expected(_ index: Int, _ x: Int, _ y: Int) -> SIMD4<Float> {
            let pixel = SIMD2(Float(x) + 0.5, Float(y) + 0.5)
            let local = pixel - origin
            switch index {
            case 0: return SIMD4(SIMD3(local / extent, local.x / 100), 1)
            case 1: return SIMD4(local.y / 100, pixel.x / 100, pixel.y / 100, 1)
            default: return SIMD4(SIMD3(SIMD2(repeating: Float(Self.size)) / 100, extent.y / 100), 1)
            }
        }
        for (index, graph) in graphs.enumerated() {
            let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -1.5)), mesh: try emissive(graph, try .plane(width: 4, height: 4)))
            let image = try rasterize([node]) { scene, pass, buffer in
                try scene.encode(pass: pass, commandBuffer: buffer, clipFromWorld: clip, cameraPosition: cameraTransform.translation,
                                 viewport: viewport)
            }
            for (x, y) in [(11, 7), (15, 10), (30, 30), (48, 56)] {
                #expect(close(image[x, y], expected(index, x, y)), "graph \(index) at \(x), \(y): \(image[x, y])")
            }
            // Outside the viewport nothing is drawn.
            #expect(image[5, 5] == .zero && image[55, 62] == .zero)
        }

        // Without a viewport the screen and viewport inputs coincide.
        let whole = try rasterize([SceneNode(
            transform: .init(translation: SIMD3(0, 0, -1.5)), mesh: try emissive(graphs[2], try .plane(width: 4, height: 4))
        )])
        #expect(close(whole[32, 32], SIMD4(0.65, 0.65, 0.65, 1)), "\(whole[32, 32])")

        // Post-processing covers its whole output.
        let fixture = try PostProcessFixture()
        let post = try PostProcessGraph { p in
            p.vector(p.screenCoordinate, p.screenSize.x + p.screenSize.y * 10, p.constant(Float(1)))
        }
        let pass = try PostProcessPass(graph: post, device: fixture.device)
        let source = try fixture.color(width: 4, height: 2) { _, _ in SIMD4(0, 0, 0, 1) }
        let destination = try fixture.color(width: 4, height: 2)
        try fixture.run { try pass.encode(source: source, destination: destination, commandBuffer: $0) }
        let texels = try fixture.read(destination)
        for y in 0..<2 {
            for x in 0..<4 {
                #expect(texels[y * 4 + x] == SIMD4(Float(x) + 0.5, Float(y) + 0.5, 24, 1), "post \(x), \(y): \(texels[y * 4 + x])")
            }
        }

        // The path tracer has no pixel grid at secondary bounces.
        for input in [MaterialGraph.Input.screenCoordinate, .screenSize] {
            let graph = try SurfaceShaderGraph(nodes: [.input(input), .swizzle(0, "xyx")], outputs: .init(emission: 1))
            let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -1.5)), mesh: try emissive(graph, try .plane()))
            let rejection = #expect(throws: PathTracingScene.Failure.self) {
                try PathTracingScene(queue: queue, graph: SceneGraph(roots: [node], camera: camera))
            }
            if case .unsupported(_, let feature) = rejection {
                #expect(feature.contains("node 0") && feature.contains("screenCoordinate"), "\(feature)")
            } else {
                Issue.record("Unexpected failure \(String(describing: rejection))")
            }
        }
        // Compute stages have no render target.
        #expect(throws: MaterialGraph.Failure.unavailable(node: 0, stage: .compute)) {
            try BufferShaderGraph(fieldCount: 1) { b in [0: b.vector(b.screenCoordinate, b.screenSize)] }
        }
    }

    // MARK: - Graph cameras

    @Test("Path graph matrices follow the camera model's perspective shift and orthographic scale")
    func pathCameraProjection() throws {
        let graph = try SurfaceShaderGraph { s in
            .init(emission: s.vector(
                s.projectionMatrix.column(2).x + 0.5,
                s.projectionMatrix.column(0).x / 4,
                s.cameraFar / 40
            ))
        }
        let node = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -1.5)),
            mesh: try emissive(graph, try .plane(width: 4, height: 4))
        )
        let perspective = PathTracer.CameraModel(
            projection: .perspective(verticalFieldOfView: camera.verticalFieldOfView, shift: SIMD2(0.25, 0)), far: 20
        )
        let orthographic = PathTracer.CameraModel(projection: .orthographic(height: 2), far: 20)
        #expect(close(
            try trace([node], samples: 1, cameraModel: perspective)[32, 32],
            SIMD4(0.75, 1 / (4 * Self.halfExtent), 0.5, 1)
        ))
        #expect(close(
            try trace([node], samples: 1, cameraModel: orthographic)[32, 32],
            SIMD4(0.5, 0.25, 0.5, 1)
        ))
        let tile = try PathTracer.Region(imageWidth: 130, imageHeight: 65, x: 32, y: 0)
        #expect(close(
            try trace([node], samples: 1, cameraModel: perspective, region: tile)[32, 32],
            SIMD4(0.75, 1 / (8 * Self.halfExtent), 0.5, 1)
        ))
        #expect(throws: PathTracer.Failure.graphCameraProjection) {
            try trace([node], samples: 1, cameraModel: .init(projection: .equirectangular))
        }
    }

    @Test("Clip-matrix encodes read the view, projection and clip distances from a ShaderGraphCamera, including glTF camera views")
    func graphCameras() throws {
        let graph = try SurfaceShaderGraph { s in
            let view = s.viewMatrix.transformPoint(s.worldPosition)
            let clip = s.projectionMatrix * s.vector(view, s.constant(Float(1)))
            return .init(emission: s.vector(-view.z / 4 + s.cameraNear, clip.w / 4 + s.projectionMatrix.column(1).y / 8, s.cameraFar / 40))
        }
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -1.5)), mesh: try emissive(graph, try .plane(width: 4, height: 4)))
        // Two units in front: -view.z = 2 and clip.w = 2; near 0.1, far 20.
        let focal = 1 / Self.halfExtent
        let expected = SIMD4<Float>(0.6, 0.5 + focal / 8, 0.5, 1)
        let reference = try rasterize([node])[32, 32]
        #expect(close(reference, expected), "\(reference)")

        let viewMatrix = simd_inverse(cameraTransform.matrix)
        let projection = try camera.projection(aspectRatio: 1)
        let graphCamera = try ShaderGraphCamera(viewMatrix: viewMatrix, projectionMatrix: projection, near: 0.1, far: 20)
        let clipped = try rasterize([node]) { scene, pass, buffer in
            try scene.encode(pass: pass, commandBuffer: buffer, clipFromWorld: projection * viewMatrix,
                             cameraPosition: cameraTransform.translation, graphCamera: graphCamera)
        }[32, 32]
        #expect(close(clipped, expected), "\(clipped)")

        // Without a graph camera, or with one that disagrees with the clip matrix, the encode fails.
        let scene = SceneGraph(roots: [node], camera: camera, cameraTransform: cameraTransform)
        let (pass, _) = try target()
        let buffer = try #require(queue.makeCommandBuffer())
        #expect(throws: MeshRenderer.Failure.graphCameraInputs) {
            try scene.encode(pass: pass, commandBuffer: buffer, clipFromWorld: projection * viewMatrix,
                             cameraPosition: cameraTransform.translation)
        }
        let shifted = try ShaderGraphCamera(
            viewMatrix: simd_inverse(SceneTransform(translation: SIMD3(0, 0, 0.6)).matrix), projectionMatrix: projection,
            near: 0.1, far: 20
        )
        #expect(throws: MeshRenderer.Failure.graphCameraInputs) {
            try scene.encode(pass: pass, commandBuffer: buffer, clipFromWorld: projection * viewMatrix,
                             cameraPosition: cameraTransform.translation, graphCamera: shifted)
        }
        #expect(throws: ShaderGraphCamera.Failure.invalid) {
            try ShaderGraphCamera(viewMatrix: viewMatrix, projectionMatrix: projection, near: 1, far: 1)
        }
        #expect(throws: ShaderGraphCamera.Failure.invalid) {
            try ShaderGraphCamera(viewMatrix: simd_float4x4(), projectionMatrix: projection, near: 0.1, far: 1)
        }

        // A glTF camera at the same pose with the same lens supplies its own graph camera.
        let document = try JSONSerialization.data(withJSONObject: [
            "asset": ["version": "2.0"], "scenes": [["nodes": [0, 1]]],
            "nodes": [["camera": 0, "translation": [0, 0, 0.5]], ["camera": 1, "translation": [0, 0, 0.5]]],
            "cameras": [
                ["type": "perspective", "perspective": ["yfov": Double.pi / 3, "aspectRatio": 1, "znear": 0.1, "zfar": 20]],
                ["type": "perspective", "perspective": ["yfov": Double.pi / 3, "znear": 0.1]],
            ],
        ])
        let prepared = try PreparedGLTFScene(
            device: device, colorPixelFormat: .rgba16Float, scene: try GLTFMeshLoader.loadScene(data: document, includeCameras: true)
        )
        var gltfGraph = try prepared.graph(camera: camera)
        gltfGraph.roots.append(node)
        let view = try prepared.cameraView(nodeIndex: 0, in: gltfGraph, viewportAspectRatio: 1)
        #expect(view.graphCamera.projectionMatrix * view.graphCamera.viewMatrix == view.clipFromWorld)
        #expect(view.graphCamera.near == 0.1 && view.graphCamera.far == 20)
        let viewed = try rasterize(gltfGraph.roots, scene: gltfGraph) { scene, pass, buffer in
            try scene.encode(pass: pass, commandBuffer: buffer, clipFromWorld: view.clipFromWorld,
                             cameraPosition: view.cameraPosition, graphCamera: view.graphCamera)
        }[32, 32]
        #expect(close(viewed, expected), "glTF camera: \(viewed)")
        // An infinite projection reports an infinite far distance.
        let infinite = try prepared.cameraView(nodeIndex: 1, in: gltfGraph, viewportAspectRatio: 1)
        #expect(infinite.graphCamera.far == .infinity && infinite.graphCamera.near == 0.1)
    }

    // MARK: - Helpers

    private struct Image {
        let pixels: [SIMD4<Float>]
        subscript(x: Int, y: Int) -> SIMD4<Float> { pixels[y * ShaderGraphViewTests.size + x] }
    }

    private func emissive(
        _ graph: SurfaceShaderGraph, _ geometry: MeshGeometry, textures: [String: any MTLTexture] = [:]
    ) throws -> MeshRenderer {
        try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: geometry,
            material: .physicallyBased(try .init(
                lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, surfaceGraph: graph, surfaceTextures: textures
            ))
        )
    }

    /// A 4 × 4 RGBA8 texture of one value, readable by both renderers.
    private func dataTexture(_ value: SIMD4<Float>) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 4, height: 4, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        let texel = SIMD4<UInt8>((value * 255).rounded(.toNearestOrEven))
        let bytes = [SIMD4<UInt8>](repeating: texel, count: 16)
        bytes.withUnsafeBytes {
            texture.replace(region: MTLRegionMake2D(0, 0, 4, 4), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 16)
        }
        return texture
    }

    private func target() throws -> (MTLRenderPassDescriptor, any MTLTexture) {
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: Self.size, height: Self.size, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        let color = try #require(device.makeTexture(descriptor: colorDescriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: Self.size, height: Self.size, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = try #require(device.makeTexture(descriptor: depthDescriptor))
        pass.depthAttachment.clearDepth = 1
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        return (pass, color)
    }

    /// Raster PBR through the default camera encode, or through `encode` when given.
    private func rasterize(
        _ roots: [SceneNode], scene: SceneGraph? = nil,
        encode: ((SceneGraph, MTLRenderPassDescriptor, any MTLCommandBuffer) throws -> Void)? = nil
    ) throws -> Image {
        let graph = scene ?? SceneGraph(roots: roots, camera: camera, cameraTransform: cameraTransform)
        let (pass, color) = try target()
        let buffer = try #require(queue.makeCommandBuffer())
        if let encode {
            try encode(graph, pass, buffer)
        } else {
            try graph.encode(pass: pass, commandBuffer: buffer)
        }
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)
        return try read(color)
    }

    private func trace(
        _ roots: [SceneNode], samples: Int, lighting: PathTracer.Lighting? = nil, bounces: Int = 4,
        cameraModel: PathTracer.CameraModel? = nil, region: PathTracer.Region? = nil
    ) throws -> Image {
        let graph = SceneGraph(roots: roots, camera: camera, cameraTransform: cameraTransform)
        let tracer = try PathTracer(queue: queue, scene: try PathTracingScene(queue: queue, graph: graph))
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: Self.size, height: Self.size, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderWrite
        let traced = try #require(device.makeTexture(descriptor: descriptor))
        let accumulation = try tracer.makeAccumulation(width: Self.size, height: Self.size)
        let model = cameraModel ?? .init(
            projection: .perspective(verticalFieldOfView: camera.verticalFieldOfView), near: camera.near, far: camera.far
        )
        // One encode traces at most 16 samples per pixel; accumulate the rest.
        var history: PathTracer.History?
        var remaining = samples
        while remaining > 0 {
            let count = min(remaining, 16)
            let buffer = try #require(queue.makeCommandBuffer())
            history = try tracer.encode(cameraModel: model, cameraTransform: cameraTransform, destination: traced,
                                        accumulation: accumulation, previous: history, commandBuffer: buffer,
                                        samplesPerPixel: count, settings: .init(maxBounces: bounces), lighting: lighting,
                                        region: region)
            buffer.commit()
            buffer.waitUntilCompleted()
            try #require(buffer.status == .completed)
            remaining -= count
        }
        return try read(traced)
    }

    private func read(_ texture: any MTLTexture) throws -> Image {
        var halves = [Float16](repeating: 0, count: Self.size * Self.size * 4)
        try halves.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: Self.size * 8,
                from: MTLRegionMake2D(0, 0, Self.size, Self.size), mipmapLevel: 0
            )
        }
        return Image(pixels: (0..<(Self.size * Self.size)).map {
            SIMD4(Float(halves[$0 * 4]), Float(halves[$0 * 4 + 1]), Float(halves[$0 * 4 + 2]), Float(halves[$0 * 4 + 3]))
        })
    }

    /// Half-float output resolves about 1e-3 near 1; allow 2e-3 plus `slack`.
    private func close(_ a: SIMD4<Float>, _ b: SIMD4<Float>, slack: Float = 0) -> Bool {
        simd_reduce_max(simd_abs(a - b)) <= 2e-3 + slack + 1e-3 * simd_reduce_max(simd_abs(b))
    }
}
