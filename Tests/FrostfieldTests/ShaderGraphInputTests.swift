import Foundation
import Frostfield
import Metal
import simd
import Testing

/// The graph clock, camera, model and screen inputs through every renderer hook:
/// buffer and texture compute, post-processing, vertex deformation, raster PBR and
/// the path tracer, plus the inputs each hook rejects.
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ShaderGraphInputTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let camera: PerspectiveCamera
    private let cameraTransform = SceneTransform(translation: SIMD3(0, 0, 0.5))
    private let color: any MTLTexture
    private let pass: MTLRenderPassDescriptor
    private static let size = 65

    init() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        self.device = device
        queue = try #require(device.makeCommandQueue())
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20)
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: Self.size, height: Self.size, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        color = try #require(device.makeTexture(descriptor: colorDescriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: Self.size, height: Self.size, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))
        pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.clearDepth = 1
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
    }

    // MARK: - Clock

    @Test("A display clock advances every attached uniform set and rejects a negative interval")
    func displayClock() throws {
        let graph = try BufferShaderGraph(fieldCount: 1) { builder in
            [0: builder.vector(builder.time, builder.deltaTime, builder.frame, builder.constant(Float(0)))]
        }
        let uniforms = try ShaderUniforms(graph: graph)
        try ShaderDisplayClock.advance([uniforms], by: 0.25)
        #expect(uniforms.frame == ShaderFrame(time: 0.25, deltaTime: 0.25, index: 1))
        try ShaderDisplayClock.advance([uniforms], by: 0.25)
        #expect(uniforms.frame.index == 2)
        #expect(uniforms.frame.time == 0.5)
        #expect(throws: ShaderUniforms.Failure.nonfinite("frame")) { try ShaderDisplayClock.advance([uniforms], by: -1) }
    }

    @Test("The graph clock reaches buffer, texture, post-process, vertex, raster and path-traced graphs")
    func clock() throws {
        let frame = ShaderFrame(time: 2.5, deltaTime: 0.25, index: 3)

        let buffer = try BufferShaderGraph(fieldCount: 1) { b in
            [0: b.vector(b.time, b.deltaTime, b.frame, b.oscillator(.triangle, b.time))]
        }
        let bufferUniforms = try ShaderUniforms(graph: buffer)
        try bufferUniforms.setFrame(frame)
        let probe = try GraphProbe()
        // The triangle wave peaks at half-integers.
        let first = try probe.run(buffer, elements: [[.zero]], uniforms: bufferUniforms)[0][0]
        #expect(first == SIMD4(2.5, 0.25, 3, 1), "\(first)")
        let revision = bufferUniforms.revision
        let frameRevision = bufferUniforms.frameRevision
        try bufferUniforms.advanceFrame(by: 0.5)
        // The clock has its own revision; the declared values did not change.
        #expect(bufferUniforms.frameRevision == frameRevision + 1 && bufferUniforms.revision == revision)
        #expect(bufferUniforms.frame == ShaderFrame(time: 3, deltaTime: 0.5, index: 4))
        let second = try probe.run(buffer, elements: [[.zero]], uniforms: bufferUniforms)[0][0]
        #expect(second == SIMD4(3, 0.5, 4, 0), "\(second)")
        #expect(throws: ShaderUniforms.Failure.nonfinite("frame")) { try bufferUniforms.advanceFrame(by: .infinity) }
        // A graph that never reads the clock still compiles without the clock slot.
        #expect(!(try BufferShaderGraph(fieldCount: 1) { b in [0: b.elementField(0)] }.metalFunction(named: "plain"))
            .contains("frostfield_graph_uniform"))

        let texture = try TextureShaderGraph { t in t.vector(t.time, t.deltaTime, t.frame, t.constant(Float(1))) }
        let textureUniforms = try ShaderUniforms(graph: texture)
        try textureUniforms.setFrame(frame)
        let generated = try ShaderTextureGenerator(graph: texture, device: device)
            .makeTexture(width: 2, height: 2, format: .rgba32Float, uniforms: textureUniforms, queue: queue)
        var texel = SIMD4<Float>.zero
        withUnsafeMutableBytes(of: &texel) {
            generated.getBytes($0.baseAddress!, bytesPerRow: 32, from: MTLRegionMake2D(1, 1, 1, 1), mipmapLevel: 0)
        }
        #expect(texel == SIMD4(2.5, 0.25, 3, 1))

        let fixture = try PostProcessFixture()
        let post = try PostProcessGraph { p in p.vector(p.time, p.deltaTime, p.frame, p.constant(Float(1))) }
        let postUniforms = try ShaderUniforms(graph: post)
        try postUniforms.setFrame(frame)
        let postPass = try PostProcessPass(graph: post, device: fixture.device, uniforms: postUniforms)
        let source = try fixture.color(width: 4, height: 4) { _, _ in SIMD4(0, 0, 0, 1) }
        let destination = try fixture.color(width: 4, height: 4)
        try fixture.run { try postPass.encode(source: source, destination: destination, commandBuffer: $0) }
        #expect(try fixture.read(destination).allSatisfy { $0 == SIMD4(2.5, 0.25, 3, 1) })

        // Vertex deformation reads the clock from the material's live values, which the
        // material creates because the graph reads the clock without declaring uniforms.
        let wave = try SurfaceShaderGraph { s in .init(positionOffset: s.displacement(s.time)) }
        let waveMaterial = try PhysicallyBasedMaterial(surfaceGraph: wave)
        let clock = try #require(waveMaterial.surfaceUniforms)
        try clock.setFrame(frame)
        let plane = try MeshGeometry.plane()
        let deformed = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: plane, material: .physicallyBased(waveMaterial)
        ).deformedGeometry(queue: queue)
        for (original, moved) in zip(plane.vertices, deformed.vertices) {
            #expect(simd_distance(moved.position, original.position + original.normal * 2.5) < 1e-6)
        }

        // Raster PBR and the path tracer read the same clock.
        let glow = try SurfaceShaderGraph { s in
            .init(emission: s.vector(s.time * 0.2, s.deltaTime, s.frame * 0.1))
        }
        let glowMaterial = try PhysicallyBasedMaterial(lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, surfaceGraph: glow)
        try #require(glowMaterial.surfaceUniforms).setFrame(frame)
        let renderer = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 3, height: 3),
            material: .physicallyBased(glowMaterial)
        )
        let nodes = [SceneNode(transform: .init(translation: SIMD3(0, 0, -1.5)), mesh: renderer)]
        let (raster, path) = try renderBoth(nodes)
        let expected = SIMD4<Float>(0.5, 0.25, 0.3, 1)
        #expect(close(raster(32), expected), "raster \(raster(32))")
        #expect(close(path(32), expected), "path \(path(32))")
    }

    @Test("Clock changes rerun deformation and restart path accumulation only where a graph reads the clock")
    func clockConsumers() throws {
        // One live block shared by a clock-driven emission and a static displacement.
        let shared = try SurfaceShaderGraph { s in
            .init(emission: s.vector(s.time * 0.1, s.uniform("Gain", Float(0.25)), s.constant(Float(0))),
                  positionOffset: s.objectNormal * 0.01)
        }
        let animated = try SurfaceShaderGraph { s in
            .init(emission: s.vector(s.constant(Float(0.2)), s.uniform("Gain", Float(0.25)), s.constant(Float(0))),
                  positionOffset: s.objectNormal * (s.time * 0.01))
        }
        let uniforms = try ShaderUniforms(graph: shared)
        func renderer(_ graph: SurfaceShaderGraph) throws -> MeshRenderer {
            try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.6, height: 0.6),
                material: .physicallyBased(try .init(
                    lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, surfaceGraph: graph, surfaceUniforms: uniforms
                ))
            )
        }
        let still = try renderer(shared)
        let moving = try renderer(animated)
        let nodes = [SceneNode(transform: .init(translation: SIMD3(-0.4, 0, -1.5)), mesh: still),
                     SceneNode(transform: .init(translation: SIMD3(0.4, 0, -1.5)), mesh: moving)]
        try render(nodes)
        #expect(still.deformationEncodeCount == 1 && moving.deformationEncodeCount == 1)
        try uniforms.advanceFrame(by: 1 / 60)
        try render(nodes)
        #expect(still.deformationEncodeCount == 1, "a static offset ignores the clock")
        #expect(moving.deformationEncodeCount == 2, "an offset that reads the clock deforms again")
        try uniforms.set("Gain", to: 0.5)
        try render(nodes)
        #expect(still.deformationEncodeCount == 2 && moving.deformationEncodeCount == 3, "declared values deform both")

        // The path tracer keeps accumulating across clock ticks unless a traced graph reads the clock.
        func accumulates(_ graph: SurfaceShaderGraph) throws -> Bool {
            let live = try ShaderUniforms(graph: graph)
            let mesh = try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.6, height: 0.6),
                material: .physicallyBased(try .init(
                    lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, surfaceGraph: graph, surfaceUniforms: live
                ))
            )
            let scene = try PathTracingScene(queue: queue, graph: SceneGraph(
                roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -1.5)), mesh: mesh)], camera: camera,
                cameraTransform: cameraTransform
            ))
            let tracer = try PathTracer(queue: queue, scene: scene)
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 8, height: 8, mipmapped: false)
            descriptor.storageMode = .private
            descriptor.usage = .shaderWrite
            let destination = try #require(device.makeTexture(descriptor: descriptor))
            let accumulation = try tracer.makeAccumulation(width: 8, height: 8)
            let first = try #require(queue.makeCommandBuffer())
            let history = try tracer.encode(camera: camera, cameraTransform: cameraTransform, destination: destination,
                                            accumulation: accumulation, commandBuffer: first)
            first.commit()
            first.waitUntilCompleted()
            try live.advanceFrame(by: 0.5)
            let second = try #require(queue.makeCommandBuffer())
            do {
                _ = try tracer.encode(camera: camera, cameraTransform: cameraTransform, destination: destination,
                                      accumulation: accumulation, previous: history, commandBuffer: second)
            } catch PathTracer.Failure.surfaceUniformsChanged {
                return false
            }
            second.commit()
            second.waitUntilCompleted()
            return true
        }
        #expect(try accumulates(try SurfaceShaderGraph { s in .init(emission: s.vector(s.uv0, s.uniform("Gain", Float(0.5)))) }))
        #expect(!(try accumulates(try SurfaceShaderGraph { s in .init(emission: s.vector(s.uv0, s.time)) })))
        // A clock read only by the vertex offset does not change what the tracer shades.
        #expect(try accumulates(try SurfaceShaderGraph { s in
            .init(emission: s.vector(s.uv0, s.uniform("Gain", Float(0.5))), positionOffset: s.objectNormal * s.time)
        }))
    }

    @Test("A particle simulation that reads the clock gets live values it can advance, even when created without any")
    func particleClock() throws {
        let simulation = try BufferShaderGraph(fieldCount: 2) { b in
            let state = b.elementField(0)
            let velocity = b.elementField(1)
            return [0: b.vector(state.xyz + velocity.xyz * b.deltaTime, state.w + b.deltaTime)]
        }
        let system = try ShaderParticleSystem(
            device: device, capacity: 1, simulation: simulation, initial: [SIMD4(0, 1, 2, 0), SIMD4(2, 0, -4, 3)]
        )
        let clock = try #require(system.uniforms)
        try clock.advanceFrame(by: 0.5)
        let step = try #require(queue.makeCommandBuffer())
        try system.encodeUpdate(commandBuffer: step)
        step.commit()
        step.waitUntilCompleted()
        #expect(system.fields()[0] == SIMD4(1, 1, 0, 0.5))
        // A simulation without the clock and without declarations keeps nil values.
        let plain = try BufferShaderGraph(fieldCount: 2) { b in [0: b.elementField(0) + b.elementField(1)] }
        #expect(try ShaderParticleSystem(device: device, capacity: 1, simulation: plain, initial: [.zero, .zero]).uniforms == nil)
    }

    // MARK: - Camera and model

    @Test("Camera, view, projection and model inputs agree between raster PBR and the path tracer")
    func cameraAndModel() throws {
        // Three planes two units in front of a camera at z = 0.5.
        let cameraGraph = try SurfaceShaderGraph { s in
            .init(emission: s.vector(s.cameraPosition.z, s.cameraNear * 10, s.cameraFar / 40))
        }
        let viewGraph = try SurfaceShaderGraph { s in
            let view = s.viewMatrix.transformPoint(s.worldPosition)
            let clip = s.projectionMatrix * s.vector(view, s.constant(Float(1)))
            return .init(emission: s.vector(-view.z / 4, s.projectionMatrix.column(1).y / 4, clip.w / 4))
        }
        let modelGraph = try SurfaceShaderGraph { s in
            let normal = s.normalMatrix * s.constant(SIMD3<Float>(0, 0, 1))
            return .init(emission: s.vector(
                s.modelMatrix.column(3).x * 0.5 + 0.5, normal.z, s.modelMatrix.determinant() / 4
            ))
        }
        let nodes = try [(cameraGraph, Float(-0.7)), (viewGraph, 0), (modelGraph, 0.7)].map { graph, x in
            SceneNode(
                transform: .init(
                    translation: SIMD3(x, 0, -1.5), rotation: simd_quatf(angle: 0.3, axis: SIMD3(0, 0, 1)),
                    scale: SIMD3(1, 1, 2)
                ),
                mesh: try emissive(graph)
            )
        }
        let (raster, path) = try renderBoth(nodes)
        let expectations: [(Int, SIMD4<Float>)] = [
            (13, SIMD4(0.5, 1, 0.5, 1)),
            (32, SIMD4(0.5, Float(1 / tan(Double.pi / 6)) / 4, 0.5, 1)),
            (51, SIMD4(0.85, 0.5, 0.5, 1)),
        ]
        for (x, expected) in expectations {
            #expect(close(raster(x), expected), "raster x \(x): \(raster(x)) vs \(expected)")
            #expect(close(path(x), expected), "path x \(x): \(path(x)) vs \(expected)")
        }
    }

    @Test("Raster graphs read screenUV, viewportSize and derivatives, which the path tracer and clip-only views reject")
    func screenSpace() throws {
        // A plane facing the camera two units away spans 4 tan(30°) world units over 65 pixels.
        let worldPerPixel = Float(4 * tan(Double.pi / 6)) / Float(Self.size)
        let screen = try SurfaceShaderGraph { s in
            .init(emission: s.vector(s.screenUV.x, s.screenUV.y, s.worldPosition.x.fwidth() / worldPerPixel))
        }
        let pixels = try SurfaceShaderGraph { s in
            .init(emission: s.vector(s.viewportSize / 100, s.fragmentCoordinate.x / 100))
        }
        let screenNode = SceneNode(transform: .init(translation: SIMD3(0, 0, -1.5)), mesh: try emissive(screen))
        let pixelNode = SceneNode(transform: .init(translation: SIMD3(-0.7, 0, -1.5)), mesh: try emissive(pixels))
        try render([screenNode, pixelNode])
        let center = try readPixel(x: 32)
        let left = try readPixel(x: 13)
        #expect(close(center, SIMD4(32.5 / 65, 32.5 / 65, 1, 1)), "\(center)")
        #expect(close(left, SIMD4(0.65, 0.65, 0.135, 1)), "\(left)")

        let traced = SceneGraph(roots: [screenNode], camera: camera, cameraTransform: cameraTransform)
        let rejection = #expect(throws: PathTracingScene.Failure.self) { try PathTracingScene(queue: queue, graph: traced) }
        guard case .unsupported(let id, let feature) = rejection else {
            Issue.record("Unexpected failure \(String(describing: rejection))")
            return
        }
        #expect(id == screenNode.id && feature.contains("screen-space derivative"), "\(feature)")

        let coordinates = try SurfaceShaderGraph { s in
            .init(emission: s.vector(s.screenUV, s.viewportSize.x / 100))
        }
        let coordinateNode = SceneNode(transform: .init(translation: SIMD3(0, 0, -1.5)), mesh: try emissive(coordinates))
        let tracedCoordinates = try renderBoth([coordinateNode])
        let expectedUV = SIMD4<Float>(32.5 / 65, 32.5 / 65, 0.65, 1)
        #expect(close(tracedCoordinates.raster(32), expectedUV), "\(tracedCoordinates.raster(32))")
        #expect(abs(tracedCoordinates.path(32).x - expectedUV.x) < 0.02, "\(tracedCoordinates.path(32))")
        #expect(abs(tracedCoordinates.path(32).y - expectedUV.y) < 0.02)
        #expect(abs(tracedCoordinates.path(32).z - 0.65) < 0.02)

        // The path tracer has no single model matrix for the instances of an instanced node.
        let model = try SurfaceShaderGraph { s in .init(emission: s.modelMatrix.column(3).xyz) }
        let instanced = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(),
            material: .physicallyBased(try .init(lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, surfaceGraph: model)),
            supportsInstancing: true
        )
        let set = try InstanceSet(device: device, transforms: [matrix_identity_float4x4])
        let instancedGraph = SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -1.5)), mesh: instanced, instances: set)],
            camera: camera
        )
        #expect(throws: PathTracingScene.Failure.self) { try PathTracingScene(queue: queue, graph: instancedGraph) }

        // A caller-supplied clip matrix carries no separate view or projection.
        let view = try SurfaceShaderGraph { s in .init(emission: s.viewMatrix.column(3).xyz) }
        let viewNode = SceneNode(transform: .init(translation: SIMD3(0, 0, -1.5)), mesh: try emissive(view))
        let clip = try camera.projection(aspectRatio: 1) * simd_inverse(cameraTransform.matrix)
        let clipGraph = SceneGraph(roots: [viewNode], camera: camera, cameraTransform: cameraTransform)
        let commandBuffer = try #require(queue.makeCommandBuffer())
        #expect(throws: MeshRenderer.Failure.graphCameraInputs) {
            try clipGraph.encode(pass: pass, commandBuffer: commandBuffer, clipFromWorld: clip, cameraPosition: SIMD3(0, 0, 0.5))
        }
        // Camera position and screen inputs still work with a clip matrix.
        let positionGraph = SceneGraph(roots: [screenNode], camera: camera, cameraTransform: cameraTransform)
        try positionGraph.encode(pass: pass, commandBuffer: commandBuffer, clipFromWorld: clip, cameraPosition: SIMD3(0, 0, 0.5))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        let clipped = try readPixel(x: 32)
        #expect(close(clipped, SIMD4(32.5 / 65, 32.5 / 65, 1, 1)), "\(clipped)")
    }

    @Test("Branches, loops and noise compile into raster PBR and the path tracer and agree with ShaderNoise")
    func controlFlowInRenderers() throws {
        let graph = try SurfaceShaderGraph { s in
            let x = s.parameters.x
            let octaves = s.loop(count: 5, initial: s.constant(Float(0))) { i, total in
                total + s.noise(s.vector(x + i, s.constant(Float(0.5)), s.constant(Float(0.25))), kind: .value) / 5
            }
            let chosen = s.branch(x.greaterThan(s.constant(0.5))) {
                s.fractalNoise(s.vector(x, x, x) * 3, octaves: 4)
            } else: {
                x
            }
            return .init(emission: s.vector(octaves, chosen, s.oscillator(.triangle, x)))
        }
        let parameters: [Float] = [0.3, 0.8]
        let nodes = try zip(parameters, [Float(-0.5), 0.5]).map { value, offset in
            SceneNode(transform: .init(translation: SIMD3(offset, 0, -1.5)), mesh: try emissive(graph),
                      materialParameters: SIMD4(value, 0, 0, 1))
        }
        let (raster, path) = try renderBoth(nodes)
        for (value, x) in zip(parameters, [Int(32 - 0.5 / 1.1547 * 32), Int(32 + 0.5 / 1.1547 * 32)]) {
            let octaves = (0..<5).reduce(Float(0)) { $0 + ShaderNoise.value(SIMD3(value + Float($1), 0.5, 0.25)) / 5 }
            let chosen = value > 0.5 ? ShaderNoise.fractal(SIMD3(value, value, value) * 3, octaves: 4) : value
            let triangle = abs(ShaderPatternReference.fract(value + 0.5) * 2 - 1)
            let expected = SIMD4(octaves, chosen, triangle, 1)
            #expect(close(raster(x), expected), "raster \(value): \(raster(x)) vs \(expected)")
            #expect(close(path(x), expected), "path \(value): \(path(x)) vs \(expected)")
        }
    }

    @Test("Archived procedural surfaces compile together in raster PBR and the path tracer")
    func proceduralSurfaces() throws {
        let nodes = try [false, true].map { swap in
            let authored = try SurfaceShaderGraph { s in
                let p = s.parameters
                let position = s.vector(p.x, p.y, p.z)
                let curl = s.curlNoise(position).x * 0.1
                let simplex = s.simplexNoise(position)
                let distance = s.animatedVoronoi(position, time: p.w)
                return .init(emission: swap ? s.vector(distance, curl, simplex) : s.vector(simplex, curl, distance))
            }
            let graph = try JSONDecoder().decode(SurfaceShaderGraph.self, from: JSONEncoder().encode(authored))
            #expect(graph == authored)
            return SceneNode(
                transform: .init(translation: SIMD3(swap ? 0.5 : -0.5, 0, -1.5)), mesh: try emissive(graph),
                materialParameters: SIMD4(0.125, 0.25, 0.5, swap ? 1.75 : 0)
            )
        }
        let (raster, path) = try renderBoth(nodes)
        let left = SIMD4<Float>(0.316197455, 0.280470729, 0.203125, 1)
        let right = SIMD4<Float>(0.714444697, 0.280470729, 0.316197455, 1)
        #expect(close(raster(18), left), "\(raster(18))")
        #expect(close(path(18), left), "\(path(18))")
        #expect(close(raster(46), right), "\(raster(46))")
        #expect(close(path(46), right), "\(path(46))")
    }

    // MARK: - Helpers

    private func emissive(_ graph: SurfaceShaderGraph) throws -> MeshRenderer {
        try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.6, height: 0.6),
            material: .physicallyBased(try .init(lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, surfaceGraph: graph))
        )
    }

    private func render(_ roots: [SceneNode]) throws {
        let graph = SceneGraph(roots: roots, camera: camera, cameraTransform: cameraTransform)
        let buffer = try #require(queue.makeCommandBuffer())
        try graph.encode(pass: pass, commandBuffer: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)
    }

    /// Renders with raster PBR and four path-traced samples per pixel; returns
    /// center-row pixel readers for both images.
    private func renderBoth(_ roots: [SceneNode]) throws -> (raster: (Int) -> SIMD4<Float>, path: (Int) -> SIMD4<Float>) {
        try render(roots)
        let rasterRow = try row(color)
        let graph = SceneGraph(roots: roots, camera: camera, cameraTransform: cameraTransform)
        let scene = try PathTracingScene(queue: queue, graph: graph)
        let tracer = try PathTracer(queue: queue, scene: scene)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: Self.size, height: Self.size, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderWrite
        let traced = try #require(device.makeTexture(descriptor: descriptor))
        let accumulation = try tracer.makeAccumulation(width: Self.size, height: Self.size)
        let buffer = try #require(queue.makeCommandBuffer())
        _ = try tracer.encode(camera: camera, cameraTransform: cameraTransform, destination: traced,
                              accumulation: accumulation, commandBuffer: buffer, samplesPerPixel: 4)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)
        let pathRow = try row(traced)
        return ({ rasterRow[$0] }, { pathRow[$0] })
    }

    private func row(_ texture: any MTLTexture) throws -> [SIMD4<Float>] {
        var halves = [Float16](repeating: 0, count: Self.size * 4)
        try halves.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: Self.size * 8,
                from: MTLRegionMake2D(0, Self.size / 2, Self.size, 1), mipmapLevel: 0
            )
        }
        return (0..<Self.size).map { SIMD4(Float(halves[$0 * 4]), Float(halves[$0 * 4 + 1]),
                                           Float(halves[$0 * 4 + 2]), Float(halves[$0 * 4 + 3])) }
    }

    private func readPixel(x: Int) throws -> SIMD4<Float> { try row(color)[x] }

    /// Half-float output resolves about 1e-3 near 1; allow 2e-3.
    private func close(_ a: SIMD4<Float>, _ b: SIMD4<Float>) -> Bool {
        simd_reduce_max(simd_abs(a - b)) <= 2e-3 + 1e-3 * simd_reduce_max(simd_abs(b))
    }
}

/// CPU helpers shared by the graph input tests.
enum ShaderPatternReference {
    static func fract(_ x: Float) -> Float { min(x - x.rounded(.down), 0x1.fffffep-1) }
}
