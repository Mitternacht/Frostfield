import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ShaderGraphRenderingTests {
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let camera: PerspectiveCamera
    private let color: any MTLTexture
    private let pass: MTLRenderPassDescriptor
    private static let size = 65
    // Four vertical stripes: red, green, blue, white. V does not change the color.
    private static let stripes: [SIMD4<Float>] = [SIMD4(1, 0, 0, 1), SIMD4(0, 1, 0, 1), SIMD4(0, 0, 1, 1), SIMD4(1, 1, 1, 1)]

    init() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        self.device = device
        queue = try #require(device.makeCommandQueue())
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20)
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: Self.size, height: Self.size, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = [.renderTarget, .shaderWrite]
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

    @Test("Uniforms and textures drive both renderers, and edits reuse compiled shaders")
    func uniformsAndTexturesAcrossRenderers() throws {
        let marker = "uniform-reuse-probe-4c1e"
        ShaderLibraryCache.countCompilations(containing: marker)
        let graph = try SurfaceShaderGraph { s in
            let tint = s.uniform("Tint", SIMD3<Float>(1, 1, 1))
            let stripes = s.texture("Stripes", content: .data, addressing: .repeat, filtering: .nearest)
            return .init(emission: (stripes.sample(s.uv0).rgb * tint).named(marker))
        }
        let material = try PhysicallyBasedMaterial(
            lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, surfaceGraph: graph,
            surfaceTextures: ["Stripes": try stripeTexture()]
        )
        let uniforms = try #require(material.surfaceUniforms)
        let plane = try MeshGeometry.plane(width: 2, height: 4)
        let first = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: plane,
                                     material: .physicallyBased(material))
        // A second renderer with the same graph shares the compiled library.
        _ = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: plane,
                             material: .physicallyBased(material))
        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: first)
        let graphScene = SceneGraph(roots: [node], camera: camera)
        let tracer = try PathTracer(queue: queue, scene: PathTracingScene(queue: queue, graph: graphScene))
        let accumulation = try tracer.makeAccumulation(width: Self.size, height: Self.size)
        let compiled = ShaderLibraryCache.compilationCount(containing: marker)
        #expect(compiled == 2, "one raster and one path library")

        // u≈0.21 is the red stripe and u≈0.79 the white stripe.
        try render(graphScene)
        var history = try trace(tracer, accumulation: accumulation, previous: nil)
        for (x, expected) in [(16, SIMD3<Float>(1, 0, 0)), (48, SIMD3<Float>(1, 1, 1))] {
            #expect(near(try readPixel(x: x), expected), "raster \(x)")
            #expect(near(try readPixel(from: history.texture, x: x), expected), "path \(x)")
        }

        try uniforms.set("Tint", to: SIMD3(0.5, 0.25, 1))
        try render(graphScene)
        #expect(near(try readPixel(x: 48), SIMD3(0.5, 0.25, 1)))
        #expect(throws: PathTracer.Failure.surfaceUniformsChanged) { try trace(tracer, accumulation: accumulation, previous: history.history) }
        history = try trace(tracer, accumulation: accumulation, previous: nil)
        #expect(near(try readPixel(from: history.texture, x: 16), SIMD3(0.5, 0, 0)))
        #expect(near(try readPixel(from: history.texture, x: 48), SIMD3(0.5, 0.25, 1)))
        // Unchanged values keep accumulating.
        _ = try trace(tracer, accumulation: accumulation, previous: history.history)

        // Editing values and re-preparing the geometry snapshot never recompile.
        _ = try PathTracingScene(queue: queue, graph: graphScene)
        #expect(ShaderLibraryCache.compilationCount(containing: marker) == compiled)
    }

    @Test("Graph normals and occlusion change raster lighting in the UV0 tangent frame")
    func graphNormalAndOcclusion() throws {
        func brightness(tilt: Float?, occlusion: Float? = nil, ambient: Float = 0) throws -> Float {
            let graph = try SurfaceShaderGraph { s in
                .init(
                    baseColor: s.constant(SIMD3<Float>(0.8, 0.8, 0.8)),
                    roughness: s.constant(Float(1)),
                    normal: tilt.map { s.constant(SIMD3<Float>($0, 0, 0.8)) },
                    occlusion: occlusion.map { s.constant($0) }
                )
            }
            let material = try PhysicallyBasedMaterial(
                directionToLight: SIMD3(1, 0, 1), lightIntensity: ambient == 0 ? 1 : 0, ambientIntensity: ambient,
                alphaMode: .opaque, surfaceGraph: graph
            )
            let mesh = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float,
                                        geometry: .plane(width: 2, height: 4), material: .physicallyBased(material))
            try render(SceneGraph(roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)],
                                  camera: camera))
            return try readPixel(x: Self.size / 2).x
        }
        let flat = try brightness(tilt: nil)
        let toward = try brightness(tilt: 0.6)
        let away = try brightness(tilt: -0.6)
        // N·L rises from 0.71 to 0.99 toward the light and falls to 0.14 away from it.
        #expect(toward > flat * 1.2, "\(toward) vs \(flat)")
        #expect(away < flat * 0.5, "\(away) vs \(flat)")

        let open = try brightness(tilt: nil, ambient: 1)
        let occluded = try brightness(tilt: nil, occlusion: 0.25, ambient: 1)
        #expect(abs(occluded - open * 0.25) < 0.01, "\(occluded) vs \(open)")
    }

    @Test("Path-traced graph normals keep raster's tangent frame on back faces")
    func backFaceGraphNormal() throws {
        // A unit tangent-space normal tilted along both U and V.
        let graph = try SurfaceShaderGraph { s in
            .init(baseColor: s.constant(SIMD3<Float>(0.8, 0.8, 0.8)), normal: s.constant(SIMD3<Float>(0.48, 0.36, 0.8)))
        }
        let material = try PhysicallyBasedMaterial(alphaMode: .opaque, isDoubleSided: true, surfaceGraph: graph)
        let mesh = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float,
                                    geometry: .plane(width: 2, height: 4), material: .physicallyBased(material))
        func tracedNormal(rotation: simd_quatf) throws -> SIMD3<Float> {
            let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -2), rotation: rotation), mesh: mesh)
            let tracer = try PathTracer(queue: queue, scene: PathTracingScene(queue: queue, graph: SceneGraph(roots: [node], camera: camera)))
            let accumulation = try tracer.makeAccumulation(width: Self.size, height: Self.size)
            func output() throws -> any MTLTexture {
                let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .rgba16Float, width: Self.size, height: Self.size, mipmapped: false
                )
                descriptor.storageMode = .shared
                descriptor.usage = [.shaderWrite, .shaderRead]
                return try #require(device.makeTexture(descriptor: descriptor))
            }
            let guides = PathTracer.Guides(normalDepth: try output(), albedoRoughness: try output())
            let buffer = try #require(queue.makeCommandBuffer())
            _ = try tracer.encode(camera: camera, cameraTransform: .init(), destination: try output(),
                                  accumulation: accumulation, guides: guides, commandBuffer: buffer,
                                  settings: .init(maxBounces: 1, seed: 1))
            buffer.commit()
            buffer.waitUntilCompleted()
            try #require(buffer.status == .completed)
            let pixel = try readPixel(from: guides.normalDepth, x: Self.size / 2)
            return SIMD3(pixel.x, pixel.y, pixel.z)
        }
        // Front: T = +X, B = -Y (V runs down the plane), N = +Z.
        let front = try tracedNormal(rotation: simd_quatf(angle: 0, axis: SIMD3(0, 1, 0)))
        #expect(simd_distance(front, SIMD3(0.48, -0.36, 0.8)) < 0.01, "\(front)")
        // Back, turned half a turn about Y: T = -X, B = -Y, N = -Z. Raster and glTF negate
        // T·x + B·y + N·z, giving (0.48, 0.36, 0.8); mirroring would give (-0.48, -0.36, 0.8).
        let back = try tracedNormal(rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0)))
        #expect(simd_distance(back, SIMD3(0.48, 0.36, 0.8)) < 0.01, "\(back)")
    }

    @Test("Path-traced clamp and mirror addressing stay at the edge at coarse mip levels")
    func coarseLevelAddressing() throws {
        // Left half black, right half white. Sampling at u = 0 must stay black for clamp and
        // mirror addressing, even where the bilinear footprint of a coarse level spans the edge.
        let width = 256
        let ramp = try makeTexture(format: .rgba8Unorm, width: width)
        var texels: [UInt8] = []
        for _ in 0..<width { for x in 0..<width { texels += x < width / 2 ? [0, 0, 0, 255] : [255, 255, 255, 255] } }
        texels.withUnsafeBytes { bytes in
            ramp.replace(region: MTLRegionMake2D(0, 0, width, width), mipmapLevel: 0,
                         withBytes: bytes.baseAddress!, bytesPerRow: width * 4)
        }
        func traced(_ addressing: ShaderTextureInput.Addressing) throws -> Float {
            let graph = try SurfaceShaderGraph { s in
                let map = s.texture("Ramp", content: .data, addressing: addressing, filtering: .linear)
                return .init(emission: map.sample(s.constant(SIMD2<Float>(0, 0.5))).rgb)
            }
            let material = try PhysicallyBasedMaterial(
                lightIntensity: 0, ambientIntensity: 0, alphaMode: .opaque, surfaceGraph: graph,
                surfaceTextures: ["Ramp": ramp]
            )
            let mesh = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float,
                                        geometry: .plane(width: 2, height: 4), material: .physicallyBased(material))
            // Two units at two units' distance: about four baked texels per pixel, mip level two.
            let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: mesh)
            let tracer = try PathTracer(queue: queue, scene: PathTracingScene(queue: queue, graph: SceneGraph(roots: [node], camera: camera)))
            let result = try trace(tracer, accumulation: try tracer.makeAccumulation(width: Self.size, height: Self.size), previous: nil)
            return try readPixel(from: result.texture, x: Self.size / 2).x
        }
        #expect(try traced(.clampToEdge) < 0.01)
        #expect(try traced(.mirroredRepeat) < 0.01)
        // Repeat addressing wraps by definition: u = 0 lies between the black and white edges.
        #expect(abs(try traced(.repeat) - 0.5) < 0.1)
    }

    @Test("Materials validate graph texture bindings and shared uniforms")
    func bindingValidation() throws {
        let graph = try SurfaceShaderGraph { s in
            // This fixture validates bindings, independent of 32-bit float filtering support.
            .init(baseColor: s.texture("Albedo", filtering: .nearest).sample(s.uv0).rgb * s.uniform("Gain", Float(1)))
        }
        let texture = try stripeTexture()
        #expect(throws: PhysicallyBasedMaterial.Failure.surfaceTexture("Albedo")) {
            try PhysicallyBasedMaterial(surfaceGraph: graph)
        }
        #expect(throws: PhysicallyBasedMaterial.Failure.surfaceTexture("Extra")) {
            try PhysicallyBasedMaterial(surfaceGraph: graph, surfaceTextures: ["Albedo": texture, "Extra": texture])
        }
        let wrong = try ShaderUniforms([ShaderUniform("Gain", SIMD2<Float>(1, 1))])
        #expect(throws: PhysicallyBasedMaterial.Failure.surfaceUniforms) {
            try PhysicallyBasedMaterial(surfaceGraph: graph, surfaceUniforms: wrong, surfaceTextures: ["Albedo": texture])
        }
        // Two materials may share one live uniform set.
        let shared = try ShaderUniforms(graph: graph)
        let a = try PhysicallyBasedMaterial(surfaceGraph: graph, surfaceUniforms: shared, surfaceTextures: ["Albedo": texture])
        let b = try PhysicallyBasedMaterial(surfaceGraph: graph, surfaceUniforms: shared, surfaceTextures: ["Albedo": texture])
        #expect(a.surfaceUniforms === b.surfaceUniforms)

        let dataGraph = try SurfaceShaderGraph { s in
            .init(roughness: s.texture("Mask", content: .data).sample(s.uv0).x)
        }
        let srgb = try makeTexture(format: .rgba8Unorm_srgb, width: 4)
        #expect(throws: PhysicallyBasedMaterial.Failure.surfaceTexture("Mask")) {
            try PhysicallyBasedMaterial(surfaceGraph: dataGraph, surfaceTextures: ["Mask": srgb])
        }
    }

    @Test("Linear filtering of 32-bit float graph textures follows the GPU's support")
    func floatFiltering() throws {
        func material(_ filtering: ShaderTextureInput.Filtering) throws -> PhysicallyBasedMaterial {
            let graph = try SurfaceShaderGraph { s in
                .init(emission: s.texture("Height", content: .data, filtering: filtering).sample(s.uv0).rgb)
            }
            return try PhysicallyBasedMaterial(alphaMode: .opaque, surfaceGraph: graph,
                                               surfaceTextures: ["Height": try makeTexture(format: .rgba32Float, width: 4)])
        }
        _ = try material(.nearest)
        if device.supports32BitFloatFiltering {
            _ = try material(.linear)
        } else {
            #expect(throws: PhysicallyBasedMaterial.Failure.surfaceTexture("Height")) { try material(.linear) }
        }
    }

    private func stripeTexture() throws -> any MTLTexture {
        let width = 256
        let texture = try makeTexture(format: .rgba32Float, width: width)
        var texels: [SIMD4<Float>] = []
        for _ in 0..<width {
            for x in 0..<width { texels.append(Self.stripes[x * 4 / width]) }
        }
        texels.withUnsafeBytes { bytes in
            texture.replace(region: MTLRegionMake2D(0, 0, width, width), mipmapLevel: 0,
                            withBytes: bytes.baseAddress!, bytesPerRow: width * 16)
        }
        return texture
    }

    private func makeTexture(format: MTLPixelFormat, width: Int) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: width, mipmapped: false)
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        return try #require(device.makeTexture(descriptor: descriptor))
    }

    private func render(_ graph: SceneGraph) throws {
        let buffer = try #require(queue.makeCommandBuffer())
        try graph.encode(pass: pass, commandBuffer: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)
    }

    private func trace(
        _ tracer: PathTracer, accumulation: PathTracer.Accumulation, previous: PathTracer.History?
    ) throws -> (history: PathTracer.History, texture: any MTLTexture) {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: Self.size, height: Self.size, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderWrite
        let traced = try #require(device.makeTexture(descriptor: descriptor))
        let buffer = try #require(queue.makeCommandBuffer())
        let history = try tracer.encode(camera: camera, cameraTransform: .init(), destination: traced,
                                        accumulation: accumulation, previous: previous, commandBuffer: buffer,
                                        samplesPerPixel: 2)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)
        return (history, traced)
    }

    private func readPixel(from texture: (any MTLTexture)? = nil, x: Int) throws -> SIMD4<Float> {
        var pixel = SIMD4<Float16>.zero
        try withUnsafeMutableBytes(of: &pixel) { bytes in
            (texture ?? color).getBytes(
                try #require(bytes.baseAddress), bytesPerRow: 8,
                from: MTLRegionMake2D(x, Self.size / 2, 1, 1), mipmapLevel: 0
            )
        }
        return SIMD4<Float>(pixel)
    }

    private func near(_ pixel: SIMD4<Float>, _ expected: SIMD3<Float>) -> Bool {
        (0..<3).allSatisfy { abs(pixel[$0] - expected[$0]) < 0.004 }
    }
}
