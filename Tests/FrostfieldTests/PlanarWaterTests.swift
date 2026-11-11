import Foundation
import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PlanarWaterTests {
    private let gpu: GeometryRenderFixture

    init() throws { gpu = try GeometryRenderFixture(side: 64) }

    @Test("Flowing water combines two real LDR or HDR captures using Fresnel and linear tint",
          arguments: [false, true])
    func capturedReflectionAndRefraction(hdr: Bool) throws {
        let reflection = try captureTarget()
        let refraction = try captureTarget()
        let flat = try texture([SIMD4(0.5, 0.5, 1, 1)])
        var options = PlanarWater.FlowOptions()
        options.reflectivity = 0.25
        options.tint = SIMD3(0.5, 0.75, 0.25)
        let material = try PlanarWater.flowing(
            reflection: reflection.color, refraction: refraction.color,
            firstNormal: flat, secondNormal: flat, options: options
        )
        let water = try surface(material)
        let reflectionColor = hdr ? SIMD3<Float>(8, 2, 4) : SIMD3<Float>(0.75, 0.0625, 0.25)
        let refractionColor = hdr ? SIMD3<Float>(4, 8, 2) : SIMD3<Float>(0.125, 0.5, 0.75)
        let above = try marker(z: -2, color: reflectionColor, towardMirror: true, hdr: hdr)
        let below = try marker(z: -6, color: refractionColor, towardMirror: false, hdr: hdr)
        let graph = SceneGraph(roots: [water, above, below], camera: gpu.camera)
        let camera = try camera()
        let plane = try Plane3D(normal: SIMD3(0, 0, 1), through: SIMD3(0, 0, -4))
        let mirror = PlanarReflection(plane: plane, reflector: water.id)
        let refractor = try PlanarRefraction(plane: plane, refractor: water.id)
        let output = try gpu.makeTarget()
        let command = try #require(gpu.queue.makeCommandBuffer())

        try mirror.encode(graph, camera: camera, color: reflection.color, depth: reflection.depth, commandBuffer: command)
        try refractor.encode(graph, camera: camera, color: refraction.color, depth: refraction.depth, commandBuffer: command)
        try encode(water, camera: camera, directionToCamera: SIMD3(0, 0, 1), to: output, command: command)
        try gpu.finish(command)

        #expect(isNearly(try center(of: reflection.color), SIMD4(reflectionColor, 1), tolerance: 0.003))
        #expect(isNearly(try center(of: refraction.color), SIMD4(refractionColor, 1), tolerance: 0.003))
        let expected = (0.25 * reflectionColor + 0.75 * refractionColor) * options.tint
        #expect(isNearly(try center(of: output.color), SIMD4(expected, 1), tolerance: 0.004))
    }

    @Test("Viewport refraction rejects a real nearer occluder and keeps a displaced background sample",
          arguments: [MeshDepthMode.forward, .reversed, .logarithmic])
    func safeViewportRefraction(mode: MeshDepthMode) throws {
        let black = try texture([SIMD4(0, 0, 0, 1)])
        let normal = try texture([SIMD4(0.875, 0.5, 1, 1)])
        var options = PlanarWater.FlowOptions()
        options.distortion = 0.20
        options.reflectivity = 0
        let material = try PlanarWater.flowingViewport(
            reflection: black, firstNormal: normal, secondNormal: normal, options: options
        )
        let water = try surface(material, depthMode: mode)
        let green = SIMD3<Float>(0.125, 0.875, 0.125)
        let blue = SIMD3<Float>(0.125, 0.25, 0.875)
        let red = SIMD3<Float>(0.875, 0.125, 0.125)
        func marker(_ center: SIMD3<Float>, size: Float, color: SIMD3<Float>) throws -> SceneNode {
            let material = try PhysicallyBasedMaterial(baseColorFactor: SIMD4(color, 1), shadingModel: .unlit)
            return SceneNode(
                transform: .init(translation: center),
                mesh: try gpu.unlit(MeshGeometry.plane(width: size, height: size), material: .physicallyBased(material), depthMode: mode)
            )
        }
        let backdrop = try marker(SIMD3(0, 0, -7), size: 14, color: green)
        let fartherPatch = try marker(SIMD3(-0.9375, 0, -6), size: 0.55, color: blue)
        let occluder = try marker(SIMD3(0.75, 0, -3), size: 0.45, color: red)
        var scene = SceneGraph(roots: [water, backdrop, fartherPatch, occluder], camera: gpu.camera)
        let snapshot = try captureTarget()
        scene.viewportSnapshot = try SceneViewportSnapshot(color: snapshot.color, depth: snapshot.depth, source: .opaquePass)
        let output = try gpu.makeTarget()
        let camera = try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: gpu.camera.projection(aspectRatio: 1, depthMode: mode),
            near: gpu.camera.near, far: gpu.camera.far
        )
        let viewport = mode == .logarithmic ? nil : MTLViewport(
            originX: 8, originY: 4, width: 48, height: 48, znear: 0.15, zfar: 0.85
        )
        let command = try #require(gpu.queue.makeCommandBuffer())
        try scene.encode(
            pass: gpu.pass(color: output.color, depth: output.depth, depthMode: mode), commandBuffer: command,
            clipFromWorld: camera.projectionMatrix, cameraPosition: .zero, viewport: viewport,
            depthMode: mode, graphCamera: camera
        )
        try gpu.finish(command)

        let sourcePixels = try gpu.pixels(of: snapshot.color)
        let resultPixels = try gpu.pixels(of: output.color)
        let origin = Int(viewport?.originX ?? 0)
        let size = Int(viewport?.width ?? Double(gpu.side))
        let row = Int(viewport?.originY ?? 0) + size / 2
        func pixel(_ values: [SIMD4<Float>], u: Float) -> SIMD4<Float> {
            gpu.pixel(values, x: origin + Int(u * Float(size)), y: row)
        }
        // The same positive normal offset reaches two different depths.
        #expect(isNearly(pixel(sourcePixels, u: 0.50), SIMD4(green, 1)))
        #expect(isNearly(pixel(sourcePixels, u: 0.62), SIMD4(red, 1)))
        #expect(isNearly(pixel(sourcePixels, u: 0.30), SIMD4(green, 1)))
        #expect(isNearly(pixel(sourcePixels, u: 0.42), SIMD4(blue, 1)))
        #expect(isNearly(pixel(resultPixels, u: 0.50), SIMD4(green, 1)))
        #expect(isNearly(pixel(resultPixels, u: 0.30), SIMD4(blue, 1)))
        #expect(isNearly(pixel(resultPixels, u: 0.62), SIMD4(red, 1)))

        if mode == .forward {
            let archive = try SceneArchive(capturing: scene, commandQueue: gpu.queue)
            var json = try #require(try JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
            #expect(json["version"] as? Int == 28)
            let restored = try archive.load(device: gpu.device, commandQueue: gpu.queue).graph
            guard case .physicallyBased(let restoredMaterial) = restored.roots.first?.mesh?.material else {
                Issue.record("Expected the archived viewport water material.")
                return
            }
            #expect(restoredMaterial.surfaceGraph == material.surfaceGraph)
            #expect(restored.viewportSnapshot == nil)
            json["version"] = 27
            #expect(throws: SceneArchive.Failure.decoding("depth-safe viewport coordinates require archive version 28")) {
                try SceneArchive(manifest: JSONSerialization.data(withJSONObject: json), binary: archive.binary)
            }
        }
    }

    @Test("Flowing-water reflectance rises at a grazing view angle")
    func grazingFresnel() throws {
        let reflection = try texture([SIMD4(1, 0, 0, 1)])
        let refraction = try texture([SIMD4(0, 0, 1, 1)])
        let flat = try texture([SIMD4(0.5, 0.5, 1, 1)])
        let material = try PlanarWater.flowing(
            reflection: reflection, refraction: refraction, firstNormal: flat, secondNormal: flat
        )
        let water = try surface(material)
        let facing = try render(water, camera: camera())
        let cosine: Float = 0.1
        let direction = SIMD3<Float>(sqrt(1 - cosine * cosine), 0, cosine)
        let rotation = simd_quatf(from: SIMD3(0, 0, 1), to: direction)
        let origin = water.transform.translation + direction * 5
        let camera = try camera(transform: .init(translation: origin, rotation: rotation))
        let grazing = try render(water, camera: camera, directionToCamera: direction)
        let expected = 0.02 + 0.98 * pow(1 - cosine, 5)
        #expect(isNearly(facing, SIMD4(0.02, 0, 0.98, 1), tolerance: 0.003))
        #expect(isNearly(grazing, SIMD4(expected, 0, 1 - expected, 1), tolerance: 0.003))
    }

    @Test("The live two-normal phase distorts both captures and wraps in forward and reversed depth",
          arguments: [MeshDepthMode.forward, .reversed])
    func animatedFlow(mode: MeshDepthMode) throws {
        let (reflection, refraction) = try ramps()
        let first = try texture([SIMD4(0.875, 0.5, 1, 1)])
        let second = try texture([SIMD4(0.125, 0.5, 1, 1)])
        var options = PlanarWater.FlowOptions()
        options.speed = 0.15
        options.distortion = 3.125 // The surface is at forward projective depth 0.08.
        options.reflectivity = 0.5
        let material = try PlanarWater.flowing(
            reflection: reflection, refraction: refraction,
            firstNormal: first, secondNormal: second, options: options
        )
        let water = try surface(material, depthMode: mode)
        let clock = try #require(material.surfaceUniforms)
        let camera = try camera(mode: mode)
        let firstFrame = try render(water, camera: camera, depthMode: mode)
        try clock.setFrame(.init(time: 0.5))
        let halfCycle = try render(water, camera: camera, depthMode: mode)
        try clock.setFrame(.init(time: 0.9999))
        let beforeWrap = try render(water, camera: camera, depthMode: mode)
        try clock.setFrame(.init(time: 1))
        let afterWrap = try render(water, camera: camera, depthMode: mode)

        // Refraction contributes the red ramp and the X-flipped reflection the green ramp.
        #expect(halfCycle.x > firstFrame.x + 0.13)
        #expect(halfCycle.y > firstFrame.y + 0.13)
        #expect(isNearly(beforeWrap, afterWrap, tolerance: 0.003))
        #expect(isNearly(firstFrame, afterWrap, tolerance: 0.003))
    }

    @Test("An RG flow texture reverses the normal advection independently of authored direction")
    func flowMap() throws {
        let (reflection, refraction) = try ramps()
        let normal = try texture([SIMD4(0, 0.5, 1, 1), SIMD4(1, 0.5, 1, 1)])
        let reverseX = try texture([SIMD4(0, 0.5, 0, 1)])
        var options = PlanarWater.FlowOptions()
        options.speed = 0.15
        options.distortion = 4.375
        options.reflectivity = 0
        let camera = try camera()
        var pixels: [SIMD4<Float>] = []
        for flowMap in [nil, reverseX] {
            let material = try PlanarWater.flowing(
                reflection: reflection, refraction: refraction, firstNormal: normal, secondNormal: normal,
                flowMap: flowMap, options: options
            )
            let clock = try #require(material.surfaceUniforms)
            try clock.setFrame(.init(time: 0.5))
            pixels.append(try render(surface(material), camera: camera))
        }
        #expect(pixels[1].x > pixels[0].x + 0.15)
    }

    @Test("Reflective water adds sun and scatter in linear HDR and advances four world-space waves")
    func reflectiveWater() throws {
        let reflectedColor = SIMD3<Float>(2, 0.25, 0.5)
        let reflection = try texture([SIMD4(reflectedColor, 1)])
        let normal = try texture([
            SIMD4(0.5, 0.5, 1, 1), SIMD4(0.9, 0.5, 1, 1),
            SIMD4(0.9, 0.5, 1, 1), SIMD4(0.5, 0.5, 1, 1),
        ])
        var options = PlanarWater.ReflectionOptions()
        options.tint = SIMD3(0.25, 0.125, 0.5)
        options.sunDirection = SIMD3(0, 5, 0)
        options.sunColor = SIMD3(0.5, 0.25, 0.125)
        options.distortion = 0
        options.opacity = 0.5
        let material = try PlanarWater.reflective(reflection: reflection, normalMap: normal, options: options)
        let rotation = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
        let water = try surface(material, transform: .init(rotation: rotation))
        let camera = try camera(transform: .init(translation: SIMD3(0, 4, 0), rotation: rotation))
        let still = try render(water, camera: camera, directionToCamera: SIMD3(0, 1, 0))
        let scatter = options.tint + options.sunColor * options.sunColor * 0.15
        let reflectionAndSun = reflectedColor + options.sunColor * 2
        let expected = (scatter * 0.98 + reflectionAndSun * 0.02) * options.opacity
        #expect(isNearly(still, SIMD4(expected, 0.5), tolerance: 0.004))

        let clock = try #require(material.surfaceUniforms)
        try clock.setFrame(.init(time: 8))
        let moving = try render(water, camera: camera, directionToCamera: SIMD3(0, 1, 0))
        #expect(moving.x < still.x - 0.015)
        #expect(moving.w == still.w)
    }

    @Test("Reflective water preserves HDR after applying authored alpha")
    func reflectiveHDROpacity() throws {
        let reflectedColor = SIMD3<Float>(200, 160, 120)
        let reflection = try texture([SIMD4(reflectedColor, 1)])
        let normal = try texture([SIMD4(0.5, 0.5, 1, 1)])
        var options = PlanarWater.ReflectionOptions()
        options.tint = .zero
        options.sunColor = .zero
        options.opacity = 0.5
        let material = try PlanarWater.reflective(reflection: reflection, normalMap: normal, options: options)
        let rotation = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
        let water = try surface(material, transform: .init(rotation: rotation))
        let camera = try camera(transform: .init(translation: SIMD3(0, 4, 0), rotation: rotation))
        let pixel = try render(water, camera: camera, directionToCamera: SIMD3(0, 1, 0))
        #expect(isNearly(pixel, SIMD4(reflectedColor * 0.01, 0.5), tolerance: 0.004))
    }

    @Test("Unlit graph radiance is opt-in and preserves the authored coverage through native archive and path tracing")
    func archivedGraphRadiance() throws {
        let base = SIMD3<Float>(0.25, 0.5, 0.125)
        let radiance = SIMD3<Float>(4, 2, 8)
        let graph = try SurfaceShaderGraph { surface in
            .init(baseColor: surface.constant(base), emission: surface.constant(radiance), opacity: surface.constant(Float(0.5)))
        }
        let ordinary = try PhysicallyBasedMaterial(surfaceGraph: graph, shadingModel: .unlit)
        let camera = try camera()
        #expect(isNearly(try render(surface(ordinary), camera: camera), SIMD4(base * 0.5, 0.5)))
        let optedIn = try PhysicallyBasedMaterial(
            surfaceGraph: graph, shadingModel: .unlit, unlitUsesGraphEmission: true
        ).withLiveFactors()
        let scene = try SceneGraph(roots: [surface(optedIn)], camera: gpu.camera)
        let archive = try SceneArchive(capturing: scene, commandQueue: gpu.queue)
        var json = try #require(try JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(json["version"] as? Int == 27)
        json["version"] = 22
        let downgraded = try JSONSerialization.data(withJSONObject: json)
        #expect(throws: SceneArchive.Failure.decoding("unlit graph emission requires archive version 27")) {
            try SceneArchive(manifest: downgraded, binary: archive.binary)
        }
        let restored = try archive.load(device: gpu.device, commandQueue: gpu.queue).graph
        let node = try #require(restored.roots.first)
        let expected = SIMD4(radiance * 0.5, 0.5)
        #expect(isNearly(try render(node, camera: camera), expected))

        let path = try PathTracingScene(queue: gpu.queue, graph: restored, options: .init(acceleration: .software))
        let tracer = try PathTracer(queue: gpu.queue, scene: path)
        let accumulation = try tracer.makeAccumulation(width: gpu.side, height: gpu.side)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: gpu.side, height: gpu.side, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderWrite
        let target = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let command = try #require(gpu.queue.makeCommandBuffer())
        try tracer.encode(
            camera: gpu.camera, cameraTransform: .init(), destination: target, accumulation: accumulation,
            commandBuffer: command
        )
        try gpu.finish(command)
        #expect(isNearly(try center(of: target), expected))
    }

    @Test("Both water models apply linear or exponential-squared fog after shading and keep alpha",
          arguments: [false, true])
    func fog(exponential: Bool) throws {
        let tint = SIMD3<Float>(0.125, 0.375, 0.5)
        let fogColor = SIMD3<Float>(0.75, 0.25, 0.0625)
        let reflection = try texture([SIMD4(tint, 1)])
        let refraction = try texture([SIMD4(tint, 1)])
        let normal = try texture([SIMD4(0.5, 0.5, 1, 1)])
        let fog = try PlanarWater.Fog(
            color: fogColor, falloff: exponential ? .exponentialSquared(density: 0.25) : .linear(near: 1, far: 5)
        )
        let factor: Float = exponential ? 1 - exp(-1 as Float) : 0.84375
        let expected = tint * (1 - factor) + fogColor * factor

        var flow = PlanarWater.FlowOptions()
        flow.fog = fog
        let flowing = try PlanarWater.flowing(
            reflection: reflection, refraction: refraction, firstNormal: normal, secondNormal: normal, options: flow
        )
        let flowingPixel = try render(surface(flowing), camera: camera())
        #expect(isNearly(flowingPixel, SIMD4(expected, 1), tolerance: 0.004))

        var mirror = PlanarWater.ReflectionOptions()
        mirror.tint = tint
        mirror.sunColor = .zero
        mirror.opacity = 0.5
        mirror.fog = fog
        let reflective = try PlanarWater.reflective(reflection: reflection, normalMap: normal, options: mirror)
        let rotation = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
        let overhead = try camera(transform: .init(translation: SIMD3(0, 4, 0), rotation: rotation))
        let reflectivePixel = try render(
            surface(reflective, transform: .init(rotation: rotation)), camera: overhead,
            directionToCamera: SIMD3(0, 1, 0)
        )
        #expect(isNearly(reflectivePixel, SIMD4(expected * 0.5, 0.5), tolerance: 0.004))
    }

    @Test("A directional caster shadows the water scatter while reflected radiance stays visible")
    func sunShadow() throws {
        let reflected = SIMD3<Float>(0.5, 0.25, 0.125)
        let reflection = try texture([SIMD4(reflected, 1)])
        let normal = try texture([SIMD4(0.5, 0.5, 1, 1)])
        var options = PlanarWater.ReflectionOptions()
        options.sunDirection = SIMD3(1, 1, 0)
        options.sunColor = SIMD3(repeating: 0.5)
        options.tint = SIMD3(repeating: 0.25)
        options.receivesSunShadows = true
        let material = try PlanarWater.reflective(reflection: reflection, normalMap: normal, options: options)
        let rotation = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
        var water = try surface(material, transform: .init(rotation: rotation))
        water.castsShadow = false
        let caster = SceneNode(
            transform: .init(translation: SIMD3(0.625, 1, 0), rotation: rotation),
            mesh: try gpu.unlit(MeshGeometry.plane(width: 0.5, height: 0.5))
        )
        let sun = try DirectionalShadowMap(
            queue: gpu.queue, resolution: 96, directionToLight: options.sunDirection, halfExtent: 3, sampling: .basic
        )
        let camera = try camera(transform: .init(translation: SIMD3(0, 4, 0), rotation: rotation))
        let target = try gpu.makeTarget()
        let command = try #require(gpu.queue.makeCommandBuffer())
        let stats = try encode(
            water, camera: camera, directionToCamera: SIMD3(0, 1, 0), to: target, command: command,
            otherNodes: [caster], shadowMap: sun
        )
        #expect(stats.shadowDrawCount == 1)
        try gpu.finish(command)

        let image = try gpu.pixels(of: target.color)
        let shadow = gpu.pixel(image, x: 26, y: 32)
        let sunlit = gpu.pixel(image, x: 10, y: 32)
        let reflectionTerm = reflected * 0.02
        let scatterTerm = options.tint + options.sunColor * options.sunColor * (0.15 / sqrt(2 as Float))
        #expect(isNearly(shadow, SIMD4(reflectionTerm, 1), tolerance: 0.004))
        #expect(isNearly(sunlit, SIMD4(scatterTerm * 0.98 + reflectionTerm, 1), tolerance: 0.004))
    }

    @Test("A custom Water2 graph replaces color and coverage using bound captures, its own texture and live phase")
    func customFlowGraph() throws {
        let reflected = try texture([SIMD4(0.9, 0, 0, 1)])
        let transmitted = SIMD3<Float>(0.125, 0.25, 0.5)
        let refraction = try texture([SIMD4(transmitted, 1)])
        let flat = try texture([SIMD4(0.5, 0.5, 1, 1)])
        let foam = SIMD3<Float>(0.75, 0.5, 0.25)
        let foamTexture = try texture([SIMD4(foam, 1)])
        var options = PlanarWater.FlowOptions()
        options.speed = 0.15
        let material = try PlanarWater.flowing(
            reflection: reflected, refraction: refraction, firstNormal: flat, secondNormal: flat,
            options: options, additionalTextures: ["Foam": foamTexture]
        ) { surface, water in
            let base = water.refraction.samplePlanarRefraction(at: water.coordinates).rgb
            let foam = surface.texture("Foam").sample(surface.uv0).rgb
            return .init(baseColor: base.mix(foam, by: water.phase / 0.15), opacity: water.opacity * 0.5)
        }
        let water = try surface(material)
        let camera = try camera()
        let first = try render(water, camera: camera)
        let clock = try #require(material.surfaceUniforms)
        try clock.setFrame(.init(time: 0.5))
        let second = try render(water, camera: camera)
        #expect(isNearly(first, SIMD4(transmitted * 0.5, 0.5), tolerance: 0.004))
        #expect(isNearly(second, SIMD4((transmitted + foam) * 0.25, 0.5), tolerance: 0.004))
    }

    @Test("Invalid water optics and capture aliases fail before rendering")
    func invalidInput() throws {
        let capture = try texture([SIMD4(0.5, 0.5, 1, 1)])
        let other = try texture([SIMD4(0.25, 0.5, 1, 1)])
        var options = PlanarWater.FlowOptions()
        options.reflectivity = -.infinity
        #expect(throws: PlanarWater.Failure.flowOptions) {
            try PlanarWater.flowing(
                reflection: capture, refraction: other, firstNormal: capture, secondNormal: capture, options: options
            )
        }
        #expect(throws: PlanarWater.Failure.textures) {
            try PlanarWater.flowing(reflection: capture, refraction: capture, firstNormal: other, secondNormal: other)
        }
        #expect(throws: PlanarWater.Failure.fog) {
            try PlanarWater.Fog(color: .zero, falloff: .linear(near: 3, far: 2))
        }
        #expect(throws: PlanarWater.Failure.reservedTexture("WaterReflection")) {
            try PlanarWater.flowing(
                reflection: capture, refraction: other, firstNormal: capture, secondNormal: other,
                additionalTextures: ["WaterReflection": other]
            )
        }
        #expect(throws: SurfaceShaderGraph.Failure.directionalShadowCoverage) {
            try SurfaceShaderGraph { .init(opacity: $0.directionalShadowVisibility) }
        }
        #expect(throws: PhysicallyBasedMaterial.Failure.unlitGraphEmission) {
            try PhysicallyBasedMaterial(shadingModel: .unlit, unlitUsesGraphEmission: true)
        }
        var reflective = PlanarWater.ReflectionOptions()
        reflective.sunDirection = .zero
        #expect(throws: PlanarWater.Failure.reflectionOptions) {
            try PlanarWater.reflective(reflection: capture, normalMap: other, options: reflective)
        }
    }

    private func camera(transform: SceneTransform = .init(), mode: MeshDepthMode = .forward) throws -> ShaderGraphCamera {
        let depthScale: Float = mode == .forward ? -0.02 : 0.02
        let projection = simd_float4x4(
            SIMD4(0.5, 0, 0, 0), SIMD4(0, 0.5, 0, 0), SIMD4(0, 0, depthScale, 0),
            SIMD4(0, 0, mode == .forward ? 0 : 1, 1)
        )
        return try ShaderGraphCamera(viewMatrix: simd_inverse(transform.matrix), projectionMatrix: projection, near: 0, far: 50)
    }

    private func surface(
        _ material: PhysicallyBasedMaterial, transform: SceneTransform = .init(translation: SIMD3(0, 0, -4)),
        depthMode: MeshDepthMode = .forward
    ) throws -> SceneNode {
        SceneNode(
            transform: transform,
            mesh: try gpu.unlit(MeshGeometry.plane(width: 4, height: 4), material: .physicallyBased(material), depthMode: depthMode)
        )
    }

    private func marker(z: Float, color: SIMD3<Float>, towardMirror: Bool, hdr: Bool) throws -> SceneNode {
        let material: MeshMaterial
        if hdr {
            let graph = try SurfaceShaderGraph { .init(emission: $0.constant(color)) }
            material = .physicallyBased(try PhysicallyBasedMaterial(
                surfaceGraph: graph, shadingModel: .unlit, unlitUsesGraphEmission: true
            ))
        } else {
            material = .classic(try ClassicMaterial(model: .basic, color: color))
        }
        return SceneNode(
            transform: .init(
                translation: SIMD3(0, 0, z),
                rotation: simd_quatf(angle: towardMirror ? .pi : 0, axis: SIMD3(0, 1, 0))
            ),
            mesh: try gpu.unlit(MeshGeometry.plane(width: 4, height: 4), material: material),
            frustumCulled: false
        )
    }

    private func captureTarget() throws -> (color: any MTLTexture, depth: any MTLTexture) {
        let color = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: gpu.side, height: gpu.side, mipmapped: false)
        color.storageMode = .shared
        color.usage = [.shaderRead, .renderTarget]
        let depth = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: gpu.side, height: gpu.side, mipmapped: false)
        depth.storageMode = .private
        depth.usage = [.renderTarget, .shaderRead]
        return (try #require(gpu.device.makeTexture(descriptor: color)), try #require(gpu.device.makeTexture(descriptor: depth)))
    }

    private func render(
        _ water: SceneNode, camera: ShaderGraphCamera, directionToCamera: SIMD3<Float> = SIMD3(0, 0, 1),
        depthMode: MeshDepthMode = .forward
    ) throws -> SIMD4<Float> {
        let output = try gpu.makeTarget()
        let command = try #require(gpu.queue.makeCommandBuffer())
        try encode(water, camera: camera, directionToCamera: directionToCamera, to: output, command: command, depthMode: depthMode)
        try gpu.finish(command)
        return try center(of: output.color)
    }

    @discardableResult
    private func encode(
        _ water: SceneNode, camera: ShaderGraphCamera, directionToCamera: SIMD3<Float>,
        to target: (color: any MTLTexture, depth: any MTLTexture), command: any MTLCommandBuffer,
        depthMode: MeshDepthMode = .forward, otherNodes: [SceneNode] = [], shadowMap: DirectionalShadowMap? = nil
    ) throws -> SceneDrawStatistics {
        let position = simd_inverse(camera.viewMatrix)[3]
        return try SceneGraph(roots: [water] + otherNodes, camera: gpu.camera).encode(
            pass: gpu.pass(color: target.color, depth: target.depth, depthMode: depthMode), commandBuffer: command,
            clipFromWorld: camera.projectionMatrix * camera.viewMatrix,
            cameraPosition: SIMD3(position.x, position.y, position.z) / position.w,
            directionToCamera: directionToCamera, depthMode: depthMode, shadowMap: shadowMap, graphCamera: camera
        )
    }

    private func center(of texture: any MTLTexture) throws -> SIMD4<Float> {
        gpu.pixel(try gpu.pixels(of: texture), x: gpu.side / 2, y: gpu.side / 2)
    }

    private func ramps() throws -> (any MTLTexture, any MTLTexture) {
        let steps = 32
        let ramp = (0..<steps).map { (Float($0) + 0.5) / Float(steps) }
        return (try texture(ramp.map { SIMD4(0, 1 - $0, 0, 1) }), try texture(ramp.map { SIMD4($0, 0, 0, 1) }))
    }

    private func texture(_ pixels: [SIMD4<Float>]) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: pixels.count, height: 1, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let texture = try #require(gpu.device.makeTexture(descriptor: descriptor))
        let packed = pixels.map(SIMD4<Float16>.init)
        try packed.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, pixels.count, 1), mipmapLevel: 0,
                withBytes: try #require(bytes.baseAddress), bytesPerRow: bytes.count
            )
        }
        return texture
    }
}
