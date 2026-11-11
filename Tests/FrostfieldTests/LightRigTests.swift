import Foundation
import Frostfield
import Metal
import simd
import Testing

struct LightRigTests {
    typealias Light = LightRig.Light

    @Test("Power converts to each kind's intensity through its emitting solid angle or area")
    func powerConversion() throws {
        let point = Light(id: "p", kind: .point(), position: .zero, target: SIMD3(0, 0, -1), emission: .power(100))
        #expect(abs(try point.intensity() - Float(100 / (4 * Double.pi))) < 1e-4)
        #expect(abs(try #require(try point.power()) - 100) < 1e-3)

        // The closed-form spot solid angle equals a numerical integral of the shading falloff.
        let inner: Float = 0.3
        let outer: Float = 0.7
        var integral = 0.0
        let steps = 200_000
        for step in 0..<steps {
            let cosine = cos(Double(outer)) + (1 - cos(Double(outer))) * (Double(step) + 0.5) / Double(steps)
            let width = cos(Double(inner)) - cos(Double(outer))
            let cone = min(max((cosine - cos(Double(outer))) / width, 0), 1)
            integral += cone * cone * (1 - cos(Double(outer))) / Double(steps)
        }
        #expect(abs(Light.spotSolidAngle(inner: inner, outer: outer) - 2 * .pi * integral) < 1e-6)
        let spot = Light(id: "s", kind: .spot(innerAngle: inner, outerAngle: outer), position: .zero,
                         target: SIMD3(0, 0, -1), emission: .power(50))
        #expect(abs(Double(try spot.intensity()) - 50 / (2 * .pi * integral)) < 1e-3)
        // Narrowing the cone at constant power raises intensity.
        var narrow = spot
        narrow.kind = .spot(innerAngle: 0.1, outerAngle: 0.3)
        #expect(try narrow.intensity() > spot.intensity())

        let panel = Light(id: "r", kind: .rectangle(width: 2, height: 0.5), position: .zero, target: SIMD3(0, 0, -1),
                          emission: .power(Float.pi))
        #expect(abs(try panel.intensity() - 1) < 1e-5)
        let sun = Light(id: "d", kind: .directional(), position: .zero, target: SIMD3(0, -1, 0), emission: .power(1))
        #expect(throws: LightRig.Failure.directionalPower("d")) { try sun.intensity() }
    }

    @Test("Color temperature, aim and validation")
    func authoring() throws {
        let daylight = try Light.color(temperature: 6_500)
        #expect(simd_reduce_max(daylight) == 1)
        #expect(simd_reduce_min(daylight) > 0.85, "\(daylight)")
        let tungsten = try Light.color(temperature: 2_700)
        #expect(tungsten.x == 1 && tungsten.z < 0.5 && tungsten.y < tungsten.x, "\(tungsten)")
        #expect(throws: LightRig.Failure.temperature) { try Light.color(temperature: 500) }

        let light = Light(id: "a", kind: .spot(innerAngle: 0.2, outerAngle: 0.5), position: SIMD3(1, 2, 3),
                          target: SIMD3(1, 0, 3), emission: .intensity(5))
        let transform = try light.transform()
        #expect(simd_distance(transform.rotation.act(SIMD3(0, 0, -1)), SIMD3(0, -1, 0)) < 1e-5)
        let level = Light(id: "b", kind: .rectangle(width: 1, height: 1), position: .zero, target: SIMD3(3, 0.5, -2),
                          emission: .intensity(1))
        // The panel's width axis stays horizontal.
        #expect(abs(try level.transform().rotation.act(SIMD3(1, 0, 0)).y) < 1e-5)

        #expect(throws: LightRig.Failure.duplicateID("a")) { try LightRig(lights: [light, light]) }
        var bad = light
        bad.target = bad.position
        #expect(throws: LightRig.Failure.aim("a")) { try LightRig(lights: [bad]) }
        bad = light
        bad.color = SIMD3(2, 0, 0)
        #expect(throws: LightRig.Failure.color("a")) { try LightRig(lights: [bad]) }
        bad = light
        bad.kind = .spot(innerAngle: 0.6, outerAngle: 0.5)
        #expect(throws: LightRig.Failure.size("a")) { try LightRig(lights: [bad]) }
    }

    @Test("Editing a light changes the tracer's lighting; an unchanged rig keeps accumulating")
    func editSemantics() throws {
        var rig = try LightRig(lights: [
            Light(id: "key", kind: .rectangle(width: 1, height: 0.5), position: SIMD3(0, 2, 2), target: .zero,
                  emission: .power(40)),
            Light(id: "sun", kind: .directional(angularDiameter: 0.01), position: SIMD3(1, 1, 1), target: .zero,
                  emission: .intensity(2)),
        ])
        let before = try rig.pathLighting()
        #expect(before == (try rig.pathLighting()))
        #expect(before.lights.count == 1 && before.analyticLights.count == 1)
        rig["key"]?.emission = .power(60)
        #expect(before != (try rig.pathLighting()))
        rig["sun"]?.isEnabled = false
        #expect(try rig.pathLighting().analyticLights.isEmpty)
        #expect(try rig.sceneNodes().count == 1)
        #expect(rig.directionalShadow == nil)
    }

    @Test("A rig fails at construction when a rectangle is brighter than the path tracer accepts")
    func pathTracerLimits() throws {
        // 10,000 W over 0.04 m² is about 79,577 nits, above the 65,504 half-float radiance limit.
        let panel = Light(id: "panel", kind: .rectangle(width: 0.2, height: 0.2), position: SIMD3(0, 2, 0), target: .zero,
                          emission: .power(10_000))
        func isUnsupported(_ failure: LightRig.Failure?, _ id: String) -> Bool {
            if case .unsupported(id, let detail)? = failure { return detail.contains("65,504") }
            return false
        }
        let overPowered = #expect(throws: LightRig.Failure.self) { try LightRig(lights: [panel]) }
        #expect(isUnsupported(overPowered, "panel"), "\(String(describing: overPowered))")
        var bright = panel
        bright.emission = .intensity(70_000)
        let overBright = #expect(throws: LightRig.Failure.self) { try LightRig(lights: [bright]) }
        #expect(isUnsupported(overBright, "panel"))
        // A disabled light is still validated.
        bright.isEnabled = false
        #expect(throws: LightRig.Failure.self) { try LightRig(lights: [bright]) }

        // The limit applies to luminance times the brightest color component.
        var tinted = panel
        tinted.emission = .intensity(70_000)
        tinted.color = SIMD3(0.9, 0.5, 0.2)
        var rig = try LightRig(lights: [tinted])
        #expect(try rig.pathLighting().lights.first?.radiance == SIMD3<Float>(0.9, 0.5, 0.2) * 70_000)
        // An edit after construction fails with the same clear error in either renderer.
        rig["panel"]?.color = SIMD3(1, 0.5, 0.2)
        let traced = #expect(throws: LightRig.Failure.self) { try rig.pathLighting() }
        #expect(isUnsupported(traced, "panel"))
        let raster = #expect(throws: LightRig.Failure.self) { try rig.sceneNodes() }
        #expect(isUnsupported(raster, "panel"))

        // The widest sun the rig accepts keeps its full diameter in the path tracer, while
        // the raster shadow radius stays within what DirectionalShadowMap accepts.
        let sun = try LightRig(lights: [Light(id: "sun", kind: .directional(angularDiameter: .pi / 4),
                                              position: SIMD3(1, 1, 1), target: .zero, emission: .intensity(3))])
        #expect(try sun.pathLighting().analyticLights.first?.angularDiameter == .pi / 4)
        #expect(sun.directionalShadow?.angularRadius == DirectionalShadowMap.maximumLightAngularRadius)
    }

    @Test("One rig lights raster PBR and the path tracer with matching radiance",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func rasterMatchesPathTracing() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 20)
        let plane = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 4, height: 4),
            material: .physicallyBased(try PhysicallyBasedMaterial(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0.8, 0.8, 0.8, 1),
                alphaMode: .opaque
            ))
        )
        let target = SIMD3<Float>(0, 0, -2)
        let rigs = [
            try LightRig(lights: [Light(
                id: "spot", kind: .spot(innerAngle: 0.25, outerAngle: 0.6), position: SIMD3(0.6, 0.2, -1.2),
                target: target, color: try Light.color(temperature: 3_200), emission: .power(12), castsShadow: false
            )]),
            try LightRig(lights: [Light(
                id: "panel", kind: .rectangle(width: 0.4, height: 0.25), position: SIMD3(-0.7, 0.3, -1.1),
                target: target, emission: .power(6), castsShadow: false
            )]),
        ]
        let size = 33
        for (index, rig) in rigs.enumerated() {
            let graph = SceneGraph(
                roots: [SceneNode(transform: .init(translation: target), mesh: plane)] + (try rig.sceneNodes()),
                camera: camera
            )
            let raster = try Self.rasterCenter(graph, device: device, queue: queue, size: size)
            let tracer = try PathTracer(
                queue: queue, scene: PathTracingScene(queue: queue, graph: graph), lighting: try rig.pathLighting()
            )
            let traced = try Self.tracedCenter(tracer, camera: camera, device: device, queue: queue, size: size)
            for channel in 0..<3 {
                let relative = abs(traced[channel] - raster[channel]) / max(raster[channel], 1e-4)
                // Spots are deterministic with one shadow ray; area samples converge with noise.
                #expect(relative < (index == 0 ? 0.01 : 0.04), "rig \(index) channel \(channel): \(raster) vs \(traced)")
            }
            #expect(raster.x > 0.01)
        }
    }

    private static func rasterCenter(
        _ graph: SceneGraph, device: any MTLDevice, queue: any MTLCommandQueue, size: Int
    ) throws -> SIMD4<Float> {
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: size, height: size, mipmapped: false)
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        let color = try #require(device.makeTexture(descriptor: colorDescriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: size, height: size, mipmapped: false)
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = try #require(device.makeTexture(descriptor: depthDescriptor))
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        let buffer = try #require(queue.makeCommandBuffer())
        try graph.encode(pass: pass, commandBuffer: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed)
        var sum = SIMD4<Float>.zero
        for dy in -1...1 { for dx in -1...1 { sum += try pixel(color, size / 2 + dx, size / 2 + dy) } }
        return sum / 9
    }

    private static func tracedCenter(
        _ tracer: PathTracer, camera: PerspectiveCamera, device: any MTLDevice, queue: any MTLCommandQueue, size: Int
    ) throws -> SIMD4<Float> {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: size, height: size, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderWrite
        let output = try #require(device.makeTexture(descriptor: descriptor))
        let accumulation = try tracer.makeAccumulation(width: size, height: size)
        var history: PathTracer.History?
        for _ in 0..<64 {
            let buffer = try #require(queue.makeCommandBuffer())
            history = try tracer.encode(camera: camera, cameraTransform: .init(), destination: output,
                                        accumulation: accumulation, previous: history, commandBuffer: buffer,
                                        samplesPerPixel: 16, settings: .init(maxBounces: 1))
            buffer.commit()
            buffer.waitUntilCompleted()
            try #require(buffer.status == .completed)
        }
        // Average a 3×3 block to steady the sampled estimate.
        var sum = SIMD4<Float>.zero
        for dy in -1...1 { for dx in -1...1 { sum += try pixel(output, size / 2 + dx, size / 2 + dy) } }
        return sum / 9
    }

    private static func pixel(_ texture: any MTLTexture, _ x: Int, _ y: Int? = nil) throws -> SIMD4<Float> {
        var value = SIMD4<Float16>.zero
        try withUnsafeMutableBytes(of: &value) { bytes in
            texture.getBytes(try #require(bytes.baseAddress), bytesPerRow: 8,
                             from: MTLRegionMake2D(x, y ?? x, 1, 1), mipmapLevel: 0)
        }
        return SIMD4<Float>(value)
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SceneEnvironmentTests {
    @Test("Scene environment intensity scales image-based light without new preparation")
    func environmentIntensity() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 32, height: 16, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let panorama = try #require(device.makeTexture(descriptor: descriptor))
        let pixels = [SIMD4<Float16>](repeating: SIMD4(0.5, 0.5, 0.5, 1), count: 32 * 16)
        pixels.withUnsafeBytes { panorama.replace(region: MTLRegionMake2D(0, 0, 32, 16), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 32 * 8) }
        let lighting = try EnvironmentLighting(queue: queue, equirectangular: panorama)
        let material = try PhysicallyBasedMaterial(
            roughness: 1, lightIntensity: 0, alphaMode: .opaque,
            environment: try .init(lighting: lighting, intensity: 1, yaw: 0)
        )
        let mesh = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: .plane(), material: .physicallyBased(material))
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        var graph = SceneGraph(roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -1)), mesh: mesh)], camera: camera)

        func center() throws -> Float {
            let color = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 9, height: 9, mipmapped: false)
            color.storageMode = .shared
            color.usage = .renderTarget
            let target = try #require(device.makeTexture(descriptor: color))
            let depth = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: 9, height: 9, mipmapped: false)
            depth.storageMode = .private
            depth.usage = .renderTarget
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = target
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = try #require(device.makeTexture(descriptor: depth))
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.clearDepth = 1
            let buffer = try #require(queue.makeCommandBuffer())
            try graph.encode(pass: pass, commandBuffer: buffer)
            buffer.commit()
            buffer.waitUntilCompleted()
            var value = SIMD4<Float16>.zero
            try withUnsafeMutableBytes(of: &value) { target.getBytes(try #require($0.baseAddress), bytesPerRow: 8, from: MTLRegionMake2D(4, 4, 1, 1), mipmapLevel: 0) }
            return Float(value.x)
        }
        let base = try center()
        graph.environmentIntensity = 0.25
        let dimmed = try center()
        #expect(base > 0.05 && abs(dimmed - base * 0.25) < base * 0.01, "\(base) → \(dimmed)")
        graph.environmentIntensity = -1
        #expect(throws: MeshRenderer.Failure.environment) { try center() }
    }
}
