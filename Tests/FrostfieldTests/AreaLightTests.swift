import Foundation
import Frostfield
import Metal
import Testing
import simd

struct AreaLightConventionTests {
    @Test("Rectangle power, size, and facing follow the path tracer's area light conventions")
    func conventions() throws {
        let light = try PunctualLight(rectangleWidth: 2, height: 0.5, power: 30, color: SIMD3(1, 0.5, 0.25))
        #expect(light.kind == .rectangle(width: 2, height: 0.5))
        #expect(abs(light.intensity - 30 / .pi) < 1e-5)
        let traced = try PathTracer.AreaLight(
            center: .zero, aim: SIMD3(0, 0, -1), size: SIMD2(2, 0.5), radiantPower: SIMD3(repeating: 30)
        )
        #expect(abs(traced.radiance.x - light.intensity) <= 1e-6 * light.intensity)

        // An unrotated rectangle node emits toward local -Z, the traced light's front face.
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let resolved = try #require(try SceneGraph(roots: [SceneNode(light: light)], camera: camera).resolve()[0].light)
        let front = simd_normalize(simd_cross(traced.halfU, traced.halfV))
        #expect(simd_distance(resolved.direction, front) < 1e-6)

        #expect(throws: PunctualLight.Failure.rectangle) { try PunctualLight(kind: .rectangle(width: 0, height: 1)) }
        #expect(throws: PunctualLight.Failure.rectangle) { try PunctualLight(kind: .rectangle(width: 1, height: .nan)) }
        #expect(throws: PunctualLight.Failure.rectangle) { try PunctualLight(rectangleWidth: -1, height: 1, power: 1) }
        #expect(throws: PunctualLight.Failure.power) { try PunctualLight(rectangleWidth: 1, height: 1, power: -1) }
        #expect(throws: PunctualLight.Failure.power) {
            try PunctualLight(rectangleWidth: 1e-20, height: 1e-20, power: 1e30)
        }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct AreaLightTests {
    private static let side = 33
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue

    init() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        self.device = device
        queue = try #require(device.makeCommandQueue())
    }

    /// The form factor from a parallel rectangle `a` by `b` to a differential area
    /// a distance `c` below one of its corners.
    private static func cornerFormFactor(_ a: Double, _ b: Double, _ c: Double) -> Double {
        let x = a / c
        let y = b / c
        let rx = (1 + x * x).squareRoot()
        let ry = (1 + y * y).squareRoot()
        return (x / rx * atan(y / rx) + y / ry * atan(x / ry)) / (2 * .pi)
    }

    /// Renders a 6 m square facing +Z at z = -3, viewed from the origin, and returns the center pixel,
    /// which sees the receiver point (0, 0, -3) along the view direction +Z.
    private func centerPixel(_ material: PhysicallyBasedMaterial, lights: [SceneNode]) throws -> SIMD4<Float> {
        let corners: [SIMD3<Float>] = [SIMD3(-3, -3, 0), SIMD3(3, -3, 0), SIMD3(3, 3, 0), SIMD3(-3, 3, 0)]
        let geometry = try MeshGeometry(
            vertices: corners.map { MeshVertex(position: $0, normal: SIMD3(0, 0, 1)) }, indices: [0, 1, 2, 0, 2, 3]
        )
        let renderer = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(material)
        )
        let receiver = SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -3)), mesh: renderer)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 20)
        let graph = SceneGraph(roots: [receiver] + lights, camera: camera)

        let side = Self.side
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: side, height: side, mipmapped: false
        )
        colorDescriptor.storageMode = .shared
        colorDescriptor.usage = .renderTarget
        let color = try #require(device.makeTexture(descriptor: colorDescriptor))
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: side, height: side, mipmapped: false
        )
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = .renderTarget
        let depth = try #require(device.makeTexture(descriptor: depthDescriptor))
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        let buffer = try #require(queue.makeCommandBuffer())
        try graph.encode(pass: pass, commandBuffer: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        try #require(buffer.status == .completed, "\(buffer.error?.localizedDescription ?? "The area light draw failed.")")
        var pixel = SIMD4<Float16>.zero
        withUnsafeMutablePointer(to: &pixel) {
            color.getBytes(
                $0, bytesPerRow: MemoryLayout<SIMD4<Float16>>.stride,
                from: MTLRegionMake2D(side / 2, side / 2, 1, 1), mipmapLevel: 0
            )
        }
        return SIMD4<Float>(pixel)
    }

    private func rectangle(
        at position: SIMD3<Float>, rotation: simd_quatf = simd_quatf(angle: 0, axis: SIMD3(0, 1, 0)),
        width: Float, height: Float, luminance: Float, color: SIMD3<Float> = SIMD3(repeating: 1)
    ) throws -> SceneNode {
        SceneNode(
            transform: SceneTransform(translation: position, rotation: rotation),
            light: try PunctualLight(kind: .rectangle(width: width, height: height), color: color, intensity: luminance)
        )
    }

    @Test("Diffuse radiance under a rectangle matches the analytic form factor", arguments: [false, true])
    func analyticIrradiance(cornerAligned: Bool) throws {
        let luminance: Float = 1.5
        // The receiver point is under the center of a 2 m square, or under the
        // corner of a 2 m by 1 m rectangle, 1 m below the emitter.
        let light = cornerAligned
            ? try rectangle(at: SIMD3(1, 0.5, -2), width: 2, height: 1, luminance: luminance)
            : try rectangle(at: SIMD3(0, 0, -2), width: 2, height: 2, luminance: luminance)
        let formFactor = cornerAligned ? Self.cornerFormFactor(2, 1, 1) : 4 * Self.cornerFormFactor(1, 1, 1)
        // A white Lambertian receiver reflects albedo / π times irradiance π·L·F.
        let expected = Float(Double(luminance) * formFactor)
        let pixel = try centerPixel(PunctualLightingFixture.diffuse(), lights: [light])
        #expect(abs(pixel.x - expected) <= 0.02 * expected, "Expected \(expected), rendered \(pixel.x).")
        #expect(pixel.x == pixel.y && pixel.y == pixel.z && pixel.w == 1)

        let tinted = cornerAligned
            ? try rectangle(at: SIMD3(1, 0.5, -2), width: 2, height: 1, luminance: luminance, color: SIMD3(1, 0.5, 0))
            : try rectangle(at: SIMD3(0, 0, -2), width: 2, height: 2, luminance: luminance, color: SIMD3(1, 0.5, 0))
        let colored = try centerPixel(PunctualLightingFixture.diffuse(), lights: [tinted])
        #expect(abs(colored.y - pixel.y * 0.5) <= 0.01 * pixel.y && colored.z == 0)
    }

    @Test("Irradiance clips an emitter that crosses the receiver's horizon")
    func horizonClipping() throws {
        let luminance: Float = 2
        // A 2 m square 1 m above the receiver point, facing down, spanning 0.5 m
        // behind the receiver's plane and 1.5 m in front of it.
        let light = try rectangle(
            at: SIMD3(0, 1, -2.5), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0)),
            width: 2, height: 2, luminance: luminance
        )
        // Integrate cos(receiver) cos(emitter) / d² over the visible part numerically.
        let steps = 600
        var integral = 0.0
        for i in 0..<steps {
            let x = -1 + 2 * (Double(i) + 0.5) / Double(steps)
            for j in 0..<steps {
                let w = 1.5 * (Double(j) + 0.5) / Double(steps)
                let d2 = x * x + 1 + w * w
                integral += w / (d2 * d2)
            }
        }
        integral *= (2.0 / Double(steps)) * (1.5 / Double(steps))
        let expected = Float(Double(luminance) * integral / .pi)
        let pixel = try centerPixel(PunctualLightingFixture.diffuse(), lights: [light])
        #expect(abs(pixel.x - expected) <= 0.02 * expected, "Expected \(expected), rendered \(pixel.x).")
    }

    @Test("A rectangle light is one-sided")
    func oneSided() throws {
        let facing = try rectangle(at: SIMD3(0, 0, -2), width: 2, height: 2, luminance: 3)
        let away = try rectangle(
            at: SIMD3(0, 0, -2), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0)), width: 2, height: 2, luminance: 3
        )
        let material = try PhysicallyBasedMaterial(roughness: 0.4, lightIntensity: 0, ambientIntensity: 0)
        #expect(try centerPixel(material, lights: [facing]).x > 0.5)
        #expect(try centerPixel(material, lights: [away]) == SIMD4(0, 0, 0, 1))
    }

    @Test("Small rectangles converge to a point light with intensity luminance times area", arguments: [Float(0.001), 0.002, 0.01])
    func smallEmitter(size: Float) throws {
        // The 1 mm square uses the far-field approximation; the others integrate exactly.
        let luminance = 1 / (size * size)
        let rectangle = try rectangle(at: SIMD3(0, 0, -2), width: size, height: size, luminance: luminance)
        let point = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -2)), light: try PunctualLight(kind: .point(), intensity: 1)
        )
        let expected = try centerPixel(PunctualLightingFixture.diffuse(), lights: [point])
        let pixel = try centerPixel(PunctualLightingFixture.diffuse(), lights: [rectangle])
        #expect(abs(expected.x - 1 / .pi) <= 0.002)
        #expect(abs(pixel.x - expected.x) <= 0.005 * expected.x, "Rendered \(pixel.x), expected \(expected.x).")
    }

    @Test("A mirror reflects the emitter radiance through its specular lobe")
    func mirrorSpecular() throws {
        let luminance: Float = 4
        let light = try rectangle(at: SIMD3(0, 0, -2), width: 1, height: 1, luminance: luminance)
        let mirror = try PhysicallyBasedMaterial(metallic: 1, roughness: 0, lightIntensity: 0, ambientIntensity: 0)
        let pixel = try centerPixel(mirror, lights: [light])
        #expect(abs(pixel.x - luminance) <= 0.05 * luminance, "Rendered \(pixel.x).")

        // Rough metal spreads the same light, so the peak falls well below the mirror's.
        let rough = try PhysicallyBasedMaterial(metallic: 1, roughness: 0.7, lightIntensity: 0, ambientIntensity: 0)
        let spread = try centerPixel(rough, lights: [light])
        #expect(spread.x > 0.01 && spread.x < 0.5 * pixel.x)
    }

    @Test("Clearcoat reflects a rectangle light over its base")
    func clearcoat() throws {
        let luminance: Float = 10
        let light = try rectangle(at: SIMD3(0, 0, -2), width: 1, height: 1, luminance: luminance)
        // Disable the whole base specular lobe. An IOR of one only clears normal-incidence Fresnel.
        let specular = try PhysicallyBasedMaterial.Specular(factor: 0)
        let black = SIMD4<Float>(0, 0, 0, 1)
        let base = try PhysicallyBasedMaterial(
            roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: black, specular: specular
        )
        let coated = try PhysicallyBasedMaterial(
            roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: black,
            clearcoat: PhysicallyBasedMaterial.Clearcoat(factor: 1, roughness: 0), specular: specular
        )
        #expect(try centerPixel(base, lights: [light]) == SIMD4(0, 0, 0, 1))
        // At normal incidence the coat's Fresnel weight is 0.04 of the mirrored radiance.
        let pixel = try centerPixel(coated, lights: [light])
        #expect(abs(pixel.x - 0.04 * luminance) <= 0.05 * 0.04 * luminance, "Rendered \(pixel.x).")
    }

    @Test("A shadowed rectangle light is blocked by an occluder seen from its center")
    func rectangleShadow() throws {
        let fixture = try PunctualLightingFixture()
        let floor = SceneNode(mesh: try fixture.floor(halfExtent: 4))
        let light = SceneNode(
            transform: PunctualLightingFixture.aim(SIMD3(0, 2, 0), at: .zero),
            light: try PunctualLight(
                kind: .rectangle(width: 1, height: 1), intensity: 6, shadow: PunctualLight.Shadow()
            )
        )
        let occluder = SceneNode(
            transform: SceneTransform(translation: SIMD3(1, 1, 0), scale: SIMD3(repeating: 0.4)), mesh: try fixture.box()
        )
        let maps = try PunctualShadowMaps(queue: fixture.queue, resolution: 256, spotCapacity: 0, pointCapacity: 1)
        let shadowed = try fixture.render([floor, light, occluder], maps: maps)
        let unshadowed = try fixture.render([floor, light, occluder], maps: nil)
        let behind = fixture.pixel(2.06, 0)
        let opposite = fixture.pixel(-2.06, 0)
        #expect(unshadowed[behind].x > 0.05)
        #expect(abs(unshadowed[behind].x - unshadowed[opposite].x) <= 0.01 * unshadowed[opposite].x)
        #expect(shadowed[behind].x < 0.05 * unshadowed[behind].x)
        #expect(abs(shadowed[opposite].x - unshadowed[opposite].x) <= 0.005 * unshadowed[opposite].x)
        // The floor directly below the emitter is the brightest point.
        #expect(unshadowed[fixture.pixel(0, 0)].x > unshadowed[behind].x)
    }
}
