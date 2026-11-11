import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct RectangleLightLTCTests {
    struct Reference: Sendable {
        let name: String
        let view: SIMD3<Float>
        let center: SIMD3<Float>
        let roll: Float
        let tilt: Float
        let roughness: Float
        let expected: SIMD3<Float>

        static let cases = [
            Self(name: "narrow", view: SIMD3(0.42, 0.23, 0.88), center: SIMD3(-0.65, -0.12, 1.25),
                 roll: 0.18, tilt: 0.26, roughness: 0.26, expected: SIMD3(2.865218, 1.034877, 0.278736)),
            Self(name: "rolled", view: SIMD3(0.42, 0.23, 0.88), center: SIMD3(-0.65, -0.12, 1.25),
                 roll: 1.11, tilt: 0.26, roughness: 0.26, expected: SIMD3(3.015901, 1.089301, 0.293395)),
            Self(name: "rough", view: SIMD3(0.42, 0.23, 0.88), center: SIMD3(-0.65, -0.12, 1.25),
                 roll: 1.11, tilt: 0.26, roughness: 0.65, expected: SIMD3(0.520983, 0.188301, 0.050787)),
            Self(name: "grazing", view: SIMD3(0.97, 0.1, 0.23), center: SIMD3(-2.5, -0.18, 0.79),
                 roll: 0.83, tilt: -0.2, roughness: 0.41, expected: SIMD3(0.934113, 0.398229, 0.140013)),
        ]
    }

    // These values come from a JS evaluation of pinned LTC.js and
    // PhysicalLightingModel.directRectArea with RectAreaLightTexturesLib's half tables.
    @Test("An asymmetric panel's stock and graph specular pixels match r187 LTC", arguments: Reference.cases)
    func metalReflection(_ reference: Reference) throws {
        let baseColor = SIMD4<Float>(0.8, 0.4, 0.2, 1)
        let stock = try PhysicallyBasedMaterial(
            metallic: 1, roughness: reference.roughness, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: baseColor
        )
        let stockPixel = try render(reference, material: stock)
        #expect(isNearly(stockPixel, SIMD4(reference.expected, 1), tolerance: 0.003), "\(reference.name)")

        if reference.name == "grazing" {
            let graph = try SurfaceShaderGraph { builder in .init(roughness: builder.constant(reference.roughness)) }
            let graphMaterial = try PhysicallyBasedMaterial(
                metallic: 1, roughness: 1, lightIntensity: 0, ambientIntensity: 0, receivesShadows: true,
                baseColorFactor: baseColor, surfaceGraph: graph
            )
            let graphPixel = try render(reference, material: graphMaterial)
            #expect(graphPixel == stockPixel)
        }
    }

    @Test("Clearcoat uses its own LTC roughness and integrated grazing Fresnel")
    func clearcoatReflection() throws {
        let reference = try #require(Reference.cases.last)
        let coated = try PhysicallyBasedMaterial(
            roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0, 0, 0, 1),
            clearcoat: .init(factor: 0.6, roughness: 0.34), specular: .init(factor: 0)
        )
        let pixel = try render(reference, material: coated)
        #expect(isNearly(pixel, SIMD4(0.207771, 0.150057, 0.080800, 1), tolerance: 0.002))
    }

    private func render(_ reference: Reference, material: PhysicallyBasedMaterial) throws -> SIMD4<Float> {
        let gpu = try GeometryRenderFixture(side: 65)
        let receiverPosition = SIMD3<Float>(0, 0, -3)
        let receiver = SceneNode(
            transform: .init(translation: receiverPosition),
            mesh: try gpu.unlit(GeometryRenderFixture.square(1), material: .physicallyBased(material)),
            castsShadow: false
        )
        let tilt = simd_quatf(angle: reference.tilt, axis: SIMD3(0, 1, 0))
        let roll = simd_quatf(angle: reference.roll, axis: SIMD3(0, 0, 1))
        let light = SceneNode(
            transform: .init(translation: receiverPosition + reference.center, rotation: tilt * roll),
            light: try PunctualLight(kind: .rectangle(width: 1.7, height: 0.55), color: SIMD3(0.9, 0.65, 0.35), intensity: 6)
        )
        let view = simd_normalize(reference.view)
        let camera = SceneTransform(
            translation: receiverPosition + view * 3, rotation: simd_quatf(from: SIMD3(0, 0, -1), to: -view)
        )
        let graph = SceneGraph(roots: [receiver, light], camera: gpu.camera, cameraTransform: camera)
        let map = try material.receivesShadows ? DirectionalShadowMap(
            queue: gpu.queue, resolution: 16, directionToLight: SIMD3(0, 0, 1), halfExtent: 6, intensity: 0
        ) : nil
        // Exercise the table's lifetime on unretained draws. Shadow maps require retained references.
        let target = try gpu.makeTarget()
        let command = try #require(map == nil ? gpu.queue.makeCommandBufferWithUnretainedReferences() : gpu.queue.makeCommandBuffer())
        try graph.encode(pass: gpu.pass(color: target.color, depth: target.depth), commandBuffer: command, shadowMap: map)
        try gpu.finish(command)
        return try gpu.pixel(gpu.pixels(of: target.color), x: 32, y: 32)
    }
}
