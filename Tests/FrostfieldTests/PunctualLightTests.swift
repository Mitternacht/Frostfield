import Foundation
import Frostfield
import Metal
import simd
import Testing

struct PunctualLightTests {
    @Test("Punctual inputs and the scene light count are bounded")
    func validation() throws {
        #expect(throws: PunctualLight.Failure.color) {
            try PunctualLight(kind: .directional, color: SIMD3(1, 1.1, 1))
        }
        #expect(throws: PunctualLight.Failure.intensity) {
            try PunctualLight(kind: .directional, intensity: .infinity)
        }
        #expect(throws: PunctualLight.Failure.range) {
            try PunctualLight(kind: .point(range: 0))
        }
        #expect(throws: PunctualLight.Failure.cone) {
            try PunctualLight(kind: .spot(innerConeAngle: 0.3, outerConeAngle: 0.3))
        }
        let light = try PunctualLight(kind: .point())
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 20)
        let maximum = SceneGraph.maximumPunctualLightCount
        var graph = SceneGraph(roots: (0..<maximum).map { _ in SceneNode(light: light) }, camera: camera)
        #expect(try graph.resolve().count == maximum)
        graph.roots.append(SceneNode(light: light))
        #expect(throws: SceneGraph.ValidationError.tooManyLights(maximum: maximum)) { try graph.resolve() }
    }

    @Test("Node hierarchy transforms a light's position and aim while preserving its properties")
    func resolvesThroughHierarchy() throws {
        let light = try PunctualLight(kind: .spot(range: 5), color: SIMD3(0.1, 0.3, 1), intensity: 12)
        let child = SceneNode(
            transform: SceneTransform(
                translation: SIMD3(1, 0, 1), rotation: simd_quatf(angle: .pi / 4, axis: SIMD3(0, 1, 0))
            ),
            light: light
        )
        let parent = SceneNode(
            transform: SceneTransform(translation: SIMD3(2, 3, 4), scale: SIMD3(-2, 3, -4)),
            children: [child]
        )
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 20)
        let graph = SceneGraph(roots: [parent], camera: camera)
        let resolved = try graph.resolve()
        #expect(resolved[0].light == nil)
        let world = try #require(resolved[1].light)
        #expect(world.position == SIMD3(0, 3, 0))
        #expect(simd_distance(world.direction, simd_normalize(SIMD3<Float>(1, 0, 2))) < 0.00001)
        #expect(world.light == light)
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PunctualLightRenderingTests {
    @Test("Colored directional, point and spot lights add physical falloff to PBR")
    func shading() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let geometry = try MeshGeometry(
            vertices: [SIMD3<Float>(-1, -1, -3), SIMD3(1, -1, -3), SIMD3(0, 1, -3)].map {
                MeshVertex(position: $0, normal: SIMD3(0, 0, 1))
            },
            indices: [0, 1, 2]
        )
        let material = try PhysicallyBasedMaterial(roughness: 0.8, lightIntensity: 0, ambientIntensity: 0)
        let renderer = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: geometry, material: .physicallyBased(material)
        )
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 20)
        let receiver = SceneNode(mesh: renderer)
        let side = 33
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

        func render(_ lights: [SceneNode]) throws -> SIMD4<Float> {
            let graph = SceneGraph(roots: [receiver] + lights, camera: camera)
            let buffer = try #require(queue.makeCommandBuffer())
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = color
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = depth
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.clearDepth = 1
            try graph.encode(pass: pass, commandBuffer: buffer)
            buffer.commit()
            buffer.waitUntilCompleted()
            try #require(buffer.status == .completed, "\(buffer.error?.localizedDescription ?? "Punctual light draw failed.")")
            var pixel = SIMD4<Float16>.zero
            withUnsafeMutablePointer(to: &pixel) {
                color.getBytes(
                    $0, bytesPerRow: MemoryLayout<SIMD4<Float16>>.stride,
                    from: MTLRegionMake2D(side / 2, side / 2, 1, 1), mipmapLevel: 0
                )
            }
            return SIMD4<Float>(pixel)
        }

        func point(_ position: SIMD3<Float>, range: Float? = nil) throws -> SceneNode {
            SceneNode(
                transform: SceneTransform(translation: position),
                light: try PunctualLight(kind: .point(range: range), color: SIMD3(1, 0, 0), intensity: 4)
            )
        }

        let unlit = try render([])
        let near = try render([point(SIMD3(0, 0, -2))])
        let far = try render([point(SIMD3(0, 0, -1))])
        let outside = try render([point(SIMD3(0, 0, -1), range: 1.5)])
        let inRange = try render([point(SIMD3(0, 0, -2), range: 2)])
        #expect(unlit == SIMD4(0, 0, 0, 1))
        #expect(near.x > 1 && near.y == 0 && near.z == 0)
        #expect(abs(near.x - 4 * far.x) < 0.01)
        #expect(outside == unlit)
        #expect(abs(inRange.x - near.x * 0.9375) < 0.01)

        let sun = SceneNode(
            transform: SceneTransform(translation: SIMD3(-40, 30, -25)),
            light: try PunctualLight(kind: .directional, color: SIMD3(0, 1, 0), intensity: 2)
        )
        let mixed = try render([point(SIMD3(0, 0, -2)), sun])
        #expect(abs(mixed.x - near.x) < 0.01 && mixed.y > 0.5 && mixed.z == 0)

        let offAxis = SIMD3<Float>(1 / sqrt(3), 0, -2)
        let spot = try PunctualLight(
            kind: .spot(innerConeAngle: .pi / 12, outerConeAngle: .pi / 4),
            color: SIMD3(1, 0, 0), intensity: 4
        )
        let spotNode = SceneNode(transform: SceneTransform(translation: offAxis), light: spot)
        let penumbra = try render([spotNode])
        let barePoint = try render([point(offAxis)])
        let cone = (cos(Float.pi / 6) - cos(Float.pi / 4)) / (cos(Float.pi / 12) - cos(Float.pi / 4))
        #expect(abs(penumbra.x - barePoint.x * cone * cone) < 0.015)
        let facingAway = SceneNode(
            transform: SceneTransform(
                translation: offAxis, rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
            ),
            light: spot
        )
        #expect(try render([facingAway]) == unlit)
    }
}
