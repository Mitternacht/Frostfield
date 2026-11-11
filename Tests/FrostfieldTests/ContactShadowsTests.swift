import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ContactShadowsTests {
    @Test("A contact layer darkens a receiver by closest height, blurs the footprint and respects scene depth",
          arguments: [MeshDepthMode.forward, .reversed])
    func floorContact(mode: MeshDepthMode) async throws {
        let gpu = try RasterLightingFixture(width: 96, height: 96)
        let horizontal = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
        let tilt = simd_quatf(angle: 0.41, axis: simd_normalize(SIMD3<Float>(1, 0.3, 0.4)))
        let pose = SceneTransform(translation: SIMD3(0.2, -0.7, 0.3), rotation: mode == .forward ? horizontal : tilt * horizontal)
        let extent = SIMD2<Float>(2.5, 2)
        let shadows = try ContactShadows(queue: gpu.queue, plane: pose, size: extent, maximumHeight: 1, resolution: 96)
        let floorColor = SIMD4<Float>(0.75, 0.625, 0.5, 1)
        let tint = SIMD3<Float>(0.125, 0.25, 0.375)
        let colorFormat = MTLPixelFormat.rgba16Float
        let unlit = MeshMaterial.physicallyBased(try PhysicallyBasedMaterial(shadingModel: .unlit))
        let floor = try MeshRenderer(device: gpu.device, colorPixelFormat: colorFormat,
                                     geometry: .plane(width: extent.x, height: extent.y), material: unlit, depthMode: mode)
        let floorNode = SceneNode(transform: pose, mesh: floor, materialParameters: floorColor, castsShadow: false)
        let receiver = try shadows.makeReceiver(colorPixelFormat: colorFormat, depthMode: mode, tint: tint, opacity: 0.75, offset: 0.001)
        let caster = try MeshRenderer(device: gpu.device, colorPixelFormat: colorFormat,
                                      geometry: .box(width: 0.44, height: 0.3, depth: 0.24), material: unlit, depthMode: mode)
        let footprint = SIMD2<Float>(-0.48, 0.31)
        func box(_ xy: SIMD2<Float>, height: Float, castsShadow: Bool = true, visible: Bool = true) -> SceneNode {
            SceneNode(
                transform: .init(translation: pose.translation + pose.rotation.act(SIMD3(xy, height)), rotation: pose.rotation),
                mesh: caster, materialParameters: SIMD4(0.25, 0.5, 0.875, 1), castsShadow: castsShadow, isVisible: visible
            )
        }
        let closest = box(footprint, height: 0.17)
        let elevated = box(footprint, height: 0.59)
        let upper = SIMD2<Float>(0.62, 0.31), lower = SIMD2<Float>(0.62, -0.32)
        let ignored = SIMD2<Float>(-0.48, -0.45), hidden = SIMD2<Float>(0, -0.45)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 10)
        let scene = SceneGraph(roots: [
            floorNode, receiver, closest, elevated, box(upper, height: 1.2), box(lower, height: -0.25),
            box(ignored, height: 0.18, castsShadow: false), box(hidden, height: 0.18, visible: false),
        ], camera: camera)
        let eye = pose.translation + pose.rotation.act(SIMD3(0, 0, 3))
        let view = simd_inverse(SceneTransform(translation: eye, rotation: pose.rotation).matrix)
        var projection = simd_float4x4(diagonal: SIMD4(2 / extent.x, 2 / extent.y, mode == .forward ? -0.125 : 0.125, 1))
        if mode == .reversed { projection[3].z = 1 }
        let color = try gpu.colorTarget(), depth = try gpu.depthTarget()
        func render(
            captures: Bool, showReceiver: Bool, blur: Float = 0, excluding: Set<UUID> = [], occluder: Bool = false
        ) throws -> (Int, [SIMD4<Float>]) {
            var draws = 0
            var sceneView = SceneGraph(roots: [floorNode], camera: camera)
            if showReceiver { sceneView.roots.append(receiver) }
            if occluder { sceneView.roots.append(closest) }
            try gpu.run { command in
                if captures {
                    draws = try shadows.encode(
                        graph: scene, settings: .init(blurSigma: blur), excluding: excluding, commandBuffer: command
                    )
                }
                try sceneView.encode(
                    pass: gpu.pass(color: color, depth: depth, depthMode: mode), commandBuffer: command,
                    clipFromWorld: projection * view, cameraPosition: eye, depthMode: mode
                )
            }
            return (draws, try gpu.pixels(of: color))
        }
        func pixel(_ pixels: [SIMD4<Float>], _ xy: SIMD2<Float>) -> SIMD4<Float> {
            let x = Int((xy.x / extent.x + 0.5) * Float(gpu.width))
            let y = Int((0.5 - xy.y / extent.y) * Float(gpu.height))
            return pixels[y * gpu.width + x]
        }
        func shaded(_ coverage: Float) -> SIMD4<Float> {
            floorColor * (1 - coverage * 0.75) + SIMD4(tint, 1) * (coverage * 0.75)
        }

        let sharp = try render(captures: true, showReceiver: true)
        let receiverDepth = try await TextureTransfer.read(from: depth, commandQueue: gpu.queue).bytes
        #expect(sharp.0 == 4)
        #expect(simd_reduce_max(abs(pixel(sharp.1, footprint) - shaded(0.95))) < 0.002)
        for sample in [upper, lower, ignored, hidden, SIMD2(-footprint.x, footprint.y), SIMD2(footprint.x, -footprint.y)] {
            #expect(pixel(sharp.1, sample) == floorColor)
        }
        let outside = footprint + SIMD2(0.29, 0)
        #expect(pixel(sharp.1, outside) == floorColor)
        let soft = try render(captures: true, showReceiver: true, blur: 4)
        #expect(pixel(soft.1, outside).x < floorColor.x - 0.04)
        #expect(pixel(soft.1, footprint).x < floorColor.x - 0.2)
        #expect(soft.1.allSatisfy { $0.w == 1 })
        let raised = try render(captures: true, showReceiver: true, excluding: [closest.id])
        #expect(raised.0 == 3)
        #expect(simd_reduce_max(abs(pixel(raised.1, footprint) - shaded(0.53))) < 0.002)
        let empty = try render(captures: true, showReceiver: true, excluding: [closest.id, elevated.id])
        #expect(pixel(empty.1, footprint) == floorColor)
        let baseline = try render(captures: true, showReceiver: false)
        #expect(pixel(baseline.1, footprint) == floorColor)
        let floorDepth = try await TextureTransfer.read(from: depth, commandQueue: gpu.queue).bytes
        #expect(floorDepth == receiverDepth)
        let blocked = try render(captures: false, showReceiver: true, occluder: true)
        #expect(pixel(blocked.1, footprint) == closest.materialParameters)
    }

    @Test("Contact shadow captures graph cutouts and rejects wrong queues and settings")
    func cutoutAndValidation() async throws {
        let gpu = try RasterLightingFixture(width: 48, height: 48)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 10)
        let shadows = try ContactShadows(queue: gpu.queue, size: SIMD2(repeating: 2), maximumHeight: 1, resolution: 48)
        let graph = try SurfaceShaderGraph { input in
            .init(mask: input.uv0.x.lessThan(input.constant(Float(0.5))))
        }
        let material = try PhysicallyBasedMaterial(alphaMode: .mask(cutoff: 0.5), surfaceGraph: graph)
        let renderer = try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float,
                                        geometry: .plane(width: 1.5, height: 1.5), material: .physicallyBased(material))
        let node = SceneNode(
            transform: .init(translation: SIMD3(0, 0, 0.2), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))),
            mesh: renderer
        )
        let scene = SceneGraph(roots: [node], camera: camera)
        try gpu.run { command in
            let count = try shadows.encode(graph: scene, settings: .init(blurSigma: 0), commandBuffer: command)
            #expect(count == 1)
        }
        let bytes = try await TextureTransfer.read(from: shadows.texture, commandQueue: gpu.queue).bytes
        func sample(_ x: Int) -> SIMD4<Float> {
            bytes.withUnsafeBytes { raw in SIMD4<Float>(raw.loadUnaligned(fromByteOffset: (24 * 48 + x) * 8, as: SIMD4<Float16>.self)) }
        }
        #expect(simd_reduce_max(abs(sample(14) - SIMD4<Float>(repeating: 0.8))) < 0.001)
        #expect(sample(34) == .zero)

        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: ContactShadows.Failure.settings) {
            try shadows.encode(graph: scene, settings: .init(blurSigma: .infinity), commandBuffer: command)
        }
        let otherQueue = try #require(gpu.device.makeCommandQueue())
        let unrelated = try #require(otherQueue.makeCommandBuffer())
        #expect(throws: ContactShadows.Failure.commandBuffer) {
            try shadows.encode(graph: scene, commandBuffer: unrelated)
        }
    }
}
