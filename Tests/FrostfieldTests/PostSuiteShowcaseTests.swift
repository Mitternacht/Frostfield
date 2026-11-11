import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Renders a lit scene through the post-processing suite and times each effect.
/// Both tests run only when their environment variable is set, because they write
/// images and measure the GPU rather than assert behavior covered by other suites.
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PostSuiteShowcaseTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    private struct Scene {
        let graph: SceneGraph
        let selected: UUID
        let lamp: SIMD3<Float>
    }

    private func scene() throws -> Scene {
        let camera = try PerspectiveCamera(verticalFieldOfView: 0.6, near: 0.1, far: 60)
        func pbr(_ color: SIMD3<Float>, metallic: Float = 0, roughness: Float = 0.4, emissive: SIMD3<Float> = .zero,
                 strength: Float = 1) throws -> MeshMaterial {
            .physicallyBased(try PhysicallyBasedMaterial(
                metallic: metallic, roughness: roughness, directionToLight: SIMD3(-0.4, 0.8, -0.6), lightIntensity: 3,
                ambientIntensity: 0.25, baseColorFactor: SIMD4(color, 1), emissiveFactor: emissive,
                emissiveStrength: strength
            ))
        }
        func mesh(_ geometry: MeshGeometry, _ material: MeshMaterial) throws -> MeshRenderer {
            try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: material)
        }
        let sphere = try MeshGeometry.uvSphere(radius: 0.5, segments: 64, rings: 32)
        let ground = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, -0.5, -12), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))),
            mesh: try mesh(try .plane(width: 30, height: 40), try pbr(SIMD3(0.35, 0.33, 0.3), roughness: 0.8))
        )
        var roots = [ground]
        let colors: [SIMD3<Float>] = [SIMD3(0.8, 0.2, 0.15), SIMD3(0.9, 0.75, 0.3), SIMD3(0.2, 0.5, 0.8), SIMD3(0.3, 0.7, 0.35)]
        var selected = UUID()
        for (index, z) in [Float(-1.25), -2.4, -5, -10].enumerated() {
            let node = SceneNode(
                transform: SceneTransform(translation: SIMD3(Float(index) * 0.8 - 1.1, 0, z)),
                mesh: try mesh(sphere, try pbr(colors[index], metallic: index == 1 ? 1 : 0, roughness: index == 1 ? 0.25 : 0.45))
            )
            if index == 1 { selected = node.id }
            roots.append(node)
        }
        let lamp = SIMD3<Float>(1.6, 0.4, -7)
        roots.append(SceneNode(
            transform: SceneTransform(translation: lamp, scale: SIMD3(repeating: 0.3)),
            mesh: try mesh(sphere, try pbr(SIMD3(1, 0.9, 0.7), emissive: SIMD3(1, 0.85, 0.6), strength: 40))
        ))
        roots.append(SceneNode(
            transform: SceneTransform(translation: SIMD3(-2.6, 1.5, -16)),
            mesh: try mesh(try .bevelBox(size: SIMD3(1.2, 4, 1.2), bevel: 0.1), try pbr(SIMD3(0.5, 0.5, 0.55)))
        ))
        let graph = SceneGraph(roots: roots, camera: camera, cameraTransform: SceneTransform(translation: SIMD3(0, 0.4, 0)))
        return Scene(graph: graph, selected: selected, lamp: lamp)
    }

    private func renderScene(_ scene: Scene, width: Int, height: Int) throws -> (color: any MTLTexture, depth: any MTLTexture) {
        let color = try gpu.color(width: width, height: height)
        let depth = try gpu.depthTarget(width: width, height: height)
        try gpu.run { buffer in
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = color
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.35, green: 0.5, blue: 0.75, alpha: 1)
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = depth
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.clearDepth = 1
            pass.depthAttachment.storeAction = .store
            try scene.graph.encode(pass: pass, commandBuffer: buffer)
        }
        return (color, depth)
    }

    @Test("Showcase: a scene through bokeh, god rays, flare, bloom, outline, tone mapping and a LUT",
          .enabled(if: ProcessInfo.processInfo.environment["FROSTFIELD_POSTSUITE_IMAGES"] != nil))
    func showcase() throws {
        let (width, height) = (1440, 810)
        let scene = try scene()
        let frame = try renderScene(scene, width: width, height: height)
        let aspect = Float(width) / Float(height)
        let lens = try CameraLens(matching: scene.graph.camera, aspectRatio: aspect, fNumber: 1.2, focusDistance: 2.4)
        let bokeh = try BokehDepthOfField(device: gpu.device, lens: lens, camera: scene.graph.camera,
                                          settings: .init(maximumRadius: 16, sampleCount: 160, bladeCount: 6))
        let sun = try #require(try GodRays.screenPosition(
            ofDirection: SIMD3(-0.3, 0.3, -1), camera: scene.graph.camera, cameraTransform: scene.graph.cameraTransform,
            aspectRatio: aspect
        ))
        let rays = try GodRays(device: gpu.device, settings: .init(
            lightPosition: sun, color: SIMD3(1, 0.85, 0.6), intensity: 0.7, density: 0.9, decay: 0.965, samples: 48,
            source: .sun(radius: 0.18)
        ))
        let lamp = try #require(try LensFlare.Light(
            worldPosition: scene.lamp, camera: scene.graph.camera, cameraTransform: scene.graph.cameraTransform,
            aspectRatio: aspect, color: SIMD3(1, 0.85, 0.6), intensity: 1.5
        ))
        let flare = try LensFlare(device: gpu.device, lights: [lamp])
        let bloom = try MipmapBloom(device: gpu.device, settings: .init(strength: 0.06, radius: 0.85, levels: 7))
        let outline = try SelectionOutline(device: gpu.device, settings: .init(thickness: 1.5))
        let warm = try ColorLookupTable(size: 33) { c in
            simd_clamp(SIMD3(c.x * 1.06 + 0.02, c.y * 1.0 + 0.01, c.z * 0.9 + 0.03), .zero, .one)
        }
        let hdr = try PostProcessChain(queue: gpu.queue, effects: [
            .pass(bokeh), .pass(rays), .pass(flare), .pass(bloom), .pass(outline),
        ])
        let display = try HDRDisplay(device: gpu.device, destinationPixelFormat: .rgba16Float)
        let grade = try PostProcessChain(queue: gpu.queue, effects: [
            .pass(try ColorLookupPass(device: gpu.device, table: warm)), .graph(try .vignette(offset: 1, darkness: 1.1)),
        ])
        let graded = try gpu.color(width: width, height: height)
        let mapped = try gpu.color(width: width, height: height)
        let final = try gpu.color(width: width, height: height)
        try gpu.run { buffer in
            try outline.encodeSelection(graph: scene.graph, selection: [scene.selected], width: width, height: height, commandBuffer: buffer)
            try hdr.encode(source: frame.color, depth: frame.depth, destination: graded, commandBuffer: buffer)
            try display.encode(source: graded, destination: mapped, commandBuffer: buffer)
            try grade.encode(source: mapped, destination: final, commandBuffer: buffer)
        }
        let plain = try gpu.color(width: width, height: height)
        try gpu.run { try display.encode(source: frame.color, destination: plain, commandBuffer: $0) }
        gpu.savePNG(try gpu.read(final), width: width, height: height, name: "showcase")
        gpu.savePNG(try gpu.read(plain), width: width, height: height, name: "showcase-plain")
    }

    @Test("Timing: GPU time per effect at 1920 × 1080",
          .enabled(if: ProcessInfo.processInfo.environment["FROSTFIELD_POSTSUITE_TIMING"] != nil))
    func timing() throws {
        let (width, height) = (1920, 1080)
        let scene = try scene()
        let frame = try renderScene(scene, width: width, height: height)
        let destination = try gpu.color(width: width, height: height)
        let camera = scene.graph.camera
        let lens = try CameraLens(matching: camera, aspectRatio: 16.0 / 9, fNumber: 1.4, focusDistance: 3)
        let outline = try SelectionOutline(device: gpu.device)
        let effects: [(String, any PostProcessEffect)] = [
            ("GaussianBlur sigma 4", try GaussianBlur(device: gpu.device, settings: .init(sigma: 4))),
            ("KawaseBlur medium, half resolution", try KawaseBlur(device: gpu.device)),
            ("BilateralBlur sigma 3", try BilateralBlur(device: gpu.device, camera: camera)),
            ("MipmapBloom 6 levels", try MipmapBloom(device: gpu.device)),
            ("MipmapBloom 6 levels, threshold 1", try MipmapBloom(device: gpu.device, settings: .init(threshold: 1))),
            ("KawaseBlur medium, quarter resolution", try KawaseBlur(device: gpu.device, settings: .init(resolutionScale: 0.25))),
            ("BokehDepthOfField 96 taps, radius 16", try BokehDepthOfField(device: gpu.device, lens: lens, camera: camera)),
            ("GodRays 2 x 32 taps, half resolution", try GodRays(device: gpu.device, settings: .init(lightPosition: SIMD2(0.3, 0.2)))),
            ("LensFlare 1 light, 6 ghosts", try LensFlare(device: gpu.device, lights: [LensFlare.Light(position: SIMD2(0.7, 0.3))])),
            ("SelectionOutline thickness 2", outline),
            ("ColorLookupPass 33^3 tetrahedral", try ColorLookupPass(device: gpu.device, table: try .identity(size: 33))),
            ("Afterimage", try Afterimage(device: gpu.device)),
        ]
        var lines: [String] = []
        for (name, effect) in effects {
            var times: [Double] = []
            for run in 0..<9 {
                let buffer = try #require(gpu.queue.makeCommandBuffer())
                if effect === outline {
                    try outline.encodeSelection(graph: scene.graph, selection: [scene.selected], width: width, height: height,
                                                commandBuffer: buffer)
                }
                try effect.encode(source: frame.color, depth: frame.depth, destination: destination, commandBuffer: buffer)
                buffer.commit()
                buffer.waitUntilCompleted()
                try #require(buffer.status == .completed)
                if run >= 2 { times.append((buffer.gpuEndTime - buffer.gpuStartTime) * 1000) }
            }
            times.sort()
            lines.append(String(format: "%@: median %.3f ms (min %.3f, max %.3f)", name, times[times.count / 2], times[0], times[times.count - 1]))
        }
        for line in lines { print("POSTSUITE-TIMING \(line)") }
    }
}
