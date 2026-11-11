import Foundation
import Frostfield
import Metal
import Testing
import simd

/// GPU durations behind docs/evidence/rasterlight.md. Opt in with
/// `TEST_RUNNER_FROSTFIELD_BENCHMARKS=1` and run without the Metal debug layer; each figure
/// is the median command-buffer GPU time of 20 runs after 5 warm-up runs.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["FROSTFIELD_BENCHMARKS"] != nil, "Set FROSTFIELD_BENCHMARKS to measure"))
struct RasterLightingBenchmarks {
    @Test("Raster lighting passes at 1920×1080")
    func passDurations() throws {
        let fixture = try RasterLightingFixture(width: 1920, height: 1080)
        let device = fixture.device
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 200)
        let cameraTransform = RasterLightingFixture.cameraTransform(position: SIMD3(0, 1.5, 2.5), pitch: -atan2(1.5, 2.5))
        func median(_ body: (any MTLCommandBuffer) throws -> Void) throws -> Double {
            var times: [Double] = []
            for run in 0..<25 {
                let buffer = try #require(fixture.queue.makeCommandBuffer())
                try body(buffer)
                buffer.commit()
                buffer.waitUntilCompleted()
                if run >= 5 { times.append((buffer.gpuEndTime - buffer.gpuStartTime) * 1000) }
            }
            return times.sorted()[times.count / 2]
        }
        let material = try PhysicallyBasedMaterial(metallic: 0.2, roughness: 0.3, lightIntensity: 1, ambientIntensity: 0.1)
        let plane = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 20, height: 20, columns: 8, rows: 8),
            material: .physicallyBased(material)
        )
        let box = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .bevelBox(bevel: 0.05), material: .physicallyBased(material)
        )
        var roots = [
            SceneNode(
                transform: SceneTransform(translation: SIMD3(0, 0, 10), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))),
                mesh: plane
            ),
            SceneNode(transform: SceneTransform(translation: SIMD3(0, 10, 0)), mesh: plane),
        ]
        for index in 0..<6 {
            roots.append(SceneNode(transform: SceneTransform(translation: SIMD3(Float(index) - 2.5, 0.5, 1)), mesh: box))
        }
        var graph = SceneGraph(roots: roots, camera: camera)
        graph.cameraTransform = cameraTransform

        let attributes = try fixture.texture(SceneGraph.surfaceAttributesPixelFormat, usage: [.renderTarget, .shaderRead])
        let prepassDepth = try fixture.depthTarget()
        let color = try fixture.texture(.rgba16Float, usage: [.renderTarget, .shaderRead])
        let depth = try fixture.depthTarget()
        let visibility = try fixture.texture(.rgba16Float, usage: [.renderTarget, .shaderRead])
        let output = try fixture.texture(.rgba16Float, usage: [.renderTarget, .shaderRead])
        let attributePass = MTLRenderPassDescriptor()
        attributePass.colorAttachments[0].texture = attributes
        attributePass.colorAttachments[0].loadAction = .clear
        attributePass.colorAttachments[0].storeAction = .store
        attributePass.depthAttachment.texture = prepassDepth
        attributePass.depthAttachment.loadAction = .clear
        attributePass.depthAttachment.storeAction = .store
        attributePass.depthAttachment.clearDepth = 1
        let colorPass = fixture.pass(color: color, depth: depth)

        let prepass = try median { try graph.encodeSurfaceAttributes(pass: attributePass, commandBuffer: $0) }
        let shading = try median { try graph.encode(pass: colorPass, commandBuffer: $0) }
        let occlusion = try AmbientOcclusion(device: device)
        let gtaoDefault = try median {
            try occlusion.encodeVisibility(
                depth: prepassDepth, surfaceAttributes: attributes, destination: visibility, camera: camera,
                cameraTransform: cameraTransform, settings: .init(radius: 0.5, strength: 1), commandBuffer: $0
            )
        }
        let gtaoHigh = try median {
            try occlusion.encodeVisibility(
                depth: prepassDepth, surfaceAttributes: attributes, destination: visibility, camera: camera,
                cameraTransform: cameraTransform,
                settings: .init(radius: 0.5, strength: 1, sliceCount: 8, stepCount: 32, denoise: false), commandBuffer: $0
            )
        }
        for (slices, steps, denoise) in [(3, 8, false), (3, 2, false), (1, 2, false), (1, 2, true), (3, 4, true), (2, 4, true)] {
            let time = try median {
                try occlusion.encodeVisibility(
                    depth: prepassDepth, surfaceAttributes: attributes, destination: visibility, camera: camera,
                    cameraTransform: cameraTransform,
                    settings: .init(radius: 0.5, strength: 1, sliceCount: slices, stepCount: steps, denoise: denoise), commandBuffer: $0
                )
            }
            #expect(time > 0)
            print(String(format: "BENCHMARK GTAO %d×%d per side, filter %@: %.2f ms", slices, steps, denoise ? "on" : "off", time))
        }
        let shadedOccluded = try median {
            try graph.encode(pass: colorPass, commandBuffer: $0, screenSpaceOcclusion: visibility)
        }
        let reflections = try ScreenSpaceReflections(device: device)
        let ssr = try median {
            try reflections.encode(
                source: color, depth: depth, surfaceAttributes: attributes, destination: output, camera: camera,
                cameraTransform: cameraTransform, commandBuffer: $0
            )
        }

        let shadowMaterial = try PhysicallyBasedMaterial(roughness: 0.5, receivesShadows: true)
        let shadowPlane = try MeshRenderer(
            device: device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 200, height: 200), material: .physicallyBased(shadowMaterial)
        )
        var shadowGraph = graph
        shadowGraph.roots[0].mesh = shadowPlane
        let light = simd_normalize(SIMD3<Float>(1, 2, 0.5))
        let single = try DirectionalShadowMap(queue: fixture.queue, resolution: 4096, directionToLight: light, center: SIMD3(0, 0, -20), halfExtent: 40)
        let cascaded = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 2048, directionToLight: light, cascades: .init(count: 4, maximumDistance: 80)
        )
        try cascaded.fitCascades(camera: camera, cameraTransform: cameraTransform, aspectRatio: 1920.0 / 1080.0)
        let singleShadow = try median { try shadowGraph.encode(pass: colorPass, commandBuffer: $0, shadowMap: single) }
        let cascadedShadow = try median { try shadowGraph.encode(pass: colorPass, commandBuffer: $0, shadowMap: cascaded) }
        let soft = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 2048, directionToLight: light, cascades: .init(count: 4, maximumDistance: 80),
            lightAngularRadius: 0.02
        )
        try soft.fitCascades(camera: camera, cameraTransform: cameraTransform, aspectRatio: 1920.0 / 1080.0)
        let cascadedSoft = try median { try shadowGraph.encode(pass: colorPass, commandBuffer: $0, shadowMap: soft) }
        #expect([prepass, shading, shadedOccluded, gtaoDefault, gtaoHigh, ssr, singleShadow, cascadedShadow].allSatisfy { $0 > 0 })
        print(String(
            format: "BENCHMARK %@ 1920×1080: attribute prepass %.2f ms, PBR color %.2f, PBR color reading GTAO %.2f, GTAO default %.2f, GTAO 8×32 unfiltered %.2f, SSR default %.2f, single 4096² shadow + color %.2f, 4 × 2048² cascades + color %.2f, cascades with contact hardening + color %.2f",
            device.name, prepass, shading, shadedOccluded, gtaoDefault, gtaoHigh, ssr, singleShadow, cascadedShadow, cascadedSoft
        ))
    }
}
