import Foundation
@testable import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ScreenSpaceGlobalIlluminationTests {
    private let gpu: RasterLightingFixture
    private let camera: PerspectiveCamera
    private let transform = SceneTransform(
        translation: SIMD3(0, 1.5, 0), rotation: simd_quatf(angle: -0.08, axis: SIMD3(1, 0, 0))
    )
    private let albedo = SIMD4<Float>(0.4, 0.8, 0.6, 1)

    init() throws {
        gpu = try RasterLightingFixture(width: 128, height: 128)
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 35)
    }

    @Test("An emissive wall bounces red light onto the facing floor with three normal sources",
          arguments: [MeshDepthMode.forward, .reversed], [Float(1), 0.5])
    func diffuseBounce(mode: MeshDepthMode, scale: Float) throws {
        let layout = try SurfaceRenderTargets(outputs: [
            .init("normal", value: .worldNormal, pixelFormat: .rgba16Float),
            .init("albedo", value: .baseColor, pixelFormat: .rgba16Float),
        ])
        var graph = try scene(layout: layout, mode: mode)
        graph.projectionJitter = mode == .reversed ? SIMD2(0.35, -0.25) : .zero
        let explicit = try graphCamera(mode, jitter: graph.projectionJitter)
        let beauty = try gpu.colorTarget(), normals = try gpu.colorTarget(), base = try gpu.colorTarget()
        let depth = try gpu.depthTarget(), packedDepth = try gpu.depthTarget()
        let packed = try gpu.texture(.rg32Uint, usage: [.renderTarget, .shaderRead])
        let indirect = try gpu.colorTarget(), packedIndirect = try gpu.colorTarget(), depthIndirect = try gpu.colorTarget()
        let tooFar = try gpu.colorTarget(), composite = try gpu.colorTarget()
        let ssgi = try ScreenSpaceGlobalIllumination(device: gpu.device)
        var settings = ScreenSpaceGlobalIllumination.Settings(radius: 2.2, thickness: 0.32, sliceCount: 5, stepCount: 16, intensity: 6, resolutionScale: scale)
        try gpu.run { command in
            let primary = gpu.pass(color: beauty, depth: depth, depthMode: mode)
            let pass = try layout.makePass(colorAttachments: [
                SurfaceRenderTargets.colorName: primary.colorAttachments[0]!,
                "normal": gpu.pass(color: normals, depth: depth).colorAttachments[0]!,
                "albedo": gpu.pass(color: base, depth: depth).colorAttachments[0]!,
            ], depthAttachment: primary.depthAttachment!)
            try graph.encode(pass: pass, commandBuffer: command, depthMode: mode)
            try graph.encodeSurfaceAttributes(pass: gpu.pass(color: packed, depth: packedDepth, depthMode: mode),
                                              commandBuffer: command, depthMode: mode)
            try ssgi.encode(source: beauty, depth: depth, normals: .worldNormals(normals), destination: indirect,
                            camera: explicit, depthMode: mode, settings: settings, commandBuffer: command)
            try ssgi.encode(source: beauty, depth: depth, normals: .surfaceAttributes(packed), destination: packedIndirect,
                            camera: explicit, depthMode: mode, settings: settings, commandBuffer: command)
            try ssgi.encode(source: beauty, depth: depth, destination: depthIndirect,
                            camera: explicit, depthMode: mode, settings: settings, commandBuffer: command)
            try ssgi.composite(source: beauty, illumination: indirect, diffuseAlbedo: base, destination: composite, commandBuffer: command)
            settings.radius = 0.25
            settings.denoise = false
            try ssgi.encode(source: beauty, depth: depth, normals: .worldNormals(normals), destination: tooFar,
                            camera: explicit, depthMode: mode, settings: settings, commandBuffer: command)
        }

        if scale < 1 {
            #expect(ssgi.traceTextureSize == SIMD2(gpu.width / 2, gpu.height / 2))
        }
        let near = pixel(SIMD3(0, 0, -4.05), camera: explicit)
        let far = pixel(SIMD3(0, 0, -2.6), camera: explicit)
        let beautyPixels = try gpu.pixels(of: beauty)
        let lightPixels = try gpu.pixels(of: indirect)
        let reflected = mean(lightPixels, at: near)
        let nearBeauty = mean(beautyPixels, at: near)
        let merged = try mean(gpu.pixels(of: composite), at: near)
        #expect(nearBeauty == SIMD4(0, 0, 0, 1))
        #expect(reflected.x > 0.07, "Near GI \(reflected)")
        #expect(reflected.x > 8 * max(reflected.y, reflected.z))
        #expect(reflected.w < 0.99 && reflected.w > 0.1)
        #expect(abs(merged.x - reflected.x * albedo.x) < 0.002 && merged.w == 1)
        #expect(try abs(mean(gpu.pixels(of: packedIndirect), at: near).x - reflected.x) < 0.015)
        #expect(try mean(gpu.pixels(of: depthIndirect), at: near).x > 0.05)
        #expect(try mean(gpu.pixels(of: tooFar), at: near).x < 0.001)
        let open = mean(lightPixels, at: far)
        #expect(open.x < 0.001 && open.w > 0.94, "Open floor \(open)")
        let uncovered = beautyPixels.indices.filter { beautyPixels[$0].w == 0 }
        #expect(uncovered.count > 100)
        #expect(uncovered.allSatisfy { lightPixels[$0] == SIMD4(0, 0, 0, 1) })
        #expect(try gpu.pixels(of: composite)[0] == .zero)
    }

    @Test("Temporal GI reprojects prior bounce and drops it at disocclusion",
          arguments: [MeshDepthMode.forward, .reversed])
    func temporalHistory(mode: MeshDepthMode) throws {
        let layout = try SurfaceRenderTargets(outputs: [.init("normal", value: .worldNormal, pixelFormat: .rgba16Float)])
        let beauty = try gpu.colorTarget(), normals = try gpu.colorTarget(), depth = try gpu.depthTarget()
        let output = try gpu.colorTarget()
        let ssgi = try ScreenSpaceGlobalIllumination(device: gpu.device)
        let settings = ScreenSpaceGlobalIllumination.Settings(
            radius: 8, thickness: 0.08, sliceCount: 5, stepCount: 16, intensity: 6,
            resolutionScale: 0.5, screenSpaceSampling: true, linearThickness: true, temporal: .init(historyWeight: 0.85)
        )
        func frame(
            emission: SIMD3<Float>, floorHeight: Float, motion: Float, jitter: SIMD2<Float>
        ) throws -> (view: ShaderGraphCamera, pixels: [SIMD4<Float>]) {
            let moved = SceneTransform(translation: transform.translation + SIMD3(motion, 0, 0), rotation: transform.rotation)
            var graph = try scene(layout: layout, mode: mode, emission: emission, floorHeight: floorHeight, pose: moved)
            graph.projectionJitter = jitter
            let explicit = try graphCamera(mode, jitter: jitter, pose: moved)
            try gpu.run { command in
                let primary = gpu.pass(color: beauty, depth: depth, depthMode: mode)
                let pass = try layout.makePass(colorAttachments: [
                    SurfaceRenderTargets.colorName: primary.colorAttachments[0]!,
                    "normal": gpu.pass(color: normals, depth: depth).colorAttachments[0]!,
                ], depthAttachment: primary.depthAttachment!)
                try graph.encode(pass: pass, commandBuffer: command, depthMode: mode)
                try ssgi.encode(
                    source: beauty, depth: depth, normals: .worldNormals(normals), destination: output,
                    camera: explicit, depthMode: mode, settings: settings, commandBuffer: command
                )
            }
            let pixels = try gpu.pixels(of: output)
            #expect(pixels[0] == SIMD4(0, 0, 0, 1))
            return (explicit, pixels)
        }
        func receiver(_ frame: (view: ShaderGraphCamera, pixels: [SIMD4<Float>]), at height: Float = 0) -> SIMD4<Float> {
            mean(frame.pixels, at: pixel(SIMD3(0, height, -4.05), camera: frame.view))
        }

        let emission = SIMD3<Float>(1, 0.04, 0.01)
        let first = try frame(emission: emission, floorHeight: 0, motion: 0, jitter: .zero)
        let carried = try frame(emission: .zero, floorHeight: 0, motion: 0.06, jitter: SIMD2(0.35, -0.25))
        let bounced = receiver(first)
        #expect(bounced.x > 0.04, "First bounce \(bounced)")
        let carriedPixel = pixel(SIMD3(0, 0, -4.05), camera: carried.view)
        let fromFirst = reprojectedFloorRed(first.pixels, previous: first.view, current: carried.view, center: carriedPixel)
        let reprojection = receiver(carried)
        #expect(fromFirst > 0.04 && abs(reprojection.x - 0.85 * fromFirst) < 0.003,
                "Reprojected bounce \(reprojection.x), predicted \(0.85 * fromFirst)")
        let uncovered = try frame(emission: .zero, floorHeight: -0.45, motion: 0.06, jitter: SIMD2(-0.25, 0.4))
        let exposed = receiver(uncovered, at: -0.45)
        #expect(exposed.x < 0.001, "Disocclusion carried stale bounce \(exposed)")
        _ = try frame(emission: emission, floorHeight: 0, motion: 0, jitter: .zero)
        ssgi.resetTemporalHistory()
        let reset = try frame(emission: .zero, floorHeight: 0, motion: 0, jitter: .zero)
        #expect(receiver(reset).x < 0.001)
    }

    @Test("Raster-format motion tracks a moving bounce receiver and rejects occluded and invalid history",
          arguments: [MeshDepthMode.forward, .reversed])
    func movingReceiver(mode: MeshDepthMode) throws {
        let layout = try SurfaceRenderTargets(outputs: [.init("normal", value: .worldNormal, pixelFormat: .rgba16Float)])
        let beauty = try gpu.colorTarget(), normals = try gpu.colorTarget(), depth = try gpu.depthTarget()
        let output = try gpu.colorTarget(), vectors = try gpu.colorTarget()
        let ssgi = try ScreenSpaceGlobalIllumination(device: gpu.device)
        let settings = ScreenSpaceGlobalIllumination.Settings(
            radius: 4, thickness: 0.32, sliceCount: 5, stepCount: 20, intensity: 6,
            resolutionScale: 0.5, temporal: .init()
        )
        var before = try scene(layout: layout, mode: mode)
        before.projectionJitter = SIMD2(0.125, -0.375)
        let floor = before.roots[0]
        let receiver = SIMD3<Float>(-0.55, 0, -4.05)
        let hidden = SIMD3<Float>(0.72, 0, -4.05)
        // Put a parallel blocker at a different depth on the ray to the old receiver.
        let blocker = transform.translation + (hidden - transform.translation) * 0.78
        before.roots.append(.init(transform: .init(
            translation: blocker, rotation: floor.transform.rotation, scale: SIMD3(0.19, 0.08, 1)
        ), mesh: floor.mesh))
        var after = try scene(layout: layout, mode: mode, emission: .zero)
        after.projectionJitter = SIMD2(-0.375, 0.25)
        let displacement = SIMD3<Float>(0.48, 0.31, 0.20)
        after.roots[0] = floor
        after.roots[0].transform.translation += displacement
        let previous = try graphCamera(mode, jitter: before.projectionJitter)
        let current = try graphCamera(mode, jitter: after.projectionJitter)
        let emptyMotion = [SIMD4<Float16>](repeating: .zero, count: gpu.width * gpu.height)
        let authored = movingFloorVectors(from: previous, to: current, displacement: displacement)

        func render(
            _ graph: SceneGraph, camera: ShaderGraphCamera, motion: [SIMD4<Float16>]
        ) throws -> [SIMD4<Float>] {
            motion.withUnsafeBytes {
                vectors.replace(region: MTLRegionMake2D(0, 0, gpu.width, gpu.height), mipmapLevel: 0,
                                withBytes: $0.baseAddress!, bytesPerRow: gpu.width * MemoryLayout<SIMD4<Float16>>.stride)
            }
            try gpu.run { command in
                let primary = gpu.pass(color: beauty, depth: depth, depthMode: mode)
                let pass = try layout.makePass(colorAttachments: [
                    SurfaceRenderTargets.colorName: primary.colorAttachments[0]!,
                    "normal": gpu.pass(color: normals, depth: depth).colorAttachments[0]!,
                ], depthAttachment: primary.depthAttachment!)
                try graph.encode(pass: pass, commandBuffer: command, depthMode: mode)
                try ssgi.encode(
                    source: beauty, depth: depth, normals: .worldNormals(normals), motionTexture: vectors,
                    destination: output, camera: camera, depthMode: mode, settings: settings, commandBuffer: command
                )
            }
            return try gpu.pixels(of: output)
        }

        let first = try render(before, camera: previous, motion: emptyMotion)
        let second = try render(after, camera: current, motion: authored)
        let lit = pixel(receiver + displacement, camera: current)
        let uncovered = pixel(hidden + displacement, camera: current)
        let vector = SIMD4<Float>(authored[lit.y * gpu.width + lit.x])
        #expect(abs(vector.x) > 4 && abs(vector.y) > 3 && vector.w == 1)
        let oldClip = previous.projectionMatrix * previous.viewMatrix * SIMD4(receiver, 1)
        #expect(abs(vector.z - oldClip.z / oldClip.w) < 0.001)
        #expect(mean(first, at: pixel(receiver, camera: previous)).x > 0.06)
        #expect(mean(second, at: lit).x > 0.04)
        let priorBlocker = mean(first, at: pixel(hidden, camera: previous))
        #expect(priorBlocker.x > 0.02, "The prior occluder held bounce \(priorBlocker)")
        #expect(authored[uncovered.y * gpu.width + uncovered.x].w == 1)
        #expect(mean(second, at: uncovered).x < 0.001)

        ssgi.resetTemporalHistory()
        _ = try render(before, camera: previous, motion: emptyMotion)
        // Same depth and normal as the old receiver, but explicitly invalid history.
        after.roots[0] = .init(transform: floor.transform, mesh: floor.mesh)
        let third = try render(after, camera: current, motion: emptyMotion)
        let invalid = pixel(receiver, camera: current)
        #expect(mean(third, at: invalid).x < 0.001)
    }

    private func movingFloorVectors(
        from previous: ShaderGraphCamera, to current: ShaderGraphCamera, displacement: SIMD3<Float>
    ) -> [SIMD4<Float16>] {
        let worldFromView = simd_inverse(current.viewMatrix)
        let worldFromClip = worldFromView * simd_inverse(current.projectionMatrix)
        let previousClip = previous.projectionMatrix * previous.viewMatrix
        let eye = worldFromView[3]
        let size = SIMD2(Float(gpu.width), Float(gpu.height))
        var vectors = [SIMD4<Float16>](repeating: .zero, count: gpu.width * gpu.height)
        for y in 0..<gpu.height {
            for x in 0..<gpu.width {
                let currentPixel = SIMD2(Float(x) + 0.5, Float(y) + 0.5)
                let ndc = currentPixel / size * SIMD2(2, -2) + SIMD2(-1, 1)
                let point = worldFromClip * SIMD4(ndc.x, ndc.y, 0.5, 1)
                let ray = point / point.w - eye
                let t = (displacement.y - eye.y) / ray.y
                guard t > 0, t.isFinite else { continue }
                let prior = previousClip * (eye + ray * t - SIMD4(displacement, 0))
                guard prior.w > 0, prior.z >= 0, prior.z <= prior.w else { continue }
                let projected = SIMD2(prior.x, prior.y) / prior.w
                guard abs(projected.x) < 1, abs(projected.y) < 1 else { continue }
                let previousPixel = (projected * SIMD2(0.5, -0.5) + 0.5) * size
                let offset = previousPixel - currentPixel
                vectors[y * gpu.width + x] = SIMD4(Float16(offset.x), Float16(offset.y), Float16(prior.z / prior.w), 1)
            }
        }
        return vectors
    }

    // Independent ray/plane reprojection and bilinear lookup of the previous HDR pixels.
    private func reprojectedFloorRed(
        _ pixels: [SIMD4<Float>], previous: ShaderGraphCamera, current: ShaderGraphCamera, center: SIMD2<Int>
    ) -> Float {
        let worldFromCurrent = simd_inverse(current.viewMatrix)
        let eye = worldFromCurrent[3]
        let worldFromClip = worldFromCurrent * simd_inverse(current.projectionMatrix)
        let previousClip = previous.projectionMatrix * previous.viewMatrix
        let size = SIMD2(Float(gpu.width), Float(gpu.height))
        var red: Float = 0
        for y in (center.y - 2)...(center.y + 2) {
            for x in (center.x - 2)...(center.x + 2) {
                let ndc = (SIMD2(Float(x), Float(y)) + 0.5) / size * SIMD2(2, -2) + SIMD2(-1, 1)
                let world = worldFromClip * SIMD4(ndc.x, ndc.y, 0.5, 1)
                let ray = world / world.w - eye
                let clip = previousClip * (eye - ray * (eye.y / ray.y))
                let prior = (SIMD2(clip.x, clip.y) / clip.w * SIMD2(0.5, -0.5) + 0.5) * size - 0.5
                let base = SIMD2(Int(floor(prior.x)), Int(floor(prior.y)))
                let f = prior - SIMD2(Float(base.x), Float(base.y))
                let row = base.y * gpu.width + base.x
                let upper = pixels[row].x * (1 - f.x) + pixels[row + 1].x * f.x
                let lower = pixels[row + gpu.width].x * (1 - f.x) + pixels[row + gpu.width + 1].x * f.x
                red += upper * (1 - f.y) + lower * f.y
            }
        }
        return red / 25
    }

    @Test("SSGI rejects incompatible normals, depth mode and aliased output")
    func validation() throws {
        let ssgi = try ScreenSpaceGlobalIllumination(device: gpu.device)
        let color = try gpu.colorTarget(), output = try gpu.colorTarget(), depth = try gpu.depthTarget()
        let view = try graphCamera(.forward)
        let command = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: ScreenSpaceGlobalIllumination.Failure.input) {
            try ssgi.encode(source: color, depth: depth, normals: .surfaceAttributes(color), destination: output,
                            camera: view, commandBuffer: command)
        }
        #expect(throws: ScreenSpaceGlobalIllumination.Failure.input) {
            try ssgi.encode(source: color, depth: depth, motionTexture: depth, destination: output,
                            camera: view, settings: .init(temporal: .init()), commandBuffer: command)
        }
        #expect(throws: ScreenSpaceGlobalIllumination.Failure.settings) {
            try ssgi.encode(source: color, depth: depth, motionTexture: color, destination: output,
                            camera: view, commandBuffer: command)
        }
        #expect(throws: ScreenSpaceGlobalIllumination.Failure.camera) {
            try ssgi.encode(source: color, depth: depth, destination: output, camera: view, depthMode: .reversed, commandBuffer: command)
        }
        let alias = try #require(color.makeTextureView(pixelFormat: .rgba16Float))
        #expect(throws: ScreenSpaceGlobalIllumination.Failure.destination) {
            try ssgi.encode(source: color, depth: depth, destination: alias, camera: view, commandBuffer: command)
        }
        #expect(throws: ScreenSpaceGlobalIllumination.Failure.settings) {
            try ssgi.encode(source: color, depth: depth, destination: output, camera: view,
                            settings: .init(stepCount: 0), commandBuffer: command)
        }
    }

    private func scene(
        layout: SurfaceRenderTargets, mode: MeshDepthMode, emission: SIMD3<Float> = SIMD3(1, 0.04, 0.01),
        floorHeight: Float = 0, pose: SceneTransform? = nil
    ) throws -> SceneGraph {
        func mesh(_ geometry: MeshGeometry, albedo: SIMD4<Float>, emission: SIMD3<Float> = .zero) throws -> MeshRenderer {
            let material = try PhysicallyBasedMaterial(
                roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: albedo,
                emissiveFactor: emission, emissiveStrength: 3
            )
            return try MeshRenderer(
                device: gpu.device, surfaceRenderTargets: layout, geometry: geometry, material: material, depthMode: mode
            )
        }
        let floor = SceneNode(transform: .init(
            translation: SIMD3(0, floorHeight, -4.5), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
        ), mesh: try mesh(MeshGeometry.plane(width: 4, height: 7), albedo: albedo))
        let wall = SceneNode(transform: .init(translation: SIMD3(0, 0.9, -5)), mesh: try mesh(
            MeshGeometry.plane(width: 3, height: 1.8), albedo: SIMD4(1, 0, 0, 1), emission: emission
        ))
        return SceneGraph(roots: [floor, wall], camera: camera, cameraTransform: pose ?? transform)
    }

    private func graphCamera(
        _ mode: MeshDepthMode, jitter: SIMD2<Float> = .zero, pose: SceneTransform? = nil
    ) throws -> ShaderGraphCamera {
        let projection = try camera.projection(aspectRatio: 1, depthMode: mode)
        return try ShaderGraphCamera(
            viewMatrix: simd_inverse((pose ?? transform).matrix), projectionMatrix: TemporalAntiAliasing.jitteredClipFromWorld(
                projection, jitter: jitter, width: Double(gpu.width), height: Double(gpu.height)
            ), near: camera.near, far: camera.far
        )
    }

    private func pixel(_ point: SIMD3<Float>, camera: ShaderGraphCamera) -> SIMD2<Int> {
        let clip = camera.projectionMatrix * camera.viewMatrix * SIMD4(point, 1)
        return SIMD2(Int((0.5 + clip.x / clip.w * 0.5) * Float(gpu.width)),
                     Int((0.5 - clip.y / clip.w * 0.5) * Float(gpu.height)))
    }

    private func mean(_ pixels: [SIMD4<Float>], at center: SIMD2<Int>) -> SIMD4<Float> {
        var total = SIMD4<Float>.zero
        for y in (center.y - 2)...(center.y + 2) {
            for x in (center.x - 2)...(center.x + 2) { total += pixels[y * gpu.width + x] }
        }
        return total / 25
    }
}
