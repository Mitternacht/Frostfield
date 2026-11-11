import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ScreenSpaceShadowsTests {
    @Test("A rendered caster shadows the floor's directional light while preserving ambient and alpha",
          arguments: [MeshDepthMode.forward, .reversed, .logarithmic], [false, true])
    func floorShadow(mode: MeshDepthMode, orthographic: Bool) throws {
        let gpu = try RasterLightingFixture(width: 144, height: 112)
        let shadows = try ScreenSpaceShadows(device: gpu.device)
        let pose = RasterLightingFixture.cameraTransform(position: SIMD3(0, 2.2, 3.5), pitch: -atan2(2.2, 3.5))
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.07, far: 70)
        let light = simd_normalize(SIMD3<Float>(0.8, 0.45, 0.2))
        let aspect = Float(gpu.width) / Float(gpu.height)
        let view = simd_inverse(pose.matrix)
        let projection: simd_float4x4
        let jitter = SIMD2<Float>(0.35, -0.25)
        if orthographic {
            let depthScale: Float = mode == .reversed ? 0.0625 : -0.0625
            let depthOffset: Float = mode == .reversed ? 1 : 0
            projection = simd_float4x4(
                SIMD4(1 / (1.7 * aspect), 0, 0, 0), SIMD4(0, 1 / 1.7, 0, 0),
                SIMD4(0, 0, depthScale, 0), SIMD4(-0.12, 0.10, depthOffset, 1)
            )
        } else {
            projection = TemporalAntiAliasing.jitteredClipFromWorld(
                try camera.projection(aspectRatio: aspect, depthMode: mode), jitter: jitter,
                width: Double(gpu.width), height: Double(gpu.height)
            )
        }
        let graphCamera = try ShaderGraphCamera(viewMatrix: view, projectionMatrix: projection,
                                                near: orthographic ? 0 : camera.near, far: orthographic ? 16 : camera.far)
        let floor = try MeshGeometry.plane(width: 4, height: 3.8)
        let caster = try MeshGeometry.box(width: 0.30, height: 0.52, depth: 0.34)
        let materialGraph = try SurfaceShaderGraph { graph in .init(roughness: graph.constant(Float(1))) }
        let point = SceneNode(
            transform: .init(translation: SIMD3(-0.4, 1.2, -0.2)),
            light: try PunctualLight(kind: .point(), color: SIMD3(0.1, 0.5, 1), intensity: 2)
        )
        func render(
            direct: Float, ambient: Float, includeCaster: Bool = true, otherLighting: Bool = true,
            directionalVisibility: ScreenSpaceShadows.DirectionalVisibility? = nil,
            directionalMap: DirectionalShadowMap? = nil, ambientVisibility: (any MTLTexture)? = nil
        ) throws -> (color: any MTLTexture, depth: any MTLTexture) {
            let material = MeshMaterial.physicallyBased(try .init(
                roughness: 1, directionToLight: light, lightIntensity: direct, ambientIntensity: ambient,
                receivesShadows: directionalMap != nil, baseColorFactor: SIMD4(0.76, 0.58, 0.4, 1),
                emissiveFactor: otherLighting ? SIMD3(0.07, 0.018, 0.005) : .zero,
                surfaceGraph: orthographic ? materialGraph : nil, specular: .init(factor: 0)
            ))
            func mesh(_ geometry: MeshGeometry) throws -> MeshRenderer {
                try MeshRenderer(device: gpu.device, colorPixelFormat: .rgba16Float, geometry: geometry,
                                 material: material, depthMode: mode)
            }
            var graph = SceneGraph(roots: [
                SceneNode(transform: .init(rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))), mesh: try mesh(floor), castsShadow: false)
            ], camera: camera)
            if includeCaster {
                graph.roots.append(SceneNode(transform: .init(translation: SIMD3(0.24, 0.26, 0)), mesh: try mesh(caster)))
            }
            if otherLighting { graph.roots.append(point) }
            graph.cameraTransform = pose
            graph.projectionJitter = mode == .forward && !orthographic ? jitter : .zero
            let color = try gpu.colorTarget(), depth = try gpu.depthTarget()
            try gpu.run { command in
                let pass = gpu.pass(color: color, depth: depth, depthMode: mode)
                if mode == .logarithmic { pass.depthAttachment.clearDepth = 1 }
                if mode == .forward && !orthographic {
                    try graph.encode(pass: pass, commandBuffer: command, shadowMap: directionalMap, depthMode: mode,
                                     screenSpaceOcclusion: ambientVisibility, screenSpaceDirectionalShadow: directionalVisibility)
                } else {
                    try graph.encode(pass: pass, commandBuffer: command, clipFromWorld: projection * view,
                                     cameraPosition: pose.translation, depthMode: mode, shadowMap: directionalMap,
                                     graphCamera: graphCamera, screenSpaceOcclusion: ambientVisibility,
                                     screenSpaceDirectionalShadow: directionalVisibility)
                }
            }
            return (color, depth)
        }

        let beauty = try render(direct: 2.5, ambient: 0.42)
        let directional = try render(direct: 2.5, ambient: 0, otherLighting: false)
        let ambient = try render(direct: 0, ambient: 0.42)
        let bareFloor = try render(direct: 2.5, ambient: 0.42, includeCaster: false)
        let settings = ScreenSpaceShadows.Settings(maximumDistance: 0.82, thickness: 0.12,
                                                    depthBias: 0.002, intensity: 1, quality: 1)
        func capture(
            _ settings: ScreenSpaceShadows.Settings, depth: any MTLTexture, direction: SIMD3<Float>,
            reduced: Bool = false, wholeBeauty: Bool = false
        ) throws -> (mask: any MTLTexture, color: [SIMD4<Float>]) {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: reduced ? .r8Unorm : .rgba16Float, width: reduced ? gpu.width / 2 : gpu.width,
                height: reduced ? gpu.height / 2 : gpu.height, mipmapped: false
            )
            descriptor.usage = [.renderTarget, .shaderRead]
            descriptor.storageMode = reduced ? .private : .shared
            let mask = try #require(gpu.device.makeTexture(descriptor: descriptor))
            let output = try gpu.colorTarget()
            try gpu.run { command in
                try shadows.encodeVisibility(
                    depth: depth, destination: mask, camera: graphCamera, depthMode: mode, directionToLight: direction,
                    settings: settings, commandBuffer: command
                )
                try shadows.encodeComposite(beauty: beauty.color, directionalRadiance: wholeBeauty ? beauty.color : directional.color,
                                            visibility: mask, destination: output, commandBuffer: command)
            }
            return (mask, try gpu.pixels(of: output))
        }
        func pixel(_ point: SIMD3<Float>) -> Int {
            let p = gpu.project(point, clipFromView: projection, worldFromView: pose.matrix)
            return Int(p.y) * gpu.width + Int(p.x)
        }
        let receiver = pixel(SIMD3(-0.32, 0, -0.05))
        let away = pixel(SIMD3(0.60, 0, -0.10))
        let reference = try gpu.pixels(of: beauty.color)
        let ambientOnly = try gpu.pixels(of: ambient.color)
        let full = try capture(settings, depth: beauty.depth, direction: light)
        let visible = try gpu.pixels(of: full.mask)
        #expect(visible[receiver].x == 0)
        #expect(visible[away] == SIMD4(repeating: 1))
        #expect(visible[0] == SIMD4(repeating: 1) && reference[0].w == 0)
        #expect(reference[receiver].x - ambientOnly[receiver].x > 0.15)
        #expect(simd_reduce_max(abs(full.color[receiver] - ambientOnly[receiver])) < 0.003)
        #expect(full.color[away] == reference[away])
        #expect(zip(full.color, reference).allSatisfy { $0.w == $1.w })

        let binding = try ScreenSpaceShadows.DirectionalVisibility(texture: full.mask, directionToLight: light)
        let integrated = try render(direct: 2.5, ambient: 0.42, directionalVisibility: binding)
        let integratedPixels = try gpu.pixels(of: integrated.color)
        #expect(integratedPixels[receiver] == ambientOnly[receiver])
        #expect(integratedPixels[receiver].z > 0.1)
        #expect(integratedPixels[away] == reference[away])
        #expect(zip(integratedPixels, reference).allSatisfy { $0.w == $1.w })

        let scaled = try capture(settings, depth: beauty.depth, direction: light, reduced: true)
        #expect(scaled.mask.width * 2 == gpu.width && scaled.mask.height * 2 == gpu.height)
        #expect(scaled.color[receiver].x < reference[receiver].x - 0.09)
        let reducedBinding = try ScreenSpaceShadows.DirectionalVisibility(texture: scaled.mask, directionToLight: light)
        let integratedReduced = try render(direct: 2.5, ambient: 0.42, directionalVisibility: reducedBinding)
        let reducedPixels = try gpu.pixels(of: integratedReduced.color)
        #expect(simd_reduce_max(abs(reducedPixels[receiver] - scaled.color[receiver])) < 0.003)
        let removed = try capture(settings, depth: bareFloor.depth, direction: light)
        #expect(removed.color[receiver] == reference[receiver])
        let otherSide = try capture(settings, depth: beauty.depth, direction: SIMD3(-light.x, light.y, light.z))
        #expect(otherSide.color[receiver] == reference[receiver])

        var shorter = settings
        shorter.maximumDistance = 0.25
        let short = try capture(shorter, depth: beauty.depth, direction: light)
        #expect(short.color[receiver] == reference[receiver])
        var disabled = settings
        disabled.intensity = 0
        let off = try capture(disabled, depth: beauty.depth, direction: light, wholeBeauty: true)
        #expect(off.color == reference)
        if mode == .forward {
            let allVisible = try ScreenSpaceShadows.DirectionalVisibility(texture: off.mask, directionToLight: light)
            let unchanged = try render(direct: 2.5, ambient: 0.42, directionalVisibility: allVisible)
            #expect(try gpu.pixels(of: unchanged.color) == reference)
            let mismatch = try ScreenSpaceShadows.DirectionalVisibility(texture: full.mask, directionToLight: SIMD3(-light.x, light.y, light.z))
            let unmatched = try render(direct: 2.5, ambient: 0.42, directionalVisibility: mismatch)
            #expect(try gpu.pixels(of: unmatched.color) == reference)
        }
        if mode == .forward && orthographic {
            let aoMask = try gpu.colorTarget()
            let zeros = [SIMD4<Float16>](repeating: .zero, count: gpu.width * gpu.height)
            zeros.withUnsafeBytes {
                aoMask.replace(region: MTLRegionMake2D(0, 0, gpu.width, gpu.height), mipmapLevel: 0,
                               withBytes: $0.baseAddress!, bytesPerRow: gpu.width * MemoryLayout<SIMD4<Float16>>.stride)
            }
            let ambientOnlyWithAO = try render(direct: 0, ambient: 0.42, ambientVisibility: aoMask)
            let withBothMasks = try render(direct: 2.5, ambient: 0.42, directionalVisibility: binding, ambientVisibility: aoMask)
            let aoOnlyPixel = try gpu.pixels(of: ambientOnlyWithAO.color)[receiver]
            let bothMasksPixel = try gpu.pixels(of: withBothMasks.color)[receiver]
            #expect(bothMasksPixel == aoOnlyPixel)
            #expect(bothMasksPixel.x < ambientOnly[receiver].x - 0.03 && bothMasksPixel.z > 0.01)

            let directionalMap = try DirectionalShadowMap(
                queue: gpu.queue, resolution: 128, directionToLight: light, halfExtent: 3, intensity: 2.5, sampling: .basic
            )
            let mapOnly = try render(direct: 2.5, ambient: 0.42, directionalMap: directionalMap)
            let onlyMapPixels = try gpu.pixels(of: mapOnly.color)
            #expect(onlyMapPixels[receiver].x < reference[receiver].x - 0.1)
            #expect(onlyMapPixels[away] == reference[away])
            let partial = try gpu.texture(.r16Float, usage: .shaderRead, storage: .shared)
            let values = [Float16](repeating: 0.5, count: gpu.width * gpu.height)
            values.withUnsafeBytes {
                partial.replace(region: MTLRegionMake2D(0, 0, gpu.width, gpu.height), mipmapLevel: 0,
                                withBytes: $0.baseAddress!, bytesPerRow: gpu.width * MemoryLayout<Float16>.stride)
            }
            let half = try ScreenSpaceShadows.DirectionalVisibility(texture: partial, directionToLight: light)
            let combined = try render(direct: 2.5, ambient: 0.42, directionalVisibility: half, directionalMap: directionalMap)
            let combinedPixels = try gpu.pixels(of: combined.color)
            for address in [receiver, away] {
                let expected = (onlyMapPixels[address] + ambientOnly[address]) * 0.5
                #expect(simd_reduce_max(abs(combinedPixels[address] - expected)) < 0.003)
            }
        }
        if mode == .forward && !orthographic {
            var seeded = settings
            seeded.frameIndex = 7
            let one = try capture(seeded, depth: beauty.depth, direction: light)
            let two = try capture(seeded, depth: beauty.depth, direction: light)
            #expect(one.color == two.color)
            #expect(one.color != full.color)
        }
    }

    @Test("Screen-space shadow parameters and texture roles are validated before encoding")
    func validation() throws {
        let gpu = try RasterLightingFixture(width: 24, height: 24)
        let pass = try ScreenSpaceShadows(device: gpu.device)
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 10)
        let projection = try camera.projection(aspectRatio: 1, depthMode: .reversed)
        let view = try ShaderGraphCamera(viewMatrix: matrix_identity_float4x4, projectionMatrix: projection,
                                         near: camera.near, far: camera.far)
        let depth = try gpu.depthTarget()
        let output = try gpu.texture(.r16Float, usage: [.renderTarget, .shaderRead])
        let color = try gpu.colorTarget()
        let command = try #require(gpu.queue.makeCommandBuffer())
        let scene = SceneGraph(roots: [], camera: camera)
        let feedback = try ScreenSpaceShadows.DirectionalVisibility(texture: color, directionToLight: SIMD3(1, 1, 0))
        #expect(throws: ScreenSpaceShadows.Failure.attachments) {
            try scene.encode(pass: gpu.pass(color: color, depth: depth), commandBuffer: command,
                             screenSpaceDirectionalShadow: feedback)
        }
        let partial = MTLViewport(originX: 1, originY: 0, width: 23, height: 24, znear: 0, zfar: 1)
        let partialMask = try ScreenSpaceShadows.DirectionalVisibility(texture: output, directionToLight: SIMD3(1, 1, 0))
        #expect(throws: ScreenSpaceShadows.Failure.attachments) {
            try scene.encode(pass: gpu.pass(color: color, depth: depth), commandBuffer: command,
                             clipFromWorld: projection, cameraPosition: .zero, viewport: partial, depthMode: .reversed,
                             screenSpaceDirectionalShadow: partialMask)
        }
        #expect(throws: ScreenSpaceShadows.Failure.parameters) {
            try pass.encodeVisibility(depth: depth, destination: output, camera: view, depthMode: .reversed,
                                      directionToLight: .zero, commandBuffer: command)
        }
        #expect(throws: ScreenSpaceShadows.Failure.parameters) {
            try pass.encodeVisibility(depth: depth, destination: output, camera: view, depthMode: .reversed,
                                      directionToLight: SIMD3(1, 1, 0), settings: .init(maximumSteps: 0), commandBuffer: command)
        }
        #expect(throws: ScreenSpaceShadows.Failure.camera) {
            try pass.encodeVisibility(depth: depth, destination: output, camera: view,
                                      directionToLight: SIMD3(1, 1, 0), commandBuffer: command)
        }
        #expect(throws: ScreenSpaceShadows.Failure.attachments) {
            try pass.encodeComposite(beauty: color, directionalRadiance: color, visibility: output,
                                     destination: color, commandBuffer: command)
        }
        try gpu.run { command in
            try pass.encodeVisibility(depth: depth, destination: output, camera: camera, cameraTransform: .init(),
                                      directionToLight: SIMD3(1, 1, 0), settings: .init(quality: 0), commandBuffer: command)
        }
    }
}
