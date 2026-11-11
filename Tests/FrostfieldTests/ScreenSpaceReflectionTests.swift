import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

/// A unit box standing on a metal floor, seen by a level camera at the box's height. The
/// mirror image of the box's front face lies below the floor at the reflected heights, so
/// each floor pixel that reflects it is the projection of a point on the virtual face.
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct ScreenSpaceReflectionTests {
    private let fixture: RasterLightingFixture
    private let camera: PerspectiveCamera
    private let cameraTransform = SceneTransform(translation: SIMD3(0, 1, 0))
    private let boxFront: Float = -6

    init() throws {
        fixture = try RasterLightingFixture(width: 256, height: 256)
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 60)
    }

    private func scene(
        floor material: PhysicallyBasedMaterial, depthMode: MeshDepthMode = .forward, extra: [SceneNode] = []
    ) throws -> SceneGraph {
        let floor = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 40, height: 40),
            material: .physicallyBased(material), depthMode: depthMode
        )
        let box = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .bevelBox(bevel: 0), material: .unlit,
            depthMode: depthMode
        )
        var graph = SceneGraph(roots: [
            SceneNode(
                transform: SceneTransform(
                    translation: SIMD3(0, 0, -20), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
                ),
                mesh: floor
            ),
            SceneNode(
                transform: SceneTransform(translation: SIMD3(0, 0.5, boxFront - 0.5)), mesh: box,
                materialParameters: SIMD4(1, 0.25, 0.1, 1)
            ),
        ] + extra, camera: camera)
        graph.cameraTransform = cameraTransform
        return graph
    }

    private func render(
        _ graph: SceneGraph, settings: ScreenSpaceReflections.Settings = .init(),
        environment: PhysicallyBasedMaterial.Environment? = nil, depthMode: MeshDepthMode = .forward,
        clipFromView: simd_float4x4? = nil, receiverMask: (any MTLTexture)? = nil,
        expectedTraceSize: SIMD2<Int>? = nil, previousFrame: ScreenSpaceReflections.PreviousFrame? = nil
    ) throws -> (source: [SIMD4<Float>], reflected: [SIMD4<Float>]) {
        let attributes = try fixture.texture(SceneGraph.surfaceAttributesPixelFormat, usage: [.renderTarget, .shaderRead])
        let prepassDepth = try fixture.depthTarget()
        let color = try fixture.colorTarget()
        let depth = try fixture.depthTarget()
        let destination = try fixture.colorTarget()
        let reflections = try ScreenSpaceReflections(device: fixture.device)
        try fixture.run { buffer in
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = attributes
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = prepassDepth
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.storeAction = .dontCare
            pass.depthAttachment.clearDepth = depthMode == .forward ? 1 : 0
            let colorPass = fixture.pass(color: color, depth: depth, depthMode: depthMode)
            if let clipFromView {
                // A caller-supplied projection for every pass, including reflections.
                let clip = clipFromView * simd_inverse(cameraTransform.matrix)
                try graph.encodeSurfaceAttributes(
                    pass: pass, commandBuffer: buffer, clipFromWorld: clip, cameraPosition: cameraTransform.translation,
                    depthMode: depthMode
                )
                try graph.encode(
                    pass: colorPass, commandBuffer: buffer, clipFromWorld: clip,
                    cameraPosition: cameraTransform.translation, depthMode: depthMode
                )
                try reflections.encode(
                    source: color, depth: depth, surfaceAttributes: attributes, receiverMask: receiverMask,
                    previousFrame: previousFrame, destination: destination,
                    clipFromView: clipFromView, worldFromView: cameraTransform.matrix, depthMode: depthMode,
                    settings: settings, environment: environment, commandBuffer: buffer
                )
                return
            }
            if depthMode == .forward {
                try graph.encodeSurfaceAttributes(pass: pass, commandBuffer: buffer)
                try graph.encode(pass: colorPass, commandBuffer: buffer)
            } else {
                // Reversed depth through the caller-supplied view, as visionOS hosts use it.
                var reverse = matrix_identity_float4x4
                reverse.columns.2.z = -1
                reverse.columns.3.z = 1
                let clip = reverse * (try camera.projection(aspectRatio: 1)) * simd_inverse(cameraTransform.matrix)
                try graph.encodeSurfaceAttributes(
                    pass: pass, commandBuffer: buffer, clipFromWorld: clip, cameraPosition: cameraTransform.translation,
                    depthMode: .reversed
                )
                try graph.encode(
                    pass: colorPass, commandBuffer: buffer, clipFromWorld: clip,
                    cameraPosition: cameraTransform.translation, depthMode: .reversed
                )
            }
            try reflections.encode(
                source: color, depth: depth, surfaceAttributes: attributes, receiverMask: receiverMask,
                previousFrame: previousFrame, destination: destination,
                camera: camera, cameraTransform: cameraTransform, depthMode: depthMode, settings: settings,
                environment: environment, commandBuffer: buffer
            )
        }
        if let expectedTraceSize { #expect(reflections.traceTextureSize == expectedTraceSize) }
        return (try fixture.pixels(of: color), try fixture.pixels(of: destination))
    }

    /// The pixel rectangle of the box front face's mirror image: x in ±0.5 and y in −1...0.
    private func predictedReflection() -> (left: Float, right: Float, top: Float, bottom: Float) {
        let topLeft = fixture.project(SIMD3(-0.5, 0, boxFront), camera: camera, transform: cameraTransform)
        let bottomRight = fixture.project(SIMD3(0.5, -1, boxFront), camera: camera, transform: cameraTransform)
        return (topLeft.x, bottomRight.x, topLeft.y, bottomRight.y)
    }

    @Test(
        "A mirror floor reflects the box at the pixels its mirror image projects to",
        arguments: [MeshDepthMode.forward, .reversed]
    )
    func mirrorReflection(depthMode: MeshDepthMode) throws {
        let mirror = try PhysicallyBasedMaterial(metallic: 1, roughness: 0, lightIntensity: 0, ambientIntensity: 0)
        let (source, reflected) = try render(try scene(floor: mirror, depthMode: depthMode), depthMode: depthMode)
        fixture.writeImage(source, name: "ssr-source-\(depthMode)")
        fixture.writeImage(reflected, name: "ssr-mirror-\(depthMode)")
        let box = source[Int(fixture.height / 2 + 15) * fixture.width + fixture.width / 2]
        #expect(box.x > 0.2 && box.x > box.y)
        let rectangle = predictedReflection()
        var inside = 0
        var outside = 0
        for y in 0..<fixture.height {
            for x in 0..<fixture.width {
                let center = SIMD2(Float(x) + 0.5, Float(y) + 0.5)
                let pixel = reflected[y * fixture.width + x]
                let original = source[y * fixture.width + x]
                // Above the rectangle the box itself and the far floor are unchanged; its top
                // edge is the box's base, so skip the two pixels on either side of it.
                if center.y < rectangle.top - 2 {
                    #expect(pixel == original)
                    continue
                }
                guard center.y > rectangle.top + 2 else { continue }
                if center.x > rectangle.left + 2, center.x < rectangle.right - 2, center.y < rectangle.bottom - 2 {
                    // A white metal at roughness zero reflects everything: the box's own color.
                    #expect(simd_reduce_max(abs(pixel - box)) < 2e-3)
                    inside += 1
                } else if center.x < rectangle.left - 2 || center.x > rectangle.right + 2 || center.y > rectangle.bottom + 2 {
                    #expect(pixel == original)
                    outside += 1
                }
            }
        }
        #expect(inside > 900)
        #expect(outside > 20_000)
        // The unreflected floor is black: no light, no environment.
        #expect(source[(fixture.height - 10) * fixture.width + fixture.width / 2] == SIMD4(0, 0, 0, 1))
    }

    @Test(
        "With an off-center, anisotropic projection the mirror image lands where that projection predicts",
        arguments: [MeshDepthMode.forward, .reversed]
    )
    func offCenterProjection(depthMode: MeshDepthMode) throws {
        let mirror = try PhysicallyBasedMaterial(metallic: 1, roughness: 0, lightIntensity: 0, ambientIntensity: 0)
        let clipFromView = try RasterLightingFixture.offCenterProjection(
            camera, aspectRatio: 1, center: SIMD2(-0.2, 0.3), horizontalScale: 0.85, depthMode: depthMode
        )
        let worldFromView = cameraTransform.matrix
        let (source, reflected) = try render(
            try scene(floor: mirror, depthMode: depthMode), depthMode: depthMode, clipFromView: clipFromView
        )
        fixture.writeImage(reflected, name: "ssr-off-center-\(depthMode)")
        let boxPixel = fixture.project(SIMD3(0, 0.5, boxFront), clipFromView: clipFromView, worldFromView: worldFromView)
        let box = source[Int(boxPixel.y) * fixture.width + Int(boxPixel.x)]
        #expect(box.x > 0.2 && box.x > box.y)
        let topLeft = fixture.project(SIMD3(-0.5, 0, boxFront), clipFromView: clipFromView, worldFromView: worldFromView)
        let bottomRight = fixture.project(SIMD3(0.5, -1, boxFront), clipFromView: clipFromView, worldFromView: worldFromView)
        var inside = 0
        var outside = 0
        for y in 0..<fixture.height {
            for x in 0..<fixture.width {
                let center = SIMD2(Float(x) + 0.5, Float(y) + 0.5)
                let pixel = reflected[y * fixture.width + x]
                let original = source[y * fixture.width + x]
                guard center.y > topLeft.y + 2 else { continue }
                if center.x > topLeft.x + 2, center.x < bottomRight.x - 2, center.y < bottomRight.y - 2 {
                    #expect(simd_reduce_max(abs(pixel - box)) < 2e-3)
                    inside += 1
                } else if center.x < topLeft.x - 2 || center.x > bottomRight.x + 2 || center.y > bottomRight.y + 2 {
                    #expect(pixel == original)
                    outside += 1
                }
            }
        }
        #expect(inside > 600)
        #expect(outside > 10_000)
    }

    @Test("A reduced reflection grid respects a receiver mask and preserves the full-size beauty",
          arguments: [MeshDepthMode.forward, .reversed])
    func reducedSelection(mode: MeshDepthMode) throws {
        let mirror = try PhysicallyBasedMaterial(metallic: 1, roughness: 0, lightIntensity: 0, ambientIntensity: 0)
        let mask = try fixture.texture(.r8Unorm, usage: [.shaderRead], storage: .shared)
        let texels = (0..<(fixture.width * fixture.height)).map { index in
            index % fixture.width < fixture.width / 2 ? UInt8.max : 0
        }
        texels.withUnsafeBytes {
            mask.replace(region: MTLRegionMake2D(0, 0, fixture.width, fixture.height), mipmapLevel: 0,
                         withBytes: $0.baseAddress!, bytesPerRow: fixture.width)
        }
        let result = try render(
            scene(floor: mirror, depthMode: mode), settings: .init(resolutionScale: 0.5),
            depthMode: mode, receiverMask: mask, expectedTraceSize: SIMD2(128, 128)
        )
        let rect = predictedReflection()
        let y = Int((rect.top + rect.bottom) * 0.5)
        let left = y * fixture.width + fixture.width / 2 - 10
        let right = y * fixture.width + fixture.width / 2 + 10
        #expect(result.reflected[left].x > result.source[left].x + 0.2)
        #expect(result.reflected[right] == result.source[right])
        #expect(result.reflected.indices.allSatisfy { result.reflected[$0].w == result.source[$0].w })
        #expect(result.reflected[0] == result.source[0])
    }

    @Test("Receiver roughness softens the reflection edge while retaining the reflected center")
    func roughReflectionBlur() throws {
        let material = try PhysicallyBasedMaterial(metallic: 1, roughness: 0.5, lightIntensity: 0, ambientIntensity: 0)
        let graph = try scene(floor: material)
        let sharp = try render(graph, settings: .init(maximumRoughness: 1))
        let blur = try render(graph, settings: .init(maximumRoughness: 1, roughnessBlurRadius: 32), expectedTraceSize: SIMD2(256, 256))
        let rect = predictedReflection()
        let y = Int((rect.top + rect.bottom) * 0.5)
        let exterior = y * fixture.width + Int(rect.left) - 3
        let interior = y * fixture.width + Int(rect.left) + 3
        let center = y * fixture.width + fixture.width / 2
        #expect(sharp.reflected[exterior] == sharp.source[exterior])
        #expect(blur.reflected[exterior].x > sharp.reflected[exterior].x + 0.02)
        #expect(blur.reflected[interior].x + 0.02 < sharp.reflected[interior].x)
        #expect(abs(blur.reflected[center].x - sharp.reflected[center].x) < 0.03)
        #expect(blur.reflected.indices.allSatisfy { blur.reflected[$0].w == sharp.source[$0].w })
    }

    @Test("Registered prior radiance changes the reflection and rejects a prior occluder",
          arguments: [MeshDepthMode.forward, .reversed])
    func priorRadiance(mode: MeshDepthMode) throws {
        let mirror = try PhysicallyBasedMaterial(metallic: 1, roughness: 0, lightIntensity: 0, ambientIntensity: 0)
        let blocker = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.42, height: 1.2),
            material: .unlit, depthMode: mode
        )
        var prior = try scene(floor: mirror, depthMode: mode, extra: [
            .init(transform: .init(translation: SIMD3(0.28, 0.55, -5)), mesh: blocker,
                  materialParameters: SIMD4(0.1, 0.1, 1, 1)),
        ])
        prior.roots[1].materialParameters = SIMD4(0.12, 1, 0.08, 1)
        let radiance = try fixture.colorTarget(), depth = try fixture.depthTarget()
        try fixture.run { buffer in
            try prior.encode(pass: fixture.pass(color: radiance, depth: depth, depthMode: mode),
                             commandBuffer: buffer, depthMode: mode)
        }
        let projection = try camera.projection(aspectRatio: 1, depthMode: mode)
        let view = try ShaderGraphCamera(
            viewMatrix: simd_inverse(cameraTransform.matrix), projectionMatrix: projection,
            near: camera.near, far: camera.far
        )
        let history = ScreenSpaceReflections.PreviousFrame(radiance: radiance, depth: depth, camera: view, depthMode: mode)
        let graph = try scene(floor: mirror, depthMode: mode)
        let settings = ScreenSpaceReflections.Settings(resolutionScale: mode == .forward ? 1 : 0.5)
        let original = try render(graph, settings: settings, depthMode: mode)
        let bounced = try render(graph, settings: settings, depthMode: mode, previousFrame: history)
        let left = fixture.project(SIMD3(-0.27, -0.5, boxFront), camera: camera, transform: cameraTransform)
        let right = fixture.project(SIMD3(0.27, -0.5, boxFront), camera: camera, transform: cameraTransform)
        let leftIndex = Int(left.y) * fixture.width + Int(left.x)
        let rightIndex = Int(right.y) * fixture.width + Int(right.x)
        let oldHit = fixture.project(SIMD3(0.27, 0.5, boxFront), camera: camera, transform: cameraTransform)
        let oldColor = try fixture.pixels(of: radiance)[Int(oldHit.y) * fixture.width + Int(oldHit.x)]
        #expect(oldColor.z > 4 * oldColor.x)
        #expect(original.reflected[leftIndex].x > 0.2 && original.reflected[rightIndex].x > 0.2)
        #expect(bounced.reflected[leftIndex].y > 4 * bounced.reflected[leftIndex].x)
        #expect(bounced.reflected[rightIndex] == original.reflected[rightIndex])
        #expect(bounced.source == original.source)
        #expect(bounced.reflected.indices.allSatisfy { bounced.reflected[$0].w == bounced.source[$0].w })

        let moved = try ShaderGraphCamera(
            viewMatrix: simd_inverse(SceneTransform(translation: SIMD3(0.1, 1, 0)).matrix),
            projectionMatrix: projection, near: camera.near, far: camera.far
        )
        let incompatible = ScreenSpaceReflections.PreviousFrame(radiance: radiance, depth: depth, camera: moved, depthMode: mode)
        #expect(throws: ScreenSpaceReflections.Failure.historyView) {
            try render(graph, settings: settings, depthMode: mode, previousFrame: incompatible)
        }
    }

    @Test("Prior reflection radiance reprojects translated and rotated cameras across depth conventions",
          arguments: [MeshDepthMode.forward, .reversed])
    func priorCameraMotion(mode: MeshDepthMode) throws {
        let mirror = try PhysicallyBasedMaterial(metallic: 1, roughness: 0, lightIntensity: 0, ambientIntensity: 0)
        let previousMode: MeshDepthMode = mode == .forward ? .reversed : .forward
        let previousPose = SceneTransform(
            translation: SIMD3(0.75, 1.05, 0.60),
            rotation: simd_quatf(angle: 0.0583, axis: simd_normalize(SIMD3<Float>(1, 2, 3)))
        )
        let beforeProjection = try RasterLightingFixture.offCenterProjection(
            camera, aspectRatio: 1, center: SIMD2(0.19, -0.08), horizontalScale: 0.94, depthMode: previousMode
        )
        let jittered = TemporalAntiAliasing.jitteredClipFromWorld(beforeProjection, jitter: SIMD2(0.35, -0.25),
                                                               width: Double(fixture.width), height: Double(fixture.height))
        let previousView = try ShaderGraphCamera(
            viewMatrix: simd_inverse(previousPose.matrix), projectionMatrix: jittered,
            near: camera.near, far: camera.far
        )
        let leftHit = SIMD3<Float>(-0.27, 0.5, boxFront)
        let rightHit = SIMD3<Float>(0.27, 0.5, boxFront)
        let blockerCenter = previousPose.translation + (rightHit - previousPose.translation) * 0.82
        let blocker = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.40, height: 1.2),
            material: .unlit, depthMode: previousMode
        )
        var prior = try scene(floor: mirror, depthMode: previousMode, extra: [
            .init(transform: .init(translation: blockerCenter), mesh: blocker, materialParameters: SIMD4(0.1, 0.1, 1, 1)),
        ])
        prior.roots[1].materialParameters = SIMD4(0.10, 1, 0.1, 1)
        let radiance = try fixture.colorTarget(), depth = try fixture.depthTarget()
        try fixture.run { buffer in
            try prior.encode(
                pass: fixture.pass(color: radiance, depth: depth, depthMode: previousMode), commandBuffer: buffer,
                clipFromWorld: jittered * previousView.viewMatrix, cameraPosition: previousPose.translation,
                depthMode: previousMode
            )
        }
        let history = ScreenSpaceReflections.PreviousFrame(
            radiance: radiance, depth: depth, camera: previousView, depthMode: previousMode, reprojection: .cameraMotion
        )
        func index(_ p: SIMD2<Float>) -> Int { Int(p.y) * fixture.width + Int(p.x) }
        let oldLeft = fixture.project(leftHit, clipFromView: jittered, worldFromView: previousPose.matrix)
        let oldRight = fixture.project(rightHit, clipFromView: jittered, worldFromView: previousPose.matrix)
        let newLeft = fixture.project(leftHit, camera: camera, transform: cameraTransform)
        let priorPixels = try fixture.pixels(of: radiance)
        #expect(abs(oldLeft.x - newLeft.x) > 8)
        let oldLeftColor = priorPixels[index(oldLeft)]
        let oldRightColor = priorPixels[index(oldRight)]
        // The unlit fixture multiplies channels by the +Z surface-normal tint.
        #expect(simd_reduce_max(abs(oldLeftColor - SIMD4(0.022, 0.22, 0.1, 1))) < 0.002)
        #expect(simd_reduce_max(abs(oldRightColor - SIMD4(0.022, 0.022, 1, 1))) < 0.002)

        let graph = try scene(floor: mirror, depthMode: mode)
        let settings = ScreenSpaceReflections.Settings(resolutionScale: mode == .forward ? 1 : 0.5)
        let current = try render(graph, settings: settings, depthMode: mode)
        let reprojected = try render(graph, settings: settings, depthMode: mode, previousFrame: history)
        let left = index(fixture.project(SIMD3(leftHit.x, -leftHit.y, leftHit.z), camera: camera, transform: cameraTransform))
        let right = index(fixture.project(SIMD3(rightHit.x, -rightHit.y, rightHit.z), camera: camera, transform: cameraTransform))
        let redBox = SIMD4<Float>(0.22, 0.055, 0.1, 1)
        #expect(simd_reduce_max(abs(current.reflected[left] - redBox)) < 0.002)
        #expect(simd_reduce_max(abs(current.reflected[right] - redBox)) < 0.002)
        #expect(simd_reduce_max(abs(reprojected.reflected[left] - oldLeftColor)) < 0.002)
        #expect(reprojected.reflected[right] == current.reflected[right])
        #expect(reprojected.reflected.indices.allSatisfy { reprojected.reflected[$0].w == current.source[$0].w })
    }

    @Test("Seeded GGX repeats exactly and spreads rough rays beyond the mirror footprint")
    func stochasticGloss() throws {
        let rough = try PhysicallyBasedMaterial(metallic: 1, roughness: 0.35, lightIntensity: 0, ambientIntensity: 0)
        let graph = try scene(floor: rough)
        let sharp = try render(graph, settings: .init(maximumRoughness: 1, resolutionScale: 0.5))
        var settings = ScreenSpaceReflections.Settings(maximumRoughness: 1, resolutionScale: 0.5, stochasticSeed: 0xa412_0072)
        let first = try render(graph, settings: settings)
        let same = try render(graph, settings: settings)
        settings.stochasticSeed = 0xa513_0072
        let second = try render(graph, settings: settings)
        #expect(first.reflected == same.reflected)
        var changed = 0
        var spread = 0
        let rect = predictedReflection()
        for y in Int(rect.top + 5)..<Int(rect.bottom + 12) {
            for x in Int(rect.left - 20)..<Int(rect.right + 20) {
                let p = y * fixture.width + x
                if abs(first.reflected[p].x - second.reflected[p].x) > 0.02 { changed += 1 }
                if sharp.reflected[p] == sharp.source[p], first.reflected[p].x > 0.03 { spread += 1 }
            }
        }
        #expect(changed > 100)
        #expect(spread > 40)
        #expect(first.reflected.indices.allSatisfy { first.reflected[$0].w == first.source[$0].w })

        let mirror = try PhysicallyBasedMaterial(metallic: 1, roughness: 0, lightIntensity: 0, ambientIntensity: 0)
        let smooth = try scene(floor: mirror)
        let deterministic = try render(smooth, settings: .init())
        let stochastic = try render(smooth, settings: .init(stochasticSeed: 123))
        #expect(stochastic.reflected == deterministic.reflected)
    }

    @Test("Reflections fade between half and all of the maximum roughness, weighted by the stored reflectance")
    func roughnessFade() throws {
        let rough = try PhysicallyBasedMaterial(metallic: 1, roughness: 0.45, lightIntensity: 0, ambientIntensity: 0)
        let graph = try scene(floor: rough)
        let (source, reflected) = try render(graph, settings: .init(maximumRoughness: 0.6))
        let rectangle = predictedReflection()
        let x = Int((rectangle.left + rectangle.right) / 2)
        let y = Int((rectangle.top + rectangle.bottom) / 2)
        let box = source[Int(fixture.height / 2 + 15) * fixture.width + fixture.width / 2]
        // Karis's split-sum fit for a white metal (F0 = F90 = 1): A + B.
        let ray = fixture.ray(x: x, y: y, camera: camera, transform: cameraTransform)
        let noV = -ray.direction.y
        let r = Float(0.45) * SIMD4<Float>(-1, -0.0275, -0.572, 0.022) + SIMD4<Float>(1, 0.0425, 1.04, -0.04)
        let a004 = min(r.x * r.x, exp2(-9.28 * noV)) * r.x + r.y
        let reflectance = (-1.04 * a004 + r.z) + (1.04 * a004 + r.w)
        // 1 − smoothstep(0.3, 0.6, 0.45) = 0.5.
        let expected = 0.5 * reflectance * box.x
        let measured = reflected[y * fixture.width + x].x
        #expect(abs(measured - expected) < 0.02 * expected)
        let tooRough = try PhysicallyBasedMaterial(metallic: 1, roughness: 0.6, lightIntensity: 0, ambientIntensity: 0)
        let (roughSource, roughReflected) = try render(try scene(floor: tooRough), settings: .init(maximumRoughness: 0.6))
        #expect(roughSource == roughReflected)
    }

    @Test("With the environment supplied, a hit replaces the environment's reflection instead of adding to it")
    func replacesEnvironmentReflection() throws {
        let width = 16
        let height = 8
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let panorama = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let texels = [SIMD4<Float16>](repeating: SIMD4(0.5, 0.6, 0.7, 1), count: width * height)
        try texels.withUnsafeBytes { bytes in
            panorama.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: try #require(bytes.baseAddress), bytesPerRow: width * 8
            )
        }
        let lighting = try EnvironmentLighting(
            queue: fixture.queue, equirectangular: panorama,
            settings: .init(cubeSize: 16, diffuseSize: 8, brdfSize: 32, sampleCount: 64)
        )
        let environment = try PhysicallyBasedMaterial.Environment(lighting: lighting)
        let mirror = try PhysicallyBasedMaterial(
            metallic: 1, roughness: 0, lightIntensity: 0, ambientIntensity: 0, environment: environment
        )
        let (source, reflected) = try render(try scene(floor: mirror), environment: environment)
        fixture.writeImage(reflected, name: "ssr-environment")
        let rectangle = predictedReflection()
        let x = Int((rectangle.left + rectangle.right) / 2)
        let y = Int((rectangle.top + rectangle.bottom) / 2)
        let box = source[Int(fixture.height / 2 + 15) * fixture.width + fixture.width / 2]
        let floorBefore = source[y * fixture.width + x]
        // The floor first showed the sky; the reflected box replaces it, within the
        // reflectance's square-root unorm8 storage.
        #expect(floorBefore.z > 0.6)
        #expect(simd_reduce_max(abs(reflected[y * fixture.width + x] - box)) < 0.01)
        let open = (fixture.height - 10) * fixture.width + fixture.width / 2
        #expect(reflected[open] == source[open])
    }

    /// For each pixel well inside the mirror rectangle: the reflected ray's world length
    /// from the floor to the box's front face, and the continuous pixel where it lands.
    private func mirrorHits(columns: [Int]) -> [(x: Int, y: Int, length: Float, hit: SIMD2<Float>)] {
        let rectangle = predictedReflection()
        var hits: [(x: Int, y: Int, length: Float, hit: SIMD2<Float>)] = []
        for x in columns {
            for y in 0..<fixture.height {
                let center = SIMD2(Float(x) + 0.5, Float(y) + 0.5)
                guard center.y > rectangle.top + 2, center.y < rectangle.bottom - 2,
                      center.x > rectangle.left + 2, center.x < rectangle.right - 2 else { continue }
                let ray = fixture.ray(x: x, y: y, camera: camera, transform: cameraTransform)
                let floorPoint = ray.origin + (-ray.origin.y / ray.direction.y) * ray.direction
                let mirrored = SIMD3(ray.direction.x, -ray.direction.y, ray.direction.z)
                let length = (boxFront - floorPoint.z) / mirrored.z
                let hit = floorPoint + length * mirrored
                hits.append((x, y, length, fixture.project(hit, camera: camera, transform: cameraTransform)))
            }
        }
        return hits
    }

    @Test("Reflections fade over the last 20% of the maximum distance, measured along the reflected ray")
    func distanceFade() throws {
        let mirror = try PhysicallyBasedMaterial(metallic: 1, roughness: 0, lightIntensity: 0, ambientIntensity: 0)
        // The longest reflected ray to the box is √10 ≈ 3.16 units, so rays reach into the fade.
        let maximumDistance: Float = 3.3
        let (source, reflected) = try render(
            try scene(floor: mirror), settings: .init(maximumDistance: maximumDistance, edgeFade: 0)
        )
        fixture.writeImage(reflected, name: "ssr-distance-fade")
        let box = source[Int(fixture.height / 2 + 15) * fixture.width + fixture.width / 2]
        var partial = 0
        var largest: Float = 0
        let hits = mirrorHits(columns: [120, 128, 136])
        for hit in hits {
            let fade = 1 - RasterLightingFixture.smoothstep(0.8, 1, hit.length / maximumDistance)
            let measured = reflected[hit.y * fixture.width + hit.x]
            // The floor is black, so the faded reflection is all its color; alpha is preserved.
            largest = max(largest, simd_reduce_max(abs(measured - SIMD4(fade * SIMD3(box.x, box.y, box.z), 1))))
            if fade > 0.1, fade < 0.9 { partial += 1 }
        }
        #expect(hits.count > 60)
        #expect(partial >= 9)
        // Bisection places the hit within a small fraction of a pixel along the ray.
        #expect(largest < 0.01)
        fixture.note("SSR distance fade: \(hits.count) pixels, \(partial) partly faded, largest error \(largest)")
    }

    @Test("Hits fade linearly toward the screen edges over the edge fraction")
    func edgeFade() throws {
        let mirror = try PhysicallyBasedMaterial(metallic: 1, roughness: 0, lightIntensity: 0, ambientIntensity: 0)
        let edgeFade: Float = 0.5
        let (source, reflected) = try render(try scene(floor: mirror), settings: .init(edgeFade: edgeFade))
        let box = source[Int(fixture.height / 2 + 15) * fixture.width + fixture.width / 2]
        var largest: Float = 0
        var smallest: Float = 1
        var worst = ""
        let hits = mirrorHits(columns: [116, 128, 140])
        for hit in hits {
            let uv = hit.hit / SIMD2(Float(fixture.width), Float(fixture.height))
            let edge = min(max(min(min(uv.x, 1 - uv.x), min(uv.y, 1 - uv.y)) / edgeFade, 0), 1)
            let measured = reflected[hit.y * fixture.width + hit.x]
            let error = simd_reduce_max(abs(measured - SIMD4(edge * SIMD3(box.x, box.y, box.z), 1)))
            if error > largest { worst = "pixel (\(hit.x), \(hit.y)) hit \(hit.hit) measured \(measured.x / box.x) expected \(edge)" }
            largest = max(largest, error)
            smallest = min(smallest, edge)
        }
        #expect(hits.count > 60)
        // The hits span the lower part of the box face, between 0.36 and 0.5 of the screen from its bottom edge.
        #expect(smallest < 0.8)
        #expect(largest < 0.01, "\(worst)")
        fixture.note("SSR edge fade: \(hits.count) pixels, weights down to \(smallest), largest error \(largest)")
    }

    @Test("A ray that passes behind a thin card by more than the thickness continues to the box")
    func thickness() throws {
        let mirror = try PhysicallyBasedMaterial(metallic: 1, roughness: 0, lightIntensity: 0, ambientIntensity: 0)
        // A card facing the camera at z = −2 covers screen rows 170–180 over the mirror
        // rectangle's columns. Rays leaving the floor below it are 1.5 units or more behind it there.
        let card = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.6, height: 0.09),
            material: .unlit
        )
        let cardNode = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0.575, -2)), mesh: card,
            materialParameters: SIMD4(0.1, 0.3, 1, 1)
        )
        let (source, reflected) = try render(try scene(floor: mirror, extra: [cardNode]))
        fixture.writeImage(reflected, name: "ssr-thickness")
        let box = source[Int(fixture.height / 2 + 15) * fixture.width + fixture.width / 2]
        let cardTop = fixture.project(SIMD3(0, 0.62, -2), camera: camera, transform: cameraTransform)
        let cardBottom = fixture.project(SIMD3(0, 0.53, -2), camera: camera, transform: cameraTransform)
        let cardColor = source[Int((cardTop.y + cardBottom.y) / 2) * fixture.width + fixture.width / 2]
        #expect(cardColor.z > 0.5 && cardColor.z > cardColor.x)
        var compared = 0
        for hit in mirrorHits(columns: [120, 128, 136]) where Float(hit.y) > cardBottom.y + 2 {
            // The ray's screen path crosses the card before it reaches the box.
            #expect(hit.hit.y < cardTop.y)
            #expect(simd_reduce_max(abs(reflected[hit.y * fixture.width + hit.x] - box)) < 2e-3)
            compared += 1
        }
        #expect(compared > 30)
    }

    @Test("A material's occlusion map scales the replaced environment reflection and the hit alike")
    func materialOcclusion() throws {
        let environment = try fixture.environment { _, _ in SIMD3(0.5, 0.6, 0.7) }
        let occlusionTexture = try fixture.constantTexture(SIMD4(128, 255, 255, 255))
        let occlusion: Float = 128 / 255
        let mirror = try PhysicallyBasedMaterial(
            metallic: 1, roughness: 0, lightIntensity: 0, ambientIntensity: 0,
            occlusionTexture: .init(texture: occlusionTexture), environment: environment
        )
        let (source, reflected) = try render(try scene(floor: mirror), environment: environment)
        let box = source[Int(fixture.height / 2 + 15) * fixture.width + fixture.width / 2]
        let hits = mirrorHits(columns: [120, 128, 136])
        for hit in hits {
            let index = hit.y * fixture.width + hit.x
            // Shading showed the occluded sky; the hit replaces exactly that, at the same occlusion.
            #expect(source[index].z > 0.3 && source[index].z < 0.4)
            #expect(simd_reduce_max(abs(reflected[index] - SIMD4(occlusion * SIMD3(box.x, box.y, box.z), 1))) < 0.01)
        }
        #expect(hits.count > 60)
    }

    @Test("Reflection inputs are validated before encoding")
    func validation() throws {
        let reflections = try ScreenSpaceReflections(device: fixture.device)
        let color = try fixture.colorTarget()
        let depth = try fixture.depthTarget()
        let attributes = try fixture.texture(SceneGraph.surfaceAttributesPixelFormat, usage: [.renderTarget, .shaderRead])
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: ScreenSpaceReflections.Failure.invalidSettings) {
            try reflections.encode(
                source: color, depth: depth, surfaceAttributes: attributes, destination: try fixture.colorTarget(),
                camera: camera, cameraTransform: cameraTransform, settings: .init(thickness: 0), commandBuffer: buffer
            )
        }
        #expect(throws: ScreenSpaceReflections.Failure.incompatibleTextures) {
            try reflections.encode(
                source: color, depth: depth, surfaceAttributes: color, destination: try fixture.colorTarget(),
                camera: camera, cameraTransform: cameraTransform, commandBuffer: buffer
            )
        }
        #expect(throws: ScreenSpaceReflections.Failure.incompatibleTextures) {
            try reflections.encode(
                source: color, depth: depth, surfaceAttributes: attributes, destination: color,
                camera: camera, cameraTransform: cameraTransform, commandBuffer: buffer
            )
        }
        var sheared = try camera.projection(aspectRatio: 1)
        sheared.columns.0.y = 0.1
        #expect(throws: ScreenSpaceReflections.Failure.cameraRange) {
            try reflections.encode(
                source: color, depth: depth, surfaceAttributes: attributes, destination: try fixture.colorTarget(),
                clipFromView: sheared, worldFromView: cameraTransform.matrix, commandBuffer: buffer
            )
        }
        #expect(throws: ScreenSpaceReflections.Failure.cameraRange) {
            try reflections.encode(
                source: color, depth: depth, surfaceAttributes: attributes, destination: try fixture.colorTarget(),
                clipFromView: try camera.projection(aspectRatio: 1), worldFromView: cameraTransform.matrix,
                depthMode: .reversed, commandBuffer: buffer
            )
        }
    }
}
