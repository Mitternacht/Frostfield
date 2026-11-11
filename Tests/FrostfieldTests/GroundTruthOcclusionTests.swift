import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

/// A floor (y = 0, z ≥ 0) meeting a wall (z = 0, y ≥ 0) in a 90° crease along X, seen
/// from in front of the wall and above the floor.
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct GroundTruthOcclusionTests {
    private let fixture: RasterLightingFixture
    private let camera: PerspectiveCamera
    private let cameraTransform = RasterLightingFixture.cameraTransform(
        position: SIMD3(0, 1.5, 2.5), pitch: -atan2(1.5, 2.5)
    )

    init() throws {
        fixture = try RasterLightingFixture(width: 256, height: 256)
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 50)
    }

    private func corner(_ material: MeshMaterial, depthMode: MeshDepthMode = .forward) throws -> SceneGraph {
        let plane = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 20, height: 20),
            material: material, depthMode: depthMode
        )
        var graph = SceneGraph(roots: [
            SceneNode(
                transform: SceneTransform(
                    translation: SIMD3(0, 0, 10), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
                ),
                mesh: plane
            ),
            SceneNode(transform: SceneTransform(translation: SIMD3(0, 10, 0)), mesh: plane),
        ], camera: camera)
        graph.cameraTransform = cameraTransform
        return graph
    }

    private func prepass(_ graph: SceneGraph) throws -> (depth: any MTLTexture, attributes: any MTLTexture) {
        let attributes = try fixture.texture(
            SceneGraph.surfaceAttributesPixelFormat, usage: [.renderTarget, .shaderRead], storage: .shared
        )
        let depth = try fixture.depthTarget()
        try fixture.run { buffer in
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = attributes
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = depth
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.storeAction = .store
            pass.depthAttachment.clearDepth = 1
            try graph.encodeSurfaceAttributes(pass: pass, commandBuffer: buffer)
        }
        return (depth, attributes)
    }

    private func visibility(
        depth: any MTLTexture, attributes: (any MTLTexture)?, settings: AmbientOcclusion.Settings
    ) throws -> (texture: any MTLTexture, pixels: [SIMD4<Float>]) {
        let occlusion = try AmbientOcclusion(device: fixture.device)
        let destination = try fixture.colorTarget()
        try fixture.run { buffer in
            try occlusion.encodeVisibility(
                depth: depth, surfaceAttributes: attributes, destination: destination, camera: camera,
                cameraTransform: cameraTransform, settings: settings, commandBuffer: buffer
            )
        }
        return (destination, try fixture.pixels(of: destination))
    }

    /// Cosine-weighted visibility of a floor point `distance` from an infinitely tall wall,
    /// counting wall points within `radius`: the half-disk of radius √(R² − d²) on the wall
    /// subtends a projected solid angle of atan(ρ/d) − ρd/R².
    static func cornerVisibility(distance d: Double, radius r: Double) -> Double {
        guard d < r else { return 1 }
        let rho = (r * r - d * d).squareRoot()
        return 1 - (atan(rho / d) - rho * d / (r * r)) / .pi
    }

    /// The cosine-weighted mean unoccluded direction at that point, as its tilt in radians
    /// from the floor normal away from the wall, by quadrature over the hemisphere.
    static func cornerBentTilt(distance d: Double, radius r: Double) -> Double {
        var mean = SIMD2<Double>.zero
        let rings = 400
        let sectors = 800
        for i in 0..<rings {
            let theta = (Double(i) + 0.5) / Double(rings) * .pi / 2
            for j in 0..<sectors {
                let phi = (Double(j) + 0.5) / Double(sectors) * 2 * .pi
                let towardWall = -sin(theta) * sin(phi)
                if towardWall > 0, d / towardWall <= r { continue }
                let weight = cos(theta) * sin(theta)
                mean += weight * SIMD2(sin(theta) * sin(phi), cos(theta))
            }
        }
        return atan2(mean.x, mean.y)
    }

    @Test("The prepass stores each surface's normal, roughness and split-sum specular reflectance")
    func surfaceAttributes() throws {
        let material = try PhysicallyBasedMaterial(roughness: 0.4)
        let graph = try corner(.physicallyBased(material))
        let (_, attributes) = try prepass(graph)
        var words = [SIMD2<UInt32>](repeating: .zero, count: fixture.width * fixture.height)
        try words.withUnsafeMutableBytes { bytes in
            attributes.getBytes(
                try #require(bytes.baseAddress), bytesPerRow: fixture.width * 8,
                from: MTLRegionMake2D(0, 0, fixture.width, fixture.height), mipmapLevel: 0
            )
        }
        func decode(_ word: SIMD2<UInt32>) -> (normal: SIMD3<Float>, roughness: Float, specular: SIMD3<Float>) {
            let x = Float(Int16(truncatingIfNeeded: word.x & 0xFFFF)) / 32767
            let y = Float(Int16(truncatingIfNeeded: word.x >> 16)) / 32767
            var n = SIMD3(x, y, 1 - abs(x) - abs(y))
            let t = max(-n.z, 0)
            n.x += n.x >= 0 ? -t : t
            n.y += n.y >= 0 ? -t : t
            let bytes = (0..<4).map { Float((word.y >> (8 * UInt32($0))) & 0xFF) / 255 }
            return (simd_normalize(n), bytes[3], SIMD3(bytes[0] * bytes[0], bytes[1] * bytes[1], bytes[2] * bytes[2]))
        }
        // Karis's fit of the split-sum terms for a dielectric with F0 = 0.04 and F90 = 1.
        func reflectance(roughness: Float, noV: Float) -> Float {
            let r = roughness * SIMD4<Float>(-1, -0.0275, -0.572, 0.022) + SIMD4<Float>(1, 0.0425, 1.04, -0.04)
            let a004 = min(r.x * r.x, exp2(-9.28 * noV)) * r.x + r.y
            let ab = SIMD2<Float>(-1.04, 1.04) * a004 + SIMD2(r.z, r.w)
            return 0.04 * ab.x + ab.y
        }
        for (x, y, normal) in [(128, 200, SIMD3<Float>(0, 1, 0)), (128, 60, SIMD3<Float>(0, 0, 1)), (40, 180, SIMD3<Float>(0, 1, 0))] {
            let attribute = decode(words[y * fixture.width + x])
            #expect(simd_distance(attribute.normal, normal) < 1e-3)
            #expect(abs(attribute.roughness - 0.4) < 0.5 / 255)
            let ray = fixture.ray(x: x, y: y, camera: camera, transform: cameraTransform)
            let expected = reflectance(roughness: 0.4, noV: simd_dot(-ray.direction, normal))
            // Square-root unorm8 storage resolves reflectance to about 2·√R/255.
            #expect(abs(attribute.specular.x - expected) < 2 * expected.squareRoot() / 255 + 1e-4)
            #expect(attribute.specular.x == attribute.specular.y && attribute.specular.y == attribute.specular.z)
        }
    }

    @Test("Surface attributes read graph coordinates and preserve PBR and classic alpha cutouts")
    func graphAndClassicSurfaceCoverage() throws {
        let graph = try SurfaceShaderGraph { s in .init(roughness: s.screenUV.x) }
        let geometry = try MeshGeometry.plane(width: 4, height: 4)
        for green: UInt8 in [0, 255] {
            let alpha = try fixture.constantTexture(SIMD4(255, green, 255, 255))
            let materials: [(MeshMaterial, Float)] = [
                (.physicallyBased(try .init(
                    alphaMode: .mask(cutoff: 0.5), surfaceGraph: graph, alphaTexture: .init(texture: alpha)
                )), 80.5 / Float(fixture.width)),
                (.classic(try .init(model: .basic, alphaMap: .init(texture: alpha), alphaTest: 0.5)), 1),
            ]
            for (material, expectedRoughness) in materials {
                let renderer = try MeshRenderer(
                    device: fixture.device, colorPixelFormat: .rgba16Float, geometry: geometry, material: material
                )
                let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -2)), mesh: renderer)
                let scene = SceneGraph(roots: [node], camera: camera)
                let (_, attributes) = try prepass(scene)
                var word = SIMD2<UInt32>.zero
                attributes.getBytes(&word, bytesPerRow: 8, from: MTLRegionMake2D(80, 128, 1, 1), mipmapLevel: 0)
                if green == 0 {
                    #expect(word == .zero)
                } else {
                    let roughness = Float(word.y >> 24) / 255
                    #expect(abs(roughness - expectedRoughness) < 0.51 / 255)
                }
            }
        }
    }

    @Test("Converged GTAO matches the analytic visibility and bent normal of a 90° corner and reads one on open floor")
    func cornerMatchesAnalytic() throws {
        let graph = try corner(.physicallyBased(try PhysicallyBasedMaterial()))
        let (depth, attributes) = try prepass(graph)
        let radius: Float = 0.5
        // The most slices and steps, unfiltered, with a hard radius: the closed form's assumptions.
        let settings = AmbientOcclusion.Settings(
            radius: radius, strength: 1, sliceCount: 8, stepCount: 32, falloff: 0, maximumScreenRadius: 256,
            denoise: false
        )
        let result = try visibility(depth: depth, attributes: attributes, settings: settings)
        fixture.writeImage(result.pixels.map { SIMD4(repeating: $0.w) }, name: "gtao-corner")

        let crease = fixture.project(.zero, camera: camera, transform: cameraTransform)
        var largestError: Double = 0
        var samples: [(distance: Double, measured: Float, expected: Double)] = []
        for column in [96, 128, 160] {
            for row in Int(crease.y.rounded(.up))..<fixture.height {
                let ray = fixture.ray(x: column, y: row, camera: camera, transform: cameraTransform)
                let t = -ray.origin.y / ray.direction.y
                let point = ray.origin + t * ray.direction
                let distance = Double(point.z)
                // Skip the pixels whose footprint straddles the crease.
                let footprint = Double(t) * 2 * tan(Double(camera.verticalFieldOfView) / 2) / Double(fixture.height)
                guard distance > 3 * footprint, distance < 1.5 * Double(radius) else { continue }
                let measured = result.pixels[row * fixture.width + column].w
                let expected = Self.cornerVisibility(distance: distance, radius: Double(radius))
                largestError = max(largestError, abs(Double(measured) - expected))
                samples.append((distance, measured, expected))
            }
        }
        #expect(samples.count > 60)
        #expect(largestError < 0.02)

        // At the crease the wall hides exactly half the cosine-weighted hemisphere; the
        // visibility grows as 0.5 + (2/π)·d/R away from it.
        let nearest = try #require(samples.min { $0.distance < $1.distance })
        #expect(abs(Double(nearest.measured) - (0.5 + 2 / Double.pi * nearest.distance / Double(radius))) < 0.01)

        // Beyond the radius the open floor is exactly unoccluded and its normal is not bent.
        let open = fixture.project(SIMD3(0, 0, 0.9), camera: camera, transform: cameraTransform)
        #expect(result.pixels[Int(open.y) * fixture.width + Int(open.x)] == SIMD4(0, 0, 0, 1))

        // Near the crease the bent normal tilts away from the wall by the quadrature angle.
        let near = fixture.project(SIMD3(0, 0, 0.1), camera: camera, transform: cameraTransform)
        let bent = Self.floorBentNormal(result.pixels[Int(near.y) * fixture.width + Int(near.x)])
        let ray = fixture.ray(x: Int(near.x), y: Int(near.y), camera: camera, transform: cameraTransform)
        let distance = Double((ray.origin + (-ray.origin.y / ray.direction.y) * ray.direction).z)
        let tilt = atan2(Double(bent.z), Double(bent.y))
        let expectedTilt = Self.cornerBentTilt(distance: distance, radius: Double(radius))
        #expect(abs(bent.x) < 0.02)
        #expect(abs(tilt - expectedTilt) < 1 * .pi / 180)
    }

    @Test(
        "An off-center, anisotropic projection, as a headset eye uses, still reproduces the analytic corner",
        arguments: [MeshDepthMode.forward, .reversed]
    )
    func offCenterProjection(depthMode: MeshDepthMode) throws {
        // A yawed view puts the crease diagonally on screen, so a misplaced principal point
        // or focal length would shear both planes and bend the floor away from its stored normal.
        let yaw: Float = 0.5
        let view = RasterLightingFixture.cameraTransform(
            position: SIMD3(2.5 * sin(yaw), 1.5, 2.5 * cos(yaw)), pitch: -atan2(1.5, 2.5), yaw: yaw
        )
        var graph = try corner(.physicallyBased(try PhysicallyBasedMaterial()), depthMode: depthMode)
        graph.cameraTransform = view
        let clipFromView = try RasterLightingFixture.offCenterProjection(
            camera, aspectRatio: 1, center: SIMD2(0.35, -0.25), horizontalScale: 1.3, depthMode: depthMode
        )
        let worldFromView = view.matrix
        let attributes = try fixture.texture(SceneGraph.surfaceAttributesPixelFormat, usage: [.renderTarget, .shaderRead])
        let depth = try fixture.depthTarget()
        try fixture.run { buffer in
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = attributes
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = depth
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.storeAction = .store
            pass.depthAttachment.clearDepth = depthMode == .forward ? 1 : 0
            try graph.encodeSurfaceAttributes(
                pass: pass, commandBuffer: buffer, clipFromWorld: clipFromView * simd_inverse(worldFromView),
                cameraPosition: view.translation, depthMode: depthMode
            )
        }
        let radius: Float = 0.5
        let occlusion = try AmbientOcclusion(device: fixture.device)
        let destination = try fixture.colorTarget()
        try fixture.run { buffer in
            try occlusion.encodeVisibility(
                depth: depth, surfaceAttributes: attributes, destination: destination, clipFromView: clipFromView,
                worldFromView: worldFromView, depthMode: depthMode,
                settings: .init(radius: radius, strength: 1, sliceCount: 8, stepCount: 32, falloff: 0, maximumScreenRadius: 256, denoise: false),
                commandBuffer: buffer
            )
        }
        let pixels = try fixture.pixels(of: destination)
        fixture.writeImage(pixels.map { SIMD4(repeating: $0.w) }, name: "gtao-off-center-\(depthMode)")

        var largestError: Double = 0
        var largestExcessOcclusion: Double = 0
        var samples = 0
        var clippedSamples = 0
        var openPixels = 0
        var openExact = 0
        for row in 0..<fixture.height {
            for column in stride(from: 8, to: fixture.width - 8, by: 16) {
                let ray = fixture.ray(x: column, y: row, clipFromView: clipFromView, worldFromView: worldFromView)
                let t = -ray.origin.y / ray.direction.y
                guard t > 0 else { continue }
                let point = ray.origin + t * ray.direction
                let distance = Double(point.z)
                let footprint = Double(t) * 2 * tan(Double(camera.verticalFieldOfView) / 2) / Double(fixture.height)
                let value = pixels[row * fixture.width + column]
                if distance > 3 * footprint, distance < 1.5 * Double(radius) {
                    let expected = Self.cornerVisibility(distance: distance, radius: Double(radius))
                    // The analytic wall is infinite, but the screen-space pass can only see
                    // its visible part. Project a bounding rectangle of the occluding half-disk.
                    let reach = sqrt(max(0, radius * radius - point.z * point.z))
                    let patch = [point.x - reach, point.x + reach].flatMap { x in
                        [Float(0), reach].map { y in
                            fixture.project(SIMD3(x, y, 0), clipFromView: clipFromView, worldFromView: worldFromView)
                        }
                    }
                    if patch.allSatisfy({
                        $0.x >= 0.5 && $0.x <= Float(fixture.width) - 0.5 &&
                        $0.y >= 0.5 && $0.y <= Float(fixture.height) - 0.5
                    }) {
                        largestError = max(largestError, abs(Double(value.w) - expected))
                        samples += 1
                    } else {
                        // Missing offscreen blockers can only make the estimate brighter.
                        largestExcessOcclusion = max(largestExcessOcclusion, expected - Double(value.w))
                        clippedSamples += 1
                    }
                } else if distance > 1.2 * Double(radius), distance < 3, abs(point.x) < 5 {
                    // Open floor, wholly inside the screen: reconstruction must agree with the stored normal.
                    openPixels += 1
                    if value == SIMD4(0, 0, 0, 1) { openExact += 1 }
                }
            }
        }
        #expect(samples > 60)
        #expect(largestError < 0.02)
        #expect(clippedSamples > 30)
        #expect(largestExcessOcclusion < 0.02)
        #expect(openPixels > 100)
        #expect(openExact == openPixels, "\(openExact) of \(openPixels) open floor pixels exact")

        let near = fixture.project(SIMD3(0, 0, 0.1), clipFromView: clipFromView, worldFromView: worldFromView)
        let bent = Self.floorBentNormal(pixels[Int(near.y) * fixture.width + Int(near.x)])
        let ray = fixture.ray(x: Int(near.x), y: Int(near.y), clipFromView: clipFromView, worldFromView: worldFromView)
        let distance = Double((ray.origin + (-ray.origin.y / ray.direction.y) * ray.direction).z)
        let tilt = atan2(Double(bent.z), Double(bent.y))
        #expect(abs(bent.x) < 0.02)
        #expect(abs(tilt - Self.cornerBentTilt(distance: distance, radius: Double(radius))) < 1 * .pi / 180)
        fixture.note(
            "off-center GTAO \(depthMode): \(samples) fully visible samples, largest error \(largestError), "
                + "\(clippedSamples) clipped samples, largest excess occlusion \(largestExcessOcclusion), "
                + "\(openExact) of \(openPixels) open pixels exact, tilt \(tilt * 180 / .pi)° "
                + "against \(Self.cornerBentTilt(distance: distance, radius: Double(radius)) * 180 / .pi)°"
        )
    }

    @Test("Explicit views reject projections and transforms the passes cannot invert")
    func explicitViewValidation() throws {
        let occlusion = try AmbientOcclusion(device: fixture.device)
        let depth = try fixture.depthTarget()
        let destination = try fixture.colorTarget()
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        let projection = try camera.projection(aspectRatio: 1)
        var invalidOrthographic = matrix_identity_float4x4
        invalidOrthographic.columns.2.z = -0.1
        invalidOrthographic.columns.3.z = 0.2 // Negative view-space near distance.
        var sheared = projection
        sheared.columns.1.x = 0.2
        var reversedMismatch = projection
        reversedMismatch.columns.2.z = -1 - projection.columns.2.z
        reversedMismatch.columns.3.z = -projection.columns.3.z
        var scaled = cameraTransform.matrix
        scaled.columns.0 *= 2
        for (clipFromView, worldFromView) in [
            (invalidOrthographic, cameraTransform.matrix), (sheared, cameraTransform.matrix),
            (reversedMismatch, cameraTransform.matrix), (projection, scaled),
        ] {
            #expect(throws: AmbientOcclusion.Failure.cameraRange) {
                try occlusion.encodeVisibility(
                    depth: depth, destination: destination, clipFromView: clipFromView, worldFromView: worldFromView,
                    commandBuffer: buffer
                )
            }
        }
    }

    private func orthographicProjection(_ mode: MeshDepthMode) -> simd_float4x4 {
        let reversed = mode == .reversed
        return simd_float4x4(
            SIMD4(1 / 3.2, 0, 0, 0), SIMD4(0, 1 / 2.2, 0, 0),
            SIMD4(0, 0, reversed ? 0.0625 : -0.0625, 0),
            SIMD4(0.23, -0.16, reversed ? 1 : 0, 1)
        )
    }

    @Test("Shifted orthographic GTAO keeps its world footprint and composites color and sky",
          arguments: [MeshDepthMode.forward, .reversed], [Float(1), 0.5])
    func orthographicVisibilityAndColor(mode: MeshDepthMode, scale: Float) throws {
        let projection = orthographicProjection(mode)
        let ao = try AmbientOcclusion(device: fixture.device)
        var graph = try corner(.unlit, depthMode: mode)
        graph.roots[0].transform.translation.z = 3
        graph.roots[0].transform.scale = SIMD3(0.3, 0.3, 1)
        graph.roots[1].transform.translation.y = 3
        graph.roots[1].transform.scale = SIMD3(0.3, 0.3, 1)
        let settings = AmbientOcclusion.Settings(
            radius: 0.5, strength: 1, sliceCount: 8, stepCount: 24, falloff: 0, resolutionScale: scale
        )
        let sky = SIMD4<Float>(0.125, 0.25, 0.5, 0.75)
        func render(_ pose: SceneTransform) throws -> (source: [SIMD4<Float>], visibility: [SIMD4<Float>], color: [SIMD4<Float>]) {
            let source = try fixture.colorTarget(), depth = try fixture.depthTarget()
            let visibility = try fixture.colorTarget(), destination = try fixture.colorTarget()
            try fixture.run { command in
                let pass = fixture.pass(color: source, depth: depth, depthMode: mode)
                pass.colorAttachments[0].clearColor = MTLClearColorMake(Double(sky.x), Double(sky.y), Double(sky.z), Double(sky.w))
                try graph.encode(
                    pass: pass, commandBuffer: command, clipFromWorld: projection * simd_inverse(pose.matrix),
                    cameraPosition: pose.translation, depthMode: mode
                )
                try ao.encodeVisibility(
                    depth: depth, destination: visibility, clipFromView: projection, worldFromView: pose.matrix,
                    depthMode: mode, settings: settings, commandBuffer: command
                )
                try ao.encode(
                    source: source, depth: depth, destination: destination, clipFromView: projection,
                    depthMode: mode, settings: settings, commandBuffer: command
                )
            }
            return (try fixture.pixels(of: source), try fixture.pixels(of: visibility), try fixture.pixels(of: destination))
        }

        let first = try render(cameraTransform)
        let displacement = cameraTransform.rotation.act(SIMD3<Float>(0, 0, 0.7))
        let moved = SceneTransform(translation: cameraTransform.translation + displacement, rotation: cameraTransform.rotation)
        let second = try render(moved)
        let point = fixture.project(SIMD3(0, 0, 0.12), clipFromView: projection, worldFromView: cameraTransform.matrix)
        let index = Int(point.y) * fixture.width + Int(point.x)
        let value = first.visibility[index]
        #expect((0.52...0.79).contains(value.w) && value.z > 0.05)
        #expect(simd_reduce_max(abs(value - second.visibility[index])) < 0.004)
        let expected = first.source[index] * SIMD4(value.w, value.w, value.w, 1)
        #expect(simd_reduce_max(abs(first.color[index] - expected)) < 0.002)
        let background = first.source.indices.filter { first.source[$0] == sky }
        #expect(background.count > 100)
        #expect(background.allSatisfy { first.color[$0] == sky && first.visibility[$0] == SIMD4(0, 0, 0, 1) })
        #expect(ao.horizonTextureSize == (scale == 1 ? SIMD2(256, 256) : SIMD2(128, 128)))
    }

    @Test("Default filtered GTAO settings stay within a few percent of the analytic corner")
    func defaultSettingsAccuracy() throws {
        let graph = try corner(.physicallyBased(try PhysicallyBasedMaterial()))
        let (depth, attributes) = try prepass(graph)
        var settings = AmbientOcclusion.Settings(radius: 0.5, strength: 1)
        settings.falloff = 0
        let result = try visibility(depth: depth, attributes: attributes, settings: settings)
        let crease = fixture.project(.zero, camera: camera, transform: cameraTransform)
        var errors: [Double] = []
        for row in Int(crease.y.rounded(.up))..<fixture.height {
            let ray = fixture.ray(x: 128, y: row, camera: camera, transform: cameraTransform)
            let t = -ray.origin.y / ray.direction.y
            let distance = Double((ray.origin + t * ray.direction).z)
            let footprint = Double(t) * 2 * tan(Double(camera.verticalFieldOfView) / 2) / Double(fixture.height)
            guard distance > 3 * footprint, distance < 0.75 else { continue }
            let expected = Self.cornerVisibility(distance: distance, radius: 0.5)
            errors.append(abs(Double(result.pixels[row * fixture.width + 128].w) - expected))
        }
        let mean = errors.reduce(0, +) / Double(errors.count)
        #expect(mean < 0.03)
        #expect((errors.max() ?? 1) < 0.06)
    }

    @Test("Depth-only GTAO reconstructs normals and still reads one on open floor")
    func depthOnlyNormals() throws {
        let graph = try corner(.physicallyBased(try PhysicallyBasedMaterial()))
        let (depth, attributes) = try prepass(graph)
        let settings = AmbientOcclusion.Settings(radius: 0.5, strength: 1, sliceCount: 4, stepCount: 16, falloff: 0)
        let withNormals = try visibility(depth: depth, attributes: attributes, settings: settings).pixels
        let fromDepth = try visibility(depth: depth, attributes: nil, settings: settings).pixels
        let open = fixture.project(SIMD3(0, 0, 0.9), camera: camera, transform: cameraTransform)
        #expect(fromDepth[Int(open.y) * fixture.width + Int(open.x)].w == 1)
        let near = fixture.project(SIMD3(0, 0, 0.12), camera: camera, transform: cameraTransform)
        let index = Int(near.y) * fixture.width + Int(near.x)
        #expect(fromDepth[index].w < 0.8)
        #expect(abs(fromDepth[index].w - withNormals[index].w) < 0.03)
        // Screen-border pixels reconstruct their normals from one side and are occluded too.
        let row = Int(near.y)
        for column in [0, fixture.width - 1] {
            let border = fromDepth[row * fixture.width + column].w
            #expect(border < 0.8)
            #expect(abs(border - withNormals[row * fixture.width + column].w) < 0.03)
        }
    }

    @Test("View thickness and normal bias reject crease occluders, and exponent remaps visibility")
    func horizonControls() throws {
        let (depth, attributes) = try prepass(corner(.unlit))
        var settings = AmbientOcclusion.Settings(radius: 0.5, strength: 1, sliceCount: 8, stepCount: 24, falloff: 0, denoise: false)
        let full = try visibility(depth: depth, attributes: attributes, settings: settings).pixels
        let point = fixture.project(SIMD3(0, 0, 0.12), camera: camera, transform: cameraTransform)
        let index = Int(point.y) * fixture.width + Int(point.x)
        let base = full[index]
        #expect(base.w > 0.5 && base.w < 0.8)
        settings.thickness = 0.005
        let thin = try visibility(depth: depth, attributes: attributes, settings: settings).pixels[index]
        #expect(thin.w > base.w + 0.12)
        settings.thickness = nil
        settings.normalBias = 0.36
        let biased = try visibility(depth: depth, attributes: attributes, settings: settings).pixels[index]
        #expect(biased.w > base.w + 0.05)
        settings.normalBias = 0.51
        #expect(try visibility(depth: depth, attributes: attributes, settings: settings).pixels[index] == SIMD4(0, 0, 0, 1))
        settings.normalBias = 0
        settings.visibilityExponent = 2
        let shaped = try visibility(depth: depth, attributes: attributes, settings: settings).pixels[index]
        #expect(abs(shaped.w - base.w * base.w) < 0.004)
        #expect(simd_length(SIMD3(shaped.x - base.x, shaped.y - base.y, shaped.z - base.z)) < 0.003)
        settings.thickness = .infinity
        #expect(throws: AmbientOcclusion.Failure.invalidSettings) {
            try visibility(depth: depth, attributes: attributes, settings: settings)
        }
    }

    @Test("Fixed pixel radius replaces the projected world radius and respects its screen cap")
    func fixedScreenRadius() throws {
        let (depth, attributes) = try prepass(corner(.unlit))
        let point = fixture.project(SIMD3(0, 0, 0.12), camera: camera, transform: cameraTransform)
        let index = Int(point.y) * fixture.width + Int(point.x)
        var settings = AmbientOcclusion.Settings(radius: 0.005, strength: 1, sliceCount: 8, stepCount: 24, falloff: 0)
        let subpixel = try visibility(depth: depth, attributes: attributes, settings: settings).pixels
        #expect(subpixel[index] == SIMD4(0, 0, 0, 1))
        settings.screenSpaceRadius = 48
        let fixed = try visibility(depth: depth, attributes: attributes, settings: settings).pixels
        #expect(fixed[index].w < 0.82)
        settings.maximumScreenRadius = 4
        let capped = try visibility(depth: depth, attributes: attributes, settings: settings).pixels
        #expect(capped[index] == SIMD4(0, 0, 0, 1))
        settings.screenSpaceRadius = 4
        settings.maximumScreenRadius = 128
        #expect(try visibility(depth: depth, attributes: attributes, settings: settings).pixels == capped)
        settings.resolutionScale = 0
        #expect(throws: AmbientOcclusion.Failure.invalidSettings) {
            try visibility(depth: depth, attributes: attributes, settings: settings)
        }
    }

    @Test("GTAO clips world-space receivers with a smooth radius fade and preserves HDR alpha",
          arguments: [false, true])
    func sceneClipMask(reducedPoisson: Bool) throws {
        let graph = try corner(.unlit)
        let projection = try camera.projection(aspectRatio: 1)
        let view = try ShaderGraphCamera(
            viewMatrix: simd_inverse(cameraTransform.matrix), projectionMatrix: projection,
            near: camera.near, far: camera.far
        )
        let ao = try AmbientOcclusion(device: fixture.device)
        let source = try fixture.colorTarget(), depth = try fixture.depthTarget()
        let unbounded = try fixture.colorTarget(), bounded = try fixture.colorTarget(), color = try fixture.colorTarget()
        let box = try AxisAlignedBox(minimum: SIMD3(-0.2, -0.05, -0.2), maximum: SIMD3(0.2, 0.05, 0.3))
        var settings = AmbientOcclusion.Settings(radius: 0.5, strength: 1, sliceCount: 6, stepCount: 16, falloff: 0)
        if reducedPoisson {
            settings.resolutionScale = 0.5
            settings.screenSpaceRadius = 48
            settings.poisson = .init()
        }
        try fixture.run { command in
            let scene = fixture.pass(color: source, depth: depth)
            try graph.encode(pass: scene, commandBuffer: command)
            try FramebufferClear(color: SIMD4(2.5, 1.25, 0.625, 0.375)).encode(in: scene, commandBuffer: command)
            try ao.encodeVisibility(
                depth: depth, destination: unbounded, camera: view, settings: settings, commandBuffer: command
            )
            settings.sceneClipBox = box
            try ao.encodeVisibility(
                depth: depth, destination: bounded, camera: view, settings: settings, commandBuffer: command
            )
            try ao.encode(
                source: source, depth: depth, destination: color, camera: view, settings: settings, commandBuffer: command
            )
        }
        let pixels = try fixture.pixels(of: source)
        let full = try fixture.pixels(of: unbounded)
        let clipped = try fixture.pixels(of: bounded)
        let shaded = try fixture.pixels(of: color)
        let open = SIMD4<Float>(0, 0, 0, 1)
        for x: Float in [0, 0.4, 0.85] {
            let pixel = fixture.project(SIMD3(x, 0, 0.12), camera: camera, transform: cameraTransform)
            let column = Int(pixel.x), row = Int(pixel.y)
            let address = row * fixture.width + column
            let ray = fixture.ray(x: column, y: row, camera: camera, transform: cameraTransform)
            let point = ray.origin - ray.direction * (ray.origin.y / ray.direction.y)
            let viewPoint = view.viewMatrix * SIMD4(point, 1)
            let focal = Float(fixture.height) / (2 * tan(camera.verticalFieldOfView * 0.5))
            let radius = settings.screenSpaceRadius.map { $0 * -viewPoint.z / focal } ?? settings.radius
            let t = min(1, Float(try box.distance(to: point)) / radius)
            let weight = 1 - t * t * (3 - 2 * t)
            let expected = open + (full[address] - open) * weight
            #expect(full[address].w < 0.9)
            #expect(simd_reduce_max(abs(expected - clipped[address])) < 0.002)
            if weight == 0 { #expect(clipped[address] == open) }
            if weight == 1 { #expect(clipped[address] == full[address]) }
            let visibility = clipped[address].w
            #expect(simd_reduce_max(abs(shaded[address] - pixels[address] * SIMD4(visibility, visibility, visibility, 1))) < 0.004)
            #expect(shaded[address].w == 0.375)
        }
        #expect(pixels.indices.allSatisfy { pixels[$0].w == shaded[$0].w })
    }

    @Test("GTAO horizon spacing changes samples and bounded history respects the prior box")
    func boundedHistoryAndDistanceExponent() throws {
        let (depth, attributes) = try prepass(corner(.unlit))
        let view = try ShaderGraphCamera(
            viewMatrix: simd_inverse(cameraTransform.matrix), projectionMatrix: camera.projection(aspectRatio: 1),
            near: camera.near, far: camera.far
        )
        let ao = try AmbientOcclusion(device: fixture.device)
        var settings = AmbientOcclusion.Settings(radius: 0.5, strength: 1, sliceCount: 2, stepCount: 5, denoise: false)
        settings.sceneClipBox = try AxisAlignedBox(minimum: SIMD3(-0.2, -0.05, -0.2), maximum: SIMD3(0.2, 0.05, 0.3))
        func render(
            _ settings: AmbientOcclusion.Settings, index: UInt64, prior: AmbientOcclusion.TemporalFrame.Previous? = nil
        ) throws -> any MTLTexture {
            let target = try fixture.colorTarget()
            try fixture.run { command in
                try ao.encodeVisibility(
                    depth: depth, surfaceAttributes: attributes, destination: target, camera: view,
                    settings: settings, temporal: .init(index: index, previous: prior, historyWeight: 0.8), commandBuffer: command
                )
            }
            return target
        }
        let first = try render(settings, index: 0)
        let current = try fixture.pixels(of: render(settings, index: 1))
        let validHistory = AmbientOcclusion.TemporalFrame.Previous(
            visibility: first, depth: depth, surfaceAttributes: attributes, camera: view, settings: settings
        )
        let history = try fixture.pixels(of: render(settings, index: 1, prior: validHistory))
        let open = SIMD4<Float>(0, 0, 0, 1)
        #expect(history != current)
        let outside = fixture.project(SIMD3(0.85, 0, 0.12), camera: camera, transform: cameraTransform)
        let outsideAddress = Int(outside.y) * fixture.width + Int(outside.x)
        #expect(history[outsideAddress] == open)
        let omittedSettings = AmbientOcclusion.TemporalFrame.Previous(
            visibility: first, depth: depth, surfaceAttributes: attributes, camera: view
        )
        #expect(try fixture.pixels(of: render(settings, index: 1, prior: omittedSettings)) == current)
        var shifted = settings
        shifted.sceneClipBox = try AxisAlignedBox(minimum: SIMD3(0.2, -0.05, -0.2), maximum: SIMD3(0.5, 0.05, 0.3))
        let fresh = try fixture.pixels(of: render(shifted, index: 1))
        #expect(try fixture.pixels(of: render(shifted, index: 1, prior: validHistory)) == fresh)
        settings.sampleDistanceExponent = 1
        #expect(try fixture.pixels(of: render(settings, index: 1)) != current)
        settings.sampleDistanceExponent = 0
        #expect(throws: AmbientOcclusion.Failure.invalidSettings) { try render(settings, index: 1) }
    }

    @Test("Poisson spiral offsets and visibility weights match an independent floor reference")
    func poissonReference() throws {
        let (depth, attributes) = try prepass(corner(.unlit))
        var settings = AmbientOcclusion.Settings(
            radius: 0.5, strength: 1, sliceCount: 3, stepCount: 8, falloff: 0, denoise: false
        )
        let raw = try visibility(depth: depth, attributes: attributes, settings: settings).pixels
        let poisson = AmbientOcclusion.PoissonSettings(
            sampleCount: 12, rings: 3, radius: 8, radiusExponent: 1.4, lumaPhi: 0.16,
            depthPhi: 0.1, normalPhi: 5, sampleIndex: 1
        )
        settings.poisson = poisson
        #expect(try visibility(depth: depth, attributes: attributes, settings: settings).pixels == raw)
        settings.denoise = true
        let filtered = try visibility(depth: depth, attributes: attributes, settings: settings).pixels

        let projected = fixture.project(SIMD3(0, 0, 0.32), camera: camera, transform: cameraTransform)
        let x = Int(projected.x), y = Int(projected.y)
        let index = y * fixture.width + x
        let bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
        let rotation = 2 * Double.pi * ((Double(bayer[(y % 4) * 4 + x % 4]) + 0.5) / 16 + Double(poisson.sampleIndex) / 4)
        var reference = raw[index]
        var total: Float = 1
        for tap in 0..<poisson.sampleCount {
            let angle = 2 * Double.pi * Double(poisson.rings * tap) / Double(poisson.sampleCount) + rotation
            let ring = pow(Double(tap) / Double(poisson.sampleCount - 1), Double(poisson.radiusExponent))
            let distance = 1 + (Double(poisson.radius) - 1) * ring
            let px = Int(floor(Double(x) + 0.5 + cos(angle) * distance))
            let py = Int(floor(Double(y) + 0.5 + sin(angle) * distance))
            let neighbor = raw[py * fixture.width + px]
            // These taps all stay on the same planar floor, so depth and normal weights are one.
            let weight = max(1 - abs(neighbor.w - raw[index].w) / poisson.lumaPhi, 0)
            reference += neighbor * weight
            total += weight
        }
        reference /= total
        #expect(simd_reduce_max(abs(filtered[index] - reference)) < 0.004)
        #expect(simd_reduce_max(abs(filtered[index] - raw[index])) > 0.005)
        settings.poisson?.sampleIndex = 2
        let rotated = try visibility(depth: depth, attributes: attributes, settings: settings).pixels
        #expect(zip(filtered, rotated).contains { abs($0.0.w - $0.1.w) > 0.02 })
        settings.poisson?.depthPhi = 0
        #expect(throws: AmbientOcclusion.Failure.invalidSettings) {
            try visibility(depth: depth, attributes: attributes, settings: settings)
        }
    }

    @Test("Logarithmic raster depth preserves a distant GTAO corner and HDR color",
          arguments: [false, true])
    func logarithmicOcclusion(explicitView: Bool) async throws {
        let distanceScale: Float = 125_000
        let wideCamera = try PerspectiveCamera(verticalFieldOfView: camera.verticalFieldOfView, near: 0.0005, far: 5e8)
        let pose = SceneTransform(translation: cameraTransform.translation * distanceScale, rotation: cameraTransform.rotation)
        let ao = try AmbientOcclusion(device: fixture.device)
        let scale: Float = explicitView ? 0.5 : 1
        let settings = AmbientOcclusion.Settings(
            radius: 0.5 * distanceScale, strength: 1, sliceCount: 8, stepCount: 20, falloff: 0, denoise: true,
            resolutionScale: scale, poisson: explicitView ? .init(radius: 3, depthPhi: 0.05 * distanceScale) : nil
        )
        let source = try fixture.colorTarget()
        let hdr = SIMD4<Float16>(3.5, 0.5, 1.25, 0.375)
        let pixels = [SIMD4<Float16>](repeating: hdr, count: fixture.width * fixture.height)
        pixels.withUnsafeBytes {
            source.replace(region: MTLRegionMake2D(0, 0, fixture.width, fixture.height), mipmapLevel: 0,
                           withBytes: $0.baseAddress!, bytesPerRow: fixture.width * MemoryLayout<SIMD4<Float16>>.stride)
        }
        func render(_ mode: MeshDepthMode) throws -> (
            depth: any MTLTexture, projection: simd_float4x4, visibility: [SIMD4<Float>], color: [SIMD4<Float>]
        ) {
            var graph = try corner(.unlit, depthMode: mode)
            graph.roots = [SceneNode(transform: .init(scale: SIMD3(repeating: distanceScale)), children: graph.roots)]
            graph.camera = wideCamera
            graph.cameraTransform = pose
            var projection = try wideCamera.projection(aspectRatio: 1, depthMode: mode)
            if explicitView {
                projection = TemporalAntiAliasing.jitteredClipFromWorld(
                    projection, jitter: SIMD2(6.35, -4.25), width: Double(fixture.width), height: Double(fixture.height)
                )
            }
            let view = try ShaderGraphCamera(
                viewMatrix: simd_inverse(pose.matrix), projectionMatrix: projection, near: wideCamera.near, far: wideCamera.far
            )
            let depth = try fixture.depthTarget(), output = try fixture.colorTarget(), color = try fixture.colorTarget()
            let attributes = try fixture.texture(SceneGraph.surfaceAttributesPixelFormat, usage: [.renderTarget, .shaderRead])
            let scissor = MTLScissorRect(x: 2, y: 2, width: fixture.width - 4, height: fixture.height - 4)
            try fixture.run { command in
                let pass = fixture.pass(color: attributes, depth: depth, depthMode: mode)
                if mode == .logarithmic { pass.depthAttachment.clearDepth = 1 }
                if explicitView {
                    try graph.encodeSurfaceAttributes(
                        pass: pass, commandBuffer: command, clipFromWorld: projection * view.viewMatrix,
                        cameraPosition: pose.translation, depthMode: mode, graphCamera: view, scissor: scissor
                    )
                    try ao.encodeVisibility(
                        depth: depth, destination: output, camera: view, depthMode: mode, settings: settings, commandBuffer: command
                    )
                    try ao.encode(
                        source: source, depth: depth, destination: color, camera: view, depthMode: mode,
                        settings: settings, commandBuffer: command
                    )
                } else {
                    try graph.encodeSurfaceAttributes(pass: pass, commandBuffer: command, depthMode: mode, scissor: scissor)
                    try ao.encodeVisibility(
                        depth: depth, destination: output, camera: wideCamera, cameraTransform: pose, depthMode: mode,
                        settings: settings, commandBuffer: command
                    )
                    try ao.encode(
                        source: source, depth: depth, destination: color, camera: wideCamera, depthMode: mode,
                        settings: settings, commandBuffer: command
                    )
                }
            }
            return (depth, projection, try fixture.pixels(of: output), try fixture.pixels(of: color))
        }
        let logarithmic = try render(.logarithmic)
        let reversed = try render(.reversed)
        #expect(logarithmic.projection[2].z == -1, "The finite far plane is lost in the forward Float projection")
        #expect(ao.horizonTextureSize == (explicitView ? SIMD2(128, 128) : SIMD2(256, 256)))
        #expect(logarithmic.visibility[0] == SIMD4(0, 0, 0, 1))
        #expect(logarithmic.color[0] == SIMD4<Float>(hdr))
        #expect(logarithmic.color.allSatisfy { $0.w == Float(hdr.w) })
        for distance: Float in [0.12, 0.22, 0.8] {
            let point = SIMD3<Float>(0, 0, distance * distanceScale)
            let screen = fixture.project(point, clipFromView: logarithmic.projection, worldFromView: pose.matrix)
            let address = Int(screen.x) + Int(screen.y) * fixture.width
            let bend = logarithmic.visibility[address]
            #expect(simd_reduce_max(abs(bend - reversed.visibility[address])) < 0.012)
            #expect(abs(Double(bend.w) - Self.cornerVisibility(distance: Double(distance), radius: 0.5)) < 0.07)
            #expect(abs(logarithmic.color[address].x - Float(hdr.x) * bend.w) < 0.006)
            if distance < 0.3 { #expect(bend.w < 0.9 && bend.z > 0.05) }
            let raw = try await TextureTransfer.read(
                from: logarithmic.depth, region: MTLRegionMake2D(Int(screen.x), Int(screen.y), 1, 1), commandQueue: fixture.queue
            ).bytes.withUnsafeBytes { $0.loadUnaligned(as: Float.self) }
            let viewDistance = -(simd_inverse(pose.matrix) * SIMD4(point, 1)).z
            let expectedRaw = log2(Double(viewDistance) / Double(wideCamera.near)) / log2(Double(wideCamera.far) / Double(wideCamera.near))
            #expect(abs(Double(raw) - expectedRaw) < 0.001)
        }
        let command = try #require(fixture.queue.makeCommandBuffer())
        let destination = try fixture.colorTarget()
        #expect(throws: AmbientOcclusion.Failure.cameraRange) {
            try ao.encode(source: source, depth: logarithmic.depth, destination: destination,
                          clipFromView: logarithmic.projection, depthMode: .logarithmic, commandBuffer: command)
        }
    }

    @Test("Logarithmic perspective history mixes with reversed and linear orthographic frames")
    func logarithmicTemporal() throws {
        try verifyTemporal(mode: .logarithmic, scale: 0.5, priorOrthographic: false, currentOrthographic: false,
                           poisson: .init(radius: 4, depthPhi: 0.1), priorDepthMode: .logarithmic)
        try verifyTemporal(mode: .reversed, scale: 1, priorOrthographic: false, currentOrthographic: false,
                           priorDepthMode: .logarithmic)
        try verifyTemporal(mode: .logarithmic, scale: 0.5, priorOrthographic: true, currentOrthographic: false,
                           priorDepthMode: .reversed)
        try verifyTemporal(mode: .logarithmic, scale: 0.5, priorOrthographic: false, currentOrthographic: true,
                           priorDepthMode: .logarithmic)
    }

    @Test("Poisson visibility stays compatible with reduced mixed-projection temporal history",
          arguments: [MeshDepthMode.forward, .reversed])
    func poissonTemporal(mode: MeshDepthMode) throws {
        try verifyTemporal(
            mode: mode, scale: mode == .forward ? 1 : 0.5, priorOrthographic: true, currentOrthographic: false,
            poisson: .init(radius: 4, depthPhi: 0.1)
        )
    }

    @Test("Temporal GTAO reprojects world bends and rejects prior depth and normal discontinuities",
          arguments: [MeshDepthMode.forward, .reversed], [Float(1), 0.5])
    func temporalOcclusion(mode: MeshDepthMode, scale: Float) throws {
        try verifyTemporal(mode: mode, scale: scale, priorOrthographic: false, currentOrthographic: false)
    }

    @Test("Reduced temporal GTAO follows orthographic and mixed projection history",
          arguments: [MeshDepthMode.forward, .reversed])
    func orthographicTemporal(mode: MeshDepthMode) throws {
        try verifyTemporal(mode: mode, scale: 0.5, priorOrthographic: true, currentOrthographic: true)
        try verifyTemporal(mode: mode, scale: 0.5, priorOrthographic: true, currentOrthographic: false)
        try verifyTemporal(mode: mode, scale: 0.5, priorOrthographic: false, currentOrthographic: true)
    }

    private func verifyTemporal(
        mode: MeshDepthMode, scale: Float, priorOrthographic: Bool, currentOrthographic: Bool,
        poisson: AmbientOcclusion.PoissonSettings? = nil, priorDepthMode: MeshDepthMode? = nil
    ) throws {
        let previousMode = priorDepthMode ?? (mode == .forward ? .reversed : .forward)
        let earlier = try corner(.unlit, depthMode: previousMode)
        var later = try corner(.unlit, depthMode: mode)
        later.roots.removeLast()
        // Extend the now-open floor past the old corner to remove its silhouette.
        later.roots[0].transform.translation.z = 0
        let currentPose = SceneTransform(
            translation: cameraTransform.translation + SIMD3(0.13, 0, 0.08), rotation: cameraTransform.rotation
        )
        let oldProjection = try priorOrthographic ? orthographicProjection(previousMode)
            : camera.projection(aspectRatio: 1, depthMode: previousMode)
        let baseProjection = try currentOrthographic ? orthographicProjection(mode)
            : camera.projection(aspectRatio: 1, depthMode: mode)
        let newProjection = TemporalAntiAliasing.jitteredClipFromWorld(
            baseProjection, jitter: SIMD2(0.35, -0.25), width: Double(fixture.width), height: Double(fixture.height)
        )
        let oldCamera = try ShaderGraphCamera(
            viewMatrix: simd_inverse(cameraTransform.matrix), projectionMatrix: oldProjection,
            near: priorOrthographic ? 0 : camera.near, far: priorOrthographic ? 16 : camera.far
        )
        let settings = AmbientOcclusion.Settings(
            radius: 0.5, strength: 1, sliceCount: 6, stepCount: 16, falloff: 0, resolutionScale: scale, poisson: poisson
        )
        let ao = try AmbientOcclusion(device: fixture.device)
        func capture(
            _ graph: SceneGraph, pose: SceneTransform, projection: simd_float4x4, mode: MeshDepthMode,
            frame: AmbientOcclusion.TemporalFrame
        ) throws -> (visibility: any MTLTexture, depth: any MTLTexture, attributes: any MTLTexture, pixels: [SIMD4<Float>]) {
            let depth = try fixture.depthTarget(), output = try fixture.colorTarget()
            let attributes = try fixture.texture(SceneGraph.surfaceAttributesPixelFormat, usage: [.renderTarget, .shaderRead])
            let view = simd_inverse(pose.matrix)
            let graphCamera = try ShaderGraphCamera(
                viewMatrix: view, projectionMatrix: projection, near: projection[2].w == 0 ? 0 : camera.near,
                far: projection[2].w == 0 ? 16 : camera.far
            )
            try fixture.run { command in
                let pass = fixture.pass(color: attributes, depth: depth, depthMode: mode)
                if mode == .logarithmic { pass.depthAttachment.clearDepth = 1 }
                try graph.encodeSurfaceAttributes(
                    pass: pass, commandBuffer: command,
                    clipFromWorld: projection * view, cameraPosition: pose.translation, depthMode: mode,
                    graphCamera: mode == .logarithmic ? graphCamera : nil
                )
                if mode == .logarithmic {
                    try ao.encodeVisibility(
                        depth: depth, surfaceAttributes: attributes, destination: output, camera: graphCamera, depthMode: mode,
                        settings: settings, temporal: frame, commandBuffer: command
                    )
                } else {
                    try ao.encodeVisibility(
                        depth: depth, surfaceAttributes: attributes, destination: output,
                        clipFromView: projection, worldFromView: pose.matrix, depthMode: mode,
                        settings: settings, temporal: frame, commandBuffer: command
                    )
                }
            }
            return (output, depth, attributes, try fixture.pixels(of: output))
        }

        let first = try capture(earlier, pose: cameraTransform, projection: oldProjection, mode: previousMode, frame: .init(index: 0))
        #expect(ao.horizonTextureSize == (scale == 1 ? SIMD2(256, 256) : SIMD2(128, 128)))
        #expect(first.visibility.width == 256 && first.visibility.height == 256)
        let previous = AmbientOcclusion.TemporalFrame.Previous(
            visibility: first.visibility, depth: first.depth, surfaceAttributes: first.attributes, camera: oldCamera, depthMode: previousMode
        )
        let phase = AmbientOcclusion.TemporalFrame(index: 1, previous: previous)
        let second = try capture(later, pose: currentPose, projection: newProjection, mode: mode, frame: phase)
        let clean = try capture(later, pose: currentPose, projection: newProjection, mode: mode, frame: .init(index: 1))
        func sample(_ data: [SIMD4<Float>], world: SIMD3<Float>) -> SIMD4<Float> {
            let p = fixture.project(world, clipFromView: newProjection, worldFromView: currentPose.matrix)
            return data[Int(p.y) * fixture.width + Int(p.x)]
        }
        let receiver = SIMD3<Float>(0, 0, 0.12)
        let carried = sample(second.pixels, world: receiver)
        #expect(sample(clean.pixels, world: receiver) == SIMD4(0, 0, 0, 1))
        #expect(carried.w < 0.9 && carried.z > 0.05)
        let reconstructed = AmbientOcclusion.TemporalFrame.Previous(
            visibility: first.visibility, depth: first.depth, camera: oldCamera, depthMode: previousMode
        )
        let depthOnly = try capture(
            later, pose: currentPose, projection: newProjection, mode: mode, frame: .init(index: 1, previous: reconstructed)
        )
        #expect(simd_reduce_max(abs(sample(depthOnly.pixels, world: receiver) - carried)) < 0.01)

        let alternate = try fixture.texture(SceneGraph.surfaceAttributesPixelFormat, usage: [.shaderRead], storage: .shared)
        let normals = [SIMD2<UInt32>](repeating: .zero, count: fixture.width * fixture.height) // Packed +Z.
        normals.withUnsafeBytes {
            alternate.replace(region: MTLRegionMake2D(0, 0, fixture.width, fixture.height), mipmapLevel: 0,
                              withBytes: $0.baseAddress!, bytesPerRow: fixture.width * MemoryLayout<SIMD2<UInt32>>.stride)
        }
        let wrongNormal = AmbientOcclusion.TemporalFrame.Previous(
            visibility: first.visibility, depth: first.depth, surfaceAttributes: alternate, camera: oldCamera, depthMode: previousMode
        )
        let differentNormal = try capture(
            later, pose: currentPose, projection: newProjection, mode: mode, frame: .init(index: 1, previous: wrongNormal)
        )
        #expect(sample(differentNormal.pixels, world: receiver) == SIMD4(0, 0, 0, 1))

        later.roots[0].transform.translation.y = -0.45
        let uncovered = try capture(later, pose: currentPose, projection: newProjection, mode: mode, frame: phase)
        #expect(sample(uncovered.pixels, world: SIMD3(0, -0.45, 0.12)) == SIMD4(0, 0, 0, 1))

        let command = try #require(fixture.queue.makeCommandBuffer())
        #expect(throws: AmbientOcclusion.Failure.history) {
            try ao.encodeVisibility(
                depth: first.depth, surfaceAttributes: first.attributes, destination: first.visibility,
                camera: oldCamera, depthMode: previousMode,
                settings: settings, temporal: phase, commandBuffer: command
            )
        }
        if previousMode == .logarithmic {
            let destination = try fixture.colorTarget()
            #expect(throws: AmbientOcclusion.Failure.history) {
                try ao.encodeVisibility(
                    depth: first.depth, surfaceAttributes: first.attributes, destination: destination,
                    camera: oldCamera, depthMode: previousMode, settings: settings,
                    temporal: .init(index: 1, previous: previous, motionTexture: second.visibility), commandBuffer: command
                )
            }
        }
    }

    @Test("Raster motion carries a moved receiver's GTAO and rejects its disoccluded prior depth",
          arguments: [MeshDepthMode.forward, .reversed], [false, true])
    func movingOcclusionReceiver(mode: MeshDepthMode, orthographic: Bool) throws {
        var earlier = try corner(.unlit, depthMode: mode)
        var later = earlier
        later.roots.removeLast()
        let receiver = SIMD3<Float>(-0.48, 0, 0.14)
        let hidden = SIMD3<Float>(0.59, 0, 0.14)
        // Same normal as the old floor, with a closer depth on the hidden receiver's ray.
        let blocker = orthographic ? hidden + (cameraTransform.translation / 24)
            : hidden + (cameraTransform.translation - hidden) * 0.05
        earlier.roots.append(.init(
            transform: .init(translation: blocker, rotation: earlier.roots[0].transform.rotation, scale: SIMD3(0.025, 0.009, 1)),
            mesh: earlier.roots[0].mesh
        ))
        let displacement = SIMD3<Float>(0.39, 0.32, 0.61)
        later.roots[0].transform.translation += displacement
        let currentPose = SceneTransform(translation: cameraTransform.translation + SIMD3(0.11, 0, 0),
                                         rotation: cameraTransform.rotation)
        let baseProjection = try orthographic ? orthographicProjection(mode) : camera.projection(aspectRatio: 1, depthMode: mode)
        func projection(_ jitter: SIMD2<Float>) -> simd_float4x4 {
            TemporalAntiAliasing.jitteredClipFromWorld(
                baseProjection, jitter: jitter, width: Double(fixture.width), height: Double(fixture.height)
            )
        }
        let oldProjection = projection(SIMD2(0.15, -0.35))
        let newProjection = projection(SIMD2(-0.35, 0.25))
        let oldCamera = try ShaderGraphCamera(
            viewMatrix: simd_inverse(cameraTransform.matrix), projectionMatrix: oldProjection,
            near: orthographic ? 0 : camera.near, far: orthographic ? 16 : camera.far
        )
        let ao = try AmbientOcclusion(device: fixture.device)
        let recorder = try RasterMotionVectors(queue: fixture.queue)
        let settings = AmbientOcclusion.Settings(radius: 0.5, strength: 1, sliceCount: 6, stepCount: 16,
                                                  falloff: 0, normalBias: 0.002, resolutionScale: orthographic ? 0.5 : 1)
        func capture(
            _ graph: SceneGraph, pose: SceneTransform, clipFromView: simd_float4x4, index: UInt64,
            prior: RasterMotionVectors.History? = nil
        ) throws -> (visibility: any MTLTexture, depth: any MTLTexture, attributes: any MTLTexture,
                     vectors: any MTLTexture, history: RasterMotionVectors.History) {
            let depth = try fixture.depthTarget(), output = try fixture.colorTarget()
            let color = try fixture.colorTarget(), vectors = try fixture.colorTarget()
            let attributes = try fixture.texture(SceneGraph.surfaceAttributesPixelFormat, usage: [.renderTarget, .shaderRead])
            let recording = recorder.recording(to: vectors, previous: prior)
            let clip = clipFromView * simd_inverse(pose.matrix)
            try fixture.run { command in
                try graph.encodeSurfaceAttributes(
                    pass: fixture.pass(color: attributes, depth: depth, depthMode: mode), commandBuffer: command,
                    clipFromWorld: clip, cameraPosition: pose.translation, depthMode: mode
                )
                try graph.encode(
                    pass: fixture.pass(color: color, depth: depth, depthMode: mode), commandBuffer: command,
                    clipFromWorld: clip, cameraPosition: pose.translation, depthMode: mode, motionVectors: recording
                )
                try ao.encodeVisibility(
                    depth: depth, surfaceAttributes: attributes, destination: output,
                    clipFromView: clipFromView, worldFromView: pose.matrix, depthMode: mode,
                    settings: settings, temporal: .init(index: index), commandBuffer: command
                )
            }
            return (output, depth, attributes, vectors, try #require(recording.history))
        }
        let first = try capture(earlier, pose: cameraTransform, clipFromView: oldProjection, index: 0)
        let second = try capture(later, pose: currentPose, clipFromView: newProjection, index: 1, prior: first.history)
        let previous = AmbientOcclusion.TemporalFrame.Previous(
            visibility: first.visibility, depth: first.depth, surfaceAttributes: first.attributes,
            camera: oldCamera, depthMode: mode
        )
        let clean = try fixture.pixels(of: second.visibility)
        func resolve(_ frame: AmbientOcclusion.TemporalFrame) throws -> [SIMD4<Float>] {
            let output = try fixture.colorTarget()
            try fixture.run { command in
                try ao.encodeVisibility(
                    depth: second.depth, surfaceAttributes: second.attributes, destination: output,
                    clipFromView: newProjection, worldFromView: currentPose.matrix, depthMode: mode,
                    settings: settings, temporal: frame, commandBuffer: command
                )
            }
            return try fixture.pixels(of: output)
        }
        let withMotion = try resolve(.init(index: 1, previous: previous, motionTexture: second.vectors))
        let cameraOnly = try resolve(.init(index: 1, previous: previous))
        #expect(try resolve(.init(index: 1, motionTexture: second.vectors)) == clean)
        #expect(try resolve(.init(index: 1, previous: previous, motionTexture: first.vectors)) == clean)

        func address(_ p: SIMD2<Float>) -> Int { Int(p.x) + Int(p.y) * fixture.width }
        func currentPixel(_ oldPoint: SIMD3<Float>) -> SIMD2<Float> {
            fixture.project(oldPoint + displacement, clipFromView: newProjection, worldFromView: currentPose.matrix)
        }
        let receiverPixel = address(currentPixel(receiver))
        let hiddenPixel = address(currentPixel(hidden))
        let vectors = try fixture.pixels(of: second.vectors)
        let oldAO = try fixture.pixels(of: first.visibility)
        let oldReceiver = fixture.project(receiver, clipFromView: oldProjection, worldFromView: cameraTransform.matrix)
        let oldHidden = fixture.project(hidden, clipFromView: oldProjection, worldFromView: cameraTransform.matrix)
        for (pixel, priorPoint, point) in [(receiverPixel, oldReceiver, receiver), (hiddenPixel, oldHidden, hidden)] {
            let xy = SIMD2(Float(pixel % fixture.width) + 0.5, Float(pixel / fixture.width) + 0.5)
            let motion = vectors[pixel]
            let clip = oldCamera.projectionMatrix * oldCamera.viewMatrix * SIMD4(point, 1)
            #expect(motion.w == 1)
            #expect(simd_distance(xy + SIMD2(motion.x, motion.y), priorPoint) < 1.4)
            #expect(abs(motion.z - clip.z / clip.w) < 0.001)
            #expect(oldAO[address(priorPoint)].w < 0.94)
        }
        // The rasterized plane can have one half-float step of residual occlusion.
        #expect(simd_reduce_max(abs(clean[receiverPixel] - SIMD4(0, 0, 0, 1))) < 0.001)
        #expect(cameraOnly[receiverPixel] == clean[receiverPixel])
        #expect(withMotion[receiverPixel].w < 0.91 && withMotion[receiverPixel].z > 0.03)
        #expect(withMotion[hiddenPixel] == clean[hiddenPixel])

        let vector = vectors[receiverPixel]
        let xy = SIMD2(Float(receiverPixel % fixture.width), Float(receiverPixel / fixture.width))
        let priorTexel = xy + SIMD2(vector.x, vector.y)
        let corner = floor(priorTexel)
        let fraction = priorTexel - corner
        let top = oldAO[address(corner)] * (1 - fraction.x) + oldAO[address(corner + SIMD2(1, 0))] * fraction.x
        let bottom = oldAO[address(corner + SIMD2(0, 1))] * (1 - fraction.x) + oldAO[address(corner + SIMD2(1, 1))] * fraction.x
        let expected = clean[receiverPixel] * 0.15 + (top * (1 - fraction.y) + bottom * fraction.y) * 0.85
        #expect(simd_reduce_max(abs(withMotion[receiverPixel] - expected)) < 0.001)

        let badVectors = try fixture.colorTarget()
        let rect = MTLRegionMake2D(Int(xy.x), Int(xy.y), 1, 1)
        for bad in [
            SIMD4<Float16>(Float16(vector.x), Float16(vector.y), Float16(vector.z), 0),
            SIMD4<Float16>(Float16(vector.x), Float16(vector.y), 2, 1),
            SIMD4<Float16>(Float16(vector.x), Float16(vector.y), .nan, 1),
            SIMD4<Float16>(-1024, Float16(vector.y), Float16(vector.z), 1),
        ] {
            withUnsafeBytes(of: bad) {
                badVectors.replace(region: rect, mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: $0.count)
            }
            let rejected = try resolve(.init(index: 1, previous: previous, motionTexture: badVectors))
            #expect(rejected[receiverPixel] == clean[receiverPixel])
        }

        let command = try #require(fixture.queue.makeCommandBuffer())
        for invalid in [second.visibility, second.depth] {
            #expect(throws: AmbientOcclusion.Failure.history) {
                try ao.encodeVisibility(
                    depth: second.depth, surfaceAttributes: second.attributes, destination: second.visibility,
                    clipFromView: newProjection, worldFromView: currentPose.matrix, depthMode: mode,
                    settings: settings, temporal: .init(index: 1, previous: previous, motionTexture: invalid), commandBuffer: command
                )
            }
        }
    }

    @Test("Screen-space occlusion darkens only image-based light, by the multiple-bounce visibility")
    func pbrIndirectOcclusion() throws {
        let width = 16
        let height = 8
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let panorama = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let texels = [SIMD4<Float16>](repeating: SIMD4(0.8, 0.8, 0.8, 1), count: width * height)
        try texels.withUnsafeBytes { bytes in
            panorama.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: try #require(bytes.baseAddress), bytesPerRow: width * 8
            )
        }
        let lighting = try EnvironmentLighting(
            queue: fixture.queue, equirectangular: panorama,
            settings: .init(cubeSize: 16, diffuseSize: 8, brdfSize: 16, sampleCount: 32)
        )
        let albedo = SIMD3<Float>(0.7, 0.45, 0.2)
        // Zero specular leaves only diffuse image-based light.
        let diffuse = try PhysicallyBasedMaterial(
            roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(albedo, 1),
            environment: try .init(lighting: lighting), specular: try .init(factor: 0)
        )
        let graph = try corner(.physicallyBased(diffuse))
        let (depth, attributes) = try prepass(graph)
        let occlusion = try visibility(
            depth: depth, attributes: attributes,
            settings: .init(radius: 0.5, strength: 1, sliceCount: 4, stepCount: 16, falloff: 0)
        )
        func shade(_ graph: SceneGraph, occluded: Bool) throws -> [SIMD4<Float>] {
            let color = try fixture.colorTarget()
            try fixture.run { buffer in
                try graph.encode(
                    pass: fixture.pass(color: color, depth: try fixture.depthTarget()), commandBuffer: buffer,
                    screenSpaceOcclusion: occluded ? occlusion.texture : nil
                )
            }
            return try fixture.pixels(of: color)
        }
        let open = try shade(graph, occluded: false)
        let occluded = try shade(graph, occluded: true)
        fixture.writeImage(open, name: "gtao-pbr-open")
        fixture.writeImage(occluded, name: "gtao-pbr-occluded")
        // Jimenez et al. 2016's fit for light interreflected by surfaces of this albedo.
        func multipleBounce(_ v: Float, _ albedo: Float) -> Float {
            let a = 2.0404 * albedo - 0.3324
            let b = -4.7951 * albedo + 0.6417
            let c = 2.7552 * albedo + 0.6903
            return max(v, ((v * a + b) * v + c) * v)
        }
        var compared = 0
        var darkest: Float = 1
        for index in stride(from: 0, to: open.count, by: 7) where open[index].w == 1 {
            let v = occlusion.pixels[index].w
            for channel in 0..<3 {
                let expected = open[index][channel] * multipleBounce(v, albedo[channel])
                #expect(abs(occluded[index][channel] - expected) <= 2e-3 * open[index][channel] + 1e-4)
            }
            darkest = min(darkest, v)
            compared += 1
        }
        #expect(compared > 1000)
        #expect(darkest < 0.6)

        // Direct light is never occluded.
        let lit = try PhysicallyBasedMaterial(
            roughness: 0.5, directionToLight: SIMD3(0.3, 1, 0.6), lightIntensity: 2, ambientIntensity: 0,
            baseColorFactor: SIMD4(albedo, 1)
        )
        let direct = try corner(.physicallyBased(lit))
        #expect(try shade(direct, occluded: false) == shade(direct, occluded: true))
    }

    /// Image-based light that varies with azimuth in red and with elevation in green, so
    /// diffuse light depends on which way the irradiance normal points.
    private func gradientEnvironment() throws -> PhysicallyBasedMaterial.Environment {
        try fixture.environment { column, row in
            SIMD3(0.2 + 0.8 * Float(column) / 15, 0.1 + 0.9 * Float(row) / 7, 0.5)
        }
    }

    private func shade(
        _ graph: SceneGraph, occlusion: (any MTLTexture)?, size: (width: Int, height: Int)? = nil
    ) throws -> [SIMD4<Float>] {
        let target = try size.map { try RasterLightingFixture(width: $0.width, height: $0.height) } ?? fixture
        let color = try target.colorTarget()
        try target.run { buffer in
            try graph.encode(
                pass: target.pass(color: color, depth: try target.depthTarget()), commandBuffer: buffer,
                screenSpaceOcclusion: occlusion
            )
        }
        return try target.pixels(of: color)
    }

    /// The world bent normal at a floor pixel: the surface normal plus the stored bend.
    private static func floorBentNormal(_ value: SIMD4<Float>) -> SIMD3<Float> {
        simd_normalize(SIMD3(0, 1, 0) + SIMD3(value.x, value.y, value.z))
    }

    @Test("Zero strength leaves PBR shading bit-identical, with or without surface attributes")
    func zeroStrengthIsNeutral() throws {
        let occlusionMap = try fixture.constantTexture(SIMD4(128, 255, 255, 255))
        let material = try PhysicallyBasedMaterial(
            metallic: 0.25, roughness: 0.5, lightIntensity: 0, ambientIntensity: 0,
            baseColorFactor: SIMD4(0.8, 0.6, 0.4, 1), occlusionTexture: .init(texture: occlusionMap),
            environment: try gradientEnvironment()
        )
        let graph = try corner(.physicallyBased(material))
        let (depth, attributes) = try prepass(graph)
        let open = try shade(graph, occlusion: nil)
        for surfaceAttributes in [attributes, nil] as [(any MTLTexture)?] {
            let occlusion = try visibility(
                depth: depth, attributes: surfaceAttributes, settings: .init(radius: 0.5, strength: 0)
            )
            #expect(occlusion.pixels.allSatisfy { $0 == SIMD4(0, 0, 0, 1) })
            #expect(try shade(graph, occlusion: occlusion.texture) == open)
        }
    }

    @Test("Strength scales the occlusion and the bend linearly")
    func strengthScalesLinearly() throws {
        let graph = try corner(.physicallyBased(try PhysicallyBasedMaterial()))
        let (depth, attributes) = try prepass(graph)
        let full = try visibility(depth: depth, attributes: attributes, settings: .init(radius: 0.5, strength: 1)).pixels
        let half = try visibility(depth: depth, attributes: attributes, settings: .init(radius: 0.5, strength: 0.5)).pixels
        var largestBend: Float = 0
        for (whole, part) in zip(full, half) {
            // Half-float storage resolves these values to about 5e-4.
            #expect(abs(part.w - (1 - 0.5 * (1 - whole.w))) < 1e-3)
            let bend = SIMD3(whole.x, whole.y, whole.z)
            #expect(simd_reduce_max(abs(SIMD3(part.x, part.y, part.z) - 0.5 * bend)) < 1e-3)
            largestBend = max(largestBend, simd_length(bend))
        }
        #expect(largestBend > 0.2)
        #expect((full.map(\.w).min() ?? 1) < 0.6)
    }

    @Test(
        "Depth-only GTAO on an open, normal-mapped floor leaves shading unchanged to the screen border",
        arguments: [false, true]
    )
    func depthOnlyKeepsShadingNormal(denoise: Bool) throws {
        // A constant normal map tilts the shading normal 30° across the floor.
        let normalMap = try fixture.constantTexture(SIMD4(191, 128, 238, 255))
        let material = try PhysicallyBasedMaterial(
            roughness: 0.6, lightIntensity: 0, ambientIntensity: 0, normalTexture: .init(texture: normalMap),
            environment: try gradientEnvironment()
        )
        let floor = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 20, height: 20),
            material: .physicallyBased(material)
        )
        var graph = SceneGraph(
            roots: [SceneNode(transform: RasterLightingFixture.groundTransform(), mesh: floor)], camera: camera
        )
        graph.cameraTransform = RasterLightingFixture.cameraTransform(position: SIMD3(0, 2, 1), pitch: -1.1)
        let depth = try fixture.depthTarget()
        let color = try fixture.colorTarget()
        try fixture.run { buffer in
            try graph.encode(pass: fixture.pass(color: color, depth: depth), commandBuffer: buffer)
        }
        let open = try fixture.pixels(of: color)
        let occlusion = try AmbientOcclusion(device: fixture.device)
        let map = try fixture.colorTarget()
        try fixture.run { buffer in
            try occlusion.encodeVisibility(
                depth: depth, destination: map, camera: camera, cameraTransform: graph.cameraTransform,
                settings: .init(radius: 0.5, strength: 1, denoise: denoise), commandBuffer: buffer
            )
        }
        let occluded = try shade(graph, occlusion: map)
        let values = try fixture.pixels(of: map)
        #expect(open.allSatisfy { $0.w == 1 })
        // The normal map matters here: without it the floor's center is lit noticeably differently.
        let flat = try PhysicallyBasedMaterial(
            roughness: 0.6, lightIntensity: 0, ambientIntensity: 0, environment: try gradientEnvironment()
        )
        var flatGraph = SceneGraph(roots: [
            SceneNode(transform: RasterLightingFixture.groundTransform(), mesh: try MeshRenderer(
                device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 20, height: 20),
                material: .physicallyBased(flat)
            )),
        ], camera: camera)
        flatGraph.cameraTransform = graph.cameraTransform
        let center = fixture.height / 2 * fixture.width + fixture.width / 2
        let unmapped = try shade(flatGraph, occlusion: nil)[center]
        let mapEffect = (0..<3).map { abs(unmapped[$0] - open[center][$0]) / open[center][$0] }.max() ?? 0
        #expect(mapEffect > 0.1, "normal map changes the center by \(mapEffect)")

        // Nothing occludes an open plane. Depth-only normals at grazing distances carry
        // Depth32Float rounding, which leaves at most a half-precision step of occlusion and
        // a tiny bend; the shading normal is never replaced.
        let lowest = values.map(\.w).min() ?? 0
        let largestBend = values.map { simd_length(SIMD3($0.x, $0.y, $0.z)) }.max() ?? 1
        #expect(lowest >= 0.999, "lowest visibility \(lowest)")
        #expect(largestBend < 0.01, "largest bend \(largestBend)")
        var largest: Float = 0
        var unchanged = 0
        for index in open.indices {
            largest = max(largest, simd_reduce_max(abs(occluded[index] - open[index]) / open[index]))
            if occluded[index] == open[index] { unchanged += 1 }
        }
        #expect(largest < 0.01, "largest relative change \(largest)")
        #expect(unchanged > open.count * 9 / 10)
        fixture.note(
            "depth-only open floor, denoise \(denoise): normal map effect \(mapEffect), lowest visibility \(lowest), "
                + "largest bend \(largestBend), largest relative change \(largest), unchanged \(unchanged) of \(open.count)"
        )
    }

    @Test("PBR diffuse image light arrives around the bent normal, matching a surface that faces that way")
    func bentNormalIrradiance() throws {
        let albedo = SIMD3<Float>(0.7, 0.45, 0.2)
        let environment = try gradientEnvironment()
        let diffuse = try PhysicallyBasedMaterial(
            roughness: 1, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(albedo, 1),
            environment: environment, specular: try .init(factor: 0)
        )
        let graph = try corner(.physicallyBased(diffuse))
        let (depth, attributes) = try prepass(graph)
        let occlusion = try visibility(
            depth: depth, attributes: attributes,
            settings: .init(radius: 0.5, strength: 1, sliceCount: 4, stepCount: 16, falloff: 0)
        )
        let open = try shade(graph, occlusion: nil)
        let occluded = try shade(graph, occlusion: occlusion.texture)
        let plane = try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: .plane(width: 0.5, height: 0.5),
            material: .physicallyBased(diffuse)
        )
        func multipleBounce(_ v: Float, _ albedo: Float) -> Float {
            min(1, max(v, ((v * (2.0404 * albedo - 0.3324) + (-4.7951 * albedo + 0.6417)) * v + (2.7552 * albedo + 0.6903)) * v))
        }
        for distance: Float in [0.05, 0.1, 0.2] {
            let pixel = fixture.project(SIMD3(0, 0, distance), camera: camera, transform: cameraTransform)
            let index = Int(pixel.y) * fixture.width + Int(pixel.x)
            let value = occlusion.pixels[index]
            let bent = Self.floorBentNormal(value)
            // Tilted away from the wall.
            #expect(bent.z > 0.2)
            // The same material on a small plane facing the bent normal, seen at the image center.
            var reference = SceneGraph(roots: [
                SceneNode(transform: SceneTransform(rotation: simd_quatf(from: SIMD3(0, 0, 1), to: bent)), mesh: plane),
            ], camera: camera)
            reference.cameraTransform = cameraTransform
            let center = fixture.project(.zero, camera: camera, transform: cameraTransform)
            let facing = try shade(reference, occlusion: nil)[Int(center.y) * fixture.width + Int(center.x)]
            // The bent normal sees measurably different light from the floor normal.
            let change = (0..<3).map { abs(facing[$0] - open[index][$0]) / open[index][$0] }.max() ?? 0
            #expect(change > 0.05)
            for channel in 0..<3 {
                let expected = facing[channel] * multipleBounce(value.w, albedo[channel])
                #expect(abs(occluded[index][channel] - expected) < 3e-3 * facing[channel])
                fixture.note(
                    "bent irradiance z=\(distance) channel \(channel): change from floor normal \(change), "
                        + "relative error \(abs(occluded[index][channel] - expected) / facing[channel])"
                )
            }
        }
    }

    @Test("Specular occlusion is the reflection cone's overlap with the visibility cone relative to the open hemisphere's, with multiple bounce")
    func specularOcclusionMatchesCones() throws {
        let environment = try fixture.environment { _, _ in SIMD3(1, 1, 1) }
        let roughness: Float = 0.5
        let metal = try PhysicallyBasedMaterial(
            metallic: 1, roughness: roughness, lightIntensity: 0, ambientIntensity: 0,
            baseColorFactor: SIMD4(0.5, 0.5, 0.5, 1), environment: environment
        )
        let graph = try corner(.physicallyBased(metal))
        let (depth, attributes) = try prepass(graph)
        let occlusion = try visibility(
            depth: depth, attributes: attributes,
            settings: .init(radius: 0.5, strength: 1, sliceCount: 4, stepCount: 16, falloff: 0)
        )
        let open = try shade(graph, occlusion: nil)
        let occluded = try shade(graph, occlusion: occlusion.texture)
        // Oat and Sander 2007 cap intersection with a smoothstep lune, as documented.
        func capIntersection(_ cosFirst: Float, _ cosSecond: Float, _ cosDistance: Float) -> Float {
            let first = acos(min(max(cosFirst, -1), 1))
            let second = acos(min(max(cosSecond, -1), 1))
            let distance = acos(min(max(cosDistance, -1), 1))
            let smaller = 2 * Float.pi * (1 - max(cosFirst, cosSecond))
            if min(first, second) <= max(first, second) - distance { return smaller }
            if first + second <= distance { return 0 }
            let delta = abs(first - second)
            let x = 1 - min(max((distance - delta) / max(first + second - delta, 1e-4), 0), 1)
            return smaller * x * x * (3 - 2 * x)
        }
        var compared = 0
        var partial = 0
        for distance in stride(from: Float(0.02), through: 0.46, by: 0.02) {
            let pixel = fixture.project(SIMD3(0, 0, distance), camera: camera, transform: cameraTransform)
            let (x, y) = (Int(pixel.x), Int(pixel.y))
            let value = occlusion.pixels[y * fixture.width + x]
            let ray = fixture.ray(x: x, y: y, camera: camera, transform: cameraTransform)
            let reflection = SIMD3(ray.direction.x, -ray.direction.y, ray.direction.z)
            let alpha = max(roughness * roughness, 1e-3)
            let exponent = max(2 / (alpha * alpha) - 2, 0)
            let cosLobe = pow(0.25, 1 / (exponent + 1))
            let cosVisible = max(1 - value.w, 0).squareRoot()
            // The lobe's overlap with the unoccluded hemisphere around the floor normal.
            let openOverlap = capIntersection(0, cosLobe, reflection.y)
            let cone = min(max(capIntersection(cosVisible, cosLobe, simd_dot(Self.floorBentNormal(value), reflection)) / openOverlap, 0), 1)
            let v = value.w >= 1 ? 1 : cone
            // Multiple bounce with the metal's F0 of 0.5.
            let bounce = min(1, max(v, ((v * (2.0404 * 0.5 - 0.3324) + (-4.7951 * 0.5 + 0.6417)) * v + (2.7552 * 0.5 + 0.6903)) * v))
            let index = y * fixture.width + x
            #expect(abs(occluded[index].x - open[index].x * bounce) < 2e-3 * open[index].x + 1e-4)
            fixture.note(
                "specular cones z=\(distance): visibility \(value.w), expected factor \(bounce), "
                    + "measured \(occluded[index].x / open[index].x)"
            )
            if bounce > 0.05, bounce < 0.95 { partial += 1 }
            compared += 1
        }
        #expect(compared == 23)
        // Near the wall the reflection is wholly hidden, then partly, then not at all.
        #expect(partial >= 4)
    }

    @Test("Specular occlusion darkens reflections inside the crease and leaves open reflections unchanged")
    func specularOcclusion() throws {
        let width = 16
        let height = 8
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        let panorama = try #require(fixture.device.makeTexture(descriptor: descriptor))
        let texels = [SIMD4<Float16>](repeating: SIMD4(1, 1, 1, 1), count: width * height)
        try texels.withUnsafeBytes { bytes in
            panorama.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                withBytes: try #require(bytes.baseAddress), bytesPerRow: width * 8
            )
        }
        let lighting = try EnvironmentLighting(
            queue: fixture.queue, equirectangular: panorama,
            settings: .init(cubeSize: 16, diffuseSize: 8, brdfSize: 16, sampleCount: 32)
        )
        let metal = try PhysicallyBasedMaterial(
            metallic: 1, roughness: 0.3, lightIntensity: 0, ambientIntensity: 0,
            baseColorFactor: SIMD4(0.5, 0.5, 0.5, 1), environment: try .init(lighting: lighting)
        )
        let graph = try corner(.physicallyBased(metal))
        let (depth, attributes) = try prepass(graph)
        let occlusion = try visibility(
            depth: depth, attributes: attributes,
            settings: .init(radius: 0.5, strength: 1, sliceCount: 4, stepCount: 16, falloff: 0)
        )
        func shade(_ occluded: Bool) throws -> [SIMD4<Float>] {
            let color = try fixture.colorTarget()
            try fixture.run { buffer in
                try graph.encode(
                    pass: fixture.pass(color: color, depth: try fixture.depthTarget()), commandBuffer: buffer,
                    screenSpaceOcclusion: occluded ? occlusion.texture : nil
                )
            }
            return try fixture.pixels(of: color)
        }
        let open = try shade(false)
        let occluded = try shade(true)
        let crease = fixture.project(SIMD3(0, 0, 0.03), camera: camera, transform: cameraTransform)
        let creaseIndex = Int(crease.y) * fixture.width + Int(crease.x)
        let far = fixture.project(SIMD3(0, 0, 1.2), camera: camera, transform: cameraTransform)
        let farIndex = Int(far.y) * fixture.width + Int(far.x)
        #expect(occluded[creaseIndex].x < 0.9 * open[creaseIndex].x)
        #expect(occluded[farIndex] == open[farIndex])
        for index in open.indices where open[index].w == 1 {
            #expect(occluded[index].x <= open[index].x * 1.001 + 1e-4)
        }
    }
}
