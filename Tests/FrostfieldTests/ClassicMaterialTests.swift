import Foundation
import Frostfield
import Metal
import Testing
import simd

/// Three.js basic, Lambert, Phong, toon, matcap, normal, depth, distance and shadow
/// materials against their closed-form outputs.
struct ClassicMaterialTests {
    let fixture: GeometryRenderFixture

    init() throws {
        fixture = try GeometryRenderFixture()
    }

    func mesh(_ material: ClassicMaterial, state: MeshRenderState? = nil, halfExtent: Float = 1) throws -> MeshRenderer {
        try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: try GeometryRenderFixture.square(halfExtent),
            material: .classic(material), renderState: state
        )
    }

    /// A directional light of π lux whose direction toward the light leans `angle`
    /// radians from +Z toward +X, so N·L = cos(angle) on a quad facing the camera.
    func light(angle: Float, intensity: Float = .pi) throws -> SceneNode {
        SceneNode(
            transform: SceneTransform(rotation: simd_quatf(angle: angle, axis: SIMD3(0, 1, 0))),
            light: try PunctualLight(kind: .directional, intensity: intensity)
        )
    }

    func pixels(
        _ material: ClassicMaterial, rotation: simd_quatf? = nil, lights: [SceneNode] = [], state: MeshRenderState? = nil,
        halfExtent: Float = 1
    ) throws -> [SIMD4<Float>] {
        let quad = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -2), rotation: rotation ?? simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))),
            mesh: try mesh(material, state: state, halfExtent: halfExtent)
        )
        return try fixture.render(SceneGraph(roots: [quad] + lights, camera: fixture.camera)).pixels
    }

    func center(
        _ material: ClassicMaterial, rotation: simd_quatf? = nil, lights: [SceneNode] = [], state: MeshRenderState? = nil
    ) throws -> SIMD4<Float> {
        fixture.pixel(try pixels(material, rotation: rotation, lights: lights, state: state), x: 32, y: 32)
    }

    /// The world point a pixel center sees on the plane z = −2 for the fixture camera.
    func worldPoint(x: Int, y: Int) -> SIMD3<Float> {
        let side = Float(fixture.side)
        return SIMD3(((Float(x) + 0.5) / side * 2 - 1) * 2, (1 - (Float(y) + 0.5) / side * 2) * 2, -2)
    }

    func texture(
        width: Int, height: Int, format: MTLPixelFormat = .rgba8Unorm, bytes: [UInt8]
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        let texture = try #require(fixture.device.makeTexture(descriptor: descriptor))
        texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: bytes, bytesPerRow: bytes.count / height)
        return texture
    }

    func nearest() throws -> any MTLSamplerState {
        let descriptor = MTLSamplerDescriptor()
        descriptor.minFilter = .nearest
        descriptor.magFilter = .nearest
        descriptor.sAddressMode = .clampToEdge
        descriptor.tAddressMode = .clampToEdge
        return try #require(fixture.device.makeSamplerState(descriptor: descriptor))
    }

    // MARK: Toon

    @Test("Toon gradient bands switch at the N·L thresholds of a three-texel gradient map")
    func toonBands() throws {
        let gradient = try texture(width: 3, height: 1, bytes: [51, 0, 0, 255, 128, 0, 0, 255, 255, 0, 0, 255])
        let binding = ClassicMaterial.TextureBinding(texture: gradient, sampler: try nearest())
        let color = SIMD3<Float>(0.8, 0.6, 0.4)
        let material = try ClassicMaterial(model: .toon(gradient: binding), color: color)
        // Texel i covers N·L × 0.5 + 0.5 in [i/3, (i + 1)/3): thresholds at N·L = ±1/3.
        let cases: [(cosine: Float, band: Float)] = [
            (0.9, 1), (0.36, 1), (0.30, 128 / 255), (0, 128 / 255), (-0.30, 128 / 255), (-0.36, 0.2), (-0.9, 0.2),
        ]
        for (cosine, band) in cases {
            let pixel = try center(material, lights: [try light(angle: acos(cosine))])
            #expect(isNearly(pixel, SIMD4(color * band, 1)), "N·L \(cosine): \(pixel)")
        }
        // Without a map, Three.js uses 0.7 below N·L = 0.4 and 1 above.
        let plain = try ClassicMaterial(model: .toon(), color: color)
        #expect(isNearly(try center(plain, lights: [try light(angle: acos(0.6))]), SIMD4(color, 1)))
        #expect(isNearly(try center(plain, lights: [try light(angle: acos(0.2))]), SIMD4(color * 0.7, 1)))
    }

    // MARK: Lambert and Phong

    @Test("Lambert and Phong match Three.js direct, ambient and Blinn-Phong terms")
    func lambertAndPhong() throws {
        let color = SIMD3<Float>(0.8, 0.6, 0.4)
        let lambert = try ClassicMaterial(model: .lambert, color: color, ambient: SIMD3(repeating: 0.2 * .pi))
        // E = π at N·L = 0.5 gives 0.5; ambient 0.2π gives 0.2.
        #expect(isNearly(try center(lambert, lights: [try light(angle: .pi / 3)]), SIMD4(color * 0.7, 1)))
        // Facing away, the back face of a double-sided quad shades with its flipped normal.
        let backLit = try center(
            try ClassicMaterial(model: .lambert, color: color), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0)),
            lights: [try light(angle: 0)], state: MeshRenderState(side: .double)
        )
        #expect(isNearly(backLit, SIMD4(color, 1)))

        let specular = SIMD3<Float>(0.2, 0.3, 0.4)
        let shininess: Float = 30
        let phong = try ClassicMaterial(model: .phong(specular: specular, shininess: shininess), color: color)
        let angle = Float.pi / 6
        let image = try pixels(phong, lights: [try light(angle: angle)])
        for (x, y) in [(32, 32), (20, 40), (45, 25)] {
            let n = SIMD3<Double>(0, 0, 1)
            let l = SIMD3<Double>(Double(sin(angle)), 0, Double(cos(angle)))
            let v = simd_normalize(-SIMD3<Double>(worldPoint(x: x, y: y)))
            let h = simd_normalize(l + v)
            let noL = simd_dot(n, l)
            let fresnel = SIMD3<Double>(specular) + (1 - SIMD3<Double>(specular)) * pow(1 - simd_dot(v, h), 5)
            let distribution = (Double(shininess) / 2 + 1) / .pi * pow(simd_dot(n, h), Double(shininess))
            let expected = SIMD3<Double>(color) * noL + .pi * noL * fresnel * 0.25 * distribution
            #expect(isNearly(fixture.pixel(image, x: x, y: y), SIMD4(SIMD3<Float>(expected), 1)), "\(x), \(y)")
        }
    }

    @Test("Lambert and Phong node graphs replace the diffuse color before lighting")
    func nodeGraphColor() throws {
        let color = SIMD3<Float>(0.2, 0.4, 0.6)
        let graph = try SurfaceShaderGraph { s in
            .init(baseColor: s.uniform("Tint", color))
        }
        let uniforms = try ShaderUniforms(graph: graph)
        let lights = [try light(angle: .pi / 3)]
        let models: [ClassicMaterial.Model] = [
            .lambert, .phong(specular: SIMD3(0.2, 0.3, 0.4), shininess: 30),
        ]
        for model in models {
            let node = try ClassicMaterial(
                model: model, color: SIMD3(repeating: 1), surfaceGraph: graph, surfaceUniforms: uniforms
            )
            let reference = try ClassicMaterial(model: model, color: color)
            #expect(isNearly(try center(node, lights: lights), try center(reference, lights: lights)), "\(model)")
        }
        let scene = SceneGraph(
            roots: [
                SceneNode(
                    transform: SceneTransform(translation: SIMD3(0, 0, -2)),
                    mesh: try mesh(try ClassicMaterial(
                        model: .lambert, color: SIMD3(repeating: 1), surfaceGraph: graph, surfaceUniforms: uniforms
                    ))
                ),
            ] + lights,
            camera: fixture.camera
        )
        let archive = try SceneArchive(capturing: scene, commandQueue: fixture.queue)
        let loaded = try archive.load(device: fixture.device, commandQueue: fixture.queue).graph
        let restored = try fixture.render(loaded).pixels
        let original = try fixture.render(scene).pixels
        #expect(restored == original)
        try uniforms.set("Tint", to: SIMD3(repeating: 0.5))
        let edited = try ClassicMaterial(model: .lambert, surfaceGraph: graph, surfaceUniforms: uniforms)
        let gray = try ClassicMaterial(model: .lambert, color: SIMD3(repeating: 0.5))
        #expect(isNearly(try center(edited, lights: lights), try center(gray, lights: lights)))
        #expect(throws: ClassicMaterial.Failure.nodeGraph("A classic node material reads base color, opacity, and normal.")) {
            try ClassicMaterial(
                model: .lambert,
                surfaceGraph: try SurfaceShaderGraph { s in
                    .init(baseColor: s.constant(SIMD3(1, 0, 0)), emission: s.constant(SIMD3(1, 0, 0)))
                }
            )
        }
    }

    @Test("A classic node graph scales opacity before blending and alpha test")
    func nodeGraphOpacity() throws {
        let color = SIMD3<Float>(0.8, 0.6, 0.4)
        let graph = try SurfaceShaderGraph { s in
            .init(baseColor: s.constant(color), opacity: s.uniform("Opacity", Float(0.25)))
        }
        let uniforms = try ShaderUniforms(graph: graph)
        let node = try ClassicMaterial(model: .basic, surfaceGraph: graph, surfaceUniforms: uniforms)
        let blended = try ClassicMaterial(model: .basic, color: color, opacity: 0.25)
        #expect(isNearly(try center(node, state: .transparent), try center(blended, state: .transparent)))
        let tested = try ClassicMaterial(model: .basic, alphaTest: 0.5, surfaceGraph: graph, surfaceUniforms: uniforms)
        #expect(isNearly(try center(tested), .zero))
        try uniforms.set("Opacity", to: 0.8)
        let visible = try ClassicMaterial(model: .basic, color: color, opacity: 0.8, alphaTest: 0.5)
        #expect(isNearly(try center(tested), try center(visible)))
    }

    @Test("A classic node graph replaces the shading normal in the UV0 tangent frame")
    func nodeGraphNormal() throws {
        let graph = try SurfaceShaderGraph { s in
            .init(baseColor: s.constant(SIMD3(repeating: 1)), normal: s.uniform("Normal", SIMD3(0, 0, 1)))
        }
        let uniforms = try ShaderUniforms(graph: graph)
        let node = try ClassicMaterial(model: .lambert, surfaceGraph: graph, surfaceUniforms: uniforms)
        let lights = [try light(angle: 0)]
        #expect(isNearly(try center(node, lights: lights), SIMD4(1, 1, 1, 1)))
        try uniforms.set("Normal", to: SIMD3(1, 0, 0))
        #expect(isNearly(try center(node, lights: lights), SIMD4(0, 0, 0, 1), tolerance: 0.02))
    }

    @Test("Classic lighting reads a spot cookie through the shared light tables")
    func spotCookie() throws {
        let material = try ClassicMaterial(model: .lambert, color: SIMD3(repeating: 1))
        let kind = PunctualLight.Kind.spot(innerConeAngle: 0.2, outerConeAngle: 0.5)
        let plain = SceneNode(light: try PunctualLight(kind: kind, intensity: 4 * .pi))
        let cookie = try LightCookie(width: 1, height: 1, transmittance: [SIMD3(0.25, 0.5, 1)])
        let profiled = SceneNode(light: try PunctualLight(kind: kind, intensity: 4 * .pi, cookie: cookie))
        let reference = try center(material, lights: [plain])
        #expect(reference.x > 0.99)
        let actual = try center(material, lights: [profiled])
        #expect(isNearly(actual, reference * SIMD4(0.25, 0.5, 1, 1), tolerance: 0.002))
    }

    // MARK: Matcap

    @Test("Matcap samples the texel that the view-space normal points at")
    func matcap() throws {
        // Each texel's red encodes its column and green its row.
        var bytes: [UInt8] = []
        for row in 0..<3 { for column in 0..<3 { bytes += [UInt8(50 + 100 * column), UInt8(50 + 100 * row), 128, 255] } }
        let lookup = try texture(width: 3, height: 3, bytes: bytes)
        let material = try ClassicMaterial(model: .matcap(.init(texture: lookup, sampler: try nearest())))
        func texel(_ column: Int, _ row: Int) -> SIMD4<Float> {
            SIMD4(Float(50 + 100 * column) / 255, Float(50 + 100 * row) / 255, 128 / 255, 1)
        }
        let tilt = Float.pi * 40 / 180
        #expect(isNearly(try center(material), texel(1, 1)))
        // Turning the normal toward +X reads the right column; toward +Y the top row.
        #expect(isNearly(try center(material, rotation: simd_quatf(angle: tilt, axis: SIMD3(0, 1, 0))), texel(2, 1)))
        #expect(isNearly(try center(material, rotation: simd_quatf(angle: -tilt, axis: SIMD3(1, 0, 0))), texel(1, 0)))
        #expect(isNearly(try center(material, rotation: simd_quatf(angle: tilt, axis: SIMD3(1, 0, 0))), texel(1, 2)))
        #expect(isNearly(try center(material, rotation: simd_quatf(angle: -tilt, axis: SIMD3(0, 1, 0))), texel(0, 1)))
    }

    @Test("Off-center matcap lookups follow Three.js's per-fragment view direction")
    func matcapViewDirection() throws {
        // A 256 × 256 lookup whose red and green store each texel's center U and V.
        let size = 256
        var texels: [Float16] = []
        for row in 0..<size {
            for column in 0..<size {
                texels += [Float16((Float(column) + 0.5) / Float(size)), Float16((Float(row) + 0.5) / Float(size)), 0, 1]
            }
        }
        let bytes = texels.withUnsafeBytes { Array($0) }
        let lookup = try texture(width: size, height: size, format: .rgba16Float, bytes: bytes)
        let material = try ClassicMaterial(model: .matcap(.init(texture: lookup, sampler: try nearest())))
        let image = try pixels(material, halfExtent: 3)
        for (x, y) in [(6, 52), (58, 9), (48, 40), (32, 32)] {
            // Three.js meshmatcap: viewDir = normalize(vViewPosition) toward the camera,
            // x = normalize(viewDir.z, 0, −viewDir.x), y = viewDir × x, uv = (x·n, y·n) × 0.495 + 0.5,
            // with V flipped for the top-left texture. The camera is at the origin looking down −Z.
            let viewDirection = simd_normalize(-SIMD3<Double>(worldPoint(x: x, y: y)))
            let n = SIMD3<Double>(0, 0, 1)
            let xAxis = simd_normalize(SIMD3(viewDirection.z, 0, -viewDirection.x))
            let yAxis = simd_cross(viewDirection, xAxis)
            let u = 0.5 + 0.495 * simd_dot(xAxis, n)
            let v = 0.5 - 0.495 * simd_dot(yAxis, n)
            let pixel = fixture.pixel(image, x: x, y: y)
            // The nearest texel's center lies within half a texel of (u, v).
            let tolerance = 0.5 / Double(size) + 0.001
            #expect(abs(Double(pixel.x) - u) <= tolerance && abs(Double(pixel.y) - v) <= tolerance,
                    "pixel \(x), \(y): \(pixel) vs \(u), \(v)")
            if (x, y) != (32, 32) {
                #expect(abs(u - 0.5) > 0.05 || abs(v - 0.5) > 0.05, "pixel \(x), \(y) is far enough off center")
            }
        }
    }

    // MARK: Normal, depth and distance

    @Test("Normal, depth and distance materials write their packed closed-form values")
    func normalDepthDistance() throws {
        let normal = try ClassicMaterial(model: .normal)
        #expect(isNearly(try center(normal), SIMD4(0.5, 0.5, 1, 1)))
        let tilt = Float.pi * 40 / 180
        #expect(isNearly(
            try center(normal, rotation: simd_quatf(angle: tilt, axis: SIMD3(0, 1, 0))),
            SIMD4(0.5 + 0.5 * sin(tilt), 0.5, 0.5 + 0.5 * cos(tilt), 1)
        ))
        // The camera turned toward −X sees the same quad with its normal rotated into view space.
        let turned = SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(-2, 0, 0), rotation: simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0))),
                              mesh: try mesh(normal))],
            camera: fixture.camera, cameraTransform: SceneTransform(rotation: simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0)))
        )
        #expect(isNearly(fixture.pixel(try fixture.render(turned).pixels, x: 32, y: 32), SIMD4(0.5, 0.5, 1, 1)))
        // A caller-supplied view matrix yields the same view axes, from its projection rows.
        let target = try fixture.makeTarget()
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        let clip = try fixture.camera.projection(aspectRatio: 1) * simd_inverse(turned.cameraTransform.matrix)
        try turned.encode(
            pass: fixture.pass(color: target.color, depth: target.depth), commandBuffer: buffer,
            clipFromWorld: clip, cameraPosition: .zero
        )
        try fixture.finish(buffer)
        #expect(isNearly(fixture.pixel(try fixture.pixels(of: target.color), x: 32, y: 32), SIMD4(0.5, 0.5, 1, 1)))
        // A quad tilted 40° about the world Y axis reads the same view-space X under that view.
        let tiltedGraph = SceneGraph(
            roots: [SceneNode(transform: SceneTransform(translation: SIMD3(0, 0, -2), rotation: simd_quatf(angle: tilt, axis: SIMD3(0, 1, 0))),
                              mesh: try mesh(normal))],
            camera: fixture.camera
        )
        let tiltedTarget = try fixture.makeTarget()
        let tiltedBuffer = try #require(fixture.queue.makeCommandBuffer())
        try tiltedGraph.encode(
            pass: fixture.pass(color: tiltedTarget.color, depth: tiltedTarget.depth), commandBuffer: tiltedBuffer,
            clipFromWorld: try fixture.camera.projection(aspectRatio: 1), cameraPosition: .zero
        )
        try fixture.finish(tiltedBuffer)
        #expect(isNearly(
            fixture.pixel(try fixture.pixels(of: tiltedTarget.color), x: 32, y: 32), SIMD4(0.5 + 0.5 * sin(tilt), 0.5, 0.5 + 0.5 * cos(tilt), 1)
        ))

        let near: Float = 0.1
        let far: Float = 50
        let depth = far / (far - near) * (1 - near / 2)
        #expect(isNearly(try center(try ClassicMaterial(model: .depth)), SIMD4(1 - depth, 1 - depth, 1 - depth, 1)))

        let distance = try ClassicMaterial(model: .distance(reference: .zero, near: 1, far: 5))
        let image = try pixels(distance, halfExtent: 3)
        for (x, y) in [(32, 32), (10, 50), (60, 5), (0, 0)] {
            let value = min(max((simd_length(worldPoint(x: x, y: y)) - 1) / 4, 0), 1)
            #expect(isNearly(fixture.pixel(image, x: x, y: y), SIMD4(value, value, value, 1), tolerance: 0.002), "\(x), \(y)")
        }
    }

    // MARK: Basic maps

    @Test("Basic material applies light map irradiance over π, occlusion, and environment blends")
    func basicMaps() throws {
        let color = SIMD3<Float>(0.8, 0.6, 0.4)
        let irradiance: [Float16] = [0.9, 1.8, 2.7, 1]
        let lightMap = try texture(width: 1, height: 1, format: .rgba16Float, bytes: irradiance.withUnsafeBytes { Array($0) })
        let occlusion = try texture(width: 1, height: 1, bytes: [128, 0, 0, 255])
        let material = try ClassicMaterial(
            model: .basic, color: color, aoMap: .init(texture: occlusion), aoMapIntensity: 0.5,
            lightMap: .init(texture: lightMap), lightMapIntensity: 2
        )
        let ao = (Float(128) / 255 - 1) * 0.5 + 1
        let baked = SIMD3(Float(irradiance[0]), Float(irradiance[1]), Float(irradiance[2])) * 2 / .pi
        #expect(isNearly(try center(material), SIMD4(color * baked * ao, 1)))

        // One color per cube face: +X, −X, +Y, −Y, +Z, −Z.
        let faces: [SIMD3<Float>] = [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1), SIMD3(1, 1, 0), SIMD3(0, 1, 1), SIMD3(1, 0, 1)]
        // Eight texels per side keep the face-center lookups clear of seamless edge filtering.
        let descriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: .rgba8Unorm, size: 8, mipmapped: false)
        descriptor.usage = .shaderRead
        let cube = try #require(fixture.device.makeTexture(descriptor: descriptor))
        for (face, value) in faces.enumerated() {
            let texel: [UInt8] = [UInt8(value.x * 255), UInt8(value.y * 255), UInt8(value.z * 255), 255]
            let texels: [UInt8] = Array(repeating: texel, count: 64).flatMap { $0 }
            cube.replace(region: MTLRegionMake2D(0, 0, 8, 8), mipmapLevel: 0, slice: face, withBytes: texels, bytesPerRow: 32, bytesPerImage: 0)
        }
        // Facing the camera, the reflection is +Z.
        let mixed = try ClassicMaterial(model: .basic, color: color, environmentMap: .init(texture: cube, blend: .mix))
        let mixedPixel = try center(mixed)
        #expect(isNearly(mixedPixel, SIMD4(faces[4], 1)), "\(mixedPixel)")
        let multiplied = try ClassicMaterial(
            model: .basic, color: color, environmentMap: .init(texture: cube, blend: .multiply, reflectivity: 0.5)
        )
        #expect(isNearly(try center(multiplied), SIMD4(simd_mix(color, color * faces[4], SIMD3(repeating: 0.5)), 1)))
        let added = try ClassicMaterial(
            model: .basic, color: color, environmentMap: .init(texture: cube, blend: .add, reflectivity: 0.25)
        )
        #expect(isNearly(try center(added), SIMD4(color + faces[4] * 0.25, 1)))
        // A quarter-turn yaw maps the +Z reflection to the −X face, as EnvironmentLighting does.
        let turned = try ClassicMaterial(model: .basic, color: color, environmentMap: .init(texture: cube, blend: .mix, yaw: .pi / 2))
        #expect(isNearly(try center(turned), SIMD4(faces[1], 1)))
        #expect(throws: ClassicMaterial.Failure.environmentTexture) {
            try ClassicMaterial.EnvironmentMap(texture: occlusion)
        }
    }

    // MARK: Alpha

    @Test("Opacity, alpha maps and alpha test follow Three.js transparent and opaque output rules")
    func alpha() throws {
        let color = SIMD3<Float>(0.8, 0.6, 0.4)
        let alphaMap = try texture(width: 1, height: 1, bytes: [0, 64, 0, 255])
        let material = try ClassicMaterial(model: .basic, color: color, opacity: 0.8, alphaMap: .init(texture: alphaMap))
        let alpha = 0.8 * Float(64) / 255
        #expect(isNearly(try center(material, state: .transparent), SIMD4(color * alpha, alpha)))
        // Without blending the surface is opaque, as with Three.js `transparent = false`.
        #expect(isNearly(try center(material), SIMD4(color, 1)))
        let tested = try ClassicMaterial(model: .basic, color: color, opacity: 0.8, alphaMap: .init(texture: alphaMap), alphaTest: 0.5)
        #expect(isNearly(try center(tested), .zero))
    }

    // MARK: Shadow

    @Test("The shadow catcher is black by default and clears away without a shadow map")
    func shadowCatcherDefaults() throws {
        let material = try ClassicMaterial.shadowCatcher()
        #expect(material.color == .zero && material.opacity == 1)
        #expect(try center(material) == .zero)
        #expect(throws: ClassicMaterial.Failure.invalidValue("opacity")) {
            try ClassicMaterial.shadowCatcher(opacity: -0.1)
        }
    }

    enum CatcherLight: CaseIterable, Sendable { case directional, spot, point }

    @Test("Shadow catchers blend tinted directional and punctual shadows over a background with explicit depth policy",
          arguments: CatcherLight.allCases, [ShadowMapSampling.basic, .filtered])
    func shadowCatcherCompositing(_ light: CatcherLight, sampling: ShadowMapSampling) async throws {
        let tint = SIMD3<Float>(0.125, 0.25, 0.375)
        let backgroundColor = SIMD3<Float>(0.625, 0.375, 0.875)
        let material = try ClassicMaterial.shadowCatcher(
            color: tint, opacity: 0.5, receivesDirectionalShadows: light == .directional
        )
        let receiverTransform = SceneTransform(translation: SIMD3(0, 0, -3))
        let background = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -5)),
            mesh: try mesh(ClassicMaterial(model: .basic, color: backgroundColor), halfExtent: 6),
            castsShadow: false
        )
        let directional: DirectionalShadowMap?
        let punctual: PunctualShadowMaps?
        let lightNode: SceneNode?
        let blockerDepth: Float
        if light == .directional {
            // The light is behind the catcher and emits no direct radiance. Its
            // assigned shadow depth still contributes to the receiver's mask.
            directional = try DirectionalShadowMap(
                queue: fixture.queue, resolution: 128, directionToLight: SIMD3(0.6, 0, -0.8),
                center: SIMD3(0, 0, -3.5), halfExtent: 3, intensity: 0, sampling: sampling
            )
            punctual = nil
            lightNode = nil
            blockerDepth = -4
        } else {
            directional = nil
            let spot = light == .spot
            punctual = try PunctualShadowMaps(
                queue: fixture.queue, resolution: 128, spotCapacity: spot ? 1 : 0, pointCapacity: spot ? 0 : 1
            )
            let position = SIMD3<Float>(2.25, 0, 0)
            let rotation = simd_quatf(from: SIMD3(0, 0, -1), to: simd_normalize(receiverTransform.translation - position))
            let shadow = try PunctualLight.Shadow(normalBias: 0, filterRadius: 1, sampling: sampling)
            let source = try PunctualLight(
                kind: spot ? .spot(innerConeAngle: 0.45, outerConeAngle: 0.6) : .point(), intensity: 8, shadow: shadow
            )
            lightNode = SceneNode(transform: .init(translation: position, rotation: rotation), light: source)
            blockerDepth = -2
        }
        let blocker = SceneNode(
            transform: .init(translation: SIMD3(0.75, 0, blockerDepth)),
            mesh: try mesh(
                ClassicMaterial(model: .basic), state: MeshRenderState(depthWrite: false, colorWriteMask: [], side: .double),
                halfExtent: 0.32
            )
        )
        let projection = try fixture.camera.projection(aspectRatio: 1)
        for depthWrite in [false, true] {
            let state: MeshRenderState? = depthWrite ? MeshRenderState(blending: .normal, transparent: true, depthWrite: true) : nil
            let receiver = SceneNode(
                transform: receiverTransform, mesh: try mesh(material, state: state, halfExtent: 1.65), castsShadow: false
            )
            let graph = SceneGraph(roots: [background, blocker, receiver] + [lightNode].compactMap { $0 }, camera: fixture.camera)
            let target = try fixture.makeTarget()
            let pass = fixture.pass(color: target.color, depth: target.depth)
            pass.depthAttachment.storeAction = .store
            let command = try #require(fixture.queue.makeCommandBuffer())
            let stats = try graph.encode(pass: pass, commandBuffer: command, shadowMap: directional, punctualShadowMaps: punctual)
            try fixture.finish(command)
            #expect(stats.shadowPassCount == (light == .point ? 6 : 1) && stats.shadowDrawCount > 0)
            let pixels = try fixture.pixels(of: target.color)
            let center = fixture.side / 2
            let clear = fixture.column(x: -1.25, depth: 3)
            let actual = fixture.pixel(pixels, x: center, y: center)
            #expect(isNearly(actual, SIMD4((backgroundColor + tint) * 0.5, 1)), "\(light) \(sampling): \(actual)")
            #expect(isNearly(fixture.pixel(pixels, x: clear, y: center), SIMD4(backgroundColor, 1)))

            let row = try await TextureTransfer.read(
                from: target.depth, region: MTLRegionMake2D(0, center, fixture.side, 1), commandQueue: fixture.queue
            )
            let clip = projection * SIMD4<Float>(0, 0, depthWrite ? -3 : -5, 1)
            let expectedDepth = clip.z / clip.w
            for x in [center, clear] {
                let actualDepth = row.bytes.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: x * 4, as: Float.self) }
                #expect(abs(actualDepth - expectedDepth) < 0.00001)
            }
        }
    }

    @Test("Shadow material shows only the shadowed region as translucent color")
    func shadowMaterial() throws {
        let receiver = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            mesh: try mesh(try ClassicMaterial(model: .shadow, color: SIMD3(0.1, 0.2, 0.3), opacity: 0.5, receivesShadows: true), halfExtent: 1.6),
            castsShadow: false
        )
        // Behind the camera, this blocker shades the receiver's center along (0.6, 0, 0.8);
        // it writes depth only, so the color pass never shows it.
        let blocker = SceneNode(
            transform: SceneTransform(translation: SIMD3(2.625, 0, 0.5)),
            mesh: try mesh(try ClassicMaterial(model: .basic), state: MeshRenderState(colorWriteMask: []), halfExtent: 0.25)
        )
        let shadowMap = try DirectionalShadowMap(
            queue: fixture.queue, resolution: 512, directionToLight: SIMD3(0.6, 0, 0.8), center: SIMD3(0, 0, -3), halfExtent: 5
        )
        let image = try fixture.render(SceneGraph(roots: [receiver, blocker], camera: fixture.camera), shadowMap: shadowMap).pixels
        #expect(isNearly(fixture.pixel(image, x: 32, y: 32), SIMD4(0.05, 0.1, 0.15, 0.5)))
        #expect(isNearly(fixture.pixel(image, x: fixture.column(x: -1, depth: 3), y: 32), .zero))

        // A lit model that receives the same map keeps only its ambient term in shadow.
        let lit = SceneNode(
            transform: SceneTransform(translation: SIMD3(0, 0, -3)),
            mesh: try mesh(try ClassicMaterial(model: .lambert, ambient: SIMD3(repeating: 0.1 * .pi), receivesShadows: true), halfExtent: 1.6),
            castsShadow: false
        )
        let litImage = try fixture.render(SceneGraph(roots: [lit, blocker], camera: fixture.camera), shadowMap: shadowMap).pixels
        #expect(isNearly(fixture.pixel(litImage, x: 32, y: 32), SIMD4(0.1, 0.1, 0.1, 1)))
        // Unshadowed, the map's unit light adds N·L / π = 0.8 / π.
        let level = 0.1 + 0.8 / Float.pi
        #expect(isNearly(fixture.pixel(litImage, x: fixture.column(x: -1, depth: 3), y: 32), SIMD4(level, level, level, 1)))
        #expect(throws: MeshRenderer.Failure.shadowMap) {
            try fixture.render(SceneGraph(roots: [receiver], camera: fixture.camera))
        }
    }

    @Test("Classic materials reject values outside their ranges and maps their model ignores")
    func validation() throws {
        let white = try texture(width: 1, height: 1, bytes: [255, 255, 255, 255])
        #expect(throws: ClassicMaterial.Failure.invalidValue("opacity")) { try ClassicMaterial(model: .basic, opacity: 2) }
        #expect(throws: ClassicMaterial.Failure.unusedProperty("emissive")) {
            try ClassicMaterial(model: .basic, emissive: SIMD3(1, 0, 0))
        }
        let descriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: .rgba8Unorm, size: 1, mipmapped: false)
        descriptor.usage = .shaderRead
        let cube = try #require(fixture.device.makeTexture(descriptor: descriptor))
        #expect(throws: ClassicMaterial.Failure.unusedProperty("environmentMap")) {
            try ClassicMaterial(model: .toon(), environmentMap: try .init(texture: cube))
        }
        #expect(throws: ClassicMaterial.Failure.unusedProperty("aoMap or lightMap")) {
            try ClassicMaterial(model: .normal, aoMap: .init(texture: white))
        }
        #expect(throws: ClassicMaterial.Failure.invalidValue("distance reference or range")) {
            try ClassicMaterial(model: .distance(reference: .zero, near: 2, far: 1))
        }
        #expect(throws: ClassicMaterial.Failure.unusedProperty("receivesShadows")) {
            try ClassicMaterial(model: .depth, receivesShadows: true)
        }

        // The alpha map reads green, so single-channel formats are rejected for it but not for occlusion.
        let red = try texture(width: 1, height: 1, format: .r8Unorm, bytes: [255])
        #expect(throws: ClassicMaterial.Failure.texture(1)) {
            try mesh(try ClassicMaterial(model: .basic, alphaMap: .init(texture: red)))
        }
        _ = try mesh(try ClassicMaterial(model: .basic, aoMap: .init(texture: red)))
        let redGreen = try texture(width: 1, height: 1, format: .rg8Unorm, bytes: [0, 255])
        _ = try mesh(try ClassicMaterial(model: .basic, alphaMap: .init(texture: redGreen)))
        // A map on the secondary UV set needs secondary coordinates on every vertex.
        #expect(throws: MeshRenderer.Failure.missingSecondaryTextureCoordinates) {
            try mesh(try ClassicMaterial(model: .basic, lightMap: .init(texture: white, textureCoordinateSet: .secondary)))
        }
        #expect(throws: MeshRenderer.Failure.missingSecondaryTextureCoordinates) {
            try mesh(try ClassicMaterial(model: .lambert, aoMap: .init(texture: white, textureCoordinateSet: .secondary)))
        }
    }

    @Test("Phong's default specular is Three.js 0x111111 decoded from sRGB")
    func phongDefaults() throws {
        let encoded = 17.0 / 255
        let linear = Float(pow((encoded + 0.055) / 1.055, 2.4))
        guard case .phong(let specular, let shininess) = try ClassicMaterial(model: .phong()).model else {
            Issue.record("Expected a Phong model")
            return
        }
        #expect(simd_reduce_max(abs(specular - SIMD3(repeating: linear))) < 1e-6, "\(specular) vs \(linear)")
        #expect(shininess == 30)
        #expect(ClassicMaterial.defaultPhongSpecular == specular)
    }
}
