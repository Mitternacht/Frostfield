import Foundation
import Frostfield
import Metal
import Testing
import simd

struct TextureTransformValidationTests {
    @Test("Texture transforms scale, rotate, then offset, and reject degenerate values")
    func formula() throws {
        typealias Transform = PhysicallyBasedMaterial.TextureTransform
        let transform = try Transform(offset: SIMD2(0.25, -0.5), rotation: .pi / 2, scale: SIMD2(2, 3))
        // u' = cos(r) sx u + sin(r) sy v + ox, v' = -sin(r) sx u + cos(r) sy v + oy.
        let mapped = transform.apply(to: SIMD2(0.5, 0.25))
        #expect(simd_distance(mapped, SIMD2(0.75 + 0.25, -1 - 0.5)) < 1e-6)
        #expect(Transform.identity.apply(to: SIMD2(0.3, 0.7)) == SIMD2(0.3, 0.7))
        #expect(Transform.identity == (try Transform()))
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidTextureTransform) { try Transform(scale: SIMD2(0, 1)) }
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidTextureTransform) { try Transform(offset: SIMD2(.nan, 0)) }
        #expect(throws: PhysicallyBasedMaterial.Failure.invalidTextureTransform) { try Transform(rotation: .infinity) }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct TextureTransformTests {
    private let fixture: MaterialRenderFixture
    private let camera: PerspectiveCamera
    private let side = MaterialRenderFixture.side
    // A 2 × 2 checker of 8 × 8 texel blocks: red, green on top; blue, white below.
    private let colors: [SIMD3<Float>] = [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1), SIMD3(1, 1, 1)]

    init() throws {
        fixture = try MaterialRenderFixture()
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
    }

    private func checker() throws -> any MTLTexture {
        var pixels: [UInt8] = []
        for y in 0..<16 {
            for x in 0..<16 {
                let color = colors[(y / 8) * 2 + x / 8]
                pixels += [UInt8(color.x * 255), UInt8(color.y * 255), UInt8(color.z * 255), 255]
            }
        }
        return try fixture.dataTexture(width: 16, height: 16, pixels: pixels)
    }

    @Test("A transform moves checker texels to the expected pixels in both renderers")
    func checkerMoves() throws {
        // A 2 × 2 plane at distance 1 fills a 90° view, so pixel centers sample
        // UV ((x + 0.5) / 33, (y + 0.5) / 33).
        let plane = try MeshGeometry.plane(width: 2, height: 2)
        let placement = SceneTransform(translation: SIMD3(0, 0, -1))
        let texture = try checker()
        let transforms = [
            PhysicallyBasedMaterial.TextureTransform.identity,
            try .init(offset: SIMD2(0.5, 0)),
            try .init(offset: SIMD2(0.2, -0.15), rotation: 0.6, scale: SIMD2(1.7, -1.3)),
        ]
        var images: [[SIMD4<Float>]] = []
        for transform in transforms {
            let material = try PhysicallyBasedMaterial(
                baseColorTexture: texture, alphaMode: .opaque, baseColorTextureTransform: transform, shadingModel: .unlit
            )
            let raster = try fixture.raster(material, geometry: plane, transform: placement, camera: camera)
            let traced = try fixture.trace(
                [(plane, material, placement)], camera: camera, lighting: PathTracer.Lighting(environment: .zero),
                samples: 8, bounces: 1, textureSize: 16
            )
            var checked = 0
            for y in 0..<side {
                for x in 0..<side {
                    let uv = SIMD2(Float(x) + 0.5, Float(y) + 0.5) / Float(side)
                    let mapped = transform.apply(to: uv)
                    let cell = mapped * 2 - (mapped * 2).rounded(.down)
                    // Skip pixels whose footprint can reach a block edge or the bilinear seam.
                    guard (0..<2).allSatisfy({ min(cell[$0], 1 - cell[$0]) > 0.2 }) else { continue }
                    let wrapped = mapped - mapped.rounded(.down)
                    let block = Int(wrapped.y * 2) * 2 + Int(wrapped.x * 2)
                    let expected = SIMD4(colors[block], 1)
                    let index = y * side + x
                    #expect(simd_distance(raster[index], expected) < 0.01, "raster pixel \(x), \(y)")
                    #expect(simd_distance(traced[index], expected) < 0.01, "path pixel \(x), \(y)")
                    checked += 1
                }
            }
            #expect(checked > 150)
            images.append(raster)
        }
        // The offset moves the red quadrant's texels half a texture to the right.
        #expect(simd_distance(images[0][8 * side + 8], SIMD4(1, 0, 0, 1)) < 0.01)
        #expect(simd_distance(images[1][8 * side + 8], SIMD4(0, 1, 0, 1)) < 0.01)
    }

    @Test("Normal-map tangent frames follow a rotated texture transform in both renderers")
    func rotatedNormalFrame() throws {
        // A uniform normal map leaning toward +U: tangent-space (0.6, 0, 0.8).
        let normalMap = try fixture.dataTexture(width: 1, height: 1, pixels: [204, 128, 230, 255])
        let placement = SceneTransform(translation: SIMD3(0, 0, -3))
        let planeWithoutTangents = try MeshGeometry.plane(width: 4, height: 4)
        let planeWithTangents = try MeshGeometry(vertices: planeWithoutTangents.vertices.map { vertex in
            // +X increases U and -Y increases V, so the bitangent sign is -1.
            MeshVertex(
                position: vertex.position, normal: vertex.normal, textureCoordinate: vertex.textureCoordinate,
                tangent: SIMD4(1, 0, 0, -1)
            )
        }, indices: planeWithoutTangents.indices)
        let fromRight = simd_normalize(SIMD3<Float>(1, 0, 1))
        let fromBelow = simd_normalize(SIMD3<Float>(0, -1, 1))
        let center = side / 2

        func brightness(
            _ geometry: MeshGeometry, rotation: Float, light: SIMD3<Float>, traced: Bool
        ) throws -> Float {
            let binding = PhysicallyBasedMaterial.TextureBinding(
                texture: normalMap, transform: try .init(rotation: rotation)
            )
            let material = try PhysicallyBasedMaterial(
                roughness: 1, directionToLight: light, ambientIntensity: 0, normalTexture: binding, alphaMode: .opaque,
                specular: .init(factor: 0)
            )
            let image: [SIMD4<Float>]
            if traced {
                let analytic = try PathTracer.AnalyticLight(PunctualLight(kind: .directional), direction: -light)
                image = try fixture.trace(
                    [(geometry, material, placement)], camera: camera,
                    lighting: PathTracer.Lighting(analyticLights: [analytic], environment: .zero), samples: 4, bounces: 1
                )
            } else {
                image = try fixture.raster(material, geometry: geometry, transform: placement, camera: camera)
            }
            return image[center * side + center].x
        }

        // Lambertian white: brightness is cos(normal, light) / pi. The tangent-space
        // normal (t, b, n) maps to world (t, -b, n) unrotated, where +U is +X and +V is -Y.
        // A 90° rotation makes +U' run along +V and +V' along -U: world (-b, -t, n).
        let local = simd_normalize(SIMD3<Float>(204, 128, 230) / 255 * 2 - 1)
        let unrotated = SIMD3(local.x, -local.y, local.z)
        let rotated = SIMD3(-local.y, -local.x, local.z)
        for (geometry, traced) in [(planeWithoutTangents, false), (planeWithTangents, false), (planeWithoutTangents, true)] {
            for (rotation, normal) in [(Float(0), unrotated), (.pi / 2, rotated)] {
                for light in [fromRight, fromBelow] {
                    let expected = simd_dot(normal, light) / .pi
                    let measured = try brightness(geometry, rotation: rotation, light: light, traced: traced)
                    #expect(abs(measured - expected) < 0.01 * expected, "rotation \(rotation), light \(light), traced \(traced)")
                }
            }
        }
        // The rotation moves the lean from the right-lit side to the below-lit side.
        #expect(simd_dot(unrotated, fromRight) > simd_dot(unrotated, fromBelow) + 0.4)
        #expect(simd_dot(rotated, fromBelow) > simd_dot(rotated, fromRight) + 0.4)
    }

    @Test("Specular and clearcoat maps follow their own transforms and UV sets in both renderers")
    func layerMapTransform() throws {
        // Half-black, half-white masks in alpha (specular) and red (clearcoat). Moving a
        // mask by half a texture swaps which side of the plane keeps its highlight.
        var alphaMask: [UInt8] = []
        var redMask: [UInt8] = []
        for _ in 0..<16 {
            for x in 0..<16 {
                alphaMask += [255, 255, 255, x < 8 ? 0 : 255]
                redMask += [x < 8 ? 0 : 255, 0, 0, 255]
            }
        }
        let alpha = try fixture.dataTexture(width: 16, height: 16, pixels: alphaMask)
        let red = try fixture.dataTexture(width: 16, height: 16, pixels: redMask)
        let source = try MeshGeometry.plane(width: 2, height: 2)
        // UV0 runs vertically, so only the secondary set can select the left or right half.
        let plane = try MeshGeometry(vertices: source.vertices.map { vertex in
            MeshVertex(
                position: vertex.position, normal: vertex.normal,
                textureCoordinate: SIMD2(vertex.textureCoordinate.y, vertex.textureCoordinate.x),
                secondaryTextureCoordinate: vertex.textureCoordinate
            )
        }, indices: source.indices)
        let placement = SceneTransform(translation: SIMD3(0, 0, -1))

        func material(specular: Bool, offset: Float) throws -> PhysicallyBasedMaterial {
            let transform = try PhysicallyBasedMaterial.TextureTransform(offset: SIMD2(offset, 0))
            let mask = specular
                ? PhysicallyBasedMaterial.TextureBinding(texture: alpha, textureCoordinateSet: .secondary, transform: transform)
                // The red mask uses UV0 with a quarter turn that maps V back onto U.
                : PhysicallyBasedMaterial.TextureBinding(
                    texture: red, transform: try .init(offset: SIMD2(offset, 0), rotation: -.pi / 2, scale: SIMD2(1, -1))
                )
            return try PhysicallyBasedMaterial(
                roughness: 0.5, directionToLight: SIMD3(0, 0, 1), ambientIntensity: 0,
                baseColorFactor: SIMD4(0, 0, 0, 1), alphaMode: .opaque,
                clearcoat: specular ? nil : .init(factor: 1, roughness: 0.4, factorTexture: mask),
                specular: specular ? .init(factorTexture: mask) : .init(factor: 0)
            )
        }
        func image(_ material: PhysicallyBasedMaterial, traced: Bool) throws -> [SIMD4<Float>] {
            if traced {
                let light = try PathTracer.AnalyticLight(PunctualLight(kind: .directional), direction: SIMD3(0, 0, -1))
                return try fixture.trace(
                    [(plane, material, placement)], camera: camera,
                    lighting: PathTracer.Lighting(analyticLights: [light], environment: .zero), samples: 4, bounces: 1,
                    textureSize: 16
                )
            }
            return try fixture.raster(material, geometry: plane, transform: placement, camera: camera)
        }
        let left = 16 * side + 8
        let right = 16 * side + 24
        for specular in [true, false] {
            for traced in [false, true] {
                let plain = try image(material(specular: specular, offset: 0), traced: traced)
                let moved = try image(material(specular: specular, offset: 0.5), traced: traced)
                #expect(plain[left].x == 0 && plain[right].x > 0.005, "specular \(specular), traced \(traced)")
                #expect(moved[right].x == 0 && moved[left].x > 0.005, "specular \(specular), traced \(traced)")
            }
        }

        // The path tracer traces clearcoat normal maps (PathMediaTests checks their shading).
        let coatNormal = try PhysicallyBasedMaterial(
            alphaMode: .opaque, clearcoat: .init(factor: 1, normalTexture: .init(texture: red))
        )
        let node = SceneNode(transform: placement, mesh: try MeshRenderer(
            device: fixture.device, colorPixelFormat: .rgba16Float, geometry: plane, material: .physicallyBased(coatNormal)
        ))
        #expect(try PathTracingScene(queue: fixture.queue, graph: SceneGraph(roots: [node], camera: camera), textureSize: 16)
            .materialCount == 1)
    }
}
