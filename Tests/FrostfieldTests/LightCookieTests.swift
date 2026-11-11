import CoreGraphics
import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

struct LightCookieTests {
    @Test("Nonfinite transmittance in any channel is rejected",
          arguments: [SIMD3<Float>(.nan, 0.5, 0.5), SIMD3<Float>(0.5, .nan, 0.5), SIMD3<Float>(0.5, 0.5, .nan)])
    func nonfiniteTexels(pixel: SIMD3<Float>) {
        #expect(throws: LightCookie.Failure.values) {
            try LightCookie(width: 1, height: 1, transmittance: [pixel])
        }
    }

    @Test("Sampling at a NaN coordinate blocks light")
    func invalidCoordinates() throws {
        let cookie = try LightCookie(width: 1, height: 1, transmittance: [SIMD3(1, 1, 1)])
        #expect(cookie.transmittance(u: .nan, v: 0.5) == .zero)
        #expect(cookie.transmittance(u: 0.5, v: .nan) == .zero)
        #expect(cookie.transmittance(u: -.infinity, v: .infinity) == SIMD3(1, 1, 1))
    }

    @Test("Image alpha multiplies linear color")
    func linearAlpha() throws {
        let bytes: [UInt8] = [255, 188, 0, 128, 255, 255, 255, 0]
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let provider = try #require(CGDataProvider(data: Data(bytes) as CFData))
        let image = try #require(CGImage(
            width: 2, height: 1, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 8, space: space,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent
        ))
        let cookie = try LightCookie(cgImage: image)
        let expected = SIMD3<Float>(1, 0.5029, 0) * (128 / 255)
        #expect(simd_reduce_max(abs(cookie.transmittance[0] - expected)) < 0.002)
        #expect(cookie.transmittance[1] == .zero)
    }

    @Test("Cookies validate their texels, decode sRGB images and sample bilinearly with clamped edges")
    func values() throws {
        #expect(throws: LightCookie.Failure.dimensions) { try LightCookie(width: 2, height: 2, transmittance: [SIMD3(1, 1, 1)]) }
        #expect(throws: LightCookie.Failure.values) {
            try LightCookie(width: 1, height: 1, transmittance: [SIMD3(1.5, 0, 0)])
        }
        let cookie = try LightCookie(width: 2, height: 1, transmittance: [SIMD3(1, 0, 0), SIMD3(0, 0, 1)])
        #expect(cookie.transmittance(u: 0.25, v: 0.5) == SIMD3(1, 0, 0))
        #expect(cookie.transmittance(u: 0.5, v: 0.5) == SIMD3(0.5, 0, 0.5))
        #expect(cookie.transmittance(u: -3, v: 9) == SIMD3(1, 0, 0))

        // sRGB 188 is linear 0.5029; the top row of the image is the cookie's first row.
        let bytes: [UInt8] = [188, 188, 188, 255, 0, 0, 0, 255, 255, 0, 0, 255, 0, 255, 0, 255]
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let provider = try #require(CGDataProvider(data: Data(bytes) as CFData))
        let image = try #require(CGImage(
            width: 2, height: 2, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 8, space: space,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue), provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent
        ))
        let decoded = try LightCookie(cgImage: image)
        #expect(abs(decoded.transmittance[0].x - 0.5029) < 0.002 && decoded.transmittance[1] == .zero)
        #expect(decoded.transmittance[2] == SIMD3(1, 0, 0) && decoded.transmittance[3] == SIMD3(0, 1, 0))

        #expect(throws: PunctualLight.Failure.cookieKind) { try PunctualLight(kind: .point(), cookie: cookie) }
        #expect(throws: PunctualLight.Failure.cookieKind) {
            try PunctualLight(kind: .spot(innerConeAngle: 0, outerConeAngle: .pi / 2), cookie: cookie)
        }
        let rig = LightRig.Light(id: "gobo", kind: .point(), position: .zero, target: SIMD3(0, -1, 0), emission: .intensity(1), cookie: cookie)
        #expect(throws: LightRig.Failure.lightTables("gobo")) { try LightRig(lights: [rig]) }

        // glTF has no cookies, so export keeps the spot and reports the loss.
        let spot = try PunctualLight(kind: .spot(innerConeAngle: 0.2, outerConeAngle: 0.5), cookie: cookie)
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        let output = try GLTFExporter().export(SceneGraph(roots: [SceneNode(light: spot)], camera: camera))
        #expect(output.warnings.contains(.lightProfileDropped))
        let plain = try GLTFExporter().export(SceneGraph(
            roots: [SceneNode(light: PunctualLight(kind: .spot(innerConeAngle: 0.2, outerConeAngle: 0.5)))], camera: camera
        ))
        #expect(!plain.warnings.contains(.lightProfileDropped))
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct LightCookieRenderTests {
    private let fixture: PathCameraFixture

    init() throws {
        fixture = try PathCameraFixture()
    }

    @Test("A light table cache releases old sets after replacement and its current set with its owner")
    func tableLifetime() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cookie = try LightCookie(width: 1, height: 1, transmittance: [SIMD3(1, 0, 0)])
        weak var active: LightTextureSet?
        do {
            let cache = LightTextureSet.Cache()
            weak var original: LightTextureSet?
            do {
                let initial = try cache.make(device: device, profiles: [nil], cookies: [cookie])
                let reused = try cache.make(device: device, profiles: [nil], cookies: [cookie])
                #expect(initial === reused)
                original = initial
            }
            #expect(original != nil)
            active = try cache.make(device: device, profiles: [], cookies: [])
            #expect(original == nil)
            withExtendedLifetime(cache) { #expect(active != nil) }
        }
        #expect(active == nil)
    }

    @Test("A spot cookie projects its texels onto the predicted floor pixels in both renderers")
    func projection() throws {
        // Four colored quadrants of 8 × 8 texels each.
        let size = 16
        let colors: [SIMD3<Float>] = [SIMD3(1, 0.2, 0.2), SIMD3(0.2, 1, 0.2), SIMD3(0.2, 0.2, 1), SIMD3(1, 1, 0.2)]
        let texels = (0..<(size * size)).map { index in
            colors[(index / size >= size / 2 ? 2 : 0) + (index % size >= size / 2 ? 1 : 0)]
        }
        let cookie = try LightCookie(width: size, height: size, transmittance: texels)
        let floor = try fixture.lambert(.plane(width: 10, height: 10), albedo: 0.8)
        let floorNode = SceneNode(transform: .init(rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))), mesh: floor)
        let side = 64
        let height: Float = 4
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 3, near: 0.1, far: 30)
        let cameraTransform = SceneTransform(translation: SIMD3(0, height, 0), rotation: simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0)))
        let outer: Float = 0.6
        let position = SIMD3<Float>(0.2, 2, 0.1)
        let target = SIMD3<Float>(0.5, 0, -0.2)

        func render(_ cookie: LightCookie?) throws -> (raster: [SIMD4<Float>], traced: [SIMD4<Float>], light: LightRig.Light) {
            let light = LightRig.Light(
                id: "gobo", kind: .spot(innerAngle: outer - 0.01, outerAngle: outer), position: position, target: target,
                emission: .intensity(8), cookie: cookie
            )
            let rig = try LightRig(lights: [light])
            let graph = SceneGraph(roots: [floorNode] + (try rig.sceneNodes()), camera: camera, cameraTransform: cameraTransform)
            let raster = try fixture.base.readback(fixture.rasterize(graph, width: side, height: side).color)
            let scene = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
            let tracer = try PathTracer(queue: fixture.queue, scene: scene)
            // Pixel-center rays only, so each pixel samples the floor at the predicted point.
            let traced = try fixture.trace(
                tracer, model: .init(projection: .perspective(verticalFieldOfView: .pi / 3), near: 0.1, far: 30),
                transform: cameraTransform, width: side, height: side, samples: 1, lighting: rig.pathLighting()
            )
            return (raster, traced, light)
        }
        let plain = try render(nil)
        let gobo = try render(cookie)
        PathCameraFixture.writeImage(gobo.raster, width: side, height: side, name: "cookie-raster")
        PathCameraFixture.writeImage(gobo.traced, width: side, height: side, name: "cookie-traced")

        let rotation = try gobo.light.transform().rotation
        let right = rotation.act(SIMD3<Float>(1, 0, 0))
        let up = rotation.act(SIMD3<Float>(0, 1, 0))
        let forward = rotation.act(SIMD3<Float>(0, 0, -1))
        var seen = Set<Int>()
        var checked = 0
        for row in 0..<side {
            for column in 0..<side {
                let ndc = SIMD2((Float(column) + 0.5) / Float(side) * 2 - 1, 1 - (Float(row) + 0.5) / Float(side) * 2)
                let tangent = tan(Float.pi / 6)
                let point = SIMD3(ndc.x * tangent * height, 0, -ndc.y * tangent * height)
                let toSurface = simd_normalize(point - position)
                let z = simd_dot(toSurface, forward)
                // Inside the inner cone, away from the quadrant seams.
                guard z > cos(outer - 0.05) else { continue }
                let uv = SIMD2(
                    0.5 + 0.5 * simd_dot(toSurface, right) / (z * tan(outer)),
                    0.5 - 0.5 * simd_dot(toSurface, up) / (z * tan(outer))
                )
                guard abs(uv.x - 0.5) > 0.05, abs(uv.y - 0.5) > 0.05 else { continue }
                let expected = cookie.transmittance(u: uv.x, v: uv.y)
                let quadrant = (uv.y >= 0.5 ? 2 : 0) + (uv.x >= 0.5 ? 1 : 0)
                #expect(expected == colors[quadrant])
                seen.insert(quadrant)
                checked += 1
                let index = row * side + column
                for (name, lit, reference) in [("raster", gobo.raster[index], plain.raster[index]), ("traced", gobo.traced[index], plain.traced[index])] {
                    let ratio = SIMD3(lit.x, lit.y, lit.z) / SIMD3(reference.x, reference.y, reference.z)
                    #expect(simd_reduce_max(abs(ratio - expected)) < 0.01, "\(name) pixel \(column), \(row): \(ratio) vs \(expected)")
                }
            }
        }
        #expect(seen.count == 4 && checked > 300, "\(checked) pixels over quadrants \(seen)")
    }
}
