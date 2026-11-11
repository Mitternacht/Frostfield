import Foundation
import Frostfield
import Metal
import Testing
import simd

struct PhysicalCameraTests {
    @Test("Field of view follows focal length and the fitted film, as three.js setFocalLength computes it")
    func fieldOfView() throws {
        let camera = try PhysicalCamera(lens: PhysicalLens(focalLength: 50))
        // A 3:2 image covers the whole 36 × 24 mm frame: 2 atan(24 / 100) and 2 atan(36 / 100).
        #expect(abs(try camera.verticalFieldOfView(aspectRatio: 1.5) - Float(2 * atan(24.0 / 100))) < 1e-6)
        #expect(abs(try camera.horizontalFieldOfView(aspectRatio: 1.5) - Float(2 * atan(36.0 / 100))) < 1e-6)
        // A 16:9 image crops the frame to 36 × 20.25 mm; a square image to 24 × 24 mm.
        #expect(try camera.filmSize(aspectRatio: 16.0 / 9) == SIMD2(36, 20.25))
        #expect(try camera.filmSize(aspectRatio: 1) == SIMD2(24, 24))
        var vertical = camera
        vertical.sensorFit = .vertical
        #expect(try vertical.filmSize(aspectRatio: 16.0 / 9) == SIMD2(24 * 16.0 / 9, 24))
        var horizontal = camera
        horizontal.sensorFit = .horizontal
        #expect(try horizontal.filmSize(aspectRatio: 1) == SIMD2(36, 36))
        // three.js PerspectiveCamera: fov = 2 atan(filmGauge / max(aspect, 1) / (2 f)) with a 35 mm gauge.
        let threeFOV = 2 * atan(35.0 / 1.5 / (2 * 35))
        let three = try PhysicalCamera(lens: PhysicalLens(focalLength: 35, focusDistance: 2), sensorSize: SIMD2(35, 35 / 1.5))
        #expect(abs(Double(try three.verticalFieldOfView(aspectRatio: 1.5)) - threeFOV) < 1e-6)
        #expect(abs(try PhysicalCamera.focalLength(verticalFieldOfView: Float(threeFOV), filmHeight: 35 / 1.5) - 35) < 1e-4)
        let perspective = try camera.perspectiveCamera(aspectRatio: 1.5)
        #expect(perspective.verticalFieldOfView == (try camera.verticalFieldOfView(aspectRatio: 1.5)))
        #expect(perspective.near == camera.near && perspective.far == camera.far)
    }

    @Test("Circle of confusion, aperture radii and exposure values follow their closed forms")
    func closedForms() throws {
        let lens = try PhysicalLens(focalLength: 50, fNumber: 2, focusDistance: 2)
        let camera = try PhysicalCamera(lens: lens)
        // c = f² |d − s| / (N d (s − f)) = 0.05² × 2 / (2 × 4 × 1.95) m.
        #expect(abs(Double(try camera.circleOfConfusionDiameter(distance: 4)) - 1000 * 0.0025 * 2 / (2 * 4 * 1.95)) < 1e-6)
        #expect(try camera.circleOfConfusionDiameter(distance: 2) == 0)
        // Pixels: half the diameter times 1,000 px per 24 mm of film height at 3:2.
        let radius = try camera.circleOfConfusionRadius(distance: 4, imageHeight: 1_000, aspectRatio: 1.5)
        #expect(abs(Double(radius) - 1000 * 0.0025 * 2 / (2 * 4 * 1.95) / 2 * 1000 / 24) < 1e-4)
        #expect(abs(camera.apertureRadius - 0.0125) < 1e-7)
        #expect(abs(Double(camera.pathApertureRadius) - 0.0125 * 2 / 1.95) < 1e-7)
        let dof = try camera.depthOfFieldSettings(aspectRatio: 1.5, maximumRadius: 20)
        #expect(dof.focusDistance == 2 && dof.fNumber == 2 && abs(dof.sensorHeight - 0.024) < 1e-7 && dof.maximumRadius == 20)

        // EV100 = log2(N² / t × 100 / S): sunny sixteen at ISO 100 and 1/100 s is log2(25,600).
        var sunny = try PhysicalCamera(lens: PhysicalLens(fNumber: 16), shutterTime: 1.0 / 100, iso: 100)
        #expect(abs(Double(sunny.ev100) - log2(25_600)) < 1e-5)
        sunny.iso = 200
        #expect(abs(Double(sunny.ev100) - (log2(25_600) - 1)) < 1e-5)
        sunny.exposureCompensation = 1
        #expect(abs(Double(sunny.ev100) - (log2(25_600) - 2)) < 1e-5)
        // Saturation: 1 / (1.2 × 2^EV); middle grey: 0.18 × 100 / (12.5 × 2^EV).
        #expect(abs(Double(try sunny.exposure(calibration: .saturation)) - 1 / (1.2 * exp2(log2(25_600) - 2))) < 1e-9)
        #expect(abs(Double(try sunny.exposure(calibration: .middleGrey)) - 1.44 / exp2(log2(25_600) - 2)) < 1e-9)
        #expect(abs(try sunny.exposureStops(calibration: .middleGrey) - log2(try sunny.exposure(calibration: .middleGrey))) < 1e-6)
        // Meters: reflected log2(L × 100 / 12.5), incident log2(E × 100 / 250).
        #expect(abs(try PhysicalCamera.ev100(luminance: 12.5) - log2(100)) < 1e-12)
        #expect(abs(try PhysicalCamera.ev100(illuminance: 250) - log2(100)) < 1e-12)
        let time = try PhysicalCamera.shutterTime(ev100: 10, fNumber: 8, iso: 400)
        #expect(throws: PhysicalCamera.Failure.exposure) {
            try PhysicalCamera.shutterTime(ev100: 10, fNumber: -8, iso: 400)
        }
        #expect(throws: PhysicalCamera.Failure.exposure) {
            try PhysicalCamera.shutterTime(ev100: 10, fNumber: 8, iso: 0.5)
        }
        #expect(abs(PhysicalCamera.ev100(fNumber: 8, shutterTime: time, iso: 400) - 10) < 1e-6)
    }

    @Test("Lens, sensor, exposure and display ranges are validated")
    func validation() throws {
        #expect(throws: PhysicalCamera.Failure.lens) { try PhysicalLens(focalLength: 0) }
        #expect(throws: PhysicalCamera.Failure.lens) { try PhysicalLens(fNumber: .nan) }
        #expect(throws: PhysicalCamera.Failure.lens) { try PhysicalLens(focalLength: 50, focusDistance: 0.04) }
        #expect(throws: PhysicalCamera.Failure.lens) { try PhysicalLens(apertureBlades: 2) }
        #expect(throws: PhysicalCamera.Failure.lens) { try PhysicalLens(apertureBlades: 17) }
        #expect(throws: PhysicalCamera.Failure.lens) { try PhysicalLens(anamorphicRatio: 20) }
        let lens = try PhysicalLens()
        #expect(throws: PhysicalCamera.Failure.sensor) { try PhysicalCamera(lens: lens, sensorSize: SIMD2(0, 24)) }
        #expect(throws: PhysicalCamera.Failure.exposure) { try PhysicalCamera(lens: lens, shutterTime: 0) }
        #expect(throws: PhysicalCamera.Failure.exposure) { try PhysicalCamera(lens: lens, iso: 0) }
        #expect(throws: PhysicalCamera.Failure.clipping) { try PhysicalCamera(lens: lens, near: 1, far: 1) }
        let camera = try PhysicalCamera(lens: lens)
        #expect(throws: PhysicalCamera.Failure.aspectRatio) { try camera.filmSize(aspectRatio: 0) }
        #expect(throws: PhysicalCamera.Failure.distance) { try camera.circleOfConfusionDiameter(distance: -1) }
        // f/1 at 1 s and ISO 1,000,000 is EV100 −13.3: about +13.5 stops, inside ±16; f/256 at 1 µs and ISO 1 is not.
        let dark = try PhysicalCamera(lens: PhysicalLens(fNumber: 256), shutterTime: 1e-6, iso: 1)
        #expect(throws: PhysicalCamera.Failure.self) { try dark.displaySettings() }
        let stereo = #expect(throws: PhysicalCamera.Failure.self) {
            try camera.stereoPair(cameraTransform: .init(), eyeSeparation: 0, aspectRatio: 1)
        }
        #expect(stereo == .stereo)
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PhysicalCameraRenderTests {
    private let fixture: PathCameraFixture

    init() throws {
        fixture = try PathCameraFixture()
    }

    /// A one-pixel emissive sphere six meters ahead, the only thing in an empty scene.
    private func pointSource() throws -> (graph: SceneGraph, tracer: PathTracer) {
        // The sphere subtends about one pixel of radius at 128 pixels over 2 atan(0.12).
        let emitter = try fixture.lambert(.uvSphere(radius: 0.0112, segments: 24, rings: 12), albedo: 0, emission: SIMD3(1, 1, 1))
        let camera = try PerspectiveCamera(verticalFieldOfView: 0.5, near: 0.1, far: 30)
        let graph = SceneGraph(roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -6)), mesh: emitter)], camera: camera)
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
        return (graph, try PathTracer(queue: fixture.queue, scene: scene))
    }

    @Test("Path-traced bokeh of a point takes the polygonal aperture with the thin-lens circumradius")
    func polygonalBokeh() throws {
        let side = 128
        let blades = 5
        let rotation: Float = 0.3
        let camera = try PhysicalCamera(
            lens: PhysicalLens(focalLength: 100, fNumber: 1.4, focusDistance: 0.6, apertureBlades: blades, apertureRotation: rotation),
            near: 0.1, far: 30
        )
        let (_, tracer) = try pointSource()
        let model = try camera.pathCameraModel(aspectRatio: 1)
        var pinholeModel = model
        pinholeModel.lens = nil
        let pinhole = try fixture.trace(tracer, model: pinholeModel, width: side, height: side, samples: 256)
        let bokeh = try fixture.trace(tracer, model: model, width: side, height: side, samples: 4_096)
        PathCameraFixture.writeImage(bokeh, width: side, height: side, name: "bokeh-pentagon", scale: 300)

        // c = f² |d − s| / (N d (s − f)) on 24 mm of film, 128 pixels tall.
        let predicted = try camera.circleOfConfusionRadius(distance: 6, imageHeight: side, aspectRatio: 1)
        #expect(abs(Double(predicted) - 1000 * 0.01 * 5.4 / (1.4 * 6 * 0.5) / 2 * 128 / 24) < 1e-3)
        let energy = pinhole.reduce(0) { $0 + $1.x }
        let total = bokeh.reduce(0) { $0 + $1.x }
        #expect(abs(total / energy - 1) < 0.03, "bokeh \(total) vs pinhole \(energy)")

        // Polar coordinates about the image center with +y up, matching the lens axes.
        func polar(_ index: Int) -> (radius: Float, angle: Float) {
            let x = Float(index % side) + 0.5 - Float(side) / 2
            let y = Float(side) / 2 - (Float(index / side) + 0.5)
            return (simd_length(SIMD2(x, y)), atan2(y, x))
        }
        func angularDistance(_ a: Float, _ b: Float) -> Float {
            abs(remainder(a - b, 2 * .pi))
        }
        let sector = 2 * Float.pi / Float(blades)
        let inradius = predicted * cos(.pi / Float(blades))
        // Uniform aperture sampling spreads the point's energy evenly: the interior level is E / area.
        let interior = bokeh.indices.filter { polar($0).radius < inradius - 2.5 }.map { bokeh[$0].x }
        let level = interior.reduce(0, +) / Float(interior.count)
        let measuredArea = energy / level
        let measuredRadius = sqrt(measuredArea / (Float(blades) / 2 * sin(sector)))
        #expect(abs(measuredRadius / predicted - 1) < 0.03, "circumradius \(measuredRadius) vs \(predicted)")
        // Beyond the inscribed circle only the corners remain: no energy toward the edge midpoints.
        var corners: Float = 0
        var edges: Float = 0
        var outside: Float = 0
        for index in bokeh.indices {
            let (radius, angle) = polar(index)
            if radius > predicted + 2.5 { outside += bokeh[index].x }
            guard radius > inradius + 2, radius < predicted + 2.5 else { continue }
            let offset = angularDistance(angle - rotation, (((angle - rotation) / sector).rounded()) * sector)
            if offset < 0.2 { corners += bokeh[index].x }
            if offset > sector / 2 - 0.2 { edges += bokeh[index].x }
        }
        #expect(corners > 0.025 * total, "corner energy \(corners) of \(total)")
        #expect(edges < 0.02 * corners, "edge-midpoint energy \(edges) vs corners \(corners)")
        #expect(outside < 0.005 * total, "energy past the circumradius \(outside)")
    }

    @Test("The raster depth-of-field pass and the path tracer blur by the same circle of confusion")
    func rasterAndPathTracerAgree() throws {
        let side = 128
        // f = 100 mm, f/1.4, focus 0.985 m: a point at 6 m blurs to an 18.0 px radius, under the raster limit of 24,
        // so the gather's round(radius) resolves errors above 2.8%.
        let camera = try PhysicalCamera(
            lens: PhysicalLens(focalLength: 100, fNumber: 1.4, focusDistance: 0.985), near: 0.1, far: 30
        )
        let radius = try camera.circleOfConfusionRadius(distance: 6, imageHeight: side, aspectRatio: 1)
        #expect(abs(Double(radius) - 1000 * 0.01 * 5.015 / (1.4 * 6 * 0.885) / 2 * 128 / 24) < 1e-3)
        #expect(abs(radius - 17.99) < 0.01)

        // Path tracing: a round aperture; measure the radius from the interior level.
        let (_, tracer) = try pointSource()
        let model = try camera.pathCameraModel(aspectRatio: 1)
        var pinholeModel = model
        pinholeModel.lens = nil
        let energy = try fixture.trace(tracer, model: pinholeModel, width: side, height: side, samples: 256).reduce(0) { $0 + $1.x }
        let bokeh = try fixture.trace(tracer, model: model, width: side, height: side, samples: 4_096)
        PathCameraFixture.writeImage(bokeh, width: side, height: side, name: "bokeh-round", scale: 300)
        var interior: [Float] = []
        for index in bokeh.indices {
            let offset = SIMD2(Float(index % side) + 0.5, Float(index / side) + 0.5) - Float(side) / 2
            if simd_length(offset) < radius - 2.5 { interior.append(bokeh[index].x) }
        }
        let level = interior.reduce(0, +) / Float(interior.count)
        let traced = sqrt(energy / level / .pi)
        #expect(abs(traced / radius - 1) < 0.03, "path-traced radius \(traced) vs \(radius)")

        // Raster: a lit pixel on a black plane at the same depth, then DepthOfField with the camera's settings.
        let black = try fixture.lambert(.plane(width: 20, height: 20), albedo: 0)
        // Centered on pixel (64, 64): half a pixel right and down of the axis, 0.6 px wide.
        let pixel: Float = 6 * 0.12 / 64
        let dot = try fixture.lambert(.plane(width: 0.6 * pixel, height: 0.6 * pixel), albedo: 0, emission: SIMD3(1, 1, 1))
        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -6)), mesh: black),
            SceneNode(transform: .init(translation: SIMD3(0.5 * pixel, -0.5 * pixel, -5.999)), mesh: dot),
        ], camera: try camera.perspectiveCamera(aspectRatio: 1))
        let (color, depth) = try fixture.rasterize(graph, width: side, height: side)
        let blurred = try fixture.base.output(width: side, height: side, usage: [.renderTarget, .shaderRead])
        let effect = try DepthOfField(device: fixture.device)
        let buffer = try #require(fixture.queue.makeCommandBuffer())
        try effect.encode(
            source: color, depth: depth, destination: blurred, camera: camera.perspectiveCamera(aspectRatio: 1),
            settings: camera.depthOfFieldSettings(aspectRatio: 1, maximumRadius: 24), commandBuffer: buffer
        )
        try fixture.base.finish(buffer)
        let sharp = try fixture.base.readback(color)
        let gathered = try fixture.base.readback(blurred)
        PathCameraFixture.writeImage(gathered, width: side, height: side, name: "raster-dof", scale: 30)
        #expect(sharp[64 * side + 64].x > 0.9 && sharp[64 * side + 65].x == 0)
        // The gather's farthest axial taps sit at round(radius) pixels; nothing lies beyond.
        let reach = Int(radius.rounded())
        let lit = (0..<side).filter { gathered[64 * side + $0].x > 1e-4 }
        #expect(lit == [64 - reach, 64, 64 + reach], "lit columns \(lit) for radius \(radius)")
        let litRows = (0..<side).filter { gathered[$0 * side + 64].x > 1e-4 }
        #expect(litRows == [64 - reach, 64, 64 + reach])
    }

    @Test("An anamorphic ratio squeezes the aperture's horizontal extent")
    func anamorphicBokeh() throws {
        let side = 96
        let (_, tracer) = try pointSource()
        let lens = try PhysicalLens(focalLength: 100, fNumber: 1.4, focusDistance: 0.7, anamorphicRatio: 0.5)
        let camera = try PhysicalCamera(lens: lens, near: 0.1, far: 30)
        let image = try fixture.trace(tracer, model: camera.pathCameraModel(aspectRatio: 1), width: side, height: side, samples: 2_048)
        PathCameraFixture.writeImage(image, width: side, height: side, name: "bokeh-anamorphic", scale: 200)
        var moments = SIMD3<Float>.zero
        for index in image.indices {
            let offset = SIMD2(Float(index % side) + 0.5, Float(index / side) + 0.5) - Float(side) / 2
            moments += image[index].x * SIMD3(1, offset.x * offset.x, offset.y * offset.y)
        }
        // An ellipse with half the width has a quarter of the horizontal second moment.
        let ratio = moments.y / moments.z
        #expect(abs(ratio - 0.25) < 0.03, "σx² / σy² = \(ratio)")
    }

    @Test("Camera exposure maps a metered 18% grey card under 1,000 lux to middle grey in both renderers")
    func middleGrey() throws {
        let side = 16
        let card = try fixture.lambert(.plane(width: 4, height: 4), albedo: 0.18)
        let lux: Float = 1_000
        let sun = try PunctualLight(kind: .directional, intensity: lux)
        let lens = try PhysicalLens(focalLength: 50, fNumber: 4, focusDistance: 3)
        let luminance = 0.18 * lux / .pi
        // Spot-meter the card at f/4 and ISO 100.
        let time = try PhysicalCamera.shutterTime(ev100: PhysicalCamera.ev100(luminance: luminance), fNumber: 4, iso: 100)
        let camera = try PhysicalCamera(lens: lens, shutterTime: time, iso: 100, near: 0.1, far: 30)
        #expect(abs(Double(try camera.exposure(calibration: .middleGrey)) * Double(luminance) - 0.18) < 1e-5)
        #expect(abs(Double(try camera.exposure(calibration: .saturation)) * Double(luminance) - 1 / 9.6) < 1e-5)

        let graph = SceneGraph(roots: [
            SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: card),
            SceneNode(light: sun),
        ], camera: try camera.perspectiveCamera(aspectRatio: 1))
        let raster = try fixture.rasterize(graph, width: side, height: side).color
        let scene = try PathTracingScene(queue: fixture.queue, graph: graph, textureSize: 16)
        let tracer = try PathTracer(queue: fixture.queue, scene: scene)
        let traced = try fixture.trace(
            tracer, model: camera.pathCameraModel(aspectRatio: 1), width: side, height: side, samples: 16,
            lighting: .init(analyticLights: PathTracer.AnalyticLight.lights(in: graph))
        )
        let tracedTexture = try fixture.base.output(width: side, height: side, usage: [.shaderRead, .shaderWrite])
        traced.map(SIMD4<Float16>.init).withUnsafeBytes { bytes in
            tracedTexture.replace(region: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0, withBytes: bytes.baseAddress!, bytesPerRow: side * 8)
        }
        let center = side / 2 * side + side / 2
        let rasterHDR = try fixture.base.readback(raster)[center]
        #expect(abs(rasterHDR.x / luminance - 1) < 0.005, "raster \(rasterHDR.x) vs \(luminance) cd/m²")
        #expect(abs(traced[center].x / luminance - 1) < 0.005, "traced \(traced[center].x) vs \(luminance) cd/m²")

        // PBR Neutral leaves 0.18 at 0.18 − 0.04 = 0.14 linear, sRGB code 105; 1 / 9.6 becomes 0.0642, code 72.
        func display(_ source: any MTLTexture, format: MTLPixelFormat, calibration: PhysicalCamera.ExposureCalibration) throws -> SIMD4<Float> {
            let hdr = try HDRDisplay(device: fixture.device, destinationPixelFormat: format)
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: side, height: side, mipmapped: false)
            descriptor.storageMode = .shared
            descriptor.usage = [.renderTarget, .shaderRead]
            let output = try #require(fixture.device.makeTexture(descriptor: descriptor))
            let buffer = try #require(fixture.queue.makeCommandBuffer())
            try hdr.encode(source: source, destination: output, settings: camera.displaySettings(calibration: calibration), commandBuffer: buffer)
            try fixture.base.finish(buffer)
            if format == .rgba16Float { return try fixture.base.readback(output)[center] }
            var bytes = [UInt8](repeating: 0, count: side * side * 4)
            output.getBytes(&bytes, bytesPerRow: side * 4, from: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0)
            return SIMD4(Float(bytes[center * 4]), Float(bytes[center * 4 + 1]), Float(bytes[center * 4 + 2]), Float(bytes[center * 4 + 3]))
        }
        func srgbCode(_ linear: Double) -> Double { (1.055 * pow(linear, 1 / 2.4) - 0.055) * 255 }
        for source in [raster, tracedTexture] {
            let linear = try display(source, format: .rgba16Float, calibration: .middleGrey)
            #expect(abs(linear.x - 0.14) < 0.002, "display-linear \(linear.x)")
            let code = try display(source, format: .rgba8Unorm_srgb, calibration: .middleGrey)
            #expect(abs(Double(code.x) - srgbCode(0.14)) <= 1, "sRGB code \(code.x) vs \(srgbCode(0.14))")
            let saturation = try display(source, format: .rgba16Float, calibration: .saturation)
            #expect(abs(saturation.x - Float(1 / 9.6 - 0.04)) < 0.002, "saturation display-linear \(saturation.x)")
        }
    }
}
