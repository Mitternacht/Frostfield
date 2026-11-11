import Foundation
import Frostfield
import Metal
import Testing

@Suite("Camera lens")
struct CameraLensTests {
    @Test("Field of view, hyperfocal distance and depth-of-field limits follow the thin-lens formulas")
    func optics() throws {
        let lens = try CameraLens(focalLength: 50, fNumber: 2.8, focusDistance: 3, sensorWidth: 36)
        #expect(abs(lens.horizontalFieldOfView - 2 * atan(Float(18) / 50)) < 1e-6)
        #expect(abs(lens.verticalFieldOfView(aspectRatio: 1.5) - 2 * atan(Float(12) / 50)) < 1e-6)
        #expect(abs(lens.apertureDiameter - 50 / 2.8) < 1e-5)

        // H = f² / (N c) + f = 2500 / 0.084 + 50 mm.
        let hyperfocal = Double(lens.hyperfocalDistance(circleOfConfusion: 0.03))
        #expect(abs(hyperfocal - (2500 / 0.084 + 50) / 1000) < 1e-5)
        let h = 2500 / 0.084 + 50
        let limits = lens.depthOfFieldLimits(circleOfConfusion: 0.03)
        #expect(abs(Double(limits.near) - 3000 * (h - 50) / (h + 3000 - 100) / 1000) < 1e-5)
        #expect(abs(Double(limits.far) - 3000 * (h - 50) / (h - 3000) / 1000) < 1e-5)
        var atHyperfocal = lens
        atHyperfocal.focusDistance = Float(hyperfocal)
        let far = atHyperfocal.depthOfFieldLimits(circleOfConfusion: 0.03)
        #expect(far.far == .infinity)
        #expect(abs(Double(far.near) - hyperfocal / 2) < 1e-4)
    }

    @Test("Circle of confusion matches A·f·|d − s| / (d·(s − f)) in millimeters and pixels")
    func circleOfConfusion() throws {
        let lens = try CameraLens(focalLength: 85, fNumber: 1.8, focusDistance: 2, sensorWidth: 36)
        let aperture = 85.0 / 1.8
        for distance in [0.5, 1.0, 2.0, 4.0, 40.0] {
            let d = distance * 1000
            let expected = aperture * 85 * abs(d - 2000) / (d * (2000 - 85))
            #expect(abs(Double(lens.circleOfConfusionDiameter(atDistance: Float(distance))) - expected) < 1e-6 * max(1, expected))
            let radius = Double(lens.circleOfConfusionRadius(atDistance: Float(distance), imageWidth: 1920))
            let signed = (d < 2000 ? -1.0 : 1.0) * expected / 2 * 1920 / 36
            #expect(abs(radius - signed) < 1e-4 * max(1, abs(signed)), "distance \(distance)")
        }
        // Centimeter scenes scale distances, not the optics.
        let centimeters = try CameraLens(focalLength: 85, fNumber: 1.8, focusDistance: 200, sensorWidth: 36, metersPerUnit: 0.01)
        #expect(abs(centimeters.circleOfConfusionDiameter(atDistance: 400) - lens.circleOfConfusionDiameter(atDistance: 4)) < 1e-6)
    }

    @Test("A lens matching a camera reproduces its field of view, and invalid lenses throw")
    func matching() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: 0.7, near: 0.1, far: 100)
        let lens = try CameraLens(matching: camera, aspectRatio: 16.0 / 9, fNumber: 2, focusDistance: 5)
        #expect(abs(lens.verticalFieldOfView(aspectRatio: 16.0 / 9) - 0.7) < 1e-6)
        #expect(abs(try lens.perspectiveCamera(aspectRatio: 16.0 / 9, near: 0.1, far: 100).verticalFieldOfView - 0.7) < 1e-6)
        #expect(throws: CameraLens.Failure.focusInsideFocalLength) {
            try CameraLens(focalLength: 50, fNumber: 2, focusDistance: 0.04)
        }
        #expect(throws: CameraLens.Failure.invalidValues) { try CameraLens(focalLength: 50, fNumber: 0, focusDistance: 2) }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct BokehDepthOfFieldTests {
    private let gpu: PostProcessFixture
    private let camera: PerspectiveCamera
    /// 200 mm f/1.4 focused at 2 m: K = A·f/(s − f) · W/(2·sensor) = 14.1 px at 64 px wide.
    private let lens: CameraLens

    init() throws {
        gpu = try PostProcessFixture()
        camera = try PerspectiveCamera(verticalFieldOfView: 0.4, near: 0.1, far: 100)
        lens = try CameraLens(focalLength: 200, fNumber: 1.4, focusDistance: 2, sensorWidth: 36)
    }

    @Test("The GPU circle of confusion equals the thin-lens radius at sample depths", arguments: [false, true])
    func circleOfConfusion(reversed: Bool) throws {
        let distances: [Float] = [0.3, 0.5, 1, 1.5, 1.9, 2, 2.1, 3, 5, 10, 20, 50, 90]
        let depth = try gpu.depth(width: distances.count, height: 1) { x, _ in
            storedDepth(distance: distances[x], near: 0.1, far: 100, reversed: reversed)
        }
        let output = try gpu.scalarTexture(width: distances.count, height: 1)
        let effect = try BokehDepthOfField(
            device: gpu.device, lens: lens, camera: camera, depthMode: reversed ? .reversed : .forward
        )
        try gpu.run { try effect.encodeCircleOfConfusion(depth: depth, destination: output, commandBuffer: $0) }
        let radii = try gpu.readScalars(output)
        for (index, distance) in distances.enumerated() {
            let expected = lens.circleOfConfusionRadius(atDistance: distance, imageWidth: distances.count)
            #expect(abs(radii[index] - expected) < 1e-3 * max(1, abs(expected)), "z = \(distance): \(radii[index]) ≠ \(expected)")
        }
        #expect(radii[distances.firstIndex(of: 2)!] == 0 || abs(radii[distances.firstIndex(of: 2)!]) < 1e-4)
    }

    private func render(
        width: Int, height: Int, settings: BokehDepthOfField.Settings, distance: @escaping (Int, Int) -> Float,
        color: @escaping (Int, Int) -> SIMD4<Float>
    ) throws -> (input: [SIMD4<Float>], output: [SIMD4<Float>]) {
        let effect = try BokehDepthOfField(device: gpu.device, lens: lens, camera: camera, settings: settings)
        let depth = try gpu.depth(width: width, height: height) { storedDepth(distance: distance($0, $1), near: 0.1, far: 100) }
        let source = try gpu.color(width: width, height: height, pixel: color)
        let destination = try gpu.color(width: width, height: height)
        try gpu.run { try effect.encode(source: source, depth: depth, destination: destination, commandBuffer: $0) }
        return (try gpu.read(source), try gpu.read(destination))
    }

    @Test("A focused plane is untouched and a defocused edge spreads exactly over its circle of confusion")
    func edgeSpread() throws {
        let width = 64
        let height = 24
        let settings = BokehDepthOfField.Settings(maximumRadius: 16, sampleCount: 256, dither: false)
        let step: (Int, Int) -> SIMD4<Float> = { x, _ in x < 32 ? SIMD4(0, 0, 0, 1) : SIMD4(1, 1, 1, 1) }

        let focused = try render(width: width, height: height, settings: settings, distance: { _, _ in 2 }, color: step)
        #expect(focused.output == focused.input)

        let distance: Float = 4
        let radius = lens.circleOfConfusionRadius(atDistance: distance, imageWidth: width)
        #expect(abs(radius - 7.05) < 0.05)
        let blurred = try render(width: width, height: height, settings: settings, distance: { _, _ in distance }, color: step)
        gpu.savePNG(blurred.output, width: width, height: height, name: "bokeh-edge")
        let row = Array(blurred.output[(height / 2) * width..<(height / 2 + 1) * width]).map(\.x)
        // Beyond the radius plus the half-pixel coverage ramp, nothing changes.
        let reach = Int((radius + 0.5).rounded(.up))
        for x in 0..<(32 - reach - 1) { #expect(row[x] == 0, "x = \(x)") }
        for x in (32 + reach + 1)..<width { #expect(row[x] == 1, "x = \(x)") }
        // Inside, the profile follows the fraction of a disk of that radius past the edge.
        func whiteFraction(_ t: Float) -> Float {
            let h = min(abs(t) / radius, 1)
            let segment = (acos(h) - h * (1 - h * h).squareRoot()) / .pi
            return t < 0 ? segment : 1 - segment
        }
        for x in (32 - reach)..<(32 + reach) {
            let expected = whiteFraction(Float(x) + 0.5 - 32)
            #expect(abs(row[x] - expected) < 0.09, "x = \(x): \(row[x]) vs \(expected)")
        }
        for x in (32 - reach)..<(32 + reach - 1) { #expect(row[x] <= row[x + 1] + 1e-3) }
    }

    @Test("Blurred background stays off a sharp subject, and blurred foreground spreads over a sharp background")
    func layers() throws {
        let width = 64
        let height = 24
        let settings = BokehDepthOfField.Settings(maximumRadius: 16, sampleCount: 256, dither: false)
        // Left: subject in focus at 2 m (red). Right: background at 8 m (blue), radius 10.6 px.
        let background = try render(
            width: width, height: height, settings: settings, distance: { x, _ in x < 32 ? 2 : 8 },
            color: { x, _ in x < 32 ? SIMD4(1, 0, 0, 1) : SIMD4(0, 0, 4, 1) }
        )
        let middle = (height / 2) * width
        for x in 0..<32 {
            #expect(background.output[middle + x].z < 0.02, "subject x = \(x): \(background.output[middle + x])")
        }
        #expect(background.output[middle + 45].z > 3.9)

        // Left: foreground at 1 m (green, radius 14.1 px). Right: background in focus at 2 m (white).
        let foreground = try render(
            width: width, height: height, settings: settings, distance: { x, _ in x < 32 ? 1 : 2 },
            color: { x, _ in x < 32 ? SIMD4(0, 1, 0, 1) : SIMD4(1, 1, 1, 1) }
        )
        gpu.savePNG(foreground.output, width: width, height: height, name: "bokeh-foreground")
        // The defocused foreground covers the first few in-focus pixels behind its edge.
        let covered = foreground.output[middle + 34]
        #expect(covered.x < 0.85 && covered.y > 0.95, "\(covered)")
        #expect(foreground.output[middle + 60] == SIMD4(1, 1, 1, 1))
        // Inside its silhouette the defocused foreground is translucent too: 1.5 px in,
        // a 14.1 px disk is 57% foreground, and coverage is continuous across the edge.
        let inside = foreground.output[middle + 30].x
        #expect(abs(inside - (1 - 0.567)) < 0.1, "\(inside)")
        #expect(abs(foreground.output[middle + 31].x - foreground.output[middle + 32].x) < 0.15)
        #expect(foreground.output[middle + 5] == SIMD4(0, 1, 0, 1))
    }

    @Test("Invalid lens settings throw before encoding")
    func invalid() throws {
        var badLens = lens
        badLens.focusDistance = 0.05
        let effect = try BokehDepthOfField(device: gpu.device, lens: badLens, camera: camera)
        let depth = try gpu.depth(width: 4, height: 4) { _, _ in 0.5 }
        let source = try gpu.color(width: 4, height: 4)
        let destination = try gpu.color(width: 4, height: 4)
        let buffer = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PostProcessEffectError.self) {
            try effect.encode(source: source, depth: depth, destination: destination, commandBuffer: buffer)
        }
        let valid = try BokehDepthOfField(device: gpu.device, lens: lens, camera: camera)
        #expect(throws: PostProcessEffectError.missingDepth) {
            try valid.encode(source: source, depth: nil, destination: destination, commandBuffer: buffer)
        }
        // The circle-of-confusion pass names command-buffer problems as such.
        let radii = try gpu.scalarTexture(width: 4, height: 4)
        let unretained = try #require(gpu.queue.makeCommandBufferWithUnretainedReferences())
        #expect(throws: PostProcessEffectError.commandBuffer) {
            try valid.encodeCircleOfConfusion(depth: depth, destination: radii, commandBuffer: unretained)
        }
        let committed = try #require(gpu.queue.makeCommandBuffer())
        committed.commit()
        committed.waitUntilCompleted()
        #expect(throws: PostProcessEffectError.commandBuffer) {
            try valid.encodeCircleOfConfusion(depth: depth, destination: radii, commandBuffer: committed)
        }
    }
}
