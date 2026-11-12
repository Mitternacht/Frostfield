import Metal
import Testing
import simd
@testable import Frostfield

@Suite struct StereoPresentationTests {
    @Test func anaglyphEyesAndCombines() throws {
        let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 2, near: 0.1, far: 10)
        let eyes = try StereoPresentation.anaglyphEyes(camera: camera, aspectRatio: 1)
        #expect(abs(eyes.left.position.x + 0.032) < 1e-6)
        #expect(eyes.left.position.y == 0 && eyes.left.position.z == 0)
        #expect(abs(eyes.right.position.x - 0.032) < 1e-6)
        #expect(eyes.left.clipFromWorld != eyes.right.clipFromWorld)

        let red = SIMD4<Float>(1, 0, 0, 1)
        let black = SIMD4<Float>(0, 0, 0, 0)
        let leftOnly = StereoPresentation.anaglyph(left: red, right: black)
        #expect(leftOnly.x == 0.456100)
        #expect(leftOnly.y == 0)
        #expect(leftOnly.z == 0)
        #expect(leftOnly.w == 1)

        #expect(StereoPresentation.parallaxBarrier(left: red, right: black, fragmentY: 0.5) == black)
        #expect(StereoPresentation.parallaxBarrier(left: red, right: black, fragmentY: 1.5) == red)

        let left = try StereoPresentation.stereoEye(
            camera: camera, aspectRatio: 2, eye: .left, aspectScale: 0.5
        )
        let right = try StereoPresentation.stereoEye(
            camera: camera, aspectRatio: 2, eye: .right, aspectScale: 0.5
        )
        let zoomed = try StereoPresentation.stereoEye(
            camera: camera, aspectRatio: 2, eye: .left, zoom: 2, aspectScale: 0.5
        )
        #expect(abs(left.position.x + 0.032) < 1e-6)
        #expect(abs(right.position.x - 0.032) < 1e-6)
        let focus = SIMD4<Float>(0, 0, -10, 1)
        #expect(abs(ndcX(left.clipFromWorld, focus)) < 1e-5)
        #expect(abs(ndcX(right.clipFromWorld, focus)) < 1e-5)
        #expect(abs(ndcX(zoomed.clipFromWorld, focus)) < 1e-5)
        #expect(abs(zoomed.clipFromWorld.columns.0.x - left.clipFromWorld.columns.0.x * 2) < 1e-5)

        let scaled = SceneTransform(scale: SIMD3(2, 1, 1))
        let scaledLeft = try StereoPresentation.stereoEye(
            camera: camera, transform: scaled, aspectRatio: 2, eye: .left, aspectScale: 0.5
        )
        let scaledPoint = scaled.matrix * focus
        #expect(abs(ndcX(scaledLeft.clipFromWorld, scaledPoint)) < 1e-4)
        #expect(abs(scaledLeft.position.x + 0.064) < 1e-5)

        let device = try #require(MTLCreateSystemDefaultDevice())
        let pixels = [
            StereoPresentation.Pixel(combine: .anaglyph, left: red, right: black),
            StereoPresentation.Pixel(combine: .parallaxBarrier, left: red, right: black, fragmentY: 0.5),
            StereoPresentation.Pixel(combine: .parallaxBarrier, left: red, right: black, fragmentY: 1.5),
        ]
        let colors = try StereoPresentation.evaluateOnGPU(pixels, device: device)
        #expect(colors[0] == leftOnly)
        #expect(colors[1] == black)
        #expect(colors[2] == red)
    }

    private func ndcX(_ clip: simd_float4x4, _ point: SIMD4<Float>) -> Float {
        let projected = clip * point
        return projected.x / projected.w
    }
}
