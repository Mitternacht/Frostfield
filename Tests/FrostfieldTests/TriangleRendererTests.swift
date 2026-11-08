import Foundation
import FrostfieldShaderTypes
import Metal
import Testing
@testable import Frostfield

private let systemDevice = MTLCreateSystemDefaultDevice()

/// Construction failures must surface as typed errors, never as a crash or a
/// renderer that silently draws nothing.
struct TriangleRendererFailureTests {
    @Test func `A missing device throws noMetalDevice`() {
        #expect(throws: RendererError.noMetalDevice) {
            try TriangleRenderer(device: nil, colorPixelFormat: .bgra8Unorm)
        }
    }

    @Test(.enabled(if: systemDevice != nil))
    func `A missing vertex function names the function`() {
        #expect(throws: RendererError.shaderFunctionMissing(name: "ff_missing_vertex")) {
            try TriangleRenderer(
                device: systemDevice,
                colorPixelFormat: .bgra8Unorm,
                maxFramesInFlight: 1,
                vertexFunctionName: "ff_missing_vertex",
                fragmentFunctionName: TriangleRenderer.fragmentFunctionName,
                onFrameCompleted: { _ in }
            )
        }
    }

    @Test(.enabled(if: systemDevice != nil))
    func `A missing fragment function names the function`() {
        #expect(throws: RendererError.shaderFunctionMissing(name: "ff_missing_fragment")) {
            try TriangleRenderer(
                device: systemDevice,
                colorPixelFormat: .bgra8Unorm,
                maxFramesInFlight: 1,
                vertexFunctionName: TriangleRenderer.vertexFunctionName,
                fragmentFunctionName: "ff_missing_fragment",
                onFrameCompleted: { _ in }
            )
        }
    }

    @Test func `Errors describe themselves for display`() {
        #expect(RendererError.noMetalDevice.description == "No Metal device is available.")
        #expect(RendererError.shaderFunctionMissing(name: "ff_x").description.contains("ff_x"))
    }
}

/// The shaders ship in the package's own resource bundle. The renderer must
/// load that library rather than whatever default library the host app has.
struct ShaderPackagingTests {
    @Test func `The package bundle contains a compiled default library`() {
        let bundle = TriangleRenderer.shaderBundle
        #expect(bundle != Bundle.main)
        #expect(bundle.url(forResource: "default", withExtension: "metallib") != nil)
    }

    @Test(.enabled(if: systemDevice != nil))
    func `The resolved library holds the triangle functions`() throws {
        let device = try #require(systemDevice)
        let library = try TriangleRenderer.makeShaderLibrary(device: device)
        #expect(library.functionNames.contains(TriangleRenderer.vertexFunctionName))
        #expect(library.functionNames.contains(TriangleRenderer.fragmentFunctionName))
        let vertex = try #require(library.makeFunction(name: TriangleRenderer.vertexFunctionName))
        #expect(vertex.functionType == .vertex)
        let fragment = try #require(library.makeFunction(name: TriangleRenderer.fragmentFunctionName))
        #expect(fragment.functionType == .fragment)
    }
}

/// The uniform record crosses the CPU/GPU boundary. Check its host layout
/// field by field; the render tests below check the values arrive.
struct TriangleUniformsLayoutTests {
    @Test func `Host layout matches the shader contract`() {
        typealias Uniforms = FFTriangleUniforms
        #expect(MemoryLayout<Uniforms>.size == 16)
        #expect(MemoryLayout<Uniforms>.stride == 16)
        #expect(MemoryLayout<Uniforms>.alignment == 8)
        #expect(MemoryLayout<Uniforms>.offset(of: \.scale) == 0)
        #expect(MemoryLayout<Uniforms>.offset(of: \.rotation) == 8)
        #expect(MemoryLayout<Uniforms>.offset(of: \.padding) == 12)
        #expect(FF_TRIANGLE_UNIFORMS_BUFFER_INDEX == 0)
    }

    @Test(arguments: [
        (200, 100, SIMD2<Float>(0.5, 1)),
        (100, 400, SIMD2<Float>(1, 0.25)),
        (64, 64, SIMD2<Float>(1, 1)),
    ])
    func `Aspect scale fits the shorter axis`(width: Int, height: Int, expected: SIMD2<Float>) {
        #expect(TriangleRenderer.aspectScale(width: width, height: height) == expected)
    }
}

/// Renders off-screen on the system GPU and reads pixels back. Each sample
/// point lies well inside or outside the triangle, away from its edges.
@Suite(.enabled(if: systemDevice != nil))
struct TriangleRenderingTests {
    let device: any MTLDevice
    let completions: AsyncStream<FrameCompletion>
    let reportCompletion: AsyncStream<FrameCompletion>.Continuation

    init() throws {
        device = try #require(systemDevice)
        (completions, reportCompletion) = AsyncStream.makeStream(of: FrameCompletion.self)
    }

    func makeRenderer(maxFramesInFlight: Int = 2) throws -> TriangleRenderer {
        let report = reportCompletion
        return try TriangleRenderer(
            device: device,
            colorPixelFormat: .bgra8Unorm,
            maxFramesInFlight: maxFramesInFlight,
            onFrameCompleted: { report.yield($0) }
        )
    }

    @Test func `Draws the triangle apex up with red, green, and blue corners`() async throws {
        let renderer = try makeRenderer()
        let target = try OffscreenTarget(device: device, width: 64, height: 64)
        try await renderAndWait(renderer, rotation: 0, into: target)

        let apex = target.pixel(clipX: 0, clipY: 0.6)
        #expect(apex.red > 180 && apex.green < 60 && apex.blue < 60, "near the red apex: \(apex)")
        let left = target.pixel(clipX: -0.5, clipY: -0.33)
        #expect(left.green > 180 && left.red < 60 && left.blue < 60, "near the green corner: \(left)")
        let right = target.pixel(clipX: 0.5, clipY: -0.33)
        #expect(right.blue > 180 && right.red < 60 && right.green < 60, "near the blue corner: \(right)")

        #expect(target.pixel(clipX: 0, clipY: -0.6).isClear, "below the base")
        #expect(target.pixel(clipX: -0.9, clipY: 0.9).isClear, "outside the corner")
    }

    @Test func `Rotation reaches the shader`() async throws {
        let renderer = try makeRenderer()
        let target = try OffscreenTarget(device: device, width: 64, height: 64)
        try await renderAndWait(renderer, rotation: .pi, into: target)

        #expect(target.pixel(clipX: 0, clipY: 0.6).isClear, "the apex moved away")
        let apex = target.pixel(clipX: 0, clipY: -0.6)
        #expect(apex.red > 180 && apex.green < 60 && apex.blue < 60, "the red apex now points down: \(apex)")
    }

    @Test func `A wide target narrows the triangle instead of stretching it`() async throws {
        let renderer = try makeRenderer()
        let target = try OffscreenTarget(device: device, width: 128, height: 64)
        try await renderAndWait(renderer, rotation: 0, into: target)

        // Without the 0.5 horizontal scale this point would be inside.
        #expect(target.pixel(clipX: 0.5, clipY: -0.3).isClear)
        #expect(!target.pixel(clipX: 0.2, clipY: -0.3).isClear)
    }

    @Test func `A tall target shortens the triangle instead of stretching it`() async throws {
        let renderer = try makeRenderer()
        let target = try OffscreenTarget(device: device, width: 64, height: 128)
        try await renderAndWait(renderer, rotation: 0, into: target)

        // Without the 0.5 vertical scale this point would be inside.
        #expect(target.pixel(clipX: 0, clipY: 0.6).isClear)
        let apex = target.pixel(clipX: 0, clipY: 0.3)
        #expect(apex.red > 180, "the scaled apex region: \(apex)")
    }

    @Test func `A frame requested while every slot is busy is skipped without waiting`() async throws {
        let renderer = try makeRenderer(maxFramesInFlight: 1)
        let target = try OffscreenTarget(device: device, width: 16, height: 16)

        // The outer frame holds the only slot while its target is requested,
        // so the nested request must be declined immediately.
        var nested: FrameOutcome?
        let outer = renderer.drawFrame(rotation: 0) {
            nested = renderer.drawFrame(rotation: 0) { target.frameTarget }
            return target.frameTarget
        }
        #expect(nested == .skippedNoSlot)
        #expect(outer == .submitted(frame: 1))
        #expect(await nextCompletion() == FrameCompletion(frame: 1, succeeded: true))
        #expect(renderer.statistics.skippedNoSlot == 1)
        #expect(renderer.statistics.submitted == 1)
    }

    @Test func `A missing target releases its slot for the next frame`() async throws {
        let renderer = try makeRenderer(maxFramesInFlight: 1)
        let target = try OffscreenTarget(device: device, width: 16, height: 16)

        #expect(renderer.drawFrame(rotation: 0) { nil } == .skippedNoTarget)
        #expect(renderer.drawFrame(rotation: 0) { target.frameTarget } == .submitted(frame: 1))
        #expect(await nextCompletion()?.succeeded == true)
        #expect(renderer.statistics.skippedNoTarget == 1)
    }

    @Test func `Rapid requests never exceed the in-flight bound`() async throws {
        let renderer = try makeRenderer(maxFramesInFlight: 2)
        let target = try OffscreenTarget(device: device, width: 32, height: 32)
        let attempts = 200

        var submitted = 0
        for index in 0..<attempts {
            let outcome = renderer.drawFrame(rotation: Float(index) * 0.01) { target.frameTarget }
            switch outcome {
            case .submitted: submitted += 1
            case .skippedNoSlot: break
            case .skippedNoTarget, .encodingFailed: Issue.record("Unexpected outcome \(outcome)")
            }
            #expect(renderer.statistics.inFlight <= 2)
        }

        var completed = 0
        while completed < submitted, let completion = await nextCompletion() {
            #expect(completion.succeeded)
            completed += 1
        }
        let statistics = renderer.statistics
        #expect(statistics.submitted == submitted)
        #expect(statistics.completed == submitted)
        #expect(statistics.submitted + statistics.skippedNoSlot == attempts)
        #expect(statistics.failed == 0)
    }

    /// With `MTL_DEBUG_LAYER=1` (set by the shared Frostfield scheme's test
    /// action) the system device must be Metal's validation wrapper; otherwise
    /// the GPU tests above would run without API validation.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MTL_DEBUG_LAYER"] == "1"))
    func `API validation wraps the device when requested`() {
        #expect(String(describing: type(of: device)).contains("Debug"), "device type: \(type(of: device))")
    }

    // MARK: Helpers

    private func nextCompletion() async -> FrameCompletion? {
        var iterator = completions.makeAsyncIterator()
        return await iterator.next()
    }

    private func renderAndWait(_ renderer: TriangleRenderer, rotation: Float, into target: OffscreenTarget) async throws {
        let outcome = renderer.drawFrame(rotation: rotation) { target.frameTarget }
        guard case .submitted(let frame) = outcome else {
            Issue.record("Frame was not submitted: \(outcome)")
            return
        }
        let completion = try #require(await nextCompletion())
        #expect(completion == FrameCompletion(frame: frame, succeeded: true))
    }
}

/// A CPU-readable BGRA8 render target cleared to opaque black.
private struct OffscreenTarget {
    struct Pixel: CustomStringConvertible {
        var blue: UInt8, green: UInt8, red: UInt8, alpha: UInt8
        var isClear: Bool { red < 8 && green < 8 && blue < 8 }
        var description: String { "rgba(\(red), \(green), \(blue), \(alpha))" }
    }

    let texture: any MTLTexture

    init(device: any MTLDevice, width: Int, height: Int) throws {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false
        )
        descriptor.usage = .renderTarget
        descriptor.storageMode = .shared
        texture = try #require(device.makeTexture(descriptor: descriptor))
    }

    var frameTarget: FrameTarget {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        return FrameTarget(passDescriptor: pass, drawable: nil)
    }

    /// The pixel containing a clip-space point (y up, as in the shader).
    func pixel(clipX: Float, clipY: Float) -> Pixel {
        let x = min(texture.width - 1, Int((clipX + 1) / 2 * Float(texture.width)))
        let y = min(texture.height - 1, Int((1 - clipY) / 2 * Float(texture.height)))
        var bytes = [UInt8](repeating: 0, count: 4)
        texture.getBytes(&bytes, bytesPerRow: 4, from: MTLRegionMake2D(x, y, 1, 1), mipmapLevel: 0)
        return Pixel(blue: bytes[0], green: bytes[1], red: bytes[2], alpha: bytes[3])
    }
}
