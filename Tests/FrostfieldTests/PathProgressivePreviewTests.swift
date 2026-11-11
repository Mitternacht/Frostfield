import Foundation
@testable import Frostfield
import Metal
import Testing
import simd

struct PathPreviewResolutionPolicyTests {
    @Test("Completed timing windows adjust area with a deadband and bounded steps")
    func timingWindows() throws {
        let configuration = PathProgressivePreview.Configuration(
            targetGPUSeconds: 0.02, renderScales: 0.25...1, initialScale: 0.5, measurementFrames: 3
        )
        var policy = try PathProgressivePreview.ResolutionPolicy(configuration: configuration, width: 97, height: 65)
        #expect(policy.renderSize.width == 48 && policy.renderSize.height == 32)
        for seconds: Double? in [nil, 0, .infinity, 0.08, .nan, 0.08] {
            policy.record(seconds: seconds)
            #expect(policy.scale == 0.5)
        }
        policy.record(seconds: 0.08)
        #expect(abs(policy.scale - sqrt(0.5) * 0.5) < 1e-12)
        #expect(policy.renderSize.width == 34 && policy.renderSize.height == 22)
        let lower = policy.scale
        for _ in 0..<3 { policy.record(seconds: 0.021) }
        #expect(policy.scale == lower)
        for _ in 0..<3 { policy.record(seconds: 0.001) }
        #expect(abs(policy.scale - 0.5) < 1e-12)

        policy.record(seconds: 0.2)
        policy.record(seconds: 0.2)
        try policy.resize(width: 31, height: 13)
        policy.record(seconds: 0.2)
        #expect(abs(policy.scale - 0.5) < 1e-12)
        #expect(policy.renderSize.width == 15 && policy.renderSize.height == 6)
    }

    @Test("The preview bounds native allocations even for an oversized canvas")
    func bounds() throws {
        let config = PathProgressivePreview.Configuration()
        var policy = try PathProgressivePreview.ResolutionPolicy(configuration: config, width: 7_681, height: 4_321)
        #expect(policy.renderSize.width <= 4_096 && policy.renderSize.height <= 4_096)
        #expect(policy.renderSize.width * policy.renderSize.height <= 4_194_304)
        #expect(policy.scale >= 0.25 && policy.scale < 1)
        #expect(throws: PathProgressivePreview.Failure.dimensions) { try policy.resize(width: 0, height: 13) }
        #expect(policy.canvasWidth == 7_681 && policy.canvasHeight == 4_321)
        #expect(throws: PathProgressivePreview.Failure.dimensions) {
            try PathProgressivePreview.ResolutionPolicy(
                configuration: .init(renderScales: 0.9...1), width: 7_681, height: 4_321
            )
        }
        #expect(throws: PathProgressivePreview.Failure.configuration) {
            try PathProgressivePreview.ResolutionPolicy(configuration: .init(measurementFrames: 1), width: 9, height: 7)
        }
    }
}

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PathProgressivePreviewTests {
    private let model = PathTracer.CameraModel(projection: .orthographic(height: 2))

    private struct Fixture {
        let gpu: PathTestFixture
        let tracer: PathTracer

        init(readProjection: Bool = false) throws {
            gpu = try PathTestFixture()
            let plane = try MeshGeometry.plane(width: 8, height: 8)
            let mesh: MeshRenderer
            if readProjection {
                let graph = try SurfaceShaderGraph { s in
                    .init(emission: s.vector(s.projectionMatrix.column(0).x, s.constant(Float(0)), s.constant(Float(0))))
                }
                mesh = try MeshRenderer(
                    device: gpu.device, colorPixelFormat: .rgba16Float, geometry: plane,
                    material: .physicallyBased(try .init(alphaMode: .opaque, surfaceGraph: graph))
                )
            } else {
                mesh = try gpu.pbr(plane, emission: SIMD3(1, 0.25, 0))
            }
            let graph = SceneGraph(
                roots: [SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh)],
                camera: gpu.camera
            )
            let scene = try PathTracingScene(queue: gpu.queue, graph: graph, options: .init(acceleration: .software))
            tracer = try PathTracer(queue: gpu.queue, scene: scene)
        }

        func command() throws -> any MTLCommandBuffer { try #require(gpu.queue.makeCommandBuffer()) }

        func pixels(_ texture: any MTLTexture) throws -> [SIMD4<Float>] {
            let shared = try gpu.output(width: texture.width, height: texture.height)
            let buffer = try command()
            let blit = try #require(buffer.makeBlitCommandEncoder())
            blit.copy(from: texture, to: shared)
            blit.endEncoding()
            try gpu.finish(buffer)
            return try gpu.readback(shared)
        }
    }

    @Test("Completed GPU timing changes size once per window and restarts actual accumulation")
    func measuredChange() throws {
        let fixture = try Fixture()
        let preview = try PathProgressivePreview(
            tracer: fixture.tracer, canvasWidth: 97, canvasHeight: 65,
            configuration: .init(targetGPUSeconds: 1e-12, measurementFrames: 2)
        )
        let hasPassTiming = (try? PathTracer.Timing(device: fixture.gpu.device)) != nil
        #expect(preview.adaptsToGPUTime == hasPassTiming)
        for expectedSamples in 1...2 {
            let buffer = try fixture.command()
            let output = try #require(try preview.encode(cameraModel: model, commandBuffer: buffer))
            #expect(output.width == 97 && output.height == 65)
            try fixture.gpu.finish(buffer)
            let frame = try #require(preview.latestFrame)
            #expect(frame.sampleCount == expectedSamples)
            if preview.adaptsToGPUTime {
                let seconds = try #require(frame.gpuSeconds)
                #expect(seconds > 0)
                print("Preview GPU pass \(expectedSamples): \(seconds * 1_000) ms")
                #expect(abs(preview.renderScale - (expectedSamples == 1 ? 1 : sqrt(0.5))) < 1e-12)
            } else {
                #expect(frame.gpuSeconds == nil)
                #expect(preview.renderScale == 1)
            }
        }
        let old = try #require(preview.latestFrame)
        let original = try fixture.pixels(old.texture)
        let buffer = try fixture.command()
        let next = try #require(try preview.encode(cameraModel: model, commandBuffer: buffer))
        #expect(preview.latestFrame?.texture === old.texture)
        if preview.adaptsToGPUTime {
            #expect(next.width == 68 && next.height == 45)
        }
        try fixture.gpu.finish(buffer)
        let latest = try #require(preview.latestFrame)
        #expect(latest.sampleCount == (preview.adaptsToGPUTime ? 1 : 3))
        #expect(latest.texture === next)
        if preview.adaptsToGPUTime {
            #expect(try fixture.pixels(old.texture) == original)
        }
    }

    @Test("Sample caps, reset, resize and discarded commands preserve the last successful image")
    func lifecycle() throws {
        let fixture = try Fixture()
        let preview = try PathProgressivePreview(
            tracer: fixture.tracer, canvasWidth: 11, canvasHeight: 7,
            configuration: .init(renderScales: 0.5...0.5, samplesPerFrame: 2, maximumSamples: 3)
        )
        let first = try fixture.command()
        let output = try #require(try preview.encode(cameraModel: model, commandBuffer: first))
        #expect(output.width == 5 && output.height == 3)
        #expect(preview.latestFrame == nil)
        let second = try fixture.command()
        #expect(throws: PathProgressivePreview.Failure.frameInFlight) {
            try preview.encode(cameraModel: model, commandBuffer: second)
        }
        try fixture.gpu.finish(first)
        #expect(preview.latestFrame?.sampleCount == 2)
        let refined = try #require(try preview.encode(cameraModel: model, commandBuffer: second))
        #expect(refined !== output)
        #expect(preview.latestFrame?.texture === output)
        try fixture.gpu.finish(second)
        let old = try #require(preview.latestFrame)
        #expect(old.texture === refined && old.sampleCount == 3)
        let capped = try fixture.command()
        #expect(try preview.encode(cameraModel: model, commandBuffer: capped) == nil)

        try preview.resize(width: 13, height: 9)
        let invalidated = try #require(try preview.encode(cameraModel: model, commandBuffer: capped))
        #expect(invalidated.width == 6 && invalidated.height == 4)
        preview.reset()
        try fixture.gpu.finish(capped)
        #expect(preview.latestFrame?.texture === old.texture)

        let abandoned = try fixture.command()
        _ = try preview.encode(cameraModel: model, commandBuffer: abandoned)
        try preview.discardUnsubmittedFrame(abandoned)
        #expect(preview.latestFrame?.texture === old.texture)
        let completed = try fixture.command()
        let next = try #require(try preview.encode(cameraModel: model, commandBuffer: completed))
        try fixture.gpu.finish(completed)
        #expect(preview.latestFrame?.texture === next && preview.latestFrame?.sampleCount == 2)

        let moved = try fixture.command()
        _ = try preview.encode(
            cameraModel: model, cameraTransform: .init(translation: SIMD3(0.5, 0, 0)), commandBuffer: moved
        )
        try fixture.gpu.finish(moved)
        #expect(preview.latestFrame?.sampleCount == 2)
    }

    @Test("An odd canvas retains its projection aspect after the trace grid rounds down")
    func oddAspect() throws {
        let fixture = try Fixture(readProjection: true)
        let preview = try PathProgressivePreview(
            tracer: fixture.tracer, canvasWidth: 5, canvasHeight: 3, configuration: .init(renderScales: 0.5...0.5)
        )
        let command = try fixture.command()
        let output = try #require(try preview.encode(cameraModel: model, commandBuffer: command))
        #expect(output.width == 2 && output.height == 1)
        try fixture.gpu.finish(command)
        // Orthographic half-height 1, half-width 5/3. The graph sees a
        // projection coefficient of 3/5 even though the grid's aspect is 2.
        let pixels = try fixture.pixels(output)
        #expect(pixels.allSatisfy { abs($0.x - 0.6) < 0.001 && $0.w == 1 })
    }
}
