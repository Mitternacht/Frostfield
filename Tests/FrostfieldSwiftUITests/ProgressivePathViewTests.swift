import Frostfield
@testable import FrostfieldSwiftUI
import Metal
import Testing
import simd

@MainActor
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires Metal"))
struct ProgressivePathViewTests {
    private struct Fixture {
        let queue: any MTLCommandQueue
        let graph: SceneGraph
        let scene: PathTracingScene
        let tracer: PathTracer
        let camera = PathTracer.CameraModel(projection: .orthographic(height: 1))

        init(queue: (any MTLCommandQueue)? = nil, swapColors: Bool = false) throws {
            let device = try #require(MTLCreateSystemDefaultDevice())
            self.queue = try #require(queue ?? device.makeCommandQueue())
            let geometry = try MeshGeometry.plane(width: 4, height: 8)
            func mesh(_ color: SIMD4<Float>) throws -> MeshRenderer {
                try MeshRenderer(
                    device: device, colorPixelFormat: .rgba16Float, geometry: geometry,
                    material: .physicallyBased(try .init(
                        baseColorFactor: color, alphaMode: .opaque, shadingModel: .unlit
                    ))
                )
            }
            let red = SIMD4<Float>(1, 0, 0, 1), blue = SIMD4<Float>(0, 0, 1, 1)
            graph = SceneGraph(roots: [
                SceneNode(transform: .init(translation: SIMD3(-2, 0, -2)), mesh: try mesh(swapColors ? blue : red)),
                SceneNode(transform: .init(translation: SIMD3(2, 0, -2)), mesh: try mesh(swapColors ? red : blue)),
            ], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.01, far: 30))
            scene = try PathTracingScene(queue: self.queue, graph: graph, options: .init(acceleration: .software))
            tracer = try PathTracer(queue: self.queue, scene: scene)
        }

        func input(
            paused: Bool = false, invalidation: UInt64 = 0, pose: SceneTransform = .init(),
            lighting: PathTracer.Lighting? = nil
        ) -> PathViewContent {
            PathViewContent(
                tracer: tracer, camera: camera, pose: pose, settings: .init(maxBounces: 1), lighting: lighting,
                preview: .init(renderScales: 0.5...0.5, maximumSamples: 2), isPaused: paused,
                sceneRevision: tracer.sceneRevision, invalidationID: invalidation
            )
        }
    }

    @Test("The native path presentation scales linear pixels, pauses, and restarts for view or scene edits")
    func renderedLifecycle() throws {
        let fixture = try Fixture()
        let renderer = PathViewRenderer()
        let device = fixture.queue.device
        let display = try HDRDisplay(device: device, destinationPixelFormat: .rgba16Float)
        var settings = HDRDisplay.Settings(toneMapping: .none)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 5, height: 1, mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = [.renderTarget, .shaderRead]
        let target = try #require(device.makeTexture(descriptor: descriptor))

        func render(_ input: PathViewContent) throws -> (refined: Bool, pixels: [SIMD4<Float>]) {
            let command = try #require(fixture.queue.makeCommandBuffer())
            let refined = try renderer.encode(
                input, display: display, displaySettings: settings, destination: target, commandBuffer: command
            )
            command.commit()
            command.waitUntilCompleted()
            try #require(command.status == .completed, "\(command.error?.localizedDescription ?? "")")
            var values = [SIMD4<Float16>](repeating: .zero, count: 5)
            values.withUnsafeMutableBytes {
                target.getBytes($0.baseAddress!, bytesPerRow: 5 * 8,
                                from: MTLRegionMake2D(0, 0, 5, 1), mipmapLevel: 0)
            }
            return (refined, values.map(SIMD4<Float>.init))
        }

        let first = try render(fixture.input())
        #expect(first.refined && renderer.sampleCount == 1 && renderer.shouldRefine)
        #expect(first.pixels.first == SIMD4(1, 0, 0, 1) && first.pixels.last == SIMD4(0, 0, 1, 1))
        #expect(first.pixels[2] == SIMD4(0.5, 0, 0.5, 1))
        let paused = try render(fixture.input(paused: true))
        #expect(!paused.refined && paused.pixels == first.pixels && renderer.sampleCount == 1)
        #expect(try render(fixture.input()).refined && renderer.sampleCount == 2 && !renderer.shouldRefine)
        #expect(try !render(fixture.input()).refined)

        let moved = SceneTransform(translation: SIMD3(0.25, 0, 0))
        #expect(try render(fixture.input(pose: moved)).refined && renderer.sampleCount == 1)
        let lit = try PathTracer.Lighting(environment: SIMD3(0.1, 0.1, 0.1))
        #expect(try render(fixture.input(pose: moved, lighting: lit)).refined && renderer.sampleCount == 1)

        var graph = fixture.graph
        graph.roots[0].transform.translation.x -= 10
        #expect(try fixture.scene.updateTransforms(graph))
        let changed = try render(fixture.input(pose: moved, lighting: lit))
        #expect(changed.refined && renderer.sampleCount == 1)
        #expect(changed.pixels.first != first.pixels.first)

        settings.exposureStops = .nan
        #expect(throws: HDRDisplay.Failure.invalidExposure) { try render(fixture.input(invalidation: 1)) }
        settings.exposureStops = 0
        #expect(try render(fixture.input(invalidation: 1)).refined && renderer.sampleCount == 1)

        let replacement = try Fixture(queue: fixture.queue, swapColors: true)
        let replaced = try render(replacement.input())
        #expect(replaced.refined && renderer.sampleCount == 1 && replaced.pixels.first == SIMD4(0, 0, 1, 1))
    }


    @Test("The fixed trace reaches a completed sample gate before a GPU fade, even after reaching its cap")
    func stagedHandoff() throws {
        let fixture = try Fixture()
        let renderer = PathViewRenderer()
        let display = try HDRDisplay(device: fixture.queue.device, destinationPixelFormat: .rgba16Float)
        let start = ContinuousClock.now
        let configuration = FrostfieldPathTracingView.Transition(
            renderDelay: .milliseconds(10), minimumSamples: 2, fadeDuration: .milliseconds(100)
        )

        func render(
            at milliseconds: Int, paused: Bool = false, invalidate: UInt64 = 0, width: Int = 5, invalidDisplay: Bool = false
        ) throws -> (traced: Bool, pixel: SIMD4<Float>) {
            var input = fixture.input(paused: paused, invalidation: invalidate)
            input.transition = configuration
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: 1, mipmapped: false)
            descriptor.storageMode = .shared
            descriptor.usage = .renderTarget
            let target = try #require(fixture.queue.device.makeTexture(descriptor: descriptor))
            let command = try #require(fixture.queue.makeCommandBuffer())
            let traced = try renderer.encode(
                input, display: display, displaySettings: .init(exposureStops: invalidDisplay ? .nan : 0, toneMapping: .none),
                destination: target, commandBuffer: command, time: start + .milliseconds(milliseconds)
            )
            command.commit()
            command.waitUntilCompleted()
            try #require(command.status == .completed, "\(command.error?.localizedDescription ?? "")")
            var values = [SIMD4<Float16>](repeating: .zero, count: width)
            values.withUnsafeMutableBytes {
                target.getBytes($0.baseAddress!, bytesPerRow: width * 8,
                                from: MTLRegionMake2D(0, 0, width, 1), mipmapLevel: 0)
            }
            return (traced, SIMD4<Float>(values[1]))
        }

        // x=1 is solid red at width 5 but includes a blue contribution when
        // the two-pixel low-resolution red/blue trace is linearly scaled.
        let low = try render(at: 0)
        #expect(low.traced && renderer.sampleCount == 1 && renderer.shouldRefine)
        #expect(simd_distance(low.pixel, SIMD4(0.9, 0, 0.1, 1)) < 0.004)
        #expect(try render(at: 9).pixel == low.pixel && renderer.sampleCount == 2)

        // Failed first final command cannot publish samples or start the fade.
        #expect(throws: HDRDisplay.Failure.invalidExposure) { try render(at: 10, invalidDisplay: true) }
        #expect(try render(at: 11).pixel == low.pixel && renderer.sampleCount == 1)
        let reachingGate = try render(at: 12)
        #expect(reachingGate.traced && reachingGate.pixel == low.pixel && renderer.sampleCount == 2)
        let fadeStart = try render(at: 13)
        #expect(!fadeStart.traced && fadeStart.pixel == low.pixel && renderer.shouldRefine)

        let halfway = try render(at: 63)
        let final = SIMD4<Float>(1, 0, 0, 1)
        #expect(!halfway.traced && simd_distance(halfway.pixel, (low.pixel + final) / 2) < 0.004)
        #expect(try render(at: 1000, paused: true).pixel == halfway.pixel)
        #expect(try render(at: 2000).pixel == halfway.pixel)
        let finished = try render(at: 2050)
        #expect(!finished.traced && finished.pixel == final && !renderer.shouldRefine)

        let reset = try render(at: 2200, invalidate: 1)
        #expect(reset.traced && reset.pixel == low.pixel && renderer.sampleCount == 1 && renderer.shouldRefine)
        let resized = try render(at: 2201, invalidate: 1, width: 7)
        #expect(resized.traced && renderer.sampleCount == 1 && renderer.shouldRefine)
    }

    @Test("Transition settings reject an unreachable sample gate and cap large final allocations")
    func transitionValidation() throws {
        let preview = PathProgressivePreview.Configuration(maximumSamples: 2)
        #expect(throws: FrostfieldPathTracingView.TransitionFailure.configuration) {
            try PathPreviewHandoff(.init(minimumSamples: 3), preview: preview, width: 5, height: 1)
        }
        #expect(throws: FrostfieldPathTracingView.TransitionFailure.configuration) {
            try PathPreviewHandoff(.init(renderDelay: .milliseconds(-1)), preview: preview, width: 5, height: 1)
        }
        let handoff = try PathPreviewHandoff(.init(), preview: preview, width: 7_681, height: 4_321)
        #expect(handoff.finalConfiguration.renderScales.lowerBound == handoff.finalConfiguration.renderScales.upperBound)
        #expect(handoff.finalConfiguration.renderScales.upperBound < 1)
        #expect(handoff.lowConfiguration.renderScales.upperBound <= handoff.finalConfiguration.renderScales.upperBound)
    }
}
