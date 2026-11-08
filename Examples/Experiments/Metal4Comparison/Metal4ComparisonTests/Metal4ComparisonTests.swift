import Foundation
import Frostfield
import Metal
import Testing

// The shared experiment sources (Shared/) compile into this test bundle, so
// their internal types need no import.

private let systemDevice = MTLCreateSystemDefaultDevice()
private let deviceSupportsMetal4 = systemDevice.map { TriangleBackend.select(device: $0) == .metal4 } ?? false

/// The runtime decision between the two command paths.
struct BackendSelectionTests {
    @Test(arguments: [true, false])
    func `Metal 4 is chosen only when the device reports the family`(reportsMetal4: Bool) {
        let backend = TriangleBackend.select { family in
            #if targetEnvironment(simulator)
            return reportsMetal4
            #else
            return family == .metal4 ? reportsMetal4 : true
            #endif
        }
        #if targetEnvironment(simulator)
        // The simulator SDK has no Metal 4 API, whatever a probe reports.
        #expect(backend == .conventional)
        #else
        #expect(backend == (reportsMetal4 ? .metal4 : .conventional))
        #endif
    }

    @Test(.enabled(if: systemDevice != nil))
    func `The selection agrees with Frostfield's capability survey`() throws {
        let device = try #require(systemDevice)
        let survey = RenderCapabilities(device: device)
        #expect((TriangleBackend.select(device: device) == .metal4) == survey.supportsMetal4)
    }
}

/// Renders the same frames through Frostfield's shipping renderer, the
/// conventional mirror, and the Metal 4 path, then compares every byte.
@Suite(.enabled(if: systemDevice != nil))
struct OutputEquivalenceTests {
    let device: any MTLDevice
    let library: any MTLLibrary

    init() throws {
        device = try #require(systemDevice)
        library = try TriangleFunctions.makeLibrary(device: device)
    }

    @Test(arguments: [
        (64, 64, Float(0)),
        (200, 100, Float.pi / 3),
        (90, 160, Float(2)),
        (1206, 2622, Float(0.7)),
    ])
    func `Conventional mirror matches the shipping renderer`(width: Int, height: Int, rotation: Float) async throws {
        let reference = try await renderShipping(width: width, height: height, rotation: rotation)
        let mirror = try await renderConventional(width: width, height: height, rotation: rotation)
        let difference = PixelDifference(reference, mirror)
        #expect(difference.differingPixels == 0, "\(difference)")
        #expect(reference.containsAllThreeCorners)
    }

    @Test(.enabled(if: deviceSupportsMetal4), arguments: [
        (64, 64, Float(0)),
        (200, 100, Float.pi / 3),
        (90, 160, Float(2)),
        (1206, 2622, Float(0.7)),
    ])
    func `Metal 4 output matches conventional Metal byte for byte`(width: Int, height: Int, rotation: Float) async throws {
        #if !targetEnvironment(simulator)
        let conventional = try await renderConventional(width: width, height: height, rotation: rotation)
        let metal4 = try await renderMetal4(width: width, height: height, rotation: rotation)
        let difference = PixelDifference(conventional, metal4)
        #expect(difference.differingPixels == 0, "\(difference)")
        #expect(metal4.containsAllThreeCorners)
        #endif
    }

    // MARK: Renders

    private func renderShipping(width: Int, height: Int, rotation: Float) async throws -> Readback {
        let target = try SharedTarget(device: device, width: width, height: height)
        let (stream, report) = AsyncStream.makeStream(of: FrameCompletion.self)
        let renderer = try TriangleRenderer(device: device, colorPixelFormat: .bgra8Unorm) { report.yield($0) }
        let outcome = renderer.drawFrame(rotation: rotation) {
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = target.texture
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            return FrameTarget(passDescriptor: pass, drawable: nil)
        }
        #expect(outcome == .submitted(frame: 1))
        var iterator = stream.makeAsyncIterator()
        #expect(await iterator.next()?.succeeded == true)
        return target.readback()
    }

    private func renderConventional(width: Int, height: Int, rotation: Float) async throws -> Readback {
        let target = try SharedTarget(device: device, width: width, height: height)
        let (stream, report) = AsyncStream.makeStream(of: ExperimentFrameCompletion.self)
        let renderer = try ConventionalTriangle(device: device, library: library, colorPixelFormat: .bgra8Unorm) { report.yield($0) }
        #expect(renderer.drawFrame(rotation: rotation) { _ in target.texture } == .submitted(frame: 1))
        var iterator = stream.makeAsyncIterator()
        #expect(await iterator.next()?.succeeded == true)
        return target.readback()
    }

    #if !targetEnvironment(simulator)
    private func renderMetal4(width: Int, height: Int, rotation: Float) async throws -> Readback {
        let target = try SharedTarget(device: device, width: width, height: height)
        let (stream, report) = AsyncStream.makeStream(of: ExperimentFrameCompletion.self)
        let renderer = try Metal4Triangle(device: device, library: library, colorPixelFormat: .bgra8Unorm) { report.yield($0) }
        renderer.makeResident(target.texture)
        #expect(renderer.drawFrame(rotation: rotation) { _ in target.texture } == .submitted(frame: 1))
        var iterator = stream.makeAsyncIterator()
        let completion = try #require(await iterator.next())
        #expect(completion.succeeded)
        #expect(completion.gpuSeconds > 0)
        return target.readback()
    }
    #endif
}

#if !targetEnvironment(simulator)
/// Slot, allocator, residency, and teardown behavior of the Metal 4 path.
@Suite(.enabled(if: deviceSupportsMetal4))
struct Metal4LifetimeTests {
    let device: any MTLDevice
    let library: any MTLLibrary
    let completions: AsyncStream<ExperimentFrameCompletion>
    let report: AsyncStream<ExperimentFrameCompletion>.Continuation

    init() throws {
        device = try #require(systemDevice)
        library = try TriangleFunctions.makeLibrary(device: device)
        (completions, report) = AsyncStream.makeStream(of: ExperimentFrameCompletion.self)
    }

    private func makeRenderer(maxFramesInFlight: Int = 2) throws -> Metal4Triangle {
        let report = self.report
        return try Metal4Triangle(
            device: device, library: library, colorPixelFormat: .bgra8Unorm,
            maxFramesInFlight: maxFramesInFlight, onFrameCompleted: { report.yield($0) }
        )
    }

    @Test func `A frame requested while every slot is busy is skipped without waiting`() async throws {
        let renderer = try makeRenderer(maxFramesInFlight: 1)
        let target = try SharedTarget(device: device, width: 16, height: 16)
        renderer.makeResident(target.texture)

        var nested: ExperimentFrameOutcome?
        let outer = renderer.drawFrame(rotation: 0) { _ in
            nested = renderer.drawFrame(rotation: 0) { _ in target.texture }
            return target.texture
        }
        #expect(nested == .skippedNoSlot)
        #expect(outer == .submitted(frame: 1))
        var iterator = completions.makeAsyncIterator()
        #expect(await iterator.next()?.succeeded == true)
        #expect(renderer.statistics.skippedNoSlot == 1)
    }

    @Test func `A missing target releases its slot and allocator unused`() async throws {
        let renderer = try makeRenderer(maxFramesInFlight: 1)
        let target = try SharedTarget(device: device, width: 16, height: 16)
        renderer.makeResident(target.texture)

        #expect(renderer.drawFrame(rotation: 0) { _ in nil } == .skippedNoTarget)
        #expect(renderer.drawFrame(rotation: 0) { _ in target.texture } == .submitted(frame: 1))
        var iterator = completions.makeAsyncIterator()
        #expect(await iterator.next()?.succeeded == true)
        #expect(renderer.statistics.skippedNoTarget == 1)
    }

    /// Two slots, two allocators, two targets, 300 back-to-back requests.
    /// Every allocator reset follows the completion of the frame that last
    /// used it (the pool reports reuse only of completed slots), the bound is
    /// never exceeded, and with API validation on every frame completes
    /// without an error.
    @Test func `Allocators are reused only after their frames complete`() async throws {
        let renderer = try makeRenderer(maxFramesInFlight: 2)
        let targets = try (0..<2).map { _ in try SharedTarget(device: device, width: 64, height: 64) }
        for target in targets { renderer.makeResident(target.texture) }
        var iterator = completions.makeAsyncIterator()

        var submitted = 0
        var completed = 0
        for index in 0..<300 {
            switch renderer.drawFrame(rotation: Float(index) * 0.02, target: { targets[$0].texture }) {
            case .submitted: submitted += 1
            case .skippedNoSlot:
                // Let one frame finish, then continue; no CPU wait inside the renderer.
                let completion = try #require(await iterator.next())
                #expect(completion.succeeded)
                completed += 1
            case let other: Issue.record("Unexpected outcome \(other)")
            }
            #expect(renderer.statistics.inFlight <= 2)
        }
        while completed < submitted {
            let completion = try #require(await iterator.next())
            #expect(completion.succeeded)
            completed += 1
        }
        let statistics = renderer.statistics
        #expect(statistics.submitted == submitted)
        #expect(statistics.completed == submitted)
        #expect(statistics.failed == 0)
        #expect(statistics.slotReuses == submitted - 2)
        #expect(renderer.allocatorBytes > 0)
    }

    /// Resizing means new targets. The old ones may be evicted from residency
    /// only after the frames that used them complete.
    @Test func `A resized target renders with the new aspect after the old one is evicted`() async throws {
        let renderer = try makeRenderer()
        var iterator = completions.makeAsyncIterator()

        let square = try SharedTarget(device: device, width: 64, height: 64)
        renderer.makeResident(square.texture)
        #expect(renderer.drawFrame(rotation: 0) { _ in square.texture } == .submitted(frame: 1))
        #expect(await iterator.next()?.succeeded == true)
        #expect(!square.readback().pixel(clipX: 0.5, clipY: -0.3).isClear)

        renderer.evictTargets()
        let wide = try SharedTarget(device: device, width: 128, height: 64)
        renderer.makeResident(wide.texture)
        #expect(renderer.drawFrame(rotation: 0) { _ in wide.texture } == .submitted(frame: 2))
        #expect(await iterator.next()?.succeeded == true)
        // The 0.5 horizontal scale moves this point outside the triangle.
        let readback = wide.readback()
        #expect(readback.pixel(clipX: 0.5, clipY: -0.3).isClear)
        #expect(!readback.pixel(clipX: 0.2, clipY: -0.3).isClear)
    }
}
#endif

/// With `MTL_DEBUG_LAYER=1` (set by the `Metal4Experiment` scheme's test
/// action) the device must be Metal's validation wrapper.
@Test(.enabled(if: systemDevice != nil && ProcessInfo.processInfo.environment["MTL_DEBUG_LAYER"] == "1"))
func `API validation wraps the device when requested`() throws {
    let device = try #require(systemDevice)
    #expect(String(describing: type(of: device)).contains("Debug"), "device type: \(type(of: device))")
}

// MARK: - Helpers

/// A CPU-readable BGRA8 render target.
private struct SharedTarget {
    let texture: any MTLTexture

    init(device: any MTLDevice, width: Int, height: Int) throws {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false
        )
        descriptor.usage = .renderTarget
        descriptor.storageMode = .shared
        texture = try #require(device.makeTexture(descriptor: descriptor))
    }

    func readback() -> Readback {
        let bytesPerRow = texture.width * 4
        var bytes = [UInt8](repeating: 0, count: bytesPerRow * texture.height)
        texture.getBytes(&bytes, bytesPerRow: bytesPerRow, from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0)
        return Readback(width: texture.width, height: texture.height, bytes: bytes)
    }
}

private struct Readback {
    struct Pixel {
        var blue: UInt8, green: UInt8, red: UInt8, alpha: UInt8
        var isClear: Bool { red < 8 && green < 8 && blue < 8 }
    }

    let width: Int, height: Int
    let bytes: [UInt8]

    func pixel(clipX: Float, clipY: Float) -> Pixel {
        let x = min(width - 1, Int((clipX + 1) / 2 * Float(width)))
        let y = min(height - 1, Int((1 - clipY) / 2 * Float(height)))
        let i = (y * width + x) * 4
        return Pixel(blue: bytes[i], green: bytes[i + 1], red: bytes[i + 2], alpha: bytes[i + 3])
    }

    /// Strongly red, green, and blue pixels all occur, so a comparison is not
    /// between two blank images.
    var containsAllThreeCorners: Bool {
        var red = false, green = false, blue = false
        for i in stride(from: 0, to: bytes.count, by: 4) {
            let (b, g, r) = (bytes[i], bytes[i + 1], bytes[i + 2])
            if r > 240 && g < 15 && b < 15 { red = true }
            if g > 240 && r < 15 && b < 15 { green = true }
            if b > 240 && r < 15 && g < 15 { blue = true }
        }
        return red && green && blue
    }
}

private struct PixelDifference: CustomStringConvertible {
    var differingPixels = 0
    var maxChannelDelta = 0

    init(_ a: Readback, _ b: Readback) {
        precondition(a.width == b.width && a.height == b.height)
        for pixel in 0..<(a.width * a.height) {
            var differs = false
            for channel in 0..<4 {
                let delta = abs(Int(a.bytes[pixel * 4 + channel]) - Int(b.bytes[pixel * 4 + channel]))
                if delta > 0 { differs = true }
                maxChannelDelta = max(maxChannelDelta, delta)
            }
            if differs { differingPixels += 1 }
        }
    }

    var description: String { "\(differingPixels) differing pixels, max channel delta \(maxChannelDelta)" }
}
