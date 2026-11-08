import Foundation
import Frostfield
import Metal

/// Off-screen timing harness for FF-04. It renders the same triangle through
/// Frostfield's shipping `TriangleRenderer` (CPU encode time only, since it
/// does not report GPU time), the conventional mirror, and the Metal 4 path,
/// and prints one `RESULT` line per measured block.
///
/// The numbers describe the machine that runs it. On a Mac they are Mac
/// characterization, never iPhone or iPad evidence.
public enum Metal4Benchmark {
    public struct Settings: Sendable {
        public var width = 1206
        public var height = 2622
        public var warmupFrames = 100
        public var measuredFrames = 1000
        public var rounds = 3
        public init() {}
    }

    public static func run(settings: Settings = Settings()) async {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("FAIL no Metal device")
            return
        }
        let capabilities = RenderCapabilities(device: device)
        #if DEBUG
        let configuration = "Debug"
        #else
        let configuration = "Release"
        #endif
        // The validation layer wraps the device in a type whose name contains
        // "Debug"; timings taken through it are not representative.
        let validation = String(describing: type(of: device)).contains("Debug")
        let metalEnvironment = ProcessInfo.processInfo.environment
            .filter { $0.key.hasPrefix("MTL") || $0.key.hasPrefix("METAL") }
            .map { "\($0.key)=\($0.value)" }.sorted().joined(separator: ",")
        print("METAL_ENV \(metalEnvironment.isEmpty ? "none" : metalEnvironment) deviceType=\(type(of: device))")
        print("ENV device=\(capabilities.deviceName) os=\(ProcessInfo.processInfo.operatingSystemVersionString) metal4=\(capabilities.supportsMetal4) apple=\(capabilities.highestAppleFamily.map(String.init) ?? "none") configuration=\(configuration) apiValidation=\(validation) thermal=\(ProcessInfo.processInfo.thermalState.rawValue) lowPower=\(ProcessInfo.processInfo.isLowPowerModeEnabled)")
        print("SETTINGS size=\(settings.width)x\(settings.height) format=bgra8Unorm storage=private warmup=\(settings.warmupFrames) measured=\(settings.measuredFrames) rounds=\(settings.rounds) framesInFlight=2")

        let library: any MTLLibrary
        do {
            library = try TriangleFunctions.makeLibrary(device: device)
        } catch {
            print("FAIL \(error)")
            return
        }
        let backend = TriangleBackend.select(device: device)
        print("BACKEND selected=\(backend.rawValue)")

        for round in 1...settings.rounds {
            // Alternate the order so a warming GPU or cache favors neither path.
            var paths: [String] = ["frostfield", "conventional", "metal4"]
            if round.isMultiple(of: 2) { paths.reverse() }
            for path in paths {
                do {
                    try await measure(path: path, round: round, device: device, library: library, settings: settings)
                } catch {
                    print("FAIL round=\(round) path=\(path) \(error)")
                }
            }
        }
        print("DONE")
    }

    private static func measure(
        path: String, round: Int, device: any MTLDevice, library: any MTLLibrary, settings: Settings
    ) async throws {
        let (completions, report) = AsyncStream.makeStream(of: ExperimentFrameCompletion.self)
        var iterator = completions.makeAsyncIterator()
        let targets = try (0..<2).map { _ in try makeTarget(device: device, settings: settings) }
        let allocatedBefore = device.currentAllocatedSize

        let created = ContinuousClock.now
        let draw: (Float) -> ExperimentFrameOutcome
        var allocatorBytes: () -> UInt64 = { 0 }
        var keepAlive: AnyObject?
        switch path {
        case "frostfield":
            let renderer = try TriangleRenderer(device: device, colorPixelFormat: .bgra8Unorm, maxFramesInFlight: 2) { completion in
                // Shipping renderer: no GPU interval available; report NaN.
                report.yield(ExperimentFrameCompletion(frame: completion.frame, succeeded: completion.succeeded, gpuSeconds: .nan))
            }
            var next = 0
            draw = { rotation in
                let texture = targets[next % targets.count]
                next += 1
                let outcome = renderer.drawFrame(rotation: rotation) {
                    let pass = MTLRenderPassDescriptor()
                    pass.colorAttachments[0].texture = texture
                    pass.colorAttachments[0].loadAction = .clear
                    pass.colorAttachments[0].storeAction = .store
                    pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
                    return FrameTarget(passDescriptor: pass, drawable: nil)
                }
                return switch outcome {
                case .submitted(let frame): .submitted(frame: frame)
                case .skippedNoSlot: .skippedNoSlot
                case .skippedNoTarget: .skippedNoTarget
                case .encodingFailed: .encodingFailed
                }
            }
            keepAlive = renderer
        case "conventional":
            let renderer = try ConventionalTriangle(device: device, library: library, colorPixelFormat: .bgra8Unorm) { report.yield($0) }
            draw = { rotation in renderer.drawFrame(rotation: rotation) { targets[$0] } }
            keepAlive = renderer
        case "metal4":
            #if targetEnvironment(simulator)
            throw ExperimentError.metal4Unsupported
            #else
            let renderer = try Metal4Triangle(device: device, library: library, colorPixelFormat: .bgra8Unorm) { report.yield($0) }
            for texture in targets { renderer.makeResident(texture) }
            draw = { rotation in renderer.drawFrame(rotation: rotation) { targets[$0] } }
            allocatorBytes = { renderer.allocatorBytes }
            keepAlive = renderer
            #endif
        default:
            preconditionFailure("unknown path \(path)")
        }
        let creation = ContinuousClock.now - created
        let allocatedAfterCreate = device.currentAllocatedSize

        // Paced: one frame at a time, as in requested drawing.
        var cpu: [Double] = []
        var gpu: [Double] = []
        var failures = 0
        for index in 0..<(settings.warmupFrames + settings.measuredFrames) {
            let start = ContinuousClock.now
            let outcome = draw(Float(index) * 0.01)
            let elapsed = ContinuousClock.now - start
            guard case .submitted = outcome else {
                print("FAIL round=\(round) path=\(path) paced outcome=\(outcome)")
                return
            }
            guard let completion = await iterator.next() else { return }
            if !completion.succeeded { failures += 1 }
            if index >= settings.warmupFrames {
                cpu.append(elapsed.microseconds)
                gpu.append(completion.gpuSeconds * 1_000_000)
            }
        }

        // Throughput: keep both slots busy; skipped requests wait for a completion.
        var throughputCPU: [Double] = []
        var skips = 0
        var inFlight = 0
        let throughputStart = ContinuousClock.now
        var submitted = 0
        while submitted < settings.measuredFrames {
            let start = ContinuousClock.now
            let outcome = draw(Float(submitted) * 0.01)
            let elapsed = ContinuousClock.now - start
            switch outcome {
            case .submitted:
                submitted += 1
                inFlight += 1
                throughputCPU.append(elapsed.microseconds)
            case .skippedNoSlot:
                skips += 1
                guard let completion = await iterator.next() else { return }
                if !completion.succeeded { failures += 1 }
                inFlight -= 1
            case .skippedNoTarget, .encodingFailed:
                print("FAIL round=\(round) path=\(path) throughput outcome=\(outcome)")
                return
            }
        }
        while inFlight > 0, let completion = await iterator.next() {
            if !completion.succeeded { failures += 1 }
            inFlight -= 1
        }
        let throughputSeconds = (ContinuousClock.now - throughputStart).seconds
        let allocatedPeak = device.currentAllocatedSize

        let c = Percentiles(cpu), g = Percentiles(gpu), t = Percentiles(throughputCPU)
        print(String(
            format: "RESULT round=%d path=%@ create_ms=%.3f paced_cpu_us p50=%.1f p95=%.1f p99=%.1f paced_gpu_us p50=%.1f p95=%.1f p99=%.1f throughput_cpu_us p50=%.1f p95=%.1f p99=%.1f throughput_fps=%.0f slot_skips=%d failures=%d alloc_create_kib=%lld alloc_peak_kib=%lld allocator_kib=%llu",
            round, path, creation.seconds * 1000,
            c.p50, c.p95, c.p99, g.p50, g.p95, g.p99, t.p50, t.p95, t.p99,
            Double(settings.measuredFrames) / throughputSeconds, skips, failures,
            Int64(allocatedAfterCreate) / 1024 - Int64(allocatedBefore) / 1024,
            Int64(allocatedPeak) / 1024 - Int64(allocatedBefore) / 1024,
            allocatorBytes() / 1024
        ))
        withExtendedLifetime(keepAlive) {}
    }

    private static func makeTarget(device: any MTLDevice, settings: Settings) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: settings.width, height: settings.height, mipmapped: false
        )
        descriptor.usage = .renderTarget
        descriptor.storageMode = .private
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw ExperimentError.bufferAllocationFailed }
        return texture
    }
}

/// Nearest-rank percentiles of a sample.
struct Percentiles {
    let p50: Double, p95: Double, p99: Double

    init(_ values: [Double]) {
        let sorted = values.filter { !$0.isNaN }.sorted()
        func rank(_ p: Double) -> Double {
            guard !sorted.isEmpty else { return .nan }
            let index = Int((p * Double(sorted.count)).rounded(.up)) - 1
            return sorted[min(max(index, 0), sorted.count - 1)]
        }
        p50 = rank(0.50)
        p95 = rank(0.95)
        p99 = rank(0.99)
    }
}

extension Duration {
    var seconds: Double {
        let (s, attoseconds) = components
        return Double(s) + Double(attoseconds) / 1e18
    }
    var microseconds: Double { seconds * 1_000_000 }
}
