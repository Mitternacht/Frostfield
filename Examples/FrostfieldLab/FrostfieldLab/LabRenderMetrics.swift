import Foundation
import QuartzCore
#if os(macOS)
import Darwin
#endif

/// Opt-in lab instrumentation. Keep the collection window bounded and log once per run.
@MainActor
final class LabRenderMetrics {
    private static let warmupFrames = 60
    private static let measuredFrames = 240

    private struct Sample {
        var cpu: Double?
        var gpu: Double?
        var presented: Double?
    }

    private struct Activity: Equatable {
        let active: Bool
        let continuous: Bool
    }

    private struct Histogram {
        private var bins = Array(repeating: 0, count: 1_001)
        private var count = 0

        mutating func add(_ seconds: Double) {
            guard seconds.isFinite, seconds >= 0 else { return }
            let index = Int(min(seconds * 4_000, 1_000))
            bins[index] += 1
            count += 1
        }

        var summary: String {
            guard count > 0 else { return "n/a" }
            func percentile(_ fraction: Double) -> Double {
                let rank = Int((Double(count) * fraction).rounded(.up))
                var total = 0
                for index in bins.indices {
                    total += bins[index]
                    if total >= rank { return min(Double(index + 1) * 0.25, 250) }
                }
                return 250
            }
            return String(format: "n:%d,p50:%.3f,p95:%.3f,p99:%.3f,overflow:%d", Int32(count),
                          percentile(0.5), percentile(0.95), percentile(0.99), Int32(bins[1_000]))
        }
    }

    private final class Sustained {
        var window: Range<Double>?
        var firstFrame: Int?
        var lastFrame: Int?
        var cpu = Histogram()
        var gpu = Histogram()
        var presentationIntervals = Histogram()
        var lastPresentation: Double?
        var presentations = 0

        func contains(_ frame: Int) -> Bool {
            guard let firstFrame, let lastFrame else { return false }
            return firstFrame...lastFrame ~= frame
        }
    }

    private let runID = UUID().uuidString
    private let sustained: Sustained?
    private var samples = Array(repeating: Sample(), count: LabRenderMetrics.measuredFrames)
    private var activity: Activity?
    private var idleTask: Task<Void, Never>?
    private var reportTask: Task<Void, Never>?
    private var submitted = 0
    private var completed = 0
    private var missingDrawables = 0
    private var suspensions = 0
    private var sizeChanges = 0
    private var firstSize: SIMD2<Int>?
    private var lastSize: SIMD2<Int>?
    private var thermalAtStart: String?
    #if os(macOS)
    private var residentAtStart: UInt64?
    #endif

    init(arguments: [String]) {
        let fixtureFlags = ["--mesh", "--lit", "--particles", "--grade", "--grade-bypass", "--continuous"]
        let workload = fixtureFlags.filter(arguments.contains).joined(separator: ",")
        sustained = arguments.contains("--render-metrics-sustained") ? Sustained() : nil
        let window = sustained == nil
            ? "mode=short warmup_frames=\(Self.warmupFrames) measured_frames=\(Self.measuredFrames)"
            : "mode=sustained warmup_seconds=30 measured_seconds=600 histogram_bin_ms=0.25 histogram_limit_ms=250"
        write("event=start workload=\(workload.isEmpty ? "default" : workload) \(window)")
    }

    func wantsSample(_ frame: Int) -> Bool {
        if let sustained {
            return sustained.window?.contains(CACurrentMediaTime()) == true
        }
        return sampleIndex(frame) != nil
    }

    func updateActivity(active: Bool, continuous: Bool) {
        let next = Activity(active: active, continuous: continuous)
        guard next != activity else { return }
        if activity?.active == true, !active, wantsSample(submitted + 1) {
            suspensions += 1
        }
        activity = next
        restartIdleProbe()
    }

    func missedDrawable() {
        if wantsSample(submitted + 1) { missingDrawables += 1 }
    }

    func recordSubmission(frame: Int, cpuSeconds: Double?, width: Int, height: Int) {
        submitted += 1
        if let sustained {
            if sustained.window == nil, activity?.continuous == true {
                let start = CACurrentMediaTime() + 30
                sustained.window = start..<(start + 600)
                reportTask = Task { @MainActor [weak self] in
                    do { try await Task.sleep(for: .milliseconds(630_500)) } catch { return }
                    self?.report()
                }
            }
            if let cpuSeconds {
                if sustained.firstFrame == nil { sustained.firstFrame = frame }
                sustained.lastFrame = frame
                sustained.cpu.add(cpuSeconds)
                recordSize(width: width, height: height)
            }
        } else if let index = sampleIndex(frame) {
            samples[index].cpu = cpuSeconds
            recordSize(width: width, height: height)
        }
        if activity?.continuous == false { restartIdleProbe() }
    }

    func recordCompletion(frame: Int, succeeded: Bool, gpuSeconds: Double?) {
        completed += 1
        if let sustained {
            if succeeded, let gpuSeconds, sustained.contains(frame) { sustained.gpu.add(gpuSeconds) }
            return
        }
        guard let index = sampleIndex(frame) else { return }
        if succeeded { samples[index].gpu = gpuSeconds }
        if index == samples.count - 1 {
            // A drawable's presentation callback may follow its command buffer completion.
            reportTask = Task { @MainActor [weak self] in
                do { try await Task.sleep(for: .milliseconds(500)) } catch { return }
                self?.report()
            }
        }
    }

    func recordPresentation(frame: Int, time: Double) {
        guard time.isFinite, time > 0 else { return }
        if let sustained {
            guard sustained.contains(frame) else { return }
            sustained.presentations += 1
            if let previous = sustained.lastPresentation {
                guard time > previous else { return }
                sustained.presentationIntervals.add(time - previous)
            }
            sustained.lastPresentation = time
        } else if let index = sampleIndex(frame) {
            samples[index].presented = time
        }
    }

    private func recordSize(width: Int, height: Int) {
        let size = SIMD2(width, height)
        if firstSize == nil {
            firstSize = size
            thermalAtStart = Self.thermalState
            #if os(macOS)
            residentAtStart = Self.residentBytes()
            #endif
        }
        if let lastSize, lastSize != size { sizeChanges += 1 }
        lastSize = size
    }

    private func sampleIndex(_ frame: Int) -> Int? {
        let index = frame - Self.warmupFrames - 1
        return samples.indices.contains(index) ? index : nil
    }

    private func restartIdleProbe() {
        idleTask?.cancel()
        idleTask = nil
        guard let activity, !activity.continuous else { return }
        let submittedAtStart = submitted
        let completedAtStart = completed
        idleTask = Task { @MainActor [weak self] in
            do { try await Task.sleep(for: .seconds(2)) } catch { return }
            guard let self else { return }
            let mode = activity.active ? "on-demand" : "inactive"
            write("event=idle mode=\(mode) seconds=2 new_submissions=\(submitted - submittedAtStart) completions=\(completed - completedAtStart)")
            idleTask = nil
        }
    }

    private func report() {
        let presentationCount: Int
        let timing: String
        if let sustained {
            presentationCount = sustained.presentations
            timing = "cpu_submit=\(sustained.cpu.summary) gpu_buffer=\(sustained.gpu.summary) presentation_interval=\(sustained.presentationIntervals.summary)"
        } else {
            let presentations = samples.compactMap(\.presented).sorted()
            let intervals = zip(presentations, presentations.dropFirst()).compactMap { earlier, later in
                later > earlier ? later - earlier : nil
            }
            presentationCount = presentations.count
            timing = "cpu_submit=\(Self.stats(samples.compactMap(\.cpu))) gpu_buffer=\(Self.stats(samples.compactMap(\.gpu))) presentation_interval=\(Self.stats(intervals))"
        }
        let dimensions = "\(firstSize?.x ?? 0)x\(firstSize?.y ?? 0)->\(lastSize?.x ?? 0)x\(lastSize?.y ?? 0)"
        write("event=summary resolution=\(dimensions) resize_count=\(sizeChanges) suspensions=\(suspensions) missed_drawables=\(missingDrawables) presented=\(presentationCount) thermal=\(thermalAtStart ?? "unknown")->\(Self.thermalState)")
        write("event=timing unit=ms \(timing)")
        #if os(macOS)
        let start = residentAtStart.map(String.init) ?? "n/a"
        let end = Self.residentBytes().map(String.init) ?? "n/a"
        write("event=resident_memory unit=bytes start=\(start) end=\(end)")
        #endif
        reportTask = nil
    }

    private func write(_ message: String) {
        FileHandle.standardError.write(Data("FrostfieldMetrics run=\(runID) \(message)\n".utf8))
    }

    private static func stats(_ seconds: [Double]) -> String {
        let sorted = seconds.filter { $0.isFinite && $0 >= 0 }.sorted()
        guard !sorted.isEmpty else { return "n/a" }
        func percentile(_ fraction: Double) -> Double {
            let index = Int((Double(sorted.count) * fraction).rounded(.up)) - 1
            return sorted[index] * 1_000
        }
        return String(format: "n:%d,p50:%.3f,p95:%.3f,p99:%.3f", Int32(sorted.count),
                      percentile(0.5), percentile(0.95), percentile(0.99))
    }

    private static var thermalState: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    #if os(macOS)
    private static func residentBytes() -> UInt64? {
        var info = mach_task_basic_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<integer_t>.size)
        let capacity = Int(count)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: capacity) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return status == KERN_SUCCESS ? UInt64(info.resident_size) : nil
    }
    #endif
}
