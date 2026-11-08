import Foundation
import Metal
import Synchronization

/// Why an experiment renderer could not be created.
enum ExperimentError: Error, Equatable, CustomStringConvertible {
    /// The device does not report `MTLGPUFamily.metal4`, or this build has no
    /// Metal 4 API (the iOS Simulator SDK declares none).
    case metal4Unsupported
    case queueUnavailable
    case commandBufferUnavailable
    case allocatorUnavailable
    case bufferAllocationFailed
    case shaderLibraryUnavailable(reason: String)
    case shaderFunctionMissing(name: String)
    case pipelineCreationFailed(reason: String)
    case argumentTableUnavailable(reason: String)
    case residencySetUnavailable(reason: String)

    var description: String {
        switch self {
        case .metal4Unsupported: "The device or build does not support the Metal 4 command path."
        case .queueUnavailable: "The device could not create a command queue."
        case .commandBufferUnavailable: "The device could not create a command buffer."
        case .allocatorUnavailable: "The device could not create a command allocator."
        case .bufferAllocationFailed: "The device could not allocate a per-frame buffer."
        case .shaderLibraryUnavailable(let reason): "The experiment's shader library is unavailable: \(reason)"
        case .shaderFunctionMissing(let name): "The shader library has no function named \(name)."
        case .pipelineCreationFailed(let reason): "The render pipeline could not be created: \(reason)"
        case .argumentTableUnavailable(let reason): "The argument table could not be created: \(reason)"
        case .residencySetUnavailable(let reason): "The residency set could not be created: \(reason)"
        }
    }
}

/// What happened to one frame request in either experiment renderer.
enum ExperimentFrameOutcome: Equatable, Sendable {
    case submitted(frame: UInt64)
    case skippedNoSlot
    case skippedNoTarget
    case encodingFailed
}

/// The GPU's report for one submitted frame, with the GPU execution interval
/// Metal reported for it (`gpuEndTime - gpuStartTime`, host seconds).
struct ExperimentFrameCompletion: Equatable, Sendable {
    var frame: UInt64
    var succeeded: Bool
    var gpuSeconds: Double
}

/// Counters for one experiment renderer.
struct ExperimentStatistics: Equatable, Sendable {
    var submitted = 0
    var completed = 0
    var failed = 0
    var skippedNoSlot = 0
    var skippedNoTarget = 0
    var encodingFailures = 0
    /// Slot acquisitions that found the slot's previous frame already
    /// completed, so its allocator or buffer could be reused.
    var slotReuses = 0
    var inFlight: Int { submitted - completed }
}

/// A fixed set of per-frame slots shared by the encoding thread and Metal's
/// completion thread, mirroring Frostfield's internal `FrameSlots`.
///
/// A slot is handed out only after the GPU completed the last frame that used
/// it, so the per-slot uniform buffer (and, on the Metal 4 path, the per-slot
/// command allocator) is never reused while the GPU may still read it.
/// Acquisition never waits.
final class SlotPool: Sendable {
    private struct State {
        var busy: [Bool]
        var everUsed: [Bool]
        var next = 0
        var statistics = ExperimentStatistics()
    }

    let capacity: Int
    private let state: Mutex<State>

    init(capacity: Int) {
        precondition(capacity >= 1)
        self.capacity = capacity
        self.state = Mutex(State(
            busy: Array(repeating: false, count: capacity),
            everUsed: Array(repeating: false, count: capacity)
        ))
    }

    var statistics: ExperimentStatistics { state.withLock { $0.statistics } }

    /// A free slot in round-robin order, or `nil` without waiting.
    func acquire() -> Int? {
        state.withLock { state in
            for offset in 0..<capacity {
                let slot = (state.next + offset) % capacity
                if !state.busy[slot] {
                    state.busy[slot] = true
                    state.next = (slot + 1) % capacity
                    if state.everUsed[slot] { state.statistics.slotReuses += 1 }
                    state.everUsed[slot] = true
                    return slot
                }
            }
            state.statistics.skippedNoSlot += 1
            return nil
        }
    }

    /// Whether `slot` is currently acquired or in flight.
    func isBusy(_ slot: Int) -> Bool { state.withLock { $0.busy[slot] } }

    func releaseUnused(_ slot: Int, missingTarget: Bool) {
        state.withLock { state in
            precondition(state.busy[slot])
            state.busy[slot] = false
            if missingTarget {
                state.statistics.skippedNoTarget += 1
            } else {
                state.statistics.encodingFailures += 1
            }
        }
    }

    func markSubmitted(_ slot: Int) -> UInt64 {
        state.withLock { state in
            precondition(state.busy[slot])
            state.statistics.submitted += 1
            return UInt64(state.statistics.submitted)
        }
    }

    func complete(_ slot: Int, failed: Bool) {
        state.withLock { state in
            precondition(state.busy[slot], "Completing slot \(slot), which is not in flight")
            state.busy[slot] = false
            state.statistics.completed += 1
            if failed { state.statistics.failed += 1 }
        }
    }
}

/// The 16-byte uniform record of `ff_triangle_vertex`: `float2 scale` at
/// offset 0, `float rotation` at 8, padding at 12 (Frostfield's
/// `FFTriangleUniforms`). The pixel-identity tests against Frostfield's
/// `TriangleRenderer` confirm this layout reaches the shader unchanged.
func triangleUniforms(width: Int, height: Int, rotation: Float) -> SIMD4<Float> {
    let scale: SIMD2<Float> = width >= height
        ? SIMD2(Float(height) / Float(width), 1)
        : SIMD2(1, Float(width) / Float(height))
    return SIMD4(scale.x, scale.y, rotation, 0)
}

/// Names of the shared shader functions (from Frostfield's Triangle.metal).
enum TriangleFunctions {
    static let vertex = "ff_triangle_vertex"
    static let fragment = "ff_triangle_fragment"
    static let uniformsBufferIndex = 0

    /// The experiment's compiled library: Frostfield's Triangle.metal,
    /// included by `Shared/Shaders/Triangle.metal` and compiled by Xcode into
    /// the `default.metallib` of whichever target (test bundle or tool)
    /// contains this file.
    static func makeLibrary(device: any MTLDevice) throws(ExperimentError) -> any MTLLibrary {
        do {
            return try device.makeDefaultLibrary(bundle: Bundle(for: SlotPool.self))
        } catch {
            throw .shaderLibraryUnavailable(reason: String(describing: error))
        }
    }
}

/// Which command path to use for a device, decided at run time.
enum TriangleBackend: String, Sendable, CaseIterable {
    case conventional
    case metal4

    /// Metal 4 requires both the API in this build and the device's
    /// `MTLGPUFamily.metal4`. The OS version alone does not decide it: the
    /// iOS 26 floor runs on GPUs without the family, and the iOS Simulator SDK
    /// declares no Metal 4 API at all.
    static func select(supportsFamily: (MTLGPUFamily) -> Bool) -> TriangleBackend {
        #if targetEnvironment(simulator)
        return .conventional
        #else
        return supportsFamily(.metal4) ? .metal4 : .conventional
        #endif
    }

    static func select(device: any MTLDevice) -> TriangleBackend {
        select { device.supportsFamily($0) }
    }
}
