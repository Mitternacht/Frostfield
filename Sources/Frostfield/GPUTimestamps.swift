import Metal

// The GPU writes the buffer during the pass. Only its completion handler resolves it.
struct GPUTimingSampleBuffer: @unchecked Sendable {
    let buffer: any MTLCounterSampleBuffer
}

enum GPUTimestamps {
    static func counterSet(on device: any MTLDevice) -> (any MTLCounterSet)? {
        guard device.supportsCounterSampling(.atStageBoundary) else { return nil }
        return device.counterSets?.first { $0.name == MTLCommonCounterSet.timestamp.rawValue }
    }

    static func samples(_ buffer: any MTLCounterSampleBuffer, count: Int) -> [MTLTimestamp]? {
        let stride = MemoryLayout<MTLCounterResultTimestamp>.stride
        guard let data = try? buffer.resolveCounterRange(0..<count), data.count >= count * stride else { return nil }
        return data.withUnsafeBytes { bytes in
            (0..<count).map {
                bytes.loadUnaligned(fromByteOffset: $0 * stride, as: MTLCounterResultTimestamp.self).timestamp
            }
        }
    }

    static func interval(_ start: MTLTimestamp, _ end: MTLTimestamp) -> ClosedRange<MTLTimestamp>? {
        guard start != MTLCounterErrorValue, end != MTLCounterErrorValue, end >= start else { return nil }
        return start...end
    }

    static func attachRenderSamples(
        _ buffer: any MTLCounterSampleBuffer, to attachment: MTLRenderPassSampleBufferAttachmentDescriptor, at base: Int
    ) {
        attachment.sampleBuffer = buffer
        attachment.startOfVertexSampleIndex = base
        attachment.endOfVertexSampleIndex = base + 1
        attachment.startOfFragmentSampleIndex = base + 2
        attachment.endOfFragmentSampleIndex = base + 3
    }

    static func renderStages(
        _ samples: [MTLTimestamp], at base: Int, calibration: GPUTimestampCalibration
    ) -> RenderGPUTiming.Measurement {
        let vertex = interval(samples[base], samples[base + 1])
        let fragment = interval(samples[base + 2], samples[base + 3])
        let span: ClosedRange<MTLTimestamp>?
        if let vertex, let fragment {
            span = min(vertex.lowerBound, fragment.lowerBound)...max(vertex.upperBound, fragment.upperBound)
        } else {
            span = vertex ?? fragment
        }
        return RenderGPUTiming.Measurement(
            vertexSeconds: vertex.map(calibration.seconds), fragmentSeconds: fragment.map(calibration.seconds),
            elapsedSeconds: span.map(calibration.seconds)
        )
    }
}

struct GPUTimestampCalibration: Sendable {
    let secondsPerTick: Double

    init?(start: (cpu: MTLTimestamp, gpu: MTLTimestamp), end: (cpu: MTLTimestamp, gpu: MTLTimestamp)) {
        guard end.cpu > start.cpu, end.gpu > start.gpu else { return nil }
        // Metal CPU timestamps use nanoseconds. Scale GPU ticks by the observed clock ratio.
        secondsPerTick = Double(end.cpu - start.cpu) * 1e-9 / Double(end.gpu - start.gpu)
    }

    func seconds(_ interval: ClosedRange<MTLTimestamp>) -> Double {
        Double(interval.upperBound - interval.lowerBound) * secondsPerTick
    }
}
