import Foundation
import Frostfield
import Testing

struct PerformanceMonitorTests {
    @Test("Slow frames lower the factor, fast frames raise it, and too many reversals stop")
    func stepsAndFallback() throws {
        var monitor = try PerformanceMonitor(
            lowerFramesPerSecond: 50, upperFramesPerSecond: 60, iterations: 2, step: 0.25, maximumReversals: 2
        )
        #expect(monitor.factor == 1)
        #expect(try monitor.pixelScale(minimum: 0.5, maximum: 1) == 1)
        #expect(try monitor.record(frameTime: 1 / Float(30)) == nil)
        #expect(try monitor.record(frameTime: 1 / Float(30)) == .decline(factor: 0.75))
        #expect(try monitor.pixelScale(minimum: 0.5, maximum: 1) == 0.875)

        #expect(try monitor.record(frameTime: 1 / Float(120)) == nil)
        #expect(try monitor.record(frameTime: 1 / Float(120)) == .incline(factor: 1))
        // The next decline is the second reversal, so the factor stays put.
        #expect(try monitor.record(frameTime: 1 / Float(30)) == nil)
        #expect(try monitor.record(frameTime: 1 / Float(30)) == nil)
        #expect(monitor.fellBack)
        #expect(monitor.factor == 1)
        #expect(try monitor.record(frameTime: 1 / Float(30)) == nil)
        #expect(monitor.factor == 1)
    }

    @Test("A rate inside the bounds and a factor already at its limit do not move")
    func steadyAndClamped() throws {
        var steady = try PerformanceMonitor(
            lowerFramesPerSecond: 40, upperFramesPerSecond: 60, iterations: 1, step: 0.25
        )
        #expect(try steady.record(frameTime: 1 / Float(50)) == nil)
        #expect(steady.factor == 1)
        var low = try PerformanceMonitor(
            lowerFramesPerSecond: 40, upperFramesPerSecond: 60, iterations: 1, step: 0.25, factor: 0
        )
        #expect(try low.record(frameTime: 1 / Float(10)) == nil)
        #expect(low.factor == 0)
        #expect(throws: PerformanceMonitor.Failure.frameTime) { try low.record(frameTime: 0) }
        #expect(throws: PerformanceMonitor.Failure.bounds) {
            try PerformanceMonitor(lowerFramesPerSecond: 60, upperFramesPerSecond: 40)
        }
        #expect(throws: PerformanceMonitor.Failure.pixelScale) { try low.pixelScale(minimum: 2, maximum: 1) }
    }
}
