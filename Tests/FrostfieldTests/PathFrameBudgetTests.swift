import Frostfield
import Testing

struct PathFrameBudgetTests {
    @Test
    func `Frame budget recommends a whole batch and follows changing pass costs`() throws {
        var budget = try PathTracer.FrameBudget(targetSeconds: 0.025, range: 2...12, responsiveness: 0.5)
        #expect(budget.samplesPerFrame == 2)
        #expect(budget.secondsPerSample == nil)
        #expect(budget.record(seconds: 0.016, samples: 4) == 6)
        #expect(budget.record(seconds: 0.048, samples: 6) == 4)
        #expect(abs(try #require(budget.secondsPerSample) - 0.006) < 1e-12)
        #expect(budget.record(seconds: 0.004, samples: 4) == 7)
        #expect(budget.record(seconds: 1, samples: 2) == 2)
    }

    @Test
    func `Unavailable timings leave the recommendation unchanged`() throws {
        var budget = try PathTracer.FrameBudget(targetSeconds: 0.01, initialSamples: 3)
        let original = budget
        for seconds: Double? in [nil, 0, -1, .infinity, .nan] {
            #expect(budget.record(seconds: seconds, samples: 3) == 3)
        }
        for samples in [0, 17] {
            #expect(budget.record(seconds: 0.01, samples: samples) == 3)
        }
        #expect(budget == original)
    }

    @Test
    func `Finite extreme timings and budgets clamp before integer conversion`() throws {
        var budget = try PathTracer.FrameBudget(targetSeconds: .greatestFiniteMagnitude, responsiveness: 1)
        #expect(budget.record(seconds: .leastNonzeroMagnitude, samples: 16) == 16)
        #expect(budget.secondsPerSample == .leastNonzeroMagnitude)
        #expect(budget.record(seconds: .greatestFiniteMagnitude, samples: 1) == 1)
        #expect(budget.record(seconds: .leastNonzeroMagnitude, samples: 2) == 16)

        var tiny = try PathTracer.FrameBudget(targetSeconds: .leastNonzeroMagnitude)
        #expect(tiny.record(seconds: .greatestFiniteMagnitude, samples: 1) == 1)
    }

    @Test
    func `Frame budget rejects invalid targets, sample ranges, and weights`() {
        #expect(throws: PathTracer.Failure.frameBudget) {
            try PathTracer.FrameBudget(targetSeconds: .infinity)
        }
        #expect(throws: PathTracer.Failure.frameBudget) {
            try PathTracer.FrameBudget(targetSeconds: 0.01, range: 1...17)
        }
        #expect(throws: PathTracer.Failure.frameBudget) {
            try PathTracer.FrameBudget(targetSeconds: 0.01, range: 2...8, initialSamples: 1)
        }
        #expect(throws: PathTracer.Failure.frameBudget) {
            try PathTracer.FrameBudget(targetSeconds: 0.01, responsiveness: .nan)
        }
    }
}
