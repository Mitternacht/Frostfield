import Frostfield
import Testing
import simd

struct SeededRandomTests {
    @Test("Park-Miller samples match the Three.js seeded sequence")
    func matchesThreeJS() throws {
        var first = SeededRandom(seed: 1)
        #expect(abs(first.next() - 7.825903601782307e-06) < 1e-18)
        #expect(abs(first.next() - 0.13153778773875702) < 1e-18)
        #expect(first.integer(in: 0...3) == 3)
        var standard = SeededRandom()
        #expect(abs(standard.next() - 0.6621772173439872) < 1e-15)
        var wrapped = SeededRandom(seed: -5)
        #expect(abs(wrapped.next() - 0.9999530417844216) < 1e-15)
        #expect(try first.next(from: 4, through: 4) == 4)
        #expect(throws: SeededRandom.Failure.nonfinite) { try first.next(from: .nan, through: 1) }
        let direction = first.direction()
        #expect(abs(simd_length(direction) - 1) < 1e-12)
    }
}
