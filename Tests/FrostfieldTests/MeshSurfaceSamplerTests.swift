import Frostfield
import simd
import Testing

struct MeshSurfaceSamplerTests {
    @Test("Surface samples follow triangle area and interpolate local attributes")
    func areaAndAttributes() throws {
        let firstNormal = SIMD3<Float>(0, 0, 1)
        let geometry = try MeshGeometry(vertices: [
            MeshVertex(position: SIMD3(0, 0, 0), normal: firstNormal, textureCoordinate: .zero),
            MeshVertex(position: SIMD3(1, 0, 0), normal: SIMD3(1, 0, 1), textureCoordinate: SIMD2(1, 0)),
            MeshVertex(position: SIMD3(0, 1, 0), normal: SIMD3(0, 1, 1), textureCoordinate: SIMD2(0, 1)),
            MeshVertex(position: SIMD3(3, 0, 0), normal: firstNormal),
            MeshVertex(position: SIMD3(5, 0, 0), normal: firstNormal),
            MeshVertex(position: SIMD3(3, 2, 0), normal: firstNormal),
            MeshVertex(position: SIMD3(100, 0, 0), normal: firstNormal),
        ], indices: [0, 1, 2, 3, 4, 5, 6, 6, 6])
        let sampler = try MeshSurfaceSampler(geometry: geometry)
        let samples = try sampler.samples(count: 4_096, seed: 42)
        #expect(try sampler.samples(count: 4_096, seed: 42) == samples)
        #expect(try sampler.samples(count: 4_096, seed: 43) != samples)

        let small = samples.filter { $0.position.x < 2 }
        let large = samples.filter { $0.position.x >= 3 }
        #expect(small.count + large.count == samples.count)
        #expect((0.77...0.83).contains(Double(large.count) / Double(samples.count)))
        #expect(large.allSatisfy { ($0.position.x - 3) + $0.position.y <= 2.000001 })
        #expect(small.allSatisfy { sample in
            let position = sample.position
            let expectedNormal = simd_normalize(SIMD3(position.x, position.y, 1))
            return position.x >= 0 && position.y >= 0 && position.x + position.y <= 1.000001
                && position.z == 0
                && simd_distance(sample.normal, expectedNormal) < 0.000001
                && sample.textureCoordinate == SIMD2(position.x, position.y)
        })
    }

    @Test("Degenerate surfaces and out-of-range sample counts fail explicitly")
    func validation() throws {
        let vertex = MeshVertex(position: .zero, normal: SIMD3(0, 1, 0))
        let geometry = try MeshGeometry(vertices: [vertex], indices: [0, 0, 0])
        #expect(throws: MeshSurfaceSampler.Failure.emptySurface) {
            try MeshSurfaceSampler(geometry: geometry)
        }

        let sampler = try MeshSurfaceSampler(geometry: .cube())
        #expect(try sampler.samples(count: 0, seed: 0).isEmpty)
        #expect(throws: MeshSurfaceSampler.Failure.count(-1)) {
            try sampler.samples(count: -1, seed: 0)
        }
        let overLimit = MeshSurfaceSampler.maximumSampleCount + 1
        #expect(throws: MeshSurfaceSampler.Failure.count(overLimit)) {
            try sampler.samples(count: overLimit, seed: 0)
        }
    }
}
