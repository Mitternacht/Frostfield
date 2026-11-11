import Foundation
import Frostfield
import Synchronization
import Testing
import simd

struct MeshBVHBackgroundBuildTests {
    @MainActor
    @Test func `The concurrent build shares the source and exactly matches synchronous topology and queries`() async throws {
        let source = try mesh()
        let synchronous = try MeshBVH(geometry: source)
        let reports = Mutex<[(MeshBVH.BuildProgress, Bool)]>([])
        let accelerated = try await MeshBVH.build(geometry: source) { report in
            let isMain = Thread.isMainThread
            reports.withLock { $0.append((report, isMain)) }
        }
        #expect(accelerated.bounds == synchronous.bounds)
        #expect(accelerated.depth == synchronous.depth)
        #expect(accelerated.nodeCount == synchronous.nodeCount)
        #expect(accelerated.triangleCount == synchronous.triangleCount)
        #expect(accelerated.geometry.groups == source.groups)
        #expect(accelerated.geometry.vertices.withUnsafeBufferPointer { actual in
            source.vertices.withUnsafeBufferPointer { $0.baseAddress == actual.baseAddress }
        })
        #expect(accelerated.geometry.indices.withUnsafeBufferPointer { actual in
            source.indices.withUnsafeBufferPointer { $0.baseAddress == actual.baseAddress }
        })
        #expect(try leaves(accelerated) == leaves(synchronous))

        for position: SIMD3<Float> in [SIMD3(0.25, 0.125, 3), SIMD3(-1.75, 0.25, 3), SIMD3(5, 0, 3)] {
            let ray = try SceneRay(origin: position, direction: SIMD3(0, 0, -1), maximumDistance: 3)
            #expect(try accelerated.intersection(ray: ray) == source.intersection(ray: ray, localToWorld: matrix_identity_float4x4))
            #expect(try accelerated.intersections(ray: ray) == synchronous.intersections(ray: ray))
            #expect(try accelerated.closestPoint(to: position, maximumDistance: 3)
                == synchronous.closestPoint(to: position, maximumDistance: 3))
        }

        let recorded = reports.withLock { $0 }
        #expect(recorded.allSatisfy { !$0.1 })
        let progress = recorded.map { $0.0 }
        let last = try #require(progress.last)
        #expect(progress.count > 3)
        #expect(progress.allSatisfy { $0.triangleCount == accelerated.triangleCount })
        #expect(progress.contains { $0.completedBoundsCount < $0.triangleCount && $0.indexedTriangleCount == 0 })
        #expect(progress.contains { $0.indexedTriangleCount > 0 && $0.indexedTriangleCount < $0.triangleCount })
        #expect(zip(progress, progress.dropFirst()).allSatisfy { $0.fractionCompleted < $1.fractionCompleted })
        #expect(last.fractionCompleted == 1)
        #expect(last.completedBoundsCount == last.triangleCount)
        #expect(last.indexedTriangleCount == last.triangleCount)
    }

    enum CancellationStage: CaseIterable, Sendable {
        case triangleBounds, treeIndexing, completion

        func reached(_ report: MeshBVH.BuildProgress) -> Bool {
            switch self {
            case .triangleBounds: report.completedBoundsCount < report.triangleCount && report.indexedTriangleCount == 0
            case .treeIndexing: report.indexedTriangleCount > 0 && report.indexedTriangleCount < report.triangleCount
            case .completion: report.indexedTriangleCount == report.triangleCount
            }
        }
    }

    @Test(arguments: CancellationStage.allCases)
    func `Cancellation during the build prevents publication and leaves the source reusable`(stage: CancellationStage) async throws {
        let source = try mesh()
        let reports = Mutex<[MeshBVH.BuildProgress]>([])
        let task = Task {
            try await MeshBVH.build(geometry: source) { report in
                reports.withLock { $0.append(report) }
                if stage.reached(report) {
                    withUnsafeCurrentTask { $0?.cancel() }
                }
            }
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        let records = reports.withLock { $0 }
        let stoppedAt = try #require(records.last)
        #expect(stage.reached(stoppedAt))
        #expect(records.filter { stage.reached($0) }.count == 1)

        let retry = try await MeshBVH.build(geometry: source)
        #expect(retry.geometry.indices == source.indices)
        let ray = try SceneRay(origin: SIMD3(0.25, 0.125, 1), direction: SIMD3(0, 0, -1), maximumDistance: 1)
        #expect(try retry.intersection(ray: ray) == source.intersection(ray: ray, localToWorld: matrix_identity_float4x4))
    }

    @Test func `A pre-cancelled task reports no progress and can still use the synchronous initializer`() async throws {
        let source = try MeshGeometry.plane()
        let reports = Mutex(0)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            let synchronous = try MeshBVH(geometry: source)
            #expect(synchronous.triangleCount == 2)
            return try await MeshBVH.build(geometry: source) { _ in reports.withLock { $0 += 1 } }
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(reports.withLock { $0 } == 0)
    }

    private func mesh() throws -> MeshGeometry {
        try MeshGeometry.plane(width: 6, height: 4, columns: 48, rows: 40)
            .withGroups([MeshGeometryGroup(start: 0, count: 6, materialIndex: 7)])
    }

    private func leaves(_ bvh: MeshBVH) throws -> [Int: [Int]] {
        var ranges: [Int: [Int]] = [:]
        _ = try bvh.shapecast(classifyBounds: { _, _ in .intersecting }, visitRange: { node, range, _ in
            ranges[node.index] = Array(range)
            return false
        })
        return ranges
    }
}
