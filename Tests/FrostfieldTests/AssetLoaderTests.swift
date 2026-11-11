import Foundation
import Frostfield
import Metal
import Synchronization
import Testing

/// A latch that suspends loads until a test opens it.
private actor Gate {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        if isOpen { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func open() {
        isOpen = true
        for waiter in waiters { waiter.resume() }
        waiters = []
    }
}

private struct Expected: Error, Equatable {}

private func url(_ name: String) -> URL { URL(fileURLWithPath: "/assets/\(name)") }

/// Waits until a condition on the loader holds, polling between yields.
private func eventually(_ condition: () async -> Bool) async throws {
    for _ in 0..<2_000 {
        if await condition() { return }
        try await Task.sleep(for: .milliseconds(1))
    }
    Issue.record("Condition did not become true in time")
}

struct AssetLoaderTests {
    @Test func `A full integer budget evicts before adding a new cost`() async throws {
        let loader = try AssetLoader(memoryBudget: .max)
        let first = url("full"), second = url("next")
        _ = try await loader.load(first, cost: { _ in Int.max }) { _ in 1 }
        _ = try await loader.load(second, cost: { _ in 16 }) { _ in 2 }
        #expect(await loader.cachedCost == 16)
        #expect(await loader.cachedValue(for: loader.key(for: first), as: Int.self) == nil)
        #expect(await loader.cachedValue(for: loader.key(for: second), as: Int.self) == 2)
    }

    @Test
    func `Concurrent requests for one asset share a single load`() async throws {
        let loader = try AssetLoader()
        let runs = Mutex(0)
        let gate = Gate()
        let values = try await withThrowingTaskGroup(of: [Int].self) { group in
            for _ in 0..<24 {
                group.addTask {
                    try await loader.load(url("mesh.bin"), cost: { $0.count }) { _ in
                        runs.withLock { $0 += 1 }
                        await gate.wait()
                        return [1, 2, 3]
                    }
                }
            }
            try await eventually { await loader.statistics.misses + loader.statistics.shared == 24 }
            await gate.open()
            return try await group.reduce(into: []) { $0.append($1) }
        }
        #expect(values.count == 24 && values.allSatisfy { $0 == [1, 2, 3] })
        #expect(runs.withLock { $0 } == 1)
        var statistics = await loader.statistics
        #expect(statistics.misses == 1 && statistics.shared == 23)

        let cached = try await loader.load(url("mesh.bin"), as: [Int].self) { _ in Issue.record("reloaded"); return [] }
        #expect(cached == [1, 2, 3])
        statistics = await loader.statistics
        #expect(statistics.hits == 1)
        #expect(await loader.cachedCost == 3)
    }

    @Test
    func `Progress counts and fractions never decrease within a batch`() async throws {
        let loader = try AssetLoader()
        let events = await loader.events()
        let gates = (0..<5).map { _ in Gate() }
        let tasks = (0..<5).map { index in
            Task {
                try await loader.load(url("item\(index)"), as: Int.self) { context in
                    context.reportProgress(0.25)
                    context.reportProgress(0.5)
                    context.reportProgress(0.1)
                    await gates[index].wait()
                    if index == 3 { throw Expected() }
                    return index
                }
            }
        }
        try await eventually { await loader.progress.total == 5 }
        for gate in gates { await gate.open() }
        for (index, task) in tasks.enumerated() {
            if index == 3 {
                await #expect(throws: Expected.self) { try await task.value }
            } else {
                #expect(try await task.value == index)
            }
        }

        var previous = AssetLoader.Progress()
        var finished: AssetLoader.Progress?
        for await event in events {
            let progress: AssetLoader.Progress
            switch event {
            case .started(_, let value), .progressed(_, let value), .loaded(_, let value), .cancelled(_, let value):
                progress = value
            case .failed(let key, _, let value):
                #expect(key.url == url("item3"))
                progress = value
            case .finished(let value):
                finished = value
                progress = value
            }
            #expect(progress.loaded >= previous.loaded && progress.failed >= previous.failed)
            #expect(progress.total >= previous.total)
            // Every item starts before any finishes here, so the fraction is monotonic too.
            if progress.total == 5, previous.total == 5 {
                #expect(progress.fractionCompleted >= previous.fractionCompleted)
            }
            previous = progress
            if finished != nil { break }
        }
        #expect(finished?.loaded == 4 && finished?.failed == 1 && finished?.total == 5)
        #expect(finished?.fractionCompleted == 1)
        // The next load starts a new batch.
        _ = try await loader.load(url("later"), as: Int.self) { _ in 9 }
        #expect(await loader.progress == AssetLoader.Progress())
    }

    @Test
    func `Cancelling the last requester cancels the load and failures are not cached`() async throws {
        let loader = try AssetLoader()
        let sawCancellation = Mutex(false)
        let operation: @Sendable (AssetLoader.Context) async throws -> Int = { _ in
            do {
                try await Task.sleep(for: .seconds(30))
            } catch {
                sawCancellation.withLock { $0 = true }
                throw error
            }
            return 1
        }
        let first = Task { try await loader.load(url("slow"), operation: operation) }
        let second = Task { try await loader.load(url("slow"), operation: operation) }
        try await eventually { await loader.statistics.shared == 1 }
        first.cancel()
        await #expect(throws: CancellationError.self) { try await first.value }
        // The other requester keeps the load alive.
        #expect(await loader.isLoading(loader.key(for: url("slow"))))
        #expect(!sawCancellation.withLock { $0 })
        second.cancel()
        await #expect(throws: CancellationError.self) { try await second.value }
        try await eventually { sawCancellation.withLock { $0 } }
        try await eventually { await !loader.isLoading(loader.key(for: url("slow"))) }
        #expect(await loader.cachedValue(for: loader.key(for: url("slow")), as: Int.self) == nil)

        // cancelAll aborts every load and every requester.
        let third = Task { try await loader.load(url("slow-2"), operation: operation) }
        try await eventually { await loader.isLoading(loader.key(for: url("slow-2"))) }
        await loader.cancelAll()
        await #expect(throws: CancellationError.self) { try await third.value }

        let attempts = Mutex(0)
        for expected in 1...2 {
            await #expect(throws: Expected.self) {
                try await loader.load(url("broken"), as: Int.self) { _ in
                    attempts.withLock { $0 += 1 }
                    throw Expected()
                }
            }
            #expect(attempts.withLock { $0 } == expected)
        }
    }

    @Test
    func `A request after its key's load was cancelled starts a new load instead of joining it`() async throws {
        let loader = try AssetLoader()
        let key = loader.key(for: url("model"))
        let runs = Mutex(0)
        // The first run ignores cancellation until the gate opens, like an operation stuck in I/O.
        let unwinding = Gate()
        let operation: @Sendable (AssetLoader.Context) async throws -> Int = { _ in
            let run = runs.withLock { $0 += 1; return $0 }
            if run == 1 {
                await unwinding.wait()
                throw CancellationError()
            }
            return 7
        }
        let events = await loader.events()
        let first = Task { try await loader.load(url("model"), operation: operation) }
        try await eventually { await loader.isLoading(key) }
        first.cancel()
        await #expect(throws: CancellationError.self) { try await first.value }
        // The cancelled load is forgotten at once, while its operation is still running.
        #expect(await !loader.isLoading(key))
        #expect(await loader.progress == AssetLoader.Progress())

        let second = Task { try await loader.load(url("model"), operation: operation) }
        try await eventually { await loader.statistics.misses + loader.statistics.shared == 2 }
        await unwinding.open()
        #expect(try await second.value == 7)
        #expect(runs.withLock { $0 } == 2)
        #expect(await loader.statistics.shared == 0)
        // The first run's late failure neither replaces the value nor reaches anyone.
        try await Task.sleep(for: .milliseconds(20))
        #expect(await loader.cachedValue(for: key, as: Int.self) == 7)

        // One batch for the cancelled load, then one for the fresh load.
        var batches: [AssetLoader.Progress] = []
        for await event in events {
            if case .finished(let progress) = event { batches.append(progress) }
            if batches.count == 2 { break }
        }
        #expect(batches.map(\.cancelled) == [1, 0] && batches.map(\.loaded) == [0, 1])

        // cancelAll forgets loads the same way.
        let stuck = Gate()
        let slow = Task {
            try await loader.load(url("slow"), as: Int.self) { _ in
                await stuck.wait()
                throw CancellationError()
            }
        }
        try await eventually { await loader.isLoading(loader.key(for: url("slow"))) }
        await loader.cancelAll()
        await #expect(throws: CancellationError.self) { try await slow.value }
        #expect(try await loader.load(url("slow"), as: Int.self) { _ in 3 } == 3)
        await stuck.open()
    }

    @Test
    func `Eviction stays least recently used and fast with thousands of cached values`() async throws {
        let count = 8_000
        let loader = try AssetLoader(memoryBudget: count)
        for index in 0..<count {
            _ = try await loader.load(url("value\(index)"), as: Int.self, cost: { _ in 1 }) { _ in index }
        }
        // Touch the first hundred so they become the most recently used.
        for index in 0..<100 {
            _ = try await loader.load(url("value\(index)"), as: Int.self, cost: { _ in 1 }) { _ in -1 }
        }
        let clock = ContinuousClock()
        let elapsed = try await clock.measure { try await loader.setMemoryBudget(150) }
        #expect(await loader.cachedCost == 150)
        #expect(await loader.statistics.evictions == count - 150)
        // Survivors: the 50 newest untouched values and the 100 touched ones.
        for index in [0, 99, 7_950, 7_999] {
            #expect(await loader.cachedValue(for: loader.key(for: url("value\(index)")), as: Int.self) == index)
        }
        for index in [100, 7_949] {
            #expect(await loader.cachedValue(for: loader.key(for: url("value\(index)")), as: Int.self) == nil)
        }
        // Evicting 7,850 values scans each once; the old per-victim scan took seconds here.
        #expect(elapsed < .milliseconds(500), "\(elapsed)")

        // Removal and clearing keep the recency order consistent.
        await loader.removeValue(for: loader.key(for: url("value0")))
        _ = try await loader.load(url("fresh"), as: Int.self, cost: { _ in 1 }) { _ in 1 }
        try await loader.setMemoryBudget(1)
        #expect(await loader.cachedValue(for: loader.key(for: url("fresh")), as: Int.self) == 1)
        await loader.removeAll()
        _ = try await loader.load(url("after"), as: Int.self, cost: { _ in 1 }) { _ in 2 }
        _ = try await loader.load(url("later"), as: Int.self, cost: { _ in 1 }) { _ in 3 }
        #expect(await loader.cachedValue(for: loader.key(for: url("after")), as: Int.self) == nil)
        #expect(await loader.cachedValue(for: loader.key(for: url("later")), as: Int.self) == 3)
    }

    @Test
    func `The cache evicts least recently used values to stay within its budget`() async throws {
        let loader = try AssetLoader(memoryBudget: 100)
        func load(_ name: String, cost: Int) async throws -> Int {
            try await loader.load(url(name), as: Int.self, cost: { _ in cost }) { _ in cost }
        }
        _ = try await load("a", cost: 60)
        _ = try await load("b", cost: 30)
        _ = try await load("a", cost: 60)
        _ = try await load("c", cost: 40)
        #expect(await loader.cachedValue(for: loader.key(for: url("a")), as: Int.self) == 60)
        #expect(await loader.cachedValue(for: loader.key(for: url("b")), as: Int.self) == nil)
        #expect(await loader.cachedValue(for: loader.key(for: url("c")), as: Int.self) == 40)
        #expect(await loader.cachedCost == 100)
        #expect(await loader.statistics.evictions == 1)

        // A value larger than the whole budget is returned but not kept.
        #expect(try await load("huge", cost: 500) == 500)
        #expect(await loader.cachedValue(for: loader.key(for: url("huge")), as: Int.self) == nil)
        try await loader.setMemoryBudget(50)
        #expect(await loader.cachedCost == 40)
        await loader.removeAll()
        #expect(await loader.cachedCost == 0)
    }

    @Test
    func `URL modifiers redirect requests and requests that resolve alike share one load`() async throws {
        let loader = try AssetLoader { url in
            URL(fileURLWithPath: "/cdn/" + url.lastPathComponent.replacingOccurrences(of: "@2x", with: ""))
        }
        let seen = Mutex<[URL]>([])
        let operation: @Sendable (AssetLoader.Context) async throws -> String = { context in
            seen.withLock { $0.append(context.url) }
            return context.url.lastPathComponent
        }
        #expect(try await loader.load(url("icon.png"), operation: operation) == "icon.png")
        #expect(try await loader.load(url("icon@2x.png"), operation: operation) == "icon.png")
        #expect(seen.withLock { $0 } == [URL(fileURLWithPath: "/cdn/icon.png")])
        await #expect(throws: AssetLoader.Failure.self) {
            try await loader.load(url("icon.png"), as: Int.self) { _ in 1 }
        }
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func `Textures load once per device and decoding and report their allocated size`() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        // A 2 × 1 PNG with sRGB-encoded red and green pixels.
        let png = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAADklEQVR4nGP4z8DwHwQBEPgD/U6VwW8AAAAASUVORK5CYII=")
        let directory = FileManager.default.temporaryDirectory.appending(path: "frostfield-loader-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appending(path: "pair.png")
        try #require(png).write(to: file)

        let loader = try AssetLoader()
        async let first = loader.texture(contentsOf: file, device: device)
        async let second = loader.texture(contentsOf: file, device: device)
        let (a, b) = try await (first, second)
        #expect(a === b)
        #expect(a.texture.width == 2 && a.texture.pixelFormat == .rgba8Unorm_srgb)
        let linear = try await loader.texture(contentsOf: file, device: device, interpretation: .linearData)
        #expect(linear !== a && linear.texture.pixelFormat == .rgba8Unorm)
        #expect(await loader.cachedCost == a.texture.allocatedSize + linear.texture.allocatedSize)
        var texels = [UInt8](repeating: 0, count: 8)
        texels.withUnsafeMutableBytes { bytes in
            a.texture.getBytes(bytes.baseAddress!, bytesPerRow: 8, from: MTLRegionMake2D(0, 0, 2, 1), mipmapLevel: 0)
        }
        #expect(texels == [255, 0, 0, 255, 0, 255, 0, 255])
    }
}
