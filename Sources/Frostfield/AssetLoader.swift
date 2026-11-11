import Foundation
import Metal

/// An asynchronous loading manager and memory-budgeted cache, like Three.js
/// `LoadingManager` with `Cache`, built on structured concurrency.
///
/// - Concurrent requests for the same key share one load: the operation runs once and
///   every requester receives its value or error.
/// - A requester whose task is cancelled stops waiting immediately with
///   `CancellationError`. The shared load is cancelled when its last requester leaves,
///   and `cancelAll()` aborts everything, like Three.js `LoadingManager.abort`. A
///   cancelled load is forgotten at once, so a later request for the same key starts a
///   new load instead of joining one that is still unwinding; the cancelled operation's
///   eventual result is discarded.
/// - Loaded values stay cached by key until their total cost exceeds `memoryBudget`;
///   then the least recently used values are evicted, at amortized constant cost per
///   value. A value costing more than the whole budget is returned without being
///   cached. Failures are never cached.
/// - `events()` streams start, finish, failure and progress events. Within a batch
///   (from the first start until nothing is in flight) the started, loaded, failed and
///   cancelled counts never decrease, and each item's reported fraction never decreases.
///   `Progress.fractionCompleted` can drop when a load starts while others are in
///   flight, because the batch's total grows, as with Three.js `itemsTotal`.
/// - `urlModifier` rewrites every URL before keying and loading, like
///   `LoadingManager.setURLModifier`; requests that resolve to the same URL share a load.
public actor AssetLoader {
    /// Identifies a cached asset: its resolved URL plus a variant that distinguishes
    /// different decodings of the same file, such as sRGB and linear textures.
    public struct Key: Hashable, Sendable, CustomStringConvertible {
        public let url: URL
        public let variant: String

        public init(url: URL, variant: String = "") {
            self.url = url
            self.variant = variant
        }

        public var description: String { variant.isEmpty ? url.absoluteString : "\(url.absoluteString) [\(variant)]" }
    }

    /// Counts for the current batch, like the `itemsLoaded` and `itemsTotal` arguments of
    /// Three.js `LoadingManager.onProgress`.
    public struct Progress: Equatable, Sendable {
        /// Loads started in this batch.
        public var total = 0
        public var loaded = 0
        public var failed = 0
        public var cancelled = 0
        /// Finished items plus the reported fractions of items still loading, over `total`.
        public var fractionCompleted: Double = 0

        public init() {}

        public var finished: Int { loaded + failed + cancelled }
    }

    public enum Event: Equatable, Sendable {
        case started(Key, Progress)
        /// An item reported a new fraction of its own work.
        case progressed(Key, Progress)
        case loaded(Key, Progress)
        case failed(Key, message: String, Progress)
        case cancelled(Key, Progress)
        /// Nothing is in flight; the next start begins a new batch. Like Three.js `onLoad`.
        case finished(Progress)
    }

    /// Cache and sharing counters since the loader was created.
    public struct Statistics: Equatable, Sendable {
        public var hits = 0
        public var misses = 0
        /// Requests that joined a load already in flight.
        public var shared = 0
        public var evictions = 0
    }

    /// What an operation receives: the resolved URL and a way to report its own progress.
    public struct Context: Sendable {
        public let key: Key
        public var url: URL { key.url }
        let report: @Sendable (Double) -> Void

        /// Reports the fraction of this item's work that is done, from zero to one.
        /// Values below an earlier report are ignored.
        public func reportProgress(_ fraction: Double) {
            report(fraction)
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case typeMismatch(String)
        case invalidBudget
        case httpStatus(Int)

        public var errorDescription: String? {
            switch self {
            case .typeMismatch(let key): "Asset \(key) is cached as a different type than requested."
            case .invalidBudget: "The memory budget must be nonnegative."
            case .httpStatus(let code): "The asset request returned HTTP status \(code)."
            }
        }
    }

    private struct Entry {
        let value: any Sendable
        let cost: Int
        var lastUse: UInt64
    }

    private struct Waiter {
        let id: UInt64
        let continuation: CheckedContinuation<any Sendable, any Error>
    }

    private struct Flight {
        let id: UInt64
        var task: Task<Void, Never>?
        var waiters: [Waiter] = []
        var fraction: Double = 0
        let cost: @Sendable (any Sendable) -> Int
    }

    /// The largest total cost the cache keeps, in the units `cost` returns (bytes for the built-in loaders).
    public private(set) var memoryBudget: Int
    /// The total cost of cached values.
    public private(set) var cachedCost = 0
    public private(set) var progress = Progress()
    public private(set) var statistics = Statistics()
    private let urlModifier: @Sendable (URL) -> URL
    private var entries: [Key: Entry] = [:]
    /// Each cached entry's key by its `lastUse` tick, for least recently used eviction.
    private var recency: [UInt64: Key] = [:]
    /// No cached entry has a `lastUse` tick below this, so eviction scans forward from it.
    private var oldestTick: UInt64 = 1
    private var flights: [Key: Flight] = [:]
    private var subscribers: [UInt64: AsyncStream<Event>.Continuation] = [:]
    private var clock: UInt64 = 0
    private var nextID: UInt64 = 0

    /// - Parameters:
    ///   - memoryBudget: The cache's capacity, 256 MiB by default.
    ///   - urlModifier: Rewrites each request URL before it is keyed and loaded.
    public init(memoryBudget: Int = 256 << 20, urlModifier: @escaping @Sendable (URL) -> URL = { $0 }) throws {
        guard memoryBudget >= 0 else { throw Failure.invalidBudget }
        self.memoryBudget = memoryBudget
        self.urlModifier = urlModifier
    }

    /// A stream of loading events from now on. Finishing the stream's iteration unsubscribes.
    public func events() -> AsyncStream<Event> {
        let (stream, continuation) = AsyncStream<Event>.makeStream(bufferingPolicy: .unbounded)
        let id = makeID()
        subscribers[id] = continuation
        continuation.onTermination = { [weak self] _ in
            Task { await self?.unsubscribe(id) }
        }
        return stream
    }

    /// The key a URL and variant resolve to after the URL modifier.
    public nonisolated func key(for url: URL, variant: String = "") -> Key {
        Key(url: urlModifier(url), variant: variant)
    }

    /// Returns the cached value for a key, or loads it once for every concurrent requester.
    ///
    /// The operation runs outside the actor and should check for cancellation. `cost`
    /// measures the value for the memory budget.
    public func load<Value: Sendable>(
        _ url: URL, variant: String = "", as _: Value.Type = Value.self,
        cost: @escaping @Sendable (Value) -> Int = { _ in 0 },
        operation: @escaping @Sendable (Context) async throws -> Value
    ) async throws -> Value {
        try Task.checkCancellation()
        let key = key(for: url, variant: variant)
        if var entry = entries[key] {
            guard let value = entry.value as? Value else { throw Failure.typeMismatch(key.description) }
            statistics.hits += 1
            recency[entry.lastUse] = nil
            entry.lastUse = tick()
            recency[entry.lastUse] = key
            entries[key] = entry
            return value
        }
        if flights[key] == nil {
            statistics.misses += 1
            start(key, cost: { ($0 as? Value).map(cost) ?? 0 }, operation: operation)
        } else {
            statistics.shared += 1
        }
        let waiterID = makeID()
        let value = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<any Sendable, any Error>) in
                if flights[key] != nil {
                    flights[key]?.waiters.append(Waiter(id: waiterID, continuation: continuation))
                } else if let entry = entries[key] {
                    // The load finished between starting and waiting.
                    continuation.resume(returning: entry.value)
                } else {
                    continuation.resume(throwing: CancellationError())
                }
            }
        } onCancel: {
            Task { await self.cancelWaiter(waiterID, key: key) }
        }
        try Task.checkCancellation()
        guard let typed = value as? Value else { throw Failure.typeMismatch(key.description) }
        return typed
    }

    /// The cached value for a key, without loading and without counting a hit.
    public func cachedValue<Value: Sendable>(for key: Key, as _: Value.Type = Value.self) -> Value? {
        entries[key]?.value as? Value
    }

    /// Whether a load for the key is in flight.
    public func isLoading(_ key: Key) -> Bool {
        flights[key] != nil
    }

    /// Removes one cached value, like Three.js `Cache.remove`.
    public func removeValue(for key: Key) {
        guard let entry = entries.removeValue(forKey: key) else { return }
        cachedCost -= entry.cost
        recency[entry.lastUse] = nil
    }

    /// Empties the cache, like Three.js `Cache.clear`. Loads in flight continue.
    public func removeAll() {
        entries.removeAll()
        recency.removeAll()
        oldestTick = clock + 1
        cachedCost = 0
    }

    /// Changes the budget and evicts least recently used values until the cache fits.
    public func setMemoryBudget(_ budget: Int) throws {
        guard budget >= 0 else { throw Failure.invalidBudget }
        memoryBudget = budget
        evict(to: budget)
    }

    /// Cancels every load in flight; every requester receives `CancellationError`, and
    /// the next request for any of those keys starts a new load.
    public func cancelAll() {
        for key in Array(flights.keys) { abandon(key) }
    }

    // MARK: Internals

    private func start(
        _ key: Key, cost: @escaping @Sendable (any Sendable) -> Int,
        operation: @escaping @Sendable (Context) async throws -> some Sendable
    ) {
        let id = makeID()
        flights[key] = Flight(id: id, cost: cost)
        progress.total += 1
        updateFraction()
        emit(.started(key, progress))
        // The task and context hold the loader only until the operation finishes.
        let context = Context(key: key) { fraction in
            Task { await self.report(fraction, key: key, flight: id) }
        }
        // A detached task keeps the operation off this actor; its handle cancels the shared load.
        flights[key]?.task = Task.detached {
            do {
                let value = try await operation(context)
                await self.finish(key, flight: id, result: .success(value))
            } catch {
                await self.finish(key, flight: id, result: .failure(error))
            }
        }
    }

    private func finish(_ key: Key, flight id: UInt64, result: Result<any Sendable, any Error>) {
        guard let flight = flights[key], flight.id == id else { return }
        flights[key] = nil
        switch result {
        case .success(let value):
            let cost = max(0, flight.cost(value))
            if cost <= memoryBudget {
                if let old = entries[key] {
                    cachedCost -= old.cost
                    recency[old.lastUse] = nil
                }
                evict(to: memoryBudget - cost)
                let entry = Entry(value: value, cost: cost, lastUse: tick())
                entries[key] = entry
                recency[entry.lastUse] = key
                cachedCost += cost
            }
            progress.loaded += 1
            updateFraction()
            emit(.loaded(key, progress))
            for waiter in flight.waiters { waiter.continuation.resume(returning: value) }
        case .failure(let error):
            if error is CancellationError || flight.task?.isCancelled == true {
                progress.cancelled += 1
                updateFraction()
                emit(.cancelled(key, progress))
            } else {
                progress.failed += 1
                updateFraction()
                emit(.failed(key, message: error.localizedDescription, progress))
            }
            for waiter in flight.waiters { waiter.continuation.resume(throwing: error) }
        }
        finishBatchIfIdle()
    }

    private func finishBatchIfIdle() {
        guard flights.isEmpty else { return }
        emit(.finished(progress))
        progress = Progress()
    }

    /// Cancels a load and forgets it immediately, so a new request for the key starts a
    /// fresh load. `finish` ignores the cancelled task's late result by its flight ID.
    private func abandon(_ key: Key) {
        guard let flight = flights.removeValue(forKey: key) else { return }
        flight.task?.cancel()
        for waiter in flight.waiters { waiter.continuation.resume(throwing: CancellationError()) }
        progress.cancelled += 1
        updateFraction()
        emit(.cancelled(key, progress))
        finishBatchIfIdle()
    }

    private func report(_ fraction: Double, key: Key, flight id: UInt64) {
        guard let flight = flights[key], flight.id == id, fraction.isFinite else { return }
        let clamped = min(max(fraction, 0), 1)
        guard clamped > flight.fraction else { return }
        flights[key]?.fraction = clamped
        updateFraction()
        emit(.progressed(key, progress))
    }

    private func cancelWaiter(_ waiterID: UInt64, key: Key) {
        guard let flight = flights[key], let index = flight.waiters.firstIndex(where: { $0.id == waiterID }) else { return }
        flights[key]?.waiters.remove(at: index)
        flight.waiters[index].continuation.resume(throwing: CancellationError())
        if flights[key]?.waiters.isEmpty == true { abandon(key) }
    }

    /// Makes room before insertion so adding the new cost cannot overflow.
    /// The scan passes each recency tick only once.
    private func evict(to availableCost: Int) {
        while cachedCost > availableCost, oldestTick <= clock {
            guard let key = recency[oldestTick] else {
                oldestTick += 1
                continue
            }
            if let victim = entries.removeValue(forKey: key) {
                cachedCost -= victim.cost
                statistics.evictions += 1
            }
            recency[oldestTick] = nil
            oldestTick += 1
        }
    }

    private func updateFraction() {
        guard progress.total > 0 else {
            progress.fractionCompleted = 0
            return
        }
        let partial = flights.values.reduce(0) { $0 + $1.fraction }
        progress.fractionCompleted = min(1, (Double(progress.finished) + partial) / Double(progress.total))
    }

    private func emit(_ event: Event) {
        for subscriber in subscribers.values { subscriber.yield(event) }
    }

    private func unsubscribe(_ id: UInt64) {
        subscribers[id] = nil
    }

    private func tick() -> UInt64 {
        clock += 1
        return clock
    }

    private func makeID() -> UInt64 {
        nextID += 1
        return nextID
    }
}

/// A texture whose upload has completed and whose contents will not be written again,
/// so tasks may share it. `AssetLoader` returns textures in this wrapper because Metal
/// textures are not `Sendable`. Writing to the texture after loading is the caller's
/// responsibility to synchronize.
public final class LoadedTexture: @unchecked Sendable {
    public let texture: any MTLTexture

    public init(_ texture: any MTLTexture) {
        self.texture = texture
    }
}

public extension AssetLoader {
    /// Loads a file's bytes once. Its cost is the byte count. Local files are read
    /// directly; other URLs use `URLSession.shared`.
    func data(contentsOf url: URL) async throws -> Data {
        try await load(url, variant: "data", as: Data.self, cost: { $0.count }) { context in
            try await Self.fetch(context.url)
        }
    }

    private static func fetch(_ url: URL) async throws -> Data {
        try Task.checkCancellation()
        if url.isFileURL { return try Data(contentsOf: url, options: .mappedIfSafe) }
        let (data, response) = try await URLSession.shared.data(from: url)
        if let response = response as? HTTPURLResponse, !(200..<300).contains(response.statusCode) {
            throw Failure.httpStatus(response.statusCode)
        }
        return data
    }

    /// Loads a texture once per device, interpretation and mip setting with `TextureLoader`.
    /// Its cost is the texture's allocated size.
    func texture(
        contentsOf url: URL, device: any MTLDevice, interpretation: TextureLoader.Interpretation = .color,
        mipmapped: Bool = false
    ) async throws -> LoadedTexture {
        let name = switch interpretation {
        case .color: "color"
        case .encodedSRGB: "srgb"
        case .linearData: "linear"
        }
        let variant = "texture:\(device.registryID):\(name):\(mipmapped ? "mipmapped" : "single")"
        return try await load(url, variant: variant, as: LoadedTexture.self, cost: { $0.texture.allocatedSize }) { context in
            let data = try await Self.fetch(context.url)
            try Task.checkCancellation()
            context.reportProgress(0.5)
            let texture = try TextureLoader(device: device).load(data: data, interpretation: interpretation, mipmapped: mipmapped)
            return LoadedTexture(texture)
        }
    }
}
