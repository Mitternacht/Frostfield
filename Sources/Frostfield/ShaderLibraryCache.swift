import Foundation
import Metal
import Synchronization

/// Shares runtime-compiled Metal libraries between preparations of identical
/// generated source on the same device, so several materials or passes built
/// from one graph compile it once. Bounded to the most recently used entries.
package enum ShaderLibraryCache {
    package static let capacity = 64

    private struct Key: Hashable {
        let device: ObjectIdentifier
        let source: String
    }

    // MTLLibrary is documented as thread-safe for function creation; the cache only hands out references.
    private struct Entry: @unchecked Sendable {
        let library: any MTLLibrary
        var lastUse: UInt64
        var compilations: Int
    }

    private struct State {
        var entries: [Key: Entry] = [:]
        var clock: UInt64 = 0
        /// Compilations of sources containing each watched marker, kept across evictions.
        var watched: [String: Int] = [:]

        mutating func recordCompilation(of source: String) {
            for marker in watched.keys where source.contains(marker) { watched[marker, default: 0] += 1 }
        }
    }

    private static let state = Mutex(State())

    /// Returns a cached library or compiles one with safe math. Compilation errors propagate.
    package static func library(source: String, device: any MTLDevice) throws -> any MTLLibrary {
        let key = Key(device: ObjectIdentifier(device), source: source)
        if let cached = state.withLock({ state -> (any MTLLibrary)? in
            guard var entry = state.entries[key] else { return nil }
            state.clock &+= 1
            entry.lastUse = state.clock
            state.entries[key] = entry
            return entry.library
        }) {
            return cached
        }
        let options = MTLCompileOptions()
        options.mathMode = .safe
        let library = try device.makeLibrary(source: source, options: options)
        return state.withLock { state in
            state.recordCompilation(of: source)
            if var existing = state.entries[key] {
                // Another thread compiled the same source concurrently; keep the first library.
                existing.compilations += 1
                state.entries[key] = existing
                return existing.library
            }
            state.clock &+= 1
            state.entries[key] = Entry(library: library, lastUse: state.clock, compilations: 1)
            if state.entries.count > capacity,
               let oldest = state.entries.min(by: { $0.value.lastUse < $1.value.lastUse })?.key {
                state.entries.removeValue(forKey: oldest)
            }
            return library
        }
    }

    /// Starts counting compilations of sources that contain the marker. Counts survive
    /// eviction, so concurrent compilations elsewhere cannot hide a recompilation.
    package static func countCompilations(containing marker: String) {
        state.withLock { $0.watched[marker] = 0 }
    }

    /// Compilations of sources containing a watched marker since `countCompilations(containing:)`.
    /// For an unwatched marker, the compilations of entries still cached.
    package static func compilationCount(containing marker: String) -> Int {
        state.withLock { state in
            if let count = state.watched[marker] { return count }
            return state.entries.reduce(0) { $0 + ($1.key.source.contains(marker) ? $1.value.compilations : 0) }
        }
    }

    /// Cached libraries for one device: how many, and how many source bytes they hold.
    package static func snapshot(for device: any MTLDevice) -> (entries: Int, sourceBytes: Int) {
        let key = ObjectIdentifier(device)
        return state.withLock { state in
            let entries = state.entries.filter { $0.key.device == key }
            return (entries.count, entries.reduce(0) { $0 + $1.key.source.utf8.count })
        }
    }
}
