import Foundation
import Metal
import simd

/// Controls when a native raster shadow map renders its depth passes.
public enum ShadowMapUpdateMode: Sendable {
    /// Render the required shadow views every time a scene uses them.
    case automatic
    /// Reuse completed depth until requested or the light-space layout changes.
    case onDemand
}

public enum ShadowMapUpdateError: LocalizedError, Equatable {
    case commandBufferInUse
    case depthUnavailable
    case lightUpdateModesUnavailable

    public var errorDescription: String? {
        switch self {
        case .commandBufferInUse:
            "Finish or discard the command buffer already using this on-demand shadow map before encoding another use."
        case .depthUnavailable:
            "The selected on-demand shadow slice has no valid depth. Render the scene's shadow views before inspecting it."
        case .lightUpdateModesUnavailable:
            "Supply lightUpdateModes when creating an automatic punctual shadow set to enable per-light update policies."
        }
    }
}

/// A map has one physical allocation. Hold its read or write lease until
/// completion so submitting another command buffer out of order is safe.
final class ShadowMapUpdateCache: @unchecked Sendable {
    struct Layout: Equatable {
        struct Group: Equatable {
            struct RelativeProjection: Equatable {
                let origin: SIMD3<Double>
                let clipFromOrigin: simd_double4x4
            }

            let lightID: UUID?
            let slices: Range<Int>
            let momentBlurRadius: Int?
            private let relativeProjections: [RelativeProjection]
            private let projections: [SIMD4<Float>]

            init(
                lightID: UUID? = nil, slices: Range<Int>, views: ArraySlice<simd_float4x4>,
                momentBlurRadius: Int? = nil, relativeProjections: [RelativeProjection] = []
            ) {
                self.lightID = lightID
                self.slices = slices
                self.momentBlurRadius = momentBlurRadius
                self.relativeProjections = relativeProjections
                projections = views.flatMap { [$0.columns.0, $0.columns.1, $0.columns.2, $0.columns.3] }
            }
        }

        let groups: [Group]

        init(groups: [Group]) {
            self.groups = groups.sorted { $0.slices.lowerBound < $1.slices.lowerBound }
        }

        func contains(slice: Int) -> Bool {
            groups.contains { $0.slices.contains(slice) }
        }
    }

    struct Use: Sendable {
        static let automatic = Use(writeSlices: nil)

        /// Nil writes every assigned slice, as the automatic path does.
        fileprivate let writeSlices: Set<Int>?
        fileprivate var cache: ShadowMapUpdateCache?
        fileprivate var ticket: UInt64 = 0

        var writesDepth: Bool { writeSlices?.isEmpty != true }
        var writesAll: Bool { writeSlices == nil }
        func writes(slice: Int) -> Bool { writeSlices?.contains(slice) ?? true }

        func didEncode() { cache?.didEncode(ticket) }
        func cancelBeforeAccess() { cache?.cancelBeforeAccess(ticket) }
    }

    private struct Revision: Equatable {
        let layout: Layout
        let request: UInt64
        let lightRequests: [UUID: UInt64]
        let automaticSlices: Set<Int>
    }

    private struct Active {
        weak var commandBuffer: (any MTLCommandBuffer)?
        let ticket: UInt64
        let revision: Revision
        var writeSlices: Set<Int>
        let publishesLayout: Bool
        var additionalSlices: Set<Int> = []
        var encoded = false

        var isReady: Bool { encoded && additionalSlices.isEmpty }
    }

    private let lock = NSLock()
    private let defaultMode: ShadowMapUpdateMode
    private var lightModes: [UUID: ShadowMapUpdateMode]
    private var request: UInt64 = 0
    private var lightRequests: [UUID: UInt64] = [:]
    private var nextTicket: UInt64 = 0
    private var completed: Revision?
    /// Keep the previous owners' slots after failed writes, but never read their invalid depth.
    private var invalidSlices: Set<Int> = []
    private var active: Active?

    init(defaultMode: ShadowMapUpdateMode = .onDemand, lightModes: [UUID: ShadowMapUpdateMode] = [:]) {
        self.defaultMode = defaultMode
        self.lightModes = lightModes
    }

    func updateMode(for lightID: UUID) -> ShadowMapUpdateMode {
        lock.withLock { lightModes[lightID] ?? defaultMode }
    }

    func setUpdateMode(_ mode: ShadowMapUpdateMode?, for lightID: UUID) {
        lock.withLock { lightModes[lightID] = mode }
    }

    func requestUpdate() {
        lock.withLock { request &+= 1 }
    }

    func requestUpdate(for lightID: UUID) -> Bool {
        lock.withLock {
            pruneRequests()
            guard knows(lightID) else { return false }
            lightRequests[lightID, default: 0] &+= 1
            return true
        }
    }

    func existingSlots(commandBuffer: any MTLCommandBuffer) -> [UUID: Int] {
        lock.withLock {
            let layout = if let active, active.isReady, active.commandBuffer === commandBuffer {
                active.revision.layout
            } else {
                completed?.layout
            }
            guard let layout else { return [:] }
            return Dictionary(uniqueKeysWithValues: layout.groups.compactMap { group in
                group.lightID.map { ($0, group.slices.lowerBound) }
            })
        }
    }

    func begin(_ layout: Layout, commandBuffer: any MTLCommandBuffer) throws -> Use {
        try lock.withLock {
            let ids = Set(layout.groups.compactMap(\.lightID))
            let automatic = layout.groups.filter { group in
                guard let id = group.lightID else { return false }
                return (lightModes[id] ?? defaultMode) == .automatic
            }
            let revision = Revision(
                layout: layout, request: request, lightRequests: lightRequests.filter { ids.contains($0.key) },
                automaticSlices: Set(automatic.flatMap(\.slices))
            )
            let slices = slicesToWrite(revision)
            if var active, let owner = active.commandBuffer {
                guard owner === commandBuffer, active.isReady, active.revision == revision,
                      active.publishesLayout || slices.isEmpty else {
                    throw ShadowMapUpdateError.commandBufferInUse
                }
                guard !revision.automaticSlices.isEmpty else { return Use(writeSlices: []) }
                // Manual slices already recorded in this buffer stay reusable.
                // Record the automatic passes again for the next scene use.
                active.additionalSlices = revision.automaticSlices
                self.active = active
                return Use(writeSlices: active.additionalSlices, cache: self, ticket: active.ticket)
            }

            return reserve(revision, writeSlices: slices, publishesLayout: true, commandBuffer: commandBuffer)
        }
    }

    func beginRead(slice: Int, commandBuffer: any MTLCommandBuffer) throws -> Use {
        try lock.withLock {
            if let active, let owner = active.commandBuffer {
                guard owner === commandBuffer, active.isReady else { throw ShadowMapUpdateError.commandBufferInUse }
                guard active.revision.layout.contains(slice: slice),
                      active.publishesLayout || !invalidSlices.contains(slice) else {
                    throw ShadowMapUpdateError.depthUnavailable
                }
                return Use(writeSlices: [])
            }
            guard let completed, completed.layout.contains(slice: slice), !invalidSlices.contains(slice) else {
                throw ShadowMapUpdateError.depthUnavailable
            }
            return reserve(completed, writeSlices: [], publishesLayout: false, commandBuffer: commandBuffer)
        }
    }

    private func slicesToWrite(_ revision: Revision) -> Set<Int> {
        let dirty = revision.layout.groups.filter { group in
            guard revision.automaticSlices.isDisjoint(with: group.slices),
                  let completed, completed.request == revision.request,
                  completed.layout.groups.contains(group), invalidSlices.isDisjoint(with: group.slices) else { return true }
            guard let id = group.lightID else { return false }
            return completed.lightRequests[id, default: 0] != revision.lightRequests[id, default: 0]
        }
        return Set(dirty.flatMap(\.slices))
    }

    private func knows(_ lightID: UUID) -> Bool {
        if active?.commandBuffer != nil, active?.revision.layout.groups.contains(where: { $0.lightID == lightID }) == true {
            return true
        }
        return completed?.layout.groups.contains { $0.lightID == lightID } == true
    }

    private func pruneRequests() {
        lightRequests = lightRequests.filter { knows($0.key) }
    }

    /// Called with the lock held. A weak owner disappears if an unsubmitted
    /// buffer was discarded. Live leases prevent out-of-order texture access.
    private func reserve(
        _ revision: Revision, writeSlices: Set<Int>, publishesLayout: Bool, commandBuffer: any MTLCommandBuffer
    ) -> Use {
        nextTicket &+= 1
        let ticket = nextTicket
        active = Active(
            commandBuffer: commandBuffer, ticket: ticket, revision: revision, writeSlices: writeSlices, publishesLayout: publishesLayout
        )
        commandBuffer.addCompletedHandler { [self] command in
            finish(ticket, succeeded: command.status == .completed)
        }
        return Use(writeSlices: writeSlices, cache: self, ticket: ticket)
    }

    private func didEncode(_ ticket: UInt64) {
        lock.withLock {
            guard var active, active.ticket == ticket else { return }
            active.encoded = true
            active.writeSlices.formUnion(active.additionalSlices)
            active.additionalSlices.removeAll()
            self.active = active
        }
    }

    private func cancelBeforeAccess(_ ticket: UInt64) {
        lock.withLock {
            guard var active, active.ticket == ticket else { return }
            if active.additionalSlices.isEmpty {
                self.active = nil
                pruneRequests()
            } else {
                active.additionalSlices.removeAll()
                self.active = active
            }
        }
    }

    private func finish(_ ticket: UInt64, succeeded: Bool) {
        lock.withLock {
            guard let active, active.ticket == ticket else { return }
            if active.publishesLayout, active.isReady, succeeded {
                completed = active.revision
                invalidSlices.removeAll()
            } else {
                invalidSlices.formUnion(active.writeSlices)
                invalidSlices.formUnion(active.additionalSlices)
            }
            self.active = nil
            pruneRequests()
        }
    }
}
