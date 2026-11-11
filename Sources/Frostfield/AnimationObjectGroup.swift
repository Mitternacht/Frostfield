import Foundation

/// Several objects or subtrees that share a mixer's playback state.
///
/// Each member maps the same template nodes to its own scene IDs. Add, remove,
/// or uncache members on the same serialized owner that advances and samples
/// the mixer. These operations keep the group's playing actions intact.
public final class AnimationObjectGroup {
    public let sourceNodes: Set<UUID>
    public private(set) var activeMembers: Set<UUID> = []
    public var memberCount: Int { mappings.count }
    public var cachedMemberCount: Int { mappings.count - activeMembers.count }

    private var mappings: [UUID: [UUID: UUID]] = [:]
    private var owners: [UUID: (member: UUID, source: UUID)] = [:]
    private var destinations: [UUID: [UUID]] = [:]

    public init(sourceNodes: Set<UUID>) throws {
        guard !sourceNodes.isEmpty else { throw Failure.emptyTemplate }
        self.sourceNodes = sourceNodes
    }

    /// Adds a member or reactivates its cached mapping. Use a stable identifier
    /// for the member, typically its root node ID. Every source and destination
    /// occurs once per mapping. Destinations belong to only one member, including
    /// cached members. Uncache a member before replacing its mapping.
    public func add(member: UUID, mapping: [UUID: UUID]) throws {
        if let existing = mappings[member] {
            guard existing == mapping else { throw Failure.changedMapping(member) }
        } else {
            guard Set(mapping.keys) == sourceNodes, Set(mapping.values).count == sourceNodes.count else {
                throw Failure.invalidMapping(member)
            }
            if let duplicate = mapping.values.first(where: { owners[$0] != nil }) {
                throw Failure.duplicateDestination(duplicate)
            }
            mappings[member] = mapping
            for (source, destination) in mapping { owners[destination] = (member, source) }
        }
        guard activeMembers.insert(member).inserted else { return }
        for (source, destination) in mapping { destinations[source, default: []].append(destination) }
    }

    /// Stops applying group animation to a member while retaining its mapping.
    public func remove(member: UUID) {
        guard activeMembers.remove(member) != nil, let mapping = mappings[member] else { return }
        for (source, destination) in mapping { destinations[source]?.removeAll { $0 == destination } }
    }

    /// Removes a member and releases its mapping and destination IDs.
    public func uncache(member: UUID) {
        remove(member: member)
        guard let mapping = mappings.removeValue(forKey: member) else { return }
        for destination in mapping.values { owners.removeValue(forKey: destination) }
    }

    func destinationNodes(for source: UUID) -> [UUID] { destinations[source] ?? [] }

    func sourceNode(for destination: UUID) -> UUID? {
        guard let owner = owners[destination], activeMembers.contains(owner.member) else { return nil }
        return owner.source
    }

    public enum Failure: LocalizedError, Equatable {
        case emptyTemplate
        case invalidMapping(UUID)
        case changedMapping(UUID)
        case duplicateDestination(UUID)
        case unknownClipNode(UUID)

        public var errorDescription: String? {
            switch self {
            case .emptyTemplate: "An animation group needs at least one template node."
            case .invalidMapping(let id): "Member \(id) must map each template node to one distinct scene node."
            case .changedMapping(let id): "Uncache animation group member \(id) before changing its mapping."
            case .duplicateDestination(let id): "Scene node \(id) already belongs to an animation group member."
            case .unknownClipNode(let id): "The animation group template does not contain clip node \(id)."
            }
        }
    }
}
