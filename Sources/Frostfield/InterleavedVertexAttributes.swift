import Foundation

/// Packs separately stored vertex channels into one interleaved float buffer,
/// like Three.js `interleaveAttributes`.
public struct InterleavedVertexAttributes: Equatable, Sendable {
    public struct Channel: Equatable, Sendable {
        public var name: String
        public var components: Int
        public var values: [Float]

        public init(name: String, components: Int, values: [Float]) {
            self.name = name
            self.components = components
            self.values = values
        }
    }

    public struct Layout: Equatable, Sendable {
        public var name: String
        /// Float index of this channel inside one vertex.
        public var offset: Int
        public var components: Int
    }

    public let vertexCount: Int
    /// Floats per vertex.
    public let stride: Int
    public let layout: [Layout]
    /// Vertex-major floats: vertex 0's channels, then vertex 1's, in layout order.
    public let values: [Float]

    public init(_ channels: [Channel]) throws {
        guard let first = channels.first else { throw Failure.empty }
        guard channels.allSatisfy({ !$0.name.isEmpty }) else { throw Failure.emptyName }
        var seen = Set<String>()
        for channel in channels where !seen.insert(channel.name).inserted {
            throw Failure.duplicateName(channel.name)
        }
        guard (1...4).contains(first.components), first.values.count.isMultiple(of: first.components), !first.values.isEmpty else {
            throw Failure.components(first.name)
        }
        let count = first.values.count / first.components
        var layouts: [Layout] = []
        var cursor = 0
        for channel in channels {
            guard (1...4).contains(channel.components), channel.values.count == count * channel.components else {
                throw channel.components < 1 || channel.components > 4 ? Failure.components(channel.name) : Failure.count(channel.name)
            }
            guard channel.values.allSatisfy(\.isFinite) else { throw Failure.nonfinite(channel.name) }
            layouts.append(Layout(name: channel.name, offset: cursor, components: channel.components))
            cursor += channel.components
        }
        var packed = [Float](repeating: 0, count: count * cursor)
        for layout in layouts {
            let source = channels.first { $0.name == layout.name }!
            for vertex in 0..<count {
                for component in 0..<layout.components {
                    packed[vertex * cursor + layout.offset + component] = source.values[vertex * layout.components + component]
                }
            }
        }
        vertexCount = count
        stride = cursor
        layout = layouts
        values = packed
    }

    /// The channel's values in vertex order, not interleaved.
    public func values(named name: String) throws -> [Float] {
        guard let channel = layout.first(where: { $0.name == name }) else { throw Failure.missing(name) }
        var extracted = [Float]()
        extracted.reserveCapacity(vertexCount * channel.components)
        for vertex in 0..<vertexCount {
            let start = vertex * stride + channel.offset
            extracted.append(contentsOf: values[start..<(start + channel.components)])
        }
        return extracted
    }

    public enum Failure: LocalizedError, Equatable {
        case empty
        case emptyName
        case duplicateName(String)
        case components(String)
        case count(String)
        case nonfinite(String)
        case missing(String)

        public var errorDescription: String? {
            switch self {
            case .empty: "Interleaving needs at least one vertex channel."
            case .emptyName: "An interleaved vertex channel needs a name."
            case .duplicateName(let name): "Vertex channel \(name) is listed more than once."
            case .components(let name): "Vertex channel \(name) needs one to four components and a matching value count."
            case .count(let name): "Vertex channel \(name) has a different vertex count from the first channel."
            case .nonfinite(let name): "Vertex channel \(name) must be finite."
            case .missing(let name): "Interleaved attributes have no channel named \(name)."
            }
        }
    }
}

public extension MeshGeometry {
    /// Position, normal and primary UV as an interleaved float buffer.
    func interleavedAttributes() throws -> InterleavedVertexAttributes {
        try InterleavedVertexAttributes([
            .init(name: "position", components: 3, values: vertices.flatMap { [$0.position.x, $0.position.y, $0.position.z] }),
            .init(name: "normal", components: 3, values: vertices.flatMap { [$0.normal.x, $0.normal.y, $0.normal.z] }),
            .init(name: "uv", components: 2, values: vertices.flatMap { [$0.textureCoordinate.x, $0.textureCoordinate.y] }),
        ])
    }
}
