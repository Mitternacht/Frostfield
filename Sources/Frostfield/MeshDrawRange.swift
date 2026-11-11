import Foundation

/// A window into a prepared geometry's indices or source vertices.
public struct GeometryDrawRange: Hashable, Sendable {
    public static let all = Self(validatedStart: 0, count: nil)
    public let start: Int
    /// Nil selects all remaining elements. An incomplete trailing primitive is omitted.
    public let count: Int?

    public init(start: Int, count: Int? = nil) throws {
        guard start >= 0, count.map({ $0 >= 0 }) != false else { throw Failure.negative }
        self.init(validatedStart: start, count: count)
    }

    private init(validatedStart: Int, count: Int?) {
        start = validatedStart
        self.count = count
    }

    func resolve(indexCount: Int) -> Range<Int> {
        resolve(elementCount: indexCount, elementsPerPrimitive: 3)
    }

    func resolve(elementCount: Int, elementsPerPrimitive: Int = 1) -> Range<Int> {
        let first = min(start, elementCount)
        let available = elementCount - first
        let length = min(count ?? available, available)
        return first..<(first + length - length % elementsPerPrimitive)
    }

    static func intersection(of selected: Range<Int>, in group: Range<Int>) -> Self {
        let start = min(group.upperBound, max(group.lowerBound, selected.lowerBound))
        let end = max(start, min(group.upperBound, selected.upperBound))
        if start == group.lowerBound, end == group.upperBound { return .all }
        return Self(validatedStart: start - group.lowerBound, count: end - start)
    }

    public enum Failure: LocalizedError, Equatable {
        case negative

        public var errorDescription: String? { "Geometry draw range start and count must be nonnegative." }
    }
}

/// An index window for an indexed triangle mesh.
public typealias MeshDrawRange = GeometryDrawRange
