public extension MeshBVH {
    /// A bounds classifier's decision for a shapecast node.
    enum ShapeCastClassification: Sendable {
        /// Skip this node and every triangle under it.
        case outside
        /// Descend to children, or offer this leaf's triangles for testing.
        case intersecting
        /// Offer all triangles in this subtree as one contained range.
        case contained
    }

    /// A node snapshot in the mesh's local coordinates.
    struct ShapeCastNode: Equatable, Sendable {
        public let bounds: AxisAlignedBox
        public let isLeaf: Bool
        /// Depth in the hierarchy, with the root at one.
        public let depth: Int
        /// An identifier within this tree, preserved across refits.
        public let index: Int
    }

    /// Source triangle indices in the hierarchy's order. Collection offsets
    /// start at zero. Indices keep their original geometry meaning.
    struct ShapeCastRange: RandomAccessCollection, Sendable {
        private let order: ArraySlice<UInt32>

        init(_ order: ArraySlice<UInt32>) { self.order = order }

        public var startIndex: Int { 0 }
        public var endIndex: Int { order.count }

        public func index(after i: Int) -> Int { i + 1 }
        public func index(before i: Int) -> Int { i - 1 }

        public subscript(position: Int) -> Int {
            precondition(position >= 0 && position < order.count)
            return Int(order[order.startIndex + position])
        }
    }
}
