import simd

/// Optional per-face UV0 callbacks for an extrusion. Positions and normals use
/// the output mesh's coordinate system. Each corner's texture coordinate is the
/// default UV0 from the requested shape mapping.
public struct ShapeExtrusionUVGenerator: Sendable {
    public enum CapSide: Equatable, Sendable {
        /// The +Z cap of a straight extrusion or the end of a path sweep.
        case front
        /// The −Z cap of a straight extrusion or the start of a path sweep.
        case back
    }

    public struct Cap: Sendable {
        public let shapeIndex: Int
        public let side: CapSide
        public let triangleIndex: Int
        /// Corners in emitted triangle order.
        public let corners: (MeshVertex, MeshVertex, MeshVertex)
    }

    public struct Wall: Sendable {
        public let shapeIndex: Int
        /// The outer ring is zero. Holes follow in authored order.
        public let ringIndex: Int
        public let edgeIndex: Int
        /// Wall intervals count from the front cap, including bevel levels.
        public let levelIndex: Int
        /// Front start, front end, back end, back start along this outline edge.
        /// A collapsed bevel corner can repeat a position.
        public let corners: (MeshVertex, MeshVertex, MeshVertex, MeshVertex)
    }

    public typealias CapCallback = @Sendable (Cap) throws -> (SIMD2<Float>, SIMD2<Float>, SIMD2<Float>)
    public typealias WallCallback = @Sendable (Wall) throws -> (SIMD2<Float>, SIMD2<Float>, SIMD2<Float>, SIMD2<Float>)

    public let cap: CapCallback?
    public let wall: WallCallback?

    /// Return finite top-left UVs in the same corner order. Values may lie
    /// outside 0...1. A nil callback preserves that surface's default mapping.
    /// Callback errors propagate to the caller.
    public init(cap: CapCallback? = nil, wall: WallCallback? = nil) {
        self.cap = cap
        self.wall = wall
    }

    var hasCallbacks: Bool { cap != nil || wall != nil }
}

/// Collects coordinates in index order so the existing cap and wall emitters
/// keep their topology. Distinct coordinates at a shared source vertex split
/// at the seam before tangent generation.
final class ExtrusionUVBuilder {
    private let generator: ShapeExtrusionUVGenerator
    private var coordinates: [SIMD2<Float>] = []

    init?(_ generator: ShapeExtrusionUVGenerator) {
        guard generator.hasCallbacks else { return nil }
        self.generator = generator
    }

    func appendCap(
        _ face: [UInt32], vertices: [MeshVertex], shape: Int, front: Bool, triangle: Int
    ) throws {
        let corners = (vertices[Int(face[0])], vertices[Int(face[1])], vertices[Int(face[2])])
        let mapped = try generator.cap?(.init(
            shapeIndex: shape, side: front ? .front : .back, triangleIndex: triangle, corners: corners
        )) ?? (corners.0.textureCoordinate, corners.1.textureCoordinate, corners.2.textureCoordinate)
        coordinates += try validate([mapped.0, mapped.1, mapped.2], shape: shape)
    }

    func wallCoordinates(
        _ face: [UInt32], vertices: [MeshVertex], shape: Int, ring: Int, edge: Int, level: Int
    ) throws -> [SIMD2<Float>] {
        let corners = (vertices[Int(face[0])], vertices[Int(face[1])], vertices[Int(face[2])], vertices[Int(face[3])])
        let mapped = try generator.wall?(.init(
            shapeIndex: shape, ringIndex: ring, edgeIndex: edge, levelIndex: level, corners: corners
        )) ?? (corners.0.textureCoordinate, corners.1.textureCoordinate, corners.2.textureCoordinate, corners.3.textureCoordinate)
        return try validate([mapped.0, mapped.1, mapped.2, mapped.3], shape: shape)
    }

    func appendWall(_ uv: [SIMD2<Float>], corners: SIMD3<Int>) {
        coordinates += [uv[corners.x], uv[corners.y], uv[corners.z]]
    }

    func applying(to geometry: MeshGeometry) throws -> MeshGeometry {
        assert(coordinates.count == geometry.indices.count)
        let corners = zip(geometry.indices, coordinates)
        guard corners.contains(where: { geometry.vertices[Int($0)].textureCoordinate != $1 }) else { return geometry }

        struct Key: Hashable {
            let source: UInt32
            let coordinate: SIMD2<Float>
        }
        var vertices: [MeshVertex] = []
        var remaps: [Key: UInt32] = [:]
        let indices = corners.map { source, uv in
            let key = Key(source: source, coordinate: uv)
            if let previous = remaps[key] { return previous }
            let old = geometry.vertices[Int(source)]
            let index = UInt32(vertices.count)
            remaps[key] = index
            vertices.append(MeshVertex(
                position: old.position, normal: old.normal, textureCoordinate: uv,
                secondaryTextureCoordinate: old.secondaryTextureCoordinate, color: old.color
            ))
            return index
        }
        let mesh = try MeshGeometry(vertices: vertices, indices: indices).withGroups(geometry.groups)
        return try mesh.generatingTangents(convention: .directX)
    }

    private func validate(_ uv: [SIMD2<Float>], shape: Int) throws -> [SIMD2<Float>] {
        guard uv.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else {
            throw MeshGeometry.ShapeMeshError.invalidGeneratedUV(shape)
        }
        return uv
    }
}
