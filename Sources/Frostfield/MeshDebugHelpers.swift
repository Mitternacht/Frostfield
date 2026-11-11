import Foundation
import simd

public extension LineGeometry {
    /// One world-space normal segment per authored vertex, including unused
    /// vertices. The inverse transpose preserves normal directions under
    /// nonuniform scale. Add the result at the scene root.
    static func vertexNormals(
        of geometry: MeshGeometry, worldFromObject: simd_float4x4 = matrix_identity_float4x4,
        size: Float = 1, color: SIMD4<Float> = LineGeometry.helperColor(0xff0000)
    ) throws -> LineGeometry {
        try vertexDirections(of: geometry, worldFromObject: worldFromObject, size: size, color: color, tangents: false)
    }

    /// One world-space tangent segment per authored vertex. Directions use the
    /// linear transform and ignore tangent handedness, which describes the
    /// bitangent. Every vertex needs an authored tangent.
    static func vertexTangents(
        of geometry: MeshGeometry, worldFromObject: simd_float4x4 = matrix_identity_float4x4,
        size: Float = 1, color: SIMD4<Float> = LineGeometry.helperColor(0x00ffff)
    ) throws -> LineGeometry {
        try vertexDirections(of: geometry, worldFromObject: worldFromObject, size: size, color: color, tangents: true)
    }

    /// Snapshots the selected ordinary mesh with its resolved hierarchy
    /// transform. Returns nil for a missing node or one without a mesh.
    static func vertexNormals(
        of node: UUID, in graph: SceneGraph, size: Float = 1,
        color: SIMD4<Float> = LineGeometry.helperColor(0xff0000)
    ) throws -> LineGeometry? {
        guard let source = try vertexDirectionSource(of: node, in: graph) else { return nil }
        return try vertexNormals(of: source.geometry, worldFromObject: source.world, size: size, color: color)
    }

    /// Snapshots the selected ordinary mesh's authored tangents with its
    /// resolved hierarchy transform. Rebuild the lines after the mesh moves.
    static func vertexTangents(
        of node: UUID, in graph: SceneGraph, size: Float = 1,
        color: SIMD4<Float> = LineGeometry.helperColor(0x00ffff)
    ) throws -> LineGeometry? {
        guard let source = try vertexDirectionSource(of: node, in: graph) else { return nil }
        return try vertexTangents(of: source.geometry, worldFromObject: source.world, size: size, color: color)
    }

    private static func vertexDirectionSource(
        of id: UUID, in graph: SceneGraph
    ) throws -> (geometry: MeshGeometry, world: simd_float4x4)? {
        guard let node = try graph.resolve().first(where: { $0.id == id }) else { return nil }
        guard node.batch == nil, node.instances == nil else { throw MeshDebugFailure.unsupportedNode(id) }
        guard let renderer = node.mesh else { return nil }
        if case .rawMetal = renderer.material { throw MeshDebugFailure.unsupportedNode(id) }
        guard renderer.skin == nil, renderer.morphTargets == nil,
              renderer.pathTracingMaterial?.surfaceGraph?.hasVertexOutputs != true else {
            throw MeshDebugFailure.unsupportedNode(id)
        }
        return (renderer.geometry, node.worldTransform)
    }

    private static func vertexDirections(
        of geometry: MeshGeometry, worldFromObject: simd_float4x4,
        size: Float, color: SIMD4<Float>, tangents: Bool
    ) throws -> LineGeometry {
        guard size.isFinite, size > 0 else { throw PrimitiveGeometryError.invalidHelperParameter("vertex helper size") }
        guard isValidPrimitiveColor(color) else { throw PrimitiveGeometryError.invalidColor(0) }
        guard geometry.vertices.count <= maximumPointCount / 2 else { throw PrimitiveGeometryError.pointCount }
        guard let transform = BoundsAffine(worldFromObject) else { throw MeshDebugFailure.transform }
        let determinant = simd_determinant(transform.linear)
        guard determinant.isFinite, determinant != 0 else { throw MeshDebugFailure.transform }
        let directions = tangents ? transform.linear : transform.linear.inverse.transpose
        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(geometry.vertices.count * 2)
        for (index, vertex) in geometry.vertices.enumerated() {
            let attribute: SIMD3<Float>
            if tangents {
                guard let tangent = vertex.tangent else { throw MeshDebugFailure.missingTangent(index) }
                attribute = SIMD3(tangent.x, tangent.y, tangent.z)
            } else {
                attribute = vertex.normal
            }
            let origin = transform.linear * SIMD3<Double>(vertex.position) + transform.translation
            let direction = directions * SIMD3<Double>(attribute)
            let scale = simd_reduce_max(abs(direction))
            guard scale.isFinite, scale > 0 else { throw MeshDebugFailure.transform }
            positions.append(SIMD3<Float>(origin))
            positions.append(SIMD3<Float>(origin + simd_normalize(direction / scale) * Double(size)))
        }
        return try LineGeometry(
            positions: positions, colors: Array(repeating: color, count: positions.count), topology: .segments
        )
    }
}

/// Triangle contours and annotations of authored UVs in a unit layout.
/// The layout preserves UVs outside the unit square so tiled and
/// out-of-range charts remain visible when the view includes them.
public struct UVLayoutDebug: Sendable {
    public struct Label: Equatable, Sendable {
        public enum Kind: Equatable, Sendable {
            case face
            /// The corner's order in the face (zero to two) and its source UV index.
            case corner(order: Int, vertex: UInt32)
        }

        public let triangle: Int
        public let kind: Kind
        /// Normalized top-left UV coordinates. Multiply by the display width
        /// and height to place a pixel overlay at this anchor.
        public let position: SIMD2<Float>

        public var text: String {
            switch kind {
            case .face:
                return String(triangle)
            case .corner(let order, let vertex):
                let letter = switch order {
                case 0: "a"
                case 1: "b"
                default: "c"
                }
                return "\(letter)\(vertex)"
            }
        }

        public var color: SIMD4<Float> {
            switch kind {
            case .face: LineGeometry.helperColor(0x3f3f3f)
            case .corner: LineGeometry.helperColor(0xbfbfbf)
            }
        }
    }

    /// Triangle edges at (u, 1 − v, 0) so top-left UVs draw top-left
    /// when viewed along −Z. Use `DebugHelpers.node` to render the layout.
    public let lines: LineGeometry
    /// The face number at each centroid, then three corner labels halfway
    /// toward each vertex in index order.
    public let labels: [Label]

    public init(
        geometry: MeshGeometry, coordinateSet: MeshTextureCoordinateSet = .primary,
        color: SIMD4<Float> = LineGeometry.helperColor(0x3f3f3f)
    ) throws {
        let coordinates = try geometry.vertices.enumerated().map { index, vertex in
            if coordinateSet == .primary { return vertex.textureCoordinate }
            guard let uv = vertex.secondaryTextureCoordinate else { throw MeshDebugFailure.missingSecondaryUV(index) }
            return uv
        }
        try self.init(coordinates: coordinates, indices: geometry.indices, color: color)
    }

    /// Reads triangles in index order. With nil indices each consecutive
    /// triple of UV coordinates forms one triangle.
    public init(
        coordinates: [SIMD2<Float>], indices: [UInt32]? = nil,
        color: SIMD4<Float> = LineGeometry.helperColor(0x3f3f3f)
    ) throws {
        let count = indices?.count ?? coordinates.count
        guard count > 0, count.isMultiple(of: 3) else { throw MeshGeometry.ValidationError.invalidTriangleIndexCount }
        guard count <= LineGeometry.maximumPointCount / 2 else { throw PrimitiveGeometryError.pointCount }
        guard isValidPrimitiveColor(color) else { throw PrimitiveGeometryError.invalidColor(0) }
        if let invalid = coordinates.firstIndex(where: { !$0.x.isFinite || !$0.y.isFinite }) {
            throw MeshDebugFailure.invalidUV(invalid)
        }
        if let invalid = indices?.first(where: { Int($0) >= coordinates.count }) {
            throw MeshGeometry.ValidationError.indexOutOfBounds(invalid)
        }

        func vertex(_ index: Int) -> UInt32 { indices?[index] ?? UInt32(index) }
        func point(_ uv: SIMD2<Double>) -> SIMD3<Float> { SIMD3(Float(uv.x), Float(1 - uv.y), 0) }
        var positions: [SIMD3<Float>] = []
        var labels: [Label] = []
        positions.reserveCapacity(count * 2)
        labels.reserveCapacity(count / 3 * 4)
        for offset in stride(from: 0, to: count, by: 3) {
            let face = SIMD3(vertex(offset), vertex(offset + 1), vertex(offset + 2))
            let a = SIMD2<Double>(coordinates[Int(face.x)])
            let b = SIMD2<Double>(coordinates[Int(face.y)])
            let c = SIMD2<Double>(coordinates[Int(face.z)])
            let p = point(a), q = point(b), r = point(c)
            positions += [p, q, q, r, r, p]
            let triangle = offset / 3
            let center = (a + b + c) / 3
            labels.append(Label(triangle: triangle, kind: .face, position: SIMD2<Float>(center)))
            for (order, uv) in [a, b, c].enumerated() {
                labels.append(Label(
                    triangle: triangle, kind: .corner(order: order, vertex: face[order]),
                    position: SIMD2<Float>((center + uv) * 0.5)
                ))
            }
        }
        self.lines = try LineGeometry(
            positions: positions, colors: Array(repeating: color, count: positions.count), topology: .segments
        )
        self.labels = labels
    }
}

public enum MeshDebugFailure: LocalizedError, Equatable {
    case transform
    case unsupportedNode(UUID)
    case missingTangent(Int)
    case missingSecondaryUV(Int)
    case invalidUV(Int)

    public var errorDescription: String? {
        switch self {
        case .transform: "Vertex helpers need a finite, invertible affine world transform."
        case .unsupportedNode(let id): "Vertex helper node \(id) needs ordinary mesh geometry and stock vertex inputs."
        case .missingTangent(let index): "Vertex \(index) needs an authored tangent."
        case .missingSecondaryUV(let index): "Vertex \(index) needs a secondary UV coordinate."
        case .invalidUV(let index): "UV coordinate \(index) must be finite."
        }
    }
}
