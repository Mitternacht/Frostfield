import Foundation
import simd

/// Polygons over shared vertex positions, with face-varying texture coordinates and
/// edge creases: the input and output of subdivision surfaces.
///
/// Faces list position indices counterclockwise when viewed from the front, like USD
/// `faceVertexCounts` and `faceVertexIndices`. Texture coordinates are face varying:
/// each face corner has its own index into a value array, so corners that share a
/// position can carry different coordinates across a UV seam.
public struct PolygonMesh: Sendable {
    /// Face-varying values: one value index per face corner, in face order.
    public struct FaceVaryingChannel: Sendable, Equatable {
        public let values: [SIMD2<Float>]
        public let indices: [UInt32]

        public init(values: [SIMD2<Float>], indices: [UInt32]) {
            self.values = values
            self.indices = indices
        }
    }

    /// Linear RGBA values with one value index per face corner.
    public struct ColorChannel: Sendable, Equatable {
        public let values: [SIMD4<Float>]
        public let indices: [UInt32]

        public init(values: [SIMD4<Float>], indices: [UInt32]) {
            self.values = values
            self.indices = indices
        }
    }

    /// A contiguous face range with one material slot.
    public struct FaceGroup: Sendable, Equatable {
        public let start: Int
        public let count: Int
        public let materialIndex: Int

        public init(start: Int, count: Int, materialIndex: Int = 0) {
            self.start = start
            self.count = count
            self.materialIndex = materialIndex
        }
    }

    /// A sharp edge between two positions. Sharpness `s` applies sharp rules for `⌊s⌋`
    /// subdivision levels and blends by the fraction, as in DeRose, Kass and Truong 1998;
    /// `Float.infinity` keeps the edge sharp at every level.
    public struct Crease: Sendable, Hashable {
        public let first: UInt32
        public let second: UInt32
        public let sharpness: Float

        public init(_ first: UInt32, _ second: UInt32, sharpness: Float = .infinity) {
            self.first = first
            self.second = second
            self.sharpness = sharpness
        }
    }

    public let positions: [SIMD3<Float>]
    /// The number of corners in each face, at least three.
    public let faceVertexCounts: [Int]
    public let faceVertexIndices: [UInt32]
    public let textureCoordinates: FaceVaryingChannel?
    public let secondaryTextureCoordinates: FaceVaryingChannel?
    public let colors: ColorChannel?
    /// Material groups in face order. Gaps between groups remain ungrouped.
    public let faceGroups: [FaceGroup]
    /// Creases on edges of the mesh. Open boundaries are always sharp and need no crease.
    public let creases: [Crease]
    /// Positions that stay fixed under subdivision.
    public let corners: [UInt32]

    /// Validates finite positions and coordinates, faces of three or more distinct
    /// corners, in-range indices, one channel index per corner, creases on existing edges
    /// with positive sharpness, and in-range corners.
    public init(
        positions: [SIMD3<Float>], faceVertexCounts: [Int], faceVertexIndices: [UInt32],
        textureCoordinates: FaceVaryingChannel? = nil, secondaryTextureCoordinates: FaceVaryingChannel? = nil,
        colors: ColorChannel? = nil, faceGroups: [FaceGroup] = [], creases: [Crease] = [], corners: [UInt32] = []
    ) throws {
        typealias Failure = MeshGeometry.GeometryProcessingError
        guard !positions.isEmpty, positions.allSatisfy(\.isFiniteVector) else {
            throw Failure.invalidPolygonMesh("positions must be finite and nonempty")
        }
        guard positions.count <= Int(Int32.max) else { throw Failure.tooManyVertices }
        guard faceVertexIndices.count <= Int(Int32.max) else { throw Failure.tooManyTriangles }
        guard !faceVertexCounts.isEmpty else {
            throw Failure.invalidPolygonMesh("a mesh needs at least one face")
        }
        guard faceVertexIndices.allSatisfy({ Int($0) < positions.count }) else {
            throw Failure.invalidPolygonMesh("a face index is outside the positions")
        }
        var start = 0
        var edges = Set<UInt64>()
        for count in faceVertexCounts {
            guard count >= 3, count <= faceVertexIndices.count - start else {
                throw Failure.invalidPolygonMesh("each face needs at least three corners, and the counts must match the indices")
            }
            let face = faceVertexIndices[start..<(start + count)]
            guard Set(face.map { positions[Int($0)].exactKey }).count == count else {
                throw Failure.invalidPolygonMesh("a face repeats a position")
            }
            for k in 0..<count {
                edges.insert(Self.edgeKey(face[start + k], face[start + (k + 1) % count]))
            }
            start += count
        }
        guard start == faceVertexIndices.count else {
            throw Failure.invalidPolygonMesh("the face counts must cover all indices")
        }
        for channel in [textureCoordinates, secondaryTextureCoordinates].compactMap({ $0 }) {
            guard channel.values.count <= Int(Int32.max), channel.indices.count == faceVertexIndices.count,
                  channel.indices.allSatisfy({ Int($0) < channel.values.count }),
                  channel.values.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else {
                throw Failure.invalidPolygonMesh("texture coordinates need finite values and one in-range index per corner")
            }
        }
        if let colors {
            guard colors.values.count <= Int(Int32.max), colors.indices.count == faceVertexIndices.count,
                  colors.indices.allSatisfy({ Int($0) < colors.values.count }),
                  colors.values.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite && $0.w.isFinite }) else {
                throw Failure.invalidPolygonMesh("colors need finite RGBA values and one in-range index per corner")
            }
        }
        let sortedGroups = faceGroups.sorted { $0.start < $1.start }
        var groupEnd = 0
        for group in sortedGroups {
            guard group.start >= groupEnd, group.start <= faceVertexCounts.count, group.count > 0,
                  group.count <= faceVertexCounts.count - group.start, group.materialIndex >= 0 else {
                throw Failure.invalidPolygonMesh("face groups need valid, nonoverlapping ranges and nonnegative material indices")
            }
            groupEnd = group.start + group.count
        }
        for crease in creases {
            guard crease.sharpness > 0, !crease.sharpness.isNaN,
                  edges.contains(Self.edgeKey(crease.first, crease.second)) else {
                throw Failure.invalidPolygonMesh("a crease needs positive sharpness on an edge of the mesh")
            }
        }
        guard corners.allSatisfy({ Int($0) < positions.count }) else {
            throw Failure.invalidPolygonMesh("a corner is outside the positions")
        }
        self.positions = positions
        self.faceVertexCounts = faceVertexCounts
        self.faceVertexIndices = faceVertexIndices
        self.textureCoordinates = textureCoordinates
        self.secondaryTextureCoordinates = secondaryTextureCoordinates
        self.colors = colors
        self.faceGroups = sortedGroups
        self.creases = creases
        self.corners = corners
    }

    /// Triangles of `geometry` as faces over welded positions. Vertices with equal
    /// positions share a position, and texture coordinates become face-varying values
    /// with seams where coordinates differ. Zero-area faces are discarded before
    /// welding. Secondary coordinates must be present on all referenced vertices
    /// when used. Colors and material groups follow the surviving faces. Vertices
    /// without an authored color use white when the mesh has colors. With
    /// `hardEdgesAsCreases`, opposite faces with mismatching normals form sharp edges.
    public init(_ input: MeshGeometry, hardEdgesAsCreases: Bool = false) throws {
        guard input.vertices.count <= Int(Int32.max) else { throw MeshGeometry.GeometryProcessingError.tooManyVertices }
        guard input.indices.count <= Int(Int32.max) else { throw MeshGeometry.GeometryProcessingError.tooManyTriangles }
        let originalTopology = EdgeTopology(input)
        let geometry: MeshGeometry
        let topology: EdgeTopology
        if originalTopology.faceNormals.contains(.zero) {
            var indices: [UInt32] = []
            var offsets = [0]
            for (face, normal) in originalTopology.faceNormals.enumerated() {
                if normal != .zero { indices.append(contentsOf: input.indices[(3 * face)..<(3 * face + 3)]) }
                if !input.groups.isEmpty { offsets.append(indices.count) }
            }
            guard !indices.isEmpty else { throw MeshGeometry.GeometryProcessingError.noTrianglesRemain }
            geometry = try MeshGeometry(vertices: input.vertices, indices: indices)
                .withGroups(MeshGeometry.remappedGroups(input.groups, offsets: offsets))
            topology = EdgeTopology(geometry)
        } else {
            geometry = input
            topology = originalTopology
        }
        var positionIndex = [UInt32](repeating: .max, count: geometry.vertices.count)
        var positions: [SIMD3<Float>] = []
        for vertex in geometry.vertices.indices {
            let id = Int(topology.positionID[vertex])
            if positionIndex[id] == .max {
                positionIndex[id] = UInt32(positions.count)
                positions.append(geometry.vertices[id].position)
            }
            positionIndex[vertex] = positionIndex[id]
        }
        let faceIndices = geometry.indices.map { positionIndex[Int($0)] }

        func channel<Value: Hashable>(_ value: (MeshVertex) -> Value) -> (values: [Value], indices: [UInt32]) {
            var lookup: [ChannelKey<Value>: UInt32] = [:]
            var values: [Value] = []
            var indices: [UInt32] = []
            indices.reserveCapacity(geometry.indices.count)
            for index in geometry.indices {
                let attribute = value(geometry.vertices[Int(index)])
                let key = ChannelKey(position: positionIndex[Int(index)], value: attribute)
                if let existing = lookup[key] {
                    indices.append(existing)
                } else {
                    let entry = UInt32(values.count)
                    lookup[key] = entry
                    indices.append(entry)
                    values.append(attribute)
                }
            }
            return (values, indices)
        }
        let primary = channel(\.textureCoordinate)
        var secondary: FaceVaryingChannel?
        if geometry.indices.contains(where: { geometry.vertices[Int($0)].secondaryTextureCoordinate != nil }) {
            guard geometry.indices.allSatisfy({ geometry.vertices[Int($0)].secondaryTextureCoordinate != nil }) else {
                throw MeshGeometry.GeometryProcessingError.invalidPolygonMesh("referenced vertices must agree on whether UV1 is present")
            }
            let uv1 = channel { $0.secondaryTextureCoordinate ?? .zero }
            secondary = FaceVaryingChannel(values: uv1.values, indices: uv1.indices)
        }
        var colors: ColorChannel?
        if geometry.hasVertexColors {
            let rgba = channel { $0.color ?? SIMD4(repeating: 1) }
            colors = ColorChannel(values: rgba.values, indices: rgba.indices)
        }
        let groups = geometry.groups.map { FaceGroup(start: $0.start / 3, count: $0.count / 3, materialIndex: $0.materialIndex) }

        var creases: [Crease] = []
        if hardEdgesAsCreases {
            for corner in 0..<geometry.indices.count where topology.isConsistentManifold(corner) {
                let other = Int(topology.neighbor[corner])
                guard corner < other else { continue }
                let a = geometry.vertices[Int(geometry.indices[corner])].normal
                let b = geometry.vertices[Int(geometry.indices[EdgeTopology.next(corner)])].normal
                let a2 = geometry.vertices[Int(geometry.indices[EdgeTopology.next(other)])].normal
                let b2 = geometry.vertices[Int(geometry.indices[other])].normal
                if a != a2 || b != b2 {
                    creases.append(Crease(faceIndices[corner], faceIndices[EdgeTopology.next(corner)]))
                }
            }
        }
        try self.init(
            positions: positions, faceVertexCounts: Array(repeating: 3, count: geometry.indices.count / 3),
            faceVertexIndices: faceIndices, textureCoordinates: FaceVaryingChannel(values: primary.values, indices: primary.indices),
            secondaryTextureCoordinates: secondary, colors: colors, faceGroups: groups, creases: creases
        )
    }

    private struct ChannelKey<Value: Hashable>: Hashable {
        let position: UInt32
        let value: Value
    }

    static func edgeKey(_ a: UInt32, _ b: UInt32) -> UInt64 {
        UInt64(min(a, b)) << 32 | UInt64(max(a, b))
    }

    var faceStarts: [Int] {
        var starts = [0]
        starts.reserveCapacity(faceVertexCounts.count + 1)
        for count in faceVertexCounts { starts.append(starts[starts.count - 1] + count) }
        return starts
    }

    /// Triangulates each face in the plane of its dominant normal component and computes
    /// corner-angle-weighted normals. Creases with sharpness one or more split normals.
    /// Concave faces retain winding. Texture coordinates, colors and material ranges are kept.
    /// Throws when a face cannot be triangulated in its projection.
    public func triangulated() throws -> MeshGeometry {
        let starts = faceStarts
        var triangleCorners: [Int] = []
        triangleCorners.reserveCapacity((faceVertexIndices.count - faceVertexCounts.count) * 3)
        var triangleOffsets = [0]
        for face in faceVertexCounts.indices {
            let start = starts[face], count = faceVertexCounts[face]
            let local = try triangulateFace(start: start, count: count)
            triangleCorners += local.map { start + $0 }
            triangleOffsets.append(triangleCorners.count)
        }
        let groups = faceGroups.map { group in
            let start = triangleOffsets[group.start]
            let count = triangleOffsets[group.start + group.count] - start
            return MeshGeometryGroup(start: start, count: count, materialIndex: group.materialIndex)
        }
        // Temporary vertices: one per distinct (position, coordinate indices) combination.
        var vertexKey: [SIMD4<UInt32>: UInt32] = [:]
        var vertices: [MeshVertex] = []
        var positionOfVertex: [UInt32] = []
        var indices: [UInt32] = []
        indices.reserveCapacity(triangleCorners.count)
        for corner in triangleCorners {
            let position = faceVertexIndices[corner]
            let uv = textureCoordinates?.indices[corner] ?? .max
            let uv2 = secondaryTextureCoordinates?.indices[corner] ?? .max
            let color = colors?.indices[corner] ?? .max
            let key = SIMD4(position, uv, uv2, color)
            if let existing = vertexKey[key] {
                indices.append(existing)
                continue
            }
            guard vertices.count < Int(UInt32.max) else { throw MeshGeometry.GeometryProcessingError.tooManyVertices }
            vertexKey[key] = UInt32(vertices.count)
            indices.append(UInt32(vertices.count))
            positionOfVertex.append(position)
            vertices.append(MeshVertex(
                position: positions[Int(position)], normal: SIMD3(0, 0, 1),
                textureCoordinate: textureCoordinates.map { $0.values[Int(uv)] } ?? .zero,
                secondaryTextureCoordinate: secondaryTextureCoordinates.map { $0.values[Int(uv2)] },
                color: colors.map { $0.values[Int(color)] }
            ))
        }
        let geometry = try MeshGeometry(vertices: vertices, indices: indices).withGroups(groups)
        var sharp = Set<UInt64>()
        for crease in creases where crease.sharpness >= 1 { sharp.insert(Self.edgeKey(crease.first, crease.second)) }
        let topology = EdgeTopology(geometry)
        let normals = SmoothingGroups(geometry: geometry, topology: topology) { corner, _ in
            guard !sharp.isEmpty else { return false }
            let a = positionOfVertex[Int(indices[corner])]
            let b = positionOfVertex[Int(indices[EdgeTopology.next(corner)])]
            return sharp.contains(Self.edgeKey(a, b))
        }
        return try normals.remapped(keepingNormalsOfUnsplitVertices: false).geometry
    }

    /// Local triangle corners, preserving the face's orientation.
    private func triangulateFace(start: Int, count: Int) throws -> [Int] {
        let points = faceVertexIndices[start..<(start + count)].map { SIMD3<Double>(positions[Int($0)]) }
        let origin = points[0]
        var normal = SIMD3<Double>.zero
        for k in 1..<(count - 1) {
            normal += simd_cross(points[k] - origin, points[k + 1] - origin)
        }
        guard normal != .zero else { throw MeshGeometry.GeometryProcessingError.invalidPolygonMesh("a face has no area") }
        if count == 3 { return [0, 1, 2] }
        let absolute = simd_abs(normal)
        let axis = absolute.x >= absolute.y && absolute.x >= absolute.z ? 0 : (absolute.y >= absolute.z ? 1 : 2)
        let side = normal[axis] > 0 ? 1.0 : -1.0
        let ring = points.map { SIMD2($0[(axis + 1) % 3], $0[(axis + 2) % 3] * side) }
        if count == 4 {
            func oriented(_ a: Int, _ b: Int, _ c: Int) -> Bool {
                PolygonTriangulator.cross(ring[b] - ring[a], ring[c] - ring[a]) > 0
            }
            let diagonal02 = oriented(0, 1, 2) && oriented(0, 2, 3)
            let diagonal13 = oriented(0, 1, 3) && oriented(1, 2, 3)
            if diagonal02 && (!diagonal13 || simd_distance_squared(points[0], points[2]) <= simd_distance_squared(points[1], points[3])) {
                return [0, 1, 2, 0, 2, 3]
            }
            if diagonal13 { return [0, 1, 3, 1, 2, 3] }
        }
        do {
            return try PolygonTriangulator.triangulate(outer: ring).flatMap { [$0.x, $0.y, $0.z] }
        } catch {
            throw MeshGeometry.GeometryProcessingError.invalidPolygonMesh("a face could not be triangulated without reversing winding")
        }
    }
}
