// SPDX-FileCopyrightText: 2011 Morten S. Mikkelsen
// SPDX-FileCopyrightText: 2022 Blender Authors
// SPDX-FileCopyrightText: 2026 Frostfield contributors
// SPDX-License-Identifier: Apache-2.0
// Swift adaptation of Blender's intern/mikktspace implementation.

import Foundation
import simd

// MikkTSpace tangent generation.
//
// This is a Swift implementation of the MikkTSpace algorithm by Morten S. Mikkelsen
// ("Simulation of Wrinkled Surfaces Revisited", 2008; http://www.mikktspace.com),
// following the behavior of Blender's maintained implementation (intern/mikktspace),
// which Blender uses to bake and display tangent-space normal maps. The steps match
// that implementation for triangle input: corners with identical position, normal and
// texture coordinate are welded; triangles with coincident corner positions are
// degenerate and copy a neighbor's frame; each triangle's tangent is dP/du projected
// and normalized, signed by its UV orientation; tangents accumulate per vertex over
// groups of edge-connected triangles with the same UV orientation, weighted by the
// corner angle (with Blender's polynomial arccosine), then normalize.

extension MeshGeometry {
    /// Which way a tangent-space normal map's green channel points.
    public enum NormalMapConvention: Sendable, Hashable {
        /// Green points up the image, toward decreasing top-left V. glTF, Blender,
        /// Three.js, Unity, and the OpenGL setting in Substance use this convention.
        case openGL
        /// Green points down the image, toward increasing top-left V. Unreal, the
        /// DirectX setting in Substance, `ProceduralSurface`, and Frostfield's
        /// derivative fallback for meshes without tangents use this convention.
        case directX
    }

    /// Tangent generation output and the original vertex index for each output vertex.
    public struct TangentGeneration: Sendable {
        public let geometry: MeshGeometry
        /// Each original vertex keeps its index. Additional vertices share
        /// the source index of the vertex they copy. Use this map to copy
        /// per-vertex channels stored separately from the geometry.
        public let sourceVertexIndices: [Int]
    }

    /// Returns the mesh with MikkTSpace tangents for one texture coordinate set,
    /// like Three.js `BufferGeometryUtils.computeMikkTSpaceTangents`.
    ///
    /// Tangent XYZ follows increasing U. The handedness sign makes
    /// `cross(normal, tangent.xyz) * tangent.w` point along the green channel of a
    /// normal map in `convention`. With `.openGL`, the result matches the tangents
    /// Blender computes for the same triangles and writes to glTF, so normal maps baked
    /// against them decode as baked. With `.directX`, signs are negated.
    ///
    /// A vertex whose triangles need different tangent frames, as on a mirrored UV
    /// seam, is split into one vertex per frame; split copies follow the original
    /// vertices. Normals are used normalized. Where MikkTSpace has no frame, because the
    /// vertex's triangles have no UV area or none references it, the tangent is a
    /// unit vector perpendicular to the normal with handedness +1 in the convention
    /// (MikkTSpace itself would return +X or zero there). Existing tangents are replaced.
    /// Material groups and other vertex attributes are preserved. Use
    /// ``generatingTangentsWithRemap(textureCoordinateSet:convention:)`` to carry
    /// separate per-vertex channels such as skin weights and morph deltas.
    ///
    /// Throws when the secondary set is requested and a referenced vertex lacks it,
    /// or when splitting would exceed the 32-bit index range.
    public func generatingTangents(
        textureCoordinateSet: MeshTextureCoordinateSet = .primary,
        convention: NormalMapConvention = .openGL
    ) throws -> MeshGeometry {
        try generatingTangentsWithRemap(textureCoordinateSet: textureCoordinateSet, convention: convention).geometry
    }

    /// Generates tangents and reports how the output vertices map to the source.
    public func generatingTangentsWithRemap(
        textureCoordinateSet: MeshTextureCoordinateSet = .primary,
        convention: NormalMapConvention = .openGL
    ) throws -> TangentGeneration {
        let triangleCount = indices.count / 3
        var uvs: [SIMD2<Float>] = []
        uvs.reserveCapacity(vertices.count)
        let flip: Float = convention == .openGL ? -1 : 1
        for vertex in vertices {
            let uv: SIMD2<Float>
            switch textureCoordinateSet {
            case .primary: uv = vertex.textureCoordinate
            case .secondary: uv = vertex.secondaryTextureCoordinate ?? SIMD2(.nan, .nan)
            }
            // MikkTSpace's sign follows the orientation of the UV map. Blender runs it on
            // bottom-left UVs, so the OpenGL convention negates top-left V. Negation is exact.
            uvs.append(SIMD2(uv.x, uv.y * flip))
        }
        if textureCoordinateSet == .secondary,
           let missing = indices.first(where: { vertices[Int($0)].secondaryTextureCoordinate == nil }) {
            throw TangentGenerationError.missingSecondaryTextureCoordinates(vertex: Int(missing))
        }
        let normals = vertices.map { vertex -> SIMD3<Float> in
            let lengthSquared = Mikk.dot(vertex.normal, vertex.normal)
            if abs(lengthSquared - 1) <= 4 * Float.ulpOfOne { return vertex.normal }
            // Scaling first keeps length computation in range for finite nonzero input.
            let scale = simd_reduce_max(simd_abs(vertex.normal))
            return simd_normalize(vertex.normal / scale)
        }

        var mikk = Mikk(
            positions: vertices.map(\.position), normals: normals, textureCoordinates: uvs,
            indices: indices, triangleCount: triangleCount
        )
        let frames = mikk.generate()

        // One output vertex per distinct (source vertex, frame). The first frame keeps the
        // source index; later frames append copies.
        var output = vertices
        var sourceVertexIndices = Array(vertices.indices)
        var assigned = [SIMD4<Float>?](repeating: nil, count: vertices.count)
        var copies: [VertexFrameKey: UInt32] = [:]
        var newIndices = indices
        for corner in 0..<(triangleCount * 3) {
            let source = Int(indices[corner])
            var frame = frames[corner]
            let direction = SIMD3(frame.x, frame.y, frame.z)
            if !(Mikk.dot(direction, direction) > 0.25) || !direction.x.isFinite || !direction.y.isFinite || !direction.z.isFinite {
                frame = SIMD4(Self.perpendicularUnit(to: normals[source]), 1)
            }
            if let existing = assigned[source] {
                if existing == frame { continue }
                let key = VertexFrameKey(source: UInt32(source), frame: frame)
                if let copy = copies[key] {
                    newIndices[corner] = copy
                    continue
                }
                guard output.count < Int(UInt32.max) else { throw TangentGenerationError.tooManyVertices }
                let copy = UInt32(output.count)
                copies[key] = copy
                output.append(vertices[source].replacingTangent(frame))
                sourceVertexIndices.append(source)
                newIndices[corner] = copy
            } else {
                assigned[source] = frame
                output[source] = vertices[source].replacingTangent(frame)
            }
        }
        for index in 0..<vertices.count where assigned[index] == nil {
            output[index] = vertices[index].replacingTangent(SIMD4(Self.perpendicularUnit(to: normals[index]), 1))
        }
        let geometry = try MeshGeometry(vertices: output, indices: newIndices).withGroups(groups)
        return TangentGeneration(geometry: geometry, sourceVertexIndices: sourceVertexIndices)
    }

    /// A deterministic unit vector perpendicular to a unit normal (Duff et al. 2017).
    static func perpendicularUnit(to normal: SIMD3<Float>) -> SIMD3<Float> {
        let sign: Float = normal.z >= 0 ? 1 : -1
        let a = -1 / (sign + normal.z)
        let b = normal.x * normal.y * a
        return simd_normalize(SIMD3(1 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x))
    }

    public enum TangentGenerationError: LocalizedError, Equatable {
        case missingSecondaryTextureCoordinates(vertex: Int)
        case tooManyVertices

        public var errorDescription: String? {
            switch self {
            case .missingSecondaryTextureCoordinates(let vertex):
                "Vertex \(vertex) has no secondary texture coordinates for tangent generation."
            case .tooManyVertices:
                "Splitting vertices for tangent frames would exceed the 32-bit index range."
            }
        }
    }
}

private struct VertexFrameKey: Hashable {
    let source: UInt32
    let x: UInt32, y: UInt32, z: UInt32, w: UInt32

    init(source: UInt32, frame: SIMD4<Float>) {
        self.source = source
        x = frame.x.bitPattern
        y = frame.y.bitPattern
        z = frame.z.bitPattern
        w = frame.w.bitPattern
    }
}

private extension MeshVertex {
    func replacingTangent(_ tangent: SIMD4<Float>?) -> MeshVertex {
        MeshVertex(
            position: position, normal: normal, textureCoordinate: textureCoordinate,
            secondaryTextureCoordinate: secondaryTextureCoordinate, tangent: tangent, color: color
        )
    }
}

/// The MikkTSpace computation for indexed triangles. Single precision throughout, with
/// the reference's operation order, so results agree with it to rounding.
private struct Mikk {
    static let unset = UInt32.max

    struct Triangle {
        var neighbor: (UInt32, UInt32, UInt32) = (Mikk.unset, Mikk.unset, Mikk.unset)
        var group: (UInt32, UInt32, UInt32) = (Mikk.unset, Mikk.unset, Mikk.unset)
        /// Welded vertex identifiers of the three corners.
        var vertices: (UInt32, UInt32, UInt32)
        var tangent = SIMD3<Float>(0, 0, 0)
        /// The source triangle, whose corners are `3 * face + i`.
        var face: UInt32
        var degenerate = false
        var groupWithAny = true
        var orientPreserving = false

        func vertex(_ i: Int) -> UInt32 { i == 0 ? vertices.0 : (i == 1 ? vertices.1 : vertices.2) }
        func neighborAt(_ i: Int) -> UInt32 { i == 0 ? neighbor.0 : (i == 1 ? neighbor.1 : neighbor.2) }
        func groupAt(_ i: Int) -> UInt32 { i == 0 ? group.0 : (i == 1 ? group.1 : group.2) }
        mutating func setNeighbor(_ i: Int, _ value: UInt32) {
            switch i { case 0: neighbor.0 = value; case 1: neighbor.1 = value; default: neighbor.2 = value }
        }
        mutating func setGroup(_ i: Int, _ value: UInt32) {
            switch i { case 0: group.0 = value; case 1: group.1 = value; default: group.2 = value }
        }
    }

    struct Group {
        var tangent = SIMD3<Float>(0, 0, 0)
        let vertexRepresentative: UInt32
        let orientPreserving: Bool
    }

    let positions: [SIMD3<Float>]
    let normals: [SIMD3<Float>]
    let textureCoordinates: [SIMD2<Float>]
    let indices: [UInt32]
    let triangleCount: Int
    /// Per source vertex, the welded identifier: the first vertex with equal attributes.
    var welded: [UInt32] = []
    var triangles: [Triangle] = []
    var groups: [Group] = []
    var goodCount = 0

    init(positions: [SIMD3<Float>], normals: [SIMD3<Float>], textureCoordinates: [SIMD2<Float>], indices: [UInt32], triangleCount: Int) {
        self.positions = positions
        self.normals = normals
        self.textureCoordinates = textureCoordinates
        self.indices = indices
        self.triangleCount = triangleCount
    }

    /// Per corner, the tangent and handedness (+1 orientation preserving, −1 otherwise).
    mutating func generate() -> [SIMD4<Float>] {
        weld()
        degeneratePrologue()
        // Zero marks corners without a frame; the caller substitutes a perpendicular tangent.
        var frames = [SIMD4<Float>](repeating: .zero, count: triangleCount * 3)
        guard goodCount > 0 else { return frames }
        initializeTriangles()
        buildNeighbors()
        buildGroups()
        accumulate(into: &frames)
        degenerateEpilogue(&frames)
        return frames
    }

    // Corners with equal position, normal and texture coordinate share one identifier.
    // Equality is by value, so −0 and +0 weld, as in the reference.
    private mutating func weld() {
        func bits(_ value: Float) -> UInt32 { (value == 0 ? Float(0) : value).bitPattern }
        var first: [SIMD8<UInt32>: UInt32] = [:]
        first.reserveCapacity(positions.count)
        welded = [UInt32](repeating: 0, count: positions.count)
        for index in positions.indices {
            let p = positions[index], n = normals[index], t = textureCoordinates[index]
            let key = SIMD8(bits(p.x), bits(p.y), bits(p.z), bits(n.x), bits(n.y), bits(n.z), bits(t.x), bits(t.y))
            if let existing = first[key] {
                welded[index] = existing
            } else {
                first[key] = UInt32(index)
                welded[index] = UInt32(index)
            }
        }
    }

    private func position(_ id: UInt32) -> SIMD3<Float> { positions[Int(id)] }
    private func normal(_ id: UInt32) -> SIMD3<Float> { normals[Int(id)] }
    private func uv(_ id: UInt32) -> SIMD2<Float> { textureCoordinates[Int(id)] }

    /// Marks triangles with coincident corner positions and moves them, in order, after the good ones.
    private mutating func degeneratePrologue() {
        var good: [Triangle] = []
        var bad: [Triangle] = []
        good.reserveCapacity(triangleCount)
        for face in 0..<triangleCount {
            let v = (welded[Int(indices[3 * face])], welded[Int(indices[3 * face + 1])], welded[Int(indices[3 * face + 2])])
            var triangle = Triangle(vertices: v, face: UInt32(face))
            let p0 = position(v.0), p1 = position(v.1), p2 = position(v.2)
            if p0 == p1 || p0 == p2 || p1 == p2 {
                triangle.degenerate = true
                bad.append(triangle)
            } else {
                good.append(triangle)
            }
        }
        goodCount = good.count
        triangles = good + bad
    }

    private mutating func initializeTriangles() {
        for t in 0..<goodCount {
            var triangle = triangles[t]
            let v1 = position(triangle.vertices.0), v2 = position(triangle.vertices.1), v3 = position(triangle.vertices.2)
            let t1 = uv(triangle.vertices.0), t2 = uv(triangle.vertices.1), t3 = uv(triangle.vertices.2)
            let t21x = t2.x - t1.x
            let t31x = t3.x - t1.x
            let t21y = t2.y - t1.y
            let t31y = t3.y - t1.y
            let d1 = v2 - v1, d2 = v3 - v1
            let signedArea = t21x * t31y - t21y * t31x
            let os = (t31y * d1) - (t21y * d2)
            let ot = (-t31x * d1) + (t21x * d2)
            triangle.orientPreserving = signedArea > 0
            if Self.notZero(signedArea) {
                let lengthOs2 = Self.dot(os, os)
                let lengthOt2 = Self.dot(ot, ot)
                let sign: Float = triangle.orientPreserving ? 1 : -1
                if Self.notZero(lengthOs2) {
                    triangle.tangent = os * (sign / lengthOs2.squareRoot())
                }
                if Self.notZero(lengthOs2) && Self.notZero(lengthOt2) {
                    triangle.groupWithAny = false
                }
            }
            triangles[t] = triangle
        }
    }

    /// Pairs each directed edge with the earliest unmatched opposite edge, in corner
    /// order, which is the reference's matching after its stable sort.
    private mutating func buildNeighbors() {
        var pending: [UInt64: Int32] = [:]
        pending.reserveCapacity(goodCount * 2)
        // A FIFO queue of unmatched corners per directed edge: head in `pending`, links in `next`.
        var next = [Int32](repeating: -1, count: goodCount * 3)
        var tail: [UInt64: Int32] = [:]
        for t in 0..<goodCount {
            for i in 0..<3 {
                let a = triangles[t].vertex(i), b = triangles[t].vertex(i == 2 ? 0 : i + 1)
                let reverse = UInt64(b) << 32 | UInt64(a)
                if let head = pending[reverse] {
                    let other = Int(head)
                    let link = next[other]
                    if link < 0 {
                        pending[reverse] = nil
                        tail[reverse] = nil
                    } else {
                        pending[reverse] = link
                    }
                    triangles[t].setNeighbor(i, UInt32(other / 3))
                    triangles[other / 3].setNeighbor(other % 3, UInt32(t))
                } else {
                    let key = UInt64(a) << 32 | UInt64(b)
                    let corner = Int32(3 * t + i)
                    if let last = tail[key] {
                        next[Int(last)] = corner
                    } else {
                        pending[key] = corner
                    }
                    tail[key] = corner
                }
            }
        }
    }

    private mutating func buildGroups() {
        var stack: [UInt32] = []
        for t in 0..<goodCount {
            for i in 0..<3 {
                let triangle = triangles[t]
                if triangle.groupWithAny || triangle.groupAt(i) != Self.unset { continue }
                let id = UInt32(groups.count)
                triangles[t].setGroup(i, id)
                groups.append(Group(vertexRepresentative: triangle.vertex(i), orientPreserving: triangle.orientPreserving))
                // Depth first, left neighbor before right, as the reference's recursion.
                stack.removeAll(keepingCapacity: true)
                stack.append(triangle.neighborAt(i == 0 ? 2 : i - 1))
                stack.append(triangle.neighborAt(i))
                while let next = stack.popLast() {
                    guard next != Self.unset else { continue }
                    let n = Int(next)
                    let group = groups[Int(id)]
                    var triangle = triangles[n]
                    let corner = triangle.vertex(0) == group.vertexRepresentative ? 0
                        : (triangle.vertex(1) == group.vertexRepresentative ? 1 : 2)
                    if triangle.groupAt(corner) != Self.unset { continue }
                    if triangle.groupWithAny, triangle.group.0 == Self.unset,
                       triangle.group.1 == Self.unset, triangle.group.2 == Self.unset {
                        // The first group to reach a group-with-any triangle sets its orientation.
                        triangle.orientPreserving = group.orientPreserving
                    }
                    if triangle.orientPreserving != group.orientPreserving {
                        triangles[n] = triangle
                        continue
                    }
                    triangle.setGroup(corner, id)
                    triangles[n] = triangle
                    stack.append(triangle.neighborAt(corner == 0 ? 2 : corner - 1))
                    stack.append(triangle.neighborAt(corner))
                }
            }
        }
    }

    private mutating func accumulate(into frames: inout [SIMD4<Float>]) {
        for t in 0..<goodCount {
            let triangle = triangles[t]
            if triangle.groupWithAny { continue }
            let n0 = normal(triangle.vertices.0), n1 = normal(triangle.vertices.1), n2 = normal(triangle.vertices.2)
            let p0 = position(triangle.vertices.0), p1 = position(triangle.vertices.1), p2 = position(triangle.vertices.2)
            let corners = [
                (n0, Self.dot(Self.project(n0, p1 - p0), Self.project(n0, p2 - p0))),
                (n1, Self.dot(Self.project(n1, p2 - p1), Self.project(n1, p0 - p1))),
                (n2, Self.dot(Self.project(n2, p0 - p2), Self.project(n2, p1 - p2))),
            ]
            for (i, (n, cosine)) in corners.enumerated() {
                let group = triangle.groupAt(i)
                guard group != Self.unset else { continue }
                let weight = Self.fastArccosine(min(max(cosine, -1), 1))
                groups[Int(group)].tangent += Self.project(n, triangle.tangent) * weight
            }
        }
        for index in groups.indices {
            groups[index].tangent = Self.normalize(groups[index].tangent)
        }
        for t in 0..<goodCount {
            let triangle = triangles[t]
            for i in 0..<3 {
                let group = triangle.groupAt(i)
                guard group != Self.unset else { continue }
                let value = groups[Int(group)]
                frames[3 * Int(triangle.face) + i] = SIMD4(value.tangent, value.orientPreserving ? 1 : -1)
            }
        }
    }

    /// Degenerate triangles copy the frame of the first good corner with the same welded vertex.
    private mutating func degenerateEpilogue(_ frames: inout [SIMD4<Float>]) {
        guard goodCount < triangles.count else { return }
        var firstGood: [UInt32: Int] = [:]
        for t in 0..<goodCount {
            for i in 0..<3 {
                let id = triangles[t].vertex(i)
                if firstGood[id] == nil { firstGood[id] = 3 * Int(triangles[t].face) + i }
            }
        }
        for t in goodCount..<triangles.count {
            for i in 0..<3 {
                guard let source = firstGood[triangles[t].vertex(i)] else { continue }
                frames[3 * Int(triangles[t].face) + i] = frames[source]
            }
        }
    }

    static func dot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    static func normalize(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let length = dot(v, v).squareRoot()
        return length != 0 ? v * (1 / length) : v
    }

    /// `v` projected onto the plane with unit normal `n`, normalized.
    static func project(_ n: SIMD3<Float>, _ v: SIMD3<Float>) -> SIMD3<Float> {
        normalize(v - n * dot(n, v))
    }

    static func notZero(_ value: Float) -> Bool {
        abs(value) > Float.leastNormalMagnitude
    }

    /// Blender's polynomial arccosine (from Cycles' fast math), within 4.6e-5 radians.
    static func fastArccosine(_ x: Float) -> Float {
        let f = abs(x)
        let m: Float = f < 1 ? 1 - (1 - f) : 1
        let a = (1 - m).squareRoot() * (1.5707963267 + m * (-0.213300989 + m * (0.077980478 + m * -0.02164095)))
        return x < 0 ? Float.pi - a : a
    }
}
