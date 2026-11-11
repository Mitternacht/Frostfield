import Foundation
import simd

/// How filled and extruded shapes assign texture coordinates.
public enum ShapeTextureMapping: Equatable, Sendable {
    /// Caps map the bounds of all supplied shapes to 0...1 on each axis, with
    /// V increasing downward. Walls run U from 0 to 1 around each ring's
    /// perimeter and V from 0 at the front cap edge to 1 at the back.
    case normalized
    /// Every coordinate is a distance in world units divided by
    /// `unitsPerTile`, so a tiling material keeps one texel density on caps,
    /// bevels and walls. Caps use `(x, −y)`; walls use perimeter distance and
    /// distance along the bevel and wall profile from the front cap edge.
    case world(unitsPerTile: Float)
}

/// The depth, rounded bevel and shading of an extruded shape.
///
/// The walls stand on the shape's own outline, so the extrusion's XY bounds
/// equal the outline's. A bevel insets the caps by `bevelSize` and rounds the
/// edge with a quarter ellipse that is `bevelDepth` deep, so the straight wall
/// spans `depth − 2 × bevelDepth`.
public struct ShapeExtrusion: Equatable, Sendable {
    /// The full thickness along Z. The front cap faces +Z at `depth / 2`.
    public var depth: Float
    /// How far the bevel insets the caps from the outline, in the XY plane.
    public var bevelSize: Float
    /// How far the bevel reaches along Z from each cap.
    public var bevelDepth: Float
    /// Segments in each quarter-ellipse bevel, 1...32.
    public var bevelSegments: Int
    /// Axial intervals in the straight portion, from 1 to 4,096.
    public var steps: Int
    /// Wall vertices are split where adjacent outline edges turn by more than
    /// this angle in radians, giving hard corners; smaller turns share normals.
    public var creaseAngle: Float
    public var textureMapping: ShapeTextureMapping

    public init(
        depth: Float, bevelSize: Float = 0, bevelDepth: Float? = nil, bevelSegments: Int = 4,
        creaseAngle: Float = .pi / 6, textureMapping: ShapeTextureMapping = .normalized, steps: Int = 1
    ) {
        self.depth = depth
        self.bevelSize = bevelSize
        self.bevelDepth = bevelDepth ?? bevelSize
        self.bevelSegments = bevelSegments
        self.steps = steps
        self.creaseAngle = creaseAngle
        self.textureMapping = textureMapping
    }
}

extension MeshGeometry {
    /// A flat mesh of shapes in the XY plane at Z = 0, facing +Z, with holes
    /// left open. Cap UVs follow `textureMapping`; tangents point along +X.
    public static func shape(
        _ shapes: [PlanarShape], textureMapping: ShapeTextureMapping = .normalized
    ) throws -> MeshGeometry {
        guard !shapes.isEmpty else { throw ShapeMeshError.noShapes }
        try validate(textureMapping)
        let mapping = CapMapping(shapes: shapes, mapping: textureMapping)
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        for (shapeIndex, shape) in shapes.enumerated() {
            let rings = ([shape.outer] + shape.holes).map { $0.map(SIMD2<Double>.init) }
            let flat = rings.flatMap { $0 }.map { SIMD3<Float>(Float($0.x), Float($0.y), 0) }
            try appendCap(flat, rings: rings, front: true, shape: shapeIndex, mapping: mapping, into: &vertices, &indices)
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    /// Extrudes shapes along Z between caps at `±depth / 2`, with an optional
    /// rounded bevel. The triangles wind counterclockwise seen from outside,
    /// and cap rims and wall rims share exact positions. Disjoint shapes
    /// with non-touching rings form closed solids. Rings or shapes that
    /// touch at a vertex can produce walls that share an edge and are
    /// unsuitable as inputs to ``MeshBoolean``.
    ///
    /// The bevel insets each outline point along its miter. Edges that shrink
    /// to nothing merge their points, as edge events of a straight skeleton
    /// do, and reflex corners move at most twice the bevel size. Throws
    /// ``ShapeMeshError/bevelTooLarge(_:)`` when the inset would make rings
    /// cross, turn inside out or vanish, as when the bevel exceeds half a
    /// stroke's width.
    ///
    /// Optional UV callbacks can split per-face seams. Changed UVs regenerate
    /// V-down tangents, with a budget of 1,048,576 index corners.
    public static func extrude(
        _ shapes: [PlanarShape], _ extrusion: ShapeExtrusion, uvGenerator: ShapeExtrusionUVGenerator? = nil
    ) throws -> MeshGeometry {
        guard !shapes.isEmpty else { throw ShapeMeshError.noShapes }
        let depth = Double(extrusion.depth)
        guard extrusion.depth.isFinite, extrusion.depth > 0, (Float(depth / 2)).isNormal else {
            throw ShapeMeshError.invalidDepth
        }
        let size = Double(extrusion.bevelSize), bevelDepth = Double(extrusion.bevelDepth)
        guard size.isFinite, bevelDepth.isFinite, size >= 0, bevelDepth >= 0, (size == 0) == (bevelDepth == 0),
              2 * bevelDepth <= depth else {
            throw ShapeMeshError.invalidBevel
        }
        guard (1...32).contains(extrusion.bevelSegments) else { throw ShapeMeshError.invalidBevelSegments }
        guard (1...4_096).contains(extrusion.steps) else { throw ShapeMeshError.invalidSteps }
        try validate(extrusion.textureMapping, creaseAngle: extrusion.creaseAngle)

        let profile = BevelProfile(
            depth: depth, size: size, bevelDepth: bevelDepth, segments: extrusion.bevelSegments, steps: extrusion.steps
        )
        let uv = uvGenerator.flatMap(ExtrusionUVBuilder.init)
        if extrusion.steps > 1 || uv != nil {
            try validateSweepBudget(shapes, levels: profile.levels.count, capped: true, customUV: uv != nil)
        }
        let mapping = CapMapping(shapes: shapes, mapping: extrusion.textureMapping)
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        for (shapeIndex, shape) in shapes.enumerated() {
            let rings = ([shape.outer] + shape.holes).map { $0.map(SIMD2<Double>.init) }
            let frames = try rings.map { try RingFrame(ring: $0, shape: shapeIndex) }
            // Each distinct inset of the profile, shallowest first, with every ring's points at that inset.
            let insets = Array(Set(profile.levels.map(\.inset))).sorted()
            let insetRings = try frames.map { try $0.inset(by: insets, shape: shapeIndex) }
            if size > 0 {
                // Edge collapses are handled; a ring that crosses itself or another ring is not.
                let deepest = insetRings.map { PlanarShape.removingRepeats($0[insets.count - 1]) }
                guard deepest.allSatisfy({ $0.count >= 3 }), !PlanarArrangement.hasCrossing(deepest),
                      zip(rings, deepest).allSatisfy({ PolygonTriangulator.doubledArea($0) * PolygonTriangulator.doubledArea($1) > 0 })
                else {
                    throw ShapeMeshError.bevelTooLarge(shapeIndex)
                }
            }

            // Positions of every ring at every profile level, rounded once and shared by caps and walls.
            let levels = insetRings.map { byInset in
                profile.levels.map { level in
                    byInset[insets.firstIndex(of: level.inset)!].map { SIMD3<Float>(Float($0.x), Float($0.y), Float(level.z)) }
                }
            }
            for (ringIndex, frame) in frames.enumerated() {
                try appendWalls(
                    frame, positions: levels[ringIndex], profile: profile,
                    creaseCosine: cos(Double(extrusion.creaseAngle)), mapping: extrusion.textureMapping,
                    shape: shapeIndex, ring: ringIndex, uv: uv, into: &vertices, &indices
                )
            }
            let capRings = insetRings.map { $0[insets.count - 1] }
            try appendCap(
                levels.flatMap { $0[0] }, rings: capRings, front: true, shape: shapeIndex,
                mapping: mapping, uv: uv, into: &vertices, &indices
            )
            try appendCap(
                levels.flatMap { $0[profile.levels.count - 1] }, rings: capRings, front: false, shape: shapeIndex,
                mapping: mapping, uv: uv, into: &vertices, &indices
            )
            guard vertices.count < Int(UInt32.max) else { throw ShapeMeshError.vertexBudget }
        }
        let mesh = try MeshGeometry(vertices: vertices, indices: indices)
        return try uv?.applying(to: mesh) ?? mesh
    }

    /// Sweeps XY shapes along a spatial curve with one ring per approximate
    /// arc-length step. The shape's X/Y axes follow the curve's parallel-transported
    /// normal and binormal. Open sweeps have flat endpoint caps, material slot 0.
    /// Walls use material slot 1. A closed sweep welds its positions and frames
    /// at the seam, with separate UVs there. Choose a profile that clears
    /// nonadjacent parts of the path. Locally folded triangles are rejected.
    ///
    /// Supply 1...4,096 steps (at least three if closed) and 1...65,536 sampling
    /// intervals for the arc-length approximation. Path sweeps use flat rims.
    /// Optional UV callbacks follow the same per-face and tangent conventions
    /// as straight extrusions, including their index-corner budget.
    public static func extrude(
        _ shapes: [PlanarShape], along path: Curve3D, steps: Int = 1, closed: Bool = false,
        arcLengthSubdivisions: Int = 200, creaseAngle: Float = .pi / 6,
        textureMapping: ShapeTextureMapping = .normalized, uvGenerator: ShapeExtrusionUVGenerator? = nil
    ) throws -> MeshGeometry {
        guard !shapes.isEmpty else { throw ShapeMeshError.noShapes }
        guard ((closed ? 3 : 1)...4_096).contains(steps) else { throw ShapeMeshError.invalidSteps }
        try validate(textureMapping, creaseAngle: creaseAngle)
        let uv = uvGenerator.flatMap(ExtrusionUVBuilder.init)
        try validateSweepBudget(shapes, levels: steps + 1, capped: !closed, customUV: uv != nil)
        let sampled = try path.arcSamples(segments: steps, subdivisions: arcLengthSubdivisions)
        let pathFrames: [CurveSweepFrame]
        do {
            pathFrames = try CurveSweepFrame.make(samples: sampled.samples, length: sampled.length, closed: closed)
        } catch {
            switch error {
            case .invalidClosedPath: throw ShapeMeshError.invalidClosedPath
            case .invalidSample(let index): throw ShapeMeshError.invalidPathFrame(index)
            }
        }
        // The wall builder's front-to-back convention starts at the final cap.
        let frames = Array(pathFrames.reversed())
        let profile = BevelProfile(depth: sampled.length, size: 0, bevelDepth: 0, segments: 1, steps: steps)
        let mapping = CapMapping(shapes: shapes, mapping: textureMapping)
        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        var groups: [MeshGeometryGroup] = []
        for (shapeIndex, shape) in shapes.enumerated() {
            let rings = ([shape.outer] + shape.holes).map { $0.map(SIMD2<Double>.init) }
            let levels = try rings.map { ring in
                try frames.map { frame in
                    try ring.map { point in
                        let position = SIMD3<Float>(frame.center + frame.map(point))
                        guard boundsFinite(position) else { throw ShapeMeshError.foldedPathSurface(shapeIndex) }
                        return position
                    }
                }
            }
            if !closed {
                let start = indices.count
                try appendCap(
                    levels.flatMap { $0[steps] }, rings: rings, front: false, shape: shapeIndex,
                    mapping: mapping, pathFrame: frames[steps], uv: uv, into: &vertices, &indices
                )
                try appendCap(
                    levels.flatMap { $0[0] }, rings: rings, front: true, shape: shapeIndex,
                    mapping: mapping, pathFrame: frames[0], uv: uv, into: &vertices, &indices
                )
                groups.append(.init(start: start, count: indices.count - start, materialIndex: 0))
            }
            let start = indices.count
            for (ringIndex, ring) in rings.enumerated() {
                try appendWalls(
                    RingFrame(ring: ring, shape: shapeIndex), positions: levels[ringIndex], profile: profile,
                    creaseCosine: cos(Double(creaseAngle)), mapping: textureMapping, shape: shapeIndex, ring: ringIndex,
                    sweep: (frames, closed), uv: uv, into: &vertices, &indices
                )
            }
            groups.append(.init(start: start, count: indices.count - start, materialIndex: 1))
        }
        let mesh = try MeshGeometry(vertices: vertices, indices: indices).withGroups(groups)
        return try uv?.applying(to: mesh) ?? mesh
    }

    private static func validateSweepBudget(_ shapes: [PlanarShape], levels: Int, capped: Bool, customUV: Bool) throws {
        // Each profile point can split into two wall vertices. Caps may add
        // two more; holes add at most two bridge vertices per ring during triangulation.
        let vertexLimit = 1_048_576 / (2 * levels + (capped ? 2 : 0))
        let indexBudget = customUV ? 1_048_576 : 6_291_456
        let indexLimit = indexBudget / (6 * (levels - 1) + (capped ? 12 : 0))
        var remaining = min(vertexLimit, indexLimit)
        for shape in shapes {
            for ring in [shape.outer] + shape.holes {
                guard ring.count <= remaining else { throw ShapeMeshError.vertexBudget }
                remaining -= ring.count
            }
        }
    }

    public enum ShapeMeshError: LocalizedError, Equatable {
        case noShapes
        case invalidDepth
        case invalidBevel
        case invalidBevelSegments
        case invalidSteps
        case invalidClosedPath
        case invalidPathFrame(Int)
        case foldedPathSurface(Int)
        case invalidCreaseAngle
        case invalidTextureScale
        case invalidGeneratedUV(Int)
        case uvGeneratorNeedsExtrusion
        case reversingOutline(Int)
        case bevelTooLarge(Int)
        case triangulation(Int)
        case vertexBudget

        public var errorDescription: String? {
            switch self {
            case .noShapes: "Supply at least one shape."
            case .invalidDepth: "Extrusion depth must be finite and positive, with a normal Float half-depth."
            case .invalidBevel:
                "Bevel size and depth must be finite, both zero or both positive, with twice the bevel depth no greater than the extrusion depth."
            case .invalidBevelSegments: "A bevel needs 1 to 32 segments."
            case .invalidSteps: "Use 1 to 4096 extrusion steps, or at least 3 for a closed path."
            case .invalidClosedPath: "A closed extrusion needs matching endpoint positions and tangents."
            case .invalidPathFrame(let index): "Extrusion path sample \(index) coincides with its neighbor or reverses its frame."
            case .foldedPathSurface(let shape): "Shape \(shape) collapses or folds at a bend, or exceeds Float position precision."
            case .invalidCreaseAngle: "The crease angle must be between 0 and pi radians."
            case .invalidTextureScale: "World texture mapping needs a finite, positive number of units per tile."
            case .invalidGeneratedUV(let shape): "The UV callback returned a nonfinite coordinate for shape \(shape)."
            case .uvGeneratorNeedsExtrusion: "Supply text extrusion settings when using an extrusion UV generator."
            case .reversingOutline(let shape): "Shape \(shape) has an outline that turns back on itself."
            case .bevelTooLarge(let shape):
                "The bevel is too large for shape \(shape): insetting its outline makes rings cross, turn inside out or vanish."
            case .triangulation(let shape): "The caps of shape \(shape) could not be triangulated."
            case .vertexBudget: "The extrusion exceeds the geometry budget for its shape and step counts."
            }
        }
    }

    private static func validate(_ mapping: ShapeTextureMapping, creaseAngle: Float? = nil) throws {
        if let creaseAngle, !(0...Float.pi).contains(creaseAngle) { throw ShapeMeshError.invalidCreaseAngle }
        if case .world(let units) = mapping, !(units.isFinite && units > 0) {
            throw ShapeMeshError.invalidTextureScale
        }
    }

    private static func appendCap(
        _ positions: [SIMD3<Float>], rings: [[SIMD2<Double>]], front: Bool, shape: Int, mapping: CapMapping,
        pathFrame: CurveSweepFrame? = nil, uv: ExtrusionUVBuilder? = nil, into vertices: inout [MeshVertex], _ indices: inout [UInt32]
    ) throws {
        // Collapsed bevel edges leave repeated points; triangulate each distinct point once.
        var unique: [[SIMD2<Double>]] = []
        var sources: [(index: Int, point: SIMD2<Double>)] = []
        var offset = 0
        for ring in rings {
            var kept: [SIMD2<Double>] = []
            for (index, point) in ring.enumerated() {
                let position = positions[offset + index]
                if let last = sources.last, kept.count > 0, positions[last.index] == position { continue }
                kept.append(point)
                sources.append((offset + index, point))
            }
            if kept.count > 1, positions[sources[sources.count - kept.count].index] == positions[sources[sources.count - 1].index] {
                kept.removeLast()
                sources.removeLast()
            }
            unique.append(kept)
            offset += ring.count
        }
        let triangles: [SIMD3<Int>]
        do {
            triangles = try PolygonTriangulator.triangulate(outer: unique[0], holes: Array(unique.dropFirst()))
        } catch {
            throw ShapeMeshError.triangulation(shape)
        }
        let base = UInt32(vertices.count)
        let normal = SIMD3<Float>(pathFrame?.tangent ?? SIMD3(0, 0, 1)) * (front ? 1 : -1)
        let tangent = SIMD4(SIMD3<Float>(pathFrame?.normal ?? SIMD3(1, 0, 0)), front ? -1 : 1)
        for source in sources {
            let position = positions[source.index]
            vertices.append(MeshVertex(
                position: position, normal: normal,
                textureCoordinate: mapping.coordinate(SIMD2<Float>(source.point)), tangent: tangent
            ))
        }
        for (triangleIndex, triangle) in triangles.enumerated() {
            let a = base + UInt32(triangle.x), b = base + UInt32(triangle.y), c = base + UInt32(triangle.z)
            let face = front ? [a, b, c] : [a, c, b]
            if pathFrame != nil { try validatePathFace(face, vertices: vertices, shape: shape) }
            try uv?.appendCap(face, vertices: vertices, shape: shape, front: front, triangle: triangleIndex)
            indices += face
        }
    }

    private static func appendWalls(
        _ frame: RingFrame, positions: [[SIMD3<Float>]], profile: BevelProfile, creaseCosine: Double,
        mapping: ShapeTextureMapping, shape: Int, ring: Int,
        sweep: (frames: [CurveSweepFrame], closed: Bool)? = nil, uv: ExtrusionUVBuilder? = nil,
        into vertices: inout [MeshVertex], _ indices: inout [UInt32]
    ) throws {
        let count = frame.ring.count
        let perimeter = frame.distances[count]
        let uScale: Double, vScale: Double
        switch mapping {
        case .normalized:
            uScale = 1 / perimeter
            vScale = 1 / profile.distances[profile.levels.count - 1]
        case .world(let units):
            uScale = 1 / Double(units)
            vScale = uScale
        }
        // A point is a hard corner when its edges turn by more than the crease angle.
        let corner = (0..<count).map { index in
            simd_dot(frame.directions[(index + count - 1) % count], frame.directions[index]) < creaseCosine
        }

        // For every level: the vertex ending the incoming edge and the one starting the outgoing edge.
        var incoming = [[UInt32]](repeating: [UInt32](repeating: 0, count: count), count: profile.levels.count)
        var outgoing = incoming
        for (levelIndex, level) in profile.levels.enumerated() {
            let v = Float(profile.distances[levelIndex] * vScale)
            func append(_ index: Int, planar: SIMD2<Double>, along: SIMD2<Double>, u: Double) throws -> UInt32 {
                var normal = SIMD3<Double>(planar * level.planarWeight, level.axialWeight)
                var tangent = SIMD3<Double>(along, 0)
                if let sweep {
                    let pathFrame = sweep.frames[levelIndex]
                    let last = positions.count - 1
                    let previous = levelIndex == 0 && sweep.closed ? last - 1 : max(0, levelIndex - 1)
                    let next = levelIndex == last && sweep.closed ? 1 : min(last, levelIndex + 1)
                    let longitudinal = SIMD3<Double>(positions[next][index]) - SIMD3<Double>(positions[previous][index])
                    guard simd_dot(longitudinal, pathFrame.tangent) < 0 else {
                        throw ShapeMeshError.foldedPathSurface(shape)
                    }
                    tangent = pathFrame.map(along)
                    normal = simd_cross(longitudinal, tangent)
                }
                vertices.append(MeshVertex(
                    position: positions[levelIndex][index], normal: SIMD3<Float>(simd_normalize(normal)),
                    textureCoordinate: SIMD2(Float(u * uScale), v),
                    tangent: SIMD4(SIMD3<Float>(tangent), -1)
                ))
                return UInt32(vertices.count - 1)
            }
            for index in 0..<count {
                let before = frame.directions[(index + count - 1) % count], after = frame.directions[index]
                let u = frame.distances[index]
                if corner[index] {
                    incoming[levelIndex][index] = try append(index, planar: outward(before), along: before, u: index == 0 ? perimeter : u)
                    outgoing[levelIndex][index] = try append(index, planar: outward(after), along: after, u: u)
                } else {
                    let along = simd_normalize(before + after)
                    let first = try append(index, planar: outward(along), along: along, u: u)
                    outgoing[levelIndex][index] = first
                    incoming[levelIndex][index] = index == 0
                        ? try append(index, planar: outward(along), along: along, u: perimeter) : first
                }
            }
        }
        for levelIndex in 0..<(profile.levels.count - 1) {
            for index in 0..<count {
                let next = (index + 1) % count
                let upperA = outgoing[levelIndex][index], upperB = incoming[levelIndex][next]
                let lowerA = outgoing[levelIndex + 1][index], lowerB = incoming[levelIndex + 1][next]
                let quad = [upperA, upperB, lowerB, lowerA]
                var mapped: [SIMD2<Float>]?
                // An edge that collapsed at one level leaves a triangle, not a quad.
                for corners in [SIMD3(0, 3, 1), SIMD3(1, 3, 2)] {
                    let triangle = [quad[corners.x], quad[corners.y], quad[corners.z]]
                    if sweep != nil {
                        try validatePathFace(triangle, vertices: vertices, shape: shape)
                    } else {
                        let p = triangle.map { vertices[Int($0)].position }
                        guard p[0] != p[1], p[1] != p[2], p[2] != p[0] else { continue }
                    }
                    if let uv {
                        let coordinates = try mapped ?? uv.wallCoordinates(
                            quad, vertices: vertices, shape: shape, ring: ring, edge: index, level: levelIndex
                        )
                        mapped = coordinates
                        uv.appendWall(coordinates, corners: corners)
                    }
                    indices += triangle
                }
            }
        }
    }

    private static func validatePathFace(_ face: [UInt32], vertices: [MeshVertex], shape: Int) throws {
        let a = vertices[Int(face[0])], b = vertices[Int(face[1])], c = vertices[Int(face[2])]
        let origin = SIMD3<Double>(a.position)
        let cross = simd_cross(SIMD3<Double>(b.position) - origin, SIMD3<Double>(c.position) - origin)
        guard simd_dot(cross, SIMD3<Double>(a.normal + b.normal + c.normal)) > 0 else {
            throw ShapeMeshError.foldedPathSurface(shape)
        }
    }

    /// The outward normal of an edge direction on a ring whose filled side is on the left.
    private static func outward(_ direction: SIMD2<Double>) -> SIMD2<Double> {
        SIMD2(direction.y, -direction.x)
    }
}

/// Edge directions and perimeter distances of one ring, and its inward offsets.
private struct RingFrame {
    /// How far a reflex corner may move, in units of the inset distance.
    static let reflexMiterLimit = 2.0

    let ring: [SIMD2<Double>]
    let directions: [SIMD2<Double>]
    /// Cumulative perimeter distance at each point, with the full perimeter appended.
    let distances: [Double]

    init(ring: [SIMD2<Double>], shape: Int) throws {
        let count = ring.count
        var directions: [SIMD2<Double>] = []
        var distances = [0.0]
        for index in 0..<count {
            let edge = ring[(index + 1) % count] - ring[index]
            let length = simd_length(edge)
            directions.append(edge / length)
            distances.append(distances[index] + length)
        }
        for index in 0..<count {
            let first = directions[(index + count - 1) % count], second = directions[index]
            guard 1 + simd_dot(first, second) > 1e-9 else { throw MeshGeometry.ShapeMeshError.reversingOutline(shape) }
        }
        self.ring = ring
        self.directions = directions
        self.distances = distances
    }

    /// The ring moved inward so every edge shifts by each distance along its normal.
    ///
    /// Each point slides along its miter. When an edge shrinks to nothing its
    /// two points merge and continue along the miter of the surviving edges,
    /// as at an edge event of a straight skeleton, so every original point
    /// keeps a position. Reflex corners move at most ``reflexMiterLimit``
    /// times the inset, which cuts the arc an exact offset would round them
    /// with by one chord. [Aichholzer et al. 1995](https://doi.org/10.3217/jucs-001-12-0752)
    func inset(by targets: [Double], shape: Int) throws -> [[SIMD2<Double>]] {
        var position = ring
        var members = ring.indices.map { [$0] }
        var outgoing = directions
        var current = 0.0
        let scale = ring.reduce(0) { max($0, simd_reduce_max(simd_abs($1))) }
        let epsilon = 1e-12 * max(scale, 1e-300)
        var results: [[SIMD2<Double>]] = []
        func outward(_ direction: SIMD2<Double>) -> SIMD2<Double> { SIMD2(direction.y, -direction.x) }
        for target in targets {
            while true {
                let count = position.count
                guard count >= 3 else { throw MeshGeometry.ShapeMeshError.bevelTooLarge(shape) }
                var miters: [SIMD2<Double>] = []
                for index in 0..<count {
                    let incoming = outgoing[(index + count - 1) % count]
                    let first = outward(incoming), second = outward(outgoing[index])
                    let denominator = 1 + simd_dot(first, second)
                    guard denominator > 1e-9 else { throw MeshGeometry.ShapeMeshError.bevelTooLarge(shape) }
                    var miter = (first + second) / denominator
                    // At a reflex corner the exact offset is an arc of the inset radius, not a
                    // spike; limit the miter there as a stroke miter limit does.
                    let reflex = incoming.x * outgoing[index].y - incoming.y * outgoing[index].x < 0
                    if reflex, simd_length(miter) > Self.reflexMiterLimit {
                        miter = simd_normalize(miter) * Self.reflexMiterLimit
                    }
                    miters.append(miter)
                }
                // An edge keeps its direction and loses length at this rate per unit of inset.
                var step = target - current
                for index in 0..<count {
                    let next = (index + 1) % count
                    let rate = simd_dot(miters[next] - miters[index], outgoing[index])
                    guard rate > 0 else { continue }
                    let length = max(simd_dot(position[next] - position[index], outgoing[index]), 0)
                    step = min(step, length / rate)
                }
                for index in 0..<count { position[index] -= miters[index] * step }
                current += step
                var merged = false
                var index = 0
                while index < position.count, position.count >= 3 {
                    let next = (index + 1) % position.count
                    guard simd_dot(position[next] - position[index], outgoing[index]) <= epsilon else {
                        index += 1
                        continue
                    }
                    position[index] = (position[index] + position[next]) / 2
                    members[index] += members[next]
                    outgoing[index] = outgoing[next]
                    position.remove(at: next)
                    members.remove(at: next)
                    outgoing.remove(at: next)
                    merged = true
                    if next < index { index -= 1 }
                }
                if !merged && current >= target { break }
            }
            var placed = [SIMD2<Double>](repeating: .zero, count: ring.count)
            for (node, list) in members.enumerated() {
                for member in list { placed[member] = position[node] }
            }
            results.append(placed)
        }
        return results
    }
}

/// The front-to-back levels of a wall: a quarter-ellipse bevel, the straight wall and the mirrored bevel.
private struct BevelProfile {
    struct Level {
        let inset: Double
        let z: Double
        /// The normal's weights on the outward planar direction and on Z.
        let planarWeight: Double
        let axialWeight: Double
    }

    let levels: [Level]
    /// Cumulative profile length from the front cap edge.
    let distances: [Double]

    init(depth: Double, size: Double, bevelDepth: Double, segments: Int, steps: Int) {
        let wallTop = depth / 2 - bevelDepth
        var levels: [Level] = []
        if size > 0 {
            // The profile (inset, z) = (s(1 − sin φ), wallTop + t cos φ) has normal ∝ (t sin φ, s cos φ).
            for step in 0...segments {
                let phi = Double.pi / 2 * Double(step) / Double(segments)
                let last = step == segments
                levels.append(Level(
                    inset: last ? 0 : size * (1 - sin(phi)),
                    z: step == 0 ? depth / 2 : (last ? wallTop : wallTop + bevelDepth * cos(phi)),
                    planarWeight: bevelDepth * sin(phi), axialWeight: last ? 0 : size * cos(phi)
                ))
            }
            let front = levels
            if wallTop == 0 { levels.removeLast() }
            else { appendWallInterior(halfDepth: wallTop, steps: steps, into: &levels) }
            for level in front.reversed() {
                levels.append(Level(inset: level.inset, z: -level.z, planarWeight: level.planarWeight, axialWeight: -level.axialWeight))
            }
        } else {
            levels = [Level(inset: 0, z: depth / 2, planarWeight: 1, axialWeight: 0)]
            appendWallInterior(halfDepth: depth / 2, steps: steps, into: &levels)
            levels.append(Level(inset: 0, z: -depth / 2, planarWeight: 1, axialWeight: 0))
        }
        var distances = [0.0]
        for index in 1..<levels.count {
            let a = levels[index - 1], b = levels[index]
            distances.append(distances[index - 1] + hypot(b.inset - a.inset, b.z - a.z))
        }
        self.levels = levels
        self.distances = distances
    }
}

/// Adds equally spaced levels between the existing front and back wall edges.
private func appendWallInterior(halfDepth: Double, steps: Int, into levels: inout [BevelProfile.Level]) {
    for step in 1..<steps {
        let z = halfDepth * (1 - 2 * Double(step) / Double(steps))
        levels.append(.init(inset: 0, z: z, planarWeight: 1, axialWeight: 0))
    }
}

/// Cap texture coordinates for a set of shapes.
private struct CapMapping {
    let lower: SIMD2<Float>
    let upper: SIMD2<Float>
    let mapping: ShapeTextureMapping

    init(shapes: [PlanarShape], mapping: ShapeTextureMapping) {
        let points = shapes.flatMap(\.outer)
        lower = points.reduce(points[0]) { simd_min($0, $1) }
        upper = points.reduce(points[0]) { simd_max($0, $1) }
        self.mapping = mapping
    }

    func coordinate(_ point: SIMD2<Float>) -> SIMD2<Float> {
        switch mapping {
        case .normalized:
            let extent = upper - lower
            return SIMD2((point.x - lower.x) / extent.x, (upper.y - point.y) / extent.y)
        case .world(let units):
            return SIMD2(point.x / units, -point.y / units)
        }
    }
}
