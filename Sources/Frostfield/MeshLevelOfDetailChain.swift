import Foundation
import simd

extension MeshGeometry {
    /// Perspective projection settings for estimating the switch distances of generated meshes.
    public struct ScreenSpaceErrorBudget: Sendable, Equatable {
        /// The target projected quadric error in pixels. Positive and finite.
        public var pixels: Float
        /// The drawable height in pixels. Positive and finite.
        public var viewportHeight: Float
        /// The vertical field of view in radians, greater than zero and less than π.
        public var verticalFieldOfView: Float

        public init(pixels: Float = 1, viewportHeight: Float = 1080, verticalFieldOfView: Float = .pi / 3) {
            self.pixels = pixels
            self.viewportHeight = viewportHeight
            self.verticalFieldOfView = verticalFieldOfView
        }
    }

    /// One mesh in a generated level-of-detail chain.
    public struct DetailLevel: Sendable {
        public let geometry: MeshGeometry
        /// The cumulative quadric error estimate relative to the full mesh, in mesh units.
        public let error: Float
        /// The estimated switch distance in world units, measured to the node origin.
        public let distance: Float
        /// The full mesh's representative for each vertex. Optimal placement can move
        /// the representative and update its attributes, as in `Simplification`.
        public let sourceVertexIndices: [Int]
    }

    /// Builds up to `levelCount` meshes whose requested triangle counts fall by
    /// `triangleRatio` per step. One simplifier maintains cumulative error from the
    /// full mesh, which remains at distance zero. Distinct levels have strictly
    /// increasing switch distances. The coarsest mesh at an equal distance is used.
    ///
    /// The estimate uses a perspective camera aimed at the node origin. At camera
    /// distance d, the nearest depth in an origin-centered bounding sphere of radius R
    /// is d − R. A transverse displacement e at that depth projects to
    /// e × viewportHeight / (2 tan(verticalFieldOfView / 2) × (d − R)).
    /// The switch distance therefore adds R to the estimated depth needed for the
    /// error to reach the pixel budget. The sphere contains both the source and the
    /// reduced mesh, and the world transform's linear part scales the error estimate.
    ///
    /// The quadric error incorporates position and attributes. It is separate from a
    /// geometric displacement bound or a guarantee for arbitrary camera directions.
    /// Use the node's complete affine world transform, including ancestor scale.
    /// Translation does not affect the estimate. Distances are fixed when prepared.
    /// Recompute them when the combined world transform stretches the mesh differently,
    /// or when the viewport height or field of view changes.
    public func levelOfDetailChain(
        levelCount: Int, triangleRatio: Float = 0.5, budget: ScreenSpaceErrorBudget = ScreenSpaceErrorBudget(),
        worldTransform: simd_float4x4 = matrix_identity_float4x4, options: SimplificationOptions = SimplificationOptions()
    ) throws -> [DetailLevel] {
        guard (1...LevelOfDetail.maximumLevelCount).contains(levelCount), triangleRatio > 0, triangleRatio < 1 else {
            throw GeometryProcessingError.invalidLevelOfDetail
        }
        try options.validate()
        let projection = try DetailProjection(budget: budget, worldTransform: worldTransform)
        var levels = [DetailLevel(geometry: self, error: 0, distance: 0, sourceVertexIndices: Array(vertices.indices))]
        let fullTriangleCount = indices.count / 3
        guard levelCount > 1, fullTriangleCount > 1 else { return levels }
        try validateSimplificationCapacity()

        var simplifier = QuadricSimplifier(self, options: options)
        let sourceRadius = projection.radius(of: self)
        var requested = Double(fullTriangleCount)
        for _ in 1..<levelCount {
            let previousCount = simplifier.triangleCount
            guard previousCount > 1 else { break }
            requested *= Double(triangleRatio)
            // When rounding repeats a count, request one fewer triangle so a ratio
            // close to one can still generate a distinct next level.
            let target = max(1, min(previousCount - 1, Int(requested.rounded())))
            simplifier.simplify(toTriangleCount: target)
            guard simplifier.triangleCount < previousCount else { break }

            let result = try simplifier.result()
            let radius = max(sourceRadius, projection.radius(of: result.geometry))
            let estimate = try projection.distance(forError: result.error, radius: radius)
            let last = levels[levels.count - 1]
            let distance = max(estimate, last.distance)
            let level = DetailLevel(
                geometry: result.geometry, error: result.error, distance: distance, sourceVertexIndices: result.sourceVertexIndices
            )
            if levels.count > 1, distance == last.distance {
                levels[levels.count - 1] = level
            } else {
                levels.append(level)
            }
        }
        return levels
    }
}

extension LevelOfDetail {
    /// Prepares a generated chain for `SceneNode.levelsOfDetail`, using the same
    /// distance selection as authored levels. Nonzero hysteresis can retain a coarser
    /// mesh inside its nominal switch distance. The factory prepares the renderer
    /// and chooses the material for each level.
    public init(
        chain: [MeshGeometry.DetailLevel], hysteresis: Float = 0,
        makeRenderer: (MeshGeometry) throws -> MeshRenderer
    ) throws {
        guard (1...Self.maximumLevelCount).contains(chain.count) else { throw Failure.levelCount }
        guard hysteresis.isFinite, (0..<1).contains(hysteresis) else { throw Failure.hysteresis(0) }
        for index in chain.indices {
            let distance = chain[index].distance
            guard distance.isFinite, distance >= 0, index == 0 || distance > chain[index - 1].distance else {
                throw Failure.distance(index)
            }
        }
        try self.init(levels: chain.map { level in
            Level(mesh: try makeRenderer(level.geometry), distance: level.distance, hysteresis: hysteresis)
        })
    }
}

private struct DetailProjection {
    let linearTransform: simd_double3x3
    let errorScale: Double
    let depthPerError: Double

    init(budget: MeshGeometry.ScreenSpaceErrorBudget, worldTransform: simd_float4x4) throws {
        let (x, y, z, offset) = worldTransform.columns
        guard budget.pixels.isFinite, budget.pixels > 0, budget.viewportHeight.isFinite, budget.viewportHeight > 0,
              budget.verticalFieldOfView > 0, budget.verticalFieldOfView < .pi,
              [x, y, z, offset].allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }),
              x.w == 0, y.w == 0, z.w == 0, offset.w == 1 else {
            throw MeshGeometry.GeometryProcessingError.invalidLevelOfDetail
        }
        let a = SIMD3(Double(x.x), Double(x.y), Double(x.z))
        let b = SIMD3(Double(y.x), Double(y.y), Double(y.z))
        let c = SIMD3(Double(z.x), Double(z.y), Double(z.z))
        let linear = simd_double3x3(a, b, c)
        guard linear.determinant != 0 else { throw MeshGeometry.GeometryProcessingError.invalidLevelOfDetail }
        linearTransform = linear

        // The square root of the largest absolute row sum of AᵀA bounds how
        // much A can stretch an error vector. It is exact for orthogonal axes.
        let ab = abs(simd_dot(a, b))
        let ac = abs(simd_dot(a, c))
        let bc = abs(simd_dot(b, c))
        errorScale = sqrt(max(simd_dot(a, a) + ab + ac, simd_dot(b, b) + ab + bc, simd_dot(c, c) + ac + bc))
        depthPerError = Double(budget.viewportHeight) / (2 * tan(Double(budget.verticalFieldOfView) / 2) * Double(budget.pixels))
    }

    func radius(of geometry: MeshGeometry) -> Double {
        var maximumSquared: Double = 0
        for index in geometry.indices {
            let worldOffset = linearTransform * SIMD3<Double>(geometry.vertices[Int(index)].position)
            maximumSquared = max(maximumSquared, simd_length_squared(worldOffset))
        }
        return maximumSquared.squareRoot()
    }

    func distance(forError error: Float, radius: Double) throws -> Float {
        let estimate = radius + Double(error) * errorScale * depthPerError
        let rounded = Float(estimate)
        let distance = Double(rounded) <= estimate ? rounded.nextUp : rounded
        guard distance.isFinite else { throw MeshGeometry.GeometryProcessingError.unrepresentableLevelOfDetailDistance }
        return distance
    }
}
