import Foundation
import simd

/// A world ray's closest approach to a point or segment, or a hit on a sprite.
public struct PrimitiveRayHit: Equatable, Sendable {
    /// The source point index, drawn segment index, or sprite index.
    public let element: PrimitiveHit.Element
    /// Distance from the ray origin to the closest point on the ray.
    public let distance: Float
    /// The point center, closest point on a segment, or sprite intersection in world space.
    public let position: SIMD3<Float>
    /// The closest point on the ray, in world space.
    public let rayPosition: SIMD3<Float>
    /// World-space separation of the two positions.
    public let distanceToRay: Float
    /// The closest point's parameter on the segment, from zero at its start to one at its end.
    public let segmentParameter: Float?
    /// The sprite's top-left texture coordinate, including its atlas region.
    public let textureCoordinate: SIMD2<Float>?

    fileprivate func onNode(_ id: UUID) -> PrimitiveHit {
        PrimitiveHit(
            nodeID: id, element: element, distance: distance, position: position,
            distanceToRay: distanceToRay, segmentParameter: segmentParameter, textureCoordinate: textureCoordinate
        )
    }
}

public extension PointCloud {
    /// Returns every point center within the inclusive world-space threshold of
    /// the ray. Zero-size points are hidden. Drawable size and point shape do
    /// not affect the query. The closest ray position must be within the ray's
    /// maximum distance. Hits are sorted by distance, then by point index.
    func intersections(
        ray: SceneRay, localToWorld: simd_float4x4 = matrix_identity_float4x4, threshold: Float = 1
    ) throws -> [PrimitiveRayHit] {
        try intersections(ray: ray, localToWorld: localToWorld, threshold: threshold, drawRange: .all)
    }

    /// Restricts the linear scan to source indices selected by the draw range.
    func intersections(
        ray: SceneRay, localToWorld: simd_float4x4 = matrix_identity_float4x4, threshold: Float = 1,
        drawRange: GeometryDrawRange
    ) throws -> [PrimitiveRayHit] {
        let query = try PrimitiveWorldRay(ray, threshold: threshold, localToWorld: localToWorld)
        var hits: [PrimitiveRayHit] = []
        try query.scan(self, range: drawRange.resolve(elementCount: positions.count)) { hits.append($0) }
        return hits.sorted { $0.distance < $1.distance }
    }
}

public extension LineGeometry {
    /// Returns every segment within the inclusive world-space threshold of the
    /// ray. The query uses centerlines, regardless of stroke width and dashes.
    /// The closest ray position must be within the ray's maximum distance.
    /// Hits are sorted by distance, then by the index of the drawn segment.
    func intersections(
        ray: SceneRay, localToWorld: simd_float4x4 = matrix_identity_float4x4, threshold: Float = 1
    ) throws -> [PrimitiveRayHit] {
        let query = try PrimitiveWorldRay(ray, threshold: threshold, localToWorld: localToWorld)
        var hits: [PrimitiveRayHit] = []
        try query.scan(positions: positions, segments: segments) { hits.append($0) }
        return hits.sorted { $0.distance < $1.distance }
    }
}

public extension SpriteRenderer {
    /// Intersects an arbitrary world ray with each billboard as it faces the
    /// supplied render camera. Pixel sizes use the viewport's width and height.
    /// The query tests the sprite rectangle before texture alpha.
    func intersections(
        ray: SceneRay, localToWorld: simd_float4x4 = matrix_identity_float4x4,
        camera: ShaderGraphCamera, drawableSize: SIMD2<Float>
    ) throws -> [PrimitiveRayHit] {
        _ = try PickingAffine(localToWorld)
        let view = try SpriteWorldRay(ray: ray, camera: camera, drawableSize: drawableSize)
        var hits: [PrimitiveRayHit] = []
        try view.scan(self, localToWorld: localToWorld) { hits.append($0) }
        return hits.sorted { $0.distance < $1.distance }
    }
}

public extension SceneGraph {
    /// Casts an arbitrary world ray against visible meshes, instances, batched
    /// slots, points and lines. Point and line thresholds use world units.
    /// The supplied layer mask replaces the graph camera's mask. The graph
    /// camera position selects mesh levels of detail. A group restricts hits to
    /// its direct children. Use the viewport overload for camera-facing sprites.
    ///
    /// Hits sort by distance from the ray origin. Ties preserve depth-first
    /// node order, with mesh hits before primitives on the same node.
    func raycast(
        ray: SceneRay, pointThreshold: Float = 1, lineThreshold: Float = 1,
        layers: SceneLayers? = nil, in group: UUID? = nil
    ) throws -> [SceneRaycastHit] {
        try worldRaycast(ray: ray, pointThreshold: pointThreshold, lineThreshold: lineThreshold,
                         layers: layers, group: group, spriteView: nil)
    }

    /// Casts a world ray and includes sprites facing the supplied camera.
    /// Use the dimensions of its viewport for pixel-sized sprites.
    func raycast(
        ray: SceneRay, camera: ShaderGraphCamera, drawableSize: SIMD2<Float>,
        pointThreshold: Float = 1, lineThreshold: Float = 1, layers: SceneLayers? = nil, in group: UUID? = nil
    ) throws -> [SceneRaycastHit] {
        let view = try SpriteWorldRay(ray: ray, camera: camera, drawableSize: drawableSize)
        return try worldRaycast(ray: ray, pointThreshold: pointThreshold, lineThreshold: lineThreshold,
                                layers: layers, group: group, spriteView: view)
    }

    private func worldRaycast(
        ray: SceneRay, pointThreshold: Float, lineThreshold: Float,
        layers: SceneLayers?, group: UUID?, spriteView: SpriteWorldRay?
    ) throws -> [SceneRaycastHit] {
        try PrimitiveWorldRay.validate(pointThreshold)
        try PrimitiveWorldRay.validate(lineThreshold)
        let children = try pickingChildren(in: group)
        guard children?.isEmpty != true else { return [] }
        let mask = layers ?? cameraLayers
        var result: [SceneRaycastHit] = []
        var meshHits: [SceneHit] = []
        let position = spriteView?.cameraPosition ?? cameraTransform.translation
        for node in try resolve(cameraPosition: position)
            where node.isVisible && node.layers.intersects(mask) && children?.contains(node.id) != false {
            try node.appendMeshHits(ray: ray, to: &meshHits)
            result.append(contentsOf: meshHits.lazy.map(SceneRaycastHit.mesh))
            meshHits.removeAll(keepingCapacity: true)
            func append(_ hit: PrimitiveRayHit) { result.append(.primitive(hit.onNode(node.id))) }
            switch node.primitive {
            case .points(let renderer):
                let query = try PrimitiveWorldRay(ray, threshold: pointThreshold, localToWorld: node.worldTransform)
                if let bvh = renderer.pickingBVH {
                    try bvh.intersections(query, range: renderer.activePoints).forEach(append)
                } else {
                    try query.scan(renderer.cloud, range: renderer.activePoints, report: append)
                }
            case .lines(let renderer):
                let query = try PrimitiveWorldRay(ray, threshold: lineThreshold, localToWorld: node.worldTransform)
                try query.scan(positions: renderer.geometry.positions, segments: renderer.drawnSegments, report: append)
            case .sprites(let renderer):
                try spriteView?.scan(renderer, localToWorld: node.worldTransform, report: append)
            case nil: break
            }
        }
        result.sort { $0.distance < $1.distance }
        return result
    }
}

struct PrimitiveWorldRay {
    let origin: SIMD3<Double>
    let direction: SIMD3<Double>
    let maximumDistance: Double
    private let threshold: Double
    private let thresholdSquared: Double
    let linear: simd_double3x3
    let translationFromOrigin: SIMD3<Double>

    init(_ ray: SceneRay, threshold: Float, localToWorld: simd_float4x4) throws {
        try Self.validate(threshold)
        let transform = try PickingAffine(localToWorld)
        origin = SIMD3<Double>(ray.origin)
        direction = simd_normalize(SIMD3<Double>(ray.direction))
        maximumDistance = Double(ray.maximumDistance)
        self.threshold = Double(threshold)
        thresholdSquared = Double(threshold) * Double(threshold)
        linear = transform.linear
        translationFromOrigin = transform.translation - origin
    }

    static func validate(_ threshold: Float) throws {
        guard threshold.isFinite, threshold >= 0 else { throw PrimitivePickingError.tolerance }
    }

    func scan(_ cloud: PointCloud, range: Range<Int>, report: (PrimitiveRayHit) -> Void) throws {
        for index in range {
            if let hit = try pointHit(in: cloud, index: index) { report(hit) }
        }
    }

    func pointHit(in cloud: PointCloud, index: Int) throws -> PrimitiveRayHit? {
        guard cloud.sizes?[index] != 0 else { return nil }
        return try hit(.point(index), offset: offset(of: cloud.positions[index]))
    }

    /// A conservative broad phase for the world-space point tolerance tube.
    /// Expansion also covers the near cap at the ray origin.
    func overlapsPointBounds(_ bounds: AxisAlignedBox) -> Bool {
        let relative = CollisionBounds.transformed(
            lower: bounds.minimum, upper: bounds.maximum, linear: linear, translation: translationFromOrigin
        )
        let scale = simd_reduce_max(simd_max(simd_abs(relative.minimum), simd_abs(relative.maximum)))
        // Cross products, projection and slab division can round differently
        // at an exact tolerance or ray limit.
        let padding = threshold + (32 * Double.ulpOfOne) * max(scale, threshold)
        let lower = relative.minimum - padding
        let upper = relative.maximum + padding
        var entry = 0.0, exit = maximumDistance
        for axis in 0..<3 {
            let step = direction[axis]
            if step == 0 {
                if lower[axis] > 0 || upper[axis] < 0 { return false }
            } else {
                let first = lower[axis] / step, second = upper[axis] / step
                entry = max(entry, min(first, second))
                exit = min(exit, max(first, second))
                if entry > exit { return false }
            }
        }
        return true
    }

    func scan(positions: [SIMD3<Float>], segments: [LineSegment], report: (PrimitiveRayHit) -> Void) throws {
        for (index, segment) in segments.enumerated() {
            let a = positions[segment.start]
            let b = positions[segment.end]
            let start = offset(of: a)
            let edge = linear * (SIMD3<Double>(b) - SIMD3<Double>(a))
            let parameter = raySegmentParameter(start: start, edge: edge, direction: direction, maximumDistance: .infinity)
            if let hit = try hit(.segment(index), offset: start + edge * parameter, parameter: parameter) { report(hit) }
        }
    }

    private func offset(of local: SIMD3<Float>) -> SIMD3<Double> {
        linear * SIMD3<Double>(local) + translationFromOrigin
    }

    private func hit(
        _ element: PrimitiveHit.Element, offset: SIMD3<Double>, parameter: Double? = nil
    ) throws -> PrimitiveRayHit? {
        let projected = simd_dot(offset, direction)
        let along = max(0, projected)
        guard along <= maximumDistance else { return nil }
        let closest = direction * along
        let separation = projected >= 0
            ? simd_length_squared(simd_cross(offset, direction)) : simd_length_squared(offset)
        guard separation <= thresholdSquared else { return nil }
        let distance = Float(along)
        let position = SIMD3<Float>(origin + offset)
        let rayPosition = SIMD3<Float>(origin + closest)
        let distanceToRay = Float(separation.squareRoot())
        guard distance.isFinite, position.isFiniteVector, rayPosition.isFiniteVector, distanceToRay.isFinite else {
            throw ScenePickingError.unrepresentableResult
        }
        return PrimitiveRayHit(
            element: element, distance: distance, position: position, rayPosition: rayPosition,
            distanceToRay: distanceToRay, segmentParameter: parameter.map(Float.init), textureCoordinate: nil
        )
    }
}

/// A billboard lies at its anchor's clip depth and uses the render viewport's
/// pixel scale. A controller ray can cross this plane at any angle.
private struct SpriteWorldRay {
    let clipFromWorld: simd_double4x4
    let origin: SIMD3<Double>
    let direction: SIMD3<Double>
    let maximumDistance: Double
    let clipOrigin: SIMD4<Double>
    let clipDirection: SIMD4<Double>
    let halfSize: SIMD2<Double>
    let pixelsPerUnit: Double
    let cameraPosition: SIMD3<Float>

    init(ray: SceneRay, camera: ShaderGraphCamera, drawableSize: SIMD2<Float>) throws {
        guard drawableSize.x.isFinite, drawableSize.y.isFinite, drawableSize.x > 0, drawableSize.y > 0 else {
            throw PrimitivePickingError.drawableSize
        }
        let projection = camera.projectionMatrix * camera.viewMatrix
        guard primitiveFinite(projection), let scale = PrimitiveView.verticalScale(of: projection) else {
            throw PrimitiveRendererError.projection
        }
        clipFromWorld = simdPickingMatrix(projection)
        origin = SIMD3<Double>(ray.origin)
        direction = simd_normalize(SIMD3<Double>(ray.direction))
        maximumDistance = Double(ray.maximumDistance)
        clipOrigin = clipFromWorld * SIMD4(origin, 1)
        clipDirection = clipFromWorld * SIMD4(direction, 0)
        halfSize = SIMD2<Double>(drawableSize) / 2
        pixelsPerUnit = scale * halfSize.y
        cameraPosition = try camera.pickingPosition()
    }

    func scan(
        _ renderer: SpriteRenderer, localToWorld: simd_float4x4, report: (PrimitiveRayHit) -> Void
    ) throws {
        let clipFromLocal = clipFromWorld * simdPickingMatrix(localToWorld)
        for (index, sprite) in renderer.sprites.enumerated() {
            let anchor = clipFromLocal * SIMD4(SIMD3<Double>(sprite.position), 1)
            let rounding = Double(Float.ulpOfOne) * anchor.w
            guard anchor.w > 0, anchor.z >= -rounding, anchor.z <= anchor.w + rounding else { continue }
            // Solve z/w = anchor.z/anchor.w along the world ray.
            let denominator = clipDirection.z * anchor.w - anchor.z * clipDirection.w
            guard denominator != 0 else { continue }
            let distance = (anchor.z * clipOrigin.w - clipOrigin.z * anchor.w) / denominator
            guard distance.isFinite, distance >= 0, distance <= maximumDistance else { continue }
            let hit = clipOrigin + clipDirection * distance
            guard hit.w > 0 else { continue }
            let offset = (SIMD2(hit.x, hit.y) / hit.w - SIMD2(anchor.x, anchor.y) / anchor.w) * halfSize
            let scale = renderer.material.sizeAttenuation ? pixelsPerUnit / anchor.w : 1
            guard let uv = spriteTextureCoordinate(sprite, offset: offset, scale: scale, tolerance: 0) else { continue }
            let position = SIMD3<Float>(origin + direction * distance)
            guard Float(distance).isFinite, position.isFiniteVector else { throw ScenePickingError.unrepresentableResult }
            report(PrimitiveRayHit(
                element: .sprite(index), distance: Float(distance), position: position, rayPosition: position,
                distanceToRay: 0, segmentParameter: nil, textureCoordinate: uv
            ))
        }
    }
}
