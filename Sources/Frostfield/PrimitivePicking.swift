import Foundation
import simd

/// How far from a drawn point, line or sprite a pick may land and still hit it,
/// beyond the element's own drawn extent.
public enum PrimitivePickingTolerance: Equatable, Sendable {
    /// Drawable pixels, independent of the element's depth.
    case pixels(Float)
    /// World units projected at the element's depth.
    case world(Float)
}

/// A point, line segment or sprite hit.
public struct PrimitiveHit: Equatable, Sendable {
    public enum Element: Equatable, Sendable {
        /// An index into the cloud's positions.
        case point(Int)
        /// An index into ``LineRenderer/drawnSegments`` for scene hits or
        /// ``LineGeometry/segments`` for a query on the geometry itself.
        case segment(Int)
        /// An index into the renderer's sprites.
        case sprite(Int)
    }

    public let nodeID: UUID
    public let element: Element
    /// Distance along the ray from its origin. Viewport rays start on the near plane.
    public let distance: Float
    /// The point's center, the nearest centerline point to the ray or viewport
    /// cursor, or the intersection of the viewport ray with the sprite's plane.
    public let position: SIMD3<Float>
    /// World distance from `position` to the closest point on a world ray, or
    /// to the viewport ray at the same projected depth.
    public let distanceToRay: Float
    /// The segment parameter from start (0) to end (1), for line hits.
    public let segmentParameter: Float?
    /// Top-left texture coordinate, including the sprite's texture region.
    /// A hit in the tolerance margin samples the nearest edge of the region.
    public let textureCoordinate: SIMD2<Float>?
}

/// An intersection with a drawable in a scene graph.
public enum SceneRaycastHit: Equatable, Sendable {
    case mesh(SceneHit)
    case primitive(PrimitiveHit)

    public var nodeID: UUID {
        switch self {
        case .mesh(let hit): hit.nodeID
        case .primitive(let hit): hit.nodeID
        }
    }

    public var distance: Float {
        switch self {
        case .mesh(let hit): hit.intersection.distance
        case .primitive(let hit): hit.distance
        }
    }

    /// The primary mesh UV or sprite texture coordinate, with a top-left origin.
    public var textureCoordinate: SIMD2<Float>? {
        switch self {
        case .mesh(let hit): hit.intersection.textureCoordinate
        case .primitive(let hit): hit.textureCoordinate
        }
    }
}

public extension SceneGraph {
    /// Picks the nearest point, line segment or sprite at top-left normalized
    /// coordinates. Points use their drawn square or disc, sprites their rotated
    /// rectangle, and lines their screen-space centerline and nominal half width.
    /// Meshes do not occlude this query. Use `pick(at:drawableSize:...)` to
    /// compare meshes and primitives by distance.
    func pickPrimitive(
        at normalized: SIMD2<Float>, drawableSize: SIMD2<Float>,
        tolerance: PrimitivePickingTolerance = .pixels(2)
    ) throws -> PrimitiveHit? {
        let query = try primitiveQuery(at: normalized, drawableSize: drawableSize, tolerance: tolerance)
        var nearest: PrimitiveHit?
        for node in try resolve(cameraPosition: query.cameraPosition) where renders(node) {
            try query.scan(node) { hit in
                if nearest.map({ hit.distance < $0.distance }) != false { nearest = hit }
            }
        }
        return nearest
    }

    /// Returns every mesh triangle and primitive element hit in one viewport,
    /// nearest first. Ties keep depth-first node order, then mesh hits before
    /// primitive hits on the same node. Alpha, dash gaps and render order are
    /// outside this geometric query. A custom render camera must use the depth
    /// mode of its projection. With no camera the graph's perspective is used.
    ///
    /// Pass the viewport's width and height in pixels. A supplied group limits
    /// hits to its direct children. An unknown group throws `EditingError.missingNode`.
    func raycast(
        at normalized: SIMD2<Float>, drawableSize: SIMD2<Float>, tolerance: PrimitivePickingTolerance = .pixels(2),
        camera: ShaderGraphCamera? = nil, depthMode: MeshDepthMode = .forward, in group: UUID? = nil
    ) throws -> [SceneRaycastHit] {
        let query = try primitiveQuery(
            at: normalized, drawableSize: drawableSize, tolerance: tolerance, camera: camera, depthMode: depthMode
        )
        let children = try pickingChildren(in: group)
        guard children?.isEmpty != true else { return [] }
        var result: [SceneRaycastHit] = []
        var meshHits: [SceneHit] = []
        for node in try resolve(cameraPosition: query.cameraPosition)
            where children?.contains(node.id) != false && renders(node) {
            try node.appendMeshHits(ray: query.ray, to: &meshHits)
            result.append(contentsOf: meshHits.lazy.map(SceneRaycastHit.mesh))
            meshHits.removeAll(keepingCapacity: true)
            try query.scan(node) { result.append(.primitive($0)) }
        }
        result.sort { $0.distance < $1.distance }
        return result
    }

    /// Returns the first hit of `raycast(at:drawableSize:...)` without collecting
    /// or sorting the other hits.
    func pick(
        at normalized: SIMD2<Float>, drawableSize: SIMD2<Float>, tolerance: PrimitivePickingTolerance = .pixels(2),
        camera: ShaderGraphCamera? = nil, depthMode: MeshDepthMode = .forward, in group: UUID? = nil
    ) throws -> SceneRaycastHit? {
        let query = try primitiveQuery(
            at: normalized, drawableSize: drawableSize, tolerance: tolerance, camera: camera, depthMode: depthMode
        )
        let children = try pickingChildren(in: group)
        guard children?.isEmpty != true else { return nil }
        var nearest: SceneRaycastHit?
        func consider(_ hit: SceneRaycastHit) {
            if nearest.map({ hit.distance < $0.distance }) != false { nearest = hit }
        }
        for node in try resolve(cameraPosition: query.cameraPosition)
            where children?.contains(node.id) != false && renders(node) {
            if let hit = try node.nearestMeshHit(ray: query.ray) { consider(.mesh(hit)) }
            try query.scan(node) { consider(.primitive($0)) }
        }
        return nearest
    }

    private func primitiveQuery(
        at normalized: SIMD2<Float>, drawableSize: SIMD2<Float>, tolerance: PrimitivePickingTolerance,
        camera explicit: ShaderGraphCamera? = nil, depthMode: MeshDepthMode = .forward
    ) throws -> PrimitivePickQuery {
        guard drawableSize.x.isFinite, drawableSize.y.isFinite, drawableSize.x > 0, drawableSize.y > 0 else {
            throw PrimitivePickingError.drawableSize
        }
        switch tolerance {
        case .pixels(let value), .world(let value):
            guard value.isFinite, value >= 0 else { throw PrimitivePickingError.tolerance }
        }
        let ray: SceneRay
        let view: simd_float4x4
        let cameraPosition: SIMD3<Float>
        if let explicit {
            ray = try explicit.ray(at: normalized, depthMode: depthMode)
            view = explicit.projectionMatrix * explicit.viewMatrix
            cameraPosition = try explicit.pickingPosition()
        } else {
            let aspect = drawableSize.x / drawableSize.y
            ray = try camera.ray(at: normalized, aspectRatio: aspect, cameraTransform: cameraTransform)
            view = try camera.projection(aspectRatio: aspect) * simd_inverse(cameraTransform.matrix)
            cameraPosition = cameraTransform.translation
        }
        return try PrimitivePickQuery(
            normalized: normalized, drawableSize: drawableSize, clipFromWorld: view,
            ray: ray, cameraPosition: cameraPosition, depthMode: explicit == nil ? .forward : depthMode,
            tolerance: tolerance
        )
    }
}

public enum PrimitivePickingError: LocalizedError, Equatable {
    case drawableSize
    case tolerance

    public var errorDescription: String? {
        switch self {
        case .drawableSize: "Picking primitives needs a finite, positive drawable size in pixels."
        case .tolerance: "A picking tolerance must be finite and nonnegative."
        }
    }
}

/// Mirrors the raster primitive viewport mapping. Working in homogeneous clip
/// coordinates preserves both off-axis perspective and orthographic sizes.
private struct PrimitivePickQuery {
    let clipFromWorld: simd_double4x4
    let worldFromClip: simd_double4x4
    let ndc: SIMD2<Double>
    let cursor: SIMD2<Double>
    let halfSize: SIMD2<Double>
    let pixelsPerUnit: Double
    let ray: SceneRay
    let cameraPosition: SIMD3<Float>
    let depthMode: MeshDepthMode
    let tolerance: PrimitivePickingTolerance

    init(
        normalized: SIMD2<Float>, drawableSize: SIMD2<Float>, clipFromWorld: simd_float4x4,
        ray: SceneRay, cameraPosition: SIMD3<Float>, depthMode: MeshDepthMode, tolerance: PrimitivePickingTolerance
    ) throws {
        guard primitiveFinite(clipFromWorld), let scale = PrimitiveView.verticalScale(of: clipFromWorld) else {
            throw PrimitiveRendererError.projection
        }
        self.clipFromWorld = simdPickingMatrix(clipFromWorld)
        worldFromClip = self.clipFromWorld.inverse
        halfSize = SIMD2<Double>(drawableSize) / 2
        ndc = SIMD2(2 * Double(normalized.x) - 1, 1 - 2 * Double(normalized.y))
        cursor = ndc * halfSize
        pixelsPerUnit = scale * halfSize.y
        self.ray = ray
        self.cameraPosition = cameraPosition
        self.depthMode = depthMode
        self.tolerance = tolerance
    }

    func pixels(_ world: Double, at clip: SIMD4<Double>) -> Double {
        world * pixelsPerUnit / clip.w
    }

    func slack(at clip: SIMD4<Double>) -> Double {
        switch tolerance {
        case .pixels(let pixels): Double(pixels)
        case .world(let units): self.pixels(Double(units), at: clip)
        }
    }

    func project(_ clip: SIMD4<Double>) -> SIMD2<Double> {
        SIMD2(clip.x, clip.y) / clip.w * halfSize
    }

    func inDepthRange(_ clip: SIMD4<Double>) -> Bool {
        let rounding = Double(Float.ulpOfOne) * clip.w
        return clip.w > 0 && clip.z >= -rounding && clip.z <= clip.w + rounding
    }

    func scan(_ node: ResolvedSceneNode, report: (PrimitiveHit) -> Void) throws {
        guard let primitive = node.primitive else { return }
        let worldFromLocal = simdPickingMatrix(node.worldTransform)
        let clipFromLocal = clipFromWorld * worldFromLocal
        func world(_ point: SIMD3<Double>) -> SIMD3<Double> {
            let transformed = worldFromLocal * SIMD4(point, 1)
            return SIMD3(transformed.x, transformed.y, transformed.z)
        }
        func consider(
            _ clip: SIMD4<Double>, _ element: PrimitiveHit.Element, position: SIMD3<Double>? = nil,
            parameter: Double? = nil, uv: SIMD2<Float>? = nil
        ) throws {
            guard inDepthRange(clip) else { return }
            let z = min(max(clip.z / clip.w, 0), 1)
            let homogeneous = worldFromClip * SIMD4(ndc.x, ndc.y, z, 1)
            let onRay = SIMD3(homogeneous.x, homogeneous.y, homogeneous.z) / homogeneous.w
            let along = simd_dot(onRay - SIMD3<Double>(ray.origin), SIMD3<Double>(ray.direction))
            let point = position ?? onRay
            let separation = Float(simd_distance(onRay, point))
            let rounded = SIMD3<Float>(point)
            guard along.isFinite, Float(along).isFinite, separation.isFinite,
                  rounded.x.isFinite, rounded.y.isFinite, rounded.z.isFinite else {
                throw ScenePickingError.unrepresentableResult
            }
            report(PrimitiveHit(
                nodeID: node.id, element: element, distance: max(0, Float(along)), position: rounded,
                distanceToRay: separation, segmentParameter: parameter.map(Float.init), textureCoordinate: uv
            ))
        }

        switch primitive {
        case .points(let renderer):
            let material = renderer.material
            for index in renderer.activePoints {
                let point = renderer.cloud.positions[index]
                let local = SIMD3<Double>(point)
                let clip = clipFromLocal * SIMD4(local, 1)
                let size = Double(renderer.cloud.sizes?[index] ?? 1)
                guard inDepthRange(clip), size > 0 else { continue }
                let diameter: Double = switch material.size {
                case .world(let width): pixels(Double(width) * size, at: clip)
                case .pixels(let width): Double(width) * size
                }
                let limit = max(diameter, 1) / 2 + slack(at: clip)
                let offset = cursor - project(clip)
                let inside = material.shape == .circle ? simd_length(offset) <= limit
                    : max(abs(offset.x), abs(offset.y)) <= limit
                if inside { try consider(clip, .point(index), position: world(local)) }
            }
        case .lines(let renderer):
            let positions = renderer.geometry.positions
            for (index, segment) in renderer.drawnSegments.enumerated() {
                let first = SIMD3<Double>(positions[segment.start])
                let last = SIMD3<Double>(positions[segment.end])
                let start = clipFromLocal * SIMD4(first, 1)
                let end = clipFromLocal * SIMD4(last, 1)
                var lo = 0.0, hi = 1.0
                guard Self.trim(start.z, end.z, lo: &lo, hi: &hi),
                      Self.trim(start.w - start.z, end.w - end.z, lo: &lo, hi: &hi) else { continue }
                let a = start + (end - start) * lo
                let b = start + (end - start) * hi
                guard a.w > 0, b.w > 0 else { continue }
                let pixelA = project(a)
                let delta = project(b) - pixelA
                let lengthSquared = simd_length_squared(delta)
                let u = lengthSquared > 0 ? min(max(simd_dot(cursor - pixelA, delta) / lengthSquared, 0), 1)
                    : closerEndpoint(a, b)
                let separation = simd_distance(cursor, pixelA + delta * u)
                // Convert linear screen interpolation to the original segment parameter.
                let t = u * a.w / ((1 - u) * b.w + u * a.w)
                let parameter = lo + (hi - lo) * t
                let clip = a + (b - a) * t
                let halfWidth: Double = switch renderer.material.width {
                case .hairline: 0.5
                case .pixels(let width): Double(width) / 2
                case .world(let width): pixels(Double(width) / 2, at: clip)
                }
                guard separation <= halfWidth + slack(at: clip) else { continue }
                try consider(clip, .segment(index), position: world(first + (last - first) * parameter), parameter: parameter)
            }
        case .sprites(let renderer):
            for (index, sprite) in renderer.sprites.enumerated() {
                let clip = clipFromLocal * SIMD4(SIMD3<Double>(sprite.position), 1)
                guard inDepthRange(clip) else { continue }
                let unit = renderer.material.sizeAttenuation ? pixels(1, at: clip) : 1
                guard let uv = spriteTextureCoordinate(sprite, offset: cursor - project(clip), scale: unit,
                                                       tolerance: slack(at: clip)) else { continue }
                try consider(clip, .sprite(index), uv: uv)
            }
        }
    }

    /// Keeps the part of the segment inside a homogeneous clip plane.
    private static func trim(_ start: Double, _ end: Double, lo: inout Double, hi: inout Double) -> Bool {
        if start < 0 && end < 0 { return false }
        if start < 0 { lo = max(lo, start / (start - end)) }
        if end < 0 { hi = min(hi, start / (start - end)) }
        return lo <= hi
    }

    private func closerEndpoint(_ a: SIMD4<Double>, _ b: SIMD4<Double>) -> Double {
        let aDepth = a.z / a.w
        let bDepth = b.z / b.w
        let aIsNearer = depthMode == .reversed ? aDepth >= bDepth : aDepth <= bDepth
        return aIsNearer ? 0 : 1
    }
}

/// The renderer rotates a pixel-sized rectangle around its authored anchor.
/// Both viewport and spatial raycasts invert that mapping into atlas UVs.
func spriteTextureCoordinate(
    _ sprite: Sprite, offset: SIMD2<Double>, scale: Double, tolerance: Double
) -> SIMD2<Float>? {
    let size = SIMD2<Double>(sprite.size) * scale
    guard size.x > 0, size.y > 0 else { return nil }
    let angle = Double(sprite.rotation)
    let c = cos(angle), s = sin(angle)
    let local = SIMD2(c * offset.x + s * offset.y, s * offset.x - c * offset.y)
    let corner = SIMD2<Double>(sprite.center) + local / size
    let margin = tolerance / size
    guard corner.x >= -margin.x, corner.x <= 1 + margin.x,
          corner.y >= -margin.y, corner.y <= 1 + margin.y else { return nil }
    let region = sprite.textureRegion
    let clamped = SIMD2<Float>(simd_clamp(corner, .zero, SIMD2(repeating: 1)))
    return SIMD2(region.x, region.y) + clamped * SIMD2(region.z, region.w)
}
