import Foundation
import simd

/// Positions and optional per-point attributes for a point cloud, like the
/// geometry of a Three.js `Points` object.
public struct PointCloud: Sendable {
    /// The most points one cloud can hold: 2²⁶, about 1.6 GB of point records.
    public static let maximumPointCount = 1 << 26

    /// Positions in the owning node's local space.
    public let positions: [SIMD3<Float>]
    /// Optional linear RGBA per point, with straight alpha, multiplied by the
    /// material color, like `PointsMaterial.vertexColors`. Uploaded as half floats.
    public let colors: [SIMD4<Float>]?
    /// Optional nonnegative per-point multipliers of the material size. Zero
    /// hides a point in every blending mode and from picking.
    public let sizes: [Float]?
    /// The box around every position.
    public let bounds: AxisAlignedBox

    /// Positions must be finite. Colors need finite components in 0…65,504 and
    /// alpha in 0…1. Sizes must be finite and nonnegative. Attribute arrays must
    /// match the position count.
    public init(positions: [SIMD3<Float>], colors: [SIMD4<Float>]? = nil, sizes: [Float]? = nil) throws {
        guard (1...Self.maximumPointCount).contains(positions.count) else { throw PrimitiveGeometryError.pointCount }
        if let index = positions.firstIndex(where: { !primitiveFinite($0) }) {
            throw PrimitiveGeometryError.invalidPosition(index)
        }
        try validateColors(colors, count: positions.count)
        if let sizes {
            guard sizes.count == positions.count else { throw PrimitiveGeometryError.attributeCount }
            if let index = sizes.firstIndex(where: { !$0.isFinite || $0 < 0 }) {
                throw PrimitiveGeometryError.invalidSize(index)
            }
        }
        self.positions = positions
        self.colors = colors
        self.sizes = sizes
        bounds = try AxisAlignedBox(enclosing: positions)
    }
}

/// One drawn line segment and the points it joins with.
public struct LineSegment: Equatable, Sendable {
    /// Indices into ``LineGeometry/positions``.
    public let start: Int
    public let end: Int
    /// The point before `start` whose segment shares a join with this one, or nil for an end cap.
    public let previous: Int?
    /// The point after `end` whose segment shares a join with this one, or nil for an end cap.
    public let next: Int?
    /// The accumulated world-space length at each end, like Three.js
    /// `computeLineDistances()`. Dash patterns are measured in these units.
    public let startDistance: Float
    public let endDistance: Float
}

/// Points connected as a polyline, a closed loop or separate segments, like the
/// geometry of a Three.js `Line`, `LineLoop` or `LineSegments`.
public struct LineGeometry: Sendable {
    /// The most points one line can hold.
    public static let maximumPointCount = 1 << 24

    public enum Topology: Equatable, Sendable {
        /// Consecutive points form a polyline with joins, like `Line`.
        case strip
        /// A polyline that also joins the last point back to the first, like `LineLoop`.
        case loop
        /// Each pair of points forms an independent segment with caps, like `LineSegments`.
        case segments
    }

    /// Positions in the owning node's local space.
    public let positions: [SIMD3<Float>]
    /// Optional linear RGBA per point, straight alpha, interpolated along each
    /// segment, like `LineBasicMaterial.vertexColors`. Uploaded as half floats.
    public let colors: [SIMD4<Float>]?
    public let topology: Topology
    /// The drawn segments in order. A strip or loop skips segments whose ends
    /// coincide, so the segments around them join directly.
    public let segments: [LineSegment]
    /// The box around every position.
    public let bounds: AxisAlignedBox

    /// A strip needs at least two points, a loop at least three, and separate
    /// segments a positive even count. Positions must be finite; colors follow
    /// ``PointCloud`` rules. Distances accumulate in `Double` along the drawn
    /// segments; separate segments continue the previous segment's distance, as
    /// Three.js does, and a loop's closing segment ends at the full perimeter.
    public init(positions: [SIMD3<Float>], colors: [SIMD4<Float>]? = nil, topology: Topology = .strip) throws {
        guard positions.count <= Self.maximumPointCount else { throw PrimitiveGeometryError.pointCount }
        switch topology {
        case .strip: guard positions.count >= 2 else { throw PrimitiveGeometryError.lineTopology }
        case .loop: guard positions.count >= 3 else { throw PrimitiveGeometryError.lineTopology }
        case .segments:
            guard positions.count >= 2, positions.count.isMultiple(of: 2) else { throw PrimitiveGeometryError.lineTopology }
        }
        if let index = positions.firstIndex(where: { !primitiveFinite($0) }) {
            throw PrimitiveGeometryError.invalidPosition(index)
        }
        try validateColors(colors, count: positions.count)
        self.positions = positions
        self.colors = colors
        self.topology = topology
        segments = try Self.segments(positions: positions, topology: topology)
        bounds = try AxisAlignedBox(enclosing: positions)
    }

    private static func segments(positions: [SIMD3<Float>], topology: Topology) throws -> [LineSegment] {
        func length(_ a: Int, _ b: Int) -> Double {
            simd_distance(SIMD3<Double>(positions[a]), SIMD3<Double>(positions[b]))
        }
        if topology == .segments {
            var distance = 0.0
            return try stride(from: 0, to: positions.count, by: 2).map { index in
                let start = distance
                distance += length(index, index + 1)
                return try segment(index, index + 1, nil, nil, start, distance)
            }
        }
        let closed = topology == .loop
        var pairs: [(Int, Int)] = []
        let count = closed ? positions.count : positions.count - 1
        for index in 0..<count {
            let end = (index + 1) % positions.count
            if positions[index] != positions[end] { pairs.append((index, end)) }
        }
        // A line whose points all coincide still draws its caps once.
        if pairs.isEmpty { return [try segment(0, 1, nil, nil, 0, 0)] }
        var distance = 0.0
        return try pairs.indices.map { offset in
            let (start, end) = pairs[offset]
            let before = offset > 0 ? pairs[offset - 1].0 : (closed ? pairs[pairs.count - 1].0 : nil)
            let after = offset + 1 < pairs.count ? pairs[offset + 1].1 : (closed ? pairs[0].1 : nil)
            let startDistance = distance
            distance += length(start, end)
            return try segment(start, end, before, after, startDistance, distance)
        }
    }

    /// Rebuilds connectivity at the range edges. The selected line retains its
    /// starting distance from the source so existing dash patterns stay in phase.
    func segments(in range: Range<Int>) throws -> [LineSegment] {
        guard range.count >= 2 else { return [] }
        if range.count == positions.count { return segments }
        let first = range.lowerBound
        let phase: Double
        if topology == .segments {
            let original = segments[first / 2]
            phase = Double(first.isMultiple(of: 2) ? original.startDistance : original.endDistance)
        } else {
            phase = Double(segments.first(where: { $0.start >= first })?.startDistance ?? segments.last?.endDistance ?? 0)
        }
        return try Self.segments(positions: Array(positions[range]), topology: topology).map { segment in
            try Self.segment(
                first + segment.start, first + segment.end, segment.previous.map { first + $0 },
                segment.next.map { first + $0 }, phase + Double(segment.startDistance), phase + Double(segment.endDistance)
            )
        }
    }

    private static func segment(
        _ start: Int, _ end: Int, _ previous: Int?, _ next: Int?, _ startDistance: Double, _ endDistance: Double
    ) throws -> LineSegment {
        let first = Float(startDistance)
        let last = Float(endDistance)
        guard first.isFinite, last.isFinite else { throw PrimitiveGeometryError.unrepresentableDistance }
        return LineSegment(
            start: start, end: end, previous: previous, next: next, startDistance: first, endDistance: last
        )
    }
}

/// One camera-facing rectangle, like a Three.js `Sprite`.
public struct Sprite: Equatable, Sendable {
    /// The anchor's position in the owning node's local space.
    public var position: SIMD3<Float>
    /// Width and height in world units, or in pixels when the material does not
    /// attenuate size. Three.js takes this from the sprite's scale.
    public var size: SIMD2<Float>
    /// Counterclockwise rotation on screen about the anchor, in radians.
    public var rotation: Float
    /// The anchor as a top-left texture coordinate: (0.5, 0.5) is the middle
    /// and (0, 0) the top-left corner. Three.js measures `center` from the
    /// bottom-left; Frostfield uses its top-left texture convention throughout.
    public var center: SIMD2<Float>
    /// Linear RGBA tint with straight alpha, multiplied by the material color.
    public var color: SIMD4<Float>
    /// The texture region the sprite shows, as a top-left origin and size in
    /// normalized coordinates, for atlases and sprite sheets.
    public var textureRegion: SIMD4<Float>

    public init(
        position: SIMD3<Float>, size: SIMD2<Float>, rotation: Float = 0, center: SIMD2<Float> = SIMD2(0.5, 0.5),
        color: SIMD4<Float> = SIMD4(repeating: 1), textureRegion: SIMD4<Float> = SIMD4(0, 0, 1, 1)
    ) {
        self.position = position
        self.size = size
        self.rotation = rotation
        self.center = center
        self.color = color
        self.textureRegion = textureRegion
    }

    var isValid: Bool {
        primitiveFinite(position) && size.x.isFinite && size.y.isFinite && size.x >= 0 && size.y >= 0
            && rotation.isFinite && center.x.isFinite && center.y.isFinite
            && isValidPrimitiveColor(color)
            && textureRegion.x.isFinite && textureRegion.y.isFinite && textureRegion.z.isFinite && textureRegion.w.isFinite
    }

    /// The largest distance from the anchor to a corner, in the size's units.
    var reach: Float {
        let x = size.x * max(abs(center.x), abs(1 - center.x))
        let y = size.y * max(abs(center.y), abs(1 - center.y))
        return (x * x + y * y).squareRoot()
    }
}

public enum PrimitiveGeometryError: LocalizedError, Equatable {
    case pointCount
    case lineTopology
    case invalidPosition(Int)
    case attributeCount
    case invalidColor(Int)
    case invalidSize(Int)
    case unrepresentableDistance
    case invalidThresholdAngle
    case invalidHelperParameter(String)

    public var errorDescription: String? {
        switch self {
        case .pointCount:
            "Use 1 to \(PointCloud.maximumPointCount) points for a cloud and at most \(LineGeometry.maximumPointCount) for a line."
        case .lineTopology:
            "A line strip needs at least two points, a loop three, and separate segments a positive even count."
        case .invalidPosition(let index): "Position \(index) is not finite."
        case .attributeCount: "Per-point attributes need one value for each position."
        case .invalidColor(let index):
            "Color \(index) needs finite RGB in 0 to 65,504 and alpha in 0 to 1."
        case .invalidSize(let index): "Point size \(index) must be finite and nonnegative."
        case .unrepresentableDistance: "The accumulated line length exceeds the Float range."
        case .invalidThresholdAngle: "The edge threshold angle must be finite and in 0 to π radians."
        case .invalidHelperParameter(let detail): "Invalid helper parameter: \(detail)."
        }
    }
}

func primitiveFinite(_ value: SIMD3<Float>) -> Bool {
    value.x.isFinite && value.y.isFinite && value.z.isFinite
}

func isValidPrimitiveColor(_ color: SIMD4<Float>) -> Bool {
    (0..<3).allSatisfy { color[$0].isFinite && color[$0] >= 0 && color[$0] <= 65_504 }
        && color.w.isFinite && (0...1).contains(color.w)
}

private func validateColors(_ colors: [SIMD4<Float>]?, count: Int) throws {
    guard let colors else { return }
    guard colors.count == count else { throw PrimitiveGeometryError.attributeCount }
    if let index = colors.firstIndex(where: { !isValidPrimitiveColor($0) }) {
        throw PrimitiveGeometryError.invalidColor(index)
    }
}
