import Foundation
import simd

/// One radius and height in a Y-axis lathe profile.
public struct LatheProfilePoint: Equatable, Sendable {
    public let radius: Float
    public let y: Float
    /// An optional authored normal in radial/Y coordinates.
    public let normal: SIMD2<Float>?
    /// Splits the incoming and outgoing normals when no authored normal is supplied.
    public let hard: Bool

    public init(radius: Float, y: Float, normal: SIMD2<Float>? = nil, hard: Bool = false) {
        self.radius = radius
        self.y = y
        self.normal = normal
        self.hard = hard
    }
}

extension MeshGeometry {
    /// Revolves a bottom-to-top profile around Y. Heights may repeat for a radial
    /// shoulder. The radius may reach zero at either endpoint, closing that end
    /// with pole triangles. Positive-radius caps use independent planar UVs.
    ///
    /// Side U wraps from +X. Side V increases down the profile by arc length.
    /// Supply 2...256 profile points and 3...256 radial segments.
    public static func lathe(
        profile: [LatheProfilePoint],
        segments: Int = 64,
        capBottom: Bool = true,
        capTop: Bool = true
    ) throws -> MeshGeometry {
        guard (2...256).contains(profile.count) else { throw SolidValidationError.profileCount }
        guard (3...256).contains(segments) else { throw SolidValidationError.latheSegments }

        for (index, point) in profile.enumerated() {
            let endpoint = index == 0 || index == profile.count - 1
            guard point.y.isFinite,
                  (point.radius.isNormal && point.radius > 0) || (endpoint && point.radius == 0) else {
                throw SolidValidationError.profilePoint(index)
            }
            if let normal = point.normal,
               !normal.x.isFinite || !normal.y.isFinite || normal == .zero {
                throw SolidValidationError.profileNormal(index)
            }
        }
        guard profile[0].y < profile[profile.count - 1].y else {
            throw SolidValidationError.profileHeight
        }

        var edgeNormals: [SIMD2<Double>] = []
        var distances: [Double] = [0]
        var previousShoulder: Double = 0
        for index in 0..<(profile.count - 1) {
            let a = profile[index]
            let b = profile[index + 1]
            let radial = Double(b.radius) - Double(a.radius)
            let vertical = Double(b.y) - Double(a.y)
            guard vertical >= 0, radial != 0 || vertical != 0,
                  a.radius != 0 || b.radius != 0 else {
                throw SolidValidationError.profileEdge(index)
            }
            if vertical == 0 {
                guard previousShoulder * radial >= 0 else { throw SolidValidationError.profileEdge(index) }
                if (capBottom && a.y == profile[0].y && radial < 0)
                    || (capTop && b.y == profile[profile.count - 1].y && radial > 0) {
                    throw SolidValidationError.overlappingCap(index)
                }
            }
            previousShoulder = vertical == 0 ? radial : 0
            let length = hypot(radial, vertical)
            edgeNormals.append(SIMD2(vertical, -radial) / length)
            distances.append(distances[index] + length)
        }

        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        var rings: [(incoming: UInt32, outgoing: UInt32)] = []
        vertices.reserveCapacity((segments + 1) * (profile.count + 2) + 2)
        indices.reserveCapacity(6 * segments * profile.count)
        let length = distances[profile.count - 1]

        for (index, point) in profile.enumerated() {
            let incoming = edgeNormals[max(0, index - 1)]
            let outgoing = edgeNormals[min(index, edgeNormals.count - 1)]
            let v = Float(1 - distances[index] / length)
            let firstNormal: SIMD2<Double>
            if let authored = point.normal {
                firstNormal = simd_normalize(SIMD2<Double>(authored))
            } else if point.hard {
                firstNormal = incoming
            } else {
                firstNormal = simd_normalize(incoming + outgoing)
            }
            let first = appendLatheRing(point, normal: firstNormal, v: v, segments: segments, into: &vertices)
            let second = point.hard && point.normal == nil && incoming != outgoing
                ? appendLatheRing(point, normal: outgoing, v: v, segments: segments, into: &vertices)
                : first
            rings.append((first, second))
        }

        for index in 0..<(profile.count - 1) {
            let lower = rings[index].outgoing
            let upper = rings[index + 1].incoming
            let lowerPole = profile[index].radius == 0
            let upperPole = profile[index + 1].radius == 0
            for sector in 0..<UInt32(segments) {
                let a = lower + sector
                let b = upper + sector
                if !lowerPole { indices += [a, b, a + 1] }
                if !upperPole { indices += [lowerPole ? a : a + 1, b, b + 1] }
            }
        }

        if capBottom, profile[0].radius > 0 {
            appendLatheCap(profile[0], top: false, segments: segments, vertices: &vertices, indices: &indices)
        }
        let top = profile[profile.count - 1]
        if capTop, top.radius > 0 {
            appendLatheCap(top, top: true, segments: segments, vertices: &vertices, indices: &indices)
        }
        return try validatedSolid(vertices: vertices, indices: indices)
    }

    /// A box with the given outer size and a circular bevel radius. Each side
    /// has a top-left UV chart. Matching vertices on chart seams share positions
    /// and normals. Segments subdivide each face's half of an edge fillet.
    ///
    /// Use a bevel smaller than half the smallest dimension. Zero makes a
    /// sharp box. Supply 1...32 segments.
    public static func bevelBox(
        size: SIMD3<Float> = SIMD3(repeating: 1),
        bevel: Float = 0.05,
        segments: Int = 4
    ) throws -> MeshGeometry {
        guard (0..<3).allSatisfy({ size[$0].isNormal && size[$0] > 0 }) else {
            throw SolidValidationError.boxSize
        }
        let half = SIMD3<Double>(size) * 0.5
        let radius = Double(bevel)
        guard radius.isFinite, radius >= 0, radius < min(half.x, min(half.y, half.z)) else {
            throw SolidValidationError.bevel
        }
        guard (1...32).contains(segments) else { throw SolidValidationError.bevelSegments }
        let core = half - radius
        if radius > 0, (0..<3).contains(where: { Float(core[$0]) == Float(half[$0]) }) {
            throw SolidValidationError.collapsedTriangle
        }
        let axes: [(normal: SIMD3<Double>, right: SIMD3<Double>, up: SIMD3<Double>)] = [
            (SIMD3(1, 0, 0), SIMD3(0, 0, -1), SIMD3(0, 1, 0)),
            (SIMD3(-1, 0, 0), SIMD3(0, 0, 1), SIMD3(0, 1, 0)),
            (SIMD3(0, 1, 0), SIMD3(1, 0, 0), SIMD3(0, 0, -1)),
            (SIMD3(0, -1, 0), SIMD3(1, 0, 0), SIMD3(0, 0, 1)),
            (SIMD3(0, 0, 1), SIMD3(1, 0, 0), SIMD3(0, 1, 0)),
            (SIMD3(0, 0, -1), SIMD3(-1, 0, 0), SIMD3(0, 1, 0)),
        ]

        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        let side = radius == 0 ? 2 : 2 * (segments + 1)
        vertices.reserveCapacity(6 * side * side)
        indices.reserveCapacity(36 * (side - 1) * (side - 1))

        for (normal, right, up) in axes {
            let faceExtent = abs(simd_dot(normal, half))
            let rightExtent = abs(simd_dot(right, half))
            let upExtent = abs(simd_dot(up, half))
            let xs = bevelCoordinates(half: rightExtent, bevel: radius, segments: segments)
            let ys = bevelCoordinates(half: upExtent, bevel: radius, segments: segments)
            let start = UInt32(vertices.count)

            for y in ys {
                for x in xs {
                    let plane = normal * faceExtent + right * x + up * y
                    let closest = simd_clamp(plane, -core, core)
                    let roundedNormal = radius == 0 ? normal : simd_normalize(plane - closest)
                    let point = radius == 0 ? plane : closest + roundedNormal * radius
                    let tangent = simd_normalize(right - roundedNormal * simd_dot(right, roundedNormal))
                    vertices.append(MeshVertex(
                        position: SIMD3<Float>(point),
                        normal: SIMD3<Float>(roundedNormal),
                        textureCoordinate: SIMD2(Float(0.5 + x / (2 * rightExtent)), Float(0.5 - y / (2 * upExtent))),
                        tangent: SIMD4(SIMD3<Float>(tangent), -1)
                    ))
                }
            }

            let stride = UInt32(side)
            for row in 0..<(stride - 1) {
                for column in 0..<(stride - 1) {
                    let a = start + row * stride + column
                    let b = a + stride
                    indices += [a, a + 1, b, a + 1, b + 1, b]
                }
            }
        }
        return try validatedSolid(vertices: vertices, indices: indices)
    }

    private static func appendLatheRing(
        _ point: LatheProfilePoint, normal: SIMD2<Double>, v: Float,
        segments: Int, into vertices: inout [MeshVertex]
    ) -> UInt32 {
        let start = UInt32(vertices.count)
        let pole = point.radius == 0
        let count = pole ? segments : segments + 1
        for sector in 0..<count {
            let u = (Double(sector) + (pole ? 0.5 : 0)) / Double(segments)
            let angle = sector == segments ? 0 : 2 * .pi * u
            let radial = SIMD2<Float>(Float(cos(angle)), Float(sin(angle)))
            vertices.append(MeshVertex(
                position: SIMD3(point.radius * radial.x, point.y, point.radius * radial.y),
                normal: SIMD3(Float(normal.x) * radial.x, Float(normal.y), Float(normal.x) * radial.y),
                textureCoordinate: SIMD2(Float(u), v),
                tangent: SIMD4(-radial.y, 0, radial.x, 1)
            ))
        }
        return start
    }

    private static func appendLatheCap(
        _ point: LatheProfilePoint, top: Bool, segments: Int,
        vertices: inout [MeshVertex], indices: inout [UInt32]
    ) {
        let center = UInt32(vertices.count)
        let normal = SIMD3<Float>(0, top ? 1 : -1, 0)
        let tangent = SIMD4<Float>(1, 0, 0, top ? 1 : -1)
        vertices.append(MeshVertex(
            position: SIMD3(0, point.y, 0), normal: normal, textureCoordinate: SIMD2(repeating: 0.5), tangent: tangent
        ))
        for sector in 0..<segments {
            let angle = 2 * .pi * (Double(sector) / Double(segments))
            let x = Float(cos(angle))
            let z = Float(sin(angle))
            vertices.append(MeshVertex(
                position: SIMD3(point.radius * x, point.y, point.radius * z),
                normal: normal, textureCoordinate: SIMD2(0.5 + 0.5 * x, 0.5 - 0.5 * z), tangent: tangent
            ))
        }
        for sector in 0..<segments {
            let a = center + 1 + UInt32(sector)
            let b = center + 1 + UInt32((sector + 1) % segments)
            indices += top ? [center, b, a] : [center, a, b]
        }
    }

    private static func bevelCoordinates(half: Double, bevel: Double, segments: Int) -> [Double] {
        guard bevel > 0 else { return [-half, half] }
        let lower = (0...segments).map { -half + bevel * (Double($0) / Double(segments)) }
        return lower + lower.reversed().map { -$0 }
    }

    private static func validatedSolid(vertices: [MeshVertex], indices: [UInt32]) throws -> MeshGeometry {
        for index in stride(from: 0, to: indices.count, by: 3) {
            let a = SIMD3<Double>(vertices[Int(indices[index])].position)
            let b = SIMD3<Double>(vertices[Int(indices[index + 1])].position)
            let c = SIMD3<Double>(vertices[Int(indices[index + 2])].position)
            guard simd_cross(b - a, c - a) != .zero else { throw SolidValidationError.collapsedTriangle }
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }

    public enum SolidValidationError: LocalizedError, Equatable {
        case profileCount
        case latheSegments
        case profileHeight
        case profilePoint(Int)
        case profileEdge(Int)
        case profileNormal(Int)
        case overlappingCap(Int)
        case boxSize
        case bevel
        case bevelSegments
        case collapsedTriangle

        public var errorDescription: String? {
            switch self {
            case .profileCount: "A lathe needs 2 to 256 profile points."
            case .latheSegments: "A lathe needs 3 to 256 radial segments."
            case .profileHeight: "A lathe profile needs positive height."
            case .profilePoint(let index): "Lathe point \(index) needs a finite height and positive radius, or zero at an endpoint."
            case .profileEdge(let index): "Lathe edge \(index) is repeated, moves downward, or reverses a radial shoulder."
            case .profileNormal(let index): "Lathe point \(index) needs a finite nonzero authored normal."
            case .overlappingCap(let index): "Lathe edge \(index) overlaps a planar cap."
            case .boxSize: "Box dimensions must be positive normal numbers."
            case .bevel: "The bevel must be finite, nonnegative, and smaller than half the shortest side."
            case .bevelSegments: "A box bevel needs 1 to 32 segments."
            case .collapsedTriangle: "The generated triangle collapsed at Float precision."
            }
        }
    }
}
