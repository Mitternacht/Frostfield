import Foundation
import simd

/// Imports Wavefront OBJ polygon, line and point geometry. Faces may contain up
/// to 256 corners and must be simple, planar, and strictly convex. Each face,
/// polyline or point record occupies one line. Positions, UVs, and normals must
/// be declared before the records that use them. Optional normalized sRGB vertex
/// colors use the XYZRGB or XYZRGBA extension. Materials and smoothing groups
/// are ignored.
public enum OBJMeshLoader {
    /// Polygons, one polyline per `l` record, and every `p` record in one cloud.
    public struct Document: Sendable {
        public var mesh: MeshGeometry?
        public var lines: [LineGeometry]
        public var points: PointCloud?
    }

    public static func load(_ source: String) throws -> MeshGeometry {
        let document: Document
        do {
            document = try loadDocument(source)
        } catch Failure.noGeometry {
            throw Failure.noFaces
        }
        guard let mesh = document.mesh else { throw Failure.noFaces }
        return mesh
    }

    public static func load(contentsOf url: URL) throws -> MeshGeometry {
        try load(String(contentsOf: url, encoding: .utf8))
    }

    /// Reads polygons, polylines and points. A closing line index makes a loop.
    /// Throws when the file has none of the three.
    public static func loadDocument(_ source: String) throws -> Document {
        var parser = Parser()
        let text = source.first == "\u{FEFF}" ? source.dropFirst() : source[...]
        for (offset, rawLine) in text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline).enumerated() {
            try parser.read(rawLine, line: offset + 1)
        }
        return try parser.document()
    }

    public static func loadDocument(contentsOf url: URL) throws -> Document {
        try loadDocument(String(contentsOf: url, encoding: .utf8))
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidLine(Int, String)
        case noFaces
        case noGeometry

        public var errorDescription: String? {
            switch self {
            case .invalidLine(let line, let reason):
                "OBJ line \(line): \(reason)"
            case .noFaces:
                "The OBJ contains no polygon faces."
            case .noGeometry:
                "The OBJ contains no polygons, lines or points."
            }
        }
    }

    private struct Corner: Hashable {
        let position: Int
        let textureCoordinate: Int?
        let normal: Int?
    }

    private struct VertexKey: Hashable {
        let corner: Corner
        let generatedNormalLine: Int?
    }

    private struct Parser {
        private static let maximumFaceCorners = 256

        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        private var positions: [SIMD3<Float>] = []
        private var colors: [Int: SIMD4<Float>] = [:]
        private var lineRecords: [(indices: [Int], line: Int)] = []
        private var pointRecords: [(index: Int, line: Int)] = []
        private var textureCoordinates: [SIMD2<Float>] = []
        private var normals: [SIMD3<Float>] = []
        private var indexByVertex: [VertexKey: UInt32] = [:]

        mutating func read(_ rawLine: Substring, line: Int) throws {
            let fields = rawLine.prefix { $0 != "#" }.split(
                maxSplits: Self.maximumFaceCorners + 1, whereSeparator: \.isWhitespace
            )
            guard let record = fields.first else { return }
            let values = fields.dropFirst()
            switch record {
            case "v":
                guard (3...4).contains(values.count) || (6...7).contains(values.count) else {
                    throw Failure.invalidLine(line, "A position requires XYZ with an optional weight or RGB(A) color.")
                }
                let xyz = try Self.components(values, line: line)
                if xyz.count >= 6 {
                    guard xyz.dropFirst(3).allSatisfy({ (0...1).contains($0) }) else {
                        throw Failure.invalidLine(line, "Vertex color components must be in 0...1.")
                    }
                    colors[positions.count] = SIMD4(
                        GLTFColorTables.sRGBToLinear(xyz[3]), GLTFColorTables.sRGBToLinear(xyz[4]),
                        GLTFColorTables.sRGBToLinear(xyz[5]), xyz.count == 7 ? xyz[6] : 1
                    )
                }
                // The optional OBJ weight is used by rational free-form geometry.
                positions.append(SIMD3(xyz[0], xyz[1], xyz[2]))
            case "vt":
                guard (1...3).contains(values.count) else {
                    throw Failure.invalidLine(line, "A texture coordinate requires one to three components.")
                }
                let uvw = try Self.components(values, line: line)
                let v = uvw.count > 1 ? uvw[1] : 0
                // The mesh uses a top-left texture origin and two coordinates.
                textureCoordinates.append(SIMD2(uvw[0], 1 - v))
            case "vn":
                guard values.count == 3 else {
                    throw Failure.invalidLine(line, "A normal requires x, y, and z.")
                }
                let xyz = try Self.components(values, line: line)
                let normal = SIMD3(xyz[0], xyz[1], xyz[2])
                let scale = max(abs(normal.x), abs(normal.y), abs(normal.z))
                guard scale > 0 else { throw Failure.invalidLine(line, "A normal must be nonzero.") }
                normals.append(simd_normalize(normal / scale))
            case "f":
                try appendFace(values, line: line)
            case "l":
                guard values.count >= 2 else {
                    throw Failure.invalidLine(line, "A line requires at least two vertices.")
                }
                lineRecords.append((try values.map { try corner($0, line: line).position }, line))
            case "p":
                guard !values.isEmpty else { throw Failure.invalidLine(line, "A point requires a vertex index.") }
                pointRecords.append(contentsOf: try values.map { (try corner($0, line: line).position, line) })
            default:
                break
            }
        }

        func document() throws -> OBJMeshLoader.Document {
            let mesh = indices.isEmpty ? nil : try MeshGeometry(vertices: vertices, indices: indices)
            let lines = try lineRecords.map { record in
                let loop = record.indices.count >= 4 && record.indices.first == record.indices.last
                let indices = loop ? Array(record.indices.dropLast()) : record.indices
                return try LineGeometry(
                    positions: indices.map { positions[$0] },
                    colors: try attributeColors(indices, line: record.line),
                    topology: loop ? .loop : .strip
                )
            }
            let points = pointRecords.isEmpty ? nil : try PointCloud(
                positions: pointRecords.map { positions[$0.index] },
                colors: try attributeColors(pointRecords.map(\.index), line: pointRecords[0].line)
            )
            guard mesh != nil || !lines.isEmpty || points != nil else { throw Failure.noGeometry }
            return OBJMeshLoader.Document(mesh: mesh, lines: lines, points: points)
        }

        private func attributeColors(_ indices: [Int], line: Int) throws -> [SIMD4<Float>]? {
            let values = indices.map { colors[$0] }
            if values.allSatisfy({ $0 == nil }) { return nil }
            guard values.allSatisfy({ $0 != nil }) else {
                throw Failure.invalidLine(line, "Color every vertex of a line or point set, or none of them.")
            }
            return values.map { $0! }
        }

        private mutating func appendFace(_ fields: ArraySlice<Substring>, line: Int) throws {
            guard (3...Self.maximumFaceCorners).contains(fields.count) else {
                throw Failure.invalidLine(line, "A face requires 3...\(Self.maximumFaceCorners) corners.")
            }
            let corners = try fields.map { try corner($0, line: line) }
            let first = corners[0]
            guard corners.allSatisfy({
                ($0.textureCoordinate == nil) == (first.textureCoordinate == nil) &&
                    ($0.normal == nil) == (first.normal == nil)
            }) else {
                throw Failure.invalidLine(line, "Face corners must use the same UV and normal fields.")
            }
            let faceNormal = try self.faceNormal(corners, line: line)
            var faceIndices: [UInt32] = []
            faceIndices.reserveCapacity(corners.count)
            for corner in corners {
                let key = VertexKey(corner: corner, generatedNormalLine: corner.normal == nil ? line : nil)
                if let existing = indexByVertex[key] {
                    faceIndices.append(existing)
                } else {
                    guard let index = UInt32(exactly: vertices.count) else {
                        throw Failure.invalidLine(line, "The mesh exceeds the UInt32 vertex index range.")
                    }
                    let uv = corner.textureCoordinate.map { textureCoordinates[$0] } ?? .zero
                    let normal = corner.normal.map { normals[$0] } ?? faceNormal
                    vertices.append(MeshVertex(
                        position: positions[corner.position], normal: normal, textureCoordinate: uv,
                        color: colors[corner.position]
                    ))
                    indexByVertex[key] = index
                    faceIndices.append(index)
                }
            }

            for corner in 1..<(faceIndices.count - 1) {
                indices.append(faceIndices[0])
                indices.append(faceIndices[corner])
                indices.append(faceIndices[corner + 1])
            }
        }

        private func corner(_ token: Substring, line: Int) throws -> Corner {
            let fields = token.split(separator: "/", maxSplits: 3, omittingEmptySubsequences: false)
            guard (1...3).contains(fields.count), let first = fields.first, !first.isEmpty,
                  fields.count == 1 || !fields[fields.count - 1].isEmpty else {
                throw Failure.invalidLine(line, "Invalid face corner '\(token)'.")
            }
            return Corner(
                position: try Self.index(first, count: positions.count, kind: "position", line: line),
                textureCoordinate: fields.count > 1 && !fields[1].isEmpty
                    ? try Self.index(fields[1], count: textureCoordinates.count, kind: "UV", line: line)
                    : nil,
                normal: fields.count == 3
                    ? try Self.index(fields[2], count: normals.count, kind: "normal", line: line)
                    : nil
            )
        }

        private static func index(_ token: Substring, count: Int, kind: String, line: Int) throws -> Int {
            guard let value = Int(token), value != 0 else {
                throw Failure.invalidLine(line, "Invalid \(kind) index '\(token)'.")
            }
            let index = value > 0 ? value - 1 : count + value
            guard (0..<count).contains(index) else {
                throw Failure.invalidLine(line, "\(kind.capitalized) index \(value) is out of range.")
            }
            return index
        }

        private static func components(_ values: ArraySlice<Substring>, line: Int) throws -> [Float] {
            try values.map { token in
                guard let number = Float(token), number.isFinite else {
                    throw Failure.invalidLine(line, "Expected a finite number, found '\(token)'.")
                }
                return number
            }
        }

        private func faceNormal(_ corners: [Corner], line: Int) throws -> SIMD3<Float> {
            func doublePosition(_ index: Int) -> SIMD3<Double> {
                let p = positions[index]
                return SIMD3(Double(p.x), Double(p.y), Double(p.z))
            }
            let origin = doublePosition(corners[0].position)
            let offsets = corners.map { doublePosition($0.position) - origin }
            let extent = offsets.reduce(0.0) { max($0, abs($1.x), abs($1.y), abs($1.z)) }
            guard extent > 0 else { throw Failure.invalidLine(line, "A face must enclose an area.") }

            let points = offsets.map { $0 / extent }
            var area = SIMD3<Double>.zero
            for index in points.indices {
                area += simd_cross(points[index], points[(index + 1) % points.count])
            }
            let length = simd_length(area)
            guard length > 0 else { throw Failure.invalidLine(line, "A face must enclose an area.") }
            let normal = area / length
            guard points.allSatisfy({ abs(simd_dot($0, normal)) <= 0.00001 }) else {
                throw Failure.invalidLine(line, "A face must be planar.")
            }

            // Every other corner must lie strictly inside each oriented edge.
            for index in points.indices {
                let next = (index + 1) % points.count
                let edge = points[next] - points[index]
                for other in points.indices where other != index && other != next {
                    guard simd_dot(simd_cross(edge, points[other] - points[index]), normal) > 0 else {
                        throw Failure.invalidLine(line, "A face must be simple and strictly convex.")
                    }
                }
            }
            return SIMD3(Float(normal.x), Float(normal.y), Float(normal.z))
        }
    }
}
