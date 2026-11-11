import Foundation
import simd

/// Imports Stanford PLY 1.0 triangle meshes and face-free point clouds.
public enum PLYLoader {
    public struct Options: Sendable {
        /// Maps names stored in the file to the names used by the loader.
        public var propertyNameMapping: [String: String]
        /// Selects extra vertex scalars by their mapped names, in output order.
        /// All selected properties must exist and have a scalar type.
        public var customVertexProperties: [String]

        public init(propertyNameMapping: [String: String] = [:], customVertexProperties: [String] = []) {
            self.propertyNameMapping = propertyNameMapping
            self.customVertexProperties = customVertexProperties
        }
    }

    public enum Geometry: Sendable {
        case mesh(MeshGeometry)
        case pointCloud(PointCloud)
    }

    public struct Asset: Sendable {
        public let geometry: Geometry
        /// Selected scalars in output vertex order, with their original numeric types.
        /// Exportable names can be passed to `PLYExporter.Options.typedScalarProperties`.
        public let customVertexProperties: [PLYExporter.TypedScalarProperty]
        /// Source indices when face UVs or colors cause the mesh to split vertices.
        /// Each output vertex refers to one row of the original vertex element.
        public let sourceVertexIndices: [UInt32]?
        /// Authored normals on a point cloud, in point order. These are source data.
        public let pointNormals: [SIMD3<Float>]?
        /// Authored top-left UVs on a point cloud, in point order. These are source data.
        public let pointTextureCoordinates: [SIMD2<Float>]?
    }

    public enum Failure: LocalizedError, Equatable {
        case fileURL
        case size
        case unsupportedFormat
        case invalidHeader(Int)
        case invalidBody(Int)
        case missingPositions
        case customProperty(String)
        case invalidVertex(Int)
        case invalidFace(Int)

        public var errorDescription: String? {
            switch self {
            case .fileURL: "PLY import requires a regular local file."
            case .size: "The PLY file or geometry exceeds the supported size."
            case .unsupportedFormat: "Use PLY 1.0 ASCII, binary little-endian, or binary big-endian."
            case .invalidHeader(let line): "The PLY header is invalid at line \(line)."
            case .invalidBody(let offset): "The PLY body is invalid at byte \(offset)."
            case .missingPositions: "A PLY vertex needs three scalar position properties."
            case .customProperty(let name): "Select a unique PLY scalar vertex property for '\(name)'."
            case .invalidVertex(let row): "PLY vertex \(row) has attributes outside the supported range."
            case .invalidFace(let row): "PLY face \(row) needs three or four valid vertex indices and matching corner UVs when supplied."
            }
        }
    }

    public static func load(contentsOf url: URL, options: Options = .init()) throws -> Asset {
        guard url.isFileURL else { throw Failure.fileURL }
        let info = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard info.isRegularFile == true else { throw Failure.fileURL }
        guard let size = info.fileSize, size <= PLYFileReader.maximumFileBytes else { throw Failure.size }
        return try load(Data(contentsOf: url, options: .mappedIfSafe), options: options)
    }

    /// Supports ASCII and both binary byte orders, with triangle and quad faces.
    /// Vertex RGB is decoded from sRGB and bottom-left UVs become top-left UVs.
    /// Limits input to 256 MiB and 2²² total records, including unknown elements.
    /// Selected custom scalar storage may occupy up to 128 MiB.
    public static func load(_ data: Data, options: Options = .init()) throws -> Asset {
        guard data.count <= PLYFileReader.maximumFileBytes, options.customVertexProperties.count <= 32 else {
            throw Failure.size
        }
        return try data.withUnsafeBytes { bytes in
            var reader = PLYFileReader(bytes)
            let elements = try reader.header(renaming: options.propertyNameMapping)
            guard let vertex = elements.first(where: { $0.name == "vertex" }) else { throw Failure.missingPositions }
            var vertices = try VertexBuffer(vertex, customProperties: options.customVertexProperties)
            var faces = FaceBuffer()
            for element in elements where element.count > 0 {
                var values = [Double](repeating: 0, count: element.properties.count)
                if element.name == "vertex" {
                    var lists: [Int: [Double]] = [:]
                    for row in 0..<element.count {
                        try reader.record(element, values: &values, lists: &lists)
                        try vertices.append(values, row: row)
                    }
                } else if element.name == "face" {
                    let layout = try FaceLayout(element)
                    var lists = [layout.indices: [Double]()]
                    if let uv = layout.uv { lists[uv] = [] }
                    for row in 0..<element.count {
                        try reader.record(element, values: &values, lists: &lists)
                        try faces.append(layout, lists: lists, values: values, row: row, vertexCount: vertex.count)
                    }
                } else {
                    var lists: [Int: [Double]] = [:]
                    for _ in 0..<element.count { try reader.record(element, values: &values, lists: &lists) }
                }
            }
            try reader.finish()
            return try vertices.asset(faces: faces)
        }
    }

    private struct ColorLayout {
        let channels: SIMD3<Int>
        let scale: Float

        init?(_ element: PLYFileReader.Element) {
            guard let r = element.scalar(["red", "diffuse_red", "r", "diffuse_r"]),
                  let g = element.scalar(["green", "diffuse_green", "g", "diffuse_g"]),
                  let b = element.scalar(["blue", "diffuse_blue", "b", "diffuse_b"]) else { return nil }
            channels = SIMD3(r, g, b)
            scale = switch element.properties[r].scalar {
            case .uint16: 1 / 65_535
            case .float32, .float64: 1
            default: 1 / 255
            }
        }

        func color(_ values: [Double]) -> SIMD4<Float> {
            SIMD4(
                GLTFColorTables.sRGBToLinear(Float(values[channels.x]) * scale),
                GLTFColorTables.sRGBToLinear(Float(values[channels.y]) * scale),
                GLTFColorTables.sRGBToLinear(Float(values[channels.z]) * scale), 1
            )
        }
    }

    private struct CustomColumn {
        let index: Int
        let property: PLYFileReader.Property
        var values: [Double] = []

        func typed(remapping indices: [UInt32]?) -> PLYExporter.TypedScalarProperty {
            func converted<T>(_ transform: (Double) -> T) -> [T] {
                if let indices { return indices.map { transform(values[Int($0)]) } }
                return values.map(transform)
            }
            let result: PLYExporter.TypedScalarProperty.Values = switch property.scalar {
            case .int8: .int8(converted { Int8($0) })
            case .uint8: .uint8(converted { UInt8($0) })
            case .int16: .int16(converted { Int16($0) })
            case .uint16: .uint16(converted { UInt16($0) })
            case .int32: .int32(converted { Int32($0) })
            case .uint32: .uint32(converted { UInt32($0) })
            case .float32: .float32(converted { Float($0) })
            case .float64: .float64(converted { $0 })
            }
            return .init(name: property.name, values: result)
        }
    }

    private struct VertexBuffer {
        let position: SIMD3<Int>
        let normal: SIMD3<Int>?
        let uv: SIMD2<Int>?
        let color: ColorLayout?
        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var uvs: [SIMD2<Float>] = []
        var colors: [SIMD4<Float>] = []
        var columns: [CustomColumn] = []

        init(_ element: PLYFileReader.Element, customProperties: [String]) throws {
            guard let x = element.scalar(["x", "px", "posx"]),
                  let y = element.scalar(["y", "py", "posy"]),
                  let z = element.scalar(["z", "pz", "posz"]) else { throw Failure.missingPositions }
            position = SIMD3(x, y, z)
            if let x = element.scalar(["nx", "normalx"]), let y = element.scalar(["ny", "normaly"]),
               let z = element.scalar(["nz", "normalz"]) {
                normal = SIMD3(x, y, z)
            } else {
                normal = nil
            }
            if let s = element.scalar(["s", "u", "texture_u", "tx"]),
               let t = element.scalar(["t", "v", "texture_v", "ty"]) {
                uv = SIMD2(s, t)
            } else {
                uv = nil
            }
            color = ColorLayout(element)
            guard element.count * customProperties.count * MemoryLayout<Double>.stride <= 128 * 1_024 * 1_024 else {
                throw Failure.size
            }
            var requested: Set<String> = []
            for name in customProperties {
                guard let index = element.scalar([name]), requested.insert(name).inserted else {
                    throw Failure.customProperty(name)
                }
                columns.append(CustomColumn(index: index, property: element.properties[index]))
            }
            positions.reserveCapacity(element.count)
        }

        mutating func append(_ values: [Double], row: Int) throws {
            let p = Self.vector(values, indices: position)
            guard Self.finite(p) else { throw Failure.invalidVertex(row) }
            positions.append(p)
            if let normal {
                let n = Self.vector(values, indices: normal)
                guard Self.finite(n) else { throw Failure.invalidVertex(row) }
                normals.append(n)
            }
            if let uv {
                let coordinates = SIMD2(Float(values[uv.x]), Float(1 - values[uv.y]))
                guard coordinates.x.isFinite, coordinates.y.isFinite else { throw Failure.invalidVertex(row) }
                uvs.append(coordinates)
            }
            if let color {
                let rgba = color.color(values)
                guard rgba.x.isFinite, rgba.y.isFinite, rgba.z.isFinite else { throw Failure.invalidVertex(row) }
                colors.append(rgba)
            }
            for index in columns.indices {
                columns[index].values.append(values[columns[index].index])
            }
        }

        func asset(faces: FaceBuffer) throws -> Asset {
            if faces.indices.isEmpty {
                let points = try PointCloud(positions: positions, colors: color == nil ? nil : colors)
                return Asset(
                    geometry: .pointCloud(points), customVertexProperties: columns.map { $0.typed(remapping: nil) },
                    sourceVertexIndices: nil, pointNormals: normal == nil ? nil : normals,
                    pointTextureCoordinates: uv == nil ? nil : uvs
                )
            }
            let split = faces.hasCornerUVs || !faces.triangleColors.isEmpty
            let remap = split ? faces.indices : nil
            let count = remap?.count ?? positions.count
            let customStride = columns.reduce(0) { $0 + $1.property.scalar.width }
            guard count <= PLYFileReader.maximumRecords, count * customStride <= 128 * 1_024 * 1_024 else {
                throw Failure.size
            }
            let resolvedNormals = meshNormals(faces.indices)
            let meshVertices: [MeshVertex] = (0..<count).map { output in
                let source = remap.map { Int($0[output]) } ?? output
                let faceUV = faces.hasCornerUVs ? faces.cornerUVs[output] : nil
                let faceColor = faces.triangleColors.isEmpty ? nil : faces.triangleColors[output / 3]
                return MeshVertex(
                    position: positions[source], normal: resolvedNormals[source],
                    textureCoordinate: faceUV ?? (uv == nil ? .zero : uvs[source]),
                    color: faceColor ?? (color == nil ? nil : colors[source])
                )
            }
            let indices = split ? (0..<count).map { UInt32($0) } : faces.indices
            let mesh = try MeshGeometry(vertices: meshVertices, indices: indices)
            return Asset(
                geometry: .mesh(mesh), customVertexProperties: columns.map { $0.typed(remapping: remap) },
                sourceVertexIndices: remap, pointNormals: nil, pointTextureCoordinates: nil
            )
        }

        private func meshNormals(_ indices: [UInt32]) -> [SIMD3<Float>] {
            if !normals.isEmpty, normals.allSatisfy({ $0 != .zero }) { return normals }
            var sums = [SIMD3<Double>](repeating: .zero, count: positions.count)
            for i in stride(from: 0, to: indices.count, by: 3) {
                let a = Int(indices[i]), b = Int(indices[i + 1]), c = Int(indices[i + 2])
                let origin = SIMD3<Double>(positions[a])
                let normal = simd_cross(SIMD3<Double>(positions[b]) - origin, SIMD3<Double>(positions[c]) - origin)
                sums[a] += normal
                sums[b] += normal
                sums[c] += normal
            }
            return sums.indices.map { index in
                if !normals.isEmpty, normals[index] != .zero { return normals[index] }
                let normal = sums[index]
                return normal == .zero ? SIMD3(0, 1, 0) : SIMD3<Float>(simd_normalize(normal))
            }
        }

        private static func vector(_ values: [Double], indices: SIMD3<Int>) -> SIMD3<Float> {
            SIMD3(Float(values[indices.x]), Float(values[indices.y]), Float(values[indices.z]))
        }

        private static func finite(_ vector: SIMD3<Float>) -> Bool {
            vector.x.isFinite && vector.y.isFinite && vector.z.isFinite
        }
    }

    private struct FaceLayout {
        let indices: Int
        let uv: Int?
        let color: ColorLayout?

        init(_ element: PLYFileReader.Element) throws {
            guard let index = element.property(["vertex_indices", "vertex_index"]),
                  element.properties[index].countType != nil,
                  element.properties[index].scalar.integerBounds != nil else { throw Failure.invalidFace(0) }
            indices = index
            uv = element.property(["texcoord"])
            if let uv, element.properties[uv].countType == nil { throw Failure.invalidFace(0) }
            color = ColorLayout(element)
        }
    }

    private struct FaceBuffer {
        private static let triangleCorners = [0, 1, 2]
        private static let quadCorners = [0, 1, 2, 0, 2, 3]

        var indices: [UInt32] = []
        var cornerUVs: [SIMD2<Float>?] = []
        var triangleColors: [SIMD4<Float>] = []
        var hasCornerUVs = false

        mutating func append(
            _ layout: FaceLayout, lists: [Int: [Double]], values: [Double], row: Int, vertexCount: Int
        ) throws {
            guard let corners = lists[layout.indices], (3...4).contains(corners.count),
                  corners.allSatisfy({ $0 >= 0 && $0 < Double(vertexCount) }) else { throw Failure.invalidFace(row) }
            let triangleCorners = corners.count == 3 ? Self.triangleCorners : Self.quadCorners
            let limit = layout.uv != nil || layout.color != nil ? PLYFileReader.maximumRecords : PLYFileReader.maximumIndices
            guard triangleCorners.count <= limit - indices.count else { throw Failure.size }
            let cornerUVs = layout.uv.flatMap { lists[$0] } ?? []
            guard cornerUVs.isEmpty || cornerUVs.count == corners.count * 2 else { throw Failure.invalidFace(row) }
            hasCornerUVs = hasCornerUVs || !cornerUVs.isEmpty
            for corner in triangleCorners {
                indices.append(UInt32(corners[corner]))
                if layout.uv != nil {
                    if cornerUVs.isEmpty {
                        self.cornerUVs.append(nil)
                    } else {
                        let u = Float(cornerUVs[2 * corner]), v = Float(1 - cornerUVs[2 * corner + 1])
                        guard u.isFinite, v.isFinite else { throw Failure.invalidFace(row) }
                        self.cornerUVs.append(SIMD2(u, v))
                    }
                }
            }
            if let color = layout.color {
                let rgba = color.color(values)
                guard rgba.x.isFinite, rgba.y.isFinite, rgba.z.isFinite else { throw Failure.invalidFace(row) }
                triangleColors.append(rgba)
                if corners.count == 4 { triangleColors.append(rgba) }
            }
        }
    }
}
