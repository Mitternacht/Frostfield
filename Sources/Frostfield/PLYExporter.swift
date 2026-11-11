import Foundation
import simd

/// Writes static triangles and point clouds in Stanford PLY 1.0 format.
public enum PLYExporter {
    public enum Format: String, CaseIterable, Sendable {
        case ascii
        case binaryLittleEndian = "binary_little_endian"
        case binaryBigEndian = "binary_big_endian"
    }

    /// An additional Float32 column on the vertex element.
    public struct ScalarProperty: Sendable {
        /// A unique ASCII identifier starting with a letter or underscore.
        public var name: String
        /// One finite value per exported vertex, in output order.
        public var values: [Float]

        public init(name: String, values: [Float]) {
            self.name = name
            self.values = values
        }
    }

    /// An additional vertex column with a fixed-width PLY scalar type.
    public struct TypedScalarProperty: Sendable {
        public enum Values: Sendable {
            case int8([Int8])
            case uint8([UInt8])
            case int16([Int16])
            case uint16([UInt16])
            case int32([Int32])
            case uint32([UInt32])
            case float32([Float])
            case float64([Double])

            fileprivate var layout: (type: String, width: Int) {
                switch self {
                case .int8: ("char", 1)
                case .uint8: ("uchar", 1)
                case .int16: ("short", 2)
                case .uint16: ("ushort", 2)
                case .int32: ("int", 4)
                case .uint32: ("uint", 4)
                case .float32: ("float", 4)
                case .float64: ("double", 8)
                }
            }

            fileprivate var count: Int {
                switch self {
                case .int8(let values): values.count
                case .uint8(let values): values.count
                case .int16(let values): values.count
                case .uint16(let values): values.count
                case .int32(let values): values.count
                case .uint32(let values): values.count
                case .float32(let values): values.count
                case .float64(let values): values.count
                }
            }

            fileprivate var firstNonfinite: Int? {
                switch self {
                case .float32(let values): values.firstIndex { !$0.isFinite }
                case .float64(let values): values.firstIndex { !$0.isFinite }
                default: nil
                }
            }
        }

        /// A unique ASCII identifier starting with a letter or underscore.
        public var name: String
        /// One value per exported vertex, in output order. Floats must be finite.
        public var values: Values

        public init(name: String, values: Values) {
            self.name = name
            self.values = values
        }
    }

    public struct Options: Sendable {
        public var format: Format
        public var includeNormals: Bool
        /// Writes primary UVs as bottom-left `s` and `t` properties.
        public var includeTextureCoordinates: Bool
        /// Writes present vertex colors as sRGB RGB bytes. Missing colors use white.
        public var includeColors: Bool
        /// Set to false to export mesh vertices as a point cloud.
        public var includeFaces: Bool
        /// Float32 columns written in the supplied order.
        public var scalarProperties: [ScalarProperty]
        /// Fixed-width columns, written in the supplied order after `scalarProperties`.
        public var typedScalarProperties: [TypedScalarProperty]

        public init(
            format: Format = .ascii, includeNormals: Bool = true, includeTextureCoordinates: Bool = true,
            includeColors: Bool = true, includeFaces: Bool = true, scalarProperties: [ScalarProperty] = []
        ) {
            self.init(
                format: format, includeNormals: includeNormals, includeTextureCoordinates: includeTextureCoordinates,
                includeColors: includeColors, includeFaces: includeFaces, scalarProperties: scalarProperties,
                typedScalarProperties: []
            )
        }

        public init(
            format: Format = .ascii, includeNormals: Bool = true, includeTextureCoordinates: Bool = true,
            includeColors: Bool = true, includeFaces: Bool = true, scalarProperties: [ScalarProperty] = [],
            typedScalarProperties: [TypedScalarProperty]
        ) {
            self.format = format
            self.includeNormals = includeNormals
            self.includeTextureCoordinates = includeTextureCoordinates
            self.includeColors = includeColors
            self.includeFaces = includeFaces
            self.scalarProperties = scalarProperties
            self.typedScalarProperties = typedScalarProperties
        }
    }

    /// Writes all vertices and, when requested, triangle faces from the mesh.
    public static func data(from mesh: MeshGeometry, options: Options = .init()) throws -> Data {
        try encode([.mesh(mesh)], options: options)
    }

    /// Writes positions and optional colors with no face element. To preserve
    /// per-point sizes, supply them as a named scalar property.
    public static func data(from cloud: PointCloud, options: Options = .init()) throws -> Data {
        try encode([.points(cloud, worldFromLocal: nil)], options: options)
    }

    /// Flattens meshes and point clouds into world space in resolved node order.
    /// Each node writes its mesh instances in index order, then its point cloud.
    /// In mixed scenes, mesh faces stay indexed and points are unreferenced
    /// vertices. Missing normals and UVs on those points use zero.
    public static func data(from graph: SceneGraph, options: Options = .init()) throws -> Data {
        var segments: [Segment] = []
        for node in try graph.resolve() {
            for part in try MeshFileSnapshot.meshes(for: node) { segments.append(.mesh(part.geometry)) }
            if case .points(let renderer)? = node.primitive {
                segments.append(.points(renderer.cloud, worldFromLocal: node.worldTransform))
            }
        }
        return try encode(segments, options: options)
    }

    public enum Failure: LocalizedError, Equatable {
        case emptyGeometry
        case size
        case propertyName(String)
        case propertyCount(String)
        case propertyValue(String, vertex: Int)

        public var errorDescription: String? {
            switch self {
            case .emptyGeometry: "Supply a mesh or point cloud for PLY export."
            case .size: "The PLY vertex or face data exceeds the supported size."
            case .propertyName(let name): "Use a unique ASCII PLY property name for '\(name)'."
            case .propertyCount(let name): "Supply one '\(name)' value per PLY vertex."
            case .propertyValue(let name, let vertex): "The PLY property '\(name)' is nonfinite at vertex \(vertex)."
            }
        }
    }

    private enum Segment {
        case mesh(MeshGeometry)
        case points(PointCloud, worldFromLocal: simd_float4x4?)

        var vertexCount: Int {
            switch self {
            case .mesh(let mesh): mesh.vertices.count
            case .points(let cloud, _): cloud.positions.count
            }
        }

        var hasColors: Bool {
            switch self {
            case .mesh(let mesh): mesh.hasVertexColors
            case .points(let cloud, _): cloud.colors != nil
            }
        }
    }

    private static func encode(_ segments: [Segment], options: Options) throws -> Data {
        guard !segments.isEmpty else { throw Failure.emptyGeometry }
        var vertexCount = 0
        var faceCount = 0
        var hasMeshes = false
        for segment in segments {
            guard segment.vertexCount <= Int(Int32.max) - vertexCount else { throw Failure.size }
            vertexCount += segment.vertexCount
            if case .mesh(let mesh) = segment {
                hasMeshes = true
                if options.includeFaces {
                    let faces = mesh.indices.count / 3
                    guard faces <= Int(Int32.max) - faceCount else { throw Failure.size }
                    faceCount += faces
                }
            }
        }
        let properties = options.scalarProperties.map {
            TypedScalarProperty(name: $0.name, values: .float32($0.values))
        } + options.typedScalarProperties
        try validate(properties, vertexCount: vertexCount)
        let normals = hasMeshes && options.includeNormals
        let uvs = hasMeshes && options.includeTextureCoordinates
        let colors = options.includeColors && segments.contains { $0.hasColors }

        var header = "ply\nformat \(options.format.rawValue) 1.0\nelement vertex \(vertexCount)\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        if normals { header += "property float nx\nproperty float ny\nproperty float nz\n" }
        if uvs { header += "property float s\nproperty float t\n" }
        if colors { header += "property uchar red\nproperty uchar green\nproperty uchar blue\n" }
        for property in properties { header += "property \(property.values.layout.type) \(property.name)\n" }
        if faceCount > 0 { header += "element face \(faceCount)\nproperty list uchar int vertex_index\n" }
        header += "end_header\n"

        var writer = Writer(format: options.format, data: Data(header.utf8))
        if options.format != .ascii {
            let standardWidth = 4 * (3 + (normals ? 3 : 0) + (uvs ? 2 : 0)) + (colors ? 3 : 0)
            let stride = properties.reduce(standardWidth) { $0 + $1.values.layout.width }
            let fixedBytes = writer.data.count + faceCount * 13
            guard vertexCount <= (Int.max - fixedBytes) / stride else { throw Failure.size }
            writer.data.reserveCapacity(fixedBytes + vertexCount * stride)
        }

        var row = 0
        for segment in segments {
            switch segment {
            case .mesh(let mesh):
                for vertex in mesh.vertices {
                    let uv = vertex.textureCoordinate
                    writer.vertex(
                        position: vertex.position, normal: normals ? vertex.normal : nil,
                        uv: uvs ? SIMD2(uv.x, 1 - uv.y) : nil,
                        color: colors ? vertex.color ?? SIMD4(repeating: 1) : nil,
                        properties: properties, row: row
                    )
                    row += 1
                }
            case .points(let cloud, let worldFromLocal):
                for (index, local) in cloud.positions.enumerated() {
                    let position = try worldFromLocal.map { try MeshFileSnapshot.position(local, in: $0) } ?? local
                    writer.vertex(
                        position: position, normal: normals ? .zero : nil, uv: uvs ? .zero : nil,
                        color: colors ? cloud.colors?[index] ?? SIMD4(repeating: 1) : nil,
                        properties: properties, row: row
                    )
                    row += 1
                }
            }
        }
        if faceCount > 0 {
            var offset = 0
            for segment in segments {
                if case .mesh(let mesh) = segment {
                    for index in stride(from: 0, to: mesh.indices.count, by: 3) {
                        writer.appendInteger(UInt8(3))
                        writer.appendInteger(UInt32(offset) + mesh.indices[index])
                        writer.appendInteger(UInt32(offset) + mesh.indices[index + 1])
                        writer.appendInteger(UInt32(offset) + mesh.indices[index + 2])
                        writer.endRow()
                    }
                }
                offset += segment.vertexCount
            }
        }
        return writer.data
    }

    private static func validate(_ properties: [TypedScalarProperty], vertexCount: Int) throws {
        var used: Set<String> = [
            "x", "y", "z", "nx", "ny", "nz", "s", "t", "u", "v", "texture_u", "texture_v", "red", "green", "blue",
            "alpha", "r", "g", "b", "a", "diffuse_red", "diffuse_green", "diffuse_blue", "vertex_index", "vertex_indices",
            "end_header",
        ]
        for property in properties {
            let name = property.name
            guard let first = name.utf8.first, isIdentifierStart(first),
                  name.utf8.dropFirst().allSatisfy({ isIdentifierStart($0) || (48...57).contains($0) }),
                  used.insert(name.lowercased()).inserted else { throw Failure.propertyName(name) }
            guard property.values.count == vertexCount else { throw Failure.propertyCount(name) }
            if let index = property.values.firstNonfinite {
                throw Failure.propertyValue(name, vertex: index)
            }
        }
    }

    private static func isIdentifierStart(_ byte: UInt8) -> Bool {
        (65...90).contains(byte) || (97...122).contains(byte) || byte == 95
    }

    private struct Writer {
        let format: Format
        var data: Data
        private var startsRow = true

        init(format: Format, data: Data) {
            self.format = format
            self.data = data
        }

        mutating func vertex(
            position: SIMD3<Float>, normal: SIMD3<Float>?, uv: SIMD2<Float>?, color: SIMD4<Float>?,
            properties: [TypedScalarProperty], row: Int
        ) {
            appendFloat(position.x)
            appendFloat(position.y)
            appendFloat(position.z)
            if let normal {
                appendFloat(normal.x)
                appendFloat(normal.y)
                appendFloat(normal.z)
            }
            if let uv {
                appendFloat(uv.x)
                appendFloat(uv.y)
            }
            if let color {
                let srgb = MeshFileSnapshot.color(color)
                appendInteger(UInt8((srgb.x * 255).rounded()))
                appendInteger(UInt8((srgb.y * 255).rounded()))
                appendInteger(UInt8((srgb.z * 255).rounded()))
            }
            for property in properties { appendProperty(property.values, row: row) }
            endRow()
        }

        mutating func appendFloat(_ value: Float) {
            if format == .ascii { appendField(String(value)) } else { appendBinary(value.bitPattern) }
        }

        private mutating func appendProperty(_ values: TypedScalarProperty.Values, row: Int) {
            switch values {
            case .int8(let values): appendInteger(values[row])
            case .uint8(let values): appendInteger(values[row])
            case .int16(let values): appendInteger(values[row])
            case .uint16(let values): appendInteger(values[row])
            case .int32(let values): appendInteger(values[row])
            case .uint32(let values): appendInteger(values[row])
            case .float32(let values): appendFloat(values[row])
            case .float64(let values):
                let value = values[row]
                if format == .ascii { appendField(String(value)) } else { appendBinary(value.bitPattern) }
            }
        }

        mutating func appendInteger<T: FixedWidthInteger>(_ value: T) {
            if format == .ascii { appendField(String(value)) } else { appendBinary(value) }
        }

        private mutating func appendField(_ value: String) {
            if !startsRow { data.append(0x20) }
            data.append(contentsOf: value.utf8)
            startsRow = false
        }

        private mutating func appendBinary<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: format == .binaryLittleEndian ? value.littleEndian : value.bigEndian) {
                data.append(contentsOf: $0)
            }
        }

        mutating func endRow() {
            if format == .ascii {
                data.append(0x0A)
                startsRow = true
            }
        }
    }
}
