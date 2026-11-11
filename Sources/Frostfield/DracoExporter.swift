import Foundation
internal import FrostfieldDraco

/// Encodes local mesh geometry and point clouds as standalone Draco `.drc` data.
/// Encoding runs on the caller's thread using the pinned native Draco library.
public enum DracoExporter {
    public enum MeshMethod: Sendable {
        /// Keeps vertex, face and corner order, including isolated vertices and repeated indices.
        case sequential
        /// Compresses connectivity and may reorder points and faces or split
        /// vertices at topology seams. Requires referenced vertices and three
        /// distinct indices in each face.
        case edgeBreaker
    }

    public enum PointCloudMethod: Sendable {
        /// Keeps the input point order and supports unquantized attributes.
        case sequential
        /// Compresses spatially and may reorder points. Requires quantization
        /// for every exported attribute.
        case kdTree
    }

    public enum ColorEncoding: Sendable {
        /// Stores the original linear RGB component values.
        case linear
        /// Converts linear RGB to sRGB for standalone Draco consumers.
        case sRGB
    }

    /// Bit depth for each named attribute. Zero disables quantization for that
    /// attribute. The supported range is 0...30, with 1 excluded for normals.
    public struct Quantization: Equatable, Sendable {
        public var position: Int
        public var normal: Int
        public var textureCoordinate: Int
        public var color: Int

        public init(position: Int = 16, normal: Int = 8, textureCoordinate: Int = 8, color: Int = 8) {
            self.position = position
            self.normal = normal
            self.textureCoordinate = textureCoordinate
            self.color = color
        }

        public static let none = Quantization(position: 0, normal: 0, textureCoordinate: 0, color: 0)
    }

    public struct Options: Sendable {
        /// Each speed is 0 (slowest) through 10 (fastest).
        public var encodeSpeed: Int
        public var decodeSpeed: Int
        public var quantization: Quantization
        /// Nil omits vertex colors. Encoding preserves alpha without applying an RGB transfer to it.
        public var colorEncoding: ColorEncoding?

        public init(
            encodeSpeed: Int = 5, decodeSpeed: Int = 5, quantization: Quantization = .init(),
            colorEncoding: ColorEncoding? = nil
        ) {
            self.encodeSpeed = encodeSpeed
            self.decodeSpeed = decodeSpeed
            self.quantization = quantization
            self.colorEncoding = colorEncoding
        }

        fileprivate func native(method: FFDracoEncodeMethod) throws -> FFDracoEncodeOptions {
            guard (0...10).contains(encodeSpeed), (0...10).contains(decodeSpeed),
                  (0...30).contains(quantization.position), (0...30).contains(quantization.normal),
                  quantization.normal != 1, (0...30).contains(quantization.textureCoordinate),
                  (0...30).contains(quantization.color) else { throw Failure.options }
            return FFDracoEncodeOptions(
                method: method, encode_speed: UInt32(encodeSpeed), decode_speed: UInt32(decodeSpeed),
                position_bits: UInt32(quantization.position), normal_bits: UInt32(quantization.normal),
                uv_bits: UInt32(quantization.textureCoordinate), color_bits: UInt32(quantization.color)
            )
        }
    }

    public static let maximumPointCount = 1_000_000
    public static let maximumIndexCount = 999_999

    /// Encodes indexed triangles with positions, optional normals, a selected
    /// UV set and optional RGBA colors. Mesh colors missing from a partially
    /// colored mesh are white. Quantized normals store direction.
    public static func encode(
        _ mesh: MeshGeometry, method: MeshMethod = .edgeBreaker, options: Options = .init(),
        includeNormals: Bool = true, textureCoordinateSet: MeshTextureCoordinateSet? = .primary
    ) throws -> Data {
        let native = try options.native(method: method == .sequential ? FFDracoMeshSequential : FFDracoMeshEdgeBreaker)
        guard mesh.vertices.count <= maximumPointCount, mesh.indices.count <= maximumIndexCount else {
            throw Failure.sizeLimit
        }
        let colorEncoding = mesh.hasVertexColors ? options.colorEncoding : nil
        let stride = 3 + (includeNormals ? 3 : 0) + (textureCoordinateSet == nil ? 0 : 2) + (colorEncoding == nil ? 0 : 4)
        var values: [Float] = []
        values.reserveCapacity(mesh.vertices.count * stride)
        for (index, vertex) in mesh.vertices.enumerated() {
            append(vertex.position, to: &values)
            if includeNormals { append(vertex.normal, to: &values) }
            switch textureCoordinateSet {
            case .primary:
                values.append(vertex.textureCoordinate.x)
                values.append(vertex.textureCoordinate.y)
            case .secondary:
                guard let uv = vertex.secondaryTextureCoordinate else { throw Failure.missingUV1(vertex: index) }
                values.append(uv.x)
                values.append(uv.y)
            case nil:
                break
            }
            if let colorEncoding {
                append(color: vertex.color ?? SIMD4(repeating: 1), encoding: colorEncoding, to: &values)
            }
        }
        return try encode(
            values: values, pointCount: mesh.vertices.count, indices: mesh.indices,
            normals: includeNormals, uvs: textureCoordinateSet != nil, colors: colorEncoding != nil, options: native
        )
    }

    /// Encodes positions and optional RGBA colors. Per-point sizes are renderer data.
    public static func encode(
        _ pointCloud: PointCloud, method: PointCloudMethod = .kdTree, options: Options = .init()
    ) throws -> Data {
        let native = try options.native(method: method == .sequential ? FFDracoPointsSequential : FFDracoPointsKDTree)
        let count = pointCloud.positions.count
        guard count <= maximumPointCount else { throw Failure.sizeLimit }
        let colorEncoding = pointCloud.colors == nil ? nil : options.colorEncoding
        if method == .kdTree {
            guard native.position_bits > 0, colorEncoding == nil || native.color_bits > 0 else {
                throw Failure.pointCloudQuantization
            }
        }
        var values: [Float] = []
        values.reserveCapacity(count * (colorEncoding == nil ? 3 : 7))
        if let colorEncoding, let colors = pointCloud.colors {
            for (position, color) in zip(pointCloud.positions, colors) {
                append(position, to: &values)
                append(color: color, encoding: colorEncoding, to: &values)
            }
        } else {
            for position in pointCloud.positions { append(position, to: &values) }
        }
        return try encode(
            values: values, pointCount: count, indices: [], normals: false,
            uvs: false, colors: colorEncoding != nil, options: native
        )
    }

    private static func append(_ value: SIMD3<Float>, to values: inout [Float]) {
        values.append(value.x)
        values.append(value.y)
        values.append(value.z)
    }

    private static func append(color: SIMD4<Float>, encoding: ColorEncoding, to values: inout [Float]) {
        switch encoding {
        case .linear: append(SIMD3(color.x, color.y, color.z), to: &values)
        case .sRGB:
            append(
                SIMD3(
                    GLTFColorTables.linearToSRGB(color.x), GLTFColorTables.linearToSRGB(color.y),
                    GLTFColorTables.linearToSRGB(color.z)
                ),
                to: &values
            )
        }
        values.append(color.w)
    }

    private static func encode(
        values: [Float], pointCount: Int, indices: [UInt32], normals: Bool, uvs: Bool, colors: Bool,
        options: FFDracoEncodeOptions
    ) throws -> Data {
        var output: OpaquePointer?
        var options = options
        let status = values.withUnsafeBufferPointer { values in
            indices.withUnsafeBufferPointer { indices in
                var input = FFDracoEncodeInput(
                    values: values.baseAddress, value_count: values.count, point_count: pointCount,
                    indices: indices.baseAddress, index_count: indices.count,
                    has_normals: normals, has_uvs: uvs, has_colors: colors
                )
                return ff_draco_encode(&input, &options, &output)
            }
        }
        switch status {
        case FFDracoStatusOK: break
        case FFDracoStatusSizeLimit: throw Failure.sizeLimit
        case FFDracoStatusUnsupportedGeometry: throw Failure.edgeBreakerTopology
        case FFDracoStatusAttributeRange: throw Failure.attributeRange
        default: throw Failure.encodingFailed
        }
        guard let output else { throw Failure.encodingFailed }
        defer { ff_draco_release_encoded(output) }
        guard let bytes = ff_draco_encoded_bytes(output), ff_draco_encoded_size(output) > 0 else {
            throw Failure.encodingFailed
        }
        return Data(bytes: bytes, count: ff_draco_encoded_size(output))
    }

    public enum Failure: LocalizedError, Equatable {
        case options
        case sizeLimit
        case missingUV1(vertex: Int)
        case pointCloudQuantization
        case edgeBreakerTopology
        case attributeRange
        case encodingFailed

        public var errorDescription: String? {
            switch self {
            case .options: "Use speed settings in 0...10 and quantization in 0...30, with at least two bits when quantizing normals."
            case .sizeLimit: "The Draco geometry exceeds one million points, 999,999 indices or the 64 MiB output limit."
            case .missingUV1(let vertex): "Vertex \(vertex) has no secondary texture coordinate."
            case .pointCloudQuantization: "KD-tree encoding needs quantization for positions and any exported colors."
            case .edgeBreakerTopology: "Use sequential Draco encoding for repeated face indices or unreferenced vertices."
            case .attributeRange: "The exported attribute contains a non-finite value or a range that cannot be quantized."
            case .encodingFailed: "The native Draco encoder could not write this geometry."
            }
        }
    }
}
