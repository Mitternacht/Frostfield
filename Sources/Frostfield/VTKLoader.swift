import Foundation
import simd

/// Imports indexed surfaces from VTK POLYDATA datasets.
public enum VTKLoader {
    public enum Failure: LocalizedError, Equatable {
        case fileURL
        case size
        case header
        case unsupported(String)
        case malformed(String)
        case truncated
        case empty

        public var errorDescription: String? {
            switch self {
            case .fileURL: "VTK import requires a regular local file."
            case .size: "The VTK file or surface exceeds the supported size."
            case .header: "The file has an invalid VTK header."
            case .unsupported(let feature): "The VTK feature \(feature) is unsupported."
            case .malformed(let section): "The VTK section \(section) is malformed."
            case .truncated: "The VTK file ends before a declared array is complete."
            case .empty: "The VTK dataset contains no polygon or strip triangles."
            }
        }
    }

    static let maximumBytes = 128 * 1_024 * 1_024
    static let maximumVertices = 2_097_152
    static let maximumCells = 2_097_152
    static let maximumConnectivity = 8_388_608
    static let maximumIndices = 6_291_456

    public static func load(contentsOf url: URL) throws -> MeshGeometry {
        guard url.isFileURL else { throw Failure.fileURL }
        let info = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard info.isRegularFile == true else { throw Failure.fileURL }
        if let size = info.fileSize, size > maximumBytes { throw Failure.size }
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var data = Data()
        while data.count <= maximumBytes {
            let length = min(1 << 20, maximumBytes + 1 - data.count)
            guard let chunk = try file.read(upToCount: length), !chunk.isEmpty else { break }
            data.append(chunk)
        }
        return try load(data)
    }

    /// Reads a legacy or XML PolyData surface.
    /// Point and cell colors are converted from sRGB to linear vertex colors.
    public static func load(_ data: Data) throws -> MeshGeometry {
        guard data.count <= maximumBytes else { throw Failure.size }
        if VTKXMLReader.recognizes(data) { return try VTKXMLReader(data: data).read().makeMesh() }
        let parsed = try data.withUnsafeBytes { bytes in
            var reader = VTKLegacyReader(bytes: bytes)
            return try reader.read()
        }
        return try parsed.makeMesh()
    }

    enum CellKind: Int, CaseIterable {
        case vertices, lines, polygons, strips
    }

    struct Cells {
        let kind: CellKind
        var indices: [UInt32] = []
        var offsets: [Int] = []
    }

    struct Attributes {
        var normals: [SIMD3<Float>]?
        var colors: [SIMD4<Float>]?
    }

    struct PolyData {
        var points: [SIMD3<Float>] = []
        var cells: [Cells] = []
        var pointData = Attributes()
        var cellData = Attributes()
        var cellCount: Int { cells.reduce(0) { $0 + $1.offsets.count } }

        func makeMesh() throws -> MeshGeometry {
            var bases: [CellKind: Int] = [:]
            var next = 0
            for kind in CellKind.allCases {
                bases[kind] = next
                next += cells.first { $0.kind == kind }?.offsets.count ?? 0
            }

            let expandsCells = cellData.colors != nil || cellData.normals != nil
            var indices: [UInt32] = []
            var owners: [Int] = []
            for group in cells where group.kind == .polygons || group.kind == .strips {
                var start = 0
                for (cell, end) in group.offsets.enumerated() {
                    let count = end - start
                    guard count >= 3 else { throw Failure.malformed("surface cell") }
                    let added = (count - 2) * 3
                    let limit = expandsCells ? maximumVertices : maximumIndices
                    guard added <= limit - indices.count else { throw Failure.size }
                    for i in 0..<(count - 2) {
                        if group.kind == .polygons {
                            indices.append(group.indices[start])
                            indices.append(group.indices[start + i + 1])
                            indices.append(group.indices[start + i + 2])
                        } else {
                            indices.append(group.indices[start + i])
                            indices.append(group.indices[start + i + (i.isMultiple(of: 2) ? 1 : 2)])
                            indices.append(group.indices[start + i + (i.isMultiple(of: 2) ? 2 : 1)])
                        }
                        if expandsCells { owners.append(bases[group.kind, default: 0] + cell) }
                    }
                    start = end
                }
            }
            guard !indices.isEmpty else { throw Failure.empty }

            let normalFallback = SIMD3<Float>(0, 0, 1)
            let authored = pointData.normals
            let needsNormals = authored == nil || authored?.contains(.zero) == true
            let vertices = points.indices.map { index in
                let normal = authored?[index] ?? .zero
                return MeshVertex(
                    position: points[index], normal: normal == .zero ? normalFallback : normal,
                    color: pointData.colors?[index]
                )
            }
            var mesh = try MeshGeometry(vertices: vertices, indices: indices)
            if needsNormals {
                let smooth = mesh.generatingVertexNormals()
                if let authored {
                    let filled = vertices.indices.map { authored[$0] == .zero ? smooth.vertices[$0] : vertices[$0] }
                    mesh = try MeshGeometry(vertices: filled, indices: indices)
                } else {
                    mesh = smooth
                }
            }
            guard expandsCells else { return mesh }
            let expanded = indices.enumerated().map { offset, source in
                let point = mesh.vertices[Int(source)]
                let owner = owners[offset / 3]
                let cellNormal = cellData.normals?[owner] ?? .zero
                return MeshVertex(
                    position: point.position, normal: cellNormal == .zero ? point.normal : cellNormal,
                    color: cellData.colors?[owner] ?? point.color
                )
            }
            return try MeshGeometry(vertices: expanded, indices: expanded.indices.map(UInt32.init))
        }
    }

    static func linearColor(_ values: SIMD4<Float>) -> SIMD4<Float> {
        SIMD4(
            GLTFColorTables.sRGBToLinear(values.x), GLTFColorTables.sRGBToLinear(values.y),
            GLTFColorTables.sRGBToLinear(values.z), values.w
        )
    }
}
