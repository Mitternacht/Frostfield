import Foundation
import FrostfieldXAtlas
import simd

extension MeshGeometry {
    /// An automatically unwrapped geometry and its single, square UV1 atlas.
    public struct UVAtlas: Sendable {
        public let geometry: MeshGeometry
        public let resolution: Int
        public let padding: Int
        public let chartCount: Int
        /// For each output vertex, the original vertex it copies. Use this to
        /// copy external channels such as skin weights and morph target offsets.
        public let sourceVertexIndices: [Int]
    }

    /// Generates a single square atlas for baking with ``MeshTextureBaker`` or
    /// ``MeshLightmapBaker``. The result replaces UV1 and preserves UV0, all other
    /// vertex attributes, triangle order and winding, and material groups.
    ///
    /// Seams can duplicate vertices. Unreferenced vertices are omitted. Packing
    /// reserves the requested gutter in texels between chart interiors at the
    /// supplied resolution. It reduces chart density if needed to fit one page.
    /// Bake at this resolution using `.secondary` and at most the reserved padding.
    /// Tangents are copied for the existing UV0 material, and do not describe
    /// the new UV1 parameterization.
    ///
    /// This synchronous CPU operation is deterministic for the same ordered
    /// mesh, options and toolchain. It rejects degenerate geometry, invalid
    /// parameterizations, and atlases that do not fit or have empty charts at the
    /// selected resolution. Individual small triangles can be subpixel.
    public func unwrappedUVAtlas(resolution: Int = 512, padding: Int = 4) throws -> UVAtlas {
        guard (16...4096).contains(resolution) else { throw UVAtlasError.resolution }
        guard (0...MeshTextureBaker.maximumPadding).contains(padding),
              padding * 2 + 2 < resolution else { throw UVAtlasError.padding }
        guard vertices.count <= 750_000, indices.count <= 750_000 else { throw UVAtlasError.sizeLimit }

        var input = [Int]()
        var compactIndex = [UInt32](repeating: .max, count: vertices.count)
        var minimum = SIMD3<Double>(repeating: .infinity)
        var maximum = SIMD3<Double>(repeating: -.infinity)
        let compactIndices = indices.map { index in
            let i = Int(index)
            if compactIndex[i] == .max {
                compactIndex[i] = UInt32(input.count)
                input.append(i)
                let p = SIMD3<Double>(vertices[i].position)
                minimum = simd_min(minimum, p)
                maximum = simd_max(maximum, p)
            }
            return compactIndex[i]
        }
        let extent = simd_reduce_max(maximum - minimum)
        guard extent > 0 else { throw UVAtlasError.degenerateTriangle(0) }
        let center = minimum + (maximum - minimum) * 0.5
        // Normalize the input range before passing it to the Float chart solver.
        let positions = input.map { SIMD3<Float>((SIMD3<Double>(vertices[$0].position) - center) * (1024 / extent)) }
        let normals = input.map { SIMD3<Float>(simd_normalize(SIMD3<Double>(vertices[$0].normal))) }
        var totalArea: Double = 0
        for i in stride(from: 0, to: compactIndices.count, by: 3) {
            let a = positions[Int(compactIndices[i])]
            let b = positions[Int(compactIndices[i + 1])]
            let c = positions[Int(compactIndices[i + 2])]
            let area = simd_length(simd_cross(b - a, c - a)) * 0.5
            // xatlas otherwise silently excludes these faces from the atlas.
            guard area > Float.ulpOfOne else { throw UVAtlasError.degenerateTriangle(i / 3) }
            totalArea += Double(area)
        }

        var faceGroups = [UInt32](repeating: 0, count: indices.count / 3)
        for (index, group) in groups.enumerated() {
            for face in (group.start / 3)..<(group.range.upperBound / 3) {
                faceGroups[face] = UInt32(index + 1)
            }
        }
        let texelsPerUnit = Float(Double(resolution) * sqrt(0.75 / totalArea))
        var atlas: OpaquePointer?
        let status = positions.withUnsafeBytes { positionBytes in
            normals.withUnsafeBytes { normalBytes in
                ff_uv_atlas_create(
                    positionBytes.baseAddress, normalBytes.baseAddress, UInt32(positions.count),
                    UInt32(MemoryLayout<SIMD3<Float>>.stride), compactIndices, UInt32(compactIndices.count),
                    faceGroups, UInt32(resolution), UInt32(padding), texelsPerUnit, &atlas
                )
            }
        }
        guard status != FFUVAtlasStatusCannotPack else { throw UVAtlasError.cannotPack }
        guard status == FFUVAtlasStatusOK, let atlas else { throw UVAtlasError.invalidParameterization }
        defer { ff_uv_atlas_destroy(atlas) }

        let chartCount = Int(ff_uv_atlas_chart_count(atlas))
        let vertexCount = Int(ff_uv_atlas_vertex_count(atlas))
        var output = [MeshVertex]()
        var sourceIndices = [Int]()
        var charts = [Int]()
        output.reserveCapacity(vertexCount)
        sourceIndices.reserveCapacity(vertexCount)
        charts.reserveCapacity(vertexCount)
        for index in 0..<vertexCount {
            let vertex = ff_uv_atlas_vertex(atlas, UInt32(index))
            let chart = Int(vertex.chartIndex)
            let uv = SIMD2(vertex.u, vertex.v) / Float(resolution)
            guard input.indices.contains(Int(vertex.sourceIndex)), (0..<chartCount).contains(chart),
                  (0...1).contains(uv.x), (0...1).contains(uv.y) else { throw UVAtlasError.invalidParameterization }
            let sourceIndex = input[Int(vertex.sourceIndex)]
            let source = vertices[sourceIndex]
            output.append(MeshVertex(
                position: source.position, normal: source.normal, textureCoordinate: source.textureCoordinate,
                secondaryTextureCoordinate: uv, tangent: source.tangent, color: source.color
            ))
            sourceIndices.append(sourceIndex)
            charts.append(chart)
        }
        let outputIndices = Array(UnsafeBufferPointer(start: ff_uv_atlas_indices(atlas), count: indices.count))
        let mesh = try MeshGeometry(vertices: output, indices: outputIndices).withGroups(groups)
        try Self.validatePackedUVs(mesh, charts: charts, chartCount: chartCount, resolution: resolution)
        return UVAtlas(
            geometry: mesh, resolution: resolution, padding: padding, chartCount: chartCount,
            sourceVertexIndices: sourceIndices
        )
    }

    private static func validatePackedUVs(
        _ mesh: MeshGeometry, charts: [Int], chartCount: Int, resolution: Int
    ) throws {
        var coverage = try UVBakeAtlas(width: resolution, height: resolution, padding: 0)
        let uv = try coverage.coordinates(of: mesh, set: .secondary)
        var chartCovered = [Bool](repeating: false, count: chartCount)
        do {
            for i in stride(from: 0, to: mesh.indices.count, by: 3) {
                let a = Int(mesh.indices[i]), b = Int(mesh.indices[i + 1]), c = Int(mesh.indices[i + 2])
                let chart = charts[a]
                guard charts[b] == chart, charts[c] == chart else { throw UVAtlasError.invalidParameterization }
                try coverage.rasterize(uv[a], uv[b], uv[c], triangle: i / 3, overlap: .reject) { _, _ in
                    chartCovered[chart] = true
                }
            }
        } catch MeshTextureBaker.Failure.overlap {
            throw UVAtlasError.invalidParameterization
        }
        guard chartCovered.allSatisfy({ $0 }) else { throw UVAtlasError.cannotPack }
    }

    public enum UVAtlasError: LocalizedError, Equatable {
        case resolution
        case padding
        case sizeLimit
        case degenerateTriangle(Int)
        case cannotPack
        case invalidParameterization

        public var errorDescription: String? {
            switch self {
            case .resolution: "A UV atlas needs a square resolution from 16 to 4,096 texels."
            case .padding: "A UV atlas needs from zero to 64 padding texels and room for a chart between the gutters."
            case .sizeLimit: "Automatic UV atlases support up to 750,000 input vertices and 250,000 triangles."
            case .degenerateTriangle(let index): "Triangle \(index) is degenerate at the mesh's working scale."
            case .cannotPack: "The charts could not fit into one page with usable texel coverage."
            case .invalidParameterization: "The mesh could not be unwrapped into a valid, nonoverlapping UV atlas."
            }
        }
    }
}
