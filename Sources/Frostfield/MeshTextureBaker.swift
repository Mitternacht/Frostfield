import Foundation
import Metal
import simd

/// Bakes the vertex attributes of an authored UV atlas into shader-readable textures.
///
/// Atlas coordinates use the top-left origin and the chosen UV set. Referenced UVs must
/// be in 0...1 on both axes. Coverage is sampled at texel centers with a top-left rule,
/// so adjacent triangles share an edge without overlapping at shared samples. Edges
/// use 12 fractional bits per texel. Degenerate UV triangles and triangles that cover
/// no texel centers contribute no samples. Either UV winding is supported.
public struct MeshTextureBaker {
    public enum NormalSpace: Sendable {
        /// Retains object-space directions.
        case object
        /// Transforms directions by the inverse transpose of the affine transform.
        case world(worldFromObject: simd_float4x4)
    }

    /// Determines which triangle supplies attributes when their UV interiors share a texel.
    public enum Overlap: Sendable {
        case reject
        case firstTriangle
        case lastTriangle
    }

    public static let maximumDimension = ShaderTextureGenerator.maximumDimension
    public static let maximumTexelCount = ShaderTextureGenerator.maximumTexelCount
    public static let maximumPadding = 64

    /// The UV set to use when sampling these textures on the input mesh.
    public let textureCoordinateSet: MeshTextureCoordinateSet
    /// Straight linear RGBA in `rgba16Float`. Missing vertex colors sample as opaque white.
    public let vertexColorTexture: any MTLTexture
    /// Object- or world-space normals encoded as `0.5 × normal + 0.5` in `rgba16Float`.
    /// Alpha is one on covered and padded texels. This is a directional data map.
    public let normalTexture: any MTLTexture
    /// `r8Unorm` with stored bytes 255 for interior samples, 128 for padding, and 0 for empty texels.
    public let coverageTexture: any MTLTexture

    /// Rasterizes on the CPU and uploads three complete, shared-storage, single-level textures.
    ///
    /// Corner normals are normalized in the selected space before interpolation and the result
    /// is normalized per sample. If opposing directions cancel, the first corner supplies the
    /// normal. `padding` copies the nearest covered texel up to that many four-neighbor steps,
    /// with ties going to the earliest covered texel in row-major order. Padding cannot overwrite
    /// covered texels. Every unfilled texel stays zero. Overlap policy applies to covered samples
    /// before padding and follows triangle order in `geometry.indices`.
    public init(
        device: any MTLDevice, geometry: MeshGeometry, width: Int, height: Int,
        textureCoordinateSet: MeshTextureCoordinateSet = .primary,
        normalSpace: NormalSpace = .object, overlap: Overlap = .reject, padding: Int = 0
    ) throws {
        var atlas = try UVBakeAtlas(width: width, height: height, padding: padding)
        let coordinates = try atlas.coordinates(of: geometry, set: textureCoordinateSet)
        let normalMatrix = try Self.normalMatrix(normalSpace)
        var vertices: [BakeVertex?] = Array(repeating: nil, count: geometry.vertices.count)
        for index in geometry.indices where vertices[Int(index)] == nil {
            let vertex = geometry.vertices[Int(index)]
            let color = vertex.color ?? SIMD4(repeating: 1)
            let maxHalf = Float(Float16.greatestFiniteMagnitude)
            guard simd_reduce_max(simd_abs(color)) <= maxHalf else { throw Failure.colorRange(vertex: Int(index)) }
            let direction = normalMatrix * SIMD3<Double>(vertex.normal)
            let maximum = simd_reduce_max(simd_abs(direction))
            guard maximum.isFinite, maximum > 0 else { throw Failure.normalTransform }
            vertices[Int(index)] = BakeVertex(color: SIMD4<Double>(color), normal: simd_normalize(direction / maximum))
        }

        var colors = [SIMD4<Float16>](repeating: .zero, count: width * height)
        var normals = [SIMD4<Float16>](repeating: .zero, count: width * height)
        for start in stride(from: 0, to: geometry.indices.count, by: 3) {
            let ia = Int(geometry.indices[start]), ib = Int(geometry.indices[start + 1]), ic = Int(geometry.indices[start + 2])
            // Every referenced vertex was prepared above.
            let a = vertices[ia]!, b = vertices[ib]!, c = vertices[ic]!
            try atlas.rasterize(coordinates[ia], coordinates[ib], coordinates[ic], triangle: start / 3, overlap: overlap) { index, weights in
                let color = a.color * weights.x + b.color * weights.y + c.color * weights.z
                let normal = UVBakeAtlas.interpolatedNormal(a.normal, b.normal, c.normal, weights: weights)
                colors[index] = SIMD4<Float16>(SIMD4<Float>(color))
                normals[index] = SIMD4<Float16>(SIMD4(SIMD3<Float>(normal * 0.5 + 0.5), 1))
            }
        }
        try atlas.pad(by: padding) { source, destination in
            colors[destination] = colors[source]
            normals[destination] = normals[source]
        }

        self.textureCoordinateSet = textureCoordinateSet
        vertexColorTexture = try atlas.upload(colors, device: device, format: .rgba16Float, label: "Frostfield baked vertex color")
        normalTexture = try atlas.upload(normals, device: device, format: .rgba16Float, label: "Frostfield baked normal")
        coverageTexture = try atlas.coverageTexture(device: device)
    }

    private static func normalMatrix(_ space: NormalSpace) throws -> simd_double3x3 {
        guard case .world(let transform) = space else { return matrix_identity_double3x3 }
        let (x, y, z, offset) = transform.columns
        guard [x, y, z, offset].allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }),
              x.w == 0, y.w == 0, z.w == 0, offset.w == 1 else { throw Failure.normalTransform }
        let linear = simd_double3x3(
            SIMD3(Double(x.x), Double(x.y), Double(x.z)),
            SIMD3(Double(y.x), Double(y.y), Double(y.z)),
            SIMD3(Double(z.x), Double(z.y), Double(z.z))
        )
        guard linear.determinant != 0 else { throw Failure.normalTransform }
        return linear.inverse.transpose
    }

    public enum Failure: LocalizedError, Equatable {
        case dimensions
        case padding
        case normalTransform
        case missingUV1(vertex: Int)
        case invalidUV(vertex: Int)
        case colorRange(vertex: Int)
        case overlap(triangle: Int, x: Int, y: Int)
        case emptyAtlas
        case textureAllocation

        public var errorDescription: String? {
            switch self {
            case .dimensions: "A mesh bake needs up to 8,192 texels per side and 16,777,216 texels total."
            case .padding: "Mesh bake padding must be from zero to 64 texels."
            case .normalTransform: "World-space normals need a finite, invertible affine transform."
            case .missingUV1(let vertex): "Mesh bake vertex \(vertex) has no secondary texture coordinates."
            case .invalidUV(let vertex): "Mesh bake vertex \(vertex) has UV coordinates outside the atlas."
            case .colorRange(let vertex): "Mesh bake vertex \(vertex) has a color outside the finite RGBA16Float range."
            case .overlap(let triangle, let x, let y): "Mesh bake triangle \(triangle) overlaps an earlier triangle at texel (\(x), \(y))."
            case .emptyAtlas: "The mesh UV atlas covers no texel centers at this resolution."
            case .textureAllocation: "Metal could not allocate a mesh bake texture."
            }
        }
    }
}

private struct BakeVertex {
    let color: SIMD4<Double>
    let normal: SIMD3<Double>
}

/// Oriented edge equations in fixed-point atlas texel coordinates.
private struct BakeEdge {
    static let unit: Int64 = 1 << 12
    static let half = unit / 2
    let origin: SIMD2<Int64>
    let dx: Int64
    let dy: Int64
    let inclusive: Bool

    init(_ a: SIMD2<Int64>, _ b: SIMD2<Int64>) {
        origin = a
        dx = b.x - a.x
        dy = b.y - a.y
        // Positive oriented triangles in a Y-down atlas include top and left edges.
        inclusive = dy < 0 || (dy == 0 && dx > 0)
    }

    var stepX: Int64 { -dy * Self.unit }
    var stepY: Int64 { dx * Self.unit }

    func value(at p: SIMD2<Int64>) -> Int64 {
        dx * (p.y - origin.y) - dy * (p.x - origin.x)
    }

    func contains(_ value: Int64) -> Bool {
        value > 0 || (value == 0 && inclusive)
    }
}

/// Shared UV coverage, interpolation coordinates and padding for the asset bakers.
struct UVBakeAtlas {
    let width: Int
    let height: Int
    private var coverage: [UInt8]

    init(width: Int, height: Int, padding: Int) throws {
        guard (1...MeshTextureBaker.maximumDimension).contains(width),
              (1...MeshTextureBaker.maximumDimension).contains(height),
              width <= MeshTextureBaker.maximumTexelCount / height else { throw MeshTextureBaker.Failure.dimensions }
        guard (0...MeshTextureBaker.maximumPadding).contains(padding) else { throw MeshTextureBaker.Failure.padding }
        self.width = width
        self.height = height
        coverage = Array(repeating: 0, count: width * height)
    }

    func coordinates(of geometry: MeshGeometry, set: MeshTextureCoordinateSet) throws -> [SIMD2<Int64>] {
        var coordinates = [SIMD2<Int64>](repeating: SIMD2(repeating: -1), count: geometry.vertices.count)
        let scale = SIMD2(Double(width), Double(height)) * Double(BakeEdge.unit)
        for index in geometry.indices where coordinates[Int(index)].x < 0 {
            let vertex = geometry.vertices[Int(index)]
            let uv: SIMD2<Float>
            switch set {
            case .primary:
                uv = vertex.textureCoordinate
            case .secondary:
                guard let secondary = vertex.secondaryTextureCoordinate else {
                    throw MeshTextureBaker.Failure.missingUV1(vertex: Int(index))
                }
                uv = secondary
            }
            guard (0...1).contains(uv.x), (0...1).contains(uv.y) else {
                throw MeshTextureBaker.Failure.invalidUV(vertex: Int(index))
            }
            coordinates[Int(index)] = SIMD2<Int64>((SIMD2<Double>(uv) * scale).rounded(.toNearestOrAwayFromZero))
        }
        return coordinates
    }

    static func interpolatedNormal(
        _ a: SIMD3<Double>, _ b: SIMD3<Double>, _ c: SIMD3<Double>, weights: SIMD3<Double>
    ) -> SIMD3<Double> {
        let direction = a * weights.x + b * weights.y + c * weights.z
        return simd_length_squared(direction) > 1e-24 ? simd_normalize(direction) : a
    }


    mutating func rasterize(
        _ a: SIMD2<Int64>, _ second: SIMD2<Int64>, _ third: SIMD2<Int64>,
        triangle: Int, overlap: MeshTextureBaker.Overlap, sampleOffset: SIMD2<Double> = .zero,
        acceptsSample: ((SIMD3<Double>) -> Bool)? = nil,
        write: (Int, SIMD3<Double>) throws -> Void
    ) throws {
        let signedArea = BakeEdge(a, second).value(at: third)
        guard signedArea != 0 else { return }
        let b = signedArea > 0 ? second : third
        let c = signedArea > 0 ? third : second
        let area = Double(abs(signedArea))
        let edgeA = BakeEdge(b, c)
        let edgeB = BakeEdge(c, a)
        let edgeC = BakeEdge(a, b)
        let minX = min(a.x, b.x, c.x)
        let minY = min(a.y, b.y, c.y)
        let maxX = max(a.x, b.x, c.x)
        let maxY = max(a.y, b.y, c.y)
        let unit = Double(BakeEdge.unit)
        let shift = SIMD2<Int64>((sampleOffset * unit).rounded(.toNearestOrAwayFromZero))
        let center = SIMD2<Int64>(repeating: BakeEdge.half) &+ shift
        let offset = SIMD2<Double>(center) / unit
        let left = max(0, Int(ceil(Double(minX) / unit - offset.x)))
        let top = max(0, Int(ceil(Double(minY) / unit - offset.y)))
        let right = min(width - 1, Int(floor(Double(maxX) / unit - offset.x)))
        let bottom = min(height - 1, Int(floor(Double(maxY) / unit - offset.y)))
        guard left <= right, top <= bottom else { return }
        let first = SIMD2(Int64(left) * BakeEdge.unit, Int64(top) * BakeEdge.unit) &+ center
        var row = SIMD3(edgeA.value(at: first), edgeB.value(at: first), edgeC.value(at: first))
        let across = SIMD3(edgeA.stepX, edgeB.stepX, edgeC.stepX)
        let down = SIMD3(edgeA.stepY, edgeB.stepY, edgeC.stepY)
        let inverseArea = 1 / area
        for y in top...bottom {
            var weights = row
            row &+= down
            for x in left...right {
                let sample = weights
                weights &+= across
                guard edgeA.contains(sample.x), edgeB.contains(sample.y), edgeC.contains(sample.z) else { continue }
                let barycentric = SIMD3(Double(sample.x), Double(sample.y), Double(sample.z)) * inverseArea
                let originalOrder = signedArea > 0 ? barycentric : SIMD3(barycentric.x, barycentric.z, barycentric.y)
                guard acceptsSample?(originalOrder) != false else { continue }
                let index = y * width + x
                if coverage[index] != 0 {
                    switch overlap {
                    case .reject: throw MeshTextureBaker.Failure.overlap(triangle: triangle, x: x, y: y)
                    case .firstTriangle: continue
                    case .lastTriangle: break
                    }
                }
                try write(index, originalOrder)
                markInterior(at: index)
            }
        }
    }

    mutating func markInterior(at index: Int) { coverage[index] = 255 }

    mutating func pad(by radius: Int, copySample: (Int, Int) -> Void) throws {
        guard coverage.contains(255) else { throw MeshTextureBaker.Failure.emptyAtlas }
        guard radius > 0, coverage.contains(0) else { return }
        var frontier: [Int] = []
        for index in coverage.indices where coverage[index] == 255 {
            let x = index % width
            if (x > 0 && coverage[index - 1] == 0) || (x + 1 < width && coverage[index + 1] == 0)
                || (index >= width && coverage[index - width] == 0)
                || (index < coverage.count - width && coverage[index + width] == 0) {
                frontier.append(index)
            }
        }
        var next: [Int] = []
        for _ in 0..<radius {
            if frontier.isEmpty { break }
            next.removeAll(keepingCapacity: true)
            for index in frontier {
                let x = index % width
                if x > 0 { copy(from: index, to: index - 1, into: &next, copySample: copySample) }
                if x + 1 < width { copy(from: index, to: index + 1, into: &next, copySample: copySample) }
                if index >= width { copy(from: index, to: index - width, into: &next, copySample: copySample) }
                if index < coverage.count - width { copy(from: index, to: index + width, into: &next, copySample: copySample) }
            }
            swap(&frontier, &next)
        }
    }

    private mutating func copy(from source: Int, to destination: Int, into next: inout [Int], copySample: (Int, Int) -> Void) {
        guard coverage[destination] == 0 else { return }
        coverage[destination] = 128
        copySample(source, destination)
        next.append(destination)
    }

    func coverageTexture(device: any MTLDevice) throws -> any MTLTexture {
        try upload(coverage, device: device, format: .r8Unorm, label: "Frostfield bake coverage")
    }

    func upload<Value>(
        _ values: [Value], device: any MTLDevice, format: MTLPixelFormat, label: String
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = .shaderRead
        descriptor.hazardTrackingMode = .tracked
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw MeshTextureBaker.Failure.textureAllocation }
        texture.label = label
        values.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: bytes.baseAddress!,
                bytesPerRow: width * MemoryLayout<Value>.stride
            )
        }
        return texture
    }

}
