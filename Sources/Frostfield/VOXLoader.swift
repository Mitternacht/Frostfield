import Foundation
import Metal
import simd

/// Reads MagicaVoxel models and their static scene hierarchy.
public enum VOXLoader {
    public struct Voxel: Equatable, Sendable {
        /// Authored X-right, Y-forward, Z-up grid coordinates.
        public let position: SIMD3<UInt8>
        public let paletteIndex: UInt8
    }

    public struct Model: Sendable {
        public let size: SIMD3<Int>
        public let voxels: [Voxel]
    }

    public struct Node: Sendable {
        public let name: String?
        /// A local transform in Frostfield's X-right, Y-up, Z-back coordinates.
        public let transform: SceneTransform
        /// An index into the asset's models on mesh-bearing leaves.
        public let modelIndex: Int?
        public let children: [Node]
    }

    public struct Asset: Sendable {
        public let version: Int
        public let models: [Model]
        /// All 256 sRGB RGBA entries, including the reserved zero entry.
        public let palette: [SIMD4<UInt8>]
        /// First-frame scene hierarchy, when the file includes scene chunks.
        public let scene: Node?

        /// Builds a greedy, colored surface for one model. Empty models have no mesh.
        public func makeMesh(modelIndex: Int) throws -> MeshGeometry? {
            guard models.indices.contains(modelIndex) else { throw Failure.invalidModelIndex(modelIndex) }
            let model = models[modelIndex]
            guard !model.voxels.isEmpty else { return nil }
            return try model.makeMesh(palette: palette, maximumVertices: VOXLoader.maximumVertices)
        }

        /// Prepares the imported hierarchy, or an untransformed group of all models in a legacy file.
        /// One renderer is shared among every reference to the same model.
        public func makeSceneNode(
            device: any MTLDevice, colorPixelFormat: MTLPixelFormat, material: MeshMaterial? = nil,
            depthMode: MeshDepthMode = .forward, sampleCount: Int = 1
        ) throws -> SceneNode {
            let material = try material ?? .physicallyBased(PhysicallyBasedMaterial(roughness: 1, alphaMode: .opaque))
            var renderers: [Int: MeshRenderer] = [:]
            var remainingVertices = VOXLoader.maximumSceneVertices
            var remainingCells = VOXLoader.maximumSceneCells

            func prepare(_ node: Node) throws -> SceneNode {
                var renderer: MeshRenderer?
                if let index = node.modelIndex, !models[index].voxels.isEmpty {
                    if let prepared = renderers[index] {
                        renderer = prepared
                    } else {
                        let size = models[index].size
                        let cells = size.x * size.y * size.z
                        guard cells <= remainingCells else { throw Failure.meshLimit }
                        remainingCells -= cells
                        let geometry = try models[index].makeMesh(
                            palette: palette, maximumVertices: min(remainingVertices, VOXLoader.maximumVertices)
                        )
                        remainingVertices -= geometry.vertices.count
                        let prepared = try MeshRenderer(
                            device: device, colorPixelFormat: colorPixelFormat, geometry: geometry,
                            material: material, depthMode: depthMode, sampleCount: sampleCount
                        )
                        renderers[index] = prepared
                        renderer = prepared
                    }
                }
                return try SceneNode(
                    transform: node.transform, mesh: renderer, children: node.children.map(prepare), name: node.name
                )
            }
            if let scene { return try prepare(scene) }
            let nodes = models.indices.map { Node(name: nil, transform: SceneTransform(), modelIndex: $0, children: []) }
            return try SceneNode(children: nodes.map(prepare))
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case fileURL
        case size
        case header
        case version(Int)
        case chunk(String)
        case invalidVoxel(Int)
        case invalidModelIndex(Int)
        case scene
        case meshLimit
        case textureAllocation

        public var errorDescription: String? {
            switch self {
            case .fileURL: "VOX import requires a regular local file."
            case .size: "The VOX file or voxel count exceeds the supported size."
            case .header: "The file has an invalid MagicaVoxel header."
            case .version(let version): "VOX version \(version) is unsupported."
            case .chunk(let id): "The VOX file has an invalid or truncated \(id) chunk."
            case .invalidVoxel(let index): "VOX voxel \(index) has an invalid or duplicate position or palette index."
            case .invalidModelIndex(let index): "The VOX asset has no model at index \(index)."
            case .scene: "The VOX scene hierarchy has invalid references, transforms or depth."
            case .meshLimit: "The VOX surface exceeds the supported mesh size."
            case .textureAllocation: "Metal could not allocate the VOX occupancy texture."
            }
        }
    }

    private static let maximumBytes = 64 * 1_024 * 1_024
    private static let maximumModels = 4_096
    private static let maximumVoxels = 4_194_304
    private static let maximumVertices = 1_048_576
    private static let maximumSceneVertices = 2_097_152
    private static let maximumSceneCells = 67_108_864
    private static let maximumChunks = 131_072
    private static let maximumSceneNodes = 16_384
    private static let maximumExpandedNodes = 65_536

    public static func load(contentsOf url: URL) throws -> Asset {
        guard url.isFileURL else { throw Failure.fileURL }
        let info = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard info.isRegularFile == true else { throw Failure.fileURL }
        if let size = info.fileSize, size > maximumBytes { throw Failure.size }
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var data = Data()
        while data.count <= maximumBytes {
            let count = min(1 << 20, maximumBytes + 1 - data.count)
            guard let chunk = try file.read(upToCount: count), !chunk.isEmpty else { break }
            data.append(chunk)
        }
        return try load(data)
    }

    public static func load(_ data: Data) throws -> Asset {
        guard data.count <= maximumBytes else { throw Failure.size }
        return try data.withUnsafeBytes { bytes in
            guard bytes.count >= 20 else { throw Failure.header }
            var reader = Reader(bytes: bytes, cursor: 0, end: bytes.count, chunk: "MAIN")
            guard try reader.string(count: 4) == "VOX " else { throw Failure.header }
            let version = try reader.integer()
            guard version == 150 || version == 200 else { throw Failure.version(version) }
            guard try reader.string(count: 4) == "MAIN", try reader.integer() == 0,
                  try reader.integer() == reader.remaining else { throw Failure.chunk("MAIN") }
            return try parse(reader, version: version)
        }
    }

    private struct Reader {
        let bytes: UnsafeRawBufferPointer
        var cursor: Int
        let end: Int
        let chunk: String
        var remaining: Int { end - cursor }

        mutating func take(_ count: Int) throws -> Range<Int> {
            guard count >= 0, count <= remaining else { throw Failure.chunk(chunk) }
            let start = cursor
            cursor += count
            return start..<cursor
        }

        mutating func integer() throws -> Int {
            let offset = try take(4).lowerBound
            let value = bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
            return Int(Int32(bitPattern: UInt32(littleEndian: value)))
        }

        mutating func string(count: Int) throws -> String {
            let range = try take(count)
            guard let text = String(bytes: bytes[range], encoding: .utf8) else { throw Failure.chunk(chunk) }
            return text
        }

        mutating func count(stride: Int, limit: Int = .max) throws -> Int {
            let count = try integer()
            guard count >= 0, count <= limit, count <= remaining / stride else { throw Failure.chunk(chunk) }
            return count
        }

        mutating func dictionary() throws -> [String: String] {
            let count = try count(stride: 8, limit: 4_096)
            var result: [String: String] = [:]
            for _ in 0..<count {
                let keySize = try integer()
                guard keySize <= 65_536 else { throw Failure.chunk(chunk) }
                let key = try string(count: keySize)
                let valueSize = try integer()
                guard valueSize <= 65_536 else { throw Failure.chunk(chunk) }
                let value = try string(count: valueSize)
                guard result.updateValue(value, forKey: key) == nil else { throw Failure.chunk(chunk) }
            }
            return result
        }
    }

    private enum SceneRecord {
        case transform(String?, SceneTransform, Int)
        case group([Int])
        case shape([Int])
    }

    private static func parse(_ input: Reader, version: Int) throws -> Asset {
        var reader = input
        var models: [Model] = []
        var pendingSize: SIMD3<Int>?
        var packedCount: Int?
        var palette: [SIMD4<UInt8>]?
        var sceneRecords: [Int: SceneRecord] = [:]
        var remainingVoxels = maximumVoxels
        var remainingChunks = maximumChunks

        while reader.remaining > 0 {
            guard remainingChunks > 0 else { throw Failure.size }
            remainingChunks -= 1
            let id = try reader.string(count: 4)
            let byteCount = try reader.integer()
            let childCount = try reader.integer()
            let content = try reader.take(byteCount)
            _ = try reader.take(childCount)
            var payload = Reader(bytes: reader.bytes, cursor: content.lowerBound, end: content.upperBound, chunk: id)
            switch id {
            case "PACK":
                let count = try payload.integer()
                guard packedCount == nil, models.isEmpty, pendingSize == nil,
                      (1...maximumModels).contains(count) else { throw Failure.chunk(id) }
                packedCount = count
            case "SIZE":
                guard pendingSize == nil, models.count < maximumModels else { throw Failure.chunk(id) }
                let size = try SIMD3(payload.integer(), payload.integer(), payload.integer())
                guard (0..<3).allSatisfy({ (1...256).contains(size[$0]) }) else { throw Failure.chunk(id) }
                pendingSize = size
            case "XYZI":
                guard let size = pendingSize else { throw Failure.chunk(id) }
                let count = try payload.count(stride: 4)
                guard count <= remainingVoxels else { throw Failure.size }
                guard count <= size.x * size.y * size.z else { throw Failure.chunk(id) }
                let voxels = try readVoxels(count: count, size: size, payload: &payload)
                models.append(Model(size: size, voxels: voxels))
                remainingVoxels -= count
                pendingSize = nil
            case "RGBA":
                guard palette == nil, content.count == 1_024 else { throw Failure.chunk(id) }
                palette = [.zero] + stride(from: content.lowerBound, to: content.upperBound - 4, by: 4).map {
                    SIMD4(reader.bytes[$0], reader.bytes[$0 + 1], reader.bytes[$0 + 2], reader.bytes[$0 + 3])
                }
                _ = try payload.take(content.count)
            case "nTRN", "nGRP", "nSHP":
                let node = try payload.integer()
                guard node >= 0, sceneRecords[node] == nil, sceneRecords.count < maximumSceneNodes else {
                    throw Failure.scene
                }
                sceneRecords[node] = try readSceneRecord(id, payload: &payload)
            default:
                continue
            }
            guard payload.remaining == 0, childCount == 0 else { throw Failure.chunk(id) }
        }
        guard pendingSize == nil, !models.isEmpty else { throw Failure.chunk("XYZI") }
        if let packedCount, packedCount != models.count { throw Failure.chunk("PACK") }
        let scene = try makeScene(records: sceneRecords, modelCount: models.count)
        return Asset(version: version, models: models, palette: palette ?? defaultPalette, scene: scene)
    }

    private static func readVoxels(count: Int, size: SIMD3<Int>, payload: inout Reader) throws -> [Voxel] {
        var voxels: [Voxel] = []
        var occupied = Set<UInt32>()
        occupied.reserveCapacity(count)
        voxels.reserveCapacity(count)
        let start = try payload.take(count * 4).lowerBound
        let bytes = payload.bytes
        for index in 0..<count {
            let offset = start + index * 4
            let position = SIMD3(bytes[offset], bytes[offset + 1], bytes[offset + 2])
            let color = bytes[offset + 3]
            let x = Int(position.x), y = Int(position.y), z = Int(position.z)
            guard x < size.x, y < size.y, z < size.z, color > 0 else { throw Failure.invalidVoxel(index) }
            let cell = UInt32(position.x) | (UInt32(position.y) << 8) | (UInt32(position.z) << 16)
            guard occupied.insert(cell).inserted else { throw Failure.invalidVoxel(index) }
            voxels.append(Voxel(position: position, paletteIndex: color))
        }
        return voxels
    }

    private static func readSceneRecord(_ id: String, payload: inout Reader) throws -> SceneRecord {
        let attributes = try payload.dictionary()
        switch id {
        case "nTRN":
            let child = try payload.integer()
            _ = try payload.integer() // Reserved.
            _ = try payload.integer() // Layer ID.
            let count = try payload.count(stride: 4, limit: maximumSceneNodes)
            var transform = SceneTransform()
            for frame in 0..<count {
                let data = try payload.dictionary()
                if frame == 0 { transform = try frameTransform(data) }
            }
            return .transform(attributes["_name"], transform, child)
        case "nGRP":
            let count = try payload.count(stride: 4, limit: maximumSceneNodes)
            return .group(try (0..<count).map { _ in try payload.integer() })
        default:
            let count = try payload.count(stride: 8, limit: maximumModels)
            let models = try (0..<count).map { _ in
                let index = try payload.integer()
                _ = try payload.dictionary()
                return index
            }
            return .shape(models)
        }
    }

    private static func frameTransform(_ frame: [String: String]) throws -> SceneTransform {
        var result = SceneTransform()
        if let translation = frame["_t"] {
            let words = translation.split(whereSeparator: \.isWhitespace)
            guard words.count == 3, let x = Int32(words[0]), let y = Int32(words[1]),
                  let z = Int32(words[2]) else { throw Failure.scene }
            result.translation = SIMD3(Float(x), Float(z), -Float(y))
        }
        if let rotation = frame["_r"] {
            guard let code = UInt8(rotation), code < 128 else { throw Failure.scene }
            let a = Int(code & 3), b = Int((code >> 2) & 3)
            guard a < 3, b < 3, a != b else { throw Failure.scene }
            var rows = [SIMD3<Float>](repeating: .zero, count: 3)
            rows[0][a] = code & 16 == 0 ? 1 : -1
            rows[1][b] = code & 32 == 0 ? 1 : -1
            rows[2][3 - a - b] = code & 64 == 0 ? 1 : -1
            let basis = simd_float3x3(rows: [SIMD3(1, 0, 0), SIMD3(0, 0, 1), SIMD3(0, -1, 0)])
            var oriented = basis * simd_float3x3(rows: rows) * basis.transpose
            if simd_determinant(oriented) < 0 {
                result.scale.x = -1
                oriented.columns.0 = -oriented.columns.0
            }
            result.rotation = simd_normalize(simd_quatf(oriented))
        }
        return result
    }

    private static func makeScene(records: [Int: SceneRecord], modelCount: Int) throws -> Node? {
        guard !records.isEmpty else { return nil }
        var active: Set<Int> = []
        var reached: Set<Int> = []
        var remainingNodes = maximumExpandedNodes

        func build(_ id: Int, depth: Int) throws -> Node {
            guard depth < 128, remainingNodes > 0, let record = records[id],
                  active.insert(id).inserted else { throw Failure.scene }
            defer { active.remove(id) }
            reached.insert(id)
            remainingNodes -= 1
            switch record {
            case .transform(let name, let transform, let child):
                let node = try build(child, depth: depth + 1)
                // A transform directly over a single-model shape belongs to its mesh.
                if case .shape? = records[child], let model = node.modelIndex {
                    return Node(name: name, transform: transform, modelIndex: model, children: [])
                }
                return Node(name: name, transform: transform, modelIndex: nil, children: [node])
            case .group(let children):
                return try Node(
                    name: nil, transform: SceneTransform(), modelIndex: nil,
                    children: children.map { try build($0, depth: depth + 1) }
                )
            case .shape(let models):
                guard models.allSatisfy({ (0..<modelCount).contains($0) }) else { throw Failure.scene }
                if models.count == 1 {
                    return Node(name: nil, transform: SceneTransform(), modelIndex: models[0], children: [])
                }
                guard models.count <= remainingNodes else { throw Failure.scene }
                remainingNodes -= models.count
                let children = models.map { Node(name: nil, transform: SceneTransform(), modelIndex: $0, children: []) }
                return Node(name: nil, transform: SceneTransform(), modelIndex: nil, children: children)
            }
        }
        let root = try build(0, depth: 0)
        guard reached.count == records.count else { throw Failure.scene }
        return root
    }

    // The format's default palette has a 6-level RGB cube followed by
    // red, green, blue and grayscale ramps. Index zero stays empty.
    private static let defaultPalette: [SIMD4<UInt8>] = {
        let levels: [UInt8] = [255, 204, 153, 102, 51, 0]
        let ramp: [UInt8] = [238, 221, 187, 170, 136, 119, 85, 68, 34, 17]
        var colors = [SIMD4<UInt8>.zero]
        for r in levels {
            for g in levels {
                for b in levels where r != 0 || g != 0 || b != 0 {
                    colors.append(SIMD4(r, g, b, 255))
                }
            }
        }
        colors += ramp.map { SIMD4($0, 0, 0, 255) }
        colors += ramp.map { SIMD4(0, $0, 0, 255) }
        colors += ramp.map { SIMD4(0, 0, $0, 255) }
        colors += ramp.map { SIMD4($0, $0, $0, 255) }
        return colors
    }()
}
