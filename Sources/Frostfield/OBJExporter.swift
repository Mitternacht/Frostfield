import Foundation
import simd

/// Writes Wavefront OBJ triangles, lines and points with optional vertex colors.
/// UV0 is written in OBJ's bottom-left convention. Material groups get named
/// OBJ groups. Material definitions and textures require a different format.
public enum OBJExporter {
    public static func data(from geometry: MeshGeometry, name: String = "mesh") -> Data {
        var writer = Writer()
        writer.mesh(geometry, name: name)
        return Data(writer.text.utf8)
    }

    /// Writes a named material reference and, optionally, one MTL file token.
    /// Supply the corresponding MTL file separately when sharing the OBJ.
    public static func data(
        from geometry: MeshGeometry, name: String = "mesh", materialName: String, materialLibrary: String? = nil
    ) throws -> Data {
        var writer = Writer(text: try libraryHeader(materialLibrary))
        try writer.mesh(geometry, name: name, materialName: materialLabel(materialName))
        return Data(writer.text.utf8)
    }

    public static func data(from cloud: PointCloud, name: String = "points") -> Data {
        var writer = Writer()
        writer.points(cloud.positions, colors: cloud.colors, name: name)
        return Data(writer.text.utf8)
    }

    public static func data(from geometry: LineGeometry, name: String = "lines") -> Data {
        var writer = Writer()
        writer.lines(geometry, positions: geometry.positions, name: name)
        return Data(writer.text.utf8)
    }

    /// Captures a scene's resolved geometry in world coordinates, including
    /// hidden nodes, selected LODs, posed meshes and expanded mesh instances.
    /// Sprites and render-time appearance are omitted. Vertex colors are
    /// clamped and written as sRGB, independent of materials or instance tints.
    public static func data(from graph: SceneGraph) throws -> Data {
        try scene(graph, materialNames: [:], defaultMaterialName: nil, header: "")
    }

    /// Writes material references for all emitted objects. A node's instances
    /// share its material name. Objects without a mapping use the supplied
    /// default, so an earlier object's material cannot carry into them.
    public static func data(
        from graph: SceneGraph, materialNames: [UUID: String], defaultMaterialName: String, materialLibrary: String? = nil
    ) throws -> Data {
        try scene(
            graph, materialNames: materialNames.mapValues(materialLabel),
            defaultMaterialName: materialLabel(defaultMaterialName), header: libraryHeader(materialLibrary)
        )
    }

    private static func scene(
        _ graph: SceneGraph, materialNames: [UUID: String], defaultMaterialName: String?, header: String
    ) throws -> Data {
        var writer = Writer(text: header)
        for node in try graph.resolve() {
            let materialName = materialNames[node.id] ?? defaultMaterialName
            for part in try MeshFileSnapshot.meshes(for: node) {
                writer.mesh(part.geometry, name: part.name, materialName: materialName)
            }
            let name = node.name ?? node.id.uuidString
            switch node.primitive {
            case .points(let renderer):
                let cloud = renderer.cloud
                let positions = try cloud.positions.map { try MeshFileSnapshot.position($0, in: node.worldTransform) }
                writer.points(positions, colors: cloud.colors, name: name, materialName: materialName)
            case .lines(let renderer):
                let geometry = renderer.geometry
                let positions = try geometry.positions.map { try MeshFileSnapshot.position($0, in: node.worldTransform) }
                writer.lines(geometry, positions: positions, name: name, materialName: materialName)
            case .sprites, nil: break
            }
        }
        guard writer.vertexCount > 0 else { throw Failure.noGeometry }
        return Data(writer.text.utf8)
    }

    private static func materialLabel(_ name: String) throws -> String {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, validReference(trimmed) else {
            throw MaterialReferenceFailure.invalidName
        }
        return trimmed
    }

    private static func libraryHeader(_ name: String?) throws -> String {
        guard let name else { return "" }
        guard !name.isEmpty, !name.contains(" "), validReference(name) else {
            throw MaterialReferenceFailure.invalidLibraryName
        }
        return "mtllib \(name)\n"
    }

    private static func validReference(_ name: String) -> Bool {
        name.unicodeScalars.allSatisfy { scalar in
            scalar == " " || (
                scalar != "#" && scalar != "\\"
                    && !CharacterSet.whitespacesAndNewlines.contains(scalar)
                    && !CharacterSet.controlCharacters.contains(scalar)
            )
        }
    }

    public enum Failure: LocalizedError, Equatable {
        case noGeometry

        public var errorDescription: String? { "The scene contains no OBJ meshes, lines or point clouds." }
    }

    public enum MaterialReferenceFailure: LocalizedError, Equatable {
        case invalidName
        case invalidLibraryName

        public var errorDescription: String? {
            switch self {
            case .invalidName: "Use a nonempty printable OBJ material name without comments or line escapes."
            case .invalidLibraryName: "Use one printable MTL file token without spaces, comments or line escapes."
            }
        }
    }

    private struct Writer {
        var text = ""
        var vertexCount = 0
        var meshVertexCount = 0

        mutating func mesh(_ geometry: MeshGeometry, name: String, materialName: String? = nil) {
            let name = MeshFileSnapshot.name(name, fallback: "mesh")
            beginObject(name, materialName: materialName)
            for vertex in geometry.vertices { position(vertex.position, color: vertex.color) }
            for vertex in geometry.vertices {
                let uv = vertex.textureCoordinate
                text += "vt \(uv.x) \(1 - uv.y)\n"
            }
            for vertex in geometry.vertices {
                let n = simd_normalize(SIMD3<Double>(vertex.normal))
                text += "vn \(Float(n.x)) \(Float(n.y)) \(Float(n.z))\n"
            }

            var groups = geometry.groups[...]
            var material: Int?
            for start in stride(from: 0, to: geometry.indices.count, by: 3) {
                while let group = groups.first, group.range.upperBound <= start { groups = groups.dropFirst() }
                let next = groups.first.flatMap { $0.range.contains(start) ? $0.materialIndex : nil }
                if next != material {
                    text += "g \(name)\(next.map { "_material_\($0)" } ?? "")\n"
                    material = next
                }
                text += "f"
                for corner in geometry.indices[start..<(start + 3)] {
                    let position = vertexCount + Int(corner) + 1
                    let attribute = meshVertexCount + Int(corner) + 1
                    text += " \(position)/\(attribute)/\(attribute)"
                }
                text += "\n"
            }
            if !geometry.groups.isEmpty { text += "g default\n" }
            vertexCount += geometry.vertices.count
            meshVertexCount += geometry.vertices.count
        }

        mutating func points(
            _ positions: [SIMD3<Float>], colors: [SIMD4<Float>]?, name: String, materialName: String? = nil
        ) {
            beginObject(MeshFileSnapshot.name(name, fallback: "points"), materialName: materialName)
            for (index, point) in positions.enumerated() { position(point, color: colors?[index]) }
            for index in positions.indices { text += "p \(vertexCount + index + 1)\n" }
            vertexCount += positions.count
        }

        mutating func lines(
            _ geometry: LineGeometry, positions: [SIMD3<Float>], name: String, materialName: String? = nil
        ) {
            beginObject(MeshFileSnapshot.name(name, fallback: "lines"), materialName: materialName)
            for (index, point) in positions.enumerated() { position(point, color: geometry.colors?[index]) }
            if geometry.topology == .segments {
                for segment in geometry.segments {
                    text += "l \(vertexCount + segment.start + 1) \(vertexCount + segment.end + 1)\n"
                }
            } else {
                text += "l"
                for index in positions.indices { text += " \(vertexCount + index + 1)" }
                if geometry.topology == .loop { text += " \(vertexCount + 1)" }
                text += "\n"
            }
            vertexCount += positions.count
        }

        private mutating func beginObject(_ name: String, materialName: String?) {
            text += "o \(name)\n"
            if let materialName { text += "usemtl \(materialName)\n" }
        }

        private mutating func position(_ p: SIMD3<Float>, color: SIMD4<Float>?) {
            text += "v \(p.x) \(p.y) \(p.z)"
            if let color {
                let rgb = MeshFileSnapshot.color(color)
                text += " \(rgb.x) \(rgb.y) \(rgb.z)"
            }
            text += "\n"
        }
    }
}
