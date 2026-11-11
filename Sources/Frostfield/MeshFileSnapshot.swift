import Foundation
import simd

enum MeshFileSnapshot {
    typealias Part = (name: String, geometry: MeshGeometry)

    static func meshes(in graph: SceneGraph) throws -> [Part] {
        try graph.resolve().flatMap(meshes(for:))
    }

    static func meshes(for node: ResolvedSceneNode) throws -> [Part] {
        let label = node.name ?? node.id.uuidString
        if let batch = node.batch {
            return try batch.objects(world: node.worldTransform).map { object in
                let geometry = try batch.geometry(for: object.instance.geometryID)
                return (label + "_\(object.instance.id.index)", try geometry.transformed(by: object.world))
            }
        }
        guard let mesh = node.mesh, node.instances?.count != 0 else { return [] }
        let geometry = try mesh.pickingGeometry(jointPalette: node.jointPalette, morphWeights: node.morphWeights)
        if let instances = node.instances {
            return try instances.transforms.enumerated().map { index, transform in
                (label + "_\(index)", try geometry.transformed(by: node.worldTransform * transform))
            }
        }
        return [(label, try geometry.transformed(by: node.worldTransform))]
    }

    static func position(_ point: SIMD3<Float>, in matrix: simd_float4x4) throws -> SIMD3<Float> {
        let value = SIMD4<Double>(SIMD3<Double>(point), 1)
        let result = SIMD4<Double>(matrix.columns.0) * value.x
            + SIMD4<Double>(matrix.columns.1) * value.y
            + SIMD4<Double>(matrix.columns.2) * value.z
            + SIMD4<Double>(matrix.columns.3)
        let position = SIMD3<Float>(Float(result.x), Float(result.y), Float(result.z))
        guard (0..<3).allSatisfy({ position[$0].isFinite }) else { throw Failure.position }
        return position
    }

    static func color(_ rgb: SIMD4<Float>) -> SIMD3<Float> {
        SIMD3(
            GLTFColorTables.linearToSRGB(min(1, max(0, rgb.x))),
            GLTFColorTables.linearToSRGB(min(1, max(0, rgb.y))),
            GLTFColorTables.linearToSRGB(min(1, max(0, rgb.z)))
        )
    }

    static func name(_ value: String, fallback: String) -> String {
        let safe = String(value.unicodeScalars.map { scalar -> Character in
            (32...126).contains(scalar.value) && scalar != "#" && scalar != "\\" ? Character(scalar) : "_"
        }).trimmingCharacters(in: .whitespaces)
        return safe.isEmpty ? fallback : safe
    }

    private enum Failure: LocalizedError {
        case position

        var errorDescription: String? { "The exported world-space position exceeds the Float range." }
    }
}
