import simd

extension GLTFScene {
    /// A source light definition, shared by every node referring to its index.
    public struct Light {
        public let sourceIndex: Int
        public let name: String?
        public let light: PunctualLight
    }
}

struct GLTFLightDocument: Decodable {
    static let extensionName = "KHR_lights_punctual"

    private typealias Failure = GLTFMeshLoader.Failure

    private let extensions: DocumentExtensions?
    private let nodes: [Node]?

    private struct DocumentExtensions: Decodable {
        let punctual: Definitions?

        private enum CodingKeys: String, CodingKey { case punctual = "KHR_lights_punctual" }
    }

    private struct Definitions: Decodable {
        let lights: [Record]
    }

    private struct Record: Decodable {
        let name: String?
        let type: String
        let color: [Float]?
        let intensity: Float?
        let range: Float?
        let spot: Spot?
    }

    private struct Spot: Decodable {
        let innerConeAngle: Float?
        let outerConeAngle: Float?
    }

    private struct Node: Decodable {
        let extensions: NodeExtensions?
    }

    private struct NodeExtensions: Decodable {
        let punctual: LightReference?

        private enum CodingKeys: String, CodingKey { case punctual = "KHR_lights_punctual" }
    }

    private struct LightReference: Decodable {
        let light: Int
    }

    func nodeReferences(required: Bool, maximumElements: Int) throws -> [Int: Int] {
        let definitions = extensions?.punctual
        let count = definitions?.lights.count ?? 0
        if required || definitions != nil {
            guard count > 0 else { throw Failure.invalidLightExtension }
        }
        let records = nodes ?? []
        guard count <= maximumElements, records.count <= maximumElements else { throw Failure.sceneSizeLimit }

        var references: [Int: Int] = [:]
        for (nodeIndex, node) in records.enumerated() {
            guard let index = node.extensions?.punctual?.light else { continue }
            guard (0..<count).contains(index) else { throw Failure.missingLight(index) }
            references[nodeIndex] = index
        }
        return references
    }

    func resolve(nodes: [GLTFScene.Node]) throws -> [GLTFScene.Light] {
        let instances = nodes.compactMap(\.lightIndex)
        let limit = SceneGraph.maximumPunctualLightCount
        guard instances.count <= limit else { throw Failure.lightInstanceLimit(limit) }
        let records = extensions?.punctual?.lights ?? []
        return try Set(instances).sorted().map { index in
            guard records.indices.contains(index) else { throw Failure.missingLight(index) }
            let source = records[index]
            let kind: PunctualLight.Kind
            switch source.type {
            case "directional":
                guard source.range == nil else { throw Failure.invalidLight(index, "directional lights cannot specify a range") }
                kind = .directional
            case "point":
                kind = .point(range: source.range)
            case "spot":
                guard let spot = source.spot else { throw Failure.invalidLight(index, "spot lights require a spot object") }
                kind = .spot(
                    range: source.range, innerConeAngle: spot.innerConeAngle ?? 0,
                    outerConeAngle: spot.outerConeAngle ?? .pi / 4
                )
            default:
                throw Failure.unsupportedFeature("punctual light type \(source.type)")
            }

            let color = source.color ?? [1, 1, 1]
            guard color.count == 3 else { throw Failure.invalidLight(index, "color requires three linear components") }
            let light: PunctualLight
            do {
                light = try PunctualLight(kind: kind, color: SIMD3(color[0], color[1], color[2]), intensity: source.intensity ?? 1)
            } catch {
                throw Failure.invalidLight(index, error.localizedDescription)
            }
            return GLTFScene.Light(sourceIndex: index, name: source.name, light: light)
        }
    }
}
