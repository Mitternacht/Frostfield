import Foundation
import Frostfield
import Testing

struct SurfaceShaderGraphTests {
    @Test("Multiple outputs validate together and share one reachable expression")
    func multiOutputCode() throws {
        let nodes: [MaterialGraph.Node] = [
            .uv(.u), .worldPosition(.y), .scalar(0.25),
            .noise(x: 0, y: 1, z: 2), .scalar(0.5), .subtract(3, 4),
            .cosine(5), .floor(6), .fract(5),
            .smoothstep(edge0: 2, edge1: 4, value: 3), .divide(4, 2),
            .rgb(red: 7, green: 8, blue: 10), .color(.zero),
        ]
        let graph = try SurfaceShaderGraph(
            nodes: nodes, outputs: .init(baseColor: 11, metallic: 3, roughness: 9)
        )
        let source = try graph.metalFunction(named: "fixture_surface")
        #expect(source.contains("static inline void fixture_surface("))
        #expect(source.contains("thread const FrostfieldSurfaceInput& input"))
        #expect(source.contains("thread FrostfieldSurfaceOutput& surface"))
        #expect(source.components(separatedBy: "float n3 =").count == 2)
        #expect(source.contains("float n5 = n3 - n4"))
        #expect(source.contains("float n6 = cos(n5)"))
        #expect(source.contains("float n7 = floor(n6)"))
        #expect(source.contains("float n8 = fract(n5)"))
        #expect(source.contains("frostfield_graph_smoothstep(n2, n4, n3)"))
        #expect(source.contains("frostfield_graph_finite(n2) && n2 != 0.0f"))
        #expect(source.contains("// node 3"))
        #expect(source.contains("baseColor = clamp("))
        #expect(source.contains("metallic = saturate("))
        #expect(source.contains("roughness = saturate("))
        #expect(!source.contains("surface.emission ="))
        #expect(!source.contains("n12"))
        #expect(try graph.metalFunction(named: "fixture_surface") == source)
        let other = try graph.metalFunction(named: "other_surface")
        #expect(other.contains("static inline void other_surface("))
        #expect(other.contains("frostfield_graph_value_noise("))
        #expect(!other.contains("fixture_surface"))
    }

    @Test("Surface graphs reject mismatched outputs, bad references and disconnected cycles")
    func validation() {
        #expect(throws: SurfaceShaderGraph.Failure.noOutputs) {
            try SurfaceShaderGraph(nodes: [], outputs: .init())
        }
        #expect(throws: SurfaceShaderGraph.Failure.outputType(channel: .roughness, node: 0, actual: .color)) {
            try SurfaceShaderGraph(nodes: [.color(.zero)], outputs: .init(baseColor: 0, roughness: 0))
        }
        #expect(throws: SurfaceShaderGraph.Failure.outputType(channel: .emission, node: 0, actual: .scalar)) {
            try SurfaceShaderGraph(nodes: [.scalar(1)], outputs: .init(emission: 0))
        }
        #expect(throws: SurfaceShaderGraph.Failure.outputType(channel: .transmissionMask, node: 0, actual: .color)) {
            try SurfaceShaderGraph(nodes: [.color(.zero)], outputs: .init(transmissionMask: 0))
        }
        #expect(throws: MaterialGraph.Failure.invalidOutput(2)) {
            try SurfaceShaderGraph(nodes: [.scalar(1)], outputs: .init(metallic: 2))
        }
        #expect(throws: MaterialGraph.Failure.cycle(1)) {
            try SurfaceShaderGraph(nodes: [.scalar(1), .cosine(1)], outputs: .init(metallic: 0))
        }
        #expect(throws: MaterialGraph.Failure.nonfiniteConstant(1)) {
            try SurfaceShaderGraph(nodes: [.scalar(1), .scalar(.infinity)], outputs: .init(roughness: 0))
        }
        #expect(throws: MaterialGraph.Failure.typeMismatch(node: 1, input: 0, expected: .scalar, actual: .color)) {
            try SurfaceShaderGraph(nodes: [.color(.zero), .divide(0, 0)], outputs: .init(baseColor: 1))
        }
    }

    @Test("Transmission graph outputs clamp to the material's optical range")
    func transmissionMaskCode() throws {
        let graph = try SurfaceShaderGraph(nodes: [.scalar(2)], outputs: .init(transmissionMask: 0))
        let source = try graph.metalFunction(named: "ice_surface")
        #expect(source.contains("surface.transmissionMask = saturate(frostfield_graph_finite(n0) ? n0 : finite_transmissionMask)"))
    }

    @Test("Generated function names reject reserved names, non-ASCII and source delimiters",
          arguments: ["", "9surface", "surface);", "sürface", "float3", "thread", "ff__graph", "ff_"])
    func invalidName(_ name: String) throws {
        let graph = try SurfaceShaderGraph(nodes: [.scalar(0)], outputs: .init(roughness: 0))
        #expect(throws: SurfaceShaderGraph.Failure.invalidFunctionName(name)) {
            try graph.metalFunction(named: name)
        }
    }
}
