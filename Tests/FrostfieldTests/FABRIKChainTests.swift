import Foundation
import Frostfield
import Testing
import simd

struct FABRIKChainTests {
    @Test
    func `A spatial chain matches the pinned FABRIK joint positions without changing the input graph`() throws {
        let tip = SceneNode(transform: .init(translation: SIMD3(1, 0, 0)))
        let wrist = SceneNode(
            transform: .init(translation: SIMD3(1, 0, 0), rotation: simd_quatf(angle: .pi / 2, axis: SIMD3(0, 1, 0))),
            children: [tip]
        )
        let elbow = SceneNode(
            transform: .init(translation: SIMD3(1, 0, 0), rotation: simd_quatf(angle: .pi / 2, axis: SIMD3(0, 0, 1))),
            children: [wrist]
        )
        let branch = SceneNode(transform: .init(translation: SIMD3(0, 0, 1)))
        let base = SceneNode(children: [elbow, branch])
        let nodes = [base, elbow, wrist, tip]
        let graph = try scene(base)
        let solver = try FABRIKChain(joints: nodes.map(\.id), iterations: 64)
        let target = SIMD3<Float>(0.6, 1.4, -1.1)
        let original = try anchors(nodes, in: graph)
        let solved = try solver.solve(in: graph, target: target)
        let posed = try anchors(nodes, in: solved.graph)
        // pmndrs/math eda3d8c, solved with a 1e-5 distance threshold.
        let reference: [SIMD3<Float>] = [
            .zero, SIMD3(0.99502197, 0.09780041, -0.01914030),
            SIMD3(0.92338621, 1.07666605, -0.21068762), target
        ]
        #expect(solved.reachedTarget && solved.iterations > 0 && solved.iterations <= 64)
        for (actual, expected) in zip(posed, reference) {
            #expect(simd_distance(actual, expected) < 3e-5)
        }
        #expect(abs(solved.distance - Double(simd_distance(posed[3], target))) < 1e-6)
        #expect(try anchors(nodes, in: graph) == original)
        try expectStableLengths(original, posed)
        for node in nodes {
            let result = try #require(solved.graph.node(withID: node.id))
            #expect(result.transform.translation == node.transform.translation)
            #expect(result.transform.scale == node.transform.scale)
            #expect(graph.node(withID: node.id)?.transform == node.transform)
        }
        #expect(solved.graph.node(withID: tip.id)?.transform == tip.transform)
        #expect(solved.graph.node(withID: branch.id)?.transform == branch.transform)

        let unchanged = try solver.solve(in: solved.graph, target: target)
        #expect(unchanged.reachedTarget && unchanged.iterations == 0)
        #expect(try anchors(nodes, in: unchanged.graph) == posed)
    }

    @Test
    func `Pivoted joints preserve their lengths under nested rotated and scaled parents`() throws {
        let tip = SceneNode(
            transform: .init(
                translation: SIMD3(0.5, 0.1, -0.25), rotation: simd_quatf(angle: -0.25, axis: SIMD3(0, 1, 0)),
                scale: SIMD3(repeating: 1.1)
            ),
            pivot: SIMD3(0.09, 0, 0.1)
        )
        let joint = SceneNode(
            transform: .init(
                translation: SIMD3(0.8, 0.05, 0.1), rotation: simd_quatf(angle: 0.4, axis: SIMD3(0, 1, 0)),
                scale: SIMD3(repeating: 0.75)
            ),
            children: [tip], pivot: SIMD3(-0.08, 0.12, 0)
        )
        let base = SceneNode(
            transform: .init(
                translation: SIMD3(0.2, -0.1, 0.3), rotation: simd_quatf(angle: 0.3, axis: SIMD3(0, 0, 1)),
                scale: SIMD3(repeating: 1.5)
            ),
            children: [joint], pivot: SIMD3(0.1, 0.15, -0.05)
        )
        let parent = SceneNode(
            transform: .init(
                translation: SIMD3(3, 2, -4), rotation: simd_quatf(angle: 0.25, axis: SIMD3(0, 0, 1)),
                scale: SIMD3(repeating: 2)
            ),
            children: [base], pivot: SIMD3(0.2, -0.4, 0.1)
        )
        let nodes = [base, joint, tip]
        let graph = try scene(parent)
        // A separately authored pose supplies a provably reachable world target.
        var witness = graph
        try witness.updateNode(withID: base.id) { $0.transform.rotation = simd_quatf(angle: 0.55, axis: SIMD3(0, 0, 1)) }
        try witness.updateNode(withID: joint.id) { $0.transform.rotation = simd_quatf(angle: -1.1, axis: SIMD3(0, 1, 0)) }
        let target = try witness.worldPosition(of: tip.id)
        let solver = try FABRIKChain(joints: nodes.map(\.id), iterations: 128, tolerance: 3e-5)
        let initial = try anchors(nodes, in: graph)
        let result = try solver.solve(in: graph, target: target)
        let posed = try anchors(nodes, in: result.graph)
        #expect(result.reachedTarget)
        #expect(simd_distance(posed[2], target) < 3e-5)
        #expect(simd_distance(posed[0], initial[0]) < 3e-6)
        try expectStableLengths(initial, posed)
        #expect(result.graph.node(withID: parent.id)?.transform == parent.transform)
        #expect(result.graph.node(withID: base.id)?.pivot == base.pivot)
        #expect(result.graph.node(withID: joint.id)?.pivot == joint.pivot)
        #expect(result.graph.node(withID: tip.id)?.transform == tip.transform)
    }

    @Test("Straight and folded interior targets, and both unreachable bounds",
          arguments: [SIMD3<Float>(1.1, 0, 0), .zero, SIMD3(0, 5, 0), SIMD3(0.3, 0, 0)])
    func radialCases(_ target: SIMD3<Float>) throws {
        let inner = target.x == 0.3
        let outer = target.y == 5
        let (graph, nodes) = try straightArm(first: inner ? 2 : 1, second: inner ? 0.5 : 1)
        let solver = try FABRIKChain(joints: nodes.map(\.id), iterations: 128)
        let result = try solver.solve(in: graph, target: target)
        let initial = try anchors(nodes, in: graph)
        let posed = try anchors(nodes, in: result.graph)
        #expect(posed[0] == .zero)
        try expectStableLengths(initial, posed)
        if outer {
            #expect(!result.reachedTarget && result.iterations == 1)
            #expect(simd_distance(posed[2], SIMD3(0, 2, 0)) < 1e-6)
            #expect(abs(result.distance - 3) < 1e-6)
        } else if inner {
            #expect(!result.reachedTarget && result.iterations == 1)
            #expect(simd_distance(posed[2], SIMD3(1.5, 0, 0)) < 1e-6)
            #expect(abs(result.distance - 1.2) < 1e-6)
        } else {
            #expect(result.reachedTarget)
            #expect(simd_distance(posed[2], target) < 1e-5)
        }
    }

    @Test
    func `A bounded solve reports the actual remaining error and a single bone can reverse`() throws {
        let (graph, nodes) = try straightArm(first: 1, second: 1)
        let limited = try FABRIKChain(joints: nodes.map(\.id), iterations: 1, tolerance: 0)
        let target = SIMD3<Float>(0.85, 0.65, 0.1)
        let result = try limited.solve(in: graph, target: target)
        let actual = try result.graph.worldPosition(of: nodes[2].id)
        #expect(!result.reachedTarget && result.iterations == 1)
        #expect(abs(result.distance - Double(simd_distance(actual, target))) < 1e-6)
        #expect(result.distance < Double(simd_distance(SIMD3(2, 0, 0), target)))

        let single = try FABRIKChain(joints: [nodes[0].id, nodes[1].id])
        let reversed = try single.solve(in: graph, target: SIMD3(-1, 0, 0))
        #expect(reversed.reachedTarget && reversed.iterations == 1)
        #expect(simd_distance(try reversed.graph.worldPosition(of: nodes[1].id), SIMD3(-1, 0, 0)) < 1e-6)
    }

    @Test
    func `Invalid chains and poses fail without mutating the input`() throws {
        let (graph, nodes) = try straightArm(first: 1, second: 1)
        let ids = nodes.map(\.id)
        let solver = try FABRIKChain(joints: ids)
        #expect(throws: FABRIKChain.Failure.invalidJoints) { try FABRIKChain(joints: [ids[0]]) }
        #expect(throws: FABRIKChain.Failure.invalidJoints) { try FABRIKChain(joints: [ids[0], ids[0]]) }
        #expect(throws: FABRIKChain.Failure.invalidSettings) { try FABRIKChain(joints: ids, iterations: 0) }
        #expect(throws: FABRIKChain.Failure.invalidSettings) { try FABRIKChain(joints: ids, tolerance: .infinity) }
        #expect(throws: FABRIKChain.Failure.invalidTarget) { try solver.solve(in: graph, target: SIMD3(.nan, 0, 0)) }
        #expect(throws: FABRIKChain.Failure.disconnectedChain) {
            try FABRIKChain(joints: [ids[0], ids[2]]).solve(in: graph, target: .zero)
        }
        let missing = UUID()
        #expect(throws: FABRIKChain.Failure.missingEffector(missing)) {
            try FABRIKChain(joints: [ids[0], missing]).solve(in: graph, target: .zero)
        }

        var collapsed = graph
        try collapsed.updateNode(withID: ids[1]) { $0.transform.translation = .zero }
        #expect(throws: FABRIKChain.Failure.zeroLengthBone(ids[0])) { try solver.solve(in: collapsed, target: .zero) }
        var scaled = graph
        try scaled.updateNode(withID: ids[0]) { $0.transform.scale.x = 2 }
        #expect(throws: FABRIKChain.Failure.nonuniformScale(ids[0])) { try solver.solve(in: scaled, target: .zero) }
        var worldOriented = graph
        try worldOriented.updateNode(withID: ids[1]) { $0.orientationMode = .world }
        #expect(throws: FABRIKChain.Failure.worldOrientedNode(ids[1])) {
            try solver.solve(in: worldOriented, target: .zero)
        }
        var invalid = graph
        try invalid.updateNode(withID: ids[1]) { $0.transform.rotation = simd_quatf(vector: .zero) }
        #expect(throws: FABRIKChain.Failure.invalidTransform(ids[1])) { try solver.solve(in: invalid, target: .zero) }
        #expect(try graph.worldPosition(of: ids[2]) == SIMD3(2, 0, 0))
    }

    private func scene(_ root: SceneNode) throws -> SceneGraph {
        SceneGraph(roots: [root], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 30))
    }

    private func straightArm(first: Float, second: Float) throws -> (SceneGraph, [SceneNode]) {
        let tip = SceneNode(transform: .init(translation: SIMD3(second, 0, 0)))
        let joint = SceneNode(transform: .init(translation: SIMD3(first, 0, 0)), children: [tip])
        let base = SceneNode(children: [joint])
        return (try scene(base), [base, joint, tip])
    }

    private func anchors(_ nodes: [SceneNode], in graph: SceneGraph) throws -> [SIMD3<Float>] {
        try nodes.enumerated().map { index, node in
            let anchor = index == nodes.count - 1 ? .zero : node.pivot ?? .zero
            return try graph.localToWorld(anchor, in: node.id)
        }
    }

    private func expectStableLengths(_ initial: [SIMD3<Float>], _ posed: [SIMD3<Float>]) throws {
        try #require(initial.count == posed.count)
        for index in 1..<initial.count {
            let before = simd_distance(initial[index - 1], initial[index])
            let after = simd_distance(posed[index - 1], posed[index])
            #expect(abs(before - after) < 5e-6)
        }
    }
}
