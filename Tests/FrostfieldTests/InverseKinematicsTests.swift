import Foundation
import Frostfield
import Testing
import simd

struct InverseKinematicsTests {
    /// A planar arm along +X: shoulder at the origin, then unit bones to the elbow, wrist and tip.
    private func arm(rest: [simd_quatf]? = nil) throws -> (graph: SceneGraph, ids: [UUID]) {
        let tip = SceneNode(transform: .init(translation: SIMD3(1, 0, 0)))
        let wrist = SceneNode(transform: .init(translation: SIMD3(1, 0, 0), rotation: rest?[2] ?? .identity), children: [tip])
        let elbow = SceneNode(transform: .init(translation: SIMD3(1, 0, 0), rotation: rest?[1] ?? .identity), children: [wrist])
        let shoulder = SceneNode(transform: .init(rotation: rest?[0] ?? .identity), children: [elbow])
        let root = SceneNode(transform: .init(translation: SIMD3(0, 1, 0)), children: [shoulder])
        let graph = SceneGraph(roots: [root], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
        return (graph, [shoulder.id, elbow.id, wrist.id, tip.id])
    }

    private func position(_ id: UUID, in graph: SceneGraph) throws -> SIMD3<Float> {
        let world = try #require(graph.resolve().first { $0.id == id }).worldTransform
        return SIMD3(world.columns.3.x, world.columns.3.y, world.columns.3.z)
    }

    private func local(_ id: UUID, in graph: SceneGraph) throws -> simd_quatf {
        var pending = graph.roots
        while let node = pending.popLast() {
            if node.id == id { return node.transform.rotation }
            pending += node.children
        }
        throw CCDIKChain.Failure.missingNode(id)
    }

    @Test
    func `Free links reach a reachable target`() throws {
        let (graph, ids) = try arm()
        let chain = try CCDIKChain(effectorID: ids[3], links: [ids[2], ids[1], ids[0]].map { .init(nodeID: $0) }, iterations: 200, tolerance: 1e-5)
        for target in [SIMD3<Float>(1.5, 2.2, 0), SIMD3(-0.4, 1.5, 1.2), SIMD3(0.2, -0.6, -1)] {
            let solution = try chain.solve(in: graph, target: target)
            #expect(solution.distance < 1e-3)
            #expect(simd_distance(try position(ids[3], in: solution.graph), target) < 1e-3)
            // Bones keep their lengths: only rotations change.
            #expect(abs(simd_distance(try position(ids[1], in: solution.graph), SIMD3(0, 1, 0)) - 1) < 1e-5)
        }
        #expect(throws: CCDIKChain.Failure.invalidTarget) { try chain.solve(in: graph, target: SIMD3(.nan, 0, 0)) }
    }

    @Test
    func `Hinges stay within their limits and stop at the limit for an unreachable angle`() throws {
        let (graph, ids) = try arm()
        let z = SIMD3<Float>(0, 0, 1)
        // One hinge between 0 and 45°: a target straight above the shoulder needs 90°.
        let single = try CCDIKChain(
            effectorID: ids[1], links: [.init(nodeID: ids[0], constraint: .hinge(axis: z, minimumAngle: 0, maximumAngle: .pi / 4))]
        )
        let solution = try single.solve(in: graph, target: SIMD3(0, 3, 0))
        let elbow = try position(ids[1], in: solution.graph)
        #expect(simd_distance(elbow, SIMD3(cos(.pi / 4), 1 + sin(.pi / 4), 0)) < 1e-5)
        #expect(abs(solution.distance - simd_distance(elbow, SIMD3(0, 3, 0))) < 1e-5)
        let rotation = try local(ids[0], in: solution.graph)
        #expect(abs(rotation.angle - .pi / 4) < 1e-5)
        #expect(simd_distance(rotation.axis, z) < 1e-5)

        // Two hinges with room to move reach the target and respect their ranges.
        let limits: [(Float, Float)] = [(-0.2, 1.2), (0, 2.0)]
        let pair = try CCDIKChain(effectorID: ids[2], links: [
            .init(nodeID: ids[1], constraint: .hinge(axis: z, minimumAngle: limits[1].0, maximumAngle: limits[1].1)),
            .init(nodeID: ids[0], constraint: .hinge(axis: z, minimumAngle: limits[0].0, maximumAngle: limits[0].1)),
        ], iterations: 200, tolerance: 1e-5)
        let target = SIMD3<Float>(0.6, 2.4, 0)
        let reached = try pair.solve(in: graph, target: target)
        #expect(reached.distance < 1e-3)
        for (index, (low, high)) in [ids[0], ids[1]].enumerated().map({ ($0.offset, limits[$0.offset]) }) {
            let rotation = try local([ids[0], ids[1]][index], in: reached.graph)
            let signed = rotation.axis.z >= 0 ? rotation.angle : -rotation.angle
            #expect(signed >= low - 1e-5 && signed <= high + 1e-5)
            #expect(abs(abs(simd_dot(rotation.axis, z)) - 1) < 1e-4 || rotation.angle < 1e-5)
        }
    }

    @Test
    func `Euler limits clamp each angle like Three.js rotationMin and rotationMax`() throws {
        let (graph, ids) = try arm()
        let chain = try CCDIKChain(effectorID: ids[3], links: [
            .init(nodeID: ids[2], constraint: .euler(minimum: SIMD3(-0.1, -0.1, -0.5), maximum: SIMD3(0.1, 0.1, 0.5))),
            .init(nodeID: ids[1], constraint: .euler(minimum: SIMD3(-0.1, -0.1, -0.5), maximum: SIMD3(0.1, 0.1, 0.5))),
        ], iterations: 50)
        let solution = try chain.solve(in: graph, target: SIMD3(0, 4, 0))
        for id in [ids[1], ids[2]] {
            let matrix = simd_float3x3(try local(id, in: solution.graph))
            // Three.js XYZ order: y = asin(m13), x = atan2(-m23, m33), z = atan2(-m12, m11).
            let y = asin(min(max(matrix.columns.2.x, -1), 1))
            let x = atan2(-matrix.columns.2.y, matrix.columns.2.z)
            let zAngle = atan2(-matrix.columns.1.x, matrix.columns.0.x)
            #expect(abs(x) <= 0.1 + 1e-4 && abs(y) <= 0.1 + 1e-4 && abs(zAngle) <= 0.5 + 1e-4)
        }
        // The links bend fully toward the out-of-reach target.
        #expect(abs(abs(atan2(-simd_float3x3(try local(ids[1], in: solution.graph)).columns.1.x,
                              simd_float3x3(try local(ids[1], in: solution.graph)).columns.0.x)) - 0.5) < 1e-3)
    }

    @Test
    func `Blend factor, validation and ancestry`() throws {
        let (graph, ids) = try arm()
        let full = try CCDIKChain(effectorID: ids[3], links: [.init(nodeID: ids[0])], iterations: 1)
        let half = try CCDIKChain(effectorID: ids[3], links: [.init(nodeID: ids[0])], iterations: 1, blendFactor: 0.5)
        let target = SIMD3<Float>(0, 4, 0)
        let fullAngle = try local(ids[0], in: full.solve(in: graph, target: target).graph).angle
        let halfAngle = try local(ids[0], in: half.solve(in: graph, target: target).graph).angle
        #expect(abs(fullAngle - .pi / 2) < 1e-5)
        #expect(abs(halfAngle - .pi / 4) < 1e-5)
        #expect(throws: CCDIKChain.Failure.notAncestor(ids[3])) {
            try CCDIKChain(effectorID: ids[1], links: [.init(nodeID: ids[3])]).solve(in: graph, target: target)
        }
        #expect(throws: CCDIKChain.Failure.invalidLinks) { try CCDIKChain(effectorID: ids[1], links: []) }
        #expect(throws: CCDIKChain.Failure.invalidConstraint(ids[0])) {
            try CCDIKChain(effectorID: ids[1], links: [.init(nodeID: ids[0], constraint: .hinge(axis: .zero, minimumAngle: 0, maximumAngle: 1))])
        }
    }
}

private extension simd_quatf {
    static var identity: simd_quatf { simd_quatf(angle: 0, axis: SIMD3(0, 1, 0)) }
}
