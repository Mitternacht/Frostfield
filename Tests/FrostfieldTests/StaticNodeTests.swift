import Foundation
import Frostfield
import Metal
import simd
import Testing

struct StaticNodeTests {
    @Test("A frozen static node keeps its world matrix when its parent or local pose changes")
    func frozenWorldMatrix() throws {
        let childID = UUID()
        var child = SceneNode(id: childID, transform: .init(translation: SIMD3(1, 0, 0)))
        child.isStatic = true
        let siblingID = UUID()
        let sibling = SceneNode(id: siblingID, transform: .init(translation: SIMD3(0, 1, 0)))
        var parent = SceneNode(transform: .init(translation: SIMD3(2, 0, 0)), children: [child, sibling])
        var graph = try SceneGraph(roots: [parent], camera: PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
        try graph.freezeStaticNodes()
        #expect(graph.roots[0].children[0].staticWorldMatrix?.columns.3.x == 3)

        graph.roots[0].transform.translation = SIMD3(10, 0, 0)
        graph.roots[0].children[0].transform.translation = SIMD3(8, 0, 0)
        let moved = try graph.resolve()
        #expect(worldX(moved, childID) == 3)
        #expect(worldX(moved, siblingID) == 10)
        #expect(worldX(moved, graph.roots[0].id) == 10)

        graph.roots[0].children[0].isStatic = false
        #expect(worldX(try graph.resolve(), childID) == 18)

        parent = graph.roots[0]
        parent.children[0].isStatic = true
        parent.children[0].staticWorldMatrix = nil
        parent.transform.translation = SIMD3(2, 0, 0)
        parent.children[0].transform.translation = SIMD3(1, 0, 0)
        graph.roots = [parent]
        try graph.freezeStaticNodes()
        graph.roots[0].transform.translation = SIMD3(10, 0, 0)
        try graph.freezeStaticNodes()
        #expect(worldX(try graph.resolve(), childID) == 3)
    }

    @Test("A child of a static parent still composes under the frozen parent")
    func childOfStaticParent() throws {
        let parentID = UUID()
        let childID = UUID()
        var parent = SceneNode(id: parentID, transform: .init(translation: SIMD3(2, 0, 0)))
        parent.isStatic = true
        parent.children = [SceneNode(id: childID, transform: .init(translation: SIMD3(1, 0, 0)))]
        var graph = try SceneGraph(roots: [parent], camera: PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10))
        try graph.freezeStaticNodes()
        graph.roots[0].transform.translation = SIMD3(9, 0, 0)
        graph.roots[0].children[0].transform.translation = SIMD3(4, 0, 0)
        let resolved = try graph.resolve()
        #expect(worldX(resolved, parentID) == 2)
        #expect(worldX(resolved, childID) == 6)
    }

    @Test("An archive keeps the frozen matrix")
    func archive() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let childID = UUID()
        var child = SceneNode(id: childID, transform: .init(translation: SIMD3(1, 0, 0)))
        child.isStatic = true
        var graph = try SceneGraph(
            roots: [SceneNode(transform: .init(translation: SIMD3(2, 0, 0)), children: [child])],
            camera: PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 10)
        )
        try graph.freezeStaticNodes()
        let archive = try SceneArchive(capturing: graph, commandQueue: queue)
        let json = try #require(JSONSerialization.jsonObject(with: archive.manifest) as? [String: Any])
        #expect(json["version"] as? Int == 42)
        var loaded = try archive.load(device: device, commandQueue: queue).graph
        loaded.roots[0].transform.translation = SIMD3(10, 0, 0)
        #expect(worldX(try loaded.resolve(), childID) == 3)
    }

    private func worldX(_ nodes: [ResolvedSceneNode], _ id: UUID) -> Float {
        nodes.first { $0.id == id }?.worldTransform.columns.3.x ?? .nan
    }
}
