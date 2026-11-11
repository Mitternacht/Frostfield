import Foundation
import Frostfield
import Testing
import simd

struct SceneGraphTests {
    private let camera: PerspectiveCamera

    init() throws {
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 4, near: 0.1, far: 100)
    }

    @Test
    func `Scene resolution preserves depth-first order and composes parent transforms`() throws {
        let grandchild = SceneNode(transform: SceneTransform(translation: SIMD3<Float>(1, 0, 0)))
        let parameters = SIMD4<Float>(0.2, 0.4, 0.8, 1)
        let child = SceneNode(
            transform: SceneTransform(translation: SIMD3<Float>(1, 2, 0)),
            materialParameters: parameters,
            children: [grandchild]
        )
        let sibling = SceneNode()
        let parent = SceneNode(
            transform: SceneTransform(
                translation: SIMD3<Float>(5, 0, 0),
                rotation: simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(0, 0, 1)),
                scale: SIMD3<Float>(2, 3, 1)
            ),
            children: [child, sibling]
        )
        let anotherRoot = SceneNode()
        let graph = SceneGraph(roots: [parent, anotherRoot], camera: camera)

        let nodes = try graph.resolve()

        try #require(nodes.count == 5)
        #expect(nodes.map(\.id) == [parent.id, child.id, grandchild.id, sibling.id, anotherRoot.id])
        let origin = SIMD4<Float>(0, 0, 0, 1)
        #expect(simd_length(nodes[1].worldTransform * origin - SIMD4<Float>(-1, 2, 0, 1)) < 0.0001)
        #expect(simd_length(nodes[2].worldTransform * origin - SIMD4<Float>(-1, 4, 0, 1)) < 0.0001)
        #expect(nodes[1].materialParameters == parameters)
        #expect(nodes[0].mesh == nil)
    }

    @Test
    func `An empty graph resolves no nodes`() throws {
        let graph = SceneGraph(roots: [], camera: camera)
        let nodes = try graph.resolve()

        #expect(nodes.isEmpty)
    }

    @Test
    func `A node identifier cannot be reused in a different branch`() {
        let id = UUID()
        let graph = SceneGraph(
            roots: [SceneNode(children: [SceneNode(id: id)]), SceneNode(id: id)],
            camera: camera
        )

        #expect(throws: SceneGraph.ValidationError.duplicateID(id)) {
            try graph.resolve()
        }
    }

    @Test
    func `Invalid local transforms identify the failing node`() {
        for transform in [
            SceneTransform(scale: SIMD3<Float>(1, 0, 1)),
            SceneTransform(translation: SIMD3<Float>(.nan, 0, 0)),
            SceneTransform(rotation: simd_quatf(vector: .zero)),
        ] {
            let node = SceneNode(transform: transform)
            let graph = SceneGraph(roots: [node], camera: camera)

            #expect(throws: SceneGraph.ValidationError.invalidLocalTransform(node.id)) {
                try graph.resolve()
            }
        }
    }

    @Test
    func `Finite local transforms can overflow when composed`() {
        let transform = SceneTransform(scale: SIMD3<Float>(1e20, 1, 1))
        let child = SceneNode(transform: transform)
        let graph = SceneGraph(roots: [SceneNode(transform: transform, children: [child])], camera: camera)

        #expect(throws: SceneGraph.ValidationError.invalidWorldTransform(child.id)) {
            try graph.resolve()
        }
    }
}
