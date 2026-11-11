import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct SceneUtilsTests {
    private let gpu: GeometryRenderFixture

    init() throws { gpu = try GeometryRenderFixture() }

    @Test("Instance expansion preserves reflected poses, tints, picking and the wrapper subtree")
    func instancedExpansion() throws {
        let geometry = try GeometryRenderFixture.square(0.28)
        let mesh = try gpu.unlit(geometry, supportsInstancing: true)
        let left = SceneTransform(translation: SIMD3(-0.55, 0, 0), scale: SIMD3(-1, 1, 1))
        let right = SceneTransform(translation: SIMD3(0.55, 0, 0), scale: SIMD3(-1, 1, 1))
        let tints = [SIMD4<Float>(1, 0, 0, 1), SIMD4<Float>(0, 1, 0, 1)]
        let set = try InstanceSet(device: gpu.device, transforms: [left.matrix, right.matrix], tints: tints)
        let originalChild = SceneNode(name: "Already present")
        let parameters = SIMD4<Float>(0.5, 0.75, 0.5, 1)
        let node = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -3)), mesh: mesh, materialParameters: parameters,
            children: [originalChild], castsShadow: false, instances: set, name: "Tiles", renderOrder: 4
        )
        let source = scene(node)
        let expanded = try SceneUtils.expandedInstances(of: node)
        #expect(expanded.id == node.id && expanded.transform == node.transform)
        #expect(expanded.mesh == nil && expanded.instances == nil)
        #expect(expanded.children.map(\.name) == ["Tiles_0", "Tiles_1", "Already present"])
        #expect(expanded.children[2].id == originalChild.id)
        #expect(expanded.children[0].mesh === mesh && expanded.children[1].mesh === mesh)
        #expect(expanded.children[0].instances == nil && expanded.children[1].instances == nil)
        #expect(expanded.children[0].materialParameters == parameters * tints[0])
        #expect(expanded.children[1].materialParameters == parameters * tints[1])
        #expect(expanded.children[0].renderOrder == 4 && !expanded.children[0].castsShadow)
        let detached = scene(expanded)
        let before = try gpu.render(source)
        let after = try gpu.render(detached)
        #expect(before.pixels == after.pixels)
        #expect(isNearly(
            gpu.pixel(before.pixels, x: gpu.column(x: -0.55, depth: 3), y: 32),
            GeometryRenderFixture.unlitFacingCamera(parameters * tints[0])
        ))
        #expect(isNearly(
            gpu.pixel(after.pixels, x: gpu.column(x: 0.55, depth: 3), y: 32),
            GeometryRenderFixture.unlitFacingCamera(parameters * tints[1])
        ))
        #expect(before.statistics.drawnInstanceCount == 2 && after.statistics.drawnMeshCount == 2)

        let ray = try cameraRay(0.55)
        let originalHit = try #require(try source.pick(ray: ray))
        let expandedHit = try #require(try detached.pick(ray: ray))
        #expect(originalHit.instanceIndex == 1 && expandedHit.instanceIndex == nil)
        #expect(expandedHit.nodeID == expanded.children[1].id)
        #expect(originalHit.intersection == expandedHit.intersection)

        let originalPositions = try positions(source)
        #expect(originalPositions.count == 2 * geometry.vertices.count)
        #expect(try positions(detached) == originalPositions)

        var edited = detached
        try edited.updateNode(withID: expanded.children[1].id) { $0.transform.translation.x += 1 }
        #expect(try edited.pick(ray: ray) == nil)
        #expect(try source.pick(ray: ray)?.instanceIndex == 1)
    }

    @Test("Batch expansion retains stable slot order, hidden entries and per-geometry renderer sharing")
    func batchExpansion() throws {
        let square = try GeometryRenderFixture.square(0.28)
        let other = try MeshGeometry.plane(width: 0.25, height: 0.35)
        let batch = try BatchedMesh(
            device: gpu.device, colorPixelFormat: .rgba16Float, geometries: [square, other], capacity: 4
        )
        batch.sortsObjects = false
        let red = SIMD4<Float>(1, 0, 0, 1), green = SIMD4<Float>(0, 1, 0, 1), blue = SIMD4<Float>(0, 0, 1, 1)
        try batch.addInstance(geometryIndex: 0, transform: translationMatrix(-0.6), tint: red)
        let removed = try batch.addInstance(geometryIndex: 0, transform: translationMatrix(-6))
        let hidden = try batch.addInstance(geometryIndex: 1, tint: blue, isVisible: false)
        try batch.remove(removed)
        let reused = try batch.addInstance(geometryIndex: 0, transform: translationMatrix(0.6), tint: green)
        #expect(reused.index == removed.index && reused != removed && hidden.index == 2)

        let node = SceneNode(transform: .init(translation: SIMD3(0, 0, -3)), name: "Batch", batch: batch)
        let expanded = try SceneUtils.expandedBatch(of: node)
        #expect(expanded.id == node.id && expanded.batch == nil)
        #expect(expanded.children.map(\.name) == ["Batch_0", "Batch_1", "Batch_2"])
        #expect(expanded.children.map(\.materialParameters) == [red, green, blue])
        #expect(expanded.children.map(\.isVisible) == [true, true, false])
        #expect(expanded.children[0].mesh === expanded.children[1].mesh)
        #expect(expanded.children[0].mesh !== expanded.children[2].mesh)
        #expect(expanded.children[2].mesh?.geometry.indices == other.indices)

        let source = scene(node), detached = scene(expanded)
        let before = try gpu.render(source)
        let after = try gpu.render(detached)
        #expect(before.pixels == after.pixels)
        #expect(isNearly(
            gpu.pixel(before.pixels, x: gpu.column(x: -0.6, depth: 3), y: 32),
            GeometryRenderFixture.unlitFacingCamera(red)
        ))
        #expect(isNearly(
            gpu.pixel(after.pixels, x: gpu.column(x: 0.6, depth: 3), y: 32),
            GeometryRenderFixture.unlitFacingCamera(green)
        ))
        let right = try cameraRay(0.6)
        #expect(try source.pick(ray: right)?.batchInstanceID == reused)
        #expect(try detached.pick(ray: right)?.nodeID == expanded.children[1].id)
        #expect(try source.pick(ray: cameraRay(0)) == nil && detached.pick(ray: cameraRay(0)) == nil)
        #expect(try positions(source) == positions(detached))
    }

    @Test("Stable sorting keeps transforms and tints paired and rejects reentrant edits")
    func sortInstances() throws {
        let transforms = [2, 1, 2, 0].map { translationMatrix(Float($0)) }
        let tints = (0..<4).map { SIMD4(Float($0) / 4, 0, 0, 1) }
        let set = try InstanceSet(device: gpu.device, transforms: transforms, tints: tints)
        let originalRevision = set.revision
        let indices = try set.sort { set.transforms[$0].columns.3.x < set.transforms[$1].columns.3.x }
        #expect(indices == [3, 1, 0, 2])
        #expect(set.transforms == indices.map { transforms[$0] })
        #expect(set.tints == indices.map { tints[$0] })
        #expect(set.revision == originalRevision + 1)
        let sortedRevision = set.revision
        let identity = try set.sort { set.transforms[$0].columns.3.x < set.transforms[$1].columns.3.x }
        #expect(identity == [0, 1, 2, 3] && set.revision == sortedRevision)
        #expect(throws: InstanceSet.Failure.reentrantSort) {
            try set.sort { first, _ in
                try set.setTint(.zero, at: first)
                return false
            }
        }
        #expect(set.tints == indices.map { tints[$0] } && set.revision == sortedRevision)
    }

    @Test("An unrepresentable shear rejects expansion without editing the source")
    func shear() throws {
        let mesh = try gpu.unlit(try GeometryRenderFixture.square(0.4), supportsInstancing: true)
        var matrix = matrix_identity_float4x4
        matrix.columns.1.x = 0.5
        let set = try InstanceSet(device: gpu.device, transforms: [matrix_identity_float4x4, matrix])
        let node = SceneNode(mesh: mesh, instances: set)
        let revision = set.revision
        #expect(throws: SceneUtils.Failure.unrepresentableTransform(instance: 1)) {
            try SceneUtils.expandedInstances(of: node)
        }
        #expect(node.mesh === mesh && node.instances === set)
        #expect(set.transforms[1] == matrix && set.revision == revision)
    }

    @Test("Graph materials keep independent parameters when instance tint is neutral")
    func graphTint() throws {
        let graph = try SurfaceShaderGraph { scope in
            .init(baseColor: scope.constant(SIMD3<Float>(0.3, 0.6, 0.2)))
        }
        let mesh = try gpu.unlit(
            GeometryRenderFixture.square(0.4), supportsInstancing: true, material: .physicallyBased(.init(surfaceGraph: graph))
        )
        let set = try InstanceSet(device: gpu.device, transforms: [matrix_identity_float4x4])
        let node = SceneNode(mesh: mesh, materialParameters: SIMD4(0.3, 0.2, 0.4, 1), instances: set)
        #expect(try SceneUtils.expandedInstances(of: node).children[0].materialParameters == node.materialParameters)
        try set.setTint(SIMD4(1, 0, 0, 1), at: 0)
        #expect(throws: SceneUtils.Failure.unsupportedTint(instance: 0)) {
            try SceneUtils.expandedInstances(of: node)
        }
    }

    @Test("Whole-mesh material copies and visible point/line reduction keep authored geometry")
    func materialsAndReduction() throws {
        let geometry = try GeometryRenderFixture.square(0.4).withGroups([
            .init(start: 0, count: 3, materialIndex: 1),
        ])
        let object = try SceneUtils.multiMaterialObject(geometry, materials: [.unlit, .lambert], device: gpu.device)
        #expect(object.children.count == 2)
        #expect(object.children.allSatisfy { $0.mesh?.geometry.indices == geometry.indices })
        #expect(object.children[0].mesh !== object.children[1].mesh)

        var points = SceneNode(transform: .init(translation: SIMD3(1, 0, 0)))
        points.primitive = .points(try PointsRenderer(
            device: gpu.device, cloud: PointCloud(positions: [.zero, SIMD3(0, 1, 0)]), material: PointsMaterial()
        ))
        let line = try DebugHelpers(device: gpu.device).node(LineGeometry(positions: [.zero, SIMD3(1, 0, 0)]))
        let hidden = SceneNode(children: [pointsCopy(points)], isVisible: false)
        let root = SceneNode(transform: .init(translation: SIMD3(0, 2, -3)), children: [points, hidden, line])
        let graph = scene(root)
        #expect(try positions(graph) == [SIMD3(1, 2, -3), SIMD3(1, 3, -3), SIMD3(0, 2, -3), SIMD3(1, 2, -3)])
        #expect(try positions(graph, from: points.id) == [SIMD3(1, 2, -3), SIMD3(1, 3, -3)])
        #expect(try positions(graph, from: hidden.id).isEmpty)
    }

    private func pointsCopy(_ source: SceneNode) -> SceneNode {
        var node = SceneNode(transform: source.transform)
        node.primitive = source.primitive
        return node
    }

    private func scene(_ node: SceneNode) -> SceneGraph { SceneGraph(roots: [node], camera: gpu.camera) }

    private func cameraRay(_ x: Float) throws -> SceneRay {
        try SceneRay(origin: SIMD3(x, 0, 0), direction: SIMD3(0, 0, -1))
    }

    private func positions(_ graph: SceneGraph, from id: UUID? = nil) throws -> [SIMD3<Float>] {
        try SceneUtils.reduceVertices(in: graph, from: id, into: []) { $0.append($1) }
    }
}
