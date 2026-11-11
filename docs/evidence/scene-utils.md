# Scene graph utilities

`SceneUtils.expandedInstances(of:)` converts an instanced draw into editable
child nodes. Each child keeps the current placement and tint and shares the
original mesh renderer, including its current geometry and draw range.
`SceneUtils.expandedBatch(of:)` converts active batch slots in numeric slot
order, retaining hidden entries and sharing one prepared renderer per used
geometry. Both preserve the original wrapper ID, local transform, visibility,
light, primitive, and existing child subtrees. Replace the source node to
keep its identity. The children have ordinary scene sorting and culling.
[Three.js r187 SceneUtils](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/utils/SceneUtils.js)

Expansion copies the current per-object values. Wrapper transform animation
continues to affect its children. Draw order, layers, material parameters,
pose, and shadow settings live on the generated draws. Affine shear cannot
be stored as a local TRS, and nonwhite tints for custom or surface graph
materials require a separate shader input. Those cases return typed errors
before replacing any source scene state.

`InstanceSet.sort(by:)` stably reorders source indices and applies the same
order to transforms and tints. It returns the previous index for each new
slot. A throwing comparison or an attempted reentrant edit leaves the set
unchanged. Native instance attributes are transforms and tints.
[Swift stable sorting guarantee](https://github.com/swiftlang/swift-evolution/blob/main/proposals/0372-document-sorting-as-stable.md)

`SceneUtils.multiMaterialObject` draws the full geometry once per material.
`SceneNode.grouped` already handles per-geometry material groups.
`SceneUtils.reduceVertices` walks visible world-space mesh, point, and line
positions, including each visible instance or batch slot. Camera-facing
sprites are outside this position reduction. The existing
`SceneGraph.forEachNode`, `forEachVisibleNode`, and `ancestors(of:)`
provide depth-first, visible, and ancestor traversal.

`SceneUtilsTests` checks pixel equality and independent mesh picking before
and after expansion, recycled slot order, hidden instances, per-geometry
renderer sharing, stable paired sorting, transactional shear rejection, and
world-space vertex reduction.

A pinned r187 reference run confirmed stable source order `[3, 1, 0, 2]`
for the repeated sort key fixture and kept the existing child after the
expanded instances. The nine focused suites passed 66 tests on the Apple
Silicon Mac and 66 on the iPad Pro M5 simulator with Metal API Validation
enabled.
