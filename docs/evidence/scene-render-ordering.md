# Scene render ordering

Both color `SceneGraph.encode` overloads accept `SceneRenderOrdering`.
Opaque and transparent comparators receive the node ID, drawable kind, group
order, render order, and projected depth when available. The depth convention
is consistent for forward and reversed depth targets. Equal comparisons keep
graph traversal order. Set `sortObjects` to false to use traversal order
within each draw phase. Depth-disabled overlays still draw after scene geometry.
Use `BatchedMesh.customSort` to order the items inside one batch.

Metal readback in `PrimitiveSceneGraphTests` verifies transparent mesh and sprite
composition, stable ties, traversal order with sorting disabled, and custom
opaque ordering with equal-depth meshes. The focused package run with Metal API
Validation passed 58 tests in eight suites on macOS and on iPhone 18 Pro
Simulator. It also exercised transmission, surface render targets, batch
submission, occlusion, GPU timing and layered render targets.

[Three.js r187 WebGLRenderLists](https://github.com/mrdoob/three.js/blob/r187/src/renderers/webgl/WebGLRenderLists.js)
