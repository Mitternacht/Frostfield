# Batched mesh rendering

`BatchedMesh` packs a bounded palette of indexed triangle geometries into one
vertex and index allocation and prepares one material pipeline. A
`SceneNode(batch:)` supplies the parent transform and uses the regular color,
surface attribute, selection depth, and shadow passes. Insert, remove, or edit
an instance's geometry index, affine transform, tint and visibility without
rebuilding the palette. The instance handle survives edits. Deleting the
instance invalidates it even if another instance reuses the same numeric slot.

Use `addGeometry`, `replaceGeometry`, and `removeGeometry` to edit the
palette. Replacement preserves the geometry handle and updates every referencing
instance. Removal deletes its dependent instances. The batch can start empty and
can use smaller vertex and index budgets. A palette edit repacks its geometry and
prepares a new renderer and material pipeline. Encoded passes keep their original
palette. The batch allows three live palette generations, and an edit that needs
a new generation can retry after an earlier command completes or is discarded.

```swift
let batch = try BatchedMesh(
    device: device, colorPixelFormat: .rgba16Float,
    geometries: [box, sphere], capacity: 100
)
let first = try batch.addInstance(geometryIndex: 0, transform: boxTransform)
try batch.addInstance(geometryIndex: 1, transform: sphereTransform)
let node = SceneNode(batch: batch)
try batch.update(first, tint: SIMD4(0.8, 0.4, 0.2, 1))
```

Color passes can cull each visible instance and sort by the projected depth of
its bounds center. Opaque runs are front to back and blended runs back to front.
Ties use slot order, as does turning off sorting. Shadow passes cull in each
light's view. A custom sort receives the eligible instances and the pass's
clip matrix for color, surface attributes, shadows or selection depth. Return
every supplied handle once. An invalid permutation or an edit to the sorted
batch fails before its draw. A different batch can be edited from the callback.
Previously prepared draws continue using their captured palettes. Adjacent items with the same index range and world winding are
grouped into one indexed Metal instanced draw. Mixed mirrored and ordinary
placements use separate runs. An encoded pass retains an immutable instance
snapshot, including for command buffers with unretained references.
[Three.js r187 BatchedMesh](https://github.com/mrdoob/three.js/blob/r187/src/objects/BatchedMesh.js)

Batch-local bounds and `SceneGraph.worldBounds` union all active slots by
default. Pass `visibleOnly: true` to exclude hidden slots. Graph bounds also
apply the graph's visibility and layer filter in that mode.

Nearest and all-hit ray picking skip hidden slots and report the stable
`SceneHit.batchInstanceID`. Collision snapshots include active slots according
to their `visibleOnly` flag. Box selection tests each active slot's transformed
origin and returns its stable handle in slot order. It includes active slots
whose individual visibility is off, within the graph's visible nodes and
selected layers.
[Three.js r187 SelectionBox](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/interactive/SelectionBox.js)

OBJ, PLY and STL mesh exports flatten the
visible batch slots to world-space geometry. The scene archive, glTF/USDZ
scene export and path tracer report an error when the batch would otherwise
be lost.

The palette accepts up to 4,096 geometries, 1,048,576 vertices and 6,291,456
indices, subject to device buffer limits. Each batch supports up to 262,144
slots and each graph encode reserves up to 64 MiB for all batch instance
snapshots across passes. Independent per-geometry reservations, resizing,
and transmitting or vertex-deformed materials remain open.

Xcode 27.2 ran the eight batch tests and seven existing instancing tests on
arm64 macOS with Metal API Validation. The batch tests read color, normal
attributes and directional shadow pixels. They also cover transparent order,
mixed winding, stable slot reuse, picking, snapshot exports and unretained
command buffers after teardown. The Xcode MCP timed out resolving the
workspace's initial destination. Verification used `xcodebuild` with
isolated DerivedData.

The eight core batch tests also passed on the iPad Pro 13-inch (M5)
iOS 27.2 simulator with Metal API Validation enabled. The follow-up on both hosts
passed all 15 batch and box-selection tests, including transformed origins,
deleted slots, reused handles and hidden ancestors.

The palette follow-up passed 23 batch, selection and bounds tests on arm64
macOS and the same iPad simulator with Metal API Validation. The tests cover
replacement while older index ranges are in flight, geometry and instance
handle reuse, budget and lifetime errors, hidden-slot bounds, invalid custom
orders, and a callback replacing a different batch between its prepared color
and shadow draws.

## Indirect submission

Set `submission: .indirectCommandBuffer` when creating a batch to submit its
ordered indexed draws through a reusable Metal indirect command buffer (ICB).
Direct submission is the default. Check `BatchedMesh.supportsIndirectSubmission(on:)`
for the active device. An unavailable opt-in reports `.indirectUnsupported`.
[Apple Metal feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

The CPU records one indexed command per run. The normal render encoder binds
the current pass's pipeline, material, transforms and instance records. It
executes consecutive spans of equal winding in the prepared order. Color,
surface attributes, shadows and selection depth share this submission path.
The built-in textured fragment uses a Metal argument buffer for its texture
and sampler. Materials need a fragment pipeline that Metal can compile with
indirect command support.
`SceneDrawStatistics.indirectBatchExecutionCount` counts the actual indirect
executions reported by the color, attribute and shadow graph encoders.
[Apple Metal CPU indirect command buffers](https://developer.apple.com/documentation/metal/encoding-indirect-command-buffers-on-the-cpu)

Each palette caches up to four ordered layouts, with a Frostfield budget of
16,384 commands per layout. Changing transforms or tints reuses the ICB when
the run layout is identical. The renderer still determines the order and
uploads instance records for each view. Encoded draws retain the exact bundle
and palette through completion, including bundles evicted from the cache and command
buffers with unretained references.

This covers reusable indexed batch commands. GPU-written draw arguments with
byte offsets and reusable render bundles for arbitrary scene subtrees remain
separate capabilities.
[Three.js r187 BufferGeometry](https://github.com/mrdoob/three.js/blob/r187/src/core/BufferGeometry.js) ·
[Three.js r187 BundleGroup](https://github.com/mrdoob/three.js/blob/r187/src/renderers/common/BundleGroup.js)

Xcode 27.2 passed 27 tests on arm64 macOS with Metal API Validation,
including actual ICB output, direct versus indirect texture pixels, mixed
winding and blended order, per-pass bindings, layout reuse, and submission
after palette teardown and cache eviction.
The iPadOS 27.2 simulator passed 15 direct and capability tests with Metal API
Validation. Its device rejected the ICB opt-in and skipped five ICB execution
tests. Physical iPad and Vision Pro execution remain unverified.
[Apple Metal Simulator capabilities](https://developer.apple.com/documentation/metal/developing-metal-apps-that-run-in-simulator)
