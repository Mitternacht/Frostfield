# Scene draw and resource statistics

`SceneGraph.encode` returns `SceneDrawStatistics` for the submitted view.
Use its triangle, point, line-segment and sprite counts to inspect the
draws that passed scene culling. Mesh triangles include CPU-counted
instances and batched index ranges. Sprites contribute two triangles each.
Point and wide-line quads count under their logical point and segment
fields. `shadowTriangleCount` counts submitted casters across the shadow
passes. GPU-authored indirect arguments remain on the GPU and are reported
as draw records separately. These are submission counts before clipping,
depth testing or fragment rejection.
[Three.js r187 renderer information](https://github.com/mrdoob/three.js/blob/r187/src/renderers/common/Info.js)

```swift
let resources = graph.resourceStatistics()
let submitted = try graph.encode(pass: pass, commandBuffer: commandBuffer)
```

`resourceStatistics()` walks the current scene's prepared meshes, batches,
primitives, and instance sets. It includes hidden nodes and alternate LODs
because the graph still holds those resources. Each prepared drawable,
pipeline, and Metal resource identity counts once. Texture views count with
their root texture. Buffer-backed texture storage counts with its buffer,
and externally owned IOSurface textures have a separate count.
`bufferLengthBytes` sums logical buffer lengths. The allocation totals sum
the sizes Metal reports for the tracked resources, which can include
overlapping heap placements. A zero report for an existing resource leaves
the corresponding allocation total unavailable. The iPhone simulator
returned zero allocation bytes for nonempty buffers in the fixture.
The snapshot covers prepared draw resources and material textures.
Caller-owned attachments, global caches,
retired palettes, and transient command resources are outside its scope.
[Metal resource allocation size](https://developer.apple.com/documentation/metal/mtlresource/allocatedsize)

The resource fixture references one mesh and instance set from two nodes,
places an alternate mesh in a hidden LOD, and binds views of the same
texture storage. It verifies deduplication, logical buffer lengths, reported
allocation bytes and an empty graph snapshot. Draw tests verify mixed primitives, frustum culling,
instances, CPU-authored indirect bundles, GPU-authored argument records,
and directional shadow triangles.

The integrated `dae857d` run passed 1,366 package tests on the M4 Max
with five optional checks skipped. The focused suites passed 101 tests
with 14 skips on iPhone 18 Pro Simulator and 28 tests with one skip on
Vision Pro Simulator. All runs used Metal API Validation.
