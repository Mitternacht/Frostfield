# Dynamic raster geometry

Prepare a `MeshRenderer` with the `dynamicGeometry:` initializer and capacities
in vertices and triangles. Keep that renderer on its scene node and pass each
new indexed mesh to `updateGeometry`. This changes the draw count and the local
bounds together. World culling and CPU mesh picking use the new geometry, and
the material and prepared pipeline stay in place.

A replacement uploads the active vertex, index and applicable color data on its
first draw. Up to three different revisions can have draws in flight. A draw
retains its storage until the command completes or is discarded. Later draws of
the same revision share its slot. If all slots are in use, encode reports
`dynamicGeometryBuffersBusy`. Discard a failed command that could contain earlier
work and retry on a new command after an earlier revision is released.
`SceneGraph.resourceStatistics()` includes each resident slot's buffers and
their logical lengths. Metal-reported allocation sizes remain optional.

This provides the raster update path for a regenerated `MarchingCubes.mesh`.
Callers choose the capacity and rebuild the scalar field and indexed surface
on the CPU. For an empty field, hide the node until a surface becomes available.
Dynamic geometry is a full indexed-mesh replacement. Skinning, morphs,
instancing, vertex graph deformation, GPU indirect arguments and attribute-range
updates are outside this API. Static renderers retain their existing behavior.

`DynamicGeometryTests` exercises a blocked GPU draw across an update, vertex
color and triangle count changes, ring exhaustion and recovery, updated mesh
picking and culling, directional shadow and selection depth pixels, and
regenerated metaballs rendered through one prepared object.

The nine focused suites passed 50 tests on the Apple Silicon Mac and 50 on
the iPad Pro M5 simulator, with Metal API Validation enabled.
