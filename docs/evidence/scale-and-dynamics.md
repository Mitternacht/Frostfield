# Instancing, culling, LOD, and a dynamic path tracer

## Raster scale

| Capability | API | Verified by |
| --- | --- | --- |
| GPU instancing with per-instance transforms and tints in color, directional and punctual shadow passes, and picking | `InstanceSet`, `MeshRenderer(..., supportsInstancing: true)`, `SceneHit.instanceIndex` | `InstancingTests` |
| Per-draw morph target weights on the GPU, before skinning, in color and shadow passes | `MeshRenderer(..., morphTargets:)`, `SceneNode.morphWeights` | `MorphTargetRenderingTests` |
| Frustum culling with draw statistics; shadow casters use each light's own view | `SceneNode.frustumCulled`, `SceneNode.cullingBounds`, `SceneDrawStatistics` | `CullingAndLODTests` |
| Distance LOD with Three.js hysteresis, used by color, shadows, picking and path tracing | `LevelOfDetail`, `SceneNode.levelsOfDetail`, `SceneGraph.updateLevelsOfDetail` | `CullingAndLODTests` |
| A static node keeps a captured world matrix while its parent and local pose change | `SceneNode.isStatic`, `SceneNode.staticWorldMatrix`, `SceneGraph.freezeStaticNodes()` | `StaticNodeTests` |

The instancing tests compare one instanced draw with the same instances drawn as
separate nodes, and they hold an edit buffer ring under five encodes in flight.
GPU morph weights, including negative ones, render the same pixels as geometry
morphed on the CPU. Culled frames match frames with culling off, pixel for pixel.

Three deliberate breakages were caught by these tests: reusing a busy instance
buffer, culling shadow casters by the camera frustum, and skinning before
morphing.

Limits: instanced nodes are culled as a whole, with no per-instance GPU culling
or compaction. One instance set shares one handedness. LOD uses world distance,
not screen size. Skinned and graph-deformed meshes are culled only when the node
sets `cullingBounds`.

## Path tracer dynamics

| Capability | API | Verified by |
| --- | --- | --- |
| Two-level BVH: one bottom level per distinct mesh, shared by instances and by nodes with the same renderer | `PathTracingScene.statistics` | `PathSceneUpdateTests`, `InstancingTests` |
| Transform updates that refit the top level only | `PathTracingScene.updateTransforms(_:)` | `PathSceneUpdateTests` |
| Geometry updates for skinned, morphed or deformed meshes | `PathTracingScene.updateGeometry(for:in:replacements:)` | `PathSceneUpdateTests` |
| Stale history detection | `PathTracingScene.revision`, `PathTracer.Failure.sceneChanged` | `PathSceneUpdateTests` |
| Owen-scrambled Sobol sampling and a firefly clamp | `PathTracer.Settings.sampler`, `Settings.maximumSampleRadiance` | `PathSamplingTests` |
| GPU pass timing | `PathTracer.Timing` | `PathSamplingTests` |

Measured in the tests on an M4 Max Mac:

- Instanced, scaled and mirrored boxes and a mirrored glass block trace
  bit-identically to the same geometry baked into world space.
- A moved instance after `updateTransforms` matches a fresh preparation within
  1e-3, and no bottom level is rebuilt.
- RMSE against a 4,096-sample reference: at 16 samples per pixel, 0.0313 with
  independent sampling and 0.0108 with Sobol; at 64, 0.0157 and 0.0048.
- With a clamp of 1, a mirror caustic's peak falls from 14.37 to 1.0.
  Camera-visible emission is never clamped.
- Path-traced skinned silhouettes differ from raster at 1–2 of 1,536 pixels,
  all on edges.
- `PathTracer.Timing` reported 3.16 ms against 3.16 ms of command-buffer GPU time.

Limits: bottom-level updates are full rebuilds on the CPU, and skinning and
morphing for the path tracer run on the CPU. Updates cannot add or remove mesh
nodes or change a node's renderer or material parameters; those need a new
`PathTracingScene`. Sobol points are stratified per pixel with no blue-noise
ordering across pixels.

All tests ran with `xcodebuild` and Metal API Validation on an M4 Max Mac with
macOS 27.2. Nothing here was measured on iPhone, iPad or Vision Pro hardware, and
no comparison with Three.js was made.
