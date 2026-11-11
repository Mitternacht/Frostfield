# Render hooks and the scene encode inspector

Frostfield has native equivalents for two of the three Three.js r187 per-object render
hooks. They are CPU-side observers, not a way to change a draw. The third,
`setRenderObjectFunction`, has no faithful native equivalent and stays unimplemented.

| r187 capability | Frostfield | Status |
| --- | --- | --- |
| `Object3D.onBeforeRender`, `onAfterRender` | `SceneNode.onBeforeEncode`, `onAfterEncode` | partial |
| `Renderer.inspector` (`InspectorBase`) | `SceneGraph.encodeInspector`, `SceneEncodeInspector` | partial |
| `Renderer.setRenderObjectFunction` | none | missing, see [below](#setrenderobjectfunction-is-not-implemented) |

## API

```swift
var node = SceneNode(mesh: renderer)
node.onBeforeEncode = { context in
    // The draw's commands have not reached the encoder yet.
    drawLog.append((context.nodeID, context.drawIndex, context.viewIndex))
}
node.onAfterEncode = { context in
    // The draw's commands are in the encoder. The GPU has not run them.
}

var graph = SceneGraph(roots: [node], camera: camera)
graph.encodeInspector = myInspector          // an object that conforms to SceneEncodeInspector
let statistics = try graph.encode(pass: pass, commandBuffer: commandBuffer)
```

`SceneRenderObjectContext` is an `Equatable`, `Sendable` value. It holds no renderer, no
encoder and no scene reference.

| Field | Meaning |
| --- | --- |
| `nodeID` | The drawn node's ID. |
| `kind` | `SceneRenderOrdering.Kind`: `.mesh` (plain, instanced or skinned), `.batch`, `.points`, `.lines`, `.sprites`. A node with a mesh and a primitive reports two contexts. |
| `pass` | `.color` or `.surfaceAttributes` (the `encodeSurfaceAttributes` prepass). |
| `drawIndex` | The draw's position in this encode's draw order, from 0. It counts every draw, hooked or not, mesh and primitive together. It continues across the opaque and overlay passes of a transmission encode and restarts for each array-camera view. |
| `viewIndex`, `viewCount` | The array-camera view being encoded, or nil outside an array-camera encode. |
| `depthMode` | The encode's depth mode. |
| `worldTransform` | The node's resolved world matrix, as the draw uses it. With `highPrecision` the GPU draws from a camera-relative Double matrix derived from it. |
| `materialParameters` | The parameters the draw is encoded with: the node's, or the material override's when one replaced them. |
| `usesMaterialOverride`, `isBlended` | Whether an override replaced the material, and whether the draw is in the blended phase. |
| `instanceCount` | The instance count of an instanced mesh, else nil. |

`SceneEncodeInspector` has four methods, all with empty defaults so that an inspector
overrides only what it needs:

- `sceneEncodeDidBegin(_:graph:)` when an encode starts. Like r187 `beginRender`.
- `sceneEncode(_:didEncode:statistics:)` after each draw, with a `SceneDrawStatistics` that
  holds only that draw's counts.
- `sceneEncodeDidFinish(_:statistics:)` with exactly the statistics the encode returns. Like r187 `finishRender`.
- `sceneEncodeDidFail(_:error:)` when the encode throws after it began.

`SceneEncodeInfo` carries an `id` that is distinct for every observed encode, the pass, the
view index and count, and the depth mode.

## When a hook runs

A hook runs for a draw that is really encoded, in encode order, once per encode and once per
array-camera view. It does not run for:

- a hidden node or a descendant of one, or a node whose layers miss the graph's `cameraLayers`
  or the array-camera view's layers;
- a node outside the camera frustum, unless its `frustumCulled` is false. Such an
  uncullable node still draws, so its hooks run, as in r187;
- an instanced node with no instances, or any draw whose index range is empty (the code skips it, and no test covers an empty index range);
- a group node that only carries a transform, even when it has hooks;
- any draw of an encode with an empty scissor rectangle;
- the validation-only pass that an array-camera encode runs over every view after the first.

Hidden, filtered and culled nodes follow r187. The empty-draw, zero-instance and empty-scissor rules are
Frostfield's own. If the encode throws while a draw is encoded, that node's before hook has run and its
after hook does not; the inspector gets the failure. Each hook pair brackets everything the node's draw
encodes in that pass. By code, that includes
a toon outline hull drawn with it; no test covers a toon hull. A material override changes the
context, not the call count.

The hooks run in the color pass and in the surface attribute prepass. They do not run in
shadow passes, in the toon outline pass that draws hulls after the scene, in the motion vector
pass, in depth-only selection captures, or in the single-mesh `MeshRenderer.encode(MeshScene)`
path, which has no node.

Frostfield runs other captures through the same color encode: planar reflections, cube
captures, occlusion queries and GPU timing. A hook on a node in the graph they encode runs
there too, and an inspector on that graph sees each capture as a separate encode. That follows
from the shared code path. No test covers those captures.

## What the hooks cannot do

- They get no renderer, camera, scene, geometry, material or group object. `nodeID` finds the
  node, whose mesh is its geometry and material. A Three.js multi-material mesh maps to
  `SceneNode.grouped`, whose per-material children each report their own context.
- They cannot change the draw. The scene validates every draw and prepares its transform,
  material parameters, instances, joint palette and morph weights before any hook runs. Change
  those before calling `encode`.
- They cannot change GPU state of a draw. The hook runs on the CPU while the scene's render
  encoder is open, and the GPU executes the commands later. A write to a buffer that a shader
  reads reaches every draw in the command buffer that reads it, not only the hooked draw.
  Use the hooks for bookkeeping, counters and deciding per-frame values.
- A hook must not encode into the same command buffer, because the scene's encoder is open.
  Encoding into another command buffer is possible.
- Closures are not stored by `SceneArchive`.

## setRenderObjectFunction is not implemented

In r187 `renderer.setRenderObjectFunction(fn)` replaces the function that renders each object.
`fn(object, scene, camera, geometry, material, group, lightsNode, clippingContext, passId)` must
call `renderer.renderObject(...)`, and it may call it with another material, more than once, or
after other work, for example a pre-pass with an override material
([Renderer.js](https://github.com/mrdoob/three.js/blob/r187/src/renderers/common/Renderer.js#L3165-L3205)).
A faithful native version would have to let a closure run the default draw and encode its own
commands around it. Frostfield does not offer that, for these reasons:

1. A draw is a prepared, immutable `MeshRenderer`. Its pipeline states come from the material,
   attachment formats, depth mode and sample count when it is created. There is no mutable
   material object that a closure could swap per call. The nearest existing feature,
   `SceneMaterialOverride` with `excluding`, replaces the material for a whole graph. It does not
   give per-object, per-call control.
2. A pre-pass or a second draw needs the draw's vertex, index, palette, instance, deformed
   vertex and indirect-argument buffers. They are private to the encode.
3. A closure that is given the `MTLRenderCommandEncoder` can leave state behind that later
   default draws do not reset, such as visibility result mode, scissor, fill mode or stencil
   reference. The scene encoder tracks some of this state in locals. Making the callback safe
   would need a state snapshot and restore for every draw, and a test for each kind of draw. That
   design is not written or tested, so no API is exposed.

What exists instead: `onBeforeEncode` and `onAfterEncode` to observe each draw,
`SceneMaterialOverride` and `ScenePrimitiveMaterialOverride` to render a graph with another
material, and `isVisible` and layers to skip nodes.

## What r187 does

- `Object3D.onBeforeRender` and `onAfterRender` are empty methods that a program replaces
  ([Object3D.js](https://github.com/mrdoob/three.js/blob/r187/src/core/Object3D.js#L418-L440)).
  `WebGLRenderer.renderObjects` calls `renderObject` for an object only when its layers test
  against the camera's, and `renderObject` brackets the draw with the two callbacks
  ([WebGLRenderer.js](https://github.com/mrdoob/three.js/blob/r187/src/renderers/WebGLRenderer.js#L2141-L2198)).
  Objects culled by `projectObject` never reach it
  ([`projectObject`](https://github.com/mrdoob/three.js/blob/r187/src/renderers/WebGLRenderer.js#L1873)).
  An `ArrayCamera` renders the list once per sub-camera
  ([lines 1767-1790](https://github.com/mrdoob/three.js/blob/r187/src/renderers/WebGLRenderer.js#L1767-L1790)).
- The WebGPU `Renderer.renderObject` calls the same two hooks around its draw
  ([Renderer.js](https://github.com/mrdoob/three.js/blob/r187/src/renderers/common/Renderer.js#L3924)
  and [L4020](https://github.com/mrdoob/three.js/blob/r187/src/renderers/common/Renderer.js#L4020)),
  and `_renderObjects` goes through `_currentRenderObjectFunction`, which is the function that
  `setRenderObjectFunction` sets
  ([L3659-L3668](https://github.com/mrdoob/three.js/blob/r187/src/renderers/common/Renderer.js#L3659-L3668)).
- `onBeforeShadow` and `onAfterShadow` are called around shadow-map draws
  ([WebGLShadowMap.js](https://github.com/mrdoob/three.js/blob/r187/src/renderers/webgl/WebGLShadowMap.js#L548-L566)).
- `InspectorBase` is the base class of `Renderer.inspector`. Its `beginRender(uid, scene, camera,
  renderTarget)` and `finishRender(uid)` are called around each `render()`
  ([Renderer.js L2091](https://github.com/mrdoob/three.js/blob/r187/src/renderers/common/Renderer.js#L2091)
  and [L2268](https://github.com/mrdoob/three.js/blob/r187/src/renderers/common/Renderer.js#L2268)).
  `begin()` and `finish()` bracket an animation-loop frame
  ([Animation.js](https://github.com/mrdoob/three.js/blob/r187/src/renderers/common/Animation.js)).
  `beginCompute`, `finishCompute`, `computeAsync`, `copyTextureToTexture`,
  `copyFramebufferToTexture` and `inspect(node)` serve compute passes, copies and the node
  system
  ([InspectorBase.js](https://github.com/mrdoob/three.js/blob/r187/src/renderers/common/InspectorBase.js)).
  The renderer always owns an inspector, a no-op `InspectorBase` by default.

## Departures from r187

| r187 | Frostfield |
| --- | --- |
| Hooks run when the object is rendered and receive `(renderer, scene, camera, geometry, material, group)`. They may change the object, its material or render targets, as a `Reflector` does. | Hooks run while the draw is encoded and receive a plain context. They observe; the draw was already prepared. |
| `scene.onBeforeRender`, `scene.onAfterRender` and `material.onBeforeRender` exist. | Not provided. |
| `onBeforeShadow`, `onAfterShadow` run around shadow-map draws. | Not provided. Shadow passes never call a hook. |
| An `ArrayCamera` culls with one union frustum. | Each view culls with its own frustum (see `ArrayCamera`), so a node can have fewer hook calls than in r187. |
| `WebGLRenderer.renderTransmissionPass` re-renders opaque objects into the transmission render target ([L2015](https://github.com/mrdoob/three.js/blob/r187/src/renderers/WebGLRenderer.js#L2015)), which calls their hooks again. | The transmission backdrop is copied from the color attachment. Each draw's hooks run once. |
| A double-sided transparent material draws twice between one hook pair. | One hook pair covers whatever the node's draw encodes. |
| `inspect`, compute, `copyTextureToTexture`, `copyFramebufferToTexture`, `begin` and `finish`, `setRenderer`, `enabled`, `dispose`. | Not provided. Scene encodes are the only observed events. The host owns frame timing, and compute passes and copies are not scene encodes. |
| `finishRender` is skipped when `render` throws. | `sceneEncodeDidFail` reports it. |
| An inspector is always installed. | `encodeInspector` is nil by default. |
| `uid` identifies a render context. | `SceneEncodeInfo.id` identifies an encode. |

## Cost with no hook set

With no hook and no inspector, an encode does one `contains` scan over the resolved nodes, one
integer increment per draw, and an `Optional` check in the wrapper that reports the encode. The
resolved node holds two more optional closures. The scan and the increments are the only added
work per draw. No timing benchmark was run, so there is no measured overhead figure.

## Verification

All runs used Xcode 27.2 command-line `xcodebuild` with `MTL_DEBUG_LAYER=1`, because the Xcode
MCP bridge times out and its DerivedData volume is full. They ran on the Mac Studio (Apple M4 Max,
macOS 27.2) and on the iPad Pro (M5) simulator. Logs are in the recovery checkout under
`verification/render-hooks-*.log`.

### Default path unchanged

Before any hook code existed, commit `a39b0fe` added a fixture that renders eight 64 by 64
scenes (a plain, an instanced, a skinned and a batched mesh, points, a line, sprites, and all
seven together) and hashes every channel of every pixel with FNV-1a. It ran on the unmodified
tip `b756c35` (`render-hooks-baseline-mac.log`: "Test run with 1 test in 1 suite passed",
`** TEST SUCCEEDED **`) and recorded these digests on the Apple M4 Max:

| Scene | Digest |
| --- | --- |
| plain | `d56bc48b3618ada5` |
| instanced | `c13314b3449ddecc` |
| skinned | `62762605f4fbcf25` |
| batched | `e191ca19629a43fd` |
| points | `b7c837bb36d16725` |
| line | `bd56c18f0b897a1c` |
| sprites | `08f2f9703e19d625` |
| all | `42276cc6deb254fd` |

`RenderHooksDefaultPathTests` keeps them. It compares only on macOS with an Apple M4 Max, so it passes
without comparing on any other GPU, and a deliberate change to the output of these node kinds has to
update the digests. On the M4 Max it fails if any digest of the current
code differs, and all eight comparisons ran and matched (`RENDER-HOOKS-BASELINE-COMPARED`
appears eight times in `render-hooks-mac-final.log`). The iPad Pro (M5) simulator renders the
same eight digests, which `diff` confirmed against the recorded ones. On every device the test
also renders each scene a second time with a hook on every node and an inspector installed, and
requires the pixels and the `SceneDrawStatistics` to equal the render without them.

### Focused runs after the change

Thirty suites ran: the two hook suites, the array camera, scene graph, mesh renderer, picking,
occlusion, GPU timing, material override, post-process scene pass, planar reflection, cube
camera, primitive scene graph, batched mesh and indirect batch, instancing, points and sprites,
viewport snapshot, scene archive and coding, prepared glTF scene, toon outline, transmission,
surface render targets, mesh indirect draw, ambient occlusion, screen-space reflection, layered
render target and cascaded shadow map suites.

| Destination | Command line | Result line |
| --- | --- | --- |
| macOS, Apple M4 Max | `MTL_DEBUG_LAYER=1 xcodebuild test -scheme Frostfield-Package -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:FrostfieldTests/<suite> ...` | "Test run with 184 tests in 30 suites passed after 3.918 seconds." and `** TEST SUCCEEDED **` |
| iPad Pro (M5) simulator `C131684E-215A-4D00-9C92-C877CE2581C7` | the same, with `-destination 'platform=iOS Simulator,id=...'` | "Test run with 184 tests in 30 suites passed after 7.596 seconds." and `** TEST SUCCEEDED **` |

### Whole package on macOS

One run of the whole `Frostfield-Package` scheme on the Mac, on the committed sources
(`render-hooks-mac-full.log`), ended with "Test run with 2183 tests in 426 suites passed after
154.392 seconds.", "Test run with 6 tests in 2 suites passed after 0.091 seconds." and
`** TEST SUCCEEDED **`. The whole package was not run on the iPad simulator.

### The hook tests

`RenderHooksTests` has eleven tests and `RenderHooksDefaultPathTests` has one.

| Test | What it requires |
| --- | --- |
| Every drawable node kind calls its hooks once | Plain, instanced, skinned and batched meshes, points, a line and sprites each call both hooks once with the right `kind`, `instanceCount`, world transform and material parameters. The pixel at the position that the context's matrix projects to carries the node's tint, checked for the plain and skinned meshes only. |
| A node with a mesh and a primitive reports one context per draw | Two bracketed contexts. |
| Hooks run before the draw's commands reach the encoder and after | A proxy command buffer logs every draw call that the real Metal encoder receives. The log must read before, one or more draws, after for each node, in draw index order, with no draw outside a bracket. |
| A hidden, filtered, culled, empty or group node never calls its hooks | Only the visible, the uncullable and the child of a transform-only group are called. Hidden, other-layer, culled, zero-instance, hidden-parent and group nodes are not, and `drawnMeshCount` and `culledMeshCount` agree. An empty scissor calls nothing. |
| Hooks run once per drawn view of an array-camera encode | Exact event sequence over two views with view layers and a culled node, with `viewIndex`, `viewCount` and a draw index that restarts per view. A single-view array reports 0 of 1. The preflight pass adds no calls. |
| Draw indices count every draw | A node sees the same `drawIndex` whether every node or only two have hooks. |
| A material override replaces the parameters the hook sees | `materialParameters` and `usesMaterialOverride` match the override, and the pixel carries the override color. |
| The surface attribute prepass calls hooks | The four opaque mesh kinds are called with `pass == .surfaceAttributes`. Primitives are not. |
| An opaque and an overlay pass of one transmission encode share one draw count | The opaque mesh is called before the glass mesh although it is listed after it, with indices 0 and 1. |
| The inspector sees begin, every draw, and finish | Event order, one `SceneEncodeInfo` per encode, distinct increasing IDs, `didFinish` statistics equal to the returned ones, and per-draw statistics that sum to the returned draw, instance, triangle, point, line, sprite and indirect counts. A toon outline hull drawn with a node breaks the triangle sum, and no test covers a hull. |
| An encode that throws reports the failure | `begin` then `fail` and no `finish` for an out-of-range viewport. A two-view array camera reports exactly `begin 0, draw 0, finish 0, begin 1, draw 1, finish 1`. |
| Hooks and an inspector leave every pixel and every statistic unchanged | See above. |

### Broken encodes

To check that the tests can fail, twelve defects were put into the encode and the two hook suites
were run against them, then every defect was removed. The first run applied ten at once, so the
assertion that failed, not the defect list, ties each failure to its cause. Several rules, such as the
empty-scissor, view-layer and group-node rules, were never mutated, so their tests have not been seen to fail. Two rows below name
two defects each.

| Defect | Failed |
| --- | --- |
| The before hook runs after the draw is encoded | the encoder-order test |
| Culled draws are encoded and given hooks, and hidden nodes reach the draw list | the suppressed-node test and the array-camera test |
| Primitives never call the after hook or the inspector | the node-kind test and the inspector test |
| An array-camera preflight pass is reported to the inspector | the inspector failure and array-camera test (the two-finishes expectation failed) |
| Primitives are not encoded when observed | the default-path test, for the points, line, sprites and all-together scenes |
| The view count is reported as 1 | the array-camera tests |
| Unhooked draws do not advance the draw index | the stable-index test |
| The hook sees the material parameters from before an override | the override test |
| The hook sees an identity world transform | the node-kind test |
| The draw index restarts in each encode pass | the transmission test |
| The attribute prepass reports `.color` | the attribute prepass test |

`render-hooks-mutation-mac.log`: "Test run with 11 tests in 2 suites failed after 0.377 seconds
with 30 issues." In that run the test process aborted once, in the test fixture, which divided by
zero for the identity transform. `xcodebuild` restarted it, and its summary includes both
launches. The fixture now maps such a point to a corner, and that fixed fixture was only run
against correct encodes and the last two defects. Those two ran separately, in
`render-hooks-mutation2-mac.log`: "Test run with 12 tests in 2 suites failed after 0.291 seconds
with 2 issues." Not mutated: the empty-scissor check, the view-layer filter and the group-node
rule. Passing tests assert them, and no defect was injected to see them fail.


## Not implemented

- `Renderer.setRenderObjectFunction`, with the reasons above.
- `onBeforeShadow`, `onAfterShadow`, `scene.onBeforeRender`, `scene.onAfterRender` and
  `material.onBeforeRender`.
- Inspector events for compute, copies, node inspection and frame begin and finish.
- Hooks in the toon outline after-scene pass, the motion vector pass and depth-only captures.
- A benchmark of the no-hook overhead.
