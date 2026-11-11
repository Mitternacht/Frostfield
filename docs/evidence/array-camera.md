# Array camera rendering

`ArrayCamera` holds ordered views with independent camera poses, projections
and top-left pixel viewports. A perspective view can derive its projection
from the viewport aspect ratio. A split `ShaderGraphCamera` also supports
authored orthographic and off-axis views. A view can set its own scissor
and layer mask.

This is the multi-pass form: one render pass per view. Single-pass layered
view amplification and physical XR remain separate and open.

## Encoding

`SceneGraph.encode(pass:commandBuffer:arrayCamera:)` encodes every view
through the native explicit-view path and returns draw statistics in array
order. An empty array encodes no work, and the caller's pass descriptor and
scene stay intact.

| View | Load actions | Store actions |
| --- | --- | --- |
| Only view | the caller's | the caller's, the original descriptor |
| First of several | the caller's | store, resolve target removed |
| Middle | load | store, resolve target removed |
| Last of several | load | the caller's, with its resolve targets |

Every attachment that has a texture follows this table: all color
attachments, depth and stencil. Intermediate views store depth even when the
caller's depth store action is `dontCare`; without it, a later view would test
against undefined depth. Later views load the previous color, depth and
stencil, so overlapping rectangles share depth tests and the area outside all
rectangles keeps the first view's clear or the caller's earlier contents.

Several views need one depth convention and attachments that are tracked,
storable and not aliasable (`ArrayCamera.Failure.transientAttachment`
otherwise). One view has neither requirement.

## Layers

A view with a nil mask draws what the graph's `cameraLayers` allow. A view
with a mask adds a second test: a node draws in that view only when its
layers intersect both the graph's `cameraLayers` and the view's mask. The
two tests are separate, as in Three.js r187: `WebGLRenderer.render` projects
objects, lights and shadow casters with the array camera's own layers, and
each sub-camera then tests the object's layers again in `renderScene`. A node
on layers 0 and 2 therefore draws for a graph mask of layer 0 and a view
mask of layer 2, while a node on layer 2 alone does not. Lights, shadow
casters and probe volumes follow only the graph's mask, so they are the same
in every view.

## One view

One view uses the caller's pass descriptor unchanged and none of the
multi-view checks. It accepts the same command buffers as the explicit-view
overload, including one made with unretained references, and a test compares
its pixels and statistics with an explicit encode. Several views accept
unretained command buffers too; keep the attachments alive as for any such
buffer. A command buffer that was already committed throws
`ArrayCamera.Failure.commandBuffer` for any view count instead of failing in
Metal. Shadow maps and transmission still need retained references where the
ordinary path does.

## Failures

Before encoding anything, several views check mixed depth modes, the on-demand
camera-fitted sun, every viewport and scissor, camera finiteness and
invertibility, high-precision camera data and the attachment gate. Each view
after the first then runs the scene encoder's CPU-side preparation in a
validation-only mode: node and level-of-detail validation for that view's
camera position, view-dependent culling and viewport-input checks, shadow
fits, shadow-map command-buffer checks, transmission backdrop and attachment
checks, and the soft-particle projection. That mode stops before the first
encoder, deformation lease or indirect-argument pass, and shadow update-cache
reservations are cancelled. A failure in a later view therefore throws before
anything is encoded, which tests show for a level of detail that only a distant
view selects and for a transmissive scene whose first view has an empty
scissor. The first view validates itself before it creates an encoder.

The preparation of views after the first runs twice, and a camera-fitted sun
keeps the fit of the last view checked. Failures that occur only while
creating an encoder or leasing deformation vertices, and a Metal failure
during encoding, cannot be foreseen. If an encode throws after validation,
discard the command buffer.

## Shadows and transmission

Shadow depth is not shared across views. Each view encodes its own
directional and punctual shadow passes into the shared map, in order, because
its casters follow its own level-of-detail selection and a camera-fitted sun
refits to its camera. The map holds the last view's depth afterwards, each
view's statistics report its own `shadowPassCount`, and a test checks one
pass per view for one automatically updated directional map. An on-demand map
follows its update cache and draws in the first view that needs it; no test
covers that case with several views. An automatic camera-fitted sun refits
per view; an on-demand one rejects distinct camera fits in one command
buffer. Transmission copies the whole attachment into the shared backdrop
before each view's overlay pass and samples the view's own rectangle of it. A
test renders glass in two side-by-side views against a clear color.

## Explicit-view options the array encode drops

These parameters of the explicit-view overload have no array equivalent:
`screenSpaceOcclusion`, `screenSpaceDirectionalShadow`, `indirectDraws`,
`updatingSkeletons`, `updatingBoxes`, `updatingPlanes`, `toonOutline` and
`motionVectors`. Occlusion and directional visibility masks need one mask per
view, and the rest need per-view state the array does not carry.
`depthMode`, `viewport`, `scissor`, `graphCamera`, `cameraPosition`,
`directionToCamera` and `clipFromWorld` come from each view. Shadow maps,
transmission backdrop and ordering are shared by every view.

All views render into the same attachments, slice and level of the one pass,
so views cannot target different array slices or select a render-target array
index of their own.

## Picking

`SceneGraph.pick(atPixel:in:viewIndex:)` casts a ray through one chosen view at
a top-left attachment pixel. A point outside that view's viewport or scissor
hits nothing. A hit must lie on a node that the view would draw: visible, on a
layer in both the graph's `cameraLayers` and the view's mask, with levels of
detail selected at the view's position. A bad index throws
`ArrayCamera.Failure.viewIndex`. Other views never occlude a hit, and points,
lines and sprites are not picked. Choosing among overlapping views, for example
by depth or by order, is left to the caller and remains open. The
explicit-camera pick overload still uses only the graph's layers.

## Upstream

This covers ordered viewport rendering from
[Three.js r187 ArrayCamera](https://github.com/mrdoob/three.js/blob/r187/src/cameras/ArrayCamera.js)
and its [WebGL render loop](https://github.com/mrdoob/three.js/blob/r187/src/renderers/WebGLRenderer.js).
