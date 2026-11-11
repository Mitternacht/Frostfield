# Material stencil state

`MeshRenderState.Stencil` configures 8-bit reference and read/write masks,
comparison, stencil failure, depth failure and pass operations. Its default
comparison is always and its default operations preserve stencil. Set the
state to nil to disable tests and writes. Frostfield sets the native
reference value per draw and applies the same state to both faces.
[Three.js Material r187](https://github.com/mrdoob/three.js/blob/r187/src/materials/Material.js),
[Metal stencil descriptor](https://developer.apple.com/documentation/metal/mtlstencildescriptor)

Bind one `MeshRenderer.stencilDepthPixelFormat` texture in both
`MTLRenderPassDescriptor` depth and stencil slots. The color and
depth-stencil attachments use equal-sized 2D textures at mip zero with
matching sample counts. Clear stencil to an authored byte or load a
previous pass's stored stencil. The render-state initializer selects the
layout automatically when stencil is enabled. Prepare an indexed mesh
with `usesStencilAttachment: true` when it shares this pass without
testing or updating stencil.

A multisampled pass can resolve color to a separate single-sample
texture of the same size and format. It supports both
`.multisampleResolve` and `.storeAndMultisampleResolve`. Depth and
stencil stay at their original sample count. Attachment textures use
backed storage.
[Metal color resolve](https://developer.apple.com/documentation/metal/mtlrenderpassattachmentdescriptor/resolvetexture)

Indexed standalone and scene graph color draws use stencil with forward
or reversed depth, including skin, morph and dynamic geometry. Graph
nodes can use an ordinary `InstanceSet`. Dynamic draws keep the same
per-encode geometry revision as ordinary color rendering. Scene order
determines which draws produce and consume stencil values. The combined
layout excludes batched and GPU-indirect draws, primitives, and MRT.

CPU and Metal curve flows retain the vertex revision chosen when a draw
is encoded. Typed surface graph position offsets run before the stencil
color pass and lease the prepared vertices for that command. Draws that
share a displaced renderer in one scene pass share that result. Each
draw still uses its authored range of the original index buffer.
[Shader graph vertex revisions](mesh-deformation-order.md)
Shadow, surface attribute, motion, selection, transmission, and viewport
capture passes do not use this layout. Material overrides and the toon
outline share their original non-stencil pipeline layout. Selection
preserves its previous mask if a stencil draw is rejected. World-space
picking uses the underlying mesh geometry. Path tracing rejects active
raster stencil state. Scene archive version 21 stores
the stencil state and the opted-in layout, including for meshes that share
the pass with stencil tests disabled. Earlier scene files keep their
original depth layout.

## Verification

Five focused suites passed 49 tests each on arm64 macOS and iPhone 18 Pro
Simulator with Metal validation enabled. The graph fixture reads every
color texel after distinct stencil-fail, depth-fail and pass operations
in forward and reversed depth. It repeats the draw after archive
restoration. Other fixtures cover standalone stencil store/load and
comparison direction, unchanged color after rejecting incompatible
draws, selection mask retention, and malformed archive operations.

The Xcode MCP could not create DerivedData on the full external volume.
The focused runs used `xcodebuild` with internal caches.

The indexed geometry follow-up passed 50 focused tests each on arm64
macOS and iPhone 18 Pro Simulator. The posed fixture distinguishes
morph-before-skinning with a scaled joint, verifies two instance
silhouettes after archive load, and checks a standalone posed draw.
The live indexed fixture holds the old draw in flight while an update
moves its vertices and grows its triangle count. Both command buffers
produce the expected stencil silhouettes. Direct batch construction
still rejects stencil state.

The MSAA follow-up passed 58 focused tests each on arm64 macOS and iPhone
18 Pro Simulator. A stencil-writing silhouette crosses the middle of a
pixel with four samples. A second full-frame draw uses the stencil test,
and the resolved HDR color matches exactly the coverage computed from
Metal's sample positions. The test exercises both color resolve actions.
Invalid sizes, targets, missing resolves and a stencil resolve are
rejected before changing the destination.

The deformation follow-up passed 97 focused tests each on arm64 macOS
and iPhone 18 Pro Simulator. The curve fixture tests both CPU and
Metal placement updates and submits the newer command first. The graph
fixture submits a repeated uniform revision before its first command
and checks all three frames. Both fixtures mix a fixed writer with
deformed indexed draws, exercise a nonzero index range, and verify
standalone stencil writes.
