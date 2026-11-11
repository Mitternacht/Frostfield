# Post-process masks and save stages

`PostProcessChain.setMask(_:stages:)` applies a binary mask to any stage range.
An effect samples the whole current frame, then the chain copies the previous
pixel back wherever the mask excludes it. This happens after every masked
effect, so later filters also sample the unchanged pixels outside the mask.
Pass `nil` for a range to clear its mask. `inverse` selects the complement.

A mask uses the red or alpha channel of a tracked, shader-readable 2D texture
at the working frame size. Values at or above the threshold are selected.
The red channel also supports R8Unorm, R16Float and R32Float. To mask with
geometry, render the selected scene into a separate `PostProcessScenePass`
with a transparent clear and use its color alpha. Encode the capture before
the chain on the same queue. Frostfield composes the mask in a full-screen pass
after each effect, including after a reduced stage is upsampled. Three.js
writes scene masks into the composer targets' stencil attachments and copies
the unmasked regions when its buffers swap.
[Three.js MaskPass r187](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/postprocessing/MaskPass.js),
[Three.js EffectComposer r187](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/postprocessing/EffectComposer.js)

`.save(to:)` blits the current full working frame into a caller-owned,
tracked RGBA16Float texture. It preserves premultiplied linear HDR and leaves
the frame available to the next stage without advancing the ping-pong
buffers. A trailing save still presents the current frame to the chain's
destination. Set the target dimensions to the capture dimensions when a
scene is captured at reduced resolution. Keep save targets separate from
the chain source and destination.
[Three.js SavePass r187](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/postprocessing/SavePass.js)

`insertEffect(_:at:)` prepares and inserts a stage before the given index.
Pass `stageCount` to append. `removeEffect(at:)` drops a stage and shifts
the following ones. Surviving stages carry their existing masks, scale,
enabled state and uniforms. A failed preparation leaves the chain unchanged.
Changes affect subsequent encodes. An already encoded frame keeps its draws
and intermediate save even when those stages have since been removed.
[Three.js EffectComposer r187](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/postprocessing/EffectComposer.js)

## Verification

Four focused suites passed 14 tests each on arm64 macOS and iPhone 18 Pro
Simulator with Metal validation enabled. The silhouette fixture renders a
scene, applies gain and a neighboring-pixel filter, and compares every
output and intermediate save texel against the CPU result for normal and
inverse masks. It then clears the masks and disables one tap. Other
fixtures cover reduced stages, linear HDR saves with sRGB presentation,
and rejection of invalid dimensions, threshold, aliases, and save scale.

The live stage edit checkpoint passed 26 focused tests on both Mac and
iPad Simulator. The GPU fixture inserts and appends before an encode, then
removes the new stages before committing the command buffer. It reads back
both outputs and verifies that the next frame uses the revised sequence.
Invalid indices, preparation errors, and the stage cap preserve the chain.

The Xcode MCP build failed because its external DerivedData volume was
full. The focused runs used `xcodebuild` with existing internal caches.
