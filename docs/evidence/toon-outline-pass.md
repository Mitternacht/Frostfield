# Toon outline pass

Pass a `ToonOutlinePass` to either ordinary `SceneGraph.encode` view to render a back-facing hull before each visible mesh whose effective material is `ClassicMaterial.Model.toon`. The hull uses the same vertex inputs, index range, clip transform, winding and scene attachments as the toon draw. It writes depth using the selected forward or reversed mapping and blends premultiplied linear color source-over. The scene color target, or its MSAA resolve, can feed the usual post-processing chain.

```swift
let outline = try ToonOutlinePass(thickness: 0.012, color: SIMD3(0.025, 0.012, 0.008), alpha: 0.8)
let statistics = try graph.encode(pass: scenePass, commandBuffer: commandBuffer, toonOutline: outline)
```

The pass follows r187's projected inverted-hull construction. It multiplies the normal by the same clip transform as a direction, normalizes the full four-component difference, then scales the displacement by thickness and the vertex's clip W. Its defaults are thickness 0.003, black and alpha 1. Zero thickness or alpha bypasses the hull. Frostfield exposes values set per encode and blends translucent hulls over the color present when the draw occurs. Draw a backdrop first to see it through the hull. The r187 helper accepts TSL nodes and uses an opaque outline material by default. [Three.js r187 ToonOutlinePassNode](https://github.com/mrdoob/three.js/blob/r187/src/nodes/display/ToonOutlinePassNode.js), [Three.js r187 NodeMaterial opacity handling](https://github.com/mrdoob/three.js/blob/r187/src/materials/nodes/NodeMaterial.js#L865-L871)

The default mode targets classic toon on indexed triangles, including direct batched draws, instancing, skinning and morphs. Programmable TSL outline inputs and toon node-material position overrides remain outside this entry point. Classic toon fragments are incompatible with Metal indirect command buffers.

## OutlineEffect mode

`ToonOutlinePass.outlineEffect` draws the same hulls after ordinary scene
color. It defaults to black and a clip-space thickness of 0.003. Nil alpha
uses classic material opacity, live or authored PBR base alpha, or the
draw tint alpha for the stock unlit, textured and shadowed Lambert
materials. A node override selects thickness, color, alpha, or visibility
by UUID. A grouped mesh can use a separate override for each material
child. The scene graph and source materials stay intact.
[Three.js r187 OutlineEffect](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/effects/OutlineEffect.js)

Eligible nodes use indexed meshes with the stock vertex inputs and an
enabled depth test. Color-output materials can use standard, posed,
deformed, instanced and batched vertices. Raw Metal vertex programs,
texture helpers, depth and distance output, and shadow catchers are
excluded. The final hulls use the scene's ordering policy with their own
opaque and translucent phases. Hulls replay the existing direct batch
runs while the ordinary color draw can keep indirect command buffers.
Use a single color attachment and ordinary forward or reversed depth.
The settings belong to the encode call. Per-material clipping and fog
remain outside this mode.

## Verification

- The pinned r187 WebGPU-coordinate matrix oracle predicts right edges at 86.4, 90.4 and 94.4 pixels on a 128-pixel orthographic view for thickness 0, 0.0625 and 0.125. Metal samples cover the corresponding edge pixels in forward and reversed depth.
- Metal readback checks ordinary toon front color, a neighboring basic material, a nearer occluder in both graph orders, premultiplied alpha over a transparent clear, source-over against a drawn opaque backdrop, and bypass with zero thickness or alpha.
- The perspective fixture compares a skinned and morphed instanced hull with the equivalent CPU-posed mesh. Direct batches with two different index ranges and a mirrored transform match individually drawn reference meshes.
- Xcode 27.2 ran 58 focused tests in six suites on macOS and iPhone 18 Pro Simulator with Metal API and shader validation enabled. The Xcode MCP workspace call timed out while resolving the package destination, so these runs used headless `xcodebuild`.
- OutlineEffect mode passed all 64 focused Mac tests with Metal validation. The iPhone 18 Pro Simulator passed 59 and skipped five hardware ICB tests. Both rendered classic and live PBR opacity, node overrides, a late backdrop, and overlapping translucent hulls in forward and reversed depth. The split render matched the ordinary final image, and the opaque snapshot preserved its pre-outline color. On Mac, an ICB color batch with direct hulls matched the direct-only batch pixel for pixel. The Xcode bridge's configured DerivedData volume was full, so these runs used headless `xcodebuild` with internal DerivedData.
