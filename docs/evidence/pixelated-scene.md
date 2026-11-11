# Reduced-resolution scene pixelation

Create `PixelatedScenePass(queue:settings:)` and call `encode` with a scene
and destination. The pass uses the scene's camera and render settings by
default. Pass a `ShaderGraphCamera` for an orthographic or off-axis view.
The pass supports forward and reversed projection depth. Its returned
frame exposes the captured low-resolution color and depth for use before
the next encode.

The normalized viewport starts at the top left. Its edges round to output
pixels, and the scene camera uses the resulting viewport aspect ratio.
The pass renders one scene texel for each requested integer pixel block,
including any partial block at the right or bottom. Presentation copies
the nearest texel across each block and preserves destination pixels
outside the viewport. The reference sizes its render target by rounding
down and stretches that texture across the view with nearest filtering.
[Three.js r187 RenderPixelatedPass](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/postprocessing/RenderPixelatedPass.js)

Depth outlines use the stored beauty depth. Normal creases use a separate
native surface prepass when enabled, with its depth checked against the
beauty depth. The prepass supplies normals for opaque and masked meshes.
The four-neighbor edge resolve runs on the reduced grid and changes HDR RGB
while preserving the sampled alpha.
[Three.js r187 PixelationPassNode](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/display/PixelationPassNode.js)

The GPU tests exercise real rendered silhouettes in a subviewport, clipped
final blocks, one-texel capture, reuse, camera aspect ratio, forward and
reversed depth, and independently adjustable depth and normal edges.

All 19 focused scene pixelation, edge, scene capture and graph preset tests
passed on macOS and the Polish iPhone 18 Pro Simulator with Metal API
validation. Xcode MCP reported a full external DerivedData volume, so the
run reused the internal Xcode command-line build caches.
