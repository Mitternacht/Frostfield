# Motion-guided temporal anti-aliasing

Pass the current raster motion attachment to the `motionTexture:` overload
of `TemporalAntiAliasing.encode`. It is a single-sample, shader-readable
RGBA16Float texture matching the scene color and Depth32Float textures.
RG contains previous-minus-current motion in top-left pixels including both
frames' raster jitter. B contains the previous projected window depth and A
is one for valid motion. Consecutive motion frames use the same forward or
reversed depth mode. Encode them in order on retained buffers from one queue.

The resolve checks raw previous depth at the position predicted by RG. It
removes the change in raster jitter for the accumulated color lookup. The
motion filter compares stored depths for each previous color tap and
renormalizes the surviving bilinear weights. Exact accumulation samples the
nearest eligible history texel. A new surface, invalid vector, cleared depth,
out-of-frame coordinate or mismatched prior depth uses the current color.
The default depth tolerance of 0.0005 matches the pinned TRAA threshold and
allows for half-float depth in the motion attachment. Overlapping surfaces
with indistinguishable stored depths share this depth test.
[Three.js r187 TRAANode](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/display/TRAANode.js)

Motion resolves retain a pair of full-precision depth histories on the GPU.
The first motion frame, a resize, a depth-mode change or an explicit reset
starts a new history. Omit motion to use the camera-only resolve.

The GPU tests use authored jittered vectors to verify moving foreground
history, disocclusion and invalid-vector rejection in forward and reversed
depth, mixed-depth bilinear filtering, history reset and resource validation.

All 20 focused motion TAA, camera-only TAA, FXAA and motion blur tests
passed on macOS and iPhone 18 Pro Simulator with Metal API validation.
Xcode MCP reported a full external DerivedData volume, so verification used
Xcode command-line tests with isolated internal DerivedData.

The integrated `ebff184` checkpoint passes the raster motion texture
directly to TAA. Metal readback verifies that a translated, jittered
foreground blends its previous color and that the newly revealed
background rejects foreground history in forward and reversed depth.
This passed on Mac, iPad Simulator and Vision Pro Simulator.
