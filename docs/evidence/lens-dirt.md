# Lens dirt

The pinned [Three.js r187 lens dirt node](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/display/lensDirt.js)
multiplies the bloom RGB by the dirt RGB and an intensity. It center-crops the dirt
texture to cover the frame. The [official example](https://github.com/mrdoob/three.js/blob/r187/examples/webgpu_postprocessing_lensflare.html)
adds the scene, the original bloom and the dirt-modulated bloom.

Frostfield's `MipmapBloom.encodeGlow` writes the strength-weighted bloom field to
a shader-readable RGBA16Float target. Bind that target as `"Bloom"` and the dirt
image as `"Dirt"` in a `PostProcessPass` using `PostProcessGraph.lensDirt(dirtSize:intensity:)`.
Use the dirt texture's pixel dimensions for `dirtSize`. Pass the original scene
as the post-process source. The bloom and composite can encode in sequence on
one command buffer.

The graph adds both optical contributions in linear HDR, multiplied by scene
coverage once. It preserves scene alpha and keeps transparent pixels black.
The dirt image's sampled RGB controls the effect and `"Intensity"` can change
between frames. A zero intensity leaves the coverage-weighted plain bloom.

The GPU fixtures compare portrait and landscape outputs to a CPU reference
with bilinear sampling, unequal source sizes and partially transparent pixels.
They also generate glow and composite it in one command, then compare that
field to the contribution from the ordinary additive bloom pass. Existing bloom,
lens flare and post-process graph fixtures run in the same focused check.

The independent CPU comparison uses a per-channel bound of 0.005 or two
RGBA16Float output steps, whichever is larger. Alpha and transparent black
match exactly. All 23 focused tests passed on macOS and iPhone 18 Pro Simulator
with Metal API validation. Xcode MCP reported a full external DerivedData
volume, so Xcode command-line tests used isolated internal DerivedData.
