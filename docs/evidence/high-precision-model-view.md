# High-precision raster model-view

Set `SceneGraph.highPrecision` when the camera and scene share a large
world offset. Ordinary raster mesh, point, line, and sprite draws
subtract the camera origin at each root, compose the hierarchy and
model-view in Double, and convert the final clip matrix to Float for
Metal. Small child offsets, pivots, and world-oriented children
survive the common origin. The default render path uses the existing
Float transforms and allocates no Double hierarchy.

Camera culling tests each mesh's local bounds against its final clip
matrix. Raster level selection measures the camera-relative origin.
The world normal matrix's linear part uses a Double inverse transpose
before upload. Authored world-space shader positions and light vectors
keep their existing Float bindings. [Three.js r187 Renderer.highPrecision](https://github.com/mrdoob/three.js/blob/r187/src/renderers/common/Renderer.js) ·
[Model-view and normal-view uniforms](https://github.com/mrdoob/three.js/blob/r187/src/nodes/accessors/ModelNode.js)

Public scene queries and path tracing retain their own transform
paths. Skinning, instance placements, and batch placements still apply
their local Float data before the camera-relative model-view. Three.js
r187 documents that its high-precision option excludes instanced and
skinned meshes. Its node renderer also has a Double normal-view matrix.
The Frostfield setting is a render option to set again after loading a
scene archive.

## Explicit camera views

The explicit scene encode takes the same `clipFromWorld` and
`cameraPosition` inputs as before. Supply `graphCamera` with the split
projection and affine view when high precision is enabled. The
renderer checks the split matrices against the combined clip matrix and
checks the camera position against the view translation within Float
rounding. It uses the supplied position as the rebase origin and the
view's linear part to rotate or scale relative positions. Off-axis
projection, orthographic projection, reversed depth, and pixel jitter
use the authored settings. The position must still represent the intended
origin at its Float precision.

## Primitive paths

Points, line endpoints and sprite anchors use the same Double
camera-relative clip preparation as ordinary meshes. Primitive bounds
are transformed into camera-relative world space before culling.
World-sized shapes retain their world-unit padding, explicit node bounds
override that padding, and pixel-sized shapes cull by their anchors.
Transparent ordering uses the depth of the camera-relative bounds center.
Metal still applies the existing point size, line width, and billboard
size and rotation in each primitive shader.

The r187 points and sprite node materials use the model-view accessor
directly. Ordinary line nodes inherit it from NodeMaterial, and the
thick-line node uses it for both endpoints. Three.js limits WebGPU
point primitives to one pixel and describes a sprite path for larger
points. Its high-precision option documents limits for `InstancedMesh`
and `SkinnedMesh`. Frostfield's native point quads retain their
authored sizes, and each primitive renderer's Float anchors share the
node's camera-relative transform.
[r187 points](https://github.com/mrdoob/three.js/blob/r187/src/materials/nodes/PointsNodeMaterial.js) ·
[r187 sprites](https://github.com/mrdoob/three.js/blob/r187/src/materials/nodes/SpriteNodeMaterial.js) ·
[r187 ordinary lines](https://github.com/mrdoob/three.js/blob/r187/src/materials/nodes/LineBasicNodeMaterial.js) ·
[r187 thick lines](https://github.com/mrdoob/three.js/blob/r187/src/materials/nodes/Line2NodeMaterial.js)

## Native instance groups and batches

InstanceSet applies its existing local placement, tint, and winding
through the camera-relative draw matrix. The camera and light cull
against aggregate instance bounds. A batch also uses this camera
matrix for its direct and Metal indirect color draws and the surface
attribute pass. Camera-relative object bounds control per-object
culling, and Double depths control the built-in opaque and transparent
ordering. Custom sort callbacks continue to receive authored Float
world transforms and world bounds. This corresponds to the r187 batch
node applying the batch matrix to its local position before the
model-view. [r187 batch node](https://github.com/mrdoob/three.js/blob/r187/src/nodes/accessors/Batch.js#L126)

## Directional shadows

With high precision enabled, directional maps keep their authored light
center or fitted camera origin separate from their projections. Unskinned
mesh draws, including instance groups, subtract the origin before
projecting the local model in Double. Metal draws the caster and samples the receiver
with a precomposed model-to-light matrix. The stock receiver uses the
locally deformed vertex position. The 24-slot shadow uniform and
world-position shader bindings retain their existing layout.

Cascades snap to the world light-space texel grid using compensated
remainders of the absolute origin and Double camera-relative slice centers.
Receivers evaluate a per-draw model-local plane to choose the cascade and
blend band. On-demand maps keep their anchored projections across subtexel
refits that stay in the same grid cells. Explicit cascade fits derive the
camera origin from the supplied Float view matrix, whose translation
precision can limit large rotated poses.

This follows the r187 shadow node's model-local projection. Three.js
adds its authored world-normal offset as a separate direction in light
space and uses its world-position path for explicit receiver or context
overrides. Frostfield's directional map exposes normalized depth bias
and applies slope bias for filtered sampling. Batched and skinned
meshes retain the existing shadow coordinate path.
[r187 shadow receiver](https://github.com/mrdoob/three.js/blob/r187/src/nodes/lighting/ShadowNode.js#L450)

An on-demand map refreshes when a pass switches between coordinate
paths or changes the anchored light projection. Bias-only updates affect
receiver sampling immediately while reusing the completed caster depth.

## Punctual lighting and shadows

An ordinary high-precision draw keeps the light's root position
and small child offset separate. The caster composes its projection
about the light in Double and culls against local bounds. Stock PBR
and classic receivers use camera-relative positions for spotlight and
point attenuation, profiles, and shadow lookup. Spotlights also use
these positions for their cone and cookie. Authored normal
bias is projected separately as a direction. A point receiver selects
its cube face from the biased light-relative vector, with X, Y, then Z
precedence for equal axes. Basic, PCF, and spot variance sampling keep
their existing depth and filter rules.

The light's placement and projection identify cached depth.
Moving the viewing camera reuses completed depth. Moving a light within
its root refreshes the assigned slices even if its absolute Float
position rounds to the same value. A point light refreshes all six
faces together.

Author small offsets under a shared large root and use a camera origin
near that root. Public transforms and explicit camera positions still
accept Float coordinates. Authored world-space graph inputs retain
Float precision. Unskinned instance groups share the rebased caster
and receiver path. Batched and skinned shadow draws use the existing
world-space path.

Rectangle lighting derives its corners from the rebased center and
oriented half-axes. The view vector used for specular and clearcoat is
camera-relative. Frostfield evaluates irradiance from the solid angle
and uses the r187 LTC fit for isotropic specular and clearcoat.
See [rectangle light evidence](rectangle-ltc.md). Frostfield's optional rectangle shadow
continues to use a six-face proxy at the emitter center, so rotating
the panel can change the lighting while reusing its shadow depth.
[Three.js r187 rectangle node](https://github.com/mrdoob/three.js/blob/r187/src/nodes/lighting/RectAreaLightNode.js) ·
[Three.js r187 rectangle light](https://github.com/mrdoob/three.js/blob/r187/src/lights/RectAreaLight.js)
[Three.js r187 spotlight shadow camera](https://github.com/mrdoob/three.js/blob/r187/src/lights/SpotLightShadow.js) ·
[Receiver projection and normal bias](https://github.com/mrdoob/three.js/blob/r187/src/nodes/lighting/ShadowNode.js#L421) ·
[r187 point shadow node](https://github.com/mrdoob/three.js/blob/r187/src/nodes/lighting/PointShadowNode.js)

## Verification

At a shared origin of 2⁶⁰ world units, a Metal render preserves two
small child offsets and culls the offscreen sibling. Its pixels match
the ordinary Float render near zero. Further checks cover a rotated
camera, world-oriented pivoted children, normal color, pixel jitter,
and a level-of-detail change driven by the retained child offset.
A material graph still receives the authored world-space position.

All 85 focused tests across 12 suites passed on the M4 Max Mac and
iPad Pro M5 Simulator with Metal API Validation.

The primitive follow-up passed 94 focused tests across 12 suites on
both destinations with Metal API Validation. The 2⁶⁰ translation
fixtures compare near-origin pixels for fixed-size points, world-width
lines and a rotated sprite. They also check world-sized point padding
at the viewport edge, an authored culling bound, and back-to-front
sprite compositing at a 2⁶⁰ depth origin.

The explicit-view follow-up passed 85 focused tests across 13 suites
on both destinations with Metal API Validation. A rotated off-axis
view and a reversed-depth orthographic view match their ordinary
near-origin pixels after a 2⁶⁰ translation. The fixture includes
inherited nonuniform scale, a pivot, jitter, culling, and sorted
transparent sprites. Separate checks reject a mismatched clip column,
an inconsistent camera position, and a projective view matrix before
encoding.

The single-directional follow-up passed 59 focused tests in 12 suites
on each destination with Metal API Validation. Tests read the full
caster depth slice and receiver color. Basic Lambert, filtered
generated PBR, and variance classic receivers match their
high-precision near-origin images and depth exactly at 2⁶⁰. They also
agree with the ordinary Float frame near the origin within readback
tolerance. The on-demand test checks caster culling, mode switches,
reuse, an authored depth bias, and refitting after a center change.
Xcode MCP timed out resolving the package destination, so verification
used Xcode 27.2 through `xcodebuild`.

The cascade fixture checks two full depth slices and receiver pixels
before the first split, through both blend bands, and beyond the last
split. One tilted receiver changes split distance across its surface.
Basic and variance sampling match near-origin color and depth exactly
after a grid-aligned translation of 3 × 2⁶⁰. At an off-grid origin of
2⁶⁰, both stored slices move by the predicted −1/96 of normalized depth
while the sampled receiver colors agree. Normal-scale rotated-light
fixtures verify fixed shadow edges in both precision modes. An
on-demand perspective fit reuses depth for a 0.45-texel move and
refreshes it when the snapped grid moves.
Mac and iPhone 18 Pro Simulator each passed 61 focused tests in
12 suites with Metal API Validation. Xcode 27.2 ran the tests through
`xcodebuild` after the MCP package destination timed out.

The spotlight follow-up passed 69 focused tests in 16 suites on Mac
and iPad Pro M5 Simulator with Metal API Validation. Basic classic
and filtered or variance PBR fixtures compare exact caster depth and
receiver color at 2⁶⁰. The PBR fixture uses an off-axis perspective
projection. The variance moments match exactly. A displaced graph
caster's mask clears depth and moments and restores the lit pixel.
Other checks read the authored normal-bias response and reuse, child
movement, and selective refresh on a cached variance spot. The Xcode
MCP destination timed out, so the tests ran with `xcodebuild`.

The point-light follow-up passed 79 focused tests in 18 suites on Mac
and iPad Pro M5 Simulator with Metal API Validation. Basic and PCF
fixtures read all six cube faces and the shaded receiver at 2⁶⁰. Depth
and color match their near-origin frames exactly. A seam fixture gives
adjacent faces different depth and verifies X-first ties, biased face
selection, and six-face refresh after moving the light. A mixed
ordinary and batched receiver scene matches its normal-scale Float frame.
The Xcode MCP package destination timed out, so the tests used
`xcodebuild`.

The rectangle follow-up passed 83 focused tests in 20 suites on
Mac and iPad Pro M5 Simulator with Metal API Validation. A rolled,
tilted asymmetric emitter with a metallic and clearcoat receiver
matches its near-origin color at 2⁶⁰, including facing and roll
changes. Basic and PCF proxy depth readback matches all six
near-origin slices. A shallow blocker verifies normal bias, and the
cache fixture verifies reuse on viewing-camera motion and panel roll
and six-face refresh after a center move. Xcode MCP timed out
resolving the package destination, so the tests used `xcodebuild`.

The instance and batch follow-up passed 60 focused test declarations
in 14 suites on Mac with Metal API Validation. The iPhone 18 Pro
Simulator run passed the same selection with six capability-gated
Metal indirect tests skipped. At 2⁶⁰, the direct and indirect
batch color fixtures matched their near-origin images, including
transparent depth order across winding spans, tint, and per-object
culling. The direct surface-attribute pass matched stored pixels
with explicit node culling bounds. Directional and point maps matched
near-origin caster depth and receiver color for unskinned instance
groups. Xcode MCP timed out resolving the package destination, so the
tests used `xcodebuild`.
