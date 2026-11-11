# On-demand raster shadow updates

Directional maps and punctual map sets accept `updateMode: .onDemand`. The
first scene use initializes depth. Caster transforms, visibility and material
edits reuse the last depth until `requestUpdate()`. A changed light projection
or punctual slice assignment refreshes its affected views so sampled depth
matches the bound view. The default mode encodes shadow depth on every use.

An on-demand map holds a read or write lease for one command buffer at a time
on its owning queue. Further uses in the same buffer can sample the layout
after its depth passes are recorded. A successful scene encoding and GPU
completion publish a new cache entry. A partially encoded update cannot
publish depth, and abandoning an unsubmitted buffer releases its lease when
the buffer is discarded. `TextureDebugPass` participates in the same lease
and accepts only slices with valid depth.

Punctual map sets accept `requestUpdate(for: lightNodeID)` for a node with a
completed or pending slot assignment. Eligible nodes retain their slots across
graph reorder. New nodes fill remaining slots in traversal order and build
depth on first use. A spot refreshes one slice, while point and rectangle
lights refresh six faces together. A failed update invalidates the affected
light's faces and preserves unrelated lights' depth.

Scene statistics count actual encoded depth passes, including clear-only
updates. Reusing a map encodes no shadow passes, draws or triangles. Tests
read projected depth and lit receiver pixels for directional, spot and point
lights. They also check requests during pending uses, discarded buffers,
incomplete updates, projection and slot changes, and the shadow texture viewer.

All 54 focused tests passed on arm64 macOS and iPad Pro 13-inch (M5) Simulator
using Xcode 27.2 and Metal API Validation. The selection included shadow timing,
cached caster geometry, texture inspection and named PBR render targets.

The selective-refresh follow-up passed 51 focused tests on both platforms
with Metal API Validation. A two-spot and point-light fixture verified stable
physical slots after reorder, byte-identical depth for untouched lights, a
one-pass spot refresh, and a six-pass point refresh. Receiver color changed
only when its light refreshed. Submitting one face of an incomplete point
update invalidated all six faces and left both spot slices available.

### Mixed punctual policies

`PunctualShadowMaps` accepts `lightUpdateModes` keyed by scene node ID.
Use `setUpdateMode(_:for:)` to change an override and pass nil to restore
the set default. An automatic set needs an initial dictionary to opt in to
cached policies, since it will then hold one command-buffer lease per map set.
An on-demand set always supports overrides.

Automatic lights render their slices on each scene encode. In the same
command buffer they can render again while manual slices reuse previously
recorded depth. Manual lights refresh when requested, on first use, or
after a projection or slot change. Changing a light to manual keeps its
last valid depth. Sets without caching keep the automatic encode path.

The mixed-policy fixture passed on Mac and iPhone 18 Pro Simulator in a
53-test selection using Metal API Validation. It renders two different
caster positions in one command buffer, verifies that the automatic point
light refreshes its six faces on each use, and checks that the manual spot
slice stays byte-identical until requested. Both set defaults were tested.
Xcode 27.2 `xcodebuild` ran after the bridge timed out resolving the destination.

Three.js r187 exposes `autoUpdate` and `needsUpdate` on the shadow renderer and
individual lights. Frostfield exposes the default on each map or set, with
per-node overrides and refresh requests for punctual lights.
[Three.js shadow update gates](https://github.com/mrdoob/three.js/blob/r187/src/renderers/webgl/WebGLShadowMap.js) · [Three.js LightShadow](https://github.com/mrdoob/three.js/blob/r187/src/lights/LightShadow.js) · [Metal command buffer completion](https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted())
