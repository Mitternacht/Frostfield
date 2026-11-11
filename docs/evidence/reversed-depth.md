# Reversed depth for perspective raster passes

Prepare each mesh with `depthMode: .reversed`, clear the caller-owned
depth attachment to 0, and pass `depthMode: .reversed` to the graph's
color and surface-attribute encoders. A standalone mesh uses the
mode it was prepared for. Forward depth remains the default, with
a depth clear of 1.

`PerspectiveCamera.projection(aspectRatio:depthMode:)` and the matching
pixel-window overload map near to 1 and far to 0 when reversed. The
r187 perspective matrix uses the same two depth coefficients.
Its WebGPU pipeline mirrors material depth comparisons and bias in
reversed mode. Frostfield reuses its existing mirrored Metal states.
[r187 projection](https://github.com/mrdoob/three.js/blob/r187/src/math/Matrix4.js#L1140) ·
[r187 WebGPU depth states](https://github.com/mrdoob/three.js/blob/r187/src/renderers/webgpu/utils/WebGPUPipelineUtils.js#L949)

The graph projection also supplies camera inputs for surface graphs and
camera-relative clip assembly when `SceneGraph.highPrecision` is enabled.
The jitter shifts clip XY while preserving depth. Existing explicit-view
overloads accept caller-authored reversed projections.

Named surface targets, GPU timings and visibility queries accept the
same depth mode and use the graph's camera. Timing and visibility queries
also offer explicit-view overloads.

The SwiftUI graph initializer accepts the depth mode, clears the matching
depth attachment and redraws when the mode changes. A standalone
SwiftUI scene gets the mode from its prepared renderer. A graph can
switch modes on the same view by supplying meshes prepared for the
new mode.

Both `ParticleEmitter` and `ShaderParticleSystem` accept a depth mode.
Supply the matching projection when drawing a `ParticleEmitter`.
`ShaderParticleSystem` derives its projection from the chosen mode.
Integrated Metal fixtures checked soft quads at both depth endpoints
and composited surface particles over loaded mesh depth in both modes.

## Verification

The Metal fixture stores and reads the private Depth32Float attachment.
Two standalone draws share color and depth across passes. The nearer
surface survives in both modes, and its surface graph displays the
same projection coefficient used for rasterization. A reversed graph
matches its near-origin pixels and depth after relocation to 2⁶⁰ world
units. It checks point occlusion, clipped siblings, an unchanged
central depth sample under jitter, and the surface-attribute pass's
stored depth and clear value.

Xcode 27.2 ran 73 tests in 13 focused suites on both the M4 Max Mac
and the iPad Pro 13-inch (M5) iOS 27.2 Simulator, with Metal API
Validation. The Xcode MCP package open timed out resolving the
initial run destination, so verification used `xcodebuild`.

The owned wrapper follow-up passed 31 focused test declarations on Mac
and iPad Simulator with Metal API Validation. The iPad run skipped the
explicit stage-counter case and exercised the timing capability error.
On Mac, the timed render produced native timestamps in both modes.
The visibility fixture confirms occlusion and batch visibility in both
modes. The named-target fixture reads color, mapped normal, roughness,
and the expected Depth32Float value before and after changing a mask.

On the iPhone 18 Pro Simulator, the SwiftUI HDR graph probe switched
reversed → forward → reversed with matching prepared meshes. It kept the
same RGBA16Float layer and 480 × 288 drawable in a 160 × 96-point view.
The near cube covered the rear cube in both modes. Returning to reversed
depth reproduced every original pixel in the view. The Xcode workspace
build could not allocate DerivedData on the configured external volume.
The probe was built with internal `xcodebuild` storage and inspected
with the Xcode device-interaction session.
