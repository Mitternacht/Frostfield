# Scene transitions

`SceneTransitionPass` blends two resolved RGBA16Float frames of
equal size. Render each scene with its own camera into a separate
caller-owned frame, then encode the transition. It interpolates
premultiplied linear RGBA, including HDR values and alpha. The
same object runs as `PostProcessChain.Effect.pass`, with the
previous stage's output as its first frame. A chain can apply tone
mapping or other effects after the transition.
[Three.js r187 RenderTransitionPass](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/postprocessing/RenderTransitionPass.js)

Progress runs from the first frame at zero to the second frame at
one. The r187 shader increases from its second texture toward its
first. Frostfield names the inputs in playback order. Reversed
playback uses one minus progress. An optional linear
texture supplies the mask's red channel in normalized coordinates.
The pass uses level zero with linear filtering and clamp-to-edge
addressing, so the mask can have a different resolution. Low values
reveal the second frame first. Mask inversion reverses that order.
Positive thresholds use the r187 feathered ramp. Zero threshold
cuts when the mask is at or below progress. Both endpoints select
one complete frame regardless of the mask.

The typed `ShaderGraphBuilder.transition` helper applies the same
blend to two sampled RGBA values and an optional scalar mask. The
graph can choose coordinates and sampling for each input. Progress
and threshold can be graph uniforms. The prepared pass validates
that settings are in range and rejects destination feedback,
including views of input storage. The caller can replace the second
frame, mask or settings before the next encode. The pass uses
retained command buffers and copies settings into each draw.
[Three.js r187 TransitionNode](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/display/TransitionNode.js)

## Verification

The focused suite passed 37 tests on macOS and iPad Simulator with
Metal API and shader validation. Readback checks HDR crossfade and
alpha, exact endpoints, the pinned feathered mask ramp, hard-cut
ties, playback reversal, mask inversion, and agreement between
the native pass and graph. An end-to-end fixture draws two
different camera views into separate textures and transitions
between them on the same command buffer before a second chain
stage. Tests also cover replacing an input between encodes, a
destination view alias, a mask that aliases the output, and
incompatible resources and settings.

The Xcode 27.2 MCP timed out resolving the package destination.
Verification used `xcodebuild` with internal DerivedData.
