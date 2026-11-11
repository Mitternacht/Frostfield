# Screen-space reflections

`ScreenSpaceReflections` traces reflection rays from the packed normal and
reflectance written by `SceneGraph.encodeSurfaceAttributes`. Supply the HDR
beauty and depth from the same perspective view. The result keeps beauty alpha.
With a supplied environment cube, a screen hit replaces the existing image
reflection according to confidence and material reflectance.

The default uses a full-size sharp trace. `Settings.resolutionScale` reduces
the Metal trace attachment in both dimensions. The full-size resolve reads
depth, normal and roughness to keep the samples on the receiver's surface.
`Settings.roughnessBlurRadius` adds a receiver-dependent spatial footprint,
in full-size pixels. Squared perceptual roughness scales that radius. Set
`maximumRoughness` to choose how much of the rough range retains a reflection.

An optional `receiverMask` takes a shader-readable R8Unorm, R16Float or R32Float
texture matching the view. Values from zero to one scale the reflection.
The mask applies to the receiving surface. A masked scene object can
still appear in another surface's reflection.

To use a previous reflected frame for another bounce, supply its composited
radiance and depth with `PreviousFrame`. A prior beauty texture also works as
a radiance source. `.registered` requires identical view and projection matrices
and the same depth convention. `.cameraMotion` instead reprojects the current
ray hit into the supplied prior camera, including any prior lens shift or jitter.
The two perspective cameras may use different forward or reversed conventions.
Use the standard Metal depth range. Each previous bilinear tap contributes only
when its depth matches the reprojected point. Rejected taps and offscreen hits
use current scene radiance. This follows the camera through geometry that stays
in the same world space. Motion between indistinguishable depth layers cannot
be resolved from depth alone. Order previous-frame producers before the SSR
encode on the GPU and use a separate output.

Set `Settings.stochasticSeed` to trace one deterministic bounded GGX
visible-normal ray per trace pixel. Zero roughness remains a perfect mirror.
Change the seed between frames for an external accumulation pass. Frostfield
uses the packed split-sum surface reflectance for the hit's energy and the
original environment term. The r187 per-direction F0/F90 weighting and
environment-miss importance sampling remain separate.

[Three.js r187 SSRNode](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/display/SSRNode.js) ·
[Three.js r187 SSRPass](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/postprocessing/SSRPass.js) ·
[Three.js r187 specular helpers](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/utils/SpecularHelpers.js)

Focused Metal validation passed 91 Swift Testing declarations on macOS and
iPhone 18 Pro Simulator. The mirror scene verifies a 128×128 trace resolved
to 256×256, receiver-mask exclusion, and exact beauty alpha in forward and
reversed depth. The rough floor fixture checks that the reflection spreads
across a sharp boundary while keeping the center and original alpha. A
rendered prior frame changes the reflected box color where depth agrees,
and rejects a previous blocker. Registered mode rejects a displaced camera.
Camera reprojection follows a translated and rotated prior camera with a
jittered off-axis lens, rejects the old blocker, and handles both directions
of a forward/reversed depth change. Seeded tracing
repeats exactly for the same seed, changes with a new seed, and spreads
rough reflections beyond the mirror footprint while leaving perfect mirrors
sharp. The Xcode bridge timed out resolving the package destination, so
verification used `xcodebuild`.
