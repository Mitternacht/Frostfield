# Directional screen-space contact shadows

`ScreenSpaceShadows.encodeVisibility` marches a short view-space ray
toward a directional light through stored raster depth. Set the maximum
distance, thickness, receiver bias, sample budget and frame phase. It
supports perspective, shifted orthographic, reversed and native logarithmic
depth. The visibility target can be smaller than the source attachment.
[Three.js r187 screen-space shadow node](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/display/SSSNode.js)

For stock and generated PBR, prepare
`ScreenSpaceShadows.DirectionalVisibility(texture:directionToLight:)`
from that mask. Pass it as `screenSpaceDirectionalShadow` to the normal
scene color encode. The mask must correspond to the full attachment view.
Opaque and masked draws with a matching directional light multiply its
existing shadow visibility. Ambient occlusion has its own input, and
ambient, emission and punctual lighting keep their own contributions.
`encodeComposite` can also attenuate an explicit directional radiance pass.

All 64 focused tests passed on Mac and iPhone 18 Pro Simulator with Metal
API Validation. The fixtures compare a rendered caster's shadow to an
independent ambient-only render, combine the mask with a directional
shadow map and GTAO, and check wrong-direction and all-visible results.
The native package bridge timed out resolving the destination, so the
tests used `xcodebuild`.
