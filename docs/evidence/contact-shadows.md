# Projected contact shadows

`ContactShadows` prepares a finite shadow layer from the nearest visible
caster above an oriented plane. The receiver lies in the pose's local XY
plane. Local positive Z points toward the casters. For a horizontal floor,
rotate the receiver around X by −π/2. Choose width, height, maximum caster
height and mask resolution when preparing the layer.

Encode a capture before drawing the receiver. The capture respects
visibility, camera layers, caster flags, explicit node exclusions and
supported cutouts. Mask opacity falls linearly with caster height and
scales by darkness. An optional separable Gaussian softens the silhouette.
`makeReceiver` provides a tinted and depth-tested transparent node for
scene composition. The receiver leaves the scene depth intact.

Encode on the layer's queue in submission order. Encode each frame for
moving casters, or reuse the stored mask until the scene changes. The
tracked RGBA16Float mask is also available to custom render passes.
Auxiliary depth-capture restrictions apply to custom caster materials.
[Three.js r187 contact shadow example](https://github.com/mrdoob/three.js/blob/r187/examples/webgl_shadow_contact.html) ·
[Drei ContactShadows](https://drei.docs.pmnd.rs/staging/contact-shadows)

The macOS Metal run passed all 22 focused tests in five suites. The
fixtures cover horizontal and tilted receivers, overlapping casters,
height fade, blur, exclusions, alpha cutouts and an opaque scene
occluder. Forward and reversed scene depth remained unchanged by the
receiver. Xcode's bridge timed out resolving the package destination,
so verification used `xcodebuild` with Metal API Validation.
