# Flake normal textures

`FlakesTexture(device:seed:settings:)` creates an opaque, linear
RGBA8Unorm texture for metallic paint detail. Bind `normalTexture`
with `PhysicallyBasedMaterial.TextureBinding` as the base normal
or as the `PhysicallyBasedMaterial.Clearcoat.normalTexture`.
Material normal scale controls the strength of the detail.

The default uses the Three.js r187 layout. It draws 4,000 disks in
a 512 × 512 map with uniform independent centers, radii from 3 up to
6 texels, and normal directions from normalized `(x, y, 1.5)` with
independent uniform x and y in −1..<1. Later disks cover earlier disks.
The native rasterizer uses a one-texel distance ramp for antialiasing.
A shared SplitMix64 stream makes the base texels deterministic for a seed.
[Three.js r187 FlakesTexture](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/textures/FlakesTexture.js)
[Three.js r187 car paint material](https://github.com/mrdoob/three.js/blob/r187/examples/webgl_materials_physical_clearcoat.html)

The `.threeJS` encoding uses a neutral RGB value of 127/127/255 and
flake codes of `127 + 127x`, `127 + 127y`, and `255z`. Its blue
channel produces stronger tilt after the shader's ordinary signed RGB
decoding. Choose `.tangentSpaceRGB` to encode the unit direction in
the standard signed RGB range with 128/128/255 as neutral. The default
clips disks at the boundary. `.wrapped` continues disks across both
edges, including for textures smaller than one disk. Mips are generated
from the baked map when requested and complete before the texture returns.

Dimensions use Frostfield's 8,192-pixel side and 16,777,216-base-texel
limits. Radii are 0.5 to 64 texels. Generation allows up to 65,536 disks
within a combined 16,777,216 candidate texel budget. Zero disks produces
the selected encoding's neutral texture.

## Verification

The actual r187 JavaScript ran in Chromium with the same SplitMix64 draws.
For seed 82, its first isolated flake at texel (3, 49) encoded
RGBA 77/175/213/255. The 512 × 512 canvas covered 70.33% of texels and
averaged RGB 126.44/126.72/236.38. Native tests compare those statistics,
check exact interior and seam-crossing codes, and verify the alternative
encoding, deterministic mips, validation, and GPU sampling.

The native distance ramp covered 70.37% of texels. Both platforms matched
the browser's minimum blue code of 187 and differed by at most 0.017
in mean RGB codes. In the 64 × 64 off-axis lighting fixture, the base
normal raised 137 sampled pixels and lowered 372 relative to a flat map.
The independent clearcoat normal raised 64 and lowered 148. Each count
uses a 0.015 linear change threshold. The macOS and iPad Simulator results
agreed, including the peak increases of 47.72 and 29.18 linear.

A separate first-hit path trace centers the ray on the conventional
RGB flake. The returned signed world-space normal agrees with the
encoded texel and the plane's tangent frame. Its flat comparison returns
the geometric normal. macOS verifies software and hardware acceleration.
iPad Simulator verifies software acceleration.

The final focused run passed 20 tests on each platform. Xcode 27.2 MCP timed out resolving the package destination. Verification
used `xcodebuild` with internal DerivedData and Metal API validation.
