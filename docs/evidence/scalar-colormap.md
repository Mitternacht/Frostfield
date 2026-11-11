# Scalar color lookup tables

`ScalarColorLookupTable` maps a single scalar to linear RGB. Its rainbow,
cool-to-warm, blackbody and grayscale palettes use the stops and linear
RGB interpretation from Three.js r187. Add a custom ramp with strictly
increasing normalized stops from 0 to 1. Stops can use linear float RGB
or packed linear `0xRRGGBB` values. The RGB-to-RGB `ColorLookupTable`
remains an independent API for 3D color grading.

```swift
var table = try ScalarColorLookupTable(
    palette: .coolToWarm, intervalCount: 32, domain: -20...60
)
let color = table.color(at: 12)
try table.setDomain(-30...80)

let lookup = try table.makeTexture(device: device)
let legend = try table.makeTexture(device: device, layout: .legend)
try viewer.encode(source: .color(legend), destination: target, region: rect, commandBuffer: buffer)
```

An interval count of `n` stores `n + 1` colors, including both endpoints.
The scalar domain clamps the input, then selects the nearest table index.
Ties select the higher index. Infinite values select the corresponding
endpoint, and NaN returns nil. Changing the scalar domain keeps the
color samples.

Metal strips use shader-readable linear RGBA16Float with opaque alpha.
The horizontal strip puts the minimum at the left. For nearest shader
lookup, normalize and clamp the scalar into `t` from 0 to 1, compute
`i = floor(t * n + 0.5)`, and sample the texel center at
`((i + 0.5) / (n + 1), 0.5)`. The vertical legend puts the maximum at
the top and includes both endpoint rows. The r187 canvas has `n` rows
and skips its upper endpoint. Update an existing strip of the matching
layout after its previous GPU reads finish.

The scalar LUT, 3D color grading, viewer and post-process suites passed
42 tests each on macOS and the iPad Pro M5 simulator with Metal API
Validation. A Node run against Three.js revision 187 supplied reference
colors for all four palettes and custom stops. GPU pixels verify scalar
mapping through a nearest-filtered shader resource, plus both endpoint
rows and an in-place HDR legend update.

[Three.js r187 Lut](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/math/Lut.js)
