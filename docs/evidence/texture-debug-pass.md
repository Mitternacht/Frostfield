# Texture and shadow map viewer

`TextureDebugPass` draws an opaque image into a caller-owned render target on
the caller's Metal command buffer. Fill the target for inspection or supply
a top-left pixel rectangle to overlay the image on a completed scene color
pass. The overlay loads the existing color and keeps the pixels outside its
rectangle. A render target can be a framebuffer-only drawable.

```swift
let viewer = try TextureDebugPass(device: device, pixelFormat: color.pixelFormat)
try viewer.encode(
    source: .directionalShadow(shadowMap, cascade: 1),
    destination: color, region: MTLRegionMake2D(12, 12, 192, 192),
    commandBuffer: commandBuffer
)
```

Use `.color` for a 2D image, 2D array layer, volume Z slice, or cube face at a
selected mip. The viewer stretches the selected image with nearest texels,
shows RGB or a grayscale channel, and writes opaque alpha. Metal decodes
sRGB source texels and encodes an sRGB destination. RGBA16Float output keeps
linear HDR values without tone mapping. Color input must support
float sampling, and source and destination must be separate, tracked,
single-sample textures.

Use `.depth` for a Depth32Float image or array layer. Use `.directionalShadow`
or `.punctualShadow` to read a native shadow slice after the graph writes it.
Directional depth is forward and punctual depth is reversed. Both show
the near endpoint in white. The values are stored depth rather than
world distances. Encode on the shadow maps' owning queue. Punctual spot
slots start at slice zero. Point and rectangle slots start after the full
spot capacity and use six slices each in +X, −X, +Y, −Y, +Z, −Z order.
Select a slice that the graph rendered in the current frame.

## Scene helpers

`DebugHelpers.texture(_:size:mipLevel:sampleCount:transform:)` prepares a
scene node with an XY plane for 2D textures, a box for cube textures, or an
XY slice stack for arrays and volumes. The cube maps all six faces in
Metal's native order with top-left image coordinates. The stack draws one
child per selected mip's layer. Children share a renderer and are blended
back to front by the scene's camera sort. Each slice uses opacity
`max(1 / layerCount, 0.25)`. The first and last layers span the authored
depth along local Z. A single layer is centered and opaque.

The source texture is bound directly when the scene encodes, so a tracked
producer can write it in an earlier encoder on the same command buffer.
Helpers transform with their parents, use the scene's depth test, and
start with shadow casting disabled. They use the raster custom material
path. Scene archives and asset exporters do not preserve the debug
material. [TextureHelper](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/helpers/TextureHelper.js)

## Shadow overlay labels

Prepare a `TextureDebugPass.Label` from a scene node's name or an authored
caption, then encode it after drawing the preview. The label caches a
one-line Core Text coverage mask, so it can be drawn repeatedly at
different positions and tints. Choose a top-left pixel origin that keeps
the label inside the destination. The blend uses linear premultiplied
color and preserves the target outside the glyphs.

```swift
// Prepare once, or again after the scene node is renamed.
let label = try TextureDebugPass.Label(lightNode.name ?? "Spot", device: device)

// Encode after drawing the shadow preview.
try viewer.encode(label: label, destination: color, origin: SIMD2(16, 16), commandBuffer: commandBuffer)
```

The caller chooses the caption for the displayed shadow slot.
[ShadowMapViewer](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/utils/ShadowMapViewer.js) ·
[Core Text line drawing](https://developer.apple.com/documentation/coretext/ctlinedraw(_:_:))

## Verification

The viewer and existing shadow and post-process suites passed 44 tests
each on macOS and the iPad Pro M5 simulator with Metal API Validation.
GPU pixels verify linear HDR, sRGB transfer and ASTC decoding, overlay
clipping, exact mip and layer selection, and near-white grayscale from
rendered directional cascades, a spot and point faces.

The scene helper, viewer, renderer and graph suites passed 50 tests each
on macOS and the iPad Pro M5 simulator with Metal API Validation. The
pixel fixtures verify nested plane rotation and a same-buffer GPU write,
texel orientation on every cube face at a selected mip, and array and
volume compositing viewed from both sides.

With label support, the same verification passed 52 tests on each
destination. Label pixels verify upright glyph orientation, antialiased
coverage, premultiplied blending over depth grayscale and preservation
of surrounding pixels.
