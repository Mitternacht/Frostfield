# Texture copies and pixel readback

Use `TextureTransfer.read` to await the stored bytes of one render attachment
or texture region. The result includes the Metal pixel format, extent, row
pitch and 2D image pitch. Reads keep channel order, integer and floating-point
bit patterns, and compressed blocks. They use Metal's top-left texture
coordinates and apply no color transform.

```swift
let pixels = try await TextureTransfer.read(
    from: colorAttachment, mipLevel: 1,
    region: MTLRegionMake2D(4, 8, 32, 16), commandQueue: queue
)

try TextureTransfer.encodeCopy(
    from: colorAttachment, sourceMipLevel: 1,
    sourceRegion: MTLRegionMake2D(4, 8, 32, 16),
    to: destination, destinationOrigin: MTLOrigin(x: 16, y: 0, z: 0),
    commandBuffer: commandBuffer
)
```

The caller owns copy submission. End any preceding encoder before encoding
the copy. A read submits its own retaining command buffer on the supplied
queue and returns after successful GPU completion. Commit earlier writes on
that queue before reading. Synchronize producers on other queues separately.
Render attachments need a store action and initialized contents in the
selected region. Cancellation preserves resources until GPU completion.

The copy selects one mip and slice on each texture, or a Z range for 3D
textures. Arrays and cube faces use Metal slice indices. Copies require
matching pixel formats and dimensionality. Compressed regions use block
alignment, with partial blocks allowed at mip edges. The API rejects
overlapping copies and different views of the same allocation. It accepts
tracked, single-sample textures in persistent storage and caps each read at
256 MiB. Combined depth/stencil, PVRTC, sparse textures and lower simulator
ASTC mips are outside this slice.
[Three.js r187 renderer reference](https://github.com/mrdoob/three.js/blob/r187/src/renderers/WebGLRenderer.js)
[Apple texture-copy rules](https://developer.apple.com/documentation/metal/mtlblitcommandencoder/copy(from:sourceslice:sourcelevel:sourceorigin:sourcesize:to:destinationslice:destinationlevel:destinationorigin:))
[Apple texture-to-buffer rules](https://developer.apple.com/documentation/metal/mtlblitcommandencoder/copy(from:sourceslice:sourcelevel:sourceorigin:sourcesize:to:destinationoffset:destinationbytesperrow:destinationbytesperimage:))

## Verification

The GPU fixtures render different colors to mips 0 and 1, then read a
nonzero-origin region of mip 1 after the queued producer. An integer copy
selects a subregion of source mip 1 and writes it into destination mip 2,
leaving surrounding texels intact. Other fixtures compare bytes from an array
layer, cube face, cube-array face, 1D texture, 3D subvolume, rendered Float32
depth and EAC blocks. Invalid operations are rejected before encoding, then
a valid copy on the same command buffer succeeds.

`TextureTransferTests`, `SceneArchiveTests` and `KTX2ExporterTests` passed
with Metal API validation on the M4 Max Mac (24 tests), iPhone 18 Pro
Simulator (25), and iPad Pro 13-inch (M5) Simulator (25). Both simulators
returned the distinct rendered mip 1 bytes. The cube-array case ran on
Mac and requires Apple4 or Mac2 GPU support. The simulators exercised
2D arrays and individual cube faces.
[Apple Metal feature set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

Xcode 27.2 MCP timed out resolving the package run destination. These
runs used `xcodebuild` with isolated DerivedData.

## Partial CPU uploads

Use `TextureTransfer.encodeUpload` to patch a rectangle, mip, array layer,
cube face, or volume region from CPU data. Pack bytes tightly in the
destination's native texel or block order. A nil region updates the whole
selected mip and slice. You can encode several edits on the same retaining
command buffer before its first consumer. Earlier bytes are copied into
Metal staging buffers before the caller can reuse the source data.

The API preserves other texels, levels, and slices and uses explicit Metal
queue ordering. There is no color conversion or implicit mip generation.
For changes already on the GPU, use `TextureTransfer.encodeCopy`.
[Three.js texture update ranges](https://github.com/mrdoob/three.js/blob/r187/src/textures/Texture.js) ·
[Three.js array layer updates](https://github.com/mrdoob/three.js/blob/r187/src/textures/DataArrayTexture.js) ·
[Metal buffer-to-texture copies](https://developer.apple.com/documentation/metal/mtlblitcommandencoder/copy(from:sourceoffset:sourcebytesperrow:sourcebytesperimage:sourcesize:to:destinationslice:destinationlevel:destinationorigin:))

The fixture patches one mip and layer of a private texture, changes the
source data before submission, and reads all the affected and neighboring
bytes back. It rejects a short upload and reuses the same command buffer.
Additional fixtures check 1D, cube and volume uploads and stored EAC block
replacement. The four focused suites passed 24 tests on Mac and 21 tests on
iPhone 18 Pro Simulator, with three compressed-format capability skips on
the simulator. Xcode ran with Metal API Validation.
