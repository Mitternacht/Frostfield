# Stored viewport textures in surface graphs

Raster surface graphs can sample stored render color and depth using
`sceneColor(at:)` and `sceneDepth(at:)`. Set a `SceneViewportSnapshot` on
the consuming `SceneGraph` or standalone `MeshScene`. Color is premultiplied
linear RGBA16Float with bilinear sampling of mip zero by default. The
`sceneColor(at:mipLevel:)` overload supports trilinear explicit LOD in
surface and post-process graphs. Depth uses the nearest stored Depth32Float
value. Both sample top-left normalized UVs and clamp at
the texture edges. Use `screenUV` for a capture of the current viewport or
`screenCoordinate / screenSize` for a capture of the full attachment.

```swift
let frame = try sceneCapture.encode(
    backdropScene, width: width, height: height, commandBuffer: command
)
var composite = scene
composite.viewportSnapshot = try SceneViewportSnapshot(color: frame.color, depth: frame.depth)
try composite.encode(pass: compositePass, commandBuffer: command)
```

The producer can draw fewer layers or render at a reduced resolution. Match
its camera and normalized window to the receiving surface. End its encoder
before drawing the surface. Snapshots use tracked, single-sample textures
with storage separate from every active render and resolve attachment.
The caller can keep the snapshot for more than one pass on the same queue.
Archives preserve the graph but the caller supplies new snapshot textures
after loading. Materials that only read one input can omit the other.

Set `source: .opaquePass` on a snapshot to fill its textures during a
`SceneGraph.encode` color pass. It draws opaque meshes, cutouts and opaque
primitives, stores the attachments and blits them before transparent and
transmissive draws. Mipmapped capture color is generated from that opaque
result. Capture uses matched single-sample base-level attachments, with
the full attachment extent. A material that reads this snapshot must
draw after opaque. The capture color can also serve as the transmission
backdrop, which avoids a second copy and mip generation.

Use caller-supplied snapshots when a material needs a different producer,
such as a separately rendered scene or a pass that includes some transparent
geometry. Archives version 17 preserve explicit scene-color mip nodes.
Camera-dependent coverage can cut out the raster surface. An active shadow
pass rejects a caster whose coverage reads the main camera's viewport.
The path tracer rejects viewport-dependent material graphs. Raster
sampling uses a Metal argument buffer for color and depth, including on
devices with the smaller direct texture-binding table.

Metal readback verifies an actual half-resolution scene capture with forward
and reversed depth, distinct bilinear and nearest samples, and independence
from a receiving pass's color and depth. Automatic capture fixtures verify
opaque color and depth in both modes, with and without shared transmission,
and read fractional mip blends. Archive round-trip restores the graph and
leaves capture textures for the caller to attach. Validation tests preserve
the destination on missing inputs, attachment feedback and incompatible
capture dimensions. Separate coverage fixtures keep viewport-independent
shadow casting and reject a viewport read in a light-pass cutout.
Mac and iPad Pro (M5) Simulator each passed all 51 focused tests in eight
suites with Metal API Validation. The Xcode workspace bridge timed out
resolving the destination, so verification used headless Xcode.

[Three.js r187 viewport texture](https://github.com/mrdoob/three.js/blob/r187/src/nodes/display/ViewportTextureNode.js) ·
[Three.js r187 depth texture](https://github.com/mrdoob/three.js/blob/r187/src/nodes/display/ViewportDepthTextureNode.js) ·
[Metal feature set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
