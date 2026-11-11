# Authored sprite lens flares

`SpriteLensFlare` adds a set of textured flare elements to a linear
scene frame. Give each element a straight-alpha color texture, square
pixel size, distance along the light-to-center axis and linear RGB tint.
Distances 0, 0.5 and 1 place elements on the light, at frame center and
at the mirrored light position. Sprite UVs are top-left and the Metal
format controls sRGB decoding. Each element contributes its RGB times
alpha, tint, light color and intensity, and visibility. The composite
retains the scene alpha.
[Three.js r187 Lensflare](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/objects/Lensflare.js) ·
[Three.js r187 LensflareMesh](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/objects/LensflareMesh.js)

```swift
let flare = try SpriteLensFlare(device: queue.device, elements: [
    .init(texture: core, size: 120),
    .init(texture: ghost, size: 44, distance: 0.75, color: SIMD3(0.8, 1, 0.5))
])
flare.light = try LensFlare.Light(
    worldPosition: lampPosition, camera: camera, cameraTransform: cameraPose,
    aspectRatio: Float(width) / Float(height)
)
let chain = try PostProcessChain(queue: queue, effects: [.pass(flare)])
```

Pass matching scene depth to the chain. The sprite effect shares
`LensFlare`'s GPU depth visibility test. A finite source is visible
where stored depth reaches its projection. A source beyond the far
plane is visible through cleared depth. Out-of-frame depth taps count
as hidden. Each visible fraction scales all the light's elements,
including the ones that appear far from the light. A light behind the
camera projects to nil and the stage copies its input.

The effect supports up to 16 elements for one projected light and
forward or reversed projection depth. It samples authored sprite mips
at the pixel size of the stage. Keep the textures read-only until the
render completes and use a distinct color destination. Place the pass
before tone mapping when its highlights should contribute in HDR.

Metal readback checks axis placement, square pixel size, UV orientation,
sRGB decoding, authored mips, HDR overlap and unchanged scene alpha.
Geometry rendered in front of, behind and across a projected light
produces zero, full and half visibility in forward and reversed depth.
All 22 focused tests passed on macOS and iPhone 18 Pro Simulator with
Metal API Validation enabled.
