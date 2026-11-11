# Planar refraction

`PlanarRefraction` captures geometry behind a normalized world-space plane using the source
camera. The capture retains the source view and projection's XY and W rows. It substitutes a
Metal oblique depth row to reject the foreground. The positive plane side faces the camera.
`clipBias` optionally moves the clipping boundary farther into the negative half-space in
world units. The capture clears color and depth and returns nil when the camera is on the
plane's back side or the kept half-space misses the view.

The caller owns tracked RGBA16Float and Depth32Float 2D targets, the retained command buffer,
and capture ordering. The refractor UUID excludes its subtree from the capture. Additional
subtrees can be excluded explicitly. Storage aliases with other resources sampled in the
pass are rejected. The shared planar capture encoder supports optional MSAA resolve,
backgrounds, depth storage, shadow inputs, and color mip generation. A background uses the
original camera and its unclipped projection. Capture encoding is explicit, so rendering a
refractor surface does not trigger another capture.

```swift
let refraction = try PlanarRefraction(plane: windowPlane, refractor: windowNodeID)
try refraction.encode(
    scene, camera: sourceCamera, color: refractionColor, depth: refractionDepth,
    depthMode: depthMode, commandBuffer: commandBuffer
)

let windowSurface = try SurfaceShaderGraph { surface in
    let capture = surface.texture("Refraction", addressing: .clampToEdge)
    let dudv = surface.texture("DuDv")
    let offset = (dudv.sample(surface.uv0).xy - surface.constant(SIMD2<Float>(repeating: 0.5)))
        * 0.02
    return .init(baseColor: capture.samplePlanarRefraction(distortion: offset).rgb)
}
```

Bind the completed capture and the DuDv texture under the graph's names. For an undistorted
surface, call `samplePlanarRefraction()`. Its default coordinates are top-left viewport UVs
from the main render, so the two passes must use matching viewport coverage. A graph can
compose the sample with `blendOverlay` for the tint in Three.js's default Refractor shader.
The distortion option corresponds to the custom WaterRefractionShader's UV offset.
[Three.js r187 Refractor](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/objects/Refractor.js),
[Three.js r187 WaterRefractionShader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/shaders/WaterRefractionShader.js)

The capture accepts finite rigid perspective and orthographic camera views with forward or
reversed projection depth. The stored depth belongs to the oblique capture projection
returned by `refractedView`. Distortion is authored in the surface graph and samples
the available pixels from that source view.

The Metal fixtures use a tilted refractor in a transformed scene with an off-center camera.
They verify foreground clipping, subtree exclusions, projection alignment and a known
distortion offset for both depth mappings and both camera projections. They also verify
shared-storage rejection, world-unit clip bias, and inactive-target clearing.
