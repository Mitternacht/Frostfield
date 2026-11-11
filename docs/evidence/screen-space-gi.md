# Screen-space global illumination

`ScreenSpaceGlobalIllumination` estimates one diffuse bounce from a finished
linear HDR view. Each slice samples near to far on both sides of a pixel and
projects the front and back of each visible surface into 32 angular sectors.
Only newly covered sectors contribute, weighted by the receiver and emitter
cosines. RGB stores the estimated indirect radiance. Alpha stores ambient
visibility. Cleared depth returns zero radiance and full visibility.

Pass the same perspective `ShaderGraphCamera` used to render the source,
including any off-center projection or jitter. Forward and reversed depth use
the same view-space distances. Use `SceneGraph.encodeSurfaceAttributes`, a
signed named world-normal target from `SurfaceRenderTargets`, or the
closest-depth normal reconstruction. Beauty and depth are shader-readable
RGBA16Float and Depth32Float. The pass writes a separate RGBA16Float target
on the supplied retained-reference command buffer.

```swift
let gi = try ScreenSpaceGlobalIllumination(device: queue.device)
try gi.encode(
    source: beauty, depth: depth, normals: .worldNormals(normal),
    destination: indirect, camera: graphCamera, depthMode: .reversed,
    settings: .init(radius: 2, sliceCount: 4, stepCount: 12,
                    resolutionScale: 0.5, temporal: .init()),
    commandBuffer: commandBuffer
)
try gi.composite(
    source: beauty, illumination: indirect, diffuseAlbedo: albedo,
    destination: composed, commandBuffer: commandBuffer
)
```

The composite adds only the indirect RGB weighted by the supplied diffuse
albedo and source coverage. It preserves beauty alpha and leaves ambient
visibility available for a separate indirect-light composition. Sample
settings control world distance, thickness, screen radius, step distribution,
light intensity, backface contribution, and fade at the screen boundary.
The optional full-size spatial resolve rejects samples across depth jumps and
weights nearby samples by normal agreement. At a reduced resolution scale the
Metal gather renders one sample per smaller attachment texel, snaps the trace
guide to the nearest visible depth around the mapped center, and resolves to
the original dimensions using normal and tangent-plane agreement. Cleared
background stays neutral through the resolve.

The estimator follows the tagged Three.js addon and the visibility bitmask
paper. `screenSpaceSampling` derives the trace radius from image width, and
`linearThickness` multiplies thickness by the clamped sample distance divided
by camera far, times 100. Both are opt-in. The search still uses the configured
world radius to reject distant surfaces and a cap in full-size pixels.
Geometry outside the source view or behind the visible depth layer cannot
contribute.
[Three.js r187 SSGINode](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/display/SSGINode.js) ·
[Screen Space Indirect Lighting with Visibility Bitmask](https://cdrinmatane.github.io/cgspotlight-slides/ssilvb_slides.pdf)

Opt-in temporal filtering cycles the tagged slice and step phases. It
reprojects the current surface into the preceding jittered perspective view,
accepts bilinear history taps only where previous linear depth and normal
agree, and blends indirect RGB and ambient visibility. The temporal pass uses
its own per-pixel history on the same Metal queue. Use one pass instance per
view stream. Settings, depth convention, source extent, queue, or an unordered
producer invalidate history. Use `resetTemporalHistory()` at a camera cut or
an abrupt same-surface lighting change.

Pass the current view's `RasterMotionVectors` texture as `motionTexture` to
follow independently moving geometry. Supply vectors against the same
preceding frame used by SSGI history, with the default 0...1 viewport depth.
The pass uses previous-minus-current offsets in top-left pixels, including
raster jitter, and validates previous raw depth with Float16 rounding in mind.
An invalid vector, incompatible normal or occluded previous surface uses the
current trace. Omit the motion texture for camera-only reprojection. Motion
history has the precision of the stored vectors and previous depth.

Focused native checks passed on macOS and iPhone 18 Pro Simulator with Metal
validation enabled, 76 Swift Testing declarations per destination. The
emissive wall and separate floor verify red bleed at full size and with a
64×64 trace resolved to 128×128, all three normal sources, search radius,
open-floor visibility, albedo composition and the cleared silhouette. The
temporal fixture checks jittered camera motion against an independent
ray/plane reprojection, disocclusion, and explicit reset. The object-motion
fixture authors raster-format vectors independently from the two rendered
views, checking retained GI on a translated receiver, previous-depth rejection
at a parallel blocker, and invalid-vector rejection on matching geometry.
The fixtures run with forward and reversed depth. The Xcode bridge timed out
resolving the package destination, so verification used `xcodebuild`.
