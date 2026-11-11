# Temporal path denoising with MetalFX

MetalFX's temporal denoised scaler takes HDR color, depth, motion vectors,
diffuse and specular albedo, signed world-space normals, and roughness.
It also uses the frame's world-to-view and view-to-clip matrices and sampling
jitter. Specular hit distance and the strength mask are optional.
[MetalFX descriptor](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerdescriptor) ·
[Apple's input guidance](https://developer.apple.com/videos/play/wwdc2025/211/)

A descriptor probe against Xcode 27.2 on an M4 Max returned true from
`supportsDevice` and `supportsMetal4FX`. A 320 × 192 to 640 × 384 scaler
could be created with RGBA16Float color, albedo and normal textures,
RG16Float motion, R16Float roughness, and either R32Float or depth32Float
depth. The instance requested shader-read inputs and shader-read,
shader-write, and render-target usage on the output. This probe tested
factory creation. The 27.2 SDK excludes the temporal denoiser on visionOS.

| Input | Frostfield output |
| --- | --- |
| Color | Linear HDR `PathTracer.encode` output. Progressive accumulation and per-ray jitter belong to the existing still and preview path. |
| Depth | `AOVs.projectedDepth` stores forward Metal normalized depth in R32Float. Perspective and orthographic projections are supported. A miss stores one. `AOVs.depth` separately stores linear distance. |
| Diffuse albedo | `AOVs.albedo` stores diffuse base color, reduced for metals and specular transmission. |
| Specular albedo | `AOVs.specularAlbedo` uses the path material's view-dependent dielectric and conductor Fresnel curves, including specular modulation, iridescence, and clearcoat. This approximation excludes incident lighting and rough-lobe integration. |
| Normal and roughness | `AOVs.normal` stores signed world-space shading normals and `AOVs.roughness` stores material roughness. |
| Motion | `AOVs.cameraMotion` writes current-to-previous pixel displacement from the actual first hit, accounting for camera, rigid instance and topology-stable skin or morph motion. Perspective sky misses reproject direction. |

The specular and depth outputs use the same first hit as other AOVs.
Ordinary path tracing uses the centered pinhole guide sample. Specular albedo is
zero for unlit hits, area-light hits, and misses, with an alpha coverage
channel. Requesting planar depth with a panoramic camera throws before
encoding.

## Camera and rigid instance motion

A frame opts in through `AOVs(cameraMotion: .init(vectors:previousFrame:jitter:))`.
The same subpixel offset positions both the beauty ray and the first-hit guides.
Each RG16Float or RG32Float motion texel contains previous minus current
displacement in top-left input pixels. The shader projects the actual current
hit into the previous camera, using its translation, orientation, lens shift,
and field of view. For a moving mesh the hit's triangle and barycentric
coordinates identify the corresponding object-space point, which the shader
transforms with that instance's prior world matrix. The history retains the
completed frame's immutable instance records, so multiple refits between
renders still reproject to the frame MetalFX actually received. Displacement
excludes jitter. A perspective sky
miss uses the ray direction. An orthographic sky miss has no unique position and
uses the corresponding pixel if the view direction is unchanged. Positions
outside the previous projection produce finite out-of-image displacement.
[Apple's motion vector guidance](https://developer.apple.com/videos/play/wwdc2026/359/)

Use a full-image planar pinhole camera and one sample per pixel. Pass nil to
progressive `previous` and pass the completed prior frame through the motion
request. A cut or the first frame uses nil prior frame and writes zero vectors.
The encoder rejects unmatched dimensions and projection kinds, incomplete or
superseded histories, incompatible scene edits, and changes to the light rig
or surface graph uniforms. `updateTransforms(_:)` preserves correspondence
for matching node UUIDs and stable `InstanceSet` slots. Rebuilding a
corresponding native skin or morph pose retains temporal history. Explicit
geometry replacements, shader displacement rebuilds, material repacks and
replacing a whole instance set or its slots require nil history. The native
wrapper restarts history for those edits automatically.
Adding or removing rendered nodes requires preparing a new scene and tracer.
Callers must reset history when they replace texture contents directly.

## Native denoised frames

`PathTemporalDenoiser` owns the path accumulation, the native scaler, the guide
textures, and an RGBA16Float output. It binds the traced HDR color, normalized
depth, signed world normal, roughness, diffuse and specular albedo, and camera
motion. Inputs carry both the usage MetalFX requests and the tracer's shader
write usage. The private output is shader-readable for post processing on the
same queue. Equal input and output sizes run at native resolution. Larger
outputs preserve aspect ratio.

The denoiser shares the unjittered camera projection construction with shader
graphs. It sends the current world-to-view matrix and the jittered projection
that matches the primary ray samples. MetalFX's jitter is the image's offset,
which is the negative of the camera sample offset. The pass supplies forward
depth and unjittered motion with a scale of one input pixel per unit. Automatic
exposure uses the traced HDR image.
[Apple's temporal sampling example](https://developer.apple.com/videos/play/wwdc2022/10103/)

Check `PathTemporalDenoiser.isSupported(device:)` on the path tracer's GPU.
The public type reports unsupported on visionOS and in simulators. On native
macOS and iOS the wrapper checks the scaler's runtime device support and
creates its descriptor with the requested dimensions.

The pass uses one frame in flight. Encode it on the path tracer's retained
command buffer, append output consumers on that queue, and submit the buffer.
A succeeding command promotes its camera history. A failed command restarts
the next frame. `reset()` starts a new jitter and history sequence after the
outstanding frame finishes. If a frame will never be submitted, call
`discardUnsubmittedFrame(_:)` with its buffer and discard that entire buffer.

## Remaining integration

- Native skin and morph rebuilds can retain temporal history when source
  topology and correspondence are stable. Arbitrary replacement and shader
  displacement rebuilds still restart history. [Pose motion evidence](path-deformation-motion.md)
- The path pass uses first-hit guides and a visible background for opaque
  content. It rejects world fog. Denoising strength masks, transparency
  overlays, secondary-hit guides for mirrors and glass, and their quality
  across disocclusions remain open. The existing spatial filter covers
  unsupported devices and stills.
  [Apple's neural rendering guidance](https://developer.apple.com/videos/play/wwdc2026/359/)

## Verification

Xcode 27.2 reported seven passing tests across `PathAOVTests`, `PathRegionTests`,
and `PathTraceDenoiserTests` on arm64 macOS with Metal API Validation. The AOV
suite used both software and Metal acceleration. Four focused AOV tests passed
on iPad Pro M5 Simulator using software acceleration. The depth comparison
uses the raster camera's projection matrix. Specular checks cover dielectric
and metal mixing, viewing angle, tinted specular under clearcoat, and unlit
surfaces.

Xcode MCP opened the Lab workspace but could not create build data on the full
external volume. The package workspace open timed out. The test commands used
`xcodebuild` with DerivedData on the internal volume.

The camera motion change passed eleven macOS tests across
`PathCameraMotionTests`, `PathAOVTests`, and `PathRegionTests` with Metal API
Validation. Both ray tracing backends were exercised. Five camera motion tests
passed on iPad Pro M5 Simulator. The fixtures check both directions of camera
translation, sky rotation, prior clip rejection, zoom and lens shift,
orthographic and perspective jitter, and invalidation after scene edits.

The native wrapper passed twelve focused macOS tests on the M4 Max with Metal
API Validation. A 48 × 36 to 96 × 72 run retained HDR radiance, followed camera
pans in perspective and orthographic views, and reset to the same image as a
fresh scaler. Queue mismatch, unsubmitted-frame discard and scene-edit
history invalidation were exercised. On a static area-lit surface, the central
crop after 12 MetalFX frames had over 10% lower RGB RMSE than one path sample
at the output resolution, against a 128-sample reference.

The iOS device build and visionOS Simulator build passed. The iPad Simulator
test verified the unsupported query. Physical iOS runtime quality remains
unqualified.

The rigid-motion change passed 25 tests across five suites on arm64 macOS
with Metal API Validation and seven tests across two suites on iPad Pro M5
Simulator using Xcode 27.2. Two slots of one mesh move and rotate independently
while the camera moves and jitters. Their vectors match analytic projection on
both software and Metal ray tracing after an intervening refit. A rejected
structural update preserves valid history. A bulk slot replacement and a
material repack invalidate it. On the M4 Max, native MetalFX retained history
while an emissive mesh translated, cleared a previously covered pixel to the
background, and matched a fresh scaler after a geometry rebuild.
