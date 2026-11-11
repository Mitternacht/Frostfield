# Path tracer output

## Image regions

`PathTracer.Region` lets an accumulation render part of an image up to
32,768 pixels on either axis. Pass it to either `PathTracer.encode` camera
overload. The region's origin is measured from the top left of the full image.
The accumulation, destination, and first-hit guides cover only the region.
The camera projection, lens rays, and random sample indices use full-image
coordinates. Use nil history when switching regions.

`PathRegionTests` renders a 37 × 30 image in four uneven regions. Color and
normal/depth guides match a full-frame render bit for bit through a perspective
thin lens and a top-bottom omnidirectional stereo camera. The odd tile heights
exercise both sides of the stereo eye boundary. A second test renders the
bottom-right two-by-two region of a 32,768 × 32,768 image, rejects out-of-bounds
regions, and verifies history ownership across origins and image sizes.

On an M4 Max Mac with Xcode 27.2 (27B52), ten tests in `PathRegionTests`,
`PathProjectionTests`, and `PathTracerTests` passed with Metal API Validation.
The Xcode bridge timed out resolving the package destination. Its Lab workspace
opened, but the external DerivedData volume could not write the build arena, so
verification used `xcodebuild` with DerivedData on the internal drive.

```sh
TEST_RUNNER_MTL_DEBUG_LAYER=1 xcodebuild test -scheme Frostfield-Package \
  -destination 'platform=macOS,arch=arm64' -derivedDataPath /tmp/frostfield-path-output \
  -only-testing:FrostfieldTests/PathRegionTests \
  -only-testing:FrostfieldTests/PathProjectionTests \
  -only-testing:FrostfieldTests/PathTracerTests
```

The caller assembles regions and writes still files. Denoising each region
independently can introduce seams. Physical mobile and headset qualification
remains open.

## First-hit outputs

Supply `PathTracer.AOVs` to either camera overload to write diffuse albedo,
world-space shading normal, depth, roughness, emission, object ID, and
material ID textures. The caller can request any subset. The outputs describe
a centered pinhole ray through each pixel and use the same visibility rule as
the existing guides. They can be requested on later accumulation passes.
The existing first-hit guides retain their original layout and update timing.

Depth is measured forward from the camera plane for perspective and
orthographic views and along the ray for panoramas. A miss stores positive
infinity in depth and zero in the other layers. IDs are zero for misses and
camera-visible area lights. Object IDs derive from the node UUID, and all
instances of a node share its ID. The ID is a 32-bit hash and can collide.

`PathAOVTests` verified the material channels and IDs for a lit planar scene
on both Metal ray tracing and the software BVH. It checked misses, existing
guides, layer requests after the initial accumulation pass, and rejection of
wrong formats, sizes, and aliases. Eight tests in `PathAOVTests`,
`PathRegionTests`, and `PathTracerTests` passed on the M4 Max with Metal API
Validation, including ten runs when backend and camera arguments are counted.

The path tracer also supplies specular albedo, planar device depth, and
camera motion for temporal frames. Specular hit distance remains open.
[Temporal input evidence](path-temporal-inputs.md)

## Layered OpenEXR

`EXRExporter.data(width:height:channels:)` stores the beauty image and
auxiliary channels in one scanline file. Name channels by layer and component,
such as `normal.X`, `depth.Z`, and `objectID.id`. Each channel supplies its own
half, float, or unsigned integer samples. Depth infinity, signed normals,
and the full 32-bit IDs survive export. Uncompressed, RLE, ZIPS, ZIP, and PIZ
use the same writer as the RGBA convenience methods.
[Codec evidence](exr-codecs.md) · [OpenEXR file layout](https://openexr.com/en/latest/OpenEXRFileLayout.html)

`LayeredEXRTests` compares the raw sample bits across the supported encodings,
including a partial final ZIP block, RLE packets, PIZ channel planes, and
raw fallback cases.
The HDR suite decodes the beauty image
with ImageIO. It also checked malformed channel names, duplicates,
incomplete planes, and the 512 MiB uncompressed payload limit. The final
M4 Max run passed 17 tests across layered EXR, existing HDR loading and
export, path AOVs, and tiled rendering with Metal API Validation.
The iPad Pro 13-inch (M5) Simulator on iOS 27.2 passed seven tests across
layered export, regions, and AOVs, with ten runs counting the arguments.

The in-memory API accepts caller-supplied channels. The streaming API below
can write stills without a complete channel array. Multipart OpenEXR remains open.

## Frame budgeting

`PathTracer.FrameBudget` turns a completed `Timing.duration` and the number
of samples just rendered into a bounded sample recommendation for the next
encode. The controller smooths seconds per sample and keeps its last
recommendation when the timing is unavailable. Use the recommendation with
the tracer's normal history and maximum-sample rules.

`PathFrameBudgetTests` verifies the response to changes in sample cost, the
sample limits, invalid inputs, and extreme finite timings. Eight tests in
`PathFrameBudgetTests` and `PathSamplingTests` passed on the M4 Max with Metal
API Validation. The controller leaves render size and per-region convergence
to the caller.

## Streamed EXR output

`EXRExporter.write(to:width:height:scanlines:)` takes a provider of full-width
scanline blocks. It writes 1 to 64 mixed half, float, or UInt32 channels for
images up to 32,768 pixels per axis. The exporter holds one block and the EXR
offset table, then atomically replaces the destination after completing the
table. A provider can assemble each row band from path-traced tiles. The
writer preserves the chosen precision and channel names.

`LayeredEXRTests` checks that streamed and in-memory files are byte-identical
under all five encodings, that rows are requested once in increasing
order, and that a failed provider cannot replace an existing file. A
17,000-pixel-wide fixture exercises the streamed size range.
[OpenEXR file layout](https://openexr.com/en/latest/OpenEXRFileLayout.html)

`PathTracer.writeStillEXR` writes half-float beauty RGBA by default. Add a
`layers` selection to include first-hit channels. The layer names are
`albedo.R/G/B/A`, `normal.X/Y/Z/A`, `emission.R/G/B/A`, and
`specularAlbedo.R/G/B/A` in half precision. `depth.Z`, `projectedDepth.Z`,
and `roughness.R` store 32-bit floats. `objectID.id` and `materialID.id`
store UInt32 values. Panoramas support ray-distance `depth.Z` and reject
projected depth. Motion vectors are available through the temporal frame API.

Each GPU tile is at most 1,024 columns by sixteen rows. Only requested guide
textures are allocated, and each tile writes its guides on the first sample
batch. The CPU assembles one full-width scanline block at a time. Whole-image
projection and sample sequences match an untiled render. The file is published
when all batches finish. Scene revision and live uniform checks prevent an
edit during export from producing a mixed still.

`PathRegionTests` compares the default beauty-only file with an untiled
1,037 × 19 thin-lens render. The layered test puts a tilted surface across the
same horizontal and vertical tile boundaries and compares the file byte for
byte with full-frame beauty and all nine first-hit layers. It checks finite
hit depths, infinite miss depths, signed normals, and a UInt32 object ID that
would lose precision if converted to Float. Both exports use 17 samples split
across two dispatches. A panoramic fixture compares the selected ray-depth
export and checks that requesting projected depth preserves an existing file.

Xcode 27.2 passed all 14 focused tests in `PathRegionTests`, `PathAOVTests`,
and `LayeredEXRTests` on macOS with Metal API Validation. All five
`PathRegionTests` passed on the iPad simulator. The Xcode bridge could not
create its build arena on the external volume, so the runs used `xcodebuild`
with DerivedData on the internal volume.
