# Asset formats for Apple pipelines

Frostfield writes USDZ packages for AR Quick Look and USD tools, imports the
formats ModelIO reads, and loads and writes high-dynamic-range images. Each row
names the public API, the path the data takes, and the Swift Testing suite that
checks it. Measurements come from an M4 Max Mac Studio with macOS 27.2 (26B123)
and Xcode 27.2 (27B52), with Metal API Validation enabled for the GPU tests.

## USDZ export

| Capability | API | Path | Verified by |
| --- | --- | --- | --- |
| USDZ package: stored zip, 64-byte aligned file data, USDA root layer first, no zip64 | `USDZExporter.export(_:)` | `USDZArchive` (Swift) | `USDZExporterTests` |
| Scene graphs: `Xform` hierarchy with translate/orient/scale ops, shared mesh layers, instance sets as matrix `Xform`s, perspective camera | `export(_ graph:nodeNames:animation:includesCamera:)` | USDA text | `USDZExporterTests` |
| Imported glTF scenes: node names and transforms, primitives, materials, sampler wrap modes, cameras, GPU instances | `export(_ scene:animationIndex:)` | USDA text | `USDZExporterTests` |
| Meshes: points, normals, `st` and `st1` flipped to USD's bottom-left space, `float4` tangents, extents, `subdivisionScheme = "none"` | both | `geometries/Geometry_N.usda` referenced by each mesh prim | `USDZExporterTests` |
| UsdPreviewSurface: base color, opacity and threshold, metallic, roughness, normal, occlusion, emissive, clearcoat, clearcoat roughness, IOR, from factors or PNG/JPEG maps with UsdUVTexture `scale`/`bias` and UsdTransform2d | both | `USDMaterialNetwork` | `USDZExporterTests` |
| Transform animation as `xformOp` time samples at 60 time codes per second | `animation:`, `animationIndex:` | USDA time samples | `USDZExporterTests`, `ModelIOSceneLoaderTests` |
| Stage metadata: `metersPerUnit`, `upAxis = "Y"`, `defaultPrim`, `kind = "component"` | `USDZExporter.Options` | root layer | `USDZExporterTests` |
| Warnings for features UsdPreviewSurface cannot express | `Output.warnings` | — | `USDZExporterTests` |

Checks:

- Every file in the archive starts on a 64-byte boundary, is stored, and has a
  matching CRC-32; `unzip -v` agrees. The 80 name lengths from 1 to 80 bytes all
  align, including the cases that need a full extra 64-byte block.
- ModelIO reads the package back with the same local transforms (within
  1 × 10⁻⁵), vertex counts, index buffers, positions, normals, and UVs, the same
  camera field of view and clipping range, and the UsdPreviewSurface inputs.
  Stored PNG bytes decode to the source texture's bytes.
- `usdchecker --arkit --strict` reports success for the textured fixture, the
  unlit/masked/transformed-map fixture, and an animated stage.
- UsdTransform2d parameters reproduce `TextureTransform.apply(to:)` in `st`
  space within 1 × 10⁻⁵ for rotated, mirrored, and offset transforms.
- An animated box exported with three keys has ModelIO key times 0, 0.5, and
  1.25 s, and its sampled matrices match each key within 1 × 10⁻⁵.
- A glTF clip with 400 animated nodes and 60 LINEAR translation, rotation, and
  scale keys each (1,200 channels) exports in 0.21–0.24 s in a Debug build. Each
  node evaluates only its own channels; evaluating the whole clip for every node
  at every key took 23.2 s (14.0 s of thread CPU time) for the same scene. The
  test limits thread CPU time to 3 s, which load from other processes does not
  inflate. ModelIO reads 60 key times on the
  first, middle, and last nodes, and their matrices match the source keys within
  1 × 10⁻⁵.

`usdrecord` (Hydra Storm) renders the exported fixture with the wood texture in
place, and `usdtree` shows the expected prim hierarchy.

Limits: UsdPreviewSurface has no transmission, volume, specular, sheen,
anisotropy, iridescence, dispersion, or clearcoat normal; these are dropped with
a warning. Normal maps are written with the standard `(2, 2, 2, 1)` scale that
USD's compliance rules require, so a normal scale other than one is reported and
dropped. Unlit materials become emission over a black diffuse response.
Lights, skins, morph targets, and texture filtering are not exported. A USD
stage has one timeline, so only one clip is written; STEP and CUBICSPLINE
channels are sampled every 1/60 s. The root layer and geometry layers are USDA
text, which is larger than binary USDC. AR Quick Look itself was not run;
compatibility evidence is `usdchecker --arkit`.

## ModelIO import

| Capability | API | Path | Verified by |
| --- | --- | --- | --- |
| USD, USDA, USDC, USDZ, OBJ, PLY, STL, and Alembic into `GLTFScene` | `ModelIOSceneLoader.load(contentsOf:options:)` | `MDLAsset` with preserved topology | `ModelIOSceneLoaderTests` |
| Polygons ear-clipped in their plane, so concave faces stay inside their outline | same | `PolygonTriangulation` | `ModelIOSceneLoaderTests` |
| `doubleSided = 1` on USD meshes becomes a double-sided material | same | text root layer scan (`USDLayerMetadata`) | `ModelIOSceneLoaderTests` |
| OBJ materials from MTL: `Kd`, `d`/`Tr`, `Ke`, `Ns`, `Ni`, `Pr`, `Pm`, `Pc`, `Pcr`, `map_Kd`, `map_d`, `map_Ke`, `map_Pr`, `map_Pm`, `norm`, `bump`, `-s`, `-o`, `-clamp`, `-bm` | same | `WavefrontMaterialLibrary` | `ModelIOSceneLoaderTests` |
| Standalone named Wavefront material libraries with local maps | `MTLLoader.load(contentsOf:)` | Shared OBJ material conversion | `MTLLoaderTests` |
| UsdPreviewSurface inputs as ModelIO reports them, with USDZ textures read from the package (directory parsed once, file mapped, not copied) | same | `ModelIOMaterialImporter`, `USDZArchive` | `ModelIOSceneLoaderTests` |
| Separate metallic, roughness, and opacity maps packed into glTF channel layout | same | PNG | `ModelIOSceneLoaderTests` |
| `metersPerUnit` and Z-up conversion at the roots, cameras in meters | `Options.convertsToMeters`, `convertsToYUp` | text USD layer metadata, `MDLAsset.upAxis` | `ModelIOSceneLoaderTests` |
| Time-sampled transforms as a LINEAR clip | `Options.includeAnimations` | `MDLTransform` key times | `ModelIOSceneLoaderTests` |
| Perspective cameras | `Options.includeCameras` | `MDLCamera` | `ModelIOSceneLoaderTests` |

The result prepares with `PreparedGLTFScene` and exports with `GLTFExporter` or
`USDZExporter`. Rendered-pixel checks against the source scene graph, with the
same camera and light:

- Three constant-factor PBR spheres (dielectric, clearcoated metal, emissive)
  exported to USDZ and imported back render with a maximum channel difference
  below 1 × 10⁻³ over a 96 × 96 target (measured 0). Scaling the imported
  roughness by 0.98 raises it to 0.16.
- A wood-textured bevel box with base color, metallic-roughness, occlusion, and
  normal maps renders with a mean channel difference below 1 × 10⁻⁵ and a
  maximum below 1 × 10⁻³ (measured 2.5 × 10⁻⁸ and 1.2 × 10⁻⁴); the imported
  maps are the package's PNG bytes. Halving the imported normal scale gives a
  mean of 1.2 × 10⁻⁴ and a maximum of 4.3 × 10⁻³, which fails. The package sits
  in a folder named `a[1]`, so the `package.usdz[entry]` paths ModelIO reports
  are split at the package's own path, not at the first bracket.
- An animated box plays back through the imported clip at 0, 0.3, 0.5, 0.9, and
  1.25 s with a maximum channel difference below 2 × 10⁻³ (measured
  6.1 × 10⁻⁵), including the times between keys.
- Two squares facing away from the camera, one double-sided, export and import
  with the same `isDoubleSided` flags and render within 1 × 10⁻³ of the source:
  only the double-sided square covers pixels.
- A Z-up, centimeter USDA stage puts a 100 cm quad at 1 m in Frostfield's Y-up
  meters, and its time-sampled translation moves the rendered panel to the
  analytic column at 0, 0.5, and 1 s.

Other checks: an OBJ with two materials and a concave L-shaped face keeps its
factors and maps and covers exactly 3 square units with front-facing
triangles; ModelIO's own triangulation fans that face and covers the notch.
Ear clipping produces n − 2 triangles that all wind with the polygon and cover
its exact area for a keyhole (a square hole bridged to its outline), a
collinear run, 200 random star polygons of 5 to 64 corners, and a 1,020-corner
double spiral strip. The ear search resumes beside each clipped ear and tests
only non-convex corners, so that spiral triangulates in about 0.06 s in a Debug
build; restarting the search at the first corner after every ear took 4.9 s
(3.2 s of thread CPU time; 8.7 s in a standalone -Onone build). The test limits
thread CPU time to 0.75 s. An OBJ whose comment holds the Latin-1
byte 0xE9, with an MTL holding the byte 0xB0, keeps `Pr`, `Pm`, and `-s` from
its MTL; such files are read as ISO Latin-1 when they are not valid UTF-8. A
USDA scan finds `doubleSided` opinions in typed and untyped prims and `over`
prims, and ignores them in comments (`#`, `//`, `/* */`), strings, triple-quoted
strings, metadata dictionaries, and asset paths. A
material with `Tr 0.25`, a separate `map_d`, `map_Ke`, `bump -bm 0.5`, and
`map_Kd -o 0.25 0 -clamp on` imports as blended with alpha 0.75, the dissolve
map packed into the base color alpha byte for byte, clamped wrapping, the
offset in top-left UV space, and a normal scale of 0.5.

The standalone MTL loader returns named materials that prepare for Metal
through `GLTFScene.Material.prepare(device:)`. It uses the same local map
resolution, opacity packing and warnings as OBJ import. Phong shininess
maps to PBR roughness. Classic specular and displacement maps remain open.
[Three.js r187 MTL loader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/MTLLoader.js)

The original iOS transparency fixture isolated an ImageIO decode difference.
The packed PNG contains exact straight RGBA bytes. When ImageIO supplies
8-bit premultiplied samples, `TextureLoader` recovers the encoded channels
of static noninterlaced RGBA8 PNGs for `.linearData` and `.encodedSRGB`.
The fallback checks required chunk CRCs and the zlib stream, applies all
five PNG filters, and preserves RGB at low and zero alpha.
`TextureLoaderTests` and the original exact MTL test cover the result.

ASCII and binary little-endian PLY files written by the test import with +Z
winding. Normalized PLY sRGB vertex colors become linear mesh RGB with
opaque alpha. ASCII and binary STL import, and a ModelIO-written Alembic box
imports with its extents. A texture path outside the asset folder is refused
with a warning. [Mesh export evidence](mesh-export.md)

Limits, all from ModelIO on macOS 27.2 unless noted: UsdLux prims arrive as plain
transforms, so lights are not imported. ModelIO exposes no tangent attribute
from any format it reads (USD `primvars:tangents` and `primvars:tangent`, OBJ,
PLY, STL, Alembic), so imported primitives carry no tangents and normal maps
use the frame the renderer derives from texture coordinates; a test fails if a
later ModelIO starts reporting them. UsdUVTexture `scale`, `bias`, wrap
modes, channel outputs, and UsdTransform2d nodes are not reported, so USD maps
import with unit factors, repeat wrapping, and identity transforms, and a map
shared by `metallic` and `roughness` is read with glTF's green/blue layout.
`opacityThreshold` is not reported. ModelIO omits PLY alpha and texture coordinates,
and an ASCII PLY with UVs but without normals fails to open. A ModelIO-written
Alembic transform reads as an all-zero matrix, which becomes identity with a
warning. `metersPerUnit` and `doubleSided` are read by Frostfield from text root
layers only: a binary root layer reports `binaryRootLayer`, opinions in
sublayers or referenced layers are not found, and a double-sided mesh without a
bound material keeps the default single-sided material. Vertex colors are
dropped. ModelIO's thread safety is undocumented, so imports are serialized
process-wide.

## High-dynamic-range images

| Capability | API | Path | Verified by |
| --- | --- | --- | --- |
| OpenEXR into RGBA16Float or RGBA32Float, bit-exact in float | `TextureLoader.loadHighDynamicRange(data:precision:mipmapped:)` | ImageIO, raw copy | `HighDynamicRangeImageTests` |
| Radiance RGBE at any orientation and aspect ratio, up to 8,192 per side and 16,777,216 pixels; every RGBE value exact in `.float` | same, and `load(data:)` | `RadianceHDRImage.decodeLinear` | `HighDynamicRangeImageTests` |
| ISO 21496-1 gain-map HEIC, JPEG, and AVIF to linear HDR, upright | same | ImageIO `kCGImageSourceDecodeToHDR` | `HighDynamicRangeImageTests` |
| OpenEXR and Radiance in the ordinary loader as linear RGBA16Float | `TextureLoader.load(data:interpretation:mipmapped:)` | routed | `HighDynamicRangeImageTests` |
| OpenEXR export, half or float, uncompressed, RLE, ZIPS, ZIP, or PIZ | `EXRExporter.data(width:height:pixels:)`, `data(from:)` | Swift writer and bounded PIZ bridge | `HighDynamicRangeImageTests`, `LayeredEXRTests` |

Checks:

- EXR files from `EXRExporter` decode in ImageIO to the stored bits for all six
  pixel-type and compression combinations, including −0, a float subnormal,
  65,504, and 10⁶, on a 37 × 19 image whose last ZIP block is partial.
- `loadHighDynamicRange(precision: .float)` returns those floats bit for bit;
  `.half` returns `Float16` of the value clamped to ±65,504.
- ImageIO's own EXR encoder is not exact: it leaks up to about 2 × 10⁻⁵ of a
  pixel's largest channel into the others, so (1000, 0.1, 7.5) reads back with
  green near 0.117. The test bounds that error by one half-precision step of the
  pixel's peak and records it; `EXRExporter` avoids the conversion.
- A 3 × 2 Radiance file decodes to (m + 0.5) · 2^(e − 136), the Radiance
  reference formula, exactly in both precisions.
- Radiance values outside half precision's range load exactly with `.float`:
  102,656, 2,155,872,256 (the reported `maximumValue`), 1.87 × 10⁻⁹, and the
  single-precision subnormal 255.5 · 2⁻¹³⁵. `.half` clamps the large values to
  65,504 and flushes the small ones to zero, as Three.js `HDRLoader` does with
  `HalfFloatType`.
- A 3000 × 3000 Radiance header (9,000,000 pixels) passes the size check and
  fails at its missing first scanline; 9000 × 10 and 4097 × 4097 fail with
  `TextureLoader.Failure.imageDimensions`. `EnvironmentLighting` keeps its own
  2:1 panorama limit of 8,388,608 pixels.
- Gain-map HEIC, JPEG, and AVIF images made from a 0.5 / 4.0 split with 4×
  headroom decode with headroom 4, the 0.5 side within 0.02, and the bright
  side above 3 (ImageIO tone-maps the SDR base, so the gain map restores
  about 3.4). The ordinary loader returns the SDR base in RGBA8. ImageIO returns
  the HDR rendition as 10-bit PQ (`kCGColorSpaceITUR_2100_PQ`) on both the
  full-size and thumbnail paths; the loader's upright thumbnail path reads
  3.374 and the full-size path 3.405, one 10-bit PQ code step apart near
  4× SDR white, at 64 × 32 and 512 × 256, for HEIC and JPEG alike.
- EXIF orientation 6 turns a 64 × 32 gain-map image into 32 × 64 with the
  left column on top.
- A uniform EXR panorama of radiance (3.25, 1.625, 6.5) produces
  `EnvironmentLighting` sky and diffuse cubes within 0.1% and 1% of that
  radiance, the analytic value for cosine convolution of a constant sky.

Limits: Google Ultra HDR files that carry only the version 1.0 XMP gain-map
metadata were not tested; decoding them depends on ImageIO. Gain-map renditions
carry 10-bit PQ quantization, about 1% per code step near 4× SDR white. HDR results are in
linear sRGB primaries, so wider-gamut content outside them carries negative
components. OpenEXR alpha stays associated (premultiplied); other formats return
straight alpha. EXR export writes single-part scanline files with optional
named layers. Tiled, deep, and DWA output remain open.
[Layered EXR evidence](path-output.md#layered-openexr) · [Codec evidence](exr-codecs.md)

## Six-face cube textures

`CubeTextureLoader(queue:)` loads a cube from six local URLs or six encoded
images. Pass equal square faces in Metal slice order `+X, −X, +Y, −Y, +Z, −Z`.
Each face uses Metal's top-left texel orientation. Convert external skybox
assets to that convention before import.

`load` preserves the shared face format and samples ordinary color images as
sRGB by default. `loadHighDynamicRange` converts each face to linear RGBA16Float
or RGBA32Float. The input can use any supported HDR decoder, including Radiance
and OpenEXR. Request a generated mip chain where the GPU can filter the
chosen format. A returned cube can be sampled directly or passed to
`EnvironmentLighting(queue:cube:settings:)` for sky, diffuse and specular
filtering. KTX2 cube containers and block-compressed face uploads remain open.

`CubeTextureLoaderTests` samples asymmetric PNG faces on the GPU to check
axis order, corner orientation, linearized sRGB and the generated mip. Six
Radiance files exercise HDR values above half-float range and the half-float
clamp. Invalid face counts, sizes, formats and corrupt data fail explicitly.
Xcode 27.2 passed 23 focused tests on an M4 Max and eight on iPhone 18 Pro
Simulator, both with Metal API Validation. The Xcode bridge could not resolve
the package run destination, so verification used internal DerivedData with
`xcodebuild`.
[Three.js r187 CubeTextureLoader](https://github.com/mrdoob/three.js/blob/r187/src/loaders/CubeTextureLoader.js) ·
[Three.js r187 HDRCubeTextureLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/HDRCubeTextureLoader.js)

## Measurements

Release build (`-O`, whole-module), five runs except where noted, on the M4 Max
while about 15 other `xcodebuild` processes shared the machine, so medians are
noisy and minimums are closer to an idle machine.

| Operation | Size | Median | Minimum |
| --- | --- | --- | --- |
| `USDZExporter.export` of a 33,151-vertex, 65,024-triangle textured sphere with three 512² PNG maps | 5.3 MB package | 943 ms | 402 ms |
| `ModelIOSceneLoader.load` of that package | — | 270 ms | 50 ms |
| `EXRExporter`, 2048 × 1024, half, ZIP (3 runs) | 277 KB | 217 ms | 88 ms |
| `EXRExporter`, 2048 × 1024, float, stored (3 runs) | 33.6 MB | 16 ms | 13 ms |
| `loadHighDynamicRange` of the half ZIP file as RGBA16Float (3 runs) | — | 132 ms | 118 ms |
| `loadHighDynamicRange` of the float file as RGBA16Float (3 runs) | — | 107 ms | 27 ms |
| ImageIO HDR decode and float draw of a 2048 × 1024 gain-map HEIC, the calls `loadHighDynamicRange` makes, in a standalone `-O` program | 34 KB | 190 ms | 99 ms |

The USDZ cost is dominated by USDA text formatting and PNG encoding.

Scaling checks, Debug build unless noted, same machine:

| Operation | Before | After |
| --- | --- | --- |
| `USDZExporter.export` of a glTF clip with 400 nodes × 60 keys × 3 channels | 23.2 s wall, 14.0 s thread CPU | 0.21–0.24 s |
| Ear clipping a 1,020-corner double spiral strip (`PolygonTriangulation`) | 4.9 s wall, 3.2 s thread CPU (8.7 s standalone -Onone) | 0.06 s |
| Reading 20 textures of 64 KB from a 257 MiB stored package, `-O` | 91–122 ms, with a 257 MiB copy per texture | 2.5 ms, no copy |
