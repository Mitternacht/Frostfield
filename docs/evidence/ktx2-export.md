# Native KTX2 texture export

`KTX2Exporter.data(from:commandQueue:options:)` writes uncompressed KTX 2.0
from single-sample 2D, 2D array, cube, cube-array, and 3D Metal textures. It accepts 8-bit UNORM
and sRGB R, RG, RGBA, and BGRA formats, and Float16 and Float32 R, RG, and
RGBA formats. Metal's stored bits set the Vulkan format and transfer
function. Optional primaries and alpha association describe the data
without converting it. HDR values retain their floating-point bits.

The writer stores all allocated mip levels with tightly packed rows.
The level index starts at the base level and the payload starts at the
smallest mip. Each payload begins at the texel-aligned file offset required
by KTX2. Array layers are contiguous in ascending Metal slice order within
each mip. A one-layer array keeps its declared type. Cube faces use Metal
and KTX order (+X, −X, +Y, −Y, +Z, −Z).
Orientation metadata describes top-left rows and forward depth slices.

Populate every mip and commit GPU writes before export. Supply the same
producer queue to order readback after earlier tracked writes on it. Finish
untracked writes and writes from other queues before export. The exporter
waits for the Metal blit and uses the loader's 64 MiB file budget.

`TextureLoader.load(data:)` reads exported 2D files. Use
`TextureLoader.loadKTX2(data:)` for 2D arrays, volumes, and cubes. It uploads
existing mips and returns the same format with default interpretation. The
ordinary image loader treats a one-layer file as a 2D image for compatibility.
The writer covers the formats of the Three.js r187 exporter and adds stored mip
chains, cube textures, 2D arrays, cube arrays, and Metal-native BGRA. Block
encoding, Basis Universal output, and supercompression are outside the export
API, for cube arrays as well. Basis-encoded cube arrays are a separate codec
task. Metal on iPhone Simulator
rejects the R8 and RG8 sRGB formats. Loading these files there reports the
format limitation. The explicit linear-data interpretation preserves their
bytes in the corresponding UNORM formats.
[Three.js r187 exporter](https://raw.githubusercontent.com/mrdoob/three.js/r187/examples/jsm/exporters/KTX2Exporter.js) ·
[Khronos KTX 2.0 specification](https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html) ·
[Apple Simulator limitations](https://developer.apple.com/documentation/metal/developing-metal-apps-that-run-in-simulator)

## Cube arrays

A `.typeCubeArray` texture exports with `faceCount` 6 and a `layerCount` equal
to its cube count. Metal's `arrayLength` counts cubes, and so does KTX2. Each
level holds its cubes in order, and each cube holds +X, −X, +Y, −Y, +Z, −Z.
This is the layer, face, depth-slice order of KTX 2.0 section 3.13.1, and it is
also Metal's slice order, `6 × cube + face`, so one readback in slice order
is the file order. A level's length is the face byte count times six times the
cube count, with `uncompressedByteLength` equal to it, because supercompression
is none. The level index lists the base level first. Payloads are stored smallest
first at offsets aligned to the larger of four bytes and the texel size.

A one-cube array writes `layerCount` 1 and so stays distinct from a plain
cube, whose `layerCount` is 0. The key/value data holds `KTXorientation`
`rd` and `KTXwriter` only. [`KTXcubemapIncomplete`](https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html#_ktxcubemapincomplete) is never written, because
every exported cube is complete, and incomplete cubes are not exported. The
formats are the ones the 2D-array export accepts. Cube arrays need Apple4 or
Mac2 GPU support. The iPhone Simulator does not expose them, so the native
cube-array tests skip there.

The exporter reports typed errors for shapes it cannot write.
`cubeFaceSize(width:height:)` covers non-square faces, `arrayLength(_:)` covers
fewer than one or more than 2,048 layers or cubes, and `size` covers more than
16,777,216 texels across all six faces of every cube or a file above 64 MiB.
`pixelFormat` and `textureType` keep their earlier meaning. Metal already
guarantees square cubes and an array length from 1 to 2,048, so `cubeFaceSize`
and `arrayLength` are reachable only through the internal layout planner that
the tests call directly. Every check runs before any texel is read back.

`TextureLoader.loadKTX2` still rejects a cube array with
`unsupportedKTX2("cube map arrays")`. KTX2 stores a cube array and a 2D array
of `6 × cubes` layers identically, so the tests relabel the exported header as
that array and import it to read every face back through Metal.
[KTX 2.0 levelImages order](https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html#levelImages) ·
[KTX 2.0 texture types](https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html#_texture_type)

## Verification

Xcode 27.2 passed 16 focused tests in `KTX2ExporterTests`, `KTX2LoaderTests`,
and `CubeTextureLoaderTests` on arm64 macOS with Metal API Validation. The
parameterized fixture covered all 14 export formats, byte-exact round-trips
of odd-sized mips, descriptor fields, file alignment, and deterministic
output. A private render target retained positive and negative HDR values.
Other fixtures checked each cube face and 3D depth plane.

The iPhone 18 Pro Simulator run succeeded with 17 reported tests, including
12 export formats and rejection of both unavailable narrow sRGB formats.
BC1 sampling was capability-skipped. The initial simulator run exposed the
RG8 sRGB allocation failure, which the format guard now rejects before any
Metal allocation or readback.

The Xcode package workspace open timed out, and the Lab workspace could not
create its build arena on the full external volume. Verification used
`xcodebuild` with DerivedData on the internal volume.


## 2D array verification

R8 UNORM, RGBA8 sRGB, and RGBA16Float textures with one and three layers
retain each slice and mip through public export and import. The test checks
layer count, orientation, transfer, alpha metadata, aligned offsets, tight
per-layer storage, and exact bytes after Metal readback. It also checks the
64 MiB file limit including container overhead. Three independent export
files passed Khronos KTX-Software v4.4.0 validation with warnings treated as
errors. [Khronos KTX-Software v4.4.0](https://github.com/KhronosGroup/KTX-Software/releases/tag/v4.4.0)

The expanded native suite passed all 30 tests on macOS. The iPhone Simulator
reported 31 tests with two existing BC sampling capability skips. Both runs
used Metal API Validation.

## Cube-array verification

`KTX2CubeArrayExportTests` writes one and two cubes of R8 UNORM, RGBA8 sRGB and
RGBA16Float, with 5 × 5 faces and three mips of 5, 2 and 1 texels. Every
face of every cube and level carries different bytes. The tests check the
header words, level index, descriptor, key/value entries, zero padding, level
order, and the position of each face at `offset + (6 × cube + face) × image
bytes`. A byte-exact test compares the first 296 bytes of the two-cube RGBA8
sRGB file, which are the header, level index, 92-byte descriptor and 52 bytes
of key/value data, with literal values, and checks the 1,736-byte length. The
same bytes declared as a 2D array of 6 × cubes layers import through
`loadKTX2`, and each Metal slice matches. The loader still refuses the
cube-array declaration. Separate tests cover
the layout planner's typed errors, including the texel limit, which allows two
1,024-square cubes and rejects three, and the file budget, which allows ten
256-square RGBA32F cubes and rejects eleven.

On a cube-array GPU, a private cube-array texture is filled slice by slice
and exported through the public API. The file matches the layout encoder's
output for the same bytes. A further native test rejects an unsupported
Snorm format and oversized shapes. The previous expectation that cube arrays
fail with `textureType` was removed from the unsupported-layout test, since
that is now valid input. The 1D array rejection remains.

Xcode 27.2 command-line `xcodebuild` ran seven focused suites with Metal API
Validation. The macOS run reported "Test run with 42 tests in 7 suites passed"
with nothing skipped. The iPhone 18 Pro Simulator run reported 43 tests in
7 suites passed, with four tests skipped: the two native cube-array tests and two
BC sampling tests. The Simulator count includes its narrow-sRGB test. The
Simulator therefore covered the cube-array layout and relabeled import with
synthetic readback bytes, and the Mac covered native textures.

The Xcode MCP bridge timed out, so command-line `xcodebuild` replaced it. The
test writes its cube-array files only when the environment variable
`FROSTFIELD_KTX2_EXPORT_DIR` names a directory. The Mac run passed it through
`TEST_RUNNER_FROSTFIELD_KTX2_EXPORT_DIR`, producing six files.

### Khronos validator

The six files passed `ktx validate --warnings-as-errors <file>` from
[KTX-Software v5.0.0-rc2](https://github.com/KhronosGroup/KTX-Software/tree/v5.0.0-rc2)
(`ktx version: v5.0.0-rc2~3`), each with exit status 0. The JSON report for the
two-cube RGBA8 sRGB file said `"valid": true` with no messages, and `ktx info`
read `layerCount` 2, `faceCount` 6, three levels, and level offsets 0x218, 0x158
and 0x128, which are the literal offsets the test expects. Two corrupted
copies, one with `faceCount` 5 and one with `layerCount` 0, failed with the
validator's faceCount and level byteLength errors, so the tool does read the
cube-array layout. The earlier 2D-array validation used v4.4.0. That binary is no
longer on this machine, so the cube-array files were not checked with v4.4.0,
and v5.0.0-rc2 is a release candidate. The three 2D-array files that passed
v4.4.0 also pass v5.0.0-rc2.

```
TEST_RUNNER_FROSTFIELD_KTX2_EXPORT_DIR=<directory> MTL_DEBUG_LAYER=1 xcodebuild test \
  -scheme Frostfield-Package -destination 'platform=macOS' \
  -only-testing:FrostfieldTests/KTX2CubeArrayExportTests
ktx validate --warnings-as-errors <directory>/cube-array-rgba8-srgb-2-cubes.ktx2
```

