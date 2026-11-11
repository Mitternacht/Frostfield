# NRRD scalar volumes

`NRRDLoader.load` reads a local file or data into a typed scalar volume.
It retains signed and unsigned 8-, 16-, and 32-bit integers and Float32
or Float64 values. The first axis varies fastest. The loader supports
embedded raw data, one gzip member, ASCII aliases, and hexadecimal
payloads. Hexadecimal byte streams follow Teem's byte order and take
precedence when they match the declared size. Hexadecimal scalar tokens
are also accepted when the byte stream does not fit the declared layout.
Binary byte order defaults to little-endian when omitted.

The source affine preserves all three directions and the physical origin.
When directions are absent, scalar spacings set the axis steps, with
unit steps for unspecified values. `voxelToSpace` uses the declared
coordinate system and retains oblique directions. `voxelToRAS` is
available for RAS, LAS, and LPS and converts both the directions and
origin. Physical coordinate units are retained when supplied.
[Teem NRRD format](https://teem.sourceforge.net/nrrd/format.html)
[Three.js r187 NRRDLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/NRRDLoader.js)

`makeTexture(device:)` uploads a single-channel Metal 3D texture in the
original voxel order. Native integer formats preserve every integer bit,
and Float32 sources upload as R32Float. Explicit Float32 conversion
provides floating-point shader access for integers and Float64. Rounding
is possible during conversion. Finite doubles outside Float32 range
produce an error. Integer textures use typed shader reads. Float32
linear filtering depends on the Metal device.

The parser checks the header, scalar type, dimensions, field layout,
and exact payload lengths. SDK zlib validates gzip, including its CRC
and decoded size. Limits are 64 KiB of header, 256 MiB of encoded input,
2,048 samples per axis, and 2²⁴ voxels. Detached data, nonzero byte or
line skips, bzip2, 64-bit integers, and non-spatial or multicomponent
layouts are outside this loader.

The pinned r187 loader reads an embedded payload and uses its first three
dimensions for a scalar volume. Detached and multicomponent datasets are
additional NRRD format capabilities beyond this comparison.
[Three.js r187 NRRD parser](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/NRRDLoader.js)

## Verification

The focused suite covers big-endian input, typed text values, both
hexadecimal layouts, gzip corruption, oblique LPS and LAS affine
conversion, and malformed inputs. Metal probes read negative Int16,
UInt16, and UInt32 above the exact Float32 integer range. Float probes
sample actual 3D textures uploaded from Float32 and Float64 values.

Both NRRD files from the pinned Three.js r187 demos matched the
JavaScript loader's complete scalar SHA-256 and voxel spacing on Mac
and iPad Simulator. The scan contains 7,405,568 Int16 samples and the
stent contains 4,194,304 Float32 samples. Eight GPU samples per platform
matched the reference, including scan values 1,300 and 3,952 and a
stent value of 1. The fixture run reported 36 declarations on each
platform, with two native BC tests skipped on iPad Simulator.
[Three.js r187 NRRD fixtures](https://github.com/mrdoob/three.js/tree/r187/examples/models/nrrd)

Xcode 27.2 MCP timed out resolving the package destination. Verification
used `xcodebuild` with internal DerivedData and Metal API validation.
