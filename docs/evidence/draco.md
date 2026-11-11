# Draco geometry compression

`DracoDecoder.decode(_:)` returns a triangle index list and tightly packed,
typed attribute buffers in decoded point order. Attribute records retain
the stream's IDs, semantic kinds, component counts and normalization flags.
The public mesh value is independent of the native decoder's lifetime.

`GLTFMeshLoader.load` and `loadScene` read `KHR_draco_mesh_compression` on
triangle-list primitives. Each compressed semantic selects a Draco attribute
by ID and is validated against the corresponding accessor. Additional
attributes keep their ordinary accessor storage. Scene import uses the same
path for compressed joint indices and normalized skin weights. The loader
chooses compressed data when the extension is optional and fallback data is
also available. Malformed compressed data reports an import error.

The implementation uses Google's native Draco library pinned at
`554628821c3448319d7d4ae48db170b8c63e1aaf`. The Swift package compiles selected
C++ sources behind a private C interface in a separate namespace. The
vendored sources preserve upstream notices and the
[Apache 2.0 license](../../LICENSES/Draco.txt).
[Draco 1.5.9](https://github.com/google/draco/releases/tag/1.5.9)

Frostfield adds checks to the sequential and EdgeBreaker connectivity
decoders, attribute descriptor and storage decoders, and the rANS symbol
table reader. They cap topology and attribute storage before those decoders
allocate it. The standalone wrapper limits input and materialized output
to 64 MiB each, with up to one million points and indices and 32 attributes.
The glTF loader also counts decoded primitive buffers against its document
budget.

## Export

`DracoExporter.encode` writes standalone `.drc` data from local-space
`MeshGeometry` or `PointCloud`. Meshes support sequential and EdgeBreaker
encoding with optional normals, one selected UV set and RGBA colors.
Point clouds support sequential and KD-tree encoding with optional RGBA
colors. Color export can retain linear RGB or convert RGB to sRGB for
standalone consumers. Alpha uses the selected color precision.

Encoding and decoding speed settings range from 0 through 10. Quantization
can be disabled per attribute for meshes and sequential point clouds, or configured
from 1 through 30 bits, starting at two for normals. Quantized normals
store direction. KD-tree needs quantization for every stored attribute.
Sequential encoding preserves input order. EdgeBreaker and KD-tree may
reorder data. EdgeBreaker requires referenced vertices and distinct indices
per face, so use sequential encoding to retain isolated vertices or
repeated indices.

The native bridge validates input counts, topology and quantization
ranges before invoking the pinned encoder classes. It accepts up to
one million points and 999,999 indices and caps the resulting stream at
64 MiB. The encoder skips raw entropy tables when the symbol width
requires tagged coding, and uses bounded prediction at high bit depths.
The existing vendor subset adds 23 encoder translation units and their
required headers. Standalone streams carry
geometry rather than scene hierarchy or material data.
[Three.js r187 DRACOExporter](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/exporters/DRACOExporter.js)

## Verification

Xcode 27.2 reported 24 focused tests passing on both macOS (M4 Max) and
iPhone 18 Pro Simulator. FrostfieldLab built for arm64 visionOS Simulator.
Xcode MCP timed out opening the package, so these checks used xcodebuild
with internal DerivedData.

- `DracoDecoderTests` reads the upstream EdgeBreaker, sequential and legacy
  cube streams and compares decoded position bounds, face winding and
  authored normals. A separate triangle encoded with Draco 1.5.9 exercises
  compressed UInt16 joints and normalized byte skin weights.
- The same tests exercise required and optional glTF compression, mixed
  compressed and uncompressed attributes, missing IDs, accessor count
  mismatches, truncated data and unsupported point cloud input.
- A native AddressSanitizer and UndefinedBehaviorSanitizer smoke run exercised
  6,480 truncations and single-bit mutations of the three cube fixtures.

`DracoExporterTests` round-trips native meshes with the public decoder and
checks exact sequential topology, EdgeBreaker winding and seam attributes,
quantization precision and invalid ranges. The exact sequential and KD-tree
point-cloud outputs were decoded by the upstream Draco 1.5.9 WASM decoder
from the pinned checkout. It verified XYZ, RGBA and duplicate point
retention, including sequential input order.
Xcode 27.2 passed 15 focused tests on macOS and 11 on the iPad Pro 13-inch
(M5) iOS 27.2 simulator, including a 30-bit EdgeBreaker grid. The package
workspace bridge timed out, so verification used internal DerivedData
with `xcodebuild`.
A native AddressSanitizer and UndefinedBehaviorSanitizer smoke run passed
45 valid encode cases across methods, speeds and quantization settings,
and four malformed bridge layouts.

The baseline is Three.js r187 and Khronos' glTF extension. Public point-cloud
decoding, glTF triangle strips and Draco-compressed glTF writing remain open.
[Three.js DRACOLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/DRACOLoader.js) ·
[Khronos Draco extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/`KHR_draco_mesh_compression`/README.md)
