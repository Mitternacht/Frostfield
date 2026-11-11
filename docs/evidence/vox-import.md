# MagicaVoxel VOX import

`VOXLoader.load` reads local VOX files or data, versions 150 and 200.
It preserves model and voxel order, grid dimensions, palette indices,
and the 256-entry sRGB RGBA palette. Files without a custom palette use
the MagicaVoxel default. The static scene hierarchy retains transform
names and uses the first frame of each transform. Translation and rotation are
converted from VOX's Z-up basis to Frostfield's Y-up basis.
[MagicaVoxel VOX format](https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox.txt)
[MagicaVoxel scene extensions](https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox-extension.txt)

`Asset.makeMesh(modelIndex:)` produces centered indexed geometry with
flat outward normals and linear RGB vertex colors. It removes faces
between occupied cells and merges coplanar faces of the same palette index.
`Asset.makeSceneNode` prepares the hierarchy with one Metal renderer per
referenced nonempty model. A legacy file without a hierarchy produces a
group containing all models at the origin. `Model.makeOccupancyTexture`
uploads a shader-readable R8Unorm 3D texture with unit occupancy and
source X-fastest grid order. The mesh uses opaque palette RGB, and the
asset preserves the original palette alpha.
[Three.js r187 VOXLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/VOXLoader.js)
[Three.js MIT notice](../../LICENSES/Three-CurveExtras.txt)

Later animation frames, palette remapping, layer visibility, and material
or camera extensions are outside the static import. The reader validates
chunk extents, model counts, unique in-range coordinates, palette
indices, and scene references. It limits a file to 64 MiB, 131,072
chunks, 4,096 models, and 4,194,304 stored voxels, with dimensions up to
256 per axis. Scene expansion is bounded to 65,536 nodes and depth below
128. Meshing is capped at 1,048,576 vertices per model and 2,097,152
vertices and 67,108,864 scanned volume cells per prepared scene.

## Verification

All three pinned VOX fixtures, Menger sponge, monument and teapot,
matched Three.js r187 on macOS and iPad Simulator. The 339,175 voxel
records, palette bytes, 829,436 vertex positions and axis normals,
414,718 indexed triangles, and 1,799,505 Metal occupancy texels matched
the reference. The maximum absolute linear RGB difference was
0.00000012. First-frame mesh world transforms also matched. The
fixture run passed 12 test declarations on each target.
[Three.js r187 VOX fixtures](https://github.com/mrdoob/three.js/tree/r187/examples/models/vox)

The committed focused tests exercise multi-model files, default and
custom palettes, greedy face merging, winding, first-frame transforms,
duplicate model references, and malformed input. Metal samples verify
two correctly colored scene instances, their shared renderer, and
occupied and empty 3D texture coordinates.

The Xcode 27.2 MCP timed out resolving the package destination.
Verification used `xcodebuild` with internal DerivedData and Metal API
and shader validation enabled.
