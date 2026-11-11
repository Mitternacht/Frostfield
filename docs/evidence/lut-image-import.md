# Image-based 3D lookup tables

`ColorLookupTable(image:)` and `ColorLookupTable(imageContentsOf:)` load a
horizontal or vertical strip of square blue slices. The file initializer
`ColorLookupTable(contentsOf:)` recognizes lookup-strip image extensions
alongside `.cube` and `.3dl`. The smaller dimension is the lattice size N
and the larger dimension must be N². The existing size range is 2–65.
Red increases across each square, green down it, and blue by square.
[Three.js r187 LUTImageLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/LUTImageLoader.js)

The initializer reads source rows top-to-bottom by default. Set
`flipVertically: true` to reverse source rows before extracting slices.
For a horizontal strip this reverses green within each slice. For a
vertical strip it reverses green and the order of blue slices. A browser
probe reproduced r187's horizontal flip dropping the first slice and
adding a blank slice. Frostfield's flip preserves all samples.

The image's decoded code values map to the unit-domain table by dividing
by 255. Color profiles, transfer tags, and EXIF orientation do not affect
the mapping. The lossless path accepts static, noninterlaced PNG with
8-bit grayscale, RGB, grayscale with alpha, RGBA, or palette indices.
Grayscale replicates into RGB and PNG alpha is ignored. Other ImageIO
formats are accepted when they decode to a single opaque 8-bit RGB or
grayscale image. Image data is bounded to 64 MiB and dimensions are
validated before decoding the full image.
[Portable Network Graphics specification](https://www.w3.org/TR/png-3/)
[Core Graphics color-space tagging](https://developer.apple.com/documentation/coregraphics/cgimage/copy(colorspace:))

Tiled square LUTs, interlaced PNG, and higher-bit-depth image LUTs are
outside this implementation. The r187 cube loader requires a 3D size and
the 3DL loader creates a 3D lattice. The image loader also creates a 3D
texture, so 1D tables fall outside these loader capabilities.
[Three.js r187 LUTCubeLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/LUTCubeLoader.js)
[Three.js r187 LUT3dlLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/LUT3dlLoader.js)

## Verification

Headless Chrome ran the pinned r187 image loader on the three demo PNGs.
Frostfield matched every RGB code at all 32³ points in each table on macOS
and iPad Simulator. All five demo `.cube` files and the demo `.3dl` loaded
as 32³ tables. Seven Metal grading samples per image matched across
destinations and agreed with CPU trilinear evaluation within 0.001.
[Three.js r187 LUT demo](https://github.com/mrdoob/three.js/blob/r187/examples/webgl_postprocessing_3dlut.html)

The focused tests cover both strip orientations and flipping, exact RGB
beneath low and zero PNG alpha, grayscale and indexed values, a Display
P3 TIFF, invalid dimensions and corrupt pixels, and actual Metal grading
through both interpolation modes. The final run passed 29 tests on macOS
and iPad Simulator, including PNG texture, text LUT, and grading
regressions. Xcode 27.2 MCP timed out resolving the package destination. Verification used `xcodebuild` with internal DerivedData
and Metal API validation.
