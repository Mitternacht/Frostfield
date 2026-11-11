# Raster texture upload settings

Pass `uploadOptions:` to either `TextureLoader.load` overload to choose
row order and alpha association. The existing entry points retain their
straight-alpha output and current row order. Set `flipY: true` to reverse
the decoded rows. In `.color` mode this runs after ImageIO applies the
image orientation. In `.encodedSRGB` and `.linearData` it reverses the
decoded storage rows. Native TGA first normalizes its declared origin.

| Alpha mode | RGB operation |
| --- | --- |
| `.straight` | Retains the existing decoded RGB values |
| `.premultipliedEncoded` | Multiplies decoded 8-bit RGB codes by alpha and rounds to the nearest code |
| `.premultipliedLinear` | Multiplies in shader sampling space. For sRGB textures, decodes the transfer and re-encodes after multiplying |

Alpha itself stays linear and unchanged. Both associated modes clear RGB
at zero alpha. `.linearData` uses RGBA8Unorm, so both associated modes
produce the same stored values in this interpretation. `.color` and
`.encodedSRGB` use RGBA8Unorm sRGB. The consumer chooses an alpha
association and blending mode compatible with those samples.

Set `mipmapped: true` to generate the mip chain from the transformed base
image. Both transforms use the ordinary decoded raster path, including
ImageIO images and the native TGA fallback. OpenEXR, Radiance HDR, DDS,
KTX 1.0, KTX2, and PVR require default upload options. Stored format,
mip, cube, array, and volume rules belong to their corresponding loaders.

Three.js r187 sends `Texture.flipY` and `Texture.premultiplyAlpha` through
WebGL pixel storage or WebGPU external image copy. An actual Chromium
probe of both r187 renderers confirmed that premultiplication multiplies
stored RGB codes even for an sRGB texture. For example, input red 128
with alpha 128 stores red 64. `.premultipliedEncoded` follows that
behavior. `.premultipliedLinear` preserves linear-light association for
sampling, filtering, and compositing.
[Three.js r187 Texture](https://github.com/mrdoob/three.js/blob/r187/src/textures/Texture.js)
[Three.js r187 WebGL texture upload](https://github.com/mrdoob/three.js/blob/r187/src/renderers/webgl/WebGLTextures.js)
[Three.js r187 WebGPU texture upload](https://github.com/mrdoob/three.js/blob/r187/src/renderers/webgpu/utils/WebGPUTextureUtils.js)

## Verification

Both native destinations matched the browser's exact 24-byte reference
and sampled its linear and sRGB output through Metal. Additional tests
sample color-managed linear association, hidden RGB, TGA origin and TIFF
orientation, and a generated mip chain with known associated colors.
The focused run passed 79 tests on macOS and 76 on iPad Simulator. Three
existing simulator tests were skipped for unavailable BC sampling or an
ImageIO gain-map encoder.

Xcode 27.2 MCP timed out resolving the package destination. Verification
used `xcodebuild` with internal DerivedData and Metal API validation.
