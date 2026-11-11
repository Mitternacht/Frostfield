# Layered raster render targets

`SceneGraph.encode` and `MeshRenderer.encode` draw directly into caller-owned
Metal render subresources. Allocate a renderable texture with
`MTLTextureUsage.renderTarget` and select the mip with the color attachment's
`level`. Select a 2D array layer, cube face, or cube-array face with `slice`.
For a 3D texture use `depthPlane`, with slice zero. A volume's available
planes shrink at smaller mips, and the length of a 2D array stays fixed.
Use a Depth32Float subresource with matching mip dimensions and sample count.
[Metal render pass attachments](https://developer.apple.com/documentation/metal/mtlrenderpassattachmentdescriptor) ·
[Three.js r187 layer and mip binding](https://github.com/mrdoob/three.js/blob/r187/src/renderers/WebGLRenderer.js)

Each encode targets one plane or face per attachment. The selected extent
drives default projection, graph screen coordinates, viewport and scissor.
Different targets in a custom multi-output pass can select different layers.
Use separate storage and the color formats of the prepared pipeline. Standard
2D multisampling still resolves through the existing path. Layered subresources
use one sample.

Raster transmission blits the selected RGBA16Float layer and mip into a
separate mipmapped 2D backdrop, then blends into that same layer. Backdrop
dimensions must match the selected subresource. A neighboring layer stays
outside the copy and overlay. Cubes accept device-renderable formats through
ordinary mesh pipelines. The existing `CubeCamera` HDR capture and environment
conversion keep their public behavior.

## Verification

Metal readbacks verify mip one of a 2D array, volume, cube and cube array,
an RGBA8 cube face with array depth, out-of-range selectors, and a custom
three-output pass into independent HDR array, integer 2D and float volume
textures. Raster transmission fixtures read the selected mip, the copied
backdrop, and a neighboring layer.

The 54 focused tests passed on Apple Silicon Mac. On iPhone 18 Pro Simulator,
53 passed and the native cube-array case was skipped because that simulator
does not expose cube arrays. The 2D array, volume and cube cases ran on both
targets. The Xcode bridge timed out resolving the package destination, so
verification used Xcode 27.2 `xcodebuild` with Metal API Validation.
[Metal feature set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
