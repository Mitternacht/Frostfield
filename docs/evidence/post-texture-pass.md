# Full-screen texture pass

`PostProcessTexturePass` draws a color texture across the working frame as
a standalone effect or a `PostProcessChain.Effect.pass` stage. Opacity one
replaces the incoming color and alpha. For opacity below one, the pass
scales the premultiplied RGBA by that opacity and composites it over the
incoming frame with source-over. Zero preserves the input. The pass uses
a separate output attachment and leaves scene depth available unchanged.
[Three.js TexturePass r187](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/postprocessing/TexturePass.js),
[Three.js CopyShader r187](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/shaders/CopyShader.js)

The overlay can differ in size from the working frame. It uses top-left
normalized coordinates, bilinear filtering, edge clamp and mip level
zero. Use `.premultiplied` for prepared linear HDR frames and `.straight`
for straight-alpha color textures. Straight-alpha RGB is multiplied by
alpha before interpolation to keep transparent texels from tinting an
edge. Metal decodes sRGB textures to linear RGB. A linear RGBA16Float
destination retains HDR and alpha for subsequent stages. A chain mask
can limit which pixels the texture stage changes, and a save stage can
capture the whole masked result.

## Verification

Five focused suites passed 18 tests each on arm64 macOS and iPhone 18 Pro
Simulator with Metal validation enabled. Readbacks verified HDR
replacement and source-over, sRGB and straight-alpha resizing in both
axes, preserved depth through a masked stage followed by a save tap,
and rejection of destination texture views that share storage. The
prepared pipeline contract also ran against the straight-alpha path.

The Xcode MCP could not allocate DerivedData on the full external
volume. The focused runs used `xcodebuild` with internal caches.
