# Custom mesh multiple render targets

`MeshRenderer(device:colorPixelFormats:geometry:material:…)` prepares an
unblended custom Metal fragment with two to eight contiguous color attachment
slots. Each slot has its own `MTLPixelFormat`. The shader declares matching
`[[color(i)]]` outputs and uses the existing read-only `float4` material binding
at buffer zero. The caller supplies the textures, clear values and load/store
actions in the same attachment order. Slot zero still supplies the viewport
dimensions and the existing `colorPixelFormat` property.

Standalone and scene graph draws use the same layout validator. Each MRT pass
uses distinct persistent, renderable, equal-sized, single-sample 2D textures
at mip and slice zero with a matching Depth32Float attachment. Tests read back
the HDR color, integer object ID, float scalar and depth from a real Metal
pass. They check independent clears, depth overlap, hidden meshes, incorrect
format order, holes, bad dimensions and usage, aliases, and invalid mixtures.
The ordinary single-target attribute pass can still emit the mesh's generic
geometry attributes.

Xcode 27.2 ran all 60 focused tests on arm64 macOS and on iPad Pro 13-inch
(M5) Simulator with Metal API Validation enabled. The run included the five
MRT fixtures plus mesh rendering, render state, scene graph, texture transfer,
surface attributes and indirect draw regressions.

Three.js r187 allocates separate ordered textures with `RenderTarget.count`.
`PassNode.setMRT` binds a named output node that resolves targets by texture
name. This slice exposes explicit Metal indices. Named shader-graph outputs,
per-target blending, MRT multisampling and resolves, and mixtures with stock
or primitive pipelines remain open.

[Three.js RenderTarget](https://github.com/mrdoob/three.js/blob/r187/src/core/RenderTarget.js) · [Three.js PassNode](https://github.com/mrdoob/three.js/blob/r187/src/nodes/display/PassNode.js) · [Three.js MRTNode](https://github.com/mrdoob/three.js/blob/r187/src/nodes/core/MRTNode.js)

[Metal pass setup](https://developer.apple.com/documentation/metal/customizing-render-pass-setup) · [Metal color-renderable format limits](https://developer.apple.com/documentation/metal/understanding-color-renderable-pixel-format-sizes)
