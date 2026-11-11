# Depth-picked camera autofocus

Use `CameraLens.autofocused(from:at:camera:depthMode:commandQueue:)` to focus
a lens on a normalized top-left screen coordinate. It asynchronously reads
one Depth32Float texel and reconstructs the view-axis distance for the
perspective camera and depth direction that rendered it. The returned lens
preserves the authored optics and can be assigned to
`BokehDepthOfField.lens` before a later render.

Commit the depth producer before calling the helper on the same Metal queue.
For producers on other queues, synchronize before reading. Cleared pixels
return nil. Invalid coordinates, textures and unsupported logarithmic depth
throw. A target inside the physical focal length uses the existing lens
validation.

The pinned [Three.js r187 BokehShader2](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/shaders/BokehShader2.js)
selects its focal depth from the depth texture at an authored screen
coordinate. Frostfield exposes that operation as an explicit one-texel
readback for native camera interaction.

The GPU fixture renders two targets at known view distances, reads back each
from an off-center coordinate in forward and reversed depth, and checks that
the result sets the circle of confusion to zero. The tests also cover clear
pixels, invalid sampling and physical lens validation.

All 21 focused autofocus, lens, bokeh, tilt-shift and texture-transfer tests
passed on macOS and iPhone 18 Pro Simulator with Metal API validation. Xcode MCP
reported a full external DerivedData volume, so verification used Xcode
command-line tests with isolated internal DerivedData.
