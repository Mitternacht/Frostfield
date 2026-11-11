# Framebuffer clear control

Apply `FramebufferClear` to a caller-owned Metal render pass to set the load
actions for color, depth and stencil independently. Encode a standalone clear
before a later draw with `encode(in:commandBuffer:)`. The standalone pass
preserves the selected attachments' mip levels, slices and store or resolve
actions. Each clear covers the selected attachment subresource.

`FrostfieldSceneView.clearColor(_:)` sets the premultiplied linear drawable
clear color. A change invalidates the current host's frame and updates its native
layer opacity. The host initializes every acquired drawable.

The Metal tests read back color and depth and use a stencil comparison draw to
check the stored stencil value. They also verify that unselected attachments
keep their contents, that standalone encoding preserves the caller's descriptor,
and that invalid selections fail before changing load actions.

With Metal API Validation, the focused package run passed 48 tests in seven
suites on macOS and 47 tests on iPhone 18 Pro Simulator, which skipped the
native cube-array capability case. Both builds compiled the SwiftUI host.
These tests exercise offscreen Metal clears.

[Three.js r187 WebGLRenderer](https://github.com/mrdoob/three.js/blob/r187/src/renderers/WebGLRenderer.js)
· [Metal render pass load action](https://developer.apple.com/documentation/metal/mtlrenderpassattachmentdescriptor/loadaction)
