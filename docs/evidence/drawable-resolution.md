# SwiftUI drawable resolution

Use `FrostfieldSceneView.drawableResolution(_:)` with a finite positive
`DrawableResolution(pixelsPerPoint:)` to choose the number of rendered pixels
per SwiftUI layout point. The default `.display` follows the window's
backing resolution. SwiftUI's ordinary frame and layout modifiers choose
the view's logical size. The render target rounds each axis up to a whole pixel.

`onDrawableSizeChange(_:)` reports the dimensions of the first completed
drawable and later size changes. Rapid changes can coalesce before a frame
completes. The coordinator uses the existing Metal host and resizes its
depth texture to the acquired color drawable. Oversized drawable requests
reach the existing error callback. Zero-sized views wait for valid layout.

The SwiftUI package tests changed resolution on a real `CAMetalLayer`,
checked fixed and display-backed sizes, checked the layer's logical bounds
and validated input limits. They also acquired a real Metal drawable and
checked both texture dimensions. The Mac and iPhone 18 Pro Simulator runs
each passed all three SwiftUI tests plus seven framebuffer and scheduling
regressions with Metal API Validation. The package bridge timed out
resolving the run destination, so the tests used `xcodebuild`.

In a live iPhone 18 Pro Simulator Lab probe, the same 160×96-point view
reported completed drawables of 80×48 and 200×120 pixels after its ratio
changed from 0.5 to 1.25. A rotation then changed 2,369 pixels inside the
Metal surface. The configured external Xcode build volume was full, so
the probe built with internal DerivedData. Apple's device session
inspected the view and exercised the rotation control.

[Three.js r187 renderer sizing](https://github.com/mrdoob/three.js/blob/r187/src/renderers/common/Renderer.js)
· [Metal drawable size](https://developer.apple.com/documentation/quartzcore/cametallayer/drawablesize)
