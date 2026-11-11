# Programmable scene verification

2026-11-09. Working tree based on `e6a7144`. Xcode 27.2 (27B52), Apple Swift
6.4.2, SDK 27.2. The physical Mac was a Mac Studio with Apple M4 Max and
macOS 27.2 (26B123).

| Check | Observed |
| --- | --- |
| Prepared mesh and perspective camera | The native Mac lab showed a [three-face cube](scene/cube-macos.png). GPU readback checked distinct projected positions and normal colors after camera, transform, and scale changes. |
| Caller shader | The [custom material](scene/custom-material-macos.png) changed tint during rotation. GPU tests checked parameter updates and actionable errors for a missing function and a wrong buffer binding. |
| Lambert lighting and particles | The [Mac effects fixture](scene/combined-effects-macos.png) showed the lit cube and moving fountain. GPU readback checked Lambert intensity and equal seeded particle steps. |
| Color grade and texture output | GPU readback checked an exact four-color bypass and premultiplied alpha. Delayed GPU work verified that the pool held frames through producer and consumer completion. |
| iPhone 18 Pro Simulator, iOS 27.2 | The [expanded scene](scene/combined-effects-ios.png) rendered with Metal API Validation. The cube and particles fit the portrait window. |
| Apple Vision Pro Simulator, visionOS 27.2 | The [expanded scene](scene/combined-effects-visionos.jpg) rendered in a native visionOS window with Metal API Validation. |
| SwiftUI host | The adapter rendered on macOS and both simulators. Rotate Cube changed the pose through accessibility on Mac and touch on iPhone. [Mac](scene/swiftui-mesh-macos-rotated.png) · [iPhone](scene/swiftui-ios-rotated.png) |

The final core test run passed 26 Swift Testing tests in six suites, including
the GPU checks on the M4 Max. The headless Xcode bridge reported a missing local
package product for the lab and returned an incomplete bundle on the last native
test attempt. The final app builds and test run used `xcodebuild` with generated
files on the internal volume. Unsigned Release builds passed against both the
iOS and visionOS device SDKs. The hosted workflow has yet to run.

A separate Swift 6 macOS consumer built both products from the working-tree
package using public imports. It constructed the optional SwiftUI host and
rendered the cube twice offscreen, first with the packaged unlit material and
then with a caller MSL fragment. Both GPU readbacks matched the expected pixels
with Metal API Validation enabled. The native lab separately checked SwiftUI
presentation. This was a local package consumer, with no release tag.

The native simulator runs caught a depthless particle encoder call that passed
a nil state to Metal. The fixed pass omits the depth-state call when it has no
depth attachment. The lab adjusts its camera distance for portrait windows.
The SwiftUI adapter observes the actual UIKit window scene when embedded in
`UIHostingController`. Both simulator fixtures rendered after these fixes.

The eight-second Metal System Trace on the Mac saved successfully and recorded
nominal thermal state. It contained no FrostfieldLab command-buffer submissions
or encoder rows, so it provides no frame pacing, GPU timing, or allocation
baseline. The sustained profile, mobile hardware runs, and full backend
comparisons remain open. The sibling GlassWeave checkout at `181c846` still
contains planning documents for its custom surface and external texture
consumer, so there is no integration fixture to qualify yet.

## Scene graph and picking

The graph composes nested transforms and encodes all prepared meshes in one
color and depth pass. The GPU test checks that a nearer cube occludes a farther
one and that the picker returns the corresponding node at both overlap and
far-only pixels. It also checks a miss on the background.

The batched Xcode run passed 50 test cases, including hierarchy order, invalid
transforms, clip planes, mirrored geometry, and the GPU overlap fixture.
The lab built for macOS, iOS Simulator, and visionOS Simulator. The headless
bridge returned no lab schemes, so the app builds used `xcodebuild`.

On the Mac, clicking the nested right cube highlighted it and clicking the
background restored its original color. Each change submitted one redraw.
An iPhone 18 Pro simulator tap highlighted the center cube.
[Mac selection capture](scene/graph-selected-macos.png)

## Asset and material fixtures

The Xcode package batch passed 86 tests after adding image textures, OBJ and
static glTF geometry, transform keyframes, directional shadows, and
metallic-roughness shading. The GPU tests check the texture's top-left UV and
premultiplied tint, a cast shadow that darkens a receiver outside the caster's
camera silhouette, and the response of polished and rough metallic surfaces.

The Mac lab displayed the [textured OBJ scene](scene/asset-obj-macos.png)
and the [metallic-roughness scene](scene/pbr-macos.png). The directional
shadow preview showed stippling on tilted faces with a constant depth bias.
A bounded texel bias based on the receiver angle removed the visible pattern
while keeping the [cast shadows on the floor](scene/shadows-macos.png).
Metal API Validation was enabled for the updated shadow preview.

The updated SwiftUI adapter displayed the mesh using `CAMetalDisplayLink`.
The native button updated the pose through accessibility with Metal API
Validation enabled. [Mac capture](scene/swiftui-display-link-macos.png)

The integrated Xcode run passed 83 Swift Testing functions across 17 suites,
including parameterized cases. It checked static glTF node transforms and
material import, plane and sphere winding, grazing-angle shadow bias,
shadowing on Lambert and PBR receivers, and reversed-depth viewport occlusion.
An earlier attempt found two reference expectation errors. The corrected
assertions compare the actual unlit red intensity and allow float rounding
after glTF matrix decomposition.

A later focused glTF run passed 13 test functions. It includes a real
symlink fixture that verifies percent-decoded filenames containing a slash
and a combining character are rejected before opening an adjacent file.

The final lab built in Release for native macOS, iOS Simulator, and visionOS
Simulator. Unsigned Release builds passed against the iOS and visionOS device
SDKs. The package tests and app builds used `xcodebuild` after the external
volume rejected writes and the headless bridge required UI approval to open
the internal recovery checkout.

The iPhone 18 Pro simulator on iOS 27.2 rendered the [imported glTF scene](scene/gltf-scene-iphone.png)
with its embedded base-color texture and PBR factors, and the [procedural spheres and shadows](scene/pbr-shadow-scene-iphone.png).
The SwiftUI graph host rendered two nested cubes on the
[iPhone simulator](scene/swiftui-graph-iphone.png) and the
[iPad Pro 13-inch (M5) simulator](scene/swiftui-graph-ipad.png).
The [visionOS window fixture](scene/vision-window.png) rendered the scene graph
on Apple Vision Pro Simulator. Metal API Validation was enabled for these launches.

## visionOS immersive preview

On Apple Vision Pro Simulator with visionOS 27.2, a Debug build opened the
immersive space through the launcher's open action, dismissed the launcher,
and submitted one compositor view. The [immersive capture](scene/immersive-visionos.png)
shows the cube rendered into that view. Each view uses its compositor
projection, predicted device pose when available, a dedicated target, and
reversed depth. Interactive dismissal and launcher restoration, stereo,
tracking, and comfort still need physical Vision Pro coverage.
[Apple Compositor Services guide](https://developer.apple.com/documentation/compositorservices/drawing-fully-immersive-content-using-metal)
