# FF-02: triangle fixture

2026-11-09, implementation working tree based on `e6a7144`. Built in Debug
with Xcode 27.2 (27B52), Swift 6 mode, and the macOS, iOS, and visionOS 27.2
SDKs. The deployment floor is 27.0.

FrostfieldLab links the local package and uses its package-scoped renderer.
The renderer prepares a conventional Metal command queue and a render pipeline
from `Frostfield_Frostfield.bundle/default.metallib`. Xcode compiles
`Triangle.metal` as a package resource. Metal compiler 32023.922 defaults to
MSL 4.1 for this target. Vertices are generated in the shader, so no host
vertex buffer or shared data layout is needed. The fixture draws a single
gradient triangle with an sRGB drawable and no recurring display timer.

| Destination / configuration | Result |
| --- | --- |
| Mac Studio, Apple M4 Max, macOS 27.2 (26B123) | [Centered gradient triangle](triangle/triangle-macos.png) in a 900 × 640 native AppKit view. The executable reports `MACOS` as its build platform. |
| Same macOS build, injected setup failures | Both flags showed “Triangle unavailable” with the expected device or missing vertex function message. |
| iPhone 18 Pro Simulator, iOS 27.2 (24B8117) | [Centered gradient triangle](triangle/triangle-simulator.png). The accessibility hierarchy reports “Multicolored triangle”. |
| Earlier iOS simulator build, `--simulate-no-metal` | [Visible unavailable-device message](triangle/no-metal-simulator.png). |
| Earlier iOS simulator build, `--simulate-bad-shader` | [Visible shader setup error](triangle/bad-shader-simulator.png), naming the missing vertex function. |
| Apple Vision Pro Simulator, visionOS 27.2 (24N120) | [Centered gradient triangle](triangle/triangle-visionos.jpg) in a native visionOS window. |
| iPhone or iPad and Apple Vision Pro hardware | The device SDK builds are compilation evidence. Runtime qualification is pending. |

The Xcode MCP package builds passed. The headless bridge reported a missing local
package product when building the example. `xcodebuild` resolved the package and
built the native apps. The Mac build used ad hoc signing and launched with
`open`. `simctl` installed and launched the simulator builds after the Xcode
device tool failed to launch the iOS lab and rejected the visionOS simulator
destination. The Xcode device tool captured the iOS screenshot and hierarchy.
`simctl io` captured the visionOS window.

The [earlier Catalyst capture](triangle/triangle-mac.png) records a temporary Mac
check of the UIKit host. The current Mac app uses AppKit. The first signed
iPhone build installed successfully. iOS refused launch while the phone was
locked, and the project moved local development to the Mac.

Full lifecycle checks, GPU timing, and hardware qualification for iOS/iPadOS and
visionOS are tracked in [the task list](../../tasks/todo.md).
