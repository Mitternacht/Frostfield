# FF-03 render lifecycle evidence

## Initial MTKView host

2026-11-09. Working tree based on `e6a7144`. Mac Studio with Apple M4 Max,
macOS 27.2 (26B123), Xcode 27.2 (27B52), Debug, conventional Metal command queue.

Each lab window owns its schedule. The host derives drawable size from AppKit
backing coordinates or UIKit view bounds and display scale. It pauses when
inactive and permits one submitted frame until the Metal command buffer completes.
A missing drawable gets one delayed retry per on-demand invalidation. Continuous
mode retries on a later display tick. A command buffer error stops rendering and
shows the existing failure view.

| Mac check | Observed |
| --- | --- |
| On demand | The count held at 2 through 0.9 seconds of idle time. Resizing the drawable from 900 × 608 to 740 × 488 submitted frame 3. [Resized triangle](triangle/triangle-macos-resized.png) |
| Continuous | The count stayed at 63 for more than a second while minimized. After restoring the window, it reached 120. |
| Two windows | The first window paused at 37 with a 900 × 608 drawable. The second used a 740 × 488 drawable after resizing and paused at 76 when minimized. The first resumed from 37 and kept its original size. |

These counter checks ran before the bounded drawable retry and GPU error handling
were added. The final macOS binary displayed the centered triangle. The native
Xcode test run passed all five scheduling tests. The final lab built for macOS
and rendered on launch in the iPhone 18 Pro (iOS 27.2) and Apple Vision Pro
(visionOS 27.2) simulators. The headless Xcode bridge reported a
missing local package product for the lab, so its builds used `xcodebuild`.
The hosted CI run is pending.

This implementation bounded submissions owned by the lab. `MTKView` could still
wait internally while acquiring a drawable.
[Apple's drawable acquisition documentation](https://developer.apple.com/documentation/quartzcore/cametallayer/nextdrawable())

## CAMetalDisplayLink host

2026-11-09. The lab's AppKit and UIKit hosts now use the drawable supplied by
`CAMetalDisplayLink.Update`. Neither host calls a synchronous drawable
acquisition API. Each window keeps one submitted frame in flight, pauses the
link when inactive or after an on-demand request, and resumes on invalidation
or completion as needed. Layout and backing scale set the expected drawable
size. An update with an older size is skipped while the new frame is pending.

The port built in Release for Apple silicon macOS and iOS Simulator with
Xcode 27.2 (27B52). Both builds also passed after adding the PBR and glTF lab
fixtures. Before those additions, an iPhone 18 Pro simulator on iOS 27.2
displayed both display-link fixtures at 1206 × 2622 pixels. These captures show the rendered content at
one moment. They provide no presentation intervals or submission counters.

| Simulator launch | Capture |
| --- | --- |
| Default, on demand | [Gradient triangle](triangle/triangle-ios-display-link.png) |
| `--lit --particles --grade --continuous` | [Lit cube, particles, and color grade](scene/effects-ios-display-link.png) |

The Mac console was locked during this batch. Foreground checks for the new
Mac host, mobile resize and lifecycle coverage, a Mac display-scale change, and
a GPU trace under drawable pressure remain open. The earlier Mac counter checks
above cover the MTKView host.

[Apple's CAMetalDisplayLink documentation](https://developer.apple.com/documentation/quartzcore/cametaldisplaylink)
