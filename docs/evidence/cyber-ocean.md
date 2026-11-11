# Cyber Ocean native reference

FrostfieldLab's `--cyber-ocean` scene uses the original dolphin asset and
swim animation, an original procedural Metal tunnel, 4,096 surface-seeded
flow particles, bloom, chromatic aberration, and seeded grain. Drag or pan
to orbit. Scroll or pinch to zoom. The native Pause and Play control freezes
and resumes simulation. Camera changes can still redraw a paused scene.

| Runtime | Observed |
| --- | --- |
| Apple M4 Max Mac Studio, macOS 27.2 | [Frame 74](scene/cyber-ocean-macos-a.jpg) and [frame 108](scene/cyber-ocean-macos-b.jpg) captured at 1200 × 800 with Metal API Validation. The tail and flipper poses, tunnel phase, and particle positions differ between samples. |
| iPhone 18 Pro Simulator, iOS 27.2 | [Playing](scene/cyber-ocean-iphone-playing.jpg), [paused](scene/cyber-ocean-iphone-paused.jpg), and [orbited while paused](scene/cyber-ocean-iphone-paused-orbited.jpg). Pause held the image for about eight seconds. Orbit updated the view. Play resumed continuous submissions. |
| iPad Pro 13-inch (M5) Simulator, iPadOS 27.2 | The [portrait scene](scene/cyber-ocean-ipad.jpg) rendered with its native playback control. |
| Apple Vision Pro Simulator, visionOS 27.2 | The [windowed scene](scene/cyber-ocean-visionos.jpg) rendered with its native playback control. |

Xcode 27.2 (27B52). Release lab builds passed on macOS, iOS Simulator,
and visionOS Simulator, plus unsigned builds against the iOS and visionOS
device SDKs. The engine batch passed 108 Swift Testing functions in 25 suites.
Four focused follow-up tests covered the portable screen effects and prepared
glTF scene path. Six material tests passed, including preparing an imported
five-map surface and its samplers on the Mac GPU. The headless Xcode build attempted to place DerivedData on
the full external volume, so the final builds used `xcodebuild` with internal
output paths.

The first iPhone Simulator run rejected nonuniform compute threadgroups.
The fixed bloom pass uses ceiling-divided uniform groups and bounds checks.
The simulator emitted an IOSurface notification warning at pause while
rendering and interaction continued. Physical iPhone, iPad, and headset
runtime and sustained presentation timings remain separate qualification.

On the Mac, use the built lab executable to reproduce a frame. The capture
runs the same encode path with a fixed time step and one GPU submission
in flight.

```sh
FrostfieldLab.app/Contents/MacOS/FrostfieldLab --capture-cyber-ocean /tmp/cyber-ocean.png 108
```

The 4,102-vertex, 40-joint Dolphin model is by Alex_Pfe under CC BY 4.0.
The Lab bundles a GLB repacked from Cyber Ocean's Draco geometry using
glTF Transform 4.5.1 and draco3dgltf 1.5.9. The original swim clip lasts
2.29 seconds. The Lab applies its own placement and shading. Particle
seeds come from a centered bind surface. Runtime Draco decoding remains
outside the loader.
[Model attribution](../../Examples/FrostfieldLab/FrostfieldLab/Dolphin-Attribution.txt) · [Cyber Ocean source](https://github.com/SahilK-027/Cyber-Ocean) · [glTF Transform Draco documentation](https://gltf-transform.dev/modules/extensions/classes/KHRDracoMeshCompression)
