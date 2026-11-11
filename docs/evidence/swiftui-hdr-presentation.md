# SwiftUI HDR presentation

Prepare the scene's meshes for RGBA16Float and attach an `HDRDisplay`
on the same device using `FrostfieldSceneView.hdrDisplay(_:settings:)`.
The view renders into a reusable, size-matched HDR texture and encodes
the display pass before presenting. An RGBA16Float drawable uses the
display's extended linear color profile and enables extended dynamic
range on the layer. A BGRA8 drawable uses its matching SDR profile.

Pass the available display headroom as it changes. Exposure and other
settings invalidate the existing host when the drawable format is
stable. The source and drawable share one retained command buffer.
The host clears a standalone mesh's depth attachment according to the
renderer’s depth mode.

On iPhone 18 Pro Simulator, a 160×96-point view drew to a 480×288
RGBA16Float drawable. The layer reported extended linear Display P3
and EDR enabled. Changing exposure from −2 to +1 EV kept the same
layer and dimensions and changed 2,704 pixels in the view. The probe
used internal DerivedData because Xcode's configured external build
volume was full. Apple's device session captured the frames and
exercised the exposure control. The observation covers simulator
presentation and redraw. Physical-device luminance remains open.

The Mac and iPhone Simulator package runs each passed 42 focused tests
with Metal API Validation, including HDR display, tone mapping,
framebuffer clears, scheduling and drawable resolution.

[Wide-gamut output](wide-gamut-output.md)
· [Apple extended dynamic range](https://developer.apple.com/documentation/quartzcore/cametallayer/wantsextendeddynamicrangecontent)
