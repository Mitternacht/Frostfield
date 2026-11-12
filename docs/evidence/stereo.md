# Stereoscopic effects

`StereoPresentation` covers the three r187 example effects. The caller renders
the scene once for each returned camera, using the existing explicit-view
encode and its viewport. The color combines stop before tone mapping and
output encoding.

Anaglyph eyes share one screen. The screen sits `planeDistance` units in front
of the camera, default 0.5, and the eyes sit half of the separation to either
side of the normalized camera X axis. The default separation is 0.064. Each
view comes from `CameraUtils.frameCorners`. The Dubois matrices use the column
order Three.js uploads to `mat3`. The result clamps to zero through one, and
the alpha is the larger of the two sources.

The barrier and the side-by-side pair use the `StereoCamera` off-axis frustum.
The default focus is 10 and the default zoom is 1, matching the r187
perspective camera. Pass `aspectScale` 1 for the barrier and 0.5 for
side-by-side. Zoom multiplies the horizontal and vertical projection scales.
The eye transform is the camera matrix times a camera-space X translation, so
a scaled camera moves the eye along the scaled X column. Both eyes project a
point at the focus distance on the camera axis to the same horizontal NDC
position.

The barrier selects the left image when `mod(fragmentY, 2) > 1`, using the
GLSL `mod` and a fragment Y in the same space as `gl_FragCoord.y`. Pixel
centers at 0.5 and 1.5 therefore alternate right, then left. Side-by-side does
not crop two full-frame images. Draw the half-aspect left eye into the left
viewport and the right eye into the right viewport.

`StereoPresentationTests` passed with Metal API Validation through `xcodebuild`.
The Xcode bridge was not used. macOS: `Test run with 1 test in 1 suite passed after 0.020 seconds`.
iPad Pro (M5) simulator: `Test run with 1 test in 1 suite passed after 0.026 seconds`.
