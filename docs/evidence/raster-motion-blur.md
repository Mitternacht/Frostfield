# Screen-space motion blur

Create `MotionBlur(device:motionTexture:settings:)` and encode it directly or
use `.pass(blur)` in a `PostProcessChain`. Set `motionTexture` for the current
view before encoding. Color and motion use shader-readable RGBA16Float
textures of the same size. Motion RG contains previous-minus-current motion
in top-left pixels, B contains the previous stored depth, and A is one when
history is valid. The effect reads RG and A. Encode the blur after color and
motion production on an ordered queue, with a separate destination.

The effect uses the reference's center sample and tap positions along the
motion vector. It divides by the full sample count including the center, so
constant premultiplied HDR frames retain their brightness and coverage. Taps
read level-zero color bilinearly and clamp at the frame edge. Set
`sampleCount` between 2 and 64 to control the number of moving taps and
`shutterScale` between zero and four to scale the vector. Zero shutter, zero
motion and invalid history copy the source texel exactly.
[Three.js r187 MotionBlur](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/display/MotionBlur.js)

Set `jitter` from zero to one to shift the tap span by two blue-noise ranks
scaled by the respective motion components. Zero retains the existing
regular taps. The renderer reuses Frostfield's periodic 64×64 blue-noise
rank field and derives two spatial phases from the caller's `frameSeed`.
Reusing a seed repeats the same sample positions. Advance it with the frame
when temporal variation is desired. Set `shutterScale` explicitly to author
the exposure span from the application's timing. The effect keeps its
premultiplied HDR averaging and exact invalid-vector behavior with jitter.
[realism-effects motion blur at 061daea](https://github.com/0beqz/realism-effects/blob/061daeaefb7773d8a1a8e10fa95a184fbcda1fb7/src/motion-blur/shader/motion_blur.frag) ·
[Frame update](https://github.com/0beqz/realism-effects/blob/061daeaefb7773d8a1a8e10fa95a184fbcda1fb7/src/motion-blur/MotionBlurEffect.js)

Native Metal tests exercise horizontal and vertical impulse streaks,
constant-frame energy, fractional and invalid authored vectors, edge clamp,
mixed coverage, signed HDR, the chain API and input validation. The seeded
fixture verifies both rank-derived axes, repeatability, frame-seed changes
and exact invalid, constant and zero-shutter output.

The separate regular shader preserves zero jitter bit for bit. A Mac GPU
comparison against the original shader matched 184,320 half-float channels
across five tap counts and four shutter scales with signed HDR and mixed
validity. All 33 focused Swift Testing declarations passed on macOS and
iPhone 18 Pro Simulator with Metal API validation. Xcode MCP timed out
resolving the package destination, so verification used `xcodebuild`.
