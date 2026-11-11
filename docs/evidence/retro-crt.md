# Retro CRT post-processing

Use `PostProcessGraph.retroCRT()` in a `PostProcessPass` or as a graph
effect in a post-process chain. The preset exposes barrel curvature, screen
pixel block size, color levels, Bayer dither, vignette intensity and smoothness,
and scanline intensity, density and speed. Set `ShaderFrame.time` to animate
scanlines. Setting pixel size to one and curvature, color levels, scanline
intensity and vignette intensity to zero gives an exact bypass.

The barrel, scanline and vignette operators follow the
[Three.js r187 CRT functions](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/display/CRT.js).
The preset reuses Frostfield's ordered dither and quantization for the color
reduction in the [r187 retro example](https://github.com/mrdoob/three.js/blob/r187/examples/webgpu_postprocessing_retro.html).
It caps curvature at 0.49 to keep the corner denominator positive and uses
top-left Metal screen coordinates. Pixel blocks sample the warped image at
the block center. Color reduction operates on straight linear RGB and applies
the sampled alpha again before the vignette and scanline gains. This preserves
premultiplied alpha and HDR channel values above one.

This image preset leaves open the geometry-level clip-space snapping and
affine UV interpolation in the
[pinned r187 RetroPassNode](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/tsl/display/RetroPassNode.js).
The example's optional horizontal color bleeding is also outside the preset.

The native Metal tests compare each pixel of two animated 19 × 13 HDR frames
against a CPU reference for the combined operators. They cover nonuniform
coverage, transparent black, preserved HDR, barrel-only samples and an exact
bypass. All 29 focused CRT, graph preset and graph pass tests passed on macOS
and iPhone 18 Pro Simulator with Metal API validation. Xcode MCP reported a
full external DerivedData volume, so verification used Xcode command-line
tests with isolated internal DerivedData.
