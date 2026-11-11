# Texture-driven digital glitch

Use `PostProcessGraph.glitchWithDisplacement()` and bind a single-level linear
texture as `"Displacement"`. The red channel supplies the displacement, with
nearest filtering and clamped addressing. Use a custom map or
`GlitchRandomizer.makeDisplacementTexture(device:size:seed:)` for deterministic
R32Float noise. The generated map has its own seed, independent of frame scheduling.

The graph follows the [Three.js r187 DigitalGlitch shader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/shaders/DigitalGlitch.js)
for the seed-squared map coordinates, conditional row and column tears,
displacement, RGB shift and snow. It converts between the pinned bottom-up
coordinates and Metal's top-left texture coordinates. For a premultiplied
Frostfield frame, it applies the warped center pixel's coverage to the shifted
channels and snow and retains its alpha. `"SnowStrength"` defaults to one and
zero disables snow.

Drive the pass with `GlitchRandomizer.advance` on the graph uniforms. The
existing seeded schedule produces a strong burst, weaker frames for about the
first fifth of the 120–239-frame interval and then bypass. `goWild` applies
strong settings on every frame. The [pinned GlitchPass](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/postprocessing/GlitchPass.js)
uses an unseeded draw with an inclusive 240-frame limit. The existing hash graph
and native seeded schedule retain their behavior.

The GPU tests compare an authored nonuniform displacement map to an independent
CPU implementation for both signs of the tear branches. They also exercise
seeded map replay, RGB splitting, opacity, snow and exact bypass. A 600-frame
uniform test checks both graph modes share the same intermittent schedule.

All 18 focused glitch, graph-preset and screen-effect tests passed on macOS
and the iPhone 18 Pro Simulator with Metal API validation. Xcode MCP reported
a full external DerivedData volume, so verification used Xcode command-line
tests with isolated internal DerivedData.
