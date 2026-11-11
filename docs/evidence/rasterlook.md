# Raster image look: tone mapping and anti-aliasing

## Tone mapping

`HDRDisplay` maps exposed scene-linear color to display-referred color in one
full-screen raster pass. Every operator receives straight (unpremultiplied),
nonnegative linear sRGB multiplied by `2^exposureStops`, and the result is
premultiplied again.

| Three.js | Frostfield | Formula source | Verified by |
| --- | --- | --- | --- |
| `NoToneMapping` | `HDRDisplay.ToneMapping.none` | identity, clamped to 0...65,504 | `ToneMappingTests` |
| `LinearToneMapping` | `.linear` | three.js r187 `tonemapping_pars_fragment` | `ToneMappingTests` |
| `ReinhardToneMapping` | `.reinhard` | three.js r187 | `ToneMappingTests` |
| — | `.reinhardExtended(whitePoint:)` | Reinhard et al. 2002, equation 4, per channel | `ToneMappingTests` |
| `CineonToneMapping` | `.cineon` | three.js r187 (Hejl and Burgess-Dawson) | `ToneMappingTests` |
| `ACESFilmicToneMapping` | `.acesFilmic` | three.js r187 (Hill's RRT+ODT fit, with the 1/0.6 exposure) | `ToneMappingTests` |
| — | `.acesNarkowicz` | Narkowicz 2016 fit | `ToneMappingTests` |
| `AgXToneMapping` | `.agx()` | three.js r187 (Blender Rec.2020 matrices via Filament, sixth-order sigmoid) | `ToneMappingTests` |
| — | `.agx(look: .punchy)`, `.agx(look: .golden)` | ASC CDL values from sobotka/AgX `config.ocio` | `ToneMappingTests` |
| `NeutralToneMapping` | `.neutral` | Khronos PBR Neutral, as three.js r187 | `ToneMappingTests`, `HDRDisplayTests` |
| `CustomToneMapping` | `PostProcessGraph.toneMapping(exposureStops:curve:)` run by `PostProcessPass` | caller's curve | `ToneMappingTests` |
| `toneMappingExposure` | `HDRDisplay.Settings.exposureStops` (multiplier `2^stops`) | — | `ToneMappingTests` |
| `OutputPass` in `EffectComposer` | `PostProcessChain.Effect.toneMapping` | — | `FXAATests` |

The tests evaluate each curve on the CPU in double precision, transliterated
line by line from its source, at 20 colors: a gray ramp from 0 to 120 and
saturated colors such as (4, 2, 1), (10, 0.5, 0.1) and (60, 30, 2). The GPU
result, read back from RGBA16Float, matches within 1e-3 for every operator and
look; the largest difference is 4.9e-4, one half-float step just below 1.
Three.js's AgX maps middle gray 0.18 to 0.2145, and the Narkowicz fit maps 1 to
0.8038; both values are checked.

**Extended range.** `Settings.displayHeadroom` (1 to 64, default 1) is the peak
the destination can show as a multiple of SDR white. At 1 every operator except
`.none` writes 0...1 and is exactly its reference formula. Above 1 the
destination must be RGBA16Float presented with extended range
(`CAMetalLayer.wantsExtendedDynamicRangeContent` and an extended linear sRGB color
space), and the caller passes the screen's current headroom
(`NSScreen.maximumExtendedDynamicRangeColorComponentValue`,
`UIScreen.currentEDRHeadroom`). An 8-bit destination with a headroom above 1 is
rejected before encoding.

The headroom adds range to highlights only, so shadows and mid-tones do not move
when the headroom follows the screen's brightness:

- `.none` never clamps; `.linear` clamps to H.
- `.neutral` keeps its output below the 0.76 shoulder start and aims the PBR
  Neutral shoulder at H instead of 1.
- Every other curve T keeps its SDR output wherever all exposed scene channels are
  at most 1 (exposed diffuse white). Above that, with m the largest exposed scene
  channel and w the curve's largest output at exposed white (1, 1, 1), the
  pixel's largest display channel rises by
  `E = (H − 1) Δ² / ((Δ + H − w)(Δ + 1 − w))`, `Δ = w (m − 1)`, and RGB scales
  with it, so the SDR curve's hue and saturation are kept. E is the difference
  between two PBR Neutral shoulders leaving the knee at the curve's gain w, one
  approaching H and one approaching 1. It is 0 with zero slope at the knee, rises
  with m and approaches H − 1, so gray maps to T(x) + E < H.
- w is evaluated by the vertex stage (three times per draw) and passed flat to the
  fragments, so a headroom costs no extra time per pixel.

GPU readback of a gray ramp (RGBA16Float, exposure 0):

| Curve | 0.18, any H | 1, any H | 4 at H = 1 / 4 / 16 | 64 at H = 1 / 4 / 16 |
| --- | --- | --- | --- | --- |
| `.acesFilmic` | 0.208 | 0.763 | 0.952 / 2.08 / 2.73 | 1.000 / 3.80 / 12.3 |
| `.cineon` | 0.221 | 0.683 | 0.902 / 1.89 / 2.44 | 0.993 / 3.76 / 12.0 |
| `.acesNarkowicz` | 0.261 | 0.804 | 0.973 / 2.17 / 2.87 | 1.000 / 3.81 / 12.5 |
| `.agx()` | 0.211 | 0.590 | 0.861 / 1.69 / 2.12 | 0.997 / 3.71 / 11.5 |
| `.reinhard` | 0.150 | 0.500 | 0.800 / 1.47 / 1.79 | 0.984 / 3.64 / 10.9 |
| `.neutral` | 0.137 | 0.869 / 0.948 / 0.957 (above its 0.76 knee) | 0.983 / 2.37 / 3.40 | 0.999 / 3.84 / 13.0 |

(The ramp sample nearest 0.18 is 0.177.) The earlier rule, `H · T(x / H)`, is the
same as lowering exposure by log2 H stops before the curve, so it darkened
mid-tones: ACES filmic mapped 0.18 to 0.047 at H = 16 instead of 0.213, Cineon to
0.026 instead of 0.225 and AgX to 0.117 instead of 0.215.

`ToneMappingTests` checks this four ways. Every operator at H = 2 and H = 16
matches the CPU reference of the rule at 20 colors within 1e-3 relative. Colors
whose channels are all at most 1 (0.76 for Neutral) are bit-identical to SDR at
H = 1.5, 4, 16 and 64, and middle gray at H = 16 equals its SDR value. A
quarter-stop gray ramp from 1/64 to 4,096 at H = 16 never decreases, stays at or
below H and exceeds 0.95 H at the top. For (10, 0.5, 0.1), (0.02, 0.5, 3) and
(60, 30, 2), channel ratios match the SDR curve within 2e-3 while the brightest
channel is more than 1.5 times its SDR value.

**Extended Reinhard white point.** The CPU computes 1 / W² in double precision
and limits it to the largest finite float; the shader evaluates
`x (1 + x / W²) / (1 + x)` with that constant, so zero channels stay zero for any
positive W. With W = 1e-20, the smallest normal float and the smallest subnormal
float, (0, 0, 0) stays black and (0, 0.5, 0), (2, 0, 0.001) and (60000, 0, 1)
clip channel by channel; with the largest finite W the output is bit-identical to
`.reinhard`. Before, W below about 1e-19 made 1 / W² infinite in the shader and
turned black into white.

**Custom curves.** The graph hook unpremultiplies, applies the "ExposureStops"
uniform, calls the caller's curve and premultiplies again. Reinhard written as a
graph matches the built-in operator within 1e-3, and Hable's Uncharted 2 curve
(not in Three.js) matches its CPU formula within 1.5e-3. The curve's output is not
clamped, so it can use EDR headroom.

Deliberate breakages caught: changing the ACES 1/0.6 exposure to 1/0.65, Golden's
saturation from 1.3 to 0.8 (the iolite value), Cineon's output power from 2.2 to
2.0, Neutral's shoulder peak, the earlier `H · T(x / H)` headroom rule (345 issues
across three tests), adding the highlight extension per channel instead of
scaling RGB, and computing 1 / W² in single precision without the limit.

## Anti-aliasing

| Three.js | Frostfield | Path | Verified by |
| --- | --- | --- | --- |
| `FXAAPass`, `fxaa` node | `FXAA`, `PostProcessChain.Effect.fxaa` | two raster passes: luma to R16Float, then FXAA 3.11 quality | `FXAATests` |
| `SMAAPass`, `smaa` node | `SMAA`, `PostProcessChain.Effect.smaa` | three raster passes: edges, weights, blending | `SMAATests` |
| `traa` node (reprojecting TAA) | `TemporalAntiAliasing`, `SceneGraph.projectionJitter` | one raster pass writing history and output | `TemporalAntiAliasingTests` |
| `TAARenderPass` with `accumulate` | `TemporalAntiAliasing(settings: .accumulation)` | the same pass with equal weights, no clipping and RGBA32Float history | `TemporalAntiAliasingTests` |
| `SSAARenderPass` | partial: `reset()`, then render and encode N jittered frames with `.accumulation` in one frame | — | `TemporalAntiAliasingTests` (the accumulation itself) |
| `WebGLRenderer({ antialias: true })` (MSAA) | not implemented | — | — |

**FXAA** ports the quality path of FXAA 3.11 with presets 12, 29 and 39 and
Lottes's subpixel, edge threshold and minimum threshold parameters. Luma is the
square root of Rec.709 luminance, written by a first pass so premultiplied alpha
passes through the edge filter unchanged. The edge-end search and the preset step
tables are checked against closed-form values: on a horizontal edge that steps
down one row at x = 64, with sub-pixel blending off, the dark pixel d columns
past the step blends with the white pixel below by `0.5 − s / (s + R)`, where s
is the first sampled search distance beyond d + 0.5 and R the preset's full reach
(0 when no sample reaches it). For d = 3, 7, 11, 15 and 21 all three presets
match within 6e-3 (bilinear filter precision); preset 12 leaves d ≥ 11 unblended
because it samples only to 8.5 pixels, and preset 39 leaves d = 21 unblended.
On the rendered edges, FXAA with sub-pixel blending off lowers the error from
0.117 to 0.057 (14.3°) and from 0.103 to 0.075 (40.1°).

**SMAA** ports SMAA.hlsl for SMAA 1x with luma edge detection on sRGB-encoded
luma, local contrast adaptation, diagonal and corner detection, and the four
presets. Values above 1 continue on the sRGB curve, as in Core Graphics' extended
sRGB transfer function, so edges between EDR highlights are detected: on a 14.3°
staircase between display values 2 and 6, SMAA changes 92 pixels and lowers the
error against the supersampled staircase from 0.405 to 0.152 (it changed none when
luma was clamped to 1). Diagonal detection: on the 40.1° edge, high quality reaches
0.026 against 0.052 for medium, which has the same threshold but no diagonal or
corner search. Corner detection: on a pixel-aligned 24 × 24 white square, which a
supersampled render reproduces exactly, medium changes 92 pixels by 22.1 in total
and high by 6.0, about the quarter that SMAA.hlsl's 25% corner rounding keeps.

The area and search tables are generated in Swift by a port of `AreaTex.py` and
`SearchTex.py`. For subsample offset 0 the generated 160 × 80
RG8 area table and 64 × 16 R8 search table equal the published `AreaTex.h`
(first 80 rows) and `SearchTex.h` byte for byte; the test checks FNV-1a hashes and
sums taken from those files, which are not shipped. Generation takes under 10 ms
in an optimized build and about 0.4 s in a debug build, once per process.

**TAA** jitters the projection by Halton (2, 3) offsets, applied in clip space by
`SceneGraph.projectionJitter` (both the camera and explicit-matrix encode paths).
The resolve reconstructs each pixel's surface from stored depth with the inverse
jittered matrix, reprojects it with the unjittered current and previous matrices
using the nearest depth in the 3 × 3 neighborhood, fetches history with a nine-tap
Catmull-Rom filter, clips it to the neighborhood's YCoCg box (Playdead's clip
toward the box center) and clamps alpha, then blends. Until history builds up,
each frame has weight `1 / (frames + 1)`, an exact running mean; after that the
current frame has weight 0.1. A still camera reprojects to the exact texel, so
history is not blurred. `reset()` discards history and restarts the jitter.

Without clipping (`.accumulation`), history is RGBA32Float and is read at the
nearest texel of the reprojected position, which is the pixel itself for a still
camera; 32-bit float filtering is optional on Apple GPUs, so it is not sampled.
Over 1,024 jittered frames of the 14.3° edge the output stays within 4.9e-4 (one
half-float step of the output) of the exact mean of the same frames, computed on
the CPU in double precision, at 64, 256 and 1,024 frames. The error against the
8 × 8 reference falls from 0.0076 to 0.0037 and 0.0033. With RGBA16Float history
the same run drifted: 0.0085, 0.0122 and 0.0416, with a pixel 0.171 off the mean.

### Measured on an M4 Max

The test scene is a white half-plane over transparent black, 64 × 64 pixels, with
its edge 14.3° (0.25 rad) or 40.1° (0.7 rad) from vertical. The reference is an
8 × 8 supersampled render of the same scene, box filtered. Error is RGBA RMSE
over the pixels within two of a partially covered reference pixel.

| Method | 14.3° edge | 40.1° edge |
| --- | --- | --- |
| One sample per pixel | 0.1169 | 0.1031 |
| FXAA, preset 29 | 0.0522 | 0.0562 |
| SMAA 1x, high | 0.0438 | 0.0261 |
| TAA, 16 jittered frames | 0.0190 | 0.0172 |
| TAA, 32 jittered frames | 0.0222 | 0.0182 |
| Accumulation, 64 frames | 0.0076 | 0.0089 |
| Accumulation, 1,024 frames | 0.0033 | — |

After 24 TAA frames, moving the camera 3.36 pixels sideways gives 0.0512 on the
first moved frame (one sample at the new position: 0.1146) and 0.0269 eight frames
later. During a steady pan of 0.4 pixel per frame (17.2° edge), the mean error
over frames 21 to 40 is 0.082 against 0.115 for one sample: each frame resamples
the history between texels, which softens the edge. A single bilinear history tap
instead of Catmull-Rom gives about 0.105 in the same run, so the test bound is
0.85 of the one-sample error.

FXAA and SMAA leave every pixel whose 5 × 5 neighborhood is flat bit-identical,
and FXAA leaves a 0.25–0.75 gradient unchanged. SMAA leaves a long vertical edge
bit-identical; FXAA's subpixel term softens it slightly. A reset
TAA writes the current frame exactly. Jitter of +0.6 pixel covers the next column
or row and −0.6 uncovers it, which fixes the x-right, y-down convention.

GPU time at 1920 × 1080 on an M4 Max (macOS 27.2), Metal API Validation off.
The source is an RGBA16Float HDR pattern (horizontal ramp 0–8, vertical ramp
0–4, and blue stripes of 3 and 0.05 every 5 to 7 pixels, so most pixels lie on
an edge); each pass ran 10 warm-up rounds and then 64 command buffers
interleaved with the other passes, and the table gives the median:

| Pass | Time |
| --- | --- |
| `HDRDisplay`, ACES filmic, H = 1 / 16 | 0.100 / 0.097 ms |
| `HDRDisplay`, AgX Punchy, H = 1 / 16 | 0.134 / 0.134 ms |
| `HDRDisplay`, Neutral, H = 1 / 16 | 0.086 / 0.088 ms |
| `FXAA`, presets 12 / 29 / 39 | 0.081 / 0.088 / 0.093 ms |
| `SMAA`, low / medium / high / ultra | 0.131 / 0.152 / 0.159 / 0.471 ms |
| `TemporalAntiAliasing`, clipping (RGBA16Float history) | 0.381 ms |
| `TemporalAntiAliasing`, `.accumulation` (RGBA32Float history) | 0.357 ms |

SMAA ultra's 32-step searches are expensive on this edge-dense pattern; on
ordinary images most pixels have no edge and skip the weight pass.

Deliberate breakages caught: FXAA blending along instead of across the edge,
FXAA without its edge-end offset (`pixelOffsetGood = 0`), preset 29 given preset
12's step table, SMAA's L-pattern geometry in the table generator, an area offset
in the SMAA shader (straight edges no longer stay sharp), SMAA without blending,
SMAA high without diagonal search, SMAA high without corner detection, SMAA luma
clamped to 1, the TAA motion sign, a reset that kept history, the jitter's y sign,
a resolve that ignored history, a single bilinear history tap instead of
Catmull-Rom, and RGBA16Float accumulation history.

## Limits

- TAA reprojects camera motion only. Objects that move or deform have no motion
  vectors; neighborhood clipping limits their ghosting but does not remove it.
  Thin features below a pixel can flicker under clipping.
- During continuous sub-pixel camera motion TAA is much softer than when still:
  0.082 on the panned test edge against 0.02 to 0.03 for the same edge held
  still (one sample: 0.115). There is no velocity-weighted blending or history
  sharpening.
- `.accumulation` reads history at the nearest texel, so it is exact for a still
  camera only; reset it when the view changes.
- `SSAARenderPass` has no one-call equivalent: per-frame supersampling means
  `reset()` and N renders and encodes in one frame.
- Particles, `ShaderParticleSystem` draws and `EnvironmentBackground` use their
  own camera matrices and are not jittered. A smooth sky does not need it.
- There is no MSAA: scene pipelines are single-sample. TAA, SMAA and FXAA are the
  anti-aliasing paths.
- SMAA is 1x only: no S2x, T2x or 4x, no color, depth or predicated edge
  detection. Its luma thresholds apply to sRGB-encoded values, so above SDR white
  the same threshold corresponds to a smaller relative contrast.
- Above SDR, the filmic curves keep their SDR shadows and mid-tones and gain a
  shared highlight extension; this is Frostfield's rule, not a dedicated HDR
  display transform such as the ACES 2.0 output transforms, which also raise
  middle gray slightly with peak luminance. Highlights brighter than exposed
  white change as the headroom changes, as with any EDR mapping.
- Blender's own AgX looks use OCIO grading transforms; Punchy and Golden here use
  the CDL values of Troy Sobotka's original configuration, as Filament does.
- All tests ran with `xcodebuild` and Metal API Validation on an M4 Max Mac with
  macOS 27.2 (full suite: 462 tests). The five suites here (`ToneMappingTests`,
  `FXAATests`, `SMAATests`, `TemporalAntiAliasingTests`, `HDRDisplayTests`, 36
  tests) also pass on the iPhone 17 Simulator with iOS 27.2 and report the same
  measurements; the Simulator renders on the host GPU, so this checks API use,
  not iPhone hardware.
  Nothing was measured on iPhone, iPad or Vision Pro hardware, and no rendered
  comparison with Three.js output was made.
