# Dithering

Set `HDRDisplay.Settings.dithering` to add Three.js r187's `material.dithering` shift
to the final color of the display pass. A screen-space hash of the pixel moves each
channel by up to half of one 8-bit step, plus, minus, plus for red, green and blue, so
that quantizing to 8 bits hides banding in smooth gradients. It is off by default, and
with it off every output byte is the byte the pass wrote before the option existed.

This is a documented native equivalent, not a per-material flag. See
[Departures from r187](#departures-from-r187) for why and for every other difference.

```swift
let display = try HDRDisplay(device: device, destinationPixelFormat: .bgra8Unorm_srgb)
var settings = HDRDisplay.Settings(toneMapping: .acesFilmic)
settings.dithering = true
try display.encode(source: hdrColor, destination: drawableTexture, settings: settings, commandBuffer: commandBuffer)

// In a post-process chain, on the final, unscaled, unmasked tone-mapping stage of a chain
// whose destination is BGRA8Unorm_sRGB or RGBA8Unorm_sRGB.
let chain = try PostProcessChain(
    queue: queue, destinationPixelFormat: .bgra8Unorm_srgb,
    effects: [.colorGrade(strength: 0.5), .toneMapping(.init(dithering: true))]
)
```

`FrostfieldSceneView.hdrDisplay(_:settings:)` takes the same settings. Dithering needs
an 8-bit RGBA or BGRA destination, because that is where the color is quantized:
`.rgba8Unorm`, `.bgra8Unorm`, `.rgba8Unorm_srgb` or `.bgra8Unorm_srgb`. An RGBA16Float
destination throws `HDRDisplay.Failure.ditheringNeedsEightBitDestination` instead of
silently doing nothing. A chain throws the same error when its destination is RGBA16Float
when it is created, and when a dithered stage is not the final stage, is scaled or is
masked, because such a stage writes an RGBA16Float intermediate. It also throws when a
scene-source chain renders a reduced capture, because that capture is bilinearly resampled
into the destination after the last stage and the resample would blur the dither. Each of
these throws happens when the chain is prepared for encoding.

The reusable Metal functions are in `FrostfieldDithering.h`:
`frostfieldR187Rand(float2)`, `frostfieldR187DitherShift(float)`,
`frostfieldR187Dithering(float3, float2)` for a color and an OpenGL-convention fragment
coordinate, and `frostfieldR187DitherOutput(float3, float2, uint)` for a render target.
The header is bundled as a resource, and `DitheringShader.metalSource()` returns it for
shaders compiled at run time. That accessor is package-level API, not public.

## What r187 does

`material.dithering` adds `#define DITHERING` to the material's fragment shader
([WebGLProgram](https://github.com/mrdoob/three.js/blob/r187/src/renderers/webgl/WebGLProgram.js)).
Every built-in material ends with `opaque_fragment`, `tonemapping_fragment`,
`colorspace_fragment`, `fog_fragment`, `premultiplied_alpha_fragment` and
`dithering_fragment`
([meshphysical](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderLib/meshphysical.glsl.js)),
so the shift lands on the final, encoded, premultiplied color. The chunks are
[`dithering_pars_fragment`](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/dithering_pars_fragment.glsl.js),
[`dithering_fragment`](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/dithering_fragment.glsl.js)
and `rand` in
[`common`](https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/common.glsl.js).
The code below inlines the two dithering chunks as one equivalent sequence, so it is not a
verbatim quote:

```glsl
float grid_position = rand( gl_FragCoord.xy );
vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
gl_FragColor.rgb = gl_FragColor.rgb + dither_shift_RGB;

highp float rand( const in vec2 uv ) {
    const highp float a = 12.9898, b = 78.233, c = 43758.5453;
    highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
    return fract( sin( sn ) * c );   // #define PI 3.141592653589793
}
```

The shift is therefore `+0.5 / 255` down to `-0.5 / 255` as the grid position rises
from 0 to 1, with the sign pattern (+, -, +). The Metal port has the same constants, the
same `mix` and the same sign pattern. [MIT notice](../../LICENSES/Three-Dithering.txt)

## Where Frostfield quantizes

- `MeshRenderer`, `SceneGraph` and the surface passes write premultiplied linear color to
  the target they are given. In the HDR pipeline that is RGBA16Float, like the post-process
  intermediates, scaled and saved frames, and path-traced images.
- `HDRDisplay` converts RGBA16Float to the destination: tone mapping, output primaries,
  premultiplication and, for the `_srgb` formats, the transfer that Metal applies when it
  stores. It accepts RGBA16Float and the four 8-bit formats above.
- `PostProcessChain` writes RGBA16Float intermediates and its destination is RGBA16Float,
  BGRA8Unorm_sRGB or RGBA8Unorm_sRGB. Its `.toneMapping` stage is an `HDRDisplay`, so the
  stage inherits dithering when it is the final stage. Other final stages (`.graph`,
  `.colorGrade`, `.screenEffects`, `.fxaa`, `.smaa`, `.pass`) write the 8-bit destination
  without dithering. `.fxaa` and `.smaa` are documented to follow `.toneMapping`, so a chain
  that ends in antialiasing cannot dither.
- A `MeshRenderer` can also target an 8-bit format directly, as `FrostfieldSceneView` does
  by default (`bgra8Unorm_srgb`) when no `hdrDisplay` is set. That output is not dithered.

## Departures from r187

1. **Per output stage, not per material.** r187 compiles `DITHERING` into one material's
   fragment shader. Frostfield materials write HDR RGBA16Float color that later stages
   tone map, grade, blur, antialias and encode, so a ±0.5 / 255 shift of linear scene color is
   not a ±0.5 code shift of the stored 8-bit value, and it would pass through every later
   pass. The color is quantized in the display pass, so that is where the switch lives, and
   it applies to every pixel of the frame. You cannot dither one mesh and not another. A
   per-material flag would need a variant of every fragment function (unlit, textured,
   Lambert, shadowed, classic, PBR, generated surfaces, logarithmic depth, multiple render
   targets, indirect draws) and of the pipeline caches that key them, and it would only help
   meshes drawn straight into an 8-bit target. That change is not made.
2. **Only 8-bit RGBA and BGRA destinations.** See above. r187 shifts whatever the
   framebuffer is.
3. **sRGB destinations are shifted in encoded values.** r187's shader outputs
   sRGB-encoded numbers to an RGBA8 framebuffer, so its shift is a shift of the stored
   code. Frostfield stores linear color in an `_srgb` format and Metal encodes it on store.
   A shift of 0.5 / 255 in linear light would be about 6 codes at linear 0.002 and 0.2
   codes at white. The pass therefore computes `decode(encode(color) + shift)`, with Metal's
   piecewise sRGB functions mirrored for negative values, so that the stored code moves by
   at most half a step at every brightness. For `.rgba8Unorm` and `.bgra8Unorm` the color
   is stored as written and the shift is added to it directly.
4. **`gl_FragCoord` is reproduced, including its origin.** The input is the pixel center
   counted from the bottom-left corner, `(x + 0.5, height - y - 0.5)` for the pixel `(x, y)`
   from the top-left, so the same frame size gives the same pattern as WebGL at the same
   pixel. It is relative to the destination texture, not to a viewport.
5. **Fully transparent pixels stay zero.** r187 shifts the rgb of every fragment, even at
   alpha 0. A premultiplied pixel with alpha 0 and a nonzero color is not a valid
   premultiplied color, so `HDRDisplay` leaves it at `(0, 0, 0, 0)`. The guard is defensive:
   round-to-nearest already keeps a zero color at code 0 for a shift of at most 0.5 / 255, so
   no test can tell the guard from its absence. A pixel with alpha above 0
   is shifted in its premultiplied color, as r187 shifts it after `premultiplied_alpha_fragment`.
   Alpha is never shifted. For alpha below one code the premultiplied color can end up a
   code above alpha, as in r187.
6. **The float32 evaluation order is pinned.** GLSL leaves `dot`, `mod` and `sin` to the
   implementation, and the multiplier `c = 43758.5453` turns a single ulp of difference into a
   different result at a pixel (see [Precision](#precision)). The port computes `dot` as one
   rounded product and one fused multiply-add, `mod` as `x - y * floor( x / y )` with a
   correctly rounded divide and one fused multiply-subtract, and `sin` as `precise::sin`. The
   constants are the float32 values of the GLSL literals. Metal's `fract` stops just below 1
   where `x - floor( x )` could round to 1. Builds with fast math and with safe math produce
   identical bits.
7. **After the AA mix.** With `Settings.antiAliasing`, the shift is added to the final
   mixed color, which is the last operation of the pass. Code reading shows this order, and
   the test only checks that no code moves by more than one. It cannot tell this order from
   dithering the center sample before the mix.

## Other dithering in the package

These are not `rand` and were not reused: `BokehDepthOfField.Settings.dither` rotates tap
patterns with Jimenez interleaved gradient noise (`fract(52.9829189 * fract(dot(p, (0.06711056,
0.00583715))))`), `ShaderGraphBuilder.bayerDither` and `PostProcessGraph.retroCRT` use a 4 × 4
Bayer matrix with a chosen number of levels, and `PathTracer.Settings.sampler == .blueDither`
shifts sample sequences with a blue-noise rank field. Each solves a different problem
with different math.

## Precision

r187's `rand` cannot be compared with a Double evaluation of the same formula at most
pixels, in any float32 implementation. A rounding of `dt` by one ulp moves `sin( sn )`
by about as much, and `c` multiplies that by 43,758. At the first pixel center `(0.5, 0.5)`, `dt = 45.6`
has an ulp of 3.8e-6, which already moves the grid position by 0.17. By a pixel with `dt`
near 1,000 it is a different random number. WebGL on a GPU has the same limit, so the
pattern at a pixel is a property of the exact operation order, which the port pins. The
tests below therefore check each stage against Double arithmetic with the bound that
stage can meet, check the whole formula against Double where float32 can resolve it, and
check the whole grid against a CPU mirror of the shader's float32 operations.

Measured on an Apple M4 Max (the iPad Pro (M5) Simulator runs Metal on the same host GPU and
printed identical values):

| Quantity | Measured |
| --- | --- |
| `precise::sin` over [0, PI], 1,048,576 points, against Double | max error 1.31e-7 (2.63 ulp); `fast::sin` has the same maximum there |
| Effect of that on the grid position (`c` × 1.31e-7) | up to 0.0058, with up to 0.0020 more from rounding the product |
| `dt` at 85,849 pixel centers of a 2,048 × 2,048 image | within 0.95 ulp of Double with the float32 constants |
| `sn = mod( dt, PI )` at the same pixels | exact (0 ulp) for the shader's `dt`, never outside 0 to PI by more than 2 ulp of `dt` |
| Grid position given the shader's `sn` | within 0.0072 of Double; the stated bound is 0.008 |
| `mix` shift for 1,005 grid positions | within 3.3e-10 of Double (one 8-bit step is 3.9e-3) |
| Shift of 22,002 coordinates with `dt < 64`, including pixel centers (0.5, 0.5) and (1.5, 0.5), against Double with the same constants | worst grid position error 0.150 against a derived bound of 0.175 at the same point (0.857 of the bound), no violations. That is 0.15 of a code at worst |
| All 4,194,304 pixel centers of a 2,048 × 2,048 image against the float32 mirror | 100% within 0.01 of the grid position, worst 0.0078 (0.0078 of a code) |
| The same pixels against Double with the float32 constants | 2.01% within 0.01, which is chance (two independent uniform values fall within 0.01 of each other 2% of the time) |
| The same against Double with the decimal constants `12.9898`, `78.233`, `43758.5453`, `PI` | 2.02% |
| Distribution of the grid position over 466,489 pixels | mean 0.4985, variance 0.08324 (uniform 0.5, 0.08333), 16 bins within 0.73% and +0.80% of uniform, range 0 to 0.99988 |
| Fast math and safe math builds of the functions, 168,100 pixels | 0 pixels differ |
| sRGB encode and decode against Double | within 1.6e-7 and 2.0e-7 |

The mirror in the tests is Swift `Float` with the same fused operations and `sinf`.
Where a stage has a hardware limit, such as the `precise::sin` figure above, the test
asserts the measured bound and not a looser one.

## Verification

Both runs used command-line `xcodebuild` from Xcode 27.2 with `MTL_DEBUG_LAYER=1`, because the
Xcode MCP bridge timed out and its DerivedData volume was full. The test process printed that
variable, and `xctest` printed `Metal API Validation Enabled`.

The focused Mac run on an Apple M4 Max and the iPad Pro (M5) Simulator run each reported
`Test run with 50 tests in 6 suites passed` and `** TEST SUCCEEDED **`. The suites are the new
`DitheringShaderTests` and `HDRDisplayDitheringTests`, which define 18 test functions, and the
existing `HDRDisplayTests`, `ToneMappingTests`, `PostProcessChainTests` and `FXAATests`. The iPad
Pro (M5) Simulator is a simulator device, "Apple iOS simulator GPU", and executes Metal on the
host's GPU, so it does not measure M5 hardware. Its simulated GPU cannot dispatch non-uniform
threadgroups, so the probe kernels dispatch whole threadgroups over padded buffers.

A full Mac run of the package on the final tree reported `Test run with 2170 tests in 424 suites
passed`, `Test run with 6 tests in 2 suites passed` and `** TEST SUCCEEDED **`. The last
recorded full run of the integrated tip was 2,152 tests in 422 suites, so the 18 new test functions
and 2 new suites account for the difference. The full run was not repeated on the iPad Pro (M5)
Simulator.

**Off is byte-identical.** The SHA-256 of every destination byte of 16 renders was captured
before the change, on the integrated tip `8f473f7` with the shader sources unmodified, on both
platforms, and the digests were equal on both. The renders cover 13 combinations of tone mapping,
exposure, headroom, output primaries, antialiasing and destination format
(RGBA16Float, BGRA8Unorm_sRGB, RGBA8Unorm_sRGB, RGBA8Unorm, BGRA8Unorm), the scaled and
crossfade entry points, and a one-stage chain, from a 64 × 32 frame, or its 32 × 16 subsample for
the two entry points, with gradients, an HDR highlight, alpha 0, 0.5 and 1 bands and a diagonal edge. After the change
`HDRDisplayDitheringTests` renders the same frame with the default settings and with an explicit
`dithering: false`, and compares all of them with the stored digests.

**The shift on the GPU against independent references.** The tests above, plus: for
`.rgba8Unorm` and `.bgra8Unorm` destinations the stored code equals the float32 mirror's
expected code at 99.80% of 24,576 channel values per format and is within one code at 100%.
The frame sits half a code above whole codes, so the sign of the shift decides the code: 48.5%
of codes change, and a reference that counts rows from the top agrees at only 50.3%. For the
`_srgb` formats 96.20% of the codes equal the encoded-space reference and all are within one
code, none more than one code from the undithered code. The undithered store equals its own
reference at 95.31%, because of the hardware behavior below.

**Metal's sRGB store is uneven in the dark.** A ramp of linear values into an
`rgba8Unorm_srgb` target puts 39 code boundaries within the first 39 codes at up to 0.345
codes from the encoded midpoint, in a period-4 pattern (`-0.094 -0.290 0.319 0.124 ...`).
Dithering cannot remove the store's own curve, so the dark sRGB ramp below improves less.

**Banding.** A 256 × 64 ramp of four codes (64 pixels per code), rendered with and without
dithering, compared by 16 × 16 window averages of the stored code against the average of the
source's true value (the Float16 source and its exact encode), in 8-bit codes. Lower is better.

| Ramp | Undithered RMS | Dithered RMS | Ratio |
| --- | --- | --- | --- |
| 8-bit linear, codes 102 to 106 | 0.2694 | 0.0235 | 0.087 |
| sRGB, encoded codes 102 to 106 | 0.2616 | 0.0334 | 0.128 |
| 8-bit linear, dark, codes 5.1 to 9.1 | 0.1813 | 0.0263 | 0.145 |
| sRGB, dark, encoded codes 5.1 to 9.1 | 0.3117 | 0.2256 | 0.724 |

The mean error of the dithered result is within 0.006 of a code in the first three rows. A
sawtooth quantizer leaves about 0.29 codes RMS, and independent noise of 0.4 codes averaged over
256 pixels leaves about 0.025, which the first and third rows match. The dark sRGB row is limited by
the uneven store measured above, and the test asserts that it improves, by less than 0.85 times the
undithered error and below the store's 0.345-code deviation, not that it reaches the others.

**Other behavior.** Alpha bytes are unchanged at every pixel, pixels with alpha 0 stay zero (which the guard does not affect, see departure 5), and
half-transparent pixels shift their premultiplied color like any other. With antialiasing,
dithering moves no code by more than one and changes some. A chain whose final stage is a dithered
tone-mapping stage is byte-identical to the display pass. `HDRDisplay` with an RGBA16Float
destination, a chain with an RGBA16Float destination and a dithered stage that is not last each
throw `ditheringNeedsEightBitDestination`.

## Not implemented

- A `dithering` flag on materials, and dithering of a mesh pass that renders straight into an
  8-bit target. The default `FrostfieldSceneView` presentation, without `hdrDisplay`, is not dithered.
- Dithering from a final chain stage other than `.toneMapping`, including a chain that ends with
  `.fxaa` or `.smaa`.
- Dithering of an RGBA16Float destination, which has no 8-bit quantization to hide.
- The SwiftUI `hdrDisplay(_:settings:)` pass-through is unchanged and not separately tested.
- Physical iPad hardware and Apple Vision Pro were not run.
