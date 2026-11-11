#ifndef FROSTFIELD_DITHERING_H
#define FROSTFIELD_DITHERING_H

#include <metal_stdlib>

// Three.js r187 `material.dithering` as reusable Metal functions (MIT notice in
// LICENSES/Three-Dithering.txt).
//
// r187 adds a screen-space shift to the final fragment color, after tone mapping, color
// space conversion, fog and premultiplication, so that quantizing to 8 bits hides banding.
// ShaderChunk `dithering_pars_fragment` and `dithering_fragment`, inlined here as one
// equivalent sequence (the chunks wrap it in a `dithering( color )` function):
//
//     float grid_position = rand( gl_FragCoord.xy );
//     vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
//     dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
//     gl_FragColor.rgb = gl_FragColor.rgb + dither_shift_RGB;
//
// and ShaderChunk `common`, with `#define PI 3.141592653589793`:
//
//     highp float rand( const in vec2 uv ) {
//         const highp float a = 12.9898, b = 78.233, c = 43758.5453;
//         highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
//         return fract( sin( sn ) * c );
//     }
//
// https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/dithering_pars_fragment.glsl.js
// https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/dithering_fragment.glsl.js
// https://github.com/mrdoob/three.js/blob/r187/src/renderers/shaders/ShaderChunk/common.glsl.js
//
// `rand` is float32 arithmetic whose last step multiplies `sin` by 43758.5453: one ulp
// of error in `sin` or one half-ulp of error in `dt` moves the result by far more than
// the shift needs, so the value at one pixel is implementation defined in GLSL itself.
// These functions pin every step so that the result is the same on every Apple GPU and
// can be reproduced on the CPU: `dot` is one rounded product and one fused multiply-add,
// `mod` is GLSL's `x - y * floor( x / y )` with a correctly rounded divide and a fused
// multiply-subtract, and `sin` is `precise::sin`. The constants are the float32 values of
// the GLSL literals. Differences from r187 are listed in docs/evidence/dithering.md.

// `dot( uv.xy, vec2( a, b ) )`: one rounded product and one fused multiply-add.
static inline float frostfieldR187RandDot(metal::float2 uv) {
    constexpr float a = 12.9898f;
    constexpr float b = 78.233f;
    return metal::fma(uv.y, b, uv.x * a);
}

// `mod( dt, PI )` for the float32 value of the GLSL literal PI.
static inline float frostfieldR187RandMod(float dt) {
    constexpr float pi = 3.141592653589793f;
    return metal::fma(-pi, metal::floor(metal::precise::divide(dt, pi)), dt);
}

// `fract( sin( sn ) * c )`.
static inline float frostfieldR187RandFract(float sn) {
#pragma METAL fp contract(off)
    constexpr float c = 43758.5453f;
    return metal::fract(metal::precise::sin(sn) * c);
}

// r187 `rand( uv )`, in 0...1.
static inline float frostfieldR187Rand(metal::float2 uv) {
    return frostfieldR187RandFract(frostfieldR187RandMod(frostfieldR187RandDot(uv)));
}

// The shift r187 adds to red, green and blue for a grid position in 0...1: plus 0.5 / 255
// down to minus 0.5 / 255 as the position rises, with the sign pattern (+, -, +).
static inline metal::float3 frostfieldR187DitherShift(float gridPosition) {
    const metal::float3 shift = metal::float3(0.25f / 255.0f, -0.25f / 255.0f, 0.25f / 255.0f);
    return metal::mix(2.0f * shift, -2.0f * shift, metal::float3(gridPosition));
}

// r187 `dithering( color )` at a fragment coordinate in OpenGL window convention: the
// pixel center, counted from the lower-left corner, as `gl_FragCoord.xy`.
static inline metal::float3 frostfieldR187Dithering(metal::float3 color, metal::float2 fragmentCoordinate) {
    return color + frostfieldR187DitherShift(frostfieldR187Rand(fragmentCoordinate));
}

// The sRGB transfer functions with Metal's piecewise definition, mirrored for negative values.
static inline metal::float3 frostfieldSRGBEncode(metal::float3 linear) {
    metal::float3 magnitude = metal::abs(linear);
    metal::float3 encoded = metal::select(
        1.055f * metal::pow(magnitude, metal::float3(1.0f / 2.4f)) - 0.055f, magnitude * 12.92f,
        magnitude <= 0.0031308f
    );
    return metal::copysign(encoded, linear);
}

static inline metal::float3 frostfieldSRGBDecode(metal::float3 encoded) {
    metal::float3 magnitude = metal::abs(encoded);
    metal::float3 decoded = metal::select(
        metal::pow((magnitude + 0.055f) / 1.055f, metal::float3(2.4f)), magnitude / 12.92f,
        magnitude <= 0.04045f
    );
    return metal::copysign(decoded, encoded);
}

// Dithering for a render target. The shift is r187's, in units of the stored 8-bit code.
// Mode 1 shifts the color as stored, for a target without a transfer function. Mode 2 is for
// an sRGB target, where Metal encodes the color on store: the shift is added to the
// sRGB-encoded color and the sum is decoded again, so that the stored code moves by
// +-0.5 and not by a size that depends on brightness.
static inline metal::float3 frostfieldR187DitherOutput(
    metal::float3 color, metal::float2 fragmentCoordinate, unsigned int mode
) {
    metal::float3 shift = frostfieldR187DitherShift(frostfieldR187Rand(fragmentCoordinate));
    if (mode == 2u) { return frostfieldSRGBDecode(frostfieldSRGBEncode(color) + shift); }
    return color + shift;
}

#endif
