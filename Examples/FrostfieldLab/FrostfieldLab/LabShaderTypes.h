#ifndef LAB_SHADER_TYPES_H
#define LAB_SHADER_TYPES_H

// Records shared by the lab's Swift code (through the target's bridging
// header setting) and its Metal shaders (by #include), so both compilers see
// one byte layout. This is the pattern docs/materials.md recommends for an
// app's material parameters.

#ifdef __METAL_VERSION__
#include <metal_stdlib>
typedef metal::float4 LabFloat4;
#else
#include <simd/simd.h>
typedef simd_float4 LabFloat4;
#endif

/// Parameters of `lab_pulse_fragment`. 48 bytes, 16-byte aligned.
typedef struct LabPulseParameters {
    /// The color between bands. Byte offset 0.
    LabFloat4 coolColor;
    /// The color of the bands. Byte offset 16.
    LabFloat4 warmColor;
    /// Animation time in seconds, supplied by the app every frame. Byte offset 32.
    float time;
    /// How fast the bands travel, in radians of phase per second. Byte offset 36.
    float speed;
    /// Explicit padding to the 16-byte alignment of the colors. Byte offset 40.
    float padding[2];
} LabPulseParameters;

#endif
