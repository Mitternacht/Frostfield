#ifndef FROSTFIELD_SHADER_TYPES_H
#define FROSTFIELD_SHADER_TYPES_H

// Records shared by the Swift renderer (through this C module) and the Metal
// shaders (by #include). One declaration fixes the byte layout for both
// compilers instead of assuming that a Swift struct matches an MSL struct.
//
// Layout rules: fixed-width scalars and documented SIMD types only, explicit
// padding, no Swift-only types. FrostfieldTests asserts size, alignment, and
// offsets on the host and checks the values through a GPU render; the shaders
// and FrostfieldShaderTypes.c repeat the size checks at compile time.

#ifdef __METAL_VERSION__
#include <metal_stdlib>
typedef metal::float2 FFFloat2;
#else
#include <simd/simd.h>
typedef simd_float2 FFFloat2;
#endif

/// Vertex buffer index of `FFTriangleUniforms` in `ff_triangle_vertex`.
#define FF_TRIANGLE_UNIFORMS_BUFFER_INDEX 0

/// Per-frame values for the diagnostic triangle. 16 bytes, 8-byte aligned.
typedef struct FFTriangleUniforms {
    /// Multiplies clip-space x and y after rotation so the triangle keeps its
    /// shape in a non-square drawable. Byte offset 0.
    FFFloat2 scale;
    /// Counter-clockwise rotation in radians, applied before `scale`.
    /// Byte offset 8.
    float rotation;
    /// Explicit padding to the 8-byte alignment of `scale`. Byte offset 12.
    float padding;
} FFTriangleUniforms;

#endif
