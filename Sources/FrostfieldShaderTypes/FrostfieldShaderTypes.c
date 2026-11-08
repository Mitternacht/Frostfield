#include "FrostfieldShaderTypes.h"

#include <stddef.h>

// Compile-time layout checks on the host side. The Metal shaders repeat the
// size and alignment checks; FrostfieldTests checks the same values from Swift.
_Static_assert(sizeof(FFTriangleUniforms) == 16, "FFTriangleUniforms must be 16 bytes");
_Static_assert(_Alignof(FFTriangleUniforms) == 8, "FFTriangleUniforms must be 8-byte aligned");
_Static_assert(offsetof(FFTriangleUniforms, scale) == 0, "scale must be at offset 0");
_Static_assert(offsetof(FFTriangleUniforms, rotation) == 8, "rotation must be at offset 8");
_Static_assert(offsetof(FFTriangleUniforms, padding) == 12, "padding must be at offset 12");

_Static_assert(sizeof(FFTransformUniforms) == 64, "FFTransformUniforms must be 64 bytes");
_Static_assert(_Alignof(FFTransformUniforms) == 16, "FFTransformUniforms must be 16-byte aligned");
_Static_assert(offsetof(FFTransformUniforms, modelViewProjection) == 0, "modelViewProjection must be at offset 0");

_Static_assert(sizeof(FFUnlitParameters) == 16, "FFUnlitParameters must be 16 bytes");
_Static_assert(_Alignof(FFUnlitParameters) == 16, "FFUnlitParameters must be 16-byte aligned");
_Static_assert(offsetof(FFUnlitParameters, color) == 0, "color must be at offset 0");
