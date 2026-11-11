// Boundary traversal for mesh fog compares against infinite limits and steps to the
// next representable parameter, so the whole translation unit compiles with safe math,
// as the runtime-generated surface libraries do.
#pragma METAL fp math_mode(safe)
#include "PathTracingShading.h"
