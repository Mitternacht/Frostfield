#include <metal_stdlib>
#include "FrostfieldShadowSampling.h"
using namespace metal;

// Reuse the PBR varyings, light records, punctual shadows and the hashed alpha
// threshold without compiling a second copy of the PBR fragment functions.
#define FROSTFIELD_PBR_HELPERS_ONLY 1
#include "FrostfieldScreenSpace.h"
#include "FrostfieldDithering.h"
#include "PhysicallyBasedShading.h"
#include "ClassicMaterialShading.h"
