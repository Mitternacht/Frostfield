#ifndef FROSTFIELD_JPEGXL_H
#define FROSTFIELD_JPEGXL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum FrostfieldJPEGXLStatus {
  FrostfieldJPEGXLSuccess,
  FrostfieldJPEGXLInvalid,
  FrostfieldJPEGXLLimit,
  FrostfieldJPEGXLAnimated,
  FrostfieldJPEGXLColorProfile
} FrostfieldJPEGXLStatus;

typedef struct FrostfieldJPEGXLImage {
  uint32_t width;
  uint32_t height;
  float *rgba;
  float reference_headroom;
} FrostfieldJPEGXLImage;

FrostfieldJPEGXLStatus FrostfieldJPEGXLDecode(
    const uint8_t *input, size_t input_count, uint32_t maximum_side,
    size_t maximum_pixels, FrostfieldJPEGXLImage *result);

void FrostfieldJPEGXLFree(float *rgba);

#ifdef __cplusplus
}
#endif

#endif
