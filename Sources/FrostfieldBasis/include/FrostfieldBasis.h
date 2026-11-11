#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FrostfieldBasisContext FrostfieldBasisContext;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t levels;
    uint32_t layers;
    uint32_t faces;
    uint8_t is_srgb;
    uint8_t is_uastc;
    uint8_t has_alpha;
    uint8_t is_hdr;
} FrostfieldBasisInfo;

typedef enum {
    FrostfieldBasisRGBA8,
    FrostfieldBasisETC1RGB,
    FrostfieldBasisETC2RGBA,
    FrostfieldBasisASTC4x4,
    FrostfieldBasisBC7,
    FrostfieldBasisRGBA16Float,
} FrostfieldBasisFormat;

int32_t frostfield_basis_validate(const void *data, size_t size);

// Borrows the input through frostfield_basis_close. The caller owns the returned context.
// Status is 1 for invalid data, 2 for an unsupported layout or transfer function.
FrostfieldBasisContext *frostfield_basis_open(
    const void *data, size_t size, uint8_t allowing_spatial_textures, FrostfieldBasisInfo *info, int32_t *status);
int32_t frostfield_basis_transcode(
    FrostfieldBasisContext *context, FrostfieldBasisFormat format, uint32_t level, uint32_t layer,
    uint32_t face, void *output, size_t size);
void frostfield_basis_close(FrostfieldBasisContext *context);

typedef enum {
    FrostfieldBC1Opaque,
    FrostfieldBC2,
    FrostfieldBC3,
    FrostfieldBC6HSigned,
    FrostfieldBC6HUnsigned,
    FrostfieldBC1RGBA,
    FrostfieldBC4,
    FrostfieldBC4Signed,
    FrostfieldBC5,
    FrostfieldBC5Signed,
    FrostfieldBC7,
} FrostfieldBCFormat;

// Decodes a single 2D slice to tightly packed RGBA8, RGBA8Snorm, or RGBA16F. Returns 1 on
// success and 0 for invalid dimensions, format, or buffer sizes.
int32_t frostfield_basis_decode_bc(
    FrostfieldBCFormat format, const void *blocks, size_t block_size, uint32_t width, uint32_t height,
    void *output, size_t output_size);

// Decodes a twiddled PVRTC1 level to tightly packed RGBA8.
int32_t frostfield_basis_decode_pvrtc1(
    const void *blocks, size_t block_size, uint32_t width, uint32_t height,
    int32_t two_bpp, int32_t opaque, void *output, size_t output_size);

#ifdef __cplusplus
}
#endif
