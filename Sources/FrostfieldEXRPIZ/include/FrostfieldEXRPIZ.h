#ifndef FROSTFIELD_EXR_PIZ_H
#define FROSTFIELD_EXR_PIZ_H

#include <stddef.h>
#include <stdint.h>

enum { FF_EXR_PIZ_MAX_BLOCK_BYTES = 64 * 1024 * 1024 };

// Raw bytes are little-endian and ordered by row, channel, then sample.
// Each channel contributes either 2 or 4 bytes per pixel. Returns 0 if
// a smaller compressed payload cannot be written within outputCapacity.
size_t ff_exr_compress_piz(
    const void *raw, size_t rawCount, uint32_t width, uint32_t rows,
    const uint8_t *channelSampleBytes, size_t channelCount,
    void *output, size_t outputCapacity);

#endif
