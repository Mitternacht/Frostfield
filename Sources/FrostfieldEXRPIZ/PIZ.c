#include "FrostfieldEXRPIZ.h"

#include <stdlib.h>
#include <string.h>

typedef enum {
    EXR_ERR_SUCCESS = 0,
    EXR_ERR_ARGUMENT_OUT_OF_RANGE,
    EXR_ERR_INVALID_ARGUMENT,
} exr_result_t;

#include "Vendor/OpenEXRHuffman.inc"
#include "Vendor/OpenEXRWavelet.inc"

static void writeUInt16(uint8_t *output, uint16_t value) {
    output[0] = (uint8_t)value;
    output[1] = (uint8_t)(value >> 8);
}

size_t ff_exr_compress_piz(
    const void *raw, size_t rawCount, uint32_t width, uint32_t rows,
    const uint8_t *channelSampleBytes, size_t channelCount,
    void *output, size_t outputCapacity
) {
    if (!raw || !channelSampleBytes || !output || width == 0 || width > 32768 ||
        rows == 0 || rows > 32 || channelCount == 0 || channelCount > 64 ||
        rawCount > FF_EXR_PIZ_MAX_BLOCK_BYTES || outputCapacity < 28 ||
        outputCapacity >= rawCount) {
        return 0;
    }

    size_t bytesPerPixel = 0;
    for (size_t channel = 0; channel < channelCount; ++channel) {
        uint8_t sampleBytes = channelSampleBytes[channel];
        if (sampleBytes != 2 && sampleBytes != 4) return 0;
        bytesPerPixel += sampleBytes;
    }
    size_t pixelCount = (size_t)width * rows;
    if (pixelCount * bytesPerPixel != rawCount) return 0;

    size_t huffmanSpareBytes = (size_t)internal_exr_huf_compress_spare_bytes();
    uint8_t *scratch = malloc(huffmanSpareBytes + BITMAP_SIZE + USHORT_RANGE * sizeof(uint16_t));
    // The wavelet loop can reach one element beyond its last interleaved lane.
    uint16_t *words = malloc(rawCount + sizeof(uint16_t));
    if (!scratch || !words) {
        free(words);
        free(scratch);
        return 0;
    }
    uint8_t *bitmap = scratch + huffmanSpareBytes;
    uint16_t *lut = (uint16_t *)(bitmap + BITMAP_SIZE);
    size_t wordCount = rawCount / sizeof(uint16_t);

    // PIZ transforms one channel plane at a time. Keep each 32-bit channel's
    // two words interleaved, so the transform sees two independent sample lanes.
    const uint8_t *packed = raw;
    for (size_t row = 0; row < rows; ++row) {
        uint16_t *plane = words;
        for (size_t channel = 0; channel < channelCount; ++channel) {
            size_t rowWords = (size_t)width * (channelSampleBytes[channel] / 2);
            uint16_t *line = plane + row * rowWords;
            for (size_t i = 0; i < rowWords; ++i, packed += 2) {
                line[i] = (uint16_t)packed[0] | ((uint16_t)packed[1] << 8);
            }
            plane += rows * rowWords;
        }
    }

    uint16_t firstBitmapByte, lastBitmapByte;
    bitmapFromData(words, wordCount, bitmap, &firstBitmapByte, &lastBitmapByte);
    size_t bitmapBytes = firstBitmapByte <= lastBitmapByte ? lastBitmapByte - firstBitmapByte + 1 : 0;
    size_t prefixBytes = 8 + bitmapBytes;
    size_t written = 0;
    if (outputCapacity >= prefixBytes + 20) {
        uint16_t maxValue = forwardLutFromBitmap(bitmap, lut);
        applyLut(lut, words, wordCount);

        uint16_t *plane = words;
        for (size_t channel = 0; channel < channelCount; ++channel) {
            int lanes = channelSampleBytes[channel] / 2;
            for (int lane = 0; lane < lanes; ++lane) {
                wav_2D_encode(plane + lane, (int)width, lanes, (int)rows, (int)width * lanes, maxValue);
            }
            plane += pixelCount * (size_t)lanes;
        }

        uint8_t *bytes = output;
        uint64_t huffmanBytes = 0;
        if (internal_huf_compress(
                &huffmanBytes, bytes + prefixBytes, outputCapacity - prefixBytes,
                words, wordCount, scratch, huffmanSpareBytes) == EXR_ERR_SUCCESS) {
            writeUInt16(bytes, firstBitmapByte);
            writeUInt16(bytes + 2, lastBitmapByte);
            if (bitmapBytes > 0) memcpy(bytes + 4, bitmap + firstBitmapByte, bitmapBytes);
            writeUInt(bytes + 4 + bitmapBytes, (uint32_t)huffmanBytes);
            written = prefixBytes + (size_t)huffmanBytes;
        }
    }

    free(words);
    free(scratch);
    return written;
}
