#include "FrostfieldBasis.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <new>
#include <vector>

namespace {
#include "Vendor/pvrtc1-decompress.inc"
}

int32_t frostfield_basis_decode_pvrtc1(
    const void *blocks, size_t block_size, uint32_t width, uint32_t height,
    int32_t two_bpp, int32_t opaque, void *output, size_t output_size
) {
    static_assert(sizeof(Pixel32) == 4);
    if (!blocks || !output || !width || !height || width > 8192 || height > 8192 ||
        (width & (width - 1)) || (height & (height - 1)) || uint64_t(width) * height > 16777216) {
        return 0;
    }
    const uint32_t padded_width = std::max(width, two_bpp ? 16u : 8u);
    const uint32_t padded_height = std::max(height, 8u);
    const size_t padded_count = size_t(padded_width) * padded_height;
    if (block_size != padded_count / (two_bpp ? 4 : 2) ||
        output_size != size_t(width) * height * sizeof(Pixel32)) {
        return 0;
    }
    try {
        std::vector<uint32_t> source(block_size / sizeof(uint32_t));
        std::memcpy(source.data(), blocks, block_size);
        std::vector<Pixel32> pixels(padded_count);
        if (pvrtcDecompress(reinterpret_cast<uint8_t *>(source.data()), pixels.data(),
                            padded_width, padded_height, two_bpp ? 2 : 4) != block_size) {
            return 0;
        }
        auto *destination = static_cast<uint8_t *>(output);
        for (size_t y = 0; y < height; ++y) {
            Pixel32 *row = pixels.data() + y * padded_width;
            if (opaque) {
                for (size_t x = 0; x < width; ++x) row[x].alpha = 255;
            }
            std::memcpy(destination + y * width * sizeof(Pixel32), row, width * sizeof(Pixel32));
        }
        return 1;
    } catch (const std::bad_alloc &) {
        return 0;
    }
}
