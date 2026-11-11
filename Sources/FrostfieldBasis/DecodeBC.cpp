#include "FrostfieldBasis.h"
#include "Vendor/transcoder/basisu_transcoder.h"
#include "Vendor/transcoder/basisu_transcoder_internal.h"

#include <algorithm>
#include <cstring>

namespace {
#include "Vendor/bcdec-bc6h.inc"
#include "Vendor/bcdec-bc4.inc"
}

int32_t frostfield_basis_decode_bc(
    FrostfieldBCFormat format, const void *blocks, size_t block_size, uint32_t width, uint32_t height,
    void *output, size_t output_size
) {
    if (!blocks || !output || !width || !height || width > 8192 || height > 8192 ||
        uint64_t(width) * height > 16777216 ||
        format < FrostfieldBC1Opaque || format > FrostfieldBC7) {
        return 0;
    }
    const bool hdr = format == FrostfieldBC6HSigned || format == FrostfieldBC6HUnsigned;
    const size_t texel_bytes = hdr ? 8 : 4;
    const bool single_channel = format == FrostfieldBC4 || format == FrostfieldBC4Signed;
    const bool signed_channels = format == FrostfieldBC4Signed || format == FrostfieldBC5Signed;
    const bool bc1 = format == FrostfieldBC1Opaque || format == FrostfieldBC1RGBA;
    const size_t block_bytes = bc1 || single_channel ? 8 : 16;
    const size_t columns = (width + 3) / 4, rows = (height + 3) / 4;
    if (block_size != columns * rows * block_bytes ||
        output_size != size_t(width) * height * texel_bytes) {
        return 0;
    }
    const uint8_t *source = static_cast<const uint8_t *>(blocks);
    uint8_t *destination = static_cast<uint8_t *>(output);
    for (size_t by = 0; by < rows; ++by) {
        for (size_t bx = 0; bx < columns; ++bx, source += block_bytes) {
            const size_t used_rows = std::min<size_t>(4, height - by * 4);
            const size_t used_columns = std::min<size_t>(4, width - bx * 4);
            uint8_t *first = destination + (by * 4 * width + bx * 4) * texel_bytes;
            if (hdr) {
                uint16_t rgb[16 * 3];
                bcdec_bc6h_half(source, rgb, 4 * 3, format == FrostfieldBC6HSigned);
                for (size_t y = 0; y < used_rows; ++y) {
                    for (size_t x = 0; x < used_columns; ++x) {
                        const uint16_t *pixel = rgb + (y * 4 + x) * 3;
                        const uint16_t rgba[4] = {pixel[0], pixel[1], pixel[2], 0x3c00};
                        std::memcpy(first + (y * width + x) * texel_bytes, rgba, sizeof(rgba));
                    }
                }
            } else {
                alignas(16) uint8_t block[16];
                std::memcpy(block, source, block_bytes);
                basist::color_rgba rgba[16];
                switch (format) {
                case FrostfieldBC1Opaque:
                    basist::bcu::unpack_bc1(block, rgba, true);
                    for (auto &pixel : rgba) pixel.a = 255;
                    break;
                case FrostfieldBC1RGBA: basist::bcu::unpack_bc1(block, rgba, true); break;
                case FrostfieldBC2: basist::bcu::unpack_bc2(block, rgba); break;
                case FrostfieldBC3: basist::bcu::unpack_bc3(block, rgba); break;
                case FrostfieldBC4:
                case FrostfieldBC4Signed:
                case FrostfieldBC5:
                case FrostfieldBC5Signed: {
                    uint8_t red[16], green[16] = {};
                    bcdec__bc4_block(block, red, 4, 1, signed_channels);
                    if (!single_channel) bcdec__bc4_block(block + 8, green, 4, 1, signed_channels);
                    for (size_t i = 0; i < 16; ++i) {
                        rgba[i].set_noclamp_rgba(red[i], green[i], 0, signed_channels ? 127 : 255);
                    }
                    break;
                }
                case FrostfieldBC7:
                    if (!basist::bc7u::unpack_bc7(block, rgba)) {
                        for (auto &pixel : rgba) pixel.set_noclamp_rgba(0, 0, 0, 0);
                    }
                    break;
                default: return 0;
                }
                for (size_t y = 0; y < used_rows; ++y) {
                    std::memcpy(first + y * width * texel_bytes, rgba + y * 4, used_columns * texel_bytes);
                }
            }
        }
    }
    return 1;
}
