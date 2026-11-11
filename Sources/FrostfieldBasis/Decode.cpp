#include "FrostfieldBasis.h"
#include "Vendor/transcoder/basisu_transcoder.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

struct FrostfieldBasisContext {
    // Some UASTC paths read 32-bit fields directly. A foreign caller can
    // supply a buffer with an unaligned base even for valid KTX2 offsets.
    std::vector<uint8_t> alignedInput;
    basist::ktx2_transcoder decoder;
};

namespace {
constexpr size_t maximumInputBytes = 64 * 1024 * 1024;
constexpr uint32_t maximumDimension = 8192;
constexpr uint32_t maximumLayers = 2048;
constexpr uint64_t maximumPixels = 16 * 1024 * 1024;

uint32_t word(const uint8_t *p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

uint64_t long_word(const uint8_t *p) {
    return uint64_t(word(p)) | uint64_t(word(p + 4)) << 32;
}

bool contains(size_t size, uint64_t offset, uint64_t length) {
    return offset <= size && length <= size - offset;
}

int validate(const uint8_t *bytes, size_t size, bool allowingSpatialTextures, bool allowingHDR) {
    if (!bytes || size < 80 || size > maximumInputBytes) return 1;
    if (std::memcmp(bytes, basist::g_ktx2_file_identifier, 12) || word(bytes + 16) != 1) return 1;
    const uint32_t vkFormat = word(bytes + 12);

    uint32_t width = word(bytes + 20), height = word(bytes + 24);
    uint32_t layers = word(bytes + 32), faces = word(bytes + 36);
    uint32_t levels = word(bytes + 40), scheme = word(bytes + 44);
    if (word(bytes + 28) != 0 || (faces != 1 && faces != 6) ||
        (faces == 6 && layers != 0) || layers > maximumLayers) return 2;
    bool spatial = faces == 6 || layers != 0;
    if (spatial && !allowingSpatialTextures) return 2;
    const uint32_t imagesPerLevel = std::max(layers, 1u) * faces;
    if (!width || !height || width > maximumDimension || height > maximumDimension ||
        uint64_t(width) * height * imagesPerLevel > maximumPixels ||
        !levels || (faces == 6 && width != height)) return 1;

    uint32_t maximumLevels = 1;
    for (uint32_t extent = width > height ? width : height; extent > 1; extent >>= 1) ++maximumLevels;
    if (levels > maximumLevels) return 1;
    const uint32_t indexEnd = 80 + 24 * levels;
    if (size < indexEnd) return 1;

    uint32_t dfd = word(bytes + 48), dfdSize = word(bytes + 52);
    uint32_t kvd = word(bytes + 56), kvdSize = word(bytes + 60);
    uint64_t globalOffset = long_word(bytes + 64), globalSize = long_word(bytes + 72);
    if (!contains(size, dfd, dfdSize) || dfdSize < 28 || dfdSize > 4096 ||
        kvdSize > 256 * 1024 || !contains(size, kvd, kvdSize) ||
        !contains(size, globalOffset, globalSize)) return 1;
    uint8_t model = bytes[dfd + 12], transfer = bytes[dfd + 14], flags = bytes[dfd + 15];
    const bool etc1s = model == basist::KTX2_KDF_DF_MODEL_ETC1S;
    const bool uastc = model == basist::KTX2_KDF_DF_MODEL_UASTC_LDR_4X4;
    const bool hdr4 = model == basist::KTX2_KDF_DF_MODEL_UASTC_HDR_4X4;
    const bool astcHDR = model == basist::KTX2_KDF_DF_MODEL_ASTC;
    const bool hdrIntermediate = model == basist::KTX2_KDF_DF_MODEL_UASTC_HDR_6X6_INTERMEDIATE;
    const bool hdr = hdr4 || astcHDR || hdrIntermediate;
    if ((!etc1s && !uastc && !hdr) || (hdr && !allowingHDR) || flags != 0 ||
        (transfer != 1 && (hdr || transfer != 2))) return 2;
    const uint32_t requiredVkFormat = hdr4 ? basist::KTX2_FORMAT_ASTC_4x4_SFLOAT_BLOCK :
        astcHDR ? basist::KTX2_FORMAT_ASTC_6x6_SFLOAT_BLOCK : 0;
    if (vkFormat != requiredVkFormat) return 1;
    if (etc1s ? scheme != 1 : hdrIntermediate ? (scheme != 4 && scheme != 1) : (scheme != 0 && scheme != 2))
        return 2;
    const uint32_t blockExtent = (astcHDR || hdrIntermediate) ? 6 : 4;
    if (hdr && (bytes[dfd + 16] != blockExtent - 1 || bytes[dfd + 17] != blockExtent - 1 ||
                bytes[dfd + 18] != 0 || bytes[dfd + 19] != 0)) return 1;

    const uint8_t *imageDescriptors = nullptr;
    bool hasAlphaSlice = false;
    const uint32_t hdrDescriptorSize = scheme == 4 ? 12 : 8;
    if (hdrIntermediate) {
        if (globalSize != uint64_t(hdrDescriptorSize) * levels * imagesPerLevel) return 1;
        imageDescriptors = bytes + globalOffset;
    }
    if (etc1s) {
        // The BasisLZ codebook header is followed by one image descriptor
        // for each mip, layer and face.
        uint32_t descriptorBytes = 20 + 20 * levels * imagesPerLevel;
        if (globalSize < descriptorBytes) return 1;
        const auto *global = bytes + globalOffset;
        uint64_t endpointBytes = word(global + 4), selectorBytes = word(global + 8);
        uint64_t tablesBytes = word(global + 12), extendedBytes = word(global + 16);
        if (!endpointBytes || !selectorBytes || !tablesBytes ||
            endpointBytes + selectorBytes + tablesBytes + extendedBytes > globalSize - descriptorBytes) return 1;
        const uint32_t dfdBlockSize = word(bytes + dfd + 8) >> 16;
        if ((dfdBlockSize != 40 && dfdBlockSize != 56) || dfdBlockSize > dfdSize - 4) return 1;
        hasAlphaSlice = dfdBlockSize == 56;
        imageDescriptors = global + 20;
    }

    const uint64_t metadataEnd = std::max({
        uint64_t(indexEnd), uint64_t(dfd) + dfdSize,
        kvdSize ? uint64_t(kvd) + kvdSize : 0, globalSize ? globalOffset + globalSize : 0
    });
    uint64_t precedingLevel = size;
    for (uint32_t level = 0; level < levels; ++level) {
        auto entry = bytes + 80 + level * 24;
        uint64_t offset = long_word(entry), length = long_word(entry + 8), uncompressed = long_word(entry + 16);
        if (!length || !contains(size, offset, length)) return 1;
        if (spatial || hdr) {
            if (offset < metadataEnd || offset + length > precedingLevel) return 1;
            precedingLevel = offset;
        }
        if (etc1s) {
            if (uncompressed != 0) return 1;
            for (uint32_t image = 0; image < imagesPerLevel; ++image) {
                const auto *descriptor = imageDescriptors + 20 * (level * imagesPerLevel + image);
                uint32_t flags = word(descriptor);
                if (flags & basist::KTX2_IMAGE_IS_P_FRAME) return 2;
                if (flags != 0) return 1;
                uint32_t rgbOffset = word(descriptor + 4), rgbLength = word(descriptor + 8);
                uint32_t alphaOffset = word(descriptor + 12), alphaLength = word(descriptor + 16);
                if (!rgbLength || !contains(length, rgbOffset, rgbLength)) return 1;
                if (hasAlphaSlice) {
                    if (!alphaLength || !contains(length, alphaOffset, alphaLength)) return 1;
                } else if (alphaOffset != 0 || alphaLength != 0) {
                    return 1;
                }
            }
        } else if (hdrIntermediate) {
            if (uncompressed != 0) return 1;
            for (uint32_t image = 0; image < imagesPerLevel; ++image) {
                const auto *descriptor = imageDescriptors + hdrDescriptorSize * (level * imagesPerLevel + image);
                const uint32_t sliceOffset = word(descriptor), sliceLength = word(descriptor + 4);
                if (sliceLength < 2 || !contains(length, sliceOffset, sliceLength)) return 1;
                const auto *signature = bytes + offset + sliceOffset;
                const uint32_t profile = uint32_t(signature[0]) | uint32_t(signature[1]) << 8;
                if (profile != 0xabcd && profile != 0xabce) return 2;
                if (scheme == 4 && word(descriptor + 8) != profile) return 1;
            }
        } else {
            uint32_t levelWidth = std::max(1u, width >> level), levelHeight = std::max(1u, height >> level);
            uint64_t blocksX = (levelWidth + blockExtent - 1) / blockExtent;
            uint64_t blocksY = (levelHeight + blockExtent - 1) / blockExtent;
            uint64_t rawSize = 16 * blocksX * blocksY * imagesPerLevel;
            if (uncompressed != rawSize || (scheme == 0 && (length != rawSize || offset % 16 != 0))) return 1;
        }
    }
    return 0;
}
}

int32_t frostfield_basis_validate(const void *data, size_t size) {
    return validate(static_cast<const uint8_t *>(data), size, false, false);
}

FrostfieldBasisContext *frostfield_basis_open(
    const void *data, size_t size, uint8_t allowing_spatial_textures, FrostfieldBasisInfo *info, int32_t *status) {
    if (!info || !status) return nullptr;
    *status = validate(static_cast<const uint8_t *>(data), size, allowing_spatial_textures != 0, true);
    if (*status) return nullptr;
    *status = 1;
    try {
        static std::once_flag initialized;
        std::call_once(initialized, [] { basist::basisu_transcoder_init(); });
        auto context = std::make_unique<FrostfieldBasisContext>();
        auto &decoder = context->decoder;
        if (reinterpret_cast<uintptr_t>(data) % alignof(uint32_t) != 0) {
            const auto *bytes = static_cast<const uint8_t *>(data);
            context->alignedInput.assign(bytes, bytes + size);
            data = context->alignedInput.data();
        }
        if (!decoder.init(data, static_cast<uint32_t>(size)) ||
            (!decoder.is_etc1s() && !decoder.is_uastc() && !decoder.is_hdr())) return nullptr;
        if (decoder.get_layers() && decoder.find_key("KTXanimData")) {
            *status = 2;
            return nullptr;
        }
        if (!decoder.start_transcoding()) return nullptr;
        *info = {
            decoder.get_width(), decoder.get_height(), decoder.get_levels(),
            decoder.get_layers(), decoder.get_faces(), uint8_t(decoder.is_srgb()),
            uint8_t(decoder.is_uastc()), uint8_t(decoder.get_has_alpha() != 0), uint8_t(decoder.is_hdr())
        };
        *status = 0;
        return context.release();
    } catch (...) {
        return nullptr;
    }
}

int32_t frostfield_basis_transcode(
    FrostfieldBasisContext *context, FrostfieldBasisFormat format, uint32_t level, uint32_t layer,
    uint32_t face, void *output, size_t size) {
    if (!context || !output || level >= context->decoder.get_levels() ||
        layer >= std::max(context->decoder.get_layers(), 1u) || face >= context->decoder.get_faces()) return 1;
    try {
        using Target = basist::transcoder_texture_format;
        if ((format == FrostfieldBasisRGBA16Float) != context->decoder.is_hdr()) return 1;
        Target target;
        switch (format) {
        case FrostfieldBasisRGBA8: target = Target::cTFRGBA32; break;
        case FrostfieldBasisETC1RGB:
            if (context->decoder.get_has_alpha()) return 1;
            target = Target::cTFETC1_RGB;
            break;
        case FrostfieldBasisETC2RGBA: target = Target::cTFETC2_RGBA; break;
        case FrostfieldBasisASTC4x4: target = Target::cTFASTC_LDR_4x4_RGBA; break;
        case FrostfieldBasisBC7: target = Target::cTFBC7_RGBA; break;
        case FrostfieldBasisRGBA16Float: target = Target::cTFRGBA_HALF; break;
        default: return 1;
        }
        if (!basist::basis_is_format_supported(target, context->decoder.get_basis_tex_format())) return 1;
        basist::ktx2_image_level_info image;
        if (!context->decoder.get_image_level_info(image, level, layer, face)) return 1;
        const bool raw = format == FrostfieldBasisRGBA8 || format == FrostfieldBasisRGBA16Float;
        const uint32_t rowUnits = raw ? image.m_orig_width : image.m_num_blocks_x;
        const uint32_t rows = raw ? image.m_orig_height : image.m_num_blocks_y;
        const uint64_t units = uint64_t(rowUnits) * rows;
        if (!units || units > maximumPixels || size != units * basist::basis_get_bytes_per_block_or_pixel(target)) return 1;
        return context->decoder.transcode_image_level(
            level, layer, face, output, static_cast<uint32_t>(units), target,
            0, rowUnits, raw ? rows : 0
        ) ? 0 : 1;
    } catch (...) {
        return 1;
    }
}

void frostfield_basis_close(FrostfieldBasisContext *context) {
    delete context;
}
