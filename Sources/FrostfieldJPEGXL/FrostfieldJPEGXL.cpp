#include "FrostfieldJPEGXL.h"

#include <jxl/cms.h>
#include <jxl/decode.h>
#include "skcms.h"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>

namespace {

constexpr size_t kMaximumDecoderBytes = 512u * 1024u * 1024u;
constexpr size_t kMaximumInputBytes = 64u * 1024u * 1024u;
constexpr size_t kMaximumICCBytes = 16u * 1024u * 1024u;
constexpr float kReferenceWhiteNits = 203.0f;

struct alignas(std::max_align_t) Allocation {
  size_t bytes;
};

struct DecoderBudget {
  size_t allocated = 0;
  bool exhausted = false;
};

void *Allocate(void *opaque, size_t bytes) {
  auto &budget = *static_cast<DecoderBudget *>(opaque);
  if (bytes > kMaximumDecoderBytes - sizeof(Allocation) ||
      bytes + sizeof(Allocation) > kMaximumDecoderBytes - budget.allocated) {
    budget.exhausted = true;
    return nullptr;
  }
  const size_t total = bytes + sizeof(Allocation);
  auto *entry = static_cast<Allocation *>(std::malloc(total));
  if (!entry) {
    budget.exhausted = true;
    return nullptr;
  }
  entry->bytes = total;
  budget.allocated += total;
  return entry + 1;
}

void Release(void *opaque, void *address) {
  if (!address) return;
  auto *entry = static_cast<Allocation *>(address) - 1;
  static_cast<DecoderBudget *>(opaque)->allocated -= entry->bytes;
  std::free(entry);
}

}  // namespace

FrostfieldJPEGXLStatus FrostfieldJPEGXLDecode(
    const uint8_t *input, size_t input_count, uint32_t maximum_side,
    size_t maximum_pixels, FrostfieldJPEGXLImage *result) {
  if (!result || !input || input_count == 0) return FrostfieldJPEGXLInvalid;
  *result = {};
  if (input_count > kMaximumInputBytes || maximum_side == 0 ||
      maximum_pixels == 0 || maximum_pixels > SIZE_MAX / (4 * sizeof(float))) {
    return FrostfieldJPEGXLLimit;
  }

  DecoderBudget budget;
  const JxlMemoryManager memory = {&budget, Allocate, Release};
  std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> decoder(
      JxlDecoderCreate(&memory), &JxlDecoderDestroy);
  if (!decoder) return FrostfieldJPEGXLLimit;

  auto failure = [&] {
    return budget.exhausted ? FrostfieldJPEGXLLimit : FrostfieldJPEGXLInvalid;
  };
  const int events = JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING | JXL_DEC_FULL_IMAGE;
  if (JxlDecoderSubscribeEvents(decoder.get(), events) != JXL_DEC_SUCCESS ||
      JxlDecoderSetCms(decoder.get(), *JxlGetDefaultCms()) != JXL_DEC_SUCCESS ||
      JxlDecoderSetUnpremultiplyAlpha(decoder.get(), JXL_TRUE) != JXL_DEC_SUCCESS ||
      JxlDecoderSetInput(decoder.get(), input, input_count) != JXL_DEC_SUCCESS) {
    return failure();
  }
  JxlDecoderCloseInput(decoder.get());

  JxlBasicInfo info = {};
  const JxlPixelFormat format = {4, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  std::unique_ptr<float, decltype(&std::free)> pixels(nullptr, &std::free);
  bool has_basic_info = false;
  bool has_color_profile = false;
  float reference_headroom = 0;

  for (;;) {
    const JxlDecoderStatus status = JxlDecoderProcessInput(decoder.get());
    switch (status) {
      case JXL_DEC_BASIC_INFO:
        if (has_basic_info ||
            JxlDecoderGetBasicInfo(decoder.get(), &info) != JXL_DEC_SUCCESS) return failure();
        if (info.have_animation) return FrostfieldJPEGXLAnimated;
        result->width = info.xsize;
        result->height = info.ysize;
        if (info.xsize == 0 || info.ysize == 0 ||
            info.xsize > maximum_side || info.ysize > maximum_side ||
            info.xsize > maximum_pixels / info.ysize) {
          return FrostfieldJPEGXLLimit;
        }
        has_basic_info = true;
        break;
      case JXL_DEC_COLOR_ENCODING: {
        if (!has_basic_info || has_color_profile) return failure();
        bool display_hdr = false;
        JxlColorEncoding original;
        if (JxlDecoderGetColorAsEncodedProfile(
                decoder.get(), JXL_COLOR_PROFILE_TARGET_ORIGINAL, &original) == JXL_DEC_SUCCESS) {
          display_hdr = original.transfer_function == JXL_TRANSFER_FUNCTION_PQ ||
                        original.transfer_function == JXL_TRANSFER_FUNCTION_HLG;
        } else {
          size_t icc_size = 0;
          if (JxlDecoderGetICCProfileSize(
                  decoder.get(), JXL_COLOR_PROFILE_TARGET_ORIGINAL, &icc_size) != JXL_DEC_SUCCESS) {
            return FrostfieldJPEGXLColorProfile;
          }
          if (icc_size == 0 || icc_size > kMaximumICCBytes) return FrostfieldJPEGXLLimit;
          std::unique_ptr<uint8_t, decltype(&std::free)> icc(
              static_cast<uint8_t *>(std::malloc(icc_size)), &std::free);
          if (!icc) return FrostfieldJPEGXLLimit;
          if (JxlDecoderGetColorAsICCProfile(
                  decoder.get(), JXL_COLOR_PROFILE_TARGET_ORIGINAL, icc.get(), icc_size) != JXL_DEC_SUCCESS) {
            return FrostfieldJPEGXLColorProfile;
          }
          skcms_ICCProfile profile;
          if (skcms_Parse(icc.get(), icc_size, &profile) && profile.has_CICP) {
            // CICP transfer characteristics 16 and 18 identify PQ and HLG.
            const auto transfer = profile.CICP.transfer_characteristics;
            display_hdr = transfer == 16 || transfer == 18;
          }
        }
        if (display_hdr) {
          if (!std::isfinite(info.intensity_target) || info.intensity_target <= 0) {
            return FrostfieldJPEGXLColorProfile;
          }
          reference_headroom = info.intensity_target / kReferenceWhiteNits;
        }
        JxlColorEncoding linear = {};
        linear.color_space = info.num_color_channels == 1 ? JXL_COLOR_SPACE_GRAY : JXL_COLOR_SPACE_RGB;
        linear.white_point = JXL_WHITE_POINT_D65;
        linear.primaries = JXL_PRIMARIES_SRGB;
        linear.transfer_function = JXL_TRANSFER_FUNCTION_LINEAR;
        linear.rendering_intent = JXL_RENDERING_INTENT_RELATIVE;
        if (JxlDecoderSetOutputColorProfile(decoder.get(), &linear, nullptr, 0) != JXL_DEC_SUCCESS) {
          return budget.exhausted ? FrostfieldJPEGXLLimit : FrostfieldJPEGXLColorProfile;
        }
        has_color_profile = true;
        break;
      }
      case JXL_DEC_NEED_IMAGE_OUT_BUFFER: {
        size_t size = 0;
        if (!has_basic_info || !has_color_profile || pixels ||
            JxlDecoderImageOutBufferSize(decoder.get(), &format, &size) != JXL_DEC_SUCCESS ||
            size != static_cast<size_t>(info.xsize) * info.ysize * 4 * sizeof(float)) {
          return failure();
        }
        pixels.reset(static_cast<float *>(std::malloc(size)));
        if (!pixels) return FrostfieldJPEGXLLimit;
        if (JxlDecoderSetImageOutBuffer(decoder.get(), &format, pixels.get(), size) != JXL_DEC_SUCCESS) return failure();
        break;
      }
      case JXL_DEC_FULL_IMAGE:
        if (!pixels || !has_color_profile) return failure();
        // libjxl reports display-referred values relative to the source intensity
        // target. Rebase tagged HDR transfer functions to diffuse reference white.
        if (reference_headroom > 0) {
          const size_t samples = static_cast<size_t>(info.xsize) * info.ysize * 4;
          for (size_t i = 0; i < samples; i += 4) {
            pixels.get()[i] *= reference_headroom;
            pixels.get()[i + 1] *= reference_headroom;
            pixels.get()[i + 2] *= reference_headroom;
          }
        }
        result->width = info.xsize;
        result->height = info.ysize;
        result->rgba = pixels.release();
        result->reference_headroom = reference_headroom;
        return FrostfieldJPEGXLSuccess;
      default:
        return failure();
    }
  }
}

void FrostfieldJPEGXLFree(float *rgba) {
  std::free(rgba);
}
