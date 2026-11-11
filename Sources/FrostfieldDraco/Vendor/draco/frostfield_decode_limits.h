#ifndef FROSTFIELD_DRACO_DECODE_LIMITS_H
#define FROSTFIELD_DRACO_DECODE_LIMITS_H

#include <cstddef>
#include <cstdint>

namespace draco {
namespace frostfield {

// Mesh counts share the glTF loader's limits. Attribute topology is also
// bounded because EdgeBreaker allocates separate seam maps per attribute.
constexpr uint32_t kMaxPoints = 1'000'000;
constexpr uint32_t kMaxFaces = 1'000'000 / 3;
constexpr uint32_t kMaxAttributes = 32;
constexpr uint32_t kMaxComponents = 16;
constexpr uint64_t kMaxTopologyFaceLayers = 4'000'000;
constexpr uint64_t kMaxBytes = 64 * 1024 * 1024;
// The upstream encoder switches from raw to tagged symbols above 18 bits.
constexpr uint32_t kMaxRAnsSymbols = 1 << 18;

}  // namespace frostfield
}  // namespace draco

#endif
