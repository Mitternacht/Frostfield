#ifndef FROSTFIELD_XATLAS_H
#define FROSTFIELD_XATLAS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FFUVAtlas FFUVAtlas;
typedef enum FFUVAtlasStatus {
    FFUVAtlasStatusOK = 0,
    FFUVAtlasStatusInvalid,
    FFUVAtlasStatusCannotPack,
} FFUVAtlasStatus;

typedef struct FFUVAtlasVertex {
    uint32_t sourceIndex;
    int32_t chartIndex;
    float u;
    float v;
} FFUVAtlasVertex;

FFUVAtlasStatus ff_uv_atlas_create(
    const void *positions, const void *normals, uint32_t vertexCount, uint32_t vertexStride,
    const uint32_t *indices, uint32_t indexCount, const uint32_t *faceGroups,
    uint32_t resolution, uint32_t padding, float texelsPerUnit, FFUVAtlas **result
);
void ff_uv_atlas_destroy(FFUVAtlas *atlas);
uint32_t ff_uv_atlas_vertex_count(const FFUVAtlas *atlas);
uint32_t ff_uv_atlas_chart_count(const FFUVAtlas *atlas);
const uint32_t *ff_uv_atlas_indices(const FFUVAtlas *atlas);
FFUVAtlasVertex ff_uv_atlas_vertex(const FFUVAtlas *atlas, uint32_t index);

#ifdef __cplusplus
}
#endif

#endif
