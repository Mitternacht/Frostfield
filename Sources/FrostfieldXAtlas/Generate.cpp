#include "FrostfieldXAtlas.h"
#include "Vendor/xatlas.h"
#include <cmath>
#include <memory>

FFUVAtlasStatus ff_uv_atlas_create(
    const void *positions, const void *normals, uint32_t vertexCount, uint32_t vertexStride,
    const uint32_t *indices, uint32_t indexCount, const uint32_t *faceGroups,
    uint32_t resolution, uint32_t padding, float texelsPerUnit, FFUVAtlas **result
) {
    if (!result)
        return FFUVAtlasStatusInvalid;
    *result = nullptr;
    if (!positions || !normals || !indices || !vertexCount || !indexCount || indexCount % 3
        || vertexStride < 3 * sizeof(float) || resolution < 16 || resolution > 4096
        || padding > 64 || padding * 2 + 2 >= resolution
        || !std::isfinite(texelsPerUnit) || texelsPerUnit <= 0)
        return FFUVAtlasStatusInvalid;

    std::unique_ptr<xatlas::Atlas, decltype(&xatlas::Destroy)> atlas(xatlas::Create(), xatlas::Destroy);
    xatlas::MeshDecl input;
    input.vertexCount = vertexCount;
    input.vertexPositionData = positions;
    input.vertexPositionStride = vertexStride;
    input.vertexNormalData = normals;
    input.vertexNormalStride = vertexStride;
    input.indexCount = indexCount;
    input.indexData = indices;
    input.indexFormat = xatlas::IndexFormat::UInt32;
    input.faceMaterialData = faceGroups;
    if (xatlas::AddMesh(atlas.get(), input) != xatlas::AddMeshError::Success)
        return FFUVAtlasStatusInvalid;
    xatlas::ComputeCharts(atlas.get());

    xatlas::PackOptions options;
    options.resolution = resolution;
    options.maxChartSize = resolution - 2;
    options.padding = padding;
    options.texelsPerUnit = texelsPerUnit;
    // Retain the chart parameterization and lower density only when it will not fit.
    for (int attempt = 0; attempt < 10 && atlas->meshCount == 0; ++attempt) {
        xatlas::PackCharts(atlas.get(), options);
        options.texelsPerUnit *= 0.72f;
    }
    if (!atlas->meshCount)
        return FFUVAtlasStatusCannotPack;
    if (atlas->meshCount != 1 || atlas->atlasCount != 1 || atlas->width != resolution
        || atlas->height != resolution || !atlas->meshes || !atlas->chartCount)
        return FFUVAtlasStatusInvalid;

    const xatlas::Mesh &mesh = atlas->meshes[0];
    if (mesh.indexCount != indexCount || !mesh.indexArray || !mesh.vertexArray || !mesh.chartArray)
        return FFUVAtlasStatusInvalid;
    for (uint32_t chart = 0; chart < mesh.chartCount; ++chart)
        if (mesh.chartArray[chart].type == xatlas::ChartType::Invalid)
            return FFUVAtlasStatusInvalid;
    for (uint32_t index = 0; index < indexCount; ++index) {
        if (mesh.indexArray[index] >= mesh.vertexCount
            || mesh.vertexArray[mesh.indexArray[index]].xref != indices[index])
            return FFUVAtlasStatusInvalid;
    }
    *result = reinterpret_cast<FFUVAtlas *>(atlas.release());
    return FFUVAtlasStatusOK;
}

void ff_uv_atlas_destroy(FFUVAtlas *atlas) {
    xatlas::Destroy(reinterpret_cast<xatlas::Atlas *>(atlas));
}

uint32_t ff_uv_atlas_vertex_count(const FFUVAtlas *atlas) {
    return reinterpret_cast<const xatlas::Atlas *>(atlas)->meshes[0].vertexCount;
}

uint32_t ff_uv_atlas_chart_count(const FFUVAtlas *atlas) {
    return reinterpret_cast<const xatlas::Atlas *>(atlas)->chartCount;
}

const uint32_t *ff_uv_atlas_indices(const FFUVAtlas *atlas) {
    return reinterpret_cast<const xatlas::Atlas *>(atlas)->meshes[0].indexArray;
}

FFUVAtlasVertex ff_uv_atlas_vertex(const FFUVAtlas *atlas, uint32_t index) {
    const xatlas::Vertex &vertex = reinterpret_cast<const xatlas::Atlas *>(atlas)->meshes[0].vertexArray[index];
    return {vertex.xref, vertex.chartIndex, vertex.uv[0], vertex.uv[1]};
}
