#ifndef FROSTFIELD_DRACO_H
#define FROSTFIELD_DRACO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FFDracoMesh FFDracoMesh;
typedef struct FFDracoEncoded FFDracoEncoded;

typedef enum FFDracoStatus {
  FFDracoStatusOK = 0,
  FFDracoStatusInvalid,
  FFDracoStatusUnsupportedGeometry,
  FFDracoStatusSizeLimit,
  FFDracoStatusAttributeRange,
  FFDracoStatusEncodingFailed,
} FFDracoStatus;

typedef struct FFDracoAttribute {
  uint32_t id;
  uint32_t semantic;
  uint32_t component;
  size_t components;
  size_t width;
  bool normalized;
} FFDracoAttribute;

// The returned mesh owns the decoded geometry independently of the input.
FFDracoStatus ff_draco_decode(const void *bytes, size_t count, FFDracoMesh **mesh);
void ff_draco_release(FFDracoMesh *mesh);
size_t ff_draco_point_count(const FFDracoMesh *mesh);
size_t ff_draco_index_count(const FFDracoMesh *mesh);
size_t ff_draco_attribute_count(const FFDracoMesh *mesh);
bool ff_draco_attribute(const FFDracoMesh *mesh, size_t index, FFDracoAttribute *attribute);
bool ff_draco_copy_indices(const FFDracoMesh *mesh, uint32_t *output, size_t count);
bool ff_draco_copy_attribute(const FFDracoMesh *mesh, size_t index, void *output, size_t bytes);

typedef enum FFDracoEncodeMethod {
  FFDracoMeshSequential = 0,
  FFDracoMeshEdgeBreaker,
  FFDracoPointsSequential,
  FFDracoPointsKDTree,
} FFDracoEncodeMethod;

typedef struct FFDracoEncodeOptions {
  FFDracoEncodeMethod method;
  uint32_t encode_speed;
  uint32_t decode_speed;
  uint32_t position_bits;
  uint32_t normal_bits;
  uint32_t uv_bits;
  uint32_t color_bits;
} FFDracoEncodeOptions;

typedef struct FFDracoEncodeInput {
  // Each point stores XYZ, followed by the enabled attributes in normal XYZ,
  // texture UV and color RGBA order. value_count counts float scalars.
  const float *values;
  size_t value_count;
  size_t point_count;
  const uint32_t *indices;
  size_t index_count;
  bool has_normals;
  bool has_uvs;
  bool has_colors;
} FFDracoEncodeInput;

// Inputs are borrowed for this call. The returned handle owns its bytes.
FFDracoStatus ff_draco_encode(const FFDracoEncodeInput *input, const FFDracoEncodeOptions *options,
                             FFDracoEncoded **encoded);
const void *ff_draco_encoded_bytes(const FFDracoEncoded *encoded);
size_t ff_draco_encoded_size(const FFDracoEncoded *encoded);
void ff_draco_release_encoded(FFDracoEncoded *encoded);

#ifdef __cplusplus
}
#endif

#endif
