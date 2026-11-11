#include "FrostfieldDraco.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "draco/compression/mesh/mesh_edgebreaker_encoder.h"
#include "draco/compression/mesh/mesh_sequential_encoder.h"
#include "draco/compression/point_cloud/point_cloud_kd_tree_encoder.h"
#include "draco/compression/point_cloud/point_cloud_sequential_encoder.h"
#include "draco/frostfield_decode_limits.h"

struct FFDracoEncoded {
  draco::EncoderBuffer bytes;
};

namespace {

using draco::GeometryAttribute;
using draco::frostfield::kMaxBytes;
using draco::frostfield::kMaxFaces;
using draco::frostfield::kMaxPoints;

struct AttributeLayout {
  GeometryAttribute::Type semantic;
  uint8_t components;
  uint32_t quantization;
};

bool IsMesh(FFDracoEncodeMethod method) {
  return method == FFDracoMeshSequential || method == FFDracoMeshEdgeBreaker;
}

bool ValidOptions(const FFDracoEncodeOptions &options) {
  return (IsMesh(options.method) || options.method == FFDracoPointsSequential ||
          options.method == FFDracoPointsKDTree) &&
         options.encode_speed <= 10 && options.decode_speed <= 10 &&
         options.position_bits <= 30 && options.normal_bits <= 30 &&
         options.normal_bits != 1 && options.uv_bits <= 30 && options.color_bits <= 30;
}

FFDracoStatus BuildFaces(const FFDracoEncodeInput &input, FFDracoEncodeMethod method,
                         draco::Mesh *mesh) {
  const bool edge_breaker = method == FFDracoMeshEdgeBreaker;
  std::vector<bool> referenced(edge_breaker ? input.point_count : 0, false);
  mesh->SetNumFaces(input.index_count / 3);
  for (size_t offset = 0; offset < input.index_count; offset += 3) {
    const uint32_t a = input.indices[offset], b = input.indices[offset + 1], c = input.indices[offset + 2];
    if (a >= input.point_count || b >= input.point_count || c >= input.point_count) return FFDracoStatusInvalid;
    if (edge_breaker) {
      // EdgeBreaker discards degenerate faces and isolated vertices. Require
      // sequential encoding when the caller needs to preserve either.
      if (a == b || b == c || c == a) return FFDracoStatusUnsupportedGeometry;
      referenced[a] = referenced[b] = referenced[c] = true;
    }
    mesh->SetFace(draco::FaceIndex(static_cast<uint32_t>(offset / 3)),
                  {draco::PointIndex(a), draco::PointIndex(b), draco::PointIndex(c)});
  }
  if (edge_breaker && std::find(referenced.begin(), referenced.end(), false) != referenced.end()) {
    return FFDracoStatusUnsupportedGeometry;
  }
  return FFDracoStatusOK;
}

FFDracoStatus AddAttribute(const FFDracoEncodeInput &input, AttributeLayout layout, size_t offset,
                           size_t stride, draco::PointCloud *geometry, draco::EncoderOptions *options) {
  const uint32_t point_count = static_cast<uint32_t>(input.point_count);
  GeometryAttribute attribute;
  attribute.Init(layout.semantic, nullptr, layout.components, draco::DT_FLOAT32, false,
                 layout.components * sizeof(float), 0);
  const int id = geometry->AddAttribute(attribute, true, point_count);
  if (id < 0) return FFDracoStatusEncodingFailed;
  auto *destination = geometry->attribute(id);
  const bool octahedral = layout.semantic == GeometryAttribute::NORMAL && layout.quantization != 0;
  std::array<float, 4> minimum, maximum;
  minimum.fill(std::numeric_limits<float>::infinity());
  maximum.fill(-std::numeric_limits<float>::infinity());
  for (uint32_t point = 0; point < point_count; ++point) {
    const float *values = input.values + point * stride + offset;
    for (uint8_t component = 0; component < layout.components; ++component) {
      if (!std::isfinite(values[component])) return FFDracoStatusAttributeRange;
      minimum[component] = std::min(minimum[component], values[component]);
      maximum[component] = std::max(maximum[component], values[component]);
    }
    const draco::AttributeValueIndex index(point);
    if (octahedral) {
      // Normalize in Double so very small nonzero normals keep their direction.
      const double length = std::hypot(static_cast<double>(values[0]), static_cast<double>(values[1]),
                                       static_cast<double>(values[2]));
      if (length == 0) return FFDracoStatusAttributeRange;
      const float normal[3] = {static_cast<float>(values[0] / length), static_cast<float>(values[1] / length),
                               static_cast<float>(values[2] / length)};
      destination->SetAttributeValue(index, normal);
    } else {
      destination->SetAttributeValue(index, values);
    }
  }
  if (layout.quantization > 0 && !octahedral) {
    float range = 0;
    for (uint8_t component = 0; component < layout.components; ++component) {
      range = std::max(range, maximum[component] - minimum[component]);
    }
    // Draco stores the range and inverse quantization step in Float.
    const float levels = static_cast<float>((uint32_t{1} << layout.quantization) - 1);
    if (!std::isfinite(range) || (range > 0 && !std::isfinite(levels / range))) {
      return FFDracoStatusAttributeRange;
    }
  }
  if (layout.quantization > 0) {
    options->SetAttributeInt(id, "quantization_bits", layout.quantization);
    if (layout.quantization > 18 ||
        (octahedral && options->GetAttributeInt(0, "quantization_bits", 0) > 20)) {
      // Slow mesh predictors can allocate by correction magnitude. Geometric
      // normal prediction also sums integer triangle areas in int64.
      options->SetAttributeInt(id, "prediction_scheme", draco::PREDICTION_DIFFERENCE);
    }
  }
  return FFDracoStatusOK;
}

}  // namespace

FFDracoStatus ff_draco_encode(const FFDracoEncodeInput *input, const FFDracoEncodeOptions *options,
                             FFDracoEncoded **encoded) {
  if (encoded == nullptr) return FFDracoStatusInvalid;
  *encoded = nullptr;
  if (input == nullptr || options == nullptr || !ValidOptions(*options)) return FFDracoStatusInvalid;
  if (input->point_count > kMaxPoints || input->index_count > size_t{kMaxFaces} * 3 ||
      input->value_count > kMaxBytes / sizeof(float)) {
    return FFDracoStatusSizeLimit;
  }

  const bool mesh_input = IsMesh(options->method);
  if (input->point_count == 0 || input->values == nullptr ||
      (mesh_input && (input->indices == nullptr || input->index_count == 0 || input->index_count % 3 != 0)) ||
      (!mesh_input && (input->index_count != 0 || input->has_normals || input->has_uvs))) {
    return FFDracoStatusInvalid;
  }
  if (options->method == FFDracoPointsKDTree &&
      (options->position_bits == 0 || (input->has_colors && options->color_bits == 0))) {
    return FFDracoStatusInvalid;
  }

  std::array<AttributeLayout, 4> layout;
  size_t attribute_count = 0;
  layout[attribute_count++] = {GeometryAttribute::POSITION, 3, options->position_bits};
  if (input->has_normals) layout[attribute_count++] = {GeometryAttribute::NORMAL, 3, options->normal_bits};
  if (input->has_uvs) layout[attribute_count++] = {GeometryAttribute::TEX_COORD, 2, options->uv_bits};
  if (input->has_colors) layout[attribute_count++] = {GeometryAttribute::COLOR, 4, options->color_bits};
  size_t stride = 0;
  for (size_t i = 0; i < attribute_count; ++i) stride += layout[i].components;
  if (input->value_count != input->point_count * stride) return FFDracoStatusInvalid;

  try {
    draco::Mesh mesh;
    draco::PointCloud points;
    draco::PointCloud *geometry = mesh_input ? static_cast<draco::PointCloud *>(&mesh) : &points;
    geometry->set_num_points(static_cast<uint32_t>(input->point_count));
    if (mesh_input) {
      const FFDracoStatus status = BuildFaces(*input, options->method, &mesh);
      if (status != FFDracoStatusOK) return status;
    }
    auto settings = draco::EncoderOptions::CreateDefaultOptions();
    settings.SetSpeed(options->encode_speed, options->decode_speed);
    settings.SetGlobalBool("store_number_of_encoded_points", true);
    size_t offset = 0;
    for (size_t i = 0; i < attribute_count; ++i) {
      const FFDracoStatus status = AddAttribute(*input, layout[i], offset, stride, geometry, &settings);
      if (status != FFDracoStatusOK) return status;
      offset += layout[i].components;
    }

    std::unique_ptr<draco::PointCloudEncoder> encoder;
    if (mesh_input) {
      std::unique_ptr<draco::MeshEncoder> mesh_encoder;
      if (options->method == FFDracoMeshEdgeBreaker) {
        mesh_encoder = std::make_unique<draco::MeshEdgebreakerEncoder>();
      } else {
        mesh_encoder = std::make_unique<draco::MeshSequentialEncoder>();
      }
      mesh_encoder->SetMesh(mesh);
      encoder = std::move(mesh_encoder);
    } else {
      if (options->method == FFDracoPointsKDTree) {
        encoder = std::make_unique<draco::PointCloudKdTreeEncoder>();
      } else {
        encoder = std::make_unique<draco::PointCloudSequentialEncoder>();
      }
      encoder->SetPointCloud(points);
    }
    auto output = std::make_unique<FFDracoEncoded>();
    if (!encoder->Encode(settings, &output->bytes).ok() || output->bytes.size() == 0) {
      return FFDracoStatusEncodingFailed;
    }
    if (encoder->num_encoded_points() > kMaxPoints || output->bytes.size() > kMaxBytes) {
      return FFDracoStatusSizeLimit;
    }
    *encoded = output.release();
    return FFDracoStatusOK;
  } catch (const std::bad_alloc &) {
    return FFDracoStatusSizeLimit;
  } catch (const std::exception &) {
    return FFDracoStatusEncodingFailed;
  }
}

const void *ff_draco_encoded_bytes(const FFDracoEncoded *encoded) {
  return encoded == nullptr ? nullptr : encoded->bytes.data();
}

size_t ff_draco_encoded_size(const FFDracoEncoded *encoded) {
  return encoded == nullptr ? 0 : encoded->bytes.size();
}

void ff_draco_release_encoded(FFDracoEncoded *encoded) { delete encoded; }
