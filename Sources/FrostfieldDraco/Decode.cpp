#include "FrostfieldDraco.h"

#include <cstdint>
#include <exception>
#include <memory>
#include <new>

#include "draco/compression/decode.h"
#include "draco/frostfield_decode_limits.h"

struct FFDracoMesh {
  std::unique_ptr<draco::Mesh> value;
};

FFDracoStatus ff_draco_decode(const void *bytes, size_t count, FFDracoMesh **mesh) {
  if (mesh == nullptr) return FFDracoStatusInvalid;
  *mesh = nullptr;
  if (count > draco::frostfield::kMaxBytes) return FFDracoStatusSizeLimit;
  if (bytes == nullptr || count == 0) return FFDracoStatusInvalid;
  try {
    draco::DecoderBuffer buffer;
    buffer.Init(static_cast<const char *>(bytes), count);
    const auto kind = draco::Decoder::GetEncodedGeometryType(&buffer);
    if (!kind.ok()) return FFDracoStatusInvalid;
    if (kind.value() != draco::TRIANGULAR_MESH) return FFDracoStatusUnsupportedGeometry;

    draco::Decoder decoder;
    auto decoded = decoder.DecodeMeshFromBuffer(&buffer);
    if (!decoded.ok()) return FFDracoStatusInvalid;
    auto geometry = std::move(decoded).value();
    if (!geometry || geometry->num_points() == 0 || geometry->num_faces() == 0) {
      return FFDracoStatusInvalid;
    }
    *mesh = new FFDracoMesh{std::move(geometry)};
    return FFDracoStatusOK;
  } catch (const std::bad_alloc &) {
    return FFDracoStatusSizeLimit;
  } catch (const std::exception &) {
    return FFDracoStatusInvalid;
  }
}

void ff_draco_release(FFDracoMesh *mesh) { delete mesh; }

size_t ff_draco_point_count(const FFDracoMesh *mesh) { return mesh->value->num_points(); }
size_t ff_draco_index_count(const FFDracoMesh *mesh) { return 3 * mesh->value->num_faces(); }
size_t ff_draco_attribute_count(const FFDracoMesh *mesh) { return mesh->value->num_attributes(); }

bool ff_draco_attribute(const FFDracoMesh *mesh, size_t index, FFDracoAttribute *attribute) {
  if (index >= ff_draco_attribute_count(mesh) || attribute == nullptr) return false;
  const auto *source = mesh->value->attribute(static_cast<int>(index));
  const size_t width = draco::DataTypeLength(source->data_type());
  if (width == 0 || source->num_components() == 0) return false;
  *attribute = FFDracoAttribute{
      source->unique_id(),
      static_cast<uint32_t>(source->attribute_type()),
      static_cast<uint32_t>(source->data_type()),
      source->num_components(),
      width,
      source->normalized(),
  };
  return true;
}

bool ff_draco_copy_indices(const FFDracoMesh *mesh, uint32_t *output, size_t count) {
  if (output == nullptr || count != ff_draco_index_count(mesh)) return false;
  for (draco::FaceIndex face(0); face < mesh->value->num_faces(); ++face) {
    for (const auto point : mesh->value->face(face)) {
      if (point.value() >= mesh->value->num_points()) return false;
      *output++ = point.value();
    }
  }
  return true;
}

bool ff_draco_copy_attribute(const FFDracoMesh *mesh, size_t index, void *output, size_t bytes) {
  FFDracoAttribute info;
  if (output == nullptr || !ff_draco_attribute(mesh, index, &info)) return false;
  const size_t count = ff_draco_point_count(mesh);
  const size_t element_bytes = info.width * info.components;
  if (bytes != count * element_bytes) return false;

  const auto *source = mesh->value->attribute(static_cast<int>(index));
  const auto *buffer = source->buffer();
  if (buffer == nullptr || source->byte_offset() < 0 ||
      source->byte_stride() < static_cast<int64_t>(element_bytes) ||
      (!source->is_mapping_identity() && source->indices_map_size() != count)) {
    return false;
  }
  const size_t offset = source->byte_offset();
  const size_t stride = source->byte_stride();
  if (offset > buffer->data_size() || element_bytes > buffer->data_size() - offset) return false;

  const size_t max_index = (buffer->data_size() - offset - element_bytes) / stride;
  auto *destination = static_cast<uint8_t *>(output);
  for (draco::PointIndex point(0); point < count; ++point) {
    const size_t entry = source->mapped_index(point).value();
    if (entry >= source->size() || entry > max_index) return false;
    buffer->Read(offset + stride * entry, destination, element_bytes);
    destination += element_bytes;
  }
  return true;
}
