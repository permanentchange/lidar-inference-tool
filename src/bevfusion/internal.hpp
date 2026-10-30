#pragma once
#include <cuda_runtime.h>

#include <map>
#include <stdexcept>

#include "bevfusion_inference.hpp"

namespace inference::bev {
/// Floating-point storage used at dense network boundaries.
enum class ScalarType { Float32, Float16 };
inline size_t ScalarBytes(ScalarType type) {
  return type == ScalarType::Float16 ? 2 : 4;
}
inline void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
  void* data = nullptr;
  size_t capacity = 0;
  ~Buffer() {
    if (data) cudaFree(data);
  }
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void resize(size_t bytes) {
    if (bytes > capacity) {
      if (data) Check(cudaFree(data));
      data = nullptr;
      capacity = 0;
      Check(cudaMalloc(&data, bytes));
      capacity = bytes;
    }
  }
  template <class T>
  T* ptr() {
    return static_cast<T*>(data);
  }
};
struct Kernels {
  Buffer source_points, valid_points, points, keys, order, starts, first,
      groups, features, indices;
  Buffer depth, owners, projection, pixels, collision_count;
  Buffer geometry, ranks, geometry_order, pool_starts, pool_ends, pooled;
  Buffer camera_mask;
  Buffer rgb, horizontal, weights, bounds, image_float;
  int voxel_count = 0, total_voxel_count = 0;
};
size_t FilterPoints(Kernels&, const BevFusionConfig&, const std::vector<float>&,
                    cudaStream_t);
void Voxelize(Kernels&, const BevFusionConfig&, int n, cudaStream_t);
size_t Depth(Kernels&, const BevFusionConfig&, const BevFusionFrame&, int n,
             cudaStream_t, ScalarType = ScalarType::Float32);
void Geometry(Kernels&, const BevFusionConfig&, const BevFusionFrame&,
              cudaStream_t);
void Pool(Kernels&, const BevFusionConfig&, const void* feature,
          const void* depth, cudaStream_t,
          ScalarType input_type = ScalarType::Float32,
          ScalarType output_type = ScalarType::Float32);
void Preprocess(Kernels&, const BevFusionConfig&, const BevFusionFrame&,
                void* output, cudaStream_t, ScalarType = ScalarType::Float32);
void HalfToFloat(const void*, float*, size_t, cudaStream_t);
void FloatToHalf(const float*, void*, size_t, cudaStream_t);
void MaskCameraOutputs(const BevFusionConfig&, const BevFusionFrame&, void*, void*,
                       ScalarType, cudaStream_t);
std::array<float, 16> Inverse(const std::array<float, 16>&);
}  // namespace inference::bev
