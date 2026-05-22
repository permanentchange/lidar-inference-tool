#include "pillar_postprocess/pillar_postprocess.hpp"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

namespace inference_internal {
namespace {

constexpr int kBoxDim = 7;
constexpr int kBBoxChannelsPerTask = 8;
constexpr int kDirChannelsPerTask = 2;

struct DeviceCandidate {
  float score = -1.0f;
  int label = 0;
  float box[kBoxDim]{};
};

InferenceStatus SetError(std::string* out_error,
                         InferenceStatus status,
                         const std::string& message) {
  if (out_error != nullptr) {
    *out_error = message;
  }
  return status;
}

InferenceStatus CheckCuda(cudaError_t error,
                          const std::string& operation,
                          std::string* out_error) {
  if (error == cudaSuccess) {
    return INFERENCE_STATUS_OK;
  }
  std::ostringstream oss;
  oss << operation << " failed: " << cudaGetErrorString(error);
  return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
}

template <typename T>
InferenceStatus AllocateDeviceBuffer(size_t count,
                                     const std::string& name,
                                     T** out_buffer,
                                     std::string* out_error) {
  if (out_buffer == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    name + " output buffer pointer is null");
  }
  *out_buffer = nullptr;
  if (count == 0) {
    return INFERENCE_STATUS_OK;
  }
  return CheckCuda(cudaMalloc(reinterpret_cast<void**>(out_buffer), count * sizeof(T)),
                   "cudaMalloc(" + name + ")", out_error);
}

template <typename T>
InferenceStatus AllocatePinnedHostBuffer(size_t count,
                                         const std::string& name,
                                         T** out_buffer,
                                         std::string* out_error) {
  if (out_buffer == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    name + " output buffer pointer is null");
  }
  *out_buffer = nullptr;
  if (count == 0) {
    return INFERENCE_STATUS_OK;
  }
  return CheckCuda(cudaMallocHost(reinterpret_cast<void**>(out_buffer), count * sizeof(T)),
                   "cudaMallocHost(" + name + ")", out_error);
}

template <typename T>
void FreeDeviceBuffer(T** buffer) {
  if (buffer != nullptr && *buffer != nullptr) {
    cudaFree(*buffer);
    *buffer = nullptr;
  }
}

template <typename T>
void FreePinnedHostBuffer(T** buffer) {
  if (buffer != nullptr && *buffer != nullptr) {
    cudaFreeHost(*buffer);
    *buffer = nullptr;
  }
}

__device__ inline float Sigmoid(float value) {
  if (value >= 0.0f) {
    const float z = expf(-value);
    return 1.0f / (1.0f + z);
  }
  const float z = expf(value);
  return z / (1.0f + z);
}

__device__ inline float SafeExp(float value) {
  constexpr float kClampMin = -20.0f;
  constexpr float kClampMax = 20.0f;
  return expf(fmaxf(kClampMin, fminf(kClampMax, value)));
}

struct CandidateCompare {
  __host__ __device__ bool operator()(const DeviceCandidate& lhs,
                                      const DeviceCandidate& rhs) const {
    return lhs.score > rhs.score;
  }
};

__global__ void FinalizeDecodeCandidatesKernel(const float* bbox_pred,
                                               const float* cls_score,
                                               const float* dir_cls_pred,
                                               const int* task_segment_offsets,
                                               const int* task_num_classes,
                                               const int* task_label_offsets,
                                               int task_count,
                                               int height,
                                               int width,
                                               float score_threshold,
                                               const float* point_cloud_range,
                                               const float* post_center_range,
                                               const float* pillar_size,
                                               int out_size_factor,
                                               int decode_mode,
                                               bool norm_bbox,
                                               DeviceCandidate* candidates) {
  const int slot = blockDim.x * blockIdx.x + threadIdx.x;
  const int total_slots = task_segment_offsets[task_count];
  if (slot >= total_slots) {
    return;
  }

  int task_id = 0;
  while (task_id + 1 < task_count && slot >= task_segment_offsets[task_id + 1]) {
    ++task_id;
  }
  const int segment_offset = task_segment_offsets[task_id];
  const int task_classes = task_num_classes[task_id];
  const int task_slot = slot - segment_offset;
  const int class_in_task = task_slot / (height * width);
  const int spatial_index = task_slot % (height * width);
  if (class_in_task >= task_classes) {
    candidates[slot].score = -1.0f;
    return;
  }

  const int class_channel = task_label_offsets[task_id] + class_in_task;
  const float heat = cls_score[class_channel * height * width + spatial_index];
  if (!isfinite(heat)) {
    candidates[slot].score = -1.0f;
    return;
  }
  const float score = Sigmoid(heat);
  if (!isfinite(score) || score <= score_threshold) {
    candidates[slot].score = -1.0f;
    return;
  }

  const float* bbox = bbox_pred + task_id * kBBoxChannelsPerTask * height * width;
  const float* dir = dir_cls_pred + task_id * kDirChannelsPerTask * height * width;
  const int y_idx = spatial_index / width;
  const int x_idx = spatial_index % width;

  const float reg_x = bbox[0 * height * width + spatial_index];
  const float reg_y = bbox[1 * height * width + spatial_index];
  const float height_value = bbox[2 * height * width + spatial_index];
  const float dim_x =
      (decode_mode == static_cast<int>(inference::DecodeMode::kCenterPoint) && !norm_bbox)
          ? bbox[3 * height * width + spatial_index]
          : SafeExp(bbox[3 * height * width + spatial_index]);
  const float dim_y =
      (decode_mode == static_cast<int>(inference::DecodeMode::kCenterPoint) && !norm_bbox)
          ? bbox[4 * height * width + spatial_index]
          : SafeExp(bbox[4 * height * width + spatial_index]);
  const float dim_z =
      (decode_mode == static_cast<int>(inference::DecodeMode::kCenterPoint) && !norm_bbox)
          ? bbox[5 * height * width + spatial_index]
          : SafeExp(bbox[5 * height * width + spatial_index]);
  const float rot_sine = dir[0 * height * width + spatial_index];
  const float rot_cosine = dir[1 * height * width + spatial_index];
  const float yaw = atan2f(rot_sine, rot_cosine);
  if (!isfinite(reg_x) || !isfinite(reg_y) || !isfinite(height_value) || !isfinite(dim_x) ||
      !isfinite(dim_y) || !isfinite(dim_z) || !isfinite(yaw)) {
    candidates[slot].score = -1.0f;
    return;
  }

  const float center_x =
      (static_cast<float>(x_idx) + reg_x) * out_size_factor * pillar_size[0] +
      point_cloud_range[0];
  const float center_y =
      (static_cast<float>(y_idx) + reg_y) * out_size_factor * pillar_size[1] +
      point_cloud_range[1];
  const float center_z =
      decode_mode == static_cast<int>(inference::DecodeMode::kCenterPoint)
          ? height_value
          : height_value - dim_z * 0.5f;
  if (!isfinite(center_x) || !isfinite(center_y) || !isfinite(center_z) ||
      center_x < post_center_range[0] || center_x > post_center_range[3] ||
      center_y < post_center_range[1] || center_y > post_center_range[4] ||
      center_z < post_center_range[2] || center_z > post_center_range[5]) {
    candidates[slot].score = -1.0f;
    return;
  }

  DeviceCandidate candidate;
  candidate.score = score;
  candidate.label = class_channel;
  candidate.box[0] = center_x;
  candidate.box[1] = center_y;
  candidate.box[2] = center_z;
  candidate.box[3] = dim_x;
  candidate.box[4] = dim_y;
  candidate.box[5] = dim_z;
  candidate.box[6] = yaw;
  candidates[slot] = candidate;
}

__global__ void PackTopKCandidatesKernel(const DeviceCandidate* sorted_candidates,
                                         const int* task_segment_offsets,
                                         int task_count,
                                         int topk,
                                         DeviceCandidate* output_candidates,
                                         int* output_counts) {
  const int task_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (task_id >= task_count) {
    return;
  }
  const int start = task_segment_offsets[task_id];
  const int end = task_segment_offsets[task_id + 1];
  int count = 0;
  for (int i = start; i < end && count < topk; ++i) {
    const DeviceCandidate candidate = sorted_candidates[i];
    if (candidate.score <= 0.0f) {
      break;
    }
    output_candidates[task_id * topk + count] = candidate;
    ++count;
  }
  output_counts[task_id] = count;
}

}  // namespace

struct GpuDecodedCandidates::Impl {
  RuntimeConfig config;
  int task_count = 0;
  int height = 0;
  int width = 0;
  int total_slots = 0;
  int max_candidates = 0;

  std::vector<int> host_task_segment_offsets;
  std::vector<int> host_task_num_classes;
  std::vector<int> host_task_label_offsets;

  int* d_task_segment_offsets = nullptr;
  int* d_task_num_classes = nullptr;
  int* d_task_label_offsets = nullptr;
  float* d_point_cloud_range = nullptr;
  float* d_post_center_range = nullptr;
  float* d_pillar_size = nullptr;
  DeviceCandidate* d_workspace = nullptr;
  DeviceCandidate* d_output = nullptr;
  int* d_output_counts = nullptr;

  DeviceCandidate* h_output = nullptr;
  int* h_output_counts = nullptr;

  ~Impl() {
    FreePinnedHostBuffer(&h_output_counts);
    FreePinnedHostBuffer(&h_output);
    FreeDeviceBuffer(&d_output_counts);
    FreeDeviceBuffer(&d_output);
    FreeDeviceBuffer(&d_workspace);
    FreeDeviceBuffer(&d_pillar_size);
    FreeDeviceBuffer(&d_post_center_range);
    FreeDeviceBuffer(&d_point_cloud_range);
    FreeDeviceBuffer(&d_task_label_offsets);
    FreeDeviceBuffer(&d_task_num_classes);
    FreeDeviceBuffer(&d_task_segment_offsets);
  }
};

GpuDecodedCandidates::GpuDecodedCandidates() = default;
GpuDecodedCandidates::~GpuDecodedCandidates() = default;

InferenceStatus GpuDecodedCandidates::Initialize(const RuntimeConfig& config,
                                                 int task_count,
                                                 int bbox_channels,
                                                 int cls_channels,
                                                 int dir_channels,
                                                 int height,
                                                 int width,
                                                 std::string* out_error) {
  if (task_count <= 0 || bbox_channels <= 0 || cls_channels <= 0 || dir_channels <= 0 ||
      height <= 0 || width <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "GPU decode received invalid engine shape metadata");
  }
  if (config.decode_mode == inference::DecodeMode::kCenterPoint &&
      config.task_num_classes.size() != static_cast<size_t>(task_count)) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "GPU decode CenterPoint task layout does not match runtime config");
  }
  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->task_count = task_count;
  impl->height = height;
  impl->width = width;
  impl->max_candidates = task_count * config.bbox_max_num;

  impl->host_task_segment_offsets.reserve(static_cast<size_t>(task_count + 1));
  impl->host_task_num_classes.reserve(static_cast<size_t>(task_count));
  impl->host_task_label_offsets.reserve(static_cast<size_t>(task_count));
  impl->host_task_segment_offsets.push_back(0);

  int class_offset = 0;
  for (int task_id = 0; task_id < task_count; ++task_id) {
    const int task_classes =
        config.decode_mode == inference::DecodeMode::kCenterPoint
            ? config.task_num_classes[static_cast<size_t>(task_id)]
            : 1;
    impl->host_task_num_classes.push_back(task_classes);
    impl->host_task_label_offsets.push_back(class_offset);
    const int next_offset =
        impl->host_task_segment_offsets.back() + task_classes * height * width;
    impl->host_task_segment_offsets.push_back(next_offset);
    class_offset += task_classes;
  }
  impl->total_slots = impl->host_task_segment_offsets.back();
  if (class_offset != cls_channels) {
    std::ostringstream oss;
    oss << "GPU decode class layout mismatch: runtime classes=" << class_offset
        << " engine cls_channels=" << cls_channels;
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
  }
  if (impl->total_slots <= 0) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "GPU decode produced no candidate slots");
  }

  InferenceStatus status = AllocateDeviceBuffer(
      impl->host_task_segment_offsets.size(), "task_segment_offsets",
      &impl->d_task_segment_offsets, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = AllocateDeviceBuffer(impl->host_task_num_classes.size(), "task_num_classes",
                                &impl->d_task_num_classes, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = AllocateDeviceBuffer(impl->host_task_label_offsets.size(), "task_label_offsets",
                                &impl->d_task_label_offsets, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = AllocateDeviceBuffer(config.point_cloud_range.size(), "point_cloud_range",
                                &impl->d_point_cloud_range, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = AllocateDeviceBuffer(config.post_center_range.size(), "post_center_range",
                                &impl->d_post_center_range, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = AllocateDeviceBuffer(config.pillar_size.size(), "pillar_size", &impl->d_pillar_size,
                                out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = AllocateDeviceBuffer(static_cast<size_t>(impl->total_slots), "decode_workspace",
                                &impl->d_workspace, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = AllocateDeviceBuffer(static_cast<size_t>(impl->max_candidates), "decode_output",
                                &impl->d_output, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = AllocateDeviceBuffer(static_cast<size_t>(task_count), "decode_output_counts",
                                &impl->d_output_counts, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = AllocatePinnedHostBuffer(static_cast<size_t>(impl->max_candidates), "host_output",
                                    &impl->h_output, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = AllocatePinnedHostBuffer(static_cast<size_t>(task_count), "host_output_counts",
                                    &impl->h_output_counts, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  status = CheckCuda(cudaMemcpy(impl->d_task_segment_offsets,
                                impl->host_task_segment_offsets.data(),
                                impl->host_task_segment_offsets.size() * sizeof(int),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy(task_segment_offsets)", out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = CheckCuda(cudaMemcpy(impl->d_task_num_classes, impl->host_task_num_classes.data(),
                                impl->host_task_num_classes.size() * sizeof(int),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy(task_num_classes)", out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = CheckCuda(cudaMemcpy(impl->d_task_label_offsets,
                                impl->host_task_label_offsets.data(),
                                impl->host_task_label_offsets.size() * sizeof(int),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy(task_label_offsets)", out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = CheckCuda(cudaMemcpy(impl->d_point_cloud_range, config.point_cloud_range.data(),
                                config.point_cloud_range.size() * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy(point_cloud_range)", out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = CheckCuda(cudaMemcpy(impl->d_post_center_range, config.post_center_range.data(),
                                config.post_center_range.size() * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy(post_center_range)", out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = CheckCuda(cudaMemcpy(impl->d_pillar_size, config.pillar_size.data(),
                                config.pillar_size.size() * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy(pillar_size)", out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  impl_ = std::move(impl);
  return INFERENCE_STATUS_OK;
}

InferenceStatus GpuDecodedCandidates::Decode(const float* d_bbox_pred,
                                             const float* d_cls_score,
                                             const float* d_dir_cls_pred,
                                             cudaStream_t stream,
                                             std::string* out_error) {
  if (!impl_) {
    return SetError(out_error, INFERENCE_STATUS_NOT_READY,
                    "GPU decode workspace is not initialized");
  }
  if (d_bbox_pred == nullptr || d_cls_score == nullptr || d_dir_cls_pred == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "GPU decode input tensor is null");
  }

  const int threads = 256;
  const int blocks = (impl_->total_slots + threads - 1) / threads;
  FinalizeDecodeCandidatesKernel<<<blocks, threads, 0, stream>>>(
      d_bbox_pred, d_cls_score, d_dir_cls_pred, impl_->d_task_segment_offsets,
      impl_->d_task_num_classes, impl_->d_task_label_offsets, impl_->task_count, impl_->height,
      impl_->width, impl_->config.score_threshold, impl_->d_point_cloud_range,
      impl_->d_post_center_range, impl_->d_pillar_size, impl_->config.out_size_factor,
      static_cast<int>(impl_->config.decode_mode), impl_->config.norm_bbox, impl_->d_workspace);
  InferenceStatus status =
      CheckCuda(cudaPeekAtLastError(), "FinalizeDecodeCandidatesKernel", out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  thrust::device_ptr<DeviceCandidate> workspace_ptr(impl_->d_workspace);
  for (int task_id = 0; task_id < impl_->task_count; ++task_id) {
    const int offset = impl_->host_task_segment_offsets[static_cast<size_t>(task_id)];
    const int next_offset = impl_->host_task_segment_offsets[static_cast<size_t>(task_id + 1)];
    thrust::sort(thrust::cuda::par.on(stream), workspace_ptr + offset, workspace_ptr + next_offset,
                 CandidateCompare{});
  }

  PackTopKCandidatesKernel<<<1, impl_->task_count, 0, stream>>>(
      impl_->d_workspace, impl_->d_task_segment_offsets, impl_->task_count,
      impl_->config.bbox_max_num, impl_->d_output, impl_->d_output_counts);
  return CheckCuda(cudaPeekAtLastError(), "PackTopKCandidatesKernel", out_error);
}

InferenceStatus GpuDecodedCandidates::CopyToHost(CandidateDetections* out_candidates,
                                                 cudaStream_t stream,
                                                 std::string* out_error) const {
  if (!impl_) {
    return SetError(out_error, INFERENCE_STATUS_NOT_READY,
                    "GPU decode workspace is not initialized");
  }
  if (out_candidates == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "out_candidates is null");
  }

  InferenceStatus status = CheckCuda(
      cudaMemcpyAsync(impl_->h_output_counts, impl_->d_output_counts,
                      static_cast<size_t>(impl_->task_count) * sizeof(int),
                      cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync(output_counts)", out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = CheckCuda(cudaMemcpyAsync(
                         impl_->h_output, impl_->d_output,
                         static_cast<size_t>(impl_->max_candidates) * sizeof(DeviceCandidate),
                         cudaMemcpyDeviceToHost, stream),
                     "cudaMemcpyAsync(output_candidates)", out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(output_candidates)",
                     out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  out_candidates->scores.clear();
  out_candidates->boxes.clear();
  out_candidates->labels.clear();
  out_candidates->scores.reserve(static_cast<size_t>(impl_->max_candidates));
  out_candidates->boxes.reserve(static_cast<size_t>(impl_->max_candidates) * kBoxDim);
  out_candidates->labels.reserve(static_cast<size_t>(impl_->max_candidates));

  for (int task_id = 0; task_id < impl_->task_count; ++task_id) {
    const int task_count = impl_->h_output_counts[task_id];
    const DeviceCandidate* task_candidates =
        impl_->h_output + task_id * impl_->config.bbox_max_num;
    for (int i = 0; i < task_count; ++i) {
      const DeviceCandidate& candidate = task_candidates[i];
      out_candidates->scores.push_back(candidate.score);
      out_candidates->labels.push_back(candidate.label);
      out_candidates->boxes.insert(out_candidates->boxes.end(), candidate.box,
                                   candidate.box + kBoxDim);
    }
  }
  return INFERENCE_STATUS_OK;
}

}  // namespace inference_internal
