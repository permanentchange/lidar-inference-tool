#pragma once

#include "pillar_runtime.hpp"

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

namespace inference_internal {

class GpuDecodedCandidates {
 public:
  GpuDecodedCandidates();
  ~GpuDecodedCandidates();

  GpuDecodedCandidates(const GpuDecodedCandidates&) = delete;
  GpuDecodedCandidates& operator=(const GpuDecodedCandidates&) = delete;

  InferenceStatus Initialize(const RuntimeConfig& config,
                             const ResolvedHeadLayout& layout,
                             int task_count,
                             int bbox_channels,
                             int cls_channels,
                             int dir_channels,
                             int height,
                             int width,
                             std::string* out_error);
  InferenceStatus Decode(const float* d_bbox_pred,
                         const float* d_cls_score,
                         const float* d_dir_cls_pred,
                         cudaStream_t stream,
                         std::string* out_error);
  InferenceStatus CopyToHost(CandidateDetections* out_candidates,
                             cudaStream_t stream,
                             std::string* out_error,
                             std::vector<float>* velocities = nullptr) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace inference_internal
