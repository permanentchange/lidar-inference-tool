#pragma once

#include "pillar_runtime.hpp"

#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

namespace inference_internal {

struct EngineContract {
  int task_count = 0;
  int bbox_channels = 0;
  int cls_channels = 0;
  int dir_channels = 0;
  int height = 0;
  int width = 0;
  int num_bindings = 0;
  int min_profile_pillars = 0;
  int opt_profile_pillars = 0;
  int max_profile_pillars = 0;
  int min_runtime_pillars = 0;
  int max_runtime_pillars = 0;
  int voxels_index = -1;
  int num_points_index = -1;
  int coors_index = -1;
  int bbox_pred_index = -1;
  int cls_score_index = -1;
  int dir_cls_pred_index = -1;
};

class TrtExecutor {
 public:
  TrtExecutor();
  ~TrtExecutor();

  TrtExecutor(const TrtExecutor&) = delete;
  TrtExecutor& operator=(const TrtExecutor&) = delete;

  InferenceStatus Load(const RuntimeConfig& config,
                       const ResolvedHeadLayout& layout,
                       const std::string& engine_path,
                       std::string* out_error);
  InferenceStatus ConfigureShapes(const RuntimeConfig& config,
                                  int num_pillars,
                                  std::string* out_error);
  InferenceStatus Enqueue(void** bindings, cudaStream_t stream, std::string* out_error);

  const EngineContract& contract() const { return contract_; }

 private:
  struct Impl;

  EngineContract contract_{};
  std::unique_ptr<Impl> impl_;
};

struct CenterPointHeadContract {
  int task_count = 0;
  int bbox_channels = 0;
  int cls_channels = 0;
  int dir_channels = 0;
  int height = 0;
  int width = 0;
  int num_bindings = 0;
  int input_index = -1;
  std::vector<int> reg_indices;
  std::vector<int> height_indices;
  std::vector<int> dim_indices;
  std::vector<int> rot_indices;
  std::vector<int> vel_indices;
  std::vector<int> hm_indices;
};

class CenterPointHeadTrtExecutor {
 public:
  CenterPointHeadTrtExecutor();
  ~CenterPointHeadTrtExecutor();

  CenterPointHeadTrtExecutor(const CenterPointHeadTrtExecutor&) = delete;
  CenterPointHeadTrtExecutor& operator=(const CenterPointHeadTrtExecutor&) = delete;

  InferenceStatus Load(const RuntimeConfig& config,
                       const ResolvedHeadLayout& layout,
                       const std::string& engine_path,
                       std::string* out_error);
  InferenceStatus Enqueue(void* scn_features,
                          const std::vector<int64_t>& scn_features_shape,
                          float* d_bbox_pred,
                          float* d_cls_score,
                          float* d_dir_cls_pred,
                          cudaStream_t stream,
                          std::string* out_error);

  const CenterPointHeadContract& contract() const { return contract_; }

 private:
  struct Impl;

  CenterPointHeadContract contract_{};
  std::unique_ptr<Impl> impl_;
};

}  // namespace inference_internal
