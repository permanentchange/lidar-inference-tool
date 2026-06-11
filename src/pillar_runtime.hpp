#pragma once

#include "inference_cpp.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace inference_internal {

struct RuntimeConfig {
  std::string end2end_file;
  std::array<float, 6> point_cloud_range{};
  std::array<float, 6> post_center_range{};
  std::array<float, 3> pillar_size{};
  int max_points_per_pillar = 20;
  int min_pillars = 100;
  int max_pillars = 50000;
  int max_points = 300000;
  int frame_num = 1;
  int num_point_features = 5;
  int bbox_max_num = 500;
  int out_size_factor = 4;
  float score_threshold = 0.1f;
  inference::DecodeMode decode_mode = inference::DecodeMode::kLegacy;
  inference::YawEncoding yaw_encoding = inference::YawEncoding::kSinCos;
  int bbox_code_size = 7;
  bool norm_bbox = false;
  bool has_velocity = false;
  std::vector<int> task_num_classes;
  std::vector<std::vector<int>> task_label_ids;
  int label_count = 0;
};

struct ResolvedHeadLayout {
  inference::DecodeMode decode_mode = inference::DecodeMode::kLegacy;
  inference::YawEncoding yaw_encoding = inference::YawEncoding::kSinCos;
  bool has_velocity = false;
  bool norm_bbox = false;
  int task_count = 0;
  int label_count = 0;
  int cls_channels_total = 0;
  int bbox_channels_per_task = 0;
  int dir_channels_per_task = 2;
  int reg_offset = 0;
  int height_offset = 2;
  int dim_offset = 3;
  int theta_offset = -1;
  int vel_offset = -1;
  std::vector<int> task_num_classes;
  std::vector<int> task_class_offsets;
  std::vector<std::vector<int>> task_label_ids;
  std::vector<int> flat_global_label_ids;
};

struct PostprocessConfig {
  struct NmsLabelGroup {
    std::vector<int> labels;
    float nms_thresh = -1.0f;
    int nms_pre_maxsize = 0;
    int nms_post_maxsize = 0;
    float circle_nms_min_radius = -1.0f;
  };

  InferenceNmsType nms_type = INFERENCE_NMS_IOU_BEV;
  float nms_thresh = 0.25f;
  int nms_pre_maxsize = 2048;
  int nms_post_maxsize = 256;
  float final_score_threshold = 0.3f;
  std::vector<float> circle_nms_min_radius{4.0f, 12.0f, 10.0f};
  std::vector<float> label_score_thresh{0.45f, 0.3f, 0.45f};
  std::vector<NmsLabelGroup> nms_label_groups;
};

struct StageTimings {
  double pillarize_ms = 0.0;
  double h2d_prepare_ms = 0.0;
  double trt_forward_ms = 0.0;
  double d2h_collect_ms = 0.0;
  double decode_ms = 0.0;
  double output_pack_ms = 0.0;
  double total_forward_ms = 0.0;
};

struct PillarInputTensors {
  std::vector<float> voxels;
  std::vector<int32_t> num_points;
  std::vector<int32_t> coors;
  int num_pillars = 0;
};

struct RawHeadOutputs {
  std::vector<float> bbox_pred;
  std::vector<float> cls_score;
  std::vector<float> dir_cls_pred;
  int bbox_channels = 0;
  int cls_channels = 0;
  int dir_channels = 0;
  int height = 0;
  int width = 0;
  int task_count = 0;
};

struct RawHeadView {
  const float* bbox_pred = nullptr;
  const float* cls_score = nullptr;
  const float* dir_cls_pred = nullptr;
  int bbox_channels = 0;
  int cls_channels = 0;
  int dir_channels = 0;
  int height = 0;
  int width = 0;
  int task_count = 0;
};

struct CandidateDetections {
  std::vector<float> scores;
  std::vector<float> boxes;
  std::vector<int> labels;
};

struct DebugArtifacts {
  std::vector<float> input_points;
  PillarInputTensors pillar_inputs;
  RawHeadOutputs raw_outputs;
  CandidateDetections decoded_candidates;
  CandidateDetections final_nms_outputs;
  StageTimings timings;
};

struct ToolConfig {
  inference::PackageConfig package;
  RuntimeConfig runtime;
  std::string model_dir;
  std::string override_config_path;
  std::string pcd_path;
  std::string dump_dir;
  int warmup_iters = 10;
  int profile_iters = 50;
  float bev_resolution = 0.2f;
};

struct PointCloudLoadInfo {
  std::string path;
  std::string format;
  std::vector<std::string> fields;
  int source_points = 0;
  int loaded_points = 0;
  int num_point_features = 0;
  bool truncated = false;
};

InferenceStatus ParseRuntimeConfig(const InferenceConfig& config,
                                   RuntimeConfig* out_config,
                                   std::string* out_error);
InferenceStatus ResolveHeadLayout(const inference::ModelConfig& config,
                                  ResolvedHeadLayout* out_layout,
                                  std::string* out_error);
InferenceStatus ResolveHeadLayout(const RuntimeConfig& config,
                                  ResolvedHeadLayout* out_layout,
                                  std::string* out_error);
int GlobalLabelForTaskClass(const ResolvedHeadLayout& layout,
                            int task_id,
                            int class_index);
const std::vector<int>& GlobalLabelsForTask(const ResolvedHeadLayout& layout,
                                            size_t task_index);
void SetDefaultPostprocessConfig(PostprocessConfig* out_config);
InferenceStatus ParsePostprocessConfig(const InferencePostprocessConfig& config,
                                       PostprocessConfig* out_config,
                                       std::string* out_error);
InferenceOutputSpec MakeOutputSpec(int task_count, int bbox_max_num);

class PillarRuntime {
 public:
  explicit PillarRuntime(RuntimeConfig config);
  ~PillarRuntime();

  InferenceStatus Initialize();
  InferenceStatus Forward(const InferenceRequest& request,
                          InferenceResponse* response,
                          DebugArtifacts* debug,
                          StageTimings* timings);
  InferenceStatus ForwardNms(const InferenceRequest& request,
                             const PostprocessConfig& postprocess,
                             InferenceResponse* response,
                             DebugArtifacts* debug,
                             StageTimings* timings);
  InferenceOutputSpec output_spec() const { return output_spec_; }
  const std::string& last_error() const { return last_error_; }
  void Reset();

 private:
  struct Impl;
  InferenceStatus ForwardInternal(const InferenceRequest& request,
                                  const PostprocessConfig* postprocess,
                                  InferenceResponse* response,
                                  DebugArtifacts* debug,
                                  StageTimings* timings);

  RuntimeConfig config_;
  InferenceOutputSpec output_spec_{};
  std::string last_error_;
  std::unique_ptr<Impl> impl_;
};

InferenceStatus LoadToolConfigFromModelDir(const std::string& model_dir,
                                           const std::string& pcd_path,
                                           const std::string& dump_dir,
                                           const std::string& override_config_path,
                                           int warmup_iters,
                                           int profile_iters,
                                           float bev_resolution,
                                           ToolConfig* out_config,
                                           std::string* out_error);
InferenceStatus LoadPcdFile(const std::string& path,
                            int num_point_features,
                            int max_points,
                            std::vector<float>* out_points,
                            std::string* out_error,
                            PointCloudLoadInfo* out_info = nullptr);
InferenceStatus RunPostprocessNms(const CandidateDetections& candidates,
                                  const PostprocessConfig& config,
                                  const RuntimeConfig& runtime_config,
                                  CandidateDetections* out_result,
                                  std::string* out_error);
PostprocessConfig MakeToolPostprocessConfig(const ToolConfig& config);
InferenceStatus SaveFloatNpy(const std::string& path,
                             const float* data,
                             size_t count,
                             const std::vector<size_t>& shape,
                             std::string* out_error);
InferenceStatus SaveInt32Npy(const std::string& path,
                             const int32_t* data,
                             size_t count,
                             const std::vector<size_t>& shape,
                             std::string* out_error);
InferenceStatus SaveInternalAuxiliaryArtifacts(const DebugArtifacts& debug,
                                               const ToolConfig& config,
                                               std::string* out_error);

}  // namespace inference_internal
