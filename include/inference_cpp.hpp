#pragma once

#include "inference.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace inference_internal {
class PillarRuntime;
}  // namespace inference_internal

namespace inference {

constexpr size_t kBoxDimension = 7;

enum class DecodeMode {
  kLegacy = 0,
  kCenterPoint = 1,
};

struct ModelConfig {
  std::string engine_path;
  std::array<float, 6> point_cloud_range{{-25.6f, -25.6f, -5.0f, 25.6f, 25.6f, 3.0f}};
  std::array<float, 6> post_center_range{{-35.6f, -35.6f, -10.0f, 35.6f, 35.6f, 10.0f}};
  std::array<float, 3> pillar_size{{0.1f, 0.1f, 8.0f}};
  int num_point_features = 5;
  int out_size_factor = 4;
  DecodeMode decode_mode = DecodeMode::kLegacy;
  int bbox_code_size = 7;
  bool norm_bbox = false;
  bool has_velocity = false;
  std::vector<std::string> label_names{"car", "pedestrian", "cyclist"};
  std::vector<int> task_num_classes{1, 1, 1};
  std::vector<std::vector<int>> task_label_ids;
};

struct RuntimeConfig {
  int max_points_per_pillar = 20;
  int min_pillars = 100;
  int max_pillars = 50000;
  int max_points = 300000;
  int frame_num = 1;
  int bbox_max_num = 500;
  float score_threshold = 0.1f;
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
  std::vector<float> label_score_thresh{0.45f, 0.3f, 0.45f};
  std::vector<float> circle_nms_min_radius{4.0f, 12.0f, 10.0f};
  std::vector<NmsLabelGroup> nms_label_groups;
  std::vector<int> label_mapping;
  std::vector<std::string> mapped_label_names;
};

struct PackageConfig {
  ModelConfig model;
  RuntimeConfig runtime;
  PostprocessConfig postprocess;
};

struct PointCloudView {
  const float* points = nullptr;
  int point_count = 0;
  const float* motion_4x4 = nullptr;
  int enable_runtime = 1;
};

struct Detection {
  float score = 0.0f;
  // Box layout is [x, y, z, dx, dy, dz, yaw], where x/y/z denote the box geometric center.
  std::array<float, kBoxDimension> box{};
  int label = 0;
  // Valid while the owning runtime remains alive.
  const char* label_name = nullptr;
};

class LidarInferenceRuntime {
 public:
  ~LidarInferenceRuntime();

  LidarInferenceRuntime(const LidarInferenceRuntime&) = delete;
  LidarInferenceRuntime& operator=(const LidarInferenceRuntime&) = delete;

  static InferenceStatus LoadPackageConfigFile(const std::string& config_path,
                                               PackageConfig* out_config,
                                               std::string* out_error,
                                               const std::string& override_path = "");
  static InferenceStatus LoadPackageConfig(const std::string& model_dir,
                                           PackageConfig* out_config,
                                           std::string* out_error,
                                           const std::string& override_path = "");
  static InferenceStatus Create(const PackageConfig& config,
                                std::unique_ptr<LidarInferenceRuntime>* out_runtime,
                                std::string* out_error);
  static InferenceStatus CreateFromModelDir(const std::string& model_dir,
                                            std::unique_ptr<LidarInferenceRuntime>* out_runtime,
                                            std::string* out_error,
                                            const std::string& override_path = "");

  InferenceStatus Infer(const PointCloudView& cloud, std::vector<Detection>* out_detections);
  InferenceStatus InferDecoded(const PointCloudView& cloud,
                               std::vector<Detection>* out_detections);
  InferenceStatus InferWithPostprocess(const PointCloudView& cloud,
                                       const PostprocessConfig& postprocess,
                                       std::vector<Detection>* out_detections);
  InferenceStatus Reset();

  const PackageConfig& config() const { return config_; }
  const InferenceOutputSpec& output_spec() const { return output_spec_; }
  const std::string& last_error() const { return last_error_; }

 private:
  LidarInferenceRuntime() = default;

  InferenceStatus InferInternal(const PointCloudView& cloud,
                                const PostprocessConfig* postprocess,
                                std::vector<Detection>* out_detections,
                                bool use_nms);

  PackageConfig config_{};
  std::shared_ptr<inference_internal::PillarRuntime> runtime_;
  InferenceOutputSpec output_spec_{};
  InferenceResponse response_{};
  std::vector<float> cls_preds_;
  std::vector<float> box_preds_;
  std::vector<int> label_preds_;
  std::string last_error_;
};

}  // namespace inference
