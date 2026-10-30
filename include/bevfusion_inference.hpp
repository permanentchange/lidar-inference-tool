#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "inference_cpp.hpp"

namespace inference_internal { struct RuntimeDiagnostics; }
namespace inference {

// Separate multimodal API: existing lidar-only ABI and C API are unchanged.
struct BevFusionTensor {
  std::string name;
  std::vector<size_t> shape;
  std::vector<float> values;
};
struct BevFusionCamera {
  std::string name, image_path;
  bool enabled = true;
  // Native source ROI applied before resize; zeros retain resize-then-crop.
  std::array<int, 4> source_crop{};
  std::array<float, 16> cam2img{}, cam2lidar{}, lidar2img{}, img_aug_matrix{};
  int width = 0, height = 0;
  int resize_width = 0, resize_height = 0, crop_left = 0, crop_top = 0;
  std::vector<uint8_t> rgb;
};
struct BevFusionFrame {
  std::string manifest_path;
  std::vector<float> points, images;
  int num_point_features = 0;
  size_t source_points = 0;
  std::array<float, 16> lidar_aug_matrix{};
  std::vector<BevFusionCamera> cameras;
  bool raw_images = true;
};
struct BevFusionDetection {
  // Geometric center x,y,z; dimensions dx,dy,dz; yaw, in lidar coordinates.
  std::array<float, 7> box{};
  std::array<float, 2> velocity{};
  float score = 0;
  int label = 0;
  std::string label_name;
};
struct BevFusionTiming {
  std::string name;
  double host_ms = 0, gpu_ms = 0;
};
struct BevFusionResult {
  std::vector<BevFusionTiming> timings;
  size_t device_memory_used_bytes = 0;
  std::vector<BevFusionDetection> detections;
  std::vector<BevFusionDetection> decoded_detections;
  std::vector<BevFusionTensor> tensors;
  size_t input_points = 0, filtered_points = 0, used_points = 0, voxels = 0,
         discarded_voxels = 0;
  size_t depth_collisions = 0;
  bool geometry_cache_hit = false;
};
struct BevFusionConfig {
  std::string architecture = "centerpoint_bevfusion", bev_pool_backend = "reference";
  std::vector<std::vector<int>> task_label_ids;
  int bbox_max_num = 500;
  bool norm_bbox = true;
  PostprocessConfig centerhead_postprocess;
  std::string model_dir, sparse_model, camera_engine, head_engine;
  int num_point_features = 0, max_points = 0, max_voxels = 0,
      max_points_per_voxel = 0;
  int min_voxels = 100, frame_num = 1;
  int out_size_factor = 0, feature_channels = 0, depth_bins = 0;
  std::array<float, 6> point_cloud_range{}, post_center_range{};
  std::array<float, 3> voxel_size{}, mean{}, std{}, xbound{}, ybound{},
      zbound{}, dbound{};
  std::array<int, 3> sparse_shape{};
  std::array<int, 2> image_size{}, feature_size{};
  std::vector<std::string> camera_names, label_names;
  float score_threshold = 0;
};

// These functions throw std::runtime_error with actionable context on failure.
bool IsBevFusionModel(const std::string& model_dir);
BevFusionConfig LoadBevFusionConfig(const std::string& model_dir,
                                    const std::string& override_config = "");
BevFusionFrame LoadBevFusionFrame(const std::string& manifest,
                                  const BevFusionConfig& config,
                                  bool raw_images = true,
                                  int source_features = 0);
BevFusionTensor LoadBevFusionNpy(const std::string& path);
void SaveBevFusionNpy(const std::string& path, const BevFusionTensor& tensor);

class BevFusionInferenceRuntime {
 public:
  explicit BevFusionInferenceRuntime(const std::string& model_dir,
                                     const std::string& override_config = "");
  ~BevFusionInferenceRuntime();
  BevFusionInferenceRuntime(const BevFusionInferenceRuntime&) = delete;
  BevFusionInferenceRuntime& operator=(const BevFusionInferenceRuntime&) =
      delete;
  const BevFusionConfig& config() const;
  // One runtime owns one stream; callers must serialize use of an instance.
  // Debug copies occur only when dump_internal is true.
  BevFusionResult Infer(const BevFusionFrame& frame,
                        bool dump_internal = false);

 private:
  friend struct inference_internal::RuntimeDiagnostics;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace inference
