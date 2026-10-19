#include "inference_cpp.hpp"
#include "pillar_runtime.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::filesystem::path MakeTempDir() {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "lidar_inference_config_loader_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

void WriteTextFile(const std::filesystem::path& path, const std::string& content) {
  std::ofstream output(path);
  assert(output.is_open());
  output << content;
}

void WriteFloatBinFile(const std::filesystem::path& path, const std::vector<float>& values) {
  std::ofstream output(path, std::ios::binary);
  assert(output.is_open());
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}

void TestVoxelPackageConfigUsesVoxelSizeAndHeadEnginePath() {
  const std::filesystem::path dir = MakeTempDir() / "voxel_config";
  std::filesystem::create_directories(dir);
  const std::filesystem::path yaml_path = dir / "model.yaml";
  WriteTextFile(yaml_path, R"YAML(
model:
  runtime_backend: centerpoint_voxel
  scn_onnx_path: centerpoint.scn.onnx
  head_engine_path: rpn_centerhead.engine
  decode_mode: centerpoint_mmdeploy
  yaw_encoding: sin_cos
  num_point_features: 5
  voxel_size: [0.05, 0.05, 0.2]
  pillar_size: [9.0, 9.0, 9.0]
  sparse_shape: [41, 320, 480]
  point_cloud_range: [-8.0, -8.0, -2.0, 16.0, 8.0, 4.0]
  post_center_range: [-10.0, -10.0, -2.0, 18.0, 10.0, 4.0]
  out_size_factor: 8
  bbox_code_size: 9
  norm_bbox: true
  has_velocity: true
  label_names: [car]
  task_num_classes: [1]
  task_label_ids:
    - [0]
runtime:
  max_points_per_voxel: 10
  max_points_per_pillar: 99
  min_voxels: 1
  min_pillars: 99
  max_voxels: 160000
  max_pillars: 99
  max_points: 300000
  bbox_max_num: 500
  score_threshold: 0.1
)YAML");

  inference::PackageConfig config;
  std::string error;
  const InferenceStatus status =
      inference::LidarInferenceRuntime::LoadPackageConfigFile(yaml_path.string(), &config, &error);
  assert(status == INFERENCE_STATUS_OK && error.c_str());
  assert(config.model.engine_path == (dir / "rpn_centerhead.engine").lexically_normal().string());
  assert(config.model.voxel_size[0] == 0.05f);
  assert(config.model.voxel_size[1] == 0.05f);
  assert(config.model.voxel_size[2] == 0.2f);
  assert(config.runtime.max_points_per_voxel == 10);
}

void TestVoxelPackageConfigAcceptsLegacyPillarKeysAsFallback() {
  const std::filesystem::path dir = MakeTempDir() / "legacy_voxel_config";
  std::filesystem::create_directories(dir);
  const std::filesystem::path yaml_path = dir / "model.yaml";
  WriteTextFile(yaml_path, R"YAML(
model:
  runtime_backend: centerpoint_voxel
  scn_onnx_path: centerpoint.scn.onnx
  head_engine_path: rpn_centerhead.engine
  decode_mode: centerpoint_mmdeploy
  yaw_encoding: sin_cos
  num_point_features: 5
  pillar_size: [0.05, 0.05, 0.2]
  sparse_shape: [41, 320, 480]
  label_names: [car]
  task_num_classes: [1]
  task_label_ids:
    - [0]
  bbox_code_size: 9
  has_velocity: true
runtime:
  max_points_per_pillar: 10
  min_pillars: 1
  max_pillars: 160000
)YAML");

  inference::PackageConfig config;
  std::string error;
  const InferenceStatus status =
      inference::LidarInferenceRuntime::LoadPackageConfigFile(yaml_path.string(), &config, &error);
  assert(status == INFERENCE_STATUS_OK && error.c_str());
  assert(config.model.voxel_size[0] == 0.05f);
  assert(config.model.voxel_size[1] == 0.05f);
  assert(config.model.voxel_size[2] == 0.2f);
  assert(config.runtime.max_points_per_voxel == 10);
  assert(config.runtime.min_voxels == 1);
  assert(config.runtime.max_voxels == 160000);
}

void TestVoxelPackageConfigRequiresSparseShape() {
  const std::filesystem::path dir = MakeTempDir() / "missing_sparse_shape";
  std::filesystem::create_directories(dir);
  const std::filesystem::path yaml_path = dir / "model.yaml";
  WriteTextFile(yaml_path, R"YAML(
model:
  runtime_backend: centerpoint_voxel
  scn_onnx_path: centerpoint.scn.onnx
  head_engine_path: rpn_centerhead.engine
  label_names: [car]
  task_num_classes: [1]
  task_label_ids:
    - [0]
)YAML");

  inference::PackageConfig config;
  std::string error;
  const InferenceStatus status =
      inference::LidarInferenceRuntime::LoadPackageConfigFile(yaml_path.string(), &config, &error);
  assert(status == INFERENCE_STATUS_INVALID_ARGUMENT);
  assert(error.find("model.sparse_shape") != std::string::npos);
}

void TestDemoCenterPointVoxelPackageConfigParses() {
  const std::filesystem::path repo_root =
      std::filesystem::path(__FILE__).parent_path().parent_path();
  const std::filesystem::path yaml_path =
      repo_root / "demo" / "centerpoint_voxel" / "model.yaml";

  inference::PackageConfig config;
  std::string error;
  const InferenceStatus status =
      inference::LidarInferenceRuntime::LoadPackageConfigFile(yaml_path.string(), &config, &error);
  assert(status == INFERENCE_STATUS_OK && error.c_str());
  assert(config.model.num_point_features == 5);
  assert(config.model.engine_path ==
         (yaml_path.parent_path() / "rpn_centerhead.engine").lexically_normal().string());
  assert(config.model.voxel_size[0] == 0.05f);
  assert(config.model.voxel_size[1] == 0.05f);
  assert(config.model.voxel_size[2] == 0.2f);
  assert(config.runtime.max_points_per_voxel == 20);
}

void TestToolConfigCarriesCenterPointVoxelRuntimeExtras() {
  const std::filesystem::path repo_root =
      std::filesystem::path(__FILE__).parent_path().parent_path();
  const std::filesystem::path model_dir =
      repo_root / "demo" / "centerpoint_voxel";
  const std::filesystem::path points_path = model_dir / "sample_points.bin";

  inference_internal::ToolConfig config;
  std::string error;
  const InferenceStatus status = inference_internal::LoadToolConfigFromModelDir(
      model_dir.string(), points_path.string(), "/tmp/lidar_tool_config_test",
      "", 1, 2, 0.2f, &config, &error);
  assert(status == INFERENCE_STATUS_OK && error.empty());
  assert(config.runtime.runtime_backend ==
         inference_internal::RuntimeBackend::kCenterPointVoxel);
  assert(config.runtime.scn_onnx_file ==
         (model_dir / "centerpoint.scn.onnx").lexically_normal().string());
  assert(config.runtime.head_engine_file ==
         (model_dir / "rpn_centerhead.engine").lexically_normal().string());
  assert(config.runtime.sparse_shape.size() == 3);
  assert(config.runtime.sparse_shape[0] == 41);
  assert(config.runtime.sparse_shape[1] == 320);
  assert(config.runtime.sparse_shape[2] == 480);
}

void TestRejectsInvalidRuntimeLimits() {
  const std::filesystem::path dir = MakeTempDir() / "invalid_runtime_limits";
  std::filesystem::create_directories(dir);
  const std::filesystem::path yaml_path = dir / "model.yaml";
  WriteTextFile(yaml_path, R"YAML(
model:
  engine_path: dummy.engine
runtime:
  max_points: -1
)YAML");

  inference::PackageConfig config;
  std::string error;
  const InferenceStatus status =
      inference::LidarInferenceRuntime::LoadPackageConfigFile(yaml_path.string(), &config, &error);
  assert(status == INFERENCE_STATUS_INVALID_ARGUMENT);
  assert(error.find("runtime.max_points") != std::string::npos);
}

void TestRejectsInvalidVoxelSize() {
  const std::filesystem::path dir = MakeTempDir() / "invalid_voxel_size";
  std::filesystem::create_directories(dir);
  const std::filesystem::path yaml_path = dir / "model.yaml";
  WriteTextFile(yaml_path, R"YAML(
model:
  engine_path: dummy.engine
  voxel_size: [0.1, 0.0, 8.0]
)YAML");

  inference::PackageConfig config;
  std::string error;
  const InferenceStatus status =
      inference::LidarInferenceRuntime::LoadPackageConfigFile(yaml_path.string(), &config, &error);
  assert(status == INFERENCE_STATUS_INVALID_ARGUMENT);
  assert(error.find("model.voxel_size[1]") != std::string::npos);
}

void TestRejectsShortLabelScoreThresholds() {
  const std::filesystem::path dir = MakeTempDir() / "short_label_thresholds";
  std::filesystem::create_directories(dir);
  const std::filesystem::path yaml_path = dir / "model.yaml";
  WriteTextFile(yaml_path, R"YAML(
model:
  engine_path: dummy.engine
  label_names: [car, pedestrian, cyclist]
postprocess:
  nms_type: iou_bev
  label_score_thresh: [0.1]
)YAML");

  inference::PackageConfig config;
  std::string error;
  const InferenceStatus status =
      inference::LidarInferenceRuntime::LoadPackageConfigFile(yaml_path.string(), &config, &error);
  assert(status == INFERENCE_STATUS_INVALID_ARGUMENT);
  assert(error.find("postprocess.label_score_thresh") != std::string::npos);
}

void TestRejectsShortCircleNmsRadii() {
  const std::filesystem::path dir = MakeTempDir() / "short_circle_radii";
  std::filesystem::create_directories(dir);
  const std::filesystem::path yaml_path = dir / "model.yaml";
  WriteTextFile(yaml_path, R"YAML(
model:
  engine_path: dummy.engine
postprocess:
  nms_type: circle
  circle_nms_min_radius: [1.0]
)YAML");

  inference::PackageConfig config;
  std::string error;
  const InferenceStatus status =
      inference::LidarInferenceRuntime::LoadPackageConfigFile(yaml_path.string(), &config, &error);
  assert(status == INFERENCE_STATUS_INVALID_ARGUMENT);
  assert(error.find("postprocess.circle_nms_min_radius") != std::string::npos);
}

void TestFiveFeatureBinPreservesHasIntensityChannel() {
  const std::filesystem::path dir = MakeTempDir() / "bin";
  std::filesystem::create_directories(dir);
  const std::filesystem::path bin_path = dir / "points.bin";
  WriteFloatBinFile(bin_path, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                              6.0f, 7.0f, 8.0f, 9.0f, 10.0f});

  std::vector<float> points;
  std::string error;
  inference_internal::PointCloudLoadInfo info;
  const InferenceStatus status =
      inference_internal::LoadPcdFile(bin_path.string(), 5, 10, &points, &error, &info);
  assert(status == INFERENCE_STATUS_OK);
  assert(points.size() == 10);
  assert(points[3] == 4.0f);
  assert(points[4] == 5.0f);
  assert(points[8] == 9.0f);
  assert(points[9] == 10.0f);
}

void TestAsciiPcdPreservesUintIntensityAndHasIntensity() {
  const std::filesystem::path dir = MakeTempDir() / "pcd";
  std::filesystem::create_directories(dir);
  const std::filesystem::path pcd_path = dir / "points.pcd";
  WriteTextFile(pcd_path, R"PCD(# .PCD v0.7
VERSION 0.7
FIELDS x y z intensity has_intensity
SIZE 4 4 4 1 1
TYPE F F F U U
COUNT 1 1 1 1 1
WIDTH 2
HEIGHT 1
POINTS 2
DATA ascii
1.0 2.0 3.0 10 1
4.0 5.0 6.0 20 1
)PCD");

  std::vector<float> points;
  std::string error;
  inference_internal::PointCloudLoadInfo info;
  const InferenceStatus status =
      inference_internal::LoadPcdFile(pcd_path.string(), 5, 10, &points, &error, &info);
  assert(status == INFERENCE_STATUS_OK);
  assert(points.size() == 10);
  assert(points[3] == 10.0f);
  assert(points[4] == 1.0f);
  assert(points[8] == 20.0f);
  assert(points[9] == 1.0f);
}

}  // namespace

int main() {
  TestVoxelPackageConfigUsesVoxelSizeAndHeadEnginePath();
  TestVoxelPackageConfigAcceptsLegacyPillarKeysAsFallback();
  TestVoxelPackageConfigRequiresSparseShape();
  TestDemoCenterPointVoxelPackageConfigParses();
  TestToolConfigCarriesCenterPointVoxelRuntimeExtras();
  TestRejectsInvalidRuntimeLimits();
  TestRejectsInvalidVoxelSize();
  TestRejectsShortLabelScoreThresholds();
  TestRejectsShortCircleNmsRadii();
  TestFiveFeatureBinPreservesHasIntensityChannel();
  TestAsciiPcdPreservesUintIntensityAndHasIntensity();
  return 0;
}
