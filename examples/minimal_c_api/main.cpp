#include "inference.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kNumPointFeatures = 5;
constexpr int kBoxDim = 7;
constexpr size_t kMaxPreviewDetections = 5;

InferenceConfig MakeDemoConfig(const char* engine_path) {
  InferenceConfig config{};
  config.end2end_file = engine_path;

  config.point_cloud_range[0] = -25.6f;
  config.point_cloud_range[1] = -25.6f;
  config.point_cloud_range[2] = -5.0f;
  config.point_cloud_range[3] = 25.6f;
  config.point_cloud_range[4] = 25.6f;
  config.point_cloud_range[5] = 3.0f;

  config.post_center_range[0] = -35.6f;
  config.post_center_range[1] = -35.6f;
  config.post_center_range[2] = -10.0f;
  config.post_center_range[3] = 35.6f;
  config.post_center_range[4] = 35.6f;
  config.post_center_range[5] = 10.0f;

  config.pillar_size[0] = 0.1f;
  config.pillar_size[1] = 0.1f;
  config.pillar_size[2] = 8.0f;

  config.max_points_per_pillar = 20;
  config.min_pillars = 100;
  config.max_pillars = 50000;
  config.max_points = 300000;
  config.frame_num = 1;
  config.num_point_features = kNumPointFeatures;
  config.bbox_max_num = 500;
  config.out_size_factor = 4;
  config.score_threshold = 0.1f;
  return config;
}

bool LoadPointCloudBin(const std::string& path,
                       int num_point_features,
                       std::vector<float>* out_points,
                       std::string* out_error) {
  if (out_points == nullptr) {
    if (out_error != nullptr) {
      *out_error = "out_points is null";
    }
    return false;
  }
  if (num_point_features <= 0) {
    if (out_error != nullptr) {
      *out_error = "num_point_features must be positive";
    }
    return false;
  }

  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open point cloud: " + path;
    }
    return false;
  }

  const std::streamsize file_size = input.tellg();
  if (file_size <= 0 ||
      (file_size % static_cast<std::streamsize>(sizeof(float))) != 0) {
    if (out_error != nullptr) {
      *out_error = "Point cloud file is not aligned to float32 values: " + path;
    }
    return false;
  }

  const size_t total_floats = static_cast<size_t>(file_size) / sizeof(float);
  if ((total_floats % static_cast<size_t>(num_point_features)) != 0U) {
    if (out_error != nullptr) {
      *out_error = "Point cloud file does not match the expected 5-feature layout: " + path;
    }
    return false;
  }

  out_points->assign(total_floats, 0.0f);
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(out_points->data()), file_size);
  if (!input) {
    if (out_error != nullptr) {
      *out_error = "Failed to read point cloud payload: " + path;
    }
    return false;
  }
  return true;
}

void PrintUsage(const char* program) {
  std::cerr << "Usage: " << program << " <engine_path> <pcd_bin_path>\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    PrintUsage(argc > 0 ? argv[0] : "minimal_c_api_example");
    return 1;
  }

  const std::string engine_path = argv[1];
  const std::string point_cloud_path = argv[2];

  std::vector<float> points;
  std::string error;
  if (!LoadPointCloudBin(point_cloud_path, kNumPointFeatures, &points, &error)) {
    std::cerr << error << "\n";
    return 1;
  }

  InferenceConfig config = MakeDemoConfig(engine_path.c_str());
  InferenceHandle* handle = nullptr;
  const InferenceStatus create_status = inference_create(&config, &handle);
  if (create_status != INFERENCE_STATUS_OK) {
    std::cerr << "inference_create failed with status " << create_status << "\n";
    return 1;
  }

  InferenceOutputSpec spec{};
  const InferenceStatus spec_status = inference_get_output_spec(handle, &spec);
  if (spec_status != INFERENCE_STATUS_OK) {
    std::cerr << "inference_get_output_spec failed with status " << spec_status << ": "
              << inference_last_error(handle) << "\n";
    inference_destroy(handle);
    return 1;
  }

  std::vector<float> cls_preds(spec.cls_count, 0.0f);
  std::vector<float> box_preds(spec.box_count, 0.0f);
  std::vector<int> label_preds(spec.label_count, 0);

  InferenceRequest request{};
  request.points = points.data();
  request.point_count = static_cast<int>(points.size() / kNumPointFeatures);
  request.motion_4x4 = nullptr;
  request.enable_runtime = 1;

  InferenceResponse response{};
  response.cls_preds = cls_preds.data();
  response.cls_capacity = cls_preds.size();
  response.box_preds = box_preds.data();
  response.box_capacity = box_preds.size();
  response.label_preds = label_preds.data();
  response.label_capacity = label_preds.size();
  response.freespace = nullptr;
  response.freespace_capacity = 0;

  const InferenceStatus forward_status =
      inference_forward(handle, &request, &response);
  if (forward_status != INFERENCE_STATUS_OK) {
    std::cerr << "inference_forward failed with status " << forward_status << ": "
              << inference_last_error(handle) << "\n";
    inference_destroy(handle);
    return 1;
  }

  std::cout << "Loaded points: " << request.point_count << "\n";
  std::cout << "Detected objects: " << response.cls_count << "\n";

  const size_t preview_count = std::min(response.cls_count, kMaxPreviewDetections);
  std::cout << std::fixed << std::setprecision(4);
  for (size_t i = 0; i < preview_count; ++i) {
    const size_t box_offset = i * kBoxDim;
    std::cout << "det[" << i << "] score=" << response.cls_preds[i]
              << " label=" << response.label_preds[i]
              << " box=[";
    for (int dim = 0; dim < kBoxDim; ++dim) {
      if (dim > 0) {
        std::cout << ", ";
      }
      std::cout << response.box_preds[box_offset + static_cast<size_t>(dim)];
    }
    std::cout << "]\n";
  }

  inference_destroy(handle);
  return 0;
}
