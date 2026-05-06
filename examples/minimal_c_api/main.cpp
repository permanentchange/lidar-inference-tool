#include "inference.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kNumPointFeatures = 5;

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

  out_points->resize(total_floats);
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
  std::cerr << "Usage: " << program << " <engine_path> <pcd_bin_path> [pcd_bin_path ...]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    PrintUsage(argc > 0 ? argv[0] : "minimal_c_api_example");
    return 1;
  }

  const std::string engine_path = argv[1];
  std::vector<std::string> point_cloud_paths;
  point_cloud_paths.reserve(static_cast<size_t>(argc - 2));
  for (int i = 2; i < argc; ++i) {
    point_cloud_paths.emplace_back(argv[i]);
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

  InferencePostprocessConfig postprocess{};
  const InferenceStatus postprocess_status =
      inference_get_default_postprocess_config(&postprocess);
  if (postprocess_status != INFERENCE_STATUS_OK) {
    std::cerr << "inference_get_default_postprocess_config failed with status "
              << postprocess_status << ": " << inference_last_error(handle) << "\n";
    inference_destroy(handle);
    return 1;
  }

  std::vector<float> cls_preds(spec.cls_count, 0.0f);
  std::vector<float> box_preds(spec.box_count, 0.0f);
  std::vector<int> label_preds(spec.label_count, 0);
  std::vector<float> points;
  points.reserve(static_cast<size_t>(config.max_points) *
                 static_cast<size_t>(config.num_point_features));

  InferenceRequest request{};
  request.points = nullptr;
  request.point_count = 0;
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

  using Clock = std::chrono::steady_clock;
  const auto total_start = Clock::now();
  size_t succeeded_files = 0;
  size_t failed_files = 0;
  size_t total_loaded_points = 0;
  size_t total_consumed_points = 0;
  size_t total_detected_objects = 0;
  double total_load_ms = 0.0;
  double total_infer_ms = 0.0;
  std::vector<std::string> failure_messages;

  for (const std::string& point_cloud_path : point_cloud_paths) {
    std::string error;
    const auto load_start = Clock::now();
    const bool loaded = LoadPointCloudBin(point_cloud_path, kNumPointFeatures, &points, &error);
    const auto load_end = Clock::now();
    total_load_ms += std::chrono::duration<double, std::milli>(load_end - load_start).count();
    if (!loaded) {
      ++failed_files;
      failure_messages.push_back(point_cloud_path + ": " + error);
      continue;
    }

    request.points = points.data();
    request.point_count = static_cast<int>(points.size() / static_cast<size_t>(kNumPointFeatures));
    total_loaded_points += static_cast<size_t>(request.point_count);
    total_consumed_points += static_cast<size_t>(
        std::max(0, std::min(request.point_count, config.max_points)));

    response.cls_count = 0;
    response.box_count = 0;
    response.label_count = 0;
    response.freespace_count = 0;

    const auto infer_start = Clock::now();
    const InferenceStatus forward_status =
        inference_forward_nms(handle, &request, &postprocess, &response);
    const auto infer_end = Clock::now();
    total_infer_ms += std::chrono::duration<double, std::milli>(infer_end - infer_start).count();

    if (forward_status != INFERENCE_STATUS_OK) {
      ++failed_files;
      std::ostringstream oss;
      oss << point_cloud_path << ": inference_forward_nms failed with status " << forward_status
          << ": " << inference_last_error(handle);
      failure_messages.push_back(oss.str());
      continue;
    }

    ++succeeded_files;
    total_detected_objects += response.cls_count;
  }

  const auto total_end = Clock::now();
  const double total_elapsed_ms =
      std::chrono::duration<double, std::milli>(total_end - total_start).count();

  std::cout << "Processed files: " << point_cloud_paths.size() << "\n";
  std::cout << "Succeeded files: " << succeeded_files << "\n";
  std::cout << "Failed files: " << failed_files << "\n";
  std::cout << "Total loaded points: " << total_loaded_points << "\n";
  std::cout << "Total consumed points: " << total_consumed_points << "\n";
  std::cout << "Total detected objects: " << total_detected_objects << "\n";
  std::cout << "Total load time: " << total_load_ms << " ms\n";
  std::cout << "Total inference time: " << total_infer_ms << " ms\n";
  std::cout << "Total elapsed time: " << total_elapsed_ms << " ms\n";
  if (succeeded_files > 0) {
    const double success_count = static_cast<double>(succeeded_files);
    std::cout << "Mean load time per successful file: " << (total_load_ms / success_count)
              << " ms\n";
    std::cout << "Mean inference time per successful file: " << (total_infer_ms / success_count)
              << " ms\n";
  }
  if (!failure_messages.empty()) {
    std::cout << "Failures:\n";
    for (const std::string& failure_message : failure_messages) {
      std::cout << "  " << failure_message << "\n";
    }
  }

  inference_destroy(handle);
  return failure_messages.empty() ? 0 : 1;
}
