#include "inference_cpp.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

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
      std::ostringstream oss;
      oss << "Point cloud file does not match the expected "
          << num_point_features << "-feature layout: " << path;
      *out_error = oss.str();
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
  (void)program;
  std::cerr
      << "Usage: ./build/minimal_cpp_api_example MODEL_DIR POINT_CLOUD_BIN\n"
      << "Example: ./build/minimal_cpp_api_example demo/centerpoint_pillar demo/data/lidar/test1.bin\n";
}

void PrintDetections(const std::vector<inference::Detection>& detections) {
  std::cout << "Detections: " << detections.size() << "\n";
  for (size_t i = 0; i < detections.size(); ++i) {
    const inference::Detection& detection = detections[i];
    std::cout << "[" << i << "] label=" << detection.label;
    if (detection.label_name != nullptr) {
      std::cout << " (" << detection.label_name << ")";
    }
    std::cout << " score=" << detection.score
              << " box=[" << detection.box[0]
              << ", " << detection.box[1]
              << ", " << detection.box[2]
              << ", " << detection.box[3]
              << ", " << detection.box[4]
              << ", " << detection.box[5]
              << ", " << detection.box[6] << "]\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    PrintUsage(argc > 0 ? argv[0] : "minimal_cpp_api_example");
    return 1;
  }

  const std::string model_dir = argv[1];
  const std::string point_cloud_path = argv[2];

  std::string error;
  std::unique_ptr<inference::LidarInferenceRuntime> runtime;
  const InferenceStatus create_status =
      inference::LidarInferenceRuntime::CreateFromModelDir(model_dir, &runtime, &error);
  if (create_status != INFERENCE_STATUS_OK) {
    std::cerr << "CreateFromModelDir failed with status " << create_status << ": "
              << error << "\n";
    return 1;
  }

  std::vector<float> points;
  const int num_point_features = runtime->config().model.num_point_features;
  if (!LoadPointCloudBin(point_cloud_path, num_point_features, &points, &error)) {
    std::cerr << "LoadPointCloudBin failed: " << error << "\n";
    return 1;
  }

  inference::PointCloudView cloud{};
  cloud.points = points.data();
  cloud.point_count = static_cast<int>(points.size() / static_cast<size_t>(num_point_features));

  std::vector<inference::Detection> detections;
  const InferenceStatus infer_status = runtime->Infer(cloud, &detections);
  if (infer_status != INFERENCE_STATUS_OK) {
    std::cerr << "Infer failed with status " << infer_status << ": "
              << runtime->last_error() << "\n";
    return 1;
  }

  PrintDetections(detections);
  return 0;
}
