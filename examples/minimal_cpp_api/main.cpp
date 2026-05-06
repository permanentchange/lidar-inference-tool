#include "inference_cpp.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// 读取 float32 .bin 点云文件，并按指定点特征维度解释为紧凑数组。
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
  std::cerr << "Usage: " << program << " <model_dir> <pcd_bin_path> [pcd_bin_path ...]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    PrintUsage(argc > 0 ? argv[0] : "minimal_cpp_api_example");
    return 1;
  }

  // 第一个参数是模型目录，后续参数是一帧或多帧点云文件路径。
  const std::string model_dir = argv[1];
  std::vector<std::string> point_cloud_paths;
  point_cloud_paths.reserve(static_cast<size_t>(argc - 2));
  for (int i = 2; i < argc; ++i) {
    point_cloud_paths.emplace_back(argv[i]);
  }

  std::string error;
  std::unique_ptr<inference::LidarInferenceRuntime> runtime;
  // 在启动阶段一次性加载 model.yaml 和 TensorRT engine，运行时复用同一个 runtime。
  const InferenceStatus create_status =
      inference::LidarInferenceRuntime::CreateFromModelDir(model_dir, &runtime, &error);
  if (create_status != INFERENCE_STATUS_OK) {
    std::cerr << "CreateFromModelDir failed with status " << create_status << ": "
              << error << "\n";
    return 1;
  }

  // 点特征维度来自模型配置，而不是样例里写死，避免调用方和模型约定不一致。
  const int num_point_features = runtime->config().model.num_point_features;
  std::vector<float> points;
  // 预留最大点数对应的 host buffer，减少多帧处理时的重复扩容。
  points.reserve(static_cast<size_t>(runtime->config().runtime.max_points) *
                 static_cast<size_t>(num_point_features));
  // 输出检测结果容器在循环里重复使用，避免每帧重新分配。
  std::vector<inference::Detection> detections;

  using Clock = std::chrono::steady_clock;
  const auto total_start = Clock::now();
  size_t succeeded_files = 0;
  size_t failed_files = 0;
  size_t total_loaded_points = 0;
  size_t total_detected_objects = 0;
  double total_load_ms = 0.0;
  double total_infer_ms = 0.0;
  std::vector<std::string> failure_messages;

  for (const std::string& point_cloud_path : point_cloud_paths) {
    std::string load_error;
    // 逐帧读取点云，样例只演示最常见的 float32 .bin 输入格式。
    const auto load_start = Clock::now();
    const bool loaded = LoadPointCloudBin(point_cloud_path, num_point_features, &points, &load_error);
    const auto load_end = Clock::now();
    total_load_ms += std::chrono::duration<double, std::milli>(load_end - load_start).count();
    if (!loaded) {
      ++failed_files;
      failure_messages.push_back(point_cloud_path + ": " + load_error);
      continue;
    }

    inference::PointCloudView cloud{};
    // PointCloudView 只做轻量视图封装，不复制点云数据。
    cloud.points = points.data();
    cloud.point_count = static_cast<int>(points.size() / static_cast<size_t>(num_point_features));
    total_loaded_points += static_cast<size_t>(cloud.point_count);

    // Infer() 走的是库内封装好的完整流程，返回 NMS 之后的最终检测结果。
    const auto infer_start = Clock::now();
    const InferenceStatus infer_status = runtime->Infer(cloud, &detections);
    const auto infer_end = Clock::now();
    total_infer_ms += std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    if (infer_status != INFERENCE_STATUS_OK) {
      ++failed_files;
      failure_messages.push_back(point_cloud_path + ": " + runtime->last_error());
      continue;
    }

    ++succeeded_files;
    total_detected_objects += detections.size();
  }

  const auto total_end = Clock::now();
  const double total_elapsed_ms =
      std::chrono::duration<double, std::milli>(total_end - total_start).count();

  std::cout << "Processed files: " << point_cloud_paths.size() << "\n";
  std::cout << "Succeeded files: " << succeeded_files << "\n";
  std::cout << "Failed files: " << failed_files << "\n";
  std::cout << "Total loaded points: " << total_loaded_points << "\n";
  std::cout << "Total detected objects: " << total_detected_objects << "\n";
  std::cout << "Total load time: " << total_load_ms << " ms\n";
  std::cout << "Total inference time: " << total_infer_ms << " ms\n";
  std::cout << "Total elapsed time: " << total_elapsed_ms << " ms\n";

  if (!detections.empty()) {
    // 只打印最后一帧中的前几个目标，便于快速确认输出格式是否正确。
    const size_t preview_count = std::min<size_t>(detections.size(), 5);
    std::cout << "Last frame preview (" << preview_count << " detections):\n";
    for (size_t i = 0; i < preview_count; ++i) {
      const inference::Detection& detection = detections[i];
      std::cout << "  [" << i << "] label=" << detection.label;
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

  if (!failure_messages.empty()) {
    // 汇总失败文件，便于调用方批处理时一次性排查异常输入。
    std::cerr << "Failed inputs:\n";
    for (const std::string& message : failure_messages) {
      std::cerr << "  " << message << "\n";
    }
  }
  return failed_files == 0 ? 0 : 1;
}
