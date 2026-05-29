#include "inference_cpp.hpp"
#include "pillar_runtime.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct CliOptions {
  std::string model_dir;
  std::string pcd_path;
  std::string dump_dir = "outputs/infer_debug";
  std::string override_config_path;
  bool dump_decoded = false;
  bool dump_internal = false;
};

void PrintUsage(const char* program) {
  std::cerr
      << "Usage: " << program
      << " <model_dir> <pcd_path> [dump_dir] [--override-config PATH]"
      << " [--dump-decoded] [--dump-internal]\n";
}

bool ParseCliOptions(int argc, char** argv, CliOptions* out_options, std::string* out_error) {
  if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    PrintUsage(argv[0]);
    std::exit(0);
  }
  if (argc < 3) {
    if (out_error != nullptr) {
      *out_error = "model_dir and pcd_path are required";
    }
    return false;
  }

  CliOptions options;
  int arg_index = 1;
  options.model_dir = argv[arg_index++];
  options.pcd_path = argv[arg_index++];

  if (arg_index < argc && std::string(argv[arg_index]).rfind("--", 0) != 0) {
    options.dump_dir = argv[arg_index++];
  }

  while (arg_index < argc) {
    const std::string arg = argv[arg_index++];
    if (arg == "--override-config") {
      if (arg_index >= argc) {
        if (out_error != nullptr) {
          *out_error = "Missing path for --override-config";
        }
        return false;
      }
      options.override_config_path = argv[arg_index++];
    } else if (arg == "--dump-decoded") {
      options.dump_decoded = true;
    } else if (arg == "--dump-internal") {
      options.dump_internal = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      if (out_error != nullptr) {
        *out_error = "Unknown argument: " + arg;
      }
      return false;
    }
  }

  *out_options = options;
  return true;
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream oss;
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        oss << "\\\\";
        break;
      case '"':
        oss << "\\\"";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        oss << ch;
        break;
    }
  }
  return oss.str();
}

InferenceRequest MakeRequest(const std::vector<float>& points, int num_point_features) {
  InferenceRequest request{};
  request.points = points.data();
  request.point_count = static_cast<int>(points.size() / static_cast<size_t>(num_point_features));
  request.motion_4x4 = nullptr;
  request.enable_runtime = 1;
  return request;
}

inference::PointCloudView MakeCloudView(const std::vector<float>& points, int num_point_features) {
  inference::PointCloudView cloud{};
  cloud.points = points.data();
  cloud.point_count = static_cast<int>(points.size() / static_cast<size_t>(num_point_features));
  cloud.motion_4x4 = nullptr;
  cloud.enable_runtime = 1;
  return cloud;
}

std::string DetectionLabelName(const inference::Detection& detection) {
  if (detection.label_name != nullptr) {
    return detection.label_name;
  }
  std::ostringstream oss;
  oss << "class_" << detection.label;
  return oss.str();
}

void PackDetections(const std::vector<inference::Detection>& detections,
                    std::vector<float>* boxes,
                    std::vector<float>* scores,
                    std::vector<int32_t>* labels,
                    std::vector<std::string>* label_names) {
  boxes->clear();
  scores->clear();
  labels->clear();
  label_names->clear();
  boxes->reserve(detections.size() * inference::kBoxDimension);
  scores->reserve(detections.size());
  labels->reserve(detections.size());
  label_names->reserve(detections.size());
  for (const inference::Detection& detection : detections) {
    scores->push_back(detection.score);
    labels->push_back(static_cast<int32_t>(detection.label));
    label_names->push_back(DetectionLabelName(detection));
    boxes->insert(boxes->end(), detection.box.begin(), detection.box.end());
  }
}

bool WriteLabelNamesJson(const std::filesystem::path& path,
                         const std::vector<std::string>& label_names,
                         std::string* out_error) {
  std::ofstream output(path);
  if (!output.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open label names output: " + path.string();
    }
    return false;
  }
  output << "[\n";
  for (size_t i = 0; i < label_names.size(); ++i) {
    output << "  \"" << JsonEscape(label_names[i]) << "\"";
    if (i + 1 < label_names.size()) {
      output << ",";
    }
    output << "\n";
  }
  output << "]\n";
  return true;
}

bool WriteResultJson(const std::filesystem::path& path,
                     const CliOptions& options,
                     const std::vector<inference::Detection>& detections,
                     std::string* out_error) {
  std::ofstream output(path);
  if (!output.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open result output: " + path.string();
    }
    return false;
  }

  output << "{\n";
  output << "  \"tool\": \"infer_debug_tool\",\n";
  output << "  \"result_source\": \"high_level_cpp_api\",\n";
  output << "  \"file\": \"" << JsonEscape(options.pcd_path) << "\",\n";
  output << "  \"model_dir\": \"" << JsonEscape(options.model_dir) << "\",\n";
  output << "  \"override_config_path\": \"" << JsonEscape(options.override_config_path) << "\",\n";
  output << "  \"count\": " << detections.size() << ",\n";
  output << "  \"box_z_origin\": \"center\",\n";
  output << "  \"objects\": [\n";
  for (size_t i = 0; i < detections.size(); ++i) {
    const inference::Detection& detection = detections[i];
    output << "    {\n";
    output << "      \"box\": [";
    for (size_t dim = 0; dim < inference::kBoxDimension; ++dim) {
      if (dim > 0) {
        output << ", ";
      }
      output << detection.box[dim];
    }
    output << "],\n";
    output << "      \"score\": " << detection.score << ",\n";
    output << "      \"label\": " << detection.label << ",\n";
    output << "      \"label_name\": \"" << JsonEscape(DetectionLabelName(detection)) << "\"\n";
    output << "    }";
    if (i + 1 < detections.size()) {
      output << ",";
    }
    output << "\n";
  }
  output << "  ]\n";
  output << "}\n";
  return true;
}

bool WriteMetadataJson(const std::filesystem::path& path,
                       const CliOptions& options,
                       const inference::PackageConfig& package,
                       const inference_internal::PointCloudLoadInfo& point_cloud_info,
                       size_t final_count,
                       size_t decoded_count,
                       std::string* out_error) {
  std::ofstream output(path);
  if (!output.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open metadata output: " + path.string();
    }
    return false;
  }

  output << "{\n";
  output << "  \"tool\": \"infer_debug_tool\",\n";
  output << "  \"model_dir\": \"" << JsonEscape(options.model_dir) << "\",\n";
  output << "  \"pcd_path\": \"" << JsonEscape(options.pcd_path) << "\",\n";
  output << "  \"override_config_path\": \"" << JsonEscape(options.override_config_path) << "\",\n";
  output << "  \"final_source\": \"high_level_cpp_api\",\n";
  output << "  \"decoded_source\": \""
         << (options.dump_decoded ? "high_level_cpp_api" : "disabled") << "\",\n";
  output << "  \"internal_dump_source\": \""
         << (options.dump_internal ? "internal_auxiliary" : "disabled") << "\",\n";
  output << "  \"model\": {\n";
  output << "    \"num_point_features\": " << package.model.num_point_features << ",\n";
  output << "    \"max_points\": " << package.runtime.max_points << ",\n";
  output << "    \"label_count\": " << package.model.label_names.size() << "\n";
  output << "  },\n";
  output << "  \"input\": {\n";
  output << "    \"format\": \"" << JsonEscape(point_cloud_info.format) << "\",\n";
  output << "    \"source_points\": " << point_cloud_info.source_points << ",\n";
  output << "    \"loaded_points\": " << point_cloud_info.loaded_points << ",\n";
  output << "    \"features\": " << point_cloud_info.num_point_features << ",\n";
  output << "    \"truncated\": " << (point_cloud_info.truncated ? "true" : "false") << "\n";
  output << "  },\n";
  output << "  \"results\": {\n";
  output << "    \"final_count\": " << final_count << ",\n";
  output << "    \"decoded_count\": " << decoded_count << "\n";
  output << "  }\n";
  output << "}\n";
  return true;
}

InferenceStatus SaveDetectionArrays(const std::filesystem::path& dump_dir,
                                    const std::string& prefix,
                                    const std::vector<inference::Detection>& detections,
                                    std::string* out_error) {
  std::vector<float> boxes;
  std::vector<float> scores;
  std::vector<int32_t> labels;
  std::vector<std::string> label_names;
  PackDetections(detections, &boxes, &scores, &labels, &label_names);

  InferenceStatus status = inference_internal::SaveFloatNpy(
      (dump_dir / (prefix + "_boxes.npy")).string(), boxes.data(), boxes.size(),
      {detections.size(), inference::kBoxDimension}, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = inference_internal::SaveFloatNpy(
      (dump_dir / (prefix + "_scores.npy")).string(), scores.data(), scores.size(),
      {scores.size()}, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  return inference_internal::SaveInt32Npy(
      (dump_dir / (prefix + "_labels.npy")).string(), labels.data(), labels.size(),
      {labels.size()}, out_error);
}

InferenceStatus DumpInternalAuxiliary(const inference_internal::ToolConfig& config,
                                      const std::vector<float>& points,
                                      std::string* out_error) {
  inference_internal::PillarRuntime runtime(config.runtime);
  InferenceStatus status = runtime.Initialize();
  if (status != INFERENCE_STATUS_OK) {
    if (out_error != nullptr) {
      *out_error = runtime.last_error();
    }
    return status;
  }

  const InferenceOutputSpec spec = runtime.output_spec();
  std::vector<float> cls_preds(spec.cls_count, 0.0f);
  std::vector<float> box_preds(spec.box_count, 0.0f);
  std::vector<int> label_preds(spec.label_count, 0);

  InferenceResponse response{};
  response.cls_preds = cls_preds.data();
  response.cls_capacity = cls_preds.size();
  response.box_preds = box_preds.data();
  response.box_capacity = box_preds.size();
  response.label_preds = label_preds.data();
  response.label_capacity = label_preds.size();
  response.freespace = nullptr;
  response.freespace_capacity = 0;

  inference_internal::DebugArtifacts debug;
  inference_internal::StageTimings timings;
  const inference_internal::PostprocessConfig postprocess =
      inference_internal::MakeToolPostprocessConfig(config);
  status = runtime.ForwardNms(MakeRequest(points, config.runtime.num_point_features),
                              postprocess, &response, &debug, &timings);
  if (status != INFERENCE_STATUS_OK) {
    if (out_error != nullptr) {
      *out_error = runtime.last_error();
    }
    return status;
  }
  debug.timings = timings;
  return inference_internal::SaveInternalAuxiliaryArtifacts(debug, config, out_error);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  CliOptions options;
  std::string error;
  if (!ParseCliOptions(argc, argv, &options, &error)) {
    std::cerr << error << "\n";
    PrintUsage(argv[0]);
    return 1;
  }

  inference_internal::ToolConfig tool_config;
  InferenceStatus status = inference_internal::LoadToolConfigFromModelDir(
      options.model_dir, options.pcd_path, options.dump_dir, options.override_config_path,
      0, 1, 0.2f, &tool_config, &error);
  if (status != INFERENCE_STATUS_OK) {
    std::cerr << "Failed to load tool config: " << error << "\n";
    return 1;
  }

  std::unique_ptr<inference::LidarInferenceRuntime> runtime;
  status = inference::LidarInferenceRuntime::CreateFromModelDir(
      options.model_dir, &runtime, &error, options.override_config_path);
  if (status != INFERENCE_STATUS_OK) {
    std::cerr << "CreateFromModelDir failed with status " << status << ": "
              << error << "\n";
    return 1;
  }

  std::vector<float> points;
  inference_internal::PointCloudLoadInfo point_cloud_info;
  status = inference_internal::LoadPcdFile(
      options.pcd_path, runtime->config().model.num_point_features,
      runtime->config().runtime.max_points, &points, &error, &point_cloud_info);
  if (status != INFERENCE_STATUS_OK) {
    std::cerr << "Failed to load point cloud: " << error << "\n";
    return 1;
  }

  const std::filesystem::path dump_dir(options.dump_dir);
  std::filesystem::create_directories(dump_dir);

  status = inference_internal::SaveFloatNpy(
      (dump_dir / "input_points.npy").string(), points.data(), points.size(),
      {points.empty() ? 0UL
                      : points.size() /
                            static_cast<size_t>(runtime->config().model.num_point_features),
       static_cast<size_t>(runtime->config().model.num_point_features)},
      &error);
  if (status != INFERENCE_STATUS_OK) {
    std::cerr << "Failed to save input_points.npy: " << error << "\n";
    return 1;
  }

  const inference::PointCloudView cloud =
      MakeCloudView(points, runtime->config().model.num_point_features);
  std::vector<inference::Detection> final_detections;
  status = runtime->Infer(cloud, &final_detections);
  if (status != INFERENCE_STATUS_OK) {
    std::cerr << "Infer failed with status " << status << ": "
              << runtime->last_error() << "\n";
    return 1;
  }

  status = SaveDetectionArrays(dump_dir, "final", final_detections, &error);
  if (status != INFERENCE_STATUS_OK) {
    std::cerr << "Failed to save final detection arrays: " << error << "\n";
    return 1;
  }

  std::vector<float> final_boxes;
  std::vector<float> final_scores;
  std::vector<int32_t> final_labels;
  std::vector<std::string> final_label_names;
  PackDetections(final_detections, &final_boxes, &final_scores, &final_labels,
                 &final_label_names);
  if (!WriteLabelNamesJson(dump_dir / "final_label_names.json", final_label_names, &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  if (!WriteResultJson(dump_dir / "result.json", options, final_detections, &error)) {
    std::cerr << error << "\n";
    return 1;
  }

  size_t decoded_count = 0;
  if (options.dump_decoded) {
    std::vector<inference::Detection> decoded_detections;
    status = runtime->InferDecoded(cloud, &decoded_detections);
    if (status != INFERENCE_STATUS_OK) {
      std::cerr << "InferDecoded failed with status " << status << ": "
                << runtime->last_error() << "\n";
      return 1;
    }
    decoded_count = decoded_detections.size();
    status = SaveDetectionArrays(dump_dir, "decoded", decoded_detections, &error);
    if (status != INFERENCE_STATUS_OK) {
      std::cerr << "Failed to save decoded detection arrays: " << error << "\n";
      return 1;
    }
  }

  if (options.dump_internal) {
    status = DumpInternalAuxiliary(tool_config, points, &error);
    if (status != INFERENCE_STATUS_OK) {
      std::cerr << "Failed to save internal auxiliary artifacts: " << error << "\n";
      return 1;
    }
  }

  if (!WriteMetadataJson(dump_dir / "metadata.json", options, runtime->config(),
                         point_cloud_info, final_detections.size(), decoded_count, &error)) {
    std::cerr << error << "\n";
    return 1;
  }

  std::cout << "Inference debug run complete.\n";
  std::cout << "input_points: " << cloud.point_count << "\n";
  std::cout << "final_detections: " << final_detections.size() << "\n";
  std::cout << "dump_dir: " << dump_dir.string() << "\n";
  return 0;
}
