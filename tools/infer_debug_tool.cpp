#include "inference_cpp.hpp"
#include "pillar_runtime.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct CliOptions {
  std::string model_dir;
  std::string input_path;
  std::string dump_dir = "outputs/infer_debug";
  std::string override_config_path;
  bool dump_decoded = false;
  bool dump_internal = false;
};

struct TimestampInfo {
  uint64_t timestamp_ns = 0;
  double timestamp_seconds = 0.0;
};

struct FrameInput {
  size_t frame_index = 0;
  std::filesystem::path input_path;
  std::string file_name;
  std::string file_stem;
  std::optional<TimestampInfo> timestamp;
};

struct FrameOutputSummary {
  size_t frame_index = 0;
  std::filesystem::path input_path;
  std::filesystem::path frame_dir;
  std::string file_name;
  std::string file_stem;
  std::optional<TimestampInfo> timestamp;
  size_t final_count = 0;
  size_t decoded_count = 0;
};

void PrintUsage(const char* program) {
  std::cerr
      << "Usage: " << program
      << " <model_dir> <input_path> [dump_dir] [--override-config PATH]"
      << " [--dump-decoded] [--dump-internal]\n";
}

bool ParseCliOptions(int argc, char** argv, CliOptions* out_options, std::string* out_error) {
  if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    PrintUsage(argv[0]);
    std::exit(0);
  }
  if (argc < 3) {
    if (out_error != nullptr) {
      *out_error = "model_dir and input_path are required";
    }
    return false;
  }

  CliOptions options;
  int arg_index = 1;
  options.model_dir = argv[arg_index++];
  options.input_path = argv[arg_index++];

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
                     const std::filesystem::path& input_path,
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
  output << "  \"file\": \"" << JsonEscape(input_path.string()) << "\",\n";
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
                       const std::filesystem::path& input_path,
                       const inference::PackageConfig& package,
                       const inference_internal::PointCloudLoadInfo& point_cloud_info,
                       const char* input_path_kind,
                       const std::optional<size_t>& frame_index,
                       const std::optional<TimestampInfo>& timestamp,
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
  output << "  \"pcd_path\": \"" << JsonEscape(input_path.string()) << "\",\n";
  output << "  \"override_config_path\": \"" << JsonEscape(options.override_config_path) << "\",\n";
  output << "  \"input_path_kind\": \"" << JsonEscape(input_path_kind) << "\",\n";
  output << "  \"frame_index\": ";
  if (frame_index.has_value()) {
    output << *frame_index;
  } else {
    output << "null";
  }
  output << ",\n";
  output << "  \"timestamp_ns\": ";
  if (timestamp.has_value()) {
    output << timestamp->timestamp_ns;
  } else {
    output << "null";
  }
  output << ",\n";
  output << "  \"timestamp_seconds\": ";
  if (timestamp.has_value()) {
    output << std::fixed << std::setprecision(9) << timestamp->timestamp_seconds;
  } else {
    output << "null";
  }
  output << ",\n";
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

bool IsSupportedPointCloudPath(const std::filesystem::path& path) {
  if (!std::filesystem::is_regular_file(path)) {
    return false;
  }
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return extension == ".pcd" || extension == ".bin";
}

std::optional<TimestampInfo> ParseTimestampFromStem(const std::string& stem) {
  if (stem.empty()) {
    return std::nullopt;
  }
  for (const char ch : stem) {
    if (ch < '0' || ch > '9') {
      return std::nullopt;
    }
  }
  try {
    const uint64_t timestamp_ns = static_cast<uint64_t>(std::stoull(stem));
    TimestampInfo info;
    info.timestamp_ns = timestamp_ns;
    info.timestamp_seconds = static_cast<double>(timestamp_ns) / 1e9;
    return info;
  } catch (...) {
    return std::nullopt;
  }
}

std::string ZeroPadNumber(size_t value, size_t width) {
  std::ostringstream oss;
  oss << std::setw(static_cast<int>(width)) << std::setfill('0') << value;
  return oss.str();
}

InferenceStatus CollectFrameInputs(const std::filesystem::path& input_path,
                                   std::vector<FrameInput>* out_frames,
                                   std::string* out_error) {
  if (out_frames == nullptr) {
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  out_frames->clear();
  if (std::filesystem::is_regular_file(input_path)) {
    if (!IsSupportedPointCloudPath(input_path)) {
      if (out_error != nullptr) {
        *out_error = "Unsupported input file extension: " + input_path.string();
      }
      return INFERENCE_STATUS_INVALID_ARGUMENT;
    }
    FrameInput frame;
    frame.frame_index = 0;
    frame.input_path = input_path.lexically_normal();
    frame.file_name = input_path.filename().string();
    frame.file_stem = input_path.stem().string();
    frame.timestamp = ParseTimestampFromStem(frame.file_stem);
    out_frames->push_back(frame);
    return INFERENCE_STATUS_OK;
  }

  if (!std::filesystem::is_directory(input_path)) {
    if (out_error != nullptr) {
      *out_error = "input_path must be an existing .pcd/.bin file or directory: " +
                   input_path.string();
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  std::vector<std::filesystem::path> candidates;
  for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(input_path)) {
    if (IsSupportedPointCloudPath(entry.path())) {
      candidates.push_back(entry.path().lexically_normal());
    }
  }
  if (candidates.empty()) {
    if (out_error != nullptr) {
      *out_error = "No .pcd or .bin files found in directory: " + input_path.string();
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  bool all_numeric_stems = true;
  for (const std::filesystem::path& candidate : candidates) {
    if (!ParseTimestampFromStem(candidate.stem().string()).has_value()) {
      all_numeric_stems = false;
      break;
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [all_numeric_stems](const std::filesystem::path& lhs,
                                const std::filesystem::path& rhs) {
              if (all_numeric_stems) {
                return std::stoull(lhs.stem().string()) < std::stoull(rhs.stem().string());
              }
              return lhs.filename().string() < rhs.filename().string();
            });

  out_frames->reserve(candidates.size());
  for (size_t index = 0; index < candidates.size(); ++index) {
    FrameInput frame;
    frame.frame_index = index;
    frame.input_path = candidates[index];
    frame.file_name = candidates[index].filename().string();
    frame.file_stem = candidates[index].stem().string();
    frame.timestamp = all_numeric_stems ? ParseTimestampFromStem(frame.file_stem) : std::nullopt;
    out_frames->push_back(std::move(frame));
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus WriteSequenceManifest(const std::filesystem::path& path,
                                      const CliOptions& options,
                                      const std::filesystem::path& input_dir,
                                      bool timestamps_available,
                                      const std::vector<FrameOutputSummary>& frames,
                                      std::string* out_error) {
  std::ofstream output(path);
  if (!output.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open sequence manifest output: " + path.string();
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  output << "{\n";
  output << "  \"tool\": \"infer_debug_tool\",\n";
  output << "  \"mode\": \"batch_directory\",\n";
  output << "  \"model_dir\": \"" << JsonEscape(options.model_dir) << "\",\n";
  output << "  \"input_dir\": \"" << JsonEscape(input_dir.string()) << "\",\n";
  output << "  \"override_config_path\": \"" << JsonEscape(options.override_config_path) << "\",\n";
  output << "  \"frame_count\": " << frames.size() << ",\n";
  output << "  \"timestamp_source\": \""
         << (timestamps_available ? "filename_stem_unix_ns" : "filename_stem_unavailable")
         << "\",\n";
  output << "  \"timestamp_unit\": \"" << (timestamps_available ? "nanoseconds" : "null")
         << "\",\n";
  output << "  \"dump_decoded\": " << (options.dump_decoded ? "true" : "false") << ",\n";
  output << "  \"dump_internal\": " << (options.dump_internal ? "true" : "false") << ",\n";
  output << "  \"frames\": [\n";
  for (size_t i = 0; i < frames.size(); ++i) {
    const FrameOutputSummary& frame = frames[i];
    output << "    {\n";
    output << "      \"frame_index\": " << frame.frame_index << ",\n";
    output << "      \"file_name\": \"" << JsonEscape(frame.file_name) << "\",\n";
    output << "      \"file_path\": \"" << JsonEscape(frame.input_path.string()) << "\",\n";
    output << "      \"file_stem\": \"" << JsonEscape(frame.file_stem) << "\",\n";
    output << "      \"timestamp_ns\": ";
    if (frame.timestamp.has_value()) {
      output << frame.timestamp->timestamp_ns;
    } else {
      output << "null";
    }
    output << ",\n";
    output << "      \"timestamp_seconds\": ";
    if (frame.timestamp.has_value()) {
      output << std::fixed << std::setprecision(9) << frame.timestamp->timestamp_seconds;
    } else {
      output << "null";
    }
    output << ",\n";
    output << "      \"frame_dir\": \"" << JsonEscape(frame.frame_dir.string()) << "\",\n";
    output << "      \"final_count\": " << frame.final_count << ",\n";
    output << "      \"decoded_count\": " << frame.decoded_count << "\n";
    output << "    }";
    if (i + 1 < frames.size()) {
      output << ",";
    }
    output << "\n";
  }
  output << "  ]\n";
  output << "}\n";
  return INFERENCE_STATUS_OK;
}

InferenceStatus WriteSequenceDetectionsJsonl(const std::filesystem::path& path,
                                             const std::vector<FrameOutputSummary>& frames,
                                             const std::vector<std::vector<inference::Detection>>& all_detections,
                                             std::string* out_error) {
  std::ofstream output(path);
  if (!output.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open sequence detections output: " + path.string();
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }
  if (frames.size() != all_detections.size()) {
    if (out_error != nullptr) {
      *out_error = "Sequence detections size mismatch";
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  for (size_t i = 0; i < frames.size(); ++i) {
    const FrameOutputSummary& frame = frames[i];
    const std::vector<inference::Detection>& detections = all_detections[i];
    output << "{";
    output << "\"frame_index\":" << frame.frame_index << ",";
    output << "\"timestamp_ns\":";
    if (frame.timestamp.has_value()) {
      output << frame.timestamp->timestamp_ns;
    } else {
      output << "null";
    }
    output << ",";
    output << "\"timestamp_seconds\":";
    if (frame.timestamp.has_value()) {
      output << std::fixed << std::setprecision(9) << frame.timestamp->timestamp_seconds;
    } else {
      output << "null";
    }
    output << ",";
    output << "\"file_name\":\"" << JsonEscape(frame.file_name) << "\",";
    output << "\"file_path\":\"" << JsonEscape(frame.input_path.string()) << "\",";
    output << "\"frame_dir\":\"" << JsonEscape(frame.frame_dir.string()) << "\",";
    output << "\"detections\":[";
    for (size_t det_index = 0; det_index < detections.size(); ++det_index) {
      const inference::Detection& detection = detections[det_index];
      output << "{";
      output << "\"box\":[";
      for (size_t dim = 0; dim < inference::kBoxDimension; ++dim) {
        if (dim > 0) {
          output << ",";
        }
        output << detection.box[dim];
      }
      output << "],";
      output << "\"score\":" << detection.score << ",";
      output << "\"label\":" << detection.label << ",";
      output << "\"label_name\":\"" << JsonEscape(DetectionLabelName(detection)) << "\"";
      output << "}";
      if (det_index + 1 < detections.size()) {
        output << ",";
      }
    }
    output << "]}\n";
  }
  return INFERENCE_STATUS_OK;
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

InferenceStatus RunSingleInput(const CliOptions& options,
                               const std::filesystem::path& input_path,
                               const std::optional<size_t>& frame_index,
                               const std::optional<TimestampInfo>& timestamp,
                               const char* input_path_kind,
                               std::unique_ptr<inference::LidarInferenceRuntime>* runtime,
                               FrameOutputSummary* out_summary,
                               std::vector<inference::Detection>* out_final_detections,
                               std::string* out_error) {
  if (runtime == nullptr || runtime->get() == nullptr || out_summary == nullptr ||
      out_final_detections == nullptr) {
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  inference_internal::ToolConfig tool_config;
  InferenceStatus status = inference_internal::LoadToolConfigFromModelDir(
      options.model_dir, input_path.string(), options.dump_dir, options.override_config_path,
      0, 1, 0.2f, &tool_config, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  tool_config.dump_dir = options.dump_dir;

  std::vector<float> points;
  inference_internal::PointCloudLoadInfo point_cloud_info;
  status = inference_internal::LoadPcdFile(
      input_path.string(), (*runtime)->config().model.num_point_features,
      (*runtime)->config().runtime.max_points, &points, out_error, &point_cloud_info);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  const std::filesystem::path dump_dir(options.dump_dir);
  std::filesystem::create_directories(dump_dir);

  status = inference_internal::SaveFloatNpy(
      (dump_dir / "input_points.npy").string(), points.data(), points.size(),
      {points.empty() ? 0UL
                      : points.size() /
                            static_cast<size_t>((*runtime)->config().model.num_point_features),
       static_cast<size_t>((*runtime)->config().model.num_point_features)},
      out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  const inference::PointCloudView cloud =
      MakeCloudView(points, (*runtime)->config().model.num_point_features);
  std::vector<inference::Detection> final_detections;
  status = (*runtime)->Infer(cloud, &final_detections);
  if (status != INFERENCE_STATUS_OK) {
    if (out_error != nullptr) {
      *out_error = (*runtime)->last_error();
    }
    return status;
  }

  status = SaveDetectionArrays(dump_dir, "final", final_detections, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  std::vector<float> final_boxes;
  std::vector<float> final_scores;
  std::vector<int32_t> final_labels;
  std::vector<std::string> final_label_names;
  PackDetections(final_detections, &final_boxes, &final_scores, &final_labels,
                 &final_label_names);
  if (!WriteLabelNamesJson(dump_dir / "final_label_names.json", final_label_names, out_error)) {
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }
  if (!WriteResultJson(dump_dir / "result.json", options, input_path, final_detections, out_error)) {
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  size_t decoded_count = 0;
  if (options.dump_decoded) {
    std::vector<inference::Detection> decoded_detections;
    status = (*runtime)->InferDecoded(cloud, &decoded_detections);
    if (status != INFERENCE_STATUS_OK) {
      if (out_error != nullptr) {
        *out_error = (*runtime)->last_error();
      }
      return status;
    }
    decoded_count = decoded_detections.size();
    status = SaveDetectionArrays(dump_dir, "decoded", decoded_detections, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
  }

  if (options.dump_internal) {
    status = DumpInternalAuxiliary(tool_config, points, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
  }

  if (!WriteMetadataJson(dump_dir / "metadata.json", options, input_path, (*runtime)->config(),
                         point_cloud_info, input_path_kind, frame_index, timestamp,
                         final_detections.size(), decoded_count, out_error)) {
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  out_summary->frame_index = frame_index.value_or(0);
  out_summary->input_path = input_path;
  out_summary->frame_dir = dump_dir;
  out_summary->file_name = input_path.filename().string();
  out_summary->file_stem = input_path.stem().string();
  out_summary->timestamp = timestamp;
  out_summary->final_count = final_detections.size();
  out_summary->decoded_count = decoded_count;
  *out_final_detections = std::move(final_detections);
  return INFERENCE_STATUS_OK;
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

  std::vector<FrameInput> frames;
  InferenceStatus status = CollectFrameInputs(options.input_path, &frames, &error);
  if (status != INFERENCE_STATUS_OK) {
    std::cerr << "Failed to collect input frames: " << error << "\n";
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

  if (frames.size() == 1 && std::filesystem::is_regular_file(std::filesystem::path(options.input_path))) {
    FrameOutputSummary summary;
    std::vector<inference::Detection> final_detections;
    status = RunSingleInput(options, frames[0].input_path, std::nullopt, frames[0].timestamp,
                            "file", &runtime, &summary, &final_detections, &error);
    if (status != INFERENCE_STATUS_OK) {
      std::cerr << "Single-file inference failed with status " << status << ": "
                << error << "\n";
      return 1;
    }

    std::cout << "Inference debug run complete.\n";
    std::cout << "input_file: " << summary.input_path.string() << "\n";
    std::cout << "final_detections: " << summary.final_count << "\n";
    std::cout << "dump_dir: " << summary.frame_dir.string() << "\n";
    return 0;
  }

  const std::filesystem::path root_dump_dir(options.dump_dir);
  const std::filesystem::path frames_dir = root_dump_dir / "frames";
  std::filesystem::create_directories(frames_dir);

  std::vector<FrameOutputSummary> summaries;
  std::vector<std::vector<inference::Detection>> all_detections;
  summaries.reserve(frames.size());
  all_detections.reserve(frames.size());
  const size_t frame_index_width = std::max<size_t>(4, std::to_string(frames.size() - 1).size());
  bool timestamps_available = true;

  for (const FrameInput& frame : frames) {
    CliOptions frame_options = options;
    frame_options.dump_dir =
        (frames_dir / (ZeroPadNumber(frame.frame_index, frame_index_width) + "_" + frame.file_stem))
            .string();
    FrameOutputSummary summary;
    std::vector<inference::Detection> final_detections;
    status = RunSingleInput(frame_options, frame.input_path, frame.frame_index, frame.timestamp,
                            "directory_frame", &runtime, &summary, &final_detections, &error);
    if (status != INFERENCE_STATUS_OK) {
      std::cerr << "Batch inference failed on frame " << frame.frame_index << " ("
                << frame.input_path.string() << ") with status " << status << ": "
                << error << "\n";
      return 1;
    }
    if (!frame.timestamp.has_value()) {
      timestamps_available = false;
      summary.timestamp = std::nullopt;
    }
    summaries.push_back(std::move(summary));
    all_detections.push_back(std::move(final_detections));
  }

  status = WriteSequenceManifest(root_dump_dir / "sequence_manifest.json", options,
                                 std::filesystem::path(options.input_path), timestamps_available,
                                 summaries, &error);
  if (status != INFERENCE_STATUS_OK) {
    std::cerr << "Failed to write sequence manifest: " << error << "\n";
    return 1;
  }
  status = WriteSequenceDetectionsJsonl(root_dump_dir / "sequence_detections.jsonl",
                                        summaries, all_detections, &error);
  if (status != INFERENCE_STATUS_OK) {
    std::cerr << "Failed to write sequence detections: " << error << "\n";
    return 1;
  }

  std::cout << "Batch inference debug run complete.\n";
  std::cout << "input_dir: " << std::filesystem::path(options.input_path).string() << "\n";
  std::cout << "frame_count: " << summaries.size() << "\n";
  std::cout << "dump_dir: " << root_dump_dir.string() << "\n";
  return 0;
}
