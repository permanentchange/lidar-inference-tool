#include "pillar_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct CliOptions {
  std::string model_dir;
  std::string pcd_path;
  std::string dump_dir = "outputs/perf_debug";
  std::string override_config_path;
  int warmup_iters = 10;
  int profile_iters = 50;
};

void PrintUsage(const char* program) {
  std::cerr
      << "Usage: " << program
      << " <model_dir> <pcd_path> [dump_dir] [--warmup N] [--profile N]"
      << " [--override-config PATH]\n";
}

bool ParseInt(const std::string& value, int* out_value) {
  try {
    size_t parsed = 0;
    const int result = std::stoi(value, &parsed);
    if (parsed != value.size()) {
      return false;
    }
    *out_value = result;
    return true;
  } catch (...) {
    return false;
  }
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
    if (arg == "--warmup") {
      if (arg_index >= argc || !ParseInt(argv[arg_index++], &options.warmup_iters)) {
        if (out_error != nullptr) {
          *out_error = "Invalid value for --warmup";
        }
        return false;
      }
    } else if (arg == "--profile") {
      if (arg_index >= argc || !ParseInt(argv[arg_index++], &options.profile_iters)) {
        if (out_error != nullptr) {
          *out_error = "Invalid value for --profile";
        }
        return false;
      }
    } else if (arg == "--override-config") {
      if (arg_index >= argc) {
        if (out_error != nullptr) {
          *out_error = "Missing path for --override-config";
        }
        return false;
      }
      options.override_config_path = argv[arg_index++];
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

void WarnAboutConflictingLibraries() {
  const char* ld_library_path = std::getenv("LD_LIBRARY_PATH");
  if (ld_library_path != nullptr &&
      std::string(ld_library_path).find("IGV/devel/lib") != std::string::npos) {
    std::cerr
        << "Warning: LD_LIBRARY_PATH contains IGV/devel/lib. If debug inference fails with "
        << "TensorRT enqueueV2 binding errors, prefer launching the built tool with the "
        << "repository's build-local iou3d_nms and voxelization libraries are loaded first.\n";
  }
}

struct ProfileBucket {
  std::string name;
  std::vector<double> values;
};

double Percentile(std::vector<double> values, double ratio) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double rank = ratio * static_cast<double>(values.size() - 1);
  const size_t low = static_cast<size_t>(rank);
  const size_t high = std::min(low + 1, values.size() - 1);
  const double alpha = rank - static_cast<double>(low);
  return values[low] * (1.0 - alpha) + values[high] * alpha;
}

struct TimingStats {
  double mean = 0.0;
  double min = 0.0;
  double max = 0.0;
  double p50 = 0.0;
  double p90 = 0.0;
};

TimingStats ComputeStats(const std::vector<double>& values) {
  TimingStats stats;
  if (values.empty()) {
    return stats;
  }
  const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
  stats.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  stats.min = *min_it;
  stats.max = *max_it;
  stats.p50 = Percentile(values, 0.50);
  stats.p90 = Percentile(values, 0.90);
  return stats;
}

void PrintStats(const std::string& name, const std::vector<double>& values) {
  if (values.empty()) {
    return;
  }
  const TimingStats stats = ComputeStats(values);
  std::cout << name << ": mean=" << stats.mean << " ms"
            << ", min=" << stats.min << " ms"
            << ", max=" << stats.max << " ms"
            << ", p50=" << stats.p50 << " ms"
            << ", p90=" << stats.p90 << " ms\n";
}

void AppendStatJson(std::ostringstream* oss, const std::string& name, const TimingStats& stats, bool trailing_comma) {
  *oss << "    \"" << name << "\": {\n";
  *oss << "      \"mean\": " << stats.mean << ",\n";
  *oss << "      \"min\": " << stats.min << ",\n";
  *oss << "      \"max\": " << stats.max << ",\n";
  *oss << "      \"p50\": " << stats.p50 << ",\n";
  *oss << "      \"p90\": " << stats.p90 << "\n";
  *oss << "    }";
  if (trailing_comma) {
    *oss << ",";
  }
  *oss << "\n";
}

bool IsCenterPointVoxelBackend(const inference_internal::ToolConfig& config) {
  return config.runtime.runtime_backend ==
         inference_internal::RuntimeBackend::kCenterPointVoxel;
}

const char* PillarOrVoxelCountName(const inference_internal::ToolConfig& config) {
  return IsCenterPointVoxelBackend(config) ? "voxel_count" : "pillar_count";
}

const char* PillarOrVoxelStageName(const inference_internal::ToolConfig& config) {
  return IsCenterPointVoxelBackend(config) ? "voxelize" : "pillarize";
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

bool WriteTimingsJson(const inference_internal::ToolConfig& config,
                      double pcd_load_ms,
                      int input_points,
                      int pillar_count,
                      size_t decoded_candidates,
                      const inference_internal::StageTimings& single_run,
                      const std::vector<double>& input_pack_values,
                      const std::vector<double>& pillarize_values,
                      const std::vector<double>& h2d_values,
                      const std::vector<double>& trt_values,
                      const std::vector<double>& d2h_values,
                      const std::vector<double>& decode_values,
                      const std::vector<double>& output_pack_values,
                      const std::vector<double>& total_values,
                      std::string* out_error) {
  std::filesystem::create_directories(config.dump_dir);
  const std::filesystem::path output_path =
      std::filesystem::path(config.dump_dir) / "timings.json";
  std::ofstream output(output_path);
  if (!output.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open timings output: " + output_path.string();
    }
    return false;
  }

  std::ostringstream json;
  const char* count_name = PillarOrVoxelCountName(config);
  const char* voxelize_stage_name = PillarOrVoxelStageName(config);
  json << "{\n";
  json << "  \"config\": {\n";
  json << "    \"warmup_iters\": " << config.warmup_iters << ",\n";
  json << "    \"profile_iters\": " << config.profile_iters << ",\n";
  json << "    \"pcd_path\": \"" << JsonEscape(config.pcd_path) << "\"\n";
  json << "  },\n";
  json << "  \"debug_run\": {\n";
  json << "    \"pcd_load_ms\": " << pcd_load_ms << ",\n";
  json << "    \"input_points\": " << input_points << ",\n";
  json << "    \"" << count_name << "\": " << pillar_count << ",\n";
  json << "    \"decoded_candidates\": " << decoded_candidates << ",\n";
  json << "    \"timings_ms\": {\n";
  json << "      \"" << voxelize_stage_name << "\": " << single_run.pillarize_ms << ",\n";
  json << "      \"h2d_prepare\": " << single_run.h2d_prepare_ms << ",\n";
  json << "      \"trt_forward\": " << single_run.trt_forward_ms << ",\n";
  json << "      \"d2h_collect\": " << single_run.d2h_collect_ms << ",\n";
  json << "      \"decode\": " << single_run.decode_ms << ",\n";
  json << "      \"output_pack\": " << single_run.output_pack_ms << ",\n";
  json << "      \"total_forward\": " << single_run.total_forward_ms << "\n";
  json << "    }\n";
  json << "  },\n";
  json << "  \"profile_summary_ms\": {\n";
  AppendStatJson(&json, "pcd_load", ComputeStats(std::vector<double>{pcd_load_ms}), true);
  AppendStatJson(&json, "input_pack", ComputeStats(input_pack_values), true);
  AppendStatJson(&json, voxelize_stage_name, ComputeStats(pillarize_values), true);
  AppendStatJson(&json, "h2d_prepare", ComputeStats(h2d_values), true);
  AppendStatJson(&json, "trt_forward", ComputeStats(trt_values), true);
  AppendStatJson(&json, "d2h_collect", ComputeStats(d2h_values), true);
  AppendStatJson(&json, "decode", ComputeStats(decode_values), true);
  AppendStatJson(&json, "output_pack", ComputeStats(output_pack_values), true);
  AppendStatJson(&json, "total_forward", ComputeStats(total_values), false);
  json << "  }\n";
  json << "}\n";
  output << json.str();
  return true;
}

bool WriteMetadataJson(const inference_internal::ToolConfig& config,
                       const inference_internal::PointCloudLoadInfo& point_cloud_info,
                       int pillar_count,
                       size_t decoded_candidates,
                       std::string* out_error) {
  std::filesystem::create_directories(config.dump_dir);
  const std::filesystem::path output_path =
      std::filesystem::path(config.dump_dir) / "metadata.json";
  std::ofstream output(output_path);
  if (!output.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open metadata output: " + output_path.string();
    }
    return false;
  }

  output << "{\n";
  output << "  \"tool\": \"perf_debug_tool\",\n";
  output << "  \"model_dir\": \"" << JsonEscape(config.model_dir) << "\",\n";
  output << "  \"pcd_path\": \"" << JsonEscape(config.pcd_path) << "\",\n";
  output << "  \"override_config_path\": \"" << JsonEscape(config.override_config_path) << "\",\n";
  output << "  \"result_artifacts\": \"disabled\",\n";
  output << "  \"input\": {\n";
  output << "    \"format\": \"" << JsonEscape(point_cloud_info.format) << "\",\n";
  output << "    \"source_points\": " << point_cloud_info.source_points << ",\n";
  output << "    \"loaded_points\": " << point_cloud_info.loaded_points << ",\n";
  output << "    \"features\": " << point_cloud_info.num_point_features << ",\n";
  output << "    \"truncated\": " << (point_cloud_info.truncated ? "true" : "false") << "\n";
  output << "  },\n";
  output << "  \"runtime\": {\n";
  output << "    \"" << PillarOrVoxelCountName(config) << "\": " << pillar_count << ",\n";
  output << "    \"decoded_candidates\": " << decoded_candidates << "\n";
  output << "  }\n";
  output << "}\n";
  return true;
}

InferenceRequest MakeRequest(const std::vector<float>& points, int num_point_features) {
  InferenceRequest request{};
  request.points = points.data();
  request.point_count = static_cast<int>(points.size() / static_cast<size_t>(num_point_features));
  request.motion_4x4 = nullptr;
  request.enable_runtime = 1;
  return request;
}

void ResetResponseCounts(InferenceResponse* response) {
  response->cls_count = 0;
  response->box_count = 0;
  response->label_count = 0;
  response->freespace_count = 0;
}

void PrintSingleRunSummary(const inference_internal::ToolConfig& config,
                           double pcd_load_ms,
                           int input_points,
                           const inference_internal::DebugArtifacts& debug) {
  std::cout << "Performance debug run complete.\n";
  std::cout << "pcd_load: " << pcd_load_ms << " ms\n";
  std::cout << "input_points: " << input_points << "\n";
  std::cout << PillarOrVoxelCountName(config) << ": "
            << debug.pillar_inputs.num_pillars << "\n";
  std::cout << "decoded_candidates: " << debug.decoded_candidates.scores.size() << "\n";
  std::cout << "Single-run timings:\n";
  std::cout << "  " << PillarOrVoxelStageName(config) << "="
            << debug.timings.pillarize_ms << " ms\n";
  std::cout << "  h2d_prepare=" << debug.timings.h2d_prepare_ms << " ms\n";
  std::cout << "  trt_forward=" << debug.timings.trt_forward_ms << " ms\n";
  std::cout << "  d2h_collect=" << debug.timings.d2h_collect_ms << " ms\n";
  std::cout << "  decode=" << debug.timings.decode_ms << " ms\n";
  std::cout << "  output_pack=" << debug.timings.output_pack_ms << " ms\n";
  std::cout << "  total_forward=" << debug.timings.total_forward_ms << " ms\n";
}

std::string JoinStrings(const std::vector<std::string>& values, const std::string& separator) {
  std::ostringstream oss;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      oss << separator;
    }
    oss << values[i];
  }
  return oss.str();
}

void PrintPointCloudLoadSummary(const inference_internal::PointCloudLoadInfo& info,
                                double pcd_load_ms) {
  std::cout << "Loaded point cloud: path=" << info.path
            << ", format=" << info.format
            << ", fields=" << JoinStrings(info.fields, " ")
            << ", source_points=" << info.source_points
            << ", loaded_points=" << info.loaded_points
            << ", features=" << info.num_point_features
            << ", truncated=" << (info.truncated ? "yes" : "no")
            << ", load_ms=" << pcd_load_ms << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  WarnAboutConflictingLibraries();

  CliOptions options;
  std::string error;
  if (!ParseCliOptions(argc, argv, &options, &error)) {
    std::cerr << error << "\n";
    PrintUsage(argv[0]);
    return 1;
  }

  inference_internal::ToolConfig config;
  InferenceStatus status = inference_internal::LoadToolConfigFromModelDir(
      options.model_dir, options.pcd_path, options.dump_dir, options.override_config_path,
      options.warmup_iters, options.profile_iters, 0.2f, &config, &error);
  if (status != INFERENCE_STATUS_OK) {
    std::cerr << "Failed to load tool config: " << error << "\n";
    return 1;
  }

  std::vector<float> points;
  inference_internal::PointCloudLoadInfo point_cloud_info;
  const auto pcd_start = std::chrono::steady_clock::now();
  status = inference_internal::LoadPcdFile(config.pcd_path, config.runtime.num_point_features,
                                           config.runtime.max_points, &points, &error,
                                           &point_cloud_info);
  const auto pcd_end = std::chrono::steady_clock::now();
  if (status != INFERENCE_STATUS_OK) {
    std::cerr << "Failed to load point cloud: " << error << "\n";
    return 1;
  }
  const double pcd_load_ms =
      std::chrono::duration<double, std::milli>(pcd_end - pcd_start).count();
  PrintPointCloudLoadSummary(point_cloud_info, pcd_load_ms);

  inference_internal::PillarRuntime runtime(config.runtime);
  status = runtime.Initialize();
  if (status != INFERENCE_STATUS_OK) {
    std::cerr << "Failed to initialize runtime: " << runtime.last_error() << "\n";
    return 1;
  }

  const InferenceOutputSpec spec = runtime.output_spec();
  std::vector<float> cls_preds(spec.cls_count, 0.0f);
  std::vector<float> box_preds(spec.box_count, 0.0f);
  std::vector<int> label_preds(spec.label_count, 0);

  const InferenceRequest request = MakeRequest(points, config.runtime.num_point_features);
  const inference_internal::PostprocessConfig postprocess =
      inference_internal::MakeToolPostprocessConfig(config);

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
  inference_internal::StageTimings debug_timing;
  status = runtime.ForwardNms(request, postprocess, &response, &debug, &debug_timing);
  if (status != INFERENCE_STATUS_OK) {
    std::cerr << "Performance debug forward failed: " << runtime.last_error() << "\n";
    return 1;
  }
  debug.timings = debug_timing;

  PrintSingleRunSummary(config, pcd_load_ms, request.point_count, debug);

  for (int i = 0; i < config.warmup_iters; ++i) {
    ResetResponseCounts(&response);
    status = runtime.ForwardNms(request, postprocess, &response, nullptr, nullptr);
    if (status != INFERENCE_STATUS_OK) {
      std::cerr << "Warmup forward failed: " << runtime.last_error() << "\n";
      return 1;
    }
  }

  std::vector<double> input_pack_values(config.profile_iters, 0.0);
  std::vector<double> pillarize_values;
  std::vector<double> h2d_values;
  std::vector<double> trt_values;
  std::vector<double> d2h_values;
  std::vector<double> decode_values;
  std::vector<double> output_pack_values;
  std::vector<double> total_values;
  pillarize_values.reserve(config.profile_iters);
  h2d_values.reserve(config.profile_iters);
  trt_values.reserve(config.profile_iters);
  d2h_values.reserve(config.profile_iters);
  decode_values.reserve(config.profile_iters);
  output_pack_values.reserve(config.profile_iters);
  total_values.reserve(config.profile_iters);

  for (int i = 0; i < config.profile_iters; ++i) {
    ResetResponseCounts(&response);
    inference_internal::StageTimings timing;
    const auto pack_start = std::chrono::steady_clock::now();
    const auto pack_end = std::chrono::steady_clock::now();
    input_pack_values[static_cast<size_t>(i)] =
        std::chrono::duration<double, std::milli>(pack_end - pack_start).count();

    status = runtime.ForwardNms(request, postprocess, &response, nullptr, &timing);
    if (status != INFERENCE_STATUS_OK) {
      std::cerr << "Profile forward failed: " << runtime.last_error() << "\n";
      return 1;
    }
    pillarize_values.push_back(timing.pillarize_ms);
    h2d_values.push_back(timing.h2d_prepare_ms);
    trt_values.push_back(timing.trt_forward_ms);
    d2h_values.push_back(timing.d2h_collect_ms);
    decode_values.push_back(timing.decode_ms);
    output_pack_values.push_back(timing.output_pack_ms);
    total_values.push_back(timing.total_forward_ms);
  }

  std::cout << "\nProfile summary (" << config.profile_iters << " iters after "
            << config.warmup_iters << " warmup):\n";
  PrintStats("pcd_load", std::vector<double>{pcd_load_ms});
  PrintStats("input_pack", input_pack_values);
  PrintStats(PillarOrVoxelStageName(config), pillarize_values);
  PrintStats("h2d_prepare", h2d_values);
  PrintStats("trt_forward", trt_values);
  PrintStats("d2h_collect", d2h_values);
  PrintStats("decode", decode_values);
  PrintStats("output_pack", output_pack_values);
  PrintStats("total_forward", total_values);
  if (!WriteTimingsJson(config, pcd_load_ms, request.point_count,
                        debug.pillar_inputs.num_pillars,
                        debug.decoded_candidates.scores.size(),
                        debug.timings, input_pack_values, pillarize_values, h2d_values, trt_values,
                        d2h_values, decode_values, output_pack_values, total_values, &error)) {
    std::cerr << "Failed to save timings.json: " << error << "\n";
    return 1;
  }
  if (!WriteMetadataJson(config, point_cloud_info, debug.pillar_inputs.num_pillars,
                         debug.decoded_candidates.scores.size(), &error)) {
    std::cerr << "Failed to save metadata.json: " << error << "\n";
    return 1;
  }
  return 0;
}
