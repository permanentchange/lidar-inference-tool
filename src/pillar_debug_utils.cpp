#include "pillar_runtime.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "iou3d_nms_api.h"

namespace inference_internal {
namespace {

constexpr int kBoxDim = 7;
constexpr int kLabelCount = 3;

template <typename T>
std::string DtypeTag();

template <>
std::string DtypeTag<float>() {
  return "<f4";
}

template <>
std::string DtypeTag<int32_t>() {
  return "<i4";
}

std::string ShapeToNpyString(const std::vector<size_t>& shape) {
  std::ostringstream oss;
  oss << "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    oss << shape[i];
    if (shape.size() == 1) {
      oss << ",";
    } else if (i + 1 < shape.size()) {
      oss << ", ";
    }
  }
  oss << ")";
  return oss.str();
}

std::string ShapeToJsonArray(const std::vector<size_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
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

std::string NormalizeNmsType(const std::string& value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return normalized;
}

std::string ResolveConfigRelativePath(const std::filesystem::path& config_path,
                                      const std::string& value) {
  if (value.empty()) {
    return value;
  }

  const std::filesystem::path raw_path(value);
  if (raw_path.is_absolute()) {
    return raw_path.lexically_normal().string();
  }

  const std::filesystem::path base_dir = config_path.parent_path();
  if (base_dir.empty()) {
    return raw_path.lexically_normal().string();
  }
  return (base_dir / raw_path).lexically_normal().string();
}

bool UseCircleNms(const ToolConfig& config) {
  return NormalizeNmsType(config.nms_type) == "circle";
}

template <typename T>
InferenceStatus WriteNpy(const std::filesystem::path& path,
                         const T* data,
                         size_t count,
                         const std::vector<size_t>& shape,
                         std::string* out_error) {
  std::ofstream output(path, std::ios::binary);
  if (!output.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open dump file: " + path.string();
    }
    return INFERENCE_STATUS_BACKEND_ERROR;
  }

  std::ostringstream header_stream;
  header_stream << "{'descr': '" << DtypeTag<T>() << "', 'fortran_order': False, 'shape': "
                << ShapeToNpyString(shape) << ", }";
  std::string header = header_stream.str();
  const size_t base_length = 10 + header.size() + 1;
  const size_t padding = (16 - (base_length % 16)) % 16;
  header.append(padding, ' ');
  header.push_back('\n');

  const char magic[] = "\x93NUMPY";
  output.write(magic, 6);
  const char version[2] = {1, 0};
  output.write(version, 2);
  const uint16_t header_len = static_cast<uint16_t>(header.size());
  output.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  if (count > 0) {
    output.write(reinterpret_cast<const char*>(data),
                 static_cast<std::streamsize>(count * sizeof(T)));
  }
  return INFERENCE_STATUS_OK;
}

template <typename T>
std::string BuildNumericSummary(const std::string& name,
                                const std::string& dtype,
                                const std::vector<size_t>& shape,
                                const std::vector<T>& data) {
  std::ostringstream oss;
  oss << "    \"" << name << "\": {\n";
  oss << "      \"dtype\": \"" << dtype << "\",\n";
  oss << "      \"shape\": " << ShapeToJsonArray(shape) << ",\n";
  oss << "      \"count\": " << data.size();
  if (!data.empty()) {
    const auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
    oss << ",\n      \"min\": " << *min_it << ",\n      \"max\": " << *max_it;
    if constexpr (std::is_floating_point_v<T>) {
      const double sum = std::accumulate(data.begin(), data.end(), 0.0);
      oss << ",\n      \"mean\": " << sum / static_cast<double>(data.size());
    }
    oss << ",\n      \"preview\": [";
    const size_t preview_count = std::min<size_t>(data.size(), 8);
    for (size_t i = 0; i < preview_count; ++i) {
      if (i > 0) {
        oss << ", ";
      }
      oss << data[i];
    }
    oss << "]";
  }
  oss << "\n    }";
  return oss.str();
}

void ApplyPreMaxsize(std::vector<float>* scores,
                     std::vector<float>* boxes,
                     std::vector<int>* labels,
                     int pre_maxsize) {
  if (pre_maxsize <= 0 || scores == nullptr || boxes == nullptr || labels == nullptr ||
      scores->size() <= static_cast<size_t>(pre_maxsize)) {
    return;
  }
  std::vector<size_t> indices(scores->size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&](size_t lhs, size_t rhs) { return (*scores)[lhs] > (*scores)[rhs]; });

  std::vector<float> new_scores;
  std::vector<float> new_boxes;
  std::vector<int> new_labels;
  new_scores.reserve(pre_maxsize);
  new_boxes.reserve(static_cast<size_t>(pre_maxsize) * kBoxDim);
  new_labels.reserve(pre_maxsize);
  for (int i = 0; i < pre_maxsize; ++i) {
    const size_t src = indices[static_cast<size_t>(i)];
    new_scores.push_back((*scores)[src]);
    new_labels.push_back((*labels)[src]);
    new_boxes.insert(new_boxes.end(), boxes->begin() + static_cast<long>(src * kBoxDim),
                     boxes->begin() + static_cast<long>((src + 1) * kBoxDim));
  }
  scores->swap(new_scores);
  boxes->swap(new_boxes);
  labels->swap(new_labels);
}

std::vector<size_t> CircleNmsKeep(const std::vector<float>& scores,
                                  const std::vector<float>& boxes,
                                  float distance_threshold,
                                  int post_maxsize) {
  std::vector<size_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](size_t lhs, size_t rhs) { return scores[lhs] > scores[rhs]; });

  std::vector<size_t> keep;
  keep.reserve(order.size());
  std::vector<uint8_t> suppressed(scores.size(), 0);
  for (size_t order_index = 0; order_index < order.size(); ++order_index) {
    const size_t index = order[order_index];
    if (suppressed[index] != 0U) {
      continue;
    }
    keep.push_back(index);
    for (size_t compare_index = order_index + 1; compare_index < order.size(); ++compare_index) {
      const size_t other = order[compare_index];
      if (suppressed[other] != 0U) {
        continue;
      }
      const float dx = boxes[index * kBoxDim + 0] - boxes[other * kBoxDim + 0];
      const float dy = boxes[index * kBoxDim + 1] - boxes[other * kBoxDim + 1];
      const float squared_distance = dx * dx + dy * dy;
      if (squared_distance <= distance_threshold) {
        suppressed[other] = 1U;
      }
    }
    if (post_maxsize > 0 && keep.size() >= static_cast<size_t>(post_maxsize)) {
      break;
    }
  }
  return keep;
}

void AppendBoxes(const std::vector<float>& source_boxes,
                 size_t source_index,
                 std::vector<float>* target_boxes) {
  target_boxes->insert(target_boxes->end(),
                       source_boxes.begin() + static_cast<long>(source_index * kBoxDim),
                       source_boxes.begin() + static_cast<long>((source_index + 1) * kBoxDim));
}

void FilterByScoreThreshold(CandidateDetections* detections, float score_threshold) {
  if (detections == nullptr || score_threshold <= 0.0f) {
    return;
  }

  CandidateDetections filtered;
  filtered.scores.reserve(detections->scores.size());
  filtered.labels.reserve(detections->labels.size());
  filtered.boxes.reserve(detections->boxes.size());
  for (size_t i = 0; i < detections->scores.size(); ++i) {
    if (detections->scores[i] < score_threshold) {
      continue;
    }
    filtered.scores.push_back(detections->scores[i]);
    filtered.labels.push_back(detections->labels[i]);
    AppendBoxes(detections->boxes, i, &filtered.boxes);
  }
  detections->scores.swap(filtered.scores);
  detections->labels.swap(filtered.labels);
  detections->boxes.swap(filtered.boxes);
}

InferenceStatus LoadBinaryPcdFile(const std::string& path,
                                  int num_point_features,
                                  int max_points,
                                  std::vector<float>* out_points,
                                  std::string* out_error) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open point cloud: " + path;
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  std::string line;
  int point_count = -1;
  std::string data_format;
  while (std::getline(input, line)) {
    if (line.rfind("POINTS ", 0) == 0) {
      point_count = std::stoi(line.substr(7));
    } else if (line.rfind("DATA ", 0) == 0) {
      data_format = line.substr(5);
      break;
    }
  }

  if (point_count <= 0 || data_format != "binary") {
    if (out_error != nullptr) {
      *out_error = "Only binary x/y/z PCD files are supported";
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  point_count = std::min(point_count, max_points);
  std::vector<float> xyz(static_cast<size_t>(point_count) * 3);
  input.read(reinterpret_cast<char*>(xyz.data()),
             static_cast<std::streamsize>(xyz.size() * sizeof(float)));
  if (!input) {
    if (out_error != nullptr) {
      *out_error = "Failed to read binary PCD payload";
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  out_points->assign(static_cast<size_t>(point_count) * num_point_features, 0.0f);
  for (int i = 0; i < point_count; ++i) {
    (*out_points)[static_cast<size_t>(i) * num_point_features + 0] = xyz[static_cast<size_t>(i) * 3 + 0];
    (*out_points)[static_cast<size_t>(i) * num_point_features + 1] = xyz[static_cast<size_t>(i) * 3 + 1];
    (*out_points)[static_cast<size_t>(i) * num_point_features + 2] = xyz[static_cast<size_t>(i) * 3 + 2];
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus LoadFloatBinFile(const std::string& path,
                                 int num_point_features,
                                 int max_points,
                                 std::vector<float>* out_points,
                                 std::string* out_error) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open point cloud: " + path;
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  const std::streamsize file_size = input.tellg();
  if (file_size <= 0 || (file_size % static_cast<std::streamsize>(sizeof(float))) != 0) {
    if (out_error != nullptr) {
      *out_error = "Point cloud .bin file size is not aligned to float32: " + path;
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  const size_t total_floats = static_cast<size_t>(file_size) / sizeof(float);
  int source_features = 0;
  if (num_point_features > 0 && total_floats % static_cast<size_t>(num_point_features) == 0) {
    source_features = num_point_features;
  } else {
    constexpr std::array<int, 3> kCommonFeatureCounts = {5, 4, 3};
    for (const int candidate : kCommonFeatureCounts) {
      if (total_floats % static_cast<size_t>(candidate) == 0) {
        source_features = candidate;
        break;
      }
    }
  }

  if (source_features <= 0) {
    if (out_error != nullptr) {
      *out_error = "Failed to infer feature count for float32 point cloud: " + path;
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  input.seekg(0, std::ios::beg);
  const size_t source_point_count = total_floats / static_cast<size_t>(source_features);
  const size_t point_count = std::min(source_point_count, static_cast<size_t>(max_points));
  std::vector<float> raw_points(point_count * static_cast<size_t>(source_features), 0.0f);
  input.read(reinterpret_cast<char*>(raw_points.data()),
             static_cast<std::streamsize>(raw_points.size() * sizeof(float)));
  if (!input) {
    if (out_error != nullptr) {
      *out_error = "Failed to read float32 point cloud payload";
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  out_points->assign(point_count * static_cast<size_t>(num_point_features), 0.0f);
  size_t copy_features =
      std::min(static_cast<size_t>(source_features), static_cast<size_t>(num_point_features));
  // Match InferTools PointCloudLoader(use_timestamp=False): 5D bin input is treated as XYZI
  // and the trailing feature is zero-padded instead of forwarded verbatim.
  if (source_features >= 5 && num_point_features >= 5) {
    copy_features = 4;
  }
  for (size_t i = 0; i < point_count; ++i) {
    std::copy_n(raw_points.data() + i * static_cast<size_t>(source_features), copy_features,
                out_points->data() + i * static_cast<size_t>(num_point_features));
  }
  return INFERENCE_STATUS_OK;
}

}  // namespace

InferenceStatus LoadToolConfig(const std::string& path,
                               ToolConfig* out_config,
                               std::string* out_error) {
  if (out_config == nullptr) {
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  const std::filesystem::path config_path(path);
  YAML::Node node = YAML::LoadFile(path);
  out_config->engine_path = ResolveConfigRelativePath(
      config_path, node["engine_path"].as<std::string>());
  out_config->pcd_path = ResolveConfigRelativePath(
      config_path, node["pcd_path"].as<std::string>());
  out_config->dump_dir = ResolveConfigRelativePath(
      config_path, node["dump_dir"].as<std::string>());
  out_config->warmup_iters = node["warmup_iters"].as<int>(10);
  out_config->profile_iters = node["profile_iters"].as<int>(50);
  out_config->bev_resolution = node["bev_resolution"].as<float>(0.2f);
  out_config->nms_type = NormalizeNmsType(
      node["post_processing"]["nms_type"].as<std::string>(out_config->nms_type));
  out_config->nms_thresh = node["post_processing"]["nms_thresh"].as<float>(0.25f);
  out_config->nms_pre_maxsize = node["post_processing"]["nms_pre_maxsize"].as<int>(2048);
  out_config->nms_post_maxsize = node["post_processing"]["nms_post_maxsize"].as<int>(256);
  out_config->final_score_threshold =
      node["post_processing"]["final_score_threshold"].as<float>(0.3f);

  const YAML::Node label_thresh = node["post_processing"]["label_score_thresh"];
  out_config->label_score_thresh = {
      label_thresh["1"].as<float>(0.45f),
      label_thresh["2"].as<float>(0.3f),
      label_thresh["3"].as<float>(0.45f),
  };
  const YAML::Node circle_radius = node["post_processing"]["circle_nms_min_radius"];
  if (circle_radius && circle_radius.IsSequence()) {
    for (size_t i = 0; i < kLabelCount && i < circle_radius.size(); ++i) {
      out_config->circle_nms_min_radius[i] = circle_radius[i].as<float>();
    }
  }
  const YAML::Node label_names = node["post_processing"]["label_names"];
  if (label_names && label_names.IsSequence()) {
    for (size_t i = 0; i < kLabelCount && i < label_names.size(); ++i) {
      out_config->label_names[i] = label_names[i].as<std::string>();
    }
  }

  RuntimeConfig& runtime = out_config->runtime;
  runtime.end2end_file = out_config->engine_path;
  runtime.max_points = node["runtime"]["max_points"].as<int>(300000);
  runtime.frame_num = node["runtime"]["frame_num"].as<int>(1);
  runtime.num_point_features = node["runtime"]["num_point_features"].as<int>(5);
  runtime.max_points_per_pillar = node["runtime"]["max_points_per_pillar"].as<int>(20);
  runtime.min_pillars = node["runtime"]["min_pillars"].as<int>(100);
  runtime.max_pillars = node["runtime"]["max_pillars"].as<int>(50000);
  runtime.bbox_max_num = node["runtime"]["bbox_max_num"].as<int>(500);
  runtime.out_size_factor = node["runtime"]["out_size_factor"].as<int>(4);
  runtime.score_threshold = node["runtime"]["score_threshold"].as<float>(0.1f);

  for (size_t i = 0; i < 6; ++i) {
    runtime.point_cloud_range[i] = node["runtime"]["point_cloud_range"][i].as<float>();
    runtime.post_center_range[i] = node["runtime"]["post_center_range"][i].as<float>();
  }
  for (size_t i = 0; i < 3; ++i) {
    runtime.pillar_size[i] = node["runtime"]["pillar_size"][i].as<float>();
  }

  if (out_config->engine_path.empty() || out_config->pcd_path.empty() || out_config->dump_dir.empty()) {
    if (out_error != nullptr) {
      *out_error = "engine_path, pcd_path, and dump_dir must be configured";
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus LoadPcdFile(const std::string& path,
                            int num_point_features,
                            int max_points,
                            std::vector<float>* out_points,
                            std::string* out_error) {
  if (out_points == nullptr) {
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }
  if (num_point_features <= 0 || max_points <= 0) {
    if (out_error != nullptr) {
      *out_error = "num_point_features and max_points must be positive";
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  const std::filesystem::path input_path(path);
  if (input_path.extension() == ".bin") {
    return LoadFloatBinFile(path, num_point_features, max_points, out_points, out_error);
  }
  return LoadBinaryPcdFile(path, num_point_features, max_points, out_points, out_error);
}

InferenceStatus RunClassAgnosticNms(const CandidateDetections& candidates,
                                    const ToolConfig& config,
                                    CandidateDetections* out_result,
                                    std::string* out_error) {
  if (out_result == nullptr) {
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  if (UseCircleNms(config)) {
    out_result->scores.clear();
    out_result->boxes.clear();
    out_result->labels.clear();

    for (int label = 1; label <= kLabelCount; ++label) {
      CandidateDetections per_label;
      per_label.scores.reserve(candidates.scores.size());
      per_label.labels.reserve(candidates.labels.size());
      per_label.boxes.reserve(candidates.boxes.size());
      for (size_t i = 0; i < candidates.scores.size(); ++i) {
        if (candidates.labels[i] != label) {
          continue;
        }
        per_label.scores.push_back(candidates.scores[i]);
        per_label.labels.push_back(label);
        AppendBoxes(candidates.boxes, i, &per_label.boxes);
      }
      const std::vector<size_t> keep =
          CircleNmsKeep(per_label.scores, per_label.boxes,
                        config.circle_nms_min_radius[static_cast<size_t>(label - 1)],
                        config.nms_post_maxsize);
      for (const size_t kept_index : keep) {
        out_result->scores.push_back(per_label.scores[kept_index]);
        out_result->labels.push_back(label);
        AppendBoxes(per_label.boxes, kept_index, &out_result->boxes);
      }
    }
    FilterByScoreThreshold(out_result, config.final_score_threshold);
    return INFERENCE_STATUS_OK;
  }

  std::vector<float> scores;
  std::vector<float> boxes;
  std::vector<int> labels;
  scores.reserve(candidates.scores.size());
  boxes.reserve(candidates.boxes.size());
  labels.reserve(candidates.labels.size());
  for (size_t i = 0; i < candidates.scores.size(); ++i) {
    const int label = candidates.labels[i];
    if (label < 1 || label > kLabelCount) {
      continue;
    }
    if (candidates.scores[i] < config.label_score_thresh[static_cast<size_t>(label - 1)]) {
      continue;
    }
    scores.push_back(candidates.scores[i]);
    labels.push_back(label);
    AppendBoxes(candidates.boxes, i, &boxes);
  }
  ApplyPreMaxsize(&scores, &boxes, &labels, config.nms_pre_maxsize);
  std::vector<long> keep(scores.size(), 0);
  const int keep_count = scores.empty() ? 0 :
      nms_gpu(boxes.data(), keep.data(), config.nms_thresh, static_cast<int>(scores.size()));
  const size_t final_count = std::min(static_cast<size_t>(keep_count),
                                      static_cast<size_t>(config.nms_post_maxsize));

  out_result->scores.clear();
  out_result->boxes.clear();
  out_result->labels.clear();
  out_result->scores.reserve(final_count);
  out_result->boxes.reserve(final_count * kBoxDim);
  out_result->labels.reserve(final_count);
  for (size_t i = 0; i < final_count; ++i) {
    const long src = keep[i];
    out_result->scores.push_back(scores[static_cast<size_t>(src)]);
    out_result->labels.push_back(labels[static_cast<size_t>(src)]);
    AppendBoxes(boxes, static_cast<size_t>(src), &out_result->boxes);
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus SaveDebugArtifacts(const DebugArtifacts& debug,
                                   const ToolConfig& config,
                                   std::string* out_error) {
  const std::filesystem::path dump_dir(config.dump_dir);
  std::filesystem::create_directories(dump_dir);

  InferenceStatus status = WriteNpy<float>(
      dump_dir / "input_points.npy", debug.input_points.data(), debug.input_points.size(),
      {debug.input_points.empty() ? 0UL
                                  : debug.input_points.size() /
                                        static_cast<size_t>(config.runtime.num_point_features),
       static_cast<size_t>(config.runtime.num_point_features)},
      out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<float>(dump_dir / "pillar_voxels.npy", debug.pillar_inputs.voxels.data(),
                           debug.pillar_inputs.voxels.size(),
                           {static_cast<size_t>(debug.pillar_inputs.num_pillars),
                            static_cast<size_t>(config.runtime.max_points_per_pillar),
                            static_cast<size_t>(config.runtime.num_point_features)},
                           out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<int32_t>(dump_dir / "pillar_num_points.npy", debug.pillar_inputs.num_points.data(),
                             debug.pillar_inputs.num_points.size(),
                             {debug.pillar_inputs.num_points.size()}, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<int32_t>(dump_dir / "pillar_coors.npy", debug.pillar_inputs.coors.data(),
                             debug.pillar_inputs.coors.size(),
                             {static_cast<size_t>(debug.pillar_inputs.num_pillars), 4}, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<float>(dump_dir / "bbox_pred.npy", debug.raw_outputs.bbox_pred.data(),
                           debug.raw_outputs.bbox_pred.size(),
                           {1UL, static_cast<size_t>(debug.raw_outputs.bbox_channels),
                            static_cast<size_t>(debug.raw_outputs.height),
                            static_cast<size_t>(debug.raw_outputs.width)},
                           out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<float>(dump_dir / "cls_score.npy", debug.raw_outputs.cls_score.data(),
                           debug.raw_outputs.cls_score.size(),
                           {1UL, static_cast<size_t>(debug.raw_outputs.cls_channels),
                            static_cast<size_t>(debug.raw_outputs.height),
                            static_cast<size_t>(debug.raw_outputs.width)},
                           out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<float>(dump_dir / "dir_cls_pred.npy", debug.raw_outputs.dir_cls_pred.data(),
                           debug.raw_outputs.dir_cls_pred.size(),
                           {1UL, static_cast<size_t>(debug.raw_outputs.dir_channels),
                            static_cast<size_t>(debug.raw_outputs.height),
                            static_cast<size_t>(debug.raw_outputs.width)},
                           out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<float>(dump_dir / "decoded_scores.npy", debug.decoded_candidates.scores.data(),
                           debug.decoded_candidates.scores.size(),
                           {debug.decoded_candidates.scores.size()}, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<float>(dump_dir / "decoded_boxes.npy", debug.decoded_candidates.boxes.data(),
                           debug.decoded_candidates.boxes.size(),
                           {debug.decoded_candidates.scores.size(), static_cast<size_t>(kBoxDim)},
                           out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  std::vector<int32_t> decoded_labels(debug.decoded_candidates.labels.begin(),
                                      debug.decoded_candidates.labels.end());
  status = WriteNpy<int32_t>(dump_dir / "decoded_labels.npy", decoded_labels.data(),
                             decoded_labels.size(), {decoded_labels.size()}, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<float>(dump_dir / "final_nms_scores.npy", debug.final_nms_outputs.scores.data(),
                           debug.final_nms_outputs.scores.size(),
                           {debug.final_nms_outputs.scores.size()}, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = WriteNpy<float>(dump_dir / "final_nms_boxes.npy", debug.final_nms_outputs.boxes.data(),
                           debug.final_nms_outputs.boxes.size(),
                           {debug.final_nms_outputs.scores.size(), static_cast<size_t>(kBoxDim)},
                           out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  std::vector<int32_t> final_nms_labels(debug.final_nms_outputs.labels.begin(),
                                        debug.final_nms_outputs.labels.end());
  status = WriteNpy<int32_t>(dump_dir / "final_nms_labels.npy", final_nms_labels.data(),
                             final_nms_labels.size(), {final_nms_labels.size()}, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  std::ofstream result_json(dump_dir / "result.json");
  if (!result_json.is_open()) {
    if (out_error != nullptr) {
      *out_error = "Failed to open result.json for writing";
    }
    return INFERENCE_STATUS_BACKEND_ERROR;
  }
  result_json << "{\n";
  result_json << "  \"file\": \"" << JsonEscape(config.pcd_path) << "\",\n";
  result_json << "  \"count\": " << debug.final_nms_outputs.scores.size() << ",\n";
  result_json << "  \"objects\": [\n";
  for (size_t i = 0; i < debug.final_nms_outputs.scores.size(); ++i) {
    const int label = debug.final_nms_outputs.labels[i];
    const size_t label_index =
        label >= 1 && label <= kLabelCount ? static_cast<size_t>(label - 1) : 0UL;
    result_json << "    {\n";
    result_json << "      \"box\": [";
    for (int dim = 0; dim < kBoxDim; ++dim) {
      if (dim > 0) {
        result_json << ", ";
      }
      result_json << debug.final_nms_outputs.boxes[i * kBoxDim + static_cast<size_t>(dim)];
    }
    result_json << "],\n";
    result_json << "      \"score\": " << debug.final_nms_outputs.scores[i] << ",\n";
    result_json << "      \"label\": " << static_cast<int>(label_index) << ",\n";
    result_json << "      \"label_name\": \"" << JsonEscape(config.label_names[label_index]) << "\"\n";
    result_json << "    }";
    if (i + 1 < debug.final_nms_outputs.scores.size()) {
      result_json << ",";
    }
    result_json << "\n";
  }
  result_json << "  ]\n";
  result_json << "}\n";

  std::ofstream meta(dump_dir / "metadata.json");
  meta << "{\n";
  meta << "  \"timings_ms\": {\n";
  meta << "    \"pillarize\": " << debug.timings.pillarize_ms << ",\n";
  meta << "    \"h2d_prepare\": " << debug.timings.h2d_prepare_ms << ",\n";
  meta << "    \"trt_forward\": " << debug.timings.trt_forward_ms << ",\n";
  meta << "    \"d2h_collect\": " << debug.timings.d2h_collect_ms << ",\n";
  meta << "    \"decode\": " << debug.timings.decode_ms << ",\n";
  meta << "    \"output_pack\": " << debug.timings.output_pack_ms << ",\n";
  meta << "    \"total_forward\": " << debug.timings.total_forward_ms << "\n";
  meta << "  },\n";
  meta << "  \"tensors\": {\n";
  meta << BuildNumericSummary("decoded_scores", "float32",
                              {debug.decoded_candidates.scores.size()},
                              debug.decoded_candidates.scores) << ",\n";
  meta << BuildNumericSummary("decoded_labels", "int32",
                              {debug.decoded_candidates.labels.size()},
                              debug.decoded_candidates.labels) << ",\n";
  meta << BuildNumericSummary("final_nms_scores", "float32",
                              {debug.final_nms_outputs.scores.size()},
                              debug.final_nms_outputs.scores) << "\n";
  meta << "  }\n";
  meta << "}\n";
  return INFERENCE_STATUS_OK;
}

}  // namespace inference_internal
