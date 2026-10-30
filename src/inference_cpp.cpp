#include "runtime_diagnostics.hpp"
#include "inference_cpp.hpp"
#include "pillar_runtime.hpp"
#include "postprocess_config.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <memory>
#include <sstream>
#include <utility>

namespace inference {
namespace {

InferenceStatus SetError(std::string* out_error,
                         InferenceStatus status,
                         const std::string& message) {
  if (out_error != nullptr) {
    *out_error = message;
  }
  return status;
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

std::string NormalizeIdentifier(const std::string& value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char ch) {
                   if (ch == '-' || ch == ' ') {
                     return '_';
                   }
                   return static_cast<char>(std::tolower(ch));
                 });
  return normalized;
}

InferenceStatus ParseDecodeMode(const std::string& value,
                                DecodeMode* out_mode,
                                std::string* out_error) {
  if (out_mode == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, "out_mode is null");
  }
  const std::string normalized = NormalizeIdentifier(value);
  if (normalized.empty() || normalized == "legacy") {
    *out_mode = DecodeMode::kLegacy;
    return INFERENCE_STATUS_OK;
  }
  if (normalized == "centerpoint" || normalized == "centerpoint_mmdeploy" ||
      normalized == "mmdeploy" || normalized == "centerpoint_mmdeploy_unified") {
    *out_mode = DecodeMode::kCenterPoint;
    return INFERENCE_STATUS_OK;
  }
  return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                  "Unsupported model.decode_mode: " + value);
}

InferenceStatus ParseYawEncoding(const std::string& value,
                                 YawEncoding* out_encoding,
                                 std::string* out_error) {
  if (out_encoding == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "out_encoding is null");
  }
  const std::string normalized = NormalizeIdentifier(value);
  if (normalized.empty() || normalized == "sin_cos" || normalized == "sincos") {
    *out_encoding = YawEncoding::kSinCos;
    return INFERENCE_STATUS_OK;
  }
  if (normalized == "direction_theta") {
    *out_encoding = YawEncoding::kDirectionTheta;
    return INFERENCE_STATUS_OK;
  }
  return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                  "Unsupported model.yaw_encoding: " + value);
}

InferenceStatus ParseRuntimeBackend(const std::string& value,
                                    inference_internal::RuntimeBackend* out_backend,
                                    std::string* out_error) {
  if (out_backend == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "out_backend is null");
  }
  const std::string normalized = NormalizeIdentifier(value);
  if (normalized.empty() || normalized == "pillar_end2end" ||
      normalized == "end2end" || normalized == "pillar") {
    *out_backend = inference_internal::RuntimeBackend::kPillarEnd2End;
    return INFERENCE_STATUS_OK;
  }
  if (normalized == "centerpoint_voxel" || normalized == "voxel" ||
      normalized == "cuda_centerpoint") {
    *out_backend = inference_internal::RuntimeBackend::kCenterPointVoxel;
    return INFERENCE_STATUS_OK;
  }
  return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                  "Unsupported model.runtime_backend: " + value);
}

const char* RuntimeBackendName(inference_internal::RuntimeBackend backend) {
  switch (backend) {
    case inference_internal::RuntimeBackend::kCenterPointVoxel:
      return "centerpoint_voxel";
    case inference_internal::RuntimeBackend::kPillarEnd2End:
    default:
      return "pillar_end2end";
  }
}

template <size_t N>
InferenceStatus LoadFloatArray(const YAML::Node& node,
                               const char* field_name,
                               std::array<float, N>* out_values,
                               std::string* out_error) {
  if (!node) {
    return INFERENCE_STATUS_OK;
  }
  if (!node.IsSequence() || node.size() != N) {
    std::ostringstream oss;
    oss << field_name << " must be a sequence of " << N << " floats";
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
  }
  for (size_t i = 0; i < N; ++i) {
    (*out_values)[i] = node[i].as<float>();
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus LoadSparseShape(const YAML::Node& node,
                                std::vector<int>* out_values,
                                std::string* out_error) {
  if (!node) {
    return INFERENCE_STATUS_OK;
  }
  if (out_values == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "model.sparse_shape output is null");
  }
  if (!node.IsSequence() || node.size() != 3) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "model.sparse_shape must be a sequence of 3 positive integers");
  }
  out_values->clear();
  out_values->reserve(3);
  for (size_t i = 0; i < 3; ++i) {
    const int value = node[i].as<int>();
    if (value <= 0) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "model.sparse_shape must be a sequence of 3 positive integers");
    }
    out_values->push_back(value);
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus LoadStringVector(const YAML::Node& node,
                                 const char* field_name,
                                 std::vector<std::string>* out_values,
                                 std::string* out_error) {
  if (!node) {
    return INFERENCE_STATUS_OK;
  }
  if (!node.IsSequence() || node.size() == 0) {
    std::ostringstream oss;
    oss << field_name << " must be a non-empty sequence of strings";
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
  }
  out_values->clear();
  out_values->reserve(node.size());
  for (size_t i = 0; i < node.size(); ++i) {
    out_values->push_back(node[i].as<std::string>());
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus LoadIntVector(const YAML::Node& node,
                              const char* field_name,
                              std::vector<int>* out_values,
                              std::string* out_error) {
  if (!node) {
    return INFERENCE_STATUS_OK;
  }
  if (!node.IsSequence() || node.size() == 0) {
    std::ostringstream oss;
    oss << field_name << " must be a non-empty sequence of integers";
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
  }
  out_values->clear();
  out_values->reserve(node.size());
  for (size_t i = 0; i < node.size(); ++i) {
    out_values->push_back(node[i].as<int>());
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus LoadNestedIntVector(const YAML::Node& node,
                                    const char* field_name,
                                    std::vector<std::vector<int>>* out_values,
                                    std::string* out_error) {
  if (!node) {
    return INFERENCE_STATUS_OK;
  }
  if (!node.IsSequence() || node.size() == 0) {
    std::ostringstream oss;
    oss << field_name << " must be a non-empty sequence of integer sequences";
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
  }
  out_values->clear();
  out_values->reserve(node.size());
  for (size_t i = 0; i < node.size(); ++i) {
    if (!node[i].IsSequence() || node[i].size() == 0) {
      std::ostringstream oss;
      oss << field_name << "[" << i << "] must be a non-empty sequence of integers";
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
    }
    std::vector<int> values;
    values.reserve(node[i].size());
    for (size_t j = 0; j < node[i].size(); ++j) {
      values.push_back(node[i][j].as<int>());
    }
    out_values->push_back(std::move(values));
  }
  return INFERENCE_STATUS_OK;
}

void NormalizeTaskLayout(ModelConfig* model) {
  if (model == nullptr) {
    return;
  }
  if (model->decode_mode == DecodeMode::kCenterPoint) {
    if (model->task_num_classes.empty()) {
      model->task_num_classes.assign(model->label_names.size(), 1);
    }
    if (model->bbox_code_size <= 0) {
      model->bbox_code_size = model->has_velocity ? 9 : 7;
    }
    if (model->task_label_ids.empty()) {
      model->task_label_ids.reserve(model->task_num_classes.size());
      int label_offset = 0;
      for (const int task_classes : model->task_num_classes) {
        std::vector<int> label_ids;
        label_ids.reserve(static_cast<size_t>(task_classes));
        for (int i = 0; i < task_classes; ++i) {
          label_ids.push_back(label_offset + i);
        }
        model->task_label_ids.push_back(std::move(label_ids));
        label_offset += task_classes;
      }
    }
  } else if (model->task_num_classes.empty()) {
    model->task_num_classes = {1, 1, 1};
    model->yaw_encoding = YawEncoding::kSinCos;
  }
}

InferenceStatus ValidatePackageConfig(const PackageConfig& config, std::string* out_error) {
  if (config.model.engine_path.empty()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "model.engine_path must be configured");
  }
  if (config.model.label_names.empty()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "model.label_names must not be empty");
  }
  for (size_t i = 0; i < config.model.label_names.size(); ++i) {
    if (config.model.label_names[i].empty()) {
      std::ostringstream oss;
      oss << "model.label_names[" << i << "] must not be empty";
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
    }
  }
  if (config.model.decode_mode != DecodeMode::kLegacy &&
      config.model.decode_mode != DecodeMode::kCenterPoint) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "model.decode_mode is invalid");
  }
  if (config.model.yaw_encoding != YawEncoding::kSinCos &&
      config.model.yaw_encoding != YawEncoding::kDirectionTheta) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "model.yaw_encoding is invalid");
  }
  if (config.model.num_point_features < 3) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "model.num_point_features must be at least 3");
  }
  if (config.model.out_size_factor <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "model.out_size_factor must be positive");
  }
  if (config.runtime.max_points_per_voxel <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "runtime.max_points_per_voxel must be positive");
  }
  if (config.runtime.min_voxels <= 0 ||
      config.runtime.max_voxels < config.runtime.min_voxels) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "runtime voxel limits must satisfy 0 < min_voxels <= max_voxels");
  }
  if (config.runtime.max_points <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "runtime.max_points must be positive");
  }
  if (config.runtime.bbox_max_num <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "runtime.bbox_max_num must be positive");
  }
  if (!std::isfinite(config.runtime.score_threshold) ||
      config.runtime.score_threshold < 0.0f) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "runtime.score_threshold must be finite and non-negative");
  }

  const auto validate_range = [&](const std::array<float, 6>& range,
                                  const char* field_name) -> InferenceStatus {
    for (size_t i = 0; i < range.size(); ++i) {
      if (!std::isfinite(range[i])) {
        std::ostringstream oss;
        oss << field_name << "[" << i << "] must be finite";
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
      }
    }
    for (size_t axis = 0; axis < 3; ++axis) {
      if (range[axis] >= range[axis + 3]) {
        std::ostringstream oss;
        oss << field_name << " has invalid min/max bounds for axis " << axis;
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
      }
    }
    return INFERENCE_STATUS_OK;
  };
  InferenceStatus status = validate_range(config.model.point_cloud_range,
                                          "model.point_cloud_range");
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = validate_range(config.model.post_center_range, "model.post_center_range");
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  for (size_t i = 0; i < config.model.voxel_size.size(); ++i) {
    if (!std::isfinite(config.model.voxel_size[i]) || config.model.voxel_size[i] <= 0.0f) {
      std::ostringstream oss;
      oss << "model.voxel_size[" << i << "] must be finite and positive";
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
    }
  }

  inference_internal::RuntimeBackend backend;
  status = ParseRuntimeBackend(config.model.runtime_backend, &backend, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  if (backend == inference_internal::RuntimeBackend::kCenterPointVoxel) {
    if (config.model.decode_mode != DecodeMode::kCenterPoint) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "centerpoint_voxel requires model.decode_mode=CenterPoint");
    }
    if (config.model.scn_onnx_path.empty()) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "model.scn_onnx_path must be configured for centerpoint_voxel");
    }
    if (config.model.sparse_shape.size() != 3) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "model.sparse_shape must contain 3 positive dimensions");
    }
    for (size_t i = 0; i < config.model.sparse_shape.size(); ++i) {
      if (config.model.sparse_shape[i] <= 0) {
        std::ostringstream oss;
        oss << "model.sparse_shape[" << i << "] must be positive";
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
      }
    }
  }

  if (config.model.decode_mode == DecodeMode::kCenterPoint) {
    if (config.model.task_num_classes.empty()) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "model.task_num_classes must not be empty in CenterPoint mode");
    }
    int total_classes = 0;
    for (size_t i = 0; i < config.model.task_num_classes.size(); ++i) {
      if (config.model.task_num_classes[i] <= 0) {
        std::ostringstream oss;
        oss << "model.task_num_classes[" << i << "] must be positive";
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
      }
      total_classes += config.model.task_num_classes[i];
    }
    if (total_classes != static_cast<int>(config.model.label_names.size())) {
      std::ostringstream oss;
      oss << "Sum(model.task_num_classes)=" << total_classes
          << " does not match model.label_names.size()="
          << config.model.label_names.size();
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
    }
    if (config.model.task_label_ids.size() != config.model.task_num_classes.size()) {
      std::ostringstream oss;
      oss << "model.task_label_ids size must match model.task_num_classes size, got "
          << config.model.task_label_ids.size() << " and "
          << config.model.task_num_classes.size();
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
    }
    std::vector<bool> seen_label_ids(config.model.label_names.size(), false);
    for (size_t task_index = 0; task_index < config.model.task_label_ids.size();
         ++task_index) {
      const std::vector<int>& label_ids = config.model.task_label_ids[task_index];
      if (label_ids.size() !=
          static_cast<size_t>(config.model.task_num_classes[task_index])) {
        std::ostringstream oss;
        oss << "model.task_label_ids[" << task_index
            << "] size must match model.task_num_classes[" << task_index << "]";
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
      }
      for (const int label_id : label_ids) {
        if (label_id < 0 || label_id >= static_cast<int>(config.model.label_names.size())) {
          std::ostringstream oss;
          oss << "model.task_label_ids[" << task_index
              << "] contains invalid global label " << label_id
              << ", label_names.size()=" << config.model.label_names.size();
          return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
        }
        if (seen_label_ids[static_cast<size_t>(label_id)]) {
          std::ostringstream oss;
          oss << "model.task_label_ids contains duplicate global label " << label_id;
          return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
        }
        seen_label_ids[static_cast<size_t>(label_id)] = true;
      }
    }
    for (size_t label_id = 0; label_id < seen_label_ids.size(); ++label_id) {
      if (!seen_label_ids[label_id]) {
        std::ostringstream oss;
        oss << "model.task_label_ids must cover every global label exactly once; missing "
            << label_id;
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
      }
    }
    if (config.model.bbox_code_size < static_cast<int>(kBoxDimension)) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "model.bbox_code_size must be at least 7 in CenterPoint mode");
    }
    const int min_bbox_code_size = config.model.has_velocity ? 9 : 7;
    if (config.model.bbox_code_size < min_bbox_code_size) {
      std::ostringstream oss;
      oss << "model.bbox_code_size must be at least " << min_bbox_code_size
          << " when model.has_velocity=" << (config.model.has_velocity ? "true" : "false");
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
    }
    inference_internal::ResolvedHeadLayout layout;
    status = inference_internal::ResolveHeadLayout(config.model, &layout, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
  }
  return ValidatePostprocessConfig(config.model, config.postprocess, out_error);
}

struct PackageConfigExtras {
  inference_internal::RuntimeBackend runtime_backend =
      inference_internal::RuntimeBackend::kPillarEnd2End;
  std::string scn_onnx_path;
  std::string head_engine_path;
  std::vector<int> sparse_shape;
  bool has_label_score_thresh = false;
};

InferenceStatus ValidateExtendedPackageConfig(PackageConfig* config,
                                              const PackageConfigExtras& extras,
                                              std::string* out_error) {
  if (config == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, "config is null");
  }
  if (extras.runtime_backend != inference_internal::RuntimeBackend::kCenterPointVoxel) {
    return INFERENCE_STATUS_OK;
  }
  if (extras.scn_onnx_path.empty()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "model.scn_onnx_path must be configured for centerpoint_voxel");
  }
  if (extras.head_engine_path.empty()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "model.head_engine_path must be configured for centerpoint_voxel");
  }
  if (extras.sparse_shape.size() != 3) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "model.sparse_shape must be a sequence of 3 positive integers");
  }
  config->model.engine_path = extras.head_engine_path;
  config->model.runtime_backend = RuntimeBackendName(extras.runtime_backend);
  config->model.scn_onnx_path = extras.scn_onnx_path;
  config->model.head_engine_path = extras.head_engine_path;
  config->model.sparse_shape = extras.sparse_shape;
  return INFERENCE_STATUS_OK;
}

InferenceStatus ApplyPackageConfigNode(const YAML::Node& root,
                                       const std::filesystem::path& config_path,
                                       PackageConfig* out_config,
                                       PackageConfigExtras* out_extras,
                                       std::string* out_error) {
  if (out_config == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, "out_config is null");
  }
  if (!root || !root.IsMap()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "package config must be a YAML map");
  }

  const YAML::Node model = root["model"];
  if (model) {
    if (!model.IsMap()) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "model must be a YAML map");
    }
    if (model["engine_path"]) {
      out_config->model.engine_path =
          ResolveConfigRelativePath(config_path, model["engine_path"].as<std::string>());
    }
    if (model["runtime_backend"]) {
      inference_internal::RuntimeBackend backend =
          inference_internal::RuntimeBackend::kPillarEnd2End;
      InferenceStatus status =
          ParseRuntimeBackend(model["runtime_backend"].as<std::string>(),
                              &backend, out_error);
      if (status != INFERENCE_STATUS_OK) {
        return status;
      }
      out_config->model.runtime_backend = RuntimeBackendName(backend);
      if (out_extras != nullptr) {
        out_extras->runtime_backend = backend;
      }
    }
    if (model["scn_onnx_path"]) {
      out_config->model.scn_onnx_path =
          ResolveConfigRelativePath(config_path, model["scn_onnx_path"].as<std::string>());
      if (out_extras != nullptr) {
        out_extras->scn_onnx_path = out_config->model.scn_onnx_path;
      }
    }
    if (model["head_engine_path"]) {
      out_config->model.head_engine_path =
          ResolveConfigRelativePath(config_path, model["head_engine_path"].as<std::string>());
      if (out_extras != nullptr) {
        out_extras->head_engine_path = out_config->model.head_engine_path;
      }
    }
    if (model["num_point_features"]) {
      out_config->model.num_point_features = model["num_point_features"].as<int>();
    }
    if (model["out_size_factor"]) {
      out_config->model.out_size_factor = model["out_size_factor"].as<int>();
    }
    if (model["decode_mode"]) {
      InferenceStatus status = ParseDecodeMode(model["decode_mode"].as<std::string>(),
                                               &out_config->model.decode_mode, out_error);
      if (status != INFERENCE_STATUS_OK) {
        return status;
      }
    }
    if (model["yaw_encoding"]) {
      InferenceStatus status = ParseYawEncoding(model["yaw_encoding"].as<std::string>(),
                                                &out_config->model.yaw_encoding,
                                                out_error);
      if (status != INFERENCE_STATUS_OK) {
        return status;
      }
    }
    if (model["bbox_code_size"]) {
      out_config->model.bbox_code_size = model["bbox_code_size"].as<int>();
    }
    if (model["norm_bbox"]) {
      out_config->model.norm_bbox = model["norm_bbox"].as<bool>();
    }
    if (model["has_velocity"]) {
      out_config->model.has_velocity = model["has_velocity"].as<bool>();
    }

    InferenceStatus status =
        LoadFloatArray(model["point_cloud_range"], "model.point_cloud_range",
                       &out_config->model.point_cloud_range, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = LoadFloatArray(model["post_center_range"], "model.post_center_range",
                            &out_config->model.post_center_range, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    if (model["voxel_size"]) {
      status = LoadFloatArray(model["voxel_size"], "model.voxel_size",
                              &out_config->model.voxel_size, out_error);
    } else {
      status = LoadFloatArray(model["pillar_size"], "model.pillar_size",
                              &out_config->model.voxel_size, out_error);
    }
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = LoadSparseShape(model["sparse_shape"], &out_config->model.sparse_shape,
                             out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    if (out_extras != nullptr) {
      out_extras->sparse_shape = out_config->model.sparse_shape;
    }
    status = LoadStringVector(model["label_names"], "model.label_names",
                              &out_config->model.label_names, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = LoadIntVector(model["task_num_classes"], "model.task_num_classes",
                           &out_config->model.task_num_classes, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = LoadNestedIntVector(model["task_label_ids"], "model.task_label_ids",
                                 &out_config->model.task_label_ids, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    NormalizeTaskLayout(&out_config->model);
  }

  const YAML::Node runtime = root["runtime"];
  if (runtime) {
    if (!runtime.IsMap()) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "runtime must be a YAML map");
    }
    if (runtime["max_points_per_voxel"]) {
      out_config->runtime.max_points_per_voxel = runtime["max_points_per_voxel"].as<int>();
    } else if (runtime["max_points_per_pillar"]) {
      out_config->runtime.max_points_per_voxel = runtime["max_points_per_pillar"].as<int>();
    }
    if (runtime["min_voxels"]) {
      out_config->runtime.min_voxels = runtime["min_voxels"].as<int>();
    } else if (runtime["min_pillars"]) {
      out_config->runtime.min_voxels = runtime["min_pillars"].as<int>();
    }
    if (runtime["max_voxels"]) {
      out_config->runtime.max_voxels = runtime["max_voxels"].as<int>();
    } else if (runtime["max_pillars"]) {
      out_config->runtime.max_voxels = runtime["max_pillars"].as<int>();
    }
    if (runtime["max_points"]) {
      out_config->runtime.max_points = runtime["max_points"].as<int>();
    }
    if (runtime["frame_num"]) {
      out_config->runtime.frame_num = runtime["frame_num"].as<int>();
    }
    if (runtime["bbox_max_num"]) {
      out_config->runtime.bbox_max_num = runtime["bbox_max_num"].as<int>();
    }
    if (runtime["score_threshold"]) {
      out_config->runtime.score_threshold = runtime["score_threshold"].as<float>();
    }
  }

  const YAML::Node postprocess = root["postprocess"];
  const InferenceStatus status = LoadPostprocessConfig(postprocess, &out_config->postprocess, out_error);
  if (status != INFERENCE_STATUS_OK) return status;
  if (postprocess && postprocess["label_score_thresh"] && out_extras)
    out_extras->has_label_score_thresh = true;

  return INFERENCE_STATUS_OK;
}

InferenceStatus LoadPackageConfigFromPath(const std::filesystem::path& config_path,
                                          const std::filesystem::path& override_path,
                                          PackageConfig* out_config,
                                          PackageConfigExtras* out_extras,
                                          std::string* out_error) {
  if (out_config == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, "out_config is null");
  }

  const auto load_yaml = [&](const std::filesystem::path& path,
                             YAML::Node* out_node) -> InferenceStatus {
    try {
      *out_node = YAML::LoadFile(path.string());
      return INFERENCE_STATUS_OK;
    } catch (const YAML::BadFile&) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "Failed to open package config: " + path.string());
    } catch (const YAML::Exception& ex) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "Failed to parse package config " + path.string() + ": " + ex.what());
    }
  };

  PackageConfig config;
  PackageConfigExtras extras;
  YAML::Node base_root;
  InferenceStatus status = load_yaml(config_path, &base_root);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  try {
    InferenceStatus status =
        ApplyPackageConfigNode(base_root, config_path, &config, &extras, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }

    if (!override_path.empty()) {
      YAML::Node override_root;
      status = load_yaml(override_path, &override_root);
      if (status != INFERENCE_STATUS_OK) {
        return status;
      }
      status = ApplyPackageConfigNode(override_root, override_path, &config, &extras, out_error);
      if (status != INFERENCE_STATUS_OK) {
        return status;
      }
    }

    NormalizeTaskLayout(&config.model);
    if (config.model.decode_mode == DecodeMode::kCenterPoint &&
        config.model.label_names.size() != config.postprocess.label_score_thresh.size() &&
        !extras.has_label_score_thresh) {
      // The built-in threshold defaults target the legacy three-class model.
      // Do not apply only a prefix of them to a different CenterPoint label set.
      config.postprocess.label_score_thresh.clear();
    }
    status = ValidateExtendedPackageConfig(&config, extras, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = ValidatePackageConfig(config, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    if (out_extras != nullptr) {
      *out_extras = std::move(extras);
    }
    *out_config = std::move(config);
    return INFERENCE_STATUS_OK;
  } catch (const YAML::Exception& ex) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Failed to parse package config: " + std::string(ex.what()));
  }
}

InferenceConfig ToCConfig(const PackageConfig& config) {
  InferenceConfig c_config{};
  c_config.end2end_file = config.model.engine_path.c_str();
  std::copy(config.model.point_cloud_range.begin(), config.model.point_cloud_range.end(),
            c_config.point_cloud_range);
  std::copy(config.model.post_center_range.begin(), config.model.post_center_range.end(),
            c_config.post_center_range);
  std::copy(config.model.voxel_size.begin(), config.model.voxel_size.end(),
            c_config.voxel_size);
  c_config.max_points_per_voxel = config.runtime.max_points_per_voxel;
  c_config.min_voxels = config.runtime.min_voxels;
  c_config.max_voxels = config.runtime.max_voxels;
  c_config.max_points = config.runtime.max_points;
  c_config.frame_num = config.runtime.frame_num;
  c_config.num_point_features = config.model.num_point_features;
  c_config.bbox_max_num = config.runtime.bbox_max_num;
  c_config.out_size_factor = config.model.out_size_factor;
  c_config.score_threshold = config.runtime.score_threshold;
  return c_config;
}

inference_internal::RuntimeConfig ToInternalRuntimeConfig(
    const PackageConfig& config,
    const PackageConfigExtras* extras = nullptr) {
  inference_internal::RuntimeConfig runtime;
  runtime.end2end_file = config.model.engine_path;
  runtime.head_engine_file = config.model.engine_path;
  runtime.point_cloud_range = config.model.point_cloud_range;
  runtime.post_center_range = config.model.post_center_range;
  runtime.voxel_size = config.model.voxel_size;
  runtime.max_points_per_voxel = config.runtime.max_points_per_voxel;
  runtime.min_voxels = config.runtime.min_voxels;
  runtime.max_voxels = config.runtime.max_voxels;
  runtime.max_points = config.runtime.max_points;
  runtime.frame_num = config.runtime.frame_num;
  runtime.num_point_features = config.model.num_point_features;
  runtime.bbox_max_num = config.runtime.bbox_max_num;
  runtime.out_size_factor = config.model.out_size_factor;
  runtime.score_threshold = config.runtime.score_threshold;
  runtime.decode_mode = config.model.decode_mode;
  runtime.yaw_encoding = config.model.yaw_encoding;
  runtime.bbox_code_size = config.model.bbox_code_size;
  runtime.norm_bbox = config.model.norm_bbox;
  runtime.has_velocity = config.model.has_velocity;
  runtime.task_num_classes = config.model.task_num_classes;
  runtime.task_label_ids = config.model.task_label_ids;
  runtime.label_count = static_cast<int>(config.model.label_names.size());
  if (extras != nullptr) {
    runtime.runtime_backend = extras->runtime_backend;
    runtime.scn_onnx_file = extras->scn_onnx_path;
    runtime.head_engine_file = extras->head_engine_path;
    runtime.sparse_shape = extras->sparse_shape;
  } else {
    inference_internal::RuntimeBackend backend =
        inference_internal::RuntimeBackend::kPillarEnd2End;
    if (ParseRuntimeBackend(config.model.runtime_backend, &backend, nullptr) ==
        INFERENCE_STATUS_OK) {
      runtime.runtime_backend = backend;
    }
    runtime.scn_onnx_file = config.model.scn_onnx_path;
    runtime.head_engine_file =
        config.model.head_engine_path.empty() ? config.model.engine_path
                                              : config.model.head_engine_path;
    runtime.sparse_shape = config.model.sparse_shape;
  }
  return runtime;
}

inference_internal::PostprocessConfig ToInternalPostprocessConfig(
    const PostprocessConfig& config) {
  inference_internal::PostprocessConfig internal;
  internal.nms_type = config.nms_type;
  internal.nms_thresh = config.nms_thresh;
  internal.nms_pre_maxsize = config.nms_pre_maxsize;
  internal.nms_post_maxsize = config.nms_post_maxsize;
  internal.final_score_threshold = config.final_score_threshold;
  internal.self_car_box = config.self_car_box;
  internal.self_car_nms_thresh = config.self_car_nms_thresh;
  internal.circle_nms_min_radius = config.circle_nms_min_radius;
  internal.label_score_thresh = config.label_score_thresh;
  internal.nms_label_groups.reserve(config.nms_label_groups.size());
  for (const PostprocessConfig::NmsLabelGroup& group : config.nms_label_groups) {
    inference_internal::PostprocessConfig::NmsLabelGroup internal_group;
    internal_group.labels = group.labels;
    internal_group.nms_thresh = group.nms_thresh;
    internal_group.nms_pre_maxsize = group.nms_pre_maxsize;
    internal_group.nms_post_maxsize = group.nms_post_maxsize;
    internal_group.circle_nms_min_radius = group.circle_nms_min_radius;
    internal.nms_label_groups.push_back(std::move(internal_group));
  }
  return internal;
}

InferencePostprocessConfig ToCPostprocessConfig(const PostprocessConfig& config) {
  InferencePostprocessConfig c_config{};
  c_config.nms_type = static_cast<int>(config.nms_type);
  c_config.nms_thresh = config.nms_thresh;
  c_config.nms_pre_maxsize = config.nms_pre_maxsize;
  c_config.nms_post_maxsize = config.nms_post_maxsize;
  c_config.final_score_threshold = config.final_score_threshold;
  const size_t label_count =
      std::min(config.label_score_thresh.size(),
               sizeof(c_config.label_score_thresh) / sizeof(c_config.label_score_thresh[0]));
  std::copy_n(config.label_score_thresh.begin(), static_cast<long>(label_count),
              c_config.label_score_thresh);
  const size_t radius_count =
      std::min(config.circle_nms_min_radius.size(),
               sizeof(c_config.circle_nms_min_radius) /
                   sizeof(c_config.circle_nms_min_radius[0]));
  std::copy_n(config.circle_nms_min_radius.begin(), static_cast<long>(radius_count),
              c_config.circle_nms_min_radius);
  return c_config;
}

}  // namespace

LidarInferenceRuntime::~LidarInferenceRuntime() {
  runtime_.reset();
}

InferenceStatus LidarInferenceRuntime::LoadPackageConfigFile(const std::string& config_path,
                                                             PackageConfig* out_config,
                                                             std::string* out_error,
                                                             const std::string& override_path) {
  if (config_path.empty()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "config_path must not be empty");
  }
  return LoadPackageConfigFromPath(std::filesystem::path(config_path),
                                   override_path.empty() ? std::filesystem::path()
                                                         : std::filesystem::path(override_path),
                                   out_config, nullptr, out_error);
}

InferenceStatus LidarInferenceRuntime::LoadPackageConfig(const std::string& model_dir,
                                                         PackageConfig* out_config,
                                                         std::string* out_error,
                                                         const std::string& override_path) {
  if (model_dir.empty()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "model_dir must not be empty");
  }
  const std::filesystem::path config_path =
      std::filesystem::path(model_dir) / "model.yaml";
  return LoadPackageConfigFromPath(config_path,
                                   override_path.empty() ? std::filesystem::path()
                                                         : std::filesystem::path(override_path),
                                   out_config, nullptr, out_error);
}

InferenceStatus LidarInferenceRuntime::Create(
    const PackageConfig& config,
    std::unique_ptr<LidarInferenceRuntime>* out_runtime,
    std::string* out_error) {
  if (out_runtime == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, "out_runtime is null");
  }
  *out_runtime = nullptr;

  InferenceStatus status = ValidatePackageConfig(config, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  if (!std::filesystem::exists(config.model.engine_path)) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "TensorRT engine does not exist: " + config.model.engine_path);
  }

  std::unique_ptr<LidarInferenceRuntime> runtime(new LidarInferenceRuntime());
  runtime->config_ = config;

  runtime->runtime_ = std::make_shared<inference_internal::PillarRuntime>(
      ToInternalRuntimeConfig(runtime->config_));
  status = runtime->runtime_->Initialize();
  if (status != INFERENCE_STATUS_OK) {
    runtime->last_error_ = runtime->runtime_->last_error();
    return SetError(out_error, status, runtime->last_error_);
  }
  runtime->output_spec_ = runtime->runtime_->output_spec();

  runtime->cls_preds_.assign(runtime->output_spec_.cls_count, 0.0f);
  runtime->box_preds_.assign(runtime->output_spec_.box_count, 0.0f);
  runtime->label_preds_.assign(runtime->output_spec_.label_count, 0);

  runtime->response_.cls_preds = runtime->cls_preds_.data();
  runtime->response_.cls_capacity = runtime->cls_preds_.size();
  runtime->response_.box_preds = runtime->box_preds_.data();
  runtime->response_.box_capacity = runtime->box_preds_.size();
  runtime->response_.label_preds = runtime->label_preds_.data();
  runtime->response_.label_capacity = runtime->label_preds_.size();
  runtime->response_.freespace = nullptr;
  runtime->response_.freespace_capacity = 0;
  runtime->response_.cls_count = 0;
  runtime->response_.box_count = 0;
  runtime->response_.label_count = 0;
  runtime->response_.freespace_count = 0;

  *out_runtime = std::move(runtime);
  return INFERENCE_STATUS_OK;
}

InferenceStatus LidarInferenceRuntime::CreateFromModelDir(
    const std::string& model_dir,
    std::unique_ptr<LidarInferenceRuntime>* out_runtime,
    std::string* out_error,
    const std::string& override_path) {
  PackageConfig config;
  PackageConfigExtras extras;
  const std::filesystem::path config_path =
      std::filesystem::path(model_dir) / "model.yaml";
  InferenceStatus status = LoadPackageConfigFromPath(
      config_path,
      override_path.empty() ? std::filesystem::path() : std::filesystem::path(override_path),
      &config, &extras, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  if (out_runtime == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, "out_runtime is null");
  }
  *out_runtime = nullptr;
  if (!std::filesystem::exists(config.model.engine_path)) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "TensorRT engine does not exist: " + config.model.engine_path);
  }
  std::unique_ptr<LidarInferenceRuntime> runtime(new LidarInferenceRuntime());
  runtime->config_ = config;
  runtime->runtime_ = std::make_shared<inference_internal::PillarRuntime>(
      ToInternalRuntimeConfig(runtime->config_, &extras));
  status = runtime->runtime_->Initialize();
  if (status != INFERENCE_STATUS_OK) {
    runtime->last_error_ = runtime->runtime_->last_error();
    return SetError(out_error, status, runtime->last_error_);
  }
  runtime->output_spec_ = runtime->runtime_->output_spec();

  runtime->cls_preds_.assign(runtime->output_spec_.cls_count, 0.0f);
  runtime->box_preds_.assign(runtime->output_spec_.box_count, 0.0f);
  runtime->label_preds_.assign(runtime->output_spec_.label_count, 0);

  runtime->response_.cls_preds = runtime->cls_preds_.data();
  runtime->response_.cls_capacity = runtime->cls_preds_.size();
  runtime->response_.box_preds = runtime->box_preds_.data();
  runtime->response_.box_capacity = runtime->box_preds_.size();
  runtime->response_.label_preds = runtime->label_preds_.data();
  runtime->response_.label_capacity = runtime->label_preds_.size();
  runtime->response_.freespace = nullptr;
  runtime->response_.freespace_capacity = 0;
  runtime->response_.cls_count = 0;
  runtime->response_.box_count = 0;
  runtime->response_.label_count = 0;
  runtime->response_.freespace_count = 0;

  *out_runtime = std::move(runtime);
  return INFERENCE_STATUS_OK;
}

InferenceStatus LidarInferenceRuntime::Infer(const PointCloudView& cloud,
                                             std::vector<Detection>* out_detections) {
  return InferInternal(cloud, &config_.postprocess, out_detections, true);
}

InferenceStatus LidarInferenceRuntime::InferDecoded(const PointCloudView& cloud,
                                                    std::vector<Detection>* out_detections) {
  return InferInternal(cloud, nullptr, out_detections, false);
}

InferenceStatus LidarInferenceRuntime::InferWithPostprocess(
    const PointCloudView& cloud,
    const PostprocessConfig& postprocess,
    std::vector<Detection>* out_detections) {
  InferenceStatus status = ValidatePostprocessConfig(config_.model, postprocess, &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  return InferInternal(cloud, &postprocess, out_detections, true);
}

const char* LidarInferenceRuntime::StoreLabelName(const std::string& label_name) {
  return label_name_pool_.insert(label_name).first->c_str();
}

InferenceStatus LidarInferenceRuntime::InferInternal(
    const PointCloudView& cloud,
    const PostprocessConfig* postprocess,
    std::vector<Detection>* out_detections,
    bool use_nms,
    inference_internal::StageTimings* timings,
    inference_internal::DebugArtifacts* debug) {
  if (runtime_ == nullptr) {
    return SetError(&last_error_, INFERENCE_STATUS_NOT_READY, "runtime is not initialized");
  }
  if (out_detections == nullptr) {
    return SetError(&last_error_, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "out_detections is null");
  }

  InferenceRequest request{};
  request.points = cloud.points;
  request.point_count = cloud.point_count;
  request.motion_4x4 = cloud.motion_4x4;
  request.enable_runtime = cloud.enable_runtime;

  response_.cls_count = 0;
  response_.box_count = 0;
  response_.label_count = 0;
  response_.freespace_count = 0;

  InferenceStatus status = INFERENCE_STATUS_OK;
  inference_internal::PostprocessConfig internal_postprocess;
  if (use_nms) {
    internal_postprocess = ToInternalPostprocessConfig(
        postprocess == nullptr ? config_.postprocess : *postprocess);
    status = runtime_->ForwardNms(request, internal_postprocess, &response_, debug, timings);
  } else {
    status = runtime_->Forward(request, &response_, debug, timings);
  }
  last_error_ = runtime_->last_error();
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  out_detections->clear();
  out_detections->reserve(response_.cls_count);
  const PostprocessConfig& label_postprocess =
      (use_nms && postprocess != nullptr) ? *postprocess : config_.postprocess;
  for (size_t i = 0; i < response_.cls_count; ++i) {
    int mapped_label = label_preds_[i];
    const PostprocessConfig* names_postprocess = nullptr;
    if (use_nms) {
      if (!MapDetectionLabel(label_postprocess, label_preds_[i], &mapped_label)) {
        continue;
      }
      names_postprocess = &label_postprocess;
    }
    Detection detection;
    detection.score = cls_preds_[i];
    detection.label = mapped_label;
    const std::string* label_name = nullptr;
    if (names_postprocess == nullptr) {
      if (detection.label >= 0 &&
          detection.label < static_cast<int>(config_.model.label_names.size())) {
        label_name = &config_.model.label_names[static_cast<size_t>(detection.label)];
      }
    } else {
      label_name = LabelNameForLabel(config_.model.label_names, *names_postprocess, detection.label);
    }
    if (label_name != nullptr) {
      detection.label_name = StoreLabelName(*label_name);
    }
    const size_t box_offset = i * kBoxDimension;
    std::copy(box_preds_.begin() + static_cast<long>(box_offset),
              box_preds_.begin() + static_cast<long>(box_offset + kBoxDimension),
              detection.box.begin());
    out_detections->push_back(detection);
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus LidarInferenceRuntime::Reset() {
  if (runtime_ == nullptr) {
    return SetError(&last_error_, INFERENCE_STATUS_NOT_READY, "runtime is not initialized");
  }
  runtime_->Reset();
  last_error_.clear();
  return INFERENCE_STATUS_OK;
}

}  // namespace inference

InferenceStatus inference_internal::RuntimeDiagnostics::Lidar(
    inference::LidarInferenceRuntime& runtime, const inference::PointCloudView& cloud,
    std::vector<inference::Detection>* detections, StageTimings* timings, DebugArtifacts* debug) {
  return runtime.InferInternal(cloud, &runtime.config_.postprocess, detections, true, timings, debug);
}
