#include "inference_cpp.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cctype>
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

std::string NormalizeNmsType(const std::string& value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return normalized;
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

InferenceStatus ParseNmsType(const std::string& value,
                             InferenceNmsType* out_type,
                             std::string* out_error) {
  if (out_type == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, "out_type is null");
  }

  const std::string normalized = NormalizeNmsType(value);
  if (normalized == "circle") {
    *out_type = INFERENCE_NMS_CIRCLE;
    return INFERENCE_STATUS_OK;
  }
  if (normalized == "iou" || normalized == "iou_bev" || normalized == "ioubev") {
    *out_type = INFERENCE_NMS_IOU_BEV;
    return INFERENCE_STATUS_OK;
  }
  if (normalized == "rotate" || normalized == "rotated") {
    *out_type = INFERENCE_NMS_IOU_BEV;
    return INFERENCE_STATUS_OK;
  }
  return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                  "Unsupported postprocess.nms_type: " + value);
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

InferenceStatus LoadThresholdVector(const YAML::Node& node,
                                    const char* field_name,
                                    std::vector<float>* out_values,
                                    std::string* out_error) {
  if (!node) {
    return INFERENCE_STATUS_OK;
  }
  if (node.IsSequence()) {
    out_values->clear();
    out_values->reserve(node.size());
    for (size_t i = 0; i < node.size(); ++i) {
      out_values->push_back(node[i].as<float>());
    }
    return INFERENCE_STATUS_OK;
  }
  if (!node.IsMap()) {
    std::ostringstream oss;
    oss << field_name << " must be either a sequence or a map";
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
  }

  if (out_values->empty()) {
    size_t max_index = 0;
    for (auto it = node.begin(); it != node.end(); ++it) {
      const size_t index = static_cast<size_t>(std::stoul(it->first.as<std::string>()));
      max_index = std::max(max_index, index);
    }
    out_values->assign(max_index, 0.0f);
  }

  for (size_t i = 0; i < out_values->size(); ++i) {
    const std::string key = std::to_string(i + 1U);
    if (node[key]) {
      (*out_values)[i] = node[key].as<float>();
    }
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
  } else if (model->task_num_classes.empty()) {
    model->task_num_classes = {1, 1, 1};
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
    if (config.model.bbox_code_size < static_cast<int>(kBoxDimension)) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "model.bbox_code_size must be at least 7 in CenterPoint mode");
    }
  }
  for (const float value : config.postprocess.label_score_thresh) {
    if (value < 0.0f) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "postprocess.label_score_thresh must be non-negative");
    }
  }
  for (const float value : config.postprocess.circle_nms_min_radius) {
    if (value < 0.0f) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "postprocess.circle_nms_min_radius must be non-negative");
    }
  }
  if (config.postprocess.nms_pre_maxsize <= 0 || config.postprocess.nms_post_maxsize <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "postprocess.nms_pre_maxsize and nms_post_maxsize must be positive");
  }
  if (config.postprocess.nms_thresh < 0.0f || config.postprocess.final_score_threshold < 0.0f) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "postprocess thresholds must be non-negative");
  }
  if (config.model.decode_mode == DecodeMode::kCenterPoint &&
      config.postprocess.nms_type == INFERENCE_NMS_CIRCLE &&
      config.postprocess.circle_nms_min_radius.size() != config.model.task_num_classes.size()) {
    std::ostringstream oss;
    oss << "postprocess.circle_nms_min_radius must have one value per task in CenterPoint mode, "
        << "expected " << config.model.task_num_classes.size()
        << " got " << config.postprocess.circle_nms_min_radius.size();
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus ApplyPackageConfigNode(const YAML::Node& root,
                                       const std::filesystem::path& config_path,
                                       PackageConfig* out_config,
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
    status = LoadFloatArray(model["pillar_size"], "model.pillar_size",
                            &out_config->model.pillar_size, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
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
    NormalizeTaskLayout(&out_config->model);
  }

  const YAML::Node runtime = root["runtime"];
  if (runtime) {
    if (!runtime.IsMap()) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "runtime must be a YAML map");
    }
    if (runtime["max_points_per_pillar"]) {
      out_config->runtime.max_points_per_pillar = runtime["max_points_per_pillar"].as<int>();
    }
    if (runtime["min_pillars"]) {
      out_config->runtime.min_pillars = runtime["min_pillars"].as<int>();
    }
    if (runtime["max_pillars"]) {
      out_config->runtime.max_pillars = runtime["max_pillars"].as<int>();
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
  if (postprocess) {
    if (!postprocess.IsMap()) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "postprocess must be a YAML map");
    }
    if (postprocess["nms_type"]) {
      InferenceStatus status =
          ParseNmsType(postprocess["nms_type"].as<std::string>(),
                       &out_config->postprocess.nms_type, out_error);
      if (status != INFERENCE_STATUS_OK) {
        return status;
      }
    }
    if (postprocess["nms_thresh"]) {
      out_config->postprocess.nms_thresh = postprocess["nms_thresh"].as<float>();
    }
    if (postprocess["nms_pre_maxsize"]) {
      out_config->postprocess.nms_pre_maxsize = postprocess["nms_pre_maxsize"].as<int>();
    }
    if (postprocess["nms_post_maxsize"]) {
      out_config->postprocess.nms_post_maxsize = postprocess["nms_post_maxsize"].as<int>();
    }
    if (postprocess["final_score_threshold"]) {
      out_config->postprocess.final_score_threshold =
          postprocess["final_score_threshold"].as<float>();
    }

    InferenceStatus status =
        LoadThresholdVector(postprocess["label_score_thresh"],
                            "postprocess.label_score_thresh",
                            &out_config->postprocess.label_score_thresh, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = LoadThresholdVector(postprocess["circle_nms_min_radius"],
                                 "postprocess.circle_nms_min_radius",
                                 &out_config->postprocess.circle_nms_min_radius, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
  }

  return INFERENCE_STATUS_OK;
}

InferenceStatus LoadPackageConfigFromPath(const std::filesystem::path& config_path,
                                          const std::filesystem::path& override_path,
                                          PackageConfig* out_config,
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
  YAML::Node base_root;
  InferenceStatus status = load_yaml(config_path, &base_root);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  try {
    InferenceStatus status = ApplyPackageConfigNode(base_root, config_path, &config, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }

    if (!override_path.empty()) {
      YAML::Node override_root;
      status = load_yaml(override_path, &override_root);
      if (status != INFERENCE_STATUS_OK) {
        return status;
      }
      status = ApplyPackageConfigNode(override_root, override_path, &config, out_error);
      if (status != INFERENCE_STATUS_OK) {
        return status;
      }
    }

    NormalizeTaskLayout(&config.model);
    status = ValidatePackageConfig(config, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
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
  std::copy(config.model.pillar_size.begin(), config.model.pillar_size.end(),
            c_config.pillar_size);
  c_config.max_points_per_pillar = config.runtime.max_points_per_pillar;
  c_config.min_pillars = config.runtime.min_pillars;
  c_config.max_pillars = config.runtime.max_pillars;
  c_config.max_points = config.runtime.max_points;
  c_config.frame_num = config.runtime.frame_num;
  c_config.num_point_features = config.model.num_point_features;
  c_config.bbox_max_num = config.runtime.bbox_max_num;
  c_config.out_size_factor = config.model.out_size_factor;
  c_config.score_threshold = config.runtime.score_threshold;
  return c_config;
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

const char* LabelNameForLabel(const PackageConfig& config, int label) {
  if (label < 0 || label >= static_cast<int>(config.model.label_names.size())) {
    return nullptr;
  }
  return config.model.label_names[static_cast<size_t>(label)].c_str();
}

}  // namespace

LidarInferenceRuntime::~LidarInferenceRuntime() {
  inference_destroy(handle_);
  handle_ = nullptr;
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
                                   out_config, out_error);
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
                                   out_config, out_error);
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
  runtime->default_postprocess_ = ToCPostprocessConfig(config.postprocess);

  InferenceConfig c_config = ToCConfig(runtime->config_);
  InferenceHandle* handle = nullptr;
  status = inference_create(&c_config, &handle);
  if (status != INFERENCE_STATUS_OK) {
    return SetError(out_error, status, "inference_create failed");
  }
  runtime->handle_ = handle;
  runtime->last_error_ = inference_last_error(runtime->handle_);

  status = inference_get_output_spec(runtime->handle_, &runtime->output_spec_);
  if (status != INFERENCE_STATUS_OK) {
    runtime->last_error_ = inference_last_error(runtime->handle_);
    return SetError(out_error, status, runtime->last_error_);
  }

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
  InferenceStatus status = LoadPackageConfig(model_dir, &config, out_error, override_path);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  return Create(config, out_runtime, out_error);
}

InferenceStatus LidarInferenceRuntime::Infer(const PointCloudView& cloud,
                                             std::vector<Detection>* out_detections) {
  return InferInternal(cloud, &default_postprocess_, out_detections, true);
}

InferenceStatus LidarInferenceRuntime::InferDecoded(const PointCloudView& cloud,
                                                    std::vector<Detection>* out_detections) {
  return InferInternal(cloud, nullptr, out_detections, false);
}

InferenceStatus LidarInferenceRuntime::InferWithPostprocess(
    const PointCloudView& cloud,
    const PostprocessConfig& postprocess,
    std::vector<Detection>* out_detections) {
  const InferencePostprocessConfig c_postprocess = ToCPostprocessConfig(postprocess);
  return InferInternal(cloud, &c_postprocess, out_detections, true);
}

InferenceStatus LidarInferenceRuntime::InferInternal(
    const PointCloudView& cloud,
    const InferencePostprocessConfig* postprocess,
    std::vector<Detection>* out_detections,
    bool use_nms) {
  if (handle_ == nullptr) {
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

  const InferenceStatus status =
      use_nms ? inference_forward_nms(handle_, &request, postprocess, &response_)
              : inference_forward(handle_, &request, &response_);
  last_error_ = inference_last_error(handle_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  out_detections->clear();
  out_detections->reserve(response_.cls_count);
  for (size_t i = 0; i < response_.cls_count; ++i) {
    Detection detection;
    detection.score = cls_preds_[i];
    detection.label = label_preds_[i];
    detection.label_name = LabelNameForLabel(config_, detection.label);
    const size_t box_offset = i * kBoxDimension;
    std::copy(box_preds_.begin() + static_cast<long>(box_offset),
              box_preds_.begin() + static_cast<long>(box_offset + kBoxDimension),
              detection.box.begin());
    out_detections->push_back(detection);
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus LidarInferenceRuntime::Reset() {
  if (handle_ == nullptr) {
    return SetError(&last_error_, INFERENCE_STATUS_NOT_READY, "runtime is not initialized");
  }
  const InferenceStatus status = inference_reset(handle_);
  last_error_ = inference_last_error(handle_);
  return status;
}

}  // namespace inference
