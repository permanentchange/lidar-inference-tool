#include "postprocess_config.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

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

std::string NormalizeNmsType(const std::string& value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
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
  if (normalized == "iou_min" || normalized == "iou_min_bev" || normalized == "iouminbev") {
    *out_type = INFERENCE_NMS_IOU_MIN_BEV;
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

InferenceStatus LoadNmsLabelGroups(const YAML::Node& node,
                                   const char* field_name,
                                   std::vector<PostprocessConfig::NmsLabelGroup>* out_groups,
                                   std::string* out_error) {
  if (!node) {
    return INFERENCE_STATUS_OK;
  }
  if (out_groups == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    std::string(field_name) + " output is null");
  }
  if (!node.IsSequence() || node.size() == 0) {
    std::ostringstream oss;
    oss << field_name << " must be a non-empty sequence of maps";
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
  }

  out_groups->clear();
  out_groups->reserve(node.size());
  for (size_t i = 0; i < node.size(); ++i) {
    const YAML::Node group_node = node[i];
    if (!group_node.IsMap()) {
      std::ostringstream oss;
      oss << field_name << "[" << i << "] must be a YAML map";
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
    }
    PostprocessConfig::NmsLabelGroup group;
    InferenceStatus status =
        LoadIntVector(group_node["labels"], "postprocess.nms_label_groups[].labels",
                      &group.labels, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    if (group_node["nms_thresh"]) {
      group.nms_thresh = group_node["nms_thresh"].as<float>();
    }
    if (group_node["nms_pre_maxsize"]) {
      group.nms_pre_maxsize = group_node["nms_pre_maxsize"].as<int>();
    }
    if (group_node["nms_post_maxsize"]) {
      group.nms_post_maxsize = group_node["nms_post_maxsize"].as<int>();
    }
    if (group_node["circle_nms_min_radius"]) {
      group.circle_nms_min_radius = group_node["circle_nms_min_radius"].as<float>();
    }
    out_groups->push_back(std::move(group));
  }
  return INFERENCE_STATUS_OK;
}

}  // namespace

InferenceStatus LoadPostprocessConfig(const YAML::Node& postprocess,
    PostprocessConfig* out_postprocess, std::string* out_error) {
  if (!out_postprocess) return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, "postprocess output is null");
  try {
  if (postprocess) {
    if (!postprocess.IsMap()) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "postprocess must be a YAML map");
    }
    if (postprocess["self_car_box"]) {
      std::array<float, kBoxDimension> self_car_box{};
      InferenceStatus status =
          LoadFloatArray(postprocess["self_car_box"], "postprocess.self_car_box",
                         &self_car_box, out_error);
      if (status != INFERENCE_STATUS_OK) {
        return status;
      }
      out_postprocess->self_car_box = self_car_box;
    }
    if (postprocess["self_car_nms_thresh"]) {
      out_postprocess->self_car_nms_thresh =
          postprocess["self_car_nms_thresh"].as<float>();
    }
    if (postprocess["nms_type"]) {
      InferenceStatus status =
          ParseNmsType(postprocess["nms_type"].as<std::string>(),
                       &out_postprocess->nms_type, out_error);
      if (status != INFERENCE_STATUS_OK) {
        return status;
      }
    }
    if (postprocess["nms_thresh"]) {
      out_postprocess->nms_thresh = postprocess["nms_thresh"].as<float>();
    }
    if (postprocess["nms_pre_maxsize"]) {
      out_postprocess->nms_pre_maxsize = postprocess["nms_pre_maxsize"].as<int>();
    }
    if (postprocess["nms_post_maxsize"]) {
      out_postprocess->nms_post_maxsize = postprocess["nms_post_maxsize"].as<int>();
    }
    if (postprocess["final_score_threshold"]) {
      out_postprocess->final_score_threshold =
          postprocess["final_score_threshold"].as<float>();
    }

    InferenceStatus status =
        LoadThresholdVector(postprocess["label_score_thresh"],
                            "postprocess.label_score_thresh",
                            &out_postprocess->label_score_thresh, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = LoadThresholdVector(postprocess["circle_nms_min_radius"],
                                 "postprocess.circle_nms_min_radius",
                                 &out_postprocess->circle_nms_min_radius, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = LoadNmsLabelGroups(postprocess["nms_label_groups"],
                                "postprocess.nms_label_groups",
                                &out_postprocess->nms_label_groups, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = LoadIntVector(postprocess["label_mapping"], "postprocess.label_mapping",
                           &out_postprocess->label_mapping, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = LoadStringVector(postprocess["mapped_label_names"],
                              "postprocess.mapped_label_names",
                              &out_postprocess->mapped_label_names, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
  }

  } catch (const YAML::Exception& error) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, error.what());
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus ValidatePostprocessConfig(const ModelConfig& model,
                                          const PostprocessConfig& postprocess,
                                          std::string* out_error) {
  const int label_count = static_cast<int>(model.label_names.size());
  if (postprocess.nms_type != INFERENCE_NMS_IOU_BEV &&
      postprocess.nms_type != INFERENCE_NMS_IOU_MIN_BEV &&
      postprocess.nms_type != INFERENCE_NMS_CIRCLE) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "postprocess.nms_type is invalid");
  }
  const bool uses_label_score_thresholds =
      !postprocess.nms_label_groups.empty() ||
      model.decode_mode == DecodeMode::kCenterPoint ||
      postprocess.nms_type != INFERENCE_NMS_CIRCLE;
  if (uses_label_score_thresholds && !postprocess.label_score_thresh.empty() &&
      postprocess.label_score_thresh.size() < static_cast<size_t>(label_count)) {
    std::ostringstream oss;
    oss << "postprocess.label_score_thresh must contain at least " << label_count
        << " values when configured, got " << postprocess.label_score_thresh.size();
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
  }
  if (postprocess.self_car_box.has_value() !=
      postprocess.self_car_nms_thresh.has_value()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "postprocess.self_car_box and self_car_nms_thresh must be configured together");
  }
  if (postprocess.self_car_box.has_value()) {
    for (const float value : *postprocess.self_car_box) {
      if (!std::isfinite(value)) {
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                        "postprocess.self_car_box must contain finite values");
      }
    }
    if ((*postprocess.self_car_box)[3] <= 0.0f ||
        (*postprocess.self_car_box)[4] <= 0.0f ||
        (*postprocess.self_car_box)[5] <= 0.0f) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "postprocess.self_car_box dimensions must be positive");
    }
    if (!std::isfinite(*postprocess.self_car_nms_thresh) ||
        *postprocess.self_car_nms_thresh < 0.0f) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "postprocess.self_car_nms_thresh must be finite and non-negative");
    }
    if (postprocess.nms_type != INFERENCE_NMS_IOU_BEV &&
        postprocess.nms_type != INFERENCE_NMS_IOU_MIN_BEV) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "postprocess self-car NMS only supports iou_bev and iou_min_bev");
    }
  }
  for (const float value : postprocess.label_score_thresh) {
    if (!std::isfinite(value) || value < 0.0f) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "postprocess.label_score_thresh must be finite and non-negative");
    }
  }
  for (const float value : postprocess.circle_nms_min_radius) {
    if (!std::isfinite(value) || value < 0.0f) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "postprocess.circle_nms_min_radius must be finite and non-negative");
    }
  }
  if (postprocess.nms_pre_maxsize <= 0 || postprocess.nms_post_maxsize <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "postprocess.nms_pre_maxsize and nms_post_maxsize must be positive");
  }
  if (!std::isfinite(postprocess.nms_thresh) ||
      !std::isfinite(postprocess.final_score_threshold) ||
      postprocess.nms_thresh < 0.0f || postprocess.final_score_threshold < 0.0f) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "postprocess thresholds must be finite and non-negative");
  }

  if (!postprocess.nms_label_groups.empty()) {
    std::vector<int> seen(static_cast<size_t>(label_count), 0);
    for (size_t group_index = 0; group_index < postprocess.nms_label_groups.size();
         ++group_index) {
      const PostprocessConfig::NmsLabelGroup& group =
          postprocess.nms_label_groups[group_index];
      if (group.labels.empty()) {
        std::ostringstream oss;
        oss << "postprocess.nms_label_groups[" << group_index << "].labels must not be empty";
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
      }
      for (const int label : group.labels) {
        if (label < 0 || label >= label_count) {
          std::ostringstream oss;
          oss << "postprocess.nms_label_groups[" << group_index
              << "].labels contains out-of-range label " << label
              << ", valid range is [0, " << (label_count - 1) << "]";
          return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
        }
        if (seen[static_cast<size_t>(label)] != 0) {
          std::ostringstream oss;
          oss << "postprocess.nms_label_groups label " << label
              << " appears in more than one group";
          return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
        }
        seen[static_cast<size_t>(label)] = 1;
      }

      if (postprocess.nms_type == INFERENCE_NMS_CIRCLE) {
        if (!std::isfinite(group.circle_nms_min_radius) ||
            group.circle_nms_min_radius < 0.0f || group.nms_post_maxsize <= 0) {
          std::ostringstream oss;
          oss << "circle postprocess.nms_label_groups[" << group_index
              << "] must set finite non-negative circle_nms_min_radius and positive "
                 "nms_post_maxsize";
          return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
        }
      } else {
        if (!std::isfinite(group.nms_thresh) || group.nms_thresh < 0.0f ||
            group.nms_pre_maxsize <= 0 || group.nms_post_maxsize <= 0) {
          std::ostringstream oss;
          oss << "iou postprocess.nms_label_groups[" << group_index
              << "] must set finite non-negative nms_thresh and positive "
                 "nms_pre_maxsize/nms_post_maxsize";
          return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
        }
      }
    }
    for (int label = 0; label < label_count; ++label) {
      if (seen[static_cast<size_t>(label)] == 0) {
        std::ostringstream oss;
        oss << "postprocess.nms_label_groups must cover every original label exactly once; "
            << "missing label " << label;
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
      }
    }
  } else if (model.decode_mode == DecodeMode::kCenterPoint &&
             postprocess.nms_type == INFERENCE_NMS_CIRCLE &&
             postprocess.circle_nms_min_radius.size() != model.task_num_classes.size()) {
    std::ostringstream oss;
    oss << "postprocess.circle_nms_min_radius must have one value per task in CenterPoint mode, "
        << "expected " << model.task_num_classes.size()
        << " got " << postprocess.circle_nms_min_radius.size();
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
  } else if (model.decode_mode != DecodeMode::kCenterPoint &&
             postprocess.nms_type == INFERENCE_NMS_CIRCLE &&
             postprocess.circle_nms_min_radius.size() < static_cast<size_t>(label_count)) {
    std::ostringstream oss;
    oss << "postprocess.circle_nms_min_radius must contain at least " << label_count
        << " values in legacy mode, got " << postprocess.circle_nms_min_radius.size();
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
  }

  if (!postprocess.label_mapping.empty()) {
    if (postprocess.label_mapping.size() != model.label_names.size()) {
      std::ostringstream oss;
      oss << "postprocess.label_mapping must have one entry per original label, expected "
          << model.label_names.size() << " got " << postprocess.label_mapping.size();
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
    }
    if (postprocess.mapped_label_names.empty()) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "postprocess.mapped_label_names must be set when label_mapping is set");
    }
    for (size_t i = 0; i < postprocess.label_mapping.size(); ++i) {
      const int mapped_label = postprocess.label_mapping[i];
      if (mapped_label == -1) {
        continue;
      }
      if (mapped_label < 0 ||
          mapped_label >= static_cast<int>(postprocess.mapped_label_names.size())) {
        std::ostringstream oss;
        oss << "postprocess.label_mapping[" << i << "] maps to invalid label "
            << mapped_label << ", mapped_label_names.size()="
            << postprocess.mapped_label_names.size();
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
      }
    }
  } else if (!postprocess.mapped_label_names.empty() &&
             postprocess.mapped_label_names.size() != model.label_names.size()) {
    std::ostringstream oss;
    oss << "postprocess.mapped_label_names without label_mapping must have one entry per "
        << "original label, expected " << model.label_names.size()
        << " got " << postprocess.mapped_label_names.size();
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
  }
  for (size_t i = 0; i < postprocess.mapped_label_names.size(); ++i) {
    if (postprocess.mapped_label_names[i].empty()) {
      std::ostringstream oss;
      oss << "postprocess.mapped_label_names[" << i << "] must not be empty";
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
    }
  }

  return INFERENCE_STATUS_OK;
}

const std::string* LabelNameForLabel(const std::vector<std::string>& original_names,
                                      const PostprocessConfig& postprocess,
                                      int label) {
  const std::vector<std::string>& label_names =
      postprocess.mapped_label_names.empty() ? original_names
                                             : postprocess.mapped_label_names;
  if (label < 0 || label >= static_cast<int>(label_names.size())) {
    return nullptr;
  }
  return &label_names[static_cast<size_t>(label)];
}

bool MapDetectionLabel(const PostprocessConfig& postprocess,
                       int original_label,
                       int* out_label) {
  if (out_label == nullptr) {
    return false;
  }
  if (postprocess.label_mapping.empty()) {
    *out_label = original_label;
    return true;
  }
  if (original_label < 0 ||
      original_label >= static_cast<int>(postprocess.label_mapping.size())) {
    return false;
  }
  const int mapped_label = postprocess.label_mapping[static_cast<size_t>(original_label)];
  if (mapped_label < 0) {
    return false;
  }
  *out_label = mapped_label;
  return true;
}

}  // namespace inference
