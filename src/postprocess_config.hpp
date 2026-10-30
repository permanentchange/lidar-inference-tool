#pragma once
#include "inference_cpp.hpp"
#include <yaml-cpp/yaml.h>

namespace inference {
// Thresholds/groups index original labels; output mapping runs after NMS.
InferenceStatus LoadPostprocessConfig(const YAML::Node&, PostprocessConfig*, std::string*);
InferenceStatus ValidatePostprocessConfig(const ModelConfig&, const PostprocessConfig&, std::string*);
bool MapDetectionLabel(const PostprocessConfig&, int original_label, int* output_label);
const std::string* LabelNameForLabel(const std::vector<std::string>& original_names,
                                   const PostprocessConfig&, int output_label);
}  // namespace inference
