#include "inference_cpp.hpp"
#include "pillar_runtime.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool Check(bool condition, const char* expression, int line) {
  if (!condition) {
    std::cerr << "CHECK failed at line " << line << ": " << expression << "\n";
  }
  return condition;
}

#define CHECK(expression) Check((expression), #expression, __LINE__)

std::filesystem::path WriteNamedConfig(const std::string& name, const std::string& yaml) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / name;
  std::ofstream output(path);
  output << yaml;
  return path;
}

std::filesystem::path WriteConfig(const std::string& yaml) {
  return WriteNamedConfig("lidar_self_car_nms_test.yaml", yaml);
}

bool TestLoadsSelfCarConfig() {
  const std::filesystem::path path = WriteConfig(R"yaml(
model:
  engine_path: dummy.engine
postprocess:
  nms_type: iou_min_bev
  self_car_box: [0.5, 0.0, 0.5, 1.0, 0.8, 1.0, 0.25]
  self_car_nms_thresh: 0.2
)yaml");

  inference::PackageConfig config;
  std::string error;
  const InferenceStatus status = inference::LidarInferenceRuntime::LoadPackageConfigFile(
      path.string(), &config, &error);
  std::filesystem::remove(path);

  if (!CHECK(status == INFERENCE_STATUS_OK)) {
    std::cerr << error << "\n";
    return false;
  }
  if (!CHECK(config.postprocess.self_car_box.has_value()) ||
      !CHECK(config.postprocess.self_car_nms_thresh.has_value())) {
    return false;
  }
  return CHECK(std::fabs((*config.postprocess.self_car_box)[0] - 0.5f) < 1.0e-6f) &&
         CHECK(std::fabs((*config.postprocess.self_car_box)[6] - 0.25f) < 1.0e-6f) &&
         CHECK(std::fabs(*config.postprocess.self_car_nms_thresh - 0.2f) < 1.0e-6f);
}

bool TestOldConfigAndPartialOverrideRemainCompatible() {
  const std::filesystem::path old_path = WriteConfig(R"yaml(
model:
  engine_path: dummy.engine
postprocess:
  nms_type: iou_bev
)yaml");
  inference::PackageConfig old_config;
  std::string error;
  InferenceStatus status = inference::LidarInferenceRuntime::LoadPackageConfigFile(
      old_path.string(), &old_config, &error);
  std::filesystem::remove(old_path);
  if (!CHECK(status == INFERENCE_STATUS_OK) ||
      !CHECK(!old_config.postprocess.self_car_box.has_value()) ||
      !CHECK(!old_config.postprocess.self_car_nms_thresh.has_value())) {
    return false;
  }

  const std::filesystem::path base_path = WriteNamedConfig(
      "lidar_self_car_nms_base.yaml", R"yaml(
model:
  engine_path: dummy.engine
postprocess:
  nms_type: iou_min_bev
  self_car_box: [0.5, 0, 0.5, 1, 0.8, 1, 0]
  self_car_nms_thresh: 0.2
)yaml");
  const std::filesystem::path override_path = WriteNamedConfig(
      "lidar_self_car_nms_override.yaml", R"yaml(
postprocess:
  self_car_nms_thresh: 0.4
)yaml");
  inference::PackageConfig merged_config;
  error.clear();
  status = inference::LidarInferenceRuntime::LoadPackageConfigFile(
      base_path.string(), &merged_config, &error, override_path.string());
  std::filesystem::remove(base_path);
  std::filesystem::remove(override_path);
  return CHECK(status == INFERENCE_STATUS_OK) &&
         CHECK(merged_config.postprocess.self_car_box.has_value()) &&
         CHECK(merged_config.postprocess.self_car_nms_thresh.has_value()) &&
         CHECK(std::fabs(*merged_config.postprocess.self_car_nms_thresh - 0.4f) < 1.0e-6f);
}

bool TestRejectsIncompleteSelfCarConfig() {
  const std::string yaml_prefix = R"yaml(
model:
  engine_path: dummy.engine
postprocess:
  nms_type: iou_bev
)yaml";
  const std::string cases[] = {
      yaml_prefix + "  self_car_box: [0, 0, 0, 1, 1, 1, 0]\n",
      yaml_prefix + "  self_car_nms_thresh: 0.2\n",
  };

  for (const std::string& yaml : cases) {
    const std::filesystem::path path = WriteConfig(yaml);
    inference::PackageConfig config;
    std::string error;
    const InferenceStatus status = inference::LidarInferenceRuntime::LoadPackageConfigFile(
        path.string(), &config, &error);
    std::filesystem::remove(path);
    if (!CHECK(status == INFERENCE_STATUS_INVALID_ARGUMENT) ||
        !CHECK(error.find("must be configured together") != std::string::npos)) {
      std::cerr << error << "\n";
      return false;
    }
  }
  return true;
}

bool ExpectInvalidConfig(const std::string& postprocess_yaml,
                         const std::string& expected_error) {
  const std::filesystem::path path = WriteConfig(
      "model:\n  engine_path: dummy.engine\npostprocess:\n" + postprocess_yaml);
  inference::PackageConfig config;
  std::string error;
  const InferenceStatus status = inference::LidarInferenceRuntime::LoadPackageConfigFile(
      path.string(), &config, &error);
  std::filesystem::remove(path);
  if (!CHECK(status == INFERENCE_STATUS_INVALID_ARGUMENT) ||
      !CHECK(error.find(expected_error) != std::string::npos)) {
    std::cerr << error << "\n";
    return false;
  }
  return true;
}

bool TestValidatesSelfCarValuesAndNmsType() {
  return ExpectInvalidConfig(
             "  nms_type: iou_bev\n"
             "  self_car_box: [0, 0, 0, 1, 1, 1]\n"
             "  self_car_nms_thresh: 0.2\n",
             "must be a sequence of 7 floats") &&
         ExpectInvalidConfig(
             "  nms_type: iou_bev\n"
             "  self_car_box: [0, 0, 0, 0, 1, 1, 0]\n"
             "  self_car_nms_thresh: 0.2\n",
             "dimensions must be positive") &&
         ExpectInvalidConfig(
             "  nms_type: iou_min_bev\n"
             "  self_car_box: [.nan, 0, 0, 1, 1, 1, 0]\n"
             "  self_car_nms_thresh: 0.2\n",
             "must contain finite values") &&
         ExpectInvalidConfig(
             "  nms_type: iou_bev\n"
             "  self_car_box: [0, 0, 0, 1, 1, 1, 0]\n"
             "  self_car_nms_thresh: -0.1\n",
             "must be finite and non-negative") &&
         ExpectInvalidConfig(
             "  nms_type: circle\n"
             "  self_car_box: [0, 0, 0, 1, 1, 1, 0]\n"
             "  self_car_nms_thresh: 0.2\n",
             "only supports iou_bev and iou_min_bev");
}

void AddCandidate(inference_internal::CandidateDetections* candidates,
                  float score,
                  int label,
                  const std::array<float, 7>& box) {
  candidates->scores.push_back(score);
  candidates->labels.push_back(label);
  candidates->boxes.insert(candidates->boxes.end(), box.begin(), box.end());
}

inference_internal::PostprocessConfig MakeSelfCarPostprocess(InferenceNmsType nms_type,
                                                             float self_car_thresh) {
  inference_internal::PostprocessConfig config;
  config.nms_type = nms_type;
  config.nms_thresh = 0.5f;
  config.nms_pre_maxsize = 10;
  config.nms_post_maxsize = 10;
  config.final_score_threshold = 0.0f;
  config.label_score_thresh = {0.0f, 0.0f, 0.0f};
  config.self_car_box = std::array<float, 7>{{0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 1.0f, 0.0f}};
  config.self_car_nms_thresh = self_car_thresh;
  return config;
}

bool RunNms(const inference_internal::CandidateDetections& candidates,
            const inference_internal::PostprocessConfig& postprocess,
            inference_internal::CandidateDetections* result) {
  inference_internal::RuntimeConfig runtime;
  runtime.label_count = 3;
  std::string error;
  const InferenceStatus status = inference_internal::RunPostprocessNms(
      candidates, postprocess, runtime, result, &error);
  if (!CHECK(status == INFERENCE_STATUS_OK)) {
    std::cerr << error << "\n";
    return false;
  }
  return true;
}

bool TestSelfCarUsesConfiguredIouMetric() {
  inference_internal::CandidateDetections candidates;
  AddCandidate(&candidates, 0.9f, 0, {{0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f}});
  AddCandidate(&candidates, 0.8f, 0, {{10.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f}});

  inference_internal::CandidateDetections iou_result;
  if (!RunNms(candidates, MakeSelfCarPostprocess(INFERENCE_NMS_IOU_BEV, 0.2f),
              &iou_result)) {
    return false;
  }
  if (!CHECK(iou_result.scores.size() == 1U) || !CHECK(iou_result.scores[0] == 0.8f)) {
    return false;
  }

  inference_internal::CandidateDetections iou_min_result;
  if (!RunNms(candidates, MakeSelfCarPostprocess(INFERENCE_NMS_IOU_MIN_BEV, 0.5f),
              &iou_min_result)) {
    return false;
  }
  return CHECK(iou_min_result.scores.size() == 1U) &&
         CHECK(iou_min_result.scores[0] == 0.8f);
}

bool TestSelfCarThresholdIsStrictAndDoesNotReplaceNmsThreshold() {
  inference_internal::CandidateDetections boundary_candidate;
  AddCandidate(&boundary_candidate, 0.9f, 0,
               {{0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 1.0f, 0.0f}});
  inference_internal::CandidateDetections boundary_result;
  if (!RunNms(boundary_candidate,
              MakeSelfCarPostprocess(INFERENCE_NMS_IOU_MIN_BEV, 1.0f),
              &boundary_result)) {
    return false;
  }
  if (!CHECK(boundary_result.scores.size() == 1U)) {
    return false;
  }

  inference_internal::CandidateDetections real_candidates;
  AddCandidate(&real_candidates, 0.9f, 0,
               {{10.0f, 0.0f, 0.0f, 2.0f, 2.0f, 1.0f, 0.0f}});
  AddCandidate(&real_candidates, 0.8f, 0,
               {{10.0f, 0.0f, 0.0f, 2.0f, 2.0f, 1.0f, 0.0f}});
  inference_internal::PostprocessConfig postprocess =
      MakeSelfCarPostprocess(INFERENCE_NMS_IOU_BEV, 0.1f);
  postprocess.nms_thresh = 1.0f;
  inference_internal::CandidateDetections real_result;
  return RunNms(real_candidates, postprocess, &real_result) &&
         CHECK(real_result.scores.size() == 2U);
}

bool TestSelfCarDoesNotConsumePostMaxsize() {
  inference_internal::CandidateDetections candidates;
  AddCandidate(&candidates, 0.9f, 0,
               {{10.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f}});
  AddCandidate(&candidates, 0.8f, 0,
               {{20.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f}});
  inference_internal::PostprocessConfig postprocess =
      MakeSelfCarPostprocess(INFERENCE_NMS_IOU_BEV, 0.2f);
  postprocess.nms_post_maxsize = 1;
  inference_internal::CandidateDetections result;
  return RunNms(candidates, postprocess, &result) &&
         CHECK(result.scores.size() == 1U) && CHECK(result.scores[0] == 0.9f);
}

bool TestSelfCarDoesNotConsumePreMaxsize() {
  inference_internal::CandidateDetections candidates;
  AddCandidate(&candidates, 0.9f, 0,
               {{10.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f}});
  AddCandidate(&candidates, 0.8f, 0,
               {{20.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f}});
  inference_internal::PostprocessConfig postprocess =
      MakeSelfCarPostprocess(INFERENCE_NMS_IOU_BEV, 0.2f);
  postprocess.nms_pre_maxsize = 1;
  inference_internal::CandidateDetections result;
  return RunNms(candidates, postprocess, &result) &&
         CHECK(result.scores.size() == 1U) && CHECK(result.scores[0] == 0.9f);
}

bool TestSelfCarAppliesToGroupedNms() {
  inference_internal::CandidateDetections candidates;
  AddCandidate(&candidates, 0.9f, 0,
               {{0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f}});
  inference_internal::PostprocessConfig postprocess =
      MakeSelfCarPostprocess(INFERENCE_NMS_IOU_MIN_BEV, 0.2f);
  inference_internal::PostprocessConfig::NmsLabelGroup group;
  group.labels = {0, 1, 2};
  group.nms_thresh = 0.5f;
  group.nms_pre_maxsize = 10;
  group.nms_post_maxsize = 10;
  postprocess.nms_label_groups = {group};

  inference_internal::CandidateDetections result;
  return RunNms(candidates, postprocess, &result) && CHECK(result.scores.empty());
}

bool TestSelfCarAppliesToCenterPointTaskNms() {
  inference_internal::CandidateDetections candidates;
  AddCandidate(&candidates, 0.9f, 0,
               {{0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f}});
  const inference_internal::PostprocessConfig postprocess =
      MakeSelfCarPostprocess(INFERENCE_NMS_IOU_MIN_BEV, 0.2f);
  inference_internal::RuntimeConfig runtime;
  runtime.decode_mode = inference::DecodeMode::kCenterPoint;
  runtime.task_num_classes = {1};
  runtime.task_label_ids = {{0}};
  runtime.label_count = 1;
  inference_internal::CandidateDetections result;
  std::string error;
  const InferenceStatus status = inference_internal::RunPostprocessNms(
      candidates, postprocess, runtime, &result, &error);
  if (!CHECK(status == INFERENCE_STATUS_OK)) {
    std::cerr << error << "\n";
    return false;
  }
  return CHECK(result.scores.empty());
}

bool TestInternalNmsRejectsInvalidSelfCarConfiguration() {
  inference_internal::CandidateDetections candidates;
  AddCandidate(&candidates, 0.9f, 0,
               {{10.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f}});
  inference_internal::RuntimeConfig runtime;
  runtime.label_count = 3;
  inference_internal::CandidateDetections result;
  std::string error;

  inference_internal::PostprocessConfig incomplete;
  incomplete.self_car_box =
      std::array<float, 7>{{0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 1.0f, 0.0f}};
  InferenceStatus status = inference_internal::RunPostprocessNms(
      candidates, incomplete, runtime, &result, &error);
  if (!CHECK(status == INFERENCE_STATUS_INVALID_ARGUMENT) ||
      !CHECK(error.find("must be configured together") != std::string::npos)) {
    return false;
  }

  inference_internal::PostprocessConfig circle =
      MakeSelfCarPostprocess(INFERENCE_NMS_CIRCLE, 0.2f);
  error.clear();
  status = inference_internal::RunPostprocessNms(
      candidates, circle, runtime, &result, &error);
  return CHECK(status == INFERENCE_STATUS_INVALID_ARGUMENT) &&
         CHECK(error.find("only supports IoU") != std::string::npos);
}

}  // namespace

int main() {
  if (!TestLoadsSelfCarConfig()) {
    return 1;
  }
  if (!TestOldConfigAndPartialOverrideRemainCompatible()) {
    return 1;
  }
  if (!TestRejectsIncompleteSelfCarConfig()) {
    return 1;
  }
  if (!TestValidatesSelfCarValuesAndNmsType()) {
    return 1;
  }
  if (!TestSelfCarUsesConfiguredIouMetric()) {
    return 1;
  }
  if (!TestSelfCarThresholdIsStrictAndDoesNotReplaceNmsThreshold()) {
    return 1;
  }
  if (!TestSelfCarDoesNotConsumePostMaxsize()) {
    return 1;
  }
  if (!TestSelfCarDoesNotConsumePreMaxsize()) {
    return 1;
  }
  if (!TestSelfCarAppliesToGroupedNms()) {
    return 1;
  }
  if (!TestSelfCarAppliesToCenterPointTaskNms()) {
    return 1;
  }
  if (!TestInternalNmsRejectsInvalidSelfCarConfiguration()) {
    return 1;
  }
  std::cout << "self_car_nms_test passed\n";
  return 0;
}
