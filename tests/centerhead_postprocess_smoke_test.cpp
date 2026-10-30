#include "bevfusion_inference.hpp"
#include "pillar_runtime.hpp"
#include "postprocess_config.hpp"
#include <stdexcept>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) throw std::runtime_error("Expected repository root");
  const std::string root = argv[1];
  inference::PackageConfig cp;
  std::string error;
  if (inference::LidarInferenceRuntime::LoadPackageConfig(root + "/demo/centerpoint_voxel", &cp, &error)
      != INFERENCE_STATUS_OK) throw std::runtime_error(error);
  const auto bf = inference::LoadBevFusionConfig(root + "/demo/bevfusion_exp20261028");
  if (bf.centerhead_postprocess.label_mapping != cp.postprocess.label_mapping ||
      bf.centerhead_postprocess.mapped_label_names != cp.postprocess.mapped_label_names)
    throw std::runtime_error("Backend label mappings differ");
  inference_internal::CandidateDetections candidates;
  candidates.labels = {0, 1, 4, 2, 6, 5};
  candidates.scores = {.6f, .9f, .95f, .99f, .45f, .2f};
  candidates.boxes = {5,2,0,2,1,1,0, 5,2,0,2,1,1,0, .7f,0,0,1.8f,1.4f,.2f,0,
                      10,-4,0,2,1,1,0, 12,5,0,1,1,1,0, 14,5,0,1,1,1,0};
  inference_internal::RuntimeConfig runtime;
  runtime.decode_mode = inference::DecodeMode::kCenterPoint;
  runtime.label_count = 9;
  runtime.task_num_classes = cp.model.task_num_classes;
  runtime.task_label_ids = cp.model.task_label_ids;
  const inference::PostprocessConfig* configurations[] = {&cp.postprocess, &bf.centerhead_postprocess};
  for (const auto* config : configurations) {
    inference_internal::ToolConfig tool;
    tool.package.postprocess = *config;
    inference_internal::CandidateDetections kept;
    if (inference_internal::RunPostprocessNms(candidates,
          inference_internal::MakeToolPostprocessConfig(tool), runtime, &kept, &error)
        != INFERENCE_STATUS_OK) throw std::runtime_error(error);
    if (kept.source_indices != std::vector<size_t>({3,1}) || kept.labels != std::vector<int>({2,1}))
      throw std::runtime_error("Cross-task NMS, score ordering, ego filtering or candidate indices differ");
    int mapped = -1;
    if (!inference::MapDetectionLabel(*config, kept.labels[0], &mapped) || mapped != 0 ||
        *inference::LabelNameForLabel(cp.model.label_names, *config, mapped) != "bi")
      throw std::runtime_error("Business label mapping differs");
  }
  std::cout << "Shared postprocess smoke: passed\n";
}
