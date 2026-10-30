#pragma once
#include "inference_cpp.hpp"
#include "bevfusion_inference.hpp"
#include "pillar_runtime.hpp"
namespace inference_internal {
struct RuntimeDiagnostics {
  static InferenceStatus Lidar(inference::LidarInferenceRuntime&, const inference::PointCloudView&,
      std::vector<inference::Detection>*, StageTimings*, DebugArtifacts* = nullptr);
  static inference::BevFusionResult Fusion(inference::BevFusionInferenceRuntime&,
      const inference::BevFusionFrame&, bool profile, bool debug = false);
};
}
