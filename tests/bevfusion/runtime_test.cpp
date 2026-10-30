#include <cuda_runtime.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "bevfusion_inference.hpp"
using namespace inference;
void Expect(bool v, const char* text) {
  if (!v) throw std::runtime_error(text);
}
int main(int argc, char** argv) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || !count || argc != 3 ||
      !std::filesystem::exists(std::filesystem::path(argv[1]) / "model.yaml")) {
    std::cout << "SKIP: requires GPU and prepared BEVFusion package\n";
    return 77;
  }
  try {
    BevFusionInferenceRuntime runtime(argv[1]);
    auto frame = LoadBevFusionFrame(
        argv[2],
        runtime.config(), false);
    auto a = runtime.Infer(frame);
    auto b = runtime.Infer(frame, true);
    Expect(!a.geometry_cache_hit && b.geometry_cache_hit,
           "geometry cache reuse");
    Expect(!b.tensors.empty(), "internal dump");
    Expect(a.detections.size() == b.detections.size(), "repeat count");
    for (size_t i = 0; i < a.detections.size(); ++i) {
      Expect(a.detections[i].box == b.detections[i].box &&
                 a.detections[i].velocity == b.detections[i].velocity &&
                 a.detections[i].score == b.detections[i].score &&
                 a.detections[i].label == b.detections[i].label,
             "repeat determinism");
    }
    auto changed = frame;
    changed.cameras[0].img_aug_matrix[7] += 1;
    auto c = runtime.Infer(changed);
    Expect(!c.geometry_cache_hit, "camera augmentation cache invalidation");
    auto empty = frame;
    empty.points.assign(5, 999);
    bool rejected = false;
    try {
      runtime.Infer(empty);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    Expect(rejected, "empty ROI rejection");
    auto d = runtime.Infer(frame);
    Expect(d.detections.size() == a.detections.size(),
           "runtime reusable after input error");
    std::cout << "BEVFusion runtime repeat/cache/error checks passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
