#include "inference_cpp.hpp"

namespace inference {
namespace {

InferenceStatus SetStubError(std::string* out_error) {
  if (out_error != nullptr) {
    *out_error = "Inference library was built without CUDA or TensorRT support";
  }
  return INFERENCE_STATUS_NOT_SUPPORTED;
}

}  // namespace

LidarInferenceRuntime::~LidarInferenceRuntime() {
  inference_destroy(handle_);
  handle_ = nullptr;
}

InferenceStatus LidarInferenceRuntime::LoadPackageConfigFile(const std::string&,
                                                             PackageConfig*,
                                                             std::string* out_error,
                                                             const std::string&) {
  return SetStubError(out_error);
}

InferenceStatus LidarInferenceRuntime::LoadPackageConfig(const std::string&,
                                                         PackageConfig*,
                                                         std::string* out_error,
                                                         const std::string&) {
  return SetStubError(out_error);
}

InferenceStatus LidarInferenceRuntime::Create(
    const PackageConfig&,
    std::unique_ptr<LidarInferenceRuntime>* out_runtime,
    std::string* out_error) {
  if (out_runtime != nullptr) {
    *out_runtime = nullptr;
  }
  return SetStubError(out_error);
}

InferenceStatus LidarInferenceRuntime::CreateFromModelDir(
    const std::string&,
    std::unique_ptr<LidarInferenceRuntime>* out_runtime,
    std::string* out_error,
    const std::string&) {
  if (out_runtime != nullptr) {
    *out_runtime = nullptr;
  }
  return SetStubError(out_error);
}

InferenceStatus LidarInferenceRuntime::Infer(const PointCloudView&,
                                             std::vector<Detection>* out_detections) {
  if (out_detections != nullptr) {
    out_detections->clear();
  }
  return SetStubError(&last_error_);
}

InferenceStatus LidarInferenceRuntime::InferDecoded(const PointCloudView&,
                                                    std::vector<Detection>* out_detections) {
  if (out_detections != nullptr) {
    out_detections->clear();
  }
  return SetStubError(&last_error_);
}

InferenceStatus LidarInferenceRuntime::InferWithPostprocess(
    const PointCloudView&,
    const PostprocessConfig&,
    std::vector<Detection>* out_detections) {
  if (out_detections != nullptr) {
    out_detections->clear();
  }
  return SetStubError(&last_error_);
}

InferenceStatus LidarInferenceRuntime::InferInternal(const PointCloudView&,
                                                     const InferencePostprocessConfig*,
                                                     std::vector<Detection>*,
                                                     bool) {
  return SetStubError(&last_error_);
}

InferenceStatus LidarInferenceRuntime::Reset() {
  return SetStubError(&last_error_);
}

}  // namespace inference
