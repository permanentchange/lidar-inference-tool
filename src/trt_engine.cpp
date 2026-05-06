#include "trt_engine.hpp"

#include <array>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>

namespace inference_internal {
namespace {

constexpr int kBBoxChannelsPerTask = 8;
constexpr int kDirChannelsPerTask = 2;
constexpr int kProfileIndex = 0;
constexpr char kVoxelsName[] = "voxels";
constexpr char kNumPointsName[] = "num_points";
constexpr char kCoorsName[] = "coors";
constexpr char kBboxPredName[] = "bbox_pred0";
constexpr char kClsScoreName[] = "cls_score0";
constexpr char kDirClsPredName[] = "dir_cls_pred0";

#if defined(NV_TENSORRT_MAJOR) && \
    (NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 5))
#define INFERENCE_USE_TRT_TENSOR_API 1
#else
#define INFERENCE_USE_TRT_TENSOR_API 0
#endif

InferenceStatus SetError(std::string* out_error,
                         InferenceStatus status,
                         const std::string& message) {
  if (out_error != nullptr) {
    *out_error = message;
  }
  return status;
}

std::string DimsToString(const nvinfer1::Dims& dims) {
  if (dims.nbDims < 0) {
    return "<invalid>";
  }
  std::ostringstream oss;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (i > 0) {
      oss << 'x';
    }
    oss << dims.d[i];
  }
  return oss.str();
}

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kERROR) {
      last_error_ = msg == nullptr ? "" : msg;
    }
  }

  const std::string& last_error() const { return last_error_; }

 private:
  std::string last_error_;
};

bool SetContextInputShape(nvinfer1::IExecutionContext* context,
                          int binding_index,
                          const char* tensor_name,
                          const nvinfer1::Dims& dims) {
  if (context == nullptr) {
    return false;
  }
#if INFERENCE_USE_TRT_TENSOR_API
  (void)binding_index;
  return context->setInputShape(tensor_name, dims);
#else
  (void)tensor_name;
  return context->setBindingDimensions(binding_index, dims);
#endif
}

bool AreAllInputDimensionsSpecified(nvinfer1::IExecutionContext* context) {
  if (context == nullptr) {
    return false;
  }
  return context->allInputDimensionsSpecified();
}

bool SetContextTensorAddress(nvinfer1::IExecutionContext* context,
                             const char* tensor_name,
                             void* address) {
  if (context == nullptr) {
    return false;
  }
#if INFERENCE_USE_TRT_TENSOR_API
  return context->setTensorAddress(tensor_name, address);
#else
  (void)tensor_name;
  (void)address;
  return true;
#endif
}

bool EnqueueContext(nvinfer1::IExecutionContext* context,
                    void** bindings,
                    cudaStream_t stream) {
  if (context == nullptr) {
    return false;
  }
#if INFERENCE_USE_TRT_TENSOR_API
  (void)bindings;
  return context->enqueueV3(stream);
#else
  return context->enqueueV2(bindings, stream, nullptr);
#endif
}

struct RuntimeDeleter {
  void operator()(nvinfer1::IRuntime* runtime) const {
    if (runtime != nullptr) {
      runtime->destroy();
    }
  }
};

struct EngineDeleter {
  void operator()(nvinfer1::ICudaEngine* engine) const {
    if (engine != nullptr) {
      engine->destroy();
    }
  }
};

struct ContextDeleter {
  void operator()(nvinfer1::IExecutionContext* context) const {
    if (context != nullptr) {
      context->destroy();
    }
  }
};

InferenceStatus ValidateInputProfile(nvinfer1::ICudaEngine& engine,
                                     const char* tensor_name,
                                     int expected_nb_dims,
                                     const std::array<int, 3>& expected_static_dims,
                                     int* out_min_pillars,
                                     int* out_opt_pillars,
                                     int* out_max_pillars,
                                     std::string* out_error) {
  const nvinfer1::Dims min_dims =
      engine.getProfileShape(tensor_name, kProfileIndex, nvinfer1::OptProfileSelector::kMIN);
  const nvinfer1::Dims opt_dims =
      engine.getProfileShape(tensor_name, kProfileIndex, nvinfer1::OptProfileSelector::kOPT);
  const nvinfer1::Dims max_dims =
      engine.getProfileShape(tensor_name, kProfileIndex, nvinfer1::OptProfileSelector::kMAX);

  const auto validate_dims = [&](const char* label, const nvinfer1::Dims& dims) -> InferenceStatus {
    if (dims.nbDims != expected_nb_dims) {
      std::ostringstream oss;
      oss << "TensorRT profile " << label << " shape for " << tensor_name << " is "
          << DimsToString(dims) << ", expected " << expected_nb_dims << " dimensions";
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
    }
    for (int i = 1; i < expected_nb_dims; ++i) {
      if (dims.d[i] != expected_static_dims[static_cast<size_t>(i)]) {
        std::ostringstream oss;
        oss << "TensorRT profile " << label << " shape for " << tensor_name << " is "
            << DimsToString(dims) << ", expected dimension " << i << " to be "
            << expected_static_dims[static_cast<size_t>(i)];
        return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
      }
    }
    return INFERENCE_STATUS_OK;
  };

  InferenceStatus status = validate_dims("min", min_dims);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = validate_dims("opt", opt_dims);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = validate_dims("max", max_dims);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  const int min_pillars = min_dims.d[0];
  const int opt_pillars = opt_dims.d[0];
  const int max_pillars = max_dims.d[0];
  if (min_pillars <= 0 || opt_pillars < min_pillars || max_pillars < opt_pillars) {
    std::ostringstream oss;
    oss << "TensorRT profile for " << tensor_name << " is invalid: min=" << min_pillars
        << ", opt=" << opt_pillars << ", max=" << max_pillars;
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
  }
  *out_min_pillars = min_pillars;
  *out_opt_pillars = opt_pillars;
  *out_max_pillars = max_pillars;
  return INFERENCE_STATUS_OK;
}

InferenceStatus ValidateInputProfiles(const RuntimeConfig& config,
                                      nvinfer1::ICudaEngine& engine,
                                      EngineContract* contract,
                                      std::string* out_error) {
  int voxels_min = 0;
  int voxels_opt = 0;
  int voxels_max = 0;
  int num_points_min = 0;
  int num_points_opt = 0;
  int num_points_max = 0;
  int coors_min = 0;
  int coors_opt = 0;
  int coors_max = 0;

  InferenceStatus status =
      ValidateInputProfile(engine, kVoxelsName, 3, {0, config.max_points_per_pillar, config.num_point_features},
                           &voxels_min, &voxels_opt, &voxels_max, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = ValidateInputProfile(engine, kNumPointsName, 1, {0, 0, 0},
                                &num_points_min, &num_points_opt, &num_points_max, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = ValidateInputProfile(engine, kCoorsName, 2, {0, 4, 0},
                                &coors_min, &coors_opt, &coors_max, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  if (voxels_min != num_points_min || voxels_opt != num_points_opt || voxels_max != num_points_max ||
      voxels_min != coors_min || voxels_opt != coors_opt || voxels_max != coors_max) {
    std::ostringstream oss;
    oss << "TensorRT input profiles disagree on pillar-count range: voxels=["
        << voxels_min << ", " << voxels_opt << ", " << voxels_max << "], num_points=["
        << num_points_min << ", " << num_points_opt << ", " << num_points_max
        << "], coors=[" << coors_min << ", " << coors_opt << ", " << coors_max << "]";
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
  }

  contract->min_profile_pillars = voxels_min;
  contract->opt_profile_pillars = voxels_opt;
  contract->max_profile_pillars = voxels_max;
  contract->min_runtime_pillars = std::max(config.min_pillars, contract->min_profile_pillars);
  contract->max_runtime_pillars = std::min(config.max_pillars, contract->max_profile_pillars);
  if (contract->min_runtime_pillars > contract->max_runtime_pillars) {
    std::ostringstream oss;
    oss << "Runtime pillar range [" << config.min_pillars << ", " << config.max_pillars
        << "] does not overlap TensorRT optimization profile ["
        << contract->min_profile_pillars << ", " << contract->max_profile_pillars << "]";
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus BuildContract(const RuntimeConfig& config,
                              nvinfer1::ICudaEngine& engine,
                              EngineContract* out_contract,
                              std::string* out_error) {
  if (out_contract == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, "out_contract is null");
  }

  EngineContract contract{};
  contract.voxels_index = engine.getBindingIndex(kVoxelsName);
  contract.num_points_index = engine.getBindingIndex(kNumPointsName);
  contract.coors_index = engine.getBindingIndex(kCoorsName);
  contract.bbox_pred_index = engine.getBindingIndex(kBboxPredName);
  contract.cls_score_index = engine.getBindingIndex(kClsScoreName);
  contract.dir_cls_pred_index = engine.getBindingIndex(kDirClsPredName);
  if (contract.voxels_index < 0 || contract.num_points_index < 0 || contract.coors_index < 0 ||
      contract.bbox_pred_index < 0 || contract.cls_score_index < 0 || contract.dir_cls_pred_index < 0) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Engine bindings do not match expected CenterPoint pillar names");
  }

  if (engine.getBindingDataType(contract.voxels_index) != nvinfer1::DataType::kFLOAT ||
      engine.getBindingDataType(contract.num_points_index) != nvinfer1::DataType::kINT32 ||
      engine.getBindingDataType(contract.coors_index) != nvinfer1::DataType::kINT32 ||
      engine.getBindingDataType(contract.bbox_pred_index) != nvinfer1::DataType::kFLOAT ||
      engine.getBindingDataType(contract.cls_score_index) != nvinfer1::DataType::kFLOAT ||
      engine.getBindingDataType(contract.dir_cls_pred_index) != nvinfer1::DataType::kFLOAT) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Engine binding dtypes do not match expected float/int32 contract");
  }

  InferenceStatus status = ValidateInputProfiles(config, engine, &contract, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  const nvinfer1::Dims bbox_dims = engine.getBindingDimensions(contract.bbox_pred_index);
  const nvinfer1::Dims cls_dims = engine.getBindingDimensions(contract.cls_score_index);
  const nvinfer1::Dims dir_dims = engine.getBindingDimensions(contract.dir_cls_pred_index);
  if (bbox_dims.nbDims != 4 || cls_dims.nbDims != 4 || dir_dims.nbDims != 4) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Engine outputs must be 4D BCHW tensors");
  }

  contract.bbox_channels = bbox_dims.d[1];
  contract.cls_channels = cls_dims.d[1];
  contract.dir_channels = dir_dims.d[1];
  contract.height = cls_dims.d[2];
  contract.width = cls_dims.d[3];
  if (bbox_dims.d[2] != contract.height || bbox_dims.d[3] != contract.width ||
      dir_dims.d[2] != contract.height || dir_dims.d[3] != contract.width) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Engine output feature map shapes do not match");
  }
  if (contract.bbox_channels <= 0 || contract.cls_channels <= 0 || contract.dir_channels <= 0 ||
      contract.bbox_channels != contract.cls_channels * kBBoxChannelsPerTask ||
      contract.dir_channels != contract.cls_channels * kDirChannelsPerTask) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Engine channels do not match expected 3-task CenterPoint head layout");
  }

  contract.task_count = contract.cls_channels;
  contract.num_bindings = engine.getNbBindings();
  *out_contract = contract;
  return INFERENCE_STATUS_OK;
}

}  // namespace

struct TrtExecutor::Impl {
  TrtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime, RuntimeDeleter> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> engine;
  std::unique_ptr<nvinfer1::IExecutionContext, ContextDeleter> context;
};

TrtExecutor::TrtExecutor() = default;
TrtExecutor::~TrtExecutor() = default;

InferenceStatus TrtExecutor::Load(const RuntimeConfig& config,
                                  const std::string& engine_path,
                                  std::string* out_error) {
  contract_ = EngineContract{};

  std::ifstream input(engine_path, std::ios::binary);
  if (!input.is_open()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Failed to open engine: " + engine_path);
  }
  input.seekg(0, std::ios::end);
  const std::streamsize size = input.tellg();
  input.seekg(0, std::ios::beg);
  if (size <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Engine file is empty: " + engine_path);
  }
  std::vector<char> data(static_cast<size_t>(size));
  if (!input.read(data.data(), size)) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Failed to read engine: " + engine_path);
  }

  impl_ = std::make_unique<Impl>();
  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
  if (!impl_->runtime) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Failed to create TensorRT runtime");
  }

  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(data.data(), data.size()));
  if (!impl_->engine) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Failed to deserialize engine: " + impl_->logger.last_error());
  }

  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Failed to create execution context");
  }

  if (impl_->engine->getNbOptimizationProfiles() < 1) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "TensorRT engine must expose at least one optimization profile");
  }
  if (!impl_->context->setOptimizationProfile(kProfileIndex)) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Failed to activate TensorRT optimization profile 0");
  }

  return BuildContract(config, *impl_->engine, &contract_, out_error);
}

InferenceStatus TrtExecutor::ConfigureShapes(const RuntimeConfig& config,
                                             int num_pillars,
                                             std::string* out_error) {
  if (!impl_ || !impl_->context) {
    return SetError(out_error, INFERENCE_STATUS_NOT_READY,
                    "TensorRT executor is not initialized");
  }
  if (num_pillars < contract_.min_runtime_pillars || num_pillars > contract_.max_runtime_pillars) {
    std::ostringstream oss;
    oss << "Pillar count " << num_pillars << " is outside supported runtime/profile range ["
        << contract_.min_runtime_pillars << ", " << contract_.max_runtime_pillars << "]";
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
  }

  nvinfer1::Dims voxels_dims;
  voxels_dims.nbDims = 3;
  voxels_dims.d[0] = num_pillars;
  voxels_dims.d[1] = config.max_points_per_pillar;
  voxels_dims.d[2] = config.num_point_features;

  nvinfer1::Dims num_points_dims;
  num_points_dims.nbDims = 1;
  num_points_dims.d[0] = num_pillars;

  nvinfer1::Dims coors_dims;
  coors_dims.nbDims = 2;
  coors_dims.d[0] = num_pillars;
  coors_dims.d[1] = 4;

  if (!SetContextInputShape(impl_->context.get(), contract_.voxels_index, kVoxelsName, voxels_dims) ||
      !SetContextInputShape(impl_->context.get(), contract_.num_points_index, kNumPointsName,
                            num_points_dims) ||
      !SetContextInputShape(impl_->context.get(), contract_.coors_index, kCoorsName, coors_dims) ||
      !AreAllInputDimensionsSpecified(impl_->context.get())) {
    std::ostringstream oss;
    oss << "Failed to set TensorRT binding dimensions for pillar count " << num_pillars
        << " within optimization profile [" << contract_.min_profile_pillars << ", "
        << contract_.max_profile_pillars << "]";
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus TrtExecutor::Enqueue(void** bindings,
                                     cudaStream_t stream,
                                     std::string* out_error) {
  if (!impl_ || !impl_->context) {
    return SetError(out_error, INFERENCE_STATUS_NOT_READY,
                    "TensorRT executor is not initialized");
  }
#if INFERENCE_USE_TRT_TENSOR_API
  if (!SetContextTensorAddress(impl_->context.get(), kVoxelsName,
                               bindings[contract_.voxels_index]) ||
      !SetContextTensorAddress(impl_->context.get(), kNumPointsName,
                               bindings[contract_.num_points_index]) ||
      !SetContextTensorAddress(impl_->context.get(), kCoorsName,
                               bindings[contract_.coors_index]) ||
      !SetContextTensorAddress(impl_->context.get(), kBboxPredName,
                               bindings[contract_.bbox_pred_index]) ||
      !SetContextTensorAddress(impl_->context.get(), kClsScoreName,
                               bindings[contract_.cls_score_index]) ||
      !SetContextTensorAddress(impl_->context.get(), kDirClsPredName,
                               bindings[contract_.dir_cls_pred_index])) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Failed to bind TensorRT tensor addresses");
  }
#endif
  if (!EnqueueContext(impl_->context.get(), bindings, stream)) {
#if INFERENCE_USE_TRT_TENSOR_API
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "TensorRT enqueueV3 failed");
#else
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "TensorRT enqueueV2 failed");
#endif
  }
  return INFERENCE_STATUS_OK;
}

}  // namespace inference_internal
