#include "trt_engine.hpp"

#include <array>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>

namespace inference_internal {
namespace {

constexpr int kDirChannelsPerTask = 2;
constexpr int kProfileIndex = 0;
constexpr char kVoxelsName[] = "voxels";
constexpr char kNumPointsName[] = "num_points";
constexpr char kCoorsName[] = "coors";
constexpr char kBboxPredName[] = "bbox_pred0";
constexpr char kClsScoreName[] = "cls_score0";
constexpr char kDirClsPredName[] = "dir_cls_pred0";

#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
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
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  const bool ok = context->setBindingDimensions(binding_index, dims);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
  return ok;
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
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  const bool ok = context->enqueueV2(bindings, stream, nullptr);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
  return ok;
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

const char* YawEncodingName(inference::YawEncoding encoding) {
  switch (encoding) {
    case inference::YawEncoding::kDirectionTheta:
      return "direction_theta";
    case inference::YawEncoding::kSinCos:
    default:
      return "sin_cos";
  }
}

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
      ValidateInputProfile(engine, kVoxelsName, 3, {0, config.max_points_per_voxel, config.num_point_features},
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
  contract->min_runtime_pillars = std::max(config.min_voxels, contract->min_profile_pillars);
  contract->max_runtime_pillars = std::min(config.max_voxels, contract->max_profile_pillars);
  if (contract->min_runtime_pillars > contract->max_runtime_pillars) {
    std::ostringstream oss;
    oss << "Runtime voxel range [" << config.min_voxels << ", " << config.max_voxels
        << "] does not overlap TensorRT optimization profile ["
        << contract->min_profile_pillars << ", " << contract->max_profile_pillars << "]";
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus BuildContract(const RuntimeConfig& config,
                              const ResolvedHeadLayout& layout,
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
  if (contract.bbox_channels <= 0 || contract.cls_channels <= 0 || contract.dir_channels <= 0) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Engine output channels must be positive");
  }

  if (config.decode_mode == inference::DecodeMode::kCenterPoint) {
    if (config.task_num_classes.empty()) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "CenterPoint runtime config requires task_num_classes");
    }
    const int expected_task_count = layout.task_count;
    if (contract.bbox_channels !=
            expected_task_count * layout.bbox_channels_per_task ||
        contract.dir_channels != expected_task_count * layout.dir_channels_per_task ||
        contract.cls_channels != layout.cls_channels_total) {
      std::ostringstream oss;
      oss << "Engine channels do not match CenterPoint task layout: bbox="
          << contract.bbox_channels << " cls=" << contract.cls_channels
          << " dir=" << contract.dir_channels << " expected bbox="
          << expected_task_count * layout.bbox_channels_per_task << " cls="
          << layout.cls_channels_total << " dir="
          << expected_task_count * layout.dir_channels_per_task
          << " for yaw_encoding=" << YawEncodingName(layout.yaw_encoding);
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
    }
    contract.task_count = expected_task_count;
  } else {
    if (contract.bbox_channels != contract.cls_channels * layout.bbox_channels_per_task ||
        contract.dir_channels != contract.cls_channels * layout.dir_channels_per_task) {
      std::ostringstream oss;
      oss << "Engine channels do not match expected legacy CenterPoint head layout: bbox="
          << contract.bbox_channels << " cls=" << contract.cls_channels
          << " dir=" << contract.dir_channels << " expected bbox="
          << contract.cls_channels * layout.bbox_channels_per_task
          << " dir=" << contract.cls_channels * layout.dir_channels_per_task;
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
    }
    contract.task_count = contract.cls_channels;
  }
  contract.num_bindings = engine.getNbBindings();
  *out_contract = contract;
  return INFERENCE_STATUS_OK;
}

int FindBindingIndex(nvinfer1::ICudaEngine& engine, const std::string& name) {
  return engine.getBindingIndex(name.c_str());
}

int FindFirstInputBinding(nvinfer1::ICudaEngine& engine) {
  const int binding_count = engine.getNbBindings();
  for (int i = 0; i < binding_count; ++i) {
    if (engine.bindingIsInput(i)) {
      return i;
    }
  }
  return -1;
}

InferenceStatus ValidateHeadOutput(nvinfer1::ICudaEngine& engine,
                                   int binding_index,
                                   const std::string& tensor_name,
                                   int expected_channels,
                                   int* inout_height,
                                   int* inout_width,
                                   std::string* out_error) {
  if (binding_index < 0) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Missing CenterPoint voxel head output: " + tensor_name);
  }
  if (engine.bindingIsInput(binding_index)) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "CenterPoint voxel head binding is an input, expected output: " + tensor_name);
  }
  if (engine.getBindingDataType(binding_index) != nvinfer1::DataType::kFLOAT) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "CenterPoint voxel head outputs must be float32 for runtime decode: " +
                        tensor_name);
  }
  const nvinfer1::Dims dims = engine.getBindingDimensions(binding_index);
  if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] != expected_channels ||
      dims.d[2] <= 0 || dims.d[3] <= 0) {
    std::ostringstream oss;
    oss << "CenterPoint voxel head output " << tensor_name << " shape is "
        << DimsToString(dims) << ", expected 1x" << expected_channels << "xHxW";
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
  }
  if (*inout_height == 0 && *inout_width == 0) {
    *inout_height = dims.d[2];
    *inout_width = dims.d[3];
  } else if (*inout_height != dims.d[2] || *inout_width != dims.d[3]) {
    std::ostringstream oss;
    oss << "CenterPoint voxel head output " << tensor_name << " spatial shape is "
        << dims.d[2] << "x" << dims.d[3] << ", expected " << *inout_height
        << "x" << *inout_width;
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
  }
  return INFERENCE_STATUS_OK;
}

int FindCanonicalOrMmdetOutput(nvinfer1::ICudaEngine& engine,
                               const std::string& canonical_name,
                               int task_id,
                               int mmdet_slot) {
  const int canonical = FindBindingIndex(engine, canonical_name);
  if (canonical >= 0) {
    return canonical;
  }
  return FindBindingIndex(engine, "out" + std::to_string(task_id * 6 + mmdet_slot));
}

InferenceStatus BuildCenterPointHeadContract(const RuntimeConfig& config,
                                             const ResolvedHeadLayout& layout,
                                             nvinfer1::ICudaEngine& engine,
                                             CenterPointHeadContract* out_contract,
                                             std::string* out_error) {
  if (out_contract == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "out_contract is null");
  }
  if (layout.yaw_encoding != inference::YawEncoding::kSinCos) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "centerpoint_voxel runtime currently requires model.yaw_encoding=sin_cos");
  }
  if (!layout.has_velocity || layout.bbox_channels_per_task != 8 ||
      layout.dir_channels_per_task != 2) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "centerpoint_voxel runtime expects reg/height/dim/rot/vel/hm head layout");
  }

  CenterPointHeadContract contract{};
  contract.task_count = layout.task_count;
  contract.bbox_channels = layout.task_count * layout.bbox_channels_per_task;
  contract.cls_channels = layout.cls_channels_total;
  contract.dir_channels = layout.task_count * layout.dir_channels_per_task;
  contract.num_bindings = engine.getNbBindings();
  contract.input_index = FindBindingIndex(engine, "input");
  if (contract.input_index < 0) {
    contract.input_index = FindFirstInputBinding(engine);
  }
  if (contract.input_index < 0 || !engine.bindingIsInput(contract.input_index)) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "CenterPoint voxel head engine must expose one input binding");
  }
  const nvinfer1::DataType input_type = engine.getBindingDataType(contract.input_index);
  if (input_type != nvinfer1::DataType::kHALF) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "CenterPoint voxel head input must be float16 to consume libspconv output");
  }

  contract.reg_indices.resize(static_cast<size_t>(layout.task_count), -1);
  contract.height_indices.resize(static_cast<size_t>(layout.task_count), -1);
  contract.dim_indices.resize(static_cast<size_t>(layout.task_count), -1);
  contract.rot_indices.resize(static_cast<size_t>(layout.task_count), -1);
  contract.vel_indices.resize(static_cast<size_t>(layout.task_count), -1);
  contract.hm_indices.resize(static_cast<size_t>(layout.task_count), -1);

  int height = 0;
  int width = 0;
  for (int task_id = 0; task_id < layout.task_count; ++task_id) {
    const int task_classes = layout.task_num_classes[static_cast<size_t>(task_id)];
    const std::string suffix = "_" + std::to_string(task_id);
    const size_t index = static_cast<size_t>(task_id);
    contract.reg_indices[index] =
        FindCanonicalOrMmdetOutput(engine, "reg" + suffix, task_id, 4);
    contract.height_indices[index] =
        FindCanonicalOrMmdetOutput(engine, "height" + suffix, task_id, 2);
    contract.dim_indices[index] =
        FindCanonicalOrMmdetOutput(engine, "dim" + suffix, task_id, 0);
    contract.rot_indices[index] =
        FindCanonicalOrMmdetOutput(engine, "rot" + suffix, task_id, 1);
    contract.vel_indices[index] =
        FindCanonicalOrMmdetOutput(engine, "vel" + suffix, task_id, 5);
    contract.hm_indices[index] =
        FindCanonicalOrMmdetOutput(engine, "hm" + suffix, task_id, 3);

    InferenceStatus status =
        ValidateHeadOutput(engine, contract.reg_indices[index], "reg" + suffix, 2,
                           &height, &width, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = ValidateHeadOutput(engine, contract.height_indices[index], "height" + suffix, 1,
                                &height, &width, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = ValidateHeadOutput(engine, contract.dim_indices[index], "dim" + suffix, 3,
                                &height, &width, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = ValidateHeadOutput(engine, contract.rot_indices[index], "rot" + suffix, 2,
                                &height, &width, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = ValidateHeadOutput(engine, contract.vel_indices[index], "vel" + suffix, 2,
                                &height, &width, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = ValidateHeadOutput(engine, contract.hm_indices[index], "hm" + suffix,
                                task_classes, &height, &width, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
  }

  const int expected_height =
      static_cast<int>((config.point_cloud_range[4] - config.point_cloud_range[1]) /
                       (config.voxel_size[1] * config.out_size_factor) + 0.5f);
  const int expected_width =
      static_cast<int>((config.point_cloud_range[3] - config.point_cloud_range[0]) /
                       (config.voxel_size[0] * config.out_size_factor) + 0.5f);
  if (height != expected_height || width != expected_width) {
    std::ostringstream oss;
    oss << "CenterPoint voxel head spatial shape is " << height << "x" << width
        << ", expected " << expected_height << "x" << expected_width
        << " from point_cloud_range/voxel_size/out_size_factor";
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
  }

  contract.height = height;
  contract.width = width;
  *out_contract = std::move(contract);
  return INFERENCE_STATUS_OK;
}

}  // namespace

InferenceStatus NormalizeCenterPointHeadInputShape(
    const std::vector<int64_t>& engine_shape,
    const std::vector<int64_t>& scn_features_shape,
    std::vector<int64_t>* out_input_shape,
    std::string* out_error) {
  if (out_input_shape == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "out_input_shape is null");
  }
  if (engine_shape.empty() || scn_features_shape.empty()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "CenterPoint head input shapes must be non-empty");
  }

  const int64_t max_dimension = std::numeric_limits<int>::max();
  const auto format_shape = [](const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) {
        oss << 'x';
      }
      oss << shape[i];
    }
    oss << ']';
    return oss.str();
  };

  for (size_t i = 0; i < scn_features_shape.size(); ++i) {
    if (scn_features_shape[i] <= 0 || scn_features_shape[i] > max_dimension) {
      std::ostringstream oss;
      oss << "libspconv output dimension " << i
          << " must be positive and fit a TensorRT dimension";
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
    }
  }
  for (const int64_t dimension : engine_shape) {
    if (dimension == 0 || dimension < -1) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "TensorRT engine exposes an invalid input dimension");
    }
  }

  std::vector<int64_t> normalized_shape = scn_features_shape;
  if (engine_shape.size() != scn_features_shape.size()) {
    // CenterPoint's dense sparse-convolution output is commonly [N,C,D,H,W],
    // while exported TensorRT heads consume [N,C*D,H,W]. Both views use the
    // same contiguous memory; fold only C and D and do not copy the tensor.
    if (engine_shape.size() == 4 && scn_features_shape.size() == 5) {
      const int64_t channels = scn_features_shape[1];
      const int64_t depth = scn_features_shape[2];
      if (channels > max_dimension / depth) {
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                        "Flattened CenterPoint head channel dimension overflows TensorRT limits");
      }
      normalized_shape = {scn_features_shape[0], channels * depth,
                          scn_features_shape[3], scn_features_shape[4]};
    } else {
      std::ostringstream oss;
      oss << "Unsupported CenterPoint head input rank mismatch: TensorRT expects "
          << engine_shape.size() << " dimensions " << format_shape(engine_shape)
          << ", but libspconv output has " << scn_features_shape.size()
          << " dimensions " << format_shape(scn_features_shape);
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
    }
  }

  if (engine_shape.size() != normalized_shape.size()) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Normalized CenterPoint head input rank does not match TensorRT");
  }
  for (size_t i = 0; i < engine_shape.size(); ++i) {
    if (engine_shape[i] > 0 && engine_shape[i] != normalized_shape[i]) {
      std::ostringstream oss;
      oss << "CenterPoint head input shape mismatch: TensorRT expects "
          << format_shape(engine_shape) << ", but libspconv output "
          << format_shape(scn_features_shape) << " normalizes to "
          << format_shape(normalized_shape)
          << " when folding the depth dimension into channels";
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
    }
  }

  *out_input_shape = std::move(normalized_shape);
  return INFERENCE_STATUS_OK;
}

struct TrtExecutor::Impl {
  TrtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime, RuntimeDeleter> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> engine;
  std::unique_ptr<nvinfer1::IExecutionContext, ContextDeleter> context;
};

TrtExecutor::TrtExecutor() = default;
TrtExecutor::~TrtExecutor() = default;

InferenceStatus TrtExecutor::Load(const RuntimeConfig& config,
                                  const ResolvedHeadLayout& layout,
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

  return BuildContract(config, layout, *impl_->engine, &contract_, out_error);
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
  voxels_dims.d[1] = config.max_points_per_voxel;
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
    const std::string logger_error = impl_->logger.last_error();
#if INFERENCE_USE_TRT_TENSOR_API
    if (!logger_error.empty()) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "TensorRT enqueueV3 failed: " + logger_error);
    }
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "TensorRT enqueueV3 failed");
#else
    if (!logger_error.empty()) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "TensorRT enqueueV2 failed: " + logger_error);
    }
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "TensorRT enqueueV2 failed");
#endif
  }
  return INFERENCE_STATUS_OK;
}

struct CenterPointHeadTrtExecutor::Impl {
  TrtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime, RuntimeDeleter> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> engine;
  std::unique_ptr<nvinfer1::IExecutionContext, ContextDeleter> context;
};

CenterPointHeadTrtExecutor::CenterPointHeadTrtExecutor() = default;
CenterPointHeadTrtExecutor::~CenterPointHeadTrtExecutor() = default;

InferenceStatus CenterPointHeadTrtExecutor::Load(const RuntimeConfig& config,
                                                 const ResolvedHeadLayout& layout,
                                                 const std::string& engine_path,
                                                 std::string* out_error) {
  contract_ = CenterPointHeadContract{};

  std::ifstream input(engine_path, std::ios::binary);
  if (!input.is_open()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Failed to open CenterPoint voxel head engine: " + engine_path);
  }
  input.seekg(0, std::ios::end);
  const std::streamsize size = input.tellg();
  input.seekg(0, std::ios::beg);
  if (size <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "CenterPoint voxel head engine file is empty: " + engine_path);
  }
  std::vector<char> data(static_cast<size_t>(size));
  if (!input.read(data.data(), size)) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Failed to read CenterPoint voxel head engine: " + engine_path);
  }

  impl_ = std::make_unique<Impl>();
  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
  if (!impl_->runtime) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Failed to create TensorRT runtime for CenterPoint voxel head");
  }
  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(data.data(), data.size()));
  if (!impl_->engine) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Failed to deserialize CenterPoint voxel head engine: " +
                        impl_->logger.last_error());
  }
  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Failed to create TensorRT context for CenterPoint voxel head");
  }

  return BuildCenterPointHeadContract(config, layout, *impl_->engine, &contract_, out_error);
}

InferenceStatus CenterPointHeadTrtExecutor::Enqueue(void* scn_features,
                                                    const std::vector<int64_t>& scn_features_shape,
                                                    float* d_bbox_pred,
                                                    float* d_cls_score,
                                                    float* d_dir_cls_pred,
                                                    cudaStream_t stream,
                                                    std::string* out_error) {
  if (!impl_ || !impl_->context) {
    return SetError(out_error, INFERENCE_STATUS_NOT_READY,
                    "CenterPoint voxel head executor is not initialized");
  }
  if (scn_features == nullptr || d_bbox_pred == nullptr || d_cls_score == nullptr ||
      d_dir_cls_pred == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "CenterPoint voxel head input/output buffers must not be null");
  }
  if (scn_features_shape.empty() ||
      scn_features_shape.size() > static_cast<size_t>(nvinfer1::Dims::MAX_DIMS)) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "CenterPoint voxel head input shape must be non-empty and fit TensorRT dims");
  }

  nvinfer1::Dims input_dims = impl_->engine->getBindingDimensions(contract_.input_index);
  std::vector<int64_t> engine_shape;
  if (input_dims.nbDims > 0) {
    engine_shape.reserve(static_cast<size_t>(input_dims.nbDims));
    for (int i = 0; i < input_dims.nbDims; ++i) {
      engine_shape.push_back(input_dims.d[i]);
    }
  }
  std::vector<int64_t> normalized_input_shape;
  const InferenceStatus shape_status = NormalizeCenterPointHeadInputShape(
      engine_shape, scn_features_shape, &normalized_input_shape, out_error);
  if (shape_status != INFERENCE_STATUS_OK) {
    return shape_status;
  }

  bool has_dynamic_input_dim = false;
  for (int i = 0; i < input_dims.nbDims; ++i) {
    if (input_dims.d[i] < 0) {
      has_dynamic_input_dim = true;
      break;
    }
  }

  if (has_dynamic_input_dim) {
    nvinfer1::Dims runtime_dims{};
    runtime_dims.nbDims = static_cast<int>(normalized_input_shape.size());
    for (int i = 0; i < runtime_dims.nbDims; ++i) {
      runtime_dims.d[i] = static_cast<int>(normalized_input_shape[static_cast<size_t>(i)]);
    }
    const char* input_name = impl_->engine->getBindingName(contract_.input_index);
    if (!SetContextInputShape(impl_->context.get(), contract_.input_index, input_name,
                              runtime_dims) ||
        !AreAllInputDimensionsSpecified(impl_->context.get())) {
      std::ostringstream oss;
      oss << "Failed to set CenterPoint voxel head input shape to "
          << DimsToString(runtime_dims);
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
    }
  }

  const int hw = contract_.height * contract_.width;
  constexpr int kRegOffset = 0;
  constexpr int kHeightOffset = 2;
  constexpr int kDimOffset = 3;
  constexpr int kVelOffset = 6;
  constexpr int kBboxChannelsPerTask = 8;
  constexpr int kDirChannelsPerTask = 2;

  std::vector<void*> bindings(static_cast<size_t>(contract_.num_bindings), nullptr);
  bindings[static_cast<size_t>(contract_.input_index)] = scn_features;
  int class_offset = 0;
  for (int task_id = 0; task_id < contract_.task_count; ++task_id) {
    const size_t index = static_cast<size_t>(task_id);
    float* bbox_base =
        d_bbox_pred + static_cast<size_t>(task_id) * kBboxChannelsPerTask * hw;
    float* dir_base =
        d_dir_cls_pred + static_cast<size_t>(task_id) * kDirChannelsPerTask * hw;
    bindings[static_cast<size_t>(contract_.reg_indices[index])] =
        bbox_base + kRegOffset * hw;
    bindings[static_cast<size_t>(contract_.height_indices[index])] =
        bbox_base + kHeightOffset * hw;
    bindings[static_cast<size_t>(contract_.dim_indices[index])] =
        bbox_base + kDimOffset * hw;
    bindings[static_cast<size_t>(contract_.rot_indices[index])] = dir_base;
    bindings[static_cast<size_t>(contract_.vel_indices[index])] =
        bbox_base + kVelOffset * hw;
    bindings[static_cast<size_t>(contract_.hm_indices[index])] =
        d_cls_score + static_cast<size_t>(class_offset) * hw;

    const nvinfer1::Dims hm_dims =
        impl_->engine->getBindingDimensions(contract_.hm_indices[index]);
    class_offset += hm_dims.d[1];
  }

#if INFERENCE_USE_TRT_TENSOR_API
  for (int binding_index = 0; binding_index < contract_.num_bindings; ++binding_index) {
    if (bindings[static_cast<size_t>(binding_index)] == nullptr) {
      continue;
    }
    const char* tensor_name = impl_->engine->getBindingName(binding_index);
    if (!SetContextTensorAddress(impl_->context.get(), tensor_name,
                                 bindings[static_cast<size_t>(binding_index)])) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "Failed to bind CenterPoint voxel head tensor address: " +
                          std::string(tensor_name == nullptr ? "<null>" : tensor_name));
    }
  }
#endif

  if (!EnqueueContext(impl_->context.get(), bindings.data(), stream)) {
    const std::string logger_error = impl_->logger.last_error();
    if (!logger_error.empty()) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "CenterPoint voxel head TensorRT enqueue failed: " + logger_error);
    }
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "CenterPoint voxel head TensorRT enqueue failed");
  }
  return INFERENCE_STATUS_OK;
}

}  // namespace inference_internal
