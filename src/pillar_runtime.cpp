#include "pillar_runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include "voxelization.hpp"

namespace inference_internal {
namespace {

constexpr int kBoxDim = 7;
constexpr int kBBoxChannelsPerTask = 8;
constexpr int kDirChannelsPerTask = 2;
constexpr int kBatchIndex = 0;
constexpr int kProfileIndex = 0;
constexpr char kVoxelsName[] = "voxels";
constexpr char kNumPointsName[] = "num_points";
constexpr char kCoorsName[] = "coors";
constexpr char kBboxPredName[] = "bbox_pred0";
constexpr char kClsScoreName[] = "cls_score0";
constexpr char kDirClsPredName[] = "dir_cls_pred0";

double ElapsedMs(const std::chrono::steady_clock::time_point& start,
                 const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

InferenceStatus SetError(std::string* out_error,
                         InferenceStatus status,
                         const std::string& message) {
  if (out_error != nullptr) {
    *out_error = message;
  }
  return status;
}

bool IsFiniteArray(const float* values, int count) {
  for (int i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) {
      return false;
    }
  }
  return true;
}

float Sigmoid(float value) {
  if (value >= 0.0f) {
    const float z = std::exp(-value);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(value);
  return z / (1.0f + z);
}

float SafeExp(float value) {
  constexpr float kClampMin = -20.0f;
  constexpr float kClampMax = 20.0f;
  return std::exp(std::max(kClampMin, std::min(kClampMax, value)));
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

class TrtEngine {
 public:
  InferenceStatus Load(const RuntimeConfig& config,
                       const std::string& engine_path,
                       std::string* out_error) {
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

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "Failed to create TensorRT runtime");
    }

    engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
    if (!engine_) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "Failed to deserialize engine: " + logger_.last_error());
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "Failed to create execution context");
    }

    const int profile_count = engine_->getNbOptimizationProfiles();
    if (profile_count < 1) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "TensorRT engine must expose at least one optimization profile");
    }
    if (!context_->setOptimizationProfile(kProfileIndex)) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "Failed to activate TensorRT optimization profile 0");
    }

    voxels_index_ = engine_->getBindingIndex(kVoxelsName);
    num_points_index_ = engine_->getBindingIndex(kNumPointsName);
    coors_index_ = engine_->getBindingIndex(kCoorsName);
    bbox_pred_index_ = engine_->getBindingIndex(kBboxPredName);
    cls_score_index_ = engine_->getBindingIndex(kClsScoreName);
    dir_cls_pred_index_ = engine_->getBindingIndex(kDirClsPredName);
    if (voxels_index_ < 0 || num_points_index_ < 0 || coors_index_ < 0 ||
        bbox_pred_index_ < 0 || cls_score_index_ < 0 || dir_cls_pred_index_ < 0) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "Engine bindings do not match expected CenterPoint pillar names");
    }

    if (engine_->getBindingDataType(voxels_index_) != nvinfer1::DataType::kFLOAT ||
        engine_->getBindingDataType(num_points_index_) != nvinfer1::DataType::kINT32 ||
        engine_->getBindingDataType(coors_index_) != nvinfer1::DataType::kINT32 ||
        engine_->getBindingDataType(bbox_pred_index_) != nvinfer1::DataType::kFLOAT ||
        engine_->getBindingDataType(cls_score_index_) != nvinfer1::DataType::kFLOAT ||
        engine_->getBindingDataType(dir_cls_pred_index_) != nvinfer1::DataType::kFLOAT) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "Engine binding dtypes do not match expected float/int32 contract");
    }

    InferenceStatus status = ValidateInputProfiles(config, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }

    const nvinfer1::Dims bbox_dims = engine_->getBindingDimensions(bbox_pred_index_);
    const nvinfer1::Dims cls_dims = engine_->getBindingDimensions(cls_score_index_);
    const nvinfer1::Dims dir_dims = engine_->getBindingDimensions(dir_cls_pred_index_);
    if (bbox_dims.nbDims != 4 || cls_dims.nbDims != 4 || dir_dims.nbDims != 4) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "Engine outputs must be 4D BCHW tensors");
    }

    bbox_channels_ = bbox_dims.d[1];
    cls_channels_ = cls_dims.d[1];
    dir_channels_ = dir_dims.d[1];
    height_ = cls_dims.d[2];
    width_ = cls_dims.d[3];
    if (bbox_dims.d[2] != height_ || bbox_dims.d[3] != width_ || dir_dims.d[2] != height_ ||
        dir_dims.d[3] != width_) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "Engine output feature map shapes do not match");
    }
    if (bbox_channels_ <= 0 || cls_channels_ <= 0 || dir_channels_ <= 0 ||
        bbox_channels_ != cls_channels_ * kBBoxChannelsPerTask ||
        dir_channels_ != cls_channels_ * kDirChannelsPerTask) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "Engine channels do not match expected 3-task CenterPoint head layout");
    }
    return INFERENCE_STATUS_OK;
  }

  InferenceStatus ConfigureShapes(const RuntimeConfig& config,
                                  int num_pillars,
                                  std::string* out_error) {
    if (num_pillars < min_runtime_pillars_ || num_pillars > max_runtime_pillars_) {
      std::ostringstream oss;
      oss << "Pillar count " << num_pillars << " is outside supported runtime/profile range ["
          << min_runtime_pillars_ << ", " << max_runtime_pillars_ << "]";
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

    if (!context_->setBindingDimensions(voxels_index_, voxels_dims) ||
        !context_->setBindingDimensions(num_points_index_, num_points_dims) ||
        !context_->setBindingDimensions(coors_index_, coors_dims) ||
        !context_->allInputDimensionsSpecified()) {
      std::ostringstream oss;
      oss << "Failed to set TensorRT binding dimensions for pillar count " << num_pillars
          << " within optimization profile [" << min_profile_pillars_ << ", "
          << max_profile_pillars_ << "]";
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
    }
    return INFERENCE_STATUS_OK;
  }

  InferenceStatus Enqueue(void** bindings,
                          cudaStream_t stream,
                          std::string* out_error) {
    if (!context_->enqueueV2(bindings, stream, nullptr)) {
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                      "TensorRT enqueueV2 failed");
    }
    return INFERENCE_STATUS_OK;
  }

  int task_count() const { return cls_channels_; }
  int bbox_channels() const { return bbox_channels_; }
  int cls_channels() const { return cls_channels_; }
  int dir_channels() const { return dir_channels_; }
  int height() const { return height_; }
  int width() const { return width_; }
  int num_bindings() const { return engine_->getNbBindings(); }
  int min_runtime_pillars() const { return min_runtime_pillars_; }
  int max_runtime_pillars() const { return max_runtime_pillars_; }
  int voxels_index() const { return voxels_index_; }
  int num_points_index() const { return num_points_index_; }
  int coors_index() const { return coors_index_; }
  int bbox_pred_index() const { return bbox_pred_index_; }
  int cls_score_index() const { return cls_score_index_; }
  int dir_cls_pred_index() const { return dir_cls_pred_index_; }

 private:
  static std::string DimsToString(const nvinfer1::Dims& dims) {
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

  InferenceStatus ValidateInputProfile(const char* tensor_name,
                                       int expected_nb_dims,
                                       const std::array<int, 3>& expected_static_dims,
                                       int* out_min_pillars,
                                       int* out_opt_pillars,
                                       int* out_max_pillars,
                                       std::string* out_error) const {
    const nvinfer1::Dims min_dims =
        engine_->getProfileShape(tensor_name, kProfileIndex, nvinfer1::OptProfileSelector::kMIN);
    const nvinfer1::Dims opt_dims =
        engine_->getProfileShape(tensor_name, kProfileIndex, nvinfer1::OptProfileSelector::kOPT);
    const nvinfer1::Dims max_dims =
        engine_->getProfileShape(tensor_name, kProfileIndex, nvinfer1::OptProfileSelector::kMAX);

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
        ValidateInputProfile(kVoxelsName, 3, {0, config.max_points_per_pillar, config.num_point_features},
                             &voxels_min, &voxels_opt, &voxels_max, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = ValidateInputProfile(kNumPointsName, 1, {0, 0, 0},
                                  &num_points_min, &num_points_opt, &num_points_max, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = ValidateInputProfile(kCoorsName, 2, {0, 4, 0},
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

    min_profile_pillars_ = voxels_min;
    opt_profile_pillars_ = voxels_opt;
    max_profile_pillars_ = voxels_max;
    min_runtime_pillars_ = std::max(config.min_pillars, min_profile_pillars_);
    max_runtime_pillars_ = std::min(config.max_pillars, max_profile_pillars_);
    if (min_runtime_pillars_ > max_runtime_pillars_) {
      std::ostringstream oss;
      oss << "Runtime pillar range [" << config.min_pillars << ", " << config.max_pillars
          << "] does not overlap TensorRT optimization profile [" << min_profile_pillars_ << ", "
          << max_profile_pillars_ << "]";
      return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
    }
    return INFERENCE_STATUS_OK;
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

  TrtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime, RuntimeDeleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, ContextDeleter> context_;
  int voxels_index_ = -1;
  int num_points_index_ = -1;
  int coors_index_ = -1;
  int bbox_pred_index_ = -1;
  int cls_score_index_ = -1;
  int dir_cls_pred_index_ = -1;
  int bbox_channels_ = 0;
  int cls_channels_ = 0;
  int dir_channels_ = 0;
  int height_ = 0;
  int width_ = 0;
  int min_profile_pillars_ = 0;
  int opt_profile_pillars_ = 0;
  int max_profile_pillars_ = 0;
  int min_runtime_pillars_ = 0;
  int max_runtime_pillars_ = 0;
};

InferenceStatus CheckCuda(cudaError_t error,
                          const std::string& operation,
                          std::string* out_error) {
  if (error == cudaSuccess) {
    return INFERENCE_STATUS_OK;
  }
  std::ostringstream oss;
  oss << operation << " failed: " << cudaGetErrorString(error);
  return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR, oss.str());
}

struct HeapEntry {
  float score = 0.0f;
  int index = 0;
};

struct MinScoreCompare {
  bool operator()(const HeapEntry& lhs, const HeapEntry& rhs) const {
    return lhs.score > rhs.score;
  }
};

InferenceStatus BuildPillars(const RuntimeConfig& config,
                             int num_pillars,
                             float* d_voxels,
                             const unsigned int* d_num_points,
                             const unsigned int* d_coors,
                             PillarInputTensors* out_tensors,
                             std::string* out_error) {
  if (out_tensors == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "out_tensors is null");
  }
  if (num_pillars < 0 || num_pillars > config.max_pillars || d_voxels == nullptr ||
      d_num_points == nullptr || d_coors == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Invalid device pillar tensors");
  }

  const int voxel_stride = config.max_points_per_pillar * config.num_point_features;
  out_tensors->num_pillars = num_pillars;
  out_tensors->voxels.resize(static_cast<size_t>(num_pillars) * voxel_stride);
  out_tensors->num_points.resize(num_pillars);
  out_tensors->coors.resize(static_cast<size_t>(num_pillars) * 4);

  InferenceStatus status =
      CheckCuda(cudaMemcpy(out_tensors->voxels.data(), d_voxels,
                           out_tensors->voxels.size() * sizeof(float),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy(pillar_voxels)", out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status =
      CheckCuda(cudaMemcpy(out_tensors->num_points.data(), d_num_points,
                           out_tensors->num_points.size() * sizeof(int32_t),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy(pillar_num_points)", out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status =
      CheckCuda(cudaMemcpy(out_tensors->coors.data(), d_coors,
                           out_tensors->coors.size() * sizeof(int32_t),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy(pillar_coors)", out_error);
  return status;
}

InferenceStatus DecodeOutputs(const RuntimeConfig& config,
                              const RawHeadView& raw_outputs,
                              CandidateDetections* out_candidates,
                              std::string* out_error) {
  if (out_candidates == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "out_candidates is null");
  }

  out_candidates->scores.clear();
  out_candidates->boxes.clear();
  out_candidates->labels.clear();

  const int task_count = raw_outputs.cls_channels;
  const int height = raw_outputs.height;
  const int width = raw_outputs.width;
  const int hw = height * width;
  if (task_count <= 0 || height <= 0 || width <= 0) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Raw outputs have invalid dimensions");
  }

  out_candidates->scores.reserve(static_cast<size_t>(task_count) * config.bbox_max_num);
  out_candidates->boxes.reserve(static_cast<size_t>(task_count) * config.bbox_max_num * kBoxDim);
  out_candidates->labels.reserve(static_cast<size_t>(task_count) * config.bbox_max_num);

  for (int task_id = 0; task_id < task_count; ++task_id) {
    const float* heat = raw_outputs.cls_score + static_cast<size_t>(task_id) * hw;
    const float* bbox = raw_outputs.bbox_pred +
                        static_cast<size_t>(task_id) * kBBoxChannelsPerTask * hw;
    const float* dir = raw_outputs.dir_cls_pred +
                       static_cast<size_t>(task_id) * kDirChannelsPerTask * hw;

    std::priority_queue<HeapEntry, std::vector<HeapEntry>, MinScoreCompare> topk_heap;
    for (int index = 0; index < hw; ++index) {
      if (!std::isfinite(heat[index])) {
        continue;
      }
      const float score = Sigmoid(heat[index]);
      if (!std::isfinite(score)) {
        continue;
      }
      if (topk_heap.size() < static_cast<size_t>(config.bbox_max_num)) {
        topk_heap.push({score, index});
      } else if (score > topk_heap.top().score) {
        topk_heap.pop();
        topk_heap.push({score, index});
      }
    }

    std::vector<HeapEntry> topk;
    topk.reserve(topk_heap.size());
    while (!topk_heap.empty()) {
      topk.push_back(topk_heap.top());
      topk_heap.pop();
    }
    std::sort(topk.begin(), topk.end(),
              [](const HeapEntry& lhs, const HeapEntry& rhs) { return lhs.score > rhs.score; });

    for (const HeapEntry& entry : topk) {
      if (!std::isfinite(entry.score) || entry.score <= config.score_threshold) {
        continue;
      }
      const int y_idx = entry.index / width;
      const int x_idx = entry.index % width;
      const float reg_x = bbox[0 * hw + entry.index];
      const float reg_y = bbox[1 * hw + entry.index];
      const float height_value = bbox[2 * hw + entry.index];
      const float dim_x = SafeExp(bbox[3 * hw + entry.index]);
      const float dim_y = SafeExp(bbox[4 * hw + entry.index]);
      const float dim_z = SafeExp(bbox[5 * hw + entry.index]);
      const float rot_sine = dir[0 * hw + entry.index];
      const float rot_cosine = dir[1 * hw + entry.index];
      const float yaw = std::atan2(rot_sine, rot_cosine);

      const float center_x =
          (static_cast<float>(x_idx) + reg_x) * config.out_size_factor * config.pillar_size[0] +
          config.point_cloud_range[0];
      const float center_y =
          (static_cast<float>(y_idx) + reg_y) * config.out_size_factor * config.pillar_size[1] +
          config.point_cloud_range[1];
      const float center_z = height_value - dim_z * 0.5f;
      if (!std::isfinite(center_x) || !std::isfinite(center_y) || !std::isfinite(center_z) ||
          !std::isfinite(dim_x) || !std::isfinite(dim_y) || !std::isfinite(dim_z) ||
          !std::isfinite(yaw)) {
        continue;
      }

      if (center_x < config.post_center_range[0] || center_x > config.post_center_range[3] ||
          center_y < config.post_center_range[1] || center_y > config.post_center_range[4] ||
          center_z < config.post_center_range[2] || center_z > config.post_center_range[5]) {
        continue;
      }

      out_candidates->scores.push_back(entry.score);
      out_candidates->labels.push_back(task_id + 1);
      out_candidates->boxes.push_back(center_x);
      out_candidates->boxes.push_back(center_y);
      out_candidates->boxes.push_back(center_z);
      out_candidates->boxes.push_back(dim_x);
      out_candidates->boxes.push_back(dim_y);
      out_candidates->boxes.push_back(dim_z);
      out_candidates->boxes.push_back(yaw);
    }
  }

  std::vector<size_t> order(out_candidates->scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
    return out_candidates->scores[lhs] > out_candidates->scores[rhs];
  });

  CandidateDetections sorted;
  sorted.scores.reserve(order.size());
  sorted.labels.reserve(order.size());
  sorted.boxes.reserve(order.size() * kBoxDim);
  for (size_t index : order) {
    sorted.scores.push_back(out_candidates->scores[index]);
    sorted.labels.push_back(out_candidates->labels[index]);
    const size_t box_offset = index * kBoxDim;
    sorted.boxes.insert(sorted.boxes.end(),
                        out_candidates->boxes.begin() + static_cast<long>(box_offset),
                        out_candidates->boxes.begin() + static_cast<long>(box_offset + kBoxDim));
  }
  *out_candidates = std::move(sorted);
  return INFERENCE_STATUS_OK;
}

InferenceStatus CopyOutputsToResponse(const CandidateDetections& candidates,
                                      InferenceResponse* response,
                                      std::string* out_error) {
  if (response == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "response is null");
  }
  const size_t candidate_count = candidates.scores.size();
  const size_t box_count = candidate_count * kBoxDim;
  if (response->cls_capacity < candidate_count || response->label_capacity < candidate_count ||
      response->box_capacity < box_count) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Response capacity is smaller than decoded output size");
  }
  if ((candidate_count > 0 && (response->cls_preds == nullptr || response->box_preds == nullptr ||
                               response->label_preds == nullptr)) ||
      (response->freespace_capacity > 0 && response->freespace == nullptr)) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Response buffers contain null pointers");
  }

  if (candidate_count > 0) {
    std::memcpy(response->cls_preds, candidates.scores.data(), candidate_count * sizeof(float));
    std::memcpy(response->label_preds, candidates.labels.data(), candidate_count * sizeof(int));
    std::memcpy(response->box_preds, candidates.boxes.data(), box_count * sizeof(float));
  }
  if (response->freespace != nullptr && response->freespace_capacity > 0) {
    std::fill(response->freespace, response->freespace + response->freespace_capacity, 0.0f);
  }
  response->cls_count = candidate_count;
  response->label_count = candidate_count;
  response->box_count = box_count;
  response->freespace_count = 0;
  return INFERENCE_STATUS_OK;
}

}  // namespace

InferenceStatus ParseRuntimeConfig(const InferenceConfig& config,
                                   RuntimeConfig* out_config,
                                   std::string* out_error) {
  if (out_config == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "out_config is null");
  }
  if (config.end2end_file == nullptr || std::strlen(config.end2end_file) == 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "end2end_file is empty");
  }
  if (!IsFiniteArray(config.point_cloud_range, 6) || !IsFiniteArray(config.post_center_range, 6) ||
      !IsFiniteArray(config.pillar_size, 3)) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Config contains non-finite numeric values");
  }
  if (config.point_cloud_range[0] >= config.point_cloud_range[3] ||
      config.point_cloud_range[1] >= config.point_cloud_range[4] ||
      config.point_cloud_range[2] >= config.point_cloud_range[5]) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "point_cloud_range is invalid");
  }
  if (config.post_center_range[0] >= config.post_center_range[3] ||
      config.post_center_range[1] >= config.post_center_range[4] ||
      config.post_center_range[2] >= config.post_center_range[5]) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "post_center_range is invalid");
  }
  if (config.pillar_size[0] <= 0.0f || config.pillar_size[1] <= 0.0f ||
      config.pillar_size[2] <= 0.0f || config.max_points_per_pillar <= 0 ||
      config.min_pillars <= 0 || config.max_pillars < config.min_pillars ||
      config.max_points <= 0 || config.num_point_features < 3 || config.bbox_max_num <= 0 ||
      config.out_size_factor <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Config contains non-positive limits");
  }

  out_config->end2end_file = config.end2end_file;
  std::copy(config.point_cloud_range, config.point_cloud_range + 6,
            out_config->point_cloud_range.begin());
  std::copy(config.post_center_range, config.post_center_range + 6,
            out_config->post_center_range.begin());
  std::copy(config.pillar_size, config.pillar_size + 3, out_config->pillar_size.begin());
  out_config->max_points_per_pillar = config.max_points_per_pillar;
  out_config->min_pillars = config.min_pillars;
  out_config->max_pillars = config.max_pillars;
  out_config->max_points = config.max_points;
  out_config->frame_num = std::max(config.frame_num, 1);
  out_config->num_point_features = config.num_point_features;
  out_config->bbox_max_num = config.bbox_max_num;
  out_config->out_size_factor = config.out_size_factor;
  out_config->score_threshold = config.score_threshold;
  return INFERENCE_STATUS_OK;
}

InferenceOutputSpec MakeOutputSpec(int task_count, int bbox_max_num) {
  InferenceOutputSpec spec{};
  spec.max_candidates = static_cast<size_t>(task_count) * static_cast<size_t>(bbox_max_num);
  spec.cls_count = spec.max_candidates;
  spec.box_count = spec.max_candidates * kBoxDim;
  spec.label_count = spec.max_candidates;
  spec.freespace_count = 0;
  return spec;
}

struct PillarRuntime::Impl {
  TrtEngine engine;
  std::shared_ptr<Voxelization> voxelizer;
  cudaStream_t stream = nullptr;
  float* d_points = nullptr;
  float* d_bbox_pred = nullptr;
  float* d_cls_score = nullptr;
  float* d_dir_cls_pred = nullptr;
  std::vector<float> h_bbox_pred;
  std::vector<float> h_cls_score;
  std::vector<float> h_dir_cls_pred;

  ~Impl() {
    if (d_dir_cls_pred != nullptr) {
      cudaFree(d_dir_cls_pred);
    }
    if (d_cls_score != nullptr) {
      cudaFree(d_cls_score);
    }
    if (d_bbox_pred != nullptr) {
      cudaFree(d_bbox_pred);
    }
    if (d_points != nullptr) {
      cudaFree(d_points);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};

PillarRuntime::PillarRuntime(RuntimeConfig config) : config_(std::move(config)) {}

PillarRuntime::~PillarRuntime() = default;

InferenceStatus PillarRuntime::Initialize() {
  impl_ = std::make_unique<Impl>();
  InferenceStatus status = impl_->engine.Load(config_, config_.end2end_file, &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  output_spec_ = MakeOutputSpec(impl_->engine.task_count(), config_.bbox_max_num);

  status = CheckCuda(cudaStreamCreate(&impl_->stream), "cudaStreamCreate", &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  VoxelizationParameter voxelization_param{};
  voxelization_param.min_range = nvtype::Float3(config_.point_cloud_range[0],
                                                config_.point_cloud_range[1],
                                                config_.point_cloud_range[2]);
  voxelization_param.max_range = nvtype::Float3(config_.point_cloud_range[3],
                                                config_.point_cloud_range[4],
                                                config_.point_cloud_range[5]);
  voxelization_param.voxel_size = nvtype::Float3(config_.pillar_size[0],
                                                 config_.pillar_size[1],
                                                 config_.pillar_size[2]);
  voxelization_param.grid_size = VoxelizationParameter::compute_grid_size(
      voxelization_param.max_range, voxelization_param.min_range, voxelization_param.voxel_size);
  voxelization_param.num_feature = config_.num_point_features;
  voxelization_param.max_voxels = config_.max_pillars;
  voxelization_param.max_points_per_voxel = config_.max_points_per_pillar;
  voxelization_param.max_points = config_.max_points;
  impl_->voxelizer = create_voxelization(voxelization_param);
  if (!impl_->voxelizer) {
    return SetError(&last_error_, INFERENCE_STATUS_BACKEND_ERROR,
                    "Failed to create CUDA voxelizer");
  }

  const size_t max_points_size = static_cast<size_t>(config_.max_points) *
                                 static_cast<size_t>(config_.num_point_features);
  const size_t bbox_size = static_cast<size_t>(impl_->engine.bbox_channels()) *
                           static_cast<size_t>(impl_->engine.height()) *
                           static_cast<size_t>(impl_->engine.width());
  const size_t cls_size = static_cast<size_t>(impl_->engine.cls_channels()) *
                          static_cast<size_t>(impl_->engine.height()) *
                          static_cast<size_t>(impl_->engine.width());
  const size_t dir_size = static_cast<size_t>(impl_->engine.dir_channels()) *
                          static_cast<size_t>(impl_->engine.height()) *
                          static_cast<size_t>(impl_->engine.width());

  status = CheckCuda(cudaMalloc(reinterpret_cast<void**>(&impl_->d_points),
                                max_points_size * sizeof(float)),
                     "cudaMalloc(d_points)", &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = CheckCuda(cudaMalloc(reinterpret_cast<void**>(&impl_->d_bbox_pred),
                                bbox_size * sizeof(float)),
                     "cudaMalloc(d_bbox_pred)", &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = CheckCuda(cudaMalloc(reinterpret_cast<void**>(&impl_->d_cls_score),
                                cls_size * sizeof(float)),
                     "cudaMalloc(d_cls_score)", &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = CheckCuda(cudaMalloc(reinterpret_cast<void**>(&impl_->d_dir_cls_pred),
                                dir_size * sizeof(float)),
                     "cudaMalloc(d_dir_cls_pred)", &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  impl_->h_bbox_pred.resize(bbox_size);
  impl_->h_cls_score.resize(cls_size);
  impl_->h_dir_cls_pred.resize(dir_size);
  return INFERENCE_STATUS_OK;
}

InferenceStatus PillarRuntime::Forward(const InferenceRequest& request,
                                       InferenceResponse* response,
                                       DebugArtifacts* debug,
                                       StageTimings* timings) {
  last_error_.clear();
  if (!impl_) {
    return SetError(&last_error_, INFERENCE_STATUS_NOT_READY,
                    "Runtime is not initialized");
  }
  if (request.point_count < 0 || (request.point_count > 0 && request.points == nullptr)) {
    return SetError(&last_error_, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Invalid point cloud request");
  }
  const int bounded_point_count = std::max(0, std::min(request.point_count, config_.max_points));

  StageTimings local_timings;
  const auto total_start = std::chrono::steady_clock::now();
  if (debug != nullptr && bounded_point_count > 0) {
    debug->input_points.assign(
        request.points,
        request.points + static_cast<size_t>(bounded_point_count) * config_.num_point_features);
  } else if (debug != nullptr) {
    debug->input_points.clear();
  }

  cudaEvent_t gpu_start = nullptr;
  cudaEvent_t gpu_end = nullptr;
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_end);

  InferenceStatus status = INFERENCE_STATUS_OK;
  float elapsed_ms = 0.0f;
  if (bounded_point_count > 0) {
    const size_t input_bytes = static_cast<size_t>(bounded_point_count) *
                               static_cast<size_t>(config_.num_point_features) *
                               sizeof(float);
    cudaEventRecord(gpu_start, impl_->stream);
    status = CheckCuda(cudaMemcpyAsync(impl_->d_points, request.points, input_bytes,
                                       cudaMemcpyHostToDevice, impl_->stream),
                       "cudaMemcpyAsync(points)", &last_error_);
    cudaEventRecord(gpu_end, impl_->stream);
    cudaEventSynchronize(gpu_end);
    cudaEventElapsedTime(&elapsed_ms, gpu_start, gpu_end);
    local_timings.h2d_prepare_ms = elapsed_ms;
    if (status != INFERENCE_STATUS_OK) {
      cudaEventDestroy(gpu_start);
      cudaEventDestroy(gpu_end);
      return status;
    }
  }

  cudaEventRecord(gpu_start, impl_->stream);
  impl_->voxelizer->forward_pillars(impl_->d_points, bounded_point_count,
                                    CoordinateOrder::ZYX, impl_->stream);
  cudaEventRecord(gpu_end, impl_->stream);
  cudaEventSynchronize(gpu_end);
  cudaEventElapsedTime(&elapsed_ms, gpu_start, gpu_end);
  local_timings.pillarize_ms = elapsed_ms;

  float* d_voxels = nullptr;
  unsigned int* d_num_points = nullptr;
  unsigned int* d_coors = nullptr;
  const int actual_num_pillars =
      static_cast<int>(impl_->voxelizer->get_pillar_output(&d_voxels, &d_num_points, &d_coors));
  const int padded_num_pillars =
      std::min(impl_->engine.max_runtime_pillars(),
               std::max(actual_num_pillars, impl_->engine.min_runtime_pillars()));

  if (debug != nullptr) {
    PillarInputTensors pillar_inputs;
    status = BuildPillars(config_, padded_num_pillars, d_voxels, d_num_points, d_coors,
                          &pillar_inputs, &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      cudaEventDestroy(gpu_start);
      cudaEventDestroy(gpu_end);
      return status;
    }
    debug->pillar_inputs = std::move(pillar_inputs);
  }

  status = impl_->engine.ConfigureShapes(config_, padded_num_pillars, &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_end);
    return status;
  }

  void* bindings[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  bindings[impl_->engine.voxels_index()] = d_voxels;
  bindings[impl_->engine.num_points_index()] = d_num_points;
  bindings[impl_->engine.coors_index()] = d_coors;
  bindings[impl_->engine.bbox_pred_index()] = impl_->d_bbox_pred;
  bindings[impl_->engine.cls_score_index()] = impl_->d_cls_score;
  bindings[impl_->engine.dir_cls_pred_index()] = impl_->d_dir_cls_pred;

  cudaEventRecord(gpu_start, impl_->stream);
  status = impl_->engine.Enqueue(bindings, impl_->stream, &last_error_);
  cudaEventRecord(gpu_end, impl_->stream);
  cudaEventSynchronize(gpu_end);
  cudaEventElapsedTime(&elapsed_ms, gpu_start, gpu_end);
  local_timings.trt_forward_ms = elapsed_ms;
  if (status != INFERENCE_STATUS_OK) {
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_end);
    return status;
  }

  cudaEventRecord(gpu_start, impl_->stream);
  status = CheckCuda(cudaMemcpyAsync(impl_->h_bbox_pred.data(), impl_->d_bbox_pred,
                                     impl_->h_bbox_pred.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost, impl_->stream),
                     "cudaMemcpyAsync(bbox_pred)", &last_error_);
  if (status == INFERENCE_STATUS_OK) {
    status = CheckCuda(cudaMemcpyAsync(impl_->h_cls_score.data(), impl_->d_cls_score,
                                       impl_->h_cls_score.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, impl_->stream),
                       "cudaMemcpyAsync(cls_score)", &last_error_);
  }
  if (status == INFERENCE_STATUS_OK) {
    status = CheckCuda(cudaMemcpyAsync(impl_->h_dir_cls_pred.data(), impl_->d_dir_cls_pred,
                                       impl_->h_dir_cls_pred.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, impl_->stream),
                       "cudaMemcpyAsync(dir_cls_pred)", &last_error_);
  }
  cudaEventRecord(gpu_end, impl_->stream);
  cudaEventSynchronize(gpu_end);
  cudaEventElapsedTime(&elapsed_ms, gpu_start, gpu_end);
  local_timings.d2h_collect_ms = elapsed_ms;
  cudaEventDestroy(gpu_start);
  cudaEventDestroy(gpu_end);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  RawHeadView raw_outputs;
  raw_outputs.bbox_pred = impl_->h_bbox_pred.data();
  raw_outputs.cls_score = impl_->h_cls_score.data();
  raw_outputs.dir_cls_pred = impl_->h_dir_cls_pred.data();
  raw_outputs.bbox_channels = impl_->engine.bbox_channels();
  raw_outputs.cls_channels = impl_->engine.cls_channels();
  raw_outputs.dir_channels = impl_->engine.dir_channels();
  raw_outputs.height = impl_->engine.height();
  raw_outputs.width = impl_->engine.width();
  if (debug != nullptr) {
    debug->raw_outputs.bbox_pred = impl_->h_bbox_pred;
    debug->raw_outputs.cls_score = impl_->h_cls_score;
    debug->raw_outputs.dir_cls_pred = impl_->h_dir_cls_pred;
    debug->raw_outputs.bbox_channels = raw_outputs.bbox_channels;
    debug->raw_outputs.cls_channels = raw_outputs.cls_channels;
    debug->raw_outputs.dir_channels = raw_outputs.dir_channels;
    debug->raw_outputs.height = raw_outputs.height;
    debug->raw_outputs.width = raw_outputs.width;
  }

  CandidateDetections candidates;
  auto start = std::chrono::steady_clock::now();
  status = DecodeOutputs(config_, raw_outputs, &candidates, &last_error_);
  auto end = std::chrono::steady_clock::now();
  local_timings.decode_ms = ElapsedMs(start, end);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  if (debug != nullptr) {
    debug->decoded_candidates = candidates;
  }

  start = std::chrono::steady_clock::now();
  status = CopyOutputsToResponse(candidates, response, &last_error_);
  end = std::chrono::steady_clock::now();
  local_timings.output_pack_ms = ElapsedMs(start, end);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  local_timings.total_forward_ms = ElapsedMs(total_start, std::chrono::steady_clock::now());
  if (timings != nullptr) {
    *timings = local_timings;
  }
  if (debug != nullptr) {
    debug->timings = local_timings;
  }
  return INFERENCE_STATUS_OK;
}

void PillarRuntime::Reset() {
  last_error_.clear();
}

}  // namespace inference_internal
