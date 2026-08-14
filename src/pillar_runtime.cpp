#include "pillar_runtime.hpp"
#include "pillar_postprocess/pillar_postprocess.hpp"
#include "trt_engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <cuda_runtime_api.h>

#include "iou3d_nms_api.h"
#include "voxelization.hpp"

namespace inference_internal {
namespace {

constexpr int kBoxDim = 7;
constexpr int kDirChannelsPerTask = 2;
constexpr int kLabelCount = 3;
constexpr float kPi = 3.14159265358979323846f;

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

float WrapAnglePi(float yaw) {
  if (!std::isfinite(yaw)) {
    return yaw;
  }
  float wrapped = std::fmod(yaw + kPi, 2.0f * kPi);
  if (wrapped < 0.0f) {
    wrapped += 2.0f * kPi;
  }
  return wrapped - kPi;
}

float DecodeYaw(const ResolvedHeadLayout& layout,
                const float* bbox,
                const float* dir,
                int hw,
                int index) {
  if (layout.theta_offset >= 0) {
    const float theta = bbox[layout.theta_offset * hw + index];
    const float dir0 = dir[0 * hw + index];
    const float dir1 = dir[1 * hw + index];
    if (!std::isfinite(theta) || !std::isfinite(dir0) || !std::isfinite(dir1)) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    const int dir_label = dir1 > dir0 ? 1 : 0;
    return WrapAnglePi(theta + static_cast<float>(dir_label) * kPi);
  }
  const float rot_sine = dir[0 * hw + index];
  const float rot_cosine = dir[1 * hw + index];
  if (!std::isfinite(rot_sine) || !std::isfinite(rot_cosine)) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return std::atan2(rot_sine, rot_cosine);
}

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

template <typename T>
InferenceStatus AllocatePinnedHostBuffer(size_t count,
                                         const std::string& name,
                                         T** out_buffer,
                                         std::string* out_error) {
  if (out_buffer == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    name + " output buffer pointer is null");
  }
  *out_buffer = nullptr;
  if (count == 0) {
    return INFERENCE_STATUS_OK;
  }
  return CheckCuda(cudaMallocHost(reinterpret_cast<void**>(out_buffer), count * sizeof(T)),
                   "cudaMallocHost(" + name + ")", out_error);
}

template <typename T>
void FreePinnedHostBuffer(T** buffer) {
  if (buffer != nullptr && *buffer != nullptr) {
    cudaFreeHost(*buffer);
    *buffer = nullptr;
  }
}

InferenceStatus SyncCudaStream(cudaStream_t stream,
                               const std::string& operation,
                               std::string* out_error) {
  return CheckCuda(cudaStreamSynchronize(stream), operation, out_error);
}

InferenceStatus MeasureGpuStageBegin(cudaEvent_t start_event,
                                     cudaStream_t stream,
                                     bool enabled,
                                     std::string* out_error) {
  if (!enabled) {
    return INFERENCE_STATUS_OK;
  }
  return CheckCuda(cudaEventRecord(start_event, stream), "cudaEventRecord(stage_start)", out_error);
}

InferenceStatus MeasureGpuStageEnd(cudaEvent_t start_event,
                                   cudaEvent_t end_event,
                                   cudaStream_t stream,
                                   bool enabled,
                                   double* out_ms,
                                   std::string* out_error) {
  if (!enabled || out_ms == nullptr) {
    return INFERENCE_STATUS_OK;
  }
  InferenceStatus status =
      CheckCuda(cudaEventRecord(end_event, stream), "cudaEventRecord(stage_end)", out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = CheckCuda(cudaEventSynchronize(end_event), "cudaEventSynchronize(stage_end)", out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  float elapsed_ms = 0.0f;
  status = CheckCuda(cudaEventElapsedTime(&elapsed_ms, start_event, end_event),
                     "cudaEventElapsedTime(stage)", out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  *out_ms = static_cast<double>(elapsed_ms);
  return INFERENCE_STATUS_OK;
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

struct HeatmapEntry {
  float score = 0.0f;
  int class_index = 0;
  int spatial_index = 0;
};

struct MinHeatmapScoreCompare {
  bool operator()(const HeatmapEntry& lhs, const HeatmapEntry& rhs) const {
    return lhs.score > rhs.score;
  }
};

void ApplyPreMaxsize(std::vector<float>* scores,
                     std::vector<float>* boxes,
                     std::vector<int>* labels,
                     int pre_maxsize) {
  if (pre_maxsize <= 0 || scores == nullptr || boxes == nullptr || labels == nullptr ||
      scores->size() <= static_cast<size_t>(pre_maxsize)) {
    return;
  }
  std::vector<size_t> indices(scores->size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&](size_t lhs, size_t rhs) { return (*scores)[lhs] > (*scores)[rhs]; });

  std::vector<float> new_scores;
  std::vector<float> new_boxes;
  std::vector<int> new_labels;
  new_scores.reserve(static_cast<size_t>(pre_maxsize));
  new_boxes.reserve(static_cast<size_t>(pre_maxsize) * kBoxDim);
  new_labels.reserve(static_cast<size_t>(pre_maxsize));
  for (int i = 0; i < pre_maxsize; ++i) {
    const size_t src = indices[static_cast<size_t>(i)];
    new_scores.push_back((*scores)[src]);
    new_labels.push_back((*labels)[src]);
    new_boxes.insert(new_boxes.end(), boxes->begin() + static_cast<long>(src * kBoxDim),
                     boxes->begin() + static_cast<long>((src + 1) * kBoxDim));
  }
  scores->swap(new_scores);
  boxes->swap(new_boxes);
  labels->swap(new_labels);
}

void AppendBoxes(const std::vector<float>& source_boxes,
                 size_t source_index,
                 std::vector<float>* target_boxes) {
  target_boxes->insert(target_boxes->end(),
                       source_boxes.begin() + static_cast<long>(source_index * kBoxDim),
                       source_boxes.begin() + static_cast<long>((source_index + 1) * kBoxDim));
}

void FilterByScoreThreshold(CandidateDetections* detections, float score_threshold) {
  if (detections == nullptr || score_threshold <= 0.0f) {
    return;
  }

  CandidateDetections filtered;
  filtered.scores.reserve(detections->scores.size());
  filtered.labels.reserve(detections->labels.size());
  filtered.boxes.reserve(detections->boxes.size());
  for (size_t i = 0; i < detections->scores.size(); ++i) {
    if (detections->scores[i] < score_threshold) {
      continue;
    }
    filtered.scores.push_back(detections->scores[i]);
    filtered.labels.push_back(detections->labels[i]);
    AppendBoxes(detections->boxes, i, &filtered.boxes);
  }
  detections->scores.swap(filtered.scores);
  detections->labels.swap(filtered.labels);
  detections->boxes.swap(filtered.boxes);
}

void FilterByLabelScoreThreshold(CandidateDetections* detections,
                                 const std::vector<float>& label_score_thresh) {
  if (detections == nullptr || label_score_thresh.empty()) {
    return;
  }

  CandidateDetections filtered;
  filtered.scores.reserve(detections->scores.size());
  filtered.labels.reserve(detections->labels.size());
  filtered.boxes.reserve(detections->boxes.size());
  for (size_t i = 0; i < detections->scores.size(); ++i) {
    const int label = detections->labels[i];
    if (label >= 0 && label < static_cast<int>(label_score_thresh.size()) &&
        detections->scores[i] < label_score_thresh[static_cast<size_t>(label)]) {
      continue;
    }
    filtered.scores.push_back(detections->scores[i]);
    filtered.labels.push_back(label);
    AppendBoxes(detections->boxes, i, &filtered.boxes);
  }
  detections->scores.swap(filtered.scores);
  detections->labels.swap(filtered.labels);
  detections->boxes.swap(filtered.boxes);
}

void SortCandidatesDescending(CandidateDetections* out_candidates);

void CollectByLabelSet(const CandidateDetections& candidates,
                       const std::vector<int>& labels,
                       int label_count,
                       CandidateDetections* out_group) {
  out_group->scores.clear();
  out_group->boxes.clear();
  out_group->labels.clear();
  if (labels.empty() || label_count <= 0) {
    return;
  }

  std::vector<uint8_t> selected(static_cast<size_t>(label_count), 0);
  for (const int label : labels) {
    if (label >= 0 && label < label_count) {
      selected[static_cast<size_t>(label)] = 1U;
    }
  }

  out_group->scores.reserve(candidates.scores.size());
  out_group->labels.reserve(candidates.labels.size());
  out_group->boxes.reserve(candidates.boxes.size());
  for (size_t i = 0; i < candidates.scores.size(); ++i) {
    const int label = candidates.labels[i];
    if (label < 0 || label >= label_count || selected[static_cast<size_t>(label)] == 0U) {
      continue;
    }
    out_group->scores.push_back(candidates.scores[i]);
    out_group->labels.push_back(label);
    AppendBoxes(candidates.boxes, i, &out_group->boxes);
  }
}

InferenceStatus AppendCircleNmsResult(CandidateDetections* detections,
                                      float distance_threshold,
                                      int post_maxsize,
                                      CandidateDetections* out_result,
                                      std::string* out_error) {
  if (detections == nullptr || out_result == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Circle NMS input or output is null");
  }
  if (distance_threshold < 0.0f || post_maxsize <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Circle NMS group config is invalid");
  }
  if (detections->scores.empty()) {
    return INFERENCE_STATUS_OK;
  }

  SortCandidatesDescending(detections);
  std::vector<long> keep(detections->scores.size(), 0);
  const int keep_count =
      circle_nms_gpu(detections->boxes.data(), keep.data(), distance_threshold,
                     static_cast<int>(detections->scores.size()));
  const size_t final_count =
      std::min(static_cast<size_t>(keep_count), static_cast<size_t>(post_maxsize));
  for (size_t i = 0; i < final_count; ++i) {
    const size_t src = static_cast<size_t>(keep[i]);
    out_result->scores.push_back(detections->scores[src]);
    out_result->labels.push_back(detections->labels[src]);
    AppendBoxes(detections->boxes, src, &out_result->boxes);
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus AppendIouNmsResult(CandidateDetections* detections,
                                   InferenceNmsType nms_type,
                                   float nms_thresh,
                                   int pre_maxsize,
                                   int post_maxsize,
                                   const std::optional<std::array<float, inference::kBoxDimension>>&
                                       self_car_box,
                                   const std::optional<float>& self_car_nms_thresh,
                                   CandidateDetections* out_result,
                                   std::string* out_error) {
  if (detections == nullptr || out_result == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "IoU NMS input or output is null");
  }
  if (nms_thresh < 0.0f || pre_maxsize <= 0 || post_maxsize <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "IoU NMS group config is invalid");
  }
  if (detections->scores.empty()) {
    return INFERENCE_STATUS_OK;
  }

  ApplyPreMaxsize(&detections->scores, &detections->boxes, &detections->labels, pre_maxsize);
  const bool use_self_car = self_car_box.has_value() && self_car_nms_thresh.has_value();
  if (use_self_car) {
    detections->scores.insert(detections->scores.begin(),
                              std::numeric_limits<float>::max());
    detections->labels.insert(detections->labels.begin(), -1);
    detections->boxes.insert(detections->boxes.begin(), self_car_box->begin(),
                             self_car_box->end());
  }
  std::vector<long> keep(detections->scores.size(), 0);
  int keep_count = 0;
  if (nms_type == INFERENCE_NMS_IOU_BEV) {
    keep_count = use_self_car
                     ? nms_gpu_with_self_car(detections->boxes.data(), keep.data(), nms_thresh,
                                             *self_car_nms_thresh,
                                             static_cast<int>(detections->scores.size()))
                     : nms_gpu(detections->boxes.data(), keep.data(), nms_thresh,
                               static_cast<int>(detections->scores.size()));
  } else if (nms_type == INFERENCE_NMS_IOU_MIN_BEV) {
    keep_count = use_self_car
                     ? nms_normalized_gpu_with_self_car(
                           detections->boxes.data(), keep.data(), nms_thresh,
                           *self_car_nms_thresh, static_cast<int>(detections->scores.size()))
                     : nms_normalized_gpu(detections->boxes.data(), keep.data(), nms_thresh,
                                          static_cast<int>(detections->scores.size()));
  } else {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "IoU NMS type is invalid");
  }
  size_t appended_count = 0;
  for (size_t i = 0; i < static_cast<size_t>(keep_count) &&
                     appended_count < static_cast<size_t>(post_maxsize);
       ++i) {
    const size_t src = static_cast<size_t>(keep[i]);
    if (use_self_car && src == 0U) {
      continue;
    }
    out_result->scores.push_back(detections->scores[src]);
    out_result->labels.push_back(detections->labels[src]);
    AppendBoxes(detections->boxes, src, &out_result->boxes);
    ++appended_count;
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus RunGroupedNms(const CandidateDetections& candidates,
                              const PostprocessConfig& config,
                              const RuntimeConfig& runtime_config,
                              CandidateDetections* out_result,
                              std::string* out_error) {
  out_result->scores.clear();
  out_result->boxes.clear();
  out_result->labels.clear();
  const int label_count = runtime_config.label_count > 0 ? runtime_config.label_count : kLabelCount;
  for (const PostprocessConfig::NmsLabelGroup& group : config.nms_label_groups) {
    CandidateDetections per_group;
    CollectByLabelSet(candidates, group.labels, label_count, &per_group);
    FilterByLabelScoreThreshold(&per_group, config.label_score_thresh);
    InferenceStatus status = INFERENCE_STATUS_OK;
    if (config.nms_type == INFERENCE_NMS_CIRCLE) {
      status = AppendCircleNmsResult(&per_group, group.circle_nms_min_radius,
                                     group.nms_post_maxsize, out_result, out_error);
    } else {
      status = AppendIouNmsResult(&per_group, config.nms_type, group.nms_thresh,
                                  group.nms_pre_maxsize, group.nms_post_maxsize,
                                  config.self_car_box, config.self_car_nms_thresh,
                                  out_result, out_error);
    }
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
  }
  FilterByScoreThreshold(out_result, config.final_score_threshold);
  return INFERENCE_STATUS_OK;
}

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
  if (num_pillars < 0 || num_pillars > config.max_pillars) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Invalid pillar count for debug tensors");
  }
  if (num_pillars == 0) {
    out_tensors->num_pillars = 0;
    out_tensors->voxels.clear();
    out_tensors->num_points.clear();
    out_tensors->coors.clear();
    return INFERENCE_STATUS_OK;
  }
  if (d_voxels == nullptr || d_num_points == nullptr || d_coors == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Device pillar tensors are null for a non-empty debug capture");
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

void SortCandidatesDescending(CandidateDetections* out_candidates) {
  if (out_candidates == nullptr) {
    return;
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
    AppendBoxes(out_candidates->boxes, index, &sorted.boxes);
  }
  *out_candidates = std::move(sorted);
}

InferenceStatus DecodeLegacyOutputs(const RuntimeConfig& config,
                                    const ResolvedHeadLayout& layout,
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
                        static_cast<size_t>(task_id) *
                            layout.bbox_channels_per_task * hw;
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
      const float yaw = DecodeYaw(layout, bbox, dir, hw, entry.index);

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
      out_candidates->labels.push_back(task_id);
      out_candidates->boxes.push_back(center_x);
      out_candidates->boxes.push_back(center_y);
      out_candidates->boxes.push_back(center_z);
      out_candidates->boxes.push_back(dim_x);
      out_candidates->boxes.push_back(dim_y);
      out_candidates->boxes.push_back(dim_z);
      out_candidates->boxes.push_back(yaw);
    }
  }

  SortCandidatesDescending(out_candidates);
  return INFERENCE_STATUS_OK;
}

InferenceStatus DecodeCenterPointOutputs(const RuntimeConfig& config,
                                         const ResolvedHeadLayout& layout,
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

  const int task_count = raw_outputs.task_count;
  const int height = raw_outputs.height;
  const int width = raw_outputs.width;
  const int hw = height * width;
  if (task_count <= 0 || height <= 0 || width <= 0) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "Raw outputs have invalid dimensions");
  }
  if (layout.task_count != task_count) {
    return SetError(out_error, INFERENCE_STATUS_BACKEND_ERROR,
                    "CenterPoint task layout does not match engine contract");
  }

  out_candidates->scores.reserve(static_cast<size_t>(task_count) * config.bbox_max_num);
  out_candidates->boxes.reserve(static_cast<size_t>(task_count) * config.bbox_max_num * kBoxDim);
  out_candidates->labels.reserve(static_cast<size_t>(task_count) * config.bbox_max_num);

  int class_offset = 0;
  for (int task_id = 0; task_id < task_count; ++task_id) {
    const int task_classes = layout.task_num_classes[static_cast<size_t>(task_id)];
    const float* heat =
        raw_outputs.cls_score + static_cast<size_t>(class_offset) * static_cast<size_t>(hw);
    const float* bbox = raw_outputs.bbox_pred +
                        static_cast<size_t>(task_id) *
                            layout.bbox_channels_per_task *
                            static_cast<size_t>(hw);
    const float* dir = raw_outputs.dir_cls_pred +
                       static_cast<size_t>(task_id) * kDirChannelsPerTask * static_cast<size_t>(hw);

    std::priority_queue<HeatmapEntry, std::vector<HeatmapEntry>, MinHeatmapScoreCompare> topk_heap;
    for (int class_index = 0; class_index < task_classes; ++class_index) {
      const float* class_heat = heat + static_cast<size_t>(class_index) * static_cast<size_t>(hw);
      for (int spatial_index = 0; spatial_index < hw; ++spatial_index) {
        if (!std::isfinite(class_heat[spatial_index])) {
          continue;
        }
        const float score = Sigmoid(class_heat[spatial_index]);
        if (!std::isfinite(score)) {
          continue;
        }
        if (topk_heap.size() < static_cast<size_t>(config.bbox_max_num)) {
          topk_heap.push({score, class_index, spatial_index});
        } else if (score > topk_heap.top().score) {
          topk_heap.pop();
          topk_heap.push({score, class_index, spatial_index});
        }
      }
    }

    std::vector<HeatmapEntry> topk;
    topk.reserve(topk_heap.size());
    while (!topk_heap.empty()) {
      topk.push_back(topk_heap.top());
      topk_heap.pop();
    }
    std::sort(topk.begin(), topk.end(), [](const HeatmapEntry& lhs, const HeatmapEntry& rhs) {
      return lhs.score > rhs.score;
    });

    for (const HeatmapEntry& entry : topk) {
      if (!std::isfinite(entry.score) || entry.score <= config.score_threshold) {
        continue;
      }
      const int y_idx = entry.spatial_index / width;
      const int x_idx = entry.spatial_index % width;
      const float reg_x = bbox[0 * hw + entry.spatial_index];
      const float reg_y = bbox[1 * hw + entry.spatial_index];
      const float height_value = bbox[2 * hw + entry.spatial_index];
      const float dim_x =
          config.norm_bbox ? SafeExp(bbox[3 * hw + entry.spatial_index])
                           : bbox[3 * hw + entry.spatial_index];
      const float dim_y =
          config.norm_bbox ? SafeExp(bbox[4 * hw + entry.spatial_index])
                           : bbox[4 * hw + entry.spatial_index];
      const float dim_z =
          config.norm_bbox ? SafeExp(bbox[5 * hw + entry.spatial_index])
                           : bbox[5 * hw + entry.spatial_index];
      const float yaw = DecodeYaw(layout, bbox, dir, hw, entry.spatial_index);

      const float center_x =
          (static_cast<float>(x_idx) + reg_x) * config.out_size_factor * config.pillar_size[0] +
          config.point_cloud_range[0];
      const float center_y =
          (static_cast<float>(y_idx) + reg_y) * config.out_size_factor * config.pillar_size[1] +
          config.point_cloud_range[1];
      const float center_z = height_value;
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
      out_candidates->labels.push_back(
          GlobalLabelForTaskClass(layout, task_id, entry.class_index));
      out_candidates->boxes.push_back(center_x);
      out_candidates->boxes.push_back(center_y);
      out_candidates->boxes.push_back(center_z);
      out_candidates->boxes.push_back(dim_x);
      out_candidates->boxes.push_back(dim_y);
      out_candidates->boxes.push_back(dim_z);
      out_candidates->boxes.push_back(yaw);
    }
    class_offset += task_classes;
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus DecodeOutputs(const RuntimeConfig& config,
                              const ResolvedHeadLayout& layout,
                              const RawHeadView& raw_outputs,
                              CandidateDetections* out_candidates,
                              std::string* out_error) {
  if (config.decode_mode == inference::DecodeMode::kCenterPoint) {
    return DecodeCenterPointOutputs(config, layout, raw_outputs, out_candidates,
                                    out_error);
  }
  return DecodeLegacyOutputs(config, layout, raw_outputs, out_candidates,
                             out_error);
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

InferenceStatus ResolveHeadLayout(const inference::ModelConfig& config,
                                  ResolvedHeadLayout* out_layout,
                                  std::string* out_error) {
  RuntimeConfig runtime_config;
  runtime_config.decode_mode = config.decode_mode;
  runtime_config.yaw_encoding = config.yaw_encoding;
  runtime_config.bbox_code_size = config.bbox_code_size;
  runtime_config.norm_bbox = config.norm_bbox;
  runtime_config.has_velocity = config.has_velocity;
  runtime_config.task_num_classes = config.task_num_classes;
  runtime_config.task_label_ids = config.task_label_ids;
  runtime_config.label_count = static_cast<int>(config.label_names.size());
  return ResolveHeadLayout(runtime_config, out_layout, out_error);
}

InferenceStatus ResolveHeadLayout(const RuntimeConfig& config,
                                  ResolvedHeadLayout* out_layout,
                                  std::string* out_error) {
  if (out_layout == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "out_layout is null");
  }

  ResolvedHeadLayout layout;
  layout.decode_mode = config.decode_mode;
  layout.yaw_encoding = config.yaw_encoding;
  layout.has_velocity = config.has_velocity;
  layout.norm_bbox = config.norm_bbox;
  layout.task_num_classes = config.task_num_classes;
  layout.label_count = config.label_count;
  layout.task_count = static_cast<int>(layout.task_num_classes.size());
  layout.theta_offset =
      layout.yaw_encoding == inference::YawEncoding::kDirectionTheta ? 6 : -1;
  layout.vel_offset = layout.has_velocity ? (layout.theta_offset >= 0 ? 7 : 6) : -1;
  layout.bbox_channels_per_task =
      2 + 1 + 3 + (layout.theta_offset >= 0 ? 1 : 0) + (layout.has_velocity ? 2 : 0);
  if (layout.task_count <= 0 || layout.bbox_channels_per_task <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "resolved head layout is invalid");
  }

  layout.task_class_offsets.reserve(layout.task_num_classes.size());
  layout.task_label_ids.reserve(layout.task_num_classes.size());
  layout.flat_global_label_ids.reserve(static_cast<size_t>(layout.label_count));
  int class_offset = 0;
  for (size_t task_index = 0; task_index < layout.task_num_classes.size();
       ++task_index) {
    const int task_classes = layout.task_num_classes[task_index];
    if (task_classes <= 0) {
      std::ostringstream oss;
      oss << "task_num_classes[" << task_index << "] must be positive";
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
    }
    std::vector<int> label_ids;
    if (task_index < config.task_label_ids.size()) {
      label_ids = config.task_label_ids[task_index];
    } else {
      label_ids.reserve(static_cast<size_t>(task_classes));
      for (int i = 0; i < task_classes; ++i) {
        label_ids.push_back(class_offset + i);
      }
    }
    if (label_ids.size() != static_cast<size_t>(task_classes)) {
      std::ostringstream oss;
      oss << "task_label_ids[" << task_index
          << "] size must match task_num_classes[" << task_index << "]";
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, oss.str());
    }
    layout.task_class_offsets.push_back(class_offset);
    layout.task_label_ids.push_back(label_ids);
    for (const int label_id : label_ids) {
      layout.flat_global_label_ids.push_back(label_id);
    }
    class_offset += task_classes;
  }
  layout.cls_channels_total = class_offset;
  *out_layout = std::move(layout);
  return INFERENCE_STATUS_OK;
}

int GlobalLabelForTaskClass(const ResolvedHeadLayout& layout,
                            int task_id,
                            int class_index) {
  if (task_id < 0 || task_id >= static_cast<int>(layout.task_label_ids.size())) {
    return class_index;
  }
  const std::vector<int>& label_ids =
      layout.task_label_ids[static_cast<size_t>(task_id)];
  if (class_index < 0 || class_index >= static_cast<int>(label_ids.size())) {
    return class_index;
  }
  return label_ids[static_cast<size_t>(class_index)];
}

const std::vector<int>& GlobalLabelsForTask(const ResolvedHeadLayout& layout,
                                            size_t task_index) {
  static const std::vector<int> kEmptyLabels;
  if (task_index >= layout.task_label_ids.size()) {
    return kEmptyLabels;
  }
  return layout.task_label_ids[task_index];
}

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
  out_config->decode_mode = inference::DecodeMode::kLegacy;
  out_config->yaw_encoding = inference::YawEncoding::kSinCos;
  out_config->bbox_code_size = kBoxDim;
  out_config->norm_bbox = false;
  out_config->has_velocity = false;
  out_config->task_num_classes = {1, 1, 1};
  out_config->label_count = kLabelCount;
  return INFERENCE_STATUS_OK;
}

void SetDefaultPostprocessConfig(PostprocessConfig* out_config) {
  if (out_config == nullptr) {
    return;
  }
  *out_config = PostprocessConfig{};
}

InferenceStatus ParsePostprocessConfig(const InferencePostprocessConfig& config,
                                       PostprocessConfig* out_config,
                                       std::string* out_error) {
  if (out_config == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "out_config is null");
  }
  if (config.nms_type != INFERENCE_NMS_IOU_BEV &&
      config.nms_type != INFERENCE_NMS_IOU_MIN_BEV &&
      config.nms_type != INFERENCE_NMS_CIRCLE) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "nms_type is invalid");
  }
  if (!std::isfinite(config.nms_thresh) || !std::isfinite(config.final_score_threshold) ||
      !IsFiniteArray(config.label_score_thresh, kLabelCount) ||
      !IsFiniteArray(config.circle_nms_min_radius, kLabelCount)) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Postprocess config contains non-finite values");
  }
  if (config.nms_pre_maxsize <= 0 || config.nms_post_maxsize <= 0 || config.nms_thresh < 0.0f ||
      config.final_score_threshold < 0.0f) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "Postprocess config contains non-positive limits");
  }
  for (int i = 0; i < kLabelCount; ++i) {
    if (config.label_score_thresh[i] < 0.0f || config.circle_nms_min_radius[i] < 0.0f) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "Postprocess thresholds must be non-negative");
    }
  }

  out_config->nms_type = static_cast<InferenceNmsType>(config.nms_type);
  out_config->nms_thresh = config.nms_thresh;
  out_config->nms_pre_maxsize = config.nms_pre_maxsize;
  out_config->nms_post_maxsize = config.nms_post_maxsize;
  out_config->final_score_threshold = config.final_score_threshold;
  std::copy(config.label_score_thresh, config.label_score_thresh + kLabelCount,
            out_config->label_score_thresh.begin());
  std::copy(config.circle_nms_min_radius, config.circle_nms_min_radius + kLabelCount,
            out_config->circle_nms_min_radius.begin());
  return INFERENCE_STATUS_OK;
}

InferenceStatus RunPostprocessNms(const CandidateDetections& candidates,
                                  const PostprocessConfig& config,
                                  const RuntimeConfig& runtime_config,
                                  CandidateDetections* out_result,
                                  std::string* out_error) {
  if (out_result == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "out_result is null");
  }
  if (config.nms_type != INFERENCE_NMS_IOU_BEV &&
      config.nms_type != INFERENCE_NMS_IOU_MIN_BEV &&
      config.nms_type != INFERENCE_NMS_CIRCLE) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "nms_type is invalid");
  }
  if (config.self_car_box.has_value() != config.self_car_nms_thresh.has_value()) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "self_car_box and self_car_nms_thresh must be configured together");
  }
  if (config.self_car_box.has_value()) {
    if (config.nms_type == INFERENCE_NMS_CIRCLE) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "self-car NMS only supports IoU BEV modes");
    }
    if (!std::isfinite(*config.self_car_nms_thresh) ||
        *config.self_car_nms_thresh < 0.0f) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "self_car_nms_thresh must be finite and non-negative");
    }
    for (const float value : *config.self_car_box) {
      if (!std::isfinite(value)) {
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                        "self_car_box must contain finite values");
      }
    }
    if ((*config.self_car_box)[3] <= 0.0f || (*config.self_car_box)[4] <= 0.0f ||
        (*config.self_car_box)[5] <= 0.0f) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "self_car_box dimensions must be positive");
    }
  }
  if (config.nms_pre_maxsize <= 0 || config.nms_post_maxsize <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "NMS maxsize must be positive");
  }
  if (!config.nms_label_groups.empty()) {
    return RunGroupedNms(candidates, config, runtime_config, out_result, out_error);
  }
  if (runtime_config.decode_mode == inference::DecodeMode::kCenterPoint) {
    ResolvedHeadLayout layout;
    InferenceStatus status =
        ResolveHeadLayout(runtime_config, &layout, out_error);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    out_result->scores.clear();
    out_result->boxes.clear();
    out_result->labels.clear();

    for (size_t task_index = 0; task_index < layout.task_num_classes.size();
         ++task_index) {
      CandidateDetections per_task;
      per_task.scores.reserve(candidates.scores.size());
      per_task.labels.reserve(candidates.labels.size());
      per_task.boxes.reserve(candidates.boxes.size());
      const std::vector<int>& task_labels = GlobalLabelsForTask(layout, task_index);
      CollectByLabelSet(candidates, task_labels, runtime_config.label_count, &per_task);
      if (config.nms_type == INFERENCE_NMS_CIRCLE) {
        const float radius = task_index < config.circle_nms_min_radius.size()
                                 ? config.circle_nms_min_radius[task_index]
                                 : 0.0f;
        InferenceStatus status = AppendCircleNmsResult(&per_task, radius, config.nms_post_maxsize,
                                                       out_result, out_error);
        if (status != INFERENCE_STATUS_OK) {
          return status;
        }
      } else {
        InferenceStatus status = AppendIouNmsResult(&per_task, config.nms_type, config.nms_thresh,
                                                    config.nms_pre_maxsize,
                                                    config.nms_post_maxsize,
                                                    config.self_car_box,
                                                    config.self_car_nms_thresh,
                                                    out_result, out_error);
        if (status != INFERENCE_STATUS_OK) {
          return status;
        }
      }
    }
    FilterByScoreThreshold(out_result, config.final_score_threshold);
    return INFERENCE_STATUS_OK;
  }
  if (config.nms_type == INFERENCE_NMS_CIRCLE) {
    out_result->scores.clear();
    out_result->boxes.clear();
    out_result->labels.clear();

    for (int label = 0; label < kLabelCount; ++label) {
      CandidateDetections per_label;
      per_label.scores.reserve(candidates.scores.size());
      per_label.labels.reserve(candidates.labels.size());
      per_label.boxes.reserve(candidates.boxes.size());
      for (size_t i = 0; i < candidates.scores.size(); ++i) {
        if (candidates.labels[i] != label) {
          continue;
        }
        per_label.scores.push_back(candidates.scores[i]);
        per_label.labels.push_back(label);
        AppendBoxes(candidates.boxes, i, &per_label.boxes);
      }
      InferenceStatus status = AppendCircleNmsResult(
          &per_label, config.circle_nms_min_radius[static_cast<size_t>(label)],
          config.nms_post_maxsize, out_result, out_error);
      if (status != INFERENCE_STATUS_OK) {
        return status;
      }
    }
    FilterByScoreThreshold(out_result, config.final_score_threshold);
    return INFERENCE_STATUS_OK;
  }

  std::vector<float> scores;
  std::vector<float> boxes;
  std::vector<int> labels;
  scores.reserve(candidates.scores.size());
  boxes.reserve(candidates.boxes.size());
  labels.reserve(candidates.labels.size());
  const int label_count = runtime_config.label_count > 0 ? runtime_config.label_count : kLabelCount;
  for (size_t i = 0; i < candidates.scores.size(); ++i) {
    const int label = candidates.labels[i];
    if (label < 0 || label >= label_count) {
      continue;
    }
    if (candidates.scores[i] < config.label_score_thresh[static_cast<size_t>(label)]) {
      continue;
    }
    scores.push_back(candidates.scores[i]);
    labels.push_back(label);
    AppendBoxes(candidates.boxes, i, &boxes);
  }
  out_result->scores.clear();
  out_result->boxes.clear();
  out_result->labels.clear();
  CandidateDetections filtered;
  filtered.scores = std::move(scores);
  filtered.boxes = std::move(boxes);
  filtered.labels = std::move(labels);
  return AppendIouNmsResult(&filtered, config.nms_type, config.nms_thresh,
                            config.nms_pre_maxsize, config.nms_post_maxsize,
                            config.self_car_box, config.self_car_nms_thresh,
                            out_result, out_error);
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
  ResolvedHeadLayout layout;
  TrtExecutor trt;
  std::unique_ptr<GpuDecodedCandidates> gpu_decoder;
  std::shared_ptr<Voxelization> voxelizer;
  cudaStream_t stream = nullptr;
  cudaEvent_t profile_start = nullptr;
  cudaEvent_t profile_end = nullptr;
  float* d_points = nullptr;
  float* d_bbox_pred = nullptr;
  float* d_cls_score = nullptr;
  float* d_dir_cls_pred = nullptr;
  float* h_points_staging = nullptr;
  float* h_bbox_pred = nullptr;
  float* h_cls_score = nullptr;
  float* h_dir_cls_pred = nullptr;
  size_t h_points_staging_count = 0;
  size_t h_bbox_pred_count = 0;
  size_t h_cls_score_count = 0;
  size_t h_dir_cls_pred_count = 0;

  ~Impl() {
    FreePinnedHostBuffer(&h_dir_cls_pred);
    FreePinnedHostBuffer(&h_cls_score);
    FreePinnedHostBuffer(&h_bbox_pred);
    FreePinnedHostBuffer(&h_points_staging);
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
    if (profile_end != nullptr) {
      cudaEventDestroy(profile_end);
    }
    if (profile_start != nullptr) {
      cudaEventDestroy(profile_start);
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
  InferenceStatus status = ResolveHeadLayout(config_, &impl_->layout, &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = impl_->trt.Load(config_, impl_->layout, config_.end2end_file,
                           &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  const EngineContract& contract = impl_->trt.contract();
  output_spec_ = MakeOutputSpec(contract.task_count, config_.bbox_max_num);

  status = CheckCuda(cudaStreamCreate(&impl_->stream), "cudaStreamCreate", &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = CheckCuda(cudaEventCreate(&impl_->profile_start), "cudaEventCreate(profile_start)",
                     &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = CheckCuda(cudaEventCreate(&impl_->profile_end), "cudaEventCreate(profile_end)",
                     &last_error_);
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
  const size_t bbox_size = static_cast<size_t>(contract.bbox_channels) *
                           static_cast<size_t>(contract.height) *
                           static_cast<size_t>(contract.width);
  const size_t cls_size = static_cast<size_t>(contract.cls_channels) *
                          static_cast<size_t>(contract.height) *
                          static_cast<size_t>(contract.width);
  const size_t dir_size = static_cast<size_t>(contract.dir_channels) *
                          static_cast<size_t>(contract.height) *
                          static_cast<size_t>(contract.width);

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

  impl_->h_points_staging_count = max_points_size;
  impl_->h_bbox_pred_count = bbox_size;
  impl_->h_cls_score_count = cls_size;
  impl_->h_dir_cls_pred_count = dir_size;
  status = AllocatePinnedHostBuffer(max_points_size, "h_points_staging", &impl_->h_points_staging,
                                    &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = AllocatePinnedHostBuffer(bbox_size, "h_bbox_pred", &impl_->h_bbox_pred, &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = AllocatePinnedHostBuffer(cls_size, "h_cls_score", &impl_->h_cls_score, &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status =
      AllocatePinnedHostBuffer(dir_size, "h_dir_cls_pred", &impl_->h_dir_cls_pred, &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  impl_->gpu_decoder = std::make_unique<GpuDecodedCandidates>();
  status = impl_->gpu_decoder->Initialize(
      config_, impl_->layout, contract.task_count, contract.bbox_channels,
      contract.cls_channels, contract.dir_channels, contract.height,
      contract.width, &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  return INFERENCE_STATUS_OK;
}

InferenceStatus PillarRuntime::Forward(const InferenceRequest& request,
                                       InferenceResponse* response,
                                       DebugArtifacts* debug,
                                       StageTimings* timings) {
  return ForwardInternal(request, nullptr, response, debug, timings);
}

InferenceStatus PillarRuntime::ForwardNms(const InferenceRequest& request,
                                          const PostprocessConfig& postprocess,
                                          InferenceResponse* response,
                                          DebugArtifacts* debug,
                                          StageTimings* timings) {
  return ForwardInternal(request, &postprocess, response, debug, timings);
}

InferenceStatus PillarRuntime::ForwardInternal(const InferenceRequest& request,
                                               const PostprocessConfig* postprocess,
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

  const bool enable_gpu_timing = (timings != nullptr || debug != nullptr);
  InferenceStatus status = INFERENCE_STATUS_OK;
  if (bounded_point_count > 0) {
    const size_t input_bytes = static_cast<size_t>(bounded_point_count) *
                               static_cast<size_t>(config_.num_point_features) *
                               sizeof(float);
    std::memcpy(impl_->h_points_staging, request.points, input_bytes);
    status = MeasureGpuStageBegin(impl_->profile_start, impl_->stream, enable_gpu_timing,
                                  &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = CheckCuda(cudaMemcpyAsync(impl_->d_points, impl_->h_points_staging, input_bytes,
                                       cudaMemcpyHostToDevice, impl_->stream),
                       "cudaMemcpyAsync(points)", &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = MeasureGpuStageEnd(impl_->profile_start, impl_->profile_end, impl_->stream,
                                enable_gpu_timing, &local_timings.h2d_prepare_ms, &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
  }

  status = MeasureGpuStageBegin(impl_->profile_start, impl_->stream, enable_gpu_timing,
                                &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  impl_->voxelizer->forward_pillars(impl_->d_points, bounded_point_count,
                                    CoordinateOrder::ZYX, impl_->stream);

  float* d_voxels = nullptr;
  unsigned int* d_num_points = nullptr;
  unsigned int* d_coors = nullptr;
  const int actual_num_pillars =
      static_cast<int>(impl_->voxelizer->get_pillar_output(&d_voxels, &d_num_points, &d_coors));
  status = MeasureGpuStageEnd(impl_->profile_start, impl_->profile_end, impl_->stream,
                              enable_gpu_timing, &local_timings.pillarize_ms, &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  const EngineContract& contract = impl_->trt.contract();
  const int padded_num_pillars =
      std::min(contract.max_runtime_pillars,
               std::max(actual_num_pillars, contract.min_runtime_pillars));

  if (debug != nullptr) {
    status = SyncCudaStream(impl_->stream, "cudaStreamSynchronize(debug_pillar_capture)",
                            &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    PillarInputTensors pillar_inputs;
    status = BuildPillars(config_, actual_num_pillars, d_voxels, d_num_points, d_coors,
                          &pillar_inputs, &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    debug->pillar_inputs = std::move(pillar_inputs);
  }

  status = impl_->trt.ConfigureShapes(config_, padded_num_pillars, &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  if (contract.num_bindings <= 0) {
    return SetError(&last_error_, INFERENCE_STATUS_BACKEND_ERROR,
                    "TensorRT engine reports no bindings");
  }
  std::vector<void*> bindings(static_cast<size_t>(contract.num_bindings), nullptr);
  bindings[static_cast<size_t>(contract.voxels_index)] = d_voxels;
  bindings[static_cast<size_t>(contract.num_points_index)] = d_num_points;
  bindings[static_cast<size_t>(contract.coors_index)] = d_coors;
  bindings[static_cast<size_t>(contract.bbox_pred_index)] = impl_->d_bbox_pred;
  bindings[static_cast<size_t>(contract.cls_score_index)] = impl_->d_cls_score;
  bindings[static_cast<size_t>(contract.dir_cls_pred_index)] = impl_->d_dir_cls_pred;

  status = MeasureGpuStageBegin(impl_->profile_start, impl_->stream, enable_gpu_timing,
                                &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = impl_->trt.Enqueue(bindings.data(), impl_->stream, &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  status = MeasureGpuStageEnd(impl_->profile_start, impl_->profile_end, impl_->stream,
                              enable_gpu_timing, &local_timings.trt_forward_ms, &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  CandidateDetections candidates;
  const CandidateDetections* output_detections = &candidates;
  CandidateDetections final_detections;
  auto start = std::chrono::steady_clock::now();
  auto end = start;

  if (debug == nullptr && impl_->gpu_decoder != nullptr) {
    status = MeasureGpuStageBegin(impl_->profile_start, impl_->stream, enable_gpu_timing,
                                  &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = impl_->gpu_decoder->Decode(impl_->d_bbox_pred, impl_->d_cls_score,
                                        impl_->d_dir_cls_pred, impl_->stream, &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = MeasureGpuStageEnd(impl_->profile_start, impl_->profile_end, impl_->stream,
                                enable_gpu_timing, &local_timings.decode_ms, &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }

    status = MeasureGpuStageBegin(impl_->profile_start, impl_->stream, enable_gpu_timing,
                                  &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = impl_->gpu_decoder->CopyToHost(&candidates, impl_->stream, &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = MeasureGpuStageEnd(impl_->profile_start, impl_->profile_end, impl_->stream,
                                enable_gpu_timing, &local_timings.d2h_collect_ms, &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    if (config_.decode_mode == inference::DecodeMode::kLegacy) {
      SortCandidatesDescending(&candidates);
    }
  } else {
    status = MeasureGpuStageBegin(impl_->profile_start, impl_->stream, enable_gpu_timing,
                                  &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = CheckCuda(cudaMemcpyAsync(impl_->h_bbox_pred, impl_->d_bbox_pred,
                                       impl_->h_bbox_pred_count * sizeof(float),
                                       cudaMemcpyDeviceToHost, impl_->stream),
                       "cudaMemcpyAsync(bbox_pred)", &last_error_);
    if (status == INFERENCE_STATUS_OK) {
      status = CheckCuda(cudaMemcpyAsync(impl_->h_cls_score, impl_->d_cls_score,
                                         impl_->h_cls_score_count * sizeof(float),
                                         cudaMemcpyDeviceToHost, impl_->stream),
                         "cudaMemcpyAsync(cls_score)", &last_error_);
    }
    if (status == INFERENCE_STATUS_OK) {
      status = CheckCuda(cudaMemcpyAsync(impl_->h_dir_cls_pred, impl_->d_dir_cls_pred,
                                         impl_->h_dir_cls_pred_count * sizeof(float),
                                         cudaMemcpyDeviceToHost, impl_->stream),
                         "cudaMemcpyAsync(dir_cls_pred)", &last_error_);
    }
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = SyncCudaStream(impl_->stream, "cudaStreamSynchronize(head_outputs)", &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    status = MeasureGpuStageEnd(impl_->profile_start, impl_->profile_end, impl_->stream,
                                enable_gpu_timing, &local_timings.d2h_collect_ms, &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }

    RawHeadView raw_outputs;
    raw_outputs.bbox_pred = impl_->h_bbox_pred;
    raw_outputs.cls_score = impl_->h_cls_score;
    raw_outputs.dir_cls_pred = impl_->h_dir_cls_pred;
    raw_outputs.bbox_channels = contract.bbox_channels;
    raw_outputs.cls_channels = contract.cls_channels;
    raw_outputs.dir_channels = contract.dir_channels;
    raw_outputs.height = contract.height;
    raw_outputs.width = contract.width;
    raw_outputs.task_count = contract.task_count;
    if (debug != nullptr) {
      debug->raw_outputs.bbox_pred.assign(impl_->h_bbox_pred,
                                          impl_->h_bbox_pred + impl_->h_bbox_pred_count);
      debug->raw_outputs.cls_score.assign(impl_->h_cls_score,
                                          impl_->h_cls_score + impl_->h_cls_score_count);
      debug->raw_outputs.dir_cls_pred.assign(impl_->h_dir_cls_pred,
                                             impl_->h_dir_cls_pred + impl_->h_dir_cls_pred_count);
      debug->raw_outputs.bbox_channels = raw_outputs.bbox_channels;
      debug->raw_outputs.cls_channels = raw_outputs.cls_channels;
      debug->raw_outputs.dir_channels = raw_outputs.dir_channels;
      debug->raw_outputs.height = raw_outputs.height;
      debug->raw_outputs.width = raw_outputs.width;
      debug->raw_outputs.task_count = raw_outputs.task_count;
    }

    start = std::chrono::steady_clock::now();
    status = DecodeOutputs(config_, impl_->layout, raw_outputs, &candidates,
                           &last_error_);
    end = std::chrono::steady_clock::now();
    local_timings.decode_ms = ElapsedMs(start, end);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    if (debug != nullptr) {
      debug->decoded_candidates = candidates;
      debug->final_nms_outputs = CandidateDetections{};
    }
  }

  if (postprocess != nullptr) {
    status = RunPostprocessNms(candidates, *postprocess, config_, &final_detections, &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      return status;
    }
    if (debug != nullptr) {
      debug->final_nms_outputs = final_detections;
    }
    output_detections = &final_detections;
  }

  start = std::chrono::steady_clock::now();
  status = CopyOutputsToResponse(*output_detections, response, &last_error_);
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
