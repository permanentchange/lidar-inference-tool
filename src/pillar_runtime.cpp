#include "pillar_runtime.hpp"
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
constexpr int kBBoxChannelsPerTask = 8;
constexpr int kDirChannelsPerTask = 2;
constexpr int kLabelCount = 3;

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

std::vector<size_t> CircleNmsKeep(const std::vector<float>& scores,
                                  const std::vector<float>& boxes,
                                  float distance_threshold,
                                  int post_maxsize) {
  std::vector<size_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](size_t lhs, size_t rhs) { return scores[lhs] > scores[rhs]; });

  std::vector<size_t> keep;
  keep.reserve(order.size());
  std::vector<uint8_t> suppressed(scores.size(), 0);
  for (size_t order_index = 0; order_index < order.size(); ++order_index) {
    const size_t index = order[order_index];
    if (suppressed[index] != 0U) {
      continue;
    }
    keep.push_back(index);
    for (size_t compare_index = order_index + 1; compare_index < order.size(); ++compare_index) {
      const size_t other = order[compare_index];
      if (suppressed[other] != 0U) {
        continue;
      }
      const float dx = boxes[index * kBoxDim + 0] - boxes[other * kBoxDim + 0];
      const float dy = boxes[index * kBoxDim + 1] - boxes[other * kBoxDim + 1];
      const float squared_distance = dx * dx + dy * dy;
      if (squared_distance <= distance_threshold) {
        suppressed[other] = 1U;
      }
    }
    if (post_maxsize > 0 && keep.size() >= static_cast<size_t>(post_maxsize)) {
      break;
    }
  }
  return keep;
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
  if (config.nms_type != INFERENCE_NMS_IOU_BEV && config.nms_type != INFERENCE_NMS_CIRCLE) {
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
                                  CandidateDetections* out_result,
                                  std::string* out_error) {
  if (out_result == nullptr) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "out_result is null");
  }
  if (config.nms_type != INFERENCE_NMS_IOU_BEV && config.nms_type != INFERENCE_NMS_CIRCLE) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "nms_type is invalid");
  }
  if (config.nms_pre_maxsize <= 0 || config.nms_post_maxsize <= 0) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "NMS maxsize must be positive");
  }
  if (config.nms_type == INFERENCE_NMS_CIRCLE) {
    out_result->scores.clear();
    out_result->boxes.clear();
    out_result->labels.clear();

    for (int label = 1; label <= kLabelCount; ++label) {
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
      const std::vector<size_t> keep =
          CircleNmsKeep(per_label.scores, per_label.boxes,
                        config.circle_nms_min_radius[static_cast<size_t>(label - 1)],
                        config.nms_post_maxsize);
      for (size_t kept_index : keep) {
        out_result->scores.push_back(per_label.scores[kept_index]);
        out_result->labels.push_back(label);
        AppendBoxes(per_label.boxes, kept_index, &out_result->boxes);
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
  for (size_t i = 0; i < candidates.scores.size(); ++i) {
    const int label = candidates.labels[i];
    if (label < 1 || label > kLabelCount) {
      continue;
    }
    if (candidates.scores[i] < config.label_score_thresh[static_cast<size_t>(label - 1)]) {
      continue;
    }
    scores.push_back(candidates.scores[i]);
    labels.push_back(label);
    AppendBoxes(candidates.boxes, i, &boxes);
  }
  ApplyPreMaxsize(&scores, &boxes, &labels, config.nms_pre_maxsize);
  std::vector<long> keep(scores.size(), 0);
  const int keep_count = scores.empty()
                             ? 0
                             : nms_gpu(boxes.data(), keep.data(), config.nms_thresh,
                                       static_cast<int>(scores.size()));
  const size_t final_count = std::min(static_cast<size_t>(keep_count),
                                      static_cast<size_t>(config.nms_post_maxsize));

  out_result->scores.clear();
  out_result->boxes.clear();
  out_result->labels.clear();
  out_result->scores.reserve(final_count);
  out_result->boxes.reserve(final_count * kBoxDim);
  out_result->labels.reserve(final_count);
  for (size_t i = 0; i < final_count; ++i) {
    const long src = keep[i];
    out_result->scores.push_back(scores[static_cast<size_t>(src)]);
    out_result->labels.push_back(labels[static_cast<size_t>(src)]);
    AppendBoxes(boxes, static_cast<size_t>(src), &out_result->boxes);
  }
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
  TrtExecutor trt;
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
  InferenceStatus status = impl_->trt.Load(config_, config_.end2end_file, &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }
  const EngineContract& contract = impl_->trt.contract();
  output_spec_ = MakeOutputSpec(contract.task_count, config_.bbox_max_num);

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

  impl_->h_bbox_pred.resize(bbox_size);
  impl_->h_cls_score.resize(cls_size);
  impl_->h_dir_cls_pred.resize(dir_size);
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
  const EngineContract& contract = impl_->trt.contract();
  const int padded_num_pillars =
      std::min(contract.max_runtime_pillars,
               std::max(actual_num_pillars, contract.min_runtime_pillars));

  if (debug != nullptr) {
    PillarInputTensors pillar_inputs;
    status = BuildPillars(config_, actual_num_pillars, d_voxels, d_num_points, d_coors,
                          &pillar_inputs, &last_error_);
    if (status != INFERENCE_STATUS_OK) {
      cudaEventDestroy(gpu_start);
      cudaEventDestroy(gpu_end);
      return status;
    }
    debug->pillar_inputs = std::move(pillar_inputs);
  }

  status = impl_->trt.ConfigureShapes(config_, padded_num_pillars, &last_error_);
  if (status != INFERENCE_STATUS_OK) {
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_end);
    return status;
  }

  void* bindings[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  bindings[contract.voxels_index] = d_voxels;
  bindings[contract.num_points_index] = d_num_points;
  bindings[contract.coors_index] = d_coors;
  bindings[contract.bbox_pred_index] = impl_->d_bbox_pred;
  bindings[contract.cls_score_index] = impl_->d_cls_score;
  bindings[contract.dir_cls_pred_index] = impl_->d_dir_cls_pred;

  cudaEventRecord(gpu_start, impl_->stream);
  status = impl_->trt.Enqueue(bindings, impl_->stream, &last_error_);
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
  raw_outputs.bbox_channels = contract.bbox_channels;
  raw_outputs.cls_channels = contract.cls_channels;
  raw_outputs.dir_channels = contract.dir_channels;
  raw_outputs.height = contract.height;
  raw_outputs.width = contract.width;
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
    debug->final_nms_outputs = CandidateDetections{};
  }

  const CandidateDetections* output_detections = &candidates;
  CandidateDetections final_detections;
  if (postprocess != nullptr) {
    status = RunPostprocessNms(candidates, *postprocess, &final_detections, &last_error_);
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
