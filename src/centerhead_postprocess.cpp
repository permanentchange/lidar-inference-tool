#include "pillar_runtime.hpp"
#include "iou3d_nms_api.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace inference_internal {
namespace {
constexpr int kBoxDim = 7;
constexpr int kLabelCount = 3;
InferenceStatus SetError(std::string* error, InferenceStatus status, const std::string& message) {
  if (error) *error = message;
  return status;
}
void ApplyPreMaxsize(CandidateDetections* detections, int pre_maxsize) {
  // NMS requires score order even when there are fewer candidates than the cap.
  SortCandidatesDescending(detections);
  if (pre_maxsize > 0 && detections->scores.size() > size_t(pre_maxsize)) {
    detections->scores.resize(pre_maxsize);
    detections->labels.resize(pre_maxsize);
    detections->boxes.resize(size_t(pre_maxsize) * kBoxDim);
    if (!detections->source_indices.empty()) detections->source_indices.resize(pre_maxsize);
  }
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
    if (!detections->source_indices.empty()) filtered.source_indices.push_back(detections->source_indices[i]);
  }
  detections->scores.swap(filtered.scores);
  detections->labels.swap(filtered.labels);
  detections->boxes.swap(filtered.boxes);
  detections->source_indices.swap(filtered.source_indices);
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
    if (!detections->source_indices.empty()) filtered.source_indices.push_back(detections->source_indices[i]);
  }
  detections->scores.swap(filtered.scores);
  detections->labels.swap(filtered.labels);
  detections->boxes.swap(filtered.boxes);
  detections->source_indices.swap(filtered.source_indices);
}


void CollectByLabelSet(const CandidateDetections& candidates,
                       const std::vector<int>& labels,
                       int label_count,
                       CandidateDetections* out_group) {
  out_group->scores.clear();
  out_group->boxes.clear();
  out_group->labels.clear();
  out_group->source_indices.clear();
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
    if (!candidates.source_indices.empty()) out_group->source_indices.push_back(candidates.source_indices[i]);
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
    if (!detections->source_indices.empty()) out_result->source_indices.push_back(detections->source_indices[src]);
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

  ApplyPreMaxsize(detections, pre_maxsize);
  const bool use_self_car = self_car_box.has_value() && self_car_nms_thresh.has_value();
  if (use_self_car) {
    detections->scores.insert(detections->scores.begin(),
                              std::numeric_limits<float>::max());
    detections->labels.insert(detections->labels.begin(), -1);
    if (!detections->source_indices.empty()) detections->source_indices.insert(detections->source_indices.begin(), size_t(-1));
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
    if (!detections->source_indices.empty()) out_result->source_indices.push_back(detections->source_indices[src]);
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
  out_result->source_indices.clear();
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

}  // namespace

void SortCandidatesDescending(CandidateDetections* out_candidates) {
  if (out_candidates == nullptr) {
    return;
  }
  std::vector<size_t> order(out_candidates->scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
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
    if (!out_candidates->source_indices.empty()) sorted.source_indices.push_back(out_candidates->source_indices[index]);
  }
  *out_candidates = std::move(sorted);
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
  if (candidates.boxes.size() != candidates.scores.size() * kBoxDim ||
      candidates.labels.size() != candidates.scores.size() ||
      (!candidates.source_indices.empty() && candidates.source_indices.size() != candidates.scores.size()))
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT, "Candidate lengths differ");
  if (candidates.source_indices.empty() && !candidates.scores.empty()) {
    CandidateDetections indexed = candidates;
    indexed.source_indices.resize(indexed.scores.size());
    std::iota(indexed.source_indices.begin(), indexed.source_indices.end(), 0);
    return RunPostprocessNms(indexed, config, runtime_config, out_result, out_error);
  }
  if (config.nms_type != INFERENCE_NMS_IOU_BEV &&
      config.nms_type != INFERENCE_NMS_IOU_MIN_BEV &&
      config.nms_type != INFERENCE_NMS_CIRCLE) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "nms_type is invalid");
  }
  const int label_count = runtime_config.label_count > 0 ? runtime_config.label_count : kLabelCount;
  const bool uses_label_score_thresholds =
      !config.nms_label_groups.empty() ||
      runtime_config.decode_mode == inference::DecodeMode::kCenterPoint ||
      config.nms_type != INFERENCE_NMS_CIRCLE;
  if (uses_label_score_thresholds && !config.label_score_thresh.empty() &&
      config.label_score_thresh.size() < static_cast<size_t>(label_count)) {
    return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                    "label_score_thresh has fewer values than the runtime label count");
  }
  if (config.nms_label_groups.empty() && config.nms_type == INFERENCE_NMS_CIRCLE) {
    if (runtime_config.decode_mode == inference::DecodeMode::kCenterPoint) {
      if (config.circle_nms_min_radius.size() != runtime_config.task_num_classes.size()) {
        return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                        "circle_nms_min_radius must contain one value per CenterPoint task");
      }
    } else if (config.circle_nms_min_radius.size() < static_cast<size_t>(label_count)) {
      return SetError(out_error, INFERENCE_STATUS_INVALID_ARGUMENT,
                      "circle_nms_min_radius has fewer values than the runtime label count");
    }
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
    out_result->source_indices.clear();

    for (size_t task_index = 0; task_index < layout.task_num_classes.size();
         ++task_index) {
      CandidateDetections per_task;
      per_task.scores.reserve(candidates.scores.size());
      per_task.labels.reserve(candidates.labels.size());
      per_task.boxes.reserve(candidates.boxes.size());
      const std::vector<int>& task_labels = GlobalLabelsForTask(layout, task_index);
      CollectByLabelSet(candidates, task_labels, runtime_config.label_count, &per_task);
      FilterByLabelScoreThreshold(&per_task, config.label_score_thresh);
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
    out_result->source_indices.clear();

    for (int label = 0; label < label_count; ++label) {
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
        per_label.source_indices.push_back(candidates.source_indices[i]);
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
  std::vector<size_t> source_indices;
  scores.reserve(candidates.scores.size());
  boxes.reserve(candidates.boxes.size());
  labels.reserve(candidates.labels.size());
  for (size_t i = 0; i < candidates.scores.size(); ++i) {
    const int label = candidates.labels[i];
    if (label < 0 || label >= label_count) {
      continue;
    }
    if (!config.label_score_thresh.empty() &&
        candidates.scores[i] < config.label_score_thresh[static_cast<size_t>(label)]) {
      continue;
    }
    scores.push_back(candidates.scores[i]);
    labels.push_back(label);
    AppendBoxes(candidates.boxes, i, &boxes);
    source_indices.push_back(candidates.source_indices[i]);
  }
  out_result->scores.clear();
  out_result->boxes.clear();
  out_result->labels.clear();
    out_result->source_indices.clear();
  CandidateDetections filtered;
  filtered.scores = std::move(scores);
  filtered.boxes = std::move(boxes);
  filtered.labels = std::move(labels);
  filtered.source_indices = std::move(source_indices);
  return AppendIouNmsResult(&filtered, config.nms_type, config.nms_thresh,
                            config.nms_pre_maxsize, config.nms_post_maxsize,
                            config.self_car_box, config.self_car_nms_thresh,
                            out_result, out_error);
}

}  // namespace inference_internal
