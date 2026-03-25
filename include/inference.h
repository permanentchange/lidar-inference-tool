#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct InferenceHandle InferenceHandle;

typedef enum InferenceStatus {
  INFERENCE_STATUS_OK = 0,
  INFERENCE_STATUS_INVALID_ARGUMENT = 1,
  INFERENCE_STATUS_NOT_READY = 2,
  INFERENCE_STATUS_BACKEND_ERROR = 3,
  INFERENCE_STATUS_NOT_SUPPORTED = 4,
} InferenceStatus;

typedef struct InferenceConfig {
  const char* end2end_file;
  float point_cloud_range[6];
  float post_center_range[6];
  float pillar_size[3];
  int max_points_per_pillar;
  int min_pillars;
  int max_pillars;
  int max_points;
  int frame_num;
  int num_point_features;
  int bbox_max_num;
  int out_size_factor;
  float score_threshold;
} InferenceConfig;

typedef struct InferenceOutputSpec {
  size_t max_candidates;
  size_t cls_count;
  size_t box_count;
  size_t label_count;
  size_t freespace_count;
} InferenceOutputSpec;

typedef struct InferenceRequest {
  const float* points;
  int point_count;
  const float* motion_4x4;
  int enable_runtime;
} InferenceRequest;

typedef struct InferenceResponse {
  float* cls_preds;
  size_t cls_capacity;
  float* box_preds;
  size_t box_capacity;
  int* label_preds;
  size_t label_capacity;
  float* freespace;
  size_t freespace_capacity;

  size_t cls_count;
  size_t box_count;
  size_t label_count;
  size_t freespace_count;
} InferenceResponse;

InferenceStatus inference_create(const InferenceConfig* config,
                                 InferenceHandle** out_handle);
InferenceStatus inference_get_output_spec(const InferenceHandle* handle,
                                          InferenceOutputSpec* out_spec);
InferenceStatus inference_forward(InferenceHandle* handle,
                                  const InferenceRequest* request,
                                  InferenceResponse* response);
InferenceStatus inference_reset(InferenceHandle* handle);
void inference_destroy(InferenceHandle* handle);
const char* inference_last_error(const InferenceHandle* handle);

#ifdef __cplusplus
}
#endif
