#include "inference.h"

InferenceStatus inference_create(const InferenceConfig* config,
                                 InferenceHandle** out_handle) {
  (void)config;
  if (out_handle != nullptr) {
    *out_handle = nullptr;
  }
  return INFERENCE_STATUS_NOT_SUPPORTED;
}

InferenceStatus inference_get_output_spec(const InferenceHandle* handle,
                                          InferenceOutputSpec* out_spec) {
  (void)handle;
  if (out_spec != nullptr) {
    out_spec->max_candidates = 0;
    out_spec->cls_count = 0;
    out_spec->box_count = 0;
    out_spec->label_count = 0;
    out_spec->freespace_count = 0;
  }
  return INFERENCE_STATUS_NOT_SUPPORTED;
}

InferenceStatus inference_get_default_postprocess_config(
    InferencePostprocessConfig* out_config) {
  (void)out_config;
  return INFERENCE_STATUS_NOT_SUPPORTED;
}

InferenceStatus inference_forward(InferenceHandle* handle,
                                  const InferenceRequest* request,
                                  InferenceResponse* response) {
  (void)handle;
  (void)request;
  if (response != nullptr) {
    response->cls_count = 0;
    response->box_count = 0;
    response->label_count = 0;
    response->freespace_count = 0;
  }
  return INFERENCE_STATUS_NOT_SUPPORTED;
}

InferenceStatus inference_forward_nms(InferenceHandle* handle,
                                      const InferenceRequest* request,
                                      const InferencePostprocessConfig* postprocess,
                                      InferenceResponse* response) {
  (void)handle;
  (void)request;
  (void)postprocess;
  if (response != nullptr) {
    response->cls_count = 0;
    response->box_count = 0;
    response->label_count = 0;
    response->freespace_count = 0;
  }
  return INFERENCE_STATUS_NOT_SUPPORTED;
}

InferenceStatus inference_reset(InferenceHandle* handle) {
  (void)handle;
  return INFERENCE_STATUS_NOT_SUPPORTED;
}

void inference_destroy(InferenceHandle* handle) {
  (void)handle;
}

const char* inference_last_error(const InferenceHandle* handle) {
  (void)handle;
  return "Inference library was built without CUDA or TensorRT support";
}
