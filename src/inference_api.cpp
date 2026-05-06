#include "pillar_runtime.hpp"

#include <algorithm>
#include <memory>
#include <new>
#include <string>

using inference_internal::ParseRuntimeConfig;
using inference_internal::ParsePostprocessConfig;
using inference_internal::PillarRuntime;
using inference_internal::PostprocessConfig;
using inference_internal::RuntimeConfig;
using inference_internal::SetDefaultPostprocessConfig;

struct InferenceHandle {
  std::unique_ptr<PillarRuntime> runtime;
  std::string last_error;
};

namespace {
InferenceStatus MakeRuntime(const InferenceConfig* config,
                            std::unique_ptr<PillarRuntime>* out_runtime,
                            std::string* out_error) {
  if (config == nullptr || out_runtime == nullptr) {
    if (out_error != nullptr) {
      *out_error = "config or out_runtime is null";
    }
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  RuntimeConfig runtime_config;
  InferenceStatus status = ParseRuntimeConfig(*config, &runtime_config, out_error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  std::unique_ptr<PillarRuntime> runtime(new (std::nothrow) PillarRuntime(runtime_config));
  if (!runtime) {
    if (out_error != nullptr) {
      *out_error = "Failed to allocate PillarRuntime";
    }
    return INFERENCE_STATUS_BACKEND_ERROR;
  }

  status = runtime->Initialize();
  if (status != INFERENCE_STATUS_OK) {
    if (out_error != nullptr) {
      *out_error = runtime->last_error();
    }
    return status;
  }
  *out_runtime = std::move(runtime);
  return INFERENCE_STATUS_OK;
}

}  // namespace

InferenceStatus inference_create(const InferenceConfig* config,
                                 InferenceHandle** out_handle) {
  if (out_handle == nullptr) {
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }
  *out_handle = nullptr;

  std::string error;
  std::unique_ptr<PillarRuntime> runtime;
  InferenceStatus status = MakeRuntime(config, &runtime, &error);
  if (status != INFERENCE_STATUS_OK) {
    return status;
  }

  std::unique_ptr<InferenceHandle> handle(new (std::nothrow) InferenceHandle());
  if (!handle) {
    return INFERENCE_STATUS_BACKEND_ERROR;
  }
  handle->runtime = std::move(runtime);
  handle->last_error.clear();
  *out_handle = handle.release();
  return INFERENCE_STATUS_OK;
}

InferenceStatus inference_get_output_spec(const InferenceHandle* handle,
                                          InferenceOutputSpec* out_spec) {
  if (handle == nullptr || handle->runtime == nullptr || out_spec == nullptr) {
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }
  *out_spec = handle->runtime->output_spec();
  return INFERENCE_STATUS_OK;
}

InferenceStatus inference_get_default_postprocess_config(
    InferencePostprocessConfig* out_config) {
  if (out_config == nullptr) {
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }
  PostprocessConfig config;
  SetDefaultPostprocessConfig(&config);
  out_config->nms_type = config.nms_type;
  out_config->nms_thresh = config.nms_thresh;
  out_config->nms_pre_maxsize = config.nms_pre_maxsize;
  out_config->nms_post_maxsize = config.nms_post_maxsize;
  out_config->final_score_threshold = config.final_score_threshold;
  std::copy(config.label_score_thresh.begin(), config.label_score_thresh.end(),
            out_config->label_score_thresh);
  std::copy(config.circle_nms_min_radius.begin(), config.circle_nms_min_radius.end(),
            out_config->circle_nms_min_radius);
  return INFERENCE_STATUS_OK;
}

InferenceStatus inference_forward(InferenceHandle* handle,
                                  const InferenceRequest* request,
                                  InferenceResponse* response) {
  if (handle == nullptr || handle->runtime == nullptr || request == nullptr || response == nullptr) {
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }
  response->cls_count = 0;
  response->box_count = 0;
  response->label_count = 0;
  response->freespace_count = 0;
  InferenceStatus status = handle->runtime->Forward(*request, response, nullptr, nullptr);
  handle->last_error = handle->runtime->last_error();
  return status;
}

InferenceStatus inference_forward_nms(InferenceHandle* handle,
                                      const InferenceRequest* request,
                                      const InferencePostprocessConfig* postprocess,
                                      InferenceResponse* response) {
  if (handle == nullptr || handle->runtime == nullptr || request == nullptr || response == nullptr) {
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }

  PostprocessConfig internal_config;
  if (postprocess == nullptr) {
    SetDefaultPostprocessConfig(&internal_config);
  } else {
    InferenceStatus parse_status =
        ParsePostprocessConfig(*postprocess, &internal_config, &handle->last_error);
    if (parse_status != INFERENCE_STATUS_OK) {
      response->cls_count = 0;
      response->box_count = 0;
      response->label_count = 0;
      response->freespace_count = 0;
      return parse_status;
    }
  }

  response->cls_count = 0;
  response->box_count = 0;
  response->label_count = 0;
  response->freespace_count = 0;
  InferenceStatus status =
      handle->runtime->ForwardNms(*request, internal_config, response, nullptr, nullptr);
  handle->last_error = handle->runtime->last_error();
  return status;
}

InferenceStatus inference_reset(InferenceHandle* handle) {
  if (handle == nullptr || handle->runtime == nullptr) {
    return INFERENCE_STATUS_INVALID_ARGUMENT;
  }
  handle->runtime->Reset();
  handle->last_error.clear();
  return INFERENCE_STATUS_OK;
}

void inference_destroy(InferenceHandle* handle) {
  delete handle;
}

const char* inference_last_error(const InferenceHandle* handle) {
  if (handle == nullptr) {
    return "InferenceHandle is null";
  }
  return handle->last_error.c_str();
}
