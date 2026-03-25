#include "pillar_runtime.hpp"

#include <memory>
#include <new>
#include <string>

using inference_internal::ParseRuntimeConfig;
using inference_internal::PillarRuntime;
using inference_internal::RuntimeConfig;

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
