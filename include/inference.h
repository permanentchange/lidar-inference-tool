#pragma once

#include <stddef.h>

/**
 * @file inference.h
 * @brief Public C API for loading one TensorRT end-to-end pillar engine and running inference.
 *
 * Typical call flow:
 * 1. Fill an ::InferenceConfig and call ::inference_create().
 * 2. Call ::inference_get_output_spec() to learn the maximum response buffer sizes.
 * 3. Allocate output buffers using element counts from ::InferenceOutputSpec.
 * 4. Fill an ::InferenceRequest and call ::inference_forward().
 * 5. Read output counts from ::InferenceResponse and then call ::inference_destroy().
 *
 * The current runtime targets PointPillars / CenterPoint-style engines with fixed tensor names and
 * decoded boxes laid out as [x, y, z, dx, dy, dz, yaw].
 */

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque runtime handle created by ::inference_create() and released by ::inference_destroy(). */
typedef struct InferenceHandle InferenceHandle;

/** Status code returned by all public API functions except ::inference_destroy(). */
typedef enum InferenceStatus {
  /** The call completed successfully. */
  INFERENCE_STATUS_OK = 0,
  /** The caller provided invalid pointers, sizes, or configuration values. */
  INFERENCE_STATUS_INVALID_ARGUMENT = 1,
  /** The runtime handle exists but is not initialized or cannot execute yet. */
  INFERENCE_STATUS_NOT_READY = 2,
  /** Backend initialization or execution failed inside CUDA, TensorRT, or runtime logic. */
  INFERENCE_STATUS_BACKEND_ERROR = 3,
  /** The library was built without TensorRT support or the requested operation is unavailable. */
  INFERENCE_STATUS_NOT_SUPPORTED = 4,
} InferenceStatus;

/**
 * @brief Configuration used to create one inference runtime.
 *
 * All numeric values must be finite. Range arrays use the layout
 * [x_min, y_min, z_min, x_max, y_max, z_max].
 */
typedef struct InferenceConfig {
  /** Path to the serialized TensorRT engine file. Must be a non-empty string. */
  const char* end2end_file;
  /** Point cloud bounds used during voxelization. Each min value must be smaller than its max. */
  float point_cloud_range[6];
  /** Post-decode spatial filter range applied to predicted box centers. */
  float post_center_range[6];
  /** Pillar size in meters, laid out as [x_size, y_size, z_size]. All values must be positive. */
  float pillar_size[3];
  /** Maximum number of points stored per pillar. Must be positive. */
  int max_points_per_pillar;
  /** Minimum pillar count presented to the engine profile. Must be positive. */
  int min_pillars;
  /** Maximum pillar count presented to the engine profile. Must be >= ::min_pillars. */
  int max_pillars;
  /** Maximum number of input points consumed from each request. Extra points are ignored. */
  int max_points;
  /** Sequence length hint kept for API compatibility. Values smaller than 1 are clamped to 1. */
  int frame_num;
  /** Number of float features per input point. Must be at least 3 and match the input layout. */
  int num_point_features;
  /** Maximum decoded boxes kept per task head before final response packing. Must be positive. */
  int bbox_max_num;
  /** Backbone stride used during decode. Must be positive. */
  int out_size_factor;
  /** Score threshold applied during candidate decoding. */
  float score_threshold;
} InferenceConfig;

/**
 * @brief Maximum output sizes required by the current runtime configuration.
 *
 * All counts are element counts, not byte sizes.
 */
typedef struct InferenceOutputSpec {
  /** Maximum decoded candidate count across all tasks for one forward pass. */
  size_t max_candidates;
  /** Required capacity for ::InferenceResponse::cls_preds in float elements. */
  size_t cls_count;
  /** Required capacity for ::InferenceResponse::box_preds in float elements. Equals 7 * cls_count. */
  size_t box_count;
  /** Required capacity for ::InferenceResponse::label_preds in int elements. */
  size_t label_count;
  /** Required capacity for ::InferenceResponse::freespace in float elements. Currently always 0. */
  size_t freespace_count;
} InferenceOutputSpec;

/**
 * @brief One inference request.
 *
 * The point buffer is interpreted as a tightly packed row-major float array with
 * point_count * config.num_point_features elements.
 */
typedef struct InferenceRequest {
  /** Input point data. May be null only when ::point_count is 0. */
  const float* points;
  /** Number of points in ::points. Must be non-negative. */
  int point_count;
  /** Reserved for future ego-motion input. The current runtime ignores this field. */
  const float* motion_4x4;
  /** Reserved enable flag for future runtime selection. The current runtime ignores this field. */
  int enable_runtime;
} InferenceRequest;

/**
 * @brief Caller-owned output buffers and produced output counts.
 *
 * Capacities are element counts, not byte sizes. On entry, the caller fills buffer pointers and
 * capacities. On return, the runtime fills the corresponding *_count fields.
 */
typedef struct InferenceResponse {
  /** Decoded detection scores. Required when decoded candidate count is greater than 0. */
  float* cls_preds;
  /** Capacity of ::cls_preds in float elements. Must be >= produced ::cls_count. */
  size_t cls_capacity;
  /** Decoded boxes laid out as [x, y, z, dx, dy, dz, yaw]. */
  float* box_preds;
  /** Capacity of ::box_preds in float elements. Must be >= produced ::box_count. */
  size_t box_capacity;
  /** Decoded 1-based task labels. Required when decoded candidate count is greater than 0. */
  int* label_preds;
  /** Capacity of ::label_preds in int elements. Must be >= produced ::label_count. */
  size_t label_capacity;
  /** Optional freespace output buffer. The current runtime does not produce freespace values. */
  float* freespace;
  /** Capacity of ::freespace in float elements. If non-zero, ::freespace must be non-null. */
  size_t freespace_capacity;

  /** Number of valid score elements written to ::cls_preds. */
  size_t cls_count;
  /** Number of valid float elements written to ::box_preds. Equals 7 * cls_count. */
  size_t box_count;
  /** Number of valid label elements written to ::label_preds. */
  size_t label_count;
  /** Number of valid freespace elements written. Currently always 0. */
  size_t freespace_count;
} InferenceResponse;

/**
 * @brief Create one inference runtime from the supplied configuration.
 *
 * @param config Runtime configuration. Must be non-null and satisfy all documented constraints.
 * @param out_handle Output location for the created handle. Must be non-null. Set to null on failure.
 * @return ::INFERENCE_STATUS_OK on success. On failure, returns an error status and leaves
 *         *out_handle as null.
 */
InferenceStatus inference_create(const InferenceConfig* config,
                                 InferenceHandle** out_handle);
/**
 * @brief Query the maximum output buffer sizes required by a runtime handle.
 *
 * @param handle Valid handle returned by ::inference_create().
 * @param out_spec Output structure receiving required capacities in elements.
 * @return ::INFERENCE_STATUS_OK on success, otherwise ::INFERENCE_STATUS_INVALID_ARGUMENT.
 */
InferenceStatus inference_get_output_spec(const InferenceHandle* handle,
                                          InferenceOutputSpec* out_spec);
/**
 * @brief Run one forward pass and decode detections into caller-provided buffers.
 *
 * Before the call, the caller must provide response buffer pointers and capacities. The runtime
 * resets all output counts to 0 before validation and execution. The call fails with
 * ::INFERENCE_STATUS_INVALID_ARGUMENT if the request is invalid, if required response buffers are
 * null, or if any response capacity is smaller than the decoded output size.
 *
 * @param handle Valid handle returned by ::inference_create().
 * @param request Input request. Must be non-null.
 * @param response Caller-owned output buffers and output counts. Must be non-null.
 * @return ::INFERENCE_STATUS_OK on success, or an error status on validation or backend failure.
 *         Call ::inference_last_error() for a human-readable error string after failure.
 */
InferenceStatus inference_forward(InferenceHandle* handle,
                                  const InferenceRequest* request,
                                  InferenceResponse* response);
/**
 * @brief Reset internal runtime state associated with a handle.
 *
 * The current runtime clears the last error and resets backend state without changing the original
 * configuration.
 *
 * @param handle Valid handle returned by ::inference_create().
 * @return ::INFERENCE_STATUS_OK on success, otherwise ::INFERENCE_STATUS_INVALID_ARGUMENT.
 */
InferenceStatus inference_reset(InferenceHandle* handle);
/** Release a handle created by ::inference_create(). Accepts null. */
void inference_destroy(InferenceHandle* handle);
/**
 * @brief Return the most recent runtime error string associated with a handle.
 *
 * The returned pointer remains owned by the library and becomes invalid after the handle is
 * destroyed. For a null handle, a static fallback message is returned.
 */
const char* inference_last_error(const InferenceHandle* handle);

#ifdef __cplusplus
}
#endif
