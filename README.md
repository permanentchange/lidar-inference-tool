# lidar-inference-tool

`lidar-inference-tool` is a C++17 LiDAR inference project built around TensorRT end-to-end pillar models. The repository exposes:

- `libinference.so`: a shared library with a stable C API plus a higher-level C++ wrapper for loading one engine and running decoded or final NMS-filtered detections
- `pillar_debug_tool`: a single-sample debug executable that loads a YAML config, runs inference, dumps intermediate tensors, and writes timing reports
- `minimal_c_api_example`: a minimal executable under `examples/` that shows the lowest-level public API call flow
- `minimal_cpp_api_example`: a model-package-driven executable under `examples/` that shows the recommended C++ integration path

The current runtime is written for PointPillars / CenterPoint-style engines with fixed binding names and a 3-task head layout.

## Privacy Note

This README intentionally uses repository-relative paths and placeholders such as `/path/to/TensorRT`. Do not replace examples with host-specific absolute paths, usernames, home directories, or machine identifiers when sharing configs or logs.

## Repository Layout

- `include/inference.h`: public C API
- `include/inference_cpp.hpp`: public C++ wrapper API
- `src/inference_api.cpp`: shared-library entry points
- `src/pillar_runtime.cpp`: TensorRT loading, voxelization, inference, decode, NMS, and response packing
- `src/pillar_debug_utils.cpp`: YAML parsing, point-cloud loading, and artifact export
- `tools/pillar_debug_tool.cpp`: debug CLI entry point
- `tools/onnx2engine.sh`: helper script for `trtexec`
- `scripts/visualize_bev.py`: BEV visualization for dumped debug artifacts
- `examples/minimal_c_api/`: minimal executable example for the public C API
- `demo/model.yaml`: example model package config for the public C++ wrapper
- `demo/demo.yaml`: default debug-tool config
- `demo/end2end.onnx`: example ONNX model
- `demo/end2end.engine`: example TensorRT engine
- `demo/pcd.bin`: example point cloud
- `third_party/voxelize/`: vendored CUDA voxelization module
- `third_party/iou3d_nms/`: low-level IoU / NMS module used by the runtime and debug tool

## Runtime Contract

The TensorRT engine may execute internally in FP16 or INT8, but it must expose these external bindings:

Inputs:

- `voxels` (`float32`)
- `num_points` (`int32`)
- `coors` (`int32`)

Outputs:

- `bbox_pred0` (`float32`)
- `cls_score0` (`float32`)
- `dir_cls_pred0` (`float32`)

Additional assumptions enforced by the runtime:

- the engine must expose at least one optimization profile for pillar count
- the `voxels`, `num_points`, and `coors` inputs must agree on the same min/opt/max pillar range
- output tensors must be 4D `BCHW`
- output feature-map spatial sizes must match
- `bbox_pred0` must have `8 * task_count` channels
- `dir_cls_pred0` must have `2 * task_count` channels
- the current decode path expects 3 tasks
- decoded labels are `1`, `2`, `3` inside the C API, mapped by default to `car`, `pedestrian`, `cyclist`

Initialization fails if the engine bindings, dtypes, optimization profile, or channel layout do not match these expectations.

## Requirements

### Full native build

- Linux
- CMake >= 3.0.2
- GCC / G++ with C++17 support
- CUDA Toolkit with `nvcc`
- TensorRT headers and libraries
- `yaml-cpp` development package

Install the YAML dependency on Ubuntu:

```bash
sudo apt-get install -y libyaml-cpp-dev
```

Notes:

- the full build requires CUDA and TensorRT at configure time
- CMake rejects Conda-managed include and library paths on purpose; use system dependencies or explicit non-Conda roots
- TensorRT auto-detection is limited to standard system prefixes such as `/usr`, `/usr/local`, and `/opt`
- the build no longer scans personal directories such as `$HOME/TensorRT*` or `$HOME/workspace/TensorRT*`
- if TensorRT or `yaml-cpp` is installed in a nonstandard location, pass `-DTENSORRT_ROOT=...` and / or `-DYAML_CPP_ROOT=...`

### Python utilities

- Python 3
- `numpy`
- `pyyaml`
- `matplotlib`

Install optional Python dependencies with:

```bash
python3 -m pip install numpy pyyaml matplotlib
```

## Build

Run commands from the repository root.

### Full build

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

The example targets are enabled by default. Disable them only if you want the shared library and debug tool without the `examples/` executables:

```bash
cmake -S . -B build -DBUILD_EXAMPLES=OFF
cmake --build build -j$(nproc)
```

If TensorRT is not detected automatically from standard system prefixes:

```bash
cmake -S . -B build -DTENSORRT_ROOT=/path/to/TensorRT
cmake --build build -j$(nproc)
```

If `yaml-cpp` is outside standard system prefixes:

```bash
cmake -S . -B build \
  -DTENSORRT_ROOT=/path/to/TensorRT \
  -DYAML_CPP_ROOT=/path/to/yaml-cpp
cmake --build build -j$(nproc)
```

If the CMake cache is stale, was copied from another environment, or still points at an older TensorRT selection, reconfigure from scratch:

```bash
rm -rf build
cmake -S . -B build -DTENSORRT_ROOT=/path/to/TensorRT
cmake --build build -j$(nproc)
```

A successful full build generates:

- `build/libinference.so`
- `build/pillar_debug_tool`
- `build/minimal_c_api_example` when `BUILD_EXAMPLES=ON`
- `build/minimal_cpp_api_example` when `BUILD_EXAMPLES=ON`

### Platform notes

On some ARM64 / `aarch64` Linux systems, TensorRT headers are installed under a multiarch include directory such as `/usr/include/aarch64-linux-gnu` instead of the generic `/usr/include`. The CMake configuration in this repository now checks the active multiarch include suffix automatically, so a standard system TensorRT installation is detected without extra flags.

If you configure from a non-interactive shell where `PATH` is minimal or empty, make sure CMake can still resolve `nvcc`. The repository now falls back to `/usr/local/cuda/bin/nvcc` when it exists, but an explicit override also works:

```bash
cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j$(nproc)
```

### Stub build

If CUDA or TensorRT is unavailable but you still want the shared library to build, enable the stub explicitly:

```bash
cmake -S . -B build -DBUILD_STUB_IF_NO_TENSORRT=ON
cmake --build build -j$(nproc)
```

Stub behavior:

- only `build/libinference.so` is produced
- exported API calls return `INFERENCE_STATUS_NOT_SUPPORTED`
- `inference_last_error()` reports that the library was built without CUDA or TensorRT support
- example executables are skipped even if `BUILD_EXAMPLES=ON`

## Quick Start

The repository ships both a model package config for the public C++ wrapper and a debug-tool config.

Run one debug session:

```bash
./build/pillar_debug_tool demo/demo.yaml
```

Run the minimal C API example against the same demo assets:

```bash
./build/minimal_c_api_example demo/end2end.engine demo/pcd.bin
```

Run the recommended C++ wrapper example with the demo model directory:

```bash
./build/minimal_cpp_api_example demo demo/pcd.bin
```

The default model package reads:

- `demo/end2end.engine`
- `demo/model.yaml`

The default debug config reads:

- `demo/demo.yaml`
- `demo/pcd.bin`

and writes artifacts under:

- `outputs/demo_dump`

Typical console output includes:

- point-cloud load time
- input point count
- pillar count
- decoded candidate count
- final NMS box count
- per-stage timings for a single run
- percentile summary over `profile_iters` runs after `warmup_iters` warmup runs

## Examples

The repository includes two standalone examples under `examples/`.

`examples/minimal_cpp_api/` shows the recommended path for a realtime C++ project:

- load `model.yaml` from a model directory
- create one `inference::LidarInferenceRuntime` at startup
- reuse internal output buffers across frames
- call `runtime->Infer(...)` for final detections

`examples/minimal_c_api/` shows the low-level C integration path:

- fill `InferenceConfig` manually
- create an `InferenceHandle`
- allocate output buffers from `InferenceOutputSpec`
- call `inference_forward()` or `inference_forward_nms()`

Local READMEs:

- `examples/minimal_cpp_api/README.md`
- `examples/minimal_c_api/README.md`

## Model Package Config

The public C++ wrapper expects a model directory containing:

- `model.yaml`
- the TensorRT engine referenced by `model.engine_path`

`model.yaml` is split into three sections:

- `model`: engine path, point feature count, spatial ranges, pillar size, stride, label names
- `runtime`: pillar and point capacity limits plus decode threshold
- `postprocess`: NMS type, thresholds, and per-label filtering values

The wrapper resolves relative `model.engine_path` values against the location of `model.yaml`.
Optional override YAML files use the same schema and are applied with this precedence:

- explicit code edits to the loaded `PackageConfig`
- override YAML
- base `model.yaml`

## Debug Config

`pillar_debug_tool` expects one YAML file:

```bash
./build/pillar_debug_tool <config.yaml>
```

Top-level fields:

- `engine_path`: TensorRT engine path
- `pcd_path`: input point cloud path
- `dump_dir`: output directory for `.npy` / `.json` artifacts
- `warmup_iters`: number of warmup runs before profiling
- `profile_iters`: number of profiled runs
- `bev_resolution`: stored for visualization-oriented workflows
- `runtime`: voxelization and decode settings
- `post_processing`: debug NMS and final filtering settings

Important `runtime` fields:

- `max_points`
- `frame_num`
- `num_point_features`
- `point_cloud_range`
- `post_center_range`
- `pillar_size`
- `max_points_per_pillar`
- `min_pillars`
- `max_pillars`
- `bbox_max_num`
- `out_size_factor`
- `score_threshold`

Important `post_processing` fields:

- `nms_type`
- `nms_thresh`
- `nms_pre_maxsize`
- `nms_post_maxsize`
- `final_score_threshold`
- `circle_nms_min_radius`
- `label_names`
- `label_score_thresh`

Path handling:

- `engine_path`, `pcd_path`, and `dump_dir` are resolved relative to the YAML file if they are not absolute
- using repository-relative paths is recommended for portability and privacy

NMS behavior:

- when `nms_type` is `circle`, the tool applies per-label circle NMS using `circle_nms_min_radius`
- for any other `nms_type`, the tool uses the CUDA IoU NMS path from `third_party/iou3d_nms`

## Debug Artifacts

A successful debug run writes these files into `dump_dir`:

- `input_points.npy`
- `pillar_voxels.npy`
- `pillar_num_points.npy`
- `pillar_coors.npy`
- `bbox_pred.npy`
- `cls_score.npy`
- `dir_cls_pred.npy`
- `decoded_scores.npy`
- `decoded_boxes.npy`
- `decoded_labels.npy`
- `final_nms_scores.npy`
- `final_nms_boxes.npy`
- `final_nms_labels.npy`
- `result.json`
- `metadata.json`
- `timings.json`

Notes:

- `result.json` contains final detections after the debug tool's NMS stage
- `result.json` writes a zero-based `label` field plus `label_name`; this differs from the C API output, which uses `1..3`
- `metadata.json` contains timing data and tensor summaries
- `timings.json` contains both the single-run timing breakdown and aggregated profile statistics

## BEV Visualization

After a debug run, generate BEV plots with:

```bash
python3 scripts/visualize_bev.py demo/demo.yaml
```

The script reads `dump_dir` from the YAML config and produces:

- `01_input_points_bev.png`
- `02_heatmap_bev.png`
- `03_decoded_candidates_bev.png`
- `04_final_boxes_bev.png`

## Input Formats

### `.bin`

The loader accepts `float32` point clouds and tries to infer the source feature count from the file size.

Current behavior:

- if the file length matches `runtime.num_point_features`, that feature count is used directly
- otherwise the loader falls back to common layouts in this order: `5`, `4`, `3`
- data is copied into a buffer shaped by `runtime.num_point_features`
- if both the source file and the runtime are 5-feature layouts, only the first 4 features are copied and the 5th feature is zero-padded

That last rule matches the behavior documented in the loader comment: 5D input is treated as `XYZI` plus a zero-filled trailing feature instead of passing the 5th input channel through unchanged.

### `.pcd`

Current behavior:

- only binary PCD is supported
- only `x`, `y`, `z` are read from the payload
- remaining channels in the runtime input buffer are zero-padded

## Public C++ Wrapper API

The recommended C++ interface is declared in `include/inference_cpp.hpp`.

Typical call flow:

1. Place `model.yaml` and the TensorRT engine in one model directory.
2. Call `inference::LidarInferenceRuntime::CreateFromModelDir(...)` once at startup.
3. Keep the returned runtime on one inference thread.
4. For each frame, fill `inference::PointCloudView` and call `runtime->Infer(...)`.
5. Read `std::vector<inference::Detection>` results.

Important semantics:

- one runtime instance owns one `InferenceHandle` and reused output buffers
- do not call the same runtime instance concurrently from multiple threads
- `Detection::label_name` points into the runtime-owned config and stays valid while the runtime exists
- use `LoadPackageConfig(...)` if you want to load YAML, edit typed fields in code, and then call `Create(...)`

## Public C API

The public interface is declared in `include/inference.h`, and that header now contains the
canonical field-by-field and function-by-function contract.

Exported functions:

- `inference_create`
- `inference_get_output_spec`
- `inference_get_default_postprocess_config`
- `inference_forward`
- `inference_forward_nms`
- `inference_reset`
- `inference_destroy`
- `inference_last_error`

Typical call flow:

1. Fill an `InferenceConfig` and call `inference_create()`.
2. Call `inference_get_output_spec()` to learn the required output capacities.
3. Allocate response buffers using those element counts.
4. Fill an `InferenceRequest` with tightly packed point data and call `inference_forward_nms()` for final detections, or `inference_forward()` for decoded pre-NMS candidates.
5. Read `cls_count`, `box_count`, `label_count`, and `freespace_count` from the response.
6. Call `inference_destroy()` when done.

Minimal example:

```c
#include "inference.h"

InferenceHandle* handle = NULL;
InferenceConfig config = {0};
/* fill config.end2end_file and runtime parameters */
if (inference_create(&config, &handle) != INFERENCE_STATUS_OK) {
  return 1;
}

InferenceOutputSpec spec = {0};
inference_get_output_spec(handle, &spec);

float* cls_preds = malloc(spec.cls_count * sizeof(float));
float* box_preds = malloc(spec.box_count * sizeof(float));
int* label_preds = malloc(spec.label_count * sizeof(int));

InferencePostprocessConfig postprocess = {0};
inference_get_default_postprocess_config(&postprocess);

InferenceRequest request = {0};
request.points = points;
request.point_count = point_count;
request.enable_runtime = 1;

InferenceResponse response = {0};
response.cls_preds = cls_preds;
response.cls_capacity = spec.cls_count;
response.box_preds = box_preds;
response.box_capacity = spec.box_count;
response.label_preds = label_preds;
response.label_capacity = spec.label_count;

InferenceStatus status = inference_forward_nms(handle, &request, &postprocess, &response);
if (status != INFERENCE_STATUS_OK) {
  const char* err = inference_last_error(handle);
}

inference_destroy(handle);
```

Important semantics:

- `InferenceOutputSpec` and all `*_capacity` fields use element counts, not byte sizes
- `InferenceOutputSpec` is a safe upper bound for both `inference_forward()` and `inference_forward_nms()`
- `inference_forward()` returns decoded pre-NMS candidates
- `inference_forward_nms()` returns final detections after built-in NMS
- `box_preds` stores decoded boxes as `[x, y, z, dx, dy, dz, yaw]`
- `label_preds` stores 1-based task ids
- `box_count` is the number of float elements in `box_preds`, so it is `7 * cls_count`
- `freespace_count` is always `0` in the current implementation
- `freespace` is optional and is only zero-filled if a caller supplies that buffer
- `motion_4x4` and `enable_runtime` are present in the public request struct, but the current runtime does not read them
- `inference_get_default_postprocess_config()` returns the library's default NMS settings for the current fixed 3-label layout
- call `inference_last_error()` after a failing API call to retrieve a human-readable backend error

## ONNX to TensorRT

Use the helper script to build an engine and then run a quick `trtexec` profile:

```bash
bash tools/onnx2engine.sh <onnx_path> <engine_path> <min_points> <opt_points> <max_points> [fp32|fp16|int8]
```

FP16 example:

```bash
bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end.engine 10000 40000 50000 fp16
```

INT8 example for a quantized / Q-DQ ONNX model:

```bash
bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end_int8.engine 10000 40000 50000 int8
```

Requirements and behavior:

- `trtexec` must be available in `PATH`
- the ONNX file must already exist
- the generated engine must preserve the current external binding contract: `voxels=float32`, `num_points=int32`, `coors=int32`, outputs `float32`
- the generated engine must keep a pillar-count optimization profile compatible with the runtime min/max settings
- `int8` export assumes the ONNX graph is already quantized or otherwise buildable by TensorRT without adding a calibration path in this helper
- if the target engine already exists, regeneration is skipped
- the script always runs a follow-up `trtexec --loadEngine ... --dumpProfile --verbose` pass
