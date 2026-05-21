# lidar-inference-tool

`lidar-inference-tool` is a C++17 LiDAR inference project built around TensorRT end-to-end pillar models. The repository exposes:

- `libinference.so`: a shared library exposing the recommended C++ wrapper for loading one model package and running decoded or final NMS-filtered detections
- `pillar_debug_tool`: a single-sample debug executable that loads one model directory plus one point cloud, runs inference, dumps intermediate tensors, and writes timing reports
- `lidar_model_downloader`: an installed command-line helper for fetching model assets after package installation
- `minimal_cpp_api_example`: a model-package-driven executable under `examples/` that shows the recommended C++ integration path

## v1.2.0

- Fixed the Decoder height parsing issue.
- Accelerated NMS on GPU.

The current runtime is written for PointPillars / CenterPoint-style engines with fixed binding names and a 3-task head layout.

## Privacy Note

This README intentionally uses repository-relative paths and placeholders such as `/path/to/TensorRT`. Do not replace examples with host-specific absolute paths, usernames, home directories, or machine identifiers when sharing configs or logs.

## Repository Layout

- `include/inference_cpp.hpp`: public C++ wrapper API
- `include/inference.h`: internal C compatibility header used by the wrapper and runtime implementation
- `src/inference_api.cpp`: internal compatibility-layer entry points used by the wrapper
- `src/pillar_runtime.cpp`: TensorRT loading, voxelization, inference, decode, NMS, and response packing
- `src/pillar_debug_utils.cpp`: debug-tool config assembly, point-cloud loading, and artifact export
- `tools/pillar_debug_tool.cpp`: debug CLI entry point
- `tools/onnx2engine.sh`: helper script for `trtexec`
- `tools/run_pillar_debug.sh`: recommended wrapper for launching `pillar_debug_tool` with the build-local third-party libraries first in `LD_LIBRARY_PATH`
- `scripts/visualize_bev.py`: BEV visualization for dumped debug artifacts
- `examples/minimal_cpp_api/`: minimal executable example for the public C++ wrapper API
- `demo/model.yaml`: example model package config for the public C++ wrapper
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
- decoded labels are internally represented as `1`, `2`, `3`, mapped by default to `car`, `pedestrian`, `cyclist`

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
- `build/minimal_cpp_api_example` when `BUILD_EXAMPLES=ON`

### DEB package

Use the repository helper to rebuild and package:

```bash
./rebuild.sh
```

The generated `.deb` installs its runtime commands under `/usr/local/bin`, including:

- `pillar_debug_tool`
- `lidar_model_downloader`

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

Run one debug session:

```bash
./tools/run_pillar_debug.sh demo demo/pcd.bin
```

Write artifacts to a custom output directory:

```bash
./tools/run_pillar_debug.sh demo demo/pcd.bin outputs/demo_dump
```

Run the recommended C++ wrapper example with the demo model directory:

```bash
./build/minimal_cpp_api_example demo demo/pcd.bin
```

The wrapper script is recommended because interactive shells may prepend unrelated library directories in `LD_LIBRARY_PATH`. `tools/run_pillar_debug.sh` forces the repository's freshly built `third_party/iou3d_nms` and `third_party/voxelize` shared libraries to win lookup order before launching the debug binary.

The demo assets used by both the public C++ API and the debug tool are:

- `demo/model.yaml`
- `demo/end2end.engine`
- `demo/pcd.bin`

Typical console output includes:

- point-cloud load time
- input point count
- pillar count
- decoded candidate count
- final NMS box count
- per-stage timings for a single run
- percentile summary over `profile_iters` runs after `warmup_iters` warmup runs

## Examples

The repository includes one standalone example under `examples/`.

`examples/minimal_cpp_api/` shows the recommended path for a realtime C++ project:

- load `model.yaml` from a model directory
- create one `inference::LidarInferenceRuntime` at startup
- reuse internal output buffers across frames
- call `runtime->Infer(...)` for final detections
- use `runtime->InferDecoded(...)` only when you explicitly need decoded pre-NMS candidates

Local README:

- `examples/minimal_cpp_api/README.md`

## Model Package Config

The public C++ wrapper expects a model directory containing:

- `model.yaml`
- the TensorRT engine referenced by `model.engine_path`

`model.yaml` is split into three sections:

- `model`: engine path, point feature count, spatial ranges, pillar size, stride, label names
- `runtime`: pillar and point capacity limits plus decode threshold
- `postprocess`: NMS type, thresholds, and per-label filtering values

The current demo `model.yaml` includes:

- `model.engine_path`
- `model.num_point_features`
- `model.point_cloud_range`
- `model.post_center_range`
- `model.pillar_size`
- `model.out_size_factor`
- `model.label_names`
- `runtime.max_points_per_pillar`
- `runtime.min_pillars`
- `runtime.max_pillars`
- `runtime.max_points`
- `runtime.frame_num`
- `runtime.bbox_max_num`
- `runtime.score_threshold`
- `postprocess.nms_type`
- `postprocess.nms_thresh`
- `postprocess.nms_pre_maxsize`
- `postprocess.nms_post_maxsize`
- `postprocess.final_score_threshold`
- `postprocess.circle_nms_min_radius`
- `postprocess.label_score_thresh`

The wrapper resolves relative `model.engine_path` values against the location of `model.yaml`.
Optional override YAML files use the same schema and are applied with this precedence:

- explicit code edits to the loaded `PackageConfig`
- override YAML
- base `model.yaml`

## Debug CLI

`pillar_debug_tool` now follows the same model-directory entry style as the public C++ wrapper:

```bash
./build/pillar_debug_tool <model_dir> <pcd_path> [dump_dir]
```

Supported options:

- `--warmup N`
- `--profile N`
- `--bev-resolution X`
- `--override-config PATH`

Examples:

```bash
./build/pillar_debug_tool demo demo/pcd.bin
./build/pillar_debug_tool demo demo/pcd.bin outputs/demo_dump --warmup 10 --profile 100
./build/pillar_debug_tool demo demo/pcd.bin outputs/demo_dump --override-config overrides/model_override.yaml
```

Behavior:

- `model_dir` must contain `model.yaml`
- runtime and postprocess settings come from `<model_dir>/model.yaml`
- `pcd_path` is the input point cloud for this one debug run
- `dump_dir` controls where `.npy` and `.json` artifacts are written
- `override_config_path` applies the same package-schema YAML override used by the public C++ wrapper

Default values:

- `dump_dir`: `outputs/pillar_debug`
- `warmup_iters`: `10`
- `profile_iters`: `50`
- `bev_resolution`: `0.2`

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
- decoded and final boxes use `[x, y, z, dx, dy, dz, yaw]` with `z` at the box geometric center; `result.json` writes `box_z_origin: "center"` explicitly
- `result.json` writes a zero-based `label` field plus `label_name`; this differs from the internal response label ids, which use `1..3`
- `metadata.json` contains timing data and tensor summaries
- `timings.json` contains both the single-run timing breakdown and aggregated profile statistics

## BEV Visualization

After a debug run, generate BEV plots with:

```bash
python3 scripts/visualize_bev.py demo outputs/pillar_debug
```

The script reads `demo/model.yaml` for the point-cloud range and uses the given `dump_dir`.
It produces:

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

## Public C++ API

The recommended C++ interface is declared in `include/inference_cpp.hpp`.

Typical call flow:

1. Place `model.yaml` and the TensorRT engine in one model directory.
2. Call `inference::LidarInferenceRuntime::CreateFromModelDir(...)` once at startup.
3. Keep the returned runtime on one inference thread.
4. For each frame, fill `inference::PointCloudView` and call `runtime->Infer(...)`.
5. Read `std::vector<inference::Detection>` results.

Additional wrapper entry points:

- `LoadPackageConfigFile(...)`: load one package config file directly
- `LoadPackageConfig(...)`: load `<model_dir>/model.yaml` into typed structs
- `Create(...)`: create a runtime from an already prepared `PackageConfig`
- `InferWithPostprocess(...)`: override postprocess settings at call time while still returning final NMS-filtered detections
- `InferDecoded(...)`: return decoded pre-NMS candidates

Important semantics:

- one runtime instance owns one `InferenceHandle` and reused output buffers
- do not call the same runtime instance concurrently from multiple threads
- `Detection::label_name` points into the runtime-owned config and stays valid while the runtime exists
- use `LoadPackageConfig(...)` if you want to load YAML, edit typed fields in code, and then call `Create(...)`
- `Infer()` uses the runtime's configured `postprocess` section and returns final detections after built-in NMS
- `InferDecoded()` and `Infer()` both return boxes as `[x, y, z, dx, dy, dz, yaw]` with `z` at the box geometric center

## ONNX to TensorRT

Use the helper script to build an engine and then run a quick `trtexec` profile:

```bash
bash tools/onnx2engine.sh <onnx_path> <engine_path> <min_points> <opt_points> <max_points> [fp32|fp16|int8]
```

Show the built-in help:

```bash
bash tools/onnx2engine.sh --help
```

Arguments:

- `<onnx_path>`: input ONNX model path
- `<engine_path>`: output TensorRT engine path
- `<min_points>`: minimum optimization-profile pillar/point count
- `<opt_points>`: optimal optimization-profile pillar/point count
- `<max_points>`: maximum optimization-profile pillar/point count
- `[precision]`: optional precision mode, one of `fp32`, `fp16`, `int8`, default `fp16`

The helper assumes the current repository contract:

- `voxels`: `<N>x20x5`
- `num_points`: `<N>`
- `coors`: `<N>x4`

FP16 example:

```bash
bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end.engine 10000 40000 50000
```

FP32 example:

```bash
bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end_fp32.engine 10000 40000 50000 fp32
```

INT8 example for a quantized / Q-DQ ONNX model:

```bash
bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end_int8.engine 10000 40000 50000 int8
```

Requirements and behavior:

- `trtexec` must be available in `PATH`
- the ONNX file must already exist
- `<min_points>`, `<opt_points>`, and `<max_points>` must be positive integers
- the helper requires `min_points <= opt_points <= max_points`
- the generated engine must preserve the current external binding contract: `voxels=float32`, `num_points=int32`, `coors=int32`, outputs `float32`
- the generated engine must keep a pillar-count optimization profile compatible with the runtime min/max settings
- `int8` export assumes the ONNX graph is already quantized or otherwise buildable by TensorRT without adding a calibration path in this helper
- the script prints the resolved min/opt/max TensorRT shapes before it starts building
- if the target engine already exists, regeneration is skipped
- the script still runs a follow-up `trtexec --loadEngine ... --dumpProfile --verbose` pass against the existing engine
- the final profiling pass always uses the opt shape
