# lidar-inference-tool

`lidar-inference-tool` is a C++17 LiDAR inference project built around TensorRT LiDAR detection models. It supports the original end-to-end pillar package path and a CenterPoint voxel package path. The repository exposes:

- `libinference.so`: a shared library exposing the recommended C++ wrapper for loading one model package and running decoded or final NMS-filtered detections
- `infer_debug_tool`: a single-sample debug executable that validates high-level C++ API inference results and writes visualization-friendly artifacts
- `perf_debug_tool`: a profiling executable for single-sample inference performance breakdowns
- `lidar_model_downloader`: an ARM64 package command-line helper for fetching model assets after package installation
- `minimal_cpp_api_example`: a model-package-driven executable under `examples/` that shows the recommended C++ integration path

## v1.8.0

- Added `centerpoint_voxel` model-package support through the same high-level C++ API used by pillar packages.
- Added bundled `libspconv` sparse-convolution integration for voxel sparse encoder / SCN execution.
- Reorganized demo assets into pillar, voxel, and shared point-cloud input directories.
- Added `scripts/centerpoint_voxel_head2engine.sh` for building the voxel RPN/head TensorRT engine.

See [CHANGELOG.md](./CHANGELOG.md) for version history.

The current runtime is written for PointPillars / CenterPoint-style engines with fixed binding names and a configurable CenterPoint task head layout.

## Privacy Note

This README intentionally uses repository-relative paths and placeholders such as `/path/to/TensorRT`. Do not replace examples with host-specific absolute paths, usernames, home directories, or machine identifiers when sharing configs or logs.

## Repository Layout

- `include/inference_cpp.hpp`: public C++ wrapper API
- `include/inference.h`: internal C compatibility header used by the wrapper and runtime implementation
- `src/inference_api.cpp`: internal compatibility-layer entry points used by the wrapper
- `src/pillar_runtime.cpp`: TensorRT loading, voxelization, inference, decode, NMS, and response packing
- `src/pillar_debug_utils.cpp`: debug-tool config assembly, point-cloud loading, and artifact export
- `tools/infer_debug_tool.cpp`: high-level C++ API result debug CLI entry point
- `tools/perf_debug_tool.cpp`: performance debug CLI entry point
- `scripts/lidar_model_downloader.py`: source implementation of the model downloader utility
- `scripts/dist/`: prebuilt downloader binary that is shipped in ARM64 DEB packages
- `scripts/onnx2engine.sh`: helper script for `trtexec`
- `scripts/centerpoint_voxel_head2engine.sh`: helper script for CenterPoint voxel RPN/head TensorRT engines
- `tools/visualize_infer_debug.py`: BEV and interactive 3D visualization for dumped infer-debug artifacts
- `examples/minimal_cpp_api/`: minimal executable example for the public C++ wrapper API
- `demo/centerpoint_pillar/`: example end-to-end pillar model package assets
- `demo/centerpoint_voxel/`: example CenterPoint voxel model package assets
- `demo/data/`: shared example point-cloud inputs
- `third_party/libspconv/`: CenterPoint voxel sparse-convolution runtime copied from CUDA-CenterPoint
- `third_party/voxelize/`: vendored CUDA voxelization module
- `third_party/iou3d_nms/`: low-level IoU / NMS module used by the runtime and debug tools
- `tests/`: optional regression tests built with `-DBUILD_TESTING=ON`

## Runtime Contract

### End-to-end pillar package

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
- `bbox_pred0` channels must match the resolved head layout for the configured decode mode, yaw encoding, and velocity fields
- `dir_cls_pred0` must have `2 * task_count` channels
- `cls_score0` channels must match the sum of `model.task_num_classes`
- decoded labels are global label ids resolved through `model.task_label_ids`
- final labels are mapped through `postprocess.label_mapping` when configured

Initialization fails if the engine bindings, dtypes, optimization profile, or channel layout do not match these expectations.

### CenterPoint voxel package

Set `model.runtime_backend: centerpoint_voxel` in `model.yaml`. The high-level C++ API remains the same as the pillar package path; downstream code can continue using `LidarInferenceRuntime::CreateFromModelDir(...)`, `runtime->config().model.num_point_features`, `PointCloudView`, and `runtime->Infer(...)`.

The voxel package runtime executes:

```text
points -> CUDA voxelization -> libspconv SCN -> TensorRT RPN/head -> decode -> NMS
```

Required model files in the model directory:

- `model.scn_onnx_path`: sparse encoder / SCN ONNX consumed by `libspconv`
- `model.head_engine_path`: TensorRT RPN/head engine

Required voxel YAML fields:

- `model.voxel_size`: `[x, y, z]`; legacy `model.pillar_size` is still accepted for existing pillar packages
- `model.sparse_shape`: sparse grid shape passed to `libspconv`
- `model.yaw_encoding: sin_cos`
- `model.has_velocity: true`

The head engine must expose one FP16 input from `libspconv` and separate FP32 CenterPoint head outputs. Output bindings may use canonical names such as `reg_0`, `height_0`, `dim_0`, `rot_0`, `vel_0`, `hm_0`, or the mmdetection3d export order `out0..out35`.

Point-cloud loading supports 3D, 4D, and 5D float `.bin` files and PCD files with `FIELDS x y z intensity has_intensity`. Missing feature channels are padded as `[x, y, z, 0, 0]`; a present `intensity` without `has_intensity` is loaded as `[x, y, z, intensity, 1]`.

## Requirements

### Full native build

- Linux
- CMake >= 3.0.2
- GCC / G++ with C++17 support
- CUDA Toolkit with `nvcc`
- TensorRT headers and libraries
- `yaml-cpp` development package
- Protobuf headers, libraries, and a matching `protoc`
- architecture- and CUDA-compatible `third_party/libspconv/lib/<arch>/libspconv.so`

Install common development dependencies on Ubuntu:

```bash
sudo apt-get install -y libyaml-cpp-dev protobuf-compiler libprotobuf-dev
```

Notes:

- the full build requires CUDA, TensorRT, Protobuf, `yaml-cpp`, and the bundled `libspconv` assets at configure time
- the full build also requires the vendored `third_party/libspconv` headers and parser proto files
- CMake uses `find_path` / `find_library` for TensorRT and `yaml-cpp`
- if TensorRT is installed in a nonstandard location, pass `-DTENSORRT_INCLUDE_DIR=...`, `-DNVINFER_LIB=...`, and `-DNVINFER_PLUGIN_LIB=...`
- `rebuild.sh` selects `third_party/libspconv/lib/<arch>_cuda<cuda>/libspconv.so` from the active `nvcc` version and refreshes `third_party/libspconv/lib/<arch>/libspconv.so`
- `rebuild.sh` preserves your environment, including `PATH`, `LD_LIBRARY_PATH`, compiler variables, and CMake search paths; use `CUDA_HOME`, `TENSORRT_ROOT`, or explicit CMake arguments for custom native dependencies
- the selected Protobuf headers/libraries and `protoc` must correspond to the same release (newer releases use different version numbering for the C++ runtime and compiler)
- CMake prefers `protoc` from the Protobuf headers' installation prefix before searching `PATH`; incompatible versions stop configuration with the selected paths
- to select a compiler explicitly, use `PROTOC=/usr/bin/protoc bash rebuild.sh` (for system Protobuf), or pass `-DProtobuf_PROTOC_EXECUTABLE=/path/to/matching/protoc`; explicit CMake arguments take priority over `PROTOC`
- direct CMake configuration also accepts `PROTOC`; an existing cached `Protobuf_PROTOC_EXECUTABLE` takes priority, so use `-UProtobuf_PROTOC_EXECUTABLE` when changing it through the environment

### Python utilities

- Python 3
- `numpy`
- `pyyaml`
- `matplotlib`
- `plotly` for 3D HTML visualization
- `minio` only when running `scripts/lidar_model_downloader.py` from source

Install optional Python dependencies with:

```bash
python3 -m pip install numpy pyyaml matplotlib plotly minio
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

If TensorRT is not detected automatically, pass the exact include and library paths:

```bash
cmake -S . -B build \
  -DTENSORRT_INCLUDE_DIR=/path/to/TensorRT/include \
  -DNVINFER_LIB=/path/to/TensorRT/lib/libnvinfer.so \
  -DNVINFER_PLUGIN_LIB=/path/to/TensorRT/lib/libnvinfer_plugin.so
cmake --build build -j$(nproc)
```

If the CMake cache is stale, was copied from another environment, or still points at an older TensorRT selection, reconfigure from scratch:

```bash
rm -rf build
cmake -S . -B build \
  -DTENSORRT_INCLUDE_DIR=/path/to/TensorRT/include \
  -DNVINFER_LIB=/path/to/TensorRT/lib/libnvinfer.so \
  -DNVINFER_PLUGIN_LIB=/path/to/TensorRT/lib/libnvinfer_plugin.so
cmake --build build -j$(nproc)
```

A successful full build generates:

- `build/libinference.so`
- `build/infer_debug_tool`
- `build/perf_debug_tool`
- `build/minimal_cpp_api_example` when `BUILD_EXAMPLES=ON`

Build and run regression tests when CUDA, TensorRT, `yaml-cpp`, Protobuf, and the vendored runtime libraries are available:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

### DEB package

Use the repository helper to rebuild and package:

```bash
./rebuild.sh
```

The native build uses the installed C++/CUDA tools and libraries. The helper
preserves your shell environment and does not activate or search for a Python
environment. If dependency discovery fails, set the relevant paths and rerun:

```bash
export CUDA_HOME=/usr/local/cuda
export TENSORRT_ROOT=/path/to/TensorRT
export PROTOC=/usr/bin/protoc
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$TENSORRT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash rebuild.sh
```

You can also set `CC`, `CXX`, `PATH`, and `CMAKE_PREFIX_PATH` for your toolchain,
or pass CMake options directly to `rebuild.sh`. Configuration prints the selected
Protobuf include directory, library, and compiler version. A version mismatch
stops configuration with instructions for selecting a matching compiler.

The generated `.deb` installs its runtime commands under `/usr/local/bin`, including:

- `infer_debug_tool`
- `perf_debug_tool`
- `lidar_model_downloader` on ARM64 / `aarch64` packages only

The downloader command in ARM64 DEB packages comes from the prebuilt binary under `scripts/dist/`. The Python source file `scripts/lidar_model_downloader.py` remains the repository-side source utility.

Installed CMake package files include a version file, so downstream projects can require this release with `find_package(lidar-inference-tool 1.8.0 CONFIG REQUIRED)`.

The package build now runs from `build/`, so CPack temporary files stay under `build/_CPack_Packages/` instead of the repository root.

### Platform notes

If you configure from a non-interactive shell where `PATH` is minimal or empty, pass the CUDA compiler explicitly:

```bash
cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j$(nproc)
```

## Quick Start

Run the recommended pillar API result debug session:

```bash
./build/infer_debug_tool --model-dir demo/centerpoint_pillar --input demo/data/test1.bin --source-num-point-features 5 --dump-dir demo/centerpoint_pillar_outputs --dump-decoded --dump-internal --warmup 10 --profile-iters 100
```

For the voxel demo, first build the TensorRT head engine referenced by `demo/centerpoint_voxel/model.yaml`, then run the same debug flow:

```bash
bash scripts/centerpoint_voxel_head2engine.sh demo/centerpoint_voxel/rpn_centerhead.onnx demo/centerpoint_voxel/rpn_centerhead.engine
./build/infer_debug_tool --model-dir demo/centerpoint_voxel --input demo/data/test1.bin --source-num-point-features 5 --dump-dir demo/centerpoint_voxel_outputs --dump-decoded --dump-internal --warmup 10 --profile-iters 100
```

The demo assets used by the public C++ API and debug tools are:

- `demo/centerpoint_pillar/model.yaml`
- `demo/centerpoint_pillar/end2end.engine`
- `demo/centerpoint_voxel/model.yaml`
- `demo/centerpoint_voxel/centerpoint.scn.onnx`
- `demo/centerpoint_voxel/rpn_centerhead.onnx`
- `demo/centerpoint_voxel/rpn_centerhead.engine` after running `scripts/centerpoint_voxel_head2engine.sh`
- `demo/data/test1.bin`
- `demo/data/test1.pcd`
- `demo/data/test2.pcd`
- `demo/data/test3.pcd`

Typical `perf_debug_tool` console output includes:

- point-cloud load time
- input point count
- pillar count
- decoded candidate count
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
- for `pillar_end2end`: the TensorRT engine referenced by `model.engine_path`
- for `centerpoint_voxel`: the SCN ONNX referenced by `model.scn_onnx_path` and the TensorRT head engine referenced by `model.head_engine_path`

`model.yaml` is split into three sections:

- `model`: engine/backend paths, point feature count, spatial ranges, voxel size, stride, label names, and head layout
- `runtime`: voxel and point capacity limits plus decode threshold
- `postprocess`: NMS type, thresholds, and per-label filtering values

Across the current pillar and voxel demo configs, `model.yaml` includes:

- `model.engine_path` for `pillar_end2end` packages
- `model.runtime_backend` for CenterPoint voxel packages
- `model.scn_onnx_path` for CenterPoint voxel packages
- `model.head_engine_path` for CenterPoint voxel packages
- `model.num_point_features`
- `model.point_cloud_range`
- `model.post_center_range`
- `model.voxel_size`; legacy `model.pillar_size` is still accepted
- `model.sparse_shape` for CenterPoint voxel packages
- `model.out_size_factor`
- `model.label_names`
- `model.yaw_encoding`
- `model.task_num_classes`
- `model.task_label_ids`
- `runtime.max_points_per_voxel`; legacy `runtime.max_points_per_pillar` is still accepted
- `runtime.min_voxels`; legacy `runtime.min_pillars` is still accepted
- `runtime.max_voxels`; legacy `runtime.max_pillars` is still accepted
- `runtime.max_points`
- `runtime.frame_num`
- `runtime.bbox_max_num`
- `runtime.score_threshold`
- `postprocess.nms_type`
- `postprocess.nms_thresh`
- `postprocess.nms_pre_maxsize`
- `postprocess.nms_post_maxsize`
- `postprocess.final_score_threshold`
- `postprocess.self_car_box`
- `postprocess.self_car_nms_thresh`
- `postprocess.nms_label_groups`
- `postprocess.label_score_thresh`
- `postprocess.label_mapping`
- `postprocess.mapped_label_names`

`postprocess.nms_type` supports `iou_bev`, `iou_min_bev`, and `circle`.
`iou_min_bev` uses `overlap / min(area_a, area_b)` in BEV, which suppresses fully-contained small boxes more aggressively than standard IoU.

`postprocess.self_car_box` optionally configures the ego-vehicle box as
`[x, y, z, dx, dy, dz, yaw]`. It must be configured together with
`postprocess.self_car_nms_thresh`; omitting both fields preserves the original NMS behavior.
When enabled, the ego box is inserted as the first candidate in each IoU NMS pass. Comparisons
from the ego box use `self_car_nms_thresh`, while comparisons between detected boxes continue to
use the normal group or global `nms_thresh`. A detected box is suppressed when its configured BEV
overlap metric is strictly greater than the ego threshold. The ego box is removed from the result
and does not consume `nms_pre_maxsize` or `nms_post_maxsize` capacity. Ego-box filtering supports
`iou_bev` and `iou_min_bev`; it is rejected with `circle` NMS.

The current demo config uses lower per-label score thresholds than the previous release and runs one shared NMS pass over all nine global labels. Override these values in `model.yaml` or an override YAML when a deployment needs class-specific NMS groups.

Label-related config uses two stages:

- `model.label_names` defines global labels.
- `model.task_label_ids` maps each task-local ONNX class channel to a global label id.
- `postprocess.nms_label_groups` selects which global labels share one NMS pass.
- `postprocess.label_mapping` maps global labels to final business labels returned by the public API.
- `postprocess.mapped_label_names` names those final business labels.

CenterPoint head layout also uses one explicit yaw field:

- `model.yaw_encoding: sin_cos` keeps the legacy decode path where yaw is reconstructed from `dir_cls_pred`.
- `model.yaw_encoding: direction_theta` uses the newer layout where `bbox_pred` carries `theta` and `dir_cls_pred` carries 2-way direction logits.

The recommended public C++ example flow is unchanged in this release:

- create one runtime with `LidarInferenceRuntime::CreateFromModelDir(...)`
- run final detections with `runtime->Infer(...)`
- use `runtime->InferDecoded(...)` only when pre-NMS debug candidates are needed

The wrapper resolves relative model asset paths, including `model.engine_path`, `model.scn_onnx_path`, and `model.head_engine_path`, against the location of `model.yaml`.
Optional override YAML files use the same schema and are applied with this precedence:

- explicit code edits to the loaded `PackageConfig`
- override YAML
- base `model.yaml`

## Debug CLIs

`infer_debug_tool` follows the same model-directory entry style as the public C++ wrapper and treats `runtime->Infer(...)` as the authoritative final result:

```bash
./build/infer_debug_tool --model-dir MODEL_DIR --input INPUT_PATH --dump-dir DUMP_DIR [--source-num-point-features N] [--override-config PATH] [--dump-decoded] [--dump-internal] [--warmup N] [--profile-iters N]
```

Supported options:

- `--source-num-point-features N`: source feature count for float32 `.bin` files; specify `3`, `4`, or `5` when the layout is known
- `--override-config PATH`
- `--warmup N`
- `--profile-iters N`
- `--dump-decoded`
- `--dump-internal`

Recommended examples:

```bash
# Raw float32 .bin: declare its source feature layout.
./build/infer_debug_tool --model-dir demo/centerpoint_pillar --input demo/data/test1.bin --source-num-point-features 5 --dump-dir demo/centerpoint_pillar_outputs --dump-decoded --dump-internal --warmup 10 --profile-iters 100

# PCD: no source feature-count option is needed; fields are read from the PCD header.
./build/infer_debug_tool --model-dir demo/centerpoint_pillar --input demo/data/test1.pcd --dump-dir demo/centerpoint_pcd_outputs --dump-decoded --dump-internal --warmup 10 --profile-iters 100

./build/infer_debug_tool --model-dir demo/centerpoint_voxel --input demo/data/test1.bin --source-num-point-features 5 --dump-dir demo/centerpoint_voxel_outputs --dump-decoded --dump-internal --warmup 10 --profile-iters 100
```

The voxel example requires `demo/centerpoint_voxel/rpn_centerhead.engine`; build it from `demo/centerpoint_voxel/rpn_centerhead.onnx` with `scripts/centerpoint_voxel_head2engine.sh` before running inference.

Behavior:

- `--model-dir` must point to a directory containing `model.yaml`
- runtime and postprocess settings come from `--model-dir` `model.yaml`
- `--input` may be one `.pcd` / `.bin` file or a directory containing top-level `.pcd` / `.bin` files
- `--dump-dir` controls where result `.npy` and `.json` artifacts are written
- `--override-config PATH` applies the same package-schema YAML override used by the public C++ wrapper
- final detections come from the high-level C++ API, so `mapped_label_names` is reflected in `result.json`
- `--dump-internal` writes raw internal tensors for auxiliary diagnosis only
- directory mode sorts numerically by filename stem when all stems are unsigned integers; otherwise it falls back to lexicographic filename order
- numeric filename stems are treated as Unix nanosecond timestamps in per-frame metadata and sequence outputs
- directory mode reuses one runtime instance and writes one subdirectory per frame under `dump_dir/frames/`
- `--warmup` and `--profile-iters` repeat the high-level `runtime->Infer(...)` call and print total latency statistics

`perf_debug_tool` uses the internal runtime path to profile one sample:

```bash
./build/perf_debug_tool <model_dir> <pcd_path> [dump_dir] [--warmup N] [--profile N] [--source-num-point-features N] [--override-config PATH]
```

It writes timing/profile data only and does not write detection result artifacts.

Default values:

- `infer_debug_tool` `warmup_iters`: `0`
- `infer_debug_tool` `profile_iters`: `1`
- `perf_debug_tool` `dump_dir`: `outputs/perf_debug`
- `perf_debug_tool` `warmup_iters`: `10`
- `perf_debug_tool` `profile_iters`: `50`

## Debug Artifacts

A successful `infer_debug_tool` run writes these files into `dump_dir`:

- `input_points.npy`
- `final_scores.npy`
- `final_boxes.npy`
- `final_labels.npy`
- `final_label_names.json`
- `result.json`
- `metadata.json`

In directory batch mode, the root `dump_dir` additionally contains:

- `frames/<frame_index>_<file_stem>/...` per-frame artifact directories
- `sequence_manifest.json`
- `sequence_detections.jsonl`

With `--dump-decoded`, it also writes:

- `decoded_scores.npy`
- `decoded_boxes.npy`
- `decoded_labels.npy`

With `--dump-internal`, it also writes auxiliary internal tensors:

- `pillar_voxels.npy`
- `pillar_num_points.npy`
- `pillar_coors.npy`
- `bbox_pred.npy`
- `cls_score.npy`
- `dir_cls_pred.npy`
- `internal_decoded_scores.npy`
- `internal_decoded_boxes.npy`
- `internal_decoded_labels.npy`

A successful `perf_debug_tool` run writes:

- `timings.json`
- `metadata.json`

Notes:

- `result.json` contains final detections returned by `runtime->Infer(...)`
- final boxes use `[x, y, z, dx, dy, dz, yaw]` with `z` at the box geometric center
- `label_name` fields in `result.json` come from the high-level API and honor `postprocess.mapped_label_names`
- per-frame `metadata.json` now also includes `input_path_kind`, optional `frame_index`, and optional `timestamp_ns` / `timestamp_seconds`
- `sequence_manifest.json` records ordered frame metadata for downstream consumers
- `sequence_detections.jsonl` writes one JSON object per frame with ordered detections for downstream Python Kalman-filter code
- `perf_debug_tool/timings.json` contains both the single-run timing breakdown and aggregated profile statistics

## Infer Debug Visualization

After an `infer_debug_tool` run, generate detailed BEV plots, an optional interactive 3D HTML view, an index page, and a visualization summary with one command:

```bash
python3 tools/visualize_infer_debug.py demo/centerpoint_pillar demo/centerpoint_pillar_outputs
```

For the common single-sample flow:

```bash
./build/infer_debug_tool --model-dir demo/centerpoint_pillar --input demo/data/test1.bin --source-num-point-features 5 --dump-dir demo/centerpoint_pillar_outputs --dump-decoded --dump-internal --warmup 10 --profile-iters 100
python3 tools/visualize_infer_debug.py demo/centerpoint_pillar demo/centerpoint_pillar_outputs --zmin -0.2 --zmax 3
```

The script reads the selected model directory's `model.yaml` for the point-cloud range and reads final detections from `result.json`, so final box labels use the `label_name` values returned by the high-level C++ API and therefore honor `postprocess.mapped_label_names`.

Typical outputs in the dump directory:

- `01_input_points_height_bev.png`
- `02_input_points_density_bev.png`
- `03_cls_heatmap_bev.png` when `cls_score.npy` exists from `--dump-internal`
- `04_decoded_candidates_bev.png` when decoded arrays exist from `--dump-decoded` or `--dump-internal`
- `05_final_boxes_bev.png`
- `06_3d_final_result.html` when Plotly is available
- `index.html`
- `visualization_summary.json`

For a more detailed run with decoded candidates and internal debug heatmap:

```bash
bash scripts/centerpoint_voxel_head2engine.sh demo/centerpoint_voxel/rpn_centerhead.onnx demo/centerpoint_voxel/rpn_centerhead.engine
./build/infer_debug_tool --model-dir demo/centerpoint_voxel --input demo/data/test1.bin --source-num-point-features 5 --dump-dir demo/centerpoint_voxel_outputs --dump-decoded --dump-internal --warmup 10 --profile-iters 100
python3 tools/visualize_infer_debug.py demo/centerpoint_voxel demo/centerpoint_voxel_outputs --zmin -0.2 --zmax 3
```

Use `--max-candidates` and `--candidate-score-threshold` to reduce clutter when decoded candidate dumps are large.
The default 3D HTML view shows only the final high-level API result. Add `--show-decoded-3d`, `--show-internal-3d`, or `--show-heatmap-3d` only when you explicitly want those debug layers overlaid in 3D.

## Input Formats

### `.bin`

The debug tools accept float32 `.bin` files with 3, 4, or 5 features per point. Use `--source-num-point-features 3|4|5` to declare the source layout explicitly; the option is supported by both `infer_debug_tool` and `perf_debug_tool`.

If the option is omitted, the loader accepts a layout only when exactly one of the 3-, 4-, or 5-feature layouts divides the file's float count. Ambiguous files are rejected with an error asking for an explicit source feature count instead of guessing from file size.

The loader copies the first `min(source features, model.num_point_features)` values for each point and zero-pads any remaining model input channels. Extra source channels are truncated. For example, a 4-feature source loaded into a 5-feature model becomes `[f0, f1, f2, f3, 0]`; a 5-feature source loaded into a 5-feature model preserves all five channels.

The public C++ API does not infer file layouts: callers must provide `PointCloudView` data already packed according to `model.num_point_features`.

### `.pcd`

Current behavior:

- ASCII and binary PCD are supported
- `x`, `y`, `z` must be float32 fields
- optional float32 `intensity` is copied into the 4th runtime feature
- remaining channels in the runtime input buffer are zero-padded
- unsupported headers report the file path, `DATA` format, fields, and point counts

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
- `PostprocessConfig::self_car_box` and `self_car_nms_thresh` are optional together and can also be supplied through `InferWithPostprocess(...)`
- `InferDecoded()` and `Infer()` both return boxes as `[x, y, z, dx, dy, dz, yaw]` with `z` at the box geometric center

## ONNX to TensorRT

Use the helper script to build an engine and then run a quick `trtexec` profile. It supports both a short form with default profile sizes and a full form with explicit profile sizes:

```bash
bash scripts/onnx2engine.sh <onnx_path> <engine_path> [precision] [--cali-data DIR] [--report-dir DIR]
bash scripts/onnx2engine.sh <onnx_path> <engine_path> <min_points> <opt_points> <max_points> [precision] [--cali-data DIR] [--report-dir DIR]
```

Show the built-in help:

```bash
bash scripts/onnx2engine.sh --help
```

Arguments:

- `<onnx_path>`: input ONNX model path
- `<engine_path>`: output TensorRT engine path
- `<min_points>`: optional minimum optimization-profile pillar/point count, default `20000`
- `<opt_points>`: optional optimal optimization-profile pillar/point count, default `30000`
- `<max_points>`: optional maximum optimization-profile pillar/point count, default `40000`
- `[precision]`: optional precision mode, one of `fp32`, `fp16`, `int8`, `fp16int8`, `fp16in8`, or `best`, default `fp16`
- `--cali-data DIR`: optional TensorRT PTQ calibration data directory for `int8`, `fp16int8`, or `best`
- `--report-dir DIR`: optional report directory, default `<engine_path>.reports`

The helper assumes the current repository contract:

- `voxels`: `<N>x20x5`
- `num_points`: `<N>`
- `coors`: `<N>x4`

Default FP16 example:

```bash
bash scripts/onnx2engine.sh demo/centerpoint_pillar/end2end.onnx demo/centerpoint_pillar/end2end.engine
```

Explicit-profile FP16 example:

```bash
bash scripts/onnx2engine.sh demo/centerpoint_pillar/end2end.onnx demo/centerpoint_pillar/end2end.engine 10000 40000 50000
```

FP32 example:

```bash
bash scripts/onnx2engine.sh demo/centerpoint_pillar/end2end.onnx demo/centerpoint_pillar/end2end_fp32.engine 10000 40000 50000 fp32
```

INT8 example for a quantized / Q-DQ ONNX model:

```bash
bash scripts/onnx2engine.sh demo/centerpoint_pillar/end2end.onnx demo/centerpoint_pillar/end2end_int8.engine 10000 40000 50000 int8
```

FP16 plus INT8 PTQ calibration example:

```bash
bash scripts/onnx2engine.sh demo/centerpoint_pillar/end2end.onnx demo/centerpoint_pillar/end2end_fp16int8.engine fp16int8 --cali-data /path/to/cali_data
```

`best` precision example:

```bash
bash scripts/onnx2engine.sh demo/centerpoint_pillar/end2end.onnx demo/centerpoint_pillar/end2end_best.engine best
```

Requirements and behavior:

- `trtexec` must be available in `PATH`
- the ONNX file must already exist
- if profile sizes are omitted, the helper uses `20000 / 30000 / 40000`
- `<min_points>`, `<opt_points>`, and `<max_points>` must be positive integers
- the helper requires `min_points <= opt_points <= max_points`
- the generated engine must preserve the current external binding contract: `voxels=float32`, `num_points=int32`, `coors=int32`, outputs `float32`
- the generated engine must keep a pillar-count optimization profile compatible with the runtime min/max settings
- `--cali-data` samples must contain `pillar_voxels.npy`, `pillar_num_points.npy`, and `pillar_coors.npy`
- without `--cali-data`, if the target engine already exists, regeneration is skipped
- with `--cali-data`, the helper rebuilds the engine and writes a calibration cache next to the engine
- the script still runs a follow-up `trtexec --loadEngine ... --dumpProfile --verbose` pass against the existing engine
- the final profiling pass always uses the opt shape
- report files are written under `--report-dir` or `<engine_path>.reports`, including logs, metadata, summary, and supported `trtexec` layer/profile JSON exports

Build a CenterPoint voxel RPN/head engine from an exported head ONNX:

```bash
bash scripts/centerpoint_voxel_head2engine.sh <head_onnx_path> <head_engine_path> [--report-dir DIR] [--fp32]
```

The CenterPoint voxel head helper expects one FP16 input from `libspconv` and separate FP32 task-head outputs. By default it builds with FP16 compute, writes reports to `<head_engine_path>.reports`, and accepts `--fp32` to disable the FP16 build flag.
The helper adds common TensorRT, Conda, and PyTorch library directories to `LD_LIBRARY_PATH` before running `trtexec`. If `libnvinfer_plugin.so.8` still reports missing dependencies such as `libcudnn.so.8`, install the matching cuDNN 8 runtime or export `LD_LIBRARY_PATH` to the directory that contains it.
