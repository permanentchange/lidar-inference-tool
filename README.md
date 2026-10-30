# lidar-inference-tool

`lidar-inference-tool` is a C++17 LiDAR inference project built around TensorRT LiDAR detection models. It supports the original end-to-end pillar package path, a CenterPoint voxel package path, and a separate camera/LiDAR BEVFusion runtime. The repository exposes:

- `libinference.so`: a shared library exposing the recommended C++ wrapper for loading one model package and running decoded or final NMS-filtered detections
- `infer_debug_tool`: a single-file or batch debug executable that validates high-level C++ API inference results and writes visualization-friendly artifacts
- `perf_debug_tool`: a profiling executable for single-sample inference performance breakdowns
- `lidar_model_downloader`: an ARM64 package command-line helper for fetching model assets after package installation
- `minimal_cpp_api_example`: a model-package-driven executable under `examples/` that shows the recommended C++ integration path

## v1.8.0

- Added `centerpoint_voxel` model-package support through the same high-level C++ API used by pillar packages.
- Added bundled `libspconv` sparse-convolution integration for voxel sparse encoder / SCN execution.
- Reorganized demo assets into pillar, voxel, and shared point-cloud input directories.
- Added `scripts/centerpoint_voxel_head2engine.sh` for building the voxel RPN/head TensorRT engine.

See [CHANGELOG.md](./CHANGELOG.md) for version history.

PointPillars / CenterPoint use the existing LiDAR C++ wrapper. BEVFusion uses `include/bevfusion_inference.hpp`, a sparse ONNX graph and two TensorRT engines, calibrated camera manifests and the same two debug tools. See [BEVFusion deployment](scripts/bevfusion/README.md) for checkpoint preparation, raw/preprocessed image inputs, precision and validation.

## Shared CenterHead deployment

CenterPoint Voxel and BEVFusion use the same deployment configuration fields,
GPU decoder, globally grouped NMS, ego-box filtering and final business-label mapping.
See [configuration synchronization and run commands](docs/centerhead_deployment.md).
The checked-in default profile is `scripts/configs/centerpoint_deployment.yaml`.

## Privacy Note

This README intentionally uses repository-relative paths and placeholders such as `/path/to/TensorRT`. Do not replace examples with host-specific absolute paths, usernames, home directories, or machine identifiers when sharing configs or logs.

## Repository Layout

- `include/bevfusion_inference.hpp`: public multimodal C++ API
- `scripts/bevfusion/`: local ONNX export, independent TensorRT build, numerical checks, full evaluation and a YAML-driven Bash pipeline
- `include/inference_cpp.hpp`: public C++ wrapper API
- `include/inference.h`: internal C compatibility header used by the wrapper and runtime implementation
- `src/inference_api.cpp`: internal compatibility-layer entry points used by the wrapper
- `src/pillar_runtime.cpp`: TensorRT loading, voxelization, inference, decode, and response packing
- `src/centerhead_postprocess.cpp`: common candidate sorting, filtering and grouped NMS
- `src/postprocess_config.cpp`: common postprocess parsing, validation and output-label mapping
- `src/pillar_debug_utils.cpp`: debug-tool config assembly, point-cloud loading, and artifact export
- `tools/debug_cli.hpp`: shared debug CLI argument parsing, defaults, and help
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
- `demo/data/`: original point-cloud and multimodal frame inputs
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
sudo apt-get install -y libyaml-cpp-dev protobuf-compiler libprotobuf-dev libjpeg-dev
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

Run commands from the repository root. Single-config builds default to `Release`; use `-DCMAKE_BUILD_TYPE=Debug` when needed.

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
- `build/minimal_cpp_api_example` and `build/bevfusion_cpp_api_example` when `BUILD_EXAMPLES=ON`

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

Run a pillar API result debug session. Decoded detections and internal tensors are dumped by default; the BIN sample requires its source feature count:

```bash
./build/infer_debug_tool \
  --model-dir demo/centerpoint_pillar \
  --input demo/data/lidar/test1.bin \
  --dump-dir demo/centerpoint_pillar_outputs \
  --source-num-point-features 5
```

For the voxel demo, build the TensorRT head engine if it is not already available, then run inference on a PCD file. PCD fields are read from the file header, so no source feature-count argument is needed:

```bash
bash scripts/centerpoint_voxel_head2engine.sh demo/centerpoint_voxel/rpn_centerhead.onnx demo/centerpoint_voxel/rpn_centerhead.engine
./build/infer_debug_tool \
  --model-dir demo/centerpoint_voxel \
  --input demo/data/lidar/test3.pcd \
  --dump-dir demo/centerpoint_voxel_outputs
```

Profile the same input with the performance tool. Its defaults are 10 warmup iterations and 50 measured iterations:

```bash
./build/perf_debug_tool \
  --model-dir demo/centerpoint_voxel \
  --input demo/data/lidar/test3.pcd \
  --dump-dir outputs/voxel_perf
```

Use `./build/infer_debug_tool -h` or `./build/perf_debug_tool -h` for grouped help and defaults. See [Debug CLIs](#debug-clis) for batch inference, dump controls, and timing options.

The demo assets used by the public C++ API and debug tools are:

- `demo/centerpoint_pillar/model.yaml`
- `demo/centerpoint_pillar/end2end.engine`
- `demo/centerpoint_voxel/model.yaml`
- `demo/centerpoint_voxel/centerpoint.scn.onnx`
- `demo/centerpoint_voxel/rpn_centerhead.onnx`
- `demo/centerpoint_voxel/rpn_centerhead.engine` after running `scripts/centerpoint_voxel_head2engine.sh`
- `demo/data/lidar/test1.bin`
- `demo/data/lidar/test1.pcd`
- `demo/data/lidar/test2.pcd`
- `demo/data/lidar/test3.pcd`

Typical `perf_debug_tool` console output includes:

- point-cloud load time
- input point count
- pillar or voxel count, depending on the model backend
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

The CenterPoint Voxel and BEVFusion demos share the configured per-label score thresholds and run one NMS pass over all nine global labels. Override these values in `model.yaml` or an override YAML when a deployment needs class-specific NMS groups.

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

## BEVFusion deployment and shared demos

BEVFusion uses the current CenterHead implementation: a libspconv ZYX sparse
backbone, camera TensorRT engine, CUDA geometry/pooling, and a fusion/CenterHead
TensorRT engine. See [deployment](scripts/bevfusion/README.md) and
[model details](scripts/bevfusion/MODEL.md). The old TransFusion package is no
longer supported. Its existing engine/ONNX files remain untouched.

All example inputs live under [demo/data](demo/data/README.md). Four multimodal
frames retain their original PCD, five JPEGs, camera configuration, annotations,
and source hashes. Three selected scenes cover mixed traffic, sparse pedestrians,
and a disabled camera. The same original PCD can be passed directly to CenterPoint.

```bash
conda activate openmmlab_py38
export CUDA_VISIBLE_DEVICES=5
export TENSORRT_ROOT=/home/zhouxiufeng/TensorRT-8.6.1.6
export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib:/usr/local/cuda-11.7/lib64:$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

./build/infer_debug_tool --model-dir demo/centerpoint_voxel \
  --input demo/data/bevfusion/frames/1776582550193446516/lidar_point_cloud_0/1776582550193446516.pcd \
  --dump-dir outputs/mixed/centerpoint --no-dump-internal --no-dump-decoded
./build/infer_debug_tool --model-dir demo/bevfusion_exp20261028 \
  --input demo/data/bevfusion/manifests/1776582550193446516.yaml \
  --dump-dir outputs/mixed/bevfusion --no-dump-internal --no-dump-decoded
python scripts/compare_demo_frames.py
```

## Debug tools and reports

Both tools accept `--model-dir`, `--input`, `--dump-dir`,
`--override-config`, `--source-num-point-features`, `--warmup`, and
`--profile-iters`. BEVFusion also accepts `--image-input raw|preprocessed`.
PCD fields come from its header; ambiguous BIN layouts require the source feature
count. Runtime backend selection comes from the model package.

| Behavior | infer_debug_tool | perf_debug_tool |
|---|---|---|
| Purpose | Inspect detections and tensors | Measure latency and stages |
| Input | Single frame or flat directory | Single frame |
| Default warmup / samples | 0 / 1 | 10 / 50 per pass |
| Default result / decoded / internal dumps | On / on / on | Off / off / off |
| Profile | Normal inference total | Normal total, then a separate stage pass |

Each dump type has common positive/negative switches:
`--dump-result` / `--no-dump-result`, `--dump-decoded` /
`--no-dump-decoded`, `--dump-internal` / `--no-dump-internal`.
The last switch wins. Extra dump inference and file writes occur after profiling.

```bash
./build/perf_debug_tool --model-dir demo/centerpoint_voxel \
  --input demo/data/lidar/test3.pcd --dump-dir outputs/centerpoint_perf
./build/perf_debug_tool --model-dir demo/bevfusion_exp20261028 \
  --input demo/data/bevfusion/manifests/1776582550193446516.yaml \
  --dump-dir outputs/bevfusion_perf
./build/infer_debug_tool --model-dir demo/bevfusion_exp20261028 \
  --input demo/data/bevfusion/manifests --dump-dir outputs/bevfusion_sequence \
  --no-dump-internal --no-dump-decoded
```

Total latency is wall time for the complete synchronous inference call, including
upload, preprocessing, networks, decoding, NMS and result packing. It excludes
runtime initialization, file/JPEG loading, extra diagnostic inference, and disk
writes. Stage collection is disabled during the total pass. Perf then repeats
warmup and measurement with stage collection enabled; both passes reuse one
runtime and loaded frame. `--warmup 0` performs no hidden inference. Geometry
cache misses are reported separately for each pass. CUDA events are collected
at the end rather than adding synchronization after every stage.

Terminal output and JSON reports use the same backend-independent statistics:
`n`, `mean`, `min`, `max`, `p50`, `p90`, in milliseconds. Host stage time includes
submission/waits; CUDA event spans measure stream elapsed time, not exclusively
GPU busy time. Stage and total values come from separate passes and must not be
summed or mixed. CPU-only stages have null CUDA statistics.

Every frame writes `metadata.json` and `timings.json` with `schema_version: 2`:

- Metadata: mode/backend/model, source input, counts, device, effective switches,
  frame ID/timestamp and artifact list.
- Timings: `setup.runtime_init_ms`, `setup.input_load_ms`, `total_infer_ms`
  statistics, pass settings/cache misses, and `stage_profile.stages` (null in
  infer mode). Each stage contains `name`, `host_ms` and optional `gpu_ms` stats.
- The previous `stage_timings.json` and timing fields embedded in metadata have
  been replaced; repository consumers use the new report.

With result dumps enabled, tools also write `result.json`, `input_points.npy`,
`final_{boxes,scores,labels}.npy`, `final_label_names.json` and available velocities.
`result.json.objects` contains labels/names, scores and geometric-center
`[x,y,z,dx,dy,dz,yaw]` boxes; velocity is optional. Decoded arrays are independent
of final result dumps. Backend-specific internal NPY tensors live in `internal/`.

Directory inference sorts numeric stems numerically (otherwise by filename),
rejects duplicate frame IDs, and reuses the runtime. Per-frame outputs live in
`frames/<six-digit-index>_<stem>/`; the root writes `sequence_manifest.json` and,
when results are enabled, `sequence_detections.jsonl`. Numeric filename stems are
interpreted as nanosecond timestamps.

## Infer visualization

```bash
python tools/visualize_infer_debug.py demo/centerpoint_voxel outputs/mixed/centerpoint
```

The viewer reads final JSON/arrays and optional tensors from `internal/`. For
same-frame model comparisons, use `scripts/compare_demo_frames.py`: it generates
native result directories and ground-truth/CenterPoint/BEVFusion BEV plots.

## Input Formats

### `.bin`

The debug tools accept float32 `.bin` files with 3, 4, or 5 features per point. Use `--source-num-point-features 3|4|5` to declare the source layout explicitly; the option is supported by both `infer_debug_tool` and `perf_debug_tool`.

These BIN files contain raw float32 values without a header describing point boundaries. The source feature count describes the stored file layout, independently of `model.num_point_features`. Use the layout specified by the data exporter: a different count may still divide the file size while grouping coordinates and features incorrectly. The repository's `demo/data/lidar/test1.bin` examples use `--source-num-point-features 5`.

If the option is omitted, the loader accepts a layout only when exactly one of the 3-, 4-, or 5-feature layouts divides the file's float count. Ambiguous files are rejected with an error asking for an explicit source feature count instead of guessing from file size.

The loader copies the first `min(source features, model.num_point_features)` values for each point and zero-pads any remaining model input channels. Extra source channels are truncated. For example, a 4-feature source loaded into a 5-feature model becomes `[f0, f1, f2, f3, 0]`; a 5-feature source loaded into a 5-feature model preserves all five channels.

The public C++ API does not infer file layouts: callers must provide `PointCloudView` data already packed according to `model.num_point_features`.

### `.pcd`

Current behavior:

- ASCII and binary PCD are supported; `binary_compressed` is not supported
- fields and their storage types are read from `FIELDS`, `SIZE`, `TYPE`, and `COUNT`, so `--source-num-point-features` is unnecessary for PCD, including binary PCD
- `x`, `y`, `z` must be float32 scalar fields
- optional scalar `intensity` and `has_intensity` may use float32/float64 or signed/unsigned 8-, 16-, 32-, or 64-bit integers; values are converted to float32 and copied into the 4th and 5th runtime features when those channels exist
- when `intensity` exists but `has_intensity` is absent, the 5th runtime feature is set to `1`; other missing channels are zero-padded
- unsupported headers report the file path, `DATA` format, fields, and point counts

For a model with five input features, common PCD layouts become:

| PCD fields | Runtime point |
| --- | --- |
| `x y z` | `[x, y, z, 0, 0]` |
| `x y z intensity` | `[x, y, z, intensity, 1]` |
| `x y z intensity has_intensity` | `[x, y, z, intensity, has_intensity]` |

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
