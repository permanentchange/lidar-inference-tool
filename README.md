# lidar-inference-tool

`lidar-inference-tool` provides two main deliverables:

- A shared C API library: `libinference.so`
- A single-sample debug executable: `pillar_debug_tool`

The current implementation targets PointPillars / CenterPoint-style TensorRT
end-to-end engines. It takes point clouds as input and returns decoded
detections sorted by score.

## Repository Metadata

- Contribution guidelines: `CONTRIBUTING.md`
- License: `LICENSE`
- Local build products and debug outputs are covered by `.gitignore`

## Repository Layout

- `include/inference.h`: stable public C API
- `src/pillar_runtime.cpp`: TensorRT engine loading, voxelization, inference, and decode
- `src/pillar_debug_utils.cpp`: debug config parsing, NMS, and artifact export
- `tools/pillar_debug_tool.cpp`: single-sample debug entry point
- `tools/onnx2engine.sh`: helper script that converts ONNX to TensorRT and profiles the engine with `trtexec`
- `demo/demo.yaml`: default demo configuration
- `demo/end2end.onnx`: example ONNX model
- `demo/end2end.engine`: example TensorRT engine
- `demo/pcd.bin`: example point cloud
- `scripts/visualize_bev.py`: BEV visualization script
- `third_party/voxelize/`: vendored CUDA voxelization module
- `third_party/iou3d_nms/`: low-level IoU and NMS module used by the debug tool

## Requirements

### Native inference build

- Linux
- CMake >= 3.0.2
- GCC / G++ with C++17 support
- CUDA Toolkit
- TensorRT
- System `yaml-cpp` development package

Install the system YAML dependency on Ubuntu:

```bash
sudo apt-get install -y libyaml-cpp-dev
```

The build does not support Conda-provided include or library paths. If your
existing `build/` directory was configured with Conda flags, remove it and
configure again.

### Python utilities

- Python 3
- `numpy`
- `pyyaml`
- `matplotlib`

Install Python dependencies with:

```bash
python3 -m pip install numpy pyyaml matplotlib
```

## Build

Run all commands from the repository root.

### Fast path

Start with this exact command:

```bash
cmake -S . -B build && cmake --build build -j$(nproc)
```

This passes when TensorRT auto-detection succeeds. The current CMake logic
checks common system paths, common extracted TensorRT directories, and paths
that can be inferred from `trtexec`.

### If the fast path fails

If configure fails because TensorRT was not found, rerun with an explicit
TensorRT root:

```bash
cmake -S . -B build -DTENSORRT_ROOT=/path/to/TensorRT && cmake --build build -j$(nproc)
```

### If the cache is stale or polluted

If `build/` was copied from another machine or was previously configured with
Conda-managed paths, remove it and reconfigure locally:

```bash
rm -rf build
cmake -S . -B build -DTENSORRT_ROOT=/path/to/TensorRT && cmake --build build -j$(nproc)
```

### Optional stub build

The default configuration is strict: if CUDA or TensorRT is missing, CMake
fails during configure. To build the stub library explicitly, enable the stub
switch:

```bash
cmake -S . -B build -DBUILD_STUB_IF_NO_TENSORRT=ON
cmake --build build -j$(nproc)
```

Successful full builds generate:

- `build/libinference.so`
- `build/pillar_debug_tool`

Stub builds only generate `build/libinference.so`, and exported API calls such
as `inference_create()` and `inference_forward()` return
`INFERENCE_STATUS_NOT_SUPPORTED`.

## Engine Contract

The runtime expects the TensorRT engine to expose the following bindings.

Input bindings:

- `voxels`
- `num_points`
- `coors`

Output bindings:

- `bbox_pred0`
- `cls_score0`
- `dir_cls_pred0`

The decode logic assumes three tasks:

- `1 -> car`
- `2 -> pedestrian`
- `3 -> cyclist`

Expected head layout:

- `bbox_pred0`: 8 channels per task
- `dir_cls_pred0`: 2 channels per task
- `cls_score0`: 1 heatmap channel per task

Initialization fails if the engine bindings, dtypes, or output tensor layout do
not match these assumptions.

## Debug Tool Workflow

### Configuration

The default debug config is `demo/demo.yaml`.

All path fields in the YAML are resolved relative to the YAML file itself, so
the demo stays portable across machines.

Important fields:

- `engine_path`: TensorRT engine file
- `pcd_path`: input point cloud, supports `.bin` and `.pcd`
- `dump_dir`: output directory for debug artifacts
- `runtime.*`: voxelization and decode parameters
- `post_processing.*`: NMS and final score thresholds

The shipped demo config points at files inside `demo/`:

- `end2end.engine`
- `pcd.bin`
- `demo_output`

### Run a single-sample debug session

```bash
./build/pillar_debug_tool demo/demo.yaml
```

Typical console output includes:

- point cloud load time
- input point count
- pillar count
- decoded candidate count
- final NMS box count
- single-run inference latency
- profile summary timings

### Generate BEV visualizations

```bash
python3 scripts/visualize_bev.py demo/demo.yaml
```

Generated files under `dump_dir`:

- `01_input_points_bev.png`
- `02_heatmap_bev.png`
- `03_decoded_candidates_bev.png`
- `04_final_boxes_bev.png`

### Common debug artifacts

`pillar_debug_tool` usually writes the following files into `dump_dir`:

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

- `result.json` contains the exported final detection output
- `timings.json` contains the profile summary
- `metadata.json` contains tensor summaries for the dump

## ONNX to TensorRT

Use `tools/onnx2engine.sh` to convert an ONNX model into a TensorRT engine and
run a quick `trtexec` profile.

```bash
bash tools/onnx2engine.sh <onnx_path> <engine_path> <min_points> <opt_points> <max_points>
```

Example:

```bash
bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end.engine 10000 40000 50000
```

Requirements:

- `trtexec` must be available in `PATH`
- the ONNX file must already exist

Behavior:

- if the target engine already exists, the script skips regeneration
- after conversion, it runs `trtexec --loadEngine ... --dumpProfile --verbose`

## Input Formats

### `.bin`

The loader supports `float32` point clouds with 3, 4, or 5 features per point.

Current behavior:

- 3D input is interpreted as `x y z`
- 4D input is interpreted as `x y z intensity`
- 5D input matches `InferTools PointCloudLoader(use_timestamp=False)`: only the
  first 4 values are forwarded and the 5th feature is zero-padded

In other words, a 5D input file does not pass the 5th feature through to the
model verbatim.

### `.pcd`

Only binary PCD is supported.

Current behavior:

- `x y z` are loaded
- intensity is filled with `0`

## Public API Usage

Public header:

- `include/inference.h`

Exported symbols:

- `inference_create`
- `inference_get_output_spec`
- `inference_forward`
- `inference_reset`
- `inference_destroy`
- `inference_last_error`
