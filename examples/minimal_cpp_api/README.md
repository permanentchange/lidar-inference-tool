# Minimal C++ API Example

This example shows the recommended high-level integration path for a C++ project:

- place `model.yaml` and the TensorRT engine referenced by `model.engine_path` in one model directory
- create one `inference::LidarInferenceRuntime` at startup
- reuse that runtime for every point cloud frame

## Build

From the repository root:

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

The example executable is written to:

- `build/minimal_cpp_api_example`

## Run

Use the repository demo model package:

```bash
./build/minimal_cpp_api_example demo demo/pcd.bin
```

Process multiple point clouds in one run:

```bash
./build/minimal_cpp_api_example demo demo/pcd.bin demo/pcd.bin
```

Arguments:

- `<model_dir>`: directory containing `model.yaml` and the TensorRT engine
- `<pcd_bin_path>`: one or more float32 `.bin` point cloud paths

What the example does:

- loads `demo/model.yaml`
- creates one runtime via `CreateFromModelDir`
- reuses internal output buffers across all frames
- runs final NMS-filtered inference through `runtime->Infer(...)`
- prints aggregate timings and a preview of the last frame detections
