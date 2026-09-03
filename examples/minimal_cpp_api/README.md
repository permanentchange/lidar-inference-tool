# Minimal C++ API Example

This example shows the smallest high-level C++ integration path for the library:

- load `model.yaml` from a model directory
- create one `inference::LidarInferenceRuntime`
- load one float32 `.bin` point cloud
- call `runtime->Infer(...)`
- print the final detections

## Build

From the repository root:

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

The example executable is written to:

- `build/minimal_cpp_api_example`

## Run

Run the executable built under `build/`; `examples/minimal_cpp_api/` is the source directory, not an executable:

```bash
./build/minimal_cpp_api_example demo/centerpoint_pillar demo/data/test.bin
./build/minimal_cpp_api_example demo/centerpoint_voxel demo/data/test.bin
```

Arguments:

- `MODEL_DIR`: directory containing `model.yaml` and the referenced TensorRT / runtime assets
- `POINT_CLOUD_BIN`: one float32 `.bin` point-cloud path

Core API flow shown by the example:

- `inference::LidarInferenceRuntime::CreateFromModelDir(...)`
- `inference::PointCloudView`
- `runtime->Infer(...)`
- `std::vector<inference::Detection>` for the final NMS-filtered result

Important semantics:

- `runtime->Infer(...)` returns final detections after built-in NMS using the `postprocess` section from `model.yaml`
- if you need decoded pre-NMS candidates instead, use `runtime->InferDecoded(...)`
- one runtime instance should be owned by one inference thread; create multiple runtime instances for concurrent inference
