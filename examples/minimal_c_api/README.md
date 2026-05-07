# Minimal C API Example

This example shows the smallest end-to-end use of `include/inference.h` from a standalone executable.
For the recommended higher-level C++ integration path, see `examples/minimal_cpp_api/`.

It deliberately uses the low-level API so that callers can see the exact C entry points:

- fill one `InferenceConfig`
- create one `InferenceHandle`
- query `InferenceOutputSpec`
- allocate output buffers
- call `inference_forward_nms()` for final detections

## Build

From the repository root:

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

The example executable is written to:

- `build/minimal_c_api_example`

## Run

Use the repository demo assets by default:

```bash
./build/minimal_c_api_example demo/end2end.engine demo/pcd.bin
```

Arguments:

- `<engine_path>`: TensorRT engine path
- `<pcd_bin_path>`: one float32 `.bin` point cloud path

Input expectations:

- the point cloud file must contain `float32` values
- this example uses the same runtime settings as `demo/model.yaml`
- the example reads `.bin` input with 5 features per point
- the example uses `inference_get_default_postprocess_config()` and `inference_forward_nms()`
- reported detections are therefore final post-NMS results, not decoded pre-NMS candidates
- the example prints the final detections directly
