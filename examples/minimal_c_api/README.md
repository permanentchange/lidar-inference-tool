# Minimal C API Example

This example shows the shortest end-to-end use of `include/inference.h` from a standalone executable.

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
- `<pcd_bin_path>`: float32 `.bin` point cloud path

Input expectations:

- the point cloud file must contain `float32` values
- this example uses the same runtime settings as `demo/demo.yaml`
- the example reads `.bin` input with 5 features per point

The program prints the total detection count and up to five decoded detections.
