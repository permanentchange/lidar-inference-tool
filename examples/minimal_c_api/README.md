# Minimal C API Example

This example shows the shortest end-to-end use of `include/inference.h` from a standalone executable.
For the recommended higher-level C++ integration path, see `examples/minimal_cpp_api/`.

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

Process multiple point clouds in one run:

```bash
./build/minimal_c_api_example demo/end2end.engine demo/pcd.bin demo/pcd.bin
```

Arguments:

- `<engine_path>`: TensorRT engine path
- `<pcd_bin_path>`: one or more float32 `.bin` point cloud paths

Input expectations:

- the point cloud file must contain `float32` values
- this example uses the same runtime settings as `demo/model.yaml`
- the example reads `.bin` input with 5 features per point
- files are loaded one by one into a reused host buffer to reduce repeated allocations
- output buffers are allocated once from `InferenceOutputSpec` and reused for every file
- load or inference failures are reported and skipped; the program returns nonzero if any file fails

The program prints a batch summary with processed file counts, total points, total detected
objects, aggregate timings, and any failed file paths.
