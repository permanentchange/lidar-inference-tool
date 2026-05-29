# Contributing

## Scope

This repository contains a C++/CUDA TensorRT inference runtime, auxiliary kernels,
and Python scripts for single-sample debugging and result comparison.

## Development setup

### Native build

Use an out-of-tree CMake build:

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

When CUDA or TensorRT is unavailable, the top-level build falls back to the stub
library and can still be used for API-level compile checks.

### Python utilities

The debug scripts in `scripts/` require:

- `numpy`
- `pyyaml`
- `matplotlib`

Install them with:

```bash
python3 -m pip install numpy pyyaml matplotlib
```

## Repository conventions

- Keep generated files in `artifacts/`, `outputs/`, or another local-only folder.
- Do not commit point cloud dumps, profiling outputs, or debug images.
- Avoid committing machine-specific `build/` directories or CMake cache files.
- `third_party/` contains dependency-style modules; update them deliberately and
  keep any upstream license headers intact.

## Change expectations

- Keep functional changes and refactors in separate commits when practical.
- Update `README.md` when build steps, engine contracts, or debug workflows change.
- Prefer adding or extending verification steps when changing inference logic or
  tensor layout assumptions.

## Verification

Before submitting changes, run whichever of the following applies to your
environment:

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

If you have a valid engine and sample data available, also run:

```bash
./build/infer_debug_tool demo demo/pcd.bin outputs/infer_debug --dump-decoded
./build/perf_debug_tool demo demo/pcd.bin outputs/perf_debug --warmup 10 --profile 50
```
