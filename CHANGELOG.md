# Changelog

## Unreleased

- Unified debug CLI named arguments: both tools require `--model-dir`, `--input`, and `--dump-dir`, and use `--profile-iters` for profiling iterations. Removed `perf_debug_tool` positional arguments and `--profile`.
- Enabled decoded and internal dumps by default in `infer_debug_tool`; added `--no-dump-decoded` and `--no-dump-internal` to disable them independently.
- Added grouped CLI help with defaults, shared argument validation, and regression tests for both tools.

## v1.8.0

- Added `model.runtime_backend: centerpoint_voxel` for CenterPoint voxel model packages while keeping the high-level C++ API entry points unchanged.
- Added `libspconv` sparse-convolution integration, generated ONNX protobuf build support, and bundled CUDA-platform `libspconv.so` assets.
- Reorganized demo assets into `demo/centerpoint_pillar/`, `demo/centerpoint_voxel/`, and shared point-cloud inputs under `demo/data/`.
- Added CenterPoint voxel demo model assets, including the SCN ONNX, RPN/head ONNX, and package `model.yaml`.
- Added `scripts/centerpoint_voxel_head2engine.sh` for building the voxel RPN/head TensorRT engine from an exported head ONNX.
- Added regression coverage for package config loading, point-cloud input loading, debug CLI argument handling, minimal C++ API usage text, CenterPoint voxel engine-helper dependency checks, and package architecture contents.

## v1.7.1

- Added optional `postprocess.self_car_box` and `postprocess.self_car_nms_thresh` filtering for `iou_bev` and `iou_min_bev` NMS.
- Kept ego-box filtering inside the existing NMS pass and removed the temporary ego box without consuming configured pre/post NMS capacity.
- C++ integrations using `PostprocessConfig` must rebuild against the updated public header.
- Fixed CUDA compilation of the C++17 public API on ARM CUDA 11.4 and CMake 3.16 by passing `-std=c++17` directly to nvcc.
- Fixed amd64 DEB packaging by excluding the ARM64-only `lidar_model_downloader`; ARM64 packages retain the downloader.

## v1.7.0

- Added directory batch mode to `infer_debug_tool` for top-level `.pcd` / `.bin` input sequences.
- Added ordered per-frame dump directories under `dump_dir/frames/`.
- Added `sequence_manifest.json` and `sequence_detections.jsonl` outputs for downstream Python tracking workflows.
- Added per-frame metadata fields for input path kind, frame index, and filename-derived timestamps.
- Documented `postprocess.nms_type` modes, including `iou_min_bev`.
- Updated the demo postprocess defaults with lower per-label score thresholds and one shared NMS label group.
- Removed `lidar_model_uploader` source and packaged uploader binary from the repository.
- Updated `lidar_model_downloader` to discover and download all release files by default, with `--files` for selective downloads.

## v1.6.1

- Improved the model download helper and ONNX-to-engine build helper workflow for release packaging and deployment.
- Added `lidar_model_uploader` for publishing model release assets to MinIO with upload-time integrity metadata.
- Fixed downloader/uploader MD5 validation to prefer real MinIO object metadata instead of unreliable multipart ETag values.
- Added packaged prebuilt `lidar_model_downloader` and `lidar_model_uploader` binaries from `scripts/dist/`, including ARM release artifacts.

## v1.6.0

- Added `model.yaw_encoding` so CenterPoint model packages can declare `sin_cos` or `direction_theta` yaw decoding explicitly.
- Added compatibility for the newer CenterPoint/MMDeploy `direction_theta` head layout while keeping existing `sin_cos` model packages working unchanged.
- Centralized internal head-layout resolution so package validation, TensorRT output contract checks, CPU decode, and CUDA decode share the same task and channel semantics.
- Kept the recommended high-level C++ API flow unchanged, including `LidarInferenceRuntime::CreateFromModelDir(...)` and `runtime->Infer(...)`.

## v1.5.0

- Added `model.task_label_ids` so CenterPoint task-local output channels can be mapped back to global label ids during decode.
- Updated postprocess to run task/group-aware NMS on global labels and then apply final label mapping for merged business labels.
- Updated the demo model package for the 6-task / 9-label layout, including per-label score thresholds, grouped NMS settings, and mapped final label names.
- Updated infer-debug visualization so final detections use mapped labels while decoded/internal debug candidates still use global labels.
- Added a CMake package version file so downstream projects can use versioned `find_package(...)` checks.
- Updated the package version to `1.5.0`.

## v1.4.0

- Split the old `pillar_debug_tool` workflow into `infer_debug_tool` for high-level C++ API result validation and `perf_debug_tool` for performance profiling.
- Added visualization-friendly inference debug artifacts whose final detections come from `runtime->Infer(...)` and honor mapped label names.
- Consolidated BEV and 3D debug visualization into `tools/visualize_infer_debug.py`.
- Updated the interactive 3D view to show only final high-level API results by default, with explicit debug-layer overlays for decoded candidates and heatmaps.
- Added BEV box heading arrows for final detections using box yaw.
- Updated the demo model package sample and debug-tool documentation.

## v1.3.0

- Changed inference API labels from 1-based output IDs to 0-based output IDs.
- Added ASCII PCD loading support to `pillar_debug_tool`.
- Improved PCD header validation and diagnostics for unsupported point cloud formats.
- Added point cloud load summaries that report the format, fields, point counts, truncation state, feature count, and load time.

## v1.2.1

- Added `lidar_model_downloader` to the DEB package install set.
- Installed `lidar_model_downloader` into `/usr/local/bin` so it can be invoked directly on ARM Linux systems after package installation.
- Updated the rebuild packaging flow to run CPack from the build directory and avoid leaving a repository-level `_CPack_Packages` directory.

## v1.2.0

- Fixed the Decoder height parsing issue.
- Accelerated NMS on GPU.
