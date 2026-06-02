# Changelog

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
