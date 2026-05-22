# Changelog

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
