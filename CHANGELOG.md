# Changelog

## v1.2.1

- Added `lidar_model_downloader` to the DEB package install set.
- Installed `lidar_model_downloader` into `/usr/local/bin` so it can be invoked directly on ARM Linux systems after package installation.
- Updated the rebuild packaging flow to run CPack from the build directory and avoid leaving a repository-level `_CPack_Packages` directory.

## v1.2.0

- Fixed the Decoder height parsing issue.
- Accelerated NMS on GPU.
