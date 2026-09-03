# Third-Party Modules

This directory contains dependency-style modules that are versioned inside the
repository.

- `voxelize/`: CUDA voxelization implementation used by the runtime and debug tool
- `iou3d_nms/`: IoU and NMS implementation used by both the public inference path and the debug workflow when IoU BEV NMS is selected
- `libspconv/`: NVIDIA 3DSparseConvolution sparse-convolution runtime, headers, ONNX parser, and ONNX protobuf schemas used by `model.runtime_backend: centerpoint_voxel`

Keep upstream license headers intact when modifying these modules.

The CenterPoint voxel backend generates the ONNX protobuf C++ sources in the
build tree using the `protoc` selected by CMake's `find_package(Protobuf)`, then
compiles the NVIDIA ONNX parser into `libinference.so`. The selected `protoc`,
headers, and `libprotobuf` must have the same version. Generated protobuf files
are build artifacts and are not versioned in the source tree.

The backend links directly against the bundled `libspconv.so` plus system
`libprotobuf`. Deployment packages that use the voxel backend must ship a
`libspconv.so` build compatible with the target CUDA platform and its dependent
libraries.
