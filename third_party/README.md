# Third-Party Modules

This directory contains dependency-style modules that are versioned inside the
repository.

- `voxelize/`: CUDA voxelization implementation used by the runtime and debug tool
- `iou3d_nms/`: IoU and NMS implementation used by both the public inference path and the debug workflow when IoU BEV NMS is selected

Keep upstream license headers intact when modifying these modules.
