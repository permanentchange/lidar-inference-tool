#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "${script_dir}/.." && pwd)
build_dir="${repo_root}/build"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model_dir> <pcd_path> [dump_dir] [--warmup N] [--profile N] [--bev-resolution X] [--override-config PATH]" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${build_dir}/third_party/iou3d_nms:${build_dir}/third_party/voxelize${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${build_dir}/pillar_debug_tool" "$@"
