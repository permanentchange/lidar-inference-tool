#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "${script_dir}/.." && pwd)
build_dir="${repo_root}/build"

if [[ $# -gt 0 ]]; then
  config_path=$1
  shift
else
  config_path="${repo_root}/demo/demo.yaml"
fi

export LD_LIBRARY_PATH="${build_dir}/third_party/iou3d_nms:${build_dir}/third_party/voxelize${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${build_dir}/pillar_debug_tool" "${config_path}" "$@"
