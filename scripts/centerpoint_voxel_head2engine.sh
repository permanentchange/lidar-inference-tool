#!/bin/bash

set -euo pipefail

prepend_ld_library_path() {
  local lib_dir=$1
  if [ -d "$lib_dir" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$lib_dir:"*) ;;
      *) export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
  fi
}

configure_runtime_library_path() {
  if [ -n "${CONDA_PREFIX:-}" ]; then
    prepend_ld_library_path "${CONDA_PREFIX}/lib"
  fi

  local torch_lib_dir=""
  if command -v python >/dev/null 2>&1; then
    torch_lib_dir=$(python - <<'PY'
import os

try:
    import torch
except Exception:
    raise SystemExit(0)

lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.isdir(lib_dir):
    print(lib_dir)
PY
)
  fi
  prepend_ld_library_path "$torch_lib_dir"

  local fallback_conda_env="${ONNX2ENGINE_CONDA_ENV:-openmmlab_py38}"
  local fallback_torch_lib
  for fallback_torch_lib in "$HOME"/miniconda3/envs/"$fallback_conda_env"/lib/python*/site-packages/torch/lib; do
    prepend_ld_library_path "$fallback_torch_lib"
  done

  local trtexec_path=""
  trtexec_path=$(command -v trtexec 2>/dev/null || true)
  if [ -n "$trtexec_path" ]; then
    local trt_lib_dir=""
    trt_lib_dir=$(dirname "$(dirname "$trtexec_path")")/lib
    prepend_ld_library_path "$trt_lib_dir"
  fi
}

find_trt_plugin_library() {
  local trtexec_path=""
  trtexec_path=$(command -v trtexec 2>/dev/null || true)
  if [ -z "$trtexec_path" ]; then
    return 1
  fi

  local trt_lib_dir
  trt_lib_dir=$(dirname "$(dirname "$trtexec_path")")/lib
  local candidate
  for candidate in "$trt_lib_dir"/libnvinfer_plugin.so*; do
    if [ -f "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

check_trtexec_runtime_dependencies() {
  if ! command -v ldd >/dev/null 2>&1; then
    return 0
  fi

  local plugin_library=""
  plugin_library=$(find_trt_plugin_library || true)
  if [ -z "$plugin_library" ]; then
    return 0
  fi

  local ldd_output=""
  ldd_output=$(ldd "$plugin_library" 2>&1 || true)
  if grep -Fq -- "libcudnn.so.8 => not found" <<<"$ldd_output"; then
    echo "Missing TensorRT runtime dependency: libcudnn.so.8" >&2
    echo "TensorRT plugin library: $plugin_library" >&2
    echo "Install cuDNN 8 or export LD_LIBRARY_PATH to the directory containing libcudnn.so.8." >&2
    exit 1
  fi

  local missing_deps=""
  missing_deps=$(awk '/=> not found/ { print $1 }' <<<"$ldd_output")
  if [ -n "$missing_deps" ]; then
    echo "Missing TensorRT runtime dependencies for $plugin_library:" >&2
    sed 's/^/  - /' <<<"$missing_deps" >&2
    echo "Install the missing libraries or export LD_LIBRARY_PATH to their directory." >&2
    exit 1
  fi
}

usage() {
  cat <<'EOF_USAGE'
Usage:
  scripts/centerpoint_voxel_head2engine.sh <head_onnx_path> <head_engine_path> [--report-dir DIR] [--fp32]

Build the TensorRT engine used by model.runtime_backend=centerpoint_voxel.

The runtime expects:
  - one float16 input from libspconv SparseEncoder/SCN
  - separate float32 outputs for each CenterPoint task head:
      reg/height/dim/rot/vel/hm, or TensorRT out0..out35 in mmdetection3d export order

Default precision is fp16 compute with fp16 input and fp32 outputs.
EOF_USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -lt 2 ]; then
  usage >&2
  exit 1
fi

onnx_path=$1
engine_path=$2
shift 2

precision_flag="--fp16"
report_dir="${engine_path}.reports"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --report-dir)
      if [ "$#" -lt 2 ]; then
        echo "--report-dir requires a value" >&2
        exit 1
      fi
      report_dir=$2
      shift 2
      ;;
    --fp32)
      precision_flag=""
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ ! -f "$onnx_path" ]; then
  echo "ONNX file not found: $onnx_path" >&2
  exit 1
fi

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found in PATH" >&2
  exit 1
fi

configure_runtime_library_path
check_trtexec_runtime_dependencies

mkdir -p "$(dirname "$engine_path")" "$report_dir"

set +e
trtexec \
  --onnx="$onnx_path" \
  --saveEngine="$engine_path" \
  ${precision_flag} \
  --inputIOFormats=fp16:chw \
  --outputIOFormats=fp32:chw \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  --dumpProfile \
  --separateProfileRun \
  >"${report_dir}/build.log" 2>&1
trtexec_status=$?
set -e

if [ "$trtexec_status" -ne 0 ]; then
  echo "trtexec failed while building CenterPoint voxel head engine." >&2
  echo "Build log: ${report_dir}/build.log" >&2
  echo "Last build log lines:" >&2
  tail -n 40 "${report_dir}/build.log" >&2
  exit "$trtexec_status"
fi

{
  echo "onnx_path=$onnx_path"
  echo "engine_path=$engine_path"
  echo "precision=${precision_flag:---fp32}"
  echo "input_io_formats=fp16:chw"
  echo "output_io_formats=fp32:chw"
} >"${report_dir}/metadata.txt"
