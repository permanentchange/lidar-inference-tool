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
  prepend_ld_library_path "${CONDA_PREFIX:-}/lib"

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
}

configure_runtime_library_path

usage() {
  cat <<'EOF'
Usage:
  bash tools/onnx2engine.sh <onnx_path> <engine_path> <min_points> <opt_points> <max_points> [fp32|fp16|int8]

Arguments:
  <onnx_path>     Input ONNX model path.
  <engine_path>   Output TensorRT engine path.
  <min_points>    Min optimization-profile pillar/point count.
  <opt_points>    Opt optimization-profile pillar/point count.
  <max_points>    Max optimization-profile pillar/point count.
  [precision]     Optional precision mode. Default: fp16.

This helper assumes the exported ONNX/TensorRT contract used by this repository:
  - voxels:     <N>x20x5
  - num_points: <N>
  - coors:      <N>x4

Examples:
  bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end.engine 10000 40000 50000
  bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end_fp32.engine 10000 40000 50000 fp32
  bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end_int8.engine 10000 40000 50000 int8

Behavior:
  1. If <engine_path> does not exist, build the TensorRT engine with trtexec.
  2. Load the resulting engine and run a short profile at the opt shape.
  3. Always print the profiling output from the final trtexec run.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
  echo "Invalid argument count: expected 5 or 6, got $#." >&2
  usage >&2
  exit 1
fi

onnx_path=$1
trt_path=$2
min_num_points=$3
opt_num_points=$4
max_num_points=$5
precision=${6:-fp16}

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec was not found in PATH." >&2
  echo "Install TensorRT tools or export PATH to include the trtexec binary." >&2
  exit 1
fi

if [ ! -f "$onnx_path" ]; then
  echo "ONNX model file not found: $onnx_path" >&2
  exit 1
fi

for value_name in min_num_points opt_num_points max_num_points; do
  value=${!value_name}
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -le 0 ]; then
    echo "$value_name must be a positive integer, got: $value" >&2
    exit 1
  fi
done

if [ "$min_num_points" -gt "$opt_num_points" ] || [ "$opt_num_points" -gt "$max_num_points" ]; then
  echo "Require min_points <= opt_points <= max_points, got: $min_num_points <= $opt_num_points <= $max_num_points" >&2
  exit 1
fi

precision_flag=""
case "$precision" in
  fp32)
    ;;
  fp16)
    precision_flag="--fp16"
    ;;
  int8)
    precision_flag="--int8"
    ;;
  *)
    echo "Unsupported precision '$precision'. Use fp32, fp16, or int8." >&2
    exit 1
    ;;
esac

shape_min="coors:${min_num_points}x4,voxels:${min_num_points}x20x5,num_points:${min_num_points}"
shape_opt="coors:${opt_num_points}x4,voxels:${opt_num_points}x20x5,num_points:${opt_num_points}"
shape_max="coors:${max_num_points}x4,voxels:${max_num_points}x20x5,num_points:${max_num_points}"

echo "ONNX input:      $onnx_path"
echo "Engine output:   $trt_path"
echo "Precision mode:  $precision"
echo "Min TRT shape:   $shape_min"
echo "Opt TRT shape:   $shape_opt"
echo "Max TRT shape:   $shape_max"

if [ -f "$trt_path" ]; then
  echo "TensorRT engine already exists, skipping build: $trt_path"
else
  echo "Building TensorRT engine with trtexec..."
  trtexec \
    --onnx="$onnx_path" \
    --minShapes="$shape_min" \
    --optShapes="$shape_opt" \
    --maxShapes="$shape_max" \
    --saveEngine="$trt_path" \
    $precision_flag
fi

echo "Profiling engine at opt shape..."
trtexec \
  --loadEngine="$trt_path" \
  --shapes="$shape_opt" \
  --warmUp=100 \
  --duration=10 \
  --iterations=100 \
  --dumpProfile \
  --verbose
