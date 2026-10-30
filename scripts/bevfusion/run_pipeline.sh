#!/usr/bin/env bash
# Default pipeline: local ONNX export -> TensorRT engines -> numerical checks.
set -euo pipefail

usage() {
  echo "Usage: bash scripts/bevfusion/run_pipeline.sh --config EXPERIMENT.yaml [--overwrite]"
  echo "Environment: PYTHON_BIN (default: python), TENSORRT_ROOT, CUDA_VISIBLE_DEVICES"
  echo "Full dataset evaluation is separate: python scripts/bevfusion/evaluate.py --model-dir MODEL_DIR"
}

experiment=""
overwrite=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --config)
      if [ "$#" -lt 2 ] || [[ "$2" == --* ]] || [ -z "$2" ]; then
        echo "Missing path for --config" >&2; exit 1
      fi
      experiment=$2; shift 2 ;;
    --overwrite) overwrite=(--overwrite); shift ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done
if [ -z "$experiment" ]; then echo "--config is required" >&2; usage >&2; exit 1; fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
arguments_file="$(mktemp)"
trap 'rm -f "$arguments_file"' EXIT
# No eval or sourcing generated shell code: each value is an array element.
PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -m _lib.pipeline_config \
  --config "$experiment" --output "$arguments_file"
mapfile -d '' -t settings < "$arguments_file"
if [ "${#settings[@]}" -ne 9 ]; then echo "Invalid normalized experiment settings" >&2; exit 1; fi
model_dir=${settings[3]}
sample_args=()
if [ "${settings[7]}" != "$model_dir/sample" ]; then sample_args=(--sample-dir "${settings[7]}"); fi
mkdir -p "$model_dir/logs"

run_step() {
  local step=$1
  shift
  echo "==> $step"
  "$python_bin" "$script_dir/$step.py" "$@" 2>&1 | tee "$model_dir/logs/$step.log"
}

run_step export --mmdet3d-root "${settings[0]}" --config "${settings[1]}" \
  --checkpoint "${settings[2]}" --output-dir "$model_dir" --device "${settings[4]}" \
  --deployment-config "${settings[8]}" --sample-index "${settings[5]}" "${sample_args[@]}" "${overwrite[@]}"
run_step build_engines --model-dir "$model_dir" --device "${settings[4]}" "${overwrite[@]}"
run_step validate --model-dir "$model_dir" --device "${settings[4]}" \
  --frames "${settings[6]}" "${overwrite[@]}"
echo "Pipeline complete: $model_dir"
