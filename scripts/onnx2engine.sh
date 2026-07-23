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

configure_runtime_library_path

resolve_python_for_tensorrt() {
  local candidate="${PYTHON_BIN:-python3}"
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" - <<'PY' >/dev/null 2>&1
import tensorrt  # noqa: F401
PY
    then
      PYTHON_BIN="$candidate"
      return 0
    fi
  fi

  local conda_sh="${HOME}/miniconda3/etc/profile.d/conda.sh"
  local conda_env="${ONNX2ENGINE_CONDA_ENV:-openmmlab_py38}"
  if [ -f "$conda_sh" ]; then
    # shellcheck disable=SC1090
    source "$conda_sh"
    conda activate "$conda_env"
    configure_runtime_library_path
    if python - <<'PY' >/dev/null 2>&1
import tensorrt  # noqa: F401
PY
    then
      PYTHON_BIN="python"
      return 0
    fi
  fi

  echo "Unable to import TensorRT Python bindings. Set PYTHON_BIN to a Python" >&2
  echo "environment that can import 'tensorrt', or install the bindings into" >&2
  echo "the current environment." >&2
  exit 1
}

usage() {
  cat <<'EOF_USAGE'
Usage:
  bash tools/onnx2engine.sh <onnx_path> <engine_path> [precision] [--cali-data DIR] [--report-dir DIR]
  bash tools/onnx2engine.sh <onnx_path> <engine_path> <min_points> <opt_points> <max_points> [precision] [--cali-data DIR] [--report-dir DIR]

Arguments:
  <onnx_path>     Input ONNX model path.
  <engine_path>   Output TensorRT engine path.
  <min_points>    Optional min optimization-profile pillar/point count.
                  Default: 10000.
  <opt_points>    Optional opt optimization-profile pillar/point count.
                  Default: 30000.
  <max_points>    Optional max optimization-profile pillar/point count.
                  Default: 50000.
  [precision]     Optional precision mode: fp32, fp16, int8, fp16int8,
                  fp16in8, or best. Default: fp16.
                  fp16in8 is accepted as a backward-compatible alias for
                  fp16int8.
  --cali-data DIR Optional calibration-data directory for TensorRT PTQ.
                  The directory must contain per-sample subdirectories with:
                    pillar_voxels.npy
                    pillar_num_points.npy
                    pillar_coors.npy
  --report-dir DIR
                  Optional report output directory. Default:
                    <engine_path>.reports

Supported calling forms:
  1. ONNX + engine only:
       bash tools/onnx2engine.sh model.onnx model.engine

  2. ONNX + engine + precision:
       bash tools/onnx2engine.sh model.onnx model.engine fp32
       bash tools/onnx2engine.sh model.onnx model.engine fp16
       bash tools/onnx2engine.sh model.onnx model.engine int8
       bash tools/onnx2engine.sh model.onnx model.engine fp16int8
       bash tools/onnx2engine.sh model.onnx model.engine fp16in8
       bash tools/onnx2engine.sh model.onnx model.engine best

  3. ONNX + engine + min/opt/max:
       bash tools/onnx2engine.sh model.onnx model.engine 10000 30000 50000

  4. ONNX + engine + min/opt/max + precision:
       bash tools/onnx2engine.sh model.onnx model.engine 10000 30000 50000 fp16
       bash tools/onnx2engine.sh model.onnx model.engine 10000 30000 50000 fp16int8
       bash tools/onnx2engine.sh model.onnx model.engine 10000 30000 50000 best

  5. PTQ build with calibration data:
       bash tools/onnx2engine.sh model.onnx model.engine best --cali-data cali_data
       bash tools/onnx2engine.sh model.onnx model.engine fp16int8 --cali-data cali_data
       bash tools/onnx2engine.sh model.onnx model.engine 10000 30000 50000 int8 --cali-data cali_data

This helper assumes the exported ONNX/TensorRT contract used by this repository:
  - voxels:     <N>x20x5
  - num_points: <N>
  - coors:      <N>x4

Examples:
  bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end.engine
  bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end_fp32.engine fp32
  bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end_fp16.engine fp16
  bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end_int8.engine int8
  bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end_fp16int8.engine fp16int8 --cali-data demo/cali_data
  bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end_best.engine best
  bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end.engine 10000 30000 50000
  bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end.engine 10000 30000 50000 fp16
  bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end.engine 10000 30000 50000 best
  bash tools/onnx2engine.sh demo/end2end.onnx demo/end2end.engine best --cali-data demo/cali_data

Behavior:
  1. If <engine_path> does not exist, build the TensorRT engine.
  2. If --cali-data is provided, rebuild with TensorRT INT8 calibration and
     overwrite the existing engine.
  3. Load the resulting engine and run a short profile at the opt shape.
  4. Always keep detailed report files under --report-dir or <engine_path>.reports:
       metadata.txt
       build.log or build_skipped.log
       build_layer_info.json, when supported by this trtexec version
       build_profile.json, when supported by this trtexec version
       profile.log
       profile_layer_info.json, when supported by this trtexec version
       profile_profile.json, when supported by this trtexec version
       summary.txt
       trtexec_commands.sh
EOF_USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -lt 2 ]; then
  echo "Invalid argument count: expected at least 2 arguments, got $#." >&2
  usage >&2
  exit 1
fi

onnx_path=$1
trt_path=$2
shift 2

default_min_num_points=20000
default_opt_num_points=30000
default_max_num_points=40000
default_precision=fp16

min_num_points=$default_min_num_points
opt_num_points=$default_opt_num_points
max_num_points=$default_max_num_points
precision=$default_precision
cali_data_dir=""
report_dir=""
extra_args=()

is_precision() {
  case "$1" in
    fp32|fp16|int8|fp16int8|fp16in8|best) return 0 ;;
    *) return 1 ;;
  esac
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --cali-data)
      if [ "$#" -lt 2 ]; then
        echo "--cali-data requires a directory argument." >&2
        exit 1
      fi
      cali_data_dir=$2
      shift 2
      ;;
    --report-dir)
      if [ "$#" -lt 2 ]; then
        echo "--report-dir requires a directory argument." >&2
        exit 1
      fi
      report_dir=$2
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

case "${#extra_args[@]}" in
  0)
    ;;
  1)
    if is_precision "${extra_args[0]}"; then
      precision=${extra_args[0]}
    else
      echo "Invalid precision argument: ${extra_args[0]}" >&2
      usage >&2
      exit 1
    fi
    ;;
  3)
    min_num_points=${extra_args[0]}
    opt_num_points=${extra_args[1]}
    max_num_points=${extra_args[2]}
    ;;
  4)
    min_num_points=${extra_args[0]}
    opt_num_points=${extra_args[1]}
    max_num_points=${extra_args[2]}
    precision=${extra_args[3]}
    if ! is_precision "$precision"; then
      echo "Invalid precision argument: $precision" >&2
      usage >&2
      exit 1
    fi
    ;;
  *)
    echo "Invalid positional argument count after parsing flags: ${#extra_args[@]}." >&2
    usage >&2
    exit 1
    ;;
esac

if [ "$precision" = "fp16in8" ]; then
  precision="fp16int8"
fi

if [ -z "$report_dir" ]; then
  report_dir="${trt_path}.reports"
fi
mkdir -p "$report_dir"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec was not found in PATH." >&2
  echo "Install TensorRT tools or export PATH to include the trtexec binary." >&2
  exit 1
fi

if [ ! -f "$onnx_path" ]; then
  echo "ONNX model file not found: $onnx_path" >&2
  exit 1
fi

if [ -n "$cali_data_dir" ] && [ ! -d "$cali_data_dir" ]; then
  echo "Calibration data directory not found: $cali_data_dir" >&2
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

precision_flags=()
case "$precision" in
  fp32)
    ;;
  fp16)
    precision_flags+=(--fp16)
    ;;
  int8)
    precision_flags+=(--int8)
    ;;
  fp16int8)
    precision_flags+=(--fp16 --int8)
    ;;
  best)
    precision_flags+=(--best)
    ;;
  *)
    echo "Unsupported precision '$precision'. Use fp32, fp16, int8, fp16int8, fp16in8, or best." >&2
    exit 1
    ;;
esac

shape_min="coors:${min_num_points}x4,voxels:${min_num_points}x20x5,num_points:${min_num_points}"
shape_opt="coors:${opt_num_points}x4,voxels:${opt_num_points}x20x5,num_points:${opt_num_points}"
shape_max="coors:${max_num_points}x4,voxels:${max_num_points}x20x5,num_points:${max_num_points}"

trtexec_help=""
trtexec_help=$(trtexec --help 2>&1 || true)
trtexec_supports() {
  local option=$1
  grep -F -- "$option" <<<"$trtexec_help" >/dev/null 2>&1
}

append_common_report_flags() {
  local phase=$1
  local -n cmd_ref=$2

  if trtexec_supports "--profilingVerbosity"; then
    cmd_ref+=(--profilingVerbosity=detailed)
  fi
  if trtexec_supports "--dumpLayerInfo"; then
    cmd_ref+=(--dumpLayerInfo)
  fi
  if trtexec_supports "--exportLayerInfo"; then
    cmd_ref+=(--exportLayerInfo="${report_dir}/${phase}_layer_info.json")
  fi
  if trtexec_supports "--dumpProfile"; then
    cmd_ref+=(--dumpProfile)
  fi
  if trtexec_supports "--exportProfile"; then
    cmd_ref+=(--exportProfile="${report_dir}/${phase}_profile.json")
  fi
}

write_command() {
  local label=$1
  shift
  {
    echo "# ${label}"
    printf '%q ' "$@"
    echo
    echo
  } >>"${report_dir}/trtexec_commands.sh"
}

run_and_log() {
  local log_path=$1
  shift
  "$@" 2>&1 | tee "$log_path"
}

write_metadata() {
  {
    echo "ONNX input:      $onnx_path"
    echo "Engine output:   $trt_path"
    echo "Precision mode:  $precision"
    echo "Min TRT shape:   $shape_min"
    echo "Opt TRT shape:   $shape_opt"
    echo "Max TRT shape:   $shape_max"
    echo "Report dir:      $report_dir"
    echo "trtexec path:    $(command -v trtexec)"
    echo "Generated at:    $(date -Iseconds)"
    if [ -n "$cali_data_dir" ]; then
      echo "Calibration dir: $cali_data_dir"
    fi
    echo
    echo "Report flag support detected from trtexec --help:"
    for option in --profilingVerbosity --dumpLayerInfo --exportLayerInfo --dumpProfile --exportProfile; do
      if trtexec_supports "$option"; then
        echo "  ${option}: yes"
      else
        echo "  ${option}: no"
      fi
    done
  } >"${report_dir}/metadata.txt"
}

summarize_reports() {
  {
    echo "Report directory: $report_dir"
    echo "Engine: $trt_path"
    echo "Precision: $precision"
    echo
    echo "Generated files:"
    find "$report_dir" -maxdepth 1 -type f -printf '  %f\n' | sort
    echo
    echo "Selected profile/build highlights:"
    for log_file in "${report_dir}/build.log" "${report_dir}/profile.log"; do
      if [ -f "$log_file" ]; then
        echo
        echo "== $(basename "$log_file") =="
        grep -E "GPU latency|Host latency|Throughput|Latency|Enqueue Time|H2D Latency|D2H Latency|Total Host Walltime|Total GPU Compute Time|Average on|ERROR|Error|FAILED|failed|WARNING|Warning" "$log_file" || true
      fi
    done
  } >"${report_dir}/summary.txt"
}

write_metadata
: >"${report_dir}/trtexec_commands.sh"
chmod +x "${report_dir}/trtexec_commands.sh"

cat "${report_dir}/metadata.txt"

build_with_calibration() {
  if [ "$precision" != "int8" ] && [ "$precision" != "fp16int8" ] && [ "$precision" != "best" ]; then
    echo "--cali-data requires precision int8, fp16int8, or best, got: $precision" >&2
    exit 1
  fi

  resolve_python_for_tensorrt

  local calib_cache_path="${trt_path}.calib.cache"
  rm -f "$trt_path" "$calib_cache_path"

  "$PYTHON_BIN" - "$onnx_path" "$trt_path" "$cali_data_dir" "$min_num_points" "$opt_num_points" "$max_num_points" "$precision" "$calib_cache_path" <<'PY'
import os
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

onnx_path = sys.argv[1]
engine_path = sys.argv[2]
cali_dir = Path(sys.argv[3])
min_points = int(sys.argv[4])
opt_points = int(sys.argv[5])
max_points = int(sys.argv[6])
precision = sys.argv[7]
cache_path = sys.argv[8]

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def set_workspace(config, bytes_size):
    if hasattr(config, 'set_memory_pool_limit') and hasattr(trt, 'MemoryPoolType'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, bytes_size)
    elif hasattr(config, 'max_workspace_size'):
        config.max_workspace_size = bytes_size


def list_sample_dirs(root: Path):
    sample_dirs = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / 'pillar_voxels.npy').is_file() and \
           (child / 'pillar_num_points.npy').is_file() and \
           (child / 'pillar_coors.npy').is_file():
            sample_dirs.append(child)
    if not sample_dirs and (root / 'pillar_voxels.npy').is_file():
        sample_dirs.append(root)
    if not sample_dirs:
        raise RuntimeError(
            f'No calibration samples found under {root}. Expected per-sample '
            'directories containing pillar_voxels.npy, pillar_num_points.npy, '
            'and pillar_coors.npy.')
    return sample_dirs


class PillarEntropyCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, sample_dirs, opt_points, cache_path):
        super().__init__()
        self.sample_dirs = sample_dirs
        self.opt_points = opt_points
        self.cache_path = cache_path
        self.index = 0
        self.buffers = {
            'voxels': torch.zeros((opt_points, 20, 5), dtype=torch.float32, device='cuda'),
            'num_points': torch.zeros((opt_points,), dtype=torch.int32, device='cuda'),
            'coors': torch.zeros((opt_points, 4), dtype=torch.int32, device='cuda'),
        }

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.index >= len(self.sample_dirs):
            return None

        sample_dir = self.sample_dirs[self.index]
        self.index += 1

        voxels = np.load(sample_dir / 'pillar_voxels.npy')
        num_points = np.load(sample_dir / 'pillar_num_points.npy')
        coors = np.load(sample_dir / 'pillar_coors.npy')

        if voxels.ndim != 3 or voxels.shape[1:] != (20, 5):
            raise RuntimeError(
                f'Unexpected pillar_voxels.npy shape {voxels.shape} in {sample_dir}')
        if num_points.ndim != 1:
            raise RuntimeError(
                f'Unexpected pillar_num_points.npy shape {num_points.shape} in {sample_dir}')
        if coors.ndim != 2 or coors.shape[1] != 4:
            raise RuntimeError(
                f'Unexpected pillar_coors.npy shape {coors.shape} in {sample_dir}')

        count = min(int(voxels.shape[0]), self.opt_points)
        self.buffers['voxels'].zero_()
        self.buffers['num_points'].zero_()
        self.buffers['coors'].zero_()

        if count > 0:
            self.buffers['voxels'][:count].copy_(torch.from_numpy(
                np.ascontiguousarray(
                    voxels[:count].astype(np.float32, copy=False))).to('cuda'))
            self.buffers['num_points'][:count].copy_(torch.from_numpy(
                np.ascontiguousarray(
                    num_points[:count].astype(np.int32, copy=False))).to('cuda'))
            self.buffers['coors'][:count].copy_(torch.from_numpy(
                np.ascontiguousarray(
                    coors[:count].astype(np.int32, copy=False))).to('cuda'))

        ptrs = {
            name: int(tensor.data_ptr())
            for name, tensor in self.buffers.items()
        }
        return [ptrs[name] for name in names]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_path, 'wb') as f:
            f.write(cache)


sample_dirs = list_sample_dirs(cali_dir)
print(f'[INFO] Found {len(sample_dirs)} calibration samples under {cali_dir}')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required for TensorRT PTQ calibration')

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)
with open(onnx_path, 'rb') as f:
    if not parser.parse(f.read()):
        errors = '\n'.join(
            str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f'Failed to parse ONNX:\n{errors}')

config = builder.create_builder_config()
set_workspace(config, 4 << 30)
profile = builder.create_optimization_profile()
profile.set_shape('voxels', (min_points, 20, 5), (opt_points, 20, 5),
                  (max_points, 20, 5))
profile.set_shape('num_points', (min_points,), (opt_points,), (max_points,))
profile.set_shape('coors', (min_points, 4), (opt_points, 4),
                  (max_points, 4))
config.add_optimization_profile(profile)
config.set_flag(trt.BuilderFlag.INT8)
if precision in ('best', 'fp16int8'):
    config.set_flag(trt.BuilderFlag.FP16)
if hasattr(config, 'set_calibration_profile'):
    config.set_calibration_profile(profile)
config.int8_calibrator = PillarEntropyCalibrator(sample_dirs, opt_points,
                                                 cache_path)

serialized = builder.build_serialized_network(network, config)
if serialized is None:
    raise RuntimeError('TensorRT failed to build the calibrated engine')

engine_path = Path(engine_path)
engine_path.parent.mkdir(parents=True, exist_ok=True)
with open(engine_path, 'wb') as f:
    f.write(serialized)
print(f'[INFO] Saved calibrated TensorRT engine to {engine_path}')
PY
}

if [ -n "$cali_data_dir" ]; then
  echo "Building TensorRT engine with TensorRT PTQ calibration..."
  run_and_log "${report_dir}/build.log" build_with_calibration
elif [ -f "$trt_path" ]; then
  echo "TensorRT engine already exists, skipping build: $trt_path"
  {
    echo "TensorRT engine already exists, skipping build: $trt_path"
    echo "Generated at: $(date -Iseconds)"
  } | tee "${report_dir}/build_skipped.log"
else
  echo "Building TensorRT engine with trtexec..."
  build_cmd=(
    trtexec
    --onnx="$onnx_path"
    --minShapes="$shape_min"
    --optShapes="$shape_opt"
    --maxShapes="$shape_max"
    --saveEngine="$trt_path"
  )
  build_cmd+=("${precision_flags[@]}")
  append_common_report_flags build build_cmd
  write_command build "${build_cmd[@]}"
  run_and_log "${report_dir}/build.log" "${build_cmd[@]}"
fi

if [ ! -f "$trt_path" ]; then
  echo "TensorRT engine was not produced: $trt_path" >&2
  exit 1
fi

echo "Profiling engine at opt shape..."
profile_cmd=(
  trtexec
  --loadEngine="$trt_path"
  --shapes="$shape_opt"
  --warmUp=100
  --duration=10
  --iterations=100
)
append_common_report_flags profile profile_cmd
write_command profile "${profile_cmd[@]}"
run_and_log "${report_dir}/profile.log" "${profile_cmd[@]}"

summarize_reports

echo "Detailed reports saved under: $report_dir"
echo "Summary report: ${report_dir}/summary.txt"
