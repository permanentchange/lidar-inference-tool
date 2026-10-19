#!/usr/bin/env bash
set -euo pipefail

tool="$1"

run_expect_failure() {
  local expected="$1"
  shift
  local output
  local status=0
  output="$("$tool" "$@" 2>&1)" || status=$?
  if [ "$status" -eq 0 ]; then
    echo "Expected failure for: $*" >&2
    exit 1
  fi
  if ! grep -Fq -- "$expected" <<<"$output"; then
    echo "Expected output to contain: $expected" >&2
    echo "Actual output:" >&2
    echo "$output" >&2
    exit 1
  fi
}

help_output="$("$tool" --help 2>&1)"
named_usage="Usage: $tool --model-dir MODEL_DIR --input INPUT_PATH --dump-dir DUMP_DIR [--source-num-point-features N] [--override-config PATH] [--dump-decoded] [--dump-internal] [--warmup N] [--profile-iters N]"
if ! grep -Fq -- "$named_usage" <<<"$help_output"; then
  echo "Named-argument usage missing from --help output" >&2
  echo "$help_output" >&2
  exit 1
fi
if grep -Fq -- "<model_dir> <input_path>" <<<"$help_output"; then
  echo "Positional usage unexpectedly present in --help output" >&2
  echo "$help_output" >&2
  exit 1
fi

run_expect_failure "Unexpected positional argument: demo/centerpoint_pillar" \
  demo/centerpoint_pillar demo/data/test1.bin
run_expect_failure "--model-dir is required" \
  --input demo/data/test1.bin --dump-dir /tmp/infer_debug_cli_test
run_expect_failure "--input is required" \
  --model-dir demo/centerpoint_pillar --dump-dir /tmp/infer_debug_cli_test
run_expect_failure "--dump-dir is required" \
  --model-dir demo/centerpoint_pillar --input demo/data/test1.bin
run_expect_failure "must be one of 3, 4, or 5" \
  --model-dir demo/centerpoint_pillar --input demo/data/test1.bin \
  --dump-dir /tmp/infer_debug_cli_test --source-num-point-features 6
