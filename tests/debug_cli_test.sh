#!/usr/bin/env bash
set -euo pipefail

tool="$1"
kind="$2"

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
named_usage="Usage: $tool --model-dir MODEL_DIR --input INPUT_PATH --dump-dir DUMP_DIR [--source-num-point-features N] [--override-config PATH] [--warmup N] [--profile-iters N]"
if ! grep -Fq -- "$named_usage" <<<"$help_output"; then
  echo "Named-argument usage missing from --help output" >&2
  echo "$help_output" >&2
  exit 1
fi
if grep -Fq -- "<model_dir>" <<<"$help_output"; then
  echo "Positional usage unexpectedly present in --help output" >&2
  echo "$help_output" >&2
  exit 1
fi
if [ "$("$tool" -h 2>&1)" != "$help_output" ]; then
  echo "-h and --help must print the same help" >&2
  exit 1
fi

required=(--model-dir unused-model --input unused-input --dump-dir /tmp/debug_cli_test)
run_expect_failure "Usage:"

run_expect_failure "Unexpected positional argument: demo/centerpoint_pillar" \
  demo/centerpoint_pillar demo/data/lidar/test1.bin
run_expect_failure "--model-dir is required" \
  --input demo/data/lidar/test1.bin --dump-dir /tmp/infer_debug_cli_test
run_expect_failure "--input is required" \
  --model-dir demo/centerpoint_pillar --dump-dir /tmp/infer_debug_cli_test
run_expect_failure "--dump-dir is required" \
  --model-dir demo/centerpoint_pillar --input demo/data/lidar/test1.bin
run_expect_failure "use --profile-iters N instead" "${required[@]}" --profile 2
run_expect_failure "Unknown argument: --unknown" "${required[@]}" --unknown

run_expect_failure "--image-input must be raw or preprocessed" "${required[@]}" --image-input invalid

for option in --model-dir --input --dump-dir --override-config --image-input; do
  run_expect_failure "Missing path for $option" "${required[@]}" "$option"
  run_expect_failure "Missing path for $option" "${required[@]}" "$option" --warmup 0
  run_expect_failure "Missing path for $option" "${required[@]}" "$option" ""
done
for option in --warmup --profile-iters --source-num-point-features; do
  run_expect_failure "Missing count for $option" "${required[@]}" "$option"
  run_expect_failure "Missing count for $option" "${required[@]}" "$option" --override-config ignored
  for value in abc 1x 1.5 99999999999999999999 -99999999999999999999 ""; do
    run_expect_failure "Invalid integer for $option" "${required[@]}" "$option" "$value"
  done
done
run_expect_failure "--warmup must be >= 0" "${required[@]}" --warmup -1
for value in 0 -1; do
  run_expect_failure "--profile-iters must be > 0" "${required[@]}" --profile-iters "$value"
done
for value in 0 2 6 -1; do
  run_expect_failure "must be one of 3, 4, or 5" "${required[@]}" --source-num-point-features "$value"
done

if [ "$kind" = infer ]; then
  grep -Fq -- "Warmup iterations >= 0 (default: 0)" <<<"$help_output"
  grep -Fq -- "Profile iterations > 0 (default: 1)" <<<"$help_output"
else
  grep -Fq -- "Warmup iterations >= 0 (default: 10)" <<<"$help_output"
  grep -Fq -- "Profile iterations > 0 (default: 50)" <<<"$help_output"
fi
grep -Fq -- "--dump-result / --no-dump-result" <<<"$help_output"
grep -Fq -- "separate passes" <<<"$help_output"
