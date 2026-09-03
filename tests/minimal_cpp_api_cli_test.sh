#!/usr/bin/env bash
set -euo pipefail

tool="$1"
output="$("$tool" 2>&1)" && {
  echo "Expected usage failure with no arguments" >&2
  exit 1
}

if ! grep -Fq -- "Usage: ./build/minimal_cpp_api_example MODEL_DIR POINT_CLOUD_BIN" <<<"$output"; then
  echo "Expected build executable usage missing" >&2
  echo "$output" >&2
  exit 1
fi
if ! grep -Fq -- "./build/minimal_cpp_api_example demo/centerpoint_pillar demo/data/test.bin" <<<"$output"; then
  echo "Expected pillar demo command missing" >&2
  echo "$output" >&2
  exit 1
fi
if grep -Fq -- "examples/minimal_cpp_api" <<<"$output"; then
  echo "Source directory path should not be presented as an executable command" >&2
  echo "$output" >&2
  exit 1
fi
