#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <build-dir> <target-processor>" >&2
  exit 2
fi

build_dir="$1"
target_processor="$2"
staging_dir="$(mktemp -d "${TMPDIR:-/tmp}/lidar-package-arch.XXXXXX")"
trap 'rm -rf -- "${staging_dir}"' EXIT

DESTDIR="${staging_dir}" cmake --install "${build_dir}" >/dev/null

mapfile -t downloaders < <(find "${staging_dir}" -type f -name lidar_model_downloader -print)

case "${target_processor}" in
  aarch64|arm64)
    if [[ ${#downloaders[@]} -ne 1 ]]; then
      echo "expected one ARM64 lidar_model_downloader, found ${#downloaders[@]}" >&2
      exit 1
    fi
    if ! file "${downloaders[0]}" | grep -q 'ARM aarch64'; then
      echo "installed lidar_model_downloader is not an ARM64 executable" >&2
      file "${downloaders[0]}" >&2
      exit 1
    fi
    ;;
  *)
    if [[ ${#downloaders[@]} -ne 0 ]]; then
      echo "non-ARM64 package contains an unusable ARM64 lidar_model_downloader" >&2
      file "${downloaders[@]}" >&2
      exit 1
    fi
    ;;
esac
