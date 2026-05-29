#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX:-/usr/local}"
JOBS="${JOBS:-$(nproc)}"

rm -rf "${BUILD_DIR}"

cmake \
  -S "${SCRIPT_DIR}" \
  -B "${BUILD_DIR}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  "$@"

cmake --build "${BUILD_DIR}" -j "${JOBS}"

rm -rf "${BUILD_DIR}/_CPack_Packages"

cd "${BUILD_DIR}"
cpack --config CPackConfig.cmake -G DEB
