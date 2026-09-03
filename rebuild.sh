#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${BUILD_DIR:-$DIR/build}"

case "$(uname -m)" in
  x86_64) ARCH=x86_64 ;;
  aarch64|arm64) ARCH=aarch64 ;;
  *) echo "ERROR: unsupported architecture: $(uname -m)"; exit 1 ;;
esac

CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
NVCC="$CUDA_ROOT/bin/nvcc"

[ -x "$NVCC" ] || { echo "ERROR: nvcc not found: $NVCC"; exit 1; }

CUDA="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
[ -n "$CUDA" ] || { echo "ERROR: failed to detect CUDA version"; exit 1; }

SPCONV="$DIR/third_party/libspconv/lib"
SRC="$SPCONV/${ARCH}_cuda${CUDA}/libspconv.so"
DST="$SPCONV/$ARCH/libspconv.so"

[ -f "$SRC" ] || { echo "ERROR: libspconv not found: $SRC"; exit 1; }

mkdir -p "$(dirname "$DST")"
ln -sfn "../${ARCH}_cuda${CUDA}/libspconv.so" "$DST"

echo "ARCH   : $ARCH"
echo "CUDA   : $CUDA"
echo "CUDA_DIR: $CUDA_ROOT"
echo "SPCONV : ${ARCH}_cuda${CUDA}/libspconv.so"
[ -n "${TENSORRT_ROOT:-}" ] && echo "TensorRT: $TENSORRT_ROOT"

rm -rf "$BUILD"

if [ -n "${TENSORRT_ROOT:-}" ]; then
  cmake -S "$DIR" -B "$BUILD" \
    -DCMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX:-/usr/local}" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" \
    -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DTENSORRT_INCLUDE_DIR="$TENSORRT_ROOT/include" \
    -DNVINFER_LIB="$TENSORRT_ROOT/targets/$ARCH-linux-gnu/lib/libnvinfer.so" \
    -DNVINFER_PLUGIN_LIB="$TENSORRT_ROOT/targets/$ARCH-linux-gnu/lib/libnvinfer_plugin.so" \
    "$@"
else
  cmake -S "$DIR" -B "$BUILD" \
    -DCMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX:-/usr/local}" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" \
    -DCMAKE_CUDA_COMPILER="$NVCC" \
    "$@"
fi

cmake --build "$BUILD" -j "${JOBS:-$(nproc)}"

cd "$BUILD"
cpack -G DEB