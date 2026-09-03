#!/usr/bin/env bash
set -euo pipefail

repo_root=$1
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

mkdir -p "$tmp_dir/TensorRT/bin" "$tmp_dir/TensorRT/lib" "$tmp_dir/tools" "$tmp_dir/model"
touch "$tmp_dir/TensorRT/lib/libnvinfer_plugin.so.8"
touch "$tmp_dir/model/head.onnx"

cat >"$tmp_dir/TensorRT/bin/trtexec" <<'EOF_TRTEXEC'
#!/usr/bin/env bash
echo "trtexec should not be reached" >&2
exit 99
EOF_TRTEXEC
chmod +x "$tmp_dir/TensorRT/bin/trtexec"

cat >"$tmp_dir/tools/ldd" <<'EOF_LDD'
#!/usr/bin/env bash
cat <<'EOF_LDD_OUTPUT'
linux-vdso.so.1 (0x00007ffc00000000)
libcudnn.so.8 => not found
EOF_LDD_OUTPUT
EOF_LDD
chmod +x "$tmp_dir/tools/ldd"

output=""
status=0
output="$(
  PATH="$tmp_dir/TensorRT/bin:$tmp_dir/tools:$PATH" \
    bash "$repo_root/scripts/centerpoint_voxel_head2engine.sh" \
      "$tmp_dir/model/head.onnx" "$tmp_dir/model/head.engine" 2>&1
)" || status=$?

if [ "$status" -eq 0 ]; then
  echo "Expected dependency check failure" >&2
  exit 1
fi
if ! grep -Fq -- "Missing TensorRT runtime dependency: libcudnn.so.8" <<<"$output"; then
  echo "Expected missing cuDNN dependency message" >&2
  echo "$output" >&2
  exit 1
fi
if ! grep -Fq -- "Install cuDNN 8" <<<"$output"; then
  echo "Expected cuDNN install or LD_LIBRARY_PATH guidance" >&2
  echo "$output" >&2
  exit 1
fi
if grep -Fq -- "trtexec should not be reached" <<<"$output"; then
  echo "Dependency check should fail before invoking trtexec" >&2
  echo "$output" >&2
  exit 1
fi

rm -rf "$tmp_dir"
tmp_dir=$(mktemp -d)
mkdir -p "$tmp_dir/TensorRT/bin" "$tmp_dir/TensorRT/lib" "$tmp_dir/tools" "$tmp_dir/model"
touch "$tmp_dir/TensorRT/lib/libnvinfer_plugin.so.8"
touch "$tmp_dir/model/head.onnx"

cat >"$tmp_dir/TensorRT/bin/trtexec" <<'EOF_TRTEXEC_FAIL'
#!/usr/bin/env bash
echo "Cuda failure: no CUDA-capable device is detected" >&2
exit 134
EOF_TRTEXEC_FAIL
chmod +x "$tmp_dir/TensorRT/bin/trtexec"

cat >"$tmp_dir/tools/ldd" <<'EOF_LDD_OK'
#!/usr/bin/env bash
cat <<'EOF_LDD_OK_OUTPUT'
linux-vdso.so.1 (0x00007ffc00000000)
libcudnn.so.8 => /tmp/libcudnn.so.8 (0x00007f0000000000)
EOF_LDD_OK_OUTPUT
EOF_LDD_OK
chmod +x "$tmp_dir/tools/ldd"

output=""
status=0
output="$(
  PATH="$tmp_dir/TensorRT/bin:$tmp_dir/tools:$PATH" \
    bash "$repo_root/scripts/centerpoint_voxel_head2engine.sh" \
      "$tmp_dir/model/head.onnx" "$tmp_dir/model/head.engine" 2>&1
)" || status=$?

if [ "$status" -eq 0 ]; then
  echo "Expected trtexec failure to propagate" >&2
  exit 1
fi
if ! grep -Fq -- "trtexec failed while building CenterPoint voxel head engine." <<<"$output"; then
  echo "Expected trtexec failure summary" >&2
  echo "$output" >&2
  exit 1
fi
if ! grep -Fq -- "Cuda failure: no CUDA-capable device is detected" <<<"$output"; then
  echo "Expected build log tail to be printed" >&2
  echo "$output" >&2
  exit 1
fi
