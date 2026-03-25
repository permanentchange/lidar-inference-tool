#!/bin/bash

### bash tools/onnx2engine.sh end2end.onnx end2end.engine 10000 40000 50000 [fp32|fp16|int8]
onnx_path=$1
trt_path=$2
min_num_points=$3
opt_num_points=$4
max_num_points=$5
precision=${6:-fp16}

if [ ! -f "$onnx_path" ]; then
  echo "ONNX model file not found!"
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
    echo "Unsupported precision '$precision'. Use fp32, fp16, or int8."
    exit 1
    ;;
esac

if [ -f "$trt_path" ]; then
  echo "TensorRT engine file $trt_path already exists. Skipping. "
else
  trtexec --onnx="$onnx_path"     --minShapes="coors:${min_num_points}x4,voxels:${min_num_points}x20x5,num_points:${min_num_points}"     --optShapes="coors:${opt_num_points}x4,voxels:${opt_num_points}x20x5,num_points:${opt_num_points}"     --maxShapes="coors:${max_num_points}x4,voxels:${max_num_points}x20x5,num_points:${max_num_points}"     --saveEngine="$trt_path" $precision_flag
fi

trtexec --loadEngine="$trt_path"   --shapes="coors:${opt_num_points}x4,voxels:${opt_num_points}x20x5,num_points:${opt_num_points}"   --warmUp=100 --duration=10 --iterations=100 --dumpProfile --verbose
