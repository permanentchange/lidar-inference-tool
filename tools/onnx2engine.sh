#!/bin/bash


### bash onnx2trt.sh end2end.onnx end2end.trt 10000 40000 50000
onnx_path=$1
trt_path=$2


min_num_points=$3
opt_num_points=$4
max_num_points=$5

if [ ! -f "$onnx_path" ]; then
  echo "ONNX model file not found!"
  exit 1
fi

if [ -f "$trt_path" ]; then
  echo "TensorRT engine file $trt_path already exists. Skipping. "
else
  trtexec --onnx=$onnx_path \
    --minShapes=coors:${min_num_points}x4,voxels:${min_num_points}x20x5,num_points:${min_num_points} \
    --optShapes=coors:${opt_num_points}x4,voxels:${opt_num_points}x20x5,num_points:${opt_num_points} \
    --maxShapes=coors:${max_num_points}x4,voxels:${max_num_points}x20x5,num_points:${max_num_points} \
    --saveEngine=$trt_path --fp16
fi

trtexec --loadEngine=$trt_path --shapes=coors:${opt_num_points}x4,voxels:${opt_num_points}x20x5,num_points:${opt_num_points} --warmUp=100 --duration=10 --iterations=100 --dumpProfile --verbose

