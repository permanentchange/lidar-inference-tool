# BEVFusion deployment

The only supported architecture is the current CenterHead BEVFusion trained with
`projects/BEVFusion/configs/exp20261028/lidar_cam.py`. Runtime backend is
`bevfusion`, architecture is `centerpoint_bevfusion`; old TransFusion packages
are rejected. See [model details](MODEL.md) and [source demo frames](../../demo/data/bevfusion/README.md).

## Environment and independent steps

```bash
conda activate openmmlab_py38
export CUDA_VISIBLE_DEVICES=5
export TENSORRT_ROOT=/home/zhouxiufeng/TensorRT-8.6.1.6
export CUDA_HOME=/usr/local/cuda-11.7
export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib:$CUDA_HOME/lib64:$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
```

Existing model engines can be used directly. To create a new package:

```bash
python scripts/bevfusion/export.py --mmdet3d-root ../mmdetection3d \
  --config projects/BEVFusion/configs/exp20261028/lidar_cam.py \
  --checkpoint work_dirs/exp20261028_lidar_cam_200e/epoch_200.pth \
  --output-dir outputs/new_bevfusion --sample-dir demo/data/bevfusion/export_sample \
  --deployment-config /home/zhouxiufeng/lidar_perception_config/model.yaml
python scripts/bevfusion/build_engines.py --model-dir outputs/new_bevfusion
python scripts/bevfusion/validate.py --model-dir outputs/new_bevfusion --frames 20
```

Export, native TensorRT build, numerical validation, pooling benchmark and full
metric evaluation remain independent commands. `--overwrite` is required when
replacing a stage's existing output. The configured pipeline runs export/build/
validate through `bash scripts/bevfusion/run_pipeline.sh --config EXPERIMENT.yaml`;
`sample_dir` optionally separates demo inputs from the model directory.
`deployment_config` selects the shared CenterPoint deployment profile; both export
flows use one generator. See [shared configuration and commands](../../docs/centerhead_deployment.md).

The exporter writes three graphs: `lidar.backbone.zyx.onnx`,
`camera.backbone.onnx`, and `fusion_centerhead.onnx`. Native TensorRT builds only
the latter two; libspconv loads the sparse graph directly. FP16 is enabled, with
FP32 final heatmap convolutions and output bindings to prevent logit overflow.
The builder verifies layer precision inspection before replacing an engine.

## Inference, timing and outputs

```bash
./build/infer_debug_tool --model-dir demo/bevfusion_exp20261028 \
  --input demo/data/bevfusion/manifests/1776582550193446516.yaml \
  --dump-dir outputs/bevfusion --no-dump-internal --no-dump-decoded
./build/perf_debug_tool --model-dir demo/bevfusion_exp20261028 \
  --input demo/data/bevfusion/manifests/1776582550193446516.yaml \
  --dump-dir outputs/bevfusion_perf
python scripts/bevfusion/benchmark_pooling.py --model-dir demo/bevfusion_exp20261028
python scripts/bevfusion/evaluate.py --model-dir demo/bevfusion_exp20261028
```

The native tools share execution and [versioned report contracts](../../README.md#debug-tools-and-reports).
Perf defaults to reports only and separately profiles normal total latency and
instrumented stages. Both passes use the requested warmup and sample counts.
Loading/JPEG decode, initialization and debug writes are outside inference timing.
Infer directory inputs are flat manifest directories; output frames use the same
`frames/<index>_<stem>` layout as LiDAR models. Results include geometric-center
boxes and velocities, and internal tensors are opt-in for perf.

Manifest paths are relative to the manifest. Raw mode loads original PCD/JPG;
preprocessed mode uses the recorded NPY image cache. Camera slots retain training
order and the per-camera `enabled` mask. Disabled cameras require no usable image
or calibration, even though the source demo retains all five original JPEGs.
Crop-first preprocessing records `source_crop`, `resize_size`, and augmentation
matrices; the CUDA path reproduces Pillow uint8 bicubic rounding.

## Validation and retention

Validation checks raw/preprocessed parity, finite tensors, independent voxel and
pooling operators, independent NumPy/OpenCV deployment decode/NMS, same-engine Python/C++ outputs,
and selected dataset frames including camera mask policies. Reference voxel caps and
postprocessing follow the local deployment YAML, including global NMS and final
five-class mapping. Native head-policy outputs are kept separately. Full metric
evaluation explicitly retains the original nine-class label space through an identity
output mapping; those metrics are not five-class business metrics. Reports retain actual
candidate differences; full mAP evaluation is separate. Successful validation
keeps `validation/report.json` by default; `--keep-intermediates` retains fixtures
and tensor dumps for diagnosis. Failed checks retain available diagnostics.

`provenance.json` records source/config/checkpoint hashes and export/build/validate/
evaluate stages. Demo inputs are recorded separately as `input_artifacts` and
resolved via `sample.manifest`. A verified data-only relocation records migration
identifiers and preserves build timestamps; it never rebuilds or changes engines.
Historical validation and pooling summaries live in `outputs/reports/`.

Pooling supports `reference` and `nvidia_pack10` (80-channel FP16). The benchmark
requires tensor/detection equality and improved pooling latency without total-p50
regression before selecting tiled pooling. Its selected YAML override is stored
beside `pooling_benchmark/report.json`.

NVIDIA-derived pooling code retains [its license](../../src/bevfusion/NVIDIA-BEVPOOL-LICENSE).
