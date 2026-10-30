# CenterPoint / BEVFusion 共用部署配置

本文对应 `demo/centerpoint_voxel` 与 `demo/bevfusion_exp20261028`。两者采用
相同的 CenterHead 结构和类别定义，各自使用自己的模型权重及网络文件。
公共配置以 `/home/zhouxiufeng/lidar_perception_config/model.yaml` 为基准。

仓库内的 [默认配置](../scripts/configs/centerpoint_deployment.yaml) 是该基准的
快照。命令不指定 `--deployment-config` 时读取仓库快照，**不会自动读取或跟踪
外部文件的修改**。同步和新导出时显式指定外部路径，才能采用外部基准的新值。
运行时读取各模型目录内的 `model.yaml`，不读取外部基准或配置快照。

## 1. 共用内容与保留差异

两个后端共用后处理配置解析及校验、GPU 解码器、候选排序、分组 NMS 和最终
标签映射。模型加载、体素化实现、相机计算和公共 C++ 推理入口仍各有对应实现；
相同配置不表示两者的整个推理过程或输出数值相同。

| 内容 | 当前公共配置 |
|---|---|
| 版本、解码与朝向 | `v8.0`、`centerpoint_mmdeploy`、`sin_cos` |
| 点云范围 | `[-8, -8, -1, 16, 8, 3]` |
| 解码后中心范围 | `[-10, -10, -1, 18, 10, 3]` |
| 体素大小、稀疏网格 | `[0.05, 0.05, 0.1]`、`[41, 320, 480]` |
| 体素及点数上限 | 每体素 `16` 点、最多 `100000` 个体素、最多 `300000` 个输入点 |
| 解码分数与 TopK | `runtime.score_threshold: 0.1`；每个 task 最多 `500` 个候选 |
| 类别分数阈值 | `[0.52, 0.63, 0.55, 0.60, 0.50, 0.10, 0.47, 0.40, 0.60]` |
| NMS 类型 | `iou_min_bev` |
| 显式 NMS 分组 | 原始标签 `0..8` 共一组，阈值 `0.3`，预选上限 `500`，保留上限 `50` |
| 最终分数阈值 | `0.3` |
| 自车框及阈值 | `[0.7, 0, 0, 1.8, 1.4, 0.2, 0]`、`0.01` |
| 输出标签映射 | `[0, 1, 0, 0, 3, 4, 0, 2, 2]` |
| 映射后名称 | `bi / ped / tri / car / sight`，编号为 `0..4` |

表中数值是当前基准的内容；之后修改基准时，以同步后的模型 YAML 为准。
每体素 `16` 点是指定的部署选择，当前对应训练配置记录的上限为 `10`。
这会影响体素特征和预测，不能视为纯格式调整。

保留的模型差异包括 `runtime_backend`、稀疏 ONNX 和 Head engine 路径。
BEVFusion 额外保留 `architecture`、相机 engine、相机参数、精度约束及
`runtime.bev_pool_backend`。`file_list` 按各模型运行所需文件生成，因此不要求
两份 YAML 完全相同，也不代表目录内所有导出文件的清单。

## 2. 解码、NMS 与标签含义

模型的九个原始类别由 `model.label_names` 定义；六个 task 的原始类别编号为
`[[0,2], [1], [3,6], [4], [5], [7,8]]`。`task_label_ids` 将 Head 局部通道映射到
这九类，`postprocess.label_mapping` 再将最终检测映射到业务五类，两者用途不同。

当前正常 GPU 推理路径按以下顺序处理：

1. 解码框、分数和朝向，按解码分数与中心范围过滤，再对每个 task 取 TopK。
2. 使用原始九类编号执行 `label_score_thresh` 过滤。
3. 汇总同一 NMS 组内各 task 的候选，按分数降序排列，截取组内预选上限。
4. 插入自车框进行过滤，并对检测框执行组内 NMS；自车框不返回、不占检测保留名额。
5. 对 NMS 结果执行最终分数过滤，再映射为业务五类。

`iou_min_bev` 使用旋转 BEV 框的交集面积除以两框中较小的面积；`iou_bev`
则除以并集面积。重叠值**严格大于**对应阈值时抑制。
`nms_label_groups` 内的参数决定当前显式分组的 NMS；因此表中的 `0.3/500/50`
生效，不能把外层的 `0.2/1000/83` 当作本组的实际参数。

最终 `result.json` 和 `final_labels.npy` 使用业务五类；启用解码导出后，
`decoded_labels.npy` 保留原始九类。NMS 保留候选索引，用于关联 BEVFusion 的
速度；当前 BEVFusion 结果包含 `[vx, vy]`，LiDAR C++ 结果接口不输出速度字段。
两者框坐标均为 LiDAR 坐标系下的几何中心 `[x,y,z,dx,dy,dz,yaw]`。

## 3. 同步现有配置与新导出

同步脚本只更新配置和来源记录，不导出 ONNX、不构建 engine。它将覆盖公共模型
字段、公共运行参数和整个 `postprocess`，因此模型目录中的手工后处理修改也会被
基准替换。此命令适用于已经确认与基准匹配的两套模型；同步入口本身不会重新
加载 checkpoint 核验网络几何，新导出入口才执行几何和类别匹配检查。

**同步校验：**模型目录中若有 `provenance.json`，脚本先校验已有 `export/build`
记录。ONNX、engine、已有 `model.yaml` 及其他必需产物缺失、哈希不符或阶段不完整时
会中止。同步负责重新生成的 `deployment/reference.yaml`、
`deployment/configuration.json` 不要求旧文件完整；缺失的历史构建日志也不阻塞同步。
现存构建日志若哈希不符，仍会中止。

多个 `--model-dir` 会先全部完成预检查，再开始写配置，避免后一模型校验失败时
前一模型已经被修改。写入阶段本身不是跨目录事务，磁盘等写入故障不会自动回滚。

```bash
cd /home/zhouxiufeng/workspace/openmmlab/lidar-inference-tool
conda activate openmmlab_py38

python scripts/deployment_config.py \
  --deployment-config /home/zhouxiufeng/lidar_perception_config/model.yaml \
  --model-dir demo/centerpoint_voxel \
  --model-dir demo/bevfusion_exp20261028
```

同步成功后写入以下记录：

| 文件 | 内容 |
|---|---|
| `model.yaml` | 实际生效的部署配置，标准 YAML 格式 |
| `deployment/reference.yaml` | 本次采用的基准原文 |
| `deployment/configuration.json` | 基准路径及哈希、生成配置哈希，以及被覆盖字段的旧值 |
| `deployment/previous_configuration.json` | 最近一次同步前的完整配置及阶段标识 |

已有 BEVFusion 台账会更新配置产物哈希和 build 的 export 依赖标识，保留原构建的
模型产物哈希、完成时间及工具记录，并移除 `validate/evaluate` 完成记录。
旧台账中将 `logs/build/*.log` 列为必需产物的记录会迁入 `diagnostic_artifacts`；
日志原哈希保留，缺失的日志不会伪造补写。后续构建也将这些日志记录为诊断产物。
若 `validation/`、`evaluation/`、`pooling_benchmark/` 已存在，则写入 `STALE.json`，
已有报告不因此删除；其他 `outputs/` 下的报告不会被同步脚本自动标记。
没有 `provenance.json` 的目录不会凭空生成 export/build 完成记录。

如果之前清理了配置快照和构建日志，重新运行上述同步命令即可重建配置记录并迁移
日志记录；无需为这些文件重新导出网络或构建 engine。同步不会掩盖模型文件的哈希
错误，也不会为缺失的 export/build 完成记录生成通过凭证。

新导出时，两条入口都调用同一个配置生成器，均可显式传入：

```text
--deployment-config /home/zhouxiufeng/lidar_perception_config/model.yaml
```

- CenterPoint：从 MMDetection3D 根目录调用
  `projects/CenterPoint/tools/export_voxel_split.py`；生成器默认位于同级
  `lidar-inference-tool/scripts/`，其他布局使用 `--deployment-tools-root` 指定工具仓库根目录。
- BEVFusion：从工具仓库根目录调用 `scripts/bevfusion/export.py`；实验配置
  `scripts/bevfusion/exp20261028.yaml` 的 `deployment_config` 字段已指向外部基准。

导出与配置同步不同：导出会写模型网络文件，应使用新的输出目录。
`run_pipeline.sh` 会依次执行导出、engine 构建和验证，当前实验默认验证 `20` 帧；
只想同步配置或复用现有 engine 时，不需要运行整个 pipeline。
BEVFusion 各独立步骤见 [部署工具说明](../scripts/bevfusion/README.md)。

## 4. 使用现有模型推理、对比和测速

以下命令从工具仓库根目录执行，先设置环境。Python 使用 `openmmlab_py38`；
只暴露物理 GPU 5 后，脚本中的 `cuda:0` 对应这张卡，不应再指定 `cuda:5`。

```bash
cd /home/zhouxiufeng/workspace/openmmlab/lidar-inference-tool
conda activate openmmlab_py38
export CUDA_VISIBLE_DEVICES=5
export TENSORRT_ROOT=/home/zhouxiufeng/TensorRT-8.6.1.6
export CUDA_HOME=/usr/local/cuda-11.7
export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib:$CUDA_HOME/lib64:$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
```

### 最小单帧推理

两条命令使用同一帧原始 PCD；BEVFusion 的 manifest 还引用同步 JPG 和标定信息。
PCD 按头信息读取，无需指定 `--source-num-point-features`。这些原始示例 PCD 没有
强度字段，加载为五维 `[x,y,z,0,0]`。帧来源见 [demo 数据](../demo/data/bevfusion/README.md)。

```bash
./build/infer_debug_tool --model-dir demo/centerpoint_voxel \
  --input demo/data/bevfusion/frames/1776582550193446516/lidar_point_cloud_0/1776582550193446516.pcd \
  --dump-dir outputs/shared_policy/centerpoint --warmup 0 --profile-iters 1 \
  --no-dump-internal --no-dump-decoded

./build/infer_debug_tool --model-dir demo/bevfusion_exp20261028 \
  --input demo/data/bevfusion/manifests/1776582550193446516.yaml \
  --dump-dir outputs/shared_policy/bevfusion --warmup 0 --profile-iters 1 \
  --no-dump-internal --no-dump-decoded
```

每模型只执行一次正常推理。输出包括 `result.json`、`final_*.npy`、
`final_label_names.json`、`input_points.npy`、`metadata.json` 和 `timings.json`；
BEVFusion 额外包含 `final_velocities.npy`。这两条命令关闭了解码和内部张量导出。
启用它们可能在计时之外追加推理，不属于这里的最小运行方式。

BEVFusion 默认 `--image-input raw`，加载原始 JPG；如需使用已有图像 NPY 缓存，
给 BEVFusion 命令添加 `--image-input preprocessed`。该参数不适用于 CenterPoint。

### 三帧可视化对比（按需执行）

```bash
python scripts/compare_demo_frames.py --output-dir outputs/comparison_shared_policy

# 或只选一帧；多个 --frame 可重复指定。
python scripts/compare_demo_frames.py \
  --frame 1776582550193446516 --output-dir outputs/comparison_shared_policy_one
```

默认选择混行、稀疏行人、相机屏蔽三帧。结果位于输出目录的
`<时间戳>/comparison.png`，同时保存各模型原始结果、配置摘要和 `comparison.json`；
根目录的 `README.md` 提供图片索引。标注从九类映射为五类，预测已是五类，不再次映射。
显示范围取两个模型点云范围的交集；三帧图用于观察差异，不代表完整数据集精度。

### 性能采集（按需执行）

```bash
./build/perf_debug_tool --model-dir demo/centerpoint_voxel \
  --input demo/data/bevfusion/frames/1776582550193446516/lidar_point_cloud_0/1776582550193446516.pcd \
  --dump-dir outputs/shared_policy_perf/centerpoint

./build/perf_debug_tool --model-dir demo/bevfusion_exp20261028 \
  --input demo/data/bevfusion/manifests/1776582550193446516.yaml \
  --dump-dir outputs/shared_policy_perf/bevfusion
```

Perf 默认只保存 `metadata.json` 和 `timings.json`，分两轮采集：正常推理轮关闭
阶段事件，阶段分析轮单独开启。**每轮**预热 `10` 次、统计 `50` 次，因此默认每模型
共执行 `120` 次推理；`--warmup`、`--profile-iters` 同时作用于两轮。

比较 `timings.json` 的 `total_infer_ms`；初始化与输入加载分别位于
`setup.runtime_init_ms`、`setup.input_load_ms`。读盘/JPEG 解码和输出写入不计入
正常推理总耗时，图像归一化及上传等推理内操作仍计入。阶段 host/CUDA 时间来自
另一轮，不能相加替代正常总延迟。单次无预热耗时也不能作为稳定性能结论。

建议为不同配置选用新的输出目录：工具会覆盖同名产物，但不会清理上次遗留的其他文件。

## 5. 验证含义与已验证范围

`scripts/bevfusion/validate.py` 使用部署体素限制和独立 NumPy 解码/OpenCV 几何参考，
检查当前部署策略下的分数、NMS 和五类输出，并检查原始图像/缓存、算子及相同引擎的
数值一致性。它不是简单的单次推理，即使选一帧也会执行多个检查，且依赖完整的
export/build 台账、源配置、checkpoint 和数据集。

成功时默认仅保留 `validation/report.json`；只有指定 `--keep-intermediates`
才保留参考结果及中间张量。检查失败时保留已有诊断文件。
参考中的 `native_head_policy_predictions.json` 表示采用部署体素限制、使用原生
Head 后处理的对照，不是完整复现训练默认配置的结果。

`scripts/bevfusion/evaluate.py` 使用完整验证集和现有九类 evaluator。它生成
`evaluation/metric_label_space.yaml`，仅在评测调用中将输出映射改为九类恒等映射，
其他部署后处理参数保持不变；原始模型参考也应用同一部署策略。
报告的 `original` 和 `deployment` 分别对应 PyTorch 网络与 C++ 部署，使用原始
九类指标，**不是业务五类指标，也不是训练默认后处理与部署策略的比较**。
评测不会改写模型目录的 `model.yaml`。评测报告保存在 `evaluation/report.json`，
阶段完成记录写入 `provenance.json`。

此前共享后处理变更只完成了受影响目标编译、公共配置和模型哈希核对、固定后处理
样例，以及两个模型各一次同帧推理。记录见
`outputs/centerhead_unification/checks.json`；没有据此验证完整精度、批量数值一致性或
稳定性能。变更前的验证、测速和三帧图属于历史结果，不能代表当前配置。

配置记录恢复修复仅使用一个 CPU 回归样例，覆盖快照缺失恢复和 engine 改动拒绝，
再执行配置同步；没有为此运行推理、测速、批量测试或模型重建。
