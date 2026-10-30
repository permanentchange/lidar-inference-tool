# 当前 BEVFusion 的计算与部署契约

唯一实现为 `CenterPointBEVFusion + CenterHead`，对应 exp20261028。
训练配置及 checkpoint 的模型参数、类别顺序必须严格一致。

## 计算路径

1. 原始 PCD 转为 `[x,y,z,intensity,has_intensity]`，按模型范围过滤；CUDA
   进行 HardSimpleVFE 体素均值，索引为 `[batch,z,y,x]`。
2. `lidar.backbone.zyx.onnx` 由 libspconv FP16 执行，输出 LiDAR BEV。
3. 使用实际 PCD 投影生成稀疏深度；像素冲突保留原始点序列最后一个点。
   相机按训练方式先裁剪再缩放到 224×384，保持 Pillow bicubic 的 uint8 舍入。
4. `camera.backbone.engine` 执行 ResNet50、FPN、稀疏深度 dtransform、depthnet
   和深度 softmax，输出 80 维相机特征及 58 个深度 bin 权重。
5. CUDA 根据标定及增强矩阵构建 geometry，按相机掩码聚合到 XY 布局 BEV。
   geometry 缓存键包含标定、增强和掩码，禁用相机不参与投影或 pooling。
6. `fusion_centerhead.engine` 执行相机 BEV 下采样、转置到 YX、按
   `[LiDAR,camera]` 拼接，再执行 SECOND、SECONDFPN 和六个 CenterHead task。
7. CUDA 解码和范围过滤后，使用与 CenterPoint 相同的候选排序和跨 task
   分组 NMS，按部署基准执行 `iou_min_bev`、自车过滤及分数过滤；最后将
   九类映射为 `bi/ped/tri/car/sight`。输出几何中心框和 `[vx,vy]`；调试解码结果保留九类。

模型有 9 类，task 类别索引依次为 `[[0,2],[1],[3,6],[4],[5],[7,8]]`。
不得将 task 局部通道直接当作全局类别。公共部署配置、同步方法及验证边界
见 [共用部署配置](../../docs/centerhead_deployment.md)。每体素 16 点和最多
100000 个体素属于指定的部署限制；独立参考同样采用这些限制。

## 图与精度

| 文件 | 执行器 | 输入 / 输出 |
|---|---|---|
| lidar.backbone.zyx.onnx | libspconv FP16 | 体素特征、ZYX 索引 → LiDAR BEV |
| camera.backbone.engine | TensorRT FP16 | 五路图像和稀疏深度 → 特征、深度权重 |
| fusion_centerhead.engine | TensorRT FP16 | LiDAR BEV、相机 XY BEV → 六 task 原始头张量 |

CenterHead 的 36 个输出绑定使用 FP32。最终六个 heatmap Conv 也强制 FP32，
避免 FP16 背景 logit 超过 65504 而产生 Inf。导出记录精确 Conv 名称，构建器
使用 precision constraints，并核验 engine layer inspection，不修改训练权重。

范围为 `[-8,-8,-1,16,8,3]`，voxel 为 `[0.05,0.05,0.1]`。
`20260612_pici2` 和 `20260613_pici1` 的 `camera_image_3` 按训练配置禁用；
原始图像可以保留，但计算必须使用原掩码。

`reference` pooling 是正确性基准；`nvidia_pack10` 使用 NVIDIA 分块 CUDA
聚合，保留 FP32 累加和 exclusive interval end。是否采用由独立 benchmark
依据张量/检测一致性和性能结果选择。

## 计时和验证

参见 [部署工具](README.md)。正常推理总延迟关闭阶段采集，阶段分析单独运行；
阶段 host/CUDA 时间不代表可相加的总耗时。输入读盘/JPEG 解码另计。
验证通过原始模型、独立算子、相同引擎及多帧检查确认实际部署行为。
原始 demo 的 PCD/JPG/相机 JSON 与标注见 [demo 数据](../../demo/data/bevfusion/README.md)。
