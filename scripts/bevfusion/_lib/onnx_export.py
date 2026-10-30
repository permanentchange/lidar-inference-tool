# SPDX-License-Identifier: Apache-2.0
# Portions adapted from MMDetection3D, copyright OpenMMLab contributors.
# See licenses/Apache-2.0.txt and NOTICE for source attribution.
"""Maintain local split-network ONNX exporters and package samples, references, and model configuration."""
import shutil
import copy
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import onnx
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .common import write
from .model import get_data_batch, metas_to_tensors, is_centerhead
from .images import add_preprocess
from .reference import original_reference
from .sparse_export import export_sparse_encoder_onnx

class CameraBackboneExport(nn.Module):
    """Export the image backbone, neck, and sparse-depth fusion network.

    Reuse the supplied model's modules and weights, treating each camera as a
    batch element. Output channels-last image features and depth probabilities.
    Geometry and BEV pooling run outside the graph; this wrapper performs neither
    sparse-depth projection nor image preprocessing.
    """

    def __init__(self, model: nn.Module):
        """Reference the fusion model image/depth networks and record depth bins D and feature channels C."""
        super().__init__()
        self.backbone = model.img_backbone
        self.neck = model.img_neck
        self.dtransform = model.view_transform.dtransform
        self.depthnet = model.view_transform.depthnet
        self.depth_bins = model.view_transform.D
        self.feature_channels = model.view_transform.C

    def forward(self, images: Tensor, sparse_depth: Tensor) -> Tuple[Tensor, Tensor]:
        """Return image features and depth-normalized probabilities for CUDA pooling.

        Args:
            images: Normalized images [Ncam, 3, Himg, Wimg].
            sparse_depth: Projected depth aligned with images [Ncam, 1, Himg, Wimg].

        Returns:
            camera_features [Ncam, Hf, Wf, C] and depth_weights [Ncam, D, Hf, Wf],
            preserving input camera order.
        """
        features = self.neck(self.backbone(images))
        if not isinstance(features, Tensor):
            features = features[0]
        depth_features = self.dtransform(sparse_depth)
        depth_and_features = self.depthnet(torch.cat([depth_features, features], dim=1))
        depth_weights = depth_and_features[:, :self.depth_bins].softmax(dim=1)
        # CUDA BEV pooling consumes channels-last image features, one camera per
        # batch element: [N, H, W, C].
        camera_features = depth_and_features[:, self.depth_bins:
                                             self.depth_bins + self.feature_channels]
        return camera_features.permute(0, 2, 3, 1).contiguous(), depth_weights


class FusionCenterHeadExport(nn.Module):
    """Keep trained XY downsampling before YX fusion and emit raw task maps."""

    def __init__(self, model):
        super().__init__()
        self.downsample = model.view_transform.downsample
        self.backbone = model.pts_backbone
        self.neck = model.pts_neck
        self.head = model.pts_bbox_head

    def forward(self, lidar_bev, camera_bev):
        camera = self.downsample(camera_bev).transpose(-1, -2).contiguous()
        features = self.neck(self.backbone(torch.cat([lidar_bev, camera], dim=1)))
        predictions = self.head(features)
        return tuple(task[0][key].float() for task in predictions
                     for key in ('reg', 'height', 'dim', 'rot', 'vel', 'heatmap'))


def _numpy(value: Tensor) -> np.ndarray:
    """Detach a tensor and convert it to a CPU float32 NumPy array for reference serialization."""
    return value.detach().float().cpu().numpy()


def _copy_sample_images(output_dir: Path, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Copy original images into sample/ in metadata order and return camera names and relative paths."""
    sample_dir = output_dir / 'sample'
    sample_dir.mkdir(parents=True, exist_ok=True)
    cameras = []
    for index, source_path in enumerate(meta['img_path']):
        enabled = bool(meta.get('camera_valid_mask', [True] * len(meta['img_path']))[index])
        if not enabled:
            cameras.append({'name': f'camera_image_{index}', 'enabled': False})
            continue
        source = Path(source_path)
        suffix = source.suffix.lower() or '.jpg'
        destination_name = f'camera_image_{index}/{source.name}'
        destination = sample_dir / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        cameras.append({'name': f'camera_image_{index}', 'image': destination_name, 'enabled': True})
    return cameras


def _export_onnx(module: nn.Module, inputs: Tuple[Tensor, ...], path: Path,
                 input_names: List[str], output_names: List[str], opset: int) -> None:
    """Export fixed-shape dense ONNX with the requested bindings/opset and check graph validity.

    Put module in eval mode and use actual input shapes with constant folding.
    Do not create parent directories, check overwrite permission, or manage
    stage metadata.
    """
    module.eval()
    with torch.inference_mode():
        torch.onnx.export(module, inputs, str(path), input_names=input_names,
                          output_names=output_names, opset_version=opset,
                          do_constant_folding=True)
    onnx.checker.check_model(str(path))


def export_package(model, cfg, output_dir, sample_index=0, device="cuda:0", deployment_config=None):
    """Export the supported CenterHead BEVFusion package."""
    if not is_centerhead(model):
        raise RuntimeError('BEVFusion requires CenterHead')
    return export_centerhead_package(model, cfg, Path(output_dir), sample_index, device, deployment_config)


def export_centerhead_package(model, cfg, output_dir, sample_index, device, deployment_config=None):
    """Export the aligned model with GPU pooling between two dense engines."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    batch = get_data_batch(cfg, sample_index)
    data = model.data_preprocessor(copy.deepcopy(batch), training=False)
    images = data['inputs']['imgs'].to(device).float()
    points = [p.to(device).float() for p in data['inputs']['points']]
    samples = [s.to(device) for s in data['data_samples']]
    metas, l2i, intr, c2l, aug, laug = metas_to_tensors(images, samples)
    if images.shape[0] != 1 or tuple(images.shape[-2:]) != tuple(model.view_transform.image_size):
        raise RuntimeError('Expected batch one and pipeline/view-transform image sizes to agree')
    count = images.shape[1]
    enabled = torch.as_tensor(metas[0].get('camera_valid_mask', [True] * count),
                              dtype=torch.bool, device=device)
    # The original model replaces disabled calibration with identity matrices.
    for matrices in (l2i, intr, c2l, aug):
        matrices[:, ~enabled] = torch.eye(4, device=device)
    original_reference(model, batch, output_dir / 'reference_original')
    depth = torch.from_numpy(np.load(output_dir / 'reference_original/sparse_depth.npy')).to(device)
    camera = CameraBackboneExport(model).to(device).eval()
    fusion = FusionCenterHeadExport(model).to(device).eval()
    with torch.inference_mode():
        vox = data['inputs']['voxels']
        features = model.pts_voxel_encoder(vox['voxels'], vox['num_points'], vox['coors'])
        coordinates = vox['coors']
        lidar = model.pts_middle_encoder(features, coordinates, 1)
        feat, weights = camera(images[0], depth)
        feat = feat * enabled[:, None, None, None]
        weights = weights * enabled[:, None, None, None]
        geometry = model.view_transform.get_geometry(
            c2l[..., :3, :3], c2l[..., :3, 3], intr[..., :3, :3],
            aug[..., :3, :3], aug[..., :3, 3],
            extra_rots=laug[..., :3, :3], extra_trans=laug[..., :3, 3])
        weighted = weights.unsqueeze(-1) * feat.unsqueeze(1)
        pooled = model.view_transform.bev_pool(geometry, weighted.unsqueeze(0), enabled[None])
        outputs = fusion(lidar, pooled)
    names = [f'{name}_{i}' for i in range(len(model.pts_bbox_head.task_heads))
             for name in ('reg', 'height', 'dim', 'rot', 'vel', 'hm')]
    export_sparse_encoder_onnx(model.pts_middle_encoder, features, coordinates, 1,
                               output_dir / 'lidar.backbone.zyx.onnx')
    _export_onnx(camera, (images[0], depth), output_dir / 'camera.backbone.onnx',
                 ['img', 'depth'], ['camera_feature', 'camera_depth_weights'], 13)
    _export_onnx(fusion, (lidar, pooled), output_dir / 'fusion_centerhead.onnx',
                 ['input', 'camera_bev'], names, 13)
    # A float32 output binding alone does not stop TensorRT choosing FP16 for
    # the preceding Conv. Strong negative background logits can exceed 65504.
    graph = onnx.load(str(output_dir / 'fusion_centerhead.onnx')).graph
    producers = {value: node for node in graph.node for value in node.output}
    heatmap_layers = []
    for name in (value for value in names if value.startswith('hm_')):
        producer = producers[name]
        while producer.op_type in ('Cast', 'Identity'):
            producer = producers[producer.input[0]]
        if producer.op_type != 'Conv':
            raise RuntimeError('Expected a final heatmap Conv for ' + name)
        heatmap_layers.append(producer.name)
    sample_dir = output_dir / 'sample'
    sample_dir.mkdir(exist_ok=True)
    point_source = Path(metas[0]['lidar_path']).with_suffix('.pcd')
    point_input = 'points.npy'
    if point_source.is_file():
        point_input = 'lidar_point_cloud_0/' + point_source.name
        (sample_dir / 'lidar_point_cloud_0').mkdir(exist_ok=True)
        shutil.copy2(point_source, sample_dir / point_input)
        camera_config = point_source.parent.parent / 'camera_config' / (point_source.stem + '.json')
        if camera_config.is_file():
            (sample_dir / 'camera_config').mkdir(exist_ok=True)
            shutil.copy2(camera_config, sample_dir / 'camera_config' / camera_config.name)
    else:
        np.save(sample_dir / 'points.npy', _numpy(points[0]))
    np.save(sample_dir / 'images.npy', _numpy(images[0]))
    cameras = _copy_sample_images(output_dir, metas[0])
    for i, cam in enumerate(cameras):
        for name, tensor in (('cam2img', intr), ('cam2lidar', c2l),
                             ('lidar2img', l2i), ('img_aug_matrix', aug)):
            cam[name] = _numpy(tensor[0, i]).tolist()
    write(sample_dir / 'sample.yaml', dict(version=2, point_cloud=point_input,
          preprocessed_images='images.npy', num_point_features=points[0].shape[1],
          lidar_aug_matrix=_numpy(laug[0]).tolist(), cameras=cameras))
    add_preprocess(sample_dir / 'sample.yaml', cfg)
    references = dict(images=images[0], sparse_depth=depth, sparse_bev=lidar,
                      camera_feature=feat, camera_depth_weights=weights,
                      camera_bev_before_downsample=pooled)
    references.update(zip(names, outputs))
    (output_dir / 'reference').mkdir(exist_ok=True)
    for name, tensor in references.items():
        np.save(output_dir / 'reference' / (name + '.npy'), _numpy(tensor))
    head = model.pts_bbox_head
    test = head.test_cfg
    view = model.view_transform
    voxel = cfg.model.data_preprocessor.voxel_layer
    task_ids = [[cfg.class_names.index(name) for name in task['class_names']]
                for task in cfg.model.pts_bbox_head.tasks]
    artifact_names = ['lidar.backbone.zyx.onnx', 'camera.backbone.onnx',
                      'fusion_centerhead.onnx', 'model.yaml', 'sample/sample.yaml']
    node = dict(version='8.1', file_list=artifact_names, model=dict(
        runtime_backend='bevfusion', architecture='centerpoint_bevfusion',
        scn_onnx_path='lidar.backbone.zyx.onnx', camera_engine_path='camera.backbone.engine',
        head_engine_path='fusion_centerhead.engine', num_point_features=points[0].shape[1],
        head_fp32_layers=heatmap_layers,
        point_cloud_range=list(cfg.point_cloud_range), post_center_range=list(cfg.post_center_range),
        voxel_size=list(cfg.voxel_size), sparse_shape=list(model.pts_middle_encoder.sparse_shape),
        out_size_factor=int(head.bbox_coder.out_size_factor), bbox_code_size=9,
        has_velocity=True, norm_bbox=bool(head.norm_bbox), label_names=list(cfg.class_names),
        task_label_ids=task_ids, camera=dict(num_cameras=count,
        camera_names=[c['name'] for c in cameras], image_size=list(view.image_size),
        feature_size=list(view.feature_size), feature_channels=int(view.C), depth_bins=int(view.D),
        mean=_numpy(model.data_preprocessor.mean.flatten()).tolist(),
        std=_numpy(model.data_preprocessor.std.flatten()).tolist(),
        xbound=list(view.xbound), ybound=list(view.ybound), zbound=list(view.zbound), dbound=list(view.dbound))),
        runtime=dict(max_points_per_voxel=int(voxel.max_num_points), max_voxels=int(voxel.max_voxels[1]),
                     max_points=300000, bbox_max_num=int(head.bbox_coder.max_num), bev_pool_backend='reference'),
        postprocess=dict(nms_type=test.nms_type, score_threshold=float(head.bbox_coder.score_threshold),
            nms_thresh=float(test.nms_thr), nms_pre_maxsize=int(test.pre_max_size),
            nms_post_maxsize=int(test.post_max_size), final_score_threshold=float(test.final_score_threshold),
            label_score_thresh=list(test.label_score_thresh), nms_label_groups=copy.deepcopy(test.nms_label_groups)))
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from deployment_config import DEFAULT_CONFIG, write_package
    write_package(output_dir, node, deployment_config or DEFAULT_CONFIG, check_geometry=True)
    print('Exported CenterPoint BEVFusion package:', output_dir, flush=True)
