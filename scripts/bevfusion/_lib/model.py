# SPDX-License-Identifier: Apache-2.0
# Includes MMDetection3D dataset/metainfo adaptations; see NOTICE.
"""Load models and datasets through MMDetection3D and organize camera metadata."""
import copy
from typing import List, Any
from torch import Tensor
import numpy as np
import torch
from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules

def is_centerhead(model):
    """Check the supported BEVFusion detection head."""
    return hasattr(model, 'pts_bbox_head') and model.pts_bbox_head.__class__.__name__ == 'CenterHead'

def load_model(config, checkpoint, device):
    """Strictly load a fusion model compatible with the deployment baseline.

    Args:
        config: MMEngine configuration file matching the checkpoint.
        checkpoint: Fusion checkpoint containing a complete state_dict.
        device: Torch device onto which the model is moved.

    Returns:
        (cfg, model), with the model in eval mode.

    Clear the image backbone init_cfg in a configuration copy before construction
    to avoid fetching separate pretrained weights. Require DepthLSSTransform,
    RGB input and CenterHead with task-aware IoU NMS. If the checkpoint
    stores its configuration, also verify the model settings and class ordering.
    Incompatible structures or incomplete weights raise RuntimeError.
    """
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config))
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    if cfg.model.view_transform.type != 'DepthLSSTransform':
        raise RuntimeError('Baseline requires DepthLSSTransform')
    if cfg.model.type != 'CenterPointBEVFusion':
        raise RuntimeError('BEVFusion requires CenterPointBEVFusion with CenterHead')
    head_cfg = cfg.model.pts_bbox_head
    if head_cfg.type != 'CenterHead':
        raise RuntimeError('Aligned deployment requires CenterHead')
    if cfg.model.data_preprocessor.get('bgr_to_rgb', False):
        raise RuntimeError('Baseline requires RGB pipeline with bgr_to_rgb=False')
    build_cfg = copy.deepcopy(cfg.model)
    # Strict checkpoint loading supplies every parameter; no pretrained
    # initialization download/file is needed just to construct the model.
    if build_cfg.get('img_backbone') is not None:
        build_cfg.img_backbone.init_cfg = None
    model = MODELS.build(build_cfg)
    state = torch.load(str(checkpoint), map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=True)
    if (cfg.model.pts_voxel_encoder.type != 'HardSimpleVFE' or
                    head_cfg.get('yaw_encoding', 'sin_cos') != 'sin_cos' or
                    model.pts_bbox_head.test_cfg.get('nms_type') != 'iou_bev'):
        raise RuntimeError('Aligned deployment requires HardSimpleVFE, sin_cos yaw and iou_bev NMS')
    saved_cfg = state.get('meta', {}).get('cfg')
    if saved_cfg:
        prior = Config.fromstring(saved_cfg, '.py')
        if prior.model != cfg.model:
            raise RuntimeError('Checkpoint model configuration differs from requested config')
        if prior.get('class_names') != cfg.get('class_names'):
            raise RuntimeError('Checkpoint class ordering differs from requested config')
    return cfg, model.to(device).eval()

def build_dataloader(cfg: Config):
    """Build an ordered, single-sample loader for export and numerical validation.

    Prefer test_dataloader and fall back to val_dataloader. Copy its configuration,
    set batch_size=1 and num_workers=0, keep the final batch, and disable shuffle
    in dictionary sampler configurations. Return the MMEngine loader without
    modifying cfg.
    """
    if cfg.get('test_dataloader') is not None:
        dataloader_cfg = copy.deepcopy(cfg.test_dataloader)
    elif cfg.get('val_dataloader') is not None:
        dataloader_cfg = copy.deepcopy(cfg.val_dataloader)
    else:
        raise RuntimeError('config has neither test_dataloader nor val_dataloader')
    dataloader_cfg.update(
        batch_size=1, num_workers=0, persistent_workers=False, drop_last=False)
    if isinstance(dataloader_cfg.get('sampler'), dict):
        dataloader_cfg['sampler']['shuffle'] = False
    return Runner.build_dataloader(dataloader_cfg, seed=0)


def get_data_batch(cfg: Config, sample_index: int):
    """Iterate to a zero-based sample index and return its batch before model preprocessing.

    Use build_dataloader ordering. Negative indices raise ValueError and indices
    beyond the loader length raise IndexError. The returned batch can be passed
    directly to model.data_preprocessor.
    """
    if sample_index < 0:
        raise ValueError('--sample-index must be non-negative')
    iterator = iter(build_dataloader(cfg))
    for _ in range(sample_index + 1):
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise IndexError(f'sample index {sample_index} exceeds dataset length') from exc
    return batch


def metas_to_tensors(images: Tensor, data_samples: List[Any]):
    """Convert calibration and augmentation metadata to the device and dtype of images.

    Return the original metainfo list, lidar2img, cam2img, cam2lidar,
    img_aug_matrix, and lidar_aug_matrix, in that order. Preserve batch and camera
    axes in camera matrices. Missing augmentation matrices default to a 4x4
    identity; this helper does not check invertibility or expand camera counts.
    """
    metas = [sample.metainfo for sample in data_samples]
    lidar2image = images.new_tensor(np.asarray([meta['lidar2img'] for meta in metas]))
    camera_intrinsics = images.new_tensor(np.asarray([meta['cam2img'] for meta in metas]))
    camera2lidar = images.new_tensor(np.asarray([meta['cam2lidar'] for meta in metas]))
    img_aug_matrix = images.new_tensor(np.asarray(
        [meta.get('img_aug_matrix', np.eye(4)) for meta in metas]))
    lidar_aug_matrix = images.new_tensor(np.asarray(
        [meta.get('lidar_aug_matrix', np.eye(4)) for meta in metas]))
    return metas, lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix


def library_sources(root):
    """Return relative paths and digests for BEVFusion model sources and sparse_block.py.

    root is the training project directory. Exclude its deployment tools. The
    mapping identifies whether export and later reference inference use the
    same model implementation.
    """
    from .common import digest
    files = list((root / 'projects/BEVFusion/bevfusion').rglob('*.py'))
    files.append(root / 'mmdet3d/models/layers/sparse_block.py')
    return {str(p.relative_to(root)): digest(p) for p in sorted(files)}
