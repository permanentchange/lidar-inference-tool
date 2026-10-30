"""Capture references from original PyTorch prediction and generate per-frame validation manifests."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from .common import write
from .images import add_preprocess
from .model import metas_to_tensors

def prediction(sample):
    """Serialize a prediction sample as geometric center boxes, scores, and labels.

    Replace box centers with bboxes_3d.gravity_center and preserve all remaining
    components, including velocity, in their original order. Detach tensors and
    move them to CPU without modifying the prediction sample.
    """
    p = sample.pred_instances_3d
    boxes = p.bboxes_3d
    values = torch.cat([boxes.gravity_center, boxes.tensor[:, 3:]], dim=1).detach().float().cpu().numpy()
    return dict(boxes=values.tolist(), scores=p.scores_3d.detach().float().cpu().tolist(),
                labels=p.labels_3d.detach().cpu().tolist())


def original_reference(model, batch, directory, tensors=True, deployment=None):
    """Observe original model.predict and return (serialized_result, prediction_sample).

    Args:
        model: Original fusion model with loaded weights in eval mode, without
            deployment export wrappers.
        batch: Single-sample batch, deep-copied before model preprocessing.
        directory: Output directory; predictions.json is always written.
        deployment: Optional package config selecting the independent deployment policy.
        tensors: Capture images, sparse depth, BEV features, and other intermediate
            values through hooks as CPU FP32 NPY when true. Otherwise save only
            the prediction result.

    Do not replace model forward methods. Remove registered hooks on both normal
    and exceptional exits. Preserve original pixel-collision behavior in depth
    projection rather than rewriting it to be deterministic.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    captures, hooks = {}, []
    enabled = None
    head_tensors = []
    if deployment is not None:
        def capture_head(module, inputs, output):
            head_tensors[:] = [{k: v.detach().float().cpu().numpy().copy()
                                for k, v in task[0].items()} for task in output]
        hooks.append(model.pts_bbox_head.register_forward_hook(capture_head))
    def save(name, value):
        """Capture an independent FP32 CPU copy when enabled, taking the first element of sequences."""
        if tensors:
            if isinstance(value, (tuple, list)):
                value = value[0]
            captures[name] = value.detach().float().cpu().numpy().copy()
    if tensors:
        for module, name in ((model.pts_middle_encoder, 'sparse_bev'),
                             (model.pts_neck, 'fused_features'),
                             (model.view_transform.downsample, 'camera_bev')):
            hooks.append(module.register_forward_hook(lambda m, a, out, n=name: save(n, out)))
        def save_camera(name, value):
            if enabled is not None and value.shape[0] != len(enabled):
                restored = value.new_zeros((len(enabled), *value.shape[1:]))
                restored[enabled] = value
                value = restored
            save(name, value)
        hooks.append(model.view_transform.dtransform.register_forward_pre_hook(lambda m, a: save_camera('sparse_depth', a[0])))
        hooks.append(model.view_transform.downsample.register_forward_pre_hook(lambda m, a: save('camera_bev_before_downsample', a[0])))
        def depthnet(m, a, out):
            """Split original depthnet output into channels-last camera features and softmax depth probabilities."""
            save_camera('camera_feature', out[:, model.view_transform.D:].permute(0, 2, 3, 1).contiguous())
            save_camera('camera_depth_weights', out[:, :model.view_transform.D].softmax(1))
        hooks.append(model.view_transform.depthnet.register_forward_hook(depthnet))
    try:
        with torch.inference_mode():
            data = model.data_preprocessor(copy.deepcopy(batch), training=False)
            enabled = torch.tensor(data['data_samples'][0].metainfo.get(
                'camera_valid_mask', [True] * data['inputs']['imgs'].shape[1]),
                dtype=torch.bool, device=data['inputs']['imgs'].device)
            save('images', data['inputs']['imgs'][0])
            pred = model.predict(data['inputs'], data['data_samples'])[0]
            result = prediction(pred)
            if deployment is not None:
                # Retain the native head policy separately; it does not define deployment output.
                write(directory / 'native_head_policy_predictions.json', result)
                from .deployment_reference import predict
                from mmengine.structures import InstanceData
                from mmdet3d.structures import LiDARInstance3DBoxes
                result = predict(head_tensors, deployment)
                device = pred.pred_instances_3d.scores_3d.device
                values = torch.tensor(result['boxes'], dtype=torch.float32, device=device).reshape(-1, 9)
                pred.pred_instances_3d = InstanceData(
                    bboxes_3d=LiDARInstance3DBoxes(values, box_dim=9, origin=(.5, .5, .5)),
                    scores_3d=torch.tensor(result['scores'], dtype=torch.float32, device=device),
                    labels_3d=torch.tensor(result['labels'], dtype=torch.long, device=device))
    finally:
        for hook in hooks:
            hook.remove()
    for name, array in captures.items():
        np.save(directory / (name + '.npy'), array)
    write(directory / 'predictions.json', result)
    return result, pred


def fixture(cfg, model, batch, folder):
    """Save one preprocessed batch as deployment inputs and return folder/sample.yaml.

    Use the original preprocessor to obtain single-frame points and images as
    FP32 NPY, and record calibration, augmentation, and integer resize/crop
    metadata. Reference original images by absolute path without copying JPEGs;
    points and preprocessed images use manifest-relative paths. These validation
    fixtures therefore depend on dataset images, unlike the self-contained sample/
    in an exported package.
    """
    folder.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        data = model.data_preprocessor(copy.deepcopy(batch), training=False)
        images = data['inputs']['imgs']
        meta, _, intrinsic, camera2lidar, img_aug, lidar_aug = metas_to_tensors(images, data['data_samples'])
        np.save(folder / 'images.npy', images[0].float().cpu().numpy())
        points = data['inputs']['points'][0].float().cpu().numpy()
        np.save(folder / 'points.npy', points)
    cameras = []
    for i, image in enumerate(meta[0]['img_path']):
        cameras.append(dict(name=f'camera_image_{i}', image=str(Path(image).resolve()),
                            enabled=bool(meta[0].get('camera_valid_mask', [True] * len(meta[0]['img_path']))[i]),
                            cam2img=intrinsic[0, i].cpu().tolist(), cam2lidar=camera2lidar[0, i].cpu().tolist(),
                            lidar2img=np.asarray(meta[0]['lidar2img'][i]).tolist(), img_aug_matrix=img_aug[0, i].cpu().tolist()))
    manifest = folder / 'sample.yaml'
    write(manifest, dict(version=2, point_cloud='points.npy', preprocessed_images='images.npy', num_point_features=points.shape[1],
                         lidar_aug_matrix=lidar_aug[0].cpu().tolist(), cameras=cameras))
    add_preprocess(manifest, cfg)
    return manifest
