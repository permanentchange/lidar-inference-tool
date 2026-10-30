#!/usr/bin/env python3
"""CPU checks for the aligned graph, using independent non-square expectations."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/bevfusion'))
from _lib.onnx_export import FusionCenterHeadExport, CameraBackboneExport
from _lib.images import add_preprocess
from _lib.common import read, write
from mmdet3d.utils import register_all_modules
from mmdet3d.registry import TRANSFORMS
import projects.BEVFusion.bevfusion  # registers the training crop implementation


class Neck(nn.Module):
    def forward(self, x):
        return [x]


class Head(nn.Module):
    def forward(self, features):
        return [[{key: features[0] for key in
                  ('reg', 'height', 'dim', 'rot', 'vel', 'heatmap')}]]


class CenterHeadExportTest(unittest.TestCase):
    def test_downsample_axes_and_lidar_first_concatenation(self):
        model = nn.Module()
        model.view_transform = nn.Module()
        model.view_transform.downsample = nn.Conv2d(1, 1, 3, 2, 1, bias=False)
        with torch.no_grad():
            model.view_transform.downsample.weight.copy_(torch.arange(9).reshape(1, 1, 3, 3))
        model.pts_backbone = nn.Identity()
        model.pts_neck = Neck()
        model.pts_bbox_head = Head()
        lidar = torch.full((1, 2, 2, 3), 101.)
        camera = torch.arange(24, dtype=torch.float32).reshape(1, 1, 6, 4)
        output = FusionCenterHeadExport(model)(lidar, camera)
        expected_camera = model.view_transform.downsample(camera).permute(0, 1, 3, 2)
        expected = torch.cat([lidar, expected_camera], dim=1)
        self.assertEqual(len(output), 6)
        for value in output:
            torch.testing.assert_close(value, expected)

    def test_camera_graph_keeps_learned_depth_encoder_and_softmax(self):
        model = nn.Module()
        model.img_backbone = nn.Identity()
        model.img_neck = Neck()
        model.view_transform = nn.Module()
        view = model.view_transform
        view.D, view.C = 2, 1
        view.dtransform = nn.Conv2d(1, 1, 1, bias=False)
        view.depthnet = nn.Conv2d(4, 3, 1, bias=False)
        with torch.no_grad():
            view.dtransform.weight.fill_(2.)
            view.depthnet.weight.zero_()
            view.depthnet.weight[0, 0] = 1
            view.depthnet.weight[1, 0] = -1
            view.depthnet.weight[2, 0] = 3
        camera = CameraBackboneExport(model)
        # Depth features are concatenated before the three image channels.
        feature, probability = camera(torch.zeros(2, 3, 2, 3), torch.ones(2, 1, 2, 3))
        torch.testing.assert_close(feature, torch.full((2, 2, 3, 1), 6.))
        torch.testing.assert_close(probability.sum(1), torch.ones(2, 2, 3))
        self.assertTrue(torch.all(probability[:, 0] > .98))

    def test_native_crop_metadata_matches_training_transform(self):
        register_all_modules(init_default_scope=True)
        aug = dict(type='AspectRatioCropResize3D', final_dim=[224, 384],
                   is_train=False, center_width_keep=.9, rand_flip=True, rot_lim=[-5.4, 5.4])
        transform = TRANSFORMS.build(aug)
        cfg = dict(test_dataloader=dict(dataset=dict(pipeline=[aug])))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cameras = []
            for i, (width, height) in enumerate(((640, 360), (1920, 1536))):
                filename = f'camera_{i}.jpg'
                Image.new('RGB', (width, height)).save(root / filename)
                _, rotation, translation = transform.img_transform(
                    np.zeros((height, width, 3), dtype=np.uint8), torch.eye(2), torch.zeros(2),
                    *transform.sample_augmentation((height, width)))
                matrix = np.eye(4, dtype=np.float32)
                matrix[:2, :2] = rotation.numpy()
                matrix[:2, 3] = translation.numpy()
                cameras.append(dict(image=filename, img_aug_matrix=matrix.tolist()))
            cameras.append(dict(enabled=False))  # no file or calibration needed
            manifest = root / 'sample.yaml'
            write(manifest, dict(cameras=cameras))
            add_preprocess(manifest, cfg)
            node = read(manifest)
            for camera, shape in zip(node['cameras'], ((360, 640), (1536, 1920))):
                self.assertEqual(camera['preprocess']['operation'], 'crop_resize')
                self.assertEqual(camera['preprocess']['source_crop'], list(transform.crop_box(shape)))
                self.assertEqual(camera['preprocess']['resize_size'], [384, 224])


if __name__ == '__main__':
    unittest.main()
