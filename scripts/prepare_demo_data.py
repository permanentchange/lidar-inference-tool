#!/usr/bin/env python3
"""Copy four fixed, source-identifiable frames and prepare portable BEVFusion manifests."""
import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
SELECTIONS = (
    ('20260419_pici1', '1776574418716918507', 'original_demo'),
    ('20260419_pici1', '1776582550193446516', 'mixed_traffic'),
    ('20260419_pici1', '1776580417398125945', 'sparse_pedestrians'),
    ('20260613_pici1', '1781344892413737925', 'masked_camera'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default='/training/zhouxiufeng/data/zhonghengzhilian')
    parser.add_argument('--mmdet3d-root', default=str(REPO.parent / 'mmdetection3d'))
    parser.add_argument('--output-dir', default=str(REPO / 'demo/data/bevfusion'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.mmdet3d_root).resolve()))
    sys.path.insert(0, str(REPO / 'scripts/bevfusion'))
    from _lib.common import digest, write
    from _lib.images import add_preprocess
    import numpy as np
    from mmengine import Config
    from mmengine.utils import import_modules_from_strings
    from mmdet3d.registry import DATASETS
    from mmdet3d.utils import register_all_modules
    cfg_path = Path(args.mmdet3d_root) / 'projects/BEVFusion/configs/exp20261028/lidar_cam.py'
    cfg = Config.fromfile(str(cfg_path))
    register_all_modules()
    import_modules_from_strings(**cfg.custom_imports)
    output = Path(args.output_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    if output == dataset_root or dataset_root in output.parents:
        raise RuntimeError('Demo output must be separate from the source dataset')
    children = {Path(c.data_root).name: c for c in cfg.test_dataloader.dataset.datasets}
    datasets = {}
    catalog = []
    for dataset_name, stamp, reason in SELECTIONS:
        root = dataset_root / dataset_name
        info_path = root / 'zhzl_bevfusion_infos_test.pkl'
        with info_path.open('rb') as stream:
            raw = pickle.load(stream)['data_list']
        original = next(row for row in raw if Path(row['lidar_points']['lidar_path']).stem == stamp)
        if dataset_name not in datasets:
            child = children[dataset_name].copy()
            child.data_root = str(root)
            child.ann_file = str(info_path)
            datasets[dataset_name] = DATASETS.build(child)
        dataset = datasets[dataset_name]
        index = next(i for i in range(len(dataset)) if dataset.get_data_info(i)['token'] == original['token'])
        item = dataset[index]
        meta = item['data_samples'].metainfo
        frame = output / 'frames' / stamp
        records = []
        def copy(source, relative):
            destination = frame / relative
            if destination.exists() and digest(destination) != digest(source) and not args.overwrite:
                raise RuntimeError('Different demo data already exists: ' + str(destination))
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or digest(destination) != digest(source):
                shutil.copy2(source, destination)
            records.append(dict(source=str(source), file=relative, sha256=digest(destination)))
            return destination
        pcd = copy(root / 'samples/lidar_point_cloud_0' / (stamp + '.pcd'), 'lidar_point_cloud_0/' + stamp + '.pcd')
        copy(root / 'samples/camera_config' / (stamp + '.json'), 'camera_config/' + stamp + '.json')
        cameras = []
        for i, (name, camera) in enumerate(original['images'].items()):
            image = copy(root / 'samples' / name / camera['img_path'], name + '/' + camera['img_path'])
            c = dict(name=name, image=os.path.relpath(image, output / 'manifests'),
                     enabled=bool(meta.get('camera_valid_mask', [True]*5)[i]))
            for field in ('cam2img', 'cam2lidar', 'lidar2img', 'img_aug_matrix'):
                c[field] = np.asarray(meta[field][i], dtype=np.float32).tolist()
            cameras.append(c)
        cache = output / 'derived' / stamp / 'images.npy'
        cache.parent.mkdir(parents=True, exist_ok=True)
        images = item['inputs']['img'].float()
        mean = images.new_tensor(cfg.model.data_preprocessor.mean).reshape(1,3,1,1)
        std = images.new_tensor(cfg.model.data_preprocessor.std).reshape(1,3,1,1)
        np.save(cache, ((images-mean)/std).numpy())
        manifest = output / 'manifests' / (stamp + '.yaml')
        node = dict(version=2, point_cloud=os.path.relpath(pcd, manifest.parent),
                    preprocessed_images=os.path.relpath(cache, manifest.parent), num_point_features=5,
                    lidar_aug_matrix=np.eye(4, dtype=np.float32).tolist(), cameras=cameras)
        write(manifest, node)
        add_preprocess(manifest, cfg)
        annotations = []
        for ann in original['instances']:
            label = int(ann['bbox_label_3d'])
            if label < 0: continue
            velocity = ann.get('velocity', [float('nan')]*2)
            annotations.append(dict(box=ann['bbox_3d'], label=label, label_name=cfg.class_names[label],
                valid=bool(ann.get('bbox_3d_isvalid', True)), num_lidar_points=int(ann.get('num_lidar_pts',0)),
                velocity=[float(v) if np.isfinite(v) else None for v in velocity]))
        write(frame / 'annotations.json', dict(box_z_origin='center', coordinates='lidar', objects=annotations))
        # Read binary PCD and compare every point to the actual training BIN.
        with pcd.open('rb') as stream:
            while True:
                line = stream.readline()
                if not line: raise RuntimeError('PCD DATA header missing')
                if line.startswith(b'DATA '): break
            if line.strip() != b'DATA binary': raise RuntimeError('Expected source binary XYZ PCD')
            xyz = np.fromfile(stream, dtype=np.float32).reshape(-1,3)
        train_points = np.fromfile(root / 'samples/lidar_point_cloud_0' / (stamp+'.bin'), dtype=np.float32).reshape(-1,5)
        if not np.array_equal(xyz, train_points[:,:3]) or np.any(train_points[:,3:]):
            raise RuntimeError('Source PCD differs from the training point features: ' + stamp)
        write(frame / 'source.json', dict(dataset=dataset_name, split='test', dataset_index=index,
            annotation_index=next(i for i,row in enumerate(raw) if row['token']==original['token']),
            sample_token=original['token'], scene_token=original['scene_token'], scene_name=original['scene_name'],
            timestamp_ns=int(stamp), annotation_source=str(info_path), annotation_sha256=digest(info_path),
            config=str(cfg_path), config_sha256=digest(cfg_path), reason=reason,
            source_points=len(xyz), training_point_features_equal=True, files=records))
        catalog.append(dict(frame_id=stamp, dataset=dataset_name, reason=reason,
            manifest='manifests/'+stamp+'.yaml', point_cloud=os.path.relpath(pcd, output)))
        print('Prepared', stamp, reason, len(xyz), flush=True)
    write(output / 'index.json', dict(version=1, frames=catalog))
    print('Demo data ready:', output)

if __name__ == '__main__':
    main()
