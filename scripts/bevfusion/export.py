#!/usr/bin/env python3
"""Export a trained BEVFusion model using this project's local ONNX exporters."""
import argparse
import sys
from pathlib import Path
from _lib.common import (REPO_ROOT, begin_stage, digest, finish_stage, nonnegative,
                         require_overwrite, reset_directory, run, versions, read)
from _lib.environment import training_context

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from deployment_config import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mmdet3d-root', required=True)
    parser.add_argument('--config', required=True, help='Model config, relative to MMDetection3D root')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint, relative to MMDetection3D root')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--deployment-config', type=Path, default=DEFAULT_CONFIG, help='Shared deployment profile YAML')
    parser.add_argument('--sample-dir', help='Separate destination for generated demo inputs')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sample-index', type=nonnegative, default=0)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    args.deployment_config = args.deployment_config.resolve()
    root, output = Path(args.mmdet3d_root).resolve(), Path(args.output_dir).resolve()
    config, checkpoint = (root / args.config).resolve(), (root / args.checkpoint).resolve()
    owned = [output / name for name in ('model.yaml', 'lidar.backbone.zyx.onnx',
             'camera.backbone.onnx', 'fusion_centerhead.onnx', 'sample', 'reference', 'reference_original', 'deployment')]
    require_overwrite(owned, args.overwrite)
    training_context(root, args.device)
    from _lib.model import library_sources, load_model
    from _lib.onnx_export import export_package
    cfg, model = load_model(config, checkpoint, args.device)
    source = dict(mmdet3d_root=str(root), config=str(config), checkpoint=str(checkpoint),
                  config_sha256=digest(config), checkpoint_sha256=digest(checkpoint),
                  deployment_config=str(args.deployment_config.resolve()),
                  deployment_config_sha256=digest(args.deployment_config),
                  library_sources=library_sources(root))
    begin_stage(output, 'export', source)
    for name in ('sample', 'reference', 'reference_original'):
        reset_directory(output / name, args.overwrite)
    export_package(model, cfg, output, args.sample_index, args.device, args.deployment_config)
    finish_stage(output, 'export', dict(config=source['config_sha256'], checkpoint=source['checkpoint_sha256'],
                 library_sources=source['library_sources'], sample_index=args.sample_index,
                 deployment_config=source['deployment_config_sha256']), owned,
                 versions('torch', 'mmcv', 'mmengine', 'mmdet', 'mmdet3d', 'spconv-cu117', 'Pillow', 'onnx'))
    if args.sample_dir and Path(args.sample_dir).resolve() != (output / 'sample').resolve():
        import shutil
        from _lib.common import register_sample
        destination = Path(args.sample_dir).resolve()
        require_overwrite([destination], args.overwrite)
        destination.mkdir(parents=True, exist_ok=True)
        for source_file in (output / 'sample').iterdir():
            if source_file.is_dir():
                shutil.copytree(source_file, destination / source_file.name, dirs_exist_ok=True)
            else:
                shutil.copy2(source_file, destination / source_file.name)
        register_sample(output, destination / 'sample.yaml')
        shutil.rmtree(output / 'sample')
    print('Export complete:', output, flush=True)


if __name__ == '__main__':
    run(main)
