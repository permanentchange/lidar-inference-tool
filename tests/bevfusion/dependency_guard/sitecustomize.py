"""Opt-in subprocess dependency audit for real deployment integration tests."""
import os
import sys

root = os.environ.get('BEVFUSION_TRAIN_ROOT', '').rstrip('/')
blocked = [root + suffix for suffix in
           ('/projects/BEVFusion/tools/', '/projects/CenterPoint/tools/', '/tools/')]


def guard(event, args):
    if event == 'open' and isinstance(args[0], str) and root:
        path = os.path.abspath(args[0])
        if any(path.startswith(prefix) for prefix in blocked):
            raise RuntimeError('Forbidden external deployment tool access: ' + path)
    if event != 'import':
        return
    entry, package = os.path.basename(sys.argv[0]), args[0].split('.')[0]
    if entry == 'export.py' and package == 'tensorrt':
        raise RuntimeError('Forbidden export-stage TensorRT import')
    if entry == 'build_engines.py' and package in {'torch', 'tensorrt', 'mmdet3d', 'mmengine', 'mmcv', 'mmdet'}:
        raise RuntimeError('Forbidden build-stage framework import: ' + args[0])


sys.addaudithook(guard)
