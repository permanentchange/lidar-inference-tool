#!/usr/bin/env python3
"""Build the selected dense TensorRT engines from an exported BEVFusion package."""
import argparse
from pathlib import Path
from _lib.common import (begin_stage, digest, finish_stage, identifier, require_overwrite,
                         run, verify_stage)
from _lib.environment import bootstrap_native, device_index, select_cuda
from _lib.engine_build import build_artifacts, build_engines
from _lib.trtexec import find_trtexec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--trtexec', help='Path to TensorRT trtexec (otherwise TENSORRT_ROOT/bin or PATH)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    root = Path(args.model_dir).resolve()
    exported = verify_stage(root, 'export')
    owned = build_artifacts(root)
    require_overwrite(owned, args.overwrite)
    binary = find_trtexec(args.trtexec)
    bootstrap_native()
    device = select_cuda(args.device)
    begin_stage(root, 'build')
    build_engines(root, binary, device_index(args.device))
    from _lib.engine_build import networks
    finish_stage(root, 'build', dict(export=identifier(exported), head_precision='float16',
                 head_output_dtype='float32' if 'fusion_centerhead' in networks(root) else 'float16',
                 dense_precision='float16', binding_dtype='float16', tf32=True), owned,
                 dict(device=device, trtexec=dict(path=str(binary), sha256=digest(binary))))
    print('Engine build complete:', root, flush=True)


if __name__ == '__main__':
    run(main)
