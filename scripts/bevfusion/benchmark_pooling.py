#!/usr/bin/env python3
"""Compare reference/tiled pooling and write a measured runtime override."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

from _lib.common import (REPO_ROOT, digest, identifier, positive, read,
                         reset_directory, run, verify_stage, write)
from _lib.common import sample_manifest
from _lib.environment import bootstrap_native, device_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--input', help='Frame manifest; defaults to exported sample')
    parser.add_argument('--tool', default=str(REPO_ROOT / 'build/perf_debug_tool'))
    parser.add_argument('--infer-tool', default=str(REPO_ROOT / 'build/infer_debug_tool'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--repeats', type=positive, default=3)
    parser.add_argument('--warmup', type=positive, default=10)
    parser.add_argument('--profile-iters', type=positive, default=50)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    root = Path(args.model_dir).resolve()
    tool = Path(args.tool).resolve()
    infer_tool = Path(args.infer_tool).resolve()
    source = Path(args.input).resolve() if args.input else sample_manifest(root)
    exported, built = verify_stage(root, 'export'), verify_stage(root, 'build')
    bootstrap_native()
    import numpy as np
    import yaml
    config = yaml.safe_load((root / 'model.yaml').read_text())
    if config['model']['camera']['feature_channels'] != 80:
        raise RuntimeError('NVIDIA tiled pooling requires 80 feature channels')
    work = root / 'pooling_benchmark'
    reset_directory(work, args.overwrite)
    env = dict(os.environ)
    visible = env.get('CUDA_VISIBLE_DEVICES')
    index = device_index(args.device)
    env['CUDA_VISIBLE_DEVICES'] = visible.split(',')[index] if visible else str(index)
    runs = {name: [] for name in ('reference', 'nvidia_pack10')}
    for backend in runs:
        write(work / (backend + '.yaml'), dict(runtime=dict(bev_pool_backend=backend)))
        cmd = [str(infer_tool), '--model-dir', str(root), '--input', str(source),
               '--dump-dir', str(work / ('parity_' + backend)), '--override-config', str(work / (backend + '.yaml')),
               '--warmup', '0', '--profile-iters', '1', '--dump-internal', '--no-dump-decoded']
        with (work / ('parity_' + backend + '.log')).open('w') as log:
            subprocess.run(cmd, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    # Alternate policies to reduce order/temperature bias.
    for repeat in range(args.repeats):
        order = list(runs) if repeat % 2 == 0 else list(reversed(runs))
        for backend in order:
            target = work / f'{backend}_{repeat}'
            cmd = [str(tool), '--model-dir', str(root), '--input', str(source),
                   '--dump-dir', str(target), '--override-config', str(work / (backend + '.yaml')),
                   '--warmup', str(args.warmup), '--profile-iters', str(args.profile_iters)]
            with (work / f'{backend}_{repeat}.log').open('w') as log:
                subprocess.run(cmd, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
            meta, stages = read(target / 'metadata.json'), read(target / 'timings.json')
            pool = next(stage for stage in stages['stage_profile']['stages'] if stage['name'] == 'bev_pool')
            runs[backend].append(dict(pool_gpu_ms=pool['gpu_ms']['mean'],
                total_mean_ms=stages['total_infer_ms']['mean'], total_p50_ms=stages['total_infer_ms']['p50'],
                total_p90_ms=stages['total_infer_ms']['p90']))
            print(backend, repeat, runs[backend][-1], flush=True)
    a, b = work / 'parity_reference', work / 'parity_nvidia_pack10'
    for tensor in sorted((a / 'internal').glob('*.npy')):
        if not np.array_equal(np.load(tensor), np.load(b / 'internal' / tensor.name)):
            raise AssertionError('Pooling policies differ: ' + tensor.name)
    if read(a / 'result.json') != read(b / 'result.json'):
        raise AssertionError('Pooling policies produce different detections')
    medians = {backend: {key: statistics.median(row[key] for row in rows)
                        for key in rows[0]} for backend, rows in runs.items()}
    faster = (medians['nvidia_pack10']['pool_gpu_ms'] < medians['reference']['pool_gpu_ms'] and
              medians['nvidia_pack10']['total_p50_ms'] <= medians['reference']['total_p50_ms'])
    selected = 'nvidia_pack10' if faster else 'reference'
    override = work / 'selected_backend.yaml'
    write(override, dict(runtime=dict(bev_pool_backend=selected)))
    write(work / 'report.json', dict(export=identifier(exported), build=identifier(built),
        tool_sha256=digest(tool), infer_tool_sha256=digest(infer_tool),
        manifest=str(source), manifest_sha256=digest(source),
        raw_preprocessing=True, warmup=args.warmup, profile_iters=args.profile_iters,
        repeats=args.repeats, runs=runs, medians=medians, tensors_and_detections_exact=True,
        selected_backend=selected, selected_override=str(override)))
    print('Selected', selected, '; use --override-config', override, flush=True)


if __name__ == '__main__':
    run(main)
