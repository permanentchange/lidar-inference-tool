"""Numerical checks for a built deployment package, separate from full dataset metric evaluation."""
import json
import shutil
from pathlib import Path
import numpy as np
from .common import read, write


def selected_batches(loader, indices):
    """Collate requested map-style samples without decoding unselected frames.

    build_dataloader supplies a single-sample, ordered loader with deterministic
    test transforms. Preserve the requested dataset indices and collate function.
    """
    for index in indices:
        yield index, loader.collate_fn([loader.dataset[index]])


def check_finite_tensors(directory):
    """Reject NaN/Inf in any captured NPY tensor and return the checked file count."""
    paths = sorted(directory.glob('*.npy'))
    if not paths:
        raise AssertionError('Deployment tensor dump is empty: ' + str(directory))
    for path in paths:
        if not np.isfinite(np.load(path)).all():
            raise AssertionError('Nonfinite deployment tensor: ' + str(path))
    return len(paths)


def validate(args, cfg, model, output):
    """Check preprocessing, same-engine execution, independent operators/decode, and selected frames.

    Args:
        args: CLI options containing tool, device, and frames.
        cfg: Export source configuration used for an ordered test loader,
            falling back to its validation loader.
        model: Original model in eval mode with the corresponding weights.
        output: Model directory Path. The entrypoint verifies artifacts and
            resets validation/ before calling this function.

    Check the exported sample's raw/preprocessed input parity and independent
    operators, then select at most frames samples. Include both endpoints when
    selecting at least two frames and use a fixed seed for the remaining samples.
    Write tensors, all candidate differences, and report.json under validation/;
    return no value. Operator/tolerance failures raise AssertionError. Candidate
    differences are reported without a match-rate gate. Do not run the full
    val_evaluator or register stage completion.
    """
    from .model import build_dataloader
    from .reference import fixture, original_reference
    from .runtime import run_tool
    from .parity import (metrics, cpp_prediction, compare_predictions, same_engine,
                         independent_ops, independent_decode, clean_json, binding_types)

    package = read(output / 'model.yaml')
    work = output / 'validation'
    work.mkdir(exist_ok=True)
    raw, pre = work / 'raw', work / 'preprocessed'
    from .common import sample_manifest
    sample = sample_manifest(output)
    sample_node = read(sample)
    dtypes = binding_types(output)
    run_tool(args, output, sample, raw, raw=True, internal=True)
    run_tool(args, output, sample, pre, internal=True)
    check_finite_tensors(raw / 'internal')
    check_finite_tensors(pre / 'internal')
    a, b = np.load(raw / 'internal/images.npy'), np.load(sample.parent / sample_node['preprocessed_images'])
    expected_images = b.astype(dtypes['camera.backbone']['img']).astype(np.float32)
    image_error = metrics(expected_images, a)
    if image_error['max_abs'] > 1e-5:
        raise AssertionError('Raw JPEG/Pillow preprocessing parity failed: ' + str(image_error))
    for p in sorted((raw / 'internal').glob('*.npy')):
        if not np.array_equal(np.load(p), np.load(pre / 'internal' / p.name)):
            raise AssertionError('Raw/preprocessed inference differs: ' + p.name)
    report = dict(policy="model.yaml deployment voxel limits, NMS and output label mapping",
                  native_head_policy="saved separately under fixtures; not used as the deployment acceptance reference",
                  raw_preprocessing=image_error, raw_preprocessed_exact=True, binding_types=dtypes,
                  same_engine=same_engine(output, raw / 'internal'),
                  independent_ops=independent_ops(model, sample, raw / 'internal',
                      dtypes['fusion_centerhead']['camera_bev']),
                  independent_decode=independent_decode(model, raw / 'internal', raw / 'result.json', package), frames=[])
    loader = build_dataloader(cfg)
    total = len(loader.dataset)
    if total == 0:
        raise RuntimeError('Test dataset is empty')
    count = min(args.frames, total)
    chosen = [0] if count == 1 else [0, total - 1]
    if hasattr(loader.dataset, 'datasets'):
        # Include each configured disabled-camera policy before random sampling.
        chosen, offset, policies = [0], 0, set()
        for child in loader.dataset.datasets:
            transforms = getattr(getattr(child, 'pipeline', None), 'transforms', [])
            disabled = tuple(getattr(transforms[0], 'disabled_cameras', ())) if transforms else ()
            # Distinct dataset roots exercise their independent calibration.
            key = (str(getattr(child, 'data_root', '')), disabled)
            if disabled and key not in policies and offset not in chosen and len(chosen) < count:
                chosen.append(offset)
                policies.add(key)
            offset += len(child)
        if len(chosen) < count and total - 1 not in chosen:
            chosen.append(total - 1)
    if len(chosen) < count:
        remaining = np.setdiff1d(np.arange(total), chosen)
        chosen += np.random.RandomState(0).choice(remaining, count - len(chosen), replace=False).tolist()
    chosen = sorted(chosen)
    manifests = work / 'manifests'
    if manifests.exists():
        shutil.rmtree(manifests)
    manifests.mkdir(exist_ok=True)
    refs = {}
    for index, batch in selected_batches(loader, chosen):
        folder = work / 'fixtures' / f'{index:06d}'
        manifest = fixture(cfg, model, batch, folder)
        # Manifest directory is flat; its file contains absolute paths.
        node = read(manifest)
        for key in ('point_cloud', 'preprocessed_images'):
            node[key] = str((folder / node[key]).resolve())
        write(manifests / f'{index:06d}.yaml', node)
        refs[index], _ = original_reference(model, batch, folder / 'original', deployment=package)
        print('Captured original test frame', index, flush=True)
    target = work / 'batch'
    run_tool(args, output, manifests, target, internal=True)
    for sequence_index, index in enumerate(chosen):
        folder = work / 'fixtures' / f'{index:06d}'
        cpp = target / 'frames' / f'{sequence_index:06d}_{index:06d}'
        finite_count = check_finite_tensors(cpp / 'internal')
        raw_masked_check = None
        manifest_node = read(folder / 'sample.yaml')
        if any(not camera.get('enabled', True) for camera in manifest_node['cameras']):
            masked_raw = folder / 'raw_input_check'
            run_tool(args, output, folder / 'sample.yaml', masked_raw, raw=True, internal=True)
            check_finite_tensors(masked_raw / 'internal')
            for tensor in sorted((cpp / 'internal').glob('*.npy')):
                if not np.array_equal(np.load(tensor), np.load(masked_raw / 'internal' / tensor.name)):
                    raise AssertionError('Masked raw/preprocessed parity failed: ' + tensor.name)
            raw_masked_check = dict(raw_preprocessed_exact=True)
        tensor_report = {}
        for p in (folder / 'original').glob('*.npy'):
            actual = cpp / 'internal' / p.name
            if actual.exists():
                tensor_report[p.stem] = metrics(np.load(p), np.load(actual))
        diff = compare_predictions(refs[index], cpp_prediction(cpp / 'result.json'))
        write(work / f'candidates_{index:06d}.json', diff)
        report['frames'].append(dict(dataset_index=index, tensors=tensor_report,
                                     masked_raw_input=raw_masked_check,
                                     independent_decode=independent_decode(model, cpp / 'internal', cpp / 'result.json', package),
                                     finite_tensor_files=finite_count,
                                     original_count=len(refs[index]['boxes']), deployment_count=len(diff['deployment']['boxes']),
                                     matched_within_0_2m=diff.get('matched_within_0_2m', 0),
                                     depth_collisions=json.loads((cpp / 'metadata.json').read_text())['counts']['depth_collisions']))
    write(work / 'report.json', clean_json(report))
    print('Validated raw/preprocessed, independent operators, same engines,', len(chosen), 'test frames', flush=True)
