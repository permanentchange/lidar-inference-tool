#!/usr/bin/env python3
"""Generate both CenterHead packages from one deployment configuration."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / 'configs/centerpoint_deployment.yaml'
MODEL_PATHS = {'runtime_backend', 'scn_onnx_path', 'head_engine_path'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            value.update(block)
    return value.hexdigest()


def identifier(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    text = (yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=100)
            if path.suffix == '.yaml' else json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.write_text(text)
    temporary.replace(path)


def load_reference(path):
    reference = yaml.safe_load(Path(path).read_text())
    if not isinstance(reference, dict) or not all(isinstance(reference.get(k), dict)
                                                for k in ('model', 'runtime', 'postprocess')):
        raise ValueError('Deployment configuration requires model/runtime/postprocess maps')
    model = reference['model']
    if model.get('decode_mode') != 'centerpoint_mmdeploy' or model.get('yaw_encoding') != 'sin_cos':
        raise ValueError('Deployment configuration must describe a sin_cos CenterHead')
    if model.get('task_num_classes') != [len(ids) for ids in model.get('task_label_ids', [])]:
        raise ValueError('Deployment task counts do not match task_label_ids')
    return reference


def apply_reference(package, reference, check_geometry=False):
    """Keep backend assets/cameras; replace all common fields with the reference."""
    result = copy.deepcopy(package)
    native = result['model']
    if native['runtime_backend'] not in ('centerpoint_voxel', 'bevfusion'):
        raise ValueError('Only CenterPoint Voxel and CenterHead BEVFusion are supported')
    common = {k: copy.deepcopy(v) for k, v in reference['model'].items() if k not in MODEL_PATHS}
    if check_geometry:
        # Runtime caps and postprocessing are deployment choices. Network geometry is not.
        for key in ('point_cloud_range', 'post_center_range', 'voxel_size', 'sparse_shape',
                    'num_point_features', 'out_size_factor', 'bbox_code_size', 'norm_bbox',
                    'has_velocity', 'label_names', 'task_label_ids'):
            if native.get(key) != common.get(key):
                raise ValueError('Deployment/model geometry mismatch: ' + key)
    model = {}
    for key, value in reference['model'].items():
        model[key] = native[key] if key in MODEL_PATHS else copy.deepcopy(value)
    model.update({k: v for k, v in native.items() if k not in model})
    result['version'] = reference['version']
    result['model'] = model
    result['runtime'] = copy.deepcopy(reference['runtime'])
    if native['runtime_backend'] == 'bevfusion':
        result['runtime']['bev_pool_backend'] = package['runtime'].get('bev_pool_backend', 'reference')
    result['postprocess'] = copy.deepcopy(reference['postprocess'])
    result['file_list'] = [model['scn_onnx_path']]
    if native['runtime_backend'] == 'bevfusion':
        result['file_list'].append(model['camera_engine_path'])
    result['file_list'] += [model['head_engine_path'], 'model.yaml']
    return result


def write_package(root, package, reference_path=DEFAULT_CONFIG, check_geometry=False):
    """Snapshot the source profile; deployed runtimes only read their local model.yaml."""
    root, reference_path = Path(root), Path(reference_path).resolve()
    reference = load_reference(reference_path)
    updated = apply_reference(package, reference, check_geometry)
    snapshot = root / 'deployment/reference.yaml'
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.resolve() != reference_path:
        shutil.copyfile(reference_path, snapshot)
    write(root / 'model.yaml', updated)
    record = dict(version=1, reference_path=str(reference_path), reference_sha256=digest(reference_path),
                  snapshot='deployment/reference.yaml', model_config_sha256=digest(root / 'model.yaml'),
                  model_overrides={k: package['model'].get(k) for k in reference['model']
                                   if k not in MODEL_PATHS and package['model'].get(k) != reference['model'][k]},
                  runtime_overrides={k: package.get('runtime', {}).get(k) for k in reference['runtime']
                                     if package.get('runtime', {}).get(k) != reference['runtime'][k]},
                  applied_at=datetime.now(timezone.utc).isoformat())
    write(root / 'deployment/configuration.json', record)
    return updated


def prepare_synchronization(root):
    """Verify immutable artifacts before replacing regenerable configuration records."""
    root = Path(root).resolve()
    package = yaml.safe_load((root / 'model.yaml').read_text())
    provenance_path = root / 'provenance.json'
    data = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    if data:
        sys.path.insert(0, str(REPO / 'scripts/bevfusion'))
        from _lib.common import verify_stage
        for stage in ('export', 'build'):
            verify_stage(root, stage, refresh_deployment_config=True)
    return root, package, data


def synchronize(root, reference_path, prepared=None):
    """Migrate config records without exporting a network or building an engine."""
    root, package, data = prepared if prepared is not None else prepare_synchronization(root)
    provenance_path = root / 'provenance.json'
    protected = {str(p): digest(p) for p in root.rglob('*')
                 if p.is_file() and ('.onnx' in p.name or '.engine' in p.name)}
    previous = copy.deepcopy(data.get('stages', {})) if data else {}
    previous_config = copy.deepcopy(package)
    write_package(root, package, reference_path)
    history = dict(previous_model=previous_config, previous_stage_identifiers={
        k: identifier(v) for k, v in previous.items()}, models_rebuilt=False,
        reference_sha256=digest(reference_path), changed_at=datetime.now(timezone.utc).isoformat())
    write(root / 'deployment/previous_configuration.json', history)
    # Retain old reports but explicitly invalidate their applicability.
    for directory in ('validation', 'evaluation', 'pooling_benchmark'):
        path = root / directory
        if path.is_dir():
            write(path / 'STALE.json', dict(reason='Deployment configuration and postprocessing changed',
                                           reference_sha256=digest(reference_path)))
    if data:
        # Logs are historical diagnostics, not required inputs for an existing engine.
        # Preserve their recorded hashes without inventing replacement log files.
        for stage in data['stages'].values():
            for name in list(stage.get('artifacts', {})):
                if name.startswith('logs/build/') and name.endswith('.log'):
                    stage.setdefault('diagnostic_artifacts', {})[name] = stage['artifacts'].pop(name)
        exported = data['stages']['export']
        for name in ('model.yaml', 'deployment/reference.yaml', 'deployment/configuration.json'):
            exported['artifacts'][name] = digest(root / name)
        exported['inputs']['deployment_config'] = digest(reference_path)
        for name, stage in data['stages'].items():
            if name != 'export' and 'model.yaml' in stage.get('artifacts', {}):
                stage['artifacts']['model.yaml'] = digest(root / 'model.yaml')
        data['stages']['build']['inputs']['export'] = identifier(exported)
        for stale in ('validate', 'evaluate'):
            data['stages'].pop(stale, None)
        data.setdefault('migrations', []).append(dict(kind='shared_centerhead_deployment_config', **history))
        write(provenance_path, data)
    for name, expected in protected.items():
        if digest(name) != expected:
            raise RuntimeError('Protected model changed: ' + name)
    print('Synchronized configuration:', root, '(model files unchanged)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', action='append', required=True)
    parser.add_argument('--deployment-config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        args.deployment_config = args.deployment_config.resolve()
        reference = load_reference(args.deployment_config)
        prepared = [prepare_synchronization(root) for root in args.model_dir]
        # Fail before writing any model configuration when a requested package is invalid.
        for _, package, _ in prepared:
            apply_reference(package, reference)
        for state in prepared:
            synchronize(state[0], args.deployment_config, prepared=state)
    except (OSError, ValueError, RuntimeError, KeyError, yaml.YAMLError) as error:
        parser.exit(1, 'Error: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
