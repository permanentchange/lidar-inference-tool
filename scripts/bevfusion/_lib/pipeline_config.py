"""Validate experiment YAML and normalize a fixed-order argument list for safe Bash consumption."""
import argparse
from pathlib import Path
from .common import nonnegative, positive, run
from .environment import device_index


def load_config(path):
    """Read an experiment and return nine normalized string arguments.

    The order is mmdet3d_root, config, checkpoint, model_dir, device, sample_index,
    validation_frames, sample_dir, and deployment_config. The first four fields are required. Resolve the library
    root and output relative to the YAML directory, and config/checkpoint relative
    to the library root; absolute paths are resolved directly. Optional defaults
    are cuda:0, 0, and 20. Counts must be integers rather than bool, with a
    nonnegative sample index and a positive frame count. Reject unknown fields;
    do not create outputs or check that source files exist.
    """
    import yaml
    path = Path(path).resolve()
    try:
        with path.open() as stream:
            value = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError('Invalid experiment YAML: ' + str(error)) from error
    if not isinstance(value, dict):
        raise ValueError('Experiment YAML must be a mapping')
    required = {'mmdet3d_root', 'config', 'checkpoint', 'model_dir'}
    allowed = required | {'device', 'sample_index', 'validation_frames', 'sample_dir', 'deployment_config'}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError('Unknown experiment fields: ' + ', '.join(sorted(map(str, unknown))))
    for name in required:
        if not isinstance(value.get(name), str) or not value[name]:
            raise ValueError('Required experiment path: ' + name)
    root = (path.parent / value['mmdet3d_root']).resolve()
    output = (path.parent / value['model_dir']).resolve()
    device = value.get('device', 'cuda:0')
    if not isinstance(device, str):
        raise ValueError('device must be cuda:N')
    device_index(device)
    for key, default in [('sample_index', 0), ('validation_frames', 20)]:
        count = value.get(key, default)
        if not isinstance(count, int) or isinstance(count, bool):
            raise ValueError(key + ' must be an integer')
    return [str(root), str((root / value['config']).resolve()),
            str((root / value['checkpoint']).resolve()), str(output), device,
            str(nonnegative(value.get('sample_index', 0))),
            str(positive(value.get('validation_frames', 20))),
            str((path.parent / value['sample_dir']).resolve()) if value.get('sample_dir') else str(output / 'sample'),
            str((path.parent / value['deployment_config']).resolve()) if value.get('deployment_config')
            else str(Path(__file__).resolve().parents[2] / 'configs/centerpoint_deployment.yaml')]


def main():
    """Write normalized --config arguments to --output, separated by NUL bytes.

    Reject embedded NUL characters. run_pipeline.sh reads the file with mapfile;
    the output is data, not shell code, so spaces, quotes, and command substitution
    characters are not evaluated by the shell.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    values = load_config(args.config)
    if any('\0' in item for item in values):
        raise ValueError('NUL is not valid in a path')
    Path(args.output).write_bytes(b'\0'.join(item.encode() for item in values) + b'\0')


if __name__ == '__main__':
    run(main)
