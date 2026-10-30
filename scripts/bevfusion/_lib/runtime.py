"""Adapt C++ inference subprocesses for validation and evaluation without exporting or building models."""
import os
import subprocess
from pathlib import Path

def run_tool(args, output, source, target, raw=False, internal=False, override_config=None):
    """Run native inference on the selected GPU and redirect stdout/stderr to target.log.

    Args:
        args: Options containing tool (binary path) and device (visible index).
        output: Deployment model directory.
        source: A sample manifest or a flat directory of manifests.
        target: Inference output directory whose parent must already exist.
        raw: Use JPEG images if true, otherwise preprocessed NPY tensors.
        internal: Whether to dump intermediate tensors; disabled by default.

    Use zero warmups, one measured iteration, and no decoded dumps. Map device
    through the parent's visible device list to the child's sole visible GPU
    without changing the parent environment. A nonzero tool exit raises
    subprocess.CalledProcessError so the enclosing stage stops.
    """
    cmd = [str(Path(args.tool).resolve()), '--model-dir', str(output), '--input', str(source), '--dump-dir', str(target),
           '--image-input', 'raw' if raw else 'preprocessed', '--warmup', '0', '--profile-iters', '1',
           '--no-dump-decoded']
    if override_config is not None:
        cmd += ['--override-config', str(override_config)]
    cmd += ['--dump-internal' if internal else '--no-dump-internal']
    with open(str(target) + '.log', 'w') as log:
        env = dict(os.environ)
        device = int(args.device.split(':', 1)[1]) if ':' in args.device else 0
        visible = env.get('CUDA_VISIBLE_DEVICES')
        env['CUDA_VISIBLE_DEVICES'] = visible.split(',')[device] if visible else str(device)
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
