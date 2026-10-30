"""Locate and execute TensorRT's native builder for the BEVFusion deployment stage."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess


def find_trtexec(explicit=None):
    """Resolve an executable from an explicit path, TENSORRT_ROOT, or PATH.

    Reject an invalid explicit path instead of falling back to another install.
    This function neither imports TensorRT nor starts CUDA.
    """
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file() or not os.access(str(candidate), os.X_OK):
            raise RuntimeError('trtexec is not executable: ' + str(candidate))
        return candidate
    root = os.environ.get('TENSORRT_ROOT')
    candidates = [Path(root) / 'bin/trtexec'] if root else []
    found = shutil.which('trtexec')
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return candidate.resolve()
    raise RuntimeError('Cannot find trtexec; use --trtexec PATH, TENSORRT_ROOT, or PATH')


def build_command(binary, onnx, engine, layers, profile, device):
    """Return a shell-free FP16 command with linear FP16 bindings for every I/O.

    TensorRT 8.6 broadcasts a single I/O format to all bindings. Internal layer
    precision and TF32 follow TensorRT defaults. Synthetic-input profiling runs
    separately from the benchmark and does not constitute accuracy validation.
    """
    return [str(binary), '--onnx=' + str(onnx), '--saveEngine=' + str(engine),
            '--device=' + str(device), '--fp16', '--inputIOFormats=fp16:chw',
            '--outputIOFormats=fp16:chw', '--memPoolSize=workspace:2048',
            '--verbose', '--profilingVerbosity=detailed', '--dumpLayerInfo',
            '--dumpProfile', '--separateProfileRun',
            '--exportLayerInfo=' + str(layers), '--exportProfile=' + str(profile)]


def execute(command, log):
    """Run an argv list, preserve combined output, and report failures by log path.

    The shell-quoted command in the log is diagnostic text only. Execution never
    invokes a shell, so paths and arguments remain literal.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        stream.write('$ ' + ' '.join(shlex.quote(arg) for arg in command) + '\n')
        stream.flush()
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('trtexec exited with status %d; see %s' % (result.returncode, log))
