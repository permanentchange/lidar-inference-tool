"""Initialize native library search paths, CUDA devices, and the model library environment."""
import ctypes
import ctypes.util
import importlib.util
import os
from pathlib import Path
import platform
import re
import sys


def bootstrap_native():
    """Extend native library search paths and re-execute Python with the same arguments.

    Locate libraries under TENSORRT_ROOT, CUDA_HOME, and the active Python
    environment. An installed torch/lib may supply cuDNN without importing
    PyTorch. Call before loading native libraries. The first successful call
    replaces the process through execve and never returns; the restarted process
    uses _BEVFUSION_NATIVE_READY to prevent a re-execution loop.
    """
    if os.environ.get('_BEVFUSION_NATIVE_READY') == '1':
        return
    paths = []
    trt_root = os.environ.get('TENSORRT_ROOT')
    if trt_root:
        base = Path(trt_root)
        paths += [base / 'lib', base / 'lib64', base / ('targets/' + platform.machine() + '-linux-gnu/lib')]
    cuda_root = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
    paths += [cuda_root / 'lib64', Path(sys.prefix) / 'lib']
    # cuDNN may be supplied by the selected Python environment. Inspect the
    # location without importing torch or making it a build-stage dependency.
    spec = importlib.util.find_spec('torch')
    if spec and spec.origin:
        paths.append(Path(spec.origin).parent / 'lib')
    env = dict(os.environ, _BEVFUSION_NATIVE_READY='1')
    existing = env.get('LD_LIBRARY_PATH', '').split(':')
    env['LD_LIBRARY_PATH'] = ':'.join(dict.fromkeys([str(p) for p in paths if p.is_dir()] + [p for p in existing if p]))
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


def device_index(value):
    """Parse cuda or cuda:N as a visible device index; raise ValueError for other forms."""
    match = re.fullmatch(r'cuda(?::([0-9]+))?', value)
    if not match:
        raise ValueError('Expected CUDA device cuda:N')
    return int(match.group(1) or 0)


def select_cuda(device):
    """Select a CUDA device and initialize its primary context through ctypes.

    Args:
        device: cuda or cuda:N, indexed relative to CUDA_VISIBLE_DEVICES.

    Returns:
        A dictionary containing the PCI bus identifier, compute capability, and
        CUDA runtime/driver versions. This path needs neither PyTorch nor PyCUDA;
        a failed CUDA call raises RuntimeError.
    """
    index = device_index(device)
    root = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
    path = root / 'lib64/libcudart.so'
    library = ctypes.CDLL(str(path) if path.is_file() else (ctypes.util.find_library('cudart') or 'libcudart.so'))
    library.cudaGetErrorString.restype = ctypes.c_char_p
    def check(status):
        """Translate a nonzero CUDA status code into a runtime error message."""
        if status:
            raise RuntimeError(library.cudaGetErrorString(status).decode())
    check(library.cudaSetDevice(index))
    library.cudaFree.argtypes = [ctypes.c_void_p]
    check(library.cudaFree(None))  # initialize the device's primary context
    major, minor, runtime, driver = (ctypes.c_int() for _ in range(4))
    check(library.cudaDeviceGetAttribute(ctypes.byref(major), 75, index))
    check(library.cudaDeviceGetAttribute(ctypes.byref(minor), 76, index))
    check(library.cudaRuntimeGetVersion(ctypes.byref(runtime)))
    check(library.cudaDriverGetVersion(ctypes.byref(driver)))
    pci = ctypes.create_string_buffer(32)
    check(library.cudaDeviceGetPCIBusId(pci, len(pci), index))
    return dict(pci_bus_id=pci.value.decode(), compute_capability=[major.value, minor.value],
                cuda_runtime=runtime.value, cuda_driver=driver.value)


def training_context(root, device):
    """Configure the working directory, device, and floating-point options for model libraries.

    root must contain mmdet3d. Add it to sys.path and change the working directory
    so relative dataset paths resolve against the training project. Select the
    requested CUDA device, disable PyTorch TF32, and seed Torch/NumPy with zero.
    These process settings are not restored and do not eliminate nondeterministic
    pixel collisions in the original model's depth projection.
    """
    root = Path(root).resolve()
    if not (root / 'mmdet3d').is_dir():
        raise RuntimeError('MMDetection3D library root missing: ' + str(root))
    device_index(device)
    sys.path.insert(0, str(root))
    os.chdir(root)
    import torch
    import numpy as np
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required')
    torch.cuda.set_device(torch.device(device))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    np.random.seed(0)


def source_context(root, device):
    """Reload the recorded export source and return (cfg, model_in_eval_mode).

    root is the deployment model directory, not the training checkout. Verify
    config/checkpoint digests, configure training_context, then verify model
    library sources and strictly load the weights. Changed sources raise
    RuntimeError to prevent mixing references from different model versions.
    """
    from .common import metadata, digest
    source = metadata(root)['source']
    for name in ('config', 'checkpoint'):
        if digest(source[name]) != source[name + '_sha256']:
            raise RuntimeError('Export source changed: ' + source[name])
    training_context(source['mmdet3d_root'], device)
    from .model import library_sources, load_model
    if library_sources(Path(source['mmdet3d_root'])) != source['library_sources']:
        raise RuntimeError('MMDetection3D model library sources changed since export')
    cfg, model = load_model(source['config'], source['checkpoint'], device)
    from .common import read
    from .deployment_reference import configure_voxelizer
    configure_voxelizer(model, read(Path(root) / 'model.yaml'))
    return cfg, model
