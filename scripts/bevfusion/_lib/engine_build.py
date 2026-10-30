"""Build the two BEVFusion dense networks with the native FP16 TensorRT builder."""
import json
from .common import write
from .trtexec import build_command, execute

NETWORKS = ('camera.backbone', 'fusion_centerhead')

def networks(output):
    """Require the single supported BEVFusion graph contract."""
    import yaml
    node = yaml.safe_load((output / 'model.yaml').read_text())
    if node['model'].get('architecture') != 'centerpoint_bevfusion':
        raise RuntimeError('BEVFusion requires the current CenterHead package')
    return NETWORKS


def head_precision_layers(output):
    """Read exact final-logit Conv names emitted by the ONNX exporter."""
    import yaml
    model = yaml.safe_load((output / 'model.yaml').read_text())['model']
    layers = model.get('head_fp32_layers')
    if not isinstance(layers, list) or not layers or not all(isinstance(x, str) and x for x in layers):
        raise RuntimeError('CenterHead FP32 heatmap layer metadata missing; rerun export')
    return layers


def build_artifacts(output):
    """Return all build-owned engine, inspection, profile, policy, and log paths."""
    return [output / (name + suffix) for name in networks(output)
            for suffix in ('.engine', '.precision.json', '.layers.json', '.profile.json')] + [
                output / 'logs/build' / (name + '.log') for name in networks(output)]


def build_engines(output, binary, device=0):
    """Build fixed-shape networks with FP16 I/O and a 2 GiB workspace.

    The caller verifies export identity, CUDA selection, and overwrite permission.
    Use TensorRT's default internal precision selection, including TF32, without
    blanket normalization/Softmax constraints. Validate generated JSON and engine
    existence before replacing that network's artifacts. On failure preserve the
    log and existing completed engine; the caller keeps the build stage invalid.
    No training framework or TensorRT Python binding is required.
    """
    for name in networks(output):
        print('Building', name, 'FP16; log:', output / 'logs/build' / (name + '.log'), flush=True)
        destinations = [output / (name + suffix) for suffix in ('.engine', '.layers.json', '.profile.json')]
        temporary = [path.with_name(path.name + '.tmp') for path in destinations]
        command = build_command(binary, output / (name + '.onnx'), *temporary, device)
        output_dtype = 'float32' if name == 'fusion_centerhead' else 'float16'
        fp32_layers = []
        if output_dtype == 'float32':
            command[command.index('--outputIOFormats=fp16:chw')] = '--outputIOFormats=fp32:chw'
            fp32_layers = head_precision_layers(output)
            constraints = ','.join(layer + ':fp32' for layer in fp32_layers)
            command += ['--precisionConstraints=obey', '--layerPrecisions=' + constraints,
                        '--layerOutputTypes=' + constraints]
        try:
            for path in temporary:
                if path.exists():
                    path.unlink()
            execute(command, output / 'logs/build' / (name + '.log'))
            if not temporary[0].is_file() or not temporary[0].stat().st_size:
                raise RuntimeError('trtexec did not produce an engine: ' + name)
            for path in temporary[1:]:
                json.loads(path.read_text())
            if fp32_layers:
                inspection = json.loads(temporary[1].read_text())
                for requested in fp32_layers:
                    matches = [layer for layer in inspection.get('Layers', [])
                               if requested in layer.get('Name', '') and
                               'convolution' in layer.get('LayerType', '').lower()]
                    if not matches or any(not layer.get('Outputs') for layer in matches) or not all(
                            'FP32' in tensor.get('Format/Datatype', '')
                            for layer in matches for tensor in layer.get('Outputs', [])):
                        raise RuntimeError('TensorRT did not retain FP32 heatmap convolution: ' + requested)
            for source, target in zip(temporary, destinations):
                source.replace(target)
            write(output / (name + '.precision.json'), dict(
                fp16=True, binding_dtype='float16', binding_format='linear',
                output_dtype=output_dtype,
                tf32=True, fp32_layers=fp32_layers, workspace_mib=2048, command=command,
                note=('FP16 enabled; final heatmap Convs require FP32 to prevent logit overflow.'
                      if fp32_layers else 'FP16 enabled; TensorRT chooses internal precisions. No layer overrides.')))
        finally:
            for path in temporary:
                if path.exists():
                    path.unlink()
