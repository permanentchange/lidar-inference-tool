# SPDX-License-Identifier: Apache-2.0
# Portions adapted from MMDetection3D, copyright OpenMMLab contributors.
# See licenses/Apache-2.0.txt and NOTICE for source attribution.
"""Trace selected operators to build a libspconv-compatible FP16 sparse ONNX graph."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
import torch
from torch import nn

try:
    import cumm.tensorview as tv
    import spconv.pytorch as spconv
    from spconv.core import ConvAlgo
except ImportError as exc:  # pragma: no cover - validated by CLI.
    raise RuntimeError(
        'SCN export requires spconv.pytorch and cumm in the active env') from exc

from mmdet3d.models.layers.sparse_block import SparseBasicBlock


class SparseScnTrace:
    """Collect object identities, ONNX nodes, and constants for one sparse export.

    enabled controls recording; obj_to_tensor_id maps runtime tensors to graph
    edges; nodes/initializers hold graph contents; layer_counters names each
    operator family. keep_alive retains intermediates to prevent Python object
    id reuse from connecting unrelated graph edges. The module uses a global
    _TRACE instance and patches process-wide methods during tracing. Concurrent
    or nested exports in the same process are unsupported.
    """

    def __init__(self) -> None:
        """Initialize an empty, disabled tracer without installing operator patches."""
        self.enabled = False
        self.keep_alive: List[object] = []
        self.obj_to_tensor_id: Dict[int, str] = {}
        self.next_tensor_index = 0
        self.nodes: List[onnx.NodeProto] = []
        self.initializers: List[onnx.TensorProto] = []
        self.layer_counters: Dict[str, int] = {}

    def reset(self) -> None:
        """Disable tracing and clear nodes, constants, references, and counters to release intermediates."""
        self.enabled = False
        self.keep_alive = []
        self.obj_to_tensor_id = {}
        self.next_tensor_index = 0
        self.nodes = []
        self.initializers = []
        self.layer_counters = {}

    def next_layer(self, prefix: str) -> int:
        """Return the next zero-based index for a node prefix and increment its counter."""
        value = self.layer_counters.get(prefix, 0)
        self.layer_counters[prefix] = value + 1
        return value

    def object_id(self, obj: object) -> int:
        """Return the graph identity of an object, using features identity for sparse tensors."""
        if isinstance(obj, spconv.SparseConvTensor):
            return id(obj.features)
        return id(obj)

    def register_tensor(self, obj: object) -> str:
        """Allocate a new string edge identifier, store its object mapping, and return it."""
        tensor_id = str(self.next_tensor_index)
        self.next_tensor_index += 1
        self.obj_to_tensor_id[self.object_id(obj)] = tensor_id
        return tensor_id

    def tensor_id(self, obj: object) -> str:
        """Look up a registered edge identifier; raise RuntimeError for an untraced input."""
        obj_id = self.object_id(obj)
        if obj_id not in self.obj_to_tensor_id:
            raise RuntimeError(
                f'Missing traced tensor id for {type(obj).__name__}. '
                'An unsupported op was probably executed while tracing SCN.')
        return self.obj_to_tensor_id[obj_id]

    def append_initializer(self, value: torch.Tensor, name: str) -> str:
        """Append a CPU FP16 raw-byte ONNX initializer and return its supplied name."""
        array = value.detach().cpu().numpy().astype(np.float16)
        self.initializers.append(
            helper.make_tensor(
                name=name,
                data_type=TensorProto.FLOAT16,
                dims=list(array.shape),
                vals=array.tobytes(),
                raw=True,
            ))
        return name


_TRACE = SparseScnTrace()
_SPCONV_DOMAIN = 'ai.onnx.contrib'


def _as_list(value: object) -> List[int]:
    """Convert spatial parameters to integer lists, repeating scalar values along three axes."""
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)] * 3


def _activation_name(conv: nn.Module) -> str:
    """Map a fused spconv activation to its parser name; default or unknown values become the string None."""
    activation = getattr(conv, 'act_type', None)
    if activation == tv.gemm.Activation.ReLU:
        return 'ReLU'
    if activation in (None, tv.gemm.Activation.None_):
        return 'None'
    if activation == tv.gemm.Activation.Sigmoid:
        return 'Sigmoid'
    if activation == tv.gemm.Activation.LeakyReLU:
        return 'LeakyReLU'
    return 'None'


def _conv_algo_name(conv: nn.Module) -> str:
    """Map a spconv algorithm enum to a graph attribute name, or return an empty string if unknown."""
    algo = getattr(conv, 'algo', None)
    if algo == ConvAlgo.MaskImplicitGemm:
        return 'MaskImplicitGemm'
    if algo == ConvAlgo.MaskSplitImplicitGemm:
        return 'MaskSplitImplicitGemm'
    if algo == ConvAlgo.Native:
        return 'Native'
    return ''


def _fuse_bn_weights(conv_w: torch.Tensor, conv_b: torch.Tensor,
                     bn: nn.BatchNorm1d) -> Tuple[torch.nn.Parameter,
                                                  torch.nn.Parameter]:
    """Compute fused inference weights and return new (weight, bias) parameters.

    conv_w follows spconv layout [out, kernel..., in]. Permute internally to scale
    each output channel, then restore the original layout. Handle missing
    convolution bias or BN affine parameters without replacing supplied module
    parameters in place. The caller ensures BN is in eval mode.
    """
    ndim = conv_w.ndim - 2
    permute_to_oik = [0, ndim + 1] + [i + 1 for i in range(ndim)]
    conv_w_oik = conv_w.permute(*permute_to_oik)
    if conv_b is None:
        conv_b = torch.zeros_like(bn.running_mean)
    if bn.weight is None:
        bn_w = torch.ones_like(bn.running_mean)
    else:
        bn_w = bn.weight
    if bn.bias is None:
        bn_b = torch.zeros_like(bn.running_mean)
    else:
        bn_b = bn.bias
    bn_var_rsqrt = torch.rsqrt(bn.running_var + bn.eps)
    scale = (bn_w * bn_var_rsqrt).reshape([-1] +
                                          [1] * (conv_w_oik.ndim - 1))
    conv_w_oik = conv_w_oik * scale
    conv_b = (conv_b - bn.running_mean) * bn_var_rsqrt * bn_w + bn_b
    permute_to_oki = [0] + [i + 2 for i in range(ndim)] + [1]
    return (torch.nn.Parameter(conv_w_oik.permute(*permute_to_oki).contiguous()),
            torch.nn.Parameter(conv_b.contiguous()))


def _fuse_bn_into_conv(conv: nn.Module, bn: nn.BatchNorm1d) -> None:
    """Require eval mode and replace convolution weights/bias in place; use only on an export copy."""
    if conv.training or bn.training:
        raise RuntimeError('BatchNorm fusion requires eval mode')
    conv.weight, conv.bias = _fuse_bn_weights(conv.weight, conv.bias, bn)


def _sparse_basic_block_forward(self, x):
    """Run a BN-fused residual block, preserving the final feature addition and ReLU."""
    identity = x
    out = self.conv1(x)
    out = self.conv2(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out.replace_feature(out.features + identity.features)
    out = out.replace_feature(self.relu(out.features))
    return out


def _fuse_sparse_basic_block(block: SparseBasicBlock) -> None:
    """Fuse both convolution/BN pairs in place and bind a forward method for the fused residual block."""
    norm1 = getattr(block, 'norm1', getattr(block, 'bn1', None))
    norm2 = getattr(block, 'norm2', getattr(block, 'bn2', None))
    if norm1 is None or norm2 is None:
        raise RuntimeError(f'Unsupported SparseBasicBlock norms: {block}')
    _fuse_bn_into_conv(block.conv1, norm1)
    _fuse_bn_into_conv(block.conv2, norm2)
    block.conv1.act_type = tv.gemm.Activation.ReLU
    block.conv2.act_type = tv.gemm.Activation.None_
    block.forward = _sparse_basic_block_forward.__get__(block, block.__class__)


def _fuse_sparse_sequential(module: nn.Module) -> nn.Module:
    """Fuse an exact sparse Conv-BN-ReLU sequence into one convolution; return other modules unchanged."""
    if not isinstance(module, spconv.SparseSequential):
        return module
    children = list(module.children())
    if (len(children) == 3 and
            isinstance(children[0], spconv.conv.SparseConvolution) and
            isinstance(children[1], nn.BatchNorm1d) and
            isinstance(children[2], nn.ReLU)):
        conv = children[0]
        _fuse_bn_into_conv(conv, children[1])
        conv.act_type = tv.gemm.Activation.ReLU
        return conv
    return module


def fuse_sparse_encoder_for_export(model: nn.Module) -> nn.Module:
    """Deep-copy a sparse encoder, fuse supported BN/ReLU patterns, and return an eval-mode copy.

    Handle SparseBasicBlock, top-level sparse sequences, and sequences inside
    encoder_layers. Preserve original parameters and forward methods for later
    independent PyTorch references. Leave FP16 conversion and device selection
    to the export entrypoint.
    """
    exported = copy.deepcopy(model).eval()
    for name, module in list(exported.named_modules()):
        if isinstance(module, SparseBasicBlock):
            _fuse_sparse_basic_block(module)

    for name, module in list(exported.named_children()):
        setattr(exported, name, _fuse_sparse_sequential(module))

    for stage_name, stage in list(exported.encoder_layers.named_children()):
        for child_name, child in list(stage.named_children()):
            setattr(stage, child_name, _fuse_sparse_sequential(child))

    return exported


@contextmanager
def _trace_patches():
    """Temporarily patch six Torch/spconv methods to collect a sparse graph.

    Call original methods when _TRACE is disabled; otherwise run actual computation
    and record corresponding nodes. Restore every method and disable tracing on
    normal or exceptional exit. export_sparse_encoder_onnx clears retained objects
    in its outer finally block. Patches are process-wide: do not run concurrent
    models or nested exports inside this context.
    """
    originals = {
        'sparse_conv_forward': spconv.conv.SparseConvolution.forward,
        'relu_forward': torch.nn.ReLU.forward,
        'tensor_add': torch.Tensor.__add__,
        'dense': spconv.SparseConvTensor.dense,
        'view': torch.Tensor.view,
        'permute': torch.Tensor.permute,
    }

    def sparse_conv_forward(conv, x):
        """Run sparse convolution and record weights, shapes, rulebook, and FP16 operator attributes."""
        if not _TRACE.enabled:
            return originals['sparse_conv_forward'](conv, x)
        _TRACE.enabled = False
        y = originals['sparse_conv_forward'](conv, x)
        _TRACE.enabled = True
        layer = _TRACE.next_layer('conv')
        _TRACE.register_tensor(y)
        inputs = [
            _TRACE.tensor_id(x),
            _TRACE.append_initializer(conv.weight, f'spconv{layer}.weight'),
        ]
        if conv.bias is None:
            bias = torch.zeros(conv.out_channels, device=conv.weight.device,
                               dtype=conv.weight.dtype)
        else:
            bias = conv.bias
        inputs.append(_TRACE.append_initializer(bias, f'spconv{layer}.bias'))
        _TRACE.nodes.append(
            helper.make_node(
                'SparseConvolution',
                inputs,
                [_TRACE.tensor_id(y)],
                f'conv{layer}',
                domain=_SPCONV_DOMAIN,
                ndim=int(getattr(conv, 'ndim', 3)),
                input_spatial_shape=_as_list(x.spatial_shape),
                output_spatial_shape=_as_list(y.spatial_shape),
                in_channels=int(conv.in_channels),
                out_channels=int(conv.out_channels),
                kernel_size=_as_list(conv.kernel_size),
                output_bound=int(getattr(conv, 'output_bound', 200000)),
                stride=_as_list(conv.stride),
                dilation=_as_list(conv.dilation),
                padding=_as_list(conv.padding),
                transposed=bool(conv.transposed),
                inverse=bool(conv.inverse),
                output_padding=_as_list(conv.output_padding),
                groups=int(conv.groups),
                subm=bool(conv.subm),
                rulebook=str(getattr(conv, 'indice_key', '') or ''),
                activation=_activation_name(conv),
                algo=_conv_algo_name(conv),
                input_shape=list(x.features.shape),
                output_shape=list(y.features.shape),
                precision='fp16',
                output_precision='fp16',
            ))
        _TRACE.keep_alive.extend([x, y])
        return y

    def relu_forward(relu, x):
        """Record an unfused ReLU and retain input/output objects to prevent identity reuse."""
        if not _TRACE.enabled:
            return originals['relu_forward'](relu, x)
        input_id = _TRACE.tensor_id(x)
        _TRACE.enabled = False
        y = originals['relu_forward'](relu, x)
        _TRACE.enabled = True
        layer = _TRACE.next_layer('relu')
        output_id = _TRACE.register_tensor(y)
        _TRACE.nodes.append(
            helper.make_node('Relu', [input_id],
                             [output_id], f'relu{layer}',
                             domain=_SPCONV_DOMAIN))
        _TRACE.keep_alive.extend([x, y])
        return y

    def tensor_add(a, b):
        """Record residual addition of traced feature tensors as a custom-domain Add node."""
        if not _TRACE.enabled:
            return originals['tensor_add'](a, b)
        _TRACE.enabled = False
        y = originals['tensor_add'](a, b)
        _TRACE.enabled = True
        layer = _TRACE.next_layer('add')
        _TRACE.register_tensor(y)
        _TRACE.nodes.append(
            helper.make_node('Add', [_TRACE.tensor_id(a), _TRACE.tensor_id(b)],
                             [_TRACE.tensor_id(y)], f'add{layer}',
                             domain=_SPCONV_DOMAIN))
        _TRACE.keep_alive.extend([a, b, y])
        return y

    def dense(tensor, *args, **kwargs):
        """Record sparse-to-dense conversion as ScatterDense with parser output-layout attributes."""
        if not _TRACE.enabled:
            return originals['dense'](tensor, *args, **kwargs)
        _TRACE.enabled = False
        y = originals['dense'](tensor, *args, **kwargs)
        _TRACE.enabled = True
        layer = _TRACE.next_layer('scatter')
        _TRACE.register_tensor(y)
        _TRACE.nodes.append(
            helper.make_node(
                'ScatterDense',
                [_TRACE.tensor_id(tensor)],
                [_TRACE.tensor_id(y)],
                f'scatter{layer}',
                domain=_SPCONV_DOMAIN,
                input_spatial_shape=_as_list(tensor.spatial_shape),
                format='zyx',
                output_shape=list(y.shape),
                output_layout='NCHW',
            ))
        _TRACE.keep_alive.extend([tensor, y])
        return y

    def view(tensor, *dims):
        """Execute view and record explicit target dimensions in a custom-domain Reshape node."""
        if not _TRACE.enabled:
            return originals['view'](tensor, *dims)
        _TRACE.enabled = False
        y = originals['view'](tensor, *dims)
        _TRACE.enabled = True
        layer = _TRACE.next_layer('reshape')
        _TRACE.register_tensor(y)
        _TRACE.nodes.append(
            helper.make_node('Reshape', [_TRACE.tensor_id(tensor)],
                             [_TRACE.tensor_id(y)], f'reshape{layer}',
                             domain=_SPCONV_DOMAIN,
                             dims=[int(dim) for dim in dims]))
        _TRACE.keep_alive.extend([tensor, y])
        return y

    def permute(tensor, *dims):
        """Record a transpose, using a libspconv-fusible equivalent for the supported singleton-axis layout."""
        if not _TRACE.enabled:
            return originals['permute'](tensor, *dims)
        _TRACE.enabled = False
        y = originals['permute'](tensor, *dims)
        _TRACE.enabled = True
        layer = _TRACE.next_layer('transpose')
        _TRACE.register_tensor(y)
        onnx_dims = [int(dim) for dim in dims]
        # libspconv fuses ScatterDense(zyx) followed by only
        # [N, C, H, D, W].  BEVFusionSparseEncoder uses [N, C, D, H, W]
        # instead, but its final D axis is statically one.  Both layouts
        # therefore flatten to the same [N, C, H, W] tensor; emit the
        # libspconv-supported equivalent for the runtime graph.
        if (onnx_dims == [0, 1, 4, 2, 3] and tensor.ndim == 5 and
                tensor.shape[4] == 1):
            onnx_dims = [0, 1, 2, 4, 3]
        _TRACE.nodes.append(
            helper.make_node('Transpose', [_TRACE.tensor_id(tensor)],
                             [_TRACE.tensor_id(y)], f'transpose{layer}',
                             domain=_SPCONV_DOMAIN,
                             dims=onnx_dims))
        _TRACE.keep_alive.extend([tensor, y])
        return y

    spconv.conv.SparseConvolution.forward = sparse_conv_forward
    torch.nn.ReLU.forward = relu_forward
    torch.Tensor.__add__ = tensor_add
    spconv.SparseConvTensor.dense = dense
    torch.Tensor.view = view
    torch.Tensor.permute = permute
    try:
        yield
    finally:
        _TRACE.enabled = False
        spconv.conv.SparseConvolution.forward = originals['sparse_conv_forward']
        torch.nn.ReLU.forward = originals['relu_forward']
        torch.Tensor.__add__ = originals['tensor_add']
        spconv.SparseConvTensor.dense = originals['dense']
        torch.Tensor.view = originals['view']
        torch.Tensor.permute = originals['permute']


def export_sparse_encoder_onnx(middle_encoder: nn.Module,
                               voxel_features: torch.Tensor,
                               coors: torch.Tensor,
                               batch_size: int,
                               output_path: Path) -> torch.Tensor:
    """Export an XYZ-coordinate FP16 sparse backbone and return its copied encoder's dense BEV output.

    Args:
        middle_encoder: Supported sparse encoder; fuse BN only in a deep copy.
        voxel_features: Sample voxel means [Nvox, Cin], converted to contiguous FP16.
        coors: Matching [Nvox, 4] coordinates in [batch, x, y, z] order.
        batch_size: Batch size passed to the sparse encoder; deployment uses 1.
        output_path: ONNX destination; parent directories are created automatically.

    The graph declares only feature input; libspconv manages coordinates at runtime.
    Nodes use the ai.onnx.contrib domain and include required sparse convolution
    attributes. After saving the graph, return the FP16 dense tensor on the current
    device. Clear global tracing state on success or failure and restore patched
    methods through the context manager. Concurrent or nested calls are unsupported.
    """
    _TRACE.reset()
    try:
        middle_encoder = fuse_sparse_encoder_for_export(middle_encoder)
        middle_encoder = middle_encoder.to(voxel_features.device).half().eval()
        features = voxel_features.contiguous().half()
        indices = coors.contiguous().int()

        with torch.no_grad(), _trace_patches():
            _TRACE.register_tensor(features)
            _TRACE.enabled = True
            dense = middle_encoder(features, indices, batch_size)
            _TRACE.enabled = False

        if not _TRACE.nodes:
            raise RuntimeError('SCN trace produced no ONNX nodes')

        graph = helper.make_graph(
            _TRACE.nodes,
            'bevfusion_scn',
            [
                helper.make_value_info(
                    '0',
                    helper.make_tensor_type_proto(TensorProto.FLOAT16,
                                                  list(features.shape))),
            ],
            [
                helper.make_value_info(
                    _TRACE.tensor_id(dense),
                    helper.make_tensor_type_proto(TensorProto.FLOAT16,
                                                  list(dense.shape))),
            ],
            initializer=_TRACE.initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_operatorsetid('', 11),
                helper.make_operatorsetid(_SPCONV_DOMAIN, 1),
            ],
            producer_name='lidar-inference-tool-bevfusion',
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(model, str(output_path))
        return dense
    finally:
        _TRACE.reset()
