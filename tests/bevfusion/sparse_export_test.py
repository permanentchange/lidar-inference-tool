#!/usr/bin/env python3
"""GPU check for local SCN export state isolation and exception cleanup."""
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/bevfusion'))
import torch
from torch import nn
import spconv.pytorch as spconv
from _lib import sparse_export as exporter


class Encoder(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.input_conv = spconv.SparseSequential(spconv.SubMConv3d(5, 8, 3, padding=1, bias=False),
                                                 nn.BatchNorm1d(8), nn.ReLU())
        self.encoder_layers = nn.Sequential()
        self.fail = fail

    def forward(self, features, indices, batch):
        tensor = spconv.SparseConvTensor(features, indices, [4, 4, 4], batch)
        tensor = self.input_conv(tensor)
        if self.fail:
            raise RuntimeError('test trace exception')
        return tensor.dense().view(1, 8, 4, 16)


class SparseExportTest(unittest.TestCase):
    def test_fusion_and_trace_cleanup(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA required')
        encoder = Encoder().cuda().eval()
        state = {k: v.clone() for k, v in encoder.state_dict().items()}
        originals = (spconv.conv.SparseConvolution.forward, torch.Tensor.__add__,
                     torch.Tensor.view, torch.Tensor.permute, torch.nn.ReLU.forward,
                     spconv.SparseConvTensor.dense)
        features = torch.randn(4, 5, device='cuda')
        indices = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2], [0, 3, 3, 3]], dtype=torch.int32, device='cuda')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.onnx'
            exporter.export_sparse_encoder_onnx(encoder, features, indices, 1, path)
            import onnx
            onnx.checker.check_model(str(path))
            graph = onnx.load(str(path))
            self.assertTrue(any(n.op_type == 'SparseConvolution' for n in graph.graph.node))
            for key, value in encoder.state_dict().items():
                self.assertTrue(torch.equal(state[key], value), key)
            encoder.fail = True
            with self.assertRaisesRegex(RuntimeError, 'test trace exception'):
                exporter.export_sparse_encoder_onnx(encoder, features, indices, 1, path)
        restored = (spconv.conv.SparseConvolution.forward, torch.Tensor.__add__,
                    torch.Tensor.view, torch.Tensor.permute, torch.nn.ReLU.forward,
                    spconv.SparseConvTensor.dense)
        self.assertEqual(originals, restored)
        self.assertFalse(exporter._TRACE.enabled)
        self.assertEqual(exporter._TRACE.keep_alive, [])


if __name__ == '__main__':
    unittest.main()
