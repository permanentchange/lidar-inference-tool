#!/usr/bin/env python3
"""Exercise native builder discovery, literal arguments, artifacts, and failure handling."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/bevfusion'))
from _lib.engine_build import NETWORKS, build_artifacts, build_engines
from _lib.trtexec import find_trtexec


class EngineBuildTest(unittest.TestCase):
    """Use a native subprocess stub so tests do not require a CUDA device or TensorRT."""

    def setUp(self):
        """Create independent output and executable paths containing shell metacharacters."""
        self.directory = tempfile.TemporaryDirectory(prefix='bevfusion builder ')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / 'model $(touch UNEXPECTED)'
        self.output.mkdir()
        (self.output / 'model.yaml').write_text(json.dumps(dict(model=dict(
            architecture='centerpoint_bevfusion', head_fp32_layers=['heatmap/Conv']))))
        self.binary = self.root / 'trtexec with spaces'
        self.binary.write_text('#!' + sys.executable + '\n' + '''
import json, os, sys
from pathlib import Path
args = dict(arg[2:].split('=', 1) for arg in sys.argv[1:] if '=' in arg)
print(json.dumps(sys.argv[1:]))
Path(args['saveEngine']).write_bytes(b'engine')
if os.environ.get('BUILDER_TEST_FAIL'):
    raise SystemExit(17)
layers = [dict(Name=value.rsplit(':', 1)[0], LayerType='CaskConvolution',
               Outputs=[{'Format/Datatype': 'Row major linear ' + ('FP16' if os.environ.get('BUILDER_TEST_FP16_HEAD') else 'FP32')}])
          for value in args.get('layerPrecisions', '').split(',') if value]
Path(args['exportLayerInfo']).write_text('invalid' if os.environ.get('BUILDER_TEST_BAD_JSON') else json.dumps(dict(Layers=layers)))
Path(args['exportProfile']).write_text('[]')
''')
        self.binary.chmod(0o755)

    def test_discovery_precedence_and_invalid_explicit_path(self):
        """Respect explicit selection, then SDK root, then executable search paths."""
        sdk = self.root / 'sdk/bin'
        sdk.mkdir(parents=True)
        executable = sdk / 'trtexec'
        executable.symlink_to(self.binary)
        with mock.patch.dict(os.environ, {'TENSORRT_ROOT': str(sdk.parent), 'PATH': ''}, clear=True):
            self.assertEqual(find_trtexec(), self.binary.resolve())
            with self.assertRaisesRegex(RuntimeError, 'not executable'):
                find_trtexec(self.root / 'missing')
        with mock.patch.dict(os.environ, {'PATH': str(sdk)}, clear=True):
            self.assertEqual(find_trtexec(), self.binary.resolve())
        with mock.patch.dict(os.environ, {'PATH': ''}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'Cannot find trtexec'):
                find_trtexec()

    def test_all_networks_use_fp16_and_keep_literal_paths(self):
        """Produce every declared artifact with the same precision policy and device."""
        build_engines(self.output, self.binary, 2)
        self.assertTrue(all(path.is_file() for path in build_artifacts(self.output)))
        for name in NETWORKS:
            policy = json.loads((self.output / (name + '.precision.json')).read_text())
            command = policy['command']
            self.assertIn('--onnx=' + str(self.output / (name + '.onnx')), command)
            for flag in ('--fp16', '--device=2', '--inputIOFormats=fp16:chw',
                         '--memPoolSize=workspace:2048'):
                self.assertIn(flag, command)
            self.assertNotIn('--noTF32', command)
            self.assertNotIn('--int8', command)
            self.assertIn('--outputIOFormats=' + ('fp32:chw' if name=='fusion_centerhead' else 'fp16:chw'), command)
            self.assertEqual(any('layerPrecisions' in arg for arg in command), name=='fusion_centerhead')
            self.assertEqual(policy['binding_dtype'], 'float16')
            self.assertTrue(policy['tf32'])
        self.assertFalse((REPO / 'UNEXPECTED').exists())
        self.assertFalse(list(self.output.glob('*.tmp')))

    def test_failed_builder_preserves_engine_and_diagnostic(self):
        """Reject a failed subprocess while retaining its log and previous engine."""
        engine = self.output / 'camera.backbone.engine'
        engine.write_bytes(b'previous engine')
        with mock.patch.dict(os.environ, BUILDER_TEST_FAIL='1'):
            with self.assertRaisesRegex(RuntimeError, 'status 17; see'):
                build_engines(self.output, self.binary)
        self.assertEqual(engine.read_bytes(), b'previous engine')
        self.assertTrue((self.output / 'logs/build/camera.backbone.log').is_file())
        self.assertFalse(list(self.output.glob('*.tmp')))
        self.assertFalse((self.output / 'camera.backbone.precision.json').exists())

    def test_centerhead_two_engines_and_float32_head_outputs(self):
        """The aligned package builds two graphs with the CenterPoint output contract."""
        layers = ['/head/task_heads.0/heatmap/heatmap.1/Conv']
        (self.output / 'model.yaml').write_text(json.dumps(dict(model=dict(
            architecture='centerpoint_bevfusion', head_fp32_layers=layers))))
        build_engines(self.output, self.binary)
        self.assertTrue(all(path.is_file() for path in build_artifacts(self.output)))
        self.assertFalse((self.output / 'fuser.engine').exists())
        policy = json.loads((self.output / 'fusion_centerhead.precision.json').read_text())
        self.assertIn('--outputIOFormats=fp32:chw', policy['command'])
        self.assertIn('--inputIOFormats=fp16:chw', policy['command'])
        self.assertEqual(policy['output_dtype'], 'float32')
        self.assertIn('--precisionConstraints=obey', policy['command'])
        self.assertIn('--layerPrecisions=' + layers[0] + ':fp32', policy['command'])
        self.assertIn('--layerOutputTypes=' + layers[0] + ':fp32', policy['command'])
        self.assertEqual(policy['fp32_layers'], layers)

    def test_rejects_ignored_heatmap_precision_before_replacing_engine(self):
        (self.output / 'model.yaml').write_text(json.dumps(dict(model=dict(
            architecture='centerpoint_bevfusion', head_fp32_layers=['heatmap/Conv']))))
        engine = self.output / 'fusion_centerhead.engine'
        engine.write_bytes(b'previous head engine')
        with mock.patch.dict(os.environ, BUILDER_TEST_FP16_HEAD='1'):
            with self.assertRaisesRegex(RuntimeError, 'did not retain FP32'):
                build_engines(self.output, self.binary)
        self.assertEqual(engine.read_bytes(), b'previous head engine')

    def test_invalid_inspection_json_does_not_replace_engine(self):
        """Require complete parseable reports before publishing a newly built engine."""
        engine = self.output / 'camera.backbone.engine'
        engine.write_bytes(b'previous engine')
        with mock.patch.dict(os.environ, BUILDER_TEST_BAD_JSON='1'):
            with self.assertRaises(json.JSONDecodeError):
                build_engines(self.output, self.binary)
        self.assertEqual(engine.read_bytes(), b'previous engine')
        self.assertFalse(list(self.output.glob('*.tmp')))


if __name__ == '__main__':
    unittest.main()
