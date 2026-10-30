"""Small CPU regression for configuration-record recovery and model hash protection."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import deployment_config as config


class SynchronizationTest(unittest.TestCase):
    def test_missing_records_recover_but_changed_engine_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = config.load_reference(config.DEFAULT_CONFIG)
            config.write(root / 'model.yaml', reference)
            graph = root / reference['model']['scn_onnx_path']
            engine = root / reference['model']['head_engine_path']
            graph.write_bytes(b'original graph')
            engine.write_bytes(b'original engine')
            exported = dict(inputs={}, completed_at='original export time', artifacts={
                'model.yaml': config.digest(root / 'model.yaml'),
                graph.name: config.digest(graph),
                'deployment/reference.yaml': 'previous snapshot hash',
                'deployment/configuration.json': 'previous configuration record hash'})
            built = dict(inputs=dict(export=config.identifier(exported)),
                         completed_at='original build time', artifacts={
                             engine.name: config.digest(engine),
                             'logs/build/head.log': 'previous log hash'})
            config.write(root / 'provenance.json', dict(version=2, stages=dict(export=exported, build=built)))
            before = (root / 'model.yaml').read_bytes()
            engine.write_bytes(b'changed engine')
            with self.assertRaisesRegex(RuntimeError, 'Artifact hash mismatch'):
                config.prepare_synchronization(root)
            self.assertEqual((root / 'model.yaml').read_bytes(), before)
            self.assertFalse((root / 'deployment').exists())
            engine.write_bytes(b'original engine')
            config.synchronize(root, config.DEFAULT_CONFIG)
            from _lib.common import verify_stage
            verify_stage(root, 'export')
            result = verify_stage(root, 'build')
            self.assertEqual(result['completed_at'], 'original build time')
            self.assertEqual(result['diagnostic_artifacts']['logs/build/head.log'], 'previous log hash')
            self.assertFalse((root / 'logs/build/head.log').exists())
            self.assertEqual((root / 'deployment/reference.yaml').read_bytes(), config.DEFAULT_CONFIG.read_bytes())
            self.assertTrue((root / 'deployment/configuration.json').is_file())
            self.assertEqual(engine.read_bytes(), b'original engine')
            self.assertEqual(graph.read_bytes(), b'original graph')


if __name__ == '__main__':
    unittest.main()
