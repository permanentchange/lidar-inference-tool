#!/usr/bin/env python3
"""Deployment CLI, stage ownership and Bash wiring checks without a GPU."""
import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / 'scripts/bevfusion'
sys.path.insert(0, str(SCRIPTS))
from _lib import common
from _lib.pipeline_config import load_config


class ScriptsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='bevfusion scripts ')
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def config(self, **changes):
        config = dict(mmdet3d_root='training repo', config='model config.py',
                      checkpoint='weights.pth', model_dir='model output')
        config.update(changes)
        path = self.root / 'experiment.yaml'
        path.write_text(json.dumps(config))
        return path

    def command(self, arguments, **kwargs):
        return subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, **kwargs)

    def test_help_without_training_or_gpu_imports(self):
        guard = self.root / 'sitecustomize.py'
        guard.write_text('''import sys
def guard(event,args):
    if event == 'import' and args[0].split('.')[0] in {'torch','tensorrt','mmengine','mmdet3d','mmcv','mmdet'}:
        raise RuntimeError('Unexpected heavy import during help: ' + args[0])
sys.addaudithook(guard)
''')
        env = dict(os.environ, PYTHONPATH=str(self.root))
        for entry in ('export', 'build_engines', 'validate', 'evaluate', 'benchmark_pooling'):
            result = self.command([sys.executable, str(SCRIPTS / (entry + '.py')), '--help'], env=env)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn('usage:', result.stdout)
        result = self.command(['bash', str(SCRIPTS / 'run_pipeline.sh'), '--help'],
                              env=dict(env, PYTHON_BIN='/does/not/exist'))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_config_paths_and_defaults(self):
        path = self.config()
        values = load_config(path)
        self.assertEqual(values[:4], [str(self.root / 'training repo'),
                         str(self.root / 'training repo/model config.py'),
                         str(self.root / 'training repo/weights.pth'), str(self.root / 'model output')])
        self.assertEqual(values[4:7], ['cuda:0', '0', '20'])
        self.assertEqual(values[7], str(self.root / 'model output/sample'))

    def test_config_errors(self):
        for change in [dict(unknown=1), dict(device='cpu'), dict(validation_frames=0),
                       dict(validation_frames=True), dict(sample_index=-1), dict(checkpoint=None)]:
            with self.assertRaises((ValueError, argparse.ArgumentTypeError)):
                load_config(self.config(**change))
        path = self.root / 'bad.yaml'
        path.write_text('[]')
        with self.assertRaises(ValueError):
            load_config(path)

    def stage(self, root, name, inputs, filename):
        path = root / filename
        path.write_text(name)
        common.begin_stage(root, name)
        common.finish_stage(root, name, inputs, [path])
        return common.verify_stage(root, name)

    def test_stage_dependencies_and_invalidation(self):
        root = self.root / 'model'
        common.begin_stage(root, 'export', {'checkpoint': 'source'})
        (root / 'model.yaml').write_text('export')
        common.finish_stage(root, 'export', {}, [root / 'model.yaml'])
        exported = common.verify_stage(root, 'export')
        built = self.stage(root, 'build', {'export': common.identifier(exported)}, 'test.engine')
        self.stage(root, 'validate', {'build': common.identifier(built)}, 'validation.json')
        self.stage(root, 'evaluate', {'build': common.identifier(built)}, 'evaluation.json')
        with mock.patch.object(common, 'tools_for', return_value={'validation changed': 'new hash'}):
            self.assertEqual(common.verify_stage(root, 'export'), exported)
            self.assertEqual(common.verify_stage(root, 'build'), built)
        common.begin_stage(root, 'validate')
        self.assertNotIn('validate', common.metadata(root)['stages'])
        self.assertIn('evaluate', common.metadata(root)['stages'])
        common.begin_stage(root, 'build')
        self.assertEqual(set(common.metadata(root)['stages']), {'export'})
        (root / 'model.yaml').write_text('tampered')
        with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
            common.verify_stage(root, 'export')

    def test_old_or_incomplete_metadata_is_not_certified(self):
        common.write(self.root / 'provenance.json', {'fingerprint': {}, 'artifacts': {}})
        with self.assertRaisesRegex(RuntimeError, 'Stage metadata required'):
            common.verify_stage(self.root, 'export')
        common.write(self.root / 'provenance.json', {'version': 2, 'stages': {}})
        with self.assertRaisesRegex(RuntimeError, 'Incomplete stage'):
            common.verify_stage(self.root, 'export')

    def test_existing_export_requires_explicit_overwrite(self):
        (self.root / 'model.yaml').write_text('owned')
        result = self.command([sys.executable, str(SCRIPTS / 'export.py'),
                     '--mmdet3d-root', '/missing', '--config', 'a', '--checkpoint', 'b',
                     '--output-dir', str(self.root)])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--overwrite', result.stdout)
        self.assertEqual((self.root / 'model.yaml').read_text(), 'owned')

    def stub_python(self):
        path = self.root / 'python with spaces'
        real = shlex.quote(sys.executable)
        code = '''import json,os,sys
from pathlib import Path
entry=Path(sys.argv[1]).name
with open(os.environ['BEVFUSION_TEST_CALLS'],'a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
print(entry)
raise SystemExit(17 if entry==os.environ.get('BEVFUSION_TEST_FAIL') else 0)
'''
        path.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                        'if [ "${1:-}" = -m ]; then exec ' + real + ' "$@"; fi\n'
                        'exec ' + real + ' -c ' + shlex.quote(code) + ' "$@"\n')
        path.chmod(0o755)
        return path

    def pipeline(self, failure=None, overwrite=False, **changes):
        calls = self.root / 'calls.jsonl'
        env = dict(os.environ, PYTHON_BIN=str(self.stub_python()), BEVFUSION_TEST_CALLS=str(calls))
        if failure:
            env['BEVFUSION_TEST_FAIL'] = failure
        path = self.config(**changes)
        arguments = ['bash', str(SCRIPTS / 'run_pipeline.sh'), '--config', str(path)]
        if overwrite:
            arguments.append('--overwrite')
        result = self.command(arguments, cwd='/tmp', env=env)
        rows = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return result, rows

    def test_bash_runs_exactly_three_steps_and_quotes_paths(self):
        result, rows = self.pipeline(checkpoint='weights $(do-not-execute).pth')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual([Path(row[0]).name for row in rows], ['export.py', 'build_engines.py', 'validate.py'])
        self.assertIn(str(self.root / 'training repo/weights $(do-not-execute).pth'), rows[0])
        self.assertIn(str(self.root / 'model output'), rows[1])
        self.assertIn('20', rows[2])
        self.assertFalse((self.root / 'model output/evaluation').exists())

    def test_bash_forwards_only_relevant_parameters(self):
        result, rows = self.pipeline(overwrite=True, device='cuda:2', sample_index=3, validation_frames=5)
        self.assertEqual(result.returncode, 0, result.stdout)
        for row in rows:
            self.assertIn('--overwrite', row)
            self.assertIn('cuda:2', row)
        self.assertIn('--sample-index', rows[0])
        self.assertNotIn('--sample-index', rows[1])
        self.assertNotIn('--frames', rows[0])
        self.assertIn('5', rows[2])

    def test_reset_owns_only_requested_directory(self):
        owned = self.root / 'sample'
        owned.mkdir()
        (owned / 'stale.npy').write_text('old sample')
        engine = self.root / 'head.bbox.engine'
        engine.write_text('keep engine')
        with self.assertRaisesRegex(RuntimeError, '--overwrite'):
            common.reset_directory(owned, False)
        common.reset_directory(owned, True)
        self.assertEqual(list(owned.iterdir()), [])
        self.assertEqual(engine.read_text(), 'keep engine')

    def test_bash_stops_after_failed_step(self):
        result, rows = self.pipeline(failure='build_engines.py')
        self.assertEqual(result.returncode, 17, result.stdout)
        self.assertEqual([Path(row[0]).name for row in rows], ['export.py', 'build_engines.py'])
        self.assertNotIn('Pipeline complete', result.stdout)

    def test_invalid_config_stops_before_creating_outputs(self):
        result, rows = self.pipeline(unknown=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(rows, [])
        self.assertFalse((self.root / 'model output').exists())

    def test_tools_have_no_external_deployment_tool_loader(self):
        for path in SCRIPTS.rglob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse('.tools' in (node.module or ''), str(path))
                if isinstance(node, ast.Attribute):
                    self.assertNotEqual(node.attr, 'spec_from_file_location', str(path))
        self.assertFalse((REPO / 'scripts/prepare_bevfusion.py').exists())


    def test_yaml_configuration_and_json_report_formats(self):
        value = dict(model=dict(runtime_backend='bevfusion', voxel_size=[0.05,0.05,0.1]))
        config = self.root / 'model.yaml'
        report = self.root / 'report.json'
        common.write(config, value)
        common.write(report, value)
        self.assertTrue(config.read_text().startswith('model:\n'))
        self.assertFalse(config.read_text().lstrip().startswith('{'))
        self.assertTrue(report.read_text().startswith('{'))
        self.assertEqual(common.read(config), common.read(report))

    def test_sample_relocation_retains_build_and_verifiable_data(self):
        """Moving inputs must not retain a dangling export ledger or change engine bytes."""
        import shutil
        root = self.root / 'package'
        (root / 'sample').mkdir(parents=True)
        common.write(root / 'model.yaml', dict(file_list=['network.blob','sample/sample.yaml']))
        (root / 'network.blob').write_bytes(b'unchanged model')
        common.write(root / 'sample/sample.yaml', dict(point_cloud='points.npy',cameras=[]))
        (root / 'sample/points.npy').write_bytes(b'points')
        common.begin_stage(root, 'export')
        common.finish_stage(root, 'export', {}, [root/'model.yaml',root/'network.blob',root/'sample'])
        exported = common.verify_stage(root, 'export')
        common.begin_stage(root, 'build')
        common.finish_stage(root, 'build', dict(export=common.identifier(exported)), [root/'network.blob'])
        before = common.verify_stage(root, 'build')['completed_at']
        destination = self.root / 'separate demo'
        shutil.copytree(root / 'sample', destination)
        common.register_sample(root, destination / 'sample.yaml', preserve_build=True)
        shutil.rmtree(root / 'sample')
        common.verify_stage(root, 'export')
        built = common.verify_stage(root, 'build')
        self.assertEqual(before, built['completed_at'])
        self.assertEqual((root/'network.blob').read_bytes(), b'unchanged model')
        self.assertEqual(common.sample_manifest(root), destination/'sample.yaml')
        self.assertNotIn('sample/sample.yaml', common.read(root/'model.yaml')['file_list'])
        (destination/'points.npy').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'Artifact hash mismatch'):
            common.verify_stage(root, 'export')


if __name__ == '__main__':
    unittest.main()
