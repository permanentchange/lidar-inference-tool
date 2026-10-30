"""YAML configuration, JSON metadata, artifact fingerprints, and deployment stage state."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_FILES = {
    'export': ('export.py', '_lib/model.py', '_lib/onnx_export.py',
               '_lib/sparse_export.py', '_lib/images.py', '_lib/reference.py',
               '_lib/environment.py', '_lib/common.py', '../deployment_config.py',
               '../configs/centerpoint_deployment.yaml'),
    'build': ('build_engines.py', '_lib/engine_build.py', '_lib/trtexec.py', '_lib/environment.py', '_lib/common.py'),
    'validate': ('validate.py', '_lib/validation.py', '_lib/parity.py', '_lib/deployment_reference.py', '_lib/runtime.py',
                 '_lib/reference.py', '_lib/images.py', '_lib/model.py', '_lib/environment.py', '_lib/common.py'),
    'evaluate': ('evaluate.py', '_lib/evaluation.py', '_lib/parity.py', '_lib/deployment_reference.py', '_lib/runtime.py',
                 '_lib/reference.py', '_lib/images.py', '_lib/model.py', '_lib/environment.py', '_lib/common.py'),
}

def digest(path):
    """Read a file in chunks and return its hexadecimal SHA-256 digest."""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()

def identifier(value):
    """Hash JSON-serializable data with sorted keys to identify a stage dependency."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def read(path):
    """Parse YAML configurations/manifests and JSON reports by file extension."""
    path = Path(path)
    if path.suffix.lower() in ('.yaml', '.yml'):
        import yaml
        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())

def _json_default(value):
    """Convert Path and NumPy values for JSON serialization without importing NumPy."""
    if isinstance(value, Path):
        return str(value)
    if type(value).__module__.startswith('numpy'):
        return value.tolist()
    raise TypeError('Not JSON serializable: ' + type(value).__name__)

def write(path, value):
    """Write block-style YAML configs or JSON reports atomically by extension.

    Create parent directories and support Path and NumPy values. Reject NaN/Inf
    to keep the output valid JSON. Callers must avoid concurrent writes to the
    same destination.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    normalized = json.loads(json.dumps(value, allow_nan=False, default=_json_default))
    if path.suffix.lower() in ('.yaml', '.yml'):
        import yaml
        contents = yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True,
                                  default_flow_style=False, width=100)
    else:
        contents = json.dumps(normalized, indent=2) + '\n'
    temporary.write_text(contents)
    temporary.replace(path)

def files_under(path):
    """Return sorted descendant files for a directory, or a single-element list otherwise."""
    path = Path(path)
    return sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]

def artifacts(root, paths):
    """Expand stage artifacts into a mapping of model-relative paths to SHA-256 digests.

    Directories in paths are expanded recursively. Files must exist beneath root;
    missing files raise FileNotFoundError, and paths outside root raise ValueError.
    """
    result = {}
    for path in paths:
        for p in files_under(path):
            if not p.is_file():
                raise FileNotFoundError('Missing stage artifact: ' + str(p))
            result[str(p.relative_to(root))] = digest(p)
    return result

def tools_for(stage):
    """Return local tool source digests for the stage listed in STAGE_FILES without importing them."""
    base = REPO_ROOT / 'scripts/bevfusion'
    return {name: digest(base / name) for name in STAGE_FILES[stage]}

def versions(*packages):
    """Read installed distribution versions without importing packages; omit missing distributions."""
    result = {}
    for name in packages:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result

def metadata(root):
    """Read version 2 provenance.json from the model directory without checking artifacts.

    Raise RuntimeError when metadata is missing or uses an older version, requiring
    an independent export before proceeding.
    """
    path = Path(root) / 'provenance.json'
    if not path.is_file():
        raise RuntimeError('Export metadata missing; run scripts/bevfusion/export.py first')
    data = read(path)
    if data.get('version') != 2:
        raise RuntimeError('Stage metadata required; export into a new directory with scripts/bevfusion/export.py')
    return data

def verify_stage(root, name, refresh_deployment_config=False):
    """Verify a completed stage, its artifact digests, and recorded upstream identifiers.

    Return the verified stage dictionary. Raise RuntimeError for missing records,
    modified artifacts, or stale dependency identifiers. Upstream artifact files
    are not checked recursively; CLI entrypoints verify export and build separately.
    Configuration synchronization may regenerate its two owned deployment records
    and tolerate missing historical build logs. Model artifacts, model.yaml, demo
    inputs and dependency identifiers are still checked normally.
    """
    data = metadata(root)
    stage = data['stages'].get(name)
    if not stage or not stage.get('artifacts'):
        raise RuntimeError('Incomplete stage: ' + name + '; run its independent tool first')
    for relative, expected in dict(stage['artifacts'], **stage.get('input_artifacts', {})).items():
        p = Path(root) / relative
        if refresh_deployment_config:
            if name == 'export' and relative in ('deployment/reference.yaml', 'deployment/configuration.json'):
                continue
            if name == 'build' and relative.startswith('logs/build/') and relative.endswith('.log') and not p.exists():
                continue
        if not p.is_file() or digest(p) != expected:
            raise RuntimeError('Artifact hash mismatch: ' + str(p))
    for relative, expected in stage.get('diagnostic_artifacts', {}).items():
        p = Path(root) / relative
        if p.exists() and digest(p) != expected:
            raise RuntimeError('Diagnostic artifact hash mismatch: ' + str(p))
    for dependency in ('export', 'build'):
        if dependency in stage['inputs']:
            upstream = data['stages'].get(dependency)
            if upstream is None or identifier(upstream) != stage['inputs'][dependency]:
                raise RuntimeError('Stale ' + name + ' stage: dependency ' + dependency + ' changed')
    return stage

def begin_stage(root, name, source=None):
    """Persist stage invalidation before work starts, optionally updating export sources.

    Export clears every stage record; build clears build, validate, and evaluate;
    validate/evaluate clear only their own records. No artifact files are deleted.
    A stage becomes complete again only after a successful finish_stage call.
    """
    root = Path(root)
    path = root / 'provenance.json'
    data = metadata(root) if path.exists() and read(path).get('version') == 2 else {'version': 2, 'stages': {}}
    invalidated = {'export': ('export', 'build', 'validate', 'evaluate'),
                   'build': ('build', 'validate', 'evaluate'),
                   'validate': ('validate',), 'evaluate': ('evaluate',)}[name]
    for stale in invalidated:
        data['stages'].pop(stale, None)
    if source is not None:
        data['source'] = source
    write(path, data)

def finish_stage(root, name, inputs, paths, details=None):
    """Record a completed stage's inputs, artifact/tool digests, environment, and UTC time.

    root is the model directory. inputs contains upstream identifiers or source
    fingerprints, paths lists owned artifacts, and details optionally describes
    the environment. Update provenance.json only after hashing artifacts succeeds;
    the caller is responsible for running and accepting the actual stage work.
    """
    root = Path(root)
    data = metadata(root)
    owned = artifacts(root, paths)
    diagnostics = {p:h for p,h in owned.items()
                   if (name == 'export' and p.startswith(('reference/', 'reference_original/')))
                   or (name == 'build' and p.startswith('logs/build/') and p.endswith('.log'))}
    owned = {p:h for p,h in owned.items() if p not in diagnostics}
    data['stages'][name] = dict(inputs=inputs, artifacts=owned, diagnostic_artifacts=diagnostics,
                               tools=tools_for(name), details=details or {},
                               completed_at=datetime.now(timezone.utc).isoformat())
    write(root / 'provenance.json', data)

def require_overwrite(paths, overwrite):
    """Raise RuntimeError if a stage destination exists and overwrite is not explicitly enabled."""
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing and not overwrite:
        raise RuntimeError('Stage output exists; use --overwrite: ' + existing[0])

def reset_directory(path, overwrite):
    """Check overwrite permission, then recreate only the requested directory and preserve siblings."""
    path = Path(path)
    require_overwrite([path], overwrite)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

def positive(value):
    """Parse a positive integer; reject nonpositive values with argparse.ArgumentTypeError."""
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value

def nonnegative(value):
    """Parse a nonnegative integer; reject negative values with argparse.ArgumentTypeError."""
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return value

def run(main):
    """Run a zero-argument CLI main, reporting common runtime, argument, and I/O errors with exit code 1."""
    try:
        main()
    except (RuntimeError, ValueError, OSError, argparse.ArgumentTypeError) as error:
        print('Error: ' + str(error), file=sys.stderr)
        raise SystemExit(1)


def sample_manifest(root):
    """Resolve the recorded demo input separately from the engine package."""
    root = Path(root)
    sample = metadata(root).get('sample')
    return (root / sample['manifest']).resolve() if sample else root / 'sample/sample.yaml'


def register_sample(root, manifest, preserve_build=False):
    """Record a verified data-only relocation without rebuilding model artifacts.

    Preserve original stage timestamps/tools and record their previous identifiers.
    Model bytes are verified before migration; downstream validation is invalidated.
    """
    import os
    root, manifest = Path(root).resolve(), Path(manifest).resolve()
    exported = verify_stage(root, 'export')
    built = verify_stage(root, 'build') if preserve_build else None
    data = metadata(root)
    data['stages']['export'] = exported
    if built:
        data['stages']['build'] = built
    before = {k: identifier(v) for k, v in data['stages'].items()}
    node = read(manifest)
    paths = [manifest, manifest.parent / node['point_cloud']]
    if node.get('preprocessed_images'):
        paths.append(manifest.parent / node['preprocessed_images'])
    paths.extend(manifest.parent / c['image'] for c in node['cameras'] if c.get('image'))
    # file_list describes model-package files; external demo data has its own ledger.
    import yaml
    package_path = root / 'model.yaml'
    package = yaml.safe_load(package_path.read_text())
    if any(str(p).startswith('sample/') for p in package.get('file_list', [])):
        package['file_list'] = [p for p in package['file_list'] if not str(p).startswith('sample/')]
        write(package_path, package)
        exported['artifacts']['model.yaml'] = digest(package_path)
    diagnostics = {p:h for p,h in exported['artifacts'].items() if p.startswith(('reference/', 'reference_original/'))}
    exported.setdefault('diagnostic_artifacts', {}).update(diagnostics)
    exported['artifacts'] = {p: h for p, h in exported['artifacts'].items() if p not in diagnostics and not p.startswith('sample/')}
    exported['input_artifacts'] = {os.path.relpath(p.resolve(), root): digest(p) for p in paths}
    data['sample'] = dict(manifest=os.path.relpath(manifest, root))
    if built:
        built['inputs']['export'] = identifier(exported)
    data['stages'].pop('validate', None)
    data['stages'].pop('evaluate', None)
    data.setdefault('migrations', []).append(dict(kind='demo_input_relocation', previous_identifiers=before,
        at=datetime.now(timezone.utc).isoformat(), models_rebuilt=False))
    write(root / 'provenance.json', data)
