#!/usr/bin/env python3
"""Check raw/preprocessed inputs, operators, decode, engine parity and test frames."""
import argparse
from pathlib import Path
from _lib.common import (REPO_ROOT, begin_stage, digest, finish_stage, identifier,
                         positive, reset_directory, run, verify_stage)
from _lib.environment import bootstrap_native, source_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--frames', type=positive, default=20)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--tool', default=str(REPO_ROOT / 'build/infer_debug_tool'))
    parser.add_argument('--keep-intermediates', action='store_true', help='Keep tensor/fixture dumps after successful checks')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    root = Path(args.model_dir).resolve()
    args.tool = str(Path(args.tool).resolve())
    exported, built = verify_stage(root, 'export'), verify_stage(root, 'build')
    tool_hash = digest(args.tool)
    bootstrap_native()
    cfg, model = source_context(root, args.device)
    from _lib.validation import validate
    reset_directory(root / 'validation', args.overwrite)
    begin_stage(root, 'validate')
    validate(args, cfg, model, root)
    if not args.keep_intermediates:
        import shutil
        for artifact in (root / 'validation').iterdir():
            if artifact.name == 'report.json': continue
            if artifact.is_dir(): shutil.rmtree(artifact)
            else: artifact.unlink()
    finish_stage(root, 'validate', dict(export=identifier(exported), build=identifier(built),
                 infer_tool=tool_hash, frames=args.frames), [root / 'validation/report.json'])
    print('Numerical validation complete:', root / 'validation/report.json', flush=True)


if __name__ == '__main__':
    run(main)
