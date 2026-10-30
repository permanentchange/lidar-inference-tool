#!/usr/bin/env python3
"""Evaluate PyTorch and deployed BEVFusion on the complete configured val dataset."""
import argparse
from pathlib import Path
from _lib.common import (REPO_ROOT, begin_stage, digest, finish_stage, identifier,
                         reset_directory, run, verify_stage, write)
from _lib.environment import bootstrap_native, source_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--tool', default=str(REPO_ROOT / 'build/infer_debug_tool'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    root = Path(args.model_dir).resolve()
    args.tool = str(Path(args.tool).resolve())
    exported, built = verify_stage(root, 'export'), verify_stage(root, 'build')
    tool_hash = digest(args.tool)
    bootstrap_native()
    cfg, model = source_context(root, args.device)
    from _lib.evaluation import evaluate_validation
    reset_directory(root / 'evaluation', args.overwrite)
    begin_stage(root, 'evaluate')
    report = evaluate_validation(args, cfg, model, root)
    write(root / 'evaluation/report.json', report)
    finish_stage(root, 'evaluate', dict(export=identifier(exported), build=identifier(built),
                 infer_tool=tool_hash), [root / 'evaluation'])
    print('Full evaluation complete:', root / 'evaluation/report.json', flush=True)


if __name__ == '__main__':
    run(main)
