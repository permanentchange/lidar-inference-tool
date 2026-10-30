"""Compare full-dataset model and deployment metrics using configured validation loaders and evaluators."""
import copy
import json
import torch
from .common import read, write
from .model import build_dataloader
from .reference import fixture, original_reference
from .runtime import run_tool
from .parity import cpp_prediction, clean_json

def evaluate_validation(args, cfg, model, output):
    """Evaluate PyTorch and C++ deployment predictions over the complete validation dataset.

    Args:
        args: CLI options containing tool and device.
        cfg: Matching model configuration with val_dataloader and val_evaluator.
        model: Original model with strictly loaded weights, in eval mode.
        output: Model directory Path; the entrypoint handles evaluation/ overwrite.

    Create separate Evaluators with the same dataset metainfo and evaluation
    configuration, preserving its ignore filtering. Save each fixture and original
    prediction, then run the C++ tool over the manifests. Wrap deployed geometric
    center boxes in the framework's box structure for evaluation. Return a JSON-
    compatible dictionary with frames, original, and deployment metrics. The CLI
    writes evaluation/report.json and the completion record.
    """
    work = output / "evaluation"
    work.mkdir(parents=True, exist_ok=True)
    package = read(output / 'model.yaml')
    metric_package = copy.deepcopy(package)
    # The configured evaluator/annotation files use nine original classes. Keep that
    # label space explicitly, while evaluating the exact deployment NMS policy.
    metric_package['postprocess']['label_mapping'] = list(range(len(package['model']['label_names'])))
    metric_package['postprocess']['mapped_label_names'] = package['model']['label_names']
    override = work / 'metric_label_space.yaml'
    write(override, dict(postprocess={key: metric_package['postprocess'][key]
                                    for key in ('label_mapping', 'mapped_label_names')}))
    from mmengine.evaluator import Evaluator
    from mmengine.structures import InstanceData
    from mmdet3d.structures import LiDARInstance3DBoxes
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
    loader = build_dataloader(eval_cfg)
    original_eval = Evaluator(copy.deepcopy(cfg.val_evaluator))
    deploy_eval = Evaluator(copy.deepcopy(cfg.val_evaluator))
    dataset_meta = loader.dataset.metainfo
    original_eval.dataset_meta = dataset_meta
    deploy_eval.dataset_meta = dataset_meta
    manifests = work / 'val_manifests'
    manifests.mkdir(exist_ok=True)
    samples = []
    for index, batch in enumerate(loader):
        folder = work / 'val_fixtures' / f'{index:06d}'
        manifest = fixture(eval_cfg, model, batch, folder)
        node = read(manifest)
        for key in ('point_cloud', 'preprocessed_images'):
            node[key] = str((folder / node[key]).resolve())
        write(manifests / f'{index:06d}.yaml', node)
        _, pred = original_reference(model, batch, folder / 'original', tensors=False, deployment=metric_package)
        original_eval.process(data_samples=[pred], data_batch=batch)
        samples.append(pred.cpu())
        if index % 20 == 0:
            print('Full val original', index, '/', len(loader.dataset), flush=True)
    target = work / 'val_cpp'
    run_tool(args, output, manifests, target, override_config=override)
    for index, sample in enumerate(samples):
        result = cpp_prediction(target / 'frames' / f'{index:06d}_{index:06d}' / 'result.json')
        boxes = torch.tensor(result['boxes'], dtype=torch.float32).reshape(-1, 9)
        sample.pred_instances_3d = InstanceData(bboxes_3d=LiDARInstance3DBoxes(boxes, box_dim=9, origin=(.5, .5, .5)),
                                               scores_3d=torch.tensor(result['scores']), labels_3d=torch.tensor(result['labels'], dtype=torch.long))
        deploy_eval.process(data_samples=[sample], data_batch=None)
    return clean_json(dict(frames=len(samples), original=original_eval.evaluate(len(samples)),
                           deployment=deploy_eval.evaluate(len(samples)),
                           label_space='nine original classes, with an explicit identity output mapping for evaluation only',
                           production_label_mapping=package['postprocess'].get('label_mapping', []),
                           note='Both references use deployment voxel limits and NMS. Native head-policy predictions are retained separately. These are original-class metrics, not five-class business metrics.'))
