"""Compare tensors, match candidates, and check operators/decode without a candidate match-rate gate."""
import json
from pathlib import Path
from .common import read, sample_manifest
import numpy as np
import torch

def metrics(a, b):
    """Compute float64 absolute errors, normalized RMSE, and cosine similarity using a as reference.

    Inputs must be convertible to NumPy arrays. On shape mismatch return only
    the two shapes and shape_match=False. Normalize nrmse by the RMS of a with
    a denominator floor of 1e-30. Compute statistics without tolerance checks or
    nonfinite-value cleanup.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return dict(shape_a=list(a.shape), shape_b=list(b.shape), shape_match=False)
    diff = a - b
    denom = max(float(np.sqrt(np.mean(a * a))), 1e-30) if a.size else 1.
    cosdenom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-30)
    return dict(shape=list(a.shape), max_abs=float(np.max(np.abs(diff), initial=0)),
                mean_abs=float(np.mean(np.abs(diff))) if a.size else 0.,
                nrmse=float(np.sqrt(np.mean(diff * diff)) / denom) if a.size else 0.,
                cosine=float(np.sum(a * b) / cosdenom))


def cpp_prediction(path):
    """Read C++ result JSON into a boxes/scores/labels list dictionary.

    Accept objects or the older detections field. Append two velocity components
    to each seven-component geometric center box to obtain
    [x, y, z, dx, dy, dz, yaw, vx, vy], preserving result order.
    """
    node = json.loads(Path(path).read_text())
    rows = node.get('objects', node.get('detections'))
    return dict(boxes=[r['box'] + r['velocity'] for r in rows],
                scores=[r['score'] for r in rows], labels=[r['label'] for r in rows])


def compare_predictions(a, b):
    """Compare original/deployed candidates by Hungarian assignment with a class penalty.

    a and b contain nine-component boxes, scores, and labels. Use XY center
    distance as cost, add a large penalty for class mismatches, and discard
    cross-class assignments afterwards. Return both complete inputs, pair indices,
    box/velocity/score deltas, and unmatched indices. Count pairs within 0.2 m as
    a statistic, never as a pass/fail gate. If either side is empty, return no pairs.
    """
    from scipy.optimize import linear_sum_assignment
    x, y = np.asarray(a['boxes']).reshape(-1, 9), np.asarray(b['boxes']).reshape(-1, 9)
    if not len(x) or not len(y):
        return dict(original=a, deployment=b, pairs=[], unmatched_original=list(range(len(x))), unmatched_deployment=list(range(len(y))))
    dist = np.linalg.norm(x[:, None, :2] - y[None, :, :2], axis=2)
    equal = np.asarray(a['labels'])[:, None] == np.asarray(b['labels'])[None, :]
    rows, cols = linear_sum_assignment(dist + (~equal) * 1e6)
    pairs = []
    for i, j in zip(rows, cols):
        if not equal[i, j]:
            continue
        pairs.append(dict(original_index=int(i), deployment_index=int(j), center_xy_distance=float(dist[i, j]),
                          box_delta=(y[j, :7] - x[i, :7]).tolist(), velocity_delta=(y[j, 7:] - x[i, 7:]).tolist(),
                          score_delta=float(b['scores'][j] - a['scores'][i]), within_0_2m=bool(dist[i, j] <= .2)))
    used_a = {p['original_index'] for p in pairs}
    used_b = {p['deployment_index'] for p in pairs}
    return dict(original=a, deployment=b, pairs=pairs,
                unmatched_original=[i for i in range(len(x)) if i not in used_a],
                unmatched_deployment=[i for i in range(len(y)) if i not in used_b],
                matched_within_0_2m=sum(p['within_0_2m'] for p in pairs),
                note='All candidates are preserved. No arbitrary candidate-match-rate acceptance gate.')


def binding_types(output):
    """Inspect linear floating bindings and return dtype names for all four engines."""
    import tensorrt as trt
    from .engine_build import networks
    report = {}
    logger = trt.Logger(trt.Logger.WARNING)
    for name in networks(output):
        with trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine((output / (name + '.engine')).read_bytes())
            if engine is None:
                raise RuntimeError('Cannot deserialize engine: ' + name)
            bindings = {}
            for i in range(engine.num_bindings):
                dtype = engine.get_binding_dtype(i)
                if dtype not in (trt.float16, trt.float32) or engine.get_binding_format(i) != trt.TensorFormat.LINEAR:
                    raise RuntimeError('Expected linear FP16/FP32 binding: ' + name + '/' + engine.get_binding_name(i))
                bindings[engine.get_binding_name(i)] = 'float16' if dtype == trt.float16 else 'float32'
            report[name] = bindings
    return report


def same_engine(output, internal):
    """Execute the same four engines as C++ in Python to isolate runtime differences.

    Args:
        output: Model directory Path containing four TensorRT engines.
        internal: Path containing C++ input and intermediate NPY tensors.

    Allocate bindings in their engine-declared dtype on the CUDA device and feed each stage's
    captured inputs directly instead of chaining previous Python stage outputs.
    Return error statistics for ten output bindings. Raise AssertionError when
    atol=1e-4 and rtol=1e-3 are exceeded.
    """
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    report = {}
    names = sorted(p.stem for p in internal.glob('hm_*.npy'))
    outputs = {f'{name}_{i}': f'{name}_{i}' for i in range(len(names))
               for name in ('reg', 'height', 'dim', 'rot', 'vel', 'hm')}
    cases = [('camera.backbone', dict(img='images', depth='sparse_depth'),
              dict(camera_feature='camera_feature', camera_depth_weights='camera_depth_weights')),
             ('fusion_centerhead', dict(input='sparse_bev', camera_bev='camera_bev_before_downsample'), outputs)]
    for name, feeds, outputs in cases:
        with trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine((output / (name + '.engine')).read_bytes())
            context = engine.create_execution_context()
            tensors = []
            for i in range(engine.num_bindings):
                binding = engine.get_binding_name(i)
                dtype = engine.get_binding_dtype(i)
                if dtype not in (trt.float16, trt.float32):
                    raise RuntimeError('Unsupported engine binding dtype: ' + binding)
                torch_dtype = torch.float16 if dtype == trt.float16 else torch.float32
                if engine.binding_is_input(i):
                    t = torch.from_numpy(np.load(internal / (feeds[binding] + '.npy'))).to(device='cuda', dtype=torch_dtype).contiguous()
                    if not torch.isfinite(t).all():
                        raise AssertionError('Nonfinite engine input: ' + name + '/' + binding)
                else:
                    t = torch.empty(tuple(engine.get_binding_shape(i)), device='cuda', dtype=torch_dtype)
                tensors.append(t)
            stream = torch.cuda.current_stream()
            if not context.execute_async_v2([t.data_ptr() for t in tensors], stream.cuda_stream):
                raise RuntimeError('Python TensorRT execution failed: ' + name)
            stream.synchronize()
            for binding, reference in outputs.items():
                actual = tensors[engine.get_binding_index(binding)].float().cpu().numpy()
                if name == 'camera.backbone':
                    cameras = read(sample_manifest(output))['cameras']
                    actual = actual.copy()
                    actual[[not c.get('enabled', True) for c in cameras]] = 0
                expected = np.load(internal / (reference + '.npy'))
                if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                    raise AssertionError('Nonfinite engine output: ' + name + '/' + binding)
                report[name + '/' + binding] = metrics(expected, actual)
                if not np.allclose(actual, expected, atol=1e-4, rtol=1e-3):
                    raise AssertionError('Same-engine Python/C++ parity failed: ' + name + '/' + binding)
    return report


def independent_decode(model, internal, result_path, package):
    """Check deployed head tensors against an independent deployment-policy reference."""
    from .deployment_reference import predict
    heads = [{k: np.load(internal / (k + '_' + str(i) + '.npy'))
              for k in ('reg', 'height', 'dim', 'rot', 'vel', 'hm')}
             for i in range(len(package['model']['task_label_ids']))]
    expected = predict(heads, package)
    cpp = cpp_prediction(result_path)
    boxes = np.asarray(expected['boxes'], np.float32).reshape(-1, 9)
    actual = np.asarray(cpp['boxes'], np.float32).reshape(-1, 9)
    if not np.array_equal(expected['labels'], cpp['labels']) or boxes.shape != actual.shape or not np.allclose(
            boxes, actual, atol=1e-4, rtol=1e-3):
        raise AssertionError('Independent deployment decode/NMS differs: ' + str(compare_predictions(expected, cpp)))
    if not np.allclose(expected['scores'], cpp['scores'], atol=1e-7, rtol=1e-5):
        raise AssertionError('Independent deployment scores differ')
    return dict(boxes_and_velocity=metrics(boxes, actual), scores=metrics(expected['scores'], cpp['scores']),
                labels_exact=True, policy='model.yaml deployment policy',
                reference='NumPy decode and OpenCV rotated-rectangle overlap')


def independent_ops(model, manifest, internal, pooling_dtype='float32'):
    """Check the exported sample using NumPy voxelization and original geometry/pooling operators.

    manifest points to sample metadata containing a point NPY and calibration;
    internal contains C++ tensors. Rebuild voxels using inclusive lower/exclusive upper ROI bounds and
    source order, requiring exact [batch,z,y,x] indices and means after FP32
    reduction and FP16 conversion. Feed deployed camera features and depth
    probabilities into the original model's geometry/pooling path, comparing
    pooled output with atol=1e-4 and rtol=1e-3. Return index/mean agreement and
    pooling errors, or raise AssertionError.
    """
    from collections import OrderedDict
    node = read(manifest)
    from .points import load_points
    pts = load_points(manifest.parent / node['point_cloud'])
    layer = model.data_preprocessor.voxel_layer
    lo = np.asarray(layer.point_cloud_range[:3], dtype=np.float32)
    hi = np.asarray(layer.point_cloud_range[3:], dtype=np.float32)
    voxel = np.asarray(layer.voxel_size, dtype=np.float32)
    groups = OrderedDict()
    for point in pts:
        if np.all(point[:3] >= lo) and np.all(point[:3] < hi):
            key = tuple(np.floor((point[:3] - lo) / voxel).astype(np.int32))
            if key not in groups:
                if len(groups) >= layer.max_voxels[1]:
                    continue
                groups[key] = []
            if len(groups[key]) < layer.max_num_points:
                groups[key].append(point)
    indices = np.asarray([(0,) + (k[::-1]) for k in groups], dtype=np.float32)
    actual_idx = np.load(internal / 'voxel_indices.npy')
    if not np.array_equal(indices, actual_idx):
        raise AssertionError('Deterministic voxel coordinates/order differ')
    means = np.asarray([np.asarray(p, np.float32).sum(axis=0, dtype=np.float32) / np.float32(len(p)) for p in groups.values()]).astype(np.float16).astype(np.float32)
    actual_mean = np.load(internal / 'voxel_features.npy')
    if not np.array_equal(means, actual_mean):
        raise AssertionError('Deterministic voxel FP16 means differ')
    cams = node['cameras']
    def matrices(key):
        """Collect calibration matrices in manifest camera order, add a batch axis, and move to CUDA."""
        return torch.tensor([[c[key] if c.get('enabled', True) else np.eye(4).tolist() for c in cams]], dtype=torch.float32, device='cuda')
    c2l, intr, aug = matrices('cam2lidar'), matrices('cam2img'), matrices('img_aug_matrix')
    la = torch.tensor([node['lidar_aug_matrix']], device='cuda')
    with torch.inference_mode():
        geom = model.view_transform.get_geometry(c2l[..., :3, :3], c2l[..., :3, 3], intr[..., :3, :3], aug[..., :3, :3], aug[..., :3, 3],
                                                  extra_rots=la[..., :3, :3], extra_trans=la[..., :3, 3])
        feat = torch.from_numpy(np.load(internal / 'camera_feature.npy')).cuda()
        dep = torch.from_numpy(np.load(internal / 'camera_depth_weights.npy')).cuda()
        weighted = dep.unsqueeze(-1) * feat.unsqueeze(1)
        mask = torch.tensor([[c.get('enabled', True) for c in cams]], dtype=torch.bool, device='cuda')
        pooled = model.view_transform.bev_pool(geom, weighted.unsqueeze(0), mask).cpu().numpy()
    pooled = pooled.astype(pooling_dtype).astype(np.float32)
    cpp = np.load(internal / 'camera_bev_before_downsample.npy')
    report = dict(voxel_indices_exact=True, voxel_means_exact=True, pooling=metrics(pooled, cpp))
    if not np.allclose(pooled, cpp, atol=1e-4, rtol=1e-3):
        raise AssertionError('Independent geometry/pooling parity failed: ' + str(report))
    return report


def clean_json(value):
    """Normalize report data recursively, converting NumPy scalars and replacing NaN/Inf with None.

    Recurse through dictionaries and lists/tuples, emitting tuples as lists. NumPy
    arrays are not converted here. Return a new structure suitable for standard
    JSON serialization with allow_nan=False.
    """
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
