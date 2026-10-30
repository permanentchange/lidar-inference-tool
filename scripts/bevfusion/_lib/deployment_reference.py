"""Independent NumPy/OpenCV reference for the shared deployment CenterHead policy."""
import numpy as np


def configure_voxelizer(model, package):
    """Use deployment caps when comparing network tensors, without changing weights."""
    layer = model.data_preprocessor.voxel_layer
    layer.max_num_points = int(package['runtime']['max_points_per_voxel'])
    limit = int(package['runtime']['max_voxels'])
    layer.max_voxels = (limit, limit)


def _overlap(a, b, mode):
    import cv2
    ra = ((float(a[0]), float(a[1])), (float(a[3]), float(a[4])), float(np.degrees(a[6])))
    rb = ((float(b[0]), float(b[1])), (float(b[3]), float(b[4])), float(np.degrees(b[6])))
    _, polygon = cv2.rotatedRectangleIntersection(ra, rb)
    intersection = 0. if polygon is None else abs(cv2.contourArea(polygon))
    area_a, area_b = float(a[3] * a[4]), float(b[3] * b[4])
    denominator = min(area_a, area_b) if mode == 'iou_min_bev' else area_a + area_b - intersection
    return intersection / max(denominator, 1e-8)


def postprocess(decoded, package, map_labels=True):
    """Threshold original labels, globally group/sort/NMS, then map output labels."""
    config = package['postprocess']
    boxes = np.asarray(decoded['boxes'], np.float32).reshape(-1, 9)
    scores = np.asarray(decoded['scores'], np.float32)
    labels = np.asarray(decoded['labels'], np.int64)
    thresholds = np.asarray(config.get('label_score_thresh', []), np.float32)
    groups = config.get('nms_label_groups') or [dict(labels=ids) for ids in package['model']['task_label_ids']]
    kept = []
    mode = config['nms_type']
    if mode not in ('iou_bev', 'iou_min_bev'):
        raise ValueError('Independent deployment reference requires an IoU BEV NMS mode')
    for group in groups:
        indices = np.flatnonzero(np.isin(labels, group['labels']))
        if thresholds.size:
            indices = indices[scores[indices] >= thresholds[labels[indices]]]
        indices = indices[np.argsort(-scores[indices], kind='stable')]
        indices = indices[:int(group.get('nms_pre_maxsize', config['nms_pre_maxsize']))]
        threshold = float(group.get('nms_thresh', config['nms_thresh']))
        if 'self_car_box' in config and len(indices):
            ego = np.asarray(config['self_car_box'], np.float32)
            indices = np.asarray([i for i in indices if _overlap(ego, boxes[i], mode)
                                  <= config['self_car_nms_thresh']], dtype=np.int64)
        selected = []
        while len(indices) and len(selected) < int(group.get('nms_post_maxsize', config['nms_post_maxsize'])):
            index = int(indices[0])
            selected.append(index)
            indices = np.asarray([i for i in indices[1:] if _overlap(boxes[index], boxes[i], mode)
                                  <= threshold], dtype=np.int64)
        kept.extend(i for i in selected if scores[i] >= config['final_score_threshold'])
    mapping = config.get('label_mapping', []) if map_labels else []
    if mapping:
        kept = [i for i in kept if mapping[labels[i]] >= 0]
    output_labels = [int(mapping[labels[i]] if mapping else labels[i]) for i in kept]
    return dict(boxes=boxes[kept].tolist(), scores=scores[kept].tolist(), labels=output_labels)


def decode(heads, package):
    """Decode task tensors with the LiDAR runtime's range-before-TopK ordering."""
    model, runtime = package['model'], package['runtime']
    all_boxes, all_scores, all_labels = [], [], []
    lower = np.asarray(model['post_center_range'][:3], np.float32)
    upper = np.asarray(model['post_center_range'][3:], np.float32)
    for task, task_labels in zip(heads, model['task_label_ids']):
        heat = np.asarray(task.get('hm', task.get('heatmap')), np.float32)[0]
        classes, height, width = heat.shape
        flat = heat.reshape(-1)
        scores = np.empty_like(flat)
        positive = flat >= 0
        scores[positive] = 1 / (1 + np.exp(-flat[positive]))
        exponent = np.exp(flat[~positive])
        scores[~positive] = exponent / (1 + exponent)
        reg = np.asarray(task['reg'], np.float32)[0].reshape(2, -1)
        dimensions = np.asarray(task['dim'], np.float32)[0].reshape(3, -1)
        if model['norm_bbox']:
            dimensions = np.exp(np.clip(dimensions, -20, 20))
        yy, xx = np.indices((height, width), dtype=np.float32)
        xy = np.stack((xx.reshape(-1), yy.reshape(-1))) + reg
        xy *= np.float32(model['out_size_factor'])
        xy *= np.asarray(model['voxel_size'][:2], np.float32)[:, None]
        xy += np.asarray(model['point_cloud_range'][:2], np.float32)[:, None]
        z = np.asarray(task['height'], np.float32)[0].reshape(1, -1)
        rotation = np.asarray(task['rot'], np.float32)[0].reshape(2, -1)
        yaw = np.arctan2(rotation[0], rotation[1])[None]
        velocity = np.asarray(task['vel'], np.float32)[0].reshape(2, -1)
        boxes = np.concatenate((xy, z, dimensions, yaw, velocity), axis=0).T
        spatial_valid = np.isfinite(boxes[:, :7]).all(axis=1)
        spatial_valid &= (boxes[:, :3] >= lower).all(axis=1) & (boxes[:, :3] <= upper).all(axis=1)
        valid = np.isfinite(flat) & np.isfinite(scores) & (scores > runtime['score_threshold'])
        valid &= np.tile(spatial_valid, classes)
        indices = np.flatnonzero(valid)
        indices = indices[np.argsort(-scores[indices], kind='stable')][:runtime['bbox_max_num']]
        all_boxes.extend(boxes[indices % (height * width)].tolist())
        all_scores.extend(scores[indices].tolist())
        all_labels.extend(int(task_labels[i // (height * width)]) for i in indices)
    return dict(boxes=all_boxes, scores=all_scores, labels=all_labels)


def predict(heads, package, map_labels=True):
    return postprocess(decode(heads, package), package, map_labels=map_labels)
