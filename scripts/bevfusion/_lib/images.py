"""Recover integer resize/crop metadata from the test pipeline for raw-image deployment."""
import json
from .common import read, write

def image_augmentation(cfg):
    """Find a deterministic ImageAug3D configuration recursively through dataset wrappers.

    Prefer test_dataloader, falling back to val_dataloader. Return the first
    matching augmentation dictionary without modifying cfg. Raise RuntimeError
    if it is missing or is_train is not false; never infer deployment settings
    from a random training augmentation.
    """
    # Follow the actual test pipeline, including wrappers, rather than deriving
    # integer sizes from the rounded float32 img_aug_matrix.
    def find(value):
        """Find the first ImageAug3D node in configuration traversal order, or return None."""
        if isinstance(value, dict):
            if (str(value.get('type', '')).endswith('ImageAug3D') or
                    value.get('type') == 'AspectRatioCropResize3D'):
                return value
            for v in value.values():
                found = find(v)
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for v in value:
                found = find(v)
                if found is not None:
                    return found
        return None
    dataloader = cfg.get('test_dataloader') or cfg.get('val_dataloader')
    aug = find(dataloader['dataset']) if dataloader is not None else None
    if aug is None or aug.get('is_train', True):
        raise RuntimeError('Expected deterministic test ImageAug3D pipeline')
    return aug


def add_preprocess(manifest, cfg):
    """Append per-camera integer resize/crop metadata and upgrade a manifest to version 2.

    manifest is a Path to JSON-encoded sample metadata, possibly with a .yaml
    extension. Resolve image paths relative to the manifest and read dimensions
    with Pillow. Reproduce test ImageAug3D resize/crop parameters, then check
    img_aug_matrix with atol=1e-6. Do not reconstruct integer dimensions from
    rounded matrices. Raise RuntimeError for incompatible augmentation. Replace
    the manifest only after every camera passes; do not modify image files.
    """
    import numpy as np
    from PIL import Image
    aug = image_augmentation(cfg)
    node = read(manifest)
    h, w = map(int, aug['final_dim'])
    for camera in node['cameras']:
        if not camera.get('enabled', True):
            continue
        with Image.open(manifest.parent / camera['image']) as image:
            iw, ih = image.size
        if aug['type'] == 'AspectRatioCropResize3D':
            from mmdet3d.registry import TRANSFORMS
            transform = TRANSFORMS.build(dict(aug))
            left, top, right, bottom = transform.crop_box((ih, iw))
            sx, sy = w / (right - left), h / (bottom - top)
            expected = np.eye(4, dtype=np.float32)
            expected[0, 0], expected[1, 1] = sx, sy
            expected[0, 3], expected[1, 3] = -left * sx, -top * sy
            if not np.allclose(camera['img_aug_matrix'], expected, atol=1e-5, rtol=0):
                raise RuntimeError('Crop-first augmentation matrix disagrees with test transform')
            camera['preprocess'] = dict(operation='crop_resize',
                source_crop=[left, top, right, bottom], resize_size=[w, h],
                crop_offset=[0, 0], interpolation='pillow_bicubic', color_order='RGB',
                exif_orientation='ignore_like_training_loader')
            continue
        scale = float(np.mean(aug['resize_lim']))
        rw, rh = int(iw * scale), int(ih * scale)
        x = int(max(0, rw - w) / 2)
        y = int((1 - np.mean(aug['bot_pct_lim'])) * rh) - h
        expected = np.eye(4, dtype=np.float32)
        expected[0, 0] = expected[1, 1] = scale
        expected[:2, 3] = [-x, -y]
        if not np.allclose(camera['img_aug_matrix'], expected, atol=1e-6, rtol=0):
            raise RuntimeError('Image augmentation is not deterministic resize/crop')
        camera['preprocess'] = dict(resize_size=[rw, rh], crop_offset=[x, y],
                                    interpolation='pillow_bicubic', color_order='RGB',
                                    exif_orientation='ignore_like_training_loader')
    node['version'] = 2
    write(manifest, node)
