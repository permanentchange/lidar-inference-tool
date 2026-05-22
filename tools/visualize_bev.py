#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml


DEFAULT_COLORS = [
    "#00d4ff",
    "#ff8c42",
    "#7cff6b",
    "#ff5d8f",
    "#ffd166",
    "#9b5de5",
    "#06d6a0",
    "#ef476f",
    "#118ab2",
    "#f4a261",
    "#8ecae6",
    "#fb8500",
]


def load_model_config(model_dir: Path):
    model_yaml = model_dir / "model.yaml"
    with model_yaml.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def label_name_from_config(label_names, label):
    label = int(label)
    if label < 0:
        return f"class_{label}"
    if label < len(label_names):
        return str(label_names[label])
    return f"class_{label}"


def color_for_label(label):
    label = int(label)
    if label < 0:
        return "#ffffff"
    return DEFAULT_COLORS[label % len(DEFAULT_COLORS)]


def make_extent(pc_range):
    return [pc_range[0], pc_range[3], pc_range[1], pc_range[4]]


def load_optional_array(path: Path, dtype):
    if not path.exists():
        return np.empty((0,), dtype=dtype)
    return np.load(path)


def box_corners(box):
    x, y, _z, dx, dy, _dz, yaw = box
    corners = np.array(
        [
            [dx / 2, dy / 2],
            [dx / 2, -dy / 2],
            [-dx / 2, -dy / 2],
            [-dx / 2, dy / 2],
            [dx / 2, dy / 2],
        ],
        dtype=np.float32,
    )
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    corners = corners @ rot.T
    corners[:, 0] += x
    corners[:, 1] += y
    return corners


def draw_boxes(ax, boxes, scores, labels, label_names, linewidth=1.0, alpha=0.8, font_size=6):
    legend_labels = []
    for box, score, label in zip(boxes, scores, labels):
        label = int(label)
        color = color_for_label(label)
        name = label_name_from_config(label_names, label)
        corners = box_corners(box)
        ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=linewidth, alpha=alpha)
        ax.text(
            float(box[0]),
            float(box[1]),
            f"{name} {float(score):.2f}",
            color=color,
            fontsize=font_size,
            ha="left",
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.0, "edgecolor": "none"},
        )
        legend_labels.append(label)
    return sorted(set(legend_labels))


def add_legend(ax, label_ids, label_names):
    if not label_ids:
        return
    handles = [
        Line2D(
            [0],
            [0],
            color=color_for_label(label_id),
            lw=2,
            label=label_name_from_config(label_names, label_id),
        )
        for label_id in label_ids
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.85)


def setup_bev_axes(ax, pc_range, title):
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")


def save_figure(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("dump_dir", type=Path)
    args = parser.parse_args()

    cfg = load_model_config(args.model_dir)
    dump_dir = args.dump_dir
    pc_range = cfg["model"]["point_cloud_range"]
    label_names = list(cfg.get("model", {}).get("label_names", []))
    extent = make_extent(pc_range)

    points = np.load(dump_dir / "input_points.npy")
    cls_score = np.load(dump_dir / "cls_score.npy")
    decoded_boxes = load_optional_array(dump_dir / "decoded_boxes.npy", np.float32)
    decoded_scores = load_optional_array(dump_dir / "decoded_scores.npy", np.float32)
    decoded_labels = load_optional_array(dump_dir / "decoded_labels.npy", np.int32)
    final_nms_boxes = load_optional_array(dump_dir / "final_nms_boxes.npy", np.float32)
    final_nms_scores = load_optional_array(dump_dir / "final_nms_scores.npy", np.float32)
    final_nms_labels = load_optional_array(dump_dir / "final_nms_labels.npy", np.int32)

    cls_score = np.nan_to_num(cls_score, nan=-100.0, posinf=100.0, neginf=-100.0)
    heatmap = 1.0 / (1.0 + np.exp(-np.clip(cls_score[0], -50.0, 50.0)))
    heatmap = heatmap.max(axis=0)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    if points.size:
        ax.scatter(points[:, 0], points[:, 1], s=0.5, c=points[:, 2], cmap="viridis")
    setup_bev_axes(ax, pc_range, "Input Points BEV")
    save_figure(fig, dump_dir / "01_input_points_bev.png")

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.imshow(heatmap, origin="lower", extent=extent, cmap="magma")
    setup_bev_axes(ax, pc_range, "Heatmap BEV")
    save_figure(fig, dump_dir / "02_heatmap_bev.png")

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    if points.size:
        ax.scatter(points[:, 0], points[:, 1], s=0.2, c="lightgray", alpha=0.5)
    decoded_legend = []
    if decoded_boxes.size and decoded_scores.size and decoded_labels.size:
        decoded_legend = draw_boxes(
            ax,
            decoded_boxes,
            decoded_scores,
            decoded_labels,
            label_names,
            linewidth=0.8,
            alpha=0.7,
            font_size=5,
        )
    add_legend(ax, decoded_legend, label_names)
    setup_bev_axes(ax, pc_range, "Decoded Candidates BEV")
    save_figure(fig, dump_dir / "03_decoded_candidates_bev.png")

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    if points.size:
        ax.scatter(points[:, 0], points[:, 1], s=0.2, c="lightgray", alpha=0.5)
    final_legend = []
    if final_nms_boxes.size and final_nms_scores.size and final_nms_labels.size:
        final_legend = draw_boxes(
            ax,
            final_nms_boxes,
            final_nms_scores,
            final_nms_labels,
            label_names,
            linewidth=1.2,
            alpha=0.95,
            font_size=6,
        )
    add_legend(ax, final_legend, label_names)
    setup_bev_axes(ax, pc_range, "Final Boxes BEV")
    save_figure(fig, dump_dir / "04_final_boxes_bev.png")


if __name__ == "__main__":
    main()
