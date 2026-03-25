#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml

LABEL_NAMES = {
    1: "car",
    2: "pedestrian",
    3: "cyclist",
}

LABEL_COLORS = {
    1: "#00d4ff",
    2: "#ff8c42",
    3: "#7cff6b",
}


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config_path(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return config_path.parent / candidate


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


def draw_boxes(ax, boxes, scores, labels, linewidth=1.0, alpha=0.8, font_size=6):
    legend_labels = []
    for box, score, label in zip(boxes, scores, labels):
        label = int(label)
        color = LABEL_COLORS.get(label, "#ffffff")
        name = LABEL_NAMES.get(label, f"class_{label}")
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


def add_legend(ax, label_ids):
    if not label_ids:
        return
    handles = [
        Line2D([0], [0], color=LABEL_COLORS[label_id], lw=2, label=LABEL_NAMES.get(label_id, str(label_id)))
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
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dump_dir = resolve_config_path(args.config, cfg["dump_dir"])
    pc_range = cfg["runtime"]["point_cloud_range"]
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
            linewidth=0.8,
            alpha=0.7,
            font_size=5,
        )
    add_legend(ax, decoded_legend)
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
            linewidth=1.2,
            alpha=0.95,
            font_size=6,
        )
    add_legend(ax, final_legend)
    setup_bev_axes(ax, pc_range, "Final Boxes BEV")
    save_figure(fig, dump_dir / "04_final_boxes_bev.png")


if __name__ == "__main__":
    main()
