#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
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
    with (model_dir / "model.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_optional_array(path: Path, dtype):
    if not path.exists():
        return np.empty((0,), dtype=dtype)
    return np.load(path)


def load_optional_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def reshape_boxes(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    if boxes.ndim == 1:
        if boxes.size % 7 != 0:
            raise ValueError(f"Expected boxes to contain a multiple of 7 values, got {boxes.size}")
        return boxes.reshape(-1, 7)
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"Expected boxes to have shape [N, >=7], got {boxes.shape}")
    return boxes[:, :7].astype(np.float32, copy=False)


def reshape_scores(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return np.empty((0,), dtype=np.float32)
    return scores.reshape(-1)


def reshape_labels(labels):
    labels = np.asarray(labels, dtype=np.int32)
    if labels.size == 0:
        return np.empty((0,), dtype=np.int32)
    return labels.reshape(-1)


def label_name_from_config(label_names, zero_based_label):
    if zero_based_label < 0:
        return f"class_{zero_based_label}"
    if zero_based_label < len(label_names):
        return str(label_names[zero_based_label])
    return f"class_{zero_based_label}"


def color_for_label(zero_based_label):
    if zero_based_label < 0:
        return "#ffffff"
    return DEFAULT_COLORS[zero_based_label % len(DEFAULT_COLORS)]


def build_decoded_entries(boxes, scores, labels, label_names):
    boxes = reshape_boxes(boxes)
    scores = reshape_scores(scores)
    labels = reshape_labels(labels)
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError(
            "Decoded arrays must have the same length: "
            f"boxes={len(boxes)}, scores={len(scores)}, labels={len(labels)}"
        )

    entries = []
    for box, score, label in zip(boxes, scores, labels):
        label = int(label)
        entries.append(
            {
                "box": box,
                "score": float(score),
                "label_id": label,
                "label_name": label_name_from_config(label_names, label),
                "source": "decoded",
            }
        )
    return entries


def build_final_entries_from_result(result_json):
    objects = (result_json or {}).get("objects", [])
    entries = []
    for obj in objects:
        box = reshape_boxes(np.asarray(obj.get("box", []), dtype=np.float32))
        if len(box) != 1:
            continue
        zero_based_label = int(obj.get("label", -1))
        label_name = obj.get("label_name") or f"class_{zero_based_label}"
        entries.append(
            {
                "box": box[0],
                "score": float(obj.get("score", 0.0)),
                "label_id": zero_based_label,
                "label_name": str(label_name),
                "source": "final",
            }
        )
    return entries


def build_final_entries_from_npy(boxes, scores, labels, label_names):
    entries = build_decoded_entries(boxes, scores, labels, label_names)
    for entry in entries:
        entry["source"] = "final"
    return entries


def box_3d_corners(box):
    x, y, z, dx, dy, dz, yaw = box[:7]

    local = np.array(
        [
            [dx / 2, dy / 2, -dz / 2],
            [dx / 2, -dy / 2, -dz / 2],
            [-dx / 2, -dy / 2, -dz / 2],
            [-dx / 2, dy / 2, -dz / 2],
            [dx / 2, dy / 2, dz / 2],
            [dx / 2, -dy / 2, dz / 2],
            [-dx / 2, -dy / 2, dz / 2],
            [-dx / 2, dy / 2, dz / 2],
        ],
        dtype=np.float32,
    )

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    corners = local @ rot.T
    corners += np.array([x, y, z], dtype=np.float32)
    return corners


def add_3d_box(fig, box, score, label_name, label_id, source_name, show_legend):
    color = color_for_label(int(label_id))
    corners = box_3d_corners(box)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [corners[i, 0], corners[j, 0], None]
        ys += [corners[i, 1], corners[j, 1], None]
        zs += [corners[i, 2], corners[j, 2], None]

    hover_text = f"{source_name}: {label_name} {float(score):.2f}"
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=5),
            name=label_name,
            legendgroup=label_name,
            showlegend=show_legend,
            text=hover_text,
            hoverinfo="text",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[float(box[0])],
            y=[float(box[1])],
            z=[float(box[2])],
            mode="text",
            text=[hover_text],
            textfont=dict(color=color, size=10),
            showlegend=False,
            hoverinfo="skip",
        )
    )


def add_entries(fig, entries, shown_labels):
    for entry in entries:
        box = np.asarray(entry["box"], dtype=np.float32)
        if box.shape[0] < 7:
            continue
        label_name = str(entry["label_name"])
        show_legend = label_name not in shown_labels
        if show_legend:
            shown_labels.add(label_name)
        add_3d_box(
            fig,
            box,
            entry["score"],
            label_name,
            entry["label_id"],
            entry["source"],
            show_legend,
        )


def add_points(fig, points):
    if not points.size:
        return
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected input_points.npy to have shape [N, >=3], got {points.shape}")

    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=1.5,
                color=points[:, 2],
                colorscale="Viridis",
                opacity=0.65,
            ),
            name="input_points",
        )
    )


def add_pc_range_box(fig, pc_range):
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range

    corners = np.array(
        [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ],
        dtype=np.float32,
    )

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [corners[i, 0], corners[j, 0], None]
        ys += [corners[i, 1], corners[j, 1], None]
        zs += [corners[i, 2], corners[j, 2], None]

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color="#888888", width=2),
            name="point_cloud_range",
        )
    )


def add_heatmap_floor(fig, cls_score, pc_range):
    if cls_score.size == 0:
        return False

    cls_score = np.nan_to_num(cls_score, nan=-100.0, posinf=100.0, neginf=-100.0)
    if cls_score.ndim == 4:
        cls_score_for_vis = cls_score[0]
    elif cls_score.ndim == 3:
        cls_score_for_vis = cls_score
    else:
        raise ValueError(
            f"Expected cls_score.npy to have shape [B, C, H, W] or [C, H, W], got {cls_score.shape}"
        )

    heatmap = 1.0 / (1.0 + np.exp(-np.clip(cls_score_for_vis, -50.0, 50.0)))
    heatmap = heatmap.max(axis=0)

    x_min, y_min, z_min, x_max, y_max, _z_max = pc_range
    h, w = heatmap.shape
    xs = np.linspace(x_min, x_max, w)
    ys = np.linspace(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, z_min)

    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            surfacecolor=heatmap,
            colorscale="Magma",
            opacity=0.45,
            showscale=True,
            name="heatmap_floor",
        )
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show_decoded", action="store_true")
    parser.add_argument("--show_heatmap", action="store_true")
    args = parser.parse_args()

    cfg = load_model_config(args.model_dir)
    dump_dir = args.dump_dir
    pc_range = cfg["model"]["point_cloud_range"]
    label_names = list(cfg.get("model", {}).get("label_names", []))

    points = load_optional_array(dump_dir / "input_points.npy", np.float32)
    cls_score = load_optional_array(dump_dir / "cls_score.npy", np.float32)
    decoded_boxes = load_optional_array(dump_dir / "decoded_boxes.npy", np.float32)
    decoded_scores = load_optional_array(dump_dir / "decoded_scores.npy", np.float32)
    decoded_labels = load_optional_array(dump_dir / "decoded_labels.npy", np.int32)
    final_nms_boxes = load_optional_array(dump_dir / "final_nms_boxes.npy", np.float32)
    final_nms_scores = load_optional_array(dump_dir / "final_nms_scores.npy", np.float32)
    final_nms_labels = load_optional_array(dump_dir / "final_nms_labels.npy", np.int32)
    result_json = load_optional_json(dump_dir / "result.json")

    decoded_entries = build_decoded_entries(decoded_boxes, decoded_scores, decoded_labels, label_names)
    final_entries = build_final_entries_from_result(result_json)
    if not final_entries:
        final_entries = build_final_entries_from_npy(
            final_nms_boxes, final_nms_scores, final_nms_labels, label_names
        )

    fig = go.Figure()
    add_points(fig, points)
    add_pc_range_box(fig, pc_range)

    if args.show_heatmap:
        heatmap_added = add_heatmap_floor(fig, cls_score, pc_range)
        if not heatmap_added:
            print(f"[WARN] skipped heatmap: missing {(dump_dir / 'cls_score.npy')}")

    shown_labels = set()
    if args.show_decoded:
        add_entries(fig, decoded_entries, shown_labels)
    add_entries(fig, final_entries, shown_labels)

    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    fig.update_layout(
        title="3D Detection Result",
        scene=dict(
            xaxis=dict(title="X", range=[x_min, x_max]),
            yaxis=dict(title="Y", range=[y_min, y_max]),
            zaxis=dict(title="Z", range=[z_min, z_max]),
            aspectmode="data",
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    output = args.output or (dump_dir / "3d_result.html")
    fig.write_html(str(output), include_plotlyjs="cdn")
    print(f"[OK] saved html: {output}")


if __name__ == "__main__":
    main()
