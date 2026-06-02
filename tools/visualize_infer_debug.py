#!/usr/bin/env python3
import argparse
from copy import copy
import html
import json
import os
from pathlib import Path

import numpy as np
import yaml

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

OPTIONAL_ARRAYS = [
    "input_points.npy",
    "final_boxes.npy",
    "final_scores.npy",
    "final_labels.npy",
    "decoded_boxes.npy",
    "decoded_scores.npy",
    "decoded_labels.npy",
    "internal_decoded_boxes.npy",
    "internal_decoded_scores.npy",
    "internal_decoded_labels.npy",
    "cls_score.npy",
    "bbox_pred.npy",
    "dir_cls_pred.npy",
    "pillar_voxels.npy",
    "pillar_num_points.npy",
    "pillar_coors.npy",
]


def load_model_config(model_dir: Path):
    with (model_dir / "model.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_optional_array(path: Path):
    if not path.exists():
        return np.empty((0,), dtype=np.float32)
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


def reshape_vector(values, dtype):
    values = np.asarray(values, dtype=dtype)
    if values.size == 0:
        return np.empty((0,), dtype=dtype)
    return values.reshape(-1)


def label_names_from_config(cfg):
    postprocess_names = cfg.get("postprocess", {}).get("mapped_label_names", [])
    if postprocess_names:
        return [str(name) for name in postprocess_names]
    return [str(name) for name in cfg.get("model", {}).get("label_names", [])]


def global_label_names_from_config(cfg):
    return [str(name) for name in cfg.get("model", {}).get("label_names", [])]


def label_name(label_names, label_id):
    label_id = int(label_id)
    if label_id < 0:
        return f"class_{label_id}"
    if label_id < len(label_names):
        return str(label_names[label_id])
    return f"class_{label_id}"


def color_for_label(label_id):
    label_id = int(label_id)
    if label_id < 0:
        return "#ffffff"
    return DEFAULT_COLORS[label_id % len(DEFAULT_COLORS)]


def make_extent(pc_range):
    return [pc_range[0], pc_range[3], pc_range[1], pc_range[4]]


def make_grid_shape(pc_range, resolution):
    width = max(1, int(np.ceil((pc_range[3] - pc_range[0]) / resolution)))
    height = max(1, int(np.ceil((pc_range[4] - pc_range[1]) / resolution)))
    return height, width


def filter_points_by_z(points, zmin, zmax):
    if not points.size:
        return points
    return points[(points[:, 2] >= zmin) & (points[:, 2] <= zmax)]


def build_entries_from_arrays(boxes, scores, labels, label_names, source, display_names=None):
    boxes = reshape_boxes(boxes)
    scores = reshape_vector(scores, np.float32)
    labels = reshape_vector(labels, np.int32)
    if len(boxes) == 0 and len(scores) == 0 and len(labels) == 0:
        return []
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError(
            f"{source} arrays length mismatch: "
            f"boxes={len(boxes)}, scores={len(scores)}, labels={len(labels)}"
        )

    entries = []
    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        name = None
        if display_names is not None and idx < len(display_names):
            name = str(display_names[idx])
        entries.append(
            {
                "box": box,
                "score": float(score),
                "label_id": int(label),
                "label_name": name or label_name(label_names, label),
                "source": source,
            }
        )
    return entries


def build_final_entries(result_json, dump_dir, label_names):
    entries = []
    for obj in (result_json or {}).get("objects", []):
        box = reshape_boxes(np.asarray(obj.get("box", []), dtype=np.float32))
        if len(box) != 1:
            continue
        label_id = int(obj.get("label", -1))
        entries.append(
            {
                "box": box[0],
                "score": float(obj.get("score", 0.0)),
                "label_id": label_id,
                "label_name": str(obj.get("label_name") or label_name(label_names, label_id)),
                "source": "final",
            }
        )
    if entries:
        return entries

    final_names = load_optional_json(dump_dir / "final_label_names.json")
    if not isinstance(final_names, list):
        final_names = None
    return build_entries_from_arrays(
        load_optional_array(dump_dir / "final_boxes.npy"),
        load_optional_array(dump_dir / "final_scores.npy"),
        load_optional_array(dump_dir / "final_labels.npy"),
        label_names,
        "final",
        final_names,
    )


def top_entries(entries, limit):
    if limit <= 0 or len(entries) <= limit:
        return list(entries)
    return sorted(entries, key=lambda item: item["score"], reverse=True)[:limit]


def box_corners_bev(box):
    x, y, _z, dx, dy, _dz, yaw = box[:7]
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


def draw_heading_bev(ax, box, color, *, linewidth, alpha):
    x, y, _z, dx, _dy, _dz, yaw = box[:7]
    heading_length = max(float(dx) * 0.45, 0.25)
    end_x = float(x) + heading_length * float(np.cos(yaw))
    end_y = float(y) + heading_length * float(np.sin(yaw))
    ax.annotate(
        "",
        xy=(end_x, end_y),
        xytext=(float(x), float(y)),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": linewidth,
            "alpha": alpha,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
            "mutation_scale": 8,
        },
        zorder=5,
    )


def draw_point_grid(ax, points, pc_range, resolution, *, mode, cmap_name, alpha=1.0):
    if not points.size:
        return
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected input_points.npy to have shape [N, >=3], got {points.shape}")

    height, width = make_grid_shape(pc_range, resolution)
    hist_range = [[pc_range[1], pc_range[4]], [pc_range[0], pc_range[3]]]
    y = points[:, 1]
    x = points[:, 0]
    counts, _, _ = np.histogram2d(y, x, bins=[height, width], range=hist_range)

    if mode == "occupancy":
        image = np.zeros((height, width, 4), dtype=np.float32)
        image[..., :3] = 0.15
        image[..., 3] = (counts > 0) * alpha
        ax.imshow(image, origin="lower", extent=make_extent(pc_range), interpolation="nearest")
        return

    if mode == "height":
        z_sum, _, _ = np.histogram2d(y, x, bins=[height, width], range=hist_range, weights=points[:, 2])
        grid = np.divide(z_sum, counts, out=np.zeros_like(z_sum), where=counts > 0)
    else:
        grid = np.log1p(counts)

    grid = np.ma.masked_where(counts <= 0, grid)
    cmap = copy(plt.get_cmap(cmap_name))
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    ax.imshow(
        grid,
        origin="lower",
        extent=make_extent(pc_range),
        cmap=cmap,
        alpha=alpha,
        interpolation="nearest",
    )


def draw_entries_bev(
    ax,
    entries,
    *,
    linewidth,
    alpha,
    font_size,
    score_threshold=0.0,
    show_heading=False,
):
    used_labels = {}
    for entry in entries:
        if entry["score"] < score_threshold:
            continue
        box = np.asarray(entry["box"], dtype=np.float32)
        if box.shape[0] < 7:
            continue
        label_id = int(entry["label_id"])
        name = str(entry["label_name"])
        color = color_for_label(label_id)
        corners = box_corners_bev(box)
        ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=linewidth, alpha=alpha)
        if show_heading:
            draw_heading_bev(ax, box, color, linewidth=linewidth + 0.4, alpha=min(alpha + 0.05, 1.0))
        ax.text(
            float(box[0]),
            float(box[1]),
            f"{name} {entry['score']:.2f}",
            color=color,
            fontsize=font_size,
            ha="left",
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.0, "edgecolor": "none"},
        )
        used_labels.setdefault(label_id, name)
    return used_labels


def add_legend(ax, label_name_by_id):
    if not label_name_by_id:
        return
    handles = [
        Line2D([0], [0], color=color_for_label(label_id), lw=2, label=name)
        for label_id, name in sorted(label_name_by_id.items())
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.85)


def setup_bev_axes(ax, pc_range, title):
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")


def save_figure(fig, path: Path):
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.06, top=0.94)
    fig.savefig(path)
    plt.close(fig)
    return path


def heatmap_from_cls_score(cls_score):
    if cls_score.size == 0:
        return None
    cls_score = np.nan_to_num(cls_score, nan=-100.0, posinf=100.0, neginf=-100.0)
    if cls_score.ndim == 4:
        cls_score = cls_score[0]
    if cls_score.ndim != 3:
        raise ValueError(
            f"Expected cls_score.npy to have shape [B, C, H, W] or [C, H, W], got {cls_score.shape}"
        )
    heatmap = 1.0 / (1.0 + np.exp(-np.clip(cls_score, -50.0, 50.0)))
    return heatmap.max(axis=0)


def generate_bev_outputs(args, pc_range, points, final_entries, decoded_entries, internal_entries, cls_score):
    generated = []
    out_dir = args.output_dir

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    draw_point_grid(ax, points, pc_range, args.point_resolution, mode="height", cmap_name="viridis")
    setup_bev_axes(ax, pc_range, "Input Points Height BEV")
    generated.append(save_figure(fig, out_dir / "01_input_points_height_bev.png"))

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    draw_point_grid(ax, points, pc_range, args.point_resolution, mode="density", cmap_name="gray_r")
    setup_bev_axes(ax, pc_range, "Input Points Density BEV")
    generated.append(save_figure(fig, out_dir / "02_input_points_density_bev.png"))

    heatmap = heatmap_from_cls_score(cls_score)
    if heatmap is not None:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
        ax.imshow(heatmap, origin="lower", extent=make_extent(pc_range), cmap="magma")
        setup_bev_axes(ax, pc_range, "Classification Heatmap BEV")
        generated.append(save_figure(fig, out_dir / "03_cls_heatmap_bev.png"))

    candidate_entries = top_entries(decoded_entries, args.max_candidates)
    internal_candidate_entries = top_entries(internal_entries, args.max_candidates)
    if candidate_entries or internal_candidate_entries:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
        draw_point_grid(ax, points, pc_range, args.point_resolution, mode="occupancy", cmap_name="gray_r", alpha=0.25)
        label_names = {}
        label_names.update(
            draw_entries_bev(
                ax,
                internal_candidate_entries,
                linewidth=0.6,
                alpha=0.45,
                font_size=5,
                score_threshold=args.candidate_score_threshold,
            )
        )
        label_names.update(
            draw_entries_bev(
                ax,
                candidate_entries,
                linewidth=0.8,
                alpha=0.75,
                font_size=5,
                score_threshold=args.candidate_score_threshold,
            )
        )
        add_legend(ax, label_names)
        setup_bev_axes(ax, pc_range, "Decoded Candidates BEV")
        generated.append(save_figure(fig, out_dir / "04_decoded_candidates_bev.png"))

    final_points = filter_points_by_z(points, args.zmin, args.zmax)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    draw_point_grid(ax, final_points, pc_range, args.point_resolution, mode="occupancy", cmap_name="gray_r", alpha=0.35)
    final_label_names = draw_entries_bev(
        ax,
        final_entries,
        linewidth=1.2,
        alpha=0.95,
        font_size=6,
        score_threshold=args.final_score_threshold,
        show_heading=True,
    )
    add_legend(ax, final_label_names)
    setup_bev_axes(ax, pc_range, "Final Boxes BEV")
    generated.append(save_figure(fig, out_dir / "05_final_boxes_bev.png"))

    return generated


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
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    corners = local @ rot.T
    corners += np.array([x, y, z], dtype=np.float32)
    return corners


def add_points_3d(fig, go, points, max_points):
    if not points.size:
        return
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected input_points.npy to have shape [N, >=3], got {points.shape}")

    if max_points > 0 and len(points) > max_points:
        stride = int(np.ceil(len(points) / max_points))
        points = points[::stride]

    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=points[:, 2], colorscale="Viridis", opacity=0.65),
            name="input_points",
        )
    )


def add_box_3d(fig, go, entry, shown_labels, *, width, opacity):
    box = np.asarray(entry["box"], dtype=np.float32)
    if box.shape[0] < 7:
        return
    label_id = int(entry["label_id"])
    label_name_text = str(entry["label_name"])
    source = str(entry["source"])
    color = color_for_label(label_id)
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

    legend_name = f"{source}:{label_name_text}"
    hover = f"{source}: {label_name_text} {entry['score']:.3f}"
    show_legend = legend_name not in shown_labels
    shown_labels.add(legend_name)
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=width),
            opacity=opacity,
            name=legend_name,
            legendgroup=legend_name,
            showlegend=show_legend,
            text=hover,
            hoverinfo="text",
        )
    )
    if source == "final":
        fig.add_trace(
            go.Scatter3d(
                x=[float(box[0])],
                y=[float(box[1])],
                z=[float(box[2])],
                mode="text",
                text=[hover],
                textfont=dict(color=color, size=10),
                showlegend=False,
                hoverinfo="skip",
            )
        )


def add_pc_range_box_3d(fig, go, pc_range):
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


def add_heatmap_floor_3d(fig, go, cls_score, pc_range):
    heatmap = heatmap_from_cls_score(cls_score)
    if heatmap is None:
        return False

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


def generate_3d_output(args, pc_range, points, final_entries, decoded_entries, internal_entries, cls_score):
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        return None, "plotly is not installed in the active Python environment"

    fig = go.Figure()
    add_points_3d(fig, go, points, args.max_3d_points)
    add_pc_range_box_3d(fig, go, pc_range)

    if args.show_heatmap_3d and cls_score.size:
        add_heatmap_floor_3d(fig, go, cls_score, pc_range)

    shown_labels = set()
    if args.show_internal_3d:
        for entry in top_entries(internal_entries, args.max_candidates):
            if entry["score"] >= args.candidate_score_threshold:
                add_box_3d(fig, go, entry, shown_labels, width=3, opacity=0.35)
    if args.show_decoded_3d:
        for entry in top_entries(decoded_entries, args.max_candidates):
            if entry["score"] >= args.candidate_score_threshold:
                add_box_3d(fig, go, entry, shown_labels, width=4, opacity=0.60)
    for entry in final_entries:
        if entry["score"] >= args.final_score_threshold:
            add_box_3d(fig, go, entry, shown_labels, width=6, opacity=0.95)

    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    fig.update_layout(
        title="Infer Debug Final 3D Result",
        scene=dict(
            xaxis=dict(title="X", range=[x_min, x_max]),
            yaxis=dict(title="Y", range=[y_min, y_max]),
            zaxis=dict(title="Z", range=[z_min, z_max]),
            aspectmode="data",
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    output = args.output_dir / "06_3d_final_result.html"
    fig.write_html(str(output), include_plotlyjs="cdn")
    return output, None


def describe_array(path: Path):
    if not path.exists():
        return {"exists": False}
    arr = np.load(path, mmap_mode="r")
    return {
        "exists": True,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def write_index_html(output_dir, generated, summary, warnings):
    images = [path for path in generated if path.suffix.lower() == ".png"]
    html_path = output_dir / "index.html"
    result = summary.get("result", {})
    rows = []
    for name, info in summary["arrays"].items():
        if info.get("exists"):
            rows.append(
                "<tr>"
                f"<td>{html.escape(name)}</td>"
                f"<td>{html.escape(str(info.get('shape')))}</td>"
                f"<td>{html.escape(str(info.get('dtype')))}</td>"
                "</tr>"
            )

    cards = []
    for image in images:
        rel = image.name
        cards.append(
            "<section>"
            f"<h2>{html.escape(rel)}</h2>"
            f"<img src=\"{html.escape(rel)}\" alt=\"{html.escape(rel)}\">"
            "</section>"
        )

    three_d = ""
    if summary.get("generated", {}).get("3d_html"):
        name = Path(summary["generated"]["3d_html"]).name
        three_d = f"<p><a href=\"{html.escape(name)}\">Open final-result 3D view</a></p>"

    warning_html = "".join(f"<li>{html.escape(item)}</li>" for item in warnings)
    if warning_html:
        warning_html = f"<ul>{warning_html}</ul>"

    html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                "<meta charset=\"utf-8\">",
                "<title>Infer Debug Visualization</title>",
                "<style>",
                "body{font-family:Arial,sans-serif;margin:24px;background:#111;color:#eee}",
                "a{color:#6bd5ff} table{border-collapse:collapse;margin:16px 0}",
                "td,th{border:1px solid #444;padding:6px 10px} img{max-width:100%;height:auto}",
                "section{margin:28px 0} .meta{color:#bbb}",
                "</style>",
                "</head>",
                "<body>",
                "<h1>Infer Debug Visualization</h1>",
                (
                    f"<p class=\"meta\">source: {html.escape(str(result.get('result_source', 'unknown')))}, "
                    f"final boxes: {html.escape(str(result.get('final_count', 0)))}, "
                    f"decoded boxes: {html.escape(str(result.get('decoded_count', 0)))}, "
                    f"internal decoded boxes: {html.escape(str(result.get('internal_decoded_count', 0)))}</p>"
                ),
                warning_html,
                three_d,
                "<h2>Arrays</h2>",
                "<table><tr><th>file</th><th>shape</th><th>dtype</th></tr>",
                *rows,
                "</table>",
                *cards,
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )
    return html_path


def write_summary(output_dir, result_json, metadata_json, arrays, generated, warnings, final_entries, decoded_entries, internal_entries):
    summary = {
        "tool": "visualize_infer_debug.py",
        "result": {
            "result_source": (result_json or {}).get("result_source", "unknown"),
            "final_count": len(final_entries),
            "decoded_count": len(decoded_entries),
            "internal_decoded_count": len(internal_entries),
        },
        "metadata": metadata_json or {},
        "arrays": arrays,
        "generated": {
            "files": [str(path) for path in generated],
            "3d_html": str(next((path for path in generated if path.suffix.lower() == ".html" and path.name != "index.html"), "")),
        },
        "warnings": warnings,
    }
    path = output_dir / "visualization_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize artifacts produced by infer_debug_tool."
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--point-resolution", type=float, default=0.05)
    parser.add_argument("--zmin", type=float, default=-0.5)
    parser.add_argument("--zmax", type=float, default=2.5)
    parser.add_argument("--final-score-threshold", type=float, default=0.0)
    parser.add_argument("--candidate-score-threshold", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--max-3d-points", type=int, default=80000)
    parser.add_argument(
        "--show-decoded-3d",
        action="store_true",
        help="Overlay high-level API decoded pre-final candidates in the 3D HTML view.",
    )
    parser.add_argument(
        "--show-internal-3d",
        action="store_true",
        help="Overlay internal decoded candidates in the 3D HTML view.",
    )
    parser.add_argument(
        "--show-heatmap-3d",
        action="store_true",
        help="Overlay cls_score heatmap floor in the 3D HTML view.",
    )
    parser.add_argument("--no-bev", action="store_true")
    parser.add_argument("--no-3d", action="store_true")
    args = parser.parse_args()

    if args.point_resolution <= 0.0:
        parser.error("--point-resolution must be positive")
    if args.zmin > args.zmax:
        parser.error("--zmin must be less than or equal to --zmax")
    if args.max_candidates < 0:
        parser.error("--max-candidates must be non-negative")
    if args.max_3d_points < 0:
        parser.error("--max-3d-points must be non-negative")
    if args.output_dir is None:
        args.output_dir = args.dump_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def main():
    args = parse_args()
    cfg = load_model_config(args.model_dir)
    pc_range = cfg["model"]["point_cloud_range"]
    final_labels = label_names_from_config(cfg)
    global_labels = global_label_names_from_config(cfg)
    dump_dir = args.dump_dir

    points = load_optional_array(dump_dir / "input_points.npy")
    cls_score = load_optional_array(dump_dir / "cls_score.npy")
    result_json = load_optional_json(dump_dir / "result.json")
    metadata_json = load_optional_json(dump_dir / "metadata.json")

    final_entries = build_final_entries(result_json, dump_dir, final_labels)
    decoded_entries = build_entries_from_arrays(
        load_optional_array(dump_dir / "decoded_boxes.npy"),
        load_optional_array(dump_dir / "decoded_scores.npy"),
        load_optional_array(dump_dir / "decoded_labels.npy"),
        global_labels,
        "decoded",
    )
    internal_entries = build_entries_from_arrays(
        load_optional_array(dump_dir / "internal_decoded_boxes.npy"),
        load_optional_array(dump_dir / "internal_decoded_scores.npy"),
        load_optional_array(dump_dir / "internal_decoded_labels.npy"),
        global_labels,
        "internal_decoded",
    )

    warnings = []
    if not points.size:
        warnings.append("missing input_points.npy; point cloud layers are empty")
    if not decoded_entries:
        warnings.append("missing decoded_* arrays; rerun infer_debug_tool with --dump-decoded to visualize decoded candidates")
    if not internal_entries and not cls_score.size:
        warnings.append("missing internal debug arrays; rerun infer_debug_tool with --dump-internal to visualize heatmap/internal candidates")

    generated = []
    if not args.no_bev:
        generated.extend(
            generate_bev_outputs(
                args,
                pc_range,
                points,
                final_entries,
                decoded_entries,
                internal_entries,
                cls_score,
            )
        )

    if not args.no_3d:
        html_path, error = generate_3d_output(
            args,
            pc_range,
            points,
            final_entries,
            decoded_entries,
            internal_entries,
            cls_score,
        )
        if html_path is not None:
            generated.append(html_path)
        if error:
            warnings.append(f"skipped 3D visualization: {error}")

    arrays = {name: describe_array(dump_dir / name) for name in OPTIONAL_ARRAYS}
    summary_path, summary = write_summary(
        args.output_dir,
        result_json,
        metadata_json,
        arrays,
        generated,
        warnings,
        final_entries,
        decoded_entries,
        internal_entries,
    )
    index_path = write_index_html(args.output_dir, generated, summary, warnings)
    generated.extend([summary_path, index_path])

    for warning in warnings:
        print(f"[WARN] {warning}")
    for path in generated:
        print(f"[OK] saved {path}")


if __name__ == "__main__":
    main()
