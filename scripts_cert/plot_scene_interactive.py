import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import plotly.graph_objects as go


SCRIPT_DIR = Path(__file__).resolve().parent          # <root>/scripts_cert
PROJECT_ROOT = SCRIPT_DIR.parent                       # <root>
RENDER_DIR = PROJECT_ROOT / "scripts_render"

for path in (PROJECT_ROOT, RENDER_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from scripts_render.render_image import Config as RenderConfig, load_gsplat_scene
from coordinate_transform import transform_to_render_constants


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive 3D plot of certification boxes inside the 3DGS scene."
    )
    parser.add_argument(
        "--scene",
        choices=["uturn", "gate_long"],
        default="uturn",
        help="Scene name to plot.",
    )
    parser.add_argument(
        "--region",
        default="small",
        help="Region size tag used in the result filename.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Maximum number of Gaussian means to show as point cloud.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for point cloud subsampling.",
    )
    parser.add_argument(
        "--result",
        default=None,
        help="Optional explicit result .pt file. Defaults to results/{scene}_{region}_cert_result.pt.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output HTML file. Defaults to figures/{scene}_{region}_cert_scene_interactive.html.",
    )
    return parser.parse_args()


def certification_filename(scene_name, region_size):
    return f"{scene_name}_{region_size}_cert"


def load_certification_result(args):
    filename = certification_filename(args.scene, args.region)
    result_path = args.result
    if result_path is None:
        result_path = PROJECT_ROOT / "results" / f"{filename}_result.pt"
    else:
        result_path = Path(result_path)
        if not result_path.is_absolute():
            result_path = PROJECT_ROOT / result_path

    data = torch.load(result_path, map_location="cpu", weights_only=False)
    return data, result_path


def means_to_pose_frame(means, transform, scale):
    device = means.device
    dtype = means.dtype
    _, _, _, cam_const = transform_to_render_constants(
        means,
        transform,
        scale,
        device=device,
        dtype=dtype,
    )
    # transform_to_render_constants returns scale * position_in_pose_frame.
    # Divide by scale so point cloud and certified boxes share the same pose frame.
    points = cam_const.squeeze(-1) / scale
    return points


def sample_points(points, max_points, seed):
    num_points = points.shape[0]
    if max_points is None or max_points <= 0 or num_points <= max_points:
        return points

    generator = torch.Generator(device=points.device)
    generator.manual_seed(seed)
    idx = torch.randperm(num_points, generator=generator, device=points.device)[:max_points]
    return points[idx]


def box_edges(box):
    x0, x1 = box["x_lb"], box["x_ub"]
    y0, y1 = box["y_lb"], box["y_ub"]
    z0, z1 = box["z_lb"], box["z_ub"]

    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]

    edge_indices = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    xs, ys, zs = [], [], []
    for i, j in edge_indices:
        xs.extend([corners[i][0], corners[j][0], None])
        ys.extend([corners[i][1], corners[j][1], None])
        zs.extend([corners[i][2], corners[j][2], None])

    return xs, ys, zs


def collect_box_edges(boxes):
    xs, ys, zs = [], [], []
    for box in boxes:
        bx, by, bz = box_edges(box)
        xs.extend(bx)
        ys.extend(by)
        zs.extend(bz)
    return xs, ys, zs


def split_boxes_by_reason(boxes):
    failed = []
    threshold_verified = []
    decrease_verified = []

    for box in boxes:
        if not box["verified"]:
            failed.append(box)
        elif box.get("threshold_verified", False):
            threshold_verified.append(box)
        else:
            decrease_verified.append(box)

    return failed, threshold_verified, decrease_verified


def add_box_trace(fig, boxes, color, name, width=4):
    if len(boxes) == 0:
        fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color=color, width=width),
                name=f"{name} (0)",
            )
        )
        return

    xs, ys, zs = collect_box_edges(boxes)
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=width),
            name=f"{name} ({len(boxes)})",
        )
    )


def add_marker(fig, point, color, name, size=6, symbol="diamond"):
    fig.add_trace(
        go.Scatter3d(
            x=[point[0]],
            y=[point[1]],
            z=[point[2]],
            mode="markers",
            marker=dict(size=size, color=color, symbol=symbol),
            name=name,
        )
    )


def main():
    args = parse_args()
    filename = certification_filename(args.scene, args.region)

    data, result_path = load_certification_result(args)
    boxes = data["verified_boxes"]
    target = np.asarray(data["target"][:3], dtype=np.float64)
    gate = np.asarray(data["gate"][:3], dtype=np.float64)

    render_cfg = RenderConfig(scene_name=args.scene, device="cpu")
    scene = load_gsplat_scene(render_cfg)
    means, quats, opacities, scales, colors, transform, scale, world_frame = scene

    points = means_to_pose_frame(means.float().cpu(), transform, scale)
    points = sample_points(points, args.max_points, args.seed)
    points_np = points.cpu().numpy()

    failed_boxes, threshold_boxes, decrease_boxes = split_boxes_by_reason(boxes)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=points_np[:, 0],
            y=points_np[:, 1],
            z=points_np[:, 2],
            mode="markers",
            marker=dict(
                size=1.5,
                color="rgba(120,120,120,0.22)",
            ),
            name=f"3DGS points ({points_np.shape[0]})",
        )
    )

    add_box_trace(fig, decrease_boxes, "darkgreen", "Verified: V decreases", width=5)
    add_box_trace(fig, threshold_boxes, "lightgreen", "Verified: V <= threshold", width=5)
    add_box_trace(fig, failed_boxes, "red", "Failed", width=5)

    add_marker(fig, target, "red", "Target", size=7, symbol="diamond")
    add_marker(fig, gate, "black", "Gate", size=7, symbol="diamond")

    fig.update_layout(
        title=(
            f"{args.scene}: certified regions in 3DGS scene<br>"
            f"<sup>Loaded {result_path.relative_to(PROJECT_ROOT)}; "
            f"world_frame={world_frame}</sup>"
        ),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=70, b=0),
    )

    output_path = args.output
    if output_path is None:
        output_path = PROJECT_ROOT / "figures" / f"{filename}_scene_interactive.html"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")

    print(f"Loaded result : {result_path}")
    print(f"Scene         : {args.scene}")
    print(f"Region        : {args.region}")
    print(f"3DGS points   : {points_np.shape[0]}")
    print(f"Boxes         : {len(boxes)}")
    print(f"  decrease    : {len(decrease_boxes)}")
    print(f"  threshold   : {len(threshold_boxes)}")
    print(f"  failed      : {len(failed_boxes)}")
    print(f"Saved HTML    : {output_path}")


if __name__ == "__main__":
    main()
