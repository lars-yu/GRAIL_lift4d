#!/usr/bin/env python3
"""Render a top view from a saved, real-data fixed-camera HOI result."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grail.core.io import load_hoi_data
from grail.optimization.hoi_optimizer import HOIOptimizer
from grail.optimization.visualizer import HOIVisualizer


def _real_file(path: str, label: str) -> str:
    resolved = os.path.abspath(path)
    if not os.path.isfile(resolved) or os.path.getsize(resolved) == 0:
        raise FileNotFoundError(f"Missing required real {label}: {resolved}")
    return resolved


def _install_numpy_pickle_compat() -> None:
    """Allow NumPy 1.x to read array pickles written by NumPy 2.x."""
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video-file", required=True)
    parser.add_argument("--hmr-file", required=True)
    parser.add_argument("--mesh-file", required=True)
    parser.add_argument("--foundationpose-poses", required=True)
    parser.add_argument("--render-config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--optimized-hoi", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--diagnostics-csv", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    _install_numpy_pickle_compat()

    config_file = _real_file(args.config_file, "GRAIL config")
    video_file = _real_file(args.video_file, "RGB video")
    hmr_file = _real_file(args.hmr_file, "HMR motion")
    mesh_file = _real_file(args.mesh_file, "object mesh")
    fp_file = _real_file(args.foundationpose_poses, "FoundationPose poses")
    render_config = _real_file(args.render_config, "FoundationPose render config")
    optimized_file = _real_file(args.optimized_hoi, "optimized hoi_data.pkl")
    diagnostics_csv = _real_file(args.diagnostics_csv, "formal diagnostics CSV")
    cache_dir = os.path.abspath(args.cache_dir)
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Missing required real GRAIL cache directory: {cache_dir}")

    hoi_data = load_hoi_data(optimized_file)
    lift4d_meta = hoi_data.get("meta", {}).get("lift4d_depth")
    if not isinstance(lift4d_meta, dict):
        raise ValueError("Saved HOI result has no real Lift4D depth provenance")
    _real_file(lift4d_meta.get("source_path", ""), "Lift4D motion-only NPZ")
    result_meta = hoi_data.get("meta", {})
    certification = result_meta.get("formal_joint_optimization") or result_meta.get(
        "dry_run", {}
    )
    if certification.get("synthetic_data_used") is not False:
        raise ValueError("Saved HOI result does not explicitly certify synthetic_data_used=False")
    if "formal_joint_optimization" in result_meta:
        vggt_meta = result_meta.get("vggt_depth")
        if not isinstance(vggt_meta, dict):
            raise ValueError("Formal HOI result has no real VGGT depth provenance")
        _real_file(vggt_meta.get("depth_path", ""), "VGGT depth")
        if vggt_meta.get("consumed_by_loss") != "depth_pointcloud":
            raise ValueError("Formal HOI result does not prove VGGT depth loss participation")

    with open(config_file, "r") as handle:
        root_cfg = yaml.safe_load(handle)
    cfg = dict(root_cfg["optimization"])
    cfg["human_model"] = dict(root_cfg["human_model"])
    project_root = Path(config_file).resolve().parents[2]
    for key, value in list(cfg["human_model"].items()):
        if (key.endswith("_path") or key.endswith("_dir")) and isinstance(value, str):
            if value and not os.path.isabs(value):
                cfg["human_model"][key] = str(project_root / value)
    cfg.update(
        {
            "results_dir": os.path.abspath(args.results_dir),
            "vis_cfg": {"enable": False},
            "opt_stage_specs": {},
        }
    )

    output_file = Path(args.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_file.parent / "_top_view_render_work"
    optimizer = HOIOptimizer(
        exp_name=args.video_id,
        cfg=cfg,
        cache_dir=cache_dir,
        output_dir=str(work_dir / "optimizer_setup"),
        device=args.device,
    )
    data = optimizer.init_data(video_file, hmr_file, mesh_file, fp_file, render_config)

    visualizer = HOIVisualizer(
        device=args.device,
        human_model=optimizer.human_model,
        cameras=optimizer.cameras,
        image_list=optimizer.image_list,
        video_fps=optimizer.video_fps,
        log_dir=str(work_dir),
        obj_path=mesh_file,
    )
    visualizer.init_vis_meshes(data)
    visualizer.visualize(
        data,
        None,
        hoi_data,
        "optimized",
        {
            "render_video": True,
            "extra_views": ["top"],
            "export_mesh": False,
            "vis_html": False,
            "vis_contact": True,
        },
    )

    rendered = work_dir / "optimized" / "optimized_top_view.mp4"
    if not rendered.is_file() or rendered.stat().st_size == 0:
        raise RuntimeError(f"Top-view renderer did not produce a valid file: {rendered}")
    with open(diagnostics_csv, newline="") as handle:
        diagnostics = list(csv.DictReader(handle))
    capture = cv2.VideoCapture(str(rendered))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open rendered top-view video: {rendered}")
    frame_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if len(diagnostics) != frame_num:
        raise ValueError(
            f"Diagnostics/top-view frame mismatch: {len(diagnostics)} vs {frame_num}"
        )
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    incomplete = output_file.with_suffix(".incomplete.mp4")
    writer = cv2.VideoWriter(
        str(incomplete), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create annotated top-view video: {incomplete}")
    for frame_idx, row in enumerate(diagnostics):
        ok, image = capture.read()
        if not ok:
            writer.release()
            raise RuntimeError(f"Cannot read top-view frame {frame_idx}/{frame_num}")
        lines = [
            f"frame={frame_idx}",
            f"Lift4D raw Z={float(row['lift_z_raw']):.4f}",
            f"Lift4D smooth Z={float(row['lift_z_smooth']):.4f}",
            f"optimized Z={float(row['optimized_z']):.4f}",
            f"hand-object distance={float(row['hand_object_distance']):.4f} m",
        ]
        for line_idx, line in enumerate(lines):
            y = 28 + line_idx * 26
            cv2.putText(image, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(image)
    writer.release()
    capture.release()
    incomplete.replace(output_file)
    print(f"optimized_hoi={optimized_file}")
    print(f"real_mesh={mesh_file}")
    print(f"top_view_mp4={output_file}")


if __name__ == "__main__":
    main()
