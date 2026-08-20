#!/usr/bin/env python3
"""Render formal real-data front/top/contact/comparison views from one HOI result."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grail.core.io import load_hoi_data
from grail.optimization.hoi_optimizer import HOIOptimizer
from grail.optimization.visualizer import HOIVisualizer
from grail.rendering.camera import project_world_to_screen, transform_camera_to_world


def _real_file(path: str, label: str) -> str:
    resolved = os.path.abspath(path)
    if not os.path.isfile(resolved) or os.path.getsize(resolved) == 0:
        raise FileNotFoundError(f"Missing required real {label}: {resolved}")
    return resolved


def _real_dir(path: str, label: str) -> str:
    resolved = os.path.abspath(path)
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"Missing required real {label}: {resolved}")
    return resolved


def _setup_cfg(root_cfg, config_file, results_dir, lift4d_prior_path):
    cfg = dict(root_cfg["optimization"])
    cfg["human_model"] = dict(root_cfg["human_model"])
    project_root = Path(config_file).resolve().parents[2]
    for key, value in list(cfg["human_model"].items()):
        if (key.endswith("_path") or key.endswith("_dir")) and isinstance(value, str):
            if value and not os.path.isabs(value):
                cfg["human_model"][key] = str(project_root / value)
    cfg.update({
        "results_dir": os.path.abspath(results_dir),
        "use_lift4d_depth_prior": True,
        "lift4d_motion_prior_path": _real_file(
            lift4d_prior_path, "Lift4D motion-only NPZ"
        ),
        "lift4d_stable_point_count": 2500,
        "lift4d_median_window": 7,
        "lift4d_center_smooth_window": 31,
        "lift4d_savgol_polyorder": 2,
        "lift4d_depth_scale": 1.0,
        "learn_lift4d_depth_scale": False,
        "object_motion_state": {"enabled": False},
        "skip_contact_label_loading": True,
        "vis_cfg": {"enable": False},
        "opt_stage_specs": {},
    })
    return cfg


def _copy_obj_data(hoi_data):
    obj = hoi_data.get("obj_data") if isinstance(hoi_data, dict) and "obj_data" in hoi_data else hoi_data
    if not isinstance(obj, dict):
        raise ValueError("Saved HOI result has no real obj_data")
    return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in obj.items()}


def _trajectory_variants(hoi_data, data, optimizer):
    """Return FP, Lift4D-Z-only and optimized object data from real sources."""
    optimized = _copy_obj_data(hoi_data)
    fp_cam = data.obj.poses_cam.detach()
    fp_world_t = transform_camera_to_world(
        fp_cam[:, :3, 3], optimizer.opencv_cam_R, optimizer.opencv_cam_t
    ).detach().cpu().numpy()
    fp_R = data.obj.poses[:, :3, :3].detach().cpu().numpy()
    fp_R_cam = data.obj.poses_cam[:, :3, :3].detach().cpu().numpy()
    fp_obj = _copy_obj_data(optimized)
    fp_obj.update({
        "obj_R": fp_R,
        "obj_R_cam": fp_R_cam,
        "obj_t": fp_world_t,
        "obj_t_cam": fp_cam[:, :3, 3].detach().cpu().numpy(),
        "obj_z_cam": fp_cam[:, 2, 3].detach().cpu().numpy(),
    })
    lift = _copy_obj_data(fp_obj)
    prior = data.lift4d_depth
    if prior is None:
        raise ValueError("Formal renderer requires real Lift4D depth metadata")
    # The formal optimizer consumes only anchor-relative Lift4D camera-Z. The
    # raw prior depth lives in its own reconstruction scale and is not an
    # absolute GRAIL object translation.
    z = fp_cam[0, 2, 3] + (prior.z_target.detach() - prior.z_target[0].detach())
    ray = data.obj.fp_ray_cam.detach()
    lift_cam_t = ray * z[:, None]
    lift_world_t = transform_camera_to_world(
        lift_cam_t, optimizer.opencv_cam_R, optimizer.opencv_cam_t
    ).detach().cpu().numpy()
    lift.update({
        "obj_t": lift_world_t,
        "obj_t_cam": lift_cam_t.detach().cpu().numpy(),
        "obj_z_cam": z.detach().cpu().numpy(),
    })
    return {"FoundationPose": fp_obj, "Lift4D depth-only": lift, "optimized": optimized}


def _render_one(visualizer, data, hoi_data, name, work_dir, scene_center, cam_distance, extra_views):
    local = dict(hoi_data)
    local["object_path"] = visualizer.obj_path
    visualizer.log_dir = str(work_dir)
    visualizer.visualize(
        data, None, local, name,
        {
            "render_video": True,
            "extra_views": extra_views,
            "extra_view_scene_center": scene_center,
            "extra_view_camera_distance": cam_distance,
            "export_mesh": False,
            "vis_html": False,
            "vis_contact": True,
        },
    )


def _video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open rendered video {path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    if info["frames"] <= 0 or info["width"] <= 0 or info["height"] <= 0:
        raise RuntimeError(f"Invalid rendered video metadata: {path}")
    return info


def _copy_checked(src, dst):
    src = Path(src)
    if not src.is_file() or src.stat().st_size == 0:
        raise RuntimeError(f"Renderer did not produce a real output: {src}")
    shutil.copyfile(src, dst)


def _as_mask(mask, shape):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask).squeeze().astype(np.uint8)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def _draw_contours(image, mask, color, thickness=2):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)


def _projected_hull(vertices, cameras, frame, image_shape):
    projected = project_world_to_screen(vertices, cameras)[..., :2]
    points = projected.detach().cpu().numpy()
    valid = np.isfinite(points).all(axis=1)
    valid &= (points[:, 0] >= 0) & (points[:, 0] < image_shape[1])
    valid &= (points[:, 1] >= 0) & (points[:, 1] < image_shape[0])
    points = np.rint(points[valid]).astype(np.int32)
    if points.shape[0] < 3:
        raise ValueError(f"Rendered mesh has fewer than three visible vertices at frame {frame}")
    return cv2.convexHull(points)


def _make_front_overlay(
    base_path,
    output_path,
    rows,
    data,
    cameras,
    human_vertices,
    object_vertices,
):
    cap = cv2.VideoCapture(str(base_path))
    info = _video_info(base_path)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"],
        (info["width"], info["height"]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create front overlay: {output_path}")
    for frame, row in enumerate(rows):
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f"Front overlay frame mismatch at {frame}")
        shape = image.shape[:2]
        _draw_contours(image, _as_mask(data.human.masks[frame], shape), (0, 220, 0), 2)
        _draw_contours(image, _as_mask(data.obj.masks[frame], shape), (0, 255, 255), 2)
        human_hull = _projected_hull(human_vertices[frame], cameras, frame, shape)
        object_hull = _projected_hull(object_vertices[frame], cameras, frame, shape)
        cv2.polylines(image, [human_hull], True, (255, 128, 0), 2, cv2.LINE_AA)
        cv2.polylines(image, [object_hull], True, (255, 0, 255), 2, cv2.LINE_AA)
        observed = tuple(np.rint([
            float(row["observed_palm_u"]), float(row["observed_palm_v"])
        ]).astype(int))
        actual = tuple(np.rint([
            float(row["actual_palm_u"]), float(row["actual_palm_v"])
        ]).astype(int))
        cv2.drawMarker(image, observed, (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
        cv2.circle(image, actual, 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.line(image, observed, actual, (255, 255, 255), 1, cv2.LINE_AA)
        labels = [
            f"frame={frame} palm reprojection={float(row['palm_reprojection_error_px']):.2f}px",
            "real: human=green object=yellow | rendered: human=orange object=magenta",
            "palm: observed=cross actual=red",
        ]
        for line_idx, label in enumerate(labels):
            y = 28 + 23 * line_idx
            cv2.putText(image, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(image)
    cap.release()
    writer.release()


def _top_mapper(hand_vertices, object_vertices, width, height):
    xy = torch.cat((hand_vertices[..., :2].reshape(-1, 2), object_vertices[..., :2].reshape(-1, 2)))
    lo = xy.amin(dim=0).detach().cpu().numpy()
    hi = xy.amax(dim=0).detach().cpu().numpy()
    center = (lo + hi) * 0.5
    span = max(float((hi - lo).max()), 1e-4) * 1.15
    scale = min(width, height) / span

    def project(points):
        values = points.detach().cpu().numpy()[..., :2]
        u = width * 0.5 - (values[..., 0] - center[0]) * scale
        v = height * 0.5 + (values[..., 1] - center[1]) * scale
        return np.rint(np.stack((u, v), axis=-1)).astype(np.int32)

    return project, scale


def _filled_hull(image, points, color, alpha):
    if points.shape[0] < 3:
        return
    hull = cv2.convexHull(points.reshape(-1, 2))
    overlay = image.copy()
    cv2.fillConvexPoly(overlay, hull, color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)
    cv2.polylines(image, [hull], True, color, 2, cv2.LINE_AA)


def _make_contact_closeup(
    output_path,
    rows,
    fps,
    human_model,
    contact_hand,
    human_vertices,
    hand_joints,
    object_vertices,
):
    width, height = 960, 720
    hand_label = "L_Hand" if contact_hand == "left" else "R_Hand"
    if contact_hand not in ("left", "right"):
        raise ValueError("Contact closeup requires one explicitly selected contact hand")
    hand_idx = torch.as_tensor(
        human_model.get_segment_indices([hand_label]),
        dtype=torch.long,
        device=human_vertices.device,
    )
    palm_idx = torch.as_tensor(
        human_model.get_palm_patch_indices(contact_hand),
        dtype=torch.long,
        device=human_vertices.device,
    )
    finger_idx = torch.as_tensor(
        human_model.get_finger_patch_indices(contact_hand),
        dtype=torch.long,
        device=human_vertices.device,
    )
    hand_seq = human_vertices[:, hand_idx]
    palm_seq = human_vertices[:, palm_idx]
    finger_seq = human_vertices[:, finger_idx]
    palm_centers = human_model.get_palm_center_from_hand_joints(hand_joints, contact_hand)
    palm_normals = human_model.get_palm_normal_from_hand_joints(hand_joints, contact_hand)
    project, scale = _top_mapper(hand_seq, object_vertices, width, height)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create contact closeup: {output_path}")
    for frame, row in enumerate(rows):
        image = np.full((height, width, 3), 245, dtype=np.uint8)
        hand_px = project(hand_seq[frame])
        obj_px = project(object_vertices[frame])
        palm_px = project(palm_seq[frame])
        finger_px = project(finger_seq[frame])
        center_px = project(palm_centers[frame:frame + 1])[0]
        normal_end = palm_centers[frame] + 0.08 * palm_normals[frame]
        normal_px = project(normal_end[None])[0]
        _filled_hull(image, obj_px, (210, 150, 70), 0.45)
        _filled_hull(image, hand_px, (80, 170, 235), 0.50)
        for point in palm_px[::max(1, len(palm_px) // 60)]:
            cv2.circle(image, tuple(point), 2, (0, 160, 0), -1, cv2.LINE_AA)
        for point in finger_px[::max(1, len(finger_px) // 40)]:
            cv2.circle(image, tuple(point), 2, (0, 0, 220), -1, cv2.LINE_AA)
        cv2.circle(image, tuple(center_px), 7, (0, 120, 0), -1, cv2.LINE_AA)
        cv2.arrowedLine(image, tuple(center_px), tuple(normal_px), (180, 0, 180), 3, cv2.LINE_AA, tipLength=0.2)
        sampled = palm_seq[frame][::max(1, palm_seq.shape[1] // 8)][:8]
        nearest = torch.cdist(sampled, object_vertices[frame]).argmin(dim=1)
        surface = object_vertices[frame][nearest]
        sampled_px = project(sampled)
        surface_px = project(surface)
        for p0, p1 in zip(sampled_px, surface_px):
            cv2.circle(image, tuple(p1), 4, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.line(image, tuple(p0), tuple(p1), (80, 80, 80), 1, cv2.LINE_AA)
        penetration = float(row["maximum_penetration_m"])
        if penetration > 0.0:
            distances = torch.cdist(palm_seq[frame], object_vertices[frame]).amin(dim=1)
            candidates = project(palm_seq[frame][distances <= max(0.001, penetration)])
            for point in candidates:
                cv2.drawMarker(image, tuple(point), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 8, 2)
        labels = [
            f"frame={frame} contact hand={contact_hand}",
            f"palm surface median={float(row['palm_surface_median_distance_m']):.4f}m coverage<1cm={float(row['palm_patch_fraction_under_1cm']):.2f}",
            f"penetration={penetration:.4f}m reprojection={float(row['palm_reprojection_error_px']):.2f}px",
            "palm patch=green finger patch=red normal=purple surface points=magenta",
        ]
        for line_idx, label in enumerate(labels):
            y = 28 + 24 * line_idx
            cv2.putText(image, label, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(image, label, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        writer.write(image)
    writer.release()


def _combine_top_videos(inputs, output):
    caps = [cv2.VideoCapture(str(path)) for path in inputs]
    if any(not cap.isOpened() for cap in caps):
        raise RuntimeError("Cannot open one of the real top comparison videos")
    infos = [_video_info(path) for path in inputs]
    if len({(x["frames"], x["fps"], x["width"], x["height"]) for x in infos}) != 1:
        raise ValueError(f"Comparison videos do not share frame/fps/size: {infos}")
    info = infos[0]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (info["width"] * 3, info["height"]))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create comparison video: {output}")
    labels = ["FoundationPose", "Lift4D depth-only", "optimized"]
    for frame_idx in range(info["frames"]):
        panels = []
        for cap, label in zip(caps, labels):
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Comparison frame mismatch at {frame_idx}")
            cv2.putText(image, f"{label} | same human trajectory, object motion comparison", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, f"{label} | same human trajectory, object motion comparison", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(image)
        writer.write(np.concatenate(panels, axis=1))
    for cap in caps:
        cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    for name in ("config-file", "video-id", "video-file", "hmr-file", "mesh-file", "foundationpose-poses", "render-config", "cache-dir", "results-dir", "optimized-hoi", "diagnostics-csv", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config_file = _real_file(args.config_file, "GRAIL config")
    video_file = _real_file(args.video_file, "RGB video")
    hmr_file = _real_file(args.hmr_file, "HMR motion")
    mesh_file = _real_file(args.mesh_file, "real object mesh")
    fp_file = _real_file(args.foundationpose_poses, "FoundationPose poses")
    render_config = _real_file(args.render_config, "render config")
    cache_dir = _real_dir(args.cache_dir, "GRAIL cache")
    optimized_file = _real_file(args.optimized_hoi, "optimized HOI data")
    diagnostics_csv = _real_file(args.diagnostics_csv, "palm diagnostics CSV")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hoi_data = load_hoi_data(optimized_file)
    meta = hoi_data.get("meta", {})
    certification = meta.get("formal_joint_optimization", {})
    if certification.get("synthetic_data_used") is not False:
        raise ValueError("Formal renderer requires synthetic_data_used=false")
    if certification.get("formal_result") is not True:
        failed = certification.get("failed_gates", [])
        raise ValueError(
            "Saved result is debug-only because formal acceptance gates did not pass: "
            + ", ".join(map(str, failed))
        )
    if not isinstance(meta.get("lift4d_depth"), dict):
        raise ValueError("Formal renderer requires Lift4D depth provenance")
    lift4d_prior_path = _real_file(
        meta["lift4d_depth"].get("source_path", ""), "Lift4D motion-only NPZ"
    )
    with open(config_file) as handle:
        root_cfg = yaml.safe_load(handle)
    optimizer = HOIOptimizer(
        exp_name=args.video_id,
        cfg=_setup_cfg(root_cfg, config_file, args.results_dir, lift4d_prior_path),
        cache_dir=cache_dir,
        output_dir=str(output_dir / "_setup"),
        device=args.device,
    )
    data = optimizer.init_data(video_file, hmr_file, mesh_file, fp_file, render_config)
    variants = _trajectory_variants(hoi_data, data, optimizer)
    human = hoi_data.get("human_data")
    if not isinstance(human, dict):
        raise ValueError("Saved HOI result has no real human_data")
    motion_for_bounds = {
        key: torch.as_tensor(value, device=args.device)
        if isinstance(value, np.ndarray) else value
        for key, value in human.items()
    }
    human_vertices, _, _ = optimizer.human_model.generate_mesh(
        motion_for_bounds, output_joints=True, require_grad=False
    )
    hand_joints = optimizer.human_model.get_hand_joints(
        motion_for_bounds, require_grad=False
    )
    optimized_R = torch.as_tensor(
        variants["optimized"]["obj_R"], device=args.device, dtype=torch.float32
    )
    optimized_t = torch.as_tensor(
        variants["optimized"]["obj_t"], device=args.device, dtype=torch.float32
    )
    base_object = data.obj.verts.to(device=args.device, dtype=torch.float32)
    optimized_object_vertices = torch.bmm(
        base_object[None].expand(data.frame_num, -1, -1), optimized_R.transpose(1, 2)
    ) + optimized_t[:, None, :]
    bounds = [human_vertices.reshape(-1, 3)]
    for vertices in (variants["optimized"].get("obj_t"),):
        if vertices is not None:
            bounds.append(torch.as_tensor(vertices, device=args.device, dtype=torch.float32).reshape(-1, 3))
    if not bounds:
        raise ValueError("Saved HOI result has no mesh trajectory")
    all_points = torch.cat(bounds)
    scene_center = ((all_points.min(0).values + all_points.max(0).values) / 2).detach().cpu().tolist()
    cam_distance = float((all_points.max(0).values - all_points.min(0).values).max() * 2.5)
    visualizer = HOIVisualizer(args.device, optimizer.human_model, optimizer.cameras, optimizer.image_list, optimizer.video_fps, str(output_dir / "_render_work"), mesh_file)
    visualizer.init_vis_meshes(data)
    for label, obj_data in variants.items():
        variant = dict(hoi_data)
        variant["obj_data"] = obj_data
        _render_one(visualizer, data, variant, label.replace(" ", "_"), output_dir / "_render_work", scene_center, cam_distance, ["top"])
    front_src = output_dir / "_render_work" / "optimized" / "optimized.mp4"
    top_src = output_dir / "_render_work" / "optimized" / "optimized_top_view.mp4"
    rows = list(csv.DictReader(open(diagnostics_csv, newline="")))
    if len(rows) != data.frame_num:
        raise ValueError(
            f"Palm diagnostics frame mismatch: {len(rows)} != {data.frame_num}"
        )
    _make_front_overlay(
        front_src,
        output_dir / "optimized_front_overlay.mp4",
        rows,
        data,
        optimizer.cameras,
        human_vertices,
        optimized_object_vertices,
    )
    _copy_checked(top_src, output_dir / "optimized_top_view.mp4")
    _make_contact_closeup(
        output_dir / "optimized_contact_closeup.mp4",
        rows,
        optimizer.video_fps,
        optimizer.human_model,
        str(
            certification.get(
                "contact_hand", (meta.get("diagnostics", {}) or {}).get("contact_hand", "")
            )
        ).lower(),
        human_vertices,
        hand_joints,
        optimized_object_vertices,
    )
    comparison_sources = [
        output_dir / "_render_work" / "FoundationPose" / "FoundationPose_top_view.mp4",
        output_dir / "_render_work" / "Lift4D_depth-only" / "Lift4D_depth-only_top_view.mp4",
        top_src,
    ]
    _combine_top_videos(comparison_sources, output_dir / "foundationpose_vs_lift4d_vs_optimized_top.mp4")
    for path in (output_dir / "optimized_front_overlay.mp4", output_dir / "optimized_top_view.mp4", output_dir / "optimized_contact_closeup.mp4", output_dir / "foundationpose_vs_lift4d_vs_optimized_top.mp4"):
        info = _video_info(path)
        if info["frames"] != data.frame_num:
            raise ValueError(f"Formal output frame mismatch for {path}: {info['frames']} != {data.frame_num}")
    print(f"formal_output_dir={output_dir}")
    print(f"frames={data.frame_num}")
    print("synthetic_data_used=false")


if __name__ == "__main__":
    main()
