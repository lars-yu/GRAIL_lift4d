#!/usr/bin/env python3
"""Render strict real-data diagnostics for GRAIL's Lift4D depth prior."""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grail.adapters.lift4d_depth import load_lift4d_depth_prior, project_opencv_translation


def _require(path: str, label: str) -> Path:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(f"Missing required real {label}: {value}")
    return value


def _install_numpy_pickle_compat() -> None:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _pose_array(value) -> np.ndarray:
    if isinstance(value, dict):
        for key in ("poses", "poses_in_cam", "obj_poses", "object_poses"):
            if key in value:
                value = value[key]
                break
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"FoundationPose file must contain [T,4,4], got {poses.shape}")
    if not np.isfinite(poses).all() or np.any(poses[:, 2, 3] <= 0):
        raise ValueError("FoundationPose OpenCV-camera poses contain invalid/non-positive z")
    return poses


def _load_object_masks(path: Path, frame_num: int, object_id: int) -> list[np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        if "masks" not in data:
            raise KeyError(f"Mask NPZ has no 'masks' key: {path}")
        masks = data["masks"]
        if masks.shape == ():
            masks = masks.item()
    result = []
    for frame in range(frame_num):
        if isinstance(masks, dict):
            frame_masks = masks[frame]
            mask = frame_masks[object_id] if isinstance(frame_masks, dict) else frame_masks[object_id]
        else:
            mask = masks[frame, object_id] if masks.ndim >= 4 else masks[frame]
        mask = np.squeeze(np.asarray(mask, dtype=bool))
        if mask.ndim != 2:
            raise ValueError(f"Object mask frame {frame} must reduce to [H,W], got {mask.shape}")
        result.append(mask)
    return result


def _load_intrinsics(path: Path, frame_num: int) -> np.ndarray:
    K = np.load(path) if path.suffix == ".npy" else np.loadtxt(path)
    K = np.asarray(K, dtype=np.float64)
    if K.shape == (3, 3):
        K = np.repeat(K[None], frame_num, axis=0)
    if K.shape != (frame_num, 3, 3) or not np.isfinite(K).all():
        raise ValueError(f"GRAIL camera intrinsics must be [3,3] or [T,3,3], got {K.shape}")
    return K


def _project_points(points_cam: np.ndarray, K: np.ndarray):
    valid = np.isfinite(points_cam).all(axis=1) & (points_cam[:, 2] > 1e-6)
    points = points_cam[valid]
    uv = np.column_stack(
        [
            K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2],
            K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2],
        ]
    )
    return uv, valid


def _mesh_overlay(image, vertices, edges, R_cam, t_cam, K, color):
    points_cam = vertices @ R_cam.T + t_cam[None]
    uv, valid = _project_points(points_cam, K)
    full_uv = np.full((vertices.shape[0], 2), np.nan, dtype=np.float64)
    full_uv[valid] = uv
    output = image.copy()
    for a, b in edges:
        if not valid[a] or not valid[b]:
            continue
        p0 = tuple(np.rint(full_uv[a]).astype(int))
        p1 = tuple(np.rint(full_uv[b]).astype(int))
        cv2.line(output, p0, p1, color, 1, cv2.LINE_AA)
    return output


def _draw_text(image, lines):
    y = 28
    for line in lines:
        cv2.putText(image, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 27


def _huber_per_frame(error: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(error)
    return np.where(absolute < delta, 0.5 * absolute**2, delta * (absolute - 0.5 * delta))


def main():
    _install_numpy_pickle_compat()
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-file", required=True)
    parser.add_argument("--foundationpose-poses", required=True)
    parser.add_argument("--lift4d-prior", required=True)
    parser.add_argument("--optimized-hoi", required=True)
    parser.add_argument("--mesh-file", required=True)
    parser.add_argument("--mask-npz", required=True)
    parser.add_argument("--grail-camera-intrinsics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--diagnostics-csv", required=True)
    parser.add_argument("--object-id", type=int, default=0)
    parser.add_argument("--smooth-window", type=int, default=31)
    args = parser.parse_args()

    video_path = _require(args.video_file, "RGB video")
    fp_path = _require(args.foundationpose_poses, "FoundationPose poses_in_cam.pkl")
    prior_path = _require(args.lift4d_prior, "Lift4D point trajectory NPZ")
    optimized_path = _require(args.optimized_hoi, "GRAIL optimized hoi_data.pkl")
    mesh_path = _require(args.mesh_file, "object mesh")
    mask_path = _require(args.mask_npz, "object mask NPZ")
    grail_intrinsics_path = _require(args.grail_camera_intrinsics, "GRAIL camera intrinsics")
    diagnostics_csv_path = _require(args.diagnostics_csv, "formal diagnostics CSV")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open real RGB video: {video_path}")
    frame_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_num <= 0:
        raise ValueError(f"Video has no frames: {video_path}")

    fp_poses = _pose_array(_load_pickle(fp_path))
    if fp_poses.shape[0] != frame_num:
        raise ValueError(f"FoundationPose/video frame mismatch: {fp_poses.shape[0]} vs {frame_num}")
    prior = load_lift4d_depth_prior(
        prior_path, frame_num=frame_num, smooth_window=args.smooth_window
    )
    with np.load(prior_path, allow_pickle=False) as raw_prior:
        motion_confidence = np.asarray(
            raw_prior["motion_confidence"], dtype=np.float64
        ).reshape(-1)
        rigid_fit_rmse = np.asarray(
            raw_prior["rigid_fit_rmse"], dtype=np.float64
        ).reshape(-1)
    if motion_confidence.shape != (frame_num,) or rigid_fit_rmse.shape != (frame_num,):
        raise ValueError("Lift4D quality arrays must contain exactly one value per frame")
    if not np.isfinite(motion_confidence).all() or not np.isfinite(rigid_fit_rmse).all():
        raise ValueError("Lift4D quality arrays contain non-finite values")
    grail_intrinsics = _load_intrinsics(grail_intrinsics_path, frame_num)
    optimized = _load_pickle(optimized_path)
    obj_data = optimized.get("obj_data", {})
    required_optimized = ("obj_R_cam", "obj_t_cam", "obj_z_cam", "obj_scale")
    missing = [key for key in required_optimized if key not in obj_data]
    if missing:
        raise KeyError(f"GRAIL optimized output missing ray-depth fields {missing}: {optimized_path}")
    meta_prior = optimized.get("meta", {}).get("lift4d_depth")
    if not isinstance(meta_prior, dict):
        raise ValueError("GRAIL output was not produced with the real Lift4D depth prior")
    if Path(meta_prior.get("source_path", "")).resolve() != prior_path.resolve():
        raise ValueError("GRAIL output Lift4D source path does not match requested real NPZ")
    motion_state = optimized.get("meta", {}).get("object_motion_state")
    if not isinstance(motion_state, dict) or "move_start_frame" not in motion_state:
        raise ValueError("GRAIL output has no validated object motion-state metadata")
    move_start = int(motion_state["move_start_frame"])
    if move_start < 1 or move_start > frame_num:
        raise ValueError(f"Invalid static object interval [0,{move_start})")

    obj_R_cam = np.asarray(obj_data["obj_R_cam"], dtype=np.float64)
    obj_t_cam = np.asarray(obj_data["obj_t_cam"], dtype=np.float64)
    obj_z = np.asarray(obj_data["obj_z_cam"], dtype=np.float64).reshape(-1)
    if obj_R_cam.shape != (frame_num, 3, 3) or obj_t_cam.shape != (frame_num, 3):
        raise ValueError("GRAIL optimized pose arrays do not match video frame count")
    if not np.isfinite(obj_t_cam).all() or np.any(obj_z <= 0):
        raise ValueError("GRAIL optimized OpenCV-camera translation is invalid or z <= 0")

    effective_fp_poses = fp_poses.copy()
    effective_fp_poses[:move_start] = fp_poses[0]
    fp_px = project_opencv_translation(
        effective_fp_poses[:, :3, 3], grail_intrinsics
    )
    optimized_px = project_opencv_translation(obj_t_cam, grail_intrinsics)
    pixel_error = np.linalg.norm(optimized_px - fp_px, axis=1)
    print(
        "projection_pixel_error "
        f"mean={pixel_error.mean():.9g} median={np.median(pixel_error):.9g} max={pixel_error.max():.9g}"
    )
    if pixel_error.max() > 1e-3:
        raise ValueError(
            f"Projection pixel drift exceeds numerical tolerance: max={pixel_error.max():.9g} px"
        )
    static_translation_deviation = np.linalg.norm(
        obj_t_cam[:move_start] - obj_t_cam[0], axis=1
    ).max()
    static_rotation_deviation = np.abs(
        obj_R_cam[:move_start] - obj_R_cam[0]
    ).max()
    print(
        "static_pose_hard_freeze "
        f"frames=0:{move_start} translation={static_translation_deviation:.9g} "
        f"rotation_matrix={static_rotation_deviation:.9g}"
    )
    if static_translation_deviation > 1e-6 or static_rotation_deviation > 1e-6:
        raise ValueError(
            "Optimized static object pose is not hard-frozen: "
            f"translation={static_translation_deviation:.9g} m, "
            f"rotation_matrix={static_rotation_deviation:.9g}"
        )

    masks = _load_object_masks(mask_path, frame_num, args.object_id)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    scale = np.asarray(obj_data["obj_scale"], dtype=np.float64).reshape(-1)
    vertices = vertices * (scale[0] if scale.size == 1 else scale.reshape(1, 3))
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if edges.shape[0] > 5000:
        edges = edges[np.linspace(0, edges.shape[0] - 1, 5000).astype(int)]

    with diagnostics_csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != frame_num:
        raise ValueError(
            f"Diagnostics CSV/video frame mismatch: {len(rows)} vs {frame_num}"
        )
    left_hand_distance = np.asarray(
        [float(row["left_hand_object_distance"]) for row in rows], dtype=np.float64
    )
    right_hand_distance = np.asarray(
        [float(row["right_hand_object_distance"]) for row in rows], dtype=np.float64
    )
    z_target = np.asarray([float(row["lift4d_z_target"]) for row in rows], dtype=np.float64)
    motion_score = np.asarray([float(row["motion_score"]) for row in rows], dtype=np.float64)
    csv_move_start = int(rows[0]["move_start_frame"])
    if csv_move_start != move_start:
        raise ValueError(
            f"Motion-state mismatch between HOI and diagnostics: {move_start} vs "
            f"{csv_move_start}"
        )
    contact_hint = int(rows[0]["contact_hint"])
    window_start = int(rows[0]["contact_window_start"])
    window_end = int(rows[0]["contact_window_end"])
    selected_contact = int(rows[0]["selected_contact_frame"])
    depth_scale = float(meta_prior.get("depth_scale", 1.0))
    depth_error = (obj_z - obj_z[0]) - depth_scale * (z_target - z_target[0])
    depth_loss_frame = prior.frame_weight * _huber_per_frame(depth_error, 0.03)
    velocity_error = np.diff(obj_z) - depth_scale * np.diff(z_target)
    velocity_loss_frame = np.r_[0.0, np.minimum(prior.frame_weight[1:], prior.frame_weight[:-1]) * _huber_per_frame(velocity_error, 0.015)]
    smooth_loss_frame = np.r_[0.0, 0.0, _huber_per_frame(np.diff(obj_z, n=2), 0.015)]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = out_dir / "lift4d_depth_diagnostics.png"
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    frames = np.arange(frame_num)
    axes[0].plot(frames, fp_poses[:, 2, 3], label="FoundationPose z")
    axes[0].plot(frames, prior.center_cam_raw[:, 2], alpha=0.55, label="Lift4D raw center z")
    axes[0].plot(frames, prior.z, linewidth=2, label=f"Lift4D smoothed z ({args.smooth_window})")
    axes[0].plot(frames, z_target, linewidth=2, label="static-relative z target")
    axes[0].plot(frames, obj_z, linewidth=2, label="GRAIL optimized z")
    axes[0].set_ylabel("OpenCV camera z")
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.25)
    point_count = prior.stable_point_score.shape[0]
    axes[1].plot(frames, prior.valid_point_count / point_count, label="valid point ratio")
    axes[1].plot(frames, prior.frame_weight, label="frame weight")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[2].plot(frames, 50.0 * depth_loss_frame, label="50 * depth loss")
    axes[2].plot(frames, 10.0 * velocity_loss_frame, label="10 * velocity loss")
    axes[2].plot(frames, 0.5 * smooth_loss_frame, label="0.5 * smoothness loss")
    axes[2].plot(frames, motion_score, label="motion score")
    axes[2].axvline(contact_hint, color="gray", linestyle=":", label="contact hint")
    axes[2].axvspan(window_start, window_end, color="green", alpha=0.15, label="contact window")
    axes[2].axvline(move_start, color="orange", linestyle="--", label="t_move")
    axes[2].axvline(selected_contact, color="black", linestyle="--", label="selected contact")
    axes[2].set_xlabel("frame")
    axes[2].set_ylabel("weighted contribution")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(diagnostics_path, dpi=160)
    plt.close(fig)

    video_out = out_dir / "foundationpose_vs_lift4d_vs_optimized.mp4"
    video_tmp = out_dir / "foundationpose_vs_lift4d_vs_optimized.incomplete.mp4"
    rigid_out = out_dir / "lift4d_rigid_motion.mp4"
    rigid_tmp = out_dir / "lift4d_rigid_motion.incomplete.mp4"
    writer = cv2.VideoWriter(
        str(video_tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 3, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {video_out}")
    rigid_writer = cv2.VideoWriter(
        str(rigid_tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not rigid_writer.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot create output video: {rigid_out}")
    for frame in range(frame_num):
        ok, rgb = capture.read()
        if not ok:
            writer.release()
            rigid_writer.release()
            raise RuntimeError(f"Failed reading real RGB frame {frame}/{frame_num}")
        mesh_K = grail_intrinsics[frame]
        lift_z_aligned = fp_poses[0, 2, 3] + z_target[frame] - z_target[0]
        lift_pose = effective_fp_poses[frame]
        lift_t_cam = lift_pose[:3, 3] / lift_pose[2, 3] * lift_z_aligned
        left = _mesh_overlay(
            rgb,
            vertices,
            edges,
            lift_pose[:3, :3],
            fp_poses[frame, :3, 3],
            mesh_K,
            (0, 220, 0),
        )
        middle = _mesh_overlay(
            rgb,
            vertices,
            edges,
            fp_poses[frame, :3, :3],
            lift_t_cam,
            mesh_K,
            (255, 180, 0),
        )
        contour, _ = cv2.findContours(masks[frame].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(middle, contour, -1, (0, 255, 0), 2)
        right = _mesh_overlay(
            rgb,
            vertices,
            edges,
            obj_R_cam[frame],
            obj_t_cam[frame],
            mesh_K,
            (0, 140, 255),
        )
        _draw_text(left, ["FoundationPose mesh", f"frame={frame}", f"FP z={fp_poses[frame, 2, 3]:.4f}"])
        _draw_text(
            middle,
            [
                "Lift4D depth-only mesh",
                f"frame={frame}",
                f"Lift4D raw Z={prior.z_raw[frame]:.4f}",
                f"Lift4D smooth Z={prior.z[frame]:.4f}",
                f"Lift4D target Z={z_target[frame]:.4f}",
                f"state={'static' if frame < move_start else 'moving'} t_move={move_start}",
                f"hint={contact_hint} window=[{window_start},{window_end}] selected={selected_contact}",
                f"confidence={motion_confidence[frame]:.4f}",
                f"rigid-fit RMSE={rigid_fit_rmse[frame]:.5f} m",
                f"valid points={prior.valid_point_count[frame]}",
            ],
        )
        _draw_text(
            right,
            [
                "GRAIL optimized mesh",
                f"frame={frame}",
                f"optimized z={obj_z[frame]:.4f}",
                f"left hand distance={left_hand_distance[frame]:.4f} m",
                f"right hand distance={right_hand_distance[frame]:.4f} m",
            ],
        )
        rigid_writer.write(middle)
        writer.write(np.concatenate([left, middle, right], axis=1))
    writer.release()
    rigid_writer.release()
    capture.release()
    video_tmp.replace(video_out)
    rigid_tmp.replace(rigid_out)
    print(f"diagnostics_png={diagnostics_path.resolve()}")
    print(f"rigid_motion_mp4={rigid_out.resolve()}")
    print(f"comparison_mp4={video_out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
