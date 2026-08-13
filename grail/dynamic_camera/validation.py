"""Validation reports for dynamic-camera VGGT reconstruction."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np

from grail.dynamic_camera.geometry import (
    camera_motion_metrics,
    erode_mask,
    read_ply_vertices,
    resize_mask,
    rotation_angle_deg,
    unproject_opencv_depth_to_world,
    write_ply,
)
from grail.preprocessing.preprocess import load_masks_from_cache


def _axis_angle_to_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis_angle / theta
    x, y, z = axis
    skew = np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def _plot_series(path: Path, values: np.ndarray, title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(8, 3))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("frame")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_camera_top_side(output_dir: Path, centers: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(centers) == 0:
        return

    plt.figure(figsize=(5, 5))
    plt.plot(centers[:, 0], centers[:, 1], marker="o", markersize=2)
    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Camera Trajectory Top")
    plt.tight_layout()
    plt.savefig(output_dir / "camera_trajectory_top.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 3))
    plt.plot(centers[:, 0], centers[:, 2], marker="o", markersize=2)
    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title("Camera Trajectory Side")
    plt.tight_layout()
    plt.savefig(output_dir / "camera_trajectory_side.png", dpi=160)
    plt.close()


def _plot_object_trajectory(output_dir: Path, obj_t: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if obj_t.ndim != 2 or obj_t.shape[1] != 3 or len(obj_t) == 0:
        return

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(obj_t[:, 0], obj_t[:, 1], marker="o", markersize=2)
    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Object Top")
    plt.subplot(1, 2, 2)
    plt.plot(obj_t[:, 0], obj_t[:, 2], marker="o", markersize=2)
    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title("Object Side")
    plt.tight_layout()
    plt.savefig(output_dir / "object_world_trajectory.png", dpi=160)
    plt.close()


def _plot_human_root_trajectory(output_dir: Path, root_t: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if root_t.ndim != 2 or root_t.shape[1] != 3 or len(root_t) == 0:
        return

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(root_t[:, 0], root_t[:, 1], marker="o", markersize=2)
    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Human Root Top")
    plt.subplot(1, 2, 2)
    plt.plot(root_t[:, 0], root_t[:, 2], marker="o", markersize=2)
    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title("Human Root Side")
    plt.tight_layout()
    plt.savefig(output_dir / "human_root_trajectory.png", dpi=160)
    plt.close()


def _plot_multi_series(path: Path, series: dict[str, np.ndarray], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(8, 3))
    for label, values in series.items():
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size:
            plt.plot(values, label=label)
    plt.title(title)
    plt.xlabel("frame")
    plt.ylabel(ylabel)
    if series:
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except Exception:
        pass
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr


def _load_hoi_data(hoi_data_file: str | Path | None) -> dict:
    if hoi_data_file is None or not Path(hoi_data_file).exists():
        return {}
    with open(hoi_data_file, "rb") as handle:
        data = pickle.load(handle)
    return data if isinstance(data, dict) else {}


def _metric_window(hoi_data: dict, n_frames: int, precontact_end: int | None = None) -> tuple[int, int]:
    meta = hoi_data.get("meta", {}) if isinstance(hoi_data, dict) else {}
    start = meta.get("inter_start_idx", precontact_end if precontact_end is not None else 0)
    end = meta.get("inter_end_idx", n_frames)
    try:
        start = int(start)
    except Exception:
        start = 0
    try:
        end = int(end)
    except Exception:
        end = n_frames
    start = max(0, min(start, n_frames))
    end = max(start + 1, min(end, n_frames)) if n_frames > 0 else 0
    return start, end


def _min_distances_to_points(query: np.ndarray, target: np.ndarray, chunk: int = 256) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float32).reshape(-1, 3)
    if len(query) == 0 or len(target) == 0:
        return np.zeros((0,), dtype=np.float32)
    mins = []
    for start in range(0, len(query), chunk):
        q = query[start : start + chunk]
        d = np.linalg.norm(q[:, None, :] - target[None, :, :], axis=-1)
        mins.append(d.min(axis=1))
    return np.concatenate(mins, axis=0)


def _load_sim3_metrics(aligned_dir: Path) -> dict:
    path = aligned_dir / "alignment" / "sim3_vggt_to_blender.json"
    if not path.exists():
        return {}
    with open(path, "r") as handle:
        data = json.load(handle)
    meta = data.get("metadata", {})
    return {
        "sim3_scale": float(data.get("scale", 1.0)),
        "alignment_rmse_m": float(meta.get("rmse_m", 0.0)),
        "alignment_median_m": float(meta.get("median_m", 0.0)),
    }


def _load_alignment_provenance_metrics(aligned_dir: Path, *, c2w_frame_count: int | None = None) -> dict:
    path = aligned_dir / "metadata.json"
    if not path.exists():
        return {"alignment_metadata_present": False}
    try:
        with open(path, "r") as handle:
            metadata = json.load(handle)
    except Exception as exc:
        return {"alignment_metadata_present": False, "alignment_metadata_error": str(exc)}
    if not isinstance(metadata, dict):
        return {"alignment_metadata_present": False, "alignment_metadata_error": "metadata is not a JSON object"}

    inputs = metadata.get("alignment_inputs") if isinstance(metadata.get("alignment_inputs"), dict) else {}
    meta_frames = metadata.get("num_frames", metadata.get("frame_count"))
    metrics: dict[str, object] = {
        "alignment_metadata_present": True,
        "alignment_metadata_backend": metadata.get("backend"),
        "alignment_metadata_camera_mode": metadata.get("camera_mode"),
        "alignment_single_video_sim3": bool(metadata.get("single_video_sim3")),
        "alignment_metadata_frame_count": int(meta_frames) if meta_frames is not None else None,
        "alignment_inputs_present": bool(inputs),
        "alignment_vggt_dir": metadata.get("vggt_dir") or inputs.get("vggt_dir"),
        "alignment_coordinate_convention": metadata.get("coordinate_convention"),
    }
    if meta_frames is not None and c2w_frame_count is not None:
        metrics["alignment_metadata_frame_count_matches_c2w"] = int(meta_frames) == int(c2w_frame_count)
    for key in (
        "vggt_dir",
        "blender_depth_path",
        "blender_K_path",
        "render_config_file",
        "static_mask_path",
        "human_mask_path",
        "object_mask_path",
        "static_scene_ply",
        "masks_cache_file",
        "alignment_mode",
        "confidence_percentile",
        "erode_pixels",
        "ransac_threshold_m",
    ):
        if key in inputs:
            metrics[f"alignment_inputs_{key}"] = inputs[key]
    return metrics


def _object_static_metrics(pose_file: str | Path | None, precontact_end: int | None = None) -> dict:
    if pose_file is None or not Path(pose_file).exists():
        return {}
    with open(pose_file, "rb") as handle:
        poses = pickle.load(handle)
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4) or len(poses) == 0:
        return {}
    n = int(precontact_end) if precontact_end is not None else min(10, len(poses))
    n = max(1, min(n, len(poses)))
    trans = poses[:n, :3, 3]
    mean_t = trans.mean(axis=0)
    pos_std = float(np.sqrt(np.mean(np.sum((trans - mean_t) ** 2, axis=1))))
    ref_R = poses[0, :3, :3]
    rot_std = float(np.std([rotation_angle_deg(ref_R, poses[i, :3, :3]) for i in range(n)]))
    return {
        "object_static_pos_std_m": pos_std,
        "object_static_rot_std_deg": rot_std,
    }


def _object_trajectory_metrics_from_pose_file(
    pose_file: str | Path | None,
    output_dir: Path,
    precontact_end: int | None = None,
) -> dict:
    """Plot and score raw object trajectory when it is already in Blender world."""
    if pose_file is None:
        return {}
    pose_file = Path(pose_file)
    if "poses_in_world" not in pose_file.name or not pose_file.exists():
        return {}
    with open(pose_file, "rb") as handle:
        poses = pickle.load(handle)
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4) or len(poses) == 0:
        return {}

    obj_t = poses[:, :3, 3]
    obj_R = poses[:, :3, :3]
    _plot_object_trajectory(output_dir, obj_t)

    n = int(precontact_end) if precontact_end is not None else min(10, len(poses))
    n = max(1, min(n, len(poses)))
    ref_t = obj_t[:n].mean(axis=0)
    pos_error = np.linalg.norm(obj_t[:n] - ref_t.reshape(1, 3), axis=1)
    _plot_series(output_dir / "object_static_error.png", pos_error, "Object Static Error", "m")
    ref_R = obj_R[0]
    rot_error = np.asarray([rotation_angle_deg(ref_R, obj_R[i]) for i in range(n)])
    return {
        "object_static_error_mean_m": float(pos_error.mean()) if pos_error.size else 0.0,
        "object_static_error_p95_m": float(np.percentile(pos_error, 95)) if pos_error.size else 0.0,
        "object_static_rot_error_mean_deg": float(rot_error.mean()) if rot_error.size else 0.0,
        "object_static_rot_error_p95_deg": float(np.percentile(rot_error, 95)) if rot_error.size else 0.0,
        "raw_foundationpose_world_trajectory_source": str(pose_file),
        "raw_foundationpose_world_trajectory_stage": "raw_foundationpose_world",
    }


def _object_static_metrics_from_hoi(hoi_data: dict, precontact_end: int | None = None) -> dict:
    obj = hoi_data.get("obj_data", {}) if isinstance(hoi_data, dict) else {}
    obj_t = obj.get("obj_t")
    obj_R = obj.get("obj_R")
    if obj_t is None or obj_R is None:
        return {}
    obj_t = np.asarray(obj_t, dtype=np.float64)
    obj_R = np.asarray(obj_R, dtype=np.float64)
    if obj_t.ndim != 2 or obj_t.shape[1] != 3 or len(obj_t) == 0:
        return {}
    inter_start = None
    if precontact_end is not None:
        inter_start = precontact_end
    elif "meta" in hoi_data:
        inter_start = hoi_data["meta"].get("inter_start_idx")
    n = int(inter_start) if inter_start is not None else min(10, len(obj_t))
    n = max(1, min(n, len(obj_t)))
    mean_t = obj_t[:n].mean(axis=0)
    pos_std = float(np.sqrt(np.mean(np.sum((obj_t[:n] - mean_t) ** 2, axis=1))))
    if obj_R.ndim == 3 and obj_R.shape[-2:] == (3, 3):
        ref_R = obj_R[0]
        rot_std = float(np.std([rotation_angle_deg(ref_R, obj_R[i]) for i in range(n)]))
    else:
        rot_std = 0.0
    return {
        "object_static_pos_std_m": pos_std,
        "object_static_rot_std_deg": rot_std,
    }


def _object_trajectory_metrics_from_hoi(
    hoi_data: dict,
    output_dir: Path,
    precontact_end: int | None = None,
) -> dict:
    obj = hoi_data.get("obj_data", {}) if isinstance(hoi_data, dict) else {}
    obj_t = _to_numpy(obj.get("obj_t"))
    obj_R = _to_numpy(obj.get("obj_R"))
    if obj_t is None or obj_t.ndim != 2 or obj_t.shape[1] != 3 or len(obj_t) == 0:
        return {}
    _plot_object_trajectory(output_dir, obj_t.astype(np.float64))

    n = len(obj_t)
    end = precontact_end
    if end is None:
        end = hoi_data.get("meta", {}).get("inter_start_idx") if isinstance(hoi_data, dict) else None
    try:
        end = int(end)
    except Exception:
        end = min(10, n)
    end = max(1, min(end, n))

    ref_t = obj_t[:end].mean(axis=0)
    pos_error = np.linalg.norm(obj_t[:end] - ref_t.reshape(1, 3), axis=1)
    _plot_series(output_dir / "object_static_error.png", pos_error, "Object Static Error", "m")

    metrics = {
        "object_static_error_mean_m": float(pos_error.mean()) if pos_error.size else 0.0,
        "object_static_error_p95_m": float(np.percentile(pos_error, 95)) if pos_error.size else 0.0,
    }
    if obj_R is not None and obj_R.ndim == 3 and obj_R.shape[-2:] == (3, 3):
        ref_R = obj_R[0]
        rot_error = np.asarray([rotation_angle_deg(ref_R, obj_R[i]) for i in range(end)])
        metrics["object_static_rot_error_mean_deg"] = float(rot_error.mean()) if rot_error.size else 0.0
        metrics["object_static_rot_error_p95_deg"] = (
            float(np.percentile(rot_error, 95)) if rot_error.size else 0.0
        )
    return metrics


def _human_motion_metrics_from_cache(
    human_motion_file: str | Path | None,
    output_dir: Path,
    *,
    fps: float | None = None,
) -> dict:
    """Plot and score cached human root motion already in Blender world."""
    if human_motion_file is None:
        return {}
    human_motion_file = Path(human_motion_file)
    if not human_motion_file.exists():
        return {}
    try:
        cache = np.load(human_motion_file, allow_pickle=True)
        motion = cache["motion_world"].item()
        metadata = cache["metadata"].item() if "metadata" in cache else {}
    except Exception:
        return {}
    root_t = _to_numpy(motion.get("trans"))
    poses = _to_numpy(motion.get("poses"))
    if root_t is None or root_t.ndim != 2 or root_t.shape[1] != 3 or len(root_t) == 0:
        return {}
    root_t = root_t.astype(np.float64)
    _plot_human_root_trajectory(output_dir, root_t)

    if len(root_t) > 1:
        root_step = np.linalg.norm(root_t[1:] - root_t[:-1], axis=1)
        root_speed = root_step * float(fps) if fps else root_step
    else:
        root_step = np.zeros((0,), dtype=np.float64)
        root_speed = np.zeros((0,), dtype=np.float64)
    root_acc = np.linalg.norm(root_t[2:] - 2.0 * root_t[1:-1] + root_t[:-2], axis=1) if len(root_t) > 2 else np.zeros((0,), dtype=np.float64)
    _plot_series(output_dir / "human_root_speed.png", root_speed, "Human Root Translation Speed", "m/s" if fps else "m/frame")

    metrics = {
        "human_motion_file": str(human_motion_file),
        "human_motion_coordinate_space": metadata.get("output_coordinate_space", "blender_metric_world"),
        "human_root_translation_step_mean_m": float(root_step.mean()) if root_step.size else 0.0,
        "human_root_translation_step_p95_m": float(np.percentile(root_step, 95)) if root_step.size else 0.0,
        "human_root_translation_speed_mean_mps": float(root_speed.mean()) if root_speed.size else 0.0,
        "human_root_translation_speed_p95_mps": float(np.percentile(root_speed, 95)) if root_speed.size else 0.0,
        "human_root_acceleration_mean_m": float(root_acc.mean()) if root_acc.size else 0.0,
        "human_root_acceleration_p95_m": float(np.percentile(root_acc, 95)) if root_acc.size else 0.0,
        "human_motion_cache_kind": metadata.get("cache_kind", "dynamic_human_world_motion"),
    }
    if poses is not None and poses.ndim == 2 and poses.shape[1] >= 3 and len(poses) == len(root_t):
        rotations = np.asarray([_axis_angle_to_matrix_np(pose[:3]) for pose in poses], dtype=np.float64)
        if len(rotations) > 1:
            rot_steps = np.asarray(
                [rotation_angle_deg(rotations[i], rotations[i + 1]) for i in range(len(rotations) - 1)],
                dtype=np.float64,
            )
            rot_speed = rot_steps * float(fps) if fps else rot_steps
            _plot_series(
                output_dir / "human_root_rotation_speed.png",
                rot_speed,
                "Human Root Rotation Speed",
                "deg/s" if fps else "deg/frame",
            )
            metrics["human_root_rotation_step_mean_deg"] = float(rot_steps.mean()) if rot_steps.size else 0.0
            metrics["human_root_rotation_step_p95_deg"] = float(np.percentile(rot_steps, 95)) if rot_steps.size else 0.0
            metrics["human_root_rotation_speed_mean_degps"] = float(rot_speed.mean()) if rot_speed.size else 0.0
            metrics["human_root_rotation_speed_p95_degps"] = float(np.percentile(rot_speed, 95)) if rot_speed.size else 0.0
    return metrics


def _load_hoi_eval_metrics(hoi_data: dict, precontact_end: int | None = None) -> dict:
    eval_data = hoi_data.get("eval_data", {}) if isinstance(hoi_data, dict) else {}
    metrics = {}
    if "scene_static" in eval_data:
        metrics["static_scene_error_mean_m"] = float(eval_data["scene_static"])
    if "human_depth_pointcloud" in eval_data:
        metrics["human_depth_chamfer_m"] = float(eval_data["human_depth_pointcloud"])
    elif "depth_pointcloud" in eval_data:
        metrics["human_depth_chamfer_m"] = float(eval_data["depth_pointcloud"])
    if "object_depth_pointcloud" in eval_data:
        metrics["object_depth_chamfer_m"] = float(eval_data["object_depth_pointcloud"])
    elif "depth_pointcloud" in eval_data:
        metrics["object_depth_chamfer_m"] = float(eval_data["depth_pointcloud"])
    if "contact" in eval_data:
        metrics["contact_distance_mean_m"] = float(eval_data["contact"])
    if "verts_tracking" in eval_data:
        metrics["object_projection_l1_px"] = float(eval_data["verts_tracking"])
    metrics.update(_object_static_metrics_from_hoi(hoi_data, precontact_end=precontact_end))
    return metrics


def _write_scalar_diagnostic_plots(output_dir: Path, metrics: dict) -> None:
    scalar_plots = [
        ("static_scene_error_mean_m", "static_scene_error.png", "Static Scene Error", "m"),
        ("static_scene_error_mean_m", "table_stability.png", "Table / Static Scene Stability", "m"),
        ("human_depth_chamfer_m", "human_depth_error.png", "Human Depth Chamfer", "m"),
        ("object_depth_chamfer_m", "object_depth_error.png", "Object Depth Chamfer", "m"),
    ]
    for key, filename, title, ylabel in scalar_plots:
        value = metrics.get(key)
        if isinstance(value, (int, float, np.floating)) and not (output_dir / filename).exists():
            _plot_series(output_dir / filename, np.asarray([float(value)]), title, ylabel)
        elif value is None and not (output_dir / filename).exists():
            _plot_series(
                output_dir / filename,
                np.asarray([], dtype=np.float32),
                f"{title} (unavailable)",
                ylabel,
            )


def _foot_slip_metrics_from_hoi(
    hoi_data: dict,
    output_dir: Path,
    *,
    threshold: float = 0.5,
) -> dict:
    val = hoi_data.get("validation_data", {}) if isinstance(hoi_data, dict) else {}
    joints = _to_numpy(val.get("body_joints_seq"))
    if joints is None:
        joints = _to_numpy(hoi_data.get("human_data", {}).get("joints"))
    if joints is None or joints.ndim != 3 or joints.shape[1] <= 16 or len(joints) < 2:
        return {}

    left = joints[:, 15, :].astype(np.float64)
    right = joints[:, 16, :].astype(np.float64)
    left_delta = np.linalg.norm(left[1:] - left[:-1], axis=1)
    right_delta = np.linalg.norm(right[1:] - right[:-1], axis=1)

    probs = _to_numpy(val.get("foot_contact_probs"))
    method = "all_frames_no_contact_probs"
    if probs is not None and probs.ndim == 2 and probs.shape[0] >= len(joints) and probs.shape[1] >= 4:
        left_contact = np.maximum(probs[:, 0], probs[:, 1])
        right_contact = np.maximum(probs[:, 2], probs[:, 3])
        left_mask = np.minimum(left_contact[:-1], left_contact[1:]) > threshold
        right_mask = np.minimum(right_contact[:-1], right_contact[1:]) > threshold
        method = "foot_contact_probs"
    else:
        left_mask = np.ones_like(left_delta, dtype=bool)
        right_mask = np.ones_like(right_delta, dtype=bool)

    left_slip = left_delta[left_mask]
    right_slip = right_delta[right_mask]
    _plot_multi_series(
        output_dir / "foot_slip.png",
        {
            "left": left_delta,
            "right": right_delta,
        },
        "Foot Slip in Blender World",
        "frame delta (m)",
    )
    return {
        "left_foot_slip_max_m": float(left_slip.max()) if left_slip.size else 0.0,
        "right_foot_slip_max_m": float(right_slip.max()) if right_slip.size else 0.0,
        "left_foot_slip_mean_m": float(left_slip.mean()) if left_slip.size else 0.0,
        "right_foot_slip_mean_m": float(right_slip.mean()) if right_slip.size else 0.0,
        "foot_slip_method": method,
    }


def _contact_metrics_from_hoi(
    hoi_data: dict,
    output_dir: Path,
    *,
    artifact_prefix: str = "",
) -> dict:
    val = hoi_data.get("validation_data", {}) if isinstance(hoi_data, dict) else {}
    hand = _to_numpy(val.get("hand_joints_seq"))
    obj_verts = _to_numpy(val.get("obj_verts_sample_seq"))
    obj_t = _to_numpy(hoi_data.get("obj_data", {}).get("obj_t"))
    if hand is None or obj_verts is None or hand.ndim != 3 or obj_verts.ndim != 3:
        return {}
    n = min(len(hand), len(obj_verts))
    if n == 0:
        return {}
    hand = hand[:n].astype(np.float32)
    obj_verts = obj_verts[:n].astype(np.float32)

    contact_distance = np.zeros(n, dtype=np.float32)
    for i in range(n):
        d = _min_distances_to_points(hand[i], obj_verts[i], chunk=128)
        contact_distance[i] = float(d.min()) if d.size else 0.0
    _plot_series(
        output_dir / f"{artifact_prefix}contact_distance.png",
        contact_distance,
        "Hand-Object Nearest Distance",
        "m",
    )

    start, end = _metric_window(hoi_data, n)
    contact_window = contact_distance[start:end]
    metrics = {
        "contact_distance_mean_m": float(contact_window.mean()) if contact_window.size else 0.0,
        "contact_distance_p95_m": float(np.percentile(contact_window, 95))
        if contact_window.size
        else 0.0,
    }

    obj_t_valid = None
    if obj_t is not None and obj_t.ndim == 2 and obj_t.shape[1] == 3:
        obj_t = obj_t[:n].astype(np.float64)
        obj_t_valid = obj_t
        hand_center = hand.mean(axis=1).astype(np.float64)
        rel = obj_t - hand_center
        rel_window = rel[start:end]
        if len(rel_window) > 0:
            rel_err = rel_window - rel_window.mean(axis=0, keepdims=True)
            metrics["grasp_relative_translation_std_m"] = float(
                np.sqrt(np.mean(np.sum(rel_err**2, axis=1)))
            )
            metrics["grasp_relative_translation_variance_m2"] = float(
                np.mean(np.sum(rel_err**2, axis=1))
            )
            metrics["grasp_relative_transform_variance"] = float(np.mean(np.sum(rel_err**2, axis=1)))

    obj_R = _to_numpy(hoi_data.get("obj_data", {}).get("obj_R"))
    if obj_R is not None and obj_R.ndim == 3 and obj_R.shape[1:] == (3, 3) and hand.shape[1] >= 32:
        obj_R = obj_R[:n].astype(np.float64)

        def _normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
            return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)

        def _hand_frame_np(side: str) -> tuple[np.ndarray | None, np.ndarray | None]:
            offset = 0 if side == "left" else 16
            wrist = hand[:, offset, :].astype(np.float64)
            fingers = hand[:, offset + 1 : offset + 16, :].astype(np.float64)
            if fingers.shape[1] < 2:
                return None, None
            x_axis = _normalize_np(fingers.mean(axis=1) - wrist)
            spread = _normalize_np(fingers[:, 0, :] - fingers[:, -1, :])
            z_axis = _normalize_np(np.cross(x_axis, spread))
            y_axis = _normalize_np(np.cross(z_axis, x_axis))
            z_axis = _normalize_np(np.cross(x_axis, y_axis))
            return np.stack([x_axis, y_axis, z_axis], axis=-1), wrist

        side_scores = {}
        frames = slice(start, end)
        for side in ("left", "right"):
            _, wrist = _hand_frame_np(side)
            if wrist is None or obj_t_valid is None:
                continue
            side_scores[side] = float(np.mean(np.linalg.norm(obj_t_valid[frames] - wrist[frames], axis=1)))
        side = min(side_scores, key=side_scores.get) if side_scores else "right"
        hand_R, _ = _hand_frame_np(side)
        if hand_R is not None and end - start > 1:
            rel_R = np.matmul(np.swapaxes(hand_R, 1, 2), obj_R)
            rel_window_R = rel_R[start:end]
            ref_R = rel_window_R[0]
            rot_errors = np.asarray(
                [rotation_angle_deg(ref_R, rel_window_R[i]) for i in range(len(rel_window_R))],
                dtype=np.float64,
            )
            metrics["grasp_relative_rotation_side"] = side
            metrics["grasp_relative_rotation_mean_deg"] = float(np.mean(rot_errors))
            metrics["grasp_relative_rotation_std_deg"] = float(np.std(rot_errors))
            metrics["grasp_relative_rotation_variance_rad2"] = float(
                np.var(np.deg2rad(rot_errors))
            )

    sdf_ratio = _to_numpy(val.get("penetration_ratio_sdf"))
    if sdf_ratio is not None:
        sdf_ratio = sdf_ratio.reshape(-1)[:n]
        metrics["penetration_ratio"] = float(np.mean(sdf_ratio[start:end]))
        metrics["penetration_ratio_method"] = "object_sdf"
    else:
        human_verts = _to_numpy(val.get("human_verts_sample_seq"))
        if human_verts is not None and human_verts.ndim == 3:
            human_verts = human_verts[:n].astype(np.float32)
            proxy = np.zeros(n, dtype=np.float32)
            for i in range(n):
                h = human_verts[i]
                o = obj_verts[i]
                if len(h) > 512:
                    h = h[np.linspace(0, len(h) - 1, 512).astype(np.int64)]
                if len(o) > 512:
                    o = o[np.linspace(0, len(o) - 1, 512).astype(np.int64)]
                d = _min_distances_to_points(h, o, chunk=128)
                proxy[i] = float(np.mean(d < 0.005)) if d.size else 0.0
            metrics["penetration_ratio"] = float(np.mean(proxy[start:end]))
            metrics["penetration_ratio_method"] = "unsigned_vertex_proximity_proxy_5mm"

    return metrics


def _object_mask_iou_from_hoi(
    aligned_dir: Path,
    hoi_data_file: str | Path | None,
    masks_cache_file: str | Path | None,
    obj_path: str | Path | None,
    *,
    device: str = "cuda",
    max_frames: int | None = None,
) -> dict:
    if (
        hoi_data_file is None
        or masks_cache_file is None
        or obj_path is None
        or not Path(hoi_data_file).exists()
        or not Path(masks_cache_file).exists()
        or not Path(obj_path).exists()
    ):
        return {}
    try:
        import torch
        import torch.nn.functional as F

        from grail.core.io import load_mesh
        from grail.dynamic_camera.camera import camera_for_frame, get_batched_cameras_from_opencv_c2w
        from grail.preprocessing.preprocess import load_masks_from_cache
        from grail.rendering.renderer import RendererType, create_renderer, render_frame
        from grail.rendering.textures import create_colored_meshes
    except Exception:
        return {}

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    with open(hoi_data_file, "rb") as handle:
        hoi_data = pickle.load(handle)
    obj = hoi_data.get("obj_data", {})
    if "obj_R" not in obj or "obj_t" not in obj or "obj_scale" not in obj:
        return {}

    masks = load_masks_from_cache(str(masks_cache_file))
    if not masks:
        return {}
    first_mask = np.asarray(masks[0][0]).squeeze()
    image_size_hw = tuple(first_mask.shape[:2])

    c2w = np.load(aligned_dir / "c2w_blender.npy").astype(np.float32)
    intr_path = aligned_dir / "intrinsics_original.npy"
    if not intr_path.exists():
        intr_path = aligned_dir / "intrinsics.npy"
    intrinsics = np.load(intr_path).astype(np.float32)
    cameras = get_batched_cameras_from_opencv_c2w(
        torch.from_numpy(c2w).float(),
        torch.from_numpy(intrinsics).float(),
        image_size_hw,
        device=device,
    )

    obj_scale = torch.as_tensor(obj["obj_scale"], dtype=torch.float32, device=device)
    verts, faces, _ = load_mesh(str(obj_path), mesh_scale=obj_scale, target_num_verts=4000, device=device)
    obj_R = torch.as_tensor(obj["obj_R"], dtype=torch.float32, device=device)
    obj_t = torch.as_tensor(obj["obj_t"], dtype=torch.float32, device=device)
    frame_num = min(len(obj_R), len(masks), len(c2w))
    if max_frames is not None:
        frame_num = min(frame_num, int(max_frames))
    if frame_num <= 0:
        return {}

    color = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    ious = []
    for i in range(frame_num):
        verts_i = torch.matmul(verts, obj_R[i].T) + obj_t[i]
        mesh = create_colored_meshes(verts_i, faces, color)
        camera_i = camera_for_frame(cameras, i)
        renderer = create_renderer(
            camera_i,
            image_size_hw,
            renderer_type=RendererType.HARD_PHONG,
            neutral_light=True,
            background_color=[0, 0, 0],
            device=device,
        )
        _, alpha = render_frame(mesh, camera_i, renderer, require_grad=False)
        pred = (alpha > 0.1).float()
        gt = torch.from_numpy(np.asarray(masks[i][0]).squeeze() > 0).float().to(device)
        if tuple(gt.shape) != tuple(pred.shape):
            gt = F.interpolate(
                gt.reshape(1, 1, *gt.shape),
                size=pred.shape,
                mode="nearest",
            ).reshape_as(pred)
        inter = torch.logical_and(pred > 0, gt > 0).float().sum()
        union = torch.logical_or(pred > 0, gt > 0).float().sum()
        if union > 0:
            ious.append(float((inter / union).detach().cpu()))
    if not ious:
        return {}
    return {
        "object_mask_iou_mean": float(np.mean(ious)),
        "object_mask_iou_p05": float(np.percentile(ious, 5)),
    }


def _copy_if_exists(src: str | Path | None, dst: Path) -> bool:
    if src is None:
        return False
    src = Path(src)
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def _read_ply_vertices(path: Path, max_points: int = 50000) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return read_ply_vertices(path, max_points=max_points)
    except Exception:
        return None


def _copy_validation_artifacts(
    aligned_dir: Path,
    output_dir: Path,
    hoi_data_file: str | Path | None,
    foundationpose_overlay_file: str | Path | None = None,
    stage_render_manifest_file: str | Path | None = None,
) -> dict:
    artifacts: dict[str, object] = {}
    artifacts["vggt_aligned_scene_ply"] = _copy_if_exists(
        aligned_dir / "aligned_scene.ply",
        output_dir / "vggt_aligned_scene.ply",
    )

    metadata = {}
    metadata_path = aligned_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as handle:
                metadata = json.load(handle)
        except Exception:
            metadata = {}
    vggt_dir = metadata.get("vggt_dir")
    if vggt_dir:
        artifacts["vggt_raw_scene_ply"] = _copy_if_exists(
            Path(vggt_dir) / "raw_scene.ply",
            output_dir / "vggt_raw_scene.ply",
        )

    sim3_path = aligned_dir / "alignment" / "sim3_vggt_to_blender.json"
    static_scene_ply = None
    if sim3_path.exists():
        try:
            with open(sim3_path, "r") as handle:
                sim3_data = json.load(handle)
            static_scene_ply = sim3_data.get("metadata", {}).get("static_scene_ply")
        except Exception:
            static_scene_ply = None
    artifacts["blend_scene_ply"] = _copy_if_exists(static_scene_ply, output_dir / "blend_scene.ply")
    if artifacts["blend_scene_ply"]:
        blend_pts = _read_ply_vertices(output_dir / "blend_scene.ply", max_points=50000)
        aligned_pts = _read_ply_vertices(output_dir / "vggt_aligned_scene.ply", max_points=50000)
        if blend_pts is not None and aligned_pts is not None:
            pts = np.concatenate([blend_pts, aligned_pts], axis=0)
            colors = np.concatenate(
                [
                    np.tile(np.asarray([[0, 180, 255]], dtype=np.uint8), (len(blend_pts), 1)),
                    np.tile(np.asarray([[255, 120, 0]], dtype=np.uint8), (len(aligned_pts), 1)),
                ],
                axis=0,
            )
            write_ply(output_dir / "alignment_overlay.ply", pts, colors)
            artifacts["alignment_overlay_ply"] = True
        else:
            artifacts["alignment_overlay_ply"] = False
    else:
        artifacts["alignment_overlay_ply"] = False

    fp_overlay = Path(foundationpose_overlay_file) if foundationpose_overlay_file else None
    if fp_overlay is not None and fp_overlay.exists():
        artifacts["foundationpose_overlay_mp4"] = _copy_if_exists(
            fp_overlay,
            output_dir / "foundationpose_overlay.mp4",
        )
        artifacts["foundationpose_overlay_source"] = str(fp_overlay)
        artifacts["foundationpose_overlay_note"] = "Copied from FoundationPose tracking output."
    else:
        artifacts["foundationpose_overlay_mp4"] = False

    stage_manifest = None
    if stage_render_manifest_file is not None and Path(stage_render_manifest_file).exists():
        try:
            with open(stage_render_manifest_file, "r") as handle:
                stage_manifest = json.load(handle)
            artifacts["stage_render_manifest_file"] = str(stage_render_manifest_file)
        except Exception:
            stage_manifest = None

    final_camera_file = None
    if isinstance(stage_manifest, dict):
        final_camera_file = (
            stage_manifest.get("visualization", {})
            .get("final_camera_view", {})
            .get("video_file")
        )
    if final_camera_file and Path(final_camera_file).exists():
        artifacts["smplx_overlay_mp4"] = _copy_if_exists(
            final_camera_file,
            output_dir / "smplx_overlay.mp4",
        )
        artifacts["smplx_overlay_source"] = str(final_camera_file)
        artifacts["smplx_overlay_trajectory_stage"] = "final"
        artifacts["smplx_overlay_note"] = "Rendered directly from final hoi_data.pkl."
        return artifacts

    if hoi_data_file is None or not Path(hoi_data_file).exists():
        artifacts["smplx_overlay_mp4"] = False
        return artifacts
    artifacts["smplx_overlay_mp4"] = False
    artifacts["smplx_overlay_note"] = (
        "No final-stage render manifest was supplied; legacy arbitrary-stage video fallback is disabled."
    )
    return artifacts


def _static_scene_ply_from_alignment(aligned_dir: Path, output_dir: Path) -> Path | None:
    copied = output_dir / "blend_scene.ply"
    if copied.exists():
        return copied
    sim3_path = aligned_dir / "alignment" / "sim3_vggt_to_blender.json"
    if not sim3_path.exists():
        return None
    try:
        with open(sim3_path, "r") as handle:
            sim3_data = json.load(handle)
        static_scene_ply = sim3_data.get("metadata", {}).get("static_scene_ply")
    except Exception:
        return None
    if static_scene_ply and Path(static_scene_ply).exists():
        return Path(static_scene_ply)
    return None


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if max_points > 0 and len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[idx]
    return points


def _axis_angle_root_rotations(hoi_data: dict) -> np.ndarray | None:
    poses = _to_numpy(hoi_data.get("human_data", {}).get("poses"))
    if poses is None or poses.ndim != 2 or poses.shape[1] < 3 or len(poses) == 0:
        return None
    return np.asarray([_axis_angle_to_matrix_np(pose[:3]) for pose in poses], dtype=np.float64)


def _rotation_velocity_deg(rotations: np.ndarray | None) -> np.ndarray:
    if rotations is None or len(rotations) < 2:
        return np.zeros((0,), dtype=np.float64)
    return np.asarray(
        [rotation_angle_deg(rotations[i], rotations[i + 1]) for i in range(len(rotations) - 1)],
        dtype=np.float64,
    )


def _translation_acceleration(values: np.ndarray | None) -> np.ndarray:
    if values is None or values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(values[2:] - 2.0 * values[1:-1] + values[:-2], axis=1)


def _rotation_acceleration_deg(rotations: np.ndarray | None) -> np.ndarray:
    vel = _rotation_velocity_deg(rotations)
    if len(vel) < 2:
        return np.zeros((0,), dtype=np.float64)
    return np.abs(vel[1:] - vel[:-1])


def _trajectory_temporal_diagnostics_from_hoi(hoi_data: dict, output_dir: Path) -> dict:
    val = hoi_data.get("validation_data", {}) if isinstance(hoi_data, dict) else {}
    human_root = _to_numpy(val.get("human_root_trans"))
    if human_root is None:
        human_root = _to_numpy(hoi_data.get("human_data", {}).get("trans"))
    obj_t = _to_numpy(hoi_data.get("obj_data", {}).get("obj_t"))
    obj_R = _to_numpy(hoi_data.get("obj_data", {}).get("obj_R"))

    human_t_acc = _translation_acceleration(human_root.astype(np.float64) if human_root is not None else None)
    human_R_acc = _rotation_acceleration_deg(_axis_angle_root_rotations(hoi_data))
    object_t_acc = _translation_acceleration(obj_t.astype(np.float64) if obj_t is not None else None)
    object_R_acc = _rotation_acceleration_deg(obj_R.astype(np.float64) if obj_R is not None else None)

    _plot_series(output_dir / "human_root_translation_acc.png", human_t_acc, "Human Root Translation Acceleration", "m/frame^2")
    _plot_series(output_dir / "human_root_rotation_acc.png", human_R_acc, "Human Root Rotation Acceleration", "deg/frame^2")
    _plot_series(output_dir / "object_translation_acc.png", object_t_acc, "Object Translation Acceleration", "m/frame^2")
    _plot_series(output_dir / "object_rotation_acc.png", object_R_acc, "Object Rotation Acceleration", "deg/frame^2")

    return {
        "human_root_translation_acc_mean_m": float(human_t_acc.mean()) if human_t_acc.size else 0.0,
        "human_root_translation_acc_p95_m": float(np.percentile(human_t_acc, 95)) if human_t_acc.size else 0.0,
        "human_root_rotation_acc_mean_deg": float(human_R_acc.mean()) if human_R_acc.size else 0.0,
        "human_root_rotation_acc_p95_deg": float(np.percentile(human_R_acc, 95)) if human_R_acc.size else 0.0,
        "object_translation_acc_mean_m": float(object_t_acc.mean()) if object_t_acc.size else 0.0,
        "object_translation_acc_p95_m": float(np.percentile(object_t_acc, 95)) if object_t_acc.size else 0.0,
        "object_rotation_acc_mean_deg": float(object_R_acc.mean()) if object_R_acc.size else 0.0,
        "object_rotation_acc_p95_deg": float(np.percentile(object_R_acc, 95)) if object_R_acc.size else 0.0,
    }


def _object_observation_diagnostics(
    observations_dir: str | Path | None,
    hoi_data: dict,
    output_dir: Path,
) -> dict:
    val = hoi_data.get("validation_data", {}) if isinstance(hoi_data, dict) else {}
    raw_count = _to_numpy(val.get("object_raw_point_count"))
    filtered_count = _to_numpy(val.get("object_filtered_point_count"))
    fused_count = _to_numpy(val.get("object_fused_point_count"))
    depth_weight = _to_numpy(val.get("object_depth_weights"))
    metadata = {}
    if observations_dir is not None:
        metadata_file = Path(observations_dir) / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as handle:
                    metadata = json.load(handle)
            except Exception:
                metadata = {}
    if raw_count is None and metadata.get("object_raw_point_count") is not None:
        raw_count = np.asarray(metadata.get("object_raw_point_count"), dtype=np.float64)
    if filtered_count is None and metadata.get("object_filtered_point_count") is not None:
        filtered_count = np.asarray(metadata.get("object_filtered_point_count"), dtype=np.float64)
    if fused_count is None and metadata.get("object_fused_point_count") is not None:
        fused_count = np.asarray(metadata.get("object_fused_point_count"), dtype=np.float64)
    if depth_weight is None:
        weights = metadata.get("quality_weights", {}).get("object_smoothed")
        if weights is not None:
            depth_weight = np.asarray(weights, dtype=np.float64)

    count_series = {}
    if raw_count is not None:
        count_series["raw"] = raw_count.reshape(-1)
    if filtered_count is not None:
        count_series["filtered"] = filtered_count.reshape(-1)
    if fused_count is not None:
        count_series["fused"] = fused_count.reshape(-1)
    if count_series:
        _plot_multi_series(output_dir / "object_point_count.png", count_series, "Object VGGT Point Count", "points/frame")
    if depth_weight is not None:
        _plot_series(output_dir / "object_depth_weight.png", depth_weight.reshape(-1), "Object Depth Observation Weight", "weight")
    contact_weight = _to_numpy(val.get("contact_weight"))
    grasp_weight = _to_numpy(val.get("grasp_weight"))
    if contact_weight is not None:
        _plot_series(output_dir / "contact_weight.png", contact_weight.reshape(-1), "Contact Weight Ramp", "weight")
    if grasp_weight is not None:
        _plot_series(output_dir / "grasp_weight.png", grasp_weight.reshape(-1), "Grasp Weight Ramp", "weight")

    overlay = dict(count_series)
    if depth_weight is not None:
        overlay["object_depth_weight"] = depth_weight.reshape(-1)
    if contact_weight is not None:
        overlay["contact_weight"] = contact_weight.reshape(-1)
    if grasp_weight is not None:
        overlay["grasp_weight"] = grasp_weight.reshape(-1)
    if overlay:
        _plot_multi_series(output_dir / "temporal_diagnostics.png", overlay, "Temporal Diagnostics", "normalized/count")

    metrics = {}
    for name, values in (("raw", raw_count), ("filtered", filtered_count), ("fused", fused_count)):
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        metrics[f"object_{name}_point_count_median"] = float(np.median(values)) if values.size else 0.0
        metrics[f"object_{name}_point_count_mean"] = float(np.mean(values)) if values.size else 0.0
        metrics[f"object_{name}_point_count_valid_frames"] = int(np.count_nonzero(values > 0))
    return metrics


def _load_frame_points(points_dir: Path, frame_idx: int, max_points: int = 5000) -> np.ndarray:
    path = points_dir / f"{frame_idx:06d}.npy"
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float32)
    try:
        return _sample_points(np.load(path), max_points)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)


def _write_debug_view_png(path: Path, groups: dict[str, tuple[np.ndarray, str]], axes: tuple[int, int], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    for label, (points, color) in groups.items():
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if points.size == 0:
            continue
        plt.scatter(points[:, axes[0]], points[:, axes[1]], s=1, c=color, label=label, alpha=0.75)
    plt.axis("equal")
    plt.title(title)
    if groups:
        plt.legend(loc="best", markerscale=4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _write_dynamic_observation_debug(
    observations_dir: str | Path | None,
    init_hoi_data: dict,
    final_hoi_data: dict,
    output_dir: Path,
    *,
    frame_stride: int = 10,
) -> dict:
    if observations_dir is None:
        return {}
    observations_dir = Path(observations_dir)
    if not observations_dir.exists():
        return {}
    init_val = init_hoi_data.get("validation_data", {}) if isinstance(init_hoi_data, dict) else {}
    final_val = final_hoi_data.get("validation_data", {}) if isinstance(final_hoi_data, dict) else {}

    def _seq(key, source):
        arr = _to_numpy(source.get(key))
        return arr if arr is not None and arr.ndim == 3 and arr.shape[-1] == 3 else None

    human_init = _seq("human_verts_sample_seq", init_val)
    human_final = _seq("human_verts_sample_seq", final_val)
    obj_init = _seq("obj_verts_sample_seq", init_val)
    obj_final = _seq("obj_verts_sample_seq", final_val)
    lengths = [len(x) for x in (human_init, human_final, obj_init, obj_final) if x is not None]
    if not lengths:
        return {}
    frame_count = max(lengths)
    debug_root = output_dir / "dynamic_points"
    written = 0
    for frame_idx in range(0, frame_count, max(1, int(frame_stride))):
        frame_dir = debug_root / f"frame_{frame_idx:03d}"
        human_raw = _load_frame_points(observations_dir / "human_raw_points", frame_idx)
        human_loss = _load_frame_points(observations_dir / "human_points", frame_idx)
        object_raw = _load_frame_points(observations_dir / "object_raw_points", frame_idx)
        object_filtered = _load_frame_points(observations_dir / "object_points", frame_idx)
        object_fused = _load_frame_points(observations_dir / "object_fused_points", frame_idx)
        items = {
            "human_raw.ply": human_raw,
            "human_loss_points.ply": human_loss,
            "object_raw.ply": object_raw,
            "object_filtered.ply": object_filtered,
            "object_fused.ply": object_fused,
        }
        if human_init is not None and frame_idx < len(human_init):
            items["human_init_mesh.ply"] = _sample_points(human_init[frame_idx], 2048)
        if human_final is not None and frame_idx < len(human_final):
            items["human_final_mesh.ply"] = _sample_points(human_final[frame_idx], 2048)
        if obj_init is not None and frame_idx < len(obj_init):
            items["object_init_mesh.ply"] = _sample_points(obj_init[frame_idx], 2048)
        if obj_final is not None and frame_idx < len(obj_final):
            items["object_final_mesh.ply"] = _sample_points(obj_final[frame_idx], 2048)
        for filename, points in items.items():
            write_ply(frame_dir / filename, points)

        groups = {
            "human_raw": (human_raw, "tab:blue"),
            "human_final": (items.get("human_final_mesh.ply", np.zeros((0, 3))), "tab:cyan"),
            "object_raw": (object_raw, "tab:orange"),
            "object_fused": (object_fused, "tab:red"),
            "object_final": (items.get("object_final_mesh.ply", np.zeros((0, 3))), "tab:green"),
        }
        _write_debug_view_png(frame_dir / "camera_view.png", groups, (0, 2), f"Frame {frame_idx} Camera/Side")
        _write_debug_view_png(frame_dir / "top_view.png", groups, (0, 1), f"Frame {frame_idx} Top")
        _write_debug_view_png(frame_dir / "side_view.png", groups, (1, 2), f"Frame {frame_idx} Side")
        written += 1
    return {
        "dynamic_points_dir": str(debug_root),
        "dynamic_points_frame_stride": int(frame_stride),
        "dynamic_points_frame_count": int(written),
    }


def _unavailable_static_scene_metrics(reason: str) -> dict:
    return {
        "static_scene_error_available": False,
        "static_scene_error_mean_m": None,
        "static_scene_error_median_m": None,
        "static_scene_error_p95_m": None,
        "static_scene_error_num_frames": 0,
        "static_scene_error_frame_ids": [],
        "static_scene_error_frame_coverage": 0.0,
        "static_scene_error_total_aligned_frames": None,
        "static_scene_error_inlier_ratio_3cm": None,
        "static_scene_error_inlier_ratio_5cm": None,
        "static_scene_error_inlier_ratio_10cm": None,
        "static_scene_error_method": None,
        "static_scene_error_unavailable_reason": reason,
    }


def _static_scene_metrics_from_aligned_geometry(
    aligned_dir: Path,
    output_dir: Path,
    *,
    masks_cache_file: str | Path | None = None,
    max_query_points_per_frame: int = 5000,
    max_scene_points: int = 250000,
) -> dict:
    """Measure aligned metric-depth background against the Blender static scene.

    ``metric_depth/depth.npy`` is camera-Z depth recomputed by the alignment
    stage from VGGT-Omega's official point map. It is therefore unprojected via
    the shared OpenCV metric-depth helper, rather than by guessing VGGT's raw
    depth convention.
    """
    sim3_path = aligned_dir / "alignment" / "sim3_vggt_to_blender.json"
    if not sim3_path.exists():
        return _unavailable_static_scene_metrics("alignment Sim(3) metadata is missing")
    try:
        with open(sim3_path, "r") as handle:
            sim3_metadata = json.load(handle).get("metadata", {})
    except Exception as exc:
        return _unavailable_static_scene_metrics(f"could not read alignment metadata: {exc}")

    scene_path = _static_scene_ply_from_alignment(aligned_dir, output_dir)
    if scene_path is None or not scene_path.exists():
        return _unavailable_static_scene_metrics("Blender static_scene.ply is missing")
    masks_cache_file = masks_cache_file or sim3_metadata.get("masks_cache_file")
    if masks_cache_file is None or not Path(masks_cache_file).exists():
        return _unavailable_static_scene_metrics("multi-frame human/object mask cache is missing")

    required = {
        "metric depth": aligned_dir / "metric_depth" / "depth.npy",
        "intrinsics": aligned_dir / "intrinsics.npy",
        "camera poses": aligned_dir / "c2w_blender.npy",
        "confidence": aligned_dir / "confidence.npy",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        return _unavailable_static_scene_metrics(f"aligned geometry is missing: {', '.join(missing)}")

    try:
        depth = np.load(required["metric depth"], mmap_mode="r")
        intrinsics = np.load(required["intrinsics"], mmap_mode="r")
        c2w = np.load(required["camera poses"], mmap_mode="r")
        confidence = np.load(required["confidence"], mmap_mode="r")
        masks = load_masks_from_cache(str(masks_cache_file))
        scene_points = read_ply_vertices(scene_path, max_points=max_scene_points)
    except Exception as exc:
        return _unavailable_static_scene_metrics(f"could not load geometry validation inputs: {exc}")

    if depth.ndim != 3:
        return _unavailable_static_scene_metrics(f"metric depth has invalid shape {depth.shape}")
    frame_count, height, width = depth.shape
    if intrinsics.shape != (frame_count, 3, 3):
        return _unavailable_static_scene_metrics(f"intrinsics have invalid shape {intrinsics.shape}")
    if c2w.shape != (frame_count, 4, 4):
        return _unavailable_static_scene_metrics(f"camera poses have invalid shape {c2w.shape}")
    if confidence.shape != depth.shape:
        return _unavailable_static_scene_metrics(f"confidence has invalid shape {confidence.shape}")

    try:
        from scipy.spatial import cKDTree

        scene_tree = cKDTree(np.asarray(scene_points, dtype=np.float64))

        def nearest_distance(points: np.ndarray) -> np.ndarray:
            distances, _ = scene_tree.query(np.asarray(points, dtype=np.float64), k=1, workers=-1)
            return np.asarray(distances, dtype=np.float32)

        distance_backend = "scipy_cKDTree"
    except Exception:

        def nearest_distance(points: np.ndarray) -> np.ndarray:
            return _min_distances_to_points(points, scene_points, chunk=256)

        distance_backend = "chunked_numpy"

    confidence_percentile = float(sim3_metadata.get("confidence_percentile", 50.0))
    erode_pixels = int(sim3_metadata.get("erode_pixels", 5))
    frame_keys = []
    if isinstance(masks, dict):
        for key in masks:
            try:
                frame_idx = int(key)
            except (TypeError, ValueError):
                continue
            if 0 <= frame_idx < frame_count:
                frame_keys.append((frame_idx, key))
    frame_keys.sort(key=lambda item: item[0])

    frame_ids = []
    per_frame_mean = []
    per_frame_median = []
    per_frame_p95 = []
    all_distances = []
    for frame_idx, mask_key in frame_keys:
        frame_masks = masks[mask_key]
        if not isinstance(frame_masks, dict):
            continue
        object_mask = frame_masks.get(0, frame_masks.get("0"))
        human_mask = frame_masks.get(1, frame_masks.get("1"))
        if object_mask is None or human_mask is None:
            continue
        object_mask = resize_mask(object_mask, (height, width))
        human_mask = resize_mask(human_mask, (height, width))
        # Erode the static region itself so human/object boundary pixels are excluded.
        static_mask = erode_mask(~(object_mask | human_mask), erode_pixels)

        depth_t = np.asarray(depth[frame_idx], dtype=np.float32)
        confidence_t = np.asarray(confidence[frame_idx], dtype=np.float32)
        valid_depth = np.isfinite(depth_t) & (depth_t > 0)
        valid_confidence = np.isfinite(confidence_t)
        if valid_confidence.any():
            cutoff = np.percentile(confidence_t[valid_confidence], confidence_percentile)
            valid_confidence &= confidence_t >= cutoff
        valid = static_mask & valid_depth & valid_confidence
        if not valid.any():
            continue

        points_world = unproject_opencv_depth_to_world(
            depth_t,
            np.asarray(intrinsics[frame_idx]),
            np.asarray(c2w[frame_idx]),
        )[valid]
        points_world = points_world[np.isfinite(points_world).all(axis=1)]
        if max_query_points_per_frame > 0 and len(points_world) > max_query_points_per_frame:
            rng = np.random.default_rng(frame_idx)
            choice = rng.choice(len(points_world), size=max_query_points_per_frame, replace=False)
            points_world = points_world[choice]
        if len(points_world) == 0:
            continue

        distances = nearest_distance(points_world)
        distances = distances[np.isfinite(distances)]
        if distances.size == 0:
            continue
        frame_ids.append(frame_idx)
        per_frame_mean.append(float(np.mean(distances)))
        per_frame_median.append(float(np.median(distances)))
        per_frame_p95.append(float(np.percentile(distances, 95)))
        all_distances.append(distances)

    if not all_distances:
        return _unavailable_static_scene_metrics("no valid masked static points were reconstructed")

    distances = np.concatenate(all_distances)
    inlier_thresholds = {"3cm": 0.03, "5cm": 0.05, "10cm": 0.10}
    inlier_ratios = {
        name: float(np.mean(distances <= threshold))
        for name, threshold in inlier_thresholds.items()
    }
    per_frame_inlier_ratios = {
        name: [float(np.mean(frame_distances <= threshold)) for frame_distances in all_distances]
        for name, threshold in inlier_thresholds.items()
    }
    _plot_multi_series(
        output_dir / "static_scene_error.png",
        {
            "mean": np.asarray(per_frame_mean),
            "median": np.asarray(per_frame_median),
            "p95": np.asarray(per_frame_p95),
        },
        "Aligned Metric Static Scene Distance to Blender Reference",
        "m",
    )
    _plot_multi_series(
        output_dir / "table_stability.png",
        {
            "median": np.asarray(per_frame_median),
            "p95": np.asarray(per_frame_p95),
        },
        "Static Scene / Table Stability",
        "m",
    )
    return {
        "static_scene_error_available": True,
        "static_scene_error_unavailable_reason": None,
        "static_scene_error_mean_m": float(np.mean(distances)),
        "static_scene_error_median_m": float(np.median(distances)),
        "static_scene_error_p95_m": float(np.percentile(distances, 95)),
        "static_scene_error_num_frames": int(len(frame_ids)),
        "static_scene_error_frame_ids": frame_ids,
        "static_scene_error_per_frame_mean_m": per_frame_mean,
        "static_scene_error_per_frame_median_m": per_frame_median,
        "static_scene_error_per_frame_p95_m": per_frame_p95,
        "static_scene_error_inlier_ratio_3cm": inlier_ratios["3cm"],
        "static_scene_error_inlier_ratio_5cm": inlier_ratios["5cm"],
        "static_scene_error_inlier_ratio_10cm": inlier_ratios["10cm"],
        "static_scene_error_per_frame_inlier_ratio_3cm": per_frame_inlier_ratios["3cm"],
        "static_scene_error_per_frame_inlier_ratio_5cm": per_frame_inlier_ratios["5cm"],
        "static_scene_error_per_frame_inlier_ratio_10cm": per_frame_inlier_ratios["10cm"],
        "static_scene_error_frame_coverage": float(len(frame_ids) / max(frame_count, 1)),
        "static_scene_error_total_aligned_frames": int(frame_count),
        "static_scene_error_mask_cache_file": str(Path(masks_cache_file)),
        "static_scene_error_static_scene_ply": str(scene_path),
        "static_scene_error_confidence_percentile": confidence_percentile,
        "static_scene_error_static_mask_erode_pixels": erode_pixels,
        "static_scene_error_points_per_frame_max": int(max_query_points_per_frame),
        "static_scene_error_distance_backend": distance_backend,
        "static_scene_error_method": (
            "aligned_metric_camera_z_unprojection_masked_static_to_blender_static_ply_nearest_neighbor"
        ),
        "static_scene_error_depth_convention": (
            "metric_depth camera-Z in aligned OpenCV camera; metric depth was recomputed from "
            "VGGT-Omega official point-map unprojection"
        ),
    }


def _static_scene_metrics_from_observations(
    observations_dir: str | Path | None,
    aligned_dir: Path,
    output_dir: Path,
    *,
    max_query_points: int = 5000,
    max_scene_points: int = 50000,
) -> dict:
    if observations_dir is None:
        return {}
    static_dir = Path(observations_dir) / "static_points"
    if not static_dir.is_dir():
        return {}
    scene_path = _static_scene_ply_from_alignment(aligned_dir, output_dir)
    if scene_path is None:
        return {}
    scene_pts = _read_ply_vertices(scene_path, max_points=max_scene_points)
    if scene_pts is None or len(scene_pts) == 0:
        return {}

    means = []
    medians = []
    p95s = []
    frame_ids = []
    for path in sorted(static_dir.glob("*.npy")):
        try:
            pts = _sample_points(np.load(path), max_query_points)
        except Exception:
            continue
        if len(pts) == 0:
            continue
        d = _min_distances_to_points(pts, scene_pts, chunk=256)
        if d.size == 0:
            continue
        means.append(float(np.mean(d)))
        medians.append(float(np.median(d)))
        p95s.append(float(np.percentile(d, 95)))
        try:
            frame_ids.append(int(path.stem))
        except Exception:
            frame_ids.append(len(frame_ids))

    if not means:
        return {}
    _plot_multi_series(
        output_dir / "static_scene_error.png",
        {
            "mean": np.asarray(means),
            "median": np.asarray(medians),
            "p95": np.asarray(p95s),
        },
        "Static Scene Distance to Blender Reference",
        "m",
    )
    _plot_multi_series(
        output_dir / "table_stability.png",
        {
            "median": np.asarray(medians),
            "p95": np.asarray(p95s),
        },
        "Static Scene / Table Stability",
        "m",
    )
    return {
        "static_scene_error_available": True,
        "static_scene_error_unavailable_reason": None,
        "static_scene_error_mean_m": float(np.mean(means)),
        "static_scene_error_median_m": float(np.median(medians)),
        "static_scene_error_p95_m": float(np.percentile(p95s, 95)),
        "static_scene_error_num_frames": int(len(means)),
        "static_scene_error_frame_ids": frame_ids,
        "static_scene_error_method": "vggt_static_points_to_blender_static_ply_nearest_neighbor",
    }


def _trimmed_distance_summary(
    mesh_points: np.ndarray,
    observation_points: np.ndarray,
    *,
    trim_pct: float = 0.2,
) -> tuple[float, float, float] | None:
    mesh_points = np.asarray(mesh_points, dtype=np.float32).reshape(-1, 3)
    observation_points = np.asarray(observation_points, dtype=np.float32).reshape(-1, 3)
    if len(mesh_points) == 0 or len(observation_points) == 0:
        return None

    mesh_to_obs = _min_distances_to_points(mesh_points, observation_points)
    obs_to_mesh = _min_distances_to_points(observation_points, mesh_points)
    if mesh_to_obs.size == 0 or obs_to_mesh.size == 0:
        return None

    trim_pct = float(np.clip(trim_pct, 0.0, 0.95))

    def _trim(values: np.ndarray) -> np.ndarray:
        keep = max(1, int(values.size * (1.0 - trim_pct)))
        return np.partition(values, keep - 1)[:keep]

    distances = np.concatenate([_trim(mesh_to_obs), _trim(obs_to_mesh)], axis=0)
    if distances.size == 0:
        return None
    return float(np.mean(distances)), float(np.median(distances)), float(np.percentile(distances, 95))


def _depth_observation_metrics_from_hoi(
    observations_dir: str | Path | None,
    hoi_data: dict,
    output_dir: Path,
    *,
    max_mesh_points: int = 2048,
    max_observation_points: int = 5000,
    trim_pct: float = 0.2,
    artifact_prefix: str = "",
) -> dict:
    """Compare optimized mesh samples against saved VGGT world-space observations."""
    if observations_dir is None or not isinstance(hoi_data, dict):
        return {}
    val = hoi_data.get("validation_data", {})
    if not isinstance(val, dict):
        return {}

    observations_dir = Path(observations_dir)
    object_points_dir = observations_dir / "object_fused_points"
    if not object_points_dir.is_dir():
        object_points_dir = observations_dir / "object_points"
    specs = [
        (
            "human",
            "human_verts_sample_seq",
            observations_dir / "human_points",
            output_dir / f"{artifact_prefix}human_depth_error.png",
            "Human Depth Chamfer",
        ),
        (
            "object",
            "obj_verts_sample_seq",
            object_points_dir,
            output_dir / f"{artifact_prefix}object_depth_error.png",
            "Object Depth Chamfer",
        ),
    ]

    metrics: dict[str, object] = {}
    for prefix, mesh_key, points_dir, plot_path, plot_title in specs:
        mesh_seq = _to_numpy(val.get(mesh_key))
        if mesh_seq is None or mesh_seq.ndim != 3 or mesh_seq.shape[-1] != 3:
            continue
        if not points_dir.is_dir():
            continue

        means = []
        medians = []
        p95s = []
        frame_ids = []
        for path in sorted(points_dir.glob("*.npy")):
            try:
                frame_idx = int(path.stem)
            except Exception:
                continue
            if frame_idx < 0 or frame_idx >= mesh_seq.shape[0]:
                continue
            try:
                obs_points = _sample_points(np.load(path), max_observation_points)
            except Exception:
                continue
            mesh_points = _sample_points(mesh_seq[frame_idx], max_mesh_points)
            summary = _trimmed_distance_summary(mesh_points, obs_points, trim_pct=trim_pct)
            if summary is None:
                continue
            mean_d, median_d, p95_d = summary
            means.append(mean_d)
            medians.append(median_d)
            p95s.append(p95_d)
            frame_ids.append(frame_idx)

        if not means:
            continue
        _plot_multi_series(
            plot_path,
            {
                "mean": np.asarray(means),
                "median": np.asarray(medians),
                "p95": np.asarray(p95s),
            },
            plot_title,
            "m",
        )

        metric_prefix = f"{prefix}_depth_chamfer"
        metrics[f"{metric_prefix}_m"] = float(np.mean(means))
        metrics[f"{metric_prefix}_median_m"] = float(np.median(medians))
        metrics[f"{metric_prefix}_p95_m"] = float(np.percentile(p95s, 95))
        metrics[f"{metric_prefix}_num_frames"] = int(len(means))
        metrics[f"{metric_prefix}_frame_ids"] = frame_ids
        metrics[f"{metric_prefix}_method"] = (
            "optimized_mesh_samples_to_vggt_observation_bidirectional_trimmed_nn"
        )
    return metrics


def _stage_scalar_comparison(
    init_metrics: dict,
    final_metrics: dict,
    key: str,
    *,
    init_source: str | None,
    final_source: str | None,
    unit: str,
) -> dict:
    """Return init/final values with an explicit final-minus-init delta."""
    init_value = init_metrics.get(key)
    final_value = final_metrics.get(key)
    available = init_value is not None and final_value is not None
    result = {
        "available": bool(available),
        "delta_definition": "final_minus_init",
        "negative_delta_means_improvement": True,
        "unit": unit,
        "init": {
            "trajectory_stage": "init",
            "trajectory_source": init_source,
            "value": None if init_value is None else float(init_value),
        },
        "final": {
            "trajectory_stage": "final",
            "trajectory_source": final_source,
            "value": None if final_value is None else float(final_value),
        },
        "delta_final_minus_init": None,
        "improvement": None,
    }
    if available:
        result["delta_final_minus_init"] = float(final_value) - float(init_value)
        result["improvement"] = float(init_value) - float(final_value)
    return result


def _build_stage_metric_details(
    *,
    init_hoi_data_file: str | Path | None,
    final_hoi_data_file: str | Path | None,
    observations_dir: str | Path | None,
    foundationpose_pose_camera_file: str | Path | None,
    foundationpose_pose_world_file: str | Path | None,
    foundationpose_overlay_file: str | Path | None,
    human_motion_file: str | Path | None,
    init_depth_metrics: dict,
    final_depth_metrics: dict,
    init_contact_metrics: dict,
    final_contact_metrics: dict,
) -> dict:
    init_source = str(init_hoi_data_file) if init_hoi_data_file is not None else None
    final_source = str(final_hoi_data_file) if final_hoi_data_file is not None else None
    observations = Path(observations_dir) if observations_dir is not None else None
    human_observations = str(observations / "human_points") if observations is not None else None
    object_observations = None
    if observations is not None:
        fused = observations / "object_fused_points"
        object_observations = str(fused if fused.is_dir() else observations / "object_points")

    human_depth = _stage_scalar_comparison(
        init_depth_metrics,
        final_depth_metrics,
        "human_depth_chamfer_m",
        init_source=init_source,
        final_source=final_source,
        unit="m",
    )
    human_depth["observation_source"] = human_observations
    object_depth = _stage_scalar_comparison(
        init_depth_metrics,
        final_depth_metrics,
        "object_depth_chamfer_m",
        init_source=init_source,
        final_source=final_source,
        unit="m",
    )
    object_depth["observation_source"] = object_observations

    contact = _stage_scalar_comparison(
        init_contact_metrics,
        final_contact_metrics,
        "contact_distance_mean_m",
        init_source=init_source,
        final_source=final_source,
        unit="m",
    )
    contact["inputs"] = "hand_joints_seq_vs_obj_verts_sample_seq"

    grasp_translation = _stage_scalar_comparison(
        init_contact_metrics,
        final_contact_metrics,
        "grasp_relative_translation_std_m",
        init_source=init_source,
        final_source=final_source,
        unit="m",
    )
    grasp_rotation_std = _stage_scalar_comparison(
        init_contact_metrics,
        final_contact_metrics,
        "grasp_relative_rotation_std_deg",
        init_source=init_source,
        final_source=final_source,
        unit="deg",
    )
    grasp_rotation_mean = _stage_scalar_comparison(
        init_contact_metrics,
        final_contact_metrics,
        "grasp_relative_rotation_mean_deg",
        init_source=init_source,
        final_source=final_source,
        unit="deg",
    )

    return {
        "human_depth_chamfer": human_depth,
        "object_depth_chamfer": object_depth,
        "contact_distance": contact,
        "grasp_relative": {
            "trajectory_stage": "init_vs_final",
            "trajectory_sources": {
                "init": init_source,
                "final": final_source,
            },
            "translation_std": grasp_translation,
            "rotation_std": grasp_rotation_std,
            "rotation_mean": grasp_rotation_mean,
        },
        "foundationpose_tracking": {
            "trajectory_stage": "raw_foundationpose",
            "trajectory_source": (
                str(foundationpose_pose_camera_file)
                if foundationpose_pose_camera_file is not None
                else None
            ),
            "overlay_source": (
                str(foundationpose_overlay_file)
                if foundationpose_overlay_file is not None
                else None
            ),
            "coordinate_space": "opencv_camera",
        },
        "initializer_human_motion": {
            "trajectory_stage": "init",
            "trajectory_source": str(human_motion_file) if human_motion_file is not None else None,
            "coordinate_space": "blender_metric_world",
        },
        "initializer_object_motion": {
            "trajectory_stage": "raw_foundationpose_world",
            "trajectory_source": (
                str(foundationpose_pose_world_file)
                if foundationpose_pose_world_file is not None
                else None
            ),
            "coordinate_space": "blender_metric_world",
        },
    }


def write_dynamic_camera_validation(
    aligned_dir: str | Path,
    output_dir: str | Path,
    *,
    fps: float | None = None,
    object_pose_file: str | Path | None = None,
    human_motion_file: str | Path | None = None,
    hoi_data_file: str | Path | None = None,
    init_hoi_data_file: str | Path | None = None,
    foundationpose_overlay_file: str | Path | None = None,
    foundationpose_pose_camera_file: str | Path | None = None,
    stage_render_manifest_file: str | Path | None = None,
    masks_cache_file: str | Path | None = None,
    obj_path: str | Path | None = None,
    observations_dir: str | Path | None = None,
    device: str = "cuda",
    precontact_end: int | None = None,
    validation_scope: str | None = None,
) -> dict:
    """Write validation plots and ``metrics.json`` for a VGGT dynamic run."""
    if validation_scope is None:
        validation_scope = (
            "full"
            if any(
                value is not None
                for value in (object_pose_file, human_motion_file, hoi_data_file, observations_dir)
            )
            else "geometry"
        )
    if validation_scope not in {"geometry", "full"}:
        raise ValueError(f"Unsupported validation_scope: {validation_scope!r}")
    aligned_dir = Path(aligned_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    c2w = np.load(aligned_dir / "c2w_blender.npy")
    motion = camera_motion_metrics(c2w, fps=fps)
    centers = np.asarray(motion["camera_center"], dtype=np.float64)
    trans_speed = np.asarray(motion["camera_translation_speed_mps"], dtype=np.float64)
    rot_speed = np.asarray(motion["camera_rotation_speed_degps"], dtype=np.float64)

    _plot_camera_top_side(output_dir, centers)
    _plot_series(output_dir / "camera_speed.png", trans_speed, "Camera Translation Speed", "m/s")
    _plot_series(output_dir / "camera_rotation_speed.png", rot_speed, "Camera Rotation Speed", "deg/s")
    _plot_series(
        output_dir / "camera_acceleration.png",
        np.asarray(motion.get("camera_acceleration_delta_m", []), dtype=np.float64),
        "Camera Translation Acceleration",
        "delta m/frame^2",
    )

    hoi_data = _load_hoi_data(hoi_data_file)
    init_hoi_data = _load_hoi_data(init_hoi_data_file)
    artifact_metrics = _copy_validation_artifacts(
        aligned_dir,
        output_dir,
        hoi_data_file,
        foundationpose_overlay_file=foundationpose_overlay_file,
        stage_render_manifest_file=stage_render_manifest_file,
    )

    metrics = {
        **artifact_metrics,
        **_load_sim3_metrics(aligned_dir),
        **_load_alignment_provenance_metrics(aligned_dir, c2w_frame_count=int(c2w.shape[0])),
        "validation_scope": validation_scope,
        "aligned_dir": str(aligned_dir),
        "frame_count": int(c2w.shape[0]),
        "camera_jump_frames": motion["camera_jump_frames"],
        "camera_translation_speed_mean_mps": float(trans_speed.mean()) if trans_speed.size else 0.0,
        "camera_translation_speed_p95_mps": float(np.percentile(trans_speed, 95)) if trans_speed.size else 0.0,
        "camera_rotation_speed_mean_degps": float(rot_speed.mean()) if rot_speed.size else 0.0,
        "camera_rotation_speed_p95_degps": float(np.percentile(rot_speed, 95)) if rot_speed.size else 0.0,
        **_unavailable_static_scene_metrics("geometry validation has not been evaluated"),
        "object_mask_iou_mean": 0.0,
        "human_depth_chamfer_m": 0.0,
        "object_depth_chamfer_m": 0.0,
        "left_foot_slip_max_m": 0.0,
        "right_foot_slip_max_m": 0.0,
        "contact_distance_mean_m": 0.0,
        "penetration_ratio": 0.0,
    }
    metrics.update(_object_static_metrics(object_pose_file, precontact_end=precontact_end))
    metrics.update(_load_hoi_eval_metrics(hoi_data, precontact_end=precontact_end))
    metrics.update(
        _object_trajectory_metrics_from_pose_file(
            object_pose_file,
            output_dir,
            precontact_end=precontact_end,
        )
    )
    metrics.update(
        _object_trajectory_metrics_from_hoi(
            hoi_data,
            output_dir,
            precontact_end=precontact_end,
        )
    )
    metrics.update(
        _human_motion_metrics_from_cache(
            human_motion_file,
            output_dir,
            fps=fps,
        )
    )
    static_scene_metrics = _static_scene_metrics_from_aligned_geometry(
        aligned_dir,
        output_dir,
        masks_cache_file=masks_cache_file,
    )
    if not static_scene_metrics.get("static_scene_error_available"):
        observation_static_metrics = _static_scene_metrics_from_observations(
            observations_dir,
            aligned_dir,
            output_dir,
        )
        if observation_static_metrics:
            static_scene_metrics = observation_static_metrics
    metrics.update(static_scene_metrics)
    final_depth_metrics = _depth_observation_metrics_from_hoi(
        observations_dir,
        hoi_data,
        output_dir,
    )
    init_depth_metrics = _depth_observation_metrics_from_hoi(
        observations_dir,
        init_hoi_data,
        output_dir,
        artifact_prefix="init_",
    )
    metrics.update(final_depth_metrics)
    metrics.update(_foot_slip_metrics_from_hoi(hoi_data, output_dir))
    final_contact_metrics = _contact_metrics_from_hoi(hoi_data, output_dir)
    init_contact_metrics = _contact_metrics_from_hoi(
        init_hoi_data,
        output_dir,
        artifact_prefix="init_",
    )
    metrics.update(final_contact_metrics)
    metrics.update(_trajectory_temporal_diagnostics_from_hoi(hoi_data, output_dir))
    metrics.update(_object_observation_diagnostics(observations_dir, hoi_data, output_dir))
    metrics.update(_write_dynamic_observation_debug(observations_dir, init_hoi_data, hoi_data, output_dir))
    metrics.update(
        _object_mask_iou_from_hoi(
            aligned_dir,
            hoi_data_file,
            masks_cache_file,
            obj_path,
            device=device,
        )
    )
    metrics.setdefault("object_static_pos_std_m", 0.0)
    metrics.setdefault("object_static_rot_std_deg", 0.0)
    if observations_dir is not None:
        metrics["observations_dir"] = str(observations_dir)
    if hoi_data_file is not None:
        metrics["hoi_data_file"] = str(hoi_data_file)
    if object_pose_file is not None:
        metrics["object_pose_file"] = str(object_pose_file)
        metrics["object_pose_coordinate_space"] = (
            "blender_world" if "poses_in_world" in Path(object_pose_file).name else "camera"
        )
    if human_motion_file is not None:
        metrics.setdefault("human_motion_file", str(human_motion_file))
        metrics["human_motion_trajectory_stage"] = "init"
        metrics["human_motion_scope"] = "initializer_diagnostic_only"
    if object_pose_file is not None:
        metrics["object_pose_trajectory_stage"] = "raw_foundationpose_world"
        metrics["object_pose_scope"] = "initializer_diagnostic_only"
    if init_hoi_data_file is not None:
        metrics["init_hoi_data_file"] = str(init_hoi_data_file)
    if stage_render_manifest_file is not None:
        metrics["stage_render_manifest_file"] = str(stage_render_manifest_file)

    visualization = {}
    if stage_render_manifest_file is not None and Path(stage_render_manifest_file).exists():
        try:
            with open(stage_render_manifest_file, "r") as handle:
                visualization = json.load(handle).get("visualization", {})
        except Exception:
            visualization = {}
    metrics["metrics_schema_version"] = 3
    metrics["visualization"] = visualization
    metrics["metrics"] = _build_stage_metric_details(
        init_hoi_data_file=init_hoi_data_file,
        final_hoi_data_file=hoi_data_file,
        observations_dir=observations_dir,
        foundationpose_pose_camera_file=foundationpose_pose_camera_file,
        foundationpose_pose_world_file=object_pose_file,
        foundationpose_overlay_file=foundationpose_overlay_file,
        human_motion_file=human_motion_file,
        init_depth_metrics=init_depth_metrics,
        final_depth_metrics=final_depth_metrics,
        init_contact_metrics=init_contact_metrics,
        final_contact_metrics=final_contact_metrics,
    )
    _write_scalar_diagnostic_plots(output_dir, metrics)

    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(output_dir / "camera_motion.json", "w") as handle:
        json.dump(motion, handle, indent=2)
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate VGGT dynamic-camera reconstruction")
    parser.add_argument("--aligned_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--object_pose_file", default=None)
    parser.add_argument("--human_motion_file", default=None)
    parser.add_argument("--hoi_data_file", default=None)
    parser.add_argument("--init_hoi_data_file", default=None)
    parser.add_argument("--foundationpose_overlay_file", default=None)
    parser.add_argument("--foundationpose_pose_camera_file", default=None)
    parser.add_argument("--stage_render_manifest_file", default=None)
    parser.add_argument("--masks_cache_file", default=None)
    parser.add_argument("--obj_path", default=None)
    parser.add_argument("--observations_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precontact_end", type=int, default=None)
    parser.add_argument("--validation_scope", choices=("geometry", "full"), default=None)
    args = parser.parse_args()
    write_dynamic_camera_validation(
        args.aligned_dir,
        args.output_dir,
        fps=args.fps,
        object_pose_file=args.object_pose_file,
        human_motion_file=args.human_motion_file,
        hoi_data_file=args.hoi_data_file,
        init_hoi_data_file=args.init_hoi_data_file,
        foundationpose_overlay_file=args.foundationpose_overlay_file,
        foundationpose_pose_camera_file=args.foundationpose_pose_camera_file,
        stage_render_manifest_file=args.stage_render_manifest_file,
        masks_cache_file=args.masks_cache_file,
        obj_path=args.obj_path,
        observations_dir=args.observations_dir,
        device=args.device,
        precontact_end=args.precontact_end,
        validation_scope=args.validation_scope,
    )


if __name__ == "__main__":
    main()
