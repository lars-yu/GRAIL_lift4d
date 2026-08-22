#!/usr/bin/env python3
"""Run formal fixed-camera GRAIL optimization with a real Lift4D depth prior."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grail.core.io import save_hoi_data
from grail.optimization.hoi_optimizer import HOIOptimizer
from grail.optimization.loss_computer import LossComputer
from grail.optimization.approach import hand_to_mesh_surface_distance
from grail.rendering.camera import project_world_to_screen, transform_world_to_camera


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


def _inverse_vggt_preprocess(
    image: torch.Tensor, transform: dict, source_hw: tuple[int, int]
) -> torch.Tensor:
    source_h, source_w = source_hw
    if list(transform["source_size_hw"]) != [source_h, source_w]:
        raise ValueError(
            "VGGT source size mismatch: "
            f"metadata={transform['source_size_hw']} expected={[source_h, source_w]}"
        )

    out_h, out_w = map(int, transform["output_size_hw"])
    if tuple(image.shape) != (out_h, out_w):
        raise ValueError(
            f"VGGT output shape mismatch: tensor={tuple(image.shape)} metadata={(out_h, out_w)}"
        )
    left, top, right, bottom = map(int, transform["pad_ltrb"])
    unpadded = image[top : out_h - bottom, left : out_w - right]
    resized_h, resized_w = map(int, transform["resized_size_hw"])
    if tuple(unpadded.shape) != (resized_h, resized_w):
        raise ValueError(
            "VGGT unpadded shape mismatch: "
            f"tensor={tuple(unpadded.shape)} metadata={(resized_h, resized_w)}"
        )

    crop_x, crop_y, crop_w, crop_h = map(int, transform["crop_xywh"])
    restored_crop = F.interpolate(
        unpadded[None, None], size=(crop_h, crop_w), mode="bilinear", align_corners=False
    )[0, 0]
    restored = image.new_zeros((source_h, source_w))
    restored[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w] = restored_crop
    return restored


def _load_real_vggt_depth(
    cache_dir: str,
    expected_video: str,
    data,
    device: str,
    confidence_percentile: float,
) -> tuple[list[torch.Tensor], dict]:
    metadata_path = _real_file(os.path.join(cache_dir, "metadata.json"), "VGGT metadata")
    depth_path = _real_file(os.path.join(cache_dir, "depth.npy"), "VGGT depth")
    confidence_path = _real_file(
        os.path.join(cache_dir, "confidence.npy"), "VGGT confidence"
    )
    intrinsics_path = _real_file(
        os.path.join(cache_dir, "intrinsics_original.npy"), "VGGT original intrinsics"
    )
    with open(metadata_path, "r") as handle:
        metadata = json.load(handle)

    source_video = os.path.abspath(metadata.get("source_video", ""))
    if source_video != os.path.abspath(expected_video):
        raise ValueError(
            f"VGGT source video mismatch: metadata={source_video} expected={expected_video}"
        )
    if metadata.get("backend") != "vggt" or metadata.get("use_all_frames") is not True:
        raise ValueError("VGGT cache must use backend=vggt and use_all_frames=true")
    frame_num = int(data.frame_num)
    if int(metadata.get("num_frames", -1)) != frame_num:
        raise ValueError(
            f"VGGT frame count mismatch: metadata={metadata.get('num_frames')} GRAIL={frame_num}"
        )

    transforms = metadata.get("vggt_preprocess")
    if not isinstance(transforms, list) or len(transforms) != frame_num:
        raise ValueError("VGGT metadata is missing one preprocess transform per frame")
    depth = np.load(depth_path, mmap_mode="r")
    confidence = np.load(confidence_path, mmap_mode="r")
    intrinsics = np.load(intrinsics_path, mmap_mode="r")
    if depth.shape[0] != frame_num or confidence.shape != depth.shape:
        raise ValueError(
            f"VGGT array mismatch: depth={depth.shape} confidence={confidence.shape}"
        )
    if intrinsics.shape != (frame_num, 3, 3):
        raise ValueError(f"VGGT original intrinsics must be [T,3,3], got {intrinsics.shape}")

    source_h = int(data.camera.frame_height)
    source_w = int(data.camera.frame_width)
    restored_depths: list[torch.Tensor] = []
    confidence_thresholds: list[float] = []
    object_depth_medians: list[float] = []
    for frame_idx in range(frame_num):
        depth_i = torch.from_numpy(np.asarray(depth[frame_idx]).copy()).float()
        confidence_i = torch.from_numpy(np.asarray(confidence[frame_idx]).copy()).float()
        depth_full = _inverse_vggt_preprocess(
            depth_i, transforms[frame_idx], (source_h, source_w)
        )
        confidence_full = _inverse_vggt_preprocess(
            confidence_i, transforms[frame_idx], (source_h, source_w)
        )
        object_mask = torch.from_numpy(np.asarray(data.obj.masks[frame_idx])).squeeze().bool()
        valid_geometry = torch.isfinite(depth_full) & (depth_full > 0)
        object_confidence = confidence_full[object_mask & valid_geometry]
        if object_confidence.numel() < 32:
            raise ValueError(
                f"VGGT frame {frame_idx} has only {object_confidence.numel()} finite object pixels"
            )
        threshold = float(
            torch.quantile(object_confidence, confidence_percentile / 100.0)
        )
        confidence_thresholds.append(threshold)
        valid = valid_geometry & (confidence_full >= threshold)
        depth_full = torch.where(valid, depth_full, torch.zeros_like(depth_full))

        object_values = depth_full[object_mask & valid]
        if object_values.numel() < 32:
            raise ValueError(
                f"VGGT frame {frame_idx} has only {object_values.numel()} valid object pixels"
            )
        object_depth_medians.append(float(object_values.median()))
        restored_depths.append(depth_full)

    fp_z = data.obj.poses_cam[:, 2, 3].detach().cpu().float()
    vggt_obj_z = torch.tensor(object_depth_medians, dtype=torch.float32)
    ratios = fp_z / vggt_obj_z
    finite = torch.isfinite(ratios) & (ratios > 0)
    if int(finite.sum()) < max(3, int(0.8 * frame_num)):
        raise ValueError("Too few valid frames to calibrate the VGGT global depth scale")
    depth_scale = float(ratios[finite].median())
    if not np.isfinite(depth_scale) or not (0.1 <= depth_scale <= 10.0):
        raise ValueError(f"Invalid VGGT-to-GRAIL depth scale: {depth_scale}")

    scaled_depths = [(frame * depth_scale).to(device) for frame in restored_depths]
    provenance = {
        "source_cache_dir": os.path.abspath(cache_dir),
        "source_video": source_video,
        "metadata_path": metadata_path,
        "depth_path": depth_path,
        "confidence_path": confidence_path,
        "intrinsics_original_path": intrinsics_path,
        "backend": metadata["backend"],
        "model": metadata.get("model"),
        "checkpoint_path": metadata.get("checkpoint_path"),
        "num_frames": frame_num,
        "source_indices": list(metadata.get("source_indices", [])),
        "use_all_frames": True,
        "source_size_hw": [source_h, source_w],
        "processed_size_hw": list(depth.shape[1:]),
        "inverse_preprocess": "metadata crop/pad inversion then bilinear depth restoration",
        "depth_unprojection": metadata.get("depth_unprojection"),
        "coordinate_convention": metadata.get("coordinate_convention"),
        "confidence_percentile": confidence_percentile,
        "confidence_threshold_domain": "per-frame object mask",
        "confidence_threshold_median": float(np.median(confidence_thresholds)),
        "scale_method": "global median of FoundationPose object-center-z / VGGT object-mask median-z",
        "depth_scale": depth_scale,
        "raw_object_depth_median": float(np.median(object_depth_medians)),
        "target_foundationpose_z_median": float(fp_z.median()),
    }
    return scaled_depths, provenance


def _stage_metrics(optimizer, data, loss_cfg):
    pred = optimizer.forward(data, optimizer.params)
    total, losses = optimizer.loss_computer.compute_loss(data, pred, loss_cfg)
    return pred, float(total.detach()), losses


@torch.no_grad()
def _hand_object_distances(optimizer, data, pred, hand, top_k=32):
    labels = {"right": ["R_Hand"], "left": ["L_Hand"]}[hand]
    hand_seq = optimizer.human_model.get_verts_segment(pred.human.verts_seq, labels)
    values = []
    for frame in range(data.frame_num):
        values.append(
            hand_to_mesh_surface_distance(
                hand_seq[frame], pred.obj.verts_seq[frame], data.obj.faces, top_k=top_k
            )
        )
    return torch.stack(values)


@torch.no_grad()
def _human_mask_iou_diagnostics(optimizer, data, pred):
    """Cheap real-mask diagnostic using projected human-vertex bounding boxes."""
    projected = project_world_to_screen(
        pred.human.body_joints_seq.reshape(-1, 3), optimizer.cameras
    ).reshape(data.frame_num, -1, 3)[..., :2]
    values = []
    height = int(data.camera.frame_height)
    width = int(data.camera.frame_width)
    for frame in range(data.frame_num):
        points = projected[frame].detach().cpu().numpy()
        valid = np.isfinite(points).all(axis=1)
        if not valid.any():
            values.append(0.0)
            continue
        x0, y0 = np.floor(points[valid].min(axis=0)).astype(int)
        x1, y1 = np.ceil(points[valid].max(axis=0)).astype(int)
        estimated = np.zeros((height, width), dtype=bool)
        estimated[max(0, y0):min(height, y1 + 1), max(0, x0):min(width, x1 + 1)] = True
        gt = np.asarray(data.human.masks[frame]).astype(bool)
        gt = np.squeeze(gt)
        if gt.ndim != 2:
            raise ValueError(f"Human mask frame {frame} must reduce to [H,W], got {gt.shape}")
        if gt.shape != estimated.shape:
            gt = torch.from_numpy(gt.astype(np.float32))[None, None]
            gt = torch.nn.functional.interpolate(
                gt, size=estimated.shape, mode="nearest"
            )[0, 0].numpy().astype(bool)
        union = np.logical_or(estimated, gt).sum()
        values.append(float(np.logical_and(estimated, gt).sum() / max(union, 1)))
    return np.asarray(values, dtype=np.float32)


@torch.no_grad()
def _keypoint_rmse(predicted, target, confidence_threshold=0.2):
    confidence = target[..., 2]
    valid = torch.isfinite(target).all(dim=-1) & (confidence >= confidence_threshold)
    if not bool(valid.any()):
        raise ValueError("No reliable keypoints are available for RMSE diagnostics")
    squared_pixel_error = (predicted - target[..., :2]).square().sum(dim=-1)
    return float(torch.sqrt(squared_pixel_error[valid].mean()))


@torch.no_grad()
def _foot_sliding(optimizer, data, pred, threshold=0.5):
    probs = data.human.foot_contact_probs
    if probs is None:
        return float("nan")
    left_idx, right_idx = optimizer.human_model.get_foot_joint_indices()
    joints = pred.human.body_joints_seq
    left_velocity = torch.linalg.norm(joints[1:, left_idx] - joints[:-1, left_idx], dim=-1)
    right_velocity = torch.linalg.norm(joints[1:, right_idx] - joints[:-1, right_idx], dim=-1)
    left_contact = torch.max(probs[:, 0], probs[:, 1])
    right_contact = torch.max(probs[:, 2], probs[:, 3])
    left_weight = (torch.minimum(left_contact[:-1], left_contact[1:]) > threshold).float()
    right_weight = (torch.minimum(right_contact[:-1], right_contact[1:]) > threshold).float()
    denominator = (left_weight.sum() + right_weight.sum()).clamp_min(1.0)
    return float(
        ((left_velocity * left_weight).sum() + (right_velocity * right_weight).sum())
        / denominator
    )


@torch.no_grad()
def _hand_trajectory_arrays(optimizer, data, pred):
    hand = optimizer.human_model.get_palm_center_from_hand_joints(
        pred.human.hand_joints_seq, data.contact_hand
    )
    hand_cam = transform_world_to_camera(
        hand, optimizer.opencv_cam_R, optimizer.opencv_cam_t
    )
    fps = float(optimizer.video_fps)
    velocity = torch.zeros_like(hand_cam)
    velocity[1:] = (hand_cam[1:] - hand_cam[:-1]) * fps
    acceleration = torch.zeros_like(hand_cam)
    acceleration[2:] = (hand_cam[2:] - 2.0 * hand_cam[1:-1] + hand_cam[:-2]) * fps * fps
    jerk = torch.zeros_like(hand_cam)
    jerk[3:] = (
        hand_cam[3:] - 3.0 * hand_cam[2:-1] + 3.0 * hand_cam[1:-2] - hand_cam[:-3]
    ) * fps * fps * fps
    desired_world = (
        data.hand_ray_target_world.detach()
        if data.hand_ray_target_world is not None
        else torch.zeros_like(hand)
    )
    pose_norm = torch.linalg.norm(pred.human.pose_res, dim=-1).mean(dim=-1)
    return {
        "world": hand.detach().cpu().numpy(),
        "cam": hand_cam.detach().cpu().numpy(),
        "velocity": velocity.detach().cpu().numpy(),
        "acceleration": acceleration.detach().cpu().numpy(),
        "jerk": jerk.detach().cpu().numpy(),
        "desired_world": desired_world.detach().cpu().numpy(),
        "pose_residual_norm": pose_norm.detach().cpu().numpy(),
    }


@torch.no_grad()
def _palm_contact_arrays(optimizer, data, pred):
    """Compute semantic palm/finger diagnostics, separate from legacy hand metrics."""
    palm_idx = optimizer.human_model.get_palm_patch_indices(data.contact_hand)
    finger_idx = optimizer.human_model.get_finger_patch_indices(data.contact_hand)
    palm = pred.human.verts_seq[:, list(palm_idx)]
    finger = pred.human.verts_seq[:, list(finger_idx)]
    obj = pred.obj.verts_seq.detach()
    palm_dist = []
    finger_dist = []
    exact_surface = []
    max_penetration = []
    penetrating_fraction = []
    faces_np = data.obj.faces.detach().cpu().numpy()
    for i in range(data.frame_num):
        palm_dist.append(torch.cdist(palm[i], obj[i]).amin(dim=1))
        finger_dist.append(torch.cdist(finger[i], obj[i]).amin(dim=1))
        exact_surface.append(
            hand_to_mesh_surface_distance(
                palm[i], obj[i], data.obj.faces, top_k=64, candidate_faces=64
            )
        )
        mesh = trimesh.Trimesh(
            vertices=obj[i].detach().cpu().numpy(), faces=faces_np, process=False
        )
        try:
            # trimesh uses positive signed distance for points inside a watertight mesh.
            signed = trimesh.proximity.signed_distance(
                mesh, palm[i].detach().cpu().numpy()
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not compute signed palm-object penetration at frame {i}"
            ) from exc
        inside = np.maximum(np.asarray(signed, dtype=np.float64), 0.0)
        max_penetration.append(float(inside.max(initial=0.0)))
        penetrating_fraction.append(float(np.mean(inside > 0.0)))
    palm_dist = torch.stack(palm_dist)
    finger_dist = torch.stack(finger_dist)
    palm_center = optimizer.human_model.get_palm_center_from_hand_joints(
        pred.human.hand_joints_seq, data.contact_hand
    )
    palm_cam = transform_world_to_camera(
        palm_center, optimizer.opencv_cam_R, optimizer.opencv_cam_t
    )
    palm_px = project_world_to_screen(palm_center, optimizer.cameras)[:, :2]
    observed = data.observed_palm_pixels
    reproj = torch.linalg.norm(palm_px - observed, dim=-1) if observed is not None else torch.full(
        (data.frame_num,), float("nan"), device=palm_px.device
    )
    return {
        "center_cam": palm_cam.detach().cpu().numpy(),
        "actual_px": palm_px.detach().cpu().numpy(),
        "observed_px": None if observed is None else observed.detach().cpu().numpy(),
        "reprojection_px": reproj.detach().cpu().numpy(),
        "surface_mean": torch.stack(exact_surface).detach().cpu().numpy(),
        "surface_median": torch.stack(exact_surface).detach().cpu().numpy(),
        "palm_fraction_under_1cm": (palm_dist < 0.01).float().mean(dim=1).detach().cpu().numpy(),
        "finger_fraction_under_1cm": (finger_dist < 0.01).float().mean(dim=1).detach().cpu().numpy(),
        "maximum_penetration": np.asarray(max_penetration, dtype=np.float32),
        "penetrating_fraction": np.asarray(penetrating_fraction, dtype=np.float32),
    }


def _write_diagnostics(
    output_dir, data, optimizer, pred, initial_hand_distances=None, initial_pred=None
):
    prior = data.lift4d_depth
    if prior is None:
        raise ValueError("Formal diagnostics require the real Lift4D depth prior")
    frame_num = int(data.frame_num)
    supervised = int(prior.prior_used.sum())
    if supervised != frame_num:
        raise AssertionError(f"Lift4D supervised frames: {supervised} / {frame_num}")
    left_distance = _hand_object_distances(
        optimizer, data, pred, "left"
    ).detach().cpu().numpy()
    right_distance = _hand_object_distances(
        optimizer, data, pred, "right"
    ).detach().cpu().numpy()
    selected_distance = right_distance if data.contact_hand == "right" else left_distance
    if data.contact_hand == "both":
        selected_distance = np.minimum(left_distance, right_distance)
    human_mask_iou = _human_mask_iou_diagnostics(optimizer, data, pred)
    initial_human_mask_iou = (
        _human_mask_iou_diagnostics(optimizer, data, initial_pred)
        if initial_pred is not None
        else human_mask_iou.copy()
    )
    initial_body_rmse = _keypoint_rmse(
        initial_pred.human.body_keypoints_seq if initial_pred is not None else pred.human.body_keypoints_seq,
        data.human.body_keypoints_seq,
    )
    final_body_rmse = _keypoint_rmse(
        pred.human.body_keypoints_seq, data.human.body_keypoints_seq
    )
    initial_hand_rmse = _keypoint_rmse(
        initial_pred.human.hand_keypoints_seq if initial_pred is not None else pred.human.hand_keypoints_seq,
        data.human.hand_keypoints_seq,
    )
    final_hand_rmse = _keypoint_rmse(
        pred.human.hand_keypoints_seq, data.human.hand_keypoints_seq
    )
    frame = np.arange(frame_num)
    prior_idx = prior.frame_indices.detach().cpu().numpy()
    prior_used = prior.prior_used.detach().cpu().numpy().astype(bool)
    lift_raw = prior.z_raw.detach().cpu().numpy()
    lift_smooth = prior.z.detach().cpu().numpy()
    lift_target = prior.z_target.detach().cpu().numpy()
    center_raw = prior.center_cam_raw.detach().cpu().numpy()
    center_detection = prior.center_cam_detection.detach().cpu().numpy()
    state = data.object_motion_state
    if state is None:
        move_start = int(data.contact_hint)
        motion_score_3d = np.zeros(frame_num, dtype=np.float64)
        motion_score_mask = np.zeros(frame_num, dtype=np.float64)
        motion_score = np.zeros(frame_num, dtype=np.float64)
        moving = np.zeros(frame_num, dtype=bool)
        motion_confidence = float("nan")
    else:
        move_start = int(state.move_start_frame)
        motion_score_3d = state.motion_score_3d
        motion_score_mask = state.motion_score_mask
        motion_score = state.motion_score
        moving = state.moving
        motion_confidence = float(state.confidence)
    fp_z = data.obj.poses_cam[:, 2, 3].detach().cpu().numpy()
    optimized_z = pred.obj.z_cam.detach().cpu().numpy()
    depth_res = optimizer.params.obj_depth_res.detach().cpu().numpy()
    ramp = pred.human.approach_ramp.detach().cpu().numpy()
    offset = torch.linalg.norm(pred.human.approach_offset, dim=1).detach().cpu().numpy()
    trajectory = _hand_trajectory_arrays(optimizer, data, pred)
    palm_diag = _palm_contact_arrays(optimizer, data, pred)
    selected_distance = palm_diag["surface_median"]
    hand_cam = trajectory["cam"]
    hand_world = trajectory["world"]
    desired_world = trajectory["desired_world"]
    hand_speed = np.linalg.norm(trajectory["velocity"], axis=1)
    hand_acceleration = np.linalg.norm(trajectory["acceleration"], axis=1)
    hand_jerk = np.linalg.norm(trajectory["jerk"], axis=1)
    palm_center_cam = palm_diag["center_cam"]
    palm_speed = np.linalg.norm(np.vstack([np.zeros((1, 3)), np.diff(palm_center_cam, axis=0)]) * float(optimizer.video_fps), axis=1)
    palm_acceleration = np.linalg.norm(np.vstack([np.zeros((2, 3)), np.diff(palm_center_cam, n=2, axis=0)]) * float(optimizer.video_fps) ** 2, axis=1)
    palm_jerk = np.linalg.norm(np.vstack([np.zeros((3, 3)), np.diff(palm_center_cam, n=3, axis=0)]) * float(optimizer.video_fps) ** 3, axis=1)
    approach_start = max(0, move_start - int(data.approach_window))
    boundary_tail = int((optimizer.cfg.get("contact", {}) or {}).get("boundary_tail", 2))
    metric_start = max(0, approach_start - 1)
    metric_end = min(frame_num, move_start + boundary_tail + 1)
    selected_window = selected_distance[metric_start:metric_end]
    selected_window_steps = np.abs(np.diff(selected_window))
    boundary_step_tmove = float(np.linalg.norm(hand_cam[move_start] - hand_cam[move_start - 1]))
    boundary_velocity_change = float(
        np.linalg.norm(trajectory["velocity"][move_start] - trajectory["velocity"][move_start - 1])
    )

    trajectory_csv_path = os.path.join(output_dir, "hand_trajectory_diagnostics.csv")
    with open(trajectory_csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "frame", "actual_hand_cam_x", "actual_hand_cam_y", "actual_hand_cam_z",
            "actual_hand_world_x", "actual_hand_world_y", "actual_hand_world_z",
            "hand_speed_mps", "hand_acceleration_mps2", "hand_jerk",
            "hand_object_distance", "desired_palm_target_world_x", "desired_palm_target_world_y",
            "desired_palm_target_world_z", "pose_residual_norm", "approach_ramp",
            "boundary_step_tmove", "boundary_velocity_change_tmove", "ray_surface_fallback",
        ])
        fallback = (
            np.zeros(frame_num, dtype=bool)
            if data.hand_ray_surface_fallback is None
            else data.hand_ray_surface_fallback.detach().cpu().numpy().astype(bool)
        )
        for i in range(frame_num):
            writer.writerow([
                i, *hand_cam[i], *hand_world[i], hand_speed[i], hand_acceleration[i],
                hand_jerk[i], selected_distance[i], *desired_world[i],
                trajectory["pose_residual_norm"][i], ramp[i],
                boundary_step_tmove if i == move_start else "",
                boundary_velocity_change if i == move_start else "", int(fallback[i]),
            ])

    palm_csv_path = os.path.join(output_dir, "palm_contact_diagnostics.csv")
    target_cam = (
        np.full((frame_num, 3), np.nan, dtype=np.float32)
        if data.palm_target_cam is None
        else data.palm_target_cam.detach().cpu().numpy()
    )
    target_world = (
        np.full((frame_num, 3), np.nan, dtype=np.float32)
        if data.palm_target_world is None
        else data.palm_target_world.detach().cpu().numpy()
    )
    pixel_fallback = (
        np.zeros(frame_num, dtype=bool)
        if data.palm_pixel_fallback is None
        else data.palm_pixel_fallback.detach().cpu().numpy().astype(bool)
    )
    surface_fallback = (
        np.zeros(frame_num, dtype=bool)
        if data.palm_surface_fallback is None
        else data.palm_surface_fallback.detach().cpu().numpy().astype(bool)
    )
    with open(palm_csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "frame", "t_move", "contact_hand", "observed_palm_u", "observed_palm_v",
            "actual_palm_u", "actual_palm_v", "palm_reprojection_error_px",
            "palm_pixel_fallback", "actual_palm_cam_x", "actual_palm_cam_y", "actual_palm_cam_z",
            "target_palm_cam_x", "target_palm_cam_y", "target_palm_cam_z", "palm_depth_error_m",
            "palm_target_3d_error_m", "palm_surface_mean_distance_m", "palm_surface_median_distance_m",
            "palm_patch_fraction_under_1cm", "finger_patch_fraction_under_1cm",
            "maximum_penetration_m", "penetrating_vertex_fraction", "palm_speed_mps",
            "palm_acceleration_mps2", "palm_jerk", "approach_ramp", "human_approach_offset",
            "surface_pixel_fallback",
        ])
        observed_px = palm_diag["observed_px"]
        for i in range(frame_num):
            obs = (np.nan, np.nan) if observed_px is None else observed_px[i]
            writer.writerow([
                i, move_start, data.contact_hand, obs[0], obs[1],
                *palm_diag["actual_px"][i], palm_diag["reprojection_px"][i], int(pixel_fallback[i]),
                *palm_center_cam[i], *target_cam[i], palm_center_cam[i, 2] - target_cam[i, 2],
                float(np.linalg.norm(palm_center_cam[i] - target_cam[i])),
                palm_diag["surface_mean"][i], palm_diag["surface_median"][i],
                palm_diag["palm_fraction_under_1cm"][i], palm_diag["finger_fraction_under_1cm"][i],
                palm_diag["maximum_penetration"][i], palm_diag["penetrating_fraction"][i],
                palm_speed[i], palm_acceleration[i], palm_jerk[i], ramp[i], offset[i], int(surface_fallback[i]),
            ])

    palm_plot_path = os.path.join(output_dir, "palm_contact_diagnostics.png")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(frame, palm_diag["center_cam"][:, 2], label="actual palm Z")
    axes[0].plot(frame, target_cam[:, 2], label="target palm Z")
    axes[0].axvline(move_start, color="orange", linestyle="--", label="t_move")
    axes[0].set_ylabel("camera Z (m)")
    axes[0].legend()
    axes[1].plot(frame, palm_diag["surface_median"], label="palm surface median")
    axes[1].plot(frame, palm_diag["palm_fraction_under_1cm"], label="palm coverage <1cm")
    axes[1].plot(frame, palm_diag["finger_fraction_under_1cm"], label="finger coverage <1cm")
    axes[1].legend()
    axes[1].set_ylabel("distance / fraction")
    axes[2].plot(frame, palm_diag["reprojection_px"], label="palm reprojection error (px)")
    axes[2].plot(frame, palm_speed, label="palm speed (m/s)")
    axes[2].plot(frame, palm_acceleration, label="palm acceleration")
    axes[2].legend()
    axes[2].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(palm_plot_path, dpi=160)
    plt.close(fig)

    reproj_plot_path = os.path.join(output_dir, "palm_reprojection_diagnostics.png")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(frame, palm_diag["reprojection_px"], label="reprojection error (px)")
    ax.axhline(5.0, color="green", linestyle="--", label="median gate 5px")
    ax.axhline(10.0, color="red", linestyle=":", label="p95 gate 10px")
    ax.axvline(move_start, color="orange", linestyle="--", label="t_move")
    ax.set_xlabel("frame")
    ax.set_ylabel("pixels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reproj_plot_path, dpi=160)
    plt.close(fig)

    trajectory_plot_path = os.path.join(output_dir, "hand_trajectory_diagnostics.png")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(frame, hand_cam[:, 0], label="actual hand cam X")
    axes[0].plot(frame, hand_cam[:, 1], label="actual hand cam Y")
    axes[0].plot(frame, hand_cam[:, 2], label="actual hand cam Z")
    axes[0].axvline(move_start, color="orange", linestyle="--", label="t_move")
    axes[0].legend(ncol=4)
    axes[0].set_ylabel("camera position (m)")
    axes[1].plot(frame, hand_speed, label="speed (m/s)")
    axes[1].plot(frame, hand_acceleration, label="acceleration (m/s2)")
    axes[1].plot(frame, hand_jerk, label="jerk")
    axes[1].legend(ncol=3)
    axes[1].set_ylabel("trajectory derivatives")
    axes[2].plot(frame, selected_distance, label="hand-object distance")
    axes[2].plot(frame, ramp, label="minimum-jerk ramp")
    axes[2].axvline(move_start, color="orange", linestyle="--")
    axes[2].legend(ncol=2)
    axes[2].set_xlabel("frame")
    axes[2].set_ylabel("m / ramp")
    fig.tight_layout()
    fig.savefig(trajectory_plot_path, dpi=160)
    plt.close(fig)

    csv_path = os.path.join(output_dir, "lift4d_motion_diagnostics.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame", "prior_frame_idx", "prior_used",
                "center_cam_raw_x", "center_cam_raw_y", "center_cam_raw_z",
                "center_cam_detection_x", "center_cam_detection_y", "center_cam_detection_z",
                "lift4d_z_smooth", "lift4d_z_target", "motion_score_3d",
                "motion_score_mask", "motion_score", "mask_iou_drop",
                "centroid_displacement_px", "area_change_ratio", "lift4d_center_speed",
                "iou_threshold", "centroid_threshold_px", "area_threshold", "lift4d_speed_threshold",
                "moving", "move_start_frame",
                "motion_confidence", "static", "moving_evidence", "contact_hand",
                "hand_selection_reason", "foundationpose_z", "optimized_z", "obj_depth_res",
                "left_hand_object_distance", "right_hand_object_distance",
                "approach_ramp", "human_approach_offset", "contact_or_grasp_grad_obj_depth_res",
            ]
        )
        for i in range(frame_num):
            writer.writerow(
                [
                    i, int(prior_idx[i]), int(prior_used[i]), *center_raw[i],
                    *center_detection[i], lift_smooth[i], lift_target[i],
                    motion_score_3d[i], motion_score_mask[i], motion_score[i],
                    (0.0 if state is None else state.mask_iou_drop[i]),
                    (0.0 if state is None else state.mask_centroid_displacement_px[i]),
                    (0.0 if state is None else state.mask_area_change_ratio[i]),
                    (0.0 if state is None else state.lift4d_center_speed[i]),
                    (np.nan if state is None else state.thresholds["iou_drop"]),
                    (np.nan if state is None else state.thresholds["centroid_displacement_px"]),
                    (np.nan if state is None else state.thresholds["area_change_ratio"]),
                    (np.nan if state is None else state.thresholds["lift4d_center_speed_m"]),
                    int(moving[i]), move_start, motion_confidence,
                    int(not moving[i]), int(state is not None and state.moving_evidence[i]),
                    data.contact_hand, data.hand_selection_reason,
                    fp_z[i], optimized_z[i], depth_res[i], left_distance[i], right_distance[i],
                    ramp[i], offset[i], 0.0,
                ]
            )

    motion_state_csv_path = os.path.join(output_dir, "motion_state_diagnostics.csv")
    with open(motion_state_csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "frame", "mask_iou_drop", "centroid_displacement_px",
            "area_change_ratio", "lift4d_center_speed", "moving_evidence",
            "static", "moving", "move_start_frame",
        ])
        for i in range(frame_num):
            writer.writerow([
                i,
                0.0 if state is None else state.mask_iou_drop[i],
                0.0 if state is None else state.mask_centroid_displacement_px[i],
                0.0 if state is None else state.mask_area_change_ratio[i],
                0.0 if state is None else state.lift4d_center_speed[i],
                int(state is not None and state.moving_evidence[i]),
                int(not moving[i]), int(moving[i]), move_start,
            ])

    hand_csv_path = os.path.join(output_dir, "hand_ray_ik_diagnostics.csv")
    with open(hand_csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "t_move", "contact_hand", "left_mask_distance_px",
                         "right_mask_distance_px", "approach_ramp", "human_approach_offset",
                         "hand_initial_cam_depth", "hand_target_cam_depth", "object_surface_depth"])
        left_mask = data.hand_selection_left_distance_px
        right_mask = data.hand_selection_right_distance_px
        for i in range(frame_num):
            writer.writerow([
                i, move_start, data.contact_hand,
                "" if left_mask is None or not np.isfinite(left_mask[i]) else float(left_mask[i]),
                "" if right_mask is None or not np.isfinite(right_mask[i]) else float(right_mask[i]),
                ramp[i], offset[i],
                "" if data.hand_initial_cam_depth is None else float(data.hand_initial_cam_depth[i]),
                "" if data.hand_target_cam_depth is None else float(data.hand_target_cam_depth[i]),
                "" if data.object_surface_depth is None else float(data.object_surface_depth[i]),
            ])

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(frame, data.hand_selection_left_distance_px, label="left mask distance (px)")
    axes[0].plot(frame, data.hand_selection_right_distance_px, label="right mask distance (px)")
    axes[0].axvline(move_start, color="black", linestyle=":", label="t_move")
    axes[0].legend()
    axes[0].set_ylabel("distance (px)")
    if data.hand_initial_cam_depth is not None:
        axes[1].plot(frame, data.hand_initial_cam_depth.detach().cpu(), label="hand initial Z")
        axes[1].plot(frame, data.hand_target_cam_depth.detach().cpu(), label="hand target Z")
        axes[1].plot(frame, data.object_surface_depth.detach().cpu(), label="object surface Z")
    axes[1].plot(frame, ramp, label="approach ramp")
    axes[1].legend(ncol=3)
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("camera Z / ramp")
    fig.tight_layout()
    hand_plot = os.path.join(output_dir, "hand_ray_ik_diagnostics.png")
    fig.savefig(hand_plot, dpi=160)
    plt.close(fig)

    human_mask_csv_path = os.path.join(output_dir, "human_mask_iou_diagnostics.csv")
    with open(human_mask_csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "initial_human_mask_iou", "optimized_human_mask_iou", "delta"])
        writer.writerows(
            (i, float(initial_human_mask_iou[i]), float(human_mask_iou[i]),
             float(human_mask_iou[i] - initial_human_mask_iou[i]))
            for i in range(frame_num)
        )

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(frame, lift_raw, alpha=0.45, label="Lift4D raw Z")
    axes[0].plot(frame, lift_smooth, linewidth=2, label="Lift4D smooth Z")
    axes[0].plot(frame, lift_target, linewidth=2, label="static-relative Z target")
    axes[0].plot(frame, fp_z, label="FoundationPose Z")
    axes[0].plot(frame, optimized_z, linewidth=2, label="optimized Z")
    axes[0].legend(ncol=2)
    axes[0].set_ylabel("camera Z (m)")
    axes[1].plot(frame, depth_res, label="obj_depth_res")
    axes[1].plot(frame, np.gradient(optimized_z), label="optimized dZ")
    axes[1].legend()
    axes[1].set_ylabel("m")
    axes[2].plot(frame, motion_score, label="motion score")
    if state is not None:
        axes[2].plot(frame, state.mask_iou_drop / state.thresholds["iou_drop"], label="IoU drop / threshold")
        axes[2].plot(frame, state.mask_centroid_displacement_px / state.thresholds["centroid_displacement_px"], label="centroid / threshold")
        axes[2].plot(frame, state.mask_area_change_ratio / state.thresholds["area_change_ratio"], label="area / threshold")
        axes[2].plot(frame, state.lift4d_center_speed / state.thresholds["lift4d_center_speed_m"], label="Lift4D speed / threshold")
    axes[2].axvline(move_start, color="gray", linestyle=":", label="t_move (mask evidence)")
    axes[2].axvspan(data.contact_window_start, data.contact_window_end, alpha=0.15, color="green", label="contact window")
    axes[2].axvline(move_start, color="orange", linestyle="--", label="t_move")
    axes[2].legend()
    axes[2].set_xlabel("frame")
    fig.tight_layout()
    motion_plot = os.path.join(output_dir, "lift4d_motion_diagnostics.png")
    fig.savefig(motion_plot, dpi=160)
    motion_state_plot = os.path.join(output_dir, "motion_state_diagnostics.png")
    fig.savefig(motion_state_plot, dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(frame, left_distance, label="left hand-object distance")
    ax1.plot(frame, right_distance, label="right hand-object distance")
    ax1.axhline(0.02, color="green", linestyle="--", label="2 cm target")
    ax1.axhline(0.03, color="red", linestyle=":", label="3 cm check")
    ax1.axvline(data.contact_hint, color="gray", linestyle=":", label="contact hint")
    ax1.axvspan(data.contact_window_start, data.contact_window_end, alpha=0.15, color="green", label="contact window")
    ax1.axvline(move_start, color="orange", linestyle="--", label="t_move")
    ax1.set_xlabel("frame")
    ax1.set_ylabel("distance (m)")
    ax2 = ax1.twinx()
    ax2.plot(frame, ramp, color="gray", alpha=0.5, label="approach ramp")
    ax2.set_ylabel("ramp")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], ncol=3)
    fig.tight_layout()
    contact_plot = os.path.join(output_dir, "contact_distance_diagnostics.png")
    fig.savefig(contact_plot, dpi=160)
    plt.close(fig)

    steps = np.abs(np.diff(optimized_z))
    accelerations = np.abs(np.diff(optimized_z, n=2))
    destination_frames = np.arange(1, frame_num)
    late = destination_frames >= frame_num // 2
    late_steps = steps[late]
    jump_threshold = max(0.03, 4.0 * float(np.median(late_steps)))
    late_jump_frames = destination_frames[late][late_steps > jump_threshold]
    periodic_jump_count = max(
        (int(np.sum(late_jump_frames % 4 == phase)) for phase in range(4)), default=0
    )
    metrics = {
        "lift4d_supervised_frames": supervised,
        "frame_num": frame_num,
        "maximum_optimized_depth_step": float(steps.max(initial=0.0)),
        "maximum_optimized_depth_acceleration": float(accelerations.max(initial=0.0)),
        "maximum_late_optimized_depth_step": float(late_steps.max(initial=0.0)),
        "every_four_frame_periodic_jump_count": periodic_jump_count,
        "contact_hint": int(data.contact_hint),
        "contact_hint_source": data.contact_hint_source,
        "move_start_frame": move_start,
        "motion_confidence": motion_confidence,
        "contact_window_start": int(data.contact_window_start),
        "contact_window_end": int(data.contact_window_end),
        "contact_frame_hand_object_distance": float(selected_distance[move_start]),
        "moving_fraction_under_5cm": float(np.mean(selected_distance[move_start:] < 0.05)),
        "maximum_adjacent_hand_object_distance_change": float(
            selected_window_steps.max(initial=0.0)
        ),
        "diagnostic_distance_window_start": metric_start,
        "diagnostic_distance_window_end": metric_end - 1,
        "boundary_step_tmove": boundary_step_tmove,
        "boundary_velocity_change_tmove": boundary_velocity_change,
        "approach_max_hand_step": float(
            np.linalg.norm(
                np.diff(hand_cam[max(0, approach_start - 1) : move_start + 1], axis=0), axis=1
            ).max(initial=0.0)
        ),
        "maximum_hand_speed_mps": float(hand_speed.max(initial=0.0)),
        "maximum_hand_acceleration_mps2": float(hand_acceleration.max(initial=0.0)),
        "maximum_hand_jerk": float(hand_jerk.max(initial=0.0)),
        "static_raw_z_std": float(lift_raw[:move_start].std()),
        "static_target_z_std": float(lift_target[:move_start].std()),
        "optimized_static_z_std": float(optimized_z[:move_start].std()),
        "pre_optimization_hand_object_distance": float(
            initial_hand_distances[data.contact_hand][move_start]
            if initial_hand_distances is not None and data.contact_hand in initial_hand_distances
            else selected_distance[move_start]
        ),
        "post_optimization_hand_object_distance": float(selected_distance[move_start]),
        "contact_hand": data.contact_hand,
        "hand_selection_reason": data.hand_selection_reason,
        "contact_or_grasp_grad_obj_depth_res": 0.0,
        "human_approach_distance": float(pred.human.approach_distance.detach()),
        "foot_sliding": _foot_sliding(optimizer, data, pred),
        "human_mask_iou_mean": float(human_mask_iou.mean()),
        "human_mask_iou_min": float(human_mask_iou.min()),
        "initial_human_mask_iou_mean": float(initial_human_mask_iou.mean()),
        "human_mask_iou_mean_delta": float(
            human_mask_iou.mean() - initial_human_mask_iou.mean()
        ),
        "initial_body_keypoint_rmse_px": initial_body_rmse,
        "optimized_body_keypoint_rmse_px": final_body_rmse,
        "body_keypoint_rmse_increase_px": final_body_rmse - initial_body_rmse,
        "initial_hand_keypoint_rmse_px": initial_hand_rmse,
        "optimized_hand_keypoint_rmse_px": final_hand_rmse,
        "hand_keypoint_rmse_increase_px": final_hand_rmse - initial_hand_rmse,
        "median_palm_reprojection_error_px": float(np.nanmedian(palm_diag["reprojection_px"])),
        "p95_palm_reprojection_error_px": float(np.nanpercentile(palm_diag["reprojection_px"], 95)),
        "palm_depth_error_at_t_move_m": float(abs(palm_center_cam[move_start, 2] - target_cam[move_start, 2])),
        "palm_target_3d_error_at_t_move_m": float(np.linalg.norm(palm_center_cam[move_start] - target_cam[move_start])),
        "moving_median_palm_surface_distance_m": float(np.median(palm_diag["surface_median"][move_start:])),
        "moving_fraction_palm_surface_under_1p5cm": float(np.mean(palm_diag["surface_median"][move_start:] <= 0.015)),
        "moving_mean_palm_patch_fraction_under_1cm": float(np.mean(palm_diag["palm_fraction_under_1cm"][move_start:])),
        "maximum_penetration_m": float(np.max(palm_diag["maximum_penetration"])),
        "diagnostics_csv": csv_path,
        "motion_state_diagnostics_csv": motion_state_csv_path,
        "motion_plot": motion_plot,
        "motion_state_plot": motion_state_plot,
        "contact_plot": contact_plot,
        "human_mask_iou_csv": human_mask_csv_path,
        "hand_ray_plot": hand_plot,
        "hand_trajectory_csv": trajectory_csv_path,
        "hand_trajectory_plot": trajectory_plot_path,
        "palm_contact_csv": palm_csv_path,
        "palm_contact_plot": palm_plot_path,
        "palm_reprojection_plot": reproj_plot_path,
    }
    metrics["acceptance_gates"] = {
        "static_optimized_z_std_under_2mm": metrics["optimized_static_z_std"] < 0.002,
        "no_four_frame_periodic_jump": periodic_jump_count < 3,
        "contact_frame_under_3cm": metrics["contact_frame_hand_object_distance"] < 0.03,
        "moving_frames_under_5cm_at_least_80pct": metrics["moving_fraction_under_5cm"] >= 0.80,
        "adjacent_hand_object_change_under_5cm": (
            metrics["maximum_adjacent_hand_object_distance_change"] <= 0.05
        ),
        "boundary_hand_step_under_3cm": metrics["boundary_step_tmove"] < 0.03,
        "approach_hand_step_under_3cm": metrics["approach_max_hand_step"] < 0.03,
        "lift4d_all_frames": supervised == frame_num,
        "positive_opencv_z": bool(np.all(fp_z > 0) and np.all(optimized_z > 0)),
        "human_mask_iou_decrease_under_002": (
            metrics["human_mask_iou_mean_delta"] >= -0.02
        ),
        "body_keypoint_rmse_increase_under_5px": (
            metrics["body_keypoint_rmse_increase_px"] <= 5.0
        ),
        "hand_keypoint_rmse_increase_under_5px": (
            metrics["hand_keypoint_rmse_increase_px"] <= 5.0
        ),
        "palm_reprojection_median_under_5px": metrics["median_palm_reprojection_error_px"] <= 5.0,
        "palm_reprojection_p95_under_10px": metrics["p95_palm_reprojection_error_px"] <= 10.0,
        "palm_depth_at_move_under_1cm": metrics["palm_depth_error_at_t_move_m"] <= 0.01,
        "palm_3d_at_move_under_1p5cm": metrics["palm_target_3d_error_at_t_move_m"] <= 0.015,
        "moving_palm_surface_median_under_1cm": metrics["moving_median_palm_surface_distance_m"] <= 0.01,
        "moving_palm_surface_under_1p5cm_at_least_90pct": metrics["moving_fraction_palm_surface_under_1p5cm"] >= 0.90,
        "palm_patch_coverage_at_least_30pct": metrics["moving_mean_palm_patch_fraction_under_1cm"] >= 0.30,
        "maximum_penetration_under_3mm": metrics["maximum_penetration_m"] <= 0.003,
        "boundary_palm_step_under_1p6cm": metrics["boundary_step_tmove"] <= 0.016,
        "maximum_adjacent_palm_object_change_under_1p3cm": metrics["maximum_adjacent_hand_object_distance_change"] <= 0.013,
    }
    return metrics


def _build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--lift4d-prior", required=True)
    parser.add_argument("--vggt-cache", default=None)
    parser.add_argument("--use-vggt-human-depth", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage-a-niter", type=int, default=400)
    parser.add_argument("--stage-b-niter", type=int, default=600)
    parser.add_argument("--stage-c-niter", type=int, default=600)
    parser.add_argument("--contact-frame", type=int, default=None)
    parser.add_argument(
        "--contact-hand", choices=("auto", "left", "right", "both"), default="auto"
    )
    parser.add_argument("--confidence-percentile", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    return parser


def _build_stage_loss_configs(motion_state_enabled, use_vggt_human_depth):
    depth_weight = 50.0 if motion_state_enabled else 30.0
    velocity_weight = 10.0 if motion_state_enabled else 5.0
    fp_anchor_weight = 5.0 if motion_state_enabled else 10.0
    stage_a_loss = {
        "lift4d_depth": {"weight": depth_weight, "delta": 0.02},
        "lift4d_velocity": {"weight": velocity_weight, "delta": 0.02},
        "lift4d_acceleration": {"weight": 20.0},
        "fp_depth_anchor": {"weight": fp_anchor_weight, "delta": 0.02},
    }
    if motion_state_enabled:
        stage_a_loss["object_static_pre_motion"] = {"weight": 100.0, "delta": 0.01}
    if use_vggt_human_depth:
        stage_a_loss["depth_pointcloud"] = {
            "weight": 0.2,
            "trim_pct": 0.25,
            "interval": 1,
            "require_full_frame": True,
            "num_gt_samples": 1000,
            "include_human": True,
            "include_object": False,
        }

    contact_anchor_cfg = {
        "weight": 5000.0,
        "target_distance": 0.005,
        "delta": 0.005,
        "top_k": 32,
    }
    if motion_state_enabled:
        contact_anchor_cfg["continuous_frames"] = True
    else:
        contact_anchor_cfg["frame_radius"] = 2
    stage_b_loss = {
        "contact_anchor": {**contact_anchor_cfg, "phase": "precontact"},
        "hand_ray_ik": {"weight": 1000.0, "delta": 0.03, "phase": "precontact"},
        "hand_path": {"weight": 500.0, "delta": 0.03, "phase": "precontact"},
        "hand_velocity": {"weight": 500.0, "delta": 0.02, "phase": "precontact", "max_step": 0.025, "max_step_weight": 25.0, "max_step_reduction": "max"},
        "hand_acceleration": {"weight": 100.0, "delta": 0.02, "phase": "precontact"},
        "hand_jerk": {"weight": 20.0, "delta": 0.02, "phase": "precontact"},
        "pose_residual_acceleration": {"weight": 50.0, "phase": "precontact"},
        "approach_monotonic": {"weight": 500.0, "top_k": 32},
        "human_smoothness": {"weight": 300.0},
        "human_pose_reg": {"weight": 100.0},
        "human_foot_contact": {"weight": 1000.0},
        "body_keypoint_reprojection": {"weight": 1.0},
        "hand_keypoint_reprojection": {"weight": 0.3},
        "human_silhouette": {"weight": 0.5, "output_size": (64, 64), "silhouette_num_samples": 512},
        "palm_reprojection": {"weight": 5.0, "delta": 5.0, "phase": "precontact", "window_start": 0},
        # Use a normalized smooth terminal ramp instead of a single-frame pull.
        "palm_depth": {"weight": 30.0, "delta": 0.01, "terminal_weight": 2000.0, "terminal_window": 8, "terminal_loss": "squared", "terminal_frame": "contact", "phase": "precontact", "ramp_with_hand_ray": True},
        "palm_target_3d": {"weight": 5.0, "delta": 0.015, "terminal_weight": 500.0, "terminal_window": 8, "terminal_loss": "squared", "terminal_frame": "contact", "phase": "precontact", "ramp_with_hand_ray": True},
        "palm_surface": {"weight": 100.0, "target_distance": 0.005, "delta": 0.005, "terminal_weight": 10.0, "terminal_window": 8, "terminal_frame": "contact", "phase": "precontact", "ramp_with_hand_ray": True},
        "contact_coverage": {"weight": 50.0, "threshold": 0.01, "target_fraction": 0.30, "phase": "precontact"},
        "hand_object_penetration": {"weight": 50000.0, "minimum_clearance": 0.004, "signed_proxy": True, "worst_fraction": 0.10, "worst_weight": 5.0, "phase": "precontact"},
        "hand_roi_reprojection": {"weight": 1.0, "roi_radius_px": 32.0, "phase": "precontact"},
        "boundary_velocity": {"weight": 10000.0, "delta": 0.01, "target": "ray"},
    }
    stage_c_loss = {
        "lift4d_depth": {"weight": depth_weight, "delta": 0.02},
        "lift4d_velocity": {"weight": velocity_weight, "delta": 0.02},
        "lift4d_acceleration": {"weight": 20.0},
        "fp_depth_anchor": {"weight": fp_anchor_weight, "delta": 0.02},
        "contact_anchor": {**contact_anchor_cfg, "weight": 10000.0, "phase": "joint", "overlap_frames": 5},
        "hand_ray_ik": {"weight": 5000.0, "delta": 0.03, "phase": "joint", "overlap_frames": 5},
        "hand_path": {"weight": 1500.0, "delta": 0.03, "phase": "joint", "overlap_frames": 5},
        "hand_velocity": {"weight": 1000.0, "delta": 0.02, "phase": "joint", "overlap_frames": 5, "max_step": 0.025, "max_step_weight": 25.0, "max_step_reduction": "max"},
        "hand_acceleration": {"weight": 100.0, "delta": 0.02, "phase": "joint", "overlap_frames": 5},
        "hand_jerk": {"weight": 20.0, "delta": 0.02, "phase": "joint", "overlap_frames": 5},
        "pose_residual_acceleration": {"weight": 50.0, "phase": "joint", "overlap_frames": 5},
        "postcontact_relative": {"weight": 500.0, "delta": 0.01},
        "boundary_position": {"weight": 1000.0, "delta": 0.01},
        "boundary_velocity": {"weight": 500.0, "delta": 0.01},
        "pose_residual_continuity": {"weight": 100.0, "delta": 0.01},
        "human_smoothness": {"weight": 300.0},
        "human_pose_reg": {"weight": 100.0},
        "human_foot_contact": {"weight": 1000.0},
        "body_keypoint_reprojection": {"weight": 5.0},
        "hand_keypoint_reprojection": {"weight": 1.0},
        "human_silhouette": {"weight": 0.5, "output_size": (64, 64), "silhouette_num_samples": 512},
        "palm_reprojection": {"weight": 10.0, "delta": 5.0, "phase": "joint", "overlap_frames": 5},
        "palm_depth": {"weight": 100.0, "delta": 0.01, "terminal_weight": 5000.0, "terminal_frame": "contact", "phase": "joint", "overlap_frames": 5},
        "palm_target_3d": {"weight": 20.0, "delta": 0.015, "terminal_weight": 500.0, "terminal_frame": "contact", "phase": "joint", "overlap_frames": 5},
        "palm_surface": {"weight": 1000.0, "target_distance": 0.005, "delta": 0.005, "terminal_weight": 10.0, "terminal_frame": "contact", "phase": "joint", "overlap_frames": 5},
        "palm_normal": {"weight": 10.0, "phase": "joint", "overlap_frames": 5},
        "contact_coverage": {"weight": 300.0, "threshold": 0.01, "target_fraction": 0.30, "terminal_weight": 10.0, "terminal_frame": "contact", "phase": "joint", "overlap_frames": 5},
        "hand_object_penetration": {"weight": 100000.0, "minimum_clearance": 0.004, "signed_proxy": True, "worst_fraction": 0.10, "worst_weight": 5.0, "phase": "joint", "overlap_frames": 5},
        "hand_roi_reprojection": {"weight": 1.0, "roi_radius_px": 32.0, "phase": "joint", "overlap_frames": 5},
        "hand_pose_reg": {"weight": 100.0},
        "hand_pose_velocity": {"weight": 10.0},
        "hand_pose_acceleration": {"weight": 20.0},
    }
    if motion_state_enabled:
        stage_c_loss["object_static_pre_motion"] = {"weight": 100.0, "delta": 0.01}
    return stage_a_loss, stage_b_loss, stage_c_loss


def main() -> None:
    args = _build_parser().parse_args()
    if min(
        args.stage_a_niter,
        args.stage_b_niter,
        args.stage_c_niter,
    ) < 1:
        raise ValueError("All stage iteration counts must be >= 1")
    if not 0.0 <= args.confidence_percentile < 100.0:
        raise ValueError("--confidence-percentile must be in [0, 100)")

    config_file = _real_file(args.config_file, "GRAIL config")
    video_file = _real_file(args.video_file, "RGB video")
    hmr_file = _real_file(args.hmr_file, "HMR motion")
    mesh_file = _real_file(args.mesh_file, "object mesh")
    fp_file = _real_file(args.foundationpose_poses, "FoundationPose poses")
    render_config = _real_file(args.render_config, "FoundationPose render config")
    lift4d_prior = _real_file(args.lift4d_prior, "Lift4D motion-only NPZ")
    cache_dir = _real_dir(args.cache_dir, "GRAIL cache")
    if args.use_vggt_human_depth and args.vggt_cache is None:
        raise ValueError("--use-vggt-human-depth requires --vggt-cache")
    vggt_cache = (
        _real_dir(args.vggt_cache, "VGGT cache")
        if args.use_vggt_human_depth
        else None
    )

    with open(config_file, "r") as handle:
        root_cfg = yaml.safe_load(handle)
    cfg = dict(root_cfg["optimization"])
    cfg["human_model"] = dict(root_cfg["human_model"])
    project_root = Path(config_file).resolve().parents[2]
    for key, value in list(cfg["human_model"].items()):
        if (key.endswith("_path") or key.endswith("_dir")) and isinstance(value, str):
            if value and not os.path.isabs(value):
                cfg["human_model"][key] = str(project_root / value)
    motion_state_cfg = {
        "enabled": True,
        "detection_median_window": 5,
        "baseline_frames": 15,
        "vote_window": 5,
        "min_votes": 3,
        "threshold_mad_scale": 4.0,
        "centroid_displacement_floor_px": 2.0,
        "area_change_floor": 0.02,
        "iou_drop_floor": 0.03,
        "strong_iou_drop_floor": 0.20,
        "lift4d_speed_floor_m": 0.002,
        "required_consecutive_mask_frames": 3,
        "transition_frames": 0,
        "latch_moving": True,
        "low_confidence_action": "error",
    }
    motion_state_cfg.update(dict(cfg.get("object_motion_state", {}) or {}))
    contact_cfg = {
        "frame": None,
        "hand": "auto",
        "approach_window": None,
        "max_hand_speed_mps": 0.4,
        "min_approach_frames": 20,
        "max_approach_frames": 60,
        "max_approach_distance": 0.35,
        "max_root_approach_distance": 0.08,
        "boundary_tail": 2,
        "hand_selection_lookback": 5,
        "keypoint_confidence": 0.2,
        "both_distance_px": 12.0,
        "both_ratio": 1.25,
        "target_distance": 0.005,
    }
    contact_cfg.update(dict(cfg.get("contact", {}) or {}))
    contact_cfg["frame"] = args.contact_frame
    contact_cfg["hand"] = args.contact_hand
    contact_cfg["hint_source"] = "cli"
    cfg.update(
        {
            "results_dir": os.path.abspath(args.results_dir),
            "use_lift4d_depth_prior": True,
            "lift4d_motion_prior_path": lift4d_prior,
            "freeze_foundationpose_image_plane_translation": True,
            "lift4d_stable_point_count": 2500,
            "lift4d_median_window": 7,
            "lift4d_center_smooth_window": 31,
            "lift4d_savgol_polyorder": 2,
            "lift4d_depth_scale": 1.0,
            "learn_lift4d_depth_scale": False,
            "max_human_approach_distance": float(contact_cfg.get("max_approach_distance", 0.35)),
            "max_root_approach_distance": float(
                contact_cfg.get("max_root_approach_distance", 0.03)
            ),
            "object_motion_state": motion_state_cfg,
            "contact": contact_cfg,
            "vis_cfg": {"enable": False},
            "opt_stage_specs": {},
        }
    )

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    optimizer = HOIOptimizer(
        exp_name=args.video_id,
        cfg=cfg,
        cache_dir=cache_dir,
        output_dir=output_dir,
        device=args.device,
    )
    data = optimizer.init_data(video_file, hmr_file, mesh_file, fp_file, render_config)
    if args.use_vggt_human_depth:
        vggt_depths, vggt_provenance = _load_real_vggt_depth(
            vggt_cache,
            video_file,
            data,
            args.device,
            args.confidence_percentile,
        )
        data.depth_maps = vggt_depths
        optimizer.depth_list = vggt_depths
    else:
        vggt_provenance = {
            "enabled": False,
            "source_cache_dir": None,
            "consumed_by_loss": None,
        }
    optimizer.init_params(data)
    optimizer.initialize_obj_depth_from_lift4d(data)
    optimizer.loss_computer = LossComputer(
        cameras=optimizer.cameras,
        human_model=optimizer.human_model,
        device=optimizer.device,
        get_contact_labels_for_frame_fn=optimizer.get_contact_labels_for_frame,
        num_body_joints=optimizer.num_body_joints,
        logger=optimizer.logger,
    )
    initial_pred = optimizer.forward(data, optimizer.params)
    initial_hand_distances = {
        hand: _hand_object_distances(optimizer, data, initial_pred, hand)
        .detach()
        .cpu()
        .numpy()
        for hand in ("left", "right")
    }
    if data.contact_hand == "both":
        initial_hand_distances["both"] = np.minimum(
            initial_hand_distances["left"], initial_hand_distances["right"]
        )

    motion_state_enabled = bool(motion_state_cfg.get("enabled", False))
    stage_a_loss, stage_b_loss, stage_c_loss = _build_stage_loss_configs(
        motion_state_enabled, args.use_vggt_human_depth
    )
    stages = [
        {
            "stage": "stage_3a_lift4d_full_frame_object_depth",
            "opt_vars": {
                "obj_depth_res": {"lr": 0.005, "freeze_anchor": True}
            },
            "niter": args.stage_a_niter,
            "loss_cfg": stage_a_loss,
            "restore_best_state": True,
        },
        {
            "stage": "stage_3b_human_precontact_approach",
            "overlap_frames": 5,
            "opt_vars": {
                "human_approach_distance": {"lr": 0.00005},
                "human_pose_res": {
                    "lr": 0.001, "joint_scope": "arms", "frame_radius": 2,
                },
            },
            "niter": args.stage_b_niter,
            "loss_cfg": stage_b_loss,
            "restore_best_state": True,
        },
        {
            "stage": "stage_3c_joint_contact_refinement",
            "opt_vars": {
                # Post-contact residual translation must close the observed
                # frame-90+ depth jump within the fixed Stage-C window.
                "human_trans_res": {"lr": 0.008},
                "human_pose_res": {
                    "lr": 0.00005, "joint_scope": "arms", "frame_radius": 2,
                },
                "hand_pose_res": {"lr": 0.0001, "hand": "contact", "frame_radius": 5},
            },
            "niter": args.stage_c_niter,
            "loss_cfg": stage_c_loss,
            "restore_best_state": True,
        },
    ]
    stage_records = []
    stage_b_base_pose_residual = None
    for stage in stages:
        if stage["stage"] == "stage_3b_human_precontact_approach":
            optimizer.initialize_human_approach_direction(data, gravity_axis="z")
            # Start from the observed HMR trajectory.  The single scalar is
            # learned gradually in Stage B; directly seeding it from the
            # palm target can move the whole body out of image alignment.
            optimizer.capture_approach_target_boundary(data)
        elif stage["stage"] == "stage_3c_joint_contact_refinement":
            optimizer.initialize_postcontact_pose_residuals(
                data,
                stage["opt_vars"]["human_pose_res"]["joint_scope"],
                base_pose_residual=stage_b_base_pose_residual,
            )
        if stage["stage"] == "stage_3b_human_precontact_approach":
            stage_b_base_pose_residual = optimizer.params.human_pose_res.detach().clone()
        _, initial_total, initial_losses = _stage_metrics(
            optimizer, data, stage["loss_cfg"]
        )
        stage.update({
        "gradient_log_interval": 25,
        "save_motion_progress": True,
        "motion_progress_interval": 50,
        })
        optimizer.optimize_main(data, stage)
        if stage["stage"] == "stage_3a_lift4d_full_frame_object_depth":
            optimizer.refresh_hand_ray_targets_after_object_stage(data)
        elif stage["stage"] == "stage_3b_human_precontact_approach":
            optimizer.capture_stage_boundary_state(data)
        _, final_total, final_losses = _stage_metrics(
            optimizer, data, stage["loss_cfg"]
        )
        stage_records.append({
            "stage": stage["stage"],
            "niter": stage["niter"],
            "opt_vars": copy.deepcopy(stage["opt_vars"]),
            "loss_cfg": copy.deepcopy(stage["loss_cfg"]),
            "initial_total_loss": initial_total,
            "final_total_loss": final_total,
            "initial_losses": initial_losses,
            "final_losses": final_losses,
        })

    final_pred = optimizer.forward(data, optimizer.params)
    output = optimizer.get_optimized_data(data, final_pred, smooth=False)
    output["meta"]["vggt_depth"] = {
        **vggt_provenance,
        "consumed_by_loss": "human depth_pointcloud" if args.use_vggt_human_depth else None,
        "loss_config": dict(stage_a_loss.get("depth_pointcloud", {})),
    }
    output["meta"]["formal_joint_optimization"] = {
        "stages": stage_records,
        "lift4d_prior_path": lift4d_prior,
        "mesh_path": mesh_file,
        "synthetic_data_used": False,
    }
    output_path = os.path.join(output_dir, "hoi_data.pkl")
    metrics_path = os.path.join(output_dir, "optimization_metrics.json")
    # Keep the exact fixed-camera intrinsics used by the formal optimizer next
    # to the serialized result so renderers do not have to guess them.
    grail_camera_intrinsics = data.grail_camera_intrinsics.detach().cpu().numpy()
    lift4d_camera_intrinsics = data.lift4d_depth.camera_intrinsics.detach().cpu().numpy()
    np.save(os.path.join(output_dir, "grail_camera_intrinsics.npy"), grail_camera_intrinsics)
    np.save(os.path.join(output_dir, "lift4d_camera_intrinsics.npy"), lift4d_camera_intrinsics)
    diagnostics = _write_diagnostics(
        output_dir, data, optimizer, final_pred, initial_hand_distances, initial_pred
    )
    formal_result = bool(all(diagnostics.get("acceptance_gates", {}).values()))
    output["meta"]["formal_joint_optimization"]["formal_result"] = formal_result
    output["meta"]["formal_joint_optimization"]["failed_gates"] = [
        name for name, passed in diagnostics.get("acceptance_gates", {}).items()
        if not passed
    ]
    output["meta"]["diagnostics"] = diagnostics
    save_hoi_data(output, output_path)
    with open(metrics_path, "w") as handle:
        json.dump(
            {
                "stages": stage_records,
                "diagnostics": diagnostics,
                "vggt_depth": output["meta"]["vggt_depth"],
            },
            handle,
            indent=2,
        )
    if not formal_result:
        failed = output["meta"]["formal_joint_optimization"]["failed_gates"]
        raise RuntimeError(
            "Formal acceptance gates failed; output retained as debug only: "
            + ", ".join(failed)
        )
    print(f"Lift4D supervised frames: {diagnostics['lift4d_supervised_frames']} / {data.frame_num}")
    print(f"maximum optimized depth step={diagnostics['maximum_optimized_depth_step']:.9g}")
    print(f"maximum optimized depth acceleration={diagnostics['maximum_optimized_depth_acceleration']:.9g}")
    print(f"t_move={diagnostics['move_start_frame']}")
    print(f"motion confidence={diagnostics['motion_confidence']:.9g}")
    print(f"contact hand={diagnostics['contact_hand']} ({diagnostics['hand_selection_reason']})")
    print(f"contact-frame hand-object distance={diagnostics['contact_frame_hand_object_distance']:.9g}")
    print(f"moving frames under 5cm={diagnostics['moving_fraction_under_5cm']:.3%}")
    print(f"static raw z std={diagnostics['static_raw_z_std']:.9g}")
    print(f"static target z std={diagnostics['static_target_z_std']:.9g}")
    print(f"optimized static z std={diagnostics['optimized_static_z_std']:.9g}")
    print(f"human approach distance={diagnostics['human_approach_distance']:.9g}")
    print(f"foot sliding={diagnostics['foot_sliding']:.9g}")
    print(f"acceptance gates={json.dumps(diagnostics['acceptance_gates'], sort_keys=True)}")
    if args.use_vggt_human_depth:
        print(f"vggt_depth_scale={vggt_provenance['depth_scale']:.9g}")
    print(f"output={output_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
