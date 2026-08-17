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


def _write_diagnostics(output_dir, data, optimizer, pred, initial_hand_distances=None):
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
    soft_weight = np.zeros(frame_num, dtype=np.float64)
    if data.contact_soft_weight is not None:
        start = int(data.contact_window_start)
        end = int(data.contact_window_end) + 1
        soft_weight[start:end] = data.contact_soft_weight.detach().cpu().numpy()
    selected_contact = int(
        data.selected_contact_frame
        if data.selected_contact_frame is not None
        else int(data.contact_window_start) + int(np.argmax(soft_weight[int(data.contact_window_start):int(data.contact_window_end) + 1]))
    )
    fp_z = data.obj.poses_cam[:, 2, 3].detach().cpu().numpy()
    optimized_z = pred.obj.z_cam.detach().cpu().numpy()
    depth_res = optimizer.params.obj_depth_res.detach().cpu().numpy()
    ramp = pred.human.approach_ramp.detach().cpu().numpy()
    offset = torch.linalg.norm(pred.human.approach_offset, dim=1).detach().cpu().numpy()

    csv_path = os.path.join(output_dir, "lift4d_motion_diagnostics.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame", "prior_frame_idx", "prior_used",
                "center_cam_raw_x", "center_cam_raw_y", "center_cam_raw_z",
                "center_cam_detection_x", "center_cam_detection_y", "center_cam_detection_z",
                "lift4d_z_smooth", "lift4d_z_target", "motion_score_3d",
                "motion_score_mask", "motion_score", "moving", "move_start_frame",
                "motion_confidence", "contact_hint", "contact_hint_source",
                "contact_window_start", "contact_window_end", "contact_soft_weight",
                "selected_contact_frame", "foundationpose_z", "optimized_z", "obj_depth_res",
                "left_hand_object_distance", "right_hand_object_distance",
                "approach_ramp", "human_approach_offset",
            ]
        )
        for i in range(frame_num):
            writer.writerow(
                [
                    i, int(prior_idx[i]), int(prior_used[i]), *center_raw[i],
                    *center_detection[i], lift_smooth[i], lift_target[i],
                    motion_score_3d[i], motion_score_mask[i], motion_score[i],
                    int(moving[i]), move_start, motion_confidence, int(data.contact_hint),
                    data.contact_hint_source, int(data.contact_window_start),
                    int(data.contact_window_end), soft_weight[i], selected_contact,
                    fp_z[i], optimized_z[i], depth_res[i], left_distance[i], right_distance[i],
                    ramp[i], offset[i],
                ]
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
    axes[2].axvline(data.contact_hint, color="gray", linestyle=":", label="contact hint")
    axes[2].axvspan(data.contact_window_start, data.contact_window_end, alpha=0.15, color="green", label="contact window")
    axes[2].axvline(move_start, color="orange", linestyle="--", label="t_move")
    axes[2].axvline(selected_contact, color="black", linestyle="--", label="selected contact")
    axes[2].legend()
    axes[2].set_xlabel("frame")
    fig.tight_layout()
    motion_plot = os.path.join(output_dir, "lift4d_motion_diagnostics.png")
    fig.savefig(motion_plot, dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(frame, left_distance, label="left hand-object distance")
    ax1.plot(frame, right_distance, label="right hand-object distance")
    ax1.axhline(0.02, color="green", linestyle="--", label="2 cm target")
    ax1.axhline(0.03, color="red", linestyle=":", label="3 cm check")
    ax1.axvline(data.contact_hint, color="gray", linestyle=":", label="contact hint")
    ax1.axvspan(data.contact_window_start, data.contact_window_end, alpha=0.15, color="green", label="contact window")
    ax1.axvline(move_start, color="orange", linestyle="--", label="t_move")
    ax1.axvline(selected_contact, color="black", linestyle="--", label="selected contact")
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
        "selected_contact_frame": selected_contact,
        "selected_contact_hand_object_distance": float(selected_distance[selected_contact]),
        "static_raw_z_std": float(lift_raw[:move_start].std()),
        "static_target_z_std": float(lift_target[:move_start].std()),
        "optimized_static_z_std": float(optimized_z[:move_start].std()),
        "pre_optimization_hand_object_distance": float(
            initial_hand_distances[data.contact_hand][selected_contact]
            if initial_hand_distances is not None and data.contact_hand in initial_hand_distances
            else selected_distance[selected_contact]
        ),
        "post_optimization_hand_object_distance": float(selected_distance[selected_contact]),
        "human_approach_distance": float(pred.human.approach_distance.detach()),
        "foot_sliding": _foot_sliding(optimizer, data, pred),
        "diagnostics_csv": csv_path,
        "motion_plot": motion_plot,
        "contact_plot": contact_plot,
    }
    if metrics["selected_contact_hand_object_distance"] >= 0.03:
        raise AssertionError(
            "Contact-frame hand-object surface distance must be below 3 cm; "
            f"got {metrics['selected_contact_hand_object_distance']:.6f} m"
        )
    if periodic_jump_count >= 3:
        raise AssertionError(
            "Detected repeated late single-frame depth jumps at the same frame%4 phase: "
            f"frames={late_jump_frames.tolist()}"
        )
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
    parser.add_argument("--stage-c-niter", type=int, default=250)
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
        "target_distance": 0.02,
        "delta": 0.02,
        "top_k": 32,
    }
    if motion_state_enabled:
        contact_anchor_cfg.update(
            {"hint_sigma": 5.0, "hint_floor": 0.2, "softmin_temperature": 0.01}
        )
    else:
        contact_anchor_cfg["frame_radius"] = 2
    stage_b_loss = {
        "contact_anchor": dict(contact_anchor_cfg),
        "approach_monotonic": {"weight": 500.0, "top_k": 32},
        "human_smoothness": {"weight": 300.0},
        "human_pose_reg": {"weight": 100.0},
        "human_foot_contact": {"weight": 1000.0},
        "keypoint_tracking": {"weight": 0.3},
    }
    stage_c_loss = {
        "lift4d_depth": {"weight": depth_weight, "delta": 0.02},
        "lift4d_velocity": {"weight": velocity_weight, "delta": 0.02},
        "lift4d_acceleration": {"weight": 20.0},
        "fp_depth_anchor": {"weight": fp_anchor_weight, "delta": 0.02},
        "contact_anchor": dict(contact_anchor_cfg),
        "postcontact_relative": {"weight": 500.0, "delta": 0.01},
        "human_smoothness": {"weight": 300.0},
        "human_pose_reg": {"weight": 100.0},
        "human_foot_contact": {"weight": 1000.0},
        "keypoint_tracking": {"weight": 0.3},
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
        "persistence_window": 7,
        "min_persistence": 5,
        "motion_score_threshold": 3.0,
        "transition_frames": 4,
        "latch_moving": True,
        "low_confidence_action": "error",
    }
    motion_state_cfg.update(dict(cfg.get("object_motion_state", {}) or {}))
    contact_cfg = {
        "frame": None,
        "hand": "auto",
        "hand_fallback": "right",
        "frames_before_motion": 8,
        "frames_after_motion": 2,
        "hint_sigma": 5.0,
        "hint_floor": 0.2,
        "softmin_temperature": 0.01,
        "approach_window": 30,
        "target_distance": 0.02,
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
            "max_human_approach_distance": 0.35,
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
        },
        {
            "stage": "stage_3b_human_precontact_approach",
            "opt_vars": {
                "human_approach_distance": {"lr": 0.003},
                "human_pose_res": {
                    "lr": 0.0001, "joint_scope": "lower_body_and_arms", "frame_radius": 2,
                },
            },
            "niter": args.stage_b_niter,
            "loss_cfg": stage_b_loss,
        },
        {
            "stage": "stage_3c_joint_contact_refinement",
            "opt_vars": {
                "obj_depth_res": {
                    "lr": 0.0002,
                    "freeze_anchor": True,
                    "max_delta": 0.02,
                },
                "human_approach_distance": {"lr": 0.0003},
                "human_pose_res": {
                    "lr": 0.00003, "joint_scope": "lower_body_and_arms", "frame_radius": 2,
                },
            },
            "niter": args.stage_c_niter,
            "loss_cfg": stage_c_loss,
        },
    ]
    stage_records = []
    for stage in stages:
        if stage["stage"] == "stage_3b_human_precontact_approach":
            optimizer.initialize_human_approach_direction(data, gravity_axis="z")
        _, initial_total, initial_losses = _stage_metrics(
            optimizer, data, stage["loss_cfg"]
        )
        stage.update({
        "gradient_log_interval": 25,
        "save_motion_progress": True,
        "motion_progress_interval": 50,
        })
        optimizer.optimize_main(data, stage)
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
    diagnostics = _write_diagnostics(
        output_dir, data, optimizer, final_pred, initial_hand_distances
    )
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
    print(f"Lift4D supervised frames: {diagnostics['lift4d_supervised_frames']} / {data.frame_num}")
    print(f"maximum optimized depth step={diagnostics['maximum_optimized_depth_step']:.9g}")
    print(f"maximum optimized depth acceleration={diagnostics['maximum_optimized_depth_acceleration']:.9g}")
    print(f"contact hint={diagnostics['contact_hint']} source={diagnostics['contact_hint_source']}")
    print(f"t_move={diagnostics['move_start_frame']}")
    print(f"motion confidence={diagnostics['motion_confidence']:.9g}")
    print(f"contact window=[{diagnostics['contact_window_start']},{diagnostics['contact_window_end']}]")
    print(f"selected contact frame={diagnostics['selected_contact_frame']}")
    print(f"selected-contact hand-object distance={diagnostics['selected_contact_hand_object_distance']:.9g}")
    print(f"static raw z std={diagnostics['static_raw_z_std']:.9g}")
    print(f"static target z std={diagnostics['static_target_z_std']:.9g}")
    print(f"optimized static z std={diagnostics['optimized_static_z_std']:.9g}")
    print(f"human approach distance={diagnostics['human_approach_distance']:.9g}")
    print(f"foot sliding={diagnostics['foot_sliding']:.9g}")
    if args.use_vggt_human_depth:
        print(f"vggt_depth_scale={vggt_provenance['depth_scale']:.9g}")
    print(f"output={output_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
