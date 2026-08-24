#!/usr/bin/env python3
"""Scan real local contact reachability before the full GRAIL optimization.

Every candidate frame/mode starts from the same initialized SMPL-X/object state.
Only human pose residuals and, for ray modes, one bounded camera-ray scalar are
optimized. Object pose/depth and all formal full-sequence stages remain frozen.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from dataclasses import fields

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from grail.optimization.approach import hand_to_mesh_surface_distance
from grail.optimization.data_types import OptParams
from grail.optimization.hand_object_ray_ik import camera_ray_world_directions
from grail.optimization.hoi_optimizer import HOIOptimizer
from grail.rendering.camera import project_world_to_screen
from scripts.run_lift4d_vggt_optimization import (
    _human_mask_iou_diagnostics,
    _keypoint_rmse,
)


MODE_ORDER = ["A_arms", "B_arms_shoulders", "C_arms_shoulders_ray_02",
              "D_arms_shoulders_ray_03", "E_torso_ray_03", "F_torso_ray_05"]


def _load_cfg(path):
    with open(path, "r") as handle:
        root = yaml.safe_load(handle)
    cfg = dict(root["optimization"])
    cfg["human_model"] = dict(root["human_model"])
    project_root = os.path.abspath(os.path.join(os.path.dirname(path), "../.."))
    for key, value in list(cfg["human_model"].items()):
        if (key.endswith("_path") or key.endswith("_dir")) and isinstance(value, str):
            if value and not os.path.isabs(value):
                cfg["human_model"][key] = os.path.join(project_root, value)
    return cfg


def _clone_params(params: OptParams) -> OptParams:
    values = {}
    for field in fields(params):
        value = getattr(params, field.name)
        values[field.name] = (
            value.detach().clone().requires_grad_(True)
            if isinstance(value, torch.Tensor) else value
        )
    return OptParams(**values)


def _mode_spec(optimizer, mode):
    arms = set(optimizer._human_pose_joint_indices("arms", optimizer.num_body_joints))
    shoulders = set(optimizer._human_pose_joint_indices("upper_body_and_arms", optimizer.num_body_joints))
    # G1-SMPL-X torso/spine indices are kept explicit; no free XYZ root is used.
    torso = shoulders | {1, 2, 4, 5, 7, 8, 10, 11}
    specs = {
        "A_arms": (arms, False, 0.0),
        "B_arms_shoulders": (shoulders, False, 0.0),
        "C_arms_shoulders_ray_02": (shoulders, True, 0.02),
        "D_arms_shoulders_ray_03": (shoulders, True, 0.03),
        "E_torso_ray_03": (torso, True, 0.03),
        "F_torso_ray_05": (torso, True, 0.05),
    }
    if mode not in specs:
        raise ValueError(f"unknown mode {mode!r}")
    joints, ray, limit = specs[mode]
    return sorted(joints), ray, limit


def _patch_metrics(optimizer, data, pred, frame):
    palm_idx = list(optimizer.human_model.get_palm_patch_indices(data.contact_hand))
    finger_idx = list(optimizer.human_model.get_finger_patch_indices(data.contact_hand))
    palm = pred.human.verts_seq[frame, palm_idx]
    finger = pred.human.verts_seq[frame, finger_idx]
    obj = pred.obj.verts_seq[frame].detach()
    palm_vertex_dist = torch.cdist(palm, obj).amin(dim=1)
    finger_vertex_dist = torch.cdist(finger, obj).amin(dim=1)
    faces = data.obj.faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=obj.cpu().numpy(), faces=faces, process=False)
    signed = np.asarray(trimesh.proximity.signed_distance(mesh, palm.detach().cpu().numpy()))
    penetration = np.maximum(signed, 0.0)
    center = optimizer.human_model.get_palm_center_from_hand_joints(
        pred.human.hand_joints_seq, data.contact_hand
    )[frame]
    pixel = project_world_to_screen(center[None], optimizer.cameras)[0, :2]
    observed = data.observed_palm_pixels[frame]
    return {
        "palm_center_target_error_m": float(torch.linalg.norm(center - data.palm_target_world[frame].detach())),
        "palm_patch_min_surface_distance_m": float(palm_vertex_dist.min()),
        "palm_patch_median_surface_distance_m": float(palm_vertex_dist.median()),
        "palm_patch_fraction_under_1cm": float((palm_vertex_dist <= 0.01).float().mean()),
        "finger_patch_fraction_under_1cm": float((finger_vertex_dist <= 0.01).float().mean()),
        "maximum_penetration_m": float(penetration.max(initial=0.0)),
        "penetrating_vertex_fraction": float(np.mean(penetration > 0.0)),
        "palm_reprojection_error_px": float(torch.linalg.norm(pixel - observed)),
        "palm_world": center.detach().cpu().tolist(),
        "root_residual_m": float(pred.human.approach_distance.detach()),
    }


def _run_mode(optimizer, data, base_params, initial_pred, frame, mode, iterations, ray):
    joints, use_ray, max_root = _mode_spec(optimizer, mode)
    params = _clone_params(base_params)
    optimizer.cfg["max_root_approach_distance"] = max_root if use_ray else 0.0
    variables = [params.human_pose_res]
    if use_ray:
        variables.append(params.human_approach_distance)
    local = torch.optim.Adam(variables, lr=0.01)
    identity = torch.tensor([1., 0., 0., 0., 1., 0.], dtype=params.human_pose_res.dtype, device=params.human_pose_res.device)
    for _ in range(int(iterations)):
        local.zero_grad(set_to_none=True)
        pred = optimizer.forward(data, params)
        center = optimizer.human_model.get_palm_center_from_hand_joints(pred.human.hand_joints_seq, data.contact_hand)[frame]
        pixel = project_world_to_screen(center[None], optimizer.cameras)[0, :2]
        target = data.palm_target_world[frame].detach()
        observed = data.observed_palm_pixels[frame].detach()
        center_loss = torch.linalg.norm(center - target).square()
        pixel_loss = torch.linalg.norm(pixel - observed).square()
        pose_delta = params.human_pose_res[frame, joints] - identity
        loss = 1.0e4 * center_loss + pixel_loss + 1.0e-2 * pose_delta.square().mean()
        if use_ray:
            loss = loss + 1.0e-2 * params.human_approach_distance.square()
        loss.backward()
        if params.human_pose_res.grad is not None:
            mask = torch.zeros_like(params.human_pose_res.grad)
            mask[frame, joints] = 1.0
            params.human_pose_res.grad.mul_(mask)
        if use_ray and params.human_approach_distance.grad is not None:
            params.human_approach_distance.grad.clamp_(-1.0, 1.0)
        local.step()
        with torch.no_grad():
            if use_ray:
                params.human_approach_distance.clamp_(0.0, max_root)
    with torch.no_grad():
        pred = optimizer.forward(data, params)
        metrics = _patch_metrics(optimizer, data, pred, frame)
        metrics.update({
            "candidate_frame": int(frame),
            "optimization_mode": mode,
            "optimized_joints": joints,
            "iterations": int(iterations),
            "ray_residual_enabled": bool(use_ray),
            "camera_ray_root_residual_limit_m": float(max_root),
        })
        initial_body = _keypoint_rmse(initial_pred.human.body_keypoints_seq, data.human.body_keypoints_seq)
        final_body = _keypoint_rmse(pred.human.body_keypoints_seq, data.human.body_keypoints_seq)
        initial_hand = _keypoint_rmse(initial_pred.human.hand_keypoints_seq, data.human.hand_keypoints_seq)
        final_hand = _keypoint_rmse(pred.human.hand_keypoints_seq, data.human.hand_keypoints_seq)
        initial_iou = _human_mask_iou_diagnostics(optimizer, data, initial_pred)
        final_iou = _human_mask_iou_diagnostics(optimizer, data, pred)
        metrics.update({
            "hand_keypoint_rmse_increase_px": float(final_hand - initial_hand),
            "body_keypoint_rmse_increase_px": float(final_body - initial_body),
            "human_mask_iou_decrease": float(np.mean(initial_iou - final_iou)),
            "pose_residual_magnitude": float(torch.linalg.norm(params.human_pose_res[frame, joints] - identity)),
        })
        metrics["physical_contact_feasible"] = bool(
            metrics["palm_patch_median_surface_distance_m"] <= 0.015
            and metrics["palm_patch_fraction_under_1cm"] >= 0.30
            and metrics["maximum_penetration_m"] <= 0.003
            and metrics["palm_reprojection_error_px"] <= 5.0
            and metrics["hand_keypoint_rmse_increase_px"] <= 5.0
            and metrics["body_keypoint_rmse_increase_px"] <= 5.0
            and metrics["human_mask_iou_decrease"] <= 0.03
        )
        metrics["pareto_score"] = (
            metrics["palm_patch_median_surface_distance_m"] / 0.015
            + max(0.0, 0.30 - metrics["palm_patch_fraction_under_1cm"]) / 0.30
            + metrics["maximum_penetration_m"] / 0.003
            + metrics["palm_reprojection_error_px"] / 5.0
            + metrics["hand_keypoint_rmse_increase_px"] / 5.0
            + metrics["body_keypoint_rmse_increase_px"] / 5.0
        )
    return metrics


def _select(results):
    priority = {mode: i for i, mode in enumerate(["B_arms_shoulders", "C_arms_shoulders_ray_02", "D_arms_shoulders_ray_03", "E_torso_ray_03", "F_torso_ray_05"])}
    feasible = [row for row in results if row["physical_contact_feasible"]]
    if feasible:
        return min(feasible, key=lambda row: (row["candidate_frame"], priority.get(row["optimization_mode"], 999)))
    return min(results, key=lambda row: (row["pareto_score"], row["candidate_frame"]))


def main():
    parser = argparse.ArgumentParser()
    for name in ("config_file", "video_id", "video_file", "hmr_file", "mesh_file", "foundationpose_poses", "render_config", "cache_dir", "lift4d_prior", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--t-move", type=int, default=89)
    args = parser.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = _load_cfg(args.config_file)
    cfg.update({
        "results_dir": os.path.dirname(args.output_dir),
        "use_lift4d_depth_prior": True,
        "lift4d_motion_prior_path": os.path.abspath(args.lift4d_prior),
        "freeze_foundationpose_image_plane_translation": True,
        "lift4d_depth_scale": 1.0,
        "learn_lift4d_depth_scale": False,
        "object_motion_state": {"enabled": True, "low_confidence_action": "error"},
        "contact": {**dict(cfg.get("contact", {}) or {}), "frame": None, "hand": "auto"},
        "vis_cfg": {"enable": False},
    })
    optimizer = HOIOptimizer(args.video_id, cfg, args.cache_dir, args.output_dir, args.device)
    data = optimizer.init_data(args.video_file, args.hmr_file, args.mesh_file, args.foundationpose_poses, args.render_config)
    move = int(data.object_motion_state.move_start_frame)
    if move != int(args.t_move):
        raise RuntimeError(f"automatic t_move={move}, expected {args.t_move}")
    if data.contact_hand != "right":
        raise RuntimeError(f"expected right contact hand, got {data.contact_hand}")
    optimizer.init_params(data)
    optimizer.initialize_obj_depth_from_lift4d(data)
    base_params = _clone_params(optimizer.params)
    results = []
    for frame in range(max(0, move - 8), move + 1):
        data.selected_contact_frame = frame
        optimizer.refresh_hand_ray_targets_after_object_stage(data)
        base_params = _clone_params(optimizer.params)
        optimizer.initialize_human_approach_direction(data)
        ground = optimizer._human_approach_direction.detach().clone()
        base_pred = optimizer.forward(data, base_params)
        for mode in MODE_ORDER:
            _, use_ray, _ = _mode_spec(optimizer, mode)
            if use_ray:
                ray = camera_ray_world_directions(data.observed_palm_pixels[frame:frame + 1], data.grail_camera_intrinsics[frame], data.camera.pose[:3, :3])[0]
                optimizer._human_approach_direction = ray.detach()
            else:
                optimizer._human_approach_direction = ground
            results.append(_run_mode(optimizer, data, base_params, base_pred, frame, mode, args.iterations, optimizer._human_approach_direction))
    selected = _select(results)
    report = {
        "sequence": args.video_id,
        "frame_num": int(data.frame_num),
        "t_move": move,
        "candidate_frames": [max(0, move - 8), move],
        "contact_hand": data.contact_hand,
        "local_contact_feasible": bool(any(row["physical_contact_feasible"] for row in results)),
        "selected_contact_frame": int(selected["candidate_frame"]),
        "selected_contact_mode": selected["optimization_mode"],
        "selected_contact_metrics": selected,
        "results": results,
        "note": "Real local scan only; no full 121-frame optimization or rendering was run by this scanner.",
    }
    with open(os.path.join(args.output_dir, "contact_reachability_scan.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    fields_out = sorted({key for row in results for key in row})
    with open(os.path.join(args.output_dir, "contact_reachability_scan.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_out, extrasaction="ignore")
        writer.writeheader(); writer.writerows(results)
    frames = sorted(set(row["candidate_frame"] for row in results))
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for mode in MODE_ORDER:
        rows = [row for row in results if row["optimization_mode"] == mode]
        axes[0].plot([row["candidate_frame"] for row in rows], [row["palm_patch_median_surface_distance_m"] for row in rows], marker=".", label=mode)
        axes[1].plot([row["candidate_frame"] for row in rows], [row["palm_patch_fraction_under_1cm"] for row in rows], marker=".", label=mode)
    axes[0].axhline(0.015, color="black", linestyle="--", label="median gate")
    axes[1].axhline(0.30, color="black", linestyle="--", label="coverage gate")
    axes[0].set_ylabel("palm patch median distance (m)")
    axes[1].set_ylabel("palm patch fraction <=1cm")
    axes[1].set_xlabel("candidate contact frame")
    axes[0].legend(fontsize=7, ncol=2); axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Local physical contact scan; selected={selected['candidate_frame']} {selected['optimization_mode']}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "contact_reachability_scan.png"), dpi=160)
    plt.close(fig)
    print(json.dumps({"selected_contact_frame": selected["candidate_frame"], "selected_contact_mode": selected["optimization_mode"], "local_contact_feasible": report["local_contact_feasible"]}, indent=2))


if __name__ == "__main__":
    main()
