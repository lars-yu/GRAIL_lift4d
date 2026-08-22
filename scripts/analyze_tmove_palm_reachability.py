#!/usr/bin/env python3
"""Bounded single-frame palm reachability analysis for a real GRAIL sequence.

This intentionally initializes the real scene but optimizes only residuals at the
automatically detected motion frame. It never runs the 121-frame optimization
stages or changes object pose/depth variables.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from dataclasses import fields

import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from grail.optimization.data_types import OptParams
from grail.optimization.hand_object_ray_ik import camera_ray_world_directions
from grail.optimization.hoi_optimizer import HOIOptimizer
from grail.rendering.camera import project_world_to_screen


def _load_cfg(config_file: str) -> dict:
    with open(config_file, "r") as handle:
        root = yaml.safe_load(handle)
    cfg = dict(root["optimization"])
    cfg["human_model"] = dict(root["human_model"])
    project_root = os.path.abspath(os.path.join(os.path.dirname(config_file), "../.."))
    for key, value in list(cfg["human_model"].items()):
        if (key.endswith("_path") or key.endswith("_dir")) and isinstance(value, str):
            if value and not os.path.isabs(value):
                cfg["human_model"][key] = os.path.join(project_root, value)
    return cfg


def _clone_params(params: OptParams) -> OptParams:
    values = {}
    for field in fields(params):
        value = getattr(params, field.name)
        if isinstance(value, torch.Tensor):
            values[field.name] = value.detach().clone().requires_grad_(True)
        else:
            values[field.name] = value
    return OptParams(**values)


def _palm_and_pixel(optimizer, data, pred, frame: int):
    palm = optimizer.human_model.get_palm_center_from_hand_joints(
        pred.human.hand_joints_seq, data.contact_hand
    )[frame]
    pixel = project_world_to_screen(palm[None], optimizer.cameras)[0, :2]
    return palm, pixel


def _run_mode(optimizer, data, base_params, frame, target, observed, mode, iterations):
    params = _clone_params(base_params)
    arm = optimizer._human_pose_joint_indices("arms", optimizer.num_body_joints)
    upper = optimizer._human_pose_joint_indices("upper_body_and_arms", optimizer.num_body_joints)
    allowed = upper if mode == "arm_shoulder" else arm
    optimize_root = mode == "arm_root_residual"
    optimizer.cfg["max_root_approach_distance"] = 0.05
    variables = [params.human_pose_res]
    if optimize_root:
        variables.append(params.human_approach_distance)
    local = torch.optim.Adam(variables, lr=0.01)
    identity = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=params.human_pose_res.dtype,
        device=params.human_pose_res.device,
    )
    for _ in range(iterations):
        local.zero_grad(set_to_none=True)
        pred = optimizer.forward(data, params)
        palm, pixel = _palm_and_pixel(optimizer, data, pred, frame)
        distance_error = torch.linalg.norm(palm - target)
        pixel_error = torch.linalg.norm(pixel - observed)
        pose_delta = params.human_pose_res[frame, allowed] - identity
        loss = 1.0e4 * distance_error.square() + pixel_error.square() + 1.0e-2 * pose_delta.square().mean()
        if optimize_root:
            loss = loss + 1.0e-2 * params.human_approach_distance.square()
        loss.backward()
        if params.human_pose_res.grad is not None:
            mask = torch.zeros_like(params.human_pose_res.grad)
            mask[frame, allowed] = 1.0
            params.human_pose_res.grad.mul_(mask)
        if optimize_root and params.human_approach_distance.grad is not None:
            params.human_approach_distance.grad.clamp_(-1.0, 1.0)
        local.step()
        with torch.no_grad():
            if optimize_root:
                params.human_approach_distance.clamp_(0.0, 0.05)
    with torch.no_grad():
        pred = optimizer.forward(data, params)
        palm, pixel = _palm_and_pixel(optimizer, data, pred, frame)
        distance_error = torch.linalg.norm(palm - target)
        pixel_error = torch.linalg.norm(pixel - observed)
        root_residual = float(params.human_approach_distance.detach()) if optimize_root else 0.0
    return {
        "mode": mode,
        "optimized_joints": allowed,
        "iterations": int(iterations),
        "palm_world": palm.detach().cpu().tolist(),
        "palm_target_world": target.detach().cpu().tolist(),
        "palm_distance_m": float(distance_error),
        "palm_reprojection_px": float(pixel_error),
        "root_residual_m": root_residual,
        "meets_5mm_and_5px": bool(float(distance_error) <= 0.005 and float(pixel_error) < 5.0),
    }


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
    parser.add_argument("--lift4d-prior", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--expected-t-move", type=int, default=89)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = _load_cfg(args.config_file)
    cfg.update({
        "results_dir": os.path.dirname(args.output_dir),
        "use_lift4d_depth_prior": True,
        "lift4d_motion_prior_path": os.path.abspath(args.lift4d_prior),
        "freeze_foundationpose_image_plane_translation": True,
        "lift4d_depth_scale": 1.0,
        "learn_lift4d_depth_scale": False,
        "max_root_approach_distance": 0.05,
        "object_motion_state": {"enabled": True, "low_confidence_action": "error"},
        "contact": {**dict(cfg.get("contact", {}) or {}), "frame": None, "hand": "auto"},
        "vis_cfg": {"enable": False},
    })
    optimizer = HOIOptimizer(args.video_id, cfg, args.cache_dir, args.output_dir, args.device)
    data = optimizer.init_data(
        args.video_file, args.hmr_file, args.mesh_file,
        args.foundationpose_poses, args.render_config,
    )
    frame = int(data.object_motion_state.move_start_frame)
    if frame != int(args.expected_t_move):
        raise RuntimeError(f"Automatic t_move={frame}, expected {args.expected_t_move}; refusing analysis")
    if data.contact_hand != "right":
        raise RuntimeError(f"Automatic contact hand={data.contact_hand!r}, expected 'right'")
    optimizer.init_params(data)
    optimizer.initialize_obj_depth_from_lift4d(data)
    optimizer.refresh_hand_ray_targets_after_object_stage(data)
    optimizer.initialize_human_approach_direction(data)
    with torch.no_grad():
        baseline_pred = optimizer.forward(data, optimizer.params)
        actual, actual_pixel = _palm_and_pixel(optimizer, data, baseline_pred, frame)
    target = data.palm_target_world[frame].detach()
    observed = data.observed_palm_pixels[frame].detach()
    delta = target - actual
    ray = camera_ray_world_directions(
        data.observed_palm_pixels[frame:frame + 1].detach(),
        data.grail_camera_intrinsics[frame],
        data.camera.pose[:3, :3],
    )[0]
    ground = optimizer._human_approach_direction.detach()
    ray_component = torch.dot(delta, ray) * ray
    ground_component = torch.dot(delta, ground) * ground
    report = {
        "sequence": args.video_id,
        "frame_num": int(data.frame_num),
        "t_move": frame,
        "contact_hand": data.contact_hand,
        "actual_palm_world": actual.cpu().tolist(),
        "target_palm_world": target.cpu().tolist(),
        "delta_world": delta.cpu().tolist(),
        "delta_norm_m": float(torch.linalg.norm(delta)),
        "observed_palm_pixel": observed.cpu().tolist(),
        "actual_palm_pixel": actual_pixel.cpu().tolist(),
        "baseline_reprojection_px": float(torch.linalg.norm(actual_pixel - observed)),
        "camera_ray_world_unit": ray.cpu().tolist(),
        "ground_direction_world_unit": ground.cpu().tolist(),
        "delta_camera_ray_component_world": ray_component.cpu().tolist(),
        "delta_ground_component_world": ground_component.cpu().tolist(),
        "delta_camera_ray_projection_m": float(torch.dot(delta, ray)),
        "delta_ground_projection_m": float(torch.dot(delta, ground)),
        "modes": [],
        "note": "Single-frame diagnostic only; no 121-frame optimization or formal gates were run.",
    }
    for mode in ("arm_only", "arm_shoulder", "arm_root_residual"):
        report["modes"].append(
            _run_mode(optimizer, data, optimizer.params, frame, target, observed, mode, args.iterations)
        )
    with open(os.path.join(args.output_dir, "tmove_palm_reachability.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    with open(os.path.join(args.output_dir, "tmove_palm_reachability.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mode", "palm_distance_m", "palm_reprojection_px", "root_residual_m", "meets_5mm_and_5px"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(report["modes"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
