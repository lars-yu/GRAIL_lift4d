#!/usr/bin/env python3
"""Render cached GENMO motion with the original GRAIL HOIVisualizer.

The cached GENMO motion is in OpenCV camera coordinates.  This adapter keeps
the v11 human first-frame alignment and the v10 object camera trajectory, then
maps both into the same GRAIL camera/world frame before rendering.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grail.core.io import load_human_motion_data, load_init_rendering_data, load_mesh
from grail.optimization.hoi_optimizer import HOIOptimizer
from grail.optimization.visualizer import HOIVisualizer
from grail.rendering.camera import cam_pose_blender_to_opencv


def _axis_angle_to_matrix(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(v)
    if theta < 1e-10:
        return np.eye(3, dtype=np.float32)
    axis = v / theta
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return (np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)).astype(np.float32)


def _matrix_to_axis_angle(R):
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis /= max(2 * np.sin(theta), 1e-8)
    return (axis * theta).astype(np.float32)


def _align_motion(source, reference):
    out = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in source.items()}
    src_pose, ref_pose = out["poses"], np.asarray(reference["poses"])
    src_trans, ref_trans = out["trans"], np.asarray(reference["trans"])
    A = _axis_angle_to_matrix(ref_pose[0, :3]) @ _axis_angle_to_matrix(src_pose[0, :3]).T
    b = ref_trans[0] - src_trans[0] @ A.T
    src_pose[:, :3] = np.asarray([_matrix_to_axis_angle(A @ _axis_angle_to_matrix(x)) for x in src_pose[:, :3]])
    src_trans[:] = src_trans @ A.T + b
    if src_pose.shape[1] >= 151:
        src_pose[:, 148:151] = src_pose[:, 148:151] @ A.T
    return out


def _camera_to_world_motion(motion, R, t, human_model=None):
    """Convert aligned camera motion using GRAIL's root-joint convention.

    ``trans`` in the GENMO/SMPL-X cache is the SMPL-X translation parameter,
    not necessarily the pelvis joint position.  GRAIL's optimizer therefore
    carries the pelvis offset through the camera transform.  Omitting that
    term produces a visibly displaced mesh even when the translation and root
    orientation appear aligned.
    """
    out = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in motion.items()}
    trans = np.asarray(out["trans"], dtype=np.float32)
    if human_model is not None:
        motion_t = {
            "poses": torch.as_tensor(out["poses"], device=human_model.device).float(),
            "betas": torch.as_tensor(out["betas"], device=human_model.device).float(),
            "trans": torch.as_tensor(trans, device=human_model.device).float(),
            "scale": float(out.get("scale", 1.0)),
        }
        for key in ("left_hand_pose", "right_hand_pose"):
            if out.get(key) is not None:
                motion_t[key] = torch.as_tensor(out[key], device=human_model.device).float()
        with torch.no_grad():
            _, _, joints = human_model.generate_mesh(
                motion_t, output_joints=True, require_grad=False
            )
        pelvis_offset = joints[:, 0, :].detach().cpu().numpy() - trans
    else:
        pelvis_offset = np.zeros_like(trans)
    out["trans"] = trans @ R.T + t + pelvis_offset @ R.T - pelvis_offset
    roots = np.asarray(out["poses"][:, :3], dtype=np.float32)
    out["poses"][:, :3] = np.asarray([_matrix_to_axis_angle(R @ _axis_angle_to_matrix(x)) for x in roots])
    if out["poses"].shape[1] >= 151:
        out["poses"][:, 148:151] = out["poses"][:, 148:151] @ R.T
    return out


def _camera_world_pose(render_config):
    loaded = load_init_rendering_data(render_config, to_tensor=True, device="cpu")
    _, _, _, blender_R, blender_t = loaded[:5]
    opencv_R, opencv_t = cam_pose_blender_to_opencv(blender_R, blender_t)
    return opencv_R.detach().cpu().numpy(), opencv_t.detach().cpu().numpy().reshape(3)


def _object_world_data(poses_cam, c2w_R, c2w_t, scale):
    poses_cam = np.asarray(poses_cam, dtype=np.float32)
    R_cam, t_cam = poses_cam[:, :3, :3], poses_cam[:, :3, 3]
    R_world = np.einsum("ij,fjk->fik", c2w_R, R_cam)
    t_world = t_cam @ c2w_R.T + c2w_t
    return {
        "obj_R": R_world.astype(np.float32),
        "obj_t": t_world.astype(np.float32),
        "obj_R_cam": R_cam.astype(np.float32),
        "obj_t_cam": t_cam.astype(np.float32),
        "obj_z_cam": t_cam[:, 2].astype(np.float32),
        "obj_scale": np.asarray(scale, dtype=np.float32),
    }


def _side_by_side(left_path, right_path, output_path, fps):
    readers = [imageio.get_reader(str(left_path)), imageio.get_reader(str(right_path))]
    writer = imageio.get_writer(str(output_path), fps=float(fps), codec="libx264", quality=8)
    try:
        frame_count = min(reader.count_frames() for reader in readers)
        for frame_idx in range(frame_count):
            left = readers[0].get_data(frame_idx)
            right = readers[1].get_data(frame_idx)
            if left.shape != right.shape:
                raise ValueError(
                    f"Comparison frame shape mismatch at {frame_idx}: "
                    f"{left.shape} vs {right.shape}"
                )
            writer.append_data(np.hstack((left, right)))
    finally:
        writer.close()
        for reader in readers:
            reader.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--reference-motion-dir", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--object-mesh", type=Path, required=True)
    p.add_argument("--render-config", type=Path, required=True)
    p.add_argument("--hmr-file", type=Path, required=True)
    p.add_argument("--config-file", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--human-body-model",
        choices=("smplx", "g1_smplx"),
        default="smplx",
        help="Human mesh backend. GENMO caches use the neutral SMPL-X convention.",
    )
    args = p.parse_args()

    source = args.source_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (source / "diagnostics.json").open() as f:
        diagnostics = json.load(f)
    initial = load_human_motion_data(str(source / "initial_genmo_motion.npz"), is_global=False)
    guided = load_human_motion_data(str(source / "selected_guided_motion.npz"), is_global=False)
    reference = load_human_motion_data(
        str(args.reference_motion_dir.resolve() / "initial_genmo_motion.npz"), is_global=False
    )
    initial = _align_motion(initial, reference)
    guided = _align_motion(guided, reference)

    with np.load(source / "anchored_lift4d_object_motion.npz", allow_pickle=True) as z:
        poses_cam = np.asarray(z["poses_in_cam"], dtype=np.float32)
    _, _, object_scale, _, _, _ = load_init_rendering_data(str(args.render_config), to_tensor=False)
    c2w_R, c2w_t = _camera_world_pose(str(args.render_config))
    with args.config_file.open() as f:
        root_cfg = yaml.safe_load(f)
    cfg = dict(root_cfg["optimization"])
    cfg["human_model"] = dict(root_cfg["human_model"])
    cfg["human_model"]["body_model"] = args.human_body_model
    cfg["results_dir"] = str(PROJECT_ROOT)
    cfg["skip_contact_label_loading"] = True
    cfg["object_motion_state"] = {"enabled": False}
    cfg["use_lift4d_depth_prior"] = False
    cfg["vis_cfg"] = {"enable": False}
    cfg["opt_stage_specs"] = {}
    for key, value in list(cfg["human_model"].items()):
        if isinstance(value, str) and (key.endswith("_path") or key.endswith("_dir")) and not os.path.isabs(value):
            cfg["human_model"][key] = str(PROJECT_ROOT / value)

    video_id = args.video.stem
    exp_name = f"dl300_delta/pickup_table/{video_id}"
    optimizer = HOIOptimizer(exp_name, cfg, str(args.cache_dir), str(out / "_setup"), args.device)
    camera, cameras, _, _, object_scale, static_objects = optimizer._load_camera_config(str(args.render_config))
    # Match HOIOptimizer._load_motion exactly, including the SMPL-X pelvis
    # offset term that is required for first-frame projection alignment.
    initial = _camera_to_world_motion(initial, c2w_R, c2w_t, optimizer.human_model)
    guided = _camera_to_world_motion(guided, c2w_R, c2w_t, optimizer.human_model)
    frame_dir = args.video.parent / "frames" / video_id
    image_list = sorted(str(x) for x in frame_dir.glob("*.jpg"))
    if not image_list:
        raise FileNotFoundError(f"Missing extracted frame cache: {frame_dir}")
    if isinstance(object_scale, torch.Tensor):
        object_scale = object_scale.detach().cpu().numpy()
    obj_verts, obj_faces, _ = load_mesh(str(args.object_mesh), mesh_scale=object_scale,
                                        target_num_verts=6000, device=args.device)
    data = SimpleNamespace(
        obj=SimpleNamespace(scale=object_scale),
        static_objects=static_objects,
        camera=camera,
        inter_start_idx=int(diagnostics.get("contact_frame", 0)),
        inter_end_idx=len(initial["poses"]),
    )
    obj_world = _object_world_data(poses_cam, c2w_R, c2w_t, object_scale)

    def hoi_for(motion):
        return {
            "human_data": motion,
            "obj_data": obj_world,
            "object_path": str(args.object_mesh),
            "scene_data": None,
        }

    fps = float(imageio.get_reader(str(args.video)).get_meta_data().get("fps", 30.0))
    visualizer = HOIVisualizer(args.device, optimizer.human_model, cameras,
                               image_list, fps, str(out / "_render_work"),
                               str(args.object_mesh))
    visualizer.init_vis_meshes(data)
    for name, motion in (("initial_genmo", initial), ("guided_genmo", guided)):
        visualizer.visualize(data, None, hoi_for(motion), name,
                             {"render_video": True, "extra_views": ["top"],
                              "export_mesh": False, "vis_html": False, "vis_contact": True})

    import shutil
    for name in ("initial_genmo", "guided_genmo"):
        shutil.copyfile(out / "_render_work" / name / f"{name}.mp4", out / f"{name}.mp4")
        shutil.copyfile(out / "_render_work" / name / f"{name}_top_view.mp4", out / f"{name}_top_view.mp4")
    _side_by_side(
        out / "initial_genmo.mp4",
        out / "guided_genmo.mp4",
        out / "initial_vs_guided.mp4",
        fps,
    )
    _side_by_side(
        out / "initial_genmo_top_view.mp4",
        out / "guided_genmo_top_view.mp4",
        out / "initial_vs_guided_top.mp4",
        fps,
    )
    diagnostics["hybrid_renderer"] = "original_grail_HOIVisualizer"
    diagnostics["hybrid_human_body_model"] = args.human_body_model
    diagnostics["hybrid_human_model_path"] = cfg["human_model"]["smplx_model_path"]
    diagnostics["hybrid_coordinate_policy"] = {
        "human": (
            "v11 first-frame rigid alignment, then GRAIL SMPL-X camera-to-world "
            "transform including per-frame pelvis offset"
        ),
        "object": "v10 anchored camera trajectory, then OpenCV camera-to-GRAIL world",
        "object_transform_not_aligned_with_human": True,
        "top_view": "original GRAIL HOIVisualizer top camera; no horizontal flip",
    }
    with (out / "diagnostics.json").open("w") as f:
        json.dump(diagnostics, f, indent=2, allow_nan=False)
    print(f"saved={out}")


if __name__ == "__main__":
    main()
