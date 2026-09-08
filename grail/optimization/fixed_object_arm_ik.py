"""Post-GENMO fixed-object grasp + arm inverse kinematics.

After the GENMO contact guidance produces a good *approach* and a ~2 cm contact,
the object is picked up and moves.  The guidance only nudges the palm *point*, so
the hand *mesh* keeps clipping through the object during the hold/lift.

This stage replaces the loose post-contact follow with the "fixed-object grasp"
approach: at the contact frame the palm-centre contact is frozen in the object's
LOCAL frame (offset outward by the palm-shell thickness so the mesh rests ON the
surface, not through it) and transported rigidly by the known object SE(3)
trajectory.  A small per-frame IK then optimises the arm (collar/shoulder/elbow/
wrist) so the FK palm centre rides those transported targets, the palm normal
tracks the object surface, and the hand does not penetrate — while the fingers are
lightly fitted.  Only frames [contact_frame, end] are touched; the pre-contact
approach is left identical.

Works entirely in the contact-frame camera coordinate system (like
``finger_grasp.refine_finger_grasp``): the object poses are ``poses_in_cam`` and
the SMPL-X FK is evaluated on the in-camera motion.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from grail.models.smplx_model import generate_smplx_mesh, get_smplx_segment_indices
from grail.models.smplx_constants import SMPLX_BONE_ORDER_NAMES
from grail.optimization.finger_grasp import signed_point_mesh_distance, _object_mesh_cam
from grail.optimization.fixed_object_grasp import (
    build_fixed_object_grasp_targets,
    build_palm_center_contact_target,
)


_HAND_SEGMENT = {"left": ["L_Hand"], "right": ["R_Hand"]}
# Palm-centre joints (wrist + the four finger MCPs) and fingertips, by side.
_PALM_JOINT_NAMES = {
    "left": ["L_Wrist", "L_Index1", "L_Middle1", "L_Ring1", "L_Pinky1"],
    "right": ["R_Wrist", "R_Index1", "R_Middle1", "R_Ring1", "R_Pinky1"],
}
_FINGERTIP_JOINTS = {
    "left": [27, 30, 33, 36, 39],
    "right": [42, 45, 48, 51, 54],
}
# Body-pose axis-angle slices, derived by joint name (never hardcoded).
_ARM_JOINT_NAMES = {
    "left": {"collar": "L_Thorax", "shoulder": "L_Shoulder", "elbow": "L_Elbow", "wrist": "L_Wrist"},
    "right": {"collar": "R_Thorax", "shoulder": "R_Shoulder", "elbow": "R_Elbow", "wrist": "R_Wrist"},
}


def _body_pose_slice(joint_name: str) -> slice:
    """poses[3:66] axis-angle slice for a named body joint (1..21)."""
    j = SMPLX_BONE_ORDER_NAMES.index(joint_name)
    return slice(3 + (j - 1) * 3, 3 + j * 3)


def _joint_index(joint_name: str) -> int:
    return SMPLX_BONE_ORDER_NAMES.index(joint_name)


@dataclass(frozen=True)
class FixedObjectArmIKConfig:
    contact_frame: int
    selected_hand: str
    iterations: int = 200
    learning_rate: float = 0.02
    # Follow the transported object-local palm target (main driver) + palm normal.
    position_weight: float = 150.0
    normal_weight: float = 2.0
    # Keep the hand mesh out of the object (worst-vertex emphasised).
    penetration_weight: float = 800.0
    min_clearance: float = 0.0
    penetration_worst_weight: float = 8.0
    penetration_worst_fraction: float = 0.1
    # Light finger fit (fingertips rest ~5mm off the surface).
    finger_contact_weight: float = 4.0
    target_clearance: float = 0.005
    # Regularisation / smoothness / boundary continuity.  Kept light so the arm
    # can actually reach the transported target and pull the hand out.
    collar_reg_weight: float = 3.0
    shoulder_reg_weight: float = 1.0
    elbow_reg_weight: float = 1.0
    wrist_reg_weight: float = 0.5
    hand_reg_weight: float = 0.5
    smooth_weight: float = 3.0
    velocity_smooth_weight: float = 2.0
    boundary_weight: float = 4.0
    # Palm-shell clearance estimate (metres) for the target offset.
    palm_clearance_min: float = 0.018
    palm_clearance_max: float = 0.045
    palm_clearance_quantile: float = 0.85
    # Per-group clamps (radians) and misc.
    max_collar_change_rad: float = 0.30
    max_shoulder_change_rad: float = 0.80
    max_elbow_change_rad: float = 0.80
    max_wrist_change_rad: float = 0.80
    max_hand_change_rad: float = 1.2
    candidate_faces: int = 64
    max_hand_vertices: int = 160
    grad_clip_norm: float = 1.0


def refine_fixed_object_grasp_ik(
    motion_incam: Dict,
    object_vertices_local: torch.Tensor,
    object_faces: torch.Tensor,
    object_poses_cam: np.ndarray,
    contact_frame: int,
    selected_hand: str,
    smplx_model,
    config: FixedObjectArmIKConfig,
    device: str = "cuda",
) -> Tuple[Dict, Dict]:
    """Freeze the grasp in object-local coords and IK the arm to ride the object.

    Returns a new motion dict (same schema) and a diagnostics dict.  Does not
    mutate the input.  Only frames [contact_frame, end] change.
    """
    hand = str(selected_hand).lower()
    if hand not in _ARM_JOINT_NAMES:
        raise ValueError(f"selected_hand must be left/right, got {selected_hand!r}")
    names = _ARM_JOINT_NAMES[hand]
    sl_collar = _body_pose_slice(names["collar"])
    sl_shoulder = _body_pose_slice(names["shoulder"])
    sl_elbow = _body_pose_slice(names["elbow"])
    sl_wrist = _body_pose_slice(names["wrist"])
    hand_key = "left_hand_pose" if hand == "left" else "right_hand_pose"
    other_key = "right_hand_pose" if hand == "left" else "left_hand_pose"
    hand_offset = 66 if hand == "left" else 111

    poses0 = torch.as_tensor(np.asarray(motion_incam["poses"]), device=device, dtype=torch.float32)
    betas = torch.as_tensor(np.asarray(motion_incam["betas"]), device=device, dtype=torch.float32)
    trans = torch.as_tensor(np.asarray(motion_incam["trans"]), device=device, dtype=torch.float32)
    scale = float(motion_incam.get("scale", 1.0))
    frame_num = poses0.shape[0]
    hand0 = torch.as_tensor(
        np.asarray(motion_incam[hand_key]), device=device, dtype=torch.float32
    ).reshape(frame_num, 45)
    other_hand = motion_incam.get(other_key)
    other_hand = (
        torch.as_tensor(np.asarray(other_hand), device=device, dtype=torch.float32).reshape(frame_num, 45)
        if other_hand is not None else torch.zeros(frame_num, 45, device=device)
    )

    frame = int(contact_frame)
    sel = slice(frame, frame_num)
    n_sel = frame_num - frame
    if n_sel <= 1:
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in motion_incam.items()}, {"frames": int(max(0, n_sel))}

    obj_verts = torch.as_tensor(np.asarray(object_vertices_local), device=device, dtype=torch.float32)
    obj_faces = torch.as_tensor(np.asarray(object_faces), device=device, dtype=torch.long)
    poses_cam = torch.as_tensor(np.asarray(object_poses_cam), device=device, dtype=torch.float32)
    obj_R = poses_cam[:, :3, :3].contiguous()      # [T,3,3]
    obj_t = poses_cam[:, :3, 3].contiguous()       # [T,3]

    hand_vert_idx = torch.as_tensor(
        get_smplx_segment_indices(_HAND_SEGMENT[hand]), device=device, dtype=torch.long
    )
    if hand_vert_idx.numel() > config.max_hand_vertices:
        pick = torch.linspace(0, hand_vert_idx.numel() - 1, config.max_hand_vertices, device=device).long()
        hand_vert_idx = hand_vert_idx[pick]
    fingertip_idx = torch.as_tensor(_FINGERTIP_JOINTS[hand], device=device, dtype=torch.long)
    palm_idx = torch.as_tensor(
        [_joint_index(n) for n in _PALM_JOINT_NAMES[hand]], device=device, dtype=torch.long
    )
    wrist_j, index_j, _mid_j, _ring_j, pinky_j = palm_idx.tolist()

    # ---- residual variables (frames [contact_frame, end]) ----
    collar_res = torch.zeros(n_sel, 3, device=device, requires_grad=True)
    shoulder_res = torch.zeros(n_sel, 3, device=device, requires_grad=True)
    elbow_res = torch.zeros(n_sel, 3, device=device, requires_grad=True)
    wrist_res = torch.zeros(n_sel, 3, device=device, requires_grad=True)
    hand_res = torch.zeros(n_sel, 45, device=device, requires_grad=True)
    variables = [collar_res, shoulder_res, elbow_res, wrist_res, hand_res]
    optim = torch.optim.Adam(variables, lr=config.learning_rate)

    def decode():
        poses = poses0[sel].clone()
        poses[:, sl_collar] = poses[:, sl_collar] + collar_res
        poses[:, sl_shoulder] = poses[:, sl_shoulder] + shoulder_res
        poses[:, sl_elbow] = poses[:, sl_elbow] + elbow_res
        poses[:, sl_wrist] = poses[:, sl_wrist] + wrist_res
        this_hand = hand0[sel] + hand_res
        md = {
            "poses": poses,
            "betas": betas,
            "trans": trans[sel],
            "scale": scale,
            hand_key: this_hand,
            other_key: other_hand[sel],
        }
        verts, _faces, joints = generate_smplx_mesh(
            smplx_model, md, output_joints=True, require_grad=True, device=device
        )
        return verts, joints

    def palm_center_and_normal(joints):
        palm = joints[:, palm_idx, :].mean(dim=1)                  # [n_sel,3]
        wrist = joints[:, wrist_j, :]
        idx1 = joints[:, index_j, :]
        pky1 = joints[:, pinky_j, :]
        normal = torch.cross(idx1 - wrist, pky1 - wrist, dim=-1)
        normal = torch.nn.functional.normalize(normal, dim=-1, eps=1e-8)
        return palm, normal

    # ---- Build the fixed-object palm target from the INITIAL guided pose ----
    with torch.no_grad():
        v0, j0 = decode()
        palm0, _n0 = palm_center_and_normal(j0)
        palm_center_contact = palm0[0]                             # cam, contact frame
        palm_patch = v0[0, hand_vert_idx]                          # hand verts at contact
        obj_cam_c = _object_mesh_cam(obj_verts, poses_cam[frame])
        # nearest object surface point + outward normal at the contact frame
        d = torch.linalg.vector_norm(obj_cam_c - palm_center_contact[None], dim=-1)
        surface_pt = obj_cam_c[int(d.argmin())]
        outward = torch.nn.functional.normalize(
            surface_pt - obj_cam_c.mean(dim=0), dim=0, eps=1e-8
        )
        palm_target = build_palm_center_contact_target(
            surface_pt, outward, palm_center_contact, palm_patch,
            clearance_quantile=config.palm_clearance_quantile,
            minimum_clearance=config.palm_clearance_min,
            maximum_clearance=config.palm_clearance_max,
        )
        grasp = build_fixed_object_grasp_targets(
            palm_target.center_position_world, obj_R, obj_t, frame,
            contact_normal_world=palm_target.surface_to_hand_world,
        )
        target_pos = grasp.position_world[sel].detach()            # [n_sel,3]
        target_normal = grasp.normal_world[sel].detach()           # [n_sel,3]

    def losses():
        verts, joints = decode()
        palm, palm_normal = palm_center_and_normal(joints)
        hand_verts = verts[:, hand_vert_idx]
        fingertips = joints[:, fingertip_idx]
        # position + normal follow
        position = (palm - target_pos).square().sum(dim=-1).mean()
        dot = (palm_normal * target_normal).sum(dim=-1)
        normal_loss = (1.0 - dot.abs()).square().mean()
        # penetration + finger contact (per frame, moving object mesh)
        pen_terms, contact_terms, min_signed = [], [], []
        for i in range(n_sel):
            obj_cam = _object_mesh_cam(obj_verts, poses_cam[frame + i])
            sd_hand = signed_point_mesh_distance(
                hand_verts[i], obj_cam, obj_faces, candidate_faces=config.candidate_faces
            )
            sd_tip = signed_point_mesh_distance(
                fingertips[i], obj_cam, obj_faces, candidate_faces=config.candidate_faces
            )
            pen = torch.relu(config.min_clearance - sd_hand).square()
            k = max(1, int(round(config.penetration_worst_fraction * pen.numel())))
            worst = torch.topk(pen, k=min(k, pen.numel())).values.mean()
            pen_terms.append(pen.mean() + config.penetration_worst_weight * worst)
            contact_terms.append((sd_tip - config.target_clearance).square().mean())
            min_signed.append(sd_hand.min())
        penetration = torch.stack(pen_terms).mean()
        finger_contact = torch.stack(contact_terms).mean()
        # regularisation + smoothness + boundary continuity
        reg = (
            config.collar_reg_weight * collar_res.square().mean()
            + config.shoulder_reg_weight * shoulder_res.square().mean()
            + config.elbow_reg_weight * elbow_res.square().mean()
            + config.wrist_reg_weight * wrist_res.square().mean()
            + config.hand_reg_weight * hand_res.square().mean()
        )
        smooth = palm.new_zeros(())
        if n_sel > 2:
            for r in (collar_res, shoulder_res, elbow_res, wrist_res):
                smooth = smooth + (r[2:] - 2 * r[1:-1] + r[:-2]).square().mean()
        velocity = palm.new_zeros(())
        if n_sel > 2:
            velocity = (palm[2:] - 2 * palm[1:-1] + palm[:-2]).square().sum(dim=-1).mean()
        boundary = (
            collar_res[0].square().mean() + shoulder_res[0].square().mean()
            + elbow_res[0].square().mean() + wrist_res[0].square().mean()
        )
        total = (
            config.position_weight * position
            + config.normal_weight * normal_loss
            + config.penetration_weight * penetration
            + config.finger_contact_weight * finger_contact
            + reg
            + config.smooth_weight * smooth
            + config.velocity_smooth_weight * velocity
            + config.boundary_weight * boundary
        )
        return total, position, penetration, torch.stack(min_signed), dot

    with torch.no_grad():
        _, pos0, pen0, ms0, dot0 = losses()
        before = {
            "follow_error_m": float(pos0.sqrt().detach().cpu()),
            "penetration_loss": float(pen0.detach().cpu()),
            "min_signed_distance_m": float(ms0.min().detach().cpu()),
            "normal_alignment": float(dot0.abs().mean().detach().cpu()),
        }

    for _ in range(int(config.iterations)):
        optim.zero_grad(set_to_none=True)
        total, *_ = losses()
        if not torch.isfinite(total):
            raise FloatingPointError("fixed-object arm-IK loss contains NaN/Inf")
        total.backward()
        torch.nn.utils.clip_grad_norm_(variables, config.grad_clip_norm)
        optim.step()
        with torch.no_grad():
            collar_res.clamp_(-config.max_collar_change_rad, config.max_collar_change_rad)
            shoulder_res.clamp_(-config.max_shoulder_change_rad, config.max_shoulder_change_rad)
            elbow_res.clamp_(-config.max_elbow_change_rad, config.max_elbow_change_rad)
            wrist_res.clamp_(-config.max_wrist_change_rad, config.max_wrist_change_rad)
            hand_res.clamp_(-config.max_hand_change_rad, config.max_hand_change_rad)

    with torch.no_grad():
        _, pos1, pen1, ms1, dot1 = losses()
        per_frame_err = torch.linalg.vector_norm(
            palm_center_and_normal(decode()[1])[0] - target_pos, dim=-1
        )

    # ---- write back (frames >= contact_frame) ----
    out = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in motion_incam.items()}
    poses_out = np.array(motion_incam["poses"], copy=True)
    poses_out[frame:, sl_collar] += collar_res.detach().cpu().numpy()
    poses_out[frame:, sl_shoulder] += shoulder_res.detach().cpu().numpy()
    poses_out[frame:, sl_elbow] += elbow_res.detach().cpu().numpy()
    poses_out[frame:, sl_wrist] += wrist_res.detach().cpu().numpy()
    hand_out = np.array(motion_incam[hand_key], copy=True).reshape(frame_num, 45)
    hand_out[frame:] += hand_res.detach().cpu().numpy()
    poses_out[:, hand_offset:hand_offset + 45] = hand_out
    out["poses"] = poses_out
    out[hand_key] = hand_out

    diagnostics = {
        "frames": int(n_sel),
        "hand": hand,
        "follow_error_before_m": before["follow_error_m"],
        "follow_error_after_m": float(pos1.sqrt().detach().cpu()),
        "follow_error_mean_m": float(per_frame_err.mean().detach().cpu()),
        "follow_error_max_m": float(per_frame_err.max().detach().cpu()),
        "penetration_before": before["penetration_loss"],
        "penetration_after": float(pen1.detach().cpu()),
        "min_signed_distance_before_m": before["min_signed_distance_m"],
        "min_signed_distance_after_m": float(ms1.min().detach().cpu()),
        "normal_alignment_before": before["normal_alignment"],
        "normal_alignment_after": float(dot1.abs().mean().detach().cpu()),
        "palm_shell_clearance_m": float(palm_target.center_clearance.detach().cpu()),
    }
    return out, diagnostics
