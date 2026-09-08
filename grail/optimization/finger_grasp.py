"""Post-GENMO finger-grasp refinement.

The GENMO contact guidance only positions the *arm* (it moves the palm point
near the object); the finger pose is copied verbatim from the HMR source motion,
so the fingers do not wrap the object and the flat hand mesh clips through it.

This module runs a small per-frame optimization *after* guidance: it optimizes
the grasping hand's finger pose (plus wrist orientation and a little elbow) so
that the fingertips contact the object surface while no hand vertex penetrates
the object.  It uses the differentiable SMPL-X forward kinematics and an
analytic signed point-to-triangle distance (the same math as
``loss_computer._hand_object_penetration_loss``'s ``signed_proxy`` branch).

Object rotation/translation and the rest of the body stay fixed constants; only
the grasping hand/wrist/elbow pose residuals are optimized.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from grail.models.smplx_model import generate_smplx_mesh, get_smplx_segment_indices


# SMPL-X body_pose (poses[3:66]) is 21 joints x 3 axis-angle; joint j (1..21) is
# at poses[3 + (j-1)*3 : ...].  Full SMPL joint ids: L/R elbow = 18/19,
# L/R wrist = 20/21.  Fingertip joints are indices into the 131-joint output.
_ARM_POSE_SLICES = {
    "left": {"elbow": slice(54, 57), "wrist": slice(60, 63), "hand_key": "left_hand_pose"},
    "right": {"elbow": slice(57, 60), "wrist": slice(63, 66), "hand_key": "right_hand_pose"},
}
_FINGERTIP_JOINTS = {
    "left": [27, 30, 33, 36, 39],   # L Index3/Middle3/Pinky3/Ring3/Thumb3
    "right": [42, 45, 48, 51, 54],  # R Index3/Middle3/Pinky3/Ring3/Thumb3
}
_HAND_SEGMENT = {"left": ["L_Hand"], "right": ["R_Hand"]}


@dataclass(frozen=True)
class FingerGraspConfig:
    contact_frame: int
    selected_hand: str
    iterations: int = 150
    learning_rate: float = 0.02
    contact_weight: float = 8.0
    penetration_weight: float = 300.0
    target_clearance: float = 0.005      # fingertips rest ~5mm off the surface
    # A grasp requires the palm/fingers to lightly touch the surface, so we only
    # punish real penetration (verts going inside); a positive min_clearance
    # would forbid contact outright. Keep it ~0 to allow touch, forbid inside.
    min_clearance: float = 0.0           # penalize verts inside the object (sd<0)
    hand_pose_reg_weight: float = 0.5
    wrist_reg_weight: float = 1.0
    elbow_reg_weight: float = 3.0
    temporal_weight: float = 2.0
    # Penetration is dominated by the few deepest vertices, so emphasize the
    # worst ones (mean over the deepest fraction) in addition to the mean.
    penetration_worst_weight: float = 8.0
    penetration_worst_fraction: float = 0.1
    max_hand_change_rad: float = 1.2
    max_wrist_change_rad: float = 0.8
    max_elbow_change_rad: float = 0.45
    candidate_faces: int = 64
    max_hand_vertices: int = 160         # subsample hand verts for penetration
    grad_clip_norm: float = 1.0


def signed_point_mesh_distance(
    points: torch.Tensor,
    object_vertices: torch.Tensor,
    object_faces: torch.Tensor,
    *,
    candidate_faces: int = 64,
) -> torch.Tensor:
    """Signed distance (>0 outside, <0 inside) of each point to a mesh.

    Candidate triangles are found by KNN on face centroids; the nearest one gives
    an outward-oriented signed plane distance (edge fallback outside the triangle).
    Same formulation as loss_computer's signed_proxy penetration term.
    """
    from pytorch3d.ops import knn_points

    triangles = object_vertices[object_faces.long()].float()  # [F,3,3]
    centroids = triangles.mean(dim=1)
    object_center = object_vertices.mean(dim=0)
    candidate_count = min(int(candidate_faces), triangles.shape[0])
    idx = knn_points(points[None].float(), centroids[None].float(), K=candidate_count).idx[0]
    candidate = triangles[idx]  # [N,K,3,3]
    p = points[:, None, :]
    a, b, c = candidate.unbind(dim=2)
    ab, ac = b - a, c - a
    normal = torch.cross(ab, ac, dim=-1)
    normal_sq_raw = normal.square().sum(dim=-1)
    normal_sq = normal_sq_raw.clamp_min(1e-12)
    outward = (normal * (candidate.mean(dim=2) - object_center)).sum(dim=-1)
    normal = torch.where(outward[..., None] < 0.0, -normal, normal)
    signed_num = ((p - a) * normal).sum(dim=-1)
    projected = p - (signed_num / normal_sq)[..., None] * normal
    v0, v1, v2 = ab, ac, projected - a
    d00 = (v0 * v0).sum(dim=-1)
    d01 = (v0 * v1).sum(dim=-1)
    d11 = (v1 * v1).sum(dim=-1)
    d20 = (v2 * v0).sum(dim=-1)
    d21 = (v2 * v1).sum(dim=-1)
    denom_raw = d00 * d11 - d01.square()
    denom = denom_raw.clamp_min(1e-12)
    bary_v = (d11 * d20 - d01 * d21) / denom
    bary_w = (d00 * d21 - d01 * d20) / denom
    bary_u = 1.0 - bary_v - bary_w
    inside = (
        (normal_sq_raw > 1e-12) & (denom_raw > 1e-12)
        & (bary_u >= 0.0) & (bary_v >= 0.0) & (bary_w >= 0.0)
    )

    def edge_sq(s, e):
        edge = e - s
        alpha = ((p - s) * edge).sum(dim=-1) / edge.square().sum(dim=-1).clamp_min(1e-12)
        closest = s + alpha.clamp(0.0, 1.0)[..., None] * edge
        return (p - closest).square().sum(dim=-1)

    edge_min = torch.minimum(edge_sq(a, b), torch.minimum(edge_sq(b, c), edge_sq(c, a)))
    plane_sq = signed_num.square() / normal_sq
    squared = torch.where(inside, plane_sq, edge_min)
    nearest = squared.argmin(dim=1)
    rows = torch.arange(points.shape[0], device=points.device)
    signed = signed_num[rows, nearest] / normal_sq[rows, nearest].sqrt()
    return signed


def _object_mesh_cam(object_vertices_local, pose):
    """Object vertices in camera space at one frame: v @ R^T + t."""
    R = pose[:3, :3]
    t = pose[:3, 3]
    return object_vertices_local @ R.T + t


def refine_finger_grasp(
    motion_incam: Dict,
    object_vertices_local: torch.Tensor,
    object_faces: torch.Tensor,
    object_poses_cam: np.ndarray,
    contact_frame: int,
    selected_hand: str,
    smplx_model,
    config: FingerGraspConfig,
    device: str = "cuda",
) -> Tuple[Dict, Dict]:
    """Optimize the grasping hand so fingers contact the object without
    penetrating, for frames [contact_frame, end].  Returns a new motion dict
    (same schema) and a diagnostics dict.  Does not mutate the input."""
    hand = str(selected_hand).lower()
    if hand not in _ARM_POSE_SLICES:
        raise ValueError(f"selected_hand must be left/right, got {selected_hand!r}")
    arm = _ARM_POSE_SLICES[hand]
    hand_key = arm["hand_key"]

    poses0 = torch.as_tensor(np.asarray(motion_incam["poses"]), device=device, dtype=torch.float32)
    betas = torch.as_tensor(np.asarray(motion_incam["betas"]), device=device, dtype=torch.float32)
    trans = torch.as_tensor(np.asarray(motion_incam["trans"]), device=device, dtype=torch.float32)
    scale = float(motion_incam.get("scale", 1.0))
    frame_num = poses0.shape[0]
    hand0 = torch.as_tensor(
        np.asarray(motion_incam[hand_key]), device=device, dtype=torch.float32
    ).reshape(frame_num, 45)
    other_key = "right_hand_pose" if hand == "left" else "left_hand_pose"
    other_hand = motion_incam.get(other_key)
    other_hand = (
        torch.as_tensor(np.asarray(other_hand), device=device, dtype=torch.float32).reshape(frame_num, 45)
        if other_hand is not None else torch.zeros(frame_num, 45, device=device)
    )

    frame = int(contact_frame)
    sel = slice(frame, frame_num)
    n_sel = frame_num - frame
    if n_sel <= 0:
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in motion_incam.items()}, {"frames": 0}

    obj_verts = torch.as_tensor(np.asarray(object_vertices_local), device=device, dtype=torch.float32)
    obj_faces = torch.as_tensor(np.asarray(object_faces), device=device, dtype=torch.long)
    poses_cam = torch.as_tensor(np.asarray(object_poses_cam), device=device, dtype=torch.float32)

    hand_vert_idx = torch.as_tensor(
        get_smplx_segment_indices(_HAND_SEGMENT[hand]), device=device, dtype=torch.long
    )
    if hand_vert_idx.numel() > config.max_hand_vertices:
        pick = torch.linspace(0, hand_vert_idx.numel() - 1, config.max_hand_vertices, device=device).long()
        hand_vert_idx = hand_vert_idx[pick]
    fingertip_idx = torch.as_tensor(_FINGERTIP_JOINTS[hand], device=device, dtype=torch.long)

    hand_res = torch.zeros(n_sel, 45, device=device, requires_grad=True)
    wrist_res = torch.zeros(n_sel, 3, device=device, requires_grad=True)
    elbow_res = torch.zeros(n_sel, 3, device=device, requires_grad=True)
    optim = torch.optim.Adam([hand_res, wrist_res, elbow_res], lr=config.learning_rate)

    def decode():
        poses = poses0[sel].clone()
        poses[:, arm["wrist"]] = poses[:, arm["wrist"]] + wrist_res
        poses[:, arm["elbow"]] = poses[:, arm["elbow"]] + elbow_res
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

    def losses():
        verts, joints = decode()
        hand_verts = verts[:, hand_vert_idx]        # [n_sel, Hv, 3]
        fingertips = joints[:, fingertip_idx]       # [n_sel, 5, 3]
        contact_terms, pen_terms, min_signed = [], [], []
        for i in range(n_sel):
            obj_cam = _object_mesh_cam(obj_verts, poses_cam[frame + i])
            sd_tip = signed_point_mesh_distance(
                fingertips[i], obj_cam, obj_faces, candidate_faces=config.candidate_faces
            )
            sd_hand = signed_point_mesh_distance(
                hand_verts[i], obj_cam, obj_faces, candidate_faces=config.candidate_faces
            )
            contact_terms.append((sd_tip - config.target_clearance).square().mean())
            pen = torch.relu(config.min_clearance - sd_hand).square()   # [Hv]
            # Penetration is dominated by a few deep vertices; the plain mean
            # dilutes their gradient across the many non-penetrating verts, so
            # add a worst-fraction term that focuses on the deepest offenders.
            k = max(1, int(round(config.penetration_worst_fraction * pen.numel())))
            worst = torch.topk(pen, k=min(k, pen.numel())).values.mean()
            pen_terms.append(pen.mean() + config.penetration_worst_weight * worst)
            min_signed.append(sd_hand.min())
        contact = torch.stack(contact_terms).mean()
        penetration = torch.stack(pen_terms).mean()
        reg = (
            config.hand_pose_reg_weight * hand_res.square().mean()
            + config.wrist_reg_weight * wrist_res.square().mean()
            + config.elbow_reg_weight * elbow_res.square().mean()
        )
        temporal = hand_res.new_zeros(())
        if n_sel > 2:
            temporal = (hand_res[2:] - 2 * hand_res[1:-1] + hand_res[:-2]).square().mean()
        total = (
            config.contact_weight * contact
            + config.penetration_weight * penetration
            + reg
            + config.temporal_weight * temporal
        )
        return total, contact, penetration, torch.stack(min_signed)

    with torch.no_grad():
        _, c0, p0, ms0 = losses()
        before = {
            "contact_error_m": float(c0.sqrt().detach().cpu()),
            "penetration_loss": float(p0.detach().cpu()),
            "min_signed_distance_m": float(ms0.min().detach().cpu()),
        }

    for _ in range(int(config.iterations)):
        optim.zero_grad(set_to_none=True)
        total, contact, penetration, _ = losses()
        if not torch.isfinite(total):
            raise FloatingPointError("finger grasp loss contains NaN/Inf")
        total.backward()
        torch.nn.utils.clip_grad_norm_([hand_res, wrist_res, elbow_res], config.grad_clip_norm)
        optim.step()
        with torch.no_grad():
            hand_res.clamp_(-config.max_hand_change_rad, config.max_hand_change_rad)
            wrist_res.clamp_(-config.max_wrist_change_rad, config.max_wrist_change_rad)
            elbow_res.clamp_(-config.max_elbow_change_rad, config.max_elbow_change_rad)

    with torch.no_grad():
        _, c1, p1, ms1 = losses()

    # Write the refined pose back (frames >= contact_frame).
    out = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in motion_incam.items()}
    poses_out = np.array(motion_incam["poses"], copy=True)
    poses_out[frame:, arm["wrist"]] += wrist_res.detach().cpu().numpy()
    poses_out[frame:, arm["elbow"]] += elbow_res.detach().cpu().numpy()
    hand_out = np.array(motion_incam[hand_key], copy=True).reshape(frame_num, 45)
    hand_out[frame:] += hand_res.detach().cpu().numpy()
    # keep poses hand slots in sync with the hand key (offset 66=left, 111=right)
    hand_offset = 66 if hand == "left" else 111
    poses_out[:, hand_offset:hand_offset + 45] = hand_out
    out["poses"] = poses_out
    out[hand_key] = hand_out

    diagnostics = {
        "frames": int(n_sel),
        "hand": hand,
        "contact_error_before_m": before["contact_error_m"],
        "contact_error_after_m": float(c1.sqrt().detach().cpu()),
        "penetration_before": before["penetration_loss"],
        "penetration_after": float(p1.detach().cpu()),
        "min_signed_distance_before_m": before["min_signed_distance_m"],
        "min_signed_distance_after_m": float(ms1.min().detach().cpu()),
        "min_clearance_m": float(config.min_clearance),
        "target_clearance_m": float(config.target_clearance),
    }
    return out, diagnostics
