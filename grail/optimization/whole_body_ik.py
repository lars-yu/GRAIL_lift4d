"""Anatomical helpers for fixed-object whole-body grasp IK.

The fixed-object solve is deliberately asymmetric: the recovered object
trajectory is immutable and every contact correction is absorbed by the human
root, legs, torso, and contact-side arm.  This module keeps the small pieces of
that solve independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


@dataclass(frozen=True)
class SupportFootAnchors:
    """World-space ankle targets for the detected support episodes."""

    reference_world: torch.Tensor  # (L, 2, 3), left/right episode anchors
    support_mask: torch.Tensor  # (L, 2), left/right support flags


@dataclass(frozen=True)
class FootstepPlan:
    """Detached world-space targets for a step-aware approach.

    ``stance_mask`` and ``swing_mask`` are mutually exclusive.  A foot is
    anchored while it is in stance and tracks the smooth target (including a
    vertical clearance arc) while it is in swing.  After touchdown the new
    target becomes the immutable anchor for the next stance episode; it is
    deliberately not tied to the original HMR foot location.
    """

    foot_targets_world: torch.Tensor  # (L,2,3), stance anchors + swing path
    stance_mask: torch.Tensor  # (L,2)
    swing_mask: torch.Tensor  # (L,2)
    touchdown_mask: torch.Tensor  # (L,2), last frame of each swing
    root_translation_residual: torch.Tensor  # (L,3), pelvis path from feet
    step_lengths: torch.Tensor  # (S,), distance of each individual footstep
    step_intervals: tuple[tuple[int, int, int], ...]  # (start,end,side)
    approach_start: int
    move_start: int
    first_swing_side: int
    gravity_axis: int
    max_step_length: float
    swing_clearance: float

    @property
    def step_count(self) -> int:
        return len(self.step_intervals)


def _feet_from_joints(
    body_joints_world: torch.Tensor, foot_joint_indices: tuple[int, int]
) -> torch.Tensor:
    if body_joints_world.ndim != 3 or body_joints_world.shape[-1] != 3:
        raise ValueError("body_joints_world must have shape [L,J,3]")
    left_idx, right_idx = map(int, foot_joint_indices)
    if min(left_idx, right_idx) < 0 or max(left_idx, right_idx) >= body_joints_world.shape[1]:
        raise ValueError("foot joint index is outside body_joints_world")
    return torch.stack(
        (body_joints_world[:, left_idx], body_joints_world[:, right_idx]), dim=1
    )


def build_alternating_footstep_plan(
    body_joints_world: torch.Tensor,
    foot_joint_indices: tuple[int, int],
    root_displacement: torch.Tensor,
    move_start: int,
    approach_window: int,
    *,
    first_swing_side: int = 1,
    gravity_axis: int = 2,
    max_step_length: float = 0.22,
    max_steps: int = 4,
    swing_clearance: float = 0.05,
    min_swing_frames: int = 8,
    settle_frames: int = 3,
) -> FootstepPlan:
    """Plan alternating left/right steps that realize a ground root shift.

    Each round advances both feet by the same fraction of the desired root
    displacement.  Consequently the mean planned foot displacement provides
    a physically interpretable pelvis path: the root advances only as the feet
    advance, instead of sliding independently through the world.
    """

    feet = _feet_from_joints(body_joints_world, foot_joint_indices).detach()
    frame_num = int(feet.shape[0])
    move_start = int(move_start)
    approach_window = int(approach_window)
    first_swing_side = int(first_swing_side)
    gravity_axis = int(gravity_axis)
    max_steps = int(max_steps)
    min_swing_frames = int(min_swing_frames)
    settle_frames = int(settle_frames)
    if not 0 < move_start < frame_num:
        raise ValueError("move_start must be inside the sequence and greater than zero")
    if approach_window < 1:
        raise ValueError("approach_window must be positive")
    if first_swing_side not in (0, 1):
        raise ValueError("first_swing_side must be 0 (left) or 1 (right)")
    if gravity_axis not in (0, 1, 2):
        raise ValueError("gravity_axis must be 0, 1, or 2")
    if not math.isfinite(float(max_step_length)) or float(max_step_length) <= 0.0:
        raise ValueError("max_step_length must be finite and positive")
    if max_steps < 2:
        raise ValueError("max_steps must allow at least one left/right step pair")
    if not math.isfinite(float(swing_clearance)) or float(swing_clearance) <= 0.0:
        raise ValueError("swing_clearance must be finite and positive")
    if min_swing_frames < 3:
        raise ValueError("min_swing_frames must be at least three")
    if settle_frames < 1:
        raise ValueError("settle_frames must be positive")

    displacement = torch.as_tensor(
        root_displacement, device=feet.device, dtype=feet.dtype
    ).detach().reshape(3).clone()
    displacement[gravity_axis] = 0.0
    distance = torch.linalg.norm(displacement)
    if not torch.isfinite(distance) or float(distance) < 1e-4:
        raise ValueError("A step-aware solve requires a non-zero ground displacement")

    rounds = int(math.ceil(float(distance) / float(max_step_length)))
    step_count = 2 * rounds
    if step_count > max_steps:
        raise ValueError(
            "Required root displacement needs "
            f"{step_count} steps, exceeding max_steps={max_steps}; increase "
            "--max-approach-steps or --max-step-length"
        )

    approach_start = max(0, move_start - approach_window)
    swing_end_exclusive = move_start - settle_frames + 1
    available = swing_end_exclusive - approach_start
    if available < step_count * min_swing_frames:
        raise ValueError(
            "Approach window is too short for the planned steps: "
            f"available={available}, steps={step_count}, "
            f"min_swing_frames={min_swing_frames}"
        )

    # Rounded linspace boundaries distribute spare frames without creating a
    # long last step.  The endpoint is exclusive, so each interval remains
    # contiguous and the final settle frames are double support.
    boundaries = torch.linspace(
        float(approach_start),
        float(swing_end_exclusive),
        step_count + 1,
        device=feet.device,
    ).round().to(torch.long)
    initial_feet = feet[approach_start].clone()
    targets = initial_feet.reshape(1, 2, 3).repeat(frame_num, 1, 1)
    stance = torch.zeros(frame_num, 2, dtype=torch.bool, device=feet.device)
    stance[approach_start:] = True
    swing = torch.zeros_like(stance)
    touchdown = torch.zeros_like(stance)
    intervals: list[tuple[int, int, int]] = []
    lengths: list[torch.Tensor] = []

    for step_index in range(step_count):
        side = (first_swing_side + step_index) % 2
        round_index = step_index // 2 + 1
        start = int(boundaries[step_index])
        end = int(boundaries[step_index + 1]) - 1
        if end - start + 1 < min_swing_frames:
            raise AssertionError("Rounded footstep interval violated min_swing_frames")
        takeoff = targets[start, side].clone()
        landing = initial_feet[side] + displacement * (round_index / rounds)
        phase = torch.linspace(
            0.0, 1.0, end - start + 1, device=feet.device, dtype=feet.dtype
        )
        smooth = phase.square() * (3.0 - 2.0 * phase)
        path = takeoff[None] + smooth[:, None] * (landing - takeoff)[None]
        path[:, gravity_axis] += float(swing_clearance) * torch.sin(math.pi * phase)
        targets[start : end + 1, side] = path
        targets[end + 1 :, side] = landing
        stance[start : end + 1, side] = False
        swing[start : end + 1, side] = True
        touchdown[end, side] = True
        intervals.append((start, end, side))
        lengths.append(torch.linalg.norm(landing - takeoff))

    # The pelvis follows the mean progress of the two feet.  Its vertical
    # coordinate remains an optimizable small residual rather than bobbing by
    # half the swing clearance.
    root_path = targets.mean(dim=1) - initial_feet.mean(dim=0, keepdim=True)
    root_path[:, gravity_axis] = 0.0
    root_path[:approach_start] = 0.0
    root_path[move_start:] = displacement
    step_lengths = torch.stack(lengths).detach()
    if float(step_lengths.max()) > float(max_step_length) + 1e-5:
        raise AssertionError("Generated footstep exceeded max_step_length")

    return FootstepPlan(
        foot_targets_world=targets.detach(),
        stance_mask=stance.detach(),
        swing_mask=swing.detach(),
        touchdown_mask=touchdown.detach(),
        root_translation_residual=root_path.detach(),
        step_lengths=step_lengths,
        step_intervals=tuple(intervals),
        approach_start=approach_start,
        move_start=move_start,
        first_swing_side=first_swing_side,
        gravity_axis=gravity_axis,
        max_step_length=float(max_step_length),
        swing_clearance=float(swing_clearance),
    )


def footstep_target_error(
    body_joints_world: torch.Tensor,
    foot_joint_indices: tuple[int, int],
    plan: FootstepPlan,
    *,
    phase: str,
) -> torch.Tensor:
    """Return stance, swing, or touchdown target errors in metres."""

    feet = _feet_from_joints(body_joints_world, foot_joint_indices)
    if feet.shape != plan.foot_targets_world.shape:
        raise ValueError("Footstep target shape does not match predicted feet")
    masks = {
        "stance": plan.stance_mask,
        "swing": plan.swing_mask,
        "touchdown": plan.touchdown_mask,
    }
    if phase not in masks:
        raise ValueError("phase must be stance, swing, or touchdown")
    mask = masks[phase].to(device=feet.device)
    distance = torch.linalg.norm(
        feet
        - plan.foot_targets_world.to(device=feet.device, dtype=feet.dtype),
        dim=-1,
    )
    return distance[mask]


def build_support_foot_anchors(
    body_joints_world: torch.Tensor,
    foot_joint_indices: tuple[int, int],
    foot_contact_probs: torch.Tensor | None,
    *,
    threshold: float = 0.5,
) -> SupportFootAnchors:
    """Build one constant ankle anchor for every contiguous support episode.

    Tracking the original per-frame ankle position would preserve HMR jitter.
    An episode mean instead encodes the physical statement that a supporting
    foot is planted at one world-space location while root and leg IK change.
    """

    if body_joints_world.ndim != 3 or body_joints_world.shape[-1] != 3:
        raise ValueError("body_joints_world must have shape [L,J,3]")
    left_idx, right_idx = map(int, foot_joint_indices)
    if min(left_idx, right_idx) < 0 or max(left_idx, right_idx) >= body_joints_world.shape[1]:
        raise ValueError("foot joint index is outside body_joints_world")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("support threshold must be in [0,1]")

    feet = torch.stack(
        (body_joints_world[:, left_idx], body_joints_world[:, right_idx]), dim=1
    ).detach()
    frame_num = feet.shape[0]
    if foot_contact_probs is None:
        return SupportFootAnchors(
            reference_world=feet.clone(),
            support_mask=torch.zeros(
                frame_num, 2, dtype=torch.bool, device=feet.device
            ),
        )
    probs = foot_contact_probs.detach().to(device=feet.device, dtype=feet.dtype)
    if probs.shape != (frame_num, 4):
        raise ValueError("foot_contact_probs must have shape [L,4]")
    support = torch.stack(
        (probs[:, :2].amax(dim=1), probs[:, 2:].amax(dim=1)), dim=1
    ) > float(threshold)
    reference = feet.clone()

    for side in range(2):
        active = support[:, side].tolist()
        start = 0
        while start < frame_num:
            if not active[start]:
                start += 1
                continue
            end = start + 1
            while end < frame_num and active[end]:
                end += 1
            anchor = feet[start:end, side].mean(dim=0)
            reference[start:end, side] = anchor
            start = end

    return SupportFootAnchors(
        reference_world=reference.detach(), support_mask=support.detach()
    )


def support_foot_anchor_error(
    body_joints_world: torch.Tensor,
    foot_joint_indices: tuple[int, int],
    anchors: SupportFootAnchors,
) -> torch.Tensor:
    """Return active support-foot distances in metres."""

    left_idx, right_idx = map(int, foot_joint_indices)
    feet = torch.stack(
        (body_joints_world[:, left_idx], body_joints_world[:, right_idx]), dim=1
    )
    if feet.shape != anchors.reference_world.shape:
        raise ValueError("support-foot reference shape does not match predicted feet")
    mask = anchors.support_mask.to(device=feet.device)
    if mask.shape != feet.shape[:2]:
        raise ValueError("support-foot mask shape does not match predicted feet")
    distances = torch.linalg.norm(
        feet - anchors.reference_world.to(device=feet.device, dtype=feet.dtype), dim=-1
    )
    return distances[mask]


def elbow_angles_degrees(body_joints_world: torch.Tensor) -> torch.Tensor:
    """Return left/right COCO17 elbow angles as an ``[L,2]`` tensor."""

    if body_joints_world.ndim != 3 or body_joints_world.shape[1] < 11:
        raise ValueError("elbow anatomy requires COCO17 body joints")

    def angle(shoulder: int, elbow: int, wrist: int) -> torch.Tensor:
        upper = body_joints_world[:, shoulder] - body_joints_world[:, elbow]
        lower = body_joints_world[:, wrist] - body_joints_world[:, elbow]
        cosine = torch.nn.functional.cosine_similarity(upper, lower, dim=-1, eps=1e-8)
        return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))

    return torch.stack((angle(5, 7, 9), angle(6, 8, 10)), dim=1)


def contact_elbow_angles_degrees(
    body_joints_world: torch.Tensor, contact_hand: str
) -> torch.Tensor:
    """Select the elbow angle(s) belonging to the configured contact hand."""

    angles = elbow_angles_degrees(body_joints_world)
    hand = str(contact_hand).lower()
    if hand == "left":
        return angles[:, :1]
    if hand == "right":
        return angles[:, 1:]
    if hand == "both":
        return angles
    raise ValueError(f"Unsupported contact hand: {contact_hand!r}")


def rotation_residual_angle_degrees(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Geodesic magnitude of identity-centred 6D rotation residuals."""

    matrix = rotation_6d_to_matrix(rotation_6d)
    axis_angle = matrix_to_axis_angle(matrix)
    return torch.rad2deg(torch.linalg.norm(axis_angle, dim=-1))


@torch.no_grad()
def project_rotation_residuals_(
    rotation_6d: torch.Tensor,
    joint_limits_degrees: dict[int, float],
    *,
    frame_mask: torch.Tensor | None = None,
) -> None:
    """Project selected identity-centred residual rotations into trust regions."""

    if rotation_6d.ndim != 3 or rotation_6d.shape[-1] != 6:
        raise ValueError("rotation_6d must have shape [L,J,6]")
    if frame_mask is not None and frame_mask.shape != (rotation_6d.shape[0],):
        raise ValueError("frame_mask must have shape [L]")
    active_frames = (
        torch.ones(rotation_6d.shape[0], dtype=torch.bool, device=rotation_6d.device)
        if frame_mask is None
        else frame_mask.to(device=rotation_6d.device, dtype=torch.bool)
    )
    for joint, limit_degrees in joint_limits_degrees.items():
        joint = int(joint)
        limit = math.radians(float(limit_degrees))
        if not 0 <= joint < rotation_6d.shape[1]:
            continue
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError("rotation trust-region limits must be finite and positive")
        selected = rotation_6d[active_frames, joint]
        if selected.numel() == 0:
            continue
        axis_angle = matrix_to_axis_angle(rotation_6d_to_matrix(selected))
        magnitude = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
        axis_angle = axis_angle * torch.clamp(limit / magnitude.clamp_min(1e-8), max=1.0)
        rotation_6d[active_frames, joint] = matrix_to_rotation_6d(
            axis_angle_to_matrix(axis_angle)
        )
