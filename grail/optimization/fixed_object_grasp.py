"""Utilities for fixed-object human grasp inverse kinematics.

The object trajectory is treated as immutable.  A physical palm target at the
contact frame is converted to object-local coordinates and then transported by
the known object SE(3) trajectory.  This prevents the world-offset grasp drift
that occurs when a rotating object is tracked with translation only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FixedObjectGraspTargets:
    """Detached object-local grasp anchor and its world-space trajectory."""

    contact_frame: int
    anchor_object: torch.Tensor
    normal_object: torch.Tensor | None
    position_world: torch.Tensor
    normal_world: torch.Tensor | None


def _validate_pose_sequence(
    object_rotation: torch.Tensor,
    object_translation: torch.Tensor,
) -> None:
    if object_rotation.ndim != 3 or object_rotation.shape[-2:] != (3, 3):
        raise ValueError("object_rotation must have shape [L,3,3]")
    if object_translation.ndim != 2 or object_translation.shape[-1] != 3:
        raise ValueError("object_translation must have shape [L,3]")
    if object_rotation.shape[0] != object_translation.shape[0]:
        raise ValueError("object rotation and translation lengths must match")


def world_point_to_object_local(
    point_world: torch.Tensor,
    object_rotation: torch.Tensor,
    object_translation: torch.Tensor,
) -> torch.Tensor:
    """Convert a world point to the repository's row-vector object frame."""
    if point_world.shape != (3,):
        raise ValueError("point_world must have shape [3]")
    if object_rotation.shape != (3, 3) or object_translation.shape != (3,):
        raise ValueError("object pose must be one [3,3] rotation and [3] translation")
    return torch.matmul(point_world - object_translation, object_rotation)


def object_local_point_to_world(
    point_object: torch.Tensor,
    object_rotation: torch.Tensor,
    object_translation: torch.Tensor,
) -> torch.Tensor:
    """Transport one object-local point through a full object trajectory."""
    _validate_pose_sequence(object_rotation, object_translation)
    if point_object.shape != (3,):
        raise ValueError("point_object must have shape [3]")
    return torch.matmul(
        point_object.reshape(1, 1, 3),
        object_rotation.transpose(1, 2),
    ).reshape(-1, 3) + object_translation


def build_fixed_object_grasp_targets(
    contact_position_world: torch.Tensor,
    object_rotation: torch.Tensor,
    object_translation: torch.Tensor,
    contact_frame: int,
    *,
    contact_normal_world: torch.Tensor | None = None,
) -> FixedObjectGraspTargets:
    """Freeze a contact anchor in object coordinates and transport it over time."""
    _validate_pose_sequence(object_rotation, object_translation)
    frame_num = int(object_translation.shape[0])
    contact_frame = int(contact_frame)
    if not 0 <= contact_frame < frame_num:
        raise ValueError("contact_frame is outside the object trajectory")
    if contact_position_world.shape != (3,):
        raise ValueError("contact_position_world must have shape [3]")

    rotation = object_rotation.detach()
    translation = object_translation.detach()
    position = contact_position_world.detach()
    anchor_object = world_point_to_object_local(
        position,
        rotation[contact_frame],
        translation[contact_frame],
    ).detach()
    position_world = object_local_point_to_world(
        anchor_object, rotation, translation
    ).detach()

    normal_object = None
    normal_world = None
    if contact_normal_world is not None:
        if contact_normal_world.shape != (3,):
            raise ValueError("contact_normal_world must have shape [3]")
        normal = torch.nn.functional.normalize(
            contact_normal_world.detach(), dim=0, eps=1e-8
        )
        normal_object = torch.matmul(normal, rotation[contact_frame]).detach()
        normal_world = torch.matmul(
            normal_object.reshape(1, 1, 3), rotation.transpose(1, 2)
        ).reshape(-1, 3)
        normal_world = torch.nn.functional.normalize(
            normal_world, dim=-1, eps=1e-8
        ).detach()

    return FixedObjectGraspTargets(
        contact_frame=contact_frame,
        anchor_object=anchor_object,
        normal_object=normal_object,
        position_world=position_world,
        normal_world=normal_world,
    )


def ground_alignment_delta(
    actual_palm_world: torch.Tensor,
    target_palm_world: torch.Tensor,
    *,
    gravity_axis: int = 2,
    max_distance: float = 0.35,
) -> torch.Tensor:
    """Return a bounded constant ground-plane correction for the full human track."""
    if actual_palm_world.shape != (3,) or target_palm_world.shape != (3,):
        raise ValueError("palm points must each have shape [3]")
    if gravity_axis not in (0, 1, 2):
        raise ValueError("gravity_axis must be 0, 1, or 2")
    max_distance = float(max_distance)
    if not torch.isfinite(actual_palm_world).all() or not torch.isfinite(target_palm_world).all():
        raise ValueError("palm points must be finite")
    if max_distance <= 0.0:
        raise ValueError("max_distance must be positive")
    delta = target_palm_world - actual_palm_world
    delta = delta.clone()
    delta[gravity_axis] = 0.0
    distance = torch.linalg.norm(delta)
    scale = torch.clamp(delta.new_tensor(max_distance) / distance.clamp_min(1e-8), max=1.0)
    return delta * scale


def fixed_object_grasp_position_error(
    palm_world: torch.Tensor,
    target_world: torch.Tensor,
    contact_frame: int,
) -> torch.Tensor:
    """Per-frame post-contact palm error used by formal acceptance gates."""
    if palm_world.shape != target_world.shape or palm_world.ndim != 2 or palm_world.shape[1] != 3:
        raise ValueError("palm_world and target_world must have matching [L,3] shapes")
    contact_frame = int(contact_frame)
    if not 0 <= contact_frame < palm_world.shape[0]:
        raise ValueError("contact_frame is outside the palm trajectory")
    return torch.linalg.norm(
        palm_world[contact_frame:] - target_world[contact_frame:].detach(), dim=-1
    )

