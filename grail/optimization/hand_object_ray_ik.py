"""Mask-driven contact-hand selection and camera-ray IK targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class ContactHandSelection:
    hand: str
    left_distance_px: np.ndarray
    right_distance_px: np.ndarray
    reason: str
    used_fallback: bool


def _hand_distance_curve(points, masks, start, end, confidence_threshold):
    curve = np.full(masks.shape[0], np.nan, dtype=np.float32)
    reliable = 0
    for frame in range(start, end + 1):
        kp = points[frame]
        valid = (
            np.isfinite(kp).all(axis=1)
            & (kp[:, 2] >= confidence_threshold)
        )
        if not valid.any():
            continue
        distance = distance_transform_edt(~masks[frame])
        xy = np.rint(kp[valid, :2]).astype(np.int64)
        inside = (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < masks.shape[2])
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < masks.shape[1])
        )
        if not inside.any():
            continue
        xy = xy[inside]
        curve[frame] = float(np.median(distance[xy[:, 1], xy[:, 0]]))
        reliable += int(xy.shape[0])
    return curve, reliable


def _as_confident_points(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] not in (2, 3):
        raise ValueError(f"hand points must have shape [T,J,2/3], got {points.shape}")
    if points.shape[-1] == 2:
        confidence = np.ones((*points.shape[:2], 1), dtype=points.dtype)
        points = np.concatenate([points, confidence], axis=-1)
    return points


def select_contact_hand_from_masks(
    hand_keypoints_2d: np.ndarray,
    object_masks: np.ndarray | list,
    t_move: int,
    *,
    projected_hand_points_2d: np.ndarray | None = None,
    lookback_frames: int = 5,
    confidence_threshold: float = 0.2,
    both_distance_px: float = 12.0,
    both_ratio: float = 1.25,
) -> ContactHandSelection:
    """Select the contact hand from raw 2D evidence, independently of GPT labels."""
    points = _as_confident_points(hand_keypoints_2d)
    masks = np.asarray(object_masks).astype(bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[0] != points.shape[0]:
        raise ValueError("Object masks and hand keypoints must be frame-aligned")
    if points.shape[1] < 2 or points.shape[1] % 2:
        raise ValueError("Hand keypoints must contain equal left/right groups")
    if not 1 <= int(t_move) < points.shape[0]:
        raise ValueError(f"t_move must be in [1,{points.shape[0] - 1}]")
    split = points.shape[1] // 2
    start = max(0, int(t_move) - int(lookback_frames))
    end = int(t_move)
    left, left_count = _hand_distance_curve(
        points[:, :split], masks, start, end, confidence_threshold
    )
    right, right_count = _hand_distance_curve(
        points[:, split:], masks, start, end, confidence_threshold
    )
    used_fallback = False
    reason = "raw_2d_keypoints_distance_transform"
    required = max(2, end - start + 1)
    if left_count < required or right_count < required:
        if projected_hand_points_2d is None:
            raise ValueError(
                "Raw hand keypoints are unreliable and projected SMPL-X fallback is missing"
            )
        fallback = _as_confident_points(projected_hand_points_2d)
        if fallback.shape[:2] != points.shape[:2]:
            raise ValueError("Projected SMPL-X hand fallback shape does not match raw keypoints")
        left, left_count = _hand_distance_curve(
            fallback[:, :split], masks, start, end, 0.0
        )
        right, right_count = _hand_distance_curve(
            fallback[:, split:], masks, start, end, 0.0
        )
        used_fallback = True
        reason = "projected_initial_smplx_hand_fallback"
    left_median = float(np.nanmedian(left[start : end + 1]))
    right_median = float(np.nanmedian(right[start : end + 1]))
    if not np.isfinite(left_median) or not np.isfinite(right_median):
        raise ValueError("Could not measure either hand against the object mask")
    near = max(left_median, right_median) <= float(both_distance_px)
    similar = max(left_median, right_median) <= float(both_ratio) * max(
        min(left_median, right_median), 1.0
    )
    if near and similar:
        hand = "both"
    elif left_median < right_median:
        hand = "left"
    else:
        hand = "right"
    reason += f"; left_median={left_median:.3f}px right_median={right_median:.3f}px"
    return ContactHandSelection(hand, left, right, reason, used_fallback)


def approach_window_from_fps(fps: float, seconds: float = 0.67) -> int:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return int(np.clip(round(float(fps) * float(seconds)), 10, 30))


def smoothstep_ramp(frame_num: int, t_move: int, window: int, *, device=None, dtype=None):
    if not 1 <= int(t_move) < int(frame_num):
        raise ValueError("t_move must lie inside the sequence")
    window = int(np.clip(window, 1, t_move))
    frames = torch.arange(frame_num, device=device, dtype=dtype or torch.float32)
    u = ((frames - (t_move - window)) / float(window)).clamp(0.0, 1.0)
    return u.square() * (3.0 - 2.0 * u)


def camera_ray_hand_targets(
    initial_hand_cam: torch.Tensor,
    detached_surface_depth: torch.Tensor,
    t_move: int,
    window: int,
    *,
    target_distance: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move depth along the initial OpenCV ray without changing image coordinates."""
    if initial_hand_cam.ndim != 2 or initial_hand_cam.shape[1] != 3:
        raise ValueError("initial_hand_cam must have shape [T,3]")
    if detached_surface_depth.shape != (initial_hand_cam.shape[0],):
        raise ValueError("detached_surface_depth must have shape [T]")
    if torch.any(initial_hand_cam[:, 2] <= 0) or torch.any(detached_surface_depth <= 0):
        raise ValueError("Ray IK requires positive OpenCV camera depth")
    surface_depth = detached_surface_depth.detach()
    ray = initial_hand_cam / initial_hand_cam[:, 2:3]
    target_z = (surface_depth - float(target_distance)).clamp_min(1e-4)
    ramp = smoothstep_ramp(
        initial_hand_cam.shape[0],
        t_move,
        window,
        device=initial_hand_cam.device,
        dtype=initial_hand_cam.dtype,
    )
    z = initial_hand_cam[:, 2] + ramp * (target_z - initial_hand_cam[:, 2])
    return ray * z[:, None], ramp


def mesh_surface_depth_at_pixels(
    object_vertices_cam: torch.Tensor,
    query_pixels: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    *,
    top_k: int = 32,
) -> torch.Tensor:
    """Return detached positive mesh-surface depth near each hand pixel."""
    if object_vertices_cam.ndim != 3 or object_vertices_cam.shape[-1] != 3:
        raise ValueError("object_vertices_cam must have shape [T,V,3]")
    frame_num, vertex_num, _ = object_vertices_cam.shape
    if query_pixels.shape != (frame_num, 2):
        raise ValueError("query_pixels must have shape [T,2]")
    if camera_intrinsics.shape == (3, 3):
        camera_intrinsics = camera_intrinsics[None].expand(frame_num, -1, -1)
    if camera_intrinsics.shape != (frame_num, 3, 3):
        raise ValueError("camera_intrinsics must have shape [3,3] or [T,3,3]")
    vertices = object_vertices_cam.detach()
    if torch.any(vertices[..., 2] <= 0):
        raise ValueError("Object mesh contains non-positive OpenCV camera depth")
    u = camera_intrinsics[:, None, 0, 0] * vertices[..., 0] / vertices[..., 2]
    u = u + camera_intrinsics[:, None, 0, 2]
    v = camera_intrinsics[:, None, 1, 1] * vertices[..., 1] / vertices[..., 2]
    v = v + camera_intrinsics[:, None, 1, 2]
    projected = torch.stack([u, v], dim=-1)
    pixel_distance = torch.linalg.norm(projected - query_pixels[:, None], dim=-1)
    count = min(max(1, int(top_k)), vertex_num)
    nearest = torch.topk(pixel_distance, count, dim=1, largest=False).indices
    nearest_z = torch.gather(vertices[..., 2], 1, nearest)
    surface_z = nearest_z.median(dim=1).values.detach()
    if torch.any(surface_z <= 0) or not torch.isfinite(surface_z).all():
        raise ValueError("Mesh surface lookup produced invalid OpenCV camera depth")
    return surface_z


def continuous_grasp_losses(
    hand_center: torch.Tensor,
    detached_object_translation: torch.Tensor,
    t_move: int,
    *,
    delta: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Translation-only continuous grasp losses; object gradients are impossible."""
    object_translation = detached_object_translation.detach()
    hand = hand_center[int(t_move) :]
    obj = object_translation[int(t_move) :]
    if hand.shape[0] < 2:
        raise ValueError("Continuous grasp requires at least two moving frames")
    relative = hand - obj
    anchor = relative[0].detach()
    relative_error = torch.linalg.norm(relative - anchor, dim=-1)
    velocity_error = torch.linalg.norm(
        (hand[1:] - hand[:-1]) - (obj[1:] - obj[:-1]), dim=-1
    )
    acceleration = hand[2:] - 2.0 * hand[1:-1] + hand[:-2]
    zero = hand.new_zeros(())
    return {
        "relative": torch.nn.functional.huber_loss(
            relative_error, torch.zeros_like(relative_error), delta=delta
        ),
        "velocity": torch.nn.functional.huber_loss(
            velocity_error, torch.zeros_like(velocity_error), delta=delta
        ),
        "acceleration": acceleration.square().mean() if acceleration.numel() else zero,
    }
