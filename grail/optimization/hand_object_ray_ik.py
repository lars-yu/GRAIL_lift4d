"""Mask-driven contact-hand selection and camera-ray IK targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from grail.optimization.approach import minimum_jerk_ramp


@dataclass(frozen=True)
class ContactHandSelection:
    hand: str
    left_distance_px: np.ndarray
    right_distance_px: np.ndarray
    reason: str
    used_fallback: bool


def observed_palm_pixels_from_keypoints(
    hand_keypoints_2d: np.ndarray,
    contact_hand: str,
    *,
    fallback_pixels: np.ndarray | None = None,
    confidence_threshold: float = 0.2,
    smooth_window: int = 3,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Build a wrist/MCP palm pixel track from video keypoint evidence.

    The 32-point HMR hand layout is fixed as 16 left points followed by 16
    right points. Palm points are wrist plus the four MCP points. Missing or
    low-confidence frames use the explicitly supplied projected-model fallback
    and are returned in the per-frame fallback mask.
    """
    points = _as_confident_points(hand_keypoints_2d)
    if points.shape[1] != 32:
        raise ValueError(f"Expected the verified 32-point hand layout, got {points.shape}")
    hand = str(contact_hand).lower()
    local = (0, 1, 4, 7, 10)
    if hand == "left":
        indices = tuple(local)
    elif hand == "right":
        indices = tuple(i + 16 for i in local)
    elif hand == "both":
        indices = tuple(local) + tuple(i + 16 for i in local)
    else:
        raise ValueError(f"Unsupported contact hand: {contact_hand!r}")
    pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    fallback = np.zeros(points.shape[0], dtype=bool)
    for frame in range(points.shape[0]):
        selected = points[frame, list(indices)]
        valid = np.isfinite(selected[:, :2]).all(axis=1) & (
            selected[:, 2] >= float(confidence_threshold)
        )
        if valid.any():
            weights = np.maximum(selected[valid, 2], 1e-6)
            pixels[frame] = (selected[valid, :2] * weights[:, None]).sum(axis=0) / weights.sum()
        elif fallback_pixels is not None:
            pixels[frame] = np.asarray(fallback_pixels)[frame]
            fallback[frame] = True
        else:
            raise ValueError(f"No observed palm keypoints and no fallback at frame {frame}")
    if not np.isfinite(pixels).all():
        raise ValueError("Observed palm pixels contain unresolved invalid frames")
    window = max(1, int(smooth_window))
    if window > 1:
        smoothed = pixels.copy()
        radius = window // 2
        for frame in range(points.shape[0]):
            start = max(0, frame - radius)
            end = min(points.shape[0], frame + radius + 1)
            smoothed[frame] = np.median(pixels[start:end], axis=0)
        pixels = smoothed
    return pixels, fallback, indices


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


def approach_window_from_fps(
    fps: float,
    required_hand_displacement: float = 0.0,
    *,
    max_hand_speed_mps: float = 0.4,
    min_approach_frames: int = 20,
    max_approach_frames: int = 60,
) -> int:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if max_hand_speed_mps <= 0:
        raise ValueError("max_hand_speed_mps must be positive")
    if min_approach_frames < 1 or max_approach_frames < min_approach_frames:
        raise ValueError("invalid approach frame bounds")
    required = max(0.0, float(required_hand_displacement))
    frames = int(np.ceil(required / float(max_hand_speed_mps) * float(fps)))
    return int(np.clip(max(frames, int(min_approach_frames)), 1, int(max_approach_frames)))


def smoothstep_ramp(frame_num: int, t_move: int, window: int, *, device=None, dtype=None):
    if not 1 <= int(t_move) < int(frame_num):
        raise ValueError("t_move must lie inside the sequence")
    window = int(np.clip(window, 1, t_move))
    frames = torch.arange(frame_num, device=device, dtype=dtype or torch.float32)
    u = ((frames - (t_move - window)) / float(window)).clamp(0.0, 1.0)
    return 10.0 * u.pow(3) - 15.0 * u.pow(4) + 6.0 * u.pow(5)


def camera_ray_hand_targets(
    initial_hand_cam: torch.Tensor,
    detached_surface_depth: torch.Tensor,
    t_move: int,
    window: int,
    *,
    target_distance: float = 0.005,
    query_pixels: torch.Tensor | None = None,
    camera_intrinsics: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move depth along the initial OpenCV ray without changing image coordinates."""
    if initial_hand_cam.ndim != 2 or initial_hand_cam.shape[1] != 3:
        raise ValueError("initial_hand_cam must have shape [T,3]")
    if detached_surface_depth.shape != (initial_hand_cam.shape[0],):
        raise ValueError("detached_surface_depth must have shape [T]")
    if torch.any(initial_hand_cam[:, 2] <= 0) or torch.any(detached_surface_depth <= 0):
        raise ValueError("Ray IK requires positive OpenCV camera depth")
    surface_depth = detached_surface_depth.detach()
    if query_pixels is not None:
        if camera_intrinsics is None:
            raise ValueError("camera_intrinsics is required with observed query_pixels")
        K = camera_intrinsics
        if K.shape == (3, 3):
            K = K[None].expand(initial_hand_cam.shape[0], -1, -1)
        if K.shape != (initial_hand_cam.shape[0], 3, 3):
            raise ValueError("camera_intrinsics must be [3,3] or [T,3,3]")
        ray = torch.stack(
            [
                (query_pixels[:, 0] - K[:, 0, 2]) / K[:, 0, 0],
                (query_pixels[:, 1] - K[:, 1, 2]) / K[:, 1, 1],
                torch.ones_like(query_pixels[:, 0]),
            ],
            dim=1,
        )
    else:
        ray = initial_hand_cam / initial_hand_cam[:, 2:3]
    side = torch.where(
        initial_hand_cam[:, 2] - surface_depth >= 0.0,
        torch.ones_like(surface_depth),
        -torch.ones_like(surface_depth),
    )
    target_z = (surface_depth + side * float(target_distance)).clamp_min(1e-4)
    ramp = smoothstep_ramp(
        initial_hand_cam.shape[0],
        t_move,
        window,
        device=initial_hand_cam.device,
        dtype=initial_hand_cam.dtype,
    )
    # A per-frame surface target can jump when the image ray crosses from one
    # mesh side to another.  The pre-contact path therefore uses the single
    # physical endpoint at t_move; moving frames follow their own object
    # surface after contact.
    contact_target_z = target_z[int(t_move)].detach()
    z = initial_hand_cam[:, 2] + ramp * (contact_target_z - initial_hand_cam[:, 2])
    frames = torch.arange(initial_hand_cam.shape[0], device=z.device)
    z = torch.where(frames > int(t_move), target_z, z)
    return ray * z[:, None], ramp


def mesh_surface_depth_at_pixels(
    object_vertices_cam: torch.Tensor,
    query_pixels: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    *,
    object_faces: torch.Tensor | None = None,
    current_hand_depth: torch.Tensor | None = None,
    top_k: int = 32,
    max_fallback_pixel_distance: float = 15.0,
    return_normals: bool = False,
    strict_frames: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return real ray/triangle surface depth with a nearest-surface fallback.

    The ray uses OpenCV coordinates with camera origin at zero and z-normalized
    direction.  If multiple triangles are hit, the hit nearest the current
    hand depth is selected so front/back surfaces are not mixed.  The optional
    boolean return records frames that needed the projected-vertex fallback.
    """
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
    if current_hand_depth is None:
        current_hand_depth = torch.full(
            (frame_num,), float("nan"), device=vertices.device, dtype=vertices.dtype
        )
    current_hand_depth = current_hand_depth.detach().reshape(frame_num)
    if object_faces is not None:
        faces = object_faces.detach().long()
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("object_faces must have shape [F,3]")
    else:
        faces = None

    surface_z = torch.empty(frame_num, device=vertices.device, dtype=vertices.dtype)
    surface_normal = torch.empty((frame_num, 3), device=vertices.device, dtype=vertices.dtype)
    fallback = torch.zeros(frame_num, device=vertices.device, dtype=torch.bool)
    for frame in range(frame_num):
        K = camera_intrinsics[frame]
        ray = torch.stack(
            [
                (query_pixels[frame, 0] - K[0, 2]) / K[0, 0],
                (query_pixels[frame, 1] - K[1, 2]) / K[1, 1],
                torch.ones((), device=vertices.device, dtype=vertices.dtype),
            ]
        )
        hits = None
        if faces is not None and faces.numel():
            triangles = vertices[frame][faces]
            v0, v1, v2 = triangles.unbind(dim=1)
            edge1 = v1 - v0
            edge2 = v2 - v0
            h = torch.cross(ray.expand_as(edge2), edge2, dim=1)
            a = (edge1 * h).sum(dim=1)
            valid_a = a.abs() > 1e-8
            inv_a = torch.where(valid_a, 1.0 / a, torch.zeros_like(a))
            s = -v0
            bary_u = inv_a * (s * h).sum(dim=1)
            q = torch.cross(s, edge1, dim=1)
            bary_v = inv_a * (ray.expand_as(q) * q).sum(dim=1)
            t = inv_a * (edge2 * q).sum(dim=1)
            valid = (
                valid_a
                & (bary_u >= 0.0)
                & (bary_u <= 1.0)
                & (bary_v >= 0.0)
                & (bary_u + bary_v <= 1.0)
                & (t > 1e-6)
            )
            if bool(valid.any()):
                hits = t[valid]
                hit_faces = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if hits is not None and hits.numel():
            hand_z = current_hand_depth[frame]
            if torch.isfinite(hand_z):
                selected = (hits - hand_z).abs().argmin()
            else:
                selected = hits.argmin()
            surface_z[frame] = hits[selected]
            tri = triangles[hit_faces[selected]]
            surface_normal[frame] = torch.nn.functional.normalize(
                torch.cross(tri[1] - tri[0], tri[2] - tri[0], dim=0), dim=0, eps=1e-8
            )
            continue

        fallback[frame] = True
        frame_vertices = vertices[frame]
        projected = torch.stack(
            [
                K[0, 0] * frame_vertices[:, 0] / frame_vertices[:, 2] + K[0, 2],
                K[1, 1] * frame_vertices[:, 1] / frame_vertices[:, 2] + K[1, 2],
            ],
            dim=1,
        )
        pixel_distance = torch.linalg.norm(projected - query_pixels[frame], dim=1)
        nearest_pixel_distance = float(pixel_distance.min().detach())
        strict = True if strict_frames is None else bool(strict_frames[frame])
        if strict and nearest_pixel_distance > float(max_fallback_pixel_distance):
            raise ValueError(
                "Palm ray missed the object surface and nearest projected mesh "
                f"point is {nearest_pixel_distance:.2f}px away, exceeding the "
                f"{float(max_fallback_pixel_distance):.2f}px fallback limit at frame {frame}"
            )
        count = min(max(1, int(top_k)), vertex_num)
        nearest = torch.topk(pixel_distance, count, largest=False).indices
        candidates = frame_vertices[nearest, 2]
        hand_z = current_hand_depth[frame]
        if torch.isfinite(hand_z):
            surface_z[frame] = candidates[(candidates - hand_z).abs().argmin()]
        else:
            surface_z[frame] = candidates[0]
        if faces is None:
            surface_normal[frame] = torch.tensor(
                [0.0, 0.0, 1.0], device=vertices.device, dtype=vertices.dtype
            )
        else:
            nearest_vertex = nearest[0]
            incident = torch.nonzero((faces == nearest_vertex).any(dim=1), as_tuple=False).squeeze(1)
            if incident.numel() == 0:
                raise ValueError(f"No incident face for fallback vertex at frame {frame}")
            tri = frame_vertices[faces[incident[0]]]
            surface_normal[frame] = torch.nn.functional.normalize(
                torch.cross(tri[1] - tri[0], tri[2] - tri[0], dim=0), dim=0, eps=1e-8
            )
    if torch.any(surface_z <= 0) or not torch.isfinite(surface_z).all():
        raise ValueError("Mesh surface lookup produced invalid OpenCV camera depth")
    if object_faces is not None:
        if return_normals:
            return surface_z.detach(), fallback.detach(), surface_normal.detach()
        return surface_z.detach(), fallback.detach()
    return surface_z.detach()


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
