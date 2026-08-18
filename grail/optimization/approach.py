"""Low-dimensional human pre-contact approach utilities."""

from __future__ import annotations

import torch


def minimum_jerk_ramp(
    frame_num: int,
    contact_frame: int,
    approach_window: int,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    frame_num = int(frame_num)
    contact_frame = int(contact_frame)
    approach_window = int(approach_window)
    if frame_num < 1:
        raise ValueError(f"frame_num must be positive, got {frame_num}")
    if contact_frame < 0 or contact_frame >= frame_num:
        raise ValueError(
            f"contact_frame must be in [0,{frame_num - 1}], got {contact_frame}"
        )
    if approach_window < 1:
        raise ValueError(f"approach_window must be positive, got {approach_window}")

    frames = torch.arange(frame_num, device=device, dtype=dtype)
    start = float(contact_frame - approach_window)
    u = ((frames - start) / float(approach_window)).clamp(0.0, 1.0)
    return 10.0 * u.pow(3) - 15.0 * u.pow(4) + 6.0 * u.pow(5)


def smoothstep_approach_ramp(
    frame_num: int,
    contact_frame: int,
    approach_window: int,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Compatibility name for the shared minimum-jerk approach ramp."""
    return minimum_jerk_ramp(
        frame_num,
        contact_frame,
        approach_window,
        device=device,
        dtype=dtype,
    )


def ground_approach_direction(
    object_center: torch.Tensor,
    human_root: torch.Tensor,
    gravity_axis: str | int = "z",
) -> torch.Tensor:
    axis_lookup = {"x": 0, "y": 1, "z": 2}
    axis = axis_lookup.get(gravity_axis, gravity_axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"gravity_axis must be x/y/z or 0/1/2, got {gravity_axis!r}")
    direction = object_center - human_root
    direction = direction.clone()
    direction[int(axis)] = 0.0
    norm = torch.linalg.norm(direction)
    if not torch.isfinite(norm) or float(norm.detach()) < 1e-8:
        raise ValueError("Cannot define ground approach direction from coincident centers")
    return direction / norm


def approach_offsets(
    ramp: torch.Tensor,
    distance: torch.Tensor,
    direction: torch.Tensor,
    *,
    max_distance: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_distance = float(max_distance)
    if max_distance <= 0:
        raise ValueError(f"max_distance must be positive, got {max_distance}")
    constrained = distance.reshape(()).clamp(0.0, max_distance)
    offsets = ramp.reshape(-1, 1) * constrained * direction.reshape(1, 3)
    return offsets, constrained


def hand_to_mesh_surface_distance(
    hand_points: torch.Tensor,
    object_vertices: torch.Tensor,
    object_faces: torch.Tensor,
    *,
    top_k: int = 32,
    candidate_faces: int = 64,
) -> torch.Tensor:
    """Mean distance of closest hand vertices to exact candidate triangles.

    Candidate lookup uses face centroids, then distance is evaluated against the
    triangle surface (interior plane projection plus all three edge segments).
    This avoids the packed PyTorch3D CUDA kernel, which can return false zeros for
    large meshes on some builds.
    """
    from pytorch3d.ops import knn_points

    if hand_points.ndim != 2 or hand_points.shape[1] != 3:
        raise ValueError(f"hand_points must be [N,3], got {tuple(hand_points.shape)}")
    if object_vertices.ndim != 2 or object_vertices.shape[1] != 3:
        raise ValueError(
            f"object_vertices must be [V,3], got {tuple(object_vertices.shape)}"
        )
    if object_faces.ndim != 2 or object_faces.shape[1] != 3:
        raise ValueError(f"object_faces must be [F,3], got {tuple(object_faces.shape)}")
    if hand_points.shape[0] == 0 or object_faces.shape[0] == 0:
        raise ValueError("Hand points and object faces must be non-empty")

    points = hand_points.float().contiguous()
    triangles = object_vertices[object_faces.long()].float().contiguous()
    face_count = int(triangles.shape[0])
    candidate_count = min(max(1, int(candidate_faces)), face_count)
    centroids = triangles.mean(dim=1)
    candidate_idx = knn_points(
        points.unsqueeze(0), centroids.unsqueeze(0), K=candidate_count
    ).idx[0]
    candidate = triangles[candidate_idx]
    p = points[:, None, :]
    a, b, c = candidate.unbind(dim=2)

    def segment_distance_sq(start, end):
        edge = end - start
        alpha = ((p - start) * edge).sum(dim=-1) / edge.square().sum(dim=-1).clamp_min(1e-12)
        closest = start + alpha.clamp(0.0, 1.0)[..., None] * edge
        return (p - closest).square().sum(dim=-1)

    ab = b - a
    ac = c - a
    normal = torch.cross(ab, ac, dim=-1)
    normal_sq_raw = normal.square().sum(dim=-1)
    normal_sq = normal_sq_raw.clamp_min(1e-12)
    signed_numerator = ((p - a) * normal).sum(dim=-1)
    projected = p - (signed_numerator / normal_sq)[..., None] * normal

    v0 = ab
    v1 = ac
    v2 = projected - a
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
        (normal_sq_raw > 1e-12)
        & (denom_raw > 1e-12)
        & (bary_u >= 0.0)
        & (bary_v >= 0.0)
        & (bary_w >= 0.0)
    )
    plane_distance_sq = signed_numerator.square() / normal_sq
    edge_distance_sq = torch.minimum(
        segment_distance_sq(a, b),
        torch.minimum(segment_distance_sq(b, c), segment_distance_sq(c, a)),
    )
    squared = torch.where(inside, plane_distance_sq, edge_distance_sq).min(dim=1).values
    distances = squared.clamp_min(0.0).sqrt()
    keep = min(max(1, int(top_k)), int(distances.numel()))
    return distances.topk(keep, largest=False).values.mean()
