"""Offline object motion-onset detection for Lift4D depth supervision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter


@dataclass(frozen=True)
class ObjectMotionState:
    detection_center_cam: np.ndarray
    motion_score_3d: np.ndarray
    motion_score_mask: np.ndarray
    motion_score: np.ndarray
    moving: np.ndarray
    move_start_frame: int
    confidence: float
    static_z: float
    z_target: np.ndarray


def resolve_contact_hint(
    explicit_frame: int | None,
    contact_start_idx: int | None,
    inter_start_idx: int,
    frame_num: int,
    *,
    explicit_source: str = "cli",
) -> tuple[int, str]:
    if explicit_frame is not None:
        hint, source = int(explicit_frame), explicit_source
    elif isinstance(contact_start_idx, (int, np.integer)) and 0 <= int(contact_start_idx) < frame_num:
        hint, source = int(contact_start_idx), "cache"
    else:
        hint, source = int(inter_start_idx), "inter_start"
    if not 0 <= hint < frame_num:
        raise ValueError(f"contact hint must be in [0,{frame_num - 1}], got {hint}")
    if source not in ("cli", "cache", "inter_start"):
        raise ValueError(f"Invalid contact hint source {source!r}")
    return hint, source


def infer_contact_hand(
    configured: str, contact_labels: list, *, fallback: str = "right"
) -> str:
    configured = str(configured).lower()
    if configured in ("left", "right", "both"):
        return configured
    if configured != "auto":
        raise ValueError(f"contact.hand must be auto/left/right/both, got {configured!r}")
    for labels in contact_labels:
        labels = labels or []
        has_left = "L_Hand" in labels
        has_right = "R_Hand" in labels
        if has_left and has_right:
            return "both"
        if has_left:
            return "left"
        if has_right:
            return "right"
    fallback = str(fallback).lower()
    if fallback not in ("left", "right", "both"):
        raise ValueError(f"contact.hand_fallback must be left/right/both, got {fallback!r}")
    return fallback


def _odd_window(value: int, frame_num: int) -> int:
    value = int(value)
    if value < 1 or value % 2 == 0:
        raise ValueError(f"detection_median_window must be a positive odd integer, got {value}")
    largest = frame_num if frame_num % 2 else frame_num - 1
    return min(value, largest)


def _robust_scale(values: np.ndarray, floor: float = 1e-6) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, floor
    center = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - center)))
    return center, max(scale, floor)


def _rolling_forward_sum(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(values)
    for frame in range(values.size):
        result[frame] = values[frame : min(values.size, frame + window)].sum()
    return result


def _mask_signals(masks: np.ndarray | list, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    masks = np.asarray(masks).astype(bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"object masks must have shape [T,H,W], got {masks.shape}")
    frame_num, height, width = masks.shape
    centroids = np.full((frame_num, 2), np.nan, dtype=np.float64)
    areas = masks.reshape(frame_num, -1).sum(axis=1).astype(np.float64)
    for frame, mask in enumerate(masks):
        y, x = np.nonzero(mask)
        if x.size < 4:
            raise ValueError(f"object mask has fewer than four pixels at frame {frame}")
        centroids[frame] = [x.mean() / max(width, 1), y.mean() / max(height, 1)]

    centroid_motion = np.zeros(frame_num, dtype=np.float64)
    shape_motion = np.zeros(frame_num, dtype=np.float64)
    for frame in range(frame_num):
        end = min(frame_num - 1, frame + horizon)
        centroid_motion[frame] = np.linalg.norm(centroids[end] - centroids[frame])
        union = np.logical_or(masks[frame], masks[end]).sum()
        intersection = np.logical_and(masks[frame], masks[end]).sum()
        iou_change = 1.0 - (intersection / max(union, 1))
        area_change = abs(areas[end] - areas[frame]) / max(areas[frame], 1.0)
        shape_motion[frame] = iou_change + area_change
    return centroid_motion, shape_motion


def build_static_relative_depth_target(
    z: np.ndarray, move_start_frame: int, transition_frames: int = 4
) -> tuple[float, np.ndarray]:
    """Lock the pre-motion target and retain only post-onset Lift4D relative depth."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    frame_num = z.size
    move_start_frame = int(move_start_frame)
    if not 1 <= move_start_frame < frame_num:
        raise ValueError(
            f"move_start_frame must be in [1,{frame_num - 1}], got {move_start_frame}"
        )
    if not np.isfinite(z).all():
        raise ValueError("Lift4D z contains NaN or Inf")
    static_z = float(np.median(z[:move_start_frame]))
    relative = static_z + z - z[move_start_frame]
    target = relative.copy()
    target[:move_start_frame] = static_z

    transition_frames = max(0, int(transition_frames))
    if transition_frames:
        end = min(frame_num, move_start_frame + transition_frames)
        u = np.linspace(0.0, 1.0, end - move_start_frame, dtype=np.float64)
        weight = u * u * (3.0 - 2.0 * u)
        target[move_start_frame:end] = (
            (1.0 - weight) * static_z + weight * relative[move_start_frame:end]
        )
    return static_z, target.astype(np.float32)


def detect_object_motion(
    center_cam_raw: np.ndarray,
    object_masks: np.ndarray | list,
    *,
    smoothed_z: np.ndarray | None = None,
    contact_hint: int | None = None,
    config: dict | None = None,
) -> ObjectMotionState:
    """Detect the first persistent static-to-moving transition without SG leakage."""
    cfg = dict(config or {})
    centers = np.asarray(center_cam_raw, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] < 12:
        raise ValueError(f"center_cam_raw must be [T,3] with T>=12, got {centers.shape}")
    if not np.isfinite(centers).all() or np.any(centers[:, 2] <= 0):
        raise ValueError("center_cam_raw must contain finite positive-depth OpenCV points")
    frame_num = centers.shape[0]
    if contact_hint is not None and not 0 <= int(contact_hint) < frame_num:
        raise ValueError(f"contact_hint must be in [0,{frame_num - 1}], got {contact_hint}")

    detection_window = _odd_window(cfg.get("detection_median_window", 5), frame_num)
    detection = median_filter(centers, size=(detection_window, 1), mode="nearest")
    vote_window = int(cfg.get("vote_window", 5))
    min_votes = int(cfg.get("min_votes", 3))
    persistence_window = int(cfg.get("persistence_window", 7))
    min_persistence = int(cfg.get("min_persistence", 5))
    threshold = float(cfg.get("motion_score_threshold", 3.0))
    baseline_frames = int(cfg.get("baseline_frames", 15))
    if contact_hint is not None:
        # Contact may precede lift-off by many frames. Use the known pre-contact
        # interval to estimate stationary noise, but keep onset search global so
        # a genuinely earlier physical transition can still be detected.
        baseline_frames = max(baseline_frames, int(contact_hint))
    baseline_frames = min(baseline_frames, frame_num - 1)
    if vote_window < 1 or not 1 <= min_votes <= vote_window:
        raise ValueError("min_votes must be in [1, vote_window]")
    if persistence_window < 1 or not 1 <= min_persistence <= persistence_window:
        raise ValueError("min_persistence must be in [1, persistence_window]")

    step_3d = np.linalg.norm(np.diff(detection, axis=0, prepend=detection[:1]), axis=1)
    cumulative_3d = _rolling_forward_sum(step_3d, vote_window)
    centroid_motion, shape_motion = _mask_signals(object_masks, vote_window)

    first_indices = np.arange(baseline_frames)
    quiet_fraction = float(cfg.get("fallback_quiet_fraction", 0.25))
    if not 0.2 <= quiet_fraction <= 0.3:
        raise ValueError("fallback_quiet_fraction must be in [0.2,0.3]")
    scales = np.array(
        [
            max(float(np.median(cumulative_3d)), 1e-6),
            max(float(np.median(centroid_motion)), 1e-6),
            max(float(np.median(shape_motion)), 1e-6),
        ]
    )
    raw_activity = (
        cumulative_3d / scales[0]
        + centroid_motion / scales[1]
        + shape_motion / scales[2]
    )
    quiet_cut = float(np.quantile(raw_activity, quiet_fraction))
    first_baseline_is_quiet = float(np.median(raw_activity[first_indices])) <= max(
        1e-6, 1.5 * quiet_cut
    )
    if first_baseline_is_quiet:
        baseline_indices = first_indices
    else:
        quiet_count = max(3, int(round(frame_num * quiet_fraction)))
        baseline_indices = np.argsort(raw_activity, kind="stable")[:quiet_count]

    base3, scale3 = _robust_scale(cumulative_3d[baseline_indices])
    base_centroid, scale_centroid = _robust_scale(centroid_motion[baseline_indices])
    base_shape, scale_shape = _robust_scale(shape_motion[baseline_indices])
    score_3d = np.maximum(0.0, (cumulative_3d - base3) / scale3)
    score_centroid = np.maximum(0.0, (centroid_motion - base_centroid) / scale_centroid)
    score_shape = np.maximum(0.0, (shape_motion - base_shape) / scale_shape)
    score_mask = (2.0 / 3.0) * score_centroid + (1.0 / 3.0) * score_shape
    score = 0.70 * score_3d + 0.20 * score_centroid + 0.10 * score_shape

    static_center = np.median(detection[baseline_indices], axis=0)
    baseline_radius = np.linalg.norm(detection[baseline_indices] - static_center, axis=1)
    radius_center, radius_scale = _robust_scale(baseline_radius)
    departure = np.linalg.norm(detection - static_center, axis=1)
    departed = departure > (radius_center + threshold * radius_scale)
    above = score > threshold

    move_start = None
    vote_fraction = 0.0
    persistence_fraction = 0.0
    search_end = frame_num - persistence_window + 1
    for candidate in range(1, max(1, search_end)):
        votes = int(above[candidate : min(frame_num, candidate + vote_window)].sum())
        persistence = int(
            departed[candidate : min(frame_num, candidate + persistence_window)].sum()
        )
        if votes >= min_votes and persistence >= min_persistence:
            move_start = candidate
            vote_fraction = votes / vote_window
            persistence_fraction = persistence / persistence_window
            break

    if move_start is None:
        confidence = 0.0
        action = str(cfg.get("low_confidence_action", "error")).lower()
        if action == "error":
            raise ValueError(
                "Low-confidence object motion onset: no persistent static-to-moving "
                f"transition found; max_score={float(score.max()):.4f}"
            )
        move_start = frame_num - 1
    else:
        margin = min(1.0, max(0.0, float(score[move_start] / max(threshold, 1e-6) - 1.0)))
        confidence = 0.4 * vote_fraction + 0.4 * persistence_fraction + 0.2 * margin

    min_confidence = float(cfg.get("min_confidence", 0.55))
    if confidence < min_confidence and str(cfg.get("low_confidence_action", "error")).lower() == "error":
        raise ValueError(
            f"Low-confidence object motion onset at frame {move_start}: "
            f"confidence={confidence:.4f} < {min_confidence:.4f}"
        )

    target_source_z = detection[:, 2] if smoothed_z is None else np.asarray(smoothed_z)
    if target_source_z.shape != (frame_num,):
        raise ValueError(f"smoothed_z must have shape [{frame_num}], got {target_source_z.shape}")
    static_z, z_target = build_static_relative_depth_target(
        target_source_z, move_start, cfg.get("transition_frames", 4)
    )
    moving = np.zeros(frame_num, dtype=bool)
    if cfg.get("latch_moving", True):
        moving[move_start:] = True
    else:
        moving[move_start:] = above[move_start:]
    return ObjectMotionState(
        detection_center_cam=detection.astype(np.float32),
        motion_score_3d=score_3d.astype(np.float32),
        motion_score_mask=score_mask.astype(np.float32),
        motion_score=score.astype(np.float32),
        moving=moving,
        move_start_frame=int(move_start),
        confidence=float(confidence),
        static_z=static_z,
        z_target=z_target,
    )
