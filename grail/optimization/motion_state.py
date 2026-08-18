"""Mask-first object motion-state detection for formal Lift4D supervision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter


@dataclass(frozen=True)
class ObjectMotionState:
    detection_center_cam: np.ndarray
    lift4d_center_speed: np.ndarray
    mask_iou_drop: np.ndarray
    mask_centroid_displacement_px: np.ndarray
    mask_area_change_ratio: np.ndarray
    motion_score_3d: np.ndarray
    motion_score_mask: np.ndarray
    motion_score: np.ndarray
    moving_evidence: np.ndarray
    moving: np.ndarray
    static: np.ndarray
    move_start_frame: int
    confidence: float
    static_z: float
    z_target: np.ndarray
    thresholds: dict[str, float]


def resolve_contact_hint(
    explicit_frame: int | None,
    contact_start_idx: int | None,
    inter_start_idx: int,
    frame_num: int,
    *,
    explicit_source: str = "cli",
) -> tuple[int, str]:
    """Legacy-only contact hint resolver.

    Formal mask-motion/ray-IK code must not call this function.
    """
    if explicit_frame is not None:
        hint, source = int(explicit_frame), explicit_source
    elif isinstance(contact_start_idx, (int, np.integer)) and 0 <= int(contact_start_idx) < frame_num:
        hint, source = int(contact_start_idx), "cache"
    else:
        hint, source = int(inter_start_idx), "inter_start"
    if not 0 <= hint < frame_num:
        raise ValueError(f"contact hint must be in [0,{frame_num - 1}], got {hint}")
    return hint, source


def infer_contact_hand(
    configured: str, contact_labels: list, *, fallback: str = "right"
) -> str:
    """Legacy-only label-based hand resolver."""
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


def _robust_threshold(values: np.ndarray, floor: float, mad_scale: float) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot estimate a motion threshold from empty/non-finite values")
    median = float(np.median(finite))
    mad = float(1.4826 * np.median(np.abs(finite - median)))
    return max(float(floor), median + float(mad_scale) * max(mad, 1e-9)), median, mad


def _normalize_signal(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) / max(float(threshold), 1e-9)


def _adjacent_mask_signals(masks: np.ndarray | list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = np.asarray(masks).astype(bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"object masks must have shape [T,H,W], got {masks.shape}")
    frame_num = masks.shape[0]
    centroids = np.empty((frame_num, 2), dtype=np.float64)
    areas = masks.reshape(frame_num, -1).sum(axis=1).astype(np.float64)
    for frame, mask in enumerate(masks):
        y, x = np.nonzero(mask)
        if x.size < 4:
            raise ValueError(f"object mask has fewer than four pixels at frame {frame}")
        centroids[frame] = (float(x.mean()), float(y.mean()))

    iou_drop = np.zeros(frame_num, dtype=np.float64)
    centroid_disp = np.zeros(frame_num, dtype=np.float64)
    area_change = np.zeros(frame_num, dtype=np.float64)
    for frame in range(1, frame_num):
        previous = masks[frame - 1]
        current = masks[frame]
        union = np.logical_or(previous, current).sum()
        intersection = np.logical_and(previous, current).sum()
        iou_drop[frame] = 1.0 - intersection / max(int(union), 1)
        centroid_disp[frame] = np.linalg.norm(centroids[frame] - centroids[frame - 1])
        area_change[frame] = abs(areas[frame] - areas[frame - 1]) / max(areas[frame - 1], 1.0)
    return iou_drop, centroid_disp, area_change


def build_static_relative_depth_target(
    z: np.ndarray, move_start_frame: int, transition_frames: int = 0
) -> tuple[float, np.ndarray]:
    """Hard-lock static frames, then retain Lift4D motion relative to onset."""
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
    """Detect pickup onset from adjacent masks; GPT/contact hints are ignored."""
    del contact_hint
    cfg = dict(config or {})
    centers = np.asarray(center_cam_raw, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] < 16:
        raise ValueError(f"center_cam_raw must be [T,3] with T>=16, got {centers.shape}")
    if not np.isfinite(centers).all() or np.any(centers[:, 2] <= 0):
        raise ValueError("center_cam_raw must contain finite positive-depth OpenCV points")
    frame_num = centers.shape[0]
    detection_window = _odd_window(cfg.get("detection_median_window", 5), frame_num)
    detection = median_filter(centers, size=(detection_window, 1), mode="nearest")
    lift4d_speed = np.linalg.norm(
        np.diff(detection, axis=0, prepend=detection[:1]), axis=1
    )
    iou_drop, centroid_disp, area_change = _adjacent_mask_signals(object_masks)
    if iou_drop.shape != (frame_num,):
        raise ValueError(
            f"object mask frame count {iou_drop.size} does not match Lift4D {frame_num}"
        )

    baseline_frames = int(cfg.get("baseline_frames", 15))
    if not 5 <= baseline_frames < frame_num:
        raise ValueError(f"baseline_frames must be in [5,{frame_num - 1}], got {baseline_frames}")
    baseline = slice(1, baseline_frames)
    mad_scale = float(cfg.get("threshold_mad_scale", 4.0))
    iou_threshold, iou_median, iou_mad = _robust_threshold(
        iou_drop[baseline], cfg.get("iou_drop_floor", 0.03), mad_scale
    )
    centroid_threshold, centroid_median, centroid_mad = _robust_threshold(
        centroid_disp[baseline], cfg.get("centroid_displacement_floor_px", 2.0), mad_scale
    )
    area_threshold, area_median, area_mad = _robust_threshold(
        area_change[baseline], cfg.get("area_change_floor", 0.02), mad_scale
    )
    speed_threshold, speed_median, speed_mad = _robust_threshold(
        lift4d_speed[baseline], cfg.get("lift4d_speed_floor_m", 0.002), mad_scale
    )
    strong_iou_threshold = max(
        float(cfg.get("strong_iou_drop_floor", 0.20)),
        float(cfg.get("strong_iou_threshold_scale", 2.0)) * iou_threshold,
    )

    mask_motion = (
        (iou_drop > iou_threshold)
        | (centroid_disp > centroid_threshold)
        | (area_change > area_threshold)
    )
    strong_mask_motion = iou_drop > strong_iou_threshold
    lift4d_motion = lift4d_speed > speed_threshold
    evidence = strong_mask_motion | (mask_motion & lift4d_motion)
    evidence[:baseline_frames] = False

    vote_window = int(cfg.get("vote_window", 5))
    min_votes = int(cfg.get("min_votes", 3))
    if vote_window < 1 or not 1 <= min_votes <= vote_window:
        raise ValueError("min_votes must be in [1, vote_window]")
    move_start = None
    vote_fraction = 0.0
    for end in range(baseline_frames, frame_num):
        start = max(baseline_frames, end - vote_window + 1)
        window_indices = np.arange(start, end + 1)
        active = window_indices[evidence[window_indices]]
        if active.size >= min_votes:
            move_start = int(active[0])
            vote_fraction = float(active.size / vote_window)
            break
    thresholds = {
        "iou_drop": iou_threshold,
        "centroid_displacement_px": centroid_threshold,
        "area_change_ratio": area_threshold,
        "lift4d_center_speed_m": speed_threshold,
        "strong_iou_drop": strong_iou_threshold,
        "baseline_iou_drop_median": iou_median,
        "baseline_iou_drop_mad": iou_mad,
        "baseline_centroid_median_px": centroid_median,
        "baseline_centroid_mad_px": centroid_mad,
        "baseline_area_change_median": area_median,
        "baseline_area_change_mad": area_mad,
        "baseline_lift4d_speed_median_m": speed_median,
        "baseline_lift4d_speed_mad_m": speed_mad,
    }
    if move_start is None:
        summary = ", ".join(f"{key}={value:.6g}" for key, value in thresholds.items())
        raise ValueError(
            "Low-confidence object motion onset: no reliable 3/5 adjacent-frame "
            f"pickup evidence; {summary}"
        )

    confidence = min(1.0, vote_fraction + 0.2 * float(strong_mask_motion[move_start]))
    min_confidence = float(cfg.get("min_confidence", 0.55))
    if confidence < min_confidence:
        raise ValueError(
            f"Low-confidence object motion onset at frame {move_start}: "
            f"confidence={confidence:.4f} < {min_confidence:.4f}"
        )
    target_source_z = detection[:, 2] if smoothed_z is None else np.asarray(smoothed_z)
    if target_source_z.shape != (frame_num,):
        raise ValueError(f"smoothed_z must have shape [{frame_num}], got {target_source_z.shape}")
    static_z, z_target = build_static_relative_depth_target(
        target_source_z, move_start, cfg.get("transition_frames", 0)
    )

    moving = np.zeros(frame_num, dtype=bool)
    moving[move_start:] = True
    static = ~moving
    score_3d = _normalize_signal(lift4d_speed, speed_threshold)
    score_mask = np.maximum.reduce(
        [
            _normalize_signal(iou_drop, iou_threshold),
            _normalize_signal(centroid_disp, centroid_threshold),
            _normalize_signal(area_change, area_threshold),
        ]
    )
    score = np.maximum(score_mask, score_3d)
    return ObjectMotionState(
        detection_center_cam=detection.astype(np.float32),
        lift4d_center_speed=lift4d_speed.astype(np.float32),
        mask_iou_drop=iou_drop.astype(np.float32),
        mask_centroid_displacement_px=centroid_disp.astype(np.float32),
        mask_area_change_ratio=area_change.astype(np.float32),
        motion_score_3d=score_3d.astype(np.float32),
        motion_score_mask=score_mask.astype(np.float32),
        motion_score=score.astype(np.float32),
        moving_evidence=evidence,
        moving=moving,
        static=static,
        move_start_frame=move_start,
        confidence=confidence,
        static_z=static_z,
        z_target=z_target,
        thresholds=thresholds,
    )
