"""Lift4D point-trajectory depth prior for the real GRAIL optimizer.

This adapter deliberately ignores every rigid-pose field.  Lift4D contributes
only a robust OpenCV-camera-space center depth trajectory and per-frame support
weights derived from tracked Gaussian points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


REQUIRED_DEPTH_KEYS = (
    "frame_indices",
    "point_trajectories_cam",
    "canonical_points",
    "point_visibility",
    "point_fit_inliers",
    "point_opacity",
    "valid_point_count",
    "camera_intrinsics",
    "camera_convention",
)
EXPECTED_CAMERA_CONVENTIONS = {"opencv", "opencv_camera"}


@dataclass(frozen=True)
class Lift4DDepthPrior:
    frame_indices: np.ndarray
    prior_used: np.ndarray
    center_cam_raw: np.ndarray
    center_cam: np.ndarray
    z_raw: np.ndarray
    z: np.ndarray
    delta_z: np.ndarray
    frame_weight: np.ndarray
    valid_point_count: np.ndarray
    stable_point_ids: np.ndarray
    stable_point_score: np.ndarray
    camera_intrinsics: np.ndarray
    source_path: str
    camera_convention: str
    diagnostics: dict


def _odd_window(window: int, frame_num: int, *, minimum: int = 3) -> int:
    window = int(window)
    if window < minimum:
        raise ValueError(f"Smoothing window must be >= {minimum}, got {window}")
    if window % 2 == 0:
        raise ValueError(f"Smoothing window must be odd, got {window}")
    largest = frame_num if frame_num % 2 == 1 else frame_num - 1
    return min(window, largest)


def _local_median_filter(centers: np.ndarray, window: int = 7) -> np.ndarray:
    return median_filter(centers, size=(window, 1), mode="nearest")


def project_opencv_translation(translation_cam: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    translation_cam = np.asarray(translation_cam, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if translation_cam.ndim != 2 or translation_cam.shape[1] != 3:
        raise ValueError(f"translation_cam must be [T,3], got {translation_cam.shape}")
    if intrinsics.shape != (translation_cam.shape[0], 3, 3):
        raise ValueError(
            f"intrinsics must be [T,3,3] for {translation_cam.shape[0]} frames, got {intrinsics.shape}"
        )
    if np.any(translation_cam[:, 2] <= 0):
        raise ValueError("OpenCV camera projection requires z > 0")
    x = translation_cam[:, 0] / translation_cam[:, 2]
    y = translation_cam[:, 1] / translation_cam[:, 2]
    return np.stack(
        [
            intrinsics[:, 0, 0] * x + intrinsics[:, 0, 2],
            intrinsics[:, 1, 1] * y + intrinsics[:, 1, 2],
        ],
        axis=1,
    )


def load_lift4d_depth_prior(
    path: str | Path,
    *,
    frame_num: int,
    median_window: int = 7,
    smooth_window: int = 31,
    savgol_polyorder: int = 2,
    stable_point_count: int = 2500,
    min_stable_points: int = 64,
) -> Lift4DDepthPrior:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required real Lift4D depth NPZ is missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_DEPTH_KEYS if key not in data]
        if missing:
            raise KeyError(f"Lift4D depth NPZ missing required keys: {missing}; file={path}")
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int64).reshape(-1)
        trajectories = np.asarray(data["point_trajectories_cam"], dtype=np.float64)
        canonical_points = np.asarray(data["canonical_points"], dtype=np.float64)
        visibility = np.asarray(data["point_visibility"], dtype=bool)
        inliers = np.asarray(data["point_fit_inliers"], dtype=bool)
        opacity = np.asarray(data["point_opacity"], dtype=np.float64).reshape(-1)
        valid_point_count = np.asarray(data["valid_point_count"], dtype=np.int64).reshape(-1)
        intrinsics = np.asarray(data["camera_intrinsics"], dtype=np.float64)
        convention = str(np.asarray(data["camera_convention"]).item()).lower()

    if trajectories.ndim != 3 or trajectories.shape[2] != 3:
        raise ValueError(f"point_trajectories_cam must be [T,N,3], got {trajectories.shape}")
    t_count, point_count, _ = trajectories.shape
    if t_count != int(frame_num):
        raise ValueError(
            f"Lift4D/GRAIL frame count mismatch: NPZ={t_count}, GRAIL={frame_num}; file={path}"
        )
    expected_indices = np.arange(frame_num, dtype=np.int64)
    if not np.array_equal(frame_indices, expected_indices):
        raise ValueError(
            "Lift4D frame_indices must be exactly np.arange(frame_num); "
            f"got shape={frame_indices.shape}, first={frame_indices[:8].tolist()}"
        )
    if canonical_points.shape != (point_count, 3):
        raise ValueError(f"canonical_points must be [{point_count},3], got {canonical_points.shape}")
    if visibility.shape != (t_count, point_count) or inliers.shape != visibility.shape:
        raise ValueError(
            "point_visibility and point_fit_inliers must match point_trajectories_cam [T,N]"
        )
    if opacity.shape != (point_count,):
        raise ValueError(f"point_opacity must be [{point_count}], got {opacity.shape}")
    if valid_point_count.shape != (t_count,):
        raise ValueError(f"valid_point_count must be [{t_count}], got {valid_point_count.shape}")
    if intrinsics.shape != (t_count, 3, 3):
        raise ValueError(f"camera_intrinsics must be [{t_count},3,3], got {intrinsics.shape}")
    if convention not in EXPECTED_CAMERA_CONVENTIONS:
        raise ValueError(
            f"Lift4D camera_convention must be OpenCV camera, got {convention!r}; file={path}"
        )
    for name, value in (
        ("canonical_points", canonical_points),
        ("point_opacity", opacity),
        ("camera_intrinsics", intrinsics),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"Lift4D {name} contains NaN or Inf: {path}")
    if np.any(valid_point_count < 0) or np.any(valid_point_count > point_count):
        raise ValueError("valid_point_count contains values outside [0,N]")

    finite_positive = np.isfinite(trajectories).all(axis=2) & (trajectories[:, :, 2] > 0)
    usable = visibility & inliers & finite_positive
    score = (
        visibility.astype(np.float64).mean(axis=0)
        * inliers.astype(np.float64).mean(axis=0)
        * np.sqrt(np.clip(opacity, 0.0, 1.0))
    )
    positive_ids = np.flatnonzero(score > 0)
    if positive_ids.size < min_stable_points:
        raise ValueError(
            f"Too few Lift4D points have positive stability score: {positive_ids.size} < {min_stable_points}"
        )
    stable_point_count = int(stable_point_count)
    if stable_point_count < min_stable_points:
        raise ValueError(
            f"stable_point_count must be >= {min_stable_points}, got {stable_point_count}"
        )
    selected_count = min(stable_point_count, positive_ids.size)
    ranked = positive_ids[np.argsort(score[positive_ids], kind="stable")]
    stable_ids = np.sort(ranked[-selected_count:].astype(np.int64))
    threshold = float(score[stable_ids].min())

    centers = np.full((t_count, 3), np.nan, dtype=np.float64)
    frame_support = np.zeros(t_count, dtype=np.float64)
    visibility_support = visibility[:, stable_ids].mean(axis=1)
    inlier_support = inliers[:, stable_ids].mean(axis=1)
    for frame in range(t_count):
        active = usable[frame, stable_ids]
        frame_support[frame] = active.mean()
        if int(active.sum()) < max(6, min_stable_points // 8):
            continue
        values = trajectories[frame, stable_ids[active]]
        centers[frame] = np.median(values, axis=0)

    valid_frames = np.isfinite(centers).all(axis=1) & (centers[:, 2] > 0)
    if not valid_frames.all():
        bad = np.flatnonzero(~valid_frames)
        raise ValueError(
            f"Lift4D stable center is invalid for frames {bad.tolist()}; no trajectory interpolation fallback is allowed"
        )

    median_window = _odd_window(median_window, t_count, minimum=3)
    median_center = _local_median_filter(centers, window=median_window)
    sg_window = _odd_window(smooth_window, t_count, minimum=5)
    savgol_polyorder = int(savgol_polyorder)
    if savgol_polyorder < 1 or savgol_polyorder >= sg_window:
        raise ValueError(
            f"savgol_polyorder must be in [1, {sg_window - 1}], got {savgol_polyorder}"
        )
    smoothed = savgol_filter(
        median_center,
        window_length=sg_window,
        polyorder=savgol_polyorder,
        axis=0,
        mode="interp",
    )
    if not np.isfinite(smoothed).all() or np.any(smoothed[:, 2] <= 0):
        raise ValueError("Smoothed Lift4D center contains non-finite or non-positive camera depth")

    declared_support = valid_point_count.astype(np.float64) / max(float(point_count), 1.0)
    frame_weight = (
        frame_support
        * np.sqrt(np.clip(visibility_support * inlier_support, 0.0, 1.0))
        * np.sqrt(np.clip(declared_support, 0.0, 1.0))
    )
    if not np.isfinite(frame_weight).all() or frame_weight.max() <= 0:
        raise ValueError("Lift4D frame weights are all invalid or zero")
    frame_weight = np.clip(frame_weight / frame_weight.max(), 1e-6, 1.0)

    return Lift4DDepthPrior(
        frame_indices=frame_indices,
        prior_used=np.ones(t_count, dtype=bool),
        center_cam_raw=centers.astype(np.float32),
        center_cam=smoothed.astype(np.float32),
        z_raw=centers[:, 2].astype(np.float32),
        z=smoothed[:, 2].astype(np.float32),
        delta_z=(smoothed[:, 2] - smoothed[0, 2]).astype(np.float32),
        frame_weight=frame_weight.astype(np.float32),
        valid_point_count=valid_point_count.astype(np.int64),
        stable_point_ids=stable_ids,
        stable_point_score=score.astype(np.float32),
        camera_intrinsics=intrinsics.astype(np.float32),
        source_path=str(path.resolve()),
        camera_convention=convention,
        diagnostics={
            "stable_point_count": int(stable_ids.size),
            "stable_score_threshold": threshold,
            "median_window": int(median_window),
            "smooth_window": int(sg_window),
            "savgol_polyorder": int(savgol_polyorder),
            "supervised_frame_count": int(t_count),
            "min_frame_support": float(frame_support.min()),
            "median_frame_support": float(np.median(frame_support)),
        },
    )
