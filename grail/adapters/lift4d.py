"""Lift4D motion-prior utilities for GRAIL.

This module intentionally does not import Lift4D, PyTorch3D, or rendering code.
It handles the motion-only contract between a Lift4D export NPZ and GRAIL:
rigid fitting from tracked points, fixed-camera alignment to FoundationPose, and
diagnostic trajectory quantities.  Keeping it dependency-light ensures that
GRAIL behaves exactly as before when Lift4D supervision is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np


@dataclass
class RigidFitResult:
    R: np.ndarray
    t: np.ndarray
    scale: float
    rmse: float
    inlier_mask: np.ndarray
    confidence: float
    valid: bool
    valid_point_count: int


@dataclass
class Lift4DMotionNPZ:
    frame_indices: np.ndarray
    object_poses_cam: np.ndarray
    motion_confidence: np.ndarray
    rigid_fit_rmse: np.ndarray
    object_scales: np.ndarray
    camera_convention: str
    image_size: tuple[int, int] | None
    canonical_object_center: np.ndarray | None
    source_path: str


@dataclass
class AlignedLift4DMotionPrior:
    object_poses: np.ndarray
    motion_valid: np.ndarray
    motion_confidence: np.ndarray
    rigid_fit_rmse: np.ndarray
    object_scales: np.ndarray
    source_path: str
    anchor_frame: int
    translation_scale: float
    camera_convention: str
    frame_indices: np.ndarray
    diagnostics: dict[str, Any]


def _as_float_array(value: Any, shape_last: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if shape_last is not None and (arr.ndim == 0 or arr.shape[-1] != shape_last):
        raise ValueError(f"Expected array with last dimension {shape_last}, got {arr.shape}")
    return arr


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray, scale: float = 1.0) -> np.ndarray:
    points = _as_float_array(points, 3)
    return scale * (points @ np.asarray(R, dtype=np.float64).T) + np.asarray(t, dtype=np.float64)


def weighted_kabsch_umeyama(
    canonical_points: np.ndarray,
    target_points: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    estimate_scale: bool = False,
    min_points: int = 6,
    outlier_sigma: float = 3.0,
    robust_iters: int = 2,
    rmse_good: float = 0.01,
    rmse_bad: float = 0.08,
) -> RigidFitResult:
    """Fit ``target ~= scale * R @ canonical + t`` with robust weighted Kabsch.

    The returned ``scale`` is diagnostic only.  GRAIL must not scale the real
    object mesh with this value.
    """

    src = _as_float_array(canonical_points, 3)
    dst = _as_float_array(target_points, 3)
    if src.shape != dst.shape:
        raise ValueError(f"Point shape mismatch: {src.shape} vs {dst.shape}")
    n = src.shape[0]
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != n:
            raise ValueError(f"weights length {w.shape[0]} does not match {n} points")
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1) & np.isfinite(w) & (w > 0)
    active = finite.copy()

    def _invalid() -> RigidFitResult:
        return RigidFitResult(
            R=np.eye(3, dtype=np.float64),
            t=np.zeros(3, dtype=np.float64),
            scale=1.0,
            rmse=float("inf"),
            inlier_mask=np.zeros(n, dtype=bool),
            confidence=0.0,
            valid=False,
            valid_point_count=int(active.sum()),
        )

    if int(active.sum()) < min_points:
        return _invalid()

    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    scale = 1.0
    residuals = np.full(n, np.inf, dtype=np.float64)

    for _ in range(max(1, robust_iters + 1)):
        idx = np.flatnonzero(active)
        if idx.shape[0] < min_points:
            return _invalid()
        src_i = src[idx]
        dst_i = dst[idx]
        w_i = w[idx]
        w_i = w_i / np.clip(w_i.sum(), 1e-12, None)

        mu_src = (src_i * w_i[:, None]).sum(axis=0)
        mu_dst = (dst_i * w_i[:, None]).sum(axis=0)
        X = src_i - mu_src
        Y = dst_i - mu_dst
        H = X.T @ (Y * w_i[:, None])
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T
        if estimate_scale:
            var_src = (w_i * np.sum(X * X, axis=1)).sum()
            scale = float(S.sum() / np.clip(var_src, 1e-12, None))
        else:
            scale = 1.0
        t = mu_dst - scale * (mu_src @ R.T)
        pred = transform_points(src, R, t, scale=scale)
        residuals = np.linalg.norm(pred - dst, axis=1)
        active_res = residuals[active]
        med = np.median(active_res)
        mad = np.median(np.abs(active_res - med))
        robust_std = 1.4826 * mad
        cutoff = med + outlier_sigma * max(robust_std, 1e-12)
        new_active = finite & (residuals <= cutoff)
        if np.array_equal(new_active, active):
            break
        active = new_active

    if int(active.sum()) < min_points:
        return _invalid()
    rmse = float(np.sqrt(np.average(residuals[active] ** 2, weights=w[active])))
    visible_ratio = float(active.sum() / max(1, finite.sum()))
    point_score = float(np.clip((active.sum() - min_points) / max(float(min_points), 1.0), 0.0, 1.0))
    rmse_score = float(np.clip((rmse_bad - rmse) / max(rmse_bad - rmse_good, 1e-12), 0.0, 1.0))
    confidence = float(np.clip(visible_ratio * point_score * rmse_score, 0.0, 1.0))
    return RigidFitResult(
        R=R.astype(np.float64),
        t=t.astype(np.float64),
        scale=float(scale),
        rmse=rmse,
        inlier_mask=active.astype(bool),
        confidence=confidence,
        valid=confidence > 0.0,
        valid_point_count=int(active.sum()),
    )


def load_motion_npz(path: str | Path) -> Lift4DMotionNPZ:
    path = str(path)
    data = np.load(path, allow_pickle=True)
    required = [
        "frame_indices",
        "object_poses_cam",
        "motion_confidence",
        "rigid_fit_rmse",
        "object_scales",
        "camera_convention",
        "image_size",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Lift4D motion NPZ missing required keys: {missing}")
    frame_indices = np.asarray(data["frame_indices"], dtype=np.int64).reshape(-1)
    poses = np.asarray(data["object_poses_cam"], dtype=np.float64)
    if poses.shape != (frame_indices.shape[0], 4, 4):
        raise ValueError(
            f"object_poses_cam must have shape ({frame_indices.shape[0]}, 4, 4), got {poses.shape}"
        )
    image_size_arr = np.asarray(data["image_size"]).reshape(-1)
    image_size = None
    if image_size_arr.size >= 2 and np.all(np.isfinite(image_size_arr[:2].astype(float))):
        image_size = (int(image_size_arr[0]), int(image_size_arr[1]))
    center = None
    if "canonical_object_center" in data:
        center = np.asarray(data["canonical_object_center"], dtype=np.float64).reshape(3)
    return Lift4DMotionNPZ(
        frame_indices=frame_indices,
        object_poses_cam=poses,
        motion_confidence=np.asarray(data["motion_confidence"], dtype=np.float64).reshape(-1),
        rigid_fit_rmse=np.asarray(data["rigid_fit_rmse"], dtype=np.float64).reshape(-1),
        object_scales=np.asarray(data["object_scales"], dtype=np.float64).reshape(-1),
        camera_convention=str(np.asarray(data["camera_convention"]).item()),
        image_size=image_size,
        canonical_object_center=center,
        source_path=path,
    )


def _camera_rotation(camera_c2w: np.ndarray) -> np.ndarray:
    cam = np.asarray(camera_c2w, dtype=np.float64)
    if cam.shape == (3, 3):
        return cam
    if cam.shape == (4, 4):
        return cam[:3, :3]
    raise ValueError(f"camera_c2w must be (3,3) or (4,4), got {cam.shape}")


def _relative_center_deltas_cam(poses_cam: np.ndarray, anchor_pos: int, center: np.ndarray) -> np.ndarray:
    R = poses_cam[:, :3, :3]
    t = poses_cam[:, :3, 3]
    centers = np.einsum("fij,j->fi", R, center) + t
    return centers - centers[anchor_pos : anchor_pos + 1]


def fit_translation_scale(
    fp_poses_world: np.ndarray,
    lift4d_poses_cam: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray,
    anchor_pos: int,
    camera_c2w: np.ndarray,
    *,
    canonical_center: np.ndarray | None = None,
    min_motion: float = 1e-5,
) -> float:
    """Robustly fit one scalar mapping Lift4D relative translations to FP scale."""

    fp = np.asarray(fp_poses_world, dtype=np.float64)
    center = np.zeros(3, dtype=np.float64) if canonical_center is None else np.asarray(canonical_center, dtype=np.float64)
    R_c2w = _camera_rotation(camera_c2w)
    lift_delta_cam = _relative_center_deltas_cam(lift4d_poses_cam, anchor_pos, center)
    lift_delta_world = lift_delta_cam @ R_c2w.T
    anchor_frame = anchor_pos
    fp_delta = fp[:, :3, 3] - fp[anchor_frame : anchor_frame + 1, :3, 3]
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    valid[anchor_frame] = False
    denom = np.sum(lift_delta_world * lift_delta_world, axis=1)
    motion_valid = valid & np.isfinite(denom) & (denom > min_motion**2)
    if int(motion_valid.sum()) < 1:
        raise ValueError("Cannot fit Lift4D translation scale: no valid non-anchor motion frames")
    numer = np.sum(lift_delta_world[motion_valid] * fp_delta[motion_valid], axis=1)
    candidates = numer / np.clip(denom[motion_valid], 1e-12, None)
    weights = np.clip(confidence[motion_valid], 1e-6, None)
    order = np.argsort(candidates)
    c_sorted = candidates[order]
    w_sorted = weights[order]
    cutoff = 0.5 * w_sorted.sum()
    scale = float(c_sorted[np.searchsorted(np.cumsum(w_sorted), cutoff)])
    if not np.isfinite(scale) or abs(scale) < 1e-12:
        raise ValueError(f"Invalid fitted Lift4D translation scale: {scale}")
    return scale


def align_lift4d_motion_to_foundationpose(
    motion: Lift4DMotionNPZ,
    fp_poses_world: np.ndarray,
    camera_c2w: np.ndarray,
    *,
    frame_num: int | None = None,
    min_confidence: float = 0.2,
    translation_scale: float | None = None,
    camera_mode: str = "fixed",
) -> AlignedLift4DMotionPrior:
    """Map a motion-only Lift4D NPZ onto the full GRAIL timeline.

    ``fp_poses_world`` is FoundationPose transformed to GRAIL world coordinates.
    Lift4D remains relative: the first valid/high-confidence frame anchors to
    FoundationPose, then only relative rotation and center displacement are used.
    """

    if camera_mode != "fixed":
        raise ValueError("Lift4D motion prior currently supports fixed camera only; got dynamic camera")

    fp = np.asarray(fp_poses_world, dtype=np.float64)
    if fp.ndim != 3 or fp.shape[1:] != (4, 4):
        raise ValueError(f"fp_poses_world must be (F,4,4), got {fp.shape}")
    if frame_num is None:
        frame_num = fp.shape[0]
    if frame_num != fp.shape[0]:
        raise ValueError(f"frame_num {frame_num} does not match FP poses {fp.shape[0]}")

    convention = motion.camera_convention.lower()
    if convention not in {"opencv", "opencv_camera", "opencv_camera_z", "camera", "camera_space"}:
        warnings.warn(f"Unrecognized Lift4D camera convention {motion.camera_convention!r}; assuming OpenCV-like camera space")

    poses_full = np.repeat(np.eye(4, dtype=np.float64)[None], frame_num, axis=0)
    conf_full = np.zeros(frame_num, dtype=np.float64)
    rmse_full = np.full(frame_num, np.inf, dtype=np.float64)
    scale_full = np.ones(frame_num, dtype=np.float64)
    valid_full = np.zeros(frame_num, dtype=bool)

    in_range = (motion.frame_indices >= 0) & (motion.frame_indices < frame_num)
    if not np.all(in_range):
        bad = motion.frame_indices[~in_range]
        warnings.warn(f"Ignoring Lift4D prior frames outside GRAIL timeline: {bad.tolist()}")
    src_pos = np.flatnonzero(in_range)
    full_idx = motion.frame_indices[src_pos]
    conf_full[full_idx] = motion.motion_confidence[src_pos]
    rmse_full[full_idx] = motion.rigid_fit_rmse[src_pos]
    scale_full[full_idx] = motion.object_scales[src_pos]
    valid_full[full_idx] = np.isfinite(conf_full[full_idx]) & (conf_full[full_idx] >= min_confidence)
    if not valid_full.any():
        raise ValueError(f"No Lift4D motion frames meet min_confidence={min_confidence}")
    anchor_frame = int(np.flatnonzero(valid_full)[0])
    anchor_src = int(np.where(motion.frame_indices == anchor_frame)[0][0])

    center = motion.canonical_object_center
    if center is None:
        center = np.zeros(3, dtype=np.float64)
        warnings.warn(
            "Lift4D motion NPZ has no canonical_object_center; translation prior falls back to canonical origin. "
            "Re-export for the pure-rotation-safe center displacement path."
        )

    source_poses_full = np.repeat(np.eye(4, dtype=np.float64)[None], frame_num, axis=0)
    source_poses_full[full_idx] = motion.object_poses_cam[src_pos]
    if translation_scale is None:
        translation_scale = fit_translation_scale(
            fp,
            source_poses_full,
            valid_full.copy(),
            conf_full,
            anchor_frame,
            camera_c2w,
            canonical_center=center,
        )

    R_c2w = _camera_rotation(camera_c2w)
    R_a = motion.object_poses_cam[anchor_src, :3, :3]
    center_a_cam = R_a @ center + motion.object_poses_cam[anchor_src, :3, 3]
    fp_anchor_R = fp[anchor_frame, :3, :3]
    fp_anchor_t = fp[anchor_frame, :3, 3]

    for src_i, frame_idx in zip(src_pos, full_idx):
        T = motion.object_poses_cam[src_i]
        R_t = T[:3, :3]
        center_t_cam = R_t @ center + T[:3, 3]
        R_rel_cam = R_t @ R_a.T
        R_rel_world = R_c2w @ R_rel_cam @ R_c2w.T
        poses_full[frame_idx, :3, :3] = R_rel_world @ fp_anchor_R
        delta_world = R_c2w @ (center_t_cam - center_a_cam)
        poses_full[frame_idx, :3, 3] = fp_anchor_t + float(translation_scale) * delta_world
    # Missing frames are invalid and retain identity placeholders; consumers must gate on valid.
    return AlignedLift4DMotionPrior(
        object_poses=poses_full.astype(np.float32),
        motion_valid=valid_full,
        motion_confidence=conf_full.astype(np.float32),
        rigid_fit_rmse=rmse_full.astype(np.float32),
        object_scales=scale_full.astype(np.float32),
        source_path=motion.source_path,
        anchor_frame=anchor_frame,
        translation_scale=float(translation_scale),
        camera_convention=motion.camera_convention,
        frame_indices=motion.frame_indices.copy(),
        diagnostics={
            "anchor_source_index": anchor_src,
            "canonical_object_center": center.astype(np.float32),
            "min_confidence": float(min_confidence),
        },
    )


def load_aligned_lift4d_motion_prior(
    path: str | Path,
    fp_poses_world: np.ndarray,
    camera_c2w: np.ndarray,
    *,
    frame_num: int | None = None,
    min_confidence: float = 0.2,
    translation_scale: float | None = None,
    camera_mode: str = "fixed",
) -> AlignedLift4DMotionPrior:
    motion = load_motion_npz(path)
    return align_lift4d_motion_to_foundationpose(
        motion,
        fp_poses_world,
        camera_c2w,
        frame_num=frame_num,
        min_confidence=min_confidence,
        translation_scale=translation_scale,
        camera_mode=camera_mode,
    )


def motion_diagnostics(poses: np.ndarray, valid: np.ndarray, anchor_frame: int) -> dict[str, np.ndarray]:
    poses = np.asarray(poses, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    t = poses[:, :3, 3]
    R = poses[:, :3, :3]
    rel_t = np.linalg.norm(t - t[anchor_frame : anchor_frame + 1], axis=1)
    rel_angle = np.zeros(poses.shape[0], dtype=np.float64)
    linear_velocity = np.full(poses.shape[0], np.nan, dtype=np.float64)
    angular_velocity = np.full(poses.shape[0], np.nan, dtype=np.float64)
    R_anchor = R[anchor_frame]
    for i in range(poses.shape[0]):
        rel = R[i] @ R_anchor.T
        cos = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
        rel_angle[i] = np.arccos(cos)
    valid_idx = np.flatnonzero(valid)
    for prev, cur in zip(valid_idx[:-1], valid_idx[1:]):
        dt = max(1, int(cur - prev))
        linear_velocity[cur] = np.linalg.norm(t[cur] - t[prev]) / dt
        rel = R[cur] @ R[prev].T
        cos = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
        angular_velocity[cur] = np.arccos(cos) / dt
    return {
        "tx": t[:, 0].copy(),
        "ty": t[:, 1].copy(),
        "tz": t[:, 2].copy(),
        "translation_from_anchor": rel_t,
        "rotation_from_anchor_rad": rel_angle,
        "linear_velocity": linear_velocity,
        "angular_velocity_rad": angular_velocity,
    }


def save_motion_npz(
    path: str | Path,
    *,
    frame_indices: np.ndarray,
    object_poses_cam: np.ndarray,
    motion_confidence: np.ndarray,
    rigid_fit_rmse: np.ndarray,
    object_scales: np.ndarray,
    image_size: tuple[int, int] | None,
    camera_convention: str = "opencv_camera",
    canonical_object_center: np.ndarray | None = None,
    **extra: Any,
) -> None:
    payload: dict[str, Any] = {
        "format_version": np.asarray("lift4d_motion_npz_v1"),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "object_poses_cam": np.asarray(object_poses_cam, dtype=np.float32),
        "motion_confidence": np.asarray(motion_confidence, dtype=np.float32),
        "rigid_fit_rmse": np.asarray(rigid_fit_rmse, dtype=np.float32),
        "object_scales": np.asarray(object_scales, dtype=np.float32),
        "camera_convention": np.asarray(camera_convention),
        "image_size": np.asarray(image_size if image_size is not None else (-1, -1), dtype=np.int64),
    }
    if canonical_object_center is not None:
        payload["canonical_object_center"] = np.asarray(canonical_object_center, dtype=np.float32)
    payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
