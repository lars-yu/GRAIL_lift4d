"""Coordinate-convention adapters for Blender, VGGT and FoundationPose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from grail.dynamic_camera.geometry import ensure_4x4, invert_transform


BLENDER_CAMERA_TO_OPENCV = np.asarray(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, -1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float32,
)


@dataclass
class CameraPoseConversion:
    source: str
    c2w: np.ndarray
    w2c: np.ndarray


def _pose_from_c2w(source: str, c2w: np.ndarray) -> CameraPoseConversion:
    c2w = ensure_4x4(np.asarray(c2w, dtype=np.float32))
    return CameraPoseConversion(source=source, c2w=c2w, w2c=invert_transform(c2w).astype(np.float32))


def _pose_from_w2c(source: str, w2c: np.ndarray) -> CameraPoseConversion:
    w2c = ensure_4x4(np.asarray(w2c, dtype=np.float32))
    return CameraPoseConversion(source=source, c2w=invert_transform(w2c).astype(np.float32), w2c=w2c)


def convert_vggt_to_internal(*, c2w: np.ndarray | None = None, w2c: np.ndarray | None = None) -> CameraPoseConversion:
    """VGGT extrinsics already use the internal OpenCV camera axes."""
    if c2w is not None:
        return _pose_from_c2w("vggt", c2w)
    if w2c is not None:
        return _pose_from_w2c("vggt", w2c)
    raise ValueError("convert_vggt_to_internal requires c2w or w2c")


def convert_blender_to_internal(*, c2w: np.ndarray | None = None, w2c: np.ndarray | None = None) -> CameraPoseConversion:
    """Convert Blender camera axes to the internal OpenCV convention."""
    flip = BLENDER_CAMERA_TO_OPENCV
    if c2w is not None:
        return _pose_from_c2w("blender", ensure_4x4(np.asarray(c2w, dtype=np.float32)) @ flip)
    if w2c is not None:
        return _pose_from_w2c("blender", flip @ ensure_4x4(np.asarray(w2c, dtype=np.float32)))
    raise ValueError("convert_blender_to_internal requires c2w or w2c")


def convert_foundationpose_to_internal(pose_cam: np.ndarray) -> np.ndarray:
    """FoundationPose object poses are already OpenCV camera-space transforms."""
    return np.asarray(pose_cam, dtype=np.float32)


def convert_genmo_to_internal(motion: dict[str, Any]) -> dict[str, Any]:
    """Annotate GENMO/HMR motion as OpenCV camera-space without changing values."""
    out = dict(motion)
    out.setdefault("camera_convention", "opencv_camera")
    out.setdefault("source_convention", "genmo_camera")
    return out
