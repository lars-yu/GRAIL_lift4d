"""Geometry helpers for dynamic-camera VGGT reconstruction.

Coordinate conventions used by this package:

* VGGT camera extrinsics decoded by ``encoding_to_camera`` are OpenCV
  camera-from-world matrices.
* ``c2w`` arrays saved by GRAIL are camera-to-world transforms that map OpenCV
  camera coordinates into the named world frame:
  ``X_world = c2w @ X_camera``.
* The global alignment is a single Sim(3), ``X_B = s * R_BV @ X_V + t_BV``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Sim3:
    """Similarity transform from source world to target world."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self):
        self.scale = float(self.scale)
        self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(self.translation, dtype=np.float64).reshape(3)

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        out = self.scale * (points @ self.rotation.T) + self.translation
        return out.astype(points.dtype, copy=False) if np.issubdtype(points.dtype, np.floating) else out

    def apply_c2w(self, c2w: np.ndarray) -> np.ndarray:
        """Apply this Sim(3) to camera-to-world matrices.

        Rotation stays an SE(3) rotation. Only camera centers are scaled.
        """
        c2w = ensure_4x4(c2w)
        out = np.array(c2w, dtype=np.float64, copy=True)
        out[..., :3, :3] = self.rotation @ c2w[..., :3, :3]
        centers = c2w[..., :3, 3]
        out[..., :3, 3] = self.apply(centers)
        return out.astype(np.float32)

    def to_dict(self, metadata: dict | None = None) -> dict:
        data = {
            "scale": self.scale,
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "convention": "X_blender = scale * rotation @ X_vggt + translation",
        }
        if metadata:
            data["metadata"] = metadata
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Sim3":
        return cls(data["scale"], data["rotation"], data["translation"])

    @classmethod
    def load(cls, path: str | Path) -> "Sim3":
        with open(path, "r") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(metadata), handle, indent=2)


def ensure_4x4(transform: np.ndarray) -> np.ndarray:
    arr = np.asarray(transform)
    if arr.shape[-2:] == (4, 4):
        return arr
    if arr.shape[-2:] == (3, 4):
        eye = np.broadcast_to(np.eye(4, dtype=arr.dtype), arr.shape[:-2] + (4, 4)).copy()
        eye[..., :3, :4] = arr
        return eye
    raise ValueError(f"Expected (..., 3, 4) or (..., 4, 4), got {arr.shape}")


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = ensure_4x4(transform)
    return np.linalg.inv(transform)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = ensure_4x4(transform)
    points = np.asarray(points)
    return points @ transform[..., :3, :3].swapaxes(-1, -2) + transform[..., :3, 3]


def estimate_umeyama_sim3(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Closed-form Umeyama alignment for ``dst ~= s R src + t``."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/dst must both be (N, 3), got {src.shape} and {dst.shape}")
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 correspondences for Sim(3) alignment")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / src.shape[0]
    U, singular, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1.0
    R = U @ D @ Vt

    if with_scale:
        var_src = np.mean(np.sum(src_c**2, axis=1))
        if var_src <= 1e-12:
            raise ValueError("Degenerate source point cloud for Sim(3) alignment")
        scale = float(np.sum(singular * np.diag(D)) / var_src)
    else:
        scale = 1.0
    t = mu_dst - scale * (R @ mu_src)
    return Sim3(scale, R, t)


def ransac_umeyama_sim3(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float = 0.03,
    max_iterations: int = 512,
    min_samples: int = 6,
    seed: int = 0,
) -> tuple[Sim3, np.ndarray, dict]:
    """Robust Sim(3) fit using RANSAC over Umeyama hypotheses."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] != dst.shape[0]:
        raise ValueError("src and dst correspondence counts differ")
    n = src.shape[0]
    if n < min_samples:
        raise ValueError(f"Need at least {min_samples} correspondences, got {n}")

    rng = np.random.default_rng(seed)
    best_inliers = None
    best_error = np.inf
    best_sim3 = None

    for _ in range(max_iterations):
        sample = rng.choice(n, size=min_samples, replace=False)
        try:
            sim3 = estimate_umeyama_sim3(src[sample], dst[sample], with_scale=True)
        except ValueError:
            continue
        residual = np.linalg.norm(sim3.apply(src) - dst, axis=1)
        inliers = residual < threshold
        count = int(inliers.sum())
        if count < min_samples:
            continue
        median = float(np.median(residual[inliers]))
        score = (-count, median)
        best_score = (-(int(best_inliers.sum())) if best_inliers is not None else 0, best_error)
        if best_inliers is None or score < best_score:
            best_inliers = inliers
            best_error = median
            best_sim3 = sim3

    if best_inliers is None:
        best_sim3 = estimate_umeyama_sim3(src, dst, with_scale=True)
        residual = np.linalg.norm(best_sim3.apply(src) - dst, axis=1)
        best_inliers = residual < threshold

    if int(best_inliers.sum()) >= min_samples:
        best_sim3 = estimate_umeyama_sim3(src[best_inliers], dst[best_inliers], with_scale=True)

    residual = np.linalg.norm(best_sim3.apply(src) - dst, axis=1)
    stats = {
        "num_correspondences": int(n),
        "num_inliers": int(best_inliers.sum()),
        "inlier_ratio": float(best_inliers.mean()),
        "rmse_m": float(np.sqrt(np.mean(residual[best_inliers] ** 2)))
        if np.any(best_inliers)
        else float(np.sqrt(np.mean(residual**2))),
        "median_m": float(np.median(residual[best_inliers]))
        if np.any(best_inliers)
        else float(np.median(residual)),
        "threshold_m": float(threshold),
    }
    return best_sim3, best_inliers, stats


def scale_intrinsics(K: np.ndarray, from_hw: tuple[int, int], to_hw: tuple[int, int]) -> np.ndarray:
    """Scale pinhole intrinsics between image sizes with the same crop/aspect."""
    K = np.asarray(K, dtype=np.float64).copy()
    from_h, from_w = from_hw
    to_h, to_w = to_hw
    sx = float(to_w) / float(from_w)
    sy = float(to_h) / float(from_h)
    K[..., 0, 0] *= sx
    K[..., 0, 2] *= sx
    K[..., 1, 1] *= sy
    K[..., 1, 2] *= sy
    return K.astype(np.float32)


def erode_mask(mask: np.ndarray, pixels: int = 5) -> np.ndarray:
    mask = np.asarray(mask).astype(np.uint8)
    if pixels <= 0:
        return mask > 0
    kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), dtype=np.uint8)
    return cv2.erode(mask, kernel, iterations=1) > 0


def resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    if mask.shape == shape_hw:
        return mask > 0
    return cv2.resize(mask.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST) > 0


def resize_depth(depth: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape == shape_hw:
        return depth
    return cv2.resize(depth, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)


def unproject_opencv_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Unproject camera-Z depth to OpenCV camera coordinates."""
    depth = np.asarray(depth, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    H, W = depth.shape
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth
    X = (x.astype(np.float32) - K[0, 2]) / K[0, 0] * z
    Y = (y.astype(np.float32) - K[1, 2]) / K[1, 1] * z
    return np.stack([X, Y, z], axis=-1)


def unproject_opencv_depth_to_world(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    cam = unproject_opencv_depth(depth, K)
    flat = cam.reshape(-1, 3)
    world = transform_points(c2w, flat).reshape(cam.shape)
    return world.astype(np.float32)


def world_points_to_camera_depth(points_world: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    H, W = points_world.shape[:2]
    w2c = invert_transform(c2w)
    cam = transform_points(w2c, points_world.reshape(-1, 3)).reshape(H, W, 3)
    return cam[..., 2].astype(np.float32)


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R = R_a.T @ R_b
    cos = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cos))


def camera_motion_metrics(c2w: np.ndarray, fps: float | None = None) -> dict:
    c2w = ensure_4x4(c2w)
    centers = c2w[:, :3, 3]
    trans_delta = np.linalg.norm(centers[1:] - centers[:-1], axis=1) if len(c2w) > 1 else np.array([])
    rot_delta = (
        np.array([rotation_angle_deg(c2w[i, :3, :3], c2w[i + 1, :3, :3]) for i in range(len(c2w) - 1)])
        if len(c2w) > 1
        else np.array([])
    )
    accel = np.linalg.norm(centers[2:] - 2.0 * centers[1:-1] + centers[:-2], axis=1) if len(c2w) > 2 else np.array([])
    scale = float(fps) if fps and fps > 0 else 1.0
    jump_threshold = max(0.25, 5.0 * float(np.median(trans_delta))) if trans_delta.size else np.inf
    jump_frames = (np.where(trans_delta > jump_threshold)[0] + 1).astype(int).tolist() if trans_delta.size else []
    return {
        "camera_center": centers.tolist(),
        "camera_translation_delta_m": trans_delta.tolist(),
        "camera_rotation_delta_deg": rot_delta.tolist(),
        "camera_translation_speed_mps": (trans_delta * scale).tolist(),
        "camera_rotation_speed_degps": (rot_delta * scale).tolist(),
        "camera_acceleration_delta_m": accel.tolist(),
        "camera_jump_frames": jump_frames,
    }


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if colors is not None:
        colors = np.asarray(colors).reshape(-1, 3)[valid]
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    with open(path, "w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        if colors is None:
            for p in points:
                handle.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            for p, c in zip(points, colors):
                handle.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def _ply_numpy_dtype(type_name: str, endian: str) -> str:
    type_name = type_name.lower()
    mapping = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "i2",
        "int16": "i2",
        "ushort": "u2",
        "uint16": "u2",
        "int": "i4",
        "int32": "i4",
        "uint": "u4",
        "uint32": "u4",
        "float": "f4",
        "float32": "f4",
        "double": "f8",
        "float64": "f8",
    }
    if type_name not in mapping:
        raise ValueError(f"Unsupported PLY property type: {type_name}")
    dtype = mapping[type_name]
    return dtype if dtype.endswith("1") else endian + dtype


def read_ply_vertices(path: str | Path, max_points: int | None = None) -> np.ndarray:
    """Read vertex positions from ASCII or binary PLY files.

    Only vertex ``x/y/z`` properties are returned. Face data and colors are
    intentionally ignored because dynamic-camera alignment and validation use
    PLY files as static-scene point anchors.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PLY file missing: {path}")
    with open(path, "rb") as handle:
        first = handle.readline().decode("ascii", errors="replace").strip()
        if first != "ply":
            raise RuntimeError(f"Not a PLY file: {path}")
        fmt = None
        vertex_count = None
        vertex_properties: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            raw = handle.readline()
            if not raw:
                raise RuntimeError(f"PLY header missing end_header: {path}")
            line = raw.decode("ascii", errors="replace").strip()
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format" and len(parts) >= 2:
                fmt = parts[1]
            elif parts[0] == "element" and len(parts) >= 3:
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and parts[0] == "property" and len(parts) >= 3 and parts[1] != "list":
                vertex_properties.append((parts[2], parts[1]))
            elif parts[0] == "end_header":
                break

        if fmt is None or vertex_count is None:
            raise RuntimeError(f"PLY header missing format or vertex count: {path}")
        prop_names = [name for name, _ in vertex_properties]
        try:
            xyz_indices = [prop_names.index(axis) for axis in ("x", "y", "z")]
        except ValueError:
            if len(vertex_properties) < 3:
                raise RuntimeError(f"PLY vertex properties do not contain x/y/z: {path}")
            xyz_indices = [0, 1, 2]

        if fmt == "ascii":
            step = max(1, vertex_count // max_points) if max_points and max_points > 0 else 1
            points = []
            for idx in range(vertex_count):
                raw = handle.readline()
                if not raw:
                    break
                if idx % step != 0:
                    continue
                parts = raw.decode("ascii", errors="replace").split()
                if len(parts) > max(xyz_indices):
                    points.append([float(parts[i]) for i in xyz_indices])
            if not points:
                raise RuntimeError(f"No vertices loaded from {path}")
            pts = np.asarray(points, dtype=np.float32)
        elif fmt in {"binary_little_endian", "binary_big_endian"}:
            endian = "<" if fmt == "binary_little_endian" else ">"
            dtype = np.dtype(
                [(name, _ply_numpy_dtype(type_name, endian)) for name, type_name in vertex_properties]
            )
            raw = handle.read(vertex_count * dtype.itemsize)
            data = np.frombuffer(raw, dtype=dtype, count=vertex_count)
            fields = [prop_names[i] for i in xyz_indices]
            pts = np.stack([data[field].astype(np.float32, copy=False) for field in fields], axis=1)
            if max_points and max_points > 0 and len(pts) > max_points:
                idx = np.linspace(0, len(pts) - 1, max_points).astype(np.int64)
                pts = pts[idx]
        else:
            raise RuntimeError(f"Unsupported PLY format {fmt!r}: {path}")

    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        raise RuntimeError(f"No finite vertices loaded from {path}")
    return pts.astype(np.float32, copy=False)


def load_depth_image(path: str | Path) -> np.ndarray:
    path = str(path)
    if path.lower().endswith(".npy"):
        return np.load(path).astype(np.float32)
    if path.lower().endswith(".exr"):
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        if path.lower().endswith(".exr"):
            raise RuntimeError(
                "Could not read EXR depth image with OpenCV. Set "
                "OPENCV_IO_ENABLE_OPENEXR=1 before reading, or export depth "
                "as .npy/.png."
            ) from exc
        raise
    if img is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")
    depth = img.astype(np.float32)
    if path.lower().endswith(".png") and float(np.nanmax(depth)) > 100.0:
        depth = depth / 1000.0
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth
