"""Extract VGGT world-space observations for dynamic-camera optimization."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from grail.dynamic_camera.geometry import erode_mask, resize_mask, unproject_opencv_depth_to_world
from grail.preprocessing.preprocess import load_masks_from_cache


def _frame_mask(video_masks: dict, frame_idx: int, obj_id: int, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = video_masks[frame_idx][obj_id]
    return resize_mask(mask, shape_hw)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
    return points


def _adaptive_erosion_pixels(mask: np.ndarray, max_erode_pixels: int) -> int:
    """Use less erosion for small masks so tiny objects keep usable pixels."""
    area = int(np.count_nonzero(mask))
    max_erode_pixels = max(int(max_erode_pixels), 0)
    if area > 5000:
        erosion = 4
    elif area > 1500:
        erosion = 2
    elif area > 300:
        erosion = 1
    else:
        erosion = 0
    return min(erosion, max_erode_pixels)


def _object_quality_weight(num_points: int, low_conf_points: int, normal_conf_points: int) -> float:
    """Discrete object observation quality; point count is not used repeatedly."""
    if num_points < low_conf_points:
        return 0.0
    if num_points < normal_conf_points:
        return 0.5
    return 1.0


def _same_path(a: str | Path | None, b: str | Path | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return Path(a).resolve(strict=False) == Path(b).resolve(strict=False)
    except Exception:
        return str(a) == str(b)


def extract_vggt_observations(
    aligned_dir: str | Path,
    masks_cache_file: str | Path,
    output_dir: str | Path,
    *,
    confidence_percentile: float = 40.0,
    erode_pixels: int = 5,
    max_points_per_mask: int = 5000,
    object_min_points: int = 10,
    object_normal_points: int = 50,
    skip_done: bool = False,
) -> dict:
    """Save masked VGGT observations in Blender metric world coordinates."""
    aligned_dir = Path(aligned_dir)
    output_dir = Path(output_dir)
    metadata_path = output_dir / "metadata.json"
    required_dirs = (
        output_dir / "human_raw_points",
        output_dir / "human_points",
        output_dir / "object_raw_points",
        output_dir / "object_points",
        output_dir / "static_points",
        output_dir / "human_masks",
        output_dir / "object_masks",
        output_dir / "confidence_masks",
    )
    if skip_done and metadata_path.exists() and all(path.is_dir() for path in required_dirs):
        with open(metadata_path, "r") as handle:
            cached = json.load(handle)
        frame_count = int(cached.get("num_frames") or cached.get("frame_count") or 0)
        provenance_ok = (
            _same_path(cached.get("aligned_dir"), aligned_dir)
            and _same_path(cached.get("masks_cache_file"), masks_cache_file)
            and float(cached.get("confidence_percentile", confidence_percentile)) == float(confidence_percentile)
            and int(cached.get("erode_pixels", erode_pixels)) == int(erode_pixels)
            and int(cached.get("max_points_per_mask", max_points_per_mask)) == int(max_points_per_mask)
            and int(cached.get("object_min_points", object_min_points)) == int(object_min_points)
            and int(cached.get("object_normal_points", object_normal_points)) == int(object_normal_points)
        )
        if frame_count > 0 and provenance_ok:
            complete = True
            for path, pattern in (
                (output_dir / "human_points", "*.npy"),
                (output_dir / "human_raw_points", "*.npy"),
                (output_dir / "object_points", "*.npy"),
                (output_dir / "object_raw_points", "*.npy"),
                (output_dir / "static_points", "*.npy"),
                (output_dir / "human_masks", "*.png"),
                (output_dir / "object_masks", "*.png"),
                (output_dir / "confidence_masks", "*.png"),
            ):
                complete &= len(list(path.glob(pattern))) >= frame_count
            if complete:
                return cached

    depth = np.load(aligned_dir / "metric_depth" / "depth.npy")
    c2w = np.load(aligned_dir / "c2w_blender.npy")
    intrinsics = np.load(aligned_dir / "intrinsics.npy")
    confidence = np.load(aligned_dir / "confidence.npy")
    masks = load_masks_from_cache(str(masks_cache_file))
    T, H, W = depth.shape

    dirs = {
        "human_raw_points": output_dir / "human_raw_points",
        "human_points": output_dir / "human_points",
        "object_raw_points": output_dir / "object_raw_points",
        "object_points": output_dir / "object_points",
        "static_points": output_dir / "static_points",
        "human_masks": output_dir / "human_masks",
        "object_masks": output_dir / "object_masks",
        "confidence_masks": output_dir / "confidence_masks",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    counts = {"human": [], "object": [], "static": []}
    raw_counts = {"human": [], "object": []}
    object_erosion_px = []
    mask_area_px = {"human": [], "object": [], "static": []}
    confidence_mean = {"human": [], "object": [], "static": []}

    def _confidence_mean(mask: np.ndarray, conf: np.ndarray) -> float:
        values = conf[mask & np.isfinite(conf)]
        if values.size == 0:
            return 0.0
        return float(np.mean(values))
    for t in range(T):
        points_world = unproject_opencv_depth_to_world(depth[t], intrinsics[t], c2w[t])
        human_raw = _frame_mask(masks, t, 1, (H, W))
        obj_raw = _frame_mask(masks, t, 0, (H, W))
        human = erode_mask(human_raw, erode_pixels)
        object_erode = _adaptive_erosion_pixels(obj_raw, erode_pixels)
        obj = erode_mask(obj_raw, object_erode)
        object_erosion_px.append(int(object_erode))
        valid_depth = np.isfinite(depth[t]) & (depth[t] > 0)
        valid_conf = np.isfinite(confidence[t])
        if valid_conf.any():
            cutoff = np.percentile(confidence[t][valid_conf], confidence_percentile)
        else:
            cutoff = -np.inf
        conf_mask = valid_conf & (confidence[t] >= cutoff)

        human_mask = human & conf_mask & valid_depth
        obj_mask = obj & conf_mask & valid_depth
        static_mask = (~human) & (~obj) & conf_mask & valid_depth

        human_raw_mask = human_raw & valid_depth
        obj_raw_mask = obj_raw & valid_depth
        human_area_mask = human & valid_depth
        obj_area_mask = obj & valid_depth
        static_area_mask = (~human) & (~obj) & valid_depth

        human_raw_points = _sample_points(points_world[human_raw_mask], max_points_per_mask, seed=t * 5 + 1)
        object_raw_points = _sample_points(points_world[obj_raw_mask], max_points_per_mask, seed=t * 5 + 2)
        human_points = _sample_points(points_world[human_mask], max_points_per_mask, seed=t * 3 + 1)
        object_points = _sample_points(points_world[obj_mask], max_points_per_mask, seed=t * 3 + 2)
        if len(object_points) < int(object_min_points):
            object_points = np.zeros((0, 3), dtype=np.float32)
        static_points = _sample_points(points_world[static_mask], max_points_per_mask, seed=t * 3 + 3)

        np.save(dirs["human_raw_points"] / f"{t:06d}.npy", human_raw_points)
        np.save(dirs["object_raw_points"] / f"{t:06d}.npy", object_raw_points)
        np.save(dirs["human_points"] / f"{t:06d}.npy", human_points)
        np.save(dirs["object_points"] / f"{t:06d}.npy", object_points)
        np.save(dirs["static_points"] / f"{t:06d}.npy", static_points)
        _write_mask(dirs["human_masks"] / f"{t:06d}.png", human_mask)
        _write_mask(dirs["object_masks"] / f"{t:06d}.png", obj_mask)
        _write_mask(dirs["confidence_masks"] / f"{t:06d}.png", conf_mask)

        counts["human"].append(int(len(human_points)))
        counts["object"].append(int(len(object_points)))
        counts["static"].append(int(len(static_points)))
        raw_counts["human"].append(int(len(human_raw_points)))
        raw_counts["object"].append(int(len(object_raw_points)))
        mask_area_px["human"].append(int(np.count_nonzero(human_area_mask)))
        mask_area_px["object"].append(int(np.count_nonzero(obj_area_mask)))
        mask_area_px["static"].append(int(np.count_nonzero(static_area_mask)))
        confidence_mean["human"].append(_confidence_mean(human_area_mask, confidence[t]))
        confidence_mean["object"].append(_confidence_mean(obj_area_mask, confidence[t]))
        confidence_mean["static"].append(_confidence_mean(static_area_mask, confidence[t]))

    def _quality_weights(kind: str) -> list[float]:
        if kind == "object":
            return [
                _object_quality_weight(int(n), int(object_min_points), int(object_normal_points))
                for n in counts["object"]
            ]
        areas = np.asarray(mask_area_px[kind], dtype=np.float32)
        conf = np.asarray(confidence_mean[kind], dtype=np.float32)
        if areas.size == 0:
            return []
        area_norm = np.sqrt(areas / max(float(np.max(areas)), 1.0))
        valid_conf = conf[np.isfinite(conf) & (conf > 0)]
        conf_ref = float(np.median(valid_conf)) if valid_conf.size else 1.0
        conf_norm = np.clip(conf / max(conf_ref, 1e-6), 0.05, 2.0)
        weights = area_norm * conf_norm
        weights[areas <= 0] = 0.0
        return weights.astype(float).tolist()

    metadata = {
        "aligned_dir": str(aligned_dir),
        "masks_cache_file": str(masks_cache_file),
        "num_frames": int(T),
        "frame_count": int(T),
        "shape": [int(H), int(W)],
        "confidence_percentile": float(confidence_percentile),
        "erode_pixels": int(erode_pixels),
        "object_erosion_mode": "adaptive_by_mask_area",
        "object_erosion_px": object_erosion_px,
        "max_points_per_mask": int(max_points_per_mask),
        "object_min_points": int(object_min_points),
        "object_normal_points": int(object_normal_points),
        "counts": counts,
        "raw_counts": raw_counts,
        "object_raw_point_count": raw_counts["object"],
        "object_filtered_point_count": counts["object"],
        "mask_area_px": mask_area_px,
        "confidence_mean": confidence_mean,
        "quality_weights": {
            "human": _quality_weights("human"),
            "object": _quality_weights("object"),
            "static": _quality_weights("static"),
        },
        "coordinate_convention": "All saved points are in Blender metric world coordinates B.",
    }
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract VGGT dynamic-camera observations")
    parser.add_argument("--aligned_dir", required=True)
    parser.add_argument("--masks_cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--object_min_points", type=int, default=10)
    parser.add_argument("--object_normal_points", type=int, default=50)
    parser.add_argument("--skip_done", action="store_true")
    args = parser.parse_args()
    extract_vggt_observations(
        args.aligned_dir,
        args.masks_cache,
        args.output_dir,
        object_min_points=args.object_min_points,
        object_normal_points=args.object_normal_points,
        skip_done=args.skip_done,
    )


if __name__ == "__main__":
    main()
