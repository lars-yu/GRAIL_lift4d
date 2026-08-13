"""Prepare GRAIL videos for CameraAnything auxiliary-view generation.

This is intentionally a standalone entry point. It does not register a new
GRAIL pipeline step; it packages the artifacts produced after gen_2dhoi video
generation into CameraAnything's existing caption/camera JSON format and saves
the exact command that can run inference.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import pickle
import shlex
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cameraanything" / "aux_view.yaml"
WAN21_14B_WEIGHT_FILES = [
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
    "models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
]


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _read_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg(cfg: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    value = cfg.get(section, {}).get(key, default)
    return default if value is None else value


def _arg_or_cfg(arg_value: Any, cfg: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    return arg_value if arg_value is not None else _cfg(cfg, section, key, default)


def _resolve_path(value: str | Path | None, *, base: Path = PROJECT_ROOT) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _resolve_under_results(results_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return results_dir / path


def _strip_mp4(video_id: str) -> str:
    return video_id[:-4] if video_id.endswith(".mp4") else video_id


def _sanitize_case_key(video_id: str) -> str:
    keep = []
    for ch in _strip_mp4(video_id):
        keep.append(ch if ch.isalnum() or ch in ("-", "_") else "__")
    out = "".join(keep).strip("_")
    while "____" in out:
        out = out.replace("____", "__")
    return out or "grail_video"


def _video_meta(video_path: Path) -> dict[str, Any]:
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if frame_count > 0 and width > 0 and height > 0:
                return {"fps": fps, "frame_count": frame_count, "width": width, "height": height}
        cap.release()

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"Could not read video metadata for {video_path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found in {video_path}")
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _rate_to_float(stream.get("avg_frame_rate")) or _rate_to_float(stream.get("r_frame_rate"))
    frame_count = int(stream.get("nb_frames") or 0)
    if frame_count <= 0 and fps > 0 and stream.get("duration"):
        frame_count = int(round(float(stream["duration"]) * fps))
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid video metadata for {video_path}")
    return {"fps": fps, "frame_count": frame_count, "width": width, "height": height}


def _rate_to_float(rate: str | None) -> float:
    if not rate:
        return 0.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_float = float(den)
        return 0.0 if den_float == 0.0 else float(num) / den_float
    return float(rate)


def _read_render_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    for key in ("cam_R", "cam_t", "obj_t"):
        if key not in data:
            raise KeyError(f"{path} missing required key: {key}")
    return data


def _read_k(path: Path) -> np.ndarray:
    K = np.loadtxt(path, dtype=np.float64).reshape(3, 3)
    if K.shape != (3, 3):
        raise ValueError(f"Invalid K shape from {path}: {K.shape}")
    if abs(float(K[2, 2]) - 1.0) > 1e-6:
        raise ValueError(f"Invalid K[2,2] from {path}: {K[2,2]}")
    return K


def _validate_c2w(name: str, mat: np.ndarray, atol: float = 1e-4) -> None:
    if mat.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {mat.shape}")
    if not np.allclose(mat[3], np.array([0, 0, 0, 1], dtype=mat.dtype), atol=atol):
        raise ValueError(f"{name} last row is not homogeneous: {mat[3]}")
    R = mat[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        raise ValueError(f"{name} rotation is not orthonormal")
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > 1e-3:
        raise ValueError(f"{name} rotation determinant is {det:.6f}, expected 1")


def _blender_c2w_to_opencv_c2w(cam_R: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
    """Convert Blender camera-to-world axes to OpenCV-style camera-to-world.

    Blender camera local axes are X right, Y up, -Z forward. CameraAnything's
    ray code uses OpenCV-like camera axes: X right, Y down, +Z forward.
    """
    c2w = np.eye(4, dtype=np.float64)
    blender_to_opencv = np.diag([1.0, -1.0, -1.0])
    c2w[:3, :3] = np.asarray(cam_R, dtype=np.float64).reshape(3, 3) @ blender_to_opencv
    c2w[:3, 3] = np.asarray(cam_t, dtype=np.float64).reshape(3)
    _validate_c2w("source_c2w_opencv", c2w)
    return c2w


def _grail_center_look_at(render_data: dict[str, Any]) -> np.ndarray:
    """Match GRAIL/Blender camera target selection.

    Blender rendering targets the midpoint of object and character locations,
    then keeps the target height at the object's z location.
    """
    obj_t = np.asarray(render_data["obj_t"], dtype=np.float64).reshape(3)
    char_t = render_data.get("character_t")
    if char_t is None:
        return obj_t
    char_t = np.asarray(char_t, dtype=np.float64).reshape(3)
    center = 0.5 * (obj_t + char_t)
    center[2] = obj_t[2]
    return center


def _auto_look_at(render_data: dict[str, Any]) -> np.ndarray:
    return _grail_center_look_at(render_data)


def _look_at_opencv_c2w(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        raise ValueError("Target camera eye equals look-at point")
    forward = forward / norm

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    _validate_c2w("target_c2w_opencv", c2w)
    return c2w


def _orbit_camera_c2w(
    *,
    look_at: np.ndarray,
    elevation_deg: float,
    azimuth_deg: float,
    radius: float,
) -> np.ndarray:
    elevation = math.radians(float(elevation_deg))
    azimuth = math.radians(float(azimuth_deg))
    offset = np.array(
        [
            radius * math.cos(elevation) * math.cos(azimuth),
            radius * math.cos(elevation) * math.sin(azimuth),
            radius * math.sin(elevation),
        ],
        dtype=np.float64,
    )
    eye = look_at + offset
    return _look_at_opencv_c2w(eye, look_at)


def _orbit_offset(
    *,
    elevation_deg: float,
    azimuth_deg: float,
    radius: float,
    camera_offset: np.ndarray | None = None,
) -> np.ndarray:
    elevation = math.radians(float(elevation_deg))
    azimuth = math.radians(float(azimuth_deg))
    offset = np.array(
        [
            radius * math.cos(elevation) * math.cos(azimuth),
            radius * math.cos(elevation) * math.sin(azimuth),
            radius * math.sin(elevation),
        ],
        dtype=np.float64,
    )
    if camera_offset is not None:
        offset = offset + np.asarray(camera_offset, dtype=np.float64).reshape(3)
    return offset


def _look_at_from_source_orbit(
    source_abs_c2w: np.ndarray,
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    source_elevation = _arg_or_cfg(
        args.source_elevation_deg, cfg, "view", "source_elevation_deg"
    )
    source_azimuth = _arg_or_cfg(
        args.source_azimuth_deg, cfg, "view", "source_azimuth_deg"
    )
    source_radius = _arg_or_cfg(args.source_radius, cfg, "view", "source_radius")
    if source_elevation is None or source_azimuth is None or source_radius is None:
        return None, None

    source_camera_offset = (
        args.source_camera_offset
        if args.source_camera_offset is not None
        else _cfg(cfg, "view", "source_camera_offset", [0.0, 0.0, 0.0])
    )
    offset = _orbit_offset(
        elevation_deg=float(source_elevation),
        azimuth_deg=float(source_azimuth),
        radius=float(source_radius),
        camera_offset=np.asarray(source_camera_offset, dtype=np.float64).reshape(3),
    )
    look_at = source_abs_c2w[:3, 3] - offset
    return look_at, {
        "method": "source_camera_orbit",
        "source_elevation_deg": float(source_elevation),
        "source_azimuth_deg": float(source_azimuth),
        "source_radius": float(source_radius),
        "source_camera_offset": np.asarray(source_camera_offset, dtype=np.float64).reshape(3).tolist(),
    }


def _hfov_from_k(K: np.ndarray, width: int) -> float:
    fx = float(K[0, 0])
    if fx <= 0:
        raise ValueError(f"Invalid fx: {fx}")
    return math.degrees(2.0 * math.atan(float(width) / (2.0 * fx)))


def _transformed_k_for_resize_crop(
    K: np.ndarray,
    *,
    orig_height: int,
    orig_width: int,
    final_height: int,
    final_width: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mirror CameraAnything's resize-to-cover + center-crop K transform."""
    scale = max(float(final_width) / float(orig_width), float(final_height) / float(orig_height))
    resize_height = int(math.ceil(float(orig_height) * scale))
    resize_width = int(math.ceil(float(orig_width) * scale))
    if resize_width < final_width or resize_height < final_height:
        raise ValueError(
            f"Invalid resize/crop sizes: resize=({resize_height},{resize_width}) "
            f"final=({final_height},{final_width})"
        )

    scale_x = float(resize_width) / float(orig_width)
    scale_y = float(resize_height) / float(orig_height)
    crop_offset_x = (resize_width - int(final_width)) // 2
    crop_offset_y = (resize_height - int(final_height)) // 2

    out = np.zeros_like(K, dtype=np.float64)
    out[0, 0] = K[0, 0] * scale_x
    out[1, 1] = K[1, 1] * scale_y
    out[0, 2] = K[0, 2] * scale_x - crop_offset_x
    out[1, 2] = K[1, 2] * scale_y - crop_offset_y
    out[2, 2] = 1.0
    return out, {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "resize_height": resize_height,
        "resize_width": resize_width,
        "crop_offset_x": crop_offset_x,
        "crop_offset_y": crop_offset_y,
    }


def _project_world_point(c2w: np.ndarray, K: np.ndarray, point: np.ndarray) -> dict[str, Any]:
    w2c = np.linalg.inv(c2w)
    point = np.asarray(point, dtype=np.float64).reshape(3)
    point_cam = w2c[:3, :3] @ point + w2c[:3, 3]
    depth = float(point_cam[2])
    if abs(depth) < 1e-8:
        uv = [None, None]
    else:
        uv_arr = (K @ point_cam)[:2] / depth
        uv = [float(uv_arr[0]), float(uv_arr[1])]
    return {
        "uv": uv,
        "depth": depth,
        "in_front": depth > 0.0,
    }


def _round_nested(value: Any, ndigits: int = 6) -> Any:
    if isinstance(value, dict):
        return {k: _round_nested(v, ndigits=ndigits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_nested(v, ndigits=ndigits) for v in value]
    if isinstance(value, tuple):
        return [_round_nested(v, ndigits=ndigits) for v in value]
    if isinstance(value, np.ndarray):
        return _round_nested(value.tolist(), ndigits=ndigits)
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            return None
        return round(float(value), ndigits)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _with_frame_check(proj: dict[str, Any], *, width: int, height: int) -> dict[str, Any]:
    out = dict(proj)
    uv = out.get("uv")
    if uv is None or uv[0] is None or uv[1] is None:
        out["inside_frame"] = False
    else:
        out["inside_frame"] = bool(
            0.0 <= float(uv[0]) <= float(width) and 0.0 <= float(uv[1]) <= float(height)
        )
    return out


def _build_camera_validation(
    *,
    render_data: dict[str, Any],
    K: np.ndarray,
    source_abs_c2w: np.ndarray,
    target_abs_c2w: np.ndarray,
    source_ca_c2w: np.ndarray,
    target_ca_c2w: np.ndarray,
    look_at: np.ndarray,
    look_at_method: dict[str, Any],
    meta: dict[str, Any],
    cond_height: int,
    cond_width: int,
    target_height: int,
    target_width: int,
    task_name: str,
) -> dict[str, Any]:
    points = {
        "look_at": np.asarray(look_at, dtype=np.float64).reshape(3),
        "object": np.asarray(render_data["obj_t"], dtype=np.float64).reshape(3),
    }
    if render_data.get("character_t") is not None:
        points["character"] = np.asarray(render_data["character_t"], dtype=np.float64).reshape(3)

    source_K_cond, source_transform = _transformed_k_for_resize_crop(
        K,
        orig_height=int(meta["height"]),
        orig_width=int(meta["width"]),
        final_height=int(cond_height),
        final_width=int(cond_width),
    )
    if task_name in {"resolution_only", "multi-shot-focal-resolution"}:
        ca_target_height = int(target_width)
        ca_target_width = int(target_height)
    else:
        ca_target_height = int(target_height)
        ca_target_width = int(target_width)
    target_K_out, target_transform = _transformed_k_for_resize_crop(
        K,
        orig_height=int(meta["height"]),
        orig_width=int(meta["width"]),
        final_height=ca_target_height,
        final_width=ca_target_width,
    )

    def project_all(c2w: np.ndarray, proj_K: np.ndarray, width: int, height: int) -> dict[str, Any]:
        return {
            name: _with_frame_check(
                _project_world_point(c2w, proj_K, point),
                width=width,
                height=height,
            )
            for name, point in points.items()
        }

    validation = {
        "conclusion": "camera_parameters_self_consistent",
        "look_at_method": look_at_method,
        "points_world": points,
        "source_pose": {
            "opencv_c2w_absolute": source_abs_c2w,
            "cameraanything_c2w": source_ca_c2w,
        },
        "target_pose": {
            "opencv_c2w_absolute": target_abs_c2w,
            "cameraanything_c2w": target_ca_c2w,
        },
        "original_intrinsics": {
            "K": K,
            "width": int(meta["width"]),
            "height": int(meta["height"]),
            "hfov_deg": _hfov_from_k(K, int(meta["width"])),
        },
        "cameraanything_intrinsics": {
            "condition_canvas": {
                "height": int(cond_height),
                "width": int(cond_width),
                "K": source_K_cond,
                "resize_crop": source_transform,
            },
            "target_canvas": {
                "height": ca_target_height,
                "width": ca_target_width,
                "K": target_K_out,
                "resize_crop": target_transform,
            },
        },
        "projections": {
            "original_source_canvas": project_all(
                source_abs_c2w, K, int(meta["width"]), int(meta["height"])
            ),
            "original_target_canvas": project_all(
                target_abs_c2w, K, int(meta["width"]), int(meta["height"])
            ),
            "cameraanything_condition_canvas": project_all(
                source_abs_c2w, source_K_cond, int(cond_width), int(cond_height)
            ),
            "cameraanything_target_canvas": project_all(
                target_abs_c2w, target_K_out, ca_target_width, ca_target_height
            ),
        },
    }
    return _round_nested(validation)


def _repeat_mat(mat: np.ndarray, frames: int) -> np.ndarray:
    return np.repeat(mat[None, :, :], int(frames), axis=0)


def _safe_link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _quote_cmd(cmd: list[str], cwd: Path | None = None) -> str:
    body = " ".join(shlex.quote(str(x)) for x in cmd)
    if cwd is None:
        return body
    return f"cd {shlex.quote(str(cwd))}\n{body}\n"


def _load_prompt(prompt_path: Path | None, suffix: str) -> str:
    prompt = ""
    if prompt_path is not None and prompt_path.is_file():
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    if suffix:
        prompt = f"{prompt} {suffix.strip()}".strip()
    return prompt or "A person interacts with an object in a static indoor scene."


def _build_camera_json(
    *,
    case_key: str,
    task_name: str,
    source_c2w: np.ndarray,
    target_c2w: np.ndarray,
    hfov: float,
    frame_count: int,
) -> dict[str, Any]:
    data = {}
    source = {"c2w": source_c2w.tolist(), "hfov": float(hfov)}
    target = {"c2w": target_c2w.tolist(), "hfov": float(hfov)}
    for idx in range(int(frame_count)):
        data[f"frame{idx}"] = {case_key: source, task_name: target}
    return data


def _expected_cameraanything_pred(output_dir: Path, case_key: str, task_name: str) -> Path:
    return output_dir / case_key / f"{case_key}_{task_name}_pred.mp4"


def _file_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _cameraanything_weight_status(repo_dir: Path, checkpoint: Path) -> dict[str, Any]:
    files = {"cameraanything_checkpoint": _file_status(checkpoint)}
    for rel_path in WAN21_14B_WEIGHT_FILES:
        files[rel_path] = _file_status(repo_dir / rel_path)
    missing = [name for name, info in files.items() if not info["exists"]]
    return {
        "generator": "CameraAnything inference.py",
        "base_model": "Wan-AI/Wan2.1-T2V-14B",
        "camera_adapter_checkpoint": str(checkpoint),
        "files": files,
        "missing": missing,
        "ready_for_inference": not missing,
    }


def _process_one(video_id: str | None, args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    results_dir = _resolve_path(
        args.results_dir or _cfg(cfg, "paths", "results_dir", "results")
    )
    assert results_dir is not None
    video_dir = args.video_dir or _cfg(cfg, "paths", "video_dir", "generation/videos_wan")
    prompt_dir = args.prompt_dir or _cfg(cfg, "paths", "prompt_dir", "generation/prompts")
    fp_dir = args.foundation_pose_dir or _cfg(
        cfg, "paths", "foundation_pose_dir", "generation/foundation_pose"
    )
    out_root = _resolve_under_results(
        results_dir, args.output_dir or _cfg(cfg, "paths", "output_dir", "generation/cameraanything_aux")
    )

    direct_source = _resolve_path(args.source_video) if args.source_video else None
    if video_id is None and direct_source is None:
        raise ValueError("Provide --video_id or --source_video")
    video_id_clean = _strip_mp4(video_id) if video_id else direct_source.stem

    source_video = direct_source or (results_dir / video_dir / f"{video_id_clean}.mp4")
    if not source_video.is_file():
        raise FileNotFoundError(f"Source video not found: {source_video}")

    render_config = (
        _resolve_path(args.render_config)
        if args.render_config
        else results_dir / fp_dir / video_id_clean / "first_frame_pose.pickle"
    )
    cam_k_path = (
        _resolve_path(args.cam_k)
        if args.cam_k
        else results_dir / fp_dir / video_id_clean / "cam_K.txt"
    )
    prompt_path = (
        _resolve_path(args.prompt_path)
        if args.prompt_path
        else results_dir / prompt_dir / f"{video_id_clean}.txt"
    )
    if not render_config.is_file():
        raise FileNotFoundError(f"Render camera pickle not found: {render_config}")
    if not cam_k_path.is_file():
        raise FileNotFoundError(f"Camera intrinsics not found: {cam_k_path}")

    view_name = args.view_name or _cfg(cfg, "view", "name", "oblique_45")
    aux_dir = out_root / video_id_clean / view_name
    manifest_path = aux_dir / "manifest.json"
    run_requested = bool(args.run or _cfg(cfg, "io", "run", False))
    overwrite = bool(args.overwrite or _cfg(cfg, "io", "overwrite", False))
    if manifest_path.is_file() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not run_requested or manifest.get("generated"):
            print(f"Skip existing package: {aux_dir}")
            return aux_dir
    if aux_dir.exists() and overwrite:
        shutil.rmtree(aux_dir)

    started_at = _now()
    input_dir = aux_dir / "input"
    camera_dir = aux_dir / "cameras"
    ca_output_dir = aux_dir / "cameraanything_output"
    final_output_dir = aux_dir / "output"
    log_dir = aux_dir / "logs"
    for d in (input_dir, camera_dir, ca_output_dir, final_output_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    meta = _video_meta(source_video)
    render_data = _read_render_pickle(render_config)
    K = _read_k(cam_k_path)
    source_abs_c2w = _blender_c2w_to_opencv_c2w(render_data["cam_R"], render_data["cam_t"])

    look_at_method = {"method": "explicit"}
    look_at_cfg = args.look_at if args.look_at is not None else _cfg(cfg, "view", "look_at")
    if look_at_cfg is None:
        look_at_mode = str(
            args.look_at_mode or _cfg(cfg, "view", "look_at_mode", "source_orbit")
        )
        if look_at_mode == "grail_center":
            look_at = _grail_center_look_at(render_data)
            look_at_method = {"method": "grail_object_character_center"}
        elif look_at_mode == "source_orbit":
            look_at, look_at_method = _look_at_from_source_orbit(source_abs_c2w, cfg, args)
            if look_at is None:
                look_at = _auto_look_at(render_data)
                look_at_method = {"method": "saved_object_character_center"}
        else:
            raise ValueError(
                f"Unknown look_at_mode={look_at_mode!r}; expected 'source_orbit' or 'grail_center'"
            )
    else:
        look_at = np.asarray(look_at_cfg, dtype=np.float64).reshape(3)

    source_radius = float(np.linalg.norm(source_abs_c2w[:3, 3] - look_at))
    radius_cfg = args.target_radius if args.target_radius is not None else _cfg(cfg, "view", "radius")
    target_radius = float(radius_cfg) if radius_cfg is not None else source_radius
    target_elevation_deg = float(_arg_or_cfg(args.target_elevation_deg, cfg, "view", "elevation_deg", 45.0))
    target_azimuth_deg = float(_arg_or_cfg(args.target_azimuth_deg, cfg, "view", "azimuth_deg", 300.0))
    target_abs_c2w = _orbit_camera_c2w(
        look_at=look_at,
        elevation_deg=target_elevation_deg,
        azimuth_deg=target_azimuth_deg,
        radius=target_radius,
    )

    normalize = bool(_cfg(cfg, "view", "normalize_to_source_camera", True))
    if args.no_normalize_to_source:
        normalize = False
    if normalize:
        source_to_world = source_abs_c2w
        world_to_source = np.linalg.inv(source_to_world)
        source_ca_c2w = world_to_source @ source_abs_c2w
        target_ca_c2w = world_to_source @ target_abs_c2w
    else:
        source_ca_c2w = source_abs_c2w
        target_ca_c2w = target_abs_c2w
    _validate_c2w("cameraanything_source_c2w", source_ca_c2w)
    _validate_c2w("cameraanything_target_c2w", target_ca_c2w)

    hfov = _hfov_from_k(K, meta["width"])
    source_frames = meta["frame_count"]
    camera_json_frames = max(
        source_frames,
        int(_arg_or_cfg(args.camera_json_frames, cfg, "cameraanything", "num_frames", 81)),
    )
    case_key = args.case_key or _sanitize_case_key(video_id_clean)
    task_name = args.task_name or _cfg(cfg, "cameraanything", "task_name", "multi-shot-only_1")

    copy_source = bool(args.copy_source_video or _cfg(cfg, "io", "copy_source_video", False))
    packaged_source = input_dir / "source.mp4"
    _safe_link_or_copy(source_video, packaged_source, copy=copy_source)

    prompt_suffix = args.prompt_suffix
    if prompt_suffix is None:
        prompt_suffix = _cfg(cfg, "prompt", "suffix", "")
    caption = _load_prompt(prompt_path, prompt_suffix)
    caption_json = {case_key: {"path": str(packaged_source.absolute()), "caption": caption}}
    caption_path = input_dir / "caption.json"
    _write_json(caption_path, caption_json)

    camera_json = _build_camera_json(
        case_key=case_key,
        task_name=task_name,
        source_c2w=source_ca_c2w,
        target_c2w=target_ca_c2w,
        hfov=hfov,
        frame_count=camera_json_frames,
    )
    camera_json_path = camera_dir / "cameraanything_camera.json"
    _write_json(camera_json_path, camera_json)

    np.save(camera_dir / "source_c2w_absolute.npy", _repeat_mat(source_abs_c2w, source_frames))
    np.save(camera_dir / "target_c2w_absolute.npy", _repeat_mat(target_abs_c2w, source_frames))
    np.save(camera_dir / "cameraanything_source_c2w.npy", _repeat_mat(source_ca_c2w, camera_json_frames))
    np.save(camera_dir / "cameraanything_target_c2w.npy", _repeat_mat(target_ca_c2w, camera_json_frames))
    np.save(camera_dir / "source_K.npy", K)
    np.save(camera_dir / "target_K.npy", K.copy())

    repo_dir = _resolve_path(args.cameraanything_repo or _cfg(cfg, "cameraanything", "repo_dir"))
    if repo_dir is None:
        raise ValueError("CameraAnything repo_dir is required")
    inference_script = args.inference_script or _cfg(cfg, "cameraanything", "inference_script", "inference.py")
    script_path = _resolve_path(inference_script, base=repo_dir)
    checkpoint = _resolve_path(args.checkpoint or _cfg(cfg, "cameraanything", "checkpoint"), base=repo_dir)
    if checkpoint is None:
        raise ValueError("CameraAnything checkpoint is required")
    weight_status = _cameraanything_weight_status(repo_dir, checkpoint)
    python_exe = args.python_executable or _cfg(cfg, "cameraanything", "python_executable", "python3")
    cond_height = int(_arg_or_cfg(args.cond_height, cfg, "cameraanything", "cond_height", meta["height"]))
    cond_width = int(_arg_or_cfg(args.cond_width, cfg, "cameraanything", "cond_width", meta["width"]))
    target_height = int(_arg_or_cfg(args.target_height, cfg, "cameraanything", "target_height", meta["height"]))
    target_width = int(_arg_or_cfg(args.target_width, cfg, "cameraanything", "target_width", meta["width"]))
    camera_validation = _build_camera_validation(
        render_data=render_data,
        K=K,
        source_abs_c2w=source_abs_c2w,
        target_abs_c2w=target_abs_c2w,
        source_ca_c2w=source_ca_c2w,
        target_ca_c2w=target_ca_c2w,
        look_at=look_at,
        look_at_method=look_at_method,
        meta=meta,
        cond_height=cond_height,
        cond_width=cond_width,
        target_height=target_height,
        target_width=target_width,
        task_name=task_name,
    )
    camera_validation_path = camera_dir / "validation_camera_parameters.json"
    _write_json(camera_validation_path, camera_validation)

    command = [
        python_exe,
        str(script_path),
        "--real_case_json",
        str(caption_path.resolve()),
        "--camera_real_json",
        str(camera_json_path.resolve()),
        "--ckpt_path",
        str(checkpoint),
        "--output_dir",
        str(ca_output_dir.resolve()),
        "--tasks",
        task_name,
        "--cam_type",
        str(args.cam_type or _cfg(cfg, "cameraanything", "cam_type", "pluc_adaln")),
        "--cond_height",
        str(cond_height),
        "--cond_width",
        str(cond_width),
        "--target_height",
        str(target_height),
        "--target_width",
        str(target_width),
        "--cfg_scale",
        str(args.cfg_scale if args.cfg_scale is not None else _cfg(cfg, "cameraanything", "cfg_scale", 5.0)),
        "--start_frame_id",
        str(args.start_frame_id if args.start_frame_id is not None else _cfg(cfg, "cameraanything", "start_frame_id", 0)),
        "--num_inference_steps",
        str(args.num_inference_steps if args.num_inference_steps is not None else _cfg(cfg, "cameraanything", "num_inference_steps", 50)),
        "--seed",
        str(args.seed if args.seed is not None else _cfg(cfg, "cameraanything", "seed", 0)),
        "--dataloader_num_workers",
        str(args.dataloader_num_workers if args.dataloader_num_workers is not None else _cfg(cfg, "cameraanything", "dataloader_num_workers", 1)),
        "--shard_id",
        "0",
        "--num_shards",
        "1",
    ]

    request = {
        "argv": sys.argv,
        "config_path": str(args.config.resolve()) if args.config else None,
        "video_id": video_id_clean,
        "case_key": case_key,
        "task_name": task_name,
        "source_video": str(source_video.resolve()),
        "render_config": str(render_config.resolve()),
        "cam_K": str(cam_k_path.resolve()),
        "prompt_path": str(prompt_path.resolve()) if prompt_path is not None else None,
    }
    _write_json(aux_dir / "request.json", request)
    _write_json(
        aux_dir / "effective_config.json",
        {
            "paths": {
                "results_dir": str(results_dir),
                "video_dir": video_dir,
                "prompt_dir": prompt_dir,
                "foundation_pose_dir": fp_dir,
                "output_dir": str(out_root),
            },
            "cameraanything": {
                "repo_dir": str(repo_dir),
                "python_executable": python_exe,
                "inference_script": str(script_path),
                "checkpoint": str(checkpoint),
                "weights": weight_status,
                "task_name": task_name,
                "cond_height": cond_height,
                "cond_width": cond_width,
                "target_height": target_height,
                "target_width": target_width,
            },
            "view": {
                "name": view_name,
                "look_at": look_at.tolist(),
                "look_at_method": look_at_method,
                "elevation_deg": target_elevation_deg,
                "azimuth_deg": target_azimuth_deg,
                "radius": target_radius,
                "normalize_to_source_camera": normalize,
            },
            "camera_validation": str(camera_validation_path.resolve()),
        },
    )
    _write_json(aux_dir / "command.json", {"cwd": str(repo_dir), "command": command})
    _write_text(aux_dir / "run_cameraanything.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" + _quote_cmd(command, cwd=repo_dir))
    os.chmod(aux_dir / "run_cameraanything.sh", 0o755)

    expected_pred = _expected_cameraanything_pred(ca_output_dir, case_key, task_name)
    final_video = final_output_dir / f"{view_name}.mp4"
    status = {
        "success": True,
        "prepared": True,
        "generated": False,
        "started_at": started_at,
        "finished_at": None,
        "command": command,
        "weights": weight_status,
        "error": None,
    }

    if run_requested and weight_status["missing"]:
        status["success"] = False
        status["error"] = "Missing CameraAnything weight files: " + ", ".join(weight_status["missing"])
    elif run_requested:
        status["run_started_at"] = _now()
        result = subprocess.run(command, cwd=str(repo_dir), capture_output=True, text=True)
        _write_text(log_dir / "cameraanything_stdout.log", result.stdout)
        _write_text(log_dir / "cameraanything_stderr.log", result.stderr)
        status["returncode"] = result.returncode
        if result.returncode != 0:
            status["success"] = False
            status["error"] = f"CameraAnything failed with return code {result.returncode}"
        elif expected_pred.is_file():
            shutil.copy2(expected_pred, final_video)
            status["generated"] = True
        else:
            status["success"] = False
            status["error"] = f"Expected output not found: {expected_pred}"

    status["finished_at"] = _now()
    _write_json(aux_dir / "status.json", status)

    manifest = {
        "success": bool(status["success"]),
        "prepared": True,
        "generated": bool(status["generated"]),
        "model": "CameraAnything",
        "generation_state": "generated_by_cameraanything" if status["generated"] else "prepared_inputs_only",
        "weights": weight_status,
        "view_name": view_name,
        "video_id": video_id_clean,
        "case_key": case_key,
        "task_name": task_name,
        "source_video": str(source_video.resolve()),
        "packaged_source_video": str(packaged_source.absolute()),
        "caption_json": str(caption_path.resolve()),
        "camera_json": str(camera_json_path.resolve()),
        "command_sh": str((aux_dir / "run_cameraanything.sh").resolve()),
        "cameraanything_output_dir": str(ca_output_dir.resolve()),
        "aux_video": str(final_video.resolve()) if final_video.exists() else None,
        "source_c2w_absolute": str((camera_dir / "source_c2w_absolute.npy").resolve()),
        "target_c2w_absolute": str((camera_dir / "target_c2w_absolute.npy").resolve()),
        "cameraanything_source_c2w": str((camera_dir / "cameraanything_source_c2w.npy").resolve()),
        "cameraanything_target_c2w": str((camera_dir / "cameraanything_target_c2w.npy").resolve()),
        "source_K": str((camera_dir / "source_K.npy").resolve()),
        "target_K": str((camera_dir / "target_K.npy").resolve()),
        "camera_validation": str(camera_validation_path.resolve()),
        "source_frame_count": source_frames,
        "camera_json_frame_count": camera_json_frames,
        "fps": meta["fps"],
        "width": meta["width"],
        "height": meta["height"],
        "hfov_deg": hfov,
        "camera_convention": {
            "saved_absolute": "opencv_c2w_in_grail_world",
            "cameraanything_json": "source-normalized opencv_c2w" if normalize else "opencv_c2w_in_grail_world",
        },
        "look_at": look_at.tolist(),
        "look_at_method": look_at_method,
        "target_elevation_deg": target_elevation_deg,
        "target_azimuth_deg": target_azimuth_deg,
        "target_radius": target_radius,
        "notes": [
            "This package uses CameraAnything's existing multi-shot task slot for the target view.",
            "The imported CameraAnything inference script currently samples/saves its own frame count and FPS unless patched upstream.",
        ],
    }
    _write_json(manifest_path, manifest)
    print(f"Prepared CameraAnything package: {aux_dir}")
    if run_requested and not status["success"]:
        raise RuntimeError(status["error"])
    return aux_dir


def _discover_video_ids(args: argparse.Namespace, cfg: dict[str, Any]) -> list[str]:
    ids = list(args.video_id or [])
    if args.video_glob:
        results_dir = _resolve_path(
            args.results_dir or _cfg(cfg, "paths", "results_dir", "results")
        )
        assert results_dir is not None
        video_dir = args.video_dir or _cfg(cfg, "paths", "video_dir", "generation/videos_wan")
        base = results_dir / video_dir
        for path in sorted(base.glob(args.video_glob)):
            if path.suffix.lower() == ".mp4":
                ids.append(str(path.relative_to(base).with_suffix("")))
    if args.source_video:
        if ids:
            raise ValueError("--source_video cannot be combined with --video_id/--video_glob")
        return [None]
    return [_strip_mp4(x) for x in ids]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--video_id", nargs="*", help="GRAIL video id(s), e.g. dataset/category/name")
    p.add_argument("--video_glob", help="Glob under results_dir/video_dir, e.g. 'dataset/category/*.mp4'")
    p.add_argument("--source_video", help="Direct source video path for one-off packaging")
    p.add_argument("--render_config", help="Direct first_frame_pose.pickle path")
    p.add_argument("--cam_k", help="Direct cam_K.txt path")
    p.add_argument("--prompt_path", help="Direct prompt txt path")
    p.add_argument("--results_dir")
    p.add_argument("--video_dir")
    p.add_argument("--prompt_dir")
    p.add_argument("--foundation_pose_dir")
    p.add_argument("--output_dir")
    p.add_argument("--cameraanything_repo")
    p.add_argument("--python_executable")
    p.add_argument("--inference_script")
    p.add_argument("--checkpoint")
    p.add_argument("--view_name")
    p.add_argument("--target_elevation_deg", type=float)
    p.add_argument("--target_azimuth_deg", type=float)
    p.add_argument("--target_radius", type=float)
    p.add_argument("--source_elevation_deg", type=float)
    p.add_argument("--source_azimuth_deg", type=float)
    p.add_argument("--source_radius", type=float)
    p.add_argument("--source_camera_offset", type=float, nargs=3)
    p.add_argument("--look_at_mode", choices=["source_orbit", "grail_center"])
    p.add_argument("--look_at", type=float, nargs=3)
    p.add_argument("--task_name")
    p.add_argument("--case_key")
    p.add_argument("--prompt_suffix")
    p.add_argument("--cond_height", type=int)
    p.add_argument("--cond_width", type=int)
    p.add_argument("--target_height", type=int)
    p.add_argument("--target_width", type=int)
    p.add_argument("--camera_json_frames", type=int)
    p.add_argument("--cam_type")
    p.add_argument("--cfg_scale", type=float)
    p.add_argument("--start_frame_id", type=int)
    p.add_argument("--num_inference_steps", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--dataloader_num_workers", type=int)
    p.add_argument("--copy_source_video", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--run", action="store_true", help="Run CameraAnything after preparing files")
    p.add_argument("--no_normalize_to_source", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _read_yaml(args.config)
    try:
        video_ids = _discover_video_ids(args, cfg)
        if not video_ids:
            raise ValueError("No videos selected. Use --video_id, --video_glob, or --source_video.")
        outputs = [_process_one(video_id, args, cfg) for video_id in video_ids]
        print("Prepared packages:")
        for path in outputs:
            print(f"  {path}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if _cfg(cfg, "io", "verbose_errors", False):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
