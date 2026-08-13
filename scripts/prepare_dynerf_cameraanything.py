"""Prepare a DyNeRF/LLFF scene pair for CameraAnything inference.

The DyNeRF data under ~/datasets/dynerf stores one static camera video per
`camXX.mp4` and a LLFF-style `poses_bounds.npy`. This script converts one
source camera and one target camera into CameraAnything's `caption.json` and
`camera.json` format.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERAANYTHING_ROOT = PROJECT_ROOT / "imports" / "CameraAnything"
DEFAULT_PYTHON = "/home/jiaoyufei_insta360.com/miniconda3/envs/Grail/bin/python"


def _read_video_meta(path: Path) -> dict[str, Any]:
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            meta = {
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            cap.release()
            if meta["frame_count"] and meta["width"] and meta["height"]:
                return meta
        cap.release()
    raise RuntimeError(f"Could not read video metadata: {path}")


def _load_llff_poses(scene_dir: Path) -> tuple[list[str], np.ndarray]:
    videos = sorted(p.stem for p in scene_dir.glob("cam*.mp4"))
    poses_bounds = np.load(scene_dir / "poses_bounds.npy")
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    if len(videos) != poses.shape[0]:
        raise ValueError(
            f"Found {len(videos)} cam*.mp4 files but {poses.shape[0]} poses in poses_bounds.npy"
        )
    return videos, poses


def _llff_pose_to_opencv_c2w(pose_3x5: np.ndarray) -> np.ndarray:
    """Convert LLFF/DyNeRF camera axes to CameraAnything c2w axes.

    LLFF `poses_bounds.npy` stores camera-to-world poses with columns
    `[down, right, back, translation, HWF]`. CameraAnything's ray code builds
    rays as OpenCV-like camera directions `[x_right, y_down, z_forward]`, so the
    corresponding c2w rotation is `[right, down, -back]`.
    """
    pose = np.asarray(pose_3x5[:, :4], dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = pose[:3, 1]
    c2w[:3, 1] = pose[:3, 0]
    c2w[:3, 2] = -pose[:3, 2]
    c2w[:3, 3] = pose[:3, 3]
    return c2w


def _hfov_from_focal(width: float, focal: float) -> float:
    return math.degrees(2.0 * math.atan(float(width) / (2.0 * float(focal))))


def _sanitize_case_key(scene: str, source_cam: str, target_cam: str) -> str:
    raw = f"dynerf_{scene}_{source_cam}_to_{target_cam}"
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _quote_cmd(cmd: list[str], cwd: Path) -> str:
    return "cd " + shlex.quote(str(cwd)) + "\n" + " ".join(shlex.quote(x) for x in cmd) + "\n"


def prepare(args: argparse.Namespace) -> Path:
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    scene_name = args.scene_name or scene_dir.name
    videos, poses = _load_llff_poses(scene_dir)
    if args.source_cam not in videos:
        raise ValueError(f"Unknown source cam {args.source_cam}; available={videos}")
    if args.target_cam not in videos:
        raise ValueError(f"Unknown target cam {args.target_cam}; available={videos}")

    src_idx = videos.index(args.source_cam)
    tgt_idx = videos.index(args.target_cam)
    source_video = scene_dir / f"{args.source_cam}.mp4"
    source_meta = _read_video_meta(source_video)

    source_abs = _llff_pose_to_opencv_c2w(poses[src_idx])
    target_abs = _llff_pose_to_opencv_c2w(poses[tgt_idx])
    world_to_source = np.linalg.inv(source_abs)
    source_ca = world_to_source @ source_abs
    target_ca = world_to_source @ target_abs

    hwf = poses[src_idx, :, 4]
    pose_h, pose_w, focal = float(hwf[0]), float(hwf[1]), float(hwf[2])
    if abs(pose_h - source_meta["height"]) > 1 or abs(pose_w - source_meta["width"]) > 1:
        raise ValueError(
            f"Pose H/W ({pose_h},{pose_w}) does not match source video "
            f"({source_meta['height']},{source_meta['width']})"
        )
    hfov = _hfov_from_focal(source_meta["width"], focal)
    case_key = args.case_key or _sanitize_case_key(scene_name, args.source_cam, args.target_cam)
    task_name = args.task_name

    out_dir = Path(args.output_dir).expanduser().resolve()
    input_dir = out_dir / "input"
    output_dir = out_dir / "cameraanything_output"
    final_dir = out_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    caption = args.caption or (
        "A real indoor scene with a person preparing a drink at a counter. "
        "Preserve the same person, objects, room layout and motion while changing only camera position."
    )
    caption_json = {case_key: {"path": str(source_video), "caption": caption}}
    caption_path = input_dir / "caption.json"
    _write_json(caption_path, caption_json)

    frame_count = min(source_meta["frame_count"], int(args.num_frames))
    camera_json = {}
    source_entry = {"c2w": source_ca.tolist(), "hfov": hfov}
    target_entry = {"c2w": target_ca.tolist(), "hfov": hfov}
    for i in range(frame_count):
        camera_json[f"frame{i}"] = {case_key: source_entry, task_name: target_entry}
    camera_path = input_dir / "camera.json"
    _write_json(camera_path, camera_json)

    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = CAMERAANYTHING_ROOT / checkpoint
    command = [
        args.python_executable,
        str(CAMERAANYTHING_ROOT / "inference.py"),
        "--real_case_json",
        str(caption_path),
        "--camera_real_json",
        str(camera_path),
        "--ckpt_path",
        str(checkpoint),
        "--output_dir",
        str(output_dir),
        "--tasks",
        task_name,
        "--cam_type",
        args.cam_type,
        "--cond_height",
        str(args.cond_height),
        "--cond_width",
        str(args.cond_width),
        "--target_height",
        str(args.target_height),
        "--target_width",
        str(args.target_width),
        "--cfg_scale",
        str(args.cfg_scale),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--seed",
        str(args.seed),
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
        "--shard_id",
        "0",
        "--num_shards",
        "1",
    ]

    manifest = {
        "scene_dir": str(scene_dir),
        "scene_name": scene_name,
        "source_cam": args.source_cam,
        "target_cam": args.target_cam,
        "source_video": str(source_video),
        "case_key": case_key,
        "task_name": task_name,
        "caption_json": str(caption_path),
        "camera_json": str(camera_path),
        "cameraanything_output_dir": str(output_dir),
        "final_output_dir": str(final_dir),
        "source_meta": source_meta,
        "pose_hwf": [pose_h, pose_w, focal],
        "hfov_deg": hfov,
        "source_c2w_cameraanything": source_ca.tolist(),
        "target_c2w_cameraanything": target_ca.tolist(),
        "target_intrinsics_policy": "same_hfov_as_source",
        "pose_conversion": "LLFF columns [down,right,back,t] -> CameraAnything/OpenCV [right,down,forward,t]",
        "command": command,
    }
    _write_json(out_dir / "manifest.json", manifest)
    (out_dir / "run_cameraanything.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + _quote_cmd(command, CAMERAANYTHING_ROOT),
        encoding="utf-8",
    )
    os.chmod(out_dir / "run_cameraanything.sh", 0o755)

    if args.run:
        result = subprocess.run(command, cwd=str(CAMERAANYTHING_ROOT), capture_output=True, text=True)
        (out_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
        (out_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"CameraAnything failed with return code {result.returncode}")
        pred = output_dir / case_key / f"{case_key}_{task_name}_pred.mp4"
        src = output_dir / case_key / f"{case_key}_{task_name}_source.mp4"
        if pred.is_file():
            shutil.copy2(pred, final_dir / f"{case_key}_{task_name}_pred.mp4")
        if src.is_file():
            shutil.copy2(src, final_dir / f"{case_key}_{task_name}_source.mp4")
    return out_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene_dir", default="~/datasets/dynerf/coffee_martini")
    p.add_argument("--scene_name")
    p.add_argument("--source_cam", default="cam00")
    p.add_argument("--target_cam", default="cam04")
    p.add_argument("--case_key")
    p.add_argument("--caption")
    p.add_argument("--output_dir", default=str(PROJECT_ROOT / "cameraanything_dynerf_runs" / "coffee_martini_cam00_to_cam04"))
    p.add_argument("--checkpoint", default="models/cameraanything.ckpt")
    p.add_argument("--python_executable", default=DEFAULT_PYTHON)
    p.add_argument("--task_name", default="multi-shot-only_1")
    p.add_argument("--cam_type", default="pluc_adaln")
    p.add_argument("--cond_height", type=int, default=240)
    p.add_argument("--cond_width", type=int, default=416)
    p.add_argument("--target_height", type=int, default=240)
    p.add_argument("--target_width", type=int, default=416)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=5)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataloader_num_workers", type=int, default=1)
    p.add_argument("--run", action="store_true")
    return p.parse_args()


def main() -> int:
    try:
        out = prepare(parse_args())
        print(f"Prepared DyNeRF CameraAnything package: {out}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
