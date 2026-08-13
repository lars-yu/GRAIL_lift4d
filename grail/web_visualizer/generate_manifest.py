"""Generate a static web visualizer for a GRAIL motion library.

Scans a retargeted (or data-export-merged) motion library and produces:

    <output>/
        index.html          copy of grail/web_visualizer/assets/index.html
        main.js             copy of grail/web_visualizer/assets/main.js
        style.css           copy of grail/web_visualizer/assets/style.css
        manifest.json       per-motion metadata (name, frames, duration, meta keys)
        vis/                symlinks (or copies) to per-motion MP4s
        thumbs/ (optional)  extracted first-frame JPEGs

Point any static file server at `<output>/` and open `index.html`. Zero
server-side code.

Usage:
    python -m grail.web_visualizer.generate_manifest \\
        --motion_lib data/motion_lib/benchmark_v3_0126 \\
        --output out/viz/benchmark_v3_0126 \\
        --copy_videos       # off by default — symlinks instead (fast, space-efficient)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import joblib  # motion-lib pkls are joblib-serialized, not raw pickle

_ASSET_FILES = ("index.html", "main.js", "style.css")


def _load(path: Path):
    """Load a motion-lib pkl (joblib or pickle). Returns None on failure."""
    try:
        return joblib.load(path)
    except Exception:
        return None


def _unwrap_single_key(d):
    """Retarget/export pkls are often {seq_name: {...actual...}}. Unwrap."""
    if isinstance(d, dict) and len(d) == 1:
        inner = next(iter(d.values()))
        if isinstance(inner, dict):
            return inner
    return d


def _read_meta(path: Path) -> dict:
    """Best-effort read of a meta pkl produced by process.py or retarget.py.

    Returns a dict with whichever of {object_name, table_pos, table_quat,
    table_size} are present. Unknown keys ignored so the manifest stays stable.
    """
    data = _load(path)
    if data is None:
        return {}
    data = _unwrap_single_key(data)
    if not isinstance(data, dict):
        return {}
    keys = ("object_name", "table_pos", "table_quat", "table_size")
    out = {}
    for k in keys:
        if k in data:
            v = data[k]
            try:
                # Coerce numpy arrays / tensors into plain lists for JSON.
                v = v.tolist() if hasattr(v, "tolist") else v
            except Exception:
                continue
            out[k] = v
    return out


def _read_robot_len(path: Path) -> int | None:
    """Return number of frames in a robot pkl, or None on error."""
    data = _load(path)
    if data is None:
        return None
    data = _unwrap_single_key(data)
    if not isinstance(data, dict):
        return None
    for key in ("joint_pos", "dof_pos", "poses", "qpos", "body_pos", "root_trans_offset"):
        if key in data:
            v = data[key]
            try:
                return int(v.shape[0])
            except Exception:
                try:
                    return len(v)
                except Exception:
                    continue
    return None


def _stage_video(video_src: Path, video_dst: Path, copy: bool) -> None:
    video_dst.parent.mkdir(parents=True, exist_ok=True)
    if video_dst.exists() or video_dst.is_symlink():
        video_dst.unlink()
    if copy:
        shutil.copy2(video_src, video_dst)
    else:
        # Relative symlink so the output dir is portable.
        rel = os.path.relpath(video_src, video_dst.parent)
        os.symlink(rel, video_dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_lib",
        required=True,
        help="Path to motion library (retargeted or data-export output). Expects "
        "robot/*.pkl, vis/*.mp4, and optionally objects/, meta/.",
    )
    parser.add_argument("--output", required=True, help="Output directory (static site root).")
    parser.add_argument(
        "--copy_videos",
        action="store_true",
        help="Copy MP4s into the output (default: relative symlinks).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Site title (default: motion_lib basename).",
    )
    args = parser.parse_args()

    motion_lib = Path(args.motion_lib).resolve()
    if not motion_lib.is_dir():
        print(f"ERROR: {motion_lib} is not a directory", file=sys.stderr)
        return 2

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vis").mkdir(exist_ok=True)

    # --- copy static assets -----------------------------------------------
    assets_dir = Path(__file__).parent / "assets"
    for name in _ASSET_FILES:
        src = assets_dir / name
        if not src.exists():
            print(f"WARNING: asset not found: {src}", file=sys.stderr)
            continue
        shutil.copy2(src, out_dir / name)

    # --- enumerate motions ------------------------------------------------
    robot_dir = motion_lib / "robot"
    vis_dir = motion_lib / "vis"
    meta_dir = motion_lib / "meta"
    if not robot_dir.is_dir():
        print(f"ERROR: {robot_dir} missing (expected robot/*.pkl)", file=sys.stderr)
        return 2

    motions = []
    for pkl in sorted(robot_dir.glob("*.pkl")):
        stem = pkl.stem
        entry: dict = {"name": stem}
        nframes = _read_robot_len(pkl)
        if nframes is not None:
            entry["frames"] = nframes
        # Video
        video_src = vis_dir / f"{stem}.mp4" if vis_dir.exists() else None
        if video_src and video_src.exists():
            video_dst = out_dir / "vis" / f"{stem}.mp4"
            _stage_video(video_src.resolve(), video_dst, copy=args.copy_videos)
            entry["video"] = f"vis/{stem}.mp4"
        # Meta
        if meta_dir.exists():
            mpath = meta_dir / f"{stem}.pkl"
            if mpath.exists():
                entry["meta"] = _read_meta(mpath)
        motions.append(entry)

    title = args.title or motion_lib.name

    manifest = {
        "title": title,
        "source": str(motion_lib),
        "num_motions": len(motions),
        "num_with_video": sum(1 for m in motions if "video" in m),
        "motions": motions,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Wrote {out_dir}/manifest.json ({manifest['num_motions']} motions, "
        f"{manifest['num_with_video']} with video)"
    )
    print(f"Serve with:  cd {out_dir} && python -m http.server 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
