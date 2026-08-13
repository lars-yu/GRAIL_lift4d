"""Unified FoundationPose + MoGe reconstruction runner.

This module is installed as ``grail.pipelines.recon_fp_moge``.  It follows the
task-selection interface of ``recon_4dhoi`` while resolving the mesh exactly
once and passing that absolute path to every downstream stage.
"""

import argparse
import hashlib
import json
import runpy
import sys
from glob import glob
from pathlib import Path

from grail.core.config import load_recon_config


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_ids(args):
    if args.video_id:
        return [args.video_id.removesuffix(".mp4")]
    pattern = (
        Path(args.results_dir)
        / args.video_dir
        / args.dataset
        / args.category
        / f"{args.character or ''}*.mp4"
    )
    video_root = Path(args.results_dir) / args.video_dir
    ids = sorted(
        str(Path(path).relative_to(video_root)).removesuffix(".mp4")
        for path in glob(str(pattern))
    )
    if not ids:
        raise FileNotFoundError(f"No videos matched: {pattern}")
    return ids


def _resolve_mesh(args, video_id):
    dataset, category, _ = video_id.split("/", 2)
    if args.mesh:
        mesh_path = Path(args.mesh)
        if not mesh_path.is_absolute():
            mesh_path = Path.cwd() / mesh_path
        if not mesh_path.is_file():
            raise FileNotFoundError(f"--mesh does not exist: {mesh_path}")
        return mesh_path.resolve()

    mesh_dir = Path(args.results_dir) / "generation" / "mesh" / dataset / category
    candidates = sorted(mesh_dir.glob("*.obj"))
    if not candidates:
        raise FileNotFoundError(f"No OBJ mesh found in: {mesh_dir}")
    hashes = {_sha256(path) for path in candidates}
    if len(hashes) > 1:
        names = ", ".join(str(path) for path in candidates)
        raise RuntimeError(
            "Multiple non-identical meshes found. Select the intended mesh explicitly "
            f"with --mesh: {names}"
        )
    return candidates[0].resolve()


def _resolve_reference_depth(project_root, args, video_id):
    if args.reference_depth_path:
        depth_path = Path(args.reference_depth_path).expanduser()
        if not depth_path.is_absolute():
            depth_path = Path.cwd() / depth_path
    elif args.reference_depth_source == "moge":
        depth_path = (
            Path(args.results_dir)
            / "generation"
            / "4dhoi_recon_cache"
            / "depth"
            / f"{video_id}.pt"
        )
    else:
        depth_path = project_root / "depth_vda" / "depth" / f"{video_id}.pt"
    return _require(depth_path.resolve(), "reference depth cache")


def _require(path, label):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _load_script(project_root, name):
    namespace = runpy.run_path(
        str(project_root / "scripts" / name), run_name=f"fp_moge_{name}"
    )
    # runpy may return a dictionary distinct from the function's actual globals.
    # Override the latter so the existing stage script really receives this
    # task's video, mesh and output paths.
    return namespace["main"].__globals__


def _run_reference(project_root, args, video_id, mesh_path, depth_path, work_dir):
    module = _load_script(project_root, "analyze_fp_moge.py")
    generation = Path(args.results_dir) / "generation"
    module["DEVICE"] = args.device
    module["G"] = str(generation)
    module["V"] = video_id
    module["FP"] = str(generation / "foundation_pose_output" / video_id)
    module["FP_POSES"] = str(Path(module["FP"]) / "pose_estimation_output" / "poses_in_cam.pkl")
    module["MESH"] = str(mesh_path)
    module["CONTACT_CACHE"] = str(generation / "4dhoi_recon_cache" / "contact_labels" / f"{video_id}.json")
    module["DEPTH_CACHE"] = str(depth_path)
    module["DEPTH_SOURCE"] = args.reference_depth_source
    module["MAX_TRANSLATION_SHIFT"] = args.reference_max_shift_m
    module["OUT_DIR"] = str(work_dir)
    module["NUM_FRAMES"] = args.window
    module["main"]()


def _run_optimizer(project_root, args, video_id, mesh_path, work_dir):
    module = _load_script(project_root, "run_optimizer_fp_moge.py")
    generation = Path(args.results_dir) / "generation"
    module["G"] = args.results_dir
    module["GG"] = str(generation)
    module["V"] = video_id
    module["ALIGNED"] = str(work_dir / "poses_in_cam_aligned.pkl")
    module["OUT_DIR"] = str(work_dir)
    module["WINDOW"] = args.window
    module["MESH_GLOB"] = str(mesh_path)
    module["CONFIG"] = args.config
    module["PRE_EVAL"] = {
        "per_frame_tol": args.pre_eval_tol,
        "total_tol": args.pre_eval_total_tol,
        "min_frames": args.window,
    }
    module["main"]()


def _run_render(project_root, args, video_id, mesh_path, work_dir):
    module = _load_script(project_root, "render_optimizer_fp_moge.py")
    generation = Path(args.results_dir) / "generation"
    module["G"] = args.results_dir
    module["GG"] = str(generation)
    module["V"] = video_id
    module["ALIGNED"] = str(work_dir / "poses_in_cam_aligned.pkl")
    module["OUT_DIR"] = str(work_dir)
    module["FINAL_DIR"] = str(work_dir / "opt_out" / "final_render")
    module["MESH_GLOB"] = str(mesh_path)
    module["CONFIG"] = args.config
    module["VIS_OBJ_TARGET_VERTS"] = args.render_verts
    module["main"]()


def _write_manifest(work_dir, args, video_id, mesh_path, depth_path):
    manifest = {
        "video_id": video_id,
        "mesh_path": str(mesh_path),
        "mesh_sha256": _sha256(mesh_path),
        "reference_depth": {
            "source": args.reference_depth_source,
            "path": str(depth_path),
            "max_translation_shift_m": args.reference_max_shift_m,
        },
        "window": args.window,
        "config": args.config,
        "stages": {
            "reference": "FoundationPose + MoGe relative depth",
            "optimizer": "HOI optimizer",
            "render": "camera and top-view visualization",
        },
    }
    with open(work_dir / "task_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="configs/recon_4dhoi/pickup_smplx.yaml")
    pre_args, _ = pre.parse_known_args()
    _, cfg_flat = load_recon_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Unified FoundationPose + MoGe reconstruction")
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--character", default=None)
    parser.add_argument("--video_id", default=None)
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--output_dir", default="logs/fp_moge")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window", type=int, default=121)
    parser.add_argument("--render_verts", type=int, default=8000)
    parser.add_argument(
        "--reference_depth_source",
        "--reference-depth-source",
        dest="reference_depth_source",
        choices=("moge", "vda"),
        default="moge",
        help="Depth source for FoundationPose relative-depth correction only.",
    )
    parser.add_argument(
        "--reference_depth_path",
        "--reference-depth-path",
        dest="reference_depth_path",
        default=None,
        help="Optional explicit .pt depth cache; defaults to the selected source's cache path.",
    )
    parser.add_argument(
        "--reference_max_shift_m",
        "--reference-max-shift-m",
        dest="reference_max_shift_m",
        type=float,
        default=None,
        help="Maximum per-frame FoundationPose translation correction along its camera ray.",
    )
    parser.add_argument("--pre_eval_tol", type=float, default=1.01)
    parser.add_argument("--pre_eval_total_tol", type=float, default=1.01)
    # Compatibility with recon_4dhoi. Upstream HMR/mask/depth/FP outputs are
    # prerequisites for this runner and are validated rather than regenerated.
    parser.add_argument("--skip_step1", action="store_true")
    parser.add_argument("--skip_step2", action="store_true")
    parser.add_argument("--skip_step3", action="store_true", help="Skip reference trajectory")
    parser.add_argument("--skip_step4", action="store_true", help="Skip optimization")
    parser.add_argument("--skip_step6", action="store_true", help="Skip final rendering")
    parser.add_argument("--skip_done", action="store_true")
    parser.set_defaults(**cfg_flat)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    for video_id in _video_ids(args):
        dataset, category, _ = video_id.split("/", 2)
        if (dataset, category) != (args.dataset, args.category):
            raise ValueError(f"Video {video_id} does not match --dataset/--category")
        work_dir = Path(args.output_dir) / video_id
        work_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = _resolve_mesh(args, video_id)
        depth_path = _resolve_reference_depth(project_root, args, video_id)
        _write_manifest(work_dir, args, video_id, mesh_path, depth_path)
        print(f"Task {video_id}\n  mesh: {mesh_path}\n  output: {work_dir}")

        generation = Path(args.results_dir) / "generation"
        _require(generation / "videos_kling" / f"{video_id}.mp4", "video")
        _require(generation / "hmr_smplx" / f"{video_id}.npz", "HMR output")
        _require(generation / "4dhoi_recon_cache" / "masks" / f"{video_id}.npz", "mask cache")
        _require(generation / "4dhoi_recon_cache" / "depth" / f"{video_id}.pt", "depth cache")
        _require(
            generation / "foundation_pose_output" / video_id / "pose_estimation_output" / "poses_in_cam.pkl",
            "FoundationPose trajectory",
        )

        aligned = work_dir / "poses_in_cam_aligned.pkl"
        hoi_data = work_dir / "opt_out" / "hoi_data.pkl"
        video = work_dir / "opt_out" / "final_render" / "final" / "final.mp4"
        if not args.skip_step3 and not (args.skip_done and aligned.exists()):
            _run_reference(project_root, args, video_id, mesh_path, depth_path, work_dir)
        if not args.skip_step4 and not (args.skip_done and hoi_data.exists()):
            _require(aligned, "aligned FoundationPose + MoGe trajectory")
            _run_optimizer(project_root, args, video_id, mesh_path, work_dir)
        if not args.skip_step6 and not (args.skip_done and video.exists()):
            _require(aligned, "aligned FoundationPose + MoGe trajectory")
            _require(hoi_data, "optimized HOI result")
            _run_render(project_root, args, video_id, mesh_path, work_dir)


if __name__ == "__main__":
    main()
