"""
GEM-SOMA (GEM-X) human pose estimation adapter.

Provides `infer_human_pose()` with the same interface expected by
`run_human_pose_est.py`, using the GEM-SOMA demo pipeline.

Note: GEM-SOMA and GEM-SMPL both export a package named `gem`.
Since GEM-SMPL is installed as the default `gem` package, this adapter
imports GEM-SOMA via sys.path manipulation to avoid conflicts.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# GEM-SOMA submodule root
_GEM_SOMA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "imports", "GEM-SOMA")
_GEM_SOMA_DEMO_DIR = os.path.join(_GEM_SOMA_ROOT, "scripts", "demo")

# Default checkpoint paths
_DEFAULT_CKPT = os.path.join(_GEM_SOMA_ROOT, "inputs", "pretrained", "gem_soma.ckpt")


def infer_human_pose(
    video_path,
    cache_dir,
    is_static_cam=False,
    verbose=False,
    render_mhr=False,
):
    """
    Run GEM-SOMA human pose estimation on a video.

    Returns a dict compatible with run_human_pose_est.py:
        {
            "smpl_params_global": {body_pose, global_orient, transl, betas, ...},
            "smpl_params_incam":  {body_pose, global_orient, transl, betas, ...},
            "vitpose": (L, 17, 3) or None,
            "foot_contact_probs": (L, 4) or None,
            "K_fullimg": (L, 3, 3),
        }
    """
    # Import GEM-SOMA by temporarily prepending its path
    # This overrides the installed GEM-SMPL 'gem' package for this function's scope
    saved_gem = sys.modules.pop("gem", None)
    saved_gem_submodules = {k: v for k, v in sys.modules.items() if k.startswith("gem.")}
    for k in saved_gem_submodules:
        sys.modules.pop(k, None)

    sys.path.insert(0, _GEM_SOMA_ROOT)
    sys.path.insert(0, _GEM_SOMA_DEMO_DIR)

    try:
        from demo_soma import _build_cfg, _copy_video_if_needed, load_data_dict, run_preprocess
        from gem.utils.net_utils import detach_to_cpu
        from gem.utils.pylogger import Log
        from gem.utils.video_io_utils import get_video_lwh

        output_dir = os.path.abspath(cache_dir)
        output_root_resolved = os.path.dirname(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Check for cached results
        cached_results = os.path.join(output_dir, "gem_soma_pred.pt")
        if os.path.exists(cached_results):
            print(f"[GEM-SOMA] Loading cached results from {cached_results}")
            return torch.load(cached_results, map_location="cpu")

        t0 = time.time()

        # Build Hydra config via Args mock
        class _Args:
            pass

        args = _Args()
        args.video = str(Path(video_path).resolve())
        args.output_root = output_root_resolved
        args.static_cam = is_static_cam
        args.verbose = verbose
        args.render_mhr = render_mhr
        args.ckpt = _DEFAULT_CKPT if os.path.exists(_DEFAULT_CKPT) else None
        args.exp = "gem_soma_regression"
        args.detector_name = "vitdet"
        args.sam3d_ckpt_path = None
        args.sam3d_mhr_path = None
        args.retarget = False

        cfg = _build_cfg(args)

        # Preprocessing
        L, W, H = get_video_lwh(cfg.video_path)
        print(f"[GEM-SOMA] Video: {L} frames, {W}x{H}")
        _copy_video_if_needed(cfg)
        run_preprocess(cfg)
        print(f"[GEM-SOMA] Preprocessing done ({time.time() - t0:.1f}s)")

        # Load data
        data = load_data_dict(cfg)

        # Load model and run inference
        import hydra

        Log.info("[GEM-SOMA] Loading model...")
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        if cfg.ckpt_path and os.path.exists(cfg.ckpt_path):
            model.load_pretrained_model(cfg.ckpt_path)
        else:
            from gem.utils.hf_utils import download_checkpoint

            ckpt_path = download_checkpoint()
            model.load_pretrained_model(ckpt_path)

        model = model.eval().cuda()
        pred = model.predict(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)
        print(f"[GEM-SOMA] Inference done ({time.time() - t0:.1f}s)")

        # Remap to expected output format (CPU tensors)
        def _to_cpu(d):
            return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in d.items()}

        result = {
            "smpl_params_global": _to_cpu(
                pred.get("body_params_global", pred.get("smpl_params_global", {}))
            ),
            "smpl_params_incam": _to_cpu(
                pred.get("body_params_incam", pred.get("smpl_params_incam", {}))
            ),
            "K_fullimg": pred.get("K_fullimg"),
        }

        # Add vitpose if available
        vitpose_path = cfg.paths.get("vitpose", os.path.join(cfg.preprocess_dir, "vitpose.pt"))
        if os.path.exists(vitpose_path):
            kp2d = torch.load(vitpose_path, map_location="cpu")
            if isinstance(kp2d, tuple):
                kp2d = kp2d[0]
            result["vitpose"] = kp2d
        else:
            result["vitpose"] = None

        # Extract foot contact probabilities if available
        foot_contact_probs = None
        if "net_outputs" in pred and "model_output" in pred.get("net_outputs", {}):
            model_output = pred["net_outputs"]["model_output"]
            if "static_conf_logits" in model_output:
                static_conf_logits = model_output["static_conf_logits"]
                foot_contact_probs = torch.sigmoid(static_conf_logits[:, :, :4])
        result["foot_contact_probs"] = foot_contact_probs

        # Cache results
        torch.save(result, cached_results)
        print(f"[GEM-SOMA] Results cached to {cached_results}")

        return result

    finally:
        # Restore GEM-SMPL as the 'gem' package
        if _GEM_SOMA_ROOT in sys.path:
            sys.path.remove(_GEM_SOMA_ROOT)
        if _GEM_SOMA_DEMO_DIR in sys.path:
            sys.path.remove(_GEM_SOMA_DEMO_DIR)
        # Clear GEM-SOMA modules
        for k in list(sys.modules.keys()):
            if k == "gem" or k.startswith("gem."):
                del sys.modules[k]
        # Restore GEM-SMPL modules
        if saved_gem is not None:
            sys.modules["gem"] = saved_gem
        for k, v in saved_gem_submodules.items():
            sys.modules[k] = v
