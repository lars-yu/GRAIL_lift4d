"""
GEM-SMPL human pose estimation adapter (grail branch).

Provides `infer_human_pose()` using the hmr4d package from the GEM-SMPL
grail branch (GENMO architecture with SMPL-X body model).

Reuses `demo_slam.py` from GEM-SMPL for preprocessing and data loading.
"""

import os
import sys
import time
import types
import math
from pathlib import Path

import numpy as np
import torch

_GEM_SMPL_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "imports", "GEM-SMPL")


def _setup_imports():
    """Add GEM-SMPL demo to sys.path and mock problematic modules."""
    gem_root = os.path.abspath(_GEM_SMPL_ROOT)
    demo_dir = os.path.join(gem_root, "tools", "demo")
    # This environment also has an editable hmr4d install from another GRAIL
    # checkout.  Pin both imports to the submodule owned by this repository.
    for path in (gem_root, demo_dir):
        while path in sys.path:
            sys.path.remove(path)
    sys.path[:0] = [demo_dir, gem_root]

    # Mock modules that demo_slam.py imports at module level but we don't need
    # (visualization, rendering, remote sync utilities)
    if "motiondiff.utils.vis_scenepic" not in sys.modules:
        mock = types.ModuleType("motiondiff.utils.vis_scenepic")
        mock.ScenepicVisualizer = lambda *a, **kw: None
        sys.modules["motiondiff.utils.vis_scenepic"] = mock

    if "motiondiff.utils.tools" not in sys.modules:

        class _CatchAllMock(types.ModuleType):
            """Returns a no-op for any attribute access (Timer, wandb_run_exists, etc.)."""

            def __getattr__(self, name):
                if name.startswith("__") and name.endswith("__"):
                    raise AttributeError(name)
                return lambda *a, **kw: None

        sys.modules["motiondiff.utils.tools"] = _CatchAllMock("motiondiff.utils.tools")

    if "hmr4d.utils.vis.o3d_render" not in sys.modules:
        mock = types.ModuleType("hmr4d.utils.vis.o3d_render")
        mock.Settings = type("Settings", (), {"Transparency": 0, "LIT": 1})
        mock.create_meshes = lambda *a, **kw: None
        mock.get_ground = lambda *a, **kw: None
        sys.modules["hmr4d.utils.vis.o3d_render"] = mock


def infer_human_pose(video_path, cache_dir, is_static_cam=False, verbose=False):
    """
    Run GEM-SMPL (hmr4d) human pose estimation on a video.

    Returns a dict compatible with run_human_pose_est.py:
        {
            "smpl_params_global": {body_pose, global_orient, transl, betas},
            "smpl_params_incam":  {body_pose, global_orient, transl, betas},
            "vitpose": (L, 17, 3),
            "foot_contact_probs": (L, 4) or None,
        }
    """
    output_dir = os.path.abspath(cache_dir)
    output_root_abs = os.path.dirname(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Check for cached results
    cached_results = os.path.join(output_dir, "gem_smpl_pred.pt")
    if os.path.exists(cached_results):
        print(f"[GEM-SMPL] Loading cached results from {cached_results}")
        return torch.load(cached_results, map_location="cpu")

    t0 = time.time()
    video_path_abs = os.path.abspath(video_path)

    # Setup imports and run directly
    _setup_imports()

    import cv2
    import hmr4d.model.genmo.genmo_demo  # noqa: F401 — registers genmo_demo model config
    import hydra
    from demo_slam import load_data_dict, run_preprocess
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
    from hmr4d.utils.net_utils import detach_to_cpu
    from hmr4d.utils.pylogger import Log
    from hmr4d.utils.video_io_utils import get_video_lwh
    from hydra import compose, initialize_config_module

    # Build Hydra config (replicates parse_args_to_cfg without argparse)
    video_path_obj = Path(video_path_abs)
    assert video_path_obj.exists(), f"Video not found at {video_path_obj}"
    length, width, height = get_video_lwh(video_path_obj)
    orig_fps = cv2.VideoCapture(str(video_path_obj)).get(cv2.CAP_PROP_FPS)
    Log.info(f"[GEM-SMPL] Input: {video_path_obj}, (L, W, H) = ({length}, {width}, {height})")

    register_store_gvhmr()
    overrides = [
        f"video_name={video_path_obj.stem}",
        f"static_cam={is_static_cam}",
        f"verbose={verbose}",
        f"output_root={output_root_abs}",
    ]
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(config_name="demo_genmo", overrides=overrides)

    # Override ckpt_path to absolute path so it doesn't depend on CWD
    gem_smpl_root = os.path.abspath(_GEM_SMPL_ROOT)
    cfg.ckpt_path = os.path.join(gem_smpl_root, cfg.ckpt_path)

    paths = cfg.paths
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy input video to expected location (demo convention)
    from hmr4d.utils.video_io_utils import get_video_reader, get_writer

    if (
        not Path(cfg.video_path).exists()
        or get_video_lwh(video_path_obj)[0] != get_video_lwh(cfg.video_path)[0]
    ):
        from tqdm import tqdm

        reader = get_video_reader(video_path_obj)
        writer = get_writer(cfg.video_path, fps=30, crf=23)
        for img in tqdm(reader, total=length, desc="[GEM-SMPL] Copy video"):
            writer.write_frame(img)
        writer.close()
        reader.close()

    # GEM-SMPL code uses bare "inputs/checkpoints/..." paths from CWD,
    # so temporarily chdir to GEM-SMPL root while running its functions.
    prev_cwd = os.getcwd()
    os.chdir(gem_smpl_root)
    try:
        # Preprocess (bbx tracking, vitpose, vit features, VIMO, optionally DROID-SLAM)
        run_preprocess(cfg, orig_fps)

        # Load preprocessed data
        data = load_data_dict(cfg)

        # Run HMR4D inference
        if not Path(paths.hmr4d_results).exists():
            Log.info("[GEM-SMPL] Running HMR4D prediction")
            model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
            model.load_pretrained_model(cfg.ckpt_path)
            model = model.eval().cuda()
            pred = model.predict(data, static_cam=cfg.static_cam)
            pred = detach_to_cpu(pred)
            torch.save(pred, paths.hmr4d_results)
            Log.info(f"[GEM-SMPL] Saved HMR4D results to {paths.hmr4d_results}")
        else:
            Log.info(f"[GEM-SMPL] Loading cached HMR4D results from {paths.hmr4d_results}")
            pred = torch.load(paths.hmr4d_results, map_location="cpu")
    finally:
        os.chdir(prev_cwd)

    # Load vitpose
    vitpose = None
    if os.path.exists(paths.vitpose):
        vitpose = torch.load(paths.vitpose, map_location="cpu")
        if isinstance(vitpose, tuple):
            vitpose = vitpose[0]

    # Remap to expected output format
    result = {
        "smpl_params_global": pred.get("smpl_params_global", {}),
        "smpl_params_incam": pred.get("smpl_params_incam", {}),
        "vitpose": pred.get("vitpose", vitpose),
        "foot_contact_probs": pred.get("foot_contact_probs", None),
    }

    # Extract foot contact probs if not already in pred
    if result["foot_contact_probs"] is None:
        if "net_outputs" in pred and "model_output" in pred.get("net_outputs", {}):
            model_output = pred["net_outputs"]["model_output"]
            if "static_conf_logits" in model_output:
                static_conf_logits = model_output["static_conf_logits"]
                result["foot_contact_probs"] = torch.sigmoid(static_conf_logits[:, :, :4])

    # Ensure CPU tensors
    for key in ["smpl_params_global", "smpl_params_incam"]:
        if isinstance(result[key], dict):
            result[key] = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in result[key].items()
            }

    # Cache results
    torch.save(result, cached_results)
    print(f"[GEM-SMPL] Results cached to {cached_results}")
    print(f"[GEM-SMPL] Total time: {time.time() - t0:.1f}s")

    return result


def _compact_genmo_prediction(pred):
    """Keep only tensors needed by the Stage 3.5 artifacts and renderer."""
    compact = {}
    for name in ("smpl_params_global", "smpl_params_incam"):
        compact[name] = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in pred[name].items()
        }
    for name in ("K_fullimg", "smpl24_joints_global", "smpl24_joints_incam"):
        value = pred.get(name)
        compact[name] = value.detach().cpu() if isinstance(value, torch.Tensor) else value
    model_output = pred["net_outputs"]["model_output"]
    compact["normalized_motion"] = model_output["pred_x"].detach().cpu()
    compact["guidance_steps"] = pred["net_outputs"].get("contact_guidance_steps", [])
    return compact


def _calibrate_human_points(points, camera_origin, depth_scale):
    """Apply the frame-0 camera-ray similarity used for GT depth alignment."""
    origin = torch.as_tensor(camera_origin, device=points.device, dtype=points.dtype)
    return origin + float(depth_scale) * (points - origin)


def _calibrate_compact_prediction(compact, camera_origin, depth_scale):
    """Apply one depth calibration consistently to saved params and joints."""
    scale = float(depth_scale)
    origin = torch.as_tensor(camera_origin, dtype=torch.float32).cpu()
    compact["smpl_params_global"]["transl"] = _calibrate_human_points(
        compact["smpl_params_global"]["transl"], origin, scale
    )
    compact["smpl_params_incam"]["transl"] = (
        compact["smpl_params_incam"]["transl"] * scale
    )
    compact["smpl24_joints_global"] = _calibrate_human_points(
        compact["smpl24_joints_global"], origin, scale
    )
    compact["smpl24_joints_incam"] = compact["smpl24_joints_incam"] * scale
    compact["human_depth_scale"] = scale
    return compact


def robust_human_depth_scale(predicted_depth, gt_depth, valid_mask, min_samples=32):
    """Estimate a robust camera-ray scale from projected human surface samples."""
    predicted = np.asarray(predicted_depth, dtype=np.float64).reshape(-1)
    target = np.asarray(gt_depth, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if predicted.shape != target.shape or predicted.shape != valid.shape:
        raise ValueError("predicted_depth, gt_depth, and valid_mask must have equal shape")
    valid &= np.isfinite(predicted) & np.isfinite(target)
    valid &= (predicted > 0.1) & (target > 0.1) & (target < 65.0)
    if int(valid.sum()) < int(min_samples):
        raise ValueError(
            f"GT human depth alignment has only {int(valid.sum())} valid samples; "
            f"requires at least {int(min_samples)}"
        )
    ratios = target[valid] / predicted[valid]
    scale = float(np.median(ratios))
    if not math.isfinite(scale) or not 0.4 <= scale <= 2.5:
        raise ValueError(f"implausible GT human camera-ray scale: {scale}")
    aligned_residual = np.abs(scale * predicted[valid] - target[valid])
    return scale, {
        "valid_surface_samples": int(valid.sum()),
        "predicted_surface_depth_median_m": float(np.median(predicted[valid])),
        "gt_surface_depth_median_m": float(np.median(target[valid])),
        "depth_ratio_median": scale,
        "depth_ratio_p10": float(np.quantile(ratios, 0.1)),
        "depth_ratio_p90": float(np.quantile(ratios, 0.9)),
        "aligned_surface_depth_abs_median_m": float(np.median(aligned_residual)),
        "aligned_surface_depth_abs_p90_m": float(np.quantile(aligned_residual, 0.9)),
    }


def _load_binary_render_mask(path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[..., 3] > 127
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image > 127


def _estimate_gt_human_depth_scale(model, baseline, gt_depth_path, human_mask_path):
    import cv2

    depth_mm = cv2.imread(str(gt_depth_path), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise FileNotFoundError(gt_depth_path)
    gt_depth = depth_mm.astype(np.float32) / 1000.0
    human_mask = _load_binary_render_mask(human_mask_path)
    if human_mask.shape != gt_depth.shape:
        human_mask = cv2.resize(
            human_mask.astype(np.uint8),
            (gt_depth.shape[1], gt_depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    params = {
        key: value[None, :1]
        for key, value in baseline["smpl_params_incam"].items()
    }
    with torch.no_grad():
        vertices, _ = model.smplxcoco17(**params)
    vertices = vertices[0, 0]
    K = torch.as_tensor(baseline["K_fullimg"][0], device=vertices.device)
    pixels = vertices[:, :2] / vertices[:, 2:].clamp_min(1e-6)
    pixels = pixels @ K[:2, :2].mT + K[:2, 2]
    xy = torch.round(pixels).long().detach().cpu().numpy()
    vertices_z = vertices[:, 2].detach().cpu().numpy()
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < gt_depth.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < gt_depth.shape[0])
    )
    sampled_gt = np.full(len(xy), np.nan, dtype=np.float32)
    sampled_mask = np.zeros(len(xy), dtype=bool)
    sampled_gt[inside] = gt_depth[xy[inside, 1], xy[inside, 0]]
    sampled_mask[inside] = human_mask[xy[inside, 1], xy[inside, 0]]
    scale, diagnostics = robust_human_depth_scale(
        vertices_z, sampled_gt, sampled_mask, min_samples=32
    )
    diagnostics.update({
        "enabled": True,
        "method": "frame0_gt_depth_projected_smplx437_median_ratio",
        "gt_depth_path": str(Path(gt_depth_path).resolve()),
        "human_mask_path": str(Path(human_mask_path).resolve()),
        "preserves_camera_projection": True,
    })
    return scale, diagnostics


def _guidance_step_summary(pred):
    steps = pred["net_outputs"].get("contact_guidance_steps", [])
    if not steps:
        return {"step_count": 0}
    summary = {
        "step_count": len(steps),
        "first_contact_loss": steps[0]["contact_loss"],
        "last_contact_loss": steps[-1]["contact_loss"],
        "minimum_contact_loss": min(step["contact_loss"] for step in steps),
        "maximum_gradient_norm": max(step["gradient_norm"] for step in steps),
        "last_gradient_norm": steps[-1]["gradient_norm"],
    }
    for block in ("arm", "root"):
        key = f"{block}_gradient_norm"
        if key in steps[0]:
            summary[f"maximum_{key}"] = max(step[key] for step in steps)
            summary[f"last_{key}"] = steps[-1][key]
    return summary


def _root_camera_to_global(pred, frame):
    from motiondiff.models.mdm.rotation_conversions import axis_angle_to_matrix

    global_params = pred["smpl_params_global"]
    incam_params = pred["smpl_params_incam"]
    global_R = axis_angle_to_matrix(global_params["global_orient"][frame])
    incam_R = axis_angle_to_matrix(incam_params["global_orient"][frame])
    camera_to_global_R = global_R @ incam_R.mT
    # SMPL-X ``transl`` is not the pelvis joint position because the shaped
    # template has a non-zero root offset. Solve translation from corresponding
    # decoded joints so the complete skeleton, not just the parameter vector,
    # defines the camera/global rigid transform.
    global_joints = pred["smpl24_joints_global"][frame]
    incam_joints = pred["smpl24_joints_incam"][frame]
    camera_to_global_t = (
        global_joints - incam_joints @ camera_to_global_R.mT
    ).mean(dim=0)
    return camera_to_global_R, camera_to_global_t


def _maximum_arm_rotation_change_deg(endecoder, initial_x, guided_x, hand):
    from motiondiff.models.mdm.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle
    from hmr4d.model.genmo.contact_guidance import arm_channel_slices

    initial_pose = endecoder.decode(initial_x)["body_pose"].reshape(1, initial_x.shape[1], 21, 3)
    guided_pose = endecoder.decode(guided_x)["body_pose"].reshape(1, guided_x.shape[1], 21, 3)
    joint_ids = [channel_slice.start // 6 for channel_slice in arm_channel_slices(hand).values()]
    initial_R = axis_angle_to_matrix(initial_pose[:, :, joint_ids])
    guided_R = axis_angle_to_matrix(guided_pose[:, :, joint_ids])
    relative = guided_R @ initial_R.mT
    return float(torch.rad2deg(matrix_to_axis_angle(relative).norm(dim=-1)).max().detach().cpu())


def run_contact_guided_genmo(
    video_path,
    cache_dir,
    object_vertices_cam,
    object_faces,
    contact_frame,
    selected_hand,
    *,
    seed=42,
    reach_error_threshold=0.03,
    max_arm_rotation_change_deg=30.0,
    max_root_correction=0.25,
    guidance_strength=0.7,
    root_guidance_multiplier=1.0,
    is_static_cam=True,
    verbose=False,
    diagnostics_path=None,
    gt_depth_path=None,
    human_mask_path=None,
    object_vertices_local=None,
    object_poses_cam=None,
    contact_transition_frames=8,
    guidance_temporal_weight=0.5,
    post_contact_relative_velocity_weight=8.0,
    post_contact_worst_frame_weight=0.0,
    post_contact_terminal_position_weight=1.0,
    post_contact_terminal_frames=16,
    contact_frame_position_weight=1.25,
    contact_frame_weight_radius=2,
):
    """Run baseline and two-stage contact-guided GENMO with shared noise.

    Object geometry is supplied in the contact-frame OpenCV camera coordinate
    system.  The baseline root pose determines the rigid camera-to-GENMO-global
    transform used consistently by all three samples.
    """
    import hydra
    import trimesh
    _setup_imports()
    from demo_slam import load_data_dict
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.model.genmo.contact_guidance import (
        WRIST_JOINT_INDEX,
        generate_sampling_noise,
        root_candidate_improves_hold,
        should_use_root_fallback,
    )
    from hydra import compose, initialize_config_module

    if not torch.cuda.is_available():
        raise RuntimeError("GENMO contact guidance requires CUDA for the released checkpoint")
    video_path = Path(video_path).resolve()
    cache_dir = Path(cache_dir).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    output_root_abs = str(cache_dir.parent)
    register_store_gvhmr()
    overrides = [
        f"video_name={video_path.stem}",
        f"static_cam={is_static_cam}",
        f"verbose={verbose}",
        f"output_root={output_root_abs}",
    ]
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(config_name="demo_genmo", overrides=overrides)
    cfg.ckpt_path = os.path.join(os.path.abspath(_GEM_SMPL_ROOT), cfg.ckpt_path)

    previous_cwd = os.getcwd()
    os.chdir(os.path.abspath(_GEM_SMPL_ROOT))
    try:
        data = load_data_dict(cfg)
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        length = int(data["f_imgseq"].shape[0])
        motion_dim = int(model.endecoder.get_motion_dim())
        if motion_dim != 151:
            raise ValueError(f"Stage 3.5 requires verified 151-D GENMO motion, got {motion_dim}")
        sampling_noise = generate_sampling_noise(
            (1, length, motion_dim), device="cuda", seed=seed
        )

        baseline = model.predict(
            data,
            static_cam=is_static_cam,
            sampling_noise=sampling_noise,
            postproc_override=False,
        )
        reference_x = baseline["net_outputs"]["model_output"]["pred_x"].detach()
        palm_inputs = {"cam_angvel": data["cam_angvel"][None].cuda()}
        initial_palm_global = model.pipeline.decode_palm_position(
            reference_x, palm_inputs, selected_hand
        )[0]

        frame = int(contact_frame)
        if not 0 <= frame < length:
            raise ValueError(f"contact frame {frame} outside GENMO motion length {length}")
        camera_to_global_R, camera_to_global_t = _root_camera_to_global(baseline, frame)
        baseline_root_transl = baseline["smpl_params_global"]["transl"].detach()
        if gt_depth_path is None or human_mask_path is None:
            raise ValueError(
                "Stage 3.5 requires frame-0 GT depth and the rendered human mask "
                "to align GENMO before contact guidance"
            )
        human_depth_scale, depth_alignment = _estimate_gt_human_depth_scale(
            model, baseline, gt_depth_path, human_mask_path
        )
        initial_palm_aligned = _calibrate_human_points(
            initial_palm_global, camera_to_global_t, human_depth_scale
        )
        global_joints_cam = (
            baseline["smpl24_joints_global"][frame] - camera_to_global_t
        ) @ camera_to_global_R
        global_incam_consistency = torch.linalg.vector_norm(
            global_joints_cam - baseline["smpl24_joints_incam"][frame], dim=-1
        )

        coco_incam = baseline["coco17_joints_incam"]
        K_fullimg = torch.as_tensor(baseline["K_fullimg"], device=coco_incam.device)
        projected_coco = coco_incam[..., :2] / coco_incam[..., 2:].clamp_min(1e-6)
        projected_coco = torch.einsum(
            "fij,fkj->fki", K_fullimg, torch.cat((projected_coco, torch.ones_like(projected_coco[..., :1])), dim=-1)
        )[..., :2]
        observed_kp2d = data["kp2d"].to(projected_coco.device)
        visible = observed_kp2d[..., 2] > 0.7
        reprojection_error = torch.linalg.vector_norm(
            projected_coco - observed_kp2d[..., :2], dim=-1
        )[visible]
        verts_cam = torch.as_tensor(object_vertices_cam, device="cuda", dtype=torch.float32)
        verts_global = (camera_to_global_R @ verts_cam.mT).mT + camera_to_global_t

        # trimesh's naive triangle query is deterministic and finds a true point
        # on the surface rather than substituting the object centre or a vertex.
        vertices_numpy = verts_global.detach().cpu().numpy()
        faces_numpy = torch.as_tensor(object_faces, dtype=torch.long).cpu().numpy()
        triangles = vertices_numpy[faces_numpy]
        twice_area = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=-1,
        )
        valid_faces = np.isfinite(triangles).all(axis=(1, 2)) & (twice_area > 1e-10)
        if not valid_faces.any():
            raise ValueError("object mesh has no finite, non-degenerate surface triangles")
        mesh_global = trimesh.Trimesh(
            vertices=vertices_numpy,
            faces=faces_numpy[valid_faces],
            process=False,
        )
        closest, distance, _ = trimesh.proximity.closest_point_naive(
            mesh_global, initial_palm_aligned[frame : frame + 1].detach().cpu().numpy()
        )
        target_global = torch.as_tensor(closest[0], device="cuda", dtype=torch.float32)
        initial_error = float(distance[0])
        if not torch.isfinite(target_global).all() or not math.isfinite(initial_error):
            raise ValueError("closest object surface query returned NaN/Inf")

        human_height = float(
            baseline["smpl24_joints_incam"][frame, :, 1].max()
            - baseline["smpl24_joints_incam"][frame, :, 1].min()
        ) * human_depth_scale
        object_extent = float(torch.linalg.vector_norm(verts_cam.max(0).values - verts_cam.min(0).values))
        if not math.isfinite(human_height) or not (0.5 <= human_height <= 3.0):
            raise ValueError(f"human scale is not plausibly metric: height={human_height:.4f}m")
        if not math.isfinite(object_extent) or not (0.01 <= object_extent <= 5.0):
            raise ValueError(f"object scale is not plausibly metric: extent={object_extent:.4f}m")
        if initial_error > 3.0:
            raise ValueError(
                f"human/object coordinates are inconsistent: palm-surface distance={initial_error:.4f}m"
            )

        initial_palm_cam = camera_to_global_R.mT @ (
            initial_palm_aligned[frame] - camera_to_global_t
        )
        target_cam = camera_to_global_R.mT @ (target_global - camera_to_global_t)
        palm_to_surface_delta_cam = target_cam - initial_palm_cam

        object_surface_targets_global = None
        if object_vertices_local is not None and object_poses_cam is not None:
            local_target = (
                target_cam
                - torch.as_tensor(
                    np.asarray(object_poses_cam[frame, :3, 3]).copy(), device="cuda"
                )
            ) @ torch.as_tensor(
                np.asarray(object_poses_cam[frame, :3, :3]).copy(), device="cuda"
            )
            trajectory_cam = []
            for pose in np.asarray(object_poses_cam, dtype=np.float32):
                R = torch.as_tensor(
                    np.asarray(pose[:3, :3]).copy(), device="cuda", dtype=torch.float32
                )
                t = torch.as_tensor(
                    np.asarray(pose[:3, 3]).copy(), device="cuda", dtype=torch.float32
                )
                trajectory_cam.append(local_target @ R.mT + t)
            trajectory_cam = torch.stack(trajectory_cam)
            object_surface_targets_global = (
                trajectory_cam @ camera_to_global_R.mT + camera_to_global_t
            )

        compact_baseline = _calibrate_compact_prediction(
            _compact_genmo_prediction(baseline), camera_to_global_t, human_depth_scale
        )
        del baseline

        diagnostics = {
            "contact_frame": frame,
            "selected_hand": selected_hand,
            "initial_contact_error_m": initial_error,
            "initial_palm_position": initial_palm_aligned[frame].detach().cpu().tolist(),
            "object_center": verts_global.mean(0).detach().cpu().tolist(),
            "closest_surface_point": target_global.detach().cpu().tolist(),
            "initial_palm_position_camera": initial_palm_cam.detach().cpu().tolist(),
            "closest_surface_point_camera": target_cam.detach().cpu().tolist(),
            "palm_to_surface_delta_camera": palm_to_surface_delta_cam.detach().cpu().tolist(),
            "initial_depth_error_m": float(abs(palm_to_surface_delta_cam[2]).detach().cpu()),
            "initial_camera_xy_error_m": float(
                torch.linalg.vector_norm(palm_to_surface_delta_cam[:2]).detach().cpu()
            ),
            "human_object_coordinate_system": "GENMO global (camera aligned at contact frame)",
            "human_scale": "metres",
            "object_scale": "metres",
            "frame0_human_depth_alignment": depth_alignment,
            "human_depth_scale": human_depth_scale,
            "guidance_strength": float(guidance_strength),
            "root_guidance_multiplier": float(root_guidance_multiplier),
            "human_height_m": human_height,
            "object_extent_m": object_extent,
            "motion_dimension": motion_dim,
            "wrist_joint_index": WRIST_JOINT_INDEX[selected_hand],
            "genmo_position_target_segments": {
                "pre_contact": {
                    "frames": [0, frame],
                    "shape": [3],
                    "source": "static_first_frame_gt_object_surface",
                },
                "post_contact": {
                    "frames": [frame, length],
                    "shape": [max(0, length - frame), 3],
                    "source": "frozen_object_trajectory_surface_target",
                },
            },
            "sampling_noise_shape": list(sampling_noise.shape),
            "global_incam_joint_consistency_mean_m": float(
                global_incam_consistency.mean().detach().cpu()
            ),
            "global_incam_joint_consistency_max_m": float(
                global_incam_consistency.max().detach().cpu()
            ),
            "genmo_vitpose_reprojection_median_px": float(
                reprojection_error.median().detach().cpu()
            ),
            "genmo_vitpose_reprojection_mean_px": float(
                reprojection_error.mean().detach().cpu()
            ),
        }
        print("Stage 3.5 pre-guidance coordinate check")
        for key in (
            "contact_frame",
            "selected_hand",
            "initial_palm_position",
            "object_center",
            "closest_surface_point",
            "initial_contact_error_m",
            "human_object_coordinate_system",
            "human_scale",
            "object_scale",
        ):
            print(f"  {key}: {diagnostics[key]}")
        if diagnostics_path is not None:
            import json

            diagnostics_file = Path(diagnostics_path)
            diagnostics_file.parent.mkdir(parents=True, exist_ok=True)
            with diagnostics_file.open("w") as handle:
                json.dump(diagnostics, handle, indent=2)

        common_spec = {
            "contact_frame": frame,
            "selected_hand": selected_hand,
            "object_surface_target": target_global,
            "object_surface_targets": object_surface_targets_global,
            "pre_contact_surface_target": target_global,
            "post_contact_surface_targets": (
                None
                if object_surface_targets_global is None
                else object_surface_targets_global[frame:]
            ),
            "contact_transition_frames": int(contact_transition_frames),
            "w_temporal": float(guidance_temporal_weight),
            "post_contact_relative_velocity_weight": float(
                post_contact_relative_velocity_weight
            ),
            "post_contact_worst_frame_weight": float(
                post_contact_worst_frame_weight
            ),
            "post_contact_terminal_position_weight": float(
                post_contact_terminal_position_weight
            ),
            "post_contact_terminal_frames": int(post_contact_terminal_frames),
            "contact_frame_position_weight": float(contact_frame_position_weight),
            "contact_frame_weight_radius": int(contact_frame_weight_radius),
            "reference_motion": reference_x,
            "guidance_strength": float(guidance_strength),
            "diffusion_steps": int(model.pipeline.denoiser3d.test_gen_only_diffusion.num_timesteps),
            "human_depth_scale": human_depth_scale,
            "camera_origin_global": camera_to_global_t,
            "root_guidance_multiplier": float(root_guidance_multiplier),
        }
        arm_only = model.predict(
            data,
            static_cam=is_static_cam,
            sampling_noise=sampling_noise,
            contact_guidance={**common_spec, "include_root": False},
            postproc_override=False,
        )
        arm_x = arm_only["net_outputs"]["model_output"]["pred_x"].detach()
        arm_palm = model.pipeline.decode_palm_position(arm_x, palm_inputs, selected_hand)[0]
        arm_palm = _calibrate_human_points(
            arm_palm, camera_to_global_t, human_depth_scale
        )
        target_trajectory = (
            object_surface_targets_global
            if object_surface_targets_global is not None
            else target_global.reshape(1, 3).expand(length, 3)
        )
        arm_errors = torch.linalg.vector_norm(arm_palm - target_trajectory, dim=-1)
        arm_error = float(arm_errors[frame].detach().cpu())
        arm_post_contact_mean_error = float(arm_errors[frame:].mean().detach().cpu())
        arm_post_contact_max_error = float(arm_errors[frame:].max().detach().cpu())
        arm_rotation_change = _maximum_arm_rotation_change_deg(
            model.endecoder, reference_x, arm_x, selected_hand
        )
        fallback = should_use_root_fallback(
            max(arm_error, arm_post_contact_mean_error, arm_post_contact_max_error),
            arm_rotation_change,
            reach_error_threshold=float(reach_error_threshold),
            max_arm_rotation_change_limit_deg=float(max_arm_rotation_change_deg),
        )

        arm_guidance_summary = _guidance_step_summary(arm_only)
        compact_arm_only = _calibrate_compact_prediction(
            _compact_genmo_prediction(arm_only), camera_to_global_t, human_depth_scale
        )

        arm_root = None
        root_palm = None
        root_errors = None
        root_error = None
        root_post_contact_mean_error = None
        root_post_contact_max_error = None
        root_candidate_rotation_change = None
        root_candidate_change = 0.0
        root_fallback_selected = False
        if fallback:
            arm_root = model.predict(
                data,
                static_cam=is_static_cam,
                sampling_noise=sampling_noise,
                contact_guidance={**common_spec, "include_root": True},
                postproc_override=False,
            )
            root_x = arm_root["net_outputs"]["model_output"]["pred_x"].detach()
            root_palm = model.pipeline.decode_palm_position(root_x, palm_inputs, selected_hand)[0]
            root_palm = _calibrate_human_points(
                root_palm, camera_to_global_t, human_depth_scale
            )
            root_errors = torch.linalg.vector_norm(
                root_palm - target_trajectory, dim=-1
            )
            root_error = float(root_errors[frame].detach().cpu())
            root_post_contact_mean_error = float(
                root_errors[frame:].mean().detach().cpu()
            )
            root_post_contact_max_error = float(
                root_errors[frame:].max().detach().cpu()
            )
            root_candidate_rotation_change = _maximum_arm_rotation_change_deg(
                model.endecoder, reference_x, root_x, selected_hand
            )
            root_delta = (
                arm_root["smpl_params_global"]["transl"]
                - baseline_root_transl
            ) * human_depth_scale
            root_candidate_change = float(
                torch.linalg.vector_norm(root_delta, dim=-1).max().detach().cpu()
            )
            root_candidate_safe = (
                root_candidate_rotation_change
                <= float(max_arm_rotation_change_deg) + 1e-6
                and root_candidate_change <= float(max_root_correction) + 1e-6
            )
            root_fallback_selected = bool(
                root_candidate_safe
                and root_candidate_improves_hold(
                    arm_error,
                    arm_post_contact_mean_error,
                    arm_post_contact_max_error,
                    root_error,
                    root_post_contact_mean_error,
                    root_post_contact_max_error,
                )
            )
        root_guidance_summary = (
            None if arm_root is None else _guidance_step_summary(arm_root)
        )
        compact_arm_root = None
        if arm_root is not None:
            compact_arm_root = _calibrate_compact_prediction(
                _compact_genmo_prediction(arm_root),
                camera_to_global_t,
                human_depth_scale,
            )
        compact_selected = (
            compact_arm_root if root_fallback_selected else compact_arm_only
        )

        selected_palm = root_palm if root_fallback_selected else arm_palm
        final_arm_rotation_change = (
            root_candidate_rotation_change
            if root_fallback_selected
            else arm_rotation_change
        )
        if final_arm_rotation_change > float(max_arm_rotation_change_deg) + 1e-6:
            raise RuntimeError(
                "no guided candidate satisfies the arm rotation safety limit: "
                f"selected={final_arm_rotation_change:.4f}deg, "
                f"limit={float(max_arm_rotation_change_deg):.4f}deg"
            )
        root_change = root_candidate_change if root_fallback_selected else 0.0
        selected_errors = torch.linalg.vector_norm(
            selected_palm - target_trajectory, dim=-1
        )
        final_error = float(selected_errors[frame].detach().cpu())
        selected_relative_position = selected_palm - target_trajectory
        selected_relative_steps = torch.linalg.vector_norm(
            selected_relative_position[frame + 1 :]
            - selected_relative_position[frame:-1],
            dim=-1,
        )
        selected_steps = torch.linalg.vector_norm(
            selected_palm[1:] - selected_palm[:-1], dim=-1
        )
        initial_steps = torch.linalg.vector_norm(
            initial_palm_aligned[1:] - initial_palm_aligned[:-1], dim=-1
        )
        selected_post_contact_errors = selected_errors[frame:]
        max_error_offset = int(selected_post_contact_errors.argmax().detach().cpu())
        relative_step_max_frame = (
            frame + 1 + int(selected_relative_steps.argmax().detach().cpu())
            if selected_relative_steps.numel()
            else frame
        )

        diagnostics.update({
            "arm_only_contact_error_m": arm_error,
            "arm_only_post_contact_mean_error_m": arm_post_contact_mean_error,
            "arm_only_post_contact_max_error_m": arm_post_contact_max_error,
            "arm_root_contact_error_m": root_error,
            "arm_root_post_contact_mean_error_m": root_post_contact_mean_error,
            "arm_root_post_contact_max_error_m": root_post_contact_max_error,
            "final_contact_error_m": final_error,
            "final_post_contact_mean_error_m": float(
                selected_post_contact_errors.mean().detach().cpu()
            ),
            "final_post_contact_max_error_m": float(
                selected_post_contact_errors.max().detach().cpu()
            ),
            "final_post_contact_max_error_frame": frame + max_error_offset,
            "final_post_contact_error_m_per_frame": (
                selected_post_contact_errors.detach().cpu().tolist()
            ),
            "final_post_contact_within_threshold_ratio": float(
                (selected_errors[frame:] <= float(reach_error_threshold))
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "final_post_contact_relative_step_mean_m": float(
                selected_relative_steps.mean().detach().cpu()
                if selected_relative_steps.numel()
                else 0.0
            ),
            "final_post_contact_relative_step_max_m": float(
                selected_relative_steps.max().detach().cpu()
                if selected_relative_steps.numel()
                else 0.0
            ),
            "final_post_contact_relative_step_max_frame": relative_step_max_frame,
            "final_post_contact_relative_step_m_per_frame": (
                selected_relative_steps.detach().cpu().tolist()
            ),
            "initial_max_adjacent_palm_step_m": float(initial_steps.max().detach().cpu()),
            "final_max_adjacent_palm_step_m": float(selected_steps.max().detach().cpu()),
            "final_contact_boundary_palm_step_m": float(
                selected_steps[max(frame - 1, 0)].detach().cpu()
            ),
            "contact_target_trajectory_enabled": object_surface_targets_global is not None,
            "contact_transition_frames": int(contact_transition_frames),
            "post_contact_relative_velocity_weight": float(
                post_contact_relative_velocity_weight
            ),
            "root_fallback_used": fallback,
            "root_fallback_selected": root_fallback_selected,
            "root_candidate_displacement_change_m": root_candidate_change,
            "root_displacement_change_m": root_change,
            "arm_only_max_arm_rotation_change_deg": arm_rotation_change,
            "arm_root_max_arm_rotation_change_deg": root_candidate_rotation_change,
            "max_arm_rotation_change_deg": final_arm_rotation_change,
            "seed": int(seed),
            "arm_only_guidance": arm_guidance_summary,
            "arm_root_guidance": root_guidance_summary,
        })
        return {
            "initial": compact_baseline,
            "arm_only": compact_arm_only,
            "arm_root": compact_arm_root,
            "selected": compact_selected,
            "sampling_noise": sampling_noise.detach().cpu(),
            "coordinate_transform": {
                "camera_to_genmo_global_R": camera_to_global_R.detach().cpu(),
                "camera_to_genmo_global_t": camera_to_global_t.detach().cpu(),
                "human_depth_scale": human_depth_scale,
            },
            "object_surface_target": target_global.detach().cpu(),
            "diagnostics": diagnostics,
        }
    finally:
        os.chdir(previous_cwd)
