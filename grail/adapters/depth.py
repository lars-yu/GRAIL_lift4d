import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def est_depth(
    image_list,
    device="cuda",
    intrinsics=None,
    gt_depth_first_frame=None,
    method="moge",
    encoder="vitb",
    metric=True,
    input_size=518,
    max_res=1280,
    target_fps=-1,
    fp32=False,
    resolution_level=9,
    use_fp16=True,
):
    """Estimate per-frame depth, dispatching on ``method``.

    The original signature (``image_list, device, intrinsics,
    gt_depth_first_frame``) is preserved; the remaining kwargs are optional and
    default to the original MoGe behaviour, so existing call sites are unchanged.

    Args:
        image_list: List of image paths.
        device: Device to run on.
        intrinsics: Optional camera intrinsics (3, 3) numpy array or path to
            intrinsics file. Used by MoGe (as horizontal FOV); ignored by VDA's
            metric model.
        gt_depth_first_frame: Optional ground truth depth for the first frame
            (H, W) in meters. Currently unused; kept for API symmetry.
        method: "moge" (per-frame metric depth, default) | "vda"
            (VideoDepthAnything, video-consistent depth) | "foundationgeo"
            (DINOv3-based per-frame metric depth; honors known intrinsics).
        encoder: VDA only — "vits" | "vitb" | "vitl".
        metric: VDA only — use the metric checkpoint (depth in meters, drop-in
            for MoGe) if True, else the relative (affine-invariant) checkpoint.
        input_size: VDA only — internal input size in pixels.
        max_res: VDA only — optionally downscale frames so max(H, W) <= max_res
            to bound memory (depth is resized back to the original image size).
        target_fps: VDA only — passed through to infer_video_depth (unused
            internally; -1 means unknown).
        fp32: VDA only — run inference in float32 (default fp16 via autocast).
        resolution_level: FoundationGeo only — integer [0-9] controlling the
            number of tokens; higher = finer detail but slower/more memory.
        use_fp16: FoundationGeo only — run inference in fp16 (default).

    Returns:
        list of (H, W) depth tensors on ``device``. Metric methods return meters.
    """
    if method == "moge":
        return _est_depth_moge(image_list, device, intrinsics, gt_depth_first_frame)
    if method == "vda":
        return _est_depth_vda(
            image_list,
            device=device,
            encoder=encoder,
            metric=metric,
            input_size=input_size,
            max_res=max_res,
            target_fps=target_fps,
            fp32=fp32,
            intrinsics=intrinsics,
        )
    if method == "foundationgeo":
        return _est_depth_foundationgeo(
            image_list,
            device=device,
            intrinsics=intrinsics,
            gt_depth_first_frame=gt_depth_first_frame,
            resolution_level=resolution_level,
            use_fp16=use_fp16,
        )
    raise ValueError(
        f"Unknown depth method: {method!r} (expected 'moge', 'vda', or 'foundationgeo')"
    )


def _est_depth_moge(image_list, device="cuda", intrinsics=None, gt_depth_first_frame=None):
    """Estimate per-frame metric depth with MoGe-2.

    Args:
        image_list: List of image paths.
        device: Device to run on.
        intrinsics: Optional camera intrinsics (3, 3) numpy array or path to
            intrinsics file. Same intrinsics will be used for all images
            (scaled appropriately if images are different sizes).
        gt_depth_first_frame: Optional ground truth depth for the first frame
            (H, W) in meters. Currently unused; kept for API symmetry.
    """
    moge_dir = os.path.join(os.path.dirname(__file__), "..", "..", "imports", "MoGe")
    moge_dir = os.path.abspath(moge_dir)
    if moge_dir not in sys.path:
        sys.path.insert(0, moge_dir)

    from moge.model.v2 import MoGeModel

    print("Loading MoGe-2 model (Ruicheng/moge-2-vitl-normal)...")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()

    original_image_size = cv2.imread(image_list[0]).shape[:2]

    # Convert camera intrinsics to horizontal FOV in degrees
    fov_x = None
    if intrinsics is not None:
        if isinstance(intrinsics, str):
            K = np.loadtxt(intrinsics)
        else:
            K = np.array(intrinsics)
        fx = K[0, 0]
        W = original_image_size[1]
        fov_x = 2 * math.atan(W / (2 * fx)) * 180.0 / math.pi
        print(f"Using known intrinsics: fx={fx:.2f}, image_width={W}, fov_x={fov_x:.2f} degrees")
    else:
        print("No intrinsics provided, MoGe will recover focal length internally")

    depth_list = []
    for image_path in tqdm(image_list, desc="Estimating depth (MoGe)"):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(
            2, 0, 1
        )

        with torch.no_grad():
            output = model.infer(image_tensor, fov_x=fov_x, resolution_level=9, use_fp16=True)

        depth = output["depth"]  # (H, W), metric meters

        # Replace invalid pixels (inf) with 0
        depth = torch.where(torch.isinf(depth), torch.zeros_like(depth), depth)

        # Resize to original image size if needed
        if depth.shape != tuple(original_image_size):
            depth = (
                F.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
                    size=original_image_size,
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
            )

        depth_list.append(depth)

    del model
    torch.cuda.empty_cache()
    print(f"MoGe: produced {len(depth_list)} metric depth frames")

    return depth_list


# ── VideoDepthAnything ───────────────────────────────────────────────────────

_VDA_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _est_depth_vda(
    image_list,
    device="cuda",
    encoder="vitb",
    metric=True,
    input_size=518,
    max_res=1280,
    target_fps=-1,
    fp32=False,
    intrinsics=None,
):
    """Estimate temporally-consistent per-frame depth with VideoDepthAnything.

    The metric variant returns depth in meters (drop-in for MoGe-2). The relative
    variant returns affine-invariant depth and must be paired with the existing
    GT-depth alignment (``align_depth_with_gt``) to recover metric scale.

    Output contract matches ``_est_depth_moge``: a list of (H, W) tensors on
    ``device``, invalid pixels replaced with 0, resized to the original image size.
    """
    vda_dir = os.path.join(os.path.dirname(__file__), "..", "..", "imports", "VideoDepthAnything")
    vda_dir = os.path.abspath(vda_dir)
    if vda_dir not in sys.path:
        sys.path.insert(0, vda_dir)

    from video_depth_anything.video_depth import VideoDepthAnything

    if encoder not in _VDA_MODEL_CONFIGS:
        raise ValueError(
            f"Unknown VDA encoder: {encoder!r} (expected one of {list(_VDA_MODEL_CONFIGS)})"
        )

    ckpt_name = "metric_video_depth_anything" if metric else "video_depth_anything"
    ckpt_path = os.path.join(vda_dir, "checkpoints", f"{ckpt_name}_{encoder}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"VDA checkpoint not found at {ckpt_path}. Download "
            f"{os.path.basename(ckpt_path)} from https://huggingface.co/depth-anything/"
            f"{'Metric-' if metric else ''}Video-Depth-Anything"
            f" and place it under imports/VideoDepthAnything/checkpoints/."
        )

    print(f"Loading VideoDepthAnything (encoder={encoder}, metric={metric}) from {ckpt_path} ...")
    model = VideoDepthAnything(**_VDA_MODEL_CONFIGS[encoder], metric=metric)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    model = model.to(device).eval()

    if intrinsics is not None:
        print(
            "VideoDepthAnything: intrinsics/FOV ignored "
            "(the metric model is self-calibrated; GT-depth alignment recovers scale when available)."
        )

    original_image_size = cv2.imread(image_list[0]).shape[:2]  # (H, W)

    # Read all frames (BGR -> RGB), optionally downscale by max_res to bound memory.
    H0, W0 = original_image_size
    scale = 1.0
    if max_res and max(H0, W0) > max_res:
        scale = float(max_res) / float(max(H0, W0))
    target_hw = (int(round(H0 * scale)), int(round(W0 * scale)))

    frames = []
    for image_path in tqdm(image_list, desc="Reading frames (VDA)"):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if scale != 1.0:
            img = cv2.resize(img, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
        frames.append(img)
    frames = np.stack(frames, axis=0)  # (N, H, W, 3) uint8

    print(
        f"Running VideoDepthAnything on {len(frames)} frames "
        f"(input_size={input_size}, max_res={'off' if scale == 1.0 else max_res}, fp32={fp32}) ..."
    )
    try:
        with torch.no_grad():
            depths, _ = model.infer_video_depth(
                frames, target_fps=target_fps, input_size=input_size, device=device, fp32=fp32
            )
    except torch.cuda.OutOfMemoryError as e:
        raise torch.cuda.OutOfMemoryError(
            f"VideoDepthAnything ran out of GPU memory on {len(frames)} frames "
            f"(input_size={input_size}, max_res={max_res}). VDA batches 32 frames per chunk, "
            f"so memory scales with input_size. Options: lower depth.input_size (e.g. 392) or "
            f"depth.max_res in the config, or set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"to reduce fragmentation."
        ) from e
    # depths: np.ndarray (N, H, W) float — meters when metric=True.

    depth_list = []
    for i in range(depths.shape[0]):
        depth = torch.from_numpy(np.ascontiguousarray(depths[i])).to(device)
        depth = torch.where(
            torch.isinf(depth) | torch.isnan(depth), torch.zeros_like(depth), depth
        )
        # Resize back to the original image size if frames were downscaled for VDA,
        # so downstream masks / intrinsics stay aligned.
        if depth.shape != tuple(original_image_size):
            depth = (
                F.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
                    size=original_image_size,
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
            )
        depth_list.append(depth)

    del model
    torch.cuda.empty_cache()
    print(
        f"VideoDepthAnything: produced {len(depth_list)} depth frames (metric={metric})"
    )

    return depth_list


# ── FoundationGeo ─────────────────────────────────────────────────────────────

_FGEO_UTILS3D_PATCHED = False


def _apply_fgeo_utils3d_patch():
    """Make ``utils3d.torch.intrinsics_from_focal_center`` device-safe for GPU inference.

    ``FoundationGeo.infer`` calls
    ``utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)`` (v1.py:482). On
    GPU, ``fx``/``fy`` are CUDA tensors but utils3d's ``@totensor`` decorator
    materializes the scalar ``0.5`` as a CPU tensor, so the internal ``torch.stack``
    raises "Expected all tensors to be on the same device, cuda:0 and cpu". This
    fires on every ``infer`` call regardless of whether ``fov_x`` is passed.

    The patch pre-converts scalar args to tensors on the reference (first-tensor)
    device before delegating; behavior is unchanged when all args are already on the
    same (or CPU) device. Applied lazily and once, only when FoundationGeo is used,
    so the MoGe/VDA paths are unaffected.
    """
    global _FGEO_UTILS3D_PATCHED
    if _FGEO_UTILS3D_PATCHED:
        return
    import utils3d

    _orig = utils3d.torch.intrinsics_from_focal_center

    def _patched(fx, fy, cx, cy):
        ref = next((a for a in (fx, fy, cx, cy) if isinstance(a, torch.Tensor)), None)
        dev = ref.device if ref is not None else None
        dt = ref.dtype if ref is not None else torch.float32

        def _t(a):
            if isinstance(a, torch.Tensor):
                return a.to(dev) if (dev is not None and a.device != dev) else a
            return torch.tensor(a, device=dev, dtype=dt)

        return _orig(_t(fx), _t(fy), _t(cx), _t(cy))

    utils3d.torch.intrinsics_from_focal_center = _patched
    _FGEO_UTILS3D_PATCHED = True


def _est_depth_foundationgeo(
    image_list,
    device="cuda",
    intrinsics=None,
    gt_depth_first_frame=None,
    resolution_level=9,
    use_fp16=True,
):
    """Estimate per-frame metric depth with FoundationGeo (DINOv3-vitl16).

    Mirrors ``_est_depth_moge``: per-frame loop, BGR->RGB on load, ``fov_x``
    computed from the *known* GRAIL intrinsics so the model honors them instead of
    self-estimating the camera, invalid pixels replaced with 0, and a resize-guard
    to the original image size. Reads ``output["depth_metric"]`` (meters).

    Args:
        image_list: List of image paths.
        device: Device to run on.
        intrinsics: Optional camera intrinsics (3, 3) numpy array or path to
            intrinsics file. Used to compute the horizontal FOV passed to the model;
            if None, FoundationGeo estimates the focal length itself (not preferred
            for GRAIL, where intrinsics are known).
        gt_depth_first_frame: Optional ground truth depth for the first frame
            (H, W) in meters. Currently unused; kept for API symmetry.
        resolution_level: integer [0-9] controlling the number of tokens; higher
            = finer detail but slower / more memory (default 9).
        use_fp16: run inference in fp16 (default True).
    """
    fgeo_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "imports", "FoundationGeo", "FoundationGeo"
    )
    fgeo_dir = os.path.abspath(fgeo_dir)
    if fgeo_dir not in sys.path:
        sys.path.insert(0, fgeo_dir)

    ckpt_path = os.path.join(fgeo_dir, "..", "model", "FoundationGeo.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"FoundationGeo checkpoint not found at {ckpt_path}. Expected the "
            f"weight at imports/FoundationGeo/model/FoundationGeo.pt."
        )

    _apply_fgeo_utils3d_patch()

    from foundationgeo.model.v1 import FoundationGeo

    print(f"Loading FoundationGeo from {ckpt_path} ...")
    model = FoundationGeo.from_pretrained(ckpt_path).to(device).eval()

    original_image_size = cv2.imread(image_list[0]).shape[:2]

    # Convert known camera intrinsics to horizontal FOV in degrees (same as MoGe).
    # Passing fov_x makes FoundationGeo use the known camera instead of estimating
    # its own focal length.
    fov_x = None
    if intrinsics is not None:
        if isinstance(intrinsics, str):
            K = np.loadtxt(intrinsics)
        else:
            K = np.array(intrinsics)
        fx = K[0, 0]
        W = original_image_size[1]
        fov_x = 2 * math.atan(W / (2 * fx)) * 180.0 / math.pi
        print(f"Using known intrinsics: fx={fx:.2f}, image_width={W}, fov_x={fov_x:.2f} degrees")
    else:
        print(
            "No intrinsics provided — FoundationGeo will estimate the focal length "
            "itself (GRAIL usually supplies known intrinsics)."
        )

    depth_list = []
    for image_path in tqdm(image_list, desc="Estimating depth (FoundationGeo)"):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(
            2, 0, 1
        )

        with torch.no_grad():
            output = model.infer(
                image_tensor,
                fov_x=fov_x,
                resolution_level=resolution_level,
                use_fp16=use_fp16,
            )

        depth = output["depth_metric"]  # (H, W), metric meters; inf where masked

        # Replace invalid pixels (inf/nan) with 0
        depth = torch.where(
            torch.isinf(depth) | torch.isnan(depth), torch.zeros_like(depth), depth
        )

        # Resize to original image size if needed (the model already returns
        # input-size output, so this is a no-op safety net).
        if depth.shape != tuple(original_image_size):
            depth = (
                F.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
                    size=original_image_size,
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
            )

        depth_list.append(depth)

    del model
    torch.cuda.empty_cache()
    print(f"FoundationGeo: produced {len(depth_list)} metric depth frames")

    return depth_list
