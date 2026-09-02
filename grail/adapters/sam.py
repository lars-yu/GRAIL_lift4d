import os
import random
import sys

import numpy as np
import torch

# Default seed used to make SAM2 video tracking reproducible across reruns on
# the same video. SAM2 uses non-deterministic CUDA kernels by default, so the
# tracked masks (and everything downstream: crop bbox, FoundationPose trajectory,
# interaction start frame, contact labels) drift slightly on each rerun. Pinning
# the seed + enabling deterministic cuDNN keeps mask output stable.
SAM2_SEED = 0


def _import_sam3(sam3_python_path=None):
    """Import SAM3, optionally adding a local checkout to ``sys.path``."""
    if sam3_python_path:
        path = os.path.abspath(os.path.expanduser(str(sam3_python_path)))
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    try:
        import sam3  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SAM3 backend requested but the `sam3` package is unavailable. "
            "Install SAM3 or set sam3.python_path to its source checkout."
        ) from exc
    return sam3


def track_masks_sam3(
    masks,
    video_path,
    device="cuda",
    output_threshold=0.0,
    frame_idx=0,
    checkpoint_path=None,
    sam3_python_path=None,
    compile_model=False,
):
    """Propagate seeded object/human masks with SAM3.

    The return value intentionally matches :func:`track_masks`: a dictionary
    mapping frame index to object id to a numpy binary mask.  This lets the
    rest of GRAIL, including depth alignment and motion-state detection, share
    the same cache format regardless of the segmentation backend.
    """
    if str(device).startswith("cuda") and not __import__("torch").cuda.is_available():
        raise RuntimeError("SAM3 video tracking requires a CUDA device")
    sam3 = _import_sam3(sam3_python_path)
    from sam3.model_builder import build_sam3_video_model

    if not os.path.isfile(video_path) and not os.path.isdir(video_path):
        raise FileNotFoundError(f"SAM3 video/frame directory not found: {video_path}")
    if not masks:
        raise ValueError("SAM3 requires at least one seed mask")

    # SAM3's video loader expects integer-named JPEG frames.  The caller passes
    # the temporary extracted frame directory, so avoid decoding the source a
    # second time and normalize the names only when needed.
    import glob
    import tempfile
    import shutil
    import torch

    if os.path.isdir(video_path):
        frame_paths = sorted(
            glob.glob(os.path.join(video_path, "*.jpg"))
            + glob.glob(os.path.join(video_path, "*.jpeg"))
            + glob.glob(os.path.join(video_path, "*.png"))
        )
        if not frame_paths:
            raise ValueError(f"No frames found for SAM3: {video_path}")
        tmp_frames = tempfile.mkdtemp(prefix="grail_sam3_frames_")
        try:
            from PIL import Image

            for index, source in enumerate(frame_paths):
                target = os.path.join(tmp_frames, f"{index}.jpg")
                if source.lower().endswith((".jpg", ".jpeg")):
                    os.symlink(os.path.abspath(source), target)
                else:
                    Image.open(source).convert("RGB").save(target, quality=95)
            video_input = tmp_frames
        except Exception:
            shutil.rmtree(tmp_frames, ignore_errors=True)
            raise
    else:
        tmp_frames = None
        video_input = video_path

    model = None
    try:
        checkpoint = checkpoint_path
        if checkpoint is not None:
            checkpoint = os.path.abspath(os.path.expanduser(str(checkpoint)))
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
        if checkpoint is None:
            here = os.path.dirname(os.path.abspath(sam3.__file__))
            candidate = os.path.abspath(
                os.path.join(here, "..", "checkpoints", "sam3", "sam3.pt")
            )
            checkpoint = candidate if os.path.isfile(candidate) else None
        load_from_hf = checkpoint is None
        sam3_root = os.path.dirname(os.path.dirname(os.path.abspath(sam3.__file__)))
        bpe_candidates = (
            os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz"),
            os.path.join(sam3_root, "clip", "bpe_simple_vocab_16e6.txt.gz"),
        )
        bpe_path = next((path for path in bpe_candidates if os.path.isfile(path)), None)
        model = build_sam3_video_model(
            checkpoint_path=checkpoint,
            load_from_HF=load_from_hf,
            bpe_path=bpe_path,
            device=device,
            compile=bool(compile_model),
        )
        predictor = model.tracker
        # The tracker needs the detector backbone for image feature extraction.
        predictor.backbone = model.detector.backbone
        inference_state = predictor.init_state(video_path=video_input)
        if hasattr(predictor, "clear_all_points_in_video"):
            predictor.clear_all_points_in_video(inference_state)

        for object_id, seed in enumerate(masks):
            seed = np.asarray(seed).squeeze()
            if seed.ndim != 2 or seed.size == 0:
                raise ValueError(f"SAM3 seed mask {object_id} must be a non-empty [H,W] array")
            predictor.add_new_mask(
                inference_state,
                frame_idx=int(frame_idx),
                obj_id=int(object_id),
                mask=torch.as_tensor(seed > 0, dtype=torch.float32),
            )

        tracked = {}
        frame_count = len(frame_paths) if os.path.isdir(video_path) else None
        kwargs = {
            "start_frame_idx": int(frame_idx),
            "max_frame_num_to_track": frame_count,
            "reverse": False,
            "propagate_preflight": True,
        }
        # Older SAM3 snapshots may not expose propagate_preflight.
        try:
            stream = predictor.propagate_in_video(inference_state, **kwargs)
        except TypeError:
            kwargs.pop("propagate_preflight", None)
            stream = predictor.propagate_in_video(inference_state, **kwargs)
        for output in stream:
            out_frame_idx, out_obj_ids, _low_res, video_res_masks, *_ = output
            tracked[int(out_frame_idx)] = {
                int(out_obj_ids[i]): (
                    (video_res_masks[i] > float(output_threshold))
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                for i in range(len(out_obj_ids))
            }
        expected_count = frame_count
        if expected_count is not None:
            tracked = {
                index: tracked.get(index, {}) for index in range(expected_count)
            }
        return tracked
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if tmp_frames is not None:
            shutil.rmtree(tmp_frames, ignore_errors=True)


def set_sam2_seed(seed=SAM2_SEED):
    """Seed all RNGs and force deterministic cuDNN so SAM2 tracking is reproducible.

    Must be called before constructing the SAM2 predictor / running propagation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN determinism: trade a little speed for reproducible mask logits.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Some ops (e.g. upsampling) need this env var set to run deterministically.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def get_bbox_from_mask(mask, padding_ratio=0.05):
    """
    Extract bounding box coordinates from a binary mask.

    Args:
        mask: Binary mask as numpy array or tensor with shape (H, W) or (H, W, 1)
              Values should be 0 (background) or 1/True (foreground)

    Returns:
        bbox: Bounding box coordinates as [x1, y1, x2, y2] where:
              - x1, y1: top-left corner coordinates
              - x2, y2: bottom-right corner coordinates
              Returns None if mask is empty (no foreground pixels)
    """
    # Convert to numpy if it's a tensor
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()

    # Ensure mask is 2D
    if len(mask.shape) == 3:
        mask = mask.squeeze()

    # Find all foreground pixel coordinates
    rows, cols = np.where(mask > 0)

    # Return None if no foreground pixels found
    if len(rows) == 0:
        return None

    # Calculate bounding box coordinates
    y1, y2 = rows.min(), rows.max()
    x1, x2 = cols.min(), cols.max()
    x_len = x2 - x1
    y_len = y2 - y1
    x_padding = x_len * padding_ratio
    y_padding = y_len * padding_ratio

    # Add padding to the bounding box
    x1 = max(0, x1 - x_padding)
    y1 = max(0, y1 - y_padding)
    x2 = min(mask.shape[1], x2 + x_padding)
    y2 = min(mask.shape[0], y2 + y_padding)

    # Return as [x1, y1, x2, y2] format (standard bbox format)
    return [x1, y1, x2, y2]


def track_masks_from_bbox(bboxes, video_path, device="cuda", output_threshold=0.0, frame_idx=0):
    """
    Track multiple bounding boxes throughout a video using SAM2.

    Args:
        bboxes: List of bounding boxes in format [x1, y1, x2, y2] for the first frame
        video_path: Path to the video file
        device: Device to run inference on
        output_threshold: Threshold for binary mask conversion
        frame_idx: Frame index to initialize bounding boxes (default: 0)

    Returns:
        video_masks: Dictionary mapping frame_idx -> obj_id -> binary_mask
        obj_ids: List of object IDs assigned to each bbox
    """
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    # Reproducibility: pin seeds + deterministic cuDNN before any SAM2 inference.
    set_sam2_seed()

    # Initialize SAM2 predictor
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large", device=device)

    # Create inference state for the video
    inference_state = predictor.init_state(video_path=video_path)
    predictor.reset_state(inference_state)

    # Add each bounding box as a separate object to track
    obj_ids = []
    for i, bbox in enumerate(bboxes):
        # Convert bbox to numpy array if it isn't already
        if not isinstance(bbox, np.ndarray):
            bbox = np.array(bbox)

        # Add bounding box for tracking
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=i,
            box=bbox,
        )
        obj_ids.extend(out_obj_ids)

    # Propagate masks through the video
    video_masks = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        for i, out_obj_id in enumerate(out_obj_ids):
            # Convert logits to binary mask
            binary_mask = (out_mask_logits[i] > output_threshold).cpu().numpy()

            # Store the mask for this frame
            if out_frame_idx not in video_masks:
                video_masks[out_frame_idx] = {}
            video_masks[out_frame_idx][out_obj_id] = binary_mask

    return video_masks


def track_masks(masks, video_path, device="cuda", output_threshold=0.0, frame_idx=0):
    """
    Track masks throughout a video using SAM2.

    Args:
        masks: List of binary masks for the first frame. Each mask should be a numpy array
               with shape (H, W) where values are 0 (background) or 1/True (foreground)
        video_path: Path to the video file
        device: Device to run inference on
        output_threshold: Threshold for binary mask conversion
        frame_idx: Frame index to initialize masks (default: 0)

    Returns:
        video_masks: Dictionary mapping frame_idx -> obj_id -> binary_mask
    """
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    # Reproducibility: pin seeds + deterministic cuDNN before any SAM2 inference.
    set_sam2_seed()

    # Initialize SAM2 predictor
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large", device=device)

    # Create inference state for the video
    inference_state = predictor.init_state(video_path=video_path)
    predictor.reset_state(inference_state)

    # Add each mask as a separate object to track
    obj_ids = []
    for i, mask in enumerate(masks):
        # Convert mask to numpy if it's a tensor
        if hasattr(mask, "cpu"):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask

        # Ensure mask is 2D
        if len(mask_np.shape) == 3:
            mask_np = mask_np.squeeze()

        # Convert to boolean/binary if needed
        mask_np = (mask_np > 0).astype(np.uint8)

        # Add mask for tracking
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=i,
            mask=mask_np,
        )
        obj_ids.extend(out_obj_ids)

    # Propagate masks through the video
    video_masks = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        for i, out_obj_id in enumerate(out_obj_ids):
            # Convert logits to binary mask
            binary_mask = (out_mask_logits[i] > output_threshold).cpu().numpy()

            # Store the mask for this frame
            if out_frame_idx not in video_masks:
                video_masks[out_frame_idx] = {}
            video_masks[out_frame_idx][out_obj_id] = binary_mask

    return video_masks
