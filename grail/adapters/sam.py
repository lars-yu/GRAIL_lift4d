import os
import random

import numpy as np
import torch

# Default seed used to make SAM2 video tracking reproducible across reruns on
# the same video. SAM2 uses non-deterministic CUDA kernels by default, so the
# tracked masks (and everything downstream: crop bbox, FoundationPose trajectory,
# interaction start frame, contact labels) drift slightly on each rerun. Pinning
# the seed + enabling deterministic cuDNN keeps mask output stable.
SAM2_SEED = 0


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
