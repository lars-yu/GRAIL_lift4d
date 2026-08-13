"""Evaluation and pre-evaluation logic for the HOI optimizer."""

import torch
import torch.nn.functional as F

from grail.rendering.renderer import RendererType, create_renderer, render_frame
from grail.rendering.textures import create_colored_meshes


@torch.no_grad()
def pre_eval(data, cameras, pre_eval_cfg, min_frames_threshold, device, logger):
    """
    Pre-evaluation: check if initial object pose is reasonable by comparing
    rendered object masks with ground truth object masks.

    Args:
        data: HOIData instance
        cameras: PyTorch3D cameras
        pre_eval_cfg: dict with per_frame_tol and total_tol
        min_frames_threshold: minimum frames before truncation is allowed (or None)
        device: torch device
        logger: logger instance

    Returns:
        tuple: (passed: bool, failed_frame: int or None)
            - passed: True if initial object pose is reasonable
            - failed_frame: frame index where alignment failed (for truncation), or None
    """
    obj_verts_seq = data.obj.verts_seq
    obj_faces = data.obj.faces
    obj_colors = torch.tensor([0.0, 0.0, 1.0], device=device)

    tol = pre_eval_cfg.get("per_frame_tol", 0.5)
    total_tol = pre_eval_cfg.get("total_tol", 0.3)

    total_err = 0
    frame_num = len(obj_verts_seq)
    failed_frame = None

    print("Pre-evaluation: Checking initial object mask alignment...")
    if min_frames_threshold is not None:
        logger.info(f"Min frames threshold for truncation: {min_frames_threshold}")

    eval_mask_renderer = create_renderer(
        cameras,
        (data.camera.frame_height, data.camera.frame_width),
        renderer_type=RendererType.HARD_PHONG,
        neutral_light=True,
        background_color=[0, 0, 0],
    )

    for i in range(frame_num):
        obj_mesh = create_colored_meshes(obj_verts_seq[i], obj_faces, obj_colors)
        _, pred_obj_mask = render_frame(obj_mesh, cameras, eval_mask_renderer, require_grad=False)
        pred_obj_mask = (pred_obj_mask > 0.1).float()

        gt_obj_mask = torch.from_numpy(data.obj.masks[i]).to(device).squeeze(0).float()

        if gt_obj_mask.shape != pred_obj_mask.shape:
            gt_obj_mask = gt_obj_mask.unsqueeze(0).unsqueeze(0)
            gt_obj_mask = F.interpolate(
                gt_obj_mask, size=pred_obj_mask.shape, mode="bilinear", align_corners=False
            )
            gt_obj_mask = gt_obj_mask.squeeze(0).squeeze(0)

        err = ((1 - pred_obj_mask) * gt_obj_mask).sum() / (gt_obj_mask.sum() + 100)
        total_err += err

        if err > tol:
            failed_frame = i
            logger.warning(f"Pre-eval: Object mask not aligned at frame {i}: {err:.4f} > {tol}")
            break

    # Handle failure
    if failed_frame is not None:
        if min_frames_threshold is None or failed_frame < min_frames_threshold:
            logger.error(
                f"Pre-eval failed: Alignment failed at frame {failed_frame}, "
                f"below minimum threshold {min_frames_threshold or 'N/A'}. Aborting."
            )
            return False, None
        else:
            logger.info(
                f"Pre-eval: Truncating to {failed_frame} frames "
                f"(alignment failed, above threshold {min_frames_threshold})"
            )
            frame_num = failed_frame
            return True, failed_frame

    # Check average error
    if frame_num > 0:
        avg_err = total_err / frame_num
        if avg_err > total_tol:
            logger.error(
                f"Pre-eval failed: Average mask difference too large: {avg_err:.4f} > {total_tol}"
            )
            return False, None
        logger.info(
            f"Pre-eval passed: Object mask aligned (avg error: {avg_err:.4f} < {total_tol}, "
            f"frames: {frame_num})"
        )

    return True, None


def truncate_data(data, new_frame_num, logger=None):
    """Truncate all sequence data in HOIData to new_frame_num frames."""
    import torch

    if logger:
        logger.info(f"Truncating data from {data.frame_num} to {new_frame_num} frames")

    data.frame_num = new_frame_num

    # Truncate human motion data tensors
    motion_data = data.human.motion_data
    for key in [
        "poses",
        "trans",
        "betas",
        "left_hand_pose",
        "right_hand_pose",
        "vitpose",
        "hand_keypoints_2d",
    ]:
        if key in motion_data and isinstance(motion_data[key], torch.Tensor):
            if motion_data[key].shape[0] >= new_frame_num:
                motion_data[key] = motion_data[key][:new_frame_num]

    data.human.body_keypoints_seq = data.human.body_keypoints_seq[:new_frame_num]
    data.human.hand_keypoints_seq = data.human.hand_keypoints_seq[:new_frame_num]
    data.human.masks = data.human.masks[:new_frame_num]

    # Truncate object data
    data.obj.verts_seq = data.obj.verts_seq[:new_frame_num]
    data.obj.poses = data.obj.poses[:new_frame_num]
    data.obj.verts_tracking_seq = data.obj.verts_tracking_seq[:new_frame_num]
    data.obj.masks = data.obj.masks[:new_frame_num]

    # Truncate other sequence data
    data.images_path = data.images_path[:new_frame_num]
    data.depth_maps = data.depth_maps[:new_frame_num]

    if logger:
        logger.info(f"Data truncation complete. New frame count: {new_frame_num}")
