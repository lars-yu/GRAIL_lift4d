"""Interaction detection and contact label management.

Provides functions to detect when human-object interaction starts/ends
in a video sequence, using either object pose trajectories or 2D mask tracking.
"""

import cv2
import numpy as np
import torch

from grail.core.contact_label import detect_interaction


def get_contact_labels_for_frame(frame_idx, contact_labels, contact_start_idx, contact_interval):
    """Return the contact labels for a specific frame based on its interval.

    Returns None if the frame is before the interaction start or if the contact interval is not set.
    """
    if frame_idx < contact_start_idx:
        return None
    if not contact_interval:
        return None
    interval_idx = (frame_idx - contact_start_idx) // contact_interval
    if interval_idx >= len(contact_labels):
        interval_idx = len(contact_labels) - 1
    labels = contact_labels[interval_idx]
    if isinstance(labels, str):  # backward compatibility with old caches
        labels = [labels]
    return labels


def identify_interaction_start_end(
    obj_poses, image_list, obj_name, logger, t_threshold=0.04, interval=8, has_interaction_end=False
):
    """Detect interaction start/end frames from object pose trajectories.

    Uses object translation velocity to detect when the object starts moving.
    Falls back to LLM-based detection if no motion is found.

    Returns:
        inter_start_idx: Frame index where interaction starts
        inter_end_idx: Frame index where interaction ends
        is_static_obj: Whether the object is static throughout
    """
    T = len(obj_poses)
    is_static_obj = True
    inter_start_idx = None

    for i in range(24, T):
        obj_pose = obj_poses[i]
        obj_t = obj_pose[:3, 3]
        last_obj_t = obj_poses[max(0, i - interval)][:3, 3]
        diff_t = obj_t - last_obj_t

        if torch.norm(diff_t, dim=0) > t_threshold:
            if i - interval < 0:
                logger.error("Interaction start frame is too close to the beginning of the video")
                raise ValueError(
                    "Interaction start frame is too close to the beginning of the video"
                )
            inter_start_idx = i - interval
            is_static_obj = False
            break

    if inter_start_idx is None:
        num_keyframe = 8
        interval = T // num_keyframe
        for i in range(T - interval, -1, -interval):
            image_path = image_list[i]
            is_interacting = detect_interaction(image_path, obj_name)
            if is_interacting:
                inter_start_idx = i
                break

    if inter_start_idx is None:
        inter_start_idx = T

    inter_end_idx = T
    return inter_start_idx, inter_end_idx, is_static_obj


def identify_interaction_start_end_with_mask(
    masks,
    image_list,
    logger,
    t_threshold=3.0,
    interval=2,
    min_moving_frames=12,
    min_static_frames=12,
    has_interaction_end=False,
):
    """Detect interaction start/end frames using 2D object mask center trajectory.

    For start: finds where the mask center starts to move constantly.
    For end: finds where it stops moving and remains stationary until the end.

    Args:
        masks: Dictionary of masks with frame_idx as keys, masks[frame_idx][1] is object mask
        image_list: List of image paths
        logger: Logger instance
        t_threshold: Threshold for mask center movement (in pixels)
        interval: Interval for computing velocity
        min_moving_frames: Minimum consecutive moving frames to confirm start
        min_static_frames: Minimum consecutive static frames to confirm end
        has_interaction_end: If True, detect interaction end

    Returns:
        inter_start_idx, inter_end_idx, is_static_obj
    """
    T = len(image_list)

    # Compute mask centers for all frames
    mask_centers = []
    for i in range(T):
        if i in masks and masks[i][1] is not None:
            obj_mask = masks[i][1]
            if isinstance(obj_mask, np.ndarray):
                obj_mask_np = obj_mask
            else:
                obj_mask_np = obj_mask.cpu().numpy() if hasattr(obj_mask, "cpu") else obj_mask
            if obj_mask_np.ndim == 3:
                obj_mask_np = obj_mask_np.squeeze()
            moments = cv2.moments(obj_mask_np.astype(np.uint8))
            if moments["m00"] > 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
                mask_centers.append(np.array([cx, cy]))
            else:
                if len(mask_centers) > 0:
                    mask_centers.append(mask_centers[-1].copy())
                else:
                    mask_centers.append(np.array([0.0, 0.0]))
        else:
            if len(mask_centers) > 0:
                mask_centers.append(mask_centers[-1].copy())
            else:
                mask_centers.append(np.array([0.0, 0.0]))

    mask_centers = np.array(mask_centers)

    # Compute velocities
    velocities = np.zeros(T)
    for i in range(interval, T):
        diff = mask_centers[i] - mask_centers[i - interval]
        velocities[i] = np.linalg.norm(diff)

    logger.info(
        f"Mask center velocity stats - min: {velocities.min():.2f}, max: {velocities.max():.2f}, "
        f"mean: {velocities.mean():.2f}, threshold: {t_threshold}"
    )

    # Identify interaction start
    is_static_obj = True
    inter_start_idx = None

    for i in range(24, T - min_moving_frames):
        if velocities[i] > t_threshold:
            moving_count = 0
            for j in range(i, min(i + min_moving_frames, T)):
                if velocities[j] > t_threshold * 0.5:
                    moving_count += 1

            if moving_count >= min_moving_frames * 0.7:
                inter_start_idx = max(0, i - interval)
                is_static_obj = False
                break

    if inter_start_idx is None:
        raise ValueError("No interaction found")

    # Identify interaction end
    inter_end_idx = T

    if has_interaction_end:
        for i in range(T - 1, inter_start_idx + min_static_frames, -1):
            static_count = 0
            for j in range(i, T):
                if velocities[j] <= t_threshold * 0.5:
                    static_count += 1
            remaining_frames = T - i
            if remaining_frames > 0 and static_count >= remaining_frames * 0.8:
                if i > inter_start_idx and velocities[i - 1] > t_threshold * 0.5:
                    inter_end_idx = i
                    break

    logger.info(
        f"Mask-based detection - Start: {inter_start_idx}, End: {inter_end_idx}, Static: {is_static_obj}"
    )

    return inter_start_idx, inter_end_idx, is_static_obj
