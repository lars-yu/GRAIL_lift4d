import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R


def smooth_pose_sequence(poses, window_length=11, polyorder=2):
    """
    Apply Savitzky-Golay filter to smooth pose sequences along the time dimension.

    Args:
        poses: tensor of shape (frame_num, ...) where ... can be any number of dimensions
        window_length: length of filter window (must be odd, >=3)
        polyorder: polynomial order for fitting (must be less than window_length)

    Returns:
        smoothed poses with same shape as input
    """
    if poses.shape[0] < window_length:
        # Not enough frames to apply smoothing, return as is
        return poses

    # Convert to numpy for savgol_filter
    poses_np = poses.detach().cpu().numpy()
    original_shape = poses_np.shape

    # Reshape to (frame_num, -1) for processing
    poses_flat = poses_np.reshape(poses_np.shape[0], -1)

    # Apply Savitzky-Golay filter to each dimension independently
    smoothed = np.zeros_like(poses_flat)
    for i in range(poses_flat.shape[1]):
        smoothed[:, i] = savgol_filter(poses_flat[:, i], window_length, polyorder, mode="nearest")

    # Reshape back to original shape and convert back to tensor
    smoothed = smoothed.reshape(original_shape)
    return torch.from_numpy(smoothed).to(poses.device).type(poses.dtype)


def smooth_pose_matrices(pose_list, window_length=9, polyorder=3):
    """
    Apply Savitzky-Golay filter to smooth 4x4 pose matrices (rotation + translation).
    Rotations are converted to quaternion representation before smoothing to maintain validity.
    Quaternions are better than axis-angle for interpolation (no discontinuities).

    Args:
        pose_list: list of numpy arrays, each of shape (4, 4) representing SE(3) poses
        window_length: length of filter window (must be odd, >=3)
        polyorder: polynomial order for fitting (must be less than window_length)

    Returns:
        list of smoothed 4x4 pose matrices
    """
    num_frames = len(pose_list)

    if num_frames < window_length:
        # Not enough frames to apply smoothing, return as is
        return pose_list

    # Extract rotations and translations
    rotations = np.array([pose[:3, :3] for pose in pose_list])  # (num_frames, 3, 3)
    translations = np.array([pose[:3, 3] for pose in pose_list])  # (num_frames, 3)

    # Convert rotation matrices to quaternions (w, x, y, z)
    quats = np.array(
        [R.from_matrix(rot).as_quat() for rot in rotations]
    )  # (num_frames, 4) - scipy uses (x,y,z,w)

    # Ensure quaternion continuity - flip quaternions that are on opposite hemisphere
    # q and -q represent the same rotation, so we need to ensure continuity
    for i in range(1, num_frames):
        # If dot product is negative, quaternions are on opposite hemispheres
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    # Smooth quaternions
    smoothed_quats = np.zeros_like(quats)
    for i in range(4):
        smoothed_quats[:, i] = savgol_filter(quats[:, i], window_length, polyorder, mode="nearest")

    # Normalize quaternions after smoothing (required to maintain valid rotations)
    smoothed_quats = smoothed_quats / np.linalg.norm(smoothed_quats, axis=1, keepdims=True)

    # Smooth translations
    smoothed_translations = np.zeros_like(translations)
    for i in range(3):
        smoothed_translations[:, i] = savgol_filter(
            translations[:, i], window_length, polyorder, mode="nearest"
        )

    # Convert back to rotation matrices and reconstruct 4x4 poses
    smoothed_poses = []
    for i in range(num_frames):
        pose = np.eye(4)
        pose[:3, :3] = R.from_quat(smoothed_quats[i]).as_matrix()
        pose[:3, 3] = smoothed_translations[i]
        smoothed_poses.append(pose)

    return smoothed_poses


def smooth_axis_angle_sequence(axis_angles, window_length=11, polyorder=2):
    """
    Apply Savitzky-Golay filter to smooth axis-angle rotation sequences.
    Converts to quaternions for smoothing to avoid discontinuities, then converts back.

    Args:
        axis_angles: tensor of shape (frame_num, ..., 3) where last dim is axis-angle
        window_length: length of filter window (must be odd, >=3)
        polyorder: polynomial order for fitting (must be less than window_length)

    Returns:
        smoothed axis-angles with same shape as input
    """
    # Handle input
    original_device = axis_angles.device
    original_dtype = axis_angles.dtype
    axis_angles_np = axis_angles.detach().cpu().numpy()
    original_shape = axis_angles_np.shape

    # Flatten to (frame_num, -1, 3) for easier processing
    frame_num = original_shape[0]
    axis_angles_flat = axis_angles_np.reshape(frame_num, -1, 3)
    num_joints = axis_angles_flat.shape[1]

    if frame_num < window_length:
        # Not enough frames to apply smoothing, return as is
        return axis_angles

    # Process each joint separately
    smoothed_flat = np.zeros_like(axis_angles_flat)

    for joint_idx in range(num_joints):
        joint_rotvecs = axis_angles_flat[:, joint_idx, :]  # (frame_num, 3)

        # Convert to quaternions
        quats = np.array([R.from_rotvec(rv).as_quat() for rv in joint_rotvecs])  # (frame_num, 4)

        # Ensure quaternion continuity - flip quaternions that are on opposite hemisphere
        for i in range(1, frame_num):
            if np.dot(quats[i], quats[i - 1]) < 0:
                quats[i] = -quats[i]

        # Smooth quaternions
        smoothed_quats = np.zeros_like(quats)
        for i in range(4):
            smoothed_quats[:, i] = savgol_filter(
                quats[:, i], window_length, polyorder, mode="nearest"
            )

        # Normalize quaternions after smoothing
        smoothed_quats = smoothed_quats / np.linalg.norm(smoothed_quats, axis=1, keepdims=True)

        # Convert back to axis-angle
        smoothed_rotvecs = np.array([R.from_quat(q).as_rotvec() for q in smoothed_quats])
        smoothed_flat[:, joint_idx, :] = smoothed_rotvecs

    # Reshape back to original shape
    smoothed = smoothed_flat.reshape(original_shape)

    # Convert back to tensor
    return torch.from_numpy(smoothed).to(original_device).type(original_dtype)
