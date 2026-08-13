#!/usr/bin/env python3
"""Object pose estimation orchestration.

Wraps FoundationPose with optional cropping and frame interpolation.
"""

import glob
import json
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np
import torch

from grail.constants.image import HEIGHT, WIDTH
from grail.core.io import load_init_rendering_data, run_subprocess, save_object_pose_data
from grail.core.video import (
    extract_frames_from_video,
    get_video_fps_and_frame_count,
    save_images_to_video,
)
from grail.dynamic_camera.conventions import convert_blender_to_internal
from grail.rendering.camera import world_to_camera_matrix

# ---------------------------------------------------------------------------
# Video interpolation & subsampling
# ---------------------------------------------------------------------------


def interpolate_video(input_video_path, output_video_path, interpolation_factor=2):
    """Interpolate video frames using FFmpeg minterpolate. Returns True on success."""
    input_fps, _ = get_video_fps_and_frame_count(input_video_path)
    target_fps = input_fps * interpolation_factor

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_video_path,
        "-filter:v",
        f"minterpolate='mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:fps={target_fps}'",
        "-pix_fmt",
        "yuv420p",
        "-q:v",
        "2",
        output_video_path,
    ]

    print(f"  Interpolating video {input_fps:.0f}fps → {target_fps:.0f}fps")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FFmpeg interpolation failed: {e.stderr.decode() if e.stderr else e}")
        return False


def _get_sample_indices(total, target):
    """Get evenly spaced indices to pick *target* items from *total*."""
    if target >= total:
        return list(range(total))
    return [round(i * (total - 1) / (target - 1)) for i in range(target)]


def _effective_interpolation_factor(interpolation_factor, intrinsics_npy=None, c2w_blender_npy=None):
    """Disable frame interpolation when using dynamic per-frame cameras.

    Dynamic VGGT mode supplies one calibrated K/c2w per generated frame. Running
    FoundationPose on ffmpeg-interpolated intermediate frames would require
    interpolated camera intrinsics and poses; repeating the last K is wrong for
    moving cameras. Keep tracking on the original frame sequence in v1.
    """
    if (intrinsics_npy is not None or c2w_blender_npy is not None) and int(interpolation_factor) > 1:
        return 1
    return int(interpolation_factor)


def _write_poses_in_cam_metadata(
    input_dir,
    *,
    pose_count,
    camera_mode="fixed",
    uses_dynamic_camera=False,
    uses_dynamic_intrinsics=False,
    c2w_blender_npy=None,
    intrinsics_npy=None,
    dynamic_intrinsics_path=None,
    init_pose_cam_path=None,
    object_init_pose_npy=None,
    is_static=False,
    requested_interpolation_factor=1,
    interpolation_factor=1,
    crop_bbox=None,
):
    """Write auditable metadata for cached FoundationPose camera-space poses."""
    pe_dir = os.path.join(input_dir, "pose_estimation_output")
    os.makedirs(pe_dir, exist_ok=True)
    metadata = {
        "pose_count": int(pose_count),
        "input_coordinate_space": "opencv_camera_T_C<-O",
        "camera_mode": camera_mode,
        "uses_dynamic_camera": bool(uses_dynamic_camera),
        "uses_dynamic_intrinsics": bool(uses_dynamic_intrinsics),
        "foundationpose_rgb_only": True,
        "vggt_depth_used_by_foundationpose": False,
        "is_static_object_path": bool(is_static),
        "requested_interpolation_factor": int(requested_interpolation_factor),
        "effective_interpolation_factor": int(interpolation_factor),
        "camera_space_smoothing_applied": False if uses_dynamic_camera else None,
    }
    if c2w_blender_npy is not None:
        metadata["c2w_blender_path"] = str(c2w_blender_npy)
    if intrinsics_npy is not None:
        metadata["source_intrinsics_path"] = str(intrinsics_npy)
    if dynamic_intrinsics_path is not None:
        metadata["dynamic_intrinsics_for_foundationpose_path"] = str(dynamic_intrinsics_path)
    if init_pose_cam_path is not None:
        metadata["dynamic_init_pose_cam_path"] = str(init_pose_cam_path)
    if object_init_pose_npy is not None:
        metadata["object_init_pose_source"] = str(object_init_pose_npy)
    if crop_bbox is not None:
        metadata["crop_bbox_xyxy"] = [int(v) for v in crop_bbox]
    with open(os.path.join(pe_dir, "poses_in_cam.metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def _subsample_files(directory, pattern, target_count):
    """Subsample files matching *pattern* in *directory* to *target_count*."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        return 0

    indices = _get_sample_indices(len(files), target_count)
    ext = os.path.splitext(files[0])[1]

    tmp = os.path.join(directory, "_subsample_tmp")
    os.makedirs(tmp, exist_ok=True)
    for new_i, src_i in enumerate(indices):
        shutil.copy2(files[src_i], os.path.join(tmp, f"{new_i:06d}{ext}"))

    for f in files:
        os.remove(f)
    for f in glob.glob(os.path.join(tmp, f"*{ext}")):
        shutil.move(f, directory)
    shutil.rmtree(tmp)
    return len(indices)


def subsample_pose_output(input_dir, target_frame_count):
    """Subsample pose estimation output to *target_frame_count* frames."""
    pe_dir = os.path.join(input_dir, "pose_estimation_output")

    # Subsample ob_in_cam txt files
    ob_dir = os.path.join(pe_dir, "debug", "ob_in_cam")
    if os.path.isdir(ob_dir):
        _subsample_files(ob_dir, "*.txt", target_frame_count)

    # Subsample track_vis images and regenerate video
    vis_dir = os.path.join(pe_dir, "debug", "track_vis")
    if os.path.isdir(vis_dir):
        n = _subsample_files(vis_dir, "*.png", target_frame_count)
        if n > 0:
            video_out = os.path.join(pe_dir, "pose_estimation_tracking.mp4")
            if os.path.exists(video_out):
                os.remove(video_out)
            from grail.core.video import compile_images_to_video

            compile_images_to_video(vis_dir, video_out, fps=24, image_pattern="*.png")

    # Subsample poses_in_cam.pkl
    pkl_path = os.path.join(pe_dir, "poses_in_cam.pkl")
    if os.path.exists(pkl_path):
        import pickle

        with open(pkl_path, "rb") as f:
            poses = pickle.load(f)
        indices = _get_sample_indices(len(poses), target_frame_count)
        with open(pkl_path, "wb") as f:
            pickle.dump([poses[i] for i in indices], f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Crop utilities
# ---------------------------------------------------------------------------


def determine_crop_bbox_centered(masks, pixel_boundary=50):
    """Compute minimal crop bbox covering all object positions across frames."""
    if not masks:
        raise ValueError("masks list cannot be empty")

    H, W = masks[0].shape[:2]
    x_min, x_max, y_min, y_max = W, 0, H, 0

    for mask in masks:
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            x_min = min(x_min, int(np.min(xs)))
            x_max = max(x_max, int(np.max(xs)))
            y_min = min(y_min, int(np.min(ys)))
            y_max = max(y_max, int(np.max(ys)))

    if x_min >= x_max or y_min >= y_max:
        return (0, 0, W, H)

    return (
        max(0, x_min - pixel_boundary),
        max(0, y_min - pixel_boundary),
        min(W, x_max + pixel_boundary),
        min(H, y_max + pixel_boundary),
    )


def crop_image(image, crop_bbox):
    """Crop image to *crop_bbox* = (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = crop_bbox
    return image[y0:y1, x0:x1] if image.ndim == 2 else image[y0:y1, x0:x1, :]


def crop_frames_in_directory(frames_dir, crop_bbox):
    """Crop all image frames in a directory in-place."""
    import cv2

    files = sorted(
        glob.glob(os.path.join(frames_dir, "*.png")) + glob.glob(os.path.join(frames_dir, "*.jpg"))
    )
    for p in files:
        cv2.imwrite(p, crop_image(cv2.imread(p, -1), crop_bbox))


def _resample_masks_to_count(masks, target_count):
    """Nearest-neighbor resample mask sequence to match extracted RGB frames."""
    if target_count <= 0:
        return []
    if len(masks) == target_count:
        return list(masks)
    if len(masks) == 1:
        return [masks[0] for _ in range(target_count)]
    indices = [round(i * (len(masks) - 1) / (target_count - 1)) for i in range(target_count)]
    return [masks[i] for i in indices]


def write_foundationpose_object_masks(masks, input_dir, crop_bbox=None, frame_count=None):
    """Write SAM2 object masks using FoundationPose's expected file layout."""
    import cv2

    masks_dir = os.path.join(input_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    for p in glob.glob(os.path.join(masks_dir, "*.png")) + glob.glob(
        os.path.join(masks_dir, "*.jpg")
    ):
        os.remove(p)

    masks_to_write = _resample_masks_to_count(masks, int(frame_count)) if frame_count else list(masks)
    for i, mask in enumerate(masks_to_write):
        mask_u8 = (np.asarray(mask).squeeze() > 0).astype(np.uint8) * 255
        if crop_bbox is not None:
            mask_u8 = crop_image(mask_u8, crop_bbox)
        cv2.imwrite(os.path.join(masks_dir, f"{i:06d}.png"), mask_u8)
    return masks_dir


def adjust_camera_intrinsics_for_crop(input_dir, crop_bbox):
    """Adjust cam_K.txt principal point for cropping offset."""
    x_start, y_start, _, _ = crop_bbox
    cam_K_path = os.path.join(input_dir, "cam_K.txt")
    if not os.path.exists(cam_K_path):
        return
    K = np.loadtxt(cam_K_path).reshape(3, 3)
    K[0, 2] -= x_start
    K[1, 2] -= y_start
    with open(cam_K_path, "w") as f:
        for row in K:
            f.write(f"{row[0]:.18e} {row[1]:.18e} {row[2]:.18e}\n")


def adjust_intrinsics_array_for_crop(K_seq, crop_bbox):
    """Return a copy of [T,3,3] intrinsics adjusted for an image crop."""
    x_start, y_start, _, _ = crop_bbox
    K_seq = np.asarray(K_seq, dtype=np.float32).copy()
    K_seq[:, 0, 2] -= float(x_start)
    K_seq[:, 1, 2] -= float(y_start)
    return K_seq


def object_world_pose_from_first_frame(input_dir, object_init_pose_npy=None):
    if object_init_pose_npy is not None:
        if not os.path.exists(object_init_pose_npy):
            raise FileNotFoundError(f"object_init_pose.npy not found: {object_init_pose_npy}")
        obj_pose = np.load(object_init_pose_npy).astype(np.float32).reshape(4, 4)
        return obj_pose

    first_frame_file = os.path.join(input_dir, "first_frame_pose.pickle")
    obj_R, obj_t, obj_scale, cam_R, cam_t, render_config = load_init_rendering_data(
        first_frame_file
    )
    obj_pose = np.eye(4, dtype=np.float32)
    obj_pose[:3, :3] = obj_R
    obj_pose[:3, 3] = obj_t.reshape(-1)
    return obj_pose


def dynamic_object_pose_in_camera(input_dir, c2w_blender_npy, frame_count, object_init_pose_npy=None):
    """Build T_{C_t<-O} from fixed Blender-world object pose and dynamic cameras."""
    c2w = np.load(c2w_blender_npy).astype(np.float32)
    if c2w.shape[0] < frame_count:
        raise ValueError(f"Dynamic camera has {c2w.shape[0]} frames, expected {frame_count}")
    obj_pose = object_world_pose_from_first_frame(input_dir, object_init_pose_npy=object_init_pose_npy)
    return [np.linalg.inv(c2w[i]) @ obj_pose for i in range(frame_count)]


def dynamic_object_pose_to_world(poses_in_cam, c2w_blender_npy):
    """Convert FoundationPose T_{C_t<-O,t} to T_{B<-O,t} with dynamic cameras."""
    c2w = np.load(c2w_blender_npy).astype(np.float32)
    poses = [np.asarray(p, dtype=np.float32).reshape(4, 4) for p in poses_in_cam]
    if c2w.shape[0] < len(poses):
        raise ValueError(f"Dynamic camera has {c2w.shape[0]} frames, expected {len(poses)}")
    return [(c2w[i] @ poses[i]).astype(np.float32) for i in range(len(poses))]


def load_dynamic_object_world_poses(input_dir):
    """Load cached T_{B<-O,t} dynamic object poses from a FoundationPose output dir."""
    pe_dir = os.path.join(input_dir, "pose_estimation_output")
    poses_file = os.path.join(pe_dir, "poses_in_world.pkl")
    if not os.path.exists(poses_file):
        raise FileNotFoundError(f"Dynamic object world pose cache not found: {poses_file}")
    with open(poses_file, "rb") as f:
        return pickle.load(f)


def save_dynamic_object_world_poses(input_dir, c2w_blender_npy, poses_file=None):
    """Save dynamic-camera object trajectory in Blender/GRAIL metric world."""
    pe_dir = os.path.join(input_dir, "pose_estimation_output")
    poses_file = poses_file or os.path.join(pe_dir, "poses_in_cam.pkl")
    with open(poses_file, "rb") as f:
        poses_in_cam = pickle.load(f)
    poses_world = dynamic_object_pose_to_world(poses_in_cam, c2w_blender_npy)
    world_pkl = os.path.join(pe_dir, "poses_in_world.pkl")
    world_npy = os.path.join(pe_dir, "poses_in_world.npy")
    save_object_pose_data(poses_world, world_pkl)
    np.save(world_npy, np.asarray(poses_world, dtype=np.float32))
    metadata = {
        "source_pose_file": str(poses_file),
        "c2w_blender_path": str(c2w_blender_npy),
        "pose_count": int(len(poses_world)),
        "input_coordinate_space": "opencv_camera_T_C<-O",
        "output_coordinate_space": "blender_metric_world_T_B<-O",
        "transform": "T_B<-O,t = T_B<-C_t @ T_C_t<-O,t",
        "camera_space_smoothing_applied": False,
    }
    with open(os.path.join(pe_dir, "poses_in_world.metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return world_pkl


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_obj_pose_est(
    video_path,
    mesh_file,
    input_dir,
    video_masks,
    debug=2,
    device="cuda",
    crop_image=False,
    interpolation_factor=1,
    is_static=False,
    intrinsics_npy=None,
    c2w_blender_npy=None,
    object_init_pose_npy=None,
):
    """Run object pose estimation with optional cropping and frame interpolation.

    Args:
        video_path: Path to the video file.
        mesh_file: Path to the object mesh (.obj).
        input_dir: Directory for outputs.
        video_masks: Dict mapping frame_idx → {obj_id → mask}.
        debug: FoundationPose debug level.
        device: Compute device.
        crop_image: Crop to minimal bbox covering all object positions.
        interpolation_factor: Frame interpolation factor (1 = none).
        is_static: If False, detect if the object is static; if True, skip FoundationPose and generate static poses directly.
        intrinsics_npy: Optional [T,3,3] per-frame K for dynamic-camera tracking.
        c2w_blender_npy: Optional [T,4,4] T_{B<-C_t}; enables dynamic-camera initialization.
        object_init_pose_npy: Optional [4,4] T_{B<-O,0} from scene_reference/object_init_pose.npy.
    """
    _, original_frame_count = get_video_fps_and_frame_count(video_path)
    requested_interpolation_factor = int(interpolation_factor)
    interpolation_factor = _effective_interpolation_factor(
        interpolation_factor,
        intrinsics_npy=intrinsics_npy,
        c2w_blender_npy=c2w_blender_npy,
    )
    if requested_interpolation_factor != interpolation_factor:
        print(
            "Dynamic-camera FoundationPose: disabling frame interpolation because "
            "per-frame VGGT intrinsics/cameras are defined only for original frames."
        )

    first_obj_mask = (video_masks[0][0].squeeze() > 0).astype(np.uint8)
    first_obj_float = first_obj_mask.astype(np.float32)

    # Save mask videos for debugging
    frame_count = len(video_masks)
    obj_masks = [video_masks[i][0].squeeze() for i in range(frame_count)]
    human_masks = [video_masks[i][1].squeeze() for i in range(frame_count)]

    debug_dir = os.path.join(input_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    save_images_to_video(obj_masks, os.path.join(debug_dir, "obj_masks.mp4"))
    save_images_to_video(human_masks, os.path.join(debug_dir, "human_masks.mp4"))

    if not is_static:
        # Check if object is static (not moving across frames)
        is_static = True
        threshold_frac = 0.1
        for i in range(frame_count):
            obj_f = (video_masks[i][0].squeeze() > 0).astype(np.float32)
            human_f = (video_masks[i][1].squeeze() > 0).astype(np.float32)
            obj_total = np.sum(first_obj_float)
            if obj_total == 0:
                break
            threshold = threshold_frac * obj_total

            # New object pixels outside first-frame mask → object moved
            if np.sum(obj_f * (1 - first_obj_float)) > threshold:
                is_static = False
                break

            # Object pixels vanished without human occlusion → object moved
            if np.sum(first_obj_float * (1 - obj_f) * (1 - human_f)) > threshold:
                is_static = False
                break

    if is_static:
        print("Object is static — skipping FoundationPose, generating static poses directly")
        if c2w_blender_npy is not None:
            pose_list = dynamic_object_pose_in_camera(
                input_dir,
                c2w_blender_npy,
                original_frame_count,
                object_init_pose_npy=object_init_pose_npy,
            )
        else:
            first_frame_file = os.path.join(input_dir, "first_frame_pose.pickle")
            obj_R, obj_t, obj_scale, cam_R, cam_t, render_config = load_init_rendering_data(
                first_frame_file
            )

            world_to_camera_blender = world_to_camera_matrix(
                torch.from_numpy(cam_R).float(), torch.from_numpy(cam_t).float()
            ).numpy()
            world_to_camera_opencv = convert_blender_to_internal(
                w2c=world_to_camera_blender
            ).w2c

            object_matrix = np.eye(4)
            object_matrix[:3, :3] = obj_R
            object_matrix[:3, 3] = obj_t.reshape(-1)
            ob_in_cam = world_to_camera_opencv @ object_matrix
            pose_list = [ob_in_cam.copy() for _ in range(original_frame_count)]
        poses_output_file = os.path.join(input_dir, "pose_estimation_output", "poses_in_cam.pkl")
        save_object_pose_data(pose_list, poses_output_file)
        _write_poses_in_cam_metadata(
            input_dir,
            pose_count=len(pose_list),
            camera_mode="dynamic" if c2w_blender_npy is not None else "fixed",
            uses_dynamic_camera=c2w_blender_npy is not None,
            uses_dynamic_intrinsics=intrinsics_npy is not None,
            c2w_blender_npy=c2w_blender_npy,
            intrinsics_npy=intrinsics_npy,
            object_init_pose_npy=object_init_pose_npy,
            is_static=True,
            requested_interpolation_factor=requested_interpolation_factor,
            interpolation_factor=interpolation_factor,
        )
        if c2w_blender_npy is not None:
            world_pose_file = save_dynamic_object_world_poses(
                input_dir,
                c2w_blender_npy,
                poses_file=poses_output_file,
            )
            print(f"Saved {len(pose_list)} static world poses to: {world_pose_file}")
        print(f"Saved {len(pose_list)} static poses to: {poses_output_file}")
        return True

    # Frame interpolation
    actual_video = video_path
    interp_path = None
    if interpolation_factor > 1:
        interp_path = os.path.join(input_dir, f"interpolated_{interpolation_factor}x.mp4")
        if interpolate_video(video_path, interp_path, interpolation_factor):
            actual_video = interp_path
        else:
            interpolation_factor = 1

    # Crop
    crop_bbox = None
    dynamic_intrinsics_path = None
    if intrinsics_npy is not None:
        K_seq = np.load(intrinsics_npy).astype(np.float32)
    if crop_image:
        crop_bbox = determine_crop_bbox_centered(obj_masks)
        cw = crop_bbox[2] - crop_bbox[0]
        ch = crop_bbox[3] - crop_bbox[1]
        print(f"  Crop {cw}x{ch} bbox={crop_bbox}")
        adjust_camera_intrinsics_for_crop(input_dir, crop_bbox)
        if intrinsics_npy is not None:
            K_seq = adjust_intrinsics_array_for_crop(K_seq, crop_bbox)

    if intrinsics_npy is not None:
        dynamic_intrinsics_path = os.path.join(input_dir, "dynamic_intrinsics_for_foundationpose.npy")
        np.save(dynamic_intrinsics_path, K_seq)

    init_pose_cam_path = None
    if c2w_blender_npy is not None:
        c2w = np.load(c2w_blender_npy).astype(np.float32)
        init_pose_cam = np.linalg.inv(c2w[0]) @ object_world_pose_from_first_frame(
            input_dir,
            object_init_pose_npy=object_init_pose_npy,
        )
        init_pose_cam_path = os.path.join(input_dir, "dynamic_init_pose_cam.npy")
        np.save(init_pose_cam_path, init_pose_cam.astype(np.float32))
        with open(os.path.join(input_dir, "dynamic_init_pose_cam.metadata.json"), "w") as f:
            json.dump(
                {
                    "object_init_pose_source": str(object_init_pose_npy)
                    if object_init_pose_npy is not None
                    else os.path.join(input_dir, "first_frame_pose.pickle"),
                    "c2w_blender_path": str(c2w_blender_npy),
                    "transform": "T_C0<-O = inv(T_B<-C0) @ T_B<-O0",
                    "output_coordinate_space": "opencv_camera_T_C0<-O",
                },
                f,
                indent=2,
            )

    # Extract frames
    rgb_dir = os.path.join(input_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    extract_frames_from_video(actual_video, rgb_dir, image_format="png")

    if crop_bbox is not None:
        crop_frames_in_directory(rgb_dir, crop_bbox)

    rgb_frame_count = len(glob.glob(os.path.join(rgb_dir, "*.png")))
    write_foundationpose_object_masks(
        obj_masks,
        input_dir,
        crop_bbox=crop_bbox,
        frame_count=rgb_frame_count,
    )

    # Run FoundationPose
    adapter_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "adapters", "foundation_pose.py"
    )
    cmd = [
        sys.executable,
        adapter_script,
        "--mesh_file",
        mesh_file,
        "--test_scene_dir",
        input_dir,
        "--debug",
        str(debug),
    ]
    if is_static:
        cmd.append("--is_static")
    if dynamic_intrinsics_path is not None:
        cmd.extend(["--intrinsics_npy", dynamic_intrinsics_path])
    if init_pose_cam_path is not None:
        cmd.extend(["--init_pose_cam_npy", init_pose_cam_path])

    success = run_subprocess(cmd, "FoundationPose tracking")

    # Subsample if interpolated
    if interpolation_factor > 1 and success:
        subsample_pose_output(input_dir, target_frame_count=original_frame_count)

    if success and c2w_blender_npy is not None:
        world_pose_file = save_dynamic_object_world_poses(input_dir, c2w_blender_npy)
        print(f"Saved dynamic object world poses to: {world_pose_file}")
    if success:
        poses_output_file = os.path.join(input_dir, "pose_estimation_output", "poses_in_cam.pkl")
        pose_count = 0
        if os.path.exists(poses_output_file):
            try:
                with open(poses_output_file, "rb") as f:
                    pose_count = len(pickle.load(f))
            except Exception:
                pose_count = original_frame_count
        _write_poses_in_cam_metadata(
            input_dir,
            pose_count=pose_count,
            camera_mode="dynamic" if c2w_blender_npy is not None else "fixed",
            uses_dynamic_camera=c2w_blender_npy is not None,
            uses_dynamic_intrinsics=dynamic_intrinsics_path is not None,
            c2w_blender_npy=c2w_blender_npy,
            intrinsics_npy=intrinsics_npy,
            dynamic_intrinsics_path=dynamic_intrinsics_path,
            init_pose_cam_path=init_pose_cam_path,
            object_init_pose_npy=object_init_pose_npy,
            is_static=False,
            requested_interpolation_factor=requested_interpolation_factor,
            interpolation_factor=interpolation_factor,
            crop_bbox=crop_bbox,
        )

    # Cleanup
    if interp_path and os.path.exists(interp_path):
        os.remove(interp_path)

    return success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Object Pose Estimation (FoundationPose)")
    parser.add_argument("--mesh_file", type=str, required=True)
    parser.add_argument("--test_scene_dir", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--debug", type=int, default=2)
    args = parser.parse_args()

    for p in (args.mesh_file, args.test_scene_dir, args.video):
        if not os.path.exists(p):
            print(f"Error: not found: {p}")
            sys.exit(1)

    # Simplified CLI — uses adapter directly
    adapter_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "adapters", "foundation_pose.py"
    )
    cmd = [
        sys.executable,
        adapter_script,
        "--mesh_file",
        args.mesh_file,
        "--test_scene_dir",
        args.test_scene_dir,
        "--debug",
        str(args.debug),
    ]
    success = run_subprocess(cmd, "FoundationPose tracking")
    sys.exit(0 if success else 1)
