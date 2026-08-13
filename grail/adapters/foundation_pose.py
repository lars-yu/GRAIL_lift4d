#!/usr/bin/env python3

import argparse
import logging
import os
import pickle
import sys

import cv2
import imageio
import numpy as np
import torch
import trimesh

# Add FoundationPose to path
foundation_pose_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "imports", "FoundationPose"
)
sys.path.insert(0, foundation_pose_dir)

import nvdiffrast.torch as dr
from datareader import *
from estimater import *

# Add GRAIL modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from grail.core.io import load_init_rendering_data, save_object_pose_data
from grail.core.video import compile_images_to_video
from grail.dynamic_camera.conventions import convert_blender_to_internal
from grail.pose_est.utils import smooth_pose_matrices
from grail.rendering.camera import world_to_camera_matrix


def blender_to_opencv_convention(world_to_camera_blender):
    """Convert camera matrix from Blender to OpenCV convention"""
    return convert_blender_to_internal(w2c=world_to_camera_blender).w2c


def transform_object_to_camera_frame(world_to_camera_matrix, object_position, object_rotation):
    """Transform object pose from world frame to camera frame"""
    object_matrix = np.eye(4)
    object_matrix[:3, :3] = object_rotation
    object_matrix[:3, 3] = object_position.reshape(-1)

    camspace_object_matrix = world_to_camera_matrix @ object_matrix
    return camspace_object_matrix[:3, :3], camspace_object_matrix[:3, 3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_file", type=str, required=True)
    parser.add_argument("--test_scene_dir", type=str, required=True)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--is_static", action="store_true", default=False)
    parser.add_argument("--debug", type=int, default=2)
    parser.add_argument("--debug_dir", type=str, default=None)
    parser.add_argument(
        "--intrinsics_npy",
        type=str,
        default=None,
        help="Optional [T,3,3] intrinsics. When set, track_one receives K_t per frame.",
    )
    parser.add_argument(
        "--init_pose_cam_npy",
        type=str,
        default=None,
        help="Optional 4x4 T_{C0<-O} initialization for dynamic-camera runs.",
    )
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)

    # Handle Scene objects (files with multiple meshes)
    if isinstance(mesh, trimesh.Scene):
        # Combine all meshes in the scene into a single mesh
        mesh = mesh.dump(concatenate=True)

    # Load first frame pose information when available. Dynamic-camera runs can
    # be initialized purely from scene_reference/object_init_pose.npy plus
    # c2w_blender.npy, so first_frame_pose.pickle must not be a hard dependency
    # in that path.
    first_frame_file = f"{args.test_scene_dir}/first_frame_pose.pickle"
    first_frame_data = None
    obj_scale = np.ones(3, dtype=np.float32)
    if os.path.exists(first_frame_file):
        first_frame_data = load_init_rendering_data(first_frame_file)
        obj_R, obj_t, obj_scale, cam_R, cam_t, render_config = first_frame_data
    elif args.init_pose_cam_npy is None:
        raise FileNotFoundError(
            "first_frame_pose.pickle is required for the legacy fixed-camera "
            f"FoundationPose path: {first_frame_file}"
        )
    else:
        logging.warning(
            "first_frame_pose.pickle not found; using dynamic init pose and assuming "
            "the provided mesh is already in exported metric scale."
        )

    if args.init_pose_cam_npy is not None:
        ob_in_cam = np.load(args.init_pose_cam_npy).reshape(4, 4)
    else:
        # Convert to camera space using the fixed first-frame Blender camera.
        if first_frame_data is None:
            raise RuntimeError("Legacy FoundationPose initialization missing first-frame data")
        world_to_camera_blender = world_to_camera_matrix(
            torch.from_numpy(cam_R).float(), torch.from_numpy(cam_t).float()
        ).numpy()
        world_to_camera_opencv = blender_to_opencv_convention(world_to_camera_blender)
        obj_R_camspace, obj_t_camspace = transform_object_to_camera_frame(
            world_to_camera_opencv, obj_t, obj_R
        )

        ob_in_cam = np.eye(4)
        ob_in_cam[:3, :3] = obj_R_camspace
        ob_in_cam[:3, 3] = obj_t_camspace
    mesh.apply_scale(obj_scale)

    debug = args.debug
    debug_dir = (
        args.debug_dir if args.debug_dir else f"{args.test_scene_dir}/pose_estimation_output/debug"
    )
    os.system(f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx,
    )
    logging.info("estimator initialization done")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
    intrinsics_seq = None
    if args.intrinsics_npy is not None:
        intrinsics_seq = np.load(args.intrinsics_npy).astype(np.float32)
        if intrinsics_seq.ndim != 3 or intrinsics_seq.shape[-2:] != (3, 3):
            raise ValueError(f"--intrinsics_npy must be [T,3,3], got {intrinsics_seq.shape}")
        if len(intrinsics_seq) != len(reader.color_files):
            raise ValueError(
                "--intrinsics_npy must provide exactly one K_t per tracked RGB frame: "
                f"intrinsics={len(intrinsics_seq)}, frames={len(reader.color_files)}"
            )

    pose_list = []
    for i in range(len(reader.color_files)):
        logging.info(f"i:{i}")
        color = reader.get_color(i)
        H, W = color.shape[:2]
        depth = np.zeros((H, W), dtype=float)
        # FoundationPose's mesh vertices are float64 for this mesh; keep the
        # per-frame K dtype aligned at the adapter boundary so crop projection
        # does not mix float32 intrinsics with float64 mesh tensors.
        K_i = np.asarray(
            reader.K if intrinsics_seq is None else intrinsics_seq[i], dtype=np.float64
        )
        if args.is_static:
            pose = ob_in_cam.copy()
        else:
            if i == 0:
                pose = ob_in_cam.copy()
                ob_in_cam_adjusted = ob_in_cam.copy()
                ob_in_cam_adjusted[:3, 3] += ob_in_cam_adjusted[:3, :3] @ est.model_center
                est.pose_last = torch.as_tensor(
                    ob_in_cam_adjusted, dtype=torch.float, device="cuda"
                )

                if debug >= 3:
                    m = mesh.copy()
                    m.apply_transform(pose)
                    m.export(f"{debug_dir}/model_tf.obj")
            else:
                pose = est.track_one(
                    rgb=color, depth=depth, K=K_i, iteration=args.track_refine_iter
                )

        pose_list.append(pose)
        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))

        # if debug >= 1:
        #     center_pose = pose @ np.linalg.inv(to_origin)
        #     vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        #     vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        #     # cv2.imshow('1', vis[..., ::-1])
        #     # cv2.waitKey(1)

        # if debug >= 2:
        #     os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
        #     imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

    # Apply Savitzky-Golay smoothing only in the original fixed-camera path.
    # In dynamic-camera runs T_{C_t<-O} is expected to change with camera motion;
    # smoothing must happen after conversion to Blender/world space.
    if intrinsics_seq is None and args.init_pose_cam_npy is None:
        logging.info("Applying temporal smoothing to fixed-camera object poses...")
        pose_list = smooth_pose_matrices(pose_list, window_length=9, polyorder=3)
    else:
        logging.info("Skipping camera-space smoothing for dynamic-camera object poses")
    for i in range(len(pose_list)):
        color = reader.get_color(i)
        K_i = np.asarray(
            reader.K if intrinsics_seq is None else intrinsics_seq[i], dtype=np.float64
        )
        if debug >= 1:
            center_pose = pose_list[i] @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(K_i, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(
                color,
                ob_in_cam=center_pose,
                scale=0.1,
                K=K_i,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            # cv2.imshow('1', vis[..., ::-1])
            # cv2.waitKey(1)

        if debug >= 2:
            os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
            imageio.imwrite(f"{debug_dir}/track_vis/{reader.id_strs[i]}.png", vis)

    # Save pose results
    poses_output_file = f"{args.test_scene_dir}/pose_estimation_output/poses_in_cam.pkl"
    os.makedirs(os.path.dirname(poses_output_file), exist_ok=True)
    save_object_pose_data(pose_list, poses_output_file)
    logging.info(f"Saved {len(pose_list)} poses to: {poses_output_file}")
    if debug >= 2:
        vis_dir = os.path.join(debug_dir, "track_vis")
        video_output_path = os.path.join(
            args.test_scene_dir, "pose_estimation_output", "pose_estimation_tracking.mp4"
        )
        compile_images_to_video(
            image_dir=vis_dir, output_video_path=video_output_path, fps=24, image_pattern="*.png"
        )
        logging.info(f"Created visualization video: {video_output_path}")
