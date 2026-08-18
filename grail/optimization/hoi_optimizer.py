import datetime
import json
import os
from glob import glob

import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)

from grail.adapters.lift4d_depth import load_lift4d_depth_prior
from grail.constants.image import FOCAL_LENGTH, HEIGHT, WIDTH
from grail.core.contact_label import detect_contact_joints_interval
from grail.core.io import (
    load_character_data,
    load_human_motion_data,
    load_init_rendering_data,
    load_mesh,
    load_object_pose_data,
)
from grail.core.logging import create_logger
from grail.core.torch_utils import tensor_to_numpy
from grail.core.video import (
    extract_frames_from_video,
    get_video_fps_and_frame_count,
)
from grail.models.human_model import create_human_model
from grail.optimization.data_types import HOIData, HOIPrediction, OptParams
from grail.optimization.evaluator import pre_eval, truncate_data
from grail.optimization.interaction import (
    get_contact_labels_for_frame,
    identify_interaction_start_end,
    identify_interaction_start_end_with_mask,
)
from grail.optimization.loss_computer import LossComputer
from grail.optimization.motion_state import (
    detect_object_motion,
    infer_contact_hand,
    resolve_contact_hint,
)
from grail.optimization.hand_object_ray_ik import (
    approach_window_from_fps,
    camera_ray_hand_targets,
    mesh_surface_depth_at_pixels,
    select_contact_hand_from_masks,
)
from grail.optimization.approach import (
    approach_offsets,
    ground_approach_direction,
    smoothstep_approach_ramp,
)
from grail.pose_est.utils import smooth_axis_angle_sequence, smooth_pose_sequence
from grail.preprocessing.preprocess import load_depth_from_cache, load_masks_from_cache
from grail.rendering.camera import (
    cam_pose_blender_to_opencv,
    cam_pose_opencv_to_pytorch3d,
    get_camera,
    project_world_to_screen,
    transform_camera_to_world,
    transform_world_to_camera,
    transform_pose_c2w,
    world_to_camera_matrix,
)
from grail.optimization.loss_terms import foundationpose_camera_rays, positive_depth_scale, ray_depth_translation
from grail.visualization.scenepic import ScenepicVisualizer


class HOIOptimizer:
    """
    Human-Object Interaction Optimizer
    Optimizes SMPL human motion and object trajectories
    """

    def __init__(self, exp_name, cfg, cache_dir, output_dir, device="cuda"):
        """
        Initialize the HOI optimizer

        Args:
            exp_name (str): Experiment name (dataset/category/video_id)
            cfg (dict): Optimizer configuration dictionary (optimization +
                        human_model sections merged from the unified YAML)
            cache_dir (str): Directory for cached data
            output_dir (str): Directory for optimization outputs and logs
            device (str): Device for computations
        """
        self.device = device
        self.cfg = cfg

        self.log_dir = os.path.join(output_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.logger = create_logger(self.log_dir)
        self.cache_list = self.cfg.get("cache_list", [])
        self.cache_dir = cache_dir

        # TODO: double check this logic when changing the exp_name format
        dataset, category, video_id = exp_name.split("/")
        self.exp_name = exp_name
        self.obj_name = category
        self.character_name = "_".join(video_id.split("_")[:3])
        print(f"Experiment name: {self.exp_name}")
        print(f"Object name: {self.obj_name}")

        self.opt_stage_specs = self.cfg["opt_stage_specs"]

        self.vis_cfg = self.cfg["vis_cfg"]
        self.enable_vis = self.vis_cfg.get("enable", True)

        self.eval_cfg = self.cfg["eval"]

        # Pre-evaluation configuration
        self.pre_eval_cfg = self.cfg.get("pre_eval", {})
        self.min_frames_threshold = self.pre_eval_cfg.get("min_frames", None)

        # Body model — polymorphic: SOMA, SMPL-X, or G1-proportioned SMPL-X
        self.human_model = create_human_model(self.cfg["human_model"], device)
        self.logger.info(f"Using {self.human_model.model_type} body model")

    def init_data(self, video_file, hmr_file, obj_path, obj_pose_file, render_config_file):
        """Load all data and assemble HOIData for optimization."""
        video_id = self.exp_name

        # 1. Camera and rendering setup
        camera, cameras, opencv_cam_R, opencv_cam_t, obj_scale, static_objects = (
            self._load_camera_config(render_config_file)
        )

        # 2. Object mesh and poses
        obj_verts, obj_faces, obj_poses_incam, obj_poses, fp_ray_cam = self._load_object(
            obj_path, obj_scale, obj_pose_file, opencv_cam_R, opencv_cam_t
        )

        # 3. Video frames and masks
        images_path, video_fps, video_frame_count = self._load_video_frames(video_file)
        human_masks, obj_masks = self._load_masks(video_id)

        # 4. Interaction detection
        inter_start_idx, inter_end_idx, is_static_obj = self._detect_interaction(
            obj_poses_incam, images_path, human_masks, obj_masks
        )

        # 5. Human motion data
        motion_data, motion_data_global_init, foot_contact_probs = self._load_motion(
            hmr_file, render_config_file, opencv_cam_R, opencv_cam_t, inter_start_idx
        )

        # Validate frame counts
        frame_num = motion_data["poses"].shape[0]
        if len(obj_poses) != frame_num:
            raise ValueError(f"Frame count mismatch - SMPL: {frame_num}, Object: {len(obj_poses)}")
        if video_frame_count != frame_num:
            raise ValueError(
                f"Frame count mismatch - Video: {video_frame_count}, SMPL: {frame_num}"
            )

        legacy_motion_cfg = self.cfg.get("lift4d_motion", {}) or {}
        if legacy_motion_cfg.get("enabled", False):
            raise ValueError(
                "Legacy lift4d_motion supervision is prohibited because it consumes "
                "object_poses_cam/Kabsch rotation. Use use_lift4d_depth_prior instead."
            )
        lift4d_motion = None
        lift4d_depth = self._load_lift4d_depth_prior(video_id=video_id, frame_num=frame_num)
        if lift4d_depth is not None:
            self._validate_ray_projection(obj_poses_incam, fp_ray_cam, lift4d_depth)

        # 6. Contact labels are legacy-only. Formal mask-motion/ray-IK must not
        # read cache/GPT labels even to choose a fallback interval.
        formal_motion_enabled = bool(
            (self.cfg.get("object_motion_state", {}) or {}).get("enabled", False)
        )
        skip_contact_labels = bool(self.cfg.get("skip_contact_label_loading", False))
        if formal_motion_enabled or skip_contact_labels:
            contact_labels, contact_interval, contact_start_idx = [], 1, None
        else:
            contact_labels, contact_interval, contact_start_idx = self._detect_contact_labels(
                video_id, frame_num, inter_start_idx, is_static_obj, images_path
            )

        # 7. Depth maps
        depth_maps = self._load_depth(video_id)

        # 8. Prepare GT keypoints
        human_faces = self.human_model.generate_mesh(
            motion_data, output_joints=False, require_grad=False
        )[1]
        gt_body_kp, gt_hand_kp = self.human_model.extract_gt_keypoints(motion_data)

        # 9. Transform object vertices to per-frame positions
        obj_verts_seq = torch.bmm(
            obj_verts.unsqueeze(0).repeat(frame_num, 1, 1),
            obj_poses[:, :3, :3].transpose(1, 2),
        ) + obj_poses[:, :3, 3].reshape(frame_num, 1, 3)

        gt_obj_verts_tracking = project_world_to_screen(
            obj_verts_seq.reshape(-1, 3), cameras
        ).reshape(frame_num, -1, 3)[:, :, :2]

        # ── Set self.* state (all in one place) ──────────────────────────────
        self.cameras = cameras
        self.opencv_cam_R = opencv_cam_R
        self.opencv_cam_t = opencv_cam_t
        self.obj_path = obj_path
        self.obj_mesh = Meshes(verts=[obj_verts], faces=[obj_faces])
        self.video_fps = video_fps
        self.image_list = images_path
        self.depth_list = depth_maps
        self.contact_labels = contact_labels
        self.contact_interval = contact_interval
        self.contact_start_idx = contact_start_idx

        contact_cfg = self.cfg.get("contact", {}) or {}
        motion_cfg = self.cfg.get("object_motion_state", {}) or {}
        motion_state = None
        hand_selection = None
        hand_ray_target_world = None
        hand_ray_ramp = None
        hand_initial_cam = None
        hand_pixels = None
        hand_ray_surface_fallback = None
        hand_initial_cam_depth = None
        hand_target_cam_depth = None
        object_surface_depth = None
        hand_approach_initial_distance = None
        if motion_cfg.get("enabled", False):
            if lift4d_depth is None:
                raise ValueError("object_motion_state.enabled requires a real Lift4D depth prior")
            motion_state = detect_object_motion(
                lift4d_depth.center_cam_raw.detach().cpu().numpy(),
                np.asarray([obj_masks[i] for i in range(frame_num)]),
                smoothed_z=lift4d_depth.z.detach().cpu().numpy(),
                config=motion_cfg,
            )
            lift4d_depth.center_cam_detection = torch.as_tensor(
                motion_state.detection_center_cam,
                dtype=lift4d_depth.center_cam_raw.dtype,
                device=self.device,
            )
            lift4d_depth.z_target = torch.as_tensor(
                motion_state.z_target,
                dtype=lift4d_depth.z.dtype,
                device=self.device,
            )
            move_start = int(motion_state.move_start_frame)
            initial_hand_joints = self.human_model.get_hand_joints(
                motion_data, require_grad=False
            )
            projected_hand = project_world_to_screen(
                initial_hand_joints.reshape(-1, 3), cameras
            ).reshape(frame_num, -1, 3)[:, :, :2]
            hand_selection = select_contact_hand_from_masks(
                gt_hand_kp.detach().cpu().numpy(),
                np.asarray([obj_masks[i] for i in range(frame_num)]),
                move_start,
                projected_hand_points_2d=projected_hand.detach().cpu().numpy(),
                lookback_frames=int(contact_cfg.get("hand_selection_lookback", 5)),
                confidence_threshold=float(contact_cfg.get("keypoint_confidence", 0.2)),
                both_distance_px=float(contact_cfg.get("both_distance_px", 12.0)),
                both_ratio=float(contact_cfg.get("both_ratio", 1.25)),
            )
            configured_hand = str(contact_cfg.get("hand", "auto")).lower()
            contact_hand = hand_selection.hand if configured_hand == "auto" else configured_hand
            if contact_hand not in ("left", "right", "both"):
                raise ValueError(f"contact.hand must be auto/left/right/both, got {contact_hand!r}")
            explicit_window = contact_cfg.get("approach_window")
            hand_joints_for_ray = initial_hand_joints
            if contact_hand == "left":
                hand_joints_for_ray = initial_hand_joints[:, : initial_hand_joints.shape[1] // 2]
            elif contact_hand == "right":
                hand_joints_for_ray = initial_hand_joints[:, initial_hand_joints.shape[1] // 2 :]
            initial_hand_world = hand_joints_for_ray.mean(dim=1)
            initial_hand_cam = transform_world_to_camera(
                initial_hand_world, opencv_cam_R, opencv_cam_t
            )
            object_cam_seq = transform_world_to_camera(
                obj_verts_seq.reshape(-1, 3), opencv_cam_R, opencv_cam_t
            ).reshape(frame_num, -1, 3)
            intrinsics = lift4d_depth.camera_intrinsics.detach()
            hand_pixel = torch.stack(
                [
                    intrinsics[:, 0, 0] * initial_hand_cam[:, 0] / initial_hand_cam[:, 2]
                    + intrinsics[:, 0, 2],
                    intrinsics[:, 1, 1] * initial_hand_cam[:, 1] / initial_hand_cam[:, 2]
                    + intrinsics[:, 1, 2],
                ],
                dim=1,
            )
            object_center_cam = object_cam_seq[move_start - 1].median(dim=0).values
            required_displacement = float(
                torch.abs(initial_hand_cam[move_start - 1, 2] - object_center_cam[2]).detach()
            )
            if explicit_window is None:
                approach_window = approach_window_from_fps(
                    video_fps,
                    required_displacement,
                    max_hand_speed_mps=float(contact_cfg.get("max_hand_speed_mps", 0.4)),
                    min_approach_frames=int(contact_cfg.get("min_approach_frames", 20)),
                    max_approach_frames=int(contact_cfg.get("max_approach_frames", 60)),
                )
            else:
                approach_window = int(
                    np.clip(
                        int(explicit_window),
                        int(contact_cfg.get("min_approach_frames", 20)),
                        int(contact_cfg.get("max_approach_frames", 60)),
                    )
                )
            # Stage A has not run yet.  Keep only the fixed input ray/pixels;
            # final detached surface targets are created after Stage A.
            hand_initial_cam = initial_hand_cam.detach()
            hand_pixels = hand_pixel.detach()
            hand_initial_cam_depth = initial_hand_cam[:, 2].detach()
            window_start = max(0, move_start - approach_window)
            window_end = frame_num - 1
            # Compatibility fields are derived from physical motion only. They do
            # not read GPT/cache contact metadata in the formal path.
            contact_hint = move_start
            contact_hint_source = "adjacent_object_mask_motion"
            self.logger.info(
                "Object motion state | t_move=%d confidence=%.4f static_z=%.6f | "
                "static_hard_freeze=[0,%d) approach=[%d,%d] grasp=[%d,%d] hand=%s (%s)",
                move_start,
                motion_state.confidence,
                motion_state.static_z,
                move_start,
                window_start,
                move_start,
                move_start,
                frame_num - 1,
                contact_hand,
                hand_selection.reason,
            )
        else:
            contact_hint, contact_hint_source = self._resolve_contact_hint(
                contact_cfg, contact_start_idx, inter_start_idx, frame_num
            )
            contact_hand = self._resolve_contact_hand(contact_cfg, contact_labels)
            approach_window = int(contact_cfg.get("approach_window") or 30)
            window_start = contact_hint
            window_end = contact_hint

        # ── Assemble HOIData ─────────────────────────────────────────────────
        return HOIData(
            frame_num=frame_num,
            inter_start_idx=inter_start_idx,
            inter_end_idx=inter_end_idx,
            human=HOIData.Human(
                faces=human_faces,
                masks=[human_masks[i] for i in range(frame_num)],
                motion_data=motion_data,
                motion_data_global_init=motion_data_global_init,
                body_keypoints_seq=gt_body_kp,
                hand_keypoints_seq=gt_hand_kp,
                foot_contact_probs=foot_contact_probs,
            ),
            obj=HOIData.Object(
                scale=obj_scale,
                verts=obj_verts,
                faces=obj_faces,
                masks=[obj_masks[i] for i in range(frame_num)],
                verts_seq=obj_verts_seq,
                poses=obj_poses,
                poses_cam=obj_poses_incam,
                fp_ray_cam=fp_ray_cam,
                verts_tracking_seq=gt_obj_verts_tracking,
            ),
            camera=camera,
            images_path=images_path,
            depth_maps=depth_maps,
            is_static_obj=is_static_obj,
            static_objects=static_objects,
            lift4d_motion=lift4d_motion,
            lift4d_depth=lift4d_depth,
            object_motion_state=motion_state,
            contact_frame=contact_hint,
            contact_hint=contact_hint,
            contact_hint_source=contact_hint_source,
            contact_window_start=window_start,
            contact_window_end=window_end,
            contact_hand=contact_hand,
            approach_window=approach_window,
            hand_selection_left_distance_px=(
                None if hand_selection is None else hand_selection.left_distance_px
            ),
            hand_selection_right_distance_px=(
                None if hand_selection is None else hand_selection.right_distance_px
            ),
            hand_selection_reason=(
                "legacy_contact_labels" if hand_selection is None else hand_selection.reason
            ),
            hand_ray_target_world=hand_ray_target_world,
            hand_ray_ramp=hand_ray_ramp,
            hand_initial_cam=hand_initial_cam,
            hand_pixels=hand_pixels,
            hand_ray_surface_fallback=hand_ray_surface_fallback,
            hand_initial_cam_depth=hand_initial_cam_depth,
            hand_target_cam_depth=hand_target_cam_depth,
            object_surface_depth=object_surface_depth,
            hand_approach_initial_distance=hand_approach_initial_distance,
        )

    @staticmethod
    def _resolve_contact_hint(contact_cfg, contact_start_idx, inter_start_idx, frame_num):
        return resolve_contact_hint(
            contact_cfg.get("frame", None),
            contact_start_idx,
            inter_start_idx,
            frame_num,
            explicit_source=str(contact_cfg.get("hint_source", "cli")),
        )

    @staticmethod
    def _resolve_contact_hand(contact_cfg, contact_labels):
        return infer_contact_hand(
            contact_cfg.get("hand", "auto"),
            contact_labels,
            fallback=contact_cfg.get("hand_fallback", "right"),
        )

    # ── init_data sub-methods (no self.* side effects) ───────────────────────

    def _load_camera_config(self, render_config_file):
        """Load rendering config and create camera."""
        _, _, obj_scale, blender_cam_R, blender_cam_t, render_config, additional_data = (
            load_init_rendering_data(
                render_config_file,
                to_tensor=True,
                with_human_data=True,
                with_scene_data=True,
                device=self.device,
            )
        )
        opencv_cam_R, opencv_cam_t = cam_pose_blender_to_opencv(blender_cam_R, blender_cam_t)
        if render_config is not None:
            frame_height, frame_width, focal_length = render_config
        else:
            frame_height, frame_width, focal_length = HEIGHT, WIDTH, FOCAL_LENGTH

        static_objects = additional_data.get("static_objects", None)

        cam_R, cam_t = cam_pose_opencv_to_pytorch3d(opencv_cam_R, opencv_cam_t)
        cameras = get_camera(
            cam_R, cam_t, focal_length, (frame_height, frame_width), device=self.device
        )

        cam_pose = torch.eye(4, device=self.device)
        cam_pose[:3, :3] = cam_R
        cam_pose[:3, 3] = cam_t.reshape((3,))

        camera = HOIData.Camera(
            pose=cam_pose,
            frame_height=frame_height,
            frame_width=frame_width,
            focal_length=focal_length,
        )
        return camera, cameras, opencv_cam_R, opencv_cam_t, obj_scale, static_objects

    def _load_object(self, obj_path, obj_scale, obj_pose_file, opencv_cam_R, opencv_cam_t):
        """Load object mesh and pose trajectory."""
        obj_verts, obj_faces, _ = load_mesh(
            obj_path, mesh_scale=obj_scale, target_num_verts=6000, device=self.device
        )
        obj_poses_incam = load_object_pose_data(obj_pose_file, to_tensor=True, device=self.device)
        fp_ray_cam = foundationpose_camera_rays(obj_poses_incam[:, :3, 3])
        obj_poses = transform_pose_c2w(obj_poses_incam, opencv_cam_R, opencv_cam_t)
        return obj_verts, obj_faces, obj_poses_incam, obj_poses, fp_ray_cam

    def _load_video_frames(self, video_file):
        """Extract video frames and return image paths and fps."""
        video_basename = os.path.basename(video_file).split(".")[0]
        frame_cache_dir = os.path.join(os.path.dirname(video_file), "frames", video_basename)

        video_fps, video_frame_count = get_video_fps_and_frame_count(video_file)

        images_path = sorted(glob(os.path.join(frame_cache_dir, "*.jpg")))
        return images_path, video_fps, video_frame_count

    def _load_masks(self, video_id):
        """Load and split preprocessed masks into human and object masks."""
        masks_cache_file = os.path.join(self.cache_dir, "masks", f"{video_id}.npz")
        if not os.path.exists(masks_cache_file):
            raise FileNotFoundError(
                f"Masks cache not found: {masks_cache_file}. Run preprocessing (step1) first."
            )
        preprocess_masks = load_masks_from_cache(masks_cache_file)
        # Preprocess saves obj=0, human=1 — split into separate dicts
        human_masks = {fi: preprocess_masks[fi][1] for fi in preprocess_masks}
        obj_masks = {fi: preprocess_masks[fi][0] for fi in preprocess_masks}
        self.logger.info(f"Loaded masks from cache: {masks_cache_file}")
        return human_masks, obj_masks

    def _detect_interaction(self, obj_poses_incam, images_path, human_masks, obj_masks):
        """Detect interaction start/end frame."""
        has_interaction_end = self.cfg.get("has_interaction_end", False)
        if self.cfg.get("detect_interaction_with_mask", False):
            self.logger.info("Using mask-based interaction detection")
            masks = {fi: {0: human_masks[fi], 1: obj_masks[fi]} for fi in human_masks}
            result = identify_interaction_start_end_with_mask(
                masks,
                images_path,
                self.logger,
                has_interaction_end=has_interaction_end,
            )
        else:
            result = identify_interaction_start_end(
                obj_poses_incam,
                images_path,
                self.obj_name,
                self.logger,
                has_interaction_end=has_interaction_end,
            )
        inter_start_idx, inter_end_idx, is_static_obj = result
        if self.cfg.get("is_static_obj", False):
            # override the result if is_static_obj is set True in the config
            is_static_obj = True
        self.logger.info(f"Interaction start frame: {inter_start_idx}/{len(images_path)}")
        self.logger.info(f"Interaction end frame: {inter_end_idx}/{len(images_path)}")
        self.logger.info(f"Is static object: {is_static_obj}")
        return inter_start_idx, inter_end_idx, is_static_obj

    def _load_motion(
        self, hmr_file, render_config_file, opencv_cam_R, opencv_cam_t, inter_start_idx
    ):
        """Load human motion data and transform to camera frame.

        Handles shape/height scaling so that motion_data is fully scaled on return.
        Two cases:
            1. GT shape params available: replace predicted shape, transform_global_motion
               handles translation scaling internally.
            2. No GT shape, but GT height: keep predicted shape, adjust body scale so
               generate_mesh produces correctly sized geometry.
        """
        global_motion_data = load_human_motion_data(
            hmr_file, is_global=True, to_tensor=True, device=self.device
        )
        incam_motion_data = load_human_motion_data(
            hmr_file, is_global=False, to_tensor=True, device=self.device
        )
        foot_contact_probs = incam_motion_data["foot_contact_probs"]

        # HMR-predicted body height is computed upstream (human_pose.py) and
        # stored in the motion dict. Fail fast if it's missing (stale cache).
        if incam_motion_data.get("predicted_body_height", None) is None:
            raise RuntimeError(
                "predicted_body_height missing from motion data. "
                "Regenerate HMR output (step 1 of grail.pipelines.recon_4dhoi) to populate this field."
            )

        # Load GT character height from rendering metadata
        character_height = None
        for candidate in [
            render_config_file.replace("first_frame_pose.pickle", "character_data.pickle"),
            render_config_file.replace("foundation_pose_output", "foundation_pose").replace(
                "first_frame_pose.pickle", "character_data.pickle"
            ),
        ]:
            character_data = load_character_data(candidate)
            if character_data is not None:
                character_height = character_data.get("character_height", None)
                break

        # Load GT shape params (from pre-fitted files)
        gt_shape_params = self.human_model.load_shape_params(
            self.cfg,
            self.character_name,
            self.device
        )

        use_global = self.cfg.get("use_global", False)
        motion_data = self.human_model.transform_global_motion(
            global_motion_data,
            incam_motion_data,
            cam_R=opencv_cam_R,
            cam_t=opencv_cam_t,
            align_frame=inter_start_idx,
            use_global=use_global,
            gt_shape_params=gt_shape_params,
            gt_height=character_height,
            device=self.device,
        )
        motion_data_global_init = self.human_model.transform_global_motion(
            global_motion_data,
            incam_motion_data,
            cam_R=opencv_cam_R,
            cam_t=opencv_cam_t,
            align_frame=inter_start_idx,
            use_global=True,
            gt_shape_params=gt_shape_params,
            gt_height=character_height,
            device=self.device,
        )
        return motion_data, motion_data_global_init, foot_contact_probs

    def _detect_contact_labels(
        self, video_id, frame_num, inter_start_idx, is_static_obj, images_path
    ):
        """Detect or load cached per-interval contact labels.

        Returns:
            (contact_labels, contact_interval, contact_start_idx)
            contact_labels is a list of per-interval label lists, e.g. [["R_Hand"], None, ["L_Hand"]]
        """
        interval = self.cfg.get("contact_interval_length", 8)
        cache_file = os.path.join(self.cache_dir, "contact_labels", f"{video_id}.json")

        # skip VLM contact detection for static objects (e.g. stairs)
        # since the contact loss is zeroed out for them anyway.
        if is_static_obj:
            contact_labels = []
            interval = []
            start_idx = inter_start_idx
            self.logger.info("Skipping contact label detection (is_static_obj=True)")
        elif "contact_labels" in self.cache_list and os.path.exists(cache_file):
            self.logger.info(f"Loading contact labels from cache: {cache_file}")
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            # Support both old and new cache key names
            contact_labels = cache_data.get(
                "contact_labels", cache_data.get("contact_labels_per_interval", [])
            )
            interval = cache_data.get(
                "contact_interval", cache_data.get("contact_interval_length", interval)
            )
            start_idx = cache_data.get(
                "contact_start_idx", cache_data.get("contact_interval_start_idx", inter_start_idx)
            )
        elif self.cfg.get("contact_labels", None) is not None:
            per_interval = self.cfg["contact_labels"]
            n_intervals = max(1, (frame_num - inter_start_idx + interval - 1) // interval)
            contact_labels = [per_interval for _ in range(n_intervals)]
            start_idx = inter_start_idx
            self.logger.info(f"Using contact labels from config: {per_interval}")
        else:
            contact_labels = detect_contact_joints_interval(
                images_path,
                self.obj_name,
                interval_length=interval,
                start_idx=inter_start_idx,
                end_idx=frame_num,
            )
            start_idx = inter_start_idx

            # Validate: ensure at least one interval has labels
            has_any = any(labels is not None and len(labels) > 0 for labels in contact_labels)
            if not has_any:
                contact_labels = [["R_Hand"]]
                self.logger.warning("No contact labels detected, using default: R_Hand")

            self.logger.info(f"Detected contact labels: {contact_labels}")

            # Save to cache
            if "contact_labels" in self.cache_list:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(
                        {
                            "contact_labels": contact_labels,
                            "contact_interval": interval,
                            "contact_start_idx": start_idx,
                        },
                        f,
                        indent=2,
                    )
                self.logger.info(f"Saved contact labels to cache: {cache_file}")

        # backward compatibility with old caches
        for i, entry in enumerate(contact_labels):
            if isinstance(entry, str):
                contact_labels[i] = [entry]

        self.logger.info(
            f"Contact labels ({len(contact_labels)} intervals, interval={interval}): {contact_labels}"
        )
        return contact_labels, interval, start_idx

    def _load_depth(self, video_id):
        """Load preprocessed depth maps from cache."""
        depth_cache_file = os.path.join(self.cache_dir, "depth", f"{video_id}.pt")
        if not os.path.exists(depth_cache_file):
            raise FileNotFoundError(
                f"Depth cache not found: {depth_cache_file}. Run preprocessing (step1) first."
            )
        depth_maps = load_depth_from_cache(depth_cache_file, device=self.device)
        self.logger.info(f"Loaded depth from cache: {depth_cache_file}")
        return depth_maps

    def _resolve_lift4d_depth_prior_path(self, video_id):
        prior_path = self.cfg.get("lift4d_motion_prior_path")
        if not prior_path:
            raise ValueError(
                "use_lift4d_depth_prior=true requires an explicit lift4d_motion_prior_path; "
                "no default or guessed NPZ path is allowed"
            )
        prior_path = str(prior_path).format(
            video_id=video_id, video_id_safe=video_id.replace("/", "__")
        )
        if not os.path.isabs(prior_path):
            results_dir = self.cfg.get("results_dir")
            if not results_dir:
                raise ValueError("Relative lift4d_motion_prior_path requires explicit results_dir")
            prior_path = os.path.join(results_dir, prior_path)
        return os.path.abspath(prior_path)

    def _load_lift4d_depth_prior(self, video_id, frame_num):
        if not self.cfg.get("use_lift4d_depth_prior", False):
            return None
        if not self.cfg.get("freeze_foundationpose_image_plane_translation", False):
            raise ValueError(
                "Lift4D depth supervision requires freeze_foundationpose_image_plane_translation=true"
            )
        if self.cfg.get("camera_mode", "fixed") != "fixed":
            raise ValueError(
                "Lift4D ray-depth optimization currently requires the real fixed-camera GRAIL path; "
                "dynamic-camera HOI optimization does not yet consume per-frame c2w matrices"
            )
        prior_path = self._resolve_lift4d_depth_prior_path(video_id)
        prior = load_lift4d_depth_prior(
            prior_path,
            frame_num=frame_num,
            median_window=self.cfg.get("lift4d_median_window", 7),
            detection_median_window=(
                self.cfg.get("object_motion_state", {}) or {}
            ).get("detection_median_window", 5),
            smooth_window=self.cfg.get("lift4d_center_smooth_window", 31),
            savgol_polyorder=self.cfg.get("lift4d_savgol_polyorder", 2),
            stable_point_count=self.cfg.get("lift4d_stable_point_count", 2500),
            min_stable_points=self.cfg.get("lift4d_min_stable_points", 64),
        )
        self.logger.info(
            "Loaded real Lift4D point-depth prior: %s | frames=%d | stable_points=%d | "
            "z=[%.5f, %.5f]",
            prior.source_path,
            frame_num,
            prior.stable_point_ids.size,
            float(prior.z.min()),
            float(prior.z.max()),
        )
        to_tensor = lambda value, dtype=torch.float32: torch.as_tensor(
            value, dtype=dtype, device=self.device
        )
        return HOIData.Lift4DDepth(
            frame_indices=to_tensor(prior.frame_indices, torch.long),
            prior_used=to_tensor(prior.prior_used, torch.bool),
            center_cam_raw=to_tensor(prior.center_cam_raw),
            center_cam_detection=to_tensor(prior.center_cam_detection),
            center_cam=to_tensor(prior.center_cam),
            z_raw=to_tensor(prior.z_raw),
            z=to_tensor(prior.z),
            z_target=to_tensor(prior.z),
            delta_z=to_tensor(prior.delta_z),
            frame_weight=to_tensor(prior.frame_weight),
            valid_point_count=to_tensor(prior.valid_point_count, torch.long),
            camera_intrinsics=to_tensor(prior.camera_intrinsics),
            stable_point_ids=to_tensor(prior.stable_point_ids, torch.long),
            source_path=prior.source_path,
            camera_convention=prior.camera_convention,
            diagnostics=prior.diagnostics,
        )

    def _validate_ray_projection(self, fp_poses_cam, fp_ray_cam, prior):
        fp_trans = fp_poses_cam[:, :3, 3]
        probe_z = fp_trans[:, 2] * 0.75
        probe_trans = ray_depth_translation(fp_ray_cam, probe_z)

        def project(trans):
            K = prior.camera_intrinsics
            return torch.stack(
                [
                    K[:, 0, 0] * trans[:, 0] / trans[:, 2] + K[:, 0, 2],
                    K[:, 1, 1] * trans[:, 1] / trans[:, 2] + K[:, 1, 2],
                ],
                dim=1,
            )

        pixel_error = torch.linalg.norm(project(probe_trans) - project(fp_trans), dim=1)
        stats = (
            float(pixel_error.mean()),
            float(pixel_error.median()),
            float(pixel_error.max()),
        )
        self.logger.info(
            "Ray-depth projection check | mean=%.8g px | median=%.8g px | max=%.8g px",
            *stats,
        )
        if stats[2] > 1e-3:
            raise ValueError(
                f"FoundationPose ray projection drift is too large: max={stats[2]:.8g} px"
            )

    def init_params(self, data):
        frame_num = data.frame_num
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=self.device)

        self.num_body_joints = self.human_model.num_body_joints
        self.num_hand_joints = self.human_model.num_hand_joints

        use_ray_depth = bool(self.cfg.get("use_lift4d_depth_prior", False))
        if use_ray_depth and data.lift4d_depth is None:
            raise ValueError("Ray-depth parameterization requested without a real Lift4D depth prior")
        fixed_depth_scale = float(self.cfg.get("lift4d_depth_scale", 1.0))
        if not np.isfinite(fixed_depth_scale) or fixed_depth_scale <= 0:
            raise ValueError(f"lift4d_depth_scale must be positive, got {fixed_depth_scale}")
        learn_depth_scale = bool(self.cfg.get("learn_lift4d_depth_scale", False))
        if learn_depth_scale and not use_ray_depth:
            raise ValueError("learn_lift4d_depth_scale requires use_lift4d_depth_prior=true")

        self.params = OptParams(
            human_trans_global=torch.zeros(3, device=self.device, requires_grad=True),
            human_trans_res=torch.zeros(frame_num, 3, device=self.device, requires_grad=True),
            human_pose_res=torch.tensor(
                identity_6d.reshape(1, 1, 6).repeat(frame_num, self.num_body_joints, 1),
                device=self.device,
                requires_grad=True,
            ),
            hand_pose_res=torch.tensor(
                identity_6d.reshape(1, 1, 6).repeat(frame_num, self.num_hand_joints, 1),
                device=self.device,
                requires_grad=True,
            ),
            obj_R_res=torch.tensor(
                identity_6d.unsqueeze(0).repeat(frame_num, 1),
                device=self.device,
                requires_grad=True,
            ),
            obj_t_res=(
                None
                if use_ray_depth
                else torch.zeros(frame_num, 3, device=self.device, requires_grad=True)
            ),
            obj_depth_res=(
                torch.zeros(frame_num, device=self.device, requires_grad=True)
                if use_ray_depth
                else None
            ),
            human_approach_distance=torch.zeros((), device=self.device, requires_grad=True),
            obj_z_opt=None,
            log_lift4d_depth_scale=(
                torch.tensor(
                    np.log(fixed_depth_scale),
                    dtype=data.obj.poses.dtype,
                    device=self.device,
                    requires_grad=learn_depth_scale,
                )
                if use_ray_depth
                else None
            ),
        )
        return self.params

    @torch.no_grad()
    def initialize_obj_depth_from_lift4d(self, data):
        """Initialize every object depth from the real anchor-relative Lift4D target."""
        if self.params.obj_depth_res is None or data.lift4d_depth is None:
            raise ValueError("Lift4D depth initialization requires obj_depth_res and a real prior")
        fp_z = data.obj.poses_cam[:, 2, 3].detach()
        lift_z = data.lift4d_depth.z_target.detach()
        if fp_z.shape != lift_z.shape or fp_z.shape != self.params.obj_depth_res.shape:
            raise ValueError(
                "Lift4D depth initialization shape mismatch: "
                f"fp={tuple(fp_z.shape)} lift={tuple(lift_z.shape)} "
                f"residual={tuple(self.params.obj_depth_res.shape)}"
            )
        motion_scale = float(self.cfg.get("lift4d_depth_scale", 1.0))
        if not np.isfinite(motion_scale) or motion_scale <= 0:
            raise ValueError(f"lift4d_depth_scale must be positive, got {motion_scale}")
        target_z = fp_z[0] + motion_scale * (lift_z - lift_z[0])
        if data.object_motion_state is not None:
            move_start = int(data.object_motion_state.move_start_frame)
            target_z[:move_start] = fp_z[0]
        if torch.any(target_z <= 0) or not torch.isfinite(target_z).all():
            raise ValueError("Lift4D initialized object depth must be finite and positive")
        self.params.obj_depth_res.copy_(target_z - fp_z)
        self.params.obj_depth_res[0] = 0.0
        self.logger.info(
            "Initialized full-frame object depth from Lift4D: max |residual|=%.6f m",
            float(self.params.obj_depth_res.abs().max()),
        )
        return target_z

    @staticmethod
    def _freeze_static_object_pose_inputs(data, obj_poses, fp_ray_cam, pred_obj_z):
        if data.object_motion_state is None:
            return obj_poses, fp_ray_cam, pred_obj_z
        static = np.asarray(
            getattr(
                data.object_motion_state,
                "static",
                np.arange(int(data.frame_num)) < int(data.object_motion_state.move_start_frame),
            ),
            dtype=bool,
        )
        if static.shape != (int(data.frame_num),) or not static[0]:
            raise ValueError("Object motion state must provide a frame-aligned static mask starting static")
        frozen_poses = obj_poses.clone()
        frozen_rays = fp_ray_cam.clone()
        frozen_z = pred_obj_z.clone()
        for frame in range(1, int(data.frame_num)):
            if static[frame]:
                frozen_poses[frame, :3, :3] = frozen_poses[frame - 1, :3, :3]
                frozen_rays[frame] = frozen_rays[frame - 1]
                frozen_z[frame] = frozen_z[frame - 1]
        return frozen_poses, frozen_rays, frozen_z

    @staticmethod
    def _freeze_static_object_rotation_residual(data, obj_R_res_mat):
        if data.object_motion_state is None:
            return obj_R_res_mat
        static = np.asarray(
            getattr(
                data.object_motion_state,
                "static",
                np.arange(int(data.frame_num)) < int(data.object_motion_state.move_start_frame),
            ),
            dtype=bool,
        )
        if static.shape != (int(data.frame_num),) or not static[0]:
            raise ValueError("Object motion state must provide a frame-aligned static mask starting static")
        frozen_residual = obj_R_res_mat.clone()
        for frame in range(1, int(data.frame_num)):
            if static[frame]:
                frozen_residual[frame] = frozen_residual[frame - 1]
        return frozen_residual

    def get_opt_params(self, params, opt_vars, is_static_obj=False):
        opt_params = {}

        for opt_var in opt_vars:
            if is_static_obj and opt_var in ("obj_R_res", "obj_t_res", "obj_depth_res", "obj_z_opt"):
                self.logger.info(f"Skipping optimization of {opt_var} (static object)")
                continue

            if hasattr(params, opt_var) and getattr(params, opt_var) is not None:
                opt_params[opt_var] = getattr(params, opt_var)
            else:
                self.logger.warning(f"Optimization parameter {opt_var} not found")

        return opt_params

    def get_contact_labels_for_frame(self, frame_idx):
        return get_contact_labels_for_frame(
            frame_idx,
            self.contact_labels,
            self.contact_start_idx,
            self.contact_interval,
        )

    def init_opt(self, data, params, opt_config):
        is_static_obj = data.is_static_obj
        configured_opt_vars = dict(opt_config["opt_vars"])
        use_ray_depth = bool(self.cfg.get("use_lift4d_depth_prior", False))
        if use_ray_depth and "obj_t_res" in configured_opt_vars:
            raise ValueError(
                "Formal Lift4D depth optimization must configure obj_depth_res, not obj_t_res"
            )
        elif not use_ray_depth:
            configured_opt_vars.pop("obj_depth_res", None)
        if use_ray_depth and "obj_z_opt" in configured_opt_vars:
            raise ValueError("Deprecated absolute obj_z_opt is prohibited; use obj_depth_res")
        if use_ray_depth and self.cfg.get("learn_lift4d_depth_scale", False):
            configured_opt_vars["log_lift4d_depth_scale"] = {
                "lr": float(self.cfg.get("lift4d_depth_scale_lr", 1e-3))
            }
        opt_params = self.get_opt_params(
            params, configured_opt_vars.keys(), is_static_obj=is_static_obj
        )
        if len(opt_params.keys()) == 0:
            return None, opt_params

        opt_params_cfg = []
        for opt_var in opt_params.keys():
            var_config = configured_opt_vars[opt_var]
            # Check if xy_only is set for human_trans_global
            if opt_var == "human_trans_global" and var_config.get("xy_only", False):
                # Register a gradient hook to zero out z-dimension gradients (1D tensor)
                def xy_only_hook_1d(grad):
                    grad[2] = 0  # Zero out z-dimension gradient
                    return grad

                opt_params[opt_var].register_hook(xy_only_hook_1d)
                opt_params_cfg.append(
                    {
                        "params": opt_params[opt_var],
                        "lr": var_config["lr"],
                    }
                )
                self.logger.info(f"Optimizing {opt_var} with xy_only=True (only x, y dimensions)")
            # Check if xy_only is set for human_trans_res
            elif opt_var == "human_trans_res" and var_config.get("xy_only", False):
                # Register a gradient hook to zero out z-dimension gradients (2D tensor)
                def xy_only_hook_2d(grad):
                    grad[:, 2] = 0  # Zero out z-dimension gradient
                    return grad

                opt_params[opt_var].register_hook(xy_only_hook_2d)
                opt_params_cfg.append(
                    {
                        "params": opt_params[opt_var],
                        "lr": var_config["lr"],
                    }
                )
                self.logger.info(f"Optimizing {opt_var} with xy_only=True (only x, y dimensions)")
            else:
                group = {
                    "params": opt_params[opt_var],
                    "lr": var_config["lr"],
                }
                if opt_var in (
                    "obj_depth_res",
                    "human_approach_distance",
                    "obj_z_opt",
                    "log_lift4d_depth_scale",
                ):
                    # These are absolute physical quantities, not zero-centered
                    # residuals. AdamW decay would introduce an unrelated depth drift.
                    group["weight_decay"] = 0.0
                opt_params_cfg.append(group)

        optimizer = torch.optim.AdamW(opt_params_cfg)

        return optimizer, opt_params

    def forward(self, data, params):
        frame_num = data.frame_num

        # Extract per-frame parameters
        human_trans_res = params.human_trans_res
        human_pose_res = params.human_pose_res
        hand_pose_res = params.hand_pose_res
        obj_R_res = params.obj_R_res
        obj_t_res = params.obj_t_res

        # Predict human trajectory
        human_trans_global = params.human_trans_global
        human_trans_res = human_trans_res.reshape(frame_num, 1, 3)

        human_pose_res = human_pose_res.reshape(frame_num, self.num_body_joints, 6)
        hand_pose_res = hand_pose_res.reshape(frame_num, self.num_hand_joints, 6)

        motion_data = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in data.human.motion_data.items()
        }

        motion_data = self.human_model.apply_pose_residuals(
            motion_data,
            human_pose_res,
            hand_pose_res,
            frame_num,
        )

        pred_root_pose, pred_body_pose = self.human_model.extract_root_body_pose(motion_data)

        # Apply translation residuals
        trans = motion_data["trans"].reshape(frame_num, 3)
        approach_end = (
            int(data.object_motion_state.move_start_frame)
            if data.object_motion_state is not None
            else int(data.contact_frame)
        )
        approach_ramp = smoothstep_approach_ramp(
            frame_num,
            approach_end,
            data.approach_window,
            device=trans.device,
            dtype=trans.dtype,
        )
        approach_direction = getattr(self, "_human_approach_direction", None)
        if approach_direction is None:
            approach_direction = torch.zeros(3, device=trans.device, dtype=trans.dtype)
        approach_offset, approach_distance = approach_offsets(
            approach_ramp,
            params.human_approach_distance,
            approach_direction,
            max_distance=self.cfg.get("max_human_approach_distance", 0.35),
        )
        motion_data["trans"] = (
            trans
            + human_trans_res.reshape(frame_num, 3)
            + human_trans_global.reshape(1, 3)
            + approach_offset
        )

        pred_human_verts_seq, _ = self.human_model.generate_mesh(
            motion_data,
            output_joints=False,
            require_grad=True,
        )

        pred_body_joints_seq = self.human_model.get_body_joints(motion_data, require_grad=True)
        pred_hand_joints_seq = self.human_model.get_hand_joints(motion_data, require_grad=True)

        pred_human_root_trans = motion_data["trans"]

        # Project body and hand joints to 2D screen coordinates
        pred_body_keypoints_seq = project_world_to_screen(
            pred_body_joints_seq.reshape(-1, 3), self.cameras
        ).reshape(frame_num, -1, 3)[:, :, :2]

        pred_hand_keypoints_seq = project_world_to_screen(
            pred_hand_joints_seq.reshape(-1, 3), self.cameras
        ).reshape(frame_num, -1, 3)[:, :, :2]

        pred_human = HOIPrediction.Human(
            trans=pred_human_root_trans,
            root_pose=pred_root_pose,
            pose=pred_body_pose,
            verts_seq=pred_human_verts_seq,
            body_joints_seq=pred_body_joints_seq,
            body_keypoints_seq=pred_body_keypoints_seq,
            hand_joints_seq=pred_hand_joints_seq,
            hand_keypoints_seq=pred_hand_keypoints_seq,
            pose_res=human_pose_res,
            trans_res=human_trans_res,
            approach_ramp=approach_ramp,
            approach_offset=approach_offset,
            approach_distance=approach_distance,
            motion_data=motion_data,
        )

        # Predict object trajectory
        obj_verts = data.obj.verts
        obj_R_res_mat = rotation_6d_to_matrix(obj_R_res)
        obj_R_res_mat = self._freeze_static_object_rotation_residual(data, obj_R_res_mat)

        obj_poses = data.obj.poses.clone()
        if params.obj_depth_res is not None:
            fp_z = data.obj.poses_cam[:, 2, 3].detach()
            pred_obj_z = fp_z + params.obj_depth_res.reshape(frame_num)
            obj_poses, fp_ray_cam, pred_obj_z = self._freeze_static_object_pose_inputs(
                data, obj_poses, data.obj.fp_ray_cam.detach(), pred_obj_z
            )
            if torch.any(pred_obj_z <= 0):
                raise ValueError("Optimized OpenCV camera depth became non-positive")
            pred_obj_t_cam = ray_depth_translation(fp_ray_cam, pred_obj_z)
            pred_obj_t = transform_camera_to_world(
                pred_obj_t_cam, self.opencv_cam_R, self.opencv_cam_t
            )
            depth_scale = positive_depth_scale(
                params.log_lift4d_depth_scale,
                minimum=self.cfg.get("lift4d_depth_scale_min", 0.25),
                maximum=self.cfg.get("lift4d_depth_scale_max", 4.0),
            )
        else:
            pred_obj_t = obj_poses[:, :3, 3].reshape(frame_num, 3) + obj_t_res.reshape(frame_num, 3)
            pred_obj_t_cam = transform_world_to_camera(
                pred_obj_t, self.opencv_cam_R, self.opencv_cam_t
            )
            pred_obj_z = pred_obj_t_cam[:, 2]
            depth_scale = pred_obj_t.new_tensor(1.0)

        pred_obj_R = torch.bmm(obj_R_res_mat, obj_poses[:, :3, :3])

        pred_obj_verts_seq = obj_verts.unsqueeze(0).repeat(frame_num, 1, 1)
        pred_obj_verts_seq = torch.bmm(
            pred_obj_verts_seq, pred_obj_R.transpose(1, 2)
        ) + pred_obj_t.reshape(frame_num, 1, 3)
        w2c = world_to_camera_matrix(self.opencv_cam_R, self.opencv_cam_t)
        pred_obj_R_cam = torch.matmul(w2c[:3, :3].unsqueeze(0), pred_obj_R)

        pred_obj = HOIPrediction.Object(
            trans=pred_obj_t,
            trans_cam=pred_obj_t_cam,
            z_cam=pred_obj_z,
            depth_scale=depth_scale,
            R=pred_obj_R,
            R_cam=pred_obj_R_cam,
            verts_seq=pred_obj_verts_seq,
        )

        return HOIPrediction(human=pred_human, obj=pred_obj)

    @torch.no_grad()
    def refresh_hand_ray_targets_after_object_stage(self, data):
        """Rebuild detached hand targets from the Stage-A optimized object mesh."""
        if data.object_motion_state is None:
            raise ValueError("Ray target refresh requires formal object motion state")
        if data.hand_initial_cam is None or data.hand_pixels is None:
            raise ValueError("init_data did not preserve hand ray inputs")
        pred = self.forward(data, self.params)
        object_cam_seq = transform_world_to_camera(
            pred.obj.verts_seq.detach().reshape(-1, 3),
            self.opencv_cam_R,
            self.opencv_cam_t,
        ).reshape(data.frame_num, -1, 3)
        intrinsics = data.lift4d_depth.camera_intrinsics.detach()
        surface_depth, fallback = mesh_surface_depth_at_pixels(
            object_cam_seq,
            data.hand_pixels.detach(),
            intrinsics,
            object_faces=data.obj.faces.detach(),
            current_hand_depth=data.hand_initial_cam[:, 2].detach(),
            top_k=32,
        )
        target_cam, ramp = camera_ray_hand_targets(
            data.hand_initial_cam.detach(),
            surface_depth.detach(),
            int(data.object_motion_state.move_start_frame),
            int(data.approach_window),
            target_distance=float((self.cfg.get("contact", {}) or {}).get("target_distance", 0.02)),
        )
        data.hand_ray_target_world = transform_camera_to_world(
            target_cam.detach(), self.opencv_cam_R, self.opencv_cam_t
        ).detach()
        data.hand_ray_ramp = ramp.detach()
        data.hand_initial_cam_depth = data.hand_initial_cam[:, 2].detach()
        data.hand_target_cam_depth = target_cam[:, 2].detach()
        data.object_surface_depth = surface_depth.detach()
        data.hand_ray_surface_fallback = fallback.detach()
        start = max(0, int(data.object_motion_state.move_start_frame) - int(data.approach_window))
        data.hand_approach_initial_distance = float(
            torch.abs(data.hand_initial_cam[start, 2] - surface_depth[start]).detach()
        )
        self.logger.info(
            "Refreshed Stage-A hand ray targets | surface_z[move]=%.6f | "
            "fallback_frames=%d | initial_distance=%.6f",
            float(surface_depth[int(data.object_motion_state.move_start_frame)]),
            int(fallback.sum()),
            float(data.hand_approach_initial_distance),
        )
        return data.hand_ray_target_world

    @torch.no_grad()
    def capture_stage_boundary_state(self, data):
        """Freeze Stage-B endpoint state for continuous Stage-C refinement."""
        if data.object_motion_state is None:
            raise ValueError("Boundary state requires formal object motion state")
        move_start = int(data.object_motion_state.move_start_frame)
        if move_start < 1:
            raise ValueError("t_move must be >= 1 for boundary state")
        pred = self.forward(data, self.params)
        hand = self.human_model.get_verts_segment(
            pred.human.verts_seq, self._contact_hand_labels_for_data(data)
        ).mean(dim=1)
        data.boundary_hand_position_at_move = hand[move_start].detach().clone()
        data.boundary_hand_velocity_at_move = (
            hand[move_start] - hand[move_start - 1]
        ).detach().clone()
        data.boundary_pose_residual_at_move = pred.human.pose_res[move_start].detach().clone()
        data.boundary_relative_anchor = (
            hand[move_start] - pred.obj.trans[move_start].detach()
        ).detach().clone()
        self.logger.info(
            "Captured Stage-B boundary state at t_move=%d | hand_step=%.6f",
            move_start,
            float(torch.linalg.norm(data.boundary_hand_velocity_at_move)),
        )

    @staticmethod
    def _contact_hand_labels_for_data(data):
        if data.contact_hand == "left":
            return ["L_Hand"]
        if data.contact_hand == "right":
            return ["R_Hand"]
        if data.contact_hand == "both":
            return ["L_Hand", "R_Hand"]
        raise ValueError(f"invalid contact hand {data.contact_hand!r}")

    @torch.no_grad()
    def initialize_human_approach_direction(self, data, gravity_axis="z"):
        """Freeze a ground direction from the optimized static object position."""
        pred = self.forward(data, self.params)
        if data.object_motion_state is None:
            frame = int(data.contact_frame)
            object_center = pred.obj.trans[frame]
            human_root = pred.human.trans[frame]
        else:
            frame = int(data.object_motion_state.move_start_frame)
            if frame < 1:
                raise ValueError("t_move must be >= 1 for pre-motion human approach")
            object_center = pred.obj.trans[:frame].median(dim=0).values
            human_root = pred.human.trans[frame - 1]
        direction = ground_approach_direction(
            object_center,
            human_root,
            gravity_axis=gravity_axis,
        )
        self._human_approach_direction = direction.detach()
        self.logger.info(
            "Human approach direction near motion frame %d: [%.6f, %.6f, %.6f]",
            frame,
            *self._human_approach_direction.tolist(),
        )
        return self._human_approach_direction

    @staticmethod
    def _human_pose_joint_indices(scope, num_body_joints):
        groups = {
            "lower_body": {1, 2, 4, 5, 7, 8, 10, 11},
            "arms": {13, 14, 16, 17, 18, 19, 20, 21},
            "upper_body_and_arms": {0, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21},
        }
        if scope == "full_body":
            allowed = set(range(num_body_joints))
        elif scope == "lower_body_and_arms":
            allowed = groups["lower_body"] | groups["arms"]
        elif scope in groups:
            allowed = groups[scope]
        else:
            raise ValueError(
                f"Invalid human_pose_res joint_scope={scope!r}; expected arms, lower_body, "
                "lower_body_and_arms, upper_body_and_arms, or full_body"
            )
        return sorted(index for index in allowed if index < num_body_joints)

    @torch.no_grad()
    def initialize_postcontact_pose_residuals(self, data, joint_scope="upper_body_and_arms"):
        """Seed every moving frame from the optimized motion-onset residual.

        Stage B only updates the approach interval through ``t_move``. Without
        this propagation, Stage C starts frame ``t_move + 1`` from the original
        HMR pose and creates a one-frame grasp discontinuity before its small
        learning rate can react.
        """
        if data.object_motion_state is None:
            raise ValueError("Post-contact residual initialization requires motion state")
        move_start = int(data.object_motion_state.move_start_frame)
        if not 0 <= move_start < data.frame_num - 1:
            raise ValueError("t_move must leave at least one post-motion frame")
        indices = self._human_pose_joint_indices(joint_scope, self.num_body_joints)
        pose_res = self.params.human_pose_res
        anchor = pose_res[move_start, indices].detach().clone()
        pose_res[move_start + 1 :, indices] = anchor.unsqueeze(0)
        self.logger.info(
            "Initialized continuous post-contact pose residuals | t_move=%d | "
            "frames=%d:%d | scope=%s | joints=%s",
            move_start,
            move_start + 1,
            data.frame_num - 1,
            joint_scope,
            indices,
        )

    def optimize_main(self, data, opt_config):
        """Run optimization for a single stage."""
        pose_opt_cfg = opt_config.get("opt_vars", {}).get("human_pose_res", {})
        if "arm_only" in opt_config or "arm_only" in pose_opt_cfg:
            raise ValueError(
                "arm_only is ambiguous and prohibited; use human_pose_res.joint_scope"
            )
        optimizer, opt_params = self.init_opt(data, self.params, opt_config)
        if len(opt_params.keys()) == 0:
            self.logger.warning("No optimization parameters found. Skipping optimization...")
            return

        opt_niter = opt_config["niter"]
        loss_cfg = opt_config["loss_cfg"]
        obj_depth_reference = self._obj_depth_stage_reference(opt_config)
        restore_best_state = bool(opt_config.get("restore_best_state", False))
        best_total = float("inf")
        best_state = None

        for cur_iter in range(opt_niter):
            for value in vars(self.params).values():
                if isinstance(value, torch.Tensor) and value.grad is not None:
                    value.grad = None
            optimizer.zero_grad(set_to_none=True)
            pred = self.forward(data, self.params)
            loss, loss_dict = self.loss_computer.compute_loss(data, pred, loss_cfg)
            if restore_best_state and float(loss.detach()) < best_total:
                best_total = float(loss.detach())
                best_state = {
                    name: value.detach().clone()
                    for name, value in vars(self.params).items()
                    if isinstance(value, torch.Tensor)
                }
            grad_log_interval = int(opt_config.get("gradient_log_interval", 25))
            should_log_grad = cur_iter in {0, opt_niter - 1} or (
                grad_log_interval > 0 and cur_iter % grad_log_interval == 0
            )
            if should_log_grad and "obj_depth_res" in opt_params:
                for loss_name, weighted_term in self.loss_computer.last_weighted_terms.items():
                    if not isinstance(weighted_term, torch.Tensor) or not weighted_term.requires_grad:
                        loss_dict[f"{loss_name}_grad_obj_depth_res"] = 0.0
                        continue
                    grad = torch.autograd.grad(
                        weighted_term,
                        self.params.obj_depth_res,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    loss_dict[f"{loss_name}_grad_obj_depth_res"] = (
                        0.0 if grad is None else float(torch.linalg.norm(grad).detach())
                    )
            if should_log_grad and self.params.obj_depth_res is not None:
                contact_grad = 0.0
                for loss_name, weighted_term in self.loss_computer.last_weighted_terms.items():
                    if loss_name not in {"contact_anchor", "approach_monotonic", "postcontact_relative"}:
                        continue
                    grad = torch.autograd.grad(
                        weighted_term,
                        self.params.obj_depth_res,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    if grad is not None:
                        contact_grad = max(contact_grad, float(torch.linalg.norm(grad).detach()))
                loss_dict["contact_or_grasp_grad_obj_depth_res"] = contact_grad
            loss.backward()
            self._apply_stage_gradient_masks(data, opt_config)
            self._apply_obj_depth_gradient_constraints(opt_config, data=data)
            if "obj_depth_res" in opt_params and self.params.obj_depth_res.grad is not None:
                loss_dict["total_grad_obj_depth_res"] = float(
                    torch.linalg.norm(self.params.obj_depth_res.grad).detach()
                )
            if (
                self.params.log_lift4d_depth_scale is not None
                and self.params.log_lift4d_depth_scale.grad is not None
            ):
                loss_dict["total_grad_log_lift4d_depth_scale"] = float(
                    torch.abs(self.params.log_lift4d_depth_scale.grad).detach()
                )
            optimizer.step()
            self._project_obj_depth_stage_constraints(
                opt_config, obj_depth_reference
            )
            if self.params.human_approach_distance is not None:
                with torch.no_grad():
                    self.params.human_approach_distance.clamp_(
                        0.0, float(self.cfg.get("max_human_approach_distance", 0.35))
                    )
                loss_dict["human_approach_distance"] = float(
                    self.params.human_approach_distance.detach()
                )
            if obj_depth_reference is not None:
                loss_dict["obj_depth_stage_max_delta"] = float(
                    (self.params.obj_depth_res.detach() - obj_depth_reference)
                    .abs()
                    .max()
                )

            self._maybe_save_motion_progress(cur_iter, pred, opt_config)

            self.write_logs(cur_iter, loss_dict, opt_config)
        if restore_best_state and best_state is not None:
            with torch.no_grad():
                for name, value in best_state.items():
                    current = getattr(self.params, name, None)
                    if isinstance(current, torch.Tensor):
                        current.copy_(value)
            self.logger.info("Restored Stage A best state with total loss %.6f", best_total)
        self.logger.info(
            f"Human pelvis after optimization stage: {pred.human.body_joints_seq[0, 0, :]}"
        )

    def _obj_depth_stage_reference(self, opt_config):
        depth_cfg = opt_config.get("opt_vars", {}).get("obj_depth_res")
        if depth_cfg is None or self.params.obj_depth_res is None:
            return None
        max_delta = depth_cfg.get("max_delta")
        if max_delta is not None and (
            not np.isfinite(max_delta) or float(max_delta) <= 0
        ):
            raise ValueError(f"obj_depth_res.max_delta must be positive, got {max_delta}")
        return self.params.obj_depth_res.detach().clone()

    def _apply_obj_depth_gradient_constraints(self, opt_config, data=None):
        depth_cfg = opt_config.get("opt_vars", {}).get("obj_depth_res")
        depth_res = self.params.obj_depth_res
        if depth_cfg is None or depth_res is None or depth_res.grad is None:
            return
        if data is not None and data.object_motion_state is not None:
            static = torch.as_tensor(
                data.object_motion_state.static,
                dtype=torch.bool,
                device=depth_res.grad.device,
            )
            if static.shape != depth_res.grad.shape:
                raise ValueError("Static motion-state mask does not match obj_depth_res")
            depth_res.grad[static] = 0.0
        if depth_cfg.get("freeze_anchor", False):
            depth_res.grad[0] = 0.0

    @torch.no_grad()
    def _project_obj_depth_stage_constraints(self, opt_config, reference):
        depth_cfg = opt_config.get("opt_vars", {}).get("obj_depth_res")
        depth_res = self.params.obj_depth_res
        if depth_cfg is None or depth_res is None or reference is None:
            return
        max_delta = depth_cfg.get("max_delta")
        if max_delta is not None:
            limit = float(max_delta)
            depth_res.copy_(
                torch.maximum(
                    torch.minimum(depth_res, reference + limit), reference - limit
                )
            )
        if depth_cfg.get("freeze_anchor", False):
            depth_res[0].copy_(reference[0])

    def _apply_stage_gradient_masks(self, data, opt_config):
        pose_cfg = opt_config.get("opt_vars", {}).get("human_pose_res")
        if pose_cfg is None or self.params.human_pose_res.grad is None:
            return
        scope = pose_cfg.get("joint_scope", "full_body")
        allowed = self._human_pose_joint_indices(scope, self.num_body_joints)
        joint_mask = torch.zeros(
            self.num_body_joints,
            dtype=self.params.human_pose_res.grad.dtype,
            device=self.params.human_pose_res.grad.device,
        )
        for index in allowed:
            joint_mask[index] = 1.0
        frame_mask = torch.zeros(
            data.frame_num,
            dtype=joint_mask.dtype,
            device=joint_mask.device,
        )
        if data.object_motion_state is None:
            motion_frame = int(data.contact_frame)
            start = max(0, motion_frame - int(data.approach_window))
            end = min(data.frame_num, motion_frame + int(pose_cfg.get("frame_radius", 2)) + 1)
        else:
            motion_frame = int(data.object_motion_state.move_start_frame)
            stage = str(opt_config.get("stage", ""))
            if "stage_3b" in stage:
                overlap = int(opt_config.get("overlap_frames", 5))
                start = max(0, motion_frame - int(data.approach_window) - overlap)
                end = min(data.frame_num, motion_frame + 1)
            elif "stage_3c" in stage:
                overlap = int(opt_config.get("overlap_frames", 5))
                start = max(0, motion_frame - overlap)
                end = data.frame_num
            else:
                start = max(0, motion_frame - int(data.approach_window))
                end = data.frame_num
        frame_mask[start:end] = 1.0
        self.params.human_pose_res.grad.mul_(
            frame_mask[:, None, None] * joint_mask[None, :, None]
        )

    def _maybe_save_motion_progress(self, cur_iter, pred, opt_config):
        if not opt_config.get("save_motion_progress", False):
            return
        stage = opt_config.get("stage", "stage")
        if "obj" not in stage:
            return
        interval = int(opt_config.get("motion_progress_interval", 50))
        niter = int(opt_config.get("niter", 0))
        if cur_iter not in {0, niter - 1} and (interval <= 0 or cur_iter % interval != 0):
            return
        out_dir = os.path.join(self.log_dir, "lift4d_motion_progress")
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, f"{stage}_iter_{cur_iter:06d}.npz"),
            obj_R=tensor_to_numpy(pred.obj.R),
            obj_t=tensor_to_numpy(pred.obj.trans),
            iteration=np.asarray(cur_iter, dtype=np.int64),
            stage=np.asarray(stage),
        )

    def optimize(self, data):
        print("Starting HOI optimization...")
        # Validate that data has been initialized
        if not hasattr(self, "obj_mesh") or self.obj_mesh is None:
            raise ValueError("Data not initialized. Call init_data() first.")
        if self.cfg.get("use_lift4d_depth_prior", False):
            required_losses = {
                "lift4d_depth",
                "lift4d_velocity",
                "fp_depth_anchor",
                "obj_depth_smoothness",
            }
            enabled_losses = set()
            for stage_specs in self.opt_stage_specs.values():
                for name, loss_cfg in stage_specs.get("loss_cfg", {}).items():
                    if loss_cfg.get("enabled", True) and float(loss_cfg.get("weight", 0.0)) > 0:
                        enabled_losses.add(name)
            missing = sorted(required_losses - enabled_losses)
            if missing:
                raise ValueError(
                    "use_lift4d_depth_prior=true requires explicit positive Stage-3 loss configs; "
                    f"missing enabled losses: {missing}"
                )

        if self.enable_vis:
            # Keep formal, headless optimization independent of optional
            # visualization and dynamic-camera modules.
            from grail.optimization.visualizer import HOIVisualizer

            self.visualizer = HOIVisualizer(
                device=self.device,
                human_model=self.human_model,
                cameras=self.cameras,
                image_list=self.image_list,
                video_fps=self.video_fps,
                log_dir=self.log_dir,
                obj_path=self.obj_path,
            )
            if self.vis_cfg.get("vis_html", False):
                self.visualizer.sp_visualizer = ScenepicVisualizer()
            self.visualizer.init_vis_meshes(data)

        pre_eval_pass, failed_frame = pre_eval(
            data,
            self.cameras,
            self.pre_eval_cfg,
            self.min_frames_threshold,
            self.device,
            self.logger,
        )
        if failed_frame is not None:
            truncate_data(data, failed_frame, self.logger)
            self.image_list = data.images_path
            self.depth_list = data.depth_maps

        # Initialize params after pre_eval (in case data was truncated)
        self.init_params(data)

        # Create loss computer after init_params (needs self.num_body_joints)
        self.loss_computer = LossComputer(
            cameras=self.cameras,
            human_model=self.human_model,
            device=self.device,
            get_contact_labels_for_frame_fn=self.get_contact_labels_for_frame,
            num_body_joints=self.num_body_joints,
            logger=self.logger,
        )

        if self.enable_vis:
            if self.vis_cfg.get("vis_init", False):
                pred = self.forward(data, self.params)
                hoi_data = self.get_optimized_data(data, pred, to_numpy=False)
                self.visualizer.visualize(data, pred, hoi_data, "init", self.vis_cfg)

        if pre_eval_pass:
            for stage, stage_specs in self.opt_stage_specs.items():
                stage_specs["stage"] = stage
                self.optimize_main(data, stage_specs)

                if self.enable_vis:
                    pred = self.forward(data, self.params)
                    hoi_data = self.get_optimized_data(data, pred, to_numpy=False)
                    self.visualizer.visualize(data, pred, hoi_data, stage, self.vis_cfg)
        else:
            print("Pre-evaluation failed. Skipping optimization...")

        pred = self.forward(data, self.params)
        optimized_data = self.get_optimized_data(data, pred)
        eval_data = self.eval(data, pred, self.eval_cfg)
        optimized_data["eval_data"] = eval_data

        return optimized_data

    @torch.no_grad()
    def eval(self, data, pred, eval_cfg):
        eval_data = {}
        _, eval_data = self.loss_computer.compute_loss(data, pred, eval_cfg)
        eval_data = tensor_to_numpy(eval_data)
        return eval_data

    def get_optimized_data(self, data, pred=None, to_numpy=True, smooth=True):
        if pred is None:
            pred = self.forward(data, self.params)

        def smooth_results(human_data, obj_R, obj_t):
            """Apply Savitzky-Golay smoothing to human motion and object trajectory."""
            self.logger.info("Applying smoothing to human motion...")
            frame_num = human_data["poses"].shape[0]

            human_data["poses"] = smooth_axis_angle_sequence(
                human_data["poses"].reshape(frame_num, -1, 3),
                window_length=11,
                polyorder=2,
            ).reshape(frame_num, -1)
            human_data["trans"] = smooth_pose_sequence(
                human_data["trans"], window_length=11, polyorder=2
            )
            for hand_key in ("left_hand_pose", "right_hand_pose"):
                if hand_key in human_data:
                    human_data[hand_key] = smooth_axis_angle_sequence(
                        human_data[hand_key].reshape(frame_num, -1, 3),
                        window_length=11,
                        polyorder=2,
                    ).reshape(frame_num, -1)

            self.logger.info("Applying smoothing to object trajectory...")
            obj_R = axis_angle_to_matrix(
                smooth_axis_angle_sequence(
                    matrix_to_axis_angle(obj_R).unsqueeze(1), window_length=11, polyorder=2
                ).squeeze(1)
            )
            if not self.cfg.get("use_lift4d_depth_prior", False):
                obj_t = smooth_pose_sequence(obj_t, window_length=11, polyorder=2)
            else:
                self.logger.info("Skipping world-space object translation smoothing to preserve FP rays")
            return human_data, obj_R, obj_t

        human_data = pred.human.motion_data
        obj_R = pred.obj.R
        obj_t = pred.obj.trans
        obj_t_cam = pred.obj.trans_cam
        obj_z_cam = pred.obj.z_cam

        if smooth:
            human_data, obj_R, obj_t = smooth_results(human_data, obj_R, obj_t)
        if data.object_motion_state is not None:
            static = np.asarray(data.object_motion_state.static, dtype=bool)
            with torch.no_grad():
                for frame in range(1, data.frame_num):
                    if static[frame]:
                        obj_R[frame].copy_(obj_R[frame - 1])
                        obj_t[frame].copy_(obj_t[frame - 1])
                        obj_t_cam[frame].copy_(obj_t_cam[frame - 1])
                        obj_z_cam[frame].copy_(obj_z_cam[frame - 1])
        obj_R_cam = torch.matmul(
            world_to_camera_matrix(self.opencv_cam_R, self.opencv_cam_t)[:3, :3].unsqueeze(0),
            obj_R,
        )

        optimized_data = {
            "human_data": tensor_to_numpy(human_data) if to_numpy else human_data,
            "obj_data": (
                tensor_to_numpy(
                    {
                        "obj_R": obj_R,
                        "obj_R_cam": obj_R_cam,
                        "obj_t": obj_t,
                        "obj_t_cam": obj_t_cam,
                        "obj_z_cam": obj_z_cam,
                        "obj_depth_res": self.params.obj_depth_res,
                        "obj_scale": data.obj.scale,
                    }
                )
                if to_numpy
                else {
                    "obj_R": obj_R,
                    "obj_R_cam": obj_R_cam,
                    "obj_t": obj_t,
                    "obj_t_cam": obj_t_cam,
                    "obj_z_cam": obj_z_cam,
                    "obj_depth_res": self.params.obj_depth_res,
                    "obj_scale": data.obj.scale,
                }
            ),
            "meta": {
                "inter_start_idx": data.inter_start_idx,
                "inter_end_idx": data.inter_end_idx,
                "optimization_config": dict(self.cfg),
                "contact_frame": data.contact_frame,
                "contact_hint": data.contact_hint,
                "contact_hint_source": data.contact_hint_source,
                "contact_window_start": data.contact_window_start,
                "contact_window_end": data.contact_window_end,
                "selected_contact_frame": data.selected_contact_frame,
                "contact_hand": data.contact_hand,
                "approach_window": data.approach_window,
                "human_approach_distance": float(pred.human.approach_distance.detach()),
                "human_approach_ramp": (
                    tensor_to_numpy(pred.human.approach_ramp)
                    if to_numpy
                    else pred.human.approach_ramp
                ),
                "human_approach_offset": (
                    tensor_to_numpy(pred.human.approach_offset)
                    if to_numpy
                    else pred.human.approach_offset
                ),
            },
        }

        if data.lift4d_depth is not None:
            K = data.lift4d_depth.camera_intrinsics
            fp_z = data.obj.poses_cam[:, 2, 3]
            _, expected_ray_cam, expected_z = self._freeze_static_object_pose_inputs(
                data,
                data.obj.poses,
                data.obj.fp_ray_cam,
                fp_z,
            )
            expected_trans_cam = ray_depth_translation(expected_ray_cam, expected_z)

            def project(trans):
                return torch.stack(
                    [
                        K[:, 0, 0] * trans[:, 0] / trans[:, 2] + K[:, 0, 2],
                        K[:, 1, 1] * trans[:, 1] / trans[:, 2] + K[:, 1, 2],
                    ],
                    dim=1,
                )

            pixel_error = torch.linalg.norm(
                project(obj_t_cam) - project(expected_trans_cam), dim=1
            )
            pixel_stats = {
                "mean": float(pixel_error.mean().detach()),
                "median": float(pixel_error.median().detach()),
                "max": float(pixel_error.max().detach()),
            }
            self.logger.info(
                "Final projection pixel error | mean=%.8g | median=%.8g | max=%.8g",
                pixel_stats["mean"],
                pixel_stats["median"],
                pixel_stats["max"],
            )
            if pixel_stats["max"] > 1e-3:
                raise ValueError(
                    "Optimized object translation left the effective FoundationPose image ray: "
                    f"max pixel error={pixel_stats['max']:.8g}"
                )
            static_pose_stats = None
            if data.object_motion_state is not None:
                move_start = int(data.object_motion_state.move_start_frame)
                static_t = pred.obj.trans[:move_start]
                static_R = pred.obj.R[:move_start]
                translation_deviation = torch.linalg.norm(static_t - static_t[0], dim=1)
                rotation_matrix_deviation = torch.amax(
                    torch.abs(static_R - static_R[0]), dim=(1, 2)
                )
                max_translation_deviation = float(translation_deviation.max().detach())
                max_rotation_matrix_deviation = float(
                    rotation_matrix_deviation.max().detach()
                )
                static_pose_stats = {
                    "frame_start": 0,
                    "frame_end_exclusive": move_start,
                    "max_translation_deviation_m": max_translation_deviation,
                    "max_rotation_matrix_deviation": max_rotation_matrix_deviation,
                }
                self.logger.info(
                    "Static object hard-freeze | frames=0:%d | max translation "
                    "deviation=%.8g m | max rotation-matrix deviation=%.8g",
                    move_start,
                    max_translation_deviation,
                    max_rotation_matrix_deviation,
                )
                if max_translation_deviation > 1e-6 or max_rotation_matrix_deviation > 1e-6:
                    raise ValueError(
                        "Static object pose changed despite hard freeze: "
                        f"translation={max_translation_deviation:.8g} m, "
                        f"rotation_matrix={max_rotation_matrix_deviation:.8g}"
                    )
            optimized_data["meta"]["lift4d_depth"] = {
                "source_path": data.lift4d_depth.source_path,
                "camera_convention": data.lift4d_depth.camera_convention,
                "stable_point_ids": (
                    tensor_to_numpy(data.lift4d_depth.stable_point_ids)
                    if to_numpy
                    else data.lift4d_depth.stable_point_ids
                ),
                "valid_point_count": (
                    tensor_to_numpy(data.lift4d_depth.valid_point_count)
                    if to_numpy
                    else data.lift4d_depth.valid_point_count
                ),
                "frame_indices": (
                    tensor_to_numpy(data.lift4d_depth.frame_indices)
                    if to_numpy
                    else data.lift4d_depth.frame_indices
                ),
                "prior_used": (
                    tensor_to_numpy(data.lift4d_depth.prior_used)
                    if to_numpy
                    else data.lift4d_depth.prior_used
                ),
                "lift4d_z_raw": (
                    tensor_to_numpy(data.lift4d_depth.z_raw)
                    if to_numpy
                    else data.lift4d_depth.z_raw
                ),
                "lift4d_z": (
                    tensor_to_numpy(data.lift4d_depth.z)
                    if to_numpy
                    else data.lift4d_depth.z
                ),
                "lift4d_z_target": (
                    tensor_to_numpy(data.lift4d_depth.z_target)
                    if to_numpy
                    else data.lift4d_depth.z_target
                ),
                "frame_weight": (
                    tensor_to_numpy(data.lift4d_depth.frame_weight)
                    if to_numpy
                    else data.lift4d_depth.frame_weight
                ),
                "depth_scale": float(pred.obj.depth_scale.detach()),
                "projection_pixel_error": pixel_stats,
                "static_pose_hard_freeze": static_pose_stats,
                "diagnostics": data.lift4d_depth.diagnostics,
            }
            if data.object_motion_state is not None:
                optimized_data["meta"]["object_motion_state"] = {
                    "move_start_frame": int(data.object_motion_state.move_start_frame),
                    "confidence": float(data.object_motion_state.confidence),
                    "static_z": float(data.object_motion_state.static_z),
                    "thresholds": dict(data.object_motion_state.thresholds),
                    "static": data.object_motion_state.static,
                    "moving_evidence": data.object_motion_state.moving_evidence,
                    "lift4d_center_speed": data.object_motion_state.lift4d_center_speed,
                    "mask_iou_drop": data.object_motion_state.mask_iou_drop,
                    "mask_centroid_displacement_px": data.object_motion_state.mask_centroid_displacement_px,
                    "mask_area_change_ratio": data.object_motion_state.mask_area_change_ratio,
                    "motion_score_3d": data.object_motion_state.motion_score_3d,
                    "motion_score_mask": data.object_motion_state.motion_score_mask,
                    "motion_score": data.object_motion_state.motion_score,
                    "moving": data.object_motion_state.moving,
                }

        if data.lift4d_motion is not None:
            optimized_data["meta"]["lift4d_motion"] = {
                "source_path": data.lift4d_motion.source_path,
                "anchor_frame": data.lift4d_motion.anchor_frame,
                "translation_scale": data.lift4d_motion.translation_scale,
                "valid_frames": (
                    tensor_to_numpy(data.lift4d_motion.motion_valid)
                    if to_numpy
                    else data.lift4d_motion.motion_valid
                ),
                "motion_confidence": (
                    tensor_to_numpy(data.lift4d_motion.motion_confidence)
                    if to_numpy
                    else data.lift4d_motion.motion_confidence
                ),
                "rigid_fit_rmse": (
                    tensor_to_numpy(data.lift4d_motion.rigid_fit_rmse)
                    if to_numpy and data.lift4d_motion.rigid_fit_rmse is not None
                    else data.lift4d_motion.rigid_fit_rmse
                ),
                "diagnostics": data.lift4d_motion.diagnostics,
            }

        if data.static_objects is not None:
            optimized_data["scene_data"] = tensor_to_numpy(data.static_objects)

        return optimized_data

    def write_logs(self, cur_iter, loss_dict, opt_config):
        opt_niters = opt_config["niter"]
        loss_str = " | ".join([f"{x}: {y:.6g}" for x, y in loss_dict.items()])
        head_str = f'{self.cfg["exp_name"]} - {opt_config["stage"]}'
        info_str = f"{head_str} | {cur_iter:4d}/{opt_niters} | {loss_str}"

        self.logger.info(info_str)
