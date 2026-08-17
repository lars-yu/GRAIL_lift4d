"""Typed dataclasses for HOI optimization data and predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from grail.optimization.motion_state import ObjectMotionState


@dataclass
class HOIData:
    """Ground-truth scene data assembled by HOIOptimizer.init_data()."""

    @dataclass
    class Camera:
        pose: torch.Tensor  # (4, 4) camera-to-world
        frame_height: int
        frame_width: int
        focal_length: float

    @dataclass
    class Human:
        faces: torch.Tensor  # (F, 3)
        masks: list  # per-frame binary masks, len=L
        motion_data: dict  # body-model-specific (poses, trans, scale, betas, ...)
        motion_data_global_init: dict  # global-frame reference motion
        body_keypoints_seq: torch.Tensor  # (L, J_body, 3) with confidence
        hand_keypoints_seq: torch.Tensor  # (L, J_hand, 3) with confidence
        foot_contact_probs: torch.Tensor | None  # (L, 4) or None

    @dataclass
    class Object:
        scale: torch.Tensor  # (3,)
        verts: torch.Tensor  # (V, 3) canonical mesh vertices
        faces: torch.Tensor  # (F, 3)
        masks: list  # per-frame binary masks, len=L
        verts_seq: torch.Tensor  # (L, V, 3) per-frame transformed vertices
        poses: torch.Tensor  # (L, 4, 4) SE(3) poses
        poses_cam: torch.Tensor  # (L, 4, 4) FoundationPose T_C<-O in OpenCV camera
        fp_ray_cam: torch.Tensor  # (L, 3), z-normalized FoundationPose image rays
        verts_tracking_seq: torch.Tensor  # (L, V, 2) projected 2D vertices

    @dataclass
    class Lift4DMotion:
        object_poses: torch.Tensor  # (L, 4, 4) aligned FP-anchor + Lift4D relative prior
        motion_valid: torch.Tensor  # (L,) bool
        motion_confidence: torch.Tensor  # (L,)
        source_path: str
        anchor_frame: int
        translation_scale: float
        rigid_fit_rmse: torch.Tensor | None = None  # (L,)
        object_scales: torch.Tensor | None = None  # (L,) diagnostic only
        diagnostics: dict[str, Any] | None = None

    @dataclass
    class Lift4DDepth:
        frame_indices: torch.Tensor  # (L,), strictly equal to arange(L)
        prior_used: torch.Tensor  # (L,) bool, all true for formal supervision
        center_cam_raw: torch.Tensor  # (L, 3), robust per-frame center before temporal smoothing
        center_cam_detection: torch.Tensor  # (L, 3), median5 only; motion onset input
        center_cam: torch.Tensor  # (L, 3), filtered/smoothed OpenCV camera center
        z_raw: torch.Tensor  # (L,), unsmoothed robust center depth
        z: torch.Tensor  # (L,)
        z_target: torch.Tensor  # (L,), static-locked then relative Lift4D motion
        delta_z: torch.Tensor  # (L,), relative to frame 0
        frame_weight: torch.Tensor  # (L,), support-derived and normalized
        valid_point_count: torch.Tensor  # (L,)
        camera_intrinsics: torch.Tensor  # (L, 3, 3)
        stable_point_ids: torch.Tensor  # (S,)
        source_path: str
        camera_convention: str
        diagnostics: dict[str, Any]

    frame_num: int
    inter_start_idx: int
    inter_end_idx: int
    human: Human
    obj: Object
    camera: Camera
    images_path: list  # list of image file paths
    depth_maps: list  # list of per-frame depth tensors
    is_static_obj: bool
    obj_sdf: Any = None
    static_objects: dict | None = None
    lift4d_motion: Lift4DMotion | None = None
    lift4d_depth: Lift4DDepth | None = None
    object_motion_state: ObjectMotionState | None = None
    contact_frame: int | None = None  # compatibility alias for contact_hint
    contact_hint: int | None = None
    contact_hint_source: str = "inter_start"
    contact_window_start: int | None = None
    contact_window_end: int | None = None
    selected_contact_frame: int | None = None
    contact_soft_weight: torch.Tensor | None = None
    contact_hand: str = "right"
    approach_window: int = 30


@dataclass
class OptParams:
    """Optimization parameters (residuals applied on top of initial estimates)."""

    human_trans_global: torch.Tensor  # (3,) — global translation offset
    human_trans_res: torch.Tensor  # (L, 3) — per-frame translation residuals
    human_pose_res: torch.Tensor  # (L, J_body, 6) — body pose residuals in 6D
    hand_pose_res: torch.Tensor  # (L, J_hand, 6) — hand pose residuals in 6D
    obj_R_res: torch.Tensor  # (L, 6) — object rotation residuals in 6D
    obj_t_res: Optional[torch.Tensor]  # (L, 3) legacy world translation residuals
    obj_depth_res: Optional[torch.Tensor] = None  # (L,) residual added to FoundationPose camera-z
    human_approach_distance: Optional[torch.Tensor] = None  # scalar, projected to [0,max]
    obj_z_opt: Optional[torch.Tensor] = None  # deprecated absolute ray depth; compatibility only
    log_lift4d_depth_scale: Optional[torch.Tensor] = None  # scalar, exp() keeps scale positive


@dataclass
class HOIPrediction:
    """Predicted HOI state from HOIOptimizer.forward()."""

    @dataclass
    class Human:
        trans: torch.Tensor  # (L, 3)
        root_pose: torch.Tensor  # (L, 3)
        pose: torch.Tensor  # (L, J*3)
        verts_seq: torch.Tensor  # (L, V, 3)
        body_joints_seq: torch.Tensor  # (L, J_body, 3)
        body_keypoints_seq: torch.Tensor  # (L, J_body, 2)
        hand_joints_seq: torch.Tensor  # (L, J_hand, 3)
        hand_keypoints_seq: torch.Tensor  # (L, J_hand, 2)
        pose_res: torch.Tensor  # (L, J_body, 6)
        trans_res: torch.Tensor  # (L, 1, 3)
        approach_ramp: torch.Tensor  # (L,)
        approach_offset: torch.Tensor  # (L, 3)
        approach_distance: torch.Tensor  # scalar
        motion_data: dict  # body-model-specific, used by get_optimized_data

    @dataclass
    class Object:
        trans: torch.Tensor  # (L, 3)
        trans_cam: torch.Tensor  # (L, 3), OpenCV camera convention
        z_cam: torch.Tensor  # (L,)
        depth_scale: torch.Tensor  # positive scalar
        R: torch.Tensor  # (L, 3, 3)
        R_cam: torch.Tensor  # (L, 3, 3), OpenCV camera convention
        verts_seq: torch.Tensor  # (L, V, 3)

    human: Human
    obj: Object
