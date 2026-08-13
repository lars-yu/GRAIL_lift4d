import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from grail.adapters.wilor import infer_hand_pose
from grail.core.torch_utils import tensor_to, tensor_to_numpy
from grail.pose_est.utils import smooth_axis_angle_sequence, smooth_pose_sequence


def compute_global_rotation(pose_axis_anges, joint_idx):
    """
    calculating joints' global rotation for SMPL-X (22 body joints)
    Args:
        pose_axis_anges (np.array): SMPLX's local pose (22,3)
    Returns:
        np.array: (3, 3)
    """
    global_rotation = np.eye(3)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    while joint_idx != -1:
        joint_rotation = R.from_rotvec(pose_axis_anges[joint_idx]).as_matrix()
        global_rotation = joint_rotation @ global_rotation
        joint_idx = parents[joint_idx]
    return global_rotation


def fuse_smplx_mano_predictions(
    smplx_params,
    mano_params,
    left_hand_pose_relaxed,
    right_hand_pose_relaxed,
    body_keypoints_2d=None,  # (17, 3) vitpose for this frame, COCO-17 format
    wrist_dist_threshold=30.0,  # pixel distance threshold for wrist sanity check
    wrist_rot_threshold=1.5,  # radians (~86°); reject hand if local wrist rotation magnitude exceeds this
):
    M = np.diag([-1, 1, 1])  # Preparing for the left hand switch

    # Assuming that your data are stored in smplx_params and hamer_mano_params
    full_body_pose = torch.concatenate(
        (
            smplx_params["global_orient"],
            smplx_params["body_pose"].reshape(21, 3),
        ),
        dim=0,
    )

    hand_keypoints_2d = torch.zeros((32, 3))

    right_detected = False
    left_detected = False
    right_hand_idx = None
    left_hand_idx = None
    right_bbox_conf_min = 0.0
    left_bbox_conf_min = 0.0

    for i in range(len(mano_params)):
        if mano_params[i]["is_right"] > 0.5:
            right_detected = True

            bbox_conf = mano_params[i]["bbox_conf"]
            if bbox_conf > right_bbox_conf_min:
                right_bbox_conf_min = bbox_conf
                right_hand_idx = i
        else:
            left_detected = True
            bbox_conf = mano_params[i]["bbox_conf"]
            if bbox_conf > left_bbox_conf_min:
                left_bbox_conf_min = bbox_conf
                left_hand_idx = i

    # map from mano hand order to smplx hand order
    mano_to_smplx_hand_order = [
        0,  # wrist
        5,
        6,
        7,  # index
        9,
        10,
        11,  # middle
        17,
        18,
        19,  # pinky
        13,
        14,
        15,  # ring
        1,
        2,
        3,  # thumb
    ]

    # Wrist distance sanity check: reject WiLoR hand if its wrist is too far
    # from the body wrist keypoint (vitpose COCO-17: left_wrist=9, right_wrist=10)
    if body_keypoints_2d is not None and wrist_dist_threshold > 0:
        body_kpts = body_keypoints_2d
        if not isinstance(body_kpts, torch.Tensor):
            body_kpts = torch.tensor(body_kpts, dtype=torch.float32)

        # Check left hand
        if left_detected:
            body_left_wrist = body_kpts[9]  # (3,) = (x, y, conf)
            if body_left_wrist[2] > 0.3:  # only check when body wrist is reliable
                left_kpts = tensor_to(
                    mano_params[left_hand_idx]["wilor_preds"]["pred_keypoints_2d"]
                ).reshape(-1, 2)
                left_wrist_kpt = left_kpts[0]  # MANO wrist = index 0
                dist = torch.norm(left_wrist_kpt - body_left_wrist[:2]).item()
                if dist > wrist_dist_threshold:
                    left_detected = False

        # Check right hand
        if right_detected:
            body_right_wrist = body_kpts[10]  # (3,) = (x, y, conf)
            if body_right_wrist[2] > 0.3:  # only check when body wrist is reliable
                right_kpts = tensor_to(
                    mano_params[right_hand_idx]["wilor_preds"]["pred_keypoints_2d"]
                ).reshape(-1, 2)
                right_wrist_kpt = right_kpts[0]  # MANO wrist = index 0
                dist = torch.norm(right_wrist_kpt - body_right_wrist[:2]).item()
                if dist > wrist_dist_threshold:
                    right_detected = False

    if left_detected:
        left_mano_params = mano_params[left_hand_idx]["wilor_preds"]
        # hand_keypoints_2d["left_hand"] = tensor_to(
        #     left_mano_params["pred_keypoints_2d"]
        # ).reshape(-1, 2)

        left_hand_keypoints_2d = tensor_to(left_mano_params["pred_keypoints_2d"]).reshape(-1, 2)[
            mano_to_smplx_hand_order
        ]
        # add conf
        left_hand_keypoints_2d = torch.cat(
            [left_hand_keypoints_2d, torch.ones((left_hand_keypoints_2d.shape[0], 1))], dim=1
        )
        hand_keypoints_2d[:16] = left_hand_keypoints_2d

        left_elbow_global_rot = compute_global_rotation(full_body_pose, 18)  # left elbow IDX: 18

        left_wrist_global_rot = left_mano_params["global_orient"][0]
        # axis angle to matrix
        left_wrist_global_rot = R.from_rotvec(left_wrist_global_rot).as_matrix()
        # left_wrist_global_rot = M @ left_wrist_global_rot @ M
        left_wrist_pose = np.linalg.inv(left_elbow_global_rot) @ left_wrist_global_rot
        left_wrist_pose_vec = R.from_matrix(left_wrist_pose).as_rotvec()

        # Reject if local wrist rotation is too large (likely bad prediction)
        left_wrist_rot_mag = np.linalg.norm(left_wrist_pose_vec)
        if wrist_rot_threshold > 0 and left_wrist_rot_mag > wrist_rot_threshold:
            # Fall back to relaxed hand
            smplx_params["left_hand_pose"] = tensor_to(left_hand_pose_relaxed).reshape(1, 15, 3)
            hand_keypoints_2d[:16] = torch.zeros((16, 3))
        else:
            smplx_params["body_pose"][0, 57:60] = tensor_to(left_wrist_pose_vec).reshape(1, 3)
            smplx_params["left_hand_pose"] = tensor_to(left_mano_params["hand_pose"]).reshape(
                1, 15, 3
            )

    else:
        smplx_params["left_hand_pose"] = tensor_to(left_hand_pose_relaxed).reshape(1, 15, 3)
        hand_keypoints_2d[:16] = torch.zeros((16, 3))

    if right_detected:
        right_mano_params = mano_params[right_hand_idx]["wilor_preds"]

        right_hand_keypoints_2d = tensor_to(right_mano_params["pred_keypoints_2d"]).reshape(-1, 2)[
            mano_to_smplx_hand_order
        ]
        # add conf
        right_hand_keypoints_2d = torch.cat(
            [right_hand_keypoints_2d, torch.ones((right_hand_keypoints_2d.shape[0], 1))], dim=1
        )
        hand_keypoints_2d[16:] = right_hand_keypoints_2d

        right_elbow_global_rot = compute_global_rotation(full_body_pose, 19)  # left elbow IDX: 19

        right_wrist_global_rot = right_mano_params["global_orient"][0]
        right_wrist_global_rot = R.from_rotvec(right_wrist_global_rot).as_matrix()
        right_wrist_pose = np.linalg.inv(right_elbow_global_rot) @ right_wrist_global_rot
        right_wrist_pose_vec = R.from_matrix(right_wrist_pose).as_rotvec()

        # Reject if local wrist rotation is too large (likely bad prediction)
        right_wrist_rot_mag = np.linalg.norm(right_wrist_pose_vec)
        if wrist_rot_threshold > 0 and right_wrist_rot_mag > wrist_rot_threshold:
            # Fall back to relaxed hand
            smplx_params["right_hand_pose"] = tensor_to(right_hand_pose_relaxed).reshape(1, 15, 3)
            hand_keypoints_2d[16:] = torch.zeros((16, 3))
        else:
            smplx_params["body_pose"][0, 60:63] = tensor_to(right_wrist_pose_vec).reshape(1, 3)
            smplx_params["right_hand_pose"] = tensor_to(right_mano_params["hand_pose"]).reshape(
                1, 15, 3
            )

    else:
        smplx_params["right_hand_pose"] = tensor_to(right_hand_pose_relaxed).reshape(1, 15, 3)
        hand_keypoints_2d[16:] = torch.zeros((16, 3))

    return smplx_params, hand_keypoints_2d


def interp_smplx_mano_predictions(smplx_params):
    """
    Interpolate missing hand poses in GENMO predictions.

    Args:
        smplx_params (dict): Dictionary containing SMPLX parameters with keys:
            - 'left_hand_pose': tensor of shape (frame_num, 15, 3)
            - 'right_hand_pose': tensor of shape (frame_num, 15, 3)
            - 'body_pose': tensor of shape (frame_num, 21, 3)

    Returns:
        dict: Updated smplx_params with interpolated hand poses
    """
    import numpy as np
    import torch

    frame_num = smplx_params["body_pose"].shape[0]
    smplx_params["body_pose"] = smplx_params["body_pose"].reshape(frame_num, -1, 3)

    # Load relaxed hand poses as fallback
    data_path = os.path.join(os.path.dirname(__file__), "..", "constants", "smplx_handposes.npz")
    with np.load(data_path, allow_pickle=True) as data:
        hand_poses = data["hand_poses"].item()
        left_hand_pose_relaxed, right_hand_pose_relaxed = hand_poses["relaxed"]

    left_hand_pose_relaxed = tensor_to(left_hand_pose_relaxed).reshape(15, 3)
    right_hand_pose_relaxed = tensor_to(right_hand_pose_relaxed).reshape(15, 3)

    def is_relaxed_pose(hand_pose, relaxed_pose, tolerance=1e-5):
        """Check if hand pose is close to relaxed pose (indicating missing detection)"""
        if hand_pose is None:
            return True
        return torch.allclose(hand_pose, relaxed_pose, atol=tolerance)

    def interpolate_pose_sequence(poses, valid_mask):
        """Interpolate poses for invalid frames using linear interpolation"""
        poses_interp = poses.clone()

        for i in range(frame_num):
            if not valid_mask[i]:
                # Find previous and next valid frames
                prev_valid = None
                next_valid = None

                # Search backwards for previous valid frame
                for j in range(i - 1, -1, -1):
                    if valid_mask[j]:
                        prev_valid = j
                        break

                # Search forwards for next valid frame
                for j in range(i + 1, frame_num):
                    if valid_mask[j]:
                        next_valid = j
                        break

                # Interpolate based on available valid frames
                if prev_valid is not None and next_valid is not None:
                    # Linear interpolation between prev and next
                    alpha = (i - prev_valid) / (next_valid - prev_valid)
                    poses_interp[i] = (1 - alpha) * poses[prev_valid] + alpha * poses[next_valid]
                elif prev_valid is not None:
                    # Use previous valid frame
                    poses_interp[i] = poses[prev_valid]
                elif next_valid is not None:
                    # Use next valid frame
                    poses_interp[i] = poses[next_valid]
                # If no valid frames exist, keep original (likely relaxed pose)

        return poses_interp

    # Process left hand pose
    if "left_hand_pose" in smplx_params:
        left_hand_poses = smplx_params["left_hand_pose"]
        left_valid_mask = torch.zeros(frame_num, dtype=torch.bool)

        # Identify valid (non-relaxed) hand poses
        for i in range(frame_num):
            if not is_relaxed_pose(left_hand_poses[i], left_hand_pose_relaxed):
                left_valid_mask[i] = True

        # Interpolate invalid frames
        if left_valid_mask.any():  # Only interpolate if we have some valid poses
            smplx_params["left_hand_pose"] = interpolate_pose_sequence(
                left_hand_poses, left_valid_mask
            )

            # Also interpolate left wrist rotation (body_pose[57:60])
            left_wrist_poses = smplx_params["body_pose"][:, 19, :]  # wrist is joint 19
            smplx_params["body_pose"][:, 19, :] = interpolate_pose_sequence(
                left_wrist_poses, left_valid_mask
            )

    # Process right hand pose
    if "right_hand_pose" in smplx_params:
        right_hand_poses = smplx_params["right_hand_pose"]
        right_valid_mask = torch.zeros(frame_num, dtype=torch.bool)

        # Identify valid (non-relaxed) hand poses
        for i in range(frame_num):
            if not is_relaxed_pose(right_hand_poses[i], right_hand_pose_relaxed):
                right_valid_mask[i] = True

        # Interpolate invalid frames
        if right_valid_mask.any():  # Only interpolate if we have some valid poses
            smplx_params["right_hand_pose"] = interpolate_pose_sequence(
                right_hand_poses, right_valid_mask
            )

            # Also interpolate right wrist rotation (body_pose[60:63] -> joint 20)

            right_wrist_poses = smplx_params["body_pose"].reshape(frame_num, -1, 3)[
                :, 20, :
            ]  # wrist is joint 20
            smplx_params["body_pose"][:, 20, :] = interpolate_pose_sequence(
                right_wrist_poses, right_valid_mask
            )
            # body_pose = smplx_params['body_pose'].reshape(frame_num, -1, 3)
            # right_wrist_poses = body_pose[:, 20, :]
            # right_wrist_poses = interpolate_pose_sequence(
            #     right_wrist_poses, right_valid_mask
            # )
            # body_pose[:, 20, :] = right_wrist_poses
            # smplx_params['body_pose'] = body_pose.reshape(frame_num, -1)

    # Apply Savitzky-Golay smoothing to hand poses and wrist rotations
    # Use quaternion-based smoothing for rotations to avoid discontinuities
    if "left_hand_pose" in smplx_params and smplx_params["left_hand_pose"] is not None:
        smplx_params["left_hand_pose"] = smooth_axis_angle_sequence(smplx_params["left_hand_pose"])
        # Smooth left wrist rotation
        left_wrist_poses = smplx_params["body_pose"][:, 19, :]
        smplx_params["body_pose"][:, 19, :] = smooth_axis_angle_sequence(left_wrist_poses)

    # Smooth right hand pose
    if "right_hand_pose" in smplx_params and smplx_params["right_hand_pose"] is not None:
        smplx_params["right_hand_pose"] = smooth_axis_angle_sequence(
            smplx_params["right_hand_pose"]
        )
        # Smooth right wrist rotation
        right_wrist_poses = smplx_params["body_pose"][:, 20, :]
        smplx_params["body_pose"][:, 20, :] = smooth_axis_angle_sequence(right_wrist_poses)

    return smplx_params


def run_human_pose_est_smplx(video_path, cache_dir, smplx_model_path=None, is_static_cam=True):
    """
    Predict human pose using GENMO
    """
    from grail.adapters.gem_smpl import infer_human_pose as infer_human_pose_smplx

    smplx_pred = infer_human_pose_smplx(video_path, cache_dir, is_static_cam=is_static_cam)
    smplx_global = smplx_pred["smpl_params_global"]
    data_path = os.path.join(os.path.dirname(__file__), "..", "constants", "smplx_handposes.npz")
    with np.load(data_path, allow_pickle=True) as data:
        hand_poses = data["hand_poses"].item()
        left_hand_pose_relaxed, right_hand_pose_relaxed = hand_poses["relaxed"]
        hand_pose_relaxed = np.concatenate(
            (left_hand_pose_relaxed, right_hand_pose_relaxed)
        ).reshape(1, -1)

    """
    Predict hand pose using WiLoR
    """
    mano_preds = infer_hand_pose(video_path)

    """
    Fuse GENMO and WiLoR predictions
    """
    frame_num = smplx_global["body_pose"].shape[0]
    if frame_num != len(mano_preds):
        raise ValueError("Frame number mismatch between GENMO and WiLoR")

    # Create fused predictions with hand poses for each frame
    fused_smplx_global = {}
    fused_smplx_incam = {}

    # Initialize arrays to store fused results
    fused_smplx_global["body_pose"] = smplx_global["body_pose"].clone()
    fused_smplx_global["global_orient"] = smplx_global["global_orient"].clone()
    fused_smplx_global["transl"] = smplx_global["transl"].clone()
    fused_smplx_global["betas"] = smplx_global["betas"].clone()
    fused_smplx_global["left_hand_pose"] = torch.zeros((frame_num, 15, 3))
    fused_smplx_global["right_hand_pose"] = torch.zeros((frame_num, 15, 3))

    fused_smplx_incam = {}
    fused_smplx_incam["body_pose"] = smplx_pred["smpl_params_incam"]["body_pose"].clone()
    fused_smplx_incam["global_orient"] = smplx_pred["smpl_params_incam"]["global_orient"].clone()
    fused_smplx_incam["transl"] = smplx_pred["smpl_params_incam"]["transl"].clone()
    fused_smplx_incam["betas"] = smplx_pred["smpl_params_incam"]["betas"].clone()
    fused_smplx_incam["left_hand_pose"] = torch.zeros((frame_num, 15, 3))
    fused_smplx_incam["right_hand_pose"] = torch.zeros((frame_num, 15, 3))

    # Process each frame for fusion
    # hand_keypoints_2d = {
    #     "left_hand": [None] * frame_num,
    #     "right_hand": [None] * frame_num,
    # }
    hand_keypoints_2d = torch.zeros((frame_num, 32, 3))
    for i in range(frame_num):
        # Extract frame-specific predictions
        frame_global = {k: v[i : i + 1] for k, v in fused_smplx_global.items()}
        frame_incam = {k: v[i : i + 1] for k, v in fused_smplx_incam.items()}

        frame_mano_preds = mano_preds[i]

        # Fuse predictions for this frame
        frame_vitpose = smplx_pred["vitpose"][i] if smplx_pred.get("vitpose") is not None else None
        fused_frame_incam, hand_keypoints_2d_frame = fuse_smplx_mano_predictions(
            frame_incam,
            frame_mano_preds,
            left_hand_pose_relaxed,
            right_hand_pose_relaxed,
            body_keypoints_2d=frame_vitpose,
        )

        fused_smplx_incam["body_pose"][i : i + 1] = fused_frame_incam["body_pose"]
        fused_smplx_incam["left_hand_pose"][i : i + 1] = fused_frame_incam["left_hand_pose"]
        fused_smplx_incam["right_hand_pose"][i : i + 1] = fused_frame_incam["right_hand_pose"]

        fused_smplx_global["body_pose"][i : i + 1] = fused_frame_incam["body_pose"]
        fused_smplx_global["left_hand_pose"][i : i + 1] = fused_frame_incam["left_hand_pose"]
        fused_smplx_global["right_hand_pose"][i : i + 1] = fused_frame_incam["right_hand_pose"]

        hand_keypoints_2d[i : i + 1] = hand_keypoints_2d_frame

        # hand_keypoints_2d["left_hand"][i] = hand_keypoints_2d_frame["left_hand"]
        # hand_keypoints_2d["right_hand"][i] = hand_keypoints_2d_frame["right_hand"]

    # Interpolate missing hand poses in GENMO predictions
    fused_smplx_global = interp_smplx_mano_predictions(fused_smplx_global)
    fused_smplx_incam = interp_smplx_mano_predictions(fused_smplx_incam)

    # Process incam motion
    body_pose = fused_smplx_incam["body_pose"].reshape(-1, 63)
    betas = fused_smplx_incam["betas"].reshape(-1, 10)
    global_orient = fused_smplx_incam["global_orient"].reshape(-1, 3)
    trans = fused_smplx_incam["transl"].reshape(-1, 3)
    left_hand_pose = fused_smplx_incam["left_hand_pose"].reshape(-1, 45)
    right_hand_pose = fused_smplx_incam["right_hand_pose"].reshape(-1, 45)

    incam_poses = torch.zeros((frame_num, 165))
    incam_poses[:, :3] = global_orient
    incam_poses[:, 3 : 3 + 21 * 3] = body_pose
    incam_poses[:, 3 + 21 * 3 : 3 + 21 * 3 + 45] = left_hand_pose
    incam_poses[:, 3 + 21 * 3 + 45 : 3 + 21 * 3 + 45 + 45] = right_hand_pose

    incam_betas = betas[0, :10]
    incam_trans = trans

    # Get foot contact probs if available (from GENMO prediction)
    foot_contact_probs = smplx_pred.get("foot_contact_probs", None)
    if foot_contact_probs is not None:
        # Shape: (1, L, 4) -> (L, 4) for [L_Ankle, L_foot, R_Ankle, R_foot]
        foot_contact_probs = foot_contact_probs.squeeze(0)

    motion_incam = dict(
        poses=incam_poses,
        betas=incam_betas,
        trans=incam_trans,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        mocap_frame_rate=30,
        gender="neutral",
        vitpose=smplx_pred["vitpose"],
        hand_keypoints_2d=hand_keypoints_2d,
        foot_contact_probs=foot_contact_probs,  # (L, 4) or None
    )

    # Process global motion
    body_pose = fused_smplx_global["body_pose"].reshape(-1, 63)
    betas = fused_smplx_global["betas"].reshape(-1, 10)
    global_orient = fused_smplx_global["global_orient"].reshape(-1, 3)
    trans = fused_smplx_global["transl"].reshape(-1, 3)
    left_hand_pose = fused_smplx_global["left_hand_pose"].reshape(-1, 45)
    right_hand_pose = fused_smplx_global["right_hand_pose"].reshape(-1, 45)

    global_poses = torch.zeros((frame_num, 165))
    global_poses[:, :3] = global_orient
    global_poses[:, 3 : 3 + 21 * 3] = body_pose
    global_poses[:, 3 + 21 * 3 : 3 + 21 * 3 + 45] = left_hand_pose
    global_poses[:, 3 + 21 * 3 + 45 : 3 + 21 * 3 + 45 + 45] = right_hand_pose

    global_betas = betas[0, :10]
    global_trans = trans

    motion_global = dict(
        poses=global_poses,
        betas=global_betas,
        trans=global_trans,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        mocap_frame_rate=30,
        gender="neutral",
    )

    motion_incam["model"] = "smplx"
    motion_global["model"] = "smplx"

    # Compute predicted body height from HMR-predicted betas (using raw SMPLX).
    # Stored upstream so downstream consumers (e.g. G1SmplxHumanModel, which
    # bakes zeroed shapedirs into v_template) can't recompute it.
    if smplx_model_path is None:
        raise ValueError(
            "smplx_model_path is required to compute predicted_body_height. "
            "Pass it from the pipeline config (human_model.smplx_model_path)."
        )
    from grail.models.smplx_model import get_tpose_human_height, setup_smplx_model

    smplx_model_dir = os.path.dirname(os.path.dirname(smplx_model_path))
    smplx_model = setup_smplx_model(
        model_path=smplx_model_dir, flat_hand_mean=True, device="cpu"
    )
    predicted_body_height = float(
        get_tpose_human_height(smplx_model, betas=incam_betas.cpu(), device="cpu")
    )
    motion_incam["predicted_body_height"] = predicted_body_height
    motion_global["predicted_body_height"] = predicted_body_height

    motion = dict(motion_global=motion_global, motion_incam=motion_incam)

    return motion


def run_human_pose_est_soma(video_path, cache_dir, soma_model_path=None, is_static_cam=True):
    """Predict human pose using GENMO NOVA model (SOMA body).

    Returns dict with motion_global and motion_incam in SOMA format.
    """
    if soma_model_path is None:
        raise ValueError(
            "soma_model_path is required to compute predicted_body_height. "
            "Pass it from the pipeline config (human_model.soma_model_path)."
        )

    from grail.adapters.gem_soma import infer_human_pose as infer_human_pose_soma

    soma_pred = infer_human_pose_soma(video_path, cache_dir, is_static_cam=is_static_cam)

    soma_global = soma_pred["smpl_params_global"]
    soma_incam = soma_pred["smpl_params_incam"]
    frame_num = soma_global["body_pose"].shape[0]

    def _build_params(src):
        params = {
            k: src[k].clone()
            for k in (
                "body_pose",
                "global_orient",
                "transl",
                "identity_coeffs",
                "scale_params",
                "left_hand_pose",
                "right_hand_pose",
            )
            if k in src
        }
        if params["body_pose"].dim() == 2:
            params["body_pose"] = params["body_pose"].reshape(frame_num, 76, 3)
        return params

    fused_soma_global = _build_params(soma_global)
    fused_soma_incam = _build_params(soma_incam)
    hand_keypoints_2d = torch.zeros((frame_num, 32, 3))

    # Get vitpose from prediction
    vitpose = soma_pred.get("vitpose", None)

    # Get foot contact probs if available
    foot_contact_probs = None
    if "net_outputs" in soma_pred and "model_output" in soma_pred["net_outputs"]:
        if "static_conf_logits" in soma_pred["net_outputs"]["model_output"]:
            static_conf_logits = soma_pred["net_outputs"]["model_output"]["static_conf_logits"]
            foot_contact_probs = torch.sigmoid(static_conf_logits[:, :, :4]).squeeze(0)

    # Build motion_incam dict with NOVA format
    # SOMA poses: global_orient (3) + body_pose (76×3 = 228) = 231 dims
    body_pose_incam = fused_soma_incam["body_pose"].reshape(frame_num, -1)  # (L, 228)
    global_orient_incam = fused_soma_incam["global_orient"].reshape(frame_num, 3)  # (L, 3)

    # Concatenate into full pose vector: global_orient + body_pose
    incam_poses = torch.cat([global_orient_incam, body_pose_incam], dim=1)  # (L, 231)

    # Get shape params (use mean across frames)
    identity_coeffs_incam = fused_soma_incam["identity_coeffs"]
    if identity_coeffs_incam.dim() == 2:
        identity_coeffs_incam = identity_coeffs_incam[0]  # Take first frame
    scale_params_incam = fused_soma_incam["scale_params"]
    if scale_params_incam.dim() == 2:
        scale_params_incam = scale_params_incam[0]  # Take first frame

    zero_hand = torch.zeros((frame_num, 72))
    motion_incam = dict(
        poses=incam_poses,  # (L, 231) = global_orient (3) + body_pose (228)
        identity_coeffs=identity_coeffs_incam,  # (45,)
        scale_params=scale_params_incam,  # (75,)
        trans=fused_soma_incam["transl"].reshape(frame_num, 3),  # (L, 3)
        left_hand_pose=fused_soma_incam.get("left_hand_pose", zero_hand).reshape(frame_num, -1),
        right_hand_pose=fused_soma_incam.get("right_hand_pose", zero_hand).reshape(frame_num, -1),
        mocap_frame_rate=30,
        gender="neutral",
        model_type="soma",  # Indicate NOVA format
        vitpose=vitpose,
        hand_keypoints_2d=hand_keypoints_2d,
        foot_contact_probs=foot_contact_probs,
        K_fullimg=soma_pred.get("K_fullimg", None),
    )

    # Build motion_global dict with NOVA format
    body_pose_global = fused_soma_global["body_pose"].reshape(frame_num, -1)  # (L, 228)
    global_orient_global = fused_soma_global["global_orient"].reshape(frame_num, 3)  # (L, 3)

    global_poses = torch.cat([global_orient_global, body_pose_global], dim=1)  # (L, 231)

    identity_coeffs_global = fused_soma_global["identity_coeffs"]
    if identity_coeffs_global.dim() == 2:
        identity_coeffs_global = identity_coeffs_global[0]
    scale_params_global = fused_soma_global["scale_params"]
    if scale_params_global.dim() == 2:
        scale_params_global = scale_params_global[0]

    motion_global = dict(
        poses=global_poses,  # (L, 231)
        identity_coeffs=identity_coeffs_global,  # (45,)
        scale_params=scale_params_global,  # (75,)
        trans=fused_soma_global["transl"].reshape(frame_num, 3),  # (L, 3)
        left_hand_pose=fused_soma_global.get("left_hand_pose", zero_hand).reshape(frame_num, -1),
        right_hand_pose=fused_soma_global.get("right_hand_pose", zero_hand).reshape(frame_num, -1),
        mocap_frame_rate=30,
        gender="neutral",
        model_type="soma",
    )

    motion_global["model"] = "soma"
    motion_incam["model"] = "soma"

    # Compute predicted body height from HMR-predicted identity/scale params
    # (using raw SOMA model, not the optimizer-baked one).
    from grail.models.soma_model import (
        get_tpose_human_height as get_tpose_human_height_soma,
        setup_soma_model,
    )

    soma_model = setup_soma_model(model_path=soma_model_path, device="cpu")
    predicted_body_height = float(
        get_tpose_human_height_soma(
            soma_model,
            identity_coeffs=identity_coeffs_incam.cpu(),
            scale_params=scale_params_incam.cpu(),
            device="cpu",
        )
    )
    motion_incam["predicted_body_height"] = predicted_body_height
    motion_global["predicted_body_height"] = predicted_body_height

    motion = dict(
        motion_global=motion_global,
        motion_incam=motion_incam,
    )

    return motion


def run_human_pose_est(
    video_path,
    model="smplx",
    cache_dir="cache",
    smplx_model_path=None,
    soma_model_path=None,
    is_static_cam=True,
):
    if model in ["smplx", "g1_smplx"]:
        return run_human_pose_est_smplx(
            video_path,
            cache_dir=cache_dir,
            smplx_model_path=smplx_model_path,
            is_static_cam=is_static_cam,
        )
    elif model == "soma":
        return run_human_pose_est_soma(
            video_path,
            cache_dir=cache_dir,
            soma_model_path=soma_model_path,
            is_static_cam=is_static_cam,
        )
    else:
        raise ValueError(f"Invalid model type: {model}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run human pose estimation from video input using SMPL-X or NOVA model."
    )
    parser.add_argument("--video_path", "-v", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="smplx",
        choices=["smplx", "soma"],
        help="Model type: smplx (SMPL-X body model) or soma (SOMA 77-joint model). Default: smplx",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default=None,
        help="Path to save results (.pt file). If not specified, results are not saved.",
    )
    parser.add_argument(
        "--save_mesh", action="store_true", help="Save mesh visualization to output directory"
    )
    parser.add_argument(
        "--mesh_dir",
        type=str,
        default="output/pose_est_meshes",
        help="Directory for mesh output. Default: output/pose_est_meshes",
    )
    args = parser.parse_args()

    print(f"[Pose Estimation] Input video: {args.video_path}")
    print(f"[Pose Estimation] Model type: {args.model}")

    # Run pose estimation based on model type
    motion_result = run_human_pose_est(args.video_path, args.model)

    # Print summary
    motion_global = motion_result["motion_global"]
    motion_incam = motion_result["motion_incam"]
    num_frames = motion_global["poses"].shape[0]
    print(f"[Pose Estimation] Processed {num_frames} frames")
    print(f"[Pose Estimation] Pose shape: {motion_global['poses'].shape}")

    # Save results if output_path specified
    if args.output_path:
        torch.save(motion_result, args.output_path)
        print(f"[Pose Estimation] Results saved to: {args.output_path}")

    # Generate mesh visualizations if requested
    if args.save_mesh:
        os.makedirs(args.mesh_dir, exist_ok=True)

        if args.model == "soma":
            # SOMA mesh generation
            from grail.core.io import save_mesh
            from grail.models.soma_model import generate_soma_mesh, setup_soma_model

            soma_model = setup_soma_model(
                model_path=None,
                device="cuda",
            )

            # Convert motion_global to format expected by generate_soma_mesh
            global_motion = motion_result["motion_global"]
            poses = tensor_to(global_motion["poses"], device="cuda")  # (L, 231)
            frame_num = poses.shape[0]

            # Split poses into global_orient and body_pose
            global_orient = poses[:, :3]  # (L, 3)
            body_pose = poses[:, 3:].reshape(frame_num, 76, 3)  # (L, 76, 3)

            soma_motion_data = {
                "global_orient": global_orient,
                "body_pose": body_pose,
                "identity_coeffs": tensor_to(global_motion["identity_coeffs"], device="cuda"),
                "scale_params": tensor_to(global_motion["scale_params"], device="cuda"),
                "transl": tensor_to(global_motion["trans"], device="cuda"),
            }

            soma_verts_seq, soma_faces = generate_soma_mesh(
                soma_model, soma_motion_data, output_joints=False, require_grad=False, device="cuda"
            )

            for i in range(soma_verts_seq.shape[0]):
                verts = tensor_to_numpy(soma_verts_seq[i])
                faces = tensor_to_numpy(soma_faces)
                mesh_path = os.path.join(args.mesh_dir, f"soma_mesh_{i:04d}.obj")
                save_mesh(verts, faces, mesh_path)

            print(f"[Pose Estimation] Saved {num_frames} NOVA meshes to: {args.mesh_dir}")
        else:
            # SMPL-X mesh generation
            from grail.core.io import save_mesh
            from grail.models.smplx_model import generate_smplx_mesh, setup_smplx_model

            smplx_model = setup_smplx_model(
                model_path=None,
                flat_hand_mean=True,
            )
            global_motion_data = tensor_to(motion_result["motion_global"], device="cuda")
            smplx_verts_seq, smplx_faces = generate_smplx_mesh(smplx_model, global_motion_data)

            for i in range(smplx_verts_seq.shape[0]):
                verts = tensor_to_numpy(smplx_verts_seq[i])
                faces = tensor_to_numpy(smplx_faces)
                mesh_path = os.path.join(args.mesh_dir, f"smplx_mesh_{i:04d}.obj")
                save_mesh(verts, faces, mesh_path)

            print(f"[Pose Estimation] Saved {num_frames} meshes to: {args.mesh_dir}")

    print("[Pose Estimation] Done!")
