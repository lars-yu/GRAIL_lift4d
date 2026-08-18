import json
import os

import numpy as np
import smplx
import torch
from hmr4d.utils.body_model.smplx_lite import (
    SmplxLite,
    SmplxLiteCoco17,
    SmplxLiteV437Coco17,
)
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from grail.models.smplx_constants import (
    SMPLH_SEGMENT,
    SMPLX_BONE_ORDER_NAMES,
)


def setup_smplxlite_coco17_model(model_path=None, device="cuda"):
    if model_path is None:
        smplxlite_coco17_model = SmplxLiteCoco17().to(device)
    else:
        smplxlite_coco17_model = SmplxLiteCoco17(model_path, device=device)
    return smplxlite_coco17_model


def get_coco17_joints(smplxlite_model, motion_data, require_grad=False, device="cuda"):
    betas = motion_data["betas"]
    body_pose = motion_data["poses"][:, 3 : 3 + 21 * 3]
    global_orient = motion_data["poses"][:, :3]
    transl = motion_data["trans"]
    scale = motion_data.get("scale", 1.0)

    frame_num = body_pose.shape[0]

    betas = betas.reshape(1, 1, 10).repeat(1, frame_num, 1)
    body_pose = body_pose.reshape(1, frame_num, 63)
    global_orient = global_orient.reshape(1, frame_num, 3)
    transl = transl.reshape(1, frame_num, 3)

    if require_grad:
        joints = smplxlite_model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
        )
    else:
        with torch.no_grad():
            joints = smplxlite_model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
            )

    transl = transl.unsqueeze(2)
    joints = (joints - transl) * scale + transl
    joints = joints.squeeze(0)
    return joints


def setup_smplx_model(
    model_path=None,
    flat_hand_mean=False,
    device="cuda",
):
    if model_path is None:
        raise ValueError("model_path is required. Set smplx_model_path in the config YAML.")

    smplx_model = smplx.create(
        model_path=model_path,
        model_type="smplx",
        use_pca=False,
        num_pca_comps=45,
        flat_hand_mean=flat_hand_mean,
    ).to(device)

    return smplx_model


def load_smplx_beta(beta_path, device="cuda"):
    """Load SMPL-X shape parameters (betas) from a numpy file.

    Args:
        beta_path: Path to the .npz file containing SMPL-X shape parameters.
                   Expected to contain 'betas' with shape (10,) or (1, 10).
        device: Device to load tensors to.

    Returns:
        torch.Tensor: Betas tensor with shape (1, 10).
    """
    data = np.load(beta_path, allow_pickle=True)

    betas = torch.tensor(data["betas"], dtype=torch.float32, device=device)
    scale = 1.0 if "scale" not in data.keys() else float(data["scale"])

    # Add batch dimension if needed
    if betas.dim() == 1:
        betas = betas.unsqueeze(0)  # (1, 10)

    return betas, scale


def forward_smplx(smplx_model, motion_data, require_grad=False, device="cuda"):
    poses = motion_data["poses"]
    betas = motion_data["betas"]
    trans = motion_data["trans"]
    left_hand_pose = motion_data.get("left_hand_pose", None)
    right_hand_pose = motion_data.get("right_hand_pose", None)

    frame_num = poses.shape[0]
    poses = poses.reshape(frame_num, -1)

    # Prepare SMPLX inputs (assuming poses, betas, trans are already torch tensors)
    betas_batch = betas.reshape(1, 10).repeat(frame_num, 1)
    global_orient = poses[:, :3]
    body_pose = poses[:, 3 : 3 + 21 * 3]
    transl = trans

    # Zero hand poses and facial expressions
    if left_hand_pose is None:
        left_hand_pose = torch.zeros((frame_num, 45), device=device).float()
    else:
        left_hand_pose = left_hand_pose.reshape(frame_num, 45)

    if right_hand_pose is None:
        right_hand_pose = torch.zeros((frame_num, 45), device=device).float()
    else:
        right_hand_pose = right_hand_pose.reshape(frame_num, 45)

    expression = torch.zeros((frame_num, 10), device=device).float()
    jaw_pose = torch.zeros((frame_num, 3), device=device).float()
    leye_pose = torch.zeros((frame_num, 3), device=device).float()
    reye_pose = torch.zeros((frame_num, 3), device=device).float()

    # Generate SMPLX mesh
    if require_grad:
        smplx_output = smplx_model(
            betas=betas_batch,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            return_verts=True,
            return_full_pose=True,
        )
    else:
        with torch.no_grad():
            smplx_output = smplx_model(
                betas=betas_batch,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                transl=transl,
                expression=expression,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                return_verts=True,
                return_full_pose=True,
            )

    return smplx_output


def generate_smplx_mesh(
    smplx_model, motion_data, output_joints=False, require_grad=False, device="cuda"
):
    scale = motion_data.get("scale", 1.0)
    transl = motion_data["trans"]

    smplx_output = forward_smplx(smplx_model, motion_data, require_grad, device)

    frame_num = smplx_output.vertices.shape[0]
    vertices_sequence = smplx_output.vertices  # (frame_num, V, 3)
    transl = transl.reshape(frame_num, 1, 3)
    vertices_sequence = (vertices_sequence - transl) * scale + transl
    faces = torch.from_numpy(smplx_model.faces).to(device).long()

    if output_joints:
        joints = smplx_output.joints[:, : len(SMPLX_BONE_ORDER_NAMES), :3]
        joints = (joints - transl) * scale + transl
        return vertices_sequence, faces, joints
    else:
        return vertices_sequence, faces


def transform_global_to_incam(global_motion_data, incam_motion_data, align_frame=0, device="cuda"):
    """
    Transform global motion data to in-camera coordinate frame

    This function computes a transformation matrix that aligns the global motion data
    with the in-camera motion data at a specific reference frame, then applies this
    transformation to all frames in the global motion data.

    Args:
        global_motion_data (dict): Global motion data containing 'poses', 'trans', 'betas'
        incam_motion_data (dict): In-camera motion data containing 'poses', 'trans', 'betas'
        align_frame (int): Reference frame for alignment (default: 82)
        device (str): Device for computations

    Returns:
        dict: Transformed global motion data in camera coordinate frame
    """
    # Make a copy to avoid modifying the original data
    transformed_motion_data = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in global_motion_data.items()
    }

    for k, v in incam_motion_data.items():
        if k not in transformed_motion_data.keys():
            transformed_motion_data[k] = v.clone()

    # Extract reference poses and translations at the alignment frame
    init_global_t = global_motion_data["trans"][align_frame]
    init_incam_t = incam_motion_data["trans"][align_frame]
    init_global_R = axis_angle_to_matrix(global_motion_data["poses"][align_frame][:3])
    init_incam_R = axis_angle_to_matrix(incam_motion_data["poses"][align_frame][:3])

    # Compute 4x4 transformation matrices for global and in-camera poses
    init_global_pose = torch.eye(4, device=device)
    init_global_pose[:3, :3] = init_global_R
    init_global_pose[:3, 3] = init_global_t

    init_incam_pose = torch.eye(4, device=device)
    init_incam_pose[:3, :3] = init_incam_R
    init_incam_pose[:3, 3] = init_incam_t

    # Compute transformation from global to in-camera coordinate frame
    global_to_incam_pose = init_incam_pose @ torch.linalg.inv(init_global_pose)

    # Transform all frames from global to in-camera coordinate frame
    trans_len = global_motion_data["trans"].shape[0]
    for i in range(trans_len):
        # Convert current frame to 4x4 transformation matrix
        global_t = global_motion_data["trans"][i]
        global_R = axis_angle_to_matrix(global_motion_data["poses"][i][:3])
        global_pose = torch.eye(4, device=device)
        global_pose[:3, :3] = global_R
        global_pose[:3, 3] = global_t

        # Apply transformation to get in-camera pose
        incam_pose = global_to_incam_pose @ global_pose

        # Extract transformed translation and rotation
        transformed_motion_data["trans"][i] = incam_pose[:3, 3].reshape(3)
        transformed_motion_data["poses"][i][:3] = matrix_to_axis_angle(incam_pose[:3, :3])

    return transformed_motion_data


def transform_global_motion(
    smplx_model,
    global_motion_data,
    incam_motion_data,
    cam_R=torch.eye(3),
    cam_t=torch.zeros(3),
    align_frame=0,
    use_global=False,
    gt_beta=None,
    gt_scale=1.0,
    device="cuda",
):
    if use_global:
        transformed_incam_motion_data = transform_global_to_incam(
            global_motion_data, incam_motion_data, align_frame=align_frame, device=device
        )
    else:
        transformed_incam_motion_data = incam_motion_data

    transformed_motion_data = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in transformed_incam_motion_data.items()
    }

    # get the root joint positions
    _, _, incam_joints = generate_smplx_mesh(
        smplx_model,
        transformed_incam_motion_data,
        output_joints=True,
        require_grad=False,
        device=device,
    )

    incam_root_joint_seq = incam_joints[:, 0, :]

    frame_num = transformed_incam_motion_data["trans"].shape[0]
    for i in range(frame_num):
        incam_t = transformed_incam_motion_data["trans"][i]
        incam_R = axis_angle_to_matrix(transformed_incam_motion_data["poses"][i][:3])
        diff = incam_root_joint_seq[i] - incam_t
        transformed_motion_data["trans"][i] = cam_R @ incam_t + cam_t + cam_R @ diff - diff
        transformed_motion_data["poses"][i][:3] = matrix_to_axis_angle(cam_R @ incam_R)

    if gt_beta is not None:
        est_tpose_height = get_tpose_human_height(
            smplx_model, transformed_incam_motion_data["betas"], device=device
        )
        gt_tpose_height = get_tpose_human_height(smplx_model, gt_beta, device=device) * gt_scale
        human_scale_ratio = gt_tpose_height / est_tpose_height

        # Scale translation so human-camera distance matches the new body size
        transformed_motion_data["trans"] = (
            transformed_motion_data["trans"] - cam_t
        ) * human_scale_ratio + cam_t
        transformed_motion_data["betas"] = gt_beta
        transformed_motion_data["scale"] = gt_scale

    return transformed_motion_data


def get_smplx_verts_segment(smplx_verts, segment_name=["L_Hand", "R_Hand"]):
    smplx_verts_segment = []

    if smplx_verts.dim() == 2:
        smplx_verts = smplx_verts.unsqueeze(0)

    if smplx_verts.dim() != 3:
        raise ValueError("smplx_verts must be a 2 or 3D tensor")

    # Load the contact segmentation JSON file
    json_path = os.path.join(os.path.dirname(__file__), "smplx_vert_segmentation_contact.json")
    with open(json_path, "r") as f:
        smplx_vert_segmentation_contact = json.load(f)

    for name in segment_name:
        segment_list = SMPLH_SEGMENT[name]
        for segment in segment_list:
            verts = smplx_vert_segmentation_contact[segment]
            smplx_verts_segment.append(smplx_verts[:, verts])

    smplx_verts_segment = torch.cat(smplx_verts_segment, dim=1)

    return smplx_verts_segment


def get_smplx_segment_indices(segment_name=["L_Hand", "R_Hand"]):
    """
    Return vertex indices for specified body segments.

    Args:
        segment_name: List of segment names (e.g., ["L_Hand", "R_Hand"])

    Returns:
        List of vertex indices for the specified segments
    """
    # Load the contact segmentation JSON file
    json_path = os.path.join(os.path.dirname(__file__), "smplx_vert_segmentation_contact.json")
    with open(json_path, "r") as f:
        smplx_vert_segmentation_contact = json.load(f)

    indices = []
    for name in segment_name:
        segment_list = SMPLH_SEGMENT[name]
        for segment in segment_list:
            indices.extend(smplx_vert_segmentation_contact[segment])
    return indices


def get_tpose_human_height(smplx_model, betas=None, device="cuda"):
    """
    Generate T-pose human and calculate height

    Args:
        smplx_model: SMPL-X model instance
        betas: Shape parameters (N, 10) tensor. If None, uses neutral shape
        device: Device for computation

    Returns:
        float: Height of T-pose human in meters
    """
    # Create T-pose motion data (all zeros = T-pose)
    tpose_data = {
        "betas": betas if betas is not None else torch.zeros(1, 10, device=device),
        "poses": torch.zeros(1, 165, device=device),  # All zeros = T-pose
        "trans": torch.zeros(1, 3, device=device),
        "left_hand_pose": torch.zeros(1, 45, device=device),
        "right_hand_pose": torch.zeros(1, 45, device=device),
    }

    # Generate T-pose mesh
    tpose_verts, _ = generate_smplx_mesh(
        smplx_model,
        tpose_data,
        output_joints=False,
        require_grad=False,
        device=device,
    )

    tpose_human_verts = tpose_verts[0]  # (N, 3)

    # Calculate height (max Y - min Y)
    # SMPL-X Y-axis is the vertical axis
    height = tpose_human_verts[:, 1].max() - tpose_human_verts[:, 1].min()

    return height
