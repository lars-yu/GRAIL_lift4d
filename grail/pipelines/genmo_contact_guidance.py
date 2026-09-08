"""Stage 3.5: scene-conditioned GENMO contact-guidance POC."""

import json
import math
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import trimesh

from grail.adapters.gem_smpl import run_contact_guided_genmo
from grail.adapters.lift4d_depth import load_lift4d_depth_prior
from grail.core.contact_label import detect_contact_labels_from_masks
from grail.core.io import (
    load_human_motion_data,
    load_init_rendering_data,
    load_object_pose_data,
    save_human_motion_data,
)
from grail.preprocessing.preprocess import load_masks_from_cache
from grail.optimization.motion_state import detect_object_motion


GENMO_GUIDANCE_OUTPUT_DIR = "generation/genmo_contact_guidance_poc"
SMPL24_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
OBJECT_TRAJECTORY_METHOD = "fp_contact_anchor_lift4d_relative_depth_v2"
HUMAN_MESH_COLOR = (0.8, 0.6, 0.4)
HUMAN_MESH_COLOR_RGB = (244, 132, 32)


def _json_ready(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _load_mesh(mesh_path, scale):
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    vertices = np.asarray(loaded.vertices, dtype=np.float32) * np.asarray(scale, dtype=np.float32).reshape(1, 3)
    return vertices, np.asarray(loaded.faces, dtype=np.int64)


def _load_object_scale(render_config_file):
    candidates = [
        Path(render_config_file),
        Path(str(render_config_file).replace("foundation_pose_output", "foundation_pose")),
    ]
    for candidate in candidates:
        if candidate.is_file():
            _, _, scale, _, _, _ = load_init_rendering_data(str(candidate))
            return np.asarray(scale, dtype=np.float32).reshape(3)
    raise FileNotFoundError(f"first-frame rendering metadata not found: {candidates}")


def _rotation_change_degrees(initial, optimized):
    relative = optimized @ np.swapaxes(initial, -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _resolve_genmo_motion_prior(video_id, args):
    explicit = getattr(args, "genmo_motion_prior", None)
    opt_cfg = args.cfg.get("optimization", {}) or {}
    configured = explicit or opt_cfg.get("lift4d_motion_prior_path")
    if configured:
        path = str(configured).format(
            video_id=video_id, video_id_safe=video_id.replace("/", "__")
        )
        if not os.path.isabs(path):
            path = os.path.join(args.results_dir, path)
    elif getattr(args, "lift4d_prior_dir", None):
        path = os.path.join(
            args.results_dir, args.lift4d_prior_dir, f"{video_id}.npz"
        )
    else:
        raise ValueError(
            "GENMO object motion requires --genmo-motion-prior, "
            "--lift4d-prior-dir, or optimization.lift4d_motion_prior_path"
        )
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Lift4D motion prior not found: {path}")
    return path


def _compose_anchored_object_trajectory(raw_poses, motion_depth_cam, move_start_frame):
    """Recover a contact-continuous path from FP motion and Lift4D depth."""
    raw_poses = np.asarray(raw_poses, dtype=np.float64)
    motion_depth_cam = np.asarray(motion_depth_cam, dtype=np.float64).reshape(-1)
    frame_num = raw_poses.shape[0]
    move_start_frame = int(move_start_frame)
    if (
        raw_poses.shape != (frame_num, 4, 4)
        or motion_depth_cam.shape != (frame_num,)
    ):
        raise ValueError("Object poses and Lift4D depth must be frame aligned")
    if not 1 <= move_start_frame < frame_num:
        raise ValueError("Object motion onset must lie strictly inside the video")
    if not np.isfinite(raw_poses).all() or not np.isfinite(motion_depth_cam).all():
        raise ValueError("Object trajectory inputs contain NaN/Inf")

    first_gt_translation = raw_poses[0, :3, 3]
    target_depth = (
        first_gt_translation[2]
        + motion_depth_cam
        - motion_depth_cam[move_start_frame]
    )
    raw_translation = raw_poses[:, :3, 3]
    if np.any(raw_translation[:, 2] <= 0) or np.any(target_depth <= 0):
        raise ValueError("FoundationPose and Lift4D object depth must be positive")
    camera_rays = raw_translation / raw_translation[:, 2:3]
    candidate_translation = camera_rays * target_depth[:, None]
    # The object is static before contact. Anchor the candidate trajectory at
    # the contact frame instead of switching from pose 0 to an unrelated
    # absolute FoundationPose estimate at move_start_frame.
    translations = (
        first_gt_translation
        + candidate_translation
        - candidate_translation[move_start_frame]
    )
    translations[:move_start_frame] = first_gt_translation

    poses = raw_poses.copy()
    anchor_rotation = raw_poses[0, :3, :3]
    contact_rotation = raw_poses[move_start_frame, :3, :3]
    relative_rotation = raw_poses[:, :3, :3] @ contact_rotation.T
    poses[:, :3, :3] = relative_rotation @ anchor_rotation
    poses[:, :3, 3] = translations
    poses[:move_start_frame] = raw_poses[0]
    if not np.isfinite(poses).all():
        raise FloatingPointError("Anchored object trajectory contains NaN/Inf")
    return poses.astype(np.float32), translations.astype(np.float32)


def _build_frozen_object_motion(
    raw_poses, lift4d_prior, motion_state, prior_path, output_dir
):
    """Build the reference Stage-3A trajectory without running an optimizer."""
    move_start = int(motion_state.move_start_frame)
    poses_cam, target_translation = _compose_anchored_object_trajectory(
        raw_poses, lift4d_prior.z, move_start
    )
    raw_poses = np.asarray(raw_poses, dtype=np.float32)
    translation_change = np.linalg.norm(
        poses_cam[:, :3, 3] - raw_poses[:, :3, 3], axis=-1
    )
    rotation_change = _rotation_change_degrees(
        raw_poses[:, :3, :3], poses_cam[:, :3, :3]
    )
    post_slice = slice(move_start, None)
    diagnostics = {
        "stage": "3.5a_anchored_lift4d_object_motion",
        "trajectory_method": OBJECT_TRAJECTORY_METHOD,
        "motion_prior_path": str(Path(prior_path).resolve()),
        "move_start_frame": move_start,
        "contact_frame": move_start,
        "motion_detection_confidence": float(motion_state.confidence),
        "first_frame_gt_anchor_depth_m": float(raw_poses[0, 2, 3]),
        "first_frame_anchor_source": "foundationpose_first_frame_gt_aligned",
        "pre_contact_hard_freeze": [0, move_start],
        "post_contact_translation_source": "contact_anchored_foundationpose_ray_plus_lift4d_relative_depth",
        "post_contact_rotation_source": "contact_anchored_foundationpose_relative_rotation",
        "lift4d_kabsch_pose_used": False,
        "contact_losses_enabled": False,
        "optimizer_run": False,
        "human_parameters_optimized": False,
        "translation_change_mean_m": float(translation_change.mean()),
        "translation_change_max_m": float(translation_change.max()),
        "rotation_change_mean_deg": float(rotation_change.mean()),
        "rotation_change_max_deg": float(rotation_change.max()),
        "pre_contact_pose_anchor_max_abs_m": float(
            np.max(np.abs(poses_cam[:move_start] - raw_poses[0]))
        ),
        "post_contact_translation_change_mean_m": float(
            translation_change[post_slice].mean()
        ),
        "post_contact_translation_change_max_m": float(
            translation_change[post_slice].max()
        ),
        "post_contact_rotation_change_mean_deg": float(
            rotation_change[post_slice].mean()
        ),
        "post_contact_rotation_change_max_deg": float(
            rotation_change[post_slice].max()
        ),
        "target_translation_min_m": target_translation.min(axis=0).tolist(),
        "target_translation_max_m": target_translation.max(axis=0).tolist(),
        "target_depth_min_m": float(target_translation[:, 2].min()),
        "target_depth_max_m": float(target_translation[:, 2].max()),
        "trajectory_frozen_before_genmo": True,
        "motion_thresholds": dict(motion_state.thresholds),
        "lift4d_stable_point_count": int(lift4d_prior.stable_point_ids.size),
    }
    np.savez_compressed(
        output_dir / "anchored_lift4d_object_motion.npz",
        poses_in_cam=poses_cam,
        target_translation_cam_m=target_translation,
        target_depth_m=target_translation[:, 2],
        move_start_frame=np.asarray(move_start, dtype=np.int64),
        trajectory_method=np.asarray(OBJECT_TRAJECTORY_METHOD),
        motion_prior_path=np.asarray(str(Path(prior_path).resolve())),
    )
    poses_cam.setflags(write=False)
    return poses_cam, diagnostics


def _select_contact_hand(configured_hand, diagnostics, contact_frame):
    hand = str(configured_hand).lower()
    if hand in ("left", "right"):
        return hand
    if hand == "both":
        raise ValueError("GENMO contact-guidance POC currently supports one selected hand")
    frame = int(contact_frame)
    left = float(diagnostics["left_distance_px"][frame])
    right = float(diagnostics["right_distance_px"][frame])
    if not np.isfinite(left) and not np.isfinite(right):
        raise ValueError("neither hand has a valid 2D projection at the contact frame")
    return "left" if left <= right else "right"


def _motion_from_prediction(prediction, source_motion, coordinate_transform):
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

    global_params = prediction["smpl_params_global"]
    frame_num = global_params["body_pose"].shape[0]
    c2g_R = coordinate_transform["camera_to_genmo_global_R"].float()
    c2g_t = coordinate_transform["camera_to_genmo_global_t"].float()
    human_depth_scale = float(coordinate_transform.get("human_depth_scale", 1.0))
    g2c_R = c2g_R.mT

    global_R = axis_angle_to_matrix(global_params["global_orient"].float())
    incam_R = g2c_R[None] @ global_R
    incam_t = torch.einsum(
        "ij,fj->fi", g2c_R, global_params["transl"].float() - c2g_t
    )

    def pack(params, transl_override=None):
        global_orient = params["global_orient"]
        transl = params["transl"] if transl_override is None else transl_override
        poses = torch.zeros((frame_num, 165), dtype=torch.float32)
        poses[:, :3] = global_orient.reshape(frame_num, 3).float()
        poses[:, 3:66] = global_params["body_pose"].reshape(frame_num, 63).float()
        for hand, offset in (("left_hand_pose", 66), ("right_hand_pose", 111)):
            if source_motion.get(hand) is not None:
                poses[:, offset : offset + 45] = torch.as_tensor(source_motion[hand]).reshape(frame_num, 45)
        betas = global_params["betas"]
        if betas.ndim > 1:
            betas = betas[0]
        return {
            "poses": poses.numpy(),
            "betas": betas.reshape(10).numpy(),
            "trans": transl.reshape(frame_num, 3).numpy(),
            "left_hand_pose": poses[:, 66:111].numpy(),
            "right_hand_pose": poses[:, 111:156].numpy(),
            "mocap_frame_rate": int(source_motion.get("mocap_frame_rate", 30)),
            "gender": source_motion.get("gender", "neutral"),
            "model": "smplx",
            "predicted_body_height": source_motion.get("predicted_body_height"),
            "scale": human_depth_scale,
            "frame0_gt_depth_aligned": True,
        }

    motion_global = pack(global_params)
    # GENMO's native in-camera parameters preserve the exact camera convention
    # used for its K_fullimg/keypoint reprojection. Recomputing these from the
    # contact-frame global transform can rotate/translate the body into a
    # different camera convention and breaks first-frame mask alignment.
    incam_params = prediction.get("smpl_params_incam")
    if incam_params is not None and "global_orient" in incam_params:
        motion_incam = pack(incam_params)
    else:
        motion_incam = pack({"global_orient": matrix_to_axis_angle(incam_R), "transl": incam_t,
                             "body_pose": global_params["body_pose"], "betas": global_params["betas"]})
    for key in ("vitpose", "hand_keypoints_2d", "foot_contact_probs"):
        if key in source_motion:
            motion_incam[key] = source_motion[key]
    return motion_global, motion_incam


def _project(points_cam, K):
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (z > 1e-4)
    projected = np.full((len(points_cam), 2), np.nan, dtype=np.float32)
    projected[valid, 0] = K[0, 0] * points_cam[valid, 0] / z[valid] + K[0, 2]
    projected[valid, 1] = K[1, 1] * points_cam[valid, 1] / z[valid] + K[1, 2]
    return projected


def _global_to_camera(points, coordinate_transform):
    c2g_R = np.asarray(
        torch.as_tensor(coordinate_transform["camera_to_genmo_global_R"]).cpu(),
        dtype=np.float32,
    )
    c2g_t = np.asarray(
        torch.as_tensor(coordinate_transform["camera_to_genmo_global_t"]).cpu(),
        dtype=np.float32,
    )
    return (points - c2g_t.reshape(1, 1, 3)) @ c2g_R


def _camera_aligned_ground_basis(camera_to_global_R):
    """Return ground-plane right/forward axes aligned with the source camera."""
    rotation = np.asarray(camera_to_global_R, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(rotation).all():
        raise ValueError("camera-to-global rotation contains NaN/Inf")

    global_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    def project_to_ground(axis):
        return axis - global_up * np.dot(axis, global_up)

    # All GRAIL transforms operate on row vectors:
    #   global = camera @ camera_to_global_R.T + t.
    # Therefore the camera basis vectors in global coordinates are rows of R.
    camera_right = rotation[0, :]
    right = project_to_ground(camera_right)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        raise ValueError("camera right axis is parallel to the GENMO vertical axis")
    right /= right_norm

    camera_forward = rotation[2, :]
    forward = project_to_ground(camera_forward)
    forward -= right * np.dot(forward, right)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        # Complete an orthonormal ground basis while preserving camera-right.
        forward = np.cross(right, global_up)
        forward_norm = np.linalg.norm(forward)
    forward /= forward_norm
    if np.dot(forward, camera_forward) < 0:
        forward *= -1.0
    return right.astype(np.float32), forward.astype(np.float32)


def _decode_smplx_mesh_sequence(
    smplx_model, motion, *, device, batch_size=32, output_joints=False
):
    """Decode a saved SMPL-X motion into full vertices without changing it."""
    from grail.models.smplx_model import generate_smplx_mesh

    frame_num = len(motion["poses"])
    vertices = []
    joints = []
    faces = None
    for start in range(0, frame_num, batch_size):
        end = min(frame_num, start + batch_size)
        chunk = {
            "poses": torch.as_tensor(motion["poses"][start:end], device=device).float(),
            "betas": torch.as_tensor(motion["betas"], device=device).float(),
            "trans": torch.as_tensor(motion["trans"][start:end], device=device).float(),
            # Translation is calibrated in camera depth, while scale still
            # controls the SMPL-X body geometry around that translation.  Keep
            # both: dropping scale makes the rendered body much taller than
            # the GT silhouette even though its pelvis depth is aligned.
            "scale": float(motion.get("scale", 1.0)),
        }
        for key in ("left_hand_pose", "right_hand_pose"):
            if motion.get(key) is not None:
                chunk[key] = torch.as_tensor(motion[key][start:end], device=device).float()
        decoded = generate_smplx_mesh(
            smplx_model,
            chunk,
            output_joints=output_joints,
            require_grad=False,
            device=device,
        )
        chunk_vertices, chunk_faces = decoded[:2]
        vertices.append(chunk_vertices.detach().cpu())
        if output_joints:
            joints.append(decoded[2][:, :22].detach().cpu())
        if faces is None:
            faces = chunk_faces.detach().cpu()
    decoded = torch.cat(vertices, dim=0).numpy().astype(np.float32, copy=False)
    if not np.isfinite(decoded).all():
        raise FloatingPointError("rendered SMPL-X vertices contain NaN/Inf")
    decoded_faces = faces.numpy().astype(np.int64, copy=False)
    if output_joints:
        return decoded, decoded_faces, torch.cat(joints, dim=0).numpy().astype(np.float32)
    return decoded, decoded_faces


def _attach_smplx_render_meshes(prediction_motion_pairs, model_path):
    """Attach camera-space full-body geometry used only by the POC renderers."""
    from grail.models.smplx_model import setup_smplx_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = setup_smplx_model(
        model_path=str(model_path), flat_hand_mean=True, device=device
    ).eval()
    try:
        for prediction, motion_incam in prediction_motion_pairs:
            vertices, faces, joints = _decode_smplx_mesh_sequence(
                model, motion_incam, device=device, output_joints=True
            )
            prediction["smplx_vertices_incam"] = vertices
            prediction["smplx_faces"] = faces
            prediction["smplx_joints_incam"] = joints
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


def _overlay_projected_mesh(frame, vertices, faces, K, *, color=HUMAN_MESH_COLOR_RGB, alpha=1.0):
    """Project a full SMPL-X mesh with a CPU-friendly opaque default.

    Alpha blending is retained only as a renderer-debug option.  The contact
    POC must render the orange body as opaque by default, so video pixels do
    not show through and look like a second, translucent human trajectory.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    projected = _project(vertices, np.asarray(K, dtype=np.float32))
    valid_vertex = np.isfinite(projected).all(axis=1) & np.isfinite(vertices[:, 2])
    face_valid = valid_vertex[faces].all(axis=1)
    if not np.any(face_valid):
        return frame
    face_ids = np.flatnonzero(face_valid)
    triangles = projected[faces[face_ids]].astype(np.int32)
    depths = vertices[faces[face_ids], 2].mean(axis=1)
    finite = np.isfinite(depths)
    triangles = triangles[finite]
    depths = depths[finite]
    if len(triangles) == 0:
        return frame
    # Painter's order: far triangles first, near triangles last.
    order = np.argsort(depths)[::-1]
    triangles = triangles[order]
    height, width = frame.shape[:2]
    overlay = np.empty_like(frame)
    overlay[:] = np.asarray(color, dtype=np.uint8)
    mesh_mask = np.zeros((height, width), dtype=np.uint8)
    # Clip only triangles whose bounding boxes can intersect the image. This
    # keeps the loop cheap when GENMO produces an off-screen body part.
    for triangle in triangles:
        x_min, y_min = triangle.min(axis=0)
        x_max, y_max = triangle.max(axis=0)
        if x_max < 0 or y_max < 0 or x_min >= width or y_min >= height:
            continue
        cv2.fillConvexPoly(mesh_mask, triangle, 255, lineType=cv2.LINE_AA)
    if not np.any(mesh_mask):
        return frame
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"mesh alpha must be in (0, 1], got {alpha}")
    if alpha == 1.0:
        return np.where(mesh_mask[..., None] > 0, overlay, frame)
    blended = cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0.0)
    return np.where(mesh_mask[..., None] > 0, blended, frame)


def _overlay_projected_object(frame, vertices, faces, K):
    """Render the recovered object as the opaque blue GRAIL mesh."""
    return _overlay_projected_mesh(
        frame, vertices, faces, K, color=(70, 130, 230), alpha=1.0
    )


def render_guided_motion(
    video_path,
    output_path,
    prediction,
    object_vertices,
    object_faces,
    object_poses,
    coordinate_transform,
    contact_frame,
    selected_hand,
    label,
    human_mask_dir=None,
    human_masks=None,
    human_mesh_alpha=1.0,
):
    """Render a synchronized orange SMPL-X mesh and object overlay."""
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))
    # Use GENMO's camera-space output directly. The global transform remains
    # the authoritative path for 3D guidance and the top-view renderer.
    joints_cam = np.asarray(
        prediction.get("smplx_joints_incam", prediction["smpl24_joints_incam"]),
        dtype=np.float32,
    )[:, :22]
    K_all = np.asarray(prediction["K_fullimg"], dtype=np.float32)
    if K_all.ndim == 2:
        K_all = np.repeat(K_all[None], len(joints_cam), axis=0)
    sample_ids = np.linspace(0, len(object_vertices) - 1, min(600, len(object_vertices))).astype(int)
    hand_index = 20 if selected_hand == "left" else 21
    human_vertices = prediction.get("smplx_vertices_incam")
    human_faces = prediction.get("smplx_faces")
    if human_vertices is not None and human_faces is not None:
        human_vertices = np.asarray(human_vertices, dtype=np.float32)
        human_faces = np.asarray(human_faces, dtype=np.int32)
    mask_files = None
    if human_masks is None and human_mask_dir is not None:
        mask_root = Path(human_mask_dir)
        if mask_root.is_dir():
            mask_files = sorted(mask_root.glob("*.png"))
            if not mask_files:
                mask_files = sorted(mask_root.glob("*.jpg"))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    try:
        for frame_index in range(len(joints_cam)):
            frame = reader.get_data(frame_index).copy()
            K = K_all[min(frame_index, len(K_all) - 1)]
            if human_vertices is not None and human_faces is not None:
                frame = _overlay_projected_mesh(
                    frame,
                    human_vertices[frame_index],
                    human_faces,
                    K,
                    alpha=human_mesh_alpha,
                )
            human_mask = None
            if human_masks is not None and frame_index < len(human_masks):
                human_mask = np.asarray(human_masks[frame_index]).squeeze()
            elif mask_files:
                mask_path = mask_files[min(frame_index, len(mask_files) - 1)]
                human_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if human_mask is not None:
                if human_mask.shape[:2] != frame.shape[:2]:
                    human_mask = cv2.resize(
                        human_mask,
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                contours, _ = cv2.findContours(
                    (human_mask > 0).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(frame, contours, -1, (40, 255, 255), 2, cv2.LINE_AA)
            joints_2d = _project(joints_cam[frame_index], K)
            pose = object_poses[min(frame_index, len(object_poses) - 1)]
            object_cam = object_vertices @ pose[:3, :3].T + pose[:3, 3]
            frame = _overlay_projected_object(frame, object_cam, object_faces, K)
            for child, parent in enumerate(SMPL24_PARENTS):
                if parent < 0 or not np.isfinite(joints_2d[[child, parent]]).all():
                    continue
                cv2.line(
                    frame,
                    tuple(np.rint(joints_2d[parent]).astype(int)),
                    tuple(np.rint(joints_2d[child]).astype(int)),
                    (60, 220, 80),
                    3,
                    cv2.LINE_AA,
                )
            if np.isfinite(joints_2d[hand_index]).all():
                cv2.circle(
                    frame,
                    tuple(np.rint(joints_2d[hand_index]).astype(int)),
                    7,
                    (255, 60, 220),
                    -1,
                )
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"{label} | SMPL-X mesh | contact={contact_frame} | {selected_hand} hand",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.append_data(frame)
    finally:
        writer.close()
        reader.close()


def mesh_mask_alignment_metrics(prediction, K, human_masks):
    """Return per-frame projected mesh/mask bbox and IoU diagnostics."""
    vertices = np.asarray(prediction.get("smplx_vertices_incam"), dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    if K.ndim == 2:
        K = np.repeat(K[None], len(vertices), axis=0)
    metrics = []
    for index, frame_vertices in enumerate(vertices):
        projected = _project(frame_vertices, K[min(index, len(K) - 1)])
        valid = np.isfinite(projected).all(axis=1)
        if not np.any(valid) or human_masks is None or index >= len(human_masks):
            continue
        mask = np.asarray(human_masks[index]).squeeze() > 0
        if mask.ndim != 2:
            continue
        ys, xs = np.where(mask)
        if not len(xs):
            continue
        points = projected[valid]
        mesh_mask = np.zeros(mask.shape, dtype=np.uint8)
        triangles = prediction["smplx_faces"]
        valid_face = valid[triangles].all(axis=1)
        for tri in projected[triangles[valid_face]].astype(np.int32):
            cv2.fillConvexPoly(mesh_mask, tri, 1)
        inter = np.logical_and(mesh_mask > 0, mask)
        union = np.logical_or(mesh_mask > 0, mask)
        metrics.append({
            "frame": int(index),
            "mesh_bbox_xyxy": [float(points[:, 0].min()), float(points[:, 1].min()),
                                float(points[:, 0].max()), float(points[:, 1].max())],
            "mask_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "mesh_mask_iou": float(inter.sum() / max(1, union.sum())),
        })
    return metrics


def make_side_by_side(initial_video, guided_video, output_path):
    readers = [imageio.get_reader(initial_video), imageio.get_reader(guided_video)]
    fps = float(readers[0].get_meta_data().get("fps", 30.0))
    count = min(reader.count_frames() for reader in readers)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    try:
        for index in range(count):
            left, right = (reader.get_data(index) for reader in readers)
            if left.shape[:2] != right.shape[:2]:
                right = cv2.resize(right, (left.shape[1], left.shape[0]))
            writer.append_data(np.hstack((left, right)))
    finally:
        writer.close()
        for reader in readers:
            reader.close()


def render_top_view_comparison(
    output_path,
    initial_prediction,
    guided_prediction,
    object_vertices,
    object_poses,
    coordinate_transform,
    object_surface_target,
    contact_frame,
    selected_hand,
    fps,
):
    """Render meshes from above with source-camera right kept on screen-right."""
    initial = np.asarray(
        initial_prediction.get("smplx_joints_global", initial_prediction["smpl24_joints_global"]),
        dtype=np.float32,
    )[:, :22]
    guided = np.asarray(
        guided_prediction.get("smplx_joints_global", guided_prediction["smpl24_joints_global"]),
        dtype=np.float32,
    )[:, :22]
    initial_mesh_cam = np.asarray(
        initial_prediction["smplx_vertices_incam"], dtype=np.float32
    )
    guided_mesh_cam = np.asarray(
        guided_prediction["smplx_vertices_incam"], dtype=np.float32
    )
    human_faces = np.asarray(initial_prediction["smplx_faces"], dtype=np.int32)
    c2g_R = np.asarray(
        torch.as_tensor(coordinate_transform["camera_to_genmo_global_R"]).cpu(),
        dtype=np.float32,
    )
    c2g_t = np.asarray(
        torch.as_tensor(coordinate_transform["camera_to_genmo_global_t"]).cpu(),
        dtype=np.float32,
    )
    sample_ids = np.linspace(
        0, len(object_vertices) - 1, min(600, len(object_vertices))
    ).astype(int)
    object_global = []
    for pose in object_poses[: len(initial)]:
        vertices_cam = object_vertices[sample_ids] @ pose[:3, :3].T + pose[:3, 3]
        object_global.append(vertices_cam @ c2g_R.T + c2g_t)
    object_global = np.asarray(object_global, dtype=np.float32)
    target = np.asarray(torch.as_tensor(object_surface_target).cpu(), dtype=np.float32)
    camera = c2g_t
    initial_mesh = initial_mesh_cam @ c2g_R.T + c2g_t
    guided_mesh = guided_mesh_cam @ c2g_R.T + c2g_t
    camera_right, camera_forward = _camera_aligned_ground_basis(c2g_R)

    def to_ground(points):
        relative = np.asarray(points) - camera
        return np.stack(
            (relative @ camera_right, relative @ camera_forward), axis=-1
        )

    all_ground = np.concatenate(
        (
            to_ground(initial).reshape(-1, 2),
            to_ground(guided).reshape(-1, 2),
            to_ground(initial_mesh[:, ::25]).reshape(-1, 2),
            to_ground(guided_mesh[:, ::25]).reshape(-1, 2),
            to_ground(object_global).reshape(-1, 2),
            to_ground(target[None]),
            np.zeros((1, 2), dtype=np.float32),
        ),
        axis=0,
    )
    lower = np.nanmin(all_ground, axis=0)
    upper = np.nanmax(all_ground, axis=0)
    center = (lower + upper) * 0.5
    span = max(float((upper - lower).max()), 1.0) * 1.12
    panel_size = 720
    margin = 52
    scale = (panel_size - 2 * margin) / span

    def to_pixel(points):
        ground = to_ground(points)
        pixel = np.empty_like(ground)
        pixel[..., 0] = panel_size * 0.5 + (ground[..., 0] - center[0]) * scale
        pixel[..., 1] = panel_size * 0.5 - (ground[..., 1] - center[1]) * scale
        return np.rint(pixel).astype(np.int32)

    def draw_panel(canvas, offset_x, joints, human_vertices, obj_points, label, frame_index):
        panel = canvas[:, offset_x : offset_x + panel_size]
        panel[:] = (245, 245, 242)
        for meter in np.arange(-10.0, 10.01, 0.5):
            x = int(round(panel_size * 0.5 + (meter - center[0]) * scale))
            z = int(round(panel_size * 0.5 - (meter - center[1]) * scale))
            if 0 <= x < panel_size:
                cv2.line(panel, (x, 0), (x, panel_size), (222, 222, 218), 1)
            if 0 <= z < panel_size:
                cv2.line(panel, (0, z), (panel_size, z), (222, 222, 218), 1)
        human_2d = to_pixel(human_vertices)
        mesh_triangles = human_2d[human_faces]
        cv2.fillPoly(panel, [triangle for triangle in mesh_triangles], HUMAN_MESH_COLOR_RGB, cv2.LINE_AA)
        for point in to_pixel(obj_points):
            cv2.circle(panel, tuple(point), 2, (245, 145, 30), -1)
        joints_2d = to_pixel(joints)
        for child, parent in enumerate(SMPL24_PARENTS):
            if parent >= 0:
                cv2.line(
                    panel,
                    tuple(joints_2d[parent]),
                    tuple(joints_2d[child]),
                    (55, 185, 75),
                    4,
                    cv2.LINE_AA,
                )
        hand_index = 20 if selected_hand == "left" else 21
        cv2.circle(panel, tuple(joints_2d[hand_index]), 8, (220, 50, 210), -1)
        cv2.drawMarker(
            panel,
            tuple(to_pixel(target[None])[0]),
            (220, 35, 20),
            cv2.MARKER_CROSS,
            18,
            3,
        )
        cv2.drawMarker(
            panel,
            tuple(to_pixel(camera[None])[0]),
            (40, 40, 40),
            cv2.MARKER_TRIANGLE_UP,
            18,
            2,
        )
        cv2.rectangle(panel, (0, 0), (panel_size, 42), (20, 20, 20), -1)
        cv2.putText(
            panel,
            f"{label} | camera-aligned top | frame={frame_index} | contact={contact_frame}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        axis_origin = (78, panel_size - 54)
        cv2.arrowedLine(panel, axis_origin, (158, panel_size - 54), (45, 45, 45), 2, tipLength=0.16)
        cv2.arrowedLine(panel, axis_origin, (78, panel_size - 134), (45, 45, 45), 2, tipLength=0.16)
        cv2.putText(panel, "camera right", (164, panel_size - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (45, 45, 45), 1, cv2.LINE_AA)
        cv2.putText(panel, "forward", (36, panel_size - 143), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (45, 45, 45), 1, cv2.LINE_AA)

    writer = imageio.get_writer(output_path, fps=float(fps), codec="libx264", quality=8)
    try:
        for frame_index in range(len(initial)):
            canvas = np.empty((panel_size, panel_size * 2, 3), dtype=np.uint8)
            draw_panel(
                canvas,
                0,
                initial[frame_index],
                initial_mesh[frame_index],
                object_global[frame_index],
                "Initial GENMO",
                frame_index,
            )
            draw_panel(
                canvas,
                panel_size,
                guided[frame_index],
                guided_mesh[frame_index],
                object_global[frame_index],
                "Guided GENMO",
                frame_index,
            )
            writer.append_data(canvas)
    finally:
        writer.close()


def run_genmo_contact_guidance_stage(video_id, args, object_mesh_path):
    """Build and freeze anchored object motion, then run GENMO guidance."""
    video_id = video_id.removesuffix(".mp4")
    hmr_file = Path(args.results_dir) / args.hmr_dir / f"{video_id}.npz"
    video_file = Path(args.results_dir) / args.video_dir / f"{video_id}.mp4"
    cache_dir = Path(args.results_dir) / args.hmr_cache_dir / video_id
    masks_file = Path(args.results_dir) / args.recon_cache_dir / "masks" / f"{video_id}.npz"
    pose_root = Path(args.results_dir) / args.foundation_pose_output_dir / video_id
    object_pose_file = pose_root / "pose_estimation_output" / "poses_in_cam.pkl"
    render_config_file = pose_root / "first_frame_pose.pickle"
    video_id_origin = video_id[: video_id.find("-end")] if "-end" in video_id else video_id
    gt_depth_file = (
        Path(args.results_dir) / args.depth_gt_dir / video_id / "000000.png"
    )
    human_mask_file = (
        Path(args.results_dir)
        / args.foundation_pose_dir
        / video_id_origin
        / "human_masks"
        / "000000.png"
    )
    if not human_mask_file.is_file():
        human_mask_file = (
            Path(args.results_dir)
            / args.foundation_pose_dir
            / video_id
            / "human_masks"
            / "000000.png"
        )
    human_mask_dir = human_mask_file.parent if human_mask_file is not None else None
    output_dir = Path(args.results_dir) / getattr(
        args, "genmo_guidance_dir", GENMO_GUIDANCE_OUTPUT_DIR
    ) / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    source_motion = load_human_motion_data(str(hmr_file), is_global=False)
    masks = load_masks_from_cache(str(masks_file))
    frame_num = len(source_motion["poses"])
    human_masks = [masks[index][1] for index in range(frame_num)]
    object_masks = [masks[index][0] for index in range(frame_num)]
    contact_cfg = (args.cfg.get("optimization", {}).get("contact", {}) or {})
    _, mask_contact_frame, contact_diagnostics = detect_contact_labels_from_masks(
        source_motion["hand_keypoints_2d"],
        human_masks,
        object_masks,
        start_idx=0,
        end_idx=frame_num,
        interval_length=int(contact_cfg.get("interval_length", 8)),
        hand=str(contact_cfg.get("hand", "auto")),
        distance_threshold_px=float(contact_cfg.get("distance_threshold_px", 12.0)),
        min_mask_iou=float(contact_cfg.get("min_mask_iou", 0.001)),
        min_hand_inside_ratio=float(contact_cfg.get("min_hand_inside_ratio", 0.20)),
        required_consecutive=int(contact_cfg.get("required_consecutive", 2)),
    )
    prior_path = _resolve_genmo_motion_prior(video_id, args)
    optimization_cfg = args.cfg.get("optimization", {}) or {}
    motion_cfg = optimization_cfg.get("object_motion_state", {}) or {}
    lift4d_prior = load_lift4d_depth_prior(
        prior_path,
        frame_num=frame_num,
        median_window=int(optimization_cfg.get("lift4d_median_window", 7)),
        detection_median_window=int(motion_cfg.get("detection_median_window", 5)),
        smooth_window=int(optimization_cfg.get("lift4d_center_smooth_window", 31)),
        savgol_polyorder=int(optimization_cfg.get("lift4d_savgol_polyorder", 2)),
        stable_point_count=int(optimization_cfg.get("lift4d_stable_point_count", 2500)),
        min_stable_points=int(optimization_cfg.get("lift4d_min_stable_points", 64)),
    )
    motion_state = detect_object_motion(
        lift4d_prior.center_cam_raw,
        object_masks,
        smoothed_z=lift4d_prior.z,
        config=motion_cfg,
    )
    contact_frame = int(motion_state.move_start_frame)
    selected_hand = _select_contact_hand(
        contact_cfg.get("hand", "auto"), contact_diagnostics, mask_contact_frame
    )
    configured_transition = getattr(args, "genmo_contact_transition_frames", None)
    guidance_temporal_weight = float(
        getattr(args, "genmo_guidance_temporal_weight", 0.5)
    )
    post_contact_relative_velocity_weight = float(
        getattr(args, "genmo_post_contact_relative_velocity_weight", 15.0)
    )
    post_contact_hold_radius = float(
        getattr(args, "genmo_post_contact_hold_radius", 0.02)
    )
    post_contact_relative_step_tolerance = float(
        getattr(args, "genmo_post_contact_relative_step_tolerance", 0.002)
    )
    max_guidance_update_norm = float(
        getattr(args, "genmo_max_guidance_update_norm", 0.25)
    )
    final_guidance_update_norm = float(
        getattr(args, "genmo_final_guidance_update_norm", 0.05)
    )
    arm_guidance_fade_fraction = float(
        getattr(args, "genmo_arm_guidance_fade_fraction", 0.20)
    )
    root_guidance_final_scale = float(
        getattr(args, "genmo_root_guidance_final_scale", 0.10)
    )
    late_inner_steps = int(getattr(args, "genmo_late_inner_steps", 2))
    final_inner_steps = int(getattr(args, "genmo_final_inner_steps", 5))
    arm_max_update_cap = float(getattr(args, "genmo_arm_max_update_cap", 0.60))
    arm_final_update_cap = float(getattr(args, "genmo_arm_final_update_cap", 0.25))
    root_max_update_cap = float(getattr(args, "genmo_root_max_update_cap", 0.05))
    root_final_update_cap = float(getattr(args, "genmo_root_final_update_cap", 0.02))
    root_gradient_smooth_kernel = int(
        getattr(args, "genmo_root_gradient_smooth_kernel", 9)
    )
    inner_line_search = bool(getattr(args, "genmo_inner_line_search", True))
    arm_gradient_smooth_kernel = int(getattr(args, "genmo_arm_gradient_smooth_kernel", 31))
    contact_standoff_m = float(getattr(args, "genmo_contact_standoff_m", 0.05))
    contact_min_clearance_m = float(getattr(args, "genmo_contact_min_clearance_m", 0.04))
    penetration_weight = float(getattr(args, "genmo_penetration_weight", 60.0))
    root_activation_distance_min = float(
        getattr(args, "genmo_root_activation_distance_min", 0.06)
    )
    root_activation_distance_max = float(
        getattr(args, "genmo_root_activation_distance_max", 0.15)
    )
    root_leg_fade_fraction = float(getattr(args, "genmo_root_leg_fade_fraction", 0.30))
    torso_max_update_cap = float(getattr(args, "genmo_torso_max_update_cap", 0.05))
    torso_final_update_cap = float(getattr(args, "genmo_torso_final_update_cap", 0.01))
    leg_max_update_cap = float(getattr(args, "genmo_leg_max_update_cap", 0.05))
    leg_final_update_cap = float(getattr(args, "genmo_leg_final_update_cap", 0.02))
    w_root_target = float(getattr(args, "genmo_root_target_weight", 1.0))
    w_root_vertical_lock = float(getattr(args, "genmo_root_vertical_lock_weight", 5.0))
    support_foot_weight = float(getattr(args, "genmo_support_foot_weight", 2.0))
    ground_contact_weight = float(getattr(args, "genmo_ground_contact_weight", 1.0))
    leg_reference_weight = float(getattr(args, "genmo_leg_reference_weight", 1.0))
    arm_reference_weight = float(getattr(args, "genmo_arm_reference_weight", 0.05))
    arm_smoothness_weight = float(getattr(args, "genmo_arm_smoothness_weight", 0.15))
    approach_smoothness_weight = float(
        getattr(args, "genmo_approach_smooth_weight", 1.0)
    )
    post_contact_follow_mode = str(
        getattr(args, "genmo_post_contact_follow_mode", "pose")
    )
    # v24 whole-body action terms (§5 root velocity, §8 torso/elbow, §10 slide).
    w_root_velocity = float(getattr(args, "genmo_root_velocity_weight", 1.0))
    torso_reference_weight = float(getattr(args, "genmo_torso_reference_weight", 0.5))
    torso_smoothness_weight = float(getattr(args, "genmo_torso_smoothness_weight", 0.5))
    elbow_direction_weight = float(getattr(args, "genmo_elbow_direction_weight", 0.5))
    foot_slide_limit = float(getattr(args, "genmo_foot_slide_limit", 0.05))
    palm_velocity_weight = float(getattr(args, "genmo_palm_velocity_weight", 0.0))
    palm_velocity_slack_m = float(getattr(args, "genmo_palm_velocity_slack_m", 0.005))
    interpolate_pre_contact_target = bool(
        getattr(args, "genmo_interpolate_approach_target", True)
    )
    approach_weight_floor = float(getattr(args, "genmo_approach_weight_floor", 0.5))
    approach_max_step_m = float(getattr(args, "genmo_approach_max_step_m", 0.0))
    post_contact_worst_frame_weight = float(
        getattr(args, "genmo_post_contact_worst_frame_weight", 0.5)
    )
    post_contact_terminal_position_weight = float(
        getattr(args, "genmo_post_contact_terminal_position_weight", 6.0)
    )
    post_contact_terminal_frames = int(
        getattr(args, "genmo_post_contact_terminal_frames", 16)
    )
    contact_frame_position_weight = float(
        getattr(args, "genmo_contact_frame_position_weight", 1.25)
    )
    contact_frame_weight_radius = int(
        getattr(args, "genmo_contact_frame_weight_radius", 2)
    )
    # §6: the approach ramp must span enough frames to produce a natural step /
    # reach.  Prefer the mask/gait-derived start, but clamp the window length to
    # [min_approach, max_approach], extending BACKWARDS only (never move the
    # contact frame).  A too-short window makes the hand lunge at the object.
    min_approach_frames = int(getattr(args, "genmo_min_approach_frames", 20))
    max_approach_frames = int(getattr(args, "genmo_max_approach_frames", 40))
    if configured_transition is not None:
        contact_transition_frames = max(0, int(configured_transition))
    else:
        detected = max(1, contact_frame - int(mask_contact_frame))
        contact_transition_frames = detected
        if min_approach_frames > 0:
            contact_transition_frames = max(contact_transition_frames, min_approach_frames)
        if max_approach_frames > 0:
            contact_transition_frames = min(contact_transition_frames, max_approach_frames)
        # Never extend past frame 0 (backwards-only, contact frame unchanged).
        contact_transition_frames = min(contact_transition_frames, contact_frame)

    print("Stage 3.5a: Anchored Lift4D Object Motion (no optimization/contact loss)")
    raw_object_poses = np.asarray(
        load_object_pose_data(str(object_pose_file)), dtype=np.float32
    )
    object_poses, object_motion_diagnostics = _build_frozen_object_motion(
        raw_object_poses, lift4d_prior, motion_state, prior_path, output_dir
    )
    with (output_dir / "object_motion_diagnostics.json").open("w") as handle:
        json.dump(
            _json_ready(object_motion_diagnostics), handle, indent=2, allow_nan=False
        )
    print("Stage 3.5b: GENMO Contact-Guided Refinement with frozen object motion")

    object_scale = _load_object_scale(render_config_file)
    object_vertices, object_faces = _load_mesh(object_mesh_path, object_scale)
    pose_at_contact = object_poses[contact_frame]
    object_vertices_cam = (
        object_vertices @ pose_at_contact[:3, :3].T + pose_at_contact[:3, 3]
    )
    result = run_contact_guided_genmo(
        str(video_file),
        str(cache_dir),
        object_vertices_cam,
        object_faces,
        contact_frame,
        selected_hand,
        seed=args.genmo_guidance_seed,
        reach_error_threshold=args.reach_error_threshold,
        max_arm_rotation_change_deg=args.max_arm_rotation_change_deg,
        max_root_correction=args.max_root_correction,
        guidance_strength=args.genmo_guidance_strength,
        root_guidance_multiplier=args.genmo_root_guidance_multiplier,
        is_static_cam=bool(args.cfg["human_model"].get("static_cam", True)),
        verbose=args.verbose,
        diagnostics_path=output_dir / "diagnostics.json",
        gt_depth_path=gt_depth_file,
        human_mask_path=human_mask_file,
        object_vertices_local=object_vertices,
        object_poses_cam=object_poses,
        contact_transition_frames=contact_transition_frames,
        guidance_temporal_weight=guidance_temporal_weight,
        post_contact_hold_radius=post_contact_hold_radius,
        post_contact_relative_velocity_weight=post_contact_relative_velocity_weight,
        post_contact_relative_step_tolerance=post_contact_relative_step_tolerance,
        max_guidance_update_norm=max_guidance_update_norm,
        final_guidance_update_norm=final_guidance_update_norm,
        arm_guidance_fade_fraction=arm_guidance_fade_fraction,
        root_guidance_final_scale=root_guidance_final_scale,
        late_inner_steps=late_inner_steps,
        final_inner_steps=final_inner_steps,
        arm_max_update_cap=arm_max_update_cap,
        arm_final_update_cap=arm_final_update_cap,
        root_max_update_cap=root_max_update_cap,
        root_final_update_cap=root_final_update_cap,
        root_gradient_smooth_kernel=root_gradient_smooth_kernel,
        arm_gradient_smooth_kernel=arm_gradient_smooth_kernel,
        inner_line_search=inner_line_search,
        contact_standoff_m=contact_standoff_m,
        contact_min_clearance_m=contact_min_clearance_m,
        penetration_weight=penetration_weight,
        root_activation_distance_min=root_activation_distance_min,
        root_activation_distance_max=root_activation_distance_max,
        root_leg_fade_fraction=root_leg_fade_fraction,
        torso_max_update_cap=torso_max_update_cap,
        torso_final_update_cap=torso_final_update_cap,
        leg_max_update_cap=leg_max_update_cap,
        leg_final_update_cap=leg_final_update_cap,
        w_root_target=w_root_target,
        w_root_vertical_lock=w_root_vertical_lock,
        support_foot_weight=support_foot_weight,
        ground_contact_weight=ground_contact_weight,
        leg_reference_weight=leg_reference_weight,
        arm_reference_weight=arm_reference_weight,
        arm_smoothness_weight=arm_smoothness_weight,
        post_contact_worst_frame_weight=post_contact_worst_frame_weight,
        post_contact_terminal_position_weight=post_contact_terminal_position_weight,
        post_contact_terminal_frames=post_contact_terminal_frames,
        contact_frame_position_weight=contact_frame_position_weight,
        contact_frame_weight_radius=contact_frame_weight_radius,
        approach_smoothness_weight=approach_smoothness_weight,
        post_contact_follow_mode=post_contact_follow_mode,
        w_root_velocity=w_root_velocity,
        torso_reference_weight=torso_reference_weight,
        torso_smoothness_weight=torso_smoothness_weight,
        elbow_direction_weight=elbow_direction_weight,
        foot_slide_limit=foot_slide_limit,
        palm_velocity_weight=palm_velocity_weight,
        palm_velocity_slack_m=palm_velocity_slack_m,
        interpolate_pre_contact_target=interpolate_pre_contact_target,
        approach_weight_floor=approach_weight_floor,
        approach_max_step_m=approach_max_step_m,
    )
    result["diagnostics"]["object_pose_at_contact_frame"] = pose_at_contact.tolist()
    result["diagnostics"]["object_pose_source"] = OBJECT_TRAJECTORY_METHOD
    result["diagnostics"]["object_motion"] = object_motion_diagnostics
    result["diagnostics"]["object_trajectory_frozen_during_genmo"] = True
    result["diagnostics"]["coordinate_transform"] = _json_ready(result["coordinate_transform"])
    result["diagnostics"]["contact_detector"] = "object_motion_onset"
    result["diagnostics"]["mask_hand_contact_candidate_frame"] = int(
        mask_contact_frame
    )
    result["diagnostics"]["hand_selection_frame"] = int(mask_contact_frame)
    result["diagnostics"]["contact_guidance_ramp_start_frame"] = int(
        max(0, contact_frame - contact_transition_frames)
    )
    result["diagnostics"]["contact_detection"] = contact_diagnostics

    variants = {
        "initial_genmo_motion.npz": result["initial"],
        "arm_only_guided_motion.npz": result["arm_only"],
        "selected_guided_motion.npz": result["selected"],
    }
    # §14: the whole-body candidate is always built now — save it for comparison.
    whole_body_pred = result.get("whole_body", result.get("arm_root"))
    if whole_body_pred is not None:
        variants["whole_body_guided_motion.npz"] = whole_body_pred
        variants["arm_root_guided_motion.npz"] = whole_body_pred  # legacy alias
    else:
        (output_dir / "whole_body_guided_motion.npz").unlink(missing_ok=True)
        (output_dir / "arm_root_guided_motion.npz").unlink(missing_ok=True)
    saved_motions = {}
    for filename, prediction in variants.items():
        motion_global, motion_incam = _motion_from_prediction(
            prediction, source_motion, result["coordinate_transform"]
        )
        save_human_motion_data(motion_global, motion_incam, str(output_dir / filename))
        saved_motions[filename] = (motion_global, motion_incam)

    # Finger-grasp refinement: optimize the grasping hand so fingers wrap the
    # object without penetrating (arm/body from GENMO stay fixed).  Updates the
    # selected motion in place (both the exported npz and the render input).
    # §12: DISABLED by default in v24 — validate body/arm/palm first.  A copy of
    # the selected motion BEFORE any finger refinement is always kept so contact/
    # penetration metrics can be recomputed when finger-grasp is re-enabled.
    sel_global0, sel_incam0 = saved_motions["selected_guided_motion.npz"]
    save_human_motion_data(
        sel_global0, sel_incam0,
        str(output_dir / "selected_guided_motion_before_finger.npz"),
    )

    # v26: fixed-object grasp + arm IK.  After the guided approach + contact, the
    # palm-centre contact is frozen in the object's LOCAL frame (offset out by the
    # palm-shell thickness so the hand mesh rests ON the surface) and transported
    # rigidly by the object trajectory; a small IK rides the arm on those targets
    # so the hand tracks the lifted object without drifting away or penetrating.
    if bool(getattr(args, "genmo_fixed_object_grasp_ik", True)):
        from grail.optimization.fixed_object_arm_ik import (
            FixedObjectArmIKConfig, refine_fixed_object_grasp_ik,
        )
        from grail.models.smplx_model import setup_smplx_model

        ik_model_path = Path(args.cfg["human_model"]["smplx_model_path"])
        if not ik_model_path.is_absolute():
            ik_model_path = Path.cwd() / ik_model_path
        ik_model = setup_smplx_model(
            model_path=str(ik_model_path), flat_hand_mean=True, device="cuda"
        )
        ik_config = FixedObjectArmIKConfig(
            contact_frame=contact_frame,
            selected_hand=selected_hand,
            iterations=int(getattr(args, "genmo_ik_iterations", 200)),
            position_weight=float(getattr(args, "genmo_ik_position_weight", 150.0)),
            normal_weight=float(getattr(args, "genmo_ik_normal_weight", 2.0)),
            penetration_weight=float(getattr(args, "genmo_ik_penetration_weight", 800.0)),
            finger_contact_weight=float(getattr(args, "genmo_ik_finger_weight", 3.0)),
            min_clearance=float(getattr(args, "genmo_ik_min_clearance_m", 0.0)),
            palm_clearance_min=float(getattr(args, "genmo_ik_palm_clearance_min_m", 0.018)),
            palm_clearance_max=float(getattr(args, "genmo_ik_palm_clearance_max_m", 0.040)),
        )
        sel_global, sel_incam = saved_motions["selected_guided_motion.npz"]
        save_human_motion_data(
            sel_global, sel_incam, str(output_dir / "selected_guided_motion_before_ik.npz")
        )
        refined_incam, ik_diag = refine_fixed_object_grasp_ik(
            sel_incam, object_vertices, object_faces, object_poses,
            contact_frame, selected_hand, ik_model, ik_config, device="cuda",
        )
        # Body + hand pose are frame-local (identical in global and in-camera);
        # copy them from the refined in-camera motion into the global track.
        refined_global = dict(sel_global)
        rg_poses = np.array(sel_global["poses"], copy=True)
        rg_poses[:, 3:] = np.asarray(refined_incam["poses"])[:, 3:]
        refined_global["poses"] = rg_poses
        for k in ("left_hand_pose", "right_hand_pose"):
            if k in refined_incam:
                refined_global[k] = refined_incam[k]
        save_human_motion_data(
            refined_global, refined_incam, str(output_dir / "selected_guided_motion.npz")
        )
        saved_motions["selected_guided_motion.npz"] = (refined_global, refined_incam)
        result["diagnostics"]["fixed_object_grasp_ik"] = ik_diag
        del ik_model

    if bool(getattr(args, "genmo_finger_grasp", False)):
        from grail.optimization.finger_grasp import FingerGraspConfig, refine_finger_grasp
        from grail.models.smplx_model import setup_smplx_model

        fg_model_path = Path(args.cfg["human_model"]["smplx_model_path"])
        if not fg_model_path.is_absolute():
            fg_model_path = Path.cwd() / fg_model_path
        fg_model = setup_smplx_model(
            model_path=str(fg_model_path), flat_hand_mean=True, device="cuda"
        )
        fg_config = FingerGraspConfig(
            contact_frame=contact_frame,
            selected_hand=selected_hand,
            iterations=int(getattr(args, "genmo_finger_iterations", 150)),
            contact_weight=float(getattr(args, "genmo_finger_contact_weight", 8.0)),
            penetration_weight=float(getattr(args, "genmo_finger_penetration_weight", 300.0)),
            min_clearance=float(getattr(args, "genmo_finger_min_clearance_m", 0.0)),
            target_clearance=float(getattr(args, "genmo_finger_target_clearance_m", 0.005)),
            hand_pose_reg_weight=float(getattr(args, "genmo_finger_pose_reg_weight", 0.5)),
            wrist_reg_weight=float(getattr(args, "genmo_finger_wrist_reg_weight", 1.0)),
            elbow_reg_weight=float(getattr(args, "genmo_finger_elbow_reg_weight", 3.0)),
        )
        sel_global, sel_incam = saved_motions["selected_guided_motion.npz"]
        refined_incam, fg_diag = refine_finger_grasp(
            sel_incam, object_vertices, object_faces, object_poses,
            contact_frame, selected_hand, fg_model, fg_config, device="cuda",
        )
        # Body pose (incl. wrist/elbow) and hand pose are frame-local, identical
        # in global and in-camera; copy them from the refined in-camera motion.
        refined_global = dict(sel_global)
        rg_poses = np.array(sel_global["poses"], copy=True)
        rg_poses[:, 3:] = np.asarray(refined_incam["poses"])[:, 3:]
        refined_global["poses"] = rg_poses
        for k in ("left_hand_pose", "right_hand_pose"):
            if k in refined_incam:
                refined_global[k] = refined_incam[k]
        save_human_motion_data(
            refined_global, refined_incam, str(output_dir / "selected_guided_motion.npz")
        )
        saved_motions["selected_guided_motion.npz"] = (refined_global, refined_incam)
        result["diagnostics"]["finger_grasp"] = fg_diag
        del fg_model

    if bool(getattr(args, "genmo_skip_poc_preview_render", False)):
        result["diagnostics"]["poc_preview_render_skipped"] = True
        torch.save(result["sampling_noise"], output_dir / "sampling_noise.pt")
        with (output_dir / "diagnostics.json").open("w") as handle:
            json.dump(
                _json_ready(result["diagnostics"]), handle, indent=2, allow_nan=False
            )
        print(f"Stage 3.5 sampling outputs: {output_dir}")
        return output_dir

    model_path = Path(args.cfg["human_model"]["smplx_model_path"])
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    # §16: render every candidate (baseline / arm-only / whole-body / selected)
    # so the acceptance comparison is honest, not just initial vs selected.
    preview_specs = [
        ("initial_genmo", "initial", "initial_genmo_motion.npz", "Initial GENMO"),
        ("arm_only_guided", "arm_only", "arm_only_guided_motion.npz", "Arm-only Guided"),
        ("whole_body_guided", "whole_body", "whole_body_guided_motion.npz", "Whole-body Guided"),
        ("selected_guided", "selected", "selected_guided_motion.npz", "Selected Guided"),
    ]
    preview_specs = [
        (vid, key, fn, label)
        for vid, key, fn, label in preview_specs
        if result.get(key) is not None and fn in saved_motions
    ]
    _attach_smplx_render_meshes(
        tuple((result[key], saved_motions[fn][1]) for _, key, fn, _ in preview_specs),
        model_path,
    )
    c2g_R = np.asarray(
        torch.as_tensor(result["coordinate_transform"]["camera_to_genmo_global_R"]).cpu(),
        dtype=np.float32,
    )
    c2g_t = np.asarray(
        torch.as_tensor(result["coordinate_transform"]["camera_to_genmo_global_t"]).cpu(),
        dtype=np.float32,
    )
    for _, key, _, _ in preview_specs:
        prediction = result[key]
        prediction["smplx_joints_global"] = (
            np.asarray(prediction["smplx_joints_incam"], dtype=np.float32)
            @ c2g_R.T
            + c2g_t
        )
    if human_masks:
        result["diagnostics"]["mesh_mask_alignment"] = {
            "initial": mesh_mask_alignment_metrics(
                result["initial"], result["initial"]["K_fullimg"], human_masks
            ),
            "guided": mesh_mask_alignment_metrics(
                result["selected"], result["selected"]["K_fullimg"], human_masks
            ),
        }
    result["diagnostics"]["human_renderer"] = "full_smplx_10475_orange_mesh"
    result["diagnostics"]["top_view_coordinate_system"] = (
        "GENMO ground plane; screen-right=camera-right; screen-up=camera-forward"
    )

    torch.save(result["sampling_noise"], output_dir / "sampling_noise.pt")
    with (output_dir / "diagnostics.json").open("w") as handle:
        json.dump(_json_ready(result["diagnostics"]), handle, indent=2, allow_nan=False)

    source_reader = imageio.get_reader(str(video_file))
    fps = float(source_reader.get_meta_data().get("fps", 30.0))
    source_reader.close()

    front_videos = {}
    for vid, key, _, label in preview_specs:
        front = output_dir / f"{vid}.mp4"
        render_guided_motion(
            str(video_file), str(front), result[key], object_vertices, object_faces,
            object_poses, result["coordinate_transform"], contact_frame, selected_hand,
            label, human_mask_dir=human_mask_dir, human_masks=human_masks,
        )
        front_videos[key] = front
        # Per-candidate top view: guided candidates are shown against the
        # baseline; the baseline itself is shown against the selected motion.
        other = result["selected"] if key == "initial" else result["initial"]
        render_top_view_comparison(
            str(output_dir / f"{vid}_top_view.mp4"),
            result[key] if key == "initial" else result["initial"],
            other if key == "initial" else result[key],
            object_vertices, object_poses, result["coordinate_transform"],
            result["object_surface_target"], contact_frame, selected_hand, fps,
        )
    # Backward-compatible aliases (guided_genmo == selected) + main comparisons.
    guided_video = output_dir / "guided_genmo.mp4"
    initial_video = front_videos.get("initial", output_dir / "initial_genmo.mp4")
    if "selected" in front_videos:
        import shutil as _shutil
        _shutil.copyfile(str(front_videos["selected"]), str(guided_video))
    make_side_by_side(str(initial_video), str(guided_video), str(output_dir / "initial_vs_guided.mp4"))
    render_top_view_comparison(
        str(output_dir / "initial_vs_guided_top.mp4"),
        result["initial"],
        result["selected"],
        object_vertices,
        object_poses,
        result["coordinate_transform"],
        result["object_surface_target"],
        contact_frame,
        selected_hand,
        fps,
    )

    diagnostics = result["diagnostics"]
    print("Stage 3.5 coordinate check")
    print(f"  contact frame: {diagnostics['contact_frame']}")
    print(f"  selected hand: {diagnostics['selected_hand']}")
    print(f"  initial palm position: {diagnostics['initial_palm_position']}")
    print(f"  object center: {diagnostics['object_center']}")
    print(f"  closest surface point: {diagnostics['closest_surface_point']}")
    print(f"  initial palm-surface distance: {diagnostics['initial_contact_error_m']:.6f} m")
    print(f"  human/object coordinate system: {diagnostics['human_object_coordinate_system']}")
    print(f"  human/object scale: {diagnostics['human_scale']}/{diagnostics['object_scale']}")
    print(f"  outputs: {output_dir}")
    return output_dir
