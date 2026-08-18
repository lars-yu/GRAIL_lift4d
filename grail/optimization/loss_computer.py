"""Loss computation for HOI optimization, extracted from HOIOptimizer."""

import torch
from pytorch3d.structures import Meshes

from grail.optimization.loss_terms import (
    approach_monotonic_loss,
    bidirectional_chamfer_loss,
    contact_center_loss,
    contact_depth_loss,
    contact_distribution_smoothness_loss,
    contact_loss,
    contact_smoothness_loss,
    contact_anchor_distance_loss,
    ground_loss,
    huber_loss,
    foundationpose_depth_anchor_loss,
    keypoint_loss,
    lift4d_depth_trend_loss,
    lift4d_depth_acceleration_loss,
    lift4d_depth_velocity_loss,
    l1_loss,
    human_silhouette_loss,
    penetration_loss,
    reg_loss,
    smoothness_loss,
    object_depth_smoothness_loss,
    relative_translation_consistency_loss,
)
from grail.optimization.approach import hand_to_mesh_surface_distance
from grail.rendering.camera import project_world_to_screen, unproject_depth_map_to_world


class LossComputer:
    """Computes all loss terms for HOI optimization."""

    def __init__(
        self,
        cameras,
        human_model,
        device,
        get_contact_labels_for_frame_fn,
        num_body_joints,
        logger,
    ):
        self.cameras = cameras
        self.human_model = human_model
        self.device = device
        self.get_contact_labels_for_frame = get_contact_labels_for_frame_fn
        self.num_body_joints = num_body_joints
        self.logger = logger
        self._depth_loss_cache = None

    # ── Dispatch ─────────────────────────────────────────────────────────────

    _LOSS_FN = {
        "contact": "_contact_loss",
        "keypoint_tracking": "_keypoint_tracking_loss",
        "body_keypoint_reprojection": "_body_keypoint_reprojection_loss",
        "hand_keypoint_reprojection": "_hand_keypoint_reprojection_loss",
        "human_silhouette": "_human_silhouette_loss",
        "ground": "_ground_loss",
        "human_global_init_reg": "_human_global_init_reg_loss",
        "human_smoothness": "_human_smoothness_loss",
        "human_traj_reg": "_human_traj_reg_loss",
        "human_pose_reg": "_human_pose_reg_loss",
        "human_foot_contact": "_human_foot_contact_loss",
        "verts_tracking": "_verts_tracking_loss",
        "obj_smoothness": "_obj_smoothness_loss",
        "obj_traj_reg": "_obj_traj_reg_loss",
        "obj_rot_reg": "_obj_rot_reg_loss",
        "depth_pointcloud": "_depth_pointcloud_loss",
        "contact_smoothness": "_contact_smoothness_loss",
        "contact_distribution_smoothness": "_contact_distribution_smoothness_loss",
        "obj_precontact_reg": "_obj_precontact_reg_loss",
        "penetration": "_penetration_loss",
        "lift4d_depth": "_lift4d_depth_loss",
        "lift4d_velocity": "_lift4d_velocity_loss",
        "lift4d_acceleration": "_lift4d_acceleration_loss",
        "fp_depth_anchor": "_fp_depth_anchor_loss",
        "obj_depth_smoothness": "_obj_depth_smoothness_loss",
        "lift4d_depth_scale_reg": "_lift4d_depth_scale_reg_loss",
        "contact_anchor": "_contact_anchor_loss",
        "approach_monotonic": "_approach_monotonic_loss",
        "postcontact_relative": "_postcontact_relative_loss",
        "hand_ray_ik": "_hand_ray_ik_loss",
        "hand_path": "_hand_path_loss",
        "hand_velocity": "_hand_velocity_loss",
        "hand_acceleration": "_hand_acceleration_loss",
        "hand_jerk": "_hand_jerk_loss",
        "pose_residual_acceleration": "_pose_residual_acceleration_loss",
        "boundary_position": "_boundary_position_loss",
        "boundary_velocity": "_boundary_velocity_loss",
        "pose_residual_continuity": "_pose_residual_continuity_loss",
        "object_static_pre_motion": "_object_static_pre_motion_loss",
    }

    def compute_loss(self, data, pred, loss_cfg):
        total_loss = 0.0
        loss_dict = {}
        self.last_weighted_terms = {}
        for loss_name, cfg in loss_cfg.items():
            if cfg.get("enabled", True) is False:
                continue
            weight = cfg["weight"]
            fn_name = self._LOSS_FN.get(loss_name)
            if fn_name is None:
                raise ValueError(f"Invalid loss name: {loss_name}")
            result = getattr(self, fn_name)(data, pred, cfg, weight)
            if isinstance(result, tuple):
                raw_loss, weighted_loss = result
                total_loss += weighted_loss
                loss_dict[f"{loss_name}_raw"] = raw_loss.detach().item()
                loss_dict[f"{loss_name}_weighted"] = weighted_loss.detach().item()
                self.last_weighted_terms[loss_name] = weighted_loss
            else:
                weighted_loss = result
                total_loss += weighted_loss
                raw_loss = weighted_loss / float(weight) if float(weight) != 0.0 else weighted_loss
                loss_dict[f"{loss_name}_raw"] = raw_loss.detach().item()
                loss_dict[f"{loss_name}_weighted"] = weighted_loss.detach().item()
                self.last_weighted_terms[loss_name] = weighted_loss
        return total_loss, loss_dict

    # ── Individual loss methods ──────────────────────────────────────────────

    def _contact_loss(self, data, pred, cfg, weight):
        human_verts_seq = pred.human.verts_seq
        obj_verts_seq = pred.obj.verts_seq

        inter_start_idx = data.inter_start_idx
        inter_end_idx = data.inter_end_idx
        depth_only = cfg.get("depth_only", False)
        # Skip a (frame, body-part) contact term when the closest human-object
        # 3D distance exceeds this threshold — guards against spurious contact
        # labels (e.g. predicted "right hand" when the hand is nowhere near the
        # object). None disables the gate (default).
        max_contact_dist = cfg.get("max_contact_dist", None)
        contact_loss_fn = contact_loss
        if cfg.get("use_center_loss", False):
            contact_loss_fn = contact_center_loss
        elif depth_only:
            contact_loss_fn = contact_depth_loss

        def _too_far(human_verts, obj_verts):
            if max_contact_dist is None or human_verts.numel() == 0 or obj_verts.numel() == 0:
                return False
            min_dist = torch.cdist(human_verts.float(), obj_verts.float()).min()
            return bool(min_dist > max_contact_dist)

        if cfg["duration"] == "start":
            loss = 0.0
            count = 0
            window_size = cfg.get("window_size", 8)
            ws = max(0, inter_start_idx - window_size // 2)
            we = min(len(human_verts_seq), inter_start_idx + window_size // 2 + 1)
            for i in range(ws, we):
                frame_labels = self.get_contact_labels_for_frame(i)
                if frame_labels is None:
                    continue
                for label in frame_labels:
                    cv = self.human_model.get_verts_segment(human_verts_seq, [label])
                    if _too_far(cv[i], obj_verts_seq[i]):
                        continue
                    if depth_only:
                        loss += weight * contact_loss_fn(cv[i], obj_verts_seq[i], self.cameras)
                    else:
                        loss += weight * contact_loss_fn(cv[i], obj_verts_seq[i])
                    count += 1
            if count > 0:
                loss /= count
            else:
                loss = human_verts_seq.new_zeros(())
        elif cfg["duration"] == "all":
            loss = 0.0
            count = 0
            for i in range(inter_start_idx, inter_end_idx):
                frame_labels = self.get_contact_labels_for_frame(i)
                if frame_labels is None:
                    continue
                for label in frame_labels:
                    cv = self.human_model.get_verts_segment(human_verts_seq, [label])
                    if _too_far(cv[i], obj_verts_seq[i]):
                        continue
                    if depth_only:
                        loss += weight * contact_loss_fn(cv[i], obj_verts_seq[i], self.cameras)
                    else:
                        loss += weight * contact_loss_fn(cv[i], obj_verts_seq[i])
                    count += 1
            if count > 0:
                loss /= count
            else:
                loss = human_verts_seq.new_zeros(())
        else:
            raise ValueError(f"Invalid duration: {cfg['duration']}")

        if data.is_static_obj:
            loss = 0.0 * loss
        return loss

    def _keypoint_tracking_loss(self, data, pred, cfg, weight):
        pred_body_kp = pred.human.body_keypoints_seq
        gt_body_kp = data.human.body_keypoints_seq
        gt_body_conf = gt_body_kp[:, :, 2]
        gt_body_kp = gt_body_kp[:, :, :2]

        pred_hand_kp = pred.human.hand_keypoints_seq
        gt_hand_kp = data.human.hand_keypoints_seq
        gt_hand_conf = gt_hand_kp[:, :, 2]
        gt_hand_kp = gt_hand_kp[:, :, :2]

        duration = cfg.get("duration", "all")
        if duration == "start":
            inter_start_idx = data.inter_start_idx
            window_size = 8
            ws = max(0, inter_start_idx - window_size // 2)
            we = min(len(pred_body_kp), inter_start_idx + window_size // 2)
            pred_body_kp = pred_body_kp[ws:we]
            gt_body_kp = gt_body_kp[ws:we]
            gt_body_conf = gt_body_conf[ws:we]
            pred_hand_kp = pred_hand_kp[ws:we]
            gt_hand_kp = gt_hand_kp[ws:we]
            gt_hand_conf = gt_hand_conf[ws:we]
        elif duration != "all":
            raise ValueError(f"Invalid duration: {duration}")

        loss_body = keypoint_loss(
            pred_body_kp.reshape(-1, 2), gt_body_kp.reshape(-1, 2), gt_body_conf.reshape(-1)
        )
        loss_hand = keypoint_loss(
            pred_hand_kp.reshape(-1, 2),
            gt_hand_kp.reshape(-1, 2),
            gt_hand_conf.reshape(-1),
            conf_thres=0.2,
        )
        beta = cfg.get("beta", 0.3)
        return weight * (loss_body + beta * loss_hand)

    def _human_silhouette_loss(self, data, pred, cfg, weight):
        target_masks = torch.stack([
            torch.as_tensor(mask, device=pred.human.body_keypoints_seq.device)
            for mask in data.human.masks
        ])
        sample_count = int(cfg.get("silhouette_num_samples", 2048))
        silhouette_vertices = pred.human.verts_seq
        if silhouette_vertices.shape[1] > sample_count:
            sample_idx = torch.linspace(
                0, silhouette_vertices.shape[1] - 1, sample_count,
                device=silhouette_vertices.device,
            ).long()
            silhouette_vertices = silhouette_vertices[:, sample_idx]
        projected_vertices = project_world_to_screen(
            silhouette_vertices.reshape(-1, 3), self.cameras
        ).reshape(data.frame_num, -1, 3)[..., :2]
        raw = human_silhouette_loss(
            projected_vertices,
            target_masks,
            (int(data.camera.frame_height), int(data.camera.frame_width)),
            output_size=tuple(cfg.get("output_size", (64, 64))),
            sigma=float(cfg.get("sigma", 1.5)),
            boundary_weight=float(cfg.get("boundary_weight", 0.25)),
        )
        return raw, float(weight) * raw

    def _body_keypoint_reprojection_loss(self, data, pred, cfg, weight):
        gt = data.human.body_keypoints_seq
        raw = keypoint_loss(
            pred.human.body_keypoints_seq.reshape(-1, 2),
            gt[:, :, :2].reshape(-1, 2),
            gt[:, :, 2].reshape(-1),
            conf_thres=cfg.get("conf_thres", 0.2),
        )
        return raw, float(weight) * raw

    def _hand_keypoint_reprojection_loss(self, data, pred, cfg, weight):
        gt = data.human.hand_keypoints_seq
        raw = keypoint_loss(
            pred.human.hand_keypoints_seq.reshape(-1, 2),
            gt[:, :, :2].reshape(-1, 2),
            gt[:, :, 2].reshape(-1),
            conf_thres=cfg.get("conf_thres", 0.2),
        )
        return raw, float(weight) * raw

    def _ground_loss(self, data, pred, cfg, weight):
        height = cfg.get("height", 0.14)
        gravity_axis = cfg.get("gravity_axis", "z")
        return weight * ground_loss(pred.human.verts_seq, gravity_axis=gravity_axis, height=height)

    def _human_global_init_reg_loss(self, data, pred, cfg, weight):
        pred_trans = pred.human.trans
        ref_trans = data.human.motion_data_global_init["trans"]

        pred_vel = pred_trans[1:] - pred_trans[:-1]
        ref_vel = ref_trans[1:] - ref_trans[:-1]

        return weight * reg_loss(pred_vel.norm(dim=-1), ref_vel.norm(dim=-1), use_l2=True)

    def _human_smoothness_loss(self, data, pred, cfg, weight):
        beta = cfg.get("beta", 1.0)
        return weight * smoothness_loss(pred.human.verts_seq, beta=beta)

    def _human_traj_reg_loss(self, data, pred, cfg, weight):
        res = pred.human.trans_res
        res = res.reshape(res.shape[0], 3)
        return weight * reg_loss(res, torch.zeros_like(res))

    def _human_pose_reg_loss(self, data, pred, cfg, weight):
        res = pred.human.pose_res
        frame_num = res.shape[0]
        reg_target = (
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            .reshape(1, 1, 6)
            .repeat(frame_num, self.num_body_joints, 1)
            .to(self.device)
        )
        return weight * reg_loss(res, reg_target)

    def _human_foot_contact_loss(self, data, pred, cfg, weight):
        body_joints_seq = pred.human.body_joints_seq
        foot_contact_probs = data.human.foot_contact_probs

        if foot_contact_probs is None:
            return body_joints_seq.new_zeros(())

        contact_threshold = cfg.get("threshold", 0.5)
        left_idx, right_idx = self.human_model.get_foot_joint_indices()
        left_pos = body_joints_seq[:, left_idx, :]
        right_pos = body_joints_seq[:, right_idx, :]

        left_vel = left_pos[1:] - left_pos[:-1]
        right_vel = right_pos[1:] - right_pos[:-1]

        left_contact = torch.max(foot_contact_probs[:, 0], foot_contact_probs[:, 1])
        right_contact = torch.max(foot_contact_probs[:, 2], foot_contact_probs[:, 3])

        left_w = (torch.min(left_contact[:-1], left_contact[1:]) > contact_threshold).float()
        right_w = (torch.min(right_contact[:-1], right_contact[1:]) > contact_threshold).float()

        left_loss = (left_vel.norm(dim=-1) * left_w).sum()
        right_loss = (right_vel.norm(dim=-1) * right_w).sum()
        num_contact = left_w.sum() + right_w.sum() + 1e-6
        return weight * (left_loss + right_loss) / num_contact

    def _verts_tracking_loss(self, data, pred, cfg, weight):
        pred_verts = pred.obj.verts_seq
        gt_verts = data.obj.verts_tracking_seq
        frame_num = pred_verts.shape[0]
        pred_2d = project_world_to_screen(pred_verts.reshape(-1, 3), self.cameras).reshape(
            frame_num, -1, 3
        )[:, :, :2]
        return weight * l1_loss(pred_2d, gt_verts)

    def _obj_smoothness_loss(self, data, pred, cfg, weight):
        verts = pred.obj.verts_seq
        beta = cfg.get("beta", 1.0)
        return weight * smoothness_loss(verts.reshape(verts.shape[0], -1, 3), beta=beta)

    def _obj_traj_reg_loss(self, data, pred, cfg, weight):
        pred_trans = pred.obj.trans.reshape(-1, 3)
        orig_trans = data.obj.poses[:, :3, 3].reshape(-1, 3)
        return weight * reg_loss(pred_trans, orig_trans)

    def _obj_rot_reg_loss(self, data, pred, cfg, weight):
        return weight * reg_loss(pred.obj.R, data.obj.poses[:, :3, :3])

    def _require_lift4d_depth(self, data, pred):
        prior = getattr(data, "lift4d_depth", None)
        if prior is None or pred.obj.z_cam is None:
            raise ValueError("Lift4D depth loss is enabled but no real Lift4D depth prior is loaded")
        frame_num = int(pred.obj.z_cam.shape[0])
        expected = torch.arange(frame_num, device=prior.frame_indices.device)
        if prior.frame_indices.shape != expected.shape or not torch.equal(prior.frame_indices, expected):
            raise ValueError("Lift4D frame_indices must be exactly torch.arange(frame_num)")
        if prior.prior_used.shape != expected.shape or not bool(prior.prior_used.all()):
            raise ValueError("Lift4D formal depth prior must supervise every frame without fallback")
        if int(prior.prior_used.sum()) != frame_num:
            raise AssertionError(
                f"Lift4D supervised frames: {int(prior.prior_used.sum())} / {frame_num}"
            )
        return prior

    def _lift4d_depth_loss(self, data, pred, cfg, weight):
        prior = self._require_lift4d_depth(data, pred)
        raw = lift4d_depth_trend_loss(
            pred.obj.z_cam,
            prior.z_target,
            prior.frame_weight,
            pred.obj.depth_scale,
            delta=cfg.get("delta", 0.03),
        )
        return raw, float(weight) * raw

    def _lift4d_velocity_loss(self, data, pred, cfg, weight):
        prior = self._require_lift4d_depth(data, pred)
        raw = lift4d_depth_velocity_loss(
            pred.obj.z_cam,
            prior.z_target,
            prior.frame_weight,
            pred.obj.depth_scale,
            delta=cfg.get("delta", 0.015),
        )
        return raw, float(weight) * raw

    def _lift4d_acceleration_loss(self, data, pred, cfg, weight):
        self._require_lift4d_depth(data, pred)
        raw = lift4d_depth_acceleration_loss(pred.obj.z_cam)
        return raw, float(weight) * raw

    def _contact_hand_labels(self, data):
        mapping = {
            "right": ["R_Hand"],
            "left": ["L_Hand"],
            "both": ["L_Hand", "R_Hand"],
        }
        hand = str(data.contact_hand).lower()
        if hand not in mapping:
            raise ValueError(f"contact.hand must be left/right/both, got {data.contact_hand!r}")
        return mapping[hand]

    def _hand_surface_distances(self, data, pred, frame_indices, cfg, *, detach_object=False):
        hand_seq = self.human_model.get_verts_segment(
            pred.human.verts_seq, self._contact_hand_labels(data)
        )
        object_seq = pred.obj.verts_seq.detach() if detach_object else pred.obj.verts_seq
        distances = []
        for frame_idx in frame_indices:
            i = int(frame_idx)
            distances.append(
                hand_to_mesh_surface_distance(
                    hand_seq[i],
                    object_seq[i],
                    data.obj.faces,
                    top_k=cfg.get("top_k", 32),
                )
            )
        return torch.stack(distances)

    def _contact_anchor_loss(self, data, pred, cfg, weight):
        if data.object_motion_state is None:
            if data.contact_frame is None:
                raise ValueError("contact_anchor requires an explicit contact frame")
            radius = int(cfg.get("frame_radius", 2))
            start = max(0, int(data.contact_frame) - radius)
            end = min(data.frame_num, int(data.contact_frame) + radius + 1)
            distances = self._hand_surface_distances(data, pred, range(start, end), cfg)
            raw = contact_anchor_distance_loss(
                distances,
                target=cfg.get("target_distance", 0.02),
                delta=cfg.get("delta", 0.02),
            )
            return raw, float(weight) * raw
        move_start = int(data.object_motion_state.move_start_frame)
        phase = str(cfg.get("phase", "moving"))
        if phase == "precontact":
            start, end = max(0, move_start - int(data.approach_window)), move_start
        elif phase == "joint":
            start = max(0, move_start - int(cfg.get("overlap_frames", 5)))
            end = data.frame_num - 1
        else:
            start, end = move_start, data.frame_num - 1
        frame_indices = torch.arange(start, end + 1, device=pred.obj.trans.device)
        distances = self._hand_surface_distances(
            data, pred, frame_indices, cfg, detach_object=True
        )
        target_distance = float(cfg.get("target_distance", 0.02))
        if phase in {"precontact", "joint"}:
            ramp = pred.human.approach_ramp[start : end + 1].detach()
            initial_distance = data.hand_approach_initial_distance
            if initial_distance is None:
                initial_distance = float(distances[0].detach())
            targets = float(initial_distance) + ramp * (
                target_distance - float(initial_distance)
            )
            if phase == "joint":
                frames = torch.arange(start, end + 1, device=distances.device)
                targets = torch.where(
                    frames > move_start,
                    torch.full_like(targets, target_distance),
                    targets,
                )
            terms = torch.nn.functional.huber_loss(
                distances,
                targets,
                delta=cfg.get("delta", 0.02),
                reduction="none",
            )
            weights = ramp.clamp_min(1e-4)
            raw = (weights * terms).sum() / weights.sum()
        else:
            targets = torch.full_like(distances, target_distance)
            raw = huber_loss(distances - targets, delta=cfg.get("delta", 0.02))
        return raw, float(weight) * raw

    def _approach_monotonic_loss(self, data, pred, cfg, weight):
        move_start = (
            int(data.object_motion_state.move_start_frame)
            if data.object_motion_state is not None
            else int(data.contact_frame)
        )
        start = max(0, move_start - int(data.approach_window))
        end = move_start + 1
        distances = self._hand_surface_distances(
            data, pred, range(start, end), cfg, detach_object=True
        )
        raw = approach_monotonic_loss(distances)
        return raw, float(weight) * raw

    def _postcontact_relative_loss(self, data, pred, cfg, weight):
        hand_seq = self.human_model.get_verts_segment(
            pred.human.verts_seq, self._contact_hand_labels(data)
        )
        if data.object_motion_state is None:
            start = int(data.contact_frame)
            relative = hand_seq[start:].mean(dim=1) - pred.obj.trans[start:].detach()
            raw = relative_translation_consistency_loss(
                relative, delta=cfg.get("delta", 0.01)
            )
            return raw, float(weight) * raw
        start = int(data.object_motion_state.move_start_frame)
        relative = hand_seq[start:].mean(dim=1) - pred.obj.trans[start:].detach()
        anchor = (
            data.boundary_relative_anchor.detach()
            if data.boundary_relative_anchor is not None
            else relative[0].detach()
        )
        relative_error = torch.linalg.norm(relative - anchor[None], dim=-1)
        hand_velocity = hand_seq[start:].mean(dim=1)
        object_velocity = pred.obj.trans[start:].detach()
        velocity_error = torch.linalg.norm(
            (hand_velocity[1:] - hand_velocity[:-1])
            - (object_velocity[1:] - object_velocity[:-1]), dim=-1
        )
        raw = huber_loss(relative_error, delta=cfg.get("delta", 0.01))
        if velocity_error.numel():
            raw = raw + float(cfg.get("velocity_weight", 1.0)) * huber_loss(
                velocity_error, delta=cfg.get("delta", 0.01)
            )
        return raw, float(weight) * raw

    def _hand_ray_ik_loss(self, data, pred, cfg, weight):
        target = data.hand_ray_target_world
        if target is None:
            raise ValueError("hand_ray_ik requires a real camera-ray target")
        hand = pred.human.hand_joints_seq
        half = hand.shape[1] // 2
        if data.contact_hand == "left":
            hand = hand[:, :half]
        elif data.contact_hand == "right":
            hand = hand[:, half:]
        predicted = hand.mean(dim=1)
        error = torch.linalg.norm(predicted - target.detach(), dim=-1)
        start, end = self._hand_loss_window(data, cfg)
        selected_error = error[start:end]
        if selected_error.numel() == 0:
            return pred.human.verts_seq.new_zeros(()), pred.human.verts_seq.new_zeros(())
        raw_terms = torch.nn.functional.huber_loss(
            selected_error,
            torch.zeros_like(selected_error),
            delta=cfg.get("delta", 0.03),
            reduction="none",
        )
        if data.hand_ray_ramp is not None and str(cfg.get("phase", "all")) == "precontact":
            weights = data.hand_ray_ramp[start:end].detach().clamp_min(1e-4)
            raw = (weights * raw_terms).sum() / weights.sum()
        else:
            raw = raw_terms.mean()
        return raw, float(weight) * raw

    def _hand_loss_window(self, data, cfg):
        move_start = int(
            data.object_motion_state.move_start_frame
            if data.object_motion_state is not None
            else data.contact_frame
        )
        phase = str(cfg.get("phase", "all"))
        if phase == "precontact":
            overlap = int(cfg.get("overlap_frames", 5))
            return max(0, move_start - int(data.approach_window) - overlap), move_start + 1
        overlap = int(cfg.get("overlap_frames", 5))
        return max(0, move_start - overlap), data.frame_num

    def _selected_hand_center(self, data, pred):
        hand = self.human_model.get_verts_segment(
            pred.human.verts_seq, self._contact_hand_labels(data)
        )
        return hand.mean(dim=1)

    def _hand_path_loss(self, data, pred, cfg, weight):
        if data.hand_ray_target_world is None:
            raise ValueError("hand_path requires refreshed hand ray targets")
        start, end = self._hand_loss_window(data, cfg)
        actual = self._selected_hand_center(data, pred)[start:end]
        target = data.hand_ray_target_world[start:end].detach()
        raw = torch.nn.functional.huber_loss(
            actual, target, delta=cfg.get("delta", 0.03), reduction="mean"
        )
        return raw, float(weight) * raw

    def _hand_velocity_loss(self, data, pred, cfg, weight):
        start, end = self._hand_loss_window(data, cfg)
        actual = self._selected_hand_center(data, pred)[start:end]
        target = data.hand_ray_target_world[start:end].detach()
        if actual.shape[0] < 2:
            zero = pred.human.verts_seq.new_zeros(())
            return zero, zero
        raw = torch.nn.functional.huber_loss(
            actual[1:] - actual[:-1], target[1:] - target[:-1],
            delta=cfg.get("delta", 0.02), reduction="mean"
        )
        return raw, float(weight) * raw

    def _hand_acceleration_loss(self, data, pred, cfg, weight):
        start, end = self._hand_loss_window(data, cfg)
        actual = self._selected_hand_center(data, pred)[start:end]
        target = data.hand_ray_target_world[start:end].detach()
        if actual.shape[0] < 3:
            zero = pred.human.verts_seq.new_zeros(())
            return zero, zero
        actual_acc = actual[2:] - 2.0 * actual[1:-1] + actual[:-2]
        target_acc = target[2:] - 2.0 * target[1:-1] + target[:-2]
        raw = torch.nn.functional.huber_loss(
            actual_acc, target_acc, delta=cfg.get("delta", 0.02), reduction="mean"
        )
        return raw, float(weight) * raw

    def _hand_jerk_loss(self, data, pred, cfg, weight):
        start, end = self._hand_loss_window(data, cfg)
        actual = self._selected_hand_center(data, pred)[start:end]
        target = data.hand_ray_target_world[start:end].detach()
        if actual.shape[0] < 4:
            zero = pred.human.verts_seq.new_zeros(())
            return zero, zero
        actual_jerk = actual[3:] - 3.0 * actual[2:-1] + 3.0 * actual[1:-2] - actual[:-3]
        target_jerk = target[3:] - 3.0 * target[2:-1] + 3.0 * target[1:-2] - target[:-3]
        raw = torch.nn.functional.huber_loss(
            actual_jerk, target_jerk, delta=cfg.get("delta", 0.02), reduction="mean"
        )
        return raw, float(weight) * raw

    def _pose_residual_acceleration_loss(self, data, pred, cfg, weight):
        start, end = self._hand_loss_window(data, cfg)
        residual = pred.human.pose_res[start:end]
        if residual.shape[0] < 3:
            zero = residual.new_zeros(())
            return zero, zero
        joints = [0, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        joints = [i for i in joints if i < residual.shape[1]]
        accel = residual[2:, joints] - 2.0 * residual[1:-1, joints] + residual[:-2, joints]
        raw = accel.abs().mean()
        return raw, float(weight) * raw

    def _boundary_position_loss(self, data, pred, cfg, weight):
        if data.boundary_hand_position_at_move is None:
            return pred.human.verts_seq.new_zeros(()), pred.human.verts_seq.new_zeros(())
        actual = self._selected_hand_center(data, pred)[int(data.object_motion_state.move_start_frame)]
        raw = torch.nn.functional.huber_loss(
            actual, data.boundary_hand_position_at_move.detach(), delta=cfg.get("delta", 0.01)
        )
        return raw, float(weight) * raw

    def _boundary_velocity_loss(self, data, pred, cfg, weight):
        if data.boundary_hand_velocity_at_move is None:
            return pred.human.verts_seq.new_zeros(()), pred.human.verts_seq.new_zeros(())
        t = int(data.object_motion_state.move_start_frame)
        hand = self._selected_hand_center(data, pred)
        actual = hand[t] - hand[t - 1]
        raw = torch.nn.functional.huber_loss(
            actual, data.boundary_hand_velocity_at_move.detach(), delta=cfg.get("delta", 0.01)
        )
        return raw, float(weight) * raw

    def _pose_residual_continuity_loss(self, data, pred, cfg, weight):
        if data.boundary_pose_residual_at_move is None:
            return pred.human.verts_seq.new_zeros(()), pred.human.verts_seq.new_zeros(())
        t = int(data.object_motion_state.move_start_frame)
        raw = torch.nn.functional.huber_loss(
            pred.human.pose_res[t], data.boundary_pose_residual_at_move.detach(),
            delta=cfg.get("delta", 0.01)
        )
        return raw, float(weight) * raw

    def _object_static_pre_motion_loss(self, data, pred, cfg, weight):
        if data.object_motion_state is None:
            raise ValueError("object_static_pre_motion requires object motion state")
        move_start = int(data.object_motion_state.move_start_frame)
        if move_start < 2:
            raw = pred.obj.z_cam.new_zeros(())
        else:
            static_target = pred.obj.z_cam[:move_start].detach().median()
            raw = huber_loss(
                pred.obj.z_cam[:move_start] - static_target,
                delta=cfg.get("delta", 0.01),
            )
        return raw, float(weight) * raw

    def _fp_depth_anchor_loss(self, data, pred, cfg, weight):
        self._require_lift4d_depth(data, pred)
        raw = foundationpose_depth_anchor_loss(
            pred.obj.z_cam,
            data.obj.poses_cam[:, 2, 3],
            delta=cfg.get("delta", 0.02),
        )
        return raw, float(weight) * raw

    def _obj_depth_smoothness_loss(self, data, pred, cfg, weight):
        self._require_lift4d_depth(data, pred)
        raw = object_depth_smoothness_loss(pred.obj.z_cam, delta=cfg.get("delta", 0.015))
        return raw, float(weight) * raw

    def _lift4d_depth_scale_reg_loss(self, data, pred, cfg, weight):
        self._require_lift4d_depth(data, pred)
        raw = (pred.obj.depth_scale - 1.0).square()
        return raw, float(weight) * raw

    def _depth_pointcloud_loss(self, data, pred, cfg, weight):
        if self._depth_loss_cache is None:
            num_gt_samples = cfg.get("num_gt_samples", 3000)
            self._build_depth_loss_cache(data, pred, num_gt_samples=num_gt_samples)
        cache = self._depth_loss_cache

        human_verts_seq = pred.human.verts_seq
        obj_verts_seq = pred.obj.verts_seq
        trim_pct = cfg.get("trim_pct", 0.2)
        include_human = bool(cfg.get("include_human", True))
        include_object = bool(cfg.get("include_object", True))
        if not include_human and not include_object:
            raise ValueError("depth_pointcloud must include human, object, or both")

        frame_num = human_verts_seq.shape[0]
        interval = cfg.get("interval", 1)
        if cfg.get("require_full_frame", False) and int(interval) != 1:
            raise ValueError(
                f"Formal depth_pointcloud optimization requires interval=1, got {interval}"
            )
        all_frames = list(range(frame_num))
        if interval > 1:
            start_offset = torch.randint(0, interval, (1,)).item()
            frame_indices = all_frames[start_offset::interval]
        else:
            frame_indices = all_frames

        loss = 0.0
        count = 0
        for i in frame_indices:
            h_visible = cache["human_vis_masks"][i]
            o_visible = cache["obj_vis_masks"][i]
            pred_human_visible = human_verts_seq[i][h_visible]
            pred_obj_visible = obj_verts_seq[i][o_visible]
            gt_human_pc = cache["gt_human_pcs"][i]
            gt_obj_pc = cache["gt_obj_pcs"][i]

            frame_loss = 0.0
            pairs = []
            if include_human:
                pairs.append((pred_human_visible, gt_human_pc))
            if include_object:
                pairs.append((pred_obj_visible, gt_obj_pc))
            for pred_vis, gt_pc in pairs:
                if pred_vis.shape[0] > 0 and gt_pc.shape[0] > 0:
                    frame_loss = frame_loss + bidirectional_chamfer_loss(
                        pred_vis, gt_pc, trim_pct=trim_pct
                    )
            loss += weight * frame_loss
            count += 1

        if count > 0:
            loss /= count
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        return loss

    def _contact_smoothness_loss(self, data, pred, cfg, weight):
        human_verts_seq = pred.human.verts_seq
        obj_verts_seq = pred.obj.verts_seq
        inter_start_idx = data.inter_start_idx
        inter_end_idx = data.inter_end_idx

        windows = self._build_windows(
            inter_start_idx, inter_end_idx, cfg.get("window", 8), cfg.get("stride", 2)
        )

        loss_accum = 0.0
        win_count = 0
        for ws, we in windows:
            frame_labels = self.get_contact_labels_for_frame((ws + we) // 2)
            if frame_labels is None:
                continue
            contact_verts_seq = self.human_model.get_verts_segment(
                human_verts_seq, frame_labels[:2]
            )
            loss_accum = loss_accum + contact_smoothness_loss(
                verts_A_seq=contact_verts_seq[ws:we], verts_B_seq=obj_verts_seq[ws:we]
            )
            win_count += 1

        if win_count == 0:
            return human_verts_seq.new_zeros(())
        return weight * (loss_accum / win_count)

    def _contact_distribution_smoothness_loss(self, data, pred, cfg, weight):
        human_verts_seq = pred.human.verts_seq
        obj_verts_seq = pred.obj.verts_seq
        inter_start_idx = data.inter_start_idx
        inter_end_idx = data.inter_end_idx
        temperature = cfg.get("temperature", 100.0)
        num_obj_verts = cfg.get("num_obj_verts", 2000)

        windows = self._build_windows(
            inter_start_idx, inter_end_idx, cfg.get("window", 8), cfg.get("stride", 2)
        )

        loss_accum = 0.0
        win_count = 0
        for ws, we in windows:
            frame_labels = self.get_contact_labels_for_frame((ws + we) // 2)
            if frame_labels is None:
                continue
            contact_verts_seq = self.human_model.get_verts_segment(
                human_verts_seq, frame_labels[:2]
            )
            loss_accum = loss_accum + contact_distribution_smoothness_loss(
                human_contact_verts_seq=contact_verts_seq[ws:we],
                obj_verts_seq=obj_verts_seq[ws:we],
                temperature=temperature,
                num_obj_verts=num_obj_verts,
            )
            win_count += 1

        if win_count == 0:
            return human_verts_seq.new_zeros(())
        return weight * (loss_accum / win_count)

    def _obj_precontact_reg_loss(self, data, pred, cfg, weight):
        orig = data.obj.verts_seq
        pred_verts = pred.obj.verts_seq
        inter_start_idx = data.inter_start_idx
        return weight * l1_loss(
            pred_verts[:inter_start_idx], orig[0:1].repeat(inter_start_idx, 1, 1)
        )

    def _penetration_loss(self, data, pred, cfg, weight):
        human_verts_seq = pred.human.verts_seq
        obj_t = pred.obj.trans
        obj_R = pred.obj.R
        obj_sdf = data.obj_sdf
        frame_num = human_verts_seq.shape[0]

        loss_accum = 0.0
        count = 0
        for i in range(frame_num):
            human_verts_centered = human_verts_seq[i] - obj_t[i].unsqueeze(0)
            human_verts_ocs = torch.matmul(human_verts_centered, obj_R[i])
            loss_accum += penetration_loss(verts_A=human_verts_ocs, sdf_B=obj_sdf)
            count += 1

        return weight * (loss_accum / count) if count > 0 else human_verts_seq.new_zeros(())

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_windows(start, end, window, stride):
        if window is None or window >= (end - start):
            return [(start, end)]
        windows = []
        t = start
        while t < end:
            w_end = min(end, t + window)
            if w_end - t >= 2:
                windows.append((t, w_end))
            if t + stride >= end:
                break
            t = t + stride
        return windows

    def _build_depth_loss_cache(self, data, pred, num_gt_samples=3000):
        """Pre-compute GT point clouds and vertex visibility masks for depth_pointcloud loss."""
        from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings

        full_h = int(data.camera.frame_height)
        full_w = int(data.camera.frame_width)
        half_h = full_h // 2
        half_w = full_w // 2
        frame_num = data.frame_num

        human_faces = data.human.faces
        obj_faces = data.obj.faces
        human_verts_seq = pred.human.verts_seq
        obj_verts_seq = pred.obj.verts_seq

        raster_settings = RasterizationSettings(
            image_size=(half_h, half_w),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=50000,
            cull_backfaces=True,
        )
        rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings)
        depth_tol = 0.02

        cache = {"gt_human_pcs": {}, "gt_obj_pcs": {}, "human_vis_masks": {}, "obj_vis_masks": {}}
        self.logger.info(f"Building depth loss cache for {frame_num} frames...")

        for i in range(frame_num):
            human_mask = torch.from_numpy(data.human.masks[i]).squeeze().bool().to(self.device)
            obj_mask = torch.from_numpy(data.obj.masks[i]).squeeze().bool().to(self.device)

            depth_map = data.depth_maps[i]
            if not isinstance(depth_map, torch.Tensor):
                depth_map = torch.tensor(depth_map, dtype=torch.float32)
            depth_map = depth_map.to(self.device)

            # GT point clouds via unprojection (full resolution)
            for mask, pc_key in [(human_mask, "gt_human_pcs"), (obj_mask, "gt_obj_pcs")]:
                valid = mask & (depth_map > 0)
                ys, xs = torch.where(valid)
                if len(xs) > 0:
                    pts = torch.stack([xs.float(), ys.float(), depth_map[ys, xs]], dim=1)
                    pc = unproject_depth_map_to_world(pts, self.cameras).detach()
                    if pc.shape[0] > num_gt_samples:
                        pc = pc[torch.randperm(pc.shape[0], device=self.device)[:num_gt_samples]]
                    cache[pc_key][i] = pc
                else:
                    cache[pc_key][i] = torch.zeros((0, 3), device=self.device)

            # Vertex visibility masks via rasterization (half resolution)
            with torch.no_grad():
                for verts, faces, vis_key in [
                    (human_verts_seq[i], human_faces, "human_vis_masks"),
                    (obj_verts_seq[i], obj_faces, "obj_vis_masks"),
                ]:
                    mesh = Meshes(verts=[verts], faces=[faces])
                    zbuf = rasterizer(mesh).zbuf[..., 0].squeeze(0)

                    screen = self.cameras.transform_points_screen(verts.unsqueeze(0)).squeeze(0)
                    cam_pts = (
                        self.cameras.get_world_to_view_transform()
                        .transform_points(verts.unsqueeze(0))
                        .squeeze(0)
                    )
                    px = (screen[:, 0] * half_w / full_w).long().clamp(0, half_w - 1)
                    py = (screen[:, 1] * half_h / full_h).long().clamp(0, half_h - 1)
                    surface_depth = zbuf[py, px]
                    cache[vis_key][i] = (surface_depth > 0) & (
                        torch.abs(cam_pts[:, 2] - surface_depth) < depth_tol
                    )

        self._depth_loss_cache = cache
        self.logger.info("Depth loss cache built successfully.")

    def invalidate_cache(self):
        """Invalidate the depth loss cache (e.g. after data truncation)."""
        self._depth_loss_cache = None
