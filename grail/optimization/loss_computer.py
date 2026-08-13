"""Loss computation for HOI optimization, extracted from HOIOptimizer."""

import torch
from pytorch3d.structures import Meshes

from grail.optimization.loss_terms import (
    bidirectional_chamfer_loss,
    contact_center_loss,
    contact_depth_loss,
    contact_distribution_smoothness_loss,
    contact_loss,
    contact_smoothness_loss,
    ground_loss,
    keypoint_loss,
    l1_loss,
    penetration_loss,
    reg_loss,
    smoothness_loss,
)
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
    }

    def compute_loss(self, data, pred, loss_cfg):
        total_loss = 0.0
        loss_dict = {}
        for loss_name, cfg in loss_cfg.items():
            weight = cfg["weight"]
            fn_name = self._LOSS_FN.get(loss_name)
            if fn_name is None:
                raise ValueError(f"Invalid loss name: {loss_name}")
            loss = getattr(self, fn_name)(data, pred, cfg, weight)
            total_loss += loss
            loss_dict[loss_name] = loss.item()
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

    def _depth_pointcloud_loss(self, data, pred, cfg, weight):
        if self._depth_loss_cache is None:
            num_gt_samples = cfg.get("num_gt_samples", 3000)
            self._build_depth_loss_cache(data, pred, num_gt_samples=num_gt_samples)
        cache = self._depth_loss_cache

        human_verts_seq = pred.human.verts_seq
        obj_verts_seq = pred.obj.verts_seq
        trim_pct = cfg.get("trim_pct", 0.2)

        frame_num = human_verts_seq.shape[0]
        interval = cfg.get("interval", 1)
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
            for pred_vis, gt_pc in [
                (pred_human_visible, gt_human_pc),
                (pred_obj_visible, gt_obj_pc),
            ]:
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
