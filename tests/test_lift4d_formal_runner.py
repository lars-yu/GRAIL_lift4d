import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from grail.optimization.hoi_optimizer import HOIOptimizer

from scripts.render_saved_hoi_top_view import (
    _build_render_setup_config,
    _validate_vggt_provenance,
)
from scripts.run_lift4d_vggt_optimization import (
    _build_parser,
    _build_stage_loss_configs,
)


class FormalRunnerArgumentTests(unittest.TestCase):
    def _required(self):
        return [
            "--config-file", "config.yaml",
            "--video-id", "video/id",
            "--video-file", "video.mp4",
            "--hmr-file", "hmr.npz",
            "--mesh-file", "mesh.obj",
            "--foundationpose-poses", "poses.pkl",
            "--render-config", "render.pkl",
            "--cache-dir", "cache",
            "--results-dir", "results",
            "--lift4d-prior", "prior.npz",
            "--output-dir", "output",
        ]

    def test_vggt_and_contact_override_are_optional(self):
        args = _build_parser().parse_args(self._required())
        self.assertIsNone(args.vggt_cache)
        self.assertFalse(args.use_vggt_human_depth)
        self.assertIsNone(args.contact_frame)
        self.assertEqual(args.contact_hand, "auto")

    def test_human_vggt_mode_is_explicit(self):
        args = _build_parser().parse_args(
            self._required() + ["--use-vggt-human-depth", "--vggt-cache", "cache/vggt"]
        )
        self.assertTrue(args.use_vggt_human_depth)
        self.assertEqual(args.vggt_cache, "cache/vggt")

    def test_vggt_depth_never_supervises_object(self):
        stage_a, _, _ = _build_stage_loss_configs(True, True)
        self.assertTrue(stage_a["depth_pointcloud"]["include_human"])
        self.assertFalse(stage_a["depth_pointcloud"]["include_object"])

    def test_disabling_motion_state_restores_legacy_loss_shape(self):
        stage_a, stage_b, stage_c = _build_stage_loss_configs(False, False)
        self.assertEqual(stage_a["lift4d_depth"]["weight"], 30.0)
        self.assertEqual(stage_a["lift4d_velocity"]["weight"], 5.0)
        self.assertEqual(stage_a["fp_depth_anchor"]["weight"], 10.0)
        self.assertNotIn("object_static_pre_motion", stage_a)
        self.assertNotIn("object_static_pre_motion", stage_c)
        self.assertEqual(stage_b["contact_anchor"]["frame_radius"], 2)

    def test_top_view_accepts_disabled_vggt(self):
        _validate_vggt_provenance(
            {
                "formal_joint_optimization": {"synthetic_data_used": False},
                "vggt_depth": {"enabled": False, "consumed_by_loss": None},
            }
        )

    def test_top_view_accepts_only_human_vggt_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            depth_path = Path(directory) / "depth.npy"
            depth_path.write_bytes(b"real-depth-placeholder")
            metadata = {
                "formal_joint_optimization": {"synthetic_data_used": False},
                "vggt_depth": {
                    "enabled": True,
                    "depth_path": str(depth_path),
                    "consumed_by_loss": "human depth_pointcloud",
                },
            }
            _validate_vggt_provenance(metadata)
            metadata["vggt_depth"]["consumed_by_loss"] = "depth_pointcloud"
            with self.assertRaisesRegex(ValueError, "only the human"):
                _validate_vggt_provenance(metadata)

    def test_top_view_setup_does_not_reload_motion_prior(self):
        root_cfg = {
            "optimization": {
                "object_motion_state": {"enabled": True},
                "use_lift4d_depth_prior": True,
            },
            "human_model": {},
        }
        cfg = _build_render_setup_config(
            root_cfg,
            "/tmp/repo/configs/recon_4dhoi/pickup_smplx.yaml",
            "/tmp/results",
        )
        self.assertFalse(cfg["use_lift4d_depth_prior"])
        self.assertFalse(cfg["object_motion_state"]["enabled"])


class ObjectDepthStageConstraintTests(unittest.TestCase):
    def _optimizer(self, values):
        optimizer = HOIOptimizer.__new__(HOIOptimizer)
        optimizer.params = SimpleNamespace(
            obj_depth_res=torch.tensor(values, dtype=torch.float32, requires_grad=True)
        )
        return optimizer

    def test_full_frame_prior_initialization_is_anchor_relative(self):
        optimizer = self._optimizer([0.0, 0.0, 0.0])
        optimizer.cfg = {"lift4d_depth_scale": 1.0}
        optimizer.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
        data = SimpleNamespace(
            frame_num=3,
            object_motion_state=SimpleNamespace(move_start_frame=2),
            obj=SimpleNamespace(
                poses_cam=torch.tensor(
                    [
                        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 3.0]],
                        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.5]],
                        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.7]],
                    ]
                )
            ),
            lift4d_depth=SimpleNamespace(z_target=torch.tensor([2.0, 2.0, 1.8])),
        )
        target = optimizer.initialize_obj_depth_from_lift4d(data)
        self.assertTrue(torch.allclose(target, torch.tensor([3.0, 3.0, 2.8])))
        self.assertTrue(
            torch.allclose(
                optimizer.params.obj_depth_res, torch.tensor([0.0, 0.5, 0.1])
            )
        )

    def test_detected_static_interval_freezes_rotation_ray_and_depth(self):
        data = SimpleNamespace(
            frame_num=4,
            object_motion_state=SimpleNamespace(move_start_frame=3),
        )
        poses = torch.eye(4).repeat(4, 1, 1)
        poses[:, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        rays = torch.tensor(
            [[0.1, 0.2, 1.0], [0.2, 0.3, 1.0], [0.3, 0.4, 1.0], [0.4, 0.5, 1.0]]
        )
        depths = torch.tensor([3.0, 2.8, 2.6, 2.4])
        frozen_poses, frozen_rays, frozen_depths = (
            HOIOptimizer._freeze_static_object_pose_inputs(data, poses, rays, depths)
        )
        self.assertTrue(
            torch.equal(
                frozen_poses[:3, :3, :3], poses[0, :3, :3].expand(3, -1, -1)
            )
        )
        self.assertTrue(torch.equal(frozen_rays[:3], rays[0].expand(3, -1)))
        self.assertTrue(torch.equal(frozen_depths[:3], depths[0].expand(3)))
        self.assertTrue(torch.equal(frozen_poses[3], poses[3]))
        self.assertTrue(torch.equal(frozen_rays[3], rays[3]))
        self.assertEqual(float(frozen_depths[3]), float(depths[3]))

    def test_detected_static_interval_freezes_rotation_residual(self):
        data = SimpleNamespace(
            frame_num=4,
            object_motion_state=SimpleNamespace(move_start_frame=3),
        )
        residual = torch.eye(3).repeat(4, 1, 1)
        residual[:, 0, 1] = torch.tensor([0.0, 0.1, 0.2, 0.3])
        frozen = HOIOptimizer._freeze_static_object_rotation_residual(data, residual)
        self.assertTrue(torch.equal(frozen[:3], residual[0].expand(3, -1, -1)))
        self.assertTrue(torch.equal(frozen[3], residual[3]))

    def test_freeze_anchor_zeros_anchor_gradient(self):
        optimizer = self._optimizer([0.0, 0.1, 0.2])
        optimizer.params.obj_depth_res.grad = torch.ones(3)
        config = {
            "opt_vars": {"obj_depth_res": {"freeze_anchor": True}}
        }
        optimizer._apply_obj_depth_gradient_constraints(config)
        self.assertEqual(float(optimizer.params.obj_depth_res.grad[0]), 0.0)
        self.assertTrue(
            torch.equal(optimizer.params.obj_depth_res.grad[1:], torch.ones(2))
        )

    def test_static_state_has_highest_priority_for_pose_and_gradient(self):
        optimizer = self._optimizer([0.0, 0.1, 0.2, 0.3])
        optimizer.params.obj_depth_res.grad = torch.ones(4)
        data = SimpleNamespace(
            frame_num=4,
            object_motion_state=SimpleNamespace(
                move_start_frame=2,
                static=np.array([True, True, False, True]),
            ),
        )
        optimizer._apply_obj_depth_gradient_constraints(
            {"opt_vars": {"obj_depth_res": {"freeze_anchor": True}}}, data=data
        )
        torch.testing.assert_close(
            optimizer.params.obj_depth_res.grad, torch.tensor([0.0, 0.0, 1.0, 0.0])
        )
        poses = torch.eye(4).repeat(4, 1, 1)
        rays = torch.tensor([[0.1 * i, 0.0, 1.0] for i in range(4)])
        depths = torch.arange(4, dtype=torch.float32) + 2.0
        _, frozen_rays, frozen_depths = HOIOptimizer._freeze_static_object_pose_inputs(
            data, poses, rays, depths
        )
        self.assertEqual(float(frozen_depths[1]), float(frozen_depths[0]))
        self.assertEqual(float(frozen_depths[3]), float(frozen_depths[2]))
        torch.testing.assert_close(frozen_rays[3], frozen_rays[2])

    def test_stage_c_keeps_every_post_motion_human_frame_trainable(self):
        optimizer = HOIOptimizer.__new__(HOIOptimizer)
        optimizer.num_body_joints = 22
        pose = torch.zeros(7, 22, 6, requires_grad=True)
        pose.grad = torch.ones_like(pose)
        optimizer.params = SimpleNamespace(human_pose_res=pose)
        data = SimpleNamespace(
            frame_num=7,
            approach_window=2,
            contact_frame=None,
            object_motion_state=SimpleNamespace(move_start_frame=3),
        )
        optimizer._apply_stage_gradient_masks(
            data,
            {
                "stage": "stage_3c_joint_contact_refinement",
                "opt_vars": {"human_pose_res": {"joint_scope": "arms"}},
            },
        )
        self.assertTrue(torch.all(pose.grad[:3] == 0))
        self.assertTrue(torch.all(torch.linalg.norm(pose.grad[3:, 13], dim=-1) > 0))

    def test_stage_c_initializes_every_post_motion_frame_from_motion_anchor(self):
        optimizer = HOIOptimizer.__new__(HOIOptimizer)
        optimizer.num_body_joints = 22
        pose = torch.zeros(7, 22, 6, requires_grad=True)
        with torch.no_grad():
            pose[3, 13] = 2.0
            pose[3, 1] = 4.0
        optimizer.params = SimpleNamespace(human_pose_res=pose)
        optimizer.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
        data = SimpleNamespace(
            frame_num=7,
            object_motion_state=SimpleNamespace(move_start_frame=3),
        )
        optimizer.initialize_postcontact_pose_residuals(data, "upper_body_and_arms")
        self.assertTrue(torch.all(pose[4:, 13] == 2.0))
        self.assertTrue(torch.all(pose[4:, 1] == 0.0))
        self.assertTrue(torch.all(pose[:3] == 0.0))

    def test_joint_stage_is_bounded_to_reference(self):
        optimizer = self._optimizer([0.0, 0.1, -0.1])
        config = {
            "opt_vars": {
                "obj_depth_res": {"freeze_anchor": True, "max_delta": 0.02}
            }
        }
        reference = optimizer._obj_depth_stage_reference(config)
        with torch.no_grad():
            optimizer.params.obj_depth_res.copy_(torch.tensor([0.3, 0.5, -0.5]))
        optimizer._project_obj_depth_stage_constraints(config, reference)
        expected = torch.tensor([0.0, 0.12, -0.12])
        self.assertTrue(torch.allclose(optimizer.params.obj_depth_res, expected))

    def test_nonpositive_joint_stage_bound_fails_fast(self):
        optimizer = self._optimizer([0.0, 0.1])
        config = {"opt_vars": {"obj_depth_res": {"max_delta": 0.0}}}
        with self.assertRaisesRegex(ValueError, "must be positive"):
            optimizer._obj_depth_stage_reference(config)


if __name__ == "__main__":
    unittest.main()
