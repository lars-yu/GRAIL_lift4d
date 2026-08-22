import unittest
from types import SimpleNamespace

import numpy as np
import torch

from grail.models.human_model import SmplxHumanModel
from grail.optimization.hand_object_ray_ik import (
    camera_ray_hand_targets,
    mesh_surface_depth_at_pixels,
    observed_palm_pixels_from_keypoints,
)
from grail.optimization.hoi_optimizer import HOIOptimizer
from grail.optimization.loss_computer import LossComputer
from grail.optimization.evaluator import truncate_data


class SemanticPalmTests(unittest.TestCase):
    def setUp(self):
        self.model = object.__new__(SmplxHumanModel)

    def test_left_and_right_joint_mapping_is_explicit(self):
        self.assertEqual(self.model.get_palm_joint_indices("left"), (0, 1, 4, 10, 7))
        self.assertEqual(self.model.get_palm_joint_indices("right"), (16, 17, 20, 26, 23))

    def test_palm_center_is_not_whole_hand_mean(self):
        joints = torch.arange(2 * 32 * 3, dtype=torch.float32).reshape(2, 32, 3)
        palm = self.model.get_palm_center_from_hand_joints(joints, "right")
        self.assertFalse(torch.allclose(palm, joints.mean(dim=1)))

    def test_observed_keypoint_layout_matches_hmr_writer(self):
        points = np.zeros((1, 32, 3), dtype=np.float32)
        points[0, :16, :2] = 10.0
        points[0, :16, 2] = 1.0
        points[0, 16:, :2] = 80.0
        points[0, 16:, 2] = 1.0
        left, _, left_idx = observed_palm_pixels_from_keypoints(points, "left")
        right, _, right_idx = observed_palm_pixels_from_keypoints(points, "right")
        np.testing.assert_allclose(left[0], [10.0, 10.0])
        np.testing.assert_allclose(right[0], [80.0, 80.0])
        self.assertTrue(max(left_idx) < 16)
        self.assertTrue(min(right_idx) >= 16)

    def test_palm_ray_uses_supplied_grail_intrinsics(self):
        initial = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
        surface = torch.tensor([1.0, 1.0])
        pixels = torch.tensor([[150.0, 100.0], [150.0, 100.0]])
        grail = torch.tensor([[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]])
        lift4d = torch.tensor([[400.0, 0.0, 100.0], [0.0, 400.0, 100.0], [0.0, 0.0, 1.0]])
        target, _ = camera_ray_hand_targets(
            initial, surface, 1, 1, query_pixels=pixels,
            camera_intrinsics=grail, target_distance=0.005,
        )
        projected_grail = grail[0, 0] * target[:, 0] / target[:, 2] + grail[0, 2]
        projected_lift = lift4d[0, 0] * target[:, 0] / target[:, 2] + lift4d[0, 2]
        self.assertTrue(torch.allclose(projected_grail, pixels[:, 0]))
        self.assertFalse(torch.allclose(projected_lift, pixels[:, 0]))

    def test_surface_fallback_over_15px_fails(self):
        vertices = torch.tensor([[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]]])
        query = torch.tensor([[100.0, 100.0]])
        K = torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "fallback limit"):
            mesh_surface_depth_at_pixels(
                vertices, query, K, max_fallback_pixel_distance=15.0
            )

    def test_surface_fallback_outside_contact_window_is_recorded(self):
        vertices = torch.tensor([[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]]])
        query = torch.tensor([[100.0, 100.0]])
        K = torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]])
        depth, fallback = mesh_surface_depth_at_pixels(
            vertices, query, K, max_fallback_pixel_distance=15.0,
            object_faces=torch.tensor([[0, 1, 2]]),
            strict_frames=torch.tensor([False]),
        )
        self.assertTrue(bool(fallback[0]))
        self.assertTrue(torch.isfinite(depth).all())


class _PatchModel:
    def get_palm_patch_indices(self, hand):
        return (0, 1)

    def get_finger_patch_indices(self, hand):
        return (2, 3)


class ContactLossTests(unittest.TestCase):
    def _computer(self):
        return LossComputer(None, _PatchModel(), "cpu", lambda _: None, 22, None)

    def _data_pred(self, obj_requires_grad=False):
        human = torch.tensor(
            [[[2.0, 0.0, 0.0], [2.0, 0.1, 0.0], [0.0, 0.0, 0.0], [0.0, 0.1, 0.0]]] * 3,
            requires_grad=True,
        )
        depth = torch.tensor(0.0, requires_grad=obj_requires_grad)
        obj = torch.zeros((3, 3, 3)) + depth
        pred = SimpleNamespace(
            human=SimpleNamespace(verts_seq=human),
            obj=SimpleNamespace(verts_seq=obj, trans=torch.zeros(3, 3)),
        )
        data = SimpleNamespace(
            frame_num=3, contact_hand="right", approach_window=1,
            contact_frame=1, object_motion_state=SimpleNamespace(move_start_frame=1),
            obj=SimpleNamespace(faces=torch.tensor([[0, 1, 2]])), obj_sdf=None,
        )
        return data, pred, depth

    def test_fingertip_only_contact_does_not_satisfy_palm_coverage(self):
        data, pred, _ = self._data_pred()
        raw, _ = self._computer()._contact_coverage_loss(
            data, pred, {"phase": "joint", "threshold": 0.01, "target_fraction": 0.30}, 1.0
        )
        self.assertGreater(float(raw), 0.0)
        raw.backward()
        self.assertGreater(float(pred.human.verts_seq.grad[:, :2].abs().sum()), 0.0)

    def test_penetration_clearance_loss_is_positive_for_overlap(self):
        data, pred, _ = self._data_pred()
        pred.human.verts_seq.data[:, :2] = 0.0
        raw, _ = self._computer()._hand_object_penetration_loss(
            data, pred, {"phase": "joint", "minimum_clearance": 0.003}, 1.0
        )
        self.assertGreater(float(raw), 0.0)

    def test_contact_surface_has_zero_object_depth_gradient(self):
        data, pred, depth = self._data_pred(obj_requires_grad=True)
        raw, _ = self._computer()._palm_surface_loss(
            data, pred, {"phase": "joint", "target_distance": 0.005}, 1.0
        )
        raw.backward()
        self.assertTrue(depth.grad is None or float(depth.grad) == 0.0)

    def test_postcontact_relative_uses_frozen_contact_offset_and_object_velocity(self):
        computer = self._computer()
        hand = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.5, 0.1, 0.0]],
            requires_grad=True,
        )
        obj = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.7, 0.0, 0.0], [0.9, 0.0, 0.0]],
            requires_grad=True,
        )
        computer._selected_palm_center = lambda data, pred: hand
        data = SimpleNamespace(
            frame_num=4,
            contact_frame=None,
            object_motion_state=SimpleNamespace(move_start_frame=1),
            palm_target_world=torch.tensor(
                [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.8, 0.0, 0.0], [0.8, 0.0, 0.0]]
            ),
        )
        pred = SimpleNamespace(obj=SimpleNamespace(trans=obj))
        raw, weighted = computer._postcontact_relative_loss(
            data, pred, {"delta": 0.01, "velocity_weight": 1.0}, 2.0
        )
        self.assertGreater(float(raw), 0.0)
        torch.testing.assert_close(weighted, 2.0 * raw)
        weighted.backward()
        self.assertIsNone(obj.grad)
        self.assertEqual(float(hand.grad[1].abs().sum()), 0.0)
        self.assertGreater(float(hand.grad[2:].abs().sum()), 0.0)

    def test_terminal_palm_depth_weight_is_not_multiplied_twice(self):
        computer = self._computer()
        actual_cam = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], requires_grad=True
        )
        computer._actual_palm_cam = lambda data, pred: actual_cam
        data = SimpleNamespace(
            frame_num=2,
            contact_frame=1,
            approach_window=1,
            object_motion_state=SimpleNamespace(move_start_frame=1),
            palm_target_cam=torch.zeros(2, 3),
        )
        cfg = {
            "phase": "precontact",
            "delta": 1.0,
            "terminal_weight": 5.0,
        }
        raw, weighted = computer._palm_depth_loss(data, None, cfg, 3.0)
        errors = actual_cam[:, 2]
        base = torch.nn.functional.huber_loss(
            errors, torch.zeros_like(errors), delta=1.0
        )
        terminal = torch.nn.functional.huber_loss(
            errors[-1:], torch.zeros_like(errors[-1:]), delta=1.0
        )
        torch.testing.assert_close(raw, base + terminal)
        torch.testing.assert_close(weighted, 3.0 * base + 5.0 * terminal)
        self.assertFalse(torch.allclose(weighted, 3.0 * (base + 5.0 * terminal)))

    def test_terminal_palm_loss_can_target_contact_frame(self):
        computer = self._computer()
        actual_cam = torch.zeros(4, 3)
        actual_cam[1, 2] = 0.5
        actual_cam[3, 2] = 0.1
        computer._actual_palm_cam = lambda data, pred: actual_cam
        data = SimpleNamespace(
            frame_num=4,
            palm_target_cam=torch.zeros(4, 3),
            object_motion_state=SimpleNamespace(move_start_frame=1),
            contact_frame=None,
        )
        cfg = {"terminal_weight": 5.0, "terminal_frame": "contact", "delta": 1.0}
        raw, weighted = computer._palm_depth_loss(data, None, cfg, 3.0)
        contact = torch.nn.functional.huber_loss(
            actual_cam[1:2, 2], torch.zeros(1), delta=1.0
        )
        last = torch.nn.functional.huber_loss(
            actual_cam[3:4, 2], torch.zeros(1), delta=1.0
        )
        base = torch.nn.functional.huber_loss(actual_cam[:, 2], torch.zeros(4), delta=1.0)
        torch.testing.assert_close(raw, base + contact)
        torch.testing.assert_close(weighted, 3.0 * base + 5.0 * contact)
        self.assertFalse(torch.allclose(weighted, 3.0 * base + 5.0 * last))

    def test_terminal_palm_loss_supports_smooth_window(self):
        computer = self._computer()
        data = SimpleNamespace(
            frame_num=6,
            contact_frame=3,
            object_motion_state=SimpleNamespace(move_start_frame=3),
        )
        start, end = computer._terminal_window_slice(
            data, {"terminal_frame": "contact", "terminal_window": 3}, 0, 6
        )
        self.assertEqual((start, end), (1, 4))

    def test_terminal_window_weights_are_monotone_and_normalized(self):
        error = torch.zeros(4, 3)
        weights = LossComputer._terminal_window_weights(error, 0, 4)
        self.assertEqual(tuple(weights.shape), (4,))
        self.assertTrue(torch.all(weights[1:] >= weights[:-1]))
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertLess(float(weights[0]), 1.0)
        self.assertGreater(float(weights[-1]), 1.0)

    def test_squared_terminal_depth_preserves_large_error_gradient(self):
        computer = self._computer()
        actual_cam = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], requires_grad=True
        )
        computer._actual_palm_cam = lambda data, pred: actual_cam
        data = SimpleNamespace(
            frame_num=2, contact_frame=1, approach_window=1,
            object_motion_state=SimpleNamespace(move_start_frame=1),
            palm_target_cam=torch.zeros(2, 3),
        )
        _, weighted = computer._palm_depth_loss(
            data, None, {"terminal_weight": 5.0, "terminal_loss": "squared"}, 0.0
        )
        weighted.backward()
        self.assertGreater(float(actual_cam.grad[-1, 2]), 1.0)

    def test_squared_terminal_3d_preserves_large_error_gradient(self):
        computer = self._computer()
        actual_world = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad=True
        )
        computer._selected_palm_center = lambda data, pred: actual_world
        data = SimpleNamespace(
            frame_num=2, contact_frame=1, approach_window=1,
            object_motion_state=SimpleNamespace(move_start_frame=1),
            palm_target_world=torch.zeros(2, 3),
        )
        _, weighted = computer._palm_target_3d_loss(
            data, None, {"terminal_weight": 5.0, "terminal_loss": "squared"}, 0.0
        )
        weighted.backward()
        self.assertGreater(float(actual_world.grad[-1, 0]), 1.0)


class StageCGradientMaskTests(unittest.TestCase):
    def test_approach_distance_initializes_from_target_projection(self):
        actual = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.3, 2.4, 3.5])
        direction = torch.tensor([3.0, 4.0, 0.0])
        distance, unit = HOIOptimizer._projected_approach_distance(
            actual, target, direction, 0.35
        )
        self.assertAlmostEqual(float(distance), 0.35, places=6)
        torch.testing.assert_close(unit, torch.tensor([0.6, 0.8, 0.0]))

    def test_approach_distance_cannot_move_away_from_target(self):
        distance, _ = HOIOptimizer._projected_approach_distance(
            torch.zeros(3), torch.tensor([-1.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]), 0.35,
        )
        self.assertEqual(float(distance), 0.0)

    def test_only_contact_hand_and_contact_window_have_gradient(self):
        optimizer = object.__new__(HOIOptimizer)
        optimizer.num_body_joints = 22
        optimizer.num_hand_joints = 30
        body = torch.zeros((8, 22, 6), requires_grad=True)
        hand = torch.zeros((8, 30, 6), requires_grad=True)
        body.grad = torch.ones_like(body)
        hand.grad = torch.ones_like(hand)
        optimizer.params = SimpleNamespace(human_pose_res=body, hand_pose_res=hand)
        data = SimpleNamespace(
            frame_num=8, contact_frame=None, approach_window=3, contact_hand="right",
            object_motion_state=SimpleNamespace(move_start_frame=4),
        )
        cfg = {
            "stage": "stage_3c_joint_contact_refinement", "overlap_frames": 1,
            "opt_vars": {
                "human_pose_res": {"joint_scope": "upper_body_and_arms"},
                "hand_pose_res": {"hand": "contact"},
            },
        }
        optimizer._apply_stage_gradient_masks(data, cfg)
        self.assertEqual(float(hand.grad[:5].abs().sum()), 0.0)
        self.assertEqual(float(hand.grad[:, :15].abs().sum()), 0.0)
        self.assertGreater(float(hand.grad[5:, 15:].abs().sum()), 0.0)

    def test_stage_b_does_not_open_hand_pose_residuals(self):
        optimizer = object.__new__(HOIOptimizer)
        optimizer.num_body_joints = 22
        optimizer.num_hand_joints = 30
        body = torch.zeros((8, 22, 6), requires_grad=True)
        hand = torch.zeros((8, 30, 6), requires_grad=True)
        body.grad = torch.ones_like(body)
        hand.grad = torch.ones_like(hand)
        optimizer.params = SimpleNamespace(human_pose_res=body, hand_pose_res=hand)
        data = SimpleNamespace(
            frame_num=8, contact_frame=None, approach_window=3, contact_hand="right",
            object_motion_state=SimpleNamespace(move_start_frame=4),
        )
        cfg = {
            "stage": "stage_3b_human_precontact_approach", "overlap_frames": 1,
            "opt_vars": {"human_pose_res": {"joint_scope": "arms"}},
        }
        optimizer._apply_stage_gradient_masks(data, cfg)
        self.assertTrue(torch.allclose(hand.grad, torch.ones_like(hand.grad)))

    def test_stage_b_rejects_hand_pose_configuration(self):
        optimizer = object.__new__(HOIOptimizer)
        with self.assertRaisesRegex(ValueError, "Stage 3B must not optimize hand_pose_res"):
            optimizer.optimize_main(
                None,
                {
                    "stage": "stage_3b_human_precontact_approach",
                    "opt_vars": {"hand_pose_res": {"lr": 1e-5}},
                },
            )


class PalmDataTruncationTests(unittest.TestCase):
    def test_all_palm_sequences_follow_frame_truncation(self):
        frames = 5
        human = SimpleNamespace(
            motion_data={},
            body_keypoints_seq=torch.zeros(frames, 1, 3),
            hand_keypoints_seq=torch.zeros(frames, 1, 3),
            masks=torch.zeros(frames, 2, 2),
        )
        obj = SimpleNamespace(
            verts_seq=torch.zeros(frames, 3, 3),
            poses=torch.eye(4).repeat(frames, 1, 1),
            verts_tracking_seq=torch.zeros(frames, 3, 3),
            masks=torch.zeros(frames, 2, 2),
        )
        data = SimpleNamespace(
            frame_num=frames,
            human=human,
            obj=obj,
            lift4d_depth=None,
            object_motion_state=None,
            hand_ray_target_world=None,
            hand_ray_ramp=None,
            images_path=list(range(frames)),
            depth_maps=list(range(frames)),
        )
        sequence_fields = {
            "hand_initial_cam": (frames, 3),
            "hand_pixels": (frames, 2),
            "hand_ray_surface_fallback": (frames,),
            "hand_initial_cam_depth": (frames,),
            "hand_target_cam_depth": (frames,),
            "object_surface_depth": (frames,),
            "observed_palm_pixels": (frames, 2),
            "palm_pixel_fallback": (frames,),
            "palm_target_cam": (frames, 3),
            "palm_target_world": (frames, 3),
            "palm_target_normal_world": (frames, 3),
            "grail_camera_intrinsics": (frames, 3, 3),
            "palm_surface_fallback": (frames,),
        }
        for name, shape in sequence_fields.items():
            setattr(data, name, torch.zeros(shape))

        truncate_data(data, 3)

        self.assertEqual(data.frame_num, 3)
        for name in sequence_fields:
            self.assertEqual(getattr(data, name).shape[0], 3, name)
        self.assertEqual(len(data.images_path), 3)
        self.assertEqual(len(data.depth_maps), 3)


if __name__ == "__main__":
    unittest.main()
