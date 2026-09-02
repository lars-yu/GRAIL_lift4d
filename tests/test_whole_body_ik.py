import unittest

import torch

from grail.optimization.whole_body_ik import (
    SupportFootAnchors,
    build_alternating_footstep_plan,
    build_support_foot_anchors,
    contact_elbow_angles_degrees,
    footstep_target_error,
    project_rotation_residuals_,
    rotation_residual_angle_degrees,
    support_foot_anchor_error,
)


class WholeBodyIKTests(unittest.TestCase):
    def _standing_joints(self, frame_num=70):
        joints = torch.zeros(frame_num, 17, 3)
        joints[:, 15, 1] = -0.10
        joints[:, 16, 1] = 0.10
        return joints

    def test_support_episode_uses_one_world_anchor(self):
        joints = torch.zeros(5, 17, 3)
        joints[:, 15, 0] = torch.tensor([0.0, 0.01, 0.02, 1.0, 1.1])
        probs = torch.zeros(5, 4)
        probs[:3, 0] = 0.9
        anchors = build_support_foot_anchors(joints, (15, 16), probs)
        torch.testing.assert_close(
            anchors.reference_world[:3, 0, 0], torch.full((3,), 0.01)
        )
        self.assertTrue(bool(anchors.support_mask[:3, 0].all()))
        self.assertFalse(bool(anchors.support_mask[3:, 0].any()))

    def test_support_anchor_error_is_absolute_not_only_velocity(self):
        reference = torch.zeros(3, 2, 3)
        mask = torch.ones(3, 2, dtype=torch.bool)
        joints = torch.zeros(3, 17, 3)
        joints[:, 15, 0] = 0.02
        joints[:, 16, 0] = 0.02
        error = support_foot_anchor_error(
            joints, (15, 16), SupportFootAnchors(reference, mask)
        )
        torch.testing.assert_close(error, torch.full((6,), 0.02))

    def test_contact_elbow_angle_uses_coco17_chain(self):
        joints = torch.zeros(1, 17, 3)
        joints[0, 6] = torch.tensor([0.0, 0.0, 0.0])
        joints[0, 8] = torch.tensor([1.0, 0.0, 0.0])
        joints[0, 10] = torch.tensor([1.0, 1.0, 0.0])
        angle = contact_elbow_angles_degrees(joints, "right")
        torch.testing.assert_close(angle, torch.tensor([[90.0]]))

    def test_rotation_projection_enforces_hard_residual_limit(self):
        residual = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0, 0.0, -1.0]]], requires_grad=True
        )
        project_rotation_residuals_(residual, {0: 10.0})
        self.assertLessEqual(float(rotation_residual_angle_degrees(residual)[0, 0]), 10.001)

    def test_large_root_shift_becomes_four_alternating_steps(self):
        plan = build_alternating_footstep_plan(
            self._standing_joints(),
            (15, 16),
            torch.tensor([0.34, 0.0, 0.0]),
            move_start=60,
            approach_window=60,
            max_step_length=0.22,
            max_steps=4,
            first_swing_side=1,
        )
        self.assertEqual(plan.step_count, 4)
        self.assertEqual([item[2] for item in plan.step_intervals], [1, 0, 1, 0])
        self.assertLessEqual(float(plan.step_lengths.max()), 0.22)
        torch.testing.assert_close(
            plan.root_translation_residual[60], torch.tensor([0.34, 0.0, 0.0])
        )
        torch.testing.assert_close(
            plan.foot_targets_world[60, :, 0], torch.full((2,), 0.34)
        )
        self.assertFalse(bool((plan.stance_mask & plan.swing_mask).any()))
        for start, end, side in plan.step_intervals:
            other = 1 - side
            self.assertTrue(bool(plan.stance_mask[start : end + 1, other].all()))

    def test_swing_target_has_clearance_and_new_touchdown_anchor(self):
        plan = build_alternating_footstep_plan(
            self._standing_joints(),
            (15, 16),
            torch.tensor([0.18, 0.0, 0.0]),
            move_start=60,
            approach_window=60,
            swing_clearance=0.05,
            first_swing_side=0,
        )
        start, end, side = plan.step_intervals[0]
        target_z = plan.foot_targets_world[start : end + 1, side, 2]
        self.assertGreater(float(target_z.max()), 0.049)
        self.assertEqual(int(plan.touchdown_mask[:, side].sum()), 1)
        torch.testing.assert_close(
            plan.foot_targets_world[end + 1, side, 0], torch.tensor(0.18)
        )

    def test_exact_planned_feet_have_zero_stance_swing_and_touchdown_error(self):
        plan = build_alternating_footstep_plan(
            self._standing_joints(),
            (15, 16),
            torch.tensor([0.18, 0.0, 0.0]),
            move_start=60,
            approach_window=60,
        )
        joints = self._standing_joints()
        joints[:, 15] = plan.foot_targets_world[:, 0]
        joints[:, 16] = plan.foot_targets_world[:, 1]
        for phase in ("stance", "swing", "touchdown"):
            error = footstep_target_error(joints, (15, 16), plan, phase=phase)
            torch.testing.assert_close(error, torch.zeros_like(error))

    def test_step_plan_fails_when_step_budget_cannot_cover_displacement(self):
        with self.assertRaisesRegex(ValueError, "exceeding max_steps"):
            build_alternating_footstep_plan(
                self._standing_joints(),
                (15, 16),
                torch.tensor([0.60, 0.0, 0.0]),
                move_start=60,
                approach_window=60,
                max_step_length=0.20,
                max_steps=4,
            )


if __name__ == "__main__":
    unittest.main()
