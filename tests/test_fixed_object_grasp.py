import math
import unittest

import torch

from grail.optimization.fixed_object_grasp import (
    build_fixed_object_grasp_targets,
    fixed_object_grasp_position_error,
    ground_alignment_delta,
    object_local_point_to_world,
    world_point_to_object_local,
)


class FixedObjectGraspTargetTests(unittest.TestCase):
    def test_world_object_round_trip_uses_row_vector_convention(self):
        angle = math.pi / 3.0
        rotation = torch.tensor(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = torch.tensor([0.4, -0.2, 1.3])
        point_world = torch.tensor([0.7, 0.1, 1.5])
        point_object = world_point_to_object_local(
            point_world, rotation, translation
        )
        recovered = object_local_point_to_world(
            point_object, rotation[None], translation[None]
        )[0]
        torch.testing.assert_close(recovered, point_world)

    def test_object_rotation_transports_anchor_and_normal(self):
        rotation = torch.eye(3).repeat(2, 1, 1)
        rotation[1] = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        translation = torch.zeros(2, 3)
        targets = build_fixed_object_grasp_targets(
            torch.tensor([1.0, 0.0, 0.0]),
            rotation,
            translation,
            0,
            contact_normal_world=torch.tensor([0.0, 1.0, 0.0]),
        )
        torch.testing.assert_close(
            targets.position_world,
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        )
        torch.testing.assert_close(
            targets.normal_world,
            torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        )

    def test_targets_are_detached_from_fixed_object_pose(self):
        rotation = torch.eye(3).repeat(3, 1, 1).requires_grad_()
        translation = torch.zeros(3, 3, requires_grad=True)
        contact = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        targets = build_fixed_object_grasp_targets(
            contact, rotation, translation, 1
        )
        self.assertFalse(targets.anchor_object.requires_grad)
        self.assertFalse(targets.position_world.requires_grad)

    def test_ground_alignment_is_horizontal_and_bounded(self):
        delta = ground_alignment_delta(
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([0.6, 0.8, 2.0]),
            gravity_axis=2,
            max_distance=0.35,
        )
        torch.testing.assert_close(delta, torch.tensor([0.21, 0.28, 0.0]))

    def test_acceptance_error_starts_at_contact_frame(self):
        target = torch.zeros(4, 3)
        palm = target.clone()
        palm[0, 0] = 1.0
        palm[2, 1] = 0.004
        error = fixed_object_grasp_position_error(palm, target, 1)
        torch.testing.assert_close(error, torch.tensor([0.0, 0.004, 0.0]))


if __name__ == "__main__":
    unittest.main()
