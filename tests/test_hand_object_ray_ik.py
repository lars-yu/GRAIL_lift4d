import unittest

import numpy as np
import torch

from grail.optimization.hand_object_ray_ik import (
    approach_window_from_fps,
    camera_ray_hand_targets,
    continuous_grasp_losses,
    mesh_surface_depth_at_pixels,
    select_contact_hand_from_masks,
    smoothstep_ramp,
)
from grail.optimization.loss_terms import human_silhouette_loss


class HandObjectRayIKTests(unittest.TestCase):
    def test_automatic_hand_selection_ignores_labels(self):
        masks = np.zeros((20, 40, 60), dtype=bool)
        masks[:, 15:25, 28:38] = True
        points = np.zeros((20, 32, 3), dtype=np.float32)
        points[..., 2] = 1.0
        points[:, :16, :2] = [8.0, 8.0]
        points[:, 16:, :2] = [32.0, 20.0]
        selection = select_contact_hand_from_masks(points, masks, 15)
        self.assertEqual(selection.hand, "right")
        self.assertLess(selection.right_distance_px[15], selection.left_distance_px[15])

    def test_ray_target_preserves_image_ray(self):
        initial = torch.tensor([[0.4, -0.2, 2.0]] * 20)
        depth = torch.linspace(2.0, 1.5, 20)
        target, ramp = camera_ray_hand_targets(initial, depth, 15, 10)
        torch.testing.assert_close(target[:, 0] / target[:, 2], initial[:, 0] / initial[:, 2])
        torch.testing.assert_close(target[:, 1] / target[:, 2], initial[:, 1] / initial[:, 2])
        self.assertEqual(float(ramp[0]), 0.0)
        self.assertEqual(float(ramp[15]), 1.0)
        self.assertEqual(float(ramp[-1]), 1.0)

    def test_approach_ramp_is_monotonic_and_holds_after_contact(self):
        ramp = smoothstep_ramp(40, 25, 15).numpy()
        np.testing.assert_array_equal(ramp[:10], 0.0)
        self.assertTrue(np.all(np.diff(ramp) >= -1e-7))
        np.testing.assert_array_equal(ramp[25:], 1.0)

    def test_continuous_grasp_has_no_object_gradient(self):
        hand = torch.randn(12, 3, requires_grad=True)
        obj = torch.randn(12, 3, requires_grad=True)
        losses = continuous_grasp_losses(hand, obj, 4)
        total = sum(losses.values())
        total.backward()
        self.assertIsNone(obj.grad)
        self.assertIsNotNone(hand.grad)
        self.assertTrue(torch.all(torch.linalg.norm(hand.grad[4:], dim=-1) > 0))

    def test_human_silhouette_loss_backpropagates(self):
        points = torch.tensor(
            [[[10.0, 10.0], [20.0, 10.0], [10.0, 20.0], [20.0, 20.0]]],
            requires_grad=True,
        )
        mask = torch.zeros(1, 32, 32)
        mask[:, 8:23, 8:23] = 1.0
        loss = human_silhouette_loss(points, mask, (32, 32), output_size=(16, 16))
        loss.backward()
        self.assertIsNotNone(points.grad)
        self.assertGreater(float(points.grad.abs().sum()), 0.0)

    def test_mesh_surface_depth_uses_vertices_near_hand_pixel(self):
        vertices = torch.tensor(
            [[[0.0, 0.0, 2.0], [0.02, 0.0, 2.1], [1.0, 0.0, 4.0]]]
        )
        K = torch.tensor([[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]])
        depth = mesh_surface_depth_at_pixels(vertices, torch.tensor([[50.0, 50.0]]), K, top_k=2)
        self.assertLess(float(depth[0]), 2.2)

    def test_ray_triangle_surface_and_hand_side(self):
        vertices = torch.tensor(
            [[[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0], [0.0, 0.5, 2.0]]]
        )
        faces = torch.tensor([[0, 1, 2]])
        K = torch.tensor([[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]])
        surface, fallback = mesh_surface_depth_at_pixels(
            vertices,
            torch.tensor([[50.0, 50.0]]),
            K,
            object_faces=faces,
            current_hand_depth=torch.tensor([2.3]),
        )
        self.assertAlmostEqual(float(surface[0]), 2.0, places=5)
        self.assertFalse(bool(fallback[0]))
        target, _ = camera_ray_hand_targets(
            torch.tensor([[0.0, 0.0, 2.3], [0.0, 0.0, 2.3]]),
            surface.repeat(2), 1, 1, target_distance=0.02
        )
        self.assertAlmostEqual(float(target[1, 2]), 2.02, places=5)

    def test_ray_target_refreshes_when_surface_depth_changes(self):
        initial = torch.tensor([[0.2, -0.1, 2.5]] * 4)
        target_a, _ = camera_ray_hand_targets(initial, torch.full((4,), 2.0), 2, 2)
        target_b, _ = camera_ray_hand_targets(initial, torch.full((4,), 2.4), 2, 2)
        self.assertGreater(float(torch.abs(target_a[2, 2] - target_b[2, 2])), 0.1)

    def test_precontact_path_uses_single_contact_surface_endpoint(self):
        initial = torch.tensor([[0.0, 0.0, 3.0]] * 4)
        surface = torch.tensor([1.0, 1.5, 2.5, 2.6])
        target, ramp = camera_ray_hand_targets(initial, surface, 2, 2, target_distance=0.02)
        contact_target = 2.52
        expected_mid = 3.0 + float(ramp[1]) * (contact_target - 3.0)
        self.assertAlmostEqual(float(target[1, 2]), expected_mid, places=5)

    def test_approach_window_uses_distance_and_speed_bounds(self):
        self.assertEqual(approach_window_from_fps(30.0, 0.1), 20)
        self.assertEqual(approach_window_from_fps(30.0, 0.8), 60)


if __name__ == "__main__":
    unittest.main()
